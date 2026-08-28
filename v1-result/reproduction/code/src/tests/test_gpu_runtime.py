from __future__ import annotations

import hashlib
from dataclasses import replace
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gpu_runtime import (
    EventConflictError,
    GpuRuntime,
    InvalidIdentifierError,
    SnapshotIntegrityError,
    ToyBackend,
    VersionConflictError,
)


def chat_request(*, n: int = 1, logprobs: bool = True) -> dict:
    return {
        "model": "reap",
        "messages": [{"role": "user", "content": "STATE:\n⊢ True\nTACTIC:"}],
        "n": n,
        "temperature": 0.99,
        "max_tokens": 32,
        "logprobs": logprobs,
    }


class GpuRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.snapshot_root = Path(self.temporary.name) / "snapshots"
        self.backend = ToyBackend()
        self.runtime = GpuRuntime(backend=self.backend, snapshot_root=self.snapshot_root)

    def tearDown(self) -> None:
        self.runtime.close()
        self.temporary.cleanup()

    def test_openai_policy_and_value_json_contract(self) -> None:
        self.runtime.create_session("contract")
        policy = self.runtime.policy("contract", chat_request(n=2))
        self.assertEqual(policy["object"], "chat.completion")
        self.assertEqual(len(policy["choices"]), 2)
        self.assertEqual(policy["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(policy["choices"][0]["message"]["content"], ToyBackend.FAILURE_TACTIC)
        self.assertIsInstance(policy["choices"][0]["logprobs"]["content"][0]["logprob"], float)

        value = self.runtime.value("contract", chat_request(logprobs=False))
        self.assertEqual(value["object"], "chat.completion")
        value_content = value["choices"][0]["message"]["content"]
        self.assertEqual(json.loads(value_content), {"score": 0.0})
        self.assertNotIn("logprobs", value["choices"][0])

    def test_learn_switches_only_one_session_to_trivial(self) -> None:
        self.runtime.create_session("alpha")
        self.runtime.create_session("beta")
        result = self.runtime.learn("alpha", expected_policy_version=0, event={"event_id": "positive-1", "reward": 1.0})
        self.assertTrue(result["applied"])
        self.assertEqual(result["policy_version"], 1)
        self.assertEqual(self.runtime.policy("alpha", chat_request())["choices"][0]["message"]["content"], "trivial")
        self.assertEqual(
            self.runtime.policy("beta", chat_request())["choices"][0]["message"]["content"],
            ToyBackend.FAILURE_TACTIC,
        )
        self.assertEqual(json.loads(self.runtime.value("alpha", chat_request())["choices"][0]["message"]["content"]), {"score": 1.0})
        self.assertEqual(self.runtime.sessions.get("alpha").policy_version, 1)
        self.assertEqual(self.runtime.sessions.get("beta").policy_version, 0)

    def test_gpu_actor_serializes_concurrent_sessions(self) -> None:
        self.runtime.close()
        self.backend = ToyBackend(compute_delay=0.02)
        self.runtime = GpuRuntime(backend=self.backend, snapshot_root=self.snapshot_root)
        session_ids = [f"parallel-{index}" for index in range(6)]
        for session_id in session_ids:
            self.runtime.create_session(session_id)
        barrier = threading.Barrier(len(session_ids))
        errors: list[BaseException] = []

        def call_policy(session_id: str) -> None:
            try:
                barrier.wait()
                self.runtime.policy(session_id, chat_request())
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=call_policy, args=(session_id,)) for session_id in session_ids]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.runtime.actor.max_active, 1)

    def test_event_idempotency_and_stale_version_conflict(self) -> None:
        self.runtime.create_session("events")
        event = {"event_id": "event-1", "reward": 1.0, "verdict": "kernel_verified"}
        first = self.runtime.learn("events", expected_policy_version=0, event=event)
        duplicate = self.runtime.learn("events", expected_policy_version=0, event=dict(event))
        self.assertTrue(first["applied"])
        self.assertFalse(duplicate["applied"])
        self.assertTrue(duplicate["idempotent"])
        self.assertEqual(duplicate["policy_version"], 1)
        self.assertEqual(self.runtime.inspect_backend("events")["optimizer_steps"], 1)
        state = self.runtime.sessions.get("events")
        self.assertEqual(state.buffer_metadata["pending_event_ids"], [])
        self.assertEqual(state.buffer_metadata["consumed_event_ids"], ["event-1"])

        with self.assertRaises(VersionConflictError):
            self.runtime.learn("events", expected_policy_version=0, event={"event_id": "event-2", "reward": 1.0})
        with self.assertRaises(EventConflictError):
            self.runtime.learn("events", expected_policy_version=1, event={"event_id": "event-1", "reward": -1.0})

    def test_identifier_and_snapshot_path_traversal_are_rejected(self) -> None:
        with self.assertRaises(InvalidIdentifierError):
            self.runtime.create_session("../escape")
        self.runtime.create_session("safe")
        with self.assertRaises(InvalidIdentifierError):
            self.runtime.snapshot("safe", "../escape")
        with self.assertRaises(InvalidIdentifierError):
            self.runtime.learn("safe", expected_policy_version=0, event={"event_id": "../../event", "reward": 1})
        self.assertFalse((self.snapshot_root.parent / "escape").exists())

    def test_failed_learn_restores_parameters_metadata_and_retryability(self) -> None:
        for failure in ("exception", "nan-detail"):
            with self.subTest(failure=failure):
                session_id = f"rollback-{failure}"
                self.runtime.create_session(session_id)
                before_backend = self.runtime.inspect_backend(session_id)
                before_session = self.runtime.sessions.get(session_id).snapshot()
                real_learn = self.backend.learn

                def fail_after_mutation(sid, event):
                    result = real_learn(sid, event)
                    if failure == "exception":
                        raise RuntimeError("injected partial optimizer failure")
                    return replace(result, detail={"loss": float("nan")})

                event = {"event_id": "retryable", "reward": 1.0}
                with patch.object(self.backend, "learn", side_effect=fail_after_mutation):
                    with self.assertRaises((RuntimeError, ValueError)):
                        self.runtime.learn(session_id, expected_policy_version=0, event=event)
                self.assertEqual(self.runtime.inspect_backend(session_id), before_backend)
                self.assertEqual(self.runtime.sessions.get(session_id).snapshot(), before_session)
                retry = self.runtime.learn(session_id, expected_policy_version=0, event=event)
                self.assertTrue(retry["applied"])
                self.assertEqual(retry["policy_version"], 1)
                self.assertEqual(self.runtime.inspect_backend(session_id)["optimizer_steps"], 1)

    def test_failed_rollback_quarantines_only_affected_session_until_deleted(self) -> None:
        self.runtime.create_session("broken")
        self.runtime.create_session("healthy")
        self.runtime.snapshot("broken", "before")
        real_learn = self.backend.learn

        def fail_after_mutation(session_id, event):
            real_learn(session_id, event)
            raise RuntimeError("injected optimizer failure")

        with patch.object(self.backend, "learn", side_effect=fail_after_mutation), patch.object(
            self.backend, "import_session", side_effect=RuntimeError("injected rollback failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                self.runtime.learn("broken", expected_policy_version=0, event={"event_id": "bad", "reward": 1})

        blocked_operations = {
            "policy": lambda: self.runtime.policy("broken", chat_request()),
            "value": lambda: self.runtime.value("broken", chat_request()),
            "learn": lambda: self.runtime.learn("broken", expected_policy_version=0, event={"event_id": "retry"}),
            "snapshot": lambda: self.runtime.snapshot("broken", "after"),
            "restore": lambda: self.runtime.restore("broken", "before"),
            "inspect": lambda: self.runtime.inspect_backend("broken"),
        }
        for name, operation in blocked_operations.items():
            with self.subTest(operation=name), self.assertRaisesRegex(RuntimeError, "quarantined"):
                operation()
        self.assertEqual(self.runtime.policy("healthy", chat_request())["policy_version"], 0)
        self.runtime.delete_session("broken")
        self.runtime.create_session("broken")
        self.assertEqual(self.runtime.inspect_backend("broken")["optimizer_steps"], 0)

    def test_restore_failure_rolls_back_and_quarantines_if_recovery_also_fails(self) -> None:
        self.runtime.create_session("restore-failure")
        self.runtime.snapshot("restore-failure", "v0")
        self.runtime.learn("restore-failure", expected_policy_version=0, event={"event_id": "first", "reward": 1})
        before_backend = self.runtime.inspect_backend("restore-failure")
        before_session = self.runtime.sessions.get("restore-failure").snapshot()
        real_import = self.backend.import_session
        calls = 0

        def fail_first_import(session_id, state):
            nonlocal calls
            calls += 1
            real_import(session_id, state)
            if calls == 1:
                raise RuntimeError("injected partial restore")

        with patch.object(self.backend, "import_session", side_effect=fail_first_import):
            with self.assertRaisesRegex(RuntimeError, "partial restore"):
                self.runtime.restore("restore-failure", "v0")
        self.assertEqual(self.runtime.inspect_backend("restore-failure"), before_backend)
        self.assertEqual(self.runtime.sessions.get("restore-failure").snapshot(), before_session)
        with patch.object(self.backend, "import_session", side_effect=RuntimeError("cannot restore")):
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                self.runtime.restore("restore-failure", "v0")
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            self.runtime.policy("restore-failure", chat_request())

    def _numeric_backend(self):
        """Exercise real learn math on CPU tensors; this is not a model/GPU gate."""
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch is unavailable; control-plane tests still run")
        from gpu_runtime.real_backend import RealProverBackend, _RealSession

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 4)
                self.head = torch.nn.Linear(4, 8)
                self.embedding.requires_grad_(False)
                self.head.requires_grad_(False)
                self.adapters = torch.nn.ModuleDict({"numeric": torch.nn.Linear(4, 4, bias=False)})
                self.config = SimpleNamespace(use_cache=True)
                self.enabled = True
                self.active_adapter = "numeric"

            def set_adapter(self, session_id):
                assert session_id in self.adapters
                self.active_adapter = session_id

            def add_adapter(self, session_id, config):
                self.adapters[session_id] = torch.nn.Linear(4, 4, bias=False)

            def delete_adapter(self, session_id):
                del self.adapters[session_id]

            @contextmanager
            def disable_adapter(self):
                self.enabled = False
                try:
                    yield
                finally:
                    self.enabled = True

            def forward(self, input_ids, labels=None, **kwargs):
                hidden = self.embedding(input_ids)
                if self.enabled:
                    hidden = hidden + self.adapters[self.active_adapter](hidden)
                logits = self.head(hidden)
                loss = None
                if labels is not None:
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, 8), labels[:, 1:].reshape(-1), ignore_index=-100,
                    )
                if kwargs.get("logits_to_keep"):
                    logits = logits[:, -kwargs["logits_to_keep"]:]
                return SimpleNamespace(logits=logits, loss=loss, hidden_states=[hidden], past_key_values=())

            def generate(self, input_ids, num_return_sequences, max_new_tokens, do_sample, **kwargs):
                sequences = input_ids.repeat(num_return_sequences, 1)
                finished = torch.zeros(num_return_sequences, dtype=torch.bool)
                for _ in range(max_new_tokens):
                    logits = self(sequences).logits[:, -1, :]
                    if do_sample:
                        next_ids = torch.multinomial(torch.softmax(logits / kwargs["temperature"], -1), 1)
                    else:
                        next_ids = logits.argmax(-1, keepdim=True)
                    next_ids[finished] = kwargs["pad_token_id"]
                    sequences = torch.cat((sequences, next_ids), 1)
                    finished |= next_ids[:, 0] == kwargs["eos_token_id"]
                    if finished.all():
                        break
                return sequences

        class TinyBatch(dict):
            def to(self, device):
                return TinyBatch({key: value.to(device) for key, value in self.items()})

        class TinyTokenizer:
            eos_token_id = 7
            pad_token_id = 7

            def __call__(self, text, **kwargs):
                return TinyBatch(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.ones(1, 2, dtype=torch.long))

            def decode(self, token_ids, skip_special_tokens=False):
                return "".join(str(int(token)) for token in token_ids if not (skip_special_tokens and int(token) == 7))

        backend = RealProverBackend.__new__(RealProverBackend)
        backend.torch = torch
        backend.functional = torch.nn.functional
        backend.device = torch.device("cpu")
        backend.model = TinyModel().eval()
        backend.tokenizer = TinyTokenizer()
        backend.hidden_size = 4
        backend.lora_config = SimpleNamespace(r=1)
        backend.kl_beta = 0.02
        backend.value_coefficient = 0.5
        backend.max_grad_norm = 1.0
        backend.learning_rate = 0.01
        backend.value_learning_rate = 0.01
        value_head = torch.nn.Sequential(torch.nn.Linear(4, 1), torch.nn.Tanh())
        parameters = list(backend.model.adapters.parameters()) + list(value_head.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=0.01)
        backend.sessions = {"numeric": _RealSession(value_head, optimizer)}
        seed, cpu_rng, device_rng = backend._new_rng("numeric")
        backend.sessions["numeric"].rng_seed = seed
        backend.sessions["numeric"].cpu_rng_state = cpu_rng
        backend.sessions["numeric"].device_rng_state = device_rng
        # This tiny fixture has no PEFT namespace. Exact production namespace
        # selection is exercised separately using representative LoRA paths.
        backend._adapter_state_dict = lambda sid: backend.model.adapters[sid].state_dict()
        backend._adapter_parameters = lambda sid: list(backend.model.adapters[sid].parameters())
        backend.set_peft_model_state_dict = lambda model, state, adapter_name: model.adapters[adapter_name].load_state_dict(state)
        return backend

    def test_real_learn_numeric_gates_and_mode_cleanup_on_cpu_tensors(self) -> None:
        for failure in ("loss", "gradient", "post-step"):
            with self.subTest(failure=failure):
                backend = self._numeric_backend()
                torch = backend.torch
                session = backend.sessions["numeric"]
                optimizer = session.optimizer
                parameter = next(backend.model.adapters.parameters())
                event = {"prompt": "original inference prompt", "tactic": "trivial", "reward": -1}
                if failure == "loss":
                    backend.value_coefficient = float("nan")
                elif failure == "gradient":
                    parameter.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
                step = optimizer.step

                def run_step():
                    step()
                    if failure == "post-step":
                        with torch.no_grad():
                            parameter.fill_(float("nan"))

                with patch.object(optimizer, "step", side_effect=run_step) as observed_step:
                    with self.assertRaises(FloatingPointError):
                        backend.learn("numeric", event)
                    self.assertEqual(observed_step.call_count, int(failure == "post-step"))
                self.assertEqual(session.optimizer_steps, 0)
                self.assertTrue(backend.model.config.use_cache)
                self.assertFalse(backend.model.training)
                self.assertFalse(session.value_head.training)
                self.assertTrue(all(parameter.grad is None for parameter in backend.model.parameters()))

    def test_real_learn_valid_update_on_cpu_tensors_preserves_frozen_parameters(self) -> None:
        backend = self._numeric_backend()
        before = {name: parameter.detach().clone() for name, parameter in backend.model.named_parameters()}
        result = backend.learn("numeric", {"prompt": "original inference prompt", "tactic": "trivial", "reward": -1})
        self.assertEqual(result.optimizer_metadata["steps"], 1)
        self.assertTrue(result.detail["finite_gradients"])
        for name, parameter in backend.model.named_parameters():
            if ".numeric." in name:
                self.assertFalse(backend.torch.equal(before[name], parameter))
            else:
                self.assertTrue(backend.torch.equal(before[name], parameter))

    def test_real_policy_raw_logprobs_stop_at_first_eos_and_keep_attention_masks(self) -> None:
        from gpu_runtime.schemas import parse_chat_request
        backend = self._numeric_backend()
        torch = backend.torch
        logits = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
        calls = []

        def score_forward(input_ids, attention_mask, past_key_values, logits_to_keep, **kwargs):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(backend.model.training)
            calls.append((input_ids.tolist(), attention_mask.tolist(), logits_to_keep))
            return SimpleNamespace(logits=logits, past_key_values=())

        raw = chat_request(n=2)
        raw.update(temperature=2.0, max_tokens=3)
        sequences = torch.tensor([[1, 2, 3, 7, 7], [1, 2, 4, 5, 7]])
        with patch.object(backend.model, "generate", return_value=sequences) as generate, patch.object(
            backend.model, "forward", side_effect=score_forward
        ):
            contents, rows = backend.policy("numeric", parse_chat_request(raw))
        self.assertEqual(contents, ["3", "45"])
        self.assertEqual([len(row) for row in rows], [2, 3])
        expected = torch.log_softmax(logits[0, 0], -1)
        for row, ids in zip(rows, ([3, 7], [4, 5, 7])):
            for token, token_id in zip(row, ids):
                self.assertAlmostEqual(token["logprob"], expected[token_id].item(), places=6)
        self.assertNotAlmostEqual(rows[0][0]["logprob"], torch.log_softmax(logits[0, 0] / 2.0, -1)[3].item())
        self.assertEqual([len(call[1][0]) for call in calls], [2, 3, 2, 3, 4])
        self.assertEqual([call[0] for call in calls], [[[1, 2]], [[3]], [[1, 2]], [[4]], [[5]]])
        self.assertTrue(all(call[2] == 1 for call in calls))
        self.assertFalse(generate.call_args.kwargs["output_scores"])
        self.assertEqual(generate.call_args.kwargs["temperature"], 2.0)

    def test_real_policy_nonfinite_score_fails_without_consuming_private_or_global_rng(self) -> None:
        from gpu_runtime.schemas import parse_chat_request
        backend = self._numeric_backend()
        torch = backend.torch
        original_private = backend.sessions["numeric"].cpu_rng_state.clone()
        original_global = torch.get_rng_state().clone()

        def generate(**kwargs):
            torch.rand(8)
            return torch.tensor([[1, 2, 3]])

        with patch.object(backend.model, "generate", side_effect=generate), patch.object(
            backend.model, "forward", return_value=SimpleNamespace(logits=torch.full((1, 1, 8), float("nan")))
        ):
            with self.assertRaisesRegex(FloatingPointError, "raw policy logits"):
                backend.policy("numeric", parse_chat_request(chat_request()))
        self.assertTrue(torch.equal(original_private, backend.sessions["numeric"].cpu_rng_state))
        self.assertTrue(torch.equal(original_global, torch.get_rng_state()))

    def test_real_session_rng_isolated_across_interleaving_and_snapshot_restore(self) -> None:
        backend = self._numeric_backend()
        torch = backend.torch
        global_before = torch.get_rng_state().clone()
        with GpuRuntime(backend=backend, snapshot_root=self.snapshot_root / "rng") as runtime:
            runtime.create_session("rng-a")
            runtime.create_session("rng-b")
            self.assertTrue(torch.equal(global_before, torch.get_rng_state()))
            runtime.snapshot("rng-a", "before")
            raw = chat_request()
            raw["max_tokens"] = 4
            first = runtime.policy("rng-a", raw)["choices"]
            runtime.policy("rng-b", raw)
            second = runtime.policy("rng-a", raw)["choices"]
            b_rng = backend.sessions["rng-b"].cpu_rng_state.clone()
            runtime.restore("rng-a", "before")
            self.assertTrue(torch.equal(b_rng, backend.sessions["rng-b"].cpu_rng_state))
            self.assertEqual(runtime.policy("rng-a", raw)["choices"], first)
            self.assertEqual(runtime.policy("rng-a", raw)["choices"], second)
            self.assertTrue(torch.equal(global_before, torch.get_rng_state()))

    def test_atomic_checksummed_snapshot_round_trip_and_corruption(self) -> None:
        self.runtime.create_session("snap")
        self.runtime.learn("snap", expected_policy_version=0, event={"event_id": "first", "reward": 1.0})
        snapshot_path = self.runtime.snapshot("snap", "v1")
        self.assertTrue(snapshot_path.samefile(self.snapshot_root / "snap" / "v1"))
        self.assertFalse(any(path.name.startswith(".v1.tmp-") for path in snapshot_path.parent.iterdir()))
        manifest = json.loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
        for filename, metadata in manifest["files"].items():
            payload = (snapshot_path / filename).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), metadata["sha256"])
            self.assertEqual(len(payload), metadata["bytes"])

        self.runtime.learn("snap", expected_policy_version=1, event={"event_id": "second", "reward": -1.0})
        self.assertEqual(self.runtime.sessions.get("snap").policy_version, 2)
        restored = self.runtime.restore("snap", "v1")
        self.assertEqual(restored["policy_version"], 1)
        self.assertEqual(self.runtime.inspect_backend("snap")["optimizer_steps"], 1)
        self.assertEqual(self.runtime.policy("snap", chat_request())["choices"][0]["message"]["content"], "trivial")

        with (snapshot_path / "backend.json").open("ab") as handle:
            handle.write(b"corrupt")
        with self.assertRaises(SnapshotIntegrityError):
            self.runtime.restore("snap", "v1")


class ModelDownloadValidationTests(unittest.TestCase):
    """Tiny local files exercise download integrity without torch or networking."""

    REPO = "FrenzyMath/REAL-Prover"
    REVISION = "fe76f68d9a88f342cb7b546307c20292fea9cced"
    SHARD = "model-00001-of-00001.safetensors"

    def setUp(self):
        from containers.gpu import download_model
        self.module = download_model
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        header = {"model.weight": {"dtype": "BF16", "shape": [1, 3584], "data_offsets": [0, 7168]}}
        self.write_shard(header, bytes(7168))
        self.write_json("config.json", {"hidden_size": 3584, "model_type": "qwen2"})
        self.write_json("tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})
        self.write_json("model.safetensors.index.json", {"metadata": {"total_size": 7168}, "weight_map": {"model.weight": self.SHARD}})
        self.manifest = self.make_manifest()

    def write_json(self, name, content):
        (self.root / name).write_text(json.dumps(content), encoding="utf-8")

    def write_shard(self, header, payload):
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        (self.root / self.SHARD).write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)

    def make_manifest(self):
        siblings = []
        for path in sorted(self.root.iterdir()):
            if path.name == self.module.LOCK_NAME:
                continue
            content = path.read_bytes()
            entry = {"rfilename": path.name, "size": len(content),
                     "blobId": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()}
            if path.suffix == ".safetensors":
                entry["lfs"] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            siblings.append(entry)
        return self.module.normalize_official_metadata(
            {"id": self.REPO, "sha": self.REVISION, "private": False, "gated": False, "siblings": siblings},
            self.REPO, self.REVISION)

    def verify(self):
        return self.module.validate_and_lock(self.root, self.manifest, self.REPO, self.REVISION)

    def assert_failure_without_lock(self, message):
        (self.root / self.module.LOCK_NAME).write_text('{"old_success":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, message):
            self.verify()
        self.assertFalse((self.root / self.module.LOCK_NAME).exists())

    def test_complete_snapshot_produces_full_local_sha256_lock(self):
        lock = self.verify()
        self.assertEqual(lock["revision"], self.REVISION)
        self.assertEqual(lock["hidden_size"], 3584)
        self.assertEqual(set(lock["files"]), set(self.manifest["files"]))
        for name, entry in lock["files"].items():
            self.assertEqual(entry["sha256"], hashlib.sha256((self.root / name).read_bytes()).hexdigest())
        self.assertEqual(lock["safetensors"]["tensor_count"], 1)
        self.assertEqual(lock["safetensors"]["total_tensor_bytes"], 7168)
        self.assertEqual(self.module.read_json(self.root / self.module.LOCK_NAME), lock)

    def test_same_size_tampered_lfs_and_git_files_fail_official_hashes(self):
        for name, message in ((self.SHARD, "LFS SHA256"), ("tokenizer_config.json", "git blob SHA1")):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
                self.assert_failure_without_lock(message)
                path.write_bytes(original)

    def test_missing_or_truncated_shard_never_leaves_success_lock(self):
        path = self.root / self.SHARD
        original = path.read_bytes()
        path.unlink()
        self.assert_failure_without_lock("missing/nonlocal")
        path.write_bytes(original[:-1])
        self.assert_failure_without_lock("file size mismatch")
        path.write_bytes(b"")
        self.assert_failure_without_lock("file size mismatch")

    def test_matching_hash_does_not_hide_invalid_safetensors_header(self):
        self.write_shard({"model.weight": {"dtype": "BF16", "shape": [1, 3584], "data_offsets": [0, 7169]}}, bytes(7168))
        self.manifest = self.make_manifest()
        self.assert_failure_without_lock("tensor offsets exceed")

    def test_index_tensor_and_shard_mapping_must_match_headers(self):
        for weight_map, message in (({"absent.weight": self.SHARD}, "tensor index/header mismatch"),
                                    ({"model.weight": "missing.safetensors"}, "missing/untrusted shard")):
            with self.subTest(weight_map=weight_map):
                self.write_json("model.safetensors.index.json", {"metadata": {"total_size": 7168}, "weight_map": weight_map})
                self.manifest = self.make_manifest()
                self.assert_failure_without_lock(message)

    def test_hidden_size_and_total_tensor_size_are_checked(self):
        self.write_json("config.json", {"hidden_size": 4096})
        self.manifest = self.make_manifest()
        self.assert_failure_without_lock("hidden_size mismatch")
        self.write_json("config.json", {"hidden_size": 3584})
        self.write_json("model.safetensors.index.json", {"metadata": {"total_size": 7167}, "weight_map": {"model.weight": self.SHARD}})
        self.manifest = self.make_manifest()
        self.assert_failure_without_lock("total_size mismatch")

    def test_manifest_rejects_main_revision_mirror_origin_and_path_escape(self):
        with self.assertRaisesRegex(ValueError, "full lowercase 40-hex"):
            self.module.validate_manifest(self.manifest, self.REPO, "main")
        original = self.manifest["source_api"]
        self.manifest["source_api"] = original.replace("huggingface.co", "mirror.invalid")
        self.assert_failure_without_lock("official HF API")
        self.manifest["source_api"] = original
        self.manifest["files"]["../escape"] = self.manifest["files"]["config.json"]
        self.assert_failure_without_lock("unsafe manifest filename")

    def test_unmanifested_files_are_not_covered_by_success_lock(self):
        (self.root / "extra-model.bin").write_bytes(b"untrusted")
        self.assert_failure_without_lock("unmanifested file")

    def test_official_metadata_source_ignores_mirror_environment(self):
        import io
        raw = {"id": self.REPO, "sha": self.REVISION, "private": False, "gated": False, "siblings": []}
        # Convert the local trusted fixture back to the actual HF API shape.
        for name, entry in self.manifest["files"].items():
            sibling = {"rfilename": name, "size": entry["size"], "blobId": entry["git_blob_sha1"]}
            if "lfs_sha256" in entry:
                sibling["lfs"] = {"size": entry["size"], "sha256": entry["lfs_sha256"]}
            raw["siblings"].append(sibling)
        response = io.BytesIO(json.dumps(raw).encode())
        response.geturl = lambda: self.module.official_url(self.REPO, self.REVISION)
        with patch.dict("os.environ", {"HF_ENDPOINT": "https://mirror.invalid"}), patch.object(self.module, "urlopen", return_value=response) as opener:
            fetched = self.module.fetch_official_manifest(self.REPO, self.REVISION)
        self.assertEqual(fetched, self.manifest)
        opener.assert_called_once_with(self.module.official_url(self.REPO, self.REVISION), timeout=30)

    def test_explicit_mirror_changes_byte_source_but_not_trust_source(self):
        import sys
        from unittest.mock import Mock
        download = Mock()
        argv = ["download_model.py", "--repo", self.REPO, "--revision", self.REVISION,
                "--output", str(self.root), "--endpoint", "https://mirror.invalid"]
        with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}), \
                patch.object(self.module, "fetch_official_manifest", return_value=self.manifest) as official, patch("builtins.print"):
            self.assertEqual(self.module.main(), 0)
        official.assert_called_once_with(self.REPO, self.REVISION)
        self.assertEqual(download.call_args.kwargs["endpoint"], "https://mirror.invalid")
        self.assertFalse(download.call_args.kwargs["token"])
        self.assertEqual(download.call_args.kwargs["revision"], self.REVISION)
        self.assertEqual(set(download.call_args.kwargs["allow_patterns"]), set(self.manifest["files"]))

    def test_download_failure_invalidates_old_success_lock(self):
        import sys
        from unittest.mock import Mock
        (self.root / self.module.LOCK_NAME).write_text('{"old_success":true}', encoding="utf-8")
        argv = ["download_model.py", "--repo", self.REPO, "--revision", self.REVISION, "--output", str(self.root)]
        download = Mock(side_effect=RuntimeError("fixture download failure"))
        with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}), \
                patch.object(self.module, "fetch_official_manifest", return_value=self.manifest), self.assertRaisesRegex(RuntimeError, "fixture download failure"):
            self.module.main()
        self.assertFalse((self.root / self.module.LOCK_NAME).exists())


if __name__ == "__main__":
    unittest.main()
