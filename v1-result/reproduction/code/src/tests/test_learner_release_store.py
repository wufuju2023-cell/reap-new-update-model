"""Stdlib integrity/transaction checks; opaque payload fixtures are not tensor acceptance."""
import base64
import errno
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_runtime.learner_release_store import LearnerReleaseStore, LearnerStoreError, canonical_bytes, content_sha256
from gpu_runtime.verified_objective import DATA_PROFILE, OBJECTIVE_KIND


def descriptor(digit="a", *, release=None):
    return {"dataset_sha256": digit * 64, "profile": DATA_PROFILE, "replay_receipt_sha256": "b" * 64,
        "trace_sha256": "c" * 64, "source_session_id": "source-" + digit, "theorem_sha256": "d" * 64,
        "rows": 3, "source_model_release_sha256": release}


def run_record():
    config = {"objective": OBJECTIVE_KIND, "base_tokenizer_sha256": "e" * 64, "hidden_size": 4,
        "lora": {"rank": 2, "alpha": 2, "dropout": 0.0, "target_modules": ["q_proj"]},
        "head": "linear-silu-linear-categorical", "support": {"distance_min": 1, "distance_max": 8,
            "return": "negative_integer_longest_generated_action_branch", "overflow": "reject"}}
    return {"schema_version": "reap.learner.run.v1", "role": "learner", "learner_id": "central",
        "backend_session_id": "learner", "initialization": {"kind": "base"}, "contract": {
            "backend": "verified-replay", "objective": OBJECTIVE_KIND, "base_sha256": "e" * 64,
            "hidden_size": 4, "lora_rank": 2, "lora_alpha": 2, "lora_dropout": 0.0,
            "target_modules": ["q_proj"], "value_head": "linear-silu-linear-categorical-v1", "verified_config": config},
        "seed": 12, "scope": {"kind": "generalist"}, "catalog": [descriptor()],
        "sampler": {"kind": "ordered-cycle", "batch_size": 2, "initial_state": {"step": 0, "cursor": 0}},
        "implementation": {"code_sha256": "f" * 64, "python": "stdlib-fixture"}}


def checkpoint_inputs(run, pin, *, parent=None, catalog=None, refs=None):
    step = 1 if parent is None else parent["manifest"]["step"] + 1
    before = run["sampler"]["initial_state"] if parent is None else parent["sampler_state"]
    catalog = deepcopy(catalog or (run["catalog"] if parent is None else parent["data_receipt"]["catalog"]))
    refs = deepcopy(refs or [{"dataset_sha256": catalog[0]["dataset_sha256"], "row": 0},
                             {"dataset_sha256": catalog[0]["dataset_sha256"], "row": 1}])
    event = {"kind": OBJECTIVE_KIND, "event_id": "event-" + str(step), "session_id": run["backend_session_id"],
             "policy_version": step - 1, "samples": refs}
    detail = {"objective": OBJECTIVE_KIND, "training_config": deepcopy(run["contract"]["verified_config"]),
        "optimizer_steps": step, "finite_loss": True, "finite_gradients": True, "finite_parameters": True,
        "finite_optimizer_state": True, "samples": deepcopy(refs)}
    receipt = {"event_id": event["event_id"], "applied": True, "idempotent": False, "policy_version": step, "detail": detail}
    if parent:
        logical = deepcopy(parent["logical_state"])
    else:
        logical = {"schema_version": "reap.gpu.session.v1", "session_id": run["backend_session_id"], "role": "learner",
            "theorem_id": None, "lineage": {}, "completed": False, "policy_version": 0, "adapter_metadata": {},
            "value_metadata": {}, "optimizer_metadata": {}, "reference_metadata": {},
            "buffer_metadata": {"events": {}, "pending_event_ids": [], "consumed_event_ids": []},
            "event_receipts": {}, "created_at": 1.0}
    logical["policy_version"] = step
    logical["optimizer_metadata"] = {"kind": "AdamW", "steps": step}
    logical["event_receipts"][event["event_id"]] = {"digest": content_sha256(event), "response": deepcopy(receipt)}
    logical["buffer_metadata"]["events"][event["event_id"]] = {"digest": content_sha256(event), "status": "consumed", "policy_version": step - 1}
    logical["buffer_metadata"]["consumed_event_ids"].append(event["event_id"])
    backend = {"schema_version": "reap.gpu.verified-replay-backend.v1", "session_id": run["backend_session_id"],
        "encoding": "torch-save-base64", "verified_config": deepcopy(run["contract"]["verified_config"]),
        "payload": base64.b64encode(f"opaque private snapshot step {step}".encode()).decode()}
    sampler = {"step": step, "cursor": before["cursor"] + len(refs)}
    data = {"schema_version": "reap.learner.data-receipt.v1", "run_sha256": pin, "step": step,
        "parent_receipt_sha256": None if parent is None else content_sha256(parent["data_receipt"]),
        "event": event, "event_sha256": content_sha256(event), "runtime_receipt": receipt,
        "runtime_receipt_sha256": content_sha256(receipt), "batch_refs": deepcopy(refs),
        "sampler_before_sha256": content_sha256(before), "sampler_after_sha256": content_sha256(sampler),
        "catalog": catalog, "catalog_sha256": content_sha256(catalog)}
    return [logical, backend, data, sampler]


class LearnerReleaseStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "registry"
        self.store = LearnerReleaseStore(self.root)
        self.run = run_record()
        self.run_pin = self.store.create_run(self.run)

    def checkpoint(self, *, parent=None, **kwargs):
        package = self.store.load_checkpoint(parent) if parent else None
        values = checkpoint_inputs(self.run, self.run_pin, parent=package, **kwargs)
        pin = self.store.create_checkpoint(self.run_pin, *values, parent_checkpoint_sha256=parent)
        return pin, values

    def weights(self):
        return {"contract": deepcopy(self.run["contract"]), "encoding": "torch-save-base64",
                "payload": base64.b64encode(b"opaque adapter and value head only").decode()}

    def test_v1_v2_immutable_release_and_new_store_recovery(self):
        one, values = self.checkpoint()
        release = self.store.publish(one, self.weights())
        original = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*.json")}
        two, _ = self.checkpoint(parent=one)
        second = self.store.publish(two, self.weights())
        self.assertNotEqual(one, two)
        self.assertNotEqual(release, second)
        for name, raw in original.items():
            self.assertEqual((self.root / name).read_bytes(), raw)
        restarted = LearnerReleaseStore(self.root)
        self.assertEqual(restarted.load_checkpoint(two)["sampler_state"], {"step": 2, "cursor": 4})
        metadata, weights = restarted.load_release(release)
        self.assertEqual(metadata["model_release_sha256"], release)
        self.assertEqual(metadata["weights_sha256"], content_sha256(weights))
        self.assertEqual(metadata["source"], {"source_kind": "learner_checkpoint", "learner_id": "central",
                         "learner_step": 1, "checkpoint_sha256": one})
        self.assertEqual(metadata["acceptance"], {"kind": "verified-replay-training", "committed_step": 1})
        self.assertNotIn("theorem_id", metadata)
        self.assertEqual(restarted.create_run(self.run), self.run_pin)
        self.assertEqual(restarted.publish(one, self.weights()), release)

    def test_loaded_packages_are_copies_and_no_live_backend_is_needed(self):
        one, _ = self.checkpoint()
        cp = self.store.load_checkpoint(one)
        cp["logical_state"]["role"] = "actor"
        cp["run"]["contract"]["hidden_size"] = 100
        self.assertEqual(self.store.load_checkpoint(one)["logical_state"]["role"], "learner")
        self.assertTrue(self.store.publish(one, self.weights()))

    def test_release_initialization_binds_existing_release_contract(self):
        one, _ = self.checkpoint()
        release = self.store.publish(one, self.weights())
        run = deepcopy(self.run)
        run["initialization"] = {"kind": "learner_release", "release_sha256": release}
        self.assertTrue(self.store.create_run(run))
        run["contract"]["verified_config"]["support"]["distance_max"] = 10
        with self.assertRaisesRegex(LearnerStoreError, "contract differs"):
            self.store.create_run(run)

    def test_run_strict_role_fields_and_finite_json(self):
        mutations = [lambda r: r.update(role="actor"), lambda r: r.update(theorem_id="invented"),
            lambda r: r.update(seed=True), lambda r: r["implementation"].update(value=float("nan")),
            lambda r: r["contract"].update(objective="search_visit_backup"),
            lambda r: r["contract"]["verified_config"]["support"].update(distance_max=True),
            lambda r: r["catalog"][0].update(source_model_release_sha256="bad"),
            lambda r: r["catalog"].append(deepcopy(r["catalog"][0]))]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                run = deepcopy(self.run); mutate(run)
                with self.assertRaises(ValueError):
                    self.store.create_run(run)
        for bad in ({1: "value"}, {"x": (1, 2)}, {"x": float("inf")}):
            with self.assertRaises(ValueError):
                canonical_bytes(bad)

    def test_full_backend_seed_integer_range_is_preserved(self):
        run = deepcopy(self.run); run["seed"] = 2**63 - 1
        pin = self.store.create_run(run)
        self.assertEqual(self.store.load_run(pin)["seed"], 2**63 - 1)
        for bad in (2**63, -1, True, float(2**63 - 1)):
            run["seed"] = bad
            with self.subTest(seed=bad), self.assertRaises(ValueError):
                self.store.create_run(run)

    def test_initialization_release_pin_must_exist(self):
        run = deepcopy(self.run)
        run["initialization"] = {"kind": "learner_release", "release_sha256": "1" * 64}
        with self.assertRaises((ValueError, OSError)):
            self.store.create_run(run)
        self.assertFalse((self.root / "runs" / content_sha256(run)).exists())

    def test_checkpoint_rejects_role_theorem_completed_pending_and_wrong_contract(self):
        mutations = [lambda v: v[0].update(role="actor"), lambda v: v[0].update(theorem_id="fake"),
            lambda v: v[0].update(completed=True), lambda v: v[0].update(policy_version=True),
            lambda v: v[0]["buffer_metadata"].update(pending_event_ids=["pending"]),
            lambda v: v[1].update(session_id="different"),
            lambda v: v[1]["verified_config"]["support"].update(distance_max=9),
            lambda v: v[1].update(payload="%%%"), lambda v: v[1].update(optimizer={})]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                values = checkpoint_inputs(self.run, self.run_pin); mutate(values)
                with self.assertRaises(ValueError):
                    self.store.create_checkpoint(self.run_pin, *values)
        self.assertFalse((self.root / "checkpoints").exists())

    def test_receipt_hashes_flags_identity_and_order_are_not_optional(self):
        mutations = [lambda v: v[2].update(event_sha256="0"*64),
            lambda v: v[2].update(runtime_receipt_sha256="0"*64),
            lambda v: v[2]["runtime_receipt"].update(applied=False),
            lambda v: v[2]["runtime_receipt"].update(idempotent=True),
            lambda v: v[2]["runtime_receipt"].update(event_id="wrong"),
            lambda v: v[2]["runtime_receipt"]["detail"].update(finite_parameters=False),
            lambda v: v[2]["runtime_receipt"]["detail"]["samples"].reverse(),
            lambda v: v[2]["batch_refs"].reverse(), lambda v: v[2].update(unverified=True),
            lambda v: v[2].update(parent_receipt_sha256="a"*64)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                values = checkpoint_inputs(self.run, self.run_pin); mutate(values)
                with self.assertRaises(ValueError):
                    self.store.create_checkpoint(self.run_pin, *values)

    def test_sampler_must_advance_exactly_and_parent_hash_must_match(self):
        one, _ = self.checkpoint()
        parent = self.store.load_checkpoint(one)
        mutations = [lambda v: v[3].update(cursor=9), lambda v: v[3].update(step=1),
            lambda v: v[2].update(sampler_before_sha256="0"*64),
            lambda v: v[2].update(parent_receipt_sha256="0"*64)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                values = checkpoint_inputs(self.run, self.run_pin, parent=parent); mutate(values)
                with self.assertRaises(ValueError):
                    self.store.create_checkpoint(self.run_pin, *values, parent_checkpoint_sha256=one)
        values = checkpoint_inputs(self.run, self.run_pin, parent=parent)
        with self.assertRaises(ValueError):
            self.store.create_checkpoint(self.run_pin, *values)

    def test_cross_run_parent_and_changed_historical_receipt_rejected(self):
        one, _ = self.checkpoint()
        run = deepcopy(self.run); run["learner_id"] = "another"
        pin = self.store.create_run(run)
        values = checkpoint_inputs(run, pin, parent=self.store.load_checkpoint(one))
        with self.assertRaisesRegex(ValueError, "parent checkpoint run/step"):
            self.store.create_checkpoint(pin, *values, parent_checkpoint_sha256=one)
        values = checkpoint_inputs(self.run, self.run_pin, parent=self.store.load_checkpoint(one))
        values[0]["event_receipts"]["event-1"]["response"]["detail"]["finite_loss"] = 1
        with self.assertRaisesRegex(ValueError, "historical event"):
            self.store.create_checkpoint(self.run_pin, *values, parent_checkpoint_sha256=one)

    def test_catalog_can_append_new_release_lineage_but_never_rewrite_prefix(self):
        one, _ = self.checkpoint()
        added = descriptor("f", release="1" * 64)
        catalog = self.run["catalog"] + [added]
        refs = [{"dataset_sha256": added["dataset_sha256"], "row": 2}]
        two, _ = self.checkpoint(parent=one, catalog=catalog, refs=refs)
        cp = self.store.load_checkpoint(two)
        self.assertEqual(cp["data_receipt"]["catalog"], catalog)
        self.assertEqual(cp["manifest"]["catalog_sha256"], content_sha256(catalog))
        self.assertIsNone(cp["run"]["catalog"][0]["source_model_release_sha256"])
        bad_catalogs = [catalog[::-1], catalog[1:], deepcopy(catalog)]
        bad_catalogs[-1][0]["source_model_release_sha256"] = "1" * 64
        for bad in bad_catalogs:
            with self.subTest(catalog=bad):
                values = checkpoint_inputs(self.run, self.run_pin, parent=cp, catalog=bad)
                with self.assertRaisesRegex(ValueError, "exact prefix"):
                    self.store.create_checkpoint(self.run_pin, *values, parent_checkpoint_sha256=two)

    def test_catalog_range_and_duplicate_sampling_are_explicit(self):
        repeated = [{"dataset_sha256": "a" * 64, "row": 0}] * 2
        pin, _ = self.checkpoint(refs=repeated)
        self.assertEqual(self.store.load_checkpoint(pin)["data_receipt"]["batch_refs"], repeated)
        for ref in ({"dataset_sha256": "a" * 64, "row": 3}, {"dataset_sha256": "f" * 64, "row": 0}):
            with self.subTest(ref=ref), self.assertRaisesRegex(ValueError, "outside fixed catalog"):
                self.checkpoint(refs=[ref])

    def test_release_rejects_extra_private_state_and_mismatched_contract(self):
        one, _ = self.checkpoint()
        for mutate in (lambda w: w.update(optimizer={}), lambda w: w.update(payload="bad!"),
                       lambda w: w["contract"]["verified_config"]["support"].update(distance_max=9)):
            weights = self.weights(); mutate(weights)
            with self.assertRaises(ValueError):
                self.store.publish(one, weights)
        self.assertFalse((self.root / "releases").exists())

    def test_missing_corrupt_and_noncanonical_files_rejected_without_repair(self):
        one, _ = self.checkpoint()
        path = self.root / "checkpoints" / one / "backend_state.json"
        original = path.read_bytes()
        for raw in (original + b" ", b"{}", b'{"x":1,"x":2}'):
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                self.store.load_checkpoint(one)
        path.unlink()
        with self.assertRaises(ValueError):
            self.store.load_checkpoint(one)
        values = checkpoint_inputs(self.run, self.run_pin)
        with self.assertRaises(ValueError):
            self.store.create_checkpoint(self.run_pin, *values)
        self.assertFalse(path.exists())

    def test_corrupt_ancestor_blocks_loading_later_checkpoint_and_release(self):
        one, _ = self.checkpoint(); two, _ = self.checkpoint(parent=one)
        release = self.store.publish(two, self.weights())
        (self.root / "checkpoints" / one / "sampler_state.json").write_bytes(b"{}")
        for load, pin in ((self.store.load_checkpoint, two), (self.store.load_release, release)):
            with self.assertRaisesRegex(ValueError, "checksum"):
                load(pin)

    def test_publish_failure_leaves_only_unadvertised_staging(self):
        one, _ = self.checkpoint()
        with patch("gpu_runtime.learner_release_store._publish_noreplace", side_effect=OSError("injected rename failure")):
            with self.assertRaises(OSError):
                self.store.publish(one, self.weights())
        entries = list((self.root / "releases").iterdir())
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0].name.startswith(".staging-"))
        self.assertEqual(self.store.load_checkpoint(one)["manifest"]["step"], 1)

    def test_existing_empty_content_address_is_not_replaced(self):
        run = deepcopy(self.run); run["seed"] += 1
        pin = content_sha256(run)
        (self.root / "runs" / pin).mkdir()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.store.create_run(run)
        self.assertEqual(list((self.root / "runs" / pin).iterdir()), [])

    def test_failed_file_write_never_publishes_checkpoint(self):
        from cpu_runtime.verified_dataset_store import _Directory
        original = _Directory.write_new
        def fail(directory, name, data):
            if name == "logical_state.json":
                raise OSError("disk full")
            return original(directory, name, data)
        values = checkpoint_inputs(self.run, self.run_pin)
        with patch.object(_Directory, "write_new", fail), self.assertRaisesRegex(OSError, "disk full"):
            self.store.create_checkpoint(self.run_pin, *values)
        self.assertTrue(all(p.name.startswith(".staging-") for p in (self.root / "checkpoints").iterdir()))

    def test_post_publication_sync_failure_is_not_false_success_or_overwrite(self):
        from cpu_runtime.verified_dataset_store import _Directory
        run = deepcopy(self.run); run["seed"] += 20
        pin = content_sha256(run)
        original = _Directory.sync
        def fail(directory):
            if directory.path == self.root / "runs":
                raise OSError("directory fsync failed after publication")
            return original(directory)
        with patch.object(_Directory, "sync", fail), self.assertRaisesRegex(OSError, "after publication"):
            self.store.create_run(run)
        self.assertEqual(self.store.load_run(pin), run)  # Read-only reconciliation, not a mutation retry.

    def test_concurrent_identical_publication_is_atomic_and_never_overwrites(self):
        run = deepcopy(self.run); run["seed"] += 5
        def create(_):
            try:
                return self.store.create_run(run)
            except FileExistsError:
                return None
        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(create, range(5)))
        self.assertIn(content_sha256(run), results)
        self.assertTrue(all(r in (None, content_sha256(run)) for r in results))
        self.assertEqual(self.store.load_run(content_sha256(run)), run)

    def test_symlink_member_and_root_are_never_followed(self):
        one, _ = self.checkpoint()
        outside = Path(self.tmp.name) / "outside"
        outside.mkdir()
        link = Path(self.tmp.name) / "linked"
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation unavailable on this host")
        with self.assertRaises((ValueError, OSError)):
            LearnerReleaseStore(link).load_checkpoint(one)
        path = self.root / "checkpoints" / one / "backend_state.json"
        copied = outside / "backend.json"; copied.write_bytes(path.read_bytes())
        path.unlink(); path.symlink_to(copied)
        with self.assertRaises((ValueError, OSError)):
            self.store.load_checkpoint(one)


class PortableLearnerStoreTests(unittest.TestCase):
    """Emulate NFS rejecting renameat2, while exercising real hard links."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)/"store"
        self.store = LearnerReleaseStore(self.root)
        self.run = run_record()
        self.rename = patch("gpu_runtime.learner_release_store._publish_noreplace",
                            side_effect=OSError(errno.EINVAL, "simulated NFS renameat2 unsupported"))
        self.rename.start(); self.addCleanup(self.rename.stop)

    def test_complete_run_checkpoint_release_chain_and_restart_keep_same_content_pins(self):
        from gpu_runtime.learner_release_store import PORTABLE_LAYOUT, PORTABLE_COMMIT
        pin = self.store.create_run(self.run)
        one = self.store.create_checkpoint(pin, *checkpoint_inputs(self.run, pin))
        cp = self.store.load_checkpoint(one)
        two = self.store.create_checkpoint(pin, *checkpoint_inputs(self.run, pin, parent=cp),
                                           parent_checkpoint_sha256=one)
        weights = {"contract": self.run["contract"], "encoding": "torch-save-base64",
                   "payload": base64.b64encode(b"opaque weights").decode()}
        release = self.store.publish(two, weights)
        self.assertEqual(LearnerReleaseStore(self.root).load_release(release)[1], weights)
        self.assertEqual(self.store.create_run(self.run), pin)
        self.assertEqual(self.store.load_checkpoint(two)["sampler_state"]["step"], 2)
        for kind, object_pin in (("runs", pin), ("checkpoints", one), ("releases", release)):
            directory = self.root/kind/object_pin
            self.assertTrue((directory/PORTABLE_LAYOUT).is_file())
            self.assertTrue((directory/PORTABLE_COMMIT).is_file())
        # Identical API data must keep the old semantic content pin. The marker
        # is a publication protocol detail, not part of the model release ID.
        self.rename.stop()
        legacy = LearnerReleaseStore(self.root.parent/"legacy")
        self.assertEqual(legacy.create_run(self.run), pin)
        self.assertEqual(legacy.create_checkpoint(pin, *checkpoint_inputs(self.run, pin)), one)

    def test_readers_reject_complete_payloads_until_final_commit_link(self):
        from gpu_runtime import learner_release_store as module
        link = module._link_noreplace
        seen = []
        def inspect(source, name, target, destination):
            if destination == module.PORTABLE_COMMIT:
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    self.store.load_run(content_sha256(self.run))
                seen.append(True)
            return link(source, name, target, destination)
        with patch.object(module, "_link_noreplace", side_effect=inspect):
            pin = self.store.create_run(self.run)
        self.assertEqual(seen, [True])
        self.assertEqual(self.store.load_run(pin), self.run)

    def test_partial_target_is_never_repaired_or_retried(self):
        from gpu_runtime import learner_release_store as module
        pin = content_sha256(self.run)
        link = module._link_noreplace
        def fail(source, name, target, destination):
            if destination == module.PORTABLE_COMMIT:
                raise OSError("injected failure before commit")
            return link(source, name, target, destination)
        with patch.object(module, "_link_noreplace", side_effect=fail), self.assertRaises(OSError):
            self.store.create_run(self.run)
        before = {p.name: p.read_bytes() for p in (self.root/"runs"/pin).iterdir()}
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.store.load_run(pin)
        with patch.object(module, "_link_noreplace") as retry, self.assertRaisesRegex(ValueError, "incomplete"):
            self.store.create_run(self.run)
        retry.assert_not_called()
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.root/"runs"/pin).iterdir()})

    def test_existing_empty_target_wins_race_and_is_not_populated(self):
        from gpu_runtime import learner_release_store as module
        pin = content_sha256(self.run)
        def race(root, staging, digest):
            (root.path/digest).mkdir()
            raise OSError(errno.EINVAL, "unsupported after competing mkdir")
        with patch.object(module, "_publish_noreplace", side_effect=race), self.assertRaises(FileExistsError):
            self.store.create_run(self.run)
        self.assertEqual(list((self.root/"runs"/pin).iterdir()), [])

    def test_commit_hardlink_refuses_existing_marker_without_overwrite(self):
        from gpu_runtime import learner_release_store as module
        link = module._link_noreplace
        def race(source, name, target, destination):
            if destination == module.PORTABLE_COMMIT:
                target.write_new(destination, b"existing marker must remain")
            return link(source, name, target, destination)
        with patch.object(module, "_link_noreplace", side_effect=race), self.assertRaises(FileExistsError):
            self.store.create_run(self.run)
        target = self.root/"runs"/content_sha256(self.run)/module.PORTABLE_COMMIT
        self.assertEqual(target.read_bytes(), b"existing marker must remain")
        with self.assertRaises(ValueError): self.store.load_run(content_sha256(self.run))

    def test_portable_marker_identity_hash_and_missing_marker_rejected(self):
        from gpu_runtime import learner_release_store as module
        pin = self.store.create_run(self.run)
        marker = self.root/"runs"/pin/module.PORTABLE_COMMIT
        raw = marker.read_bytes(); value = json.loads(raw)
        for mutate in (lambda x: x.update(pin="0"*64), lambda x: x["files"]["run.json"].update(bytes=0),
                       lambda x: x.update(extra=True)):
            bad = deepcopy(value); mutate(bad); marker.write_bytes(canonical_bytes(bad))
            with self.assertRaises(ValueError): self.store.load_run(pin)
        marker.unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"): self.store.load_run(pin)

    def test_post_commit_sync_failure_is_unknown_but_read_only_reconcilable(self):
        from cpu_runtime.verified_dataset_store import _Directory
        from gpu_runtime.learner_release_store import PORTABLE_COMMIT
        original = _Directory.sync
        def fail(directory):
            if directory.path.name == content_sha256(self.run) and (directory.path/PORTABLE_COMMIT).exists():
                raise OSError("sync failed after hardlink commit")
            return original(directory)
        with patch.object(_Directory, "sync", fail), self.assertRaisesRegex(OSError, "after hardlink"):
            self.store.create_run(self.run)
        self.assertEqual(self.store.load_run(content_sha256(self.run)), self.run)

    def test_io_unknown_never_enters_fallback(self):
        from gpu_runtime import learner_release_store as module
        for code in (errno.EIO, errno.ETIMEDOUT, errno.EACCES):
            with self.subTest(code=code), patch.object(module, "_publish_noreplace", side_effect=OSError(code, "unknown")), \
                    patch.object(self.store, "_commit_portable") as fallback, self.assertRaises(OSError):
                self.store.create_run(self.run)
            fallback.assert_not_called()
        self.assertFalse((self.root/"runs"/content_sha256(self.run)).exists())

    def test_concurrent_fallback_has_one_complete_winner_no_overwrite(self):
        def create(_):
            try:
                return self.store.create_run(self.run)
            except (FileExistsError, LearnerStoreError):
                return None  # Another owner may be incomplete; never retry it.
        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(create, range(5)))
        self.assertIn(content_sha256(self.run), results)
        self.assertEqual(self.store.load_run(content_sha256(self.run)), self.run)

    def test_portable_marker_symlink_is_rejected(self):
        from gpu_runtime.learner_release_store import PORTABLE_COMMIT
        pin = self.store.create_run(self.run)
        marker = self.root/"runs"/pin/PORTABLE_COMMIT
        outside = self.root/"copied-marker"; outside.write_bytes(marker.read_bytes())
        marker.unlink()
        try:
            marker.symlink_to(outside)
        except OSError:
            self.skipTest("symlink creation unavailable on this host")
        with self.assertRaises((ValueError, OSError)):
            self.store.load_run(pin)


if __name__ == "__main__":
    unittest.main()
