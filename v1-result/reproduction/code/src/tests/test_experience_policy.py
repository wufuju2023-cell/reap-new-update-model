"""Local routing/HTTP mechanism tests. Toy tensors and acceptance attestations
are explicit fixtures; no GPU, real search, or independent Lean is performed.
"""
from copy import deepcopy
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from cpu_runtime import experience_policy as policy
from cpu_runtime import online_batch as batch
from cpu_runtime.online_ttt import run_online
from cpu_runtime.http_clients import GpuHttpClient
from gpu_runtime import GpuRuntime, ToyBackend
from gpu_runtime.server import RuntimeHandler
from tests.test_experience import acceptance
from tests import test_online_batch as batch_fixtures


def contract():
    return {"backend": "real-search", "base_sha256": "e"*64, "objective": "search_visit_backup",
        "hidden_size": 4, "lora_rank": 2, "lora_alpha": 4, "lora_dropout": 0.0,
        "target_modules": ["q_proj"], "value_head": "linear-silu-linear-sigmoid-v1",
        "search_config": {"objective": "search_visit_backup", "gamma": 0.99,
            "value_semantics": "reap.search_backup_discounted_return.v1"}}


class RoutingToy(ToyBackend):
    """Deliberately emulates the real-search contract for local transport tests."""
    def experience_contract(self):
        return contract()

    def create_session(self, session_id):
        result = super().create_session(session_id)
        result["value"].update(contract()["search_config"])
        return result

    def initialize_from_experience(self, session_id, weights):
        result = super().initialize_from_experience(session_id, weights)
        result["value"].update(contract()["search_config"])
        return result


class ExperiencePolicyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); self.project = self.root/"project"; self.project.mkdir()
        self.source = self.project/"Proof.lean"; self.source.write_text("example : True := by trivial\n", encoding="utf-8")
        self.theorem = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.runtime = GpuRuntime(backend=RoutingToy(), snapshot_root=self.root/"snapshots")
        self.addCleanup(self.runtime.close)
        for sid, exp, theorem in (("source-a", "exp-a", "a"*64), ("source-b", "exp-b", "b"*64)):
            self.runtime.create_session(sid, theorem_id=theorem)
            self.runtime.learn(sid, expected_policy_version=0, event={"event_id": sid, "reward": 1})
            self.runtime.snapshot(sid, "candidate", for_experience=True)
            self.runtime.publish_experience(sid, "candidate", exp,
                acceptance(self.runtime, sid, "candidate", kind="independent-lean"))
        self.approvals = [{"experience_id": name, "family_id": "algebra", "tags": ["nat"], "priority": 3}
                          for name in ("exp-b", "exp-a")]
        self.catalog = policy.export_catalog(self.runtime.experiences.snapshots.root, self.approvals, contract())

    def query(self, mode="bank", **changes):
        return {"mode": mode, "session_id": "target", "target_theorem_sha256": self.theorem,
            "family_id": "algebra", "tags": ["nat"], "predecessor_experience_id": "exp-b" if mode == "chain" else None,
            "allow_fresh_fallback": False, **changes}

    def bundle(self, mode="bank", **changes):
        return {"catalog": deepcopy(self.catalog), "selection": policy.select(self.catalog, self.query(mode, **changes))}

    def server(self):
        handler = type("RoutingHandler", (RuntimeHandler,), {"runtime": self.runtime, "backend_name": "routing-toy"})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        return GpuHttpClient(f"http://127.0.0.1:{server.server_port}")

    def invoke(self, bundle, client, output="out"):
        return run_online(session_id="target", project_dir=self.project, theorem_file="Proof.lean",
            output_root=self.root/output, gpu_base_url="http://127.0.0.1:1", gamma=.99,
            client=client, experience_policy=bundle)

    def test_catalog_reads_real_store_hashes_without_copying_parameters(self):
        root = self.runtime.experiences.snapshots.root
        before = {p: p.read_bytes() for p in root.rglob("*.json")}
        again = policy.export_catalog(root, self.approvals, contract())
        self.assertEqual(self.catalog, again); self.assertEqual(before, {p:p.read_bytes() for p in before})
        self.assertEqual([e["metadata"]["experience_id"] for e in again["entries"]], ["exp-a", "exp-b"])
        self.assertNotIn('"adapter":', policy.encoded(again).decode())
        self.assertNotIn('"payload":', policy.encoded(again).decode())
        wrong = contract(); wrong["base_sha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "contract differs"):
            policy.export_catalog(root, self.approvals, wrong)

    def test_fresh_chain_bank_explicit_tie_rule_and_no_fallback(self):
        fresh = policy.select(self.catalog, self.query("fresh")); self.assertIsNone(fresh["chosen"])
        chain = policy.select(self.catalog, self.query("chain")); self.assertEqual(chain["chosen"]["experience_id"], "exp-b")
        bank = policy.select(self.catalog, self.query()); self.assertEqual(bank["chosen"]["experience_id"], "exp-a")
        with self.assertRaisesRegex(ValueError, "no approved"):
            policy.select(self.catalog, self.query(family_id="geometry"))
        fallback = policy.select(self.catalog, self.query(family_id="geometry", allow_fresh_fallback=True))
        self.assertEqual(fallback["outcome"], "fresh_fallback"); self.assertIsNone(fallback["chosen"])
        with self.assertRaises(ValueError): policy.select(self.catalog, self.query("chain", allow_fresh_fallback=True))

    def test_distinct_target_session_family_tags_and_chain_predecessor_are_gates(self):
        cases = [{"target_theorem_sha256": "b"*64}, {"session_id": "source-b"}, {"family_id": "other"},
                 {"tags": ["geometry"]}, {"predecessor_experience_id": "missing"}]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                policy.select(self.catalog, self.query("chain", **changes))
        with self.assertRaises(ValueError): policy.select(self.catalog, self.query(predecessor_experience_id="exp-a"))

    def test_catalog_and_selection_tamper_mixed_profile_and_nonfinite_rejected(self):
        for transform in (lambda c:c.update(profile="mixed"), lambda c:c["contract"].update(backend="mixed-replay"),
                          lambda c:c["entries"][0]["metadata"]["acceptance"].update(kind="local-mechanism"),
                          lambda c:c["entries"][0].update(priority=True)):
            changed = deepcopy(self.catalog); transform(changed)
            with self.assertRaises(ValueError): policy.select(changed, self.query())
        selected = self.bundle(); selected["selection"]["chosen"]["experience_id"] = "exp-b"
        with self.assertRaisesRegex(ValueError, "deterministic"):
            policy.validate_policy(selected, session_id="target", theorem_sha256=self.theorem)
        with self.assertRaises(ValueError): policy.decode(b'{"a":1,"a":2}')
        with self.assertRaises(ValueError): policy.decode(b'{"a":NaN}')

    def test_export_and_selection_cli_preserve_pins_and_reject_rewrite(self):
        for name,value in (("approvals",self.approvals),("contract",contract()),("query",self.query())):
            (self.root/(name+".json")).write_bytes(policy.encoded(value))
        out = self.root/"catalog.json"
        args = ["export-catalog", "--store-root", str(self.runtime.experiences.snapshots.root),
            "--approvals-json", str(self.root/"approvals.json"), "--contract-json", str(self.root/"contract.json"), "--output", str(out)]
        self.assertEqual(policy.main(args),0); original = out.read_bytes()
        with self.assertRaises(FileExistsError): policy.main(args)
        self.assertEqual(out.read_bytes(),original)
        pin = hashlib.sha256(original).hexdigest()
        self.assertEqual(policy.main(["select","--catalog",str(out),"--catalog-sha256",pin,
            "--query-json",str(self.root/"query.json"),"--output",str(self.root/"selection.json")]),0)
        selected = policy.read_json(self.root/"selection.json")
        self.assertEqual(policy.validate_policy(selected,session_id="target",theorem_sha256=self.theorem)["experience_id"],"exp-a")
        out.write_bytes(original+b" ")
        with self.assertRaisesRegex(ValueError,"file SHA"):policy.load_catalog(out,pin)

    def test_actual_http_fresh_and_chain_initialize_then_gate_before_mock_Lean(self):
        client = self.server()
        for mode in ("fresh","chain"):
            bundle = self.bundle(mode)
            with patch.object(client,"snapshot",side_effect=RuntimeError("initialization accepted")) as snap, \
                    patch("cpu_runtime.online_ttt.subprocess.Popen") as lean:
                with self.assertRaisesRegex(RuntimeError,"initialization accepted"):
                    self.invoke(bundle,client,mode)
                snap.assert_called_once(); lean.assert_not_called()
            created = policy.read_json(self.root/mode/"target/create-receipt.json")
            policy.validate_execution_contract(created,bundle)
            self.assertEqual(created["initialization_contract_sha256"],policy.digest(contract()))
            if mode == "fresh":self.assertEqual(created["lineage"],{})
            else:
                self.assertEqual(created["lineage"]["experience_id"],"exp-b")
                self.assertTrue(self.runtime.backend._states["target"].learned)
            self.runtime.delete_session("target")

    def test_fresh_contract_mismatch_refused_by_real_http_before_allocation(self):
        client=self.server();bundle=self.bundle("fresh")
        changed=contract();changed["base_sha256"]="f"*64
        bundle["catalog"]["contract"]=changed;bundle["catalog"]["contract_sha256"]=policy.digest(changed)
        bundle["selection"]=policy.select(bundle["catalog"],self.query("fresh"))
        with patch("cpu_runtime.online_ttt.subprocess.Popen") as lean, patch.object(self.runtime.backend,"create_session",wraps=self.runtime.backend.create_session) as create:
            with self.assertRaises(Exception):self.invoke(bundle,client)
            create.assert_not_called();lean.assert_not_called()
        self.assertNotIn("target",self.runtime.backend._states)

    def test_bad_post_create_contract_preserved_and_no_retry_or_Lean(self):
        bundle=self.bundle("fresh");client=self.server()
        original=client.create_session
        def wrong(*args,**kwargs):
            value=original(*args,**kwargs);value["initialization_contract_sha256"]="0"*64;return value
        with patch.object(client,"create_session",side_effect=wrong) as create,patch.object(client,"snapshot") as snap,patch("cpu_runtime.online_ttt.subprocess.Popen") as lean:
            with self.assertRaisesRegex(ValueError,"contract/base"):self.invoke(bundle,client)
            receipt=self.root/"out/target/create-receipt.json";raw=receipt.read_bytes()
            with self.assertRaises(FileExistsError):self.invoke(bundle,client)
            self.assertEqual(receipt.read_bytes(),raw);create.assert_called_once();snap.assert_not_called();lean.assert_not_called()

    def test_changed_theorem_or_selection_refused_before_create(self):
        bundle=self.bundle("chain");self.source.write_text("changed target",encoding="utf-8")
        client=SimpleNamespace(create_session=Mock())
        with self.assertRaisesRegex(ValueError,"different session/theorem"):self.invoke(bundle,client)
        client.create_session.assert_not_called();self.assertFalse((self.root/"out").exists())

    def test_materialized_manifest_pins_match_and_missing_or_changed_triplet_rejected(self):
        request={k:v for k,v in self.query("chain").items() if k!="target_theorem_sha256"};request["theorem_file"]="Proof.lean"
        rows=policy.materialize(self.catalog,[request],self.project);manifest=self.root/"batch.jsonl"
        manifest.write_bytes(b"".join(policy.encoded(row) for row in rows))
        entries,_=batch.load_manifest(manifest,self.project)
        self.assertEqual(entries[0]["experience_id"],"exp-b")
        for key in policy.PINS:
            changed=deepcopy(rows[0]);changed.pop(key);manifest.write_bytes(policy.encoded(changed))
            with self.assertRaises(batch.BatchBlocked):batch.load_manifest(manifest,self.project)
        changed=deepcopy(rows[0]);changed["experience_weights_sha256"]="0"*64;manifest.write_bytes(policy.encoded(changed))
        with self.assertRaises(batch.BatchBlocked):batch.load_manifest(manifest,self.project)

    def test_materialize_cli_is_directly_loadable_and_failure_creates_no_manifest(self):
        catalog=self.root/"catalog.json";catalog.write_bytes(policy.encoded(self.catalog))
        request={k:v for k,v in self.query("bank").items() if k!="target_theorem_sha256"};request["theorem_file"]="Proof.lean"
        queries=self.root/"queries.json";queries.write_bytes(policy.encoded([request]))
        output=self.root/"materialized.jsonl"
        args=["materialize-batch","--catalog",str(catalog),"--catalog-sha256",policy.digest(self.catalog),
            "--query-json",str(queries),"--project-dir",str(self.project),"--output",str(output)]
        self.assertEqual(policy.main(args),0)
        rows,_=batch.load_manifest(output,self.project);self.assertEqual(rows[0]["experience_id"],"exp-a")
        request["family_id"]="no-match";queries.write_bytes(policy.encoded([request]))
        args[-1]=str(self.root/"must-not-exist.jsonl")
        with self.assertRaisesRegex(ValueError,"no approved"):policy.main(args)
        self.assertFalse((self.root/"must-not-exist.jsonl").exists())

    def test_corrupt_real_store_source_refused_without_export_or_repair(self):
        backend=self.runtime.experiences.snapshots.root/"exp-a/release/backend.json"
        backend.write_bytes(b"damaged fixture")
        with self.assertRaises(Exception):
            policy.export_catalog(self.runtime.experiences.snapshots.root,self.approvals,contract())
        self.assertEqual(backend.read_bytes(),b"damaged fixture")

    def test_exact_selected_source_metadata_is_checked_after_existing_pin_gate(self):
        client=self.server();bundle=self.bundle("chain");original=client.create_session
        def wrong(*args,**kwargs):
            result=original(*args,**kwargs);result["lineage"]["source"]["policy_version"]+=1;return result
        with patch.object(client,"create_session",side_effect=wrong) as create,patch.object(client,"snapshot") as snapshot,patch("cpu_runtime.online_ttt.subprocess.Popen") as lean:
            with self.assertRaisesRegex(ValueError,"selected catalog metadata"):self.invoke(bundle,client)
            create.assert_called_once();snapshot.assert_not_called();lean.assert_not_called()


class PolicyBatchTests(unittest.TestCase):
    setUp = ExperiencePolicyTests.setUp
    query = ExperiencePolicyTests.query
    def test_batch_executes_selected_pins_and_resume_rechecks_catalog_contract(self):
        request={k:v for k,v in self.query("fresh").items() if k!="target_theorem_sha256"};request["theorem_file"]="Proof.lean"
        rows=policy.materialize(self.catalog,[request],self.project);manifest=self.root/"batch.jsonl"
        manifest.write_bytes(policy.encoded(rows[0]));calls=[]
        def execute(**kwargs):
            calls.append(kwargs)
            report=batch_fixtures.OnlineBatchTests.complete(self,kwargs,updated=False)
            directory=kwargs["output_root"]/kwargs["session_id"];p=kwargs["experience_policy"]
            session=batch.read_json(directory/"session.json")
            extra={"experience_policy_sha256":policy.digest(p),"expected_initialization_contract_sha256":p["catalog"]["contract_sha256"]}
            session.update(extra);session.update(experience_id=None,experience_candidate=False)
            (directory/"session.json").write_bytes(batch.encoded(session));(directory/"experience-policy.json").write_bytes(policy.encoded(p))
            report.update(extra);report["experience_id"]=None;(directory/"online-result.json").write_bytes(batch.encoded(report))
            created={"session_id":"target","schema_version":"reap.gpu.session.v1","theorem_id":self.theorem,"role":"theorem",
                "policy_version":0,"completed":False,"optimizer_metadata":{"steps":0},"lineage":{},"event_receipts":{},
                "buffer_metadata":{"events":{},"pending_event_ids":[],"consumed_event_ids":[]},
                "initialization_contract":contract(),"initialization_contract_sha256":policy.digest(contract())}
            (directory/"create-receipt.json").write_bytes(batch.encoded(created));return report
        args=dict(manifest=manifest,project_dir=self.project,output_root=self.root/"batch-out",gpu_base_url="http://127.0.0.1:1",gamma=.99,executor=execute,max_updates=0)
        first=batch.run_batch(**args);self.assertEqual(len(calls),1)
        batch.run_batch(**args);self.assertEqual(len(calls),1)
        path=self.root/"batch-out/target/experience-policy.json";changed=policy.read_json(path);changed["selection"]["rule"]="latest"
        path.write_bytes(policy.encoded(changed))
        with self.assertRaises(batch.BatchBlocked):batch.run_batch(**args)
        self.assertEqual(len(calls),1)


if __name__ == "__main__":
    unittest.main()
