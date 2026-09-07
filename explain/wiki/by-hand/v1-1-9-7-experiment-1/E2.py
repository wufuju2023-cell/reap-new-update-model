"""E2 — 证据权重敏感性（phi 扫描）

方法：对每题构造证据链（leanSearch 权重 w ∈ {0.05,0.15,0.30,0.55,1.00,2.00}，
其余证据固定），phi=tanh(Σw)；以「证据注入前后 value 差 ΔV」为观察。
判断：ΔV 随 phi 单调（饱和在 tanh 大权重区）→ 重加权机制敏感性;
     ΔV 随 phi 无单调 → 权重模型不稳，需要重审。
"""
import json, pathlib, sys, time, statistics, os

sys.path.insert(0, os.environ.get("V11_DRIVER", "/mnt/f/projects/v1-1-agentic-tool/driver"))
from runtime_transport import RuntimeTransport
from v1_contract import Evidence, evidence_composite

BASE = os.environ.get("V11_POLICY_URL", "http://100.91.25.4:8000")
PROBS = os.environ.get("PROBS", "/mnt/f/projects/v1-1-agentic-tool/smoke/problems.json")

def prompt_for(goal, hints):
    ps = "\n".join(f"lemma {h}" for h in hints if h)
    return "State:\n" + goal + "\n\nRelated theorems:\n" + (ps or "(none)")

W = [0.05, 0.15, 0.30, 0.55, 1.00, 2.00]
t = RuntimeTransport(BASE)
problems = json.loads(pathlib.Path(PROBS).read_text())

for p in problems:
    sid0 = t.session(p["id"]); t._last_sid = sid0
    v0 = t.value(prompt_for(p["goal"], [])); t.retire(sid0)
    out = []
    for w in W:
        evs = [Evidence(id=f"{p['id']}-s0", kind="leanSearch", weight=w,
                        payload="lemma sq_connected", sourceDesc="exp"),
               Evidence(id=f"{p['id']}-l0", kind="linkSearch", weight=0.30,
                        payload="https://example.org/lemma:sq_connected", sourceDesc="exp")]
        phi = evidence_composite([Evidence(**e.__dict__) for e in evs])
        hints = [e.payload for e in evs if e.kind == "leanSearch"]
        sid = t.session(p["id"]); t._last_sid = sid
        v1 = t.value(prompt_for(p["goal"], hints)); t.retire(sid)
        out.append((w, phi, v0, v1, v1 - v0))
    print(f"E2 {p['id'][:30]:30s} v0={v0:6.3f}")
    for w, phi, v0_, v1_, d in out:
        print(f"    w={w:5.2f} phi={phi:.4f} v1={v1_:7.3f} Δ={d:+.3f}")
print("E2 DONE")
