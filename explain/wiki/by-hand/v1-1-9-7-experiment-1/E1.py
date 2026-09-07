"""E1 — value 重复稳定性（10×/题）

方法：每题 10 次独立 value 请求（fresh session，无证据），统计 mean/std/min/max。
判断：RSE(%) = std/|mean| 若 < 5%（且无 -1000 哨兵）→ 端点稳定，后续实验可信。
"""
import json, pathlib, sys, time, statistics, os

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "driver"))
from runtime_transport import RuntimeTransport

BASE = os.environ.get("V11_POLICY_URL", "http://100.91.25.4:8000")
REP = 10
PROBS = os.environ.get("PROBS", "/mnt/f/projects/v1-1-agentic-tool/smoke/problems.json")

def build_prompt(goal):
    return "State:\n" + goal + "\n\nRelated theorems:\n(none)"

t = RuntimeTransport(BASE)
problems = json.loads(pathlib.Path(PROBS).read_text())

rows = []
for p in problems:
    vals = []
    for i in range(REP):
        sid = t.session(p["id"])
        try:
            t._last_sid = sid
            vals.append(t.value(build_prompt(p["goal"])))
        finally:
            t.retire(sid)
        time.sleep(0.3)
    mean = statistics.mean(vals); sd = statistics.stdev(vals)
    rse = (sd / abs(mean) * 100.0) if mean != 0 else float("inf")
    rows.append({"id": p["id"], "vals": vals, "mean": mean, "sd": sd, "rse": rse})
    print(f"E1 {p['id'][:36]:36s} n={REP} mean={mean:7.3f} sd={sd:6.3f} rse={rse:5.2f}% "
          f"min={min(vals):6.3f} max={max(vals):6.3f}")
print("E1 DONE")
