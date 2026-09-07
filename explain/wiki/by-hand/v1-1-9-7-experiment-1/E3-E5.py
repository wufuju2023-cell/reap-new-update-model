"""E3-E5 合并跑轮：prompt 变体敏感性 / candidates 多样性 / 温度扫描"""

import json, pathlib, sys, statistics, os

sys.path.insert(0, os.environ.get("V11_DRIVER", "/mnt/f/projects/v1-1-agentic-tool/driver"))
from runtime_transport import RuntimeTransport
import urllib.request

BASE = os.environ.get("V11_POLICY_URL", "http://100.91.25.4:8000")
PROBS = "/mnt/f/projects/v1-1-agentic-tool/smoke/problems.json"

t = RuntimeTransport(BASE)
probs = json.loads(pathlib.Path(PROBS).read_text())

def jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    u, i = sa | sb, sa & sb
    return len(i) / len(u) if u else 1.0

print("===== E3: prompt 变体（内容变化 → value 偏移） =====")
variants = {
    "full": lambda g: "State:\n" + g + "\n\nRelated theorems:\n(none)",
    "tactic-prompt": lambda g: "Simplify. State:\n" + g,
    "no-annotation": lambda g: g,
}
for p in probs:
    vals = {}
    for name, f in variants.items():
        sid = t.session(p["id"]); t._last_sid = sid
        vals[name] = t.value(f(p["goal"])); t.retire(sid)
    spread = max(vals.values()) - min(vals.values())
    print(f"E3 {p['id'][:30]:30s} full={vals['full']:6.3f} tp={vals['tactic-prompt']:6.3f} "
          f"none={vals['no-annotation']:6.3f} spread={spread:+.3f}")

print("===== E4: candidates 多样性（n=16, Jaccard） =====")
for p in probs:
    sid = t.session(p["id"]); t._last_sid = sid
    cands = t.policy("State:\n" + p["goal"], n=16)
    t.retire(sid)
    texts = [c[0] for c in cands]
    js = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            js.append(jaccard(texts[i], texts[j]))
    print(f"E4 {p['id'][:30]:30s} n={len(texts)} jaccardMed={statistics.median(js):.3f} "
          f"jaccardMean={statistics.mean(js):.3f} distinct={len(set(texts))}")

print("===== E5: 温度注入（请求级 temperature 0.4/1.0/1.6 → candidates 集合变化） =====")
def policy_with_temp(prompt, n, temperature):
    sid = t.session("tmp-temp"); t._last_sid = sid
    import json as _j, urllib.request as _rq
    body = _j.dumps({"model": t.model,
                     "messages": [{"role": "user", "content": prompt}],
                     "n": n, "temperature": temperature}).encode()
    req = _rq.Request(f"{t.base}/sessions/{sid}/policy/v1/chat/completions",
                      data=body, headers={"Content-Type": "application/json"}, method="POST")
    with _rq.urlopen(req, timeout=120) as r:
        out = _j.loads(r.read().decode())
    t.retire(sid)
    return out

p0 = probs[0]
base_set = None
for temp in (0.4, 1.0, 1.6):
    out = policy_with_temp("State:\n" + p0["goal"], n=8, temperature=temp)
    texts = [c.get("message", {}).get("content", "")
             for c in out.get("choices", [])]
    if base_set is None:
        base_set = set(texts)
    newness = len(set(texts) - base_set) / max(1, len(set(texts)))
    print(f"E5 temp={temp:.1f} n={len(texts)} unique={len(set(texts))} newness={newness:.2f} "
          f"sample={texts[0][:40]!r}")
print("E3-E5 DONE")
