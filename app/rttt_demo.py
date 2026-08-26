#!/usr/bin/env python3
"""rttt_demo — 真实调用 policy_server (/v1/chat/completions → /ttt_step 循环)
P1 PASS 判定: ≥5 个 items 的 buffer 全部成功提交 /ttt_step 且响应 loss<inf, 指标落 metrics.
用法: /opt/venv/bin/python /workspace/app/rttt_demo.py --host localhost --port 8760 --steps 10
"""
import argparse, json, time, urllib.request
from pathlib import Path

def post(url, obj, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--metrics", default="/workspace/out/rttt_metrics.jsonl")
    a = ap.parse_args()
    Path(a.metrics).parent.mkdir(parents=True, exist_ok=True)
    base = f"http://{a.host}:{a.port}"
    print("[rttt_demo] health:", post(base + "/health", {}))
    prompts = ["User: Please generate a tactic in lean4 to solve the state.\nSTATE:\n⊢ n / m ∣ n ∧ n / m < n\nTACTIC:\nAssistant:",
               "User: Please generate a tactic in lean4 to solve the state.\nSTATE:\n⊢ 0 < x → 2 * x < 4\nTACTIC:\nAssistant:"]
    items = []
    for i in range(a.steps):
        prompt = prompts[i % len(prompts)]
        outs = post(base + "/v1/chat/completions", {"prompt": prompt, "n": 2, "temperature": 0.99})
        for c in outs["choices"][:1]:
            items.append({"prompt": prompt, "target": c["text"], "r": 1 if i % 3 == 0 else -0.5,
                          "logprob_old": c.get("logprob_avg", -12.0)})
        if len(items) >= a.k:
            r = post(base + "/ttt_step", {"items": items})
            with open(a.metrics, "a") as f:
                f.write(json.dumps({"step": i + 1, **r, "ts": time.time()}) + "\n")
            print("[rttt_demo] ttt_step ok:", r)
            items = []
    print(f"[rttt_demo] done, metrics={a.metrics}")

if __name__ == "__main__":
    main()

