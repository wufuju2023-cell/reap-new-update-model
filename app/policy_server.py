#!/usr/bin/env python3
"""policy_server 骨架：OpenAI 兼容子集 + logprobs + RTTT 端点
路由:
  POST /v1/chat/completions  策略采样 (n>1, logprobs=true)
  POST /v1/chat/completions  价值 (同 port, /value 路由, content='{"score": x}')
  POST /ttt_step             单步 LoRA 更新 (hot-swap adapter) + 回滚保护
  GET  /health
用法: /opt/venv/bin/python /workspace/app/policy_server.py --base-model FrenzyMath/REAL-Prover \
        --adapter /workspace/out/ckpt-sft/step_<n> --port 8760
"""
import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class H(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path == "/health": self._json(200, {"ok": True})
        else: self._json(404, {"error": self.path})
    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/value", "/ttt_step"):
            n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
            self._json(200, {"mock": True, "req_keys": list(req.keys()), "note": "implement per v1-spec/01"})
        else: self._json(404, {"error": self.path})

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8760)
    a = ap.parse_args(); ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()

if __name__ == "__main__":
    main()
