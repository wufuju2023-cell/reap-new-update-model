#!/usr/bin/env python3
"""Deterministic OpenAI-compatible mock policy/value services for CPU smoke tests."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        return

    def send_json(self, code: int, obj: object) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"ok": True})
        else:
            self.send_json(404, {"error": self.path})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        if self.path.endswith("/chat/completions"):
            if "/value/" in self.path:
                contents = [json.dumps({"score": 0.0})]
            else:
                n = max(1, int(request.get("n", 1)))
                contents = ["trivial"] * n
            choices = []
            for index, content in enumerate(contents):
                choices.append(
                    {
                        "index": index,
                        "message": {"role": "assistant", "content": content},
                        "logprobs": {"content": [{"token": content, "logprob": -0.01}]},
                    }
                )
            self.send_json(200, {"id": "reap-cpu-mock", "choices": choices})
        else:
            self.send_json(404, {"error": self.path})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

