#!/usr/bin/env python3
"""amdrctl2.py — amdbridge 控制面 CLI（纯标准库）
子命令: push put cat ls exec out chunk-send chunk-checks
"""
import json, sys, urllib.request, base64, os

BRIDGE = "http://127.0.0.1:19826"
INST = "u-25251-d64e6c11"
BASE = f"/radeon/instances/{INST}/api/contents"
CHUNK = 1_400_000  # base64 *4/3 ≈ 1.87M < 2M cap

def bridge_req(method, path, body=None, timeout=240):
    payload = {"method": method, "path": path}
    if body is not None:
        payload["body"] = body
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{BRIDGE}/req", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.loads(r.read().decode())
    if res.get("status") is None:
        raise RuntimeError(f"bridge err: {res}")
    if res.get("status") not in (200, 201):
        raise RuntimeError(f"HTTP {res.get('status')}: {(res.get('text') or '')[:300]}")
    return json.loads(res["text"])

def put(path, text):
    return bridge_req("PUT", f"{BASE}/{path}", {"type": "file", "format": "text", "content": text})["name"]

def cat(path):
    return bridge_req("GET", f"{BASE}/{path}").get("content", "")

def ls(path=""):
    r = bridge_req("GET", f"{BASE}/{path}" if path else f"{BASE}/")
    return [(c.get("name"), c.get("type")) for c in r.get("content", [])]

def chunk_send(localfile, start, count):
    f = open(localfile, "rb").read()
    n = (len(f) + CHUNK - 1) // CHUNK
    s, c = int(start), int(count)
    for i in range(s, min(s + c, n)):
        b = base64.b64encode(f[i*CHUNK:(i+1)*CHUNK]).decode()
        put(f"dlck/{i:04d}.b64", b)
        print(f"  ok {i}/{n}", flush=True)
    print(f"DONE {s}..{min(s+c,n)-1}/{n}")

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "push":
        print(put(sys.argv[2], open(sys.argv[3]).read()))
    elif cmd == "put":
        print(put(sys.argv[2], sys.argv[3]))
    elif cmd == "cat":
        print(cat(sys.argv[2]))
    elif cmd == "ls":
        for name, typ in ls(sys.argv[2] if len(sys.argv) > 2 else ""):
            print(f"{name} ({typ})")
    elif cmd == "exec":
        print(put("q", sys.argv[2]))
    elif cmd == "out":
        print("\n".join((cat("qout") or "").splitlines()[-2500:]))
    elif cmd == "chunk-send":
        chunk_send(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "chunk-fill":
        # 逐一补传缺失块（list file: 每行一个块号）
        fdata = open(sys.argv[2], "rb").read()
        n = (len(fdata) + CHUNK - 1) // CHUNK
        miss = sorted(int(x) for x in open(sys.argv[3]).read().split())
        ok = fail = 0
        for i in miss:
            b = base64.b64encode(fdata[i*CHUNK:(i+1)*CHUNK]).decode()
            for t in range(3):
                try:
                    put(f"dlck/{i:04d}.b64", b); ok += 1; break
                except Exception:
                    if t == 2: fail += 1
                        # no-op
        print(f"FILL ok={ok} fail={fail} of {len(miss)}")
    elif cmd == "chunk-checks":
        f = open(sys.argv[2], "rb").read()
        print("n=", (len(f) + CHUNK - 1) // CHUNK)
    else:
        print(__doc__)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("CLI-ERR:", e)
        sys.exit(1)
