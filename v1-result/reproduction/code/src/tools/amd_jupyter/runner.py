#!/usr/bin/env python3
"""File-queue command runner for AMD Radeon Cloud instances.
- Watches /workspace/q  : a file containing one shell command (single line, overwrite each time).
- Executes it, appends result (`$$ done: <exit> <timestamp>`) to /workspace/qout.
- Deletes /workspace/q when done (so we know it consumed).
Run once:  nohup python3 /workspace/runner.py > /workspace/runner.log 2>&1 &
Check:     cat /workspace/qout
Queue file: /workspace/q (writes via Jupyter contents API from orchestrator side)
"""
import os, time, subprocess, datetime

Q = "/workspace/q"
QOUT = "/workspace/qout"
POLL = 1.0

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("/workspace/runner.log", "a") as f:
        f.write(f"[{ts}] {msg}\n")

def main():
    log("runner started")
    while True:
        if os.path.exists(Q):
            try:
                with open(Q, "r") as f:
                    cmd = f.read().strip()
                if cmd:
                    log(f"exec: {cmd[:120]}")
                    with open(QOUT, "a") as f:
                        f.write(f"\n$ {datetime.datetime.now().isoformat()} cmd: {cmd}\n")
                    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2800)
                    with open(QOUT, "a") as f:
                        if p.stdout: f.write(p.stdout)
                        if p.stderr: f.write("[stderr]\n" + p.stderr)
                        f.write(f"[exit] {p.returncode} @ {datetime.datetime.now().isoformat()}\n")
                os.unlink(Q)
            except Exception as ex:
                log(f"error: {ex}")
                try:
                    with open(QOUT, "a") as f:
                        f.write(f"[runner-error] {ex}\n")
                    os.unlink(Q)
                except Exception:
                    pass
        time.sleep(POLL)

if __name__ == "__main__":
    main()
