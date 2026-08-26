#!/bin/bash
# CLI 封装（tools/amdbridge/amdrctl2.sh）— 通过 bridge（127.0.0.1:19826）+ 扩展
# 用法同 amd_jupyter.sh:
#   amdrctl2.sh push <path> <localfile> | put <path> <text> | cat <path> | ls [path] | exec <cmd> | out
# 与 opencli 通道完全解耦；bridge 未启动时报错并提示。
BRIDGE=${BRIDGE:-http://127.0.0.1:19826}
INST="${INST:-u-25251-d64e6c11}"
PUBPATH="/radeon/instances/$INST/api/contents"

req() { # req <method> <path> [jsonbody]
  local method="$1" path="$2" body="${3:-}"
  local payload
  payload=$(printf '{"method":"%s","path":"%s","body":%s}' "$method" "$path" "${body:-\"\"}")
  curl -s --max-time 100 -X POST "$BRIDGE/req" -H "Content-Type: application/json" -d "$payload"
}

b64file() { base64 -w0 "$1" | tr -d '\n'; }

b64text() { printf '%s' "$1" | base64 -w0 | tr -d '\n'; }

decode() { python3 - "$1" <<'PY'
import sys, json, base64
d = json.loads(sys.argv[1])
raw = d.get("text", "")
if d.get("status", 0) in (200, 201):
    j = json.loads(raw)
    c = j.get("content", "")
    print(base64.b64decode(c).decode() if c else "NOCONTENT")
else:
    print("ERR", d.get("status"), raw[:300])
PY
}

cmd_push() { req PUT "$PUBPATH/$1" "\"$(b64file "$2")\"" | decode; }
cmd_put()  { req PUT "$PUBPATH/$1" "\"$(b64text "$2")\"" | decode; }

cmd_cat() { req GET "$PUBPATH/$1" | python3 - "$1" <<'PY'
import sys, json
d = json.loads(sys.stdin.read())
raw = d.get("text", "")
try:
    j = json.loads(raw)
    print(j.get("content", ""))
except Exception:
    print("ERR", d.get("status"), raw[:300])
PY
}

cmd_ls() { req GET "$PUBPATH/${1:-}" | python3 -c '
import sys,json
d=json.loads(sys.stdin.read())
try:
    j=json.loads(d.get("text",""))
    print(d.get("status"), "|", [c.get("name")+" ("+c.get("type")+")" for c in j.get("content",[])])
except Exception as e:
    print("ERR", d.get("status"), d.get("text","")[:200])
'; }

cmd_exec() { cmd_put "q" "$1"; }
cmd_out()  { cmd_cat "qout" | tail -2000; }

case "${1:-}" in
  push) cmd_push "$2" "$3" ;;
  put)  cmd_put "$2" "$3" ;;
  cat)  cmd_cat "$2" ;;
  ls)   cmd_ls "${2:-}" ;;
  exec) cmd_exec "$2" ;;
  out)  cmd_out ;;
  *) echo "usage: $0 push|put|cat|ls|exec|out"; exit 1 ;;
esac
