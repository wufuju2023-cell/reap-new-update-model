#!/bin/bash
# amd_jupyter.sh — WSL 侧封装工具：通过 opencli 浏览器会话驱动 AMD Radeon Cloud 实例 Jupyter
# 用法:
#   ./amd_jupyter.sh push <path> <localfile>    上传文件到实例（path 是实例内相对 workspace 路径）
#   ./amd_jupyter.sh put  <path> <text>         直接写入文本
#   ./amd_jupyter.sh cat  <path>                读取文件内容
#   ./amd_jupyter.sh ls   [path]                列出目录
#   ./amd_jupyter.sh sh   <shell-cmd>           在实例内执行命令（通过 xterm 注入，需终端开着）
#   ./amd_jupyter.sh env                         显示实例基本信息
# 依赖:  opencli (session <opencli-session>), atob 在页面端可用（base64 传输中文/二进制安全）
set -euo pipefail

SESSION=${AMD_SESSION:-<opencli-session>}
INST=${AMD_INST:-<amd-instance-id>}
BASE="/radeon/instances/$INST/api/contents"

oc() { timeout 110 opencli browser "$SESSION" eval "$1" 2>&1 | grep -vE "UNDICI|trace-warnings|Update available|npm install"; }

cmd_push() {
  local path="$1" file="$2" b64
  b64=$(base64 -w0 "$file")
  oc "(async()=>{const content=atob('$b64');const r=await fetch('$BASE/$path',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:'file',format:'text',content:content})});return r.status})()"
}

cmd_put() {
  local path="$1" text="$2" b64
  b64=$(printf '%s' "$text" | base64 -w0)
  oc "(async()=>{const content=atob('$b64');const r=await fetch('$BASE/$path',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:'file',format:'text',content:content})});return r.status})()"
}

cmd_cat() {
  local path="$1"
  oc "(async()=>{const r=await fetch('$BASE/$path',{cache:'no-store'});const j=await r.json();return r.status+'|'+((j.content||'').slice(0,20000))})()"
}

cmd_ls() {
  local path="${1:-}"
  oc "(async()=>{const r=await fetch('$BASE/$path',{cache:'no-store'});const j=await r.json();return r.status+'|'+JSON.stringify(((j.content||[]).map(function(c){return c.name+' ('+c.type+')'})))})()"
}

cmd_exec() {   # 把命令写进队列（runner.py 在实例内消费）
  local text="$1" b64
  b64=$(printf '%s' "$text" | base64 -w0)
  oc "(async()=>{const content=atob('$b64');const r=await fetch('$BASE/q',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:'file',format:'text',content:content})});return r.status})()"
}

cmd_out() {   # 读 runner 执行结果（qout 尾部）
  oc "(async()=>{const r=await fetch('$BASE/qout',{cache:'no-store'});if(r.status!==200)return r.status+'|(no output yet)';const j=await r.json();return r.status+'|'+((j.content||'').slice(-12000))})()"
}

cmd_env() {
  oc "location.href+'|'+document.title"
}

cmd_sh() {
  local cmd="$1"
  oc "var t=document.querySelector('.xterm-helper-textarea');if(!t)return 'NO-TERMINAL';t.focus();t.value='$cmd\\r';t.dispatchEvent(new Event('input',{bubbles:true}));'sent'"
}

case "${1:-}" in
  push) cmd_push "$2" "$3" ;;
  put) cmd_put "$2" "$3" ;;
  cat) cmd_cat "$2" ;;
  ls) cmd_ls "${2:-}" ;;
  sh) cmd_sh "$2" ;;
  exec) cmd_exec "$2" ;;
  out) cmd_out ;;
  env) cmd_env ;;
  *) echo "usage: $0 push|put|cat|ls|sh|exec|out|env ..."; exit 1 ;;
esac
