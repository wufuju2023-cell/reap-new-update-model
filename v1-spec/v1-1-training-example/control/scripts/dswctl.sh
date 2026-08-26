#!/usr/bin/env bash
# dswctl — 一键操控 ModelScope DSW 实例（类 SSH 封装）
# 依赖: opencli（本机已配置，Edge+扩展连接）+ curl + jq
# 底层: dsw-gateway .../dsw/commands  (认证=浏览器 httpOnly Cookie, 由 opencli 代发)
#
# 用法:
#   source dswctl.sh            # 或把函数加进 ~/.bashrc
#   dsw "uname -a"              # 在实例里执行命令并打印输出
#   dsw-file-get /path          # base64 拉取实例文件到本地 out/
#   dsw-file-put local out      # 上传小文件到实例
#   dsw-open                    # 确保 workspace/IDE 页面打开(会话未失效时)
#   dsw-refresh                 # 重新发现网关段(SEG)并缓存

: "${DSW_SESSION:=aebvmcpf}"   # opencli 浏览器会话名(本机 Edge)
: "${DSW_SEG_FILE:=${XDG_CONFIG_HOME:-$HOME/.config}/dsw/seg}"   # 网关段缓存
: "${DSW_NAME:=next-ide}"      # 网关 name 参数(保持与 ms-exec.ps1 一致)
# 宿主页: modelscope.cn code/workspace（唯一使用; 轻、稳、跨域可调 commands）。
# 旧 gateway ..?/lab?appId=MAAS&instanceId=… 版已停用(列入 control/01)。
: "${DSW_PAGE:=https://modelscope.cn/code/workspace}"
DSW_HOST="dsw-gateway-cn-hangzhou.data.aliyun.com"

# 读取/设置网关段
_dsw_seg() {
  if [[ -z "${DSW_SEG:-}" ]]; then
    DSW_SEG=$(cat "$DSW_SEG_FILE" 2>/dev/null || true)
  fi
  echo "${DSW_SEG:-}"   # 无缓存则空: 首次用 dsw-refresh 发现
}
_dsw_set_seg() { DSW_SEG="$1"; mkdir -p "$(dirname "$DSW_SEG_FILE")"; printf '%s' "$1" > "$DSW_SEG_FILE"; }

# 网关命令入口
_dsw_cmd() { # $1=command string
  local seg; seg=$(_dsw_seg)
  local payload b64 js r
  payload=$(jq -n --arg c "$1" '{command:$c}')
  b64=$(printf '%s' "$payload" | base64 -w0)
  js="(async()=>{try{const r=await fetch('https://${DSW_HOST}/${seg}/dsw/commands?type=status&name=${DSW_NAME}',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:atob('$b64')});return await r.text()}catch(e){return 'ERR:'+e.message}})()"
  r=$(opencli browser "$DSW_SESSION" eval "$js" 2>/dev/null | tr -d '\0' | grep -av 'UNDICI\|trace-warnings\|Update available\|post-quantum\|vulnerable\|upgrade\|openssh\|OpenSSH' | grep -av '^catch\|^node:\|^[[:space:]]*+' | tail -4)
  printf '%s\n' "$r"
}

# 公开命令
dsw() { # 在实例里执行命令
  local seg; seg=$(_dsw_seg)
  if [[ -z "$1" ]]; then echo "用法: dsw \"<shell 命令>\""; return 1; fi
  echo "→ ${seg} \$ $1" >&2
  _dsw_cmd "$1"
  echo
}

dsw-open() { # 保证 workspace 页开着(会话未失效)
  opencli browser "$DSW_SESSION" open "$DSW_PAGE" >/dev/null 2>&1
  sleep 8
  echo "已确保页面打开: $DSW_PAGE"
}

dsw-refresh() { # 从 modelscope API 重新发现网关段
  dsw-open
  local js seg
  js="(async()=>{const r=await fetch('/api/v1/notebooks?Channel=dsw',{cache:'no-store'});const j=await r.json();const ns=j.Data.Notebooks||[];for(const n of ns){const u=(n.Url&&(n.Url.TerminalUrl||n.Url.JupyterlabUrl))||'';const s=(u.split('/').filter(function(x){return /^dsw-[0-9]+$/.test(x)})||[])[0];if(s)return s}return 'NOTFOUND'})()"
  seg=$(opencli browser "$DSW_SESSION" eval "$js" 2>/dev/null | tr -d '\0' | grep -aoE 'dsw-[0-9]+|NOTFOUND' | tail -1)
  if [[ "$seg" =~ ^dsw-[0-9]+$ ]]; then _dsw_set_seg "$seg"; echo "SEG=${seg} (已缓存)"; else echo "未发现网关段: ${seg:-空}"; fi
}

dsw-file-get() { # 拉实例内文件到 ./out/<basename>
  local rem="$1"
  [[ -z "$rem" ]] && { echo "用法: dsw-file-get /path/file"; return 1; }
  mkdir -p out
  local b64
  b64=$(_dsw_cmd "base64 -w0 '$rem'" | grep -aoE '[A-Za-z0-9+/]{40,}={0,2}' | head -1)
  if [[ -n "$b64" ]]; then printf '%s' "$b64" | base64 -d > "out/$(basename "$rem")"; echo "已保存 out/$(basename "$rem")"; else echo "拉取失败(命令可能不在输出中)"; fi
}

dsw-file-put() { # 上传小文件到实例(<2MB, base64)
  local localf="$1" remot="$2"
  [[ -z "$localf" ]] && { echo "用法: dsw-file-put <本地> <实例绝对路径>"; return 1; }
  local b64; b64=$(base64 -w0 "$localf")
  _dsw_cmd "echo '$b64' | base64 -d > '$remot' && ls -la '$remot'"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-usage}" in
    open) dsw-open ;;
    refresh) dsw-refresh ;;
    get) shift; dsw-file-get "$@" ;;
    put) shift; dsw-file-put "$@" ;;
    *) echo "dswctl — 用法: source 本文件拿函数; 或直接: dswctl <command>|refresh|open"; dsw "$*" ;;
  esac
fi
