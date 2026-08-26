# 02 · opencli 浏览器自动化（经验/坑位）

> 目标站点：ModelScope 代码工作区 / DSW 网关（`dsw-gateway...`）、**AMD Radeon Cloud**（`developer.amd.com.cn/radeon`）。
> 统一操作模式：**复用已登录的 Edge（opencli Browser Bridge）**，不让 `dsw/commands` 的会话丢。

## 2.1 基本流程
```bash
opencli doctor                       # daemon + extension + connectivity 均 OK
opencli browser <session> open <url> # 复用登录态打开目标页
opencli browser <session> state      # 拿 url/title + 可交互 refs（[N] 数组）
opencli browser <session> eval <js>  # document.body.innerText / fetch / click JS
opencli browser <session> click <N|--name "文本">    # 真实 CDP 点击
opencli browser <session> screenshot <path>          # 读 canvas/屏幕
```

## 2.2 关键坑位（本案例真实踩过）
1. **`click` 文本≠CSS**：`click "New Terminal"` 会当 CSS 选择器 → `selector_not_found`；文本要用 `click --name "文本"` 或传数字 ref。
2. **antd/大前端 dialog**：Modal 会挡住；`document.activeElement` 常常是 dialog 按钮，需先关 `Keep Waiting / No / Close`。
3. **eval 返回值规则**：
   - 简单 `1+1`、`document.title` 稳定；
   - **重前端页（JupyterLab / Hermes / ModelScope SPA）`eval` 经常空白或 115s CDP 超时**——因为页面主线程被大量 WebSocket/渲染占满。
   - `eval` 内禁止 `return` 于顶层；用 `(function(){...})()` 或 `(async()=>{...})()`。
   - bash 引号：外层建议 `bash -c '...'`，JS 内用单引号，内嵌双引号逃逸 `\"`。
4. **多 tab / bind 混乱**：`open`/`tab new` 后 session 可能 bind 到 `about:blank`；`opencli browser <session> bind` 重新绑定；`tab list` 排查。
5. **页面被关/切走 → 会话失效**：`dsw/commands` 变 `GATEWAY_ERR` 或 `eval` 返回空 → **重新 open workspace 即可恢复**（ModelScope 已知坑）。
6. **`cdp_timeout 115s`（native dialog / 页面忙）**：先用 `opencli browser <session> dialog accept/dismiss`，或 `sleep 30–90` 等页面负载降下来再 eval。
7. **screenshot 输出到 Windows 侧路径**：opencli 在 Windows 执行时 path 是 Windows 路径（如 `C:\tmp\opencode\x.png`）；且在 WSL 读不到，需 base64 或走共享。
8. **iframe**：Hermes `/notebook/main.html`、`hermes-dispatcher` 是跨/同源 iframe；同源 `iframe[title=notebook-ide].contentDocument` 可 access；终端 canvas 文本读不到（只能截图）。

## 2.3 处理步骤模板（恢复命令通道）
```bash
opencli browser <session> open https://www.modelscope.cn/code/workspace
sleep 8
opencli browser <session> eval "(async()=>{const r=await fetch('https://dsw-gateway-cn-hangzhou.data.aliyun.com/<SEG>/dsw/commands?type=status&name=next-ide',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:'echo OK; hostname'})});return (await r.text())})()"
```

## 2.4 触发实测结论
- **确定性可用**：`dsw/commands`（返回 output JSON）；WebIDE 终端可交互。
- **不稳定**：`eval` 在 JupyterLab/Hermes 页（占用大）；`xterm-helper-textarea` 注入在部分实例（canvas/Hermes）不被接受。
- **不强求**：如果能走「个人云账号实例 SSH」或「固定 token」，尽量不再依赖 opencli。

## 2.5 其它已记录站点（AMD Radeon Cloud）
- 站点 `developer.amd.com.cn/radeon`（antd SPA）；Profile 页含 Credits/实例/Model API Key（`/api/v1` OpenAI 兼容）、SSH Public Key（贴公钥）。
- SSH-enabled 模板：`HuggingFace(788)`/`MiniCPM-v46(2568)`/`ComfyUI(645)`；模板 257 Hello ROCm `ssh_enabled:false`。
- 限流/SSO：`{"detail":"Authentication service is temporarily unavailable"}`＝平台 SSO 临时故障；`Instance is still starting` 时 SSH 未就绪要等。

## 2.6 会话恢复 SOP（2026-08-26 实测，用户定规则）
**规则：发现卡就「关掉页面重开新页面」，不要反复修复。**
1. `opencli doctor`（daemon+扩展 OK）→ 卡则 `opencli daemon restart`
2. `opencli browser <session> close`（释放 lease，或 tab 清理）→ **open 唯一宿主页**：
   - ✅ **当前唯一使用**：`https://modelscope.cn/code/workspace`（轻、稳、同会话、跨域可调 gateway commands）
   - ⛔ **gateway `/lab?appId=MAAS&instanceId=…&features=…` 版已停用**（不写具体 URL；如确需备查，用户现场获取，且勿入库）
   - **重启实例规范**：关掉旧 gateway lab 页 → 从 **modelscope「我的 Notebook」页面**重开 → `instanceId`（及网关段）会变 → 运行 `dsw-refresh` 自动发现新段。
3. **关键**：等页面 `document.readyState === "complete"`（fetch/eval 才稳，约 10–20s）；
4. 再 `dsw "echo OK"` 验证（200 + output 即恢复）。
5. 实例重开会变 `instanceId`（URL 里）与可能变 SEG——用 `dsw-refresh` 更新缓存。
