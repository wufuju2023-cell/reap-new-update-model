# AMD 控制桥（专用浏览器插件控制面）

替代 opencli CDP 通道的**专用、稳定、零依赖**方案，专门服务
`developer.amd.com.cn/radeon/instances/<id>/...` 站点（已登录态复用）。

## 架构

```
WSL: amdrctl2.sh ──HTTP──> bridge.js @127.0.0.1:19826 (Windows, 纯 node)
                               │ pending/reply（内存队列）
                               ▼ 800ms 轮询
Edge/Chrome: Extension（MV3, host_permissions 免 CORS, credentials:include 带登录 cookie）
                               ▼ fetch
developer.amd.com.cn API（contents/队列协议，同 amd_jupyter.sh 语义）
```

## 安装（一次性）

1. `mklink /D` 或复制本目录到 Windows（推荐 `D:\Apps\amdbridge`）：
   ```powershell
   Copy-Item -Recurse D:\Apps\amdbridge-extension D:\Apps\amdbridge
   node D:\Apps\amdbridge\bridge.js    # 常驻; 可注册计划任务开机自启
   ```
2. Edge/Chrome: `edge://extensions` → 开发者模式 → **Load unpacked**
   → 选择 `D:\Apps\amdbridge\extension`（`manifest.json` 所在目录）
3. 验证：
   ```bash
   ./amdrctl2.sh env   # 或 ls
   ```
   （bridge 不在时：提示 `connect refused`；扩展未授权时：`ext-fetch-error`）

## 与 opencli 对比

| 项 | opencli (CDP) | amdbridge（本方案） |
|---|---|---|
| 目标绑定 | page/targetId，易失 → about:blank | 无 target 概念（扩展常驻） |
| 长响应 | fetch 超时粘死 eval | 异步 pending/reply，2MB 截断保护 |
| 登录态 | 会话丢失需重登 | cookie 复用；过期手动重登 |
| 依赖 | npm 包+daemon+window | node（已有）+ 一条扩展 |

## CLI 子命令（同旧工具语义）

```
amdrctl2.sh push <path> <file>    上传文件
amdrctl2.sh put  <path> <text>    写文本（exec 命令 = put q）
amdrctl2.sh cat  <path>           读文件（out = cat qout | tail）
amdrctl2.sh ls   [path]           列目录
amdrctl2.sh exec <cmd>            提交命令给 runner.py 队列
amdrctl2.sh out                   取最近执行输出
```
