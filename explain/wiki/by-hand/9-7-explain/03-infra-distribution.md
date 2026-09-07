# 03 — 基础设施与分发：为什么别人能复刻

## 通道矩阵（现状 2026-09-07）
| 地点 | 身份 | 说明 |
|---|---|---|
| 本机 Windows/WSL | zhai（sudo 免密，root key 可行） | 出站全通（GitHub/hf-mirror/容器 tailnet） |
| 我们容器 dsw-2159645 | root（agent-root-mgmt key）+ zhai | MI300X 196GB；GPU server :8000 在线（339 completed 0 failed） |
| 协作者容器 | 按 next-1 自行部署 | 已成功复现（100.68.136.61） |

## 持久化/恢复事实（多轮实战验证）
- 唯一持久区 `/mnt/workspace`（NFS）：
  models/（REAL 15G、Qwen 19G 系）/ new_value_head/（47G 训练树，服务保留）/ backups/。
- `restore-container.sh`：tailscale 免授权恢复、双 SSH 用户、sshd 加固、rootkey、home 属主修。
- 重要教训：**backup 会覆盖旧归档**（已加"缺 tailscaled.state 拒写"保护前曾丢过一次；
  现脚本 v4.2 已有 guard）。

## 分发优先级（next-1 已固化）
1. GitHub 代码：`v1-1-agentic-tool`（main，含 gpu/gpu_runtime 33 模块 + torch2.11 补丁）；
   文档/Branch：`reap-new-update-model` 分支 `new-reap-mcts-ttt-public`（explain/wiki/by-hand）。
2. **HF public**：`WufuJu/v1-1-fullv3-artifact`（`backend.full.pt` 3.74MB head；
   `backend.raw.pt` 165MB adapter+head；LFS oid 均已验）。
3. 兜底：`root@100.91.25.4:/home/admin/workspace` 只读直拉（禁止写入运行树）。

## 环境事实
- GPU 栈：torch 2.11.0+gitd0c8b1f / transformers 5.14.1 / peft 0.19.1（ROCm）。
- Lean：容器评估基线 4.28.0（`lean-toolchain`）；本地 E 盘 mathlib 4.33 仅开发；
  **产物以 4.28 语法为准**（生成代码须 4.28 兼容——Executor 产物测试时注意）。
- 网络：容器 hugginface.co 直连 ✗、hf-mirror ✓、github ✗（codeload/api ✓）；
  WSL 侧全通。
