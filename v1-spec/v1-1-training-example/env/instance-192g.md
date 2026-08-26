# env · 192G DSW-AMD 实例（当前主训练机）

| 项 | 值（2026-08-25 快照） |
|---|---|
| 实例 ID | 以现场为准（**每次重开都会变化**；从「我的 Notebook」查看） |
| 网关段 | 以现场为准（`dsw-refresh` 自动发现；旧值仅参考） |
| 规格 | 23 vCPU / 200GB RAM / **AMD GPU 192G（rocm-smi 显存205.8GB）** |
| 镜像 | `ubuntu22.04-rocm7.2.3-py312-torch2.11.0-1.39.0`（DSW-AMD） |
| torch | `2.11.0+rocm`（gitd0c8b1f） |
| python | 3.12 |
| workspace | `/mnt/workspace`（NFS 共享盘，配置与数据持久） |
| 数据根 | `/mnt/workspace/v1`（models/ reap/ v1project2/ mathlib-src/ services/ runs/ ...） |
| 剩余额度 | 94h41m（AMD 类型 / 360000s） |
| 访问通道 | yang-win opencli + `dsw/commands`（ms-exec.ps1，cookie=yang-win Edge 登录态） |
| 实例自动关闭 | 闲置 >1h 自动停（AutoShutdown:true）；页面上从"连接运行时"重启 |

## 服务器 ↔ 本目录（v1-1-training-example）同步状态

| 资产 | 服务器 md5 | 本目录 md5 | 校验 |
|---|---|---|---|
| services/policy_server.py | 09f907419a99beef75abf34c8935b897 | 同 | ✅ 2026-08-25 |
| services/ps_server.py | 96b05d2760ec5f1b1a42d5b68f537403 | 同 | ✅ |
| runs/eval_sets/fatem-100.jsonl | 8e5dfd70b2f80b147ad650f13d9132dd | 同 | ✅ |
| runs/eval_sets/holdout-30.jsonl | d771e451f6f784dd8abbf5f01f919f84 | 同 | ✅ |

## 待办（Week-0 续）
1. 安装 Lean 4.28.0-rc1 **预编译** 工具链（520MB tar，之前下载未完成）→ `/opt/lean4` + PATH（新实例无 lean）；
2. `mathlib-src`（已 129M 源码）→ **`lake exe cache get` 预编译缓存**（不源码编译）；
3. policy_server.py 依赖 transformers+torch ✓ 已在镜像；启动服务 + FATE-M smoke；
4. 按 `07-runbook` 登记 runs checkpoint。
