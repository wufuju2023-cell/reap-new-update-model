# next-1 — 环境迁移 + 8 小时实验计划（可复刻给任何能 SSH 到本容器的人）

> 目标读者：能 SSH 到 `dsw-2159645-…`（tailnet `100.91.25.4`）的协作者——
> 按照本目录可以把 v1-1-agentic 环境完整部署到**你自己的工作容器**，
> 然后执行约 **8 小时**的实验矩阵，产出与「每日进度」一致的证据包。

## 一页览

| 阶段 | 内容 | 预算 | 门 |
|---|---|---|---|
| P0 | 环境准备（Lean 4.28 + Python ROCm + 网络） | 40min | `lean --version`=4.28.0; `python -c import torch` |
| P1 | 资源获取（GitHub 代码 / HF 模型与 artifact） | 60-120min | 全部 sha 校验通过 |
| P2 | 部署 GPU server | 30min | `/health` ok; backend=real-search-categorical |
| P3 | 迁移验证（E1 门 + 带证据 smoke） | 40min | E1 sd=0; 3 题 PASS |
| P4 | 实验矩阵 E1-E5 + evidence loop | 4.5h | 各表输出至进度板 |
| P5 | 产出整理 + 提交（GitHub 或回传） | 30min | 证据包齐全 |

## 分发顺序（优先级）
1. **GitHub**：`wufuju2023-cell/v1-1-agentic-tool`（main：cpulean Lean + driver + smoke + gpu/gpu_runtime 33 模块）
2. **HuggingFace**：`WufuJu/v1-1-fullv3-artifact`（public：`backend.full.pt` 3.74MB 头 / `backend.raw.pt` 165MB adapter+head）
3. **缺什么再直拉**：本容器 `/home/admin/workspace/`（NFS 持久；读只加写——⚠️ 勿覆盖他人运行树），
   可通过 `scp -P22 root@100.91.25.4:...`（密钥在 `git@github` 侧/README 咨询）

## 资源清单与哈希（P1 校验值）

| 资源 | 来源 | sha256 |
|---|---|---|
| 全库 git | `https://github.com/wufuju2023-cell/v1-1-agentic-tool` | commit `930bf6e`（参考） |
| full-v3 artifact 头（backend.full.pt） | `https://hf-mirror.com/WufuJu/v1-1-fullv3-artifact` | `c1d0255221cc1b3a6e80e1d5567f6c535ef0a62f1ab300d237aeaa2610e0385d` |
| adapter+head（backend.raw.pt） | 同上 | `5b227eb2d95893b7d590b3e270b08e6ca9c8c964384fd4a8346cdd01f3b40748` |
| REAL-Prover base | `huggingface.co/FrenzyMath/REAL-Prover`（公开 4 分片） | 模型锁 `5e5e7472…`（runtime 会自校验） |

> 其余细节/命令见 `00-`, `01-`, `02-`, `03-` 四篇子文档（本目录）。
