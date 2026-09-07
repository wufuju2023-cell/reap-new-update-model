# 04 — 资源地图：硬件、数据、代码、时序

## 硬件（实测 2026-09-07）

| 项 | 值 |
|---|---|
| GPU | AMD ROCm 7.2.3，**1 × MI300X-class**（0x74b6，VRAM **205.8 GB**，650W cap） |
| CPU | 23 核（~200GB RAM） |
| 备注 | Xetum 型 `nproc=23`；无多卡无集群。FPS/算力只够"1 个 7B 引擎"运行 |

容量含义：
- 7B fp16（~14G） + LoRA（~0.2G） + 激活/梯度 + optimizer ≈ 可同时容纳 **2–3 个样本并行 batch 处理框架**；在线 TTT 与 MCTS 推理可共存于 192G
- 但峰值推理批（16×4k tokens）会占 ~60G，建议**推理与训练分时**（进程隔离）

## 数据资产清单（/mnt/workspace）

| 目录 | size | 用途 |
|---|---|---|
| `new_value_head/features-*`, `full-extension-features` | 1.6G | full-v3 205,628 状态特征（重训/对照用） |
| `new_value_head/data*`, `dataset*` | 0.5G | 训练数据本体（保留不删） |
| `new_value_head/pell-fullv3-resume9` snapshots | 23G | 课程树（H1 + 运行中） |
| `categorical-snapshots-*` | 13.5G | 快照树（部分可压） |
| `reap-runs/ms-epoch2-*` | 15G | MS 调度 run 树（S18 候选/矩阵） |
| `models/REAL-Prover-fe76f68d` | 15G | **评测与训练必需 base**（保留） |
| `backups/*` | <0.1G | 恢复脚本/tgz/hf-token |

**AI 输入规范**：新生成的数据建议在 workspace 下建立 `datasets/verified-replay-vN/` 统一管理（by-hand 计划中记录位置），与旧目录不混用。

## 代码与复现

| 来源 | 路径 | 状态 |
|---|---|---|
| V1 runtime 源码 | `/mnt/f/projects/reap-new-update-model-value-head/v1-result/source/current/`（experiments/lean_campaign/reproduce） | 完整，与 release hash 对齐 |
| 成功案例完整包 | `v1-result/20260828-real7b-pell-success/`（code/evidence/weights/validation/proofs） | 权威 |
| 课程案例源码 | `reap-new-update-model` master `discussion/new_value_head_in7b_ex1/02.../code/` | train_value_head / course_driver / online_ttt / categorical_search_backend |
| 复现手册 | `03_复现与代码/01_完整复现手册.md` | 待随 S1 更新 |

## 时序（关键约束）

- 容器实例 12–24h 级别生命周期；实例重建后 5 分钟内可恢复（v4.4 restore）
- **每个运行窗收尾**：evidence 落盘 + `restore-container.sh backup`（防覆盖 guard 已生效）
- E2/S1 均应按"12–18h/窗"切批，跨窗状态放 workspace（NFS 持久）

## 外部镜像（已验证通）

| 目标 | 结论 |
|---|---|
| hf-mirror.com | 上传/下载全通（容器内） |
| huggingface.co | 容器不可达；WSL 可达 |
| github.com | 容器不可达；codeload/gh api 可达（WSL git 直连同） |

> 计划内的数据/artifact 落点首选 HF（WufuJu / alpha-proof-open-source orgs），代码提交用主 repo。
