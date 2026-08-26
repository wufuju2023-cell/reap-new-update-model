# 04 · 环境双向同步（本机 ↔ 云端）与多会话（ensemble）

> 用户规则：本机 `/home/zhai/project` 已有的环境，**直接复制到云端**；云端环境**最终传回本机**——保证新实例随时能从本机复制环境，而不是从零下载。

## 1. 本机环境库（中心）
```
/home/zhai/project/cloud-env/
├ dsw_deploy_key            ← GitHub 私有仓库只读 SSH key（deploy key，已在 repo 登记）
├ dsw_deploy_key.pub
└ (未来) lean-4.28.0-rc1-linux.tar.gz、REAL-Prover-7B 权重等大件（放 /mnt/e/cloud-seed/，C 盘不足）
```
- 小文件（脚本/工程/数据集 jsonl）**以 GitHub 私有仓库为真源**；仓库已挂 **deploy key**（read-only SSH），实例用
  `GIT_SSH_COMMAND="ssh -i /root/.ssh/id_dsw_key -p 443 -o StrictHostKeyChecking=no" git clone git@github.com:wufuju2023-cell/reap-alpha-proof.git`
  或 gh-proxy（只对 **public** 仓库）——**私有仓库只能用 SSH/协议直连**。
- 大文件（模型权重/lean tar）：走 **hf-mirror/gh-proxy 网络通道**（流量对称），本机侧缓存目标 **`/mnt/e`**（E 盘）。

## 2. 传输通道优先级
| 规模 | 通道 |
|---|---|
| <1MB 小包 | 封入 auth code 或分块 `echo >>` + `base64 -d`（经 `dsw`） |
| 1–50MB | GitHub 私有仓库（deploy key）+ `git clone/pull` 到 `/mnt/workspace/v1/src` |
| >50MB | 直接下载：hf-mirror（HF_ENDPOINT）、gh-proxy、ModelScope；**双向同步仅记录 SHA+路径** |

## 3. 云端 → 本机
- 训练产物（rollouts/evals/scripts）→ 分区写入 repo 可拉路径（用户目录树）→ 由本机 `git pull`（或 dsw-file-get 分块拉回小件）。
- 权重/大产物留在 `/mnt/workspace`（NAS 持久）+ 本机 `E:\cloud-seed` 另存（按用户分配盘规则）。

## 4. 多会话开放（opencode-ensemble）使用规约
- 主会话：架构决策、任务编排、代码审查（本会话）。
- 子会话（按阶段委派）：`tools/` 脚本开发、数据集管线、评测 runner、课程生成、训练监控；每子会话只碰独立目录+README 交接。
- 交接物：`v1-spec/v1-1-training-methods/*` + 本目录（控制/同步）+ `checkpoints/` 元数据；子会话产出直接 commit 到私有 repo（不碰 token/key）。
- 分工粒度建议按 `07-roadmap` 的 M0–M14 拆；避免同一文件并发编辑。

## 5. 当前状态检查表（运行中）
- [x] deploy key 已登记（`dsw-deploy-readonly`，repo 只读）
- [x] `okilo_upload.sh` 分块上传工具（修 decode：`base64 -d` 单重定向）
- [x] `setup_week0.sh` 环境重建脚本（Lean prebuilt / reap / mathlib4 / REAL-Prover 15GB / 数据集 / FATE eval）
- [x] eval sets（fatem-100 / holdout-30）已在仓库并上传至实例（md5 校验通过）
- [ ] 实例会话稳定性（网关 commands 时好时坏——**卡则关页重开**，见 02 §2.6）
- [ ] 实际触发 `setup_week0.sh`（待会话可用）
