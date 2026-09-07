# 00 — 时间线：每个阶段在做什么，留下了什么

## S0：通道与可重复（8 月下旬-9 月初）
- tailnet：laptop-ir9bs02n（100.83.92.75）、容器 dsw-*（100.87.60.87 → 现 100.91.25.4）、
  Windows sshd :22、容器 zhai+root 双通道免密。
- 备份体系：`restore-container.sh`（backup/restore/rootkey 子命令）、
  `/mnt/workspace/backups/container-config.tgz`、脚本内嵌公钥。
- 教训沉淀：容器唯一持久区 = `/mnt/workspace`（NFS）；rootfs 全部随实例丢（已有 3 轮
  销毁-恢复演练）。← 这是"长时实验可复刻"的地基。

## S1：V1 研究（8 月 28-29 日主力）
- 205,628 LeanTree 状态 → 冻结 REAL-Prover 7B → `3584→256→64` 分类头（full-v3）。
- 课程 `CourseSquareTelescope`：MCTS→step7/15 两次题内联合 TTT（v1/v2）→ 完整 Lean proof → 独立验收 → seal/publish/retire（`exp-078cea5460cb-01`）。
- F1 matched-pair（full vs exact-R64 random）：full 成功、random 失败——相对因果证据。
- 成果入 GitHub `discussion/new_value_head_in7b_ex1`（master）+ HF private 220MB 后端。

## S2：v1-1-agentic 工程（9 月 7 日）
- 决策：Lean CPU（v4.28 容器兼容，本地 mathlib 零大编译）+ Python GPU（跨设备契约）+
  opencode 证据环。仓库：`v1-1-agentic-tool`（public）。
- 真调链路：WSL→tailnet→MI300X server；E 系列 9/7 日全绿 + next-1 可迁移文档。

## S3：协作者复现（9/7 晚）
- `dsw-2160882-…/100.68.136.61` 全链自我复现（P0-P5+补充审计）。
- 三类发现：E4 弱多样性题、REAL base 上游锁 digest 变更、runtime_transport 解析/执行补丁。

## 时间线小结
```
通道/备份 ──► V1 研究 ──► v1-1 工程(Lean+GPU+证据环) ──► 复现+差异发现 ──► 深实验路线图
   S0           S1             S2                        S3              现在(v2)
```
