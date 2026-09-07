# 00 — 从 V1 scaling-plan-1（无工具调用实验）汲取的经验 → v1-1-agentic 落地动作

> 来源：`wufuju2023-cell/new-reap-mcts-ttt-public`（分支 new-reap-mcts-ttt-public, commit c804bce）
> `explain/wiki/by-hand/scaling-plan-1/10-lessons-and-artifacts.md`（7 条可迁移经验）。
> 本页把每条经验映射为 **v1-1-agentic-tool-calls 的防重犯动作**。

| # | 经验 | v1-1-agentic 防重犯动作 |
|---|---|---|
| 1 | 先诊断后扩张（E1 门拦住 50k 错误放大） | v1-1 实验始终先跑 3 题 smoke × 稳定性 → 再过策略扩展（本批 E1 已验证：value 端点确定性；任何"规模放大"前重跑 E1 门） |
| 2 | 评估集决定结论（同族 0.67 vs 跨科 0.27 → E2 改 family-matched） | v1-1 目前 3 题（IMO/Pell 同数学族）——**只作机制验证**；结论必须标注"family-matched 小集"，后续扩容按 family 分层 |
| 3 | 温度是标尺不是分辨力（T=2.0 校水平、T=1.2 最优排序） | E5 温度扫描：记录"排序指标"（delta 排序）而非单点值；ant 变体生成统一 T 分离策略 |
| 4 | 类平衡倒 U 曲线（×8 塌 0.17，×2 中性） | 证据/变体池生成时**约束类平衡**，不做无上限翻倍；把 balance 系数写进 V11Contract |
| 5 | "已训练"≠"可复刻"（记录契约优先） | v1-1 全程已记录契约：artifact SHA（becf7c4…/c1d02…）、feature fingerprint、runtime 版本；每个 release 附 contract 注 |
| 6 | 长时运行工程化（断点/COMPLETE 门/防覆盖 guard） | 本 3h 实验：每个 E 阶段写完成文件；多副本（实验目录 + GitHub）不覆盖；防覆盖 guard 在 copy 脚本中 enforce |
| 7 | 版本化=回退成本 0（registry + NFS/HF 双副本） | v1-1 使用：GitHub 仓库（reap-agentic-v1-1 + v1-1-agentic-tool）+ 云端运行时目录 + 备份归档；所有修改 commit 前 diff 记录 |

## 附加：对 v1-1 明确禁止的动作
- ❌ 不对 value 单点做"分布宽"结论（确定性端点，E1 实证）
- ❌ 不把 3 题小集结论外推到整个数学领域
- ❌ 不在无 E1 门的情况下直接重启大规模/长时运行
