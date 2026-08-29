# REAL7B full-v3 成功课程案例

本目录是一份可脱离原开发机器阅读的完整案例包。它解释并保留：value head 如何训练、如何以负剩余距离进入 MCTS、如何在题内与 policy LoRA 联合 TTT、课程为何这样设计，以及模型如何最终生成由独立 Lean 接受的证明。

## 一句话结果

REAL-Prover 7B 加载由 205,628 个隔离 LeanTree 状态训练的 full-v3 64 档距离头，在平方望远镜和课程上运行真实 MCTS；搜索在 step 7、15 两次联合更新 policy LoRA 与 value head，两个新版本都被后续生成和 value 前向真实消费，并在 checkpoint 20 自然得到完整证明。固定镜像、禁网的独立 Lean 验收通过。终端成功轨迹又完成一次 v2→v3 联合更新。

这是“课程题成功”，不是原立方 H1 已成功；本课程也没有 exact-random 同题臂，因此它展示端到端机制与能力，不单独证明 full-v3 相对随机头的因果优势。该优势请结合已有 F1 matched pair 阅读。

## 阅读顺序

1. [01_价值头训练.md](01_价值头训练.md)：数据、标签、结构、损失、离线指标与随机对照。
2. [02_MCTS与TTT设计.md](02_MCTS与TTT设计.md)：value 如何影响搜索，以及题内联合更新如何发生。
3. [03_课程设计与真实运行.md](03_课程设计与真实运行.md)：课程设计、真实 session 时间线和成功边界。
4. [04_实验结果与解释.md](04_实验结果与解释.md)：结果、能得出的结论、不能得出的结论。
5. [05_复现方法.md](05_复现方法.md)：三层复现方法。
6. [06_模型生成的Lean证明.md](06_模型生成的Lean证明.md)：模型产出的精确 proof 与验收结果。
7. [07_代码配置与证据清单.md](07_代码配置与证据清单.md)：随包代码、配置、报告清单，以及未随包分发的环境前置条件。

`code/`、`config/` 和 `results/` 已包含体积较小的真实实现、运行配置和训练报告。7B 权重、训练特征与运行快照不复制，也不提供对外不可访问的本机寻址；它们只在复现文档中作为需要另行取得的前置条件说明。

## 固定语义

```text
非终态：head 输出 d = E[k], k ∈ {1,...,64}
搜索：V(s) = -d
已证明终态：d=0, V=0
Lean 非法动作：直接丢弃
形式化死节点：-∞
预算耗尽/未找到证明：unknown，不是 -∞
```

## 封存状态

完整闭环已完成，learn 从未重发。严格离线恢复只补写既有 v3 success-learn completion；随后原 session 完成 seal、发布 experience `exp-078cea5460cb-01`、snapshot 与 receipt-backed retire，family 状态为 `completed/published/retired`。原始 proof/trace/回执的 21 文件证据已完整收入 `results/evidence/`。
