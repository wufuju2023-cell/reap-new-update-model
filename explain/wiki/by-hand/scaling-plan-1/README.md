# Scaling Plan 1：REAL7B MCTS/TTT 扩大与 AlphaProof 能力逼近

| 项 | 值 |
|---|---|
| 审计日期 | 2026-09-07 |
| 当前系统 | REAL-Prover 7B frozen + LoRA policy + 64-bin categorical value head（full-v3），MI300X 192G × 1 |
| 锚点论文 | AlphaProof 原文（以 2026-09-05 01-paper-baseline 引文为准）+ AlphaEvolve arXiv 2511.02864 |
| 入口动作 | 一个小实验门（§02），通过后才进入规模阶梯（§03） |

## 本目录

| 文档 | 内容 |
|---|---|
| [00 总览 README](README.md) | 本页；范围、决策原则、索引 |
| [01 AlphaProof 锚点](01-alpha-proof-anchor.md) | 原论文与 AlphaEvolve 的可量化对齐点与差距矩阵 |
| [02 先做的小实验](02-small-experiments.md) | E1 calibration / E2 value-on-off / E3 support & overflow 审计——通过门后才有资格谈扩张 |
| [03 规模阶梯](03-scaling-route.md) | S0→S3 四步；每步输入、运行方式、评测、gate |
| [04 资源地图](04-resource-map.md) | 硬件、数据、代码、时序与成本预算 |
| [05 并行任务与风险](05-parallel-and-risks.md) | 可并行项、失败模式、监控与事故规则 |
| [06 GPU probe 报告](06-gpu-probe-report.md) | 1h 链路验证：加载/policy/value 时延、显存、远端饱和信号 |
| [07 E1 calibration 结果](07-e1-calibration.md) | **223 样本正式结果：ρ=0.339（未过门）、系统性低估 57%、长距离饱和 → 不进 S1，转 Head 改进线** |
| [08 H1 长尾重训 + H2 温度标定](08-h1-longtail-reweight.md) | H2：T≈2.0 修水平校准（ECE 5.9→1.7）但 ρ 仅 0.30；H1：79e 长尾数据重训 head → 同分布排序 0.67、**跨科目 ρ 反降至 0.27**（family gap 是瓶颈，不是缺数据）|
| [09 混合变体矩阵与版本化](09-mix-variants.md) | 自主管线 磨头（make_features/mix_train/e1_eval + registry + HF 发布）：6 版本矩阵 E1 ρ=0.34(原头)→0.17(×8)→0.33(×2)；**v3 原头仍是未被复刻的基线；E1 转为诊断维度，E2 改为 family-matched 设计**；HF: `WufuJu/reap-value-head-v1-scaling` |

## 决策原则（本次计划的不变量）

1. **先证明有用，再扩大规模**（沿用 9-5/9-6 门：value-on 未通过前不堆 steps）。
2. **一切结果以独立 Lean 定稿**；伪成功不入"成功案例"。
3. **机制真实验证**：每次联合 TTT 必须记录参数 diff、版本递增、被后续生成/value 消费。
4. **规模信息透明**：每次扩容报告 source mix、class histogram、support coverage、compute 与 gate 结果。
5. 资源约束是核心变量：**单张 MI300X（192G）**，没有多卡/集群；算法路径必须按此设计（批时序混用、CPU search + GPU head/TTT 交替、渐进式预算）。

## 现状基线（直接引用 9-6 记录）

- 已成功：full-v3 H1 平方望远镜课程闭环（checkpoint 20，step 7/15 两次联合 TTT，独立 Lean 通过，exp-078cea5460cb-01 已发布退役）
- 已成功：F1 matched-random 因果证据（full 成功 / random 同预算失败）
- 已上线：HF `alpha-proof-open-source/alphaproof-full-v3-value-head`（64-bin；SHA 已登记）
- 未完成：heldout calibration、value-on/off、原立方 H1 完整证明、20 题矩阵
- 训练规模：full-v3 为 205,628 状态集群训练；在线 TTT 为两次 batch（9 replay + 1 Mathlib /batch）——**机制级**
