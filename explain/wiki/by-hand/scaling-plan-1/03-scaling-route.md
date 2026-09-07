# 03 — 规模阶梯 S0→S3（每步有 gate）

> 硬件基调：1 × MI300X 192G + 23 CPU 核。单卡既是引擎也是瓶颈；
> 一切调度按"单卡时间窗"设计（推理/搜索/TTT 分时复用），不假设集群。

## S0（已处于）—— 机制验证（完成）

- 已有：H1 平方望远镜闭环（checkpoint20, v1/v2/v3 消费链），F1 matched，64-bin artifact 上传
- **退出条件**：E1/E2/E3（§02）门通过

## S1：verified replay 扩容（≈1 周，E1/E2 通过后）

输入：成功/失败课程树 → 抽取 verified state–action rows（保持 9:1 Mathlib 配比）

| 增量 | 数量 | 说明 |
|---|---|---|
| 训练 rows | 10k → 50k | 从 MS 调度（ms-epoch2）与 H1/H1-variant 重新生成；state dedup + theorem-level split |
| 在线 TTT 窗 | 每 checkpoint 窗 2→8 updates | 8 窗 × (10 replay + 1 Mathlib) |
| 训练规模 | 每次 head+LoRA join（全参 freeze-check）| 单卡 batch 32 × 4 epoch ≈ 2–4h/轮 |

评测（释放 gate）：
- 训练后 classification logloss ↑/↓ 对照 holdout
- 新增 **feature-parity check**：同 prompt 同 seed 下新旧头对同一 state 的 d̂ 排序变化量
- 发布二级 artifact（V1.2）并 maintain 与 full-v3 的列联表（不会互相污染）

S1 pass 条件：holdout calibration 不变差，**且** value-on 在 E2 三题上的 solve@budget 净增益 ≥ +8%（平均）。

## S2：课程与 variants 生成（≈2–3 周，S1 后）

目标：AlphaProof 的"target + variants"的迷你版——不是 500k，而是 **1k × variants/题** 起步。

| 组件 | 设计 |
|---|---|
| 变体生成器 | prompt 示例 + 程序化变换（规模：N_evo 3–5 轮；生成 20–50 → Lean 校验 → dedup 择优） |
| 变体集 | 每题 200–1000 verified variants（先 H1-square/H1 立方族） |
| 调度 | 单卡 CPU 生成（23 核并行 Lean verdict）+ GPU head 评分分批消费 |
| 课程 | 通过率 gate：同题同预算 solve@budget 增长曲线 vs variants 数量（10/50/200/1000） |

S2 输出：paired target matrix（search-only / target-only update / variant-focused），
证据足以写 TTRL 板块的第一个"有质量的小规模"论文小节。

## S3：main-RL 风格混合训练（远期，S2 后才考虑）

- matchmaker 的轻量版：prove/disprove 混合 + 失败重试预算 + 新颖性优先
- learner 混批扩展为 50k–200k verified rows（需要多阶段课程树积累）
- **不追逐** 1M steps；先跑 10 轮 release 曲线，论证每轮增量
- 远期才有意义的前提：S2 各门通过、S1 的 calibration 方差稳定（ρ 波动 < 0.05）

## 成本预算（单卡估）

| 阶段 | GPU 时 | CPU 时 | 说明 |
|---|---|---|---|
| S1 | ~80–120h | ~60h | 生成与训练交替 |
| S2 | ~200h | ~200h（Lean 变体验证）| 分周运行 |
| S3 | ~400h+ | ~400h+ | 视 S2 结果决定是否进入 |

> 每周可调度窗口按"单次实例剩余时长"计；**每窗口收尾必须 backup（防归档空档 bug 已修）**
> 与 evidence 落盘；长任务拆成 12h 内可完成的批。
