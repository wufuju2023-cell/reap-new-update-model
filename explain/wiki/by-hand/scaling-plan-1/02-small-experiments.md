# 02 — 先做的小实验（通过门才谈扩张）

> 原则：三个实验全部使用 **uploaded full-v3（64-bin）** 同一 artifact、同一 base revision
> （FrenzyMath/REAL-Prover fe76f68d）、同一 freeze 配置。全部允许"不过"，
> 但必须"有可解释的数字"。

## E1：Critic Calibration（半天，CPU 为主）

**问题**：预测的剩余距离 d̂ 是否与 verified remaining distance 有序/校准关系。

| 项 | 设计 |
|---|---|
| 数据集 | theorem-level holdout：从已成功课程之外选 12–20 题（含平方、立方、等价替换族），每题独立 Lean 生成 20–64 步 verified tree → 抽出 (state, true_distance, d̂) 对 |
| 指标 | Spearman ρ、ECE（按 bin）、overcoverage（d̂ 低估/高估列表）、class histogram 对照 support |
| 门 | ρ ≥ 0.55 且未见类饱和 ≤ 25% → 通过；失败则记录为"value 不可靠"证据并停 E2 部分 |
| 输出 | holdout report json + 一页解释（可写入 discussion/04 状态） |

## E2：value-on/off 固定预算单点（2–3 天，GPU 为主）

**问题**：同题、同 prompt、同 policy release、同 candidate 数、同 simulation budget，value-on 是否快/多找到 verified proof。

| 项 | 设计 |
|---|---|
| 题目 | 3 题（1 题同 H1 难度、1 题平方、1 题可控立方）+ 3 题对照（同上但 exact random 头） |
| 配置 | 每题 3 臂：a) value-on（full-v3）；b) value-off（uniform V=0 或纯 policy）；c) trained 但冻结（无在线 TTT）—— 后两者分离"值头即时机 vs 值头本体" |
| 预算 | 固定 simulations（3k/题峰值）和 wall-time 上限，多 seed（≥3） |
| 指标 | verified solve@budget、nodes-to-proof、wall time、失败日志分类（dead/unknown/depletion） |
| 门 | 至少 1 题 value-on >（b）且 solve @ budget 平均正向；全部持平→记录为中性、不发"扩容" |
| 输出 | paired matrix + evidence tree（入 discussion 02_成功案例候选或 04_状态） |

## E3：support & overflow 审计（半天，只读）

**问题**：64-bin 对 d>64 的真实轨迹如何处理；分布是否偏斜到 d∈[1,4]。

- 统计训练分布中 d 直方图；上报 d=1..64 的 P95
- 对 holdout 中未覆盖的 d 段如何被处理（拒绝/统计/溢出值）——固化到 contract
- 输出：support-overflow-report.md（后续任何 support 修改（如 D=128）都以此为基线迁移文档）

## 决策分支

```
E1 过 & E2 正向 → S1（replay 扩容）
E1 过 & E2 中性 → 记录中性，做 S0.5（prefix: 分析为何中性；改 selection 温度等机制后重测）
E1 不过 → 回到 head 训练侧（看 distance 目标或特征），不做规模扩张
```

## 预计资源

| 实验 | GPU 时 | 说明 |
|---|---|---|
| E1 | ~4h | 数据生成（Lean run tree）为主，推理轻 |
| E2 | ~36–48h GPU（单卡批跑） | 3 题 × 3 臂 × 3 seed 的设计要压缩到单卡窗口；可拆多日 |
| E3 | 0 | 纯离线统计 |
