# 02 — v1-1 研究假设：Agent 证据环 = 贝叶斯先验/后验调节器

## 一句话假设
把 opencode 式 agent 工具环作为**树外证据源**，让 GPU policy 的推理分布
（先验/后验）被证明知识以受控方式重新加权，从而改善固定预算搜索。

## 数学形式（严格版，见 9-7-series 09）

$$
P(t \mid s, R) \propto P_\theta(t \mid s) \cdot P(R \mid t, s)
$$

| 通道 | 作用 | 工程落点 |
|---|---|---|
| B：证据条件注入 | 改变后验（θ 不动） | evidence→prompt/hints |
| C：selection prior 覆盖 | π'(a\|s) ∝ π(a\|s)·φ | Lean TreeSearch selection（φ=tanh(Σw)） |
| A：证据转回放 | 下一轮先验迁移（θ 更新） | replay / mixed learner |

φ 组合律（Lean `Agentic.lean` 已实现）：

$$
\phi = \tanh\!\Big(\sum_{k} w_k\Big),\qquad w_k \in [0,1]
$$

## 设计红线（RULE 集，摘要）
- RULE-0：树内采样永远由 GPU policy（K 多样）承担；Agent 永不逐节点出招。
- RULE-1：任何 Agent 产物必须过 Lean 验证门（syntax→去重→verified 才入池），零例外。
- RULE-3：回放数据必须多路线（≥2 同题路线 or 泛化自课程题数），否则仅进 evidence。
- 证据熵门：路线 Jaccard < 0.7 才算"不同"（07 系列）。

## 当前已实测量
- E2/闭环：证据内容注入 → ΔV 显著（+3.2~+8.8）——**B 通道已实证**。
- φ 尚未直通 GPU：**C 通道是下一步主攻**（φ→selection）。
- opencode 真后端（将 A/B 合并）：目前证据为骨架项（`OpencodeEvidenceBackend`）。
