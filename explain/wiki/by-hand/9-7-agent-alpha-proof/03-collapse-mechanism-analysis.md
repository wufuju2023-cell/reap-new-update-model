# 03。多样性/坍缩机制分析：什么时候"只产生一种答案"

## 区分两层"多样性"

1. **树结构多样性**：同一状态下的不同 tactic / 不同节点路径。维护者 = progressive sampling、K 采样、等价合并、AND-priority、policy prior 的熵。
2. **轨迹层多样性**：不同证明路线（不同结构选择、不同 lemma/拆分策略）之间的差异。维护者 = 数据分布（课程、variant 池、replay mix）。

## Agent 接入后，坍缩会在哪里发生？

误导性说法："Agent 是 LLM → 会被训练成一个答案"。错的。

不坍缩的条件试验：

| 场景 | 生态位 | 会坍缩吗 |
|---|---|---|
| Agent 只产**变体题**（树外），回放数据=全树所有验证轨迹 | 分布多样 | 不会 |
| Agent 产**单链证明**，且 replay 只采样这条链 | 分布=单点 | 会（学到的正是"这一种"） |
| Agent 接管**每节点出招**，K=1 单步采样 | 树=单线 | 会（不是训练坍缩，是搜索坍缩） |
| Agent 产多条不同路线，全部 Lean 验证后同时进 replay（含 90/10 混批） | 分布=多变体 | 不会 |

## REAP 已有的防坍件

- `progressive sampling`：n(s) ≤ C·N(s)^α 再采样 K —— 即使 prior 单峰也强制重采样；
- 等价后继合并 —— 合并同态，不放大重复；
- AND/OR backup：结构性分支；
- mixed learner：9 条 verified replay + 1 条 Mathlib（或按论文 90/10）——**基础混批已在**。

## 缺失的一环（Agent 补位点）

课程池内**题目级差异**由"题目选择"决定，但**解题路线差异**目前只能来自 policy/replay 的自然分歧；人工/程序化变体（TTRL）在 REAP 里是缺口（9-5 L 档 TTRL：target+variants+paired effects 尚未做）。→ Agent（尤其多 agent 并行/higher reasoning flavor）天然是"路线级差异"的生产者：同一题给出 3-5 种不同分解/引理策略，每一条过 Lean 验证后入池。

## 结论（防坍设计）

把 Agent 接到：
```
Agent(变体+多路线) → Lean 校验 → 课程池/variant 池 → course(CourseSquareTelescope...)
→ 树内仍由 GPU policy 采样，单链 agent 产物仅作为 replay 的 1/N 条
```
则 LLM 的训练分布保持多样；"只产一种答案"的担忧只在"Agent 单链回放 + 无变体池"的设置下成立——而该设置恰好不是我们推荐的接法。

## 需要监控的坍塌早期信号（放入运行门）

- 每 round：有效 proof 文本 diff / action-set Jaccard 中位数；≤1 类占 round 轨迹数比例；
- replay 中"同题不同路线"的路线数 < 2 → 触发新增变体预算；
- policy 的 action-entropy 下降速率（熵 floor 门）。
