# 00。问题与问题的表述（9-7)

日期：2026-09-07。

## 用户问题的两句话还原

1. 把 Agent 编码工作流（OpenCode 式的"终端工具调用 → 读取 → 生成"循环）跟 AlphaProof 式价值头/MCTS/TTT 系统结合，接口在哪？
2. 担心这样接入后，"LLM 会被训练成只产生一种答案"——即 Agent 做多路径生成时，是否天然坍缩到单一证明路径。

## 先给一句话结论

- 接入点存在，但**唯一的正确主接点是"课程/变体生成器"（TTRL 的 variant 层），而不是"树内采样器"或"每个节点的 tool"**。
- 坍缩风险是真实存在的，但机制不是"Agent 是 LLM"，而是 **training replay 的轨迹分布**：若反馈环只回放 Agent 单条成功链（one-chain-only），数据分布坍缩 → 该轮学到的是单点解。
- 论文级防护与 REAP 自身机制已有"树结构多样性"（progressive sampling / 等价合并 / AND-priority），缺的是**轨迹层多样性**，而这恰恰是 Agent 能补的。

## 文档系列结构

| 文件 | 内容 |
|---|---|
| 01  | REAP 当前流水线的真实接口（代码项） |
| 02  | Agent 可插入的 5 个位置 + 取舍矩阵 |
| 03  | 多样性/坍缩机制分析（为什么"只产一种答案"会发生与不发生） |
| 04  | 设计规则（结论文档：可执行条款） |
| 05  | 第一个实验（variant-driven TTRL MVP）与验收门 |

## 术语对齐

- "GPU 端生成多个 Lean 证明"—在 REAP 里 = policy 通过 gpu_runtime 批量 autoregressive 出 tactics，经 Lean 过滤，树内 progressive sample 再采样。
- "Agent 工作流"—OpenCode/类似的 agent loop：模型在长上下文中自我复盘，调 terminal/文件工具，输出可执行的复杂产物（如整段 tactic script / 变体题目 / 证明梗概）。
