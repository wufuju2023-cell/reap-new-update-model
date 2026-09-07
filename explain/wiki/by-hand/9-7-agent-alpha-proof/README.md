# Agent Coding × AlphaProof 式系统的接入（9-7 系列说明）

日期：2026-09-07。

## 提出者的问题（原话归纳）

1. 如何把 OpenCode 类 Agent 工作流（终端工具调用 GPU 端生成多种 Lean 证明）跟 AlphaProof（价值头/MCTS/TTT）结合？
2. 担心：结合后 LLM 会被训练成"只产生一种答案"，无法产生多种节点路径。

## 30 秒结论

- 正确的接法是把 Agent 放**树外**：做 TTRL 的**变体生成器**（位置 A）+ **matchmaker/挑战调度**（B）+ **证明审计**（D）。
- **不能**把 Agent 放**树内**当每节点出招器（C）：它顶不掉 GPU 批量采样，还会把 K 值坍缩成 1。
- 坍塌风险不在"Agent 是 LLM"，而在**回放数据分布**：单链回放=学到单点；多路线（Agent 天生能造）+ 90/10 混批 + 树内 progressive sampling = 分布多样，不会坍。
- 详见 `00-05` 系列文件。

## 系列文件

| 文件 | 内容 |
|---|---|
| `00-problem-and-question.md` | 问题表述与 30 秒结论 |
| `01-current-pipeline-interfaces.md` | REAP 真实接口（GPU policy/搜索/课程/训练） |
| `02-integration-points-matrix.md` | 五个插入点 + 取舍矩阵（A/B/C/D/E） |
| `03-collapse-mechanism-analysis.md` | 坍缩机制：取决于回放分布而非 LLM 本身 |
| `04-design-rules-and-verdict.md` | 6 条设计规则（RULE-0..6）与裁定 |
| `05-first-experiment-and-gates.md` | 第一个实验：Agent-driven variant TTRL 脉冲 + 交付门 |
| `06-agent-lean-repo-loop.md` | Agent 读 repo/lean 搜索/lake build/smoke → 真实 Lean 代码的标准流水线 |
| `07-does-it-produce-different-nodes.md` | "会不会产生不同节点"：来源、反坍缩条件与判定 |
| `08-engineering-constraints.md` | lake build 慢的现实、缓存/增量策略、接口清单与验收门 |
| `09-bayesian-view.md` | 贝叶斯视图：A 训练回放=先验迁移 / B 推理注入=后验加权 / C selection 覆盖 |
