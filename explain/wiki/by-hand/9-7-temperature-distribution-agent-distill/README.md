# 温度、概率分布与多样性——从数学原理到 AlphaProof+Agentic 应用

系列日期：2026-09-07。目录：`9-7-temperature-distribution-agent-distill`。

## 本系列回答

1. 为什么大模型输出**概率分布**而不是单个答案？
2. "温度"如何调控分布形状？（严格数学）
3. 这种多样性如何利用？
4. 温度在**蒸馏**中扮演什么角色？
5. 对 **AlphaProof + agentic tool calls** 的具体影响与可执行提案。

## 文件导航

| 文件 | 内容 |
|---|---|
| `00-why-distribution.md` | softmax 推导、训练目标、不确定度的来源 |
| `01-temperature-math.md` | Boltzmann 分布、温度极限、自回归联合分布、裁剪 |
| `02-using-diversity.md` | 多样性利用：采样策略、自一致性、MCTS、熵与探索 |
| `03-distillation.md` | 知识蒸馏中的温度：KL、dark knowledge、校准 |
| `04-alpha-proof-agentic-impact.md` | 落到我们系统：温度链、熵门、蒸馏/校准应用、实验提案 |

## 三处引用的共同机制（先浓缩）

- 模型本质上只输出**一组实数 logits**；概率 = 对 logits 施 softmax 的**规范性条件**（并无"生成概率的魔法"）；
- 温度是 softmax 中的一个**标度参数**，等价于玻尔兹曼分布的温度：缩放 logits 即可平滑地在"确定性"与"均匀"之间插值；
- 所有"多样性 = 优秀特质"的场景，其数学本质上都是 **熵/支撑集宽度**与**预测精度**的平衡：`sharpness–diversity` 权衡。
