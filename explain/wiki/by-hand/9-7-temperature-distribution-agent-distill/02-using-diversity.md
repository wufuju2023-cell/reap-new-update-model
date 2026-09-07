# 02。多样性如何被利用（与温度的关系）

## 1. 多样性的数学定义

设采样得到 $N$ 个序列 $\{y^{(1)}, \dots, y^{(N)}\}$，定义两两 Jaccard 或编辑距离 $d(\cdot, \cdot)$：

$$
\mathrm{div}(\mathcal{Y}) = \frac{1}{\binom{N}{2}} \sum_{i < j} d(y^{(i)}, y^{(j)})
$$

温度直接控制 token 级熵，通过**自回归放大**间接控制序列级多样性（熵越大，期望分叉越多）。

## 2. 利用它的五种经典范式

| 范式 | 机制 | 数学 | 温度角色 |
|---|---|---|---|
| Best-of-N | 采样 N 条→ 最高奖励 | $y^* = \arg\max_{y^{(i)}} r(y^{(i)})$ | $T$ 大→候选池独特；$T$ 小→池内质量高。离线最优是 $T$ 保持某种中等值 |
| 自一致性 | 多条采样→投票/多数聚合 | $y^* = \mathrm{mode}\{y^{(i)}\}$ | $T$ 升→得票结构更分散，用于"确认"而非"取一" |
| MCTS / 树搜索 | 每节点按先验采样 K 次 | PUCT: $Q + c \cdot \pi(a)^{1/\tau} \cdot \frac{\sqrt{\sum_b N_b}}{N_a+1}$ | 每节点采样数=预算；先验温度决定 K 分支的"忠实度" |
| 课程/数据增强 | 多样样本转训练集（replay） | $\mathcal{L}(\theta) = -\sum_t \log p_\theta(y_t \mid \cdot)$ 样本族 | 分布宽→数据谱广→先验宽（反坍缩） |
| 不确定性估计 | 熵/频率计 | $\mathcal{H} = -\sum_v p \log p$ | 温度=置信度缩放，用校准后熵做 OOD/门控 |

## 3. 树上：为什么"多样性"依赖温度（AlphaZero/AlphaProof 视角）

树内 sampling 理论（Van den Broeck 等, 或 AlphaZero 论文"search + sampling"）：

- 每节点树若无多样性 → 反复同一路径 → 树"线化"（这就是我们 9-7 系列 RULE-0 的搜索坍缩）；
- 多样性两途径：
  - 由**先验熵**（温度）：$\pi_T(a \mid s) = \mathrm{softmax}(z/T)$ 在节点处提供分叉率；
  - 由 **progressive sampling**：对高访问节点按 $n(s) \le C \cdot N(s)^\alpha$ 重新采样 $K$ 条——功能性解耦于温度，强化"即使先验单峰也重分叉"。

## 4. 探索–利用的精确表述

给定奖励函数 $r$，多步采样器在"高回报"与"新路径覆盖"间的权衡：

$$
\max_{G} \; \underbrace{\mathbb{E}[r \mid \text{policy 采样}]}_{\text{利用}} \; + \; \lambda \cdot \underbrace{\mathcal{H}(\text{采样分布})}_{\text{探索}}
$$

$\lambda$（探索系数）与温度 $T$ 等价地调制同一权衡：$\lambda$ 人设是"熵奖励"，$T$ 是"分布中的贝叶斯表观熵"。
在 RL 里更准确的对应是：KL-正则目标（$\pi_\theta$ vs $\pi_{\mathrm{ref}}$）与温度的联系（后面 03 蒸馏节展开）。

## 5. 我们系统要长期保持的三条（带数学）

1. **先验熵下限**：$\mathcal{H}(\pi_T(\cdot \mid s)) \ge H_{\mathrm{floor}}$ 监控（低于即判定过热→加采样温度或 progressive 加采样）；
2. **路线多样性上限**：replay 集内部 Jaccard 中位数 $\le 0.7$ 是对"路线多样性"的直接刻画（07 文件）；
3. **温度开关只在实验级**：全实验静默矩阵（$T \in \{0.8, 1.0, 1.2\}$ 与 progressive $\alpha/C$ 配比），不要在训练中途换。
