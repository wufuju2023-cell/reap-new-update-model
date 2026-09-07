# 00。为什么大模型输出概率分布而不是单个答案

## 1. 模型的真实输出：logits，而不是概率

设词表 $V$，在上下文 $x$ 下，transformer 最后产出隐藏向量 $h_T \in \mathbb{R}^d$，线性头 $W \in \mathbb{R}^{V \times d}$ 给出：

$$
z_v(x) = W_v \cdot h_T
$$

模型**并不直接产生概率**——它产生一组**未归一化得分**（logits）。概率只是对 logits 的一个规范性变换：

$$
p(v \mid x) = \frac{\exp(z_v)}{\sum_{u \in V} \exp(z_u)} = \mathrm{softmax}(z)_v
$$

## 2. 训练目标把 logits"拉成"分布

语言模型用负对数似然（交叉熵）训练：

$$
\mathcal{L}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}} \sum_{t} \log p_\theta(y_t \mid x, y_{<t})
$$

最小化该损失的最优解满足（等号仅在逐点一致时）：

$$
p_\theta(y_t \mid x, y_{<t}) = \tilde{p}(y_t \mid x, y_{<t})
$$

其中 $\tilde{p}$ 是训练分布中"上下文 $x,y_{<t}$ 下下一个 token 的真实条件分布"。关键事实：**训练数据中同一个上下文可能接不同合法续写**（同义词、不同分支、作者习惯），所以最优解是多点支撑的分布——这是"概率分布"的根源，不是模型被"灌进"了概率，而是数据本身的歧义。

## 3. 多分布中的每一种都合法

写作 $\tilde p = \sum_k \pi_k \, \delta_{y^{(k)}}$（混合点质量）时，softmax 学到的逼近未必是点质量，而是连续的：

$$
p_\theta(v) \approx \frac{\exp(z_v)}{\sum_u \exp(z_u)},\quad z_v \propto \log \tilde p(v) + \text{偏差项}
$$

一层近似下：最优 logits 近似于 $\log \tilde p(v) + C$（常数不改变概率）。这正是"softmax 是 log-probability 的归一化指数形式"的体现。

## 4. 采样为什么需要分布：期望与多样性

定义期望奖励 $\mathbb{E}[r \mid x] = \sum_v p(v \mid x)\, r(v)$。若只知道当前最佳 token（argmax），则：

- 贪心采样 $\arg\max_z$ 是"点质量"决策：无探索、零覆盖；
- 从分布 $p$ 采样则赋予每个候选其合法概率的质量，长序列下累计歧义会膨胀（自回归歧义放大），产生**多样序列集**。

## 5. 数学小结

| 概念 | 形式 |
|---|---|
| 输出单元 | logits $z \in \mathbb{R}^{V}$ |
| 概率化 | $p = \mathrm{softmax}(z)$ |
| 概率来历 | 交叉熵训练下 $p_\theta \to \tilde p$（数据的真实条件分布） |
| 不确定度的熵 | $\mathcal H = -\sum_v p(v) \log p(v)$（分布越平，熵越大） |
| 多样性的上限 | 支撑集大小与熵共同决定 |

因此"输出分布"是**训练目标 + 数据本质**的必然结果；而"如何用它"（温度、裁剪、采样）是推理时的工程自由层。
