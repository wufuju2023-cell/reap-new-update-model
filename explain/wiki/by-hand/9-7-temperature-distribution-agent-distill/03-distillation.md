# 03。蒸馏中的温度：Dark Knowledge 与校准

## 1. 知识蒸馏（Hinton et al., 2015）的形式化

教师模型给出软标签 $p_T^M$，学生最小化：

$$
\mathcal{L}_{\mathrm{KD}} = \alpha \cdot \mathrm{KL}\big(p_T^{\mathrm{teacher}} \,\|\, p_T^{\mathrm{student}}\big) + (1-\alpha) \cdot \mathrm{CE}(y_{\mathrm{hard}}, p_1^{\mathrm{student}})
$$

其中所有概率都用**温度升高后**的 softmax 计算（推理时学生回到 $T=1$）。

## 2. 为什么"温度"起作用（数学本质）

概率比（类别 $i$ vs $j$）在训练分布中通常由"类别间距离/语义重叠"决定：

$$
\log \frac{p_T(i)}{p_T(j)} = \frac{z_i - z_j}{T}
$$

- $T=1$：学生只看到"最强类别"几乎是 1（softmax 指数锐化），跨类别信息被压掉；
- $T$ 升高：把 $z_i - z_j$ 的微小差异放大到可测量区（等式右侧变小但样本量内 remain 非零），学生学到**负类之间的结构**（dark knowledge：类别间的相互相似性）。

数学上：设教师离散分布 $q$，soft target 的梯度：

$$
\nabla_{z_s} \mathrm{KL}(q_T \| p_T^S) \propto q_T - p_T^S
$$

关键：$q_T$ 的支撑集宽、梯度不触"零概率"项→**信息传递不因小概率而消失**（这就是 soft label 蒸馏优于 hard label 的原因）。

## 3. 相变点：温度过大/过小

| $T$ | 传递信息 | 风险 |
|---|---|---|
| $T \approx 1$ | 几乎只在正类上给梯度 | 学生只学"正例近邻"结构 |
| $T \approx 3$~$5$（经验） | 类别间结构充分暴露 | 太大→噪声占比升高 |
| $T \to \infty$ | 分布趋平，KL 项退化为"互相接近均匀" | 信息稀释 |
| 组合符号 | 无 | 用 $T$ 与权重 $\alpha$ 联合调优 |

## 4. 延伸到 RL/LLM 的两种"温度艺术"

### 4.1 KL-正则的 RL（RLHF 同构）

目标：

$$
\max_\theta \; \mathbb{E}_{y \sim \pi_\theta} \, r(y) \;-\; \beta \cdot \mathrm{KL}\big(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\big)
$$

这是蒸馏的"反向形式"：$\pi_{\mathrm{ref}}$ 扮演教师，$\beta$ 相当于**温度倒数**。调 $\beta$（即调"约束强度"）与调 $T$ 在数学上给出同一族分布（Log-sum-exp 对偶）。

### 4.2 温度缩放校准（Guo et al., 2017）

目标是找 $T^*$ 使 softmax 校准（ECE 最小化）：

$$
\hat p_i = \exp(z_i / T) / \sum_j \exp(z_j / T), \quad T^* = \arg\min_{T>0} \mathrm{ECE}(\hat p)
$$

这与蒸馏的"用温度做平滑"是同一数学工具的第二种用途：**校准**（不是提取知识，而是恢复置信度）。

## 5. 对蒸馏算法族的影响（我们可用）

| 方法 | 用处 | 温度设置 |
|---|---|---|
| 普通 KD | 7B teacher → 小模型学生 | $T\approx3$~$5$，KL 权重 $\alpha$ 0.7-0.9 |
| 序列蒸馏（data distillation / SeqKD） | 教师采样 N 条、学生学软标签（train on teacher-generated） | 采样 $T>1$（多样），训练 $T=1$（校准） |
| RLAIF（奖励模型蒸馏） | 语义奖励信号到轻量 policy | 多温度多评（采样多样） |
| MiniLLM（反向 KL） | $\mathrm{KL}(\pi_\theta \| \pi_{\mathrm{teacher}})$ 防小模型过拟合 | 同上需温度 |

## 6. 数学小结

- 蒸馏的价值 = 将"类别间结构"从教师转移到学生；
- 温度是转移量的旋钮：太低→只传 hard；太高→传噪声；
- RL 的 KL 正则与温度是**同一族的同构**（KL 约束系数 = 高温/低温的等价表述）；
- 温度的第二生命 = **校准**（restoring confidence），是"不确定性评估/熵门控"的硬理论基础。
