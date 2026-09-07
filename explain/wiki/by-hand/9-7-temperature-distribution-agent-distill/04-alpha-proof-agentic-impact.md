# 04。对 AlphaProof + agentic tool calls 的具体影响

> 先立边：我们的系统实际是"**GPU policy + Lean 树 + TTT + agent 库循环**"。温度在此有五个落点（T1-T5），逐一给数学理由与实验建议。

## 链路的温度落点图

$$
\underbrace{\pi_\theta}_{\text{T1: 树内采样}}
\;\to\;
\underbrace{\text{progressive resample}}_{\text{T2: 重采样热}}
\;\to\;
\underbrace{\text{agent loop}}_{\text{T3: 工具循环}}
\;\to\;
\underbrace{\text{replay/mixed learner}}_{\text{T4: 数据蒸馏}}
\;\to\;
\underbrace{\text{value head+calibration}}_{\text{T5: 校准温度}}
$$

## T1. 树内 policy 采样温度（探索-利用的轴）

- 现状：每节点 K 个候选来自 `real_backend.py` 的批量 autoregressive。当前 $T=1$（默认）。
- 数学：K 个独立同分布采样下，树"分叉熵"与 $\mathcal H(\pi_T)$ 同步；若 $T$ 过小且模型自信，树退化（先验单峰），必须靠 progressive sampling 人工注熵（见 9-7 RULE-0）。
- 建议：
  - 静默扫描 $T \in \{0.8, 1.0, 1.2\}$ × 每课程固定分支（其他全等设置），以 **verified solve@budget** 为主指标，配 entropy 报告；
  - 保持 $T=1$ 与"熵下限"的组合：$\mathcal H \ge H_{\mathrm{floor}}$（否则采样无差异）。

## T2. Progressive sampling 的相对角色

- progressive sampling 数学：$n(s) \le C \cdot N(s)^\alpha$ 后再采样 K。
- 它与温度**解耦**：温度控制"每条采样忠实于先验的程度"，progressive 控制"高访问节点重新抛硬币的频率"。
- 建议：调参分两轴——(a) 温度轴（先验忠实度），(b) $(C, \alpha)$ 轴（重抛频率）。不要让它们一个退化到"等同于另一个"——每次只动一个轴并记录。

## T3. Agent 工具循环（树外）多样性来源

- 上一系列结论：agent 单结果可接受（节点多样性由 GPU policy 保证）；agent 只负责**路线/证据多样性**。
- 但 agent 的"生成变体题"应保持**高温度采样**：每一个变体 = 一次分布采样，$T$ 高则语义分歧大；
  - 数学：变体集合的多样性 $\mathrm{div}(\mathcal{V}) \approx$ agent 采样的池熵；若 agent 只在 $T=1$ 生成，则跨会话差异≈prompt 指纹差异。
- 建议：agent loop 参数：temperature 1.0-1.3（按 9-7 系列的提示指纹 + 隔离 session），采样数次取不同类别（simplify/generalize/decompose/lemma/analogy/local-transform）。

## T4. Replay/mixed learner 的温度=数据蒸馏

- 现训练目标 = NLL + $\beta \cdot \mathrm{KL}(\pi_\theta \| \mathrm{frozen\ base})$ + $\lambda \cdot \mathrm{CE}$。
- 这已是**微分蒸馏**（frozen base 作 teacher 温度的隐式 1）。若需要更强的"路线平滑"：把 replay 信号用 teacher 分布（在 $T=3$~5 输出软标签）传给学生 → 等于标准 KD 的逐 token 版。
- 数学上，把 Replay 的 CE 换成 `soft-KD`：$\mathrm{KL}(p_T^{\mathrm{teacher}} \| p_T^{\mathrm{student}})$，$T$ 又回到"在哪取舍暗知识"的经典问题：$T$ 大→负类结构（variant/lemma 相似度结构）传递；$T$ 小→只学 yes/no。
- 建议：T4 实验级别：先不动目标；仅在 T4 评估"replay 数据里多路线"是否已足够（分布宽度由数据给,不靠温度）；若需推进 **distill (7B GPT->less)** 才启用 $T \approx 3$。

## T5. Value head 与校准温度

- value head = categorical 64-dist的 **softmax loss**。温度缩放可以在这个头上做校准：$T^*$ 使得 expected distance 的分布与"真实剩余步数"校准（ECE 最小）。
- 用途：搜索中 value-on/off 的比较应使用**校准后**的 crit 值；否则 T5 会影响"怎么算搜索增益"。
- 数学：$T^* = \arg\min_T \mathrm{ECE}\{ \hat p_T \}$，其中 $\hat p_T$ 是 value head 的 softmax 经温度缩放。如果 $T^* \neq 1$，说明 head 未校准（过度自信或胆小），这会影响 MCTS 的 backup 幅度。

## 对蒸馏（广义）的大局观

- 我们把"student 学 teacher"分成两层：**model-level distillation**（重量、可选项）与 **distribution-level distillation**（T4 的数据混合,已被发现/回放数据天然完成)；
- 若未来在 agent 上做"蒸馏一把轻量 policy"，则用 T1 老师（GPU policy）+ 高 $T$ 的离线 soft-label 采样，即"SeqKD + 温度"，是 03 中的 seq-distillation。

## 硬件与计算口径（重要）

- T1/T2 的改动 = GPU 端（成本极高：每 node × K × 温度扫描）；
- T3 的改动 = CPU/agent 端（成本中等）；
- T4 的改动 = 训练服务端（中等，需要做 versioned data）；
- T5 的改动 = 一次校准实验（便宜，建议最先做）。T5 先做：它会给 T1 一个"value 的有效校准基线"。

## 结论摘要

1. 概率分布来自 softmax + 训练目标（数据歧义）；温度 = 分布宽度旋钮；
2. 多样性利用 = 采样策略 + 熵 + 树结构 + 数据混合物（四个面）；
3. 蒸馏 = 用温度传递"类别结构"；同构于 RL 的 KL-正则；温度的另一支=校准；
4. 对 AlphaProof 系统：T1（扫描采样温）× T2（重抛频率）双轴，T5（校准）先做，T3 用高 T 造变体，T4 暂时只做数据宽度、不做 model-distill，直到 T4 前实验显示分布带宽不足。

## 下一步建议实验优先级

| 优先级 | 实验 | 成本 | 预期 |
|---|---|---|---|
| 1 | T5 价值头温度校准（T*找定，验证 ECE 下降） | 低 | value 比较更公平 |
| 2 | T1 最小扫描（0.8/1.0/1.2 × 单一课程 × 固定预算） | 中 | 先验忠实度对 solve@budget 的影响 |
| 3 | T3 变体生成加高 T（多提示指纹 × session 隔离） | 中 | agent 路线多样性 >0 |
| 4 | T4 mixed learner 做 soft-KD（T=3）对照 | 高 | 路线知识迁移 |
