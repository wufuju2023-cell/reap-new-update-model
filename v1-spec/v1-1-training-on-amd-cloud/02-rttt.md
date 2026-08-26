# 02. 持续推理时训练（RTTT）——P1 核心

> **语义（用户裁定版）**：学生模型的梯度更新**只在"被调用"期间**发生——
> 无论调用者是**人**（编辑器里跑 `reapMCTS`）还是**教师模型 API**（批量出题）。
> 调用开始 → RTTT 武装上线；调用结束 → 更新暂停（快照 + 回滚保护）。
> 不存在独立的"训练任务"；`policy/value server` 天生就是**用即学、结束即停**。

## 2.1 定义（本 spec 的实现语义）

搜索**不冻结参数**：某一道题的搜索过程中，
每次 Lean 验证反馈（tactic 成功/失败/错误）立即变成**一个小梯度步**：
- 策略：对"被验证失败或反复无效"的 (state, tactic) 推低；对"成功闭环"的推高；
- 价值：以 TD/蒙特卡洛回报校准 $V_\phi(s)$。

$$\theta \leftarrow \theta - \alpha \cdot \nabla_\theta\!\big[-\hat r \log \pi_\theta(a|s)\big],\quad
r = \begin{cases}+1 & \texttt{checkProof}\ \mathrm{ok}\\ -\eta & \mathrm{error}/\mathrm{parse}\\ 0 & \mathrm{else}\end{cases}$$
$$V_\phi(s) \leftarrow V_\phi(s) + \alpha_V\big[\hat r + \gamma V_\phi(s') - V_\phi(s)\big]$$

## 2.2 约束（工程现实 vs 理论）

| 约束 | 妥协 |
|---|---|
| 每节点梯度步（n=6 样本 × batch） | LoRA 上 1 步 ≈ 1.2s（4 卡聚集仅需 policy server 单卡算）→ 采用 **每 k=8 节点批量一步**，k 可配置 |
| 旧策略概率（重要性/PPO 需要） | 每个候选记录 `log p_old`，server 端 `logprobs` 返回 |
| KL 防忘 | 每步加 $\beta \mathrm{KL}[\pi_\theta \|\pi_{\mathrm{base}}]$, β=0.05；超界回滚该题 LoRA 快照 |
| 搜索与更新竞争 GPU | TTT 更新放**同卡（卡 0）**，值读取共享；其他卡留给训练/其它 |qI

## 2.3 RTTT 触发协议（从 Lean/搜索端视角）

```
搜索节点 s
  → policy_server /get_tactics (n=6, logprobs)
  → Lean 执行(并不可靠的省略) -> verdict (ok|error|parse|timeout|solved)
  → (s,a,r,logp_old) append 至 /workspace/out/ttt_buffer.jsonl
  → 当 |buffer| ≥ k: policy_server /ttt_step {buffer, base ref}
       → 单步更新（LoRA adapter hot-swap）→ 清 buffer
  → 每题结束：存该题 adapter 快照；若 “so much worse”(KL 超限) → 回滚
```

## 2.4 MCTS 集成节奏（v1 简化）

- v1 先**不接 Lean full MCTS**：用合成/缓存状态跑通"server 循环 + TTT 步"（状态取自 SFT 数据切片），验证 loss 曲线、回滚、hot-swap；
- v1b 再接 `reap`(Lean) 真校验环（§03《lean build》+ `reapMCTS` endpoints 指到 policy_server/value_server）。
