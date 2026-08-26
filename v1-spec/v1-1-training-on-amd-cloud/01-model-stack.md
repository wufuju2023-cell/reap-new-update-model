# 01. 模型栈：reap-7B policy + value head

## 1.1 组件

```
policy π_θ  : Qwen2.5-Math-7B 基座 + REAL-Prover 权重初始化
             ├─ LoRA 适配器（trainable; 用 BF16 全量 + LoRA rank16）
             └─ sampling 走 transformers.generate；请求端 OpenAI 兼容由 lean 侧并发
                                                 └（v1 用 Python 驱动即可，lean 集成第二级）

value v_φ  : 共享 Backbone（frozen或LoRA共享） + 标量头:
V_φ(s) = W2 · act(W1 · h(s))  （取 last hidden state → 2 层 MLP → 1 标量）

teacher T  : Radeon Token Factory API（OpenAI 兼容，BaseURL /radeon/api/v1，每日 1pt/20RPM）
             ↓ 生成：变体(变常量/弱化/推广)、提示、难度评分 → 数据 & 课程
```

## 1.2 显存预算（单卡 48GB × 4，全 BF16）

| 部件 | 显存(单卡分摊) | 说明 |
|---|---|---|
| 7B BF16 权重 | ~14.0 GB | 每卡 1 个 DP rank |
| LoRA 梯度+优化器(AdamW) | ~1.5 GB（rank=16, 全模块） | 4 卡 DDP |
| value head | 忽略 | tiny |
| 训练 ckpt/激活(seq 4096, bs 1) | ~6–10 GB | micro-batch=1×4 |
| **实际占用** | ≈26 GB/卡 | 余量充足，可提 seq/bs |

> 也可直接用 FSDP (shard) 进一步减；先 DDP-LoRA 最简单跑通。

## 1.3 学习信号：**用即学**（RTTT 为主，SFT 可选）

### (a) 策略 SFT（**可选/保险丝**，默认跳过）
$$\mathcal{L}_{\mathrm{pol}} = -\mathbb{E}_{(s,a)\sim D}\,\log \pi_\theta(a \mid s)$$
作用 = 将 7B 校准到 mathlib 格式；REAL-Prover 权重已对齐 → **V1-1 主线 epoch=0**。
仅在 P1 表现停滞（连续 10 题无提升、且错误多为格式/语法类）时启用 1 epoch 校准（P3 保险丝）。

### (b) value head 初始化（一次性微校准，10 分钟）
- 默认：**随机初始化**，头 1 s-batch 用二元信号校准（正例=验证成功，负例=验证失败）：
$$\hat y = \mathbb{1}[\mathrm{checkProof} = \mathrm{ok}],\quad \mathcal{L}_v = \mathrm{MSE}(V_\phi(s), \hat y)$$
- 之后全部为**在线 TD**（调用期间的 MCTS 事件）：
$$V_\phi(s_t) \leftarrow V_\phi(s_t) + \alpha_V\big(G_t - V_\phi(s_t)\big)$$

### (c) 上线流程（0 长训路径，5 步）

```
1. /opt/venv/bin/python app/policy_server.py --adapter none（直载 REAL-Prover）  ← 零训练
2. app/value_server 随机头 + 10 分钟校准（可选）
3. 教师/用户 → policy_server（调用即进入 RTTT 模式：buffer→ttt_step）
4. → 队列回读 rttt_metrics.jsonl（≥5 步成功更新 = P1 PASS）
5. 结束调用 → 快照 adapter + 回滚保护（KL 超限回滚）
```

## 1.4 服务与推理（TTT 用）

- 训练包 `app/` 内自带 **policy_server.py**：加载 LoRA 权重 + `generate`（n>1 时 repeats 可取）→ HTTP（localhost:8760，OpenAI 兼容子集）+ **logprobs**（TTT 需要旧概率 ρ）
- 价值 head 通过 `value_server.py`（同端口 /value 路由）返回 `{"score": float}`
- 二者与 Lean 侧（reapMCTS 的 endpoints 配置）对接时，策略地址填 localhost:8760（Lean 在实例内的话用同主机端口可行）
