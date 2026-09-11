# 02 — 模型选择与工具调用提升路线

## 两模型现状

| 维度 | REAL-Prover 7B | Qwen3.8-9B-Distill |
|---|---|---|
| 来源 | `FrenzyMath/REAL-Prover`（公开） | `empero-ai/Qwen3.8-9B-Distill`（公开） |
| 基座 | Qwen2.5-Math-7B | Qwen/Qwen3.5-9B |
| 规模 | 7B（4 分片 15G） | 9B（19G） |
| hidden | 3584 | **4096** |
| value head | **有**：full-v3 64-bin（`WufuJu/v1-1-fullv3-artifact`） | **无**（需新训，见 03） |
| 工具调用训练 | 预期无（需探测确认） | 可能保留（需探测确认） |
| 证明能力 | 论文级 Lean 证明器（强） | 通用蒸馏（证明能力未测） |
| 已有工程支持 | `real_backend.py` 全链 | 云端 runtime 有 `qwen35_backend.py`（未跑过训练） |

## 方案 1：SFT/RL 提升 REAL-Prover 的工具调用
- SFT：构造"状态→工具调用 JSON"示范集（可从 opencode/agent 轨迹蒸馏；100-1k 条即可起步）；
  训练：LoRA（rank 16 与现有 pipeline 一致）+ 保留原证明能力（混入证明数据防遗忘）。
- RL：工具调用正确性奖励（格式/参数/成功链）用 REINFORCE/GRPO 微调；风险=破坏证明先验。
- 成本：中-高（数据构造+防遗忘+多轮评估）；收益不定（证明模型未必适合工具环）。

## 方案 2：直接换 Qwen3.8-9B
- 优点：若工具调用 OK，一步到位；value head 可新训（与 V1 同方法论）。
- 风险：证明能力未知（需先做"无工具 Lean 生成"小测；若证明太弱，工具环仍可救——
  搜索外包给工具+树，policy 只需可靠产出候选）。
- 检查点：
  1. 工具调用探测（01）≥70；
  2. 证明小测：3 题 smoke 风格的 tactic 生成可用率；
  3. value head 训练（03）。

## 判定顺序（不绕路）
1. 先跑 01 探测（两模型，同一题库）；
2. REAL 结果 ≥70 → 直接进 04/05（已有 value head，省一轮）；
3. REAL <70 且 Qwen ≥70 → 走 Qwen（+03 训 head）；
4. 两者都 <70 → 方案 1 的 SFT 轻调（优先 REAL，因其证明底子强）。
