# 05 — 课程学习 + MCTS + RL 全链（v1-1-agentic-tool 各模块的角色）

## 全链图

```
[04] 2k 代数题语料（验证门后）
        │
        ▼
[课程] course_driver / target_curriculum（难度调度 + 预算）
        │  每题：初始状态 + budget（steps/samples/tokens）
        ▼
[MCTS] Reap TreeSearch（CPU Lean）←→ GPU policy/value（HTTP）
        │  value = -d̂（head）；候选 = GPU policy（RULE-0）
        │  φ 证据环（v1-1）：agent 证据 → prior 重加权（09-C，待接线）
        ▼
[TTT] 题内联合更新（现有）：LoRA+head，learn/v1 端点
        │  成功/失败轨迹 → verified replay
        ▼
[RL] 跨题策略优化（**计划扩展**）：GRPO/REINFORCE 式
        │  奖励 = verified proof + 工具调用质量 + 预算效率
        │  约束：KL(frozen base) 防遗忘（现有 mixed_objective 已有 KL 项）
        ▼
[发布] snapshot / release / retire（现有回执链）
```

## 模块对应（不重复实现）

| 环节 | 现有代码 | 状态 |
|---|---|---|
| 推理服务 | `gpu/gpu_runtime/server.py`（real-search-categorical / qwen35-search） | ✅ 可跑（REAL 验证过） |
| 树/MCTS | `cpulean/Reap/TreeSearch/{MCTS,BestFirst}` | ✅ 编译（4.28） |
| 证据环 φ | `cpulean/Reap/Agentic.lean` + driver | ✅ 数据层；selection 接线待做（A 线） |
| 题内 TTT | `learn/v1` + `learner.py`/`mixed_learner.py` | ✅ |
| 课程 | `course_driver.py` / `target_curriculum.py` | ✅（现有） |
| 变体生成 | `target_variants.py`（R1）/ agent 生成器（R2） | R1 ✅；R2 待做 |
| **跨题 RL** | —— | ❌ **需要新写**（本计划最大工程量） |
| 发布/回执 | snapshot/release/retire | ✅ |

## RL 设计草案（最小可用）
- 目标：$\max_\theta \mathbb{E}[r] - \beta\, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{frozen}})$
  （与现有 mixed_objective 的 KL 项天然兼容）；
- 奖励：`r = 1{proof verified} + λ₁·tool_call_quality − λ₂·steps/budget`；
- 数据：课程 MCTS 的树 → 采样（状态,动作,回报）→ 组内归一化（GRPO 风格）；
- 起步规模：先用 100-200 题课程跑通 1-2 个 RL 轮，再上 2k。

## 与 v1-1-agentic 的关系
- agent 工具调用是**环境的一部分**（工具=搜索/编译/读文件），其调用质量进奖励；
- φ 证据环用于先验增强（A 线），不影响 RL 主循环的正确性。
