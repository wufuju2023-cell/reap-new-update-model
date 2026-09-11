# 9-8-plan — 工具调用能力优先的 MCTS+RL 训练路线

> 日期：2026-09-08。前置事实：GPU 实例已关闭；Qwen3.8-9B 官方源 = `empero-ai/Qwen3.8-9B-Distill`
> （HF 公开，base=Qwen/Qwen3.5-9B，hidden=4096）；**它没有 value head**；
> `WufuJu/reap-value-head-v1-scaling` 是 REAL-Prover(3584) 的 head 实验，与 Qwen 无关。

## 决策树（本计划的核心）

```
                     ┌─ REAL-Prover 7B 工具调用能力？
第一步：探测两模型 ──┤
                     └─ Qwen3.8-9B-Distill 工具调用能力？

REAL-Prover:
  ├─ OK ──────────────► 直接用（加 value head 已有 full-v3）
  └─ 不行 ──┬─ 方案1：SFT/RL 提升工具调用（成本高，收益不定）
            └─ 方案2：换 Qwen3.8-9B（先测工具调用）
Qwen3.8-9B:
  ├─ 工具调用 OK
  │    ├─ 有 value head？→ 没有 → 新训（4096→256→64）
  │    └─（value head 训练完成后）
  │         └─► 2k 代数题 → 课程学习 + MCTS + RL（v1-1-agentic-tool 框架）
  └─ 工具调用不行 ──► SFT/RL 工具调用能力（同方案1）
```

## 已明确放弃
- ❌ value-head「有用性」消融实验（不再做 heldout calibration / value-on/off 对比）
- ❌ V2（忽略）
- ❌ 未在代码实现的 docs/spec（不作为执行依据）

## 系列导航
| 文件 | 内容 |
|---|---|
| `01-tool-call-probe.md` | 工具调用能力探测：协议、题库、评分、门、执行方案（等实例） |
| `02-model-options.md` | REAL-Prover vs Qwen3.8-9B 现状、SFT/RL 提升路线对比 |
| `03-qwen-value-head.md` | 为 Qwen 新训 value head 完整方案（数据/目标/预算/与框架整合） |
| `04-algebra-corpus.md` | 2k 代数题生成：R1 受限变体 / R2 agent 生成器 + 验证门 |
| `05-training-loop.md` | 课程学习 + MCTS + RL 全链（v1-1-agentic-tool 各模块的角色） |
| `06-gates-timeline.md` | 门、时间线、资源清单、回退策略 |
