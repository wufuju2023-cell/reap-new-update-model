# 02。Agent 可插入的位置：五选一矩阵

| # | 接口位置 | Agent 做啥 | 吞吐要求 | 与树的关系 | 推荐度 |
|---|---|---|---|---|---|
| A | **variant/TTRL 生成器**（课程侧） | 同一目标演化出简化/泛化/lemma/类比/分解变体题；Lean 语法校验+去重后当课程种子 | 低（每目标少量） | 树外 | ★★★★★ |
| B | matchmaker / 挑战调度 | 维护"失败前沿"，自动给失败题加预算、切换 prove/disprove 目标 | 低-中 | 树外 | ★★★★ |
| C | 树内 policy 建议者 | 每节点"建议 1 条高价值 tactic" | **极高**（每节点 K 级别采样处无法同步） | 树内 | ✗ 不适合主采样；仅可在 select 阶段外挂"advisor" |
| D | 证明翻译/审查 | auto-formalization（NL→Lean）；success 前独立审计（对 success finalization 做第二套人为可读检查） | 很低 | 树外 | ★★★★ |
| E | 运行监控与事故归因 | 慢节点/延迟误判/refresh 接线事故，生成处置建议 | 不限 | 系统 | ★★★ |

## 选 A 的理由（关键）

1. 论文 TTRL 要求"target + 数十万级相关变体"；Agent 的长上下文能力**天然适合造语义不同的变体**（不同于程序化局部变换）；
2. 变体不与 tree 采样争吞吐，Agent 慢一点无所谓；
3. 变体经 Lean 验证后进入课程池 → 与 REAP 现有 course_driver 接口完全兼容（现有课程正是从 course 声明出发的）。

## 为什么不选 C（"让 Agent 每节点出招"）

- GPU 端每秒可产出数百 candidates；Agent 单次生成为秒级且有上下文漂移，无法作为 progressive sampling 的替代；
- 若强行把它当唯一 policy 源：树的 K 值collapse 到 1 → 树变"搜索线"——这正是用户担心的"只产生一种答案"的工程版。
