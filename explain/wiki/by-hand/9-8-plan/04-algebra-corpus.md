# 04 — 2k 代数题语料生成（课程学习原料）

## 目标
生成约 **2000 道抽象代数方向 Lean 题**（statement 级），经验证门入课程池，
供"课程学习 + MCTS + RL"使用。

## 两条路线

### R1：受限确定性变换（现有代码，先跑通）
- 工具：`cpu_runtime/target_variants.py`（`nat_forall_instance` / `and_left/right` 等）
  + 手工种子（抽象代数族：群/环/域/理想/同态）；
- 规模：种子 50-100 题 × 变换 → 数百题；**达不到 2k 的量级但能立刻验证全链**；
- 验证：每个变体过 `prepare_attempt.py` 的编译预检（closed_problem preflight）。

### R2：Agent 变体生成器（计划正路，v1-1 的 B 线）
- 形态：opencode agent + 提示模板（简化/泛化/分解/引理/类比/局部变换）×隔离 session；
- 产物：Lean statement 草稿 → **验证门**（语法→编译→去重）→ 课程池；
- 规模：50 种子 × 40 变体 ≈ 2k；
- 防塌要求（教训 #4）：类平衡（题型/难度分布），不做无上限翻倍；
- 状态：生成器**尚未实现**（9-7 系列 05/08 已设计），是 2k 题的关键依赖。

## 验证门（两路共用）
```
Lean 语法检查（lean 4.28 环境）
  → 编译预检（无 sorry / 目标可陈述）
  → 去重（statement 规范化哈希）
  → 入池（variant_pool/{problem_id}/meta.json）
```
- 门：通过率 >25%（R2），>90%（R1 变换）；类平衡偏差 <30%。

## 与课程学习衔接
- 课程池 → `course_driver.py`（现有课程驱动）→ budgets（`proof-curriculum/budgets.json`）；
- 难度分层：按（是否直接 simp/omega 可解、库依赖深度）三档，课程由易到难；
- 产出：成功树（MCTS+TTT 轨迹）→ 回放池（value head 增量 + policy RL 数据）。
