# 05。第一个实验：Agent-driven variant 的 TTRL 脉冲（MVP）

## 目标
验证"Agent 造变体 → Lean 验证 → 课程池 → TTRL-style specialist"最小可运行闭环；同时用坍塌监控确认：接入后 round 轨迹路线数≥3（不放单点）。

## 范围（不做）
- 不做大 variant 规模（几百条即可）；
- 不改 GPU policy 采样路径（RULE-0）；
- 不在本实验里训练 generalist。

## 步骤
1. **agent-variant 生成器**（树外）：
   - 输入：目标题（Lean statement）+ 元信息（难度、失败前沿）。
   - Agent 提示模板：给出 5-6 类变体（简化/泛化/分解/引理前提/类比/局部变换），每类产出 Lean 草稿。
   - 输出：`variants_raw.jsonl`（含 prompt 指纹、agent run id）。
2. **Lean 验证门**：syntax validation + dedup；进 `variant_pool`（版本化：varseed）。
3. **课程池接入**：把 variant 当作 course 新种子，接 `course_driver.py` 现有入口（允许在课程声明里加 variant 题）。
4. **树内策略**：用现有 full-v3 / 评估候选，正常跑 MCTS。
5. **监控**：proof trajectory-family 统计（路线数/题、Jaccard 中位、action-entropy）+ 未验证小步验证。
6. **对照**（同时具备）：
   - 对照组：无 Agent variant（同预算、同课程）。
   - v-对照：满足 RULE-6 才有 ttrl 结论；本实验只出"闭环可运行 + 无坍缩"证据。

## 交付门
- [ ] 全链闭环：Agent 变体 → Lean 验证 → 进课程 → MCTS/TTT 运行 → 至少 1 个成功 proof + 独立 Lean；
- [ ] 路线多样性：任一题 ≥2 条不同路线回放（>=3 为佳）且坍塌监控触发数=0；
- [ ] 无 RUN-0 违规：GPU 采样路径无改动；
- [ ] 每 release：trajectory-family 统计表写入 evidence。

## 与既有方案的关系
- 该实验位置 = 9-6 计划的任务 E（TTRL）之前的"variant 生成器换代"；
- 完成后可以决定变体规模：几百 → 上千 → 数万（配合 compute 与增益门）。

## 已知风险
- Agent 生成变体的 Lean 语法失败率高 → 需要"修复循环"（agent 读取 Lean 错误再修），该循环也计入验证门；
- 线路过快增长 → 用 dedupe/diversity 门槛控制课程池大小；
- 任何一步没有把"多路线"固化到数据 → 退回 04 RULE-3 检查。
