# 04。设计规则与裁定（可执行条款）

## RULE-0（树内外边界）
树内 sampling 只由 GPU 批量 autoregressive 承担（K 每节点保持论文量级）。Agent 永不进入"每节点出招"路径。

## RULE-1（Agent 产物全过验证门）
Agent 任何输出（tactic script、变体题、证明梗概、auto-formalized statement）一律：Lean syntax validation → 去重 → 只保留 verified 的进池。零例外。

## RULE-2（主接点 = variant 生成器）
- 对 target 题：Agent + 提示模板生成（简化/泛化/分解/lemma/类比/局部变换）→ 验证 → 课程池；
- 版本化：varseed 记录 source agent run 与 prompt 指纹（与论文 L3 "以相似优质变体为种子递归演化"一致）；
- 目标是"数十万级"量级过程，先做稳健的几百条 pipeline。

## RULE-3（回放多样性不变量）
训练 replay = 全体验证轨迹（同一题多路线，作为 N 条同权记录）+ Mathlib 混批（维持 90/10）；任何 release 必须附 trajectory-family 统计表（路线数/题）。

## RULE-4（坍塌监控）
三个早期信号（见 03 末节）加入运行门；任一触发 → 停止该 course 的 learner 加速，改为扩 variant 预算（不盲目加 steps）。

## RULE-5（Agent≠policy 的前提）
除非实验目标本身是"以 agent 为单样本 policy 与 GPU policy 对比"，否则不得把 Agent 输出当 policy loss 唯一来源；对比实验也要有独立双臂（agent-policy vs gpu-policy）而不是混编。

## RULE-6（与 9-6 计划的关系）
本系列不改变 9-6 顺序；它将 TTRL 阶段的"variant curriculum"生成器升级为"Agent variant generator"，并**先于或并行于** large-variant-curriculum 阶段（9-6 计划步骤 E）。要做的第一个实验见 05。

## 裁定总结
- 主回答：能结合，接口 = 课程池/变体生成器 + 审计 + 挑战调度（位置 A/B/D）；
- 对"只产生一种答案"的担忧：在推荐接法下不会发生；若按"C 位置（树内出招）+ 单链回放"接法，会真发生——两个落点要明确分开，这是设计规则的核心。
