# 2. value、MCTS 与题内 TTT

## 2.1 搜索中的作用

policy 先生成 tactic 候选；Lean 检查候选，只把合法、非重复后继加入树。full-v3 对这些后继给出正距离 `d`，搜索保存原始距离并以 `V=-d` 做 selection 与 backup。越接近证明的状态具有更大的搜索效用。

每次运行必须同时审计：候选文本/顺序/logprob、Lean disposition、合法非重复后继、selection 与 revisit、backup 拓扑、所用模型版本和最终 proof。只看到“full 的 d 数字比较小”不能称为 value 有帮助；必须看到它改变了分支预算或关键回访，并最终带来更好的合法推进或证明。

## 2.2 本路线的联合在线更新

REAL7 full-v3 与外部 S18 critic 路线不同。这里的 AdamW 同时包含：

- policy LoRA 参数；
- categorical value head 参数；
- 对应 optimizer state。

在线 value target 来自当前题真实 MCTS 的有限 `search_visit_backup`。target 裁剪到 `[1,64]`，小数距离分配给相邻两个 bin，使用 two-hot categorical cross entropy；policy 项和对冻结 base policy 的 KL 约束同时存在。Lean 预算耗尽绝不被编码为 `-∞`。

一次 update 只有同时满足下列条件才计为有效：

1. learn 请求和回执可绑定到同一 session/event；
2. policy version 严格递增；
3. LoRA、value head、optimizer 的预期张量确实变化，冻结 base 不变；
4. 更新后出现使用新版本的 generation 和 value 前向。

终端 proof 之后的 success-learn 单列：它训练成功轨迹，但没有帮助已经完成的本次证明。

## 2.3 本案例的 refresh 限制

本场固定 `selection_value_refresh=false`。原因是当时 Lean/ReapRuntime 没有产生 learn 后的旧节点 refresh 事件；强行开启会在首个更新窗口后发生工程终止。

因此本案例可以严格证明：v1/v2 之后的新 generation 与新 value 前向消费了更新参数；不能声称树中所有旧节点缓存都被新 value 重算，也不能把旧节点后续回访全部归因于更新后的 head。该限制不影响本次数学成功和独立 Lean 验收，但属于下一版 runtime 应修复的接线能力。

## 2.4 三类证据不要混淆

- 机制证据：value 真进入 selection/backup，参数真更新，新版本真被消费。
- 单题能力证据：模型在本课程得到完整 Lean proof。
- 因果比较证据：同题、同预算、同 policy/RNG 的 trained 与 exact-random matched pair 出现可解释差异。

本包前两项完整；第三项由 F1 matched 案例和后续矩阵提供，而不是从本课程单臂结果推断。

真实实现见 `code/categorical_search_backend.py`、`code/online_ttt.py` 和 `code/course_driver.py`。
