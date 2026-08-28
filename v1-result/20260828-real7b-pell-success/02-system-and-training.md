# 当前7B、搜索训练与学生证明库

## 模型和环境

学生是`FrenzyMath/REAL-Prover`，revision `fe76f68d9a88f342cb7b546307c20292fea9cced`。本次不是9B对照或chat-thinking直出实验；使用模型原生Reap逐步tactic提示。

实际GPU为gfx1100、每卡48GiB，主机有两卡但最后Pell搜索只用一张；不需要两卡张量并行。Torch `2.10.0+rocm7.2.4.git3d3aa833`、Transformers4.57.1、PEFT0.17.1。CPU负责Lean4.28.0-rc1、Mathlib5352afc…和Reap搜索；GPU负责策略/价值与训练。完整身份见[environment.json](environment.json)。

## 一道题内怎样学习

1. Lean给出当前证明状态与固定选定的前提。
2. 7B生成候选tactic，并给出真实token概率；价值头评价状态。
3. Lean执行候选，MCTS维护合法子节点、访问和回传。
4. 每8个检查点至多触发一次搜索反馈训练；GPU回执确认新version后，同一个CPU搜索继续使用新参数。

基座冻结，LoRA rank16、alpha32、dropout0，覆盖注意力与MLP投影；隐藏维3584。价值头为Linear–SiLU–Linear–sigmoid。学习率policy 1e-4、value 3e-4，梯度裁剪1，KL系数.02，gamma .99，value floor1e-6。

题内policy学习搜索访问权重，value保持原折扣回报含义。成功后另一明确阶段执行：完整Lean独立检查→逐状态重放→同session对全部真实成功动作一次CE＋KL及折扣value更新→seal→publish→retire。实现见[search_objective.py](code/gpu_runtime/search_objective.py)、[success_finalize_objective.py](code/gpu_runtime/success_finalize_objective.py)与[成功协调器](code/experiments/proof-curriculum/runner/success_finalization.py)。不把成功动作混成未验证自训练。

## 跨课程继承

新课复制成功release的adapter/value；新建optimizer、RNG、buffer、搜索树及version0。原题create的实际来源是exp-26c724af16ee-01。同session多次更新与跨session重新计数分别记录，不以“新课v0”误判没有继承。发布参数不保证下一课立刻掌握新动作。

最终原题在第一次题内更新前就解出，**题内0次更新**，之后有1次成功更新。课程链之前确有在线更新，但不能声称原题本身发生“更新后继续搜索”。

## 最后三次搜索配置

32steps、2048nodes、8候选×256token、temperature .99、max4updates/min8、exhausted-policy stop。Lean源码整数选项`reap.c_init 1500`缩放为1.5；`reap.progressive_sampling_c 500`缩放为.5；c_base3200保持不变。实际末checkpoint c=1.5012492194004319，见[探索实证](evidence/original-target-exploration.json)。

温度影响生成什么；PUCT系数影响选哪条已有分支；progressive sampling影响何时重新生成候选，三者不能混为“增加随机性”。实际随机种子由session身份派生，不把日期标签当作Torch seed，更不保证跨机器字节级相同搜索。

## 学生库：已实现与未来计划

当前[严格来源校验器](code/cpu_runtime/student_proof_library.py)绑定完整学生proof、accepted回执、源码SHA、声明、实际Lean类型及其SHA。CPU15[闭合目标包装](code/cpu_runtime/closed_problem.py)同时把选中的完整定理放入前提通道和局部上下文；不实例化本题参数，不给见证或归纳脚手架。

每次最多明确选择4条来源合格定理，库版本固定，不热改活跃run。policy、value、observer与重放/训练使用同一前提；loader行不算模型动作。完整证明依赖重新编译，拒绝sorry、额外axiom、教师proof及直接等同完整目标的提示。

未来可让OpenCode在固定Mathlib索引＋学生证明库中检索。需记录每次查询、返回、选择、上下文hash并在训练/重放中还原；设置top-k/token/超时和缓存。**本轮只有固定选集与局部加载，尚未实现动态检索，不能把本结果宣传为检索工具有效性实验。**
