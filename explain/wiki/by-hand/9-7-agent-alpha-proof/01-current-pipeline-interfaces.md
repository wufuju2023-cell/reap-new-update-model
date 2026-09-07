# 01。REAP 当前流水线——代码层接口（用于对照）

> 依据：`reap-new-update-model` master 分支 `discussion/new_value_head_in7b_ex1` 与随附 code 包。
> 只列与"Agent 接入"有关的接口，不重复设计文档。

## 1. 生成侧（GPU）

- `gpu_runtime/real_backend.py`：REAL-Prover 7B + LoRA + 64-bin head 的推理后端；
  对外能力 = **policy tactic 批量生成** + **value 前向（categorical 期望距离）**。
- 服务形态：HTTP 端点（policy token-logprobs / value route），进程内批量吞吐。

## 2. 搜索侧（CPU + Lean patch）

- `cpu_runtime/verified_trajectory.py` 等：proof tree、AND/OR、backup。
- `Lean 过滤`：非法 tactic 丢弃；等价后继合并；`progressive sampling`（每节点 n(s)≤C·N(s)^α 再采样 K）。
- `online_ttt.py`：题内联合 TTT（LoRA + categorical head 联合更新），version v→v1→v2→… 必须被后续 generation 与 value 前向消费。

## 3. 课程侧

- `course_driver.py`：课程声明（CourseSquareTelescope 等）、checkpoint 推进、success finalization。
- `success_finalization.py` / `success_learn_recovery.py`：发布与退役链（seal→publish→retire）。

## 4. 训练侧

- `categorical_search_backend.py` / train_value_head.py 等：205,628 状态 → 64-bin head；
  学习目标 = policy NLL + KL(frozen base) + categorical CE；kernel = verified replay 轨迹（critical-path distance d）。

## 5. 关键结论（接口问题）

树内策略主采样发生在 GPU 后端；agent 工作流（terminal 工具、长上下文推理）在吞吐和同步性上**取代不了**该位置。
可行接入点围绕"课程/变体/审计"，详见 02。
