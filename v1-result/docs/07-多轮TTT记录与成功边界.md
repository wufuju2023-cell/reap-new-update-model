# 历史 F：五次更新与未证结果

## 本页记录什么

**这是历史 F 的训练记录：同一道题更新五次，最终在32步搜索上限内未证明。** 每轮参数变化和后续新版本调用都有证据，保留这些记录用于分析未证原因。

本包已有新的多轮成功案例：**第1题三次更新后证明成功，第5题两次更新后证明成功**，详见[09](09-五题尝试与并发结果.md)。

F 的题目是：`f(0)=1`，`f(n+1)=2*f(n)`，证明 `f(n)=2^n`。会话为 `online-20260827-f`，树为 `online-20260827-f.tree0`，使用与 E 相同的固定 REAL-Prover 7B 和 strict V1 训练目标。

## 五轮具体发生了什么

“张量改变数”按学习回执计数；第一轮新增 optimizer 状态，之后连续更新已有状态。所有轮次的 loss、梯度、参数和 optimizer 状态有限数值检查均通过。

| 轮次 | 搜索 step / 学习节点 | 版本变化 | LoRA 改变数 | value-head 改变数 | optimizer 状态 | 后续使用新版本的首个 step |
|---|---|---|---:|---:|---|---:|
| 1 | 1 / 0 | v0→v1 | 196 | 4 | 新增 1188 个张量 | 2 |
| 2 | 2 / 1 | v1→v2 | 392 | 4 | 改变 1188 个张量 | 3 |
| 3 | 3 / 1 | v2→v3 | 392 | 4 | 改变 1188 个张量 | 4 |
| 4 | 4 / 1 | v3→v4 | 392 | 4 | 改变 1188 个张量 | 5 |
| 5 | 5 / 1 | v4→v5 | 392 | 4 | 改变 1188 个张量 | 6 |

| 轮次 | 实际训练的正访问候选 | 累积 visits | value 目标 | 总 loss |
|---|---|---:|---:|---:|
| 1 | `intro n` | 1 | 0.591458355 | 0.667746256 |
| 2 | `induction n <;> simp_all` | 1 | 0.356377133 | 3.654008424 |
| 3 | `induction n <;> simp_all` | 2 | 0.260147721 | 1.942113927 |
| 4 | `induction n <;> simp_all` | 3 | 0.229848058 | 0.161556434 |
| 5 | `induction n <;> simp_all` | 4 | 0.239171631 | 0.170471268 |

五轮均发生在题目未证明时，reward=0；value 目标来自搜索 backup。每轮只有一个正访问候选，目标分布均为 `[1]`。第 2—5 轮重复训练同一节点的候选，访问统计继续累积；它们并非五组互不相关的新证明样本。

## 五轮更新的验证记录

原始 HTTP 共 72 个 job，含五个独立 LEARN event：`s1.n0`、`s2.n1`、`s3.n1`、`s4.n1`、`s5.n1`（均带完整 F 会话前缀）。它们分别对应 optimizer step 1—5 和版本 v1—v5；回执 `applied=true`、`idempotent=false`。

每轮 CPU 回执、HTTP 原始回执和最终快照保存的 event receipt 完全对应；adapter、value head、optimizer 的前后摘要逐轮衔接。初始 v0 与最终 v5 的真实快照张量另行读取核验。**中间轮次使用回执和连续摘要取证，没有逐轮独立读取一份中间快照。**

后续请求核验还确认 v1—v5 均被生成调用使用。v1—v4 各有一组生成，v5 有 26 组生成。实际使用新版本并不保证候选有效：v1—v4 各生成的两条候选均立即输出 EOS，v5 的 52 条中有 45 条立即输出 EOS。合计 64 条候选中 53 条为空。这解释了空输出来自哪里，尚未确定模型为何频繁提前结束。

## 在包内查哪些文件

| 文件 | 用途 |
|---|---|
| [multi-round-ttt.json](../evidence/multi-round-ttt.json) | 本页逐轮数值、event、请求 ID、版本、参数摘要和来源文件哈希 |
| [independent-wire-f.json](../evidence/independent-wire-f.json) | 72 个 HTTP job 与 CPU 记录的逐轮对应，`gaps=[]`，状态为 `MATCHED_PARTIAL_WIRE` |
| [recurrence-snapshot-audit.json](../evidence/recurrence-snapshot-audit.json) | 真实初始/最终快照及五次 event receipt、optimizer step、参数摘要 |
| [wire-snapshot-crosscheck.json](../evidence/wire-snapshot-crosscheck.json) | 每轮 CPU / HTTP / snapshot receipt 的交叉核验 |
| [f-empty-tactic-analysis.json](../evidence/f-empty-tactic-analysis.json) | 各版本原始 EOS/空输出明细 |
| [raw-evidence.tar.gz](../evidence/raw-evidence.tar.gz) | F 的 observer、checkpoint、树、结果及 HTTP 原始记录；解包方法见[证据说明](../evidence/README.md) |

## 与最新成果的关系

F保留为未证案例。后续第1、5题完成了多轮更新后证明成功，并核对了每轮新版本的后续使用、真实首末参数快照和独立Lean证明。最新成绩及并发情况统一见[09](09-五题尝试与并发结果.md)，尚未完成的工作见[10](10-原设计逐项对照与实际流程.md)。
