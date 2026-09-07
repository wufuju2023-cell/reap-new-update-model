# 06。Agent Lean 仓库循环（你设想的形态）

> 你的描述：policy 来自 GPU 端 → [写 Lean 前] 让 Agent 读 repo → 做 Lean 搜索 → lake build → smoke test → 产出真正的 Lean 代码。

## 标准流水线（每条都要过验证门）

```
目标题 Lean statement (course 出题)
  → ① repo 导航：读相关 Lean 文件（Nanoproof/REAP/所选 mathlib 区域）
  → ② 检索：mathlib 库 / 仓库自定义 lemma 检索（签名、适用条件、依赖）
  → ③ 假设生成：提出小节 lemma 候选集 + 证明骨架（分解策略）
  → ④ 生成：完整 tactic script（lean 文件级）
  → ⑤ lake build（一个文件一次）验证语法/依赖/时间
  → ⑥ smoke test（lean --run 或关键 tactic 快速执行）
  → ⑦ 产物登记：verified tactic 序列 → "candidate cart"（见下）
```

## 产物类型（三选一进系统）

| 产物 | 去向 |
|---|---|
| 已验证的**引理条目** | 库成员/先验增强（树外知识） |
| 已验证的**整条 proof script**（路线） | replay 经验 + 变体（TTRL 数据） |
| 一个可挂接的**tactic 序列骨架** | 树内子树 start（如 3-5 步骨架，后续 GPU policy 继续填） |

## 这个位置为什么"不是每节点出招"却仍影响树

- 树内 K 采样不变（RULE-0）；
- Agent 产出的是**先验知识**：更准的 lemma 指向 + 骨架，使子树更快闭合 → 树节点**结构**（哪些分支先开）改变；
- 它等价于 REAL-Prover 论文里的 retrieval-augmented 一脉（FrenzyMath/REAL-Prover 即 retrieval-augmented Lean prover）。

## 验证门实现（具体命令）

- 语法+依赖：`lake env lean <file>`（或 `lake build` 对应目标）
- smoke：选定目标推进若干步骤后用 `exact`/`sorry` 拦断，或 lean --run 执行探针
- 失败自动修复循环：读错误 → 改 → 重来（≤2 轮，防死循环）
