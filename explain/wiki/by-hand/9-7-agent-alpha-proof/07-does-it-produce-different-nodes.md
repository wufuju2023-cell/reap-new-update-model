# 07。会产出不同的节点吗：来源、条件与坍缩边界

## 直接回答：会，当且仅当"检索多样性"成立

Agent 不是"节点生成器"，节点多样性来自它的**信息径**：

| 来源 | 为什么多样 |
|---|---|
| 仓库导航路径 | 不同 session 读不同文件（同类问题在不同模块有不同 answer/API） |
| lemma 检索集 | Mathlib 同功能多版本；REAP 仓库自定义 lemma；每次返回一个候选集（非单条） |
| 证明骨架 | 分解、引理拆分、结构归纳 vs 重写策略不同 → 节点层次不同 |
| 并行 Agent | 不同上下文/温度/提示指纹 → 搜索树分叉 |

## 节点在哪个层面有差异（明确两种节点）

1. **MCTS 树内节点**（tactic 层面）：仍由 GPU policy 出 —— 不受 Agent 影响；
2. **路线层节点**（course/variant/replay 层面）：Agent 修改的是"哪些先验可用 / 哪些可行路线被发现" → 这条路线的**根化子树**在课程上表现为多条不同路线。

所以"不同节点" = 不同 route/先验/骨架，而非替换去树内 sampling。

## 真正的坍缩条件（会变单答案的）

1. **锁死检索**：固定 repo 文件白名单 + 固定 lemma 单条 + 固定骨架模板 → 每 session 同解（单点）；
2. **单 session 串行复用**：第二个 session 继承第一个 session 找到的 lemma（同模式收敛）；
3. **回放单边**：只把"最终成功那条"写进 replay（对应 03 的 One-chain-only）；
4. **K 替换**：把树内 K 采样换成 Agent 单条建议（搜索坍缩，红线）。

## 反坍缩设计（RULE-3 扩充）

- 每个 target 至少 **2 个隔离 session**（不同 profile：温度/提示/禁 lemma 清单）；
- 限制复用：session 间禁止共享"已发现 lemma"缓存（除非作为明确全局库）;
- 检索多样度公式登记：每 session 的 lemma-set Jaccard < 0.7 才算两个不同路线；
- 成功路线必须 ≥2 条才允许进 learner 回放（否则只进 evidence 不回放）。

## 判定：你要的格局

```
GPU policy（树内、每节点 K、progressive sampling）
     ↑ 共同消费
Agent Lean 库循环（树外，仓库检索 → lake build → smoke → 路线库）
     ↑ 产出被去重/验证后写入：库条目 + 骨架 + 变体 + replay
```

在该布局下：树内节点多样（GPU 保证），路线节点多样（Agent 检索多样性保证）。两者独立，不互相顶替。
