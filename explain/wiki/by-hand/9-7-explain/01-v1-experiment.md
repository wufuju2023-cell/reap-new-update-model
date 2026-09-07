# 01 — V1 研究：它到底在证明什么（证据链硬度表）

## 研究问题
REAL-Prover 7B 的价值头能否用「证明距离」语义改善固定预算的 Lean 证明搜索？

## 方法链（每一步都有回执）
```
LeanTree 成功树 × 205,628 状态
  → 7B 冻结前向缓存 3584 维表示
  → train 3584→256→64 categorical head（softmax + 期望距离 d̂ = Σ d·p(d)）
  → MCTS：value = -d̂（AND 取最深子目标，论文语义）
  → 题内 TTT：LoRA+head 联合更新（v0→v1→v2→v3，必须被后续 generation/value 消费）
  → 完整 Lean proof → 独立 Lean 验收
  → seal / publish / retire（evidence 回执）
```

## 已证明 vs 未证明（07 断言门，verbatim 摘入）

| 断言 | 状态 |
|---|---|
| V1 critic 已训练（392 LoRA + 4 head tensors 真实变化，base 冻结指纹不变） | ✅ receipt-backed |
| V1 critic 可服务（fresh-session load + policy/value HTTP 消费） | ✅ R2, 8 gates |
| full-v3 artifact 可加载（SHA `becf7c4…`/`c1d02…`，schema v3, Lean acceptance passed） | ✅ |
| **critic 有用**（heldout 校准 + value-on/off 固定预算消融） | ❌ 未做 |
| **V1 优于 base / 达 AlphaProof TTRL 规模** | ❌ 未做 |

## 为什么这很重要（对后续方向）
我们已经站在「已训练、已服务、可复刻」的土壤上——
**下一个真正的科学问题不是"能不能训练"，而是"它对搜索有没有用"**
（9-5 计划 Page 03：Does V1 value guidance improve fixed-budget verified proof search?）。
v1-1-agentic 正是冲着这个问题的"测量工具 + 增强手段"去的。
