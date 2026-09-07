# 06 — 路线图 v2：四线（A/B/C/D）与优先级判据

## 四线
| 线 | 内容 | 断言 | 门 |
|---|---|---|---|
| **A** | φ 入树：Lean TreeSearch selection 的 `π'(a\|s) ∝ π(a\|s)·φ`（φ 从 RingLog 读） | φ 会改变 MCTS 分支选择 | E2 重扫时 ΔV 随 φ 单调（或饱和合理）→ 进 B；否则查 φ 注入点 |
| **B** | TTRL 小闭环：变体池（程序化+模板）→ 双路线经验 → mixed learner 更新 → 3 题 paired 对比 search-only | 变体回放带来可测增益 | fixed-budget sol@budget 提高（同一预算；无增益必须有 pre-registered 分析） |
| **C** | 真实 opencode 证据源：`opencode run` 产出（每 target 1-2 次）→ Lean 验证门 → 池 | agent 证据质量足以替代骨架 | 路线 Jaccard<0.7 且验证通过率 > 25% |
| **D** | 长时程 8h 验证：Lean 端 session 协议全链 + TTT learn 事件运行 | 长跑下闭环稳定（不漂移/不 OOM/不丢事件） | 全程 0 门违反；产物可回放（COMPLETE 全绿） |

## 依赖与顺序
```
A（φ→selection）→ B（TTRL 增益）→ C（真证据源）→ D（长时程）
   前提：上游收口（transport 修复合并 + base pin + E4 弱多样归因）
```
- A 前置收口（今天可做，~30-60min）：合并 runtime_transport 修复 →（**git push 上游**）
  → next-1 吸收三发现（更新文档与 FAQ）。
- E4 弱多样归因可并行列队（A 完成后 30min 脚本化归因）。

## 每周的产出节拍（按四件套）
- 周一：断言主线定稿（本周=一条线）。
- 周二-周四：脚本→跑→数→门。
- 周五：证据归档放分支 + 写"本周答了什么/没答什么"。

## 风险表
| 风险 | 机制 |
|---|---|
| φ 无效果（单调没出来） | 日志记录实际 selection 前后分布；回到"φ 注入点"检查（Lean 侧） |
| 变体池太小而没有增益 | 先跑论文 90/10 混合；池质量监督（类平衡，防√x8 塌缩教训） |
| base 版本漂移 | pin live over vs `FrenzyMath/REAL-Prover`；artifact fingerprint 强制校验 |
| 证据源不可靠 | RULE-1 验证门是硬前提；失败率>75% 回炉 agent 提示模板 |

## 结束语
> 我们不是没有方向——是方向太多了被淹没。照"四件套+每周一条线+黄金三角"跑，
> 所有工作都会收敛为一个接一个可证伪的断言；输在哪一步也会清清楚楚。
> 今天的第一件事：**收口上游（A 前置），然后开 A 线**。
