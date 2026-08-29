# 成功案例

这里仅收“完整 proof + 独立 Lean 验收”通过的真实运行。两个案例承担不同证据职责，不能互相替代。

## 1. F1 匹配随机头因果案例

[01_F1匹配随机头因果案例.md](01_F1匹配随机头因果案例.md)

它比较 trained full-v3 与 exact R64-v3 random：两臂候选和初始条件匹配，full-v3 改变了关键状态回访和分支预算，最终 full 成功、random 同预算失败。它主要回答：“训练后的 value 相对随机头是否实际改善了搜索行为？”

## 2. H1 平方望远镜课程完整闭环

[02_H1平方望远镜课程成功/README.md](02_H1平方望远镜课程成功/README.md)

它展示从 pretrained full-v3、真实 MCTS、两次题内联合 TTT、更新后继续搜索，到模型 proof、独立 Lean、success-learn、seal/publish/retire 的完整链路。它主要回答：“整套课程/value/MCTS/TTT 流程能否真实跑通？”

该课程没有 exact-random 同题臂，所以不能单独证明相对因果优势；F1 提供该证据。课程成功也不等于原立方 H1 已证明。
