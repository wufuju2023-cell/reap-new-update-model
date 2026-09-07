# 01 — AlphaProof 锚点：论文可量化条款与差距矩阵

> 论文原文以 `2026-9-5-plan-original-paper/01-paper-baseline.md` 的引文（页 2–3、9–12）为准。
> 本页只做"可量化对齐"，不重复转载正文。外部可见的最新关联：AlphaEvolve
> （arXiv 2511.02864，DeepMind）：其框架明确把 AlphaProof/Deep Think 用作
> **proof-assistant 层**——即"生成器提出、自动化证明器验证"的流水线定位。

## 1. 论文可测量条款（何水平才算对齐）

| 条款 | AlphaProof 论文 | 我们的 V1 现状 | 差距 |
|---|---|---|---|
| 模型 | ~3B encoder-decoder；12T/3T tokens 预训练 | 7B causal LM（frozen）+ 4 个值头张量 | 架构路径不同但价值语义已对齐（可接受替代，须以 benchmark 证明） |
| Lean SFT | ~300k state–tactic pairs，~5M tactic tokens | full-v3 状态级训练 205,628 states；Mathlib 混批仅 2 条/窗口 | **数据维度 ~1–2 个数量级差** |
| autoformalization | ~100 万 NS → ~8000 万 Lean statements | 无内容级 formalization；只有人工题 + 少量变体 | 需要人工课程先行，formalization 靠后 |
| main RL | ~1M learner steps；matchmaker；分布式 actors | 单卡单进程；两次在线 batch；无 matchmaker | **机制已备，规模未起** |
| learner batch | 90% verified replay + 10% Mathlib | 同配比（9+1），但只有 2 个窗口 | 比例对，条数不对 |
| TTRL | target + 数十万 variants；hundreds of thousands 量级 | 无 variant curriculum | **最大缺口** |
| compute | ~80,000 TPU-days（主 RL） | 1 × MI300X 192G | 论文量级不可复现；须以"有质量的小规模"论证 |
| solve 门槛 | 有独立 proof 最终 check 的闭合证明 | 已具备（禁网 Lean 验收） | — (已对齐) |

## 2. 我们的"可复用资产"（相对 AlphaProof 的独特位置）

1. **完全开源可复现**的 Lean 验证链（fixed-image 0 network 0 mutation dry-run）
2. **冻结 7B 表征 + 轻量 LoRA/head**：单卡 192G 即可做"回放训练 + 在线 TTT"，不需要 TPU 集群
3. **categorical remaining-distance 语义与 AP 相同**（AND = longest branch）
4. 一个**通过独立验证的成功闭环**（H1 平方望远镜）+ 一个**matched-random 因果证据**（F1）

## 3. 战略结论

- 我们**不会**追逐 8 万 TPU-day；但在"小规模内的每条曲线"上可获得**与论文相同的机制学结论**：
  - verified replay 规模 → learner 更新质量曲线
  - value-on/off 的固定预算搜索收益曲线
  - target variants 数量 → solve 成功率曲线（10/100/1k/10k/100k，论文 ablation 同样画过）
- 在论文的五个可控维度中，**投入产出比排序**：target variants（TTRL）> verified replay 扩容 > main-RL steps 数。第一个小实验必须先拿 **value-on/off 单点实验** 以最小成本证明"这条曲线的起点不会为负"。
