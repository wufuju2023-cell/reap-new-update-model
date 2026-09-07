# 08 — H1 长距离重训实验（79efd240 数据 + 复测）

日期：2026-09-07

## 方法
1. 使用 `dataset-79efd240`（train 80k / val 8k / test 8k，**value_class 覆盖到 d≈38**，
   长尾：d≥8 约 900 行）预缓存特征（features-79efd240，3584-dim fp16 shards）
2. 原训练脚本 `code-79efd240/scripts/train_value_head.py` 重训 64-bin head：
   30 epochs / batch 4096 / lr 1e-3 / patience 5 / 确定性种子
3. 产出：`head-v1_3-79e.pt`（包含 initial matched-random + trained 两份，report.json 归档）

## 训练过程报告（脚本自带）
- **79e 验证/测试集**（同分布）：within_root 排序准确率 **0.669**
  （同 root 7980 可比对，正确 5337）——显著优于 full-v3 在长距离段上的表现
- 长距离段（17–32 档）仍 MAE≈13.7、argmax 准确率 0（计数 32）——尺度仍偏

## E1 复测（跨分布：223 样本 dataset/validation.jsonl 抽样 + 新 head）

| 指标 | full-v3 head | 79e head | 比较 |
|---|---|---|---|
| Spearman ρ | **0.339** | 0.266 | ❌ 79e 更差 |
| mean_pred | 4.29 | 5.58 | 更接近真值 9.98（+30%） |
| max_err | 21.5 | 18.96 | 略好 |
| per-class d∈[1,8] | 3.2–4.8 | 3.8–6.9 | 斜率更陡（d 单调性略好） |

## 解读（本文最重要的两句话）
1. **同分布重训有效**：79e 数据（含长距离）让 head 在自己 family 的树内排序
   达到 0.67——证明"补长距离样本"是该方向的正确机制；
2. **跨科目泛化不足**：一旦换到 full-v3 课程族的验证集（E1 223 样本），
   79e head 的排序弱于 full-v3（0.27 vs 0.34），且仍然低估 44%。

### 推论
- **E1 门限的真正瓶颈可能不是"数据长尾缺失"，而是 family/dataset 之间的
  distribution gap**（longtail 难度结构变了），或 validation split 的定义
  （theorem-level split 与 family 内 split 的差异）。
- 下一步不是"更多同类数据"，而是：**跨 family 混合训练 + 对每个 family
  报告 within-root 与 cross-root 排序**（双维报告），并保留 full-v3 与 79e
  两个基线作为迁移诊断。

## 与滚动计划的关系
- H2（温度）与 H1（重训）都未能让 E1 主指标（E1 sampling ρ）过 0.55 门；
- S1 扩张条件仍不满足 → 维持"head 改进线"，下一步建议动作：
  a) **混合迁移实验**：full-v3 数据集 + 79e 长尾行（按 family 分层）合并重训，
     目标：E1 ρ≥0.5
  b) 建立 **family-level 报告工具**（每 family：within-root acc + cross-root ρ）
  c) 若混合后 ρ 仍 <0.5 → 用 S18 候选（已有全套数据）对照验证 head 架构差异

## 复现
- 训练：`code-79efd240/scripts/train_value_head.py --train-limit 80000 --output head-v1_3-79e.pt`
- E1 复测脚本：`/root/exp1`（e1b-fwd + 本页脚本），日志证据在 `/root/exp1/head-v1_3-79e.pt/report.json`
