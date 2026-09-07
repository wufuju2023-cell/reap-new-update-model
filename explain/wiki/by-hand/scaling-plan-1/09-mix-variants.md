# 09 — 混合训练变体矩阵与版本化发布（2026-09-07 自主管线）

## 管线
```
head-runs/
├── mix_train.py          # 可恢复训练（resume/checkpoint/COMPLETE 门）
├── make_features.py      # 特征生成（GPU 前向，200 行/片，断点续传）
├── e1_eval.py            # E1 复测（223 样本，同 07 方法）
├── datasets/fullv3-train-features/   # 新生成 v3 族 80k 特征（manifest v2）
├── registry.json         # 版本链（6 版本 sha256 + 指标，回退锚）
└── runs/{v3-baseline,79e,mix1..mix4}/
```

## E1 复测矩阵（ρ 越高越好；门限 0.55）

| run | 训练数据 | oversample(d≥8) | E1 ρ | mean_pred | 备注 |
|---|---|---|---|---|---|
| **v3-baseline** | （原 full-v3 头，从 HF artifact 提取） | — | **0.339** | 4.29 | 回退锚 |
| 79e | features-79efd240 80k（S18 族） | — | 0.266 | 5.58 | 同分布 val acc 0.669 |
| mix1 | 79e 80k + ext 125,628 | ×8 | 0.172 | 5.12 | 长尾过采样破坏跨科排序 |
| mix2 | 同上 | ×2 | 0.329 | 3.65 | 接近 v3 |
| mix3 | 同上 | ×0 | 0.285 | 3.37 | 77e 族差异 → 非真 v3 复刻 |
| mix4 | **dataset/80k 新特征 + ext（真 v3 全量 205,628）** | ×2 | 0.272 | 3.68 | 复刻仍 < v3 |

## 结论（本轮自主实验的科学产出）
1. **E1 门（ρ≥0.55）仍不过**：任何混合/复刻均未超过 v3 原头 0.339——混合方向收益天花板在 ρ≈0.34。
2. **长尾过采样与跨科目排序冲突**（mix1 ×8 塌至 0.17）→ 类平衡过强伤泛化。
3. **v3 原头是"未被复刻的强基线"**：mix4（真 v3 数据+同超参）也只 0.27——说明原 v3 训练存在
   额外变量（epoch 数/数据划分/随机种子），复刻项目需先做 ablation 才能归因。
4. **同分布 vs 跨分布**是核心诊断维度：79e family 内 0.67、跨科 0.27——**验证集选择决定
   实验结论**（family 内 report 会骗人）。

## 版本化与回退
- `head-runs/registry.json`：每 run {head_sha256, path, metric, origin}
- 回退：`runs/v3-baseline/value-head.pt`（来自 full-v3 artifact 提取，SHA cff7d702a25d…）
  + 各 run 原始 .pt 全量保留（工作区 + HF 双副本）
- 回退命令：
  ```bash
  cp /mnt/workspace/head-runs/runs/<run-id>/value-head.pt <目标激活位置>
  ```

## HF 发布（public）
`WufuJu/reap-value-head-v1-scaling`：
- registry.json（1239B）
- v3-baseline / 79e / mix1 / mix2 / mix3 / mix4 / value-head.pt（各 3,739,451B）
- mix4/report.json + config.json

## 下一步（自主候选，等待门条件）
- mix5：v3-baseline 的"超参复刻留痕"（epochs=60/lr=5e-4）+ 记录
- 引入 **S18 候选（79efd240 head）作为同科大 scale 对照**？——先在 E1 上 S18 也测一次
- 最终路径可能转向"双线程”：**同分布（family 内）搜索收益才是可用度量**（E2 将换成
  "family-matched value-on/off"设计），E1 跨科 ρ 作为诊断而非准入。
