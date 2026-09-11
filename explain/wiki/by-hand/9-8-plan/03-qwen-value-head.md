# 03 — 为 Qwen3.8-9B 新训 value head（若无 head）

## 结论先行
`empero-ai/Qwen3.8-9B-Distill` **没有 value head**（纯 LM；其配置为
`Qwen3_5ForConditionalGeneration`，hidden 4096）。若走 Qwen 路线，需要新训一个
**4096→256→64** 的分类 head（方法学完全复用 V1）。

## 架构（与 V1 同构）
```
h_9B(s) ∈ R^4096 → Linear(4096,256) → SiLU → Linear(256,64) → softmax → d̂ = Σ d·p(d)
value(s) = -d̂   （AND 状态取子目标最大 d̂）
```

## 数据（三层来源，按优先级）
1. **现成**：`leantree_mathlib.jsonl`（205,628 状态，云端 `data/`，含 proof_depth）
   ——可直接作为 v0 训练集（与 V1 相同切分策略：80/10/10、theorem-family 隔离）；
2. **2k 代数课程**（04 产物）的成功树（课程运行产生的新状态，加入增量训练）；
3. **TTRL 变体**的成功树（后续，持续扩）。

## 训练目标
$$
\mathcal{L} = \mathcal{L}_{\text{CE}}(d) \quad (\text{64 档，two-hot 处理在线非整数回传})
$$
与 V1 完全一致（冻结 backbone 前向缓存 → head 训练；在线阶段 joint 更新可选）。

## 工程改造点（相对 REAL 版）
| 组件 | 改动 |
|---|---|
| 特征提取 | `real7_feature_fingerprint` → 泛化为 base-agnostic（Qwen 4096） |
| head 尺寸 | 3584→4096（artifact schema 字段同步） |
| runtime 后端 | 复用云端 `qwen35_backend.py`（已有骨架，需接 head 加载） |
| artifact 契约 | 新 schema `new_value_head.categorical-head.qwen-v1`（不混用 REAL artifact） |

## 预算（MI300X 级）
- 特征缓存 205k×4096（bf16）≈ 1.7TB？→ 降采样或分片（V1 用了分片缓存）；
  先用 80k 状态起步（V1 有 20k/40k/60k/80k 边界经验）。
- head 训练：小时级（小 MLP）；主要成本=前向缓存（7B→9B 略慢）。

## 验收门（与 V1 对齐）
- [ ] artifact SHA + schema + tensor 名/形状校验通过；
- [ ] 固定 prompt 下 value 确定性（E1 风格 sd=0）；
- [ ] 距离分布覆盖 d∈[1,64] 的直方图报告（class histogram）；
- [ ] 与 REAL full-v3 的数值域对比（不做"有用性"消融，只做加载与一致性检查）。
