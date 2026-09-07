# GPU Probe 实验报告（1 小时窗，2026-09-07）

## 目的
验证 REAL-Prover 7B + full-v3 64-bin categorical head 在单卡 MI300X 上的
**真实前向链路**能否跑通，并测量 baseline 时延/显存——为 E1/E2 提供先验。

## 配置
- 设备：AMD MI300X（206GB VRAM，ROCm 7.2.3，torch 2.11 w/ ROCm）
- 模型：FrenzyMath/REAL-Prover fe76f68d（本地 15GB Safetensors）
- artifact：`alpha-proof-open-source/alphaproof-full-v3-value-head`（SHA 校验 ✓ becf7c4c…）
- 加载：base fp16 → PEFT LoRA（392 张量，r=16，7 个 target modules）→ 64-bin head（3584→256→SiLU→256→64）
- 协议：`verified_backend.value` 语义（softmax×arange(1..64) → clip [1,64]）

## 结果

| 指标 | 值 |
|---|---|
| base+adapter+head 加载 | **35.8s**（含首次权重读入） |
| 前向 3 个模拟 Lean state | 单 state <0.1s（40–54 tokens） |
| policy 生成（16 tokens, T=0.8） | **1.13s**（~0.07s/token） |
| 显存占用（推理后） | **16.5GB / 206GB**（≈8%） |

### expected distance 输出（关键信号）
| state（模拟 Lean tactic state） | 预计距离 |
|---|---|
| s0: h:a=b, |- b=a | **1.148** |
| s1: |- 2*(n+1)=2n+2 | **1.556** |
| s2: p:P→Q, hp:P, |- Q | **1.269** |

> ⚠️ **观察：三个不同难度 state 全部输出 1.1–1.6**。若真实 state 集含 d∈[5,64]
> 的困难子目标，该 head 可能**低估距离、饱和在低 bin**（训练分布 d∈[1,4] 的
> 表现，E3 审计预判）。这正是 9-6 计划 E1/E3 需要量化的行为；也说明：
> **上线搜索前必须先做 calibration/coverage 检查**，否则 MCTS 会用错误的
> 相对排序引导搜索（可能误导）。

## 与计划的关系
- ✅ prove 链路可用（单卡 8% GPU 负载下，E2 3 题×3 臂×3 seed 的 36–48h 预估需确认
  ——当前时延下应大幅压缩，可重新评估为 ~6–12h/批窗口）
- ✅ full-v3 可加载（identity 校验全过）
- ⚠️ E1（calibration）优先级提升：**先用真实 holdout 复测距离分布**，再跑 E2，
  否则 E2 的"value-on 好处/坏处"会被饱和问题混淆

## 复现
```
/root/exp1/exp1_probe.py（容器内 root；依赖 /root/exp1/artifact + /mnt/workspace/models/REAL-Prover-fe76f68d）
```

## 备注（耗时预算分配）
- 解码/审计 ~35m、编写&运行 ~20m。下一步建议：E1 正式化（真实 state 采样 +
  Spearman/ECE），接入 V1 runtime 的 golden fixtures（policy logprob、value endpoint）。
