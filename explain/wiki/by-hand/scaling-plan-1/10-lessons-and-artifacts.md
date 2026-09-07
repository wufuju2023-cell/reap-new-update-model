# 10 — 训练成果与经验总结（截至 2026-09-07）

> 本文是"scaler 第一阶段"（scaling-plan-1）全部实验的浓缩成果与经验。
> 7B MCTS/TTT 价值头系统从"已训练/已发布"推进到"已诊断、可版本化、可回退、可复现"。

---

## A. 现有训练成果（事实清单）

| 类别 | 成果 | 证据 |
|---|---|---|
| 机制闭环 | full-v3 课程/MCTS/联合 TTT/Lean 验收/发布退役 全链路 | H1 平方望远镜课程（exp-078cea5460cb-01；独立 Lean 通过） |
| 因果证据 | full-v3 vs exact-random matched pair（F1：full 成功、random 失败） | 02_成功案例/01_F1 匹配随机头因果案例 |
| 可加载 artifact | 3584→256→64-bin head + LoRA，HF 已发布（SHA 登记） | alphaproof-full-v3-value-head（c59450c, 220MB backend） |
| 原生训练兜底 | 80k+125,628 特征、数据全量在 NFS、训练脚本可复现 | head-runs/（registry 6 版本） |
| 版本化体系 | 6 个 head 版本链 + 回退锚 + 双副本（工作区/HF） | head-runs/registry.json + WufuJu/reap-value-head-v1-scaling |
| 诊断基线 | E1 calibration 223 样本、温度标定、跨 family 对比矩阵 | 07/08/09 三页 |

## B. 最重要的经验（按可迁移性排序）

### 1. 先诊断，再扩张（本阶段最重要的纪律）
把"训练规模"拆成四个独立问题：**能加载→能校准→能排序→能帮助搜索**。
2026-09-05 计划断言"先评估再扩"，执行后证明这是对的一半：
- GPU probe（1h）就暴露"饱和"猜想；
- E1（4h）实锤 ρ=0.34、低估 57%；
- **如果没有 E1 直接上 S1（50k replay 扩容），会把一个未校准 value 放大成
  50k 规模的错误**——纪律的杠杆收益 > 一次实验成本。

### 2. 评估集决定结论（family-gap 陷阱）
同一 head：77e 数据家族内 within-root 排序 **0.67**（看起来很好）、
跨 family 223 样本 **0.27**（不行）。→ **任何 value 结论必须标注
"同分布/跨分布"**；对 lean-tactic 这类结构化域，家族分布 gap 是常态不是异常。
后续 E2 设计必须改为 family-matched value-on/off。

### 3. 温度是"标尺"不是"分辨力"
T≈2.0 把 mean_pred 从 4.2 扳到 9.8（≈真值）但 ρ 峰值仍只 0.30（T=1.2）。
**校准性与排序性是两个正交维度**——MCTS 依赖排序性；只做温度标定会让
"value 看起来准"但 search 无改善。

### 4. 类平衡有过热边界
长尾 ×8 过采样让 ρ 从 0.327(×2) 塌到 0.172(×8)：
**跨科泛化与类平衡呈现倒 U 关系**，×2 是无损临界点（中性），
不要为了"看见长尾"牺牲近程分辨。

### 5. "已训练"≠"可复刻"
用同数据+同超参复训（mix4）也只 0.272 < 原 v3 0.339——
**原训练留有未记录变量（epoch/划分/种子/特征版本）**。
教训：训练系统的`记录契约`（config+seed+manifest+epochs）优先级高于一切"重训成功"；
我们的管线现在把"记录"做到了 registry 里，但**上一代训练没有**——下次别再丢。

### 6. 长时运行工程学（对抗 1h 闲置回收）
- 训练进程持续吃 GPU = 不被判闲置；长窗拆 200s/段 + **断点续传**（progress.json/COMPLETE 门）
- 实例重开流程已全自动化（restore 脚本 v4.4 防覆盖 guard + 双通道 + HF 令牌持久化）
- **教训事故**：新实例上误跑 backup 覆盖了旧归档 → 失去 tailscale 旧 state 与
  PiAgent 登录态（部分已重建）→ 价值：任何破坏性动作前读 guard 文档。

### 7. 版本化 = 回退成本 0
每个 run：`{run_id, config, head_sha256, e1-metric, origin}` + 双副本（NFS+HF）。
回退=copy 一条命令。**发布即回退锚**（v3-baseline 从在线 artifact 提取）。

## C. 资产地图（哪里有什么）

| 资产 | 路径 |
|---|---|
| 经验/计划/报告 | 本目录 00–10（scaling-plan-1/） |
| 训练器/特征生成器/E1 复测 | scaling-plan-1/code/ |
| 版本链 | /mnt/workspace/head-runs/registry.json（容器持久卷） |
| 结果权重 | HF `WufuJu/reap-value-head-v1-scaling`（public，6 版本 3.7MB each） |
| 原 v3 artifact | `alpha-proof-open-source/alphaproof-full-v3-value-head`（private） |
| 运行数据 | /mnt/workspace/new_value_head/（features/dataset/snapshots） |

## D. 下一步（候选，待门）
1. **E2 改 family-matched（推荐首发）**：在同一 family 树内做 value-on/off 固定预算
2. S18 head 补测（数据已有）：跨科 ρ 报告后并入矩阵
3. `mix5`：把"未被复刻变量"捕获（epochs=60 + 记录 history 落盘）后再复刻
4. 只有当 family-matched E2 出现净增益 → 才重启 S1 扩容讨论
