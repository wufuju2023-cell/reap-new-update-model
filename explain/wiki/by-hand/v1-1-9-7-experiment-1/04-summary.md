# 汇 — v1-1×9-7 Experiment-1 结论（2026-09-07）

## 实验全景（机制验证全绿）

| # | 验证点 | 结果 | 对应 9-7 文档 |
|---|---|---|---|
| E1 | value 端点稳定性 | sd=0（确定性前向，无采样） | — |
| E2 | 证据注入改变后验 | ΔV +7.3~+8.8（内容敏感）；phi 属 Lean selection 层 | 09-B vs 09-C 分层 ✓实测 |
| E3 | prompt 措辞敏感性 | spread +2.4~+4.6（模板化注入的需求） | 03/07 证据熵 |
| E4 | GPU 候选多样性 | distinct 7-11/16（Jaccard 0.0-0.27） | RULE-0 树内多样性由 GPU 保证 ✓ |
| E5 | 温度=宽度旋钮 | unique 2→4→6（T 0.4→1.6） | 01 温度数学 ✓单实测 |

## 三条实证结论（并入 v1-1 决策）

1. **Bayesian 后验路径已可观测**：证据内容注入 → value/policy 分布确定性地改变
   （ΔV 显著）；权重组 phi 尚未直通 GPU（分层符合预期，下一步接 MCTS selection）。
2. **GPU 侧天然支撑树内 K 多样性**（E4/E5：16 候选 7-11 不同，温度调节宽度）——
   RULE-0 的"树内多样归 GPU、路线多样归证据环"分工在真实 7B 上成立。
3. **输入形态敏感**：prompt 模板化是证据注入工程的前置条件（否则措辞噪声污染结论）。

## v1-1 下一步接线清单（排序）
1. **phi 接入 MCTS**：`Reap/Tactic/TreeSearch.lean` selection 处 prior 乘 φ
   （π'∝π·φ；φ 从 RingLog 读，TTT 每 step 更新后重读）——对应 09-C。
2. **证据模板化**：Evidence.payload 进 prompt 的「Related theorems」区块按固定格式
   （`lemma <name>`），加 sourceDesc 注释行（可审计）。
3. **真实 opencode 后端**：`OpencodeEvidenceBackend.evidence_snapshot` 接 `opencode run` 产物
   （每 target 1-2 次调用，成本受控），产出 → Lean 验证门（07 的验证门复用）。
4. **E1 门复用**：任何扩容前重跑 E1+E2（先诊断后扩张，映射经验 #1）。

## 关键防护（防重演）
- 运行状态：GPU server **保持在线**（:8000，full-v3 头；weights_only/raw 修补已在运行时目录）
- 协议文件：`driver/runtime_transport.py`（OpenAI 嵌套解析已修）
- 本进度板三处同步（本目录 / GitHub new-reap-mcts-ttt-public 分支 / 云端 NFS 备份）
