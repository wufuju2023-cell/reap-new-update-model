# 05 — 并行任务与风险

## 可并行项（不互相阻塞）

| # | 任务 | 资源 | 与谁共享 | 产出 |
|---|---|---|---|---|
| P1 | E3 support & overflow 审计 | 0 GPU | 无 | support-overflow-report.md |
| P2 | holdout 数据集构造（theorem-level split，12–20 题 verified trees） | CPU 少量 | 无 | holdout-v1.tsv + 特征缓存 |
| P3 | E1 calibration 脚本（Spearman/ECE/overcoverage）开发 | CPU | Lean 校验器 | 指标脚本 + 合成报告 |
| P4 | S1 的 replay 抽取器（树 → verified rows，dedup + theorem split） | CPU | 无 | verified-replay-v1/ |
| P5 | 监控/事故规则检查脚本（延迟误判、refresh 接线、dry-run 0 network 0 mutation） | 0 | 无 | 检查清单 |
| P6 | variants 生成器 MPSC 原型（S2 预研：prompt + 程序化变换 + Lean 校验 + dedup 流水线） | CPU | P2 数据 | 变体生成 demo（10 题 × 50 variants） |
| P7 | H1 立方 `course-f86f5c4fcb50-01` 继续自然运行 | GPU 小窗 | E2（分时） | 更高难度案例或正式评估记录 |
| P8 | **HF 账号与 repo 整理**（full-v3/R2/review 系列归位、README 元数据、发布 checklist） | 0 | 无 | 一致 artifact 家族 |

串行主干（不可并行）：
```
E1 → E2 → （gate）→ S1 → S2 → S3
```
原因：每个后续步骤的数据都会污染先前结论，且 artifact contract 明确要求逐级 release。

## 风险与反制

| 风险 | 概率 | 影响 | 反制 |
|---|---|---|---|
| E2 显示 value 无效/中性 | 中 | 计划转向（不做规模扩张） | 预先接受：P4/P6 作为"降级价值"；机制分析下沉，避免大规模预算 |
| 单卡窗口撕裂长任务 | 高 | 进度丢失 | 12–18h 批 + 每窗 backup（v4.4 guard 已防覆盖） |
| Lean 变体验证慢（CPU 瓶颈） | 中 | S2 变体预算上不去 | 先建可并行 CPU 队列（23 核）+ 去重缓存；必要时 2h/题预算阶梯 |
| 生成式课程放浪（伪证明/离题） | 中 | 污染 replay | 双检：Lean 验证 + success/failure 标签双盲校验 |
| 特性变化导致跨 artifact 污染 | 中 | 结论不可信 | 任何新 run 新 identity；release gate 检查 base fingerprint |
| AI 时间超支（单实例 12h 回收规律） | 高 | 窗口中断 | 状态中断点设计：checkpoint 每 30min + 进度 json + backup 自动化 |

## 事故规则（直接引用 discussion 02_工程事故与监控规则）

- 延迟阈值（E2 期间）必须提前注册并监控——超阈值自动降级 value-off 放
- refresh 接线必须 `value_training` 两段审计字段；旧 validator 不认识时 fail-closed
- 任何 TTT 更新后跑 dry-run 证明（0 network / 0 mutation），再 seal/publish/retire

## 结档所需最低证据（每个阶段）

1. identity record（source/base/adapter/head digest）
2. 训练/生成 receipt（行数、mix、KL、histogram、compute）
3. paired fix-budget 对比 + 独立 Lean 结果
4. release 门清单通过截图/日志
5. 失败信息与边界（绝不删）
