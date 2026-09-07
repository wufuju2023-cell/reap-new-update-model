# v1-1×9-7 Experiment-1 — 进度板

> 实验窗口：2026-09-07（约 3 小时）。任务：在**我们实例**（MI300X gfx942 +
> REAL-Prover7B + full-v3 64-bin head @ `real-search-categorical`, :8000）上，围绕
> 「GPU 分布 × 证据环（Agentic prior/post reweight）」做一轮系统实验。
>
> ⚠️ 进度说明：**任何时刻打开本目录 README/进度板即可看到本任务状态**。

## 一页快照（每条结果会立即更新在下方）

| 阶段 | 内容 | 状态 |
|---|---|---|
| 00 | 汲取 V1 实验经验（7 条→v1-1 动作） | ✅（`00-lessons-from-v1.md`） |
| E1 | value 重复稳定性（10×/题） | ✅（`01-E1-stability.md`）sd=0：确定性端点 |
| E2 | 证据权重敏感性（phi 扫描） | ✅（`02-E2-weight-sensitivity.md`）内容敏感；phi 属 Lean selection 层 |
| E3 | prompt 变体分布敏感性 | ✅（`03-E3-E4-E5.md`）spread +2.4~+4.6 |
| E4 | policy candidates 多样性 | ✅（`03-…`）distinct 7-11/16，Jaccard 0.0-0.27 |
| E5 | 温度/采样扫描 | ✅（`03-…`）unique 2→4→6（T 0.4→1.6） |
| 汇 | 结论与下一步 | ✅（`04-summary.md`）后验已可观测；phi→MCTS selection 接线待做 |

## 环境

- Lean CPU 侧：`cpulean`（v4.28.0，本地编译/运行 ✓，`phi=tanh(Σw)`）
- GPU 服务：tailnet `100.91.25.4:8000`（`real-search-categorical`，full-v3 头）
- 测试题集：`smoke/problems.json`（IMO2019Q1 课程题 + Pell 不变式 + Pell 正增长）

## 用法

- 每次进展：更新本表 + 写入 `E<k>-<desc>.md`
- 结论以「分布被改变」与「是否稳健」为判断标准（9-7 系列 RULE）
