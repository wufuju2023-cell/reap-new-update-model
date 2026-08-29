# 当前状态与边界

本目录记录仍会变化的状态与工程限制，因此与稳定的设计说明、成功案例分开。

- `01_当前状态.md`：当前最好候选、已完成案例、正在进行的原 H1 难度升级和最终选择门。
- `02_工程事故与监控规则.md`：为什么曾出现“进程异常慢却未及时诊断”，以及此后必须遵守的 checkpoint/PID/unknown 门。
- `runtime-refresh-addendum.json`：Python 侧 refresh 字段和测试的历史收据。它不证明 Lean 搜索 producer 已部署；成功案例明确使用 `selection_value_refresh=false`。

状态文件可以更新，已经封存的案例原始 evidence 不应被改写。
