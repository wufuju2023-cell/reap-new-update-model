# 02 — 8 小时实验矩阵（在你自己容器执行）

> 目标：独立复现 v1-1 机制证据（E1-E5 + evidence loop），并把你自己的观测回填下述产出。
> 允许因硬件/出口差异导致绝对数值不同——**结论以"方向与结构"为准（9-7 RULE）**。

## P3 迁移验证（40min）

```bash
cd /work/v1-1-agentic-tool
V11_TRANSPORT=runtime V11_POLICY_URL=http://127.0.0.1:8000 python3 smoke/v1_smoke.py
# 期望：3 题 PASS（candidates≥1；value 为正的有限浮点）
```

## P4 实验矩阵（4.5h）

| 序 | 实验 | 脚本 | 判据 | 时间 |
|---|---|---|---|---|
| 1 | **E1 稳定性** | `driver/E1.py`（复制到实验目录后执行，PROBS=…/smoke/problems.json） | sd=0（确定端点） | 30min |
| 2 | **E2 证据权重** | `driver/evidence_loop.py` + 修改 w 列表 | ΔV 与内容相关、phi 属 Lean 层 | 40min |
| 3 | **E3 prompt 变体** | `driver/E3-E5.py` 的 E3 段 | spread>2 即稳健敏感 | 30min |
| 4 | **E4 多样性** | 同 E3-E5 E4 段 | distinct≥7/16 且 Jaccard 中位<0.3 | 30min |
| 5 | **E5 温度扫描** | 同 E3-E5 E5 段 | unique 随 T 递增 | 40min |
| 6 | **后验闭环（重点）** | `driver/evidence_loop.py`（整轮） | ΔV≠0 且与 9-7 09 文档预期方向一致 | 40min |
| 7 | **额外自选**：变体池/多题集（可选） | 自由 | 报告类平衡与 family 标注（经验 #2/#4） | 90min |

> 脚本说明：每步输出打屏；修改文件前先备份（经验 #6/#7：防覆盖 guard 与断点标记
> `<step>-COMPLETE.txt`）。

## P5 产出模板（30min)

按 `03-return-format.md` 填写：环境表（硬件/版本/网络）、E1-E5+闭环 结果表、
「方向一致性」结论（与参考值比较）、差异/异常说明。回传路径：
`git push` 到本仓库说明文件名的 branch/PR，或直接把 `*.md` 交回。

## 一门：完成判定
- [ ] 全部 6 次实验输出非退化（E1 sd=0 若你的 value 端点有采样则注明并解释）
- [ ] 后验闭环 ΔV 明显（或给出**为何不明显**的技术分析——同样可接受，需说明）
- [ ] 无未说明的断言失败；所有 sha 校验通过。
