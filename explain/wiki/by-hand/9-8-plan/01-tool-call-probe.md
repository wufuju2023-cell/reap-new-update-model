# 01 — 工具调用能力探测（Tool-Call Probe）

## 目的
判定两模型能否**以结构化工具调用**驱动 Lean 证明搜索工具链（搜索/编译/文件读取），
作为后续课程+MCTS+RL 的前置门。

## 被探模型
| 模型 | 位置 | 备注 |
|---|---|---|
| REAL-Prover 7B | HF `FrenzyMath/REAL-Prover`（15G） | 数学证明模型，预期无 tool-call 训练 |
| Qwen3.8-9B-Distill | HF `empero-ai/Qwen3.8-9B-Distill`（19G） | 通用蒸馏（Qwen3.5-9B 系），**可能**保留 tool-call 能力 |

## 工具集（探针用最小 3 件）
```json
[
 {"name":"lean_search","description":"search Mathlib for lemmas","parameters":{"query":"string","k":"int"}},
 {"name":"lean_compile","description":"compile a lean file, return errors","parameters":{"path":"string"}},
 {"name":"file_read","description":"read a repo file","parameters":{"path":"string"}}
]
```

## 测试题（8 条，覆盖 4 档能力）
| # | 场景 | 期望行为 |
|---|---|---|
| T1 | 单工具单参："搜索 `Nat.div_dvd_of_dvd`" | 调 lean_search(query=…) |
| T2 | 单工具双参："搜 `sq_connected` 取前 5 条" | k=5 |
| T3 | 选择正确工具："看看文件 `/repo/A.lean`" | file_read 而非 search |
| T4 | 工具+推理："检查 A.lean 能否编译，如有错读出文件" | compile→（条件）read 链式 |
| T5 | 无关请求："今天天气" | **不调工具**（拒绝能力） |
| T6 | 多轮：给定"上次搜索返回 X"续下一步 | 根据结果正确调用 compile |
| T7 | 格式压力：要求只输出 JSON 调用 | 格式合规 |
| T8 | 抗干扰：两个同名近似工具 | 选择语义正确的那个 |

## 评分（每模型 0-100）
- 格式合规率（可解析为 JSON/函数调用）25%
- 工具选择正确率 25%
- 参数正确率 25%
- 多轮链式与拒绝能力 25%
**门：≥70 判定"工具调用可用"；50-69 需要 SFT 轻调；<50 判不行。**

## 执行方案（GPU 实例重开后可跑）
- 脚本：加载 transformers + tokenizer（chat template），greedy/温度 0.2，max_new_tokens 256；
- 每模型 8 题 ×2 轮；输出 JSONL + 评分表；
- 产出：`tool-call-probe/{reap,qwen}/results.jsonl + summary.md`。
- 备注：REAL-Prover 若 chat template 缺失，用 Qwen 模板兜底（同源 tokenizer 家族）。
