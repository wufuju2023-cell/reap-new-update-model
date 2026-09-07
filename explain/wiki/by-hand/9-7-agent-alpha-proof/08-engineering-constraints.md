# 08。工程约束与实现顺序（lake build 慢的现实）

## 现实量级

| 操作 | 耗时量级 | 结论 |
|---|---|---|
| Mathlib 冷 build | 分钟级-1h+ | 绝不"每节点"跑；只跑"每路线" |
| 单个目标文件增量 build（依赖已建） | 10s-1min | 可接受（每路线 1-2 次） |
| smoke test（lean --run 探针） | 1-10s | 频繁使用 OK |
| repo 检索（grep/signature 解析） | 0.5-3s | 高频 OK |

## 缓存与增量策略

1. 依赖缓存区：per-repo 预建 mathlib cache（读时 build 一次，路线期共享）；
2. 文件级增量：同一目题文件只在编辑后 rebuild —— 用 `lake build` 增量（它本来就做缓存）；
3. 失败修复只能 ≤2 轮；第 3 轮失败 → 该路线置"failed"，不阻塞树；
4. 异步化：Agent 循环跑在独立队列（与 GPU policy 消费并行），candidate cart 每 5s 刷一次进树。

## 接口实现（最小改动清单）

- 新增 `lean_repo_loop/main.py` + `search.py`（检索）+ `verify.sh`（lake env lean 封装）+ `smoke.py`；
- 新增表：`candidate_cart`（id, lemma-set, route, proof script, status, jaccard-sig）；
- `course_driver.py` 在分配 course 时读取 candidate cart 作为"树外先验/骨架 entry"；
- learner 数据生成器在回放时按 RULE-3 混合（≥2 路线）。
- 全程不变：`gpu_runtime/*`、树逻辑、TTT 联更新。

## 验证门验收（首个里程碑）

- [ ] 3 题 × 每条题 ≥2 个隔离 session；至少 1 题出现"不同 lemma 集"两条合规路线；
- [ ] 每条路线都过 lake build + smoke；
- [ ] candidate cart 去重后路线 Jaccard 中位 <0.7；
- [ ] 整循环 wall time 没有阻塞树内的候选消费；
- [ ] 无 RU-0 违规（GPU sampling 路径未改）。

## 与其系列的关系

- 06 接线 = 09 系列 02 的位置 A（variant/先验生成器）的**工程化实现**；
- 03/07 的双层多样性 = 本流水线成立的前提论证；
- 完成后它成为 9-6 计划 E（TTRL）与 9-7 实验（05）的**共享基建**。
