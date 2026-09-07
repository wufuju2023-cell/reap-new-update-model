# Next Plan：Value Head 评测与规模路线（2026-09-06）

审计日期：2026-09-06。本页基于 2026-09-05 计划（`2026-9-5-plan-original-paper`）与
`reap-new-update-model` master 分支 `discussion/new_value_head_in7b_ex1` 的当前状态，给出：

- **done**：已完成的状态核对
- **next plan**：下一步计划（含顺序与门）
- **break down the task**：任务分解
- **what can be done simultaneously**：可并行事项与依赖

---

## 1. Done（已完成，截至 2026-09-06）

### 训练与发布闭环（真实、有回执）
| 项 | 证据 |
|---|---|
| full-v3 64-bin categorical head 训练 | 205,628 隔离状态；冻结 REAL-Prover 7B 表征 3584→256→64；CP20 课程证明 |
| 题内联合 TTT | `CourseSquareTelescope`：step 7→v1、step 15→v2、终端 success-learn→v3；每次更新 392 LoRA + 4 head tensors 真实变化且被后续 generation/value 消费 |
| 完整 Lean proof | 免费独立禁网 Lean 验收通过；proof/evidence 全套随仓库分发 |
| 发布链 | seal → 发布 experience `exp-078cea5460cb-01` → receipt-backed retire |
| 因果对照 | F1 matched pair：full-v3 成功 / exact R64 random 同预算失败；D1/D2 轨迹相同、P2/H1 无收益的负面结果保留 |
| local V1 runtime | R2（D=8）release + fresh-service 恢复消费，8 gates 通过 |

### Artifact 已上传
- HF：`alpha-proof-open-source/alphaproof-full-v3-value-head`（private, commit c59450c）
- backend.json 220,477,868 B，SHA-256 `becf7c4c6650fca4b11b1087ddd86f4c5dfb69f9b21913bc3f8c81c8f1c37479`
- base revision FrenzyMath/REAL-Prover `fe76f68d9a88f342cb7b546307c20292fea9cced`
- 仅证明"已加载可验"; **不**证明 calibration / value-on 增益 / 更优

### 边界（未证明项，遵循 9-5 断言门）
- 无 theorem-level heldout calibration
- 无 value-on/off 固定预算搜索消融
- 无 real-prior / uniform-prior 端到端消融
- 原立方 H1 fresh session `course-f86f5c4fcb50-01` 仍在自然运行，未完成证明

---

## 2. Next Plan（按 9-5 执行顺序 + scorecard 的 A→E）

### A. 只读加载与身份校验（先做，1-2 天）
1. 下载 HF artifact（容器出站用 `HF_ENDPOINT=https://hf-mirror.com`，已验证连通）
2. SHA-256 匹配 digest；schema `reap.gpu.snapshot.v1`；session `exp-e24fc3c0a20c-01`
3. base revision / tokenizer 匹配；adapter + head tensor names/shapes/dtypes
4. golden fixtures：policy token logprobs、expected-distance

### B. Protocol fixture（与 A 并行中的 B 步）
- policy token scoring、value decoding、trajectory target（OR/AND）、release loading、新 session 隔离
- 无任何一项通过 → 视为另一 run identity，不得借用历史结果

### C. 先评估，后扩容
1. critic calibration：theorem-level holdout、d=1..64 histogram、ranking/calibration
2. value-on/off：同 release/theorem/candidates/simulation budget
3. real-prior / uniform-prior 同预算
4. full-v3 / R2 / frozen-base 三路净效应
5. observed-distance strata（d≤4 vs 未见 bucket）
> 核心问题：**Does V1 value guidance improve fixed-budget verified proof search?**

### D. 扩容 verified replay / mixed learner（C 门过后）
- class histogram 全 D 报告；theorem-level split（禁止同证明跨 split）；Mathlib 混批；单调 holdout gate
- 每 release 记录 source mix + base/adapter/head identity + KL + overflow

### E. TTRL（论文级定义，D 后）
- generalist clone → 版本化 variants（L1 语法校验/去重/演化）→ 新 verified replay → joint learner 更新 specialist
- paired 对比：search-only / target-only / variant-focused；generalist holdout 不污染

### F. 原 H1 立方（继续自然运行，与 A-C 并行推进）
- `course-f86f5c4fcb50-01` checkpoint 49 窗继续到完整证明 + 独立 Lean（成功→进 02_成功案例；失败→正式评估）
- 完整 20 题矩阵未完成前：full-v3 保持"临时首选"，明确"谁最好"与"是否够好"是两个问题

---

## 3. Break down the task

| 工作块 | 内容 | 依赖 | 验证方法 |
|---|---|---|---|
| W1 网络/获取 | artifact 下载通道（hf-mirror）、代码库 codeload/gh api | 无 | 连通性+checksum |
| W2 校验 | SHA/schema/base/tensors、SaaS 只读 | W1 | verify_artifacts 等价校验 |
| W3 Runtime | V1 runtime 加载 + golden fixture + isolation | W2 | fixture 全部通过 |
| W4 评估 | holdout split、calibration 脚本、ablation 引擎、fixed-budget run | W3 | 门：value-on 不劣化 |
| W5 训练 | replay 扩容、mixed learner、TTRL variants、Lean validator | W4 | release gate + 增益 |
| W6 成果输出 | evidence ledger、scorecard、最小报告、上传 | W5 | 07 文档的断言门 |
| W7 工程护栏 | 监控规则（延迟误判/refresh 接线/dry-run 0 network 0 mutation）、事故复现 | 无 | 进程活性/诊断门 |

---

## 4. What can be done simultaneously（可并行）

| # | 并行项 | 依据 | 不与什么冲突 |
|---|---|---|---|
| 1 | W1+W2：artifact 下载与 SHA/schema 校验 | 纯下载+哈希，不动运行时 | 无 |
| 2 | W3 的 B 部分：protocol fixture（policy/value/trajectory/release/isolation）编写 | 只读代码+固定 golden，不需模型在线 | 无 |
| 3 | W7：监控规则脚本化（activity/diagnosis 门、refresh 字段测试历史收据整理） | 独立于模型工作 | 无 |
| 4 | W4@prep：holdout 定理集构造与 theorem-level split（CSV+确定性种子，CPU） | 只做数据，不做推断 | 无（无需 GPU） |
| 5 | W4@calibration 预研：MCTS 统计脚本/mock 数据 dry-run 编码 | 不依赖模型可用 | 无 |
| 6 | H1 立方继续自然运行（F） | 已在上游检查点跑 | 注意与 A/C 共资源时预留 GPU |
| 7 | 20 题矩阵 & random 臂的调度（ModelScope 新调度） | 完成各题预算准备 | 需先有已校验 artifact（W2） |

### 串行依赖（不能并行的）
```
W1 → W2 → W3 → W4(calibration→ablation) → W5(TTRL/api) → W6
```
关键因果：**评估（W4）出现在扩训练规模（W5）之前**——9-5 门与本次计划均要求
"先证明 value-on 收益，再扩大 steps/variants"。

### 风险提示
- 下载/评估期间保持容器 HF 端点 `hf-mirror.com`（github.com 直连不可用；gh/codeload 例外）
- full-v3 与 R2 为不同 artifact contract（64-bin vs 8-bin），任何跨支持复用都是错误归属
- 任何"成功"发布前必须独立 Lean 定稿 + evidence 清单（seal/publish/retire 链）
