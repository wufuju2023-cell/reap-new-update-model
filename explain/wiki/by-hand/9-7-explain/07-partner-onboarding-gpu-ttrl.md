# 07 — 伙伴指引：新 AMD 实例上的 GPU TTRL 部署与启动（版本地图 + 全部路径）

> 面向：新开的 clean AMD 实例（已有 ROCm/PyTorch/Python，无 Lean、无训练代码）。
> 本文档回答三件事：**训练代码在哪、模型在哪、怎么启动**。
> ⚠️ 先纠正一个易混淆点：`reap-agentic-v1-1` 是**早期骨架**（空 gpu/），
> 活跃仓库是 **`v1-1-agentic-tool`**（GPU 运行时 33 模块在其中）。

## 0. 版本地图（别用错）

| 版本 | 位置（GitHub） | 内容 | 状态 |
|---|---|---|---|
| **V1 GPU 训练运行时** | `wufuju2023-cell/reap-new-update-model-value-head` → `v1-result/20260828-real7b-pell-success/code/gpu_runtime/` | real_backend / categorical_search_backend / learner / mixed_learner / server | 真实跑过（回执在 evidence/） |
| **V1 CPU 编排（TTRL）** | 同仓库 → `…/code/cpu_runtime/` + `…/code/experiments/proof-curriculum/` | online_ttt / run_ttt / segmented_ttt / target_variants / target_curriculum / course_driver | 真实跑过 |
| **V1 打包搬运版（推荐入口）** | `wufuju2023-cell/v1-1-agentic-tool` → `gpu/gpu_runtime/`（33 模块，含 torch2.11 补丁） | 上面 GPU 运行时的 smoke-ready 副本 | 已验证（:8000 服务） |
| v1-1 CPU 侧（Lean+driver） | `wufuju2023-cell/v1-1-agentic-tool` → `cpulean/`, `driver/`, `smoke/` | Lean 4.28 证据环 + transport | 已验证 |
| 文档/计划 | `wufuju2023-cell/reap-new-update-model` 分支 `new-reap-mcts-ttt-public` → `explain/wiki/by-hand/` | next-1、9-7 系列、本文件 | — |
| V2 | —— | —— | **忽略（未实现）** |

## 1. GPU TTRL 训练代码的精确路径

**GPU 侧（真正做联合更新/TTT 的地方）**：
```
v1-1-agentic-tool/gpu/gpu_runtime/
  server.py                       # 服务入口（含 learn/v1 端点 = 训练触发）
  learner.py / mixed_learner.py   # 学习器（LoRA+head 联合）
  continual_mixed_learner.py      # 持续混合学习
  mixed_objective.py              # NLL + KL(frozen base) + categorical CE
  real_backend.py                 # 7B 推理 + policy/value
  categorical_search_backend.py   # 64-bin 头加载 + two-hot 在线目标
  actor.py                        # GPU 单线程执行器
```
**CPU 侧编排（发起 TTT 会话/课程循环）**：
```
<value-head repo>/v1-result/20260828-real7b-pell-success/code/
  cpu_runtime/run_ttt.py            # TTT 主管道入口（segmented supervisor）
  cpu_runtime/online_ttt.py         # 在线 TTT
  cpu_runtime/target_curriculum.py  # 课程目标管理
  cpu_runtime/target_variants.py    # 目标变体（受限确定性变换）
  experiments/proof-curriculum/runner/course_driver.py   # 课程驱动
  experiments/proof-curriculum/prepare_attempt.py        # 难题尝试打包
```

## 2. 模型链接（原始 + 我们的 adapter）

| 模型 | HF 链接 | 说明 |
|---|---|---|
| 原始 base | `https://huggingface.co/FrenzyMath/REAL-Prover` | 公开；7B 4 分片（~15G）；容器无直连时用 `HF_ENDPOINT=https://hf-mirror.com` |
| 我们的 LoRA adapter + 64-bin head | private: `https://huggingface.co/alpha-proof-open-source/alphaproof-full-v3-value-head` | 需授权 token（向 wufuju 要） |
| **公开镜像（推荐）** | `https://huggingface.co/WufuJu/v1-1-fullv3-artifact` | `backend.full.pt`（3.74MB，纯 head）+ `backend.raw.pt`（165MB，adapter+head）；免 token |

## 3. 启动方法（新实例，最少步骤）

```bash
# P0: 代码
git clone https://github.com/wufuju2023-cell/v1-1-agentic-tool.git
cd v1-1-agentic-tool/gpu

# P1: 模型与 artifact
export HF_ENDPOINT=https://hf-mirror.com
hf download FrenzyMath/REAL-Prover --local-dir /work/models/REAL-Prover
hf download WufuJu/v1-1-fullv3-artifact --local-dir /work/artifact
sha256sum /work/artifact/backend.full.pt   # 期望 c1d0255221cc…385d

# P2: 起 GPU server（TTRL 服务）
python -m gpu_runtime.server --backend real-search-categorical \
  --model-path /work/models/REAL-Prover \
  --categorical-value-artifact /work/artifact/backend.full.pt \
  --categorical-value-artifact-sha256 c1d0255221cc1b3a6e80e1d5567f6c535ef0a62f1ab300d237aeaa2610e0385d \
  --categorical-value-artifact-role pretrained \
  --gamma 0.999 --host 0.0.0.0 --port 8000 &

# P3: 冒烟
python3 ../smoke/v1_smoke.py   # 需要 V11_TRANSPORT=runtime V11_POLICY_URL=http://127.0.0.1:8000
```

> Lean：GPU 训练/推理**不需要** Lean；只有做"生成 tactic → 本地验证"或课程 proof 检查时才需要
> （Lean 4.28，部署法见 `next-1/01-deploy.md`）。

## 4. "生成 2k 抽象代数题做 TTRL"——现状与路线（诚实版）

**现状**：代码中**没有**"批量生成 2k 抽象代数题"的现成 pipeline。
现有最接近的三块：
1. `target_variants.py`：受限、确定性的目标变换（`nat_forall_instance` / `and_left/right` 等，输出 Lean 声明，须经编译检查）——**变体不是新题，且规模有限**；
2. `proof-curriculum/`：把难题打包为课程（budgets/attempt/prepare）——是**运行管理**不是题生成；
3. `target_curriculum.py`：课程目标管理。

**要做 2k 题的路线（二选一/组合）**：
- 路线 R1（保守，先跑通）：用 `target_variants` 的受限变换 + 手工种子题（抽象代数族），
  扩张到几十~几百条，跑通 TTRL 闭环（验证门：Lean 编译+证明）后再扩量；
- 路线 R2（计划中的正路）：等/做 **opencode agent 变体生成器**（9-7 系列位置 A），
  产出 → Lean 验证门 → 课程池（2k 量级），此部分代码属 v1-1 的 B 线（未完成）。
- ⚠️ 不要直接承诺"2k 题 pipeline 已存在"——会找不到代码。

## 5. 一页 Checklist（转发用）

- [ ] clone `v1-1-agentic-tool`
- [ ] 拉 `FrenzyMath/REAL-Prover` + `WufuJu/v1-1-fullv3-artifact`（sha 校验）
- [ ] `python -m gpu_runtime.server --backend real-search-categorical …`
- [ ] smoke 3/3（reference value 1.786/1.962/3.216）
- [ ] 训练接口：`POST /sessions/{id}/learn/v1`（event 由 CPU 编排发起；见 `cpu_runtime/run_ttt.py` 契约）
- [ ] 2k 题：R1 起步，R2 待 v1-1 变体生成器
