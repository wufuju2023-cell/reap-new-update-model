# Reap / AlphaProof 学习路线图

这份文件是整个仓库的总地图，解决两个问题：

1. 一个主题应该去哪个目录、从哪一篇文件开始；
2. 新学习者应该按什么顺序阅读，哪些重复文件可以跳过。

仓库根目录是 /mnt/f/projects/reap-new-update-model（Windows 工作区对应
F:/projects/reap-new-update-model）。下文路径均相对于这个根目录；阅读本文件时，
把路径前面加上仓库根目录即可得到绝对路径。

标记说明：

- **主线**：新学习者优先阅读的唯一入口；
- **配套**：需要实现、复现或深入时再读；
- **选读**：与主线有重叠，除非要查细节，否则跳过；
- **待补**：目前仓库中还没有成稿，不能用相似主题的文件冒充。

## 0. 原始四项纲要与目录对应

| 原始纲要 | 主目录 | 当前状态 |
|---|---|---|
| 1. AlphaGo 历史、数学和训练 | 历史章节尚无文件；机制对照在 explain/10-mcts-usage-and-alternatives.md、explain/11-selfplay-alpha-zero-vs-reap-ttt.md | 历史待补，机制已有 |
| 2. AlphaProof 数学、Nature 论文训练、与 AlphaGo 比较 | discussion/alphaproof-value-head/；补充在 explain/ 和 explain/new-deep-explain-v2/ | 主要讲义已有 |
| 3. V1 计划、代码、CPU/GPU、租 GPU、环境、value head、结果 | plan/、v1-spec/、new-v1-gather-source-code-cpu/、app/、environment-spec/、docker/、v1-result/ | 设计、原型和实证分开保存 |
| 4. V2 CPU 元编程/工具使用、代码和后续计划 | explain/reap-mcts-lean-v2/、new-v2-gather-source-code/、lean-v2/、plan/08-12 | 原型与计划并存，尚未全部验收 |

## 1. 先建立全局认识

### 1.1 仓库定位

主线：

- README.md：仓库的总体定位、CPU/GPU 分工、环境和安全约定；
- lecture/1-road-map.md：本阅读地图。

可选：

- v1-result/README.md：当前 TTT 框架的通俗总览；
- nanoproof/README.md：另一个完整的 theorem-proving pipeline，可用来补充
  预训练、SFT、RL、MCTS 和 Lean server 的背景，但不是本项目 V1/V2 的主线。

### 1.2 建议先掌握的术语

本仓库目前没有单独的术语表。第一次阅读时，至少要能区分：

- policy：根据当前 Lean 状态生成下一步 tactic；
- value head / value network：估计当前状态的可解性或剩余证明价值；
- MCTS / PUCT：安排搜索预算并汇总状态价值；
- Lean verifier：执行 tactic 并做最终 kernel 检查；
- TTT / RTTT：在搜索过程中或题目之间更新附加参数；
- experience / release / snapshot：可继承的参数、发布版本和完整恢复点。

这些概念的首次解释分散在 explain/1-价值头的作用.md、
explain/10-mcts-usage-and-alternatives.md 和 v1-result/README.md。

## 2. Alpha 系列与理论基础

### 2.1 AlphaGo 历史

**状态：待补。** 当前没有专门的 AlphaGo 历史讲义。后续写作时再加入；
现有 MCTS 对照文档只能说明机制，不能替代历史章节。

### 2.2 AlphaGo/AlphaZero 的数学、MCTS 与 self-play

主线（只读这一组即可理解本项目采用 MCTS 的理由）：

- explain/10-mcts-usage-and-alternatives.md：从 AlphaGo 到 AlphaProof、
  V1/V2 的 MCTS 角色、PUCT、价值回传和替代方案；
- explain/11-selfplay-alpha-zero-vs-reap-ttt.md：AlphaGo Zero self-play
  与 Reap-TTT 自改进的形式化对应。

配套：

- explain/new-deep-explain-1/2-mcts.md：MCTS 的树结构和数学展开；
- explain/new-deep-explain-v2/1-cpu.md、2-gpu.md：CPU/GPU 视角下的
  搜索和模型接口。

完整的 AlphaGo 历史叙述和训练史目前也没有独立成稿；这里先读机制，
不要把 explain/11 当作历史章节。

### 2.3 AlphaProof 的机制、训练和本项目差距

主线：

- discussion/alphaproof-value-head/README.md：导读，先看章节重点；
- discussion/alphaproof-value-head/07_AlphaProof从零到完整机制_搜索_Value与TTT.md：
  按一次证明实验的时间顺序解释 policy、value、Lean、MCTS、TTT，以及当前
  随机 value head 与 AlphaProof 级 value 的差距。

配套：

- explain/new-deep-explain-v2/3-value-head.md：value head 的训练设计；
- explain/new-deep-explain-v2/4-3value-head.md：三值/阶段化 value 方案，
  只在需要比较多种 value 目标时阅读；
- explain/4-alpha-proof与智能体训练是否兼容.md：MCTS-PV 与一般工具调用
  agent 训练的兼容性边界。

### 2.4 价值、难题和 harness 的基础概念

主线：

1. explain/1-价值头的作用.md：把证明搜索写成 MDP，说明 value 如何进入
   PUCT；
2. explain/2-hard-problem如何进步.md：问题难度、搜索预算和可扩展性；
3. explain/3-harness.md：局部神经网络与全局过程记忆的分工。

这三篇读完后，再进入 V1；不要一开始就钻进数百个源码文件。

## 3. V1：从 Reap 搜索到训练闭环

### 3.1 V1 的概念设计（新手主线）

新学习者只需先读 plan/ 这一套；它是较适合教学的连续叙述：

| 顺序 | 文件 | 主题 |
|---|---|---|
| 1 | plan/00-index.md | 计划总索引与整体假设 |
| 2 | plan/01-motivation.md | Reap 是什么、项目要补哪一段训练闭环 |
| 3 | plan/02-architecture.md | Lean、MCTS、policy、value、trainer 的接口 |
| 4 | plan/03-environment-mcts.md | 复用的 Lean/Reap 环境和 MCTS 方程 |
| 5 | plan/04-data.md | state-tactic 数据、清洗和 SFT |
| 6 | plan/05-rl-param-update.md | GRPO/PPO、回报、优势和参数更新 |
| 7 | plan/06-evaluation.md | benchmark、solve@B、指标和消融 |
| 8 | plan/07-roadmap.md | 里程碑、硬件、风险和近期动作 |

### 3.2 V1 的扩展计划

完成上表后，再按兴趣阅读：

- plan/08-post-training.md：继续训练、replay 和避免从零开始；
- plan/09-test-time-training.md：题内 TTT、RTTT、在线 value TD 更新；
- plan/10-recursive-self-improvement.md：递归自改进闭环；
- plan/11-curriculum-traditional-llm.md：传统 teacher 的课程/变体路线；
- plan/12-two-model-architecture.md：A=teacher、B=RSI-Reap student 的双模型
  体系。
- explain/12-teacher-evolution-co-evolution.md、
  explain/13-teacher-posttrain-vs-evolution.md：teacher 演化、共演化和传统
  post-train 的比较，属于后续研究选读。

### 3.3 V1 的正式规格（实现时查阅，不与 plan 全部重复阅读）

v1-spec/ 是规范化版本，适合已经理解概念、准备照规格实现的人：

- v1-spec/00-overview.md 至 v1-spec/07-runbook.md：系统、policy/value、
  MCTS/verifier、课程、训练、基础设施、评估和运行手册；
- v1-spec/v1-1-training-methods/：数据、SFT、RL、TTTRL、更新方案和参数目录；
- v1-spec/v1-1-training-on-amd-cloud/：AMD Cloud 上的模型栈、RTTT、数据课程、
  Lean 接入；
- v1-spec/v1-1-training-example/：实例说明、控制面、环境清单和脚本；
- v1-spec/README-render.md、v1-spec/READEME-gpu.md：渲染和 GPU 补充说明。

规则：**学习阶段读 plan/，实现阶段查 v1-spec/；不要把两套文件逐篇通读。**

## 4. V1 的代码和 CPU/GPU 分工

### 4.1 先看架构说明

主线：

- explain/reap-mcts-lean-v1/README.md；
- explain/reap-mcts-lean-v1/00-overview.md 至 05-quality-gates.md：
  overview、架构、rollout sink、batch solver、RTTT hook 和质量闸门；
- v1-result/prompt_for_agent/教学/01-整体流程与CPU-GPU分工.md：
  用一次实验解释 CPU 搜索/验证与 GPU 推理/训练如何往返。
- v1-result/prompt_for_agent/教学/README.md 及其 02-策略价值与训练信号.md、
  03-跨题经验版本与并发.md、04-成功实例与复现检查.md、
  05-跨题经验的设计与代码实现.md：通俗教学线；与 current 报告有重叠，
  只在需要逐步讲解时采用。

选读：

- explain/new-deep-explain-1/1-cpu+gpu.md：同一分工的数学化长文；
- explain/new-deep-explain-1/3-another-version-of-1.md：另一版长文，不必重复读。

### 4.2 CPU/Lean 搜索源码

推荐的完整 V1 CPU 代码归集入口：

- new-v1-gather-source-code-cpu/README.md；
- new-v1-gather-source-code-cpu/reap-upstream/：上游 Reap Lean 核心；
  重点是 Tactic/Step.lean（验证器）、Tactic/State.lean（状态）、
  Tactic/Generator.lean（模型协议）、Tactic/TreeSearch.lean（MCTS）、
  TreeSearch/ 和 PremiseSelection/；
- new-v1-gather-source-code-cpu/reap-training/：
  Reap/Training/RolloutSink.lean、Verdict.lean；
- new-v1-gather-source-code-cpu/python-driver/：
  v1_run.py、v1_sink.py、mock_policy_server.py，只负责薄编排。

注意：lean-v1/ 只有 Reap/Training/ 下的最小训练扩展，不是完整的
Reap 上游源码；不要把它当成 V1 全部实现。

### 4.3 当前可运行的 TTT 实现

实际复现和审计时使用：

- v1-result/reproduction/code/src/cpu_runtime/：Lean 搜索、采集、调度、
  经验和验证；
- v1-result/reproduction/code/src/gpu_runtime/：模型服务、value、learner、
  snapshot、release 和后端；
- v1-result/reproduction/code/src/containers/：CPU/GPU 容器相关材料；
- v1-result/reproduction/code/README.md：源码包导览；
- app/：较小的演示入口，重点看 policy_server.py、value_head.py、
  train_value_head.py、rttt_demo.py、VALUE_HEAD.md。

阅读策略：先读 new-v1-gather-source-code-cpu/ 理解原理，再按需要进入
v1-result/reproduction/code/src/；只想跑 smoke test 时先看 app/。

## 5. GPU 租用、环境搭建与部署

### 5.1 环境理论和本地/云端一致性

主线按顺序阅读：

1. environment-spec/00-index.md；
2. environment-spec/01-parity-theory.md；
3. environment-spec/02-toolchain.md；
4. environment-spec/03-concretization.md；
5. environment-spec/04-bootstrap-reuse.md；
6. environment-spec/05-state-and-reuse-layout.md。

environment-spec/06-instance-lifecycle-log.md 是某次实例的运行日志，
只在需要审计具体实例时阅读。

### 5.2 镜像和 AMD Cloud 实操

- docker/README.md：CPU-MCTS 与 GPU-Train 镜像矩阵、卷、接口和完整 runbook；
- v1-spec/05-infra.md：V1 基础设施规范；
- v1-spec/v1-1-training-on-amd-cloud/00-overview.md 至 04-lean-integration.md：
  AMD Cloud 上真实训练的模型栈、RTTT、数据和 Lean；
- v1-spec/v1-1-training-example/README.md：
  服务器、WSL、GitHub 三方同步；
- v1-spec/v1-1-training-example/control/：
  SSH、opencli、token、环境同步等控制面细节；
- v1-spec/v1-1-training-example/env/、scripts/：实例清单和脚本；
- tools/amd_jupyter/、tools/amdbridge/：远程控制工具；
- v1-result/reproduction/00-环境与代码.md：
  需要实际复现实验时的准备步骤。

新手不必先读 control 全部细节：先读 environment-spec/00-03 和
docker/README.md，真正租 GPU 或排查连接时再进入 control 子目录。

## 6. Value head、数据和训练信号

### 6.1 Value head 的主线

按以下顺序阅读：

1. discussion/alphaproof-value-head/README.md；
2. discussion/alphaproof-value-head/07_AlphaProof从零到完整机制_搜索_Value与TTT.md；
3. explain/1-价值头的作用.md；
4. app/VALUE_HEAD.md；
5. app/value_head.py、app/train_value_head.py；
6. plan/05-rl-param-update.md、plan/09-test-time-training.md；
7. 需要正式接口时查 v1-spec/01-policy-value.md 和
   v1-spec/v1-1-training-on-amd-cloud/01-model-stack.md。

必须记住：仓库中至少有两种不能混淆的 value 目标：

- 题内搜索路线：拟合 MCTS/search backup 转换出的折扣估值；
- 中央/成功证明路线：拟合已经验证的剩余证明步数类别。

它们的参数合同不同，不能只因为张量形状相同就互相加载。

### 6.2 数据、课程和 teacher

这是原始纲要没有单独列出的重要部分：

- plan/04-data.md：数据来源、清洗和 SFT；
- v1-spec/v1-1-training-methods/01-data-spec.md 至 06-catalog.md：
  数据与训练方法的正式合同；
- v1-spec/03-curriculum.md、v1-spec/v1-1-training-on-amd-cloud/03-data-curriculum.md：
  课程和数据安排；
- plan/11-curriculum-traditional-llm.md：
  传统 teacher 的变体课程；
- plan/12-two-model-architecture.md：
  teacher/student 分工、数据流和发布顺序。

当前围绕目标题自动生成变体的课程仍处于暂停状态；相关文件是设计稿，
不要当作已经跑通的 V1 训练结果。

## 7. V1 实际结果、评估与复现

### 7.1 当前结果主线

只读 v1-result/docs/current/ 这一套：

1. 00-总设计与阅读地图.md；
2. 01-题内在线TTT.md；
3. 02-跨题经验与中央学习.md；
4. 03-并发调度与模型同步.md；
5. 04-实际实例与验收结果.md；
6. 05-原文与Docker逐项对照.md；
7. 06-待办与验收范围.md。

入口是 v1-result/docs/README.md。这组文件解释当前已实现什么、
证据覆盖到哪里、哪些仍是计划。

### 7.2 一个完整案例

想从一个具体成功案例入手时，只读：

- v1-result/20260828-real7b-pell-success/README.md；
- 然后按其中的 01-problem-and-curriculum.md、
  02-system-and-training.md、03-results-and-lessons.md、
  04-reproduce.md。

该案例记录的是 2026-08-28 的 REAL-Prover 7B 课程/证明实验，
不是“未训练基座直接得到”的结果；应把它当作已封存案例，不外推成普遍
benchmark 结论。

案例目录中的 proofs/、inputs/、code/、evidence/、weights/、scripts/
分别保存证明、输入、实际代码、原始证据、参数元信息和复现脚本；需要核查
案例时按 README 给出的顺序进入，不要把这些目录当成另一条理论主线。

### 7.3 证据、测试和动手复现

- v1-result/evidence/current/README.md：按实验查看原始证据；
- v1-result/reproduction/README.md：复现总入口；
- v1-result/reproduction/00-环境与代码.md 至 04-结果检查与故障处理.md：
  环境、单题 TTT、跨题继承、多题并发和故障处理；
- v1-result/reproduction/code/：可浏览源码；
- v1-result/source/current/README.md：源码包校验、解压和扩展训练恢复；
- tests/：仓库级小测试；
- out/、.value-head-test-out/：运行生成的日志或测试产物，只用于排查，
  不作为学习材料；
- v1-result/docs/current/06-待办与验收范围.md：结果边界和未完成项。

建议先复现离线检查，再尝试 GPU；不要把历史日志、证据文件或旧 checkpoint
当成自己的新实验结果。

## 8. V2：元编程、数学驱动和工具使用涌现

### 8.1 V2 理论主线

先读：

- explain/reap-mcts-lean-v2/README.md；
- explain/reap-mcts-lean-v2/00-overview.md；
- 01-eff-channel.md；
- 02-action-space.md；
- 03-tower.md；
- 04-training-and-metrics.md。

然后按主题选择：

- 元编程和 Lean 作为智能体语言：
  explain/5-alpha-as-Lean-coding-agent.md、
  explain/6-meta-programming.md；
- 数学驱动的 agentic 技能：
  explain/8-v2-math-drive.md；
- 长链上下文：
  explain/9-context-management.md；
- 多轮工具调用：
  explain/reap-mcts-lean-v2/多轮tool-call/；
- 涌现工具使用：
  explain/reap-mcts-lean-v2/emergent-tool-use/；
- miner 的定义和实现切片：
  explain/reap-mcts-lean-v2/emergent-tool-use/what-is-the-miner/；
- agentic 视角的 rollout、LLM agent 对比和 miner 变体：
  explain/reap-mcts-lean-v2/agentic-perspective/；
- CPU 侧多轮架构计划：
  explain/reap-mcts-lean-v2/lean-cpu-多轮工具调用/。

当前 V2 仍以数学问题和可验证终局为主，no-miner/部分 T1-T5 是进行中的
设计或代码切片，阅读时不要把“计划”误写成“已完成能力”。

### 8.2 V2 代码位置

推荐唯一主入口：

- new-v2-gather-source-code/README.md；
- new-v2-gather-source-code/reap-mcts-lean-v2-code-1/README.md；
- 其 lean-v2/v2/：Eff.lean、MetaActions.lean、Tower.lean；
- 其 v2/：eff_registry.py、gate_lean.py、mcts_loop.py、
  policy_client.py、tower.py、runner.py、mine.py 和 smoke tests。

配套：

- lean-v2/：独立 Lean 项目，可用于编译和最小模块验证；
- new-v2-gather-source-code/reap-mcts-lean-v2-code-1/ 的 README 已给出
  smoke 命令和代码—spec 对照。

## 9. 当前纲要没有明确列出的补充部分

为了让路线完整，增加以下四个小部分：

### 9.1 数据与课程

没有数据清洗、课程难度、teacher/student 数据流，就无法理解 V1 如何训练、
V2 为什么以数学为主线。入口见第 6.2 节。

### 9.2 版本、经验继承和并发

这是从“单题搜索”走向“持续学习”必须掌握的状态模型：

- 理论计划：plan/08-post-training.md、plan/09-test-time-training.md；
- 实证主线：v1-result/docs/current/02-跨题经验与中央学习.md、
  03-并发调度与模型同步.md；
- 经验设计教学：v1-result/prompt_for_agent/教学/03-跨题经验版本与并发.md、
  05-跨题经验的设计与代码实现.md；
- 状态和恢复：environment-spec/05-state-and-reuse-layout.md、
  06-instance-lifecycle-log.md。

### 9.3 评估、复现和证据边界

入口见 plan/06-evaluation.md、v1-spec/06-eval.md、
v1-result/evidence/current/README.md 和
v1-result/reproduction/04-结果检查与故障处理.md。任何“成功”都应同时说明
命题、模型版本、预算、Lean 独立检查和证据位置。

### 9.4 局限、安全和运维

- 理论局限：explain/3-harness.md、explain/4-alpha-proof与智能体训练是否兼容.md；
- 已知未完成项：v1-result/docs/current/06-待办与验收范围.md；
- 仓库和环境安全：根目录 README.md、tools/scan-secrets.sh；
- 容器边界：docker/README.md。

## 10. 新学习者的推荐阅读顺序

这是最短而不跳关键概念的主线；括号内是该阶段的目标。

1. lecture/1-road-map.md → README.md（知道仓库在解决什么问题）。
2. explain/1-价值头的作用.md → explain/10-mcts-usage-and-alternatives.md
   （理解 value、PUCT、MCTS）。
3. discussion/alphaproof-value-head/README.md →
   07_AlphaProof从零到完整机制_搜索_Value与TTT.md
   （建立 AlphaProof 的完整心智模型；AlphaGo 历史暂跳过）。
4. explain/2-hard-problem如何进步.md →
   explain/3-harness.md →
   explain/4-alpha-proof与智能体训练是否兼容.md
   （理解难题、harness 和训练兼容性）。
5. plan/00-index.md → plan/01-motivation.md →
   plan/02-architecture.md → plan/03-environment-mcts.md
   （先掌握 V1 的系统边界）。
6. explain/reap-mcts-lean-v1/README.md → 00-overview.md →
   01-architecture.md → 02-rollout-sink.md →
   03-batch-solver.md → 04-rttt-hook.md → 05-quality-gates.md
   （把概念对应到 Lean/Reap 组件）。
7. plan/04-data.md → plan/05-rl-param-update.md →
   plan/06-evaluation.md → plan/07-roadmap.md
   （理解数据、训练和验收）。
8. v1-result/prompt_for_agent/教学/01-整体流程与CPU-GPU分工.md →
   new-v1-gather-source-code-cpu/README.md →
   reap-upstream/ 的四个关键 Lean 文件
   （理解 CPU/GPU 分工并看最小源码）。
9. app/VALUE_HEAD.md → app/value_head.py →
   app/train_value_head.py（动手理解 value head）。
10. environment-spec/00-index.md → 01-parity-theory.md →
    02-toolchain.md → 03-concretization.md →
    docker/README.md（准备可运行环境）。
11. v1-result/docs/README.md →
    v1-result/docs/current/00-总设计与阅读地图.md →
    01-题内在线TTT.md → 02-跨题经验与中央学习.md →
    03-并发调度与模型同步.md → 04-实际实例与验收结果.md
    （阅读已经发生的实证）。
12. v1-result/reproduction/README.md → 00-环境与代码.md →
    01-单题TTT.md → 02-跨题经验复用.md →
    03-多题并发.md → 04-结果检查与故障处理.md
    （按一条流程复现）。
13. explain/reap-mcts-lean-v2/README.md → 00-04 →
    多轮tool-call/ → emergent-tool-use/ →
    new-v2-gather-source-code/reap-mcts-lean-v2-code-1/
    （最后进入 V2）。
14. 最后读 plan/08-post-training.md →
    plan/09-test-time-training.md →
    plan/10-recursive-self-improvement.md →
    plan/11-curriculum-traditional-llm.md →
    plan/12-two-model-architecture.md
    （理解未来路线，而不是把计划当成现状）。

如果只想快速理解核心思想，读到第 7 步即可；如果要复现实验，继续到第
12 步；如果要研究 V2，再读第 13—14 步。

## 11. 重复内容的统一取舍

以下规则用于避免重复阅读。除非要审计版本差异，否则只读“保留入口”。

| 重复/相近内容 | 保留入口 | 其余文件如何处理 |
|---|---|---|
| explain/ 与 explain/explain/ 下的 1、2、3、4、9 | 外层 explain/*.md | explain/explain/ 视为旧归档；5—8 即使有文字差异，也只在查修订时读 |
| new-deep-explain-1/1-cpu+gpu.md 与 3-another-version-of-1.md | 1-cpu+gpu.md | 3-another-version-of-1.md 选读 |
| new-deep-explain-v2/3-value-head.md 与 4-3value-head.md | 3-value-head.md | 4-3value-head.md 只补充三值方案 |
| plan/01-07 与 v1-spec/00-07 | 学习读 plan/ | 实现查 v1-spec/，不要两套逐篇通读 |
| 根目录 reap-mcts-lean-v2-code-1/ 与 new-v2-gather-source-code/reap-mcts-lean-v2-code-1/ | 后者 | 两份代码逐文件相同，根目录副本跳过 |
| app/v1_run.py、app/v1_sink.py 与 new-v1-gather-source-code-cpu/python-driver/ 中的同名文件 | new-v1-gather-source-code-cpu/python-driver/ | 两个薄编排文件内容相同；app/ 只保留作便捷入口 |
| lean-v1/Reap/Training/ 与 new-v1-gather-source-code-cpu/reap-training/ | new-v1-gather-source-code-cpu/reap-training/ | 对应训练扩展文件相同；lean-v1/ 只作最小编译项目 |
| v1-result/docs/current/ 与 v1-result/docs/ 历史 01—10 | docs/current/ | 历史文件只用于审计旧结论 |
| v1-result/reproduction/ 与 v1-result/source/current/ | reproduction/ | 先按步骤复现；源码包校验时再看 source/current/ |
| v1-result/docs/current/、v1-result/README.md、prompt_for_agent/教学/ | docs/current/ | README 做总览，教学目录只在需要更通俗解释时读 |
| AlphaProof 长讲义、explain/1-价值头、app/VALUE_HEAD.md | 长讲义 + app/VALUE_HEAD.md | explain/1 作为理论索引，不必重复读全文 |

## 12. 阅读时必须区分的状态

- **理论/计划**：plan/、explain/、大部分 v1-spec/；
- **代码快照**：new-v1-gather-source-code-cpu/、new-v2-gather-source-code/、
  v1-result/reproduction/code/；
- **实证和证据**：v1-result/docs/current/、v1-result/evidence/current/；
- **历史或重复归档**：explain/explain/、v1-result/docs/ 历史文件、
  根目录 V2 代码副本。

不要用计划中的“应当实现”、案例中的一次成功或旧日志中的路径，推断整个
仓库已经具备同等能力。特别是：

1. AlphaGo 历史仍待后续撰写；
2. V2 的 miner、部分多轮工具调用和 T1—T5 仍有进行中的设计；
3. 目标变体课程和语义经验检索尚未作为稳定主线完成；
4. 活跃搜索树的无损恢复、长期吞吐和所有 GPU 容器路径仍有边界；
5. v1-result/ 的结果必须以具体实验日期、冻结代码、模型版本和独立 Lean
   验收为准。
