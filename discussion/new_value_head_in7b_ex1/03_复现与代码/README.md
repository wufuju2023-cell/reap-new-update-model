# 复现与代码导航

## 三种复现深度

1. **只读审核**：阅读成功案例，运行其 `verify_bundle.ps1` 和 `verify_evidence.ps1`。不需要 GPU、大模型或网络。
2. **价值头重训**：另行取得同 schema 的冻结特征，运行随案例提供的 `train_value_head.py`；可复现方法，但数据不同不会得到逐字相同的 checkpoint。
3. **端到端复现**：需要兼容的 REAL-Prover 7B、trained head、GPU 服务和固定 Lean 镜像，按 `01_完整复现手册.md` 运行真实 MCTS/TTT/验收。

## 文件选择

- `01_完整复现手册.md`：总体流程，适合搭建运行环境。
- `02_代码与工件说明.md`：原工作区实现、合同和 SHA，适合审计本次实验来源。
- `verify_artifacts.ps1`：只在原项目工作区有意义，对外展示不依赖。
- 成功案例 `code/`：复制了本案例实际涉及、体积较小的关键实现。
- 成功案例 `results/evidence/`：复制了 proof、tree、observer、验收和发布原始回执。

## 大工件边界

本目录不分发 7B 权重、训练特征或大型 snapshot，也不列外部读者无法访问的本机下载路径。端到端复现者需从其有权使用的来源取得兼容工件，并为自己的工件重新生成身份与 SHA 记录。
