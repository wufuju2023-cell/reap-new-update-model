# 本地 × 云端环境一致性规范 (Dev/Prod Parity Specification)

**目标**：我们操作的 cloud AMD GPU（Radeon Cloud 实例，ROCm 7.2 / Ubuntu 24.04 / py3.12）
与本地（WSL2，无独立 GPU）共享**同一份环境制品**——`同镜像、同依赖、同入口`，
可复现、可回滚、可审计。

| 文件 | 内容 |
|---|---|
| [01-parity-theory.md](01-parity-theory.md) | 概念与云计算理论（drift / immutable infra / OCI / digest） |
| [02-toolchain.md](02-toolchain.md) | 工具链矩阵与关键命令 |
| [03-concretization.md](03-concretization.md) | 本项目落地：Dockerfile / smoke test / 发布流程 |
| [04-bootstrap-reuse.md](04-bootstrap-reuse.md) | 控制面一次安装任意新实例复用（container-like） |
| [05-state-and-reuse-layout.md](05-state-and-reuse-layout.md) | **整套环境与状态放在哪**：制品/模板/归档三段布局 + 10 分钟续接剧本 |
