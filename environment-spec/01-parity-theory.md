# 01. 概念与云计算理论

## 1.0 问题定义

传统运维下，本地开发机与云端生产机的环境靠"人为同步"（虚拟机镜像快照 → 手动改配置 → 手工装包）。
随着时间推移，两端累积的差异被称作 **configuration drift（配置漂移）**：

$$\mathrm{drift}(t) = \sum_{k} \mathbb{1}\{\mathrm{state}^{\mathrm{local}}_k(t) \neq \mathrm{state}^{\mathrm{cloud}}_k(t)\}$$

漂移的直接后果：本地 `runs`、云端 `fails`（"works on my machine"），
且需要人工对根因——这在数学/ML 长任务上成本极高（一次漂移可能等价于一条训练日）。

## 1.1 解法一：Immutable Infrastructure（不可变基础设施）

原则：**绝不原地修改环境；修改 = 重建**。环境从"有状态的 pet server"
变成"声明式的制品（artifact）"。所有微差（包版本、内核参数、挂载点、缓存目录）
必须进入制品定义；线上只允许 `replace`。

表现形式：环境 = 镜像（OCI image）+ 配置（env / YAML / IaC），
不可变、可审计；允许迁移的是 **workload 状态**（数据集、checkpoint），不是环境本身。

## 1.2 解法二：容器与 OCI 标准

- 运行时语义：cgroup + namespace + overlayfs 在**任意宿主机**上提供一致的进程视图。
- 制品语义（[OCI-Image-Spec](https://github.com/opencontainers/image-spec)）：
  镜像 = manifest（分层文件系统 + 元数据），manifest 的 SHA-256 摘要即**全局唯一标识**：
  $$\mathrm{imageID} = \mathrm{SHA256}(\mathrm{manifest})$$
- 一致性证明：两端 `docker pull image@sha256:<digest>` 的是同一字节流；
  digest 匹配 == 语义等价（除宿主内核/驱动外）。
- 封装边界：一切可能漂移的（Python 解释器、依赖树、$PATH、库缓存、目录布局）**全部进镜像**；
  唯一留在镜像外的只有宿主机内核与设备驱动（属 host 层，容器无法伪装）。

## 1.3 解法三：配置即代码（12-Factor 原则）

- 代码、依赖、配置三种输入可由 Git 同一版本号重建（确定性构建）。
- 依赖必须锁定：requirements.txt 带 `===` 精确版本（或 `pip freeze > requirements.lock`）；
  基础镜像必须以 **digest 锚定**（base: `rocm/pytorch@sha256:xxxx`），不能用可变的 tag。
- 所有"环境行为"（如 `HF_HOME`、`MIOPEN_*`、`TORCH_CUDA_ARCH`）写进容器 env，
  禁止依赖外面的人肉环境。

## 1.4 分发与消费模型（GitOps / IaC）

- **构建一次，两端消费**：CI（GitHub Actions 共用 workflow）负责 `docker build & push`；
  云端与本地只是同一 registry 的两个消费者：
  ```
  [[pull 凭证]] → git → CI → registry（ACR 中国区 / GHCR）→ 云端云实例 & 本地 WSL
  ```
- 云端宿主机配置由 **Terraform / cloud-init** 声明；
  本地开发容器由 **devcontainer.json / Devbox/Nix** 声明；
  两者均由同一 git 仓库生成 —— **parity 从"约定"升级为"CI 红线"**（两端必须拉同一 digest 才能过 gate）。

## 1.5 理论结论

> 一句话：**容器是复用单元，registry 是分发机制，digest 是 parity 契约，CI 是执行者。**
> 本地与云端不再是"配置两次"，而是"消费一次"——这正对应云原生对 dev/prod parity 的标准答案
> （不可变基础设施 + OCI + GitOps），也是本项目的环境基线。
