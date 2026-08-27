# 源码包：先校验，再解包使用

从这里开始。这个目录交付当时固定的构建源码快照；它不包含镜像、模型权重、已安装依赖或运行环境。源码版本保持原样，外层指南更新不会改写历史源码身份。

使用 Edge/OpenCLI 宿主桥时，校验并解包后另按 [transport-fix 补丁说明](transport-fix/README.md) 覆盖两个桥接文件；原归档与清单保持不变，该补丁不代表 A/B 镜像已经更新。

## 1. 四个文件各做什么

| 文件 | 用途与读取顺序 |
|---|---|
| `README.md` | 先读本说明，确认目录、所需环境及成功判据 |
| `source-manifest.json` | 再看来源清单：77 个文件的相对路径、大小、SHA256，以及整包摘要和构建输入检查结果 |
| `source-snapshot.tar.gz` | 待校验的压缩源码；内部保留仓库根布局，不能当作 `docker load` 输入 |
| `verify_source.py` | Python 标准库工具；默认只校验，指定 `--extract` 后才向新的目录写入 |

固定包大小 **114,830 字节**，解压内容 **419,428 字节**，整包 SHA256：

```text
69082304c896cd5995619d9c65a8659d452db13f1cfd08f48a00d7822ab141ce
```

清单与上面的摘要应从可信交付来源取得。文件哈希检测损坏和版本差异，不替代来源信任。

## 2. 准备运行环境

校验工具只需要 Python 3 标准库；Windows 可把下文 `python3` 换为 `python`。后续 Docker 构建脚本需要 Linux/WSL、Bash、Python 3，以及已安装且可用的 Podman 或 Docker。

在交付目录根部执行，也就是同时能看到 `source/`、`docker/` 的位置：

```bash
cd /absolute/path/to/v1-result
python3 source/verify_source.py
```

成功时退出码为 0，并输出 `verified=true`、`file_count=77`、上述摘要及 `extracted_to=null`。默认不会创建解包目录，也不会构建、联网或启动服务。

工具会检查整包大小/摘要、清单数量、每个普通文件的路径/大小/摘要和成员名称集合；拒绝缺失、额外、重复文件，以及路径穿越、符号链接、硬链接、稀疏文件和跨平台路径冲突。清单和归档均通过后，才允许进入解包步骤。

## 3. 安全解包到新目录

选择一个**不存在**的绝对目标目录；其母目录必须已存在。不要先 `mkdir` 这个目标，也不要选用已有项目、符号链接或其他 Git 仓库中的目录。

```bash
python3 source/verify_source.py --extract /absolute/path/to/new-reap-source
```

成功时再次输出 `verified=true`，`extracted_to` 指向新目录。工具先在内存中核验整个源码包，再创建新目录和普通文件；不覆盖已有目录/文件，不通过链接写到外面。

校验失败返回非零，不进行解包。如果写磁盘阶段发生权限、空间或其他 I/O 错误，可能留下本次新建的部分目录；工具不自动清理或覆盖，请检查错误并选择新的目标重试。不要使用“忽略错误继续构建”。

## 4. 进入解包目录，做不联网检查

```bash
cd /absolute/path/to/new-reap-source
python3 v1-result/docker/prepare_context.py --help
python3 v1-result/docker/test_delivery_context.py
for script in v1-result/docker/build.sh v1-result/docker/run-cpu.sh v1-result/docker/run-gpu.sh; do
  bash -n "$script"
done
```

第一条确认交付 wrapper 能找到原 `tools/amd_jupyter` helper；第二条使用微型合成模型测试受控 context，不读取实际 7B 权重；最后逐个检查 shell 语法。独立解包后的 14 项 context 测试已在 WSL 全部通过。

这些检查成功意味着源码布局、导入与封装逻辑可用；GPU 库是否可运行、Dockerfile 是否构建成功仍需分别验证。新校验工具另经过损坏、路径穿越、已有目录及符号链接等 8 项小文件测试。

## 5. 接下来构建和运行

先阅读外层 [Docker 使用流程](../docker/README.md)，然后始终在上面**解包得到的仓库根目录**执行构建命令：

```bash
bash v1-result/docker/build.sh cpu localhost/reap-cpu:v1-delivery
```

这条命令会真正开始 A 镜像构建，可能拉取固定基础镜像、APT 包及 Lean/Reap/Mathlib 相关制品。Lean 使用预编译版本；本次交付没有替你执行新的完整构建。

B 需要合法远端 builder、已验证的固定 REAL-Prover 模型目录和官方 manifest。它们不在本源码包里；按 Docker 指南的 `gpu-existing` 步骤准备，权重留在远端。当前受限 DSW 没有可用容器服务，不能直接套用普通 AMD 主机运行模板。

## 6. 目录、版本和缺失项

归档包含 `containers/cpu`、`containers/gpu`、`containers/versions.lock`、`cpu_runtime`、`gpu_runtime`、必要工具、根 `.dockerignore` 和 `v1-result/docker` 构建文件。不含权重、凭据、浏览器状态、`.git`、私有 clone、pycache、历史日志/checks 或递归 source 包。

外层 `docker/` 的代码与归档内对应代码逐文件同哈希；外层 README 是新版使用说明，允许与归档中的旧说明不同。归档里的 README 可能链接到未打包的实验 docs、checks 或 source 说明；这些链接供完整历史交付使用，不是构建依赖。当前请以本目录和外层 Docker 指南指导操作。

在独立无 Git 目录中，原 helper 的 `git_state` 返回 `head=null`、`dirty=null`；不会伪造“干净 Git checkout”。如果解压到其他 Git 仓库内部，Git 可能发现上级仓库，不能把该 HEAD 当作本包来源。源码身份以 `source-manifest.json` 为准。

成功交付镜像还需：新 A/B build、合法 AMD 容器运行、真实 Lean 同树 TTT/隔离与证据复核，以及另行确认的发布操作。校验或解包成功不触发这些动作；本轮不自动 GitHub 同步，也不修改远端旧内容。
