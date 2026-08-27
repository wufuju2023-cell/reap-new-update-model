# Docker 使用流程：从源码到 A/B 验收

先完成 [源码校验与解包](../source/README.md)。下面的命令均在**解包得到的仓库根目录**执行；只拿外层 `docker/` 无法构建，因为 Dockerfile 和 wrapper 还要读取 `containers/`、runtime 与原 helper。

新A交付产物与本地运行验收已通过，ID为`623ae445c3cb8c843afb6c85a1db0dfa25f17e72ae95afaf23a49ef8844f0fce`；源码/UID、无网络双session mock及E原proof重编均通过，见[08验收报告](../docs/08-CPU交付镜像验收.md)。原Podman退出码因外层脚本转义错误未保留，保留外层exit1与实际COMMIT/inspect证据。原GPU实验仍使用增量A `df23cd6a9b5b6df30fa003285fadcdc1e5d60919eb43b823b0731c17bf135646`；新A未重复GPU实验。B实际build/run与发布仍未完成。

**远端启动失败原因：** 当时实例出现只读内核配置、`proc`挂载权限错误；后续检查也没有可用Docker服务。CPU镜像已经验收，GPU容器B仍待在受支持环境验证，详细报错与解决条件见[06](../docs/06-容器未完成原因与交付边界.md)。

## 1. 文件用途

| 文件 | 输入、作用与输出 |
|---|---|
| `cpu.Dockerfile` | 从仓库读取固定 Reap、预编译 Lean/Mathlib、3 个补丁、CPU runtime 和两道递推题，构建 A |
| `train.Dockerfile` | 固定 ROCm/PyTorch 基底、22 包哈希、REAL-Prover revision；B 的两个 release 目标默认 `real-search --gamma 0.99` |
| `prepare_context.py` | 读取原 helper、已验证模型、官方 manifest 和 recipe；产生新的 GPU 白名单 context 与哈希清单 |
| `build.sh` | `cpu` 构建 A；`gpu-existing` 先准备受控 context，再构建 B；默认引擎为 Podman |
| `run-gpu.sh` | 在普通获授权 AMD 主机启动 B，私有 IPC、8GB共享内存、仅本机 8760 端口及持久输出卷 |
| `run-cpu.sh` | 读取题单并启动 A 的双 session online batch；保存每题事件、回执、结果和 solutions |
| `batch.example.jsonl` | 两道递推题的题单模板；运行前复制并改成新的唯一 session ID |
| `test_delivery_context.py` | 微型模型的 14 项封装测试；检测模型/清单篡改、重复输出和 recipe 替换，不调用真实 GPU |

外层这八个代码/数据文件与冻结源码归档同哈希；本 README 是新版指导，可与归档内旧说明不同。若要确认版本，查 `source/source-manifest.json` 的 `v1-result/docker/...` 项。

## 2. 先检查工具和源码布局

```bash
python3 v1-result/docker/prepare_context.py --help
python3 v1-result/docker/test_delivery_context.py
podman info
```

前两条应成功，微型测试应结束为 `OK`。`podman info` 要确认容器服务可用；只安装 CLI 不够。若使用 Docker，先执行 `export CONTAINER_ENGINE=docker` 并自行确认 `docker info` 成功。这些检查不会替代平台授权。

## 3. 构建 A：CPU 镜像

```bash
bash v1-result/docker/build.sh cpu localhost/reap-cpu:v1-delivery
```

构建需要网络、磁盘及固定基础镜像/Reap/Mathlib 的可达性；不需要 GPU 或模型权重。Lean 本体使用预编译版本。成功判据是引擎退出 0，镜像可 inspect；请保存新 image ID/digest 和完整构建日志，不覆盖旧实验 tag 的身份。

随后可在无外网容器中运行内置 mock smoke（输出卷名必须新建）：

```bash
podman run --name reap-a-smoke-new --network none -v reap-a-smoke-new-out:/workspace/out \
  --entrypoint reap-cpu-smoke localhost/reap-cpu:v1-delivery
```

两 session solved 说明 CPU 验证链可用；这是 mock 测试，没有 GPU 参数更新。

## 4. 构建 B：仅使用远端既有模型

需要用户授权且具备合法容器 builder 的远端主机。模型权重不要下载本机；以下路径仅是历史示例，先确认实际模型/manifest 存在且属于固定 revision。新 context 路径必须不存在，至少额外准备约 15.25GB 用于复制模型，镜像构建还需更多空间。

```bash
export AUTHORIZED_REMOTE_BUILDER=yes
model_dir=/mnt/workspace/models/REAL-Prover-fe76f68d
manifest=/mnt/workspace/reap-v1-20260827-gpu/model-manifest.json
new_context=/mnt/workspace/replace-with-new-build-context
bash v1-result/docker/build.sh gpu-existing localhost/reap-gpu:v1-delivery \
  "$model_dir" "$manifest" \
  b20526a1d3a08365893fe937dfdec4b1d953b8c63f26cc89e922c962c11f3915 "$new_context"
```

wrapper 先调用原 helper 校验模型官方哈希、safetensors 索引和张量，再复制 GPU 源码/依赖/模型到新 context。随后仅替换其中的 recipe，更新文件/来源摘要，保留原 manifest 及全部模型条目，再复核整个 context。仓库原文件不变；失败时不覆盖或清理原目录。

脚本显式构建 `release-existing`，不会走在线下载模型阶段。成功后保存 context manifest、recipe SHA256、image ID/digest 和构建日志；`all_context_hashes_verified=true` 只说明 context 已验证，仍要等待后续引擎 build 成功。

## 5. 在受支持的 AMD 主机运行 B

此模板面向普通获授权 AMD Linux 主机；设备映射、video 组、命名卷和端口必须被平台支持。B 使用私有 IPC，未使用 `--ipc=host` 或提权/安全策略绕过。

```bash
export AUTHORIZED_GPU_RUNTIME=yes
bash v1-result/docker/run-gpu.sh localhost/reap-gpu:v1-delivery reap-b-new reap-b-new-out
```

命令前台运行。另一个终端请求 `curl --fail http://127.0.0.1:8760/health`，确认 `backend=real-search`、实际 HIP/模型状态。容器或 PID 存在不算成功；模型已烘焙，输出卷保存 snapshots。默认仅监听本机映射端口，没有额外 API 认证，跨机器应使用已授权的安全通道。

当前已检查的 DSW 实例 Docker 无法连接、三个候选 socket 不存在、Podman 未安装；UID=0、seccomp=0 不代表具有 `CAP_SYS_ADMIN`，实际该能力为 false。此前另一个实例安装过 Podman但运行 proc 挂载失败，不能混用两台的状态。DSW 还明确禁止 host IPC，并限制资源组和卷类型；不能直接套用本模板。[官方限制](https://help.aliyun.com/zh/pai/using-docker-in-dsw)

## 6. B ready 后，运行双 session A

复制 `v1-result/docker/batch.example.jsonl` 到新文件，给两行分别设置未用过的 session ID。保持两道 theorem 路径不变。然后执行：

```bash
bash v1-result/docker/run-cpu.sh localhost/reap-cpu:v1-delivery reap-a-new \
  /absolute/path/to/new-batch.jsonl reap-a-new-out http://127.0.0.1:8760
```

URL 必须从 A 主机可达；若经 Edge/OpenCLI bridge，改为已验证的 bridge 本机地址/端口，并在 bridge 白名单加入相同 session ID。不要复用历史实例 ID。默认 concurrency=2、gamma=0.99、最多5次更新；720/900秒只是请求/屏障保护，TTT 没有总时限。

每路查看 `online-result.json`、`result.json`、observer、learn 回执和 `solutions.jsonl`。提前解题可能是 `solved_without_online_update`；完整 TTT 要有实际更新、同树后续版本消费及最终 Lean proof，再独立核对 wire、参数和隔离/恢复。保存容器 inspect、日志、输出卷和 B 快照，不能只看一句 `completed`。

## 7. 当前还缺什么

仍缺B recipe的实际build、合法AMD容器运行及完整复验；宿主Torch2.11与候选B的Torch2.10需要重新做兼容性门禁。A交付镜像已有独立本地验收，不替代上述B项目。镜像发布、SBOM和目标可见性也未完成。

这些环境标记是防误操作提示，不授予权限。AMD 实例启停由用户负责；“AMD 空闲额度保护”和“AMD 启动后接管检查”自动任务已删除，不自动控制 AMD。本次按用户授权将结果包上传GitHub；后续同步、镜像push和远端实例变更须另行确认。遇到权限、未知更新结果或校验失败时停止并报告。
