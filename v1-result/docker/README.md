# CPU/GPU镜像的构建与使用

先按[当前源码指南](../source/current/README.md)校验并解压`source/current`到新目录。本文构建命令均在**解压后的源码根目录**执行；只拿外层`docker/`目录不能构建。

## 1. 已验证范围

- **CPU镜像。** 预编译Lean 4.28.0-rc1、Reap/Mathlib、题内TTT及独立证明复验已实际运行；后续采集镜像增加了固定发布版本的证明采集入口。
- **GPU运行时镜像。** 已实际本地构建，并通过断网容器机制检查；不含7B权重，模型在运行时只读挂载。
- **AMD容器运行。** 尚未完成。本次实例的Docker服务不可连接，三个候选服务套接字不存在，未安装Podman/Buildah，也没有所需的CAP_SYS_ADMIN能力。
- **已有7B实验。** 在远端宿主隔离环境运行，使用Torch2.11；镜像中的Torch2.10仍需实际AMD兼容验收。

本地GPU镜像为`localhost/reap-gpu:runtime-20260828`，ID `fc5687143dd7fd6ca276e45ed1d420fd3b611fa3299c42245216f94326c2dba3`，digest `sha256:ed6b3a035fcded94ce2c48653120d04aeb7bb7b57e9b49782ad3edd37943255d`。展开大小**29,817,042,215字节（约29.817GB）**；镜像、基础层、wheelhouse及7B权重不在交付包内，未打包、未上传镜像。

实际构建退出0，26.680秒（不含基础层拉取）。随后断网、只读容器检查退出0：运行用户UID10001、30份源码哈希、22个依赖包核对通过；3个玩具后端会话、2次终态快照复用释放及补位检查通过。容器Torch为`2.10.0+rocm7.2.4.git3d3aa833`、HIP`7.2.53211`；本地`cuda_available=false`，没有加载或训练真实模型。最小原始证据导航见[当前证据目录](../evidence/current/README.md)。

## 2. CPU镜像与Python控制代码

现有已验收的固定版本采集镜像：`localhost/reap-cpu:verified-collector-20260828`，镜像ID以原始容器检查记录为准。它在已验CPU交付镜像上依次加入选择价值刷新和固定版本采集扩展。

源码中的配方依次为：

1. `containers/cpu/Containerfile`：CPU基础，使用预编译Lean，本体不从源码编译。
2. `containers/cpu/Containerfile.selection-value-refresh`：显式增量，选择价值刷新默认关闭。
3. `containers/cpu/Containerfile.verified-collector`：固定发布版本采集的增量。

两个增量配方严格检查前一镜像的源码哈希。构建前须按源码说明准备匹配的基础镜像。历史A构建与命令见[原A验收](../docs/08-CPU交付镜像验收.md)。最新Python控制代码由单独固定的只读源码目录挂载，不能把旧镜像内Python当作当前版本。搜索时使用已绑定的GPU端点；独立Lean证明复验必须`--network=none`，证据和输出使用新命名卷。

题内TTT用`cpu_runtime.online_ttt`/`online_batch`；固定发布采集用`verified_collector`。`replica_collector`为每题固定使用哪一个独立推理服务。结果未知时保留原路由，不自动重试或迁移。每个服务有独立模型，服务内保护可变状态。完整入口见[当前复现指南](../source/current/README.md)。

### 从包内源码构建新的CPU采集镜像

交付包不含CPU镜像层。最底层另需取得配方固定的预编译Lean镜像`ghcr.io/wufuju2023-cell/reap-lean@sha256:6be053f9c1395890ca28385da56888f0735f785e6e56f54a8b96692cace8e353`，以及`ubuntu:24.04`。CPU基础构建还访问apt、固定Reap提交`0090d73c5f739e4d74000e053b00fd0148ff46aa`和Mathlib `v4.28.0-rc1`缓存；这些步骤需要联网；重新构建可能产生不同的镜像ID。若固定Lean镜像不可读取或网络/缓存失败，应保留失败并停止，不能改用另一个Lean版本。此链不需要7B权重。

以下命令以新本地tag为例，按每一步实际生成的镜像ID绑定下一步；不要覆盖历史tag。构建工具必须支持配方所用Dockerfile语法。完整命令需要在已校验解压的源码根执行，最后保留自己的镜像检查结果和构建退出记录：

```bash
set -eu
base_tag=localhost/reap-cpu:new-base
refresh_tag=localhost/reap-cpu:new-refresh
collector_tag=localhost/reap-cpu:new-collector
for tag in "$base_tag" "$refresh_tag" "$collector_tag"; do
  status=0
  podman image exists "$tag" || status=$?
  case "$status" in
    0) echo "Tag already exists; choose new names: $tag" >&2; exit 1 ;;
    1) ;;
    *) echo "Image lookup failed; do not build: $tag" >&2; exit "$status" ;;
  esac
done
podman build --jobs=1 --retry=0 \
  -f containers/cpu/Containerfile -t "$base_tag" .
base_id=$(podman image inspect --format '{{.Id}}' "$base_tag")
podman build --network=none --pull=never --jobs=1 --retry=0 \
  --build-arg "CPU_BASE_IMAGE=$base_id" \
  -f containers/cpu/Containerfile.selection-value-refresh -t "$refresh_tag" .
refresh_id=$(podman image inspect --format '{{.Id}}' "$refresh_tag")
podman build --network=none --pull=never --jobs=1 --retry=0 \
  --build-arg "CPU_BASE_IMAGE=$refresh_id" \
  -f containers/cpu/Containerfile.verified-collector -t "$collector_tag" .
podman image inspect "$collector_tag"
```

包内必要路径均已列入源码清单：基础配方消费`reap-overlay/`、`runtime/`及`patches/0001`、`0002`、`0003`；第一个增量消费`patches/0004-selection-value-refresh.patch`和`selection-value-refresh-overlay/`；第二个增量消费`verified-collector-overlay/Reap/VerifiedCollector.lean`（均位于`containers/cpu/`）。`0004`含精确混合换行上下文，禁止运行dos2unix/sed统一换行。配方中的前置SHA检查失败时不能删门或直接跳补丁，应核对新基底和解包字节。

以上提供完整构建入口，本轮没有重新执行这三段构建。当前已验采集镜像的历史ID为`d2070a64912f1a7c66ec9e4bb92e66004e8adaa58887e7be333f0bf8e3d5f3b7`；新构建须使用实际新ID并重新完成CPU环境/模块/闭合命题预检，不能冒用旧ID或旧预检收据。便携入口的显式新image重绑定会重新生成环境身份，并在两题CPU预检通过后才创建调度记录；以[便携复现说明](../source/current/README.md)的最终参数为准。

## 3. 构建无权重GPU运行时镜像

当前源码包含：

- `containers/gpu/Containerfile.runtime`：独立无权重配方。
- `containers/gpu/install_runtime.sh`：离线安装22个哈希固定wheel，不替换基础Torch。
- `tools/amd_jupyter/prepare_gpu_runtime_context.py`：严格白名单目录，不读取模型，不复制本地杂项。
- `containers/gpu/README.runtime.md`：解压后的详细操作和验收边界。

需要Linux x86_64、Python3.12、可用的Podman或Docker服务。基础ROCm镜像压缩层约10.39GB、展开约29.54GB，应预留足够磁盘。只有依赖wheel和基础镜像需要准备，**不下载7B到本机**。

以下示例目录必须是新目录。已有wheelhouse可以直接复用，并跳过下载命令；生成构建目录前会重新按锁验证。

```bash
set -eu
wheelhouse=/absolute/new-gpu-wheels
context=/absolute/new-gpu-runtime-context
mkdir "$wheelhouse"
python3.12 -m pip --isolated download --only-binary=:all: \
  --no-deps --require-hashes \
  -r containers/gpu/requirements-gpu-hashed.lock --dest "$wheelhouse"
python3.12 -B -m tools.amd_jupyter.prepare_gpu_runtime_context \
  --wheelhouse "$wheelhouse" --output "$context"
podman pull docker.io/rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e
cd "$context"
podman build --network=none --pull=never --retry=0 --jobs=1 \
  -f containers/gpu/Containerfile.runtime -t localhost/reap-gpu:runtime-local .
```

实际安装只用`--no-index --require-hashes --no-deps`及独立`/opt/reap-python`覆盖层；构建禁网且不拉取额外镜像。新构建的ID可能不同，必须保存自己的image inspect、引擎退出码与源码清单；不要给新产物套用上述历史ID。

## 4. 在支持GPU的AMD容器环境中运行

以下为**尚待真实GPU容器验收的操作模板**，不适用于本次没有可用容器服务的实例。先确认容器服务和平台GPU设备映射均获支持；只安装CLI不够。不使用`--privileged`、host IPC或安全限制绕过。

模型必须为已核验的固定`REAL-Prover` revision `fe76f68d9a88f342cb7b546307c20292fea9cced`。先核验既有模型文件，再只读挂载。Podman的`keep-groups`需要其运行时支持；设备/组权限不满足时停止，不绕过。

```bash
set -eu
model=/existing/verified/REAL-Prover
out_volume=reap-gpu-new-out
podman info
if podman volume exists "$out_volume"; then
  echo "Output volume already exists; choose a new name" >&2
  exit 1
fi
podman volume create "$out_volume"
podman run --name reap-gpu-new --pull=never --network=bridge \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --ipc=private --shm-size=8g -p 127.0.0.1:8760:8760 \
  --mount "type=bind,source=$model,destination=/opt/models/REAL-Prover,ro" \
  --mount "type=volume,source=$out_volume,destination=/workspace/out" \
  localhost/reap-gpu:runtime-local \
  --backend real-search --gamma 0.99 --model-path /opt/models/REAL-Prover \
  --snapshot-root /workspace/out/snapshots --max-resident-sessions 2
```

该例是题内搜索TTT合同。mixed/verified采集需要显式对应的server profile、只读证明数据集挂载及可写独立learner仓库，参数必须与来源发布合同完全一致；不能通过改gamma或KL门限强行兼容旧发布。输出卷应可供UID10001写入，使用新卷并在首次运行核查权限。默认模型离线加载；服务端口只映射本机，没有额外API认证，跨主机连接应走已授权的安全通道。

`/health`可用于就绪检查，但不足以证明真实GPU计算正确。后续还需HIP设备、实际7B推理/训练、数值与隔离、完整恢复、外部挂载持久化和独立Lean验证。已有宿主Torch2.11实验与容器兼容检查分别记录。

## 5. 历史配方与失败证据保留

本目录的`cpu.Dockerfile`、`train.Dockerfile`、`prepare_context.py`、`build.sh`、`run-gpu.sh`、`run-cpu.sh`、题单和封装测试是**旧V1冻结交付代码**，未修改。旧GPU配方会把模型加入镜像，当前使用上述运行时挂载方式；不要误执行旧配方默认在线模型下载阶段。历史源码仍可按[source说明](../source/README.md)校验。

本次保留首次Buildah解析Dockerfile heredoc失败（exit125），以及初次容器检查脚本snapshot_root类型错误（exit1）；修复只在新配方/新检查脚本，旧证据不覆盖。基础镜像pull外壳退出码转义丢失：outer exit1、实际pull退出码未保存；独立inspect确认固定base完整存在，**不将其改写为pull exit0**。最终build和第二次容器检查均有真实exit0。

后续复现仍需保存每次构建和运行的实际结果。平台能力不足或更新结果未知时，保留现场并核对原记录后再处理。
