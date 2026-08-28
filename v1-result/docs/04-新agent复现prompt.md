> **历史报告｜2026-08-27版本。** 本文保留当时的结果、流程和未完成项。当前成果请看[当前主报告](current/00-总设计与阅读地图.md)，最新范围见[待办](current/06-待办与验收范围.md)。本文原始字节已保存于[历史归档](history/20260827-documents.tar.gz)。

# 给另一位 agent 的独立复现 prompt

把下面分隔线内的全部内容交给 agent，并提供整个 `v1-result` 文件夹。软件、远端资源和网络需要另行准备。命令面向 Linux/Bash；Windows 控制端可使用用户授权的 WSL。所有示例路径、资源名和主机信息均由接收者填写，不是已存在的资源。

---

## 任务与验收

你收到的是 `v1-result` 结果包。先读 `docs/README.md`、`docs/01-实验设计与成功边界.md`、`docs/03-实际TTT结果与证据.md`、`docs/07-多轮TTT记录与成功边界.md`、`docs/09-五题尝试与并发结果.md`、`docs/10-原设计逐项对照与实际流程.md`、`source/README.md`、`docker/README.md`。只使用包内材料及明确获授权的外部资源，不假定原项目、历史镜像、浏览器登录、实例、模型或服务存在。

本次目标：固定 REAL-Prover 7B，在真实 Lean MCTS 的**同一个活跃树、同一 Lean 进程**内，用真实 visits/backup 反馈多轮更新，再继续生成并完成证明。要求至少两个并发、状态隔离的 session；**每个计入成功的 session 都至少两次真实 optimizer update，更新后继续消费新版本，最后获得独立 Lean 复核通过的证明**。GPU 可由单 actor 排队执行，不能把 CPU/session 并发说成 GPU kernel 并发。

最终保留两份独立镜像：A 为 CPU Lean/Reap/MCTS，B 为 AMD GPU 训练/推理并包含固定模型。沿用包中 `real-search` 的 visits 蒸馏、search-backup value/distance、冻结底座与 session 隔离设计，不静默换成 toy、旧分段 ACT/LEARN 或其他训练算法。

session 隔离针对同时解题的可变状态；共享冻结 base 不禁止将已确认的 LoRA/value 作为下一题的初始化，但当前新 session 并未继承上一题 LoRA/value。原文 explain 第9部分的跨题经验仍未完成，不能宣称完整跨题持续学习。原实验用户决定先做五题、延期跨题继承；这是当前实施范围，**本 prompt 不限定接收者必须使用这五题**。接收者可按自己的用户授权选择题目，默认不擅自增加跨题权重传递，题内多轮、同树消费和并发隔离的验收保留。

历史 E 只有一次更新后证明成功；历史 F 有五次更新但未证明成功。两者都**不满足本次“至少两轮更新后最终证明”完整门槛**。新增第1题三次更新、第5题两次更新后证明成功，通信、真实首末参数快照及独立Lean复验已通过。**下文第7/8节默认复现新增01/05题**：从包内证据归档只取原题、只读挂载到A，并用本轮生成的proof独立复核。历史E/F可另选，但须同步更换输入及复核映射。历史成绩不自动赋给新运行；没有未训练对照时不宣称能力提升。

合法提前解出时立即接受，记录“证明成功，但多轮覆盖不足”。不得拒绝有效 tactic、注入预设 proof 或强迫已解题继续训练。`--max-updates 5` 是上限，不保证发生五次更新。TTT 没有总时限，五分钟只是希望；HTTP/checkpoint 超时是故障保护。

## 1. 输入与外部依赖

| 输入/依赖 | 来源 | 用途与边界 |
|---|---|---|
| 源码、Docker recipes、哈希锁、CLI、Lean 题目 | 包内 `source/source-snapshot.tar.gz` | 校验后解包；后续命令从解包根目录运行，无需原仓库 |
| Python 3.12、venv/pip、Bash、coreutils、Git、curl、OpenSSH、Docker 或 Podman | 经用户同意安装的软件 | 构建还需固定基础镜像、Reap/Mathlib 源码和预编译缓存；不自编译 Lean 工具链 |
| AMD Linux GPU、ROCm 驱动、容器设备权限、存储 | 用户授权的远端主机/平台 | 实例启停、计费、平台账号和验证码由用户控制；不建自动启停任务 |
| GHCR、Docker Hub、GitHub、Lean/Mathlib 缓存、PyPI、官方 Hugging Face | 授权网络 | 需要代理、镜像或凭据时停问，不绕过限制 |
| 官方模型元数据及约 15.25 GB 文件 | 固定 HF revision，下文包内 CLI 获取 | 权重仅在远端下载/构建，不在控制机下载或缓存 |
| A→B 通信、源码/证据传输 | 用户授权的 SSH 或同等安全通道 | 下文 SSH 示例需用户提供真实主机与认证，不依赖特定 Edge 配置 |

依赖关系：**源包校验→A 构建；源包+远端官方模型验证→B context→B 构建→GPU 预检；A+B+可审计安全通道→并发真实 TTT→独立复核。**模型目录、context 与镜像层同时占空间，不能只预留一份权重空间。

先确认 CPU/GPU 机器、架构、磁盘、网络、`/dev/kfd`、`/dev/dri` 和合法容器能力。root 不代表有容器权限。**平台不能运行所需 GPU 容器时，暂停询问更换获授权主机，或是否先做宿主 GPU 实验。**宿主方案需另行确认 Python/ROCm/Torch、隔离依赖，标为 host-B；不能算 B 镜像 build/run 完成。不要添加 `--privileged`、`--ipc=host` 等绕过平台限制。问题和重要设计变化及时停问。

## 2. 校验与解包

先在收到的 `v1-result` 目录执行。目标目录尚不存在，父目录已存在，且不要置于其他 Git 仓库中：

```bash
PACKAGE_DIR="$PWD"
python3 source/verify_source.py
SOURCE_DIR=/absolute/new/reap-source
python3 source/verify_source.py --extract "$SOURCE_DIR"
cd "$SOURCE_DIR"
python3 -B -m unittest discover -s v1-result/docker -p test_delivery_context.py
```

冻结包 77 文件，归档 SHA256 为 `69082304c896cd5995619d9c65a8659d452db13f1cfd08f48a00d7822ab141ce`。校验器检查归档、逐文件名称/大小/哈希及路径安全，失败不得继续。A、B 各自验证并解包同一源包。包没有 `.git`，helper 的 Git head/dirty 可为 null，以文件哈希识别代码，不伪造 Git 状态。

新运行的宿主 HTTP bridge 另使用包内 `source/transport-fix/` 补丁，历史归档保留原样。先按 `source/README.md` 核对补丁来源与哈希，再在解包根目录覆盖下面两个文件并执行本地测试；文件缺失或测试失败就停下。`remote_http_job.py` 已在冻结源包中，无需包外下载：

```bash
cp "$PACKAGE_DIR/source/transport-fix/http_bridge.py" tools/amd_jupyter/http_bridge.py
cp "$PACKAGE_DIR/source/transport-fix/test_http_bridge.py" tools/amd_jupyter/test_http_bridge.py
python3 -B tools/amd_jupyter/test_http_bridge.py
sha256sum tools/amd_jupyter/http_bridge.py tools/amd_jupyter/test_http_bridge.py
```

这一步产生“冻结源码+显式宿主传输补丁”的新运行身份，记录覆盖前后哈希。它不改变历史实验身份，也不表示 A/B 镜像已包含或实测该补丁；实际启动 bridge 的主机必须执行覆盖与测试。

每台机器新开 shell 都先 `cd "$SOURCE_DIR"`，重设该机器的路径变量。只有 `verify_source.py` 在外层包中；后续 CLI 均在解包根目录。外层 Docker README 是更新说明，其文本可以不同于归档内 README，运行代码应对应。

## 3. 构建 CPU 镜像 A

在获授权 CPU 构建机执行，不依赖预先存在的实验镜像：

```bash
export CONTAINER_ENGINE=podman  # 使用 Docker 时改为 docker
A_IMAGE=localhost/reap-cpu:my-multiround-v1
bash v1-result/docker/build.sh cpu "$A_IMAGE"
"$CONTAINER_ENGINE" image inspect "$A_IMAGE"
"$CONTAINER_ENGINE" run --rm --network none --entrypoint python3 "$A_IMAGE" \
  -m cpu_runtime.online_batch --help
```

保存构建日志、退出码、镜像 ID。recipe 固定预编译 Lean 4.28.0-rc1 基础镜像，应用三个 Reap patch、使用 Mathlib 缓存并编译 Training overlay；题目位于 `/opt/reap-runtime/real-fixtures/`。GHCR 需认证时向用户说明 registry 凭据要求，不能使用 GitHub 网页密码冒充 token。A 编译成功不代表 B 或 TTT 成功。

## 4. 仅在远端下载、验证模型

在授权远端下载/构建机的解包根目录执行。`MODEL_WORK` 是远端持久存储中的全新目录。先取元数据，不下载权重：

```bash
MODEL_WORK=/absolute/remote/new-model-work
mkdir "$MODEL_WORK"
MODEL_REPO=FrenzyMath/REAL-Prover
MODEL_REV=fe76f68d9a88f342cb7b546307c20292fea9cced
MODEL_MANIFEST="$MODEL_WORK/model-manifest.json"
MODEL_SHA=b20526a1d3a08365893fe937dfdec4b1d953b8c63f26cc89e922c962c11f3915
python3 containers/gpu/download_model.py --repo "$MODEL_REPO" --revision "$MODEL_REV" \
  --manifest-only --save-manifest "$MODEL_MANIFEST"
printf '%s  %s\n' "$MODEL_SHA" "$MODEL_MANIFEST" | sha256sum --check -
```

CLI 仅从固定官方 HF API 获取信任元数据，核对 public/ungated、repo/revision、大小及官方 Git/LFS 哈希并规范化输出。预期 16 文件、15,247,184,074 字节。SHA 不一致必须停下调查，不自动改预期值。公开模型不预设需要 HF 账号。

下载需 `huggingface_hub`。创建独立 Python 3.12 venv，安装包内 22 个精确 wheel 哈希锁，不碰宿主 Torch。这是下载工具环境，不能宣称 GPU 训练环境通过：

```bash
python3.12 -m venv "$MODEL_WORK/download-venv"
DOWNLOAD_PY="$MODEL_WORK/download-venv/bin/python"
"$DOWNLOAD_PY" -m pip --isolated --disable-pip-version-check --no-cache-dir install \
  --require-hashes --no-deps -r containers/gpu/requirements-gpu-hashed.lock
MODEL_DIR="$MODEL_WORK/REAL-Prover"
"$DOWNLOAD_PY" containers/gpu/download_model.py --repo "$MODEL_REPO" --revision "$MODEL_REV" \
  --manifest "$MODEL_MANIFEST" --manifest-sha256 "$MODEL_SHA" --output "$MODEL_DIR"
python3 containers/gpu/download_model.py --repo "$MODEL_REPO" --revision "$MODEL_REV" \
  --manifest "$MODEL_MANIFEST" --manifest-sha256 "$MODEL_SHA" --output "$MODEL_DIR" --verify-only
```

成功要求官方逐文件哈希、hidden_size=3584、4 个 safetensors 分片与 339 个张量的索引/大小一致，产生 `reap-model-lock.json`。HF/CDN 或 PyPI 不通时停问；包内没有历史 CDN 签名 URL 或那次临时下载 helper，不能引用包外文件。经用户授权可另选可验证网络方案，但必须保持官方 manifest/文件哈希；不在控制机下载大权重，不把凭据或签名 URL 写进报告。

## 5. 构建 B，先做 GPU 数值预检

```bash
export CONTAINER_ENGINE=podman
export AUTHORIZED_REMOTE_BUILDER=yes  # 仅在用户已授权该远端 builder 后设置
B_IMAGE=localhost/reap-train:my-multiround-v1
GPU_CONTEXT="$MODEL_WORK/gpu-context-new"
bash v1-result/docker/build.sh gpu-existing "$B_IMAGE" "$MODEL_DIR" \
  "$MODEL_MANIFEST" "$MODEL_SHA" "$GPU_CONTEXT"
python3 v1-result/docker/prepare_context.py --authorized-remote-builder \
  --output "$GPU_CONTEXT" --verify-only
"$CONTAINER_ENGINE" image inspect "$B_IMAGE"
```

脚本调用包内 context helper，复制已验证模型，安装交付 recipe、刷新 context 哈希清单，再构建 `release-existing`。B 使用 `requirements-gpu-hashed.lock`；仅列版本的 `requirements-remote.lock` 不能代替哈希锁。固定 B 底座为 ROCm 7.2.4/PyTorch 2.10，历史宿主用过 Torch 2.11，必须重新验证。

使用新卷，在正式服务之前运行 strict 合成预检。`smoke_search_gpu.py` 通过只读挂载取得，正式 B recipe 未将其安装为独立命令：

```bash
SMOKE_VOLUME=reap-multiround-smoke-new
"$CONTAINER_ENGINE" run --name reap-multiround-smoke-new \
  --device /dev/kfd --device /dev/dri --group-add video --shm-size 8g \
  -v "$PWD/containers/gpu:/checks:ro" -v "$SMOKE_VOLUME:/workspace/out" \
  --entrypoint python "$B_IMAGE" /checks/smoke_search_gpu.py \
  --model-path /opt/models/REAL-Prover --gamma 0.99 \
  --snapshot-root /workspace/out/strict-smoke-snapshots --output /workspace/out/strict-smoke.json
```

检查真实 HIP/GPU、有限 loss/梯度/参数/optimizer、一次更新、未训练 session 不变、恢复、冻结底座全量指纹。保存报告与退出码；它是合成预检，不计入真实 Lean 多轮 TTT。

## 6. 启动 B 与可审计安全通道

在 GPU 主机新终端，从解包根目录启动全新服务，先确认端口和名字未被已有任务占用：

```bash
export AUTHORIZED_GPU_RUNTIME=yes
B_CONTAINER=reap-multiround-b-new
B_VOLUME=reap-multiround-b-output-new
bash v1-result/docker/run-gpu.sh "$B_IMAGE" "$B_CONTAINER" "$B_VOLUME"
```

模板仅在宿主 `127.0.0.1:8760` 发布服务，保持私有 IPC。镜像内入口等价于下式；这行只解释入口，不要在宿主再启动第二份模型：

```bash
python -m gpu_runtime.server --backend real-search --gamma 0.99 \
  --host 0.0.0.0 --port 8760 --model-path /opt/models/REAL-Prover \
  --snapshot-root /workspace/out/snapshots
```

为保存真实 request/result/response 字节，在 GPU 主机另一个终端执行以下包内模块组合，只绑定 loopback。`submit_job` 创建 detached worker；`Bridge` 保持单次提交、只读轮询及未知结果停止语义。选择唯一 RUN_ID，将同一 SID_A/SID_B 传给 A。这里不需要浏览器：

```bash
RUN_ID=run-$(date -u +%Y%m%dT%H%M%SZ)
SID_A="$RUN_ID-a"
SID_B="$RUN_ID-b"
RUN_DIR=/absolute/remote/new-run-directory
mkdir "$RUN_DIR"
export SID_A SID_B RUN_DIR
python3 -B - <<'PY'
import os
from pathlib import Path
from tools.amd_jupyter.http_bridge import Bridge, make_server
from tools.amd_jupyter.remote_http_job import submit_job, poll_job
jobs = Path(os.environ['RUN_DIR']) / 'http-jobs'
assert not jobs.exists(), 'Use a new run directory'
sessions = [os.environ['SID_A'], os.environ['SID_B']]
def transport(action, request=None, *, request_id='', offset=0):
    if action == 'submit':
        return submit_job(jobs, request)
    if action == 'poll':
        return poll_job(jobs, request_id, offset)
    raise ValueError('Unsupported transport action')
server = make_server(Bridge(transport, sessions, timeout_seconds=600), port=18760)
try:
    server.serve_forever()
finally:
    server.server_close()
PY
```

在 A 主机通过用户已授权的 SSH 配置转发记录端口，例如另开终端：

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18760:127.0.0.1:18760 USER@GPU_HOST
```

`USER@GPU_HOST` 由用户提供，不能猜测。可换成另一条获授权安全通道，但须继续经过记录层；不能绕过它直连 8760 后声称 wire 已审计。不要将无认证 GPU API 暴露公网。A 检查 `curl --fail http://127.0.0.1:18760/health` 应返回 `ok:true, backend:real-search`；只表示服务可达。通道未具备时先报告，不依赖本包没有的登录态或浏览器扩展。

## 7. 同时运行两个真实 Lean session

在 A 主机解包根目录，填写上节确定的 SID_A/SID_B，创建新 manifest/容器/卷。默认01为奇数递推与平方累加，05为立方累加；它们的历史成功不保证新运行同样成功。

下列两题命令是通路示例，可换成接收用户授权的其他自然题目，记录相应manifest与源码哈希。原实验五题属于用例/成绩数据，不是新agent必须预先持有的题库，也不替代本轮重新验收。

```bash
CPU_RUN=/absolute/local/new-run-directory
mkdir "$CPU_RUN"
export SID_A SID_B
python3 - "$PACKAGE_DIR/evidence/multiround" "$CPU_RUN/inputs" <<'PY'
import hashlib, io, json, sys, tarfile
from pathlib import Path, PurePosixPath
bundle = Path(sys.argv[1]); output = Path(sys.argv[2])
manifest = json.loads((bundle / 'raw-evidence-manifest.json').read_bytes())
assert manifest['schema_version'] == 'reap.multiround.evidence-manifest.v1'
raw = (bundle / 'raw-evidence.tar.gz').read_bytes()
assert len(raw) == manifest['archive']['bytes'] and len(raw) <= 16 * 1024 * 1024
assert hashlib.sha256(raw).hexdigest() == manifest['archive']['sha256']
expected = {row['path']: row for row in manifest['files']}
assert len(expected) == len(manifest['files'])
selected = {
    'inputs/01-CoupledOddSquare.lean': '1c934ede5d3f068c24d12dc19d4b887078e5d17904cb690f022044714cf244e0',
    'inputs/05-CubeAccumulator.lean': '78d3be9fb0da44c2576e6d51232c6a1f5444ac6573c48ca5269d24b8df316fad',
}
verified = {}
with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
    members = archive.getmembers(); seen = set()
    for member in members:
        name = PurePosixPath(member.name)
        assert member.isfile() and not member.issym() and not member.islnk() and not member.sparse
        assert not name.is_absolute() and name.as_posix() == member.name
        assert '..' not in name.parts and '.' not in name.parts and '\\' not in member.name and ':' not in member.name
        assert member.name not in seen and member.name in expected
        seen.add(member.name)
        assert member.size == expected[member.name]['bytes']
        if member.name in selected:
            assert member.size <= 64 * 1024
            data = archive.extractfile(member).read()
            digest = hashlib.sha256(data).hexdigest()
            assert digest == expected[member.name]['sha256'] == selected[member.name]
            verified[name.name] = data
    assert seen == set(expected) and len(verified) == 2
assert not output.exists() and not any(p.is_symlink() for p in (output, *output.parents))
output.mkdir()
for name, data in verified.items():
    with (output / name).open('xb') as out:
        out.write(data)
print('Verified and copied only the two original Lean inputs; no historical proof files extracted')
PY
python3 - "$CPU_RUN/batch.jsonl" <<'PY'
import json, os, sys
with open(sys.argv[1], 'x', encoding='utf-8') as out:
    for sid, name in ((os.environ['SID_A'], '01-CoupledOddSquare'),
                      (os.environ['SID_B'], '05-CubeAccumulator')):
        out.write(json.dumps({'session_id': sid,
                             'theorem_file': 'multiround-input/' + name + '.lean'}) + '\n')
PY
A_CONTAINER=reap-multiround-a-new
A_VOLUME=reap-multiround-a-output-new
"$CONTAINER_ENGINE" run --name "$A_CONTAINER" --network host \
  -v "$CPU_RUN/batch.jsonl:/batch.jsonl:ro" \
  -v "$CPU_RUN/inputs:/opt/reap-runtime/multiround-input:ro" \
  -v "$A_VOLUME:/workspace/out" --entrypoint python3 "$A_IMAGE" \
  -m cpu_runtime.online_batch --manifest /batch.jsonl \
  --project-dir /opt/reap-runtime --output-dir /workspace/out/sessions \
  --gpu-base-url http://127.0.0.1:18760 --gamma 0.99 --concurrency 2 --max-updates 5 \
  --http-timeout-seconds 720 --barrier-timeout-seconds 900
```

这两题不在冻结CPU recipe的默认题目目录中，所以这里显式增加只读输入挂载，不能直接使用没有该挂载的`run-cpu.sh`。题目 samples=2、max_tokens=128、max_steps=32、visit_discount=990。协调器在实时 checkpoint 屏障等更新 ACK，随后让同一 Lean 进程/树继续；不使用旧 `run_ttt` 分段入口。只提取原题，绝不将归档中的`proof-check*`答案送入搜索。

既有 `completed`、`passed_execution`、`solutions.jsonl` 检查的条件较弱，**不能单独认定本次两轮目标通过**。提前解题、步数用尽未证、空候选都须如实报告。需换自然题目或调整步数/采样时，与用户确认后另开新运行并记录源码/配置哈希；不覆盖证据、不掩盖失败。

## 8. 收集与独立复核

停止新增请求，确认 worker 结局明确后收集。CPU 无论成功失败都保留容器/卷，使用绝对目标路径导出：

```bash
mkdir "$CPU_RUN/cpu-out"
"$CONTAINER_ENGINE" cp "$A_CONTAINER:/workspace/out/." "$CPU_RUN/cpu-out"
"$CONTAINER_ENGINE" inspect "$A_CONTAINER" > "$CPU_RUN/cpu-container.json"
"$CONTAINER_ENGINE" logs "$A_CONTAINER" > "$CPU_RUN/cpu.log" 2>&1
```

冻结 `collect_http_evidence.py` 的原 CLI 白名单只支持历史名称。**不要复用历史 session。**在 GPU 主机解包根目录执行以下 wrapper，仅在当前进程内将白名单设为本轮明确授权的两个新 session，调用原函数，保留其字节/路径/大小/哈希/稳定性检查；不改源码、不采集其他会话、不用新名字重标旧数据：

```bash
python3 -B - <<'PY'
import json, os, re
from pathlib import Path
from tools.amd_jupyter import collect_http_evidence as c
from tools.amd_jupyter.remote_http_job import IDENTIFIER
sessions = [os.environ['SID_A'], os.environ['SID_B']]
assert len(set(sessions)) == 2 and all(IDENTIFIER.fullmatch(s) for s in sessions)
c.ALLOWED = tuple(sessions)
c.ROUTE = re.compile(r'/sessions/(' + '|'.join(re.escape(s) for s in sessions) + r')(?:/(.*))?')
root = Path(os.environ['RUN_DIR'])
print(json.dumps(c.collect(root / 'http-jobs', root / 'http-evidence.json.gz', sessions)))
PY
```

通过授权通道传送新归档及其 SHA256 到 A 的 CPU_RUN，核对传输前后哈希。另导出 B 输出卷中的 before/after snapshot、服务日志及镜像/容器 inspect，保持原字节并生成哈希；**snapshot API 回执不等于文件/张量已复核**。不要传输模型权重或凭据。

归档哈希确认后解压 JSON，调用包内 wire 审计：

```bash
python3 - "$CPU_RUN/http-evidence.json.gz" "$CPU_RUN/http-evidence.json" <<'PY'
import gzip, sys
with gzip.open(sys.argv[1], 'rb') as src:
    data = src.read(128 * 1024 * 1024 + 1)
assert len(data) <= 128 * 1024 * 1024
with open(sys.argv[2], 'xb') as out:
    out.write(data)
PY
for sid in "$SID_A" "$SID_B"; do
  python3 -B tools/amd_jupyter/audit_online_wire.py \
    --collection "$CPU_RUN/http-evidence.json" \
    --cpu-session-dir "$CPU_RUN/cpu-out/sessions/$sid" \
    --output "$CPU_RUN/wire-$sid.json"
done
```

它核对 observer 顺序/树、checkpoint/ACK、精确 request/receipt、版本与参数哈希链、完整 prompt、真实 response ID、tactic/logprob、value、更新后消费。它不调用 HTTP，不补发 LEARN。`MATCHED_PARTIAL_WIRE` 不是最终证明；`MATCHED_EXECUTION_WIRE` 也不自动满足两轮。

可先对本轮两份报告运行下面的更强检查；任一断言失败就记录对应门禁未满足，不改报告使其通过。它仍不能代替下一步独立 Lean 与张量复核：

```bash
python3 - "$CPU_RUN" "$SID_A" "$SID_B" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
assert len(set(sys.argv[2:])) == 2
for sid in sys.argv[2:]:
    wire = json.loads((root / ('wire-' + sid + '.json')).read_bytes())
    run = json.loads((root / 'cpu-out/sessions' / sid / 'online-result.json').read_bytes())
    assert wire['session_id'] == run['session_id'] == sid
    assert wire['status'] == 'MATCHED_EXECUTION_WIRE' and not wire['gaps']
    assert run['optimizer_updates'] >= 2 and run['root_verified'] and run['returncode'] == 0
    learns = sorted(wire['learns'], key=lambda row: row['version_after'])
    assert len(learns) == run['optimizer_updates']
    assert [row['version_after'] for row in learns] == list(range(1, len(learns) + 1))
    assert all(row['later_generation_wire_verified'] for row in learns[:2])
    print(sid, 'multi-update execution/wire gate passed; independent proof/tensors still required')
PY
```

对这两道未改动的01/05原题，用下列代码生成新复核文件：从本轮只读挂载的已校验原题读取完整上下文，只把调用位置替换为**本轮新session的**`result.json`导出的tactic，并打印对应定理依赖的公理。不能复制历史`proof-check*`答案；若改题，先同步核对源码、文件名和完整定理名：

```bash
python3 - "$CPU_RUN" "$SID_A" "$SID_B" <<'PY'
import hashlib, json, re, sys, textwrap
from pathlib import Path
root = Path(sys.argv[1]); output = root / 'proof-recheck'
output.mkdir(exist_ok=False)
for sid, name, theorem in (
        (sys.argv[2], '01-CoupledOddSquare', 'MultiroundCandidates.CoupledOddSquare.coupled_odd_square'),
        (sys.argv[3], '05-CubeAccumulator', 'MultiroundCandidates.CubeAccumulator.cube_accumulator')):
    original = root / 'inputs' / (name + '.lean')
    original_bytes = original.read_bytes()
    session_dir = root / 'cpu-out/sessions' / sid
    session = json.loads((session_dir / 'session.json').read_bytes())
    result = json.loads((session_dir / 'result.json').read_bytes())
    assert hashlib.sha256(original_bytes).hexdigest() == session['theorem_sha256']
    assert result['solved'] is True and result['session_id'] == sid
    proof = result['proof_script']; assert isinstance(proof, str) and proof.strip()
    assert not re.search(r'\b(sorry|admit|axiom)\b', proof), 'Review forbidden proof content'
    source = original_bytes.decode('utf-8'); marker = '  reapTrainingMCTS'
    assert source.count(marker) == 1
    checked = source.replace(marker, textwrap.indent(proof.strip(), '  '))
    checked += '\n#print axioms ' + theorem + '\n'
    with (output / (sid + '.lean')).open('x', encoding='utf-8') as out:
        out.write(checked)
PY
for sid in "$SID_A" "$SID_B"; do
  proof_status=0
  "$CONTAINER_ENGINE" run --rm --network none --workdir /opt/reap-runtime \
    -v "$CPU_RUN/proof-recheck:/recheck:ro" --entrypoint lake "$A_IMAGE" \
    env lean "/recheck/$sid.lean" > "$CPU_RUN/proof-$sid.log" 2>&1 || proof_status=$?
  printf '%s\n' "$proof_status" > "$CPU_RUN/proof-$sid.exit-code"
  test "$proof_status" -eq 0 || break
done
```

逐份检查零退出码与公理输出，不允许 `sorryAx`、新增未经批准的公理或其他信任绕过；上面的词检查仅是初筛。生成复核文件成功不是 Lean 编译成功，编译成功也不自动补足此前缺少的更新轮数。

逐 session 追加本次门禁：

1. Lean PID/tree_id 唯一连续；两个 session 时间确有重叠。session/tree、LoRA、value head、optimizer、RNG、快照独立。合成隔离预检和真实并发证据分别列出，不相互冒充。
2. `optimizer_updates >= 2`；至少两个不同 event 的 `applied:true, idempotent:false`，版本从 0 连续增长；参数/optimizer 哈希变化、数值有限、冻结底座不变。重复回执不计更新。
3. 对至少前两次更新，wire `learns` 中各自 `later_generation_wire_verified` 必须为 true，generation 在该 ACK 后，真实响应版本对应新版本；不能只改 observer 标签或换树重跑。
4. `result.json` 中最终 solved、实际 `proof_script`，`online-result.json` 的 root_verified/零退出码和 solutions 一致。将导出 proof 替换到**完全相同定理和上下文**的原 `reapTrainingMCTS` 位置，另用此 A 镜像 `--network none` 执行 `lake env lean <复核文件>`。禁止 sorry、新增 axiom、信任绕过；保存原定理/复核源码哈希、输出、退出码和依赖公理检查，不调用模型、不重新搜索。
5. 在已验证 B Torch 环境中，先用包内 `gpu_runtime.snapshot_store.SnapshotStore.load(session_id, name)` 核验快照文件大小/哈希，再以 `weights_only=True` 解码 backend 的 torch payload，比较实际 adapter/value/optimizer/RNG/counter 内容。可复用 `containers/gpu/smoke_search_gpu.py` 的解码/比较逻辑，新审计代码须保存并测试。wire 工具只核对回执/哈希链，没有独立重算所有张量；只有 before/after 文件时不能声称逐轮张量重算通过。

传输超时或 mutation 未知时停止该 session，保留 request UUID/作业文件，只读核查；不重发、不擅自 restore、不换 ID 掩盖未决操作。每个阻塞报告精确错误和需要用户决定的事项。

## 9. 交付

分别报告：源包校验、A 新构建、B 新构建、B 容器 GPU 预检、两个 session 各自更新次数/版本消费/最终 proof、独立 wire/张量/Lean 复核及未达门禁。保存新运行依赖/模型锁、镜像 ID、配置、原始证据与最短复现命令；准备/下载/构建、TTT、审计耗时分开。

现有包不能零准备一键执行，软件、网络资源、GPU/容器授权、安全通道必须先具备。任一计入的 session 没有达到两轮后最终证明，就不能宣布本次总体成功。用户负责实例启停；不建自动启停任务，不向 GitHub/registry 上传或发布，除非另获明确授权。阶段成功后更新材料，工作中及时简洁反馈。

---
