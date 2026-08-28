# 从零开始：另一台电脑如何复核和重跑

本说明不依赖作者电脑的目录、平台账号、端口或SSH密钥。命令假设Linux，或使用自己安装的WSL2 Linux；需要Python3、Git、可正常工作的rootless Podman。GPU训练另需兼容ROCm的AMD GPU及驱动，本次使用单卡48GiB。不要为本包修改正在服务其他任务的环境。

## A. 获取文件并校验

仓库当前为私有仓库，读者需被授予访问权限；不要在命令或文件中粘贴访问令牌。

```bash
git clone https://github.com/wufuju2023-cell/reap-new-update-model.git
cd reap-new-update-model/v1-result/20260828-real7b-pell-success
python3 scripts/verify_package.py
```

所有后续命令均从这个目录执行。proofs/和code/为原始字节，JSON历史主机路径已便携化。package-manifest校验发布内容；export-provenance另记录原件SHA和导出件SHA。历史回执里的占位符不是要创建的目录。

## B. 先复验证明，不需要GPU、7B或任何训练参数

### B1 构建CPU环境

`code/`是完整构建上下文，含Containerfile、Reap补丁、Lean runtime和Python控制代码。固定Lean基础镜像与Reap提交在配方内；构建需要访问镜像仓库、apt、Git和Mathlib缓存。若固定资源不可访问，应报告缺失资源，不换版本、不删检查。

```bash
podman info
podman build --jobs=1 --retry=0 \
  -f code/containers/cpu/Containerfile \
  -t localhost/pell-cpu:reproduction code
cpu_image=$(podman image inspect --format '{{.Id}}' localhost/pell-cpu:reproduction)
podman run --rm --network none --entrypoint lean "$cpu_image" --version
```

本次实际用的是v1-delivery基础CPU镜像，不是后来的selection-refresh/verified-collector增量。新构建可能产生不同image ID，必须记录自己的ID，不能冒用environment.json中的历史ID。Lean必须4.28.0-rc1、Mathlib实际版本核对5352afc…；若不一致，先处理环境差异。该配方是便携重建入口，本次发布没有重新下载镜像/重建全部环境。

### B2 独立编译完整原题

```bash
python3 scripts/verify_lean.py --image "$cpu_image" --output ./checks-original
# 可选：逐份验证七门课程；输出目录必须是新的。
python3 scripts/verify_lean.py --image "$cpu_image" --output ./checks-all --all
```

容器禁网；完整证明体和所有学生引理依赖重新编译，检查最终公理白名单。仅验证原题就已覆盖其完整依赖，不需要GPU或历史session。输出包含stdout、stderr、证明SHA和实际时间。输出目录存在时拒绝覆盖。

## C. 可选：重新运行7B搜索和训练

### C1 先区分两种实验

- **精确指定已训练起点**：需要另行取得对应release的adapter/value backend文件，按weights/下manifest、session及备份回执验证；原Target需exp-26c724af16ee-01。Git中没有这些张量，因此仅clone不足以复制同一训练状态。
- **fresh-base新实验**：只取得固定REAL-Prover基座，用`--fresh-base`。学生库仍显式可见，但参数未继承；必须标为新起点，不保证同样结果，也不能叫本次已训练链的精确复现。

本包已给出全部证明、真实成功训练数据、完整参数链元信息与代码。不同RNG/session、硬件和算子实现会影响生成；不承诺逐token复现。无需为了验证数学成功重做GPU实验。

### C2 准备独立GPU环境

使用已支持ROCm的Python3.12环境；核对environment.json中实际Torch/HIP，不把另一平台版本冒充本次版本。依赖锁不安装或替换Torch。以下只用于读者自己的新环境：

```bash
python3 -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
python3 -m pip install --require-hashes --no-deps \
  --target ./gpu-deps -r code/containers/gpu/requirements-gpu-hashed.lock
export PYTHONPATH="$PWD/gpu-deps:$PWD/code"
export MODEL_DIR="$PWD/model/REAL-Prover"
python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download('FrenzyMath/REAL-Prover', revision='fe76f68d9a88f342cb7b546307c20292fea9cced', local_dir=os.environ['MODEL_DIR'])"
```

模型和镜像是外部大资源，Git未附带；遵守模型访问及许可条件。已有同revision模型可复用，不必重复下载。CUDA风格的`cuda:0`是Torch在ROCm下的接口名，不表示NVIDIA硬件。

### C3 启动新GPU服务

示例让CPU和GPU在同一台机器通过loopback通信，避免绑定到公网。使用新的状态目录和空闲端口，不覆盖已有服务。

```bash
export GPU_RUN="$PWD/new-gpu-state"
mkdir "$GPU_RUN"
python3 code/start_course_gpu.py --backend real-search --gamma 0.99 \
  --host 127.0.0.1 --port 8761 --device cuda:0 --model-path "$MODEL_DIR" \
  --snapshot-root "$GPU_RUN/snapshots" --experience-root "$GPU_RUN/experiences" \
  --success-dataset-root "$GPU_RUN/trusted-success" \
  --max-resident-sessions 1 --policy-scoring tokenwise
```

在另一个终端继续。服务写出实际initialization-contract.json，后续输入必须使用它，不能复制历史合同。若使用已训练起点，把经SHA和ExperienceStore.load验证的整个`<experience_id>/release/{backend,manifest,session}.json`放入新服务experience store；不能仅复制元信息。

### C4 生成全新输入、预检并明确授权

```bash
run="$PWD/new-target-run"
python3 scripts/prepare_inputs.py --lesson original-target \
  --output "$run" --family-id pell-target-reproduction-01 \
  --cpu-image "$cpu_image" --gpu-url http://127.0.0.1:8761 \
  --contract "$GPU_RUN/initialization-contract.json" --fresh-base
family="$run/store/pell-target-reproduction-01"
python3 scripts/cpu.py --run-root "$run" preflight --family "$family"
# 仅在自己核对CPU预检、实际合同、服务连通及资源后执行：
python3 scripts/approve.py --run-root "$run" --operator "your-name" --confirm-ready
python3 scripts/cpu.py --run-root "$run" run --family "$family" \
  --approval "$run/approval.json" --authorize-remote --max-lessons 1
```

继承已训练起点时，用`--seed-metadata "$PWD/weights/exp-26c724af16ee-01/session.json"`替换`--fresh-base`，前提是GPU store中确实有校验通过的完整release。metadata本身不包含参数。

prepare_inputs重新绑定绝对路径、导出验收文件SHA、实际image和合同，保持数学source字节与proof/type pin；生成新family/session。历史学生证明验收回执仍是历史来源，不冒充本次新验收；preflight重新编译。输出路径已存在时拒绝运行，不覆盖旧证据。IndexedWitness和完整桥分别使用`--lesson indexed-witness`和`--lesson unbounded-sequence`，换新family/run目录。

默认32步、每8检查点最多一次更新，最多4次；不自动增预算。若出现真实结构推进再设计新预算；错误固定见证不硬耗上限。停止前核最新结果与已发learn，先停本family请求生产者、确认在途结果并排空，再snapshot/retire。任意活Lean树恢复尚未实现。

### C5 找到证明后完成真实成功闭环

只有run返回`awaiting_external_verification`才继续；exhausted或错误不能走成功发布。

```bash
driver="$run/code/experiments/proof-curriculum/runner/course_driver.py"
python3 scripts/cpu.py --run-root "$run" export-verification \
  --family "$family" --index 0 --output "$run/verify-export"
python3 "$driver" verify-export --export-dir "$run/verify-export" --source-root "$run/code"
python3 scripts/cpu.py --run-root "$run" import-verification \
  --family "$family" --index 0 --export-dir "$run/verify-export"
python3 scripts/cpu.py --run-root "$run" run --family "$family" \
  --approval "$run/approval.json" --authorize-remote --max-lessons 1
python3 scripts/cpu.py --run-root "$run" replay-success \
  --family "$family" --index 0 --output "$run/replay" --lean-project /opt/reap-runtime
python3 scripts/cpu.py --run-root "$run" import-success-replay \
  --family "$family" --index 0 --bundle "$run/replay"
digest=$(sha256sum "$run/replay/dataset.json" | cut -d' ' -f1)
PYTHONPATH="$run/code" python3 -m cpu_runtime.verified_dataset_store \
  --source "$run/replay" --dataset-root "$GPU_RUN/trusted-success" --sha256 "$digest"
python3 scripts/cpu.py --run-root "$run" run --family "$family" \
  --approval "$run/approval.json" --authorize-remote --max-lessons 1
```

末次run是状态机继续，不是重复搜索：同session一次success-learn→seal→publish→retire。核对applied、version、optimizer、参数diff与发布来源；若HTTP结果未知，先查询原事件/回执，不能重发learn。远程分机部署应由操作者安全传送同digest bundle，不使用本实验的任何SSH地址。

## D. 留档与限制

保存新run的完整proof、accepted、source/plan、轨迹、训练数据、create/learn/publish/retire回执与CPU/GPU版本。成功参数按manifest校验备份到持久存储；远端实例盘不是唯一副本。原题成功参数可用于后续新session，但不能恢复已经结束的Lean树。

本包脚本只做便携准备、调用冻结入口或离线检查，没有重写搜索算法、训练目标或偷偷加入教师动作。具体导出文件校验与便携检查见validation/；GPU全流程是本次历史真实运行证据，发布时没有为包装再跑一遍。
