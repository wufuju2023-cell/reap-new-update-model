# 当前TTT框架的检查与复现

**首次复现入口：[代码与复现](../../reproduction/README.md)。** 该目录将单题TTT、跨题经验和双服务并发拆成完整操作章节，附原题、可浏览代码、准备与验收脚本；[新Agent交接与教学](../../prompt_for_agent/README.md)适用于另一台电脑和全新对话。本页保留集中训练、继续学习和发布恢复的扩展入口。

在 `v1-result` 根运行以下命令。Python 3.12 或以上可做包检查；完整本地测试需要 PyTorch。GPU 运行另需与实际硬件匹配的 ROCm、Transformers、PEFT。运行环境见[Docker说明](../../docker/README.md)。

先检查整个交付包：`python3 -B source/current/check_package.py`。它检查清单、各阶段归档及成员、JSON和本地链接；Linux可加`--bash`检查命令块语法。

## 1. 校验并解压源码

```bash
python3 source/current/verify.py
python3 source/current/verify.py --extract /absolute/path/to/new-reap-source
```

目标目录必须不存在。校验内容包括整个 ZIP、每个成员的大小与 SHA、文件集合和安全路径；清单应随可信交付包取得。解压不会联网、训练或覆盖现有工作区。

## 2. 提取真实证明数据，离线检查

```bash
python3 source/current/reproduce.py prepare \
  --source-root /absolute/path/to/new-reap-source \
  --data-root /absolute/path/to/new-reap-data
python3 source/current/reproduce.py check \
  --source-root /absolute/path/to/new-reap-source \
  --data-root /absolute/path/to/new-reap-data
```

脚本从各阶段 `raw.zip` 提取完整文件。生成证明每包17文件，数学库原有证明每包22文件；检查原始字节、命题身份、证明步骤、训练标签、轨迹和历史独立验收收据。当前提供6份生成证明、3份库中原有证明，共16条＋7条证明步骤记录。Codex编写采集与核查脚本，包装辅助证明没有进入训练数据。这一步是离线复核，不重新运行 Lean 或 GPU。

## 3. 本地框架回归

```bash
python3 source/current/reproduce.py local-tests \
  --source-root /absolute/path/to/new-reap-source
```

测试覆盖题内更新、隔离、快照恢复、失败回滚、不可变经验、发布与调度。微型张量用于检查机制。缺少可选依赖或专用本地历史夹具的测试会跳过；以命令实际测试结果为准。没有PyTorch时先完成步骤1、2，GPU环境准备后再运行完整回归。

## 4. 真实 GPU：两次混合更新与恢复

使用已有的固定 REAL-Prover 模型目录，revision 为 `fe76f68d9a88f342cb7b546307c20292fea9cced`。路径指向模型目录，不指向本地下载缓存。下面不加 `--run` 时只打印命令，检查参数后加上该标志执行：

```bash
python3 source/current/reproduce.py mixed \
  --source-root /absolute/path/to/new-reap-source \
  --data-root /absolute/path/to/new-reap-data \
  --model-path /existing/model/REAL-Prover \
  --output /new/run/mixed
```

实际执行两次训练，每批9条生成证明步骤＋1条数学库原有证明步骤，比例按步骤记录数。两次训练之间完整恢复，分别发布R1和R2，并初始化使用这些版本的搜索会话。输出含 `report.json`、检查点和发布仓库。源码、输入、加载的基座模型配置合同都会检查。只加载既有权重，不自动下载。

## 5. 跨阶段持续学习

从上一步 `report.json` 读取 `release2.model_release_sha256`，将它与上一步的 `learner-store` 一起显式传入：

```bash
python3 source/current/reproduce.py continual \
  --source-root /absolute/path/to/new-reap-source \
  --data-root /absolute/path/to/new-reap-data \
  --model-path /existing/model/REAL-Prover \
  --source-release-root /new/run/mixed/learner-store \
  --source-release-sha256 REPLACE_WITH_RELEASE2_SHA256 \
  --output /new/run/continual
```

加 `--run` 执行。新学习器继承来源参数，使用新的私有训练状态；第一次训练后完整恢复，再加入包内的新成功证明数据，执行第二次更新及发布。新增证明由先前固定版本的搜索会话生成，来源记录保存在包内。

## 6. 单题与多题运行

完整命令分别见[单题TTT](../../reproduction/01-单题TTT.md)、[跨题经验复用](../../reproduction/02-跨题经验复用.md)和[多题并发](../../reproduction/03-多题并发.md)。

解压源码后的 `cpu_runtime/online_ttt.py` 是在同一搜索树内更新模型的TTT入口，`online_batch.py` 是多题调度入口；`verified_collector.py` 负责固定发布版本的证明采集。这些入口都提供 `--help`。CPU侧需要预编译Lean 4.28.0-rc1及已构建的Reap/Mathlib容器。

具体已跑通题目的原始 `.lean`、observer、模型更新回执和独立复验收据见[跨题三题实例](../../evidence/current/cross-problem/README.md)和[固定版本双题实例](../../evidence/current/collector-r3/README.md)。这些输入没有参考答案提示；模型生成结果另存于证据。

fresh/chain/bank的来源文件由 `python3 -m cpu_runtime.experience_policy --help` 管理；只传递获准的LoRA和价值头参数；每题的优化器、随机状态、缓冲区和版本独立。中央学习器记录最新已发布模型；新搜索任务预约时固定该版本，正在搜索的题保持原版。详见[跨题与中央学习](../../docs/current/02-跨题经验与中央学习.md)。

## 7. 输出与续作

每个新实验使用新的输出目录。GPU入口先独占保存启动意图，结束后保存实际退出码；已有输出或启动意图会阻止重新执行。若失去连接，先检查原进程、原意图和原报告，再按检查点恢复，避免重复训练。

模型权重、训练大快照和镜像不在本包内。复现可以生成新的检查点与发布内容哈希，实际结果以本次报告为准。当前阶段不包含变体生成课程或成功率对照。

## 8. 双服务与Lean搜索的便携复现

`portable_lean.py`准备新的独占运行目录，调用题目调度器、多服务路由、证明搜索和独立验证模块。对应源码入口为`Matchmaker`、`ReplicaCollector`、`run_collector`和`verify_collected`。固定输入是`∀ n : Nat, n = n`及`¬ (∀ n : Nat, n = 0)`，每题8步，总预算16步；不生成变体、不训练模型。原命题定义在搜索前显式展开，证明由模型重新寻找。

前置条件：使用Linux/WSL原生Python和已能运行的Podman，CPU镜像必须已在本地。两个服务地址须事先配置为两个独立GPU服务且能读取同一明确的混合训练发布版本；部署标签（deployment ID）由操作者分配，用于区分两份实际服务和固定路由；服务端不自动认证这些标签。此工具不启动GPU服务、桥接、下载模型或拉取镜像。

从`v1-result`目录执行；把示例地址和部署标签替换为实际值：

```bash
python3 -B source/current/portable_lean.py --prepare \
  --source-root /existing/verified-reap-source \
  --output /native-linux/new-lean-campaign \
  --image-id d2070a64912f1a7c66ec9e4bb92e66004e8adaa58887e7be333f0bf8e3d5f3b7 \
  --endpoint http://127.0.0.1:18767 --endpoint http://127.0.0.1:18768 \
  --deployment-id ACTUAL-DEPLOYMENT-0 --deployment-id ACTUAL-DEPLOYMENT-1 \
  --release-sha256 0103155583b260c7863ea58bbf2d3377444c272db0ea4f3a75dd49c40b65c98e \
  --project-dir /opt/reap-runtime
python3 -B source/current/portable_lean.py --check --output /native-linux/new-lean-campaign
python3 -B source/current/portable_lean.py --run --output /native-linux/new-lean-campaign
```

`--prepare`和`--check`不调用网络、Podman、Lean或GPU；Windows也可检查准备机制，实际运行的目录须在Linux原生文件系统，不能是`/mnt/c`等Windows挂载。source-root使用前面校验解包的源码，准备时复制到本次目录并记录内容哈希。默认CPU镜像为这组输入已经验证过的镜像。使用按相同配方自建的新镜像时，显式加`--rebind-cpu-image`并传入新镜像ID，走下节的新环境预检；容器需有Lean 4.28.0-rc1、ReapRuntime与VerifiedCollector。

运行顺序：两题分别在断网CPU容器预检→两服务搜索→保存结果→确认会话已释放→断网复检完整证明并逐状态重放→校验并安装数据。Git信任目录仅用容器内环境变量指定，不修改宿主全局配置。已验证数据存储放在Podman命名卷中；主机的执行日志放在原生Linux目录。

默认随机生成新的运行编号并绑定题族标签，据此生成调度身份哈希和会话ID。`--run-id`可显式固定，须保证未被已有服务用过；不复用旧实验ID。输出、启动意图、原生运行目录或数据卷已经存在时，拒绝重复启动。超时保留原容器与意图，先核原结果，不自动重搜、重训、重启或清理。

结果保存在output的`run-exit.json`、`native/campaign-report.json`和`collected/`。原生执行日志、命名卷和容器保留，便于核对未知结果。`ok`只在两个分支独立通过且两服务实际使用后成立；已知耗尽会如实失败。这套便携入口已完成本地准备、命令和未知结果处理测试；实际GPU与Lean结果仍引用对应实验，未为该入口新增运行。

原包装Windows与WSL各9项本地测试通过；新增重绑定的本地验收单独留证。运行测试可设`REAP_PORTABLE_TEST_SOURCE`指向解包源码，再执行`python3 -B source/current/test_portable_lean.py -v`。模板与固定输入说明见[lean_campaign](lean_campaign/README.md)。

### 自建CPU镜像的环境绑定

在上面的`--prepare`命令加`--rebind-cpu-image`，并将`--image-id`替换为Podman实际检查得到的新镜像ID。原默认输入保持不变。该选项重新绑定镜像、Lean版本、项目和源码哈希，并在声明中加入环境哈希注释。数学命题保持原义，环境、声明、问题及执行和预检源码的身份重新计算。

准备结果标为待验证：`run_sha256=null`、`expected_attempts=[]`、新编译收据为空；旧输入和收据移入`provenance/`。`--check`只核对待验证状态的身份；编译成功收据须由真实预检产生。

`--run`先在新镜像的断网容器核对Lean版本与Reap模块，再对两题实际执行`compile_preflight`。只有两个成功收据及原始输出匹配，才生成新的`runtime-plan.json`、题单、调度身份哈希和会话ID，然后开始GPU请求。缺模块、版本错误、编译失败或未知退出都会保留原意图与退出资料，停止相关流程，不自动重试。新镜像模式目前仅完成本地模拟收据/控制逻辑测试；实际兼容性由本次真实预检与完整运行决定。

### 两个GPU服务的启动命令

在GPU宿主已配置的Python环境中，进入已解包源码目录。两个终端分别执行以下命令；第二个把port改为8762，snapshot-root改为另一个新目录。两个进程各加载一份模型，须有足够显存。此处参数对应已验混合训练R2的配置合同：价值类别最多8步，更新后KL上限100。使用其他发布版本时须逐项匹配其合同。

```bash
python3 -B -m gpu_runtime.server \
  --host 127.0.0.1 --port 8761 --backend mixed-replay \
  --model-path /existing/REAL-Prover --device cuda:0 \
  --verified-dataset-root /existing/replay-datasets \
  --mathlib-dataset-root /existing/mathlib-datasets \
  --verified-max-distance 8 --max-post-update-kl 100 \
  --policy-scoring tokenwise --max-resident-sessions 1 \
  --learner-release-root /existing/learner-store \
  --snapshot-root /new/replica-0-snapshots
```

另一个终端使用同样命令，但设置`--port 8762 --snapshot-root /new/replica-1-snapshots`。两个快照目录都须为新目录；原发布仓库仅提供固定参数来源。CPU若不在同一主机，先建立获准的隧道或桥，再给portable脚本传CPU可访问的两个endpoint。服务器CLI没有`--deployment-id`参数；在CPU计划中自行固定两份服务的不同部署标签。

这些服务提供完整运行时接口，应限制在受信网络内。便携采集程序只执行固定版本搜索和会话释放，不调用训练接口。独立证明验证在CPU断网容器完成。停止或失联后先核原session和回执，不用同一个snapshot目录盲目重开。

判定整个批次成功要同时查看`run-exit.json`退出0和`native/campaign-report.json`的ok；`collected/`可能在失败或未知状态下导出，单独存在不表示成功。

## 9. 两代自动发布与消费的独立验证

[latest-release冻结入口](experiments/latest-release/README.md)提供完整参数化命令；先用第4节mixed复现产生自己的来源store/R2，再运行独立探针。探针、计划和恢复脚本均与各自实际部署字节一致，单独附在`experiments/`，核心188文件`source.zip`不变。

原实际运行完成两次训练提交、完整检查点保存、模型发布和最新版本预约，第二次发布后旧R1完整状态保持；子服务900秒控制等待超时使原探针退出1。冻结入口保留此限制。后续`recover_consume.py`完整恢复原R1、消费已确认R2通过8门，新训练/发布均为0；恢复需要原运行的完整快照与内容寻址发布仓库，不能只靠小型证据包执行。原失败与独立恢复分别记录，原预约保留待完成状态。见[原失败阶段](../../evidence/current/latest-release-gpu/README.md)、[恢复通过](../../evidence/current/latest-recovery-gpu/README.md)及[两条入口的完整命令](experiments/latest-release/README.md)。
