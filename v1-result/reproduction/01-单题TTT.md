# 单题TTT：在同一棵搜索树中更新模型

本篇运行原题`CoupledOddSquare`。已知两个数列的初值及递推式，目标是证明对所有自然数`n`，`f n = 2*n+1`且`g n = n*n`。输入只含题目和搜索指令，不含现成答案。

CPU用Lean检查候选证明步骤、维护搜索树；GPU根据搜索访问次数和搜索价值更新本题参数。每次收到更新确认，原Lean进程继续使用原树，后续请求使用新版本。最终导出证明，再由另一个断网CPU容器检查。

先完成[环境与代码准备](00-环境与代码.md)。以下CPU命令在Linux或WSL的原生文件系统、同一个本地rootless Podman上下文执行，使用已设置的`REPORT`、`RUN`、`CPU_IMAGE`、`GPU_URL`。GPU终端使用`GPURUN`、`GPUPY`、`MODEL`。

## 1. 核对本次输入和CPU环境

原题来自旧实验归档的`inputs/01-CoupledOddSquare.lean`，统一准备工具已将它校验并提取到`$RUN/inputs/`。先在CPU终端执行：

```bash
set -euo pipefail
python3 -B "$REPORT/reproduction/prepare.py" --check --output "$RUN"
cat "$RUN/inputs/01-CoupledOddSquare.lean"
test ! -e "$RUN/source-session-id.txt"
export SID="single-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')"
printf '%s\n' "$SID" > "$RUN/source-session-id.txt"
mkdir -p "$RUN/outputs" "$RUN/launches"
```

本题文件指定：每次生成2个候选、最多128个生成token、搜索上限32步、最多64个目标、无需检索前提，`reap.visit_discount = 990`。本篇设置`gamma = 0.99`与它一致，最多允许5次参数更新。session ID会参与随机状态初始化，新ID不会保证重现历史中的相同搜索路径。

先断网检查CPU镜像。这里显式映射到镜像内的用户10001，保留容器；命令不修改全局Git信任配置。若本机不支持此rootless映射，先处理本地容器环境，尚未创建任何GPU会话。

```bash
set -euo pipefail
podman run --name "reap-preflight-$SID" --pull never --network none \
  --userns keep-id:uid=10001,gid=10001 --user 10001:10001 \
  --volume "$RUN/source:/repro/source:ro" \
  --volume "$RUN/outputs:/workspace/out:rw" \
  --env PYTHONPATH=/repro/source --env PYTHONDONTWRITEBYTECODE=1 \
  --workdir /opt/reap-runtime --entrypoint bash "$CPU_IMAGE" -lc \
  'set -eu
   test "$(id -u)" = 10001
   test -w /workspace/out
   lake env lean --version | grep -F "version 4.28.0-rc1,"
   printf "import ReapRuntime\nimport Reap.Training\n" > /tmp/repro-preflight.lean
   lake env lean /tmp/repro-preflight.lean
   python3 -m cpu_runtime.online_ttt --help' \
  > "$RUN/launches/preflight-$SID.log" 2>&1
cat "$RUN/launches/preflight-$SID.log"
```

预检命令必须退出0。历史正式CPU镜像ID为`623ae445c3cb8c843afb6c85a1db0dfa25f17e72ae95afaf23a49ef8844f0fce`；自建镜像使用自己检查得到的完整ID。新复验收据会记录实际镜像，保留这一环境区别。

## 2. 在GPU终端启动搜索训练服务

模型与依赖已按第00篇检查。此处使用冻结基座REAL-Prover，固定模型版本为`fe76f68d9a88f342cb7b546307c20292fea9cced`。本题仅训练LoRA附加参数及价值头，基座不更新。

在GPU终端运行下列命令，并保持终端中的服务运行：

```bash
set -euo pipefail
test ! -e "$GPURUN/snapshots"
test ! -e "$GPURUN/experiences"
set -o noclobber
cd "$GPURUN/source"

PYTHONPATH="$GPURUN/source" "$GPUPY" -B -m gpu_runtime.server \
  --host 127.0.0.1 --port 8760 \
  --backend real-search --gamma 0.99 \
  --model-path "$MODEL" --device cuda:0 \
  --snapshot-root "$GPURUN/snapshots" \
  --experience-root "$GPURUN/experiences" \
  --max-resident-sessions 2 --policy-scoring tokenwise \
  > "$GPURUN/search-server.log" 2>&1
```

`real-search`选择`search_visit_backup`训练目标：候选访问次数作为策略目标，搜索backup作为价值目标。本篇不启用旧节点重新估值、不切换评分实现，也不加改变经验合同的可选KL阈值。模型身份、目标、gamma和参数形状由运行时写入快照与经验合同；CPU入口没有额外的`--base`或`--objective`参数。

另开CPU终端，恢复前面设置的变量，确认可信通道可达：

```bash
curl --fail --silent --show-error "$GPU_URL/health"
```

必须实际返回`ok: true`和`backend: real-search`。如果使用OpenCLI桥，请把上一步生成的`SID`加入桥的会话白名单。服务地址或浏览器权限未就绪时先处理连接，不提交搜索。

## 3. 运行一次搜索，保留所有结果

下面复制包内的容器启动脚本。它只调用包内原有`cpu_runtime.online_ttt`，第二篇也可用同一个脚本启动新题。输出与容器名已存在时拒绝重跑；它不会自动重发训练。

```bash
set -euo pipefail
test ! -e "$RUN/run_one.sh"
cp "$REPORT/reproduction/scripts/run_one.sh" "$RUN/run_one.sh"
bash "$RUN/run_one.sh" "$SID" 01-CoupledOddSquare.lean
```

1140秒与1200秒是请求及更新确认等待上限，不是整题耗时限制。运行期间不要重复启动同一session，不要删除正在等待确认的输出目录。

CPU结果位于`$RUN/outputs/$SID/`：`session.json`绑定原题SHA，`process.json`记录Lean进程与搜索树；`observer.jsonl`记录生成、搜索和版本使用；`online-result.json`记录实际更新次数及终态。GPU快照依次为`before-online-ttt`、`after-online-ttt`；题目成功且确有训练时，额外生成`experience-candidate`，CPU保存对应`experience-candidate.json`。

## 4. 分别检查训练过程和数学证明

先查看本次真实结果：

```bash
python3 - "$RUN" "$SID" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); sid = sys.argv[2]
report = json.loads((root / 'outputs' / sid / 'online-result.json').read_bytes())
inspection, = json.loads((root / 'launches' / sid / 'container-inspect.json').read_bytes())
for key in ('status', 'returncode', 'root_verified', 'optimizer_updates',
            'policy_version', 'online_update_consumed_by_later_generation', 'error'):
    print(key, report.get(key))
print('container_running', inspection['State']['Running'])
print('container_exit_code', inspection['State']['ExitCode'])
PY
```

完整题内TTT通过应同时看到：容器已退出且退出码0、`status=passed_execution`、`root_verified=true`、`optimizer_updates>=1`、`online_update_consumed_by_later_generation=true`。最后一项说明更新后的参数确实被同一题的后续生成使用。`solved_without_online_update`表示证明成功但没有形成这一更新闭环；未证明、训练未知或进程失败则保留记录分析，不生成验收声明。

数学证明另行检查。下面的辅助脚本仅从本次`result.json`提取`proof_script`，替换原题中的唯一搜索入口，再为原定理追加公理检查；不修改题意、不填入旧答案。

```bash
python3 -B "$REPORT/reproduction/scripts/extract_online_proof.py" \
  --run-root "$RUN" --session-id "$SID" \
  --input-file 01-CoupledOddSquare.lean \
  --theorem MultiroundCandidates.CoupledOddSquare.coupled_odd_square \
  --cpu-image "$CPU_IMAGE"
```

脚本创建独占目录`$RUN/proof-check/$SID/`，用相同CPU镜像的**新容器**执行`lake env lean /proof/proof.lean`，网络固定为`none`。同时核对原题SHA、session、实际镜像、Podman与容器退出码、容器已停止、原声明公理列表以及没有`sorryAx`。允许Lean常用的`propext`、`Classical.choice`、`Quot.sound`；公理检查或任一身份核对失败都会保留失败收据，不写成功声明。

通过后保留`proof.lean`、标准输出、标准错误、容器检查结果、`receipt.json`及`accepted.json`。其中`accepted.json`采用`reap.reproduction.independent-lean.v1`，绑定证明、输入、日志和复验收据的SHA，供第二篇明确引用。本工具自身的模拟进程测试可先运行：

```bash
python3 -B "$REPORT/reproduction/scripts/test_extract_online_proof.py" -v
```

这些本地测试检查拒绝门与文件绑定。新增入口还已用历史成功证明完成一次真实断网Lean复验，见[本地验证](validation/README.md)；你的新运行仍执行上面的实际检查命令，生成自己的收据。

## 5. 保存结果，继续跨题或结束本次运行

接着做[跨题经验继承](02-跨题经验复用.md)时，保留GPU服务、源session和两个GPU目录。第二篇会在独立验收后发布本题参数，并为另一道题创建新的私有训练状态。本题的`experience-candidate.json`只是待验收候选，尚未对所有新题自动生效。

只做单题时，先确认CPU容器已经停止、没有未确认的训练请求，保存CPU输出及GPU持久目录，然后正常结束本次服务并按平台流程停止实例。日志存在或HTTP超时均不足以判定远端训练结束；状态未知时先查原请求和原会话，不再发送一次训练。

历史中同一题在三次真实更新后证明成功，详见[同树三次更新实例](../docs/current/04-实际实例与验收结果.md#same-tree-three-updates)及[原始五题证据](../evidence/multiround/README.md)。本篇固定的是输入、训练目标和搜索预算，新运行以自己的结果和独立复验为准。
