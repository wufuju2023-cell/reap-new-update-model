# 代码与复现：从准备环境到检查新证明

本目录集中提供代码、原题和三条成功流程的操作说明。复现会重新运行模型和Lean，输出属于你这次实验；历史成功记录用于说明流程和对照检查项。

## 选择一条流程

1. **[单题TTT](01-单题TTT.md)**：7B模型提出证明步骤，CPU搜索积累反馈；GPU更新本题参数，确认后接着同一棵搜索树继续，最后独立检查证明。原例子是[奇数累加题连续更新三次后完成证明](../docs/current/04-实际实例与验收结果.md#same-tree-three-updates)。
2. **[跨题经验复用](02-跨题经验复用.md)**：沿用第一条流程的成功训练结果，检查并发布参数经验；新题复制这份参数，建立独立训练状态，再搜索、训练和证明。原实验中[三道题继承同一份经验后各自完成证明](../docs/current/04-实际实例与验收结果.md#cross-problem-inheritance)。
3. **[多题并发](03-多题并发.md)**：启动两个独立7B服务，各自支持一道题的Lean搜索；两题使用同一份已发布参数，分别检查最终证明。原例子是[双服务完成证明与反证](../docs/current/04-实际实例与验收结果.md#parallel-lean-search)。本条使用已验证证明集中训练得到的参数，搜索期间固定版本。

先完成[环境与代码准备](00-环境与代码.md)。第一、二条可以接续执行；第三条使用单独的模型服务配置，教程给出了发布参数的准备方法。无须为整理文档或查看代码启动GPU。

跨题部分可先读[设计与代码逐步说明](../prompt_for_agent/教学/05-跨题经验的设计与代码实现.md)，再执行第二篇。它明确区分来源经验、每题更新版本、完整恢复和集中训练发布。

## 文件位置

- **[code/](code/README.md)**：183份可直接浏览的程序与构建文件、完整188文件源码包、逐文件清单和代码导读。
- **[inputs/](inputs/README.md)**：五道原始搜索题目，不含参考证明。
- **[CPU配置模板](config/cpu.env.example)、[GPU配置模板](config/gpu.env.example)**：分别填写两台机器的路径、解释器和访问地址。
- **[prepare.py](prepare.py)**：离线校验代码与原题，解压到一个新的实验目录；不联网、不启动模型。
- **[test_prepare.py](test_prepare.py)**：准备工具的本地测试。
- **[scripts/](scripts/)**：新实验的证明提取、独立Lean复验和跨题发布、参数核对工具；具体调用见各篇教程。
- **[结果检查与故障处理](04-结果检查与故障处理.md)**：每条流程的成功条件、需要保存的文件及失联后的处理顺序。
- **[新增入口的本地验证](validation/README.md)**：准备工具与证明检查测试，以及已实际完成的断网Lean复验记录。

## 最短准备命令

在Linux或WSL中，把路径替换为自己的绝对路径。运行目录放在Linux原生文件系统，例如`/home/用户名/reap-runs`；并发入口不接受`/mnt/...`目录。

```bash
export REPORT=/absolute/path/to/v1-result
mkdir -p "$HOME/reap-runs"
export RUN="$HOME/reap-runs/first-reproduction"
python3 -B "$REPORT/reproduction/prepare.py" --output "$RUN"
python3 -B "$REPORT/reproduction/prepare.py" --check --output "$RUN"
```

完成后，`$RUN/source`是代码，`$RUN/inputs`是原题，`$RUN/prepared.json`是校验记录。目标目录必须原先不存在；再执行准备命令会拒绝覆盖。Windows也能做离线准备，真实CPU运行使用Linux/WSL和Podman。

这次整理增加了复现入口和操作说明，框架源码保持原冻结版本。原始实验、生成证明与复现说明分别存放；完整报告检查仍使用`python3 -B "$REPORT/source/current/check_package.py"`。更多集中训练与中断恢复入口见[扩展复现指南](../source/current/README.md)。
