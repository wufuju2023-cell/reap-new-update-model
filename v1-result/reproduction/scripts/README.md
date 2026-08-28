# 复现脚本与输入输出

这些脚本将教程中的操作整理成可重复调用的入口。训练与搜索仍调用[已冻结的框架代码](../code/README.md)，本目录不另写一套TTT算法。

## 单题搜索与证明检查

**[run_one.sh](run_one.sh)** 在CPU机器调用本地Podman，挂载本次`source/`与`inputs/`，启动一题在线TTT。参数是新会话ID、输入文件名，以及可选的跨题经验参数。它读取`RUN`、`CPU_IMAGE`、`GPU_URL`，将输出写入`$RUN/outputs/会话ID/`，将容器日志和退出码写入`$RUN/launches/会话ID/`。同名输出存在时拒绝重跑。

**[extract_online_proof.py](extract_online_proof.py)** 读取这次搜索的`result.json`，提取实际生成的证明，核对原题身份，再在新断网CPU容器中复验。它保存`proof.lean`、命令、日志、容器检查结果及`receipt.json`。所有检查通过后才写`accepted.json`。

完整调用顺序见[单题教程](../01-单题TTT.md)。

## 跨题参数发布与核对

**[cross_experience.py](cross_experience.py)** 提供四个子命令，按[跨题教程](../02-跨题经验复用.md)的顺序执行：

| 子命令 | 执行位置与输入 | 结果 |
|---|---|---|
| `publish` | CPU；来源题输出、独立证明收据、候选快照回执、原GPU服务地址 | 提交一次参数发布，保存发布意图、回执、三个来源标识和新题ID |
| `source-snapshot --phase before` | CPU；发布记录，且目标题尚未启动 | 请求GPU保存来源的运行前快照 |
| `source-snapshot --phase after` | CPU；目标题已完成独立证明检查 | 将目标题结果绑定到来源的运行后快照请求 |
| `check-tensors` | GPU机器；实际快照仓库、经验仓库、复制过来的小型跨题记录 | 在CPU上解码参数，检查新题初始参数相同、私有状态新建、来源在目标题运行期间不变 |
| `retire` | CPU；两题证明收据、参数核对结果和原服务 | 保存终态并释放这两题的模型状态，记录实际回执 |

`check-tensors`需要GPU机器上的完整参数文件，但自身不训练模型，也不调用GPU计算。不要只把小型回执复制到一台没有快照的电脑上运行它。

## 双服务并发

该流程直接使用已交付的[portable_lean.py](../../source/current/portable_lean.py)和其[固定输入与调度模板](../../source/current/lean_campaign/README.md)。`--prepare`准备本次目录，`--check`做离线检查，`--run`才执行两题预检、双服务搜索和独立证明验证。完整环境与参数见[并发教程](../03-多题并发.md)。

## 本地测试

[test_extract_online_proof.py](test_extract_online_proof.py)检查证明提取、命题绑定、容器状态、公理和失败记录。[test_cross_experience.py](test_cross_experience.py)检查发布绑定、单次提交、隔离、张量一致性与资源释放回执；其中张量测试使用本地小张量。

设置`PYTHONPATH`指向校验解压后的代码，再运行：

```bash
export PYTHONPATH="$RUN/source${PYTHONPATH:+:$PYTHONPATH}"
python3 -B "$REPORT/reproduction/scripts/test_extract_online_proof.py" -v
python3 -B "$REPORT/reproduction/scripts/test_cross_experience.py" -v
```

这些测试不启动7B、不调用远端GPU。真实独立Lean复验单独留有[本地记录](../validation/README.md)。
