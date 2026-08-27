# 宿主 HTTP 桥补丁：2026-08-27

本目录交付实际修复版 `http_bridge.py` 和对应的 `test_http_bridge.py`，与工作区文件逐字节一致。它修复本地 WSL → Windows OpenCLI → DSW 的宿主桥，不修改模型、训练目标、远端 worker 或 A/B 镜像。

## 为什么保留旧源码包

上一级 `source-snapshot.tar.gz` 与 `source-manifest.json` 记录原交付的固定身份，继续保留原样；原包摘要仍为 `69082304c896cd5995619d9c65a8659d452db13f1cfd08f48a00d7822ab141ce`。先验证旧包，再应用本补丁，不把补丁后的文件冒充原清单内容。原包的 `remote_http_job.py` 已确认与本次测试使用的依赖逐字节相同。

补丁 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `http_bridge.py` | `3e42055edc60fb4fe94b3a8c4d6b68d163be96e0dd59423f019b83544a65ce6f` |
| `test_http_bridge.py` | `2f2e451d0c9aef0d5355d7e8e936bbc27d3234ee39f96582481ff81f8b155f50` |

## 覆盖到已解包源码

以下在 Linux/WSL 执行。先按上一级说明校验并解包，目标应是独立新目录；如已完成解包，不要再次执行 `--extract`。

```bash
cd /absolute/path/to/v1-result
python3 source/verify_source.py --extract /absolute/path/to/new-reap-source
sha256sum source/transport-fix/http_bridge.py source/transport-fix/test_http_bridge.py
```

确认摘要与上表一致，且旧桥已停止后，显式覆盖解包目录中的这两个文件：

```bash
cp source/transport-fix/http_bridge.py /absolute/path/to/new-reap-source/tools/amd_jupyter/http_bridge.py
cp source/transport-fix/test_http_bridge.py /absolute/path/to/new-reap-source/tools/amd_jupyter/test_http_bridge.py
cd /absolute/path/to/new-reap-source
python3 -B -m unittest tools.amd_jupyter.test_http_bridge -v
```

预期 12 项测试通过。这些测试使用注入 transport、本地临时文件和 loopback HTTP，不连接 Edge、远端实例或 GPU。通过后沿用经过确认的桥启动命令，重新指定实际 instance ID、profile、session 白名单，并同时保存 stdout/stderr。不能仅覆盖文件却继续使用已加载旧代码的桥进程。

## 修复内容与边界

初次五 session 批次中，01/05 的 policy 已完成，但首次 value curl 长时间等待；采集时远端没有对应 value job，Lean 因等待 value 不能继续生成 observer。**最初 submit 失败的具体原因没有足够日志，尚未确认**；不能把锁竞争或浏览器超时写成既定根因。

- 进程内 OpenCLI 锁改为 FIFO，仍最多等待 60 秒，避免后来的 poll 插队到已排队 submit 前面。跨进程文件锁保持原样，不承诺跨进程公平性。
- 首次 submit 异常后仍只读查询原 UUID。连续三次 `not_found` 即报告 `UnknownOutcome` 并冻结 session，不再按真实任务的约 660 秒等待窗口空等；三次查询本身仍受锁和 CLI 的有界等待影响。
- 已存在的 `pending` 任务保留原等待窗口；丢失回执但任务存在、或前两次暂不可见的情况，仍能通过原 UUID 恢复。绝不自动重新 submit。
- stderr 输出阶段、UUID、异常类型等结构化事件，不写请求正文、认证、命令参数或异常文本。

查询不到不等于远端绝未执行。未知结果必须核对已有 UUID；不能把报错当作回滚证据，也不能直接重放 learn。

## 已验证与未验证

Windows 桥测试 12/12 通过；WSL 桥测试与既有 HTTP 证据测试合计 14/14 通过。覆盖 FIFO/超时移除、单次提交、丢回执恢复、查无任务快速失败、session 冻结、日志脱敏，以及原校验和/大响应/本地 HTTP 行为。

后续使用本补丁的01r1与05r2已分别完成三次、两次更新后证明成功；通信、实际首末参数与独立Lean检查通过，见[五题报告](../../docs/09-五题尝试与并发结果.md)。05r1创建失败也完整保留，不能称五路首发无故障通过。补丁不代表镜像重建或GPU容器验收。

仍有传输限制：外层CLI等待55秒，而已安装OpenCLI browser eval内部期限更长，可能先超时而远端任务已执行；01r1的原UUID回执恢复验证了这一点。当前通过只读恢复与未知结果停止保护训练，尚未统一全部超时层级，不承诺浏览器通信已完全稳定。
