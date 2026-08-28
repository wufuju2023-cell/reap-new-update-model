# 新增复现入口的本地检查

## 独立Lean复验已实际执行

本次使用新增的`extract_online_proof.py`，读取历史`experience-source-01`会话已生成的证明，在一个新的断网CPU容器中重新检查。输入来源是[历史来源题记录](../../evidence/current/source-experience/README.md)，没有重新搜索或训练GPU模型。

实际结果：Podman与容器退出码均为0，容器已停止，网络为`none`，镜像ID为`623ae445c3cb8c843afb6c85a1db0dfa25f17e72ae95afaf23a49ef8844f0fce`；该定理只依赖`propext`。这验证了新增证明提取与断网复验入口在现有CPU镜像中的运行。

- [完整复验收据](independent-lean/receipt.json)与[检查范围](independent-lean/scope.json)。
- [本次提取并重新检查的证明](independent-lean/proof.lean)与[Lean实际输出](independent-lean/stdout.log)。
- [实际容器检查结果](independent-lean/inspect.stdout.json)与[通过声明](independent-lean/accepted.json)。

约130秒计时包含首次容器用户映射准备和证明复验，不是TTT搜索耗时。收据中原会话的两次更新属于历史来源题，本次没有新增更新。

## 本地机制检查

最终检查使用从188文件源码包独立解压的代码：Windows共39项通过；WSL共33项通过、6项因未安装PyTorch跳过。分项结果、测试脚本及日志哈希见[测试记录](local-tests.json)，逐项输出在[测试日志](test-logs/)。

`test_prepare.py`检查代码与原题校验、拒绝覆盖、文件损坏和失败不提交最终目录。`scripts/test_extract_online_proof.py`使用模拟进程检查证明身份、退出码、离线网络、公理与失败记录。这些测试不调用模型服务；真实Lean结果单独列在上面。

`scripts/test_cross_experience.py`检查验收绑定、单次发布、来源快照的时序、模型合同、真实小型CPU张量的复制与隔离，以及资源回收确认。Windows执行了其中全部张量检查；WSL执行不需要PyTorch的部分。

本目录的通过声明只说明这一份旧证明已在本次复验中通过。按照教程运行新的题目时，应由脚本生成自己的证明检查收据。
