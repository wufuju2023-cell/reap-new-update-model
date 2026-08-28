# 发布前检查

这些是对便携包的本地检查，不是新的GPU实验或成功率重复测量。

- 包内原Target完整证明在固定CPU容器中禁网重新编译，退出0，公理仅标准三项；证明SHA与实验原件一致，耗时55.786072秒。
- IndexedWitness、UnboundedFromRecurrence、OriginalTarget三组模板均在新的目录成功prepare；不依赖历史绝对路径。测试GPU地址设为不可用的loopback端口，未发GPU调用。
- 重新绑定后的OriginalTarget实际CPU preflight也已ready；新session尚未create、未运行GPU，不能把此预检计为新模型证明。
- 发现并修正准备入口的可移植性问题：冻结driver要求容器内的Lean项目，不能直接在没有该项目的宿主执行prepare；包装现在在固定CPU容器内prepare。
- 历史回执的路径脱敏不修改证明体；新模板重新绑定相应导出验收文件SHA，不伪造旧plan签名。
- Markdown相对链接、文件大小和敏感路径/令牌模式均检查；模型、权重、镜像层、凭据未进入包。

本次没有重建所有CPU镜像或重跑GPU搜索。自己的机器仍须按复现说明构建、记录实际image ID并完成preflight；本检查不能替代自己的运行验收。
