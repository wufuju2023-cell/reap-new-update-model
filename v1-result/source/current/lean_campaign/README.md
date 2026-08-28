# 双服务Lean启动模板

`campaign.py`和`container_job.py`由已实际执行的replica-collector campaign参数化，复用生产CPU模块；它们负责容器边界与证据连接，不替换搜索、router或训练后端。

`inputs/`保存原声明、前置编译输入与原收据；没有模型证明答案。`plan-template.json`是两独立命题的固定预算与历史前置身份，新运行会再执行两次CPU预检。`template-manifest.json`绑定模板/输入和原始两个helper的SHA。

不要直接运行此目录的模板。用上一级[portable_lean说明](../README.md)准备独占bundle，并显式指定源码、CPU镜像、两个已配置endpoint和发布SHA。准备生成新的family/run/session身份；运行失败不自动重试。Python禁止`-O`，模板保留原断言门。

显式重绑定由`binding.py`分成pending身份与实际预检收据提交两步。旧默认输入保留，新镜像compatibility不在准备阶段预报成功。
