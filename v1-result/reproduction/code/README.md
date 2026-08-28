# 可复现版本的代码

[src/](src/)直接提供183份程序、测试、Lean扩展与构建文件，可点击浏览。[source.zip](source.zip)提供完整的188文件版本，另外含5份原代码说明，解压后约1.9MB；[manifest.json](manifest.json)记录每个文件的大小和SHA256。可浏览代码及压缩包均与报告原有冻结源码逐字节一致。

在上一级运行[prepare.py](../prepare.py)，它校验代码和原题后，将源码解压到新实验目录的`source/`。不需要从Git工作区挑文件，也不需要下载7B权重来查看代码。

## 主要代码位置

以下路径均相对于解压后的`source/`：

| 代码位置 | 作用 |
|---|---|
| [cpu_runtime/online_ttt.py](src/cpu_runtime/online_ttt.py) | 单题搜索与训练协调；确认更新后让同一搜索树继续 |
| [cpu_runtime/online_batch.py](src/cpu_runtime/online_batch.py) | 同时安排多道题，完成后释放并补入下一题 |
| [cpu_runtime/publish_experience.py](src/cpu_runtime/publish_experience.py) | 将经过验收的训练快照发布为不可变经验 |
| [cpu_runtime/experience_policy.py](src/cpu_runtime/experience_policy.py) | 选择重新开始、指定前题经验、按规则从经验库选取 |
| [cpu_runtime/replica_collector.py](src/cpu_runtime/replica_collector.py) | 将不同题固定分配给不同模型服务 |
| [verified_collector.py](src/cpu_runtime/verified_collector.py)、[collector_verify.py](src/cpu_runtime/collector_verify.py) | 固定模型搜索、独立验证完整证明和逐步收集训练材料 |
| [server.py](src/gpu_runtime/server.py)、[runtime.py](src/gpu_runtime/runtime.py) | 模型服务接口、各题状态管理、训练、快照与恢复 |
| [search_backend.py](src/gpu_runtime/search_backend.py)、[search_objective.py](src/gpu_runtime/search_objective.py) | 用搜索访问次数和状态估值训练策略与价值部分 |
| [experience_store.py](src/gpu_runtime/experience_store.py) | 保存经验来源、参数哈希与兼容条件 |
| [mixed_learner.py](src/gpu_runtime/mixed_learner.py)、[continual_mixed_learner.py](src/gpu_runtime/continual_mixed_learner.py) | 已验证证明的集中训练、发布及继续学习 |
| [containers/cpu/](src/containers/cpu/)、[containers/gpu/](src/containers/gpu/) | CPU和GPU环境配方、Lean扩展及运行检查 |
| [tests/](src/tests/) | 隔离、继承、恢复、发布、调度等机制测试 |

本次新增的复现准备脚本在[上一级](../README.md)。它们负责整理输入和调用既有入口，没有改动这188份框架源码。`prepare.py`同时校验`src/`可浏览副本；修改这些副本会被检查发现。实际运行统一使用校验后生成的`$RUN/source/`，新运行产物放在代码目录之外。
