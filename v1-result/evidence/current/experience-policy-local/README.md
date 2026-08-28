# 从基座、前序经验或经验库初始化：本地验收

系统提供三种显式选择：从指定配置的基座开始（`fresh`）；引用确定的前序经验（`chain`）；从获准经验库中选择（`bank`）。经验库按操作者给出的题族、标签筛选，再按优先级降序、标识升序确定唯一来源。默认没有匹配就报错，只有明确允许时才退回基座。标签由操作者声明，不代表系统理解了数学语义或完成语义检索。

[Windows回执](files/windows-receipt.json)记录105项全过；[WSL标准库测试回执](files/wsl-stdlib-receipt.json)记录99项全过。最初WSL全量运行因缺少旧测试所需的Torch出现1项错误，原日志保留。7份源码在测试前后哈希一致。测试使用简化模型和模拟Lean响应，没有运行真实7B或Lean搜索；底层跨题参数继承的真实验收另见[三题证据](../cross-problem/README.md)。

[raw.zip](raw.zip)保存17份原始材料，包括准确CLI说明README、成功/失败日志、前后SHA；[manifest](manifest.json)固定字节。files仅提供JSON阅读副本。使用方法与解包见[当前源码复现入口](../../../source/current/README.md)，解包后模块为`cpu_runtime.experience_policy`。本阶段无大张量。
