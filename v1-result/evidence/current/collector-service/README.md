# 双题服务通信与回收

## 输入、过程与结果

本目录记录前后两批双题实验所用的同一个GPU服务，共4个推理会话。各会话在搜索期间使用固定参数版本。

原始HTTP请求、响应和搜索事件逐项对应，4个会话都确认释放。服务共完成47次调用，失败0次，单进程最多同时执行1次模型相关操作。串行保护的是该进程内共享模型的可变状态，不表示多个CPU搜索不能并发；这里也没有测量GPU内核利用率。

## 核查文件

| 文件 | 核对内容 |
|---|---|
| [http-r2-audit.json](files/http-r2-audit.json) | r2原请求关联；字段`ok`、`schema_version`、`scope` |
| [http-r3-audit.json](files/http-r3-audit.json) | r3原请求关联；字段`ok`、`schema_version`、`scope` |
| [http-r3-collection.json](files/http-r3-collection.json) | r3原始通信；字段`schema_version` |
| [service-finish-01.json](files/service-finish-01.json) | 四个会话释放确认 |
| [status-after-finish-01.json](files/status-after-finish-01.json) | 服务退出与累计指标 |

## 原始材料的使用

[manifest.json](manifest.json)记录原始来源、文件大小和SHA哈希；[raw.zip](raw.zip)保存清单内完整原始材料。`files/`提供JSON与Lean文件的阅读副本。需要重新审计或加载训练数据包时，先按清单校验压缩包，再解到新的目录；加载器需要完整数据包，不能仅使用阅读副本。

模型权重、大型适配器及优化器快照、浏览器凭据均未打包。报告中的张量结论来自当时源码已固定的GPU程序，可按哈希追溯；阅读或校验这些文件不会触发训练。

回到[当前证据目录](../README.md)或[实际实例](../../../docs/current/04-实际实例与验收结果.md)。
