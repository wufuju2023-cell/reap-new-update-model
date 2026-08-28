# 跨题继承后三题证明

## 输入、过程与结果

Cube、Scaled和Affine三道题从同一个不可变经验版本复制参数，再分别建立独立会话。每题的优化器、随机状态和训练数据缓冲区各自维护。

三题分别经过4次、1次和1次题内更新后得到证明，并通过独立Lean复验。GPU最多保留两个会话，已完成会话释放后再补入下一题；实际搜索事件区间有重叠。本阶段验证了继承与多题调度，没有不继承或不训练对照，不能据此计算学习带来的提升。

## 核查文件

| 文件 | 核对内容 |
|---|---|
| [concurrency-audit-v2.json](files/concurrency-audit-v2.json) | 批次时序与输入hash；字段`ok`、`schema_version`、`source_sha256` |
| [snapshot-audit.json](files/snapshot-audit.json) | 初始来源、终态与私有状态；字段`ok`、`schema_version`、`scope` |
| [wire-audit-05.json](files/wire-audit-05.json) | Cube通信链；字段`schema_version` |
| [wire-audit-03.json](files/wire-audit-03.json) | Scaled通信链；字段`schema_version` |
| [wire-audit-02.json](files/wire-audit-02.json) | Affine通信链；字段`schema_version` |

## 原始材料的使用

[manifest.json](manifest.json)记录原始来源、文件大小和SHA哈希；[raw.zip](raw.zip)保存清单内完整原始材料。`files/`提供JSON与Lean文件的阅读副本。需要重新审计或加载训练数据包时，先按清单校验压缩包，再解到新的目录；加载器需要完整数据包，不能仅使用阅读副本。

模型权重、大型适配器及优化器快照、浏览器凭据均未打包。报告中的张量结论来自当时源码已固定的GPU程序，可按哈希追溯；阅读或校验这些文件不会触发训练。

回到[当前证据目录](../README.md)或[实际实例](../../../docs/current/04-实际实例与验收结果.md)。
