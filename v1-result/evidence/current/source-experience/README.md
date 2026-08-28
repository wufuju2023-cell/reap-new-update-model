# 来源经验与继承机制

## 输入、过程与结果

本阶段选取已训练并独立验证的来源会话，将获准参数保存为不可变经验版本，再用它初始化新的7B会话。

来源检查、发布、新会话参数继承及私有状态隔离均通过验证。同题恢复的身份限制仍然保留；跨题继承通过显式经验引用实现。后续实际三题搜索结果见[跨题继承实验](../cross-problem/README.md)，本阶段不单独声称解题能力提升。

## 核查文件

| 文件 | 核对内容 |
|---|---|
| [source-acceptance.json](files/source-acceptance.json) | 来源验收与固定身份 |
| [source-wire-audit.json](files/source-wire-audit.json) | 搜索请求与更新对应；字段`schema_version` |
| [snapshot-audit.json](files/snapshot-audit.json) | 实际来源参数检查 |
| [mechanism-report.json](files/mechanism-report.json) | 继承、隔离和恢复机制；字段`ok`、`gates`、`schema_version` |

## 原始材料的使用

[manifest.json](manifest.json)记录原始来源、文件大小和SHA哈希；[raw.zip](raw.zip)保存清单内完整原始材料。`files/`提供JSON与Lean文件的阅读副本。需要重新审计或加载训练数据包时，先按清单校验压缩包，再解到新的目录；加载器需要完整数据包，不能仅使用阅读副本。

模型权重、大型适配器及优化器快照、浏览器凭据均未打包。报告中的张量结论来自当时源码已固定的GPU程序，可按哈希追溯；阅读或校验这些文件不会触发训练。

回到[当前证据目录](../README.md)或[实际实例](../../../docs/current/04-实际实例与验收结果.md)。
