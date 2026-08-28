# 独立双题证明与反证

## 输入、过程与结果

沿用前一批的固定模型参数，但在搜索前显式展开命题定义，使模型看到数学内容。系统仍检查完整原命题的类型；展开定义不提供证明，也不计为模型生成的证明步骤。

两次尝试分别证明一个命题及另一个命题的完整否定，独立Lean验证均通过，得到1条和2条模型生成的证明步骤。本批没有训练；两个搜索事件区间没有重叠，不能用它证明搜索加速。这是预先选定的分支覆盖，不是成功率测评。

## 核查文件

| 文件 | 核对内容 |
|---|---|
| [campaign-report.json](files/collected/host/campaign-report.json) | 两题独立验证结果和训练数据来源；字段`ok`、`schema_version` |
| [batch-report.json](files/collected/host/batch-report.json) | 搜索与验证阶段时序；字段`schema_version` |
| [independent-review.json](files/independent-review.json) | 独立证据审核；字段`ok`、`schema_version`、`scope` |
| [export-manifest.json](files/collected/export-manifest.json) | 收集材料清单 |

## 原始材料的使用

[manifest.json](manifest.json)记录原始来源、文件大小和SHA哈希；[raw.zip](raw.zip)保存清单内完整原始材料。`files/`提供JSON与Lean文件的阅读副本。需要重新审计或加载训练数据包时，先按清单校验压缩包，再解到新的目录；加载器需要完整数据包，不能仅使用阅读副本。

模型权重、大型适配器及优化器快照、浏览器凭据均未打包。报告中的张量结论来自当时源码已固定的GPU程序，可按哈希追溯；阅读或校验这些文件不会触发训练。

回到[当前证据目录](../README.md)或[实际实例](../../../docs/current/04-实际实例与验收结果.md)。
