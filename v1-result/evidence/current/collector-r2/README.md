# 独立双题第一次真实搜索

## 输入、过程与结果

两道独立命题都使用此前混合训练发布的第二版参数，整个尝试期间版本固定。本次输入保留了命题定义名，没有先展开其中的数学表达式。

两题都达到搜索预算上限，未得到通过验证的证明。程序确认这是已知的未解终态，保存结果并释放两个会话，没有将其当作未知请求重新提交。搜索事件记录的时间区间重叠39.793秒；这说明搜索流程有重叠，不等于GPU内核同时运行。

## 核查文件

| 文件 | 核对内容 |
|---|---|
| [batch-report.json](files/collected/host/batch-report.json) | 并发时序和批次终态；字段`schema_version` |
| [campaign-report.json](files/collected/host/campaign-report.json) | 两题结果；字段`ok`、`schema_version` |
| [launch-intent.json](files/collected/host/launch-intent.json) | 执行意图 |

## 原始材料的使用

[manifest.json](manifest.json)记录原始来源、文件大小和SHA哈希；[raw.zip](raw.zip)保存清单内完整原始材料。`files/`提供JSON与Lean文件的阅读副本。需要重新审计或加载训练数据包时，先按清单校验压缩包，再解到新的目录；加载器需要完整数据包，不能仅使用阅读副本。

模型权重、大型适配器及优化器快照、浏览器凭据均未打包。报告中的张量结论来自当时源码已固定的GPU程序，可按哈希追溯；阅读或校验这些文件不会触发训练。

回到[当前证据目录](../README.md)或[实际实例](../../../docs/current/04-实际实例与验收结果.md)。
