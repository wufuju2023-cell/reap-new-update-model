# 初始化合同本地检查

## 输入、过程与结果

客户端可在创建会话时指定期望的模型初始化配置摘要。服务端先核对实际配置，匹配后才分配会话；不匹配或缺失时拒绝创建。默认旧请求的行为保持不变。

测试使用真实本地HTTP服务和简化模型实现，Windows与Linux检查通过。本阶段没有调用远端或GPU，也没有执行Lean证明搜索。

## 核查文件

| 文件 | 核对内容 |
|---|---|
| [receipt.json](files/receipt.json) | 结果、耗时与环境范围；字段`ok`、`scope`、`results` |
| [source-before.json](files/source-before.json) | 运行前源码哈希 |
| [source-after.json](files/source-after.json) | 运行后源码哈希 |

## 原始材料的使用

[manifest.json](manifest.json)记录原始来源、文件大小和SHA哈希；[raw.zip](raw.zip)保存清单内完整原始材料。`files/`提供JSON与Lean文件的阅读副本。需要重新审计或加载训练数据包时，先按清单校验压缩包，再解到新的目录；加载器需要完整数据包，不能仅使用阅读副本。

本阶段只验证本地机制，没有真实GPU张量结论。包内不含模型权重、大型适配器或优化器快照，也不含浏览器凭据；阅读与校验不触发训练。

回到[当前证据目录](../README.md)或[实际实例](../../../docs/current/04-实际实例与验收结果.md)。
