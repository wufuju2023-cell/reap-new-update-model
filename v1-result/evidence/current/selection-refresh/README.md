# 同树选择估值刷新

## 输入、过程与结果

本阶段在同一个Lean进程、同一棵搜索树中完成一次真实训练更新。更新确认到达后，使用新模型重新估计两个尚未解决的选择节点。

两个节点的选择缓存刷新成功，后续搜索使用新版本并完成证明。更新只改变用于下一步选择的估值缓存，保留历史访问次数、累计训练价值和先验；没有声称把整棵旧树的历史价值重新计算了一遍。

## 核查文件

| 文件 | 核对内容 |
|---|---|
| [refresh-audit.json](files/refresh-audit.json) | 更新确认、估值刷新、后续选择和生成事件的对应关系；字段`schema_version`、`source_sha256` |
| [wire-audit.json](files/wire-audit.json) | HTTP版本与输入检查；字段`schema_version` |
| [retirement-receipt.json](files/retirement-receipt.json) | 终态释放收据；字段`schema_version` |

## 原始材料的使用

[manifest.json](manifest.json)记录原始来源、文件大小和SHA哈希；[raw.zip](raw.zip)保存清单内完整原始材料。`files/`提供JSON与Lean文件的阅读副本。需要重新审计或加载训练数据包时，先按清单校验压缩包，再解到新的目录；加载器需要完整数据包，不能仅使用阅读副本。

模型权重、大型适配器及优化器快照、浏览器凭据均未打包。报告中的张量结论来自当时源码已固定的GPU程序，可按哈希追溯；阅读或校验这些文件不会触发训练。

回到[当前证据目录](../README.md)或[实际实例](../../../docs/current/04-实际实例与验收结果.md)。
