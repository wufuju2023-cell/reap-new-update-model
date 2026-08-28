# 新尝试选择最新已确认的参数版本：本地验收

## 输入、过程与结果

学习器可以显式启用“最新发布版本”记录。只有参数发布完成并确认后，该版本才供新的尝试选择；已有尝试继续固定原版本，不在搜索中途被替换。

本地检查覆盖同进程内的预约与发布顺序、未知发布结果的恢复，以及默认行为保持不变。Windows共133项、其中130项通过和3项跳过；WSL共108项、其中104项通过和4项跳过，独立审核通过。本阶段没有真实GPU运行。后续联合运行的失败与恢复分别见[原运行](../latest-release-gpu/README.md)和[独立恢复](../latest-recovery-gpu/README.md)，不回填为本阶段的验证结果。

## 核查文件

| 文件 | 核对内容 |
|---|---|
| [test-receipt.json](files/final-v2/test-receipt.json) | 测试数量和源码hash；字段`schema_version`、`source_sha256` |
| [independent-review.json](files/final-v2/independent-review.json) | 同进程锁、未知恢复与默认行为审核；字段`schema_version`、`scope`、`source_sha256` |

## 原始材料的使用

[manifest.json](manifest.json)记录原始来源、文件大小和SHA哈希；[raw.zip](raw.zip)保存清单内完整原始材料。`files/`提供JSON与Lean文件的阅读副本。需要重新审计或加载训练数据包时，先按清单校验压缩包，再解到新的目录；加载器需要完整数据包，不能仅使用阅读副本。

本阶段只验证本地机制，没有真实GPU张量结论。包内不含模型权重、大型适配器或优化器快照，也不含浏览器凭据；阅读与校验不触发训练。

回到[当前证据目录](../README.md)或[实际实例](../../../docs/current/04-实际实例与验收结果.md)。
