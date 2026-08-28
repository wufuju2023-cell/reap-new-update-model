# 中央学习器发布与证明回流

## 输入、过程与结果

本阶段将“集中训练、发布参数、用新参数搜索、验证证明、把新步骤送回训练”串起来。第一次训练已保存完整检查点并发布参数；新的推理会话使用该版本，得到AffineStrideThree的证明。

证明经独立Lean检查后，2条模型生成的证明步骤进入第二次训练。随后发布第二版参数，再创建会话核对继承结果。完整原始材料包含这份新证明的17文件数据包，以及训练、恢复和会话隔离的检查记录。

期间的失败没有覆盖：首次包装命令在执行前失败；后一次容器中的证明和逐步复验均成功，但数据包写入文件系统时失败，容器整体退出1。后来在Windows原生目录中对相同17个文件重新验证并安装，没有重复证明搜索或伪造容器退出码。

<a id="generated-proof"></a>

## 模型生成的证明

这道题给出数列起点2和每次增加3的条件，要求证明第n项为`2+3n`。[完整证明文件](files/verified-actor-01-host-recovery/registry/fae47cd667809fa30c7a87b782b98e2e4943ba47144ddc69eee930142552402e/accepted-proof.lean)保存模型生成且经独立检查的结果；逐步重放后得到的两条记录用于第二次训练。

## 核查文件

| 文件 | 核对内容 |
|---|---|
| [report.json](files/report.json) | 两步、发布、恢复与隔离；字段`ok`、`gates`、`schema_version` |
| [final-continuation.json](files/final-continuation.json) | 最终完整检查点和可继续使用的参数版本；字段`checkpoint_sha256`、`release_sha256` |
| [independent-review.json](files/independent-review.json) | 原始记录的独立审核；字段`ok`、`scope` |
| [accepted.json](files/verified-actor-01-host-recovery/accepted.json) | 相同17文件的原生目录准入；字段`ok`、`scope` |

## 原始材料的使用

[manifest.json](manifest.json)记录原始来源、文件大小和SHA哈希；[raw.zip](raw.zip)保存清单内完整原始材料。`files/`提供JSON与Lean文件的阅读副本。需要重新审计或加载训练数据包时，先按清单校验压缩包，再解到新的目录；加载器需要完整数据包，不能仅使用阅读副本。

模型权重、大型适配器及优化器快照、浏览器凭据均未打包。报告中的张量结论来自当时源码已固定的GPU程序，可按哈希追溯；阅读或校验这些文件不会触发训练。

回到[当前证据目录](../README.md)或[实际实例](../../../docs/current/04-实际实例与验收结果.md)。
