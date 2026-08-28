# 继承已有参数，再加入新证明步骤继续学习

## 输入、过程与结果

新学习任务从此前混合训练发布的第二版参数初始化。初始数据仍由模型生成的证明步骤和数学库原有证明组成；第一步训练之后，加入双题实验中新得到、经独立验证的2条反证步骤。这份新证明由初始化所引用的旧模型版本生成，其来源记录保持不变。

真实7B模型共训练两次。每批包含9条模型生成的证明步骤和1条库中原有步骤，不是9道题。第一次训练保存完整状态并恢复后，再做第二次训练；实际采样记录确认第二批使用了新加入的数据。优化器、随机状态和采样位置得以延续，两个发布版本的参数继承及已有推理会话的隔离也通过检查，共16项。

数学库数据沿用[混合学习阶段的三份原有证明](../mixed-learner/README.md)，由Codex编写的脚本采集并复验。本阶段验证通用学习任务的参数继承、完整恢复和增量数据使用；针对单一目标的专门学习与解题效果对照仍未验收。

## 核查文件

| 文件 | 核对内容 |
|---|---|
| [report.json](files/collected-final/result/report.json) | 完整GPU报告；字段`ok`、`gates`、`schema_version` |
| [worker-exit.json](files/collected-final/worker-exit.json) | 实际进程退出并绑定报告SHA；字段`returncode` |
| [final-continuation.json](files/collected-final/result/final-continuation.json) | 第二次完整检查点、发布版本与最终采样位置；字段`stage_passed`、`checkpoint_sha256`、`release_sha256` |
| [step1.json](files/collected-final/result/step1.json) | 第一步原始记录 |

## 原始材料的使用

[manifest.json](manifest.json)记录原始来源、文件大小和SHA哈希；[raw.zip](raw.zip)保存清单内完整原始材料。`files/`提供JSON与Lean文件的阅读副本。需要重新审计或加载训练数据包时，先按清单校验压缩包，再解到新的目录；加载器需要完整数据包，不能仅使用阅读副本。

模型权重、大型适配器及优化器快照、浏览器凭据均未打包。报告中的张量结论来自当时源码已固定的GPU程序，可按哈希追溯；阅读或校验这些文件不会触发训练。

回到[当前证据目录](../README.md)或[实际实例](../../../docs/current/04-实际实例与验收结果.md)。
