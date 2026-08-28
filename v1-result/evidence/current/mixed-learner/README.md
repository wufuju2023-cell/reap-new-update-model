# 模型生成步骤与数学库原有证明的混合学习

## 输入、过程与结果

输入有两种：4份模型生成并经独立验证的证明，共13条证明步骤；3份数学库原有证明，共7条证明步骤。每个训练批次取9条模型生成的证明步骤和1条库中原有步骤，计数单位是步骤，不是题目。

程序在真实7B模型上训练两次；第一次训练后保存并完整恢复学习状态，再继续第二次。每次训练的参数都发布为不可变版本，另建推理会话检查继承结果，并确认已有会话没有被后续更新改变。16项验收检查通过；本阶段没有用新参数执行证明搜索，也没有比较解题效果。

## 数学库数据来源

三份数据取自固定版本Mathlib的`Mathlib/Logic/ExistsUnique.lean`：`ExistsUnique.elim₂`、`ExistsUnique.intro₂`和`ExistsUnique.unique₂`。训练动作逐字取自库中已有证明。Codex编写了采集、状态记录和复验脚本，没有为这三份训练数据另写证明；包装文件中的等价性检查也没有进入训练动作。

来源提交为`5352afccd6866369be9de43f5b7ec47203555f44`。三份来源记录分别是[elim₂](files/frozen-data/mathlib-datasets/633da6857da2df468f5d9e6c2aee30f0b424505d95ee4c94cbbfe575fd203b5d/source.json)、[intro₂](files/frozen-data/mathlib-datasets/98c67e8f220e8f5a9227ce010a95792f838b10a9cee6f15a98f1eae695d788cb/source.json)和[unique₂](files/frozen-data/mathlib-datasets/390db65cb72310385919dfe5207659c51614962420b9a5ff39ec14fe42d874d1/source.json)。记录包含原文件、证明和每条步骤的字节范围及哈希；同目录保留原文和训练数据。每份完整数据包有22个文件，包含采集与独立复验材料。

## 核查文件

| 文件 | 核对内容 |
|---|---|
| [report.json](files/report.json) | 每批两种来源的步骤数、各自损失、参数继承和完整恢复检查；字段`ok`、`gates`、`schema_version` |
| [worker-exit.json](files/worker-exit.json) | 外层进程退出；字段`returncode` |

## 原始材料的使用

[manifest.json](manifest.json)记录原始来源、文件大小和SHA哈希；[raw.zip](raw.zip)保存清单内完整原始材料。`files/`提供JSON与Lean文件的阅读副本。需要重新审计或加载训练数据包时，先按清单校验压缩包，再解到新的目录；加载器需要完整数据包，不能仅使用阅读副本。

模型权重、大型适配器及优化器快照、浏览器凭据均未打包。报告中的张量结论来自当时源码已固定的GPU程序，可按哈希追溯；阅读或校验这些文件不会触发训练。

回到[当前证据目录](../README.md)或[实际实例](../../../docs/current/04-实际实例与验收结果.md)。
