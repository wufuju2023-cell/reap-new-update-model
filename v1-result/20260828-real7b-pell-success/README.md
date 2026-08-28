# REAL-Prover 7B：课程学习与Pell完整原题成功

本目录面向第一次接触实验、使用另一台电脑的读者。2026-08-28，REAL-Prover 7B完成了**原始Pell任意大正整数解命题**的独立Lean验收，并完成成功轨迹训练、参数发布和校验备份。

**实验设置是“课程参数继承＋学生已验证证明库辅助”。不是未训练基座直出，不是纯参数迁移，也不是OpenCode动态检索实验。** 最后一步很短，是因为此前学生证明的完整序列引理已经可用；这些引理的完整证明也在最终验收中重新编译，没有作为公理导入。

## 阅读顺序

1. [原题、七门课程与两种依赖链](01-problem-and-curriculum.md)：具体数学题面，以及课程为什么这样设计。
2. [7B、MCTS、TTT与学生库](02-system-and-training.md)：模型实际配置、训练目标、隔离与继承。
3. [实验结果、错误与结论边界](03-results-and-lessons.md)：成功、失败、纠错及计时。
4. [从零开始复现](04-reproduce.md)：先离线验证完整证明，再按需准备新的搜索训练实验。

## 最关键证据

- [原题完整Lean证明](proofs/07-original-target.lean)，SHA256：`118904a229711907a6453a0c0b11f280bfb148ebb22112a5550b3b33d6180059`。
- [独立验收](evidence/exp-23a72c86ef2a-01/proof-check/accepted.json)：退出0，完整类型为原Target，公理仅propext、Classical.choice、Quot.sound。
- [真实成功训练](evidence/exp-23a72c86ef2a-01/ops/success-learn/response.json)：v0→v1、optimizer1，392个LoRA与4个value张量变化。
- [下一题来源机制的实际证据](evidence/exp-23a72c86ef2a-01/search/create-receipt.json)：原题继承已成功桥课的发布参数，优化器、buffer、版本重新初始化。
- [最终参数备份校验回执](weights/exp-23a72c86ef2a-01/local-copy-receipt.json)：112731943字节，不含7B基座。

## 目录

```text
proofs/           七份完整学生证明；最后一份包含全部必需引理证明
inputs/           原始题面、三次关键运行的输入和可重新绑定的模板
code/             实际CPU15代码、未改动的GPU code04模块及CPU构建材料
evidence/         成功轨迹、验收、训练、继承和历史错误纠正
weights/          参数身份/依赖/校验值；不含张量文件
scripts/          包校验、离线Lean复验、跨机器输入准备与容器入口
environment.json  实际环境和模型版本
export-provenance.json  原件与便携导出件的哈希对应
package-manifest.json   本目录发布文件的最终校验清单
```

所有操作从本目录出发；不需要实验作者的电脑目录或SSH权限。历史JSON回执中的主机路径已替换为占位符，这些回执是**证据而非可执行配置**。完整证明与实际代码保持原字节；原件SHA与导出件SHA明确分列。不要用旧plan、approval或session重发训练。

Git中不包含模型、镜像层或adapter/value张量。**离线检查证明不需要这些权重；精确重跑已训练模型需要另行取得同SHA的发布参数。** 没有它们时只能做明确标注的fresh-base新实验，不能声称复现同一训练状态。
