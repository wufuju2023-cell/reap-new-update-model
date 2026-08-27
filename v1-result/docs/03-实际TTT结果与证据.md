# 历史 E/F：实际 TTT 结果与证据

本页保留历史 E/F 结果；当前主要成功案例是[第1题三次更新后证明成功](09-五题尝试与并发结果.md)，与原设计对应关系见[10](10-原设计逐项对照与实际流程.md)。

## E证明了什么

**E完成真实Lean反馈、同树在线GPU TTT，并通过独立wire、真快照和无网络proof复核。** [验收摘要](../evidence/acceptance-summary.json)汇总各项判据；以下说明证据如何支持结论。

题目：自然数函数满足`f 0=0`和`f(n+1)=f n+(2*n+1)`，证明`∀n, f n=n*n`。最终保存的proof为：

```lean
intro n
induction n <;> simp_all [hstep, mul_add, add_mul, mul_comm, mul_assoc, mul_left_comm]
ring
```

该proof来自本次搜索，随后放入原命题进行独立重验；正式实验题目调用`reapTrainingMCTS`，未预填这段解法。

| 事件 | 实际结果 | 证据 |
|---|---|---|
| 开始搜索 | session`online-20260827-e`、tree0；一个记录中的`lake env lean`启动，checkpoint树大小2→3→4 | [执行审计](../evidence/independent-execution-audit.json) |
| 搜索中学习 | step1根未证、reward=0；候选`intro n`访问1次，`π=[1]`；value目标约0.5906567362 | 执行审计及原始checkpoint/learn request |
| GPU更新 | 1次optimizer更新，v0→v1；真快照196个LoRA、4个value-head张量变化，optimizer状态更新 | [快照审计](../evidence/recurrence-snapshot-audit.json) |
| 回执身份 | 线上学习回执与实际快照记录的回执和摘要匹配 | [wire/快照交叉检查](../evidence/wire-snapshot-crosscheck.json) |
| 继续搜索 | 后续step2 policy/value请求实际使用v1，prompt、tactic和logprob与本地记录匹配 | [E wire审计](../evidence/independent-wire-e.json)，E共有10个HTTP job |
| 最终proof | 原运行Lean验证成功；原命题及保存proof在`network=none`下再次通过，无sorry或自定义公理 | 执行审计的`proof_recheck`及原始proof/log/inspect |

执行窗口**79.477140037秒**；Lean内部报告**29.085723079秒**。外层窗口包含协调与等待等时间，Lean内部有自己的计时范围；不能把两者相加或当作从下载到验收的总耗时。

## F与整批结果

F会话`online-20260827-f`运行`RecurrenceGeometric.lean`，进行5次更新，32步搜索耗尽仍未证。[F wire审计](../evidence/independent-wire-f.json)保留其线上链路。[空候选分析](../evidence/f-empty-tactic-analysis.json)统计64条候选中53条EOS/空输出；它描述观测现象，尚不能确定唯一原因。

五轮的参数变化、训练目标和后续版本消费见[07-多轮TTT记录与成功边界](07-多轮TTT记录与成功边界.md)。五次更新均有实际回执与连续状态摘要；E/F 两份历史成绩不能拼接成多轮后证明成功；新增第1题三轮、第5题两轮后成功已有独立实验，见[09](09-五题尝试与并发结果.md)。

E/F批次为`manual_intervention_required`，并非两题全解。保留F的未完成记录有助于明确实验适用范围；后续换题或重试使用新session，旧session不自动重放。

## 怎样核原始材料

[raw-evidence.tar.gz](../evidence/raw-evidence.tar.gz)集中保存E/F CPU输出、仅E/F的HTTP材料和精简proof重验产物；[raw-evidence-manifest.json](../evidence/raw-evidence-manifest.json)列出包及成员身份。先依[evidence说明](../evidence/README.md)核hash和成员路径，再解包到一个不存在的新目录，如`review/raw`；成员实际名称以manifest为准。

- CPU部分查看session/配置、`observer.jsonl`、checkpoints、learn请求/回执、树、`online-result.json`及solutions。
- HTTP部分按request ID核请求/响应bytes/hash，再对照E/F wire审计中的event、版本、prompt和tactic。
- `proof-recheck`部分查看原命题、重验proof、日志和inspect，确认原题身份、exit0与无网络运行。

这里保留本次结果所需原始材料，早期A/B历史、临时上传脚本和重复回放未随精简包附带。某份独立审计可能提到它自己的历史检查范围；最终交叉结论由验收摘要和其他对应审计补全，不能把单一审计的局部限制误读为所有证据缺失。

## 结论的范围

本次使用远端AMD宿主GPU；B容器和镜像发布仍未验收。E训练仅一项正访问候选，不能外推成广泛多候选蒸馏成绩。单次记录的启动与连续树/wire支持同树执行，没有独立历史`/proc`采样链。此次未重做全量base hash或snapshot restore；另外的合成预检与本次真实链路分开报告。没有未训练对照，不声称能力提升，也不保证新seed每次成功。
