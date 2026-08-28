# 两代自动发布与消费：原运行失败记录

本阶段用一个中央学习进程训练并发布参数，再让两个独立7B HTTP服务选择这些参数。**整体失败，主进程退出1，`report.ok=false`；不计为三进程联合验收通过。** 已完成的两次训练及发布产物保留，后续恢复单独记录。

## 已完成和失败位置

| 环节 | 实际结果 |
|---|---|
| 第一次训练、提交、发布 | 完整检查点及第一版参数已保存，新尝试已预约第一版 |
| 第一版推理 | 服务0完成HTTP创建会话、生成候选和价值估计，随后记录完整私有状态基线 |
| 第二次训练、提交、发布 | 第二个完整检查点及第二版参数已保存，新尝试最终预约第二版 |
| 旧会话跨更新保持 | 第二版发布确认后，另行保存标准安全快照；实际CPU解码得到的完整状态与原基线一致，题内版本仍为0 |
| 第二版推理 | 原服务1的控制通道等候900秒后超时退出，随后HTTP创建连接被拒绝，未完成新会话初始化及后续联合检查 |
| 退出 | 主实验进程和外层监护进程均退出1；最终核对四个相关进程均已消失 |

900秒来自该版验收脚本的子进程控制等待。HTTP请求与控制消息是两条通道，这个等待期限不会随中央训练推进而延长，因而不能作为服务空闲回收策略。原代码与失败原样保留；没有重复提交已发生的训练、发布或改变状态的HTTP请求。

## 关键证据

- [原始报告](files/collected/result/report.json)、[真实worker退出](files/collected/worker-exit.json)、[服务1超时](files/collected/result/service-1/worker-exit.json)。
- [第一次提交及发布确认](files/collected/result/step1.json)、[第二次提交及发布确认](files/collected/result/step2.json)、[第二次预约选择](files/collected/result/selection2.json)。检查点标识分别以`9359f56c…`、`700ee66f…`开头，参数版本分别以`c1631685…`、`8a2bf4d4…`开头，完整哈希保留在原文件中。
- [原R1基线](files/collected/result/actor1-before-second.json)、[安全快照状态审计](files/collected/result/operator-safety-r1/audit.json)、[终态与时序核查](files/terminal-check-01.json)。远端同一文件系统时钟显示安全快照在head2确认之后保存；第二次安全快照尝试没有产生目录或产物。
- [本地原始材料审核](files/collected/local-audit.json)：146文件、67份实际代码hash已核对。原始报告SHA为`611d46040fa26cacf393e771ea05f20d3f78221fa740c4e9ae246d08823a0354`。
- [冻结探针与复现说明](../../../source/current/experiments/latest-release/README.md)。69成员部署代码归档随本阶段raw保存，核心188文件源码归档不变。

## 验收范围与续作

本次没有Lean搜索、完整多题批次、数学验收或未知发布故障注入；两条预约仍是未完成状态。安全快照是独立补充证据，不改写原验收脚本的失败结果。原脚本没有执行完的基座参数和来源文件前后不变检查，不能记为通过。

后续[独立恢复消费](../latest-recovery-gpu/README.md)通过8项检查：从标准快照恢复旧会话，再让新会话使用已确认的第二版参数，新训练和发布均为0。原运行失败保留；此前真实双服务Lean成果仍见[固定版本双服务实验](../replica-collector/README.md)。

`files/`只散放JSON/Lean供阅读，`raw.zip`包含本阶段白名单内完整日志与原始材料。大PT快照及模型权重留在远端持久目录，不在本包；本地审核绑定实际远端审计记录，不表示已下载重算张量。
