# 完整恢复旧会话，并使用已发布的新参数推理

这是[原失败运行](../latest-release-gpu/README.md)之后的独立恢复验收。**真实7B恢复与推理通过8项检查，主进程、外层监护进程和两个子服务均正常退出。** 原运行退出1的结果保持不变；本阶段没有新增训练或发布。

恢复入口先只读核对原失败报告、退出状态、第二个完整检查点、第二版发布确认、预约记录和旧会话的安全快照。随后启动两个独立7B HTTP服务：服务0恢复旧会话的完整私有状态；服务1从已确认的第二版参数创建新会话，实际生成候选并估计价值。恢复脚本把子服务控制等待设为1800秒，不调用训练或发布接口。

## 实际结果

- 旧会话经HTTP完整恢复后，与原来的完整状态基线相同；新会话使用第二版参数推理前后，恢复后的旧会话仍保持不变。
- 第二版参数与其检查点、发布确认和预约记录一致；新会话实际完成HTTP创建、候选生成和价值估计，两个会话最后都确认释放。
- 两服务的全部冻结基座参数和缓冲区在运行前后哈希一致：339个参数张量、1个缓冲区，共15,231,233,280字节；原参数存储目录的文件清单也保持不变。
- 报告的`ok`与`real_7B_GPU_gate_passed`均为true，8项检查全部通过；主进程和外层进程退出0，两个子服务正常退出，最终四个相关进程均已消失。

## 实验收尾

远端继续点已保存，8个原运行和恢复进程、4个服务端口均已退出。AMD已通过Edge普通界面停止，只读回查为`Stopped`、`IsActive=false`；见[停止回执](files/collected/stop-receipt.json)及[状态回查](files/collected/stop-state-readback.json)。

## 原始依据

[报告](files/collected/result/report.json)、[主进程真实退出](files/collected/worker-exit.json)、[服务0退出](files/collected/result/service-0/worker-exit.json)、[服务1退出](files/collected/result/service-1/worker-exit.json)、[R1恢复基线](files/collected/result/actor1-restored-before.json)、[R2消费后R1](files/collected/result/actor1-restored-after-R2.json)、[恢复继续点](files/collected/recovery-complete.json)。报告SHA为`362f50a4063721f619d995b6be4baf6e95ad0d2b999aa2b6ffcdf8e2f2c72c71`。

[本地独立审核](files/collected/local-audit.json)核对60原始小文件和68份实际代码hash。张量结论来自冻结worker对实际张量的解码与GPU/HTTP执行，本地没有下载大PT重新计算。

[最终只读独审](files/independent-review.json)核对原R1基线、恢复后及R2消费后完整摘要相同，R2参数精确匹配原CP2/release；[操作继续点](files/collected/operator-continuation.json)保存原与恢复进程、端口及恢复资产位置。

这里验证的是在新进程中完整恢复旧会话，不能改写成原会话一直连续运行。两条原预约仍是未完成状态；没有新Lean搜索、数学验收、完整的最新版本多题批次，也没有在GPU上注入未知发布故障。

冻结恢复脚本SHA：`33641acded89106df38cbb1b5048d1b8f3b6bb6989602dd181d39738df442bbd`。见[入口与所需原资产](../../../source/current/experiments/latest-release/README.md)。本阶段raw保存实际补丁ZIP、命令、报告、退出和本地tiny测试；`files/`仅散放便于阅读的小JSON，长路径及全部原始资料保留在raw。大PT快照与CAS留在远端持久目录。
