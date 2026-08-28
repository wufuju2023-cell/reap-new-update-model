# 同卡双副本固定请求验收

两个独立进程在同一张AMD GPU上各加载一份7B模型，从同一个已发布的混合训练版本复制适配器和价值头，并各自维护会话与随机状态。输入是两条已有证明状态的提示；每条生成1个候选，最多16个词元、采样温度0.8，再计算价值估计。本阶段没有运行Lean搜索或训练。

三轮都先恢复相同的初始状态，再比较顺序执行与双进程并行执行。每个副本生成的证明步骤、逐词元对数概率、价值估计和最终私有状态均精确一致；三轮等待GPU完成的调用时间区间都有重叠。

| 轮次 | 串行调用秒数之和 | 并行调用跨度秒 | 比值 |
|---|---:|---:|---:|
| 1 | 1.688729 | 0.886617 | 1.904688 |
| 2 | 1.733476 | 0.872825 | 1.986052 |
| 3 | 1.721802 | 0.893774 | 1.926439 |

固定请求的中位耗时比为 **1.9264倍**；计入父进程文件通信后为1.9388。计时排除了模型加载、恢复、预热与状态审计；整个测试从启动到外层进程退出约183.55秒。每个进程峰值已分配显存约15.642GB，两份模型分别占用显存；设备总显存和空闲显存读数不能按进程相加。

## 原始入口

- [报告](files/collected-final/result/report.json)：三轮计时、输出、私有状态fingerprints、来源、内存和作用范围。
- [固定输入](files/collected-final/result/input.json)、[启动命令](files/collected-final/launch-intent.json)、[外层退出0](files/collected-final/worker-exit.json)。
- [冻结源码与输入清单](files/collected-final/source-manifest.json)：38成员，其中36份Python源码；运行前清单在[result/source-before](files/collected-final/result/source-before.json)。
- [来源文件清单](files/collected-final/result/source-release-files-before.json)：冻结worker重算原文件SHA并检查不变。
- [完整原始材料](raw.zip)、[成员清单](manifest.json)：111个原始文件；collection-manifest传输包装未入包。files仅供阅读，raw保存完整清单内材料。

本地独审逐一复核111原始文件、38部署成员，以及44对command/response与三轮报告绑定。真实张量比较由冻结GPUworker执行；本地未重新读取远端大PT。基座检查是参数/缓冲区身份与版本号，本轮没有新增完整基座字节哈希。

CPU上的完整双服务搜索调度见[另一个实验](../replica-collector/README.md)。本阶段测得的是等待GPU完成的宿主调用区间重叠，没有采集GPU内核执行跟踪，也没有给出整个TTT流程的吞吐提升。大模型和大张量文件保留在远端，不在此包。

[独立审计回执](files/independent-review.json)复核原始文件与上述数值。
