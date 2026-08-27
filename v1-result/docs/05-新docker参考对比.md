# 新 docker 参考与现有实现对比

新材料是参考来源，真实设计TTT决定验收标准。保留现有更可靠的实现，可吸收其A/B职责和容器入口组织。

来源：[新docker目录](https://github.com/wufuju2023-cell/reap-new-update-model/tree/master/docker)，审计固定提交`fc25669e3790f156846897f8f663103a6ae52e74`。此前仅克隆、读取和校验17个文件身份，未运行其安装脚本或构建/发布镜像。下表区分源码静态内容与我们的实测结果。

| 内容 | 新材料给出 | 当前实现与实测 |
|---|---|---|
| 双镜像 | CPU Lean/Reap与ROCm GPU服务分工 | 同方向；旧实验A `df23cd6…`已运行，新交付A `623ae445…`产物与本地运行验收通过，B容器待验；见[08](08-CPU交付镜像验收.md) |
| 可复现环境 | 使用预编译 Lean；Reap 与部分 Python 依赖浮动 | **同样使用预编译 Lean 4.28.0-rc1**，从固定 digest 的基础镜像复用二进制；编译的是 Reap、Training 扩展及项目代码。另固定源码、模型 revision 与 hash；候选 B Torch2.10 仍需独立 GPU 容器验证 |
| 权重 | B仅COPY app，模型放外部目录 | 远端固定16文件/339张量索引已验；受控context和烘焙目标已有，实际镜像未验 |
| session | 接口缺少按session路由，可变训练状态共享 | 单actor下按session路由并隔离LoRA/value/optimizer等状态；有隔离/恢复预检及真实并发E/F记录 |
| 模型接口 | value写死4096、logprob恒0、LoRA初始化注释与配置含义不符 | 固定hidden3584、真实logprob与初始化门禁；以上新材料问题来自静态审计 |
| 训练反馈 | demo循环给1/-0.5；未实现所称value训练与base KL | strict使用Lean visits/backup，训练policy/value；E与新增01r1/05r2真实在线链路通过；F未证，新增多轮成绩见[09](09-五题尝试与并发结果.md) |
| 快照/继续 | model state未含完整value/optimizer/RNG/version；缺少已接通barrier路径 | 完整session设计、真快照与wire交叉核验；E消费v1；新增01r1消费v1/v2/v3，05r2消费v1/v2 |
| 批量恢复 | workers未使用，失败也写done，恢复校验较少 | 有界并发、身份检查、未知mutation闭锁；本次整批需人工处理，未伪报全解 |

两边都采用预编译 Lean，这一点没有分歧；“编译 Reap/Training”不意味着从源码构建 Lean 编译器。

单个ENGINE本身不是关键问题：一个模型引擎或GPU actor可以承载多session。差异在于请求能否路由到正确session，以及同时训练时可变状态是否隔离，不能仅凭ENGINE数量判断。

题内隔离不禁止题间继承，冻结共享base也不要求每题永远丢弃经验。但当前新session独立初始化，没有继承上一题LoRA/value；原文explain第9部分的跨题经验尚未实现。用户本轮先完成五题，跨题继承延期，不将现有隔离/快照能力宣称为完整跨题持续学习。

这些差异支持继续当前方案。新仓库的Dockerfile/CI就绪声明没有证明其B在GPU容器运行成功；我们交付构建材料也需实际build/run。最新实验结果见[09](09-五题尝试与并发结果.md)，部署边界见[06](06-容器未完成原因与交付边界.md)。
