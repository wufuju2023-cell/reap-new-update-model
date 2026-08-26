# 04. Container-like Control Plane：一次安装，任意新实例复用

## 4.1 设计目标

把"对云 GPU 的操控"抽象为**一个可复制的入口协议**，与 Docker Devcontainer 同一哲学：

- **环境即制品**：控制面（runner、传输层、训练管线、监控）全部收进本地仓库
  `new-update-model/`（git 管理，本地 laptop 为唯一 source of truth）；
- **云端一次性消费**：任意新实例启动后执行**一条** bootstrap 命令 ⇒ 环境恢复 + 控制面在线；
- **实例生命周期无关**：实例可以 `destroy/launch` 任意次，bootstrap 幂等（幂等：重复执行结果一致）；
- **本地优先**：所有变更在本地提交；云端只是"拉取者"（无会话态依赖）。

## 4.2 Bootstrap 协议（一次 = 五件事）

```
新实例 Ready
   │  （用户/自动化执行一行）
   ▼
bash /workspace/bootstrap.sh          ← 本地仓库产物，push 到实例后执行
   ├─ (1) 控制面进程: runner.py（文件队列执行器, 纯 stdlib, 无依赖）
   ├─ (2) 环境指纹: bootstrap.stamp（python/rocm/gpu 数量/时间戳）
   ├─ (3) 目录约定: /workspace/app|data|out|logs (空则建)
   ├─ (4) 回执: 追加 /workspace/qout（"bootstrap done <ts>"）
   └─ (5) 对齐: smoke 先占位（GPU 数上报, torch 检查在 app 层）
   ▼
WSL 侧自动确认（读 qout）→ 控制面在线 → 训练/监控/数据全部按队列协议交流
```

## 4.3 复用清单（未来新实例 = 3 步）

| 步骤 | 动作 | 时长 |
|---|---|---|
| 1 | 平台 Launch（任意模板，推荐 `rocm-pytorch` / `rocm-vllm`） | 1–3 min |
| 2 | push `bootstrap.sh`（WSL 工具 `amd_jupyter.sh push` 上传） | 秒级 |
| 3 | 终端执行 `bash /workspace/bootstrap.sh`（唯一一次人工） | 秒级 |

之后：`exec`(发命令) / `out`(读结果) / `push` `/cat`(文件) —— 全部走已验证的 REST 通道，无浏览器手工操作。

## 4.4 幂等与回滚

- `pkill runner.py` 先杀后起 ⇒ 重复 bootstrap 安全（overlapping 写入由 runner 单实例保证）。
- `bootstrap.stamp` 每次覆盖 ⇒ 可对比"实例发生了几次初始化"。
- 回滚 = 重新 push 旧版本 `bootstrap.sh` + 再执行一次（版本号记录在 stamp 中）。

## 4.5 目录约定（app 层复用）

```
/workspace/
├─ bootstrap.sh        # 一次性初始化（本协议产物）
├─ runner.py           # 文件队列执行器（进程常驻）
├─ runner.log / qout   # 执行日志与回执
├─ q                   # 命令队列（WSL 侧写入）
├─ app/                # 训练管线（本地仓库同步, git）
├─ data/               # 数据集（download 缓存）
├─ out/                # checkpoint/logs（结束前必须拉回）
└─ bootstrap.stamp     # 环境指纹
```

## 4.6 与"环境即镜像"的关系

- `bootstrap.sh` 是**控制面**的一次性制品（轻量、纯 stdlib、任意实例可跑）；
- `Dockerfile.reap-env`（§03）是**计算面**的可复现制品（重量、锁定依赖、CI 构建、digest 分发）。
- 约定：控制面永远先于计算面上线（bootstrap 先行）；两者版本分别记录在
  `bootstrap.stamp` 与 `env.lock`，即"instance 状态"可审计。
