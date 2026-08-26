# 05. 整套环境与状态放在哪（随时可续）

## 0. 原则：本地是源头，云实例是执行器，模板是环境，归档使状态跨实例

新实例恢复总时长目标：**< 10 分钟**（不含权重冷下载）。

## 1. 三段布局（所有内容归属）

### A. 制品仓库（git，本地 laptop = first place）

```
/home/zhai/project/reap/new-update-model/        ← 本地 git 仓库（push 到 GitHub/Gitee 私有）
├─ tools/amd_jupyter/     控制面（runner.py / bootstrap.sh / amd_jupyter.sh）
├─ app/                   训练管线（train_sft.py / policy_server.py / value_server.py / rttt_demo.py / restore.sh）
├─ v1-spec/               开发 specs（本文档）
└─ environment-spec/      环境规格
```

- 镜像仓库选择：**Gitee 私有**（大陆访问快；云实例与本地都好拉）优先；GitHub 私有为备份。
- 大小：全部 <= 数 MB（代码不含权重/数据）。
- 版本：tag `v1-<date>`；每次训练包 `env.lock` + `requirements.lock` 同 commit。

### B. 环境容器（持久模板，不随实例消失）

| 层 | 在哪 | 持久性 |
|---|---|---|
| 实例镜像 `rocm-pytorch` | 平台 **My Templates（reap-pytorch.sfh）** 自建模板 | ✅ 永久（profile 级） |
| SSH key / 控制面 | 平台 Profile（SSH Public Key） | ✅ 永久 |
| 只读源码+锁定 | 制品仓库 A | ✅ git 永久 |
| 实例 workspace | `/workspace`（100GB NVMe） | ⚠️ **随 destroy 丢失** |

> 结论：**环境 = 模板 + 仓库**（两者均持久）；实例只是环境的一个瞬时运行实例。

### C. 状态归档（模型/日志/数据缓存）—— 必须跨实例保存

| 内容 | 大小级 | 放到哪 | 方式 |
|---|---|---|---|
| LoRA ckpt / adapters | 单份 0.2–1 GB | **ModelScope 私有 repo**（用户 wufuju 已绑定；大陆可达，加载快） | `restore.sh` 用 modelscope 认证上传/下载 |
| value head `.pt` | MB | 同 repo 或 git LFS | 同上 |
| 训练日志 / ttt_metrics / ladder | <50 MB | 制品仓库 A 的 `out/`（git commit） | 结束即 `git add && commit` |
| 数据缓存（HF sample） | 1–2 GB | 重新拉取（hf-mirror --resume），不归档 | 仅记录“数据切片哈希” |
| mathlib lake 缓存 | 8–12 GB | 不归档（重建 4 批，如 4.2） | 只记录 lean-toolchain 版本 |

## 2. 续接回放（新实例恢复剧本，10 分钟内）

```bash
# mock（全部由 WSL 工具执行）
# 1. Launch 模板 reap-pytorch（ssh:true, rocm-pytorch）
# 2. push bootstrap.sh + runner.py     # 工具一键
bash bootstrap.sh                      # 控制面在线（5s）
# 3. 仓库克隆（内部网络：gitee）
git clone <gitee://reap/new-update-model> /workspace/app-src
cp /workspace/app-src/app/* /workspace/app/  # 固化 app 版本（要求 requirements.lock 在）
# 4. 恢复权重+状态
cd /workspace/app && bash restore.sh --from modelscope:reap/rl-v1 --to /workspace/out
#    （会拉 ckpt + value_head.pt + 最近的 out_data，保持断点训练）
# 5. 一键 smoke
bash smoke.sh   # 报告：模型 hash + gpus + value head 有效 => 续训可直接开始
```

## 3. 每天结束（idempotent 收尾）

```bash
# WSL 侧（或队列 exec）
/app/archive.sh
#  → tar ckpt+Vhead+buffer → ModelScope 上传 → 失败自动重试 2 次
#  → git add/app+out 记录 → commit → push（gitee 私有）
#  → 生成 state/hash.txt（所有产物 SHA256 清单）并发进仓库
```

## 4. 冲突与防呆

- **版本对齐**：`env.lock` 记录（Linux 版本、torch/rocm 版本、reap commit、HF 快照 commit）每次归档写入 hash。
- **同一时刻只允许一个实例训练**（这是业务决策）；重开实例必须先 `archive` 两分钟。
- 若 `ModelScope` 认证 token 过期 → archive 环节发告警到 `out/degraded.md`（不静默丢数据）。
