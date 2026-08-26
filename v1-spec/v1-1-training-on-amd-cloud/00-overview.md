# V1-1：AMD Cloud 上跑通「教师模型 + 学生模型（reap-7B policy + value head + **on-demand RTTT**）」

## 0. 本次迭代的目标（按优先级排序）

> **核心理解（用户裁定）**：学生模型的"学习"发生在**使用期**（人/教师 API 调用它时才更新）；
> 不预先做长时训练。SFT 是"冷启动校准"，**可选**（REAL-Prover 权重本身已对齐 Lean 格式，
> 默认**零训练直接上线**）。
>
> **STATUS 记录（2026-08-26）**：SFT **= NOT USED**（不启用）。所有 SFT 训练步骤、`train_sft.py`
> 均标记为**停用**，仅作参考/未来保险丝；主流程 = 0 长训 + on-demand RTTT。

| 优先级 | 交付物 | 判定标准 |
|---|---|---|
| **P1** | 学生 7B policy（**REAL-Prover 权重直载，0 训练**）+ value head（随机初始化或一次性微校准）在云端跑通 **on-demand RTTT**：**每次被调用期间**（MCTS 搜索+Lean 验证反馈回流）→ 策略 LoRA 一步更新 + 价值头 TD 更新 | 一次调用（单题）10 分钟内 ≥10 节点搜索、梯度步 ≥5、指标曲线可读；**调用结束即暂停学习**（快照+回滚保护） |
| **P2** | 教师模型接入（Radeon Token Factory）：对难题生成"刚够得着"变体、难度分层；教师侧调用即触发学生 RTTT | 一个难题批量产出 ≥8 个变体并通过过滤；变体 → 学生调用 → TTT 生效 |
| **P3** | （可选增强）轻校准/数据管线：仅在 RTTT 表现不达标时，用 50k pairs 做 **≤1 epoch** 校准（4 卡约 4–6h） | 触发开关：P1 连续 10 题 solve@B 无提升 |

**P1（零长训、用即学）是本 spec 的唯一主线；P3 是保险丝，不是必做。**

## 1. 环境基线（已实证，见 environment-spec）

- 实例：`u-25251-d64e6c11`，4×Radeon PRO W7900 48GB（`torch.cuda.device_count()=4`）
- 镜像：`rocm-pytorch`（Ubuntu 24.04 + ROCm 7.2.4），Python 环境 **`/opt/venv`**
  （`torch 2.10.0+rocm7.2.4`, HIP 7.2.53211；不要用 `/usr/bin/python3`）
- 控制面：`runner.py`（文件队列执行器）+ `amd_jupyter.sh`（WSL 端工具）——已 bootstrap 就绪
- 网络：实例内 `huggingface.co` 需走 **`hf-mirror.com`**（国内镜像），PyPI 走清华 tuna（已验证 200）
- 复用：任意新实例 = Launch → push `bootstrap.sh` → `bash bootstrap.sh`（§04）

## 2. 源与锁定（版本即环境）

| 源 | 内容 | 锁定 | 下载方式 |
|---|---|---|---|
| `hf-mirror.com/FrenzyMath/REAL-Prover` | 7B policy 初始权重（Qwen2.5-Math-7B finetune） | `HF_ENDPOINT=https://hf-mirror.com`；快照 commit 记录 | `huggingface-cli download`（镜像源，可 -c 断点） |
| `hf-mirror.com/datasets/FrenzyMath/state_tactic_pairs` | ~50k (state,tactic) 训练对 | dataproc 快照 | 同上 |
| `github.com/frenzymath/REAL-Prover`（`Realprover/data/fate_m.jsonl`） | FATE-M 评估/课程池 | commit 锁定 | `git clone --depth 1`（走 github 直连或镜像） |
| Lean 4 + mathlib（后续 P1b） | Lean 验证器（checkProof） | `lean-toolchain` v4.28.0-rc1 系列 + reap@fix | elan + lake；mathlib 构建分 4 批（见 §5） |
| PyPI | transformers/peft/trl/datasets/accelerate | `requirements.lock`（pip freeze） | tuna `--index-url https://pypi.tuna.tsinghua.edu.cn/simple/` |

> 全链路遵循 AGENTS 规则：下载一律 `wget -c` / `curl -C -` / `--continue`，单条命令 ≤4 分钟，每批完成后 `touch state/batch_<n>.done` 幂等续传。
