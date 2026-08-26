# V1-1：AMD Cloud 上跑通「教师模型 + 学生模型（reap-7B policy + value head + 持续推理时训练）」

## 0. 本次迭代的目标（按优先级排序）

| 优先级 | 交付物 | 判定标准 |
|---|---|---|
| **P1** | 学生侧 7B policy（REAL-Prover 权重起步）+ **可微 value head** 在云端实例上跑通**持续推理时训练（RTTT）**：单道题的搜索中，Lean 验证反馈实时回灌 → 策略 LoRA 一步更新 + 价值头 TD 更新 | 一道题 10 分钟内完成≥10 节点搜索且梯度步数 ≥5，策略/价值指标曲线可读 |
| **P2** | 教师模型接入（Radeon Token Factory 免费 OpenAI 兼容 API）：对难题生成"刚够得着"变体、难度分层 | 一个难题批量产出 ≥8 个变体并通过 well-typed/Sim 过滤 |
| **P3** | 数据管线全自动（HF 镜像下载 → 验证 → 分片），累计训练 ≥2 轮 | 50k 样本清洗率 ≥85%，DDP 4 卡可训 |

**P1 是本 spec 的“必须跑通”，P2/P3 为并行增强。**

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
