# 03. 本项目落地（Radeon Cloud × WSL2）

## 3.1 两端现状基线

| 环境 | 现状 | 备注 |
|---|---|---|
| Cloud | Radeon Cloud 实例 `u-25251-d64e6c11`，4×W7900 48GB，ROCm 7.2（`/opt/rocm`），Ubuntu 24.04 + py3.12，**无 torch** | 云镜像池有 `rocm-pytorch` / `rocm-vllm` / `Unsloth Studio`（自带 torch+HIP） |
| Local | WSL2（i5-1135G7 / 8GB，Intel iGPU 无 ROCm） | 仅能 CPU smoke；容器用 rootless podman |

## 3.2 环境版本策略（推荐）

**版本化环境 = 一个镜像，两个消费端**：

- 云侧 `rocm-pytorch`（官方集成镜像，含 ROCm+PyTorch，**不再手装 torch**）；
- 本地构建**同义镜像**：
  - base：`rocm/pytorch:rocm7.2-ubuntu24.04`（digest 锚定 `@sha256:…`）
  - 锁定 `requirements.lock`：`transformers==x.y.z / peft==x.y.z / trl==x.y.z / datasets==x.y.z / accelerate==x.y.z / vllm==x.y.z`
  - 入口：`/app/pipeline.py`（SFT→GRPO 训练管线，参数化 `--instance=cloud|local`）
- CI（GitHub Actions）：**build 一次 → push（ACR 华东）**；两端都 `pull image@sha256`。

```dockerfile
# Dockerfile.reap-env
FROM rocm/pytorch:rocm7.2-ubuntu24.04@sha256:…   # digest 锚定
USER root
COPY requirements.lock /tmp/
RUN python -m pip install -r /tmp/requirements.lock
COPY app/ /app/
ENV HF_HOME=/workspace/hf HOME=/workspace
WORKDIR /workspace
CMD ["python", "-m", "reap.train.pipeline", "--help"]
```

## 3.3 一致性执行顺序

1. **云**：自建模板 Container Image 选 `rocm-pytorch`（与 `reap-1` 同配置 + SSH 开）→ Launch → 验证 `torch.cuda.device_count()==4`。
2. **本地**：`podman build -t reap-env:1.0`（CPU 模式）→ `podman run` 跑 `smoke_test.py`（`is_available()=False` 属预期，仅验依赖/入口）。
3. **对齐**：把云镜像 `ts.repo` 与本地 Dockerfile 用 **同一 requirements.lock + 同一 app/**（版本控制）→ `git tag env-v1.0`。
4. **gate**：CI 统一构建并 push；两端 pull 同一 digest 才允许进入训练阶段。

## 3.4 与"不再人工操作"的整合

我们已经有 `tools/amd_jupyter/`（队列执行器 runner.py + contents 传输层）。
训练管线全部代码进入 `app/`（同上容器），运行时数据流：

```
[自build镜像] → Podman(本地) / RadeonCloud rocm-pytorch(云端)
    ↓ 同一代码包 app/，仅 --device 注入不同
runner.py(文件队列) → pipeline.py（训练） → qout(回传) → 工具读回
```

## 3.5 局限（诚实陈述）

- 本地**无 AMD GPU** ⇒ 本地与云仅 `torch.cuda` 行为不同；CPU smoke 验证"依赖图/管线逻辑"，
  GPU 数值一致性只在云上验收（若未来本地有独卡，用同一镜像即完全对齐）。
- 云实例的 workspace（100GB NVMe）非持久（use_pvc=false）——
  结束前必须把 checkpoint/日志拉回（工具 `pull` + ACR push）。
