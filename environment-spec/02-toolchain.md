# 02. 工具链矩阵与命令

## 2.1 分层工具链

| 层 | 工具 | 说明 |
|---|---|---|
| 容器运行时 | Docker / Podman（rootless） | 本地 WSL 常缺特权限 → 优先 rootless Podman；HPC 备选 Apptainer |
| AMD 基础镜像 | `rocm/pytorch:rocm7.2-ubuntu24.04`、`rocm/rocm-terminal`、`rocm/vllm` 系列 | 官方 pre-built，自带 ROCm 7.x + PyTorch（HIP 运行时），勿手装 |
| GPU 设备接入（原生 Docker） | `--device /dev/kfd --device /dev/dri --group-add video` | 必须显式注入；ROCm 容器无这两项 ⇒ `torch.cuda.is_available()=False` |
| 镜像仓库 | 阿里云 ACR（华东）、GHCR、Docker Hub | 中国区网络优先 ACR/GHCR（拉取命中内网缓存） |
| 构建 | `docker build` + `dive`（层瘦身）/ `hadolint` | 构建在 CI 完成一次，绝不双端各 build |
| 开发一致性 | VS Code Dev Containers（devcontainer.json）/ Devbox / Nix | 本地"一键进入同镜像" |
| 云端编排 | Kubernetes + [amd-gpu-device-plugin](https://github.com/ROCm/k8s-device-plugin)（等价于 NVIDIA toolkit） | 资源声明 `amd.com/gpu: 1` |
| 声明式管理 | Terraform / Packer / cloud-init | host 层声明，与 devcontainer 同源 |

## 2.2 关键命令（基线）

### 本地（无 GPU，CPU smoke）
```bash
podman run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  ghcr.io/org/reap-env@sha256:xxxx \
  python -c "import torch;print(torch.__version__)"   # CPU-only smoke
```

### GPU 宿主（与云端镜像一致）
```bash
docker run --rm -it --gpus all \
  --device /dev/kfd --device /dev/dri -v /media/rocm:/rocm \
  registry-reaper@sha256:xxxx \
  /app/smoke.py --seed 42
```

### Kubernetes（device-plugin 模式）
```yaml
resources:
  limits:
    amd.com/gpu: 1          # 对应 MI300X/W7900 时可能需要 rocm/gpu 命名
```

## 2.3 Parity 验证（不可省的 gate）

两端执行**同一份** `smoke_test.py`，四步即达标：

1. `rocm-smi --showmeminfo vram`（device 数量/显存一致）
2. `torch.cuda.is_available()` 与 `torch.version.hip`（运行时匹配）
3. 固定随机种子 + 同一输入矩阵 → 输出张量 SHA256 一致（数值等价）
4. 打印环境指纹 `env-fingerprint`（torch/transformers/peft/trl + digest）→ 归档到 CI gate

$$\mathrm{parity} = (\mathrm{hostkernel}, \mathrm{driver})_{\text{裸差异之外}} \oplus \mathrm{imageID} \oplus \mathrm{lockfile} \oplus \mathrm{smoke}^{\mathrm{SHA256}}$$
