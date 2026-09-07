# 01 — P0/P1/P2 三步部署（命令级可复刻）

## P0 环境（40min）

```bash
# Lean 4.28.0（与本容器一致；mathlib 依赖仅 lake 配置，框架代码零大编译）
curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain v4.28.0 --no-modify-path
source ~/.bashrc / ~/.profile
lean --version          # → Lean (version 4.28.0, …)
# Python（ROCm / CPU 皆可——E 系列实验只做前向，CPU 很慢；推荐带 ROCm 的容器）
python3 -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
# 参考值：torch 2.11.0+git… / transformers 5.14.1 / peft 0.19.1
```

## P1 资源获取（60-120min）

```bash
# 1) 代码（含 gpu/gpu_runtime）
git clone https://github.com/wufuju2023-cell/v1-1-agentic-tool.git /work/v1-1-agentic-tool 2>/dev/null || true
cd /work/v1-1-agentic-tool

# 2) artifact（HF public，勿需 token；用 hf-mirror 直线）
export HF_ENDPOINT=https://hf-mirror.com
pip install -q huggingface_hub 2>/dev/null || true
hf download WufuJu/v1-1-fullv3-artifact --local-dir /work/artifact
sha256sum /work/artifact/backend.full.pt /work/artifact/backend.raw.pt
# 期望: c1d025… / 5b227e…

# 3) REAL-Prover base（公开；下载速度取决于你的出口；也可放缓存路径）
git -C /work/models clone https://huggingface.co/FrenzyMath/REAL-Prover 2>/dev/null \
  || HF_ENDPOINT=https://hf-mirror.com hf download FrenzyMath/REAL-Prover --local-dir /work/models/REAL-Prover
```

## P2 部署 GPU server（30min + 模型加载 2-5min）

```bash
cd /work/v1-1-agentic-tool/gpu
# 必须以包方式运行（平台已修 torch2.11 weights_only 问题；若你环境 torch<2.6 无需补丁）
python -m gpu_runtime.server --backend real-search-categorical \
  --model-path /work/models/REAL-Prover \
  --categorical-value-artifact /work/artifact/backend.full.pt \
  --categorical-value-artifact-sha256 c1d0255221cc1b3a6e80e1d5567f6c535ef0a62f1ab300d237aeaa2610e0385d \
  --categorical-value-artifact-role pretrained \
  --gamma 0.999 --host 0.0.0.0 --port 8000 > /work/server.log 2>&1 &
sleep 90
curl -s http://127.0.0.1:8000/health | head -c 120   # {"ready":true,…}
```

> ⚠️ 两个常见的坑（都已在脚本/文档固化）：
> 1. torch≥2.6 加载 legacy snapshot 报 `weights_only` → 云端 runtime 已含修复（`weights_only=False`）；
> 2. HF 上传的是 JSON+base64 wrapper → 先解包 raw（见 `00-jython` 内嵌脚本，本容器已执行过）。
