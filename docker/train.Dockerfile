# reap-train: 训练环境（torch/peft/trl on ROCm base）
# 注意: base 由 AMD 官方 rocm/pytorch 提供（大型镜像 ~8GB；中国大陆拉取慢，建议在 CI/或云内构建）。
# 构建: podman build -f docker/train.Dockerfile -t ghcr.io/wufuju2023-cell/reap-train:1.0 .
FROM rocm/pytorch:rocm7.2-ubuntu24.04
WORKDIR /workspace
COPY app/requirements.lock /tmp/requirements.lock
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.lock \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple/
COPY app/ /workspace/app/
ENV PYTHONPATH=/workspace/app
CMD ["python3", "/workspace/app/policy_server.py", "--help"]
