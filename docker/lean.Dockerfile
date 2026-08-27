# reap-lean: Lean 4.28.0-rc1 工具链镜像（本地 520MB 包 COPY，不依赖构建网络）
# 构建（仓库根目录）:
#   podman build -f docker/lean.Dockerfile -t ghcr.io/wufuju2023-cell/reap-lean:4.28.0-rc1 .
# 验证: podman run --rm <img> lean --version
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g; s|http://security.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    zstd ca-certificates git && rm -rf /var/lib/apt/lists/*
COPY tools/amdbridge/lean-4.28.0-rc1-linux.tar.zst /tmp/lean.tar.zst
RUN cd /opt && tar -I zstd -xf /tmp/lean.tar.zst && \
    mv lean-4.28.0-rc1-linux lean && rm /tmp/lean.tar.zst
ENV PATH=/opt/lean/bin:$PATH
CMD ["lean", "--version"]
