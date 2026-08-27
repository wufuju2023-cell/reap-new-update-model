# reap-lean-v2: lean 4.28.0-rc1 + reap clone + lake build（构建期网络；输出单镜像，云端免装）
# 构建: podman build -f docker/reap-lean.Dockerfile -t ghcr.io/wufuju2023-cell/reap-lean:4.28.0-rc1-reap .
FROM ghcr.io/wufuju2023-cell/reap-lean:4.28.0-rc1
RUN git clone --depth 1 https://github.com/IQuestLab/reap.git /workspace/reap
WORKDIR /workspace/reap
RUN lake build
ENV PATH=/opt/lean/bin:$PATH \
    LAKE_HOME=/workspace/reap
CMD ["lean", "--version"]
