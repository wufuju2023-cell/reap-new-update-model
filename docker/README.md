# Docker 镜像矩阵：CPU-MCTS 与 GPU-Train 严格分离

## 0. 设计原则

**镜像职责不可渗透**：任一镜像不承担对方职责；两镜像之间**只通过两个通道**交互——
(1) HTTP（MCTS←→policy/value），(2) 文件卷（产物 JSONL）。禁止把 Lean 环境塞进 GPU 镜像，反之亦然。

## 1. 矩阵

| | A: `reap-lean`（CPU-MCTS） | B: `reap-train`（GPU-Train） |
|---|---|---|
| 镜像 | `ghcr.io/wufuju2023-cell/reap-lean:4.28.0-rc1(-reap)` | `ghcr.io/wufuju2023-cell/reap-train:1.0` |
| 内容 | Lean 4.28.0-rc1 + reap + lake build + **MCTS-v1**（BatchSolver/RolloutSink/Step 验证器） | `rocm/pytorch:rocm7.2-ubuntu24.04` + `app/`（policy_server / value head / ttt_step）+ `requirements.lock` |
| 设备 | **无 GPU**：`docker run` **不**加 `--device /dev/kfd --device /dev/dri --group-add video` | **需 GPU**：必须 `--device /dev/kfd --device /dev/dri --group-add video` |
| 角色 | 批量 rollout 客户端：题目→MCTS→Lean 验证（checkProof）→写 JSONL；只发起 HTTP 调用 | 推理+在线更新服务器：策略采样(n, logprobs)、价值评分(JSON score)、TTT 单步更新 |
| 启动 | `docker run --rm -v ... hoghost reap-lean:… bash -c "lake env lean driver"` | `docker run --rm --device ... -p 8760:8760 reap-train:1.0 python app/policy_server.py` |

## 2. 接口契约（A → B，单机同网段）

| 端点 | 请求 | 响应 | 责任方 |
|---|---|---|---|
| `POST /v1/chat/completions` | `{prompt, n, temperature}` | `{choices:[{text, logprob_avg}]}` | B.policy |
| `POST /value` | `{prompt}` | `{"score": float}` | B.value |
| `POST /ttt_step` | `{items:[{prompt,target,r,logprob_old}]}` | `{loss, kl, steps}` | B.rttt |
| `POST /adapter/snapshot\|restore` | `{name}` | `{"result":…}` | B.rttt |
| （可选）`/health` | — | `{ok, device, generated}` | B |

Lean (A) 侧配置：`set_option reap.policy_endpoint "http://<B-host>:8760/v1/chat/completions"` 等。

## 3. 共享卷（仅文件产物）

```
-v $PWD/batch:/batch        # 输入 batch.jsonl（A 消费）
-v $PWD/out:/out            # 输出 solutions/failures/rttt_buffer/rttt_metrics.jsonl + checkpoint dir（A 写，训练方读）
-v $PWD/adapter:/adapter    # B 的 adapter 快照（持久，若 B 容器重启恢复用）
```
> A 与 B 的进程**不挂载彼此的代码目录**；代码更新 = 重新 build 对应镜像（不可变）。

## 4. 运行示例（单机双容器）

```bash
# B（GPU）先起：
docker run --rm -d --name reap-train --network host \
  --device /dev/kfd --device /dev/dri --group-add video \
  -v $PWD/adapter:/adapter ghcr.io/wufuju2023-cell/reap-train:1.0 \
  python app/policy_server.py --port 8760

# A（CPU）再跑 batch（Linux): 
docker run --rm --network host \
  -v $PWD/batch:/batch -v $PWD/out:/out \
  ghcr.io/wufuju2023-cell/reap-lean:4.28.0-rc1-reap \
  bash -c "cd /workspace/reap && lake env lean Reap/Training/BatchSolver.lean --run batch.jsonl"
```

## 5. 状态与下一步

| 镜像 | 状态 | 下一步 |
|---|---|---|
| reap-lean:4.28.0-rc1 | ✅ 已推 GHCR（digest 3644c9a…） | 无 |
| reap-lean:4.28.0-rc1-reap | ⏳ 构建中（job 1059e6104b57） | 完成→验证→push |
| reap-train:1.0 | Dockerfile 已写；base 8GB 未构建 | 决定构建路径（CI / 拉 base 试） |
