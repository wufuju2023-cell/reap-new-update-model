# Docker 镜像矩阵：CPU-MCTS 与 GPU-Train 严格分离 + 下次实例完整 Runbook

## 0. 设计原则

**镜像职责不可渗透**：任一镜像不承担对方职责；两镜像之间**只通过两个通道**交互——
(1) HTTP（MCTS←→policy/value），(2) 文件卷（产物 JSONL）。禁止把 Lean 环境塞进 GPU 镜像，反之亦然。

## 1. 矩阵

| | A: `reap-lean`（CPU-MCTS） | B: `reap-train`（GPU-Train） |
|---|---|---|
| 镜像 | `ghcr.io/wufuju2023-cell/reap-lean:4.28.0-rc1-reap` ✅ 已推 | `reap-train:1.0`（Dockerfile/CI 就绪；网络稳定后可推） |
| 内容 | Lean 4.28.0-rc1 + reap + lake build + **MCTS-v1** | `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` + `app/` + 3 包 |
| 设备 | **无 GPU**（不加 `--device`） | **需 GPU**（`--device /dev/kfd --device /dev/dri --group-add video`） |
| 角色 | 批量 rollout 客户端（MCTS+验证→JSONL） | 策略采样/价值评分/TTT 更新 服务端 |
| B 的等效现役 | —— | **rocm-pytorch 实例镜像（自带 torch2.10+HIP）+ `pip install peft trl accelerate`（tuna，~2min）** |

## 2. 接口契约（A → B）

| 端点 | 请求 | 响应 |
|---|---|---|
| `POST /v1/chat/completions` | `{prompt, n, temperature}` | `{choices:[{text, logprob_avg}]}` |
| `POST /value` | `{prompt}` | `{"score": float}` |
| `POST /ttt_step` | `{items:[{prompt,target,r,logprob_old}]}` | `{loss, kl, steps}` |
| `POST /adapter/snapshot\|restore` | `{name}` | `{"result":…}` |
| `GET /health` | — | `{ok, device, generated}` |

## 3. 共享卷

```
-v batch:/batch   # 输入 batch.jsonl（A 消费）
-v out:/out       # 产物 solutions/failures/rttt_buffer/rttt_metrics + checkpoint
-v adapter:/adapter  # B 的 adapter 快照（持久）
```

---

# 4. ★ 下次实跑：新实例完整 Runbook（从零到 RTTT 在线）

> 目标：**15-20 分钟内** 控制面 + 环境 + policy_server 上线并跑通 ≥5 步 RTTT。
> 全流程经 amdbridge 控制通道自动执行；仅 `Launch` 与（可选）UI 登录由人在浏览器完成。

## Phase 0 — 实例（人操作，1 次）
1. `developer.amd.com.cn/radeon` → Launch 模板 **reap-pytorch**（rocm-pytorch 镜像 + `SSH:true`）
2. Ready 后浏览器 Jupyter 可访问（登录态复用）；把**新实例 ID**（`u-…`）告知/记录

## Phase 1 — 控制面（自动，<2 min）
```bash
export INST=<实例id>          # 若实例 ID 变化
python3 amdrctl2.py put bootstrap.sh /workspace/bootstrap.sh   # 经桥上传两件套
python3 amdrctl2.py put runner.py /workspace/runner.py
# 人（1 次粘贴）：cd /workspace && bash bootstrap.sh
python3 amdrctl2.py out        # 应见 "bootstrap done" 回执；queued 协议就绪
```

## Phase 2 — 应用文件（自动，分钟级）
```bash
# app/ 三件 + lean-v1（小文件全走桥，私有库无需 PAT）
for f in app/policy_server.py app/rttt_demo.py app/v1_run.py app/v1_sink.py; do
  python3 amdrctl2.py push "$f" "$f"        # push 到 /workspace/<同路径>
done
python3 amdrctl2.py put lean-v1/.keep lean-v1/.keep   # lean-v1 目录占位（V1 batch 用）
```

## Phase 3 — GPU 环境等效安装（自动，2-5 min）
```bash
python3 amdrctl2.py exec '/opt/venv/bin/pip install -q peft trl accelerate \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ 2>&1 | tail -2'
# RTTT 主环所需仅 peft（policy_server 依赖）；trl=训练增强用
```

## Phase 4 — 策略/价值服务（自动，3-5 min）
```bash
python3 amdrctl2.py exec 'nohup /opt/venv/bin/python /workspace/app/policy_server.py \
  --base /workspace/data/real-prover --port 8760 > /workspace/logs/policy.log 2>&1 & echo STARTED=$!'
```
模型权重：`/workspace/data/real-prover` 若缺失：
```bash
python3 amdrctl2.py put ...  # 或海量路径：经队列 exec 起 hf-mirror 下载：
export HF_ENDPOINT=https://hf-mirror.com
python3 -c 'from huggingface_hub import snapshot_download as sd; sd("FrenzyMath/REAL-Prover", local_dir="/workspace/data/real-prover")'   # ~8-10 min, 自动断点
```
健康检查：
```bash
python3 amdrctl2.py exec 'curl -s http://localhost:8760/health'
# → {"ok": true, "device": "cuda:0", ...}
```

## Phase 5 — RTTT 验证（自动，2-3 min）
```bash
python3 amdrctl2.py exec '/opt/venv/bin/python /workspace/app/rttt_demo.py --steps 10 --k 2'
python3 amdrctl2.py out    # 期望：≥5 次 ttt_step ok；loss/kl 收敛
# PASS 判定 = v1-spec 00（10 min 内 ≥5 步梯度）
```

## Phase 6 — V1 batch（可选，增强）
- 拉 lean 容器（实例内需 container runtime + ghcr PAT）：
```bash
docker login ghcr.io -u wufuju2023-cell --password-stdin <PAT(读包)>
docker pull ghcr.io/wufuju2023-cell/reap-lean:4.28.0-rc1-reap
```
- 或走 lean tar 分块（老协议，网络受限时用）：`chunk-send` 372 块版
- 然后：`v1_run.py --image <镜像> --batch out/batch.jsonl`（V1 批量取证）

## Phase 7 — 持续运行 & 收尾
- 指标：`python3 amdrctl2.py out`（qout 增量、rttt_metrics 落盘）
- 每日：`app/archive.sh`（ckpt→ModelScope + hash→git）——见 environment-spec/05
- 换实例：Phase 0-5 重演（15 min），bootstrap 幂等

## 风险与兜底
| 风险 | 兜底 |
|---|---|
| 实例 ID 变化 | `amdrctl2.py` 支持 `INST` env 覆盖（第一步检查 ls 即证） |
| SSO 短暂 503 | 等 60-120s 重试（桥 polls 一直活着，不丢状态） |
| 模型下载中断 | hf snapshot 断点续传；重跑同一命令补全 |
| 实例无 docker runtime | 退回 lean tar 分块（老协议）；或跳过 Phase 6（RTTT 主环不依赖 lean） |
| 代理死/未开 | 桥/工具全走国内直连，不依赖代理；GitHub push 才需要 |
```

---

## 5. 镜像状态与遗留

| 镜像 | 状态 | 备注 |
|---|---|---|
| reap-lean:4.28.0-rc1 | ✅ GHCR | lean 纯工具链（digest 3644c9a…） |
| reap-lean:4.28.0-rc1-reap | ✅ GHCR | 含 reap+lake build；CPU MCTS 容器 |
| reap-train:1.0 | 🔶 Dockerfile/CI 就绪 | 8GB base 下载窗口不稳定；现役等价方案见矩阵 B |
