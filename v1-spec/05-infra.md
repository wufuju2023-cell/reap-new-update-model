# 05 — 基础设施（AMD Cloud Radeon Cloud 实例）

## 5.1 实例模板与网络

```
平台: developer.amd.com.cn/radeon
模板: 788 "HuggingFace"（ssh_enabled=true; instance_type=opencode; resource=16c55g1u）
GPU:  Radeon PRO W7900 48GB (DID 0x744B, gfx1100 / RDNA3, 241W cap)
配额: 18 credits (1 credit = 1 GPU·h；每账户 1 活跃实例)
接入: SSH (Profile → SSH Public Key 已配置 wsl-zhai 公钥)
```

- SSH 地址：从 `profile` → 实例条目拿 host/port（若未显示：`/api/instances/<id>` 看 `ssh_host`/`ssh_port` 字段）。
- WSL→SSH：`ssh -o StrictHostKeyChecking=no -p <port> root@<host>`；
- 实例销毁 → `profile` 页 "Destroy Instance"；不可运行时重启整实例（Jupyter 进程页面侧负责）。

## 5.2 软件栈（装于实例，含 7B 服务）

```bash
pip install --upgrade torch==2.5.1+rocm6.2 transformers==4.46.* peft accelerate
pip install trl datasets flask waitress  # waitress 可换 uvicorn(仅 CPU)
git clone https://github.com/IQuestLab/reap.git /workspace/reap && cd reap
# 依赖 openAI_client/batteries/requests: git clone 到 /workspace/libs
cd /workspace/libs && git clone https://github.com/frenzymath/openai_client.git
git clone https://github.com/leanprover-community/batteries.git
git clone https://github.com/frenzymath/requests.git
lake env lean --version   # toolchain: 见 reap/lean-toolchain
```

## 5.3 模型/数据获取（**全部走 ModelScope 或 hf-mirror**，国内云直连 HF 不通）

| 对象 | 来源 | 说明 |
|---|---|---|
| REAL-Prover 7B | modelscope: 搜 `FrenzyMath/REAL-Prover`（无则 hf-mirror.com/huggingface.co/FrenzyMath/REAL-Prover） | shards ~15GB |
| state_tactic_pairs | modelscope datasets API `FrenzyMath/state_tactic_pairs` 或 hf-mirror | 50k 对 |
| mathlib 4 | 用 reap 源码旁 `lakefile` 自动 `lake build`（可选 prebuilt elan 环境，若超时用 `lake new`手动） | 无需在线 |
| Qwen2.5-Math-7B | modelscope mirror（备胎 policy init） | 兜底 |

## 5.4 服务编排（单机进程）

```
server.py (FastAPI+transformers+peft)      :8000
 ├ model: REAL-Prover-7B, adapter_id 参数支持
 ├ /v1/chat/completions (n=6, logprobs=True)
 └ /value (state→{"score":-V_φ})
batch_solver.py (lake env lean + reapMCTS) :8001 来自 (Section 2.5)
curriculum.py (DEEPSEEK_API_KEY→批变体→闸门diff)
trainer.py    (TRL GRPO/TTTRL)
```

- 启动顺序：`server` → `reap 库检查` → `batch_solver` → `curriculum nightly`；
- 所有输出 JSONL：`/workspace/runs/<DATE>/solutions.jsonl | verdicts.jsonl | evals.jsonl`；
- 日志轮转：每轮 1GB、保留 7 天；`run_summary.json` 一次汇总。

## 5.5 时间/成本预算（实测 48GB 卡）

| 项 | 耗时 | 说明 |
|---|---|---|
| 模型下载 | ~1 h（ModelScope 镜像速） | 15GB |
| 服务启动 | ~3 min | 一次权重加载 |
| 7B 前向 1 token | ≈25–60 ms | batch 前向优化 |
| 7B batch 8 长 512tok 生成 | ≈45–90 s | n=6 一次请求 |
| 每题完整 rollout（64 nodes） | ≈3–8 min | policy+valve 混合决定 |
| GRPO 每轮 P1（360 题） | ≈2–3 h | rollouts+3 epochs |
| TTT 每题（16 步） | ≈0.5–2 min | 单题内 |
| 每日流量 | 4–8 GPU-h | ≤18 小时/month OK |
