# 04. Lean/Reap 接入（V1-1b，P1 全链路的验证环节）

## 4.1 为什么需要

RTTT 的"奖励"必须来自**可证明性**（kernel-verified），否则就是奖励黑客：
- 仅靠 SFT 信号无真值；真实信号 = Lean `checkProof` / 执行错误。

## 4.2 构建（在实例内，分批断点）

```bash
curl -fsSL https://raw.githubusercontent.com/leanprover-community/elan/master/elan-init.sh | sh -s -- -y
source ~/.elan/env
cd /workspace
git clone --depth 1 https://github.com/frenzymath/reap.git       # reap 本体
cd reap && cat lean-toolchain                                      # 记录版本（v4.28.0-rc1 系）
lake build            # 第一次最重(自带 mathlib 依赖进 .lake/packages)
```
mathlib 编译说明：
- 采用 `lake build` 分 4 批（`Reap/Basic → Options → TreeSearch → Tactic` 依赖序），
  每批前 `touch state/lean_b<n>.done`，批间允许 >4min（队列 runner timeout 2800s 已覆盖）
- 磁盘需求：mathlib 源+构建产物约 8–12GB → 放在 /workspace（100GB NVMe）ok
- 若官方仓库拉取慢：git clone 走 `gh`/GitHub 直连并 `--depth=1`，改用分支镜像不设 gitee（后续可选代理镜像）

## 4.3 对接协议（Lean 侧 ↔ policy/value 服务）

```
set_option reap.policy_endpoint "http://localhost:8760/v1/chat/completions"
set_option reap.value_endpoint  "http://localhost:8760/v1/chat/completions"   # /value 路由复用同 port
set_option reap.ps_endpoint     "http://localhost:8760/premises"              # v1 用内置 LeanSearch 兜底或本服务 mock
```
- lean 侧每次调用即触发 /ttt_step 的**缓冲蓄水口（buffer）**；
- `/ttt_step` 返回 success 后，Lean 侧继续搜索——即"验证网络在每一个节点都对模型实施了训练"（测试时训练）。

## 4.4 度量与门禁

- `ttt_metrics.jsonl`：每步 loss、KL、hot-swap 延迟、buffer 大小；
- 门禁：**在 10 分钟内完成 ≥10 节点扩展**且发生 ≥5 次成功 TTT 步 = `V1-1 PASS`；
- 评测协议：每个给定 theorem 从**冻结基础权重**开跑 TTT（避免污染）；或同题报告
  `solve@B_{static}` vs `solve@B_{static}+TTT`（对照实验）。

## 4.5 完全 Runbook（按序，每步 ≤4 分钟）

```bash
# 0) bootstrap（新实例专步）
push runner.py bootstrap.sh; bash bootstrap.sh
# 1) env 环境确认
/opt/venv/bin/python -c 'import torch;assert torch.cuda.device_count()==4'
# 2) 数据
export HF_ENDPOINT=https://hf-mirror.com; huggingface-cli download ... --resume
# 3) pip deps (tuna)
/opt/venv/bin/pip install -r /workspace/app/requirements.lock \
   --index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 4) [NOT USED] SFT（4 卡 DDP）—— 停用：主线 0 训练，RTTT on-demand
#    （如需 P3 保险丝流程，需用户决策后再启用 train_sft.py）
# 5) 启动 servers（直载 REAL-Prover，0 长训）
nohup /opt/venv/bin/python /workspace/app/policy_server.py > /workspace/logs/policy.log 2>&1 &
# 6) RTTT 演练（模拟搜索回放）
/opt/venv/bin/python /workspace/app/rttt_demo.py --steps 10
# 7)（V1-1b）lean+reap 接入 → 真校验 → PASS 报告
```

## 4.6 风险与预案（诚实）

| 风险 | 预案 |
|---|---|
| HF 模型在 hf-mirror 无缓存（首次冷拉 15GB 慢） | 分 4 批 `--resume`；同时后台拉 |
| mathlib 构建超 4 分钟/不能续 | love4batch 拆；lean cache 存盘可二次利用 |
| 7B LoRA 训练在 4 卡上 OOM（seq>4096） | 缩 seq/组数；FSDP shard |
| TTT 单步太慢（4096 seq） | 降至 1024–2048；batch=1；仅上行更新 buffer 上限 32 |
| teacher API 当日 1pt 用完 | 变体生成排后 / 人工批量一次生成预缓出 |
