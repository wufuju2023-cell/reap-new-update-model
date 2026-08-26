# 03. 数据与课程（P3 + P2 并行）

## 3.1 数据下载（hf-mirror，断点续传）

```bash
export HF_ENDPOINT=https://hf-mirror.com
/workspace/venv/bin/pip install -q huggingface_hub
# 模型（~15GB，分 4 批 w/ -c 语义: hf download 自带断点）
huggingface-cli download FrenzyMath/REAL-Prover --local-dir /workspace/data/real-prover --resume
# 数据集（~1.2GB）
huggingface-cli download FrenzyMath/state_tactic_pairs --repo-type dataset \
  --local-dir /workspace/data/pairs --resume
# FATE-M
git clone --depth 1 https://github.com/frenzymath/REAL-Prover.git /workspace/repo-realprover
cp /workspace/repo-realprover/Realprover/data/fate_m.jsonl /workspace/data/fate_m.jsonl
```
> 每批 `touch state/dl_<k>.done`；重跑幂等（-c/--resume）。

## 3.2 清洗与过滤（对齐 §04《Data》）

1. 解析/执行过滤：配对 (state,tactic) 需**能被 Lean 解析并执行**（v1 无 Lean 时用 pickle 预加载过的合法性 flag——先 50k 原样进 SFT，标注照搬）；
2. 去重：state 的 pp 哈希（对齐 Reap 的 StateKey）；
3. 错误负样本：失败 tactic 抽样 10% 保留（供 value head 负例）；
4. 输出 schema（每行 jsonl）：

```json
{"id": "...", "state_pp": "...", "context": ["lemma1_stmt", "..."],
 "tactic": "...", "logprob_old": -12.3, "solved": false, "error": "parser"}
```

## 3.3 教师模型分工（P2，用 Radeon Token Factory）

- 角色：对难题 $p_*$ 产出**变体/提示/难度标注**；frozen（不参与学生梯度）；
- 调用：OpenAI 兼容 `POST /radeon/api/v1/chat/completions`，模型从
  `GET /radeon/api/v1/models` 取（每日 1pt = 足够出 400–600 条变体）；
- 课程（分层）：输出

$$\mathrm{Diff}(v) = 1-\mathrm{solve@}B_{\mathrm{low}}(\pi_g, v),\quad \mathrm{keep}\ \mathrm{Diff}\in[0.5,0.9]$$

并按 $\mathrm{Sim}(v,p_*)\ge 0.7$ 过滤（AST 子串比），栈入 `ladder/<p>.json`，一次性给学生做"阶梯"。
- 台账：`out/teacher_calls.jsonl`（prompt-only；本 spec 不许把 API Key 写进 logs/文件——Key 出自 Profile 页环境变量注入）。

## 3.4 每日运行 cadence（student 循环）

```
T0    下载(批)          -> state/dl_*.done
T1    SFT 阶段1(12.5k)  -> ckpt-sft-1
T1.5  value head init   -> v_head.pt
T2    policy_server+value_server 起（LoRA hot）
T2.5  RTTT 直播（buffer>=k 更新）——截图指标每 60s 落 out/rttt_metrics.jsonl
T3    复用 checkpoints 再训（第 2 epoch，优先用更高质量样本）
```
