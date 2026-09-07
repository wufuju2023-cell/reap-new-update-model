# code — scaling-plan-1 训练管线脚本

3 个核心脚本（2026-09-07 在 MI300X/容器上实测；受 GPU 特征预计算管线约束）。

| 脚本 | 作用 | 复现入口 |
|---|---|---|
| `mix_train.py` | 64-bin value head 训练（可断点续传、COMPLETE 门、registry 输出） | `python3 mix_train.py --run-id mix4 --oversample 2` |
| `make_features.py` | 从 `dataset/train.jsonl` 生成 hidden 特征缓存（200 行/片，断点续传） | `python3 make_features.py` |
| `e1_eval.py` | E1 calibration 复测（223 样本、ρ/mean/maxerr） | `python3 e1_eval.py <run-id> <head-pt>` |

## 运行契约（各脚本内部已实现）
- 全部产物落持久卷 `/mnt/workspace/head-runs/`（runs/<id>/{config,progress,head,report,COMPLETE}）
- 训练依赖：7B base（frozen）、LoRA adapter + 64-bin head（HF artifact backend.json 解码）、
  特征 shards（v2 manifest；extension 段为 v3 单独加载）
- E1 复测依赖：base + LoRA + head + `dataset/validation.jsonl`（确定性 seed=7）
- 无外部网络调用（除 HF artifact 获取时）；运行时限 >240s 时用分段 + resume 驱动

## 数据约定
- `fullv3-train-features/manifest.json`：新生成特征（80,000 行，v2 分片）
- `features-79efd240/`：S18 家族特征（80k train / 8k val / 8k test，长尾 d≤38）
- `full-extension-features/`：extension 段（v3 schema；80k→205,628）

## 版本链
结果发布到 HF `WufuJu/reap-value-head-v1-scaling`（registry.json 记录 sha 链）。
回退 = `cp /mnt/workspace/head-runs/runs/<id>/value-head.pt <target>`。
