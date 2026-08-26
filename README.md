# reap RSI — local-first 仓库（F 盘 / GitHub 私有）

本地 laptop 是唯一源头；云实例只是执行器；模板是环境；ModelScope 是状态归档。

| 路径 | 内容 |
|---|---|
| `tools/amd_jupyter/` | 控制面：`runner.py` / `bootstrap.sh` / `amd_jupyter.sh`（WSL 端） |
| `app/` | 训练管线：train_sft / policy_server / value_server / rttt_demo / restore / smoke / archive |
| `environment-spec/` | 环境与状态布局（05） |
| `v1-spec/` | V1-1 实施 spec（teacher × reap-7B student + RTTT） |
| `plan/` | 早期研究计划 |

## 断点续传 / 幂等约定（AGENTS 全局规则）
- 任何多批任务先建 `state/` + `batch_<id>.done` 标记；
- 下载一律 `--resume` / `-c`；写产物用临时文件 + `mv`；
- 单条命令 ≤ 4 分钟。

## 恢复剧本（新实例 → 续接，<10 min）
```
Launch(自建模板 reap-pytorch) → push runner.py bootstrap.sh → bash bootstrap.sh
→ git clone https://github.com/wufuju2023-cell/reap-new-update-model /workspace/app-src
→ cp app-src/app/*.py app/*.sh /workspace/app/  → bash app/restore.sh --from modelscope:reap/rl-v1
→ bash app/smoke.sh  → 续训
```
