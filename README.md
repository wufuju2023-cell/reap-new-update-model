# reap-rsi（私有仓库）—— local-first 工件仓库

> 本仓库含私有环境配置（实例 ID/账号名/SESSION 等），**保持 private**；
> 公开方法论版本见同名 public repo（`reap-rsi-public`，已脱敏）。

## 定位
本地 laptop（F 盘 `/mnt/f/projects/reap-new-update-model`）是唯一源头；
云实例只是执行器；自建模板=环境；ModelScope=状态归档。

| 路径 | 内容 |
|---|---|
| `tools/amd_jupyter/` | 控制面：`runner.py` / `bootstrap.sh` / `amd_jupyter.sh`（WSL 侧 API 封装；实例 ID/SESSION 走环境变量 `AMD_INST`/`AMD_SESSION`，默认值保留本账户） |
| `tools/scan-secrets.sh` | 机密指纹扫描（pre-commit/CI gate）——**机密只允许在环境变量，禁止落盘/入 git** |
| `app/` | 训练管线：policy_server（RTTT）/ rttt_demo / restore / smoke / archive（train_sft.py 已 DEPRECATED——SFT NOT-USED） |
| `v1-spec/` | V1-1 spec：主线 = 0-long-train + on-demand RTTT |
| `environment-spec/` | 环境/状态/复用规范（05 为布局总纲） |

## 工作流（保持稳定，勿破坏）
1. 所有编辑在 F 盘仓库 → `git commit` → `git push origin master`（GHP 私有）；
2. 云实例：`bash bootstrap.sh` → 控制面在线；调用 `amd_jupyter.sh exec/out/push/cat` 自动闭环；
3. 收尾：`app/archive.sh`（ModelScope 归档 + sha256 清单）。

## 机密纪律
- Add Template/Profile 里的 Key 一律 `export AMD_*`/`MODELSCOPE_TOKEN` 环境变量；
- 提交前 `bash tools/scan-secrets.sh`，命中即修复再提交；
- 永不提交 `.env`、`*.key`、`*.pem`（见 `.gitignore`）。
