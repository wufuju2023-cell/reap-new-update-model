# v1-1 · Training Example（Checkpoint 归档 · 单一同步中心）

**本目录是"服务器 ↔ WSL ↔ GitHub"三方同步的主归档点**。

## 同步拓扑

```
192G DSW-AMD 实例 (/mnt/workspace/v1)  ←───（任意运行）───→  运行中 checkpoint
        │ 同步① 服务器→本目录（拉取示例/脚本/摘要/轻量产物）
        │ 同步② 本目录→服务器（git 脚本拉取即用）
        ▼
本目录 (repo: new-update-model/v1-spec/v1-1-training-example)
        │ 同步③ git add/commit/push
        ▼
GitHub: wufuju2023-cell/reap-alpha-proof
```

## 目录约定（新增文件必须符合）

| 子目录 | 内容 | 大小上限 |
|---|---|---|
| `env/*.md` | 服务器/本地环境清单（lean 版本、py 包、镜像、实例信息、网关段） | 无（文本） |
| `scripts/*` | 服务端脚本（policy_server/ps_server/runner 等）**以 git 为准的可复制脚本** | <500KB |
| `runs/*.jsonl` | 运行日志/摘要（solve@B、rollout 摘要、diff 指数），gzip | <10MB/文件 |
| `checkpoints/*.json` | checkpoint 元数据（adapter 的 `config.json`、metrics.jsonl、SHA256、下载地址/路径），**不存大权重** | 元数据 |

**大文件规则**：权重（≥10MB）、lean tar 等存**服务器 NFS**（`/mnt/workspace/v1/archive/`），本目录只存其 SHA256 + 路径 + 恢复命令。

## 运行中随时 checkpoint 规程（每次关键操作后）

1. 服务器：命令执行前后将关键状态追加至 `/mnt/workspace/v1/runs/checkpoint.jsonl`（一行一个对象：`ts, action, status, outputs_sha, next`）；
2. 服务器→本目录：定期（或每个 milestone）执行
   `bash sync/tar_remote_light.sh`（在 192G 实例里把 `runs/*.[jsonl|json|md]` + `scripts/*.py` 打小包 <8MB 经网关 base64 回流，详见下方工具）；
3. 本目录→GitHub：`git add new-update-model/v1-spec/v1-1-training-example && git commit -m "ckpt: ..." && git push`；
4. 服务器侧恢复：`git clone https://gh-proxy.com/https://github.com/wufuju2023-cell/reap-alpha-proof.git && cp -r new-update-model/v1-spec/v1-1-training-example/scripts /mnt/workspace/v1/services/`。

## 一键工具（wsl/yang-win 侧）

- `sync/pull_remote.sh`：经 yang-win `ms-exec` 从实例拉取轻量产物（base64 分段）+ 解包到本目录。
- `sync/push_remote.sh`：把本目录 `scripts/` 打包编码，用 ms-exec 写回实例（<2MB/次）。
- 步骤保证幂等：先查 `runs/.last_sync` 时间戳与服务器 `checkpoint.jsonl` 行数，增量同步。
