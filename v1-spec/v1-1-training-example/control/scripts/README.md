# dswctl — 云实例一键操控

把「操控 ModelScope DSW 实例」封装成几行函数（类 SSH）。底层走 `dsw/commands`（浏览器 httpOnly Cookie 由 opencli 代发），**网关段自动发现并缓存**。

## 快速开始
```bash
# 一次性: 放进 ~/.bashrc 或每次 source
source new-update-model/v1-spec/v1-1-training-example/control/scripts/dswctl.sh

dsw-refresh          # 首次: 重新发现网关段(SEG)并缓存(每次实例重开都要一下)
dsw "uname -a"       # 实例内执行命令
dsw "nvidia-smi"     # 或 rocm-smi 查 GPU
dsw "tail -50 /tmp/policy.log"
dsw-open             # 会话失效时恢复 workspace 页
dsw-file-get /tmp/x.jsonl     # 拉实例文件 → out/
dsw-file-put local.py /mnt/workspace/v1/services/x.py
```

## 配置（环境变量，非敏感）
| 变量 | 默认 | 说明 |
|---|---|---|
| `DSW_SESSION` | `aebvmcpf` | opencli 浏览器会话名 |
| `DSW_SEG` | 缓存于 `~/.config/dsw/seg`（`dsw-refresh` 自动发现；实例重开必变，**不硬编码**） | 网关段 |
| `DSW_NAME` | `next-ide` | 网关 name 参数 |

## 依赖
- `opencli`（本机，Edge+扩展连接，daemon OK）
- `curl`、`jq`、`base64`

## 关键机制
- **段(SEG)自动发现**：`dsw-refresh` 用 `opencli eval` 调 ModelScope `notebooks` API，从 `JupyterlabUrl/TerminalUrl` 提取 `dsw-<数字>` 并写缓存 → 实例重启后一条命令更新。
- **命令编码**：命令 JSON → base64 → `atob` 于页面内 fetch（规避引号转义），返回 `{output}` 直接打印。
- **阈值**：单命令 <240s；长任务用实例内 `nohup ... &` 再轮询。

## 待办 / 已知
- opencli 在重前端页（Jupyter/Hermes）`eval` 偶发空白/115s 超时 → `dsw-open` + 重试；真不行时人工在 WebIDE 终端跑。
- 彻底零浏览器仍需「个人云账号实例 SSH」或「固定 token」（见 `01-ssh-and-channels.md`）。
