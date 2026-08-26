# 01 · SSH 结论 与 类 SSH 通道

## 1.1 诊断结论（已反复验证）
- 实例内有 sshd：`/etc/dsw/runtime/bin/sshd -p 22 -D`（DSW runtime 自动，或 `apt-get install openssh-server` 可装 `/usr/sbin/sshd`）。
- **外网 22 不可达**：gateway 的 22 / 2129695 / PAI 子域名 `dsw-<id>.ds...aliyun.com:22` 全部 `Connection refused/False`（从 zhai 与 yang-win 都测过）。→ **平台容器层无公网 SSH，非配置问题**。
- 无凭据访问网关被重定向到 `account.aliyun.com/login`（检查认证）：`{"code":302,"errorType":"REDIRECT"...}`。

## 1.2 三类"类 SSH"通道（按可靠性排序）

| 通道 | 形式 | 认证 | 状态 | 用法 |
|---|---|---|---|---|
| **① `dsw/commands` RPC** | `POST https://dsw-gateway-cn-hangzhou.data.aliyun.com/<SEG>/dsw/commands?type=status&name=next-ide` body `{"command":"..."}` | 浏览器会话 Cookie（credentials:include） | ✅ 多次成功 | 批量命令/读写文件/起服务/看日志（√ 主通道） |
| **② WebIDE / Jupyter 终端（WS xterm）** | `root@<pod>:/mnt/workspace#` 交互 shell | 浏览器会话 | ✅ 可交互 | 人工调试；**程序化按键不稳定**（Hermes canvas + CDP 间歇失效） |
| **③ opencli 封装** | `opencli browser <session> eval "(async()=>fetch(dsw/commands...))()"` | 常驻 Edge + 扩展 | ✅ 全自动（依赖浏览器挂后台） | 一条命令式 `dswterm "cmd"` |

## 1.3 网关段（SEG）获取
- 从 `GET https://www.modelscope.cn/api/v1/notebooks?Channel=dsw` → `Data.Notebooks[].Url.JupyterlabUrl` 中 `dsw-<数字>` 段；**实例每次重开段都会变**（实例 ID 同理，变化即现场取；不要硬编码入库）
- 实例 ID 亦会变（`dsw-nll7... → dsw-a8gk...`）；`/mnt/workspace` 的 NFS 数据跨实例保留，**ID/段只要重开就要重新查询**。

## 1.4 认证本质（为什么不能 curl）
- 网关/Session 认证 = **浏览器在 `dsw-gateway...data.aliyun.com` 域的 httpOnly Cookie**（`document.cookie` 读不到；CDP `network --detail` 不给 headers；opencli 无 cookies 提取命令）。
- 因此：**纯 curl 直调必须先用一次浏览器**（或拿到固定 token）。

## 1.5 彻底零浏览器（正路）
| 路径 | 条件 | 结果 |
|---|---|---|
| ① 个人云账号授权实例（PAI-DSW） | 控制台迁移/新建，**原生 SSH 面板**给 `ssh root@<host> -p <port>` | ✅ 一条 ssh 搞定，完全无浏览器 |
| ② 控制台固定 Token | DSW 控制台生成**长期 token**（非 authCode） | ✅ `curl -H "token..."` 直调 |
| ③ ~~gateway `/lab?appId=MAAS&instanceId=…` 版~~ | 用 lab URL 打开网关页再 fetch | ⛔ **已停用**（2026-08-26）；统一改用 workspace 页（`modelscope.cn/code/workspace`）作宿主 | 备用不启用 |

> 决策：**优先走 ① 或 ②**（用户只需在控制台复制一条 SSH/token）；这与"整个 v1 训练控面"解耦，之后无需浏览器。
