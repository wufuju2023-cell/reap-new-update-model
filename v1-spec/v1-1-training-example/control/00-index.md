# control · 实例控制面 & 浏览器自动化笔记

> 本目录持续记录：**192G DSW-AMD 实例的访问/控制方案、SSH 结论、opencli 浏览器自动化经验、令牌/token 探测、坑位**。日后任何浏览器自动化相关内容都追加到这里（`control/`）。

## 0. 快速索引
- [01-ssh-and-channels](01-ssh-and-channels.md) —— SSH 结论 + 三种类 SSH 通道 + 认证本质
- [02-opencli-automation](02-opencli-automation.md) —— opencli 驱动浏览器自动化的做法/坑
- [03-token-probe](03-token-probe.md) —— `ms-` 令牌探测结论（待确认来源）

## 1. 当前实例（快照 2026-08-25）
- 实例：ModelScope 代码工作区 **DSW-AMD**（免费/同性质，非个人云账号）
- 实例 ID：**以现场为准**（workspace 每次重开会**变化**；「我的 Notebook」页可查）
- **网关段**：**以现场为准**（`dsw-refresh` 自动发现并缓存；见 `scripts/dswctl.sh`）
- 规格：23 vCPU / 200GB RAM / AMD GPU 192G；镜像 `ubuntu22.04-rocm7.2.3-py312-torch2.11.0-1.39.0`
- **单次实例时长**：~7.7h/次（页面显示）；自动关闭：闲置>1h；/mnt/workspace 与 /mnt/data 为 **NAS 持久挂载**
- 数据根：`/mnt/workspace/v1`（models/ reap/ v1project2/ mathlib-src/ services/ runs/）

## 2. 关键结论（务必记住）
1. **普通 `ssh`（22 端口）：实例内可有 sshd（`/etc/dsw/runtime/bin/sshd`），但公网/容器层无入口 → 不可行**（平台限制，与本实例性质无关）。
2. **"彻底零浏览器"无官方途径**：网关认证 = httpOnly Cookie，无法导出给 curl。
3. **实际可用的类 SSH（已验证）**：
   - `dsw/commands` RPC（WebSocket/HTTP）+ WebIDE 终端（`root@dsw-...` 交互 shell）
   - 二者都需**浏览器登录态常驻**（opencli + Edge）。
4. **真正告别浏览器的正路**：① 迁移到「个人云账号授权实例（PAI-DSW）」→ 控制台给 `ssh root@<host> -p <port>`；② 或 DSW/控制台生成**固定 token**（非一次性 authCode）。

## 3. 存储布局约定（用户指定）
- 本机 Windows 部分 → **D 盘**；本机大文件/模型权重 → **E 盘**（C 空间不足）
- 服务器：需持久数据一律放 **`/mnt/workspace`**；超大权重放服务器 NFS（本目录只存 SHA256+路径）
- 同步中心：**`v1-spec/v1-1-training-example/`**（服务器 ↔ WSL ↔ GitHub 三方回环）
