# 03 · 令牌（Token）探测记录

## 探测对象
- 令牌样本：`ms-<REDACTED>`（`ms-` + 36 位 UUID；**已脱敏，完整串仅存在于本地安全上下文，不入库**）
- 目的：确认它能否作为「免浏览器控制 DSW 网关」的凭据。

## 探测矩阵（全部失败）
| 端点 | 携带方式 | 结果 |
|---|---|---|
| `www.modelscope.cn/api/v1/notebooks?Channel=dsw` | `Authorization: Bearer <t>` | **401** |
| 同上 | query `?access_token=<t>` | **401** |
| 同上 | header `X-Access-Token: <t>` | **401** |
| `dsw-gateway.../dsw-&lt;SEG&gt;/dsw/commands` | `Authorization: Bearer <t>` | **302 → account.aliyun.com/login** |
| `dsw-gateway.../api/status`、`/api/me` | `Authorization: token <t>` | **302 → login** |
| `dsw-gateway.../lab?token=<t>`（含 `appId=MAAS` 组合） | query `token` | **302 → login** |

## 推断
- **不是** DSW 网关 / ModelScope Notebook API 的凭据（网关只认 httpOnly Cookie；ModelScope 该 API 也不接受此 Bearer）。
- 更可能是 **ModelScope 平台的个人 Access Token**（SDK 读写 **模型/数据集/创空间** 用），与本批次「开 Notebook shell」无关。
- 但**尚不能下最终结论**——需用户确认来源页面。

## 待确认（用户提供）
1. 该令牌在**哪个页面**复制/生成？
   - (a) 账号设置 → Access Token？（→ SDK/模型数据集用途）
   - (b) DSW / Notebook 实例详情页的「访问码/Token」？（→ 可能控制实例）
   - (c) JupyterLab 顶部 token？（→ 需配合 `appId=MAAS` 与正确 lab 根）
   - (d) 其它「令牌/Token」按钮？
2. 若确为 (a)：正确用途 = ModelScope SDK 上传下载（与本项目免浏览器控制无关），免浏览器开 shell 需改走「个人云账号实例 SSH」或「固定 token」。

## 安全提醒
- `ms-` 令牌属敏感凭据：**不写入 git/日志/仓库**；探测仅在本会话命令行临时使用。
- 若发现其对数据集/模型有写权限，请尽快在 ModelScope 设置中**限于最小权限或轮换**。
