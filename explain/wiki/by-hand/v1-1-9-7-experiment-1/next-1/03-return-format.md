# 03 — 环境与结果回填模板

## 环境表（必填）

| 项 | 你的值 |
|---|---|
| 容器 | hostname / tailscale ip |
| GPU | rocm-smi（型号/显存） |
| Python | torch/transformers/peft 版本 |
| Lean | `lean --version` |
| 网络 | hf-mirror ✓/✗；github ✓/✗ |
| artifact sha | full.pt/raw.pt 校验结果 |

## 结果表（必填）

| 实验 | 参考均值（本容器） | 你的值 | 方向一致？ |
|---|---|---|---|
| E1 IMO/Pell×2 value | 1.786 / 1.962 / 3.216 | | |
| E2 ΔV | +7.3 / +8.8 / +7.4 | | |
| E3 spread | +4.6 / +2.4 / +3.0 | | |
| E4 distinct / JaccardMed | 7-11 / 0.00-0.27 | | |
| E5 unique | 2→4→6 | | |
| 闭环 ΔV(证据注入) | +3.2~+5.4 | | |

## 差异说明（可选）
- 硬件/出口/引入改动 → 一句话列出
- 异常与规避方式

## 常见问题速查
| 现象 | 对策 |
|---|---|
| torch 加载失败（weights_only） | 云端 runtime 已修；确认你在用 `gpu/gpu_runtime`（不是旧 sync 副本） |
| artifact sha mismatch | 用 public HF 下载文件（不是 JSON wrapper）；解包见 01 P2 |
| value 恒 -1000 | policy/value 响应嵌套解析（`choices[].message.content`）——用 driver/runtime_transport.py |
| E5 温度无变化 | 与 read 温度（reap.temperature）区分：请求级 temperature 仅本实验提示层的扫描 |
