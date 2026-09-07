# 04 — 实验批次成果与未决问题（E 系列 + 复现）

## 本机 E 系列（v1-1-9-7-experiment-1，全部落盘）
| 实验 | 结果 | 结论 |
|---|---|---|
| E1 稳定性 | 10× sd=0（1.786/1.962/3.216） | value 确定性端点（无采样） |
| E2 权重组扫描 | w=0.05→2.0 下同题 ΔV 恒定（+7.315/+8.839/+7.351） | 内容敏感；phi 属 Lean 层（未直通 GPU） |
| E3 措辞敏感 | spread=+4.614/+2.363/+2.999 | 证据/提示必须模板化 |
| E4 多样性 | n=16：distinct 7-11，JaccardMed 0.00-0.27 | 树内多样=GPU 保证（RULE-0 ✓） |
| E5 温度 | unique 2→4→6（0.4→1.6） | 温度=分布宽度旋钮（合理） |
| 后验闭环 | ΔV=+3.569/+5.359/+3.244 | **B 通道实证**（09 文档 B） |

## 协作者复现（100.68.136.61）新增发现
1. **E4 弱多样性题**：Pell-positive-growth n=32 仅 5 distinct（4 非空）；
   Pell-invariant 5 轮 JaccardMed 0.369 > 0.3 门 → **E4 全绿结论需要修正**（部分题分布窄）。
2. **REAL-Prover 上游变更**：官方 manifest 锁 digest `5e5e…`→`88cf…`
   （16 文件+15.25GB 全验，仅 lock 字段变）；配合 `compatibility.patch` 重算
   canonical fingerprint 得 `e878…`（artifacts 要求的值）。→ **需 pin base 版本或固化映射**。
3. **上游代码缺口**：`RuntimeTransport.policy` 仍是 `.text`（应答为 `message.content`）；
   `retire` 用 GET 只读收据不释放会话；session id 过长触发 400 门。→ 收口并 squash 入 main。

## 因此接下来的三个"测量前题"
- 修 E4 的"部分题分布窄"（先归因：提示/参数/模型几何？）
- base 版本 pinned（避免每次环境变动）
- transport 协议稳定（不能让"读错字段"污染测量）
