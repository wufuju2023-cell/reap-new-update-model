# 固定 Mathlib 的人工 tactic 数据

独立入口是 `cpu_runtime.mathlib_trajectory`，不调用搜索、模型或生成轨迹 loader。
固定 Mathlib 提交 `5352afccd6866369be9de43f5b7ec47203555f44`，文件为
`Mathlib/Logic/ExistsUnique.lean`。第一版只接 `ExistsUnique.elim₂`、`intro₂`、`unique₂`。

动作单位为原文件中一条顶层单行 tactic。当前仅允许 `simp only`、`apply`、`exact`，
每一步必须恰有一个未解目标，最后一步必须结束证明。遇到多目标、嵌套、组合tactic、
提前结束或未结束均拒绝；不为不支持的分支估计返回值。线性路径的价值标签为
剩余原动作数的负值，与最长分支定义一致，终端不产生训练行。

每个新包依次完成原proof正常Lean检查、真实state捕获、独立state/action/return重放。
原声明源码与证明字节保留在新namespace中；额外检查声明类型相同、证明不直接引用
原目标定理、公理白名单通过。prompt由固定Reap的`mkPrompt(state, #[])`产生，明确没有检索。

在正式断网 CPU 镜像及已有 Lean 项目中执行（已有成功包不应重复生成）：

```bash
python3 -B -m cpu_runtime.mathlib_trajectory \
  --mathlib-root /opt/reap-runtime/.lake/packages/mathlib \
  --declaration ExistsUnique.elim₂ \
  --output /new/output/elim \
  --lean-project /opt/reap-runtime
```

`load_mathlib_dataset(directory, expected_sha256=...)`按22个固定文件无链接读取，完整验证
源commit/字节span/原proof、提取器和Lean模块hash、三次进程收据、trace和标签。
SHA pin必须由可信调用方提供。新profile为`mathlib_sft_linear_negative_remaining_actions_v1`，
来源为`mathlib_sft`，没有伪造generation事件；`source_policy_version=null`表示没有模型行为版本。
源码包同时保留Mathlib的LICENSE。失败不发布`dataset.json`，不自动重跑。

该接口只准备SFT数据。尚未接入90/10采样器、训练backend或GPU；包的有限样本不代表完整Mathlib训练。
