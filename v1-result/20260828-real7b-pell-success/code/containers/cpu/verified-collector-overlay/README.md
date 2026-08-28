# 独立成功轨迹profile的固定release collector

这是显式新增入口，不替换旧 `reapTrainingMCTS`/`online_ttt`。新Python collector为原题生成单独的 `run.lean`，导入 `Reap.VerifiedCollector`；原题和导出证明保留原字节关系，供既有成功轨迹replayer验收。

## 运行

本地增量镜像 `localhost/reap-cpu:verified-collector-20260828`，ID `d2070a64912f1a7c66ec9e4bb92e66004e8adaa58887e7be333f0bf8e3d5f3b7`。它基于已固定的selection镜像，仅加入Lean模块，不更新镜像内Python。构建定义见[Containerfile.verified-collector](../Containerfile.verified-collector)。运行时另外提供固定源码hash的完整 `cpu_runtime` 包，通过 `PYTHONPATH` 明确使用该副本。

```bash
python3 -m cpu_runtime.verified_collector \
  --session-id actor-new-01 \
  --project-dir /opt/reap-runtime \
  --theorem-file /inputs/first.lean \
  --theorem LearnerLoop.AffineStrideThree \
  --output-dir /out \
  --gpu-base-url http://HOST:PORT \
  --model-release-sha256 EXPLICIT_RELEASE_SHA256 \
  --puct-value-gamma 0.99
```

题源要求以 `import ReapRuntime` 开头、包含一个两空格缩进的 `reapTrainingMCTS` 标记。输出目录中的session子目录必须不存在；不支持自动重试未知结果。`return_discount`固定1；PUCT gamma在(0,1)且为Reap可表示的千分位数。

创建回执先验证actor角色、分类价值语义、完整learner release来源及local v0/空优化器计数和事件buffer。新Lean ready握手完成前不请求模型。每个checkpoint和最终证明checkpoint固定v0，禁止更新或刷新；HTTP正distance由已有Generator取负一次，新模块只校验范围。

## 结果边界

`solved_pending_independent_verification`表示本次Lean进程完成根证明与精确成功路径导出，**不表示独立证明验收通过**。`proof.lean`仍需独立断网Lean和既有 `export_verified` 逐状态重放。最终成功包仍用原17文件schema；新actor的profile、role、lineage和完整create回执嵌入session.json并受包hash约束。ready事件保留，不伪装旧TTT。

已知预算耗尽需完整终态/ACK且Lean exit1；异常退出、错误身份、缺失事件或未知create回执均为 `failed_unknown`。collector不自动learn、snapshot、retire或清理GPU session；调用方在持久化和核对终态后显式管理驻留资源。

本地验证记录见[summary.json](../../../.downloads/verified-collector-local-20260828/final/summary.json)：16项新Python机制测试包含在67项联合测试中；实际Lean+loopback mock覆盖OR/AND、距离变号、gamma分工、ready/终局错误与完整17文件导出/安装。测试fixture首次编译失败保留，修复后另有实际成功记录。没有用这些结果声称GPU训练、泛化或提速通过。
