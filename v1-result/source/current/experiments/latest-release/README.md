# 原三进程发布探针：冻结复现入口

以下文件逐字节取自各自实际部署归档：

| 文件 | SHA256 |
|---|---|
| `latest_probe.py` | `c7b54691d0d6f4c56a57d2c2bcc12bc00d39f2ad141e05c4cc4023e7a149e989` |
| `campaign-plan.json` | `b704e6be60a12345b49bb259afea6904b3733ff2f9da093857f53f97a75c06cb` |
| `recover_consume.py` | `33641acded89106df38cbb1b5048d1b8f3b6bb6989602dd181d39738df442bbd` |

核心`source.zip`仍是188文件版本。探针另放于此，不修改历史执行代码。实际原运行完成两次训练、提交、发布和head选择，随后因一个服务的900秒控制等待超时而整体退出1；详见[原失败证据](../../../../evidence/current/latest-release-gpu/README.md)。该等待限制仍在冻结文件中，此入口用于复核和复现原机制，不能视为已通过的长期服务程序。

## 准备

先按[当前源码说明](../../README.md)解包并校验源码、提取完整证明数据，再执行其中的`mixed`步骤，得到自己的`learner-store`和R2。可使用新生成的R2 SHA，不要求拥有历史远端大快照。模型须为指定revision的已有REAL-Prover；不下载权重。运行环境需具备对应PyTorch/ROCm、Transformers和PEFT，并能同时驻留三份7B模型。

以下从`v1-result`目录运行；将路径替换为自己的绝对路径。`SOURCE_R2`取此前mixed结果的`release2.model_release_sha256`。来源合同必须与固定D8、KL100混合配置兼容。

```bash
SRC=/absolute/path/to/new-reap-source
DATA=/absolute/path/to/new-reap-data
MODEL=/existing/model/REAL-Prover
STORE=/new/run/mixed/learner-store
SOURCE_R2=REPLACE_WITH_YOUR_MIXED_RELEASE2_SHA256
OUT=/new/run/latest-release
EXP="$PWD/source/current/experiments/latest-release"

CMD=(python3 -B "$EXP/latest_probe.py"
  --model-path "$MODEL"
  --expected-base-sha256 2ff73d37f6f4edad02f5c2e67bdabeeecef97a187b4747834e6acdf648980839
  --source-release-root "$STORE"
  --initial-model-release-sha256 "$SOURCE_R2"
  --replay-dataset-root "$DATA/replay"
  --mathlib-dataset-root "$DATA/mathlib"
  --replay-dataset-sha256 b89032bee7d7d75449f5395e9c34c99ff538624bd21abc057ae4be76ae8441eb
  --replay-dataset-sha256 7975e11a24d74f326387e82fd4b8f9fc2489d1d570e44c17c5f8a03e1b9555ff
  --replay-dataset-sha256 179c51f5f802311f9d1f292902fb4f0d452159dbe33afd6e14d823cae948fa0a
  --replay-dataset-sha256 fae47cd667809fa30c7a87b782b98e2e4943ba47144ddc69eee930142552402e
  --append-replay-dataset-sha256 33a3be06d72cacd9dfa78c5e3f82465f3e1f6ccbf12506fad67fd423a93114ad
  --mathlib-dataset-sha256 633da6857da2df468f5d9e6c2aee30f0b424505d95ee4c94cbbfe575fd203b5d
  --mathlib-dataset-sha256 98c67e8f220e8f5a9227ce010a95792f838b10a9cee6f15a98f1eae695d788cb
  --mathlib-dataset-sha256 390db65cb72310385919dfe5207659c51614962420b9a5ff39ec14fe42d874d1
  --sampler-seed 0
  --output-dir "$OUT"
  --campaign-plan "$EXP/campaign-plan.json"
  --campaign-plan-sha256 b704e6be60a12345b49bb259afea6904b3733ff2f9da093857f53f97a75c06cb)
PYTHONPATH="$SRC" "${CMD[@]}"
```

默认不加载模型、不创建输出。确认参数后才执行下段；输出和外层日志目录均须全新。它会复制源CAS到新输出，再在副本执行两次真实更新及发布，不能拿它恢复已提交的原失败运行。

```bash
set -o noclobber
LOG="${OUT}.launcher"
mkdir "$LOG" || exit 1
PYTHONPATH="$SRC" "${CMD[@]}" --run >"$LOG/stdout.log" 2>"$LOG/stderr.log"
RC=$?
printf '%s\n' "$RC" >"$LOG/exit-code.txt"
```

## 结果与恢复边界

只有`report.ok=true`、`real_7B_GPU_gate_passed=true`、主进程实际退出0及两服务正常结束证据同时成立，才能计为原探针通过。本次历史运行不满足这些门。连接丢失或异常后保留输出，先核原进程、CP、发布与意图；不要重新运行来“补齐”同一训练。

`campaign-plan.json`只提供预约身份。本探针没有Lean搜索、不调用`ReplicaCollector.search`、不写证明成功/耗尽；预约保留pending。未知发布故障封锁沿用既有本地事务验收，本探针没有GPU故障注入。原安全snapshot及后续只恢复、不训练的验收独立记录，不能改写本运行的exit1。

## 只恢复，不重复训练

`recover_consume.py`是另一条入口。它要求原运行已确认失败、旧PID全部退出，且原目录仍有完整CP/CAS、原R1标准snapshot、原报告/命令/head/预约及退出收据；这些大文件不在小型交付包中。应在原资产所在Linux环境运行，保留`actual-command.json`所指模型和数据路径；本入口不迁移或改写历史路径。

下面的pin对应本次历史原运行。恢复自己的运行时，须替换成事先核验的对应pin，不能套用旧值。仅设置参数不会分配GPU；加`--run`才执行。`SRC`和`EXP`沿用上面的源码及本目录路径，`ORIGINAL`指包含`result/`和`worker-exit.json`的原运行根。

```bash
ORIGINAL=/existing/original-latest-run
RECOVERY=/new/run/latest-recovery
RECOVER=(python3 -B "$EXP/recover_consume.py"
  --original-run "$ORIGINAL"
  --output-dir "$RECOVERY"
  --original-report-sha256 611d46040fa26cacf393e771ea05f20d3f78221fa740c4e9ae246d08823a0354
  --head-sha256 a9051c124a2acada42cd8a5c49b3401f48c9758d8c3e233fb37947dc35400dc9
  --selection-sha256 24783e21fe7a4bffcbd179601182d8789d26db0f56b4b33910a9ec83f0e91f26
  --snapshot-name operator-safety-r1-20260828
  --snapshot-manifest-sha256 d944c8b7de5117b46f128cc4cbb7c0245321f5ed540a78848bdcc434dca62f2e
  --expected-r1-complete-sha256 157a429fd75352a1d64e94ba7b40a22b875b066defdacd39cca5d434764768bd)
PYTHONPATH="$SRC" "${RECOVER[@]}"
```

实际执行沿用独占外层日志和真实退出记录：

```bash
set -o noclobber
LOG="${RECOVERY}.launcher"
mkdir "$LOG" || exit 1
PYTHONPATH="$SRC" "${RECOVER[@]}" --run >"$LOG/stdout.log" 2>"$LOG/stderr.log"
RC=$?
printf '%s\n' "$RC" >"$LOG/exit-code.txt"
```

恢复分别执行HTTP完整restore和新R2初始化/推理，控制等待为1800秒，不调用训练、发布或learner恢复。本次真实恢复通过8门，主/外层与两子服务均正常退出；原失败和新恢复是两个阶段，旧actor在新进程完整恢复，见[独立恢复证据](../../../../evidence/current/latest-recovery-gpu/README.md)。
