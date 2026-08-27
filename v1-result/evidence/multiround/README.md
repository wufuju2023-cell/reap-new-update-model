# 五题证据：按三个问题复核

主要结果是第1题三次更新后证明成功，第5题两次更新后证明成功。完整解释与时间口径见[五题报告](../../docs/09-五题尝试与并发结果.md)。

| 想确认什么 | 先看什么 | 原始文件在哪里 |
|---|---|---|
| 一共跑了哪些题，哪些成功 | [summary.json](summary.json) | 归档中的 `final-local-audit.json`、各 session 的 `result.json` 和 `online-result.json` |
| 模型是否真的更新 | [snapshot-audit.json](snapshot-audit.json)、[交叉核对](snapshot-wire-crosscheck.json) | 各 session 的 `checkpoints/`；实际大型参数快照留在远端，本包收录读取快照后的检查结果 |
| 更新后的模型是否参与后续搜索 | [第1题通信核验](wire-01r1.json)、[第5题通信核验](wire-05r2.json) | `observer.jsonl` 与 `transport/final-http-evidence.json` |
| 证明是否成立 | 归档中 `proof-check01/receipt.json`、`proof-check05/receipt.json` | 同目录 `.lean`、输出和容器检查记录；第3题在 `proof-check/` |
| 失败是否被保留 | summary 中八次尝试 | `initial-output/` 全五题，`retry-output/` 第1题与第5题首次重试，`retry05-output/` 第5题最终重试 |

## 解包和核验

`raw-evidence-manifest.json` 列出压缩包及每个成员的大小和 SHA256。先核对哈希，再解到新的目录。归档有241个文件，保留本次五题与同题重试的必要记录；不含模型、登录资料或其他实验。

解包后，`inputs/` 是可重新运行的五道原题，证明入口为 `reapTrainingMCTS`。`proof-check*/` 中的文件只用于事后核验，不能拿来当搜索输入。实际快照检查脚本在 `tools/audit_snapshots.py`，其中远端路径对应本次历史运行。

重新做第1题的本地通信核验（变量填写解出的实际目录）：

```bash
: "${SOURCE_DIR:?directory extracted from source/source-snapshot.tar.gz}"
: "${EVIDENCE_DIR:?new directory extracted from this raw-evidence.tar.gz}"
python3 "$SOURCE_DIR/tools/amd_jupyter/audit_online_wire.py" \
  --collection "$EVIDENCE_DIR/transport/final-http-evidence.json" \
  --cpu-session-dir "$EVIDENCE_DIR/retry-output/sessions/multi-20260827-01r1" \
  --output "$EVIDENCE_DIR/new-wire-01-audit.json"
```

预期：`MATCHED_EXECUTION_WIRE`，3条学习回执、12组生成、`gaps=[]`。只读取本地文件，不连接GPU。第5题改用 `retry05-output/sessions/multi-20260827-05r2`，预期2条学习回执、4组生成。

第2、3题的 `wire-02.json`、`wire-03.json` 使用归档中 `transport/initial-http-evidence.json`，保留其原始集合哈希；最终集合额外包含重试。证明重验见[02复现流程](../../docs/02-运行流程与复现.md)和[04新agent prompt](../../docs/04-新agent复现prompt.md)。

初次五路运行含通信异常，成功重试的两路搜索不构成同时成功训练的证据。参数检查覆盖真实首末快照，中间版本依靠连续回执指纹；未再次重算完整7B基座或执行恢复测试。各项边界见 summary 和09报告。
