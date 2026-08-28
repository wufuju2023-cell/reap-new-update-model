# 五题实验的结果与核查依据

这组实验运行五道题，包含重试共八次尝试，实际完成11次模型更新。三道题得到通过独立Lean检查的证明，其中[第1题连续训练三次后完成证明](../../docs/current/04-实际实例与验收结果.md#same-tree-three-updates)，[第3题和第5题分别训练一次、两次后完成证明](../../docs/current/04-实际实例与验收结果.md#same-tree-other-proofs)。每次训练后都确认新版本生效，再接着原来的搜索继续。详细运行编号与计时见[五题原报告](../../docs/09-五题尝试与并发结果.md)。

| 已确认的内容 | 核对记录 | 原始文件位置 |
|---|---|---|
| 五题中三题完成证明，八次尝试均有结果记录 | [运行结果汇总](summary.json) | 归档中的 `final-local-audit.json`、各次尝试的 `result.json` 和 `online-result.json` |
| 第1题和第5题真实训练后参数有变化 | [参数快照检查](snapshot-audit.json)、[训练请求与快照的交叉核对](snapshot-wire-crosscheck.json) | 对应尝试的 `checkpoints/`；大型参数快照留在远端，本包保存读取快照后的检查结果 |
| 第1题和第5题更新后的模型参与了后续搜索 | [第1题请求与回执](wire-01r1.json)、[第5题请求与回执](wire-05r2.json) | 搜索事件记录 `observer.jsonl` 与原始请求记录 `transport/final-http-evidence.json` |
| 三份完整证明通过独立Lean检查 | 归档中的 `proof-check01/receipt.json`、`proof-check05/receipt.json`，以及第3题的 `proof-check/` | 同目录的证明文件、执行输出和容器检查记录 |
| 失败和重试记录完整保留 | [八次尝试的结果](summary.json) | `initial-output/` 保存初次五题；`retry-output/` 保存第1、5题首次重试；`retry05-output/` 保存第5题最终重试 |

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

初次五路运行遇到通信异常，第1题和第5题成功重试的搜索窗口没有重叠。参数检查覆盖实际首末快照，中间版本由连续训练回执的指纹关联。本组未重新执行完整7B基础参数哈希检查和状态恢复测试；对应检查在各自的独立实验中记录。
