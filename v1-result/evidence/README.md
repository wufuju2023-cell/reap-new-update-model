# 如何阅读和复核实验凭证

这里回答三个问题：模型有没有真正训练？训练后搜索有没有用新参数？最终证明是否成立？先读报告，只有需要复核时才解包原始数据。所有检查都可以从本地开始；不需要重新打开 AMD 实例。

## 最新五题结果：先看这里

第1题三次更新后成功、第5题两次更新后成功，证据集中在 [multiround/](multiround/README.md)。[summary.json](multiround/summary.json) 给出五题八次尝试的总表；其余文件依次验证实际更新、新版本继续搜索和最终证明。

下面根目录的 E/F 文件保留较早实验的结果，不是最新五题的总表；`acceptance-summary.json` 等历史 JSON 保持原字节，不因新结果改写。

## 1. 历史 E/F：结论与依据

| 按顺序查看 | 这个文件说明什么 | 怎样理解结果 |
|---|---|---|
| `acceptance-summary.json` | 各验收项的总表 | E 的真实同树 TTT 为 true；双题整批、GPU 容器、发布、能力提升仍为 false |
| `independent-execution-audit.json` | E 的搜索树、更新位置、最终证明重检 | 树节点继续增长；一次更新之后继续生成；无网络证明重检退出码为 0 |
| `independent-wire-e.json` | E 的 CPU 记录与远端原始 HTTP 是否对应 | 10 个 job，更新一次，后续 policy/value 为 v1，`gaps=[]` |
| `recurrence-snapshot-audit.json` | 真实训练前后快照中的张量检查 | E 的 196 个 LoRA 张量、4 个 value-head 张量改变，数值有限 |
| `wire-snapshot-crosscheck.json` | CPU、HTTP 和实际快照是否描述同一次更新 | 三处学习回执、事件身份、参数摘要应一致 |
| `independent-wire-f.json` | 并发 F 的训练链 | 5 次更新可核验，最终证明未通过，保留为 partial |
| `multi-round-ttt.json` | F 五轮的集中对照 | 每轮更新、optimizer step、新版本推理和参数摘要逐项列出；最终证明为 false |
| `f-empty-tactic-analysis.json` | F 为什么出现大量空候选 | 64 条候选中 53 条直接生成结束标记；深层原因尚未做对照实验 |
| `current-container-access.json` | 最后一次只读容器环境核查 | 当前未发现 Docker daemon/代理 socket；B 的构建与运行仍待具备条件的环境 |
| `source-checks.json` | 源码快照和本地交付工具的检查 | 仅覆盖源码完整性、解包和工具测试；GPU 容器仍需另验 |

E 的训练数据来自真实搜索的访问次数和回传值。学习发生时题目还未解决，随后同一棵树继续生成，最终证明通过。完整解释见[实验结果](../docs/03-实际TTT结果与证据.md)。

## 2. 需要原始记录时，解开证据包

`raw-evidence.tar.gz` 收纳本次 E/F 的原始记录；`raw-evidence-manifest.json` 给出压缩包及每个文件的大小、SHA256。它们不含模型权重。此前 A—D 的历史实验、重复回放副本和上传临时文件保留在原本地记录中，不随本包交付。

解包前先核对压缩包 SHA256；解包后逐文件核对清单。使用全新目录，避免和新实验输出混在一起。解包后的主要路径如下：

| 路径或文件模式 | 在实验的哪个步骤使用 |
|---|---|
| `http-evidence-ef.json` | 记录 E/F 的 82 个 HTTP job；包含原始请求、结果、响应字节及摘要，是线上版本审计的输入 |
| `cpu/batch-config.json` | 冻结批次配置、源码身份和两个题目；复核时确认没有中途换题或换实现 |
| `cpu/.batch-intents/*.json` | 记录每路实验已开始，帮助恢复程序阻止重复训练 |
| `cpu/solutions.jsonl` | 只收录已验证的 E 结果；F 未证，不能添加为成功 |
| `cpu/online-20260827-{e,f}/session.json` | 会话名、题目、源码摘要、接口与 gamma；复核各文件是否属于同一实验 |
| `.../process.json` | 记录本次搜索启动命令和 launcher PID；没有独立历史进程采样证据 |
| `.../observer.jsonl` | 按顺序记录选择、生成、验证、回传、检查点与版本，用于重建同树流程 |
| `.../wall_clock.jsonl` | 连接生成请求、模型响应与计时；用 response ID 对照 HTTP 原始记录 |
| `.../checkpoints/*.request.json`、`*.receipt.json` | 一次学习请求及其真实回执；记录训练样本、版本、损失和参数摘要 |
| `.../checkpoints/*.ack.json` | 学习完成后的放行记录，让同一搜索继续；要配合 HTTP 和快照一起核对 |
| `.../raw_tree.json`、`progress.jsonl` | 最终树和搜索进度，用来理解题目走到了哪里 |
| `.../result.json`、`online-result.json` | Lean 结果、proof_script、更新次数及执行状态，是结果汇总的输入 |
| `.../create-receipt.json`、`snapshot-before.json`、`snapshot-after.json` | 会话创建及快照保存回执；实际快照张量检查见上面的审计 JSON |
| `.../stdout.log`、`stderr.log` | Lean 的输出与错误信息，辅助检查 F 耗尽预算等情况 |
| `proof-recheck/*.lean` | E 的原题和最终证明重检文件，可在相同 Lean 环境独立编译 |
| `proof-recheck/proof-recheck.log` | 无网络重检结果，含公理依赖检查 |
| `proof-recheck/*inspect.json` | 证明重检所用 CPU 镜像、用户、网络模式等容器信息 |

## 3. 如何重新做本地 HTTP 审计

先按[source 说明](../source/README.md)校验并解出源码，再解出上面的证据。下面两个路径需替换为自己的目录；输出文件应尚不存在。

```bash
source_root=/absolute/path/to/extracted-source
evidence_root=/absolute/path/to/extracted-evidence
python3 "$source_root/tools/amd_jupyter/audit_online_wire.py" \
  --collection "$evidence_root/http-evidence-ef.json" \
  --cpu-session-dir "$evidence_root/cpu/online-20260827-e" \
  --output /absolute/path/to/new-e-audit.json
```

预期为 `MATCHED_EXECUTION_WIRE`、一次 learn、`gaps=[]`。换成 F 目录时，预期为 `MATCHED_PARTIAL_WIRE`：HTTP 链可核验，证明未完成。

这一步只读取本地文件，不发请求、不触发 GPU 更新。实际张量检查是在远端读取保存的快照后得到的，本地包保留检查结果和摘要，未复制大体积张量文件。

`independent-execution-audit.json` 保留首次审核的原始字节，其中还列有当时重复回放文件的哈希。当前交付包的实际文件范围，以 `raw-evidence-manifest.json` 为准；复核所需的原始会话记录都在 `cpu/` 中。
