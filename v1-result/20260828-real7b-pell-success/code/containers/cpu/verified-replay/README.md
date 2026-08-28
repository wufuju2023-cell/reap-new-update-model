# 独立成功轨迹重放

这是可选的本地数据入口，不启动搜索、GPU 或训练，不改变
`search_visit_backup`、0004 或默认 CPU 镜像。先复用已有完整定理的独立 Lean
验收，再用保存的成功路径重新运行 Lean，逐动作核对真实目标状态。

## 验收边界

- OR 按 Reap `solvedOrChild?` 的数组顺序选第一个已解非 focus 边；不能自行选较短边。
- AND 必须包含全部按 `focusIndex` 排序的子目标。标准 focus 子树的 `isPartial=True`
  是合法的局部目标标记，只允许沿实际 focus 边进入；孤立 partial 拒绝。
- 策略数据只包含原始 `generation` 与成功 `created/eval.ok` 能对应的 OR 动作。
  合并候选不能替换最初创建边的 tactic。失败尝试不进入数据。
- 终局 `isSolved` 不能证明正确性：必须有空目标、完整证明字节与历史验收对应，
  且新 Lean 重放的每一步状态、动作、返回值和最终定理检查均通过。
- 本版本要求单目标 OR、独立 AND、向后追加的树结构，不支持共享子树。
  含 metavariable 依赖的 AND、被兄弟分支提前解决的目标、状态打印不一致一律拒绝；
  不猜测标签、不自动重试、不重新搜索。

## 返回值与论文的关系

`verified-generated-action-negative-longest-branch-v1` 每个原生成 tactic 字符串
计一次动作（即使字符串包含多个 Lean tactic），奖励为 -1；终局为 0；focus
转换不计动作；AND 取各分支返回值的最小值。因此生成四个动作、其中两个分支
长度为一和二时，根值是 -3，不能写成 -4。

这借鉴 [AlphaProof Methods](https://www.nature.com/articles/s41586-025-09833-y)
的成功轨迹和最长分支返回值原则。该入口不实现论文的完整 TTRL、任务课程、
训练混合比例或分类价值头，也没有把负返回值转换成现有 gamma/sigmoid 搜索目标。

## 本地运行

在已有 Lean 4.28.0-rc1 CPU 环境内执行，工作区须可读取；`--output` 必须是新目录。
推荐由 `--network none` 容器外壳执行并保存容器 image/网络/退出码记录。

```bash
python3 -m cpu_runtime.verified_trajectory \
  --session-dir /inputs/session \
  --source /inputs/Theorem.lean \
  --proof /inputs/proof-check/TheoremProof.lean \
  --proof-receipt /inputs/proof-check/receipt.json \
  --theorem Namespace.theorem \
  --lean-project /opt/reap-runtime \
  --output /outputs/new-verified-trajectory
```

源文件须以 `import ReapRuntime` 开头且有唯一 `  reapTrainingMCTS` 插入位置；
此限制使被历史独立验收的完整证明与新生成证明能精确对应。独立模块代码直接嵌入
新的 `replay.lean`，不需要改 Lean overlay、默认镜像或已经运行的搜索程序。

通过后才以完整文件发布 `dataset.json`，并保留输入、plan、逐步 trace、进程 stdout/stderr
和 receipt。失败保留证据且不发布 dataset；输出已存在时拒绝，人工选择新目录后才可新验收。

## 后续学习器如何读取

先把 `dataset.json` 的 SHA256 固定在可信运行清单，再调用：

```python
from cpu_runtime.verified_trajectory import load_verified_dataset
dataset = load_verified_dataset(directory, expected_sha256=pinned_digest)
```

加载器重新校验冻结输入、trace、receipt、stdout/stderr 的 hash，并从树和 observer
重新提取路径；再比较完整证明、Lean 退出码、公理白名单、实际重放和数据标签。
`rows` 包含 `state/next_state/tactic/prompt/return/node_index/child_index` 与产生动作的
policy 版本及 observer 序号。只允许非终局 OR 样本，不生成猜测成功标记。

Hash 用于绑定内容和发现损坏；不能证明一个被攻击者连同全部证据和 pin 一起替换的
本地文件是真实运行产物。必须保存原始执行证据和可信内容 pin。
