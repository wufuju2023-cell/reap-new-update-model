# 6. 模型生成的精确证明

下列 tactic block 来自实际 `proof.lean`，未人工续写：

```lean
theorem ReapCurriculumWrapper.attempt :
    _root_.ChallengeV1.H1Curriculum.CourseSquareTelescope := by
  unfold _root_.ChallengeV1.H1Curriculum.CourseSquareTelescope
  intro n
  simp [Finset.sum_range_succ, Finset.sum_range_zero]
  simp [Finset.sum_range_succ, Finset.sum_range_zero, pow_two,
    Nat.cast_add_one, Nat.cast_one]
  induction n <;> simp [Finset.sum_range_succ, *]
  linarith
```

它体现了课程想训练的完整组合：展开定义和量词、处理 `Finset.range` successor、归一化平方与 coercion、归纳并使用已有假设，最后线性算术闭合。成功路径共 5 个搜索动作，后 3 个由第二次题内更新后的 v2 生成。

原始完整文件为 [`results/evidence/proof/proof.lean`](results/evidence/proof/proof.lean)。固定 CPU 镜像、禁网验收结果为 accepted；原始请求、回执和 acceptance 分别在：

- `results/evidence/verification/request.json`
- `results/evidence/verification/receipt.json`
- `results/evidence/verification/accepted.json`

proof SHA-256：`7345d0ebdc495d2a633a1d51257b9b4017733bb9ee759ce9f830e14ba54a08cf`。
