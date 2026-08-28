# 原题与课程设计

## 1. 完整原题

本次要证明的是 **Pell 方程 x² − 2y² = 1 有任意大的正整数解，而且两个坐标都能同时超过任意给定的界。**

完整题面：任意给定一个自然数 B，证明存在自然数 x 和 y，使得 **x 大于 B、y 大于 B，并且 x² = 2y² + 1**。这里自然数包含 0；由于 B 非负，这两个见证必然都是正整数。

$$
\forall B\in\mathbb N,\quad
\exists x,y\in\mathbb N,\quad
x\gt B\quad\land\quad y\gt B\quad\land\quad x^2=2y^2+1.
$$

例如 (3, 2)、(17, 12)、(99, 70) 都满足方程，但列举这些解还不够：必须对每一个 B 都给出符合三个条件的解。因此本题要求无界性，而不只是找到一组解或检查有限范围。

为避免公式显示问题造成歧义，下面同时给出未经改写的 Lean 原目标；本实验最终验收的就是这个声明：

```lean
def Target : Prop :=
  ∀ B : ℕ, ∃ x y : ℕ, B < x ∧ B < y ∧ x ^ 2 = 2 * y ^ 2 + 1
```

Lean原声明为`CodexMathFive.Pell.Target`，见[原始声明](inputs/original/StudentDeclarations.lean)。这是基于经典Pell方程组装的练习，不声称原创，也不是IMO1988Q6。原题没有被改成有限范围、具体上界或已给定序列的弱命题。

定义变换

$$
T(x,y)=(3x+4y,\ 2x+3y),\qquad P_0=(3,2).
$$

课程设计让学生先掌握保持方程、增长、构造序列和归纳性质，再学习如何把性质转为存在见证。**下面是实际成功证明的数学依赖，不是预先输入模型的目标证明脚本。**

## 2. 七个完整声明与实际证明

以下按最终证明依赖组织，不表示实验只跑过这七次。记序列 s 的第 k 项为 (xₖ, yₖ)，所有坐标和下标均为自然数。

### 1. Invariant：变换保持 Pell 方程

任取自然数 x、y，假设 x² = 2y² + 1，证明变换后的两个坐标仍满足同一方程：

$$
x^2=2y^2+1\quad\Longrightarrow\quad(3x+4y)^2=2(2x+3y)^2+1.
$$

训练的是多项式等式推理。[完整学生证明](proofs/01-invariant.lean)

### 2. PositiveGrowth：两个坐标严格增长

任取正自然数 x、y，证明 3x + 4y 大于 x，且 2x + 3y 大于 y。本课不假设 Pell 方程。

$$
(x\gt0\quad\land\quad y\gt0)\quad\Longrightarrow\quad
(3x+4y\gt x\quad\land\quad2x+3y\gt y).
$$

训练的是正性、线性不等式与合取。[完整学生证明](proofs/02-positive-growth.lean)

### 3. SequenceExists：构造满足递推的序列

证明存在序列 s：ℕ → ℕ × ℕ，其初值为 (3, 2)，而且每一步都应用上述变换：

$$
\exists s:\mathbb N\to\mathbb N^2,\quad
s(0)=(3,2)\quad\land\quad
\forall k\in\mathbb N,\quad s(k+1)=T(s(k)).
$$

即 x₀ = 3、y₀ = 2，xₖ₊₁ = 3xₖ + 4yₖ，yₖ₊₁ = 2xₖ + 3yₖ。这里只要求构造序列，还没有要求证明其无界性。[完整学生证明](proofs/03-sequence-exists.lean)

### 4. Recurrence：任意这样的序列都具有逐项界和不变式

任给序列 s，假设 s(0) = (3, 2) 且对每个 k 都有 s(k + 1) = T(s(k))，证明对每一个 k：

$$
x_k\gt k\quad\land\quad y_k\gt k\quad\land\quad x_k^2=2y_k^2+1.
$$

这同时证明两个坐标超过下标，以及每一项都满足 Pell 方程；需要对自然数下标组织归纳。[完整学生证明](proofs/04-recurrence.lean)

### 5. IndexedWitness：把逐项性质转为任意上界的存在见证

任给序列 s，假设对所有 k 都有 xₖ 大于 k、yₖ 大于 k、xₖ² = 2yₖ² + 1，证明：

$$
\forall B\in\mathbb N,\quad\exists x,y\in\mathbb N,\quad
x\gt B\quad\land\quad y\gt B\quad\land\quad x^2=2y^2+1.
$$

本课不再需要初值或递推假设。学生必须把给定的序列性质用于选取存在见证，题面不直接给出选哪一项。[完整学生证明](proofs/05-indexed-witness.lean)

### 6. UnboundedFromRecurrence：从给定递推序列得到无界解

任给序列 s，只假设 s(0) = (3, 2) 以及对所有 k 有 s(k + 1) = T(s(k))，证明：

$$
\forall B\in\mathbb N,\quad\exists x,y\in\mathbb N,\quad
x\gt B\quad\land\quad y\gt B\quad\land\quad x^2=2y^2+1.
$$

与第 5 课不同，本课没有直接假设逐项界和方程，而是通过已证明的 Recurrence 与 IndexedWitness 衔接。[完整学生证明](proofs/06-unbounded-sequence.lean)

### 7. Target：完整原题，不再假设已经有序列

对任意自然数 B，证明存在同时大于 B 的两个自然数 x、y，满足 x² = 2y² + 1。目标就是第 1 节的完整 Lean 声明：**没有给定序列，没有递推假设，也没有有限上界限制。**

本次成功使用已验收的 SequenceExists 和 UnboundedFromRecurrence 完整学生定理，把“存在序列”与“任意这样的序列给出无界解”连接起来。[完整原题证明及其学生引理](proofs/07-original-target.lean)

课程5的实际学生动作：

```lean
intro s hs B
specialize hs B
suffices s B = s B by rw [this] at hs; aesop
simp only [Prod.mk.injEq, and_self]
```

课程6实际生成`exact fun s ↦ by tauto`；原Target实际生成`tauto`。这不是作者手工补入的动作。加载完整学生定理的两条`have student_fact_… := @…`是明确披露的环境支持，不算学生动作。

## 3. 证明依赖与参数依赖不同

证明依赖为：Recurrence＋IndexedWitness→UnboundedFromRecurrence；SequenceExists＋UnboundedFromRecurrence→Target。历史Recurrence虽然证明成功并补验收，但v23没有执行成功收尾或发布，因此**可复用它的证明，不能把v23当成功参数来源**。

实际参数链还经过先前已成功的坐标增长、显式后继组合和带见证关系课。完整九份release的顺序见[参数索引](weights/index.json)，每份都有create/发布/备份证据。不是把失败参数拼接进成功链。

最后三次新实验分别见[IndexedWitness输入](inputs/indexed-witness/)、[完整桥输入](inputs/unbounded-sequence/)、[原题输入](inputs/original-target/)。这些源文件包含获准的学生证明，不包含教师参考证明。

## 4. 为什么不是无限拆小题

前期显式后继课能解，但撤掉见证后仍选旧(x,y)；固定线性见证的原题尝试也失败。继续练算术并不训练“选择序列索引、使用存在序列”的动作。因此只补一次针对该缺口的IndexedWitness，再立即回完整桥和原Target，没有把原题替换成新微课。

这个过程提供一次实际成功路线，不是随机种子受控的课程消融，不能断言每个拆分或每次参数更新都有独立因果收益。
