# 原题与课程设计

## 1. 完整原题

证明Pell方程有任意大的正整数解：

$$\forall B\in\mathbb N,\ \exists x,y\in\mathbb N,\quad B<x\land B<y\land x^2=2y^2+1.$$

Lean原声明为`CodexMathFive.Pell.Target`，见[原始声明](inputs/original/StudentDeclarations.lean)。这是基于经典Pell方程组装的练习，不声称原创，也不是IMO1988Q6。原题没有被改成有限范围、具体上界或已给定序列的弱命题。

定义变换

$$T(x,y)=(3x+4y,\ 2x+3y),\qquad P_0=(3,2).$$

课程设计让学生先掌握保持方程、增长、构造序列和归纳性质，再学习如何把性质转为存在见证。**下面是实际成功证明的数学依赖，不是预先输入模型的目标证明脚本。**

## 2. 七个完整声明与实际证明

| 课 | 命题 | 学习动作及文件 |
| --- | --- | --- |
| 1 Invariant | 对自然数x,y，若x²=2y²+1，则T(x,y)仍满足方程 | 多项式恒等式；[证明](proofs/01-invariant.lean) |
| 2 PositiveGrowth | 若0<x且0<y，则x<T(x,y)₁且y<T(x,y)₂ | 正性、线性界与合取；[证明](proofs/02-positive-growth.lean) |
| 3 SequenceExists | 存在s:ℕ→ℕ²，s(0)=(3,2)，∀k,s(k+1)=T(s(k)) | 递归构造序列；[证明](proofs/03-sequence-exists.lean) |
| 4 Recurrence | 对任意满足上述初值、递推的s，∀k,k<s(k)₁且k<s(k)₂且s(k)₁²=2s(k)₂²+1 | 自然数归纳及后继收尾；[证明](proofs/04-recurrence.lean) |
| 5 IndexedWitness | 对任意s，若∀k,k<s(k)₁∧k<s(k)₂∧方程成立，则∀B,∃x y,B<x∧B<y∧方程成立 | 自行选择索引并封装存在，不给坐标见证；[证明](proofs/05-indexed-witness.lean) |
| 6 UnboundedFromRecurrence | 对任意满足初值、递推的s，∀B,∃x y,B<x∧B<y∧方程成立 | 把4和5合成完整序列桥；[证明](proofs/06-unbounded-sequence.lean) |
| 7 原Target | ∀B,∃x y,B<x∧B<y∧方程成立，没有给定s | 把3与6组合；[完整证明](proofs/07-original-target.lean) |

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
