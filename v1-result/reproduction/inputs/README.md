# 原始搜索输入

这里的五个`.lean`文件逐字节取自[五题实验原始归档](../../evidence/multiround/README.md)中的`inputs/`，包含命题、递推条件和搜索配置，不含参考证明。`prepare.py`会先核对归档及成员哈希，再复制这些输入。

- `01-CoupledOddSquare.lean`：奇数数列及其累加，原实验经过3次更新后完成证明。
- `02-AffineAccumulator.lean`：等差数列及其累加；后续跨题继承实验更新1次后证明。
- `03-ScaledTriangular.lean`：带系数的三角数累加，原实验更新1次后证明。
- `04-DifferenceInvariant.lean`：递推差值关系，原实验在32步预算内未完成。
- `05-CubeAccumulator.lean`：立方和累加，原实验更新2次后证明；跨题继承实验更新4次后证明。

题目中的`reapTrainingMCTS`调用Reap搜索。生成的证明另存在每次运行的输出目录，不能将历史证明文件替换为本轮搜索输入。原始结果及对应流程见[实例报告](../../docs/current/04-实际实例与验收结果.md)。

双服务并发教程使用的两个简短命题由既有`portable_lean.py`模板准备，分别是“每个自然数都等于自身”和“所有自然数都等于0”的完整否定；见[并发复现](../03-多题并发.md)。
