# 当前版本源码与复现入口

这里提供当前TTT框架的独立源码包，以及无需模型权重的证据检查脚本。Reap作为Lean搜索组件随相关补丁和构建配方交付。历史V1的 `source-snapshot.tar.gz`、`source-manifest.json`、`verify_source.py` 保持原样；旧包对应 2026-08-27 五题实验。

## 复现顺序

首次复现先看[单题、跨题与并发操作指南](../reproduction/README.md)，代码可在[可浏览代码目录](../reproduction/code/README.md)直接阅读。其他电脑上的新Agent可使用[交接prompt与教学](../prompt_for_agent/README.md)。

1. 阅读[扩展训练与恢复指南](current/README.md)。
2. 用 `current/verify.py` 校验并解压当前源码到新目录。
3. 用 `current/reproduce.py prepare` 从本包证据提取完整证明数据，再做本地检查。
4. 需要重做训练时，再按指南准备AMD环境并选择具体GPU阶段；模型目录从外部挂载，包内不含权重。

当前源码是报告所讲框架的工作版本；每项历史实测还保留自己的输入和代码哈希。新增本地机制与真实 GPU 验收范围分别见[成果实例](../docs/current/04-实际实例与验收结果.md)。

## 历史包仍可校验

```bash
python3 source/verify_source.py
```

在 `v1-result` 根运行。旧包 77 个成员，SHA256 `69082304c896cd5995619d9c65a8659d452db13f1cfd08f48a00d7822ab141ce`。历史说明已经收入[文档归档](../docs/history/archive-manifest.json)。
