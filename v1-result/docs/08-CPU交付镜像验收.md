> **历史报告｜2026-08-27版本。** 本文保留当时的结果、流程和未完成项。当前成果请看[当前主报告](current/00-总设计与阅读地图.md)，最新范围见[待办](current/06-待办与验收范围.md)。本文原始字节已保存于[历史归档](history/20260827-documents.tar.gz)。

# CPU 交付镜像验收

2026-08-27。**新 A 镜像的构建产物与本地运行验收通过。** 本次检查不使用 GPU，也没有重复 E/F 的真实训练。原 GPU 实验实际使用的 A 仍是 `df23cd6…`；新镜像有独立身份与证据。

## 镜像与输入身份

```text
tag    localhost/reap-cpu:v1-delivery-20260827
ID     623ae445c3cb8c843afb6c85a1db0dfa25f17e72ae95afaf23a49ef8844f0fce
digest sha256:597edeed42e8c22cc92fc4e46fe3e20df7e0c596227b709e4a19ca735d0fbc29
recipe SHA256
30325287bfbc8b42267d0fce37b7a8d4f1c85be3a7529b2a320d80cffa4414fb
source archive SHA256
69082304c896cd5995619d9c65a8659d452db13f1cfd08f48a00d7822ab141ce
```

构建使用本机缓存、固定预编译 Lean 和 Reap/Mathlib 构建链；镜像 inspect 大小为10,159,784,472字节。没有下载7B权重，没有发布镜像。

## 已实际执行的检查

| 检查 | 结果 |
|---|---|
| 构建产物 | 原日志有COMMIT、Successfully tagged及完整image ID；inspect返回同一ID与digest |
| 运行身份 | 镜像用户`reap`，实际UID10001 |
| CPU代码 | 镜像内13个`cpu_runtime`文件与冻结source manifest逐项匹配 |
| Lean/训练模块 | Lean4.28.0-rc1；已编译`Reap.Training.olean`存在；CPU工具目录未包含`gpu_runtime` |
| 双session mock smoke | 容器`network=none`，两路均solved、returncode0、无timeout；每路15.695181/15.920684秒，整个smoke命令20.669398秒 |
| E原proof重编 | 核对原始证据包及proof文件hash，在新镜像`network=none`重编；exit0，命令10.436555秒，axioms为`propext`与`Quot.sound`，无`sorryAx` |

proof日志只有未使用simp参数的警告。两路smoke使用本地mock服务，这一结果只证明该CPU环境的受测链路；E原proof重编证明新镜像能验证已有证明，未在新镜像重跑GPU搜索/训练。

## 构建退出码的记录限制

父agent的原执行记录显示：构建已完成COMMIT/tag后，外层启动包装脚本因变量转义问题报`exit: : numeric argument required`，外层退出码为1；原Podman退出码未保留，退出码文件为空。

因此摘要保留`outer_wrapper_exit_code=1`、`build_exit_code=null`，**没有补造构建exit0**。镜像产物通过日志/inspect核验，随后所有实际验收命令exit0；结论是“构建产物与运行验收通过”。原日志未修改，没有为了补退出码重新构建。

## 证据与复核

- [验收摘要](../evidence/cpu-delivery/acceptance-summary.json)：身份、检查结果、包装脚本限制与GPU未使用声明。
- [原始证据包](../evidence/cpu-delivery/raw-evidence.tar.gz)：34个小文件，含构建日志、实际命令、inspect、源码检查、mock轨迹、原proof及重编日志、验收脚本；25,431字节。
- [证据manifest](../evidence/cpu-delivery/raw-evidence-manifest.json)：归档与每个成员的大小/hash；所有成员已重新读取逐字节复核。

复核时先校验归档SHA256，再按manifest解包到新目录。复跑smoke可在WSL使用下面的固定ID；容器名和输出卷须全新，已有同名证据时停下检查：

```bash
: "${NEW_CPU_CHECK:?set a new container and volume prefix}"
image=623ae445c3cb8c843afb6c85a1db0dfa25f17e72ae95afaf23a49ef8844f0fce
podman run --name "$NEW_CPU_CHECK" --network none \
  -v "$NEW_CPU_CHECK-out:/workspace/out" --entrypoint reap-cpu-smoke "$image"
```

冻结源码包及旧GPU证据保持原样。源码manifest里的`image_built=false`是制包时状态；本页新增验收并不改写那份历史文件。B镜像构建、合法AMD容器实测与发布仍待完成，详见[06](06-容器未完成原因与交付边界.md)。CPU验收阶段仅在本地整理；本版结果包随后按用户授权上传GitHub，镜像发布仍待完成。
