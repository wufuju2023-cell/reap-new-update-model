# 不含模型权重的GPU运行镜像：本地构建与容器检查

GPU运行镜像已在本地WSL/Podman中构建，实际构建进程退出0；构建用时26.680秒，不含基础镜像下载。镜像约29.817GB，**未打包、未上传、未发布；本机没有下载7B权重。** 运行时应挂载已有模型目录。

构建使用固定的ROCm基础镜像和22份哈希固定的Python安装包，断网安装且不自动解析额外依赖，安装前后的Torch身份一致。容器检查也在断网、只读根目录、非特权用户下运行：30份必要源码的哈希、依赖导入、简化模型的私有会话、终态快照复用和释放后补位均通过，Podman与容器退出均为0。这里没有GPU可用，未加载7B或执行AMD计算。

## 证据与使用

镜像标签为`localhost/reap-gpu:runtime-20260828`，ID为`fc5687143dd7fd6ca276e45ed1d420fd3b611fa3299c42245216f94326c2dba3`，digest为`sha256:ed6b3a035fcded94ce2c48653120d04aeb7bb7b57e9b49782ad3edd37943255d`，准确大小29,817,042,215字节。容器用户为UID10001，构建与检查均使用`network=none`；Torch为`2.10.0+rocm7.2.4.git3d3aa833`，HIP为`7.2.53211`，本地`cuda_available=false`。

- [验收总回执](files/acceptance.json)、[实际构建命令与退出](files/evidence/attempt-2/build-result.json)、[镜像inspect](files/evidence/attempt-2/image-inspect.json)。
- [容器命令与退出](files/evidence/attempt-2/smoke-r2/smoke-result.json)、[前后源码pin](files/final-v2/source-after.json)。命令、stdout、stderr、inspect和首次失败保存在[完整小证据](raw.zip)，[成员清单](manifest.json)仅含白名单。
- 配方与挂载说明见[当前Docker使用导航](../../../docker/README.md)；解包[source/current](../../../source/current/README.md)后查看`containers/gpu/README.runtime.md`和`Containerfile.runtime`。模型应通过只读volume挂载已有远端持久目录，输出使用新的持久目录。

本次AMD实例只有Docker客户端，无法连接容器服务，检查的候选套接字不存在，也未安装Podman/Buildah，且缺少所需系统权限；没有绕过平台权限。本地镜像构建与CPU容器机制通过，**AMD实际容器验收未完成**。此前在宿主隔离Python环境中的GPU成功，不能替代容器验收。

保留失败：首次Buildah解析heredoc退出125；首次smoke因测试脚本把snapshot_root传为str退出1，修正为Path后在同一镜像通过。基础镜像pull外壳退出1且真正pull退出码未保存；完整指定digest由inspect确认，不能改称pull exit0。证据不含image/base层、wheelhouse、构建context或模型。
