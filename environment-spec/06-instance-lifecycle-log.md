# 实例生命周期记录（u-25251-d64e6c11）

## 2026-08-26 会话（rocm-pytorch 镜像，4×W7900 48GB）

### 完成并已验证
1. **bootstrap 控制面**：runner.py（文件队列）+ amdbridge（扩展 0.1.1 + bridge 计划任务 onlogon）
   —— 任意新实例 `bash /workspace/bootstrap.sh` 即恢复控制面（<30s）
2. **policy_server v1（0 长训 + on-demand RTTT）**：
   - REAL-Prover 7B 直载 (`cuda:0`, BF16)，零初始化 LoRA（等价 base）
   - `/v1/chat/completions` `/value` `/ttt_step` `/adapter/snapshot|restore` 已实现并跑通
3. **RTTT demo PASS（实证记录）**：10 events → 5× ttt_step，
   `loss 3.80 → 4.68 → −0.35 → 0.20 → 0.06`，`KL 1.86 → 2.07 → 0.030 → 0.033 → 0.0025`
4. **数据**：REAL-Prover 模型 15GB 已下载到 `/workspace/data/real-prover`（hf-mirror 可行，~8min 可重下）
5. **lean-4.28.0-rc1-linux.tar.zst**（520MB，SHA256 `a3b013a20233f51852ebeb5e54967298a37aff055546170a40f1e79234bb9f4d`）
   —— 本机 F 盘保存完整；实例侧曾传 354/372 块 → **实例已销毁，块丢失；下次重传**

### 销毁时未拉回
- `/workspace/out/rttt_metrics.jsonl`（实例已删；数值已记录于上）
- 实例 dlck 块（无需保留）

## 下次实例（计划）——恢复剧本（v2：镜像分发 / 实例内装包）
```
1. Launch 模板 reap-pytorch（rocm-pytorch 镜像 + SSH:true）→ bootstrap.sh（控制面）
2. git clone https://github.com/wufuju2023-cell/reap-new-update-model → app/
3. 实例内（rocm-pytorch 自带 torch 2.10+HIP，与 base 一致）：
   /opt/venv/bin/pip install peft trl accelerate --index-url 清华      （~2min）
   —— 等效 reap-train 镜像内容（GPU 镜像为可选增强，Dockerfile/CI 保留）
4. 镜像镜像化（按需）：docker pull ghcr.io/.../reap-lean:4.28.0-rc1-reap 跑 CPU MCTS 容器
5. policy_server 起 → rttt_demo → 续 RTTT
```
> GPU 镜像路径：训练环境 = rocm-pytorch base（实例镜像自带）+ 3 个包；容器镜像仅作为可复现制品留档。
