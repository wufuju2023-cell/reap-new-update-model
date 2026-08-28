# reap-mcts-lean-v2-code-1（V2 首个代码切片）

依据 `explain/reap-mcts-lean-v2/`：元层动作 + Eff 通道 + 塔上升的最小可运行骨架。

## 结构

```
reap-mcts-lean-v2-code-1/
├─ lean-v2/            # Lean 侧（纯 std，可在 reap-lean 容器编译）
│  └─ v2/{Eff,MetaActions,Tower}.lean
├─ v2/                 # Python 侧 harness（无第三方依赖）
│  ├─ eff_registry.py  # Eff 白名单（DeterministicE/ExistentialE 分类 + post 校验）
│  ├─ tower.py         # 塔：gate→register；δ(t,L) 深度；τ_g 塔高；JSON 持久化
│  ├─ runner.py        # min harness：元动作循环（mock policy）→ 样本 sink（v1-compatible 扩展）
│  └─ smoke_v2.py      # 3 用例断言
└─ README.md
```

## 验证

```bash
# Lean 侧（容器编译）：
podman run --rm -v $PWD/lean-v2:/ws/v2proj/src ghcr.io/wufuju2023-cell/reap-lean:4.28.0-rc1 \
  bash -c "… lake build …"          # ✔ Built v2 (3 jobs)

# Python 侧：
python3 -m v2.smoke_v2      # ALL V2 SMOKE PASS（eff 白名单 / 塔 gate / sink 事件）
python3 -m v2.runner --steps 8 --sink /tmp/v2_demo.jsonl   # 8 步元动作循环
```

## 语义要点（spec 对应）

| 代码 | 对应 spec | 关键点 |
|---|---|---|
| `runEffect` / `lookup` | 01 Eff 通道 | ExistentialE 默认拒绝；DeterministicE 白名单 + post 校验 |
| `Tower.register` | 03 塔上升 | gate bool 门控；`δ`/`τ_g` 单调定义 |
| `Action` 枚举 | 02 动作空间 | fillhole/patch/adddecl/effect 序列化；实体（internal）等价 |
| `_sink` kinds | V1 兼容 | `effect/tower/tower_reject` 事件 + node_visited 兼容格式 |

## 待接（下一切片）

- gate 接真实 `checkProof`（Lean 验证器）——当前为回调占位；
- policy 从 mock 换成 `policy_server`（GPU）；
- MCTS 循环接入（复用 V1 `v1_run` 的容器执行模型）。
