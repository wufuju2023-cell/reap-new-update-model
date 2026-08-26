#!/usr/bin/env python3
"""V1-1 最小可运行骨架：4 卡 DDP LoRA SFT（从 REAL-Prover 权重续训，不从头）。
用法（实例内）: /opt/venv/bin/torchrun --nproc 4 /workspace/app/train_sft.py \
        --data /workspace/data/pairs/train.jsonl --ckpt-out /workspace/out/ckpt-sft
"""
import argparse, json, os
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt-out", required=True)
    p.add_argument("--model", default="FrenzyMath/REAL-Prover")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--micro-bs", type=int, default=1)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--steps-max", type=int, default=0)  # 0=全量
    return p.parse_args()

def main():
    a = parse_args()
    Path(a.ckpt_out).mkdir(parents=True, exist_ok=True)
    # NOTE(骨架): 完整实现按 v1-spec/01 的显存预算组织：
    #   peft.LoraConfig(r=a.lora_r) → transformers Trainer(accelerate DDP)
    #   数据: jsonl (state_pp, tactic, context) → chat/mask 构造
    #   每 batch 写 out/ckpt-sft/step_<n>.safetensors + 断点 state/ 标记
    print(f"[train_sft] skeleton: {a.model} LoRA r={a.lora_r} -> {a.ckpt_out}")
    print(f"[train_sft] rows={sum(1 for _ in open(a.data)) if Path(a.data).exists() else 'MISSING:{a.data}'} epochs={a.epochs}")

if __name__ == "__main__":
    main()
