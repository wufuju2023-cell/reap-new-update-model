#!/usr/bin/env python3
"""mix_train.py - versioned 64-bin value-head trainer (resumable, persistent).

Runs under /mnt/workspace/head-runs; every artifact lands for rollback registry.
"""
import argparse, hashlib, json, math, os, random, sys, time, datetime
from pathlib import Path
import torch

WS = Path("/mnt/workspace/head-runs")
A_DIR = Path("/mnt/workspace/new_value_head/features-79efd240")
EXT_MANIFEST = Path("/mnt/workspace/new_value_head/full-extension-features/manifest.json")

def sha256_path(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_v2_dir(directory, manifest_name):
    manifest = json.loads((directory / manifest_name).read_text())
    xs, ys = [], []
    for sh in manifest["shards"]:
        p = directory / sh["path"]
        payload = torch.load(p, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != "new_value_head.feature-shard.v2":
            raise ValueError(f"unexpected {payload.get('schema_version')} in {p}")
        xs.append(payload["features"]); ys.append(payload["labels"])
    return torch.cat(xs, 0), torch.cat(ys, 0)

def load_ext():
    m = json.loads(EXT_MANIFEST.read_text())
    xs, ys = [], []
    for sh in m["shards"]:
        p = Path("/mnt/workspace/new_value_head/full-extension-features") / sh["path"]
        payload = torch.load(p, map_location="cpu", weights_only=True)
        xs.append(payload["features"]); ys.append(payload["labels"])
    return torch.cat(xs, 0), torch.cat(ys, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="mix1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--oversample", type=int, default=8, help="rows with d>=8 repeated xN")
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--max-seconds", type=int, default=200, help="quit after N seconds (resumable)")
    ap.add_argument("--val-size", type=int, default=8000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    run_dir = WS / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress = run_dir / "progress.json"
    cfg = {"run_id": args.run_id, "epochs": args.epochs, "batch_size": args.batch_size,
           "lr": args.lr, "weight_decay": args.weight_decay, "oversample": args.oversample,
           "seed": args.seed, "created": datetime.datetime.utcnow().isoformat() + "Z"}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    prog = json.loads(progress.read_text()) if progress.exists() else {"done_epochs": []}
    done = set(prog.get("done_epochs", []))
    t0 = time.time()

    if "x_train" not in prog:
        torch.manual_seed(args.seed); random.seed(args.seed)
        print("loading A(79e) train...", flush=True)
        x_a, y_a = load_v2_dir(A_DIR / "train", f"boundary-{80000:06d}.json")
        print("loading extension...", flush=True)
        x_b, y_b = load_ext()
        x_train = torch.cat([x_a, x_b], 0)
        y_train = torch.cat([y_a, y_b], 0)
        print("train rows", tuple(x_train.shape), "hist", torch.bincount(y_train.clamp(max=63), minlength=64).tolist()[:12], flush=True)
        if args.smoke:
            x_train = x_train[:2000]; y_train = y_train[:2000]
        # long-tail oversample index
        longtail = torch.where(y_train >= 8)[0]
        idx = torch.arange(x_train.shape[0])
        idx = torch.cat([idx, longtail.repeat(args.oversample)])
        prog["x_train"] = x_train.shape[0]; prog["y_train"] = y_train.shape[0]
        prog["idx_len"] = int(idx.shape[0])
        torch.save(idx, run_dir / "index.pt")
        prog["done_epochs"] = []
        torch.save({"done": done, "k": 0}, run_dir / "train-state.pt")
        json.dump(prog, open(progress, "w"), indent=2)

    x_val, y_val = load_v2_dir(A_DIR / "validation", "manifest.json")
    if args.val_size and args.val_size < x_val.shape[0]:
        x_val = x_val[:args.val_size]; y_val = y_val[:args.val_size]

    idx = torch.load(run_dir / "index.pt")
    torch.manual_seed(args.seed + 1)
    device = torch.device("cuda:0")
    head = torch.nn.Sequential(torch.nn.Linear(3584, 256), torch.nn.SiLU(), torch.nn.Linear(256, 64)).to(device)
    state_path = run_dir / "train-state.pt"
    if state_path.exists():
        st = torch.load(state_path, weights_only=True)
        if "head" in st:
            head.load_state_dict(st["head"])
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if state_path.exists() and "opt" in st:
        opt.load_state_dict(st["opt"])
    x_val = x_val.to(device, dtype=torch.float32)

    for ep in range(args.epochs):
        if ep in done:
            continue
        head.train()
        order = idx[torch.randperm(idx.shape[0])]
        total_loss = 0.0; nsteps = 0
        for s in range(0, order.shape[0], args.batch_size):
            if time.time() - t0 > args.max_seconds:
                print(f"time budget hit: quitting after epoch {ep} partial (resumable)", flush=True)
                torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "done": list(done)}, state_path)
                json.dump(prog, open(progress, "w"), indent=2)
                return
            index = order[s:s+args.batch_size]
            xb = x_train[index].to(device, dtype=torch.float32)
            yb = y_train[index].to(device)
            logits = head(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb.clamp(min=0, max=63))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); nsteps += 1
        print(f"epoch {ep} loss={total_loss/nsteps:.4f}", flush=True)
        head.eval()
        with torch.no_grad():
            val_logits = []
            for vs in range(0, x_val.shape[0], 4096):
                vlog = head(x_val[vs:vs+4096])
                val_logits.append(vlog)
            vl = torch.cat(val_logits, 0)
            yv = y_val[:vl.shape[0]].to(device)
            nll = torch.nn.functional.cross_entropy(vl, yv.clamp(min=0, max=63)).item()
            support = torch.arange(1, 65, device=device, dtype=torch.float32)
            dhat = (vl.softmax(-1) @ support).clamp(1, 64)
        print(f"epoch {ep} val_nll={nll:.4f} val_mean_abs_err={(dhat - (yv + 1).float()).abs().mean().item():.4f}", flush=True)
        done.add(ep)
        prog["done_epochs"] = list(sorted(done))
        prog["last_metrics"] = {"val_nll": nll, "val_mae": float((dhat - (yv + 1).float()).abs().mean().item())}
        json.dump(prog, open(progress, "w"), indent=2)
        torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "done": list(done)}, state_path)
        if ep % 5 == 0 or ep == args.epochs - 1:
            torch.save({"run_id": args.run_id, "artifact_role": "trained", "state_dict": head.state_dict(),
                        "epoch": ep, "sha256_note": ""}, run_dir / f"value-head-ckpt-{ep:02d}.pt")
        if time.time() - t0 > args.max_seconds:
            break

    if len(done) >= args.epochs:
        final = run_dir / "value-head.pt"
        torch.save({"run_id": args.run_id, "artifact_role": "trained", "episodes": args.epochs,
                    "state_dict": head.state_dict()}, final)
        print("FINAL:", final, "sha", sha256_path(final), flush=True)
        report = {"run_id": args.run_id, "config": cfg, "final_sha256": sha256_path(final),
                  "progress": prog}
        (run_dir / "report.json").write_text(json.dumps(report, indent=2))
        (run_dir / "COMPLETE").write_text("ok\n")
        print("COMPLETE", flush=True)

if __name__ == "__main__":
    main()
