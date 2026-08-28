"""Submit one explicit operator acceptance record; never retry unknown outcomes.

The GPU service checks the binding and stores its evidence digest. The operator
must independently verify Lean proof, source update and wire evidence before
creating this acceptance JSON. This client does not attest to proof correctness.
"""
import argparse
import json
from pathlib import Path

from .http_clients import GpuHttpClient
from .online_ttt import atomic_new_json
from .transport_budget import DEFAULT_BUDGET


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-base-url", required=True)
    p.add_argument("--experience-id", required=True)
    p.add_argument("--acceptance", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--http-timeout-seconds", type=float, default=DEFAULT_BUDGET.client)
    args = p.parse_args()
    acceptance = json.loads(args.acceptance.read_bytes())
    source = acceptance["source"]
    # Durable intent precedes the sole mutation. A missing receipt means inspect
    # this request and its bridge UUID; never blindly rerun into this directory.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_new_json(args.output_dir / "intent.json", {"experience_id": args.experience_id,
                                                    "acceptance": acceptance})
    client = GpuHttpClient(args.gpu_base_url, timeout_seconds=args.http_timeout_seconds)
    receipt = client.publish_experience(source["session_id"], source["snapshot"], args.experience_id, acceptance)
    if receipt.get("experience_id") != args.experience_id or receipt.get("source") != source:
        raise ValueError("publication receipt does not match intent; outcome remains unknown")
    atomic_new_json(args.output_dir / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
