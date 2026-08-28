"""Start the existing real-search server and export its actual initialization contract."""
import hashlib
import json
from pathlib import Path
from gpu_runtime import server as core
from gpu_runtime.snapshot_store import _json_bytes


def main():
    args = core.build_parser().parse_args()
    if args.backend != 'real-search' or args.host != '127.0.0.1':
        raise ValueError('Course server requires explicit real-search and loopback binding')
    destination = args.snapshot_root.parent / 'initialization-contract.json'
    if destination.exists():
        raise FileExistsError('Use a fresh run directory; preserve the old contract and state')
    destination.parent.mkdir(parents=True, exist_ok=True)
    backend = core.build_backend(args)
    runtime = core.GpuRuntime(backend=backend, snapshot_root=args.snapshot_root,
        experience_root=args.experience_root, max_resident_sessions=args.max_resident_sessions)
    http = None
    try:
        contract = runtime.actor.submit(backend.experience_contract)
        raw = _json_bytes(contract)
        with destination.open('xb') as stream:
            stream.write(raw)
        core.RuntimeHandler.runtime = runtime
        core.RuntimeHandler.backend_name = args.backend
        http = core.ThreadingHTTPServer((args.host, args.port), core.RuntimeHandler)
        print(json.dumps({'ready': True, 'host': args.host, 'port': args.port,
            'initialization_contract_sha256': hashlib.sha256(raw).hexdigest(),
            'initialization_contract_file': str(destination)}), flush=True)
        http.serve_forever()
    finally:
        if http is not None:
            http.server_close()
        runtime.close()


if __name__ == '__main__':
    main()
