"""Verify the portable publication, not the historical remote filesystem."""
import hashlib,json
from pathlib import Path

def verify(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'package-manifest.json').read_bytes())
    for rel,item in manifest['files'].items():
        path=root/rel
        assert path.resolve().is_relative_to(root) and not path.is_symlink(),rel
        raw=path.read_bytes()
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],rel
    print(json.dumps({'files_verified':len(manifest['files']),'bytes':sum(x['bytes'] for x in manifest['files'].values())}))

if __name__=='__main__':
    verify(Path(__file__).resolve().parents[1])
