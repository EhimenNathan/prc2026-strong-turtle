"""Atomic checkpoints and exact dependency fingerprints for production runs."""
import hashlib, json, os
from pathlib import Path
import numpy as np

def fingerprint(paths):
    result = {}
    for path in paths:
        path = Path(path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                digest.update(block)
        result[path.name] = digest.hexdigest()
    return result

def read_checkpoint(path, dependencies):
    path = Path(path)
    if not path.exists(): return {}
    receipt = path.with_suffix('.dependencies.json')
    if not receipt.exists() or json.loads(receipt.read_text()) != dependencies:
        raise ValueError(f'{path.name}: unverified or stale checkpoint; use a fresh PRC_WORK directory or restore the matching dependencies')
    with np.load(path,allow_pickle=False) as archive:
        return dict(archive)

def save_checkpoint(path, values, dependencies):
    path = Path(path)
    temporary = path.with_suffix('.tmp.npz')
    np.savez(temporary,**values)
    os.replace(temporary,path)
    receipt = path.with_suffix('.dependencies.json')
    receipt.write_text(json.dumps(dependencies,sort_keys=True,indent=2))
