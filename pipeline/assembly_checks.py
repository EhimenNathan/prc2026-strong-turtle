"""Strict prediction alignment and completeness checks for the final recipe."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_parts(path, airports, reference):
    path = Path(path)
    require(path.is_file(), f'Missing model output: {path}')
    outputs = []
    with np.load(path, allow_pickle=False) as archive:
        present = {key[:-7] for key in archive.files if key.endswith('_rank_p')}
        require(present == set(airports), f'{path.name}: expected airports {airports}, found {sorted(present)}')
        for airport in airports:
            ids = archive[airport + '_rank_ids']
            values = archive[airport + '_rank_p']
            expected = reference.index[reference.apt == airport]
            require(ids.dtype.kind in 'iu', f'{path.name}: noninteger IDs')
            require(len(ids) == len(values) == len(expected), f'{path.name}: incomplete {airport}')
            require(pd.Index(ids).is_unique and set(ids) == set(expected), f'{path.name}: ID mismatch {airport}')
            require(np.isfinite(values).all(), f'{path.name}: nonfinite {airport}')
            outputs.append(pd.Series(values, index=ids))
    return pd.concat(outputs)


def read_correction(path, reference):
    path = Path(path)
    require(path.is_file(), f'Missing correction: {path}')
    table = pd.read_parquet(path, columns=['mvt_id', 'c'])
    require(table.mvt_id.is_unique, f'{path.name}: duplicate IDs')
    if table.mvt_id.dtype.kind == 'f':
        ids = table.mvt_id.to_numpy()
        require(np.isfinite(ids).all() and (ids == np.floor(ids)).all() and (np.abs(ids) < 2**53).all(), f'{path.name}: inexact IDs')
        table['mvt_id'] = table.mvt_id.astype('int64')
    require(table.mvt_id.dtype.kind in 'iu', f'{path.name}: noninteger IDs')
    require(table.mvt_id.isin(reference.index).all(), f'{path.name}: unexpected IDs')
    require(np.isfinite(table.c).all(), f'{path.name}: nonfinite correction')
    return table.set_index('mvt_id').c.reindex(reference.index, fill_value=0.0)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
