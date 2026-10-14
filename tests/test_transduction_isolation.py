"""Verify holdout truth cannot enter transductive model targets or encoding."""
import importlib.util
import json
import os
import pickle
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APTS = ['LFPG', 'EDDF', 'EGLL', 'LEMD', 'EHAM', 'LIRF', 'LTFM']
calls = []
datasets = []
truth = {}

def load_airport(index, names):
    ids = np.arange(index*12+1, index*12+13)
    month = np.array([1]*4+[2]*4+[6]*4)
    y = 10000.0 + ids
    truth.update(zip(ids, y))
    return ids[:, None].astype(float), ['row_id'], pd.DataFrame(dict(mvt_id=ids, is_rank=False, ym=202500+month, y=y))

def encode(X, names, meta, train, keys):
    held = meta.ym.to_numpy() % 100 != 1
    assert (meta.y.to_numpy()[held] != np.array([truth[i] for i in meta.mvt_id[held]])).all()
    calls.append(int((held & train).sum()))
    return X, names

class Dataset:
    def __init__(self, X, label, **kwargs):
        for row, value in zip(X, label):
            row_id = int(row[0])
            if (row_id-1) % 12 >= 4:
                assert value != truth[row_id], 'True holdout label entered model fit'
        datasets.append(len(label))

class Model:
    def predict(self, X):
        return X[:, 0] + 1

with tempfile.TemporaryDirectory(dir=ROOT) as folder:
    folder = Path(folder)
    with (folder / 'vocab.pkl').open('wb') as stream:
        pickle.dump({'APT': APTS}, stream)
    rows = []
    for i, airport in enumerate(APTS):
        for k in range(4, 12):
            row_id = i*12+k+1
            rows.append(dict(ids=row_id, p=float(row_id), y=10000.0+row_id,
                             month=2 if k < 8 else 6, apt=airport,
                             base_plus_adsb=float(row_id)+5, linked=k % 2 == 0))
    pd.DataFrame(rows).to_parquet(folder / 'nested_feb_jun_predictions.parquet')
    (folder / 'nested_feb_jun_summary.json').write_text(json.dumps({'all_label_training_excludes_holdout': True}))
    sys.modules['config'] = types.SimpleNamespace(WORK_DIR=str(folder))
    sys.modules['train_apt'] = types.SimpleNamespace(load_airport=load_airport,
        feat_names=lambda airport: [], add_te=encode, TE_KEYS=[], F29_TE=[], CATCOLS=[], EXTRA_CATS=[])
    sys.modules['state_blocks'] = types.SimpleNamespace(read_state=lambda ids, files: pd.DataFrame({'state': np.zeros(len(ids))}, index=ids))
    sys.modules['lightgbm'] = types.SimpleNamespace(Dataset=Dataset, train=lambda *args: Model())
    os.environ['PRC_VALIDATION_OUTPUT'] = str(folder)
    sys.path.insert(0, str(ROOT / 'pipeline'))
    spec = importlib.util.spec_from_file_location('validation_test', ROOT / 'pipeline/validate_transduction.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.evaluate()
    assert calls == [4]*5 + [8]*2
    assert len(datasets) == 7
    results = pd.read_csv(folder / 'nested_transduction_metrics.csv')
    assert len(results) == 14 and (results.rows == 2).all()
print('PASS: holdout truth excluded from target encoding and all seven treatment fits; linked-row scope checked.')
