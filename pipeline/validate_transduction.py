"""Nested February/June surrogate test of fixed transductive steps.

Requires validate_feb_jun.py outputs. Holdout truth is used only for scoring;
pseudo-labels come from its base and ADS-B predictions. This checks the fixed
half-weight treatment, not the entire production recipe or its seed shrink.
"""
import argparse
import gc
import json
import os
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import WORK_DIR
from assembly_checks import require
from train_apt import load_airport, feat_names, add_te, TE_KEYS, F29_TE, CATCOLS, EXTRA_CATS
from state_blocks import read_state

OUT = Path(os.environ.get('PRC_VALIDATION_OUTPUT', 'validation'))
HOLD = [2, 6]
PARAMS = dict(objective='regression', learning_rate=.05, num_leaves=127,
              min_data_in_leaf=40, feature_fraction=.5, bagging_fraction=.8,
              bagging_freq=1, lambda_l2=15, max_cat_threshold=48, cat_smooth=15,
              max_bin=127, num_threads=4, verbose=-1)
SHARES = {'LFPG': .3, 'EDDF': .3, 'EGLL': .3, 'LEMD': .3, 'EHAM': 0}


def evaluate():
    summary = json.loads((OUT / 'nested_feb_jun_summary.json').read_text())
    require(summary['all_label_training_excludes_holdout'], 'Base validation is not nested')
    held_predictions = pd.read_parquet(OUT / 'nested_feb_jun_predictions.parquet').set_index('ids')
    require(held_predictions.index.is_unique, 'Duplicate validation IDs')
    airports = list(pickle.load(open(Path(WORK_DIR) / 'vocab.pkl', 'rb'))['APT'])
    metrics = []
    for airport in list(SHARES) + ['LIRF', 'LTFM']:
        checkpoint = OUT / f'transduction_{airport}.parquet'
        if checkpoint.exists():
            continue
        X, features, meta = load_airport(airports.index(airport), feat_names(airport))
        labelled = ~meta.is_rank.values
        X, meta = X[labelled], meta[labelled].reset_index(drop=True)
        ids = meta.mvt_id.to_numpy(dtype='int64')
        for filename, prefix in [('state_feats_2025.parquet', 's1_'), ('state_feats2_2025.parquet', 's2_')]:
            extra = read_state(ids, [filename])
            X = np.hstack([X, extra.to_numpy(dtype=np.float32)])
            features += [prefix + name for name in extra.columns]
            del extra
        month = meta.ym.to_numpy() % 100
        held = np.isin(month, HOLD)
        test = held_predictions.reindex(ids[held])
        require(test.p.notna().all(), airport + ': missing nested predictions')
        require(np.allclose(test.y, meta.y.to_numpy()[held]), 'Validation truth misalignment')
        linked = np.zeros(len(ids), dtype=bool)
        linked[held] = test.linked.to_numpy()
        pseudo = np.full(len(ids), np.nan)
        pseudo[held] = test.base_plus_adsb.to_numpy() if airport in SHARES else test.p.to_numpy()
        treatment_rows = held & linked if airport in SHARES else held
        require(treatment_rows.any(), airport + ': no pseudo-label rows')
        train = ~held | treatment_rows
        target = meta.y.to_numpy(dtype=float).copy()
        target[held] = pseudo[held]
        require(np.isfinite(target[train]).all(), 'Invalid training targets')
        weight = np.ones(len(ids))
        share = SHARES.get(airport, 0)
        if share:
            weight[treatment_rows] = share / (1-share) * (~held).sum() / treatment_rows.sum()
        keys = TE_KEYS + [key for key in F29_TE if all(c in features for c in key)]
        encoded, names = add_te(X, features, meta.assign(y=target), train, keys)
        categories = [names.index(c) for c in CATCOLS + EXTRA_CATS if c in names]
        model = lgb.train(PARAMS, lgb.Dataset(encoded[train], target[train],
                          weight=weight[train], categorical_feature=categories), 800)
        treatment = model.predict(encoded[held])
        test = test.copy()
        test['treatment'] = treatment
        # Production transductive changes are applied only to unlinked rows.
        test['fixed_half_blend'] = np.where(test.linked, test.p, (test.p + treatment)/2)
        test['treatment_kind'] = 'self_training' if airport in SHARES else 'base_distillation'
        require(np.isfinite(test.fixed_half_blend).all(), 'Invalid treatment predictions')
        test.reset_index().to_parquet(checkpoint, index=False)
        print(airport, 'transduction surrogate done', flush=True)
        del X, encoded, model, meta, test
        gc.collect()
    frames = [pd.read_parquet(OUT / f'transduction_{airport}.parquet') for airport in list(SHARES) + ['LIRF', 'LTFM']]
    predictions = pd.concat(frames, ignore_index=True)
    for (airport, month), group in predictions[~predictions.linked].groupby(['apt', 'month']):
        metrics.append(dict(airport=airport, month=int(month), rows=len(group),
                            treatment_kind=group.treatment_kind.iloc[0],
                            base_rmse=float(np.sqrt(np.mean((group.y-group.p)**2))),
                            half_blend_rmse=float(np.sqrt(np.mean((group.y-group.fixed_half_blend)**2)))))
    pd.DataFrame(metrics).to_csv(OUT / 'nested_transduction_metrics.csv', index=False)
    result = dict(heldout_months_2025=HOLD, holdout_truth_used_only_for_scoring=True,
                  rounds=800, blend_weight=.5, metrics=metrics,
                  scope='Nested base/ADS-B pseudo-label surrogate. No production seed-noise shrink, clock, window or tail corrections. Earlier feature and parameter choices used 2025 development evidence.')
    (OUT / 'nested_transduction_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true', help='Wait for the base/ADS-B validation summary')
    args = parser.parse_args()
    if args.wait:
        while not (OUT / 'nested_feb_jun_summary.json').exists():
            time.sleep(30)
    evaluate()
