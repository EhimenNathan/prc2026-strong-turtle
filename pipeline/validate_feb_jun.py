"""Nested 2025 Feb/Jun holdout for a base + ADS-B surrogate.

Both months are excluded from every base and residual training label. Inner
quarter folds generate residual-training targets. The outer base uses fixed
800 rounds, learning rate .05, 127 leaves; ADS-B uses the production 1600-round
Huber parameters and a fixed unit scale. This is a generalization check, not an
unbiased estimate for every correction or transductive step in the final recipe.
No 2026 labels or leaderboard scores are accessed.
"""
import os, gc, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from config import WORK_DIR
from train_apt import feat_names, load_airport, add_te, TE_KEYS, F29_TE, CATCOLS, EXTRA_CATS
from assembly_checks import require, sha256

WORK = Path(WORK_DIR)
OUT = Path(os.environ.get('PRC_VALIDATION_OUTPUT', 'validation'))
OUT.mkdir(parents=True, exist_ok=True)
APTS = list(pickle.load(open(WORK / 'vocab.pkl', 'rb'))['APT'])
HOLD = [2, 6]
BASE = dict(objective='regression', learning_rate=.05, num_leaves=127,
            min_data_in_leaf=40, feature_fraction=.5, bagging_fraction=.8,
            bagging_freq=1, lambda_l2=15, max_cat_threshold=48, cat_smooth=15,
            max_bin=127, num_threads=4, verbose=-1)

def state_features(ids):
    blocks = []
    names = []
    for filename in ['state_feats_2025.parquet', 'state_feats2_2025.parquet']:
        file = pq.ParquetFile(WORK / filename)
        fid = file.read(columns=['id']).column(0).to_numpy().astype('int64')
        index = pd.Index(fid)
        require(index.is_unique, filename + ': duplicate IDs')
        location = index.get_indexer(ids)
        columns = [name for name in file.schema_arrow.names if name != 'id']
        block = np.full((len(ids), len(columns)), np.nan, np.float32)
        good = location >= 0
        for j, name in enumerate(columns):
            values = file.read(columns=[name]).column(0).to_numpy(zero_copy_only=False)
            block[good, j] = values[location[good]]
        blocks.append(block)
        names.extend(filename.split('.')[0] + '_' + name for name in columns)
    return np.hstack(blocks), names

def build_base():
    for airport in APTS:
        path = OUT / f'base_{airport}.npz'
        if path.exists():
            continue
        start = time.time()
        X, features, meta = load_airport(APTS.index(airport), feat_names(airport))
        labelled = ~meta.is_rank.values
        X, meta = X[labelled], meta[labelled].reset_index(drop=True)
        ids = meta.mvt_id.to_numpy(dtype='int64')
        extra, names = state_features(ids)
        X = np.hstack([X, extra]); features += names
        del extra; gc.collect()
        month = meta.ym.to_numpy() % 100
        held = np.isin(month, HOLD)
        quarter = (month - 1) // 3
        y = meta.y.to_numpy(dtype='float64')
        prediction = np.full(len(y), np.nan)
        keys = TE_KEYS + [key for key in F29_TE if all(c in features for c in key)]
        for inner in sorted(set(quarter[~held])):
            train = ~held & (quarter != inner)
            test = ~held & (quarter == inner)
            require(not np.isin(month[train], HOLD).any(), 'Holdout leakage')
            encoded, encoded_names = add_te(X, features, meta, train, keys)
            categories = [encoded_names.index(c) for c in CATCOLS + EXTRA_CATS if c in encoded_names]
            model = lgb.train(BASE, lgb.Dataset(encoded[train], y[train], categorical_feature=categories), 800)
            prediction[test] = model.predict(encoded[test])
            del model, encoded; gc.collect()
            print(airport, 'inner', inner, 'done', flush=True)
        train = ~held
        encoded, encoded_names = add_te(X, features, meta, train, keys)
        categories = [encoded_names.index(c) for c in CATCOLS + EXTRA_CATS if c in encoded_names]
        model = lgb.train(BASE, lgb.Dataset(encoded[train], y[train], categorical_feature=categories), 800)
        prediction[held] = model.predict(encoded[held])
        require(np.isfinite(prediction).all(), 'Missing nested base predictions')
        np.savez(path, ids=ids, p=prediction, y=y, month=month)
        print(airport, 'outer done', round(time.time()-start), 'seconds', flush=True)
        del X, encoded, model; gc.collect()

def evaluate():
    pieces = []
    for airport in APTS:
        with np.load(OUT / f'base_{airport}.npz') as archive:
            pieces.append(pd.DataFrame({name: archive[name] for name in ['ids','p','y','month']}).assign(apt=airport))
    base = pd.concat(pieces, ignore_index=True).set_index('ids')
    require(base.index.is_unique, 'Duplicate nested IDs')
    matrix_dir = WORK / 'adsblol/mx5'
    tags = ['y2025h1','y2025h2','y2025h1r','y2025h2r']
    metadata = pd.concat([pd.read_parquet(matrix_dir / f'M_{tag}.parquet') for tag in tags], ignore_index=True)
    features = json.loads((matrix_dir / 'FE.json').read_text())
    require(metadata.mvt_id.is_unique, 'Duplicate ADS-B training IDs')
    mapped = base.reindex(metadata.mvt_id)
    require(np.isfinite(mapped.p).all(), 'Missing nested ADS-B base')
    require(np.allclose(mapped.y, metadata.y), 'Label alignment failure')
    held = np.isin(metadata.mon, HOLD)
    prediction = mapped.p.to_numpy()
    target = metadata.y.to_numpy() - prediction
    pcol = features.index('p')
    gapcols = [(features.index('gap_'+name), features.index('x_'+name)) for name in ['first','off','laststill','firstmove','lastblock']]
    scratch = OUT / 'residual_training.npy'
    mapped_matrix = np.lib.format.open_memmap(scratch, mode='w+', dtype=np.float32, shape=(int((~held).sum()),len(features)))
    offset = cursor = 0
    for tag in tags:
        matrix = np.load(matrix_dir / f'X_{tag}.npy', mmap_mode='r')
        for start in range(0, len(matrix), 20000):
            stop = min(start+20000, len(matrix))
            indices = np.arange(offset+start, offset+stop)
            select = ~held[indices]
            block = np.array(matrix[start:stop][select], dtype=np.float32)
            block[:, pcol] = prediction[indices][select]
            for gap, clock in gapcols:
                block[:, gap] = block[:, pcol] - block[:, clock]
            mapped_matrix[cursor:cursor+len(block)] = block
            cursor += len(block)
        offset += len(matrix)
    mapped_matrix.flush()
    ds = lgb.Dataset(mapped_matrix, label=target[~held], categorical_feature=[features.index('aptc')], params=dict(max_bin=255,bin_construct_sample_cnt=50000,verbose=-1))
    ds.construct()
    del mapped_matrix, matrix, block
    gc.collect()
    scratch.unlink()
    residual_params = dict(objective='huber', alpha=1500, learning_rate=.015, num_leaves=63, min_data_in_leaf=50, feature_fraction=.8, bagging_fraction=.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=4, histogram_pool_size=128)
    model = lgb.train(residual_params, ds, 1600)
    test = base[base.month.isin(HOLD)].copy()
    test['adsb'] = 0.0
    test['linked'] = False
    offset = 0
    for tag in tags:
        matrix = np.load(matrix_dir / f'X_{tag}.npy', mmap_mode='r')
        for start in range(0, len(matrix), 20000):
            stop = min(start+20000, len(matrix))
            indices = np.arange(offset+start, offset+stop)
            select = held[indices]
            if not select.any(): continue
            block = np.array(matrix[start:stop][select], dtype=np.float32)
            block[:, pcol] = prediction[indices][select]
            for gap, clock in gapcols: block[:, gap] = block[:, pcol] - block[:, clock]
            ids = metadata.mvt_id.to_numpy()[indices][select]
            test.loc[ids, 'adsb'] = model.predict(block)
            test.loc[ids, 'linked'] = True
        offset += len(matrix)
    test['base_plus_adsb'] = test.p + test.adsb
    for column in ['p','base_plus_adsb']:
        test[column+'_se'] = (test.y-test[column])**2
    groups = test.groupby(['apt','month','linked']).agg(rows=('y','size'),base_sse=('p_se','sum'),adsb_sse=('base_plus_adsb_se','sum'))
    groups['base_rmse'] = np.sqrt(groups.base_sse/groups.rows)
    groups['adsb_rmse'] = np.sqrt(groups.adsb_sse/groups.rows)
    groups.reset_index().to_csv(OUT / 'nested_feb_jun_metrics.csv', index=False)
    test.reset_index().to_parquet(OUT / 'nested_feb_jun_predictions.parquet', index=False)
    overall = test.groupby('month').agg(rows=('y','size'),base_sse=('p_se','sum'),adsb_sse=('base_plus_adsb_se','sum'))
    overall['base_rmse'] = np.sqrt(overall.base_sse/overall.rows)
    overall['adsb_rmse'] = np.sqrt(overall.adsb_sse/overall.rows)
    summary = {'heldout_months_2025':HOLD, 'all_label_training_excludes_holdout':True, 'base_rounds':800,'adsb_rounds':1600,'adsb_scale':1.0,'metrics':overall.reset_index().to_dict('records'),'scope':'Nested base+ADS-B surrogate; excludes final clock, window, tail, seed averaging and transduction. Feature gates and hyperparameters were selected in earlier 2025 work, so this is not untouched-data certification.'}
    (OUT / 'nested_feb_jun_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__':
    build_base()
    evaluate()
