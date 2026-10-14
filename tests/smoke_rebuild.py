"""Fresh-directory smoke rebuild on a small raw movement fixture.

This verifies feature entry points and alignment, not full numerical reproduction.
"""
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT
SOURCE = Path(os.environ.get('PRC_DATA', str(ROOT / 'data')))
FIXTURE = Path(os.environ.get('PRC_SMOKE_ROOT', str(ROOT / '.local_checks/smoke_rebuild')))
DATA = FIXTURE / 'data'
WORK = FIXTURE / 'work'
DATA.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)

for path in sorted(SOURCE.glob('*.parquet')):
    if 'submitting' in path.name:
        continue
    destination = DATA / path.name
    if destination.exists():
        continue
    keys = pq.read_table(path, columns=['PHASE_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt']).to_pandas()
    keys['apt'] = np.where(keys.PHASE_mvt.eq('DEP'), keys.ADEP_mvt, keys.ADES_mvt)
    keys['month'] = pd.to_datetime(keys.MVT_TIME_UTC_mvt).dt.month
    selected = np.sort(keys.groupby(['apt','PHASE_mvt','month']).head(40).index.to_numpy())
    del keys
    chunks = []
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=20000):
        local = selected[(selected >= offset) & (selected < offset + len(batch))] - offset
        if len(local):
            chunks.append(pa.Table.from_batches([batch.take(pa.array(local))]))
        offset += len(batch)
    pq.write_table(pa.concat_tables(chunks), destination)
    print('fixture', path.name, len(selected), flush=True)
    del batch, chunks
    gc.collect()

shutil.copytree(Path(os.environ.get('PRC_SOURCE_WORK', str(ROOT / 'work'))) / 'metar', WORK / 'metar', dirs_exist_ok=True)
env = os.environ.copy()
env.update(PRC_DATA=str(DATA), PRC_WORK=str(WORK))
steps = ['build_data','build_arr','f1_deltas','f2_cong','f2b_fix','f3_roll','f4_config','f5_weather',
         'f6_struct','f7_arr','f8_physics','f9_kalman','f10_struct','f11_latent','f12_dynamics',
         'f13_fid','f23c','f29_flight','state_feats','state_feats_v2']
for step in steps:
    print('STEP', step, flush=True)
    subprocess.run([sys.executable, '-u', str(PACKAGE / 'pipeline' / (step+'.py'))],
                   cwd=PACKAGE, env=env, check=True)
dep = pq.read_table(WORK / 'dep.parquet', columns=['mvt_id','is_rank']).to_pandas()
assert dep.mvt_id.is_unique
for name in ['f'+str(i) for i in range(1,14)] + ['f23','f29']:
    assert pq.ParquetFile(WORK / (name+'.parquet')).metadata.num_rows == len(dep), name
for name in ['state_feats','state_feats2']:
    for year, rank in [('2025',False), ('2026',True)]:
        ids = pq.read_table(WORK / f'{name}_{year}.parquet', columns=['id']).column(0).to_numpy()
        assert len(ids) == len(set(ids)) and set(ids) == set(dep.mvt_id[dep.is_rank == rank]), name
report = dict(passed=True, steps=steps, fixture_departures=len(dep),
              ranking_departures=int(dep.is_rank.sum()), unique_ids=True, feature_alignment=True,
              scope='Fresh raw-fixture core/state feature rebuild with cached METAR; excludes full-scale fitting, track extraction and full numerical reproduction.')
(PACKAGE / 'docs' / os.environ.get('PRC_SMOKE_REPORT','feature_smoke_rebuild.json')).write_text(json.dumps(report, indent=2))
print('PASS', json.dumps(report), flush=True)
