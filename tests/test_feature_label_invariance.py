"""Rebuild a small fixture after perturbing only historical departure targets."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / '.local_checks/smoke_rebuild'
PERTURBED = ROOT / '.local_checks/smoke_label_perturbation'
(PERTURBED / 'data').mkdir(parents=True, exist_ok=True)
changed = 0
for path in (ORIGINAL / 'data').glob('*.parquet'):
    table = pd.read_parquet(path)
    if path.name.startswith('training_'):
        departure = table.PHASE_mvt.eq('DEP')
        changed += int(departure.sum())
        table.loc[departure, 'TAXITIME_SEC_mvt'] += 1_000_000
        table.loc[departure, 'BLOCK_TIME_UTC_mvt'] -= pd.Timedelta(days=30)
    table.to_parquet(PERTURBED / 'data' / path.name, index=False)
assert changed > 0
env = os.environ.copy()
env.update(PRC_SMOKE_ROOT=str(PERTURBED), PRC_SMOKE_REPORT='feature_perturbation_rebuild.json')
subprocess.run([sys.executable, '-u', str(ROOT / 'tests/smoke_rebuild.py')], env=env, check=True)
names = ['f'+str(i)+'.parquet' for i in range(1,14)] + ['f23.parquet','f29.parquet','arr.parquet']
names += [f'{name}_{year}.parquet' for name in ['state_feats','state_feats2'] for year in ['2025','2026']]
for name in names:
    before = pd.read_parquet(ORIGINAL / 'work' / name)
    after = pd.read_parquet(PERTURBED / 'work' / name)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
report = dict(passed=True, departure_targets_perturbed=changed,
              perturbations=['TAXITIME_SEC_mvt + 1000000', 'BLOCK_TIME_UTC_mvt - 30 days'],
              unchanged_artifacts=names,
              scope='Exact frame equality on a small raw fixture after changing historical departure targets. Published arrival labels and observed departure clocks remain unchanged; target encodings belong to later model fits.')
(ROOT / 'docs/feature_label_invariance.json').write_text(json.dumps(report, indent=2))
print('PASS: changing', changed, 'historical departure targets left all', len(names), 'derived feature/arrival frames exactly unchanged.', flush=True)
