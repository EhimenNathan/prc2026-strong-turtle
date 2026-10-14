"""Row-level state features for ALL departures (2025 + 2026, all 10 airports), saved by mvt_id for use inside base models.
Uses state_core.py's featurise(): multi-scale airport state (30m/1h/3h/day deviations + lags/leads) and 30-min runway state."""
from config import RANKING, DATA_DIR, WORK_DIR
import numpy as np, pandas as pd, glob, gc, os, pickle, pyarrow.parquet as pq, warnings
warnings.filterwarnings("ignore")
from state_core import rows, featurise, OUT
D=DATA_DIR
for tag,files in [("2025",sorted(glob.glob(os.path.join(D,"training_2025-*.parquet")))),("2026",[os.path.join(D,RANKING)])]:
    Rw=rows(files); X,feats=featurise(Rw,files)
    keep=[f for f in feats if f not in ("hod","dow","dsch","heavy","aptc")]
    X[["id"]+keep].astype({k:"float32" for k in keep}).to_parquet(OUT+"state_feats_%s.parquet"%tag,index=False)
    print(tag,len(X),"rows,",len(keep),"state features",flush=True); del X,Rw; gc.collect()
print("STATEFEATS DONE")
