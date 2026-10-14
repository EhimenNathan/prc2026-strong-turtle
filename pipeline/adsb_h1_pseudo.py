"""Historical exploratory pseudo-labels: the residual layer fits Jan-Jun 2025 labels (legacy mx6 matrices),
predicting Jul-Dec 2025 linked departures. pseudo = state OOF + k * prediction (k per airport from a Jan-Jun internal holdout: train Jan-Apr,
score May-Jun). Upstream state OOF can include H2 labels; this is not fully nested validation.
This helper is outside run_all.sh. Output work/pseudo_h2.parquet (mvt_id, pseudo)."""
from config import WORK_DIR
import os, json, gc, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
MX = (WORK_DIR.rstrip("/\\") + '/adsblol/mx6/'); FE = json.load(open(MX + "FE.json")); TR = ["y2025h1", "y2025h2", "y2025h1r", "y2025h2r"]
Xs = [np.load(MX + "X_%s.npy" % t, mmap_mode="r") for t in TR]; M = pd.concat([pd.read_parquet(MX + "M_%s.parquet" % t) for t in TR], ignore_index=True)
OFF = np.cumsum([0] + [len(x) for x in Xs]); r = (M.y - M.p).values; mon = M.mon.values
def rows(idx):
    out = np.empty((len(idx), len(FE)), np.float32)
    for k, x in enumerate(Xs):
        s = (idx >= OFF[k]) & (idx < OFF[k + 1])
        if s.any(): out[s] = x[idx[s] - OFF[k]]
    return out
FULL = lgb.Dataset(MX + "train_" + "_".join(TR) + ".bin", params={"verbose": -1, "bin_construct_sample_cnt": 50000}).construct()
prm = dict(objective="huber", alpha=1500, learning_rate=0.015, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=6)
# calibration of k on an internal Jan-Jun holdout
a_tr = np.where(mon <= 4)[0]; a_te = np.where((mon >= 5) & (mon <= 6))[0]
m = lgb.train(prm, FULL.subset(a_tr), 1600); pc = m.predict(rows(a_te)); rr = r[a_te]; ap = M.apt.values[a_te]
K = {a: float((pc[ap == a] * rr[ap == a]).sum() / (pc[ap == a] ** 2).sum()) for a in np.unique(ap)}; print("k:", {a: round(v, 2) for a, v in K.items()}, flush=True)
del m; gc.collect()
h1 = np.where(mon <= 6)[0]; h2 = np.where(mon >= 7)[0]
m = lgb.train(prm, FULL.subset(h1), 1600); pr = np.empty(len(h2))
for b in range(0, len(h2), 100000): pr[b:b + 100000] = m.predict(rows(h2[b:b + 100000]))
k = np.array([K.get(a, 1.0) for a in M.apt.values[h2]]); pseudo = M.p.values[h2] + k * pr
e_b = M.y.values[h2] - M.p.values[h2]; e_p = M.y.values[h2] - pseudo
print("H2 linked rows %d: base rmse %.1f -> pseudo rmse %.1f" % (len(h2), np.sqrt(np.mean(e_b ** 2)), np.sqrt(np.mean(e_p ** 2))))
pd.DataFrame({"mvt_id": M.mvt_id.values[h2], "pseudo": pseudo}).to_parquet((WORK_DIR.rstrip("/\\") + '/pseudo_h2.parquet'), index=False); print("saved")
