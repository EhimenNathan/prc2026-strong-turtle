"""Low-memory version of adsb_fit.py (same model, same validation, same build). The training matrix is never held in RAM:
rows are streamed to CSV in chunks, LightGBM bins it with two_round loading and saves its binary dataset; folds are index subsets of the
binned data; held-out and test predictions read rows from the memory-mapped per-tag matrices.
Usage: python adsb_fit_lowmem.py <train tags,comma> <test tags,comma> [build]    env: MXDIR, OBJ, HALPHA, ROUNDS, LR, TAG"""
from config import WORK_DIR
import os, sys, json, gc, pickle, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
tr = sys.argv[1].split(","); te = sys.argv[2].split(","); BUILD = len(sys.argv) > 3
MX = os.environ.get("MXDIR", (WORK_DIR.rstrip("/\\") + '/adsblol/mx/')); FE = json.load(open(MX + "FE.json")); apts = list(pickle.load(open((WORK_DIR.rstrip("/\\") + '/vocab.pkl'), "rb"))["APT"])
OBJ = os.environ.get("OBJ", "huber"); HA = float(os.environ.get("HALPHA", "1500")); R = int(os.environ.get("ROUNDS", "1600")); LR = float(os.environ.get("LR", "0.015"))
TAG = os.environ.get("TAG", "lowmem")
Xs = [np.load(MX + "X_%s.npy" % t, mmap_mode="r") for t in tr]; M = pd.concat([pd.read_parquet(MX + "M_%s.parquet" % t) for t in tr], ignore_index=True)
assert M.mvt_id.is_unique; OFF = np.cumsum([0] + [len(x) for x in Xs])
r = (M.y - M.p).values.astype(np.float64); mon = M.mon.values; aptv = M.apt.values
def rows(idx):
    out = np.empty((len(idx), len(FE)), np.float32)
    for k, x in enumerate(Xs):
        s = (idx >= OFF[k]) & (idx < OFF[k + 1])
        if s.any(): out[s] = x[idx[s] - OFF[k]]
    return out
BIN = MX + "train_%s.bin" % "_".join(tr)
if not os.path.exists(BIN):
    csv = MX + "train_tmp.csv"
    with open(csv, "w") as f:
        for k, x in enumerate(Xs):
            for a in range(0, len(x), 100000):
                blk = np.asarray(x[a:a + 100000], dtype=np.float32); lab = r[OFF[k] + a: OFF[k] + a + len(blk)]
                np.savetxt(f, np.column_stack([lab, blk]), fmt="%.7g", delimiter=",")
                del blk; gc.collect()
    print("csv written", flush=True)
    ds = lgb.Dataset(csv, params={"two_round": True, "header": False, "label_column": 0, "max_bin": 255, "verbose": -1, "bin_construct_sample_cnt": 50000,
                                  "categorical_feature": str(FE.index("aptc"))})
    ds.construct(); ds.save_binary(BIN); del ds; gc.collect(); os.remove(csv); print("binary dataset saved", flush=True)
FULL = lgb.Dataset(BIN, params={"verbose": -1, "bin_construct_sample_cnt": 50000}).construct()
WFILE = os.environ.get("WFILE")
W = np.load(WFILE).astype(np.float64) if WFILE else np.ones(len(M))
if WFILE: FULL.set_weight(W); print("training with covariate-shift weights", WFILE, flush=True)
prm = dict(objective=OBJ, alpha=HA, learning_rate=LR, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=4, histogram_pool_size=float(os.environ.get("HPOOL", "-1")))
print("train rows %d x %d | %s alpha %g" % (len(M), len(FE), OBJ, HA), flush=True)
pr = np.full(len(M), np.nan)
SKIPCV = os.environ.get("SKIPCV") and os.path.exists(MX + "cv_%s.parquet" % TAG)
if SKIPCV:   # reuse the saved held-out predictions of the identical model (same matrices, params, folds)
    pr = pd.read_parquet(MX + "cv_%s.parquet" % TAG).set_index("mvt_id").pr.reindex(M.mvt_id.values).values; print("held-out predictions reused", flush=True)
for fm in ((1, 7) if not SKIPCV else ()):
    t = mon == fm; m = lgb.train(prm, FULL.subset(np.where(~t)[0]), num_boost_round=R)
    idx = np.where(t)[0]
    for a in range(0, len(idx), 100000): pr[idx[a:a + 100000]] = m.predict(rows(idx[a:a + 100000]))
    del m; gc.collect(); print("fold %d done" % fm, flush=True)
EV = np.isfinite(pr)
def rm(e): return float(np.sqrt(np.mean(e ** 2)))
K = {}
for a in apts:
    s = (aptv == a) & EV
    if s.sum() < 300: continue
    k = float((pr[s] * r[s]).sum() / (pr[s] * pr[s]).sum()); K[a] = k
    per = " ".join("%d:%+.1f%%" % (mo, 100 * (rm((r - k * pr)[s & (mon == mo)]) / rm(r[s & (mon == mo)]) - 1)) for mo in (1, 7) if (s & (mon == mo)).sum() > 100)
    print("%-5s n=%6d  %.1f -> %.1f (%+.1f%%) k=%.2f | %s" % (a, s.sum(), rm(r[s]), rm(r[s] - k * pr[s]), 100 * (rm(r[s] - k * pr[s]) / rm(r[s]) - 1), k, per), flush=True)
kk = np.array([K.get(a, 0.0) for a in aptv]); c = np.where(EV, kk * pr, 0)
print("total SSE saved %.3e" % ((r[EV] ** 2).sum() - ((r - c)[EV] ** 2).sum()), flush=True)
WS = np.load(os.environ.get("WEVAL", MX + "w_y2025h1_y2025h2_y2025h1r_y2025h2r.npy")).astype(np.float64) if os.path.exists(os.environ.get("WEVAL", MX + "w_y2025h1_y2025h2_y2025h1r_y2025h2r.npy")) else np.ones(len(M))
print("2026-weighted held-out SSE saved %.3e (weighted by covariate-shift ratio)" % ((WS[EV] * r[EV] ** 2).sum() - (WS[EV] * (r - c)[EV] ** 2).sum()), flush=True)
M.assign(pr=pr)[["mvt_id", "y", "p", "apt", "mon", "pr"]].to_parquet(MX + "cv_%s.parquet" % TAG, index=False)
if not BUILD: sys.exit()
GOOD = [a for a, k in K.items() if k > 0.3 and all(rm(r[(aptv == a) & (mon == mo) & EV]) > rm((r - k * pr)[(aptv == a) & (mon == mo) & EV]) * 1.01
                                                  for mo in (1, 7) if ((aptv == a) & (mon == mo) & EV).sum() > 300)]
print("airports applied:", GOOD, flush=True)
m = lgb.train(prm, FULL, num_boost_round=R); del FULL; gc.collect()
MT = pd.concat([pd.read_parquet(MX + "M_%s.parquet" % t) for t in te], ignore_index=True)
XT = np.concatenate([np.load(MX + "X_%s.npy" % t, mmap_mode="r") for t in te], axis=0)
MT.assign(raw=m.predict(XT))[["mvt_id", "apt", "p", "raw"]].to_parquet(MX + "test_%s.parquet" % TAG, index=False)
use = MT.apt.isin(GOOD).values & (MT.p.values < 7200)
c26 = np.zeros(len(MT)); c26[use] = np.array([K[a] for a in MT.apt.values[use]]) * m.predict(XT[use]); del XT; gc.collect()
def ofeat(Mx, train):
    O = pd.read_parquet((WORK_DIR.rstrip("/\\") + '/opdi/match_2025full.parquet') if train else (WORK_DIR.rstrip("/\\") + '/opdi/match_rank2026.parquet')).drop_duplicates("mvt_id").set_index("mvt_id").reindex(Mx.mvt_id.values)
    mvt = Mx.mvt.values; Z = np.full((len(Mx), 9), np.nan, np.float32)
    for k2, cn in enumerate(["t_first_taxi", "t_park_exit", "t_rwy", "t_apron_exit"]):
        v = mvt - O[cn].values; Z[:, k2] = np.where((v >= -600) & (v <= 7200), v, np.nan)
    Z[:, 4] = O.n_taxiway.values; Z[:, 5] = np.nan_to_num(O.deice.values.astype(float)); Z[:, 6] = Mx.p.values
    Z[:, 7] = pd.to_datetime(mvt, unit="s", utc=True).hour; Z[:, 8] = pd.Categorical(Mx.apt, categories=apts).codes
    return Z
if os.environ.get("NO_OPDI"):   # OPDI-matched 2026 inputs exist only for Jan/Jul -> skip the OPDI sub-correction for consistency across months
    pd.DataFrame({"mvt_id": MT.mvt_id.values, "c": c26}).to_parquet(MX + "corr_%s.parquet" % TAG, index=False); print("saved correction (no OPDI)", flush=True); sys.exit()
ZB = ofeat(M, True); prB = np.full(len(M), np.nan)
for fm in (1, 7):
    t = mon == fm; mb = lgb.train(prm, lgb.Dataset(ZB[~t], r[~t], categorical_feature=[8]), 800); prB[t] = mb.predict(ZB[t])
kB = float((prB[EV] * r[EV]).sum() / (prB[EV] * prB[EV]).sum()); mB = lgb.train(prm, lgb.Dataset(ZB, r, categorical_feature=[8]), 800); del ZB; gc.collect()
ZT = ofeat(MT, False); hoT = np.isfinite(ZT[:, 0]) | np.isfinite(ZT[:, 1]); c26[use & hoT] -= kB * mB.predict(ZT[use & hoT])
print("2026 corrected rows %d | %s | rms corr %.1f | kB %.2f" % (use.sum(), MT.apt[use].value_counts().to_dict(), np.sqrt((c26[use] ** 2).mean()), kB), flush=True)
pd.DataFrame({"mvt_id": MT.mvt_id.values, "c": c26}).to_parquet(MX + "corr_%s.parquet" % TAG, index=False)
print("saved correction", flush=True)
