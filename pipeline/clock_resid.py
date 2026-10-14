"""Clock residual layer for UNLINKED departures (no ADS-B track), non-LIRF. Residual target r = y - state OOF; features: base prediction,
hour, airport, runway, and every recorded clock relative to take-off (SCHED, LOBT, IOBT, EOBT, AOBT, ARVT1, ARVT3) - all known at test time.
Validation: scored-month folds (Jan, Jul each predicted by a model trained on the other 11 months), per-airport strength k, reported on the
UNLINKED rows it will touch. Build: final fit on all 2025 rows; 2026 prediction uses the state-model 2026 base as 'p'; correction = k * pred on
unlinked non-LIRF 2026 rows. Output work/corr_clock_unlinked.parquet."""
from config import WORK_DIR
import gc, pickle, numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
apts = list(pickle.load(open((WORK_DIR.rstrip("/\\") + '/vocab.pkl'), "rb"))["APT"]); E0 = pd.Timestamp("2024-12-01", tz="UTC").value // 10**9
RT = ["sch", "lobt", "iobt", "eobt", "aobt", "arvt1", "arvt3"]
d = pq.read_table((WORK_DIR.rstrip("/\\") + '/dep.parquet'), columns=["mvt_id", "APT", "t_mvt", "RUNWAY_mvt", "y", "is_rank"] + ["t_" + c for c in RT]).to_pandas().drop_duplicates("mvt_id")
d["apt"] = np.array(apts)[d.APT.values]; d = d[d.apt != "LIRF"].copy()
ts = pd.to_datetime(d.t_mvt + E0, unit="s", utc=True); d["mon"] = ts.dt.month.values; d["hod"] = ts.dt.hour.values
for c in RT: d["x_" + c] = (d.t_mvt - d["t_" + c]).astype(np.float32)
d = d.drop(columns=["t_" + c for c in RT] + ["RUNWAY_mvt"] if False else ["t_" + c for c in RT]); gc.collect()
d["aptc"] = d.APT.astype(np.int16); d["rwy"] = pd.to_numeric(d.RUNWAY_mvt, errors="coerce").astype(np.float32); d = d.drop(columns=["RUNWAY_mvt"])
q = np.load((WORK_DIR.rstrip("/\\") + '/oof_state_parts.npz')); O = pd.concat([pd.Series(q[a + "_oof"], index=q[a + "_ids"].astype("int64")) for a in apts if a + "_oof" in q.files])
zb = np.load((WORK_DIR.rstrip("/\\") + '/lgbstate2_rank_parts.npz')); B = pd.concat([pd.Series(zb[a + "_rank_p"], index=zb[a + "_rank_ids"].astype("int64")) for a in apts if a + "_rank_p" in zb.files])
d["p"] = np.where(d.is_rank, B.reindex(d.mvt_id.values).values, O.reindex(d.mvt_id.values).values)
LK = set(pd.concat([pd.read_parquet((WORK_DIR.rstrip("/\\") + '/adsblol/f2_%s.parquet') % t, columns=["mvt_id"]) for t in ["y2025h1", "y2025h2", "y2025h1r", "y2025h2r", "y2026", "y2026r", "y2026fj", "y2026fjr"]]).mvt_id)
d["linked"] = d.mvt_id.isin(LK)
FE = ["p", "hod", "aptc", "rwy"] + ["x_" + c for c in RT]
L = d[~d.is_rank & np.isfinite(d.y) & np.isfinite(d.p)]; r = (L.y - L.p).values.astype(np.float64); mon = L.mon.values; gc.collect()
prm = dict(objective="huber", alpha=1500, learning_rate=0.03, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=6)
X = L[FE].values.astype(np.float32); ai = FE.index("aptc"); pr = np.full(len(L), np.nan)
for fm in (1, 7):
    t = mon == fm; m = lgb.train(prm, lgb.Dataset(X[~t], r[~t], categorical_feature=[ai]), 800); pr[t] = m.predict(X[t]); del m; gc.collect()
ev = np.isfinite(pr); unl = ~L.linked.values; K = {}
def rm(e): return float(np.sqrt(np.mean(e ** 2)))
print("%-5s %8s %8s %8s %18s" % ("apt", "unlinked", "base", "+clock", "Jan / Jul"))
for a in sorted(L.apt.unique()):
    s = ev & (L.apt.values == a) & unl
    if s.sum() < 300: continue
    k = float((pr[s] * r[s]).sum() / (pr[s] ** 2).sum()); K[a] = k
    per = " / ".join("%+.1f%%" % (100 * (rm((r - k * pr)[s & (mon == mo)]) / rm(r[s & (mon == mo)]) - 1)) for mo in (1, 7) if (s & (mon == mo)).sum() > 100)
    print("%-5s %8d %8.1f %8.1f %18s k=%.2f" % (a, s.sum(), rm(r[s]), rm(r[s] - k * pr[s]), per, k))
s = ev & unl; kk = np.array([K.get(a, 0) for a in L.apt.values])
print("unlinked rows total: SSE %.3e -> %.3e (saved %.3e)" % ((r[s] ** 2).sum(), ((r - kk * pr)[s] ** 2).sum(), (r[s] ** 2).sum() - ((r - kk * pr)[s] ** 2).sum()))
GOOD = [a for a, k in K.items() if k > 0.3 and all(rm(r[ev & unl & (L.apt.values == a) & (mon == mo)]) > rm((r - k * pr)[ev & unl & (L.apt.values == a) & (mon == mo)]) for mo in (1, 7))]
print("airports where it helps in BOTH months:", GOOD)
m = lgb.train(prm, lgb.Dataset(X, r, categorical_feature=[ai]), 800)
T = d[d.is_rank & ~d.linked & d.apt.isin(GOOD) & np.isfinite(d.p) & (d.p < 7200)]
c = np.array([K[a] for a in T.apt.values]) * m.predict(T[FE].values.astype(np.float32))
pd.DataFrame({"mvt_id": T.mvt_id.values.astype("int64"), "c": c}).to_parquet((WORK_DIR.rstrip("/\\") + '/corr_clock_unlinked.parquet'), index=False)
print("2026 unlinked rows corrected %d | by airport %s | rms correction %.1f s" % (len(T), T.apt.value_counts().to_dict(), np.sqrt((c ** 2).mean())))
