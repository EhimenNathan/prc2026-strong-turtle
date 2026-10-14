"""Build the window-state correction for 2026 UNTRACKED departures: k_a * mean ADS-B correction of OTHER tracked departures in the same
airport x 30-min window. k_a fitted on 2025 held-out (Jan+Jul pooled) using held-out corrections; airport applied only if BOTH months improve
with the other month's k. Output work/corr_window_state.parquet."""
from config import WORK_DIR
import numpy as np, pandas as pd, pyarrow.parquet as pq, pickle
apts = list(pickle.load(open((WORK_DIR.rstrip("/\\") + '/vocab.pkl'), "rb"))["APT"]); W = 1800; E0 = pd.Timestamp("2024-12-01", tz="UTC").value // 10**9
dep = pq.read_table((WORK_DIR.rstrip("/\\") + '/dep.parquet'), columns=["mvt_id", "APT", "t_mvt", "y", "is_rank"]).to_pandas().drop_duplicates("mvt_id"); dep["apt"] = np.array(apts)[dep.APT.values]
def loo(d, col, lk):
    d = d.assign(w=(d.t_mvt // W).astype(int)); g = d.groupby(["apt", "w"]); s = g[col].transform("sum"); n = g[lk].transform("sum")
    m = n - d[lk].astype(int); return ((s - d[col]) / m).where(m > 0)
# 2025 held-out
L = dep[~dep.is_rank & np.isfinite(dep.y)].copy(); L["mon"] = pd.to_datetime(L.t_mvt + E0, unit="s", utc=True).dt.month; L = L[L.mon.isin([1, 7])]
q = np.load((WORK_DIR.rstrip("/\\") + '/oof_state_parts.npz')); O = pd.concat([pd.Series(q[a + "_oof"], index=q[a + "_ids"].astype("int64")) for a in apts if a + "_oof" in q.files])
cv = pd.read_parquet((WORK_DIR.rstrip("/\\") + '/adsblol/mx5/cv_slat_sb.parquet')); cv = cv[np.isfinite(cv.pr)]; K = {a: float((g.pr * (g.y - g.p)).sum() / (g.pr ** 2).sum()) for a, g in cv.groupby("apt")}
cc = cv.set_index("mvt_id"); cc = cc.pr * cc.apt.map(K)
L["cc"] = cc.reindex(L.mvt_id.values).values; L["lk"] = L.cc.notna(); L["cc"] = L.cc.fillna(0.0); L["e"] = L.y - O.reindex(L.mvt_id.values).values - L.cc
L["f"] = loo(L, "cc", "lk"); L = L[np.isfinite(L.e)]; B = (L.y <= 3600) & ~L.lk & L.f.notna()
KA = {}
for a in sorted(L.apt.unique()):
    ok = True; gains = []
    for m in (1, 7):
        fit = L[B & (L.apt == a) & (L.mon == 8 - m)]; ev = L[B & (L.apt == a) & (L.mon == m)]
        if len(fit) < 200 or len(ev) < 200: ok = False; break
        k = (fit.f * fit.e).sum() / (fit.f ** 2).sum(); g = 1 - ((ev.e - k * ev.f) ** 2).sum() / (ev.e ** 2).sum(); gains.append(g); ok &= g > 0
    if ok:
        s = L[B & (L.apt == a)]; KA[a] = float((s.f * s.e).sum() / (s.f ** 2).sum())
    print("%-5s %s" % (a, ("k=%.2f gains Jan %.2f%% Jul %.2f%%" % (KA[a], 100 * gains[0], 100 * gains[1])) if a in KA else "not applied %s" % ["%.2f%%" % (100 * x) for x in gains]))
# 2026
R = dep[dep.is_rank].copy(); c26 = pd.read_parquet((WORK_DIR.rstrip("/\\") + '/adsblol/mx5/corr_slat_sb.parquet')).set_index("mvt_id").c
R["cc"] = c26.reindex(R.mvt_id.values).fillna(0).values; R["lk"] = R.cc.abs() > 1e-9; R["f"] = loo(R, "cc", "lk")
R["c"] = np.where(~R.lk & R.f.notna() & R.apt.isin(list(KA)), R.apt.map(KA).fillna(0) * R.f.fillna(0), 0.0)
R[["mvt_id", "c"]].to_parquet((WORK_DIR.rstrip("/\\") + '/corr_window_state.parquet'), index=False)
print("2026 corrected rows %d | rms %.1f s | by airport %s" % ((R.c != 0).sum(), np.sqrt((R.c ** 2).mean()), R[R.c != 0].apt.value_counts().to_dict()))
