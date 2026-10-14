"""LOBT-tail correction (non-LIRF). Trees under-predict flights that take off long after their flight-plan off-block time.
D = MVT - LOBT. For D in [1.5 h, 2 h), [2 h, 3 h), [3 h, inf) the correction is the mean residual of the full held-out pipeline
(state-base OOF + per-airport-scaled ADS-B held-out correction) on Jan/Jul 2025. Stability is reported per month (sign must agree).
LIRF is excluded (its long-delay rows are dominated by schedule-echo labels, whose 2026 prevalence cannot be estimated).
Output: work/corr_lobt_tail_final.parquet (mvt_id, c) for the ranking rows."""
import numpy as np, pandas as pd, pyarrow.parquet as pq, pickle
from config import WORK_DIR
OUT = WORK_DIR.rstrip("/\\") + "/"
apts = list(pickle.load(open(OUT + "vocab.pkl", "rb"))["APT"]); E0 = pd.Timestamp("2024-12-01", tz="UTC").value // 10**9
BINS = [(5400, 7200), (7200, 10800), (10800, np.inf)]
d = pq.read_table(OUT + "dep.parquet", columns=["mvt_id", "APT", "t_mvt", "t_lobt", "y", "is_rank"]).to_pandas().drop_duplicates("mvt_id")
d["apt"] = np.array(apts)[d.APT.values]; d["D"] = d.t_mvt - d.t_lobt; d = d[d.apt != "LIRF"]
L = d[~d.is_rank & np.isfinite(d.y)].copy(); L["mon"] = pd.to_datetime(L.t_mvt + E0, unit="s", utc=True).dt.month; L = L[L.mon.isin([1, 7])]
q = np.load(OUT + "oof_state_parts.npz"); O = pd.concat([pd.Series(q[a + "_oof"], index=q[a + "_ids"].astype("int64")) for a in apts if a + "_oof" in q.files])
cv = pd.read_parquet(OUT + "adsblol/mx5/cv_slat_sb.parquet"); cv = cv[np.isfinite(cv.pr)]
K = {a: float((g.pr * (g.y - g.p)).sum() / (g.pr ** 2).sum()) for a, g in cv.groupby("apt")}
c = cv.set_index("mvt_id"); c = c.pr * c.apt.map(K)
L["r"] = L.y - O.reindex(L.mvt_id.values).values - c.reindex(L.mvt_id.values).fillna(0).values; L = L[np.isfinite(L.r)]
# fallback for sparse bins: 12-month base-OOF residual (all 2025 months), required to have the same sign in >= 10 of 12 months
A = d[~d.is_rank & np.isfinite(d.y)].copy(); A["mon"] = pd.to_datetime(A.t_mvt + E0, unit="s", utc=True).dt.month
A["r"] = A.y - O.reindex(A.mvt_id.values).values; A = A[np.isfinite(A.r)]
R = d[d.is_rank].copy(); R["c"] = 0.0
for lo, hi in BINS:
    s = (L.D >= lo) & (L.D < hi); m = [L.r[s & (L.mon == mo)].mean() for mo in (1, 7)]
    val = float(L.r[s].mean()) if s.sum() >= 30 and np.sign(m[0]) == np.sign(m[1]) else 0.0
    if s.sum() < 30:
        sa = (A.D >= lo) & (A.D < hi); bm = A.r[sa].groupby(A.mon[sa]).mean(); agree = int((np.sign(bm) == np.sign(A.r[sa].mean())).sum())
        if sa.sum() >= 30 and agree >= 10: val = float(A.r[sa].mean())
        print("   sparse bin: 12-month base-OOF fallback rows %d, months agreeing %d/12 -> %+.0f s" % (sa.sum(), agree, val))
    R.loc[(R.D >= lo) & (R.D < hi), "c"] = val
    print("D in [%5.0f, %5.0f): held-out rows %4d | mean residual Jan %+6.0f Jul %+6.0f -> correction %+6.0f s | 2026 rows %d"
          % (lo, hi, s.sum(), m[0], m[1], val, ((R.D >= lo) & (R.D < hi)).sum()))
R[["mvt_id", "c"]].to_parquet(OUT + "corr_lobt_tail_final.parquet", index=False); print("saved", (R.c != 0).sum(), "corrected rows")
