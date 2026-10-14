"""Movement profile v2 for every already-linked departure (icao + take-off time from f2_<tag>; no relinking).
- 24 x 5-min windows (2 h): time-weighted moving fraction (gs >= 3 kt), mean ground speed, sample count
- 10 x 1-min windows (final 10 min): moving fraction, mean ground speed
- 12 x 5-min windows (last hour): ground-speed standard deviation (stop-and-go)
- 6 x 5-min windows (last 30 min): cumulative absolute track change (deg) while moving (routing / runway turn-on)
- number of stop episodes (>= 20 s below 1 kt) in the last 30 min
Usage: python adsb_profile2.py <tag> <zip> [<zip> ...]  -> work/adsblol/prof2_<tag>.parquet"""
from config import WORK_DIR
import sys, io, zipfile, gc, numpy as np, pandas as pd, pyarrow.parquet as pq
tag = sys.argv[1]; ZS = [zipfile.ZipFile(p) for p in sys.argv[2:]]
IDX = {}; SZ = {}
for z in ZS:
    for i in z.infolist():
        n = i.filename
        if n.startswith("ground_") and i.file_size > SZ.get(n[7:17], -1): IDX[n[7:17]] = z; SZ[n[7:17]] = i.file_size
E0 = pd.Timestamp("2024-12-01", tz="UTC").value // 10**9
F = pd.read_parquet((WORK_DIR.rstrip("/\\") + '/adsblol/f2_%s.parquet') % tag, columns=["mvt_id", "icao", "dt_to"]).drop_duplicates("mvt_id")
d = pq.read_table((WORK_DIR.rstrip("/\\") + '/dep.parquet'), columns=["mvt_id", "t_mvt"]).to_pandas().drop_duplicates("mvt_id").set_index("mvt_id")
F["t_to"] = d.t_mvt.reindex(F.mvt_id.values).values + E0 + F.dt_to.values; del d; gc.collect()
F = F[np.isfinite(F.t_to)]; F["day"] = pd.to_datetime(F.t_to, unit="s", utc=True).dt.strftime("%Y-%m-%d")
COLS = (["qmov_%d" % w for w in range(24)] + ["qgs_%d" % w for w in range(24)] + ["qn_%d" % w for w in range(24)] +
        ["fmov_%d" % w for w in range(10)] + ["fgs_%d" % w for w in range(10)] + ["qsd_%d" % w for w in range(12)] + ["qturn_%d" % w for w in range(6)] + ["nstop30"])
K = len(COLS)
def load(day):
    z = IDX.get(day)
    if z is None: return None
    G = pd.read_parquet(io.BytesIO(z.read("ground_%s.parquet" % day)), columns=["icao", "t", "alt", "gs", "trk"])
    return G[G.alt == -1]
def windows(t, dt, v, mov, t_to, width, n, off=0):
    """time-weighted means per window ending at t_to"""
    res_m = np.full(n, np.nan); res_v = np.full(n, np.nan); res_n = np.zeros(n)
    for w in range(n):
        hi = t_to - off - w * width; lo = hi - width; m = (t >= lo) & (t < hi); c = dt[m].sum(); res_n[w] = m.sum()
        if c > min(30, width / 2): res_m[w] = (mov[m] * dt[m]).sum() / c; res_v[w] = (v[m] * dt[m]).sum() / c
    return res_m, res_v, res_n
F = F.reset_index(drop=True); OUTA = np.full((len(F), K), np.nan, np.float32); OUTID = F.mvt_id.values.astype("int64")
for day, g in F.groupby("day"):
    G = load(day)
    if G is None: continue
    prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"); Gp = load(prev)
    GA = pd.concat([Gp, G]) if Gp is not None else G
    GA = GA[GA.icao.isin(set(g.icao))].sort_values("t")
    gi = {k: (v.t.values, np.nan_to_num(v.gs.values.astype(float), nan=0), v.trk.values.astype(float)) for k, v in GA.groupby("icao")}
    for r in g.itertuples():
        f = OUTA[r.Index]; tv = gi.get(r.icao)
        if tv is not None:
            t, gs, trk = tv; s = (t >= r.t_to - 7800) & (t <= r.t_to); t, gs, trk = t[s], gs[s], trk[s]
            if len(t) >= 2:
                dt = np.minimum(np.diff(t), 120); tm = t[:-1]; v = gs[:-1]; mov = (v >= 3).astype(float)
                m24, g24, n24 = windows(tm, dt, v, mov, r.t_to, 300, 24); f[0:24] = m24; f[24:48] = g24; f[48:72] = n24
                m10, g10, _ = windows(tm, dt, v, mov, r.t_to, 60, 10); f[72:82] = m10; f[82:92] = g10
                for w in range(12):
                    hi = r.t_to - w * 300; m = (tm >= hi - 300) & (tm < hi)
                    if m.sum() >= 3: f[92 + w] = np.std(v[m])
                dtrk = np.abs(((np.diff(trk) + 180) % 360) - 180); movpair = (gs[:-1] >= 3) & (gs[1:] >= 3)
                for w in range(6):
                    hi = r.t_to - w * 300; m = (tm >= hi - 300) & (tm < hi) & movpair & np.isfinite(dtrk)
                    f[104 + w] = dtrk[m].sum() if m.any() else np.nan
                m30 = tm >= r.t_to - 1800; still = (v < 1) & m30
                runs = 0; cur = 0.0
                for i in range(len(still)):
                    if still[i]: cur += dt[i]
                    else:
                        if cur >= 20: runs += 1
                        cur = 0.0
                if cur >= 20: runs += 1
                f[110] = runs
    print(day, len(g), flush=True)
P = pd.DataFrame(OUTA, columns=COLS); P.insert(0, "mvt_id", OUTID); P.to_parquet((WORK_DIR.rstrip("/\\") + '/adsblol/prof2_%s.parquet') % tag, index=False)
print("wrote", len(P), "profiles")
