"""Build the adsb.lol design matrix ONCE, tag by tag, as float32 .npy files (memory-mapped later -> no commit-charge pressure).
Usage: python adsb_matrix.py <train tags,comma> <test tags,comma>
  train rows: base p = state OOF (10 airports), y known. Test rows (2026): base p = production state model.
Outputs work/adsblol/mx/X_<tag>.npy (float32, rows x FE) + M_<tag>.parquet (mvt_id, y, p, apt, mon, mvt) + FE.json + stands.parquet.
Stand coordinates are learned from TRAINING rows only (median dwell position per airport+stand, >= 5 dwells)."""
from config import WORK_DIR
import os, sys, json, gc, pickle, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from adsb_offblock import hav
tr_tags = sys.argv[1].split(","); te_tags = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else []
OUT = os.environ.get("MXDIR", (WORK_DIR.rstrip("/\\") + '/adsblol/mx/')); os.makedirs(OUT, exist_ok=True)
REC = os.environ.get("REC") == "1"   # add recorded candidate clocks (schedule / planned / estimated / NM actual / ready) and their gaps to ADS-B times
apts = list(pickle.load(open((WORK_DIR.rstrip("/\\") + '/vocab.pkl'), "rb"))["APT"]); E0 = pd.Timestamp("2024-12-01", tz="UTC").value // 10**9
POS = ["first", "off", "laststill", "firstmove", "lastblock"]
XC = ["x_air_prev", "x_first", "x_lastsample", "x_lastblock", "x_laststill", "x_firstmove", "x_off", "x_off_first"]
FE = ["p", "hod", "aptc", "rwy", "dt_to", "by_cs", "roll_to", "n_gs", "gs_first", "n_gaps300", "max_gap", "laststill_dur", "dwell", "n_long_stops",
      "stop_after_off", "path_after_off", "path_m", "move_time", "stop_time_1h", "n_stops"] + ["gap_" + n for n in POS] + ["dst_" + n for n in POS] + XC
RT = ["sch", "lobt", "iobt", "eobt", "aobt", "arvt1", "arvt3"]
if REC:
    FE = FE + ["x_" + t for t in RT] + ["d_%s_%s" % (a, b) for a in ("off", "laststill", "lastblock", "firstmove", "off_first") for b in ("sch", "eobt", "aobt")]
PROF = os.environ.get("PROF") == "1"   # last-hour movement profile (adsb_profile.py)
PCOLS = ["pmov_%d" % w for w in range(12)] + ["pgs_%d" % w for w in range(12)] + ["pn_%d" % w for w in range(12)]
PROF2 = os.environ.get("PROF") == "2"   # profile v2 (adsb_profile2.py) replaces v1
if PROF2:
    PCOLS = (["qmov_%d" % w for w in range(24)] + ["qgs_%d" % w for w in range(24)] + ["qn_%d" % w for w in range(24)] + ["fmov_%d" % w for w in range(10)] +
             ["fgs_%d" % w for w in range(10)] + ["qsd_%d" % w for w in range(12)] + ["qturn_%d" % w for w in range(6)] + ["nstop30"])
    PROF = True
if PROF: FE = FE + PCOLS
SLAT = os.environ.get("SLAT") == "1"   # per-stand arrival recording latency, same year (stand_lat_table.py)
SLCOLS = ["sl_med", "sl_w60", "sl_n", "sl_iqr", "sl_apt_med"]
if SLAT: FE = FE + SLCOLS
json.dump(FE, open(OUT + "FE.json", "w"))

# departure table as plain numpy (id -> row via searchsorted)
d = pq.read_table((WORK_DIR.rstrip("/\\") + '/dep.parquet'), columns=["mvt_id", "APT", "t_mvt", "STAND_mvt", "RUNWAY_mvt"] + ["t_" + t for t in RT]).to_pandas().drop_duplicates("mvt_id").sort_values("mvt_id")
D_REC = {t: (d.t_mvt.values - d["t_" + t].values).astype(np.float32) for t in RT}
D_ID = d.mvt_id.values.astype("int64"); D_APT = d.APT.values.astype(int); D_T = d.t_mvt.values.astype(float) + E0
D_ST = pd.to_numeric(d.STAND_mvt, errors="coerce").values.astype(float); D_RW = pd.to_numeric(d.RUNWAY_mvt, errors="coerce").values.astype(np.float32)
del d; gc.collect()
def drow(ids):
    i = np.searchsorted(D_ID, ids); i = np.clip(i, 0, len(D_ID) - 1); ok = D_ID[i] == ids; return i, ok

def read_cols(tag, cols=None):
    """f2 file -> dict of float32 arrays (+ int64 mvt_id); only the requested columns"""
    pf = pq.ParquetFile((WORK_DIR.rstrip("/\\") + '/adsblol/f2_%s.parquet') % tag); names = pf.schema_arrow.names
    want = [c for c in (cols or names) if c in names and c not in ("icao", "typ")]
    t = pf.read(columns=list(dict.fromkeys(["mvt_id"] + want)))
    out = {}
    for n in t.schema.names:
        a = t.column(n)
        out[n] = a.to_numpy().astype("int64") if n == "mvt_id" else pc.cast(a, pa.float32()).to_numpy(zero_copy_only=False).astype(np.float32)
    del t; gc.collect()
    _, first = np.unique(out["mvt_id"], return_index=True)          # de-duplicate
    return {k: v[first] for k, v in out.items()}

# ---- base predictions: training = state OOF, test = production state model
q = np.load((WORK_DIR.rstrip("/\\") + '/oof_state_parts.npz'))
B_ID = np.concatenate([q[a + "_ids"].astype("int64") for a in apts if a + "_oof" in q.files])
B_P = np.concatenate([q[a + "_oof"] for a in apts if a + "_oof" in q.files]).astype(float)
B_Y = np.concatenate([q[a + "_y"] for a in apts if a + "_oof" in q.files]).astype(float)
o = np.argsort(B_ID); B_ID, B_P, B_Y = B_ID[o], B_P[o], B_Y[o]
if os.environ.get("TEST_BASE", "state") == "state":     # test-time base = the same state model the training base (state OOF) comes from
    zz = np.load((WORK_DIR.rstrip("/\\") + '/lgbstate2_rank_parts.npz'))
    T_ID = np.concatenate([zz[a + "_rank_ids"].astype("int64") for a in apts if a + "_rank_p" in zz.files]); T_P = np.concatenate([zz[a + "_rank_p"] for a in apts if a + "_rank_p" in zz.files]).astype(float)
else:
    raise ValueError("Final pipeline requires TEST_BASE=state")

o = np.argsort(T_ID); T_ID, T_P = T_ID[o], T_P[o]

# ---- stand coordinates from training dwell positions
parts = []
for tag in tr_tags:
    c = read_cols(tag, ["lat_off", "lon_off"]); i, ok = drow(c["mvt_id"]); m = ok & np.isfinite(c["lat_off"])
    parts.append(pd.DataFrame({"apt": D_APT[i[m]], "st": D_ST[i[m]], "la": c["lat_off"][m], "lo": c["lon_off"][m]}))
S = pd.concat(parts).groupby(["apt", "st"]).agg(slat=("la", "median"), slon=("lo", "median"), sn=("la", "size")).reset_index()
S = S[S.sn >= 5]; S.to_parquet(OUT + "stands.parquet", index=False); print("stands with learned coordinates:", len(S), flush=True)
SK = dict(zip(zip(S.apt, S.st), zip(S.slat, S.slon))); del parts; gc.collect()

def build(tag, train):
    c = read_cols(tag); ids = c["mvt_id"]; i, ok = drow(ids)
    if train:
        j = np.searchsorted(B_ID, ids); j = np.clip(j, 0, len(B_ID) - 1); ok &= B_ID[j] == ids; p = B_P[j]; y = B_Y[j]; ok &= np.isfinite(y)
    else:
        j = np.searchsorted(T_ID, ids); j = np.clip(j, 0, len(T_ID) - 1); ok &= T_ID[j] == ids; p = T_P[j]; y = np.full(len(ids), np.nan)
    sel = np.where(ok)[0]; n = len(sel); ii = i[sel]
    X = np.full((n, len(FE)), np.nan, np.float32); col = {f: k for k, f in enumerate(FE)}
    mvt = D_T[ii]; ts = pd.to_datetime(mvt, unit="s", utc=True)
    X[:, col["p"]] = p[sel]; X[:, col["hod"]] = ts.hour; X[:, col["aptc"]] = D_APT[ii]; X[:, col["rwy"]] = D_RW[ii]
    for f in FE:
        if f in c and f not in ("p",): X[:, col[f]] = c[f][sel]
    for f in XC:
        if f in c:
            v = c[f][sel]; X[:, col[f]] = np.where((v >= -600) & (v <= 200000), v, np.nan)
    st = np.array([SK.get((a, s), (np.nan, np.nan)) for a, s in zip(D_APT[ii], D_ST[ii])], dtype=float).reshape(-1, 2)
    for nm in POS:
        if "lat_" + nm in c:
            X[:, col["dst_" + nm]] = hav(st[:, 0], st[:, 1], c["lat_" + nm][sel].astype(float), c["lon_" + nm][sel].astype(float))
        X[:, col["gap_" + nm]] = X[:, col["p"]] - X[:, col["x_" + nm]]
    if PROF:
        pf = pq.ParquetFile(((WORK_DIR.rstrip("/\\") + '/adsblol/prof2_%s.parquet') if PROF2 else (WORK_DIR.rstrip("/\\") + '/adsblol/prof_%s.parquet')) % tag)
        pid = pf.read(columns=["mvt_id"]).column(0).to_numpy().astype("int64"); o = np.argsort(pid); ps = pid[o]
        j = np.searchsorted(ps, ids[sel]); j = np.clip(j, 0, len(ps) - 1); hit = ps[j] == ids[sel]; src = o[j]
        for cname in PCOLS:                       # one column at a time (memory)
            v = pf.read(columns=[cname]).column(0).to_numpy(zero_copy_only=False).astype(np.float32)
            X[:, col[cname]] = np.where(hit, v[src], np.nan); del v
        gc.collect()
    if SLAT:
        pf = pq.ParquetFile((WORK_DIR.rstrip("/\\") + '/stand_lat.parquet'))
        pid = pf.read(columns=["mvt_id"]).column(0).to_numpy().astype("int64"); o = np.argsort(pid); ps = pid[o]
        j = np.searchsorted(ps, ids[sel]); j = np.clip(j, 0, len(ps) - 1); hit = ps[j] == ids[sel]; src = o[j]
        for cname in SLCOLS:
            v = pf.read(columns=[cname]).column(0).to_numpy(zero_copy_only=False).astype(np.float32); X[:, col[cname]] = np.where(hit, v[src], np.nan); del v
        gc.collect()
    nm = tag + ("" if train else os.environ.get("TEST_SUFFIX", ""))
    if REC:
        for t in RT:
            v = D_REC[t][ii]; X[:, col["x_" + t]] = np.where(np.abs(v) < 200000, v, np.nan)
        for a in ("off", "laststill", "lastblock", "firstmove", "off_first"):
            for b in ("sch", "eobt", "aobt"):
                X[:, col["d_%s_%s" % (a, b)]] = X[:, col["x_" + a]] - X[:, col["x_" + b]]
    np.save(OUT + "X_%s.npy" % nm, X)
    pd.DataFrame({"mvt_id": ids[sel], "y": y[sel], "p": p[sel], "apt": np.array(apts)[D_APT[ii]], "mon": ts.month, "mvt": mvt}).to_parquet(OUT + "M_%s.parquet" % nm, index=False)
    print("%s: %d rows x %d features (%.0f MB)" % (tag, n, len(FE), X.nbytes / 1e6), flush=True)
    del X, c; gc.collect()

SUF = os.environ.get("TEST_SUFFIX", "")
if not os.environ.get("TEST_ONLY"):
    for tag in tr_tags: build(tag, True)
for tag in te_tags: build(tag, False)
