from config import WORK_DIR
import json
"""Transductive self-training build: the production state-LightGBM (lgb_state2_full.py: same features, params, rounds) retrained per airport on
2025 labels + 2026 PSEUDO labels for ADS-B-linked departures, then predicting all 2026 rows. Pseudo label = state-model 2026 prediction +
k * raw prediction of the adsb residual layer (mx6 'slat1_sb' model; k per airport from its held-out months).
Target encodings are refitted on the same (label + pseudo) training set. Output work/lgbstate2_selftrain_parts.npz (resumable)."""
import os, re, numpy as np, pandas as pd, lightgbm as lgb, pickle, gc, time, warnings
from pathlib import Path
from checkpoints import fingerprint, read_checkpoint, save_checkpoint
from train_apt import feat_names, load_airport, add_te, TE_KEYS, F29_TE, CATCOLS, EXTRA_CATS, PRM, OUT
warnings.filterwarnings("ignore")
apts = list(pickle.load(open(OUT + "vocab.pkl", "rb"))["APT"]); RF = OUT + os.environ.get("RFNAME", "lgbstate2_selftrain_parts.npz")
rounds = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_rounds.json")))   # per-airport rounds (train_apt.py validation, early stopping)
# pseudo labels for 2026 linked rows
MXS = os.environ.get("PSEUDO_MX", (WORK_DIR.rstrip("/\\") + '/adsblol/mx5/')); PTAG = os.environ.get("PSEUDO_TAG", "slat_sb")
cv = pd.read_parquet(MXS + "cv_%s.parquet" % PTAG); cv = cv[np.isfinite(cv.pr)]
K = {a: float((g.pr * (g.y - g.p)).sum() / (g.pr ** 2).sum()) for a, g in cv.groupby("apt")}
T = pd.read_parquet(MXS + "test_%s.parquet" % PTAG).drop_duplicates("mvt_id")
zb = np.load(OUT + "lgbstate2_rank_parts.npz")
B = pd.concat([pd.Series(zb[a + "_rank_p"], index=zb[a + "_rank_ids"].astype("int64")) for a in apts if a + "_rank_p" in zb.files])
T["base"] = B.reindex(T.mvt_id.values).values; T["k"] = T.apt.map(K).fillna(0).values
T = T[np.isfinite(T.base) & (T.base < 7200)]; PSEUDO = pd.Series(T.base.values + T.k.values * T.raw.values, index=T.mvt_id.values)
BASEONLY = bool(os.environ.get("BASEONLY"))       # self-distillation: pseudo label = the base's own 2026 prediction for EVERY 2026 row (no ADS-B)
if BASEONLY: PSEUDO = B[np.isfinite(B.values)]; K.update({a: 0.0 for a in apts if a not in K})
print("pseudo labels for %d linked 2026 departures" % len(PSEUDO), flush=True)
import pyarrow.parquet as pq
def block(fnames, ids):
    """state features for these ids only (2025 + 2026 files), column by column in float32"""
    parts = []
    for fn in fnames:
        pf = pq.ParquetFile(OUT + fn); cols = [c for c in pf.schema_arrow.names if c != "id"]
        fid = pf.read(columns=["id"]).column(0).to_numpy().astype("int64"); keep = np.isin(fid, ids); _, first = np.unique(fid, return_index=True)
        keep &= np.isin(np.arange(len(fid)), first); Bk = np.empty((keep.sum(), len(cols)), np.float32)
        for j, c in enumerate(cols): Bk[:, j] = pf.read(columns=[c]).column(0).to_numpy(zero_copy_only=False)[keep].astype(np.float32)
        parts.append(pd.DataFrame(Bk, columns=cols, index=fid[keep]))
    D = pd.concat(parts); return D[~D.index.duplicated()]
ORDER = [a for a in os.environ.get("APTS", "EDDF,LFPG,LEMD,EHAM,EGLL,EDDM,LEBL,LSZH,LIRF").split(",") if a in K]
SHARE = float(os.environ.get("PSHARE", "0")); SEED = int(os.environ.get("SEED", "0"))   # target share of total training weight carried by 2026 pseudo rows (0 = unweighted)
dependency_paths = sorted({OUT + fn for airport in apts for fn, _ in feat_names(airport)}) + [OUT + f for f in ["vocab.pkl", "state_feats_2025.parquet", "state_feats_2026.parquet", "state_feats2_2025.parquet", "state_feats2_2026.parquet", "lgbstate2_rank_parts.npz"]] + [MXS + "cv_%s.parquet" % PTAG, MXS + "test_%s.parquet" % PTAG, __file__, str(Path(__file__).with_name("train_apt.py")), str(Path(__file__).with_name("base_rounds.json"))]
dependencies = {"files": fingerprint(dependency_paths), "seed": SEED, "baseonly": BASEONLY, "shares": os.environ.get("PSHARES"), "share": SHARE, "airports": ORDER}
for NM in ORDER:
    rp = read_checkpoint(RF, dependencies)
    if NM + "_rank_p" in rp: continue
    t0 = time.time(); names = feat_names(NM); X, feats, meta = load_airport(apts.index(NM), names)
    ids = meta.mvt_id.values.astype("int64"); rk = meta.is_rank.values; y = meta.y.values.astype(np.float64).copy()
    ps = PSEUDO.reindex(ids).values; usep = rk & np.isfinite(ps); y[usep] = ps[usep]
    tr = (~rk) | usep
    X, feats = add_te(X, feats, meta.assign(y=y), tr, TE_KEYS + [k for k in F29_TE if all(c in feats for c in k)], seed=SEED)
    sf = block(["state_feats_2025.parquet", "state_feats_2026.parquet"], ids).join(block(["state_feats2_2025.parquet", "state_feats2_2026.parquet"], ids), how="left").reindex(ids); X = np.hstack([X, sf.values.astype(np.float32)]); feats = feats + ["st_" + c for c in sf.columns]; del sf; gc.collect()
    cats = [feats.index(c) for c in CATCOLS + EXTRA_CATS if c in feats]; nb = int(round(rounds[NM] * 1.2))
    w = np.ones(len(y))
    PSH = dict(kv.split(":") for kv in os.environ.get("PSHARES", "").split(",") if kv); SHARE = float(PSH.get(NM, SHARE)) if PSH else SHARE
    if SHARE > 0 and usep.sum() > 0:           # weight pseudo rows so they carry SHARE of the training weight (recency, as in the validated simulation)
        n_lab = ((~rk) & tr).sum(); w[usep] = SHARE / (1 - SHARE) * n_lab / usep.sum()
    m = lgb.train(dict(PRM, num_threads=6, **({'seed': SEED} if SEED else {})), lgb.Dataset(X[tr], y[tr], weight=w[tr], categorical_feature=cats), num_boost_round=nb)
    rp[NM + "_rank_ids"] = ids[rk]; rp[NM + "_rank_p"] = m.predict(X[rk]); rp[NM + "_rank_pseudo"] = usep[rk]; save_checkpoint(RF, rp, dependencies)
    print("%s self-train done | pseudo rows %d | rounds %d | mean %.1f (%.0fs)" % (NM, usep.sum(), nb, rp[NM + "_rank_p"].mean(), time.time() - t0), flush=True)
    del X, m; gc.collect()
print("SELFTRAIN DONE", flush=True)
