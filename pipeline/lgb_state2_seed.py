from config import WORK_DIR
import json
"""Seed replica of lgb_state2_full.py (env SEED: LightGBM seed + target-encoding fold seed; output lgbstate2_rank_s<SEED>.npz). Full build: production LightGBM + row-level state features, all 10 airports, trained on all 2025, predicting 2026.
Rounds = production rounds x 1.2 (LTFM: state model best 1726 vs production ~1411). One seed. Saves after each airport."""
import os, re, numpy as np, pandas as pd, lightgbm as lgb, pickle, gc, time, warnings
from pathlib import Path
from checkpoints import fingerprint, read_checkpoint, save_checkpoint
from train_apt import feat_names, load_airport, add_te, TE_KEYS, F29_TE, CATCOLS, EXTRA_CATS, PRM, OUT
warnings.filterwarnings("ignore")
SEED=int(os.environ.get("SEED","1")); apts=list(pickle.load(open(OUT+"vocab.pkl","rb"))["APT"]); RF=OUT+("lgbstate2ws" if os.environ.get("WSFEAT") else ("lgbstate2tn" if os.environ.get("TUNE") else "lgbstate2"))+"_rank_s%d.npz"%SEED
PROD=bool(os.environ.get("PRODSEED"))   # production model: LightGBM default seed, target-encoding fold seed 0
if PROD: RF=OUT+"lgbstate2_rank_parts.npz"
rounds=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_rounds.json")))   # per-airport rounds (train_apt.py validation, early stopping)
import pyarrow.parquet as pq
dependency_paths = sorted({OUT + fn for airport in apts for fn, _ in feat_names(airport)}) + [OUT + f for f in ["vocab.pkl", "state_feats_2025.parquet", "state_feats_2026.parquet", "state_feats2_2025.parquet", "state_feats2_2026.parquet"]] + [__file__, str(Path(__file__).with_name("train_apt.py")), str(Path(__file__).with_name("base_rounds.json"))]
dependencies = {"files": fingerprint(dependency_paths), "seed": SEED, "production": PROD, "tune": os.environ.get("TUNE"), "window_features": os.environ.get("WSFEAT")}
def block(fnames, ids):
    parts = []
    for fn in fnames:
        pf = pq.ParquetFile(OUT + fn); cols = [c for c in pf.schema_arrow.names if c != "id"]
        fid = pf.read(columns=["id"]).column(0).to_numpy().astype("int64"); keep = np.isin(fid, ids); _, first = np.unique(fid, return_index=True)
        keep &= np.isin(np.arange(len(fid)), first); Bk = np.empty((keep.sum(), len(cols)), np.float32)
        for j, c in enumerate(cols): Bk[:, j] = pf.read(columns=[c]).column(0).to_numpy(zero_copy_only=False)[keep].astype(np.float32)
        parts.append(pd.DataFrame(Bk, columns=cols, index=fid[keep]))
    D = pd.concat(parts); return D[~D.index.duplicated()]
def ws_features(NM, ids):
    """leave-one-out window mean of OTHER tracked departures' physical residual (x_off + offset - base prediction), 30/60 min"""
    dep = pq.read_table(OUT + "dep.parquet", columns=["mvt_id", "APT", "t_mvt", "y", "is_rank"]).to_pandas().drop_duplicates("mvt_id")
    dep = dep[np.array(apts)[dep.APT.values] == NM].copy()
    q = np.load(OUT + "oof_state_parts.npz"); zb = np.load(OUT + "lgbstate2_rank_parts.npz")
    O = pd.Series(q[NM + "_oof"], index=q[NM + "_ids"].astype("int64")); Bk = pd.Series(zb[NM + "_rank_p"], index=zb[NM + "_rank_ids"].astype("int64"))
    dep["p"] = np.where(dep.is_rank, Bk.reindex(dep.mvt_id.values).values, O.reindex(dep.mvt_id.values).values)
    F = pd.concat([pd.read_parquet((WORK_DIR.rstrip("/\\") + '/adsblol/f2_%s.parquet') % t, columns=["mvt_id", "x_off"]) for t in ["y2025h1", "y2025h2", "y2025h1r", "y2025h2r", "y2026", "y2026r", "y2026fj", "y2026fjr"]]).drop_duplicates("mvt_id").set_index("mvt_id")
    dep["xo"] = F.x_off.reindex(dep.mvt_id.values).values; dep.loc[~((dep.xo > 60) & (dep.xo < 3600)), "xo"] = np.nan
    lab = ~dep.is_rank & np.isfinite(dep.y); off = np.nanmedian((dep.y - dep.xo)[lab & dep.xo.notna()]) if (lab & dep.xo.notna()).sum() > 50 else 0.0
    dep["z"] = (dep.xo + off - dep.p).clip(-900, 900); dep["has"] = dep.z.notna(); dep["z0"] = dep.z.fillna(0)
    for W in (1800, 3600):
        dep["w"] = (dep.t_mvt // W).astype(int); g = dep.groupby("w"); s_ = g.z0.transform("sum"); n_ = g.has.transform("sum"); m_ = n_ - dep.has.astype(int)
        dep["wsz_%d" % W] = ((s_ - dep.z0) / m_).where(m_ > 0); dep["wsn_%d" % W] = m_
    return dep.set_index("mvt_id")[["wsz_1800", "wsn_1800", "wsz_3600", "wsn_3600"]].reindex(ids).values.astype(np.float32)
for NM in os.environ.get("APTS","LTFM,EGLL,LFPG,EHAM,EDDF,LEMD,LIRF,LEBL,EDDM,LSZH").split(","):
    rp=read_checkpoint(RF, dependencies)
    if NM+"_rank_p" in rp: continue
    t0=time.time(); names=feat_names(NM); X,feats,meta=load_airport(apts.index(NM),names)
    tr=~meta.is_rank.values; rk=meta.is_rank.values; y=meta.y.values.astype(np.float64)
    X,feats=add_te(X,feats,meta,tr,TE_KEYS+[k for k in F29_TE if all(c in feats for c in k)],seed=0 if PROD else SEED)
    ids=meta.mvt_id.values.astype("int64"); sf=block(["state_feats_2025.parquet","state_feats_2026.parquet"],ids).join(block(["state_feats2_2025.parquet","state_feats2_2026.parquet"],ids),how="left").reindex(ids); X=np.hstack([X,sf.values.astype(np.float32)]); feats=feats+["st_"+c for c in sf.columns]; del sf; gc.collect()
    if os.environ.get("WSFEAT"):
        X=np.hstack([X,ws_features(NM,ids)]); feats=feats+["wsz_1800","wsn_1800","wsz_3600","wsn_3600"]
    cats=[feats.index(c) for c in CATCOLS+EXTRA_CATS if c in feats]; nb=int(round(rounds[NM]*1.2))
    TP=dict(cat_smooth=50,max_cat_threshold=16,feature_fraction=0.3) if os.environ.get("TUNE") else {}
    if TP: nb=int(round(nb*1.3))
    m=lgb.train(dict(PRM,num_threads=6,**({} if PROD else {"seed":SEED}),**TP),lgb.Dataset(X[tr],y[tr],categorical_feature=cats),num_boost_round=nb)
    rp[NM+"_rank_ids"]=meta.mvt_id.values[rk].astype("int64"); rp[NM+"_rank_p"]=m.predict(X[rk]); save_checkpoint(RF,rp,dependencies)
    print("%s lgb+state2 full done rounds %d mean %.1f (%.0fs)"%(NM,nb,rp[NM+"_rank_p"].mean(),time.time()-t0),flush=True)
    del X,m; gc.collect()
print("LGBSTATE2 FULL DONE",flush=True)
