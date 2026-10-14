"""Per-airport LightGBM models: light on memory, and taxi dynamics are airport-specific."""
from config import DATA_DIR, WORK_DIR, RANKING
import pandas as pd, numpy as np, lightgbm as lgb, gc, sys, os, time, pickle, json
import pyarrow.parquet as pq
OUT=(WORK_DIR + "/")
EPOCH0 = pd.Timestamp("2024-12-01", tz="UTC").value // 10**9
CATCOLS=["RUNWAY_mvt","STAND_mvt","ADES_mvt","AIRCRAFT_TYPE_mvt","AIRCRAFT_TYPE_flt",
         "WK_TBL_CAT_flt","AIRCRAFT_OPERATOR_flt","MARKET_SEGMENT_flt","FLIGHT_TYPE_flt",
         "FLIGHT_RULE_flt","FLIGHT_RULE_mvt","ADES_FILED_flt"]
BASEF=CATCOLS+["adep_match"]
SRC=[("dep.parquet",BASEF),("f1.parquet",None),("f2.parquet",None),("f3.parquet",None)]
for extra in ["f4.parquet","f5.parquet","f6.parquet","f7.parquet","f8.parquet","f9.parquet","f10.parquet","f11.parquet","f12.parquet","f13.parquet"]:
    if os.path.exists(OUT+extra): SRC.append((extra,None))

# f23 (MVT_ID local rank vs time rank) was measured on top of the f13 base and improves
# EGLL -1.07 (-0.96 Jan / -1.17 Jul), EDDF -0.22 and LEBL -0.22 with BOTH months agreeing.
# It hurts LTFM (+0.20) and its LIRF "-1.96" is a month-disagreement artefact (+6.28 Jan
# vs -5.13 Jul), so it is applied per airport rather than globally.
# Rule applied uniformly: deploy where BOTH validation months improve.
#   EGLL -1.07 (-0.96/-1.17)  EDDF -0.22 (-0.43/-0.08)  LEBL -0.22 (-0.71/-0.04)
#   LEMD -0.15 (-0.02/-0.25)  -- in
#   EDDM +0.26 (both worse), LTFM +0.20 (both worse)    -- out
#   LIRF "-1.96" is a month-disagreement artefact (+6.28 Jan / -5.13 Jul) -- out
# Only EGLL clears the measured seed-noise floor (EDDF 1->8 seeds moves RMSE 0.736); the
# other three are sign-consistent but inside it, so they are near-zero-EV inclusions.
F23_APTS={"EGLL","EDDF","LEBL","LEMD"}

# f29 (FLIGHT_mvt: designator, airline prefix, route, frequency + their target encodings).
# FLIGHT_mvt sits on the MOVEMENT record and is never null, but reached no model until now.
# At LIRF it is worth -41.55 raw, -37.98 trimmed to y<=3h, and -33.36 with the identity rule
# applied, with BOTH validation months improving (-2.60 Jan / -47.07 Jul). At EDDF it is
# +0.24, so it is gated to LIRF.
# Mechanism: at EDDF the designator is redundant with operator x stand x runway x type. At
# LIRF the error is a WRITER-identification problem -- which timestamp was written into BLOCK
# -- and the designator identifies the handling agent that filed the record.
F29_APTS={"LIRF"}

def feat_names(apt=None):
    names=[]
    for fn,cols in SRC:
        cs=cols if cols else list(pq.ParquetFile(OUT+fn).schema.names)
        names += [(fn,c) for c in cs]
    if apt is not None and apt in F23_APTS and os.path.exists(OUT+"f23.parquet"):
        names += [("f23.parquet",c) for c in list(pq.ParquetFile(OUT+"f23.parquet").schema.names)]
    if apt is not None and apt in F29_APTS and os.path.exists(OUT+"f29.parquet"):
        names += [("f29.parquet",c) for c in list(pq.ParquetFile(OUT+"f29.parquet").schema.names)]
    return names

def load_airport(a, names):
    meta=pq.read_table(OUT+"dep.parquet",columns=["mvt_id","is_rank","y","t_mvt","APT"]).to_pandas()
    m=(meta.APT.values==a)
    meta=meta[m].reset_index(drop=True)
    ts=pd.to_datetime(meta.t_mvt.values+EPOCH0,unit="s",utc=True)
    meta["ym"]=ts.year.values*100+ts.month.values
    X=np.empty((m.sum(),len(names)),dtype=np.float32)
    for j,(fn,c) in enumerate(names):
        X[:,j]=pq.read_table(OUT+fn,columns=[c]).column(0).to_numpy(zero_copy_only=False)[m].astype(np.float32)
        if j%25==0: gc.collect()
    return X,[c for _,c in names],meta

def add_te(X,feats,meta,fit,keys_list,nfold=5,seed=0,prior_w=20):
    y=meta.y.values.astype(np.float64); gm=np.nanmean(y[fit])
    rng=np.random.default_rng(seed); fold=np.full(len(meta),-1,np.int8)
    fi=np.where(fit)[0]; fold[fi]=rng.integers(0,nfold,len(fi))
    idx={f:j for j,f in enumerate(feats)}; cols=[]; nn=[]
    for keys in keys_list:
        code=np.zeros(len(meta),np.int64)
        for c in keys:
            v=np.nan_to_num(X[:,idx[c]],nan=-1).astype(np.int64)
            code=code*(int(v.max())+2)+(v+1)
        u,inv=np.unique(code,return_inverse=True); K=len(u)
        om=np.full(len(meta),np.nan,np.float32); oc=np.full(len(meta),np.nan,np.float32)
        s=np.bincount(inv[fit],weights=y[fit],minlength=K); c0=np.bincount(inv[fit],minlength=K)
        full=(s+gm*prior_w)/(c0+prior_w); rest=~fit
        om[rest]=full[inv[rest]]; oc[rest]=c0[inv[rest]]
        for f in range(nfold):
            h=fit&(fold==f); u2=fit&(fold!=f)
            s2=np.bincount(inv[u2],weights=y[u2],minlength=K); c2=np.bincount(inv[u2],minlength=K)
            e=(s2+gm*prior_w)/(c2+prior_w)
            om[h]=e[inv[h]]; oc[h]=c2[inv[h]]
        nm="te_"+"_".join(k.replace("_mvt","").replace("_flt","") for k in keys)
        cols+=[om,np.log1p(oc).astype(np.float32)]; nn+=[nm+"_mean",nm+"_cnt"]
    return np.hstack([X,np.column_stack(cols)]).astype(np.float32), feats+nn

EXTRA_CATS=["stand_pier","ades_region","flt_prefix","flt_suffix"]
TE_KEYS=[("STAND_mvt","RUNWAY_mvt"),("STAND_mvt",),("RUNWAY_mvt",),("stand_pier","RUNWAY_mvt"),
         ("AIRCRAFT_OPERATOR_flt",),("AIRCRAFT_TYPE_mvt",),("RUNWAY_mvt","WK_TBL_CAT_flt"),
         ("RUNWAY_mvt","STAND_mvt","WK_TBL_CAT_flt")]
# f29 keys are appended only when the columns are present, i.e. at the gated airports
F29_TE=[("flt_prefix",),("flt_code",),("flt_route",),("flt_prefix","RUNWAY_mvt")]
# best of the manual sweep (EDDF 183.00 -> 180.61): more capacity, lower learning rate
PRM=dict(objective="regression",metric="rmse",learning_rate=0.015,num_leaves=255,
         min_data_in_leaf=30,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,
         lambda_l2=15.0,max_cat_threshold=48,cat_smooth=15,num_threads=6,verbose=-1,max_bin=127)

def run(mode="val"):
    vocab=pickle.load(open(OUT+"vocab.pkl","rb")); apts=list(vocab["APT"])
    preds={}; rows=[]
    for a,nm in enumerate(apts):
        t0=time.time()
        names=feat_names(nm)
        X,feats,meta=load_airport(a,names)
        tr_all=~meta.is_rank.values
        val=tr_all&np.isin(meta.ym.values,[202501,202507])
        tr=tr_all&~val if mode=="val" else tr_all
        fitmask=tr
        keys=TE_KEYS+[k for k in F29_TE if all(c in feats for c in k)]
        X,feats=add_te(X,feats,meta,fitmask,keys)
        cats=[c for c in CATCOLS+EXTRA_CATS if c in feats]
        cati=[feats.index(c) for c in cats]
        y=meta.y.values.astype(np.float64)
        dtr=lgb.Dataset(X[tr],label=y[tr],feature_name=feats,categorical_feature=cati)
        if mode=="val":
            dva=lgb.Dataset(X[val],label=y[val],feature_name=feats,categorical_feature=cati,reference=dtr)
            m=lgb.train(PRM,dtr,3000,valid_sets=[dva],valid_names=["val"],
                        callbacks=[lgb.early_stopping(100,verbose=False)])
            p=m.predict(X[val],num_iteration=m.best_iteration)
            r=float(np.sqrt(np.mean((y[val]-p)**2)))
            rows.append(dict(apt=nm,n_val=int(val.sum()),rmse=r,best=m.best_iteration,
                             base=float(np.sqrt(np.mean((y[val]-y[tr].mean())**2)))))
            preds[nm]=(meta.mvt_id.values[val],p,y[val])
            print(f"{nm}  n={val.sum():6d}  RMSE={r:7.2f}  base={rows[-1]['base']:7.2f}  it={m.best_iteration:4d}  {round(time.time()-t0)}s",flush=True)
            m.save_model(OUT+f"m_val_{nm}.txt")
        else:
            nb=int(json.load(open(OUT+"best_iters.json"))[nm])
            m=lgb.train(PRM,dtr,nb)
            rk=meta.is_rank.values
            p=m.predict(X[rk])
            preds[nm]=(meta.mvt_id.values[rk],p,None)
            m.save_model(OUT+f"m_full_{nm}.txt")
            print(f"{nm} full trained rounds={nb} n_rank={rk.sum()} {round(time.time()-t0)}s",flush=True)
        del X,dtr; gc.collect()
    if mode=="val":
        df=pd.DataFrame(rows); df.to_csv(OUT+"val_by_apt.csv",index=False)
        ids=np.concatenate([v[0] for v in preds.values()])
        pp=np.concatenate([v[1] for v in preds.values()])
        yy=np.concatenate([v[2] for v in preds.values()])
        print("\nOVERALL VAL RMSE:",round(float(np.sqrt(np.mean((yy-pp)**2))),3))
        print("OVERALL BASE    :",round(float(np.sqrt(np.mean((yy-yy.mean())**2))),3))
        np.savez(OUT+"val_preds.npz",ids=ids,p=pp,y=yy)
        json.dump({r["apt"]:int(max(50,r["best"]*1.12)) for r in rows},open(OUT+"best_iters.json","w"))
        print(df.to_string(index=False))
    else:
        ids=np.concatenate([v[0] for v in preds.values()])
        pp=np.concatenate([v[1] for v in preds.values()])
        np.savez(OUT+"rank_preds.npz",ids=ids,p=pp)
        print("saved rank preds",len(ids))

if __name__=="__main__":
    run(sys.argv[1] if len(sys.argv)>1 else "val")

# Features that help the tail/at-risk gate but measurably hurt the bulk regressor
# (f17 anchor spread: LIRF 825.67 -> 889.33). Gate scripts append these explicitly.
GATE_EXTRA=["f17.parquet","f18.parquet","f19.parquet","f20.parquet"]
def gate_feat_names():
    names=feat_names()
    for fn in GATE_EXTRA:
        if os.path.exists(OUT+fn):
            names += [(fn,c) for c in list(pq.ParquetFile(OUT+fn).schema.names)]
    return names
