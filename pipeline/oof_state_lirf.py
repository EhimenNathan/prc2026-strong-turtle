"""LIRF state-OOF (same model/settings as oof_state.py) with memory-lean state features: only LIRF rows, column by column, float32.
Appends LIRF_ids / LIRF_oof / LIRF_y to work/oof_state_parts.npz."""
import numpy as np, pandas as pd, pickle, gc, time, warnings, lightgbm as lgb, pyarrow.parquet as pq
from train_apt import feat_names, load_airport, add_te, TE_KEYS, F29_TE, CATCOLS, EXTRA_CATS, OUT
warnings.filterwarnings("ignore")
apts=list(pickle.load(open(OUT+"vocab.pkl","rb"))["APT"]); RF=OUT+"oof_state_parts.npz"
def block(fname,ids):
    pf=pq.ParquetFile(OUT+fname); cols=[c for c in pf.schema_arrow.names if c!="id"]
    fid=pf.read(columns=["id"]).column(0).to_numpy().astype("int64"); keep=np.isin(fid,ids); _,first=np.unique(fid,return_index=True)
    keep&=np.isin(np.arange(len(fid)),first); B=np.empty((keep.sum(),len(cols)),np.float32)
    for j,c in enumerate(cols): B[:,j]=pf.read(columns=[c]).column(0).to_numpy(zero_copy_only=False)[keep].astype(np.float32)
    return pd.DataFrame(B,columns=cols,index=fid[keep]).reindex(ids).values.astype(np.float32),cols
PRM=dict(objective="regression",learning_rate=0.05,num_leaves=127,min_data_in_leaf=40,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,
         lambda_l2=15,max_cat_threshold=48,cat_smooth=15,max_bin=127,num_threads=7,verbose=-1)
NM="LIRF"; t0=time.time(); names=feat_names(NM); X0,feats0,meta=load_airport(apts.index(NM),names)
ids=meta.mvt_id.values.astype("int64"); B1,c1=block("state_feats_2025.parquet",ids); B2,c2=block("state_feats2_2025.parquet",ids)
X0=np.hstack([X0,B1,B2]); feats0=feats0+["s1_"+c for c in c1]+["s2_"+c for c in c2]; del B1,B2; gc.collect()
lab=~meta.is_rank.values; y=meta.y.values.astype(np.float64); q=((meta.ym.values%100)-1)//3; oof=np.full(len(meta),np.nan)
keys=TE_KEYS+[k for k in F29_TE if all(c in feats0 for c in k)]
for Q in range(4):
    tr=lab&(q!=Q); te=lab&(q==Q)
    X,feats=add_te(X0.copy(),list(feats0),meta,tr,keys); cats=[feats.index(c) for c in CATCOLS+EXTRA_CATS if c in feats]
    m=lgb.train(PRM,lgb.Dataset(X[tr],y[tr],categorical_feature=cats),num_boost_round=800); oof[te]=m.predict(X[te]); del X,m; gc.collect()
    print("quarter",Q,"done",flush=True)
rp=dict(np.load(RF)); rp[NM+"_ids"]=ids[lab]; rp[NM+"_oof"]=oof[lab]; rp[NM+"_y"]=y[lab]; np.savez(RF,**rp)
print("LIRF state-OOF done rmse %.2f (%.0fs)"%(float(np.sqrt(np.mean((y[lab]-oof[lab])**2))),time.time()-t0),flush=True)
