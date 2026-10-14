"""OOF residuals of a STATE model: quarterly folds, LightGBM (lr 0.05, 127 leaves) on base features + state v1+v2, 9 non-LIRF airports.
Saves oof_state_parts.npz with <apt>_ids / _oof / _y after each airport."""
import os, numpy as np, pandas as pd, pickle, gc, time, warnings, lightgbm as lgb
from train_apt import feat_names, load_airport, add_te, TE_KEYS, F29_TE, CATCOLS, EXTRA_CATS, OUT
from state_blocks import read_state
warnings.filterwarnings("ignore")
apts=list(pickle.load(open(OUT+"vocab.pkl","rb"))["APT"]); RF=OUT+"oof_state_parts.npz"
PRM=dict(objective="regression",learning_rate=0.05,num_leaves=127,min_data_in_leaf=40,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,
         lambda_l2=15,max_cat_threshold=48,cat_smooth=15,max_bin=127,num_threads=7,verbose=-1)
for NM in ["LTFM","EGLL","LFPG","EHAM","EDDF","LEMD","LEBL","EDDM","LSZH","LIRF"]:
    rp=dict(np.load(RF)) if os.path.exists(RF) else {}
    if NM+"_oof" in rp: continue
    t0=time.time(); names=feat_names(NM); X0,feats0,meta=load_airport(apts.index(NM),names)
    ids=meta.mvt_id.values.astype("int64")
    SF1=read_state(ids,["state_feats_2025.parquet"]); SF2=read_state(ids,["state_feats2_2025.parquet"])
    X0=np.hstack([X0,SF1.values.astype(np.float32),SF2.values.astype(np.float32)])
    feats0=feats0+["s1_"+c for c in SF1.columns]+["s2_"+c for c in SF2.columns]
    del SF1,SF2; gc.collect()
    lab=~meta.is_rank.values; y=meta.y.values.astype(np.float64); q=((meta.ym.values%100)-1)//3; oof=np.full(len(meta),np.nan)
    keys=TE_KEYS+[k for k in F29_TE if all(c in feats0 for c in k)]
    for Q in range(4):
        tr=lab&(q!=Q); te=lab&(q==Q)
        X,feats=add_te(X0.copy(),list(feats0),meta,tr,keys); cats=[feats.index(c) for c in CATCOLS+EXTRA_CATS if c in feats]
        m=lgb.train(PRM,lgb.Dataset(X[tr],y[tr],categorical_feature=cats),num_boost_round=800); oof[te]=m.predict(X[te]); del X,m; gc.collect()
    rp[NM+"_ids"]=ids[lab]; rp[NM+"_oof"]=oof[lab]; rp[NM+"_y"]=y[lab]; np.savez(RF,**rp)
    print("%s state-OOF done rmse %.2f (%.0fs)"%(NM,float(np.sqrt(np.mean((y[lab]-oof[lab])**2))),time.time()-t0),flush=True); del X0; gc.collect()
print("OOF STATE DONE",flush=True)
