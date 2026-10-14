"""MVT_ID rank vs MVT_TIME rank, WITHIN an airport-hour bucket.

The global ID/time rank displacement is huge (sd 4800 positions) and carries nothing
(partial -0.004): the two orderings differ globally. The signal measured at EDDF was strictly
LOCAL -- inside a fixed 1-hour take-off window, MVT_ID tracked BLOCK_TIME (+0.7322) better than
MVT_TIME (+0.7050) and the target at -0.1611.

So rank within (airport, hour) only. Both ranks are percentile ranks inside the same small
bucket, so the feature is bounded in [-1,1] and carries no calendar level.
"""
from config import RANKING, DATA_DIR, WORK_DIR
import os, numpy as np, pandas as pd, glob, gc, pickle, pyarrow.parquet as pq
D=DATA_DIR
OUT=(WORK_DIR + "/")
parts=[]
for f in sorted(glob.glob(os.path.join(D, "training_2025-*.parquet")))+[os.path.join(D, RANKING)]:
    t=pq.read_table(f,columns=["MVT_ID_mvt","ADEP_mvt","PHASE_mvt","MVT_TIME_UTC_mvt"]).to_pandas()
    parts.append(t[t.PHASE_mvt.values=="DEP"])
t=pd.concat(parts,ignore_index=True); del parts; gc.collect()
t["tt"]=pd.to_datetime(t.MVT_TIME_UTC_mvt,utc=True).dt.as_unit("ns").astype("int64")/1e9
t["mid"]=t.MVT_ID_mvt.astype("int64")
apts=list(pickle.load(open(OUT+"vocab.pkl","rb"))["APT"])
t["apt"]=t.ADEP_mvt.map({v:i for i,v in enumerate(apts)})
t=t[t.apt.notna()].drop_duplicates("mid")
dep=pq.read_table(OUT+"dep.parquet",columns=["mvt_id","is_rank","y"]).to_pandas()
dep["mvt_id"]=dep.mvt_id.astype("int64")
K=t.set_index("mid").reindex(dep.mvt_id.values).reset_index(drop=True)
out={}
for W,tag in [(3600.0,"1h"),(1800.0,"30m")]:
    g=pd.DataFrame({"apt":K.apt.values,"b":np.floor(K.tt.values/W),
                    "mid":K.index.values if False else dep.mvt_id.values.astype(np.float64),
                    "tt":K.tt.values})
    grp=g.groupby(["apt","b"])
    ri=grp["mid"].rank(pct=True).values
    rt=grp["tt"].rank(pct=True).values
    cnt=grp["mid"].transform("size").values
    d=(ri-rt).astype(np.float32); d[cnt<4]=np.nan
    out["idrank_minus_timerank_"+tag]=d
    out["bucket_n_"+tag]=cnt.astype(np.float32)
pd.DataFrame(out).to_parquet(OUT+"f23.parquet",index=False)
print("f23 saved",list(out.keys()),flush=True)
tr=~dep.is_rank.values; y=dep.y.values.astype(float)
da=pq.read_table(OUT+"f1.parquet",columns=["d_aobt"]).column(0).to_numpy().astype(float)
b=pq.read_table(OUT+"dep.parquet",columns=["t_blk","t_aobt"]).to_pandas()
dl=b.t_aobt.values-b.t_blk.values
for k,v in out.items():
    if k.startswith("bucket"): continue
    m=tr&np.isfinite(v)&np.isfinite(y); m2=m&np.isfinite(dl)&np.isfinite(da)
    r_d=np.corrcoef(v[m2],dl[m2])[0,1]; r_a=np.corrcoef(v[m2],da[m2])[0,1]
    rda=np.corrcoef(da[m2],dl[m2])[0,1]
    part=(r_d-r_a*rda)/np.sqrt(max((1-r_a**2)*(1-rda**2),1e-12))
    print("  %-30s r(y)=%+.4f  r(delta)=%+.4f  PARTIAL(delta|d_aobt)=%+.4f  n=%d"%(
        k,np.corrcoef(v[m],y[m])[0,1],r_d,part,m.sum()),flush=True)
v=out["idrank_minus_timerank_1h"]
print("\nper-airport r(y):")
ap=K.apt.values
for i,nm in enumerate(apts):
    m=tr&(ap==i)&np.isfinite(v)&np.isfinite(y)&(y<=5000)
    if m.sum()<2000: continue
    print("   %-5s r=%+.4f  n=%d"%(nm,np.corrcoef(v[m],y[m])[0,1],m.sum()),end="")
print()
print("\ndrift: train mean %+.4f sd %.4f | rank mean %+.4f sd %.4f | nan tr %.3f rk %.3f"%(
    np.nanmean(v[tr]),np.nanstd(v[tr]),np.nanmean(v[~tr]),np.nanstd(v[~tr]),
    np.isnan(v[tr]).mean(),np.isnan(v[~tr]).mean()))
