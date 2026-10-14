"""Runway-configuration features via vectorized time-bucket groupbys."""
from config import DATA_DIR, WORK_DIR, RANKING
import pandas as pd, numpy as np, gc, pickle, pyarrow.parquet as pq
OUT=(WORK_DIR + "/")
BUCKET=1800.0
d = pq.read_table(OUT+"dep.parquet", columns=["APT","RUNWAY_mvt","t_mvt"]).to_pandas()
evt = pq.read_table(OUT+"events.parquet", columns=["apt","isdep","t_mvt","rwy"]).to_pandas()
n=len(d)
row=pd.DataFrame({"apt":d.APT.values.astype(np.int32),"rwy":d.RUNWAY_mvt.values.astype(np.int32),
                  "b":np.floor(d.t_mvt.values/BUCKET).astype(np.int64)})
del d; gc.collect()
evt=evt.dropna(subset=["t_mvt"])
evt["b"]=np.floor(evt.t_mvt.values/BUCKET).astype(np.int64)
dep_e=evt[evt.isdep.values]; arr_e=evt[~evt.isdep.values]

# counts per (apt, bucket, runway) for departures
g=dep_e.groupby(["apt","b","rwy"],sort=False).size().rename("c").reset_index()
tot=g.groupby(["apt","b"],sort=False)["c"].sum().rename("tot").reset_index()
ndist=g.groupby(["apt","b"],sort=False)["rwy"].nunique().rename("ndist").reset_index()
dom=g.sort_values("c").groupby(["apt","b"],sort=False).tail(1)[["apt","b","rwy"]].rename(columns={"rwy":"dom_rwy"})
arr_nd=arr_e.groupby(["apt","b"],sort=False)["rwy"].nunique().rename("arr_ndist").reset_index()
arr_tot=arr_e.groupby(["apt","b"],sort=False).size().rename("arr_tot").reset_index()

r=row.merge(g,on=["apt","b","rwy"],how="left").merge(tot,on=["apt","b"],how="left") \
     .merge(ndist,on=["apt","b"],how="left").merge(dom,on=["apt","b"],how="left") \
     .merge(arr_nd,on=["apt","b"],how="left").merge(arr_tot,on=["apt","b"],how="left")
assert len(r)==n, (len(r),n)
res={}
res["n_dep_rwy_active"]=r.ndist.values.astype(np.float32)
res["n_arr_rwy_active"]=r.arr_ndist.values.astype(np.float32)
res["my_rwy_share_dep"]=(r.c.values/np.maximum(r.tot.values,1)).astype(np.float32)
res["is_dominant_dep_rwy"]=(r.dom_rwy.values==r.rwy.values).astype(np.float32)
res["dep_in_bucket"]=r.tot.values.astype(np.float32)
res["arr_in_bucket"]=r.arr_tot.values.astype(np.float32)
res["my_rwy_cnt_bucket"]=r.c.values.astype(np.float32)

# previous-bucket dominant runway -> config change flag
prev=dom.copy(); prev["b"]=prev.b+1; prev=prev.rename(columns={"dom_rwy":"prev_dom"})
r2=row.merge(dom,on=["apt","b"],how="left").merge(prev,on=["apt","b"],how="left")
res["dep_rwy_config_changed"]=(r2.dom_rwy.values!=r2.prev_dom.values).astype(np.float32)

vocab=pickle.load(open(OUT+"vocab.pkl","rb")); rw=list(vocab["RUNWAY_mvt"])
head=np.full(len(rw),np.nan,np.float32)
for i,s in enumerate(rw):
    dg="".join(ch for ch in str(s) if ch.isdigit())
    if dg:
        v=int(dg[:2])
        if 1<=v<=36: head[i]=v*10.0
rv=row.rwy.values
res["rwy_head_deg"]=np.where(rv>=0, head[np.clip(rv,0,len(head)-1)], np.nan).astype(np.float32)
pd.DataFrame(res).to_parquet(OUT+"f4.parquet",index=False)
print("f4 saved",list(res.keys()),flush=True)
