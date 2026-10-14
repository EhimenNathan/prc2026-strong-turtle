"""Queue-discipline and network features.

 * rank_shift      : my rank in the TAKE-OFF order minus my rank in the PUSHBACK order.
                     Positive => aircraft that pushed back after me departed before me,
                     i.e. I was overtaken (held, long route, or gave way). Distinct from
                     q_ahead, which counts departures during my taxi regardless of order.
 * since_cfg_change: minutes since the airport's dominant departure runway changed - a
                     configuration swap causes a transient surge in taxi times.
 * dest_*          : congestion at the DESTINATION around my arrival. A saturated
                     destination causes ATFM ground holding, which can hold aircraft at
                     the holding point rather than the stand.
"""
from config import DATA_DIR, WORK_DIR, RANKING
import numpy as np, pandas as pd, gc, pickle
import pyarrow.parquet as pq
OUT=(WORK_DIR + "/")
v=pickle.load(open(OUT+"vocab.pkl","rb"))
ap=list(v["APT"]); ades=list(v["ADES_mvt"])
d=pq.read_table(OUT+"dep.parquet",columns=["APT","RUNWAY_mvt","ADES_mvt","t_mvt","t_arvt3","t_arvt1"]).to_pandas()
p=np.load(OUT+"p_est.npy")
n=len(d); apt=d.APT.values; rwy=d.RUNWAY_mvt.values; t=d.t_mvt.values.astype(np.float64)
R={k:np.full(n,np.nan,np.float32) for k in
   ["rank_shift","rank_shift_rwy","since_cfg_change","dest_arr_rate_1h","dest_dep_rate_1h","dest_known"]}

for a in np.unique(apt):
    pos=np.where(apt==a)[0]
    tt=t[pos]; pp=p[pos]
    # ---- overtaking: rank in take-off order vs rank in pushback order ----
    rt=np.empty(len(pos)); rt[np.argsort(tt,kind="mergesort")]=np.arange(len(pos))
    rp=np.empty(len(pos)); rp[np.argsort(pp,kind="mergesort")]=np.arange(len(pos))
    R["rank_shift"][pos]=(rt-rp).astype(np.float32)
    rw=rwy[pos]
    for g in np.unique(rw):
        m=rw==g; k=int(m.sum())
        if k<5: continue
        a1=np.empty(k); a1[np.argsort(tt[m],kind="mergesort")]=np.arange(k)
        a2=np.empty(k); a2[np.argsort(pp[m],kind="mergesort")]=np.arange(k)
        R["rank_shift_rwy"][pos[m]]=(a1-a2).astype(np.float32)
    # ---- time since the dominant departure runway changed ----
    o=np.argsort(tt,kind="mergesort")
    B=1800.0
    bid=np.floor(tt[o]/B).astype(np.int64)
    df=pd.DataFrame({"b":bid,"r":rw[o]})
    dom=df.groupby("b")["r"].agg(lambda s:s.value_counts().index[0])
    bs=dom.index.values; dv=dom.values
    chg=np.concatenate([[True],dv[1:]!=dv[:-1]])
    chg_t=bs[chg]*B
    idx=np.searchsorted(chg_t,tt[o],"right")-1
    R["since_cfg_change"][pos[o]]=np.where(idx>=0,(tt[o]-chg_t[np.clip(idx,0,len(chg_t)-1)])/60.0,np.nan).astype(np.float32)
    print("apt",a,flush=True)
    gc.collect()

# ---- destination congestion (only for the 10 dataset airports) ----
name2i={nm:i for i,nm in enumerate(ap)}
dest_code=np.full(n,-1,np.int64)
adn=np.array([ades[i] if i>=0 else "" for i in range(len(ades))]+[""])
cc=d.ADES_mvt.values.copy(); cc[cc<0]=len(ades)
dn=adn[cc]
for k,i in name2i.items(): dest_code[dn==k]=i
R["dest_known"]=(dest_code>=0).astype(np.float32)
arr=pq.read_table(OUT+"arr.parquet",columns=["apt","t_land"]).to_pandas().dropna(subset=["t_land"])
eta=d.t_arvt3.values.astype(np.float64)
eta=np.where(np.isfinite(eta),eta,d.t_arvt1.values.astype(np.float64))
dep_t=t
for i,nm in enumerate(ap):
    m=(dest_code==i)&np.isfinite(eta)
    if m.sum()<50: continue
    L=np.sort(arr.t_land.values[arr.apt.values==i].astype(np.float64))
    D=np.sort(t[apt==i])
    q=eta[m]
    R["dest_arr_rate_1h"][m]=(np.searchsorted(L,q+1800,"left")-np.searchsorted(L,q-1800,"left")).astype(np.float32)
    R["dest_dep_rate_1h"][m]=(np.searchsorted(D,q+1800,"left")-np.searchsorted(D,q-1800,"left")).astype(np.float32)
pd.DataFrame(R).to_parquet(OUT+"f12.parquet",index=False)
print("f12 saved",list(R.keys()),flush=True)
