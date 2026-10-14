from config import DATA_DIR, WORK_DIR, RANKING
import pandas as pd, numpy as np, gc
OUT=(WORK_DIR + "/")
d = pd.read_parquet(OUT+"dep.parquet", columns=["APT","RUNWAY_mvt","t_mvt"])
p = np.load(OUT+"p_est.npy")
evt = pd.read_parquet(OUT+"events.parquet")
n = len(d); print("rows", n, flush=True)
apt_all = d.APT.values; rwy_all = d.RUNWAY_mvt.values; t_all = d.t_mvt.values
del d; gc.collect()

WIN=[900,1800,3600]
names=[]
for w in WIN:
    names += [f"{k}_{w}" for k in ["to_before","to_after","ld_before","ld_after","ib_before",
                                   "pb_before","eobt_after_p","pb_before_p","to_before_p"]]
for w in [300,900,1800]:
    names += [f"rwy_to_before_{w}", f"rwy_to_after_{w}"]
names += ["q_ahead","q_ahead_rwy","taxiing_at_pb","taxiing_at_to",
          "gap_prev_to","gap_next_to","gap_prev_to_rwy","gap_next_to_rwy","pb_rank_in_q"]
C = {k: np.full(n, np.nan, dtype=np.float32) for k in names}

def cnt(arr, lo, hi):
    return (np.searchsorted(arr,hi,"left") - np.searchsorted(arr,lo,"left")).astype(np.float32)
def clean(a):
    a = a[~np.isnan(a)]; a.sort(); return a.astype(np.float64)

for a in np.unique(apt_all):
    pos = np.where(apt_all==a)[0]
    e = evt[evt.apt.values==a]
    de = e[e.isdep.values]; ae = e[~e.isdep.values]
    TO=clean(de.t_mvt.values); PB=clean(de.t_aobt.values); EO=clean(de.t_eobt.values)
    LD=clean(ae.t_mvt.values); IB=clean(ae.t_blk.values)
    t=t_all[pos].astype(np.float64); pp=p[pos].astype(np.float64)
    for w in WIN:
        C[f"to_before_{w}"][pos]=cnt(TO,t-w,t);   C[f"to_after_{w}"][pos]=cnt(TO,t,t+w)
        C[f"ld_before_{w}"][pos]=cnt(LD,t-w,t);   C[f"ld_after_{w}"][pos]=cnt(LD,t,t+w)
        C[f"ib_before_{w}"][pos]=cnt(IB,t-w,t);   C[f"pb_before_{w}"][pos]=cnt(PB,t-w,t)
        C[f"eobt_after_p_{w}"][pos]=cnt(EO,pp,pp+w)
        C[f"pb_before_p_{w}"][pos]=cnt(PB,pp-w,pp)
        C[f"to_before_p_{w}"][pos]=cnt(TO,pp-w,pp)
    C["q_ahead"][pos]=(np.searchsorted(TO,t,"left")-np.searchsorted(TO,pp,"right")).astype(np.float32)
    C["taxiing_at_pb"][pos]=(np.searchsorted(PB,pp,"right")-np.searchsorted(TO,pp,"right")).astype(np.float32)-1
    C["taxiing_at_to"][pos]=(np.searchsorted(PB,t,"right")-np.searchsorted(TO,t,"right")).astype(np.float32)
    C["pb_rank_in_q"][pos]=(np.searchsorted(PB,pp,"left")-np.searchsorted(TO,pp,"left")).astype(np.float32)
    ip=np.searchsorted(TO,t,"left"); C["gap_prev_to"][pos]=np.where(ip>0, t-TO[np.clip(ip-1,0,len(TO)-1)], np.nan)
    ni=np.searchsorted(TO,t,"right"); C["gap_next_to"][pos]=np.where(ni<len(TO), TO[np.clip(ni,0,len(TO)-1)]-t, np.nan)
    rw_row=rwy_all[pos]
    for rw in np.unique(rw_row):
        m=rw_row==rw; sub=pos[m]; tt=t[m]; qq=pp[m]
        R=clean(de.t_mvt.values[de.rwy.values==rw])
        if len(R)==0: continue
        for w in [300,900,1800]:
            C[f"rwy_to_before_{w}"][sub]=cnt(R,tt-w,tt); C[f"rwy_to_after_{w}"][sub]=cnt(R,tt,tt+w)
        C["q_ahead_rwy"][sub]=(np.searchsorted(R,tt,"left")-np.searchsorted(R,qq,"right")).astype(np.float32)
        i1=np.searchsorted(R,tt,"left"); C["gap_prev_to_rwy"][sub]=np.where(i1>0, tt-R[np.clip(i1-1,0,len(R)-1)], np.nan)
        i2=np.searchsorted(R,tt,"right"); C["gap_next_to_rwy"][sub]=np.where(i2<len(R), R[np.clip(i2,0,len(R)-1)]-tt, np.nan)
    print("apt",a,"n",len(pos),flush=True)
    del e,de,ae,TO,PB,EO,LD,IB; gc.collect()
del evt; gc.collect()
pd.DataFrame(C).to_parquet(OUT+"f2.parquet", index=False)
print("f2 saved", len(C), "cols", flush=True)
