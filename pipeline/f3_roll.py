from config import DATA_DIR, WORK_DIR, RANKING
import pandas as pd, numpy as np, gc
OUT=(WORK_DIR + "/")
d = pd.read_parquet(OUT+"dep.parquet", columns=["APT","RUNWAY_mvt","STAND_mvt","t_mvt"])
prox = pd.read_parquet(OUT+"f1.parquet", columns=["p_est_off"]).p_est_off.values.astype(np.float64)
n=len(d); apt=d.APT.values; rwy=d.RUNWAY_mvt.values; t=d.t_mvt.values
stand=d.STAND_mvt.values; del d; gc.collect()

def roll_prev(vals, ks):
    """mean of previous k non-nan values (excludes self), nan-aware, on a sorted 1-D array"""
    v=np.nan_to_num(vals,nan=0.0); m=(~np.isnan(vals)).astype(np.float64)
    cs=np.concatenate([[0.0],np.cumsum(v)]); cm=np.concatenate([[0.0],np.cumsum(m)])
    out={}
    idx=np.arange(len(vals))
    for k in ks:
        lo=np.maximum(idx-k,0)
        s=cs[idx]-cs[lo]; c=cm[idx]-cm[lo]
        out[k]=np.where(c>=3, s/np.maximum(c,1), np.nan)
    return out

KS=[10,30,100]
res={f"roll_proxy_{k}":np.full(n,np.nan,np.float32) for k in KS}
res.update({f"roll_proxy_rwy_{k}":np.full(n,np.nan,np.float32) for k in [10,30]})
res["gap_prev_same_stand"]=np.full(n,np.nan,np.float32)
res["roll_proxy_fwd30"]=np.full(n,np.nan,np.float32)   # next-30 regime (hindsight ok)

for a in np.unique(apt):
    pos=np.where(apt==a)[0]
    order=pos[np.argsort(t[pos],kind="mergesort")]
    v=prox[order]
    r=roll_prev(v,KS)
    for k in KS: res[f"roll_proxy_{k}"][order]=r[k].astype(np.float32)
    rf=roll_prev(v[::-1],[30])[30][::-1]
    res["roll_proxy_fwd30"][order]=rf.astype(np.float32)
    # per runway
    rw=rwy[order]
    for w in np.unique(rw):
        m=rw==w; sub=order[m]
        rr=roll_prev(prox[sub],[10,30])
        for k in [10,30]: res[f"roll_proxy_rwy_{k}"][sub]=rr[k].astype(np.float32)
    # previous departure from same stand
    st=stand[order]; tt=t[order]
    ordst=np.lexsort((tt,st)); s2=st[ordst]; t2=tt[ordst]
    gap=np.full(len(s2),np.nan)
    same=np.concatenate([[False], s2[1:]==s2[:-1]])
    gap[1:]=np.where(same[1:], t2[1:]-t2[:-1], np.nan)
    res["gap_prev_same_stand"][order[ordst]]=gap.astype(np.float32)
    print("apt",a,flush=True)
pd.DataFrame(res).to_parquet(OUT+"f3.parquet", index=False)
print("f3 saved", len(res), flush=True)
