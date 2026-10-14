"""Live surface-congestion index from arrival taxi-in times + same-stand turnaround."""
from config import DATA_DIR, WORK_DIR, RANKING
import pandas as pd, numpy as np, gc, pyarrow.parquet as pq
OUT=(WORK_DIR + "/")
d=pq.read_table(OUT+"dep.parquet",columns=["APT","STAND_mvt","t_mvt"]).to_pandas()
p=np.load(OUT+"p_est.npy")
arr=pq.read_table(OUT+"arr.parquet").to_pandas()
n=len(d); apt=d.APT.values; stand=d.STAND_mvt.values; t=d.t_mvt.values.astype(np.float64)
del d; gc.collect()
NM=["arr_taxiin_mean_1800","arr_taxiin_mean_3600","arr_taxiin_med_3600","arr_taxiin_n_3600",
    "arr_taxiin_mean_7200","stand_prev_arr_taxiin","stand_turnaround","stand_prev_arr_delay",
    "arr_taxiin_mean_rwy3600"]
res={k:np.full(n,np.nan,np.float32) for k in NM}

def win_mean(ts_sorted, vals_sorted, q, w):
    v=np.nan_to_num(vals_sorted,nan=0.0); m=(~np.isnan(vals_sorted)).astype(np.float64)
    cs=np.concatenate([[0.0],np.cumsum(v)]); cm=np.concatenate([[0.0],np.cumsum(m)])
    hi=np.searchsorted(ts_sorted,q,"left"); lo=np.searchsorted(ts_sorted,q-w,"left")
    s=cs[hi]-cs[lo]; c=cm[hi]-cm[lo]
    return np.where(c>0, s/np.maximum(c,1), np.nan), c

for a in np.unique(apt):
    pos=np.where(apt==a)[0]
    A=arr[arr.apt.values==a].dropna(subset=["t_in"])
    if len(A)==0: continue
    o=np.argsort(A.t_in.values,kind="mergesort")
    ti=A.t_in.values[o].astype(np.float64); tx=A.taxi_in.values[o].astype(np.float64)
    st=A.stand.values[o]; tl=A.t_land.values[o].astype(np.float64); ts_=A.t_sch.values[o].astype(np.float64)
    q=p[pos].astype(np.float64)
    for w,key in [(1800,"arr_taxiin_mean_1800"),(3600,"arr_taxiin_mean_3600"),(7200,"arr_taxiin_mean_7200")]:
        mn,c=win_mean(ti,tx,q,w); res[key][pos]=mn.astype(np.float32)
        if w==3600: res["arr_taxiin_n_3600"][pos]=c.astype(np.float32)
    # rolling median of the last 25 arrival taxi-in values (vectorized)
    hi=np.searchsorted(ti,q,"left")
    rm=pd.Series(tx).rolling(25,min_periods=1).median().to_numpy()
    res["arr_taxiin_med_3600"][pos]=np.where(hi>0, rm[np.clip(hi-1,0,len(rm)-1)], np.nan).astype(np.float32)
    # most recent arrival in-block at MY stand before my pushback (vectorized merge_asof)
    L=pd.DataFrame({"stand":stand[pos].astype(np.int64),"q":q,"ord":np.arange(len(pos))}).sort_values("q",kind="mergesort")
    R=pd.DataFrame({"stand":st.astype(np.int64),"t_in":ti,"tx":tx,"tl":tl,"ts":ts_}).sort_values("t_in",kind="mergesort")
    M=pd.merge_asof(L,R,left_on="q",right_on="t_in",by="stand",direction="backward")
    M=M.sort_values("ord")
    res["stand_prev_arr_taxiin"][pos]=M.tx.to_numpy(dtype=np.float32)
    res["stand_turnaround"][pos]=(M.q.to_numpy()-M.t_in.to_numpy()).astype(np.float32)
    res["stand_prev_arr_delay"][pos]=(M.tl.to_numpy()-M.ts.to_numpy()).astype(np.float32)
    print("apt",a,"arr",len(A),flush=True)
    del A; gc.collect()
pd.DataFrame(res).to_parquet(OUT+"f7.parquet",index=False)
print("f7 saved",flush=True)
