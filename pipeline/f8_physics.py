"""Physics-guided + signal-processing features.

Classic taxi-out decomposition:  taxi = unimpeded_taxi(stand,runway) + queuing_delay.
Plus queueing-theory load terms and EWMA-filtered congestion state at several time constants.
"""
from config import DATA_DIR, WORK_DIR, RANKING
import numpy as np, pandas as pd, gc, pickle
import pyarrow.parquet as pq
OUT=(WORK_DIR + "/")
d=pq.read_table(OUT+"dep.parquet",columns=["APT","RUNWAY_mvt","STAND_mvt","t_mvt","t_sch","is_rank"]).to_pandas()
f1=pq.read_table(OUT+"f1.parquet",columns=["p_est_off","d_aobt"]).to_pandas()
f6=pq.read_table(OUT+"f6.parquet",columns=["stand_pier"]).to_pandas()
p=np.load(OUT+"p_est.npy")
n=len(d); apt=d.APT.values; rwy=d.RUNWAY_mvt.values; st=d.STAND_mvt.values
pier=np.nan_to_num(f6.stand_pier.values,nan=-1).astype(np.int64)
t=d.t_mvt.values.astype(np.float64); tsch=d.t_sch.values.astype(np.float64)
prox=f1.d_aobt.values.astype(np.float64)          # observed takeoff-minus-pushback
istr=~d.is_rank.values
del d,f1,f6; gc.collect()

R={}
def newcol(k): R[k]=np.full(n,np.nan,np.float32); return R[k]

# ---------- 1. unimpeded (free-flow) taxi time per stand->runway, from TRAINING rows only ----------
# low percentile of the observed proxy = physical taxi with no queue
def unimpeded(keys, name, q=10, minn=20):
    code=np.zeros(n,np.int64)
    for v in keys: code=code*(int(v.max())+2)+(v+1)
    dfk=pd.DataFrame({"c":code,"v":prox})
    fit=istr&np.isfinite(prox)
    g=dfk[fit].groupby("c")["v"]
    stat=g.quantile(q/100.0); cnt=g.size()
    stat=stat[cnt>=minn]
    m=pd.Series(stat)
    out=m.reindex(code).to_numpy()
    R[name]=out.astype(np.float32)
    return out
u_sr=unimpeded([st.astype(np.int64),rwy.astype(np.int64)],"unimp_stand_rwy")
u_pr=unimpeded([pier,rwy.astype(np.int64)],"unimp_pier_rwy")
u_s =unimpeded([st.astype(np.int64)],"unimp_stand")
u_r =unimpeded([rwy.astype(np.int64)],"unimp_rwy")
best=np.where(np.isfinite(u_sr),u_sr,np.where(np.isfinite(u_pr),u_pr,u_r))
R["unimp_best"]=best.astype(np.float32)
# excess over free-flow = the queuing component actually observed
R["excess_over_unimp"]=(prox-best).astype(np.float32)
R["ratio_over_unimp"]=(prox/np.maximum(best,60.0)).astype(np.float32)
print("unimpeded done",flush=True)

# ---------- 2. queueing-theory load terms, per airport/runway ----------
for k in ["svc_rate_rwy","svc_rate_apt","rho_rwy","exp_wait_rwy","cap_rwy","demand_next30"]:
    newcol(k)
f2=pq.read_table(OUT+"f2.parquet",columns=["q_ahead","q_ahead_rwy","rwy_to_before_1800",
                                           "to_before_1800","eobt_after_p_1800"]).to_pandas()
qa=f2.q_ahead.values.astype(np.float64); qar=f2.q_ahead_rwy.values.astype(np.float64)
rwy30=f2.rwy_to_before_1800.values.astype(np.float64); apt30=f2.to_before_1800.values.astype(np.float64)
dem30=f2.eobt_after_p_1800.values.astype(np.float64)
del f2; gc.collect()
svc_r=rwy30/1800.0                                   # departures per second on my runway
svc_a=apt30/1800.0
R["svc_rate_rwy"]=svc_r.astype(np.float32); R["svc_rate_apt"]=svc_a.astype(np.float32)
R["exp_wait_rwy"]=(qar/np.maximum(svc_r,1e-4)).astype(np.float32)   # queue length / service rate
R["exp_wait_apt"]=(qa/np.maximum(svc_a,1e-4)).astype(np.float32)
# capacity = airport-runway 99th pct of observed 30-min throughput (training only)
codeR=(apt.astype(np.int64)*1000+rwy.astype(np.int64))
cap=pd.Series(rwy30[istr]).groupby(codeR[istr]).quantile(0.99)
capv=cap.reindex(codeR).to_numpy()
R["cap_rwy"]=capv.astype(np.float32)
R["rho_rwy"]=(rwy30/np.maximum(capv,1.0)).astype(np.float32)         # utilisation
R["demand_next30"]=dem30.astype(np.float32)
R["dem_cap_ratio"]=(dem30/np.maximum(capv,1.0)).astype(np.float32)
print("queueing done",flush=True)

# ---------- 3. EWMA-filtered congestion state (multiple time constants) ----------
for hl in [300,900,3600]:
    R[f"ewma_excess_{hl}"]=np.full(n,np.nan,np.float32)
    R[f"ewma_prox_{hl}"]=np.full(n,np.nan,np.float32)
exc=prox-best
for a in np.unique(apt):
    pos=np.where(apt==a)[0]
    o=pos[np.argsort(t[pos],kind="mergesort")]
    times=pd.to_datetime(t[o].astype("int64"),unit="s")
    for src,tag in [(prox,"prox"),(exc,"excess")]:
        s_=pd.Series(src[o])
        for hl in [300,900,3600]:
            # time-decayed mean of PREVIOUS flights only (shift excludes self)
            e=s_.ewm(halflife=pd.Timedelta(seconds=hl),times=times,ignore_na=True).mean().shift(1)
            R[f"ewma_{tag}_{hl}"][o]=e.to_numpy().astype(np.float32)
    print("ewma apt",a,flush=True)
pd.DataFrame(R).to_parquet(OUT+"f8.parquet",index=False)
print("f8 saved",list(R.keys()),flush=True)
