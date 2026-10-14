"""FLIGHT_ID as a latent clock.

FLIGHT_ID_mvt is assigned sequentially by the Network Manager as flights are activated, so
it encodes WHEN the record entered the system - information about the off-block time that
exists nowhere in the feature columns. It was read by build_data.py but never stored.

Within a take-off window, rho(FLIGHT_ID, hidden BLOCK_TIME) = +0.67 and
rho(FLIGHT_ID, TAXITIME) = -0.26: among aircraft departing together, a higher ID means a
later off-block, hence a shorter taxi.

Every mapping below is built from FLIGHT_ID plus timestamps that are known in BOTH the
training and ranking periods, so nothing here uses the hidden target.
"""
from config import RANKING, DATA_DIR, WORK_DIR
import numpy as np, pandas as pd, glob, gc, os
EP=pd.Timestamp("1970-01-01",tz="UTC")
D=(DATA_DIR + "/")
OUT=(WORK_DIR + "/")
E0=pd.Timestamp("2024-12-01",tz="UTC").value//10**9
cols=["MVT_ID_mvt","FLIGHT_ID_mvt","PHASE_mvt","ADEP_mvt","ADES_mvt",
      "MVT_TIME_UTC_mvt","EOBT_1_flt","SCHED_TIME_UTC_mvt"]
parts=[]
for f in sorted(glob.glob(D+"training_*.parquet"))+[D+RANKING]:
    t=pd.read_parquet(f,columns=cols)
    t=t[t.PHASE_mvt.eq("DEP")]
    parts.append(t); del t; gc.collect()
A=pd.concat(parts,ignore_index=True); del parts; gc.collect()
for c,n in [("MVT_TIME_UTC_mvt","mvt"),("EOBT_1_flt","eobt"),("SCHED_TIME_UTC_mvt","sch")]:
    A[n]=(pd.to_datetime(A[c],utc=True)-EP).dt.total_seconds()-E0
print("DEP rows:",len(A),"with FLIGHT_ID:",A.FLIGHT_ID_mvt.notna().sum(),flush=True)

# ---- 1. global monotone clock: FLIGHT_ID -> planned off-block, fitted on train+rank ----
S=A.dropna(subset=["FLIGHT_ID_mvt","eobt"]).sort_values("FLIGHT_ID_mvt")
fid_s=S.FLIGHT_ID_mvt.values.astype(np.float64)
clock=pd.Series(S.eobt.values).rolling(401,center=True,min_periods=30).median().values
ok=np.isfinite(clock)
fid_k=fid_s[ok]; clk_k=clock[ok]
u,idx=np.unique(fid_k,return_index=True)          # strictly increasing knots for interp
implied=np.interp(A.FLIGHT_ID_mvt.values.astype(np.float64),u,clk_k[idx],
                  left=np.nan,right=np.nan)
A["fid_clock"]=implied
print("clock fitted; residual sd vs EOBT =",np.nanstd(A.eobt.values-A.fid_clock.values).round(1),"s",flush=True)

# ---- 2. features ----
R={}
R["fid_anchor"]=(A.mvt.values-A.fid_clock.values).astype(np.float32)      # implied taxi from the clock
R["fid_clock_minus_eobt"]=(A.fid_clock.values-A.eobt.values).astype(np.float32)
R["fid_clock_minus_sch"]=(A.fid_clock.values-A.sch.values).astype(np.float32)
# local residual: my ID vs peers taking off around me at the same airport
apt=A.ADEP_mvt.values; t=A.mvt.values; fid=A.FLIGHT_ID_mvt.values.astype(np.float64)
loc=np.full(len(A),np.nan); rnk=np.full(len(A),np.nan); locw=np.full(len(A),np.nan)
for a in pd.unique(apt):
    m=np.where(apt==a)[0]
    o=m[np.argsort(t[m],kind="mergesort")]
    f_=fid[o]; 
    med=pd.Series(f_).rolling(41,center=True,min_periods=8).median().values
    loc[o]=f_-med
    rr=pd.Series(f_).rolling(41,center=True,min_periods=8).rank(pct=True).values
    rnk[o]=rr
    sd=pd.Series(f_).rolling(41,center=True,min_periods=8).std().values
    locw[o]=np.where(sd>0,(f_-med)/sd,np.nan)
R["fid_local_resid"]=loc.astype(np.float32)
R["fid_local_rank"]=rnk.astype(np.float32)
R["fid_local_z"]=locw.astype(np.float32)
# convert the local ID residual into seconds using the local ID rate
rate=np.full(len(A),np.nan)
for a in pd.unique(apt):
    m=np.where(apt==a)[0]; o=m[np.argsort(t[m],kind="mergesort")]
    df=pd.Series(fid[o]).rolling(41,center=True,min_periods=8).apply(lambda s: (s.iloc[-1]-s.iloc[0]),raw=False).values
    dt=pd.Series(t[o]).rolling(41,center=True,min_periods=8).apply(lambda s: (s.iloc[-1]-s.iloc[0]),raw=False).values
    rate[o]=np.where(dt>0,df/np.maximum(dt,1),np.nan)
R["fid_resid_seconds"]=(loc/np.where(np.isfinite(rate)&(rate>0),rate,np.nan)).astype(np.float32)
out=pd.DataFrame(R); out["mvt_id"]=A.MVT_ID_mvt.values
# align to dep.parquet row order
import pyarrow.parquet as pq
order=pq.read_table(OUT+"dep.parquet",columns=["mvt_id"]).column(0).to_numpy()
out=out.set_index("mvt_id").reindex(order).reset_index(drop=True)
out.to_parquet(OUT+"f13.parquet",index=False)
print("f13 saved",list(out.columns),"rows",len(out),flush=True)
print("null frac:\n",out.isna().mean().round(3).to_string(),flush=True)
