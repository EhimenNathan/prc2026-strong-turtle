"""State-space (Kalman) estimation of the latent airport surface-congestion state.

The congestion state is OBSERVABLE in the ranking period (take-off times, landings,
arrival in-block times and AOBT_3 are all given), so a filter/smoother is legitimate.

Local linear trend model per airport, run over the departure stream:
    level_t = level_{t-1} + slope_{t-1}*dt + w1      (random walk with drift)
    slope_t = slope_{t-1}                    + w2
    obs_t   = level_t                        + v     (observed d_aobt / arrival taxi-in)

Emits: filtered level & slope, posterior variance, innovation, normalised innovation,
RTS-smoothed level (uses future movements - available here), and a robust
"disrupted regime" probability from the innovation stream.
All values are shifted so a flight never observes itself.
"""
from config import DATA_DIR, WORK_DIR, RANKING
import numpy as np, pandas as pd, gc, pickle
import pyarrow.parquet as pq
OUT=(WORK_DIR + "/")

def llt_filter(t, z, q_level=2.5e-4, q_slope=1e-8, r_obs=None, huber=3.0):
    """Local linear trend KF with Student-t style robust gating. Returns filtered+smoothed state."""
    n=len(t)
    lvl=np.full(n,np.nan); slp=np.full(n,np.nan); Pv=np.full(n,np.nan)
    inno=np.full(n,np.nan); ninno=np.full(n,np.nan)
    ok=np.isfinite(z)
    if ok.sum()<20: return lvl,slp,Pv,inno,ninno,lvl.copy()
    r = r_obs if r_obs is not None else np.nanvar(np.diff(z[ok]))/2.0
    r = max(r, 1.0)
    x=np.array([np.nanmedian(z[ok][:50]),0.0])
    P=np.array([[r,0.0],[0.0,1e-6]])
    F=np.eye(2); Q=np.zeros((2,2)); H=np.array([1.0,0.0])
    xs=np.zeros((n,2)); Ps=np.zeros((n,2,2)); xp=np.zeros((n,2)); Pp=np.zeros((n,2,2)); dts=np.zeros(n)
    prev=t[0]
    for i in range(n):
        dt=max(t[i]-prev,0.0); prev=t[i]; dts[i]=dt
        F[0,1]=dt
        Q[0,0]=q_level*dt; Q[1,1]=q_slope*dt
        x=F@x; P=F@P@F.T+Q
        xp[i]=x; Pp[i]=P
        # state BEFORE incorporating this flight -> no self-observation
        lvl[i]=x[0]; slp[i]=x[1]; Pv[i]=P[0,0]
        if ok[i]:
            S=P[0,0]+r; e=z[i]-x[0]
            inno[i]=e; ninno[i]=e/np.sqrt(S)
            rr=r*max(1.0,(abs(e)/(huber*np.sqrt(S)))**2)   # robust: inflate noise for outliers
            S=P[0,0]+rr
            K=P[:,0]/S
            x=x+K*e; P=P-np.outer(K,P[0,:])
        xs[i]=x; Ps[i]=P
    # RTS smoother (legitimate: full movement history of the ranking period is given)
    sm=xs.copy()
    for i in range(n-2,-1,-1):
        F[0,1]=dts[i+1]
        try: C=Ps[i]@F.T@np.linalg.inv(Pp[i+1])
        except np.linalg.LinAlgError: continue
        sm[i]=xs[i]+C@(sm[i+1]-xp[i+1])
    smooth=np.concatenate([[np.nan],sm[:-1,0]])   # shift: use neighbours, not self
    return lvl,slp,Pv,inno,ninno,smooth

d=pq.read_table(OUT+"dep.parquet",columns=["APT","t_mvt"]).to_pandas()
f1=pq.read_table(OUT+"f1.parquet",columns=["d_aobt"]).to_pandas()
f8=pq.read_table(OUT+"f8.parquet",columns=["excess_over_unimp"]).to_pandas()
arr=pq.read_table(OUT+"arr.parquet",columns=["apt","t_in","taxi_in"]).to_pandas()
n=len(d); apt=d.APT.values; t=d.t_mvt.values.astype(np.float64)
prox=f1.d_aobt.values.astype(np.float64); exc=f8.excess_over_unimp.values.astype(np.float64)
del d,f1,f8; gc.collect()
NM=["kf_level","kf_slope","kf_var","kf_inno","kf_ninno","kf_smooth",
    "kfx_level","kfx_slope","kfx_inno","kfx_smooth","kf_regime",
    "kfa_level","kfa_slope","kfa_smooth","kf_resid_vs_level"]
R={k:np.full(n,np.nan,np.float32) for k in NM}
for a in np.unique(apt):
    pos=np.where(apt==a)[0]
    o=pos[np.argsort(t[pos],kind="mergesort")]
    tt=t[o]
    lv,sl,pv,inn,ninn,sm=llt_filter(tt,prox[o])
    R["kf_level"][o]=lv; R["kf_slope"][o]=sl*3600; R["kf_var"][o]=pv
    R["kf_inno"][o]=inn; R["kf_ninno"][o]=ninn; R["kf_smooth"][o]=sm
    R["kf_resid_vs_level"][o]=prox[o]-lv
    lv2,sl2,_,inn2,_,sm2=llt_filter(tt,exc[o])
    R["kfx_level"][o]=lv2; R["kfx_slope"][o]=sl2*3600; R["kfx_inno"][o]=inn2; R["kfx_smooth"][o]=sm2
    # disrupted-regime probability: EWMA of |normalised innovation| exceeding 2 sigma
    flag=pd.Series((np.abs(ninn)>2.0).astype(float))
    R["kf_regime"][o]=flag.ewm(halflife=pd.Timedelta(seconds=1800),
        times=pd.to_datetime(tt.astype("int64"),unit="s"),ignore_na=True).mean().shift(1).to_numpy()
    # arrival taxi-in stream (fully observed in the ranking period)
    A=arr[arr.apt.values==a].dropna(subset=["t_in"]).sort_values("t_in")
    if len(A)>50:
        at=A.t_in.values.astype(np.float64); az=A.taxi_in.values.astype(np.float64)
        al,asl,_,_,_,asm=llt_filter(at,az)
        for src,dst in [(al,"kfa_level"),(asl,"kfa_slope"),(asm,"kfa_smooth")]:
            idx=np.searchsorted(at,tt,"left")-1
            v=np.where(idx>=0,src[np.clip(idx,0,len(src)-1)],np.nan)
            R[dst][o]=(v*3600 if dst=="kfa_slope" else v)
    print("kalman apt",a,flush=True)
pd.DataFrame(R).to_parquet(OUT+"f9.parquet",index=False)
print("f9 saved",NM,flush=True)
