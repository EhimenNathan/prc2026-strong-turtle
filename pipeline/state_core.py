"""Observable airport state features, independent of research submissions."""
import numpy as np, pandas as pd, gc, pickle, pyarrow.parquet as pq
from config import WORK_DIR
OUT = WORK_DIR.rstrip("/\\") + "/"
apts = list(pickle.load(open(OUT + "vocab.pkl", "rb"))["APT"])
AP9 = list(apts)
E0 = pd.Timestamp("2024-12-01", tz="UTC").value // 10**9
def secs(s):
    s=pd.to_datetime(s,utc=True).dt.as_unit("ns"); v=s.astype("int64").to_numpy()/1e9-E0; v[s.isna().to_numpy()]=np.nan; return v
COLS=["PHASE_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","TAXITIME_SEC_mvt","RUNWAY_mvt","WK_TBL_CAT_flt"]
VARS=["n_dep","dsch_mean","dsch_med","dsch_p90","dsch_sd","rw_nuniq","heavy_frac","n_arr","arr_anom"]
BASEF=[v+"_dev" for v in VARS]+["n_dep","n_arr"]
LAGV=["dsch_med_dev","arr_anom_dev","n_dep_dev","dsch_p90_dev"]; LAGF=[f"{v}_{s}" for v in LAGV for s in ["lag1","lead1","lag2"]]
def frames(files):
    dep=[];arr=[]
    for f in files:
        t=pq.read_table(f,columns=COLS).to_pandas(); A_=t[t.PHASE_mvt.values=="ARR"]; D_=t[t.PHASE_mvt.values=="DEP"]
        dep.append(pd.DataFrame({"apt":D_.ADEP_mvt.astype(str).values,"tm":secs(D_.MVT_TIME_UTC_mvt),"tsch":secs(D_.SCHED_TIME_UTC_mvt),
            "rw":D_.RUNWAY_mvt.astype(str).values,"wk":D_.WK_TBL_CAT_flt.astype(str).values}))
        arr.append(pd.DataFrame({"apt":A_.ADES_mvt.astype(str).values,"tm":secs(A_.MVT_TIME_UTC_mvt),
            "ti":pd.to_numeric(A_.TAXITIME_SEC_mvt,errors="coerce").values.astype(float)}))
        del t,A_,D_; gc.collect()
    DP=pd.concat(dep,ignore_index=True); AR=pd.concat(arr,ignore_index=True)
    DP=DP[DP.apt.isin(apts)&np.isfinite(DP.tm)].copy(); AR=AR[AR.apt.isin(apts)&np.isfinite(AR.tm)&np.isfinite(AR.ti)&(AR.ti>0)&(AR.ti<7200)].copy()
    DP["dsch"]=DP.tm-DP.tsch; return DP,AR
def build(DP,AR,width):
    for X in (DP,AR):
        X["c"]=np.floor(X.tm/width).astype("int64"); X["hod"]=pd.to_datetime(X.tm.values+E0,unit="s",utc=True).hour
    g=DP.groupby(["apt","c"],observed=True)
    ST=pd.DataFrame({"n_dep":g.size(),"dsch_mean":g.dsch.mean(),"dsch_med":g.dsch.median(),"dsch_p90":g.dsch.quantile(.9),
        "dsch_sd":g.dsch.std(),"rw_nuniq":g.rw.nunique(),"heavy_frac":g.wk.apply(lambda s:s.isin(["H","J"]).mean())})
    a=AR.join(AR.groupby(["apt","hod"],observed=True).ti.median().rename("nrm"),on=["apt","hod"]); a["anom"]=a.ti-a.nrm
    ga=a.groupby(["apt","c"],observed=True)
    ST=ST.join(pd.DataFrame({"n_arr":ga.size(),"arr_anom":ga.anom.mean()})).reset_index()
    ST["hod"]=pd.to_datetime(ST.c.values*width+E0,unit="s",utc=True).hour
    for v in VARS: ST[v+"_dev"]=ST[v]-ST.groupby(["apt","hod"],observed=True)[v].transform("median")
    ST=ST.sort_values(["apt","c"])
    for v in LAGV:
        gb=ST.groupby("apt",observed=True)[v]; ST[v+"_lag1"]=gb.shift(1); ST[v+"_lead1"]=gb.shift(-1); ST[v+"_lag2"]=gb.shift(2)
    return ST


def rows(files):
    out=[]
    for f in files:
        t=pq.read_table(f,columns=["MVT_ID_mvt","PHASE_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","RUNWAY_mvt","WK_TBL_CAT_flt"]).to_pandas()
        t=t[(t.PHASE_mvt=="DEP")&t.ADEP_mvt.isin(AP9)]
        out.append(pd.DataFrame({"id":t.MVT_ID_mvt.values.astype("int64"),"apt":t.ADEP_mvt.values,"t":secs(t.MVT_TIME_UTC_mvt),
            "dsch":secs(t.MVT_TIME_UTC_mvt)-secs(t.SCHED_TIME_UTC_mvt),"rw":t.RUNWAY_mvt.astype(str).values,"heavy":t.WK_TBL_CAT_flt.isin(["H","J"]).values.astype(np.float32)}))
        del t; gc.collect()
    return pd.concat(out,ignore_index=True).drop_duplicates("id")
def featurise(Rw,files):
    DP,AR=frames(files)
    ST={w:build(DP.copy(),AR.copy(),w) for w in [1800,3600,10800,86400]}
    del DP,AR; gc.collect()
    X=Rw.copy(); feats=[]
    for w in [1800,3600,10800,86400]:
        cols=[v+"_dev" for v in VARS]+([c for c in LAGF] if w in (1800,3600) else [])
        E=ST[w][["apt","c"]+cols].rename(columns={c:"%s@%d"%(c,w) for c in cols})
        X["c"]=np.floor(X.t/w).astype("int64"); X=X.merge(E,on=["apt","c"],how="left"); feats+=["%s@%d"%(c,w) for c in cols]
    X["c"]=np.floor(X.t/1800).astype("int64")
    g=X.groupby(["apt","rw","c"])
    X["rw_n30"]=g.id.transform("size"); X["rw_dsch_med30"]=g.dsch.transform("median")
    X["hod"]=pd.to_datetime(X.t,unit="s",utc=True).dt.hour; X["dow"]=pd.to_datetime(X.t,unit="s",utc=True).dt.dayofweek
    X["rw_n30_dev"]=X.rw_n30-X.groupby(["apt","rw","hod"]).rw_n30.transform("median")
    X["rw_dsch_dev"]=X.rw_dsch_med30-X.groupby(["apt","rw","hod"]).rw_dsch_med30.transform("median")
    X["apt_n30"]=X.groupby(["apt","c"]).id.transform("size"); X["rw_share"]=X.rw_n30/X.apt_n30
    X["aptc"]=pd.Categorical(X.apt,categories=apts).codes
    feats+=["rw_n30","rw_dsch_med30","rw_n30_dev","rw_dsch_dev","rw_share","hod","dow","dsch","heavy","aptc"]
    return X,feats
