from config import DATA_DIR, WORK_DIR, RANKING
import pandas as pd, numpy as np, gc
OUT=(WORK_DIR + "/")
EPOCH0 = pd.Timestamp("2024-12-01", tz="UTC").value // 10**9
tc=["t_mvt","t_blk","t_sch","t_lobt","t_iobt","t_eobt","t_arvt1","t_aobt","t_arvt3"]
d = pd.read_parquet(OUT+"dep.parquet", columns=tc)
T = d.t_mvt.values
p = d.t_aobt.values.copy()
for fb in ["t_eobt","t_lobt","t_sch"]:
    m=np.isnan(p); p[m]=d[fb].values[m]
m=np.isnan(p); p[m]=T[m]-900.0
o = {}
o["p_est_off"]  = (T - p)                       # taxi estimate from best pushback proxy
o["d_aobt"]  = T - d.t_aobt.values
o["d_eobt"]  = T - d.t_eobt.values
o["d_lobt"]  = T - d.t_lobt.values
o["d_iobt"]  = T - d.t_iobt.values
o["d_sch"]   = T - d.t_sch.values
o["aobt_m_eobt"] = d.t_aobt.values - d.t_eobt.values
o["aobt_m_sch"]  = d.t_aobt.values - d.t_sch.values
o["aobt_m_lobt"] = d.t_aobt.values - d.t_lobt.values
o["lobt_m_iobt"] = d.t_lobt.values - d.t_iobt.values
o["eobt_m_sch"]  = d.t_eobt.values - d.t_sch.values
o["air_actual"]  = d.t_arvt3.values - T
o["blk_plan"]    = d.t_arvt1.values - d.t_eobt.values
o["blk_actual"]  = d.t_arvt3.values - d.t_aobt.values
o["taxi_planned"]= o["blk_plan"] - o["air_actual"]
o["blk_slip"]    = o["blk_actual"] - o["blk_plan"]
o["arvt1_m_arvt3"] = d.t_arvt1.values - d.t_arvt3.values
o["aobt_isna"]   = np.isnan(d.t_aobt.values).astype(np.float32)
ts = pd.to_datetime(T + EPOCH0, unit="s", utc=True)
o["hour"]=ts.hour.values.astype(np.float32)
o["minofday"]=(ts.hour.values*60+ts.minute.values).astype(np.float32)
o["dow"]=ts.dayofweek.values.astype(np.float32)
o["month"]=ts.month.values.astype(np.float32)
o["doy"]=ts.dayofyear.values.astype(np.float32)
o["sin_tod"]=np.sin(2*np.pi*o["minofday"]/1440); o["cos_tod"]=np.cos(2*np.pi*o["minofday"]/1440)
o["sin_doy"]=np.sin(2*np.pi*o["doy"]/365.25);   o["cos_doy"]=np.cos(2*np.pi*o["doy"]/365.25)
out = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k,v in o.items()})
np.save(OUT+"p_est.npy", p.astype(np.float64))
out.to_parquet(OUT+"f1.parquet", index=False)
print("f1", out.shape, flush=True)
