"""Arrivals table: taxi-in times are available in BOTH train and ranking -> live congestion index."""
from config import RANKING, DATA_DIR, WORK_DIR
import pandas as pd, numpy as np, glob, os, gc, pickle
D=(DATA_DIR + "/")
OUT=(WORK_DIR + "/")
EPOCH0=pd.Timestamp("2024-12-01",tz="UTC").value//10**9
vocab=pickle.load(open(OUT+"vocab.pkl","rb"))
def tsec(s):
    s=pd.to_datetime(s,utc=True).dt.as_unit("ns")
    v=s.astype("int64").to_numpy().astype("float64")/1e9-EPOCH0
    v[s.isna().to_numpy()]=np.nan; return v
parts=[]
for f in sorted(glob.glob(D+"training_*.parquet"))+[D+RANKING]:
    df=pd.read_parquet(f,columns=["PHASE_mvt","ADES_mvt","STAND_mvt","RUNWAY_mvt",
        "MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","TAXITIME_SEC_mvt","WK_TBL_CAT_flt"])
    df=df[df.PHASE_mvt.eq("ARR")]
    parts.append(pd.DataFrame({
        "apt": vocab["APT"].get_indexer(df.ADES_mvt.to_numpy(dtype=object)).astype(np.int16),
        "stand": vocab["STAND_mvt"].get_indexer(df.STAND_mvt.to_numpy(dtype=object)).astype(np.int32),
        "rwy": vocab["RUNWAY_mvt"].get_indexer(df.RUNWAY_mvt.to_numpy(dtype=object)).astype(np.int16),
        "t_land": tsec(df.MVT_TIME_UTC_mvt), "t_in": tsec(df.BLOCK_TIME_UTC_mvt),
        "t_sch": tsec(df.SCHED_TIME_UTC_mvt),
        "taxi_in": df.TAXITIME_SEC_mvt.to_numpy().astype(np.float32)}))
    print(os.path.basename(f), len(parts[-1]), flush=True)
    del df; gc.collect()
arr=pd.concat(parts,ignore_index=True); del parts; gc.collect()
arr.to_parquet(OUT+"arr.parquet",index=False)
print("arr",arr.shape,"taxi_in null",arr.taxi_in.isna().mean().round(4),flush=True)
