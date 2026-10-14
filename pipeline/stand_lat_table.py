"""Per-(airport, stand, year) arrival recording latency (recorded in-block - ADS-B stand arrival) from arr_link.parquet (Jan/Jul arrivals),
mapped onto every departure of the same airport/stand/year. Output work/stand_lat.parquet: mvt_id, sl_med, sl_w60, sl_n, sl_iqr, sl_apt_med."""
from config import WORK_DIR
from config import DATA_DIR, WORK_DIR, RANKING
import glob,os,numpy as np,pandas as pd,pyarrow.parquet as pq
D0=DATA_DIR
ids=[];ph=[];st=[];ap=[];yr=[]
for f in sorted(glob.glob(os.path.join(D0,"*.parquet"))):
    if "submitting" in f: continue
    t=pq.read_table(f,columns=["MVT_ID_mvt","PHASE_mvt","STAND_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt"]).to_pandas()
    ids.append(t.MVT_ID_mvt.values.astype("int64")); ph.append(t.PHASE_mvt.values); st.append(t.STAND_mvt.astype(str).values)
    ap.append(np.where(t.PHASE_mvt.values=="DEP",t.ADEP_mvt.values,t.ADES_mvt.values)); yr.append(pd.to_datetime(t.MVT_TIME_UTC_mvt,utc=True).dt.year.values); del t
R=pd.DataFrame({"mvt_id":np.concatenate(ids),"phase":np.concatenate(ph),"stand":np.concatenate(st),"apt":np.concatenate(ap),"yr":np.concatenate(yr)}).drop_duplicates("mvt_id")
A=pd.read_parquet((WORK_DIR.rstrip("/\\") + '/adsblol/arr_link.parquet')); A=A[np.isfinite(A.t_stand)]; A=A.merge(R[["mvt_id","stand"]],on="mvt_id"); A["lat"]=A.blk-A.t_stand
g=A.groupby(["apt","stand","yr"]).lat
T=pd.DataFrame({"sl_med":g.median(),"sl_w60":g.apply(lambda s:(s.abs()<=60).mean()),"sl_n":g.size(),"sl_iqr":g.quantile(.75)-g.quantile(.25)}).reset_index()
T.loc[T.sl_n<5,["sl_med","sl_w60","sl_iqr"]]=np.nan
AP=A.groupby(["apt","yr"]).lat.median().rename("sl_apt_med").reset_index()
Dp=R[R.phase=="DEP"].merge(T,on=["apt","stand","yr"],how="left").merge(AP,on=["apt","yr"],how="left")
Dp[["mvt_id","sl_med","sl_w60","sl_n","sl_iqr","sl_apt_med"]].to_parquet((WORK_DIR.rstrip("/\\") + '/stand_lat.parquet'),index=False)
print("departures %d | with stand latency %.3f (2025 %.3f, 2026 %.3f)"%(len(Dp),Dp.sl_med.notna().mean(),Dp[Dp.yr==2025].sl_med.notna().mean(),Dp[Dp.yr==2026].sl_med.notna().mean()))
