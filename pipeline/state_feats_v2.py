"""State features v2 (observable in 2026): pier-level 30-min state (deps, arrs, delay median + deviations), runway mixed-mode
(arrivals on same runway in the 30-min cell), runway lead/lag (departures on same runway next/previous cell), 15-min airport windows.
Pier = stand string with trailing digits stripped to its leading part (letters, or first digit for numeric stands)."""
from config import WORK_DIR
from config import RANKING, DATA_DIR, WORK_DIR
import numpy as np, pandas as pd, glob, gc, os, pyarrow.parquet as pq, re
D=DATA_DIR; OUT=(WORK_DIR.rstrip("/\\") + '/')
APTS=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH","LTFM"]
def ns(s):
    v=pd.to_datetime(s,utc=True).dt.as_unit("ns"); x=v.astype("int64").to_numpy()/1e9; x[v.isna().to_numpy()]=np.nan; return x
def pier(s):
    s=str(s)
    m=re.match(r"^([A-Za-z]+)",s)
    return m.group(1) if m else (s[0] if s and s!="None" else "?")
for tag,files in [("2025",sorted(glob.glob(os.path.join(D,"training_2025-*.parquet")))),("2026",[os.path.join(D,RANKING)])]:
    parts=[]
    for f in files:
        t=pq.read_table(f,columns=["MVT_ID_mvt","PHASE_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","RUNWAY_mvt","STAND_mvt"]).to_pandas()
        t["apt"]=np.where(t.PHASE_mvt=="DEP",t.ADEP_mvt,t.ADES_mvt); t=t[t.apt.isin(APTS)]
        mv=ns(t.MVT_TIME_UTC_mvt)
        parts.append(pd.DataFrame({"id":t.MVT_ID_mvt.values.astype("int64"),"ph":t.PHASE_mvt.values,"apt":t.apt.values,"t":mv,
            "dsch":mv-ns(t.SCHED_TIME_UTC_mvt),"rw":t.RUNWAY_mvt.astype(str).values,"pier":[pier(s) for s in t.STAND_mvt.values]}))
        del t; gc.collect()
    A=pd.concat(parts,ignore_index=True).drop_duplicates(["id","ph"]); del parts
    A=A[np.isfinite(A.t)]
    A["c30"]=np.floor(A.t/1800).astype("int64"); A["c15"]=np.floor(A.t/900).astype("int64"); A["hod"]=pd.to_datetime(A.t,unit="s",utc=True).dt.hour
    dep=A[A.ph=="DEP"].copy(); arr=A[A.ph=="ARR"]
    out=dep[["id","apt","rw","pier","c30","c15","hod"]].copy()
    # pier-level
    gd=dep.groupby(["apt","pier","c30"]); out["pier_dep30"]=gd.id.transform("size").values; out["pier_dsch30"]=gd.dsch.transform("median").values
    pa=arr.groupby(["apt","pier","c30"]).size().rename("pier_arr30")
    out=out.join(pa,on=["apt","pier","c30"]); out["pier_arr30"]=out.pier_arr30.fillna(0)
    for c in ["pier_dep30","pier_arr30","pier_dsch30"]:
        out[c+"_dev"]=out[c]-out.groupby(["apt","pier","hod"])[c].transform("median")
    # runway mixed-mode and lead/lag
    ra=arr.groupby(["apt","rw","c30"]).size().rename("rw_arr30"); out=out.join(ra,on=["apt","rw","c30"]); out["rw_arr30"]=out.rw_arr30.fillna(0)
    rd=dep.groupby(["apt","rw","c30"]).size().rename("n")
    for lab,sh in [("rw_dep_next30",1),("rw_dep_prev30",-1)]:
        s=rd.copy(); s.index=pd.MultiIndex.from_arrays([s.index.get_level_values(0),s.index.get_level_values(1),s.index.get_level_values(2)-sh])
        out=out.join(s.rename(lab),on=["apt","rw","c30"]); out[lab]=out[lab].fillna(0)
    # 15-min airport windows
    g15=dep.groupby(["apt","c15"]); out["apt_dep15"]=g15.id.transform("size").values; out["apt_dsch15"]=g15.dsch.transform("median").values
    out["apt_arr15"]=out.join(arr.groupby(["apt","c15"]).size().rename("a"),on=["apt","c15"])["a"].fillna(0).values
    for c in ["apt_dep15","apt_dsch15","apt_arr15","rw_arr30","rw_dep_next30"]:
        out[c+"_dev"]=out[c]-out.groupby(["apt","hod"])[c].transform("median")
    keep=[c for c in out.columns if c not in ("apt","rw","pier","c30","c15","hod")]
    out[keep].astype({k:"float32" for k in keep if k!="id"}).to_parquet(OUT+"state_feats2_%s.parquet"%tag,index=False)
    print(tag,len(out),"rows",len(keep)-1,"v2 features",flush=True); del A,dep,arr,out; gc.collect()
print("STATEFEATS2 DONE")
