"""v2 adsb.lol session features. Usage: python adsb_feats2.py <tag> <zip> [<zip> ...]
Link: callsign (when the extraction has one) matched to FLIGHT_mvt / CALLSIGN_flt at ADEP with take-off within +-300 s, else take-off time
(+-60 s, greedy). Session = ground samples after the previous clearly-airborne sample (<=30 h). Adds positions (first sample, final stop,
stand dwell) for stand-distance features built later. Output work/adsblol/f2_<tag>.parquet."""
from config import WORK_DIR
from config import DATA_DIR, WORK_DIR, RANKING
import sys, io, zipfile, pickle, numpy as np, pandas as pd, pyarrow.parquet as pq
from adsb_link import takeoffs, link
from adsb_offblock import hav
tag=sys.argv[1]; ZS=[zipfile.ZipFile(p) for p in sys.argv[2:]]
import os as _os
SKIP=set(pd.read_parquet((WORK_DIR.rstrip("/\\") + '/adsblol/f2_%s.parquet')%_os.environ["ADDON_OF"],columns=["mvt_id"]).mvt_id) if _os.environ.get("ADDON_OF") else None
IDX={}; SZ={}
for z in ZS:
    for i in z.infolist():
        n=i.filename
        if n.startswith("ground_") and i.file_size>SZ.get(n[7:17],-1): IDX[n[7:17]]=z; SZ[n[7:17]]=i.file_size   # prefer the non-empty copy of a day
apts=list(pickle.load(open((WORK_DIR.rstrip("/\\") + '/vocab.pkl'),"rb"))["APT"]); E0=pd.Timestamp("2024-12-01",tz="UTC").value//10**9
d=pq.read_table((WORK_DIR.rstrip("/\\") + '/dep.parquet'),columns=["mvt_id","APT","t_mvt","CALLSIGN_flt"]).to_pandas().drop_duplicates("mvt_id")
d["apt"]=np.array(apts)[d.APT.values]; d["mvt"]=d.t_mvt+E0; d["day"]=pd.to_datetime(d.mvt,unit="s",utc=True).dt.strftime("%Y-%m-%d")
FL=None
try:
    import glob, os
    D0=DATA_DIR
    FL=pd.concat([pq.read_table(f,columns=["MVT_ID_mvt","PHASE_mvt","FLIGHT_mvt"]).to_pandas() for f in sorted(glob.glob(os.path.join(D0,"*.parquet"))) if "submitting" not in f])
    FL=FL[FL.PHASE_mvt=="DEP"].drop_duplicates("MVT_ID_mvt"); FL=pd.Series(FL.FLIGHT_mvt.values,index=FL.MVT_ID_mvt.astype("int64").values)
except Exception as e: print("FLIGHT_mvt unavailable:",e)
d["flt"]=FL.reindex(d.mvt_id.values).values if FL is not None else None
def norm(s): return s.astype(str).str.upper().str.replace(" ","",regex=False)
def load(day):
    z=IDX.get(day)
    if z is None: return None
    G=pd.read_parquet(io.BytesIO(z.read("ground_%s.parquet"%day)))
    for c in ["cs","reg","typ"]:
        if c not in G: G[c]=None
    return G
def session_feats(g, mvt, t_to):
    air=g[(g.alt>=300)&(g.t<t_to-120)]; t_air=air.t.values[-1] if len(air) else np.nan
    if not np.isfinite(t_air):
        # no airborne samples (filtered at high-elevation airports): previous landing = fast ground sample followed by slow ones within 120 s
        gg=g[(g.alt==-1)&(g.t<t_to-120)]; tt=gg.t.values; vv=np.nan_to_num(gg.gs.values.astype(float),nan=0)
        land=[tt[i] for i in range(len(tt)-1) if vv[i]>=60 and tt[i+1]-tt[i]<120 and vv[i+1]<vv[i]]
        if land: t_air=land[-1]
    s=g[(g.alt==-1)&(g.t<t_to)&(g.t>(t_air if np.isfinite(t_air) else -np.inf))]
    f={"x_air_prev":mvt-t_air,"n_gs":len(s)}
    if len(s)==0: return f
    t=s.t.values; gs=np.nan_to_num(s.gs.values.astype(float),nan=0); la=s.lat.values.astype(float); lo=s.lon.values.astype(float)
    f.update(x_first=mvt-t[0],x_lastsample=mvt-t[-1],gs_first=gs[0],lat_first=la[0],lon_first=lo[0])
    gap=np.diff(t); brk=np.where(gap>300)[0]; st=t[brk[-1]+1] if len(brk) else t[0]; ib=brk[-1]+1 if len(brk) else 0
    f.update(x_lastblock=mvt-st,lat_lastblock=la[ib],lon_lastblock=lo[ib],n_gaps300=len(brk),max_gap=gap.max() if len(gap) else 0)
    still=np.where(gs<1)[0]
    if len(still):
        i=still[-1]; f["x_laststill"]=mvt-t[i]; f["lat_laststill"]=la[i]; f["lon_laststill"]=lo[i]
        j=i
        while j>0 and (gs[j-1]<1 or (t[j]-t[j-1]>=120 and hav(la[j-1],lo[j-1],la[j],lo[j])<60)): j-=1
        f["laststill_dur"]=t[i]-t[j]
    mv=np.where(gs>=3)[0]
    if len(mv): f["x_firstmove"]=mvt-t[mv[0]]; f["lat_firstmove"]=la[mv[0]]; f["lon_firstmove"]=lo[mv[0]]
    if len(t)>=2:
        dd=hav(la[:-1],lo[:-1],la[1:],lo[1:]); stl=((gs[:-1]<1)&(gs[1:]<1)&(dd<60))|((gap>=120)&(dd<60))
        stops=[]; i=0
        while i<len(t)-1:
            if stl[i]:
                j=i
                while j<len(t)-1 and stl[j]: j+=1
                stops.append((t[i],t[j],j)); i=j
            else: i+=1
        lg=[x for x in stops if x[1]-x[0]>=600]
        if lg:
            f.update(x_off=mvt-lg[-1][1],dwell=lg[-1][1]-lg[-1][0],n_long_stops=len(lg),x_off_first=mvt-lg[0][1],lat_off=la[lg[-1][2]],lon_off=lo[lg[-1][2]])
            f["stop_after_off"]=sum(b-a for a,b,_ in stops if a>=lg[-1][1])
            f["path_after_off"]=float(dd[t[:-1]>=lg[-1][1]].sum())
        f["path_m"]=float(dd.sum()); f["move_time"]=float(gap[(gs[:-1]>=3)].sum())
        f["stop_time_1h"]=float(sum(max(0,min(b,t_to)-max(a,mvt-3600)) for a,b,_ in stops if b>mvt-3600))
        f["n_stops"]=len(stops)
    return f
import os
def flush(month,rows):
    if rows: pd.DataFrame(rows).to_parquet((WORK_DIR.rstrip("/\\") + '/adsblol/f2_%s_%s.parquet')%(tag,month),index=False)
out=[]; cur=None
for day in sorted(IDX):
    if os.path.exists((WORK_DIR.rstrip("/\\") + '/adsblol/f2_%s_%s.parquet')%(tag,day[:7])): continue      # month already built (resume)
    if cur is not None and day[:7]!=cur:
        flush(cur,out); out=[]
    cur=day[:7]
    G=load(day)
    prev=(pd.Timestamp(day)-pd.Timedelta(days=1)).strftime("%Y-%m-%d"); Gp=load(prev)
    dep=d[d.day==day]
    if SKIP is not None: dep=dep[~dep.mvt_id.isin(SKIP)]
    if not len(dep): continue
    TO=takeoffs(G); ROLL={k:set(v.t_to) for k,v in TO[TO.gap_to==-1].groupby(["icao","apt"])}
    # callsign link first
    Lc=pd.DataFrame(columns=["mvt_id","icao","t_to","dt"])
    if G.cs.notna().any():
        cst=G[G.cs.notna()][["icao","apt","cs"]].drop_duplicates(); cst["k"]=norm(cst.cs); cst=cst[["icao","apt","k"]].drop_duplicates()
        T2=TO.merge(cst,on=["icao","apt"])          # one row per (take-off, callsign heard on that aircraft)
        rows=[]
        for key in ["CALLSIGN_flt","flt"]:
            X=dep[["mvt_id","apt","mvt",key]].dropna(subset=[key]); X=X.assign(k=norm(X[key]))
            M=X[["mvt_id","apt","mvt","k"]].merge(T2,on=["apt","k"]); M=M[(M.t_to-M.mvt).abs()<=300]; M["dt"]=M.t_to-M.mvt; rows.append(M)
        M=pd.concat(rows); M["ad"]=M.dt.abs(); M=M.sort_values("ad").drop_duplicates("mvt_id").drop_duplicates(["icao","t_to"])
        Lc=M[["mvt_id","icao","t_to","dt"]]
    rest=dep[~dep.mvt_id.isin(Lc.mvt_id)]
    TOr=TO.merge(Lc[["icao","t_to"]],how="left",indicator=True); TOr=TOr[TOr._merge=="left_only"].drop(columns="_merge")
    Lt=link(rest[["mvt_id","apt","mvt"]],TOr,TOL=60)
    L=pd.concat([Lc.assign(by_cs=1),Lt.assign(by_cs=0)]).merge(dep[["mvt_id","apt","mvt"]],on="mvt_id")
    GA=pd.concat([Gp,G]) if Gp is not None else G
    GA=GA[GA.icao.isin(set(L.icao))].sort_values("t"); gi=dict(tuple(GA.groupby(["icao","apt"])))
    for r in L.itertuples():
        g=gi.get((r.icao,r.apt))
        if g is None: continue
        g=g[(g.t>=r.mvt-108000)&(g.t<=r.t_to)]
        f=session_feats(g,r.mvt,r.t_to); f.update(mvt_id=r.mvt_id,dt_to=r.dt,by_cs=r.by_cs,icao=r.icao,roll_to=float(r.t_to in ROLL.get((r.icao,r.apt),())))
        tt=g.typ.dropna(); f["typ"]=tt.iloc[-1] if len(tt) else None
        out.append(f)
    print(day,"departures",len(dep),"linked",len(L),"(callsign %d)"%len(Lc),flush=True)
flush(cur,out); out=[]
import glob
F=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob((WORK_DIR.rstrip("/\\") + '/adsblol/f2_%s_20??-??.parquet')%tag))],ignore_index=True)
F.to_parquet((WORK_DIR.rstrip("/\\") + '/adsblol/f2_%s.parquet')%tag,index=False)
print("wrote",len(F),"rows; by airport:",F.merge(d[["mvt_id","apt"]],on="mvt_id").apt.value_counts().to_dict())
