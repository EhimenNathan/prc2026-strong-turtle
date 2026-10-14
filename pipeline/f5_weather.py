"""Join METAR weather to each departure (latest observation at or before takeoff)."""
from config import DATA_DIR, WORK_DIR, RANKING
import pandas as pd, numpy as np, glob, os, pickle, gc, pyarrow.parquet as pq
OUT=(WORK_DIR + "/")
EPOCH0 = pd.Timestamp("2024-12-01", tz="UTC").value // 10**9
vocab=pickle.load(open(OUT+"vocab.pkl","rb"))
apt_names=list(vocab["APT"])

d = pq.read_table(OUT+"dep.parquet", columns=["APT","t_mvt","RUNWAY_mvt"]).to_pandas()
f4 = pq.read_table(OUT+"f4.parquet", columns=["rwy_head_deg"]).to_pandas()
n=len(d); apt=d.APT.values; t=d.t_mvt.values.astype(np.float64); head=f4.rwy_head_deg.values
del d,f4; gc.collect()

NUM=["tmpf","dwpf","relh","drct","sknt","gust","p01i","alti","vsby","skyl1",
     "ice_accretion_1hr","ice_accretion_3hr","feel"]
OUTN=["temp_c","dewp_c","relh","wdir","wspd_kt","gust_kt","precip_mm","alti","vis_km","ceil_ft",
      "ice1h","ice3h","feels_c","wx_snow","wx_freez","wx_rain","wx_ts","wx_fog","lowvis",
      "deice_risk","crosswind","headwind","wx_any","obs_age_min"]
res={k: np.full(n, np.nan, np.float32) for k in OUTN}

for ai, name in enumerate(apt_names):
    f=OUT+f"metar/{name}.csv"
    if not os.path.exists(f): print("MISSING",name,flush=True); continue
    w=pd.read_csv(f, low_memory=False)
    w["ts"]=pd.to_datetime(w["valid"],utc=True).view("int64")/1e9 - EPOCH0
    w=w.sort_values("ts")
    for c in NUM:
        w[c]=pd.to_numeric(w.get(c), errors="coerce")
    wx=w.get("wxcodes").fillna("").astype(str).str.upper() if "wxcodes" in w else pd.Series([""]*len(w))
    w["wx_snow"]=wx.str.contains("SN|SG|PL|GS|IC").astype(float)
    w["wx_freez"]=wx.str.contains("FZ").astype(float)
    w["wx_rain"]=wx.str.contains("RA|DZ").astype(float)
    w["wx_ts"]=wx.str.contains("TS").astype(float)
    w["wx_fog"]=wx.str.contains("FG|BR|HZ").astype(float)
    w["wx_any"]=(wx.str.strip()!="").astype(float)
    w["temp_c"]=(w.tmpf-32)*5/9; w["dewp_c"]=(w.dwpf-32)*5/9
    w["feels_c"]=(w.feel-32)*5/9
    w["vis_km"]=w.vsby*1.60934
    w["precip_mm"]=w.p01i*25.4
    w["lowvis"]=(w.vis_km<1.5).astype(float)
    w["deice_risk"]=((w.temp_c<=3.0)&((w.precip_mm>0)|(w.wx_snow>0)|(w.wx_freez>0)|(w.relh>85))).astype(float)
    W=w.ts.values.astype(np.float64)
    pos=np.where(apt==ai)[0]
    if len(pos)==0 or len(W)==0: continue
    idx=np.searchsorted(W, t[pos], "right")-1
    ok=idx>=0
    src={"temp_c":"temp_c","dewp_c":"dewp_c","relh":"relh","wdir":"drct","wspd_kt":"sknt",
         "gust_kt":"gust","precip_mm":"precip_mm","alti":"alti","vis_km":"vis_km","ceil_ft":"skyl1",
         "ice1h":"ice_accretion_1hr","ice3h":"ice_accretion_3hr","feels_c":"feels_c",
         "wx_snow":"wx_snow","wx_freez":"wx_freez","wx_rain":"wx_rain","wx_ts":"wx_ts",
         "wx_fog":"wx_fog","lowvis":"lowvis","deice_risk":"deice_risk","wx_any":"wx_any"}
    ii=np.clip(idx,0,len(W)-1)
    for dst,s in src.items():
        v=w[s].values.astype(np.float64)[ii]
        res[dst][pos]=np.where(ok, v, np.nan).astype(np.float32)
    res["obs_age_min"][pos]=np.where(ok,(t[pos]-W[ii])/60.0,np.nan).astype(np.float32)
    # wind components relative to the departure runway
    hd=head[pos]; wd=res["wdir"][pos]; ws=res["wspd_kt"][pos]
    ang=np.deg2rad(wd-hd)
    res["crosswind"][pos]=(ws*np.sin(ang)).astype(np.float32)
    res["headwind"][pos]=(ws*np.cos(ang)).astype(np.float32)
    print("wx",name,len(w),"obs ->",len(pos),"deps",flush=True)
    del w; gc.collect()
res["crosswind"]=np.abs(res["crosswind"])
pd.DataFrame(res).to_parquet(OUT+"f5.parquet", index=False)
print("f5 saved",flush=True)
