"""Structural / operational-interference features.

Physics of the departure process that generic congestion counts miss:
 * same-SID (route-direction) departures need a larger minimum departure interval (MDI)
 * a heavy pushing back blocks a pier/alleyway for everyone behind it
 * wake separation on the runway depends on the sequence of weight categories
 * ramp contention: simultaneous pushbacks in the same pier, arrivals inbound to my pier
 * airline bank structure: a carrier pushing many aircraft at once self-congests
All computed from quantities known at prediction time (take-off/pushback times, stands, wake).
"""
from config import DATA_DIR, WORK_DIR, RANKING
import numpy as np, pandas as pd, gc, pickle, re
import pyarrow.parquet as pq
OUT=(WORK_DIR + "/")
v=pickle.load(open(OUT+"vocab.pkl","rb"))
d=pq.read_table(OUT+"dep.parquet",columns=["APT","RUNWAY_mvt","STAND_mvt","ADES_mvt","t_mvt",
                                           "WK_TBL_CAT_flt","AIRCRAFT_OPERATOR_flt"]).to_pandas()
f6=pq.read_table(OUT+"f6.parquet",columns=["stand_pier","ades_region"]).to_pandas()
p=np.load(OUT+"p_est.npy")
n=len(d)
apt=d.APT.values.astype(np.int64); rwy=d.RUNWAY_mvt.values.astype(np.int64)
t=d.t_mvt.values.astype(np.float64)
pier=np.nan_to_num(f6.stand_pier.values,nan=-1).astype(np.int64)
reg=np.nan_to_num(f6.ades_region.values,nan=-1).astype(np.int64)
oper=d.AIRCRAFT_OPERATOR_flt.values.astype(np.int64)
wk=list(v["WK_TBL_CAT_flt"]); heavy_codes={i for i,s in enumerate(wk) if str(s).upper() in ("H","J")}
wkv=d.WK_TBL_CAT_flt.values.astype(np.int64)
isheavy=np.isin(wkv,list(heavy_codes)) if heavy_codes else np.zeros(n,bool)
del d,f6; gc.collect()

# arrival stands -> pier (same parsing as f6)
st=list(v["STAND_mvt"]); pmap={}; spref=np.full(len(st),-1,np.int64)
for i,s in enumerate(st):
    m=re.match(r"^([A-Za-z]*)",str(s)); q=m.group(1).upper() if m else ""
    if q not in pmap: pmap[q]=len(pmap)
    spref[i]=pmap[q]
arr=pq.read_table(OUT+"arr.parquet",columns=["apt","stand","t_in"]).to_pandas().dropna(subset=["t_in"])
arr["pier"]=np.where(arr.stand.values>=0, spref[np.clip(arr.stand.values,0,len(spref)-1)], -1)

NM=["sid_coincide_600","sid_coincide_rwy_600","sid_ahead_q","pier_push_300","pier_push_900",
    "heavy_pier_push_900","airline_push_900","airline_share_bank","heavy_ahead_rwy",
    "heavy_frac_ahead","arr_pier_next30","arr_pier_prev30","surface_accum_1800","pier_push_share"]
R={k:np.full(n,np.nan,np.float32) for k in NM}

def cnt(a,lo,hi):
    return (np.searchsorted(a,hi,"left")-np.searchsorted(a,lo,"left")).astype(np.float64)

for a in np.unique(apt):
    pos=np.where(apt==a)[0]
    tt=t[pos]; pp=p[pos]
    TO=np.sort(tt); PB=np.sort(pp)
    # --- same route-direction (SID proxy) ---
    for key,arr_name,win in [(reg[pos],"sid_coincide_600",600.0)]:
        out=np.full(len(pos),np.nan)
        for g in np.unique(key):
            m=key==g; T=np.sort(tt[m])
            out[m]=cnt(T,tt[m]-win,tt[m]+win)-1.0
        R["sid_coincide_600"][pos]=out
    key2=reg[pos]*1000+rwy[pos]
    out=np.full(len(pos),np.nan); outq=np.full(len(pos),np.nan)
    for g in np.unique(key2):
        m=key2==g; T=np.sort(tt[m])
        out[m]=cnt(T,tt[m]-600.0,tt[m]+600.0)-1.0
        outq[m]=(np.searchsorted(T,tt[m],"left")-np.searchsorted(T,pp[m],"right")).astype(np.float64)
    R["sid_coincide_rwy_600"][pos]=out; R["sid_ahead_q"][pos]=outq
    # --- ramp: pushback concurrency in my pier, heavies among them ---
    kp=pier[pos]
    o3=np.full(len(pos),np.nan); o9=np.full(len(pos),np.nan); oh=np.full(len(pos),np.nan)
    for g in np.unique(kp):
        m=kp==g; P=np.sort(pp[m])
        o3[m]=cnt(P,pp[m]-300.0,pp[m]+300.0)-1.0
        o9[m]=cnt(P,pp[m]-900.0,pp[m]+900.0)-1.0
        hm=m&isheavy[pos]
        if hm.sum(): 
            H=np.sort(pp[hm]); oh[m]=cnt(H,pp[m]-900.0,pp[m]+900.0)
        else: oh[m]=0.0
    R["pier_push_300"][pos]=o3; R["pier_push_900"][pos]=o9; R["heavy_pier_push_900"][pos]=oh
    allpb9=cnt(PB,pp-900.0,pp+900.0)
    R["pier_push_share"][pos]=o9/np.maximum(allpb9,1.0)
    # --- airline bank concurrency ---
    ko=oper[pos]; oa=np.full(len(pos),np.nan)
    for g in np.unique(ko):
        m=ko==g; P=np.sort(pp[m])
        oa[m]=cnt(P,pp[m]-900.0,pp[m]+900.0)-1.0
    R["airline_push_900"][pos]=oa
    R["airline_share_bank"][pos]=oa/np.maximum(allpb9,1.0)
    # --- wake separation: heavies that departed ahead of me on my runway ---
    kr=rwy[pos]; oh2=np.full(len(pos),np.nan); qr=np.full(len(pos),np.nan)
    for g in np.unique(kr):
        m=kr==g
        Tg=np.sort(tt[m]); qr[m]=(np.searchsorted(Tg,tt[m],"left")-np.searchsorted(Tg,pp[m],"right")).astype(np.float64)
        hm=m&isheavy[pos]
        H=np.sort(tt[hm]) if hm.sum() else np.array([])
        oh2[m]=(np.searchsorted(H,tt[m],"left")-np.searchsorted(H,pp[m],"right")).astype(np.float64) if len(H) else 0.0
    R["heavy_ahead_rwy"][pos]=oh2; R["heavy_frac_ahead"][pos]=oh2/np.maximum(qr,1.0)
    # --- downstream gate interference: arrivals inbound to MY pier ---
    A=arr[arr.apt.values==a]
    ap_=A.pier.values.astype(np.int64); at=A.t_in.values.astype(np.float64)
    onx=np.full(len(pos),np.nan); opv=np.full(len(pos),np.nan)
    for g in np.unique(kp):
        m=kp==g; S=np.sort(at[ap_==g])
        if len(S)==0: onx[m]=0.0; opv[m]=0.0; continue
        onx[m]=cnt(S,pp[m],pp[m]+1800.0); opv[m]=cnt(S,pp[m]-1800.0,pp[m])
    R["arr_pier_next30"][pos]=onx; R["arr_pier_prev30"][pos]=opv
    # --- net surface accumulation: pushbacks minus take-offs over the last 30 min ---
    R["surface_accum_1800"][pos]=cnt(PB,tt-1800.0,tt)-cnt(TO,tt-1800.0,tt)
    print("apt",a,flush=True)
    gc.collect()
pd.DataFrame(R).to_parquet(OUT+"f10.parquet",index=False)
print("f10 saved",NM,flush=True)
