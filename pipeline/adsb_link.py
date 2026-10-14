"""Link departures to adsb.lol aircraft by take-off time (no callsign needed). Take-off event = first airborne sample with alt>=100 ft
that follows a ground sample of the same aircraft at the same airport within 900 s; t_to = that sample time. Each departure takes the
nearest unused take-off at its ADEP within +-TOL s (greedy by |dt|)."""
import numpy as np, pandas as pd
def takeoffs(G):
    G=G.sort_values(["icao","apt","t"]); out=[]
    for (ic,ap),g in G.groupby(["icao","apt"],sort=False):
        t=g.t.values; a=g.alt.values; gr=a==-1
        if not gr.any(): continue
        last_g=-1e18
        for i in range(len(t)):
            if gr[i]: last_g=t[i]
            elif a[i]>=100 and t[i]-last_g<=900:
                out.append((ic,ap,t[i],t[i]-last_g)); last_g=-1e18
        # ground-roll take-offs: last ground sample >= 60 kt with no further ground sample for 300 s (elevation-proof: works where
        # climb-out samples were filtered away, e.g. LEMD at ~2000 ft); kept only if no airborne-detected take-off within 120 s after it
        gsv=np.nan_to_num(g.gs.values.astype(float),nan=0); tg=t[gr]; fast=gr&(gsv>=60)
        air_to=[o[2] for o in out if o[0]==ic and o[1]==ap]
        for i in np.where(fast)[0]:
            nxt=tg[tg>t[i]]
            if len(nxt) and nxt[0]-t[i]<300: continue
            if any(0<=x-t[i]<=120 for x in air_to): continue
            out.append((ic,ap,t[i],-1.0))       # gap_to=-1 marks a ground-roll event
    return pd.DataFrame(out,columns=["icao","apt","t_to","gap_to"])
def link(dep, TO, TOL=120):
    """dep: DataFrame mvt_id, apt, mvt. returns mvt_id, icao, t_to, dt"""
    # banded join: bucket times by TOL seconds; a take-off within +-TOL of the departure lies in the same or an adjacent bucket
    B=pd.concat([TO.assign(b=np.floor(TO.t_to/TOL).astype("int64")+k) for k in (-1,0,1)])
    M=dep.assign(b=np.floor(dep.mvt/TOL).astype("int64")).merge(B,on=["apt","b"]); M["dt"]=M.t_to-M.mvt; M=M[M.dt.abs()<=TOL]; M["ad"]=M.dt.abs()
    M=M.sort_values("ad"); used_d=set(); used_t=set(); keep=[]
    for r in M.itertuples():
        k=(r.icao,r.t_to)
        if r.mvt_id in used_d or k in used_t: continue
        used_d.add(r.mvt_id); used_t.add(k); keep.append(r.Index)
    return M.loc[keep,["mvt_id","icao","t_to","dt"]]
