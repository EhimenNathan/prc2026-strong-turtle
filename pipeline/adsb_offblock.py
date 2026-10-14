"""Off-block from adsb.lol ground tracks. For a linked departure (icao24, ADEP, MVT): ground points at ADEP in [MVT-30h, MVT+120];
stops = runs of consecutive stationary samples (gs<1 kt, or a >=120 s sampling gap with <60 m displacement); stand dwell = last stop >= STOPMIN s
before take-off; t_off = first sample after that stop. x_off = MVT - t_off. Also: n_stops after dwell, longest later stop, first ground sample."""
import numpy as np, pandas as pd
def hav(la1,lo1,la2,lo2):
    p=np.pi/180; a=np.sin((la2-la1)*p/2)**2+np.cos(la1*p)*np.cos(la2*p)*np.sin((lo2-lo1)*p/2)**2; return 12742e3*np.arcsin(np.sqrt(a))
def offblock_feats(g, mvt, STOPMIN=600):
    """g: ground samples (t,lat,lon,gs) of one aircraft at the airport, sorted by t, restricted to [mvt-30h, mvt+120]"""
    t=g.t.values; n=len(t)
    if n<3: return None
    la=g.lat.values.astype(float); lo=g.lon.values.astype(float); gs=np.nan_to_num(g.gs.values.astype(float),nan=0)
    # takeoff side: keep samples before mvt+120
    stat=gs<1.0
    # stationary interval i->i+1: both slow, or long gap with small displacement
    dt=np.diff(t); dd=hav(la[:-1],lo[:-1],la[1:],lo[1:])
    still=(stat[:-1]&stat[1:]&(dd<60))|((dt>=120)&(dd<60))
    # runs of still intervals
    stops=[]; i=0
    while i<n-1:
        if still[i]:
            j=i
            while j<n-1 and still[j]: j+=1
            stops.append((t[i],t[j],i,j)); i=j
        else: i+=1
    long=[s for s in stops if s[1]-s[0]>=STOPMIN]
    f={"n":n,"t_first":t[0],"t_last":t[-1]}
    if long:
        s=long[-1]; f["t_off"]=s[1]; f["dwell"]=s[1]-s[0]; f["stand_lat"]=la[s[3]]; f["stand_lon"]=lo[s[3]]
        later=[x for x in stops if x[0]>=s[1]]; f["n_later_stops"]=len(later); f["max_later_stop"]=max([x[1]-x[0] for x in later],default=0)
        f["later_stop_time"]=sum(x[1]-x[0] for x in later)
        if len(long)>=2: f["t_off_prev"]=long[-2][1]; f["dwell_prev"]=long[-2][1]-long[-2][0]
    mv=np.where(gs>=3)[0]
    if len(mv): f["t_firstmove"]=t[mv[0]]
    return f
