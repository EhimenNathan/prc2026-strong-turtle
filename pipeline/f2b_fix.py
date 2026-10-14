"""Fix the surface-occupancy features.

Bug: PB (pushback times) dropped NaNs while TO (take-off times) did not, so
#{PB<=t} - #{TO<=t} drifted negative in proportion to the cumulative number of
AOBT-missing rows - a calendar-time proxy, not a congestion measure, which
extrapolates badly into the 2026 ranking period.

Fix: restrict BOTH arrays to the same subset (departures with a known pushback),
so the difference is a genuine count of aircraft between pushback and take-off.
"""
from config import DATA_DIR, WORK_DIR, RANKING
import numpy as np, pandas as pd, gc
import pyarrow.parquet as pq
OUT=(WORK_DIR + "/")
d=pq.read_table(OUT+"dep.parquet",columns=["APT","t_mvt","t_aobt"]).to_pandas()
p=np.load(OUT+"p_est.npy")
n=len(d); apt=d.APT.values; t=d.t_mvt.values.astype(np.float64); ao=d.t_aobt.values.astype(np.float64)
del d; gc.collect()
out={k:np.full(n,np.nan,np.float32) for k in ["taxiing_at_pb","taxiing_at_to","pb_rank_in_q"]}
for a in np.unique(apt):
    pos=np.where(apt==a)[0]
    has=np.isfinite(ao[pos])
    PB=np.sort(ao[pos][has]); TO=np.sort(t[pos][has])     # SAME subset -> no drift
    tt=t[pos].astype(np.float64); pp=p[pos].astype(np.float64)
    out["taxiing_at_to"][pos]=(np.searchsorted(PB,tt,"right")-np.searchsorted(TO,tt,"right")).astype(np.float32)
    out["taxiing_at_pb"][pos]=(np.searchsorted(PB,pp,"right")-np.searchsorted(TO,pp,"right")).astype(np.float32)-1
    out["pb_rank_in_q"][pos] =(np.searchsorted(PB,pp,"left") -np.searchsorted(TO,pp,"left")).astype(np.float32)
    print("apt",a,flush=True)
f2=pq.read_table(OUT+"f2.parquet").to_pandas()
for k,v in out.items():
    print(f"  {k}: old median={np.nanmedian(f2[k].values):9.1f} -> new median={np.nanmedian(v):7.1f}  "
          f"new min={np.nanmin(v):6.1f} max={np.nanmax(v):7.1f}")
    f2[k]=v
f2.to_parquet(OUT+"f2.parquet",index=False)
print("f2 repaired",flush=True)
