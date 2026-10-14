"""Physical -> recorded transfer function per airport and year, measured on ARRIVALS (labelled in 2025 AND in the 2026 test period).
From the adsb.lol ground tracks (which include the inbound legs of the departing aircraft): landing = first ground sample after >=2 airborne
samples below 1500 ft in the previous 180 s; stand arrival = start of the first stop >= 600 s after landing (stationary samples or a >=120 s
sampling gap with <60 m displacement). Arrival records are linked by landing time (|MVT - t_land| <= 60 s, greedy nearest).
Outputs per airport x year: median / MAD of (recorded in-block - ADS-B stand arrival), share within 30/60/120 s, recorded vs physical taxi-in.
Usage: python adsb_arrivals.py <zip> [<zip> ...]  -> work/adsblol/arr_link.parquet (+ summary printed)"""
from config import WORK_DIR
from config import DATA_DIR, WORK_DIR, RANKING
import sys, io, zipfile, glob, os, gc, numpy as np, pandas as pd, pyarrow.parquet as pq
from adsb_offblock import hav
ZS = [zipfile.ZipFile(p) for p in sys.argv[1:]]; IDX = {}; SZ = {}
for z in ZS:
    for i in z.infolist():
        n = i.filename
        if n.startswith("ground_") and i.file_size > SZ.get(n[7:17], -1): IDX[n[7:17]] = z; SZ[n[7:17]] = i.file_size
MONTHS = os.environ.get("MONTHS", "01,07").split(",")
IDX = {k: v for k, v in IDX.items() if k[5:7] in MONTHS}          # scored months only (speed)
D0 = DATA_DIR
def ns(s): v = pd.to_datetime(s, utc=True).dt.as_unit("ns"); x = v.astype("int64").to_numpy() / 1e9; x[v.isna().to_numpy()] = np.nan; return x
AR = []
for f in sorted(glob.glob(os.path.join(D0, "*.parquet"))):
    if "submitting" in f: continue
    t = pq.read_table(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]).to_pandas()
    t = t[t.PHASE_mvt == "ARR"]; AR.append(pd.DataFrame({"mvt_id": t.MVT_ID_mvt.values.astype("int64"), "apt": t.ADES_mvt.values, "mvt": ns(t.MVT_TIME_UTC_mvt), "blk": ns(t.BLOCK_TIME_UTC_mvt)}))
    del t; gc.collect()
AR = pd.concat(AR).drop_duplicates("mvt_id"); AR = AR[np.isfinite(AR.mvt) & np.isfinite(AR.blk)]
AR["day"] = pd.to_datetime(AR.mvt, unit="s", utc=True).dt.strftime("%Y-%m-%d")
out = []
for day in sorted(IDX):
    G = pd.read_parquet(io.BytesIO(IDX[day].read("ground_%s.parquet" % day)), columns=["icao", "t", "lat", "lon", "alt", "gs", "apt"]).sort_values("t")
    ev = []
    for (ic, ap), g in G.groupby(["icao", "apt"], sort=False):
        t = g.t.values; a = g.alt.values; gr = a == -1; gs = np.nan_to_num(g.gs.values.astype(float), nan=0); la = g.lat.values.astype(float); lo = g.lon.values.astype(float)
        for i in np.where(gr[1:] & ~gr[:-1])[0] + 1:            # ground sample preceded by an airborne one
            prev = (t >= t[i] - 180) & (t < t[i]) & ~gr & (a < 1500)
            if prev.sum() < 2: continue
            j = i; stand = np.nan                                  # first stop >= 600 s after landing
            while j < len(t) - 1 and t[j] - t[i] < 3 * 3600:
                if not gr[j + 1]: break
                k = j;
                while k < len(t) - 1 and gr[k + 1] and ((gs[k] < 1 and gs[k + 1] < 1 and hav(la[k], lo[k], la[k + 1], lo[k + 1]) < 60) or (t[k + 1] - t[k] >= 120 and hav(la[k], lo[k], la[k + 1], lo[k + 1]) < 60)):
                    k += 1
                if t[k] - t[j] >= 600: stand = t[j]; break
                j = k + 1 if k > j else j + 1
            ev.append((ic, ap, t[i], stand))
    E = pd.DataFrame(ev, columns=["icao", "apt", "t_land", "t_stand"])
    A = AR[AR.day == day]
    M = A.merge(E, on="apt"); M["d"] = M.t_land - M.mvt; M = M[M.d.abs() <= 60]; M["ad"] = M.d.abs()
    M = M.sort_values("ad").drop_duplicates("mvt_id").drop_duplicates(["icao", "t_land"])
    out.append(M[["mvt_id", "apt", "mvt", "blk", "t_land", "t_stand"]]); print(day, "arrivals linked", len(M), flush=True)
L = pd.concat(out); L["yr"] = pd.to_datetime(L.mvt, unit="s", utc=True).dt.year; L.to_parquet((WORK_DIR.rstrip("/\\") + '/adsblol/arr_link.parquet'), index=False)
L = L[np.isfinite(L.t_stand)]; L["off"] = L.blk - L.t_stand; L["rec_taxi"] = L.blk - L.mvt; L["phys_taxi"] = L.t_stand - L.mvt
def mad(x): return float(np.median(np.abs(x - np.median(x))))
rows = []
for (ap, yr), g in L.groupby(["apt", "yr"]):
    if len(g) < 200: continue
    rows.append(dict(apt=ap, yr=yr, n=len(g), med_off=np.median(g.off), mad_off=mad(g.off), w30=(g.off.abs() <= 30).mean(), w60=(g.off.abs() <= 60).mean(),
                     w120=(g.off.abs() <= 120).mean(), med_rec_taxi=np.median(g.rec_taxi), med_phys_taxi=np.median(g.phys_taxi)))
print(pd.DataFrame(rows).round(3).to_string(index=False))
