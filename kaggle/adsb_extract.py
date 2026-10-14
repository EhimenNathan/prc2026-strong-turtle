# PRC Data Challenge 2026 - adsb.lol near-airport ground-track extraction (run on Kaggle, Internet ON, CPU).
# Set the date range and output name below, run the script, download the zip from the kernel output.
# Used ranges: 2025-01-01..2025-06-30 (adsb_2025h1.zip), 2025-07-01..2025-12-31 (adsb_2025h2.zip), 2026-01-01..2026-01-31 + 2026-07-01..2026-07-31,
#              2026-02-01..2026-02-28 (adsb_feb26.zip), 2026-06-01..2026-06-30 (adsb_jun26.zip).
# Data: adsb.lol globe_history (ODbL 1.0, https://github.com/adsblol); OPDI flight lists (EUROCONTROL, open data).
START, END, ZIPNAME = "2026-02-01", "2026-02-28", "adsb_feb26.zip"
# adsb.lol ground-track extractor for the PRC Data Challenge 2026 (headless Kaggle "Save & Run All" version).
# Streams each day's globe_history tar (nothing stored), keeps aircraft that departed one of the 10 challenge airports within +-1 day
# (icao24 from the open OPDI flight list), and saves their on-ground / <1500 ft points within ~8 km of those airports.
# Data: adsb.lol globe_history (ODbL 1.0, https://github.com/adsblol); OPDI flight lists (EUROCONTROL, open data).
import os, gzip, json, time, tarfile, zipfile, subprocess, urllib.request, traceback
import numpy as np, pandas as pd

OUT = "/kaggle/working/adsb_out2" if os.path.isdir("/kaggle/working") else "adsb_out"
os.makedirs(OUT, exist_ok=True)
APT = {"EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543), "EHAM": (52.3086, 4.7639),
       "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626), "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389),
       "LSZH": (47.4647, 8.5492), "LTFM": (41.2753, 28.7519)}
NAMES = list(APT); ALAT = np.array([APT[a][0] for a in NAMES]); ALON = np.array([APT[a][1] for a in NAMES])
_pref, _fl = {}, {}
LOG = open(os.path.join(OUT, "log.txt"), "a")

def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def days_between(a, b):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(a, b)]

def preferred_urls(day):
    yr = day[:4]
    if yr not in _pref:
        u = f"https://raw.githubusercontent.com/adsblol/globe_history_{yr}/main/PREFERRED_RELEASES.txt"
        _pref[yr] = urllib.request.urlopen(u, timeout=120).read().decode().splitlines()
    tag = "v" + day.replace("-", ".") + "-"
    for line in _pref[yr]:
        if tag in line:
            urls = [x for x in line.strip().split(",") if x.startswith("http")]
            # A release may list a standalone tar AND its split representation.
            # They are alternative archives, not consecutive pieces of one tar.
            split = sorted(x for x in urls if '.tar.a' in x.rsplit('/', 1)[-1])
            return split or [x for x in urls if x.endswith('.tar')]
    raise RuntimeError("no preferred release for " + day)

def other_replica(urls):
    """same day, other replica (prod <-> staging); part count can differ, so list .aa-.af and let curl -f skip missing parts"""
    u = urls[0].rsplit(".tar", 1)[0]
    u = u.replace("-prod-", "-staging-") if "-prod-" in u else u.replace("-staging-", "-prod-")
    return [u + ".tar.a" + c for c in "abcdef"]

def flight_list(ym):
    if ym not in _fl:
        fn = f"flight_list_{ym}.parquet"
        try:
            if not os.path.exists(fn):
                urllib.request.urlretrieve(f"https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/{fn}", fn)
            L = pd.read_parquet(fn, columns=["icao24", "adep", "first_seen"])
            L = L[L.adep.isin(APT)].copy()
            L["d"] = pd.to_datetime(L.first_seen).dt.strftime("%Y-%m-%d")
            _fl[ym] = L[["icao24", "d"]]
        except Exception as e:
            say("flight list", ym, "unavailable:", e); _fl[ym] = pd.DataFrame(columns=["icao24", "d"])
    return _fl[ym]

def aircraft_for(day):
    """icao24 departing the 10 airports +-1 day (OPDI). None = no aircraft filter (list missing/near-empty): geography filter only"""
    dd = pd.Timestamp(day); keep = set()
    for k in (-1, 0, 1):
        d = dd + pd.Timedelta(days=k); L = flight_list(d.strftime("%Y%m"))
        keep |= set(L[L.d == d.strftime("%Y-%m-%d")].icao24.dropna().str.lower())
    return keep if len(keep) >= 500 else None

def trace_rows(icao, js, thin_s=5.0):
    """near-airport ground / <1500 ft samples of one trace. Keeps registration + type (trace level) and the callsign (forward-filled from
    the aircraft-details element). Thinning: keep a sample if >= thin_s since the last kept one of this trace, or if ground state or the
    moving/stationary state (gs 1 kt / 3 kt) changes, so stop/move boundaries stay exact while cruise-taxi samples are sparser."""
    ts0 = float(js.get("timestamp", 0)); tr = js.get("trace", [])
    if not tr:
        return None
    reg = js.get("r"); typ = js.get("t")
    rows = []; cs = None; last_t = -1e18; last_state = None; pend = None
    for p in tr:
        if len(p) > 8 and isinstance(p[8], dict):
            f = p[8].get("flight")
            if f: cs = f.strip()
        lat, lon, alt = p[1], p[2], p[3]
        if lat is None or lon is None:
            continue
        gnd = (alt == "ground")
        if not gnd and (not isinstance(alt, (int, float)) or alt > 1500):
            continue
        dl = np.abs(ALAT - lat) + np.abs(ALON - lon) * 0.7; k = int(dl.argmin())
        if dl[k] > 0.08:
            continue
        gs = p[4] if len(p) > 4 and p[4] is not None else np.nan
        t = ts0 + p[0]
        state = (gnd, 0 if not (gs >= 1) else (1 if gs < 3 else 2))
        row = (t, lat, lon, -1.0 if gnd else float(alt), gs, p[5] if len(p) > 5 and p[5] is not None else np.nan, NAMES[k], cs)
        if state == last_state and t - last_t < thin_s:
            pend = row; continue
        if state != last_state and pend is not None:
            rows.append(pend)          # last sample of the previous state: keeps stop/move boundaries exact
        pend = None; last_state = state; last_t = t
        rows.append(row)
    if not rows:
        return None
    D = pd.DataFrame(rows, columns=["t", "lat", "lon", "alt", "gs", "trk", "apt", "cs"])
    D.insert(0, "icao", icao); D["reg"] = reg; D["typ"] = typ
    for c in ["lat", "lon", "alt", "gs", "trk"]:
        D[c] = D[c].astype(np.float32)
    return D

def extract_day(urls, keep):
    """stream all tar parts of one day, in order, through curl -> tarfile (no disk use)"""
    cmd = " ; ".join(f"curl -fsS -L --retry 5 --connect-timeout 60 '{u}'" for u in urls)
    proc = subprocess.Popen(["bash", "-c", "( " + cmd + " )"], stdout=subprocess.PIPE, bufsize=1 << 20)
    parts, seen, parsed = [], 0, 0
    try:
        tf = tarfile.open(fileobj=proc.stdout, mode="r|*")
        for m in tf:
            if not m.isfile():
                continue
            seen += 1
            nm = m.name.rsplit("/", 1)[-1]
            if not nm.startswith("trace_full_"):
                continue
            icao = nm[len("trace_full_"):].split(".")[0].lower()
            if keep is not None and icao not in keep:
                continue
            raw = tf.extractfile(m).read(); parsed += 1
            try:
                js = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
            except Exception:
                continue
            r = trace_rows(icao, js)
            if r is not None:
                parts.append(r)
    finally:
        proc.kill(); proc.wait()
    if seen < 1000:
        raise RuntimeError("archive stream too short (%d files)" % seen)
    D = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["icao", "t", "lat", "lon", "alt", "gs", "trk", "apt", "cs", "reg", "typ"])
    return D, seen, parsed

def run_days(days, budget_h=11.3):
    t_start = time.time()
    for day in days:
        out = f"{OUT}/ground_{day}.parquet"
        if os.path.exists(out):
            say(day, "done already"); continue
        if time.time() - t_start > budget_h * 3600:
            say("time budget reached; stopping before", day); break
        t0 = time.time()
        try:
            keep = aircraft_for(day); urls = preferred_urls(day)
        except Exception as e:
            say(day, "SKIP (setup failed):", e); continue
        res = None
        for attempt, u in enumerate([urls, other_replica(urls), urls]):
            try:
                res = extract_day(u, keep); break
            except Exception as e:
                say(day, "attempt", attempt, "failed:", repr(e)[:200])
        if res is None:
            say(day, "FAILED all replicas"); continue
        D, seen, parsed = res
        D.to_parquet(out, index=False, compression="zstd")
        g = D[D.alt == -1]
        cov = {a: int(g[g.apt == a].icao.nunique()) for a in APT}
        say("%s | files %d | aircraft of interest %d | traces parsed %d | points %d | on-ground aircraft per airport %s | %.0fs"
            % (day, seen, (len(keep) if keep is not None else -1), parsed, len(D), cov, time.time() - t0))

def zip_outputs(name="/kaggle/working/adsb_out2.zip"):
    with zipfile.ZipFile(name, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(os.listdir(OUT)):
            z.write(os.path.join(OUT, f), f)
    say("wrote", name, "%.1f MB" % (os.path.getsize(name) / 1e6))


import os, time, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
r = subprocess.run(['curl', '-sS', '-o', '/dev/null', '-w', 'github %{http_code} %{time_total}s', 'https://github.com/'], capture_output=True, text=True)
print(r.stdout, r.stderr)
DAYS = days_between(START, END)
DAYS = sorted(DAYS, key=lambda d: (int(d[8:10]), d))   # spread over months first, so a partial run still covers every month
BUDGET_H = 11.3
WORKERS = os.cpu_count() or 4
for d in DAYS:
    preferred_urls(d)
for ym in sorted({(pd.Timestamp(d) + pd.Timedelta(days=k)).strftime("%Y%m") for d in DAYS for k in (-1, 0, 1)}):
    flight_list(ym)
T0 = time.time()

def do_day(day):
    out = f"{OUT}/ground_{day}.parquet"
    if os.path.exists(out):
        return f"{day} done already"
    if time.time() - T0 > BUDGET_H * 3600:
        return f"{day} skipped (time budget)"
    t0 = time.time()
    try:
        keep = aircraft_for(day); urls = preferred_urls(day)
        if keep is None: say(day, 'OPDI list near-empty -> geography filter only')
    except Exception as e:
        return f"{day} SKIP (setup failed): {e}"
    res = None
    for attempt, u in enumerate([urls, other_replica(urls), urls]):
        try:
            res = extract_day(u, keep); break
        except Exception as e:
            say(day, "attempt", attempt, "failed:", repr(e)[:200])
    if res is None:
        return f"{day} FAILED all replicas"
    D, seen, parsed = res
    D.to_parquet(out, index=False, compression="zstd", compression_level=9)
    g = D[D.alt == -1]
    cov = {a: int(g[g.apt == a].icao.nunique()) for a in APT}
    return ("%s | files %d | aircraft %d | traces %d | points %d | with callsign %.0f%% | on-ground aircraft %s | %.0fs"
            % (day, seen, (len(keep) if keep is not None else -1), parsed, len(D), 100 * D.cs.notna().mean() if len(D) else 0, cov, time.time() - t0))

say(f"days to process: {len(DAYS)} | workers: {WORKERS}")
with ProcessPoolExecutor(max_workers=WORKERS) as ex:
    futs = {ex.submit(do_day, d): d for d in DAYS}
    for i, f in enumerate(as_completed(futs), 1):
        try:
            msg = f.result()
        except Exception as e:
            msg = f"{futs[f]} worker crashed: {e!r}"
        say(f"[{i}/{len(DAYS)}] {msg}  (elapsed {(time.time() - T0) / 60:.0f} min)")
zip_outputs("/kaggle/working/" + ZIPNAME)
