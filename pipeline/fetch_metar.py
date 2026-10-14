from config import DATA_DIR, WORK_DIR, RANKING
import urllib.request, os, time, sys
OUT=(WORK_DIR + "/metar/")
os.makedirs(OUT, exist_ok=True)
APTS=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LTFM","LSZH"]
FIELDS=["tmpf","dwpf","relh","drct","sknt","gust","p01i","alti","vsby","skyc1","skyl1",
        "wxcodes","ice_accretion_1hr","ice_accretion_3hr","peak_wind_gust","feel"]
BASE=("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={st}"
      + "".join(f"&data={f}" for f in FIELDS)
      + "&year1=2025&month1=1&day1=1&year2=2026&month2=8&day2=1"
      + "&tz=UTC&format=onlycomma&latlon=no&missing=empty&trace=0.0001"
      + "&report_type=3&report_type=4")
for st in APTS:
    dst=OUT+st+".csv"
    if os.path.exists(dst) and os.path.getsize(dst)>50000:
        print("skip",st,flush=True); continue
    for attempt in range(4):
        try:
            with urllib.request.urlopen(BASE.format(st=st), timeout=300) as r:
                data=r.read()
            open(dst,"wb").write(data)
            print(st, len(data)//1024, "KB", data.count(b"\n"), "rows", flush=True)
            break
        except Exception as e:
            print(st,"retry",attempt,e,flush=True); time.sleep(10)
print("done",flush=True)
