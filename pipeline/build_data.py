from config import RANKING, DATA_DIR, WORK_DIR
import pandas as pd, numpy as np, glob, os, gc, pickle

D   = (DATA_DIR + "/")
OUT = (WORK_DIR + "/")
os.makedirs(OUT, exist_ok=True)
EPOCH0 = pd.Timestamp("2024-12-01", tz="UTC").value // 10**9

TIMES = {"t_mvt":"MVT_TIME_UTC_mvt","t_blk":"BLOCK_TIME_UTC_mvt","t_sch":"SCHED_TIME_UTC_mvt",
         "t_lobt":"LOBT_flt","t_iobt":"IOBT_flt","t_eobt":"EOBT_1_flt",
         "t_arvt1":"ARVT_1_flt","t_aobt":"AOBT_3_flt","t_arvt3":"ARVT_3_flt"}
SRC = ["ADES_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt","AIRCRAFT_TYPE_flt",
       "WK_TBL_CAT_flt","AIRCRAFT_OPERATOR_flt","MARKET_SEGMENT_flt","FLIGHT_TYPE_flt",
       "FLIGHT_RULE_flt","FLIGHT_RULE_mvt","ADES_FILED_flt","CALLSIGN_flt"]
CATS = ["APT"] + SRC
KEEP = list(dict.fromkeys(["MVT_ID_mvt","FLIGHT_ID_mvt","PHASE_mvt","ADEP_mvt","ADEP_flt","ADES_flt",
        "TAXITIME_SEC_mvt"] + SRC + list(TIMES.values())))
files = sorted(glob.glob(D+"training_*.parquet")) + [D+RANKING]

# ---- pass 1: global vocabularies ----
uniq = {c: set() for c in CATS}
vocab_files = sorted(glob.glob(D + "training_*.parquet")) + [os.path.join(DATA_DIR, "ranking.parquet")]
for f in vocab_files:
    df = pd.read_parquet(f, columns=list(dict.fromkeys(SRC+["PHASE_mvt","ADEP_mvt"])))
    df["AIRCRAFT_OPERATOR_flt"] = df["AIRCRAFT_OPERATOR_flt"].astype("string")
    apt = pd.Series(np.where(df.PHASE_mvt.eq("DEP"), df.ADEP_mvt, df.ADES_mvt))
    uniq["APT"].update(apt.dropna().unique().tolist())
    for c in SRC:
        uniq[c].update(pd.Series(df[c]).dropna().unique().tolist())
    del df; gc.collect()
vocab = {c: pd.Index(sorted(v)) for c, v in uniq.items()}
# Reconstruct the historical Jan/Jul vocabulary deterministically, then append
# final-only categories. A pre-existing work/vocab.pkl must not change a rebuild.
df = pd.read_parquet(os.path.join(DATA_DIR, RANKING), columns=list(dict.fromkeys(SRC + ["PHASE_mvt", "ADEP_mvt"])))
df["AIRCRAFT_OPERATOR_flt"] = df["AIRCRAFT_OPERATOR_flt"].astype("string")
final_values = {c: set(df[c].dropna().unique()) for c in SRC}
final_values["APT"] = set(pd.Series(np.where(df.PHASE_mvt.eq("DEP"), df.ADEP_mvt, df.ADES_mvt)).dropna().unique())
vocab = {c: pd.Index(list(v) + sorted(final_values[c] - set(v))) for c, v in vocab.items()}
del df; gc.collect()
print("vocab sizes", {c: len(v) for c, v in vocab.items()}, flush=True)

def tsec(s):
    s = pd.to_datetime(s, utc=True).dt.as_unit("ns")
    v = s.astype("int64").to_numpy().astype("float64") / 1e9 - EPOCH0
    v[s.isna().to_numpy()] = np.nan
    return v

def enc(series, c):
    return vocab[c].get_indexer(pd.Series(series).to_numpy(dtype=object)).astype(np.int32)

# ---- pass 2 ----
dep_parts, evt_parts = [], []
for f in files:
    rank = "ranking" in f
    df = pd.read_parquet(f, columns=KEEP)
    df["AIRCRAFT_OPERATOR_flt"] = df["AIRCRAFT_OPERATOR_flt"].astype("string")
    aptv = np.where(df.PHASE_mvt.eq("DEP"), df.ADEP_mvt, df.ADES_mvt)
    T = {k: tsec(pd.to_datetime(df[v], utc=True)) for k, v in TIMES.items()}
    isdep = df.PHASE_mvt.eq("DEP").to_numpy()
    apt = enc(aptv, "APT"); rwy = enc(df.RUNWAY_mvt, "RUNWAY_mvt")

    evt_parts.append(pd.DataFrame({
        "apt": apt.astype(np.int16), "isdep": isdep,
        "t_mvt": T["t_mvt"], "t_blk": T["t_blk"],
        "t_aobt": T["t_aobt"], "t_eobt": T["t_eobt"],
        "t_sch": T["t_sch"], "rwy": rwy.astype(np.int16)}))

    d = {"mvt_id": df.MVT_ID_mvt.to_numpy(), "is_rank": np.full(len(df), rank),
         "y": df.TAXITIME_SEC_mvt.to_numpy().astype(np.float32),
         "APT": apt.astype(np.int16), "RUNWAY_mvt": rwy.astype(np.int16)}
    for k, v in T.items(): d[k] = v
    for c in SRC:
        if c == "RUNWAY_mvt": continue
        d[c] = enc(df[c], c)
    d["adep_match"] = (df.ADEP_flt == df.ADEP_mvt).to_numpy()
    dep_parts.append(pd.DataFrame(d)[isdep].reset_index(drop=True))
    print(os.path.basename(f), "dep", int(isdep.sum()), flush=True)
    del df, T, d; gc.collect()

evt = pd.concat(evt_parts, ignore_index=True); del evt_parts; gc.collect()
evt.to_parquet(OUT+"events.parquet", index=False); print("events", evt.shape, flush=True)
del evt; gc.collect()
dep = pd.concat(dep_parts, ignore_index=True); del dep_parts; gc.collect()
dep.to_parquet(OUT+"dep.parquet", index=False)
print("dep", dep.shape, "rank", int(dep.is_rank.sum()), flush=True)
pickle.dump(vocab, open(OUT+"vocab.pkl","wb"))
