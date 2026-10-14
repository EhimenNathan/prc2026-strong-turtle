"""f29: features from FLIGHT_mvt, the never-null flight designator no model has ever seen.

FLIGHT_mvt sits on the MOVEMENT record: 14,100 distinct values, 0.0001 null in both periods. It
is absent from dep.parquet and from CATCOLS, so no model has used it.

Two independent reasons it matters:

1. OPERATOR RECOVERY. Rows with no NM flight record (1.56% of rows, 59.5% of squared error) have
   AIRCRAFT_OPERATOR_flt null on 100% of rows. The airline prefix of FLIGHT_mvt is present on
   99.1% of them and identifies the operator with 99.54% purity, explaining R2 = 0.158 of y
   within that population. This restores a categorical exactly where the model is worst.

2. SERVICE-LEVEL SIGNAL. An out-of-fold target encoding of FLIGHT_mvt gives
   PARTIAL(delta | d_aobt) = +0.4201, against +0.2875 for the operator encoding already in the
   model, and the two correlate only 0.74. For scale, f13's best column was ~0.43 and the whole
   physics stack sits at 0.02-0.18.

Nothing here uses the target: the codes are identifiers, and the encoding is left to the
pipeline's own out-of-fold add_te().
"""
from config import RANKING, DATA_DIR, WORK_DIR
import os, numpy as np, pandas as pd, glob, gc, pyarrow.parquet as pq

D = DATA_DIR
OUT = (WORK_DIR + "/")
COLS = ["MVT_ID_mvt", "PHASE_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt"]
parts = []
for f in sorted(glob.glob(os.path.join(D, "training_2025-*.parquet"))) + [os.path.join(D, RANKING)]:
    t = pq.read_table(f, columns=COLS).to_pandas()
    parts.append(t[t.PHASE_mvt.values == "DEP"])
    del t; gc.collect()
A = pd.concat(parts, ignore_index=True); del parts; gc.collect()
A["mvt_id"] = A.MVT_ID_mvt.astype("int64")
A = A.drop_duplicates("mvt_id").set_index("mvt_id")
fm = A.FLIGHT_mvt.astype("string")

pref = fm.str.extract(r"^([A-Za-z]+)", expand=False)
num = pd.to_numeric(fm.str.extract(r"(\d+)", expand=False), errors="coerce")
suf = fm.str.extract(r"([A-Za-z])$", expand=False)

# integer codes: stable across train and rank because they are fitted on the union
def code(s):
    c = s.astype("category")
    return c.cat.codes.astype(np.int32).where(s.notna(), -1).astype(np.int32)

out = pd.DataFrame(index=A.index)
out["flt_code"] = code(fm)
out["flt_prefix"] = code(pref)
out["flt_suffix"] = code(suf)
out["flt_number"] = num.astype(np.float32)
# service frequency: how often this exact designator flies, and from this airport
out["flt_freq"] = fm.map(fm.value_counts()).astype(np.float32)
key = pref.fillna("") + "|" + A.ADEP_mvt.astype(str)
out["flt_prefix_apt_freq"] = key.map(key.value_counts()).astype(np.float32)
# route identity: designator + destination, a finer service key
rt = fm.fillna("") + ">" + A.ADES_mvt.astype(str)
out["flt_route"] = code(rt)

dep = pq.read_table(OUT + "dep.parquet", columns=["mvt_id", "is_rank", "y"]).to_pandas()
ids = dep.mvt_id.values.astype("int64")
res = out.reindex(ids).reset_index(drop=True)
for c in ["flt_code", "flt_prefix", "flt_suffix", "flt_route"]:
    res[c] = res[c].fillna(-1).astype(np.int32)
res.to_parquet(OUT + "f29.parquet", index=False)
print("f29 saved:", list(res.columns))
print("   rows %d | flt_code known %.4f | prefix known %.4f"
      % (len(res), (res.flt_code.values >= 0).mean(), (res.flt_prefix.values >= 0).mean()))

y = dep.y.values.astype(float)
tr = (~dep.is_rank.values) & np.isfinite(y)
rk = dep.is_rank.values.astype(bool)
print("\ndrift check (train vs rank):")
for c in res.columns:
    v = res[c].values.astype(float)
    v = np.where(v < 0, np.nan, v) if c.endswith(("code", "prefix", "suffix", "route")) else v
    print("   %-22s train mean %12.1f  rank mean %12.1f  nan tr %.3f rk %.3f"
          % (c, np.nanmean(v[tr]), np.nanmean(v[rk]), np.isnan(v[tr]).mean(), np.isnan(v[rk]).mean()))

print("\ncardinality: flt_code %d | flt_prefix %d | flt_route %d"
      % (res.flt_code.nunique(), res.flt_prefix.nunique(), res.flt_route.nunique()))
print("ranking rows whose flt_code appears in training: %.4f"
      % pd.Series(res.flt_code.values[rk]).isin(set(res.flt_code.values[tr])).mean())
