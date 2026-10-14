"""Assemble the final prediction for every departure in the ranking file from the pipeline components (no leaderboard-fitted weights).

  base        mean of the production state-LightGBM seeds (lgbstate2_rank_parts.npz + lgbstate2_rank_s*.npz)
  + adsb      per-airport-scaled ADS-B residual layer on linked departures (adsblol/mx5/corr_slat_sb.parquet; scales from 2025 held-out months)
  + selftrain W_ST * (self-trained base - base) on UNLINKED departures at the airports where the 2025 simulation improved (ADS-B pseudo-labels)
  + distil    optional research comparison only; disabled after nested February/June validation
  + window    window-state correction for untracked departures (corr_window_state.parquet; scales from 2025 held-out months)
  + clock     clock-residual correction for unlinked departures (corr_clock_unlinked.parquet; 2025-validated airports)
  + lobt      LOBT-tail correction (corr_lobt_tail_final.parquet; 2025 held-out bin means)
  then the exact LOBT window: BLOCK is within +-3606 s of LOBT for 100% of 2025 departures -> clip to [D-3606, D+3606], D = MVT-LOBT; and >= 0.
Writes submission/<team>_final.parquet in the template order, plus a Jan/Jul subset for an honest check against the regular leaderboard.
Env: W_ST (default 0.5), W_D (default 0, research comparison), TEAM (strong-turtle)."""
import os, glob, numpy as np, pandas as pd, pyarrow.parquet as pq, pickle, json
from pathlib import Path
from assembly_checks import require, read_parts, read_correction, sha256
from config import WORK_DIR, DATA_DIR, SUBMIT_TEMPLATE
OUT = WORK_DIR.rstrip("/\\") + "/"; TEAM = os.environ.get("TEAM", "strong-turtle")
W_ST = float(os.environ.get("W_ST", "0.5")); W_D = float(os.environ.get("W_D", "0.0"))
require(np.isfinite(W_ST) and np.isfinite(W_D), "Invalid transductive weights")
apts = list(pickle.load(open(OUT + "vocab.pkl", "rb"))["APT"]); E0 = pd.Timestamp("2024-12-01", tz="UTC").value // 10**9
d = pq.read_table(OUT + "dep.parquet", columns=["mvt_id", "APT", "t_mvt", "t_lobt", "is_rank"]).to_pandas()
require(d.mvt_id.is_unique, "Duplicate departure IDs")
R = d[d.is_rank].copy(); R["apt"] = np.array(apts)[R.APT.values]; R = R.set_index("mvt_id")
require(len(R) == 670790, "Final ranking must contain 670790 departures")
def parts(fn, key="_rank_p"):
    expected = ["LFPG", "EDDF", "EGLL", "LEMD", "EHAM"] if "selftrain" in fn else (["LTFM", "LIRF"] if "distil" in fn else apts)
    return read_parts(fn, expected, R)
# ---- base: seed mean ----
seed_files = [OUT + f for f in ["lgbstate2_rank_parts.npz", "lgbstate2_rank_s1.npz", "lgbstate2_rank_s2.npz"]]
S = pd.concat([parts(f).rename(i) for i, f in enumerate(seed_files)], axis=1).reindex(R.index)
R["base0"] = S[0]; R["base"] = S.mean(axis=1); print("base: %d seeds | rows with a base prediction %.4f" % (S.shape[1], R.base.notna().mean()))
# ---- linked (ADS-B) ----
LK = set(pd.concat([pd.read_parquet(OUT + "adsblol/f2_%s.parquet" % t, columns=["mvt_id"]) for t in ["y2026", "y2026r", "y2026fj", "y2026fjr"]
                    ]).mvt_id)
R["linked"] = R.index.isin(list(LK))
def corr(fn):
    return read_correction(OUT + fn, R)
R["adsb"] = corr("adsblol/mx5/corr_slat_sb.parquet")
# ---- self-training on unlinked rows ----
R["st"] = 0.0
if True:
    st = parts(OUT + "lgbstate2_selftrain_final.npz").reindex(R.index)
    m = st.notna() & ~R.linked; R.loc[m, "st"] = W_ST * (st[m] - R.base0[m]); print("self-training rows %d" % m.sum())
# ---- self-distillation (LTFM all rows, LIRF unlinked), signal-shrunk ----
R["dist"] = 0.0
dfiles = [OUT + "lgbstate2_distil.npz", OUT + "lgbstate2_distil_s1.npz"] if W_D else []
if dfiles:
    Dm = pd.concat([parts(f).rename(i) for i, f in enumerate(dfiles)], axis=1).reindex(R.index)
    for a, sel in [("LTFM", R.apt == "LTFM"), ("LIRF", (R.apt == "LIRF") & ~R.linked)]:
        s = sel & Dm.notna().all(axis=1)
        if not s.any(): continue
        delta = Dm[s].mean(axis=1) - R.base[s]
        noise = (np.mean((S[s].iloc[:, 0] - S[s].iloc[:, 1]) ** 2) / 2 if S.shape[1] > 1 else 0.0)
        noise = noise / len(dfiles) + noise / S.shape[1]
        signal = float(np.mean(delta ** 2))
        shrink = max(0.0, 1 - noise / signal) if signal > 0 else 0.0
        R.loc[s, "dist"] = W_D * shrink * delta; print("distillation %s rows %d | delta rms %.1f | shrink %.2f" % (a, s.sum(), np.sqrt(np.mean(delta ** 2)), shrink))
R["window"] = corr("corr_window_state.parquet"); R["clock"] = corr("corr_clock_unlinked.parquet"); R["lobt"] = corr("corr_lobt_tail_final.parquet")
R["pred"] = R.base + R.adsb + R.st + R.dist + R.window + R.clock + R.lobt
D = R.t_mvt - R.t_lobt; lo = D - 3606; hi = D + 3606; has = D.notna()
before = R.pred.copy(); R.loc[has, "pred"] = np.clip(R.pred[has], lo[has], hi[has]); R["pred"] = R.pred.clip(lower=0)
print("LOBT window moved %d predictions" % (np.abs(R.pred - before) > 1e-9).sum())
for c in ["adsb", "st", "dist", "window", "clock", "lobt"]: print("  %-7s rows changed %7d | rms %.1f s" % (c, (R[c] != 0).sum(), np.sqrt((R[c] ** 2).mean())))
require(np.isfinite(R.pred).all(), "Missing or infinite predictions")
require((hi[has] >= 0).all(), "LOBT window incompatible with nonnegative taxi time")
require(((R.pred[has] >= lo[has] - 1e-8) & (R.pred[has] <= hi[has] + 1e-8)).all(), "LOBT window violation")
T = pq.read_table(os.path.join(DATA_DIR, SUBMIT_TEMPLATE)).to_pandas(); T["TAXITIME_SEC_mvt"] = R.pred.reindex(T.MVT_ID_mvt.astype("int64").values).values.astype("float64")
require(T.MVT_ID_mvt.is_unique and set(T.MVT_ID_mvt) == set(R.index), "Template ID mismatch")
require(np.isfinite(T.TAXITIME_SEC_mvt).all() and len(T) == 670790, "Incomplete final template")
os.makedirs("submission", exist_ok=True); T.to_parquet("submission/%s_final.parquet" % TEAM, index=False); print("wrote submission/%s_final.parquet" % TEAM, T.shape)
mo = pd.to_datetime(R.t_mvt + E0, unit="s", utc=True).dt.month; print("rows by month:", mo.value_counts().sort_index().to_dict())
old = pq.read_table(os.path.join(DATA_DIR, "submitting.parquet")).to_pandas(); old["TAXITIME_SEC_mvt"] = R.pred.reindex(old.MVT_ID_mvt.astype("int64").values).values.astype("float64")
old.to_parquet("submission/%s_final_janjul_check.parquet" % TEAM, index=False); print("Jan/Jul subset for the regular leaderboard check:", old.shape, old.TAXITIME_SEC_mvt.isna().sum(), "missing")
require(np.isfinite(old.TAXITIME_SEC_mvt).all(), "Incomplete regular subset")
R["month"] = mo
R.reset_index().to_parquet("submission/%s_final_components.parquet" % TEAM, index=False)
component_files = seed_files + dfiles + [OUT + "lgbstate2_selftrain_final.npz"] + [OUT + f for f in ["adsblol/mx5/corr_slat_sb.parquet", "corr_window_state.parquet", "corr_clock_unlinked.parquet", "corr_lobt_tail_final.parquet"]]
manifest = {
    "rows": len(T), "seeds": len(seed_files), "weights": {"selftrain": W_ST, "distillation": W_D},
    "months": {str(k): int(v) for k, v in mo.value_counts().sort_index().items()},
    "component_sha256": {Path(f).name: sha256(f) for f in component_files},
    "template_sha256": sha256(Path(DATA_DIR) / SUBMIT_TEMPLATE),
    "submission_sha256": sha256("submission/%s_final.parquet" % TEAM),
    "checks": {"exact_template_ids": True, "unique_ids": True, "finite": True, "nonnegative": True, "lobt_window": True},
    "uploaded": False,
}
Path("submission/%s_final_manifest.json" % TEAM).write_text(json.dumps(manifest, indent=2))
