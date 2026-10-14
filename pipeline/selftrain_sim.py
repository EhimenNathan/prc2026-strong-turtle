"""Transductive self-training, simulated inside 2025 (second half plays '2026').
Base model = the state LightGBM of oof_state.py (same features/params). For one airport:
  baseline : train on Jan-Jun labels                       -> score on Jul-Dec UNLINKED departures (true labels)
  treatment: + Jul-Dec LINKED departures with PSEUDO labels -> same score rows
  ceiling  : + Jul-Dec LINKED departures with TRUE labels   -> same score rows
Pseudo label = state-OOF + k * adsb correction (the physics layer's held-out prediction for those rows is not available for all months,
so the pseudo label uses the ADS-B physical estimate  y_ps = median-calibrated physical taxi where a stand dwell was detected, else the
state OOF + adsb-layer correction predicted by a model trained on Jan-Jun only)."""
from config import WORK_DIR
import os, sys, gc, pickle, numpy as np, pandas as pd, lightgbm as lgb, pyarrow.parquet as pq, warnings
from train_apt import feat_names, load_airport, add_te, TE_KEYS, F29_TE, CATCOLS, EXTRA_CATS, OUT
warnings.filterwarnings("ignore")
NM = sys.argv[1] if len(sys.argv) > 1 else "EDDF"
apts = list(pickle.load(open(OUT + "vocab.pkl", "rb"))["APT"])
def block(fname, ids):
    pf = pq.ParquetFile(OUT + fname); cols = [c for c in pf.schema_arrow.names if c != "id"]
    fid = pf.read(columns=["id"]).column(0).to_numpy().astype("int64"); keep = np.isin(fid, ids); _, first = np.unique(fid, return_index=True)
    keep &= np.isin(np.arange(len(fid)), first); B = np.empty((keep.sum(), len(cols)), np.float32)
    for j, c in enumerate(cols): B[:, j] = pf.read(columns=[c]).column(0).to_numpy(zero_copy_only=False)[keep].astype(np.float32)
    return pd.DataFrame(B, columns=cols, index=fid[keep]).reindex(ids).values.astype(np.float32), cols
names = feat_names(NM); X0, feats0, meta = load_airport(apts.index(NM), names)
ids = meta.mvt_id.values.astype("int64"); B1, c1 = block("state_feats_2025.parquet", ids); B2, c2 = block("state_feats2_2025.parquet", ids)
X0 = np.hstack([X0, B1, B2]); feats0 = feats0 + ["s1_" + c for c in c1] + ["s2_" + c for c in c2]; del B1, B2; gc.collect()
if os.environ.get("NOSEASON"):                  # drop calendar features: removes the unseen-month artifact of a H1 -> H2 simulation
    keep = [i for i, f in enumerate(feats0) if f not in ("month", "doy", "sin_doy", "cos_doy")]; X0 = X0[:, keep]; feats0 = [feats0[i] for i in keep]; print("calendar features dropped", flush=True)
lab = ~meta.is_rank.values; y = meta.y.values.astype(np.float64); mon = (meta.ym.values % 100)
H1 = lab & (mon <= 6); H2 = lab & (mon >= 7)
# linked rows + ADS-B physical estimate (x_off = take-off minus end of stand dwell), calibrated to labels on H1 by a single median offset
F = pd.concat([pd.read_parquet((WORK_DIR.rstrip("/\\") + '/adsblol/f2_%s.parquet') % t, columns=["mvt_id", "x_off", "x_lastblock"]) for t in ["y2025h1", "y2025h2", "y2025h1r", "y2025h2r"]]).drop_duplicates("mvt_id").set_index("mvt_id")
lk = F.x_lastblock.reindex(ids).notna().values; xo = F.x_off.reindex(ids).values; xo = np.where((xo >= 60) & (xo <= 7200), xo, np.nan)
q = np.load((WORK_DIR.rstrip("/\\") + '/oof_state_parts.npz')); O = pd.Series(q[NM + "_oof"], index=q[NM + "_ids"].astype("int64")).reindex(ids).values
off = np.nanmedian((y - xo)[H1 & np.isfinite(xo) & np.isfinite(y)])
PS = pd.read_parquet((WORK_DIR.rstrip("/\\") + '/pseudo_h2.parquet')).set_index("mvt_id").pseudo; ps = PS.reindex(ids).values; ps = np.where(np.isfinite(ps), ps, O)   # H1-trained adsb layer
print("%s: H1 rows %d | H2 rows %d | H2 linked %.2f | H2 linked with stand dwell %.2f | physical->label offset %.0f s" % (NM, H1.sum(), H2.sum(), lk[H2].mean(), np.isfinite(xo[H2]).mean(), off))
print("pseudo-label quality on H2 linked: rmse(ps - y) %.1f vs rmse(base - y) %.1f" % (np.sqrt(np.nanmean(((ps - y)[H2 & lk]) ** 2)), np.sqrt(np.nanmean(((O - y)[H2 & lk]) ** 2))))
PRM = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=40, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=15, max_cat_threshold=48, cat_smooth=15, max_bin=127, num_threads=6, verbose=-1)
keys = TE_KEYS + [k for k in F29_TE if all(c in feats0 for c in k)]
test = H2 & ~lk
import os
SHARES = [float(x) for x in os.environ.get("SHARES", "").split(",") if x]
PRED_LINKED = {}
def fit_eval(train_mask, target, label, w=None, prm=None, nr=800):
    prm = PRM if prm is None else prm
    X, feats = add_te(X0.copy(), list(feats0), meta.assign(y=target), train_mask, keys); cats = [feats.index(c) for c in CATCOLS + EXTRA_CATS if c in feats]
    m = lgb.train(prm, lgb.Dataset(X[train_mask], target[train_mask], weight=None if w is None else w[train_mask], categorical_feature=cats), nr); p = m.predict(X[test])
    e = y[test] - p; b = y[test] < 3600
    LKR = H2 & lk
    if os.environ.get("LINKED_EVAL"): PRED_LINKED[label] = m.predict(X[H2 & lk & np.isfinite(ps - O)])
    print("  %-10s unlinked H2 rows %d: RMSE %.1f | bulk(y<1h) %.1f" % (label, test.sum(), np.sqrt(np.mean(e ** 2)), np.sqrt(np.mean(e[b] ** 2))), flush=True)
    del X, m; gc.collect()
TAUS = [float(x) for x in os.environ.get("TAUS", "").split(",") if x]
SEAS = [float(x) for x in os.environ.get("SEAS", "").split(",") if x]
EXCL = [int(x) for x in os.environ.get("EXCLM", "").split(",") if x]
if EXCL:                                        # regime exclusion: held-out normal-configuration months, trained with / without the excluded months
    for m0 in [int(x) for x in os.environ.get("TESTM", "1,6").split(",")]:
        test = lab & (mon == m0); trm = lab & (mon != m0)
        fit_eval(trm, y, "m%d all months" % m0)
        fit_eval(trm & ~np.isin(mon, EXCL), y, "m%d excl %s" % (m0, EXCL))
        fit_eval(trm, y, "m%d works w=0.3" % m0, np.where(np.isin(mon, EXCL), 0.3, 1.0))
elif SEAS:                                        # seasonal-analogue weighting: target month m0 predicted from the other 11 months
    for m0 in (1, 7):
        test = lab & (mon == m0); trm = lab & (mon != m0)
        fit_eval(trm, y, "m%d uniform" % m0)
        cd = np.minimum(np.abs(mon - m0), 12 - np.abs(mon - m0))
        for tau in SEAS: fit_eval(trm, y, "m%d season tau=%g" % (m0, tau), np.exp(-cd / tau))
elif TAUS:                                        # time-decay weighting of H1 labels; score on ALL H2 rows
    test = H2.copy()
    fit_eval(H1, y, "uniform")
    for tau in TAUS:
        w = np.exp((mon - 6.0) / tau); fit_eval(H1, y, "decay tau=%g mo" % tau, w)
elif SHARES:
    for sh in SHARES:
        w = np.ones(len(y)); pm = H2 & lk; w[pm] = sh / (1 - sh) * H1.sum() / pm.sum()
        fit_eval(H1 | pm, np.where(pm, ps, y), "treat s=%.2f" % sh, w)
elif os.environ.get("REGS"):                       # regularisation ladder of the base (no pseudo labels), H1 -> H2 unlinked
    test = H2 & ~lk
    fit_eval(H1, y, "baseline")
    for lab_, p_, nr_ in [("min_leaf 300", dict(min_data_in_leaf=300), 800), ("min_leaf 1500", dict(min_data_in_leaf=1500), 800),
                          ("l2 300", dict(lambda_l2=300), 800), ("rounds 250", {}, 250), ("leaves 31 + min_leaf 300", dict(num_leaves=31, min_data_in_leaf=300), 800),
                          ("huber 1500", dict(objective="huber", alpha=1500), 800), ("clip y<=7200 target", None, 800)]:
        if p_ is None: fit_eval(H1, np.minimum(y, 7200), lab_)
        else: fit_eval(H1, y, lab_, prm=dict(PRM, **p_), nr=nr_)
elif os.environ.get("HONEST"):                     # leak-free pseudo labels: H1-trained base + H1-trained adsb correction (ps - O)
    X, feats = add_te(X0.copy(), list(feats0), meta, H1, keys); cats = [feats.index(c) for c in CATCOLS + EXTRA_CATS if c in feats]
    mb = lgb.train(PRM, lgb.Dataset(X[H1], y[H1], categorical_feature=cats), 800); pb = mb.predict(X); del X, mb; gc.collect()
    ps_h = pb + (ps - O); pm = H2 & lk & np.isfinite(ps_h)
    if os.environ.get("PSALL"):                      # transductive: pseudo-label ALL H2 rows (incl. the scored ones) with the H1 base itself
        pa = H2 & np.isfinite(pb); fit_eval(H1, y, "baseline"); fit_eval(H1 | pa, np.where(pa, pb, y), "self-distil ALL H2")
        if os.environ.get("LINKED_EVAL"):           # final prediction on LINKED H2 rows = base + H1-trained adsb correction (ps - O)
            LKR = H2 & lk & np.isfinite(ps - O); corr = (ps - O)[LKR]; yl = y[LKR]
            def rm(e): return float(np.sqrt(np.mean(e ** 2)))
            bo = PRED_LINKED["baseline"] + corr; bn = PRED_LINKED["self-distil ALL H2"] + corr
            print("  LINKED H2 rows %d (base + adsb corr): baseline %.1f | self-distil %.1f (%+.2f%%) | base alone %.1f -> %.1f" % (LKR.sum(), rm(yl - bo), rm(yl - bn), 100 * (rm(yl - bn) / rm(yl - bo) - 1), rm(yl - PRED_LINKED["baseline"]), rm(yl - PRED_LINKED["self-distil ALL H2"])))
        else:
            w = np.ones(len(y)); w[pa] = 0.3 / 0.7 * H1.sum() / pa.sum(); fit_eval(H1 | pa, np.where(pa, pb, y), "self-distil ALL H2 w30", w)
        sys.exit()
    print("honest pseudo quality on H2 linked: rmse(ps_h - y) %.1f vs rmse(H1 base - y) %.1f" % (np.sqrt(np.nanmean(((ps_h - y)[pm]) ** 2)), np.sqrt(np.nanmean(((pb - y)[pm]) ** 2))))
    fit_eval(H1, y, "baseline")
    fit_eval(H1 | pm, np.where(pm, pb, y), "base-only pseudo")
    for sh in [float(x) for x in os.environ.get("PSH", "").split(",") if x]:
        w = np.ones(len(y)); w[pm] = sh / (1 - sh) * H1.sum() / pm.sum(); fit_eval(H1 | pm, np.where(pm, pb, y), "base-only share %.2f" % sh, w)
    if not os.environ.get("PSH"): fit_eval(H1 | pm, np.where(pm, ps_h, y), "honest treatment")
else:
    fit_eval(H1, y, "baseline")
    fit_eval(H1 | (H2 & lk), np.where(H2 & lk, ps, y), "treatment")
    fit_eval(H1 | (H2 & lk), y, "ceiling")

if os.environ.get("LINKED_EVAL") and "baseline" in PRED_LINKED and "treatment" in PRED_LINKED:
    LKR = H2 & lk; yl = y[LKR]; old_final = ps[LKR]                 # old base + H1-trained adsb correction (the pseudo label itself)
    new_final = old_final + (PRED_LINKED["treatment"] - PRED_LINKED["baseline"])
    def rm(e): return float(np.sqrt(np.mean(e ** 2)))
    print("  LINKED H2 rows %d: old base+adsb %.1f | + self-trained base change %.1f (%+.2f%%)" % (LKR.sum(), rm(yl - old_final), rm(yl - new_final), 100 * (rm(yl - new_final) / rm(yl - old_final) - 1)))
    bb = yl < 3600; print("     bulk y<1h: %.1f -> %.1f" % (rm((yl - old_final)[bb]), rm((yl - new_final)[bb])))
