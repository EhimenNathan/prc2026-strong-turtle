# PRC Data Challenge 2026: taxi-out time prediction (team `strong-turtle`)

This repository holds our solution to the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/) (EUROCONTROL Performance Review Commission and OpenSky Network). The task is to predict the **taxi-out time** (`MVT_TIME − BLOCK_TIME`) of departures at 10 major European airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM) for Jan/Feb/Jun/Jul 2026. Models are trained on 2025 only.

- **Model:** per-airport gradient-boosted trees on rich operational features, plus a residual layer built from open **adsb.lol** ADS-B ground tracks, plus corrections validated on 2025, plus label-free transductive steps on the 2026 inputs.
- **Final combination:** scales use 2025 component validation or label-free quantities; step sizes are fixed. Regular-phase leaderboard-fitted blends are excluded. Development history and validation limits are disclosed in [`docs/VALIDATION_LIMITS.md`](docs/VALIDATION_LIMITS.md).
- **Details:** [`docs/METHOD.md`](docs/METHOD.md) describes the method and its validation, and [`docs/DATA.md`](docs/DATA.md) covers data sources and licences.

## Results

| Set | RMSE (s) |
|---|---|
| Regular leaderboard, Jan+Jul 2026, final pipeline | Not submitted for scoring |
| Final phase, Jan/Feb/Jun/Jul 2026 | Unknown; organiser scoring pending |

The locally assembled candidate contains 670,790 rows. Its manifest records exact template-ID, finite-value, nonnegative-value and LOBT-window checks plus component SHA-256 hashes. A 2026 RMSE is unavailable without organiser scoring. Existing twelve-month base cross-fit diagnostics are in `docs/base_oof_2025_by_month.csv`. The nested February/June surrogate test is run with `python pipeline/validate_feb_jun.py`; its results are kept separate from the final-pipeline score.

The completed nested 2025 review excluded both February and June departure labels from base and ADS-B training. Its base + ADS-B + self-training surrogate improved RMSE from **306.832 to 296.242 seconds in February** and from **330.149 to 316.660 in June**. All five self-training airports improved in both months. Base distillation regressed at LIRF in both months and is excluded from the final recipe. See [`docs/NESTED_VALIDATION.md`](docs/NESTED_VALIDATION.md) for scope, bootstrap intervals and the deployment decision. This diagnostic informed selection; it is not an untouched final test or a score of the complete production pipeline.

## Repository layout

```
run_all.sh                 full reproduction, in order
requirements.txt           pinned Python packages (CPU only)
pipeline/config.py         paths: PRC_DATA (default ./data), PRC_WORK (default ./work), PRC_RANKING
pipeline/*.py              production steps plus separate validation helpers (see run_all.sh)
pipeline/base_rounds.json  per-airport boosting rounds (early stopping in train_apt.py validation)
pipeline/selftrain_sim.py  2025 simulation used to validate self-training / self-distillation
pipeline/state_core.py     standalone observable state-feature functions
pipeline/validate_feb_jun.py nested February/June 2025 holdout diagnostic
pipeline/validate_transduction.py separate nested pseudo-label diagnostic
check_source.py            pre-publication syntax and source-content checks
kaggle/adsb_extract.py     adsb.lol extraction (run on Kaggle, Internet on)
docs/METHOD.md             method, validation evidence, negative results
docs/DATA.md               data sources and licences
```

## Reproducing the final submission

1. **Environment:** Python 3.13, then `pip install -r requirements.txt`. Use a CPU machine with at least 16 GB of RAM (peak about 6 GB per step) and about 15 GB of free disk.
2. **Organiser data:** put `training_2025-*.parquet`, `final_ranking.parquet`, `final_submitting.parquet`, `ranking.parquet` and `submitting.parquet` in `data/`. They come from the organisers' bucket `prc-2026-datasets` and are not redistributed here.
3. **ADS-B extracts:** on Kaggle (CPU, Internet ON), run `kaggle/adsb_extract.py` once per date range. For each run, set `START`, `END` and `ZIPNAME` at the top of the script. Put the zips in `data/adsb/`.

   | Range | Zip |
   |---|---|
   | 2025-01-01 → 2025-06-30 | `adsb_2025h1.zip` |
   | 2025-07-01 → 2025-12-31 | `adsb_2025h2.zip` |
   | 2026-01-01 → 2026-01-31 and 2026-07-01 → 2026-07-31 (two runs, one zip) | `adsb_2026janjul.zip` |
   | 2026-02-01 → 2026-02-28 | `adsb_feb26.zip` |
   | 2026-06-01 → 2026-06-30 | `adsb_jun26.zip` |

   Each month takes about 40 minutes on a 4-CPU Kaggle session and produces about 100–130 MB.
   Keep the archive manifests and logs. The candidate used an additional seven-day January repair extract and a 2025 H2 extract without December 31; see `docs/REPRODUCTION.md`. An optional `adsb_2026janfix.zip` is merged by the feature builders using the largest copy of each day.
4. **Run everything:** `bash run_all.sh`. It downloads METAR, builds all features, trains every model and writes `submission/strong-turtle_final.parquet`. It takes about 10–12 hours on a 6-core CPU; the base model is about 1.5 h per seed and the ADS-B layer about 1 h.
   Use a fresh `work/` directory for a new reproduction. Production seed and self-training checkpoints contain dependency fingerprints and reject unverified or stale outputs. `PRC_DATA` and `PRC_WORK` override local directories. The existing candidate was assembled from completed research checkpoints; rebuilding the entire dataset and training sequence on a fresh machine has not been timed or verified end to end.

## Main ideas

- **ADS-B ground tracks.** We use the aircraft's own track from adsb.lol: stand dwell, the first push, the last stop and stand distances. This is direct physical evidence of the off-block event and was the single largest gain.
- **LOBT support window.** In the supplied 2025 data BLOCK is always within ±3,606 s of LOBT, so predictions are projected into that empirically supported window. This observation alone does not prove the support of arbitrary future datasets.
- **Shared surface state.** Departures in the same 30-minute window share unobserved congestion. Tracked flights' ADS-B corrections are passed to untracked flights in the same window.
- **Transductive learning without 2026 labels.** We self-train on ADS-B pseudo-labels and apply the change to untracked flights at five airports. The nested February/June surrogate supports this step. Base distillation was tested and excluded after that review.
- **Generalisation discipline.**
  - Historical Jan/Jul 2025 component validation, plus nested February/June checks. Historical ADS-B gates use available qualifying months; some airports lack sufficient coverage in one month.
  - A drift audit of every ADS-B feature between 2025 and 2026.
  - Rejection of components that only fit 2025 bookkeeping, such as LIRF schedule-echo models.

## Licence

Code: **GNU GPL v3** (see `LICENSE`).
Data used: adsb.lol (ODbL 1.0), EUROCONTROL OPDI flight lists (organiser-confirmed open data; licence-identifier limit disclosed), IEM ASOS METAR (public domain). Organiser data is subject to the challenge terms and is not included.

## Verification

```bash
python check_source.py
python tests/test_assembly_checks.py
python tests/test_transduction_isolation.py
```

A fresh raw-data fixture passed all 20 core/state feature stages, with aligned IDs across 6,402 departures. Changing 4,802 historical departure targets and block times left all 20 saved feature/arrival frames exactly unchanged. Aggregate evidence is in `docs/feature_smoke_rebuild.json` and `docs/feature_label_invariance.json`. To rerun these optional fixture checks, provide organiser data through `PRC_DATA` and cached METAR through `PRC_SOURCE_WORK`, then run `python tests/smoke_rebuild.py` followed by `python tests/test_feature_label_invariance.py`. They write disposable data under ignored `.local_checks/` and do not replace production inputs.
