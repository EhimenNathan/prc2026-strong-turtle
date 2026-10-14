# Method: team strong-turtle, PRC Data Challenge 2026

**Task:** predict departure taxi-out time, `MVT_TIME − BLOCK_TIME`, at 10 European airports for Jan/Feb/Jun/Jul 2026.
**Training data:** 2025 labels only.
**Metric:** RMSE.

The component gains below are historical tuning evidence. Read [VALIDATION_LIMITS.md](VALIDATION_LIMITS.md) for fold overlap, scale-selection optimism and the scope of the additional nested February/June check. They are not claimed as an independent score for the complete final recipe.

## 1. Design principles

1. **Validate on January/July 2025.** Deployment gates generally require gains in available qualifying months. ADS-B coverage is insufficient in one month at some airports, so the implementation does not enforce two qualifying months everywhere. Stored base and residual folds are not fully nested.
2. **Prefer physics and exact identities over bookkeeping.** Features describing what the aircraft physically did (ADS-B ground tracks) and exact identities in the data transferred to 2026. Patterns in how clocks were *recorded* in 2025 did not transfer (see §6).
3. **Exclude direct leaderboard-fitted final blends.** Final weights, scales and step sizes are one of:
   - estimated on held-out 2025 months;
   - derived from label-free quantities, for example seed-noise variance;
   - a fixed constant documented here.
4. **Use the unlabelled 2026 inputs transductively, never 2026 labels.** Self-training and self-distillation use only the published 2026 features.

## 2. Pipeline (`run_all.sh`)

### 2.1 Core tables and features
`build_data.py`, `build_arr.py`, f1–f13, f23, f29 and the state features. They cover:
- **clock offsets**: take-off minus SCHED, LOBT, IOBT, EOBT, AOBT and ARVT;
- **congestion counts**: take-offs, landings, pushbacks and taxiing aircraft in 15/30/60-minute windows;
- **runway configuration**;
- **METAR weather**;
- **stand/pier structure and queueing features**: unimpeded times, service rates, expected wait;
- **a Kalman-filtered airport delay state**;
- **flight-identifier features**;
- **target encodings**, computed inside the training fold only;
- **row-level airport and runway congestion state**, built from the movement table itself (state features).

### 2.2 Base model
- Per-airport LightGBM, L2 loss, trained on all of 2025.
- Rounds come from early stopping in `train_apt.py` validation (`pipeline/base_rounds.json`) × 1.2.
- **Three seeds are averaged** (the production seed plus seeds 1 and 2) to remove training noise. Two seeds disagree by 20–35 s RMS per flight, and 60–85 s at LIRF.

### 2.3 ADS-B residual layer
- **Data:** adsb.lol ground tracks are linked to departures by callsign plus take-off time, or by ground-roll take-off at high-elevation LEMD.
- **Features (176 in total):**
  - session features: stand dwell and stand departure, first push, last stop, last block, recorded-clock gaps, stand distances;
  - movement profiles: 2-hour window, 1-minute final bins, speed spread, turning;
  - per-stand recording latency learned from arrivals.
- **Model:** a LightGBM (Huber) on the residual of the 2025 out-of-fold base, applied as `k_a × prediction`.
- **Scale `k_a`:** the per-airport least-squares scale on Jan/Jul 2025 component validation. Applied where available qualifying months improve. Scale fitting and gain reporting reuse the same component-validation rows.
- **Held-out result on 2025:**

  | Airport | Change in RMSE |
  |---|---|
  | LSZH | −24% |
  | EDDM | −42% (ground-roll rows) |
  | EHAM | about −26% |
  | Total | 3.1e9 SSE saved |

### 2.4 Corrections validated on 2025

**Exact LOBT window (identity).**
- For 100% of the 2,062,577 departures of 2025 that have a LOBT, BLOCK lies within ±3,606 s of LOBT.
- Predictions are therefore projected into `[D − 3606, D + 3606]`, where `D = MVT − LOBT`.
- This is an empirically verified support rule in the supplied 2025 data; arbitrary future datasets are not guaranteed to obey it.

**LOBT tail.**
- Trees under-predict flights that take off long after their flight-plan off-block time.
- Bin offsets for D in 1.5–2 h and 2–3 h are the mean held-out residual of the full pipeline, with the sign required to agree in Jan and Jul 2025. The results are +244 s and +1,384 s, excluding LIRF.
- Bins with too little evidence get 0.

**Clock residual.**
- A residual model on the recorded clocks, applied to departures without ADS-B tracks.
- Used only at airports where both held-out months improve: EDDM, LEMD, LFPG, LSZH and LTFM.

**Window state.**
- Departures in the same airport and 30-minute window share an unobserved surface state. Oracle: the leave-one-out mean of other flights' residuals explains 4.8% of SSE.
- For untracked departures we use the mean ADS-B correction of the *other tracked* departures in the window.
- The per-airport scale is estimated on 2025 and applied where both months improve. Held-out gains (Jan/Jul):

  | Airport | Gain |
  |---|---|
  | EDDF | 5.5% / 11.3% |
  | LEBL | 3.3% / 6.1% |
  | LSZH, EHAM, EGLL | 0.9–4% |

### 2.5 Transductive steps (2026 inputs, no 2026 labels)

**Self-training.**
- The base is retrained with 2026 tracked departures pseudo-labelled by base + ADS-B layer.
- The resulting base change is applied only to **untracked** departures; applying it to tracked ones double-counts the ADS-B evidence.
- Earlier 2025 simulation (first half trains, second half plays "2026"): EDDF 191.8 → 184.7 on untracked rows. This is exploratory evidence: the ADS-B residual fit uses first-half labels, but its upstream quarterly base predictions can depend on second-half labels. It is not a fully nested estimate of self-training gain.
- Pseudo-label shares are the simulation's: 30% at LFPG, EDDF, EGLL and LEMD; the natural share at EHAM.

**Self-distillation (tested and excluded from deployment).**
- The base is retrained with every 2026 row labelled by its own prediction, at LTFM and LIRF.
- The base self-distillation simulation uses first-half-only base pseudo-labels and removes calendar features to reduce the unseen-month artifact:
  - LIRF untracked: 858 → 800, −6.7%;
  - LTFM: −1.1%.
- The change is shrunk by its label-free signal fraction, `1 − seed-noise variance / change variance`.

The exploratory candidate applied both changes at half weight. The final reviewed recipe retains self-training at **half weight** (`W_ST = 0.5`) and disables base distillation (`W_D = 0`). In the nested February/June check, all five self-training airports improved in both months, while LIRF base distillation worsened both months. See `NESTED_VALIDATION.md`. This additional holdout has now informed deployment selection and is not an untouched final test.

## 3. Validation evidence

| Component | Evidence (2025 held-out unless stated) |
|---|---|
| State features in the base | LTFM 252.97 → 244.87 (largest base gain) |
| ADS-B layer | 3.1e9 SSE saved; per airport up to −42% |
| LOBT window | exact identity, 100% of 2.06M departures |
| LOBT tail | both months same sign; +244 / +1,384 s |
| Clock residual | both months improve at 5 airports |
| Window state | both months improve at 5 airports |
| Self-training | simulated, untracked rows, e.g. EDDF −3.7% |
| Self-distillation | Earlier simulation suggested gains; nested February/June LIRF results regressed in both months. Excluded from final deployment. |
| Seed averaging | label-free variance reduction; the (1 − 1/K) expression assumes independent seed noise |

## 4. Generalisation safeguards

- **Feature drift between years.** It was audited with same-calendar-month AUC between 2025 and 2026 inputs. The adsb.lol receiver network changed in 2026: EGLL July is much denser, EDDF sparser, and LIRF July uncovered. Dropping coverage-describing features hurt even on held-out coverage regimes (EGLL dense vs sparse months), so the trees keep them as "how much to trust the movement features".
- **Regime shifts.** LFPG ran a runway-works configuration from Aug to Nov 2025. The 2026 runway mix matches normal months, and excluding the works months hurts, so all months are kept.
- **Recency or seasonal weighting** helped only during that temporary LFPG regime and is not used.

## 5. What did not work (measured)

- Leaderboard-independent but negative:
  - neural nets (MLP, deep MLP, physics-structured NN, 1-D CNN on ADS-B sequences, MC-dropout, Bayesian stacking);
  - CatBoost residual models;
  - Tweedie loss;
  - hyperparameter re-tuning (gains vanish at production learning rate);
  - neighbour ADS-B features;
  - arrival taxi-in surprises;
  - daily ATFM data;
  - published monthly levels;
  - covariate-shift weights;
  - runway-sequence physics.
- **LIRF "schedule echo".** About 9% of delayed LIRF departures have BLOCK = SCHED, so y equals the delay. A 2025-trained echo classifier reaches AUC 0.89 but does not transfer to 2026. Every LIRF echo/tail model was rejected; the base L2 model predicts the conditional mean.

## 6. Regular-phase note (transparency)

During the regular phase we submitted leaderboard probes and exact Gram-matrix re-blends of scored files: any linear blend of scored submissions has a computable score. Those blends are fitted to the Jan/Jul leaderboard and are **not used** in this pipeline or in the final submission.

Before the nested review, the candidate's label-free shrink already set LTFM self-distillation to zero. LIRF had affected 35,847 untracked departures with a shrink of about 0.46 before the half-weight step. Both distillation treatments are excluded from the final reviewed recipe; their checkpoints remain local research artifacts.
