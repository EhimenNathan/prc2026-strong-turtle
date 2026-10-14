# Validation evidence and limits

The final candidate is trained on supplied 2025 departure labels and uses published 2026 inputs, including arrival taxi-in observations and public ground tracks. All final departure taxi-out and block-time labels were checked to be blank. No hidden 2026 departure labels were obtained.

Earlier model development included regular-phase leaderboard probes. The final combination does not load the exact Gram blends or their leaderboard-fitted weights. This exclusion does not make the entire development history blind to leaderboard feedback; component-selection history is disclosed in METHOD.md.

## Existing evidence

- `audit_existing.json` verifies the final template, month counts, feature row counts and category-code reconstruction.
- `base_oof_2025_by_month.csv` and `base_oof_2025_by_airport_month.csv` report the quarterly cross-fit base surrogate over all twelve months. That surrogate uses 800 rounds at learning rate 0.05, rather than the production model's lower learning rate and airport-specific rounds.
- Existing ADS-B and clock residual validation uses held-out January and July folds on stored base cross-fit predictions. Base models supplying some residual-training rows included labels from the residual validation month. Consequently these results are **not fully nested end-to-end cross-validation**, and reported gains can be optimistic.
- Airport scales and deployment gates are selected on the same 2025 component validation predictions on which their gains are reported. Those gains are tuning evidence, not untouched-test estimates.
- Some airports have sufficient ADS-B validation coverage in only one of January and July. The code's gate checks available months with more than 300 rows. It does not require two qualifying months at every airport.
- Earlier ADS-B self-training simulations use a first-half residual fit on upstream quarterly cross-fit base predictions. Those upstream predictions can include second-half labels in their training sets, so the reported self-training gains are exploratory rather than fully nested. The historical helper `adsb_h1_pseudo.py` also expects an earlier `mx6` feature matrix; it is not part of the production rebuild in `run_all.sh`.
- Weather, traffic, flight identifiers and arrival observations are available retrospectively in the challenge. This solution reconstructs taxi-out from a complete movement dataset; it is not a pre-departure forecast validated using only real-time information.

## Additional February/June check

`pipeline/validate_feb_jun.py` excludes **both February and June 2025 labels from every base and ADS-B fit**. Inner quarter folds on the remaining months generate residual-training targets. The ADS-B features containing the base prediction and base-minus-clock gaps are rebuilt with those nested predictions. The outer ADS-B scale is fixed at 1.0 and is not calibrated on the test labels.

This diagnostic tests a base + ADS-B surrogate, with results separated by airport, month and track coverage. It excludes seed averaging, clock/window/tail corrections and transductive steps. Earlier feature gates and hyperparameters were developed on 2025, so this check is useful additional generalization evidence, not an untouched-data certification of the full production recipe.

`pipeline/validate_transduction.py` separately uses that diagnostic's predictions as pseudo-labels. It retrains on the remaining true labels plus pseudo-labelled February/June rows and scores unlinked departures at the fixed half-weight blend. Both held-out months' true labels are excluded from the treatment fits and their target encodings. This is a surrogate check of self-training and base distillation; it does not reproduce the production seed-noise shrink or the complete correction stack. A synthetic target-isolation regression check passed before the job was queued.

## Submission integrity

Assembly requires all three complete base seeds, all five self-training airports, the four track tags, and all four correction artifacts. The two distillation seeds are required only for an explicitly enabled research comparison; production distillation is disabled following the nested check. Assembly rejects missing components, mismatched IDs, duplicates and nonfinite values. Exactly represented integral floating-point correction IDs are normalized safely to integers. The final output matches the organiser template order and has 670,790 rows.

The ±3,606-second LOBT support rule was checked on all 2,062,577 supplied 2025 departures with both timestamps, with zero violations. It is a verified property of that dataset, not a proof about an unrelated future dataset.

No 2026 final RMSE, final ranking or competition win is asserted before organiser scoring. No source push or prediction upload is performed by the local training and finalization jobs.
