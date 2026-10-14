# Nested February/June 2025 review

Both months were excluded from every base and ADS-B label fit. Transductive pseudo-labels come from those nested predictions, and true held-out departure labels are used only for scoring. ADS-B scale is fixed at 1.0; self-training blend weight is fixed at 0.5.

| Month | Rows | Base RMSE (s) | Base + ADS-B | Base + ADS-B + self-training |
|---|---:|---:|---:|---:|
| 2 | 143,732 | 306.832 | 296.761 | 296.242 |
| 6 | 183,142 | 330.149 | 316.968 | 316.660 |

All five self-training airports improved on unlinked departures in both months. LIRF base distillation regressed from 676.365 to 678.855 seconds in February and from 636.487 to 654.935 in June. The reviewed recipe disables distillation. LTFM production distillation was already shrunk to zero.

Paired UTC-day bootstrap intervals for the base-to-reviewed-surrogate RMSE reduction (2,000 replicates):

- Month 2: 7.90 to 15.28 seconds.
- Month 6: 9.73 to 19.39 seconds.

See the aggregate CSV/JSON reports for coverage strata and airport results. This surrogate excludes production seed averaging, clock/window/tail corrections and LOBT projection. Earlier feature/parameter choices used 2025 development evidence, and this review now informs deployment selection. It is not an untouched final test and provides no 2026 RMSE.
