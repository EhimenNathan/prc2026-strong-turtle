# Reproduction notes

Run from the repository root using Python 3.13 and the pinned requirements. `run_all.sh` expects Bash; a Windows machine can use Git Bash, WSL, or run the documented Python steps with equivalent environment variables. Path construction inside Python is portable. Use a fresh work directory; existing research checkpoints do not have the new dependency receipts and cannot be used as verified resumable package checkpoints.

## Supplied derived archives

The local candidate used these files. Public users regenerate them using `kaggle/adsb_extract.py`; no organiser or per-flight data are committed.

| Supplied filename | Reproduction filename | Daily files |
|---|---|---:|
| `adsb_out2 (2).zip` | `data/adsb/adsb_2025h1.zip` | 181 |
| `adsb_out2 (1).zip` | `data/adsb/adsb_2025h2.zip` | 183 |
| `adsb_out2.zip` | `data/adsb/adsb_2026janjul.zip` | 62 |
| `adsb_fix_2026jan.zip` | `data/adsb/adsb_2026janfix.zip` | 7 |
| `adsb_feb26.zip` | `data/adsb/adsb_feb26.zip` | 28 |
| `adsb_jun26.zip` | `data/adsb/adsb_jun26.zip` | 30 |

The H2 archive ends on December 30, 2025. December 31 still has organiser movements and base-model labels, but no ground-track residual links in these supplied extracts. The January repair contains January 3–9, 2026. When a day appears twice, the feature builders prefer the larger daily file; this rule is also implemented by `merge_adsb_archives.py`.

To combine separate January and July extractions:

```bash
python merge_adsb_archives.py data/adsb/adsb_2026janjul.zip january.zip july.zip
```

The archive manifest records original archive hashes and each daily file's size and CRC32. Fresh extraction from mutable public releases may produce different ground samples; a matching archive manifest is necessary for strict numerical comparison. The current extractor has a fallback for nearly empty OPDI aircraft lists. Historical archive logs should accompany any numerical reproduction.

## Category-code ordering

The research project first sorted categories from all 2025 files and the regular January/July ranking file, then appended sorted categories first seen in the final ranking file. Sorting the whole four-month union from scratch changes the old integer codes. `build_data.py` now reconstructs the historical order deterministically from raw files, independent of any cached vocabulary. All fourteen vocabulary fields were checked against the existing candidate's vocabulary, with exact matches.

## State-feature dependency

The original state feature script dynamically read the observable functions from two older research scripts. The final package contains those functions in `pipeline/state_core.py`. It does not execute the earlier leaderboard-probe entry points. AST comparisons confirmed that all five extracted functions preserve the research definitions exactly; see `state_source_verification.json`.

## Verification scope

Python syntax and strict assembly gates are tested. Existing data were audited for final row IDs, blank departure labels and feature row counts. The final candidate was assembled from completed, checked research checkpoints. The entire raw-data pipeline has not yet been rerun from an empty directory, and no bit-for-bit full-rebuild claim is made. The nested February/June diagnostic runs separately and records its own prediction and metric files.

All 20 core/state entry points passed a fresh-directory raw-fixture rebuild (6,402 departures). Perturbing 4,802 historical departure targets and block times left all 20 feature/arrival frames unchanged exactly. See `feature_smoke_rebuild.json` and `feature_label_invariance.json`. Model target isolation and assembly failure cases have portable regression checks in `tests/`.

The reviewed candidate disables base distillation following the nested February/June 2025 regression at LIRF. `candidate_manifest.json` records the exact final file hash; `candidate_review_change.json` records the change from the earlier candidate.

The extractor now chooses split tar parts instead of concatenating a standalone archive with its split representation. A separate June 6/11 replica probe recovered more June 6 samples, but the repair artifact could not be retrieved locally and was **not used in this candidate**. June 11 was sparse in both tested replicas. The candidate's archive manifest describes its actual supplied extracts; a fresh extraction can differ. `track_archive_quality.json` records label-free screening of all 58 February/June days.
