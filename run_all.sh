#!/bin/bash
# Full reproduction of team strong-turtle's final submission (PRC Data Challenge 2026).
# Run from the repository root:  bash run_all.sh
# Prerequisites (see README.md):
#   data/                 organiser files: training_2025-*.parquet, final_ranking.parquet, final_submitting.parquet, submitting.parquet
#   data/adsb/            adsb.lol extracts made with kaggle/adsb_extract.py:
#                         adsb_2025h1.zip, adsb_2025h2.zip, adsb_2026janjul.zip, adsb_feb26.zip, adsb_jun26.zip
set -euo pipefail
cd "$(dirname "$0")"
export PRC_DATA="${PRC_DATA:-$PWD/data}"
export PRC_WORK="${PRC_WORK:-$PWD/work}"
mkdir -p "$PRC_WORK/adsblol" submission
A="$PRC_DATA/adsb"
JJ=("$A/adsb_2026janjul.zip")
if [[ -f "$A/adsb_2026janfix.zip" ]]; then JJ+=("$A/adsb_2026janfix.zip"); fi
step() { echo "=== $(date -u +%H:%M:%S) $*"; "$@"; }

# 1. weather + organiser data -> core tables
step python pipeline/fetch_metar.py                       # IEM ASOS METAR (public), 2025-01-01..2026-07-31, 10 airports
step python pipeline/build_data.py                        # dep.parquet / events.parquet / vocab.pkl
step python pipeline/build_arr.py                         # arrivals table (taxi-in is given for arrivals in every period)

# 2. base features (row-aligned with dep.parquet)
for s in f1_deltas f2_cong f2b_fix f3_roll f4_config f5_weather f6_struct f7_arr f8_physics f9_kalman f10_struct f11_latent f12_dynamics f13_fid f23c f29_flight; do
  step python pipeline/$s.py
done
step python pipeline/state_feats.py                       # airport/runway congestion state (movement table only)
step python pipeline/state_feats_v2.py                    # pier / runway mixed-mode / 15-min state

# 3. base model: 2025 out-of-fold residuals (target of the residual layers) + 3 seeds on all 2025 -> ranking rows
step python pipeline/oof_state.py
PRODSEED=1 step python pipeline/lgb_state2_seed.py        # production seed -> work/lgbstate2_rank_parts.npz
SEED=1 step python pipeline/lgb_state2_seed.py
SEED=2 step python pipeline/lgb_state2_seed.py

# 4. adsb.lol ground tracks -> per-departure session / movement-profile / stand-latency features
step python pipeline/adsb_feats2.py y2025h1 "$A/adsb_2025h1.zip"
step python pipeline/adsb_feats2.py y2025h2 "$A/adsb_2025h2.zip"
step python pipeline/adsb_feats2.py y2026 "${JJ[@]}"
step python pipeline/adsb_feats2.py y2026fj "$A/adsb_feb26.zip" "$A/adsb_jun26.zip"
ADDON_OF=y2025h1 step python pipeline/adsb_feats2.py y2025h1r "$A/adsb_2025h1.zip"      # ground-roll take-off links (add-on)
ADDON_OF=y2025h2 step python pipeline/adsb_feats2.py y2025h2r "$A/adsb_2025h2.zip"
ADDON_OF=y2026 step python pipeline/adsb_feats2.py y2026r "${JJ[@]}"
ADDON_OF=y2026fj step python pipeline/adsb_feats2.py y2026fjr "$A/adsb_feb26.zip" "$A/adsb_jun26.zip"
for t in y2025h1 y2025h1r; do step python pipeline/adsb_profile2.py $t "$A/adsb_2025h1.zip"; done
for t in y2025h2 y2025h2r; do step python pipeline/adsb_profile2.py $t "$A/adsb_2025h2.zip"; done
for t in y2026 y2026r; do step python pipeline/adsb_profile2.py $t "${JJ[@]}"; done
for t in y2026fj y2026fjr; do step python pipeline/adsb_profile2.py $t "$A/adsb_feb26.zip" "$A/adsb_jun26.zip"; done
step python pipeline/adsb_arrivals.py "${JJ[@]}" "$A/adsb_2025h1.zip" "$A/adsb_2025h2.zip"
step python pipeline/stand_lat_table.py

# 5. ADS-B residual layer (trained on 2025 only; per-airport scale from held-out Jan/Jul 2025)
TR=y2025h1,y2025h2,y2025h1r,y2025h2r
MXDIR="$PRC_WORK/adsblol/mx5/" REC=1 PROF=2 SLAT=1 step python pipeline/adsb_matrix.py $TR ""
MXDIR="$PRC_WORK/adsblol/mx5/" REC=1 PROF=2 SLAT=1 TEST_BASE=state TEST_ONLY=1 TEST_SUFFIX=s step python pipeline/adsb_matrix.py $TR y2026,y2026r,y2026fj,y2026fjr
MXDIR="$PRC_WORK/adsblol/mx5/" TAG=slat_sb NO_OPDI=1 step python pipeline/adsb_fit_lowmem.py $TR y2026s,y2026rs,y2026fjs,y2026fjrs build

# 6. corrections validated on 2025 + transductive steps on the unlabelled 2026 inputs
step python pipeline/lobt_tail.py
step python pipeline/clock_resid.py
step python pipeline/window_state_build.py
RFNAME=lgbstate2_selftrain_final.npz APTS=LFPG,EDDF,EGLL,LEMD,EHAM PSHARES="LFPG:0.3,EDDF:0.3,EGLL:0.3,LEMD:0.3,EHAM:0" \
  step python pipeline/lgb_state2_selftrain.py
# Base distillation is excluded: it regressed at LIRF in both nested holdout months.

# 7. final assembly (fixed recipe, exact LOBT window) -> submission/strong-turtle_final.parquet
W_D=0 step python pipeline/final_assemble.py
echo "DONE $(date -u)"
