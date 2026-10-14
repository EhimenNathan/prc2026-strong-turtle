# Data sources and licences

The final pipeline is trained on 2025 labels only. From the ranking period it uses only what the organisers publish in the ranking
files. The 2026 departure block and taxi times are blanked and were never obtained. No weight, scale or step size in the final pipeline
was fitted to leaderboard scores: every one is estimated on held-out 2025 months, derived from label-free quantities such as seed
noise, or a fixed documented constant. (During the regular phase we also submitted leaderboard-blended files; that blending is
deliberately **not** part of this pipeline. See `docs/METHOD.md`.)

## Organiser data (not redistributed)

PRC Data Challenge 2026 files, downloaded from the organisers' S3 bucket `prc-2026-datasets` with team credentials:

- `training_2025-MM-01_*.parquet` (12 files): 2025 movements with labels.
- `final_ranking.parquet`: Jan/Feb/Jun/Jul 2026 movements. Departure block time and taxi time are blanked; arrivals are complete.
- `final_submitting.parquet`: the template.
- `ranking.parquet` and `submitting.parquet`: the regular-phase Jan/Jul subset, used only for a consistency check.

Put them in `data/`.

## External data (all openly available)

| Source | What we use | Licence | How it is obtained |
|---|---|---|---|
| **adsb.lol `globe_history_2025` / `globe_history_2026`** (https://github.com/adsblol) | ADS-B / MLAT traces. Kept: on-ground and below-1500-ft samples within about 8 km of the 10 airports, for aircraft departing them. Used for stand dwell, push, stop, take-off and ground-speed features. | **ODbL 1.0** | `kaggle/adsb_extract.py` streams each day's release on Kaggle. Nothing large is stored; it outputs one zip of per-day parquet files per date range. |
| **EUROCONTROL OPDI flight lists** (`https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/`) | Only the list of `icao24` addresses departing the 10 airports, to pre-filter traces during extraction. | Open-data eligibility confirmation from the organisers, as quoted in the supplied handoff. No specific data licence identifier was supplied. The code repository's MIT licence is not asserted as the data licence. | Downloaded automatically by `kaggle/adsb_extract.py`. |
| **Iowa Environmental Mesonet ASOS/METAR** (https://mesonet.agron.iastate.edu/request/download.phtml) | Hourly METAR for the 10 airports: temperature, wind, gusts, visibility, ceiling, precipitation, icing. | Public domain, according to the [IEM disclaimer](https://mesonet.agron.iastate.edu/disclaimer.php). | `pipeline/fetch_metar.py` |

## Data we tested and did not use in the final pipeline

- **EUROCONTROL daily ATFM files** (slot adherence, airport arrival ATFM delay). Their apparent gain was entirely an airport-month level shift that does not generalise.
- **EUROCONTROL monthly taxi-out additional time.** Our own predictions already beat it as a level anchor.
- **OPDI flight events (runway/parking events).** Coverage is too thin at most airports, and it is unavailable for Feb/Jun at the time of the final phase.
