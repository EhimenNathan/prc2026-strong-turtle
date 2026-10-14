"""Central paths for the PRC Data Challenge 2026 pipeline. Run every script from the repository root (python pipeline/<script>.py)."""
import os
DATA_DIR = os.environ.get("PRC_DATA", os.path.abspath("data"))          # organiser parquet files (training_2025-*.parquet, final_ranking.parquet, ...)
WORK_DIR = os.environ.get("PRC_WORK", os.path.abspath("work"))          # derived artifacts
RANKING = os.environ.get("PRC_RANKING", "final_ranking.parquet")        # the ranking file to predict
SUBMIT_TEMPLATE = os.environ.get("PRC_SUBMIT_TEMPLATE", "final_submitting.parquet")
