"""Developer entrypoint: run ML clustering pipeline only."""

from __future__ import annotations

import argparse

from ml.clustering.team_clustering_pipeline import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run team clustering and stability analysis")
    parser.add_argument("--smoke", action="store_true", help="Use smoke-sized bootstrap iterations")
    arguments = parser.parse_args()
    result = main("ml/configs/team_clustering_config.yaml", smoke=arguments.smoke)
    print(f"[INFO] Run {result.run_id} completed with status {result.status}")
