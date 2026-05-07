"""Developer entrypoint: run ML clustering pipeline only."""

from __future__ import annotations

from ml.clustering.team_clustering_pipeline import main


if __name__ == "__main__":
    main("ml/configs/team_clustering_config.yaml")
