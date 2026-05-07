"""Visualization module for clustering results."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_clusters(X_proc: np.ndarray, best_labels: pd.DataFrame, config: dict[str, Any]) -> None:
    clusters = best_labels["cluster"].to_numpy()
    out_dir = config["output"].get("plots_dir", "output/ml/plots")
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    x = X_proc[:, 0]
    y = X_proc[:, 1] if X_proc.shape[1] > 1 else np.zeros_like(x)
    scatter = plt.scatter(x, y, c=clusters, cmap="tab10", alpha=0.8)
    plt.title("Team Clusters Projection")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_scatter.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    pd.Series(clusters).value_counts().sort_index().plot(kind="bar", color="#3f7fbf")
    plt.title("Cluster Sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Teams")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_sizes.png"), dpi=160)
    plt.close()
