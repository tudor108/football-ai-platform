"""Visualization module for clustering results."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot_labeled_scatter(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(11, 8))
    scatter = plt.scatter(df["pca_x"], df["pca_y"], c=df["cluster"], cmap="tab10", alpha=0.82, s=70)
    for _, row in df.iterrows():
        plt.text(row["pca_x"] + 0.015, row["pca_y"] + 0.015, str(row["team_name"]), fontsize=7, alpha=0.85)
    plt.title("Team Clusters (Labeled PCA Projection)")
    plt.xlabel("PCA X")
    plt.ylabel("PCA Y")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_scatter_labeled.png"), dpi=180)
    plt.close()


def _plot_cluster_sizes(df: pd.DataFrame, out_dir: str) -> None:
    sizes = df.groupby(["cluster", "cluster_label"], dropna=False).size().reset_index(name="count")
    sizes = sizes.sort_values("cluster")
    plt.figure(figsize=(10, 5))
    bars = plt.bar(sizes["cluster_label"], sizes["count"], color="#3f7fbf")
    for bar, cnt in zip(bars, sizes["count"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(int(cnt)), ha="center", va="bottom")
    plt.title("Cluster Sizes (Labeled)")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Teams")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_sizes_labeled.png"), dpi=180)
    plt.close()


def _plot_feature_heatmap(df: pd.DataFrame, features: list[str], out_dir: str) -> None:
    agg = df.groupby("cluster")[features].mean().copy()
    z = (agg - agg.mean()) / agg.std(ddof=0).replace(0, np.nan)
    z = z.fillna(0.0)

    plt.figure(figsize=(max(10, len(features) * 0.5), 5))
    plt.imshow(z.values, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="z-score vs cluster mean")
    plt.yticks(range(len(z.index)), [f"Cluster {c}" for c in z.index])
    plt.xticks(range(len(features)), features, rotation=45, ha="right")
    plt.title("Cluster Feature Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_feature_heatmap.png"), dpi=180)
    plt.close()


def _plot_cluster_radar(df: pd.DataFrame, features: list[str], out_dir: str) -> None:
    agg = df.groupby("cluster")[features].mean().copy()
    norm = (agg - agg.min()) / (agg.max() - agg.min()).replace(0, np.nan)
    norm = norm.fillna(0.5)

    top_features = norm.var().sort_values(ascending=False).index.tolist()[:6]
    angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for cluster_id, row in norm[top_features].iterrows():
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, values, linewidth=2, label=f"Cluster {cluster_id}")
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_features)
    ax.set_title("Cluster Radar (Top Variable Features)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_radar.png"), dpi=180)
    plt.close()


def _plot_top_feature_bars(df: pd.DataFrame, out_dir: str) -> None:
    records: list[dict[str, Any]] = []
    feature_slots = sorted(
        {c.replace("_name", "") for c in df.columns if c.startswith("feat_") and c.endswith("_name")}
    )
    for _, row in df.iterrows():
        for slot in feature_slots:
            feat_name = row.get(f"{slot}_name")
            feat_value = row.get(f"{slot}_value")
            if feat_name is None or feat_value is None:
                continue
            records.append({"cluster_label": row["cluster_label"], "feature": feat_name, "value": float(feat_value)})
    if not records:
        return

    plot_df = pd.DataFrame(records)
    plot_df = plot_df.groupby(["cluster_label", "feature"], as_index=False)["value"].mean()
    pivot = plot_df.pivot(index="cluster_label", columns="feature", values="value").fillna(0.0)
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.title("Top Differentiating Features by Cluster")
    plt.ylabel("Absolute deviation vs league mean")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_top_features.png"), dpi=180)
    plt.close()


def visualize_clusters(
    best_labels: pd.DataFrame,
    features_df: pd.DataFrame,
    features: list[str],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    merged = features_df.merge(best_labels, on=["team_id", "team_name"], how="inner")
    _plot_labeled_scatter(merged, out_dir)
    _plot_cluster_sizes(merged, out_dir)
    _plot_feature_heatmap(merged, features, out_dir)
    _plot_cluster_radar(merged, features, out_dir)
    _plot_top_feature_bars(best_labels, out_dir)
