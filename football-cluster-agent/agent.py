from __future__ import annotations

from google.adk.agents import Agent

from tools import (
    compare_teams,
    explain_cluster,
    list_latest_clustering_run,
    query_bigquery,
    read_gcs_file,
    read_latest_best_model_summary,
    read_latest_cluster_interpretation,
    read_latest_cluster_report,
    read_latest_metrics,
)

INSTRUCTION = """
You are a Football Analytics Explanation Agent.

Behavior requirements:
- Use GCS and BigQuery as source of truth.
- Never invent missing values; explicitly say when data is missing.
- Explain results in beginner-friendly language.
- Explain clusters, PCA scatter, heatmap, cluster sizes, and model metrics.
- Help compare teams and clusters using available artifacts.

Metrics interpretation:
- Silhouette: higher is better.
- Davies-Bouldin: lower is better.
- Calinski-Harabasz: higher is better.

Recommended tool usage:
1) list_latest_clustering_run
2) read_latest_cluster_interpretation / read_latest_metrics / read_latest_best_model_summary
3) read_latest_cluster_report or read_gcs_file for artifact details
4) explain_cluster / compare_teams
5) query_bigquery for supporting table context
"""

root_agent = Agent(
    name="football_cluster_explainer",
    model="gemini-2.0-flash",
    description="Explains football clustering artifacts using GCS + BigQuery data.",
    instruction=INSTRUCTION,
    tools=[
        list_latest_clustering_run,
        read_gcs_file,
        read_latest_cluster_interpretation,
        read_latest_cluster_report,
        read_latest_metrics,
        read_latest_best_model_summary,
        query_bigquery,
        explain_cluster,
        compare_teams,
    ],
)
