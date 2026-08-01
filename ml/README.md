# ML Clustering Pipeline (Consultant Edition)

## Scope
This ML package builds high-quality football team clusters from the data-engineering outputs and can also run from BigQuery.

Primary goal: produce stable, interpretable team archetypes for analytics and downstream modeling.

## Data Source Strategy
Default mode is **local-first**:
- Input: `output/derived/fact_match_features.csv`
- The pipeline converts match-level home/away rows into team-level aggregates.

Optional modes:
- `input.source: bigquery`
- `input.source: csv`

## Feature Engineering
Team-level features include:
- Scoring and conceding profile: `avg_goals_scored`, `avg_goals_conceded`, `avg_goal_diff`, `std_goal_diff`
- Outcomes: `win_rate`, `draw_rate`, `loss_rate`, `avg_points`
- Recent form: `form_pts_avg`, `form_gf_avg`, `form_ga_avg`, `form_gd_avg`
- Form composition: `form_wins_avg`, `form_draws_avg`, `form_losses_avg`
- Context: `home_ratio`
- Composite strengths: `attack_strength`, `defense_strength`, `form_score`

## Model Search
The pipeline runs a parameter sweep across enabled algorithms:
- KMeans
- MiniBatchKMeans
- GaussianMixture
- AgglomerativeClustering
- SpectralClustering
- Birch
- DBSCAN
- MeanShift
- HDBSCAN (if installed)

## Model Selection
Each candidate gets:
- Silhouette
- Davies-Bouldin
- Calinski-Harabasz
- Noise ratio
- Cluster balance ratio

Best model is selected via `comparison.select_by`:
- `composite` (recommended default)
- `silhouette`
- `davies_bouldin`
- `calinski_harabasz`

## Outputs
Each successful run is immutable under `output/ml/runs/<run_id>/`; the flat
paths remain as temporary backward-compatible copies:
- `output/ml/clusters/clusters.csv`
- `output/ml/metrics/metrics.json`
- `output/ml/metrics/best_model_summary.json`
- `output/ml/metrics/cluster_interpretation.json`
- `output/ml/plots/pca_scatter.png`
- `output/ml/plots/cluster_sizes.png`
- `output/ml/search/agent_search_documents.jsonl`

`output/ml/latest_run.json` advances only after required artifacts validate and
the run status is `SUCCEEDED`.

## Local Run (Recommended First)
```bash
py -m ml.clustering.team_clustering_pipeline --config ml/configs/team_clustering_config.yaml
```

## Full Pipeline Integration
`data_engineering/run_all_local.py` now includes ML as Step 5:
1. Extract
2. Upload raw
3. Transform
4. Derive
5. **ML clustering**
6. Upload processed outputs (includes `output/ml/*`)
7. Load to BigQuery

## Configuration Notes
Main config file:
- `ml/configs/team_clustering_config.yaml`

Key knobs:
- `preprocessing`: fill strategy, scaler, PCA
- `algorithms`: enable/disable + parameter grids
- `comparison`: model selection rule and cluster bounds
- `output`: artifact destinations
