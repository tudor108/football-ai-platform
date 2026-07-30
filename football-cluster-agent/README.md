# Football Cluster Agent

Vertex AI ADK + Gemini agent layer for explaining football clustering outputs.

## Scope
This module is read-only and sits on top of existing ETL + clustering outputs.
It does not modify ingestion, transformations, model training, or upload pipeline.

## Files
- `agent.py`: ADK agent definition
- `tools.py`: GCS/BigQuery read tools and helper explainers
- `requirements.txt`: dependencies
- `.env.example`: required environment variables

## Required environment variables
- `GOOGLE_CLOUD_PROJECT`
- `GCP_LOCATION`
- `CLUSTER_BUCKET`
- `BQ_DATASET`
- `CLUSTER_OUTPUT_PREFIX`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with real values.

## Run locally (ADK Web)
```bash
adk web
```

Then load this agent and ask:
- Explain the latest clustering run.
- Explain cluster 2 in simple terms.
- Why is Team X outlier-like?
- Compare Team A and Team B.
- I like attacking, in-form teams. Which clubs are most similar to Barcelona
  through that lens?
- Give me a statistical forecast for Barcelona at home against Real Madrid.

## Tools exposed
- `list_latest_clustering_run()`
- `read_gcs_file(path)`
- `read_latest_cluster_interpretation()`
- `read_latest_cluster_report()`
- `read_latest_metrics()`
- `read_latest_best_model_summary()`
- `query_bigquery(sql)` (SELECT-only safety)
- `explain_cluster(cluster_id)`
- `compare_teams(team_a, team_b)`
- `personalized_team_similarity(...)`
- `predict_match_from_stats(home_team, away_team)`

## Personalized similarity

The official cluster assignment remains unchanged. The agent maps a user's
stated preferences to five bounded dimensions (`attack`, `defense`, `results`,
`recent_form`, and `consistency`), reweights the feature-space distance, and
returns:

- nearest teams under that personal lens;
- distance to every official cluster centroid;
- preference-fit recommendations.

All similarity values are relative indexes, not probabilities. The current MVP
uses the active conversation context and does not persist a user profile across
Cloud Run restarts.

## Match forecasts

The prediction tool produces uncalibrated 1/X/2 and goal probabilities from
historical home/away scoring rates, Bayesian smoothing, recent form, and an
independent Poisson goals baseline. It also returns expected goals, likely
scores, data recency, confidence, and explicit limitations. It does not use
confirmed lineups, injuries, suspensions, bookmaker odds, or tactical matchup
data.

## Cloud Run deployment (later)
1. Build container image.
2. Deploy Cloud Run service with `football-agent-sa`.
3. Set env vars listed above.
4. Ensure IAM:
   - BigQuery Data Viewer + Job User
   - Storage Object Viewer
   - Vertex AI User
