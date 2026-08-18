# Football Cluster Agent

Vertex AI ADK + Gemini agent layer for explaining football clustering outputs.

## Scope
This module is read-only and sits on top of existing ETL + clustering outputs.
It does not modify ingestion, transformations, model training, or upload pipeline.

## Files
- `agent.py`: ADK agent definition
- `tools.py`: GCS/BigQuery adapters and all agent tool entrypoints
- `analytics.py`: deterministic personalized similarity and Poisson forecast logic
- `intelligence.py`: dossier, audit, alerts, scouting, scenarios and fan products
- `agent_search.py`: read-only historical semantic search adapter
- `business_api/main.py`: FastAPI/OpenAPI surface over the same business functions
- `requirements.txt`: dependencies
- `.env.example`: required environment variables

## Required environment variables
- `GOOGLE_CLOUD_PROJECT`
- `GCP_LOCATION`
- `CLUSTER_BUCKET`
- `BQ_DATASET`
- `CLUSTER_OUTPUT_PREFIX`

Optional feature variables:

- `GOOGLE_GENAI_MODEL` (code default: `gemini-2.5-flash`)
- `AGENT_SEARCH_DATA_STORE_ID`
- `AGENT_SEARCH_ENGINE_ID`
- `AGENT_SEARCH_LOCATION`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with real values.

PowerShell equivalent:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
Copy-Item .env.example .env
```

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
- What are the latest completed matches and newest date available in the data?

## Tools exposed
- `list_latest_clustering_run()`
- `read_gcs_file(path)`
- `read_latest_cluster_interpretation()`
- `read_latest_cluster_report()`
- `read_latest_metrics()`
- `read_latest_best_model_summary()`
- `get_latest_available_matches(limit)`
- `query_bigquery(sql)` (SELECT-only safety)
- `explain_cluster(cluster_id)`
- `compare_teams(team_a, team_b)`
- `personalized_team_similarity(...)`
- `predict_match_from_stats(home_team, away_team)`
- `generate_opponent_dossier(home_team, away_team)`
- `audit_match_forecast_quality(max_evaluated)`
- `list_business_alerts()`
- `recommend_players_for_team(...)`
- `simulate_match_absences(...)`
- `create_fan_briefing(...)`
- `get_match_companion(fixture_id)`
- `get_video_evidence(fixture_id)`
- `search_historical_analytics(...)`

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

## Business API

The same deterministic tools are available as a white-label FastAPI service:

```bash
uvicorn business_api.main:app --host 0.0.0.0 --port 8081
```

Interactive OpenAPI documentation is at `/docs`.

## Cloud Run deployment

Stable service URL:
`https://football-agent-hzdooca3ia-uc.a.run.app/dev-ui/`

The current Cloud Run revision is `football-agent-00013-nx4`.
Cost-safe serving configuration currently uses `minInstanceCount=1`,
`maxInstanceCount=5`, `cpuIdle=true`, 512 MiB memory and 1 vCPU. The minimum
instance is intentionally kept warm because the ADK Web UI can return transient
429 responses during cold-start/recovery; reduce to scale-to-zero only after
verifying the UI remains reliable.

After a billing reactivation, Cloud Run may return `429 no available instance`
for up to approximately 30 minutes while serving capacity recovers.

1. Build container image.
2. Deploy Cloud Run service with `football-agent-sa`.
3. Set env vars listed above.
4. Ensure IAM:
   - BigQuery Data Viewer + Job User
   - Storage Object Viewer
   - Vertex AI User
   - Discovery Engine Viewer (for historical Agent Search)
