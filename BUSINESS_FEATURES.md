# Football Analytics – business intelligence layer

## Ce s-a adăugat

| Capabilitate | Funcția pură | Tool Gemini / endpoint HTTP | Valoare business |
|---|---|---|---|
| Dossier adversar | `build_opponent_dossier` | `generate_opponent_dossier` / `GET /v1/dossiers/opponent` | pregătire pre-meci: formă, stil, jucători, H2H și forecast |
| Audit forecast | `backtest_match_forecasts` | `audit_match_forecast_quality` / `GET /v1/forecasts/audit` | măsoară Brier, log-loss, accuracy și calibrare fără data leakage |
| Alerte | `generate_business_alerts` | `list_business_alerts` / `GET /v1/alerts` | date stale, schimbări de formă, volatilitate și mișcare între clustere |
| Player–team fit | `rank_player_team_fit` | `recommend_players_for_team` / `GET /v1/scouting/player-fit` | shortlist transparent pentru scouting |
| Scenarii lot | `simulate_lineup_absences` | `simulate_match_absences` / `POST /v1/scenarios/absences` | impact estimativ al absențelor asupra xG și 1/X/2 |
| Fan briefing | `build_personalized_fan_briefing` | `create_fan_briefing` / `POST /v1/fans/briefing` | conținut personalizat pentru cluburile urmărite |
| Match companion | `build_live_match_companion` | `get_match_companion` / `GET /v1/matches/{id}/companion` | timeline și statistici din cel mai nou snapshot stocat |
| Video evidence | `attach_video_evidence` | `get_video_evidence` / `GET /v1/matches/{id}/video-evidence` | contract sigur pentru clipuri licențiate; nu inventează URL-uri |
| Căutare istorică | `build_agent_search_documents` | `search_historical_analytics` / `GET /v1/history/search` | căutare semantică peste rulările ML imutabile |

Funcțiile deterministe sunt în `football-cluster-agent/football_agent/intelligence.py`.
Adaptorul GCS/BigQuery și tool-urile agentului sunt în
`football-cluster-agent/football_agent/tools.py`. API-ul white-label este în
`football-cluster-agent/business_api/main.py`.

## Ce înseamnă „subiectiv”

Clusterizarea oficială rămâne neschimbată. Preferințele utilizatorului, fiecare
între `-1` și `1`, modifică numai ponderile folosite pentru o comparație
personală: atac, apărare, rezultate, formă recentă și consistență. Rezultatul
este un index relativ de similaritate/fit, nu o probabilitate și nu rescrie
artefactul oficial.

Player fit folosește aceeași separare: statisticile jucătorilor produc patru
indecși explicabili (atac, apărare, creație, fiabilitate), iar preferințele
reponderează shortlist-ul. Nu sunt incluse automat taxa de transfer, contractul,
personalitatea, medicalul sau tracking-ul off-ball.

## Forecast și responsabilitate

Forecast-ul 1/X/2 este un baseline Poisson necalibrat. Auditul este walk-forward:
pentru fiecare meci folosește numai meciuri cu dată anterioară. Simularea
absențelor limitează efectele individuale și este etichetată explicit drept
euristică, nu efect cauzal și nu recomandare de pariuri.

## Fluxul unei rulări zilnice

1. GitHub Actions sau containerul CronJob rulează
   `python data_engineering/run_all_local.py`.
2. „Local” înseamnă filesystem-ul procesului: laptop, runner GitHub sau pod GKE.
3. Se extrag datele, se urcă raw în GCS, apoi transformarea citește din GCS și
   produce CSV-uri local în `output/`.
4. ML scrie mai întâi într-un director nou
   `output/ml/runs/<run_id>/`; rulările istorice reușite nu sunt suprascrise.
5. În rulare se generează și
   `search/agent_search_documents.jsonl`.
6. Numai după succes se publică directorul versiunii în GCS și se actualizează
   controlat `latest_run.json` plus căile plate compatibile.
7. Dacă `AGENT_SEARCH_DATA_STORE_ID` există, pornește un import incremental al
   JSONL-ului. Eșecul opțional de indexare nu invalidează rularea ML deja
   publicată.
8. Tabelele transformate și derivate sunt încărcate în BigQuery.

## Agent Search și creditul GenAI App Builder

Provisionare Standard, cost-safe:

```powershell
python scripts/setup_agent_search.py --project project-73d1e32a-8e68-4750-93c
```

Pentru Enterprise Search cu add-on LLM, numai după verificarea eligibilității
creditului în Billing:

```powershell
python scripts/setup_agent_search.py --project project-73d1e32a-8e68-4750-93c --enterprise-generative
```

Apoi se setează în mediul pipeline-ului și al agentului:

```text
AGENT_SEARCH_DATA_STORE_ID=football-analytics-history
AGENT_SEARCH_ENGINE_ID=football-analytics-search
AGENT_SEARCH_LOCATION=global
```

Service agent-ul Agent Search are nevoie numai de citire pe bucket:

```powershell
gcloud storage buckets add-iam-policy-binding gs://football-ai-raw-data-tudor `
  --member="serviceAccount:service-205718552008@gcp-sa-discoveryengine.iam.gserviceaccount.com" `
  --role="roles/storage.objectViewer"
```

Contul agentului primește `roles/discoveryengine.viewer`, iar identitatea care
face importul zilnic primește `roles/discoveryengine.editor`. Primul rol caută
read-only; al doilea este necesar pentru importarea documentelor noi.

Creditul GenAI App Builder este potrivit pentru această componentă de căutare;
nu trebuie presupus că acoperă GKE, Artifact Registry, BigQuery sau API-Football.
Eligibilitatea exactă se verifică în pagina Billing → Credits a proiectului.

## API local

Din `football-cluster-agent/`, după configurarea `.env`:

```powershell
uvicorn business_api.main:app --reload --port 8081
```

OpenAPI: `http://localhost:8081/docs`. API-ul și agentul folosesc exact aceleași
funcții business, astfel încât definițiile nu diverg.
