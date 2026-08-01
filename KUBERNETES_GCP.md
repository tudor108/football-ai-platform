# Kubernetes minimal pe Google Cloud

## Arhitectură

- `Deployment/football-agent`: interfața conversațională ADK + Gemini.
- `Deployment/football-business-api`: API FastAPI pentru integrare white-label.
- `CronJob/football-daily-pipeline`: ETL + ML + publicare GCS + BigQuery + import
  opțional Agent Search.
- două Kubernetes Service Accounts legate prin Workload Identity Federation de
  conturile GCP; nu există chei JSON în manifeste.
- servicii `ClusterIP`, deci nu se creează automat un load balancer public.

Manifestele sunt în `k8s/`. CronJob-ul are inițial `suspend: true`, deoarece
repo-ul are deja scheduler GitHub la `02:32 UTC`. Activează un singur scheduler,
altfel două rulări pot concura pentru pointerul `latest`.

## Cerințe

1. Un cluster GKE cu Workload Identity activ.
2. API-urile GKE, Cloud Build și Artifact Registry activate.
3. Roluri minime pentru GSA-urile alese: Storage object read/write după nevoie,
   BigQuery Job User/Data Editor pentru pipeline, BigQuery Data Viewer și
   Storage Object Viewer pentru agent/API, Vertex AI User pentru agent.
   Pentru istoric: Discovery Engine Editor pe GSA-ul pipeline și Discovery
   Engine Viewer pe GSA-ul agent/API.
4. Secretul Kubernetes creat separat:

```powershell
kubectl create namespace football-analytics --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic football-api-secrets `
  --namespace football-analytics `
  --from-literal=API_FOOTBALL_KEY=$env:API_FOOTBALL_KEY
```

## Build și deploy

Scriptul nu creează clusterul; acesta trebuie ales explicit:

```powershell
.\scripts\deploy_gke.ps1 -Cluster NUMELE_CLUSTERULUI
```

Pentru a muta și programarea zilnică din GitHub Actions în GKE:

```powershell
.\scripts\deploy_gke.ps1 -Cluster NUMELE_CLUSTERULUI -EnableCronJob
```

În acel moment dezactivează schedule-ul GitHub sau păstrează CronJob-ul
suspendat. Verificare manuală înainte de activare:

```powershell
kubectl create job --from=cronjob/football-daily-pipeline pipeline-smoke `
  --namespace football-analytics
kubectl logs --namespace football-analytics job/pipeline-smoke --follow
```

GKE și imaginile container nu sunt în mod normal costuri Agent Search/GenAI App
Builder. Pentru învățare, păstrează un cluster mic/Autopilot, servicii interne și
șterge resursele de laborator când nu le mai folosești.
