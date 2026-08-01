param(
    [Parameter(Mandatory = $true)][string]$Cluster,
    [string]$Region = "europe-central2",
    [string]$Project = "project-73d1e32a-8e68-4750-93c",
    [string]$Repository = "football-analytics",
    [string]$PipelineGsa = "github-actions-uploader@project-73d1e32a-8e68-4750-93c.iam.gserviceaccount.com",
    [string]$AgentGsa = "football-agent-sa@project-73d1e32a-8e68-4750-93c.iam.gserviceaccount.com",
    [switch]$EnableCronJob
)

$ErrorActionPreference = "Stop"
$registry = "$Region-docker.pkg.dev/$Project/$Repository"

gcloud container clusters get-credentials $Cluster --region $Region --project $Project
gcloud artifacts repositories describe $Repository --location $Region --project $Project 2>$null
if ($LASTEXITCODE -ne 0) {
    gcloud artifacts repositories create $Repository --repository-format docker --location $Region --project $Project
}

gcloud builds submit --project $Project --config k8s/cloudbuild-pipeline.yaml --substitutions "_IMAGE=$registry/pipeline:latest" .
gcloud builds submit --project $Project --tag "$registry/agent:latest" football-cluster-agent
gcloud builds submit --project $Project --config k8s/cloudbuild-api.yaml --substitutions "_IMAGE=$registry/business-api:latest" .

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/service-accounts.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/cronjob.yaml
kubectl apply -f k8s/agent.yaml
kubectl apply -f k8s/business-api.yaml

gcloud iam service-accounts add-iam-policy-binding $PipelineGsa --project $Project --role roles/iam.workloadIdentityUser --member "serviceAccount:$Project.svc.id.goog[football-analytics/football-pipeline]"
gcloud iam service-accounts add-iam-policy-binding $AgentGsa --project $Project --role roles/iam.workloadIdentityUser --member "serviceAccount:$Project.svc.id.goog[football-analytics/football-agent]"
kubectl annotate serviceaccount football-pipeline --namespace football-analytics "iam.gke.io/gcp-service-account=$PipelineGsa" --overwrite
kubectl annotate serviceaccount football-agent --namespace football-analytics "iam.gke.io/gcp-service-account=$AgentGsa" --overwrite

if ($EnableCronJob) {
    kubectl patch cronjob football-daily-pipeline --namespace football-analytics --type merge --patch '{"spec":{"suspend":false}}'
}

kubectl rollout status deployment/football-agent --namespace football-analytics
kubectl rollout status deployment/football-business-api --namespace football-analytics
kubectl get all --namespace football-analytics
