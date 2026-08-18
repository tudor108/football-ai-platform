# Football Analytics - clustering, MLOps and Gemini agent

Pipeline batch pentru La Liga care transforma date API-Football in tabele
analitice, compara mai multi algoritmi de clustering si publica rezultate
explicabile prin GCS, BigQuery, Agent Search, un agent Gemini si un API FastAPI.

## Starea verificata

- audit local al codului si documentatiei: `2026-08-10`;
- ultima rulare ML locala si publicata valida: `20260817T115844Z`;
- date pana la `2025-05-25`: 380 meciuri, 20 echipe;
- model selectat: `AgglomerativeClustering`, 4 clustere, `linkage=ward`;
- fiecare rulare reusita este imutabila sub `output/ml/runs/<run_id>/`;
- `latest_run.json` avanseaza numai dupa validarea unei rulari `SUCCEEDED`;
- suita curenta: 30 teste `pytest` trecute;
- modulul de stabilitate exista, dar `stability.enabled` este `false`;
- manifestele GKE exista, dar nu creeaza si nu confirma un cluster activ;
- ultima rulare nu are ploturi versionate, deoarece mediul ei nu avea `matplotlib`.
- bucketul si BigQuery au fost sincronizate la `2026-08-17`;
- Agent Search a importat documentele noului run (`21/21`);
- Cloud Run revision curent: `football-agent-00013-nx4`, cu `min=1`, `max=5`, CPU idle;
- dupa reactivarea billingului, Cloud Run poate raspunde temporar cu 429 pana la propagarea completa.

## Pornire rapida

```powershell
py -m pip install -r requirements.txt
py run_features.py
py run_ml.py
py -m pytest -q
```

Pipeline complet, cu servicii cloud configurate:

```powershell
py data_engineering/run_all_local.py
```

Agent local:

```powershell
Set-Location football-cluster-agent
py -m pip install -r requirements.txt
adk web
```

API local: `uvicorn business_api.main:app --reload --port 8081`, apoi OpenAPI la
`http://localhost:8081/docs`.

## Harta documentatiei

- [Documentatie tehnica completa](DOCUMENTATIE_PROIECT.md)
- [ML, Cloud si Data](DOCUMENTATIE_ML_CLOUD_DATA.md)
- [Data Science, model, features si prompturi](DOCUMENTATIE_DATA_SCIENCE_MODEL_PROMPTS.md)
- [Functionalitati business si personalizare](BUSINESS_FEATURES.md)
- [Kubernetes minimal pe GCP](KUBERNETES_GCP.md)
- [Pipeline-ul ML](ml/README.md)
- [Agentul Gemini si API-ul business](football-cluster-agent/README.md)

Documentele descriu separat ce este activ, ce este optional si ce este doar
pregatit pentru urmatoarea iteratie, pentru a evita confundarea codului existent
cu resurse cloud deja deployate.

## Structura pentru GitHub

Entry-point-urile `run_extract.py`, `run_features.py`, `run_ml.py` si
`run_reporting.py` raman in root ca sa poata fi rulate direct si ca sa pastreze
compatibilitatea cu workflow-urile existente. Scripturile de deployment si
setup sunt grupate in `scripts/`.

Datele brute si output-urile generate nu sunt sursa de adevar pentru GitHub:

- `data/raw/`, `output/tables/`, `output/derived/` si `output/ml/` sunt fisiere
  locale generate si ignorate de Git pentru rulările viitoare;
- fiecare run ML complet este publicat imutabil in GCS la
  `output/ml/runs/<run_id>/`;
- `latest_run.json` indica ultima rulare validata;
- pentru vizitatori exista doar exemplul redus din `examples/sample_run/`.

Astfel, repository-ul public contine codul, testele si documentatia, iar
artefactele voluminoase raman versionate in cloud. Nu comite niciodata `.env`,
chei API sau credentiale GCP.

Variabilele necesare local sunt exemplificate în [`.env.example`](.env.example);
copiează-l ca `.env` și completează valorile doar pe stația ta.

La prima curatare a unui clone in care datele au fost deja urmarite de Git,
ruleaza o singura data (fara sa stergi fisierele locale):

```powershell
git rm -r --cached data/raw output/tables output/derived
git commit -m "Keep generated data out of source repository"
git push origin main
```
