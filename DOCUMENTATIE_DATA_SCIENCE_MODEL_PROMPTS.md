# Documentatie Data Science - prompturi, model, cod si features

Document pentru prezentare tehnica in fata unui Data Scientist.
Stare proiect verificata local: `2026-08-10`.
Ultima rulare valida si publicata observata: `20260817T115844Z`, cu date pana la
`2025-05-25`, 380 de meciuri si 20 de echipe.

## 1. Ce face partea de Data Science

Proiectul foloseste date de fotbal din API-Football pentru La Liga, sezonul 2024, apoi construieste un pipeline ML de **clustering nesupervizat** la nivel de echipa.

Obiectivul modelului:

```text
sa grupeze echipele in arhetipuri statistice similare
```

Clustering-ul nu este un model de predictie directa a scorului. Este un model de segmentare care raspunde la intrebari de tip:

- Ce echipe seamana statistic intre ele?
- Care sunt echipele cu profil ofensiv dominant?
- Care echipe sunt defensive/control-oriented?
- Exista outlieri?
- Cat de stabile sunt clusterele?
- Ce feature-uri explica diferentierea?

Rezultatul curent:

| Element | Valoare |
|---|---|
| Model ales | `AgglomerativeClustering` |
| Candidat | `AgglomerativeClustering#2` |
| Parametri | `n_clusters=4`, `linkage=ward` |
| Unitati clusterizate | 20 echipe |
| Feature-uri folosite | 19 feature-uri numerice |
| Output principal | `output/ml/clusters/clusters.csv` |

Separat de clustering, proiectul are un baseline Poisson pentru forecast 1/X/2
si goluri. Este necalibrat si este auditat walk-forward; nu trebuie confundat cu
modelul de clustering sau prezentat ca recomandare de pariuri.

## 2. Prompturi si comenzi folosite in proiect

In acest proiect exista doua categorii de "prompturi":

1. **Comenzi/prompturi de rulare**: ce comenzi ruleaza pipeline-ul de data science.
2. **Prompturi conversationale pentru agent**: intrebari date agentului AI care explica rezultatele ML.

### 2.1 Prompturi de rulare pentru pipeline

#### Prompt 1 - construirea feature-urilor

```powershell
py run_features.py
```

Ce face:

- ruleaza `data_engineering/transform_data_local.py`;
- transforma JSON-urile raw in tabele base `dim`/`fact`;
- ruleaza `data_engineering/derive_tables_local.py`;
- produce tabelele din `output/derived`;
- produce `fact_match_features.csv`, care este input-ul principal pentru ML.

De ce este folosit:

- separa partea de data engineering de modelare;
- permite verificarea datelor inainte de model;
- creeaza features temporale fara leakage.

Artefacte importante:

```text
output/tables/*.csv
output/derived/fact_match_features.csv
output/derived/fact_team_form.csv
output/derived/fact_match_long.csv
```

#### Prompt 2 - rularea modelului ML

```powershell
py run_ml.py
```

Cod executat:

```python
from ml.clustering.team_clustering_pipeline import main

if __name__ == "__main__":
    main("ml/configs/team_clustering_config.yaml")
```

Ce face:

- citeste config-ul ML;
- incarca `output/derived/fact_match_features.csv`;
- construieste agregari la nivel de echipa;
- ruleaza mai multi algoritmi de clustering;
- evalueaza fiecare candidat;
- selecteaza modelul cu scorul cel mai bun;
- genereaza clustere, metrici, raport si, daca dependintele sunt disponibile, ploturi;
- creeaza un run imutabil in `output/ml/runs/<run_id>/`;
- finalizeaza manifestul si modifica `latest_run.json` numai daca artefactele obligatorii sunt valide;
- produce documente JSONL pentru istoricul Agent Search.

Artefacte:

```text
output/ml/runs/<run_id>/manifest.json
output/ml/runs/<run_id>/clusters/clusters.csv
output/ml/runs/<run_id>/metrics/*.json
output/ml/runs/<run_id>/report/cluster_report.md
output/ml/runs/<run_id>/plots/*.png
output/ml/runs/<run_id>/search/agent_search_documents.jsonl
output/ml/latest_run.json
```

Caile flat `output/ml/clusters`, `metrics`, `report` si `plots` raman copii
compatibile pentru codul vechi; sursa canonica este rularea indicata de pointer.
Rularea `20260817T115844Z` nu are ploturi versionate, deoarece manifestul retine
warning-ul `No module named 'matplotlib'`; un plot flat poate fi ramas de la un
run anterior.

#### Prompt 3 - rularea directa a pipeline-ului ML

```powershell
py -m ml.clustering.team_clustering_pipeline --config ml/configs/team_clustering_config.yaml
```

De ce este util:

- permite testarea altui config YAML;
- este modul explicit si reproductibil de rulare;
- e bun pentru prezentare tehnica deoarece arata unde se afla configuratia.

#### Prompt 4 - regenerarea raportului

```powershell
py run_reporting.py
```

Ce face:

- regenereaza raportul Markdown din artefactele existente;
- nu reantreneaza modelul;
- este util cand vrem doar un raport actualizat dupa modificari de interpretare.

#### Prompt 5 - pipeline complet

```powershell
py data_engineering/run_all_local.py
```

Ce face end-to-end:

1. extractie API-Football;
2. upload raw in GCS;
3. transformare JSON in CSV;
4. derivare tabele analitice;
5. rulare ML clustering;
6. upload output in GCS;
7. import optional al documentelor in Agent Search si incarcare CSV-uri in BigQuery.

Este promptul complet pentru demo operational, dar pentru discutia cu un data scientist este mai clar sa rulam separat `run_features.py` si `run_ml.py`.

### 2.2 Promptul de sistem al agentului AI

Agentul este definit in:

```text
football-cluster-agent/football_agent/agent.py
```

Modelul LLM folosit:

```python
model=os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.5-flash")
```

Promptul/instructiunile agentului spun, pe scurt:

- foloseste GCS si BigQuery ca sursa de adevar;
- nu inventa valori lipsa;
- explica rezultatele in limbaj usor de inteles;
- explica clustere, PCA scatter, heatmap, cluster sizes si metrici;
- compara echipe folosind artefactele disponibile;
- ramane in aria football analytics;
- pentru predictii, raspunde probabilistic, cu confidence si incertitudine;
- separa lentila personala de clusterizarea oficiala;
- nu numeste indexurile de similaritate sau fit „probabilitati”;
- marcheaza forecast-ul Poisson ca necalibrat;
- foloseste ultimul snapshot stocat si nu pretinde date live;
- nu inventeaza URL-uri video si nu atribuie cauzalitate tranzitiilor de cluster;
- pentru SQL, permite doar `SELECT`;
- blocheaza DML/DDL si dataset-uri publice.

Instrumente folosite de agent:

| Tool | Rol |
|---|---|
| `list_latest_clustering_run()` | Gaseste cea mai recenta rulare de clustering din GCS. |
| `list_output_artifact_timestamps()` | Listeaza timestamp-uri pentru artefactele ML. |
| `read_latest_cluster_interpretation()` | Citeste interpretarile clusterelor. |
| `read_latest_metrics()` | Citeste metricile tuturor candidatilor. |
| `read_latest_best_model_summary()` | Citeste modelul castigator. |
| `read_latest_cluster_report()` | Citeste raportul Markdown. |
| `explain_cluster(cluster_id)` | Explica un cluster. |
| `compare_teams(team_a, team_b)` | Compara doua echipe. |
| `query_bigquery(sql)` | Ruleaza query-uri SELECT controlate. |
| `get_latest_available_matches(limit)` | Verifica data maxima si cele mai noi meciuri din dataset. |
| `personalized_team_similarity(...)` | Repondereaza comparatia dupa preferinte, fara sa schimbe clusterul oficial. |
| `predict_match_from_stats(home_team, away_team)` | Baseline Poisson necalibrat pentru 1/X/2 si goluri. |
| `generate_opponent_dossier(...)` | Combina forma, stilul, H2H, jucatorii si forecast-ul. |
| `audit_match_forecast_quality(max_evaluated)` | Backtest walk-forward cu Brier, log-loss, accuracy si calibrare. |
| `list_business_alerts()` | Semnaleaza date stale, forma, volatilitate si miscari intre clustere. |
| `recommend_players_for_team(...)` | Ranking transparent player-team fit. |
| `simulate_match_absences(...)` | Scenariu euristic de absente, fara pretentie cauzala. |
| `create_fan_briefing(...)` | Briefing personalizat pentru cluburile urmarite. |
| `get_match_companion(fixture_id)` | Timeline/statistici din cel mai nou snapshot stocat. |
| `get_video_evidence(fixture_id)` | Returneaza numai clipuri licentiate deja indexate. |
| `search_historical_analytics(...)` | Cauta semantic in rularile ML imutabile prin Agent Search. |

### 2.3 Prompturi demo pentru agent

Aceste prompturi sunt utile in prezentare:

#### Prompt agent 1

```text
Explain the latest clustering run.
```

Ce ar trebui sa raspunda:

- modelul ales;
- parametrii;
- scorurile principale;
- cate clustere au rezultat;
- ce artefacte au fost citite.

#### Prompt agent 2

```text
Explain cluster 1 in simple terms.
```

Ce demonstreaza:

- interpretare de cluster;
- echipe in cluster;
- strengths si weaknesses;
- top differentiating features.

#### Prompt agent 3

```text
Compare Barcelona and Real Madrid.
```

Ce demonstreaza:

- citire `clusters.csv`;
- comparare cluster;
- suport din BigQuery, daca este disponibil;
- explicatie business-friendly.

#### Prompt agent 4

```text
Why is Valladolid outlier-like?
```

Ce demonstreaza:

- explicarea unui singleton cluster;
- folosirea diferentelor fata de media globala;
- marcarea incertitudinii.

#### Prompt agent 5

```text
Give me a statistical forecast for Barcelona at home against Real Madrid.
```

Ce demonstreaza:

- apeleaza baseline-ul Poisson separat de clustering;
- returneaza 1/X/2, expected goals si scoruri probabile;
- marcheaza probabilitatile ca necalibrate;
- afiseaza recenta datelor, confidence si informatiile lipsa.

#### Prompt agent 6 - similaritate subiectiva

```text
I like attacking, in-form teams even if they are less consistent. Starting from Barcelona, which other teams fit my taste?
```

Ce demonstreaza:

- transformarea preferintelor intr-un vector bounded intre `-1` si `1`;
- reponderarea atacului, apararii, rezultatelor, formei si consistentei;
- recomandari personale fara modificarea clusterului oficial;
- index relativ de similaritate/fit, nu probabilitate.

#### Prompt agent 7 - dossier si audit predictiv

```text
Build an opponent dossier for Barcelona vs Atletico Madrid, then tell me how reliable the forecast baseline has been historically.
```

Ce demonstreaza:

- folosirea functiei dedicate de dossier, nu asamblarea unor presupuneri de catre LLM;
- audit walk-forward fara folosirea meciurilor viitoare;
- diferenta dintre accuracy, Brier/log-loss si calibrare.

#### Prompt agent 8 - istoric versionat

```text
How did Barcelona's cluster profile change across the stored ML runs?
```

Ce demonstreaza:

- cautarea prin Agent Search in documentele JSONL ale rularilor imutabile;
- citarea run ID-urilor si a datei datelor;
- raspuns explicit daca Agent Search nu este configurat.

## 3. Datele care intra in model

Input-ul ML curent este:

```yaml
input:
  source: local_derived
  path: output/derived/fact_match_features.csv
  table: fact_match_features
```

`fact_match_features.csv` are 380 randuri si 28 coloane.

Coloane importante:

| Coloana | Rol |
|---|---|
| `fixture_id` | Cheia meciului. |
| `date` | Data meciului. |
| `league_id`, `season`, `round` | Context competitie. |
| `home_team_id`, `home_team_name` | Echipa gazda. |
| `away_team_id`, `away_team_name` | Echipa oaspete. |
| `goals_home`, `goals_away` | Rezultatul final. |
| `home_form_*_lastN` | Forma echipei gazda inainte de meci. |
| `away_form_*_lastN` | Forma echipei oaspete inainte de meci. |
| `target_result` | Target potential pentru modele predictive viitoare. |
| `target_total_goals` | Target potential pentru modele de goluri. |

Important: modelul de clustering nu foloseste direct `target_result` sau `target_total_goals`. Aceste coloane sunt pregatite pentru modele predictive viitoare.

## 4. Transformarea din meciuri in echipe

Modelul clusterizeaza echipe, nu meciuri. De aceea, pipeline-ul transforma datele match-level in date team-level.

Fisier:

```text
ml/features/build_team_features.py
```

Pasul 1: fiecare meci este impartit in doua observatii:

- perspectiva echipei gazda;
- perspectiva echipei oaspete.

Cod relevant:

```python
home = pd.DataFrame({
    "team_id": df["home_team_id"],
    "team_name": df["home_team_name"],
    "is_home": 1,
    "goals_for": df.get("goals_home"),
    "goals_against": df.get("goals_away"),
})

away = pd.DataFrame({
    "team_id": df["away_team_id"],
    "team_name": df["away_team_name"],
    "is_home": 0,
    "goals_for": df.get("goals_away"),
    "goals_against": df.get("goals_home"),
})
```

Pasul 2: se calculeaza rezultat si puncte:

```python
long_df["goal_diff"] = long_df["goals_for"] - long_df["goals_against"]
long_df["is_win"] = (long_df["goals_for"] > long_df["goals_against"]).astype(float)
long_df["is_draw"] = (long_df["goals_for"] == long_df["goals_against"]).astype(float)
long_df["is_loss"] = (long_df["goals_for"] < long_df["goals_against"]).astype(float)
long_df["points"] = long_df["is_win"] * 3 + long_df["is_draw"]
```

Pasul 3: se agregheaza pe echipa:

```python
team_df = (
    long_df.groupby(["team_id", "team_name"], dropna=False)
    .agg(
        games=("fixture_id", "nunique"),
        avg_goals_scored=("goals_for", "mean"),
        avg_goals_conceded=("goals_against", "mean"),
        avg_goal_diff=("goal_diff", "mean"),
        std_goal_diff=("goal_diff", "std"),
        win_rate=("is_win", "mean"),
        draw_rate=("is_draw", "mean"),
        loss_rate=("is_loss", "mean"),
        avg_points=("points", "mean"),
        home_ratio=("is_home", "mean"),
    )
    .reset_index()
)
```

## 5. Feature-uri folosite in model

Config-ul contine 19 feature-uri:

```yaml
features:
  - avg_goals_scored
  - avg_goals_conceded
  - avg_goal_diff
  - std_goal_diff
  - win_rate
  - draw_rate
  - loss_rate
  - avg_points
  - form_pts_avg
  - form_gf_avg
  - form_ga_avg
  - form_gd_avg
  - form_wins_avg
  - form_draws_avg
  - form_losses_avg
  - home_ratio
  - attack_strength
  - defense_strength
  - form_score
```

### 5.1 Tabel detaliat de features

| Feature | Formula / sursa | De ce este folosit | Cum este folosit in model |
|---|---|---|---|
| `avg_goals_scored` | media `goals_for` per echipa | masoara puterea ofensiva directa | intra numeric in clustering; diferentiaza echipele ofensive |
| `avg_goals_conceded` | media `goals_against` per echipa | masoara vulnerabilitatea defensiva | valori mici sustin profil defensiv bun |
| `avg_goal_diff` | media `goals_for - goals_against` | combina atac si aparare intr-un singur indicator | separa echipe dominante de echipe slabe |
| `std_goal_diff` | deviatia standard a diferentei de goluri | masoara volatilitatea rezultatelor | identifica echipe high-variance |
| `win_rate` | medie `is_win` | masoara consistenta in victorii | ajuta la separarea elitei |
| `draw_rate` | medie `is_draw` | indica profil de control sau blocaje tactice | folosit si in `defense_strength` |
| `loss_rate` | medie `is_loss` | masoara frecventa infrangerilor | ajuta la detectarea outlierilor negativi |
| `avg_points` | media punctelor per meci | indicator standard de performanta fotbalistica | unul dintre cele mai interpretabile semnale |
| `form_pts_avg` | media `form_pts_lastN` dupa agregare pe echipa | surprinde forma recenta, nu doar sezonul total | diferentiaza echipe cu momentum |
| `form_gf_avg` | media golurilor marcate in rolling window | forma ofensiva recenta | semnal temporal pentru atac |
| `form_ga_avg` | media golurilor primite in rolling window | forma defensiva recenta | separa defensive form vs defensive weakness |
| `form_gd_avg` | media goal-diff in rolling window | forma recenta neta | intra si in `form_score` |
| `form_wins_avg` | media victoriilor in ultimele N meciuri | consistenta recenta a victoriilor | sustine clusterele de performanta ridicata |
| `form_draws_avg` | media egalurilor in ultimele N meciuri | profil conservator sau echilibrat | ajuta la profilarea echipelor de mijloc |
| `form_losses_avg` | media infrangerilor in ultimele N meciuri | forma negativa recenta | important pentru outlieri |
| `home_ratio` | media `is_home` | control contextual home/away | verifica daca agregarea este balansata |
| `attack_strength` | `avg_goals_scored * (1 + win_rate)` | combina output ofensiv cu eficienta in rezultate | amplifica echipele care marcheaza si castiga |
| `defense_strength` | `(1 / (1 + avg_goals_conceded)) * (1 + draw_rate)` | scor compozit pentru control defensiv | valori mari indica echipe care primesc putin si controleaza jocul |
| `form_score` | `form_pts_avg + form_gd_avg + avg_points` | sumar al formei si performantei | foarte important in interpretarea clusterelor curente |

### 5.2 Feature-uri compozite

#### `attack_strength`

Formula:

```python
team_df["attack_strength"] = team_df["avg_goals_scored"] * (1 + team_df["win_rate"])
```

Interpretare:

- daca o echipa marcheaza mult, scorul creste;
- daca si castiga des, scorul este amplificat;
- diferentiaza echipe care marcheaza mult dar nu castiga de echipe cu atac eficient.

Exemplu conceptual:

```text
echipa A: marcheaza mult + castiga des => attack_strength mare
echipa B: marcheaza decent + castiga rar => attack_strength moderat
```

#### `defense_strength`

Formula:

```python
team_df["defense_strength"] = (
    1 / (1 + team_df["avg_goals_conceded"].clip(lower=0))
) * (1 + team_df["draw_rate"])
```

Interpretare:

- cu cat primeste mai putine goluri, cu atat scorul creste;
- `draw_rate` adauga semnal de control si capacitate de a evita infrangerea;
- este util pentru echipe defensive sau pragmatice.

#### `form_score`

Formula:

```python
team_df["form_score"] = (
    team_df["form_pts_avg"].fillna(0)
    + team_df["form_gd_avg"].fillna(0)
    + team_df["avg_points"].fillna(0)
)
```

Interpretare:

- combina forma recenta cu performanta globala;
- este util pentru separarea echipelor aflate in momentum pozitiv sau negativ;
- apare ca top differentiating feature in rezultatul curent.

## 6. Control anti-leakage

Feature-urile de forma sunt construite in:

```text
data_engineering/derive_tables_local.py
```

Cod relevant:

```python
df["form_pts_lastN"] = grp["points"].apply(
    lambda s: s.shift(1).rolling(window, min_periods=1).sum()
)
```

De ce este important:

- `shift(1)` exclude meciul curent;
- rolling window foloseste doar meciuri anterioare;
- feature-urile descriu starea echipei la intrarea in meci, nu dupa meci.

Aceasta este o alegere corecta pentru Data Science, mai ales daca ulterior se construieste un model predictiv pentru:

- `target_result`;
- `target_total_goals`.

## 7. Preprocesare

Config:

```yaml
preprocessing:
  fillna: median
  scaler: robust
  pca: true
  pca_components: 6
```

### 7.1 Imputare valori lipsa

Cod:

```python
out[features] = out[features].fillna(out[features].median())
```

De ce `median`:

- este robusta la outlieri;
- potrivita pentru date sportive unde pot exista extreme;
- previne erori in algoritmii scikit-learn.

### 7.2 Scalare

Cod:

```python
if scaler_type == "robust":
    scaler = RobustScaler()
elif scaler_type == "minmax":
    scaler = MinMaxScaler()
elif scaler_type == "none":
    scaler = None
else:
    scaler = StandardScaler()
```

De ce `RobustScaler`:

- foloseste median si IQR;
- reduce influenta outlierilor;
- este potrivit cand unele echipe sunt mult peste media ligii.

### 7.3 PCA

Cod:

```python
pca = PCA(n_components=max_components, random_state=config.get("random_state", 42))
X_values = pca.fit_transform(X_values)
```

De ce este folosit:

- reduce dimensionalitatea de la 19 feature-uri la maximum 6 componente;
- reduce zgomotul si corelatiile dintre feature-uri;
- ofera coordonate pentru vizualizare (`pca_x`, `pca_y`);
- face clustering-ul mai stabil cand features sunt corelate.

Atentie pentru prezentare:

- PCA imbunatateste compactarea, dar reduce interpretabilitatea directa;
- interpretarea clusterelor este facuta ulterior pe feature-urile originale agregate, nu doar pe componentele PCA.

## 8. Algoritmii testati

Fisier:

```text
ml/clustering/algorithms.py
```

Pipeline-ul ruleaza toti algoritmii activati in YAML.

| Algoritm | De ce este inclus | Cum este folosit |
|---|---|---|
| `KMeans` | baseline clasic pentru clustering compact/sferic | testeaza `n_clusters` 3, 4, 5 |
| `MiniBatchKMeans` | varianta eficienta si scalabila de KMeans | util daca datele cresc |
| `GaussianMixture` | permite apartenenta probabilistica si clustere eliptice | testeaza `n_components` 3, 4, 5 |
| `AgglomerativeClustering` | clustering ierarhic, bun pentru seturi mici | testeaza 3, 4, 5 clustere cu `ward` |
| `SpectralClustering` | detecteaza structuri non-liniare | foloseste nearest neighbors |
| `Birch` | incremental, bun pentru date mai mari | testeaza 3, 4, 5 clustere |
| `DBSCAN` | detecteaza zgomot/outlieri fara numar fix de clustere | testeaza `eps` 0.7 si 0.9 |
| `MeanShift` | gaseste moduri/densitati automat | fara numar fix de clustere |
| `HDBSCAN` | clustering densitate avansat | doar daca libraria este instalata |

Cod relevant:

```python
for algo_name, algo_cfg in algorithms.items():
    if algo_cfg.get("enabled", True) is False:
        continue
    param_sets = _expand_grid(algo_cfg)
    for idx, params in enumerate(param_sets, start=1):
        model, labels = _fit_predict(algo_name, params, X_proc)
```

De ce se face model sweep:

- nu stim apriori ce algoritm separa cel mai bine echipele;
- dataset-ul este mic, deci putem testa mai multe variante;
- comparatia pe metrici reduce alegerea subiectiva.

## 9. Modelul ales si justificarea

Rezultatul curent din `best_model_summary.json`:

| Camp | Valoare |
|---|---|
| Algoritm | `AgglomerativeClustering` |
| Candidat | `AgglomerativeClustering#2` |
| Parametri | `n_clusters=4`, `linkage=ward` |
| Silhouette | `0.4480` |
| Davies-Bouldin | `0.4632` |
| Calinski-Harabasz | `16.3186` |
| Composite | `0.5186` |
| Cluster sizes | `15 / 2 / 1 / 2` |

Top candidati dupa scor compozit:

| Candidat | Algoritm | Clustere | Silhouette | Davies-Bouldin | Calinski-Harabasz | Composite |
|---|---|---:|---:|---:|---:|---:|
| `AgglomerativeClustering#2` | AgglomerativeClustering | 4 | 0.4480 | 0.4632 | 16.3186 | 0.5186 |
| `Birch#2` | Birch | 4 | 0.4480 | 0.4632 | 16.3186 | 0.5186 |
| `KMeans#1` | KMeans | 3 | 0.4708 | 0.5907 | 18.7006 | 0.5173 |
| `AgglomerativeClustering#1` | AgglomerativeClustering | 3 | 0.4722 | 0.5840 | 16.8935 | 0.5169 |
| `Birch#1` | Birch | 3 | 0.4722 | 0.5840 | 16.8935 | 0.5169 |

Explicatia trade-off-ului:

- modelele cu 3 clustere au silhouette usor mai mare;
- modelul cu 4 clustere are Davies-Bouldin mai bun, deci clustere mai putin suprapuse;
- scorul compozit favorizeaza echilibrat separarea, suprapunerea, zgomotul si balansul;
- 4 clustere ofera segmentare mai bogata pentru interpretare fotbalistica.

De ce `AgglomerativeClustering` este rezonabil aici:

- dataset-ul este mic: 20 echipe;
- clustering-ul ierarhic este potrivit pentru observatii putine;
- `ward` minimizeaza varianta intra-cluster;
- rezultatul este usor de explicat ca grupare pe similaritate.

Limitare:

- clusterele sunt dezechilibrate;
- exista un singleton cluster;
- interpretarea clusterelor mici trebuie tratata ca fragila.

## 10. Metricile de evaluare

Fisier:

```text
ml/clustering/evaluation.py
```

### 10.1 Silhouette

Directie buna: mai mare.

Ce masoara:

- cat de apropiat este un punct de propriul cluster;
- cat de departe este de alte clustere.

Interpretare:

```text
0.4480 = separare moderata, rezonabila pentru date sportive agregate
```

### 10.2 Davies-Bouldin

Directie buna: mai mic.

Ce masoara:

- media similaritatii intre fiecare cluster si cel mai apropiat alt cluster;
- valori mici inseamna clustere mai bine separate.

Rezultat curent:

```text
0.4632 = bun comparativ cu ceilalti candidati din sweep
```

### 10.3 Calinski-Harabasz

Directie buna: mai mare.

Ce masoara:

- raport intre dispersia intre clustere si dispersia in interiorul clusterelor.

Rezultat curent:

```text
16.3186
```

### 10.4 Noise ratio

Directie buna: mai mic.

Folosit mai ales pentru DBSCAN/HDBSCAN, unde label `-1` inseamna zgomot.

Rezultat curent:

```text
0.0
```

### 10.5 Balance ratio

Formula:

```text
min(cluster_size) / max(cluster_size)
```

Rezultat curent:

```text
1 / 15 = 0.0667
```

Interpretare:

- balans slab;
- exista un cluster dominant si clustere mici;
- poate fi acceptabil daca scopul este detectia de outlieri, dar trebuie mentionat.

### 10.6 Scor compozit

Cod:

```python
return float(
    0.45 * float(metric["silhouette"])
    + 0.25 * db_component
    + 0.15 * ch_component
    + 0.10 * noise_penalty
    + 0.05 * balance
)
```

Formula:

```text
0.45 * silhouette
+ 0.25 * (1 / (1 + davies_bouldin))
+ 0.15 * (log1p(calinski_harabasz) / 10)
+ 0.10 * (1 - noise_ratio)
+ 0.05 * balance_ratio
```

De ce este folosit:

- nu depinde de o singura metrica;
- penalizeaza clusterele cu zgomot;
- include o penalizare mica pentru dezechilibru;
- ofera selectie automata reproductibila.

## 11. Interpretarea clusterelor

Fisier:

```text
ml/clustering/interpretation.py
```

Interpretarea nu este generata de un LLM. Este generata deterministic din cod, pe baza:

- mediei feature-urilor per cluster;
- mediei globale;
- diferentei cluster vs global;
- marimii clusterului.

Cod conceptual:

```python
means = cluster_df[features].mean(numeric_only=True)
diff = (means - global_means).sort_values(key=lambda s: s.abs(), ascending=False)
top_features = diff.index.tolist()[:6]
```

Profiluri calculate:

```python
profile = {
    "attack": avg_goals_scored_cluster - avg_goals_scored_global,
    "defense": avg_goals_conceded_global - avg_goals_conceded_cluster,
    "results": win_rate_cluster - win_rate_global,
    "control": draw_rate_cluster - draw_rate_global,
    "volatility": std_goal_diff_cluster - std_goal_diff_global,
}
```

Reguli de narativ:

- cluster de 1 echipa => `Outlier underperforming team`;
- cluster de 2 echipe cu atac si rezultate peste medie => `Elite attacking giants`;
- cluster de 2 echipe cu aparare peste medie => `Defensive elite contenders`;
- cluster mare cu profil intermediar => `Competitive mixed-profile teams`;
- profil volatil => `High-variance transition teams`.

Confidence score:

```python
separation = diff.abs().head(5).mean()
size_factor = min(1.0, cluster_size / 6.0)
raw = 0.45 + 0.35 * tanh(2.0 * separation) + 0.20 * size_factor
```

Interpretare:

- creste daca top feature-urile separa clar clusterul;
- creste daca clusterul are mai multe echipe;
- scade ca incredere practica pentru singleton/pairs, chiar daca scorul numeric poate fi decent.

## 12. Rezultatele curente pe clustere

| Cluster | Label | Echipe | Observatie |
|---:|---|---|---|
| 0 | `Competitive mixed-profile teams` | Alaves, Celta Vigo, Espanyol, Getafe, Girona, Las Palmas, Leganes, Mallorca, Osasuna, Rayo Vallecano, Real Betis, Real Sociedad, Sevilla, Valencia, Villarreal | cluster mare, profil mixt |
| 1 | `Elite attacking giants` | Barcelona, Real Madrid | atac si rezultate peste medie |
| 2 | `Outlier underperforming team` | Valladolid | singleton, outlier |
| 3 | `Defensive elite contenders` | Athletic Club, Atletico Madrid | profil defensiv/control |

Observatie importanta pentru Data Scientist:

```text
clusterul 0 contine 15 din 20 echipe, deci segmentarea este utila mai ales pentru separarea elitei/outlierilor,
nu pentru o taxonomie perfect balansata a tuturor echipelor.
```

## 13. Artefacte produse si cum se folosesc

| Artefact | Continut | Cum se foloseste |
|---|---|---|
| `clusters.csv` | echipa, cluster, PCA, label, strengths, weaknesses | dashboard, agent, analiza echipe |
| `metrics.json` | metrici pentru toti candidatii | audit model selection |
| `best_model_summary.json` | modelul castigator si metricile lui | prezentare si reproducibilitate |
| `cluster_interpretation.json` | interpretari structurate | agent AI si rapoarte |
| `cluster_report.md` | raport citibil | livrabil business |
| `pca_scatter.png` (optional) | distributie 2D a echipelor | vizualizare separare |
| `cluster_feature_heatmap.png` (optional) | heatmap feature-uri | explicabilitate |
| `cluster_radar.png` (optional) | profil radar clustere | comparatie vizuala |
| `cluster_top_features.png` (optional) | top feature-uri diferentiatoare | interpretabilitate |
| `manifest.json` | status, hash-uri, input, config, model, artifact inventory | lineage si audit |
| `agent_search_documents.jsonl` | documente despre run/model/clustere/echipe | cautare istorica semantica |

Artefactele canonice sunt sub `output/ml/runs/<run_id>/`. Pointerul
`output/ml/latest_run.json` avanseaza numai la o rulare `SUCCEEDED`; o rulare
esuată nu suprascrie ultima versiune valida. Caile flat raman fallback de
compatibilitate.

`ml/stability/` poate produce ARI intre rulari, assignment strength, bootstrap,
tranzitii si rapoarte de stabilitate. Configuratia curenta are insa
`stability.enabled: false`, deci aceste rezultate nu trebuie pretinse pentru
rularea curenta. Assignment strength si bootstrap stability sunt scoruri de
consistenta, nu probabilitati de corectitudine.

Coloane importante in `clusters.csv`:

| Coloana | Rol |
|---|---|
| `team_id`, `team_name` | identificare echipa |
| `cluster` | label numeric cluster |
| `best_algorithm`, `candidate_id` | lineage model |
| `pca_x`, `pca_y` | coordonate vizualizare |
| `cluster_label` | nume business-friendly |
| `cluster_description` | descriere scurta |
| `strengths`, `weaknesses` | interpretare fotbalistica |
| `is_outlier_like` | flag pentru clustere mici/outlier |
| `cluster_warning` | avertisment interpretare |
| `cluster_confidence_score` | scor explicabilitate cluster |
| `feat_1_name` ... `feat_5_name` | top feature-uri |
| `feat_1_value` ... `feat_5_value` | magnitudinea diferentei |

## 14. De ce clustering si nu supervised learning

Clustering este potrivit aici deoarece:

- scopul este segmentarea echipelor, nu predictia unui target;
- avem doar un sezon si 20 echipe, deci supervised learning la nivel de echipa ar fi fragil;
- feature-urile sunt bune pentru descoperirea profilurilor;
- rezultatul poate fi explicat ca arhetipuri pentru analiza sportiva.

Pentru supervised learning ar fi nevoie de:

- mai multe sezoane;
- train/test temporal;
- target clar: `target_result`, `target_total_goals`, over/under, BTTS etc.;
- baseline-uri predictive;
- backtesting;
- calibrare probabilistica.

## 15. Intrebari probabile de la un Data Scientist si raspunsuri

### De ce folosim `RobustScaler`?

Pentru ca datele sportive au outlieri naturali: Barcelona si Real Madrid pot avea valori mult peste restul ligii. `RobustScaler` reduce influenta acestor extreme, folosind median si IQR.

### Exista data leakage?

Pentru feature-urile de forma, riscul este redus deoarece se foloseste `shift(1)` inainte de rolling window. Astfel, forma pentru un meci este calculata doar din meciuri anterioare.

### De ce folosim PCA daca vrem interpretabilitate?

PCA este folosit pentru matricea de clustering si vizualizare. Interpretarea clusterelor este facuta ulterior pe feature-urile originale, comparand mediile clusterelor cu media globala.

### De ce 4 clustere daca 3 clustere are silhouette usor mai mare?

Selectia se face dupa scor compozit, nu doar silhouette. Modelul cu 4 clustere are Davies-Bouldin mai bun si ofera segmentare mai granulara: mixed teams, attacking giants, defensive contenders si outlier.

### De ce exista cluster singleton?

Pentru ca Valladolid este suficient de diferit de restul echipelor in feature space. Totusi, singleton-ul trebuie tratat ca outlier, nu ca segment stabil.

### Sunt feature-urile independente?

Nu complet. Unele sunt corelate, de exemplu `avg_points`, `win_rate`, `avg_goal_diff` si `form_score`. PCA ajuta la reducerea redundantei. Pentru o versiune viitoare se poate face feature selection sau correlation pruning.

### Modelul prezice rezultate?

Modelul de clustering nu. El grupeaza echipe si produce arhetipuri. Proiectul are
separat un baseline Poisson care produce estimari 1/X/2 si goluri, dar acestea
sunt necalibrate, nu provin din cluster ca probabilitate si nu sunt garantii.
Calitatea baseline-ului se verifica prin backtest walk-forward cu accuracy,
Brier score, log-loss si sumar de calibrare.

## 16. Ce as imbunatati pentru o versiune Data Science mai matura

Prioritate mare:

- validare statistica pe mai multe sezoane;
- calibrare temporala si comparatie a baseline-ului Poisson cu modele supervised;
- feature selection pentru reducerea coliniaritatii;
- raport separat de distributii feature;
- activarea controlata a modulului de stabilitate deja implementat;
- validarea pragurilor pentru ARI, assignment strength, bootstrap si tranzitii.

Prioritate medie:

- comparatie cu metode fara PCA;
- comparatie cu `StandardScaler` si `MinMaxScaler`;
- analiza SHAP-like nu se aplica direct la clustering, dar se poate face interpretare prin diferenta fata de media globala;
- adaugare expected goals daca API-ul permite;
- adaugare features de posesie, suturi, cartonase, pressing proxy.

Pentru supervised learning viitor:

- target: `target_result`;
- target alternativ: `target_total_goals`;
- split temporal pe runde;
- baseline: major class / Poisson goals / logistic regression;
- modele: Logistic Regression, RandomForest, XGBoost/LightGBM;
- metrici: accuracy, log loss, Brier score, calibration curve.

Engineering/MLOps ramas:

- contracte de schema si teste directe pentru parsing, deduplicare si non-leakage;
- CI de PR pentru pytest, lint si data contracts;
- profiluri personale persistente numai opt-in;
- mai multe ligi/sezoane si snapshot-uri comparabile;
- date licentiate pentru lineups, accidentari, video sau tracking.

## 17. Mini-script verbal pentru prezentare

Poti explica proiectul asa:

```text
Am construit un pipeline Data Science pentru football analytics. Datele vin din API-Football,
sunt salvate raw, apoi transformate in tabele dim/fact si tabele derivate. Pentru ML folosesc
fact_match_features, care include forma echipelor calculata fara leakage, prin shift(1) si rolling
window pe ultimele 5 meciuri.

Modelul curent este de clustering nesupervizat la nivel de echipa. Transform fiecare meci in doua
perspective, home si away, apoi agreg pe echipa feature-uri de atac, aparare, rezultate si forma.
Rulez un sweep de algoritmi: KMeans, GaussianMixture, Agglomerative, Birch, DBSCAN si altii.
Selectia se face automat printr-un scor compozit care combina silhouette, Davies-Bouldin,
Calinski-Harabasz, noise ratio si balance ratio.

Modelul ales este AgglomerativeClustering cu 4 clustere. Rezultatul separa majoritatea echipelor
intr-un cluster mixt, Barcelona si Real Madrid ca elite attacking giants, Valladolid ca outlier si
Athletic Club plus Atletico Madrid ca defensive elite contenders. Interpretarea este deterministica,
prin compararea mediilor clusterelor cu media globala a feature-urilor.

Artefactele sunt salvate in CSV/JSON/Markdown si pot fi consumate de BigQuery, GCS si un agent
Vertex AI ADK/Gemini care explica rezultatele conversational, fara sa modifice datele.

Fiecare executie are un run ID si un manifest imutabil. Pointerul latest avanseaza doar dupa succes,
iar documentele run-ului pot fi indexate in Agent Search pentru comparatii istorice. Peste acest strat
obiectiv, agentul poate aplica preferintele utilizatorului ca ponderi pentru o similaritate personala,
fara sa schimbe clusterul oficial. Predictia este o componenta separata: un baseline Poisson necalibrat,
auditat walk-forward si prezentat cu incertitudine.
```

## 18. Concluzie tehnica

Partea de Data Science este construita corect pentru un prim pipeline de clustering:

- are feature engineering temporal;
- evita leakage in forma recenta;
- testeaza mai multi algoritmi;
- selecteaza modelul pe metrici;
- produce interpretari si artefacte;
- versioneaza failure-safe fiecare rulare;
- are teste pentru lifecycle, personalizare, forecast si functiile business;
- poate fi explicata si auditata.

Principala limitare este dimensiunea mica a dataset-ului la nivel de echipa: 20
observatii dintr-un singur sezon. De aceea, clustering-ul trebuie prezentat ca
segmentare exploratorie si explicabila. Stabilitatea este implementata dar
dezactivata implicit, iar forecast-ul separat ramane un baseline necalibrat pana
la validare multi-sezon si calibrare temporala.
