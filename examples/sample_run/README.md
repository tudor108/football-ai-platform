# Public sample run

This directory contains a deliberately small, redacted example of an ML run
for repository visitors. It is not the source of truth for production
artifacts and it is not used by the pipeline.

The complete immutable run is published to GCS under:

```text
gs://football-ai-raw-data-tudor/output/ml/runs/20260817T115844Z/
```

The full history stays in GCS so that GitHub contains source code and a small
executable-looking example instead of hundreds of generated JSON/CSV files.
See `output/ml/runs/<run_id>/manifest.json` in the cloud artifact for the
provenance, feature-schema hash, preprocessing hash and selected model.
