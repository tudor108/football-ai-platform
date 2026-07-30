# Project Agent Rules

- Preserve `run_extract.py`, `run_features.py`, `run_ml.py`, `run_reporting.py`, and `data_engineering/run_all_local.py` as working entrypoints.
- Do not perform destructive filesystem or cloud operations.
- Historical successful ML runs are immutable.
- `latest_run.json` may point only to a completed valid run.
- Do not call an uncalibrated distance score a probability.
- Do not interpret ARI, bootstrap stability, or cluster movement as causality.
- Do not compare incompatible feature schemas or preprocessing configurations without marking the comparison as method-changed.
- Add tests for every new business rule.
- Run focused tests before the full suite.
- Keep cloud integration optional and mockable.
- Preserve temporary backward compatibility with the flat output paths.
