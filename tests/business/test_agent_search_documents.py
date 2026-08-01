from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from business.agent_search_cloud import import_documents_from_gcs
from business.agent_search_documents import build_agent_search_documents, write_agent_search_documents


def test_historical_search_is_optional_without_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    agent_root = Path(__file__).resolve().parents[2] / "football-cluster-agent"
    if str(agent_root) not in sys.path:
        sys.path.insert(0, str(agent_root))
    from football_agent.agent_search import search_historical_analytics

    monkeypatch.delenv("AGENT_SEARCH_ENGINE_ID", raising=False)
    result = search_historical_analytics("Barcelona history")
    assert result["status"] == "not_configured"


def test_build_documents_has_stable_ids_and_safety_language(tmp_path: Path) -> None:
    clusters = pd.DataFrame(
        [{"team_id": 1, "team_name": "FC Álpha", "cluster": 2, "avg_points": 2.1}]
    )
    documents = build_agent_search_documents(
        run_id="20260801T120000Z-abcd",
        clusters=clusters,
        interpretations={2: {"label": "Control", "strengths": ["passing"]}},
        metrics={"silhouette": 0.4},
        best_model={"algorithm": "kmeans", "candidate_id": "k3"},
        run_metadata={"data_as_of_date": "2026-07-31"},
    )
    assert len(documents) == 2
    assert all(1 <= len(document["id"]) <= 63 for document in documents)
    assert all(set(document["id"]) <= set("abcdefghijklmnopqrstuvwxyz0123456789-") for document in documents)
    assert "do not establish causality" in documents[0]["content"]
    assert "official cluster was not changed" in documents[1]["content"]

    output = write_agent_search_documents(tmp_path / "documents.jsonl", documents)
    parsed = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert parsed == documents


def test_agent_search_import_validates_gcs_contract_before_sdk_call() -> None:
    with pytest.raises(ValueError, match="JSONL"):
        import_documents_from_gcs(
            project_id="project",
            data_store_id="store",
            gcs_uri="https://example.com/documents.csv",
        )
