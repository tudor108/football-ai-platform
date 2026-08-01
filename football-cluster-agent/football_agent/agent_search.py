"""Optional semantic search over immutable historical ML-run documents."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any


def _escape_filter(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _plain(value: Any) -> Any:
    """Convert proto-plus map/list wrappers into JSON-safe Python values."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value


def search_historical_analytics(
    query: str,
    team_name: str = "",
    run_id: str = "",
    max_results: int = 5,
) -> dict[str, Any]:
    """Semantically search indexed historical cluster runs.

    This tool is optional. It returns a configuration message instead of
    failing agent startup when no Agent Search engine has been provisioned.
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    engine_id = os.getenv("AGENT_SEARCH_ENGINE_ID", "").strip()
    location = os.getenv("AGENT_SEARCH_LOCATION", "global").strip() or "global"
    if not engine_id:
        return {
            "status": "not_configured",
            "message": "Set AGENT_SEARCH_ENGINE_ID after provisioning Agent Search.",
        }
    if not query.strip():
        raise ValueError("A non-empty historical search query is required.")

    try:
        from google.api_core.client_options import ClientOptions
        from google.cloud import discoveryengine_v1 as discoveryengine
    except ImportError as error:
        raise RuntimeError(
            "Historical search requires google-cloud-discoveryengine."
        ) from error

    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )
    client = discoveryengine.SearchServiceClient(client_options=client_options)
    serving_config = (
        f"projects/{project_id}/locations/{location}/collections/default_collection/"
        f"engines/{engine_id}/servingConfigs/default_serving_config"
    )
    filters: list[str] = []
    if team_name.strip():
        filters.append(f'team_name: ANY("{_escape_filter(team_name.strip())}")')
    if run_id.strip():
        filters.append(f'run_id: ANY("{_escape_filter(run_id.strip())}")')
    result_limit = max(1, min(int(max_results), 10))
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query.strip(),
        filter=" AND ".join(filters),
        page_size=result_limit,
        content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True,
            ),
        ),
    )
    response = client.search(request=request)
    results: list[dict[str, Any]] = []
    for index, item in enumerate(response):
        if index >= result_limit:
            break
        document = item.document
        structured = dict(document.struct_data) if document.struct_data else {}
        derived = dict(document.derived_struct_data) if document.derived_struct_data else {}
        results.append(
            {
                "id": document.id,
                "title": structured.get("title"),
                "content": structured.get("content"),
                "run_id": structured.get("run_id"),
                "team_name": structured.get("team_name"),
                "source_artifact": structured.get("source_artifact"),
                "snippets": _plain(derived.get("snippets", [])),
            }
        )
    return {
        "status": "ok",
        "query": query.strip(),
        "filters": {"team_name": team_name or None, "run_id": run_id or None},
        "result_count": len(results),
        "results": results,
        "grounding_note": "Results come from indexed immutable ML-run documents.",
    }
