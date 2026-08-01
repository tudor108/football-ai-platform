"""Provision the optional football-history Agent Search data store and app.

Usage:
  python scripts/setup_agent_search.py --project PROJECT_ID

The default is Standard Search. Pass ``--enterprise-generative`` only when the
credit terms and expected query volume have been checked in Billing.
"""

from __future__ import annotations

import argparse

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import NotFound
from google.cloud import discoveryengine_v1 as discoveryengine


def provision(
    project_id: str,
    data_store_id: str,
    engine_id: str,
    location: str = "global",
    enterprise_generative: bool = False,
) -> dict[str, str]:
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )
    data_client = discoveryengine.DataStoreServiceClient(client_options=client_options)
    parent = data_client.collection_path(project_id, location, "default_collection")
    data_store_name = data_client.data_store_path(project_id, location, data_store_id)
    try:
        data_client.get_data_store(name=data_store_name)
        data_store_status = "exists"
    except NotFound:
        operation = data_client.create_data_store(
            request=discoveryengine.CreateDataStoreRequest(
                parent=parent,
                data_store_id=data_store_id,
                data_store=discoveryengine.DataStore(
                    display_name="Football analytics immutable run history",
                    industry_vertical=discoveryengine.IndustryVertical.GENERIC,
                    solution_types=[discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH],
                    # JSONL records are structured documents; the field named
                    # "content" is part of struct_data, not Document.content.
                    content_config=discoveryengine.DataStore.ContentConfig.NO_CONTENT,
                ),
            )
        )
        operation.result()
        data_store_status = "created"

    engine_client = discoveryengine.EngineServiceClient(client_options=client_options)
    engine_name = engine_client.engine_path(
        project_id, location, "default_collection", engine_id
    )
    try:
        engine_client.get_engine(name=engine_name)
        engine_status = "exists"
    except NotFound:
        tier = (
            discoveryengine.SearchTier.SEARCH_TIER_ENTERPRISE
            if enterprise_generative
            else discoveryengine.SearchTier.SEARCH_TIER_STANDARD
        )
        add_ons = (
            [discoveryengine.SearchAddOn.SEARCH_ADD_ON_LLM]
            if enterprise_generative
            else []
        )
        operation = engine_client.create_engine(
            request=discoveryengine.CreateEngineRequest(
                parent=parent,
                engine_id=engine_id,
                engine=discoveryengine.Engine(
                    display_name="Football Analytics History Search",
                    industry_vertical=discoveryengine.IndustryVertical.GENERIC,
                    solution_type=discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH,
                    search_engine_config=discoveryengine.Engine.SearchEngineConfig(
                        search_tier=tier,
                        search_add_ons=add_ons,
                    ),
                    data_store_ids=[data_store_id],
                ),
            )
        )
        operation.result()
        engine_status = "created"
    return {
        "data_store": data_store_status,
        "engine": engine_status,
        "data_store_id": data_store_id,
        "engine_id": engine_id,
        "location": location,
        "tier": "enterprise-generative" if enterprise_generative else "standard",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--data-store-id", default="football-analytics-history")
    parser.add_argument("--engine-id", default="football-analytics-search")
    parser.add_argument("--location", default="global")
    parser.add_argument("--enterprise-generative", action="store_true")
    args = parser.parse_args()
    print(
        provision(
            project_id=args.project,
            data_store_id=args.data_store_id,
            engine_id=args.engine_id,
            location=args.location,
            enterprise_generative=args.enterprise_generative,
        )
    )


if __name__ == "__main__":
    main()
