"""Optional Google Cloud Agent Search publication helpers.

The module deliberately imports the Google SDK lazily so local ML/test runs do
not require Agent Search credentials or its client library.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AgentSearchImportResult:
    status: str
    operation_name: str
    gcs_uri: str
    data_store_id: str
    completed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def import_documents_from_gcs(
    *,
    project_id: str,
    data_store_id: str,
    gcs_uri: str,
    location: str = "global",
    wait: bool = False,
) -> AgentSearchImportResult:
    """Incrementally import custom JSONL documents from one immutable run.

    Every JSON object must contain a valid RFC-1034 ``id`` field. ``wait`` is
    false by default because imports are long-running and should not keep the
    daily data pipeline alive unnecessarily.
    """
    if not project_id.strip() or not data_store_id.strip():
        raise ValueError("project_id and data_store_id are required.")
    if not gcs_uri.startswith("gs://") or not gcs_uri.endswith(".jsonl"):
        raise ValueError("gcs_uri must point to one JSONL object in Cloud Storage.")

    try:
        from google.api_core.client_options import ClientOptions
        from google.cloud import discoveryengine_v1 as discoveryengine
    except ImportError as error:
        raise RuntimeError(
            "Agent Search support requires google-cloud-discoveryengine."
        ) from error

    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )
    client = discoveryengine.DocumentServiceClient(client_options=client_options)
    parent = client.branch_path(
        project=project_id,
        location=location,
        data_store=data_store_id,
        branch="default_branch",
    )
    request = discoveryengine.ImportDocumentsRequest(
        parent=parent,
        gcs_source=discoveryengine.GcsSource(
            input_uris=[gcs_uri],
            data_schema="custom",
        ),
        reconciliation_mode=(
            discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL
        ),
        auto_generate_ids=False,
        id_field="id",
    )
    operation = client.import_documents(request=request)
    if wait:
        operation.result()
    return AgentSearchImportResult(
        status="completed" if wait else "started",
        operation_name=operation.operation.name,
        gcs_uri=gcs_uri,
        data_store_id=data_store_id,
        completed=wait,
    )
