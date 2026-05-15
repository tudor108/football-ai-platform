import base64
import json
import os

import google.auth
from google.auth.transport.requests import AuthorizedSession


PROJECT_ID = os.environ.get("PROJECT_ID", "")
TARGET_REGION = os.environ.get("TARGET_REGION", "us-central1")
TARGET_SERVICE = os.environ.get("TARGET_SERVICE", "football-agent-v2")
SHUTDOWN_AT_USD = float(os.environ.get("SHUTDOWN_AT_USD", "40"))


def _decode_pubsub(event: dict) -> dict:
    payload = event.get("data", "")
    if not payload:
        return {}
    raw = base64.b64decode(payload).decode("utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _extract_cost_usd(msg: dict) -> float:
    amount = (
        msg.get("costAmount")
        or msg.get("cost_amount")
        or msg.get("cost", {}).get("amount")
        or 0
    )
    try:
        return float(amount)
    except (TypeError, ValueError):
        return 0.0


def _set_max_instances_zero() -> None:
    if not PROJECT_ID:
        raise ValueError("Missing PROJECT_ID env var.")

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    session = AuthorizedSession(creds)

    service_path = (
        f"projects/{PROJECT_ID}/locations/{TARGET_REGION}/services/{TARGET_SERVICE}"
    )
    url = f"https://run.googleapis.com/v2/{service_path}?updateMask=scaling.maxInstanceCount"
    body = {"scaling": {"maxInstanceCount": 0}}
    resp = session.patch(url, json=body, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to update service scaling: {resp.status_code} {resp.text}")


def budget_guard(event, context):  # pylint: disable=unused-argument
    msg = _decode_pubsub(event)
    cost = _extract_cost_usd(msg)
    if cost < SHUTDOWN_AT_USD:
        print(
            f"No action. Current cost ${cost:.2f} is below threshold ${SHUTDOWN_AT_USD:.2f}."
        )
        return

    _set_max_instances_zero()
    print(
        f"Threshold reached (${cost:.2f}). Set max instances to 0 for {TARGET_SERVICE}."
    )
