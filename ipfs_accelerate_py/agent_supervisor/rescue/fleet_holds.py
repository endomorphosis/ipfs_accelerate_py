"""Re-evaluate operator stop markers. Never forge source admission or completion."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from .fleet_watchdog import repair_hold_paths, write_json

ASEH_SOURCE_HOLD_SCHEMA = "aseh/root-active-source-repair-hold@1"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _native_owner_live(observation: Mapping[str, Any]) -> bool:
    details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
    return bool(
        observation.get("health") in {"healthy", "complete"}
        and details.get("owner_ready") is True
        and details.get("authenticated_task_observation") is True
    )


def _operation_consumed(payload: Mapping[str, Any]) -> bool:
    operation = payload.get("operation")
    if not isinstance(operation, str) or not operation:
        return False
    directory = Path(operation)
    if not directory.is_dir():
        return False
    return any(path.name.endswith("source_operation_consumed.json") for path in directory.glob("*.json"))


def _archive_unverified_source_hold(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the unverified admission record; drop only the stop-marker name."""
    archive = path.with_name("HOLD.unverified-source-admission.json")
    record = dict(payload)
    record["source_admission_verified"] = False
    record["callback_settlement_authority"] = False
    record["stop_marker_released_at"] = time.time()
    record["stop_marker_release"] = "native_owner_live_operation_consumed"
    record["stop_marker_release_schema"] = "ipfs_accelerate_py/agent-supervisor/hold-review@1"
    tmp = archive.with_name(archive.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(archive)
    path.unlink()
    return {"status": "released", "path": str(path), "archive": str(archive),
            "source_admission_verified": False}


def review_one_hold(path: str, board: Mapping[str, Any],
                    observation: Mapping[str, Any]) -> dict[str, Any]:
    marker = Path(path)
    payload = _read_json(marker)
    if payload and payload.get("schema") == ASEH_SOURCE_HOLD_SCHEMA:
        if payload.get("source_admission_verified") is True:
            return {"status": "retained", "path": path,
                    "reason": "verified_hold_requires_operator_release"}
        if _native_owner_live(observation) and _operation_consumed(payload):
            return _archive_unverified_source_hold(marker, payload)
        return {"status": "retained", "path": path,
                "reason": "source_admission_unverified"}
    return {"status": "retained", "path": path, "reason": "unstructured_or_unknown_hold"}


def review_board_holds(board: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    released, retained = [], []
    for path in repair_hold_paths(dict(board)):
        result = review_one_hold(path, board, observation)
        (released if result.get("status") == "released" else retained).append(result)
    return {
        "status": "applied" if released else "wait",
        "recipe": "hold_review",
        "released": released,
        "retained": retained,
    }
