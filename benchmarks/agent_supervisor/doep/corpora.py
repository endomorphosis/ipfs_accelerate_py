"""DOEP-113/114/115 corpus loaders.

Fixtures stay offline. Loaders never open DuckDB, never use the network,
and never complete board tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class CorpusError(ValueError):
    """Closed corpus-contract violation."""


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "test"
    / "fixtures"
    / "agent_supervisor_doep"
)


def _load(path: Path, schema: str, key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise CorpusError(f"{path.name} schema must be {schema}")
    rows = payload.get(key)
    if not isinstance(rows, list) or not rows:
        raise CorpusError(f"{path.name} is empty")
    for row in rows:
        if not isinstance(row, Mapping):
            raise CorpusError(f"{path.name} rows must be objects")
        if row.get("network") is True:
            raise CorpusError(f"{path.name} cannot require network")
        if row.get("secrets") is True:
            raise CorpusError(f"{path.name} cannot embed secrets")
    return payload


def load_hermetic_objectives(path: str | Path | None = None) -> dict[str, Any]:
    payload = _load(
        Path(path) if path is not None else FIXTURE_ROOT / "hermetic_objectives.json",
        "doep-hermetic-objectives@1",
        "objectives",
    )
    for objective in payload["objectives"]:
        if objective.get("hermetic") is not True:
            raise CorpusError("hermetic objective must be hermetic")
        if "objective_id" not in objective:
            raise CorpusError("hermetic objective_id is required")
    return {
        **payload,
        "n": len(payload["objectives"]),
        "objective_ids": [item["objective_id"] for item in payload["objectives"]],
        "completion_authority": False,
    }


def load_historical_replays(path: str | Path | None = None) -> dict[str, Any]:
    payload = _load(
        Path(path) if path is not None else FIXTURE_ROOT / "historical_replays.json",
        "doep-historical-replays@1",
        "replays",
    )
    for replay in payload["replays"]:
        if replay.get("mutates_live_store") is True:
            raise CorpusError("historical replay cannot mutate the live store")
        if "replay_id" not in replay or "generation" not in replay:
            raise CorpusError("historical replay_id and generation are required")
    return {
        **payload,
        "n": len(payload["replays"]),
        "replay_ids": [item["replay_id"] for item in payload["replays"]],
        "completion_authority": False,
        "mutates_live_store": False,
    }


def load_held_out_objectives(
    path: str | Path | None = None,
    *,
    hermetic_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = _load(
        Path(path) if path is not None else FIXTURE_ROOT / "held_out_objectives.json",
        "doep-held-out-objectives@1",
        "objectives",
    )
    hermetic = load_hermetic_objectives(hermetic_path)
    held_ids = {item["objective_id"] for item in payload["objectives"]}
    hermetic_ids = set(hermetic["objective_ids"])
    leaked = sorted(held_ids & hermetic_ids)
    if leaked:
        raise CorpusError(f"held-out corpus leaked hermetic ids {leaked}")
    return {
        **payload,
        "n": len(payload["objectives"]),
        "objective_ids": [item["objective_id"] for item in payload["objectives"]],
        "held_out": True,
        "leaked_from_hermetic": False,
        "completion_authority": False,
    }
