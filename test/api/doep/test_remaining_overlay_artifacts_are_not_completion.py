"""Remaining DOEP overlay artifacts must not claim DuckDB completion."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
)
IDS = [
    "DOEP-044",
    "DOEP-046",
    "DOEP-063",
    "DOEP-064",
    "DOEP-065",
    "DOEP-066",
    "DOEP-092",
    "DOEP-093",
    "DOEP-094",
    "DOEP-104",
    *[f"DOEP-{i}" for i in list(range(110, 118)) + list(range(120, 127))],
]


def _denies_completion(payload: dict) -> bool:
    if payload.get("cas_completed") is True:
        return False
    authority = payload.get("completion_authority")
    authoritative = payload.get("completion_authoritative")
    if authority is True or authoritative is True:
        return False
    return authority is False or authoritative is False


def test_remaining_doep_artifacts_deny_completion_authority() -> None:
    for tid in IDS:
        out = json.loads((ROOT / "outputs" / f"{tid}.json").read_text(encoding="utf-8"))
        rec = json.loads((ROOT / "receipts" / f"{tid}.json").read_text(encoding="utf-8"))
        assert _denies_completion(out), tid
        assert _denies_completion(rec), tid
        assert out.get("cas_completed") is not True
