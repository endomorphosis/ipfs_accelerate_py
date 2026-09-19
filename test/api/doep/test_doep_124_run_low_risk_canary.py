"""DOEP-124 run low-risk canary."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.agent_supervisor.doep.low_risk_canary import run_low_risk_canary


RESULT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "benchmarks"
    / "low_risk_canary.json"
)


def test_canary_run_stays_low_risk_and_non_completing() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["risk_class"] == "R4"
    assert payload["completion_authority"] is False
    assert payload["duckdb_written"] is False
    live = run_low_risk_canary([{"case_id": "c1", "risk_class": "R4"}])
    assert live["completion_authority"] is False
