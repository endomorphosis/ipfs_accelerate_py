"""DOEP-123 live shadow campaign."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.agent_supervisor.doep.paired import run_live_shadow_campaign


RESULT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "benchmarks"
    / "live_shadow.json"
)


def test_live_shadow_campaign_does_not_write_or_complete() -> None:
    snapshot = json.loads(RESULT.read_text(encoding="utf-8"))
    payload = run_live_shadow_campaign()
    assert payload["influences_live"] is snapshot["influences_live"] is False
    assert payload["completion_authority"] is False
    assert payload["duckdb_written"] is False
    assert payload["cases"]
