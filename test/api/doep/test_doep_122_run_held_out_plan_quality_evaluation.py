"""DOEP-122 held-out plan-quality evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.agent_supervisor.doep.paired import run_held_out_plan_quality


RESULT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "benchmarks"
    / "held_out_plan_quality.json"
)


def test_held_out_evaluation_does_not_leak_or_complete() -> None:
    snapshot = json.loads(RESULT.read_text(encoding="utf-8"))
    payload = run_held_out_plan_quality()
    assert payload["held_out"] is snapshot["held_out"] is True
    assert payload["leaked_from_hermetic"] is False
    assert payload["completion_authority"] is False
    assert payload["scores"]
    assert all(item["admitted"] is False for item in payload["scores"])
