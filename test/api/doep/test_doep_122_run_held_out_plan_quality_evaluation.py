"""DOEP-122 held-out plan-quality evaluation."""

from __future__ import annotations

import json
from pathlib import Path


RESULT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "benchmarks"
    / "held_out_plan_quality.json"
)


def test_held_out_evaluation_does_not_leak_or_complete() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["held_out"] is True
    assert payload["leaked_from_hermetic"] is False
    assert payload["completion_authority"] is False
    assert payload["scores"]
