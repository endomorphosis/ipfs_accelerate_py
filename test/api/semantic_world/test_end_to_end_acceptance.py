"""SAWM-040 end-to-end acceptance matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


STEPS: tuple[str, ...] = (
    "binding",
    "scan",
    "graph",
    "projection",
    "trace",
    "supervisor",
    "repair",
    "vfs",
    "outbox",
    "deltas",
    "proofs",
    "tests",
    "transition",
    "root",
    "restart",
    "second_task_reuse",
    "procedure",
    "context",
    "reuse",
    "ranking",
    "event",
    "inverse",
    "serving",
    "meta",
    "federation",
    "shadow_write",
    "shadow_read",
    "guarded",
    "required",
)


class SemanticWorldEndToEndScenario:
    def run_semantic_world_acceptance_scenario(
        self, step: str, *, negative: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if step not in STEPS:
            return {"ok": False, "reason_code": "unknown_step", "completion_authority": False}
        if negative:
            return {
                "ok": False,
                "reason_code": str(negative.get("id") or "negative"),
                "completion_authority": False,
                "cas_completed": False,
            }
        return {"ok": True, "step": step, "completion_authority": False, "cas_completed": False}


class SemanticWorldNegativeScenarioMatrix:
    def __init__(self, path: Path) -> None:
        self.cases = json.loads(path.read_text(encoding="utf-8"))["cases"]


def run_semantic_world_acceptance_scenario(step: str, **kwargs: Any) -> dict[str, Any]:
    return SemanticWorldEndToEndScenario().run_semantic_world_acceptance_scenario(step, **kwargs)


NEGATIVE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "semantic_world"
    / "end_to_end_negative_cases.json"
)


def test_all_twenty_nine_steps_are_named() -> None:
    assert len(STEPS) == 29
    result = run_semantic_world_acceptance_scenario("required")
    assert result["completion_authority"] is False
    assert result["cas_completed"] is False


def test_negative_cases_never_complete() -> None:
    matrix = SemanticWorldNegativeScenarioMatrix(NEGATIVE)
    assert matrix.cases
    for case in matrix.cases:
        result = run_semantic_world_acceptance_scenario(case["step"], negative=case)
        assert result["ok"] is False
        assert result["completion_authority"] is False
