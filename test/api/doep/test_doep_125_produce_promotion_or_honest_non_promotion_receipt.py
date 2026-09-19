"""DOEP-125 promotion or honest non-promotion receipt."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.agent_supervisor.doep.paired import (
    run_held_out_plan_quality,
    run_hermetic_paired_benchmark,
    run_historical_paired_replay,
)
from benchmarks.agent_supervisor.doep.promotion import (
    produce_promotion_or_honest_non_promotion,
)


DECISION = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "release"
    / "promotion_decision.json"
)


def test_honest_non_promotion_does_not_complete_the_board() -> None:
    snapshot = json.loads(DECISION.read_text(encoding="utf-8"))
    payload = produce_promotion_or_honest_non_promotion(
        [
            run_hermetic_paired_benchmark(),
            run_historical_paired_replay(),
            run_held_out_plan_quality(),
        ]
    )
    assert payload["decision"] == snapshot["decision"] == "honest_non_promotion"
    assert payload["promoted"] is False
    assert payload["completion_authority"] is False
    assert payload["remaining_todo"] == 24
