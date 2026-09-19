"""DOEP-125 promotion or honest non-promotion receipt."""

from __future__ import annotations

import json
from pathlib import Path


DECISION = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "release"
    / "promotion_decision.json"
)


def test_honest_non_promotion_does_not_complete_the_board() -> None:
    payload = json.loads(DECISION.read_text(encoding="utf-8"))
    assert payload["decision"] == "honest_non_promotion"
    assert payload["promoted"] is False
    assert payload["completion_authority"] is False
