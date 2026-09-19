"""DOEP-115 held-out objective corpus."""

from __future__ import annotations

import json
from pathlib import Path


HELD = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "agent_supervisor_doep"
    / "held_out_objectives.json"
)
HERMETIC = HELD.with_name("hermetic_objectives.json")


def test_held_out_corpus_is_disjoint_from_hermetic() -> None:
    held = json.loads(HELD.read_text(encoding="utf-8"))
    hermetic = json.loads(HERMETIC.read_text(encoding="utf-8"))
    held_ids = {item["objective_id"] for item in held["objectives"]}
    hermetic_ids = {item["objective_id"] for item in hermetic["objectives"]}
    assert held_ids.isdisjoint(hermetic_ids)
    for objective in held["objectives"]:
        assert objective["network"] is False
        assert objective["hermetic"] is True
