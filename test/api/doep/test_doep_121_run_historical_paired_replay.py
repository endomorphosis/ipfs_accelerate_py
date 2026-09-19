"""DOEP-121 historical paired replay."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.agent_supervisor.doep.paired import run_historical_paired_replay


RESULT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "benchmarks"
    / "historical_paired.json"
)


def test_historical_paired_replay_does_not_mutate_or_complete() -> None:
    snapshot = json.loads(RESULT.read_text(encoding="utf-8"))
    payload = run_historical_paired_replay()
    assert payload["schema"] == snapshot["schema"] == "doep-historical-paired-replay@1"
    assert payload["pairs"]
    assert payload["mutates_live_store"] is False
    for pair in payload["pairs"]:
        assert pair["mutates_live_store"] is False
        assert pair["completion_authority"] is False
        assert "replay_id" in pair
