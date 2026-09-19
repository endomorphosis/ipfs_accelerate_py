"""DOEP-114 historical replay corpus."""

from __future__ import annotations

import json
from pathlib import Path


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "agent_supervisor_doep"
    / "historical_replays.json"
)


def test_historical_replays_are_offline_and_non_mutating() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["schema"] == "doep-historical-replays@1"
    assert payload["replays"]
    for replay in payload["replays"]:
        assert replay["network"] is False
        assert replay["mutates_live_store"] is False
        assert "replay_id" in replay
        assert "generation" in replay
