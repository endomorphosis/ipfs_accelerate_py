"""DOEP-114 historical replay corpus."""

from __future__ import annotations

from pathlib import Path

from benchmarks.agent_supervisor.doep.corpora import load_historical_replays


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "agent_supervisor_doep"
    / "historical_replays.json"
)


def test_historical_replays_are_offline_and_non_mutating() -> None:
    payload = load_historical_replays(FIXTURE)
    assert payload["schema"] == "doep-historical-replays@1"
    assert payload["replays"]
    assert payload["mutates_live_store"] is False
    assert payload["completion_authority"] is False
    for replay in payload["replays"]:
        assert replay["network"] is False
        assert replay["mutates_live_store"] is False
        assert "replay_id" in replay
        assert "generation" in replay
