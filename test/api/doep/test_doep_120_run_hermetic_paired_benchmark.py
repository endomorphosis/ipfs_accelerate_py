"""DOEP-120 hermetic paired benchmark."""

from __future__ import annotations

import json
from pathlib import Path


RESULT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "benchmarks"
    / "hermetic_paired.json"
)


def test_hermetic_paired_benchmark_is_offline_and_non_completing() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["schema"] == "doep-hermetic-paired-benchmark@1"
    assert payload["pairs"]
    for pair in payload["pairs"]:
        assert pair["network"] is False
        assert pair["completion_authority"] is False
        assert "baseline" in pair and "candidate" in pair
