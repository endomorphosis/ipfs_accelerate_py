"""DOEP-126 residual-gap and marginal-return report."""

from __future__ import annotations

import json
from pathlib import Path


REPORT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "release"
    / "release_report.json"
)


def test_residual_report_does_not_complete_or_promote() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["schema"] == "doep-residual-gap-and-marginal-return@1"
    assert payload["completion_authority"] is False
    assert payload["promoted"] is False
    assert payload["remaining_todo"] == 24
    assert payload["gaps"]
    assert payload["marginal_return"]["forged_cas"] == "forbidden"
