"""DOEP-113 hermetic objective corpus."""

from __future__ import annotations

import json
from pathlib import Path


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "agent_supervisor_doep"
    / "hermetic_objectives.json"
)


def test_hermetic_corpus_has_no_network_or_secrets() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["schema"] == "doep-hermetic-objectives@1"
    assert payload["objectives"]
    for objective in payload["objectives"]:
        assert objective["hermetic"] is True
        assert objective["network"] is False
        assert objective["secrets"] is False
        assert "objective_id" in objective
