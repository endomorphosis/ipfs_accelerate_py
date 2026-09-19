"""DOEP-113 hermetic objective corpus."""

from __future__ import annotations

from pathlib import Path

from benchmarks.agent_supervisor.doep.corpora import load_hermetic_objectives


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "agent_supervisor_doep"
    / "hermetic_objectives.json"
)


def test_hermetic_corpus_has_no_network_or_secrets() -> None:
    payload = load_hermetic_objectives(FIXTURE)
    assert payload["schema"] == "doep-hermetic-objectives@1"
    assert payload["objectives"]
    assert payload["completion_authority"] is False
    for objective in payload["objectives"]:
        assert objective["hermetic"] is True
        assert objective["network"] is False
        assert objective["secrets"] is False
        assert "objective_id" in objective
