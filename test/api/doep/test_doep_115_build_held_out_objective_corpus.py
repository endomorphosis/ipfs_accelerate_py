"""DOEP-115 held-out objective corpus."""

from __future__ import annotations

from pathlib import Path

from benchmarks.agent_supervisor.doep.corpora import load_held_out_objectives


HELD = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "agent_supervisor_doep"
    / "held_out_objectives.json"
)
HERMETIC = HELD.with_name("hermetic_objectives.json")


def test_held_out_corpus_is_disjoint_from_hermetic() -> None:
    held = load_held_out_objectives(HELD, hermetic_path=HERMETIC)
    assert held["held_out"] is True
    assert held["leaked_from_hermetic"] is False
    assert held["completion_authority"] is False
    for objective in held["objectives"]:
        assert objective["network"] is False
        assert objective["hermetic"] is True
