"""Contracts for DCR-010 current-tree evidence reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_current_state import (
    CURRENT_IMPLEMENTATION_EVIDENCE_SCHEMA,
    ReuseClassification,
    reconcile_current_evidence,
    write_current_evidence,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def test_reconciliation_classifies_every_reused_component_and_reports_synthetics() -> None:
    evidence = reconcile_current_evidence(REPOSITORY_ROOT)
    report = evidence.to_dict()

    assert report["schema"] == CURRENT_IMPLEMENTATION_EVIDENCE_SCHEMA
    assert report["authoritative"] is False
    assert report["completion_authorized"] is False
    assert report["report_id"].startswith("sha256:")
    assert report["components"]
    allowed = {item.value for item in ReuseClassification}
    for component in report["components"]:
        assert component["classification"] in allowed
        assert component["legacy_programs"]
        assert component["evidence"]

    synthetic = {item["evidence_id"]: item for item in report["synthetic_evidence"]}
    assert set(synthetic) == {
        "legacy_residual_packet",
        "synthetic_planner_view",
        "synthetic_doctor_view",
    }
    assert all(item["authoritative"] is False for item in synthetic.values())
    assert all(item["present"] is True for item in synthetic.values())
    assert all(
        item["classification"] == ReuseClassification.CONFLICTING.value
        for item in synthetic.values()
    )


def test_current_state_artifact_is_valid_projection() -> None:
    artifact = REPOSITORY_ROOT / "data/agent_supervisor/deterministic_contract_repair/current-state.json"
    stored = json.loads(artifact.read_text(encoding="utf-8"))
    generated = reconcile_current_evidence(REPOSITORY_ROOT).to_dict()

    assert stored == generated
    assert stored["classification_counts"] == {
        classification.value: sum(
            item["classification"] == classification.value
            for item in stored["components"]
        )
        for classification in ReuseClassification
    }


def test_missing_current_evidence_fails_closed(tmp_path: Path) -> None:
    report = reconcile_current_evidence(tmp_path).to_dict()

    assert all(
        item["classification"] == ReuseClassification.INCOMPLETE.value
        for item in report["components"]
    )
    assert report["repository_commit"] == "unavailable"


def test_writer_is_atomic_projection(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "current-state.json"
    written = write_current_evidence(output, REPOSITORY_ROOT)

    assert json.loads(output.read_text(encoding="utf-8")) == written.to_dict()
