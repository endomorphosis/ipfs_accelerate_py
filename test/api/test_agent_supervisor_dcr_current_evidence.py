"""Contracts for DCR-010 current-tree evidence reconciliation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_current_state import (
    CURRENT_IMPLEMENTATION_EVIDENCE_SCHEMA,
    ReuseClassification,
    reconcile_current_evidence,
    validate_current_evidence,
    write_current_evidence,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _git(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repository_root: Path, message: str) -> str:
    _git(repository_root, "add", "-A")
    _git(repository_root, "commit", "-qm", message)
    return _git(repository_root, "rev-parse", "HEAD")


def _temporary_git_repository(tmp_path: Path) -> Path:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    _git(repository_root, "init", "-q")
    _git(repository_root, "config", "user.name", "DCR test")
    _git(repository_root, "config", "user.email", "dcr-test@example.invalid")
    (repository_root / "README.md").write_text("baseline\n", encoding="utf-8")
    _commit(repository_root, "baseline")
    return repository_root


def _recompute_report_id(report: dict[str, Any]) -> None:
    payload = {key: value for key, value in report.items() if key != "report_id"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    report["report_id"] = "sha256:" + hashlib.sha256(encoded).hexdigest()


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


def test_current_state_artifact_remains_valid_at_a_merged_descendant_head() -> None:
    artifact = (
        REPOSITORY_ROOT
        / "data/agent_supervisor/deterministic_contract_repair/current-state.json"
    )
    stored = json.loads(artifact.read_text(encoding="utf-8"))
    validation = validate_current_evidence(artifact, REPOSITORY_ROOT)

    assert validation.valid, validation.reason_codes
    assert validation.observed_repository_commit == stored["repository_commit"]
    assert stored["classification_counts"] == {
        classification.value: sum(
            item["classification"] == classification.value
            for item in stored["components"]
        )
        for classification in ReuseClassification
    }


def test_current_state_validation_accepts_unchanged_descendant_evidence(
    tmp_path: Path,
) -> None:
    repository_root = _temporary_git_repository(tmp_path)
    stored = reconcile_current_evidence(repository_root).to_dict()
    (repository_root / "README.md").write_text("descendant only\n", encoding="utf-8")
    current_commit = _commit(repository_root, "unchanged evidence descendant")

    validation = validate_current_evidence(stored, repository_root)

    assert validation.valid, validation.reason_codes
    assert validation.observed_repository_commit == stored["repository_commit"]
    assert validation.current_repository_commit == current_commit


def test_current_state_validation_rejects_tampered_self_cid(tmp_path: Path) -> None:
    repository_root = _temporary_git_repository(tmp_path)
    stored = reconcile_current_evidence(repository_root).to_dict()
    stored["report_id"] = "sha256:" + "0" * 64

    validation = validate_current_evidence(stored, repository_root)

    assert not validation.valid
    assert validation.reason_codes == ("stored_report_id_mismatch",)


def test_current_state_validation_rejects_component_tamper_even_with_new_cid(
    tmp_path: Path,
) -> None:
    repository_root = _temporary_git_repository(tmp_path)
    stored = deepcopy(reconcile_current_evidence(repository_root).to_dict())
    stored["components"][0]["evidence"][0]["exists"] = True
    _recompute_report_id(stored)

    validation = validate_current_evidence(stored, repository_root)

    assert not validation.valid
    assert validation.reason_codes == ("current_evidence_drift",)


def test_current_state_validation_rejects_current_evidence_drift(
    tmp_path: Path,
) -> None:
    repository_root = _temporary_git_repository(tmp_path)
    stored = reconcile_current_evidence(repository_root).to_dict()
    source = repository_root / (
        "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/"
        "todo_daemon/implementation_disposition.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text("current evidence changed\n", encoding="utf-8")
    _commit(repository_root, "change observed source bytes")

    validation = validate_current_evidence(stored, repository_root)

    assert not validation.valid
    assert validation.reason_codes == ("current_evidence_drift",)


def test_current_state_validation_rejects_non_ancestor_observation(
    tmp_path: Path,
) -> None:
    repository_root = _temporary_git_repository(tmp_path)
    stored = reconcile_current_evidence(repository_root).to_dict()
    _git(repository_root, "checkout", "--orphan", "unrelated")
    (repository_root / "unrelated.txt").write_text("unrelated\n", encoding="utf-8")
    _commit(repository_root, "unrelated history")

    validation = validate_current_evidence(stored, repository_root)

    assert not validation.valid
    assert validation.reason_codes == ("observed_repository_commit_not_ancestor",)


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
