from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualIntelligenceError,
    ResidualTaskFamily,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.release import (
    FORBIDDEN_CLAIMS,
    ResidualIntelligenceReleaseReport,
    validate_release_claims,
)

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "docs" / "architecture" / "residual_intelligence_inventory"
REPORT_PATH = INVENTORY / "final_release_report.json"
MARKDOWN_PATH = INVENTORY / "final_release_report.md"
START_COMMIT = "84a056e41e48a81d4484be43840196578d6c87da"
START_TREE = "40f0771e77d394ac91d92cc1edb02f7860f6131b"
FINAL_MERGED_COMMIT = "d694b164cf196c2df48b45b494b3df4fdd3f3e87"
FINAL_MERGED_TREE = "d121b4339de73a1a079c32c97b4c5a41f164e128"


def _payload() -> dict[str, object]:
    value = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256_file(path: str) -> str:
    return "sha256:" + hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_final_release_report_is_closed_current_tree_observation() -> None:
    payload = _payload()
    report = ResidualIntelligenceReleaseReport.from_dict(payload)

    assert validate_release_claims(report) is report
    assert report.to_dict() == payload
    assert report.start_tree == START_TREE
    assert report.end_tree == FINAL_MERGED_TREE
    assert report.rollback_target == START_COMMIT
    assert subprocess.check_output(
        ["git", "rev-parse", f"{FINAL_MERGED_COMMIT}^{{tree}}"],
        cwd=ROOT,
        text=True,
    ).strip() == FINAL_MERGED_TREE
    assert set(report.expert_dispositions) == {
        family.value for family in ResidualTaskFamily
    }
    assert set(report.expert_dispositions.values()) == {"CAPABILITY_UNAVAILABLE"}


def test_report_binds_exact_producer_files_symbols_and_benchmark() -> None:
    report = ResidualIntelligenceReleaseReport.from_dict(_payload())
    producer_artifacts = report.producer_artifacts
    tasks = producer_artifacts["tasks"]
    assert isinstance(tasks, list)
    assert [task["task_alias"] for task in tasks] == [
        "VRIF-028",
        "VRIF-029",
        "VRIF-030",
        "VRIF-031",
    ]
    for task in tasks:
        assert isinstance(task, dict)
        artifacts = task["artifacts"]
        assert isinstance(artifacts, list)
        assert artifacts
        for artifact in artifacts:
            assert isinstance(artifact, dict)
            assert artifact["blob_identity"] == _sha256_file(str(artifact["path"]))

    assert report.files_symbols["declared_output_paths"] == [
        "docs/architecture/residual_intelligence_inventory/final_release_report.json",
        "docs/architecture/residual_intelligence_inventory/final_release_report.md",
        "test/api/residual_intelligence/test_release_report.py",
    ]
    assert report.files_symbols["declared_symbols"] == [
        "ResidualIntelligenceReleaseReport",
        "ResidualGapReport",
        "validate_release_claims",
    ]
    manifest = json.loads(
        (ROOT / "benchmarks/agent_supervisor/residual_intelligence/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    freeze = manifest["benchmark_freeze"]
    proof = report.proof_validation
    assert proof["benchmark_freeze_id"] == freeze["freeze_id"]
    assert proof["benchmark_case_root"] == freeze["case_root"]
    assert proof["benchmark_binding_set_id"] == freeze["binding_set_id"]
    assert proof["paired_baseline_id"] == freeze["paired_baseline"]["paired_baseline_id"]
    assert proof["benchmark_case_payload_disposition"] == (
        freeze["case_payload_disposition"]
    )
    assert proof["benchmark_evaluation_disposition"] == freeze[
        "evaluation_disposition"
    ]


def test_report_preserves_data_denominators_costs_and_blocked_disposition() -> None:
    report = ResidualIntelligenceReleaseReport.from_dict(_payload())
    admission = json.loads(
        (
            ROOT
            / "benchmarks/agent_supervisor/residual_intelligence/"
            "synthetic_training_admission.json"
        ).read_text(encoding="utf-8")
    )
    corpus = report.corpus_rights_splits
    assert report.corpus_admission_id == admission["admission_id"]
    assert corpus["corpus_root"] == admission["corpus_root"]
    assert corpus["source_rights_root"] == admission["source_rights_root"]
    assert corpus["split_root"] == admission["split_root"]
    assert corpus["hidden_test_bodies_accessed"] is False
    assert report.architecture_tokenizer_checkpoint == {
        "disposition": "training_unavailable",
        "architecture": "not_selected",
        "tokenizer": "no_learned_tokenizer_admitted",
        "checkpoint": "not_created",
        "training": "not_attempted",
    }
    assert report.before == report.after
    assert report.before["total"] == 96
    assert report.before["accept"] == 0
    assert report.before["abstain"] == 96
    assert set(report.before["denominators_by_family"].values()) == {4}
    assert report.costs == {"tokens": 0, "break_even": 0}
    assert report.promotion_eligible is False
    assert report.gaps.blockers == ("training_unavailable",)
    assert report.gaps.not_run == ("gpu_live_qualification", "promotion", "training")
    assert set(report.gaps.unsupported_claims) == FORBIDDEN_CLAIMS
    assert report.drift["detectors_run"] == []
    assert report.rollback_blocker_eligibility["report_authority"] == "non_authoritative"


def test_release_report_rejects_promotional_and_evidence_mutations() -> None:
    payload = _payload()
    promotional = copy.deepcopy(payload)
    promotional["promotion_eligible"] = True
    with pytest.raises(ResidualIntelligenceError, match="cannot promote"):
        ResidualIntelligenceReleaseReport.from_dict(promotional)

    wrong_cost = copy.deepcopy(payload)
    wrong_cost["costs"]["tokens"] = 1
    with pytest.raises(ResidualIntelligenceError, match="costs must be exactly"):
        validate_release_claims(ResidualIntelligenceReleaseReport.from_dict(wrong_cost))

    missing_nonclaim = copy.deepcopy(payload)
    missing_nonclaim["gaps"]["unsupported_claims"].pop()
    with pytest.raises(ResidualIntelligenceError, match="unsupported claim token"):
        validate_release_claims(
            ResidualIntelligenceReleaseReport.from_dict(missing_nonclaim)
        )


def test_human_report_matches_machine_report_and_explicit_gaps() -> None:
    text = MARKDOWN_PATH.read_text(encoding="utf-8")
    for value in (
        START_COMMIT,
        START_TREE,
        FINAL_MERGED_COMMIT,
        FINAL_MERGED_TREE,
        "training_unavailable",
        "CAPABILITY_UNAVAILABLE",
        "Promotion eligibility is false",
        "gpu_live_qualification",
        "production-ready",
        "does not recursively qualify",
    ):
        assert value in text
