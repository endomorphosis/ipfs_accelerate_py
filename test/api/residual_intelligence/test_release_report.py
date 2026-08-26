from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualIntelligenceError,
    ResidualTaskFamily,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.release import (
    FORBIDDEN_CLAIMS,
    REPORT_SCHEMA,
    ResidualIntelligenceReleaseReport,
    render_vrif_release_report_markdown,
    validate_release_claims,
)

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "docs" / "architecture" / "residual_intelligence_inventory"
REPORT_PATH = INVENTORY / "final_release_report.json"
MARKDOWN_PATH = INVENTORY / "final_release_report.md"
BASELINE_PATH = INVENTORY / "baseline.json"
ADMISSION_PATH = (
    ROOT
    / "benchmarks"
    / "agent_supervisor"
    / "residual_intelligence"
    / "synthetic_training_admission.json"
)
SPLIT_PATH = (
    ROOT
    / "benchmarks"
    / "agent_supervisor"
    / "residual_intelligence"
    / "synthetic_split_manifest.json"
)
MANIFEST_PATH = (
    ROOT / "benchmarks" / "agent_supervisor" / "residual_intelligence" / "manifest.json"
)
PRODUCER_OUTPUT_PATHS: dict[str, tuple[str, ...]] = {
    "VRIF-028": (
        "docs/architecture/residual_intelligence_inventory/adversarial_campaign_report.json",
        "ipfs_accelerate_py/agent_supervisor/residual_intelligence/adversarial.py",
        "test/api/residual_intelligence/test_adversarial.py",
    ),
    "VRIF-029": (
        "ipfs_accelerate_py/agent_supervisor/control/control_plane.py",
        "ipfs_accelerate_py/agent_supervisor/residual_intelligence/cli.py",
        "test/api/residual_intelligence/test_control_surface.py",
    ),
    "VRIF-030": (
        "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl",
        "benchmarks/agent_supervisor/residual_intelligence/manifest.json",
        "test/api/residual_intelligence/test_benchmark.py",
    ),
    "VRIF-031": (
        "ipfs_accelerate_py/agent_supervisor/residual_intelligence/promotion.py",
        "test/api/residual_intelligence/test_promotion.py",
    ),
}
DECLARED_OUTPUT_PATHS = (
    "docs/architecture/residual_intelligence_inventory/final_release_report.json",
    "docs/architecture/residual_intelligence_inventory/final_release_report.md",
    "test/api/residual_intelligence/test_release_report.py",
)
REQUIRED_REPORT_PATHS = DECLARED_OUTPUT_PATHS[:2]
DECLARED_SYMBOLS = (
    "ResidualIntelligenceReleaseReport",
    "ResidualGapReport",
    "validate_release_claims",
)
VALIDATION_COMMANDS = [
    [
        "python3 -m pytest -q test/api/residual_intelligence/test_release_report.py "
        "&& python3 scripts/validate_agent_supervisor_residual_intelligence_board.py "
        "--check-all"
    ]
]
UNSUPPORTED_CLAIMS = (
    "learned",
    "verified",
    "safe",
    "autonomous",
    "token-efficient",
    "production-ready",
)
NOT_RUN = ("gpu_live_qualification", "promotion", "training")
BLOCKERS = ("training_unavailable",)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_identity(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256_file(path: str) -> str:
    return _sha256_bytes((ROOT / path).read_bytes())


def _producer_artifacts() -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for alias, paths in PRODUCER_OUTPUT_PATHS.items():
        artifacts = [
            {"path": path, "blob_identity": _sha256_file(path)} for path in paths
        ]
        task_body = {"task_alias": alias, "artifacts": artifacts}
        tasks.append({**task_body, "bundle_id": _canonical_identity(task_body)})
    body = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/goal-terminal-producer-artifacts@1"
        ),
        "digest_algorithm": "sha256",
        "tasks": tasks,
    }
    return {**body, "bundle_id": _canonical_identity(body)}


def _frozen_scores(families: list[str]) -> dict[str, Any]:
    return {
        "accept": 0,
        "abstain": len(families) * 4,
        "total": len(families) * 4,
        "denominators_by_family": {family: 4 for family in families},
    }


def build_current_tree_release_report() -> dict[str, Any]:
    """Join the current HEAD tree with admitted producer and freeze identities."""

    baseline = _json(BASELINE_PATH)
    source = baseline["source"]
    assert isinstance(source, dict)
    start_tree = str(source["tree"])
    rollback_target = str(source["commit"])
    end_tree = _git("rev-parse", "HEAD^{tree}")
    assert _git("rev-parse", f"{rollback_target}^{{tree}}") == start_tree
    admission = _json(ADMISSION_PATH)
    split = _json(SPLIT_PATH)
    manifest = _json(MANIFEST_PATH)
    freeze = manifest["benchmark_freeze"]
    assert isinstance(freeze, dict)
    paired_baseline = freeze["paired_baseline"]
    assert isinstance(paired_baseline, dict)
    families = [family.value for family in ResidualTaskFamily]
    scores = _frozen_scores(families)
    producer_artifacts = _producer_artifacts()
    bundle_id = producer_artifacts["bundle_id"]
    admission_id = str(admission["admission_id"])
    blockers = list(BLOCKERS)
    not_run = list(NOT_RUN)
    return {
        "schema": REPORT_SCHEMA,
        "start_tree": start_tree,
        "end_tree": end_tree,
        "corpus_admission_id": admission_id,
        "expert_dispositions": {
            family: "CAPABILITY_UNAVAILABLE" for family in families
        },
        "before": copy.deepcopy(scores),
        "after": copy.deepcopy(scores),
        "costs": {"tokens": 0, "break_even": 0},
        "promotion_eligible": False,
        "rollback_target": rollback_target,
        "gaps": {
            "blockers": blockers,
            "unsupported_claims": list(UNSUPPORTED_CLAIMS),
            "not_run": not_run,
        },
        "producer_artifacts": producer_artifacts,
        "files_symbols": {
            "disposition": "current_tracked_blobs_bound",
            "declared_output_paths": list(DECLARED_OUTPUT_PATHS),
            "required_report_paths": list(REQUIRED_REPORT_PATHS),
            "declared_symbols": list(DECLARED_SYMBOLS),
            "producer_artifact_bundle_id": bundle_id,
        },
        "corpus_rights_splits": {
            "disposition": "training_unavailable",
            "admission_id": admission_id,
            "corpus_root": str(admission["corpus_root"]),
            "source_rights_root": str(admission["source_rights_root"]),
            "split_root": str(split["split_root"]),
            "partitions": ["training", "development", "held_out", "adversarial"],
            "hidden_test_bodies_accessed": False,
            "privacy_disposition": "public_report_bounded",
        },
        "architecture_tokenizer_checkpoint": {
            "disposition": "training_unavailable",
            "architecture": "not_selected",
            "tokenizer": "no_learned_tokenizer_admitted",
            "checkpoint": "not_created",
            "training": "not_attempted",
        },
        "proof_validation": {
            "disposition": "owner_receipts_required",
            "validation_commands": [list(command) for command in VALIDATION_COMMANDS],
            "producer_artifact_bundle_id": bundle_id,
            "benchmark_freeze_id": freeze["freeze_id"],
            "benchmark_case_root": freeze["case_root"],
            "benchmark_binding_set_id": freeze["binding_set_id"],
            "paired_baseline_id": paired_baseline["paired_baseline_id"],
            "benchmark_case_payload_disposition": freeze["case_payload_disposition"],
            "benchmark_evaluation_disposition": freeze["evaluation_disposition"],
            "producer_database_portal_validations": "required",
            "terminal_database_portal_validation": "required",
            "report_authoritative": False,
        },
        "drift": {
            "disposition": "not_run_training_unavailable",
            "reference_tree": start_tree,
            "evaluated_tree": end_tree,
            "checkpoint_available": False,
            "detectors_run": [],
            "reason_codes": ["no_admitted_checkpoint", "training_unavailable"],
        },
        "rollback_blocker_eligibility": {
            "promotion_eligible": False,
            "rollback_target": rollback_target,
            "blockers": blockers,
            "not_run": not_run,
            "report_authority": "non_authoritative",
        },
    }


def _write_release_reports(report: Mapping[str, Any]) -> None:
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_PATH.write_text(
        render_vrif_release_report_markdown(report),
        encoding="utf-8",
    )


@pytest.fixture(scope="module", autouse=True)
def current_tree_reports() -> dict[str, Any]:
    report = build_current_tree_release_report()
    typed = validate_release_claims(ResidualIntelligenceReleaseReport.from_dict(report))
    assert typed.to_dict() == report
    _write_release_reports(report)
    return report


def _payload() -> dict[str, Any]:
    value = _json(REPORT_PATH)
    assert isinstance(value, dict)
    return value


def test_final_release_report_binds_exact_start_and_end_trees(
    current_tree_reports: dict[str, Any],
) -> None:
    payload = _payload()
    report = ResidualIntelligenceReleaseReport.from_dict(payload)
    baseline = _json(BASELINE_PATH)["source"]
    assert isinstance(baseline, dict)
    start_commit = str(baseline["commit"])
    start_tree = str(baseline["tree"])
    end_tree = _git("rev-parse", "HEAD^{tree}")

    assert validate_release_claims(report) is report
    assert report.to_dict() == payload == current_tree_reports
    assert report.start_tree == start_tree == current_tree_reports["start_tree"]
    assert report.end_tree == end_tree == current_tree_reports["end_tree"]
    assert report.start_tree != report.end_tree
    assert report.rollback_target == start_commit
    assert _git("rev-parse", f"{start_commit}^{{tree}}") == start_tree
    assert set(report.expert_dispositions) == {
        family.value for family in ResidualTaskFamily
    }
    assert set(report.expert_dispositions.values()) == {"CAPABILITY_UNAVAILABLE"}


def test_report_binds_exact_files_symbols_and_producer_blobs() -> None:
    report = ResidualIntelligenceReleaseReport.from_dict(_payload())
    producer_artifacts = report.producer_artifacts
    tasks = producer_artifacts["tasks"]
    assert isinstance(tasks, list)
    assert [task["task_alias"] for task in tasks] == list(PRODUCER_OUTPUT_PATHS)
    for task, (alias, paths) in zip(tasks, PRODUCER_OUTPUT_PATHS.items(), strict=True):
        assert isinstance(task, dict)
        assert task["task_alias"] == alias
        artifacts = task["artifacts"]
        assert isinstance(artifacts, list)
        assert [artifact["path"] for artifact in artifacts] == list(paths)
        for artifact, path in zip(artifacts, paths, strict=True):
            assert isinstance(artifact, dict)
            assert artifact["path"] == path
            assert artifact["blob_identity"] == _sha256_file(path)
        task_body = {"task_alias": alias, "artifacts": artifacts}
        assert task["bundle_id"] == _canonical_identity(task_body)

    assert report.files_symbols["declared_output_paths"] == list(DECLARED_OUTPUT_PATHS)
    assert report.files_symbols["required_report_paths"] == list(REQUIRED_REPORT_PATHS)
    assert report.files_symbols["declared_symbols"] == list(DECLARED_SYMBOLS)
    assert report.files_symbols["disposition"] == "current_tracked_blobs_bound"
    assert (
        report.files_symbols["producer_artifact_bundle_id"]
        == producer_artifacts["bundle_id"]
    )
    body = {
        "schema": producer_artifacts["schema"],
        "digest_algorithm": producer_artifacts["digest_algorithm"],
        "tasks": tasks,
    }
    assert producer_artifacts["bundle_id"] == _canonical_identity(body)


def test_report_preserves_corpus_architecture_denominators_and_costs() -> None:
    report = ResidualIntelligenceReleaseReport.from_dict(_payload())
    admission = _json(ADMISSION_PATH)
    split = _json(SPLIT_PATH)
    corpus = report.corpus_rights_splits
    assert report.corpus_admission_id == admission["admission_id"]
    assert corpus["disposition"] == "training_unavailable"
    assert corpus["admission_id"] == admission["admission_id"]
    assert corpus["corpus_root"] == admission["corpus_root"]
    assert corpus["source_rights_root"] == admission["source_rights_root"]
    assert corpus["split_root"] == split["split_root"] == admission["split_root"]
    assert corpus["partitions"] == [
        "training",
        "development",
        "held_out",
        "adversarial",
    ]
    assert corpus["hidden_test_bodies_accessed"] is False
    assert corpus["privacy_disposition"] == "public_report_bounded"
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
    assert set(report.before["denominators_by_family"]) == {
        family.value for family in ResidualTaskFamily
    }
    assert set(report.before["denominators_by_family"].values()) == {4}
    assert report.costs == {"tokens": 0, "break_even": 0}


def test_report_binds_proof_validation_drift_and_blocked_eligibility() -> None:
    report = ResidualIntelligenceReleaseReport.from_dict(_payload())
    manifest = _json(MANIFEST_PATH)
    freeze = manifest["benchmark_freeze"]
    assert isinstance(freeze, dict)
    proof = report.proof_validation
    assert proof["disposition"] == "owner_receipts_required"
    assert proof["validation_commands"] == VALIDATION_COMMANDS
    assert proof["benchmark_freeze_id"] == freeze["freeze_id"]
    assert proof["benchmark_case_root"] == freeze["case_root"]
    assert proof["benchmark_binding_set_id"] == freeze["binding_set_id"]
    assert proof["paired_baseline_id"] == freeze["paired_baseline"]["paired_baseline_id"]
    assert proof["benchmark_case_payload_disposition"] == freeze[
        "case_payload_disposition"
    ]
    assert proof["benchmark_evaluation_disposition"] == freeze["evaluation_disposition"]
    assert proof["producer_database_portal_validations"] == "required"
    assert proof["terminal_database_portal_validation"] == "required"
    assert proof["report_authoritative"] is False
    assert proof["producer_artifact_bundle_id"] == report.producer_artifacts["bundle_id"]
    assert report.drift == {
        "disposition": "not_run_training_unavailable",
        "reference_tree": report.start_tree,
        "evaluated_tree": report.end_tree,
        "checkpoint_available": False,
        "detectors_run": [],
        "reason_codes": ["no_admitted_checkpoint", "training_unavailable"],
    }
    assert report.promotion_eligible is False
    assert report.gaps.blockers == BLOCKERS
    assert report.gaps.not_run == NOT_RUN
    assert report.gaps.unsupported_claims == UNSUPPORTED_CLAIMS
    assert set(report.gaps.unsupported_claims) == FORBIDDEN_CLAIMS
    assert report.rollback_blocker_eligibility == {
        "promotion_eligible": False,
        "rollback_target": report.rollback_target,
        "blockers": list(BLOCKERS),
        "not_run": list(NOT_RUN),
        "report_authority": "non_authoritative",
    }


def test_release_report_rejects_promotional_and_evidence_mutations() -> None:
    payload = _payload()
    promotional = copy.deepcopy(payload)
    promotional["promotion_eligible"] = True
    with pytest.raises(ResidualIntelligenceError, match="cannot promote"):
        ResidualIntelligenceReleaseReport.from_dict(promotional)

    wrong_cost = copy.deepcopy(payload)
    costs = wrong_cost["costs"]
    assert isinstance(costs, dict)
    costs["tokens"] = 1
    with pytest.raises(ResidualIntelligenceError, match="costs must be exactly"):
        validate_release_claims(ResidualIntelligenceReleaseReport.from_dict(wrong_cost))

    missing_nonclaim = copy.deepcopy(payload)
    gaps = missing_nonclaim["gaps"]
    assert isinstance(gaps, dict)
    unsupported = gaps["unsupported_claims"]
    assert isinstance(unsupported, list)
    unsupported.pop()
    with pytest.raises(ResidualIntelligenceError, match="unsupported claim token"):
        validate_release_claims(
            ResidualIntelligenceReleaseReport.from_dict(missing_nonclaim)
        )


def test_human_report_is_owner_canonical_renderer_bytes() -> None:
    payload = _payload()
    rendered = render_vrif_release_report_markdown(payload)
    assert MARKDOWN_PATH.read_text(encoding="utf-8") == rendered
    assert MARKDOWN_PATH.read_bytes() == rendered.encode("utf-8")
    for token in (
        payload["start_tree"],
        payload["end_tree"],
        payload["rollback_target"],
        "training_unavailable",
        "CAPABILITY_UNAVAILABLE",
        "gpu_live_qualification",
        "production-ready",
        "non-authoritative",
        "cannot promote",
    ):
        assert str(token) in rendered
