from __future__ import annotations

import copy
import hashlib
import json

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualIntelligenceError,
    ResidualTaskFamily,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.release import (
    REPORT_SCHEMA,
    ResidualIntelligenceReleaseReport,
    validate_release_claims,
)


def _identity(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _producer_artifacts() -> dict[str, object]:
    tasks: list[dict[str, object]] = []
    for alias in ("VRIF-028", "VRIF-029", "VRIF-030", "VRIF-031"):
        bundle: dict[str, object] = {
            "task_alias": alias,
            "artifacts": [
                {
                    "path": f"producer/{alias.lower()}.json",
                    "blob_identity": "sha256:" + "a" * 64,
                }
            ],
        }
        bundle["bundle_id"] = _identity(bundle)
        tasks.append(bundle)
    result: dict[str, object] = {
        "schema": ("ipfs_accelerate_py/agent-supervisor/goal-terminal-producer-artifacts@1"),
        "digest_algorithm": "sha256",
        "tasks": tasks,
    }
    result["bundle_id"] = _identity(result)
    return result


def _payload() -> dict[str, object]:
    families = [family.value for family in ResidualTaskFamily]
    scores = {
        "accept": 0,
        "abstain": len(families) * 4,
        "total": len(families) * 4,
        "denominators_by_family": {family: 4 for family in families},
    }
    producer_artifacts = _producer_artifacts()
    bundle_id = producer_artifacts["bundle_id"]
    start_tree = "1" * 40
    end_tree = "2" * 40
    rollback_target = "3" * 40
    blockers = ["training_unavailable"]
    not_run = ["gpu_live_qualification", "promotion", "training"]
    return {
        "schema": REPORT_SCHEMA,
        "start_tree": start_tree,
        "end_tree": end_tree,
        "corpus_admission_id": "baguqeera-admission-fixture",
        "expert_dispositions": {family: "CAPABILITY_UNAVAILABLE" for family in families},
        "before": scores,
        "after": copy.deepcopy(scores),
        "costs": {"tokens": 0, "break_even": 0},
        "promotion_eligible": False,
        "rollback_target": rollback_target,
        "gaps": {
            "blockers": blockers,
            "unsupported_claims": [
                "learned",
                "verified",
                "safe",
                "autonomous",
                "token-efficient",
                "production-ready",
            ],
            "not_run": not_run,
        },
        "producer_artifacts": producer_artifacts,
        "files_symbols": {
            "disposition": "current_tracked_blobs_bound",
            "declared_output_paths": [
                "docs/architecture/residual_intelligence_inventory/final_release_report.json",
                "docs/architecture/residual_intelligence_inventory/final_release_report.md",
                "test/api/residual_intelligence/test_release_report.py",
            ],
            "required_report_paths": [
                "docs/architecture/residual_intelligence_inventory/final_release_report.json",
                "docs/architecture/residual_intelligence_inventory/final_release_report.md",
            ],
            "declared_symbols": [
                "ResidualIntelligenceReleaseReport",
                "ResidualGapReport",
                "validate_release_claims",
            ],
            "producer_artifact_bundle_id": bundle_id,
        },
        "corpus_rights_splits": {
            "disposition": "training_unavailable",
            "admission_id": "baguqeera-admission-fixture",
            "corpus_root": "baguqeera-corpus-fixture",
            "source_rights_root": "baguqeera-rights-fixture",
            "split_root": "baguqeera-split-fixture",
            "partitions": [
                "training",
                "development",
                "held_out",
                "adversarial",
            ],
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
            "validation_commands": [["python3 -m pytest -q release-test"]],
            "producer_artifact_bundle_id": bundle_id,
            "benchmark_freeze_id": "sha256:" + "4" * 64,
            "benchmark_case_root": "sha256:" + "5" * 64,
            "benchmark_binding_set_id": "sha256:" + "6" * 64,
            "paired_baseline_id": "sha256:" + "7" * 64,
            "benchmark_case_payload_disposition": ("payload_unavailable_training_unavailable"),
            "benchmark_evaluation_disposition": "all_abstain_not_run",
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
            "reason_codes": [
                "no_admitted_checkpoint",
                "training_unavailable",
            ],
        },
        "rollback_blocker_eligibility": {
            "promotion_eligible": False,
            "rollback_target": rollback_target,
            "blockers": blockers,
            "not_run": not_run,
            "report_authority": "non_authoritative",
        },
    }


def test_release_model_round_trips_the_exact_owner_projection() -> None:
    payload = _payload()
    report = ResidualIntelligenceReleaseReport.from_dict(payload)

    assert validate_release_claims(report) is report
    assert report.to_dict() == payload
    assert set(payload) == {
        "schema",
        "start_tree",
        "end_tree",
        "corpus_admission_id",
        "expert_dispositions",
        "before",
        "after",
        "costs",
        "promotion_eligible",
        "rollback_target",
        "gaps",
        "producer_artifacts",
        "files_symbols",
        "corpus_rights_splits",
        "architecture_tokenizer_checkpoint",
        "proof_validation",
        "drift",
        "rollback_blocker_eligibility",
    }


def test_release_model_rejects_the_legacy_projection() -> None:
    payload = _payload()
    payload["lineage"] = {}

    with pytest.raises(ResidualIntelligenceError, match="unknown fields: lineage"):
        ResidualIntelligenceReleaseReport.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("costs", {"tokens": 1, "break_even": 0}, "costs must be exactly"),
        (
            "gaps",
            {
                "blockers": ["training_unavailable"],
                "unsupported_claims": [
                    "verified",
                    "learned",
                    "safe",
                    "autonomous",
                    "token-efficient",
                    "production-ready",
                ],
                "not_run": ["gpu_live_qualification", "promotion", "training"],
            },
            "unsupported claim token in owner order",
        ),
    ],
)
def test_release_model_rejects_owner_semantic_mismatches(
    field: str, replacement: object, message: str
) -> None:
    payload = _payload()
    payload[field] = replacement
    report = ResidualIntelligenceReleaseReport.from_dict(payload)

    with pytest.raises(ResidualIntelligenceError, match=message):
        validate_release_claims(report)


def test_release_model_requires_capability_unavailable_for_every_family() -> None:
    payload = _payload()
    dispositions = payload["expert_dispositions"]
    assert isinstance(dispositions, dict)
    dispositions[ResidualTaskFamily.TASK_CLASSIFICATION.value] = "ABSTAIN"
    report = ResidualIntelligenceReleaseReport.from_dict(payload)

    with pytest.raises(ResidualIntelligenceError, match="CAPABILITY_UNAVAILABLE"):
        validate_release_claims(report)
