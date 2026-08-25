from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualIntelligenceError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.release import (
    FORBIDDEN_CLAIMS,
    ResidualIntelligenceReleaseReport,
    validate_release_claims,
)

ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = ROOT / "docs/architecture/residual_intelligence_inventory/final_release_report.json"
MARKDOWN_PATH = ROOT / "docs/architecture/residual_intelligence_inventory/final_release_report.md"
START_TREE = "40f0771e77d394ac91d92cc1edb02f7860f6131b"
FINAL_MERGED_COMMIT = "06d4958dc96983f16bdb0cb12ce165b5159c90d9"
FINAL_MERGED_TREE = "4b8d07d8f7f25c28e3aa645e79e23b0ea23290f5"


EXPECTED_RELEASE_SYMBOLS = {
    "contracts.py:",
    "residual_ir.py:",
    "rights.py:",
    "splits.py:",
    "expert_specs.py:",
    "baselines.py:",
    "calibration.py:",
    "abstention.py:",
    "ood.py:",
    "cascade.py:",
    "benchmark.py:",
    "corpus.py:",
    "labels.py:",
    "distillation.py:",
    "active_learning.py:",
    "continual_learning.py:",
    "checkpoint.py:",
    "packaging.py:",
    "runtime.py:",
    "router.py:",
    "structured_decoding.py:",
    "structured_specialist.py:",
    "procedure_experts.py:",
    "proof_experts.py:",
    "patch_experts.py:",
    "local_experts.py:",
    "privacy.py:",
    "adversarial.py:",
    "promotion.py:",
    "drift.py:",
    "cli.py:",
    "release.py:",
}


def _payload() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_final_release_report_is_complete_current_tree_observation() -> None:
    payload = _payload()
    report = ResidualIntelligenceReleaseReport.from_dict(payload)
    assert validate_release_claims(report) is report
    assert report.to_dict() == payload
    assert report.start_tree == START_TREE
    assert report.end_tree == FINAL_MERGED_TREE
    assert report.lineage["end"]["commit"] == FINAL_MERGED_COMMIT
    assert report.lineage["end"]["tree"] == report.end_tree
    assert "does not recursively qualify" in report.lineage["snapshot_scope"]
    assert report.corpus_rights_splits["decision"] == "training_unavailable"
    assert report.architecture_tokenizer_checkpoint["checkpoint"]["created"] is False
    assert report.architecture_tokenizer_checkpoint["checkpoint"]["simulated"] is False
    assert report.expert_dispositions["registered_expert_count"] == 0
    assert report.promotion_eligible is False
    assert report.promotion["eligible"] is False
    assert report.rollback["target"] == report.rollback_target
    assert report.report_authority == {
        "completion_authoritative": False,
        "promotion_authoritative": False,
        "proof_authoritative": False,
    }


def test_report_enumerates_the_final_tree_release_surface_and_symbols() -> None:
    report = ResidualIntelligenceReleaseReport.from_dict(_payload())
    symbols = report.files_and_symbols["implementation_symbols"]
    assert {
        symbol.partition(":")[0] + ":" for symbol in symbols
    } == EXPECTED_RELEASE_SYMBOLS
    changed_files = set(report.files_and_symbols["snapshot_changed_files"])
    assert set(report.files_and_symbols["declared_outputs"]) <= changed_files
    assert {
        "benchmarks/agent_supervisor/residual_intelligence/manifest.json",
        "docs/architecture/residual_intelligence_inventory/pgir_training_gate.json",
        "ipfs_accelerate_py/agent_supervisor/residual_intelligence/release.py",
        "ipfs_accelerate_py/agent_supervisor/residual_intelligence/promotion.py",
        "test/api/residual_intelligence/test_release_report.py",
    } <= changed_files
    actual_changed_files = set(
        subprocess.check_output(
            [
                "git",
                "diff",
                "--name-only",
                START_TREE,
                FINAL_MERGED_COMMIT,
                "--",
                "ipfs_accelerate_py/agent_supervisor/residual_intelligence",
                "test/api/residual_intelligence",
                "docs/architecture/residual_intelligence_inventory",
                "benchmarks/agent_supervisor/residual_intelligence",
            ],
            cwd=ROOT,
            text=True,
        ).splitlines()
    )
    assert changed_files == actual_changed_files


def test_report_preserves_before_after_and_cost_denominators() -> None:
    report = ResidualIntelligenceReleaseReport.from_dict(_payload())
    assert report.metrics["before"] == report.before
    assert report.metrics["after"] == report.after
    assert report.before["total_denominator"] == 384
    assert report.after["total_denominator"] == 384
    assert report.before["evaluated"] == report.after["evaluated"] == 0
    assert report.costs["denominators"]["frozen_benchmark_cases"] == 384
    assert report.costs["break_even"]["status"] == "not_applicable"
    assert report.costs["denominators"] == {
        "frozen_benchmark_cases": 384,
        "evaluated_cases": 0,
        "local_attempts": 0,
        "remote_attempts": 0,
        "validation_runs": 0,
        "training_runs": 0,
        "shadow_runs": 0,
        "human_reviews": 0,
        "rollback_events": 0,
    }
    assert set(report.gaps.unsupported_claims) == FORBIDDEN_CLAIMS


def test_release_report_rejects_promotional_or_incomplete_mutations() -> None:
    payload = _payload()
    promotional = copy.deepcopy(payload)
    promotional["promotion_eligible"] = True
    with pytest.raises(ResidualIntelligenceError, match="cannot promote"):
        ResidualIntelligenceReleaseReport.from_dict(promotional)

    denominator_lost = copy.deepcopy(payload)
    denominator_lost["metrics"]["after"]["total_denominator"] = 383
    with pytest.raises(ResidualIntelligenceError, match="same denominator"):
        validate_release_claims(ResidualIntelligenceReleaseReport.from_dict(denominator_lost))

    missing_nonclaim = copy.deepcopy(payload)
    missing_nonclaim["gaps"]["unsupported_claims"].pop()
    with pytest.raises(ResidualIntelligenceError, match="every unsupported claim"):
        validate_release_claims(ResidualIntelligenceReleaseReport.from_dict(missing_nonclaim))


def test_human_report_matches_machine_blocked_disposition() -> None:
    text = MARKDOWN_PATH.read_text(encoding="utf-8")
    assert FINAL_MERGED_COMMIT in text
    assert FINAL_MERGED_TREE in text
    assert "training_unavailable" in text
    assert "Promotion eligibility is false" in text
    assert "no-promoted-expert-route" in text
    assert "384 cases" in text
    assert "Unsupported claims:" in text
