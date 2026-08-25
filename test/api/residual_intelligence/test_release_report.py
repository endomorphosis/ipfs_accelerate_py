from __future__ import annotations

import copy
import json
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


def _payload() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_final_release_report_is_complete_current_tree_observation() -> None:
    payload = _payload()
    report = ResidualIntelligenceReleaseReport.from_dict(payload)
    assert validate_release_claims(report) is report
    assert report.to_dict() == payload
    assert report.start_tree == "40f0771e77d394ac91d92cc1edb02f7860f6131b"
    assert report.end_tree == "59b11091b143293b028390b8c949b0ee7c21f6f5"
    assert report.lineage["end"]["tree"] == report.end_tree
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


def test_report_preserves_before_after_and_cost_denominators() -> None:
    report = ResidualIntelligenceReleaseReport.from_dict(_payload())
    assert report.metrics["before"] == report.before
    assert report.metrics["after"] == report.after
    assert report.before["total_denominator"] == 384
    assert report.after["total_denominator"] == 384
    assert report.before["evaluated"] == report.after["evaluated"] == 0
    assert report.costs["denominators"]["frozen_benchmark_cases"] == 384
    assert report.costs["break_even"]["status"] == "not_applicable"
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
    assert "training_unavailable" in text
    assert "Promotion eligibility is false" in text
    assert "no-promoted-expert-route" in text
    assert "384 cases" in text
