from __future__ import annotations

import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualIntelligenceError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.release import (
    ResidualGapReport,
    ResidualIntelligenceReleaseReport,
    validate_release_claims,
)

ROOT = Path(__file__).resolve().parents[3]


def test_release_report_cannot_promote_and_preserves_denominators() -> None:
    report = ResidualIntelligenceReleaseReport(
        start_tree="tree:start",
        end_tree="tree:end",
        corpus_admission_id="admission:fixture",
        expert_dispositions={"exp:1": "candidate"},
        before={"accept": 1, "abstain": 1},
        after={"accept": 1, "abstain": 2},
        costs={"tokens": 10, "break_even": 0},
        promotion_eligible=False,
        rollback_target="tree:start",
        gaps=ResidualGapReport(
            blockers=("training_unavailable",),
            unsupported_claims=("learned", "verified", "safe", "autonomous", "token-efficient", "production-ready"),
            not_run=("gpu_live_qualification",),
        ),
    )
    validate_release_claims(report)
    payload = report.to_dict()
    assert payload["promotion_eligible"] is False
    assert payload["before"]["accept"] == 1
    json_path = ROOT / "docs/architecture/residual_intelligence_inventory/final_release_report.json"
    md_path = ROOT / "docs/architecture/residual_intelligence_inventory/final_release_report.md"
    stored = json.loads(json_path.read_text(encoding="utf-8"))
    assert stored["promotion_eligible"] is False
    assert "training_unavailable" in md_path.read_text(encoding="utf-8")
    with pytest.raises(ResidualIntelligenceError, match="cannot promote"):
        ResidualIntelligenceReleaseReport(
            start_tree="tree:start",
            end_tree="tree:end",
            corpus_admission_id="admission:fixture",
            expert_dispositions={},
            before={},
            after={},
            costs={},
            promotion_eligible=True,
            rollback_target="tree:start",
            gaps=ResidualGapReport(blockers=(), unsupported_claims=(), not_run=()),
        )
