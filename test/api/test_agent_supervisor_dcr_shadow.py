"""DCR-101: report-only and shadow execution against current repositories.

Acceptance:
* No writes outside runtime paths.
* Every proposal is explainable/replayable.
* Shadow metrics meet reviewed release thresholds.
* Never publish or project completions from shadow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_shadow import (
    DEFAULT_SHADOW_REPORT_PATH,
    DCR_SHADOW_EVIDENCE,
    DCR_TASK_ID,
    REPAIR_SHADOW_REPORT_INTERFACE,
    SHADOW_THRESHOLDS,
    ComparisonLabel,
    DeterministicRepairShadowRun,
    ShadowProposal,
    compare_shadow_to_truth,
    materialize_shadow_report,
    run_deterministic_repair_shadow,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return here.parents[4]


@pytest.fixture(scope="module")
def shadow_run() -> DeterministicRepairShadowRun:
    return run_deterministic_repair_shadow(repo_root=_repo_root())


def test_interfaces_and_symbols() -> None:
    assert REPAIR_SHADOW_REPORT_INTERFACE == "RepairShadowReport@1"
    assert DeterministicRepairShadowRun.INTERFACE == REPAIR_SHADOW_REPORT_INTERFACE
    assert DCR_TASK_ID == "DCR-101"
    assert DCR_SHADOW_EVIDENCE == "dcr/repair-shadow-report@1"
    assert callable(compare_shadow_to_truth)
    assert callable(run_deterministic_repair_shadow)
    assert SHADOW_THRESHOLDS["max_source_mutations"] == 0


def test_shadow_passes_report_only(shadow_run: DeterministicRepairShadowRun) -> None:
    assert shadow_run.passed is True
    assert shadow_run.mode == "report_only"
    assert shadow_run.thresholds_met is True
    assert shadow_run.source_mutations == 0
    assert shadow_run.writes_outside_runtime == 0
    assert shadow_run.runtime_model_calls == 0
    assert shadow_run.provider_calls == 0
    assert shadow_run.metrics["published_completions"] == 0
    assert "never_projected_completed" in shadow_run.reason_codes


def test_proposals_explainable_and_replayable(
    shadow_run: DeterministicRepairShadowRun,
) -> None:
    assert shadow_run.proposals
    for proposal in shadow_run.proposals:
        assert proposal.explanation
        assert proposal.replay_seed.startswith("sha256:")
        assert proposal.to_dict()["published"] is False
        assert proposal.to_dict()["applied"] is False
    assert all(c.explainable and c.replayable for c in shadow_run.comparisons)


def test_compare_shadow_to_truth_labels() -> None:
    proposals = (
        ShadowProposal(
            proposal_id="p1",
            operator="op:test",
            target_key="conformance/live-three-service",
            disposition="propose",
            explanation="test",
            replay_seed="sha256:" + "0" * 64,
        ),
        ShadowProposal(
            proposal_id="p2",
            operator="op:abstain",
            target_key="residual/x",
            disposition="abstain",
            explanation="review",
            replay_seed="sha256:" + "1" * 64,
        ),
    )
    truth = {
        "conformance/live-three-service": "repaired",
        "residual/x": "residual",
    }
    comps = compare_shadow_to_truth(proposals=proposals, truth=truth)
    by_id = {c.proposal_id: c for c in comps}
    assert by_id["p1"].comparison is ComparisonLabel.MATCH
    assert by_id["p2"].comparison is ComparisonLabel.ABSTAIN


def test_no_conflicts_or_extra_proposals(
    shadow_run: DeterministicRepairShadowRun,
) -> None:
    assert not any(
        c.comparison is ComparisonLabel.CONFLICT for c in shadow_run.comparisons
    )
    assert not any(
        c.comparison is ComparisonLabel.EXTRA_PROPOSAL for c in shadow_run.comparisons
    )


def test_materialize_shadow_report(tmp_path: Path) -> None:
    dest = tmp_path / "shadow-report.json"
    payload = materialize_shadow_report(repo_root=_repo_root(), destination=dest)
    assert dest.is_file()
    on_disk = json.loads(dest.read_text(encoding="utf-8"))
    assert on_disk["interface"] == REPAIR_SHADOW_REPORT_INTERFACE
    assert on_disk["task_id"] == DCR_TASK_ID
    assert on_disk["result"]["passed"] is True
    assert on_disk["source_mutations"] == 0
    assert on_disk["published_completions"] == 0
    assert payload["result"]["mode"] == "report_only"


def test_default_path() -> None:
    assert DEFAULT_SHADOW_REPORT_PATH.endswith("shadow-report.json")
