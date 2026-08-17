"""DCR-093: adversarial, mutation, stale-state, and authority negatives.

Acceptance:
* Every safety mutation is killed.
* Unknown/unsupported/error never grants mutation/completion.
* Provider/model tripwires remain untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_adversarial import (
    ADVERSARIAL_CONFORMANCE_INTERFACE,
    DEFAULT_ADVERSARIAL_REPORT_PATH,
    DCR_ADVERSARIAL_EVIDENCE,
    DCR_TASK_ID,
    MUTATION_SCORE_INTERFACE,
    AuthorityMutationSuite,
    ContractRepairAdversary,
    DcrAdversarialError,
    DcrAdversarialReport,
    MutationScore,
    evaluate_dcr_adversarial,
    materialize_adversarial_report,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return here.parents[4]


@pytest.fixture(scope="module")
def report() -> DcrAdversarialReport:
    return evaluate_dcr_adversarial(repo_root=_repo_root())


def test_interfaces_and_symbols() -> None:
    assert ADVERSARIAL_CONFORMANCE_INTERFACE == "AdversarialConformance@1"
    assert MUTATION_SCORE_INTERFACE == "MutationScore@1"
    assert DcrAdversarialReport.INTERFACE == ADVERSARIAL_CONFORMANCE_INTERFACE
    assert MutationScore.INTERFACE == MUTATION_SCORE_INTERFACE
    assert AuthorityMutationSuite.INTERFACE == "AuthorityMutationSuite@1"
    assert ContractRepairAdversary.INTERFACE == "ContractRepairAdversary@1"
    assert DCR_TASK_ID == "DCR-093"
    assert DCR_ADVERSARIAL_EVIDENCE == "dcr/adversarial-conformance@1"
    assert callable(evaluate_dcr_adversarial)


def test_every_safety_mutation_is_killed(report: DcrAdversarialReport) -> None:
    assert report.passed is True
    assert report.mutation_score.total >= 15
    assert report.mutation_score.survived == 0
    assert report.mutation_score.killed == report.mutation_score.total
    assert report.mutation_score.score == 1.0
    assert all(item.killed for item in report.outcomes)
    assert all(
        v == "killed" for v in report.killed_survivor_matrix.values()
    )


def test_no_grants_on_unknown_or_error(report: DcrAdversarialReport) -> None:
    for item in report.outcomes:
        assert item.detail.get("grants_mutation") in (False, None)
        assert item.detail.get("grants_completion") in (False, None)
    assert "no_unknown_grants_mutation_or_completion" in report.reason_codes


def test_provider_and_model_tripwires_untouched(report: DcrAdversarialReport) -> None:
    assert report.runtime_model_calls == 0
    assert report.provider_calls == 0
    trip = next(
        item for item in report.outcomes if item.mutation_id == "mut:provider-tripwire"
    )
    assert trip.killed is True
    assert trip.detail.get("runtime_model_calls") == 0
    assert trip.detail.get("provider_calls") == 0


def test_authority_suite_all_killed(report: DcrAdversarialReport) -> None:
    assert report.authority_suite.all_killed is True
    assert report.authority_suite.outcomes
    families = {item.family for item in report.authority_suite.outcomes}
    assert "forged_evidence" in families
    assert "synthetic_evidence" in families
    assert "stale_span" in families


def test_positive_control_and_rollback(report: DcrAdversarialReport) -> None:
    assert report.positive_control_ok is True
    assert report.rollback_verification.get("verified") is True


def test_cannot_pass_with_survivors() -> None:
    score = MutationScore(
        total=2,
        killed=1,
        survived=1,
        errors=0,
        score=0.5,
    )
    with pytest.raises(DcrAdversarialError):
        DcrAdversarialReport(
            passed=True,
            positive_control_ok=True,
            mutation_score=score,
            outcomes=(),
            authority_suite=AuthorityMutationSuite(
                outcomes=(), all_killed=True
            ),
            killed_survivor_matrix={},
            rollback_verification={"verified": True},
            reason_codes=("bad",),
        )


def test_materialize_adversarial_report(tmp_path: Path) -> None:
    dest = tmp_path / "adversarial-report.json"
    payload = materialize_adversarial_report(
        repo_root=_repo_root(),
        destination=dest,
    )
    assert dest.is_file()
    on_disk = json.loads(dest.read_text(encoding="utf-8"))
    assert on_disk["interface"] == ADVERSARIAL_CONFORMANCE_INTERFACE
    assert on_disk["task_id"] == DCR_TASK_ID
    assert on_disk["result"]["passed"] is True
    assert on_disk["result"]["mutation_score"]["survived"] == 0
    assert payload["result"]["passed"] is True


def test_default_path() -> None:
    assert DEFAULT_ADVERSARIAL_REPORT_PATH.endswith("adversarial-report.json")
