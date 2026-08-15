"""Analysis CLI tests for AAE-057 gaps/vacuity/remediate/evaluate/promote/benchmark.

Acceptance covered here:

* Commands preserve candidate versus authority status.
* Promotion requires explicit ``--authorize-promote`` authority.
* Absolute host paths / external repository roots are rejected.
* No network service is started or claimed.
* Honest unavailable / inconclusive results are returned (never fabricated passes).
* No production policy change is claimed.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance import cli as assurance_cli
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance import (
    cli_assurance as analysis,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


REPO_ID = "repository:sha256:test-repo-identity-aae057"
REPO_STATE = _cid("repo-state-aae057")


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------


class _FakeApi:
    """Injectable campaign API for hermetic CLI tests."""

    def __init__(self) -> None:
        self.gaps_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.vacuity_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.remediate_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.evaluate_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.promote_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def diagnose_surviving_mutant(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.gaps_calls.append((args, kwargs))
        return {
            "interface_id": "SurvivorDiagnosisRun@1",
            "candidate_id": "cand-1",
            "candidate_cid": _cid("cand-1"),
            "outcome_cid": _cid("out-1"),
            "repository_state_cid": REPO_STATE,
            "risk_class": "authorization",
            "high_risk": True,
            "gap_cid": _cid("gap-1"),
            "assurance_gap": {
                "gap_cid": _cid("gap-1"),
                "gap_class": "missing_test",
                "severity": "critical",
            },
            "requires_human_review": False,
            "minimization_failed": False,
            "production_policy_changed": False,
            "reason_codes": [
                "assurance_gap_sealed",
                "production_policy_unchanged",
            ],
            "terminal_status": "complete",
        }

    def analyze_vacuity(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.vacuity_calls.append((args, kwargs))
        return {
            "interface_id": "analyze_vacuity@1",
            "repository_state_cid": REPO_STATE,
            "assurance_manifest_cid": _cid("manifest"),
            "families_analyzed": ["formal_proof"],
            "findings": [
                {
                    "finding_id": "f1",
                    "vacuity_family": "formal_proof",
                    "residual_properties": ["goal remains proven under assumptions"],
                }
            ],
            "finding_cids": [_cid("finding-1")],
            "residual_properties": ["goal remains proven under assumptions"],
            "precise_nonclaims": ["does not prove completeness"],
            "reason_codes": [
                "vacuity_analyzed",
                "no_production_policy_change",
                "no_arbitrary_path_exposure",
            ],
            "terminal_status": "complete",
            "production_policy_changed": False,
        }

    def propose_gap_remediation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.remediate_calls.append((args, kwargs))
        return {
            "interface_id": "RemediationProposalRun@1",
            "proposal_cid": _cid("proposal"),
            "plan_cid": _cid("plan"),
            "gap_cid": _cid("gap-1"),
            "survivor_report_cid": _cid("survivor"),
            "candidate_cids": [_cid("rem-cand-1")],
            "candidate_kinds": ["test_strengthening"],
            "all_heuristic": True,
            "requires_held_out_evaluation": True,
            "production_policy_changed": False,
            "reason_codes": [
                "candidates_proposed",
                "heuristic_candidate_only",
                "held_out_evaluation_required",
                "production_policy_unchanged",
            ],
        }

    def evaluate_remediation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.evaluate_calls.append((args, kwargs))
        return {
            "interface_id": "RemediationEvaluationRun@1",
            "plan_cid": _cid("plan"),
            "candidate_cids": [_cid("rem-cand-1")],
            "evaluation_report_cid": _cid("eval"),
            "qualification_cid": _cid("qual"),
            "disposition": "qualified",
            "verdict": "pass",
            "qualified": True,
            "partitions_covered": ["held_out", "diagnosis"],
            "missing_partitions": [],
            "failed_partitions": [],
            "one_mutant_overfit": False,
            "mock_bypass": False,
            "production_policy_changed": False,
            "reason_codes": [
                "remediation_qualified",
                "held_out_evaluation_required",
                "production_policy_unchanged",
            ],
            "rejection_reasons": [],
        }

    def promote_assurance_policy(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.promote_calls.append((args, kwargs))
        # Default honest reject without disposable repository injection path.
        if kwargs.get("policy_repository") is None:
            return {
                "status": "rejected",
                "head_mutated": False,
                "blocking_reasons": ["missing_policy_repository"],
                "operation_id": kwargs.get("operation_id"),
                "workspace": kwargs.get("workspace", "default"),
                "candidate_cid": _cid("cand-promo"),
                "evaluation_report_cid": _cid("eval"),
                "authorization_cid": args[2]
                if len(args) > 2 and isinstance(args[2], str)
                else _cid("auth"),
                "promoted_policy_cid": None,
                "seal_evidence_cid": kwargs.get("seal_evidence_cid"),
                "held_out_result": None,
                "production_policy_changed": False,
                "diagnostic": "policy_repository is required",
            }
        return {
            "status": "promoted",
            "head_mutated": True,
            "blocking_reasons": [],
            "operation_id": kwargs.get("operation_id"),
            "workspace": kwargs.get("workspace", "default"),
            "candidate_cid": _cid("cand-promo"),
            "evaluation_report_cid": _cid("eval"),
            "authorization_cid": _cid("auth-external"),
            "promoted_policy_cid": _cid("policy-v2"),
            "seal_evidence_cid": kwargs.get("seal_evidence_cid"),
            "held_out_result": "pass",
            "production_policy_changed": False,
            "diagnostic": None,
        }


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    return path


def _ns(**kwargs: Any) -> SimpleNamespace:
    defaults = {
        "output_human": False,
        "output_json": True,
        "timeout_seconds": None,
        "max_candidates": None,
        "max_worktrees": None,
        "cancel": False,
        "cancel_file": None,
        "notes": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Descriptor / registration
# ---------------------------------------------------------------------------


def test_analysis_descriptor_exposes_typed_apis() -> None:
    desc = analysis.assurance_analysis_cli_descriptor()
    assert desc["interface"] == "AssuranceAnalysisCLI@1"
    assert desc["evidence"] == "aae/cli-assurance@1"
    assert desc["explicit_promote_authority_required"] is True
    assert desc["production_policy_change"] is False
    assert desc["network_service"] is False
    assert desc["preserves_candidate_versus_authority"] is True
    assert desc["honest_unavailable"] is True
    assert set(desc["commands"]) == set(analysis.ASSURANCE_ANALYSIS_COMMANDS)
    assert desc["apis"]["gaps"] == "diagnose_surviving_mutant"
    assert desc["apis"]["vacuity"] == "analyze_vacuity"
    assert desc["apis"]["remediate"] == "propose_gap_remediation"
    assert desc["apis"]["evaluate-remediation"] == "evaluate_remediation"
    assert desc["apis"]["promote"] == "promote_assurance_policy"
    assert desc["apis"]["benchmark"] == "benchmark_assurance"


def test_register_analysis_commands_is_parser_only() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    group = analysis.register_assurance_analysis_cli(sub)
    help_io = io.StringIO()
    group.print_help(help_io)
    text = help_io.getvalue()
    for name in analysis.ASSURANCE_ANALYSIS_COMMANDS:
        assert name in text

    # Banned free-form external repo root flags.
    lowered = text.lower()
    for banned in (
        "--repository ",
        "--repository-root",
        "--repo-root",
        "--workdir",
        "--worktree",
    ):
        assert banned not in lowered

    args = parser.parse_args(
        [
            "assurance",
            "promote",
            "--remediation-json",
            "r.json",
            "--evaluation-json",
            "e.json",
            "--campaign-receipt-json",
            "c.json",
            "--promotion-signature-json",
            "s.json",
            "--seal-evidence-cid",
            _cid("seal"),
            "--operation-id",
            "op-1",
            "--authorize-promote",
            "--authorization-cid",
            _cid("auth"),
        ]
    )
    assert args.assurance_command == "promote"
    assert args.authorize_promote is True


def test_cold_import_is_side_effect_free() -> None:
    # Re-import path: module already loaded; descriptor remains static.
    assert analysis.ASSURANCE_ANALYSIS_CLI_INTERFACE == "AssuranceAnalysisCLI@1"
    assert analysis.ASSURANCE_HANDLERS.keys() == set(analysis.ASSURANCE_ANALYSIS_COMMANDS)


# ---------------------------------------------------------------------------
# gaps
# ---------------------------------------------------------------------------


def test_gaps_reaches_diagnose_api(tmp_path: Path) -> None:
    api = _FakeApi()
    args = _ns(
        mutation_json=str(
            _write_json(
                tmp_path / "mutation.json",
                {
                    "candidate_id": "cand-1",
                    "candidate_cid": _cid("cand-1"),
                    "operator_id": "auth_bypass",
                },
            )
        ),
        outcome_json=str(
            _write_json(
                tmp_path / "outcome.json",
                {
                    "outcome_cid": _cid("out-1"),
                    "terminal_status": "survivor",
                    "disposition": "survived_full_verification",
                },
            )
        ),
        repository_state_json=str(
            _write_json(
                tmp_path / "state.json",
                {
                    "repository_id": REPO_ID,
                    "repository_state_cid": REPO_STATE,
                },
            )
        ),
        signals_json=None,
        comparison_json=None,
        minimized_evidence_json=None,
        survivor_report_json=None,
        always_persist_gap=False,
    )
    result = analysis.handle_gaps(args, api=api)
    assert result["api"] == "diagnose_surviving_mutant"
    assert result["production_policy_change"] is False
    assert result["network_service"] is False
    assert result["authority_status"] == analysis.AUTHORITY_NOT_AUTHORITY
    assert result["gap_cid"] == _cid("gap-1")
    assert result["high_risk"] is True
    assert len(api.gaps_calls) == 1


def test_gaps_rejects_absolute_host_path(tmp_path: Path) -> None:
    args = _ns(
        mutation_json=str(
            _write_json(
                tmp_path / "mutation.json",
                {
                    "candidate_id": "cand-1",
                    "worktree_path": "/home/other/external-repo",
                },
            )
        ),
        outcome_json=str(_write_json(tmp_path / "outcome.json", {"ok": True})),
        repository_state_json=str(
            _write_json(tmp_path / "state.json", {"repository_state_cid": REPO_STATE})
        ),
        signals_json=None,
        comparison_json=None,
        minimized_evidence_json=None,
        survivor_report_json=None,
        always_persist_gap=False,
    )
    with pytest.raises(assurance_cli.AssuranceCLIPathError):
        analysis.handle_gaps(args, api=_FakeApi())


# ---------------------------------------------------------------------------
# vacuity
# ---------------------------------------------------------------------------


def test_vacuity_reaches_analyze_api(tmp_path: Path) -> None:
    api = _FakeApi()
    args = _ns(
        manifest_json=str(
            _write_json(
                tmp_path / "manifest.json",
                {
                    "repository_id": REPO_ID,
                    "repository_state_cid": REPO_STATE,
                },
            )
        ),
        repository_state_json=str(
            _write_json(
                tmp_path / "state.json",
                {
                    "repository_id": REPO_ID,
                    "repository_state_cid": REPO_STATE,
                },
            )
        ),
        formal_subject_json=str(
            _write_json(
                tmp_path / "formal.json",
                {
                    "statement": "forall x. P(x)",
                    "assumptions": ["domain_nonempty"],
                },
            )
        ),
        policy_subject_json=None,
        test_subject_json=None,
        zk_receipt_subject_json=None,
        subjects_json=None,
        header_json=None,
    )
    result = analysis.handle_vacuity(args, api=api)
    assert result["api"] == "analyze_vacuity"
    assert result["production_policy_change"] is False
    assert result["authority_status"] == analysis.AUTHORITY_NOT_AUTHORITY
    assert result["finding_count"] == 1
    assert result["residual_properties"]
    assert len(api.vacuity_calls) == 1
    assert "formal_subject" in api.vacuity_calls[0][1]


def test_vacuity_requires_subjects(tmp_path: Path) -> None:
    args = _ns(
        manifest_json=str(_write_json(tmp_path / "m.json", {"ok": True})),
        repository_state_json=str(_write_json(tmp_path / "s.json", {"ok": True})),
        formal_subject_json=None,
        policy_subject_json=None,
        test_subject_json=None,
        zk_receipt_subject_json=None,
        subjects_json=None,
        header_json=None,
    )
    with pytest.raises(assurance_cli.AssuranceCLIUsageError) as exc:
        analysis.handle_vacuity(args, api=_FakeApi())
    assert exc.value.reason_code == "missing_vacuity_subjects"


def test_vacuity_rejects_host_path(tmp_path: Path) -> None:
    args = _ns(
        manifest_json=str(
            _write_json(
                tmp_path / "m.json",
                {"repository_path": "/var/lib/external"},
            )
        ),
        repository_state_json=str(_write_json(tmp_path / "s.json", {"ok": True})),
        formal_subject_json=str(_write_json(tmp_path / "f.json", {"ok": True})),
        policy_subject_json=None,
        test_subject_json=None,
        zk_receipt_subject_json=None,
        subjects_json=None,
        header_json=None,
    )
    with pytest.raises(assurance_cli.AssuranceCLIPathError):
        analysis.handle_vacuity(args, api=_FakeApi())


# ---------------------------------------------------------------------------
# remediate
# ---------------------------------------------------------------------------


def test_remediate_preserves_candidate_status(tmp_path: Path) -> None:
    api = _FakeApi()
    args = _ns(
        surviving_mutant_json=str(
            _write_json(
                tmp_path / "survivor.json",
                {
                    "candidate_id": "cand-1",
                    "candidate_cid": _cid("cand-1"),
                    "risk_class": "authorization",
                },
            )
        ),
        assurance_gap_json=str(
            _write_json(
                tmp_path / "gap.json",
                {
                    "gap_cid": _cid("gap-1"),
                    "gap_class": "missing_test",
                    "severity": "critical",
                },
            )
        ),
    )
    result = analysis.handle_remediate(args, api=api)
    assert result["api"] == "propose_gap_remediation"
    assert result["authority_status"] == analysis.AUTHORITY_CANDIDATE
    assert result["candidate_status"] == "heuristic_candidate"
    assert result["all_heuristic"] is True
    assert result["requires_held_out_evaluation"] is True
    assert result["production_policy_change"] is False
    assert len(api.remediate_calls) == 1


def test_remediate_honors_max_candidates(tmp_path: Path) -> None:
    class _ManyCandidates(_FakeApi):
        def propose_gap_remediation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            base = super().propose_gap_remediation(*args, **kwargs)
            base["candidate_cids"] = [_cid(f"c{i}") for i in range(5)]
            return base

    args = _ns(
        surviving_mutant_json=str(
            _write_json(tmp_path / "s.json", {"candidate_cid": _cid("c")})
        ),
        assurance_gap_json=str(
            _write_json(tmp_path / "g.json", {"gap_cid": _cid("g")})
        ),
        max_candidates=2,
    )
    budget = assurance_cli.resource_budget_from_args(args)
    with pytest.raises(assurance_cli.AssuranceCLIResourceError):
        analysis.handle_remediate(args, api=_ManyCandidates(), budget=budget)


# ---------------------------------------------------------------------------
# evaluate-remediation
# ---------------------------------------------------------------------------


def test_evaluate_remediation_stays_candidate_even_when_qualified(
    tmp_path: Path,
) -> None:
    api = _FakeApi()
    args = _ns(
        remediation_json=str(
            _write_json(
                tmp_path / "rem.json",
                {
                    "plan_cid": _cid("plan"),
                    "candidate_cids": [_cid("rem-cand-1")],
                    "requires_held_out_evaluation": True,
                },
            )
        ),
        held_out_campaign_json=str(
            _write_json(
                tmp_path / "campaign.json",
                {
                    "campaign_id": "held-out-1",
                    "partition_results": [
                        {"partition": "held_out", "passed": True},
                    ],
                },
            )
        ),
        max_cost_delta_bp=None,
        report_id=None,
        raise_on_hard_reject=False,
    )
    result = analysis.handle_evaluate_remediation(args, api=api)
    assert result["api"] == "evaluate_remediation"
    assert result["qualified"] is True
    # Held-out qualification is not production authority.
    assert result["authority_status"] == analysis.AUTHORITY_CANDIDATE
    assert result["candidate_status"] == "heuristic_candidate"
    assert result["production_policy_change"] is False
    assert len(api.evaluate_calls) == 1


def test_evaluate_rejects_absolute_path(tmp_path: Path) -> None:
    args = _ns(
        remediation_json=str(
            _write_json(
                tmp_path / "rem.json",
                {"plan_cid": _cid("plan"), "absolute_path": "/tmp/evil"},
            )
        ),
        held_out_campaign_json=str(
            _write_json(tmp_path / "c.json", {"campaign_id": "c"})
        ),
        max_cost_delta_bp=None,
        report_id=None,
        raise_on_hard_reject=False,
    )
    with pytest.raises(assurance_cli.AssuranceCLIPathError):
        analysis.handle_evaluate_remediation(args, api=_FakeApi())


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------


def test_promote_requires_explicit_authority(tmp_path: Path) -> None:
    args = _ns(
        remediation_json=str(_write_json(tmp_path / "r.json", {"ok": True})),
        evaluation_json=str(_write_json(tmp_path / "e.json", {"ok": True})),
        authorization_json=None,
        authorization_cid=_cid("auth"),
        campaign_receipt_json=str(_write_json(tmp_path / "c.json", {"ok": True})),
        promotion_signature_json=str(_write_json(tmp_path / "s.json", {"ok": True})),
        seal_evidence_cid=_cid("seal"),
        operation_id="op-1",
        authorize_promote=False,
        seal_status=None,
        workspace=None,
        expected_generation=None,
        expected_policy_cid=None,
        promoted_policy_cid=None,
        promoted_policy_version=None,
        base_policy_cid=None,
        base_policy_version=None,
        repository_state_cid=None,
        repository_id=None,
    )
    with pytest.raises(assurance_cli.AssuranceCLIAuthorityError) as exc:
        analysis.handle_promote(args, api=_FakeApi())
    assert exc.value.reason_code == "promote_authority_required"
    assert exc.value.exit_code == assurance_cli.EXIT_AUTHORITY


def test_promote_requires_authorization_binding(tmp_path: Path) -> None:
    args = _ns(
        remediation_json=str(_write_json(tmp_path / "r.json", {"ok": True})),
        evaluation_json=str(_write_json(tmp_path / "e.json", {"ok": True})),
        authorization_json=None,
        authorization_cid=None,
        campaign_receipt_json=str(_write_json(tmp_path / "c.json", {"ok": True})),
        promotion_signature_json=str(_write_json(tmp_path / "s.json", {"ok": True})),
        seal_evidence_cid=_cid("seal"),
        operation_id="op-1",
        authorize_promote=True,
        seal_status=None,
        workspace=None,
        expected_generation=None,
        expected_policy_cid=None,
        promoted_policy_cid=None,
        promoted_policy_version=None,
        base_policy_cid=None,
        base_policy_version=None,
        repository_state_cid=None,
        repository_id=None,
    )
    with pytest.raises(assurance_cli.AssuranceCLIUsageError) as exc:
        analysis.handle_promote(args, api=_FakeApi())
    assert exc.value.reason_code == "missing_authorization"


def test_promote_rejects_self_promotion_authorization(tmp_path: Path) -> None:
    cand = _cid("cand-self")
    args = _ns(
        remediation_json=str(
            _write_json(
                tmp_path / "r.json",
                {"candidate_cid": cand, "plan_cid": _cid("plan")},
            )
        ),
        evaluation_json=str(
            _write_json(tmp_path / "e.json", {"evaluation_report_cid": _cid("eval")})
        ),
        authorization_json=None,
        authorization_cid=cand,  # same as candidate → self-promote
        campaign_receipt_json=str(_write_json(tmp_path / "c.json", {"ok": True})),
        promotion_signature_json=str(_write_json(tmp_path / "s.json", {"ok": True})),
        seal_evidence_cid=_cid("seal"),
        operation_id="op-self",
        authorize_promote=True,
        seal_status=None,
        workspace=None,
        expected_generation=None,
        expected_policy_cid=None,
        promoted_policy_cid=None,
        promoted_policy_version=None,
        base_policy_cid=None,
        base_policy_version=None,
        repository_state_cid=None,
        repository_id=None,
    )
    with pytest.raises(assurance_cli.AssuranceCLIAuthorityError) as exc:
        analysis.handle_promote(args, api=_FakeApi())
    assert exc.value.reason_code == "self_promotion_forbidden"


def test_promote_without_repository_is_honest_non_authority(tmp_path: Path) -> None:
    api = _FakeApi()
    auth = _cid("auth-external")
    args = _ns(
        remediation_json=str(
            _write_json(
                tmp_path / "r.json",
                {
                    "candidate_cid": _cid("cand"),
                    "proposed_policy_cid": _cid("policy-v2"),
                },
            )
        ),
        evaluation_json=str(
            _write_json(
                tmp_path / "e.json",
                {
                    "evaluation_report_cid": _cid("eval"),
                    "held_out_result": "pass",
                    "qualified": True,
                },
            )
        ),
        authorization_json=None,
        authorization_cid=auth,
        campaign_receipt_json=str(
            _write_json(
                tmp_path / "c.json",
                {"receipt_cid": _cid("campaign-receipt")},
            )
        ),
        promotion_signature_json=str(
            _write_json(
                tmp_path / "s.json",
                {
                    "signer_id": "operator",
                    "key_id": "key-1",
                    "audience": "adversarial_assurance.store",
                    "action": "promote_policy",
                    "verification_status": "verified",
                },
            )
        ),
        seal_evidence_cid=_cid("seal"),
        operation_id="op-no-repo",
        authorize_promote=True,
        seal_status=None,
        workspace=None,
        expected_generation=None,
        expected_policy_cid=None,
        promoted_policy_cid=None,
        promoted_policy_version=None,
        base_policy_cid=None,
        base_policy_version=None,
        repository_state_cid=None,
        repository_id=None,
    )
    result = analysis.handle_promote(args, api=api)
    assert result["authorized"] is True
    assert result["head_mutated"] is False
    assert result["status"] == "rejected"
    assert result["authority_status"] == analysis.AUTHORITY_NOT_AUTHORITY
    assert result["production_policy_change"] is False
    assert "missing_policy_repository" in (result["blocking_reasons"] or [])
    assert len(api.promote_calls) == 1
    assert api.promote_calls[0][1]["metadata"]["cli_authorize_promote"] is True


def test_promote_with_repository_can_claim_authority_only_on_head_mutation(
    tmp_path: Path,
) -> None:
    api = _FakeApi()
    repo = object()  # presence is enough for the fake
    args = _ns(
        remediation_json=str(
            _write_json(tmp_path / "r.json", {"candidate_cid": _cid("cand")})
        ),
        evaluation_json=str(
            _write_json(tmp_path / "e.json", {"evaluation_report_cid": _cid("eval")})
        ),
        authorization_json=None,
        authorization_cid=_cid("auth-external"),
        campaign_receipt_json=str(_write_json(tmp_path / "c.json", {"ok": True})),
        promotion_signature_json=str(
            _write_json(
                tmp_path / "s.json",
                {
                    "signer_id": "operator",
                    "key_id": "key-1",
                    "audience": "adversarial_assurance.store",
                    "action": "promote_policy",
                    "verification_status": "verified",
                },
            )
        ),
        seal_evidence_cid=_cid("seal"),
        operation_id="op-promote",
        authorize_promote=True,
        seal_status="released",
        workspace="default",
        expected_generation=1,
        expected_policy_cid=_cid("policy-v1"),
        promoted_policy_cid=_cid("policy-v2"),
        promoted_policy_version="2",
        base_policy_cid=_cid("policy-v1"),
        base_policy_version="1",
        repository_state_cid=REPO_STATE,
        repository_id=REPO_ID,
    )
    result = analysis.handle_promote(args, api=api, policy_repository=repo)
    assert result["status"] == "promoted"
    assert result["head_mutated"] is True
    assert result["authority_status"] == analysis.AUTHORITY_AUTHORITY
    assert result["production_policy_change"] is False


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def test_benchmark_returns_honest_unavailable_without_surface(
    tmp_path: Path,
) -> None:
    args = _ns(
        campaign_result_json=str(
            _write_json(
                tmp_path / "campaign.json",
                {
                    "plan_id": "plan-1",
                    "killed_count": 1,
                    "survivor_count": 0,
                    "terminal_status": "complete",
                },
            )
        ),
        metrics_json=None,
    )
    result = analysis.handle_benchmark(args)
    assert result["api"] == "benchmark_assurance"
    assert result["status"] == "unavailable"
    assert result["available"] is False
    assert result["fabricated_pass"] is False
    assert result["metrics_available"] is False
    assert result["authority_status"] == analysis.AUTHORITY_NOT_AUTHORITY
    assert result["production_policy_change"] is False
    assert result["network_service"] is False
    assert result["reason_code"] == "benchmark_unavailable"


def test_benchmark_with_injected_runner_projects_result(tmp_path: Path) -> None:
    def _runner(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "complete",
            "available": True,
            "metrics_available": True,
            "economics_available": True,
            "cases": [{"case_id": "c1", "status": "pass"}],
            "fabricated_pass": False,
            "production_policy_changed": False,
            "authority_status": analysis.AUTHORITY_NOT_AUTHORITY,
        }

    args = _ns(
        campaign_result_json=str(
            _write_json(tmp_path / "c.json", {"plan_id": "p"})
        ),
        metrics_json=None,
    )
    result = analysis.handle_benchmark(args, benchmark_runner=_runner)
    assert result["status"] == "complete"
    assert result["available"] is True
    assert result["fabricated_pass"] is False
    assert result["authority_status"] == analysis.AUTHORITY_NOT_AUTHORITY


def test_benchmark_runner_failure_is_inconclusive(tmp_path: Path) -> None:
    def _boom(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("economics not sealed")

    args = _ns(campaign_result_json=None, metrics_json=None)
    result = analysis.handle_benchmark(args, benchmark_runner=_boom)
    assert result["status"] == "inconclusive"
    assert result["available"] is False
    assert result["fabricated_pass"] is False
    assert result["authority_status"] == analysis.AUTHORITY_INCONCLUSIVE


def test_benchmark_rejects_host_path(tmp_path: Path) -> None:
    args = _ns(
        campaign_result_json=str(
            _write_json(
                tmp_path / "c.json",
                {"repo_root": "/home/other/external"},
            )
        ),
        metrics_json=None,
    )
    with pytest.raises(assurance_cli.AssuranceCLIPathError):
        analysis.handle_benchmark(args)


# ---------------------------------------------------------------------------
# Cancellation / end-to-end dispatch
# ---------------------------------------------------------------------------


def test_cancellation_flag_short_circuits_dispatch() -> None:
    out = io.StringIO()
    args = _ns(
        assurance_command="benchmark",
        campaign_result_json=None,
        metrics_json=None,
        cancel=True,
    )
    code = analysis.run_assurance_analysis_cli(args, stdout=out)
    assert code == assurance_cli.EXIT_CANCELLED
    payload = json.loads(out.getvalue())
    assert payload["ok"] is False
    assert payload["reason_code"] == "cancelled"


def test_end_to_end_promote_authority_gate_json_envelope(tmp_path: Path) -> None:
    out = io.StringIO()
    code = analysis.main(
        [
            "assurance",
            "promote",
            "--remediation-json",
            str(_write_json(tmp_path / "r.json", {"candidate_cid": _cid("c")})),
            "--evaluation-json",
            str(_write_json(tmp_path / "e.json", {"evaluation_report_cid": _cid("e")})),
            "--authorization-cid",
            _cid("auth"),
            "--campaign-receipt-json",
            str(_write_json(tmp_path / "c.json", {"receipt_cid": _cid("cr")})),
            "--promotion-signature-json",
            str(
                _write_json(
                    tmp_path / "s.json",
                    {
                        "signer_id": "operator",
                        "key_id": "k",
                        "audience": "adversarial_assurance.store",
                        "action": "promote_policy",
                        "verification_status": "verified",
                    },
                )
            ),
            "--seal-evidence-cid",
            _cid("seal"),
            "--operation-id",
            "op-e2e",
            # intentionally omit --authorize-promote
        ],
        stdout=out,
        api=_FakeApi(),
    )
    assert code == assurance_cli.EXIT_AUTHORITY
    payload = json.loads(out.getvalue())
    assert payload["ok"] is False
    assert payload["reason_code"] == "promote_authority_required"
    assert payload["production_policy_change"] is False
    assert payload["side_effects"]["network"] is False


def test_end_to_end_benchmark_unavailable_exit_code() -> None:
    out = io.StringIO()
    code = analysis.main(
        ["assurance", "benchmark"],
        stdout=out,
        api=_FakeApi(),
    )
    assert code == assurance_cli.EXIT_UNAVAILABLE
    payload = json.loads(out.getvalue())
    assert payload["ok"] is False
    assert payload["command"] == "benchmark"
    assert payload["status"] == "unavailable"
    assert payload["result"]["fabricated_pass"] is False
    assert payload["result"]["authority_status"] == analysis.AUTHORITY_NOT_AUTHORITY


def test_end_to_end_remediate_preserves_candidate(tmp_path: Path) -> None:
    out = io.StringIO()
    code = analysis.main(
        [
            "assurance",
            "remediate",
            "--surviving-mutant-json",
            str(
                _write_json(
                    tmp_path / "s.json",
                    {"candidate_cid": _cid("c"), "risk_class": "high"},
                )
            ),
            "--assurance-gap-json",
            str(
                _write_json(
                    tmp_path / "g.json",
                    {"gap_cid": _cid("g"), "gap_class": "missing_test"},
                )
            ),
        ],
        stdout=out,
        api=_FakeApi(),
    )
    assert code == assurance_cli.EXIT_SUCCESS
    payload = json.loads(out.getvalue())
    assert payload["ok"] is True
    assert payload["result"]["authority_status"] == analysis.AUTHORITY_CANDIDATE
    assert payload["result"]["candidate_status"] == "heuristic_candidate"
    assert payload["production_policy_change"] is False


def test_no_network_or_process_side_effects_in_envelope(tmp_path: Path) -> None:
    out = io.StringIO()
    analysis.main(
        ["assurance", "benchmark"],
        stdout=out,
        api=_FakeApi(),
    )
    payload = json.loads(out.getvalue())
    effects = payload["side_effects"]
    assert effects["network"] is False
    assert effects["process_spawn"] is False
    assert effects["key_generation"] is False
    assert effects["production_policy_change"] is False
    assert payload["path_exposure"] is False
