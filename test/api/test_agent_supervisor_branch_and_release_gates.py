"""PCPR-057: produce Accelerate branch and release gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.assurance.branch_and_release_gates import (
    BRANCH_POLICY_KIND,
    CLOSED_RELEASE_OUTCOMES,
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID,
    HERMETIC_CANDIDATE_SUITES,
    INTERFACE,
    OPERATOR_BLOCKING_TASK_ID,
    OutcomeProbe,
    PCPR_057_GOAL_ID,
    PCPR_057_TASK_ID,
    PINNED_BRANCH_POLICY_CID,
    PINNED_RELEASE_GATE_CID,
    RELEASE_GATE_KIND,
    REQUIRED_STATUS_CHECKS,
    SCHEMA,
    SEALED_PATH,
    SEALED_PYTHON,
    BranchAndReleaseGatesError,
    current_head_static_probes,
    pcpr_057_receipt_promotion,
    platform_gates_catalog,
    qualify_branch_and_release_gates,
    qualify_current_head_branch_and_release_gates,
    refuse_gate_remint,
    refuse_policy_remint,
    render_declared_branch_protection_policy,
    render_declared_release_gate,
    verify_branch_and_release_gate_files,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_closed_vocabularies_match_pcpr_057_requirements() -> None:
    assert PCPR_057_TASK_ID == "PCPR-057"
    assert PCPR_057_GOAL_ID == "PCPR-G600"
    assert INTERFACE == "AccelerateBranchAndReleaseGates@1"
    assert SCHEMA == "ipfs_accelerate_py/assurance/branch-and-release-gates@1"
    assert BRANCH_POLICY_KIND == "declared_branch_protection_policy"
    assert RELEASE_GATE_KIND == "declared_release_gate"
    assert OPERATOR_BLOCKING_TASK_ID == "pcpr-057-operator-github-governance"
    assert "pcpr.current-head-qualification" in REQUIRED_STATUS_CHECKS
    assert "pcpr.branch-and-release-gates" in REQUIRED_STATUS_CHECKS
    assert "release_candidate_qualified" in CLOSED_RELEASE_OUTCOMES
    assert "rnd_non_promoted" not in CLOSED_RELEASE_OUTCOMES
    assert HERMETIC_CANDIDATE_SUITES[-1].endswith(
        "test_agent_supervisor_branch_and_release_gates.py"
    )
    assert SEALED_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert SEALED_PYTHON == "/usr/bin/python3.12"
    policy = render_declared_branch_protection_policy()
    gate = render_declared_release_gate()
    assert policy["policy_kind"] == BRANCH_POLICY_KIND
    assert policy["applied"] is False
    assert policy["live"] is False
    assert policy["release_claim"] is False
    assert policy["closed_release_outcome"] is None
    assert policy["governance_gate_complete"] is False
    assert policy["source"]["mutable_main_reference"] is False
    assert policy["branch_policy_cid"] == PINNED_BRANCH_POLICY_CID
    assert policy["operator_blocking_task"]["status"] == "typed_blocked"
    assert gate["gate_kind"] == RELEASE_GATE_KIND
    assert gate["partial_required_build_failure_prohibits_release"] is True
    assert gate["release_gate_cid"] == PINNED_RELEASE_GATE_CID
    assert refuse_policy_remint(PINNED_BRANCH_POLICY_CID) == PINNED_BRANCH_POLICY_CID
    assert refuse_gate_remint(PINNED_RELEASE_GATE_CID) == PINNED_RELEASE_GATE_CID
    with pytest.raises(BranchAndReleaseGatesError, match="remints"):
        refuse_policy_remint(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )


def test_current_head_is_rnd_non_promoted_and_not_a_release() -> None:
    verdict = qualify_current_head_branch_and_release_gates()
    assert verdict.promotion_status == "rnd_non_promoted"
    assert verdict.supervisor_disposition == "supervisor_non_promoted"
    assert verdict.closed_release_outcome is None
    assert verdict.release_claim is False
    assert verdict.completion_authoritative is False
    assert verdict.contracts_frozen is False
    assert verdict.duckdb_or_quack_state_written is False
    assert verdict.sibling_source_required is False
    assert verdict.hashes_invented is False
    assert verdict.signatures_invented is False
    assert verdict.live_branch_protection is False
    assert verdict.live_branch_protection_evidence_kind == "unavailable"
    assert verdict.live_tag_protection is False
    assert verdict.live_github_admin is False
    assert verdict.governance_gate_complete is False
    assert verdict.operator_blocking_task == OPERATOR_BLOCKING_TASK_ID
    assert verdict.mutable_main_reference is False
    assert verdict.simulated_results_represented_as_live is False
    assert verdict.this_task_created_competing_authority is False
    assert verdict.promotion_status not in CLOSED_RELEASE_OUTCOMES
    assert verdict.verdict_cid.startswith("baguqeera")
    assert verdict.verdict_cid == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID
    assert verdict.branch_policy_cid == PINNED_BRANCH_POLICY_CID
    assert verdict.release_gate_cid == PINNED_RELEASE_GATE_CID
    assert verdict.blockers == ()
    section = pcpr_057_receipt_promotion(verdict)
    assert section["promotion_status"] == "rnd_non_promoted"
    assert section["closed_release_outcome"] is None
    assert section["release_claim"] is False
    assert section["live_branch_protection"] is False
    assert section["governance_gate_complete"] is False
    assert section["verdict_cid"] == CURRENT_HEAD_NON_PROMOTION_VERDICT_CID


def test_static_probes_show_declared_gate_constraints() -> None:
    probes = {item.probe_id: item for item in current_head_static_probes()}
    assert probes["branch_policy_files_match_generator"].present is True
    assert probes["release_gate_files_match_generator"].present is True
    assert probes["pyproject_branch_and_release_gates_table"].present is True
    assert probes["required_status_checks_declared"].present is True
    assert probes["partial_required_build_failure_prohibits_release"].present is True
    assert probes["no_mutable_main_reference"].present is True
    assert probes["branch_policy_kind_is_declared"].present is True
    assert probes["release_gate_kind_is_declared"].present is True
    assert probes["pcpr_056_lock_not_reminted"].present is True
    assert probes["governance_gate_is_not_complete"].present is True
    assert probes["operator_blocking_task_emitted"].present is True
    assert probes["named_git_extras_are_not_this_gate"].present is True
    assert probes["mutable_main_reference"].present is False
    assert probes["governance_gate_represented_as_complete"].present is False
    assert probes["live_branch_protection"].evidence_kind == "unavailable"
    assert probes["live_tag_protection"].evidence_kind == "unavailable"
    for probe in probes.values():
        assert probe.live is False
        assert probe.simulated_represented_as_live is False


def test_committed_files_match_generator() -> None:
    verified = verify_branch_and_release_gate_files()
    assert verified["ok"] is True
    assert verified["missing"] == []
    catalog = platform_gates_catalog()
    assert catalog["interface"] == "PlatformBranchAndReleaseGates@1"
    assert catalog["release_claim"] is False
    assert catalog["closed_release_outcome"] is None
    assert catalog["live_branch_protection"] is False
    assert catalog["governance_gate_complete"] is False
    assert catalog["branch_policy_cid"] == PINNED_BRANCH_POLICY_CID
    assert catalog["release_gate_cid"] == PINNED_RELEASE_GATE_CID
    assert catalog["components"]["ipfs_accelerate_py"]["status"] == "observed"
    assert catalog["components"]["ipfs_datasets_py"]["binding"]["status"] == "observed"
    assert catalog["components"]["ipfs_kit_py"]["binding"]["status"] == "observed"
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'interface = "AccelerateBranchAndReleaseGates@1"' in pyproject


def test_simulated_live_probe_is_rejected() -> None:
    with pytest.raises(BranchAndReleaseGatesError, match="simulated"):
        qualify_branch_and_release_gates(
            (
                OutcomeProbe(
                    probe_id="bogus",
                    present=False,
                    evidence_kind="simulated",
                    live=False,
                    simulated_represented_as_live=True,
                    reason="must fail",
                ),
            ),
            branch_policy_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            release_gate_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            catalog_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            lock_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )


def test_live_claim_without_measured_live_evidence_is_rejected() -> None:
    with pytest.raises(BranchAndReleaseGatesError, match="measured_live"):
        qualify_branch_and_release_gates(
            (
                OutcomeProbe(
                    probe_id="bogus",
                    present=True,
                    evidence_kind="measured",
                    live=True,
                    simulated_represented_as_live=False,
                    reason="must fail",
                ),
            ),
            branch_policy_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            release_gate_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            catalog_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            lock_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
