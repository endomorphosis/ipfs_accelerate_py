"""Tests for SCG-028 governor verification bridge.

Acceptance criteria enforced here:

* Patch / model / one-test / receipt / aggregate presence cannot accept.
* Missing task-class policy or any required check fails closed.
* Stale / simulated / unavailable evidence remains nonaccepting.
* Unknown check mappings are rejected.
* Selected/full/proof conflict signals bind without upgrading statuses.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_governor.verification import (
    AUDIT_VERIFICATION_EVIDENCE_INTERFACE,
    CHECK_FULL_SUITE,
    CHECK_HUMAN_REVIEW,
    CHECK_SELECTED_TESTS,
    CHECK_STATIC,
    CHECK_TYPE,
    ConflictSignal,
    GOVERNOR_VERIFICATION_BRIDGE_INTERFACE,
    KNOWN_CHECK_KINDS,
    PresenceClaim,
    SCG_VERIFICATION_BRIDGE_EVIDENCE,
    AuditVerificationEvidence,
    GovernorVerificationBridge,
    GovernorVerificationBridgeError,
    build_audit_verification_evidence,
    reject_unknown_check_kinds,
    required_check_kinds,
    resolve_task_class_acceptance,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    TerminalStatus,
    TestReceipt,
    VerificationReceiptKind,
)
from ipfs_accelerate_py.agent_supervisor.verification.executor import (
    CheckRunOutcome,
    execute_verification_plan,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.policy_contracts import (
    TaskClassAcceptanceRequirements,
)
from test.api.test_agent_supervisor_verification_contracts import (
    _key,
    _observation,
    _plan,
    _route,
)
from test.api.test_agent_supervisor_verification_executor import (
    _passing,
    _plan_for_keys,
    _status_receipt,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_governor/verification.py"
)


# ---------------------------------------------------------------------------
# Fixtures / recipes
# ---------------------------------------------------------------------------


def _acceptance(
    *,
    task_class: str = "implementation",
    risk_class: str = "medium",
    selected: bool = True,
    full: bool = False,
    static: bool = False,
    type_checks: bool = True,
    proofs: bool = False,
    review: bool = False,
) -> TaskClassAcceptanceRequirements:
    return TaskClassAcceptanceRequirements(
        task_class=task_class,
        risk_class=risk_class,
        require_selected_tests=selected,
        require_full_suite_fallback=full,
        require_static_checks=static,
        require_type_checks=type_checks,
        require_proofs=proofs,
        require_human_review=review,
    )


def _type_key():
    return _key(VerificationReceiptKind.TYPE_CHECK)


def _test_key(**changes: object):
    return _key(VerificationReceiptKind.TEST, **changes)


def _static_key():
    return _key(VerificationReceiptKind.STATIC_ANALYSIS)


def _runner_for(outcomes: dict[str, TerminalStatus | CheckRunOutcome]):
    def _runner(key, **_kwargs):
        value = outcomes.get(key.key_id)
        if value is None:
            return CheckRunOutcome(
                receipt=_passing(key, label="default"),
                publication_allowed=True,
            )
        if isinstance(value, TerminalStatus):
            receipt = _status_receipt(key, value, label=value.value)
            return CheckRunOutcome(
                receipt=receipt,
                publication_allowed=value
                in {TerminalStatus.PASSED, TerminalStatus.PROVED},
                timed_out=value is TerminalStatus.TIMEOUT,
                unavailable=value is TerminalStatus.UNAVAILABLE,
                cancelled=value is TerminalStatus.CANCELLED,
                reason_codes=(value.value,),
            )
        return value

    return _runner


def _execute(plan, outcomes: dict[str, TerminalStatus] | None = None, **kwargs):
    kwargs.setdefault("require_resource_lease", False)
    kwargs.setdefault("model_route_decision", _route())
    kwargs.setdefault("minimize_failures", True)
    if outcomes is not None:
        kwargs["check_runner"] = _runner_for(outcomes)
    elif "check_runner" not in kwargs:

        def _runner(key, **_kw):
            return CheckRunOutcome(
                receipt=_passing(key),
                publication_allowed=True,
            )

        kwargs["check_runner"] = _runner
    return execute_verification_plan(plan, **kwargs)


def _matrix_plan(
    *keys,
    full_suite_keys=(),
    human_review: bool = False,
    affected_tests: tuple[str, ...] | None = None,
):
    """Plan covering provided keys with optional full-suite designation."""

    all_keys = tuple(keys) + tuple(full_suite_keys)
    base = _plan_for_keys(*all_keys)
    tests = tuple(
        f"test_{index}"
        for index, key in enumerate(all_keys)
        if key.receipt_kind is VerificationReceiptKind.TEST
    )
    static = tuple(
        "src/example.py"
        for key in all_keys
        if key.receipt_kind is VerificationReceiptKind.STATIC_ANALYSIS
    )
    types = tuple(
        "src/example.py"
        for key in all_keys
        if key.receipt_kind is VerificationReceiptKind.TYPE_CHECK
    )
    full_ids = tuple(key.key_id for key in full_suite_keys)
    return replace(
        base,
        affected_tests=affected_tests if affected_tests is not None else tests,
        required_static_checks=static,
        required_type_checks=types or ("src/example.py",),
        full_suite_receipt_key_cids=full_ids,
        full_suite_required=bool(full_ids),
        full_suite_reason_codes=("policy_full_suite",) if full_ids else (),
        human_review_required=human_review,
        human_review_reason_codes=("policy_review",) if human_review else (),
    )


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def test_module_surface_and_constants() -> None:
    bridge = GovernorVerificationBridge()
    assert bridge.INTERFACE == GOVERNOR_VERIFICATION_BRIDGE_INTERFACE
    assert bridge.EVIDENCE == SCG_VERIFICATION_BRIDGE_EVIDENCE
    assert callable(build_audit_verification_evidence)
    assert CHECK_SELECTED_TESTS in KNOWN_CHECK_KINDS
    assert CHECK_HUMAN_REVIEW in KNOWN_CHECK_KINDS
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert "GovernorVerificationBridge" in names
    assert "build_audit_verification_evidence" in names


def test_unknown_check_mapping_rejected() -> None:
    with pytest.raises(GovernorVerificationBridgeError, match="unknown"):
        reject_unknown_check_kinds(["selected_tests", "magic_oracle"])
    assert reject_unknown_check_kinds(["selected_tests", "type_checks"]) == (
        CHECK_SELECTED_TESTS,
        CHECK_TYPE,
    )


def test_missing_task_class_policy_fails_closed() -> None:
    type_key = _type_key()
    plan = _matrix_plan(type_key)
    result = _execute(plan)
    evidence = build_audit_verification_evidence(
        task_class="implementation",
        risk_class="medium",
        execution_result=result,
        # no policy / acceptance row
    )
    assert evidence.production_acceptance is False
    assert evidence.production_eligible is False
    assert evidence.acceptance_matrix_satisfied is False
    assert "absent_or_unknown_task_class_mapping" in evidence.reason_codes


def test_mismatched_acceptance_row_is_unknown() -> None:
    row = _acceptance(task_class="other", risk_class="high")
    resolved = resolve_task_class_acceptance(
        task_class="implementation",
        risk_class="medium",
        acceptance_requirements=row,
    )
    assert resolved is None


# ---------------------------------------------------------------------------
# Presence claims cannot accept
# ---------------------------------------------------------------------------


def test_presence_claims_cannot_accept_without_matrix() -> None:
    type_key = _type_key()
    plan = _matrix_plan(type_key)
    result = _execute(plan)
    # Green IVP leaves but no task-class policy + only presence claims.
    evidence = build_audit_verification_evidence(
        task_class="implementation",
        risk_class="medium",
        execution_result=result,
        presence_claims={
            "patch_cid": "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "model_route": "small_local_model",
            "one_test_passed": True,
            "receipt_present": True,
            "aggregate_passed": True,
        },
    )
    assert evidence.production_acceptance is False
    assert PresenceClaim.PATCH.value in evidence.presence_claims_observed
    assert PresenceClaim.MODEL.value in evidence.presence_claims_observed
    assert PresenceClaim.ONE_TEST.value in evidence.presence_claims_observed
    assert PresenceClaim.RECEIPT.value in evidence.presence_claims_observed
    assert PresenceClaim.AGGREGATE.value in evidence.presence_claims_observed
    assert "presence_claims_cannot_accept" in evidence.reason_codes


def test_aggregate_or_one_test_alone_cannot_accept() -> None:
    """A single type-check pass cannot satisfy selected-test requirement."""

    type_key = _type_key()
    plan = _matrix_plan(type_key)
    result = _execute(plan)
    # Policy requires selected tests, but plan only has type check.
    acceptance = _acceptance(selected=True, type_checks=True, full=False, static=False)
    evidence = build_audit_verification_evidence(
        task_class="implementation",
        risk_class="medium",
        execution_result=result,
        acceptance_requirements=acceptance,
        presence_claims=["one_test", "receipt", "aggregate"],
    )
    assert evidence.production_acceptance is False
    assert CHECK_SELECTED_TESTS in evidence.missing_checks or any(
        code.startswith("plan_missing:") or code.startswith("missing:")
        for code in evidence.reason_codes
    )


# ---------------------------------------------------------------------------
# Happy path with full matrix satisfaction
# ---------------------------------------------------------------------------


def test_happy_path_type_only_matrix_accepts() -> None:
    type_key = _type_key()
    plan = _matrix_plan(type_key)
    result = _execute(plan)
    acceptance = _acceptance(
        selected=False,
        full=False,
        static=False,
        type_checks=True,
        proofs=False,
        review=False,
    )
    evidence = build_audit_verification_evidence(
        task_class="implementation",
        risk_class="medium",
        execution_result=result,
        acceptance_requirements=acceptance,
    )
    assert evidence.production_acceptance is True
    assert evidence.acceptance_matrix_satisfied is True
    assert evidence.production_eligible is True
    assert CHECK_TYPE in evidence.satisfied_checks
    assert evidence.verification.production_eligible is True
    assert evidence.verification.acceptance_matrix_satisfied is True
    assert evidence.verification.verification_bundle_cid == evidence.verification_bundle_cid
    assert isinstance(evidence, AuditVerificationEvidence)
    assert evidence.INTERFACE == AUDIT_VERIFICATION_EVIDENCE_INTERFACE
    # Round-trip identity.
    payload = evidence.to_dict()
    assert payload["production_acceptance"] is True
    assert payload["evidence_cid"] == evidence.evidence_cid


def test_open_bundle_path_matches_execution() -> None:
    type_key = _type_key()
    plan = _matrix_plan(type_key)
    result = _execute(plan)
    acceptance = _acceptance(
        selected=False, full=False, static=False, type_checks=True
    )
    via_result = build_audit_verification_evidence(
        task_class="implementation",
        risk_class="medium",
        execution_result=result,
        acceptance_requirements=acceptance,
    )
    via_bundle = build_audit_verification_evidence(
        task_class="implementation",
        risk_class="medium",
        verification_bundle=result.bundle,
        acceptance_requirements=acceptance,
        model_route_decision=result.model_route_decision,
    )
    assert via_result.production_acceptance == via_bundle.production_acceptance
    assert via_result.verification_bundle_cid == via_bundle.verification_bundle_cid


def test_run_plan_via_bridge() -> None:
    type_key = _type_key()
    plan = _matrix_plan(type_key)
    acceptance = _acceptance(
        selected=False, full=False, static=False, type_checks=True
    )
    bridge = GovernorVerificationBridge(minimize_failures=False)
    evidence = bridge.run_plan(
        plan,
        task_class="implementation",
        risk_class="medium",
        acceptance_requirements=acceptance,
        check_runner=lambda key, **_kw: CheckRunOutcome(
            receipt=_passing(key), publication_allowed=True
        ),
        model_route_decision=_route(),
    )
    assert evidence.production_acceptance is True


# ---------------------------------------------------------------------------
# Required checks fail closed
# ---------------------------------------------------------------------------


def test_missing_required_check_fails_closed() -> None:
    type_key = _type_key()
    plan = _matrix_plan(type_key)
    result = _execute(plan)
    acceptance = _acceptance(
        selected=False,
        full=False,
        static=True,  # required but not in plan
        type_checks=True,
    )
    evidence = build_audit_verification_evidence(
        task_class="implementation",
        risk_class="medium",
        execution_result=result,
        acceptance_requirements=acceptance,
    )
    assert evidence.production_acceptance is False
    assert CHECK_STATIC in evidence.missing_checks or any(
        "static" in code for code in evidence.reason_codes
    )


def test_failed_required_check_fails_closed() -> None:
    type_key = _type_key()
    plan = _matrix_plan(type_key)
    result = _execute(plan, outcomes={type_key.key_id: TerminalStatus.FAILED})
    acceptance = _acceptance(
        selected=False, full=False, static=False, type_checks=True
    )
    evidence = build_audit_verification_evidence(
        task_class="implementation",
        risk_class="medium",
        execution_result=result,
        acceptance_requirements=acceptance,
    )
    assert evidence.production_acceptance is False
    assert CHECK_TYPE in evidence.failed_checks
    assert evidence.counterexamples  # minimized failure evidence bound


@pytest.mark.parametrize(
    "status",
    [
        TerminalStatus.STALE,
        TerminalStatus.SIMULATED,
        TerminalStatus.UNAVAILABLE,
        TerminalStatus.TIMEOUT,
        TerminalStatus.CANCELLED,
        TerminalStatus.INVALID,
    ],
)
def test_non_production_terminals_remain_nonaccepting(status: TerminalStatus) -> None:
    type_key = _type_key()
    plan = _matrix_plan(type_key)
    result = _execute(plan, outcomes={type_key.key_id: status})
    acceptance = _acceptance(
        selected=False, full=False, static=False, type_checks=True
    )
    evidence = build_audit_verification_evidence(
        task_class="implementation",
        risk_class="medium",
        execution_result=result,
        acceptance_requirements=acceptance,
    )
    assert evidence.production_acceptance is False
    assert evidence.production_eligible is False
    assert (
        "non_production_terminal_present" in evidence.reason_codes
        or "ivp_production_acceptance_false" in evidence.reason_codes
        or "failed_required_checks" in evidence.reason_codes
        or any(code.startswith("failed:") for code in evidence.reason_codes)
    )


def test_human_review_required_never_accepts() -> None:
    type_key = _type_key()
    plan = _matrix_plan(type_key, human_review=True)
    result = _execute(plan)
    acceptance = _acceptance(
        selected=False,
        full=False,
        static=False,
        type_checks=True,
        review=True,
    )
    evidence = build_audit_verification_evidence(
        task_class="implementation",
        risk_class="medium",
        execution_result=result,
        acceptance_requirements=acceptance,
    )
    assert evidence.production_acceptance is False
    assert ConflictSignal.HUMAN_REVIEW_REQUIRED.value in evidence.conflict_signals
    assert "human_review_required" in evidence.reason_codes


# ---------------------------------------------------------------------------
# Selected / full suite conflicts
# ---------------------------------------------------------------------------


def test_selected_pass_full_fail_conflict_blocks_acceptance() -> None:
    selected = _test_key()
    full_argv = (
        "/usr/bin/python3.12",
        "-m",
        "pytest",
        "tests/test_full_suite.py",
    )
    # Distinct full-suite key via selector identity (tool name stays pytest).
    full = _key(
        VerificationReceiptKind.TEST,
        selector_argv=full_argv,
    )
    plan = _matrix_plan(selected, full_suite_keys=(full,))

    def runner(key, **_kwargs):
        if key.key_id == full.key_id:
            receipt = TestReceipt(
                full,
                _observation(
                    full,
                    TerminalStatus.FAILED,
                    label="full-fail",
                    command_argv=full_argv,
                ),
            )
            return CheckRunOutcome(
                receipt=receipt,
                publication_allowed=False,
                reason_codes=("failed",),
            )
        return CheckRunOutcome(
            receipt=_passing(key, label="selected-pass"),
            publication_allowed=True,
        )

    result = _execute(plan, check_runner=runner)
    acceptance = _acceptance(
        selected=True, full=True, static=False, type_checks=False
    )
    evidence = build_audit_verification_evidence(
        task_class="implementation",
        risk_class="medium",
        execution_result=result,
        acceptance_requirements=acceptance,
    )
    assert evidence.production_acceptance is False
    assert ConflictSignal.SELECTED_PASS_FULL_FAIL.value in evidence.conflict_signals
    assert (
        ConflictSignal.SELECTED_FULL_OUTCOME_DISCREPANCY.value
        in evidence.conflict_signals
    )
    assert evidence.counterexamples


def test_full_suite_pending_blocks_acceptance() -> None:
    selected = _test_key()
    full = _key(
        VerificationReceiptKind.TEST,
        selector_argv=(
            "/usr/bin/python3.12",
            "-m",
            "pytest",
            "tests/test_full_suite.py",
        ),
    )
    # Plan requires full suite key but we only execute selected by opening a
    # bundle that omits the full-suite receipt (simulate incomplete evidence).
    plan = _matrix_plan(selected, full_suite_keys=(full,))
    from ipfs_accelerate_py.agent_supervisor.verification.bundle import (
        build_verification_bundle,
    )

    selected_receipt = TestReceipt(
        selected, _observation(selected, TerminalStatus.PASSED, label="sel")
    )
    # Bundle without full-suite receipt → mandatory_fallback_pending / unresolved.
    incomplete = build_verification_bundle(
        plan,
        executed_receipts=(selected_receipt,),
        human_review_required=False,
    )
    acceptance = _acceptance(
        selected=True, full=True, static=False, type_checks=False
    )
    evidence = build_audit_verification_evidence(
        task_class="implementation",
        risk_class="medium",
        verification_bundle=incomplete,
        acceptance_requirements=acceptance,
    )
    assert evidence.production_acceptance is False
    assert (
        ConflictSignal.FULL_SUITE_PENDING.value in evidence.conflict_signals
        or CHECK_FULL_SUITE in evidence.missing_checks
        or "mandatory_full_suite_fallback_pending" in evidence.reason_codes
        or "ivp_production_acceptance_false" in evidence.reason_codes
    )


# ---------------------------------------------------------------------------
# Status never upgraded
# ---------------------------------------------------------------------------


def test_bridge_never_upgrades_failed_receipt_status() -> None:
    type_key = _type_key()
    plan = _matrix_plan(type_key)
    result = _execute(plan, outcomes={type_key.key_id: TerminalStatus.FAILED})
    acceptance = _acceptance(
        selected=False, full=False, static=False, type_checks=True
    )
    evidence = build_audit_verification_evidence(
        task_class="implementation",
        risk_class="medium",
        execution_result=result,
        acceptance_requirements=acceptance,
    )
    assert result.bundle.receipts[0].status is TerminalStatus.FAILED
    assert evidence.production_acceptance is False
    # Projection does not claim type checks passed.
    assert evidence.verification.static_checks_passed is None
    # type checks present but failed → False
    # selected empty → None
    assert evidence.verification.selected_tests_passed is None


def test_required_check_kinds_order() -> None:
    row = _acceptance(
        selected=True,
        full=True,
        static=True,
        type_checks=True,
        proofs=True,
        review=True,
    )
    assert required_check_kinds(row) == (
        CHECK_SELECTED_TESTS,
        CHECK_FULL_SUITE,
        CHECK_STATIC,
        CHECK_TYPE,
        "proofs",
        CHECK_HUMAN_REVIEW,
    )


def test_missing_source_raises() -> None:
    with pytest.raises(GovernorVerificationBridgeError, match="required"):
        build_audit_verification_evidence(
            task_class="implementation",
            risk_class="medium",
            acceptance_requirements=_acceptance(selected=False, type_checks=True),
        )


def test_declared_unknown_checks_on_open_rejected() -> None:
    type_key = _type_key()
    plan = _matrix_plan(type_key)
    result = _execute(plan)
    with pytest.raises(GovernorVerificationBridgeError, match="unknown"):
        build_audit_verification_evidence(
            task_class="implementation",
            risk_class="medium",
            verification_bundle=result.bundle,
            acceptance_requirements=_acceptance(
                selected=False, type_checks=True, full=False, static=False
            ),
            declared_checks=["not_a_real_check"],
        )
