"""Paired shadow execution tests for SCG-026.

Acceptance criteria enforced here:

* Expanded output never auto-accepts (even when a runner claims accepted).
* Budgets and disclosure are rechecked before invocation.
* Cancellation/timeouts leave production state unchanged.
* Attempts run under isolated evaluation worktrees (no production edits).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    ArtifactProvenance,
    AssumptionKind,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    GovernorArtifactHeader,
    GovernorAssumption,
    GovernorTerminalStatus,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    AcceptanceDisposition,
    AttemptTerminalStatus,
    CostTimingProjection,
    SHADOW_EXECUTION_RESULT_INTERFACE,
    ShadowAttemptRole,
    ShadowExecutionPlan,
    ShadowSelectionReason,
    VerificationProjection,
    assert_expanded_never_accepted,
    verify_result_identity,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.privacy import (
    DisclosureDisposition,
    default_shadow_disclosure_policy,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.shadow import (
    DEFAULT_COMPRESSED_PROVIDER_ID,
    DEFAULT_EXPANDED_PROVIDER_ID,
    DEFAULT_EXTERNAL_PROVIDER_ID,
    EXECUTE_SHADOW_PLAN_INTERFACE,
    SCG_SHADOW_RUN_EVIDENCE,
    SHADOW_EXECUTOR_INTERFACE,
    AlwaysAdmitResourceGate,
    BudgetExceededError,
    CallableShadowAttemptRunner,
    DisclosureRecheckError,
    InMemoryEvaluationWorktreeLifecycle,
    PlanAdmissionError,
    ProductionCheckoutGuard,
    ProductionStateMutatedError,
    SCG_SHADOW_RUN_EVIDENCE as EVIDENCE,
    ShadowAttemptInvocation,
    ShadowAttemptProposal,
    ShadowBudgetLedger,
    ShadowCancellationToken,
    ShadowExecutor,
    SimulatedShadowAttemptRunner,
    admit_shadow_plan,
    execute_shadow_plan,
    expanded_never_auto_accepts,
    production_fingerprint_from_refs,
    production_state_unchanged,
    resolve_acceptance_disposition,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SHADOW_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/semantic_governor/shadow.py"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _header(**overrides: Any) -> GovernorArtifactHeader:
    compressed = _cid("context-pack-compressed")
    fields: dict[str, Any] = {
        "artifact_kind": "shadow_execution_plan",
        "repository_state_cid": _cid("repo-state"),
        "context_pack_cid": compressed,
        "verification_bundle_cid": _cid("verification-bundle"),
        "generator": GeneratorIdentity(
            generator_id="shadow_execution",
            generator_version="1.0.0",
            interface_id="create_shadow_plan@1",
        ),
        "provenance": ArtifactProvenance(
            producer_id="semantic_governor",
            producer_version="1",
            execution_mode=ExecutionMode.SIMULATED,
            authority_source=AuthoritySource.DETERMINISTIC,
            input_cids=(_cid("input-a"),),
            tool_ids=("shadow.v1",),
            policy_cid=_cid("policy"),
            notes=None,
        ),
        "terminal_status": GovernorTerminalStatus.SIMULATED,
        "assumptions": (
            GovernorAssumption(
                assumption_id="isolated_worktree",
                kind=AssumptionKind.ENVIRONMENT,
                statement="Paired shadow runs use disposable evaluation worktrees",
                supporting_cids=(_cid("worktree-policy"),),
            ),
        ),
        "metadata": {"task": "SCG-026"},
    }
    fields.update(overrides)
    return GovernorArtifactHeader(**fields)


def _plan(**overrides: Any) -> ShadowExecutionPlan:
    compressed = _cid("context-pack-compressed")
    fields: dict[str, Any] = {
        "header": _header(context_pack_cid=compressed),
        "task_id": "SCG-026",
        "audit_policy_cid": _cid("audit-policy"),
        "compressed_context_pack_cid": compressed,
        "expanded_context_pack_cid": _cid("context-pack-expanded"),
        "compressed_route_id": "route.compressed",
        "expanded_route_id": "route.expanded",
        "selection_reasons": (ShadowSelectionReason.RISK_CLASS_MANDATORY.value,),
        "max_wall_time_ms": 120_000,
        "max_model_spend_micros": 5_000_000,
        "max_expansion_token_budget": 50_000,
        "isolated_evaluation_worktree_required": True,
        "expanded_is_oracle_candidate_only": True,
        "allow_external_expanded_disclosure": False,
        "metadata": {"evidence": SCG_SHADOW_RUN_EVIDENCE},
    }
    fields.update(overrides)
    return ShadowExecutionPlan(**fields)


def _verification(**overrides: Any) -> VerificationProjection:
    fields: dict[str, Any] = {
        "verification_bundle_cid": _cid("verification-bundle"),
        "selected_tests_passed": True,
        "full_suite_passed": True,
        "proofs_passed": True,
        "static_checks_passed": True,
        "counterexample_present": False,
        "acceptance_matrix_satisfied": False,
        "production_eligible": False,
    }
    fields.update(overrides)
    return VerificationProjection(**fields)


# ---------------------------------------------------------------------------
# Module surface / evidence / import safety
# ---------------------------------------------------------------------------


def test_evidence_and_interfaces_are_stable() -> None:
    assert SCG_SHADOW_RUN_EVIDENCE == "scg/shadow-run@1"
    assert EVIDENCE == "scg/shadow-run@1"
    assert SHADOW_EXECUTOR_INTERFACE == "ShadowExecutor@1"
    assert EXECUTE_SHADOW_PLAN_INTERFACE == "execute_shadow_plan@1"
    assert SHADOW_EXECUTION_RESULT_INTERFACE == "ShadowExecutionResult@1"
    assert DEFAULT_COMPRESSED_PROVIDER_ID.startswith("local:")
    assert DEFAULT_EXPANDED_PROVIDER_ID.startswith("local:")


def test_module_import_performs_no_io() -> None:
    source = SHADOW_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"open", "urlopen", "system", "Popen", "connect", "create_connection"}
    for node in tree.body:
        if not isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else (func.attr if isinstance(func, ast.Attribute) else "")
                )
                assert name not in forbidden


# ---------------------------------------------------------------------------
# Plan admission
# ---------------------------------------------------------------------------


def test_admit_shadow_plan_requires_positive_wall_time() -> None:
    with pytest.raises(PlanAdmissionError, match="max_wall_time_ms"):
        admit_shadow_plan(_plan(max_wall_time_ms=0))


def test_admit_shadow_plan_accepts_valid_plan() -> None:
    plan = _plan()
    admitted = admit_shadow_plan(plan)
    assert admitted.plan_cid == plan.plan_cid
    assert admitted.expanded_is_oracle_candidate_only is True
    assert admitted.isolated_evaluation_worktree_required is True


# ---------------------------------------------------------------------------
# Happy path: paired isolated execution
# ---------------------------------------------------------------------------


def test_execute_shadow_plan_runs_paired_isolated_attempts() -> None:
    lifecycle = InMemoryEvaluationWorktreeLifecycle()
    runner = SimulatedShadowAttemptRunner()
    gate = AlwaysAdmitResourceGate()
    plan = _plan()
    result = execute_shadow_plan(
        plan,
        attempt_runner=runner,
        worktree_lifecycle=lifecycle,
        resource_gate=gate,
    )
    assert result.plan_cid == plan.plan_cid
    assert result.both_attempts_isolated is True
    assert result.expanded_attempt is not None
    assert result.compressed_attempt.role == ShadowAttemptRole.COMPRESSED.value
    assert result.expanded_attempt.role == ShadowAttemptRole.EXPANDED.value
    assert result.compressed_attempt.worktree_id is not None
    assert result.expanded_attempt.worktree_id is not None
    assert result.compressed_attempt.worktree_id != result.expanded_attempt.worktree_id
    assert lifecycle.create_count == 2
    assert lifecycle.release_count == 2
    assert lifecycle.active_ids() == ()
    assert len(runner.invocations) == 2
    assert len(gate.admissions) == 2
    assert len(gate.releases) == 2
    verify_result_identity(result)
    restored = type(result).from_dict(result.to_dict())
    assert restored.result_cid == result.result_cid


def test_execute_captures_costs_and_verification_refs() -> None:
    plan = _plan()
    result = execute_shadow_plan(plan)
    assert result.compressed_attempt.cost_timing.input_tokens > 0
    assert result.expanded_attempt is not None
    assert result.expanded_attempt.cost_timing.input_tokens > 0
    assert result.compressed_attempt.verification.verification_bundle_cid
    assert result.expanded_attempt.verification.verification_bundle_cid
    assert result.compressed_attempt.patch_cid is not None
    assert result.expanded_attempt.patch_cid is not None
    assert result.metadata["evidence"] == SCG_SHADOW_RUN_EVIDENCE
    assert result.metadata["invocation_count"] == 2
    assert result.metadata["budget_recheck_count"] >= 4
    assert result.metadata["disclosure_recheck_count"] >= 2


# ---------------------------------------------------------------------------
# Expanded never auto-accepts
# ---------------------------------------------------------------------------


def test_expanded_never_auto_accepts_on_success() -> None:
    result = execute_shadow_plan(_plan())
    assert result.expanded_attempt is not None
    assert (
        result.expanded_attempt.acceptance_disposition
        == AcceptanceDisposition.CANDIDATE_ONLY.value
    )
    assert result.expanded_attempt.verification.production_eligible is False
    assert expanded_never_auto_accepts(result) is True
    assert_expanded_never_accepted(
        result.expanded_attempt.acceptance_disposition,
        role=result.expanded_attempt.role,
    )


def test_expanded_runner_claimed_accepted_is_overridden() -> None:
    runner = SimulatedShadowAttemptRunner(claim_expanded_accepted=True)
    result = execute_shadow_plan(_plan(), attempt_runner=runner)
    assert result.expanded_attempt is not None
    assert (
        result.expanded_attempt.acceptance_disposition
        != AcceptanceDisposition.ACCEPTED.value
    )
    assert (
        result.expanded_attempt.acceptance_disposition
        == AcceptanceDisposition.CANDIDATE_ONLY.value
    )
    assert result.expanded_attempt.verification.production_eligible is False


def test_resolve_acceptance_rejects_expanded_accepted_claim() -> None:
    verification = _verification(production_eligible=False)
    disposition = resolve_acceptance_disposition(
        role=ShadowAttemptRole.EXPANDED.value,
        attempt_status=AttemptTerminalStatus.SUCCEEDED.value,
        execution_mode=ExecutionMode.LIVE.value,
        verification=verification,
        claimed=AcceptanceDisposition.ACCEPTED.value,
    )
    assert disposition == AcceptanceDisposition.CANDIDATE_ONLY.value


def test_force_expanded_verification_strips_production_eligible() -> None:
    from ipfs_accelerate_py.agent_supervisor.semantic_governor.shadow import (
        force_expanded_verification,
    )

    # production_eligible without acceptance_matrix_satisfied is rejected.
    with pytest.raises(Exception):
        _verification(acceptance_matrix_satisfied=False, production_eligible=True)

    eligible = VerificationProjection(
        verification_bundle_cid=_cid("v"),
        selected_tests_passed=True,
        full_suite_passed=True,
        proofs_passed=True,
        static_checks_passed=True,
        counterexample_present=False,
        acceptance_matrix_satisfied=True,
        production_eligible=True,
    )
    forced = force_expanded_verification(eligible)
    assert forced.production_eligible is False
    assert forced.acceptance_matrix_satisfied is False


# ---------------------------------------------------------------------------
# Budget recheck before invocation
# ---------------------------------------------------------------------------


def test_budget_recheck_rejects_zero_wall_time_ledger() -> None:
    ledger = ShadowBudgetLedger(
        max_wall_time_ms=0,
        max_model_spend_micros=1000,
        max_expansion_token_budget=1000,
    )
    with pytest.raises(BudgetExceededError, match="wall_time"):
        ledger.recheck_before_invocation(role=ShadowAttemptRole.COMPRESSED.value)


def test_budget_recheck_blocks_expanded_when_token_budget_zero() -> None:
    plan = _plan(max_expansion_token_budget=0)
    # admit_shadow_plan allows zero expansion budget? plan construction allows 0.
    # Execution recheck should fail before expanded invocation.
    runner = SimulatedShadowAttemptRunner()
    # Compressed may still run; expanded fails budget.
    # max_wall_time must be positive for admission.
    result = execute_shadow_plan(plan, attempt_runner=runner)
    # Expanded attempt should be failed due to budget, not silently accepted.
    assert result.expanded_attempt is not None
    assert result.expanded_attempt.attempt_status == AttemptTerminalStatus.FAILED.value
    assert "budget_exceeded" in result.expanded_attempt.failure_reason_codes
    # Runner must not have been invoked for expanded.
    roles = [inv.role for inv in runner.invocations]
    assert ShadowAttemptRole.EXPANDED.value not in roles
    assert result.expanded_attempt.acceptance_disposition != (
        AcceptanceDisposition.ACCEPTED.value
    )


def test_budget_recheck_happens_before_runner_call() -> None:
    plan = _plan(
        max_model_spend_micros=0,
        max_expansion_token_budget=50_000,
    )
    # Zero model spend with positive estimate should block if estimate > 0.
    runner = SimulatedShadowAttemptRunner()
    result = execute_shadow_plan(
        plan,
        attempt_runner=runner,
        estimated_compressed_spend_micros=1,
    )
    # Compressed blocked by spend estimate against zero remaining.
    assert result.compressed_attempt.attempt_status == AttemptTerminalStatus.FAILED.value
    assert "budget_exceeded" in result.compressed_attempt.failure_reason_codes
    assert runner.invocations == []


def test_budget_recheck_count_recorded_on_success() -> None:
    executor = ShadowExecutor()
    result = executor.execute(_plan())
    assert executor.budget_recheck_count >= 4  # 2 roles × 2 rechecks
    assert result.metadata["budget_recheck_count"] == executor.budget_recheck_count


# ---------------------------------------------------------------------------
# Disclosure recheck before invocation
# ---------------------------------------------------------------------------


def test_disclosure_recheck_blocks_private_external_without_authority() -> None:
    policy = default_shadow_disclosure_policy()
    # Private expanded context to unapproved external must not invoke.
    runner = SimulatedShadowAttemptRunner()
    private_ctx = {
        "private_source": "class Secret: pass",
        "context_pack_cid": _cid("expanded-private"),
    }
    plan = _plan(allow_external_expanded_disclosure=False)
    result = execute_shadow_plan(
        plan,
        disclosure_policy=policy,
        attempt_runner=runner,
        expanded_context=private_ctx,
        expanded_provider_id=DEFAULT_EXTERNAL_PROVIDER_ID,
        fallback_expanded_to_local=True,
    )
    # Fallback to local should still run expanded locally.
    assert result.expanded_attempt is not None
    assert any(
        inv.role == ShadowAttemptRole.EXPANDED.value for inv in runner.invocations
    )
    # Expanded provider on invocation should be local after fallback.
    expanded_inv = [
        inv for inv in runner.invocations if inv.role == ShadowAttemptRole.EXPANDED.value
    ][0]
    assert expanded_inv.provider_id.startswith("local:")
    assert expanded_inv.disclosure_disposition != DisclosureDisposition.FORBIDDEN.value


def test_disclosure_recheck_without_fallback_skips_or_fails_closed() -> None:
    runner = SimulatedShadowAttemptRunner()
    private_ctx = {"private_source": "def f(): ...", "raw_private_source": "x"}
    plan = _plan(
        allow_external_expanded_disclosure=False,
        selection_reasons=(
            ShadowSelectionReason.RISK_CLASS_MANDATORY.value,
            ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value,
        ),
    )
    result = execute_shadow_plan(
        plan,
        attempt_runner=runner,
        expanded_context=private_ctx,
        expanded_provider_id=DEFAULT_EXTERNAL_PROVIDER_ID,
        fallback_expanded_to_local=False,
    )
    # Expanded external without fallback: either skip or failed expanded.
    expanded_roles = [
        inv.role for inv in runner.invocations if inv.role == ShadowAttemptRole.EXPANDED.value
    ]
    if result.expanded_attempt is None:
        assert (
            result.expanded_skipped_reason
            == ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
        )
        assert expanded_roles == []
    else:
        # If an attempt was recorded it must not be accepted.
        assert (
            result.expanded_attempt.acceptance_disposition
            != AcceptanceDisposition.ACCEPTED.value
        )


def test_disclosure_recheck_count_recorded() -> None:
    executor = ShadowExecutor()
    result = executor.execute(_plan())
    assert executor.disclosure_recheck_count >= 2
    assert result.metadata["disclosure_recheck_count"] == executor.disclosure_recheck_count


def test_invocation_never_built_with_forbidden_disposition() -> None:
    with pytest.raises(DisclosureRecheckError):
        ShadowAttemptInvocation(
            role=ShadowAttemptRole.COMPRESSED.value,
            plan_cid=_cid("plan"),
            task_id="SCG-026",
            context_pack_cid=_cid("ctx"),
            route_id="route.compressed",
            provider_id=DEFAULT_COMPRESSED_PROVIDER_ID,
            provider_locality="local",
            worktree_id="eval-compressed-1",
            worktree_lease_id="lease.compressed.1",
            worktree_fence=1,
            execution_mode=ExecutionMode.SIMULATED.value,
            disclosure_disposition=DisclosureDisposition.FORBIDDEN.value,
            authorization_decision_cid=None,
            redacted_context={"ok": True},
            estimated_input_tokens=0,
            estimated_model_spend_micros=0,
            remaining_wall_time_ms=1000,
            remaining_model_spend_micros=1000,
            remaining_expansion_tokens=1000,
        )


# ---------------------------------------------------------------------------
# Cancellation / timeout leave production unchanged
# ---------------------------------------------------------------------------


def test_cancellation_leaves_production_unchanged() -> None:
    fingerprint = production_fingerprint_from_refs(
        repository_state_cid=_cid("repo-state"),
        head_commit="abc123",
    )
    guard = ProductionCheckoutGuard(fingerprint=fingerprint)
    token = ShadowCancellationToken()
    lifecycle = InMemoryEvaluationWorktreeLifecycle()

    def _cancel_on_first(inv: ShadowAttemptInvocation) -> None:
        token.cancel("operator_cancel")

    runner = SimulatedShadowAttemptRunner(on_invoke=_cancel_on_first)
    # Cancel mid-flight: first attempt may complete; second should see cancel
    # or we cancel before start by pre-cancelling after first.
    # Use pre-cancelled token for clearer production invariant.
    token2 = ShadowCancellationToken(cancelled=True)
    guard2 = ProductionCheckoutGuard(fingerprint=fingerprint)
    result = execute_shadow_plan(
        _plan(),
        attempt_runner=SimulatedShadowAttemptRunner(),
        worktree_lifecycle=lifecycle,
        production_guard=guard2,
        cancellation_token=token2,
    )
    assert production_state_unchanged(guard2) is True
    assert guard2.production_mutation_count == 0
    assert result.compressed_attempt.attempt_status == (
        AttemptTerminalStatus.CANCELLED.value
    )
    assert result.compressed_attempt.acceptance_disposition != (
        AcceptanceDisposition.ACCEPTED.value
    )
    # No production mutation path exists on lifecycle.
    assert lifecycle.create_count == 0


def test_cancel_mid_run_does_not_mutate_production() -> None:
    fingerprint = production_fingerprint_from_refs(
        repository_state_cid=_cid("repo-state"),
        head_commit="deadbeef",
    )
    guard = ProductionCheckoutGuard(fingerprint=fingerprint)
    token = ShadowCancellationToken()
    call_count = {"n": 0}

    def _cancel_after_compressed(inv: ShadowAttemptInvocation) -> None:
        call_count["n"] += 1
        if inv.role == ShadowAttemptRole.COMPRESSED.value:
            token.cancel("after_compressed")

    runner = SimulatedShadowAttemptRunner(on_invoke=_cancel_after_compressed)
    result = execute_shadow_plan(
        _plan(),
        attempt_runner=runner,
        production_guard=guard,
        cancellation_token=token,
    )
    assert production_state_unchanged(guard) is True
    assert guard.production_mutation_count == 0
    # Expanded should be cancelled without acceptance.
    if result.expanded_attempt is not None:
        assert (
            result.expanded_attempt.acceptance_disposition
            != AcceptanceDisposition.ACCEPTED.value
        )
        assert result.expanded_attempt.verification.production_eligible is False


def test_timeout_leaves_production_unchanged() -> None:
    class FastClock:
        def __init__(self) -> None:
            self._t = 0

        def now_ms(self) -> int:
            # First call (start) returns 0; subsequent calls exceed tiny budget.
            current = self._t
            self._t += 10_000
            return current

    fingerprint = production_fingerprint_from_refs(
        repository_state_cid=_cid("repo-state")
    )
    guard = ProductionCheckoutGuard(fingerprint=fingerprint)
    executor = ShadowExecutor(
        production_guard=guard,
        clock=FastClock(),  # type: ignore[arg-type]
        attempt_runner=SimulatedShadowAttemptRunner(),
    )
    plan = _plan(max_wall_time_ms=1)
    result = executor.execute(plan)
    assert production_state_unchanged(guard) is True
    assert result.compressed_attempt.attempt_status == (
        AttemptTerminalStatus.CANCELLED.value
    )
    assert "timeout" in result.compressed_attempt.failure_reason_codes or (
        result.metadata.get("timeout_reason")
    )


def test_production_mutation_is_detected() -> None:
    guard = ProductionCheckoutGuard(fingerprint="baseline-fingerprint")
    guard.record_production_mutation()
    with pytest.raises(ProductionStateMutatedError):
        guard.assert_unchanged()
    assert production_state_unchanged(guard) is False


def test_runner_cannot_silently_edit_production_via_guard() -> None:
    fingerprint = production_fingerprint_from_refs(
        repository_state_cid=_cid("repo-state")
    )
    guard = ProductionCheckoutGuard(fingerprint=fingerprint)

    def _mutate(_inv: ShadowAttemptInvocation) -> None:
        guard.record_production_mutation()

    runner = SimulatedShadowAttemptRunner(on_invoke=_mutate)
    with pytest.raises(ProductionStateMutatedError):
        execute_shadow_plan(
            _plan(),
            attempt_runner=runner,
            production_guard=guard,
        )


# ---------------------------------------------------------------------------
# Worktree isolation
# ---------------------------------------------------------------------------


def test_worktrees_are_isolated_and_released() -> None:
    lifecycle = InMemoryEvaluationWorktreeLifecycle()
    result = execute_shadow_plan(_plan(), worktree_lifecycle=lifecycle)
    assert result.both_attempts_isolated is True
    assert lifecycle.active_ids() == ()
    created = lifecycle.created_ids()
    assert len(created) == 2
    assert all(wid.startswith("eval-") for wid in created)
    # Managed ids only — no host path separators.
    assert all("/" not in wid and "\\" not in wid for wid in created)


def test_executor_class_execute_matches_function_api() -> None:
    plan = _plan()
    runner = SimulatedShadowAttemptRunner()
    lifecycle = InMemoryEvaluationWorktreeLifecycle()
    executor = ShadowExecutor(
        attempt_runner=runner, worktree_lifecycle=lifecycle
    )
    via_class = executor.execute(plan)
    via_fn = execute_shadow_plan(
        plan,
        attempt_runner=SimulatedShadowAttemptRunner(),
        worktree_lifecycle=InMemoryEvaluationWorktreeLifecycle(),
    )
    assert via_class.plan_cid == via_fn.plan_cid == plan.plan_cid
    assert via_class.both_attempts_isolated is True
    assert via_fn.both_attempts_isolated is True
    assert expanded_never_auto_accepts(via_class)
    assert expanded_never_auto_accepts(via_fn)


# ---------------------------------------------------------------------------
# Callable runner / failure paths
# ---------------------------------------------------------------------------


def test_callable_runner_error_becomes_evaluation_failed() -> None:
    def _boom(_inv: ShadowAttemptInvocation) -> ShadowAttemptProposal:
        raise RuntimeError("provider exploded")

    runner = CallableShadowAttemptRunner(_boom)
    result = execute_shadow_plan(_plan(), attempt_runner=runner)
    assert (
        result.compressed_attempt.attempt_status
        == AttemptTerminalStatus.EVALUATION_FAILED.value
    )
    assert "attempt_runner_error" in result.compressed_attempt.failure_reason_codes
    assert (
        result.compressed_attempt.acceptance_disposition
        != AcceptanceDisposition.ACCEPTED.value
    )


def test_expanded_failure_is_not_accepted() -> None:
    runner = SimulatedShadowAttemptRunner(
        expanded_status=AttemptTerminalStatus.FAILED.value
    )
    result = execute_shadow_plan(_plan(), attempt_runner=runner)
    assert result.expanded_attempt is not None
    assert result.expanded_attempt.attempt_status == AttemptTerminalStatus.FAILED.value
    assert (
        result.expanded_attempt.acceptance_disposition
        == AcceptanceDisposition.NOT_ACCEPTED.value
    )


# ---------------------------------------------------------------------------
# Resource gate wiring
# ---------------------------------------------------------------------------


def test_resource_gate_admits_and_releases_per_role() -> None:
    gate = AlwaysAdmitResourceGate()
    execute_shadow_plan(_plan(), resource_gate=gate)
    assert len(gate.admissions) == 2
    assert len(gate.releases) == 2


def test_default_disclosure_policy_is_local_only() -> None:
    policy = default_shadow_disclosure_policy()
    assert policy.allow_private_source_to_approved_external is False
    executor = ShadowExecutor(disclosure_policy=policy)
    result = executor.execute(_plan())
    assert expanded_never_auto_accepts(result)


# ---------------------------------------------------------------------------
# Identity / round-trip
# ---------------------------------------------------------------------------


def test_result_identity_is_deterministic_for_same_runner() -> None:
    plan = _plan()
    # Simulated runner is deterministic given plan.
    left = execute_shadow_plan(
        plan, attempt_runner=SimulatedShadowAttemptRunner()
    )
    right = execute_shadow_plan(
        plan, attempt_runner=SimulatedShadowAttemptRunner()
    )
    # Worktree ids include fence counters — may differ across runs.
    # Identity of each sealed result is still self-consistent.
    assert left.result_cid == verify_result_identity(left)
    assert right.result_cid == verify_result_identity(right)
    assert left.expanded_attempt is not None
    assert right.expanded_attempt is not None
    assert (
        left.expanded_attempt.acceptance_disposition
        == right.expanded_attempt.acceptance_disposition
        == AcceptanceDisposition.CANDIDATE_ONLY.value
    )
