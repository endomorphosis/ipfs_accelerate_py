"""Tests for verification plan execution and acceptance recomputation (IVP-014)."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
    ResourceScheduler,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    CacheReuseDecision,
    CacheReuseDisposition,
    ModelRoute,
    TerminalStatus,
    TestReceipt,
    TypeCheckReceipt,
    VerificationReceiptKind,
)
from ipfs_accelerate_py.agent_supervisor.verification.executor import (
    EXECUTION_BUNDLE_EVIDENCE,
    VERIFICATION_EXECUTOR_INTERFACE,
    CheckRunOutcome,
    ObservedPlanIdentities,
    ResourceRejectionKind,
    VerificationExecutionResult,
    VerificationExecutor,
    VerificationExecutorIdentityError,
    compute_production_acceptance,
    create_verification_executor,
    execute_verification_plan,
)
from ipfs_accelerate_py.agent_supervisor.verification.process_runner import (
    VerificationCancellation,
    fence_process_tree,
)
from ipfs_accelerate_py.agent_supervisor.verification.receipt_cache import (
    VerificationReceiptCache,
)
from ipfs_accelerate_py.agent_supervisor.verification.receipt_store import (
    HermeticVerificationReceiptStore,
)
from test.api.test_agent_supervisor_verification_contracts import (
    _artifact,
    _key,
    _observation,
    _plan,
    _route,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passing(key, *, label: str = "pass"):
    if key.receipt_kind is VerificationReceiptKind.TEST:
        return TestReceipt(key, _observation(key, TerminalStatus.PASSED, label=label))
    return TypeCheckReceipt(
        key, _observation(key, TerminalStatus.PASSED, label=label)
    )


def _failed(key, *, label: str = "fail"):
    if key.receipt_kind is VerificationReceiptKind.TEST:
        return TestReceipt(key, _observation(key, TerminalStatus.FAILED, label=label))
    return TypeCheckReceipt(
        key, _observation(key, TerminalStatus.FAILED, label=label)
    )


def _status_receipt(key, status: TerminalStatus, *, label: str = "run"):
    if key.receipt_kind is VerificationReceiptKind.TEST:
        return TestReceipt(key, _observation(key, status, label=label))
    return TypeCheckReceipt(key, _observation(key, status, label=label))


def _plan_for_keys(*keys, processes: int = 2, max_ms: int = 60_000):
    """Build a plan whose DAG steps are the receipt key ids (1:1 binding)."""

    if not keys:
        raise ValueError("at least one key is required")
    base = _plan(keys[0], *keys[1:]) if len(keys) > 1 else _plan(keys[0])
    dag = {key.key_id: () for key in keys}
    timeouts = {key.key_id: min(30_000, max_ms) for key in keys}
    type_checks = tuple(
        dict.fromkeys(
            "src/example.py"
            for key in keys
            if key.receipt_kind is VerificationReceiptKind.TYPE_CHECK
        )
    )
    tests = tuple(
        dict.fromkeys(
            f"test_{index}"
            for index, key in enumerate(keys)
            if key.receipt_kind is VerificationReceiptKind.TEST
        )
    )
    return replace(
        base,
        expected_processes=processes,
        max_execution_time_ms=max_ms,
        dependency_dag=dag,
        step_timeouts_ms=timeouts,
        required_type_checks=type_checks or base.required_type_checks,
        affected_tests=tests,
    )


def _runner_map(outcomes: dict[str, CheckRunOutcome | TerminalStatus | Exception]):
    """Return a check_runner that resolves by key_id."""

    def _runner(key, *, step_id, timeout_ms, cancellation, plan):
        value = outcomes.get(key.key_id)
        if value is None:
            return CheckRunOutcome(
                receipt=_passing(key, label="default"),
                publication_allowed=True,
            )
        if isinstance(value, Exception):
            raise value
        if isinstance(value, TerminalStatus):
            receipt = _status_receipt(key, value, label=value.value)
            return CheckRunOutcome(
                receipt=receipt,
                publication_allowed=value in {TerminalStatus.PASSED, TerminalStatus.PROVED},
                cancelled=value is TerminalStatus.CANCELLED,
                timed_out=value is TerminalStatus.TIMEOUT,
                unavailable=value is TerminalStatus.UNAVAILABLE,
                reason_codes=(value.value,),
            )
        return value

    return _runner


def _execute(plan, **kwargs):
    kwargs.setdefault("require_resource_lease", False)
    kwargs.setdefault("model_route_decision", _route())
    kwargs.setdefault("minimize_failures", False)
    return execute_verification_plan(plan, **kwargs)


# ---------------------------------------------------------------------------
# Smoke / surface
# ---------------------------------------------------------------------------


def test_module_surface_and_factory() -> None:
    executor = create_verification_executor(require_resource_lease=False)
    assert executor.INTERFACE == VERIFICATION_EXECUTOR_INTERFACE
    assert executor.EVIDENCE == EXECUTION_BUNDLE_EVIDENCE
    assert callable(execute_verification_plan)


def test_happy_path_reuses_nothing_executes_pass_and_accepts() -> None:
    key = _key()
    plan = _plan_for_keys(key)

    def runner(k, **_kwargs):
        return CheckRunOutcome(
            receipt=_passing(k, label="exec"),
            publication_allowed=True,
        )

    result = _execute(plan, check_runner=runner)
    assert isinstance(result, VerificationExecutionResult)
    assert result.production_acceptance is True
    assert len(result.executed_receipts) == 1
    assert len(result.reused_receipts) == 0
    assert result.executed_receipts[0].status is TerminalStatus.PASSED
    assert result.bundle.structurally_complete is True
    assert result.summary.model_route_decision.route is ModelRoute.SMALL_LOCAL_MODEL
    assert result.commitment.IS_ZERO_KNOWLEDGE_PROOF is False
    assert result.identity_revalidation.matched is True
    assert result.cancelled is False
    assert result.timed_out is False


# ---------------------------------------------------------------------------
# Cache reuse vs execute
# ---------------------------------------------------------------------------


def test_plan_approved_reuse_is_distinguished_from_executed() -> None:
    first = _key()
    second = _key(receipt_schema_version=2)
    reused = _passing(first, label="reused")
    plan = _plan_for_keys(first, second)
    plan = replace(
        plan,
        cache_reuse_decisions=(
            CacheReuseDecision(
                key_cid=first.key_id,
                disposition=CacheReuseDisposition.REUSED,
                reason_codes=("exact_current_production_receipt",),
                candidate_receipt=reused,
            ),
            CacheReuseDecision(
                key_cid=second.key_id,
                disposition=CacheReuseDisposition.MISSING,
                reason_codes=("cache_miss",),
            ),
        ),
    )

    def runner(k, **_kwargs):
        assert k.key_id == second.key_id
        return CheckRunOutcome(
            receipt=_passing(k, label="fresh"),
            publication_allowed=True,
        )

    result = _execute(plan, check_runner=runner)
    assert result.production_acceptance is True
    assert {item.receipt_id for item in result.reused_receipts} == {reused.receipt_id}
    assert len(result.executed_receipts) == 1
    assert result.executed_receipts[0].key.key_id == second.key_id
    assert set(result.bundle.reused_receipt_cids) == {reused.receipt_id}
    assert set(result.bundle.executed_receipt_cids) == {
        result.executed_receipts[0].receipt_id
    }


def test_simulated_and_timeout_and_unavailable_cannot_accept() -> None:
    for status in (
        TerminalStatus.SIMULATED,
        TerminalStatus.TIMEOUT,
        TerminalStatus.UNAVAILABLE,
        TerminalStatus.CANCELLED,
        TerminalStatus.FAILED,
    ):
        key = _key(receipt_schema_version=hash(status.value) % 50 + 3)
        plan = _plan_for_keys(key)
        result = _execute(
            plan,
            check_runner=_runner_map({key.key_id: status}),
        )
        assert result.production_acceptance is False, status
        assert result.bundle.receipts[0].status is status


# ---------------------------------------------------------------------------
# Dependencies and bounded parallelism
# ---------------------------------------------------------------------------


def test_dependency_dag_orders_execution() -> None:
    parent = _key()
    child = _key(receipt_schema_version=2)
    plan = _plan_for_keys(parent, child, processes=1)
    plan = replace(
        plan,
        dependency_dag={
            parent.key_id: (),
            child.key_id: (parent.key_id,),
        },
        step_timeouts_ms={
            parent.key_id: 30_000,
            child.key_id: 30_000,
        },
    )
    order: list[str] = []
    barrier = threading.Event()

    def runner(k, **_kwargs):
        order.append(k.key_id)
        if k.key_id == parent.key_id:
            barrier.set()
        else:
            assert barrier.is_set(), "child ran before parent completed"
        return CheckRunOutcome(
            receipt=_passing(k, label=k.key_id[:8]),
            publication_allowed=True,
        )

    result = _execute(plan, check_runner=runner)
    assert result.production_acceptance is True
    assert order == [parent.key_id, child.key_id]


def test_bounded_parallelism_caps_concurrent_workers() -> None:
    keys = tuple(_key(receipt_schema_version=i + 1) for i in range(4))
    plan = _plan_for_keys(*keys, processes=2)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def runner(k, **_kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return CheckRunOutcome(
            receipt=_passing(k, label=k.key_id[:8]),
            publication_allowed=True,
        )

    result = _execute(plan, check_runner=runner)
    assert result.production_acceptance is True
    assert max_active <= 2
    assert max_active >= 1


# ---------------------------------------------------------------------------
# Identity revalidation
# ---------------------------------------------------------------------------


def test_pre_identity_mismatch_raises() -> None:
    key = _key()
    plan = _plan_for_keys(key)
    bad = ObservedPlanIdentities(
        repository_tree_cid=_artifact("wrong-tree"),
        semantic_state_root_cid=plan.semantic_state_root_cid,
        environment_cid=plan.environment_cid,
        dependency_lock_cid=plan.dependency_lock_cid,
    )
    with pytest.raises(VerificationExecutorIdentityError, match="pre-execution"):
        _execute(
            plan,
            observed_identities=bad,
            check_runner=_runner_map({}),
        )


def test_post_identity_mismatch_invalidates_success_and_rejects_acceptance() -> None:
    key = _key()
    plan = _plan_for_keys(key)
    post = ObservedPlanIdentities(
        repository_tree_cid=_artifact("drifted-tree"),
        semantic_state_root_cid=plan.semantic_state_root_cid,
        environment_cid=plan.environment_cid,
        dependency_lock_cid=plan.dependency_lock_cid,
    )
    result = _execute(
        plan,
        post_observed_identities=post,
        check_runner=_runner_map(
            {
                key.key_id: CheckRunOutcome(
                    receipt=_passing(key, label="late"),
                    publication_allowed=True,
                )
            }
        ),
    )
    assert result.production_acceptance is False
    assert result.identity_revalidation.post_matched is False
    assert "repository_tree_cid" in result.identity_revalidation.post_mismatches
    assert result.bundle.receipts[0].status is TerminalStatus.INVALID


# ---------------------------------------------------------------------------
# Typed resource rejection
# ---------------------------------------------------------------------------


def test_plan_resource_rejection_is_typed_and_unavailable() -> None:
    key = _key()
    plan = _plan_for_keys(key, processes=4)
    exhausted = HostResourceSnapshot(
        worker_limit=0,
        available_worker_capacity=0,
        active_workers=0,
        memory_available_bytes=0,
        disk_available_bytes=0,
    )
    result = execute_verification_plan(
        plan,
        require_resource_lease=True,
        resource_scheduler=ResourceScheduler(),
        host_snapshot=exhausted,
        model_route_decision=_route(),
        minimize_failures=False,
        check_runner=_runner_map(
            {
                key.key_id: CheckRunOutcome(
                    receipt=_passing(key),
                    publication_allowed=True,
                )
            }
        ),
    )
    assert result.production_acceptance is False
    assert result.resource_rejections
    assert result.resource_rejections[0].kind in {
        ResourceRejectionKind.PLAN_LEASE_DENIED,
        ResourceRejectionKind.CAPACITY_EXHAUSTED,
    }
    assert all(
        item.status is TerminalStatus.UNAVAILABLE for item in result.bundle.receipts
    )


# ---------------------------------------------------------------------------
# Cancellation / timeout / late receipt fencing
# ---------------------------------------------------------------------------


def test_cancellation_before_execution_emits_cancelled_and_fences() -> None:
    key = _key()
    plan = _plan_for_keys(key)
    cancel = VerificationCancellation()
    cancel.cancel(reason="operator_abort")
    called = {"n": 0}

    def runner(k, **_kwargs):
        called["n"] += 1
        return CheckRunOutcome(
            receipt=_passing(k),
            publication_allowed=True,
        )

    result = _execute(plan, check_runner=runner, cancellation=cancel)
    assert called["n"] == 0
    assert result.cancelled is True
    assert result.cancellation_fenced is True
    assert result.production_acceptance is False
    assert result.bundle.receipts[0].status is TerminalStatus.CANCELLED


def test_late_success_after_cancellation_is_fenced() -> None:
    key = _key()
    plan = _plan_for_keys(key)
    cancel = VerificationCancellation()

    def runner(k, *, cancellation, **_kwargs):
        # Simulate a check that "succeeds" while cancellation is already set.
        if cancellation is not None:
            cancellation.cancel(reason="mid_flight")
        return CheckRunOutcome(
            receipt=_passing(k, label="late-success"),
            publication_allowed=True,
        )

    result = _execute(plan, check_runner=runner, cancellation=cancel)
    assert result.production_acceptance is False
    assert result.late_receipts_fenced >= 1
    assert result.bundle.receipts[0].status is TerminalStatus.CANCELLED
    assert result.cancellation_fenced is True


def test_timeout_status_never_accepts() -> None:
    key = _key()
    plan = _plan_for_keys(key)
    result = _execute(
        plan,
        check_runner=_runner_map({key.key_id: TerminalStatus.TIMEOUT}),
    )
    assert result.production_acceptance is False
    assert result.bundle.receipts[0].status is TerminalStatus.TIMEOUT


def test_fence_process_tree_is_exported_and_callable() -> None:
    # Shared runner helper remains available for grandchildren / escaped sessions.
    assert callable(fence_process_tree)
    assert fence_process_tree(None) is True


# ---------------------------------------------------------------------------
# Unavailable tools remain unavailable
# ---------------------------------------------------------------------------


def test_missing_check_runner_is_unavailable_not_pass() -> None:
    key = _key()
    plan = _plan_for_keys(key)
    result = _execute(plan, check_runner=None, check_runners=None)
    assert result.production_acceptance is False
    assert result.bundle.receipts[0].status is TerminalStatus.UNAVAILABLE
    assert "tool_unavailable" in result.step_outcomes[key.key_id]["reason_codes"] or any(
        "tool_unavailable" in codes
        for codes in [
            result.step_outcomes[key.key_id].get("reason_codes", [])
        ]
    )


# ---------------------------------------------------------------------------
# Failures carry compact counterexamples
# ---------------------------------------------------------------------------


def test_selected_failure_carries_compact_counterexample() -> None:
    key = _key()
    plan = _plan_for_keys(key)
    result = execute_verification_plan(
        plan,
        require_resource_lease=False,
        model_route_decision=_route(),
        minimize_failures=True,
        check_runner=_runner_map(
            {
                key.key_id: CheckRunOutcome(
                    receipt=_failed(key, label="boom"),
                    publication_allowed=True,
                )
            }
        ),
    )
    assert result.production_acceptance is False
    assert result.bundle.receipts[0].status is TerminalStatus.FAILED
    assert result.counterexamples
    cx = result.counterexamples[0]
    assert cx.failed_receipt_cid == result.bundle.receipts[0].receipt_id
    assert cx.failed_key_cid == key.key_id
    assert cx.counterexample_id
    assert result.summary.counterexample_cids


# ---------------------------------------------------------------------------
# Production acceptance rules
# ---------------------------------------------------------------------------


def test_acceptance_requires_all_required_leaves_and_no_fallback_or_review() -> None:
    key = _key()
    plan = _plan_for_keys(key)
    plan = replace(
        plan,
        human_review_required=True,
        human_review_reason_codes=("policy_review",),
    )
    result = _execute(
        plan,
        check_runner=_runner_map(
            {
                key.key_id: CheckRunOutcome(
                    receipt=_passing(key),
                    publication_allowed=True,
                )
            }
        ),
    )
    assert result.production_acceptance is False
    assert result.bundle.human_review_required is True
    assert result.model_route_decision.route is ModelRoute.HUMAN_REVIEW_REQUIRED


def test_mandatory_full_suite_pending_blocks_acceptance() -> None:
    primary = _key(kind=VerificationReceiptKind.TEST)
    suite = _key(kind=VerificationReceiptKind.TEST, receipt_schema_version=2)
    plan = _plan_for_keys(primary, suite)
    plan = replace(
        plan,
        full_suite_required=True,
        full_suite_receipt_key_cids=(suite.key_id,),
        full_suite_reason_codes=("uncertain_selection",),
        cache_reuse_decisions=(
            CacheReuseDecision(
                key_cid=primary.key_id,
                disposition=CacheReuseDisposition.MISSING,
                reason_codes=("cache_miss",),
            ),
            CacheReuseDecision(
                key_cid=suite.key_id,
                disposition=CacheReuseDisposition.MISSING,
                reason_codes=("cache_miss",),
            ),
        ),
    )

    # Only the primary check runs successfully; force suite to stay unresolved
    # by cancelling mid-plan after primary? Simpler: suite returns unavailable.
    def runner(k, **_kwargs):
        if k.key_id == primary.key_id:
            return CheckRunOutcome(
                receipt=_passing(k, label="primary"),
                publication_allowed=True,
            )
        return CheckRunOutcome(
            receipt=_status_receipt(k, TerminalStatus.UNAVAILABLE, label="suite"),
            publication_allowed=False,
            unavailable=True,
            reason_codes=("suite_unavailable",),
        )

    result = _execute(plan, check_runner=runner)
    assert result.production_acceptance is False
    # Suite key has a receipt (unavailable) so mandatory_fallback_pending is
    # false, but unavailable still blocks acceptance.
    assert any(
        item.status is TerminalStatus.UNAVAILABLE for item in result.bundle.receipts
    )


def test_compute_production_acceptance_helper_matches_bundle_rules() -> None:
    key = _key()
    plan = _plan_for_keys(key)
    result = _execute(
        plan,
        check_runner=_runner_map(
            {
                key.key_id: CheckRunOutcome(
                    receipt=_passing(key),
                    publication_allowed=True,
                )
            }
        ),
    )
    assert compute_production_acceptance(result.bundle) is True
    assert (
        compute_production_acceptance(
            result.bundle, advisory_key_cids=(key.key_id,)
        )
        is True
    )


# ---------------------------------------------------------------------------
# Advisory obligations: unresolved OK, never upgraded
# ---------------------------------------------------------------------------


def test_advisory_obligation_may_remain_unresolved_without_blocking() -> None:
    required = _key()
    advisory = _key(receipt_schema_version=2)
    plan = _plan_for_keys(required, advisory)

    def runner(k, **_kwargs):
        if k.key_id == required.key_id:
            return CheckRunOutcome(
                receipt=_passing(k, label="req"),
                publication_allowed=True,
            )
        return CheckRunOutcome(
            receipt=_status_receipt(k, TerminalStatus.NOT_MODELED, label="adv"),
            publication_allowed=False,
            reason_codes=("advisory_skip",),
        )

    result = _execute(
        plan,
        check_runner=runner,
        advisory_key_cids=(advisory.key_id,),
    )
    assert result.production_acceptance is True
    assert advisory.key_id in result.advisory_unresolved_key_cids or any(
        item.key.key_id == advisory.key_id
        and item.status is TerminalStatus.NOT_MODELED
        for item in result.bundle.receipts
    )


def test_advisory_success_is_never_upgraded_to_production_leaf() -> None:
    required = _key()
    advisory = _key(receipt_schema_version=2)
    plan = _plan_for_keys(required, advisory)

    def runner(k, **_kwargs):
        # Advisory runner tries to report PASSED — executor must collapse it.
        return CheckRunOutcome(
            receipt=_passing(k, label="try-upgrade"),
            publication_allowed=True,
        )

    result = _execute(
        plan,
        check_runner=runner,
        advisory_key_cids=(advisory.key_id,),
    )
    advisory_receipt = next(
        item for item in result.bundle.receipts if item.key.key_id == advisory.key_id
    )
    assert advisory_receipt.status is TerminalStatus.NOT_MODELED
    assert "advisory_never_upgraded" in set(advisory_receipt.reason_codes) or any(
        "advisory" in code for code in advisory_receipt.reason_codes
    )
    # Required still passes → acceptance may remain true with advisory collapsed.
    assert result.production_acceptance is True


# ---------------------------------------------------------------------------
# Summary, route, commitment always emitted
# ---------------------------------------------------------------------------


def test_emits_compact_summary_provider_neutral_route_and_commitment() -> None:
    key = _key()
    plan = _plan_for_keys(key)
    result = _execute(
        plan,
        check_runner=_runner_map(
            {
                key.key_id: CheckRunOutcome(
                    receipt=_passing(key),
                    publication_allowed=True,
                )
            }
        ),
    )
    payload = result.to_dict()
    assert payload["evidence"] == EXECUTION_BUNDLE_EVIDENCE
    assert payload["production_acceptance"] is True
    assert payload["model_route"] == ModelRoute.SMALL_LOCAL_MODEL.value
    assert "openai" not in str(payload).lower()
    assert "anthropic" not in str(payload).lower()
    assert result.summary.aggregate_terminal_status is TerminalStatus.PASSED
    assert result.commitment.merkle_root
    assert result.commitment.IS_ZERO_KNOWLEDGE_PROOF is False
    assert result.wall_time_ms >= 0


# ---------------------------------------------------------------------------
# Cache admit + tombstone post-plan
# ---------------------------------------------------------------------------


def test_admits_success_and_publishes_stale_tombstone(tmp_path: Path) -> None:
    key = _key()
    stale_candidate = _status_receipt(key, TerminalStatus.STALE, label="stale")
    # STALE disposition carries candidate for audit; plan requires miss execute.
    plan = _plan_for_keys(key)
    plan = replace(
        plan,
        cache_reuse_decisions=(
            CacheReuseDecision(
                key_cid=key.key_id,
                disposition=CacheReuseDisposition.STALE,
                reason_codes=("scoped_staleness_tombstone",),
                candidate_receipt=stale_candidate,
            ),
        ),
    )
    store = HermeticVerificationReceiptStore(tmp_path / "receipts")
    cache = VerificationReceiptCache(store)

    def runner(k, **_kwargs):
        return CheckRunOutcome(
            receipt=_passing(k, label="fresh"),
            publication_allowed=True,
        )

    result = execute_verification_plan(
        plan,
        cache=cache,
        check_runner=runner,
        require_resource_lease=False,
        model_route_decision=_route(),
        minimize_failures=False,
        admit_successes=True,
    )
    assert result.production_acceptance is True
    assert key.key_id in result.tombstones_published
    decision = cache.lookup(key, for_production=True)
    assert decision.disposition is CacheReuseDisposition.REUSED


# ---------------------------------------------------------------------------
# Class API
# ---------------------------------------------------------------------------


def test_verification_executor_class_execute() -> None:
    key = _key()
    plan = _plan_for_keys(key)
    executor = VerificationExecutor(
        require_resource_lease=False,
        model_route_decision=_route(),
        minimize_failures=False,
        check_runner=_runner_map(
            {
                key.key_id: CheckRunOutcome(
                    receipt=_passing(key),
                    publication_allowed=True,
                )
            }
        ),
    )
    result = executor.execute(plan)
    assert result.production_acceptance is True
    assert result.execution_id.startswith("exec:")
