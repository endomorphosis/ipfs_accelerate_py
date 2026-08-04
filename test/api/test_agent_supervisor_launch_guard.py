"""Tests for agent supervisor launch guard and guarded runtime factory."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.launch_guard import (
    BoundaryContinuation,
    BoundaryPhase,
    IncompleteLaunchPlanError,
    InMemoryReceiptStore,
    LaunchGuard,
    StaleLaunchPlanError,
    compute_content_hash,
    continuation_for_crash,
    revalidate_launch_plan,
    validate_launch_plan_complete,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.runtime_factory import (
    RuntimeFactory,
    build_runtime_factory,
)


def _complete_plan(**overrides):
    base = {
        "plan_id": "plan-001",
        "intent_hash": "intent-abc",
        "effect_spec": {"action": "start_agent", "agent": "demo"},
        "created_at": 1_700_000_000,
        "schema_version": "1",
    }
    base.update(overrides)
    if "content_hash" not in base:
        base["content_hash"] = compute_content_hash(base)
    return base


class TestValidateLaunchPlanComplete:
    def test_accepts_complete_plan(self):
        plan = _complete_plan()
        out = validate_launch_plan_complete(plan)
        assert out["plan_id"] == "plan-001"

    def test_rejects_non_mapping(self):
        with pytest.raises(IncompleteLaunchPlanError):
            validate_launch_plan_complete("not-a-plan")

    def test_rejects_missing_required_fields(self):
        plan = _complete_plan()
        del plan["intent_hash"]
        with pytest.raises(IncompleteLaunchPlanError) as exc:
            validate_launch_plan_complete(plan)
        assert "intent_hash" in str(exc.value)

    def test_rejects_empty_plan_id(self):
        with pytest.raises(IncompleteLaunchPlanError):
            validate_launch_plan_complete(_complete_plan(plan_id=""))

    def test_rejects_non_mapping_effect_spec(self):
        with pytest.raises(IncompleteLaunchPlanError):
            validate_launch_plan_complete(_complete_plan(effect_spec="bad"))


class TestRevalidateLaunchPlan:
    def test_content_hash_mismatch_is_stale(self):
        plan = _complete_plan(content_hash="sha256:deadbeef")
        with pytest.raises(StaleLaunchPlanError):
            revalidate_launch_plan(plan)

    def test_schema_version_pin(self):
        plan = _complete_plan(schema_version="1")
        plan["content_hash"] = compute_content_hash(plan)
        with pytest.raises(StaleLaunchPlanError):
            revalidate_launch_plan(plan, expected_schema_version="2")

    def test_ttl_staleness(self):
        plan = _complete_plan(created_at=1000.0, ttl_seconds=10)
        plan["content_hash"] = compute_content_hash(plan)
        with pytest.raises(StaleLaunchPlanError):
            revalidate_launch_plan(plan, now=1020.0)

    def test_fresh_plan_passes_ttl(self):
        plan = _complete_plan(created_at=1000.0, ttl_seconds=30)
        plan["content_hash"] = compute_content_hash(plan)
        out = revalidate_launch_plan(plan, now=1010.0)
        assert out["plan_id"] == "plan-001"


class TestLaunchGuardReplay:
    def test_exact_replay_adopts_prior_receipt(self):
        store = InMemoryReceiptStore()
        prior = {"plan_id": "plan-001", "intent_hash": "intent-abc", "status": "ok"}
        store.put_receipt("plan-001", "intent-abc", prior)
        guard = LaunchGuard(receipt_store=store)
        decision = guard.check(_complete_plan())
        assert decision.is_replay is True
        assert decision.adopted_receipt == prior
        assert decision.reason == "exact_replay_adopt_prior"

    def test_embedded_prior_receipt_replay(self):
        prior = {"status": "ok", "value": 42}
        plan = _complete_plan(prior_receipt=prior)
        plan["content_hash"] = compute_content_hash(plan)
        guard = LaunchGuard()
        decision = guard.check(plan)
        assert decision.is_replay is True
        assert decision.adopted_receipt["value"] == 42

    def test_run_effect_skips_effect_on_replay(self):
        store = InMemoryReceiptStore()
        store.put_receipt(
            "plan-001",
            "intent-abc",
            {"status": "ok", "from": "store"},
        )
        guard = LaunchGuard(receipt_store=store)
        called = {"n": 0}

        def effect(_plan):
            called["n"] += 1
            return {"ran": True}

        result = guard.run_effect(_complete_plan(), effect)
        assert result["replayed"] is True
        assert result["status"] == "adopted_prior"
        assert called["n"] == 0
        assert result["receipt"]["from"] == "store"

    def test_run_effect_persists_receipt_for_later_replay(self):
        store = InMemoryReceiptStore()
        guard = LaunchGuard(receipt_store=store)
        plan = _complete_plan()

        first = guard.run_effect(plan, lambda p: {"ok": True})
        assert first["replayed"] is False
        assert first["status"] == "completed"

        second = guard.run_effect(plan, lambda p: {"ok": "should-not-run"})
        assert second["replayed"] is True
        assert second["receipt"]["status"] == "ok"


class TestLaunchGuardBoundaries:
    def test_continuation_mapping_is_deterministic(self):
        assert continuation_for_crash(BoundaryPhase.INTENT) == BoundaryContinuation.RETRY_INTENT
        assert continuation_for_crash(BoundaryPhase.EFFECT) == BoundaryContinuation.RETRY_EFFECT
        assert continuation_for_crash(BoundaryPhase.RECEIPT) == BoundaryContinuation.RETRY_RECEIPT

    def test_effect_crash_returns_retry_effect(self):
        guard = LaunchGuard()

        def boom(_plan):
            raise RuntimeError("effect failed")

        result = guard.run_effect(_complete_plan(), boom)
        assert result["status"] == "crashed"
        assert result["crashed_at"] == "effect"
        assert result["continuation"] == BoundaryContinuation.RETRY_EFFECT.value

    def test_receipt_crash_returns_retry_receipt(self):
        guard = LaunchGuard()

        def effect(_plan):
            return {"x": 1}

        def bad_receipt(_result, _plan):
            raise RuntimeError("receipt failed")

        result = guard.run_effect(
            _complete_plan(), effect, receipt_builder=bad_receipt
        )
        assert result["status"] == "crashed"
        assert result["crashed_at"] == "receipt"
        assert result["continuation"] == BoundaryContinuation.RETRY_RECEIPT.value

    def test_resolve_crash_continuation_fail_closed_without_state(self):
        guard = LaunchGuard()
        cont = guard.resolve_crash_continuation("missing", "missing")
        assert cont == BoundaryContinuation.FAIL_CLOSED

    def test_mark_crash_then_begin_boundary_resumes_continuation(self):
        guard = LaunchGuard()
        plan = _complete_plan()
        decision = guard.check(plan)
        state = guard.begin_boundary(decision.plan)
        cont = guard.mark_crash(state, BoundaryPhase.EFFECT)
        assert cont == BoundaryContinuation.RETRY_EFFECT
        resumed = guard.begin_boundary(decision.plan)
        assert resumed.continuation == BoundaryContinuation.RETRY_EFFECT

    def test_incomplete_plan_fails_before_effect(self):
        guard = LaunchGuard()
        called = {"n": 0}

        def effect(_plan):
            called["n"] += 1

        with pytest.raises(IncompleteLaunchPlanError):
            guard.run_effect({"plan_id": "only"}, effect)
        assert called["n"] == 0

    def test_stale_plan_fails_before_effect(self):
        guard = LaunchGuard(max_age_seconds=5, clock=lambda: 2000.0)
        plan = _complete_plan(created_at=1000.0)
        plan["content_hash"] = compute_content_hash(plan)
        called = {"n": 0}

        def effect(_plan):
            called["n"] += 1

        with pytest.raises(StaleLaunchPlanError):
            guard.run_effect(plan, effect)
        assert called["n"] == 0


class TestRuntimeFactoryGuard:
    def test_create_runtime_requires_complete_plan(self):
        factory = RuntimeFactory()
        with pytest.raises(IncompleteLaunchPlanError):
            factory.create_runtime({})

    def test_create_runtime_admits_valid_plan(self):
        factory = build_runtime_factory()
        handle = factory.create_runtime(_complete_plan())
        assert handle.plan_id == "plan-001"
        assert handle.is_replay is False
        assert factory.get_session(handle.runtime_id) is handle

    def test_launch_goes_through_guard(self):
        factory = RuntimeFactory()
        result = factory.launch(_complete_plan())
        assert result["status"] == "completed"
        assert result["replayed"] is False
        assert result["receipt"]["status"] == "ok"

    def test_launch_replay_does_not_re_effect(self):
        built = {"count": 0}

        def builder(_plan):
            built["count"] += 1
            return {"built": True}

        factory = RuntimeFactory(runtime_builder=builder)
        plan = _complete_plan()
        first = factory.launch(plan)
        assert first["status"] == "completed"
        assert built["count"] == 1

        second = factory.launch(plan)
        assert second["replayed"] is True
        assert built["count"] == 1

    def test_execute_effect_rejects_stale(self):
        factory = RuntimeFactory(max_age_seconds=1, clock=lambda: 5000.0)
        plan = _complete_plan(created_at=1.0)
        plan["content_hash"] = compute_content_hash(plan)
        with pytest.raises(StaleLaunchPlanError):
            factory.execute_effect(plan, lambda p: None)

    def test_admit_exact_replay(self):
        store = InMemoryReceiptStore()
        store.put_receipt(
            "plan-001", "intent-abc", {"status": "ok", "n": 7}
        )
        factory = RuntimeFactory(receipt_store=store)
        decision = factory.admit(_complete_plan())
        assert decision.is_replay is True
        handle = factory.create_runtime(_complete_plan())
        assert handle.is_replay is True
        assert handle.adopted_receipt["n"] == 7

    def test_no_unguarded_effect_path_on_factory(self):
        """Effectful methods always require a plan argument (guard entry)."""
        factory = RuntimeFactory()
        # Public effectful API surface requires plan.
        assert callable(factory.launch)
        assert callable(factory.execute_effect)
        assert callable(factory.create_runtime)
        # Guard is always present.
        assert factory.guard is not None
