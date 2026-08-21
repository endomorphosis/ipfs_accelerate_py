"""Tests for LaunchGuard: complete revalidated LaunchPlan at every effect boundary."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.launch_guard import (
    BoundaryKind,
    ContinuationAction,
    LaunchGuard,
    LaunchGuardError,
    get_launch_guard,
    require_launch_plan,
    reset_launch_guard,
    validate_launch_plan_complete,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.runtime_factory import (
    GuardedFacade,
    RuntimeFactory,
    create_runtime_factory,
)


def _complete_plan(**overrides):
    base = {
        "plan_id": "plan-001",
        "intent_id": "intent-001",
        "created_at": "2026-01-01T00:00:00Z",
        "config_fingerprint": "fp-abc123",
        "effect_sequence": ["step-a", "step-b"],
        "payload": {"action": "start"},
        "revalidated": True,
        "revalidated_at": "2026-01-01T00:00:01Z",
        "stale": False,
    }
    base.update(overrides)
    return base


class TestValidateLaunchPlanComplete:
    def test_accepts_complete_revalidated_plan(self):
        plan = _complete_plan()
        out = validate_launch_plan_complete(plan)
        assert out["plan_id"] == "plan-001"

    def test_rejects_none(self):
        with pytest.raises(LaunchGuardError) as ei:
            validate_launch_plan_complete(None)
        assert ei.value.code == "missing_launch_plan"

    def test_rejects_non_mapping(self):
        with pytest.raises(LaunchGuardError) as ei:
            validate_launch_plan_complete("not-a-plan")
        assert ei.value.code == "invalid_launch_plan_type"

    def test_rejects_incomplete_missing_fields(self):
        plan = _complete_plan()
        del plan["intent_id"]
        with pytest.raises(LaunchGuardError) as ei:
            validate_launch_plan_complete(plan)
        assert ei.value.code == "incomplete_launch_plan"
        assert "intent_id" in ei.value.message

    def test_rejects_empty_plan_id(self):
        plan = _complete_plan(plan_id="")
        with pytest.raises(LaunchGuardError) as ei:
            validate_launch_plan_complete(plan)
        assert ei.value.code == "incomplete_launch_plan"

    def test_rejects_missing_effect_sequence_type(self):
        plan = _complete_plan(effect_sequence="bad")
        with pytest.raises(LaunchGuardError) as ei:
            validate_launch_plan_complete(plan)
        assert ei.value.code == "incomplete_launch_plan"

    def test_rejects_not_revalidated(self):
        plan = _complete_plan()
        del plan["revalidated"]
        del plan["revalidated_at"]
        with pytest.raises(LaunchGuardError) as ei:
            validate_launch_plan_complete(plan)
        assert ei.value.code == "not_revalidated"

    def test_accepts_revalidated_at_only(self):
        plan = _complete_plan()
        del plan["revalidated"]
        plan["revalidated_at"] = "2026-01-02T00:00:00Z"
        out = validate_launch_plan_complete(plan)
        assert out["revalidated_at"] == "2026-01-02T00:00:00Z"

    def test_rejects_stale(self):
        plan = _complete_plan(stale=True)
        with pytest.raises(LaunchGuardError) as ei:
            validate_launch_plan_complete(plan)
        assert ei.value.code == "stale_launch_plan"

    def test_rejects_invalid_flag(self):
        plan = _complete_plan(invalid=True)
        with pytest.raises(LaunchGuardError) as ei:
            validate_launch_plan_complete(plan)
        assert ei.value.code == "invalid_launch_plan"

    def test_stale_fails_before_effect(self):
        guard = LaunchGuard()
        calls = []

        def effect(p):
            calls.append(p)
            return "done"

        with pytest.raises(LaunchGuardError) as ei:
            guard.run_guarded(_complete_plan(stale=True), BoundaryKind.EFFECT, effect)
        assert ei.value.code == "stale_launch_plan"
        assert calls == []

    def test_incomplete_fails_before_effect(self):
        guard = LaunchGuard()
        calls = []

        def effect(p):
            calls.append(1)
            return "done"

        bad = _complete_plan()
        del bad["payload"]
        with pytest.raises(LaunchGuardError):
            guard.run_guarded(bad, "effect", effect)
        assert calls == []


class TestExactReplay:
    def test_exact_replay_returns_prior_result(self):
        guard = LaunchGuard()
        calls = []

        def effect(p):
            calls.append("ran")
            return {"receipt": "r1", "plan_id": p["plan_id"]}

        plan = _complete_plan()
        first = guard.run_guarded(plan, BoundaryKind.EFFECT, effect)
        second = guard.run_guarded(plan, BoundaryKind.EFFECT, effect)
        assert first == second
        assert first["receipt"] == "r1"
        assert calls == ["ran"]  # effect not re-executed on exact replay

    def test_different_fingerprint_is_not_replay(self):
        guard = LaunchGuard()
        calls = []

        def effect(p):
            calls.append(p["config_fingerprint"])
            return p["config_fingerprint"]

        p1 = _complete_plan(config_fingerprint="fp-1")
        p2 = _complete_plan(config_fingerprint="fp-2")
        assert guard.run_guarded(p1, "effect", effect) == "fp-1"
        assert guard.run_guarded(p2, "effect", effect) == "fp-2"
        assert calls == ["fp-1", "fp-2"]

    def test_adopt_prior_via_check_replay(self):
        guard = LaunchGuard()
        plan = _complete_plan()

        def effect(p):
            return 42

        guard.run_guarded(plan, BoundaryKind.EFFECT, effect)
        is_replay, prior = guard.check_replay(plan, BoundaryKind.EFFECT)
        assert is_replay is True
        assert prior == 42


class TestCrashContinuation:
    def test_crash_incomplete_is_fail_incomplete(self):
        guard = LaunchGuard()
        action = guard.continuation_for_crash(None, BoundaryKind.INTENT)
        assert action == ContinuationAction.FAIL_INCOMPLETE

    def test_crash_stale_is_fail_stale(self):
        guard = LaunchGuard()
        action = guard.continuation_for_crash(
            _complete_plan(stale=True),
            BoundaryKind.EFFECT,
        )
        assert action == ContinuationAction.FAIL_STALE

    def test_crash_not_revalidated_is_fail_stale(self):
        guard = LaunchGuard()
        plan = _complete_plan()
        del plan["revalidated"]
        del plan["revalidated_at"]
        action = guard.continuation_for_crash(plan, BoundaryKind.RECEIPT)
        assert action == ContinuationAction.FAIL_STALE

    def test_crash_with_prior_success_returns_prior(self):
        guard = LaunchGuard()
        plan = _complete_plan()

        def effect(p):
            return "ok"

        guard.run_guarded(plan, BoundaryKind.EFFECT, effect)
        action = guard.continuation_for_crash(
            plan,
            BoundaryKind.EFFECT,
            crash_phase="before_effect",
        )
        assert action == ContinuationAction.RETURN_PRIOR

    def test_crash_after_effect_replays_receipt(self):
        guard = LaunchGuard()
        plan = _complete_plan()

        def effect(p):
            return "ok"

        guard.run_guarded(plan, BoundaryKind.RECEIPT, effect)
        action = guard.continuation_for_crash(
            plan,
            BoundaryKind.RECEIPT,
            crash_phase="during_receipt",
        )
        assert action == ContinuationAction.REPLAY_RECEIPT

    def test_crash_before_effect_no_receipt_proceeds(self):
        guard = LaunchGuard()
        plan = _complete_plan()
        action = guard.continuation_for_crash(
            plan,
            BoundaryKind.INTENT,
            crash_phase="before_effect",
        )
        assert action == ContinuationAction.PROCEED

    def test_deterministic_single_continuation_per_state(self):
        guard = LaunchGuard()
        plan = _complete_plan()
        a1 = guard.continuation_for_crash(plan, "effect", crash_phase="during_effect")
        a2 = guard.continuation_for_crash(plan, "effect", crash_phase="during_effect")
        assert a1 == a2


class TestNoBypass:
    def test_require_launch_plan_blocks_incomplete(self):
        with pytest.raises(LaunchGuardError):
            require_launch_plan({"plan_id": "x"})

    def test_assert_no_bypass_on_guard(self):
        guard = LaunchGuard()
        with pytest.raises(LaunchGuardError) as ei:
            guard.assert_no_bypass(None)
        assert ei.value.code == "missing_launch_plan"

    def test_facade_cannot_bypass_guard(self):
        factory = RuntimeFactory()
        calls = []

        def raw_effect(plan, **kwargs):
            calls.append(True)
            return "effected"

        facade = factory.build_facade("write", effect_fn=raw_effect)
        with pytest.raises(LaunchGuardError):
            facade.invoke({"plan_id": "only"})
        assert calls == []

        result = facade.invoke(_complete_plan())
        assert result == "effected"
        assert calls == [True]

    def test_facade_exact_replay_adopts_prior(self):
        factory = RuntimeFactory()
        n = {"count": 0}

        def raw_effect(plan, **kwargs):
            n["count"] += 1
            return n["count"]

        facade = factory.build_facade("mut", effect_fn=raw_effect)
        plan = _complete_plan()
        assert facade.invoke(plan) == 1
        assert facade.invoke(plan) == 1
        assert n["count"] == 1

    def test_guarded_call_on_factory(self):
        factory = create_runtime_factory()
        out = factory.guarded_call(
            _complete_plan(),
            lambda p: p["plan_id"],
            boundary=BoundaryKind.INTENT,
        )
        assert out == "plan-001"

    def test_create_runtime_wires_guarded_facades(self):
        factory = RuntimeFactory()

        def eff(plan, **kw):
            return "v"

        factory.register_effect("launch", eff)
        handles = factory.create_runtime(facade_names=["launch"], metadata={"v": 1})
        assert "launch" in handles.facades
        assert isinstance(handles.facades["launch"], GuardedFacade)
        assert handles.metadata["v"] == 1
        assert handles.facades["launch"].invoke(_complete_plan()) == "v"

    def test_reset_and_get_launch_guard(self):
        g1 = reset_launch_guard()
        g2 = get_launch_guard()
        assert g1 is g2
        plan = _complete_plan()
        require_launch_plan(plan, guard=g2)

    def test_boundaries_intent_effect_receipt(self):
        guard = LaunchGuard()
        plan = _complete_plan()
        seen = []

        for b in (BoundaryKind.INTENT, BoundaryKind.EFFECT, BoundaryKind.RECEIPT):

            def effect(p, _b=b):
                seen.append(_b.value)
                return _b.value

            assert guard.run_guarded(plan, b, effect) == b.value

        assert seen == ["intent", "effect", "receipt"]

    def test_facade_continuation_after_crash(self):
        factory = RuntimeFactory()
        facade = factory.build_facade("x", effect_fn=lambda p, **k: 1)
        action = facade.continuation_after_crash(
            None,
            crash_phase="during_effect",
        )
        assert action == ContinuationAction.FAIL_INCOMPLETE
