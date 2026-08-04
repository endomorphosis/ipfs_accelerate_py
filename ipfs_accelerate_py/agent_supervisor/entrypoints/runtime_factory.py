"""Runtime factory that only constructs effectful runtimes through LaunchGuard.

No effectful facade may bypass the guard: every factory entry that can
perform effects requires a complete revalidated LaunchPlan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from ipfs_accelerate_py.agent_supervisor.entrypoints.launch_guard import (
    GuardDecision,
    IncompleteLaunchPlanError,
    LaunchGuard,
    LaunchGuardError,
    LaunchReceiptStore,
    StaleLaunchPlanError,
    require_launch_plan,
)


class RuntimeFactoryError(LaunchGuardError):
    """Errors raised by the runtime factory."""


@dataclass
class RuntimeHandle:
    """Opaque handle to a guarded runtime session."""

    runtime_id: str
    plan_id: str
    intent_hash: str
    schema_version: str
    effect_spec: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    adopted_receipt: Optional[Dict[str, Any]] = None

    @property
    def is_replay(self) -> bool:
        return self.adopted_receipt is not None


class RuntimeFactory:
    """Build agent-supervisor runtimes only after LaunchGuard admission.

    All public effectful methods require a complete revalidated LaunchPlan.
    Bypassing the guard is not supported: there is no unguarded constructor
    path for effectful work.
    """

    def __init__(
        self,
        *,
        launch_guard: Optional[LaunchGuard] = None,
        receipt_store: Optional[LaunchReceiptStore] = None,
        expected_schema_version: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
        clock: Optional[Callable[[], float]] = None,
        runtime_builder: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    ) -> None:
        if launch_guard is not None:
            self._guard = launch_guard
        else:
            self._guard = LaunchGuard(
                receipt_store=receipt_store,
                expected_schema_version=expected_schema_version,
                max_age_seconds=max_age_seconds,
                clock=clock,
            )
        self._runtime_builder = runtime_builder
        self._sessions: Dict[str, RuntimeHandle] = {}

    @property
    def guard(self) -> LaunchGuard:
        """The LaunchGuard every effectful path must pass."""
        return self._guard

    def admit(self, plan: Any) -> GuardDecision:
        """Admit a LaunchPlan without performing effects."""
        return self._guard.check(plan)

    def create_runtime(self, plan: Any) -> RuntimeHandle:
        """Create a runtime handle only after complete revalidated LaunchPlan.

        Stale/incomplete plans raise before any session is recorded.
        Exact replay returns a handle that adopts the prior receipt.
        """
        decision = self._guard.check(plan)
        handle = RuntimeHandle(
            runtime_id=f"rt:{decision.plan['plan_id']}:{decision.plan['intent_hash']}",
            plan_id=decision.plan["plan_id"],
            intent_hash=decision.plan["intent_hash"],
            schema_version=decision.plan["schema_version"],
            effect_spec=dict(decision.plan["effect_spec"]),
            metadata=dict(decision.plan.get("metadata") or {}),
            adopted_receipt=decision.adopted_receipt,
        )
        self._sessions[handle.runtime_id] = handle
        return handle

    def launch(self, plan: Any) -> Dict[str, Any]:
        """Effectful launch: guarded intent → effect → receipt.

        Uses LaunchGuard.run_effect so no effect can bypass the guard.
        """
        builder = self._runtime_builder

        def _effect(validated_plan: Mapping[str, Any]) -> Any:
            if builder is not None:
                return builder(validated_plan)
            # Default effect: materialize a RuntimeHandle payload.
            return {
                "runtime_id": f"rt:{validated_plan['plan_id']}:{validated_plan['intent_hash']}",
                "effect_spec": dict(validated_plan["effect_spec"]),
                "status": "launched",
            }

        def _receipt(effect_result: Any, validated_plan: Mapping[str, Any]) -> Dict[str, Any]:
            return {
                "plan_id": validated_plan["plan_id"],
                "intent_hash": validated_plan["intent_hash"],
                "runtime": effect_result,
                "status": "ok",
            }

        return self._guard.run_effect(plan, _effect, receipt_builder=_receipt)

    def execute_effect(
        self,
        plan: Any,
        effect_fn: Callable[[Mapping[str, Any]], Any],
        *,
        receipt_builder: Optional[
            Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]
        ] = None,
    ) -> Dict[str, Any]:
        """Run an arbitrary effect only through the launch guard."""
        return self._guard.run_effect(
            plan, effect_fn, receipt_builder=receipt_builder
        )

    def get_session(self, runtime_id: str) -> Optional[RuntimeHandle]:
        """Return a previously admitted session handle, if any."""
        return self._sessions.get(runtime_id)

    def require_plan(self, plan: Any, **kwargs: Any) -> Dict[str, Any]:
        """Fail closed unless plan is complete and revalidated."""
        return require_launch_plan(plan, **kwargs)


def build_runtime_factory(
    *,
    receipt_store: Optional[LaunchReceiptStore] = None,
    expected_schema_version: Optional[str] = None,
    max_age_seconds: Optional[float] = None,
    clock: Optional[Callable[[], float]] = None,
    runtime_builder: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    launch_guard: Optional[LaunchGuard] = None,
) -> RuntimeFactory:
    """Factory helper: always wires LaunchGuard into the runtime factory."""
    return RuntimeFactory(
        launch_guard=launch_guard,
        receipt_store=receipt_store,
        expected_schema_version=expected_schema_version,
        max_age_seconds=max_age_seconds,
        clock=clock,
        runtime_builder=runtime_builder,
    )


__all__ = [
    "RuntimeFactory",
    "RuntimeFactoryError",
    "RuntimeHandle",
    "build_runtime_factory",
    "IncompleteLaunchPlanError",
    "StaleLaunchPlanError",
    "LaunchGuard",
    "LaunchGuardError",
]
