"""Runtime factory that composes agent supervisor entrypoints behind LaunchGuard.

Every effectful facade path must pass through a complete revalidated LaunchPlan
at the intent, effect, and receipt boundaries. Exact replay adopts or returns
the prior result; crash continuations are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Optional

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


@dataclass
class RuntimeHandles:
    """Bundled runtime handles produced by the factory."""

    guard: LaunchGuard
    facades: Mapping[str, "GuardedFacade"]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class GuardedFacade:
    """Effectful facade that cannot bypass LaunchGuard."""

    name: str
    guard: LaunchGuard
    _effect_fn: Callable[[Mapping[str, Any]], Any]
    _boundary: str = BoundaryKind.EFFECT.value

    def invoke(self, plan: Any, **kwargs: Any) -> Any:
        """Invoke the effect only after LaunchPlan revalidation.

        Stale or incomplete inputs fail before effects. Exact replay returns
        the prior result without re-running the effect.
        """

        def _run(validated: Mapping[str, Any]) -> Any:
            return self._effect_fn(validated, **kwargs)

        return self.guard.run_guarded(
            plan,
            self._boundary,
            _run,
            require_revalidated=True,
            adopt_prior_on_replay=True,
        )

    def invoke_at(
        self,
        plan: Any,
        boundary: str | BoundaryKind,
        **kwargs: Any,
    ) -> Any:
        def _run(validated: Mapping[str, Any]) -> Any:
            return self._effect_fn(validated, **kwargs)

        return self.guard.run_guarded(
            plan,
            boundary,
            _run,
            require_revalidated=True,
            adopt_prior_on_replay=True,
        )

    def continuation_after_crash(
        self,
        plan: Any,
        *,
        boundary: str | BoundaryKind | None = None,
        crash_phase: str = "during_effect",
    ) -> ContinuationAction:
        b = boundary if boundary is not None else self._boundary
        return self.guard.continuation_for_crash(
            plan,
            b,
            crash_phase=crash_phase,
        )


def _default_noop_effect(plan: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]:
    return {
        "ok": True,
        "plan_id": plan.get("plan_id"),
        "intent_id": plan.get("intent_id"),
        "kwargs": dict(kwargs),
    }


@dataclass
class RuntimeFactory:
    """Compose runtime entrypoints with mandatory launch guarding."""

    guard: LaunchGuard = field(default_factory=LaunchGuard)
    _effects: MutableMapping[str, Callable[..., Any]] = field(default_factory=dict)

    def register_effect(
        self,
        name: str,
        effect_fn: Callable[..., Any],
    ) -> None:
        if not name or not isinstance(name, str):
            raise LaunchGuardError(
                "invalid_facade_name",
                "Facade name must be a non-empty string",
            )
        self._effects[name] = effect_fn

    def build_facade(
        self,
        name: str,
        *,
        effect_fn: Callable[..., Any] | None = None,
        boundary: str | BoundaryKind = BoundaryKind.EFFECT,
    ) -> GuardedFacade:
        fn = effect_fn if effect_fn is not None else self._effects.get(name)
        if fn is None:
            fn = _default_noop_effect
        b = boundary.value if isinstance(boundary, BoundaryKind) else str(boundary)
        return GuardedFacade(name=name, guard=self.guard, _effect_fn=fn, _boundary=b)

    def create_runtime(
        self,
        *,
        facade_names: list[str] | tuple[str, ...] | None = None,
        effects: Mapping[str, Callable[..., Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
        use_process_guard: bool = False,
    ) -> RuntimeHandles:
        """Create runtime handles; every facade is launch-guarded."""
        if use_process_guard:
            guard = get_launch_guard()
            self.guard = guard
        else:
            guard = self.guard

        if effects:
            for name, fn in effects.items():
                self.register_effect(name, fn)

        names = list(facade_names) if facade_names is not None else list(self._effects.keys())
        if not names:
            names = ["default"]

        facades: dict[str, GuardedFacade] = {}
        for name in names:
            facades[name] = self.build_facade(name)

        return RuntimeHandles(
            guard=guard,
            facades=facades,
            metadata=dict(metadata or {}),
        )

    def guarded_call(
        self,
        plan: Any,
        effect_fn: Callable[[Mapping[str, Any]], Any],
        *,
        boundary: str | BoundaryKind = BoundaryKind.EFFECT,
    ) -> Any:
        """One-shot guarded effect: no bypass of LaunchPlan revalidation."""
        return self.guard.run_guarded(
            plan,
            boundary,
            effect_fn,
            require_revalidated=True,
            adopt_prior_on_replay=True,
        )


def create_runtime_factory(
    *,
    reset_guard: bool = False,
    effects: Mapping[str, Callable[..., Any]] | None = None,
) -> RuntimeFactory:
    """Factory helper for lifecycle/orchestrator composition."""
    if reset_guard:
        guard = reset_launch_guard()
    else:
        guard = LaunchGuard()
    factory = RuntimeFactory(guard=guard)
    if effects:
        for name, fn in effects.items():
            factory.register_effect(name, fn)
    return factory


__all__ = [
    "ContinuationAction",
    "GuardedFacade",
    "LaunchGuard",
    "LaunchGuardError",
    "RuntimeFactory",
    "RuntimeHandles",
    "BoundaryKind",
    "create_runtime_factory",
    "get_launch_guard",
    "require_launch_plan",
    "reset_launch_guard",
    "validate_launch_plan_complete",
]
