"""Fail-closed composition for the prompt-to-run runtime.

Production construction deliberately has no convenience implementation.  A
caller must supply every effect boundary used by a launch; this prevents a
missing integration from looking like a completed supervisor run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json

from .contracts import LaunchPlan
from .run_registry import RunRegistry


REQUIRED_RUNTIME_HANDLERS: Final[tuple[str, ...]] = (
    "resolve", "preview", "authorize", "materialize", "start", "adopt",
    "observe", "steer", "validate", "stop",
)


class RuntimeConstructionError(RuntimeError):
    """The configured runtime cannot safely perform a requested operation."""


class MissingRuntimeHandlerError(RuntimeConstructionError):
    """A production effect boundary was not explicitly installed."""

    def __init__(self, missing: tuple[str, ...]) -> None:
        self.missing = missing
        super().__init__("missing required runtime handlers: " + ", ".join(missing))


class RuntimeEffectError(RuntimeConstructionError):
    """An installed handler did not produce a durable successful receipt."""


@dataclass(frozen=True)
class CompleteLaunchPlan:
    """A launch plan plus the immutable bindings needed to resume it exactly."""

    launch_plan: LaunchPlan
    task_source_cid: str
    task_source_revision_cid: str
    objective_cid: str = ""
    objective_revision_cid: str = ""

    @property
    def launch_plan_cid(self) -> str:
        return self.launch_plan.launch_plan_cid


@dataclass(frozen=True)
class RuntimeEffectReceipt:
    """Normalized receipt returned by a real effect adapter."""

    receipt_cid: str
    effect_applied: bool
    values: Mapping[str, Any]

    @classmethod
    def coerce(cls, value: Any, *, handler: str) -> "RuntimeEffectReceipt":
        if isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            try:
                result = cls(
                    receipt_cid=str(value["receipt_cid"]),
                    effect_applied=bool(value["effect_applied"]),
                    values=dict(value),
                )
            except KeyError as exc:
                raise RuntimeEffectError(
                    f"{handler} handler omitted durable receipt field {exc.args[0]}"
                ) from exc
        else:
            raise RuntimeEffectError(f"{handler} handler returned no effect receipt")
        if not result.receipt_cid:
            raise RuntimeEffectError(f"{handler} handler returned an empty receipt CID")
        if not result.effect_applied:
            raise RuntimeEffectError(f"{handler} handler did not apply its declared effect")
        return result


class StandardSupervisorRuntimeFactory:
    """Build a runtime only from explicit, real adapters and durable storage."""

    def __init__(
        self,
        *,
        registry: RunRegistry,
        handlers: Mapping[str, Callable[..., Any]],
        production: bool = True,
    ) -> None:
        if not isinstance(registry, RunRegistry):
            raise RuntimeConstructionError("registry must be a durable RunRegistry")
        self.registry = registry
        self.handlers = dict(handlers)
        self.production = bool(production)
        if self.production:
            self.require_handlers(REQUIRED_RUNTIME_HANDLERS)
            # Mapping receipts / placeholder callables are not production truth.
            for name, handler in self.handlers.items():
                if getattr(handler, "__name__", "") in {"fixture_handler", "noop"}:
                    raise RuntimeConstructionError(
                        f"handler {name!r} is a fixture/no-op and cannot authorize effects"
                    )

    def require_handlers(self, names: tuple[str, ...] | list[str]) -> None:
        missing = tuple(sorted(name for name in names if not callable(self.handlers.get(name))))
        if missing:
            raise MissingRuntimeHandlerError(missing)

    def handler_manifest(self) -> Mapping[str, bool]:
        return {name: callable(self.handlers.get(name)) for name in REQUIRED_RUNTIME_HANDLERS}

    def invoke(self, name: str, *args: Any, **kwargs: Any) -> RuntimeEffectReceipt:
        handler = self.handlers.get(name)
        if not callable(handler):
            raise MissingRuntimeHandlerError((name,))
        return RuntimeEffectReceipt.coerce(handler(*args, **kwargs), handler=name)

    def create_intent_service(self) -> "SupervisorIntentService":
        from .intent_service import SupervisorIntentService

        return SupervisorIntentService(factory=self)




@dataclass(frozen=True)
class RequiredArgumentCoverageReceipt:
    """Proof every parser argument has a resolver receipt or signed default."""

    parser_identity: str
    covered_arguments: tuple[str, ...]
    signed_defaults: tuple[str, ...]
    missing_arguments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.missing_arguments:
            raise RuntimeConstructionError(
                "required parser arguments uncovered: "
                + ", ".join(self.missing_arguments)
            )
        if not self.parser_identity:
            raise RuntimeConstructionError("parser_identity is required")

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/required-argument-coverage@1",
                "parser_identity": self.parser_identity,
                "covered_arguments": list(self.covered_arguments),
                "signed_defaults": list(self.signed_defaults),
                "missing_arguments": list(self.missing_arguments),
            }
        )


@dataclass(frozen=True)
class PromptToRunSaga:
    """Complete public operation saga binding for prompt-to-run."""

    run_id: str
    planning_attempt_id: str
    program_revision_cid: str
    launch_plan_cid: str
    phases: tuple[str, ...] = (
        "PLAN_ADMITTED",
        "PROGRAM_REVISED",
        "INTENT_RESERVED",
        "EFFECT_STARTED",
        "TERMINAL_OBSERVED",
        "ADOPTED",
    )

    def __post_init__(self) -> None:
        if not self.run_id or not self.launch_plan_cid:
            raise RuntimeConstructionError("prompt-to-run saga requires run and plan")
        if "fixture" in self.launch_plan_cid.lower():
            raise RuntimeConstructionError("fixture CompleteLaunchPlan is not production truth")

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/prompt-to-run-saga@1",
                "run_id": self.run_id,
                "planning_attempt_id": self.planning_attempt_id,
                "program_revision_cid": self.program_revision_cid,
                "launch_plan_cid": self.launch_plan_cid,
                "phases": list(self.phases),
            }
        )


def reject_fixture_launch_plan(plan: CompleteLaunchPlan) -> CompleteLaunchPlan:
    """Production construction rejects fixture / mapping-only launch plans."""
    if not isinstance(plan, CompleteLaunchPlan):
        raise RuntimeConstructionError("launch plan must be CompleteLaunchPlan")
    if not plan.task_source_cid or not plan.task_source_revision_cid:
        raise RuntimeConstructionError("launch plan missing task-source bindings")
    if "fixture" in plan.launch_plan_cid.lower():
        raise RuntimeConstructionError("fixture CompleteLaunchPlan values are forbidden")
    return plan


def lifecycle_start_handler(
    orchestrator: Any,
    request_builder: Callable[[CompleteLaunchPlan, Any], Any],
) -> Callable[[CompleteLaunchPlan, Any], Mapping[str, Any]]:
    """Adapt the fenced lifecycle orchestrator to the runtime ``start`` slot.

    The request builder is intentionally injected: it owns the authenticated
    ``OperationRequest`` (including lease, authorization, and idempotency),
    while this adapter refuses a partial lifecycle receipt.
    """
    if not callable(request_builder) or not callable(getattr(orchestrator, "start", None)):
        raise RuntimeConstructionError("a lifecycle orchestrator and request builder are required")

    def start(plan: CompleteLaunchPlan, handle: Any) -> Mapping[str, Any]:
        request = request_builder(plan, handle)
        receipt = orchestrator.start(request)
        if not bool(getattr(receipt, "succeeded", False)):
            raise RuntimeEffectError("lifecycle start did not commit")
        process_cid = cid_for_dag_json({"lifecycle_receipt": receipt.receipt_id})
        return {
            "receipt_cid": process_cid,
            "effect_applied": True,
            "process_cid": process_cid,
            "lease_id": str(getattr(request, "lease_id", "")),
            "fencing_generation": int(getattr(request, "fencing_epoch", 0) or 0),
            "state_revision_cid": process_cid,
            "health_revision_cid": process_cid,
            "event_cursor": "lifecycle-started",
        }

    return start


__all__ = [
    "CompleteLaunchPlan",
    "MissingRuntimeHandlerError",
    "PromptToRunSaga",
    "REQUIRED_RUNTIME_HANDLERS",
    "RequiredArgumentCoverageReceipt",
    "RuntimeConstructionError",
    "RuntimeEffectError",
    "RuntimeEffectReceipt",
    "StandardSupervisorRuntimeFactory",
    "lifecycle_start_handler",
    "reject_fixture_launch_plan",
]
