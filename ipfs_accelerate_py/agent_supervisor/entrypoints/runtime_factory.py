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
    "CompleteLaunchPlan", "MissingRuntimeHandlerError", "REQUIRED_RUNTIME_HANDLERS",
    "RuntimeConstructionError", "RuntimeEffectError", "RuntimeEffectReceipt",
    "StandardSupervisorRuntimeFactory", "lifecycle_start_handler",
]
