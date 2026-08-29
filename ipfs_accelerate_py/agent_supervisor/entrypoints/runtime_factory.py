"""Fail-closed composition for the prompt-to-run runtime.

Production construction deliberately has no convenience implementation.  A
caller must supply every effect boundary used by a launch; this prevents a
missing integration from looking like a completed supervisor run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
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


def complete_launch_plan_from_materialization(
    *,
    launch_plan: LaunchPlan,
    task_source_cid: str,
    task_source_revision_cid: str,
    objective_cid: str = "",
    objective_revision_cid: str = "",
) -> CompleteLaunchPlan:
    """Bind a launch plan to exact task-source identities after apply."""

    if not task_source_cid or not task_source_revision_cid:
        raise RuntimeConstructionError(
            "materialization did not publish task-source identities"
        )
    return reject_fixture_launch_plan(
        CompleteLaunchPlan(
            launch_plan=launch_plan,
            task_source_cid=str(task_source_cid),
            task_source_revision_cid=str(task_source_revision_cid),
            objective_cid=str(objective_cid or ""),
            objective_revision_cid=str(objective_revision_cid or ""),
        )
    )


def launch_plan_from_observation(
    observation: Mapping[str, Any],
    *,
    invocation_cid: str,
    idempotency_key: str,
    lease_required: bool = True,
) -> LaunchPlan:
    """Construct a LaunchPlan from observed production bindings, not fixtures."""

    from .contracts import (
        CoordinationShardBinding,
        ExpectedEffect,
        ReplicationBinding,
        ReplicationMode,
    )

    repository_root = str(observation.get("repository_root") or "")
    state_root = str(observation.get("state_root") or "")
    if not repository_root or not state_root:
        raise RuntimeConstructionError(
            "launch plan requires observed repository_root and state_root"
        )
    if not invocation_cid or not idempotency_key:
        raise RuntimeConstructionError(
            "launch plan requires invocation identity and idempotency"
        )
    if "fixture" in invocation_cid.lower() or "fixture" in idempotency_key.lower():
        raise RuntimeConstructionError("fixture launch-plan identities are forbidden")
    shard = CoordinationShardBinding(
        backend="duckdb",
        database_path=str(Path(state_root) / "coord.duckdb"),
        shard_id="shard-0",
        shard_count=1,
        shard_index=0,
        owner_principal_ref=str(observation.get("caller") or "principal:local"),
        coordinator_cid=cid_for_dag_json(
            {
                "schema": "ipfs_accelerate_py.agent_supervisor.observed-coordinator@1",
                "composition_cid": str(observation.get("composition_cid") or ""),
            }
        ),
        lease_namespace=str(observation.get("board_namespace") or "prompt-workflow"),
        fencing_generation=1,
        writable=True,
    )
    replication = ReplicationBinding(
        mode=ReplicationMode.PARQUET_IPLD,
        parquet_dataset_path=str(Path(state_root) / "parquet"),
        parquet_schema_cid=cid_for_dag_json(
            {"schema": "ipfs_accelerate_py.agent_supervisor.parquet-schema@1"}
        ),
        partition_keys=("repository_id", "run_id", "event_date", "shard_id"),
        ipld_manifest_schema_cid=cid_for_dag_json(
            {"schema": "ipfs_accelerate_py.agent_supervisor.ipld-manifest@1"}
        ),
    )
    return LaunchPlan(
        invocation_cid=invocation_cid,
        target_resolution_receipt_cid=str(
            observation.get("tree_id") or invocation_cid
        ),
        resolved_profile_cid=str(
            observation.get("program_root") or invocation_cid
        ),
        working_directory=repository_root,
        state_path=str(Path(state_root) / "run.json"),
        task_source_path=str(Path(state_root) / "projections" / "tasks.duckdb"),
        supervisor_argv=("python", "-m", "ipfs_accelerate_py.cli_entry", "supervisor"),
        daemon_argv=("python", "-m", "ipfs_accelerate_py.cli_entry", "supervisor"),
        environment_names=(),
        provider_route_cid=str(
            observation.get("provider_catalog_root") or invocation_cid
        ),
        resource_budget_cid=str(observation.get("policy_root") or invocation_cid),
        validation_profile_cid=str(
            observation.get("configuration_root") or invocation_cid
        ),
        lifecycle_profile_cid=str(
            observation.get("program_root") or invocation_cid
        ),
        coordination_shard=shard,
        replication=replication,
        expected_effects=(ExpectedEffect.LAUNCH_LOCAL_PROCESS,),
        idempotency_key=idempotency_key,
        adoption_key="adoption:" + idempotency_key,
        lease_required=bool(lease_required),
        authorization_required=True,
        dry_run=False,
    )


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
    "complete_launch_plan_from_materialization",
    "launch_plan_from_observation",
    "lifecycle_start_handler",
    "reject_fixture_launch_plan",
]
