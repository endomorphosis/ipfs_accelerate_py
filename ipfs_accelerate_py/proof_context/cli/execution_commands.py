"""Governed execution CLI calls (PCCE-041).

This module is intentionally a narrow presentation boundary.  It creates a
proposal only through the closed adapter registry, then delegates execution,
verification, and recovery to the public runtime facade.  It never parses or
applies a patch, opens a worktree directly, or treats a matching repository
name as a run identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Final

from ipfs_accelerate_py.proof_context.adapters.base import (
    CancellationToken,
    execute_propose,
)
from ipfs_accelerate_py.proof_context.adapters.models import (
    ContextPack,
    ModelRouteDecision,
    TaskSpecification,
)
from ipfs_accelerate_py.proof_context.adapters.registry import (
    ADAPTER_NAMES,
    AdapterConfiguration,
    DEFAULT_ADAPTER_REGISTRY,
)
from ipfs_accelerate_py.proof_context.cli.app import CliResult
from ipfs_accelerate_py.proof_context.cli.state_commands import (
    _failure_result,
    _identities,
    _reject_escape,
    _result_from_record,
    _state_dir,
)
from ipfs_accelerate_py.proof_context.errors import (
    IdentityInconsistentError,
    MalformedError,
    ProofContextError,
    from_provider_error,
)
from ipfs_accelerate_py.proof_context.facade import EngineRecord
from ipfs_accelerate_py.proof_context.policy import admit_mode

COMMANDS: Final[tuple[str, ...]] = ("run", "verify", "resume")
# A process-local memo prevents duplicate CLI dispatch from re-entering a
# settled runtime. Durable recovery remains the lifecycle/persistence port's
# responsibility; this memo is only an idempotency guard at this call layer.
_RESUME_RESULTS: dict[tuple[str, str, str, str], CliResult] = {}


@dataclass(frozen=True)
class ExecutionContext:
    """Explicit execution identity; compatible with the PCCE-040 context shape."""

    command: str
    repository: Any
    policy: str
    task_id: str
    correlation_id: str
    output_mode: str
    run_id: str
    repository_id: str | None = None
    state_dir: Any | None = None
    patch_id: str | None = None

    def __post_init__(self) -> None:
        if self.command not in COMMANDS:
            raise MalformedError("unsupported execution command")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise MalformedError("run_id is required for execution commands")
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise MalformedError("task_id is required for execution commands")
        admit_mode(self.policy)


@dataclass(frozen=True)
class RunRequest:
    """All adapter inputs are explicit immutable wire records.

    ``adapter`` is a registry configuration, never an adapter object.  This
    prevents an unregistered implementation from being smuggled into run.
    """

    context: ExecutionContext
    adapter: AdapterConfiguration | Mapping[str, Any]
    task: TaskSpecification
    context_pack: ContextPack
    route: ModelRouteDecision
    cancellation: CancellationToken | None = None


def _open_runtime(context: ExecutionContext):
    from ipfs_accelerate_py.proof_context.bootstrap import RuntimeOptions, open_runtime

    state_dir = _reject_escape(context, _state_dir(context))
    identities = replace(_identities(context), patch_id=context.patch_id)
    return open_runtime(
        context.repository,
        identities=identities,
        mode=admit_mode(context.policy),
        options=RuntimeOptions(
            kit_root=state_dir,
            worktree_parent=state_dir / "worktrees",
            operator_id="proof-context-cli",
        ),
    )


def _failure(context: ExecutionContext, exc: BaseException) -> CliResult:
    if isinstance(exc, ProofContextError):
        return _failure_result(context, exc, status=str(exc.status), error=str(exc.code))
    typed = from_provider_error(exc)
    return _failure_result(context, typed, status=str(typed.status), error=str(typed.code))


def _identity_value(record: EngineRecord | Mapping[str, Any], field: str) -> Any:
    identities = record.identities.to_mapping() if isinstance(record, EngineRecord) else record.get("identities")
    if not isinstance(identities, Mapping):
        raise MalformedError("run evidence identities are required")
    return identities.get(field)


def _require_same(expected: Any, actual: Any, *, field: str) -> None:
    if not isinstance(expected, str) or not expected or expected != actual:
        raise IdentityInconsistentError(f"{field} does not match the exact run identity")


def _admit_adapter_request(request: RunRequest) -> AdapterConfiguration:
    context = request.context
    config = request.adapter if isinstance(request.adapter, AdapterConfiguration) else AdapterConfiguration.from_mapping(request.adapter)
    if config.name not in ADAPTER_NAMES:  # Defensive even though registry already closes names.
        raise MalformedError("adapter is not registered")
    _require_same(context.task_id, request.task.task_id, field="task_id")
    expected = _identities(context)
    _require_same(expected.repository_state_cid, request.task.repository_state_cid, field="repository_state_cid")
    return config


def cmd_run(request: RunRequest) -> CliResult:
    """Create one registry-admitted proposal and submit it to the lifecycle."""

    context = request.context
    try:
        config = _admit_adapter_request(request)
        # The registry is deliberately the only construction point.  It has no
        # acceptance authority; execute_propose admits the contract binding.
        adapter = DEFAULT_ADAPTER_REGISTRY.create(config)
        result = execute_propose(
            adapter, request.task, request.context_pack, request.route, request.cancellation
        )
        if request.cancellation is not None:
            request.cancellation.check()
        bound = replace(context, patch_id=result.proposal.patch_cid)
        proposal = {
            "declared_files": list(result.proposal.declared_files),
            # Patch bytes remain adapter evidence.  The lifecycle owns patch
            # application and is not handed an imperative CLI patch operation.
            "adapter_id": config.name,
            "adapter_proposal_cid": result.proposal.proposal_cid,
            "adapter_invocation_cid": result.invocation.invocation_cid,
        }
        record = _open_runtime(bound).engine.run(proposal)
        rendered = _result_from_record(bound, record)
        payload = dict(rendered.payload)
        payload.update(
            {
                "adapter": config.name,
                "adapter_registry_admitted": True,
                "adapter_proposal_cid": result.proposal.proposal_cid,
                "adapter_invocation_cid": result.invocation.invocation_cid,
                "patch_id": result.proposal.patch_cid,
            }
        )
        return replace(rendered, payload=payload)
    except BaseException as exc:  # typed command surface; provider errors are closed by runtime
        return _failure(context, exc)


def _admit_run_evidence(context: ExecutionContext, run: EngineRecord | Mapping[str, Any]) -> str:
    operation = run.operation if isinstance(run, EngineRecord) else run.get("operation")
    status = run.status if isinstance(run, EngineRecord) else run.get("status")
    provenance = run.provenance if isinstance(run, EngineRecord) else run.get("provenance")
    if operation != "run" or status != "succeeded" or provenance != "live":
        raise IdentityInconsistentError("verify requires an exact live succeeded run record")
    for field, value in (("task_id", context.task_id), ("run_id", context.run_id), ("trace_id", context.correlation_id)):
        _require_same(value, _identity_value(run, field), field=field)
    expected = _identities(context)
    for field, value in (("repository_id", expected.repository_id), ("repository_state_cid", expected.repository_state_cid)):
        _require_same(value, _identity_value(run, field), field=field)
    patch_id = _identity_value(run, "patch_id")
    if not isinstance(patch_id, str) or not patch_id:
        raise MalformedError("run evidence patch_id is required")
    if context.patch_id is not None:
        _require_same(context.patch_id, patch_id, field="patch_id")
    return patch_id


def cmd_verify(
    context: ExecutionContext,
    *,
    run: EngineRecord | Mapping[str, Any] | None = None,
    patch_id: str | None = None,
) -> CliResult:
    """Verify by one exact run record or one explicitly bound patch identity."""

    try:
        if (run is None) == (patch_id is None):
            raise MalformedError("verify requires exactly one of run or patch_id")
        if run is not None:
            bound_patch_id = _admit_run_evidence(context, run)
        else:
            if not isinstance(patch_id, str) or not patch_id:
                raise MalformedError("patch_id is required")
            # A standalone patch selector is safe only when the invocation was
            # itself bound to that exact patch; it cannot override a run-bound
            # context or silently select a different patch.
            _require_same(context.patch_id, patch_id, field="patch_id")
            bound_patch_id = patch_id
        bound = replace(context, patch_id=bound_patch_id)
        record = _open_runtime(bound).engine.verify()
        return _result_from_record(bound, record)
    except BaseException as exc:
        return _failure(context, exc)


def cmd_resume(
    context: ExecutionContext, *, checkpoint: Mapping[str, Any] | None = None
) -> CliResult:
    """Resume solely by the named run identity; settled runs remain idempotent."""

    try:
        cache_key = (
            str(context.repository),
            context.task_id,
            context.run_id,
            context.correlation_id,
        )
        if checkpoint is not None:
            identities = checkpoint.get("identities")
            if not isinstance(identities, Mapping):
                raise MalformedError("resume checkpoint identities are required")
            for field, value in (("task_id", context.task_id), ("run_id", context.run_id), ("trace_id", context.correlation_id)):
                _require_same(value, identities.get(field), field=field)
            checkpoint_patch = identities.get("patch_id")
            if context.patch_id is not None and checkpoint_patch is not None:
                _require_same(context.patch_id, checkpoint_patch, field="patch_id")
        cached = _RESUME_RESULTS.get(cache_key)
        if cached is not None:
            return cached
        record = _open_runtime(context).engine.resume(checkpoint)
        rendered = _result_from_record(context, record)
        result = replace(rendered, payload={**dict(rendered.payload), "resumed_by_run_id": context.run_id, "idempotent": True})
        _RESUME_RESULTS[cache_key] = result
        return result
    except BaseException as exc:
        return _failure(context, exc)


__all__ = [
    "COMMANDS",
    "ExecutionContext",
    "RunRequest",
    "cmd_run",
    "cmd_verify",
    "cmd_resume",
]
