"""Repository-state CLI commands as thin runtime calls (PCCE-040).

``init``, ``scan``, ``status``, and ``plan`` delegate to the stable runtime.
This module does not duplicate lifecycle stages, skip policy admission, or
mutate paths outside the selected repository/state directory. Importing it
performs no I/O, network, process, or filesystem mutation and does not start
a scan, plan, or repository initialization.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from ipfs_accelerate_py.proof_context.cli.app import CliContext, CliResult
from ipfs_accelerate_py.proof_context.errors import ProofContextError, redact_text
from ipfs_accelerate_py.proof_context.facade import (
    OPERATION_CONTRACTS,
    EngineIdentities,
    EngineRecord,
    FacadeError,
)
from ipfs_accelerate_py.proof_context.policy import PolicyError, admit_mode

COMMANDS: Final[tuple[str, ...]] = ("init", "scan", "status", "plan")
RUNTIME_OPEN: Final[str] = "ipfs_accelerate_py.proof_context.bootstrap.open_runtime"
RUNTIME_INIT: Final[str] = (
    "ipfs_accelerate_py.proof_context.bootstrap.create_ordinary_python_repository"
)
POLICY_ADMIT: Final[str] = "ipfs_accelerate_py.proof_context.policy.admit_mode"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _state_dir(context: CliContext) -> Path:
    if context.state_dir is not None:
        return context.state_dir
    return context.repository / ".pcce-state"


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _reject_escape(context: CliContext, path: Path) -> Path:
    """Keep mutations inside the selected repository tree."""

    repository = context.repository
    resolved = path if path.is_absolute() else repository / path
    if _is_within(resolved, repository):
        return resolved
    raise ProofContextError(
        "CLI cannot mutate paths outside the selected repository state directory",
        code="boundary_violation",
        details={"field": "repository"},
    )


def _identities(context: CliContext):
    from ipfs_accelerate_py.proof_context.lifecycle import mint_lifecycle_cid

    repository = context.repository
    repository_id = context.repository_id or f"ordinary-python:{repository.name}"
    state_cid = mint_lifecycle_cid(
        {
            "kind": "repository-state",
            "repository": str(repository),
            "task_id": context.task_id,
        }
    )
    run_id = context.run_id or mint_lifecycle_cid(
        {
            "kind": "run",
            "task_id": context.task_id,
            "correlation_id": context.correlation_id,
        }
    )
    return EngineIdentities(
        repository_id=repository_id,
        repository_state_cid=state_cid,
        task_id=context.task_id,
        run_id=run_id,
        trace_id=context.correlation_id,
    )


def _open_runtime(context: CliContext):
    from ipfs_accelerate_py.proof_context.bootstrap import RuntimeOptions, open_runtime

    mode = admit_mode(context.policy)
    state_dir = _reject_escape(context, _state_dir(context))
    options = RuntimeOptions(
        kit_root=state_dir,
        worktree_parent=state_dir / "worktrees",
        operator_id="proof-context-cli",
    )
    return open_runtime(
        context.repository,
        identities=_identities(context),
        mode=mode,
        options=options,
    )


def _result_from_record(context: CliContext, record: EngineRecord) -> CliResult:
    mapping = dict(record.to_mapping())
    payload = _jsonable(mapping.get("payload") or {})
    identities = mapping.get("identities")
    identity_map = dict(identities) if isinstance(identities, Mapping) else {}
    return CliResult(
        command=context.command,
        status=str(record.status),
        policy=context.policy,
        correlation_id=context.correlation_id,
        output_mode=context.output_mode,
        payload=payload,
        identities=identity_map,
        artifact_cid=str(record.artifact_cid),
        provenance=str(record.provenance),
        contract=str(mapping.get("contract") or OPERATION_CONTRACTS.get(record.operation)),
        error=None if record.status == "succeeded" else str(record.status),
    )


def _failure_result(
    context: CliContext,
    exc: BaseException,
    *,
    status: str,
    error: str,
) -> CliResult:
    reason = redact_text(str(exc))
    code = getattr(exc, "code", None) or getattr(exc, "reason", None) or error
    return CliResult(
        command=context.command,
        status=status,
        policy=context.policy,
        correlation_id=context.correlation_id,
        output_mode=context.output_mode,
        payload={"reason": reason, "code": str(code)},
        identities={
            "task_id": context.task_id,
            "trace_id": context.correlation_id,
        },
        error=str(code),
        provenance="live",
    )


def _run_engine(context: CliContext, operation: str) -> CliResult:
    try:
        bundle = _open_runtime(context)
        engine = bundle.engine
        if operation == "scan":
            record = engine.scan()
        elif operation == "status":
            record = engine.status()
        elif operation == "plan":
            record = engine.plan()
        else:
            raise ProofContextError(
                f"unsupported state command {operation!r}",
                code="unknown_field",
                details={"operation": operation},
            )
        return _result_from_record(context, record)
    except PolicyError as exc:
        return _failure_result(
            context,
            exc,
            status=getattr(exc, "reason", "rejected")
            if getattr(exc, "reason", None) in {"invalid", "rejected", "simulated"}
            else "rejected",
            error=str(getattr(exc, "reason", None) or "boundary_violation"),
        )
    except FacadeError as exc:
        reason = str(getattr(exc, "reason", None) or "malformed")
        status = "simulated" if reason == "simulated_promoted" else "invalid"
        if reason == "boundary_violation":
            status = "rejected"
        return _failure_result(context, exc, status=status, error=reason)
    except ProofContextError as exc:
        return _failure_result(
            context,
            exc,
            status=str(exc.status),
            error=str(exc.code),
        )
    except Exception as exc:  # noqa: BLE001 - CLI must emit typed failures
        return _failure_result(
            context,
            exc,
            status="infrastructure_failure",
            error="infrastructure_failure",
        )


def cmd_init(context: CliContext) -> CliResult:
    """Create an ordinary Python Git repository, then admit it through the runtime."""

    from ipfs_accelerate_py.proof_context.bootstrap import (
        create_ordinary_python_repository,
    )

    try:
        admit_mode(context.policy)
        target = context.repository
        parent = target.parent
        if parent.exists() and not parent.is_dir():
            raise ProofContextError(
                "repository parent must be a directory",
                code="malformed",
                details={"field": "repository"},
            )
        repo = create_ordinary_python_repository(target)
        bundle = _open_runtime(context)
        engine = bundle.engine
        status_record = engine.status()
        payload = {
            "initialized": True,
            "ordinary_python_git_repository": True,
            "repository": str(repo),
            "pyproject": str(repo / "pyproject.toml"),
            "canonical_head": status_record.payload.get("canonical_head"),
            "runtime": RUNTIME_INIT,
            "policy_admission": POLICY_ADMIT,
        }
        result = _result_from_record(context, status_record)
        return CliResult(
            command="init",
            status=result.status,
            policy=result.policy,
            correlation_id=result.correlation_id,
            output_mode=result.output_mode,
            payload={**dict(result.payload), **payload},
            identities=result.identities,
            artifact_cid=result.artifact_cid,
            provenance=result.provenance,
            contract=OPERATION_CONTRACTS.get("status"),
            error=result.error,
        )
    except PolicyError as exc:
        return _failure_result(
            context,
            exc,
            status="rejected",
            error=str(getattr(exc, "reason", None) or "boundary_violation"),
        )
    except FacadeError as exc:
        reason = str(getattr(exc, "reason", None) or "malformed")
        status = "rejected" if reason == "boundary_violation" else "invalid"
        return _failure_result(context, exc, status=status, error=reason)
    except ProofContextError as exc:
        return _failure_result(
            context,
            exc,
            status=str(exc.status),
            error=str(exc.code),
        )
    except Exception as exc:  # noqa: BLE001 - CLI must emit typed failures
        return _failure_result(
            context,
            exc,
            status="infrastructure_failure",
            error="infrastructure_failure",
        )


def cmd_scan(context: CliContext) -> CliResult:
    """Scan and persist semantic state through the runtime scan port."""

    return _run_engine(context, "scan")


def cmd_status(context: CliContext) -> CliResult:
    """Show typed runtime status. Untyped success dictionaries are not emitted."""

    return _run_engine(context, "status")


def cmd_plan(context: CliContext) -> CliResult:
    """Produce a proof-aware invalidation plan through the runtime plan port."""

    return _run_engine(context, "plan")


__all__ = [
    "COMMANDS",
    "POLICY_ADMIT",
    "RUNTIME_INIT",
    "RUNTIME_OPEN",
    "cmd_init",
    "cmd_plan",
    "cmd_scan",
    "cmd_status",
]
