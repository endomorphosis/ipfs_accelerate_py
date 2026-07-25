"""Unified ``ipfs-accelerate agent`` adapter.

The adapter owns no supervisor policy.  It builds or decodes the canonical
``OperationRequest``, invokes ``SupervisorControlService.execute`` directly,
and writes the canonical ``OperationResult`` record.  This keeps CLI behavior
identical to Python and MCP callers and avoids translating control operations
into shell commands.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TextIO

from .control_contracts import (
    MUTATION_OPERATIONS,
    AuthorizationDecision,
    ControlBounds,
    ControlContractError,
    ControlDiscoveryManifest,
    ControlSurface,
    IdempotencyKey,
    Operation,
    OperationRequest,
    OperationResult,
    OperationStatus,
    decode_operation_request,
)
from .control_plane import SupervisorControlService


AGENT_CLI_EXIT_SUCCESS = 0
AGENT_CLI_EXIT_FAILED = 1
AGENT_CLI_EXIT_INVALID = 2
AGENT_CLI_EXIT_CONFLICT = 3
AGENT_CLI_EXIT_NOT_FOUND = 4
MAX_WATCH_COUNT = 100
MAX_WATCH_INTERVAL_MS = 60_000

COMMAND_OPERATIONS: dict[str, Operation] = {
    "capabilities": Operation.CAPABILITIES,
    "status": Operation.STATUS,
    "health": Operation.HEALTH,
    "metrics": Operation.METRICS,
    "goals": Operation.GOALS,
    "tasks": Operation.TASKS,
    "bundles": Operation.BUNDLES,
    "lanes": Operation.LANES,
    "events": Operation.EVENTS,
    "receipts": Operation.RECEIPTS,
    "cache": Operation.CACHE_INSPECT,
    "artifact": Operation.ARTIFACT_QUERY,
    "preview": Operation.OBJECTIVE_PREVIEW,
    "plan": Operation.PLAN,
    "refine": Operation.OBJECTIVE_REFINE,
    "reconcile": Operation.OBJECTIVE_RECONCILE,
    "refill": Operation.BACKLOG_REFILL,
    "start": Operation.START,
    "pause": Operation.PAUSE,
    "resume": Operation.RESUME,
    "drain": Operation.DRAIN,
    "stop": Operation.STOP,
    "retry": Operation.RETRY,
    "cancel": Operation.CANCEL,
    "quarantine": Operation.QUARANTINE,
    "validation-replay": Operation.VALIDATION_REPLAY,
}

_IDENTITY_ARGUMENTS = (
    "repository_root",
    "state_root",
    "repository_id",
    "tree_id",
    "objective_id",
    "objective_revision",
    "policy_id",
    "policy_revision",
    "caller",
)

_DIRECT_PARAMETER_ARGUMENTS = (
    "path",
    "limit",
    "offset",
    "task_header_prefix",
    "target_id",
    "reason",
    "requested_state",
)


class AgentCLIError(ValueError):
    """A safe, user-correctable CLI request error."""


def _add_target_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repository-root",
        help="Explicit absolute allowlisted repository root.",
    )
    parser.add_argument(
        "--state-root",
        help="Explicit absolute allowlisted supervisor state root.",
    )
    parser.add_argument("--repository-id", help="Canonical repository identity.")
    parser.add_argument("--tree-id", help="Current repository tree identity.")
    parser.add_argument("--objective-id", help="Objective identity.")
    parser.add_argument("--objective-revision", help="Objective revision identity.")
    parser.add_argument("--policy-id", help="Control policy identity.")
    parser.add_argument("--policy-revision", help="Control policy revision.")
    parser.add_argument("--caller", help="Authenticated caller identity.")


def _add_request_arguments(
    parser: argparse.ArgumentParser, operation: Operation
) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--request-json",
        help="Complete canonical OperationRequest JSON object.",
    )
    source.add_argument(
        "--request-file",
        type=Path,
        help="File containing a complete canonical OperationRequest.",
    )
    _add_target_arguments(parser)
    parser.add_argument(
        "--parameters-json",
        help="Operation parameters as a JSON object (default: {}).",
    )
    parser.add_argument("--path", help="Explicit root-relative data path.")
    parser.add_argument("--limit", type=int, help="Bounded result window size.")
    parser.add_argument("--offset", type=int, help="Result window offset.")
    parser.add_argument(
        "--task-header-prefix",
        help="Required task ID prefix for the tasks operation.",
    )
    parser.add_argument("--target-id", help="Explicit lifecycle/task target.")
    parser.add_argument("--reason", help="Operator reason for a proposed mutation.")
    parser.add_argument(
        "--requested-state", help="Requested lifecycle state, when applicable."
    )
    parser.add_argument(
        "--expected-effects-json",
        help="ExpectedEffect records as a JSON array.",
    )
    parser.add_argument(
        "--idempotency-key",
        help=(
            "Caller-chosen replay key. Its operation, caller, repository, and "
            "objective scope are derived from the explicit target binding."
        ),
    )
    authorization = parser.add_mutually_exclusive_group()
    authorization.add_argument(
        "--authorization-json",
        help="Canonical AuthorizationDecision JSON object.",
    )
    authorization.add_argument(
        "--authorization-file",
        type=Path,
        help="File containing a canonical AuthorizationDecision.",
    )
    parser.add_argument(
        "--lease-id",
        help="Lease identity bound by the authorization decision.",
    )
    parser.add_argument(
        "--fencing-epoch",
        type=int,
        help="Non-negative fencing epoch bound by the authorization decision.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Return a proposal preview without invoking a mutation backend.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Maximum result/parameter item count.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        help="Maximum serialized request/result bytes.",
    )
    parser.add_argument(
        "--max-text-bytes",
        type=int,
        help="Maximum bytes in one text value/JSONL record.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        help="Operation timeout bound in milliseconds.",
    )
    parser.add_argument(
        "--watch-count",
        type=int,
        default=1,
        help=f"Bounded repeated reads (1-{MAX_WATCH_COUNT}); emits JSON Lines.",
    )
    parser.add_argument(
        "--watch-interval-ms",
        type=int,
        default=0,
        help=f"Delay between watched reads (0-{MAX_WATCH_INTERVAL_MS} ms).",
    )
    # The root CLI option only parses before a subcommand.  Keep this option on
    # every agent command so the conventional trailing form also works.
    parser.add_argument(
        "--output-json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit compact canonical JSON (agent output is always JSON).",
    )
    parser.set_defaults(agent_operation=operation.value)


def register_agent_cli(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register the lightweight agent group on the product CLI parser."""

    agent = subparsers.add_parser(
        "agent",
        help="Inspect and control the agent supervisor through typed contracts.",
        description=(
            "All commands use the shared OperationRequest/OperationResult "
            "contract. Real mutations require either a complete request or "
            "direct authorization, idempotency, lease, fencing, and effects."
        ),
    )
    commands = agent.add_subparsers(
        dest="agent_command", metavar="COMMAND", help="Agent control operation."
    )
    for command, operation in COMMAND_OPERATIONS.items():
        child = commands.add_parser(
            command,
            help=f"Run the {operation.value} control operation.",
        )
        _add_request_arguments(child, operation)
    return agent


def agent_cli_discovery_manifest() -> ControlDiscoveryManifest:
    """Return the static CLI vocabulary without constructing a service."""

    operations = tuple(
        sorted(COMMAND_OPERATIONS.values(), key=lambda item: item.value)
    )
    if len(operations) != len(set(operations)) or set(operations) != set(
        Operation
    ):
        raise AgentCLIError(
            "agent CLI discovery does not cover the closed operation vocabulary"
        )
    return ControlDiscoveryManifest(
        surface=ControlSurface.CLI,
        operations=operations,
    )


def _json_value(raw: str, *, noun: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AgentCLIError(f"{noun} must be valid JSON") from exc


def _request_payload(args: argparse.Namespace) -> Mapping[str, Any] | None:
    raw: str | None = getattr(args, "request_json", None)
    request_file: Path | None = getattr(args, "request_file", None)
    if request_file is not None:
        try:
            raw = request_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise AgentCLIError("request file is not readable") from exc
    if raw is None:
        return None
    value = _json_value(raw, noun="request")
    if not isinstance(value, Mapping):
        raise AgentCLIError("request JSON must contain an object")
    return value


def _json_file_value(path: Path, *, noun: str) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AgentCLIError(f"{noun} file is not readable") from exc
    return _json_value(raw, noun=noun)


def _authorization(args: argparse.Namespace) -> AuthorizationDecision | None:
    raw = getattr(args, "authorization_json", None)
    path = getattr(args, "authorization_file", None)
    if path is not None:
        value = _json_file_value(path, noun="authorization")
    elif raw is not None:
        value = _json_value(raw, noun="authorization")
    else:
        return None
    if not isinstance(value, Mapping):
        raise AgentCLIError("authorization JSON must contain an object")
    return AuthorizationDecision.from_dict(value)


def _request_overlay_arguments(args: argparse.Namespace) -> tuple[str, ...]:
    """Return request-building flags that cannot overlay a canonical request."""

    names: list[str] = []
    for name in _IDENTITY_ARGUMENTS + _DIRECT_PARAMETER_ARGUMENTS:
        if getattr(args, name, None) is not None:
            names.append("--" + name.replace("_", "-"))
    for name in (
        "idempotency_key",
        "authorization_json",
        "authorization_file",
        "lease_id",
        "fencing_epoch",
    ):
        if getattr(args, name, None) is not None:
            names.append("--" + name.replace("_", "-"))
    if bool(getattr(args, "dry_run", False)):
        names.append("--dry-run")
    if getattr(args, "parameters_json", None) is not None:
        names.append("--parameters-json")
    if getattr(args, "expected_effects_json", None) is not None:
        names.append("--expected-effects-json")

    for name in ("max_items", "max_bytes", "max_text_bytes", "timeout_ms"):
        if getattr(args, name, None) is not None:
            names.append("--" + name.replace("_", "-"))
    return tuple(names)


def build_agent_request(args: argparse.Namespace) -> OperationRequest:
    """Build the one canonical request accepted by every control surface."""

    operation = Operation(str(args.agent_operation))
    payload = _request_payload(args)
    if payload is not None:
        overlays = _request_overlay_arguments(args)
        if overlays:
            raise AgentCLIError(
                "--request-json/--request-file cannot be combined with "
                "request-building options: " + ", ".join(overlays)
            )
        # Canonical decoding happens before the caller-supplied service or
        # factory is resolved, so malformed real mutations cannot dispatch.
        request = decode_operation_request(payload)
        if request.operation is not operation:
            raise AgentCLIError(
                "request operation does not match the selected CLI command"
            )
        return request

    missing = [
        "--" + name.replace("_", "-")
        for name in _IDENTITY_ARGUMENTS
        if not str(getattr(args, name, "") or "").strip()
    ]
    if missing:
        raise AgentCLIError(
            "explicit target bindings are required: " + ", ".join(missing)
        )
    for name in ("repository_root", "state_root"):
        path = Path(str(getattr(args, name)))
        if not path.is_absolute():
            raise AgentCLIError(
                "--" + name.replace("_", "-") + " must be absolute"
            )
    parameters = _json_value(
        args.parameters_json
        if args.parameters_json is not None
        else "{}",
        noun="parameters",
    )
    if not isinstance(parameters, Mapping):
        raise AgentCLIError("parameters JSON must contain an object")
    parameters = dict(parameters)
    for argument in _DIRECT_PARAMETER_ARGUMENTS:
        key = argument
        value = getattr(args, argument, None)
        if value is not None:
            if key in parameters:
                raise AgentCLIError(
                    f"{key} was supplied both directly and in --parameters-json"
                )
            parameters[key] = value
    effects = _json_value(
        args.expected_effects_json
        if args.expected_effects_json is not None
        else "[]",
        noun="expected effects",
    )
    if not isinstance(effects, list):
        raise AgentCLIError("expected effects JSON must contain an array")

    idempotency = None
    if args.idempotency_key is not None:
        idempotency = IdempotencyKey(
            key=args.idempotency_key,
            operation=operation,
            caller=args.caller,
            repository_id=args.repository_id,
            objective_id=args.objective_id,
        )
    authorization = _authorization(args)
    if operation in MUTATION_OPERATIONS and not bool(args.dry_run):
        direct_guard_arguments = (
            idempotency,
            authorization,
            args.lease_id,
            args.fencing_epoch,
            effects,
        )
        if not any(
            value is not None and value != []
            for value in direct_guard_arguments
        ):
            raise AgentCLIError(
                "real mutations require a complete --request-json/"
                "--request-file or complete direct authorization, "
                "idempotency, lease, fencing, and expected effects"
            )

    defaults = ControlBounds()
    max_items = (
        args.max_items
        if args.max_items is not None
        else defaults.max_items
    )
    bounds = ControlBounds(
        max_items=max_items,
        max_serialized_bytes=(
            args.max_bytes
            if args.max_bytes is not None
            else defaults.max_serialized_bytes
        ),
        max_depth=defaults.max_depth,
        max_text_bytes=(
            args.max_text_bytes
            if args.max_text_bytes is not None
            else defaults.max_text_bytes
        ),
        max_paths=min(defaults.max_paths, max_items),
        max_effects=min(defaults.max_effects, max_items),
        timeout_ms=(
            args.timeout_ms
            if args.timeout_ms is not None
            else defaults.timeout_ms
        ),
    )
    return OperationRequest(
        operation=operation,
        **{name: getattr(args, name) for name in _IDENTITY_ARGUMENTS},
        parameters=parameters,
        expected_effects=tuple(effects),
        dry_run=bool(args.dry_run),
        idempotency=idempotency,
        authorization=authorization,
        lease_id=args.lease_id or "",
        fencing_epoch=args.fencing_epoch,
        bounds=bounds,
    )


def default_agent_control_service(
    request: OperationRequest,
) -> SupervisorControlService:
    """Build a local service with exactly the roots selected by the operator.

    This factory is intentionally CLI-only.  The policy-controlled MCP adapter
    requires a server-configured allowlist and never derives authority from a
    tool request.
    """

    return SupervisorControlService(
        repository_allowlist=(request.repository_root,),
        state_allowlist=(request.state_root,),
    )


def exit_code_for_result(result: OperationResult) -> int:
    if result.succeeded:
        return AGENT_CLI_EXIT_SUCCESS
    if result.status is OperationStatus.CONFLICT:
        return AGENT_CLI_EXIT_CONFLICT
    if result.status is OperationStatus.NOT_FOUND:
        return AGENT_CLI_EXIT_NOT_FOUND
    if result.status is OperationStatus.DENIED:
        return AGENT_CLI_EXIT_INVALID
    return AGENT_CLI_EXIT_FAILED


def _write_record(
    stream: TextIO, record: Mapping[str, Any], *, compact: bool
) -> None:
    if compact:
        encoded = json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    else:
        encoded = json.dumps(
            record, sort_keys=True, indent=2, ensure_ascii=False
        )
    stream.write(encoded + "\n")


def run_agent_cli(
    args: argparse.Namespace,
    *,
    service: SupervisorControlService | None = None,
    service_factory: Callable[[OperationRequest], SupervisorControlService]
    | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Execute an ``agent`` namespace and return a stable process exit code."""

    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    try:
        if service is not None and service_factory is not None:
            raise AgentCLIError(
                "service and service_factory are mutually exclusive"
            )
        request = build_agent_request(args)
        count = int(args.watch_count)
        interval_ms = int(args.watch_interval_ms)
        if not 1 <= count <= MAX_WATCH_COUNT:
            raise AgentCLIError(
                f"--watch-count must be between 1 and {MAX_WATCH_COUNT}"
            )
        if not 0 <= interval_ms <= MAX_WATCH_INTERVAL_MS:
            raise AgentCLIError(
                "--watch-interval-ms must be between 0 and "
                f"{MAX_WATCH_INTERVAL_MS}"
            )
        if count == 1 and interval_ms:
            raise AgentCLIError(
                "--watch-interval-ms requires --watch-count greater than 1"
            )
        if count > 1 and request.operation not in {
            Operation.STATUS,
            Operation.HEALTH,
            Operation.METRICS,
            Operation.EVENTS,
        }:
            raise AgentCLIError(
                "bounded watch is available only for status, health, metrics, and events"
            )
        selected_service = service or (
            service_factory or default_agent_control_service
        )(request)
        result: OperationResult | None = None
        exit_code = AGENT_CLI_EXIT_SUCCESS
        for index in range(count):
            result = selected_service.execute(request)
            if not isinstance(result, OperationResult):
                raise TypeError(
                    "agent control service returned a non-OperationResult value"
                )
            # Keep the adapter fail-closed even for a caller-supplied service.
            # The standard service validates its own output as well, but a
            # custom embedding must not make the CLI emit a canonical-looking
            # result bound to another request, tree, objective, or authority.
            try:
                result.validate_against(request)
            except ControlContractError as exc:
                # This is a service-output violation, not a user request
                # error. Keep it on the redacted internal-error path.
                raise TypeError(
                    "agent control service returned a mismatched OperationResult"
                ) from exc
            _write_record(
                stdout,
                result.to_record(),
                compact=count > 1
                or bool(getattr(args, "output_json", False)),
            )
            result_exit_code = exit_code_for_result(result)
            if (
                exit_code == AGENT_CLI_EXIT_SUCCESS
                and result_exit_code != AGENT_CLI_EXIT_SUCCESS
            ):
                exit_code = result_exit_code
            if index + 1 < count and interval_ms:
                time.sleep(interval_ms / 1000)
        assert result is not None
        return exit_code
    except (AgentCLIError, ControlContractError) as exc:
        payload = {
            "schema": "ipfs_accelerate_py/agent-supervisor/cli-error@1",
            "status": "invalid_request",
            "message": str(exc),
        }
        _write_record(stderr, payload, compact=True)
        return AGENT_CLI_EXIT_INVALID
    except Exception:
        payload = {
            "schema": "ipfs_accelerate_py/agent-supervisor/cli-error@1",
            "status": "internal_error",
            "message": "agent control operation failed",
        }
        _write_record(stderr, payload, compact=True)
        return AGENT_CLI_EXIT_FAILED


__all__ = [
    "AGENT_CLI_EXIT_CONFLICT",
    "AGENT_CLI_EXIT_FAILED",
    "AGENT_CLI_EXIT_INVALID",
    "AGENT_CLI_EXIT_NOT_FOUND",
    "AGENT_CLI_EXIT_SUCCESS",
    "AgentCLIError",
    "COMMAND_OPERATIONS",
    "MAX_WATCH_COUNT",
    "MAX_WATCH_INTERVAL_MS",
    "build_agent_request",
    "agent_cli_discovery_manifest",
    "default_agent_control_service",
    "exit_code_for_result",
    "register_agent_cli",
    "run_agent_cli",
]
