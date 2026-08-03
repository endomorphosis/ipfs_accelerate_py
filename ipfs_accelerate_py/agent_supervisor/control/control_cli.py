"""Unified ``ipfs-accelerate agent`` adapter.

The adapter owns no supervisor policy.  It builds or decodes the canonical
``OperationRequest``, invokes ``SupervisorControlService.execute`` directly,
and writes the canonical ``OperationResult`` record.  This keeps CLI behavior
identical to Python and MCP callers and avoids translating control operations
into shell commands.

Prompt-workflow commands (``workflow-preview``, ``workflow-create``,
``restart``, ``rescue-preview``, ``rescue``) and plan create/steer commands
(``plan-create-preview``, ``plan-create``, ``plan-steer-preview``,
``plan-steer``) accept the same closed catalog parameters as Python/MCP.
Convenience flags never log raw prompt bodies and prefer file/stdin sources so
sensitive text does not appear in process listings.  Previews are forced to
dry-run; apply requires normal permit/lease/fence/idempotency/expected effects.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, MutableMapping
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

from .control_contracts import (
    MUTATION_OPERATIONS,
    SUPERVISOR_USAGE_READ_AUTHORITY,
    AuthorizationDecision,
    CapabilityDegradation,
    ControlBounds,
    ControlContractError,
    ControlDiscoveryManifest,
    ControlOperationCatalog,
    ControlSurface,
    IdempotencyKey,
    Operation,
    OperationRequest,
    OperationResult,
    OperationStatus,
    SupervisorUsageControlOperation,
    decode_operation_request,
    discover_usage_control_catalog,
    get_operation_catalog,
    usage_control_operations,
)
from .control_plane import (
    DIRECT_CONTROL_SERVICE_DISPATCHER_ID,
    ControlSurfacePublication,
    ProviderUsageControl,
    SupervisorControlService,
    control_operation_behavior_id,
    validate_control_surface_publication,
)


AGENT_CLI_EXIT_SUCCESS = 0
AGENT_CLI_EXIT_FAILED = 1
AGENT_CLI_EXIT_INVALID = 2
AGENT_CLI_EXIT_CONFLICT = 3
AGENT_CLI_EXIT_NOT_FOUND = 4
# Canonical results retain the distinct unavailable/timeout/cancellation
# status and typed error. Unavailable remains the legacy generic operation
# failure, while bounded timeout and cancellation have conventional stable
# process codes.
AGENT_CLI_EXIT_UNAVAILABLE = AGENT_CLI_EXIT_FAILED
AGENT_CLI_EXIT_TIMED_OUT = 124
AGENT_CLI_EXIT_CANCELLED = 130
MAX_WATCH_COUNT = 100
MAX_WATCH_INTERVAL_MS = 60_000

# Evidence identity for ASI-152 prompt CLI surface publication.
PROMPT_CLI_REQUIREMENT_ID = (
    "requirement:agent-supervisor/prompt-cli-surface@1"
)

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
    "plan-create-preview": Operation.PLAN_CREATE_PREVIEW,
    "plan-create": Operation.PLAN_CREATE_APPLY,
    "plan-steer-preview": Operation.PLAN_STEER_PREVIEW,
    "plan-steer": Operation.PLAN_STEER_APPLY,
    "refine": Operation.OBJECTIVE_REFINE,
    "reconcile": Operation.OBJECTIVE_RECONCILE,
    "refill": Operation.BACKLOG_REFILL,
    "workflow-preview": Operation.WORKFLOW_PREVIEW,
    "workflow-create": Operation.WORKFLOW_MATERIALIZE,
    "start": Operation.START,
    "pause": Operation.PAUSE,
    "resume": Operation.RESUME,
    "drain": Operation.DRAIN,
    "stop": Operation.STOP,
    "restart": Operation.RESTART,
    "retry": Operation.RETRY,
    "cancel": Operation.CANCEL,
    "quarantine": Operation.QUARANTINE,
    "validation-replay": Operation.VALIDATION_REPLAY,
    "rescue-preview": Operation.RESCUE_PREVIEW,
    "rescue": Operation.RESCUE,
}

PROMPT_WORKFLOW_CLI_COMMANDS: tuple[str, ...] = (
    "workflow-preview",
    "workflow-create",
    "restart",
    "rescue-preview",
    "rescue",
)

PLAN_CONTROL_CLI_COMMANDS: tuple[str, ...] = (
    "plan-create-preview",
    "plan-create",
    "plan-steer-preview",
    "plan-steer",
)

# Usage-governance CLI commands (thin adapters; not Operation catalog members).
USAGE_CLI_COMMANDS: dict[str, str] = {
    "usage-discover": "discover",
    "usage-status": SupervisorUsageControlOperation.STATUS.value,
    "usage-health": SupervisorUsageControlOperation.HEALTH.value,
    "usage-budgets": SupervisorUsageControlOperation.BUDGETS.value,
    "usage-headroom": SupervisorUsageControlOperation.HEADROOM.value,
    "usage-reservations": SupervisorUsageControlOperation.RESERVATIONS.value,
    "usage-receipts": SupervisorUsageControlOperation.RECEIPTS.value,
    "usage-route-preview": SupervisorUsageControlOperation.ROUTE_PREVIEW.value,
    "usage-blocked-work": SupervisorUsageControlOperation.BLOCKED_WORK.value,
    "usage-next-eligible": SupervisorUsageControlOperation.NEXT_ELIGIBLE.value,
    "usage-adapter-capabilities": (
        SupervisorUsageControlOperation.ADAPTER_CAPABILITIES.value
    ),
    "usage-set-budget": SupervisorUsageControlOperation.SET_BUDGET.value,
    "usage-set-policy": SupervisorUsageControlOperation.SET_POLICY.value,
    "usage-correct": SupervisorUsageControlOperation.CORRECT.value,
    "usage-reset": SupervisorUsageControlOperation.RESET.value,
}

_PROMPT_SURFACE_OPERATIONS: frozenset[Operation] = frozenset(
    {
        Operation.WORKFLOW_PREVIEW,
        Operation.WORKFLOW_MATERIALIZE,
        Operation.RESTART,
        Operation.RESCUE_PREVIEW,
        Operation.RESCUE,
    }
)

_PLAN_SURFACE_OPERATIONS: frozenset[Operation] = frozenset(
    {
        Operation.PLAN_CREATE_PREVIEW,
        Operation.PLAN_CREATE_APPLY,
        Operation.PLAN_STEER_PREVIEW,
        Operation.PLAN_STEER_APPLY,
    }
)

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
    "cursor",
    "event_cursor",
    "task_header_prefix",
    "target_id",
    "service_id",
    "task_id",
    "bundle_id",
    "lane_id",
    "stream_id",
    "receipt_id",
    "cache_namespace",
    "artifact_id",
    "validation_id",
    "reason",
    "requested_state",
)

_PROMPT_CONVENIENCE_ARGUMENTS = (
    "directory",
    "prompt",
    "prompt_file",
    "prompt_stdin",
    "output_mode",
    "markdown_path",
    "duckdb_path",
    "start_after",
    "allow_llm_fallback",
    "max_actions",
    "budget_json",
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
    parser.add_argument("--cursor", help="Opaque bounded query cursor.")
    parser.add_argument(
        "--event-cursor",
        help="Canonical event cursor token for replay.",
    )
    parser.add_argument(
        "--task-header-prefix",
        help="Required task ID prefix for the tasks operation.",
    )
    parser.add_argument("--target-id", help="Explicit lifecycle/task target.")
    parser.add_argument("--service-id", help="Catalog service selector.")
    parser.add_argument("--task-id", help="Catalog task selector.")
    parser.add_argument("--bundle-id", help="Catalog bundle selector.")
    parser.add_argument("--lane-id", help="Catalog lane selector.")
    parser.add_argument("--stream-id", help="Catalog event stream selector.")
    parser.add_argument("--receipt-id", help="Catalog receipt selector.")
    parser.add_argument(
        "--cache-namespace",
        help="Catalog cache namespace selector.",
    )
    parser.add_argument("--artifact-id", help="Catalog artifact selector.")
    parser.add_argument(
        "--validation-id",
        help="Catalog validation replay selector.",
    )
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
        help="Emit compact canonical JSON (default agent output).",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        default=False,
        help="Emit a concise human summary instead of full JSON.",
    )
    if operation in _PROMPT_SURFACE_OPERATIONS:
        _add_prompt_surface_arguments(parser, operation)
    if operation in _PLAN_SURFACE_OPERATIONS:
        _add_plan_surface_arguments(parser, operation)
    parser.set_defaults(agent_operation=operation.value)


def _add_plan_surface_arguments(
    parser: argparse.ArgumentParser, operation: Operation
) -> None:
    """Register thin plan create/steer convenience flags (closed catalog)."""

    if operation in {
        Operation.PLAN_CREATE_PREVIEW,
        Operation.PLAN_STEER_PREVIEW,
    }:
        parser.add_argument(
            "--mode",
            help="Optional plan mode token (deterministic / model-assisted).",
        )
    if operation in {
        Operation.PLAN_CREATE_APPLY,
        Operation.PLAN_STEER_APPLY,
    }:
        parser.add_argument(
            "--preview-ref",
            dest="preview_ref",
            help="Preview receipt reference for authorized apply.",
        )
        parser.add_argument(
            "--preview-root",
            dest="preview_root",
            help="Preview plan root for authorized apply.",
        )
        parser.add_argument(
            "--markdown-path",
            help="Root-relative Markdown task projection path.",
        )
        parser.add_argument(
            "--duckdb-path",
            help="Root-relative DuckDB task projection path.",
        )
        parser.add_argument(
            "--output-mode",
            choices=("markdown", "duckdb", "both"),
            help="Materialization projection mode: markdown, duckdb, or both.",
        )


def _add_prompt_surface_arguments(
    parser: argparse.ArgumentParser, operation: Operation
) -> None:
    """Register thin prompt-workflow / rescue convenience flags."""

    if operation is Operation.WORKFLOW_PREVIEW:
        parser.add_argument(
            "--directory",
            help="Directory parameter for workflow-preview (within repository root).",
        )
        prompt_source = parser.add_mutually_exclusive_group()
        prompt_source.add_argument(
            "--prompt",
            help=(
                "Inline prompt text. Prefer --prompt-file or --stdin so "
                "sensitive text is not visible in process listings."
            ),
        )
        prompt_source.add_argument(
            "--prompt-file",
            type=Path,
            help="Read the sole prompt body from a UTF-8 file.",
        )
        prompt_source.add_argument(
            "--stdin",
            dest="prompt_stdin",
            action="store_true",
            default=False,
            help="Read the sole prompt body from stdin.",
        )
    if operation in {
        Operation.WORKFLOW_PREVIEW,
        Operation.WORKFLOW_MATERIALIZE,
    }:
        parser.add_argument(
            "--output-mode",
            choices=("markdown", "duckdb", "both"),
            help="Materialization projection mode: markdown, duckdb, or both.",
        )
        parser.add_argument(
            "--markdown-path",
            help="Root-relative Markdown task projection path.",
        )
        parser.add_argument(
            "--duckdb-path",
            help="Root-relative DuckDB task projection path.",
        )
    if operation is Operation.WORKFLOW_MATERIALIZE:
        parser.add_argument(
            "--start",
            dest="start_after",
            action="store_true",
            default=False,
            help=(
                "Request start after materialize when the backend saga accepts "
                "it (workflow-create)."
            ),
        )
    if operation in {Operation.RESCUE_PREVIEW, Operation.RESCUE}:
        parser.add_argument(
            "--allow-llm-fallback",
            action="store_true",
            default=False,
            help="Permit bounded LLM fallback after programmatic recovery exhaustion.",
        )
        parser.add_argument(
            "--max-actions",
            type=int,
            help="Rescue action budget (model/fallback ladder bound).",
        )
    if operation in _PROMPT_SURFACE_OPERATIONS:
        parser.add_argument(
            "--budget-json",
            help=(
                "Optional model/fallback budget object merged into parameters "
                "when the catalog admits the fields (e.g. max_actions)."
            ),
        )


def register_agent_cli(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register the lightweight agent group on the product CLI parser."""

    # Refuse to publish a parser whose vocabulary or schemas drifted from the
    # immutable catalog. This is static validation: it neither constructs a
    # service nor resolves a backend/provider.
    cli_control_surface_publication()
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
    # Usage-governance commands are thin adapters over ProviderUsageControl.
    # They intentionally sit outside COMMAND_OPERATIONS so the closed Operation
    # catalog population remains exact for control-plane conformance.
    for command, operation_name in USAGE_CLI_COMMANDS.items():
        child = commands.add_parser(
            command,
            help=f"Run the {operation_name} usage-governance operation.",
        )
        child.set_defaults(
            agent_usage_operation=operation_name,
            agent_usage_command=command,
        )
        child.add_argument(
            "--authorities-json",
            help="JSON array of usage authorities.",
        )
        child.add_argument(
            "--target-id",
            help="Exact usage target / scope identifier.",
        )
        child.add_argument(
            "--limit",
            type=int,
            default=50,
            help="Bounded page size for list operations.",
        )
        child.add_argument(
            "--cursor",
            help="Opaque pagination cursor.",
        )
        child.add_argument(
            "--parameters-json",
            help="Additional JSON parameters for the usage operation.",
        )
        child.add_argument(
            "--output-json",
            action="store_true",
            help="Emit the canonical JSON envelope.",
        )
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


def validate_agent_cli_catalog(
    catalog: ControlOperationCatalog | None = None,
) -> ControlDiscoveryManifest:
    """Fail closed unless CLI publication exactly conforms to ``catalog``.

    Besides the closed operation population, publication binds every command
    to the catalog's canonical request/result schemas and transport-neutral
    behavior policy. The function is deliberately provider-free and
    process-free so parser construction and tools/list-style discovery remain
    safe.
    """

    selected = catalog or get_operation_catalog()
    if not isinstance(selected, ControlOperationCatalog):
        raise AgentCLIError("agent CLI catalog must be a ControlOperationCatalog")

    commands = tuple(COMMAND_OPERATIONS)
    if (
        len(commands) != len(set(commands))
        or any(
            not isinstance(command, str)
            or not command
            or command.strip() != command
            or any(character.isspace() for character in command)
            for command in commands
        )
    ):
        raise AgentCLIError("agent CLI command vocabulary is invalid")

    mapped = tuple(COMMAND_OPERATIONS.values())
    expected = tuple(selected.operations)
    if (
        len(mapped) != len(set(mapped))
        or frozenset(mapped) != frozenset(expected)
    ):
        missing = sorted(
            operation.value
            for operation in frozenset(expected).difference(mapped)
        )
        extra = sorted(
            str(getattr(operation, "value", operation))
            for operation in frozenset(mapped).difference(expected)
        )
        raise AgentCLIError(
            "agent CLI publication does not exactly cover the control catalog; "
            f"missing={missing}, extra={extra}"
        )

    manifest = agent_cli_discovery_manifest()
    if manifest.operations != tuple(
        sorted(selected.operations, key=lambda operation: operation.value)
    ):
        raise AgentCLIError("agent CLI discovery operation order drifted")

    for descriptor in selected:
        operation = descriptor.operation
        if (
            descriptor.request_schema_id
            != manifest.request_schema_ids[operation.value]
            or descriptor.result_schema_id
            != manifest.result_schema_ids[operation.value]
        ):
            raise AgentCLIError(
                f"agent CLI schema identity drifted for {operation.value}"
            )
        guarded = (
            descriptor.supports_dry_run,
            descriptor.requires_idempotency,
            descriptor.requires_authorization,
            descriptor.requires_lease,
            descriptor.requires_fencing,
        )
        if (
            descriptor.authority is not operation.authority
            or (operation.mutating and not all(guarded))
            or (not operation.mutating and any(guarded))
            or (
                operation.mutating
                and descriptor.degradation
                is not CapabilityDegradation.FAIL_CLOSED
            )
        ):
            raise AgentCLIError(
                f"agent CLI behavior policy drifted for {operation.value}"
            )
    return manifest


def cli_control_surface_publication(
    catalog: ControlOperationCatalog | None = None,
) -> ControlSurfacePublication:
    """Publish the complete static CLI surface or fail before registration."""

    selected = catalog or get_operation_catalog()
    manifest = validate_agent_cli_catalog(selected)
    publication = ControlSurfacePublication(
        surface=ControlSurface.CLI,
        catalog_id=selected.catalog_id,
        operations=manifest.operations,
        request_schema_ids=manifest.request_schema_ids,
        result_schema_ids=manifest.result_schema_ids,
        behavior_ids={
            descriptor.operation: control_operation_behavior_id(descriptor)
            for descriptor in selected
        },
        dispatcher_ids={
            operation: DIRECT_CONTROL_SERVICE_DISPATCHER_ID
            for operation in selected.operations
        },
        dispatch_mode="direct_service",
        provider_free=True,
        process_free=True,
    )
    return validate_control_surface_publication(publication, selected)


# Generation-2 spellings intentionally preserve function identity with the
# legacy-compatible entry points.  Both surfaces publish the one canonical
# operation catalog; these are negotiation names, not alternate behavior.
agent_cli_v2_discovery_manifest = agent_cli_discovery_manifest
v2_cli_control_surface_publication = cli_control_surface_publication


def build_agent_cli_command(
    request: OperationRequest,
    *,
    executable: str = "ipfs-accelerate",
) -> tuple[str, ...]:
    """Generate a shell-free canonical argv vector for one catalog operation."""

    if not isinstance(request, OperationRequest):
        raise AgentCLIError("request must be an OperationRequest")
    if not isinstance(executable, str) or not executable.strip():
        raise AgentCLIError("agent CLI executable must be non-empty")
    selected = get_operation_catalog()
    validate_agent_cli_catalog(selected)
    command_by_operation = {
        operation: command for command, operation in COMMAND_OPERATIONS.items()
    }
    try:
        command = command_by_operation[request.operation]
    except KeyError as exc:  # defensive after publication validation
        raise AgentCLIError(
            f"agent CLI has no command for {request.operation.value}"
        ) from exc
    return (
        executable,
        "agent",
        command,
        "--request-json",
        request.to_json(),
        "--output-json",
    )


# Concise compatibility spellings for conformance harnesses.
agent_cli_command = build_agent_cli_command
generate_agent_cli_command = build_agent_cli_command


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

    # Prompt-surface convenience flags also cannot overlay a full request.
    if getattr(args, "prompt", None) is not None:
        names.append("--prompt")
    if getattr(args, "prompt_file", None) is not None:
        names.append("--prompt-file")
    if bool(getattr(args, "prompt_stdin", False)):
        names.append("--stdin")
    for name in (
        "directory",
        "output_mode",
        "markdown_path",
        "duckdb_path",
        "max_actions",
        "budget_json",
    ):
        if getattr(args, name, None) is not None:
            flag = "--" + name.replace("_", "-")
            if flag not in names:
                names.append(flag)
    if bool(getattr(args, "start_after", False)):
        names.append("--start")
    if bool(getattr(args, "allow_llm_fallback", False)):
        names.append("--allow-llm-fallback")
    return tuple(names)


def _normalize_prompt_source(value: Any) -> dict[str, str]:
    """Project any prompt-source shape onto the closed control catalog fields.

    Durable parameters carry only ``kind`` + ``content_cid`` (and optional
    ``artifact_ref``).  Raw prompt bodies and alternate field spellings are
    never retained on the request record.
    """

    if not isinstance(value, Mapping):
        raise AgentCLIError("prompt_source must be an object")
    kind = value.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise AgentCLIError("prompt_source.kind is required")
    kind = kind.strip()
    content_cid = value.get("content_cid") or value.get("prompt_cid") or ""
    if not content_cid:
        inline_text = value.get("inline_text")
        if isinstance(inline_text, str) and inline_text:
            # Hash then drop the body so durable parameters stay body-free.
            from ..prompt.prompt_workflow import PromptSource

            content_cid = PromptSource.inline(inline_text).prompt_cid
    if not isinstance(content_cid, str) or not content_cid.strip():
        raise AgentCLIError("prompt_source requires content_cid")
    record: dict[str, str] = {
        "kind": kind,
        "content_cid": content_cid.strip(),
    }
    artifact = (
        value.get("artifact_ref")
        or value.get("artifact_handle")
        or value.get("source_path")
        or ""
    )
    if isinstance(artifact, str) and artifact.strip():
        # Reject path escape attempts; keep only a simple relative/opaque ref.
        cleaned = artifact.strip().replace("\\", "/")
        if cleaned.startswith("/") or ".." in cleaned.split("/"):
            raise AgentCLIError("prompt_source artifact_ref escapes allowed roots")
        record["artifact_ref"] = cleaned
    return record


def _resolve_prompt_source_from_args(
    args: argparse.Namespace,
    *,
    stdin_stream: TextIO | None = None,
) -> dict[str, str] | None:
    """Build a body-free prompt_source from exactly one CLI source."""

    prompt = getattr(args, "prompt", None)
    prompt_file = getattr(args, "prompt_file", None)
    prompt_stdin = bool(getattr(args, "prompt_stdin", False))
    provided = sum(
        1
        for item in (prompt is not None, prompt_file is not None, prompt_stdin)
        if item
    )
    if provided == 0:
        return None
    if provided != 1:
        raise AgentCLIError(
            "provide exactly one of --prompt, --prompt-file, or --stdin "
            "(prefer --prompt-file/--stdin so sensitive text is not listed in "
            "process arguments)"
        )
    from ..prompt.prompt_workflow import PromptSource

    if prompt is not None:
        if not isinstance(prompt, str) or not prompt:
            raise AgentCLIError("--prompt must be non-empty text")
        source = PromptSource.inline(prompt)
        return {"kind": "inline", "content_cid": source.prompt_cid}
    if prompt_file is not None:
        path = Path(prompt_file)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AgentCLIError("prompt file is not readable") from exc
        if not text:
            raise AgentCLIError("prompt file must be non-empty UTF-8 text")
        # Use the basename only; never retain absolute paths in parameters.
        source = PromptSource.file(path.name, text=text)
        return {
            "kind": "file",
            "content_cid": source.prompt_cid,
            "artifact_ref": path.name,
        }
    stream = stdin_stream if stdin_stream is not None else sys.stdin
    if stream is None or getattr(stream, "isatty", lambda: False)():
        raise AgentCLIError(
            "stdin prompt source requires piped non-empty UTF-8 text"
        )
    text = stream.read()
    if not text:
        raise AgentCLIError("stdin prompt source must be non-empty")
    source = PromptSource.stdin(text)
    return {"kind": "stdin", "content_cid": source.prompt_cid}


def _merge_plan_convenience_parameters(
    args: argparse.Namespace,
    parameters: MutableMapping[str, Any],
) -> None:
    """Fold thin plan create/steer convenience flags into catalog parameters."""

    def _set(key: str, value: Any) -> None:
        if key in parameters:
            raise AgentCLIError(
                f"{key} was supplied both directly and in --parameters-json"
            )
        parameters[key] = value

    mode = getattr(args, "mode", None)
    if mode is not None:
        _set("mode", str(mode))
    for key in (
        "preview_ref",
        "preview_root",
        "markdown_path",
        "duckdb_path",
        "output_mode",
    ):
        value = getattr(args, key, None)
        if value is not None:
            _set(key, str(value))


def _merge_prompt_convenience_parameters(
    args: argparse.Namespace,
    parameters: MutableMapping[str, Any],
    *,
    stdin_stream: TextIO | None = None,
) -> None:
    """Fold thin prompt/rescue convenience flags into catalog parameters."""

    def _set(key: str, value: Any) -> None:
        if key in parameters:
            raise AgentCLIError(
                f"{key} was supplied both directly and in --parameters-json"
            )
        parameters[key] = value

    directory = getattr(args, "directory", None)
    if directory is not None:
        _set("directory", str(directory))
    output_mode = getattr(args, "output_mode", None)
    if output_mode is not None:
        _set("output_mode", str(output_mode))
    markdown_path = getattr(args, "markdown_path", None)
    if markdown_path is not None:
        _set("markdown_path", str(markdown_path))
    duckdb_path = getattr(args, "duckdb_path", None)
    if duckdb_path is not None:
        _set("duckdb_path", str(duckdb_path))
    if bool(getattr(args, "allow_llm_fallback", False)):
        _set("allow_llm_fallback", True)
    max_actions = getattr(args, "max_actions", None)
    if max_actions is not None:
        _set("max_actions", int(max_actions))
    budget_raw = getattr(args, "budget_json", None)
    if budget_raw is not None:
        budget = _json_value(budget_raw, noun="budget")
        if not isinstance(budget, Mapping):
            raise AgentCLIError("budget JSON must contain an object")
        # Only fold catalog-admitted budget keys; ignore unknown silently-no —
        # fail closed on unknown keys that would be dropped into parameters.
        for key, value in budget.items():
            if key in {
                "max_actions",
                "allow_llm_fallback",
                "deadline_ms",
            }:
                if key in parameters and parameters[key] != value:
                    raise AgentCLIError(
                        f"budget field {key} conflicts with a direct flag"
                    )
                parameters.setdefault(key, value)
            else:
                raise AgentCLIError(
                    f"budget field {key!r} is not admitted on the control catalog"
                )
    # --start is saga intent for workflow-create; only admit it when the
    # operator also supplies it inside parameters-json for backends that
    # extend the catalog.  Direct injection of non-catalog fields is refused.
    if bool(getattr(args, "start_after", False)):
        # Keep a local marker for human output only; do not mutate parameters
        # with non-catalog keys.
        pass

    prompt_source = _resolve_prompt_source_from_args(
        args, stdin_stream=stdin_stream
    )
    if prompt_source is not None:
        if "prompt_source" in parameters:
            raise AgentCLIError(
                "prompt source was supplied both directly and in --parameters-json"
            )
        parameters["prompt_source"] = prompt_source

    if "prompt_source" in parameters:
        parameters["prompt_source"] = _normalize_prompt_source(
            parameters["prompt_source"]
        )


def _human_token(value: Any) -> str:
    """Coerce enums/records to a stable human-readable token."""

    if value is None:
        return ""
    # ``str`` enums are instances of ``str``, so check Enum before stringifying.
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _render_human_summary(record: Mapping[str, Any]) -> str:
    """Render a concise operator summary without dumping full payloads."""

    status = _human_token(record.get("status")) or "unknown"
    operation = _human_token(record.get("operation"))
    request_id = _human_token(
        record.get("request_id")
        or record.get("request_cid")
        or record.get("result_id")
        or ""
    )
    error = record.get("error") or {}
    code = ""
    if isinstance(error, Mapping):
        code = _human_token(error.get("code") or error.get("error_code") or "")
    data = record.get("data") if isinstance(record.get("data"), Mapping) else {}
    cursor = ""
    if isinstance(data, Mapping):
        cursor = _human_token(
            data.get("next_event_cursor")
            or data.get("event_cursor")
            or record.get("event_cursor")
            or ""
        )
    lines = [
        f"status={status}",
        f"operation={operation}" if operation else "",
        f"request_id={request_id}" if request_id else "",
        f"event_cursor={cursor}" if cursor else "",
        f"error={code}" if code else "",
    ]
    return "\n".join(line for line in lines if line)


def build_agent_request(
    args: argparse.Namespace,
    *,
    stdin_stream: TextIO | None = None,
) -> OperationRequest:
    """Build the one canonical request accepted by every control surface."""

    validate_agent_cli_catalog()
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
    if operation in _PROMPT_SURFACE_OPERATIONS:
        _merge_prompt_convenience_parameters(
            args, parameters, stdin_stream=stdin_stream
        )
    if operation in _PLAN_SURFACE_OPERATIONS:
        _merge_plan_convenience_parameters(args, parameters)
    elif "prompt_source" in parameters and operation not in _PROMPT_SURFACE_OPERATIONS:
        # Non-prompt operations must not smuggle prompt bodies through params.
        parameters["prompt_source"] = _normalize_prompt_source(
            parameters["prompt_source"]
        )
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
    dry_run = bool(args.dry_run)
    # Preview operations are always dry-run proposals.
    if operation in {
        Operation.WORKFLOW_PREVIEW,
        Operation.RESCUE_PREVIEW,
        Operation.PLAN_CREATE_PREVIEW,
        Operation.PLAN_STEER_PREVIEW,
        Operation.OBJECTIVE_PREVIEW,
    }:
        dry_run = True
    if operation in MUTATION_OPERATIONS and not dry_run:
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
        dry_run=dry_run,
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
    if result.status is OperationStatus.UNAVAILABLE:
        return AGENT_CLI_EXIT_UNAVAILABLE
    if result.status is OperationStatus.TIMED_OUT:
        return AGENT_CLI_EXIT_TIMED_OUT
    if result.status is OperationStatus.CANCELLED:
        return AGENT_CLI_EXIT_CANCELLED
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


def _usage_authorities_from_args(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "authorities_json", None)
    if raw is None or raw == "":
        return [SUPERVISOR_USAGE_READ_AUTHORITY]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AgentCLIError("authorities JSON is invalid") from exc
    if not isinstance(payload, list) or not all(
        isinstance(item, str) and item for item in payload
    ):
        raise AgentCLIError("authorities JSON must be an array of strings")
    return list(payload)


def run_usage_cli(
    args: argparse.Namespace,
    *,
    usage_control: ProviderUsageControl | None = None,
    service: SupervisorControlService | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Thin CLI adapter for ProviderUsageControl (schema/result/error parity)."""

    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    try:
        operation = str(getattr(args, "agent_usage_operation", "") or "")
        if not operation:
            raise AgentCLIError("usage operation is required")
        controller = usage_control
        if controller is None and service is not None:
            controller = service.usage_control
        if controller is None:
            controller = ProviderUsageControl()
        authorities = _usage_authorities_from_args(args)
        if operation == "discover":
            record = dict(controller.discover())
            record["success"] = True
            record["status"] = "success"
        else:
            parameters: dict[str, Any] = {}
            raw_params = getattr(args, "parameters_json", None)
            if raw_params:
                try:
                    loaded = json.loads(raw_params)
                except json.JSONDecodeError as exc:
                    raise AgentCLIError("parameters JSON is invalid") from exc
                if not isinstance(loaded, Mapping):
                    raise AgentCLIError("parameters JSON must be an object")
                parameters = dict(loaded)
            target_id = getattr(args, "target_id", None)
            if target_id:
                parameters.setdefault("target_id", target_id)
            if getattr(args, "limit", None) is not None:
                parameters.setdefault("limit", int(args.limit))
            if getattr(args, "cursor", None):
                parameters.setdefault("cursor", args.cursor)
            record = controller.execute(
                operation, authorities=authorities, **parameters
            )
        _write_record(
            stdout,
            record,
            compact=bool(getattr(args, "output_json", False)),
        )
        if record.get("success") is False:
            return AGENT_CLI_EXIT_FAILED
        return AGENT_CLI_EXIT_SUCCESS
    except AgentCLIError as exc:
        payload = {
            "schema": "ipfs_accelerate_py/agent-supervisor/cli-error@1",
            "status": "invalid_request",
            "error": {"code": "invalid_request", "detail": str(exc)},
            "error_code": "invalid_request",
            "success": False,
        }
        _write_record(stderr, payload, compact=True)
        return AGENT_CLI_EXIT_INVALID
    except Exception:
        payload = {
            "schema": "ipfs_accelerate_py/agent-supervisor/cli-error@1",
            "status": "internal_error",
            "error": {
                "code": "internal_error",
                "detail": "usage control operation failed",
            },
            "error_code": "internal_error",
            "success": False,
        }
        _write_record(stderr, payload, compact=True)
        return AGENT_CLI_EXIT_FAILED


def usage_cli_discovery_manifest() -> dict[str, Any]:
    """Static discovery for usage-governance CLI commands."""

    catalog = dict(discover_usage_control_catalog())
    catalog["surface"] = "cli"
    catalog["commands"] = dict(USAGE_CLI_COMMANDS)
    catalog["operations"] = list(usage_control_operations())
    return catalog


def run_agent_cli(
    args: argparse.Namespace,
    *,
    service: SupervisorControlService | None = None,
    service_factory: Callable[[OperationRequest], SupervisorControlService]
    | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdin: TextIO | None = None,
) -> int:
    """Execute an ``agent`` namespace and return a stable process exit code."""

    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    try:
        if service is not None and service_factory is not None:
            raise AgentCLIError(
                "service and service_factory are mutually exclusive"
            )
        if getattr(args, "agent_usage_operation", None):
            return run_usage_cli(
                args,
                service=service,
                stdout=stdout,
                stderr=stderr,
            )
        request = build_agent_request(args, stdin_stream=stdin)
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
        human = bool(getattr(args, "human", False)) and not bool(
            getattr(args, "output_json", False)
        )
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
            record = result.to_record()
            try:
                canonical = OperationResult.from_dict(record)
                canonical.validate_against(request)
                if canonical.canonical_bytes() != result.canonical_bytes():
                    raise ControlContractError(
                        "operation result is not canonically stable"
                    )
            except ControlContractError as exc:
                raise TypeError(
                    "agent control service returned a non-canonical "
                    "OperationResult"
                ) from exc
            payload = canonical.to_record()
            if human and count == 1:
                stdout.write(_render_human_summary(payload) + "\n")
            else:
                _write_record(
                    stdout,
                    payload,
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
        # Never echo exception arguments that might contain operator secrets;
        # AgentCLIError messages are constructed without prompt bodies.
        message = str(exc)
        payload = {
            "schema": "ipfs_accelerate_py/agent-supervisor/cli-error@1",
            "status": "invalid_request",
            "message": message,
        }
        _write_record(stderr, payload, compact=True)
        return AGENT_CLI_EXIT_INVALID
    except KeyboardInterrupt:
        payload = {
            "schema": "ipfs_accelerate_py/agent-supervisor/cli-error@1",
            "status": OperationStatus.CANCELLED.value,
            "message": "agent control operation cancelled",
        }
        _write_record(stderr, payload, compact=True)
        return AGENT_CLI_EXIT_CANCELLED
    except Exception:
        payload = {
            "schema": "ipfs_accelerate_py/agent-supervisor/cli-error@1",
            "status": "internal_error",
            "message": "agent control operation failed",
        }
        _write_record(stderr, payload, compact=True)
        return AGENT_CLI_EXIT_FAILED


__all__ = [
    "AGENT_CLI_EXIT_CANCELLED",
    "AGENT_CLI_EXIT_CONFLICT",
    "AGENT_CLI_EXIT_FAILED",
    "AGENT_CLI_EXIT_INVALID",
    "AGENT_CLI_EXIT_NOT_FOUND",
    "AGENT_CLI_EXIT_SUCCESS",
    "AGENT_CLI_EXIT_TIMED_OUT",
    "AGENT_CLI_EXIT_UNAVAILABLE",
    "AgentCLIError",
    "COMMAND_OPERATIONS",
    "MAX_WATCH_COUNT",
    "MAX_WATCH_INTERVAL_MS",
    "PROMPT_CLI_REQUIREMENT_ID",
    "PROMPT_WORKFLOW_CLI_COMMANDS",
    "USAGE_CLI_COMMANDS",
    "agent_cli_command",
    "build_agent_request",
    "build_agent_cli_command",
    "agent_cli_discovery_manifest",
    "agent_cli_v2_discovery_manifest",
    "cli_control_surface_publication",
    "default_agent_control_service",
    "exit_code_for_result",
    "generate_agent_cli_command",
    "register_agent_cli",
    "run_agent_cli",
    "run_usage_cli",
    "usage_cli_discovery_manifest",
    "validate_agent_cli_catalog",
    "v2_cli_control_surface_publication",
]
