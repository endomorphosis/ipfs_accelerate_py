"""Thin, fail-closed CLI adapter for federation post-admission control.

The CLI accepts only a complete canonical :class:`FederationCommand` record
and sends it directly to an injected ``FederationControlService``.  It never
opens a database, resolves a filesystem location, starts a state owner, or
falls back to an embedded control plane.  Federation creation remains solely
on the authenticated trigger gateway and is deliberately not published here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from typing import Any, Final, TextIO

from ..control.service import execute_federation_command
from ..task_sources.control_plane_contracts import (
    MAX_COMMAND_BYTES,
    ControlPlaneContractError,
    canonical_json_bytes,
)
from .contracts import FederationCommand, FederationContractError, FederationOperation
from .control_service import (
    POST_ADMISSION_OPERATIONS,
    FederationControlResponse,
    FederationControlService,
    FederationControlServiceError,
)

FEDERATION_CLI_INTERFACE: Final[str] = "FederationControlCLI@1"
FEDERATION_CLI_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/cli-result@1"
)
FEDERATION_CLI_ERROR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/cli-error@1"
)
FEDERATION_CLI_EXIT_SUCCESS = 0
FEDERATION_CLI_EXIT_FAILED = 1
FEDERATION_CLI_EXIT_INVALID = 2
FEDERATION_CLI_MAX_COMMAND_JSON_BYTES: Final[int] = MAX_COMMAND_BYTES


class FederationCLIError(ValueError):
    """A safe, caller-correctable federation CLI input error."""


def _command_name(operation: FederationOperation) -> str:
    return operation.value.removeprefix("federation.").replace("_", "-")


FEDERATION_CLI_COMMANDS: Final[Mapping[str, FederationOperation]] = {
    _command_name(operation): operation
    for operation in sorted(POST_ADMISSION_OPERATIONS, key=lambda item: item.value)
}


def federation_cli_discovery_manifest() -> dict[str, Any]:
    """Return static adapter vocabulary without resolving runtime capability."""

    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/causal-federation/cli-discovery@1",
        "interface": FEDERATION_CLI_INTERFACE,
        "group": "federation",
        "commands": {name: operation.value for name, operation in FEDERATION_CLI_COMMANDS.items()},
        "request_schema": FederationCommand.SCHEMA,
        "result_schema": FEDERATION_CLI_RESULT_SCHEMA,
        "dispatch": "direct_typed_service",
        "shell_out": False,
        "embedded_fallback": False,
        "create_via_trigger_gateway": True,
    }


def register_federation_cli(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register the cold, parser-only ``federation`` CLI namespace."""

    group = subparsers.add_parser(
        "federation",
        help="Control an admitted federation through typed Quack-owner commands.",
    )
    commands = group.add_subparsers(dest="federation_cli_command", metavar="COMMAND")
    for name, operation in FEDERATION_CLI_COMMANDS.items():
        parser = commands.add_parser(name, help=f"Execute {operation.value}.")
        parser.add_argument(
            "--command-json",
            required=True,
            help=(
                "Complete byte-canonical FederationCommand JSON, supplied inline "
                f"(maximum {FEDERATION_CLI_MAX_COMMAND_JSON_BYTES} UTF-8 bytes)."
            ),
        )
        parser.add_argument(
            "--output-json",
            action="store_true",
            help="Emit compact JSON instead of formatted JSON.",
        )
        parser.set_defaults(federation_operation=operation.value)
    return group


def _load_json(text: str, *, source: str) -> Mapping[str, Any]:
    if not isinstance(text, str):
        raise FederationCLIError(f"{source} must be inline JSON text")
    try:
        raw = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise FederationCLIError(f"{source} is not valid UTF-8 JSON text") from exc
    if len(raw) > FEDERATION_CLI_MAX_COMMAND_JSON_BYTES:
        raise FederationCLIError(
            f"{source} exceeds {FEDERATION_CLI_MAX_COMMAND_JSON_BYTES} UTF-8 bytes"
        )
    try:
        decoded = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise FederationCLIError(f"{source} is not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise FederationCLIError(f"{source} must contain an object")
    return decoded


def build_federation_command(args: argparse.Namespace) -> FederationCommand:
    """Decode one unmodified closed command before any service is resolved."""

    selected = getattr(args, "federation_operation", None)
    if not selected:
        raise FederationCLIError("a federation command is required")
    try:
        operation = FederationOperation(str(selected))
    except ValueError as exc:
        raise FederationCLIError("federation operation is outside the closed catalog") from exc
    if operation not in POST_ADMISSION_OPERATIONS:
        raise FederationCLIError("federation.create is accepted only by the trigger gateway")

    raw = getattr(args, "command_json", None)
    if not isinstance(raw, str):
        raise FederationCLIError("a complete inline canonical command is required")
    try:
        command = FederationCommand.from_dict(_load_json(raw, source="command"))
        if raw.encode("utf-8") != canonical_json_bytes(command.to_dict()):
            raise FederationCLIError("command must use canonical JSON encoding")
    except (ControlPlaneContractError, FederationContractError) as exc:
        raise FederationCLIError(str(exc)) from exc
    if command.operation not in POST_ADMISSION_OPERATIONS:
        raise FederationCLIError("federation.create is accepted only by the trigger gateway")
    if command.operation is not operation:
        raise FederationCLIError("command operation does not match the selected CLI command")
    return command


def federation_control_response_record(
    command: FederationCommand,
    response: FederationControlResponse,
) -> dict[str, Any]:
    """Serialize only canonical command/result/audit evidence for transport."""

    if not isinstance(command, FederationCommand):
        raise FederationControlServiceError("CLI command must be FederationCommand")
    if not isinstance(response, FederationControlResponse):
        raise FederationControlServiceError("control service returned no typed response")
    # Reconstruct records from their transport forms so a malformed custom
    # embedding cannot emit an apparently canonical result.
    result = type(response.result).from_dict(response.result.to_dict())
    audit = type(response.audit).from_dict(response.audit.to_dict())
    if result.binding != command.binding or command.cid not in result.evidence_refs:
        raise FederationControlServiceError("control response is not bound to the command")
    if audit.command_cid != command.cid or audit.result_ref != result.cid:
        raise FederationControlServiceError("control audit is not bound to the command result")
    return {
        "schema": FEDERATION_CLI_RESULT_SCHEMA,
        "interface": FEDERATION_CLI_INTERFACE,
        "command": command.to_dict(),
        "result": result.to_dict(),
        "audit": audit.to_dict(),
    }


def _write_json(stream: TextIO, payload: Mapping[str, Any], *, compact: bool) -> None:
    stream.write(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":") if compact else None,
            indent=None if compact else 2,
        )
        + "\n"
    )


def run_federation_cli(
    args: argparse.Namespace,
    *,
    service: FederationControlService | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Dispatch one post-admission command through the supplied typed service."""

    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    try:
        command = build_federation_command(args)
        if not isinstance(service, FederationControlService):
            raise FederationCLIError("a qualified FederationControlService is required")
        response = execute_federation_command(service, command)
        _write_json(
            stdout,
            federation_control_response_record(command, response),
            compact=bool(getattr(args, "output_json", False)),
        )
        return FEDERATION_CLI_EXIT_SUCCESS
    except (FederationCLIError, FederationContractError, FederationControlServiceError) as exc:
        _write_json(
            stderr,
            {
                "schema": FEDERATION_CLI_ERROR_SCHEMA,
                "interface": FEDERATION_CLI_INTERFACE,
                "status": "invalid_request",
                "message": str(exc),
            },
            compact=True,
        )
        return FEDERATION_CLI_EXIT_INVALID
    except Exception:
        # State-owner errors can carry implementation details; do not publish
        # them through a command-line transport.
        _write_json(
            stderr,
            {
                "schema": FEDERATION_CLI_ERROR_SCHEMA,
                "interface": FEDERATION_CLI_INTERFACE,
                "status": "unavailable",
                "message": "federation control operation failed",
            },
            compact=True,
        )
        return FEDERATION_CLI_EXIT_FAILED


__all__ = [
    "FEDERATION_CLI_COMMANDS",
    "FEDERATION_CLI_ERROR_SCHEMA",
    "FEDERATION_CLI_EXIT_FAILED",
    "FEDERATION_CLI_EXIT_INVALID",
    "FEDERATION_CLI_EXIT_SUCCESS",
    "FEDERATION_CLI_INTERFACE",
    "FEDERATION_CLI_MAX_COMMAND_JSON_BYTES",
    "FEDERATION_CLI_RESULT_SCHEMA",
    "FederationCLIError",
    "build_federation_command",
    "federation_cli_discovery_manifest",
    "federation_control_response_record",
    "register_federation_cli",
    "run_federation_cli",
]
