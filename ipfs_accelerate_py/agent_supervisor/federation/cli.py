"""Thin, fail-closed CLI adapter for federation control.

The CLI accepts only complete canonical records supplied inline.  CREATE is
sent only to an injected authenticated ``FederationControlGateway``; every
post-admission command is sent only to an injected ``FederationControlService``.
The adapter never opens a database, resolves a filesystem location, starts a
state owner, or falls back to an embedded control plane.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Final, TextIO

from ..control.service import execute_federation_command
from ..task_sources.control_plane_contracts import (
    ControlPlaneContractError,
    canonical_json_bytes,
)
from .contracts import (
    ClosedContract,
    FederationCommand,
    FederationCommandResult,
    FederationContractError,
    FederationIdentity,
    FederationOperation,
    FederationReceipt,
    FederationRequest,
)
from .control_service import (
    POST_ADMISSION_OPERATIONS,
    FederationControlAuditReceipt,
    FederationControlResponse,
    FederationControlService,
    FederationControlServiceError,
)
from .trigger import AuthenticationEvidence, FederationControlGateway

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

# A structurally valid create request carries the authority binding four
# binding-scale times: directly, in each of two bounded budgets, and with
# conservative transport/headroom accounting.  A binding itself has three
# 256-item compact-identity collections.  Four MiB therefore exceeds the
# closed protocol maximum while remaining a finite pre-resolution limit.
_MAX_BINDING_COLLECTIONS: Final[int] = 3
_MAX_BINDING_COLLECTION_ITEMS: Final[int] = 256
_MAX_COMPACT_ID_BYTES: Final[int] = 512
_MAX_BINDING_SCALE_COPIES: Final[int] = 4
_STRUCTURAL_IDENTITY_BYTES: Final[int] = (
    _MAX_BINDING_SCALE_COPIES
    * _MAX_BINDING_COLLECTIONS
    * _MAX_BINDING_COLLECTION_ITEMS
    * _MAX_COMPACT_ID_BYTES
)
FEDERATION_CONTROL_MAX_CANONICAL_BYTES: Final[int] = max(
    4 * 1024 * 1024,
    2 * _STRUCTURAL_IDENTITY_BYTES,
)
# Compatibility spelling retained for callers of the original CASF-035 CLI.
FEDERATION_CLI_MAX_COMMAND_JSON_BYTES: Final[int] = (
    FEDERATION_CONTROL_MAX_CANONICAL_BYTES
)

FEDERATION_TRANSPORT_INVALID_CODE: Final[str] = "federation_request_invalid"
FEDERATION_TRANSPORT_INVALID_MESSAGE: Final[str] = (
    "federation control request is invalid"
)
FEDERATION_TRANSPORT_UNAVAILABLE_CODE: Final[str] = (
    "federation_control_unavailable"
)
FEDERATION_TRANSPORT_UNAVAILABLE_MESSAGE: Final[str] = (
    "federation control operation is unavailable"
)
FEDERATION_CREATE_DISPATCH_MODE: Final[str] = "authenticated_control_gateway"
FEDERATION_POST_ADMISSION_DISPATCH_MODE: Final[str] = (
    "qualified_control_service"
)
_FEDERATION_CREATE_SUCCESS_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"accepted", "created"}
)


class FederationCLIError(ValueError):
    """A safe, caller-correctable federation CLI input error."""


@dataclass(frozen=True)
class FederationCreateTransport(ClosedContract):
    """Closed request/authentication envelope for authenticated CREATE."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/"
        "create-transport@1"
    )

    request: FederationRequest
    authentication: AuthenticationEvidence

    FIELD_DECODERS: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            "request": lambda value: (
                value
                if isinstance(value, FederationRequest)
                else FederationRequest.from_dict(value)
            ),
            "authentication": lambda value: (
                value
                if isinstance(value, AuthenticationEvidence)
                else AuthenticationEvidence.from_dict(value)
            ),
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.request, FederationRequest):
            raise FederationContractError("create request is not FederationRequest")
        if not isinstance(self.authentication, AuthenticationEvidence):
            raise FederationContractError(
                "create authentication is not AuthenticationEvidence"
            )
        if (
            self.authentication.request_cid != self.request.cid
            or self.authentication.caller_did != self.request.caller_did
            or self.authentication.audience != self.request.audience
            or self.authentication.nonce != self.request.nonce
        ):
            raise FederationContractError(
                "create authentication does not bind the request"
            )


def _command_name(operation: FederationOperation) -> str:
    return operation.value.removeprefix("federation.").replace("_", "-")


FEDERATION_CLI_COMMANDS: Final[Mapping[str, FederationOperation]] = {
    _command_name(operation): operation
    for operation in sorted(FederationOperation, key=lambda item: item.value)
}


def federation_cli_discovery_manifest() -> dict[str, Any]:
    """Return static adapter vocabulary without resolving runtime capability."""

    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/causal-federation/cli-discovery@1",
        "interface": FEDERATION_CLI_INTERFACE,
        "group": "federation",
        "commands": {name: operation.value for name, operation in FEDERATION_CLI_COMMANDS.items()},
        "request_schemas": {
            FederationOperation.CREATE.value: FederationCreateTransport.SCHEMA,
            "post_admission": FederationCommand.SCHEMA,
        },
        "result_schema": FEDERATION_CLI_RESULT_SCHEMA,
        "dispatch": {
            FederationOperation.CREATE.value: FEDERATION_CREATE_DISPATCH_MODE,
            "post_admission": FEDERATION_POST_ADMISSION_DISPATCH_MODE,
        },
        "shell_out": False,
        "embedded_fallback": False,
        "create_via_trigger_gateway": FederationControlGateway.__name__,
        "max_canonical_bytes": FEDERATION_CONTROL_MAX_CANONICAL_BYTES,
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
        argument = (
            "--request-json"
            if operation is FederationOperation.CREATE
            else "--command-json"
        )
        parser.add_argument(
            argument,
            required=True,
            help=(
                "Complete byte-canonical federation transport JSON, supplied "
                f"inline (maximum {FEDERATION_CONTROL_MAX_CANONICAL_BYTES} "
                "UTF-8 bytes)."
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
    if len(raw) > FEDERATION_CONTROL_MAX_CANONICAL_BYTES:
        raise FederationCLIError(
            f"{source} exceeds the canonical transport byte bound"
        )
    try:
        decoded = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise FederationCLIError(f"{source} is not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise FederationCLIError(f"{source} must contain an object")
    return decoded


def _bounded_canonical_bytes(
    payload: Mapping[str, Any], *, source: str
) -> bytes:
    if not isinstance(payload, Mapping):
        raise FederationCLIError(f"{source} must contain an object")
    try:
        encoded = canonical_json_bytes(dict(payload))
    except ControlPlaneContractError as exc:
        raise FederationCLIError(f"{source} is not canonical JSON") from exc
    if len(encoded) > FEDERATION_CONTROL_MAX_CANONICAL_BYTES:
        raise FederationCLIError(
            f"{source} exceeds the canonical transport byte bound"
        )
    return encoded


def decode_federation_control_request(
    payload: Mapping[str, Any],
    operation: FederationOperation | str,
) -> FederationCommand | FederationCreateTransport:
    """Bound and decode a CLI/MCP/Python request before authority resolution."""

    _bounded_canonical_bytes(payload, source="federation request")
    try:
        selected = (
            operation
            if isinstance(operation, FederationOperation)
            else FederationOperation(operation)
        )
    except ValueError as exc:
        raise FederationCLIError("federation operation is outside the catalog") from exc
    if selected is FederationOperation.CREATE:
        return FederationCreateTransport.from_dict(payload)  # type: ignore[return-value]
    if selected not in POST_ADMISSION_OPERATIONS:
        raise FederationCLIError("federation operation is outside the catalog")
    command = FederationCommand.from_dict(payload)
    if command.operation is not selected:
        raise FederationContractError(
            "request operation does not match the selected transport operation"
        )
    return command


def build_federation_command(
    args: argparse.Namespace,
) -> FederationCommand | FederationCreateTransport:
    """Decode one unmodified closed request before authority resolution."""

    selected = getattr(args, "federation_operation", None)
    if not selected:
        raise FederationCLIError("a federation command is required")
    try:
        operation = FederationOperation(str(selected))
    except ValueError as exc:
        raise FederationCLIError("federation operation is outside the closed catalog") from exc
    raw = getattr(
        args,
        "request_json" if operation is FederationOperation.CREATE else "command_json",
        None,
    )
    if not isinstance(raw, str):
        raise FederationCLIError("a complete inline canonical command is required")
    try:
        decoded = decode_federation_control_request(
            _load_json(raw, source="federation request"),
            operation,
        )
        if raw.encode("utf-8") != canonical_json_bytes(decoded.to_dict()):
            raise FederationCLIError("request must use canonical JSON encoding")
    except (ControlPlaneContractError, FederationContractError) as exc:
        raise FederationCLIError(str(exc)) from exc
    return decoded


def federation_control_response_record(
    command: FederationCommand,
    response: FederationControlResponse,
) -> dict[str, Any]:
    """Serialize only canonical command/result/audit evidence for transport."""

    if not isinstance(command, FederationCommand):
        raise FederationControlServiceError("CLI command must be FederationCommand")
    if not isinstance(response, FederationControlResponse):
        raise FederationControlServiceError("control service returned no typed response")
    if type(response.result) is not FederationCommandResult:
        raise FederationControlServiceError(
            "control service result must be an exact FederationCommandResult"
        )
    if type(response.audit) is not FederationControlAuditReceipt:
        raise FederationControlServiceError(
            "control service audit must be an exact FederationControlAuditReceipt"
        )
    # Reconstruct records from their transport forms so a malformed custom
    # embedding cannot emit an apparently canonical result.
    result = FederationCommandResult.from_dict(
        FederationCommandResult.to_dict(response.result)
    )
    audit = FederationControlAuditReceipt.from_dict(
        FederationControlAuditReceipt.to_dict(response.audit)
    )
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


def federation_create_response_record(
    transport: FederationCreateTransport,
    response: tuple[FederationIdentity, FederationReceipt],
) -> dict[str, Any]:
    """Serialize an authenticated CREATE result without authentication material."""

    if not isinstance(transport, FederationCreateTransport):
        raise FederationControlServiceError(
            "CREATE transport must be FederationCreateTransport"
        )
    if not isinstance(response, tuple) or len(response) != 2:
        raise FederationControlServiceError(
            "CREATE gateway returned no typed response"
        )
    identity = FederationIdentity.from_dict(response[0].to_dict())
    receipt = FederationReceipt.from_dict(response[1].to_dict())
    if (
        identity.binding != transport.request.binding
        or receipt.binding != transport.request.binding
        or identity.record_id != f"federation:{transport.request.cid}"
        or receipt.outcome not in _FEDERATION_CREATE_SUCCESS_OUTCOMES
        or not {
            transport.authentication.evidence_id,
            transport.authentication.cid,
        }.intersection(receipt.evidence_refs)
    ):
        raise FederationControlServiceError(
            "CREATE response is not bound to its authenticated request"
        )
    return {
        "schema": FEDERATION_CLI_RESULT_SCHEMA,
        "interface": FEDERATION_CLI_INTERFACE,
        "operation": FederationOperation.CREATE.value,
        "request": transport.request.to_dict(),
        "authentication_evidence_ref": transport.authentication.cid,
        "identity": identity.to_dict(),
        "receipt": receipt.to_dict(),
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
    gateway: FederationControlGateway | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Dispatch one post-admission command through the supplied typed service."""

    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    try:
        request = build_federation_command(args)
    except Exception:
        _write_json(
            stderr,
            {
                "schema": FEDERATION_CLI_ERROR_SCHEMA,
                "interface": FEDERATION_CLI_INTERFACE,
                "status": "invalid_request",
                "code": FEDERATION_TRANSPORT_INVALID_CODE,
                "message": FEDERATION_TRANSPORT_INVALID_MESSAGE,
            },
            compact=True,
        )
        return FEDERATION_CLI_EXIT_INVALID

    try:
        if isinstance(request, FederationCreateTransport):
            if not isinstance(gateway, FederationControlGateway):
                raise FederationControlServiceError(
                    "a qualified FederationControlGateway is required"
                )
            record = federation_create_response_record(
                request,
                gateway.create(request.request, request.authentication),
            )
        else:
            if not isinstance(service, FederationControlService):
                raise FederationControlServiceError(
                    "a qualified FederationControlService is required"
                )
            record = federation_control_response_record(
                request,
                execute_federation_command(service, request),
            )
        _write_json(
            stdout,
            record,
            compact=bool(getattr(args, "output_json", False)),
        )
        return FEDERATION_CLI_EXIT_SUCCESS
    except Exception:
        _write_json(
            stderr,
            {
                "schema": FEDERATION_CLI_ERROR_SCHEMA,
                "interface": FEDERATION_CLI_INTERFACE,
                "status": "unavailable",
                "code": FEDERATION_TRANSPORT_UNAVAILABLE_CODE,
                "message": FEDERATION_TRANSPORT_UNAVAILABLE_MESSAGE,
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
    "FEDERATION_CONTROL_MAX_CANONICAL_BYTES",
    "FEDERATION_CREATE_DISPATCH_MODE",
    "FEDERATION_POST_ADMISSION_DISPATCH_MODE",
    "FEDERATION_TRANSPORT_INVALID_CODE",
    "FEDERATION_TRANSPORT_INVALID_MESSAGE",
    "FEDERATION_TRANSPORT_UNAVAILABLE_CODE",
    "FEDERATION_TRANSPORT_UNAVAILABLE_MESSAGE",
    "FederationCLIError",
    "FederationCreateTransport",
    "build_federation_command",
    "decode_federation_control_request",
    "federation_cli_discovery_manifest",
    "federation_create_response_record",
    "federation_control_response_record",
    "register_federation_cli",
    "run_federation_cli",
]
