"""Thin autonomy CLI/Python adapter over SupervisorControlService.

``AutonomyControlSurface@1`` does not mint authority, call shell strings, or
replace the control catalog.  Reads that already exist in the closed
``Operation`` vocabulary are dispatched through
``SupervisorControlService.execute``.  Additional autonomy snapshot reads are
side-effect-free views of injected records.  Extra mutations require a current
authorization decision and a one-use confirmation nonce.

CLI and MCP adapters decode the same request and call this surface; they never
construct a subprocess.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, TextIO

from ..control.control_cli import run_agent_cli
from ..control.control_contracts import (
    AUTONOMY_CONTROL_INTERFACE,
    AUTONOMY_CONTROL_OPERATION_NAMES,
    AUTONOMY_MUTATION_OPERATION_NAMES,
    AUTONOMY_READ_OPERATION_NAMES,
    AuthorizationDecision,
    AuthorizationVerdict,
    ControlContractError,
    ControlSurface,
    Operation,
    OperationRequest,
    OperationResult,
    OperationStatus,
)
from ..control.control_plane import SupervisorControlService
from ..proof.formal_verification_contracts import content_identity

AUTONOMY_CLI_INTERFACE: Final[str] = AUTONOMY_CONTROL_INTERFACE
AUTONOMY_CONTROL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/control-surface@1"
)
AUTONOMY_CONFIRMATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/control-confirmation@1"
)
_SERVICE_OPERATIONS: Final[Mapping[str, Operation]] = MappingProxyType(
    {
        "capabilities": Operation.CAPABILITIES,
        "status": Operation.STATUS,
        "metrics": Operation.METRICS,
        "pause": Operation.PAUSE,
        "resume": Operation.RESUME,
        "cancel": Operation.CANCEL,
    }
)
_BOUNDED_LEVELS: Final[frozenset[str]] = frozenset(
    {
        "observe_only",
        "recommend",
        "dry_run",
        "execute_reversible",
        "execute_bounded_mutation",
        "self_repair_isolated",
    }
)
_SHELL_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "bash",
        "cmd.exe",
        "os.system",
        "powershell",
        "shell_command",
        "subprocess",
        "/bin/sh",
        "/bin/bash",
    }
)


class AutonomyControlError(ControlContractError):
    """Raised when the autonomy adapter is asked to exceed its authority."""


@dataclass
class AutonomyControlSurface:
    """Canonical Python methods for the APMC-017 autonomy control surface."""

    service: SupervisorControlService
    snapshots: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    _consumed_confirmations: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not isinstance(self.service, SupervisorControlService):
            raise TypeError("service must be a SupervisorControlService")
        frozen: dict[str, Mapping[str, Any]] = {}
        for name, payload in dict(self.snapshots).items():
            if name not in AUTONOMY_READ_OPERATION_NAMES:
                raise AutonomyControlError(f"unknown autonomy snapshot {name}")
            frozen[name] = MappingProxyType(dict(payload))
        self.snapshots = MappingProxyType(frozen)

    @property
    def interface(self) -> str:
        return AUTONOMY_CONTROL_INTERFACE

    @property
    def operations(self) -> frozenset[str]:
        return AUTONOMY_CONTROL_OPERATION_NAMES

    def discover(self) -> Mapping[str, Any]:
        """Side-effect-free catalog listing.  Never starts a provider."""

        return MappingProxyType(
            {
                "schema": AUTONOMY_CONTROL_SCHEMA,
                "interface": AUTONOMY_CONTROL_INTERFACE,
                "surface": ControlSurface.PYTHON.value,
                "reads": tuple(sorted(AUTONOMY_READ_OPERATION_NAMES)),
                "mutations": tuple(sorted(AUTONOMY_MUTATION_OPERATION_NAMES)),
                "shell_out": False,
                "mints_permission": False,
            }
        )

    def execute(
        self,
        operation: str,
        request: OperationRequest | Mapping[str, Any] | None = None,
        *,
        confirmation_id: str = "",
        level: str = "",
        authorization: AuthorizationDecision | None = None,
    ) -> OperationResult | Mapping[str, Any]:
        name = str(operation or "").strip()
        if name not in AUTONOMY_CONTROL_OPERATION_NAMES:
            raise AutonomyControlError(f"unknown autonomy operation {name}")
        if any(marker in name for marker in _SHELL_MARKERS):
            raise AutonomyControlError("autonomy adapters cannot invoke a shell")
        if name in AUTONOMY_READ_OPERATION_NAMES:
            return self._read(name, request)
        return self._mutate(
            name,
            request,
            confirmation_id=confirmation_id,
            level=level,
            authorization=authorization,
        )

    def _read(
        self,
        name: str,
        request: OperationRequest | Mapping[str, Any] | None,
    ) -> OperationResult | Mapping[str, Any]:
        mapped = _SERVICE_OPERATIONS.get(name)
        if mapped is not None:
            if not isinstance(request, OperationRequest):
                raise AutonomyControlError(
                    f"{name} requires a canonical OperationRequest"
                )
            if request.operation is not mapped:
                raise AutonomyControlError(
                    "request operation does not match the autonomy adapter"
                )
            return self.service.execute(request)
        return MappingProxyType(
            {
                "schema": AUTONOMY_CONTROL_SCHEMA,
                "operation": name,
                "status": OperationStatus.SUCCEEDED.value,
                "mutated": False,
                "provider_started": False,
                "snapshot": dict(self.snapshots.get(name, {})),
            }
        )

    def _mutate(
        self,
        name: str,
        request: OperationRequest | Mapping[str, Any] | None,
        *,
        confirmation_id: str,
        level: str,
        authorization: AuthorizationDecision | None,
    ) -> OperationResult | Mapping[str, Any]:
        mapped = _SERVICE_OPERATIONS.get(name)
        if mapped is not None:
            if not isinstance(request, OperationRequest):
                raise AutonomyControlError(
                    f"{name} requires a canonical OperationRequest"
                )
            if request.operation is not mapped:
                raise AutonomyControlError(
                    "request operation does not match the autonomy adapter"
                )
            if request.authorization is None and not request.dry_run:
                raise AutonomyControlError(
                    "autonomy adapters cannot mint mutation authorization"
                )
            return self.service.execute(request)
        token = str(confirmation_id or "").strip()
        if not token:
            raise AutonomyControlError("confirmation_id is required")
        if token in self._consumed_confirmations:
            raise AutonomyControlError("confirmation replay is forbidden")
        if authorization is None or authorization.verdict is not AuthorizationVerdict.PERMIT:
            raise AutonomyControlError(
                "autonomy adapters cannot mint mutation authorization"
            )
        if name == "set_level":
            selected = str(level or "").strip()
            if selected not in _BOUNDED_LEVELS:
                raise AutonomyControlError("set_level requires a bounded autonomy level")
        self._consumed_confirmations.add(token)
        receipt = {
            "schema": AUTONOMY_CONFIRMATION_SCHEMA,
            "operation": name,
            "confirmation_id": token,
            "authorization_id": authorization.content_id,
            "authorizes_merge": False,
            "mints_permission": False,
        }
        receipt["receipt_id"] = content_identity(receipt)
        return MappingProxyType(
            {
                "schema": AUTONOMY_CONTROL_SCHEMA,
                "operation": name,
                "status": OperationStatus.SUCCEEDED.value,
                "receipt": MappingProxyType(receipt),
            }
        )


def run_autonomy_cli(
    args: argparse.Namespace,
    *,
    service: SupervisorControlService | None = None,
    surface: AutonomyControlSurface | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdin: TextIO | None = None,
) -> int:
    """CLI adapter: decode and dispatch without constructing a shell."""

    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    operation = str(getattr(args, "autonomy_operation", "") or "").strip()
    if operation in _SERVICE_OPERATIONS or not operation:
        return run_agent_cli(
            args,
            service=service if surface is None else surface.service,
            stdout=stdout,
            stderr=stderr,
            stdin=stdin,
        )
    if surface is None:
        raise AutonomyControlError("autonomy snapshot operations require a bound surface")
    try:
        result = surface.execute(
            operation,
            confirmation_id=str(getattr(args, "confirmation_id", "") or ""),
            level=str(getattr(args, "level", "") or ""),
            authorization=getattr(args, "authorization", None),
        )
    except (AutonomyControlError, ControlContractError) as exc:
        stderr.write(f"{exc}\n")
        return 2
    payload = result.to_record() if hasattr(result, "to_record") else dict(result)
    if isinstance(payload.get("receipt"), Mapping):
        payload["receipt"] = dict(payload["receipt"])
    stdout.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    return 0


def build_autonomy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ipfs-accelerate autonomy")
    parser.add_argument("autonomy_operation", choices=sorted(AUTONOMY_CONTROL_OPERATION_NAMES))
    parser.add_argument("--confirmation-id", dest="confirmation_id", default="")
    parser.add_argument("--level", dest="level", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_autonomy_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_autonomy_cli(args)


__all__ = (
    "AUTONOMY_CLI_INTERFACE",
    "AUTONOMY_CONFIRMATION_SCHEMA",
    "AUTONOMY_CONTROL_INTERFACE",
    "AUTONOMY_CONTROL_OPERATION_NAMES",
    "AUTONOMY_CONTROL_SCHEMA",
    "AutonomyControlError",
    "AutonomyControlSurface",
    "build_autonomy_parser",
    "main",
    "run_autonomy_cli",
)
