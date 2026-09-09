"""Thin SPAR-044 CLI and MCP adapters.

Adapters in this module own decoding only.  The canonical
``SemanticRefactoringService.execute`` method owns every policy decision,
and MCP never converts a request into a shell command or CLI-text round trip.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Final, TextIO

from .service import (
    ALL_OPERATIONS,
    MUTATION_OPERATIONS,
    READ_OPERATIONS,
    ControlRequest,
    ControlResult,
    ControlSurfaceError,
    SemanticRefactoringService,
    register_operations,
)


MCP_NEVER_SHELLS: Final[bool] = True
CLI_PROG: Final[str] = "ipfs-accelerate agent spar"

_CLI_NAMES: Final[dict[str, str]] = {
    item.rsplit(".", 1)[1].replace("_", "-"): item for item in ALL_OPERATIONS
}


class SemanticRefactoringControlBackend:
    """A thin direct adapter over the canonical service, with no fallback."""

    def __init__(self, service: SemanticRefactoringService | None = None) -> None:
        if service is None:
            service = SemanticRefactoringService()
        if not isinstance(service, SemanticRefactoringService):
            raise TypeError("service must be a SemanticRefactoringService")
        self._service = service

    @property
    def service(self) -> SemanticRefactoringService:
        return self._service

    def execute(self, request: ControlRequest) -> ControlResult:
        if not isinstance(request, ControlRequest):
            raise TypeError("request must be a ControlRequest")
        return self._service.execute(request)


def mcp_dispatch(
    backend: SemanticRefactoringControlBackend | SemanticRefactoringService,
    request: ControlRequest | Mapping[str, Any],
) -> ControlResult:
    """Direct service dispatch; this function never executes a shell."""

    if isinstance(backend, SemanticRefactoringService):
        backend = SemanticRefactoringControlBackend(backend)
    if not isinstance(request, ControlRequest):
        request = ControlRequest.from_dict(request)
    return backend.execute(request)


def register_spar_cli(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register the ``agent spar`` group on an existing agent parser."""

    spar = subparsers.add_parser("spar", help="Typed SPAR control and diagnostics.")
    commands = spar.add_subparsers(dest="spar_command", required=True)
    for command, operation in _CLI_NAMES.items():
        child = commands.add_parser(command, help=f"Run {operation}.")
        child.set_defaults(spar_operation=operation)
        child.add_argument(
            "--request-json",
            required=True,
            help="Complete typed ControlRequest JSON.",
        )
        child.add_argument(
            "--output-json",
            action="store_true",
            help="Emit canonical result JSON.",
        )
    return spar


def build_spar_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ipfs-accelerate")
    root = parser.add_subparsers(dest="root_command", required=True)
    agent = root.add_parser("agent")
    register_spar_cli(agent.add_subparsers(dest="agent_command", required=True))
    return parser


class SemanticRefactoringCLI:
    """Typed command adapter with the same operation vocabulary as MCP."""

    def __init__(
        self,
        service: SemanticRefactoringService | SemanticRefactoringControlBackend | None = None,
    ) -> None:
        if isinstance(service, SemanticRefactoringControlBackend):
            self.backend = service
        else:
            self.backend = SemanticRefactoringControlBackend(service)

    @property
    def service(self) -> SemanticRefactoringService:
        return self.backend.service

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog=CLI_PROG)
        parser.add_argument(
            "operation",
            choices=list(_CLI_NAMES),
        )
        parser.add_argument("--request-json", help="Complete ControlRequest JSON object.")
        parser.add_argument("--target-id", default="")
        parser.add_argument("--parameters-json", default="{}")
        parser.add_argument("--authorization-json", help="Authorization JSON object.")
        parser.add_argument("--budget-json", help="Budget JSON object.")
        parser.add_argument("--idempotency-key", default="")
        parser.add_argument("--lease-id", default="")
        parser.add_argument("--fencing-epoch", type=int)
        parser.add_argument("--tree-id", default="")
        parser.add_argument("--dry-run", action="store_true")
        return parser

    @staticmethod
    def _object(value: str, label: str) -> Mapping[str, Any]:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ControlSurfaceError(f"{label} must be valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise ControlSurfaceError(f"{label} must be a JSON object")
        return decoded

    def request_from_namespace(self, namespace: argparse.Namespace) -> ControlRequest:
        operation = _CLI_NAMES[namespace.operation]
        if namespace.request_json:
            payload = dict(self._object(namespace.request_json, "request-json"))
            supplied = ControlRequest.from_dict(payload)
            if supplied.operation != operation:
                raise ControlSurfaceError(
                    "positional operation and request-json operation differ"
                )
            return supplied
        fields: dict[str, Any] = {
            "operation": operation,
            "target_id": namespace.target_id,
            "parameters": self._object(namespace.parameters_json, "parameters-json"),
            "dry_run": namespace.dry_run,
            "tree_id": namespace.tree_id,
        }
        if operation in MUTATION_OPERATIONS:
            fields["idempotency_key"] = namespace.idempotency_key
            fields["lease_id"] = namespace.lease_id
            fields["fencing_epoch"] = namespace.fencing_epoch
            if namespace.authorization_json:
                fields["authorization"] = self._object(
                    namespace.authorization_json, "authorization-json"
                )
            if namespace.budget_json:
                fields["budget"] = self._object(namespace.budget_json, "budget-json")
        return ControlRequest.from_dict(fields)

    def run(
        self,
        argv: Sequence[str] | None = None,
        *,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> int:
        stdout, stderr = stdout or sys.stdout, stderr or sys.stderr
        try:
            parsed = self.parser().parse_args(list(argv or ()))
            result = self.backend.execute(self.request_from_namespace(parsed))
        except (ControlSurfaceError, TypeError, ValueError) as exc:
            print(
                json.dumps(
                    {"ok": False, "status": "invalid", "error": str(exc)},
                    sort_keys=True,
                ),
                file=stdout,
            )
            return 2
        print(
            json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")),
            file=stdout,
        )
        return 0 if result.ok else 1


def run_spar_cli(
    argv: Sequence[str],
    backend: SemanticRefactoringControlBackend | SemanticRefactoringService,
    *,
    stdout: TextIO | None = None,
) -> int:
    """Run one CLI request through precisely the MCP/Python service adapter."""

    if isinstance(backend, SemanticRefactoringService):
        backend = SemanticRefactoringControlBackend(backend)
    args = build_spar_cli_parser().parse_args(list(argv))
    try:
        request = ControlRequest.from_dict(json.loads(args.request_json))
        if request.operation != args.spar_operation:
            raise ControlSurfaceError("CLI command and request operation differ")
        result = backend.execute(request)
    except (json.JSONDecodeError, ControlSurfaceError, TypeError, ValueError) as exc:
        if stdout is not None:
            stdout.write(
                json.dumps(
                    {"ok": False, "status": "invalid", "error": str(exc)},
                    sort_keys=True,
                )
                + "\n"
            )
        return 2
    if stdout is not None:
        stdout.write(
            json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        )
    return 0 if result.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    return SemanticRefactoringCLI().run(argv)


__all__ = [
    "ALL_OPERATIONS",
    "CLI_PROG",
    "MCP_NEVER_SHELLS",
    "MUTATION_OPERATIONS",
    "READ_OPERATIONS",
    "SemanticRefactoringCLI",
    "SemanticRefactoringControlBackend",
    "build_spar_cli_parser",
    "main",
    "mcp_dispatch",
    "register_operations",
    "register_spar_cli",
    "run_spar_cli",
]
