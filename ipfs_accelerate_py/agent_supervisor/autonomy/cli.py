"""Thin ``ipfs-accelerate agent autonomy`` adapter.

The adapter owns no autonomy policy.  It decodes caller arguments into the
canonical autonomy-control vocabulary and invokes
:class:`~ipfs_accelerate_py.agent_supervisor.control.control_plane.AutonomyControl`
directly.  It never shells out, never mints authority or confirmation, and
never starts a provider during discovery or help.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from typing import Any, TextIO

from ..control.control_contracts import (
    AUTONOMY_CONTROL_MUTATION_OPERATIONS,
    AutonomyControlOperation,
    SUPERVISOR_AUTONOMY_READ_AUTHORITY,
    discover_autonomy_control_catalog,
    autonomy_control_operations,
)
from ..control.control_plane import (
    AutonomyControl,
    SupervisorControlService,
)


AUTONOMY_CLI_EXIT_SUCCESS = 0
AUTONOMY_CLI_EXIT_FAILED = 1
AUTONOMY_CLI_EXIT_INVALID = 2

AUTONOMY_CLI_COMMANDS: dict[str, str] = {
    "discover": "discover",
    "capabilities": AutonomyControlOperation.CAPABILITIES.value,
    "status": AutonomyControlOperation.STATUS.value,
    "metrics": AutonomyControlOperation.METRICS.value,
    "graph": AutonomyControlOperation.GRAPH.value,
    "unresolved-questions": AutonomyControlOperation.UNRESOLVED_QUESTIONS.value,
    "budget": AutonomyControlOperation.BUDGET.value,
    "experience-summary": AutonomyControlOperation.EXPERIENCE_SUMMARY.value,
    "route-policy": AutonomyControlOperation.ROUTE_POLICY.value,
    "distillation-candidates": AutonomyControlOperation.DISTILLATION_CANDIDATES.value,
    "repair-history": AutonomyControlOperation.REPAIR_HISTORY.value,
    "escalations": AutonomyControlOperation.ESCALATIONS.value,
    "shadow-results": AutonomyControlOperation.SHADOW_RESULTS.value,
    "pause": AutonomyControlOperation.PAUSE.value,
    "resume": AutonomyControlOperation.RESUME.value,
    "set-level": AutonomyControlOperation.SET_LEVEL.value,
    "approve-policy-candidate": AutonomyControlOperation.APPROVE_POLICY_CANDIDATE.value,
    "reject-policy-candidate": AutonomyControlOperation.REJECT_POLICY_CANDIDATE.value,
    "rollback-policy-candidate": AutonomyControlOperation.ROLLBACK_POLICY_CANDIDATE.value,
    "approve-repair": AutonomyControlOperation.APPROVE_REPAIR.value,
    "cancel-action": AutonomyControlOperation.CANCEL_ACTION.value,
    "bind-escalation-answer": AutonomyControlOperation.BIND_ESCALATION_ANSWER.value,
}


class AutonomyCLIError(ValueError):
    """Raised when CLI arguments cannot be decoded into a typed request."""


def _write_record(stream: TextIO, record: Mapping[str, Any], *, compact: bool) -> None:
    if compact:
        encoded = json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    else:
        encoded = json.dumps(
            record, sort_keys=True, indent=2, ensure_ascii=False
        )
    stream.write(encoded + "\n")


def _authorities_from_args(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "authorities_json", None)
    if raw is None or raw == "":
        return [SUPERVISOR_AUTONOMY_READ_AUTHORITY]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AutonomyCLIError("authorities JSON is invalid") from exc
    if not isinstance(payload, list) or not all(
        isinstance(item, str) and item for item in payload
    ):
        raise AutonomyCLIError("authorities JSON must be an array of strings")
    return list(payload)


def _json_object(raw: str | None, *, noun: str) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AutonomyCLIError(f"{noun} JSON is invalid") from exc
    if not isinstance(payload, Mapping):
        raise AutonomyCLIError(f"{noun} JSON must be an object")
    return dict(payload)


def _json_list(raw: str | None, *, noun: str) -> list[Any]:
    if raw is None or raw == "":
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AutonomyCLIError(f"{noun} JSON is invalid") from exc
    if not isinstance(payload, list):
        raise AutonomyCLIError(f"{noun} JSON must be an array")
    return list(payload)


def autonomy_cli_discovery_manifest() -> dict[str, Any]:
    """Static discovery for autonomy CLI commands.  No service is constructed."""

    catalog = dict(discover_autonomy_control_catalog())
    catalog["surface"] = "cli"
    catalog["command"] = "ipfs-accelerate agent autonomy"
    catalog["commands"] = dict(AUTONOMY_CLI_COMMANDS)
    catalog["operations"] = list(autonomy_control_operations())
    catalog["dispatch_mode"] = "direct_service"
    catalog["shells_out"] = False
    catalog["mints_permission"] = False
    return catalog


def register_autonomy_cli(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register the nested ``autonomy`` group on an ``agent`` parser."""

    autonomy_cli_discovery_manifest()
    autonomy = subparsers.add_parser(
        "autonomy",
        help="Inspect and control bounded supervisor autonomy.",
        description=(
            "Thin typed adapter over AutonomyControl.  Reads are side-effect "
            "free.  Mutations require distinct authority, lease, fence, "
            "expected effects, and (when catalogued) one-use confirmation."
        ),
    )
    commands = autonomy.add_subparsers(
        dest="autonomy_command",
        metavar="COMMAND",
        help="Autonomy control operation.",
    )
    for command, operation_name in AUTONOMY_CLI_COMMANDS.items():
        child = commands.add_parser(
            command,
            help=f"Run the {operation_name} autonomy operation.",
        )
        child.set_defaults(
            autonomy_operation=operation_name,
            autonomy_cli_command=command,
        )
        child.add_argument("--authorities-json", help="JSON array of autonomy authorities.")
        child.add_argument("--target-id", help="Exact autonomy target identifier.")
        child.add_argument("--limit", type=int, default=50, help="Bounded page size.")
        child.add_argument("--cursor", help="Opaque pagination cursor.")
        child.add_argument("--parameters-json", help="Additional JSON parameters.")
        child.add_argument("--expected-revision", help="Expected autonomy revision.")
        child.add_argument("--idempotency-key", help="Scoped idempotency key.")
        child.add_argument("--lease-id", help="Current authoritative lease.")
        child.add_argument("--fence", type=int, help="Current fencing epoch.")
        child.add_argument(
            "--expected-effects-json",
            help="JSON array of expected effect identifiers.",
        )
        child.add_argument(
            "--confirmation-cid",
            help="One-use externally issued confirmation identity.",
        )
        child.add_argument("--level", help="Closed autonomy level for set-level.")
        child.add_argument("--candidate-id", help="Policy candidate identity.")
        child.add_argument("--repair-id", help="Repair plan identity.")
        child.add_argument("--action-id", help="In-flight action identity.")
        child.add_argument("--packet-id", help="Escalation packet identity.")
        child.add_argument("--option", help="Escalation answer option.")
        child.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview a mutation without applying or consuming confirmation.",
        )
        child.add_argument(
            "--output-json",
            action="store_true",
            help="Emit the canonical JSON envelope.",
        )
    return autonomy


def build_autonomy_parser() -> argparse.ArgumentParser:
    """Standalone parser for tests and nested CLI embedding."""

    parser = argparse.ArgumentParser(
        prog="ipfs-accelerate agent autonomy",
        description="Thin typed autonomy control adapter.",
    )
    subparsers = parser.add_subparsers(dest="autonomy_command")
    register_autonomy_cli(subparsers)
    return parser


def run_autonomy_cli(
    args: argparse.Namespace,
    *,
    autonomy_control: AutonomyControl | None = None,
    service: SupervisorControlService | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Thin CLI adapter for AutonomyControl (schema/result/error parity)."""

    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    try:
        nested = getattr(args, "autonomy_command", None)
        if nested == "autonomy" and hasattr(args, "autonomy_operation"):
            operation = str(getattr(args, "autonomy_operation", "") or "")
        else:
            operation = str(
                getattr(args, "autonomy_operation", None)
                or AUTONOMY_CLI_COMMANDS.get(str(nested or ""), "")
            )
        if not operation:
            raise AutonomyCLIError("autonomy operation is required")
        controller = autonomy_control
        if controller is None and service is not None:
            controller = service.autonomy_control
        if controller is None:
            controller = AutonomyControl()
        authorities = _authorities_from_args(args)
        if operation == "discover":
            record = dict(controller.discover())
            record["success"] = True
            record["status"] = "success"
        else:
            parameters = _json_object(
                getattr(args, "parameters_json", None), noun="parameters"
            )
            target_id = getattr(args, "target_id", None)
            if target_id:
                parameters.setdefault("target_id", target_id)
            if getattr(args, "limit", None) is not None:
                parameters.setdefault("limit", int(args.limit))
            if getattr(args, "cursor", None):
                parameters.setdefault("cursor", args.cursor)
            if getattr(args, "expected_revision", None):
                parameters.setdefault("expected_revision", args.expected_revision)
            if getattr(args, "idempotency_key", None):
                parameters.setdefault("idempotency_key", args.idempotency_key)
            if getattr(args, "lease_id", None):
                parameters.setdefault("lease_id", args.lease_id)
            if getattr(args, "fence", None) is not None:
                parameters.setdefault("fence", int(args.fence))
            effects = _json_list(
                getattr(args, "expected_effects_json", None),
                noun="expected effects",
            )
            if effects:
                parameters.setdefault("expected_effects", effects)
            if getattr(args, "confirmation_cid", None):
                parameters.setdefault("confirmation_cid", args.confirmation_cid)
            if getattr(args, "level", None):
                parameters.setdefault("level", args.level)
            if getattr(args, "candidate_id", None):
                parameters.setdefault("candidate_id", args.candidate_id)
            if getattr(args, "repair_id", None):
                parameters.setdefault("repair_id", args.repair_id)
            if getattr(args, "action_id", None):
                parameters.setdefault("action_id", args.action_id)
            if getattr(args, "packet_id", None):
                parameters.setdefault("packet_id", args.packet_id)
            if getattr(args, "option", None):
                parameters.setdefault("option", args.option)
            if getattr(args, "dry_run", False):
                parameters.setdefault("dry_run", True)
            mutating = any(
                item.value == operation for item in AUTONOMY_CONTROL_MUTATION_OPERATIONS
            )
            if mutating and authorities == [SUPERVISOR_AUTONOMY_READ_AUTHORITY]:
                # Adapter default remains read-only.  Callers must supply the
                # distinct mutation authority; the CLI never mints it.
                pass
            record = controller.execute(
                operation, authorities=authorities, **parameters
            )
        _write_record(
            stdout,
            record,
            compact=bool(getattr(args, "output_json", False)),
        )
        if record.get("success") is False:
            return AUTONOMY_CLI_EXIT_FAILED
        return AUTONOMY_CLI_EXIT_SUCCESS
    except AutonomyCLIError as exc:
        payload = {
            "schema": "ipfs_accelerate_py/agent-supervisor/cli-error@1",
            "status": "invalid_request",
            "error": {"code": "invalid_request", "detail": str(exc)},
            "error_code": "invalid_request",
            "success": False,
        }
        _write_record(stderr, payload, compact=True)
        return AUTONOMY_CLI_EXIT_INVALID
    except Exception:
        payload = {
            "schema": "ipfs_accelerate_py/agent-supervisor/cli-error@1",
            "status": "internal_error",
            "error": {
                "code": "internal_error",
                "detail": "autonomy control operation failed",
            },
            "error_code": "internal_error",
            "success": False,
        }
        _write_record(stderr, payload, compact=True)
        return AUTONOMY_CLI_EXIT_FAILED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ipfs-accelerate-agent-autonomy")
    subparsers = parser.add_subparsers(dest="autonomy_command")
    for command, operation_name in AUTONOMY_CLI_COMMANDS.items():
        child = subparsers.add_parser(command)
        child.set_defaults(autonomy_operation=operation_name)
        child.add_argument("--authorities-json")
        child.add_argument("--target-id")
        child.add_argument("--limit", type=int, default=50)
        child.add_argument("--cursor")
        child.add_argument("--parameters-json")
        child.add_argument("--expected-revision")
        child.add_argument("--idempotency-key")
        child.add_argument("--lease-id")
        child.add_argument("--fence", type=int)
        child.add_argument("--expected-effects-json")
        child.add_argument("--confirmation-cid")
        child.add_argument("--level")
        child.add_argument("--candidate-id")
        child.add_argument("--repair-id")
        child.add_argument("--action-id")
        child.add_argument("--packet-id")
        child.add_argument("--option")
        child.add_argument("--dry-run", action="store_true")
        child.add_argument("--output-json", action="store_true")
    args = parser.parse_args(argv)
    return run_autonomy_cli(args)


__all__ = [
    "AUTONOMY_CLI_COMMANDS",
    "AUTONOMY_CLI_EXIT_FAILED",
    "AUTONOMY_CLI_EXIT_INVALID",
    "AUTONOMY_CLI_EXIT_SUCCESS",
    "AutonomyCLIError",
    "autonomy_cli_discovery_manifest",
    "build_autonomy_parser",
    "main",
    "register_autonomy_cli",
    "run_autonomy_cli",
]


if __name__ == "__main__":
    raise SystemExit(main())
