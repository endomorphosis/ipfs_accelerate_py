"""Proof-context CLI parser and invocation context (PCCE-040).

This module is a presentation and call layer over the stable runtime. It does
not implement lifecycle stages, admit evidence, or approve patches. Importing
it performs no I/O, network, process, or filesystem mutation and does not
bind a model provider, search sibling checkouts, or infer the current
directory as a repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, TextIO

from ipfs_accelerate_py.proof_context.policy import MODES, admit_mode

SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1"
CLI_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/cli"
CLI_RESULT_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/cli-result"
CONTRACT_VERSION: Final[str] = "0.1"
PROG: Final[str] = "proof-context"
INTERFACE: Final[str] = "ProofContextCLI@0.1"
STATE_COMMANDS: Final[tuple[str, ...]] = ("init", "scan", "status", "plan")
OUTPUT_MODES: Final[tuple[str, ...]] = ("json", "human")
DEFAULT_POLICY: Final[str] = "production"
DEFAULT_OUTPUT_MODE: Final[str] = "json"
PROVIDER_BOUND: Final[bool] = False
SIBLING_LAYOUT_REQUIRED: Final[bool] = False
INFERS_CURRENT_DIRECTORY: Final[bool] = False

REQUIRED_ARGUMENTS: Final[tuple[str, ...]] = (
    "repository",
    "policy",
    "task",
    "correlation",
    "output-mode",
)

USAGE_COMMAND: Final[str] = (
    "command is required; expected one of: init, scan, status, plan"
)
USAGE_REPOSITORY: Final[str] = (
    "repository is required; the current directory is not inferred"
)
USAGE_TASK: Final[str] = "task is required"
USAGE_CORRELATION: Final[str] = "correlation is required"
USAGE_POLICY: Final[str] = (
    "policy must be one of: production, supervised, evaluation, simulation"
)
USAGE_OUTPUT_MODE: Final[str] = "output-mode must be one of: json, human"

EXIT_SUCCEEDED: Final[int] = 0
EXIT_TYPED_FAILURE: Final[int] = 1
EXIT_USAGE: Final[int] = 2
EXIT_REJECTED: Final[int] = 3
EXIT_SIMULATED: Final[int] = 4
EXIT_UNAVAILABLE: Final[int] = 5
EXIT_STALE: Final[int] = 6

EXIT_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "succeeded": EXIT_SUCCEEDED,
        "invalid": EXIT_USAGE,
        "rejected": EXIT_REJECTED,
        "simulated": EXIT_SIMULATED,
        "unavailable": EXIT_UNAVAILABLE,
        "stale": EXIT_STALE,
        "timeout": EXIT_TYPED_FAILURE,
        "cancelled": EXIT_TYPED_FAILURE,
        "verification_failed": EXIT_TYPED_FAILURE,
        "proof_failed": EXIT_TYPED_FAILURE,
        "assurance_failed": EXIT_TYPED_FAILURE,
        "context_insufficient": EXIT_TYPED_FAILURE,
        "model_escalation_required": EXIT_TYPED_FAILURE,
        "human_review_required": EXIT_TYPED_FAILURE,
        "infrastructure_failure": EXIT_TYPED_FAILURE,
        "partial_effect": EXIT_TYPED_FAILURE,
        "repair_required": EXIT_TYPED_FAILURE,
    }
)

_COMMAND_HELP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "init": "Initialize an ordinary Python Git repository through the runtime helper.",
        "scan": "Scan and persist semantic repository state through the runtime.",
        "status": "Show typed runtime status for an explicit repository and task.",
        "plan": "Produce a proof-aware invalidation plan through the runtime.",
    }
)
_VALUE_OPTIONS: Final[frozenset[str]] = frozenset(
    {
        "--repository",
        "--policy",
        "--task",
        "--correlation",
        "--output-mode",
        "--run-id",
        "--repository-id",
        "--state-dir",
    }
)


class CliUsageError(ValueError):
    """Stable argument or help-adjacent usage error. Never a typed success."""

    code = "malformed"
    status = "invalid"


@dataclass(frozen=True)
class CliContext:
    """Explicit invocation context. None of these fields default to cwd."""

    command: str
    repository: Path
    policy: str
    task_id: str
    correlation_id: str
    output_mode: str
    run_id: str | None = None
    repository_id: str | None = None
    state_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.command not in STATE_COMMANDS:
            raise CliUsageError(USAGE_COMMAND)
        if self.output_mode not in OUTPUT_MODES:
            raise CliUsageError(USAGE_OUTPUT_MODE)
        admit_mode(self.policy)


@dataclass(frozen=True)
class CliResult:
    """Typed CLI result. Generic success dictionaries are not representable."""

    command: str
    status: str
    policy: str
    correlation_id: str
    output_mode: str
    payload: Mapping[str, Any]
    identities: Mapping[str, Any] | None = None
    artifact_cid: str | None = None
    provenance: str | None = None
    error: str | None = None
    contract: str | None = None
    schema: str = CLI_RESULT_SCHEMA

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "contract_version": CONTRACT_VERSION,
                "interface": INTERFACE,
                "command": self.command,
                "status": self.status,
                "exit_code": exit_code_for(self.status, provenance=self.provenance),
                "policy": self.policy,
                "correlation_id": self.correlation_id,
                "output_mode": self.output_mode,
                "identities": dict(self.identities or {}),
                "artifact_cid": self.artifact_cid,
                "provenance": self.provenance,
                "contract": self.contract,
                "error": self.error,
                "payload": _jsonable(self.payload),
                "provider_bound": PROVIDER_BOUND,
                "sibling_layout_required": SIBLING_LAYOUT_REQUIRED,
            }
        )


class _StableParser(argparse.ArgumentParser):
    """ArgumentParser that raises typed usage errors instead of exiting."""

    def error(self, message: str) -> None:
        raise CliUsageError(_stabilize_usage(message))


def _stabilize_usage(message: str) -> str:
    text = " ".join(str(message).strip().split())
    lowered = text.lower()
    if "command" in lowered and "required" in lowered:
        return USAGE_COMMAND
    if "repository" in lowered and "required" in lowered:
        return USAGE_REPOSITORY
    if "invalid choice" in lowered and "policy" in lowered:
        return USAGE_POLICY
    if "invalid choice" in lowered and "output-mode" in lowered:
        return USAGE_OUTPUT_MODE
    if "invalid choice" in lowered and "command" in lowered:
        return USAGE_COMMAND
    return text or USAGE_COMMAND


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


def exit_code_for(status: str, *, provenance: str | None = None) -> int:
    """Map a closed status onto the PCCE-040 exit-code subset.

    Simulated, unavailable, and failed results never collapse to exit zero.
    """

    if status == "succeeded" and provenance not in {None, "live"}:
        if provenance == "simulated":
            return EXIT_SIMULATED
        return EXIT_TYPED_FAILURE
    return int(EXIT_CODES.get(status, EXIT_TYPED_FAILURE))


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repository",
        dest="repository",
        default=None,
        help="explicit repository path (required; cwd is not inferred)",
    )
    parser.add_argument(
        "--policy",
        dest="policy",
        default=DEFAULT_POLICY,
        choices=tuple(MODES),
        help="runtime policy/mode admitted by policy.admit_mode (default: production)",
    )
    parser.add_argument(
        "--task",
        dest="task",
        default=None,
        help="explicit task identity (required)",
    )
    parser.add_argument(
        "--correlation",
        dest="correlation",
        default=None,
        help="explicit correlation/trace identity (required)",
    )
    parser.add_argument(
        "--output-mode",
        dest="output_mode",
        default=DEFAULT_OUTPUT_MODE,
        choices=OUTPUT_MODES,
        help="json or human output (default: json)",
    )
    parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="optional run identity; defaults to a correlation-bound runtime id",
    )
    parser.add_argument(
        "--repository-id",
        dest="repository_id",
        default=None,
        help="optional repository identity; defaults to the repository name",
    )
    parser.add_argument(
        "--state-dir",
        dest="state_dir",
        default=None,
        help="optional kit/state directory inside the selected repository tree",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the state-command parser. Does not open a runtime or repository."""

    parser = _StableParser(
        prog=PROG,
        description=(
            "Proof-carrying context engine CLI. Commands are thin calls into "
            "the stable runtime. init creates an ordinary Python Git repository; "
            "scan persists semantic state; status emits typed status; plan emits "
            "a proof-aware invalidation plan. Repository, policy, task, "
            "correlation, and output-mode are explicit; the current directory "
            "is never inferred."
        ),
        add_help=True,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes (subset): 0 succeeded live; 1 typed failure; "
            "2 usage/invalid; 3 rejected; 4 simulated; 5 unavailable; 6 stale."
        ),
    )
    _add_common_arguments(parser)
    subparsers = parser.add_subparsers(
        dest="command",
        required=False,
        metavar="command",
        help="state command: init, scan, status, or plan",
    )
    for name in STATE_COMMANDS:
        subparsers.add_parser(
            name,
            help=_COMMAND_HELP[name],
            description=_COMMAND_HELP[name],
            allow_abbrev=False,
        )
    return parser


def _split_command(argv: Sequence[str]) -> tuple[str | None, list[str]]:
    """Accept options before or after the command without inferring cwd."""

    command: str | None = None
    remaining: list[str] = []
    index = 0
    values = list(argv)
    while index < len(values):
        item = values[index]
        if item in STATE_COMMANDS and command is None:
            command = item
            index += 1
            continue
        if item.startswith("-"):
            remaining.append(item)
            name = item.split("=", 1)[0]
            if (
                name in _VALUE_OPTIONS
                and "=" not in item
                and index + 1 < len(values)
            ):
                remaining.append(values[index + 1])
                index += 2
                continue
            index += 1
            continue
        if command is None:
            command = item
            index += 1
            continue
        raise CliUsageError(USAGE_COMMAND)
    return command, remaining


def context_from_namespace(namespace: argparse.Namespace) -> CliContext:
    """Validate explicit arguments. Never substitutes the process working directory."""

    command = getattr(namespace, "command", None)
    if not command:
        raise CliUsageError(USAGE_COMMAND)
    repository = getattr(namespace, "repository", None)
    if not isinstance(repository, str) or not repository.strip():
        raise CliUsageError(USAGE_REPOSITORY)
    task_id = getattr(namespace, "task", None)
    if not isinstance(task_id, str) or not task_id.strip():
        raise CliUsageError(USAGE_TASK)
    correlation = getattr(namespace, "correlation", None)
    if not isinstance(correlation, str) or not correlation.strip():
        raise CliUsageError(USAGE_CORRELATION)
    policy = getattr(namespace, "policy", DEFAULT_POLICY)
    try:
        admitted_policy = admit_mode(policy)
    except Exception as exc:
        raise CliUsageError(USAGE_POLICY) from exc
    output_mode = getattr(namespace, "output_mode", DEFAULT_OUTPUT_MODE)
    if output_mode not in OUTPUT_MODES:
        raise CliUsageError(USAGE_OUTPUT_MODE)
    state_dir_raw = getattr(namespace, "state_dir", None)
    run_id = getattr(namespace, "run_id", None)
    repository_id = getattr(namespace, "repository_id", None)
    repository_path = Path(repository).expanduser()
    if repository_path.is_absolute():
        repository_path = repository_path.resolve(strict=False)
    state_dir = None
    if isinstance(state_dir_raw, str) and state_dir_raw.strip():
        state_path = Path(state_dir_raw).expanduser()
        if not state_path.is_absolute():
            state_path = repository_path / state_path
        elif state_path.is_absolute():
            state_path = state_path.resolve(strict=False)
        state_dir = state_path
    return CliContext(
        command=str(command),
        repository=repository_path,
        policy=admitted_policy,
        task_id=task_id.strip(),
        correlation_id=correlation.strip(),
        output_mode=str(output_mode),
        run_id=str(run_id).strip() if isinstance(run_id, str) and run_id.strip() else None,
        repository_id=(
            str(repository_id).strip()
            if isinstance(repository_id, str) and repository_id.strip()
            else None
        ),
        state_dir=state_dir,
    )


def parse_argv(argv: Sequence[str] | None) -> CliContext:
    values = list(argv) if argv is not None else []
    command, remaining = _split_command(values)
    if command not in STATE_COMMANDS:
        raise CliUsageError(USAGE_COMMAND)
    parser = _StableParser(
        prog=PROG,
        add_help=False,
        allow_abbrev=False,
    )
    _add_common_arguments(parser)
    namespace = parser.parse_args(remaining)
    namespace.command = command
    return context_from_namespace(namespace)


def usage_result(
    *,
    command: str | None,
    message: str,
    policy: str = DEFAULT_POLICY,
    correlation_id: str = "",
    output_mode: str = DEFAULT_OUTPUT_MODE,
) -> CliResult:
    return CliResult(
        command=command or "",
        status="invalid",
        policy=policy,
        correlation_id=correlation_id,
        output_mode=output_mode,
        payload={"reason": message, "argument_error": True},
        error="malformed",
        provenance="live",
    )


def render(result: CliResult) -> str:
    payload = dict(result.to_mapping())
    if result.output_mode == "human":
        lines = [
            f"command: {payload['command'] or '(none)'}",
            f"status: {payload['status']}",
            f"exit_code: {payload['exit_code']}",
            f"policy: {payload['policy']}",
            f"correlation_id: {payload['correlation_id']}",
        ]
        if payload.get("artifact_cid"):
            lines.append(f"artifact_cid: {payload['artifact_cid']}")
        if payload.get("provenance"):
            lines.append(f"provenance: {payload['provenance']}")
        if payload.get("contract"):
            lines.append(f"contract: {payload['contract']}")
        if payload.get("error"):
            lines.append(f"error: {payload['error']}")
        identities = payload.get("identities") or {}
        if isinstance(identities, Mapping):
            for key in (
                "repository_id",
                "repository_state_cid",
                "task_id",
                "run_id",
                "trace_id",
            ):
                value = identities.get(key)
                if value:
                    lines.append(f"{key}: {value}")
        return "\n".join(lines) + "\n"
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def peek_output_mode(argv: Sequence[str]) -> str:
    values = list(argv)
    for index, item in enumerate(values):
        if item == "--output-mode" and index + 1 < len(values):
            candidate = values[index + 1]
            if candidate in OUTPUT_MODES:
                return candidate
        if item.startswith("--output-mode="):
            candidate = item.split("=", 1)[1]
            if candidate in OUTPUT_MODES:
                return candidate
    return DEFAULT_OUTPUT_MODE


def peek_policy(argv: Sequence[str]) -> str:
    values = list(argv)
    for index, item in enumerate(values):
        if item == "--policy" and index + 1 < len(values):
            return values[index + 1]
        if item.startswith("--policy="):
            return item.split("=", 1)[1]
    return DEFAULT_POLICY


def peek_correlation(argv: Sequence[str]) -> str:
    values = list(argv)
    for index, item in enumerate(values):
        if item == "--correlation" and index + 1 < len(values):
            return values[index + 1]
        if item.startswith("--correlation="):
            return item.split("=", 1)[1]
    return ""


def peek_command(argv: Sequence[str]) -> str | None:
    for item in argv:
        if item in STATE_COMMANDS:
            return item
        if item.startswith("-"):
            continue
        return item
    return None


def dispatch(context: CliContext) -> CliResult:
    """Dispatch a validated context to a state command. Runtime work starts here."""

    from ipfs_accelerate_py.proof_context.cli.state_commands import (
        cmd_init,
        cmd_plan,
        cmd_scan,
        cmd_status,
    )

    handlers = {
        "init": cmd_init,
        "scan": cmd_scan,
        "status": cmd_status,
        "plan": cmd_plan,
    }
    handler = handlers[context.command]
    return handler(context)


def _write(stream: TextIO | None, text: str) -> None:
    target = stream if stream is not None else sys.stdout
    target.write(text)
    if hasattr(target, "flush"):
        target.flush()


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """CLI entry. Help and argument errors are stable; work starts only here."""

    values = list(sys.argv[1:] if argv is None else argv)
    output_mode = peek_output_mode(values)
    policy = peek_policy(values)
    correlation = peek_correlation(values)
    command = peek_command(values)
    if "--help" in values or "-h" in values:
        parser = build_parser()
        _write(stdout if stdout is not None else sys.stdout, parser.format_help())
        return EXIT_SUCCEEDED
    try:
        context = parse_argv(values)
    except CliUsageError as exc:
        result = usage_result(
            command=command,
            message=str(exc),
            policy=policy if policy in MODES else DEFAULT_POLICY,
            correlation_id=correlation,
            output_mode=output_mode,
        )
        _write(stdout, render(result))
        _write(stderr if stderr is not None else sys.stderr, f"{PROG}: {exc}\n")
        return EXIT_USAGE
    except argparse.ArgumentError as exc:
        result = usage_result(
            command=command,
            message=_stabilize_usage(str(exc)),
            policy=policy if policy in MODES else DEFAULT_POLICY,
            correlation_id=correlation,
            output_mode=output_mode,
        )
        _write(stdout, render(result))
        _write(
            stderr if stderr is not None else sys.stderr,
            f"{PROG}: {_stabilize_usage(str(exc))}\n",
        )
        return EXIT_USAGE
    result = dispatch(context)
    _write(stdout, render(result))
    return exit_code_for(result.status, provenance=result.provenance)


__all__ = [
    "CLI_RESULT_SCHEMA",
    "CLI_SCHEMA",
    "CONTRACT_VERSION",
    "DEFAULT_OUTPUT_MODE",
    "DEFAULT_POLICY",
    "EXIT_CODES",
    "INFERS_CURRENT_DIRECTORY",
    "INTERFACE",
    "OUTPUT_MODES",
    "PROG",
    "PROVIDER_BOUND",
    "REQUIRED_ARGUMENTS",
    "SCHEMA",
    "SIBLING_LAYOUT_REQUIRED",
    "STATE_COMMANDS",
    "USAGE_COMMAND",
    "USAGE_CORRELATION",
    "USAGE_OUTPUT_MODE",
    "USAGE_POLICY",
    "USAGE_REPOSITORY",
    "USAGE_TASK",
    "CliContext",
    "CliResult",
    "CliUsageError",
    "build_parser",
    "context_from_namespace",
    "dispatch",
    "exit_code_for",
    "main",
    "parse_argv",
    "render",
]
