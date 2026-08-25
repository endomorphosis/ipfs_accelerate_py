"""Structured argv adapter for External Agent Handoff (EAAEF-111).

This CLI is a thin argparse surface over
:class:`~ipfs_accelerate_py.agent_supervisor.api.external_handoff.ExternalHandoffAPI`.
It never executes a shell and never talks to live Quack or Docker.  Large
instruction/reason/request bodies are passed as file references rather than
argv text.  ``--detach`` returns run identities without following events.
Worker principals cannot approve or reject.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, TextIO

from ipfs_accelerate_py.agent_supervisor.api.external_handoff import (
    HANDOFF_API_OPERATIONS,
    MAX_INSTRUCTION_BYTES,
    MAX_REASON_BYTES,
    ExternalHandoffAPI,
    ExternalHandoffAPIError,
    ExternalHandoffAuthorityError,
    ExternalHandoffReceipt,
    WorkerSelfApprovalError,
    get_default_api,
)


HANDOFF_CLI_INTERFACE: Final[str] = "ExternalHandoffCLI@1"
HANDOFF_CLI_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-handoff-cli@1"
)
HANDOFF_CLI_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-handoff-cli-result@1"
)
GROUP: Final[str] = "supervisor-handoff"
CONSOLE_ENTRY: Final[str] = "ipfs-accelerate"

EXIT_SUCCESS: Final[int] = 0
EXIT_ERROR: Final[int] = 1
EXIT_USAGE: Final[int] = 2
EXIT_NOT_FOUND: Final[int] = 3
EXIT_AUTHORITY: Final[int] = 4
EXIT_WORKER_APPROVAL: Final[int] = 5

MAX_REQUEST_BYTES: Final[int] = 65_536
MAX_OUTPUT_CHARS: Final[int] = 262_144

CLI_COMMANDS: Final[tuple[str, ...]] = (
    "handoff",
    "status",
    "follow",
    "attach",
    "steer",
    "pause",
    "resume",
    "approve",
    "reject",
    "cancel",
    "explain",
    "doctor",
    "report",
    "export-result",
)
CLI_TO_OPERATION: Final[dict[str, str]] = {
    command: "export" if command == "export-result" else command
    for command in CLI_COMMANDS
}
_AUTHORITY_COMMANDS: Final[frozenset[str]] = frozenset(
    {"attach", "steer", "pause", "resume", "cancel"}
)
_REVIEW_COMMANDS: Final[frozenset[str]] = frozenset({"approve", "reject"})
_ORIGIN_COMMANDS: Final[frozenset[str]] = frozenset({"handoff"})
_REQUEST_FIELDS: Final[tuple[str, ...]] = (
    "principal_id",
    "worker_principal_id",
    "reviewer_principal_id",
    "authority_id",
    "run_id",
    "session_id",
    "repository_id",
    "objective_id",
    "idempotency_key",
    "cursor",
    "instruction",
    "reason",
)
_REASON_EXITS: Final[Mapping[str, int]] = {
    "malformed": EXIT_USAGE,
    "unknown_operation": EXIT_USAGE,
    "operation_mismatch": EXIT_USAGE,
    "private_material": EXIT_USAGE,
    "unsupported_version": EXIT_USAGE,
    "identity_mismatch": EXIT_USAGE,
    "bounds": EXIT_USAGE,
    "unknown_run": EXIT_NOT_FOUND,
    "unknown_cursor": EXIT_NOT_FOUND,
    "authority_mismatch": EXIT_AUTHORITY,
    "worker_self_approval": EXIT_WORKER_APPROVAL,
    "missing_reviewer": EXIT_WORKER_APPROVAL,
    "missing_worker": EXIT_WORKER_APPROVAL,
}


class SupervisorHandoffCLIError(ValueError):
    """Typed CLI failure before API dispatch."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "malformed",
        exit_code: int = EXIT_USAGE,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.exit_code = int(exit_code)


def exit_code_for_reason(reason_code: str) -> int:
    """Map a closed API/CLI reason code to a typed process exit state."""

    return int(_REASON_EXITS.get(str(reason_code or ""), EXIT_ERROR))


def build_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    """Build the standalone structured-argv parser (no host side effects)."""

    parser = argparse.ArgumentParser(
        prog=prog or GROUP,
        description=(
            "External Agent Handoff CLI.  Structured argv only; large bodies "
            "are file references; --detach returns identities without follow."
        ),
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True
    _register_commands(subparsers)
    return parser


def register_supervisor_handoff_cli(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register the ``supervisor-handoff`` group on a host parser (parser-only)."""

    group = subparsers.add_parser(
        GROUP,
        help="External agent handoff lifecycle (structured argv, no shell).",
        description=(
            "Admit, observe, steer, and independently review an external "
            "handoff run.  Worker principals cannot approve."
        ),
        allow_abbrev=False,
    )
    commands = group.add_subparsers(dest="command", metavar="COMMAND")
    commands.required = True
    _register_commands(commands)
    return group


def _identity_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parent.add_argument("--principal-id", dest="principal_id", default="")
    parent.add_argument("--worker-principal-id", dest="worker_principal_id", default="")
    parent.add_argument(
        "--reviewer-principal-id", dest="reviewer_principal_id", default=""
    )
    parent.add_argument("--authority-id", dest="authority_id", default="")
    parent.add_argument("--run-id", dest="run_id", default="")
    parent.add_argument("--session-id", dest="session_id", default="")
    parent.add_argument("--repository-id", dest="repository_id", default="")
    parent.add_argument("--objective-id", dest="objective_id", default="")
    parent.add_argument("--idempotency-key", dest="idempotency_key", default="")
    parent.add_argument("--cursor", dest="cursor", default="")
    parent.add_argument(
        "--request-file",
        dest="request_file",
        type=Path,
        help="Large-input JSON request reference (preferred over inline bodies).",
    )
    parent.add_argument(
        "--output-json",
        dest="output_json",
        action="store_true",
        default=True,
        help="Emit a canonical JSON envelope (default).",
    )
    parent.add_argument(
        "--human",
        dest="output_json",
        action="store_false",
        help="Emit a concise human summary instead of full JSON.",
    )
    return parent


def _instruction_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    source = parent.add_mutually_exclusive_group()
    source.add_argument(
        "--instruction",
        dest="instruction",
        default="",
        help="Small inline instruction. Prefer --instruction-file for large input.",
    )
    source.add_argument(
        "--instruction-file",
        dest="instruction_file",
        type=Path,
        help="Large-input instruction reference (UTF-8 file, not argv body).",
    )
    source.add_argument(
        "--instruction-stdin",
        dest="instruction_stdin",
        action="store_true",
        help="Read instruction body from stdin (bounded).",
    )
    return parent


def _reason_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    source = parent.add_mutually_exclusive_group()
    source.add_argument("--reason", dest="reason", default="")
    source.add_argument(
        "--reason-file",
        dest="reason_file",
        type=Path,
        help="Large-input reason reference (UTF-8 file, not argv body).",
    )
    return parent


def _register_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    identity = _identity_parent()
    instruction = _instruction_parent()
    reason = _reason_parent()

    for command in CLI_COMMANDS:
        parents = [identity]
        help_text = {
            "handoff": "Admit a handoff run and return identities.",
            "status": "Observe an admitted run.",
            "follow": "Follow run events from an optional cursor.",
            "attach": "Attach to an admitted run.",
            "steer": "Steer a running admitted run.",
            "pause": "Pause a running admitted run.",
            "resume": "Resume a paused admitted run.",
            "approve": "Independently approve a run (worker cannot approve).",
            "reject": "Independently reject a run (worker cannot reject).",
            "cancel": "Cancel an admitted run.",
            "explain": "Body-free explanation of a run.",
            "doctor": "Diagnose a run without granting restart.",
            "report": "Project a public run report.",
            "export-result": "Export a public result identity.",
        }[command]
        if command in {"steer", "handoff"}:
            parents.append(instruction)
        if command in {"cancel", "reject", "pause", "approve"}:
            parents.append(reason)
        child = subparsers.add_parser(
            command,
            help=help_text,
            parents=parents,
            allow_abbrev=False,
        )
        child.set_defaults(
            command=command,
            operation=CLI_TO_OPERATION[command],
            detach=False,
            instruction="",
            instruction_file=None,
            instruction_stdin=False,
            reason="",
            reason_file=None,
        )
        if command == "handoff":
            child.add_argument(
                "--detach",
                dest="detach",
                action="store_true",
                help="Return run identities immediately without following events.",
            )


def parse_argv(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a structured argv list.  Does not execute a shell."""

    parser = build_parser()
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(list(argv))


def _read_text_reference(
    path: Path,
    *,
    field: str,
    max_bytes: int,
) -> str:
    if not path.is_file():
        raise SupervisorHandoffCLIError(
            f"{field} not found",
            reason_code="malformed",
            exit_code=EXIT_USAGE,
        )
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise SupervisorHandoffCLIError(
            f"{field} exceeds {max_bytes} UTF-8 bytes",
            reason_code="bounds",
            exit_code=EXIT_USAGE,
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SupervisorHandoffCLIError(
            f"{field} is not valid UTF-8",
            reason_code="malformed",
            exit_code=EXIT_USAGE,
        ) from exc
    if "\x00" in text:
        raise SupervisorHandoffCLIError(
            f"{field} must not contain NUL",
            reason_code="malformed",
            exit_code=EXIT_USAGE,
        )
    return text


def _read_request_file(path: Path) -> dict[str, Any]:
    raw = _read_text_reference(path, field="request-file", max_bytes=MAX_REQUEST_BYTES)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SupervisorHandoffCLIError(
            "request-file is not valid JSON",
            reason_code="malformed",
            exit_code=EXIT_USAGE,
        ) from exc
    if not isinstance(payload, Mapping):
        raise SupervisorHandoffCLIError(
            "request-file must be a JSON object",
            reason_code="malformed",
            exit_code=EXIT_USAGE,
        )
    return dict(payload)


def _merge_text(
    *,
    flag_value: str,
    file_path: Path | None,
    stdin_flag: bool,
    stdin: TextIO,
    field: str,
    max_bytes: int,
) -> str:
    sources = (bool(flag_value), file_path is not None, bool(stdin_flag))
    if sum(1 for item in sources if item) > 1:
        raise SupervisorHandoffCLIError(
            f"supply at most one {field} source",
            reason_code="malformed",
            exit_code=EXIT_USAGE,
        )
    if file_path is not None:
        return _read_text_reference(
            file_path, field=f"{field}-file", max_bytes=max_bytes
        )
    if stdin_flag:
        body = stdin.read(max_bytes + 1)
        if len(body.encode("utf-8")) > max_bytes:
            raise SupervisorHandoffCLIError(
                f"{field} exceeds {max_bytes} UTF-8 bytes",
                reason_code="bounds",
                exit_code=EXIT_USAGE,
            )
        return body
    return str(flag_value or "")


def request_from_args(
    args: argparse.Namespace,
    *,
    stdin: TextIO = sys.stdin,
) -> dict[str, Any]:
    """Build a closed API request dict from parsed argv (no shell)."""

    command = str(getattr(args, "command", "") or "")
    if command not in CLI_TO_OPERATION:
        raise SupervisorHandoffCLIError(
            "unknown handoff CLI command",
            reason_code="unknown_operation",
            exit_code=EXIT_USAGE,
        )
    operation = CLI_TO_OPERATION[command]
    payload: dict[str, Any] = {"operation": operation}
    request_file = getattr(args, "request_file", None)
    if request_file is not None:
        loaded = _read_request_file(Path(request_file))
        supplied = loaded.get("operation")
        if supplied not in (None, "", operation):
            raise SupervisorHandoffCLIError(
                "request-file operation does not match the CLI command",
                reason_code="operation_mismatch",
                exit_code=EXIT_USAGE,
            )
        for field in _REQUEST_FIELDS:
            if field in loaded and loaded[field] not in (None, ""):
                payload[field] = loaded[field]

    for field in _REQUEST_FIELDS:
        if field in {"instruction", "reason"}:
            continue
        value = getattr(args, field, None)
        if value not in (None, ""):
            payload[field] = value

    instruction = _merge_text(
        flag_value=str(getattr(args, "instruction", "") or ""),
        file_path=getattr(args, "instruction_file", None),
        stdin_flag=bool(getattr(args, "instruction_stdin", False)),
        stdin=stdin,
        field="instruction",
        max_bytes=MAX_INSTRUCTION_BYTES,
    )
    if instruction:
        payload["instruction"] = instruction
    reason = _merge_text(
        flag_value=str(getattr(args, "reason", "") or ""),
        file_path=getattr(args, "reason_file", None),
        stdin_flag=False,
        stdin=stdin,
        field="reason",
        max_bytes=MAX_REASON_BYTES,
    )
    if reason:
        payload["reason"] = reason

    if not str(payload.get("principal_id") or "").strip():
        raise SupervisorHandoffCLIError(
            "principal-id is required",
            reason_code="malformed",
            exit_code=EXIT_USAGE,
        )
    if command not in _ORIGIN_COMMANDS and not str(payload.get("run_id") or "").strip():
        raise SupervisorHandoffCLIError(
            "run-id is required",
            reason_code="malformed",
            exit_code=EXIT_USAGE,
        )
    if command in _AUTHORITY_COMMANDS and not str(
        payload.get("authority_id") or ""
    ).strip():
        raise SupervisorHandoffCLIError(
            "authority-id is required",
            reason_code="authority_mismatch",
            exit_code=EXIT_AUTHORITY,
        )
    if command == "steer" and not str(payload.get("instruction") or "").strip():
        raise SupervisorHandoffCLIError(
            "instruction is required",
            reason_code="malformed",
            exit_code=EXIT_USAGE,
        )
    if command in _REVIEW_COMMANDS:
        _reject_worker_approval(payload)
    return payload


def _reject_worker_approval(payload: Mapping[str, Any]) -> None:
    worker = str(payload.get("worker_principal_id") or "").strip()
    reviewer = str(payload.get("reviewer_principal_id") or "").strip()
    principal = str(payload.get("principal_id") or "").strip()
    if not reviewer:
        raise SupervisorHandoffCLIError(
            "approve and reject require an independent reviewer principal",
            reason_code="missing_reviewer",
            exit_code=EXIT_WORKER_APPROVAL,
        )
    if worker and (reviewer == worker or principal == worker):
        raise SupervisorHandoffCLIError(
            "worker self-approval is forbidden",
            reason_code="worker_self_approval",
            exit_code=EXIT_WORKER_APPROVAL,
        )


def envelope(
    *,
    ok: bool,
    command: str,
    exit_code: int,
    receipt: ExternalHandoffReceipt | None = None,
    error: str | None = None,
    reason_code: str | None = None,
    detached: bool = False,
) -> dict[str, Any]:
    operation = CLI_TO_OPERATION.get(command, command)
    body: dict[str, Any] = {
        "schema": HANDOFF_CLI_RESULT_SCHEMA,
        "interface": HANDOFF_CLI_INTERFACE,
        "ok": bool(ok),
        "command": command,
        "operation": operation,
        "exit_code": int(exit_code),
        "detached": bool(detached),
        "shell": False,
        "live_quack": False,
        "live_docker": False,
    }
    if receipt is not None:
        body["receipt"] = receipt.to_dict()
        body["run_id"] = receipt.run_id
        body["authority_id"] = receipt.authority_id
        body["request_id"] = receipt.request_id
        body["run_status"] = receipt.run_status
        body["reason_code"] = receipt.reason_code
        if receipt.export_id:
            body["export_id"] = receipt.export_id
    if error is not None:
        body["error"] = error
    if reason_code is not None:
        body["reason_code"] = reason_code
    return body


def emit(
    payload: Mapping[str, Any],
    *,
    output_json: bool,
    stream: TextIO,
) -> None:
    if output_json:
        text = json.dumps(dict(payload), sort_keys=True, indent=2, ensure_ascii=True)
        if len(text) > MAX_OUTPUT_CHARS:
            text = json.dumps(
                {
                    "schema": HANDOFF_CLI_RESULT_SCHEMA,
                    "ok": payload.get("ok"),
                    "command": payload.get("command"),
                    "exit_code": payload.get("exit_code"),
                    "reason_code": "bounds",
                    "error": "output exceeded bound; truncated",
                },
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
            )
        if not text.endswith("\n"):
            text += "\n"
        stream.write(text)
        stream.flush()
        return
    if payload.get("ok"):
        stream.write(
            f"{payload.get('command')} ok run_id={payload.get('run_id', '')}\n"
        )
        if payload.get("authority_id"):
            stream.write(f"authority_id={payload['authority_id']}\n")
        if payload.get("detached"):
            stream.write("detached=true\n")
    else:
        stream.write(
            f"error: {payload.get('error') or 'failed'} "
            f"reason_code={payload.get('reason_code') or ''}\n"
        )
    stream.flush()


def dispatch(
    args: argparse.Namespace,
    *,
    api: ExternalHandoffAPI | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdin: TextIO | None = None,
) -> int:
    """Dispatch one parsed command onto ExternalHandoffAPI.  No shell."""

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    stdin = stdin if stdin is not None else sys.stdin
    command = str(getattr(args, "command", "") or "")
    output_json = bool(getattr(args, "output_json", True))
    detached = bool(getattr(args, "detach", False))
    try:
        request = request_from_args(args, stdin=stdin)
        operation = str(request["operation"])
        service = api if api is not None else get_default_api()
        method = getattr(service, operation)
        receipt = method(request)
        env = envelope(
            ok=True,
            command=command,
            exit_code=EXIT_SUCCESS,
            receipt=receipt,
            detached=detached,
        )
        emit(env, output_json=output_json, stream=stdout)
        return EXIT_SUCCESS
    except SupervisorHandoffCLIError as exc:
        env = envelope(
            ok=False,
            command=command,
            exit_code=exc.exit_code,
            error=str(exc),
            reason_code=exc.reason_code,
            detached=detached,
        )
        emit(env, output_json=output_json, stream=stdout if output_json else stderr)
        return exc.exit_code
    except WorkerSelfApprovalError as exc:
        env = envelope(
            ok=False,
            command=command,
            exit_code=EXIT_WORKER_APPROVAL,
            error=str(exc),
            reason_code=exc.reason_code,
            detached=detached,
        )
        emit(env, output_json=output_json, stream=stdout if output_json else stderr)
        return EXIT_WORKER_APPROVAL
    except ExternalHandoffAuthorityError as exc:
        env = envelope(
            ok=False,
            command=command,
            exit_code=EXIT_AUTHORITY,
            error=str(exc),
            reason_code=exc.reason_code,
            detached=detached,
        )
        emit(env, output_json=output_json, stream=stdout if output_json else stderr)
        return EXIT_AUTHORITY
    except ExternalHandoffAPIError as exc:
        code = exit_code_for_reason(exc.reason_code)
        env = envelope(
            ok=False,
            command=command,
            exit_code=code,
            error=str(exc),
            reason_code=exc.reason_code,
            detached=detached,
        )
        emit(env, output_json=output_json, stream=stdout if output_json else stderr)
        return code


def main(
    argv: Sequence[str] | None = None,
    *,
    api: ExternalHandoffAPI | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdin: TextIO | None = None,
) -> int:
    """Parse structured argv and dispatch.  Returns a typed exit state."""

    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        if code in (None, 0, "0"):
            return EXIT_SUCCESS
        return EXIT_USAGE
    return dispatch(
        args,
        api=api,
        stdout=stdout,
        stderr=stderr,
        stdin=stdin,
    )


def supervisor_handoff_cli_discovery_manifest() -> dict[str, Any]:
    """Static vocabulary for help/conformance without constructing services."""

    return {
        "schema": HANDOFF_CLI_SCHEMA,
        "interface": HANDOFF_CLI_INTERFACE,
        "group": GROUP,
        "console_entry": CONSOLE_ENTRY,
        "commands": list(CLI_COMMANDS),
        "operations": [CLI_TO_OPERATION[name] for name in CLI_COMMANDS],
        "api_operations": list(HANDOFF_API_OPERATIONS),
        "cold_help": True,
        "side_effect_free_parse": True,
        "shell": False,
        "self_approval": False,
        "detach": True,
        "large_input_references": True,
        "live_quack": False,
        "live_docker": False,
    }


if __name__ == "__main__":
    sys.exit(main())
