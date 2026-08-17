"""Prompt-first product CLI: ``ipfs-accelerate supervisor …`` (ASE3-010).

Registration is parser-only and cold-safe: help/parse paths do not import the
production facade, open DuckDB, or start processes. Dispatch lazily composes
:class:`~.facade.Supervisor`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, TextIO

# Stable exit codes for typed facade outcomes.
EXIT_SUCCESS = 0
EXIT_UNAVAILABLE = 1
EXIT_INVALID = 2
EXIT_AMBIGUITY = 3
EXIT_CONFIG = 4

SUPERVISOR_COMMANDS: Final[tuple[str, ...]] = (
    "run",
    "preview",
    "steer",
    "status",
    "follow",
    "explain",
    "doctor",
    "init",
)


class SupervisorCLIError(RuntimeError):
    """Typed CLI failure before facade dispatch."""


def register_supervisor_cli(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register the lightweight ``supervisor`` group (parser-only)."""

    group = subparsers.add_parser(
        "supervisor",
        help="Prompt-first supervisor lifecycle (run/preview/steer/status/…).",
        description=(
            "Product path for prompt-only self-improvement. Normal run/preview "
            "input is a prompt; advanced flags are optional authorized overrides."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  ipfs-accelerate supervisor run "Improve validation gates"\n'
            "  ipfs-accelerate supervisor preview --prompt-file intent.txt\n"
            "  ipfs-accelerate supervisor status --run-id RUN --output-json\n"
            "  ipfs-accelerate supervisor init --consent\n"
        ),
    )
    commands = group.add_subparsers(
        dest="supervisor_command",
        metavar="COMMAND",
        help="Supervisor lifecycle operation.",
    )

    def _add_common(child: argparse.ArgumentParser) -> None:
        child.add_argument(
            "--repository",
            help="Repository root (defaults to nearest enclosing Git root).",
        )
        child.add_argument(
            "--state-root",
            help="Optional state root override.",
        )
        child.add_argument(
            "--output-json",
            action="store_true",
            help="Emit a structured JSON envelope on stdout.",
        )

    def _add_prompt(child: argparse.ArgumentParser, *, required: bool = False) -> None:
        child.add_argument(
            "prompt",
            nargs="?" if not required else None,
            help="Prompt text (positional).",
        )
        child.add_argument(
            "--prompt-file",
            type=Path,
            help="Read prompt body from a file (preferred over argv for secrets).",
        )
        child.add_argument(
            "--prompt-stdin",
            action="store_true",
            help="Read prompt body from stdin (bounded).",
        )

    run_p = commands.add_parser("run", help="Start or resume a durable run from a prompt.")
    _add_common(run_p)
    _add_prompt(run_p)

    preview_p = commands.add_parser(
        "preview", help="Preview resolution without authorizing effects."
    )
    _add_common(preview_p)
    _add_prompt(preview_p)

    steer_p = commands.add_parser("steer", help="Steer an existing run with a prompt.")
    _add_common(steer_p)
    steer_p.add_argument("--run-id", required=True, help="Exact run identifier.")
    _add_prompt(steer_p)

    status_p = commands.add_parser("status", help="Observe run status.")
    _add_common(status_p)
    status_p.add_argument("--run-id", help="Run id (optional when unique).")

    follow_p = commands.add_parser("follow", help="Follow run event cursor.")
    _add_common(follow_p)
    follow_p.add_argument("--run-id", help="Run id (optional when unique).")

    explain_p = commands.add_parser("explain", help="Body-free explanation of a run.")
    _add_common(explain_p)
    explain_p.add_argument("--run-id", help="Run id (optional when unique).")

    doctor_p = commands.add_parser(
        "doctor", help="Doctor snapshot (detection does not grant restart)."
    )
    _add_common(doctor_p)
    doctor_p.add_argument("--run-id", help="Run id (optional when unique).")

    init_p = commands.add_parser(
        "init", help="One-time local profile bootstrap (requires --consent)."
    )
    _add_common(init_p)
    init_p.add_argument(
        "--consent",
        action="store_true",
        help="Explicit consent for local initialization.",
    )
    return group


def supervisor_cli_discovery_manifest() -> dict[str, Any]:
    """Static vocabulary for help/conformance without constructing services."""

    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.supervisor-cli-discovery@1",
        "group": "supervisor",
        "commands": list(SUPERVISOR_COMMANDS),
        "console_entry": "ipfs-accelerate",
        "cold_help": True,
        "side_effect_free_parse": True,
    }


def _resolve_prompt(args: argparse.Namespace, *, stdin: TextIO = sys.stdin) -> str:
    sources = [
        bool(getattr(args, "prompt", None)),
        bool(getattr(args, "prompt_file", None)),
        bool(getattr(args, "prompt_stdin", False)),
    ]
    if sum(1 for item in sources if item) > 1:
        raise SupervisorCLIError("supply exactly one of prompt, --prompt-file, or --prompt-stdin")
    if getattr(args, "prompt_file", None) is not None:
        path = Path(args.prompt_file)
        if not path.is_file():
            raise SupervisorCLIError(f"prompt file not found: {path}")
        text = path.read_text(encoding="utf-8")
    elif getattr(args, "prompt_stdin", False):
        text = stdin.read(1_048_576)
    else:
        text = str(getattr(args, "prompt", None) or "")
    if not text or not str(text).strip():
        raise SupervisorCLIError("prompt must be a non-empty string")
    return str(text)


def _envelope(
    *,
    ok: bool,
    command: str,
    payload: Mapping[str, Any] | None = None,
    error: str | None = None,
    error_code: str | None = None,
    composition_cid: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": "ipfs_accelerate_py.agent_supervisor.supervisor-cli-result@1",
        "ok": ok,
        "command": command,
    }
    if composition_cid:
        body["composition_cid"] = composition_cid
    if payload is not None:
        body["result"] = dict(payload)
    if error is not None:
        body["error"] = error
    if error_code is not None:
        body["error_code"] = error_code
    return body


def _emit(
    payload: Mapping[str, Any],
    *,
    output_json: bool,
    stream: TextIO = sys.stdout,
) -> None:
    if output_json:
        stream.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        return
    if payload.get("ok"):
        result = payload.get("result") or {}
        if isinstance(result, Mapping):
            summary = result.get("summary") or result.get("run_id") or "ok"
            stream.write(f"{summary}\n")
            if result.get("run_id"):
                stream.write(f"run_id={result['run_id']}\n")
            if payload.get("composition_cid"):
                stream.write(f"composition_cid={payload['composition_cid']}\n")
        else:
            stream.write("ok\n")
    else:
        stream.write(f"error: {payload.get('error') or 'failed'}\n")


def run_supervisor_cli(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    stdin: TextIO = sys.stdin,
    supervisor: Any = None,
) -> int:
    """Dispatch one supervisor command through the production facade."""

    command = getattr(args, "supervisor_command", None)
    output_json = bool(getattr(args, "output_json", False))
    if not command:
        # Caller should have printed help; treat as invalid usage.
        return EXIT_INVALID

    try:
        if supervisor is None:
            from .facade import Supervisor

            if command == "init":
                receipt = Supervisor.init_local(
                    repository=getattr(args, "repository", None),
                    consent=bool(getattr(args, "consent", False)),
                )
                env = _envelope(
                    ok=True,
                    command=command,
                    payload=dict(receipt) if isinstance(receipt, Mapping) else {"receipt": receipt},
                )
                _emit(env, output_json=output_json, stream=stdout)
                return EXIT_SUCCESS

            supervisor = Supervisor.open(
                repository=getattr(args, "repository", None),
                state_root=getattr(args, "state_root", None),
            )

        composition_cid = getattr(supervisor, "composition_cid", None)

        if command == "run":
            prompt = _resolve_prompt(args, stdin=stdin)
            run = supervisor.run(prompt)
            payload = {
                "run_id": run.run_id,
                "state": run.state,
                "health": run.health,
                "event_cursor": run.event_cursor,
                "effect_receipt_cids": list(run.effect_receipt_cids),
                "summary": f"run started run_id={run.run_id}",
            }
        elif command == "preview":
            prompt = _resolve_prompt(args, stdin=stdin)
            obs = supervisor.preview(prompt)
            payload = obs.to_dict()
        elif command == "steer":
            prompt = _resolve_prompt(args, stdin=stdin)
            obs = supervisor.steer(str(args.run_id), prompt)
            payload = obs.to_dict()
        elif command == "status":
            obs = supervisor.status(getattr(args, "run_id", None))
            payload = obs.to_dict()
        elif command == "follow":
            events = []
            for obs in supervisor.follow(getattr(args, "run_id", None)):
                events.append(obs.to_dict())
            payload = {"events": events, "summary": f"followed {len(events)} event(s)"}
        elif command == "explain":
            obs = supervisor.explain(getattr(args, "run_id", None))
            payload = obs.to_dict()
        elif command == "doctor":
            obs = supervisor.doctor(getattr(args, "run_id", None))
            payload = obs.to_dict()
        elif command == "init":
            # Handled above when supervisor is None; injectable path:
            from .facade import Supervisor as _Supervisor

            receipt = _Supervisor.init_local(
                repository=getattr(args, "repository", None),
                consent=bool(getattr(args, "consent", False)),
            )
            payload = dict(receipt) if isinstance(receipt, Mapping) else {"receipt": receipt}
            composition_cid = None
        else:
            raise SupervisorCLIError(f"unknown supervisor command: {command}")

        env = _envelope(
            ok=True,
            command=command,
            payload=payload,
            composition_cid=str(composition_cid) if composition_cid else None,
        )
        _emit(env, output_json=output_json, stream=stdout)
        return EXIT_SUCCESS
    except SupervisorCLIError as exc:
        env = _envelope(ok=False, command=str(command), error=str(exc), error_code="invalid")
        _emit(env, output_json=output_json, stream=stderr if not output_json else stdout)
        return EXIT_INVALID
    except Exception as exc:  # map typed facade errors
        from .facade import (
            SupervisorAmbiguityError,
            SupervisorConfigurationError,
            SupervisorUnavailableError,
        )

        if isinstance(exc, SupervisorConfigurationError):
            code, exit_code = "configuration", EXIT_CONFIG
        elif isinstance(exc, SupervisorAmbiguityError):
            code, exit_code = "ambiguity", EXIT_AMBIGUITY
        elif isinstance(exc, SupervisorUnavailableError):
            code, exit_code = "unavailable", EXIT_UNAVAILABLE
        else:
            code, exit_code = "error", EXIT_UNAVAILABLE
        env = _envelope(
            ok=False,
            command=str(command),
            error=str(exc),
            error_code=code,
        )
        _emit(env, output_json=output_json, stream=stderr if not output_json else stdout)
        return exit_code


__all__ = [
    "EXIT_AMBIGUITY",
    "EXIT_CONFIG",
    "EXIT_INVALID",
    "EXIT_SUCCESS",
    "EXIT_UNAVAILABLE",
    "SUPERVISOR_COMMANDS",
    "SupervisorCLIError",
    "register_supervisor_cli",
    "run_supervisor_cli",
    "supervisor_cli_discovery_manifest",
]
