"""Prompt-first product CLI: ``ipfs-accelerate supervisor …`` (ASE3-010 / DOEP-014).

Registration is parser-only and cold-safe: help/parse paths do not import the
production facade, open DuckDB, or start processes. Dispatch lazily composes
:class:`~.facade.Supervisor` for lifecycle commands and delegates direct
objective submission to :func:`~.intent_service.submit_objective` without a
second planner, objective store, or admission path.
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

# ASE3-010 prompt-lifecycle vocabulary (kept MCP-parity stable).
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

# DOEP-014 thin CLI client for canonical objective submission.
CANONICAL_CLI_CLIENT: Final = "SupervisorCLI@1"
CANONICAL_CLI_OBJECTIVE_SUBMISSION_COMMAND: Final = "submit-objective"
CANONICAL_CLI_OBJECTIVE_SUBMISSION_ENTRYPOINT: Final = "submit_objective_cli"
OBJECTIVE_SUBMISSION_CLI_COMMANDS: Final[tuple[str, ...]] = (
    CANONICAL_CLI_OBJECTIVE_SUBMISSION_COMMAND,
)


class SupervisorCLIError(RuntimeError):
    """Typed CLI failure before facade or objective-submission dispatch."""


def register_supervisor_cli(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register the lightweight ``supervisor`` group (parser-only)."""

    group = subparsers.add_parser(
        "supervisor",
        help="Prompt-first supervisor lifecycle and direct objective submission.",
        description=(
            "Product path for prompt-only self-improvement and DOEP direct "
            "objective submission. Normal run/preview input is a prompt; "
            "submit-objective accepts a datasets SupervisorObjectiveIntent JSON "
            "payload and never accepts caller-supplied authoritative policy."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  ipfs-accelerate supervisor run "Improve validation gates"\n'
            "  ipfs-accelerate supervisor preview --prompt-file intent.txt\n"
            "  ipfs-accelerate supervisor status --run-id RUN --output-json\n"
            "  ipfs-accelerate supervisor init --consent\n"
            "  ipfs-accelerate supervisor submit-objective "
            "--intent-file intent.json --output-json\n"
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

    submit_p = commands.add_parser(
        CANONICAL_CLI_OBJECTIVE_SUBMISSION_COMMAND,
        help=(
            "Submit one datasets SupervisorObjectiveIntent through the "
            "canonical objective-submission service (no policy overrides)."
        ),
    )
    _add_common(submit_p)
    submit_p.add_argument(
        "--intent-file",
        type=Path,
        help="Read SupervisorObjectiveIntent JSON from a file.",
    )
    submit_p.add_argument(
        "--intent-json",
        help="Inline SupervisorObjectiveIntent JSON object.",
    )
    submit_p.add_argument(
        "--intent-stdin",
        action="store_true",
        help="Read SupervisorObjectiveIntent JSON from stdin (bounded).",
    )
    return group


def supervisor_cli_discovery_manifest() -> dict[str, Any]:
    """Static vocabulary for help/conformance without constructing services."""

    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.supervisor-cli-discovery@1",
        "group": "supervisor",
        "commands": list(SUPERVISOR_COMMANDS),
        "objective_submission_commands": list(OBJECTIVE_SUBMISSION_CLI_COMMANDS),
        "canonical_cli_client": CANONICAL_CLI_CLIENT,
        "objective_submission_entrypoint": CANONICAL_CLI_OBJECTIVE_SUBMISSION_ENTRYPOINT,
        "objective_submission_delegate": (
            "ipfs_accelerate_py.agent_supervisor.entrypoints.intent_service"
            ".submit_objective"
        ),
        "console_entry": "ipfs-accelerate",
        "cold_help": True,
        "side_effect_free_parse": True,
        "callers_supply_authoritative_policy": False,
        "completion_authority": False,
        "competing_subsystem_created": False,
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


def _resolve_intent_mapping(
    args: argparse.Namespace, *, stdin: TextIO = sys.stdin
) -> Mapping[str, Any]:
    """Load exactly one SupervisorObjectiveIntent JSON source."""

    sources = [
        bool(getattr(args, "intent_file", None)),
        bool(getattr(args, "intent_json", None)),
        bool(getattr(args, "intent_stdin", False)),
    ]
    if sum(1 for item in sources if item) != 1:
        raise SupervisorCLIError(
            "supply exactly one of --intent-file, --intent-json, or --intent-stdin"
        )
    if getattr(args, "intent_file", None) is not None:
        path = Path(args.intent_file)
        if not path.is_file():
            raise SupervisorCLIError(f"intent file not found: {path}")
        raw = path.read_text(encoding="utf-8")
    elif getattr(args, "intent_stdin", False):
        raw = stdin.read(1_048_576)
    else:
        raw = str(getattr(args, "intent_json", "") or "")
    if not raw or not str(raw).strip():
        raise SupervisorCLIError("intent JSON must be a non-empty object")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SupervisorCLIError(f"intent JSON is invalid: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SupervisorCLIError("intent JSON must decode to an object")
    # Reject caller-supplied authority fields at the CLI boundary before
    # delegating so policy overrides never reach the submission service.
    forbidden = (
        "authorization",
        "authorization_decision",
        "budget_profile",
        "budgets",
        "completion_authoritative",
        "duckdb",
        "ducklake",
        "execution_authorization",
        "fencing_epoch",
        "lease_id",
        "objective_cid",
        "objective_revision_cid",
        "plan",
        "plan_root_cid",
        "policy",
        "policy_document",
        "policy_id",
        "policy_revision",
        "risk_class",
        "terminalize",
    )
    hit = sorted(key for key in payload if key in forbidden)
    if hit:
        raise SupervisorCLIError(
            "objective submission rejects caller-supplied authority fields: "
            + ", ".join(hit)
        )
    return dict(payload)


def submit_objective_cli(
    intent: Mapping[str, Any] | Any,
    *,
    submit_objective: Any | None = None,
) -> Mapping[str, Any]:
    """Thin CLI client adapter over the canonical objective-submission service.

    This is intentionally not a second objective subsystem: semantic validation
    and identity minting stay with datasets + intent_service.submit_objective.
    The returned mapping is the materialization receipt evidence only.
    """

    if submit_objective is None:
        from .intent_service import submit_objective as _submit_objective

        submit_objective = _submit_objective
    receipt = submit_objective(intent)
    if hasattr(receipt, "to_dict"):
        payload = receipt.to_dict()
    elif isinstance(receipt, Mapping):
        payload = dict(receipt)
    else:
        raise SupervisorCLIError("objective submission returned an unreadable receipt")
    if not isinstance(payload, Mapping):
        raise SupervisorCLIError("objective submission receipt must be a mapping")
    return dict(payload)


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
    submit_objective: Any = None,
) -> int:
    """Dispatch one supervisor command through the facade or CLI client."""

    command = getattr(args, "supervisor_command", None)
    output_json = bool(getattr(args, "output_json", False))
    if not command:
        # Caller should have printed help; treat as invalid usage.
        return EXIT_INVALID

    try:
        if command == CANONICAL_CLI_OBJECTIVE_SUBMISSION_COMMAND:
            intent = _resolve_intent_mapping(args, stdin=stdin)
            receipt = submit_objective_cli(intent, submit_objective=submit_objective)
            payload = {
                **receipt,
                "summary": (
                    f"objective submitted intent_id={receipt.get('intent_id', '')} "
                    f"receipt_id={receipt.get('receipt_id', '')}"
                ),
            }
            env = _envelope(ok=True, command=command, payload=payload)
            _emit(env, output_json=output_json or True, stream=stdout)
            return EXIT_SUCCESS

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
    except Exception as exc:  # map typed facade / submission errors
        from .facade import (
            SupervisorAmbiguityError,
            SupervisorConfigurationError,
            SupervisorUnavailableError,
        )
        from .intent_service import (
            ObjectiveSubmissionContractError,
            ObjectiveSubmissionPolicyError,
            ObjectiveSubmissionUnavailableError,
        )

        if isinstance(exc, ObjectiveSubmissionPolicyError):
            code, exit_code = "policy", EXIT_CONFIG
        elif isinstance(exc, ObjectiveSubmissionContractError):
            code, exit_code = "invalid", EXIT_INVALID
        elif isinstance(exc, ObjectiveSubmissionUnavailableError):
            code, exit_code = "unavailable", EXIT_UNAVAILABLE
        elif isinstance(exc, SupervisorConfigurationError):
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
    "CANONICAL_CLI_CLIENT",
    "CANONICAL_CLI_OBJECTIVE_SUBMISSION_COMMAND",
    "CANONICAL_CLI_OBJECTIVE_SUBMISSION_ENTRYPOINT",
    "EXIT_AMBIGUITY",
    "EXIT_CONFIG",
    "EXIT_INVALID",
    "EXIT_SUCCESS",
    "EXIT_UNAVAILABLE",
    "OBJECTIVE_SUBMISSION_CLI_COMMANDS",
    "SUPERVISOR_COMMANDS",
    "SupervisorCLIError",
    "register_supervisor_cli",
    "run_supervisor_cli",
    "submit_objective_cli",
    "supervisor_cli_discovery_manifest",
]
