"""Prompt-only CLI for agent supervisor.

Usage (prompt-only defaults):
  agent-supervisor run "do the thing"
  agent-supervisor preview "plan the thing"
  agent-supervisor steer "nudge left" [--run-id RUN]
  agent-supervisor status              # infers sole compatible run
  agent-supervisor follow              # infers sole compatible run

Advanced flags are explicit overrides, never required.
Exit codes and outcome strings match the Python facade.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional, Sequence

from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import (
    EXIT_INVALID,
    AgentSupervisorFacade,
    get_facade,
    outcome_to_exit_code,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-supervisor",
        description=(
            "Prompt-only agent supervisor CLI. "
            "Normal usage: supply only a prompt (run/preview) or prompt plus "
            "optional run handle (steer). status/follow infer the sole run."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_prompt(p: argparse.ArgumentParser, required: bool = True) -> None:
        p.add_argument(
            "prompt",
            nargs=1 if required else "?",
            help="Natural-language prompt",
        )

    def add_advanced(p: argparse.ArgumentParser) -> None:
        # Advanced overrides — never required
        p.add_argument("--run-id", default=None, help="Explicit run handle (override)")
        p.add_argument("--backend", default=None, help="Backend override")
        p.add_argument("--model", default=None, help="Model override")
        p.add_argument("--timeout", type=float, default=None, help="Timeout override")
        p.add_argument(
            "--json",
            action="store_true",
            default=False,
            help="Emit machine-readable JSON on stdout",
        )

    p_run = sub.add_parser("run", help="Run from a prompt")
    add_prompt(p_run, required=True)
    add_advanced(p_run)

    p_preview = sub.add_parser("preview", help="Preview from a prompt")
    add_prompt(p_preview, required=True)
    add_advanced(p_preview)

    p_steer = sub.add_parser("steer", help="Steer a run with a prompt")
    add_prompt(p_steer, required=True)
    add_advanced(p_steer)

    p_status = sub.add_parser("status", help="Status of sole/specified run")
    p_status.add_argument("prompt", nargs="?", default=None, help=argparse.SUPPRESS)
    add_advanced(p_status)

    p_follow = sub.add_parser("follow", help="Follow sole/specified run")
    p_follow.add_argument("prompt", nargs="?", default=None, help=argparse.SUPPRESS)
    add_advanced(p_follow)

    return parser


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in ("backend", "model", "timeout"):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    return overrides


def _prompt_value(args: argparse.Namespace) -> Optional[str]:
    prompt = getattr(args, "prompt", None)
    if prompt is None:
        return None
    if isinstance(prompt, list):
        return prompt[0] if prompt else None
    return prompt


def dispatch(
    argv: Optional[Sequence[str]] = None,
    *,
    facade: Optional[AgentSupervisorFacade] = None,
) -> int:
    """Parse argv, invoke facade, print result, return exit code."""
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return int(code) if isinstance(code, int) else EXIT_INVALID

    fac = facade if facade is not None else get_facade()
    overrides = _overrides_from_args(args)
    command = args.command
    prompt = _prompt_value(args)
    run_id = getattr(args, "run_id", None)

    if command == "run":
        result = fac.run(prompt, **overrides)  # type: ignore[arg-type]
    elif command == "preview":
        result = fac.preview(prompt, **overrides)  # type: ignore[arg-type]
    elif command == "steer":
        result = fac.steer(prompt, run_id=run_id, **overrides)  # type: ignore[arg-type]
    elif command == "status":
        result = fac.status(run_id=run_id, **overrides)
    elif command == "follow":
        result = fac.follow(run_id=run_id, **overrides)
    else:
        result = {
            "outcome": "invalid",
            "exit_code": EXIT_INVALID,
            "error": f"unknown command: {command}",
        }

    as_json = bool(getattr(args, "json", False))
    if as_json:
        print(json.dumps(result, sort_keys=True))
    else:
        outcome = result.get("outcome", "failed")
        msg = result.get("message") or result.get("error") or outcome
        rid = result.get("run_id")
        if rid:
            print(f"[{outcome}] {msg} (run_id={rid})")
        else:
            print(f"[{outcome}] {msg}")

    return int(result.get("exit_code", outcome_to_exit_code(result.get("outcome", "failed"))))


def main(argv: Optional[Sequence[str]] = None) -> int:
    return dispatch(argv)


if __name__ == "__main__":
    sys.exit(main())
