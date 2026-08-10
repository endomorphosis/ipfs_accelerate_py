"""Narrow operator CLI for one prompt-v3 protected phase at a time."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Mapping, Sequence

from ..core.protected_acceptance_contracts import EvidenceHandle

CLI_COMMANDS = ("inspect", "readiness", "prepare-q", "advance-one-phase", "birth")


def _evidence(value: str) -> EvidenceHandle:
    if type(value) is not str or len(value.encode("utf-8")) > 4096:
        raise argparse.ArgumentTypeError("evidence handle exceeds CLI bound")
    try:
        payload = json.loads(value)
        return EvidenceHandle.from_mapping(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(
            "evidence must be one strict bounded handle"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prompt-v3-protected-transition")
    subcommands = parser.add_subparsers(dest="command", required=True)
    inspect = subcommands.add_parser(
        "inspect", help="inspect immutable transition capability"
    )
    inspect.add_argument("--evidence", action="append", type=_evidence, default=[])
    readiness = subcommands.add_parser(
        "readiness",
        help="report fail-closed Q construction readiness without creating Q",
    )
    readiness.add_argument(
        "--repo-root",
        default=".",
        help="repository root to assess (default: cwd)",
    )
    prepare = subcommands.add_parser("prepare-q", help="perform exactly prepare-q")
    prepare.add_argument(
        "--evidence",
        action="append",
        type=_evidence,
        default=[],
        help="optional bounded evidence handles (compat)",
    )
    prepare.add_argument(
        "--repo-root",
        default=".",
        help="repository root (default: cwd)",
    )
    prepare.add_argument(
        "--target-ref",
        default="refs/heads/main",
        help="protected target ref to publish (default: refs/heads/main)",
    )
    prepare.add_argument(
        "--dry-run",
        action="store_true",
        help="build and validate the Q candidate without publishing",
    )
    advance = subcommands.add_parser(
        "advance-one-phase", help="perform exactly advance-one-phase"
    )
    advance.add_argument(
        "--evidence",
        action="append",
        type=_evidence,
        default=[],
        help="optional bounded evidence handles (compat)",
    )
    advance.add_argument(
        "--repo-root",
        default=".",
        help="repository root (default: cwd)",
    )
    advance.add_argument(
        "--target-ref",
        default="refs/heads/main",
        help="protected target ref to publish (default: refs/heads/main)",
    )
    advance.add_argument(
        "--phase",
        default="R",
        choices=("R", "P019", "A019", "A030", "P031"),
        help="phase to advance (R through P031 auto-composed)",
    )
    advance.add_argument(
        "--dry-run",
        action="store_true",
        help="build and validate the phase candidate without publishing",
    )
    birth = subcommands.add_parser("birth", help="perform exactly birth")
    birth.add_argument(
        "--evidence",
        action="append",
        type=_evidence,
        required=True,
        help="inline canonical bounded evidence-handle JSON; repeatable",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    handlers: Mapping[str, Callable[[tuple[EvidenceHandle, ...]], Mapping[str, Any]]]
    | None = None,
) -> int:
    arguments = build_parser().parse_args(argv)
    evidence = tuple(getattr(arguments, "evidence", None) or ())
    if any(not isinstance(item, EvidenceHandle) for item in evidence):
        raise TypeError("CLI evidence was not fully transcoded")
    if handlers is None:
        if arguments.command == "readiness":
            from .protected_acceptance_q_readiness import (
                assess_prompt_v3_q_construction_readiness,
            )

            result = assess_prompt_v3_q_construction_readiness(
                getattr(arguments, "repo_root", ".")
            )
        elif arguments.command == "prepare-q":
            from .protected_acceptance_prepare_q import prepare_prompt_v3_q

            result = prepare_prompt_v3_q(
                repo_root=getattr(arguments, "repo_root", "."),
                target_ref=getattr(arguments, "target_ref", "refs/heads/main"),
                dry_run=bool(getattr(arguments, "dry_run", False)),
                publish=not bool(getattr(arguments, "dry_run", False)),
            )
        elif arguments.command == "advance-one-phase":
            phase = str(getattr(arguments, "phase", "R") or "R")
            if phase == "R":
                from .protected_acceptance_advance_r import advance_prompt_v3_r

                result = advance_prompt_v3_r(
                    repo_root=getattr(arguments, "repo_root", "."),
                    target_ref=getattr(arguments, "target_ref", "refs/heads/main"),
                    dry_run=bool(getattr(arguments, "dry_run", False)),
                    publish=not bool(getattr(arguments, "dry_run", False)),
                )
            elif phase == "P019":
                from .protected_acceptance_advance_p019 import advance_prompt_v3_p019

                result = advance_prompt_v3_p019(
                    repo_root=getattr(arguments, "repo_root", "."),
                    target_ref=getattr(arguments, "target_ref", "refs/heads/main"),
                    dry_run=bool(getattr(arguments, "dry_run", False)),
                    publish=not bool(getattr(arguments, "dry_run", False)),
                )
            elif phase == "A019":
                from .protected_acceptance_advance_a019 import advance_prompt_v3_a019

                result = advance_prompt_v3_a019(
                    repo_root=getattr(arguments, "repo_root", "."),
                    target_ref=getattr(arguments, "target_ref", "refs/heads/main"),
                    dry_run=bool(getattr(arguments, "dry_run", False)),
                    publish=not bool(getattr(arguments, "dry_run", False)),
                )
            elif phase == "A030":
                from .protected_acceptance_advance_a030 import advance_prompt_v3_a030

                result = advance_prompt_v3_a030(
                    repo_root=getattr(arguments, "repo_root", "."),
                    target_ref=getattr(arguments, "target_ref", "refs/heads/main"),
                    dry_run=bool(getattr(arguments, "dry_run", False)),
                    publish=not bool(getattr(arguments, "dry_run", False)),
                )
            elif phase == "P031":
                from .protected_acceptance_advance_p031 import advance_prompt_v3_p031

                result = advance_prompt_v3_p031(
                    repo_root=getattr(arguments, "repo_root", "."),
                    target_ref=getattr(arguments, "target_ref", "refs/heads/main"),
                    dry_run=bool(getattr(arguments, "dry_run", False)),
                    publish=not bool(getattr(arguments, "dry_run", False)),
                )
            else:
                build_parser().error(
                    f"advance-one-phase does not yet auto-compose phase {phase}"
                )
        elif arguments.command != "inspect":
            build_parser().error(
                "production command requires injected protected composition"
            )
        else:
            result = {
                "commands": list(CLI_COMMANDS),
                "run_all": False,
                "raw_key_input": False,
                "evidence_handles": [item.to_dict() for item in evidence],
            }
    else:
        if set(handlers) - set(CLI_COMMANDS):
            raise ValueError("CLI handlers contain unsupported commands")
        handler = handlers.get(arguments.command)
        if handler is None:
            raise ValueError("CLI command has no injected protected handler")
        result = handler(evidence)
        if not isinstance(result, Mapping):
            raise TypeError("CLI handler must return a mapping")
    sys.stdout.write(
        json.dumps(
            dict(result), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ("CLI_COMMANDS", "build_parser", "main")
