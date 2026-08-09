"""Narrow operator CLI for one prompt-v3 protected phase at a time."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Mapping, Sequence

from ..core.protected_acceptance_contracts import EvidenceHandle

CLI_COMMANDS = ("inspect", "prepare-q", "advance-one-phase", "birth")


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
    for name in ("prepare-q", "advance-one-phase", "birth"):
        command = subcommands.add_parser(name, help=f"perform exactly {name}")
        command.add_argument(
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
    evidence = tuple(arguments.evidence)
    if any(not isinstance(item, EvidenceHandle) for item in evidence):
        raise TypeError("CLI evidence was not fully transcoded")
    if handlers is None:
        if arguments.command != "inspect":
            build_parser().error(
                "production command requires injected protected composition"
            )
        result: Mapping[str, Any] = {
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
