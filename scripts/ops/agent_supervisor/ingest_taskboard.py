#!/usr/bin/env python3
"""Ingest a supervisor taskboard into DuckDB/Quack and emit bounded context.

Subcommands:

* ``repair`` — write deterministically repaired board text
* ``validate`` — fail-closed malformation check
* ``ingest`` — repair/validate and materialize into DuckDB
* ``context`` — query a compact DuckDB view for model context

Does not start configured-board-launch or a Quack server. Network INSTALL of
Quack is refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_repo_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest_taskboard",
        description="Repair, validate, and ingest taskboards into DuckDB/Quack.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    repair = sub.add_parser("repair", help="Repair malformed taskboard text")
    repair.add_argument("--input", required=True, help="Source JSON or Markdown path")
    repair.add_argument("--output", required=True, help="Repaired output path")
    repair.add_argument(
        "--kind",
        default="auto",
        choices=("auto", "json", "markdown"),
        help="Board text kind (default: auto)",
    )

    validate = sub.add_parser("validate", help="Validate a taskboard")
    validate.add_argument("--board", required=True, help="JSON or Markdown board path")
    validate.add_argument(
        "--no-repair",
        action="store_true",
        help="Parse without applying text repairs",
    )

    ingest = sub.add_parser("ingest", help="Ingest a taskboard into DuckDB")
    ingest.add_argument("--board", required=True, help="JSON or Markdown board path")
    ingest.add_argument("--store", required=True, help="DuckDB store path")
    ingest.add_argument(
        "--no-repair",
        action="store_true",
        help="Ingest without applying text repairs",
    )
    ingest.add_argument(
        "--allow-without-quack",
        action="store_true",
        help="Ingest even if the local Quack health check fails",
    )
    ingest.add_argument(
        "--replace-existing",
        action="store_true",
        help="Re-upsert tasks that already exist in the store",
    )

    context = sub.add_parser("context", help="Emit a bounded DuckDB context view")
    context.add_argument("--store", required=True, help="DuckDB store path")
    context.add_argument("--task-id", default="", help="Single task id")
    context.add_argument(
        "--all",
        action="store_true",
        help="Include non-ready tasks (still bounded)",
    )
    context.add_argument(
        "--max-bytes",
        type=int,
        default=4096,
        help="Hard byte budget for the context view",
    )
    return parser


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ensure_repo_path()
    from ipfs_accelerate_py.agent_supervisor.task_sources.taskboard_ingest import (
        TaskboardIngestError,
        ingest_taskboard,
        load_taskboard,
        repair_malformed_text,
        taskboard_context_view,
        write_repaired_text,
    )

    try:
        if args.command == "repair":
            raw = Path(args.input).read_bytes()
            result = write_repaired_text(args.output, raw, kind=args.kind)
            payload = result.to_dict()
            payload["input"] = args.input
            payload["output"] = args.output
            _print(payload)
            return EXIT_SUCCESS
        if args.command == "validate":
            board = load_taskboard(args.board, repair=not args.no_repair)
            _print(
                {
                    "valid": True,
                    "source_path": args.board,
                    "task_count": len(board.get("tasks") or ()),
                    "goal_count": len(board.get("goals") or ()),
                    "repair": board.get("_repair") or {},
                }
            )
            return EXIT_SUCCESS
        if args.command == "ingest":
            payload = ingest_taskboard(
                board_path=args.board,
                store_path=args.store,
                repair=not args.no_repair,
                require_quack=not args.allow_without_quack,
                replace_existing=args.replace_existing,
            )
            _print(payload)
            return EXIT_SUCCESS
        if args.command == "context":
            payload = taskboard_context_view(
                args.store,
                ready_only=not args.all,
                task_id=str(args.task_id or ""),
                max_bytes=int(args.max_bytes),
            )
            _print(payload)
            return EXIT_SUCCESS
    except TaskboardIngestError as exc:
        _print({"valid": False, "error": str(exc)})
        return EXIT_FAILURE
    except FileNotFoundError as exc:
        _print({"valid": False, "error": str(exc)})
        return EXIT_FAILURE
    parser.print_help()
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
