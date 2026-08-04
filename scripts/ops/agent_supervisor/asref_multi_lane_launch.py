#!/usr/bin/env python3
"""ASREF-G100 multi-lane launch recipe for agent_supervisor module refactor.

Autonomous supervisor execution with exact **Grok 4.5** and quota-only Codex
``gpt-5.6-terra`` medium fallback against the ASREF objective heap and board.
Autonomous supervisor execution defaults to pinned **Grok 4.5**, with
``gpt-5.6-terra`` at medium only after verified Grok quota exhaustion.

This script is the operator-facing entry under ``scripts/ops/agent_supervisor``
required by goal **ASREF-G100**. It delegates preflight/launch to
``scripts/ops/asref_module_refactor_supervisor.py`` while keeping protected-path
safety, bundle isolation, and provider selection explicit.

Related evidence goals covered by the same program surface: **ASREF-G010**
(inventory / move map) and **ASREF-G090** (public API package README / cutover).

Examples
--------
Preflight only::

    python scripts/ops/agent_supervisor/asref_multi_lane_launch.py preflight

Dry-run multi-lane launch with the default quota-routed policy::

    python scripts/ops/agent_supervisor/asref_multi_lane_launch.py launch \\
        --lanes 4 --implementation-provider auto --dry-run

Print the structured recipe JSON (no process start)::

    python scripts/ops/agent_supervisor/asref_multi_lane_launch.py recipe
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.asref_layout_evidence import (  # noqa: E402
    ASREF_DEFAULT_IMPLEMENTATION_PROVIDER,
    ASREF_DEFAULT_NAMESPACE,
    ASREF_G010,
    ASREF_G090,
    ASREF_G100,
    ASREF_IMPLEMENTATION_PROVIDER_ENV,
    ASREF_MERGE_BRANCH,
    ASREF_PROTECTED_PATHS,
    ASREF_SUPERVISOR_LAUNCH_SCRIPT,
    asref_g100_launch_recipe,
)


GOAL_ID = ASREF_G100
RELATED_EVIDENCE = (ASREF_G010, ASREF_G090, ASREF_G100)


def _load_asref_supervisor() -> ModuleType:
    """Load asref_module_refactor_supervisor without requiring scripts.ops package."""

    path = REPO_ROOT / ASREF_SUPERVISOR_LAUNCH_SCRIPT
    if not path.is_file():
        raise FileNotFoundError(f"missing ASREF supervisor launcher: {path}")
    spec = importlib.util.spec_from_file_location(
        "asref_module_refactor_supervisor",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load ASREF supervisor from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _print_recipe(
    *,
    lanes: int,
    provider: str,
    namespace: str,
    enable_objective_refill: bool,
    dry_run: bool,
) -> int:
    recipe = asref_g100_launch_recipe(
        lanes=lanes,
        provider=provider,
        namespace=namespace,
        enable_objective_refill=enable_objective_refill,
        dry_run=dry_run,
    )
    recipe["entry"] = "scripts/ops/agent_supervisor/asref_multi_lane_launch.py"
    recipe["related_evidence_goals"] = list(RELATED_EVIDENCE)
    print(json.dumps(recipe, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry for ASREF-G100 multi-lane autonomous execution."""

    parser = argparse.ArgumentParser(
        description=(
            "ASREF-G100: multi-lane implementation supervisor launch for the "
            "agent_supervisor module refactor (sealed Grok 4.5 / quota-only "
            "Terra fallback)"
            "agent_supervisor module refactor (Grok 4.5, quota-only "
            "Terra/medium fallback)"
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--lanes", type=int, default=4)
        p.add_argument("--namespace", default=ASREF_DEFAULT_NAMESPACE)
        p.add_argument("--merge-branch", default=ASREF_MERGE_BRANCH)
        p.add_argument(
            "--implementation-provider",
            default=os.environ.get(
                ASREF_IMPLEMENTATION_PROVIDER_ENV,
                ASREF_DEFAULT_IMPLEMENTATION_PROVIDER,
            )
            or ASREF_DEFAULT_IMPLEMENTATION_PROVIDER,
            help=(
                "Compatible Grok primary alias for child daemons "
                f"(default: {ASREF_DEFAULT_IMPLEMENTATION_PROVIDER}; "
                "also set via "
                f"{ASREF_IMPLEMENTATION_PROVIDER_ENV}); incompatible routes "
                "fail closed"
            ),
        )
        p.add_argument(
            "--enable-objective-refill",
            action="store_true",
            default=True,
            help="Enable objective-heap refill (default: on for ASREF-G100)",
        )
        p.add_argument(
            "--no-objective-refill",
            action="store_true",
            help="Disable objective-heap refill",
        )
        p.add_argument("--force", action="store_true")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--foreground", action="store_true")
        p.add_argument(
            "--duration-seconds",
            type=float,
            default=float("inf"),
        )

    for name in ("preflight", "launch", "recipe"):
        p = sub.add_parser(name)
        add_common(p)

    args = parser.parse_args(list(argv) if argv is not None else None)
    provider = str(args.implementation_provider or "").strip() or (
        ASREF_DEFAULT_IMPLEMENTATION_PROVIDER
    )
    enable_refill = bool(args.enable_objective_refill) and not bool(
        args.no_objective_refill
    )

    banner = {
        "schema": "ipfs_accelerate_py.asref.g100_entry.v1",
        "goal_id": GOAL_ID,
        "related_evidence_goals": list(RELATED_EVIDENCE),
        "command": args.command,
        "implementation_provider": provider,
        "protected_paths": list(ASREF_PROTECTED_PATHS),
        "merge_branch": args.merge_branch,
        "lanes": args.lanes,
        "namespace": args.namespace,
        "no_shim_rule": (
            "Workers must follow each task Validation line and never leave "
            "long-lived flat re-export stubs after a package move."
        ),
    }
    print(json.dumps(banner, indent=2, sort_keys=True))

    if args.command == "recipe":
        return _print_recipe(
            lanes=args.lanes,
            provider=provider,
            namespace=args.namespace,
            enable_objective_refill=enable_refill,
            dry_run=True,
        )

    # Propagate provider before delegating so child preflight/launch see it.
    os.environ[ASREF_IMPLEMENTATION_PROVIDER_ENV] = provider

    delegated = [
        args.command,
        "--lanes",
        str(args.lanes),
        "--namespace",
        args.namespace,
        "--merge-branch",
        args.merge_branch,
        "--implementation-provider",
        provider,
        "--duration-seconds",
        str(args.duration_seconds),
    ]
    if enable_refill and args.command == "launch":
        delegated.append("--enable-objective-refill")
    if args.force:
        delegated.append("--force")
    if args.dry_run:
        delegated.append("--dry-run")
    if args.foreground:
        delegated.append("--foreground")

    asref_supervisor = _load_asref_supervisor()
    return int(asref_supervisor.main(delegated))


if __name__ == "__main__":
    raise SystemExit(main())
