#!/usr/bin/env python3
"""Thin ops facade for IPFS Kit VFS symbolic assurance.

Contains only argument parsing, config bootstrap, and delegation into the
integrations adapter.  Cold import and ``--help`` start no process, open no
database, access no network/storage, and import no optional provider.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

# Closed subcommand names only; integration owns execution.
_SUBCOMMANDS = (
    "inventory",
    "contracts",
    "differential",
    "parity",
    "benchmark",
    "pilot",
    "rollout",
    "verify",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ipfs_kit_vfs_symbolic_assurance",
        description=(
            "Thin facade for the IPFS Kit VFS symbolic-assurance job. "
            "Validates the locked profile and delegates to generic engines."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to ipfs_kit_vfs_symbolic_assurance.json (default: config/…)",
    )
    parser.add_argument(
        "--checkout-root",
        default=None,
        help="Repository checkout root (default: auto-detect)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON on success",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inventory = sub.add_parser("inventory", help="Run surface inventory via profile")
    inventory.add_argument(
        "--relative-root",
        default=".",
        help="Safe relative root under the checkout allowlist",
    )

    for name, help_text in (
        ("contracts", "Project operation/invariant/error mappings"),
        ("differential", "Differential harness adapter bootstrap"),
        ("parity", "Interface parity adapter bootstrap"),
        ("benchmark", "Symbolic efficiency benchmark adapter bootstrap"),
        ("pilot", "Symbolic assurance pilot adapter bootstrap"),
        ("verify", "Verify adversarial gates and shadow rollout"),
    ):
        sub.add_parser(name, help=help_text)

    rollout = sub.add_parser("rollout", help="Evaluate adversarial rollout gates")
    rollout.add_argument(
        "--mode",
        default="shadow",
        choices=("off", "shadow", "assist", "automatic"),
        help="Desired rollout mode (effective mode remains shadow-default)",
    )
    return parser


def _bootstrap_integration():
    """Lazy-import the integration only after argument validation."""

    from ipfs_accelerate_py.agent_supervisor.integrations import (  # noqa: WPS433
        ipfs_kit_vfs_assurance as integration,
    )

    return integration


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return EXIT_SUCCESS
        return int(code) if isinstance(code, int) else EXIT_USAGE

    if args.command not in _SUBCOMMANDS:
        print(f"unknown command: {args.command}", file=sys.stderr)
        return EXIT_USAGE

    try:
        integration = _bootstrap_integration()
        checkout = (
            Path(args.checkout_root).resolve()
            if args.checkout_root
            else _repo_root()
        )
        config_path = (
            Path(args.config).resolve()
            if args.config
            else integration.default_config_path(checkout_root=checkout)
        )
        config = integration.load_assurance_config(
            config_path, checkout_root=checkout
        )
        kwargs = {
            "config": config,
            "checkout_root": checkout,
        }
        if args.command == "inventory":
            kwargs["relative_root"] = getattr(args, "relative_root", ".")
        if args.command == "rollout":
            kwargs["desired_mode"] = getattr(args, "mode", "shadow")
        result = integration.dispatch(args.command, **kwargs)
    except Exception as exc:  # noqa: BLE001 - facade maps all failures
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    if getattr(args, "json", False) or True:
        # Always emit JSON for machine consumers; human text is the same payload.
        sys.stdout.write(
            json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False)
            + "\n"
        )
    if args.command == "verify" and not result.get("verified", False):
        return EXIT_FAILURE
    if args.command == "rollout" and result.get("automatic_mutation_enabled") is True:
        return EXIT_FAILURE
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
