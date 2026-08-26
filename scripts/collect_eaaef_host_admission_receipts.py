#!/usr/bin/env python3
"""Collect host-controlled EAAEF S-epic admission evidence.

The safe default and ``--early-frontier`` atomically publish an immutable,
observation-only EAAEF-180..183 capture outside the checkout. Later host
evidence requires the explicit ``--full-host-evidence`` mode. No mode starts a
supervisor or configured-board launch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse an explicit collection scope; an omitted scope is early-only."""

    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--early-frontier",
        dest="scope",
        action="store_const",
        const="early_frontier",
        default=argparse.SUPPRESS,
        help=(
            "collect and publish an immutable no-go observation for "
            "EAAEF-180 through EAAEF-183 (default)"
        ),
    )
    scope.add_argument(
        "--full-host-evidence",
        dest="scope",
        action="store_const",
        const="full_host_evidence",
        default=argparse.SUPPRESS,
        help="explicitly materialize and collect EAAEF-180 through EAAEF-191",
    )
    return parser.parse_args(argv)


def _collect_host_admission() -> dict[str, object]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
        collect_early_frontier_and_publish_observation,
    )

    return collect_early_frontier_and_publish_observation()


def _collect_full_host_admission() -> dict[str, object]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
        collect_and_write,
    )

    return collect_and_write()


def main(argv: list[str] | None = None) -> int:
    # Inspection must finish before repository imports or authority/filesystem work.
    args = _parse_args(argv)
    scope = str(getattr(args, "scope", "early_frontier"))
    result = (
        _collect_full_host_admission()
        if scope == "full_host_evidence"
        else _collect_host_admission()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
