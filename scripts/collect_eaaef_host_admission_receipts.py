#!/usr/bin/env python3
"""Collect host-controlled EAAEF S-epic admission receipts.

Does not start a supervisor or configured-board launch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the intentionally argument-free collection contract."""

    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def _collect_host_admission() -> dict[str, object]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
        collect_and_write,
    )

    return collect_and_write()


def main(argv: list[str] | None = None) -> int:
    # Inspection must finish before repository imports or authority/filesystem work.
    _parse_args(argv)
    result = _collect_host_admission()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
