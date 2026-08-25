#!/usr/bin/env python3
"""Host-controlled produce/sign for EAAEF-185/186/187 admission evidence.

Uses the dedicated EAAEF local-operator profile. Does not start a supervisor
or admit live launch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the intentionally argument-free evidence-issuance contract."""

    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def _host_evidence_entrypoints():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
        collect_and_write,
        materialize_host_evidence,
    )

    return materialize_host_evidence, collect_and_write


def main(argv: list[str] | None = None) -> int:
    # Inspection must finish before repository imports or authority/filesystem work.
    _parse_args(argv)
    materialize_host_evidence, collect_and_write = _host_evidence_entrypoints()
    materialize = materialize_host_evidence()
    collection = collect_and_write()
    print(
        json.dumps(
            {
                "materialize": materialize,
                "collection": collection["decisions"],
                "process_started": False,
                "configured_board_launch": False,
                "live_launch_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
