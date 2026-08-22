#!/usr/bin/env python3
"""Host-controlled produce/sign for EAAEF-185/186/187 admission evidence.

Uses the dedicated EAAEF local-operator profile. Does not start a supervisor
or admit live launch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
    collect_and_write,
    materialize_host_evidence,
)


def main() -> int:
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
