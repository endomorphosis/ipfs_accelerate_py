#!/usr/bin/env python3
"""Collect host-controlled EAAEF S-epic admission receipts.

Does not start a supervisor or configured-board launch.
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
)


def main() -> int:
    result = collect_and_write()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
