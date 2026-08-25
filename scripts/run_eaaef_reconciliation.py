#!/usr/bin/env python3
"""Public wrapper for the fresh typed EAAEF reconciliation lifecycle."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.runtime.eaaef_reconciliation_lifecycle import (
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
