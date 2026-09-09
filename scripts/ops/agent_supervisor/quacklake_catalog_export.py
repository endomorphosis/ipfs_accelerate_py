#!/usr/bin/env python3
"""Export admitted native fleet observations to a configured QuackLake catalog."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.federation.quacklake_catalog import main

if __name__ == "__main__":
    raise SystemExit(main())
