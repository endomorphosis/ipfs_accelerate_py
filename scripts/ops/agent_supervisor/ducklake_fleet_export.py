#!/usr/bin/env python3
"""Project live Quack fleet observations into a dedicated local DuckLake."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.federation.ducklake_fleet import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
