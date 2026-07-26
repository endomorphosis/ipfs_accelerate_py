#!/usr/bin/env python3
"""CLI wrapper for the bounded Leanstral MCP++ P2P topology collector."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.mcplusplus_module.leanstral_topology_collector import main


if __name__ == "__main__":
    raise SystemExit(main())
