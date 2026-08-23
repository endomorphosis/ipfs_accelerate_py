"""Preserve ``python -m ipfs_accelerate_py.cli`` as the historical host."""

from __future__ import annotations

import sys

from ipfs_accelerate_py.cli import main

if __name__ == "__main__":
    sys.exit(main() or 0)
