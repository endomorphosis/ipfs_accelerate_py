#!/usr/bin/env python3
"""Run a sealed board operator with an origin supervisor overlay first.

SPAR inserts its checkout at sys.path[0], which hides PYTHONPATH. This
launcher keeps the overlay first so clause auto-heal can load without
amending the sealed git HEAD.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _abspath(path: str) -> str:
    return str(Path(path).resolve())


class _OverlayPath(list):
    """Keep the overlay import root ahead of a sealed checkout."""

    def __init__(self, values, *, overlay: str, source_root: str) -> None:
        super().__init__(values)
        self._overlay = overlay
        self._source_root = source_root

    def insert(self, index, path):  # type: ignore[no-untyped-def]
        resolved = _abspath(path) if isinstance(path, str) else path
        if resolved == self._source_root and index == 0:
            return super().insert(1, path)
        return super().insert(index, path)


def install_overlay(overlay: str, source_root: str) -> None:
    overlay = _abspath(overlay)
    source_root = _abspath(source_root)
    current = [str(item) for item in sys.path]
    if overlay in current:
        current.remove(overlay)
    current.insert(0, overlay)
    sys.path = _OverlayPath(current, overlay=overlay, source_root=source_root)  # type: ignore[assignment]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    overlay = os.environ.get("IPFS_ACCELERATE_SUPERVISOR_OVERLAY", "")
    source_root = os.environ.get("IPFS_ACCELERATE_SEALED_SOURCE_ROOT", "")
    while args:
        if args[0] == "--overlay":
            overlay = args[1]
            args = args[2:]
            continue
        if args[0] == "--source-root":
            source_root = args[1]
            args = args[2:]
            continue
        if args[0] == "--":
            args = args[1:]
            break
        break
    if not overlay or not source_root or not args:
        raise SystemExit(
            "usage: sealed_board_supervisor_launch.py "
            "--overlay DIR --source-root DIR -- script.py [args...]"
        )
    install_overlay(overlay, source_root)
    os.chdir(source_root)
    script = args[0]
    sys.argv = args
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
