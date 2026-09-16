"""Keep the overlay import root first when a sealed checkout inserts itself."""

from __future__ import annotations

import sys
from pathlib import Path


def overlay_root() -> str:
    return str(Path(__file__).resolve().parents[3])


class OverlayFirstPath(list):
    """sys.path stand-in that keeps the overlay ahead of sealed checkouts."""

    def __init__(self, values, *, overlay: str) -> None:
        super().__init__(values)
        self._overlay = str(Path(overlay).resolve())
        self._promote()

    def insert(self, index, path):  # type: ignore[no-untyped-def]
        super().insert(index, path)
        self._promote()

    def append(self, path):  # type: ignore[no-untyped-def]
        super().append(path)
        self._promote()

    def extend(self, paths):  # type: ignore[no-untyped-def]
        super().extend(paths)
        self._promote()

    def _promote(self) -> None:
        overlay = self._overlay
        while overlay in self:
            list.remove(self, overlay)
        list.insert(self, 0, overlay)


def pin_overlay_sys_path(overlay: str | None = None) -> str:
    """Install OverlayFirstPath on sys.path. Safe to call more than once."""
    root = str(Path(overlay or overlay_root()).resolve())
    current = sys.path
    if isinstance(current, OverlayFirstPath) and current._overlay == root:
        current._promote()
        return root
    sys.path = OverlayFirstPath(current, overlay=root)
    return root
