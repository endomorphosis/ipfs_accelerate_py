"""Compatibility re-export of neutral contracts (ASE3-029).

Preserves public and private module attributes so monkeypatched tests and
exact object identity continue to work against the compatibility path.
"""
from __future__ import annotations

from ..contracts import execution as _impl

for _name in dir(_impl):
    if _name in {"__name__", "__package__", "__loader__", "__spec__", "__file__", "__cached__", "__builtins__"}:
        continue
    globals()[_name] = getattr(_impl, _name)

__all__ = [name for name in dir(_impl) if not name.startswith("_")]
del _name
del _impl
