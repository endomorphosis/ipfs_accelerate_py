"""CID helper compatibility layer."""

from __future__ import annotations

from typing import Any, Optional

from ..ipfs_multiformats import cid_for_obj as _cid_for_obj


def cid_for_obj(value: Any, base: Optional[str] = None) -> str:
    """Delegate to the canonical multiformats CID helper."""
    return _cid_for_obj(value, base=base)
