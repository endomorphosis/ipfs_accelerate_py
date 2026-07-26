"""Compatibility alias for :mod:`ipfs_accelerate_py.embeddings_router`."""

from __future__ import annotations

import sys as _sys

from . import embeddings_router as _canonical_router


_sys.modules[__name__] = _canonical_router
