"""Compatibility re-export of the ASE3-026 activation gate (lives in validation).

Entrypoints may import lower packages; validation must never import entrypoints.
"""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.validation.protected_runtime_activation import *  # noqa: F403
from ipfs_accelerate_py.agent_supervisor.validation.protected_runtime_activation import (
    __all__ as _ALL,
)

__all__ = list(_ALL)
