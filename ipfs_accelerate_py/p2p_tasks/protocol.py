"""Shared constants/helpers for the libp2p task queue protocol.

This module is the canonical definition of the TaskQueue RPC protocol.

Compatibility:
- Accepts token env vars from both ipfs_datasets_py and ipfs_accelerate_py.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


# Keep the protocol id stable for interop with existing nodes.
PROTOCOL_V1 = "/ipfs-datasets/task-queue/1.0.0"


def get_shared_token() -> Optional[str]:
    for name in (
        "IPFS_ACCELERATE_PY_TASK_P2P_TOKEN",
        "IPFS_DATASETS_PY_TASK_P2P_TOKEN",
    ):
        token = os.environ.get(name, "").strip()
        if token:
            return token
    return None


def auth_ok(message: Dict[str, Any]) -> bool:
    expected = get_shared_token()
    if not expected:
        return True
    provided = (message.get("token") or "")
    return isinstance(provided, str) and provided == expected
