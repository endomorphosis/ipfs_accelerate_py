"""Dependency-free schema identities shared by EAAEF execution adapters.

These constants are protocol identities.  Keeping them in a leaf module lets
generic database-authoritative execution classify typed receipts without
loading the separately qualified EAAEF gateway and native-cryptography stack.
"""

from __future__ import annotations

from typing import Final

EAAEF_CONTAINER_VALIDATION_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-container-validation-evidence@1"
)
EAAEF_IDEMPOTENT_RESERVATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-idempotent-operation-reservation@1"
)

__all__ = [
    "EAAEF_CONTAINER_VALIDATION_EVIDENCE_SCHEMA",
    "EAAEF_IDEMPOTENT_RESERVATION_SCHEMA",
]
