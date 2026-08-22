"""Backup, restore, and ambiguity recovery (EAAEF-096)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final


RECOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-control-recovery@1"
)


class RecoveryError(ValueError):
    """Recovery would accept a stale write."""


def recover(
    *,
    current_epoch: int,
    backup_epoch: int,
    duplicate: bool,
    ducklake_available: bool,
) -> Mapping[str, Any]:
    if int(backup_epoch) > int(current_epoch):
        raise RecoveryError("backup from a future epoch is stale/ambiguous")
    if duplicate:
        raise RecoveryError("duplicate transaction cannot be accepted twice")
    return MappingProxyType(
        {
            "schema": RECOVERY_SCHEMA,
            "epoch": int(current_epoch),
            "backup_epoch": int(backup_epoch),
            "ducklake_available": bool(ducklake_available),
            "accepted_stale_write": False,
        }
    )
