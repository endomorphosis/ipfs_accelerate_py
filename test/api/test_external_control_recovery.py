"""EAAEF-096: recovery never accepts stale or duplicate writes."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.external_control_recovery import (
    RecoveryError,
    recover,
)


def test_recovery_refuses_future_backup_and_duplicates() -> None:
    report = recover(current_epoch=8, backup_epoch=7, duplicate=False, ducklake_available=False)
    assert report["accepted_stale_write"] is False
    with pytest.raises(RecoveryError, match="stale"):
        recover(current_epoch=8, backup_epoch=9, duplicate=False, ducklake_available=True)
    with pytest.raises(RecoveryError, match="duplicate"):
        recover(current_epoch=8, backup_epoch=8, duplicate=True, ducklake_available=True)
