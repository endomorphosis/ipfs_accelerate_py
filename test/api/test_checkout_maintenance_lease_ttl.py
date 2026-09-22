from __future__ import annotations

import json
import time

from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    CheckoutMaintenanceLease,
)


def test_expired_checkout_lease_is_stolen(tmp_path) -> None:
    lock_path = tmp_path / "implementation-main-merge.lock"
    first = CheckoutMaintenanceLease(
        lock_path=lock_path,
        metadata={"kind": "checkout-maintenance", "pid": 1},
        max_hold_seconds=30.0,
    )
    acquired, _ = first.try_acquire(owner_is_active=lambda _meta: True)
    assert acquired
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["expires_at"] = time.time() - 1
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    second = CheckoutMaintenanceLease(
        lock_path=lock_path,
        metadata={"kind": "checkout-maintenance", "pid": 2},
        max_hold_seconds=30.0,
    )
    stolen, view = second.try_acquire(owner_is_active=lambda _meta: True)
    assert stolen
    assert view["reason"] == "checkout_maintenance_lease_acquired"
    second.release()
