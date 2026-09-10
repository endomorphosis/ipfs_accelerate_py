"""UID-global hash lock contention must not look like a drifted sealed executor."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.runtime.hash_pressure import (
    HashingResourceTimeout,
)


def test_hashing_resource_timeout_is_oserror() -> None:
    """Python 3 TimeoutError subclasses OSError.

    Scope-exit receipt validation used to wrap that as
    ``sealed receipt-validation capability differs: scope exit`` and kill the
    exclusive owner when another board held ``/tmp/ipfs-accelerate-heavy-hash-*.lock``.
    """

    assert issubclass(HashingResourceTimeout, TimeoutError)
    assert issubclass(HashingResourceTimeout, OSError)
