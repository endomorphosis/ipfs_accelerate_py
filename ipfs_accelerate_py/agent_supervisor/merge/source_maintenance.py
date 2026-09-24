"""Cooperative native dispatch drain for a qualified source-maintenance owner.

This lease closes new intent admission in compatible daemons. It does not prove
an idle board, settle existing intents, acquire merge/source leases, stop an
owner, or authorize a source transition. Callers must independently qualify the
loaded consumers and retain their native custody and process-population gates.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Callable, Iterator

from .checkout_lock import (
    acquire_checkout_mutation_lease,
    board_scoped_protected_path_maintenance_lock_path,
    checkout_lock_metadata,
    read_checkout_mutation_lease,
    release_checkout_mutation_lease,
    serialized_lock_update,
)
from .worktree_lifecycle import read_process_birth


def _mirror_dispatch_drain_gate(board_namespace: str) -> None:
    """Record a drain that is still held. The path and process identity are not stored."""

    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        record_ref = str(board_namespace or "dispatch-drain")
        mirror_work_record(
            catalog_kind="metadata",
            record_kind="dispatch_drain_gate",
            record_ref=record_ref,
            subject_kind="record_cid",
            subject_ref=record_ref,
        )
    except Exception:
        pass


@contextmanager
def dispatch_drain_lease(
    repo_root: Path,
    board_namespace: str,
    *,
    admission_gate: Callable[[], None],
    timeout_seconds: float = 5.0,
) -> Iterator[Callable[[], None]]:
    """Publish explicit drain under the same native gate used by dispatch.

The mandatory admission gate must raise unless this reader/controller has
current native maintenance custody. No incumbent is reclaimed, even if its
record appears stale. The yielded gate rechecks the exact lease and custody;
invoke it at each later lease acquisition and native mutation boundary.
Existing providers and callback workspaces are left to their native owners.
"""
    if not callable(admission_gate):
        raise TypeError("native maintenance admission gate is required")
    if not isinstance(board_namespace, str) or not board_namespace.strip():
        raise ValueError("exact board namespace is required")
    if not 0 < timeout_seconds <= 60:
        raise ValueError("drain coordination timeout must be in (0, 60]")
    admission_gate()
    owner = read_process_birth(os.getpid())
    if owner is None or owner.start_time_ticks <= 0 or not owner.boot_id:
        raise RuntimeError("dispatch drain owner birth is unverified")

    def require_self() -> None:
        if os.getpid() != owner.pid or read_process_birth(os.getpid()) != owner:
            raise RuntimeError("dispatch drain owner birth changed")

    lock_path = board_scoped_protected_path_maintenance_lock_path(
        Path(repo_root), board_namespace,
    )
    metadata = checkout_lock_metadata(
        kind="implementation-protected-maintenance",
        repo_root=Path(repo_root),
        owner_script="",
        extra={"operation": "source-maintenance-dispatch-drain",
               "dispatch_admission": "drain", "board_namespace": board_namespace,
               "owner_process_birth": owner.to_dict()},
    )
    if not metadata["repository_id"]:
        raise RuntimeError("drain repository identity is unverified")
    # A zero-timeout acquisition with owner_active=True never reclaims an
    # incumbent and never nests a blocking stale-owner update under this gate.
    with serialized_lock_update(lock_path, timeout_seconds=timeout_seconds):
        require_self()
        admission_gate()
        lease, reason, _, _ = acquire_checkout_mutation_lease(
            lock_path, metadata, owner_active=lambda _: True,
        )
    if lease is None:
        raise RuntimeError("dispatch drain contended: " + reason)
    active = True

    def gate() -> None:
        if not active:
            raise RuntimeError("dispatch drain is no longer held")
        require_self()
        admission_gate()
        with serialized_lock_update(lock_path, timeout_seconds=timeout_seconds):
            if read_checkout_mutation_lease(lock_path) != lease:
                raise RuntimeError("dispatch drain lease changed")
            _mirror_dispatch_drain_gate(board_namespace)

    try:
        gate()
        yield gate
    finally:
        active = False
        require_self()
        # Compare complete metadata before using the native identity-bound
        # release. A replaced or edited record is retained for native review.
        with serialized_lock_update(lock_path, timeout_seconds=timeout_seconds):
            current = read_checkout_mutation_lease(lock_path)
            if current != lease:
                raise RuntimeError("dispatch drain release binding changed")
        if not release_checkout_mutation_lease(
            lease, timeout_seconds=timeout_seconds, require_exact_metadata=True,
        ):
            raise RuntimeError("dispatch drain release unverified")
