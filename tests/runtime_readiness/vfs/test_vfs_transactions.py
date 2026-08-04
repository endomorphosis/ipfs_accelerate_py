"""Root-level implied validation mirror for KITA-008 VFS transactions.

The authoritative suite lives under the nested ``ipfs_kit_py`` package tree.
This module asserts the declared transaction, snapshot, and nested tests exist
from a superproject checkout and cover the acceptance surface.
"""

from __future__ import annotations

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
NESTED_ROOT = WORKSPACE / "ipfs_kit_py"
TRANSACTIONS = NESTED_ROOT / "ipfs_kit_py" / "core" / "vfs" / "transactions.py"
SNAPSHOTS = NESTED_ROOT / "ipfs_kit_py" / "core" / "vfs" / "snapshots.py"
NESTED_TEST = (
    NESTED_ROOT / "tests" / "runtime_readiness" / "vfs" / "test_vfs_transactions.py"
)

REQUIRED_TRANSACTION_MARKERS = (
    "class VFSTransaction",
    "class VFSTransactionManager",
    "class VFSLockManager",
    "class IsolationLevel",
    "class CancellationDisposition",
    "class ConcurrentScheduleExecutor",
    "VFSTransaction_V1",
    "ordered_lock_paths",
    "PRE_COMMIT_ABORT",
    "POST_COMMIT_RETAINED",
    "KITA-008",
)

REQUIRED_SNAPSHOT_MARKERS = (
    "class VFSVersion",
    "class VFSSnapshot",
    "class VFSSnapshotStore",
    "class VFSVersionHistory",
    "VFSSnapshot_V1",
    "VFSVersion_V1",
    "check_version_precondition",
    "snapshot_cid_for",
    "KITA-008",
)

REQUIRED_TEST_MARKERS = (
    "test_cas_write_rejects_stale_precondition_at_stage",
    "test_snapshot_isolation_prevents_lost_update",
    "test_lock_order_is_utf8_lexicographic",
    "test_cancel_pre_commit_aborts_without_mutation",
    "test_cancel_post_commit_retains_effects",
    "test_snapshot_reproducible_cid",
    "test_snapshot_immutable",
    "test_concurrent_schedule_matches_reference_model",
    "test_concurrent_lock_conflict_is_typed_unsupported_boundary",
)


def test_declared_outputs_present_from_superproject() -> None:
    assert TRANSACTIONS.is_file(), f"missing {TRANSACTIONS}"
    assert SNAPSHOTS.is_file(), f"missing {SNAPSHOTS}"
    assert NESTED_TEST.is_file(), f"missing {NESTED_TEST}"

    txn_text = TRANSACTIONS.read_text(encoding="utf-8")
    for marker in REQUIRED_TRANSACTION_MARKERS:
        assert marker in txn_text, f"missing transaction marker {marker}"

    snap_text = SNAPSHOTS.read_text(encoding="utf-8")
    for marker in REQUIRED_SNAPSHOT_MARKERS:
        assert marker in snap_text, f"missing snapshot marker {marker}"


def test_nested_suite_covers_acceptance_surface() -> None:
    text = NESTED_TEST.read_text(encoding="utf-8")
    for marker in REQUIRED_TEST_MARKERS:
        assert marker in text, f"missing test marker {marker}"
    assert "lost update" in text.lower() or "lost_update" in text
    assert "precondition" in text.lower()
    assert "cancel" in text.lower()
    assert "snapshot" in text.lower()
    assert "lock" in text.lower()
    assert "unsupported" in text.lower()
