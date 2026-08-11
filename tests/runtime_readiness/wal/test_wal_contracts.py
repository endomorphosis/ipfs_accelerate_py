"""Root-level implied validation mirror for KITA-018 WAL contracts.

The authoritative suite lives under the nested ``ipfs_kit_py`` package tree.
This module asserts the declared contract modules and nested tests exist from a
superproject checkout.
"""

from __future__ import annotations

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
NESTED_ROOT = WORKSPACE / "ipfs_kit_py"
CONTRACTS = NESTED_ROOT / "ipfs_kit_py" / "core" / "wal" / "contracts.py"
COMPATIBILITY = NESTED_ROOT / "ipfs_kit_py" / "core" / "wal" / "compatibility.py"
NESTED_TEST = (
    NESTED_ROOT
    / "tests"
    / "runtime_readiness"
    / "wal"
    / "test_wal_contracts.py"
)

REQUIRED_CLASSES = (
    "WALRecord",
    "WALTransaction",
    "WALSegment",
    "WALCheckpoint",
    "WALRecordIdentity",
    "WALAckRequirements",
    "WALFsyncReceipt",
)

REQUIRED_STATE_LITERALS = (
    '"buffered"',
    '"queued"',
    '"appended"',
    '"committed"',
    '"prepared"',
    '"aborted"',
    '"file_synced"',
    '"parent_synced"',
)

REQUIRED_MARKERS = (
    "WAL_RECORD_SCHEMA",
    "WAL_TRANSACTION_SCHEMA",
    "WAL_SEGMENT_SCHEMA",
    "WAL_CHECKPOINT_SCHEMA",
    "WALAcknowledgementMode",
    "DURABLE_STATES",
    "COMMITTED_STATES",
    "BUFFERED_OR_QUEUED_STATES",
    "ack_requirements_for",
    "requires_file_fsync",
    "requires_parent_directory_fsync",
    "requires_backend_effect",
    "WALUnsafeEncodingError",
    "SecretMaterialError",
    "assert_sequence_monotonic",
)

COMPAT_MARKERS = (
    "map_legacy_status",
    "map_legacy_kind",
    "map_legacy_ack_mode",
    "UNKNOWN_PRESERVED",
    "CompatibilityDisposition",
    "project_legacy_operation",
    "LEGACY_MAPPED",
)


def test_declared_outputs_present_from_superproject() -> None:
    assert CONTRACTS.is_file(), f"missing {CONTRACTS}"
    assert COMPATIBILITY.is_file(), f"missing {COMPATIBILITY}"
    assert NESTED_TEST.is_file(), f"missing {NESTED_TEST}"
    text = CONTRACTS.read_text(encoding="utf-8")
    for name in REQUIRED_CLASSES:
        assert f"class {name}" in text, f"missing class {name}"
    for literal in REQUIRED_STATE_LITERALS:
        assert literal in text, f"missing state literal {literal}"
    for marker in REQUIRED_MARKERS:
        assert marker in text, f"missing marker {marker}"
    assert "KITA-018" in text
    assert "fsync" in text.lower()
    assert "parent" in text.lower()

    compat = COMPATIBILITY.read_text(encoding="utf-8")
    for marker in COMPAT_MARKERS:
        assert marker in compat, f"missing compat marker {marker}"
    assert "unknown" in compat.lower()
    assert "legacy" in compat.lower()


def test_nested_suite_mentions_acceptance_surface() -> None:
    text = NESTED_TEST.read_text(encoding="utf-8")
    assert "collision" in text.lower() or "monotonic" in text.lower()
    assert "committed" in text
    assert "buffered" in text or "queued" in text
    assert "fsync" in text.lower()
    assert "parent" in text.lower()
    assert "SecretMaterialError" in text or "secrets" in text.lower()
    assert "unknown" in text.lower()
    assert "legacy" in text.lower()
    assert "pickle" in text.lower() or "unsafe" in text.lower()
