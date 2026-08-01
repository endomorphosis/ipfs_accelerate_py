"""Root-level implied validation mirror for KITA-007 legacy VFS adapters.

The authoritative suite lives under the nested ``ipfs_kit_py`` package tree.
This module asserts declared outputs exist from a superproject checkout and
cover the acceptance surface.
"""

from __future__ import annotations

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
NESTED_ROOT = WORKSPACE / "ipfs_kit_py"
ADAPTERS = NESTED_ROOT / "ipfs_kit_py" / "core" / "vfs" / "adapters.py"
VFS_MANAGER = NESTED_ROOT / "ipfs_kit_py" / "vfs_manager.py"
MCP_VFS = NESTED_ROOT / "ipfs_kit_py" / "mcp" / "ipfs_kit" / "vfs.py"
NESTED_TEST = (
    NESTED_ROOT / "tests" / "runtime_readiness" / "vfs" / "test_vfs_legacy_adapters.py"
)

REQUIRED_ADAPTER_MARKERS = (
    "class LegacyVFSAdapter",
    "LegacyVFSAdapter_V1",
    "LEGACY_VFS_ADAPTER_SCHEMA",
    "record_operation",
    "get_entries",
    "log_operation",
    "get_recent_entries",
    "CommittedEventBuffer",
    "collision_safe_buffer_identity",
    "assert_cutover_ready",
    "CutoverBlockedError",
    "project_legacy_result",
    "is_underlying_success",
    "RESOLVED_LEGACY_CALLERS",
    "KITA-007",
)

REQUIRED_MANAGER_MARKERS = (
    "LegacyVFSAdapter",
    "record_operation",
    "get_entries",
    "_journal_record",
    "_flush_operation_buffer",
    "cutover_status",
)

REQUIRED_MCP_MARKERS = (
    "project_legacy_result",
    "committed",
    "retryable",
    "flush_to_dataset",
)

REQUIRED_TEST_MARKERS = (
    "test_resolved_legacy_callers_are_migrated_atomically",
    "test_unresolved_dynamic_callers_block_cutover",
    "test_journal_bridge_uses_record_operation_and_get_entries",
    "test_source_files_have_no_log_operation_or_get_recent_entries_calls",
    "test_false_error_results_cannot_become_success",
    "test_event_dataset_buffer_follows_committed_state_only",
    "test_unavailable_dataset_flush_retains_retryable_work_with_collision_safe_identity",
    "test_vfs_manager_uses_adapter_for_mutations",
    "test_failure_creates_no_success_event_and_no_buffer_publish",
)


def test_declared_outputs_present_from_superproject() -> None:
    assert ADAPTERS.is_file(), f"missing {ADAPTERS}"
    assert VFS_MANAGER.is_file(), f"missing {VFS_MANAGER}"
    assert MCP_VFS.is_file(), f"missing {MCP_VFS}"
    assert NESTED_TEST.is_file(), f"missing {NESTED_TEST}"

    adapters_text = ADAPTERS.read_text(encoding="utf-8")
    for marker in REQUIRED_ADAPTER_MARKERS:
        assert marker in adapters_text, f"missing adapter marker {marker}"

    manager_text = VFS_MANAGER.read_text(encoding="utf-8")
    for marker in REQUIRED_MANAGER_MARKERS:
        assert marker in manager_text, f"missing manager marker {marker}"

    mcp_text = MCP_VFS.read_text(encoding="utf-8")
    for marker in REQUIRED_MCP_MARKERS:
        assert marker in mcp_text, f"missing mcp marker {marker}"


def test_nested_suite_covers_acceptance_surface() -> None:
    text = NESTED_TEST.read_text(encoding="utf-8")
    for marker in REQUIRED_TEST_MARKERS:
        assert marker in text, f"missing test marker {marker}"
    assert "record_operation" in text
    assert "get_entries" in text
    assert "collision" in text.lower()
    assert "cutover" in text.lower()
    assert "committed" in text.lower()
    assert "success" in text.lower()
    assert "failure" in text.lower()
