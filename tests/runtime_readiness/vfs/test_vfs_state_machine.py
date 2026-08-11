"""Root-level implied validation mirror for KITA-006 VFS state machine.

The authoritative suite lives under the nested ``ipfs_kit_py`` package tree.
This module asserts the declared service, reference model, and nested tests
exist from a superproject checkout and cover the acceptance surface.
"""

from __future__ import annotations

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
NESTED_ROOT = WORKSPACE / "ipfs_kit_py"
SERVICE = NESTED_ROOT / "ipfs_kit_py" / "core" / "vfs" / "service.py"
REFERENCE = NESTED_ROOT / "tests" / "runtime_readiness" / "vfs" / "reference_model.py"
NESTED_TEST = (
    NESTED_ROOT / "tests" / "runtime_readiness" / "vfs" / "test_vfs_state_machine.py"
)

REQUIRED_SERVICE_MARKERS = (
    "class CanonicalVFSService",
    "CanonicalVFSService_V1",
    "CANONICAL_VFS_SERVICE_SCHEMA",
    "class InMemoryVFSStorage",
    "class CancellationToken",
    "class VFSEvent",
    "VFSStorageBoundary",
    "def execute",
    "def run_trace",
    "KITA-006",
)

REQUIRED_REFERENCE_MARKERS = (
    "class VFSReferenceModel",
    "VFSReferenceModel_V1",
    "REFERENCE_MODEL_SCHEMA",
    "def apply",
    "def run_trace",
    "def traces_match",
    "KITA-006",
)

REQUIRED_TEST_MARKERS = (
    "test_full_crud_trace_matches_reference_model",
    "test_failure_creates_no_success_event",
    "test_rename_changes_namespace_state",
    "test_return_and_error_types_are_stable",
    "test_cancellation_before_commit",
    "test_side_effects_confined_to_injected_storage",
    "traces_match",
)


def test_declared_outputs_present_from_superproject() -> None:
    assert SERVICE.is_file(), f"missing {SERVICE}"
    assert REFERENCE.is_file(), f"missing {REFERENCE}"
    assert NESTED_TEST.is_file(), f"missing {NESTED_TEST}"

    service_text = SERVICE.read_text(encoding="utf-8")
    for marker in REQUIRED_SERVICE_MARKERS:
        assert marker in service_text, f"missing service marker {marker}"

    ref_text = REFERENCE.read_text(encoding="utf-8")
    for marker in REQUIRED_REFERENCE_MARKERS:
        assert marker in ref_text, f"missing reference marker {marker}"


def test_nested_suite_covers_acceptance_surface() -> None:
    text = NESTED_TEST.read_text(encoding="utf-8")
    for marker in REQUIRED_TEST_MARKERS:
        assert marker in text, f"missing test marker {marker}"
    assert "rename" in text.lower()
    assert "move" in text.lower()
    assert "cancel" in text.lower()
    assert "reference" in text.lower()
    assert "success" in text.lower()
    assert "failure" in text.lower()
