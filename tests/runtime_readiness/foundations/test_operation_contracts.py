"""Root-level implied validation mirror for KITA-002 operation contracts.

The authoritative suite lives under the nested ``ipfs_kit_py`` package tree.
This module asserts the declared contract module and nested tests exist from a
superproject checkout.
"""

from __future__ import annotations

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
NESTED_ROOT = WORKSPACE / "ipfs_kit_py"
CONTRACTS = NESTED_ROOT / "ipfs_kit_py" / "core" / "operation_contracts.py"
NESTED_TEST = (
    NESTED_ROOT
    / "tests"
    / "runtime_readiness"
    / "foundations"
    / "test_operation_contracts.py"
)

REQUIRED_CLASSES = (
    "OperationRequest",
    "OperationResult",
    "StorageError",
    "StateTransitionReceipt",
    "IdentityBindings",
    "DurabilityEvidence",
    "PartialEffectRecord",
)

REQUIRED_STATE_LITERALS = (
    '"accepted"',
    '"queued"',
    '"committed"',
    '"verified"',
    '"converged"',
    '"partial_effect"',
    '"failed"',
    '"rejected"',
)

REQUIRED_MARKERS = (
    "OPERATION_REQUEST_SCHEMA",
    "OPERATION_RESULT_SCHEMA",
    "STORAGE_ERROR_SCHEMA",
    "STATE_TRANSITION_RECEIPT_SCHEMA",
    "SecretMaterialError",
    "ForgedIdentityError",
    "InconsistentStateError",
    "BodyRejectedError",
    "CycleDetectedError",
    "DURABLE_STATES",
    "VERIFIED_STATES",
    "CONVERGED_STATES",
    "FacetKind",
)


def test_declared_outputs_present_from_superproject():
    assert CONTRACTS.is_file(), f"missing {CONTRACTS}"
    assert NESTED_TEST.is_file(), f"missing {NESTED_TEST}"
    text = CONTRACTS.read_text(encoding="utf-8")
    for name in REQUIRED_CLASSES:
        assert f"class {name}" in text, f"missing class {name}"
    for literal in REQUIRED_STATE_LITERALS:
        assert literal in text, f"missing state literal {literal}"
    for marker in REQUIRED_MARKERS:
        assert marker in text, f"missing marker {marker}"
    assert "KITA-002" in text
    assert "type" in text.lower() and "resource" in text.lower() and "memory" in text.lower()


def test_nested_suite_mentions_acceptance_surface():
    text = NESTED_TEST.read_text(encoding="utf-8")
    assert "accepted" in text
    assert "committed" in text
    assert "verified" in text
    assert "converged" in text
    assert "partial" in text.lower()
    assert "SecretMaterialError" in text or "secrets" in text.lower()
    assert "idempotency" in text.lower()
    assert "durability" in text.lower()
