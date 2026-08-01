"""Root-level implied validation mirror for KITA-010 bucket contracts.

The authoritative suite lives under the nested ``ipfs_kit_py`` package tree.
This module asserts the declared contract module and nested tests exist from a
superproject checkout.
"""

from __future__ import annotations

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
NESTED_ROOT = WORKSPACE / "ipfs_kit_py"
CONTRACTS = NESTED_ROOT / "ipfs_kit_py" / "core" / "buckets" / "contracts.py"
NESTED_TEST = (
    NESTED_ROOT
    / "tests"
    / "runtime_readiness"
    / "buckets"
    / "test_bucket_contracts.py"
)

REQUIRED_CLASSES = (
    "BucketIdentity",
    "BucketCatalog",
    "BucketPolicy",
    "BucketManifest",
    "BackendBinding",
    "BackendCapabilityProfile",
    "CatalogEntry",
)

REQUIRED_MARKERS = (
    "BUCKET_IDENTITY_SCHEMA",
    "BUCKET_CATALOG_SCHEMA",
    "BUCKET_POLICY_SCHEMA",
    "BUCKET_MANIFEST_SCHEMA",
    "EXACTLY_ONE_PRIMARY",
    "VERIFIED_REPLICA_DEFINITION",
    "CONFIGURED_POLICY_NOT_ENFORCED",
    "PolicyDisposition",
    "ReplicaCountKind",
    "UnknownFieldError",
    "PolicyInvariantError",
    "CapabilityInsufficientError",
    "InvalidLifecycleTransitionError",
    "is_legal_lifecycle_transition",
)

REQUIRED_LITERALS = (
    '"exactly_one_primary"',
    '"verified_replica"',
    '"configured"',
    '"desired"',
    '"enforced"',
    '"primary"',
    '"replica"',
)


def test_declared_outputs_present_from_superproject():
    assert CONTRACTS.is_file(), f"missing {CONTRACTS}"
    assert NESTED_TEST.is_file(), f"missing {NESTED_TEST}"
    text = CONTRACTS.read_text(encoding="utf-8")
    for name in REQUIRED_CLASSES:
        assert f"class {name}" in text, f"missing class {name}"
    for marker in REQUIRED_MARKERS:
        assert marker in text, f"missing marker {marker}"
    for literal in REQUIRED_LITERALS:
        assert literal in text, f"missing literal {literal}"
    assert "KITA-010" in text
    assert "backend-scoped" in text.lower() or "backend scoped" in text.lower()


def test_nested_suite_mentions_acceptance_surface():
    text = NESTED_TEST.read_text(encoding="utf-8")
    assert "equal names" in text.lower() or "distinct backends" in text.lower()
    assert "exactly_one_primary" in text or "exactly one primary" in text.lower()
    assert "verified" in text.lower() and "replica" in text.lower()
    assert "unknown" in text.lower()
    assert "secret" in text.lower()
    assert "cycle" in text.lower()
    assert "enforced" in text.lower()
    assert "alias" in text.lower()
    assert "capability" in text.lower()
