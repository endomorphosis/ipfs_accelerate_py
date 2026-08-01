"""Root-level implied validation mirror for KITA-022 ARC reference model.

The authoritative suite lives under the nested ``ipfs_kit_py`` package tree.
This module asserts the declared contracts, reference model, and nested tests
exist from a superproject checkout and cover the acceptance surface.
"""

from __future__ import annotations

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
NESTED_ROOT = WORKSPACE / "ipfs_kit_py"
CONTRACTS = NESTED_ROOT / "ipfs_kit_py" / "cache" / "arc" / "contracts.py"
REFERENCE = NESTED_ROOT / "ipfs_kit_py" / "cache" / "arc" / "reference.py"
NESTED_TEST = (
    NESTED_ROOT / "tests" / "runtime_readiness" / "arc" / "test_arc_reference_model.py"
)

REQUIRED_CONTRACT_MARKERS = (
    "CacheKey_V1",
    "AdaptiveReplacementCache_V1",
    "ARCReferenceModel_V1",
    "class CacheKey",
    "class ARCConfig",
    "class GhostEntry",
    "class AdaptiveReplacementCache",
    "def assert_arc_invariants",
    "def lists_pairwise_disjoint",
    "def ghost_lists_have_no_values",
    "def adaptive_target_bounded",
    "def current_size_matches_live",
    "KITA-022",
)

REQUIRED_REFERENCE_MARKERS = (
    "class ARCReferenceModel",
    "ARCReferenceModel_V1",
    "REFERENCE_MODEL_SCHEMA",
    "def assert_invariants",
    "def put",
    "def get",
    "def delete",
    "def snapshot",
    "def run_trace",
    "def minimal_trace_strategy",
    "def run_seeded_trace",
    "KITA-022",
)

REQUIRED_TEST_MARKERS = (
    "test_current_size_equals_live_t1_plus_t2_and_capacity",
    "test_pairwise_disjoint_lists_after_operations",
    "test_ghost_lists_retain_no_values",
    "test_adaptive_target_bounded_on_ghost_hits",
    "test_exact_update_growth_and_ghost_accounting",
    "test_deterministic_eviction_order",
    "test_invalid_keys_sizes_capacities_reject",
    "test_property_strategy_emits_reproducible_minimal_traces",
    "minimal_trace_strategy",
    "assert_invariants",
)


def test_declared_outputs_present_from_superproject() -> None:
    assert CONTRACTS.is_file(), f"missing {CONTRACTS}"
    assert REFERENCE.is_file(), f"missing {REFERENCE}"
    assert NESTED_TEST.is_file(), f"missing {NESTED_TEST}"

    contracts_text = CONTRACTS.read_text(encoding="utf-8")
    for marker in REQUIRED_CONTRACT_MARKERS:
        assert marker in contracts_text, f"missing contracts marker {marker}"

    ref_text = REFERENCE.read_text(encoding="utf-8")
    for marker in REQUIRED_REFERENCE_MARKERS:
        assert marker in ref_text, f"missing reference marker {marker}"


def test_nested_suite_covers_acceptance_surface() -> None:
    text = NESTED_TEST.read_text(encoding="utf-8")
    for marker in REQUIRED_TEST_MARKERS:
        assert marker in text, f"missing test marker {marker}"
    assert "current_size" in text
    assert "pairwise" in text.lower() or "disjoint" in text.lower()
    assert "ghost" in text.lower()
    assert "adaptive" in text.lower()
    assert "evict" in text.lower()
    assert "reproducible" in text.lower()
    assert "reject" in text.lower()
