"""IVP-009: semantic affected-check and test selection.

Acceptance coverage:

* Direct/transitive symbol/path/test/fixture/config edges select correctly
* Unrelated edits do not expand exact selections
* Unknown/dynamic/opaque/uncovered/truncated/conflicting critical edges set
  broader or full-suite fallback
* Changed obligation dependencies select proofs
* Deterministic order and reason chains are stable
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.verification.datasets_adapter import (
    DATASETS_INVALIDATION_PLAN_SCHEMA,
    DATASETS_SEMANTIC_CAPSULE_SCHEMA,
    EdgeDisposition,
    create_datasets_verification_input_adapter,
)
from ipfs_accelerate_py.agent_supervisor.verification.selection import (
    AFFECTED_VERIFICATION_SELECTION_INTERFACE,
    AFFECTED_VERIFICATION_SELECTION_SCHEMA,
    REASON_CONFIG_EDGE,
    REASON_CONFLICTING_CRITICAL,
    REASON_DIRECT_PATH,
    REASON_DIRECT_SYMBOL,
    REASON_DYNAMIC_CRITICAL,
    REASON_FIXTURE_EDGE,
    REASON_FULL_SUITE_POLICY,
    REASON_FULL_SUITE_UNCERTAINTY,
    REASON_OPAQUE_CRITICAL,
    REASON_PROOF_DEPENDENCY,
    REASON_PROVED_BY,
    REASON_STATIC_TARGET,
    REASON_TESTED_BY,
    REASON_TRANSITIVE_DEPENDENCY,
    REASON_TRUNCATED_FRONTIER,
    REASON_TYPE_TARGET,
    REASON_UNCOVERED_IMPACT,
    REASON_UNKNOWN_CRITICAL,
    REASON_UNRELATED_NO_EXPANSION,
    REASON_VALIDATION_MAPPING_INCOMPLETE,
    SELECTION_EVIDENCE,
    AffectedCheckSelector,
    AffectedVerificationSelection,
    FallbackMode,
    SelectionError,
    SelectionPolicy,
    VerificationCatalog,
    create_affected_check_selector,
    select_affected_verification,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return content_identity({"artifact": label, "schema": "fixture-artifact@1"})


SEMANTIC = _cid("semantic-root-selection")
OPAQUE_TREE = "datasets-tree:selection-fixture-001"

TEST_A = "test/api/test_mod.py::test_fn"
TEST_B = "test/api/test_mod.py::test_helper"
TEST_C = "test/api/test_other.py::test_unrelated"
TEST_D = "test/api/test_config.py::test_configured"
TEST_E = "test/api/test_fixture.py::test_with_fixture"

PROOF_A = _cid("proof-obligation-fn")
PROOF_B = _cid("proof-obligation-helper")
PROOF_C = _cid("proof-obligation-unrelated")

STATIC_A = "static:ruff:pkg/mod.py"
STATIC_B = "static:ruff:pkg/other.py"
TYPE_A = "mypy:pkg.mod"
TYPE_B = "mypy:pkg.other"


def _catalog(**overrides: Any) -> VerificationCatalog:
    base: dict[str, Any] = {
        "tests": [TEST_A, TEST_B, TEST_C, TEST_D, TEST_E],
        "static_checks": [STATIC_A, STATIC_B],
        "type_checks": [TYPE_A, TYPE_B],
        "proof_obligations": [PROOF_A, PROOF_B, PROOF_C],
        "static_check_targets": {
            STATIC_A: ["pkg.mod.fn", "pkg/mod.py"],
            STATIC_B: ["pkg.other.g", "pkg/other.py"],
        },
        "type_check_targets": {
            TYPE_A: ["pkg.mod.fn", "pkg.mod.Helper", "pkg/mod.py"],
            TYPE_B: ["pkg.other.g", "pkg/other.py"],
        },
        "proof_obligation_dependencies": {
            PROOF_A: ["pkg.mod.fn"],
            PROOF_B: ["pkg.mod.Helper"],
            PROOF_C: ["pkg.other.g"],
        },
    }
    base.update(overrides)
    return VerificationCatalog(**base)


def _policy(**kwargs: Any) -> SelectionPolicy:
    return SelectionPolicy(**kwargs)


def _edge(
    source: str,
    target: str,
    kind: str,
    *,
    disposition: str = "exact",
    truncated: bool = False,
    opaque: bool = False,
    uncovered: bool = False,
    critical: bool = True,
    edge_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": source,
        "target": target,
        "kind": kind,
        "disposition": disposition,
        "truncated": truncated,
        "opaque": opaque,
        "uncovered": uncovered,
        "critical": critical,
    }
    if edge_id is not None:
        payload["edge_id"] = edge_id
    return payload


def _select(**kwargs: Any) -> AffectedVerificationSelection:
    if "catalog" not in kwargs:
        kwargs["catalog"] = _catalog()
    if "policy" not in kwargs:
        kwargs["policy"] = _policy(
            # Prefer broader (not full suite) for most uncertainty tests;
            # individual tests override for full-suite cases.
            critical_uncertainty_requires_full_suite=False,
        )
    return select_affected_verification(**kwargs)


# ---------------------------------------------------------------------------
# Interface / schema surface
# ---------------------------------------------------------------------------


def test_module_exports_required_interfaces() -> None:
    assert AFFECTED_VERIFICATION_SELECTION_INTERFACE == (
        "AffectedVerificationSelection@1"
    )
    assert AFFECTED_VERIFICATION_SELECTION_SCHEMA.endswith(
        "affected-verification-selection@1"
    )
    assert SELECTION_EVIDENCE == "ivp/test-selection@1"
    result = _select(
        changed_symbols=["pkg.mod.fn"],
        edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
    )
    assert isinstance(result, AffectedVerificationSelection)
    assert result.interface == AFFECTED_VERIFICATION_SELECTION_INTERFACE
    assert result.evidence == SELECTION_EVIDENCE
    assert result.authoritative is False


def test_selector_object_matches_function() -> None:
    catalog = _catalog()
    policy = _policy(critical_uncertainty_requires_full_suite=False)
    selector = create_affected_check_selector(catalog=catalog, policy=policy)
    assert isinstance(selector, AffectedCheckSelector)
    kwargs = {
        "changed_symbols": ["pkg.mod.fn"],
        "edges": [_edge("pkg.mod.fn", TEST_A, "tested_by")],
    }
    a = selector.select(**kwargs)
    b = select_affected_verification(
        catalog=catalog, policy=policy, **kwargs
    )
    assert a.to_dict() == b.to_dict()


# ---------------------------------------------------------------------------
# Direct / transitive selection
# ---------------------------------------------------------------------------


def test_direct_symbol_tested_by_edge_selects_test() -> None:
    result = _select(
        changed_symbols=["pkg.mod.fn"],
        edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
    )
    assert result.affected_tests == (TEST_A,)
    assert TEST_C not in result.affected_tests
    assert result.fallback_mode is FallbackMode.EXACT
    assert result.broader_selection_required is False
    assert result.full_suite_required is False
    assert REASON_TESTED_BY in result.selection_reason_codes
    assert REASON_DIRECT_SYMBOL in result.selection_reason_codes
    chain = result.reason_chains[TEST_A]
    assert chain[0] == "pkg.mod.fn"
    assert TEST_A in chain


def test_transitive_depends_on_selects_consumer_tests() -> None:
    # Helper depends on fn? Orientation: source depends_on target.
    # pkg.mod.fn depends_on pkg.mod.Helper means Helper is provider.
    # Changing Helper impacts fn, which is tested_by TEST_A.
    edges = [
        _edge("pkg.mod.fn", "pkg.mod.Helper", "depends_on"),
        _edge("pkg.mod.fn", TEST_A, "tested_by"),
        _edge("pkg.mod.Helper", TEST_B, "tested_by"),
    ]
    result = _select(changed_symbols=["pkg.mod.Helper"], edges=edges)
    assert TEST_B in result.affected_tests  # direct
    assert TEST_A in result.affected_tests  # transitive via depends_on reverse
    assert TEST_C not in result.affected_tests
    assert "pkg.mod.fn" in result.dependency_cone_symbols
    assert "pkg.mod.Helper" in result.dependency_cone_symbols
    assert REASON_TRANSITIVE_DEPENDENCY in result.selection_reason_codes
    # Reason chain for TEST_A should include Helper -> fn -> test
    chain = result.reason_chains[TEST_A]
    assert chain[0] == "pkg.mod.Helper"
    assert "pkg.mod.fn" in chain
    assert TEST_A in chain


def test_path_change_selects_via_path_edges_and_catalog() -> None:
    # Orientation: source depends_on target (provider). Changing util impacts mod.
    edges = [
        _edge("pkg/mod.py", "pkg/util.py", "depends_on"),
        _edge("pkg/mod.py", TEST_A, "tested_by"),
    ]
    result = _select(changed_paths=["pkg/util.py"], edges=edges)
    assert TEST_A in result.affected_tests
    assert "pkg/mod.py" in result.dependency_cone_paths
    assert REASON_DIRECT_PATH in result.selection_reason_codes
    assert STATIC_A in result.required_static_checks  # catalog targets path
    assert REASON_STATIC_TARGET in result.selection_reason_codes


def test_imports_and_calls_edges_expand_cone() -> None:
    edges = [
        _edge("pkg.consumer", "pkg.provider", "imports"),
        _edge("pkg.caller", "pkg.provider", "calls"),
        _edge("pkg.consumer", TEST_A, "tested_by"),
        _edge("pkg.caller", TEST_B, "tested_by"),
    ]
    result = _select(changed_symbols=["pkg.provider"], edges=edges)
    assert set(result.affected_tests) == {TEST_A, TEST_B}


# ---------------------------------------------------------------------------
# Fixture / config edges
# ---------------------------------------------------------------------------


def test_fixture_edge_selects_test() -> None:
    result = _select(
        changed_symbols=["fixture:db"],
        edges=[_edge("fixture:db", TEST_E, "fixtures")],
    )
    assert TEST_E in result.affected_tests
    assert REASON_FIXTURE_EDGE in result.selection_reason_codes


def test_config_edge_selects_configured_test() -> None:
    result = _select(
        changed_paths=["config/settings.toml"],
        edges=[_edge("config/settings.toml", TEST_D, "configures")],
    )
    assert TEST_D in result.affected_tests
    assert REASON_CONFIG_EDGE in result.selection_reason_codes


def test_config_edge_to_module_selects_module_tests() -> None:
    edges = [
        _edge("config/settings.toml", "pkg.mod.fn", "configures"),
        _edge("pkg.mod.fn", TEST_A, "tested_by"),
    ]
    result = _select(changed_paths=["config/settings.toml"], edges=edges)
    assert TEST_A in result.affected_tests


# ---------------------------------------------------------------------------
# Unrelated edits
# ---------------------------------------------------------------------------


def test_unrelated_edit_does_not_expand_exact_selection() -> None:
    edges = [
        _edge("pkg.mod.fn", TEST_A, "tested_by"),
        _edge("pkg.mod.Helper", TEST_B, "tested_by"),
        _edge("pkg.other.g", TEST_C, "tested_by"),
    ]
    result = _select(
        changed_symbols=["pkg.unrelated.symbol"],
        changed_paths=["docs/readme.md"],
        edges=edges,
    )
    assert result.affected_tests == ()
    assert result.required_static_checks == ()
    assert result.required_type_checks == ()
    assert result.affected_proof_obligation_cids == ()
    assert result.broader_selection_required is False
    assert result.full_suite_required is False
    assert result.fallback_mode is FallbackMode.EXACT
    assert REASON_UNRELATED_NO_EXPANSION in result.selection_reason_codes
    # Catalog-known but unrelated tests stay out of exact and fallback.
    assert TEST_A not in result.selected_tests
    assert TEST_C not in result.selected_tests


def test_unrelated_edit_preserves_empty_fallback() -> None:
    result = _select(
        changed_paths=["CHANGELOG.md"],
        edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
    )
    assert result.affected_tests == ()
    assert result.fallback_tests == ()
    assert result.selected_tests == ()


# ---------------------------------------------------------------------------
# Uncertainty / fallback
# ---------------------------------------------------------------------------


def test_opaque_critical_edge_requires_broader_fallback() -> None:
    edges = [
        _edge("pkg.mod.fn", TEST_A, "tested_by"),
        _edge(
            "pkg.mod.fn",
            "pkg.dynamic.sink",
            "opaque",
            opaque=True,
            disposition="opaque",
        ),
    ]
    result = _select(changed_symbols=["pkg.mod.fn"], edges=edges)
    assert TEST_A in result.affected_tests
    assert result.broader_selection_required is True
    assert result.fallback_mode in {
        FallbackMode.BROADER,
        FallbackMode.FULL_SUITE,
    }
    assert REASON_OPAQUE_CRITICAL in result.fallback_reason_codes
    assert result.critical_uncertain_edges


def test_dynamic_critical_edge_requires_broader_fallback() -> None:
    edges = [
        _edge("pkg.mod.fn", "runtime:plugin", "dynamic", disposition="opaque"),
    ]
    result = _select(changed_symbols=["pkg.mod.fn"], edges=edges)
    assert result.broader_selection_required is True
    assert REASON_DYNAMIC_CRITICAL in result.fallback_reason_codes


def test_unknown_critical_edge_requires_broader_fallback() -> None:
    edges = [
        _edge("pkg.mod.fn", "somewhere", "unknown", disposition="opaque"),
    ]
    result = _select(changed_symbols=["pkg.mod.fn"], edges=edges)
    assert result.broader_selection_required is True
    assert REASON_UNKNOWN_CRITICAL in result.fallback_reason_codes


def test_uncovered_impact_requires_broader() -> None:
    result = _select(
        changed_symbols=["pkg.mod.fn", "pkg.ghost.missing"],
        uncovered_symbols=["pkg.ghost.missing"],
        edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
    )
    assert TEST_A in result.affected_tests
    assert result.broader_selection_required is True
    assert REASON_UNCOVERED_IMPACT in result.fallback_reason_codes


def test_truncated_frontier_requires_broader() -> None:
    edges = [
        _edge("pkg.mod.fn", TEST_A, "tested_by"),
        _edge(
            "pkg.mod.fn",
            "pkg.deep",
            "depends_on",
            truncated=True,
            disposition="truncated",
        ),
    ]
    result = _select(changed_symbols=["pkg.mod.fn"], edges=edges, truncated=True)
    assert result.broader_selection_required is True
    assert REASON_TRUNCATED_FRONTIER in result.fallback_reason_codes


def test_conflicting_critical_edges_require_broader() -> None:
    edges = [
        _edge(
            "pkg.mod.fn",
            TEST_A,
            "tested_by",
            disposition="exact",
            edge_id="edge:exact",
        ),
        _edge(
            "pkg.mod.fn",
            TEST_A,
            "tested_by",
            disposition="opaque",
            opaque=True,
            edge_id="edge:opaque",
        ),
    ]
    result = _select(changed_symbols=["pkg.mod.fn"], edges=edges)
    assert result.broader_selection_required is True
    assert REASON_CONFLICTING_CRITICAL in result.fallback_reason_codes
    assert result.conflicting_edge_ids
    # Exact test may still appear when one edge is exact; uncertainty broadens.
    # Opaque test edge does not exact-select; the exact sibling does.
    assert TEST_A in result.affected_tests or result.fallback_tests


def test_critical_uncertainty_can_require_full_suite() -> None:
    edges = [
        _edge("pkg.mod.fn", TEST_A, "tested_by"),
        _edge(
            "pkg.mod.fn",
            "x",
            "opaque",
            opaque=True,
            disposition="opaque",
        ),
    ]
    result = _select(
        changed_symbols=["pkg.mod.fn"],
        edges=edges,
        policy=_policy(critical_uncertainty_requires_full_suite=True),
    )
    assert result.full_suite_required is True
    assert result.broader_selection_required is True
    assert result.fallback_mode is FallbackMode.FULL_SUITE
    assert REASON_FULL_SUITE_UNCERTAINTY in result.full_suite_reason_codes
    # Full suite fallback includes the catalog.
    assert set(result.fallback_tests) == {
        TEST_A,
        TEST_B,
        TEST_C,
        TEST_D,
        TEST_E,
    }


def test_policy_force_full_suite() -> None:
    result = _select(
        changed_symbols=["pkg.mod.fn"],
        edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
        policy=_policy(force_full_suite=True),
    )
    assert result.full_suite_required is True
    assert REASON_FULL_SUITE_POLICY in result.full_suite_reason_codes
    assert TEST_C in result.fallback_tests


def test_broader_sibling_expansion_without_full_suite() -> None:
    result = _select(
        changed_symbols=["pkg.mod.fn"],
        edges=[
            _edge("pkg.mod.fn", TEST_A, "tested_by"),
            _edge(
                "pkg.mod.fn",
                "x",
                "opaque",
                opaque=True,
                disposition="opaque",
            ),
        ],
        policy=_policy(
            critical_uncertainty_requires_full_suite=False,
            broader_includes_sibling_tests=True,
        ),
    )
    assert result.broader_selection_required is True
    assert result.full_suite_required is False
    assert result.fallback_mode is FallbackMode.BROADER
    # Sibling TEST_B shares test/api/test_mod.py with TEST_A.
    assert TEST_B in result.fallback_tests
    # Unrelated test file is not pulled in by sibling expansion.
    assert TEST_C not in result.fallback_tests
    assert TEST_C not in result.affected_tests


def test_validation_mapping_incomplete_forces_broader() -> None:
    result = _select(
        changed_symbols=["pkg.mod.fn"],
        edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
        validation_selection={
            "mapped_pytest_node_ids": [TEST_A],
            "unmapped_validation_ids": ["val:missing"],
            "requires_broader_selection": True,
        },
    )
    assert result.broader_selection_required is True
    assert REASON_VALIDATION_MAPPING_INCOMPLETE in result.fallback_reason_codes
    assert TEST_A in result.affected_tests


# ---------------------------------------------------------------------------
# Proof obligations / static / type checks
# ---------------------------------------------------------------------------


def test_proved_by_edge_selects_proof() -> None:
    result = _select(
        changed_symbols=["pkg.mod.fn"],
        edges=[
            _edge("pkg.mod.fn", TEST_A, "tested_by"),
            _edge("pkg.mod.fn", PROOF_A, "proved_by"),
        ],
    )
    assert PROOF_A in result.affected_proof_obligation_cids
    assert PROOF_C not in result.affected_proof_obligation_cids
    assert REASON_PROVED_BY in result.selection_reason_codes


def test_changed_obligation_dependencies_select_proofs() -> None:
    # No proved_by edge; catalog dependency mapping drives selection.
    result = _select(
        changed_symbols=["pkg.mod.Helper"],
        edges=[_edge("pkg.mod.fn", "pkg.mod.Helper", "depends_on")],
    )
    # Helper is directly changed -> PROOF_B
    assert PROOF_B in result.affected_proof_obligation_cids
    # Transitive: fn depends on Helper, so fn is in cone -> PROOF_A
    assert PROOF_A in result.affected_proof_obligation_cids
    assert PROOF_C not in result.affected_proof_obligation_cids
    assert REASON_PROOF_DEPENDENCY in result.selection_reason_codes


def test_static_and_type_checks_from_catalog_targets() -> None:
    result = _select(
        changed_symbols=["pkg.mod.fn"],
        edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
    )
    assert STATIC_A in result.required_static_checks
    assert STATIC_B not in result.required_static_checks
    assert TYPE_A in result.required_type_checks
    assert TYPE_B not in result.required_type_checks
    assert REASON_STATIC_TARGET in result.selection_reason_codes
    assert REASON_TYPE_TARGET in result.selection_reason_codes


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_order_and_reason_chains_are_stable() -> None:
    edges = [
        _edge("pkg.mod.fn", TEST_B, "tested_by"),
        _edge("pkg.mod.fn", TEST_A, "tested_by"),
        _edge("pkg.mod.Helper", "pkg.mod.fn", "depends_on"),
        _edge("pkg.mod.Helper", TEST_B, "tested_by"),
        _edge("pkg.mod.fn", PROOF_A, "proved_by"),
    ]
    # Shuffled changed symbols / edges across calls.
    results = []
    for _ in range(5):
        shuffled_edges = list(reversed(edges))
        result = _select(
            changed_symbols=["pkg.mod.Helper", "pkg.mod.fn"],
            edges=shuffled_edges,
        )
        results.append(result.to_dict())
    first = results[0]
    for other in results[1:]:
        assert other == first
    # Lexicographic ordering of collections.
    result = AffectedVerificationSelection.from_dict(first)
    assert result.affected_tests == tuple(sorted(result.affected_tests))
    assert result.affected_proof_obligation_cids == tuple(
        sorted(result.affected_proof_obligation_cids)
    )
    assert list(result.reason_chains.keys()) == sorted(result.reason_chains.keys())
    # Reason chains prefer shorter paths when both direct and transitive exist.
    if TEST_B in result.reason_chains:
        chain = result.reason_chains[TEST_B]
        assert chain  # non-empty and stable
        # Round-trip preserves chains.
        assert (
            AffectedVerificationSelection.from_dict(result.to_dict()).reason_chains
            == result.reason_chains
        )


def test_reason_chain_bounded_by_policy() -> None:
    # Build a long dependency chain.
    edges = []
    n = 12
    for i in range(n):
        edges.append(
            _edge(f"sym.{i + 1}", f"sym.{i}", "depends_on")
        )
    edges.append(_edge(f"sym.{n}", TEST_A, "tested_by"))
    result = _select(
        changed_symbols=["sym.0"],
        edges=edges,
        policy=_policy(
            critical_uncertainty_requires_full_suite=False,
            max_reason_chain=4,
        ),
    )
    assert TEST_A in result.affected_tests
    chain = result.reason_chains[TEST_A]
    assert len(chain) <= 4


def test_selection_to_dict_round_trip() -> None:
    result = _select(
        changed_symbols=["pkg.mod.fn"],
        edges=[
            _edge("pkg.mod.fn", TEST_A, "tested_by"),
            _edge("pkg.mod.fn", PROOF_A, "proved_by"),
        ],
    )
    restored = AffectedVerificationSelection.from_dict(result.to_dict())
    assert restored.to_dict() == result.to_dict()
    assert restored.selected_tests == result.selected_tests


# ---------------------------------------------------------------------------
# Integration with datasets adapter views
# ---------------------------------------------------------------------------


def test_select_from_invalidation_plan_view() -> None:
    adapter = create_datasets_verification_input_adapter()
    plan_mapping = {
        "schema": DATASETS_INVALIDATION_PLAN_SCHEMA,
        "repository_tree_id": OPAQUE_TREE,
        "semantic_state_root_cid": SEMANTIC,
        "changed_symbols": ["pkg.mod.fn", "pkg.mod.Helper"],
        "changed_paths": ["pkg/mod.py"],
        "edges": [
            {
                "source": "pkg.mod.fn",
                "target": TEST_A,
                "kind": "tested_by",
            },
            {
                "source": "pkg.mod.fn",
                "target": "pkg.mod.Helper",
                "kind": "depends_on",
            },
            {
                "source": "pkg.mod.Helper",
                "target": TEST_B,
                "kind": "tested_by",
            },
            {
                "source": "pkg.mod.fn",
                "target": PROOF_A,
                "kind": "proved_by",
            },
        ],
        "spans": [],
        "contracts": [],
        "uncertainty": {"frontier": "exact"},
        "uncovered_symbols": [],
        "uncovered_paths": [],
        "truncated": False,
    }
    normalized = adapter.normalize_invalidation_plan(plan_mapping)
    assert normalized.ok
    assert normalized.view is not None
    result = select_affected_verification(
        invalidation_plan=normalized.view,
        catalog=_catalog(),
        policy=_policy(critical_uncertainty_requires_full_suite=False),
    )
    assert TEST_A in result.affected_tests
    assert TEST_B in result.affected_tests
    assert PROOF_A in result.affected_proof_obligation_cids
    assert result.broader_selection_required is False


def test_select_from_semantic_capsule_with_opaque_edges() -> None:
    adapter = create_datasets_verification_input_adapter()
    capsule = {
        "schema": DATASETS_SEMANTIC_CAPSULE_SCHEMA,
        "semantic_state_root_cid": SEMANTIC,
        "repository_tree_id": OPAQUE_TREE,
        "edges": [
            {
                "source": "pkg.mod.fn",
                "target": TEST_A,
                "kind": "tested_by",
            },
            {
                "source": "pkg.mod.fn",
                "target": "hidden",
                "kind": "opaque",
                "opaque": True,
            },
        ],
        "spans": [],
        "contracts": [],
        "fixture_references": [],
        "truncated": False,
    }
    normalized = adapter.normalize_semantic_capsule(capsule)
    assert normalized.ok
    assert normalized.view is not None
    assert normalized.requires_broader_selection is True
    result = select_affected_verification(
        changed_symbols=["pkg.mod.fn"],
        semantic_capsule=normalized.view,
        catalog=_catalog(),
        policy=_policy(critical_uncertainty_requires_full_suite=False),
    )
    assert TEST_A in result.affected_tests
    assert result.broader_selection_required is True


def test_opaque_edge_disposition_from_adapter() -> None:
    adapter = create_datasets_verification_input_adapter()
    plan = {
        "schema": DATASETS_INVALIDATION_PLAN_SCHEMA,
        "repository_tree_id": OPAQUE_TREE,
        "semantic_state_root_cid": SEMANTIC,
        "changed_symbols": ["pkg.mod.fn"],
        "changed_paths": [],
        "edges": [
            {
                "source": "pkg.mod.fn",
                "target": "dyn",
                "kind": "dynamic",
            }
        ],
        "truncated": False,
        "uncovered_symbols": [],
        "uncovered_paths": [],
    }
    view = adapter.normalize_invalidation_plan(plan).view
    assert view is not None
    assert any(e.disposition is EdgeDisposition.OPAQUE for e in view.edges)
    result = select_affected_verification(
        invalidation_plan=view,
        catalog=_catalog(),
        policy=_policy(critical_uncertainty_requires_full_suite=True),
    )
    assert result.full_suite_required is True


# ---------------------------------------------------------------------------
# Fail-closed / error handling
# ---------------------------------------------------------------------------


def test_malformed_edges_raise() -> None:
    with pytest.raises(SelectionError):
        select_affected_verification(
            changed_symbols=["a"],
            edges=["not-an-edge"],  # type: ignore[list-item]
            catalog=_catalog(),
        )


def test_empty_change_set_selects_nothing() -> None:
    result = _select(
        changed_symbols=[],
        changed_paths=[],
        edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
    )
    assert result.affected_tests == ()
    assert result.fallback_mode is FallbackMode.EXACT


def test_non_critical_uncertain_edge_can_avoid_full_suite() -> None:
    edges = [
        _edge("pkg.mod.fn", TEST_A, "tested_by"),
        _edge(
            "pkg.mod.fn",
            "hint",
            "opaque",
            opaque=True,
            critical=False,
            disposition="opaque",
        ),
    ]
    result = _select(
        changed_symbols=["pkg.mod.fn"],
        edges=edges,
        policy=_policy(
            critical_uncertainty_requires_full_suite=True,
            non_critical_uncertainty_requires_broader=True,
        ),
    )
    assert result.broader_selection_required is True
    assert result.full_suite_required is False


def test_no_io_on_import() -> None:
    # Selection module must remain side-effect free on import (already imported).
    import ipfs_accelerate_py.agent_supervisor.verification.selection as mod

    assert callable(mod.select_affected_verification)
    # Deepcopy of a result must succeed (frozen / pure data).
    result = _select(
        changed_symbols=["pkg.mod.fn"],
        edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
    )
    cloned = copy.deepcopy(result.to_dict())
    assert cloned["affected_tests"] == list(result.affected_tests)
