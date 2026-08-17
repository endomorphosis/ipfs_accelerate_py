"""IPS-046: complete positive invalidation and reuse matrix.

Covers every normative mutation kind with exact invalidated / reused /
added / removed / fallback sets and reason-labeled dependency closures.
Roots and closures repeat deterministically.  Ordinary documentation never
invalidates execution units.  Deleted tests require authorization; added
selected tests are proven.  Trust, schema, canonicalization, key, and circuit
changes force full proof where required.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
    ParentSealContext,
    UnitPlanningInput,
    create_incremental_plan,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.dependency_graph import (
    DependencyEdgeType,
    DependencyNodeKind,
    ProofDependencyGraph,
    compute_dependency_root,
    mint_reason_cid,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.identity import (
    ABSENCE_TOKEN,
    canonical_cid,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.invalidation import (
    FULL_FALLBACK_CHANGE_CLASSES,
    InvalidationPolicy,
    UnitDispositionKind,
    classify_full_fallback,
    compute_invalidation_closure,
    explain_invalidation,
    sample_invalidation_policy,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.repository_diff import (
    ChangeAction,
    ChangeClass,
    ChangedArtifact,
)

# ---------------------------------------------------------------------------
# Fixture corpus (IPS-045)
# ---------------------------------------------------------------------------

_GEN = (
    Path(__file__).resolve().parents[2]
    / "fixtures/incremental_proof_sealer/generate_fixture_history.py"
)
_SPEC = importlib.util.spec_from_file_location("ips_fixture_generator", _GEN)
assert _SPEC is not None and _SPEC.loader is not None
_GEN_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GEN_MOD)
generate_corpus = _GEN_MOD.generate_corpus
REQUIRED_SCENARIO_KINDS = _GEN_MOD.REQUIRED_SCENARIO_KINDS

EVIDENCE_SUBSET = "ips/invalidation-positive@1"

# Known proof units for the positive matrix graph.
KNOWN_UNITS: tuple[str, ...] = (
    "aggregate/receipt_a",
    "aggregate/receipt_b",
    "unit/formal_a",
    "unit/static_a",
    "unit/static_b",
    "unit/test_a",
    "unit/test_b",
)

MODULE_A_UNITS: frozenset[str] = frozenset(
    {
        "aggregate/receipt_a",
        "unit/formal_a",
        "unit/static_a",
        "unit/test_a",
    }
)
MODULE_B_UNITS: frozenset[str] = frozenset(
    {
        "aggregate/receipt_b",
        "unit/static_b",
        "unit/test_b",
    }
)

# Mutation kinds required by acceptance / board mutation contract.
MATRIX_MUTATION_KINDS: tuple[str, ...] = (
    "source_implementation",
    "public_interface",
    "test_selector",
    "test_source",
    "test_add",
    "test_delete_unauthorized",
    "test_delete_authorized",
    "fixture",
    "relevant_configuration",
    "network_policy",
    "verification_policy",
    "dependency_lock",
    "dependency_lock_full_policy",
    "tool_prover_version",
    "circuit",
    "proving_key",
    "verification_key",
    "proof_schema",
    "canonicalization",
    "checked_specification",
    "ordinary_documentation",
    "independent_module",
    "two_module",
)


def _reason(label: str) -> str:
    return mint_reason_cid({"positive_matrix": label, "v": 1})


def _cid(label: str) -> str:
    return canonical_cid({"ips_positive_matrix": label, "v": 1})


def _changed(
    path: str,
    change_class: ChangeClass,
    *,
    action: ChangeAction = ChangeAction.MODIFIED,
) -> ChangedArtifact:
    if action is ChangeAction.ADDED:
        return ChangedArtifact(
            path=path,
            change_action=ChangeAction.ADDED,
            change_class=change_class,
            old_content_cid=ABSENCE_TOKEN,
            new_content_cid=_cid(f"{path}:new"),
            old_byte_length=ABSENCE_TOKEN,
            new_byte_length=4,
        )
    if action is ChangeAction.DELETED:
        return ChangedArtifact(
            path=path,
            change_action=ChangeAction.DELETED,
            change_class=change_class,
            old_content_cid=_cid(f"{path}:old"),
            new_content_cid=ABSENCE_TOKEN,
            old_byte_length=4,
            new_byte_length=ABSENCE_TOKEN,
        )
    return ChangedArtifact(
        path=path,
        change_action=ChangeAction.MODIFIED,
        change_class=change_class,
        old_content_cid=_cid(f"{path}:old"),
        new_content_cid=_cid(f"{path}:new"),
        old_byte_length=4,
        new_byte_length=5,
    )


def build_positive_matrix_graph() -> ProofDependencyGraph:
    """Two-module reason-labeled graph for the positive matrix.

    Module A binds source, schema/interface, fixture, config, selector, policy,
    network policy, checked-spec, circuit, and keys to test/static/formal units
    and their aggregate.  Module B is an independent island so unrelated edits
    never cross-invalidate.
    """

    graph = ProofDependencyGraph()
    nodes: Sequence[tuple[str, DependencyNodeKind, str]] = (
        ("artifact/mod_a.py", DependencyNodeKind.ARTIFACT, "pkg/mod_a.py"),
        ("symbol/mod_a.fn", DependencyNodeKind.SYMBOL, "mod_a.fn"),
        ("unit/static_a", DependencyNodeKind.UNIT, "static-a"),
        ("unit/test_a", DependencyNodeKind.UNIT, "test-a"),
        ("unit/formal_a", DependencyNodeKind.UNIT, "formal-a"),
        ("aggregate/receipt_a", DependencyNodeKind.AGGREGATE, "receipt-a"),
        ("fixture/data_a", DependencyNodeKind.FIXTURE, "tests/fixtures/data_a.json"),
        ("config/env_a", DependencyNodeKind.CONFIG, "config/app.toml"),
        ("schema/api_a", DependencyNodeKind.SCHEMA, "pkg/api_a.py"),
        ("selector/tests_a", DependencyNodeKind.CONFIG, "pytest.ini"),
        ("policy/verify_a", DependencyNodeKind.POLICY, "policy/verify.json"),
        ("network/policy_a", DependencyNodeKind.POLICY, "policy/network.json"),
        ("spec/checked_a", DependencyNodeKind.ARTIFACT, "docs/checked_spec.md"),
        ("circuit/circ_a", DependencyNodeKind.ARTIFACT, "circuits/prove.circom"),
        ("key/proving_a", DependencyNodeKind.ARTIFACT, "keys/proving.key"),
        ("key/verification_a", DependencyNodeKind.ARTIFACT, "keys/verification.key"),
        ("lock/deps", DependencyNodeKind.CONFIG, "uv.lock"),
        ("tool/prover", DependencyNodeKind.ENVIRONMENT, "tool/prover-version"),
        ("artifact/mod_b.py", DependencyNodeKind.ARTIFACT, "pkg/mod_b.py"),
        ("unit/static_b", DependencyNodeKind.UNIT, "static-b"),
        ("unit/test_b", DependencyNodeKind.UNIT, "test-b"),
        ("aggregate/receipt_b", DependencyNodeKind.AGGREGATE, "receipt-b"),
    )
    for node_id, kind, label in nodes:
        graph.add_node(node_id, kind, label=label)

    edges: Sequence[tuple[str, str, DependencyEdgeType, str]] = (
        ("artifact/mod_a.py", "symbol/mod_a.fn", DependencyEdgeType.SOURCE_DEPENDS_ON, "src-sym-a"),
        ("symbol/mod_a.fn", "unit/static_a", DependencyEdgeType.CALLS, "sym-static-a"),
        ("symbol/mod_a.fn", "unit/test_a", DependencyEdgeType.TEST_COVERS, "sym-test-a"),
        ("artifact/mod_a.py", "unit/static_a", DependencyEdgeType.IMPORTS, "import-static-a"),
        ("unit/static_a", "unit/formal_a", DependencyEdgeType.PROOF_DEPENDS_ON, "static-formal-a"),
        ("unit/test_a", "unit/formal_a", DependencyEdgeType.PROOF_DEPENDS_ON, "test-formal-a"),
        ("unit/formal_a", "aggregate/receipt_a", DependencyEdgeType.AGGREGATE_CONTAINS, "formal-agg-a"),
        ("unit/test_a", "aggregate/receipt_a", DependencyEdgeType.AGGREGATE_CONTAINS, "test-agg-a"),
        ("fixture/data_a", "unit/test_a", DependencyEdgeType.FIXTURE_DEPENDS_ON, "fixture-test-a"),
        ("config/env_a", "unit/test_a", DependencyEdgeType.CONFIG_DEPENDS_ON, "config-test-a"),
        ("schema/api_a", "unit/static_a", DependencyEdgeType.SCHEMA_DEPENDS_ON, "schema-static-a"),
        ("selector/tests_a", "unit/test_a", DependencyEdgeType.CONFIG_DEPENDS_ON, "selector-test-a"),
        ("policy/verify_a", "unit/formal_a", DependencyEdgeType.CONFIG_DEPENDS_ON, "policy-formal-a"),
        ("network/policy_a", "unit/test_a", DependencyEdgeType.CONFIG_DEPENDS_ON, "net-test-a"),
        ("spec/checked_a", "unit/formal_a", DependencyEdgeType.PROOF_DEPENDS_ON, "spec-formal-a"),
        ("circuit/circ_a", "unit/formal_a", DependencyEdgeType.PROOF_DEPENDS_ON, "circuit-formal-a"),
        ("key/proving_a", "unit/formal_a", DependencyEdgeType.PROOF_DEPENDS_ON, "pk-formal-a"),
        ("key/verification_a", "unit/formal_a", DependencyEdgeType.PROOF_DEPENDS_ON, "vk-formal-a"),
        ("lock/deps", "unit/static_a", DependencyEdgeType.CONFIG_DEPENDS_ON, "lock-static-a"),
        ("lock/deps", "unit/static_b", DependencyEdgeType.CONFIG_DEPENDS_ON, "lock-static-b"),
        ("tool/prover", "unit/formal_a", DependencyEdgeType.CONFIG_DEPENDS_ON, "tool-formal-a"),
        ("artifact/mod_b.py", "unit/static_b", DependencyEdgeType.IMPORTS, "import-static-b"),
        ("unit/static_b", "aggregate/receipt_b", DependencyEdgeType.AGGREGATE_CONTAINS, "static-agg-b"),
        ("unit/test_b", "aggregate/receipt_b", DependencyEdgeType.AGGREGATE_CONTAINS, "test-agg-b"),
    )
    for from_id, to_id, edge_type, label in edges:
        graph.add_edge(from_id, to_id, edge_type, _reason(label))
    return graph


def positive_path_map() -> dict[str, tuple[str, ...]]:
    """Repository paths bound to matrix graph seeds."""

    return {
        "pkg/mod_a.py": ("artifact/mod_a.py",),
        "pkg/api_a.py": ("schema/api_a",),
        "tests/test_a.py": ("unit/test_a",),
        "tests/test_new.py": (),
        "tests/fixtures/data_a.json": ("fixture/data_a",),
        "config/app.toml": ("config/env_a",),
        "pytest.ini": ("selector/tests_a",),
        "policy/verify.json": ("policy/verify_a",),
        "policy/network.json": ("network/policy_a",),
        "docs/checked_spec.md": ("spec/checked_a",),
        "docs/guide.md": (),
        "circuits/prove.circom": ("circuit/circ_a",),
        "keys/proving.key": ("key/proving_a",),
        "keys/verification.key": ("key/verification_a",),
        "uv.lock": ("lock/deps",),
        "tool/prover-version": ("tool/prover",),
        "pkg/mod_b.py": ("artifact/mod_b.py",),
        "proof/schema.version": (),
        "canon/version": (),
    }


@dataclass(frozen=True, slots=True)
class MatrixCase:
    """One positive-matrix mutation with exact expected sets."""

    kind: str
    change_class: ChangeClass | None
    path: str | None
    action: ChangeAction = ChangeAction.MODIFIED
    seed_node_ids: tuple[str, ...] = ()
    added_unit_ids: tuple[str, ...] = ()
    removed_unit_ids: tuple[str, ...] = ()
    authorized_removal_unit_ids: tuple[str, ...] = ()
    policy_overrides: Mapping[str, Any] | None = None
    expected_invalidated: tuple[str, ...] = ()
    expected_reused: tuple[str, ...] = ()
    expected_added: tuple[str, ...] = ()
    expected_removed: tuple[str, ...] = ()
    expected_unauthorized: tuple[str, ...] = ()
    expected_fallback_required: bool = False
    expected_fallback_reasons: tuple[str, ...] = ()
    expected_docs_only: bool = False
    expected_seed_node_ids: tuple[str, ...] = ()
    explanation_unit_id: str | None = None
    explanation_invalidated: bool | None = None


def _sorted(*items: str) -> tuple[str, ...]:
    return tuple(sorted(items))


def matrix_cases() -> tuple[MatrixCase, ...]:
    """Exact expected outcomes for every positive mutation axis."""

    a_chain = _sorted(
        "aggregate/receipt_a",
        "unit/formal_a",
        "unit/static_a",
        "unit/test_a",
    )
    a_test_chain = _sorted(
        "aggregate/receipt_a",
        "unit/formal_a",
        "unit/test_a",
    )
    a_static_chain = _sorted(
        "aggregate/receipt_a",
        "unit/formal_a",
        "unit/static_a",
    )
    a_formal_chain = _sorted("aggregate/receipt_a", "unit/formal_a")
    b_all = _sorted(*MODULE_B_UNITS)
    all_units = _sorted(*KNOWN_UNITS)
    lock_chain = _sorted(
        "aggregate/receipt_a",
        "aggregate/receipt_b",
        "unit/formal_a",
        "unit/static_a",
        "unit/static_b",
    )
    b_static_chain = _sorted("aggregate/receipt_b", "unit/static_b")

    return (
        MatrixCase(
            kind="source_implementation",
            change_class=ChangeClass.SOURCE_IMPLEMENTATION,
            path="pkg/mod_a.py",
            expected_invalidated=a_chain,
            expected_reused=b_all,
            expected_seed_node_ids=("artifact/mod_a.py",),
            explanation_unit_id="unit/formal_a",
            explanation_invalidated=True,
        ),
        MatrixCase(
            kind="public_interface",
            change_class=ChangeClass.SOURCE_INTERFACE,
            path="pkg/api_a.py",
            expected_invalidated=a_static_chain,
            expected_reused=_sorted("unit/test_a", *MODULE_B_UNITS),
            expected_seed_node_ids=("schema/api_a",),
            explanation_unit_id="unit/static_a",
            explanation_invalidated=True,
        ),
        MatrixCase(
            kind="test_selector",
            change_class=ChangeClass.TEST_SELECTOR,
            path="pytest.ini",
            expected_invalidated=a_test_chain,
            expected_reused=_sorted("unit/static_a", *MODULE_B_UNITS),
            expected_seed_node_ids=("selector/tests_a",),
        ),
        MatrixCase(
            kind="test_source",
            change_class=ChangeClass.TEST_SOURCE,
            path="tests/test_a.py",
            expected_invalidated=a_test_chain,
            expected_reused=_sorted("unit/static_a", *MODULE_B_UNITS),
            expected_seed_node_ids=("unit/test_a",),
        ),
        MatrixCase(
            kind="test_add",
            change_class=None,
            path=None,
            # Explicit selected addition: prove the new unit; reuse all known.
            added_unit_ids=("unit/test_new",),
            expected_invalidated=(),
            expected_reused=all_units,
            expected_added=("unit/test_new",),
            expected_fallback_required=False,
            explanation_unit_id="unit/test_new",
            explanation_invalidated=False,
        ),
        MatrixCase(
            kind="test_delete_unauthorized",
            change_class=ChangeClass.TEST_SOURCE,
            path="tests/test_a.py",
            action=ChangeAction.DELETED,
            removed_unit_ids=("unit/test_a",),
            authorized_removal_unit_ids=(),
            # Seed unit/test_a is removed from the active set; dependents remain.
            expected_invalidated=_sorted("aggregate/receipt_a", "unit/formal_a"),
            expected_reused=_sorted("unit/static_a", *MODULE_B_UNITS),
            expected_removed=("unit/test_a",),
            expected_unauthorized=("unit/test_a",),
            expected_seed_node_ids=("unit/test_a",),
            explanation_unit_id="unit/test_a",
            explanation_invalidated=False,
        ),
        MatrixCase(
            kind="test_delete_authorized",
            change_class=ChangeClass.TEST_SOURCE,
            path="tests/test_a.py",
            action=ChangeAction.DELETED,
            removed_unit_ids=("unit/test_a",),
            authorized_removal_unit_ids=("unit/test_a",),
            expected_invalidated=_sorted("aggregate/receipt_a", "unit/formal_a"),
            expected_reused=_sorted("unit/static_a", *MODULE_B_UNITS),
            expected_removed=("unit/test_a",),
            expected_unauthorized=(),
            expected_seed_node_ids=("unit/test_a",),
            explanation_unit_id="unit/test_a",
            explanation_invalidated=False,
        ),
        MatrixCase(
            kind="fixture",
            change_class=ChangeClass.FIXTURE,
            path="tests/fixtures/data_a.json",
            expected_invalidated=a_test_chain,
            expected_reused=_sorted("unit/static_a", *MODULE_B_UNITS),
            expected_seed_node_ids=("fixture/data_a",),
        ),
        MatrixCase(
            kind="relevant_configuration",
            change_class=ChangeClass.CONFIGURATION,
            path="config/app.toml",
            expected_invalidated=a_test_chain,
            expected_reused=_sorted("unit/static_a", *MODULE_B_UNITS),
            expected_seed_node_ids=("config/env_a",),
        ),
        MatrixCase(
            kind="network_policy",
            change_class=ChangeClass.NETWORK_POLICY,
            path="policy/network.json",
            expected_invalidated=a_test_chain,
            expected_reused=_sorted("unit/static_a", *MODULE_B_UNITS),
            expected_seed_node_ids=("network/policy_a",),
        ),
        MatrixCase(
            kind="verification_policy",
            change_class=ChangeClass.POLICY,
            path="policy/verify.json",
            expected_invalidated=a_formal_chain,
            expected_reused=_sorted(
                "unit/static_a", "unit/test_a", *MODULE_B_UNITS
            ),
            expected_seed_node_ids=("policy/verify_a",),
        ),
        MatrixCase(
            kind="dependency_lock",
            change_class=ChangeClass.DEPENDENCY_LOCK,
            path="uv.lock",
            expected_invalidated=lock_chain,
            expected_reused=_sorted("unit/test_a", "unit/test_b"),
            expected_seed_node_ids=("lock/deps",),
            expected_fallback_required=False,
        ),
        MatrixCase(
            kind="dependency_lock_full_policy",
            change_class=ChangeClass.DEPENDENCY_LOCK,
            path="uv.lock",
            policy_overrides={"treat_dependency_lock_as_full_fallback": True},
            expected_invalidated=all_units,
            expected_reused=(),
            expected_seed_node_ids=("lock/deps",),
            expected_fallback_required=True,
            expected_fallback_reasons=("dependency_lock_policy",),
        ),
        MatrixCase(
            kind="tool_prover_version",
            change_class=ChangeClass.ENVIRONMENT,
            path="tool/prover-version",
            expected_invalidated=all_units,
            expected_reused=(),
            expected_seed_node_ids=("tool/prover",),
            expected_fallback_required=True,
            expected_fallback_reasons=("environment_changed",),
        ),
        MatrixCase(
            kind="circuit",
            change_class=ChangeClass.CIRCUIT,
            path="circuits/prove.circom",
            expected_invalidated=all_units,
            expected_reused=(),
            expected_seed_node_ids=("circuit/circ_a",),
            expected_fallback_required=True,
            expected_fallback_reasons=("circuit_changed",),
        ),
        MatrixCase(
            kind="proving_key",
            change_class=ChangeClass.PROVING_KEY,
            path="keys/proving.key",
            expected_invalidated=all_units,
            expected_reused=(),
            expected_seed_node_ids=("key/proving_a",),
            expected_fallback_required=True,
            expected_fallback_reasons=("proving_key_changed",),
        ),
        MatrixCase(
            kind="verification_key",
            change_class=ChangeClass.VERIFICATION_KEY,
            path="keys/verification.key",
            expected_invalidated=all_units,
            expected_reused=(),
            expected_seed_node_ids=("key/verification_a",),
            expected_fallback_required=True,
            expected_fallback_reasons=("verification_key_changed",),
        ),
        MatrixCase(
            kind="proof_schema",
            change_class=None,
            path=None,
            policy_overrides={"proof_schema_changed": True},
            expected_invalidated=all_units,
            expected_reused=(),
            expected_fallback_required=True,
            expected_fallback_reasons=("proof_schema_changed",),
        ),
        MatrixCase(
            kind="canonicalization",
            change_class=ChangeClass.CANONICALIZATION,
            path="canon/version",
            # Unmapped + full-fallback class forces full proof of every unit.
            expected_invalidated=all_units,
            expected_reused=(),
            expected_fallback_required=True,
            expected_fallback_reasons=(
                "canonicalization_changed",
                "unmapped_relevant_change",
            ),
        ),
        MatrixCase(
            kind="checked_specification",
            change_class=ChangeClass.CHECKED_SPECIFICATION,
            path="docs/checked_spec.md",
            expected_invalidated=a_formal_chain,
            expected_reused=_sorted(
                "unit/static_a", "unit/test_a", *MODULE_B_UNITS
            ),
            expected_seed_node_ids=("spec/checked_a",),
            explanation_unit_id="unit/formal_a",
            explanation_invalidated=True,
        ),
        MatrixCase(
            kind="ordinary_documentation",
            change_class=ChangeClass.ORDINARY_DOCUMENTATION,
            path="docs/guide.md",
            expected_invalidated=(),
            expected_reused=all_units,
            expected_docs_only=True,
            expected_seed_node_ids=(),
            explanation_unit_id="unit/formal_a",
            explanation_invalidated=False,
        ),
        MatrixCase(
            kind="independent_module",
            change_class=ChangeClass.SOURCE_IMPLEMENTATION,
            path="pkg/mod_b.py",
            expected_invalidated=b_static_chain,
            expected_reused=_sorted("unit/test_b", *MODULE_A_UNITS),
            expected_seed_node_ids=("artifact/mod_b.py",),
            explanation_unit_id="unit/formal_a",
            explanation_invalidated=False,
        ),
        MatrixCase(
            kind="two_module",
            change_class=None,
            path=None,
            seed_node_ids=("artifact/mod_a.py", "artifact/mod_b.py"),
            expected_invalidated=_sorted(
                "aggregate/receipt_a",
                "aggregate/receipt_b",
                "unit/formal_a",
                "unit/static_a",
                "unit/static_b",
                "unit/test_a",
            ),
            expected_reused=("unit/test_b",),
            expected_seed_node_ids=("artifact/mod_a.py", "artifact/mod_b.py"),
        ),
    )


def _run_case(case: MatrixCase) -> Any:
    graph = build_positive_matrix_graph()
    policy = sample_invalidation_policy(**(case.policy_overrides or {}))
    artifacts: tuple[ChangedArtifact, ...] = ()
    if case.change_class is not None and case.path is not None:
        artifacts = (
            _changed(case.path, case.change_class, action=case.action),
        )
    return compute_invalidation_closure(
        graph,
        changed_node_ids=case.seed_node_ids,
        changed_artifacts=artifacts,
        path_to_node_ids=positive_path_map(),
        known_unit_ids=KNOWN_UNITS,
        added_unit_ids=case.added_unit_ids,
        removed_unit_ids=case.removed_unit_ids,
        authorized_removal_unit_ids=case.authorized_removal_unit_ids,
        policy=policy,
    )


def _assert_case(case: MatrixCase, closure: Any) -> None:
    assert tuple(closure.invalidated_unit_ids) == case.expected_invalidated, case.kind
    assert tuple(closure.preserved_unit_ids) == case.expected_reused, case.kind
    assert tuple(closure.added_unit_ids) == case.expected_added, case.kind
    assert tuple(closure.removed_unit_ids) == case.expected_removed, case.kind
    assert (
        tuple(closure.unauthorized_removal_unit_ids) == case.expected_unauthorized
    ), case.kind
    assert closure.full_fallback.required is case.expected_fallback_required, case.kind
    if case.expected_fallback_required:
        assert tuple(closure.full_fallback.reasons) == case.expected_fallback_reasons, (
            case.kind
        )
    else:
        assert closure.full_fallback.reasons == (), case.kind
    assert closure.docs_only is case.expected_docs_only, case.kind
    assert tuple(closure.seed_node_ids) == case.expected_seed_node_ids, case.kind

    # Disposition partition is exact and non-overlapping.
    by_kind: dict[str, set[str]] = {
        "invalidate": set(),
        "preserve": set(),
        "prove_new": set(),
        "remove_requires_authorization": set(),
        "remove_authorized": set(),
    }
    for disposition in closure.dispositions:
        by_kind[disposition.kind.value].add(disposition.unit_id)
    assert by_kind["invalidate"] == set(case.expected_invalidated), case.kind
    assert by_kind["preserve"] == set(case.expected_reused), case.kind
    assert by_kind["prove_new"] == set(case.expected_added), case.kind
    if case.expected_unauthorized:
        assert by_kind["remove_requires_authorization"] == set(
            case.expected_unauthorized
        ), case.kind
        assert by_kind["remove_authorized"] == set(), case.kind
    else:
        assert by_kind["remove_requires_authorization"] == set(), case.kind
        assert by_kind["remove_authorized"] == set(case.expected_removed), case.kind

    # Reused units are never also invalidated/added/removed.
    reused = set(case.expected_reused)
    assert not (reused & set(case.expected_invalidated)), case.kind
    assert not (reused & set(case.expected_added)), case.kind
    assert not (reused & set(case.expected_removed)), case.kind


# ---------------------------------------------------------------------------
# Surface and fixture alignment
# ---------------------------------------------------------------------------


def test_evidence_subset() -> None:
    assert EVIDENCE_SUBSET == "ips/invalidation-positive@1"


def test_matrix_covers_required_mutation_contract() -> None:
    kinds = {case.kind for case in matrix_cases()}
    for required in (
        "test_selector",
        "fixture",
        "relevant_configuration",
        "network_policy",
        "verification_policy",
        "dependency_lock",
        "tool_prover_version",
        "proof_schema",
        "canonicalization",
        "checked_specification",
        "source_implementation",
        "test_source",
        "circuit",
        "proving_key",
        "verification_key",
        "ordinary_documentation",
    ):
        assert required in kinds
    assert tuple(case.kind for case in matrix_cases()) == MATRIX_MUTATION_KINDS


def test_fixture_corpus_scenario_kinds_are_matrix_aligned() -> None:
    corpus = generate_corpus()
    assert tuple(corpus["scenario_kinds"]) == REQUIRED_SCENARIO_KINDS
    kinds = {item["kind"] for item in corpus["scenarios"]}
    # Every fixture scenario kind is either directly covered or a lifecycle
    # scenario (branch/merge/rollback/corruption/graph_manifest) deferred to
    # IPS-047, or folded into a matrix synonym below.
    covered_or_deferred = {
        "source_implementation",
        "public_interface",
        "test_selector",
        "test_source",
        "test_add",
        "test_delete",
        "fixture",
        "relevant_configuration",
        "network_policy",
        "verification_policy",
        "dependency_lock",
        "tool_prover_version",
        "circuit_key",
        "proof_schema",
        "canonicalization",
        "checked_specification_document",
        "ordinary_documentation",
        "graph_manifest",
        "branch",
        "merge",
        "rollback",
        "corruption",
        "independent_module",
    }
    assert kinds <= covered_or_deferred
    assert "test_selector" in kinds
    assert "checked_specification_document" in kinds
    assert "ordinary_documentation" in kinds
    # Fixture parent chain is deterministic.
    first = generate_corpus()
    second = generate_corpus()
    assert first["corpus_cid"] == second["corpus_cid"]
    assert [item["scenario_cid"] for item in first["scenarios"]] == [
        item["scenario_cid"] for item in second["scenarios"]
    ]


# ---------------------------------------------------------------------------
# Exact matrix evaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", matrix_cases(), ids=lambda c: c.kind)
def test_positive_matrix_exact_sets(case: MatrixCase) -> None:
    first = _run_case(case)
    second = _run_case(case)
    _assert_case(case, first)
    # Deterministic roots / closure CIDs.
    assert first.closure_cid() == second.closure_cid(), case.kind
    assert first.to_canonical() == second.to_canonical(), case.kind


@pytest.mark.parametrize("case", matrix_cases(), ids=lambda c: c.kind)
def test_positive_matrix_reason_labeled_explanations(case: MatrixCase) -> None:
    graph = build_positive_matrix_graph()
    closure = _run_case(case)
    unit_id = case.explanation_unit_id or (
        case.expected_invalidated[0]
        if case.expected_invalidated
        else case.expected_reused[0]
    )
    explanation = explain_invalidation(graph, unit_id, closure)
    replay = explain_invalidation(graph, unit_id, closure)
    assert explanation.to_canonical() == replay.to_canonical(), case.kind
    assert explanation.explanation_cid() == replay.explanation_cid(), case.kind
    if case.explanation_invalidated is not None:
        assert explanation.invalidated is case.explanation_invalidated, case.kind
    if explanation.invalidated:
        assert explanation.disposition is UnitDispositionKind.INVALIDATE, case.kind
        assert explanation.summary
        # Reason-labeled paths when a graph walk is possible.
        if closure.seed_node_ids and unit_id not in closure.seed_node_ids:
            # Paths may be empty under full fallback; when present they are labeled.
            for path in explanation.paths:
                assert path.edge_types
                assert path.reason_cids
                assert all(isinstance(cid, str) and cid for cid in path.reason_cids)
    if unit_id in case.expected_added:
        assert explanation.disposition is UnitDispositionKind.PROVE_NEW, case.kind
        assert "must be proven" in explanation.summary
    if unit_id in case.expected_unauthorized:
        assert (
            explanation.disposition
            is UnitDispositionKind.REMOVE_REQUIRES_AUTHORIZATION
        ), case.kind
        assert "authorization" in explanation.summary
    if unit_id in case.expected_removed and not case.expected_unauthorized:
        assert explanation.disposition is UnitDispositionKind.REMOVE_AUTHORIZED, (
            case.kind
        )
    if case.expected_docs_only and unit_id in case.expected_reused:
        assert explanation.invalidated is False, case.kind
        assert "ordinary documentation" in explanation.summary


def test_deleted_tests_require_authorization() -> None:
    unauthorized = next(
        c for c in matrix_cases() if c.kind == "test_delete_unauthorized"
    )
    authorized = next(
        c for c in matrix_cases() if c.kind == "test_delete_authorized"
    )
    bad = _run_case(unauthorized)
    good = _run_case(authorized)
    assert bad.unauthorized_removal_unit_ids == ("unit/test_a",)
    assert bad.complete is False
    assert (
        bad.disposition_for("unit/test_a").kind
        is UnitDispositionKind.REMOVE_REQUIRES_AUTHORIZATION
    )
    assert good.unauthorized_removal_unit_ids == ()
    assert (
        good.disposition_for("unit/test_a").kind
        is UnitDispositionKind.REMOVE_AUTHORIZED
    )


def test_added_selected_tests_are_proven() -> None:
    case = next(c for c in matrix_cases() if c.kind == "test_add")
    closure = _run_case(case)
    assert "unit/test_new" in closure.added_unit_ids
    assert set(closure.preserved_unit_ids) == set(KNOWN_UNITS)
    assert closure.invalidated_unit_ids == ()
    assert (
        closure.disposition_for("unit/test_new").kind is UnitDispositionKind.PROVE_NEW
    )
    assert "test_added" in closure.triggers
    explanation = explain_invalidation(
        build_positive_matrix_graph(), "unit/test_new", closure
    )
    assert explanation.disposition is UnitDispositionKind.PROVE_NEW
    assert "must be proven" in explanation.summary

    # Unmapped newly-added test source is still prove_new and never silently
    # reused, even when the change broadens to full fallback.
    graph = build_positive_matrix_graph()
    unmapped_add = compute_invalidation_closure(
        graph,
        changed_artifacts=(
            _changed(
                "tests/test_new.py",
                ChangeClass.TEST_SOURCE,
                action=ChangeAction.ADDED,
            ),
        ),
        known_unit_ids=KNOWN_UNITS,
        added_unit_ids=("unit/test_new",),
        path_to_node_ids={},
        policy=sample_invalidation_policy(),
    )
    assert (
        unmapped_add.disposition_for("unit/test_new").kind
        is UnitDispositionKind.PROVE_NEW
    )
    assert "unit/test_new" not in unmapped_add.preserved_unit_ids
    assert unmapped_add.full_fallback.required is True


def test_ordinary_documentation_reuse_remains_valid() -> None:
    case = next(c for c in matrix_cases() if c.kind == "ordinary_documentation")
    closure = _run_case(case)
    assert closure.docs_only is True
    assert closure.invalidated_unit_ids == ()
    assert set(closure.preserved_unit_ids) == set(KNOWN_UNITS)
    assert closure.full_fallback.required is False
    # Planner maps preserved+admitted units to reuse under an accepted parent.
    parent = ParentSealContext(
        seal_cid=_cid("parent-seal"),
        repository_state_cid=_cid("old-state"),
        source_root_cid=_cid("old-source"),
    )
    units = tuple(
        UnitPlanningInput(
            unit_id=unit_id,
            preserved=True,
            cache_key_complete=True,
            admitted=True,
            candidate_present=True,
        )
        for unit_id in KNOWN_UNITS
    )
    plan = create_incremental_plan(
        parent,
        _cid("old-state"),
        _cid("new-state-docs"),
        units=units,
    )
    assert set(plan.reusable_unit_ids) == set(KNOWN_UNITS)
    assert plan.invalidated_unit_ids == ()


def test_checked_specification_invalidates_consumers_only() -> None:
    case = next(c for c in matrix_cases() if c.kind == "checked_specification")
    closure = _run_case(case)
    assert "unit/formal_a" in closure.invalidated_unit_ids
    assert "aggregate/receipt_a" in closure.invalidated_unit_ids
    # Unrelated module and non-consumer local units remain reusable.
    assert "unit/static_a" in closure.preserved_unit_ids
    assert "unit/test_a" in closure.preserved_unit_ids
    assert MODULE_B_UNITS <= set(closure.preserved_unit_ids)
    explanation = explain_invalidation(
        build_positive_matrix_graph(), "unit/formal_a", closure
    )
    assert explanation.invalidated is True
    assert "spec/checked_a" in explanation.seed_node_ids


def test_trust_schema_key_circuit_force_full_proof() -> None:
    full_kinds = {
        "tool_prover_version",
        "circuit",
        "proving_key",
        "verification_key",
        "proof_schema",
        "canonicalization",
        "dependency_lock_full_policy",
    }
    for case in matrix_cases():
        if case.kind not in full_kinds:
            continue
        closure = _run_case(case)
        assert closure.full_fallback.required is True, case.kind
        assert set(closure.invalidated_unit_ids) == set(KNOWN_UNITS), case.kind
        assert closure.preserved_unit_ids == (), case.kind
        assert set(case.expected_fallback_reasons) <= set(
            closure.full_fallback.reasons
        ), case.kind

    # Closed full-fallback change classes never narrow reuse.
    for change_class in sorted(FULL_FALLBACK_CHANGE_CLASSES):
        decision = classify_full_fallback(
            change_classes=(change_class,),
            policy=sample_invalidation_policy(),
        )
        assert decision.required is True, change_class


def test_unrelated_module_edit_never_invalidates_module_a() -> None:
    case = next(c for c in matrix_cases() if c.kind == "independent_module")
    closure = _run_case(case)
    assert set(closure.invalidated_unit_ids) <= MODULE_B_UNITS
    assert MODULE_A_UNITS <= set(closure.preserved_unit_ids)


def test_dependency_roots_repeat_deterministically() -> None:
    first = build_positive_matrix_graph()
    second = build_positive_matrix_graph()
    assert first.graph_cid() == second.graph_cid()
    root_a = compute_dependency_root(first, "unit/formal_a")
    root_b = compute_dependency_root(second, "unit/formal_a")
    assert root_a.root_cid() == root_b.root_cid()
    assert root_a.prerequisite_node_ids == root_b.prerequisite_node_ids
    # Insertion-order independence: rebuild with reversed edge insertion.
    reversed_graph = ProofDependencyGraph()
    for node in reversed(first.nodes()):
        reversed_graph.add_node(
            node.node_id, node.kind, label=node.label, truncated=node.truncated
        )
    for edge in reversed(first.edges()):
        reversed_graph.add_edge(
            edge.from_id, edge.to_id, edge.edge_type, edge.reason_cid
        )
    assert reversed_graph.graph_cid() == first.graph_cid()
    assert (
        compute_dependency_root(reversed_graph, "unit/formal_a").root_cid()
        == root_a.root_cid()
    )


def test_fixture_full_fallback_kinds_align_with_engine() -> None:
    """Fixture corpus full-fallback decisions remain consistent with policy."""

    corpus = generate_corpus()
    full_kinds = {
        item["kind"]
        for item in corpus["scenarios"]
        if item["full_fallback_decision"]["required"]
    }
    # Engine-forced classes covered by the positive matrix.
    assert "circuit_key" in full_kinds
    assert "canonicalization" in full_kinds
    assert "proof_schema" in full_kinds
    assert "tool_prover_version" in full_kinds
    # Ordinary documentation must never require full fallback in the corpus.
    docs = next(
        item
        for item in corpus["scenarios"]
        if item["kind"] == "ordinary_documentation"
    )
    assert docs["full_fallback_decision"]["required"] is False
    checked = next(
        item
        for item in corpus["scenarios"]
        if item["kind"] == "checked_specification_document"
    )
    assert checked["full_fallback_decision"]["required"] is False


def test_planner_reuse_matches_preserved_sets_for_source_edit() -> None:
    case = next(c for c in matrix_cases() if c.kind == "source_implementation")
    closure = _run_case(case)
    parent = ParentSealContext(
        seal_cid=_cid("parent-seal-src"),
        repository_state_cid=_cid("old-state-src"),
        source_root_cid=_cid("old-source-src"),
    )
    units = []
    for unit_id in KNOWN_UNITS:
        preserved = unit_id in closure.preserved_unit_ids
        units.append(
            UnitPlanningInput(
                unit_id=unit_id,
                preserved=preserved,
                invalidated=not preserved,
                cache_key_complete=True,
                admitted=preserved,
                candidate_present=True,
            )
        )
    plan = create_incremental_plan(
        parent,
        _cid("old-state-src"),
        _cid("new-state-src"),
        units=tuple(units),
    )
    assert set(plan.reusable_unit_ids) == set(case.expected_reused)
    assert set(plan.invalidated_unit_ids) == set(case.expected_invalidated)


def test_policy_invalidation_policy_cid_is_stable() -> None:
    left = InvalidationPolicy()
    right = InvalidationPolicy()
    assert left.policy_cid() == right.policy_cid()
    mutated = sample_invalidation_policy(proof_schema_changed=True)
    assert mutated.policy_cid() != left.policy_cid()
