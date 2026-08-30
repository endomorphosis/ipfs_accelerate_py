"""Independent contract tests for SPAR-013 ProgramPartitionCandidate@1 generators."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_generators import (
    ADVISORY_GENERATOR_KINDS,
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    DUCKLAKE_IS_AUTHORITY,
    GENERATOR_ORDER,
    GOAL_ID,
    GRAPH_CLUSTER_EDGE_KINDS,
    IDENTITY_EXCLUDED_FIELDS,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    PARTITION_CAN_AUTHORIZE_COMPLETION,
    PARTITION_CAN_AUTHORIZE_TRANSITION,
    PARTITION_CAN_CREATE_AUTHORITY,
    PARTITION_CONTRACT_VERSION,
    PARTITION_GENERATION_RECEIPT_INTERFACE,
    PROGRAM,
    PROGRAM_PARTITION_CANDIDATE_INTERFACE,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TESTS_PROOFS_EDGE_KINDS,
    TEST_PASS_IS_NOT_COMPLETION,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    ConstraintClass,
    GeneratorKind,
    PartitionCutEdge,
    PartitionGenerationReceipt,
    PartitionGeneratorError,
    ProgramPartitionCandidate,
    assert_not_competing_capsule_family,
    decode_canonical_receipt,
    encode_canonical_receipt,
    generate_partition_candidates,
    partition_cid_profile,
    provider_free_exports,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "partition_generators.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/partition_generators.py",
    "test/api/semantic_refactoring/test_partition_generators.py",
)
PROTECTED_PATHS = (
    ".gitignore",
    "benchmarks/agent_supervisor/semantic_refactoring/preregistration.json",
    "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json",
    "config/semantic_preserving_autonomous_remodularization_dependencies.seal.json",
    "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_PLAN.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.objectives.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.todo.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/authority_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/benchmark_preregistration.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/dynamic_python_risk_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/identity_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/interface_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/overlap_gap_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/repository_baseline.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/rollout_baseline.json",
    "scripts/materialize_semantic_preserving_remodularization_program.py",
    "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py",
    "scripts/validate_semantic_preserving_remodularization_board.py",
    "scripts/validate_semantic_preserving_remodularization_dependencies.py",
    "test/api/semantic_refactoring/test_bootstrap_controls.py",
)
CAPSULE_TYPES = (
    "FunctionSemanticCapsule",
    "MethodSemanticCapsule",
    "ClassSemanticCapsule",
    "TopLevelBlockCapsule",
    "ModuleSemanticCapsule",
    "PackageSemanticCapsule",
    "CallsiteSemanticCapsule",
    "StateOwnerCapsule",
    "RegistrationCapsule",
    "ResourceLifecycleCapsule",
)
TREE_ID = "fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051"


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _component(
    *member_ids: str,
    cyclic: bool = False,
    oversized: bool = False,
    state_owner_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    members = tuple(sorted(member_ids))
    identity = {
        "cyclic": cyclic,
        "member_ids": list(members),
        "oversized": oversized,
        "state_owner_ids": list(state_owner_ids),
    }
    return {
        "scc_id": cid_for_dag_json(identity),
        "member_ids": list(members),
        "cyclic": cyclic,
        "oversized": oversized,
        "state_owner_ids": list(state_owner_ids),
    }


def _snapshot(
    components: tuple[dict[str, Any], ...],
    condensation: tuple[dict[str, str], ...] = (),
) -> dict[str, Any]:
    payload = {
        "components": list(components),
        "condensation_edges": list(condensation),
    }
    return {
        "snapshot_cid": cid_for_dag_json(payload),
        "components": list(components),
        "condensation_edges": list(condensation),
    }


def _evidence(**overrides: Any) -> dict[str, Any]:
    left_right = _component("node:left", "node:right", cyclic=True)
    leaf = _component("node:leaf")
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "analyzer_id": ANALYZER_ID,
        "scc_snapshot": _snapshot(
            (left_right, leaf),
            (
                {
                    "source_scc_id": left_right["scc_id"],
                    "target_scc_id": leaf["scc_id"],
                    "witness_kind": "calls",
                },
            ),
        ),
    }
    fields.update(overrides)
    return fields


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-013"
    assert GOAL_ID == "SPAR-G032"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert PROGRAM_PARTITION_CANDIDATE_INTERFACE == "ProgramPartitionCandidate@1"
    assert PARTITION_GENERATION_RECEIPT_INTERFACE == "PartitionGenerationReceipt@1"
    assert PARTITION_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("partition_generators@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "partition orchestration"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert PARTITION_CAN_AUTHORIZE_COMPLETION is False
    assert PARTITION_CAN_AUTHORIZE_TRANSITION is False
    assert PARTITION_CAN_CREATE_AUTHORITY is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    profile = partition_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "ProgramPartitionCandidate" in names
    assert "PartitionGenerationReceipt" in names
    assert "PartitionCutEdge" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "ProgramPartitionCandidate" in exports
    assert "generate_partition_candidates" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_scc_generator_keeps_cyclic_component_together() -> None:
    receipt = generate_partition_candidates(_evidence())
    scc_candidates = [
        item for item in receipt.candidates if item.generator_kind == GeneratorKind.SCC.value
    ]
    cyclic = next(item for item in scc_candidates if "node:left" in item.member_ids)
    assert cyclic.member_ids == ("node:left", "node:right")
    assert cyclic.admitted is True
    assert cyclic.advisory is False
    assert cyclic.can_authorize_completion is False
    leaf = next(item for item in scc_candidates if item.member_ids == ("node:leaf",))
    assert leaf.admitted is True
    assert any(
        item.kind == "calls" and item.constraint_class == "hard"
        for item in cyclic.cut_edges
    )


def test_oversized_scc_is_rejected_not_extracted() -> None:
    oversized = _component("node:a", "node:b", "node:c", cyclic=True, oversized=True)
    receipt = generate_partition_candidates(
        _evidence(scc_snapshot=_snapshot((oversized,)))
    )
    scc_candidates = [
        item for item in receipt.candidates if item.generator_kind == GeneratorKind.SCC.value
    ]
    assert len(scc_candidates) == 1
    candidate = scc_candidates[0]
    assert candidate.admitted is False
    assert ConstraintClass.OVERSIZED_CYCLE.value in candidate.hard_constraint_violations
    assert candidate.candidate_cid in receipt.rejected_candidate_cids


def test_state_unique_owner_closes_alias_set_and_scc() -> None:
    left_right = _component("node:left", "node:right", cyclic=True)
    other = _component("node:other")
    receipt = generate_partition_candidates(
        _evidence(
            scc_snapshot=_snapshot((left_right, other)),
            state_ownership={
                "graph_cid": _cid("state-graph"),
                "owners": [
                    {
                        "owner_id": "owner:cache",
                        "uniqueness": "unique",
                        "owning_symbol_id": "node:left",
                        "alias_set_id": "alias:node:left",
                    }
                ],
                "alias_sets": [
                    {
                        "alias_set_id": "alias:node:left",
                        "representative_id": "node:left",
                        "member_ids": ["node:left", "node:other"],
                        "unresolved": False,
                    }
                ],
                "unresolved": [],
            },
        )
    )
    state_candidates = [
        item
        for item in receipt.candidates
        if item.generator_kind == GeneratorKind.STATE.value
    ]
    assert len(state_candidates) == 1
    candidate = state_candidates[0]
    assert "node:left" in candidate.member_ids
    assert "node:right" in candidate.member_ids
    assert "node:other" in candidate.member_ids
    assert candidate.state_owner_ids == ("owner:cache",)
    assert candidate.admitted is True


def test_cochange_cannot_split_an_scc() -> None:
    receipt = generate_partition_candidates(
        _evidence(
            cochange_edges=[
                {
                    "source_id": "node:left",
                    "target_id": "node:leaf",
                    "evidence_class": "runtime_observation",
                }
            ]
        )
    )
    cochange = [
        item
        for item in receipt.candidates
        if item.generator_kind == GeneratorKind.COCHANGE.value
    ]
    assert len(cochange) == 1
    assert cochange[0].member_ids == ("node:leaf", "node:left", "node:right")
    assert cochange[0].admitted is True


def test_tests_proofs_clusters_exact_test_edges() -> None:
    receipt = generate_partition_candidates(
        _evidence(
            graph_view={
                "graph_view_cid": _cid("graph-view"),
                "edges": [
                    {
                        "source_id": "node:leaf",
                        "target_id": "node:left",
                        "kind": "tests",
                        "evidence_class": "test",
                        "confidence": "exact",
                    }
                ],
            }
        )
    )
    clustered = [
        item
        for item in receipt.candidates
        if item.generator_kind == GeneratorKind.TESTS_PROOFS.value
    ]
    assert len(clustered) == 1
    assert clustered[0].member_ids == ("node:leaf", "node:left", "node:right")
    assert clustered[0].evidence_class == "test"
    assert TESTS_PROOFS_EDGE_KINDS == frozenset({"tests", "proves"})


def test_graph_structural_edges_form_clusters() -> None:
    receipt = generate_partition_candidates(
        _evidence(
            graph_view={
                "graph_view_cid": _cid("graph-view"),
                "edges": [
                    {
                        "source_id": "node:leaf",
                        "target_id": "node:left",
                        "kind": "contains",
                        "evidence_class": "exact_static_fact",
                        "confidence": "exact",
                    }
                ],
            }
        )
    )
    clustered = [
        item
        for item in receipt.candidates
        if item.generator_kind == GeneratorKind.GRAPH.value
    ]
    assert len(clustered) == 1
    assert clustered[0].member_ids == ("node:leaf", "node:left", "node:right")
    assert "contains" in GRAPH_CLUSTER_EDGE_KINDS


def test_contract_groups_by_subject_and_records_consumers() -> None:
    receipt = generate_partition_candidates(
        _evidence(
            compatibility={
                "inventory_cid": _cid("compat"),
                "obligations": [
                    {
                        "obligation_id": "obl:import",
                        "subject_id": "node:leaf",
                        "consumer_id": "consumer:cli",
                        "kind": "import_path",
                    }
                ],
                "consumers": [
                    {
                        "consumer_id": "consumer:cli",
                        "module_name": "pkg.cli",
                        "role": "cli",
                    }
                ],
            }
        )
    )
    contracts = [
        item
        for item in receipt.candidates
        if item.generator_kind == GeneratorKind.CONTRACT.value
    ]
    assert len(contracts) == 1
    assert contracts[0].member_ids == ("node:leaf",)
    assert contracts[0].obligation_ids == ("obl:import",)
    assert contracts[0].consumer_ids == ("consumer:cli",)
    assert contracts[0].admitted is True


def test_projection_clusters_remain_advisory_and_never_admitted() -> None:
    pin = _cid("projection-pin")
    receipt = generate_partition_candidates(
        _evidence(
            projection_clusters=[
                {
                    "cluster_id": "proj:1",
                    "member_ids": ["node:left", "node:leaf"],
                    "model_pin_cid": pin,
                }
            ]
        )
    )
    projections = [
        item
        for item in receipt.candidates
        if item.generator_kind == GeneratorKind.PROJECTION.value
    ]
    assert len(projections) == 1
    candidate = projections[0]
    assert candidate.advisory is True
    assert candidate.admitted is False
    assert candidate.projection_is_authority is False
    assert ConstraintClass.PROJECTION_ADVISORY.value in candidate.hard_constraint_violations
    assert candidate.evidence_class == "vector_candidate"
    assert candidate.candidate_cid in receipt.advisory_candidate_cids
    assert GeneratorKind.PROJECTION.value in ADVISORY_GENERATOR_KINDS


def test_vector_cochange_cannot_admit() -> None:
    receipt = generate_partition_candidates(
        _evidence(
            cochange_edges=[
                {
                    "source_id": "node:leaf",
                    "target_id": "node:leaf",
                    "evidence_class": "vector_candidate",
                }
            ]
        )
    )
    cochange = [
        item
        for item in receipt.candidates
        if item.generator_kind == GeneratorKind.COCHANGE.value
    ]
    assert cochange
    assert all(item.admitted is False for item in cochange)
    assert all(
        ConstraintClass.VECTOR_OR_MODEL.value in item.hard_constraint_violations
        for item in cochange
    )


def test_multiple_generators_emit_multiple_candidates() -> None:
    receipt = generate_partition_candidates(
        _evidence(
            graph_view={
                "graph_view_cid": _cid("graph-view"),
                "edges": [
                    {
                        "source_id": "node:leaf",
                        "target_id": "node:left",
                        "kind": "tests",
                        "evidence_class": "test",
                        "confidence": "exact",
                    },
                    {
                        "source_id": "node:leaf",
                        "target_id": "node:left",
                        "kind": "contains",
                        "evidence_class": "exact_static_fact",
                        "confidence": "exact",
                    },
                ],
            },
            compatibility={
                "inventory_cid": _cid("compat"),
                "obligations": [
                    {
                        "obligation_id": "obl:api",
                        "subject_id": "node:leaf",
                    }
                ],
                "consumers": [],
            },
            cochange_edges=[
                {
                    "source_id": "node:leaf",
                    "target_id": "node:leaf",
                    "evidence_class": "runtime_observation",
                }
            ],
            projection_clusters=[
                {"member_ids": ["node:leaf"], "model_pin_cid": _cid("pin")}
            ],
        )
    )
    kinds = {item.generator_kind for item in receipt.candidates}
    assert kinds >= {
        GeneratorKind.SCC.value,
        GeneratorKind.CONTRACT.value,
        GeneratorKind.TESTS_PROOFS.value,
        GeneratorKind.GRAPH.value,
        GeneratorKind.COCHANGE.value,
        GeneratorKind.PROJECTION.value,
    }
    assert receipt.admitted_candidates
    assert receipt.advisory_candidates
    assert all(item.can_authorize_transition is False for item in receipt.candidates)
    assert GENERATOR_ORDER[0] == GeneratorKind.SCC.value


def test_missing_scc_snapshot_fails_closed() -> None:
    with pytest.raises(PartitionGeneratorError, match="scc_snapshot is required"):
        generate_partition_candidates({"tree_id": TREE_ID, "analyzer_id": ANALYZER_ID})


def test_unknown_member_fails_closed() -> None:
    with pytest.raises(PartitionGeneratorError, match="unknown SCC member"):
        generate_partition_candidates(
            _evidence(
                graph_view={
                    "graph_view_cid": _cid("graph-view"),
                    "edges": [
                        {
                            "source_id": "node:ghost",
                            "target_id": "node:leaf",
                            "kind": "contains",
                            "evidence_class": "exact_static_fact",
                            "confidence": "exact",
                        }
                    ],
                }
            )
        )


def test_identity_excludes_observational_fields() -> None:
    candidate = ProgramPartitionCandidate(
        tree_id=TREE_ID,
        generator_kind=GeneratorKind.SCC,
        member_ids=("node:leaf",),
        scc_ids=(_cid("scc"),),
        admitted=True,
    )
    payload = candidate.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    restored = ProgramPartitionCandidate.from_dict(payload)
    assert restored == candidate
    assert restored.candidate_cid == candidate.candidate_cid
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(PartitionGeneratorError, match="observational"):
        ProgramPartitionCandidate.from_dict(dirty)


def test_candidate_cannot_claim_authority_flags() -> None:
    candidate = ProgramPartitionCandidate(
        tree_id=TREE_ID,
        generator_kind=GeneratorKind.SCC,
        member_ids=("node:leaf",),
        admitted=True,
    )
    payload = candidate.to_dict()
    payload["can_authorize_completion"] = True
    payload["candidate_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "candidate_cid"}
    )
    with pytest.raises(PartitionGeneratorError, match="can_authorize_completion"):
        ProgramPartitionCandidate.from_dict(payload)
    with pytest.raises(PartitionGeneratorError, match="hard-constraint"):
        ProgramPartitionCandidate(
            tree_id=TREE_ID,
            generator_kind=GeneratorKind.SCC,
            member_ids=("node:leaf",),
            admitted=True,
            hard_constraint_violations=(ConstraintClass.OVERSIZED_CYCLE.value,),
        )


def test_projection_candidate_constructor_forces_advisory() -> None:
    candidate = ProgramPartitionCandidate(
        tree_id=TREE_ID,
        generator_kind=GeneratorKind.PROJECTION,
        member_ids=("node:leaf",),
        evidence_class="vector_candidate",
        admitted=True,
        advisory=False,
    )
    assert candidate.admitted is False
    assert candidate.advisory is True
    assert ConstraintClass.PROJECTION_ADVISORY.value in candidate.hard_constraint_violations


def test_receipt_identity_is_deterministic_and_round_trips() -> None:
    evidence = _evidence()
    first = generate_partition_candidates(evidence)
    second = generate_partition_candidates(evidence)
    assert first.receipt_cid == second.receipt_cid
    encoded = encode_canonical_receipt(first)
    restored = decode_canonical_receipt(encoded)
    assert restored == first
    assert restored.receipt_cid == first.receipt_cid
    assert first.can_authorize_completion is False
    assert first.projection_is_authority is False


def test_cut_edge_identity_verifies() -> None:
    edge = PartitionCutEdge(
        source_id="node:left",
        target_id="node:leaf",
        kind="calls",
        constraint_class="hard",
    )
    restored = PartitionCutEdge.from_dict(edge.to_dict())
    assert restored == edge
    with pytest.raises(PartitionGeneratorError, match="does not verify"):
        PartitionCutEdge.from_dict({**edge.to_dict(), "edge_cid": _cid("forged")})


def test_overlapping_unique_owners_fail_closed() -> None:
    with pytest.raises(PartitionGeneratorError, match="overlapping unique"):
        generate_partition_candidates(
            _evidence(
                state_ownership={
                    "graph_cid": _cid("state-graph"),
                    "owners": [
                        {
                            "owner_id": "owner:a",
                            "uniqueness": "unique",
                            "owning_symbol_id": "node:left",
                            "alias_set_id": "alias:a",
                        },
                        {
                            "owner_id": "owner:b",
                            "uniqueness": "unique",
                            "owning_symbol_id": "node:right",
                            "alias_set_id": "alias:b",
                        },
                    ],
                    "alias_sets": [
                        {
                            "alias_set_id": "alias:a",
                            "representative_id": "node:left",
                            "member_ids": ["node:left", "node:leaf"],
                            "unresolved": False,
                        },
                        {
                            "alias_set_id": "alias:b",
                            "representative_id": "node:right",
                            "member_ids": ["node:right", "node:leaf"],
                            "unresolved": False,
                        },
                    ],
                    "unresolved": [],
                }
            )
        )


def test_analyzer_id_is_pinned() -> None:
    with pytest.raises(PartitionGeneratorError, match="SPAR-013 analyzer"):
        generate_partition_candidates(_evidence(analyzer_id="other.analyzer@1"))


def test_heuristic_graph_edges_do_not_cluster() -> None:
    receipt = generate_partition_candidates(
        _evidence(
            graph_view={
                "graph_view_cid": _cid("graph-view"),
                "edges": [
                    {
                        "source_id": "node:leaf",
                        "target_id": "node:left",
                        "kind": "contains",
                        "evidence_class": "model_hypothesis",
                        "confidence": "heuristic",
                    }
                ],
            }
        )
    )
    assert [
        item
        for item in receipt.candidates
        if item.generator_kind == GeneratorKind.GRAPH.value
    ] == []


def test_datasets_snapshot_to_dict_objects_are_projected() -> None:
    class _Snapshot:
        def to_dict(self) -> dict[str, Any]:
            return _evidence()["scc_snapshot"]

    receipt = generate_partition_candidates(
        {"tree_id": TREE_ID, "analyzer_id": ANALYZER_ID, "scc_snapshot": _Snapshot()}
    )
    assert receipt.candidates
    assert receipt.tree_id == TREE_ID
