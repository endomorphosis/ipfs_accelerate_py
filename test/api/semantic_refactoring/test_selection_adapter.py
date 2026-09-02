"""Independent contract tests for SPAR-026 RefactorValidationSelectionAdapter@1."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.selection_adapter import (
    ADAPTER_IS_NOMINATION_ONLY,
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    DATASETS_OWNS_SELECTION,
    DATASETS_SELECTION_AUTHORITY,
    DUCKLAKE_IS_AUTHORITY,
    FALLBACK_BOTH,
    FALLBACK_FULL_PROOFS,
    FALLBACK_FULL_PYTEST,
    FALLBACK_NONE,
    FORBIDDEN_RESELECTION_NAMES,
    GOAL_ID,
    IDENTITY_EXCLUDED_FIELDS,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    REFACTOR_VALIDATION_SELECTION_ADAPTER_INTERFACE,
    REFACTOR_VALIDATION_SELECTION_INTERFACE,
    REFACTOR_VALIDATION_SELECTION_RECEIPT_INTERFACE,
    SELECTION_CAN_AUTHORIZE_COMPLETION,
    SELECTION_CAN_AUTHORIZE_TRANSITION,
    SELECTION_CAN_CREATE_AUTHORITY,
    SELECTION_CAN_RESELECT,
    SELECTION_CAN_WEAKEN_PRODUCER_FALLBACK,
    SELECTION_CONTRACT_VERSION,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    FallbackReason,
    RefactorValidationSelection,
    RefactorValidationSelectionAdapter,
    RefactorValidationSelectionReceipt,
    SelectionAdapterError,
    adapt_refactor_validation_selection,
    assert_fallback_not_weakened,
    assert_not_competing_capsule_family,
    combine_fallbacks,
    compile_selection_receipt,
    decode_canonical_receipt,
    decode_canonical_selection,
    encode_canonical_receipt,
    encode_canonical_selection,
    provider_free_exports,
    selection_adapter_cid_profile,
    selection_adapter_descriptor,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "selection_adapter.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/selection_adapter.py",
    "test/api/semantic_refactoring/test_selection_adapter.py",
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
OTHER_TREE = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
WRITE_PATHS = ("pkg/mod.py", "pkg/extracted.py")
VALIDATION = ("python3 -m pytest -q tests/test_mod.py",)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _packet(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "packet_cid": _cid("packet"),
        "preimage": {
            "source_cids": [_cid("source")],
            "environment_cid": _cid("env"),
            "graph_cid": _cid("graph"),
        },
        "write_paths": list(WRITE_PATHS),
        "validation_commands": list(VALIDATION),
    }
    fields.update(overrides)
    return fields


def _graph(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "graph_view_cid": _cid("graph-view"),
        "binding": {
            "tree_id": TREE_ID,
            "environment_cid": _cid("env"),
        },
        "unresolved_frontier": {"items": []},
    }
    fields.update(overrides)
    return fields


def _frontier(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "frontier_cid": _cid("frontier"),
        "unresolved_count": 0,
        "unknown_kinds": [],
        "findings": [],
        "source_cid": _cid("source"),
    }
    fields.update(overrides)
    return fields


def _inventory(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "inventory_cid": _cid("inventory"),
        "consumers": [
            {
                "consumer_id": "pkg.cli",
                "disposition": "preserve",
                "required": True,
            }
        ],
    }
    fields.update(overrides)
    return fields


def _selection(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "selection_cid": _cid("selection"),
        "previous_root_cid": _cid("prev-root"),
        "current_root_cid": _cid("curr-root"),
        "selected_pytest_node_ids": ["tests/test_mod.py::test_a"],
        "selected_proof_ids": ["proof:mod"],
        "covered_seed_obligation_ids": ["obl:1"],
        "unresolved_obligation_ids": [],
        "known_test_universe_cid": _cid("universe"),
        "known_test_universe_count": 3,
        "fallback": FALLBACK_NONE,
        "fallback_reasons": [],
    }
    fields.update(overrides)
    return fields


def _adapt(**overrides: Any) -> RefactorValidationSelection:
    fields: dict[str, Any] = {
        "selection": _selection(),
        "packet": _packet(),
        "graph": _graph(),
        "frontier": _frontier(),
        "inventory": _inventory(),
    }
    fields.update(overrides)
    return adapt_refactor_validation_selection(**fields)


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-026"
    assert GOAL_ID == "SPAR-G051"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert (
        REFACTOR_VALIDATION_SELECTION_ADAPTER_INTERFACE
        == "RefactorValidationSelectionAdapter@1"
    )
    assert REFACTOR_VALIDATION_SELECTION_INTERFACE == "RefactorValidationSelection@1"
    assert (
        REFACTOR_VALIDATION_SELECTION_RECEIPT_INTERFACE
        == "RefactorValidationSelectionReceipt@1"
    )
    assert SELECTION_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("selection_adapter@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "validation orchestration"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert DATASETS_SELECTION_AUTHORITY == "ipfs_datasets_py"
    assert SELECTION_CAN_AUTHORIZE_COMPLETION is False
    assert SELECTION_CAN_AUTHORIZE_TRANSITION is False
    assert SELECTION_CAN_CREATE_AUTHORITY is False
    assert SELECTION_CAN_WEAKEN_PRODUCER_FALLBACK is False
    assert SELECTION_CAN_RESELECT is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert ADAPTER_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert DATASETS_OWNS_SELECTION is True
    profile = selection_adapter_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "RefactorValidationSelectionAdapter" in names
    assert "RefactorValidationSelection" in names
    assert "RefactorValidationSelectionReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "RefactorValidationSelectionAdapter" in exports
    assert "adapt_refactor_validation_selection" in exports
    assert "compile_selection_receipt" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_module_does_not_implement_graph_reselection() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not (names & FORBIDDEN_RESELECTION_NAMES)
    assert "run_impact_selected(" not in source
    assert "select_tests_and_proofs(" not in source
    descriptor = selection_adapter_descriptor()
    assert descriptor["interface"] == REFACTOR_VALIDATION_SELECTION_ADAPTER_INTERFACE
    forbids = set(descriptor["forbids"])
    assert "run_impact_selected" in forbids
    assert "graph_traversal" in forbids
    assert "weaken_producer_fallback" in forbids
    assert "invent_pytest_node_ids" in forbids
    assert "suppress_raw_source" in forbids


def test_adapt_binds_roots_obligations_and_producer_selection() -> None:
    bound = _adapt()
    assert bound.tree_id == TREE_ID
    assert bound.packet_cid == _cid("packet")
    assert bound.producer_selection_cid == _cid("selection")
    assert bound.previous_root_cid == _cid("prev-root")
    assert bound.current_root_cid == _cid("curr-root")
    assert bound.graph_view_cid == _cid("graph-view")
    assert bound.frontier_cid == _cid("frontier")
    assert bound.inventory_cid == _cid("inventory")
    assert bound.selected_pytest_node_ids == ("tests/test_mod.py::test_a",)
    assert bound.selected_proof_ids == ("proof:mod",)
    assert bound.covered_seed_obligation_ids == ("obl:1",)
    assert bound.unresolved_obligation_ids == ()
    assert bound.raw_source_cids == (_cid("source"),)
    assert bound.write_paths == WRITE_PATHS
    assert bound.validation_commands == VALIDATION
    assert bound.producer_fallback == FALLBACK_NONE
    assert bound.effective_fallback == FALLBACK_NONE
    assert bound.full_suite_required is False
    assert bound.raw_source_required is True
    assert bound.adapter_is_nomination_only is True
    assert bound.datasets_owns_selection is True
    assert bound.can_authorize_completion is False
    assert bound.can_reselect is False
    adapter = RefactorValidationSelectionAdapter()
    again = adapter.adapt(
        selection=_selection(),
        packet=_packet(),
        graph=_graph(),
        frontier=_frontier(),
        inventory=_inventory(),
    )
    assert again.validation_selection_cid == bound.validation_selection_cid


def test_producer_full_pytest_is_preserved() -> None:
    bound = _adapt(selection=_selection(fallback=FALLBACK_FULL_PYTEST))
    assert bound.producer_fallback == FALLBACK_FULL_PYTEST
    assert bound.effective_fallback == FALLBACK_FULL_PYTEST
    assert bound.requires_full_pytest is True
    assert bound.requires_full_proofs is False
    assert FallbackReason.PRODUCER_FULL_PYTEST.value in bound.fallback_reasons


def test_producer_full_proofs_is_preserved() -> None:
    bound = _adapt(selection=_selection(fallback=FALLBACK_FULL_PROOFS))
    assert bound.effective_fallback == FALLBACK_FULL_PROOFS
    assert bound.requires_full_proofs is True
    assert FallbackReason.PRODUCER_FULL_PROOFS.value in bound.fallback_reasons


def test_cannot_weaken_producer_fallback() -> None:
    with pytest.raises(SelectionAdapterError, match="cannot weaken"):
        assert_fallback_not_weakened(
            producer=FALLBACK_BOTH, effective=FALLBACK_FULL_PYTEST
        )
    with pytest.raises(SelectionAdapterError, match="cannot weaken"):
        RefactorValidationSelection(
            tree_id=TREE_ID,
            packet_cid=_cid("packet"),
            producer_selection_cid=_cid("selection"),
            current_root_cid=_cid("curr-root"),
            graph_view_cid=_cid("graph-view"),
            frontier_cid=_cid("frontier"),
            inventory_cid=_cid("inventory"),
            producer_fallback=FALLBACK_BOTH,
            effective_fallback=FALLBACK_NONE,
            fallback_reasons=(),
            selected_pytest_node_ids=("tests/test_mod.py::test_a",),
            selected_proof_ids=(),
            covered_seed_obligation_ids=(),
            unresolved_obligation_ids=(),
            raw_source_cids=(_cid("source"),),
            write_paths=WRITE_PATHS,
            validation_commands=VALIDATION,
            known_test_universe_cid=_cid("universe"),
            known_test_universe_count=1,
        )


def test_unresolved_spar008_forces_full_suite_fallback() -> None:
    bound = _adapt(
        frontier=_frontier(
            unresolved_count=1,
            findings=[
                {
                    "kind": "dynamic_dispatch",
                    "unresolved": True,
                    "confidence": "opaque",
                    "presence": "unknown",
                }
            ],
        )
    )
    assert bound.producer_fallback == FALLBACK_NONE
    assert bound.effective_fallback == FALLBACK_BOTH
    assert FallbackReason.DYNAMIC_PYTHON_FRONTIER.value in bound.fallback_reasons
    assert bound.selected_pytest_node_ids == ("tests/test_mod.py::test_a",)


def test_unresolved_spar007_frontier_forces_full_suite_fallback() -> None:
    bound = _adapt(
        graph=_graph(
            unresolved_frontier={
                "items": [
                    {
                        "subject_id": "node:leaf",
                        "subject_kind": "node",
                        "reason": "dynamic_dispatch",
                        "confidence": "conservative",
                    }
                ]
            }
        )
    )
    assert bound.effective_fallback == FALLBACK_BOTH
    assert FallbackReason.UNRESOLVED_GRAPH_FRONTIER.value in bound.fallback_reasons


def test_unresolved_obligations_force_full_suite_fallback() -> None:
    bound = _adapt(
        selection=_selection(unresolved_obligation_ids=["obl:missing"])
    )
    assert bound.effective_fallback == FALLBACK_BOTH
    assert bound.unresolved_obligation_ids == ("obl:missing",)
    assert FallbackReason.UNRESOLVED_OBLIGATIONS.value in bound.fallback_reasons


def test_unknown_test_universe_forces_full_pytest() -> None:
    bound = _adapt(selection=_selection(known_test_universe_cid=""))
    assert bound.effective_fallback == FALLBACK_FULL_PYTEST
    assert FallbackReason.UNKNOWN_TEST_UNIVERSE.value in bound.fallback_reasons


def test_undispositioned_spar011_forces_fallback() -> None:
    bound = _adapt(
        inventory=_inventory(
            consumers=[
                {
                    "consumer_id": "pkg.cli",
                    "disposition": "undispositioned",
                    "required": True,
                }
            ]
        )
    )
    assert bound.effective_fallback == FALLBACK_FULL_PYTEST
    assert FallbackReason.UNDISPOSITIONED_COMPATIBILITY.value in bound.fallback_reasons


def test_dynamic_pytest_plugin_forces_full_pytest() -> None:
    bound = _adapt(
        frontier=_frontier(
            findings=[
                {
                    "kind": "dynamic_pytest_plugin",
                    "unresolved": True,
                    "presence": "unknown",
                }
            ]
        )
    )
    assert bound.requires_full_pytest is True
    assert FallbackReason.DYNAMIC_PYTEST_PLUGIN.value in bound.fallback_reasons


def test_missing_raw_source_is_typed_terminal() -> None:
    with pytest.raises(SelectionAdapterError, match="raw source"):
        _adapt(packet=_packet(preimage={"source_cids": []}))
    with pytest.raises(SelectionAdapterError, match="raw source"):
        _adapt(packet=_packet(preimage={"environment_cid": _cid("env")}))


def test_vectors_cannot_suppress_raw_source() -> None:
    with pytest.raises(SelectionAdapterError, match="raw-source"):
        _adapt(
            vector_evidence={
                "evidence_class": "vector_candidate",
                "suppress_raw_source": True,
            }
        )
    with pytest.raises(SelectionAdapterError, match="full-suite"):
        _adapt(
            vector_evidence={
                "evidence_class": "model_hypothesis",
                "skip_full_suite": True,
            }
        )
    with pytest.raises(SelectionAdapterError, match="cannot admit"):
        _adapt(selection=_selection(evidence_class="vector_candidate"))


def test_does_not_invent_pytest_node_ids() -> None:
    bound = _adapt(
        selection=_selection(selected_pytest_node_ids=["tests/test_mod.py::test_a"]),
        frontier=_frontier(unresolved_count=1, findings=[{"unresolved": True}]),
    )
    assert bound.selected_pytest_node_ids == ("tests/test_mod.py::test_a",)
    assert "tests/test_mod.py::test_invented" not in bound.selected_pytest_node_ids
    assert bound.effective_fallback == FALLBACK_BOTH


def test_empty_write_paths_are_unrestricted_scope() -> None:
    with pytest.raises(SelectionAdapterError, match="unrestricted scope"):
        _adapt(packet=_packet(write_paths=[]))
    with pytest.raises(SelectionAdapterError, match="unrestricted scope"):
        _adapt(packet=_packet(write_paths=["pkg/*.py"]))
    with pytest.raises(SelectionAdapterError, match="unrestricted scope"):
        _adapt(packet=_packet(write_paths=["/tmp/pkg/mod.py"]))
    with pytest.raises(SelectionAdapterError, match="unrestricted scope"):
        _adapt(packet=_packet(write_paths=["pkg/../secret.py"]))


def test_tree_mismatch_fails_closed() -> None:
    with pytest.raises(SelectionAdapterError, match="SPAR-007"):
        _adapt(graph=_graph(tree_id=OTHER_TREE, binding={"tree_id": OTHER_TREE}))
    with pytest.raises(SelectionAdapterError, match="SPAR-008"):
        _adapt(frontier=_frontier(tree_id=OTHER_TREE))
    with pytest.raises(SelectionAdapterError, match="SPAR-011"):
        _adapt(inventory=_inventory(tree_id=OTHER_TREE))


def test_missing_predecessors_fail_closed() -> None:
    with pytest.raises(SelectionAdapterError, match="SPAR-019"):
        adapt_refactor_validation_selection(
            selection=_selection(),
            packet=None,
            graph=_graph(),
            frontier=_frontier(),
            inventory=_inventory(),
        )
    with pytest.raises(SelectionAdapterError, match="SPAR-007"):
        adapt_refactor_validation_selection(
            selection=_selection(),
            packet=_packet(),
            graph=None,
            frontier=_frontier(),
            inventory=_inventory(),
        )
    with pytest.raises(SelectionAdapterError, match="SPAR-008"):
        adapt_refactor_validation_selection(
            selection=_selection(),
            packet=_packet(),
            graph=_graph(),
            frontier=None,
            inventory=_inventory(),
        )
    with pytest.raises(SelectionAdapterError, match="SPAR-011"):
        adapt_refactor_validation_selection(
            selection=_selection(),
            packet=_packet(),
            graph=_graph(),
            frontier=_frontier(),
            inventory=None,
        )
    with pytest.raises(SelectionAdapterError, match="datasets"):
        adapt_refactor_validation_selection(
            selection=None,
            packet=_packet(),
            graph=_graph(),
            frontier=_frontier(),
            inventory=_inventory(),
        )


def test_combine_fallbacks_escalates_and_never_weakens() -> None:
    assert combine_fallbacks(FALLBACK_NONE, FALLBACK_FULL_PYTEST) == FALLBACK_FULL_PYTEST
    assert combine_fallbacks(FALLBACK_FULL_PYTEST, FALLBACK_FULL_PROOFS) == FALLBACK_BOTH
    assert combine_fallbacks(FALLBACK_BOTH, FALLBACK_NONE) == FALLBACK_BOTH
    assert combine_fallbacks(FALLBACK_NONE) == FALLBACK_NONE


def test_round_trip_and_receipt_are_deterministic() -> None:
    first = _adapt()
    second = _adapt()
    assert first.validation_selection_cid == second.validation_selection_cid
    restored = decode_canonical_selection(encode_canonical_selection(first))
    assert restored == first
    assert restored.validation_selection_cid == first.validation_selection_cid
    receipt = compile_selection_receipt(first)
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.selection.validation_selection_cid == first.validation_selection_cid
    assert decode_canonical_receipt(
        encode_canonical_receipt(receipt), selection=first
    ) == receipt
    again = RefactorValidationSelectionReceipt.from_dict(
        receipt.to_dict(), selection=first
    )
    assert again.receipt_cid == receipt.receipt_cid


def test_identity_excludes_observational_fields() -> None:
    bound = _adapt()
    payload = bound.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(SelectionAdapterError, match="observational"):
        RefactorValidationSelection.from_dict(dirty)


def test_selection_cannot_claim_authority_flags() -> None:
    bound = _adapt()
    payload = bound.to_dict()
    payload["can_authorize_completion"] = True
    payload["validation_selection_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "validation_selection_cid"}
    )
    with pytest.raises(SelectionAdapterError, match="can_authorize_completion"):
        RefactorValidationSelection.from_dict(payload)
    payload = bound.to_dict()
    payload["adapter_is_nomination_only"] = False
    payload["validation_selection_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "validation_selection_cid"}
    )
    with pytest.raises(SelectionAdapterError, match="nomination_only"):
        RefactorValidationSelection.from_dict(payload)
    payload = bound.to_dict()
    payload["can_weaken_producer_fallback"] = True
    payload["validation_selection_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "validation_selection_cid"}
    )
    with pytest.raises(SelectionAdapterError, match="can_weaken_producer_fallback"):
        RefactorValidationSelection.from_dict(payload)
    payload = bound.to_dict()
    payload["raw_source_required"] = False
    payload["validation_selection_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "validation_selection_cid"}
    )
    with pytest.raises(SelectionAdapterError, match="raw_source_required"):
        RefactorValidationSelection.from_dict(payload)


def test_empty_validation_commands_fail_closed() -> None:
    with pytest.raises(SelectionAdapterError, match="validation_commands"):
        _adapt(packet=_packet(validation_commands=[]))


def test_effect_scope_write_paths_are_consumed() -> None:
    packet = _packet()
    packet.pop("write_paths")
    packet["effect_scope"] = {"write_paths": list(WRITE_PATHS)}
    bound = _adapt(packet=packet)
    assert bound.write_paths == WRITE_PATHS
