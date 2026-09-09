"""Independent contract tests for SPAR-040 shadow_plan rollout gate."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.rollout import (
    ALLOWED_CURRENT_MODES,
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    BOOTSTRAP_MODE,
    BOOTSTRAP_UNTIL_SPAR_040,
    CANDIDATE_COMPARISON_INTERFACE,
    CANDIDATE_DISPOSITIONS,
    COMPLETE_ANALYSIS_REQUIRED,
    COMPLETE_PLANNING_REQUIRED,
    CandidateComparison,
    CandidateDisposition,
    DECLARED_CANDIDATE_DISPOSITIONS,
    DECLARED_GATE_STATUSES,
    DECLARED_HARD_CONSTRAINTS,
    DECLARED_RECEIPT_FLOOR,
    DECLARED_ROLLOUT_MODES,
    DECLARED_TERMINAL_KINDS,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    EXISTING_ADAPTER_AUTHORITIES,
    FALSE_UNSAFE_CANDIDATES_RETAINED,
    FORBIDDEN_ROLLOUT_NAMES,
    GATE_CAN_AUTHORIZE_COMPLETION,
    GATE_CAN_AUTHORIZE_TRANSITION,
    GATE_CAN_CHANGE_MODE,
    GATE_CAN_CREATE_AUTHORITY,
    GATE_IS_NOMINATION_ONLY,
    GATE_WRITES_REPOSITORY,
    GOAL_ID,
    HARD_CONSTRAINTS,
    IDENTITY_EXCLUDED_FIELDS,
    MARKDOWN_IS_NOT_COMPLETION,
    MODE_CONTRACT_INTERFACE,
    MODE_CONTRACTS,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    ModeContract,
    NEGATIVE_EVIDENCE_RETAINED,
    NETWORK_DENIED,
    NETWORK_DENY,
    PREDECESSOR_TASK_IDS,
    PROGRAM,
    PROGRESSIVE_MODES,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    RECEIPT_FLOOR,
    ROLLOUT_BASELINE_SCHEMA,
    ROLLOUT_CONTRACT_VERSION,
    ROLLOUT_MODES,
    ROLLOUT_MODE_INTERFACE,
    RolloutError,
    RolloutMode,
    SHADOW_PLAN_CANDIDATE_INTERFACE,
    SHADOW_PLAN_GATE_INTERFACE,
    SHADOW_PLAN_INFLUENCES_ROUTING,
    SHADOW_PLAN_MERGE,
    SHADOW_PLAN_MODE,
    SHADOW_PLAN_PROMOTES_ROOT,
    SHADOW_PLAN_RECEIPT_INTERFACE,
    SHADOW_PLAN_SOURCE_MUTATION,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    ShadowPlanCandidate,
    ShadowPlanGate,
    ShadowPlanReceipt,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    TYPED_TERMINAL_INTERFACE,
    TerminalKind,
    TypedTerminal,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_MAY_CHANGE_MODE,
    WORKER_SELF_APPROVAL,
    activate_shadow_plan,
    assert_not_competing_capsule_family,
    compare_and_retain_candidates,
    decode_canonical_receipt,
    dry_run_shadow_plan,
    encode_canonical_receipt,
    provider_free_exports,
    rollout_cid_profile,
    rollout_gate_descriptor,
    run_shadow_plan,
    sealed_rollout_baseline,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "rollout.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/rollout.py",
    "test/api/semantic_refactoring/test_shadow_plan.py",
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
    "RolloutStore",
)
TREE_ID = "fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051"
OTHER_TREE = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
BASELINE_PATH = (
    ROOT
    / "docs"
    / "architecture"
    / "semantic_preserving_autonomous_remodularization_inventory"
    / "rollout_baseline.json"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _candidate(
    label: str,
    disposition: str = "admitted",
    violations: tuple[str, ...] = (),
    score: int = 0,
) -> dict[str, Any]:
    return {
        "candidate_cid": _cid(f"candidate:{label}"),
        "disposition": disposition,
        "hard_constraint_violations": list(violations),
        "score": score,
    }


def _evidence(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "current_mode": BOOTSTRAP_MODE,
        "analysis_complete": True,
        "planning_complete": True,
        "source_mutation": False,
        "merge": False,
        "routing_influence": False,
        "worker_may_change_mode": False,
        "pre_world_root_cid": _cid("world-root:pre"),
        "program_graph_snapshot_cid": _cid("graph"),
        "partition_candidate_cid": _cid("partition"),
        "boundary_contract_set_cid": _cid("boundary"),
        "transformation_packet_cid": _cid("packet"),
        "context_receipt_cid": _cid("context"),
        "route_decision_cid": _cid("route"),
        "validation_receipt_cids": [],
        "refactor_transition_cid": "",
        "expected_root_generation": 3,
        "candidates": [
            _candidate("safe-high", score=4),
            _candidate("safe-low", score=1),
            _candidate("false", "false"),
            _candidate("unsafe", "unsafe", ("scc",)),
        ],
        "network": NETWORK_DENY,
        "worktree_id": _cid("worktree"),
        "lease_id": "lease-1",
        "fence_id": "fence-1",
    }
    fields.update(overrides)
    return fields


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-040"
    assert GOAL_ID == "SPAR-G073"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert PREDECESSOR_TASK_IDS == ("SPAR-038", "SPAR-039")
    assert SHADOW_PLAN_GATE_INTERFACE == "ShadowPlanGate@1"
    assert ROLLOUT_MODE_INTERFACE == "RolloutMode@1"
    assert MODE_CONTRACT_INTERFACE == "RolloutModeContract@1"
    assert SHADOW_PLAN_CANDIDATE_INTERFACE == "ShadowPlanCandidate@1"
    assert CANDIDATE_COMPARISON_INTERFACE == "ShadowPlanCandidateComparison@1"
    assert SHADOW_PLAN_RECEIPT_INTERFACE == "ShadowPlanReceipt@1"
    assert TYPED_TERMINAL_INTERFACE == "TypedTerminal@1"
    assert ROLLOUT_CONTRACT_VERSION == "1"
    assert ROLLOUT_BASELINE_SCHEMA == "spar/rollout-baseline@1"
    assert ANALYZER_ID.endswith("rollout@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()
    assert ROLLOUT_MODES == (
        "bootstrap",
        "shadow_plan",
        "shadow_apply",
        "guarded",
        "required",
    )
    assert DECLARED_ROLLOUT_MODES == set(ROLLOUT_MODES)
    assert SHADOW_PLAN_MODE == "shadow_plan"
    assert BOOTSTRAP_MODE == "bootstrap"
    assert ALLOWED_CURRENT_MODES == {"bootstrap", "shadow_plan"}
    assert PROGRESSIVE_MODES == ("shadow_apply", "guarded", "required")
    assert DECLARED_GATE_STATUSES == {"nominated_shadow_plan", "typed_terminal"}
    assert DECLARED_TERMINAL_KINDS == {
        "unsupported",
        "human_review",
        "capability_unavailable",
    }
    assert CANDIDATE_DISPOSITIONS == ("admitted", "false", "unsafe")
    assert DECLARED_CANDIDATE_DISPOSITIONS == set(CANDIDATE_DISPOSITIONS)
    assert HARD_CONSTRAINTS == (
        "scc",
        "state",
        "consumer",
        "compatibility",
        "ordering",
        "resource",
        "frontier",
        "proof",
        "transaction",
    )
    assert DECLARED_HARD_CONSTRAINTS == set(HARD_CONSTRAINTS)
    assert RECEIPT_FLOOR == (
        "pre_world_root_cid",
        "program_graph_snapshot_cid",
        "partition_candidate_cid",
        "boundary_contract_set_cid",
        "transformation_packet_cid",
        "context_receipt_cid",
        "route_decision_cid",
        "validation_receipt_cids",
        "refactor_transition_cid",
        "post_world_root_cid",
        "expected_root_generation",
        "resulting_root_generation",
        "rollout_mode",
    )
    assert DECLARED_RECEIPT_FLOOR == set(RECEIPT_FLOOR)


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "operational refactoring authority"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert GATE_CAN_AUTHORIZE_COMPLETION is False
    assert GATE_CAN_AUTHORIZE_TRANSITION is False
    assert GATE_CAN_CREATE_AUTHORITY is False
    assert GATE_CAN_CHANGE_MODE is False
    assert GATE_WRITES_REPOSITORY is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert WORKER_MAY_CHANGE_MODE is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert GATE_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert NETWORK_DENIED is True
    assert NETWORK_DENY == "deny"
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert NEGATIVE_EVIDENCE_RETAINED is True
    assert SHADOW_PLAN_SOURCE_MUTATION is False
    assert SHADOW_PLAN_MERGE is False
    assert SHADOW_PLAN_INFLUENCES_ROUTING is False
    assert SHADOW_PLAN_PROMOTES_ROOT is False
    assert COMPLETE_ANALYSIS_REQUIRED is True
    assert COMPLETE_PLANNING_REQUIRED is True
    assert FALSE_UNSAFE_CANDIDATES_RETAINED is True
    assert BOOTSTRAP_UNTIL_SPAR_040 is True
    profile = rollout_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    assert "spar_narrow" in EXISTING_ADAPTER_AUTHORITIES
    assert "RolloutStore" in FORBIDDEN_ROLLOUT_NAMES
    assert "worker_set_mode" in FORBIDDEN_ROLLOUT_NAMES
    descriptor = rollout_gate_descriptor()
    assert descriptor["nomination_only"] is True
    assert descriptor["writes_repository"] is False
    assert descriptor["worker_may_change_mode"] is False
    assert descriptor["influences_routing"] is False
    assert descriptor["nominated_mode"] == "shadow_plan"
    assert descriptor["gate_task"] == "SPAR-040"
    assert descriptor["predecessor_task_ids"] == ["SPAR-038", "SPAR-039"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "ShadowPlanGate" in names
    assert "ShadowPlanReceipt" in names
    assert "ModeContract" in names
    assert "CandidateComparison" in names
    assert "RolloutStore" not in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "ShadowPlanGate" in exports
    assert "run_shadow_plan" in exports
    assert "activate_shadow_plan" in exports
    assert "compare_and_retain_candidates" in exports
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "authorize_mode_change" in FORBIDDEN_ROLLOUT_NAMES
    assert "def authorize_mode_change" not in source
    assert "def worker_set_mode" not in source
    assert "def mutate_source" not in source
    assert "def influence_routing" not in source


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_mode_table_matches_sealed_rollout_baseline() -> None:
    sealed = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    encoded = sealed_rollout_baseline()
    assert sealed["schema"] == ROLLOUT_BASELINE_SCHEMA
    assert sealed["current_mode"] == BOOTSTRAP_MODE
    assert sealed["worker_may_change_mode"] is False
    assert encoded["current_mode"] == sealed["current_mode"]
    assert encoded["worker_may_change_mode"] is False
    assert encoded["receipt_floor"] == sealed["receipt_floor"]
    assert list(RECEIPT_FLOOR) == sealed["receipt_floor"]
    for mode in ROLLOUT_MODES:
        assert dict(MODE_CONTRACTS[mode]) == sealed["modes"][mode]
        assert encoded["modes"][mode] == sealed["modes"][mode]
    assert MODE_CONTRACTS["shadow_plan"]["gate_task"] == "SPAR-040"
    assert MODE_CONTRACTS["shadow_plan"]["source_mutation"] is False
    assert MODE_CONTRACTS["shadow_plan"]["merge"] is False
    assert MODE_CONTRACTS["bootstrap"]["gate_task"] == "SPAR-000"


def test_shadow_plan_nominates_without_mutation_merge_or_routing() -> None:
    receipt = run_shadow_plan(_evidence())
    assert receipt.status == "nominated_shadow_plan"
    assert receipt.nominated is True
    assert receipt.accepted is False
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.can_create_authority is False
    assert receipt.current_mode == BOOTSTRAP_MODE
    assert receipt.nominated_mode == SHADOW_PLAN_MODE
    assert receipt.rollout_mode == SHADOW_PLAN_MODE
    assert receipt.mode_contract.gate_task == "SPAR-040"
    assert receipt.mode_contract.source_mutation is False
    assert receipt.mode_contract.merge is False
    assert receipt.post_world_root_cid == receipt.pre_world_root_cid
    assert receipt.resulting_root_generation == receipt.expected_root_generation
    assert receipt.refactor_transition_cid == ""
    assert receipt.root_promoted is False
    encoded = encode_canonical_receipt(receipt)
    assert encoded["source_mutation"] is False
    assert encoded["merge"] is False
    assert encoded["influences_routing"] is False
    assert encoded["worker_may_change_mode"] is False
    restored = decode_canonical_receipt(encoded)
    assert restored == receipt
    assert restored.receipt_cid == receipt.receipt_cid
    for field in RECEIPT_FLOOR:
        assert field in encoded


def test_false_and_unsafe_candidates_are_compared_and_retained() -> None:
    receipt = activate_shadow_plan(_evidence())
    comparison = receipt.comparison
    admitted = comparison.ranked_admitted_cids
    assert admitted == (
        _cid("candidate:safe-high"),
        _cid("candidate:safe-low"),
    )
    assert _cid("candidate:false") in comparison.false_candidate_cids
    assert _cid("candidate:unsafe") in comparison.unsafe_candidate_cids
    assert set(comparison.retained_negative_cids) == {
        _cid("candidate:false"),
        _cid("candidate:unsafe"),
    }
    assert tuple(receipt.negative_evidence_cids) == tuple(
        comparison.retained_negative_cids
    )
    assert FALSE_UNSAFE_CANDIDATES_RETAINED is True


def test_soft_score_cannot_admit_an_unsafe_candidate() -> None:
    with pytest.raises(RolloutError, match="hard constraint"):
        run_shadow_plan(
            _evidence(
                candidates=[
                    _candidate("promoted", "admitted", ("scc",), score=99),
                ]
            )
        )
    with pytest.raises(RolloutError, match="unsafe candidates require"):
        ShadowPlanCandidate(
            candidate_cid=_cid("candidate:bare-unsafe"),
            disposition="unsafe",
        )


def test_idempotent_when_already_in_shadow_plan() -> None:
    receipt = run_shadow_plan(_evidence(current_mode=SHADOW_PLAN_MODE))
    assert receipt.status == "nominated_shadow_plan"
    assert receipt.current_mode == SHADOW_PLAN_MODE
    assert receipt.nominated_mode == SHADOW_PLAN_MODE


def test_worker_cannot_change_rollout_mode() -> None:
    with pytest.raises(RolloutError, match="cannot change rollout"):
        run_shadow_plan(_evidence(worker_may_change_mode=True))
    with pytest.raises(RolloutError, match="cannot change rollout"):
        run_shadow_plan(_evidence(requested_mode="shadow_apply"))
    with pytest.raises(RolloutError, match="cannot change rollout"):
        run_shadow_plan(_evidence(requested_mode="guarded"))
    with pytest.raises(RolloutError, match="cannot change rollout"):
        run_shadow_plan(_evidence(requested_mode="required"))
    with pytest.raises(RolloutError, match="cannot skip or regress"):
        run_shadow_plan(_evidence(current_mode="shadow_apply"))
    with pytest.raises(RolloutError, match="cannot skip or regress"):
        run_shadow_plan(_evidence(current_mode="guarded"))
    with pytest.raises(RolloutError, match="cannot skip or regress"):
        run_shadow_plan(_evidence(current_mode="required"))


def test_source_mutation_merge_and_routing_influence_fail_closed() -> None:
    with pytest.raises(RolloutError, match="cannot mutate"):
        run_shadow_plan(_evidence(), mutate=True)
    with pytest.raises(RolloutError, match="cannot mutate"):
        run_shadow_plan(_evidence(mutate=True))
    with pytest.raises(RolloutError, match="cannot mutate source"):
        run_shadow_plan(_evidence(source_mutation=True))
    with pytest.raises(RolloutError, match="cannot merge"):
        run_shadow_plan(_evidence(merge=True))
    with pytest.raises(RolloutError, match="cannot influence routing"):
        run_shadow_plan(_evidence(routing_influence=True))
    with pytest.raises(RolloutError, match="cannot influence routing"):
        run_shadow_plan(_evidence(influences_routing=True))
    with pytest.raises(RolloutError, match="cannot influence routing"):
        run_shadow_plan(
            _evidence(proposed_route_decision_cid=_cid("route:other"))
        )


def test_world_root_and_transition_cannot_be_promoted() -> None:
    with pytest.raises(RolloutError, match="cannot mutate or promote"):
        run_shadow_plan(_evidence(post_world_root_cid=_cid("world-root:post")))
    with pytest.raises(RolloutError, match="cannot promote root generation"):
        run_shadow_plan(_evidence(resulting_root_generation=4))
    with pytest.raises(RolloutError, match="cannot publish a refactor transition"):
        run_shadow_plan(_evidence(refactor_transition_cid=_cid("transition")))


def test_incomplete_analysis_or_planning_fails_closed() -> None:
    with pytest.raises(RolloutError, match="complete analysis"):
        run_shadow_plan(_evidence(analysis_complete=False))
    with pytest.raises(RolloutError, match="complete planning"):
        run_shadow_plan(_evidence(planning_complete=False))
    with pytest.raises(RolloutError, match="requires planned candidates"):
        run_shadow_plan(_evidence(candidates=[]))


def test_capability_unavailable_is_typed_terminal() -> None:
    receipt = run_shadow_plan(_evidence(analysis_available=False))
    assert receipt.status == "typed_terminal"
    assert receipt.accepted is False
    assert receipt.nominated is False
    assert receipt.terminal is not None
    assert receipt.terminal.kind == TerminalKind.CAPABILITY_UNAVAILABLE.value
    assert receipt.rollout_mode == SHADOW_PLAN_MODE
    planning = run_shadow_plan(_evidence(planning_available=False))
    assert planning.terminal.kind == TerminalKind.CAPABILITY_UNAVAILABLE.value


def test_explicit_human_review_terminal_stops_without_completion() -> None:
    receipt = run_shadow_plan(
        _evidence(
            terminal={
                "kind": TerminalKind.HUMAN_REVIEW.value,
                "reason": "unsafe partition split requires review",
            }
        )
    )
    assert receipt.status == "typed_terminal"
    assert receipt.terminal.kind == TerminalKind.HUMAN_REVIEW.value
    assert receipt.accepted is False
    assert receipt.nominated is False


def test_dry_run_is_deterministic_and_never_mutates() -> None:
    payload = _evidence()
    first = dry_run_shadow_plan(payload)
    second = ShadowPlanGate().dry_run(payload)
    assert first.receipt_cid == second.receipt_cid
    assert DRY_RUN_MUTATES is False
    with pytest.raises(RolloutError, match="cannot mutate"):
        run_shadow_plan(payload, mutate=True)
    with pytest.raises(RolloutError, match="cannot mutate"):
        ShadowPlanGate().run(payload, mutate=True)


def test_identity_excludes_observational_fields() -> None:
    receipt = run_shadow_plan(_evidence())
    encoded = receipt.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(encoded))
    encoded["timestamp"] = "now"
    with pytest.raises(RolloutError, match="observational"):
        ShadowPlanReceipt.from_dict(encoded)
    with pytest.raises(RolloutError, match="observational"):
        run_shadow_plan(_evidence(model_output="guess"))


def test_vector_and_model_evidence_cannot_admit_a_plan() -> None:
    with pytest.raises(RolloutError, match="cannot admit"):
        run_shadow_plan(_evidence(vector_candidate={"score": 1}))
    with pytest.raises(RolloutError, match="cannot admit"):
        run_shadow_plan(_evidence(model_hypothesis={"ok": True}))
    with pytest.raises(RolloutError, match="cannot admit"):
        run_shadow_plan(_evidence(heuristic=True))


def test_network_is_denied() -> None:
    with pytest.raises(RolloutError, match="network is denied"):
        run_shadow_plan(_evidence(network="allow"))


def test_tree_mismatch_fails_closed() -> None:
    receipt = run_shadow_plan(_evidence())
    payload = receipt.to_dict()
    payload["tree_id"] = OTHER_TREE
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(RolloutError, match="tree_id"):
        ShadowPlanReceipt.from_dict(payload)
    with pytest.raises(RolloutError, match="tree_id"):
        run_shadow_plan(_evidence(tree_id="not-a-tree"))


def test_worker_cannot_self_approve_receipt() -> None:
    receipt = run_shadow_plan(_evidence())
    payload = receipt.to_dict()
    payload["accepted"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(RolloutError, match="self-approve"):
        ShadowPlanReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["can_authorize_completion"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(RolloutError, match="can_authorize_completion"):
        ShadowPlanReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["worker_may_change_mode"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(RolloutError, match="cannot claim worker_may_change_mode"):
        ShadowPlanReceipt.from_dict(payload)


def test_mode_contract_cannot_be_rewritten_by_workers() -> None:
    with pytest.raises(RolloutError, match="source_mutation does not match"):
        ModeContract(
            mode="shadow_plan",
            source_mutation=True,
            merge=False,
            gate_task="SPAR-040",
        )
    with pytest.raises(RolloutError, match="gate_task does not match"):
        ModeContract(
            mode="shadow_plan",
            source_mutation=False,
            merge=False,
            gate_task="SPAR-041",
        )
    contract = ModeContract.for_mode("shadow_plan")
    restored = ModeContract.from_dict(contract.to_dict())
    assert restored == contract
    payload = contract.to_dict()
    payload["worker_may_change_mode"] = True
    payload["contract_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "contract_cid"}
    )
    with pytest.raises(RolloutError, match="cannot change rollout"):
        ModeContract.from_dict(payload)


def test_gate_methods_match_module_functions() -> None:
    gate = ShadowPlanGate()
    payload = _evidence()
    assert gate.interface == SHADOW_PLAN_GATE_INTERFACE
    assert gate.run(payload).receipt_cid == run_shadow_plan(payload).receipt_cid
    assert gate.activate(payload).receipt_cid == activate_shadow_plan(payload).receipt_cid
    comparison = gate.compare(payload["candidates"], tree_id=TREE_ID)
    assert comparison.comparison_cid == compare_and_retain_candidates(
        payload["candidates"], tree_id=TREE_ID
    ).comparison_cid
    terminal = TypedTerminal(
        kind=TerminalKind.UNSUPPORTED.value,
        reason="required dynamic frontier remains unresolved",
    )
    assert TypedTerminal.from_dict(terminal.to_dict()) == terminal
    assert RolloutMode.SHADOW_PLAN.value == "shadow_plan"
    assert CandidateDisposition.UNSAFE.value == "unsafe"
