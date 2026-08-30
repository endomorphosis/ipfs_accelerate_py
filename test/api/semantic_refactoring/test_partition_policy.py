"""Independent contract tests for SPAR-014 PartitionObjectiveProfile@1."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_generators import (
    ANALYZER_ID as SPAR013_ANALYZER_ID,
    ConstraintClass,
    GeneratorKind,
    PartitionCutEdge,
    ProgramPartitionCandidate,
    generate_partition_candidates,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_policy import (
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    DUCKLAKE_IS_AUTHORITY,
    GOAL_ID,
    HARD_CONSTRAINT_FAMILIES,
    IDENTITY_EXCLUDED_FIELDS,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    PARTITION_CAN_AUTHORIZE_COMPLETION,
    PARTITION_CAN_AUTHORIZE_TRANSITION,
    PARTITION_CAN_CREATE_AUTHORITY,
    PARTITION_COMPARISON_RECEIPT_INTERFACE,
    PARTITION_CONTRACT_VERSION,
    PARTITION_OBJECTIVE_PROFILE_INTERFACE,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    SOFT_OBJECTIVES,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    HardConstraintFamily,
    PartitionComparisonReceipt,
    PartitionHardResult,
    PartitionObjectiveBreakdown,
    PartitionObjectiveProfile,
    PartitionPolicyError,
    PartitionSoftScore,
    SoftObjective,
    assert_not_competing_capsule_family,
    compare_partition_candidates,
    decode_canonical_receipt,
    default_partition_objective_profile,
    encode_canonical_receipt,
    evaluate_partition_candidate,
    partition_cid_profile,
    provider_free_exports,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "partition_policy.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/partition_policy.py",
    "test/api/semantic_refactoring/test_partition_policy.py",
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
        "analyzer_id": SPAR013_ANALYZER_ID,
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


def _candidate(**overrides: Any) -> ProgramPartitionCandidate:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "generator_kind": GeneratorKind.SCC,
        "member_ids": ("node:leaf",),
        "admitted": True,
    }
    fields.update(overrides)
    return ProgramPartitionCandidate(**fields)


def _family(name: str, *, passed: bool = True, violations: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "family": name,
        "passed": passed,
        "violations": list(violations),
        "witness_ids": [],
    }


def _complete_hard(*, failed: str | None = None) -> list[dict[str, Any]]:
    results = []
    for family in HARD_CONSTRAINT_FAMILIES:
        if family == failed:
            results.append(_family(family, passed=False, violations=(f"{family}_violation",)))
        else:
            results.append(_family(family))
    return results


def _complete_soft(*, applied: int = 0, observed: int = 0) -> list[dict[str, Any]]:
    return [
        {"objective": item, "observed": observed, "applied": applied}
        for item in SOFT_OBJECTIVES
    ]


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-014"
    assert GOAL_ID == "SPAR-G032"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert PARTITION_OBJECTIVE_PROFILE_INTERFACE == "PartitionObjectiveProfile@1"
    assert PARTITION_COMPARISON_RECEIPT_INTERFACE == "PartitionComparisonReceipt@1"
    assert PARTITION_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("partition_policy@1")
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
    assert "PartitionObjectiveProfile" in names
    assert "PartitionComparisonReceipt" in names
    assert "PartitionObjectiveBreakdown" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "PartitionObjectiveProfile" in exports
    assert "compare_partition_candidates" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_default_profile_is_complete_and_integer_weighted() -> None:
    profile = default_partition_objective_profile()
    assert tuple(profile.hard_constraint_families) == HARD_CONSTRAINT_FAMILIES
    assert tuple(profile.soft_objectives) == SOFT_OBJECTIVES
    assert set(HARD_CONSTRAINT_FAMILIES) == {item.value for item in HardConstraintFamily}
    assert set(SOFT_OBJECTIVES) == {item.value for item in SoftObjective}
    assert profile.weight_for("vector") == 0
    assert profile.weight_for("call") == 1
    assert profile.can_authorize_completion is False
    restored = PartitionObjectiveProfile.from_dict(profile.to_dict())
    assert restored == profile
    assert restored.profile_cid == profile.profile_cid


def test_incomplete_profile_fails_closed() -> None:
    with pytest.raises(PartitionPolicyError, match="every hard constraint family"):
        PartitionObjectiveProfile(hard_constraint_families=("scc", "state"))
    with pytest.raises(PartitionPolicyError, match="every soft objective"):
        PartitionObjectiveProfile(soft_objectives=("call", "data"))
    with pytest.raises(PartitionPolicyError, match="must be an integer"):
        PartitionObjectiveProfile(
            soft_weights={item: (0.5 if item == "call" else 1) for item in SOFT_OBJECTIVES}
        )


def test_scc_split_and_oversized_cycle_fail_closed() -> None:
    oversized = _component("node:a", "node:b", "node:c", cyclic=True, oversized=True)
    generated = generate_partition_candidates(
        _evidence(scc_snapshot=_snapshot((oversized,)))
    )
    receipt = compare_partition_candidates(generated, evidence=_evidence(scc_snapshot=_snapshot((oversized,))))
    assert receipt.ranked_candidate_cids == ()
    assert receipt.rejected_candidate_cids
    assert receipt.negative_evidence_cids == receipt.rejected_candidate_cids
    scc = next(item for item in receipt.breakdowns[0].hard_results if item.family == "scc")
    assert scc.passed is False
    assert ConstraintClass.OVERSIZED_CYCLE.value in scc.violations


def test_state_unique_owner_split_fails_closed() -> None:
    candidate = _candidate(
        member_ids=("node:left",),
        admitted=False,
        hard_constraint_violations=(ConstraintClass.UNIQUE_OWNER_SPLIT.value,),
    )
    breakdown = evaluate_partition_candidate(candidate)
    state = next(item for item in breakdown.hard_results if item.family == "state")
    assert state.passed is False
    assert breakdown.ranked is False
    assert all(item.applied == 0 for item in breakdown.soft_scores)


def test_consumer_and_compatibility_fail_closed_when_undispositioned() -> None:
    evidence = _evidence(
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
                    "subject_id": "node:leaf",
                    "module_name": "pkg.cli",
                    "role": "cli",
                }
            ],
        }
    )
    generated = generate_partition_candidates(evidence)
    receipt = compare_partition_candidates(generated, evidence=evidence)
    scc_leaf = next(
        item
        for item in generated.candidates
        if item.generator_kind == GeneratorKind.SCC.value
        and item.member_ids == ("node:leaf",)
    )
    scc_breakdown = next(
        item for item in receipt.breakdowns if item.candidate_cid == scc_leaf.candidate_cid
    )
    families = {item.family: item for item in scc_breakdown.hard_results}
    assert families["consumer"].passed is False
    assert families["compatibility"].passed is False
    assert scc_breakdown.ranked is False
    assert scc_leaf.candidate_cid in receipt.negative_evidence_cids
    contract = next(
        item
        for item in generated.candidates
        if item.generator_kind == GeneratorKind.CONTRACT.value
    )
    contract_breakdown = next(
        item for item in receipt.breakdowns if item.candidate_cid == contract.candidate_cid
    )
    contract_families = {item.family: item for item in contract_breakdown.hard_results}
    assert contract_families["consumer"].passed is True
    assert contract_families["compatibility"].passed is True
    assert contract_breakdown.ranked is True


def test_ordering_cut_fails_closed() -> None:
    candidate = _candidate(
        cut_edges=(
            PartitionCutEdge(
                source_id="node:left",
                target_id="node:leaf",
                kind="happens_before",
                constraint_class="hard",
            ),
        )
    )
    breakdown = evaluate_partition_candidate(candidate)
    ordering = next(item for item in breakdown.hard_results if item.family == "ordering")
    assert ordering.passed is False
    assert "ordering_cut" in ordering.violations
    assert breakdown.ranked is False


def test_resource_owner_split_fails_closed() -> None:
    candidate = _candidate(member_ids=("node:leaf",))
    breakdown = evaluate_partition_candidate(
        candidate,
        evidence={
            "tree_id": TREE_ID,
            "resources": {
                "graph_cid": _cid("resource-graph"),
                "owners": [
                    {
                        "owner_id": "resource:lock",
                        "uniqueness": "unique",
                        "member_ids": ["node:leaf", "node:left"],
                    }
                ],
            },
        },
    )
    resource = next(item for item in breakdown.hard_results if item.family == "resource")
    assert resource.passed is False
    assert "resource_owner_split" in resource.violations


def test_frontier_unresolved_subject_fails_closed() -> None:
    candidate = _candidate(member_ids=("node:leaf",))
    breakdown = evaluate_partition_candidate(
        candidate,
        evidence={
            "tree_id": TREE_ID,
            "frontier": {
                "frontier_cid": _cid("frontier"),
                "unresolved_subject_ids": ["node:leaf"],
            },
        },
    )
    frontier = next(item for item in breakdown.hard_results if item.family == "frontier")
    assert frontier.passed is False
    assert "unresolved_frontier" in frontier.violations


def test_missing_proof_obligation_fails_closed() -> None:
    candidate = _candidate(member_ids=("node:leaf",))
    breakdown = evaluate_partition_candidate(
        candidate,
        evidence={
            "tree_id": TREE_ID,
            "proofs": {
                "proof_cid": _cid("proofs"),
                "obligations": [
                    {"obligation_id": "proof:leaf", "subject_id": "node:leaf"}
                ],
            },
        },
    )
    proof = next(item for item in breakdown.hard_results if item.family == "proof")
    assert proof.passed is False
    assert "missing_proof_obligation" in proof.violations


def test_transaction_owner_split_fails_closed() -> None:
    candidate = _candidate(member_ids=("node:leaf",))
    breakdown = evaluate_partition_candidate(
        candidate,
        evidence={
            "tree_id": TREE_ID,
            "transactions": {
                "graph_cid": _cid("tx-graph"),
                "owners": [
                    {
                        "owner_id": "tx:batch",
                        "uniqueness": "unique",
                        "member_ids": ["node:leaf", "node:right"],
                    }
                ],
            },
        },
    )
    transaction = next(
        item for item in breakdown.hard_results if item.family == "transaction"
    )
    assert transaction.passed is False
    assert "transaction_owner_split" in transaction.violations


def test_soft_scores_never_override_hard_constraint() -> None:
    with pytest.raises(PartitionPolicyError, match="never override"):
        PartitionObjectiveBreakdown(
            tree_id=TREE_ID,
            candidate_cid=_cid("candidate"),
            hard_results=_complete_hard(failed="scc"),
            soft_scores=_complete_soft(applied=9, observed=9),
            ranked=True,
        )
    with pytest.raises(PartitionPolicyError, match="never override"):
        PartitionObjectiveBreakdown(
            tree_id=TREE_ID,
            candidate_cid=_cid("candidate"),
            hard_results=_complete_hard(failed="state"),
            soft_scores=_complete_soft(applied=3, observed=3),
            ranked=False,
        )


def test_soft_scores_apply_only_after_hard_constraints_pass() -> None:
    evidence = _evidence(
        graph_view={
            "graph_view_cid": _cid("graph-view"),
            "edges": [
                {
                    "source_id": "node:left",
                    "target_id": "node:right",
                    "kind": "calls",
                    "evidence_class": "exact_static_fact",
                    "confidence": "exact",
                }
            ],
        }
    )
    generated = generate_partition_candidates(evidence)
    receipt = compare_partition_candidates(generated, evidence=evidence)
    cyclic = next(
        item
        for item in generated.candidates
        if item.member_ids == ("node:left", "node:right")
    )
    breakdown = next(
        item for item in receipt.breakdowns if item.candidate_cid == cyclic.candidate_cid
    )
    assert breakdown.hard_passed is True
    assert breakdown.ranked is True
    call = next(item for item in breakdown.soft_scores if item.objective == "call")
    assert call.observed == 1
    assert call.applied == 1
    for item in receipt.breakdowns:
        if item.rejected or item.advisory:
            assert all(score.applied == 0 for score in item.soft_scores)


def test_rejected_comparisons_remain_negative_evidence() -> None:
    generated = generate_partition_candidates(
        _evidence(
            projection_clusters=[
                {
                    "cluster_id": "proj:1",
                    "member_ids": ["node:leaf"],
                    "model_pin_cid": _cid("pin"),
                }
            ]
        )
    )
    receipt = compare_partition_candidates(
        generated,
        evidence=_evidence(
            projection_clusters=[
                {
                    "cluster_id": "proj:1",
                    "member_ids": ["node:leaf"],
                    "model_pin_cid": _cid("pin"),
                }
            ]
        ),
    )
    assert receipt.advisory_candidate_cids
    assert receipt.ranked_candidate_cids
    assert set(receipt.advisory_candidate_cids).isdisjoint(receipt.ranked_candidate_cids)
    oversized = _component("node:a", "node:b", cyclic=True, oversized=True)
    rejected = compare_partition_candidates(
        generate_partition_candidates(_evidence(scc_snapshot=_snapshot((oversized,)))),
        evidence=_evidence(scc_snapshot=_snapshot((oversized,))),
    )
    assert rejected.rejected_candidate_cids
    assert rejected.negative_evidence_cids == rejected.rejected_candidate_cids
    assert rejected.ranked_candidate_cids == ()


def test_vector_scores_cannot_admit_or_hide_violations() -> None:
    rejected = _candidate(
        member_ids=("node:leaf",),
        admitted=False,
        hard_constraint_violations=(ConstraintClass.OVERSIZED_CYCLE.value,),
    )
    admitted = _candidate(member_ids=("node:left", "node:right"), generator_kind=GeneratorKind.STATE)
    receipt = compare_partition_candidates(
        (rejected, admitted),
        evidence={
            "tree_id": TREE_ID,
            "vector_scores": {
                rejected.candidate_cid: 999,
                admitted.candidate_cid: 0,
            },
        },
        profile=PartitionObjectiveProfile(
            soft_weights={item: (8 if item == "vector" else 1) for item in SOFT_OBJECTIVES}
        ),
    )
    assert rejected.candidate_cid in receipt.rejected_candidate_cids
    assert rejected.candidate_cid not in receipt.ranked_candidate_cids
    assert admitted.candidate_cid in receipt.ranked_candidate_cids


def test_profile_weights_reorder_only_already_ranked_candidates() -> None:
    light = _candidate(member_ids=("node:leaf",))
    heavy = _candidate(
        member_ids=("node:left", "node:right"),
        generator_kind=GeneratorKind.STATE,
        obligation_ids=("obl:api",),
    )
    default = compare_partition_candidates((light, heavy))
    assert default.ranked_candidate_cids[0] == heavy.candidate_cid
    zero_contract = compare_partition_candidates(
        (light, heavy),
        profile=PartitionObjectiveProfile(
            soft_weights={item: 0 for item in SOFT_OBJECTIVES}
        ),
    )
    assert zero_contract.ranked_candidate_cids[0] == light.candidate_cid
    assert set(zero_contract.ranked_candidate_cids) == {
        light.candidate_cid,
        heavy.candidate_cid,
    }


def test_ranking_is_deterministic_nomination_only() -> None:
    left = _candidate(member_ids=("node:leaf",))
    right = _candidate(
        member_ids=("node:left", "node:right"),
        generator_kind=GeneratorKind.STATE,
    )
    first = compare_partition_candidates((right, left))
    second = compare_partition_candidates((left, right))
    assert first.receipt_cid == second.receipt_cid
    assert first.ranked_candidate_cids == second.ranked_candidate_cids
    assert first.can_authorize_completion is False
    assert first.can_authorize_transition is False
    assert first.projection_is_authority is False
    encoded = encode_canonical_receipt(first)
    restored = decode_canonical_receipt(encoded)
    assert restored == first
    assert restored.receipt_cid == first.receipt_cid


def test_complete_breakdown_is_required() -> None:
    with pytest.raises(PartitionPolicyError, match="every hard constraint family"):
        PartitionObjectiveBreakdown(
            tree_id=TREE_ID,
            candidate_cid=_cid("candidate"),
            hard_results=[_family("scc")],
            soft_scores=_complete_soft(),
        )
    with pytest.raises(PartitionPolicyError, match="every soft objective"):
        PartitionObjectiveBreakdown(
            tree_id=TREE_ID,
            candidate_cid=_cid("candidate"),
            hard_results=_complete_hard(),
            soft_scores=[{"objective": "call", "observed": 0, "applied": 0}],
            ranked=True,
        )


def test_identity_excludes_observational_fields() -> None:
    profile = default_partition_objective_profile()
    payload = profile.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(PartitionPolicyError, match="observational"):
        PartitionObjectiveProfile.from_dict(dirty)
    candidate = _candidate()
    receipt = compare_partition_candidates((candidate,))
    encoded = receipt.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(encoded))
    encoded["model_output"] = "guess"
    with pytest.raises(PartitionPolicyError, match="observational"):
        PartitionComparisonReceipt.from_dict(encoded)


def test_receipt_cannot_claim_authority_flags() -> None:
    receipt = compare_partition_candidates((_candidate(),))
    payload = receipt.to_dict()
    payload["can_authorize_completion"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(PartitionPolicyError, match="can_authorize_completion"):
        PartitionComparisonReceipt.from_dict(payload)


def test_empty_candidates_fail_closed() -> None:
    with pytest.raises(PartitionPolicyError, match="requires candidates"):
        compare_partition_candidates(())


def test_tree_mismatch_fails_closed() -> None:
    with pytest.raises(PartitionPolicyError, match="tree_id"):
        compare_partition_candidates(
            (_candidate(),),
            evidence={"tree_id": "0" * 40},
        )


def test_spar013_generation_receipt_compares_without_extra_evidence() -> None:
    generated = generate_partition_candidates(_evidence())
    receipt = compare_partition_candidates(generated)
    assert receipt.analyzer_id == ANALYZER_ID
    assert receipt.ranked_candidate_cids
    assert all(item.hard_passed for item in receipt.ranked_breakdowns)
    assert receipt.soft_signals_cannot_override_hard_constraints is True


def test_hard_result_and_soft_score_round_trip() -> None:
    result = PartitionHardResult(family="scc", passed=True)
    assert PartitionHardResult.from_dict(result.to_dict()) == result
    score = PartitionSoftScore(objective=SoftObjective.CALL, observed=2, applied=2)
    assert PartitionSoftScore.from_dict(score.to_dict()) == score
    with pytest.raises(PartitionPolicyError, match="retain a violation"):
        PartitionHardResult(family="scc", passed=False)
    with pytest.raises(PartitionPolicyError, match="cannot retain violations"):
        PartitionHardResult(family="scc", passed=True, violations=("scc_split",))
