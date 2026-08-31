"""Independent contract tests for SPAR-036 MonolithOpportunityDetector."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_generators import (
    ConstraintClass,
    GeneratorKind,
    ProgramPartitionCandidate,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_policy import (
    ANALYZER_ID as SPAR014_ANALYZER_ID,
    compare_partition_candidates,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.opportunity_detector import (
    ACCEPTANCE_IDS,
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    AutonomyTier,
    COMPILER_ID,
    DUCKLAKE_IS_AUTHORITY,
    DurableGoal,
    EXISTING_ADAPTER_AUTHORITIES,
    FindingKind,
    GENERIC_PROMPT_FORBIDDEN,
    GOAL_COMPILATION_RECEIPT_INTERFACE,
    GOAL_COMPILER_INTERFACE,
    GOAL_ID,
    GoalCompilationReceipt,
    GoalCompiler,
    IDENTITY_EXCLUDED_FIELDS,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    MONOLITH_FINDING_INTERFACE,
    MONOLITH_OPPORTUNITY_DETECTOR_INTERFACE,
    MonolithFinding,
    MonolithOpportunityDetector,
    OPPORTUNITY_CAN_AUTHORIZE_COMPLETION,
    OPPORTUNITY_CAN_AUTHORIZE_TRANSITION,
    OPPORTUNITY_CAN_CREATE_AUTHORITY,
    OPPORTUNITY_CONTRACT_VERSION,
    OPPORTUNITY_DETECTION_RECEIPT_INTERFACE,
    OPPORTUNITY_POLICY_INTERFACE,
    OpportunityDetectionReceipt,
    OpportunityDetectorError,
    OpportunityPolicy,
    PLAN_IS_NOMINATION_ONLY,
    POLICY_ID,
    POLICY_REVISION,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    RiskKind,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    THRESHOLD_FAMILIES,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    assert_not_competing_capsule_family,
    compile_durable_goals,
    decode_canonical_detection_receipt,
    decode_canonical_goal_receipt,
    default_opportunity_policy,
    detect_monolith_opportunities,
    encode_canonical_receipt,
    opportunity_cid_profile,
    provider_free_exports,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "opportunity_detector.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/opportunity_detector.py",
    "test/api/semantic_refactoring/test_opportunity_detector.py",
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


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _component(
    *member_ids: str,
    cyclic: bool = False,
    oversized: bool = False,
) -> dict[str, Any]:
    members = tuple(sorted(member_ids))
    identity = {
        "cyclic": cyclic,
        "member_ids": list(members),
        "oversized": oversized,
    }
    return {
        "scc_id": cid_for_dag_json(identity),
        "member_ids": list(members),
        "cyclic": cyclic,
        "oversized": oversized,
        "state_owner_ids": [],
    }


def _snapshot(components: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    payload = {"components": list(components), "condensation_edges": []}
    return {
        "snapshot_cid": cid_for_dag_json(payload),
        "components": list(components),
        "condensation_edges": [],
    }


def _module(
    module_id: str,
    *,
    loc: int = 10,
    member_ids: tuple[str, ...] | None = None,
    public_export_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "module_id": module_id,
        "loc": loc,
        "member_ids": list(member_ids or (module_id,)),
        "public_export_ids": list(public_export_ids),
    }


def _evidence(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "modules": [_module("mod:small", loc=10, member_ids=("node:leaf",))],
    }
    fields.update(overrides)
    return fields


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-036"
    assert GOAL_ID == "SPAR-G071"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert OPPORTUNITY_POLICY_INTERFACE == "OpportunityPolicy@1"
    assert MONOLITH_FINDING_INTERFACE == "MonolithFinding@1"
    assert OPPORTUNITY_DETECTION_RECEIPT_INTERFACE == "OpportunityDetectionReceipt@1"
    assert GOAL_COMPILATION_RECEIPT_INTERFACE == "GoalCompilationReceipt@1"
    assert MONOLITH_OPPORTUNITY_DETECTOR_INTERFACE == "MonolithOpportunityDetector@1"
    assert GOAL_COMPILER_INTERFACE == "GoalCompiler@1"
    assert OPPORTUNITY_CONTRACT_VERSION == "1"
    assert POLICY_ID == "opportunity-policy@1"
    assert POLICY_REVISION == "1"
    assert ANALYZER_ID.endswith("opportunity_detector@1")
    assert COMPILER_ID.endswith("goal_compiler@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "opportunity detection"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert OPPORTUNITY_CAN_AUTHORIZE_COMPLETION is False
    assert OPPORTUNITY_CAN_AUTHORIZE_TRANSITION is False
    assert OPPORTUNITY_CAN_CREATE_AUTHORITY is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert PLAN_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert GENERIC_PROMPT_FORBIDDEN is True
    profile = opportunity_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    assert "spar_narrow" in EXISTING_ADAPTER_AUTHORITIES


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "MonolithOpportunityDetector" in names
    assert "GoalCompiler" in names
    assert "OpportunityPolicy" in names
    assert "MonolithFinding" in names
    assert "DurableGoal" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "MonolithOpportunityDetector" in exports
    assert "GoalCompiler" in exports
    assert "detect_monolith_opportunities" in exports
    assert "compile_durable_goals" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_default_policy_is_complete_and_integer_thresholded() -> None:
    policy = default_opportunity_policy()
    assert tuple(item["family"] for item in policy.thresholds) == THRESHOLD_FAMILIES
    assert policy.limit_for("loc") == 400
    assert policy.limit_for("public_export") == 8
    assert policy.vector_may_admit is False
    assert policy.can_authorize_completion is False
    restored = OpportunityPolicy.from_dict(policy.to_dict())
    assert restored == policy
    assert restored.policy_cid == policy.policy_cid


def test_incomplete_policy_fails_closed() -> None:
    with pytest.raises(OpportunityDetectorError, match="every threshold family"):
        OpportunityPolicy(thresholds={"loc": 400, "public_export": 8})
    with pytest.raises(OpportunityDetectorError, match="must be a positive integer"):
        OpportunityPolicy(thresholds={item: 0 for item in THRESHOLD_FAMILIES})
    with pytest.raises(OpportunityDetectorError, match="must be an integer"):
        OpportunityPolicy(
            thresholds={
                item: (0.5 if item == "loc" else 8) for item in THRESHOLD_FAMILIES
            }
        )
    with pytest.raises(OpportunityDetectorError, match="cannot admit"):
        OpportunityPolicy(vector_may_admit=True)


def test_oversized_loc_is_detected() -> None:
    receipt = detect_monolith_opportunities(
        _evidence(
            modules=[
                _module("mod:core", loc=520, member_ids=("node:core",)),
                _module("mod:small", loc=10, member_ids=("node:leaf",)),
            ]
        )
    )
    assert len(receipt.findings) == 1
    finding = receipt.findings[0]
    assert finding.module_id == "mod:core"
    assert finding.kind == FindingKind.OVERSIZED.value
    assert finding.oversized is True
    assert finding.structurally_overloaded is False
    assert RiskKind.OVERSIZED_LOC.value in finding.risks
    assert finding.autonomy_tier == AutonomyTier.B.value
    assert receipt.below_threshold_module_ids == ("mod:small",)
    assert receipt.negative_evidence_ids == ("mod:small",)
    assert finding.can_authorize_completion is False


def test_structurally_overloaded_exports_and_state_are_detected() -> None:
    receipt = detect_monolith_opportunities(
        _evidence(
            modules=[
                _module(
                    "mod:api",
                    loc=40,
                    member_ids=("node:a", "node:b"),
                    public_export_ids=tuple(f"exp:{index}" for index in range(8)),
                )
            ],
            state={
                "graph_cid": _cid("state"),
                "owners": [
                    {
                        "owner_id": "owner:one",
                        "uniqueness": "unique",
                        "member_ids": ["node:a"],
                    },
                    {
                        "owner_id": "owner:two",
                        "uniqueness": "unique",
                        "member_ids": ["node:b"],
                    },
                ],
            },
        )
    )
    finding = receipt.findings[0]
    assert finding.kind == FindingKind.STRUCTURALLY_OVERLOADED.value
    assert finding.structurally_overloaded is True
    assert finding.oversized is False
    assert finding.public_export_count == 8
    assert finding.unique_state_owner_count == 2
    assert RiskKind.PUBLIC_COMPATIBILITY.value in finding.risks
    assert RiskKind.UNIQUE_STATE_OWNER.value in finding.risks
    assert finding.autonomy_tier == AutonomyTier.C.value


def test_mixed_oversized_and_overloaded_uses_mixed_kind() -> None:
    receipt = detect_monolith_opportunities(
        _evidence(
            modules=[
                _module(
                    "mod:god",
                    loc=900,
                    member_ids=("node:a", "node:b"),
                    public_export_ids=tuple(f"exp:{index}" for index in range(8)),
                )
            ]
        )
    )
    finding = receipt.findings[0]
    assert finding.kind == FindingKind.MIXED.value
    assert finding.oversized is True
    assert finding.structurally_overloaded is True


def test_oversized_cyclic_scc_fails_into_finding() -> None:
    oversized = _component("node:a", "node:b", "node:c", cyclic=True, oversized=True)
    receipt = detect_monolith_opportunities(
        _evidence(
            modules=[_module("mod:cycle", loc=12, member_ids=("node:a", "node:b"))],
            scc_snapshot=_snapshot((oversized,)),
        )
    )
    finding = receipt.findings[0]
    assert finding.oversized is True
    assert finding.scc_member_count == 3
    assert RiskKind.OVERSIZED_SCC.value in finding.risks


def test_unresolved_frontier_lowers_autonomy_and_cannot_hide() -> None:
    receipt = detect_monolith_opportunities(
        _evidence(
            modules=[_module("mod:core", loc=520, member_ids=("node:core",))],
            frontier={
                "frontier_cid": _cid("frontier"),
                "unresolved_subject_ids": ["node:core"],
            },
        )
    )
    finding = receipt.findings[0]
    assert finding.autonomy_tier == AutonomyTier.D.value
    assert RiskKind.UNRESOLVED_FRONTIER.value in finding.risks
    assert finding.module_id == "mod:core"


def test_opaque_frontier_forces_tier_e() -> None:
    receipt = detect_monolith_opportunities(
        _evidence(
            modules=[_module("mod:core", loc=520, member_ids=("node:core",))],
            frontier={
                "frontier_cid": _cid("frontier"),
                "opaque_subject_ids": ["mod:core"],
            },
        )
    )
    assert receipt.findings[0].autonomy_tier == AutonomyTier.E.value
    assert RiskKind.OPAQUE_FRONTIER.value in receipt.findings[0].risks


def test_vector_scores_cannot_admit_or_hide_modules() -> None:
    admitted = detect_monolith_opportunities(
        _evidence(
            modules=[
                _module("mod:core", loc=520, member_ids=("node:core",)),
                _module("mod:small", loc=10, member_ids=("node:leaf",)),
            ],
            vector_scores={"mod:small": 999, "mod:core": 0},
        )
    )
    assert [item.module_id for item in admitted.findings] == ["mod:core"]
    assert "mod:small" in admitted.below_threshold_module_ids
    with pytest.raises(OpportunityDetectorError, match="cannot admit"):
        detect_monolith_opportunities(
            _evidence(vector_candidate={"mod:small": 1})
        )


def test_initialization_blocks_and_incomplete_contracts_are_risks() -> None:
    receipt = detect_monolith_opportunities(
        _evidence(
            modules=[_module("mod:boot", loc=40, member_ids=("node:a", "node:b"))],
            initialization={
                "graph_cid": _cid("init"),
                "blocks": [
                    {"block_id": f"block:{index}", "member_ids": ["node:a"]}
                    for index in range(4)
                ],
            },
            compatibility={
                "inventory_cid": _cid("compat"),
                "obligations": [
                    {
                        "obligation_id": "obl:import",
                        "subject_id": "node:a",
                        "consumer_id": "consumer:cli",
                        "kind": "import_path",
                    }
                ],
                "consumers": [],
            },
        )
    )
    finding = receipt.findings[0]
    assert finding.structurally_overloaded is True
    assert RiskKind.INITIALIZATION_ORDER.value in finding.risks
    assert RiskKind.INCOMPLETE_CONTRACT.value in finding.risks
    assert finding.autonomy_tier == AutonomyTier.C.value


def test_missing_modules_and_tree_mismatch_fail_closed() -> None:
    with pytest.raises(OpportunityDetectorError, match="requires modules"):
        detect_monolith_opportunities({"tree_id": TREE_ID, "modules": []})
    with pytest.raises(OpportunityDetectorError, match="tree_id"):
        detect_monolith_opportunities({"tree_id": "not-a-tree", "modules": [_module("mod:a")]})
    receipt = detect_monolith_opportunities(
        _evidence(modules=[_module("mod:core", loc=520, member_ids=("node:core",))])
    )
    payload = receipt.to_dict()
    payload["tree_id"] = OTHER_TREE
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(OpportunityDetectorError, match="tree_id"):
        OpportunityDetectionReceipt.from_dict(payload)


def test_unknown_scc_member_fails_closed() -> None:
    leaf = _component("node:leaf")
    with pytest.raises(OpportunityDetectorError, match="unknown SCC member"):
        detect_monolith_opportunities(
            _evidence(
                modules=[_module("mod:core", loc=520, member_ids=("node:missing",))],
                scc_snapshot=_snapshot((leaf,)),
            )
        )


def test_soft_scores_never_override_hard_constraint_on_receipt() -> None:
    finding = detect_monolith_opportunities(
        _evidence(modules=[_module("mod:core", loc=520, member_ids=("node:core",))])
    ).findings[0]
    with pytest.raises(OpportunityDetectorError, match="never override"):
        OpportunityDetectionReceipt(
            tree_id=TREE_ID,
            policy=default_opportunity_policy(),
            findings=(finding,),
            below_threshold_module_ids=("mod:core",),
        )


def test_detection_is_deterministic_nomination_only() -> None:
    evidence = _evidence(
        modules=[
            _module("mod:b", loc=401, member_ids=("node:b",)),
            _module("mod:a", loc=900, member_ids=("node:a",)),
        ]
    )
    first = detect_monolith_opportunities(evidence)
    second = MonolithOpportunityDetector().detect(evidence)
    assert first.receipt_cid == second.receipt_cid
    assert first.finding_cids == second.finding_cids
    assert first.can_authorize_completion is False
    assert first.can_authorize_transition is False
    assert first.projection_is_authority is False
    encoded = encode_canonical_receipt(first)
    restored = decode_canonical_detection_receipt(encoded)
    assert restored == first
    assert restored.receipt_cid == first.receipt_cid


def test_identity_excludes_observational_fields() -> None:
    policy = default_opportunity_policy()
    payload = policy.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(OpportunityDetectorError, match="observational"):
        OpportunityPolicy.from_dict(dirty)
    receipt = detect_monolith_opportunities(
        _evidence(modules=[_module("mod:core", loc=520, member_ids=("node:core",))])
    )
    encoded = receipt.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(encoded))
    encoded["model_output"] = "guess"
    with pytest.raises(OpportunityDetectorError, match="observational"):
        OpportunityDetectionReceipt.from_dict(encoded)


def test_receipt_cannot_claim_authority_flags() -> None:
    receipt = detect_monolith_opportunities(
        _evidence(modules=[_module("mod:core", loc=520, member_ids=("node:core",))])
    )
    payload = receipt.to_dict()
    payload["can_authorize_completion"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(OpportunityDetectorError, match="can_authorize_completion"):
        OpportunityDetectionReceipt.from_dict(payload)


def test_goal_compiler_emits_one_bounded_goal_per_finding() -> None:
    detection = detect_monolith_opportunities(
        _evidence(
            modules=[
                _module("mod:core", loc=520, member_ids=("node:core",)),
                _module("mod:api", loc=40, public_export_ids=tuple(f"exp:{i}" for i in range(8))),
            ]
        )
    )
    compiled = GoalCompiler().compile(detection)
    assert len(compiled.goals) == 2
    assert set(compiled.finding_cids) == set(detection.finding_cids)
    for goal in compiled.goals:
        assert tuple(goal.acceptance_ids) == ACCEPTANCE_IDS
        assert goal.generic_prompt_forbidden is True
        assert goal.can_authorize_completion is False
        assert goal.compiler_id == COMPILER_ID
    encoded = encode_canonical_receipt(compiled)
    restored = decode_canonical_goal_receipt(encoded)
    assert restored == compiled
    assert restored.receipt_cid == compiled.receipt_cid


def test_goal_compiler_rejects_empty_findings_and_generic_prompt() -> None:
    empty = detect_monolith_opportunities(_evidence())
    assert empty.findings == ()
    with pytest.raises(OpportunityDetectorError, match="requires findings"):
        compile_durable_goals(empty)
    detection = detect_monolith_opportunities(
        _evidence(modules=[_module("mod:core", loc=520, member_ids=("node:core",))])
    )
    compiled = compile_durable_goals(detection.findings)
    payload = compiled.goals[0].to_dict()
    payload["generic_prompt_forbidden"] = False
    payload["goal_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "goal_cid"}
    )
    with pytest.raises(OpportunityDetectorError, match="generic prompt"):
        DurableGoal.from_dict(payload)


def test_durable_goal_requires_complete_acceptance_ids() -> None:
    finding = detect_monolith_opportunities(
        _evidence(modules=[_module("mod:core", loc=520, member_ids=("node:core",))])
    ).findings[0]
    with pytest.raises(OpportunityDetectorError, match="every acceptance identifier"):
        DurableGoal(
            tree_id=TREE_ID,
            finding_cid=finding.finding_cid,
            module_id=finding.module_id,
            risks=finding.risks,
            autonomy_tier=finding.autonomy_tier,
            acceptance_ids=("exact-current-tree",),
        )


def test_finding_round_trip_and_invalid_kind_fail_closed() -> None:
    finding = detect_monolith_opportunities(
        _evidence(modules=[_module("mod:core", loc=520, member_ids=("node:core",))])
    ).findings[0]
    assert MonolithFinding.from_dict(finding.to_dict()) == finding
    with pytest.raises(OpportunityDetectorError, match="oversized or structurally"):
        MonolithFinding(
            tree_id=TREE_ID,
            module_id="mod:core",
            kind=FindingKind.OVERSIZED,
            loc=10,
            public_export_count=0,
            unique_state_owner_count=0,
            scc_member_count=1,
            initialization_block_count=0,
            responsibility_count=0,
            oversized=False,
            structurally_overloaded=False,
            risks=(RiskKind.OVERSIZED_LOC.value,),
        )


def test_graph_view_modules_are_accepted() -> None:
    receipt = detect_monolith_opportunities(
        {
            "tree_id": TREE_ID,
            "graph_view": {
                "graph_view_cid": _cid("graph"),
                "modules": [_module("mod:core", loc=520, member_ids=("node:core",))],
            },
        }
    )
    assert receipt.evidence_cids == (_cid("graph"),)
    assert receipt.findings[0].module_id == "mod:core"


def test_partition_receipt_oversized_cycle_seeds_scc_risk() -> None:
    oversized = _component("node:a", "node:b", cyclic=True, oversized=True)
    generated = compare_partition_candidates(
        (
            ProgramPartitionCandidate(
                tree_id=TREE_ID,
                generator_kind=GeneratorKind.SCC,
                member_ids=("node:a", "node:b"),
                admitted=False,
                hard_constraint_violations=(ConstraintClass.OVERSIZED_CYCLE.value,),
            ),
        )
    )
    assert generated.analyzer_id == SPAR014_ANALYZER_ID
    receipt = detect_monolith_opportunities(
        _evidence(
            modules=[_module("mod:cycle", loc=12, member_ids=("node:a",))],
            scc_snapshot=_snapshot((oversized,)),
            partition_receipt=generated.to_dict(),
        )
    )
    assert RiskKind.OVERSIZED_SCC.value in receipt.findings[0].risks
