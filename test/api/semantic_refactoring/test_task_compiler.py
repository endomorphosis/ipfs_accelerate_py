"""Independent contract tests for SPAR-037 task compiler."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.opportunity_detector import (
    ACCEPTANCE_IDS as SPAR036_ACCEPTANCE_IDS,
    AutonomyTier,
    DurableGoal,
    FindingKind,
    RiskKind,
    compile_durable_goals,
    detect_monolith_opportunities,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.task_compiler import (
    ACCEPTANCE_IDS,
    ALLOWED_EFFECTS,
    ALWAYS_KINDS,
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    BACKLOG_REPAIR_DECISION_INTERFACE,
    BacklogRepairDecision,
    COMPILED_TASK_INTERFACE,
    COMPILER_ID,
    CompiledTask,
    DECLARED_TASK_KINDS,
    DUCKLAKE_IS_AUTHORITY,
    EXISTING_ADAPTER_AUTHORITIES,
    EXTRACTION_WAVE_TASK_COMPILATION_RECEIPT_INTERFACE,
    EXTRACTION_WAVE_TASK_COMPILER_INTERFACE,
    ExtractionWaveTaskCompilationReceipt,
    ExtractionWaveTaskCompiler,
    FORBIDDEN_EFFECTS,
    GENERIC_PROMPT_FORBIDDEN,
    GOAL_ID,
    IDENTICAL_RETRY_WITHOUT_EVIDENCE,
    IDENTITY_EXCLUDED_FIELDS,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    ONE_PACKET_AT_A_TIME,
    PARTITION_SUBGOAL_REFINER_INTERFACE,
    PLAN_IS_NOMINATION_ONLY,
    POLICY_ID,
    POLICY_REVISION,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    PartitionSubgoalRefiner,
    RAW_SOURCE_REQUIRED,
    REFINED_SUBGOAL_INTERFACE,
    REFINER_ID,
    ROLLBACK_MODE,
    RefinedSubgoal,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    SUBGOAL_REFINEMENT_RECEIPT_INTERFACE,
    SubgoalRefinementReceipt,
    TASK_CAN_AUTHORIZE_COMPLETION,
    TASK_CAN_AUTHORIZE_TRANSITION,
    TASK_CAN_CREATE_AUTHORITY,
    TASK_CAN_RETIRE_FACADE,
    TASK_CONTRACT_VERSION,
    TASK_ID,
    TASK_KIND_ORDER,
    TEST_PASS_IS_NOT_COMPLETION,
    TaskCompilerError,
    TaskKind,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WAVE_KINDS,
    WORKER_SELF_APPROVAL,
    assert_not_competing_capsule_family,
    compile_backlog,
    compile_extraction_wave_tasks,
    decode_canonical_compilation_receipt,
    decode_canonical_refinement_receipt,
    encode_canonical_receipt,
    provider_free_exports,
    refine_partition_subgoals,
    retry_fingerprint,
    task_compiler_cid_profile,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.transformation_packet import (
    ROLLBACK_MODE as SPAR019_ROLLBACK_MODE,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "task_compiler.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/task_compiler.py",
    "test/api/semantic_refactoring/test_task_compiler.py",
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
VALIDATION = (
    "python3 -m pytest -q test/api/semantic_refactoring/test_task_compiler.py",
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


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
        "member_ids": list(member_ids or (module_id.replace("mod:", "node:"),)),
        "public_export_ids": list(public_export_ids),
    }


def _evidence(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "modules": [_module("mod:small", loc=10, member_ids=("node:leaf",))],
    }
    fields.update(overrides)
    return fields


def _scope(
    module_id: str,
    *,
    path: str | None = None,
    member_ids: tuple[str, ...] | None = None,
    evidence_label: str | None = None,
) -> dict[str, Any]:
    suffix = module_id.split(":", 1)[-1]
    return {
        "module_id": module_id,
        "write_paths": [path or f"pkg/{suffix}.py"],
        "member_ids": list(member_ids or (module_id.replace("mod:", "node:"),)),
        "evidence_cids": [_cid(evidence_label or f"ev:{module_id}")],
        "validation_commands": list(VALIDATION),
    }


def _goals_for(*modules: dict[str, Any], **evidence: Any):
    detection = detect_monolith_opportunities(
        _evidence(modules=list(modules), **evidence)
    )
    return compile_durable_goals(detection)


def _compile(
    *modules: dict[str, Any],
    scopes: list[dict[str, Any]] | None = None,
    prior_failures: list[dict[str, Any]] | None = None,
    **evidence: Any,
):
    goals = _goals_for(*modules, **evidence)
    resolved_scopes = scopes or [_scope(item.module_id) for item in goals.goals]
    payload: dict[str, Any] = {
        "tree_id": TREE_ID,
        "goals": goals,
        "scopes": resolved_scopes,
    }
    if prior_failures is not None:
        payload["prior_failures"] = prior_failures
    return compile_backlog(payload)


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-037"
    assert GOAL_ID == "SPAR-G071"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert REFINED_SUBGOAL_INTERFACE == "RefinedSubgoal@1"
    assert COMPILED_TASK_INTERFACE == "CompiledTask@1"
    assert BACKLOG_REPAIR_DECISION_INTERFACE == "BacklogRepairDecision@1"
    assert SUBGOAL_REFINEMENT_RECEIPT_INTERFACE == "SubgoalRefinementReceipt@1"
    assert (
        EXTRACTION_WAVE_TASK_COMPILATION_RECEIPT_INTERFACE
        == "ExtractionWaveTaskCompilationReceipt@1"
    )
    assert PARTITION_SUBGOAL_REFINER_INTERFACE == "PartitionSubgoalRefiner@1"
    assert EXTRACTION_WAVE_TASK_COMPILER_INTERFACE == "ExtractionWaveTaskCompiler@1"
    assert TASK_CONTRACT_VERSION == "1"
    assert POLICY_ID == "task-compiler-policy@1"
    assert POLICY_REVISION == "1"
    assert ANALYZER_ID.endswith("task_compiler@1")
    assert REFINER_ID.endswith("partition_subgoal_refiner@1")
    assert COMPILER_ID.endswith("extraction_wave_task_compiler@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()
    assert TASK_KIND_ORDER == (
        "inventory",
        "extraction",
        "state",
        "boundary",
        "facade",
        "validation",
        "repair",
        "rescan",
        "retirement",
    )
    assert DECLARED_TASK_KINDS == set(TASK_KIND_ORDER)
    assert ALWAYS_KINDS == ("inventory", "validation", "rescan")
    assert WAVE_KINDS == ("extraction", "state", "boundary", "facade")
    assert ACCEPTANCE_IDS == SPAR036_ACCEPTANCE_IDS
    assert ROLLBACK_MODE == SPAR019_ROLLBACK_MODE


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "task synthesis"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert TASK_CAN_AUTHORIZE_COMPLETION is False
    assert TASK_CAN_AUTHORIZE_TRANSITION is False
    assert TASK_CAN_CREATE_AUTHORITY is False
    assert TASK_CAN_RETIRE_FACADE is False
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
    assert ONE_PACKET_AT_A_TIME is True
    assert IDENTICAL_RETRY_WITHOUT_EVIDENCE is False
    profile = task_compiler_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    assert "spar_narrow" in EXISTING_ADAPTER_AUTHORITIES
    assert "bounded_source_edit" in ALLOWED_EFFECTS
    assert "network" in FORBIDDEN_EFFECTS


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "PartitionSubgoalRefiner" in names
    assert "ExtractionWaveTaskCompiler" in names
    assert "RefinedSubgoal" in names
    assert "CompiledTask" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "PartitionSubgoalRefiner" in exports
    assert "ExtractionWaveTaskCompiler" in exports
    assert "refine_partition_subgoals" in exports
    assert "compile_extraction_wave_tasks" in exports
    assert "compile_backlog" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_oversized_finding_emits_inventory_extraction_validation_rescan() -> None:
    compiled = _compile(_module("mod:core", loc=520, member_ids=("node:core",)))
    kinds = tuple(item.kind for item in compiled.tasks)
    assert kinds == (
        TaskKind.INVENTORY.value,
        TaskKind.EXTRACTION.value,
        TaskKind.VALIDATION.value,
        TaskKind.RESCAN.value,
    )
    extraction = compiled.tasks[1]
    assert extraction.module_id == "mod:core"
    assert extraction.write_paths == ("pkg/core.py",)
    assert extraction.rollback_mode == ROLLBACK_MODE
    assert extraction.generic_prompt_forbidden is True
    assert extraction.one_packet_at_a_time is True
    assert extraction.can_authorize_completion is False
    assert extraction.unrestricted_scope is False
    assert extraction.depends_on == (TaskKind.INVENTORY.value,)
    assert compiled.wave_task_cids == (extraction.task_cid,)
    assert compiled.one_packet_at_a_time is True
    assert compiled.can_retire_facade is False


def test_public_and_state_risks_emit_facade_state_boundary_and_retirement() -> None:
    compiled = _compile(
        _module(
            "mod:api",
            loc=40,
            member_ids=("node:a", "node:b"),
            public_export_ids=tuple(f"exp:{index}" for index in range(8)),
        ),
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
        scopes=[_scope("mod:api", member_ids=("node:a", "node:b"))],
    )
    kinds = tuple(item.kind for item in compiled.tasks)
    assert TaskKind.STATE.value in kinds
    assert TaskKind.BOUNDARY.value in kinds
    assert TaskKind.FACADE.value in kinds
    assert TaskKind.RETIREMENT.value in kinds
    assert TaskKind.EXTRACTION.value not in kinds
    facade = next(item for item in compiled.tasks if item.kind == TaskKind.FACADE.value)
    retirement = next(
        item for item in compiled.tasks if item.kind == TaskKind.RETIREMENT.value
    )
    assert facade.can_retire_facade is False
    assert retirement.can_retire_facade is False
    assert TaskKind.FACADE.value in retirement.depends_on
    wave_kinds = [
        item.kind
        for item in compiled.tasks
        if item.task_cid in compiled.wave_task_cids
    ]
    assert wave_kinds == [
        TaskKind.STATE.value,
        TaskKind.BOUNDARY.value,
        TaskKind.FACADE.value,
    ]


def test_missing_or_escaped_scope_fails_closed() -> None:
    goals = _goals_for(_module("mod:core", loc=520, member_ids=("node:core",)))
    with pytest.raises(TaskCompilerError, match="unrestricted scope"):
        refine_partition_subgoals(
            {"tree_id": TREE_ID, "goals": goals, "scopes": []}
        )
    with pytest.raises(TaskCompilerError, match="missing exact scope"):
        refine_partition_subgoals(
            {
                "tree_id": TREE_ID,
                "goals": goals,
                "scopes": [_scope("mod:other")],
            }
        )
    with pytest.raises(TaskCompilerError, match="exact repository-relative path"):
        refine_partition_subgoals(
            {
                "tree_id": TREE_ID,
                "goals": goals,
                "scopes": [_scope("mod:core", path="../secret.py")],
            }
        )
    with pytest.raises(TaskCompilerError, match="exact repository-relative path"):
        refine_partition_subgoals(
            {
                "tree_id": TREE_ID,
                "goals": goals,
                "scopes": [_scope("mod:core", path="/abs/core.py")],
            }
        )


def test_vector_and_generic_prompt_cannot_admit_tasks() -> None:
    goals = _goals_for(_module("mod:core", loc=520, member_ids=("node:core",)))
    with pytest.raises(TaskCompilerError, match="cannot admit"):
        refine_partition_subgoals(
            {
                "tree_id": TREE_ID,
                "goals": goals,
                "scopes": [_scope("mod:core")],
                "vector_candidate": {"mod:core": 1},
            }
        )
    with pytest.raises(TaskCompilerError, match="generic prompt"):
        refine_partition_subgoals(
            {
                "tree_id": TREE_ID,
                "goals": goals,
                "scopes": [_scope("mod:core")],
                "fix_prompt": "fix the monolith",
            }
        )


def test_identical_retry_without_new_evidence_is_blocked() -> None:
    goals = _goals_for(_module("mod:core", loc=520, member_ids=("node:core",)))
    scope = _scope("mod:core")
    first = compile_backlog(
        {"tree_id": TREE_ID, "goals": goals, "scopes": [scope]}
    )
    extraction = next(
        item for item in first.tasks if item.kind == TaskKind.EXTRACTION.value
    )
    retried = compile_backlog(
        {
            "tree_id": TREE_ID,
            "goals": goals,
            "scopes": [scope],
            "prior_failures": [
                {
                    "module_id": "mod:core",
                    "kind": TaskKind.EXTRACTION.value,
                    "write_paths": list(extraction.write_paths),
                    "evidence_cids": list(extraction.evidence_cids),
                }
            ],
        }
    )
    kinds = tuple(item.kind for item in retried.tasks)
    assert TaskKind.EXTRACTION.value not in kinds
    assert TaskKind.REPAIR.value not in kinds
    assert extraction.retry_fingerprint in retried.blocked_retry_fingerprints
    assert retried.can_authorize_completion is False
    decision = PartitionSubgoalRefiner().refine(
        {
            "tree_id": TREE_ID,
            "goals": goals,
            "scopes": [scope],
            "prior_failures": [
                {
                    "module_id": "mod:core",
                    "kind": TaskKind.EXTRACTION.value,
                    "write_paths": list(extraction.write_paths),
                    "evidence_cids": list(extraction.evidence_cids),
                }
            ],
        }
    ).repair_decisions[0]
    assert decision.retry_blocked is True
    assert decision.repair_admitted is False
    assert decision.identical_retry_without_evidence is True


def test_new_evidence_admits_repair_and_is_not_identical_retry() -> None:
    goals = _goals_for(_module("mod:core", loc=520, member_ids=("node:core",)))
    first_scope = _scope("mod:core", evidence_label="ev:old")
    first = compile_backlog(
        {"tree_id": TREE_ID, "goals": goals, "scopes": [first_scope]}
    )
    extraction = next(
        item for item in first.tasks if item.kind == TaskKind.EXTRACTION.value
    )
    repaired = compile_backlog(
        {
            "tree_id": TREE_ID,
            "goals": goals,
            "scopes": [_scope("mod:core", evidence_label="ev:new")],
            "prior_failures": [
                {
                    "module_id": "mod:core",
                    "kind": TaskKind.EXTRACTION.value,
                    "write_paths": list(extraction.write_paths),
                    "evidence_cids": list(extraction.evidence_cids),
                }
            ],
        }
    )
    kinds = tuple(item.kind for item in repaired.tasks)
    assert TaskKind.EXTRACTION.value in kinds
    assert TaskKind.REPAIR.value in kinds
    repair = next(item for item in repaired.tasks if item.kind == TaskKind.REPAIR.value)
    assert repair.evidence_cids != extraction.evidence_cids
    assert repair.retry_fingerprint != extraction.retry_fingerprint
    assert TaskKind.VALIDATION.value in repair.depends_on


def test_extraction_wave_is_one_packet_at_a_time_and_ordered() -> None:
    compiled = _compile(
        _module("mod:core", loc=520, member_ids=("node:core",)),
        _module(
            "mod:api",
            loc=40,
            public_export_ids=tuple(f"exp:{index}" for index in range(8)),
        ),
        scopes=[
            _scope("mod:core"),
            _scope("mod:api", member_ids=("node:api",)),
        ],
    )
    wave = [item for item in compiled.tasks if item.kind in WAVE_KINDS]
    assert [item.task_cid for item in wave] == list(compiled.wave_task_cids)
    assert [item.kind for item in wave] == [
        TaskKind.EXTRACTION.value,
        TaskKind.FACADE.value,
    ]
    assert all(item.one_packet_at_a_time is True for item in wave)
    indexes = [TASK_KIND_ORDER.index(item.kind) for item in compiled.tasks]
    assert indexes == sorted(indexes)
    encoded = encode_canonical_receipt(compiled)
    restored = decode_canonical_compilation_receipt(encoded)
    assert restored == compiled
    assert restored.receipt_cid == compiled.receipt_cid


def test_refinement_is_deterministic_nomination_only() -> None:
    goals = _goals_for(_module("mod:core", loc=520, member_ids=("node:core",)))
    payload = {
        "tree_id": TREE_ID,
        "goals": goals,
        "scopes": [_scope("mod:core")],
    }
    first = refine_partition_subgoals(payload)
    second = PartitionSubgoalRefiner().refine(payload)
    assert first.receipt_cid == second.receipt_cid
    assert first.can_authorize_completion is False
    assert first.can_authorize_transition is False
    assert first.can_retire_facade is False
    assert first.generic_prompt_forbidden is True
    encoded = encode_canonical_receipt(first)
    restored = decode_canonical_refinement_receipt(encoded)
    assert restored == first
    assert restored.receipt_cid == first.receipt_cid
    compiled = ExtractionWaveTaskCompiler().compile(first, scopes=payload["scopes"])
    assert compiled.compiler_id == COMPILER_ID
    assert compiled.rollback_mode == ROLLBACK_MODE


def test_identity_excludes_observational_fields() -> None:
    compiled = _compile(_module("mod:core", loc=520, member_ids=("node:core",)))
    encoded = compiled.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(encoded))
    encoded["timestamp"] = "now"
    with pytest.raises(TaskCompilerError, match="observational"):
        ExtractionWaveTaskCompilationReceipt.from_dict(encoded)
    subgoal = refine_partition_subgoals(
        {
            "tree_id": TREE_ID,
            "goals": _goals_for(_module("mod:core", loc=520, member_ids=("node:core",))),
            "scopes": [_scope("mod:core")],
        }
    ).subgoals[0]
    dirty = subgoal.to_dict()
    dirty["model_output"] = "guess"
    with pytest.raises(TaskCompilerError, match="observational"):
        RefinedSubgoal.from_dict(dirty)


def test_receipt_cannot_claim_authority_flags() -> None:
    compiled = _compile(_module("mod:core", loc=520, member_ids=("node:core",)))
    payload = compiled.to_dict()
    payload["can_authorize_completion"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(TaskCompilerError, match="can_authorize_completion"):
        ExtractionWaveTaskCompilationReceipt.from_dict(payload)
    payload = compiled.tasks[0].to_dict()
    payload["can_retire_facade"] = True
    payload["task_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "task_cid"}
    )
    with pytest.raises(TaskCompilerError, match="can_retire_facade"):
        CompiledTask.from_dict(payload)


def test_tree_mismatch_and_empty_goals_fail_closed() -> None:
    with pytest.raises(TaskCompilerError, match="requires durable goals"):
        refine_partition_subgoals(
            {
                "tree_id": TREE_ID,
                "goals": detect_monolith_opportunities(_evidence()),
                "scopes": [_scope("mod:small")],
            }
        )
    goals = _goals_for(_module("mod:core", loc=520, member_ids=("node:core",)))
    with pytest.raises(TaskCompilerError, match="tree_id"):
        refine_partition_subgoals(
            {
                "tree_id": OTHER_TREE,
                "goals": goals,
                "scopes": [_scope("mod:core")],
            }
        )
    compiled = _compile(_module("mod:core", loc=520, member_ids=("node:core",)))
    payload = compiled.to_dict()
    payload["tree_id"] = OTHER_TREE
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(TaskCompilerError, match="tree_id"):
        ExtractionWaveTaskCompilationReceipt.from_dict(payload)


def test_generic_prompt_and_incomplete_acceptance_fail_closed() -> None:
    compiled = _compile(_module("mod:core", loc=520, member_ids=("node:core",)))
    payload = compiled.tasks[0].to_dict()
    payload["generic_prompt_forbidden"] = False
    payload["task_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "task_cid"}
    )
    with pytest.raises(TaskCompilerError, match="generic prompt"):
        CompiledTask.from_dict(payload)
    with pytest.raises(TaskCompilerError, match="every acceptance identifier"):
        CompiledTask(
            tree_id=TREE_ID,
            goal_cid=compiled.tasks[0].goal_cid,
            finding_cid=compiled.tasks[0].finding_cid,
            subgoal_cid=compiled.tasks[0].subgoal_cid,
            module_id="mod:core",
            kind=TaskKind.INVENTORY,
            write_paths=("pkg/core.py",),
            member_ids=("node:core",),
            evidence_cids=compiled.tasks[0].evidence_cids,
            validation_commands=VALIDATION,
            acceptance_ids=("exact-current-tree",),
        )


def test_wave_binding_requires_all_three_identifiers() -> None:
    goals = _goals_for(_module("mod:core", loc=520, member_ids=("node:core",)))
    refinement = refine_partition_subgoals(
        {
            "tree_id": TREE_ID,
            "goals": goals,
            "scopes": [_scope("mod:core")],
        }
    )
    with pytest.raises(TaskCompilerError, match="worktree, lease, and fence"):
        compile_extraction_wave_tasks(
            refinement,
            scopes=[_scope("mod:core")],
            worktree_id=_cid("worktree"),
        )
    bound = compile_extraction_wave_tasks(
        refinement,
        scopes=[_scope("mod:core")],
        worktree_id=_cid("worktree"),
        lease_id=_cid("lease"),
        fence_id=_cid("fence"),
    )
    assert bound.worktree_id == _cid("worktree")
    assert bound.lease_id == _cid("lease")
    assert bound.fence_id == _cid("fence")
    assert bound.one_packet_at_a_time is True


def test_blocked_retry_cannot_be_re_emitted_on_receipt() -> None:
    compiled = _compile(_module("mod:core", loc=520, member_ids=("node:core",)))
    with pytest.raises(TaskCompilerError, match="blocked identical retry"):
        ExtractionWaveTaskCompilationReceipt(
            tree_id=TREE_ID,
            tasks=compiled.tasks,
            blocked_retry_fingerprints=(compiled.tasks[0].retry_fingerprint,),
        )


def test_durable_goal_round_trip_through_refiner() -> None:
    detection = detect_monolith_opportunities(
        _evidence(modules=[_module("mod:core", loc=520, member_ids=("node:core",))])
    )
    finding = detection.findings[0]
    assert finding.kind == FindingKind.OVERSIZED.value
    assert RiskKind.OVERSIZED_LOC.value in finding.risks
    compiled = compile_backlog(
        {
            "tree_id": TREE_ID,
            "goals": detection,
            "scopes": [_scope("mod:core")],
        }
    )
    restored = ExtractionWaveTaskCompilationReceipt.from_dict(compiled.to_dict())
    assert restored.receipt_cid == compiled.receipt_cid
    assert restored.tasks[0].autonomy_tier == AutonomyTier.B.value
    goal = DurableGoal.from_dict(compile_durable_goals(detection).goals[0].to_dict())
    refined = refine_partition_subgoals(
        {
            "tree_id": TREE_ID,
            "goals": [goal],
            "scopes": [_scope("mod:core")],
        }
    )
    assert refined.subgoals[0].finding_cid == finding.finding_cid


def test_repair_decision_rejects_identical_evidence_admission() -> None:
    evidence = (_cid("same"),)
    fingerprint = retry_fingerprint(
        kind=TaskKind.REPAIR.value,
        module_id="mod:core",
        write_paths=("pkg/core.py",),
        evidence_cids=evidence,
    )
    with pytest.raises(TaskCompilerError, match="repair requires new evidence"):
        BacklogRepairDecision(
            module_id="mod:core",
            fingerprint=fingerprint,
            prior_evidence_cids=evidence,
            current_evidence_cids=evidence,
            retry_blocked=False,
            repair_admitted=True,
        )
    with pytest.raises(TaskCompilerError, match="blocked retry cannot admit repair"):
        BacklogRepairDecision(
            module_id="mod:core",
            fingerprint=fingerprint,
            prior_evidence_cids=evidence,
            current_evidence_cids=(_cid("other"),),
            retry_blocked=True,
            repair_admitted=True,
        )


def test_unknown_scope_and_duplicate_module_fail_closed() -> None:
    goals = _goals_for(_module("mod:core", loc=520, member_ids=("node:core",)))
    with pytest.raises(TaskCompilerError, match="unknown scope"):
        refine_partition_subgoals(
            {
                "tree_id": TREE_ID,
                "goals": goals,
                "scopes": [_scope("mod:core"), _scope("mod:extra")],
            }
        )
    with pytest.raises(TaskCompilerError, match="duplicate scope"):
        refine_partition_subgoals(
            {
                "tree_id": TREE_ID,
                "goals": goals,
                "scopes": [_scope("mod:core"), _scope("mod:core", path="pkg/other.py")],
            }
        )
