from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_compiler import (
    CompilationIssueCode,
    CompilationIssueSeverity,
    CompilationStatus,
    FormalPlanCompiler,
    compile_formal_plan,
    compile_formal_plan_json,
    write_formal_plan_compiler_input_duckdb,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_logic_vocabulary import (
    TDFOL,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_planning_contracts import (
    EventKind,
    RefinementMode,
)


def _source() -> dict[str, object]:
    return {
        "schema": "fixture/formal-plan-input@1",
        "repository_tree_id": "tree:candidate",
        "objectives": [
            {
                "goal_id": "G12.S1",
                "goal_cid": "goal:cid:g12-s1",
                "owner_actor_id": "owner:supervisor",
                "title": "Compile formal plans",
                "acceptance_criteria": ["Every task has retained evidence."],
            }
        ],
        "taskboard": [
            {
                "task_id": "REF-275",
                "task_cid": "task:cid:275",
                "goal_id": "G12.S1",
                "actor_id": "agent:alpha",
                "resource_needs": ["cpu", "duckdb"],
                "changed_ast_scopes": ["symbol:cid:contracts"],
                "acceptance_criteria": ["contract tests pass"],
                "validation_commands": ["pytest test_contracts.py"],
                "lease": {
                    "lease_cid": "lease:cid:275",
                    "holder_id": "agent:alpha",
                    "fencing_token": 7,
                },
            },
            {
                "task_id": "REF-276",
                "task_cid": "task:cid:276",
                "goal_id": "G12.S1",
                "depends_on": ["REF-275"],
                "actor_id": "agent:beta",
                "changed_ast_scopes": ["symbol:cid:compiler"],
                "acceptance_criteria": [
                    {
                        "kind": "test",
                        "validation_commands": ["pytest test_compiler.py"],
                        "statement": "JSON and DuckDB agree",
                    }
                ],
                "deadline": 12,
            },
        ],
        "ast_records": [
            {
                "symbol_cid": "symbol:cid:contracts",
                "tree_cid": "tree:candidate",
                "task_cid": "task:cid:275",
                "symbol": "FormalWorkPlan",
            },
            {
                "symbol_cid": "symbol:cid:compiler",
                "tree_cid": "tree:candidate",
                "task_id": "REF-276",
                "symbol": "FormalPlanCompiler",
            },
        ],
        "proof_policy": {
            "policy_cid": "policy:cid:g12",
            "minimum_code_assurance": "candidate",
            "freshness_seconds": 3600,
            "fallback_check_ids": ["fallback:pytest"],
            "required_evidence": [
                {
                    "kind": "plan_check",
                    "subject_ids": ["REF-276"],
                    "source_scope_ids": ["symbol:cid:compiler"],
                }
            ],
        },
        "evidence_records": [
            {
                "evidence_cid": "evidence:cid:prior-contract-test",
                "task_cid": "task:cid:275",
                "kind": "test",
            }
        ],
    }


def test_compiles_all_supervisor_semantics_and_preserves_cids() -> None:
    result = compile_formal_plan(_source())

    assert result.status is CompilationStatus.COMPILED
    assert result.valid and result.supported and result.plan is not None
    plan = result.plan
    assert plan.repository_tree_id == "tree:candidate"
    assert {goal.goal_id for goal in plan.goals} == {"goal:cid:g12-s1"}
    assert {task.task_id for task in plan.tasks} == {
        "task:cid:275",
        "task:cid:276",
    }
    task = next(item for item in plan.tasks if item.task_id == "task:cid:276")
    assert task.depends_on == ("task:cid:275",)
    assert task.metadata["changed_ast_scope_ids"] == ["symbol:cid:compiler"]
    assert task.evidence_requirement_ids
    assert {
        "goal:cid:g12-s1",
        "task:cid:275",
        "task:cid:276",
        "symbol:cid:contracts",
        "symbol:cid:compiler",
        "policy:cid:g12",
        "evidence:cid:prior-contract-test",
        "lease:cid:275",
    }.issubset(set(plan.source_ids))
    assert any(item.kind is EventKind.ASSIGNED for item in plan.events)
    assert any(item.metadata.get("lease_ids") for item in plan.events)
    assert any(
        constraint.kind.value == "dependency_order" for constraint in plan.temporal_constraints
    )
    assert any(constraint.kind.value == "deadline" for constraint in plan.temporal_constraints)
    assert any(
        actor.actor_id == "agent:alpha" and set(actor.capabilities) == {"cpu", "duckdb"}
        for actor in plan.actors
    )
    assert result.formulas
    assert all(result.formula_by_id(item.formula_id) is item for item in result.formulas)
    graph_record_ids = {str(node["record_id"]) for node in result.graph_projection.nodes}
    assert {
        "symbol:cid:contracts",
        "symbol:cid:compiler",
        "policy:cid:g12",
        "evidence:cid:prior-contract-test",
        "lease:cid:275",
    }.issubset(graph_record_ids)
    assert any(node["kind"] == "predicate" for node in result.graph_projection.nodes)


def test_records_all_descriptive_abstractions_without_parsing_them() -> None:
    result = compile_formal_plan(_source())

    abstraction_paths = {
        issue.path
        for issue in result.issues
        if issue.severity is CompilationIssueSeverity.ABSTRACTION
    }
    assert "$.objectives[0].title" in abstraction_paths
    assert "$.ast[0].symbol" in abstraction_paths
    assert "$.ast[1].symbol" in abstraction_paths
    assert result.plan is not None
    assert set(result.plan.abstraction_ids) == set(result.abstraction_ids)


def test_json_and_duckdb_have_identical_plan_and_graph_identity(
    tmp_path: Path,
) -> None:
    pytest.importorskip("duckdb")
    source = _source()
    json_result = FormalPlanCompiler().compile(source)
    database_path = write_formal_plan_compiler_input_duckdb(tmp_path / "formal_plan.duckdb", source)
    duckdb_result = FormalPlanCompiler().compile_duckdb(database_path)

    assert json_result.status is duckdb_result.status is CompilationStatus.COMPILED
    assert json_result.source_identity == duckdb_result.source_identity
    assert json_result.plan_id == duckdb_result.plan_id
    assert json_result.graph_projection.graph_id == (duckdb_result.graph_projection.graph_id)
    assert json_result.graph_projection.canonical_records() == (
        duckdb_result.graph_projection.canonical_records()
    )


def test_malformed_json_is_an_explicit_invalid_result() -> None:
    result = FormalPlanCompiler().compile_json('{"tasks": [}')

    assert result.status is CompilationStatus.INVALID
    assert result.plan is None
    assert result.issues[0].code is CompilationIssueCode.MALFORMED_JSON


def test_cycles_and_unknown_dependencies_are_explicitly_invalid() -> None:
    source = _source()
    tasks = source["taskboard"]
    assert isinstance(tasks, list)
    tasks[0]["depends_on"] = ["REF-276"]

    cycle = compile_formal_plan(source)
    assert cycle.status is CompilationStatus.INVALID
    assert CompilationIssueCode.CYCLE in {item.code for item in cycle.issues}

    source = _source()
    tasks = source["taskboard"]
    assert isinstance(tasks, list)
    tasks[1]["depends_on"] = ["REF-999"]
    missing = compile_formal_plan(source)
    assert missing.status is CompilationStatus.INVALID
    assert CompilationIssueCode.UNKNOWN_DEPENDENCY in {item.code for item in missing.issues}


def test_ambiguous_effects_are_invalid_and_not_silently_selected() -> None:
    source = _source()
    tasks = source["taskboard"]
    assert isinstance(tasks, list)
    tasks[1]["effects"] = [
        {
            "operation": "assign",
            "fluent_id": "task:cid:276:state",
            "value": "completed",
        },
        {
            "operation": "assign",
            "fluent_id": "task:cid:276:state",
            "value": "failed",
        },
    ]

    result = compile_formal_plan(source)

    assert result.status is CompilationStatus.INVALID
    assert result.plan is None
    assert CompilationIssueCode.AMBIGUOUS_EFFECT in {item.code for item in result.issues}


def test_missing_or_unreviewed_semantics_are_explicitly_unsupported() -> None:
    missing = _source()
    missing["proof_policy"] = []
    result = compile_formal_plan(missing)
    assert result.status is CompilationStatus.UNSUPPORTED
    assert CompilationIssueCode.MISSING_SEMANTICS in {item.code for item in result.issues}

    unknown = _source()
    tasks = unknown["taskboard"]
    assert isinstance(tasks, list)
    tasks[0]["semantic_operator"] = "model-invented-operator"
    result = compile_formal_plan(unknown)
    assert result.status is CompilationStatus.UNSUPPORTED
    assert "$.tasks[0].semantic_operator" in result.unsupported_fields


def test_storage_order_does_not_change_identity() -> None:
    source = _source()
    reverse = _source()
    for key in ("objectives", "taskboard", "ast_records", "evidence_records"):
        value = reverse[key]
        assert isinstance(value, list)
        value.reverse()

    first = compile_formal_plan(source)
    second = compile_formal_plan(reverse)

    assert first.plan_id == second.plan_id
    assert first.graph_projection.graph_id == second.graph_projection.graph_id


def _source_with_reviewed_subgoals() -> dict[str, object]:
    source = _source()
    objectives = source["objectives"]
    tasks = source["taskboard"]
    assert isinstance(objectives, list)
    assert isinstance(tasks, list)
    objectives[0]["subgoals"] = [
        {
            "subgoal_id": "SG-CONTRACTS",
            "subgoal_cid": "subgoal:cid:contracts",
            "source_id": "review:subgoal:contracts",
            "acceptance_criteria": [
                {
                    "kind": "review",
                    "source_scope_ids": ["symbol:cid:contracts"],
                    "check_ids": ["review:contracts"],
                }
            ],
        },
        {
            "subgoal_id": "SG-COMPILER",
            "subgoal_cid": "subgoal:cid:compiler",
            "parent_id": "SG-CONTRACTS",
            "refinement_mode": "equivalent",
            "depends_on": ["SG-CONTRACTS"],
            "source_ids": ["review:subgoal:compiler", "criterion:compiler"],
            "scope_ids": ["symbol:cid:compiler"],
            "validation_commands": ["pytest test_compiler.py"],
        },
    ]
    tasks[0]["subgoal_id"] = "SG-CONTRACTS"
    tasks[1]["subgoal_cid"] = "subgoal:cid:compiler"
    return source


def test_compiles_reviewed_subgoal_hierarchy_with_all_canonical_bindings() -> None:
    result = compile_formal_plan(_source_with_reviewed_subgoals())

    assert result.status is CompilationStatus.COMPILED
    assert result.plan is not None
    plan = result.plan
    assert {item.subgoal_id for item in plan.subgoals} == {
        "subgoal:cid:contracts",
        "subgoal:cid:compiler",
    }
    contracts = next(item for item in plan.subgoals if item.subgoal_id == "subgoal:cid:contracts")
    compiler = next(item for item in plan.subgoals if item.subgoal_id == "subgoal:cid:compiler")
    assert contracts.goal_id == "goal:cid:g12-s1"
    assert contracts.parent_id == contracts.goal_id
    assert contracts.refinement_mode is RefinementMode.SUFFICIENT
    assert compiler.goal_id == contracts.goal_id
    assert compiler.parent_id == contracts.subgoal_id
    assert compiler.refinement_mode is RefinementMode.EQUIVALENT
    assert compiler.depends_on == (contracts.subgoal_id,)
    assert contracts.evidence_requirement_ids
    assert compiler.evidence_requirement_ids
    assert all(
        item.subgoal_id in requirement.subject_ids
        for item in plan.subgoals
        for requirement in plan.evidence_requirements
        if requirement.requirement_id in item.evidence_requirement_ids
    )
    assert set(contracts.metadata["source_ids"]) == {
        "goal:cid:g12-s1",
        "review:subgoal:contracts",
        "subgoal:cid:contracts",
    }
    assert {
        "review:subgoal:contracts",
        "review:subgoal:compiler",
        "criterion:compiler",
    }.issubset(plan.source_ids)

    for subgoal in plan.subgoals:
        expected = TDFOL.subgoal_satisfaction(subgoal.subgoal_id, plan.trace_bound)
        assert subgoal.satisfaction_formula_id == expected.formula_id
        assert expected in plan.formulas
        assert result.formula_by_id(expected.formula_id) == expected
    task_bindings = {item.task_id: item.subgoal_id for item in plan.tasks}
    assert task_bindings == {
        "task:cid:275": "subgoal:cid:contracts",
        "task:cid:276": "subgoal:cid:compiler",
    }

    node_by_record = {
        str(node["record_id"]): str(node["node_id"]) for node in result.graph_projection.nodes
    }
    edge_triples = {
        (str(edge["kind"]), str(edge["source"]), str(edge["target"]))
        for edge in result.graph_projection.edges
    }
    assert (
        "refines",
        node_by_record[compiler.subgoal_id],
        node_by_record[contracts.subgoal_id],
    ) in edge_triples
    assert (
        "depends_on",
        node_by_record[compiler.subgoal_id],
        node_by_record[contracts.subgoal_id],
    ) in edge_triples
    assert any(
        kind == "requires_evidence" and source_id == node_by_record[compiler.subgoal_id]
        for kind, source_id, _ in edge_triples
    )


def test_subgoal_storage_order_does_not_change_compilation_identity() -> None:
    first_source = _source_with_reviewed_subgoals()
    second_source = _source_with_reviewed_subgoals()
    objectives = second_source["objectives"]
    assert isinstance(objectives, list)
    objectives[0]["subgoals"].reverse()

    first = compile_formal_plan(first_source)
    second = compile_formal_plan(second_source)

    assert first.status is second.status is CompilationStatus.COMPILED
    assert first.source_identity == second.source_identity
    assert first.plan_id == second.plan_id
    assert first.graph_projection.graph_id == second.graph_projection.graph_id


def test_subgoal_claimed_formula_and_unreviewed_semantics_are_unsupported() -> None:
    claimed = _source_with_reviewed_subgoals()
    objectives = claimed["objectives"]
    assert isinstance(objectives, list)
    objectives[0]["subgoals"][0]["satisfaction_formula_id"] = "formula:model-invented"

    result = compile_formal_plan(claimed)

    assert result.status is CompilationStatus.UNSUPPORTED
    assert result.plan is None
    assert CompilationIssueCode.UNKNOWN_SEMANTIC in {issue.code for issue in result.issues}

    semantic = _source_with_reviewed_subgoals()
    objectives = semantic["objectives"]
    assert isinstance(objectives, list)
    objectives[0]["subgoals"][0]["predicate"] = "model_says_done"

    result = compile_formal_plan(semantic)

    assert result.status is CompilationStatus.UNSUPPORTED
    assert any(
        issue.path.endswith(".predicate") and issue.severity is CompilationIssueSeverity.UNSUPPORTED
        for issue in result.issues
    )


def test_subgoal_cycles_unknown_bindings_and_missing_evidence_fail_closed() -> None:
    cyclic = _source_with_reviewed_subgoals()
    objectives = cyclic["objectives"]
    assert isinstance(objectives, list)
    objectives[0]["subgoals"][0]["parent_id"] = "SG-COMPILER"

    result = compile_formal_plan(cyclic)

    assert result.status is CompilationStatus.INVALID
    assert CompilationIssueCode.CYCLE in {issue.code for issue in result.issues}

    unknown_task = _source_with_reviewed_subgoals()
    tasks = unknown_task["taskboard"]
    assert isinstance(tasks, list)
    tasks[0]["subgoal_id"] = "SG-MISSING"
    result = compile_formal_plan(unknown_task)
    assert result.status is CompilationStatus.INVALID
    assert result.plan is None

    missing_evidence = _source_with_reviewed_subgoals()
    objectives = missing_evidence["objectives"]
    assert isinstance(objectives, list)
    objectives[0]["subgoals"][0].pop("acceptance_criteria")
    result = compile_formal_plan(missing_evidence)
    assert result.status is CompilationStatus.UNSUPPORTED
    assert CompilationIssueCode.MISSING_SEMANTICS in {issue.code for issue in result.issues}


def _campaign_task(
    task_id: str = "PGIR-060",
    *,
    depends_on: tuple[str, ...] = ("PGIR-014",),
    role: str = "campaign_control",
) -> dict[str, object]:
    return {
        "schema": "IRLearningCampaignTask@1",
        "task_id": task_id,
        "title": "Implement IR learning campaign contracts and APIs",
        "status": "todo",
        "completion": "validated-implementation",
        "is_schedulable": True,
        "priority": "P0",
        "track": "campaign",
        "parent_goal": "PGIR-G080",
        "subgoal": "campaign-work-graph",
        "owning_repository": "ipfs_accelerate_py",
        "owned_paths": [
            "ipfs_accelerate_py/agent_supervisor/objectives/",
            "ipfs_accelerate_py/agent_supervisor/planning/",
        ],
        "base_source_revisions": "SRCSET-1",
        "source_dataset_revisions": "RESULT(PGIR-011)",
        "data_split_identity": "RESULT(PGIR-012)",
        "compiler_identity": "RESULT(PGIR-021)",
        "decompiler_identity": "RESULT(PGIR-022)",
        "model_checkpoint_identity": "none",
        "objective": "Add IRLearningCampaign@1 and campaign operations",
        "depends_on": list(depends_on),
        "resource_profile": "RP-CPU-M",
        "expected_inputs": "frozen input root, task metadata contract",
        "expected_outputs": "strict campaign/role/task schemas",
        "allowed_effects": "existing objective/planning/control extension and tests",
        "prohibited_effects": "semantic definitions in accelerator, hidden labels",
        "acceptance_criteria": "every task field required by this board is validated",
        "required_proof_or_evaluation_evidence": "schema/parity/adversarial plan-admission tests",
        "lease_and_checkpoint_policy": "LEASE-DEFAULT",
        "rollback_procedure": "ROLLBACK-DEFAULT",
        "result_identity": f"RESULT({task_id})",
        "outputs": [
            "ipfs_accelerate_py/agent_supervisor/objectives/",
            "ipfs_accelerate_py/agent_supervisor/planning/",
        ],
        "validation": "python -m pytest -q test/api/test_agent_supervisor_formal_plan_compiler.py",
        "bundle": "pgir/campaign/contracts",
        "parallel_lane": "campaign-contracts",
        "predicted_files": [
            "ipfs_accelerate_py/agent_supervisor/objectives/",
            "ipfs_accelerate_py/agent_supervisor/planning/",
        ],
        "conflict_policy": "campaign schema/control catalog exclusive",
        "work_graph_role": role,
    }


def _resolved_results() -> dict[str, str]:
    return {
        "PGIR-011": "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "PGIR-012": "baguqeerabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "PGIR-014": "baguqeeracccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "PGIR-021": "baguqeeradddddddddddddddddddddddddddddddddddddddddddddddddddd",
        "PGIR-022": "baguqeeraeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    }


def _campaign_payload(
    *,
    resolved: bool = True,
    tasks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    from ipfs_accelerate_py.agent_supervisor.objectives.ir_learning_campaign_contracts import (
        default_campaign_roles,
    )

    return {
        "schema": "IRLearningCampaign@1",
        "campaign_id": "campaign:pgir",
        "input_root_cid": "baguqeerarkgpz4xl663tlpfpiajjtxlya3b576lqzg5yd7nrthqgs2rm6v2q",
        "repository_tree_id": "tree:campaign",
        "owner_actor_id": "supervisor",
        "roles": [item.to_dict() for item in default_campaign_roles()],
        "tasks": tasks or [_campaign_task()],
        "operations": [
            "create",
            "plan",
            "status",
            "steer",
            "refill",
            "proof-replay",
            "compare",
            "promote",
            "reject",
            "report",
        ],
        "dependency_results": _resolved_results() if resolved else {},
    }


def test_campaign_imports_are_side_effect_free() -> None:
    from ipfs_accelerate_py.agent_supervisor.objectives import ir_learning_campaign
    from ipfs_accelerate_py.agent_supervisor.objectives import (
        ir_learning_campaign_contracts,
    )
    from ipfs_accelerate_py.agent_supervisor.planning import ir_learning_campaign_planner

    assert ir_learning_campaign.create_campaign is not None
    assert ir_learning_campaign_contracts.IR_LEARNING_CAMPAIGN_SCHEMA == "IRLearningCampaign@1"
    assert ir_learning_campaign_planner.expand_campaign_source({}) == {}


def test_campaign_compiles_all_board_fields_and_projects_dependencies() -> None:
    from ipfs_accelerate_py.agent_supervisor.objectives.ir_learning_campaign_contracts import (
        BOARD_TASK_FIELDS,
        IRLearningCampaign,
        REQUIRED_WORK_GRAPH_ROLES,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.ir_learning_campaign_planner import (
        project_campaign_for_admission,
    )

    payload = _campaign_payload()
    campaign = IRLearningCampaign.from_dict(payload)
    assert {role.role for role in campaign.roles} == set(REQUIRED_WORK_GRAPH_ROLES)
    assert set(campaign.tasks[0].to_dict()) >= set(BOARD_TASK_FIELDS)

    result = compile_formal_plan({"campaign": payload})
    assert result.status is CompilationStatus.COMPILED
    assert result.valid and result.plan is not None
    assert result.plan.repository_tree_id == "tree:campaign"
    assert result.admission_projection is not None
    assert result.admission_projection.action_ids
    assert result.admission_projection.proof_results == ()
    binding = result.plan.metadata["ir_learning_campaign_binding"]
    assert binding["campaign_schema"] == "IRLearningCampaign@1"
    assert "PGIR-060" in binding["lease_eligible_task_ids"]

    campaign_projection, formal = project_campaign_for_admission(payload)
    assert formal is not None
    assert campaign_projection.action_ids == ("PGIR-060",)
    assert campaign_projection.lease_eligible_task_ids == ("PGIR-060",)
    assert formal.action_ids
    assert formal.proof_results == ()


def test_campaign_dependency_projection_is_order_independent() -> None:
    first = _campaign_payload(
        tasks=[
            _campaign_task("PGIR-060", depends_on=("PGIR-014",)),
            _campaign_task("PGIR-061", depends_on=("PGIR-060",), role="resource"),
        ]
    )
    second = _campaign_payload(
        tasks=[
            _campaign_task("PGIR-061", depends_on=("PGIR-060",), role="resource"),
            _campaign_task("PGIR-060", depends_on=("PGIR-014",)),
        ]
    )
    first_result = compile_formal_plan({"campaign": first})
    second_result = compile_formal_plan({"campaign": second})
    assert first_result.status is second_result.status is CompilationStatus.COMPILED
    assert first_result.plan_id == second_result.plan_id
    assert first_result.admission_projection is not None
    assert second_result.admission_projection is not None
    assert (
        first_result.admission_projection.projection_id
        == second_result.admission_projection.projection_id
    )
    first_edges = first_result.plan.metadata["ir_learning_campaign_binding"]["projection_id"]
    second_edges = second_result.plan.metadata["ir_learning_campaign_binding"]["projection_id"]
    assert first_edges == second_edges


def test_campaign_unresolved_result_binds_revision_and_refuses_lease() -> None:
    from ipfs_accelerate_py.agent_supervisor.objectives.ir_learning_campaign_contracts import (
        CampaignLeaseDecision,
        IRLearningCampaign,
    )

    campaign = IRLearningCampaign.from_dict(_campaign_payload(resolved=False))
    revision = campaign.revision_for("PGIR-060")
    assert revision.lease_decision is CampaignLeaseDecision.BLOCKED
    assert "RESULT(PGIR-014)" in revision.unresolved_dependency_outputs
    assert campaign.lease_eligible_task_ids == ()

    result = compile_formal_plan({"campaign": _campaign_payload(resolved=False)})
    assert result.status is CompilationStatus.COMPILED
    assert result.plan is not None
    binding = result.plan.metadata["ir_learning_campaign_binding"]
    assert "PGIR-060" in binding["blocked_task_ids"]
    assert "PGIR-060" not in binding["lease_eligible_task_ids"]
    nested = next(item for item in result.plan.tasks).metadata.get("metadata") or {}
    assert nested.get("lease_eligible") is False


def test_campaign_missing_board_field_and_adversarial_inputs_fail_closed() -> None:
    missing = _campaign_payload()
    tasks = missing["tasks"]
    assert isinstance(tasks, list)
    assert isinstance(tasks[0], dict)
    tasks[0].pop("acceptance_criteria")
    result = compile_formal_plan({"campaign": missing})
    assert result.status is CompilationStatus.INVALID
    assert result.plan is None

    hidden = _campaign_payload()
    hidden["metadata"] = {"hidden_labels": ["secret-class"]}
    result = compile_formal_plan({"campaign": hidden})
    assert result.status is CompilationStatus.INVALID

    authority = _campaign_payload()
    authority["tasks"][0]["prompt_selected_authority"] = "promote"
    result = compile_formal_plan({"campaign": authority})
    assert result.status is CompilationStatus.INVALID

    cyclic = _campaign_payload(
        tasks=[
            _campaign_task("PGIR-060", depends_on=("PGIR-061",)),
            _campaign_task("PGIR-061", depends_on=("PGIR-060",), role="resource"),
        ]
    )
    result = compile_formal_plan({"campaign": cyclic})
    assert result.status is CompilationStatus.INVALID
    assert CompilationIssueCode.INVALID_SOURCE in {item.code for item in result.issues}

    truncated_roles = _campaign_payload()
    truncated_roles["roles"] = truncated_roles["roles"][:2]
    result = compile_formal_plan({"campaign": truncated_roles})
    assert result.status is CompilationStatus.INVALID


def test_campaign_json_and_mapping_compilation_are_parity_equivalent(tmp_path: Path) -> None:
    import json

    payload = {"campaign": _campaign_payload()}
    mapping_result = compile_formal_plan(payload)
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    json_result = compile_formal_plan_json(path)
    assert mapping_result.status is json_result.status is CompilationStatus.COMPILED
    assert mapping_result.plan_id == json_result.plan_id
    assert mapping_result.source_identity == json_result.source_identity
    assert mapping_result.admission_projection is not None
    assert json_result.admission_projection is not None
    assert (
        mapping_result.admission_projection.projection_id
        == json_result.admission_projection.projection_id
    )
