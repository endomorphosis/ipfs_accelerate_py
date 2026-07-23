from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_plan_compiler import (
    CompilationIssueCode,
    CompilationIssueSeverity,
    CompilationStatus,
    FormalPlanCompiler,
    compile_formal_plan,
    write_formal_plan_compiler_input_duckdb,
)
from ipfs_accelerate_py.agent_supervisor.formal_planning_contracts import (
    EventKind,
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
        constraint.kind.value == "dependency_order"
        for constraint in plan.temporal_constraints
    )
    assert any(
        constraint.kind.value == "deadline"
        for constraint in plan.temporal_constraints
    )
    assert any(
        actor.actor_id == "agent:alpha"
        and set(actor.capabilities) == {"cpu", "duckdb"}
        for actor in plan.actors
    )
    assert result.formulas
    assert all(result.formula_by_id(item.formula_id) is item for item in result.formulas)
    graph_record_ids = {
        str(node["record_id"]) for node in result.graph_projection.nodes
    }
    assert {
        "symbol:cid:contracts",
        "symbol:cid:compiler",
        "policy:cid:g12",
        "evidence:cid:prior-contract-test",
        "lease:cid:275",
    }.issubset(graph_record_ids)
    assert any(
        node["kind"] == "predicate" for node in result.graph_projection.nodes
    )


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
    database_path = write_formal_plan_compiler_input_duckdb(
        tmp_path / "formal_plan.duckdb", source
    )
    duckdb_result = FormalPlanCompiler().compile_duckdb(database_path)

    assert json_result.status is duckdb_result.status is CompilationStatus.COMPILED
    assert json_result.source_identity == duckdb_result.source_identity
    assert json_result.plan_id == duckdb_result.plan_id
    assert json_result.graph_projection.graph_id == (
        duckdb_result.graph_projection.graph_id
    )
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
    assert CompilationIssueCode.UNKNOWN_DEPENDENCY in {
        item.code for item in missing.issues
    }


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
    assert CompilationIssueCode.AMBIGUOUS_EFFECT in {
        item.code for item in result.issues
    }


def test_missing_or_unreviewed_semantics_are_explicitly_unsupported() -> None:
    missing = _source()
    missing["proof_policy"] = []
    result = compile_formal_plan(missing)
    assert result.status is CompilationStatus.UNSUPPORTED
    assert CompilationIssueCode.MISSING_SEMANTICS in {
        item.code for item in result.issues
    }

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
