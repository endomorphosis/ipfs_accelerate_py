from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_plan_compiler import (
    CompilationStatus,
    compile_formal_plan,
)
from ipfs_accelerate_py.agent_supervisor.formal_plan_context import (
    FORMAL_PLAN_MODEL_RESPONSE_SCHEMA,
    FormalPlanContextBudgetError,
    FormalPlanContextCapsule,
    FormalPlanContextError,
    FormalPlanContextLimits,
    FormalPlanContextMeasurement,
    FormalPlanContextQuery,
    FormalPlanContextTarget,
    FormalPlanContextUsage,
    FormalPlanResponseError,
    ImplementationOutcome,
    ImplementationOutcomeStatus,
    build_formal_plan_context_capsule,
    invoke_formal_plan_model,
    measure_formal_plan_context,
    query_formal_plan_graph,
    validate_formal_plan_model_response,
)
from ipfs_accelerate_py.agent_supervisor.formal_plan_validator import (
    PlanValidationStatus,
    validate_formal_plan,
)
from ipfs_accelerate_py.agent_supervisor.proof_context import (
    estimate_context_tokens,
)


TASK_CID = "task:cid:278"
DEPENDENCY_CID = "task:cid:277"
SYMBOL_CID = "symbol:formal-plan-context"
UNRELATED_SYMBOL_CID = "symbol:repository-wide"


def _source() -> dict[str, object]:
    return {
        "repository_tree_id": "tree:formal-context",
        "objectives": [
            {
                "goal_id": "G12.S1",
                "goal_cid": "goal:cid:g12-s1",
                "owner_actor_id": "supervisor",
                "acceptance_criteria": ["The formal plan is satisfied."],
            }
        ],
        "tasks": [
            {
                "task_id": "REF-277",
                "task_cid": DEPENDENCY_CID,
                "goal_id": "G12.S1",
                "actor_id": "codex:validator",
                "acceptance_criteria": ["Plan validation passes."],
                "validation_commands": ["pytest test_formal_plan_validator.py"],
                "changed_ast_scopes": ["symbol:formal-plan-validator"],
            },
            {
                "task_id": "REF-278",
                "task_cid": TASK_CID,
                "goal_id": "G12.S1",
                "actor_id": "codex:implementation",
                "depends_on": [DEPENDENCY_CID],
                "acceptance_criteria": [
                    "Capsules bind the selected task transition.",
                    "Capsules are bounded before model invocation.",
                ],
                "validation_commands": ["pytest test_formal_plan_context.py"],
                "changed_ast_scopes": [SYMBOL_CID],
            },
        ],
        "ast": [
            {
                "symbol_cid": SYMBOL_CID,
                "tree_cid": "tree:formal-context",
                "task_cid": TASK_CID,
                "qualified_name": (
                    "agent_supervisor.formal_plan_context."
                    "FormalPlanContextCapsule"
                ),
                "path": "agent_supervisor/formal_plan_context.py",
            },
            {
                "symbol_cid": "symbol:formal-plan-validator",
                "tree_cid": "tree:formal-context",
                "task_cid": DEPENDENCY_CID,
                "qualified_name": "agent_supervisor.formal_plan_validator.validate",
                "path": "agent_supervisor/formal_plan_validator.py",
            },
            {
                "symbol_cid": UNRELATED_SYMBOL_CID,
                "tree_cid": "tree:formal-context",
                "task_cid": "task:unrelated",
                "qualified_name": "unrelated.RepositoryWideAST",
                "path": "unrelated/repository.py",
                "full_ast": "must not enter a capsule",
            },
        ],
        "policies": [
            {
                "policy_cid": "policy:proof-carrying-plan",
                "minimum_code_assurance": "candidate",
                "fallback_check_ids": ["review:formal-plan"],
            }
        ],
        "evidence": [
            {
                "evidence_id": "evidence:validator",
                "task_cid": DEPENDENCY_CID,
                "kind": "test",
                "assurance": "candidate",
            }
        ],
    }


def _artifacts():
    compilation = compile_formal_plan(_source())
    assert compilation.status is CompilationStatus.COMPILED
    assert compilation.plan is not None
    validation = validate_formal_plan(compilation.plan, compilation.formulas)
    assert validation.status is PlanValidationStatus.CONSISTENT
    return compilation, validation


def _capsule(**kwargs):
    compilation, validation = _artifacts()
    arguments = {
        "task_id": TASK_CID,
        "target": FormalPlanContextTarget.CODEX,
        "ast_records": _source()["ast"],
        "source_excerpts": {
            SYMBOL_CID: "class FormalPlanContextCapsule:\n    pass\n" * 20,
            UNRELATED_SYMBOL_CID: "UNRELATED SOURCE",
        },
        "allowed_paths": ["test/api/test_agent_supervisor_formal_plan_context.py"],
    }
    arguments.update(kwargs)
    return build_formal_plan_context_capsule(
        compilation, validation, **arguments
    )


def _valid_response(capsule: FormalPlanContextCapsule) -> dict[str, object]:
    return {
        "schema": FORMAL_PLAN_MODEL_RESPONSE_SCHEMA,
        "bindings": capsule.bindings.to_dict(),
        "proposal": {
            "steps": [{"instruction": "Implement the selected transition."}],
            "changed_paths": ["agent_supervisor/formal_plan_context.py"],
            "tests": list(capsule.tests),
            "unresolved_obligations": [
                item["obligation_id"] for item in capsule.unresolved_obligations
            ],
            "proof_draft": "by exact capsule bindings",
            "notes": "proposal only",
        },
    }


def test_capsule_carries_complete_verified_task_slice() -> None:
    capsule = _capsule()

    assert capsule.plan_cid
    assert capsule.task_cid == TASK_CID
    assert capsule.validation_cid
    assert capsule.target is FormalPlanContextTarget.CODEX
    assert capsule.transition.task_cid == TASK_CID
    assert capsule.transition.event_kind == "completed"
    assert capsule.transition.dependency_task_ids == (DEPENDENCY_CID,)
    assert capsule.transition.from_state == "pending"
    assert capsule.transition.to_state == "completed"
    assert capsule.assumptions
    assert capsule.required_preconditions
    assert capsule.required_effects
    assert {item["symbol_id"] for item in capsule.relevant_ast_symbols} == {
        SYMBOL_CID
    }
    assert capsule.trusted_evidence
    assert capsule.trusted_evidence[0]["kind"] == "formal_plan_validation"
    assert capsule.trusted_evidence[0]["authoritative_for"] == (
        "plan_consistency_only"
    )
    assert capsule.counterexamples == ()
    assert set(capsule.allowed_paths) == {
        "agent_supervisor/formal_plan_context.py",
        "test/api/test_agent_supervisor_formal_plan_context.py",
    }
    assert "pytest test_formal_plan_context.py" in capsule.tests
    assert "review:formal-plan" in capsule.tests
    assert capsule.unresolved_obligations
    assert capsule.theorem["kind"] == "fixed_task_transition_theorem"
    assert capsule.theorem["formula_ids"]
    assert capsule.acceptance_policy["requirements"]
    assert capsule.bindings.plan_cid == capsule.plan_cid
    assert capsule.bindings.task_cid == TASK_CID
    assert capsule.bindings.theorem_cid == capsule.theorem_cid
    assert (
        capsule.bindings.acceptance_policy_cid
        == capsule.acceptance_policy_cid
    )
    assert (
        capsule.bindings.authoritative_evidence_cid
        == capsule.authoritative_evidence_cid
    )

    prompt = capsule.to_prompt()
    assert "UNRELATED SOURCE" not in prompt
    assert UNRELATED_SYMBOL_CID not in prompt
    assert "must not enter a capsule" not in prompt
    assert len(prompt.encode("utf-8")) == capsule.usage.bytes
    assert estimate_context_tokens(prompt) == capsule.usage.tokens
    assert FormalPlanContextCapsule.from_json(prompt) == capsule


def test_graph_and_source_limits_are_applied_before_dispatch() -> None:
    compilation, validation = _artifacts()
    limits = FormalPlanContextLimits(
        max_rows=7,
        max_graph_hops=1,
        max_bytes=32_000,
        max_tokens=8_000,
        max_source_excerpts=1,
        max_source_excerpt_bytes=48,
        max_source_bytes=48,
        max_ast_symbols=1,
        max_trusted_evidence=2,
        max_counterexamples=1,
        max_unresolved_obligations=8,
        max_allowed_paths=2,
        max_tests=4,
    )
    capsule = build_formal_plan_context_capsule(
        compilation,
        validation,
        FormalPlanContextQuery(TASK_CID, ast_symbol_ids=(SYMBOL_CID,)),
        target=FormalPlanContextTarget.LEANSTRAL,
        limits=limits,
        ast_records=_source()["ast"],
        source_excerpts={SYMBOL_CID: "x" * 10_000},
    )

    assert capsule.target is FormalPlanContextTarget.LEANSTRAL
    assert capsule.usage.rows <= 7
    assert capsule.usage.graph_hops <= 1
    assert capsule.graph_slice.row_count == capsule.usage.rows
    assert capsule.usage.source_excerpts == 1
    assert capsule.usage.source_bytes <= 48
    assert capsule.source_excerpts[0].byte_count <= 48
    assert capsule.source_excerpts[0].truncated
    assert capsule.usage.bytes <= limits.max_bytes
    assert capsule.usage.tokens <= limits.max_tokens

    graph = query_formal_plan_graph(
        compilation.graph_projection,
        TASK_CID,
        limits=limits,
    )
    assert graph.row_count <= limits.max_rows
    assert graph.max_hops <= limits.max_graph_hops
    assert any(node["record_id"] == TASK_CID for node in graph.nodes)
    assert all(node["record_id"] != UNRELATED_SYMBOL_CID for node in graph.nodes)

    invoked: list[str] = []

    def generate(prompt: str) -> str:
        invoked.append(prompt)
        return json.dumps(_valid_response(capsule))

    response = invoke_formal_plan_model(capsule, generate)
    assert len(invoked) == 1
    assert response.bindings == capsule.bindings

    stale = replace(
        capsule,
        usage=replace(capsule.usage, bytes=capsule.usage.bytes - 1),
    )
    invoked.clear()
    with pytest.raises(FormalPlanContextBudgetError, match="stale"):
        invoke_formal_plan_model(stale, generate)
    assert invoked == []


def test_model_response_must_bind_immutable_authority_and_scope() -> None:
    capsule = _capsule()
    response = validate_formal_plan_model_response(
        capsule, _valid_response(capsule)
    )
    assert response.changed_paths == (
        "agent_supervisor/formal_plan_context.py",
    )

    wrong_plan = _valid_response(capsule)
    wrong_plan["bindings"] = {
        **capsule.bindings.to_dict(),
        "plan_cid": "model:replacement-plan",
    }
    with pytest.raises(FormalPlanResponseError, match="plan_cid"):
        validate_formal_plan_model_response(capsule, wrong_plan)

    changed_theorem = _valid_response(capsule)
    changed_theorem["proposal"]["theorem"] = {"claim": "easier theorem"}
    with pytest.raises(FormalPlanResponseError, match="protected field"):
        validate_formal_plan_model_response(capsule, changed_theorem)

    changed_policy = _valid_response(capsule)
    changed_policy["proposal"]["steps"][0]["acceptance_policy"] = {
        "requirements": []
    }
    with pytest.raises(FormalPlanResponseError, match="protected field"):
        validate_formal_plan_model_response(capsule, changed_policy)

    changed_evidence = _valid_response(capsule)
    changed_evidence["proposal"]["trusted_evidence"] = []
    with pytest.raises(FormalPlanResponseError, match="protected field"):
        validate_formal_plan_model_response(capsule, changed_evidence)

    extra_binding = _valid_response(capsule)
    extra_binding["bindings"]["theorem"] = {"claim": "replacement"}
    with pytest.raises(FormalPlanResponseError, match="not exact"):
        validate_formal_plan_model_response(capsule, extra_binding)

    outside = _valid_response(capsule)
    outside["proposal"]["changed_paths"] = ["unrelated/repository.py"]
    with pytest.raises(FormalPlanResponseError, match="outside"):
        validate_formal_plan_model_response(capsule, outside)


def test_context_measurement_compares_size_and_implementation_outcomes() -> None:
    capsule = _capsule()
    baseline = "repository-wide planning prompt\n" + ("unbounded detail\n" * 8_000)
    capsule_outcome = ImplementationOutcome(
        status=ImplementationOutcomeStatus.ACCEPTED,
        accepted=True,
        tests_passed=3,
        changed_paths=("agent_supervisor/formal_plan_context.py",),
        obligations_resolved=2,
    )
    baseline_outcome = ImplementationOutcome(
        status=ImplementationOutcomeStatus.REJECTED,
        accepted=False,
        tests_passed=1,
        tests_failed=2,
        obligations_remaining=2,
    )

    measurement = measure_formal_plan_context(
        capsule,
        baseline,
        capsule_outcome=capsule_outcome,
        unbounded_prompt_outcome=baseline_outcome,
    )

    assert measurement.capsule_cid == capsule.capsule_cid
    assert measurement.capsule_bytes == capsule.usage.bytes
    assert measurement.unbounded_prompt_bytes == len(baseline.encode("utf-8"))
    assert measurement.bytes_saved > 0
    assert measurement.tokens_saved > 0
    assert measurement.byte_ratio_millionths < 1_000_000
    assert measurement.token_ratio_millionths < 1_000_000
    assert measurement.outcome_comparison == {
        "capsule_accepted": True,
        "unbounded_prompt_accepted": False,
        "accepted_delta": 1,
        "tests_passed_delta": 2,
        "tests_failed_delta": -2,
        "obligations_resolved_delta": 2,
        "obligations_remaining_delta": -2,
    }
    assert measurement.measurement_cid
    assert json.loads(measurement.to_json())["outcome_comparison"][
        "accepted_delta"
    ] == 1
    assert FormalPlanContextMeasurement.from_json(measurement.to_json()) == measurement


def test_countermodel_is_carried_as_bounded_repair_evidence() -> None:
    compilation, _ = _artifacts()
    assert compilation.plan is not None
    events = tuple(
        replace(item, logical_time=6)
        if item.event_id == f"{DEPENDENCY_CID}:event:completed"
        else item
        for item in compilation.plan.events
    )
    modified_plan = replace(compilation.plan, events=events)
    modified_compilation = replace(compilation, plan=modified_plan)
    validation = validate_formal_plan(
        modified_plan, modified_compilation.formulas
    )
    assert validation.countermodel is not None

    capsule = build_formal_plan_context_capsule(
        modified_compilation,
        validation,
        task_id=TASK_CID,
        ast_records=_source()["ast"],
    )

    assert capsule.counterexamples
    assert any(
        item["kind"] == "bounded_countermodel"
        for item in capsule.counterexamples
    )
    assert any(
        item["kind"] == "validation_finding"
        for item in capsule.unresolved_obligations
    )


def test_fail_closed_for_mismatched_validation_exact_selectors_and_tiny_budget() -> None:
    compilation, validation = _artifacts()

    with pytest.raises(FormalPlanContextError, match="exact"):
        FormalPlanContextQuery("*")

    with pytest.raises(FormalPlanContextError, match="absent"):
        build_formal_plan_context_capsule(
            compilation, validation, task_id="task:missing"
        )

    with pytest.raises(FormalPlanContextError, match="does not bind"):
        build_formal_plan_context_capsule(
            compilation,
            replace(validation, plan_id="plan:other"),
            task_id=TASK_CID,
        )

    with pytest.raises(FormalPlanContextBudgetError, match="mandatory"):
        build_formal_plan_context_capsule(
            compilation,
            validation,
            task_id=TASK_CID,
            ast_records=_source()["ast"],
            limits=FormalPlanContextLimits(
                max_rows=1,
                max_graph_hops=0,
                max_bytes=512,
                max_tokens=128,
                max_source_excerpts=0,
                max_source_excerpt_bytes=1,
                max_source_bytes=1,
                max_ast_symbols=1,
                max_trusted_evidence=1,
                max_counterexamples=0,
                max_unresolved_obligations=4,
                max_allowed_paths=1,
                max_tests=4,
            ),
        )

    capsule = _capsule()
    payload = capsule.to_dict()
    payload["theorem"]["claim"] = "tampered"
    with pytest.raises(FormalPlanContextError, match="identity"):
        FormalPlanContextCapsule.from_dict(payload)

    with pytest.raises(FormalPlanContextBudgetError, match="stale"):
        replace(
            capsule,
            usage=FormalPlanContextUsage(
                **{**capsule.usage.to_dict(), "tokens": capsule.usage.tokens + 1}
            ),
        ).validate_limits()
