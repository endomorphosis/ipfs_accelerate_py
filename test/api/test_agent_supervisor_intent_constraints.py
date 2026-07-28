from __future__ import annotations

from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.intent_constraint_adapter import (
    IntentAdapterBounds,
    IntentCompilationStatus,
    IntentConformanceRequest,
    IntentConformanceResult,
    IntentConformanceVerdict,
    IntentConstraintCompilationResult,
    IntentConstraintSet,
    IntentControlFlowKind,
    IntentFindingCode,
    compile_intent_constraints,
    create_intent_conformance_request,
    evaluate_intent_conformance,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_registry import (
    IRFamily,
    IRLoadRequest,
    IRLoadStatus,
    IRRegistry,
    deterministic_ir_fixture,
)


def _verified(family: IRFamily, **kwargs: object):
    reference, encoded = deterministic_ir_fixture(family, **kwargs)
    registry = IRRegistry()
    registry.register_local_artifact(reference, encoded)
    result = registry.load(IRLoadRequest(reference=reference, family=family))
    assert result.status is IRLoadStatus.VERIFIED
    return result.require_artifact()


def _compilation():
    intent = _verified(
        IRFamily.INTENT,
        declarations=[
            {"id": "goal:ship", "kind": "goal", "grounded": True},
            {
                "id": "action:fetch",
                "kind": "action",
                "goal_id": "goal:ship",
                "grounded": True,
            },
            {
                "id": "action:validate",
                "kind": "action",
                "goal_id": "goal:ship",
                "depends_on": ["action:fetch"],
                "grounded": True,
            },
            {"id": "action:index", "kind": "action", "grounded": True},
            {"id": "action:audit", "kind": "action", "grounded": True},
            {
                "id": "action:publish",
                "kind": "action",
                "depends_on": ["action:index", "action:audit"],
                "grounded": True,
            },
            {
                "id": "flow:fetch-validate",
                "kind": "sequence",
                "sequence": ["action:fetch", "action:validate"],
                "grounded": True,
            },
            {
                "id": "flow:parallel-join",
                "kind": "parallel",
                "member_action_ids": ["action:index", "action:audit"],
                "join_action_id": "action:publish",
                "grounded": True,
            },
            {
                "id": "precondition:fetch",
                "kind": "precondition",
                "action_id": "action:fetch",
                "statement_id": "statement:source-available",
                "grounded": True,
            },
            {
                "id": "guard:validate",
                "kind": "guard",
                "action_id": "action:validate",
                "formula_id": "formula:valid-input",
                "grounded": True,
            },
            {
                "id": "invariant:validate",
                "kind": "invariant",
                "action_id": "action:validate",
                "statement_id": "statement:no-data-loss",
                "grounded": True,
            },
            {
                "id": "effect:fetch",
                "kind": "effect",
                "action_id": "action:fetch",
                "operation": "assign",
                "fluent_id": "fluent:source",
                "value": "loaded",
                "grounded": True,
            },
            {
                "id": "postcondition:validate",
                "kind": "postcondition",
                "action_id": "action:validate",
                "statement_id": "statement:validated",
                "grounded": True,
            },
            {
                "id": "assumption:fetch",
                "kind": "assumption",
                "action_id": "action:fetch",
                "statement_id": "statement:network-available",
                "grounded": True,
            },
            {
                "id": "failure:fetch",
                "kind": "failure",
                "action_id": "action:fetch",
                "failure_id": "failure:source-unavailable",
                "grounded": True,
            },
            {
                "id": "retry:fetch",
                "kind": "retry",
                "action_id": "action:fetch",
                "retry_id": "retry:bounded",
                "grounded": True,
            },
            {
                "id": "verification:publish",
                "kind": "verification",
                "action_id": "action:publish",
                "verification_id": "verification:artifact",
                "grounded": True,
            },
        ],
    )
    formalization = _verified(
        IRFamily.FORMALIZATION,
        formal_views=[
            {
                "id": "formula:valid-input",
                "kind": "first_order",
                "grounded": True,
            }
        ],
        obligations=[
            {
                "id": "obligation:valid-input",
                "kind": "proof",
                "subject_id": "guard:validate",
                "evidence_ids": ["evidence:guard-proof"],
                "grounded": True,
            }
        ],
    )
    result = compile_intent_constraints(intent, formalization)
    assert result.status is IntentCompilationStatus.COMPILED
    return result


def _candidate(compilation=None):
    result = compilation or _compilation()
    constraints = result.require_constraint_set()
    return {
        "plan_id": "candidate:ship@1",
        "intent_root": dict(constraints.intent_root),
        "formalization_root": dict(constraints.formalization_root),
        "goal_ids": ["goal:ship"],
        "actions": [
            {
                "action_id": "action:fetch",
                "precondition_ids": ["statement:source-available"],
                "assumption_ids": ["statement:network-available"],
                "failure_ids": ["failure:source-unavailable"],
                "retry_ids": ["retry:bounded"],
                "effects": [
                    {
                        "operation": "assign",
                        "fluent_id": "fluent:source",
                        "value": "loaded",
                    }
                ],
            },
            {
                "action_id": "action:validate",
                "depends_on": ["action:fetch"],
                "guard_ids": ["formula:valid-input"],
                "invariant_ids": ["statement:no-data-loss"],
                "postcondition_ids": ["statement:validated"],
            },
            {"action_id": "action:index"},
            {"action_id": "action:audit"},
            {
                "action_id": "action:publish",
                "depends_on": ["action:index", "action:audit"],
                "verification_ids": ["verification:artifact"],
            },
        ],
    }


def _request(compilation=None, candidate=None, **updates):
    result = compilation or _compilation()
    constraint_set = result.require_constraint_set()
    values = {
        "discharged_obligation_ids": tuple(
            item.obligation_id for item in constraint_set.proof_obligations
        )
    }
    values.update(updates)
    return create_intent_conformance_request(
        result, candidate or _candidate(result), **values
    )


def test_compiles_every_action_contract_kind_with_source_and_proof_bindings():
    result = _compilation()
    constraint_set = result.require_constraint_set()

    assert {item.kind.value for item in constraint_set.constraints} == {
        "goal",
        "action",
        "control_flow",
        "precondition",
        "guard",
        "invariant",
        "effect",
        "postcondition",
        "assumption",
        "failure",
        "retry",
        "verification",
    }
    assert {
        item.flow_kind for item in constraint_set.control_edges
    } >= {
        IntentControlFlowKind.SEQUENCE,
        IntentControlFlowKind.JOIN,
    }
    assert constraint_set.proof_obligations
    guard = next(
        item for item in constraint_set.constraints if item.node_id == "guard:validate"
    )
    assert len(guard.source_binding_ids) == 2
    assert all(item.source_binding_ids for item in constraint_set.constraints)
    assert all(
        not item.grants_execution_authority
        for item in constraint_set.source_bindings
    )
    assert not constraint_set.grants_execution_authority
    assert not constraint_set.graph_truncated


def test_exact_candidate_conforms_but_result_never_authorizes_execution():
    request = _request()
    result = evaluate_intent_conformance(request)

    assert result.verdict is IntentConformanceVerdict.CONFORMANT
    assert result.conformant
    assert not result.findings
    assert not result.authorizes_execution
    assert set(result.checked_constraint_ids) == {
        item.constraint_id for item in request.constraint_set.constraints
    }


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda candidate: candidate["actions"].pop(0),
            IntentFindingCode.MISSING_REQUIRED_ACTION,
        ),
        (
            lambda candidate: candidate["actions"][1].pop("guard_ids"),
            IntentFindingCode.UNSATISFIED_GUARD,
        ),
        (
            lambda candidate: candidate["actions"][1].pop("invariant_ids"),
            IntentFindingCode.UNSATISFIED_INVARIANT,
        ),
        (
            lambda candidate: candidate["actions"][-1].update(depends_on=[]),
            IntentFindingCode.PARALLEL_JOIN_VIOLATION,
        ),
        (
            lambda candidate: candidate["actions"][0]["effects"].append(
                {
                    "operation": "assign",
                    "fluent_id": "fluent:undeclared",
                    "value": "changed",
                }
            ),
            IntentFindingCode.UNDECLARED_EFFECT,
        ),
        (
            lambda candidate: candidate.update(graph_complete=False),
            IntentFindingCode.GRAPH_TRUNCATED,
        ),
        (
            lambda candidate: candidate.update(
                authorization_source="intent_ir"
            ),
            IntentFindingCode.INTENT_USED_AS_AUTHORIZATION,
        ),
        (
            lambda candidate: candidate.update(
                authorization_source="graphrag-retrieval"
            ),
            IntentFindingCode.RETRIEVAL_USED_AS_AUTHORIZATION,
        ),
    ],
)
def test_conformance_fails_closed_for_plan_omissions_and_authority_confusion(
    mutation, code
):
    compilation = _compilation()
    candidate = _candidate(compilation)
    mutation(candidate)

    result = evaluate_intent_conformance(
        _request(compilation, candidate)
    )

    assert not result.conformant
    assert code in {item.code for item in result.findings}


def test_changed_intent_root_invalidates_an_otherwise_exact_candidate():
    request = _request()
    changed = dict(request.intent_root)
    changed["supervisor_digest"] = "sha256:" + "0" * 64

    result = evaluate_intent_conformance(
        IntentConformanceRequest(
            constraint_set=request.constraint_set,
            candidate_plan=request.candidate_plan,
            intent_root=changed,
            formalization_root=request.formalization_root,
            discharged_obligation_ids=request.discharged_obligation_ids,
        )
    )

    assert result.verdict is IntentConformanceVerdict.INVALID
    assert IntentFindingCode.ROOT_CHANGED in {
        item.code for item in result.findings
    }


def test_inferred_requirements_need_explicit_binding_and_proof_discharge():
    intent = _verified(
        IRFamily.INTENT,
        declarations=[
            {
                "id": "goal:inferred",
                "kind": "goal",
                "grounded": True,
            },
            {
                "id": "action:inferred",
                "kind": "action",
                "goal_id": "goal:inferred",
                "origin": "inferred",
            }
        ],
    )
    formalization = _verified(IRFamily.FORMALIZATION)
    compilation = compile_intent_constraints(intent, formalization)
    constraint_set = compilation.require_constraint_set()
    candidate = {
        "goal_ids": ["goal:inferred"],
        "actions": [{"action_id": "action:inferred"}],
        "intent_root": dict(constraint_set.intent_root),
        "formalization_root": dict(constraint_set.formalization_root),
    }

    rejected = evaluate_intent_conformance(
        create_intent_conformance_request(compilation, candidate)
    )
    assert {
        IntentFindingCode.UNBOUND_INFERRED_REQUIREMENT,
        IntentFindingCode.PROOF_OBLIGATION_UNDISCHARGED,
    } <= {item.code for item in rejected.findings}

    constraint = next(
        item for item in constraint_set.constraints if item.node_id == "action:inferred"
    )
    accepted = evaluate_intent_conformance(
        create_intent_conformance_request(
            compilation,
            candidate,
            inferred_requirement_bindings={
                constraint.constraint_id: "review-binding:1"
            },
        )
    )
    assert accepted.verdict is IntentConformanceVerdict.CONFORMANT


def test_unknown_statements_and_contradictory_effects_remain_visible_and_fail():
    intent = _verified(
        IRFamily.INTENT,
        declarations=[
            {"id": "goal:a", "kind": "goal", "grounded": True},
            {"id": "action:a", "kind": "action", "grounded": True},
            {"id": "statement:unknown", "kind": "model-invented", "grounded": True},
            {
                "id": "effect:one",
                "kind": "effect",
                "action_id": "action:a",
                "fluent_id": "fluent:x",
                "operation": "assign",
                "value": "one",
                "grounded": True,
            },
            {
                "id": "effect:two",
                "kind": "effect",
                "action_id": "action:a",
                "fluent_id": "fluent:x",
                "operation": "assign",
                "value": "two",
                "grounded": True,
            },
        ],
    )
    compilation = compile_intent_constraints(
        intent, _verified(IRFamily.FORMALIZATION)
    )
    constraint_set = compilation.require_constraint_set()

    assert compilation.status is IntentCompilationStatus.UNSUPPORTED
    assert "statement:unknown" in constraint_set.unsupported_node_ids
    assert constraint_set.contradictory_effect_groups
    result = evaluate_intent_conformance(
        create_intent_conformance_request(
            compilation,
            {
                "actions": [{"action_id": "action:a", "effects": []}],
                "intent_root": dict(constraint_set.intent_root),
                "formalization_root": dict(constraint_set.formalization_root),
            },
        )
    )
    assert {
        IntentFindingCode.UNSUPPORTED_STATEMENT,
        IntentFindingCode.CONTRADICTORY_EFFECT,
    } <= {item.code for item in result.findings}


def test_inexact_effect_and_control_flow_declarations_fail_closed():
    intent = _verified(
        IRFamily.INTENT,
        declarations=[
            {"id": "goal:a", "kind": "goal", "grounded": True},
            {"id": "action:a", "kind": "action", "grounded": True},
            {
                "id": "flow:empty",
                "kind": "sequence",
                "sequence": [],
                "grounded": True,
            },
            {
                "id": "effect:inexact",
                "kind": "effect",
                "action_id": "action:a",
                "value": "changed",
                "grounded": True,
            },
        ],
    )

    compilation = compile_intent_constraints(
        intent, _verified(IRFamily.FORMALIZATION)
    )
    constraint_set = compilation.require_constraint_set()

    assert compilation.status is IntentCompilationStatus.UNSUPPORTED
    assert set(constraint_set.unsupported_node_ids) == {
        "effect:inexact",
        "flow:empty",
    }
    assert sum(
        item.code is IntentFindingCode.UNSUPPORTED_STATEMENT
        for item in compilation.findings
    ) == 2


@pytest.mark.parametrize(
    "authorization_claim",
    [
        {"intent_authorized": True},
        {"authority": "intent_ir"},
        {"retrieval_permission": True},
    ],
)
def test_context_only_authority_claims_fail_regardless_of_field_shape(
    authorization_claim,
):
    compilation = _compilation()
    candidate = _candidate(compilation)
    candidate["metadata"] = authorization_claim

    result = evaluate_intent_conformance(_request(compilation, candidate))

    assert not result.conformant
    assert {
        IntentFindingCode.INTENT_USED_AS_AUTHORIZATION,
        IntentFindingCode.RETRIEVAL_USED_AS_AUTHORIZATION,
    } & {item.code for item in result.findings}


def test_explicitly_denied_context_authority_is_not_misread_as_a_grant():
    compilation = _compilation()
    candidate = _candidate(compilation)
    candidate["metadata"] = {
        "intent_authorized": False,
        "retrieval_permission": "denied",
    }

    result = evaluate_intent_conformance(_request(compilation, candidate))

    assert result.verdict is IntentConformanceVerdict.CONFORMANT


def test_request_bounds_and_support_claims_cannot_launder_closed_inputs():
    intent = _verified(
        IRFamily.INTENT,
        declarations=[
            {"id": "goal:a", "kind": "goal", "grounded": True},
            {"id": "action:a", "kind": "action", "grounded": True},
            {
                "id": "statement:unknown",
                "kind": "model-invented",
                "grounded": True,
            },
        ],
    )
    compilation = compile_intent_constraints(
        intent, _verified(IRFamily.FORMALIZATION)
    )
    constraint_set = compilation.require_constraint_set()
    candidate = {
        "goal_ids": ["goal:a"],
        "actions": [{"action_id": "action:a"}],
        "intent_root": dict(constraint_set.intent_root),
        "formalization_root": dict(constraint_set.formalization_root),
    }
    request = create_intent_conformance_request(
        compilation,
        candidate,
        supported_statement_ids=("statement:unknown",),
    )

    result = evaluate_intent_conformance(
        request,
        bounds=IntentAdapterBounds(max_canonical_bytes=1),
    )

    assert result.verdict is IntentConformanceVerdict.INVALID
    assert {
        IntentFindingCode.INVALID_INPUT,
        IntentFindingCode.UNSUPPORTED_STATEMENT,
    } <= {item.code for item in result.findings}


def test_canonical_compilation_request_and_result_round_trip_and_detect_tampering():
    compilation = _compilation()
    request = _request(compilation)
    result = evaluate_intent_conformance(request)

    rebuilt_set = IntentConstraintSet.from_dict(
        compilation.require_constraint_set().to_dict()
    )
    rebuilt_compilation = IntentConstraintCompilationResult.from_dict(
        compilation.to_dict()
    )
    rebuilt_request = IntentConformanceRequest.from_dict(request.to_dict())
    rebuilt_result = IntentConformanceResult.from_dict(result.to_dict())

    assert rebuilt_set.constraint_set_id == request.constraint_set.constraint_set_id
    assert rebuilt_compilation.to_dict() == compilation.to_dict()
    assert rebuilt_request.request_id == request.request_id
    assert rebuilt_result.result_id == result.result_id
    assert rebuilt_request.canonical_bytes == request.canonical_bytes
    assert rebuilt_result.canonical_bytes == result.canonical_bytes

    tampered = deepcopy(request.to_dict())
    tampered["candidate_plan"]["actions"].pop()
    with pytest.raises(ValueError, match="request_id identity mismatch"):
        IntentConformanceRequest.from_dict(tampered)
