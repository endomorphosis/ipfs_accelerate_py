"""Independent contract tests for SPAR-035 context adapter and residual routing."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.context_adapter import (
    ADAPTER_IS_NOMINATION_ONLY,
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    CONTEXT_CAN_AUTHORIZE_COMPLETION,
    CONTEXT_CAN_AUTHORIZE_TRANSITION,
    CONTEXT_CAN_CREATE_AUTHORITY,
    CONTEXT_CAN_REPLACE_COMPILER,
    CONTEXT_CAN_WEAKEN_VALIDATION,
    CONTEXT_COMPILER_AUTHORITY,
    CONTEXT_COMPILER_REMAINS_AUTHORITY,
    CONTEXT_CONTRACT_VERSION,
    DECLARED_ALLOWED_EFFECTS,
    DECLARED_QUESTION_KINDS,
    DECLARED_ROUTE_KINDS,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    FORBIDDEN_CONTEXT_NAMES,
    GOAL_ID,
    IDENTICAL_FAILURE_RETRY_WITHOUT_EVIDENCE,
    IDENTITY_EXCLUDED_FIELDS,
    INDEPENDENT_VALIDATION_REQUIRED,
    MARKDOWN_IS_NOT_COMPLETION,
    MAX_GENERAL_MODEL_QUESTIONS,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    MODEL_ROUTE_CLASS,
    NAMED_UNRESOLVED_QUESTION_INTERFACE,
    NETWORK_DENIED,
    NETWORK_DENY,
    NO_MODEL_STEPS,
    ONE_RESIDUAL_GENERAL_MODEL_QUESTION,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    RESIDUAL_MODEL_ROUTE_INTERFACE,
    RESIDUAL_ROUTE_RECEIPT_INTERFACE,
    SEMANTIC_REFACTOR_CONTEXT_ADAPTER_INTERFACE,
    SEMANTIC_REFACTOR_CONTEXT_INTERFACE,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    AffectedSlice,
    ContextAdapterError,
    NamedUnresolvedQuestion,
    QuestionKind,
    ResidualModelRoute,
    ResidualRouteReceipt,
    RouteKind,
    RouteStatus,
    SemanticRefactorContext,
    SemanticRefactorContextAdapter,
    adapt_semantic_refactor_context,
    assert_not_competing_capsule_family,
    compile_affected_slice,
    compile_context_receipt,
    compile_named_question,
    context_adapter_cid_profile,
    context_adapter_descriptor,
    decode_canonical_context,
    decode_canonical_receipt,
    decode_canonical_route,
    dry_run_context_route,
    encode_canonical_context,
    encode_canonical_receipt,
    encode_canonical_route,
    provider_free_exports,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "context_adapter.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/context_adapter.py",
    "test/api/semantic_refactoring/test_context_adapter.py",
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
SOURCE_CID_LABEL = "source"


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _packet(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "packet_cid": _cid("packet"),
        "write_paths": list(WRITE_PATHS),
        "validation_commands": list(VALIDATION),
        "raw_source_cids": [_cid(SOURCE_CID_LABEL)],
    }
    fields.update(overrides)
    return fields


def _reuse(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "decision": "revoke",
        "exact_match": False,
        "query_key_cid": _cid("query-key"),
        "matched_transition_cid": "",
        "reasons": ["no_exact_match"],
    }
    fields.update(overrides)
    return fields


def _slice(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "slice_id": "slice:mod",
        "path": "pkg/mod.py",
        "symbol": "pkg.mod.fn",
        "source_cid": _cid(SOURCE_CID_LABEL),
    }
    fields.update(overrides)
    return fields


def _question(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "question_id": "q:boundary",
        "kind": QuestionKind.BOUNDARY_CONTRACT.value,
        "statement": "Does the extracted boundary preserve input identity?",
        "slice_id": "slice:mod",
        "contract_ids": ["contract:boundary"],
        "evidence_cids": [_cid("question-evidence")],
        "typed": True,
        "unresolved": True,
    }
    fields.update(overrides)
    return fields


def _contract(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "contract_id": "contract:boundary",
        "contract_cid": _cid("contract"),
        "kind": "boundary",
    }
    fields.update(overrides)
    return fields


def _adapt(**overrides: Any) -> ResidualRouteReceipt:
    fields: dict[str, Any] = {
        "packet": _packet(),
        "reuse_decision": _reuse(),
        "slices": [_slice()],
        "questions": [_question()],
        "contracts": [_contract()],
        "counterexamples": [],
        "evidence": [],
        "analogous_refactors": [],
    }
    fields.update(overrides)
    return adapt_semantic_refactor_context(**fields)


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-035"
    assert GOAL_ID == "SPAR-G063"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert SEMANTIC_REFACTOR_CONTEXT_ADAPTER_INTERFACE == (
        "SemanticRefactorContextAdapter@1"
    )
    assert SEMANTIC_REFACTOR_CONTEXT_INTERFACE == "SemanticRefactorContext@1"
    assert NAMED_UNRESOLVED_QUESTION_INTERFACE == "NamedUnresolvedQuestion@1"
    assert RESIDUAL_MODEL_ROUTE_INTERFACE == "ResidualModelRoute@1"
    assert RESIDUAL_ROUTE_RECEIPT_INTERFACE == "ResidualRouteReceipt@1"
    assert CONTEXT_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("context_adapter@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "route/context decisions"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert CONTEXT_COMPILER_AUTHORITY.endswith("context_compiler")
    assert CONTEXT_CAN_AUTHORIZE_COMPLETION is False
    assert CONTEXT_CAN_AUTHORIZE_TRANSITION is False
    assert CONTEXT_CAN_CREATE_AUTHORITY is False
    assert CONTEXT_CAN_REPLACE_COMPILER is False
    assert CONTEXT_CAN_WEAKEN_VALIDATION is False
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
    assert NETWORK_DENIED is True
    assert NETWORK_DENY == "deny"
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert CONTEXT_COMPILER_REMAINS_AUTHORITY is True
    assert ONE_RESIDUAL_GENERAL_MODEL_QUESTION is True
    assert INDEPENDENT_VALIDATION_REQUIRED is True
    assert IDENTICAL_FAILURE_RETRY_WITHOUT_EVIDENCE is False
    assert MAX_GENERAL_MODEL_QUESTIONS == 1
    profile = context_adapter_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "SemanticRefactorContextAdapter" in names
    assert "ResidualRouteReceipt" in names
    assert "SemanticRefactorContext" in names
    assert "ResidualModelRoute" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "SemanticRefactorContextAdapter" in exports
    assert "adapt_semantic_refactor_context" in exports
    assert "compile_context_receipt" in exports
    assert "dry_run_context_route" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_module_does_not_implement_forbidden_context_shortcuts() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not (names & FORBIDDEN_CONTEXT_NAMES)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("context_compiler" in name for name in imported)
    descriptor = context_adapter_descriptor()
    assert descriptor["interface"] == SEMANTIC_REFACTOR_CONTEXT_ADAPTER_INTERFACE
    assert descriptor["context_compiler_remains_authority"] is True
    assert descriptor["nomination_only"] is True
    assert descriptor["one_residual_general_model_question"] is True
    assert descriptor["can_weaken_validation"] is False
    assert tuple(descriptor["no_model_steps"]) == NO_MODEL_STEPS
    assert tuple(descriptor["model_route_class"]) == MODEL_ROUTE_CLASS
    forbids = set(descriptor["forbids"])
    assert "replace_context_compiler" in forbids
    assert "weaken_validation" in forbids
    assert "retry_identical_failure" in forbids
    assert "route_multiple_residuals" in forbids
    assert "compile_full_task_dump" in forbids
    assert "suppress_raw_source" in forbids


def test_exact_reuse_closes_without_general_model() -> None:
    receipt = _adapt(
        reuse_decision=_reuse(
            decision="reuse",
            exact_match=True,
            matched_transition_cid=_cid("episode"),
            reasons=[],
        )
    )
    assert receipt.route_kind == RouteKind.EXACT_REUSE.value
    assert receipt.status == RouteStatus.REUSED.value
    assert receipt.general_model_invoked is False
    assert receipt.residual_question_id == ""
    assert receipt.route.no_model_step == RouteKind.EXACT_REUSE.value
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.context.adapter_is_nomination_only is True
    assert receipt.context.context_compiler_remains_authority is True
    assert receipt.mutated is False
    assert receipt.deterministic is True
    assert receipt.network == NETWORK_DENY
    assert DECLARED_ROUTE_KINDS >= {RouteKind.EXACT_REUSE.value}
    assert DECLARED_QUESTION_KINDS >= {QuestionKind.BOUNDARY_CONTRACT.value}


def test_verified_procedure_precedes_residual_model() -> None:
    receipt = _adapt(
        procedure={"available": True, "complete": True, "procedure_cid": _cid("proc")}
    )
    assert receipt.route_kind == RouteKind.VERIFIED_PROCEDURE.value
    assert receipt.general_model_invoked is False
    assert receipt.status == RouteStatus.NO_MODEL.value


def test_deterministic_analysis_and_proof_and_transform_are_no_model() -> None:
    analysis = _adapt(
        analysis={"can_close": True, "kind": "contract", "analysis_cid": _cid("an")}
    )
    assert analysis.route_kind == RouteKind.DETERMINISTIC_ANALYSIS.value
    proof = _adapt(proof={"can_close": True, "proof_cid": _cid("proof")})
    assert proof.route_kind == RouteKind.PROOF_SEARCH.value
    transform = _adapt(transform={"can_close": True, "transform_cid": _cid("tx")})
    assert transform.route_kind == RouteKind.DETERMINISTIC_TRANSFORM.value
    assert analysis.general_model_invoked is False
    assert proof.general_model_invoked is False
    assert transform.general_model_invoked is False


def test_one_named_residual_routes_to_general_model() -> None:
    receipt = _adapt()
    assert receipt.route_kind == RouteKind.RESIDUAL_GENERAL_MODEL.value
    assert receipt.status == RouteStatus.NOMINATED.value
    assert receipt.general_model_invoked is True
    assert receipt.residual_question_id == "q:boundary"
    assert receipt.route.model_output_is_proposal_only is True
    assert receipt.route.independent_validation_required is True
    assert receipt.typed_terminal is False
    context = receipt.context
    assert len(context.questions) == 1
    assert context.questions[0].typed is True
    assert context.questions[0].unresolved is True
    assert context.slices[0].path == "pkg/mod.py"
    assert set(context.allowed_effects) == DECLARED_ALLOWED_EFFECTS
    adapter = SemanticRefactorContextAdapter()
    again = adapter.adapt(
        packet=_packet(),
        reuse_decision=_reuse(),
        slices=[_slice()],
        questions=[_question()],
        contracts=[_contract()],
    )
    assert again.receipt_cid == receipt.receipt_cid


def test_multiple_residuals_are_typed_terminal_not_model_dump() -> None:
    receipt = _adapt(
        questions=[
            _question(),
            _question(
                question_id="q:state",
                kind=QuestionKind.STATE_OWNERSHIP.value,
                statement="Who owns the extracted state?",
                evidence_cids=[_cid("state-evidence")],
            ),
        ]
    )
    assert receipt.route_kind == RouteKind.BLOCKED.value
    assert receipt.status == RouteStatus.BLOCKED.value
    assert receipt.general_model_invoked is False
    assert receipt.typed_terminal is True
    assert "multiple_residuals" in receipt.route.reasons


def test_identical_failure_does_not_retry_without_new_evidence() -> None:
    question = compile_named_question(
        question_id="q:boundary",
        kind=QuestionKind.BOUNDARY_CONTRACT.value,
        statement="Does the extracted boundary preserve input identity?",
        slice_id="slice:mod",
        contract_ids=["contract:boundary"],
        evidence_cids=[_cid("question-evidence")],
    )
    receipt = _adapt(
        prior_failures=[
            {
                "question_id": "q:boundary",
                "evidence_fingerprint": question.evidence_fingerprint,
            }
        ]
    )
    assert receipt.route_kind == RouteKind.BLOCKED.value
    assert receipt.retry_blocked is True
    assert receipt.general_model_invoked is False
    assert "identical_failure_without_new_evidence" in receipt.route.reasons
    retried = _adapt(
        questions=[
            _question(evidence_cids=[_cid("new-evidence")]),
        ],
        prior_failures=[
            {
                "question_id": "q:boundary",
                "evidence_fingerprint": question.evidence_fingerprint,
            }
        ],
    )
    assert retried.route_kind == RouteKind.RESIDUAL_GENERAL_MODEL.value
    assert retried.retry_blocked is False


def test_missing_named_residual_is_typed_terminal() -> None:
    receipt = _adapt(questions=[])
    assert receipt.route_kind == RouteKind.BLOCKED.value
    assert receipt.typed_terminal is True
    assert receipt.general_model_invoked is False
    assert "missing_named_residual" in receipt.route.reasons


def test_specialist_ranking_is_nomination_only_and_cannot_close() -> None:
    receipt = _adapt(questions=[], ranking={"channel": "vector", "nomination_only": True})
    assert receipt.route_kind == RouteKind.SPECIALIST_RANKING.value
    assert receipt.general_model_invoked is False
    assert receipt.route.ranking_applied is True
    with pytest.raises(ContextAdapterError, match="nomination_only"):
        _adapt(ranking={"channel": "vector", "nomination_only": False})
    with pytest.raises(ContextAdapterError, match="cannot close"):
        _adapt(ranking={"channel": "vector", "can_close": True})
    with pytest.raises(ContextAdapterError, match="cannot suppress"):
        _adapt(ranking={"channel": "vector", "suppress_residual": True})


def test_no_model_steps_precede_ranking_and_residual() -> None:
    receipt = _adapt(
        analysis={"can_close": True, "kind": "ast", "analysis_cid": _cid("ast")},
        ranking={"channel": "hybrid", "nomination_only": True},
        questions=[_question()],
    )
    assert receipt.route_kind == RouteKind.DETERMINISTIC_ANALYSIS.value
    assert receipt.general_model_invoked is False
    assert receipt.route.ranking_applied is True


def test_vector_evidence_cannot_admit_or_weaken() -> None:
    with pytest.raises(ContextAdapterError, match="cannot admit"):
        _adapt(vector_evidence={"evidence_class": "vector_candidate", "admit_route": True})
    with pytest.raises(ContextAdapterError, match="raw-source"):
        _adapt(
            vector_evidence={
                "evidence_class": "model_hypothesis",
                "suppress_raw_source": True,
            }
        )
    with pytest.raises(ContextAdapterError, match="cannot weaken"):
        _adapt(
            vector_evidence={
                "evidence_class": "heuristic",
                "weaken_validation": True,
            }
        )
    with pytest.raises(ContextAdapterError, match="cannot weaken"):
        _adapt(skip_validation=True)
    with pytest.raises(ContextAdapterError, match="cannot weaken"):
        _adapt(weaken_validation=True)
    with pytest.raises(ContextAdapterError, match="cannot admit"):
        _adapt(
            analysis={
                "can_close": True,
                "kind": "graph",
                "evidence_class": "vector_candidate",
            }
        )


def test_analogous_refactors_are_context_only() -> None:
    receipt = _adapt(
        analogous_refactors=[
            {
                "analog_id": "analog:prior",
                "transition_cid": _cid("analog"),
                "channel": "vector",
                "evidence_class": "vector_candidate",
            }
        ]
    )
    analog = receipt.context.analogous_refactors[0]
    assert analog["context_only"] is True
    assert analog["channel"] == "vector"
    with pytest.raises(ContextAdapterError, match="cannot claim exact"):
        _adapt(
            analogous_refactors=[
                {
                    "analog_id": "analog:fake",
                    "transition_cid": _cid("analog"),
                    "channel": "exact",
                }
            ]
        )


def test_reuse_requires_exact_match() -> None:
    with pytest.raises(ContextAdapterError, match="exact key match"):
        _adapt(reuse_decision=_reuse(decision="reuse", exact_match=False))


def test_missing_predecessors_fail_closed() -> None:
    with pytest.raises(ContextAdapterError, match="SPAR-019"):
        adapt_semantic_refactor_context(
            packet=None,
            reuse_decision=_reuse(),
            slices=[_slice()],
            questions=[_question()],
            contracts=[_contract()],
        )
    with pytest.raises(ContextAdapterError, match="SPAR-033"):
        adapt_semantic_refactor_context(
            packet=_packet(),
            reuse_decision=None,
            slices=[_slice()],
            questions=[_question()],
            contracts=[_contract()],
        )


def test_raw_source_and_write_scope_are_required() -> None:
    with pytest.raises(ContextAdapterError, match="raw source"):
        _adapt(packet=_packet(raw_source_cids=[]))
    with pytest.raises(ContextAdapterError, match="unrestricted scope"):
        _adapt(packet=_packet(write_paths=[]))
    with pytest.raises(ContextAdapterError, match="unrestricted scope"):
        _adapt(packet=_packet(write_paths=["pkg/*.py"]))
    with pytest.raises(ContextAdapterError, match="unrestricted scope"):
        _adapt(packet=_packet(write_paths=["/tmp/pkg/mod.py"]))
    with pytest.raises(ContextAdapterError, match="unrestricted scope"):
        _adapt(packet=_packet(write_paths=["pkg/../secret.py"]))
    with pytest.raises(ContextAdapterError, match="validation_commands"):
        _adapt(packet=_packet(validation_commands=[]))


def test_slice_must_bind_owned_path_and_declared_source() -> None:
    with pytest.raises(ContextAdapterError, match="owned write path"):
        _adapt(slices=[_slice(path="pkg/other.py")])
    with pytest.raises(ContextAdapterError, match="declared raw source"):
        _adapt(slices=[_slice(source_cid=_cid("other-source"))])
    with pytest.raises(ContextAdapterError, match="affected slice"):
        _adapt(questions=[_question(slice_id="slice:missing")])
    with pytest.raises(ContextAdapterError, match="packed contracts"):
        _adapt(questions=[_question(contract_ids=["contract:missing"])])


def test_body_free_context_rejects_source_dumps() -> None:
    with pytest.raises(ContextAdapterError, match="body-free"):
        _adapt(questions=[_question(source="def fn():\n    return 1\n")])
    with pytest.raises(ContextAdapterError, match="body-free"):
        _adapt(packet=_packet(full_task_dump="task prose"))
    with pytest.raises(ContextAdapterError, match="body-free"):
        _adapt(questions=[_question(prompt="full prompt")])


def test_forbidden_effects_and_network_are_denied() -> None:
    with pytest.raises(ContextAdapterError, match="forbidden effects"):
        _adapt(allowed_effects=["network"])
    with pytest.raises(ContextAdapterError, match="undeclared"):
        _adapt(allowed_effects=["invented_effect"])
    with pytest.raises(ContextAdapterError, match="network is denied"):
        _adapt(network="allow")


def test_compiler_receipt_cannot_replace_context_compiler() -> None:
    receipt = _adapt(compiler_receipt={"receipt_cid": _cid("compiler")})
    assert receipt.context.compiler_receipt_cid == _cid("compiler")
    assert receipt.context.context_compiler_remains_authority is True
    with pytest.raises(ContextAdapterError, match="cannot replace ContextCompiler"):
        _adapt(compiler_receipt={"replaced": True})
    with pytest.raises(ContextAdapterError, match="cannot replace ContextCompiler"):
        _adapt(compiler_receipt={"adapter_is_context_compiler": True})
    with pytest.raises(ContextAdapterError, match="can_authorize_completion"):
        _adapt(compiler_receipt={"can_authorize_completion": True})


def test_round_trip_and_receipt_are_deterministic() -> None:
    first = _adapt()
    second = _adapt()
    assert first.receipt_cid == second.receipt_cid
    restored_context = decode_canonical_context(encode_canonical_context(first.context))
    assert restored_context == first.context
    assert restored_context.context_cid == first.context.context_cid
    restored_route = decode_canonical_route(encode_canonical_route(first.route))
    assert restored_route == first.route
    restored_receipt = decode_canonical_receipt(
        encode_canonical_receipt(first),
        context=first.context,
        route=first.route,
    )
    assert restored_receipt == first
    assert ResidualRouteReceipt.from_dict(
        first.to_dict(), context=first.context, route=first.route
    ).receipt_cid == first.receipt_cid
    slice_obj = compile_affected_slice(**_slice())
    restored_slice = AffectedSlice.from_dict(slice_obj.to_dict())
    assert restored_slice == slice_obj


def test_identity_excludes_observational_fields() -> None:
    receipt = _adapt()
    payload = receipt.context.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(ContextAdapterError, match="observational"):
        SemanticRefactorContext.from_dict(dirty)


def test_context_cannot_claim_authority_flags() -> None:
    receipt = _adapt()
    payload = receipt.context.to_dict()
    payload["can_authorize_completion"] = True
    payload["context_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "context_cid"}
    )
    with pytest.raises(ContextAdapterError, match="can_authorize_completion"):
        SemanticRefactorContext.from_dict(payload)
    payload = receipt.context.to_dict()
    payload["adapter_is_nomination_only"] = False
    payload["context_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "context_cid"}
    )
    with pytest.raises(ContextAdapterError, match="nomination_only"):
        SemanticRefactorContext.from_dict(payload)
    payload = receipt.context.to_dict()
    payload["raw_source_required"] = False
    payload["context_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "context_cid"}
    )
    with pytest.raises(ContextAdapterError, match="raw_source_required"):
        SemanticRefactorContext.from_dict(payload)
    payload = receipt.context.to_dict()
    payload["independent_validation_required"] = False
    payload["context_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "context_cid"}
    )
    with pytest.raises(ContextAdapterError, match="independent validation"):
        SemanticRefactorContext.from_dict(payload)
    payload = receipt.context.to_dict()
    payload["context_compiler_remains_authority"] = False
    payload["context_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "context_cid"}
    )
    with pytest.raises(ContextAdapterError, match="ContextCompiler"):
        SemanticRefactorContext.from_dict(payload)
    payload = receipt.context.to_dict()
    payload["can_weaken_validation"] = True
    payload["context_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "context_cid"}
    )
    with pytest.raises(ContextAdapterError, match="can_weaken_validation"):
        SemanticRefactorContext.from_dict(payload)


def test_route_cannot_claim_completion_or_weaken_validation() -> None:
    receipt = _adapt()
    payload = receipt.route.to_dict()
    payload["can_authorize_completion"] = True
    payload["route_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "route_cid"}
    )
    with pytest.raises(ContextAdapterError, match="can_authorize_completion"):
        ResidualModelRoute.from_dict(payload)
    payload = receipt.route.to_dict()
    payload["model_output_is_proposal_only"] = False
    payload["route_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "route_cid"}
    )
    with pytest.raises(ContextAdapterError, match="proposal-only"):
        ResidualModelRoute.from_dict(payload)


def test_adapter_dry_run_is_deterministic_and_does_not_mutate() -> None:
    adapter = SemanticRefactorContextAdapter()
    first = adapter.dry_run(
        packet=_packet(),
        reuse_decision=_reuse(),
        slices=[_slice()],
        questions=[_question()],
        contracts=[_contract()],
    )
    second = dry_run_context_route(
        packet=_packet(),
        reuse_decision=_reuse(),
        slices=[_slice()],
        questions=[_question()],
        contracts=[_contract()],
    )
    third = adapter.adapt(
        packet=_packet(),
        reuse_decision=_reuse(),
        slices=[_slice()],
        questions=[_question()],
        contracts=[_contract()],
    )
    assert first.receipt_cid == second.receipt_cid == third.receipt_cid
    assert first.mutated is False
    rebuilt = adapter.receipt(first.context, first.route)
    assert rebuilt.receipt_cid == first.receipt_cid


def test_packed_context_carries_contracts_counterexamples_and_effects() -> None:
    receipt = _adapt(
        counterexamples=[
            {
                "counterexample_id": "ce:identity",
                "evidence_cid": _cid("cex"),
                "kind": "boundary",
                "replayed": True,
                "source_authority": "replayed_counterexample",
            }
        ],
        evidence=[
            {
                "evidence_id": "ev:static",
                "evidence_cid": _cid("static"),
                "evidence_class": "exact_static_fact",
            }
        ],
        allowed_effects=["bounded_source_edit", "isolated_validation"],
    )
    assert receipt.context.counterexamples[0]["replayed"] is True
    assert receipt.context.evidence[0]["evidence_class"] == "exact_static_fact"
    assert receipt.context.allowed_effects == (
        "bounded_source_edit",
        "isolated_validation",
    )
    with pytest.raises(ContextAdapterError, match="cannot admit"):
        _adapt(
            counterexamples=[
                {
                    "counterexample_id": "ce:vec",
                    "evidence_cid": _cid("cex"),
                    "source_authority": "vector_candidate",
                }
            ]
        )


def test_question_must_remain_typed_and_unresolved() -> None:
    with pytest.raises(ContextAdapterError, match="typed"):
        compile_named_question(
            question_id="q:x",
            kind=QuestionKind.BOUNDARY_CONTRACT.value,
            statement="x",
            slice_id="slice:mod",
            typed=False,
        )
    with pytest.raises(ContextAdapterError, match="unresolved"):
        compile_named_question(
            question_id="q:x",
            kind=QuestionKind.BOUNDARY_CONTRACT.value,
            statement="x",
            slice_id="slice:mod",
            unresolved=False,
        )
