"""Counterevidence-refined CEGIS for ProgramRepairSynthesizer@1 (LGCVF-081).

Required evidence: CEGIS refinement from counterexamples, cores, failed
assumptions, and validated interpolants; effect/security restriction; and
zero-model coverage. Unvalidated interpolants cannot refine search.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DoctorAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.planning import (
    program_repair_synthesis as program_repair_synthesis_mod,
)
from ipfs_accelerate_py.agent_supervisor.planning import (
    repair_operator_registry as repair_operator_registry_mod,
)
from ipfs_accelerate_py.agent_supervisor.planning.program_repair_synthesis import (
    PROGRAM_REPAIR_SYNTHESIZER_INTERFACE,
    CounterevidencePacket,
    ProgramRepairDisposition,
    ProgramRepairMode,
    ProgramRepairReason,
    ProgramRepairRequest,
    ProgramRepairSynthesisError,
    ValidatedInterpolantEvidence,
    collect_counterevidence_packet,
    refine_operators_from_counterevidence,
    synthesize_program_repair,
)
from ipfs_accelerate_py.agent_supervisor.planning.repair_operator_registry import (
    CANONICAL_EFFECT_RESTRICTIONS,
    RepairEffectRestriction,
    RepairOperatorKind,
    RepairOperatorLookupReason,
    build_default_repair_operator_registry,
    candidate_effect_violations,
    refine_repair_operators,
)
from ipfs_accelerate_py.agent_supervisor.proof.counterexample_guided_tactician import (
    CandidateKind,
    CegisStopReason,
    RefinementCandidate,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_counterexamples import (
    CounterexampleKind,
    RepairClass,
    normalize_counterexample,
)


SYNTHESIZER_PATH = Path(program_repair_synthesis_mod.__file__)
REGISTRY_PATH = Path(repair_operator_registry_mod.__file__)
_FORBIDDEN_MODEL_MODULES = {
    "openai",
    "anthropic",
    "llm_router",
    "model_provider",
    "provider_router",
}


def roots() -> DoctorAuthorityRoots:
    return DoctorAuthorityRoots(
        repository_id="repository:fixture",
        forest_id="forest:fixture",
        tree_id="tree:fixture",
        overlay_id="overlay:fixture",
        file_root_id="file-root:fixture",
        ast_root_id="ast:fixture",
        graph_id="graph:fixture",
        corpus_id="corpus:fixture",
        index_id="index:fixture",
        model_id="model:fixture",
        cache_id="cache:fixture",
        operator_registry_id="operators:fixture",
        translator_id="translator:fixture",
        solver_id="solver:fixture",
        kernel_id="kernel:fixture",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
        sandbox_id="sandbox:fixture",
        environment_id="environment:fixture",
        lease_id="lease:fixture",
    )


def counterexample(**overrides: Any):
    payload = {
        "kind": CounterexampleKind.SMT_MODEL.value,
        "failure": {"code": "arity-mismatch", "predicate": "missing_argument"},
        "unsat_core": ("clause:missing_argument", "clause:call_arity"),
    }
    payload.update(overrides.pop("payload", {}))
    return normalize_counterexample(
        payload,
        kind=overrides.get("kind", CounterexampleKind.SMT_MODEL),
        violated_property=overrides.get("violated_property", "obligation:arity"),
        bindings={
            "plan_id": "plan:base",
            "task_id": "LGCVF-081",
            "ast_scope_id": "symbol:target",
            "tree_id": "tree:fixture",
            "assumption_id": "assumption:missing_argument",
            "provider_id": "tool:z3",
            "policy_id": "policy:fixture",
            "obligation_id": "obligation:arity",
        },
        assumption_ids=overrides.get(
            "assumption_ids", ("assumption:missing_argument",)
        ),
        finite_bounds={"portfolio_width": 1, "deadline": 20},
        repair_classes=overrides.get(
            "repair_classes", (RepairClass.ADD_PREMISE,)
        ),
    )


def validated_interpolant(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema": "validated-craig-interpolant/v1",
        "interface": "ValidatedCraigInterpolation@1",
        "status": "validated",
        "partition_a_cid": "cid:partition-a",
        "partition_b_cid": "cid:partition-b",
        "interpolant_cid": "cid:interpolant-arity",
        "interpolant": {"kind": "ge", "symbol": "arity", "value": "1"},
        "shared_vocabulary": ("arity", "argument", "value"),
        "interpolant_vocabulary": ("arity", "argument"),
        "predicates": ("missing_argument", "arity"),
        "a_implies_i": True,
        "i_and_b_unsat": True,
        "shared_vocabulary_ok": True,
        "identity_ok": True,
        "bounds_ok": True,
        "provider": "cvc5",
        "provider_version": "1.2.0",
        "theory": "QF_LIA",
    }
    payload.update(overrides)
    return payload


def closing_verify():
    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        return {
            "receipt_id": "receipt:ok",
            "counterexample_id": binding["counterexample_id"],
            "repository_tree_id": binding["repository_tree_id"],
            "property_id": binding["property_id"],
            "assumption_ids": list(binding.get("assumption_ids") or ()),
            "bound_digest": binding["bound_digest"],
            "tool_id": binding["tool_id"],
            "policy_id": binding["policy_id"],
            "repaired_plan_id": binding["repaired_plan_id"],
            "freshness": "current",
            "outcome": "verified",
            "available": True,
        }

    return verify


def test_registry_grammars_declare_counterevidence_and_effect_restrictions() -> None:
    registry = build_default_repair_operator_registry()
    for spec in registry.operators:
        assert spec.counterevidence_classes
        assert spec.effect_restrictions == tuple(sorted(CANONICAL_EFFECT_RESTRICTIONS))
        assert RepairEffectRestriction.NO_UNDECLARED_IMPORTS.value in spec.effect_restrictions
        assert RepairEffectRestriction.NO_AUTHORITY.value in spec.effect_restrictions
        assert spec.proposal_only is True
        restored = type(spec).from_dict(spec.to_dict())
        assert restored.content_id == spec.content_id
        assert restored.effect_restrictions == spec.effect_restrictions


def test_counterevidence_refines_reviewed_operators_and_excludes_unrelated() -> None:
    registry = build_default_repair_operator_registry()
    refined = refine_repair_operators(
        registry,
        operator_kinds=(
            RepairOperatorKind.ADD_ARGUMENT.value,
            RepairOperatorKind.EXACT_MOVE.value,
            RepairOperatorKind.ADD_IMPORT.value,
            RepairOperatorKind.EQUALITY_REWRITE.value,
        ),
        repair_classes=(RepairClass.ADD_PREMISE.value,),
        predicates=("missing_argument", "arity"),
        core_ids=("clause:missing_argument",),
        failed_assumption_ids=("assumption:missing_argument",),
        interpolant_vocabulary=("arity", "argument"),
        interpolant_predicates=("missing_argument",),
        interpolant_validated=True,
        counterexample_kind=CounterexampleKind.SMT_MODEL.value,
    )
    assert RepairOperatorKind.ADD_ARGUMENT.value in refined
    assert RepairOperatorKind.EXACT_MOVE.value not in refined
    assert RepairOperatorKind.ADD_IMPORT.value not in refined
    assert refined[0] == RepairOperatorKind.ADD_ARGUMENT.value


def test_forged_admitted_interpolant_without_checks_is_rejected() -> None:
    with pytest.raises(ProgramRepairSynthesisError):
        ValidatedInterpolantEvidence(
            admitted=True,
            status="validated",
            a_implies_i=False,
            i_and_b_unsat=False,
            shared_vocabulary_ok=False,
            identity_ok=False,
            bounds_ok=False,
        )


def test_unvalidated_interpolant_does_not_refine_search() -> None:
    registry = build_default_repair_operator_registry()
    with_invalid = refine_repair_operators(
        registry,
        operator_kinds=(
            RepairOperatorKind.ADD_ARGUMENT.value,
            RepairOperatorKind.EXACT_MOVE.value,
        ),
        interpolant_vocabulary=("arity", "argument"),
        interpolant_predicates=("missing_argument",),
        interpolant_validated=False,
    )
    assert with_invalid == (
        RepairOperatorKind.ADD_ARGUMENT.value,
        RepairOperatorKind.EXACT_MOVE.value,
    )
    evidence = ValidatedInterpolantEvidence.from_value(
        validated_interpolant(status="unavailable", a_implies_i=False)
    )
    assert evidence.admitted is False
    assert ProgramRepairReason.INTERPOLANT_UNVALIDATED.value in evidence.reason_codes


def test_collect_packet_uses_cores_assumptions_and_only_validated_interpolants() -> None:
    packet = collect_counterevidence_packet(
        counterexample=counterexample(),
        unsat_cores=({"core_id": "core:arity", "clause_ids": ("clause:arity",)},),
        failed_assumptions=("assumption:missing_argument",),
        interpolants=(
            validated_interpolant(),
            validated_interpolant(
                status="fallback",
                interpolant=None,
                interpolant_cid="",
                a_implies_i=False,
                i_and_b_unsat=False,
                shared_vocabulary_ok=False,
                identity_ok=False,
                bounds_ok=False,
                fallback_kind="unsat_core",
                fallback_core=("clause:arity",),
                fallback_validated=True,
                interpolant_vocabulary=(),
            ),
        ),
    )
    assert isinstance(packet, CounterevidencePacket)
    assert packet.interpolant_validated is True
    assert packet.validated_interpolants[0].admitted is True
    assert "clause:missing_argument" in packet.core_ids() or "core:arity" in {
        item.core_id for item in packet.cores
    }
    assert "assumption:missing_argument" in packet.failed_assumption_ids()
    kinds = refine_operators_from_counterevidence(
        packet,
        operator_kinds=(
            RepairOperatorKind.ADD_ARGUMENT.value,
            RepairOperatorKind.EXACT_MOVE.value,
            RepairOperatorKind.MANIFEST_UPDATE.value,
        ),
    )
    assert kinds[0] == RepairOperatorKind.ADD_ARGUMENT.value
    assert RepairOperatorKind.EXACT_MOVE.value not in kinds


def test_cegis_refines_with_counterexample_core_assumption_and_interpolant() -> None:
    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:arity",),
        target_paths=("pkg/mod.py",),
        operator_kinds=(
            RepairOperatorKind.ADD_ARGUMENT.value,
            RepairOperatorKind.EXACT_MOVE.value,
            RepairOperatorKind.ADD_IMPORT.value,
            RepairOperatorKind.RESTORE_TRACKED_ARTIFACT.value,
        ),
        mode=ProgramRepairMode.CEGIS,
        counterexample=counterexample(),
        unsat_cores=({"core_id": "core:arity", "predicates": ("missing_argument",)},),
        failed_assumptions=("assumption:missing_argument",),
        interpolants=(validated_interpolant(),),
        cegis_verify=closing_verify(),
    )
    receipt = synthesize_program_repair(request)
    assert receipt.disposition is ProgramRepairDisposition.SUPPORTED
    assert receipt.cegis_result is not None
    assert receipt.cegis_result.closed
    assert receipt.cegis_result.stop_reason is CegisStopReason.CLOSED
    assert receipt.refined_operator_kinds
    assert receipt.refined_operator_kinds[0] == RepairOperatorKind.ADD_ARGUMENT.value
    assert RepairOperatorKind.EXACT_MOVE.value not in receipt.refined_operator_kinds
    assert receipt.selected_candidate is not None
    assert receipt.selected_candidate.operator_kind == RepairOperatorKind.ADD_ARGUMENT.value
    assert receipt.selected_candidate.proposal_only
    assert ProgramRepairReason.CEGIS_REFINED.value in receipt.reason_codes
    assert ProgramRepairReason.INTERPOLANT_REFINED.value in receipt.reason_codes
    assert ProgramRepairReason.CORE_REFINED.value in receipt.reason_codes
    assert ProgramRepairReason.ASSUMPTION_REFINED.value in receipt.reason_codes
    assert receipt.deterministic_zero_model_calls
    assert receipt.llm_invocation_count == 0
    assert receipt.model_provider_call_count == 0
    assert receipt.counterevidence is not None
    assert receipt.counterevidence.interpolant_validated


def test_cegis_closes_only_on_fresh_matching_receipt() -> None:
    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:arity",),
        target_paths=("pkg/mod.py",),
        operator_kinds=(RepairOperatorKind.ADD_ARGUMENT.value,),
        mode=ProgramRepairMode.CEGIS,
        counterexample=counterexample(),
        interpolants=(validated_interpolant(),),
        cegis_verify=closing_verify(),
    )
    receipt = synthesize_program_repair(request)
    assert receipt.INTERFACE == PROGRAM_REPAIR_SYNTHESIZER_INTERFACE
    assert receipt.cegis_result is not None
    assert receipt.cegis_result.closed
    assert receipt.selected_candidate is not None
    assert receipt.selected_candidate.proposal_only is True


def test_effect_and_security_restrictions_reject_undeclared_candidate_effects() -> None:
    verify_calls = {"n": 0}

    def refine(witness, context):
        del witness, context
        return (
            RefinementCandidate(
                candidate_id="candidate:extra-import",
                kind=CandidateKind.REPAIR,
                goal_id="obligation:arity",
                repaired_tree_id="tree:fixture",
                repaired_plan_id="plan:x",
                statement="import os\nopen('/etc/passwd')",
                addresses_witness=True,
                parameters={
                    "operator_kind": RepairOperatorKind.ADD_ARGUMENT.value,
                    "extra_imports": ("os",),
                    "extra_files": ("pkg/other.py",),
                    "new_dependencies": ("requests",),
                    "write_authority": True,
                    "behavior_class": "dependency_changing",
                    "undeclared_effects": ("network",),
                },
            ),
        )

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        verify_calls["n"] += 1
        return {
            "receipt_id": "receipt:should-not-run",
            "counterexample_id": binding["counterexample_id"],
            "repository_tree_id": binding["repository_tree_id"],
            "property_id": binding["property_id"],
            "assumption_ids": list(binding.get("assumption_ids") or ()),
            "bound_digest": binding["bound_digest"],
            "tool_id": binding["tool_id"],
            "policy_id": binding["policy_id"],
            "repaired_plan_id": binding["repaired_plan_id"],
            "freshness": "current",
            "outcome": "verified",
            "available": True,
        }

    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:arity",),
        target_paths=("pkg/mod.py",),
        mode=ProgramRepairMode.CEGIS,
        counterexample=counterexample(),
        cegis_refine=refine,
        cegis_verify=verify,
    )
    receipt = synthesize_program_repair(request)
    assert not receipt.admitted
    assert receipt.cegis_result is not None
    assert not receipt.cegis_result.closed
    assert verify_calls["n"] == 0
    recorded_reasons = list(receipt.reason_codes)
    for iteration in receipt.cegis_result.iterations:
        if iteration.candidate is not None:
            recorded_reasons.append(iteration.candidate.validation_reason)
        recorded_reasons.append(iteration.binding.reason_code)
    assert any(
        ProgramRepairReason.EFFECT_RESTRICTED.value in str(code)
        or "undeclared" in str(code)
        for code in recorded_reasons
        if code
    )


def test_candidate_effect_violations_cover_imports_files_deps_authority_behavior() -> None:
    reasons = candidate_effect_violations(
        operator_kind=RepairOperatorKind.ADD_ARGUMENT.value,
        extra_imports=("os",),
        extra_files=("pkg/other.py",),
        new_dependencies=("requests",),
        undeclared_effects=("network",),
        behavior_class="public_api",
        write_authority=True,
        replacement="import socket\n",
        declared_paths=("pkg/mod.py",),
    )
    assert RepairOperatorLookupReason.UNDECLARED_IMPORT.value in reasons
    assert RepairOperatorLookupReason.UNDECLARED_FILE.value in reasons
    assert RepairOperatorLookupReason.UNDECLARED_DEPENDENCY.value in reasons
    assert RepairOperatorLookupReason.UNDECLARED_EFFECT.value in reasons
    allowed = candidate_effect_violations(
        operator_kind=RepairOperatorKind.ADD_IMPORT.value,
        extra_imports=("pkg.helpers",),
        declared_imports=("pkg.helpers",),
        declared_paths=("pkg/mod.py",),
        replacement="from pkg.helpers import helper\n",
        behavior_class="pure_local",
    )
    assert allowed == ()


def test_invalid_interpolant_vocabulary_or_implication_is_fail_closed() -> None:
    invalid = ValidatedInterpolantEvidence.from_value(
        validated_interpolant(
            a_implies_i=False,
            interpolant_vocabulary=("secret_symbol",),
            shared_vocabulary=("arity",),
        )
    )
    assert invalid.admitted is False
    generic = normalize_counterexample(
        {"kind": CounterexampleKind.GENERIC_FAILURE.value, "failure": {"code": "open"}},
        kind=CounterexampleKind.GENERIC_FAILURE,
        violated_property="obligation:arity",
        bindings={
            "plan_id": "plan:base",
            "task_id": "LGCVF-081",
            "tree_id": "tree:fixture",
            "policy_id": "policy:fixture",
            "obligation_id": "obligation:arity",
        },
        finite_bounds={"portfolio_width": 1},
        repair_classes=(),
    )
    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:arity",),
        target_paths=("pkg/mod.py",),
        operator_kinds=(
            RepairOperatorKind.ADD_ARGUMENT.value,
            RepairOperatorKind.EXACT_MOVE.value,
        ),
        mode=ProgramRepairMode.CEGIS,
        counterexample=generic,
        interpolants=(
            validated_interpolant(
                a_implies_i=False,
                interpolant_vocabulary=("secret_symbol",),
                shared_vocabulary=("arity",),
            ),
        ),
        cegis_verify=closing_verify(),
    )
    receipt = synthesize_program_repair(request)
    assert receipt.counterevidence is not None
    assert receipt.counterevidence.interpolant_validated is False
    assert ProgramRepairReason.INTERPOLANT_UNVALIDATED.value in receipt.reason_codes
    # Without a validated interpolant the interpolant vocabulary cannot drop
    # unrelated reviewed operators from the caller-declared grammar.
    assert RepairOperatorKind.EXACT_MOVE.value in receipt.refined_operator_kinds
    assert RepairOperatorKind.ADD_ARGUMENT.value in receipt.refined_operator_kinds


def test_unknown_operator_cannot_enter_refined_grammar() -> None:
    refined = refine_repair_operators(
        operator_kinds=("arbitrary_runtime_code", RepairOperatorKind.ADD_ARGUMENT.value),
        predicates=("missing_argument",),
        interpolant_validated=True,
        interpolant_predicates=("missing_argument",),
    )
    assert refined == (RepairOperatorKind.ADD_ARGUMENT.value,)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".", 1)[0])
            names.add(node.module)
    return names


def test_deterministic_cegis_proves_zero_model_calls_and_has_no_provider_imports() -> None:
    for path in (SYNTHESIZER_PATH, REGISTRY_PATH):
        imported = _imported_modules(path)
        assert imported.isdisjoint(_FORBIDDEN_MODEL_MODULES)
        source = path.read_text(encoding="utf-8")
        for marker in _FORBIDDEN_MODEL_MODULES:
            assert f"import {marker}" not in source
    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:arity",),
        target_paths=("pkg/mod.py",),
        operator_kinds=(RepairOperatorKind.ADD_ARGUMENT.value,),
        mode=ProgramRepairMode.CEGIS,
        counterexample=counterexample(),
        interpolants=(validated_interpolant(),),
        cegis_verify=closing_verify(),
    )
    receipt = synthesize_program_repair(request)
    assert receipt.deterministic_zero_model_calls is True
    assert receipt.llm_invocation_count == 0
    assert receipt.model_provider_call_count == 0
    assert receipt.provider_invoked is False
    assert receipt.proposal_only is True
    payload = receipt.to_dict()
    assert payload["deterministic_zero_model_calls"] is True
    assert payload["llm_invocation_count"] == 0
