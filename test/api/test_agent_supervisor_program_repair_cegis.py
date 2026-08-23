"""Counterevidence-refined CEGIS for ProgramRepairSynthesizer@1 (LGCVF-081).

Required evidence: refinement from cores/assumptions/validated interpolants,
unvalidated interpolant fail-closed, undeclared import/file/dependency/
authority/effect rejection, and zero-model coverage.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DoctorAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.planning.program_repair_synthesis import (
    ProgramRepairCounterevidence,
    ProgramRepairDisposition,
    ProgramRepairMode,
    ProgramRepairReason,
    ProgramRepairRequest,
    synthesize_program_repair,
)
from ipfs_accelerate_py.agent_supervisor.planning.repair_operator_registry import (
    CEGIS_FORBIDDEN_PARAMETER_KEYS,
    RepairOperatorKind,
    build_default_repair_operator_registry,
    cegis_restricted_operator_kinds,
)
from ipfs_accelerate_py.agent_supervisor.proof.counterexample_guided_tactician import (
    CandidateKind,
    RefinementCandidate,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_counterexamples import (
    CounterexampleKind,
    RepairClass,
    normalize_counterexample,
)

_SYNTHESIS_PATH = (
    Path(__file__).resolve().parents[2]
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "planning"
    / "program_repair_synthesis.py"
)
_BANNED_MODEL_MARKERS = (
    "openai",
    "anthropic",
    "litellm",
    "groq",
    "huggingface_hub",
    "transformers",
    "langchain",
    "invoke_model",
    "chat_completion",
)


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


def counterexample():
    return normalize_counterexample(
        {
            "kind": CounterexampleKind.GENERIC_FAILURE.value,
            "failure": {"code": "repair-required"},
        },
        kind=CounterexampleKind.GENERIC_FAILURE,
        violated_property="obligation:one",
        bindings={
            "plan_id": "plan:base",
            "task_id": "LGCVF-081",
            "ast_scope_id": "symbol:target",
            "tree_id": "tree:fixture",
            "assumption_id": "assumption:dep",
            "provider_id": "tool:z3",
            "policy_id": "policy:fixture",
            "obligation_id": "obligation:one",
        },
        finite_bounds={"portfolio_width": 1, "deadline": 20},
        repair_classes=(RepairClass.ADD_DEPENDENCY,),
    )


def closing_verify(binding: dict[str, Any]) -> dict[str, Any]:
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


def _assert_zero_model(receipt) -> None:
    assert receipt.deterministic_zero_model_calls is True
    assert receipt.llm_invocation_count == 0
    assert receipt.model_provider_call_count == 0
    assert ProgramRepairReason.ZERO_MODEL_CALLS.value in receipt.reason_codes


def test_registry_declares_cegis_restricted_kinds() -> None:
    restricted = cegis_restricted_operator_kinds()
    assert RepairOperatorKind.ADD_IMPORT.value in restricted
    assert RepairOperatorKind.ADD_EXPORT.value in restricted
    assert RepairOperatorKind.ADD_REGISTRATION.value in restricted
    assert RepairOperatorKind.EXACT_MOVE.value in restricted
    assert RepairOperatorKind.ADD_ARGUMENT.value not in restricted
    catalog = build_default_repair_operator_registry()
    assert catalog.cegis_restricted_kinds() == restricted
    assert "extra_imports" in CEGIS_FORBIDDEN_PARAMETER_KEYS


def test_cegis_refines_from_unsat_cores() -> None:
    receipt = synthesize_program_repair(
        ProgramRepairRequest(
            roots=roots(),
            obligation_refs=("obligation:one",),
            target_paths=("pkg/mod.py",),
            operator_kinds=(
                RepairOperatorKind.ADD_ARGUMENT.value,
                RepairOperatorKind.ADD_IMPORT.value,
            ),
            mode=ProgramRepairMode.CEGIS,
            counterexample=counterexample(),
            cegis_verify=closing_verify,
            counterevidence=ProgramRepairCounterevidence(
                unsat_core_refs=("core:add_argument",),
            ),
        )
    )
    assert receipt.disposition is ProgramRepairDisposition.SUPPORTED
    assert receipt.selected_candidate is not None
    assert receipt.selected_candidate.operator_kind == (
        RepairOperatorKind.ADD_ARGUMENT.value
    )
    assert receipt.selected_candidate.proposal_only
    _assert_zero_model(receipt)


def test_cegis_refines_from_failed_assumptions() -> None:
    receipt = synthesize_program_repair(
        ProgramRepairRequest(
            roots=roots(),
            obligation_refs=("obligation:one",),
            target_paths=("pkg/mod.py",),
            operator_kinds=(
                RepairOperatorKind.ADD_ARGUMENT.value,
                RepairOperatorKind.ADD_EXPORT.value,
            ),
            mode=ProgramRepairMode.CEGIS,
            counterexample=counterexample(),
            cegis_verify=closing_verify,
            counterevidence=ProgramRepairCounterevidence(
                failed_assumption_refs=("assumption:add_argument",),
            ),
        )
    )
    assert receipt.admitted
    assert receipt.selected_candidate is not None
    assert receipt.selected_candidate.operator_kind == (
        RepairOperatorKind.ADD_ARGUMENT.value
    )
    _assert_zero_model(receipt)


def test_cegis_refines_from_validated_interpolants() -> None:
    receipt = synthesize_program_repair(
        ProgramRepairRequest(
            roots=roots(),
            obligation_refs=("obligation:one",),
            target_paths=("pkg/mod.py",),
            operator_kinds=(
                RepairOperatorKind.ADD_ARGUMENT.value,
                RepairOperatorKind.ADD_REGISTRATION.value,
            ),
            mode=ProgramRepairMode.CEGIS,
            counterexample=counterexample(),
            cegis_verify=closing_verify,
            counterevidence=ProgramRepairCounterevidence(
                interpolant_refs=("interpolant:add_argument",),
                interpolants_independently_validated=True,
            ),
        )
    )
    assert receipt.admitted
    assert receipt.cegis_result is not None
    assert receipt.cegis_result.closed
    assert receipt.selected_candidate is not None
    assert receipt.selected_candidate.operator_kind == (
        RepairOperatorKind.ADD_ARGUMENT.value
    )
    parameters = receipt.cegis_result.selected_candidate.parameters
    assert "interpolant:add_argument" in parameters["counterevidence_tags"]
    _assert_zero_model(receipt)


def test_unvalidated_interpolants_fail_closed() -> None:
    verify_calls = {"n": 0}

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        verify_calls["n"] += 1
        return closing_verify(binding)

    receipt = synthesize_program_repair(
        ProgramRepairRequest(
            roots=roots(),
            obligation_refs=("obligation:one",),
            target_paths=("pkg/mod.py",),
            operator_kinds=(RepairOperatorKind.ADD_ARGUMENT.value,),
            mode=ProgramRepairMode.CEGIS,
            counterexample=counterexample(),
            cegis_verify=verify,
            counterevidence=ProgramRepairCounterevidence(
                unsat_core_refs=("core:add_argument",),
                interpolant_refs=("interpolant:add_argument",),
                interpolants_independently_validated=False,
            ),
        )
    )
    assert not receipt.admitted
    assert receipt.disposition is ProgramRepairDisposition.ABSTAIN
    assert (
        ProgramRepairReason.UNVALIDATED_INTERPOLANT.value in receipt.reason_codes
    )
    assert verify_calls["n"] == 0
    _assert_zero_model(receipt)


def test_effect_refs_restrict_sensitive_operators() -> None:
    receipt = synthesize_program_repair(
        ProgramRepairRequest(
            roots=roots(),
            obligation_refs=("obligation:one",),
            target_paths=("pkg/mod.py",),
            operator_kinds=(RepairOperatorKind.ADD_IMPORT.value,),
            mode=ProgramRepairMode.CEGIS,
            counterexample=counterexample(),
            cegis_verify=closing_verify,
            counterevidence=ProgramRepairCounterevidence(
                unsat_core_refs=("core:add_import",),
                effect_refs=("effect:network",),
            ),
        )
    )
    assert not receipt.admitted
    assert (
        ProgramRepairReason.COUNTEREVIDENCE_RESTRICTED.value in receipt.reason_codes
    )
    assert ProgramRepairReason.NO_ADMISSIBLE_OPERATOR.value in receipt.reason_codes
    _assert_zero_model(receipt)


def _restricted_refine(forbidden_key: str, forbidden_value: Any):
    def refine(witness, context):
        del witness
        return (
            RefinementCandidate(
                candidate_id=f"candidate:restricted:{forbidden_key}",
                kind=CandidateKind.REPAIR,
                goal_id="obligation:one",
                repaired_tree_id="tree:fixture",
                repaired_plan_id="plan:restricted",
                statement="undeclared surface",
                addresses_witness=True,
                parameters={
                    "operator_kind": RepairOperatorKind.ADD_ARGUMENT.value,
                    forbidden_key: forbidden_value,
                    "obligation_refs": list(context.get("obligation_refs") or ()),
                },
            ),
        )

    return refine


def _assert_parameter_rejected(
    forbidden_key: str,
    forbidden_value: Any,
    expected_reason: str,
) -> None:
    verify_calls = {"n": 0}

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        verify_calls["n"] += 1
        return closing_verify(binding)

    receipt = synthesize_program_repair(
        ProgramRepairRequest(
            roots=roots(),
            obligation_refs=("obligation:one",),
            target_paths=("pkg/mod.py",),
            operator_kinds=(RepairOperatorKind.ADD_ARGUMENT.value,),
            mode=ProgramRepairMode.CEGIS,
            counterexample=counterexample(),
            cegis_refine=_restricted_refine(forbidden_key, forbidden_value),
            cegis_verify=verify,
            counterevidence=ProgramRepairCounterevidence(
                unsat_core_refs=("core:add_argument",),
            ),
        )
    )
    assert not receipt.admitted
    assert receipt.cegis_result is not None
    assert not receipt.cegis_result.closed
    assert receipt.cegis_result.iterations
    assert receipt.cegis_result.iterations[0].binding.reason_code == expected_reason
    assert verify_calls["n"] == 0
    _assert_zero_model(receipt)


def test_undeclared_import_is_rejected() -> None:
    _assert_parameter_rejected(
        "extra_imports",
        ("undeclared.mod",),
        ProgramRepairReason.EXTRA_IMPORT.value,
    )


def test_undeclared_file_is_rejected() -> None:
    _assert_parameter_rejected(
        "extra_files",
        ("pkg/secret.py",),
        ProgramRepairReason.EXTRA_FILE.value,
    )


def test_undeclared_dependency_is_rejected() -> None:
    _assert_parameter_rejected(
        "extra_dependencies",
        ("undeclared-pkg",),
        ProgramRepairReason.EXTRA_DEPENDENCY.value,
    )


def test_undeclared_authority_is_rejected() -> None:
    _assert_parameter_rejected(
        "write_authority",
        True,
        ProgramRepairReason.AUTHORITY_CLAIM.value,
    )


def test_undeclared_effect_is_rejected() -> None:
    _assert_parameter_rejected(
        "undeclared_effects",
        ("network",),
        ProgramRepairReason.UNDECLARED_EFFECT.value,
    )


def test_cegis_path_has_zero_model_imports() -> None:
    source = _SYNTHESIS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    for marker in _BANNED_MODEL_MARKERS:
        assert marker not in imported
        assert f"import {marker}" not in source
    receipt = synthesize_program_repair(
        ProgramRepairRequest(
            roots=roots(),
            obligation_refs=("obligation:one",),
            target_paths=("pkg/mod.py",),
            operator_kinds=(RepairOperatorKind.ADD_ARGUMENT.value,),
            mode=ProgramRepairMode.CEGIS,
            counterexample=counterexample(),
            cegis_verify=closing_verify,
            counterevidence=ProgramRepairCounterevidence(
                unsat_core_refs=("core:add_argument",),
                failed_assumption_refs=("assumption:add_argument",),
                interpolant_refs=("interpolant:add_argument",),
                interpolants_independently_validated=True,
            ),
        )
    )
    assert receipt.admitted
    _assert_zero_model(receipt)
