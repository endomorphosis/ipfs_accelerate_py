"""Independent current-tree checks for DOEP-091 deterministic synthesis allowlist."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_ast_index import (
    StaticAnalysisState,
    build_analysis_ast_index,
)
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import (
    build_python_ast_blob_record,
)
from ipfs_accelerate_py.agent_supervisor.planning.program_repair_synthesis import (
    DETERMINISTIC_SYNTHESIS_ALLOWLIST_ANALYSIS_KINDS,
    DETERMINISTIC_SYNTHESIS_ALLOWLIST_DECISION_SCHEMA,
    DETERMINISTIC_SYNTHESIS_ALLOWLIST_FORBIDDEN_FIELDS,
    DETERMINISTIC_SYNTHESIS_ALLOWLIST_INTERFACE,
    DETERMINISTIC_SYNTHESIS_ALLOWLIST_MODES,
    DETERMINISTIC_SYNTHESIS_ALLOWLIST_OPERATOR_KINDS,
    DETERMINISTIC_SYNTHESIS_ALLOWLIST_SCHEMA,
    DETERMINISTIC_SYNTHESIS_ALLOWLIST_VERSION,
    PATCH_SYNTHESIS_ORIGIN_BOUNDED_MODEL_ASSISTED,
    PATCH_SYNTHESIS_ORIGIN_DETERMINISTIC_ALLOWLIST,
    PATCH_SYNTHESIS_ORIGIN_NONE,
    PROGRAM_REPAIR_SYNTHESIZER_INTERFACE,
    SUPERVISOR_PATCH_PLAN_SCHEMA,
    SUPERVISOR_PATCH_PLAN_SCHEMA_VERSION,
    DeterministicSynthesisAllowlist,
    DeterministicSynthesisAllowlistAdjudicator,
    DeterministicSynthesisAllowlistDecision,
    DeterministicSynthesisAllowlistDisposition,
    ProgramRepairAuthorityError,
    ProgramRepairMode,
    ProgramRepairReason,
    ProgramRepairRequest,
    ProgramRepairSynthesizer,
    admit_deterministic_synthesis_allowlist,
    create_program_repair_synthesizer,
    default_deterministic_synthesis_allowlist,
    operator_kind_on_deterministic_synthesis_allowlist,
    synthesize_program_repair,
)
from ipfs_accelerate_py.agent_supervisor.planning.repair_operator_registry import (
    RepairOperatorKind,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DoctorAuthorityRoots,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
SYNTHESIS_PATH = (
    ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/planning/program_repair_synthesis.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-091.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-091.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/planning/program_repair_synthesis.py",
    "test/api/doep/test_doep_091_add_deterministic_synthesis_allowlist.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-091.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-091.json",
)
TASK_CID = "sha256:906c939e49c178b51c8753e9aecab8f1aea60a5a22ab1203cb007e4243461f37"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
}


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _roots() -> DoctorAuthorityRoots:
    return DoctorAuthorityRoots(
        repository_id="repository:doep-091",
        forest_id="forest:doep-091",
        tree_id="tree:doep-091",
        overlay_id="overlay:doep-091",
        file_root_id="file-root:doep-091",
        ast_root_id="ast:doep-091",
        graph_id="graph:doep-091",
        corpus_id="corpus:doep-091",
        index_id="index:doep-091",
        model_id="model:doep-091",
        cache_id="cache:doep-091",
        operator_registry_id="operators:doep-091",
        translator_id="translator:doep-091",
        solver_id="solver:doep-091",
        kernel_id="kernel:doep-091",
        toolchain_id="toolchain:doep-091",
        policy_id="policy:doep-091",
        sandbox_id="sandbox:doep-091",
        environment_id="environment:doep-091",
        lease_id="lease:doep-091",
    )


def _index():
    return build_analysis_ast_index(
        [
            (
                "pkg/api.py",
                build_python_ast_blob_record(
                    "from pkg.service import run\n\ndef dispatch():\n    return run()\n",
                    blob_identity="blob:api",
                ),
            ),
            (
                "pkg/service.py",
                build_python_ast_blob_record(
                    "def run():\n    return 1\n",
                    blob_identity="blob:service",
                ),
            ),
        ]
    )


def _passed_route():
    return _index().route_ast_dependency_static_analysis(
        paths=("pkg/api.py",),
        query="dispatch run",
        static_analysis_state=StaticAnalysisState.PASSED,
        static_check_ids=("python-ast",),
    )


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_allowlist_extends_synthesizer_without_competing_subsystem() -> None:
    assert DETERMINISTIC_SYNTHESIS_ALLOWLIST_INTERFACE == "DeterministicSynthesisAllowlist@1"
    assert DETERMINISTIC_SYNTHESIS_ALLOWLIST_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/deterministic-synthesis-allowlist@1"
    )
    assert DETERMINISTIC_SYNTHESIS_ALLOWLIST_VERSION == (
        "deterministic-synthesis-allowlist/1.0.0"
    )
    assert SUPERVISOR_PATCH_PLAN_SCHEMA == (
        "ipfs_datasets_py/logic/external-work-plan-supervisor-patch-plan@1"
    )
    assert SUPERVISOR_PATCH_PLAN_SCHEMA_VERSION == (
        "external-work-plan-supervisor-patch-plan/v1"
    )
    assert PATCH_SYNTHESIS_ORIGIN_DETERMINISTIC_ALLOWLIST == "deterministic_allowlist"
    assert PATCH_SYNTHESIS_ORIGIN_BOUNDED_MODEL_ASSISTED == "bounded_model_assisted"
    assert PATCH_SYNTHESIS_ORIGIN_NONE == "none"
    assert DeterministicSynthesisAllowlistAdjudicator is ProgramRepairSynthesizer
    assert ProgramRepairSynthesizer.ALLOWLIST_INTERFACE == DETERMINISTIC_SYNTHESIS_ALLOWLIST_INTERFACE
    assert ProgramRepairSynthesizer.INTERFACE == PROGRAM_REPAIR_SYNTHESIZER_INTERFACE
    assert RepairOperatorKind.SEMANTIC_PATCH.value not in DETERMINISTIC_SYNTHESIS_ALLOWLIST_OPERATOR_KINDS
    assert RepairOperatorKind.ADD_ARGUMENT.value in DETERMINISTIC_SYNTHESIS_ALLOWLIST_OPERATOR_KINDS
    assert RepairOperatorKind.EQUALITY_REWRITE.value in DETERMINISTIC_SYNTHESIS_ALLOWLIST_OPERATOR_KINDS
    assert ProgramRepairMode.DETERMINISTIC.value in DETERMINISTIC_SYNTHESIS_ALLOWLIST_MODES
    assert ProgramRepairMode.HYBRID_RESIDUAL.value not in DETERMINISTIC_SYNTHESIS_ALLOWLIST_MODES
    assert "localized_exact" in DETERMINISTIC_SYNTHESIS_ALLOWLIST_ANALYSIS_KINDS
    assert "opaque" not in DETERMINISTIC_SYNTHESIS_ALLOWLIST_ANALYSIS_KINDS
    source = SYNTHESIS_PATH.read_text(encoding="utf-8").lower()
    assert "not a competing synthesizer" in source
    assert "not a second planner" in source
    assert "model assertion is never patch acceptance" in source
    assert "deterministic_allowlist" in source
    assert "completion authority" in source
    assert "class deterministicsynthesisengine" not in source
    assert "class patchplanengine" not in source
    assert "class allowlistedsynthesizer" not in source
    assert "lease_id" in DETERMINISTIC_SYNTHESIS_ALLOWLIST_FORBIDDEN_FIELDS
    assert "applied_diff" in DETERMINISTIC_SYNTHESIS_ALLOWLIST_FORBIDDEN_FIELDS
    assert "merge_decision" in DETERMINISTIC_SYNTHESIS_ALLOWLIST_FORBIDDEN_FIELDS
    assert "self_granted_acceptance" in DETERMINISTIC_SYNTHESIS_ALLOWLIST_FORBIDDEN_FIELDS
    payload = default_deterministic_synthesis_allowlist().to_dict()
    assert payload["schema"] == DETERMINISTIC_SYNTHESIS_ALLOWLIST_SCHEMA
    assert payload["completion_authoritative"] is False
    assert payload["model_assertion_is_acceptance"] is False
    assert payload["max_model_calls"] == 0
    assert "authorizes_merge" not in json.dumps(payload)


def test_allowlisted_operators_project_supervisor_patch_plan() -> None:
    first = admit_deterministic_synthesis_allowlist(
        operator_kinds=("add_argument", "exact_rename"),
        target_paths=("pkg/api.py",),
        ast_route=_passed_route(),
        patch_plan_id="patch:DOEP-091",
        task_id="DOEP-091",
        context_pack_id="contextpack:DOEP-090",
        edits=(
            {
                "edit_id": "edit:named-allowlist",
                "path": "pkg/api.py",
                "operation": "replace",
                "intent": "Thread the missing argument through the allowlisted operator.",
            },
        ),
        acceptance_conditions=(
            {
                "condition_id": "tests-pass",
                "description": "The selected current-tree tests pass.",
                "verification_method": "independent pytest",
            },
        ),
    )
    second = ProgramRepairSynthesizer().admit_deterministic_synthesis_allowlist(
        operator_kinds=("add_argument", "exact_rename"),
        target_paths=("pkg/api.py",),
        ast_route=_passed_route(),
        patch_plan_id="patch:DOEP-091",
        task_id="DOEP-091",
        context_pack_id="contextpack:DOEP-090",
        edits=(
            {
                "edit_id": "edit:named-allowlist",
                "path": "pkg/api.py",
                "operation": "replace",
                "intent": "Thread the missing argument through the allowlisted operator.",
            },
        ),
        acceptance_conditions=(
            {
                "condition_id": "tests-pass",
                "description": "The selected current-tree tests pass.",
                "verification_method": "independent pytest",
            },
        ),
    )
    assert first.disposition is DeterministicSynthesisAllowlistDisposition.ADMITTED
    assert first.admitted is True
    assert first.model_calls == 0
    assert first.completion_authoritative is False
    assert first.model_assertion_is_acceptance is False
    assert first.proposal_only is True
    assert first.synthesis_origin == PATCH_SYNTHESIS_ORIGIN_DETERMINISTIC_ALLOWLIST
    assert first.admitted_operator_kinds == ("add_argument", "exact_rename")
    plan = first.supervisor_patch_plan
    assert plan is not None
    assert plan["schema"] == SUPERVISOR_PATCH_PLAN_SCHEMA
    assert plan["schema_version"] == SUPERVISOR_PATCH_PLAN_SCHEMA_VERSION
    assert plan["kind"] == "typed_edit"
    assert plan["synthesis_origin"] == "deterministic_allowlist"
    assert plan["target_paths"] == ["pkg/api.py"]
    assert plan["semantic_nonempty"] is True
    assert plan["acceptance_requires_independent_evidence"] is True
    assert plan["completion_authoritative"] is False
    assert plan["mutation_authoritative"] is False
    assert plan["model_assertion_is_acceptance"] is False
    assert first.to_dict() == second.to_dict()
    assert DeterministicSynthesisAllowlistDecision.from_dict(first.to_dict()).to_dict() == first.to_dict()
    assert first.to_dict()["schema"] == DETERMINISTIC_SYNTHESIS_ALLOWLIST_DECISION_SCHEMA
    assert operator_kind_on_deterministic_synthesis_allowlist("missing_argument") is True
    receipt = synthesize_program_repair(
        ProgramRepairRequest(
            roots=_roots(),
            obligation_refs=("obligation:doep-091",),
            target_paths=("pkg/api.py",),
            operator_kinds=("add_argument", RepairOperatorKind.SEMANTIC_PATCH.value),
            placement_refs=("placement:exact",),
            value_refs=("value:unique",),
            proof_refs=("proof:nomination",),
            mode=ProgramRepairMode.ENUMERATIVE,
        )
    )
    assert all(item.operator_kind != RepairOperatorKind.SEMANTIC_PATCH.value for item in receipt.candidates)
    assert receipt.llm_invocation_count == 0
    assert receipt.deterministic_zero_model_calls is True


def test_unallowlisted_and_authority_paths_fail_closed() -> None:
    rejected = admit_deterministic_synthesis_allowlist(
        operator_kinds=("semantic_patch",),
        target_paths=("pkg/api.py",),
    )
    assert rejected.disposition is DeterministicSynthesisAllowlistDisposition.REJECTED
    assert ProgramRepairReason.OPERATOR_NOT_ALLOWLISTED.value in rejected.reason_codes
    assert rejected.supervisor_patch_plan is None

    hybrid = admit_deterministic_synthesis_allowlist(
        operator_kinds=("add_argument",),
        target_paths=("pkg/api.py",),
        mode=ProgramRepairMode.HYBRID_RESIDUAL,
    )
    assert hybrid.disposition is DeterministicSynthesisAllowlistDisposition.REJECTED
    assert ProgramRepairReason.PROVIDER_OR_MODEL_CALL.value in hybrid.reason_codes

    origin = admit_deterministic_synthesis_allowlist(
        operator_kinds=("add_argument",),
        target_paths=("pkg/api.py",),
        supervisor_patch_plan={
            "kind": "typed_edit",
            "synthesis_origin": PATCH_SYNTHESIS_ORIGIN_BOUNDED_MODEL_ASSISTED,
            "target_paths": ["pkg/api.py"],
        },
    )
    assert origin.disposition is DeterministicSynthesisAllowlistDisposition.REJECTED
    assert ProgramRepairReason.PATCH_PLAN_ORIGIN_MISMATCH.value in origin.reason_codes

    failed = _index().route_ast_dependency_static_analysis(
        paths=("pkg/api.py",),
        static_analysis_state=StaticAnalysisState.FAILED,
    )
    static_reject = admit_deterministic_synthesis_allowlist(
        operator_kinds=("add_argument",),
        target_paths=("pkg/api.py",),
        ast_route=failed,
    )
    assert static_reject.disposition is DeterministicSynthesisAllowlistDisposition.REJECTED
    assert ProgramRepairReason.STATIC_ANALYSIS_NOT_PASSED.value in static_reject.reason_codes

    opaque = _index().route_ast_dependency_static_analysis(
        paths=("pkg/api.py",),
        static_analysis_state=StaticAnalysisState.PASSED,
        static_check_ids=("python-ast",),
        opaque_dependencies=("generated.vendor.runtime",),
    )
    opaque_reject = admit_deterministic_synthesis_allowlist(
        operator_kinds=("add_argument",),
        target_paths=("pkg/api.py",),
        ast_route=opaque,
    )
    assert opaque_reject.disposition is DeterministicSynthesisAllowlistDisposition.REJECTED
    assert ProgramRepairReason.OPAQUE_DEPENDENCY.value in opaque_reject.reason_codes

    with pytest.raises(ProgramRepairAuthorityError, match="completion authority"):
        DeterministicSynthesisAllowlist(completion_authoritative=True)
    with pytest.raises(ProgramRepairAuthorityError, match="model assertion"):
        DeterministicSynthesisAllowlistDecision(
            disposition=DeterministicSynthesisAllowlistDisposition.ADMITTED,
            reason_codes=(ProgramRepairReason.ALLOWLIST_ADMITTED.value,),
            model_assertion_is_acceptance=True,
        )
    with pytest.raises(ProgramRepairAuthorityError, match="operational authority"):
        admit_deterministic_synthesis_allowlist(
            operator_kinds=("add_argument",),
            target_paths=("pkg/api.py",),
            supervisor_patch_plan={
                "kind": "typed_edit",
                "synthesis_origin": "deterministic_allowlist",
                "target_paths": ["pkg/api.py"],
                "lease_id": "lease:x",
            },
        )
    with pytest.raises(ProgramRepairAuthorityError, match="model assertion"):
        admit_deterministic_synthesis_allowlist(
            operator_kinds=("add_argument",),
            target_paths=("pkg/api.py",),
            supervisor_patch_plan={
                "kind": "typed_edit",
                "synthesis_origin": "deterministic_allowlist",
                "target_paths": ["pkg/api.py"],
                "model_assertion_is_acceptance": True,
            },
        )


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-091"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["entrypoint"] == "admit_deterministic_synthesis_allowlist"
    assert manifest["canonical_extension"]["carrier"] == "ProgramRepairSynthesizer"
    assert manifest["canonical_extension"]["interface"] == DETERMINISTIC_SYNTHESIS_ALLOWLIST_INTERFACE
    assert manifest["canonical_extension"]["schema"] == DETERMINISTIC_SYNTHESIS_ALLOWLIST_SCHEMA
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(SYNTHESIS_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
