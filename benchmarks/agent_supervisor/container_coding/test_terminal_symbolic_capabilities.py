"""Observational capability reports cannot promote presence into authority."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_symbolic_capabilities as caps
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity


def inputs(*, empty=False, create=False, partition=True):
    """Authored metadata fixture; no claim that this dictionary has a signature."""
    program = {} if empty else {"source.py": {"sha256": "1" * 64, "executable": False}}
    support = {name: {"sha256": str(index + 2) * 64, "executable": False}
               for index, name in enumerate(caps._SUPPORT)}
    spec = {"task_key": "TB-CODE-TASK", "scope_paths": [*support, *program],
        "outputs": [{"path": "result.py" if create else "source.py",
                     "effect": "create" if create else "modify", "media_type": "text/x-python"}],
        "dependencies": [], "validations": [{"validation_key": "public-structural-smoke",
            "argv": ["python3", "-B", ".supervisor-public-smoke.py"]}], "acceptance": []}
    manifest = {"schema": "supervisor-local-benchmark-manifest@3", "sources": {**program, **support},
                "tasks": [spec]}
    envelope = {"payload": manifest, "binding": {"identity": "authored-fixture-identity",
        "signature": "authored-fixture-signature-not-verified", "profile_id": "authored-fixture-profile"}}
    manifest_cid = content_identity(envelope)
    task_cid = content_identity({"fixture": "task"})
    result = {"schema": "supervisor-doctor-task-preparation@1", "task_cid": task_cid,
        "manifest_cid": manifest_cid, "status": "residual", "analysis_status": "available",
        "source_hashes": {name: row["sha256"] for name, row in manifest["sources"].items()},
        "reason_codes": ["no_supported_keyword_mismatch"]}
    if partition:
        item = {"schema": "doctor-terminal-source-partition@1", "manifest_cid": manifest_cid,
            "task_cid": task_cid, "profile_sha256": support[".supervisor-task-profile.json"]["sha256"],
            "program_paths": list(program), "harness_support": [
                {"path": name, "role": role, "sha256": support[name]["sha256"]}
                for name, role in caps._SUPPORT.items()],
            "execution_authority": False, "proof_authority": False, "completion_authority": False}
        item["partition_cid"] = content_identity(item)
        result["source_partition"] = item
    return {"manifest": envelope, "task_spec": spec, "task_cid": task_cid,
            "doctor_result": result, "prover_paths": {"lean": None, "z3": None}}


def test_complete_inventory_is_not_complete_semantics_or_proof():
    report = caps.assess_terminal_symbolic_capabilities(**inputs())
    assert report["inventory"]["doctor_hash_binding"] == "complete_hash_match"
    assert report["inventory"]["partition"]["program_input_count"] == 1
    assert report["inventory"]["semantic_coverage"] == "not_established_by_this_observation"
    assert report["contracts"]["named_structural_check"] is True
    assert report["contracts"]["validation_semantics_verified"] is False
    assert report["contracts"]["task_behavior_contract"] == "not_selected"
    assert report["planning"]["symbolic_selection_execution"] == "not_observed_here"
    assert "single_declared_task_no_parallel_decomposition" in report["gap_codes"]
    assert report["benchmark_solvability"] == "not_established"
    assert not any(report[key] for key in ("proof_authority", "execution_authority",
        "publication_authority", "completion_authority"))
    assert len(json.dumps(report).encode()) < caps.MAX_REPORT_BYTES


@pytest.mark.parametrize('operator,expected', [
    ('closed-local-keyword-rename@1', 'closed_local_keyword_rename'),
    ('closed-imported-alias-call@1', 'closed_imported_alias_call'),
    (None, 'not_reported'), ('PRIVATE_UNKNOWN_OPERATOR', 'unrecognized'),
])
def test_workflow_observation_distinguishes_actual_operator_without_inventing_selection(operator, expected):
    args = inputs()
    if operator is not None:
        args['doctor_result']['operator'] = operator
    report = caps.assess_terminal_symbolic_capabilities(**args)
    assert report['operators']['selected_workflow'] == expected
    assert report['operators']['candidate_ready_reported'] is False
    assert report['proof']['local_contract_proof_reported'] is False
    assert 'PRIVATE_UNKNOWN_OPERATOR' not in json.dumps(report)


def test_executable_presence_does_not_mean_a_proof_or_execution(tmp_path, monkeypatch):
    args = inputs()
    tool = tmp_path / "not-executed"
    tool.write_text("raise RuntimeError('must never run')")
    tool.chmod(0o755)
    args["prover_paths"] = {"lean": tool, "z3": tool}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("executed a tool"))
    monkeypatch.setattr(Path, "read_bytes", lambda *a, **k: pytest.fail("read a body"))
    report = caps.assess_terminal_symbolic_capabilities(**args)
    assert set(report["provers"]["presence"].values()) == {"executable_present"}
    assert report["provers"]["executed_by_assessment"] is False
    assert report["proof"]["local_contract_proof_reported"] is False
    assert "local_prover_unavailable" not in report["gap_codes"]
    assert "local_contract_proof_not_reported" in report["gap_codes"]


def test_prover_absence_nonexecutable_and_directory_are_distinct(tmp_path):
    args = inputs()
    tool = tmp_path / "tool"
    tool.write_text("data")
    args["prover_paths"] = {"lean": tool, "z3": tmp_path / "missing"}
    report = caps.assess_terminal_symbolic_capabilities(**args)
    assert report["provers"]["presence"] == {"lean": "not_executable", "z3": "unavailable"}
    args["prover_paths"]["lean"] = tmp_path
    assert caps.assess_terminal_symbolic_capabilities(**args)["provers"]["presence"]["lean"] == "not_executable"


def test_empty_and_created_source_remain_explicit_gaps():
    report = caps.assess_terminal_symbolic_capabilities(**inputs(empty=True, create=True))
    assert report["inventory"]["partition"]["program_input_count"] == 0
    assert {"empty_program_context_required", "generic_operator_output_coverage_missing"} <= set(report["gap_codes"])
    assert report["operators"]["declared_output_effect_counts"] == {"create": 1}


def test_legacy_unpartitioned_manifest_does_not_invent_code_coverage():
    report = caps.assess_terminal_symbolic_capabilities(**inputs(partition=False))
    assert report["inventory"]["partition"] is None
    assert "program_and_support_partition_not_observed" in report["gap_codes"]


def test_native_proof_report_is_observed_but_never_reverified():
    args = inputs()
    args["doctor_result"].update(status="candidate_ready", stages={"proof": {
        "status": "executed", "receipt_id": content_identity({"fixture": "proof"}),
        "mutation_capable": True, "scope": "PRIVATE_PROOF_BODY"}})
    report = caps.assess_terminal_symbolic_capabilities(**args)
    assert report["proof"]["local_contract_proof_reported"] is True
    assert report["proof"]["independently_reverified_by_assessment"] is False
    assert report["proof"]["whole_program_verified"] is False
    assert "PRIVATE" not in json.dumps(report)


def test_header_profile_remains_local_scoped_and_separate_from_generic_operator():
    args = inputs(partition=False)
    generic = args["doctor_result"]
    args["contract_profile"] = "wsgi-header-controls@1"
    args["doctor_result"] = {"schema": "supervisor-header-contract-workflow@1",
        "task_cid": args["task_cid"], "status": "candidate_ready", "analysis": {
            "manifest_cid": generic["manifest_cid"], "status": "available",
            "source_hashes": {"source.py": "1" * 64}},
        "proof": {"status": "proved_local_contract", "proof_receipt_id": content_identity({"fixture": "proof"})}}
    report = caps.assess_terminal_symbolic_capabilities(**args)
    assert report["inventory"]["doctor_hash_binding"] == "scoped_hash_match"
    assert report["contracts"]["task_behavior_contract"] == "reviewed_local_header_contract"
    assert report["operators"]["selected_workflow"] == "reviewed_header_guard"
    assert report["proof"]["local_contract_proof_reported"] is True
    assert report["contracts"]["whole_task_behavior_verified"] is False


def test_unknown_result_text_is_not_exported_and_return_value_is_detached():
    args = inputs()
    args["doctor_result"]["reason_codes"].append("PRIVATE_RUNTIME_MESSAGE")
    args["doctor_result"]["model_text"] = "PRIVATE_MODEL_BODY"
    before = deepcopy(args)
    report = caps.assess_terminal_symbolic_capabilities(**args)
    assert report["operators"]["other_reason_count"] == 1
    assert "PRIVATE" not in json.dumps(report)
    report["inventory"]["partition"]["program_input_count"] = 99
    assert args == before


def test_native_proof_bounds_remain_an_explicit_residual():
    args = inputs()
    args['doctor_result'].update(operator='closed-imported-alias-call@1',
        reason_codes=['operator_proof_bounds_exceeded'])
    report = caps.assess_terminal_symbolic_capabilities(**args)
    assert report['operators']['known_residual_reasons'] == ['operator_proof_bounds_exceeded']
    assert report['operators']['other_reason_count'] == 0
    assert report['operators']['candidate_ready_reported'] is False
    assert report['proof']['local_contract_proof_reported'] is False


@pytest.mark.parametrize("mutation", ["foreign_task", "foreign_manifest", "task_not_member",
    "duplicate_task", "stale_source", "extra_source", "unsupported_schema", "too_many_reasons",
    "oversize_reason", "missing_proof_receipt", "bad_prover_map", "bad_effect", "bad_stages"])
def test_inconsistent_or_unbounded_inputs_refused(mutation):
    args = inputs()
    result = args["doctor_result"]
    if mutation == "foreign_task": result["task_cid"] = "PRIVATE_TASK"
    elif mutation == "foreign_manifest": result["manifest_cid"] = "PRIVATE_MANIFEST"
    elif mutation == "task_not_member": args["task_spec"] = {**args["task_spec"], "task_key": "PRIVATE_TASK"}
    elif mutation == "duplicate_task": args["manifest"]["payload"]["tasks"] *= 2
    elif mutation == "stale_source": result["source_hashes"]["source.py"] = "0" * 64
    elif mutation == "extra_source": result["source_hashes"]["PRIVATE_SOURCE"] = "0" * 64
    elif mutation == "unsupported_schema": result["schema"] = "PRIVATE_SCHEMA"
    elif mutation == "too_many_reasons": result["reason_codes"] *= 129
    elif mutation == "oversize_reason": result["reason_codes"] = ["x" * 257]
    elif mutation == "missing_proof_receipt": result["stages"] = {"proof": {"status": "executed", "mutation_capable": True}}
    elif mutation == "bad_prover_map": args["prover_paths"]["PRIVATE_TOOL"] = None
    elif mutation == "bad_effect": args["task_spec"]["outputs"][0]["effect"] = "PRIVATE_EFFECT"
    elif mutation == "bad_stages": result["stages"] = "PRIVATE_STAGES"
    with pytest.raises(ValueError) as error:
        caps.assess_terminal_symbolic_capabilities(**args)
    assert "PRIVATE" not in str(error.value)


@pytest.mark.parametrize("mutation", ["missing_program", "duplicate_program", "overlap",
    "wrong_support_hash", "wrong_role", "duplicate_support", "profile_hash", "authority", "cid"])
def test_partition_cannot_hide_source_or_grant_authority(mutation):
    args = inputs()
    part = args["doctor_result"]["source_partition"]
    if mutation == "missing_program": part["program_paths"] = []
    elif mutation == "duplicate_program": part["program_paths"] *= 2
    elif mutation == "overlap": part["program_paths"].append(".supervisor-instruction.md")
    elif mutation == "wrong_support_hash": part["harness_support"][0]["sha256"] = "0" * 64
    elif mutation == "wrong_role": part["harness_support"][0]["role"] = "semantic_source"
    elif mutation == "duplicate_support": part["harness_support"][1] = deepcopy(part["harness_support"][0])
    elif mutation == "profile_hash": part["profile_sha256"] = "0" * 64
    elif mutation == "authority": part["proof_authority"] = True
    elif mutation == "cid": part["partition_cid"] = "wrong"
    if mutation != "cid":
        part["partition_cid"] = content_identity({key: value for key, value in part.items() if key != "partition_cid"})
    with pytest.raises(ValueError): caps.assess_terminal_symbolic_capabilities(**args)


def test_unknown_validation_is_not_promoted_to_behavioral_check():
    args = inputs(partition=False)
    args["task_spec"]["validations"][0]["validation_key"] = "unknown-check"
    args["doctor_result"]["manifest_cid"] = content_identity(args["manifest"])
    report = caps.assess_terminal_symbolic_capabilities(**args)
    assert report["contracts"]["named_structural_check"] is False
    assert report["contracts"]["validation_scope"] == "not_classified"
    assert report["contracts"]["whole_task_behavior_verified"] is False


@pytest.mark.parametrize("mutation", ["payload_only", "changed_signature", "changed_signer", "payload_cid"])
def test_signed_envelope_identity_cannot_be_replaced_by_payload_identity(mutation):
    args = inputs()
    if mutation == "payload_only": args["manifest"] = args["manifest"]["payload"]
    elif mutation == "changed_signature": args["manifest"]["binding"]["signature"] = "another-signature"
    elif mutation == "changed_signer": args["manifest"]["binding"]["identity"] = "another-signer"
    elif mutation == "payload_cid":
        args["doctor_result"]["manifest_cid"] = content_identity(args["manifest"]["payload"])
    with pytest.raises(ValueError): caps.assess_terminal_symbolic_capabilities(**args)


def test_named_structural_validation_cannot_self_certify_arbitrary_argv():
    args = inputs(partition=False)
    args["task_spec"]["validations"][0]["argv"] = ["arbitrary-command"]
    args["doctor_result"]["manifest_cid"] = content_identity(args["manifest"])
    report = caps.assess_terminal_symbolic_capabilities(**args)
    assert report["contracts"]["named_structural_check"] is True
    assert report["contracts"]["validation_scope"] == "named_structural_check_not_verified"
    assert report["contracts"]["validation_semantics_verified"] is False


def test_requested_header_profile_does_not_select_a_header_contract_in_generic_fallback():
    args = inputs()
    args["contract_profile"] = "wsgi-header-controls@1"
    report = caps.assess_terminal_symbolic_capabilities(**args)
    assert report["contracts"]["requested_profile"] == "wsgi-header-controls@1"
    assert report["contracts"]["selected_profile"] is None
    assert report["contracts"]["task_behavior_contract"] == "not_selected"
    assert "task_behavior_contract_not_selected" in report["gap_codes"]
