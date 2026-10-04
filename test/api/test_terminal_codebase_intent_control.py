"""Public unresolved accounting and separately authored symbolic proposal controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_control as api
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.planning.intent_requirement_adapter import (
    IntentRequirementAdapterError, build_intent_planning_materials, validate_symbolic_operations,
)
from ipfs_accelerate_py.agent_supervisor.planning.intent_symbolic_planning import build_intent_symbolic_plan
from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import (
    IntentPlanCoverageError, check_intent_plan_coverage, validate_intent_requirement_contract,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    PromptEvidenceRecord, EvidenceAuthority, RecordStatus,
)
from ipfs_datasets_py.logic.intent_ir.formalize.requirements import validate_intent_requirement_ledger


ROOT = Path(__file__).resolve().parents[4]
CAPTURE = ROOT / "artifacts/codebase_ir_terminal_bench/qualification-20261001-04"
INSTRUCTION = CAPTURE / "supervisor/repository/.supervisor-instruction.md"
LOGIC = ROOT / "artifacts/codebase_ir_terminal_bench/decoder-qualification-20261001-01/logic/decoder-logic-result.json"


@pytest.fixture
def control():
    return api.build_terminal_intent_control(public_instruction_bytes=INSTRUCTION.read_bytes())


def _unsigned_declaration(control, *, model_only=False):
    """A pure proposal declaration, never the original signature or admission."""
    fixture = json.loads((CAPTURE / "fixture.json").read_bytes())
    payload = deepcopy(fixture["prepared"]["manifest"]["payload"])
    contract = control["authored_control"]["contract"]
    payload["schema"] = "supervisor-local-benchmark-manifest@4"
    payload["sources"][api.AUTHORED_CONTROL_PATH] = {
        "sha256": control["authored_control"]["source_sha256"], "executable": False}
    payload["intent_requirements"] = {"schema": "supervisor-local-intent-requirement-artifact@1",
        "contract_json": json.dumps(contract, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False),
        "contract_cid": cid_for_dag_json(contract)}
    if model_only:
        logic = json.loads(LOGIC.read_bytes())
        evidence = PromptEvidenceRecord(evidence_key="actual-header-model-only",
            source_kind="checked_local_header_model_only",
            artifact_cid="sha256:" + logic["qualification_sha256"],
            summary="Checked conditional header model and vulnerability witnesses; runtime behavior remains unresolved",
            repository_paths=("bottle.py",), claim_keys=(), authority=EvidenceAuthority.SCAN_ADVISORY,
            status=RecordStatus.PROPOSED, provenance={"qualification_sha256": logic["qualification_sha256"],
                "source_sha256": logic["source_sha256"],
                "scope": "conditional_Boolean_header_model",
                "source_semantics_verified": False, "whole_program_proved": False,
                "proof_authority": False, "completion_authority": False})
        payload["planning_inputs"]["selected_evidence"].append(evidence.to_dict())
    return {"payload": payload}


def test_exact_public_request_is_completely_accounted_for_but_unresolved(control):
    public = control["public_request"]
    assert public["text"].encode() == INSTRUCTION.read_bytes()
    assert public["source_bytes"] == 3873
    assert public["source_sha256"] == api.PUBLIC_INSTRUCTION_SHA256
    assert public["status"] == "unresolved"
    ledger = validate_intent_requirement_ledger(public["ledger"], source_text=public["text"])
    assert ledger["requirements"] == []
    assert ledger["source_accounting_complete"] is True
    assert ledger["semantic_support_complete"] is False
    assert len(ledger["source_units"]) == 1
    assert ledger["source_units"][0]["disposition"] == "unsupported"
    assert ledger["source_units"][0]["text"] == public["text"]
    assert control["current_behavioral_facts"] == control["behavioral_satisfied_requirements"] == []
    assert all(control[key] is False for key in api._AUTHORITY)
    assert control["provider_calls"] == control["training_steps"] == control["solver_calls"] == 0


def test_authored_native_atom_has_distinct_source_and_no_public_decomposition_claim(control):
    authored = control["authored_control"]
    assert authored["text"] == "agent must repair bottle."
    assert authored["source_path"] != control["public_request"]["source_path"]
    assert authored["source_sha256"] != control["public_request"]["source_sha256"]
    assert authored["covers_complete_public_request"] is False
    assert authored["semantic_alignment_to_public_request_verified"] is False
    document = authored["native_document"]
    assert document["sources"][0]["source_uri"].startswith("authored-development-control:")
    assert document["sources"][0]["review_status"] == "trusted_fixture"
    atom = document["statements"][0]
    assert (atom["predicate"], atom["arguments"], atom["modality"]) == ("repair", ["agent", "bottle"], "required")
    requirement = authored["ledger"]["requirements"][0]
    assert requirement["semantic_support"] == "candidate"
    assert requirement["interpretation_status"] == "reviewed_candidate"
    assert validate_intent_requirement_contract(authored["contract"], source_text=authored["text"]) == authored["contract"]


def test_native_symbolic_adapter_refuses_unresolved_public_ledger(control):
    ledger = control["public_request"]["ledger"]
    operations = {"schema": "intent-symbolic-operation-contract@1", "ledger_sha256": ledger["ledger_sha256"],
        "review_ref": "test:unresolved-public-request", "interpretation_scope": "administrative_requirement_task_coverage",
        "operations": [], "semantic_alignment_verified": False, "proof_authority": False,
        "execution_authority": False, "completion_authority": False}
    with pytest.raises(IntentRequirementAdapterError, match="unresolved source units"):
        validate_symbolic_operations(operations, ledger=ledger, requirements=[])


def test_authored_atom_cannot_be_rebound_to_complete_public_instruction(control):
    with pytest.raises(IntentPlanCoverageError, match="source report is bound to a different original source"):
        validate_intent_requirement_contract(control["authored_control"]["contract"],
                                            source_text=control["public_request"]["text"])


def test_pure_native_symbolic_control_selects_declared_outputs_with_zero_behavioral_facts(control):
    declaration = _unsigned_declaration(control)
    assert set(declaration) == {"payload"}  # No historical signature is reused.
    contract = control["authored_control"]["contract"]
    materials = build_intent_planning_materials(contract, manifest=declaration)
    assert materials.current_facts == ()
    assert materials.to_dict()["current_facts"] == []
    assert materials.to_dict()["observed_source_facts"] is False
    proposed = build_intent_symbolic_plan(contract, manifest=declaration)
    assert proposed["coverage"]["accepted"]
    assert proposed["receipt"]["observed_facts_supplied"] == proposed["receipt"]["provider_calls"] == 0
    assert proposed["receipt"]["interpretation_scope"] == "administrative_requirement_task_coverage"
    assert all(proposed["receipt"][key] is False for key in (
        "semantic_alignment_verified", "source_semantics_verified", "proof_authority", "execution_authority", "completion_authority"))
    assert len(proposed["graph"].tasks) == 1
    assert proposed["graph"].tasks[0].task_key == api.TASK_KEY
    assert {(row.path, row.effect) for row in proposed["graph"].tasks[0].outputs} == {
        ("bottle.py", "modify"), ("report.jsonl", "create")}
    assert [row.validation_key for row in proposed["graph"].tasks[0].validations] == [api.VALIDATION_KEY]


def test_exact_public_coverage_still_refuses_the_authored_control_graph(control):
    proposal = build_intent_symbolic_plan(control["authored_control"]["contract"],
                                          manifest=_unsigned_declaration(control))
    coverage = check_intent_plan_coverage(control["public_request"]["contract"],
                                          graph=proposal["graph"], bindings=[])
    assert coverage["accepted"] is False
    assert "unsupported_source_units" in {row["code"] for row in coverage["errors"]}
    assert coverage["semantic_support_complete"] is False


def test_actual_checked_model_hints_do_not_discharge_behavior_or_change_task_selection(control):
    contract = control["authored_control"]["contract"]
    baseline = build_intent_symbolic_plan(contract, manifest=_unsigned_declaration(control))
    hinted_declaration = _unsigned_declaration(control, model_only=True)
    materials = build_intent_planning_materials(contract, manifest=hinted_declaration)
    assert materials.current_facts == ()
    hinted = build_intent_symbolic_plan(contract, manifest=hinted_declaration)
    assert hinted["receipt"]["observed_facts_supplied"] == 0
    assert hinted["receipt"]["schedule"] == baseline["receipt"]["schedule"]
    hints = [row for row in hinted["graph"].evidence if row.evidence_key == "actual-header-model-only"]
    assert len(hints) == 1
    assert hints[0].authority is EvidenceAuthority.SCAN_ADVISORY
    assert hints[0].claim_keys == ()
    assert hints[0].provenance["source_semantics_verified"] is False
    assert hinted["coverage"]["semantic_support_complete"] is False


@pytest.mark.parametrize("change", ["append", "replace"])
def test_public_source_drift_is_refused_before_building_ledgers(change):
    raw = INSTRUCTION.read_bytes()
    altered = raw + b"\n" if change == "append" else raw.replace(b"CWE", b"AAA", 1)
    with pytest.raises(ValueError, match="exact independently captured public Bottle instruction"):
        api.build_terminal_intent_control(public_instruction_bytes=altered)


@pytest.mark.parametrize("path", ["../instruction.md", "/instruction.md", api.AUTHORED_CONTROL_PATH])
def test_public_and_authored_source_identities_cannot_collide_or_escape(path):
    with pytest.raises(ValueError, match="distinct canonical"):
        api.build_terminal_intent_control(public_instruction_bytes=INSTRUCTION.read_bytes(), public_instruction_path=path)


def test_control_identity_is_stable_and_contains_no_hidden_authority(control):
    again = api.build_terminal_intent_control(public_instruction_bytes=INSTRUCTION.read_bytes())
    assert again == control
    value = deepcopy(control)
    identifier = value.pop("control_sha256")
    assert identifier == hashlib.sha256(api._wire(value)).hexdigest()
    assert control["canonical_state_mutated"] is False
    assert control["public_request_fully_interpreted"] is False
    assert control["official_reward"] is None
