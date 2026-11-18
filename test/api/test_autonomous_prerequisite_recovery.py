"""Isolated proposal contracts; no live task admission or runtime execution."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SOURCE = (Path(__file__).resolve().parents[2] / "ipfs_accelerate_py" /
          "agent_supervisor" / "rescue" / "gate_recovery.py")
SCHEMA = "ipfs_accelerate_py/agent-supervisor/prerequisite-recovery-plan@1"
TASK = "TEST-060"
AUTHORITY_FIELDS = (
    "completion_authority", "execution_authority", "admission_granted",
    "operator_hold_removal_allowed", "budget_reset_allowed",
)


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("_isolated_prerequisite_gate", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def record():
    return {
        "fingerprint": "f" * 64, "source_contract": "c" * 64,
        "unchanged_probes": 3, "source_refusals": 7,
        "first_seen_ms": 1000, "last_seen_ms": 3000,
        "next_retry_at_ms": 9000, "repair_attempted": True,
        "hard_failure_attempts": 2, "max_hard_failure_attempts": 3,
    }


def requirements():
    return [
        {"code": "MISSING_OWNER_POINTER", "subject": "metadata-v10"},
        {"code": "MISSING_RUNTIME_WITNESS", "subject": "torch-cpu",
         "expected_sha256": "a" * 64},
    ]


def evidence(gaps):
    return {
        "task_id": TASK, "tasks": {TASK: {"status": "pending"}},
        "dependencies": [], "pending_dependents": [],
        "allowed_source_paths": ["source.py"],
        "artifacts": {"readiness.json": {
            "schema": "fixture/prerequisite-readiness@1", "decision": "NO_GO",
            "generated_at": "original-clock", "gaps": deepcopy(gaps),
        }},
    }


def packet(gate, tmp_path, record, data):
    ledger = gate.GateRecovery(tmp_path / "attempts.json")
    path = ledger.packet(TASK, record, data)
    return json.loads(path.read_text(encoding="utf-8"))


def plan_from(packet_value):
    plan = packet_value["prerequisite_recovery"]
    assert plan["schema"] == SCHEMA
    assert plan["parent_task_id"] == TASK
    for key in AUTHORITY_FIELDS:
        assert plan[key] is False, key
    return plan


def ids_by_kind(plan):
    return {step["kind"]: step["task_id"] for step in plan["steps"]}


def test_missing_owner_and_witness_form_parallel_proposal_chains(gate, tmp_path, record):
    value = packet(gate, tmp_path, record, evidence(requirements()))
    plan = plan_from(value)
    assert plan["status"] == "recovery_required"
    assert plan["external_requirements"] == []
    by_kind = {step["kind"]: step for step in plan["steps"]}
    assert set(by_kind) == {
        "discover_registered_preparer", "verify_preparer_identity",
        "locate_approved_runtime_witness", "verify_runtime_identity",
    }
    ids = ids_by_kind(plan)
    assert len(set(ids.values())) == 4 and TASK not in ids.values()
    assert by_kind["discover_registered_preparer"]["depends_on"] == []
    assert by_kind["locate_approved_runtime_witness"]["depends_on"] == []
    assert by_kind["verify_preparer_identity"]["depends_on"] == [ids["discover_registered_preparer"]]
    assert by_kind["verify_runtime_identity"]["depends_on"] == [ids["locate_approved_runtime_witness"]]
    for kind, step in by_kind.items():
        assert TASK not in step["depends_on"]
        assert step["requires_admission"] is True
        assert step["purpose"] == (
            "metadata_validation" if kind.startswith("verify_") else "control_metadata_analysis"
        )
        for key in AUTHORITY_FIELDS:
            assert step.get(key, False) is False
    assert {ids["verify_preparer_identity"], ids["verify_runtime_identity"]} <= set(plan["recheck_after"])
    assert set(plan["recheck_after"]) <= set(ids.values())
    assert value["completion_authority"] is value["work_complete"] is False


def test_clock_order_and_duplicates_keep_the_same_proposal_ids(gate, tmp_path, record):
    original = evidence(requirements())
    first = plan_from(packet(gate, tmp_path, record, original))
    later_record = {**record, "first_seen_ms": 2000, "last_seen_ms": 9999,
                    "next_retry_at_ms": 99999, "unchanged_probes": 9}
    changed = evidence(list(reversed(requirements())) + [requirements()[0]])
    changed["artifacts"]["readiness.json"]["generated_at"] = "later-clock"
    second = plan_from(packet(gate, tmp_path, later_record, changed))
    assert len(second["steps"]) == 4
    assert ids_by_kind(first) == ids_by_kind(second)
    assert first["recheck_after"] == second["recheck_after"]


def test_expected_digest_changes_runtime_proposal_ids(gate, tmp_path, record):
    old = plan_from(packet(gate, tmp_path, record, evidence(requirements())))
    changed = requirements()
    changed[1]["expected_sha256"] = "b" * 64
    new = plan_from(packet(gate, tmp_path, record, evidence(changed)))
    for kind in ("locate_approved_runtime_witness", "verify_runtime_identity"):
        assert ids_by_kind(old)[kind] != ids_by_kind(new)[kind]


@pytest.mark.parametrize("code", [
    "MISSING_EXECUTION_AUTHORITY", "OPERATOR_HOLD", "LOOKUP_BUDGET_EXHAUSTED",
])
def test_external_requirements_never_become_runnable_proposals(gate, tmp_path, record, code):
    plan = plan_from(packet(gate, tmp_path, record, evidence([{"code": code, "subject": "scope"}])))
    assert plan["status"] == "external_dependency"
    assert plan["steps"] == [] and plan["recheck_after"] == []
    assert any(item["code"] == code for item in plan["external_requirements"])


def test_legacy_untyped_packet_keeps_its_original_shape(gate, tmp_path, record):
    data = evidence([])
    data["artifacts"]["readiness.json"] = {"decision": "NO_GO", "reasons": ["missing ordinary evidence"]}
    result = packet(gate, tmp_path, record, data)
    assert "prerequisite_recovery" not in result
    assert result["schema"] == "ipfs_accelerate_py/agent-supervisor/gate-recovery@1"
    assert result["body_free"] is True and result["completion_authority"] is False


def test_seventeen_unique_requirements_refuse_without_partial_proposals(gate, tmp_path, record):
    gaps = [{"code": "MISSING_OWNER_POINTER", "subject": f"metadata-{number}"}
            for number in range(17)]
    plan = plan_from(packet(gate, tmp_path, record, evidence(gaps)))
    assert plan["status"] == "external_dependency"
    assert plan["steps"] == [] and plan["recheck_after"] == []
    assert plan["external_requirements"]


@pytest.mark.parametrize("gaps", [
    {"code": "MISSING_OWNER_POINTER", "subject": "metadata-v10"},
    ["UNTRUSTED_PROSE_SENTINEL"],
    [{"code": "MISSING_RUNTIME_WITNESS", "subject": "torch-cpu", "expected_sha256": True}],
    [{"code": "MISSING_OWNER_POINTER", "subject": "UNTRUSTED_PROSE_SENTINEL" + "z" * 10000}],
    [{"code": "MISSING_OWNER_POINTER", "subject": "metadata-v10",
      "command": "UNTRUSTED_COMMAND_SENTINEL", "reason": "UNTRUSTED_PROSE_SENTINEL"}],
])
def test_malformed_diagnostics_refuse_without_copying_instructions(gate, tmp_path, record, gaps):
    plan = plan_from(packet(gate, tmp_path, record, evidence(gaps)))
    assert plan["status"] == "external_dependency"
    assert plan["steps"] == [] and plan["recheck_after"] == []
    assert plan["external_requirements"]
    assert all(isinstance(item, dict) and item.get("code") for item in plan["external_requirements"])
    rendered = json.dumps(plan)
    assert "UNTRUSTED_COMMAND_SENTINEL" not in rendered
    assert "UNTRUSTED_PROSE_SENTINEL" not in rendered


def test_packet_generation_preserves_records_refusals_and_budgets(gate, tmp_path, record):
    ledger = gate.GateRecovery(tmp_path / "attempts.json")
    ledger.store(TASK, record)
    ledger.store("TEST-OTHER", {"source_refusals": 12, "hard_failure_attempts": 3})
    before_records = deepcopy(ledger.records)
    before_disk = ledger.path.read_bytes()
    original_record = deepcopy(record)
    data = evidence(requirements())
    original_evidence = deepcopy(data)
    result = json.loads(ledger.packet(TASK, record, data).read_text(encoding="utf-8"))
    assert plan_from(result)["status"] == "recovery_required"
    assert ledger.records == before_records
    assert ledger.path.read_bytes() == before_disk
    assert record == original_record and data == original_evidence
    assert result["source_refusals"] == 7 and result["unchanged_probes"] == 3
    assert result["source_contract"] == record["source_contract"]
    assert ledger.records[TASK]["hard_failure_attempts"] == 2
    assert ledger.records["TEST-OTHER"]["hard_failure_attempts"] == 3


def test_declared_requirements_are_retained_and_semantically_fingerprinted(gate, tmp_path, record):
    (tmp_path / "gate.py").write_text("# fictional gate\n", encoding="utf-8")
    declared = requirements()
    declared[0]["observed_at_ms"] = 1000
    blocks = {TASK: {"status": "pending", "validation": "python3 gate.py",
                     "prerequisite_requirements": json.dumps(declared)}}
    kwargs = {"repo_root": tmp_path, "todo_path": tmp_path / "board.md", "log_path": tmp_path / "lane.log"}
    first_fingerprint, first = gate.gate_evidence(TASK, blocks, **kwargs)
    assert "prerequisite_requirements" in first["tasks"][TASK]
    retained = first["tasks"][TASK]["prerequisite_requirements"]
    assert isinstance(retained, str)
    assert json.loads(retained)[0]["code"] == "MISSING_OWNER_POINTER"
    declared[0]["observed_at_ms"] = 9000
    blocks[TASK]["prerequisite_requirements"] = json.dumps(declared)
    clock_fingerprint, clock_evidence = gate.gate_evidence(TASK, blocks, **kwargs)
    assert clock_fingerprint == first_fingerprint
    assert plan_from(packet(gate, tmp_path, record, clock_evidence))["status"] == "recovery_required"
    declared[1]["expected_sha256"] = "b" * 64
    blocks[TASK]["prerequisite_requirements"] = json.dumps(declared)
    changed_fingerprint, _ = gate.gate_evidence(TASK, blocks, **kwargs)
    assert changed_fingerprint != first_fingerprint


@pytest.mark.parametrize("include_known", [False, True])
def test_unknown_typed_artifact_gap_remains_an_external_requirement(gate, tmp_path, record, include_known):
    gaps = [{"code": "MISSING_UNREGISTERED_CAPABILITY", "subject": "unregistered-metadata"}]
    if include_known:
        gaps.extend(requirements())
    plan = plan_from(packet(gate, tmp_path, record, evidence(gaps)))
    assert plan["status"] == "external_dependency"
    assert plan["steps"] == [] and plan["recheck_after"] == []
    assert any(item["code"] == "MISSING_UNREGISTERED_CAPABILITY"
               and item["subject"] == "unregistered-metadata"
               for item in plan["external_requirements"])


@pytest.mark.parametrize("code", [
    "MISSING_EXECUTION_AUTHORITY", "OPERATOR_HOLD", "LOOKUP_BUDGET_EXHAUSTED",
])
def test_external_blocker_dominates_mixed_recoverable_requirements(gate, tmp_path, record, code):
    gaps = requirements() + [{"code": code, "subject": "selected-unit"}]
    plan = plan_from(packet(gate, tmp_path, record, evidence(gaps)))
    assert plan["status"] == "external_dependency"
    assert plan["steps"] == [] and plan["recheck_after"] == []
    assert any(item["code"] == code for item in plan["external_requirements"])
    assert {item["code"] for item in plan["requirements"]} == {
        "MISSING_OWNER_POINTER", "MISSING_RUNTIME_WITNESS", code,
    }


def rejected_declaration():
    return json.dumps([{
        "code": "MISSING_OWNER_POINTER", "subject": "metadata-v10",
        "command": "UNTRUSTED_COMMAND_SENTINEL", "reason": "UNTRUSTED_PROSE_SENTINEL",
    }])


def assert_rejected_declaration_is_safe(value, raw):
    retained = value["tasks"][TASK]
    assert json.loads(retained["prerequisite_requirements"]) == [{
        "code": "INVALID_PREREQUISITE_DIAGNOSTICS", "subject": TASK,
    }]
    assert retained["prerequisite_diagnostics_sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    rendered = json.dumps(value)
    assert "UNTRUSTED_COMMAND_SENTINEL" not in rendered
    assert "UNTRUSTED_PROSE_SENTINEL" not in rendered


def test_rejected_declaration_is_redacted_before_evidence_and_packet(gate, tmp_path, record):
    raw = rejected_declaration()
    blocks = {TASK: {"status": "pending", "prerequisite_requirements": raw}}
    kwargs = {"repo_root": tmp_path, "todo_path": tmp_path / "board.md", "log_path": tmp_path / "lane.log"}
    fingerprint, data = gate.gate_evidence(TASK, blocks, **kwargs)
    assert_rejected_declaration_is_safe(data, raw)
    result = packet(gate, tmp_path, record, data)
    assert_rejected_declaration_is_safe(result, raw)
    plan = plan_from(result)
    assert plan["status"] == "external_dependency" and plan["steps"] == []
    blocks[TASK]["prerequisite_requirements"] = raw.replace("UNTRUSTED_COMMAND_SENTINEL", "DIFFERENT_COMMAND_SENTINEL")
    changed, _ = gate.gate_evidence(TASK, blocks, **kwargs)
    assert changed != fingerprint, "safe retained digest must still distinguish changed rejected input"


def test_direct_packet_also_redacts_rejected_declaration(gate, tmp_path, record):
    raw = rejected_declaration()
    data = evidence([])
    data["artifacts"] = {}
    data["tasks"][TASK]["prerequisite_requirements"] = raw
    original = deepcopy(data)
    result = packet(gate, tmp_path, record, data)
    assert_rejected_declaration_is_safe(result, raw)
    assert data == original
    assert plan_from(result)["status"] == "external_dependency"


def test_requirement_diagnostics_do_not_reset_source_refusal_budget(gate, tmp_path):
    source = tmp_path / "gate.py"
    source.write_text("# unchanged fictional source\n", encoding="utf-8")
    declared = requirements()
    blocks = {TASK: {"status": "pending", "validation": "python3 gate.py",
                     "prerequisite_requirements": json.dumps(declared)}}
    kwargs = {"repo_root": tmp_path, "todo_path": tmp_path / "board.md", "log_path": tmp_path / "lane.log"}
    fingerprint, data = gate.gate_evidence(TASK, blocks, **kwargs)
    contract = gate.gate_source_contract(data, "unchanged command and acceptance")
    ledger = gate.GateRecovery(tmp_path / "attempts.json")
    for now in (1000, 2000, 3000):
        seen = ledger.observe(TASK, fingerprint, now_ms=now, source_contract=contract)
    assert seen["source_refusals"] == 3
    declared[1]["expected_sha256"] = "b" * 64
    blocks[TASK]["prerequisite_requirements"] = json.dumps(declared)
    changed_fingerprint, changed = gate.gate_evidence(TASK, blocks, **kwargs)
    assert changed_fingerprint != fingerprint
    assert gate.gate_source_contract(changed, "unchanged command and acceptance") == contract
    seen = ledger.observe(TASK, changed_fingerprint, now_ms=4000, source_contract=contract)
    assert seen["source_refusals"] == 4
    blocks[TASK]["prerequisite_requirements"] = rejected_declaration()
    rejected_fingerprint, rejected = gate.gate_evidence(TASK, blocks, **kwargs)
    assert gate.gate_source_contract(rejected, "unchanged command and acceptance") == contract
    seen = ledger.observe(TASK, rejected_fingerprint, now_ms=5000, source_contract=contract)
    assert seen["source_refusals"] == 5
    source.write_text("# genuine fictional implementation change\n", encoding="utf-8")
    _, actual_source_change = gate.gate_evidence(TASK, blocks, **kwargs)
    assert gate.gate_source_contract(actual_source_change, "unchanged command and acceptance") != contract


def test_deep_declared_json_refuses_explicitly_within_byte_bound(gate, tmp_path, record):
    raw = "[" * 2000 + "]" * 2000
    assert len(raw.encode("utf-8")) < 32768
    blocks = {TASK: {"status": "pending", "prerequisite_requirements": raw}}
    kwargs = {"repo_root": tmp_path, "todo_path": tmp_path / "board.md", "log_path": tmp_path / "lane.log"}
    _, data = gate.gate_evidence(TASK, blocks, **kwargs)
    assert_rejected_declaration_is_safe(data, raw)
    result = packet(gate, tmp_path, record, data)
    assert plan_from(result)["status"] == "external_dependency"
    direct = evidence([])
    direct["artifacts"] = {}
    direct["tasks"][TASK]["prerequisite_requirements"] = raw
    result = packet(gate, tmp_path, record, direct)
    assert_rejected_declaration_is_safe(result, raw)
    plan = plan_from(result)
    assert plan["status"] == "external_dependency"
    assert plan["steps"] == [] and plan["recheck_after"] == []


@pytest.mark.parametrize("raw", [False, []])
def test_present_falsy_nonstring_declaration_cannot_silently_disappear(gate, tmp_path, record, raw):
    blocks = {TASK: {"status": "pending", "prerequisite_requirements": deepcopy(raw)}}
    kwargs = {"repo_root": tmp_path, "todo_path": tmp_path / "board.md", "log_path": tmp_path / "lane.log"}
    _, data = gate.gate_evidence(TASK, blocks, **kwargs)
    marker = [{"code": "INVALID_PREREQUISITE_DIAGNOSTICS", "subject": TASK}]
    retained = data["tasks"][TASK]
    assert json.loads(retained["prerequisite_requirements"]) == marker
    digest = retained["prerequisite_diagnostics_sha256"]
    assert isinstance(digest, str) and len(digest) == 64 and set(digest) <= set("0123456789abcdef")
    direct = evidence([])
    direct["artifacts"] = {}
    direct["tasks"][TASK]["prerequisite_requirements"] = deepcopy(raw)
    result = packet(gate, tmp_path, record, direct)
    assert json.loads(result["tasks"][TASK]["prerequisite_requirements"]) == marker
    assert result["tasks"][TASK]["prerequisite_diagnostics_sha256"] == digest
    plan = plan_from(result)
    assert plan["status"] == "external_dependency"
    assert plan["steps"] == [] and plan["recheck_after"] == []


@pytest.mark.parametrize("entrypoint", ["gate_evidence", "direct_packet"])
@pytest.mark.parametrize("codepoint,escaped", [
    pytest.param(0xD800, False, id="literal-high-surrogate"),
    pytest.param(0xDC00, False, id="literal-low-surrogate"),
    pytest.param(0xD800, True, id="escaped-high-surrogate-subject"),
    pytest.param(0xDC00, True, id="escaped-low-surrogate-subject"),
])
def test_rejected_surrogate_declarations_are_redacted_without_mutating_budgets(
    gate, tmp_path, record, entrypoint, codepoint, escaped,
):
    sentinel = "UNICODE_REJECTION_SENTINEL"
    raw = json.dumps([{
        "code": "MISSING_OWNER_POINTER", "subject": sentinel + chr(codepoint),
    }], ensure_ascii=escaped)
    # This identifies rejected Python text losslessly. It is not the digest of
    # an admitted UTF8 artifact; prerequisite admission must remain strict.
    expected_digest = hashlib.sha256(raw.encode("utf-8", errors="surrogatepass")).hexdigest()
    ledger = gate.GateRecovery(tmp_path / "attempts.json")
    ledger.store(TASK, record)
    ledger.store("TEST-OTHER", {"source_refusals": 12, "hard_failure_attempts": 3})
    before_records = deepcopy(ledger.records)
    before_disk = ledger.path.read_bytes()
    original_record = deepcopy(record)
    if entrypoint == "gate_evidence":
        blocks = {TASK: {"status": "pending", "prerequisite_requirements": raw}}
        original_blocks = deepcopy(blocks)
        _, data = gate.gate_evidence(
            TASK, blocks, repo_root=tmp_path,
            todo_path=tmp_path / "board.md", log_path=tmp_path / "lane.log",
        )
        assert blocks == original_blocks
        retained = data["tasks"][TASK]
        assert retained["prerequisite_diagnostics_sha256"] == expected_digest
        assert json.loads(retained["prerequisite_requirements"]) == [{
            "code": "INVALID_PREREQUISITE_DIAGNOSTICS", "subject": TASK,
        }]
        rendered_evidence = json.dumps(data, ensure_ascii=False).encode("utf-8")
        assert sentinel.encode("ascii") not in rendered_evidence
    else:
        data = evidence([])
        data["artifacts"] = {}
        data["tasks"][TASK]["prerequisite_requirements"] = raw
    original_data = deepcopy(data)
    path = ledger.packet(TASK, record, data)
    rendered = path.read_bytes().decode("utf-8")
    value = json.loads(rendered)
    retained = value["tasks"][TASK]
    assert json.loads(retained["prerequisite_requirements"]) == [{
        "code": "INVALID_PREREQUISITE_DIAGNOSTICS", "subject": TASK,
    }]
    assert retained["prerequisite_diagnostics_sha256"] == expected_digest
    assert sentinel not in rendered
    json.dumps(value, ensure_ascii=False).encode("utf-8")
    plan = plan_from(value)
    assert plan["status"] == "external_dependency"
    assert plan["steps"] == plan["recheck_after"] == []
    assert any(item["code"] == "INVALID_PREREQUISITE_DIAGNOSTICS"
               for item in plan["external_requirements"])
    assert data == original_data and record == original_record
    assert ledger.records == before_records and ledger.path.read_bytes() == before_disk


def _oversized_ordinary_artifacts(shape):
    if shape == "65_untyped_gaps":
        return {"ordinary.json": {
            "schema": "fixture/ordinary-report@1", "decision": "NO_GO",
            "gaps": [{"message": f"ordinary observation {number}"} for number in range(65)],
        }}
    assert shape == "257_untyped_artifacts"
    return {f"ordinary-{number}.json": {
        "schema": "fixture/ordinary-report@1", "decision": "NO_GO",
        "reasons": [f"ordinary observation {number}"],
    } for number in range(257)}


@pytest.mark.parametrize("shape", ["65_untyped_gaps", "257_untyped_artifacts"])
def test_large_ordinary_diagnostics_do_not_opt_into_prerequisite_recovery(gate, tmp_path, record, shape):
    data = evidence([])
    data["artifacts"] = _oversized_ordinary_artifacts(shape)
    original_data = deepcopy(data)
    ledger = gate.GateRecovery(tmp_path / "attempts.json")
    ledger.store(TASK, record)
    ledger.store("TEST-OTHER", {"source_refusals": 12, "hard_failure_attempts": 3})
    before_records = deepcopy(ledger.records)
    before_disk = ledger.path.read_bytes()
    assert gate.prerequisite_recovery_plan(TASK, data) is None
    value = json.loads(ledger.packet(TASK, record, data).read_text(encoding="utf-8"))
    assert "prerequisite_recovery" not in value
    assert ledger.records == before_records and ledger.path.read_bytes() == before_disk
    assert data == original_data


@pytest.mark.parametrize("shape", ["65_typed_gaps", "257_typed_artifacts"])
def test_explicit_typed_diagnostics_still_enforce_raw_input_bounds(gate, tmp_path, record, shape):
    data = evidence([])
    if shape == "65_typed_gaps":
        data["artifacts"]["readiness.json"]["gaps"] = [deepcopy(requirements()[0]) for _ in range(65)]
    else:
        data["artifacts"] = {f"typed-{number}.json": {
            "schema": "fixture/prerequisite-readiness@1", "decision": "NO_GO",
            "gaps": [deepcopy(requirements()[0])] if number == 0 else [],
        } for number in range(257)}
    plan = plan_from(packet(gate, tmp_path, record, data))
    assert plan["status"] == "external_dependency"
    assert plan["steps"] == plan["recheck_after"] == []
    assert any(item["code"] == "INVALID_PREREQUISITE_DIAGNOSTICS"
               for item in plan["external_requirements"])


@pytest.mark.parametrize("shape", ["65_untyped_gaps", "257_untyped_artifacts"])
def test_invalid_declaration_is_not_hidden_by_large_ordinary_diagnostics(gate, tmp_path, record, shape):
    raw = rejected_declaration()
    data = evidence([])
    data["artifacts"] = _oversized_ordinary_artifacts(shape)
    data["tasks"][TASK]["prerequisite_requirements"] = raw
    value = packet(gate, tmp_path, record, data)
    assert_rejected_declaration_is_safe(value, raw)
    plan = plan_from(value)
    assert plan["status"] == "external_dependency"
    assert plan["steps"] == plan["recheck_after"] == []
    assert any(item["code"] == "INVALID_PREREQUISITE_DIAGNOSTICS"
               for item in plan["external_requirements"])


@pytest.mark.parametrize("origin", ["declared", "typed_artifact"])
@pytest.mark.parametrize("shape", ["65_untyped_gaps", "257_untyped_artifacts"])
def test_valid_typed_requirements_coexist_with_large_ordinary_diagnostics(
    gate, tmp_path, record, shape, origin,
):
    data = evidence([])
    data["artifacts"] = _oversized_ordinary_artifacts(shape)
    if origin == "declared":
        data["tasks"][TASK]["prerequisite_requirements"] = json.dumps(requirements())
    else:
        data["artifacts"]["typed-readiness.json"] = {
            "schema": "fixture/prerequisite-readiness@1", "decision": "NO_GO", "gaps": requirements(),
        }
    plan = plan_from(packet(gate, tmp_path, record, data))
    assert plan["status"] == "recovery_required"
    assert plan["external_requirements"] == []
    assert len(plan["steps"]) == 4
    assert {step["kind"] for step in plan["steps"]} == {
        "discover_registered_preparer", "verify_preparer_identity",
        "locate_approved_runtime_witness", "verify_runtime_identity",
    }
