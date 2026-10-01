"""Scope rejection preserves planning and precedes learned inference/replay."""
from copy import deepcopy
import hashlib
import json
import sys

import pytest

from benchmarks.agent_supervisor.container_coding.test_terminal_intent_copy import (
    DESCRIPTOR, PROBE, installed_copy_fixture,
)
from ipfs_accelerate_py.agent_supervisor.runtime import intent_autoencoder_advisor as advisor
from ipfs_datasets_py.logic.intent_ir.formalize import instruction_scope as scope


def rehash(value, key):
    value[key] = hashlib.sha256(advisor._encoded({k: v for k, v in value.items() if k != key})).hexdigest()
    return value


def legacy(advice):
    value = deepcopy(advice)
    value["schema"] = advisor.LEGACY_SCHEMA
    value.pop("scope_report")
    return rehash(value, "advice_sha256")


def assert_source_and_authority(advice, instruction):
    assert advice["instruction_sha256"] == hashlib.sha256(instruction.encode()).hexdigest()
    assert advice["instruction_bytes"] == len(instruction.encode())
    assert advice["continue_planning"] is advice["raw_instruction_preserved"] is True
    assert all(advice[key] is False for key in advisor.AUTHORITY_FIELDS)


@pytest.mark.parametrize("instruction", [
    "If checks succeed, read ledger.",
    "Update ledger unless checks fail.",
    "When compilation ends, read ledger.",
    "Read ledger and delete cache.",
    "Read ledger, then update cache.",
    "Read or delete ledger.",
    "Do it.", "Update that.",
    "Does the worker need to read ledger?",
    "Read LedgerCache from MemoryStore.",
    "<encode> read ledger.",
    "read " + "ledger " * 49,
])
def test_scope_rejects_before_model_and_keeps_descriptor_and_source(installed_copy_fixture, tmp_path, instruction):
    _, calls, _ = installed_copy_fixture
    advice = advisor.prepare_intent_advice(instruction=instruction, checkpoint_descriptor=DESCRIPTOR)
    assert advice["schema"] == advisor.SCHEMA
    assert advice["status"] == "fail_open_instruction_scope"
    assert advice["report"] is None and advice["checkpoint_descriptor"] == DESCRIPTOR
    assert advice["scope_report"]["eligible_for_inference"] is False
    assert_source_and_authority(advice, instruction)
    assert advisor.validate_intent_advice(advice, instruction=instruction) == advice
    sidecar = tmp_path / "advice.json"
    sidecar.write_text(json.dumps(advice))
    restored = advisor.load_intent_advice(path=sidecar,
        expected_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(), instruction=instruction)
    assert restored == advice
    assert advisor.intent_planner_summary(restored, instruction=instruction) == (None, advice)
    assert calls == [], "scope rejection must precede every native inference or replay"


def test_supported_advice_carries_replayed_scope_without_changing_native_report(installed_copy_fixture):
    native, calls, _ = installed_copy_fixture
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=DESCRIPTOR)
    assert advice["report"] == native
    assert advice["scope_report"] == scope.assess_intent_instruction_scope(PROBE)
    assert advice["scope_report"]["eligible_for_inference"] is True
    assert advisor.validate_intent_advice(advice, instruction=PROBE) == advice
    summary, selected = advisor.intent_planner_summary(advice, instruction=PROBE)
    assert selected == advice
    assert json.loads(summary)["instruction_scope"]["scope_acceptance_is_semantic_verification"] is False
    assert "scope_report" not in json.loads(summary)
    assert calls[0] == ("prepare", PROBE, DESCRIPTOR, None)


def test_supported_historical_envelope_still_validates_without_migration(installed_copy_fixture):
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=DESCRIPTOR)
    historical = legacy(advice)
    before = deepcopy(historical)
    assert advisor.validate_intent_advice(historical, instruction=PROBE) == before
    summary, selected = advisor.intent_planner_summary(historical, instruction=PROBE)
    assert selected == before and historical == before
    assert json.loads(summary)["instruction_scope"]["complete_consumption"] is True


@pytest.mark.parametrize("status", ["semantic_candidate_advice", "fail_open_encoder_generation"])
def test_historical_unsupported_report_cannot_bypass_scope_on_reload(installed_copy_fixture, tmp_path, status):
    report, calls, _ = installed_copy_fixture
    instruction = "Read ledger and delete cache."
    historical = advisor._base(instruction, status=status, report={**report, "status": status},
        checkpoint_descriptor=DESCRIPTOR, schema=advisor.LEGACY_SCHEMA)
    with pytest.raises(ValueError, match="single-action scope"):
        advisor.validate_intent_advice(historical, instruction=instruction)
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(historical))
    selected = advisor.load_intent_advice(path=path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), instruction=instruction)
    assert selected["status"] == "fail_open_instruction_scope"
    assert selected["checkpoint_descriptor"] == DESCRIPTOR and selected["report"] is None
    assert advisor.validate_intent_advice(selected, instruction=instruction) == selected
    assert advisor.intent_planner_summary(historical, instruction=instruction)[1] == selected
    assert_source_and_authority(selected, instruction)
    assert calls == []


@pytest.mark.parametrize("field,value", [
    ("eligible_for_inference", False), ("producer_sha256", "0" * 64),
    ("instruction_sha256", "f" * 64), ("grammar_shape", "invented_shape"),
    ("proof_authority", True), ("complete_consumption", False),
])
def test_rehashed_scope_forgery_fails_before_native_replay(installed_copy_fixture, field, value):
    _, calls, _ = installed_copy_fixture
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=DESCRIPTOR)
    calls.clear()
    advice["scope_report"][field] = value
    rehash(advice["scope_report"], "report_sha256")
    rehash(advice, "advice_sha256")
    with pytest.raises(ValueError):
        advisor.validate_intent_advice(advice, instruction=PROBE)
    summary, fallback = advisor.intent_planner_summary(advice, instruction=PROBE)
    assert summary is None and fallback["status"] == "fail_open_advice_rejected"
    assert calls == []


def test_missing_new_scope_receipt_is_not_silently_recomputed(installed_copy_fixture):
    _, calls, _ = installed_copy_fixture
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=DESCRIPTOR)
    advice["scope_report"] = None
    rehash(advice, "advice_sha256")
    calls.clear()
    with pytest.raises(ValueError, match="requires its scope"):
        advisor.validate_intent_advice(advice, instruction=PROBE)
    assert calls == []


def test_scoped_fallback_cannot_claim_rejection_for_accepted_source():
    advice = advisor._scope_fallback(PROBE, scope_report=scope.assess_intent_instruction_scope(PROBE),
        checkpoint_descriptor=DESCRIPTOR)
    with pytest.raises(ValueError, match="unsupported instruction"):
        advisor.validate_intent_advice(advice, instruction=PROBE)


def test_wrong_family_descriptor_is_not_retained_in_scope_fallback(installed_copy_fixture):
    report, calls, _ = installed_copy_fixture
    instruction = "Read ledger and delete cache."
    foreign = {**DESCRIPTOR, "schema": "intent-projection-feature-checkpoint/v1"}
    old = advisor._base(instruction, status="semantic_candidate_advice", report=report,
        checkpoint_descriptor=foreign, schema=advisor.LEGACY_SCHEMA)
    summary, selected = advisor.intent_planner_summary(old, instruction=instruction)
    assert summary is None and selected["checkpoint_descriptor"] is None
    assert selected["status"] == "fail_open_instruction_scope"
    assert advisor.validate_intent_advice(selected, instruction=instruction) == selected
    assert calls == []


def test_missing_optional_scope_module_drops_advice_and_preserves_planning(monkeypatch):
    monkeypatch.setitem(sys.modules, scope.__name__, None)
    advice = advisor.prepare_intent_advice(instruction=PROBE, checkpoint_descriptor=DESCRIPTOR)
    assert advice["status"] == "fail_open_optional_error"
    assert advice["report"] is advice["checkpoint_descriptor"] is advice["scope_report"] is None
    assert_source_and_authority(advice, PROBE)
    assert advisor.validate_intent_advice(advice, instruction=PROBE) == advice


def test_disabled_frontend_does_not_import_or_assess_scope(monkeypatch):
    monkeypatch.setitem(sys.modules, scope.__name__, None)
    advice = advisor.prepare_intent_advice(instruction="Read ledger and delete cache.",
        checkpoint_descriptor=DESCRIPTOR, enabled=False)
    assert advice["status"] == "disabled" and advice["scope_report"] is None
    assert advisor.validate_intent_advice(advice, instruction="Read ledger and delete cache.") == advice
