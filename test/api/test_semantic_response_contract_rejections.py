"""Closed refusal metadata preserves exact response checks and exception custody."""
from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_dialogue as dialogue
from test.api.test_semantic_decision_dialogue import envelope, reply, run
from test.api.test_semantic_retrieval_query_dialogue import run as run_retrieval
from test.api.test_semantic_retrieval_query_reference import build_retrieval_native_fixture

pytest_plugins = ("test.api.test_semantic_decision_dialogue",)
PRIVATE = "PRIVATE-RESPONSE-CONTENT-AND-UNKNOWN-FIELD"


def call():
    return dialogue.SemanticDecisionCall("call", "session", 0, "b" * 64,
        "codex_cli", "fixture-model", "prompt", 60, 4096, "p" * 64)


def serialized(value):
    return json.dumps(value, ensure_ascii=False)


def rejected(text, expected_code, expected_class=ValueError, expected_message=None):
    attempt = {}
    with pytest.raises(expected_class) as caught:
        dialogue._observe_rejection(attempt, "response_contract_validation",
            lambda: dialogue._response(text, call()))
    assert type(caught.value) is expected_class
    if expected_message is not None:
        assert str(caught.value) == expected_message
    assert attempt["rejection_observation"] == {
        "schema":"supervisor-semantic-decision-rejection-observation@1",
        "phase":"response_contract_validation", "reason_code":"response_contract_refused",
        "exception_type":expected_class.__name__,
    }
    assert attempt["response_contract_rejection_observation"] == {
        "schema":"supervisor-semantic-response-contract-rejection@1", "reason_code":expected_code,
    }
    assert PRIVATE not in serialized(attempt)
    return caught.value


@pytest.mark.parametrize("text", [None, "", "  \n", PRIVATE * 3000])
def test_bounded_text_refusal_preserves_original_exception(text):
    rejected(text, "bounded_response_text_refused", expected_message="bounded nonempty dialogue response required")


def test_json_and_duplicate_keys_remain_refused_without_parser_document():
    error = rejected(PRIVATE, "invalid_response_json", json.JSONDecodeError)
    assert error.doc == PRIVATE
    rejected('{"schema":0,"schema":"' + PRIVATE + '"}', "duplicate_response_key",
        expected_message="duplicate dialogue response key")


@pytest.mark.parametrize("text,code,message", [
    ('{"text":NaN}', "response_structure_refused", "projection JSON number must be finite"),
    ('[' * 42 + '0' + ']' * 42, "response_structure_refused", "projection structure exceeds bounded depth or population"),
    ('["' + PRIVATE + '"]', "response_object_required", "exact dialogue response object required"),
])
def test_structure_and_object_checks_keep_their_original_order(text, code, message):
    rejected(text, code, expected_message=message)


@pytest.mark.parametrize("field,value,reason", [
    ("schema", PRIVATE, "unknown_response_schema"),
    ("schema", [], "unknown_response_schema"),
    ("extra", PRIVATE, "response_field_set_mismatch"),
    (PRIVATE, 0, "response_field_set_mismatch"),
    ("session_id", PRIVATE, "response_session_mismatch"),
    ("generation", True, "response_generation_mismatch"),
    ("generation", 1, "response_generation_mismatch"),
    ("binding_sha256", PRIVATE, "response_binding_mismatch"),
])
def test_exact_fields_and_round_bindings_are_required(field, value, reason):
    response = envelope(call(), text=PRIVATE)
    response[field] = value
    rejected(serialized(response), reason,
        expected_message="dialogue response fields or current round binding differ")


@pytest.mark.parametrize("value", [None, "", " \n", [PRIVATE]])
def test_nonempty_literal_final_text_still_required(value):
    rejected(serialized(envelope(call(), text=value)), "invalid_final_response_text",
        expected_message="nonempty literal final response required")


@pytest.mark.parametrize("value", [None, [], [PRIVATE] * 2, [PRIVATE] * 5, [0]])
def test_requested_capsule_id_bounds_and_uniqueness_still_required(value):
    response = envelope(call(), capsule_ids=["capsule"])
    response["capsule_ids"] = value
    rejected(serialized(response), "invalid_requested_capsule_ids",
        expected_message="unique bounded requested capsule identities required")


def test_marking_does_not_replace_exception_message_or_cause(monkeypatch):
    original = ValueError(PRIVATE)
    cause = RuntimeError("private-cause")
    original.__cause__ = cause
    def refused(value, limit):
        raise original
    monkeypatch.setattr(dialogue, "_structure", refused)
    captured = rejected(serialized(envelope(call())), "response_structure_refused")
    assert captured is original and captured.args == (PRIVATE,) and captured.__cause__ is cause
    marker = captured._semantic_response_contract_rejection
    with pytest.raises(FrozenInstanceError):
        marker.reason = dialogue._ResponseContractReason.SCHEMA


def test_json_parse_marker_retains_the_exact_jsondecodeerror_and_its_cause(monkeypatch):
    original = json.JSONDecodeError("literal parser message", PRIVATE, 0)
    cause = ValueError("private-parser-cause")
    original.__cause__ = cause
    def refused(text, **kwargs):
        raise original
    monkeypatch.setattr(dialogue.json, "loads", refused)
    captured = rejected("{}", "invalid_response_json", json.JSONDecodeError)
    assert captured is original and captured.__cause__ is cause
    assert captured.msg == "literal parser message" and captured.doc == PRIVATE


@pytest.mark.parametrize("phase", ["post_return_control_check", "post_return_current_verification"])
def test_response_marker_cannot_relabel_a_different_trusted_phase(phase):
    attempt = {}
    error = dialogue._mark_response_rejection(ValueError(PRIVATE), dialogue._ResponseContractReason.BINDING)
    def refused():
        raise error
    with pytest.raises(ValueError) as caught:
        dialogue._observe_rejection(attempt, phase, refused)
    assert caught.value is error
    assert "response_contract_rejection_observation" not in attempt
    assert PRIVATE not in serialized(attempt)


def test_plain_packet_bad_binding_retains_usage_sha_and_original_generic_observation(decision_case):
    calls = []
    outputs = []
    def invoke(current):
        calls.append(current)
        value = envelope(current, text=PRIVATE)
        value["binding_sha256"] = PRIVATE
        child = reply(current, value, provider_called=True)
        outputs.append(child.text)
        return child
    with pytest.raises(dialogue.SemanticDecisionDialogueError) as caught:
        run(decision_case, invoke)
    ledger = caught.value.receipt
    attempt = ledger["attempts"][0]
    assert len(calls) == len(ledger["attempts"]) == 1
    assert ledger["failure_reason"] == "ValueError" and ledger["provider_dispatches_reported"] == 1
    assert attempt["provider_receipt_bound"] is True
    assert attempt["response_sha256"] == hashlib.sha256(outputs[0].encode()).hexdigest()
    assert attempt["response_contract_rejection_observation"]["reason_code"] == "response_binding_mismatch"
    assert PRIVATE not in serialized(ledger)


@pytest.mark.parametrize("mutation", ["source", "raw_parent"])
def test_real_retrieval_custody_refusal_precedes_malformed_response_diagnostics(tmp_path, mutation):
    case = build_retrieval_native_fixture(tmp_path)
    calls = []
    def invoke(current):
        calls.append(current)
        if mutation == "raw_parent":
            case.retrieval_artifact.write_bytes(case.retrieval_artifact.read_bytes() + b"\n")
        else:
            (case.repository / "mod.py").write_text(case.source + "\n# " + PRIVATE + "\n")
        return dialogue.SemanticDecisionReply(PRIVATE, reply(current, provider_called=True).receipt)
    with pytest.raises(dialogue.SemanticDecisionDialogueError) as caught:
        run_retrieval(case, invoke)
    ledger = caught.value.receipt
    attempt = ledger["attempts"][0]
    assert len(calls) == len(ledger["attempts"]) == 1 and attempt["provider_receipt_bound"] is True
    assert attempt["rejection_observation"]["phase"] == "post_return_current_verification"
    assert "response_contract_rejection_observation" not in attempt
    assert PRIVATE not in serialized(ledger)


def test_successful_literal_reply_has_no_rejection_metadata(decision_case):
    literal = '<b>café</b> &quot;literal&quot;  \r\n'
    result = run(decision_case, lambda current: reply(current, envelope(current, text=literal)))
    assert result.text == literal
    assert result.receipt["status"] == "returned"
    assert all("rejection_observation" not in attempt
        and "response_contract_rejection_observation" not in attempt for attempt in result.receipt["attempts"])
