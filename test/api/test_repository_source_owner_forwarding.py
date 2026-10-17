"""Real stdio pipes with authored owner exchanges; no provider or native work.

These observations report bridge behavior, never independent worker delivery
or model ingestion. Local parse errors deliberately remain outside this
owner-response-derived stdout population. No observation files are created.
"""
from copy import deepcopy
import hashlib
import json
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_stdio as stdio
from test.api.test_repository_source_owner_stdio import (
    authored, descriptor, exchange, line, ping, pipes, start_bridge, finish, drain)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _flags(receipt):
    assert receipt["worker_reported"] is True
    assert all(receipt[name] is False for name in ("independently_verified", "provider_ingestion_verified",
        "token_savings_measured", "proof_authority", "execution_authority", "completion_authority",
        "raw_content_exported"))
    assert receipt["observer_provider_calls"] == 0


def test_unused_observer_has_unknown_counts_and_no_authority():
    first, second = stdio.ForwardingObservation(), stdio.ForwardingObservation()
    receipt = first.receipt()
    assert receipt["schema"] == stdio.FORWARDING_SCHEMA
    assert receipt["status"] == "not_started" and receipt["complete"] is False
    assert receipt["reason"] == "not_started" and receipt["events"] == []
    assert all(receipt[key] is None for key in ("session_id", "descriptor_sha256", "returncode",
        "owner_exchange_attempts", "owner_exchange_unconfirmed", "observed_responses"))
    assert len(receipt["bridge_id"]) == 32 and receipt["bridge_id"] != second.receipt()["bridge_id"]
    _flags(receipt)


def test_eof_observes_empty_helper_population_without_calling_owner(authored, tmp_path):
    observation = stdio.ForwardingObservation()
    outcome, replies, raw = exchange(tmp_path, b"", observation=observation)
    receipt = observation.receipt()
    assert outcome == [0] and replies == [] and raw == b"" and authored.calls == []
    assert receipt["status"] == "observed" and receipt["complete"] is True and receipt["returncode"] == 0
    assert receipt["session_id"] == authored.descriptor["session_id"] and receipt["descriptor_sha256"] == "a" * 64
    assert receipt["owner_exchange_attempts"] == receipt["owner_exchange_unconfirmed"] == receipt["observed_responses"] == 0
    assert receipt["events"] == [] and receipt["reason"] is None
    _flags(receipt)


def test_validated_response_and_stdout_have_distinct_exact_hashes(authored, tmp_path):
    seen = []
    def reply(message, value):
        value["message"]["result"] = {"source": "PRIVATE_SOURCE_MARKER"}
        seen.append(deepcopy(value))
        return value
    authored.response = reply
    observation = stdio.ForwardingObservation()
    outcome, replies, raw = exchange(tmp_path, line(ping()), observation=observation)
    receipt = observation.receipt()
    event = receipt["events"][0]
    owner_request = stdio.protocol.make_request(authored.descriptor["session_id"], ping(), request_id=seen[0]["request_id"])
    assert outcome == [0] and event == dict(exchange_ordinal=1,
        request_sha256=_sha(stdio.protocol.canonical_json(owner_request)),
        owner_response_sha256=_sha(stdio.protocol.canonical_json(seen[0])),
        owner_response_bytes=len(stdio.protocol.canonical_json(seen[0])), owner_ok=True, owner_reason_code=None,
        output_kind="owner_message", stdout_sha256=_sha(raw), stdout_bytes=len(raw), forwarding_status="write_completed")
    assert raw.endswith(b"\n") and event["stdout_sha256"] != _sha(raw[:-1])
    assert receipt["owner_exchange_attempts"] == receipt["observed_responses"] == 1
    assert receipt["owner_exchange_unconfirmed"] == 0 and receipt["complete"] is True
    text = json.dumps(receipt)
    assert "PRIVATE_SOURCE_MARKER" not in text and str(tmp_path) not in text
    _flags(receipt)


@pytest.mark.parametrize("refused", [False, True])
def test_notification_is_explicit_no_output_even_for_owner_refusal(authored, tmp_path, refused):
    if refused:
        authored.response = lambda message, value: {**value, "ok": False, "message": None,
                                                    "reason_code": "budget_exhausted"}
    observation = stdio.ForwardingObservation()
    outcome, replies, raw = exchange(tmp_path, line({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                                     observation=observation)
    receipt = observation.receipt()
    event = receipt["events"][0]
    assert outcome == [2 if refused else 0] and raw == b"" and replies == []
    assert event["output_kind"] == event["forwarding_status"] == "no_output"
    assert event["stdout_sha256"] is None and event["stdout_bytes"] == 0
    assert event["owner_ok"] is not refused and receipt["complete"] is True


@pytest.mark.parametrize("reason", ["budget_exhausted", "invalid_request", "peer_refused"])
def test_owner_refusal_stdout_is_a_local_error_not_original_payload(authored, tmp_path, reason):
    authored.response = lambda message, value: {**value, "ok": False, "message": None, "reason_code": reason}
    observation = stdio.ForwardingObservation()
    outcome, replies, raw = exchange(tmp_path, line(ping()), observation=observation)
    receipt = observation.receipt()
    event = receipt["events"][0]
    assert outcome == [2] and event["owner_ok"] is False and event["owner_reason_code"] == reason
    assert event["output_kind"] == "local_error" and event["forwarding_status"] == "write_completed"
    assert event["stdout_sha256"] == _sha(raw) and replies[0]["error"]["message"] == reason
    assert receipt["complete"] is True


def test_local_budget_refusal_preserves_validated_reply_but_no_forward_claim(authored, tmp_path):
    authored.descriptor["max_total_response_bytes"] = 1024
    def text(message, value):
        value["message"]["result"] = {"source": "x" * 400}
        return value
    authored.response = text
    observation = stdio.ForwardingObservation()
    outcome, replies, raw = exchange(tmp_path, b"".join(line(ping(i)) for i in range(4)), observation=observation)
    receipt = observation.receipt()
    assert outcome == [2] and len(replies) == 1 and receipt["observed_responses"] == 2
    assert [event["forwarding_status"] for event in receipt["events"]] == ["write_completed", "budget_refused"]
    assert all(event["stdout_bytes"] > 0 for event in receipt["events"])
    assert receipt["complete"] is True and receipt["owner_exchange_unconfirmed"] == 0


def test_cancel_after_validated_response_keeps_planned_output_unconfirmed(authored, tmp_path):
    signal = threading.Event()
    def cancel(message, value):
        signal.set()
        return value
    authored.response = cancel
    observation = stdio.ForwardingObservation()
    outcome, replies, raw = exchange(tmp_path, line(ping()), cancel_event=signal, observation=observation)
    receipt = observation.receipt()
    assert outcome == [2] and replies == [] and raw == b""
    assert receipt["complete"] is True and receipt["observed_responses"] == 1
    event = receipt["events"][0]
    assert event["forwarding_status"] == "unconfirmed" and event["stdout_bytes"] > 0
    assert event["stdout_sha256"] == _sha(line({"jsonrpc": "2.0", "id": 1, "result": {}}))


@pytest.mark.parametrize("phase", ["before", "after_complete_write"])
def test_stdout_failure_never_infers_delivered_bytes(authored, tmp_path, monkeypatch, phase):
    original = stdio._Stdio.write
    def fail(self, raw):
        if phase == "after_complete_write": original(self, raw)
        raise OSError("PRIVATE_WRITE_FAILURE")
    monkeypatch.setattr(stdio._Stdio, "write", fail)
    observation = stdio.ForwardingObservation()
    outcome, replies, raw = exchange(tmp_path, line(ping()), observation=observation)
    receipt = observation.receipt()
    assert outcome == [2] and bool(raw) is (phase == "after_complete_write")
    assert receipt["events"][0]["forwarding_status"] == "unconfirmed"
    assert receipt["events"][0]["stdout_bytes"] > 0 and receipt["complete"] is True
    assert "PRIVATE_WRITE_FAILURE" not in json.dumps(receipt)


@pytest.mark.parametrize("failure", ["exception", "wrong_session", "wrong_id"])
def test_owner_exchange_failure_is_explicit_and_not_a_zero_operation_claim(authored, tmp_path, failure):
    def bad(message, value):
        if failure == "exception": raise RuntimeError("PRIVATE_OWNER_FAILURE")
        if failure == "wrong_session": value["session_id"] = "f" * 32
        else: value["message"]["id"] = "foreign"
        return value
    authored.response = bad
    observation = stdio.ForwardingObservation()
    outcome, replies, raw = exchange(tmp_path, line(ping()) + line(ping(2)), observation=observation)
    receipt = observation.receipt()
    assert outcome == [2] and len(authored.calls) == 1 and len(replies) == 1
    assert receipt["complete"] is True and receipt["owner_exchange_attempts"] == receipt["owner_exchange_unconfirmed"] == 1
    assert receipt["observed_responses"] == 0 and receipt["events"] == []
    assert "PRIVATE_OWNER_FAILURE" not in json.dumps(receipt)


def test_local_parse_error_stdout_is_outside_owner_response_scope(authored, tmp_path):
    observation = stdio.ForwardingObservation()
    outcome, replies, raw = exchange(tmp_path, b"invalid\n" + line(ping()), observation=observation)
    receipt = observation.receipt()
    assert outcome == [0] and len(replies) == 2 and len(receipt["events"]) == 1
    assert receipt["events"][0]["stdout_sha256"] == _sha(raw.splitlines(keepends=True)[1])
    assert receipt["owner_exchange_attempts"] == 1


@pytest.mark.parametrize("method", ["_start", "_bind", "_exchange_started", "_response_validated", "_output_finished", "_finish"])
def test_ordinary_observation_failure_preserves_successful_wire_outcome(authored, tmp_path, monkeypatch, method):
    observation = stdio.ForwardingObservation()
    def unavailable(*args, **kwargs):
        raise RuntimeError("PRIVATE_OBSERVATION_FAILURE")
    monkeypatch.setattr(observation, method, unavailable)
    outcome, replies, raw = exchange(tmp_path, line(ping()), observation=observation)
    receipt = observation.receipt()
    assert outcome == [0] and replies == [{"jsonrpc": "2.0", "id": 1, "result": {}}]
    assert receipt["status"] == "incomplete" and receipt["complete"] is False
    assert receipt["reason"] == "observation_failed"
    assert receipt["owner_exchange_attempts"] is receipt["owner_exchange_unconfirmed"] is receipt["observed_responses"] is None
    assert "PRIVATE_OBSERVATION_FAILURE" not in json.dumps(receipt)


def test_failure_after_event_append_preserves_reference_without_complete_observation(authored, tmp_path, monkeypatch):
    observation = stdio.ForwardingObservation()
    original = observation._response_validated
    def uncertain(*args, **kwargs):
        original(*args, **kwargs)
        raise ValueError("authored event publication failure")
    monkeypatch.setattr(observation, "_response_validated", uncertain)
    outcome, replies, raw = exchange(tmp_path, line(ping()), observation=observation)
    receipt = observation.receipt()
    assert outcome == [0] and len(replies) == len(receipt["events"]) == 1
    assert receipt["complete"] is False and receipt["events"][0]["forwarding_status"] == "unconfirmed"


@pytest.mark.parametrize("source", ["owner", "observation"])
@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_baseexception_propagates_after_incomplete_finalization(authored, tmp_path, monkeypatch, source, interruption):
    observation = stdio.ForwardingObservation()
    def interrupted(*args, **kwargs):
        raise interruption()
    if source == "owner": authored.response = interrupted
    else: monkeypatch.setattr(observation, "_response_validated", interrupted)
    outcome, replies, raw = exchange(tmp_path, line(ping()), observation=observation)
    receipt = observation.receipt()
    assert len(outcome) == 1 and isinstance(outcome[0], interruption)
    assert replies == [] and raw == b"" and receipt["complete"] is False
    assert receipt["returncode"] is None and receipt["status"] == "incomplete"
    assert observation._finished is True


@pytest.mark.parametrize("after", [False, True])
def test_interrupted_observation_finalizer_cannot_publish_complete_success(authored, tmp_path, monkeypatch, after):
    observation = stdio.ForwardingObservation()
    original = observation._finish
    def interrupted(*args):
        if after: original(*args)
        raise KeyboardInterrupt()
    monkeypatch.setattr(observation, "_finish", interrupted)
    outcome, replies, raw = exchange(tmp_path, line(ping()), observation=observation)
    receipt = observation.receipt()
    assert len(outcome) == 1 and isinstance(outcome[0], KeyboardInterrupt)
    assert len(replies) == 1 and raw
    assert receipt["complete"] is False and receipt["returncode"] is None


def test_observer_reuse_refuses_before_owner_and_preserves_prior_receipt(authored, tmp_path):
    observation = stdio.ForwardingObservation()
    exchange(tmp_path, line(ping()), observation=observation)
    first = observation.receipt()
    outcome, replies, raw = exchange(tmp_path, line(ping(2)), observation=observation)
    assert len(outcome) == 1 and isinstance(outcome[0], ValueError)
    assert str(outcome[0]) == "forwarding_observation_already_used"
    assert len(authored.calls) == 1 and replies == [] and raw == b""
    assert observation.receipt() == first


def test_active_snapshot_is_incomplete_and_cannot_mutate_live_evidence(authored, tmp_path):
    entered, release = threading.Event(), threading.Event()
    def blocked(message, value):
        entered.set()
        assert release.wait(2), "authored reply release was not signalled"
        return value
    authored.response = blocked
    observation = stdio.ForwardingObservation()
    with pipes() as (incoming, sender, receiver, outgoing):
        thread, outcomes = start_bridge(tmp_path, incoming, outgoing, observation=observation)
        try:
            sender.write(line(ping()))
            sender.close()
            assert entered.wait(1)
            snapshot = observation.receipt()
            assert snapshot["complete"] is False and snapshot["returncode"] is None
            snapshot["events"].append({"source": "foreign mutation"})
            snapshot["bridge_id"] = "foreign"
        finally:
            release.set()
            finish(thread)
        assert outcomes == [0] and drain(receiver)
    result = observation.receipt()
    assert result["complete"] is True and len(result["events"]) == 1
    assert "foreign mutation" not in json.dumps(result) and result["bridge_id"] != "foreign"
    result["events"][0]["forwarding_status"] = "foreign"
    assert observation.receipt()["events"][0]["forwarding_status"] == "write_completed"


def test_concurrent_reuse_cannot_enter_a_second_owner_exchange(authored, tmp_path):
    entered, release = threading.Event(), threading.Event()
    def blocked(message, value):
        entered.set()
        assert release.wait(2)
        return value
    authored.response = blocked
    observation = stdio.ForwardingObservation()
    with pipes() as (incoming, sender, receiver, outgoing):
        thread, outcomes = start_bridge(tmp_path, incoming, outgoing, observation=observation)
        try:
            sender.write(line(ping())); sender.close()
            assert entered.wait(1)
            second, replies, raw = exchange(tmp_path, line(ping(2)), observation=observation)
            assert len(second) == 1 and isinstance(second[0], ValueError)
            assert len(authored.calls) == 1 and replies == [] and raw == b""
        finally:
            release.set()
            finish(thread)
        assert outcomes == [0] and drain(receiver)
    assert observation.receipt()["complete"] is True


@pytest.mark.parametrize("value", [False, {}, object()])
def test_wrong_observer_type_refuses_before_descriptor_or_owner(authored, tmp_path, value):
    outcome, replies, raw = exchange(tmp_path, b"", observation=value)
    assert len(outcome) == 1 and isinstance(outcome[0], ValueError)
    assert authored.loaded == authored.calls == [] and replies == [] and raw == b""


def test_subclass_observer_cannot_replace_the_typed_contract(authored, tmp_path):
    class Foreign(stdio.ForwardingObservation):
        pass
    outcome, replies, raw = exchange(tmp_path, b"", observation=Foreign())
    assert len(outcome) == 1 and isinstance(outcome[0], ValueError) and authored.loaded == []


def test_descriptor_failure_has_unknown_binding_and_no_synthetic_zero(authored, tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ValueError("private descriptor failure")
    monkeypatch.setattr(stdio.protocol, "load_descriptor", unavailable)
    observation = stdio.ForwardingObservation()
    outcome, replies, raw = exchange(tmp_path, b"", observation=observation)
    receipt = observation.receipt()
    assert outcome == [2] and replies == [] and raw == b""
    assert receipt["complete"] is False and receipt["reason"] == "descriptor_unavailable"
    assert receipt["session_id"] is receipt["descriptor_sha256"] is None and receipt["events"] == []


def test_event_population_and_serialized_receipt_are_bounded(authored, tmp_path):
    observation = stdio.ForwardingObservation()
    outcome, replies, raw = exchange(tmp_path, b"".join(line(ping(i)) for i in range(65)), observation=observation)
    receipt = observation.receipt()
    assert outcome == [0] and len(replies) == len(receipt["events"]) == 64
    assert receipt["owner_exchange_attempts"] == 64 and receipt["complete"] is True
    assert len(stdio.protocol.canonical_json(receipt)) <= stdio.MAX_FORWARDING_RECEIPT_BYTES


def test_observer_limit_failure_does_not_reduce_existing_transport_budget(authored, tmp_path, monkeypatch):
    monkeypatch.setattr(stdio, "MAX_FORWARDING_EVENTS", 1)
    observation = stdio.ForwardingObservation()
    outcome, replies, raw = exchange(tmp_path, line(ping()) + line(ping(2)), observation=observation)
    receipt = observation.receipt()
    assert outcome == [0] and len(replies) == 2 and len(authored.calls) == 2
    assert receipt["complete"] is False and len(receipt["events"]) == 1


def test_optional_observer_does_not_change_default_wire(authored, tmp_path):
    raw = b"invalid\n" + line(ping()) + line({"jsonrpc": "2.0", "method": "notifications/initialized"})
    baseline = exchange(tmp_path, raw)
    observed = exchange(tmp_path, raw, observation=stdio.ForwardingObservation())
    assert baseline == observed
