"""No-model checks of the optional authenticated index resolve boundary.

The index is an authored port fixture. Native snapshot authentication and
performance are covered separately; these cases establish no qualification.
"""

from copy import deepcopy
from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import HarnessDisposition
from ipfs_accelerate_py.agent_supervisor.semantic_state.terminal_attempt_index import (
    PREFLIGHT_SCHEMA,
    RESOLVE_SCHEMA,
    TerminalAttemptIndexCapacityError,
    TerminalAttemptIndexConflictError,
    TerminalAttemptIndexError,
    TerminalAttemptIndexIntegrityError,
    TerminalAttemptIndexPendingError,
)
from test.api.semantic_state.test_harness import _cid, _harness
from test.api.semantic_state.test_harness_materialized_dispatch import CapturingMaterializer
from test.api.semantic_state.test_harness_terminal_index import FixtureIndex, setup_index_dispatch


class ResolvingFixtureIndex(FixtureIndex):
    max_entries = 64

    def __init__(self):
        super().__init__()
        self.mutate = lambda response: response
        self.resolve_error = None
        self.after_resolve = lambda: None

    def preflight(self, *_args, **_kwargs):
        pytest.fail("optimized resolution fell back to preflight")

    def lookup(self, *_args, **_kwargs):
        pytest.fail("optimized resolution fell back to lookup")

    def resolve(self, identity, *, replay_context_cid):
        self.calls.append("resolve")
        if self.resolve_error is not None:
            raise self.resolve_error
        entry = self.entries.get(identity)
        if entry and entry["replay_context_cid"] != replay_context_cid:
            raise TerminalAttemptIndexConflictError("context changed")
        if entry and entry["envelope_cid"] is None:
            raise TerminalAttemptIndexPendingError(identity, replay_context_cid, entry["pending_cid"])
        envelope = None
        if entry:
            envelope = self.blocks.get(entry["envelope_cid"])
            if envelope is None or self.record_cid(envelope) != entry["envelope_cid"]:
                raise TerminalAttemptIndexIntegrityError("missing or corrupt immutable envelope")
        entries = len(self.entries)
        completed = sum(item["envelope_cid"] is not None for item in self.entries.values())
        receipt = {
            "schema": PREFLIGHT_SCHEMA, "status": "terminal" if entry else "available",
            "namespace": self.namespace, "attempt_identity": identity,
            "replay_context_cid": replay_context_cid if entry else None,
            "envelope_cid": entry["envelope_cid"] if entry else None,
            "index_cid": _cid("fixture-index-" + str(entries + completed)) if entries else None,
            "revision": entries + completed, "entries": entries,
            "remaining_capacity": self.max_entries - entries,
            "durable": True, "completion_authority": False,
        }
        response = deepcopy({"schema": RESOLVE_SCHEMA, "receipt": receipt, "envelope": envelope})
        self.after_resolve()
        return self.mutate(response)


def test_resolve_dispatch_and_replay_use_one_response_without_legacy_reads():
    index = ResolvingFixtureIndex()
    provider, durable, harness, _, request = setup_index_dispatch(index)
    first = harness.run(request)
    assert first.terminal_persistence == "durably_indexed"
    assert index.calls == ["resolve", "reserve", "publish"]
    _, reopened = _harness(durable, providers=(provider,), terminal_attempt_index=index,
                          context_materializer=CapturingMaterializer())
    reopened._terminal_attempts[first.attempt_identity] = {"outcome": "poison"}
    cas_before = len(durable.cas_calls)
    replay = reopened.run(request)
    assert replay.historical_replay
    assert replay.terminal_persistence == "durably_indexed"
    assert replay.terminal_envelope_cid == first.terminal_envelope_cid
    assert "context_freshness_not_revalidated" in replay.result.reasons
    assert index.calls == ["resolve", "reserve", "publish", "resolve"]
    assert len(provider.calls) == 1
    assert len(durable.cas_calls) == cas_before


def test_index_without_resolve_retains_authenticated_legacy_read_path():
    index = FixtureIndex()
    provider, durable, harness, _, request = setup_index_dispatch(index)
    first = harness.run(request)
    cas_before = len(durable.cas_calls)
    index.calls.clear()
    replay = harness.run(request)
    assert replay.historical_replay
    assert replay.terminal_envelope_cid == first.terminal_envelope_cid
    assert index.calls == ["preflight", "lookup", "preflight"]
    assert len(provider.calls) == 1
    assert len(durable.cas_calls) == cas_before


@pytest.mark.parametrize("capability", [None, False, "raising"])
def test_present_invalid_resolve_never_downgrades_to_legacy_reads(capability):
    index = ResolvingFixtureIndex()
    if capability == "raising":
        def broken(*_args, **_kwargs):
            raise OSError("owner unavailable")
        index.resolve = broken
    else:
        index.resolve = capability
    provider, durable, harness, _, request = setup_index_dispatch(index)
    result = harness.run(request)
    assert result.result.disposition in {HarnessDisposition.REJECTED.value, HarnessDisposition.UNAVAILABLE.value}
    expected = "attempt_index_unavailable" if capability == "raising" else "attempt_index_integrity_failed"
    assert expected in result.result.reasons
    assert provider.calls == []
    assert len(durable.cas_calls) == 1
    assert index.calls == []


def test_malformed_resolve_responses_refuse_without_dispatch_or_cache_fallback():
    # Each mutation violates a different boundary; cached success cannot hide it.
    faults = [
        "outer_extra", "outer_schema", "receipt_extra", "receipt_namespace",
        "receipt_identity", "receipt_context", "receipt_cid", "receipt_authority",
        "receipt_durable", "receipt_bool_revision", "receipt_bad_counters", "receipt_index_cid",
        "missing_envelope", "available_with_envelope", "envelope_extra", "envelope_context",
        "envelope_identity", "invalid_typed_result", "simulated_string",
        "root_conflict_integer", "human_review_null", "historical_string", "historical_true",
    ]
    index = ResolvingFixtureIndex()
    provider, durable, harness, root, request = setup_index_dispatch(index)
    first = harness.run(request)
    cas_before = len(durable.cas_calls)
    for fault in faults:
        def mutate(response, *, fault=fault):
            receipt = response["receipt"]
            envelope = response["envelope"]
            outcome = envelope["outcome"]
            if fault == "outer_extra":
                response["cached"] = True
            elif fault == "outer_schema":
                response["schema"] = "unknown@1"
            elif fault == "receipt_extra":
                receipt["grant"] = True
            elif fault == "receipt_namespace":
                receipt["namespace"] += "/other"
            elif fault == "receipt_identity":
                receipt["attempt_identity"] = _cid("other-attempt")
            elif fault == "receipt_context":
                receipt["replay_context_cid"] = _cid("other-context")
            elif fault == "receipt_cid":
                receipt["envelope_cid"] = _cid("other-envelope")
            elif fault == "receipt_authority":
                receipt["completion_authority"] = True
            elif fault == "receipt_durable":
                receipt["durable"] = 1
            elif fault == "receipt_bool_revision":
                receipt["revision"] = True
            elif fault == "receipt_bad_counters":
                receipt["entries"] = 0
            elif fault == "receipt_index_cid":
                receipt["index_cid"] = None
            elif fault == "missing_envelope":
                response["envelope"] = None
            elif fault == "available_with_envelope":
                receipt["status"] = "available"
            elif fault == "envelope_extra":
                envelope["grant"] = True
            elif fault == "envelope_context":
                envelope["replay_context_cid"] = _cid("other-context")
            elif fault == "envelope_identity":
                envelope["attempt_identity"] = _cid("other-attempt")
            elif fault == "invalid_typed_result":
                outcome["result"] = {"disposition": "invented"}
            elif fault == "simulated_string":
                outcome["simulated"] = "false"
            elif fault == "root_conflict_integer":
                outcome["root_conflict"] = 0
            elif fault == "human_review_null":
                outcome["human_review_required"] = None
            elif fault == "historical_string":
                outcome["historical_replay"] = "false"
            else:
                outcome["historical_replay"] = True
            if fault in {
                "envelope_extra", "envelope_context", "envelope_identity", "invalid_typed_result",
                "simulated_string", "root_conflict_integer", "human_review_null",
                "historical_string", "historical_true",
            }:
                # Byte-consistent adversarial bodies must still fail the
                # schema/type gates; a CID mismatch must not mask that check.
                receipt["envelope_cid"] = index.record_cid(envelope)
            return response

        index.mutate = mutate
        harness._terminal_attempts[first.attempt_identity] = {"outcome": "poison"}
        replay = harness.run(request)
        assert "attempt_index_integrity_failed" in replay.result.reasons, fault
        assert not replay.historical_replay, fault
        assert replay.terminal_persistence == "not_published", fault
        assert len(provider.calls) == 1, fault
        assert len(durable.cas_calls) == cas_before, fault
        assert durable.read_root(request.repository_id) == root, fault
        assert index.calls[-1] == "resolve", fault


def test_resolve_errors_keep_pending_context_capacity_and_integrity_gates():
    identity = _cid("unused")
    for error in [
        TerminalAttemptIndexPendingError(identity, _cid("other-context"), _cid("pending")),
        TerminalAttemptIndexConflictError("changed"), TerminalAttemptIndexCapacityError("full"),
        TerminalAttemptIndexIntegrityError("corrupt"), TerminalAttemptIndexError("unavailable"),
    ]:
        index = ResolvingFixtureIndex()
        index.resolve_error = error
        provider, durable, harness, _, request = setup_index_dispatch(index)
        result = harness.run(request)
        code = "attempt_context_changed" if isinstance(error, TerminalAttemptIndexPendingError) else error.reason_code
        assert code in result.result.reasons
        if code == "attempt_context_changed":
            assert "new_attempt_key_required" in result.result.reasons
        assert provider.calls == []
        assert len(durable.cas_calls) == 1
        assert index.calls == ["resolve"]


def test_available_snapshot_still_reserves_current_head_before_dispatch():
    index = ResolvingFixtureIndex()
    provider, durable, harness, _, request = setup_index_dispatch(index)
    identity = request.attempt_identity(mode=harness.policy.mode)
    context = harness._replay_context_cid(request)
    index.after_resolve = lambda: index.reserve(identity, replay_context_cid=context)
    result = harness.run(request)
    assert "attempt_recovery_required" in result.result.reasons
    assert provider.calls == []
    assert len(durable.cas_calls) == 1
    assert index.calls == ["resolve", "reserve", "reserve"]
    assert index.entries[identity]["envelope_cid"] is None


def test_immutable_terminal_snapshot_survives_unrelated_head_advance():
    index = ResolvingFixtureIndex()
    provider, durable, harness, _, request = setup_index_dispatch(index)
    first = harness.run(request)
    cas_before = len(durable.cas_calls)
    index.after_resolve = lambda: index.reserve("other-attempt", replay_context_cid="other-context")
    replay = harness.run(request)
    assert replay.historical_replay
    assert replay.terminal_envelope_cid == first.terminal_envelope_cid
    assert len(provider.calls) == 1
    assert len(durable.cas_calls) == cas_before
    assert index.calls == ["resolve", "reserve", "publish", "resolve", "reserve"]


def test_changed_request_still_refuses_resolved_historical_success():
    index = ResolvingFixtureIndex()
    provider, durable, harness, _, request = setup_index_dispatch(index)
    harness.run(request)
    cas_before = len(durable.cas_calls)
    result = harness.run(replace(request, objective="changed objective"))
    assert "attempt_context_changed" in result.result.reasons
    assert "new_attempt_key_required" in result.result.reasons
    assert not result.historical_replay
    assert len(provider.calls) == 1
    assert len(durable.cas_calls) == cas_before
