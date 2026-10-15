"""Hermetic harness/index integration; not native durability qualification.

The injected discovery port explicitly models reservation and authenticated
envelopes. Real Kit close/reopen coverage lives in test_harness_native_resume.
No provider in this file runs a real model or charges for an invocation.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import HarnessDisposition
from ipfs_accelerate_py.agent_supervisor.semantic_state.harness import HARNESS_ATTEMPT_SCHEMA
from ipfs_accelerate_py.agent_supervisor.semantic_state.terminal_attempt_index import (
    PREFLIGHT_SCHEMA,
    PUBLICATION_SCHEMA,
    TICKET_SCHEMA,
    TerminalAttemptIndexCapacityError,
    TerminalAttemptIndexConflictError,
    TerminalAttemptIndexError,
    TerminalAttemptIndexIntegrityError,
    TerminalAttemptIndexPendingError,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes
from test.api.semantic_state.test_harness import _accept_request, _bootstrap, _cid, _harness
from test.api.semantic_state.test_harness_materialized_dispatch import (
    CapturingMaterializer,
    setup_dispatch,
)


class FixtureIndex:
    """Authored owner-index contract fixture with shared reopen state."""

    namespace = "terminal-attempt-index/fixture/shard-0"

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.blocks: dict[str, dict[str, Any]] = {}
        self.calls: list[str] = []
        self.preflight_error: Exception | None = None
        self.publish_error: Exception | None = None
        self.on_reserve = lambda: None

    def preflight(self, identity, *, replay_context_cid):
        self.calls.append("preflight")
        if self.preflight_error is not None:
            raise self.preflight_error
        entry = self.entries.get(identity)
        if entry and entry["replay_context_cid"] != replay_context_cid:
            raise TerminalAttemptIndexConflictError("changed context")
        if entry and entry["envelope_cid"] is None:
            raise TerminalAttemptIndexPendingError(identity, replay_context_cid, entry["pending_cid"])
        return {
            "schema": PREFLIGHT_SCHEMA, "status": "terminal" if entry else "available",
            "namespace": self.namespace, "attempt_identity": identity,
            "replay_context_cid": replay_context_cid if entry else None,
            "envelope_cid": entry["envelope_cid"] if entry else None,
            "durable": True, "completion_authority": False,
        }

    def lookup(self, identity):
        self.calls.append("lookup")
        entry = self.entries.get(identity)
        if entry is None:
            return None
        if entry["envelope_cid"] is None:
            raise TerminalAttemptIndexPendingError(identity, entry["replay_context_cid"], entry["pending_cid"])
        cid = entry["envelope_cid"]
        body = self.blocks.get(cid)
        if body is None or self.record_cid(body) != cid:
            raise TerminalAttemptIndexIntegrityError("missing or corrupt referenced envelope")
        return deepcopy(body)

    def record_cid(self, record):
        return cid_for_payload(record)

    def reserve(self, identity, *, replay_context_cid):
        self.calls.append("reserve")
        if identity in self.entries:
            entry = self.entries[identity]
            raise TerminalAttemptIndexPendingError(identity, entry["replay_context_cid"], entry["pending_cid"])
        ticket = {
            "schema": TICKET_SCHEMA, "namespace": self.namespace,
            "attempt_identity": identity, "replay_context_cid": replay_context_cid,
            "nonce": "fixture-only-not-a-native-ticket",
            "pending_cid": _cid("pending:" + identity),
        }
        self.entries[identity] = {**ticket, "envelope_cid": None}
        self.on_reserve()
        return ticket

    def publish(self, identity, record, *, reservation):
        self.calls.append("publish")
        entry = self.entries[identity]
        assert reservation["nonce"] == entry["nonce"]
        assert reservation["pending_cid"] == entry["pending_cid"]
        assert record["replay_context_cid"] == entry["replay_context_cid"]
        assert record["schema"] == HARNESS_ATTEMPT_SCHEMA
        cid = self.record_cid(record)
        self.blocks[cid] = deepcopy(record)
        if self.publish_error is not None:
            self.publish_error.envelope_cid = cid
            raise self.publish_error
        entry["envelope_cid"] = cid
        return {
            "schema": PUBLICATION_SCHEMA, "status": "published", "namespace": self.namespace,
            "attempt_identity": identity, "replay_context_cid": record["replay_context_cid"],
            "envelope_cid": cid, "pending_cid": entry["pending_cid"],
            "durable": True, "completion_authority": False,
        }


def setup_index_dispatch(index):
    provider, durable, harness, root, request = setup_dispatch(CapturingMaterializer())
    harness.terminal_attempt_index = index
    return provider, durable, harness, root, request


def test_indexed_result_uses_actual_envelope_cid_and_no_identity_alias():
    index = FixtureIndex()
    durable, harness = _harness(terminal_attempt_index=index)
    root = _bootstrap(harness)
    request = _accept_request(root)
    outcome = harness.run(request)
    assert outcome.result.disposition == HarnessDisposition.ACCEPTED.value
    assert outcome.terminal_persistence == "durably_indexed"
    record = index.blocks[outcome.terminal_envelope_cid]
    assert cid_for_payload(record) == outcome.terminal_envelope_cid
    assert record["outcome"]["result"] == outcome.result.to_dict()
    assert record["outcome"]["terminal_persistence"] == "publication_pending"
    assert outcome.terminal_reservation_cid == index.entries[outcome.attempt_identity]["pending_cid"]
    assert not durable.has(outcome.attempt_identity)
    assert harness._terminal_attempts == {}
    assert "nonce" not in outcome.to_dict()
    assert index.calls == ["preflight", "lookup", "reserve", "publish"]


def test_fresh_harness_replays_indexed_result_without_provider_or_root_cas():
    index = FixtureIndex()
    provider, durable, harness, _, request = setup_index_dispatch(index)
    first = harness.run(request)
    _, reopened = _harness(durable, providers=(provider,), terminal_attempt_index=index,
                          context_materializer=CapturingMaterializer())
    # A stale local cache cannot override the configured authority.
    reopened._terminal_attempts[first.attempt_identity] = {"outcome": "poison"}
    cas_before = len(durable.cas_calls)
    replay = reopened.run(request)
    assert len(provider.calls) == 1
    assert len(durable.cas_calls) == cas_before
    assert replay.historical_replay
    assert replay.result.current_root == first.result.current_root
    assert replay.terminal_envelope_cid == first.terminal_envelope_cid
    assert replay.terminal_persistence == "durably_indexed"
    assert "context_freshness_not_revalidated" in replay.result.reasons
    assert index.calls.count("reserve") == 1


@pytest.mark.parametrize("fault", ["missing", "corrupt", "bad_result"])
def test_indexed_envelope_failures_do_not_become_cache_misses(fault):
    index = FixtureIndex()
    provider, durable, harness, root, request = setup_index_dispatch(index)
    first = harness.run(request)
    cid = first.terminal_envelope_cid
    if fault == "missing":
        del index.blocks[cid]
    elif fault == "corrupt":
        index.blocks[cid]["terminal"] = False
    else:
        record = index.blocks.pop(cid)
        record["outcome"]["result"] = {"disposition": "invented"}
        new_cid = cid_for_payload(record)
        index.blocks[new_cid] = record
        index.entries[first.attempt_identity]["envelope_cid"] = new_cid
    harness._terminal_attempts[first.attempt_identity] = {"outcome": "poison"}
    cas_before = len(durable.cas_calls)
    result = harness.run(request)
    assert "attempt_index_integrity_failed" in result.result.reasons
    assert len(provider.calls) == 1
    assert len(durable.cas_calls) == cas_before
    assert durable.read_root(request.repository_id) == root
    assert not result.historical_replay


@pytest.mark.parametrize("error", [
    TerminalAttemptIndexCapacityError("full"), TerminalAttemptIndexError("owner unavailable"),
    TerminalAttemptIndexIntegrityError("broken"), OSError("connection unavailable"),
])
def test_preflight_errors_stop_before_worktree_or_provider(error, monkeypatch):
    index = FixtureIndex()
    index.preflight_error = error
    provider, durable, harness, root, request = setup_index_dispatch(index)
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.harness.create_isolated_worktree",
        lambda **_kwargs: pytest.fail("worktree effect before durable index admission"),
    )
    request = replace(request, repo_path="/fixture/not-opened")
    result = harness.run(request)
    assert provider.calls == []
    assert result.steps_completed == ()
    assert len(durable.cas_calls) == 1  # Bootstrap only.
    assert durable.read_root(request.repository_id) == root
    assert result.terminal_persistence == "not_published"
    assert index.calls == ["preflight"]
    assert harness._terminal_attempts == {}


@pytest.mark.parametrize("changed_context", [False, True])
def test_pending_after_interruption_requires_recovery_without_dispatch(changed_context):
    index = FixtureIndex()
    provider, durable, harness, root, request = setup_index_dispatch(index)
    identity = request.attempt_identity(mode=harness.policy.mode)
    context = harness._replay_context_cid(request)
    ticket = index.reserve(identity, replay_context_cid=context)
    _, reopened = _harness(durable, providers=(provider,), terminal_attempt_index=index,
                          context_materializer=CapturingMaterializer())
    if changed_context:
        request = replace(request, objective="changed objective")
    result = reopened.run(request)
    assert provider.calls == []
    assert len(durable.cas_calls) == 1
    assert durable.read_root(request.repository_id) == root
    assert not result.historical_replay
    assert result.result.disposition == HarnessDisposition.REJECTED.value
    expected = "attempt_context_changed" if changed_context else "attempt_recovery_required"
    assert expected in result.result.reasons
    if changed_context:
        assert "new_attempt_key_required" in result.result.reasons
    else:
        assert result.terminal_reservation_cid == ticket["pending_cid"]


def test_reservation_winner_blocks_interleaved_same_identity_before_provider():
    index = FixtureIndex()
    provider, durable, first, _, request = setup_index_dispatch(index)
    _, second = _harness(durable, providers=(provider,), terminal_attempt_index=index,
                        context_materializer=CapturingMaterializer())
    interrupted = []

    def enter_second():
        assert provider.calls == []
        interrupted.append(second.run(request))
        assert provider.calls == []

    index.on_reserve = enter_second
    completed = first.run(request)
    assert len(provider.calls) == 1
    assert completed.terminal_persistence == "durably_indexed"
    assert "attempt_recovery_required" in interrupted[0].result.reasons
    assert index.calls.count("reserve") == 1


def test_configured_index_requires_explicit_attempt_key_before_effects():
    index = FixtureIndex()
    provider, durable, harness, _, request = setup_index_dispatch(index)
    result = harness.run(replace(request, attempt_key=None))
    assert "durable_attempt_key_required" in result.result.reasons
    assert provider.calls == []
    assert len(durable.cas_calls) == 1
    assert index.calls == []


def test_configured_index_requires_explicit_owner_record_codec_before_effects():
    index = FixtureIndex()
    index.record_cid = None
    provider, durable, harness, _, request = setup_index_dispatch(index)
    result = harness.run(request)
    assert "attempt_index_integrity_failed" in result.result.reasons
    assert provider.calls == []
    assert len(durable.cas_calls) == 1
    assert index.calls == []


def test_publication_failure_preserves_accepted_root_and_exposes_orphan_cid():
    index = FixtureIndex()
    index.publish_error = TerminalAttemptIndexError("CAS not available")
    durable, harness = _harness(terminal_attempt_index=index)
    root = _bootstrap(harness)
    request = _accept_request(root)
    outcome = harness.run(request)
    assert outcome.result.disposition == HarnessDisposition.ACCEPTED.value
    assert outcome.result.current_root.generation == 2
    assert outcome.result.previous_root == root
    assert durable.read_root(request.repository_id) == outcome.result.current_root
    assert len(durable.cas_calls) == 2
    assert outcome.terminal_persistence == "index_publication_failed"
    assert outcome.terminal_persistence_error == "attempt_index_unavailable"
    assert outcome.terminal_envelope_cid in index.blocks
    assert index.blocks[outcome.terminal_envelope_cid]["outcome"]["result"] == outcome.result.to_dict()
    assert index.entries[outcome.attempt_identity]["envelope_cid"] is None
    replay = harness.run(request)
    assert "attempt_recovery_required" in replay.result.reasons
    assert len(durable.cas_calls) == 2
    assert durable.read_root(request.repository_id) == outcome.result.current_root


def test_rejected_conversion_outcome_is_indexed_without_semantic_promotion():
    index = FixtureIndex()
    provider, durable, harness, root, request = setup_index_dispatch(index)
    outcome = harness.run(request)
    assert outcome.result.disposition == HarnessDisposition.REJECTED.value
    assert "simulation_never_verifies_or_promotes" in outcome.result.reasons
    assert len(provider.calls) == 1
    assert outcome.terminal_persistence == "durably_indexed"
    assert durable.read_root(request.repository_id) == root


def test_no_index_replay_is_explicitly_memory_only_and_no_alias_is_written():
    durable, harness = _harness()
    root = _bootstrap(harness)
    request = _accept_request(root)
    first = harness.run(request)
    assert first.terminal_persistence == "memory_only"
    assert not durable.has(first.attempt_identity)
    assert cid_for_payload(durable.get(first.terminal_envelope_cid)) == first.terminal_envelope_cid
    replay = harness.run(request)
    assert replay.historical_replay
    assert replay.terminal_persistence == "memory_only"
    _, fresh = _harness(durable)
    assert fresh._load_terminal(first.attempt_identity) is None


def test_public_index_port_and_native_adapter_exports():
    from ipfs_accelerate_py.agent_supervisor import semantic_state

    assert semantic_state.HarnessTerminalAttemptIndex
    assert semantic_state.IpfsKitNativeDurableStateAdapter


def test_indexed_unicode_envelope_uses_owner_codec_without_rewriting_public_keys():
    class Utf8OwnerFixture(FixtureIndex):
        def record_cid(self, record):
            raw = json.dumps(record, ensure_ascii=False, allow_nan=False,
                             sort_keys=True, separators=(",", ":")).encode("utf-8")
            return cid_for_bytes(raw)

    index = Utf8OwnerFixture()
    durable, harness = _harness(terminal_attempt_index=index)
    root = _bootstrap(harness)
    request = _accept_request(root, changed_symbol_ids=("pkg.target.ΔVALUE",))
    first = harness.run(request)
    assert first.result.disposition == HarnessDisposition.ACCEPTED.value
    assert first.terminal_persistence == "durably_indexed"
    record = index.blocks[first.terminal_envelope_cid]
    assert first.terminal_envelope_cid == index.record_cid(record)
    assert first.terminal_envelope_cid != cid_for_payload(record)
    assert first.attempt_identity == request.attempt_identity(mode=harness.policy.mode)
    assert record["replay_context_cid"] == harness._replay_context_cid(request)
    _, reopened = _harness(durable, terminal_attempt_index=index)
    replay = reopened.run(request)
    assert replay.historical_replay
    assert replay.terminal_persistence == "durably_indexed"
    assert replay.terminal_envelope_cid == first.terminal_envelope_cid
    assert len(durable.cas_calls) == 2


@pytest.mark.parametrize("event_parent", [None, _cid("prior-event")])
def test_fenced_preflight_refusal_does_not_read_or_write_supplied_pack_or_event(event_parent):
    index = FixtureIndex()
    index.preflight_error = TerminalAttemptIndexError("owner fenced")
    durable, harness = _harness(terminal_attempt_index=index)
    root = _bootstrap(harness)
    request = _accept_request(root, event_parent_cid=event_parent)

    def forbidden(*_args, **_kwargs):
        pytest.fail("owner accessed after fencing, before durable reservation")

    durable.has = durable.get = durable.put = durable.read_root = forbidden
    result = harness.run(request)
    assert result.result.disposition == HarnessDisposition.UNAVAILABLE.value
    assert "attempt_index_unavailable" in result.result.reasons
    assert result.terminal_persistence == "not_published"
    assert len(durable.cas_calls) == 1
    assert index.calls == ["preflight"]
    if event_parent is not None:
        assert result.result.event_head_cid == event_parent
    else:
        assert result.result.event_head_cid not in durable._objects
