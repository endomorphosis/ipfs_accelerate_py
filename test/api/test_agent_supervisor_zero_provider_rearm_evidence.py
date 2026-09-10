"""A zero outer receipt count must not hide a dispatched Portal callback."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.event_log import append_jsonl_event
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalBridgeError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
)
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
    _population,
)
from test.api.test_agent_supervisor_database_portal_bridge import (
    _attempt,
    _bridge_for_projection,
    _record,
)


def _native_callback(tmp_path, *, middle=(), terminal_override=None):
    bridge = _bridge_for_projection(tmp_path)
    bridge.task_header_prefix = "## LGSWF-"
    attempt = _attempt()
    paths, binding = bridge._ensure_attempt_projection(attempt, _record())
    identity = bridge._prior_projection_identity(paths, binding)
    append_jsonl_event(paths.events, "implementation_started", {
        **identity, "attempt": 1, "provider_dispatched": False,
    })
    for kind, payload in middle:
        append_jsonl_event(paths.events, kind, {**identity, **payload})
    terminal = {
        **identity, "attempt": 1, "provider_dispatched": False,
        "attempt_consumed": False, "returncode": 78,
        "implementation_commit": "",
        "validation_result": {"attempted": False, "passed": False},
        "commit_result": {"committed": False, "commit": ""},
        "merge_result": {"merged": False, "reason": "not_attempted"},
        **(terminal_override or {}),
    }
    append_jsonl_event(paths.events, "implementation_finished", terminal)
    paths.state.write_text(json.dumps({
        "implementation_in_progress": False, "active_provider_runner": {},
        "active_attempt": 0, "last_implementation_task_id": identity["task_id"],
        "last_implementation_task_cid": identity["canonical_task_cid"],
        "last_implementation_commit": "", "last_implementation_returncode": 78,
    }))
    return bridge, attempt, paths


def test_explicit_undispatched_native_callback_remains_eligible(tmp_path):
    bridge, attempt, paths = _native_callback(tmp_path)
    before = {p.name: p.read_bytes() for p in paths.root.iterdir() if p.is_file()}
    assert bridge.zero_provider_failure_rearm_ready(attempt)
    assert before == {p.name: p.read_bytes() for p in paths.root.iterdir() if p.is_file()}


@pytest.mark.parametrize("middle", [
    (("implementation_auto_rescue_provider_started", {}),),
    (("implementation_provider_started", {}),),
    (("runner_callback", {"provider_dispatched": True}),),
    (("runner_callback", {"attempt_consumed": True}),),
    (("runner_callback", {"provider_dispatched": "false"}),),
    (("validation_started", {}),),
    (("implementation_proposal_validated", {}),),
    (("runner_callback", {"validation_result": {"attempted": True}}),),
    (("merge_started", {}),),
])
def test_execution_evidence_vetoes_even_a_later_false_terminal(tmp_path, middle):
    bridge, attempt, _paths = _native_callback(tmp_path, middle=middle)
    assert not bridge.zero_provider_failure_rearm_ready(attempt)


@pytest.mark.parametrize("override", [
    {"provider_dispatched": True},
    {"attempt_consumed": True},
    {"validation_result": {"attempted": True, "passed": False}},
    {"validation_result": {}},
    {"implementation_commit": "a" * 40},
    {"commit_result": {"committed": True}},
    {"merge_result": {"merged": False, "queued": True, "reason": "not_attempted"}},
    {"canonical_task_cid": "foreign-task"},
    {"attempt": 2},
])
def test_ambiguous_or_executed_terminal_is_not_non_dispatch(tmp_path, override):
    bridge, attempt, _paths = _native_callback(tmp_path, terminal_override=override)
    assert not bridge.zero_provider_failure_rearm_ready(attempt)


@pytest.mark.parametrize("damage", ["missing", "empty", "truncated", "hash", "state", "marker"])
def test_missing_or_inconsistent_callback_evidence_never_grants_rearm(tmp_path, damage):
    bridge, attempt, paths = _native_callback(tmp_path)
    if damage == "missing":
        paths.events.unlink()
    elif damage == "empty":
        paths.events.write_text("")
    elif damage == "truncated":
        paths.events.write_text(paths.events.read_text().splitlines()[0] + "\n")
    elif damage == "hash":
        paths.events.write_text(paths.events.read_text().replace('"returncode":78', '"returncode":79'))
    elif damage == "state":
        state = json.loads(paths.state.read_text())
        state["active_provider_runner"] = {"pid": 123}
        paths.state.write_text(json.dumps(state))
    else:
        (paths.root / "implementation-protected-path-incident.json").write_text("retained")
    assert not bridge.zero_provider_failure_rearm_ready(attempt)


def test_wrong_fence_or_missing_attempt_is_not_non_dispatch(tmp_path):
    bridge, attempt, _paths = _native_callback(tmp_path)
    assert not bridge.zero_provider_failure_rearm_ready(replace(attempt, fencing_token=8))
    assert not bridge.zero_provider_failure_rearm_ready(replace(attempt, attempt_id="other"))


@pytest.mark.parametrize("value", [None, False, "0", 0.0, -1, 1])
def test_zero_count_candidate_rejects_missing_or_coerced_counts(value):
    receipt = {
        "operation": "database_task_claim_failure",
        "failure_kind": "terminal_portal_bridge_error",
        "automatic_retry_admitted": False,
        "provider_invocation_count": value,
        "effect_claim_count": 0,
        "settlement_id": "settlement:test",
    }
    assert not DatabaseImplementationDaemon.portal_claim_failure_receipt_is_zero_provider_rearmable(receipt)


class _Provider:
    def __init__(self, verified, reason="outer callback failed after Portal work"):
        self.verified = verified
        self.reason = reason
        self.observed = []

    def run(self, _attempt):
        raise DatabasePortalBridgeError(self.reason)

    def zero_provider_failure_rearm_ready(self, attempt):
        self.observed.append(attempt.attempt_id)
        return self.verified


@pytest.mark.parametrize("verified", [False, None, "true", True])
@pytest.mark.parametrize("reason", ["outer callback failed after Portal work", "portal_provider_failed"])
def test_zero_count_failure_needs_explicit_callback_proof_before_cas(tmp_path, verified, reason):
    provider = _Provider(verified, reason)
    daemon = _open_daemon(tmp_path, provider_fn=provider.run)
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        attempt = daemon.get_attempt(first["attempt_id"])
        task = daemon.task_source.get(attempt.task_cid)
        assert task.body["completion_receipt"]["provider_invocation_count"] == 0
        assert task.body["completion_receipt"]["effect_claim_count"] == 0
        original_revision = task.revision
        daemon.authority_mode = "quack"
        rearms = daemon.reconcile_recoverable_portal_failure_rearms(
            recovery_source_validator=lambda: {"source_head": "a" * 40, "source_tree": "b" * 40},
        )
        observed = daemon.task_source.get(attempt.task_cid)
        assert bool(rearms) is (verified is True)
        assert provider.observed == [attempt.attempt_id]
        assert observed.status == ("retrying" if verified is True else "blocked")
        assert observed.revision == original_revision + (1 if verified is True else 0)
        assert daemon.get_attempt(attempt.attempt_id).status == "failed"
    finally:
        daemon.close()


def test_sidecarless_or_unbound_frontier_helper_cannot_rearm_zero_counts(tmp_path):
    def failed(_attempt):
        raise DatabasePortalBridgeError("missing callback authority")
    daemon = _open_daemon(tmp_path, provider_fn=failed)
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        attempt = daemon.get_attempt(first["attempt_id"])
        before = daemon.task_source.get(attempt.task_cid)
        daemon.authority_mode = "quack"
        assert daemon.reconcile_recoverable_portal_failure_rearms() == []
        assert not daemon._portal_zero_provider_callback_rearm_ready(None)
        after = daemon.task_source.get(attempt.task_cid)
        assert after.status == "blocked" and after.revision == before.revision
    finally:
        daemon.close()


@pytest.mark.parametrize("field,value", [
    ("claim_id", "foreign-claim"), ("fencing_token", 99), ("settlement_id", "foreign"),
])
def test_callback_proof_cannot_be_borrowed_for_another_settlement(tmp_path, field, value):
    provider = _Provider(True)
    daemon = _open_daemon(tmp_path, provider_fn=provider.run)
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        attempt = daemon.get_attempt(first["attempt_id"])
        task = daemon.task_source.get(attempt.task_cid)
        forged = {**task.body["completion_receipt"], field: value}
        assert not daemon._portal_zero_provider_callback_rearm_ready(attempt, forged)
        assert provider.observed == []
        assert daemon.task_source.get(attempt.task_cid).revision == task.revision
    finally:
        daemon.close()
