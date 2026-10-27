"""Real sealed harmless children bind post-setup execution clocks."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_execution_clock import current_execution_started_at
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import append_jsonl_event
from test.api.test_ordinary_supervisor_preparing_recovery_fence import (
    _isolated_authority, _scope, _child, _claims,
)


def _iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _execution(outer, native, task, state, child, tmp_path):
    selected = time.time()
    state.active_task_started_at = _iso(selected)
    claim, lease = _claims(outer, native, task, state, child, age=0)
    for body in (claim, lease):
        body["started_at"] = _iso(selected)
    claim["dispatch_authority_cid"] = native._implementation_dispatch_authority_cid(claim)
    for path, body in ((outer._task_dispatch_claim_path(state), claim),
            (outer._implementation_maintenance_lock_path(), lease)):
        path.write_text(json.dumps(body)); path.chmod(0o600)
    workspace = outer.config.worktree_root / "workspace_clock"
    workspace.mkdir(parents=True)
    log = outer.config.state_dir / "implementation_logs/clock.log"
    log.parent.mkdir(); log.write_text("Synthetic metadata fixture; no provider invocation\n"); log.chmod(0o600)
    state.active_attempt = 3
    state.implementation_attempts[task.task_id] = 3
    state.implementation_attempts_by_cid[state.active_task_cid] = 3
    state.implementation_in_progress = True
    state.active_phase = "validating"
    state.active_phase_started_at = _iso(selected + 60)
    state.active_worktree_path = state.last_implementation_worktree_path = str(workspace)
    state.active_log_path = state.last_implementation_log_path = str(log)
    state.last_implementation_task_id = state.active_task_id
    state.last_implementation_task_cid = state.active_task_cid
    state.last_implementation_started_at = _iso(selected)
    state.last_implementation_finished_at = ""
    state.save(outer.config.state_path)
    events = outer.config.state_dir / f"{outer.config.state_prefix}_events.jsonl"
    payload = {"timestamp": _iso(selected + 51), "task_id": task.task_id,
        "canonical_task_cid": state.active_task_cid, "canonical_task_key": state.active_task_key,
        "board_namespace": claim["board_namespace"], "attempt": 3,
        "worktree_path": str(workspace), "log_path": str(log)}
    append_jsonl_event(events, "implementation_started", payload)
    return selected, events, payload, claim


def test_cold_setup_does_not_end_existing_execution_budget_before_native_callback(tmp_path):
    outer, native, task, state = _scope(tmp_path)
    with _child(outer) as child:
        selected, events, payload, _ = _execution(outer, native, task, state, child, tmp_path)
        before = asdict(state)
        raw = events.read_bytes()
        # Selection/setup is 51 seconds earlier. This reproduces the actual
        # 06:42:35 parent exit before the implementation's 06:42:50 deadline.
        observed = selected + 936
        assert current_execution_started_at(outer, state, now_ts=observed) == selected + 51
        assert outer._implementation_attempt_is_active(state, now_ts=observed)
        assert outer.is_stuck(state, now_ts=observed) == (False, "")
        assert not outer._implementation_attempt_is_active(state, now_ts=selected + 51 + 930.01)
        assert outer.is_stuck(state, now_ts=selected + 51 + 930.01)[0]
        assert asdict(state) == before and events.read_bytes() == raw


@pytest.mark.parametrize("damage", ["cid", "key", "attempt", "worktree", "log", "namespace", "future", "replay", "outcome", "chain", "claim_birth", "repository", "alias"])
def test_foreign_replayed_or_unfinished_clock_evidence_cannot_extend_old_bound(tmp_path, damage):
    outer, native, task, state = _scope(tmp_path)
    with _child(outer) as child:
        selected, events, payload, claim = _execution(outer, native, task, state, child, tmp_path)
        observed = selected + 936
        if damage in {"cid", "key", "attempt", "worktree", "log", "namespace", "future"}:
            key, value = {"cid": ("canonical_task_cid", "foreign"), "key": ("canonical_task_key", "foreign"),
                "attempt": ("attempt", 2), "worktree": ("worktree_path", str(tmp_path / "foreign")),
                "log": ("log_path", str(tmp_path / "foreign.log")), "namespace": ("board_namespace", "foreign"),
                "future": ("timestamp", _iso(observed + 5))}[damage]
            payload[key] = value
            events.unlink(); events.with_name(events.name + ".manifest.json").unlink()
            append_jsonl_event(events, "implementation_started", payload)
        elif damage == "replay":
            append_jsonl_event(events, "implementation_started", {**payload, "timestamp": _iso(selected + 700)})
        elif damage == "outcome":
            append_jsonl_event(events, "implementation_finished", {**payload, "timestamp": _iso(selected + 100)})
        elif damage == "chain":
            event = json.loads(events.read_text()); event["timestamp"] = _iso(selected + 700)
            events.write_text(json.dumps(event) + "\n")
        elif damage in {"claim_birth", "repository"}:
            if damage == "claim_birth": claim["owner_process_birth"]["start_time_ticks"] += 1
            else: claim["repository_id"] = "repository:foreign"
            claim["dispatch_authority_cid"] = native._implementation_dispatch_authority_cid(claim)
            outer._task_dispatch_claim_path(state).write_text(json.dumps(claim))
        else:
            target = events.with_suffix(".preserved")
            events.rename(target); events.symlink_to(target)
        before = asdict(state)
        assert current_execution_started_at(outer, state, now_ts=observed) is None
        assert not outer._implementation_attempt_is_active(state, now_ts=observed)
        assert asdict(state) == before


def test_cached_execution_clock_rechecks_live_claim_and_keeps_absolute_expiry(tmp_path):
    outer, native, task, state = _scope(tmp_path)
    with _child(outer) as child:
        selected, _, _, claim = _execution(outer, native, task, state, child, tmp_path)
        observed = selected + 936
        assert current_execution_started_at(outer, state, now_ts=observed) == selected + 51
        assert current_execution_started_at(outer, state, now_ts=observed + 1) == selected + 51
        assert not outer._implementation_attempt_is_active(state, now_ts=selected + 51 + 930.01)
        claim["owner_process_birth"]["start_time_ticks"] += 1
        outer._task_dispatch_claim_path(state).write_text(json.dumps(claim))
        assert current_execution_started_at(outer, state, now_ts=observed) is None
