"""Native coordinator health uncertainty cannot manufacture track closure."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as runner,
)
from ipfs_accelerate_py.agent_supervisor.merge import worktree_lifecycle as births
from test.api.test_agent_supervisor_managed_quack_owner_integration import (
    _Lifecycle,
    _patch_track_runtime,
    _track,
)


def run(tmp_path, monkeypatch, values, *, fenced=True):
    events = []
    lines = []
    lifecycle = _Lifecycle(events, values)
    _patch_track_runtime(monkeypatch, events)
    if not fenced:

        def stop(*args, **kwargs):
            events.append("track.retained")
            return {
                "all_trees_fenced": False,
                "stopped_count": 0,
                "removed_runtime_markers": [],
            }

        monkeypatch.setattr(runner, "stop_tracks", stop)
    result = runner.run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=(),
        duration_seconds=0.06,
        heartbeat_interval_seconds=0.005,
        managed_quack_owner=lifecycle,
        output=lines.append,
    )
    return result, events, lines


@pytest.mark.parametrize("health", ["unknown", "unhealthy"])
def test_uncertain_probe_then_healthy_keeps_exact_tracks_until_run_end(
    tmp_path, monkeypatch, health
):
    result, events, lines = run(tmp_path, monkeypatch, [health, "healthy"])
    assert result["completed"] is True
    assert events.count("track.start") == 1
    assert events.count("track.stop") == 1
    assert events.count("owner.health") >= 2
    assert "owner.recover" not in events
    assert events[-2:] == ["track.stop", "owner.shutdown"]
    assert result["managed_quack_owner"]["deferred_health_count"] == 1
    assert any("preserving owner and tracks" in line for line in lines)


@pytest.mark.parametrize("health", ["unknown", "unhealthy"])
def test_persistent_uncertainty_only_fences_at_explicit_finite_deadline(
    tmp_path, monkeypatch, health
):
    result, events, _ = run(tmp_path, monkeypatch, [health] * 100)
    assert result["completed"] is True
    assert events.count("owner.health") >= 2
    assert events.count("track.start") == events.count("track.stop") == 1
    assert events[-2:] == ["track.stop", "owner.shutdown"]
    assert "owner.recover" not in events
    assert result["managed_quack_owner"]["deferred_health_count"] == events.count(
        "owner.health"
    )


@pytest.mark.parametrize("health", ["unknown", "unhealthy"])
@pytest.mark.parametrize("recovers", [False, True])
def test_health_uncertainty_blocks_exited_track_rearm_between_samples(
    tmp_path, monkeypatch, health, recovers
):
    events = []
    lifecycle = _Lifecycle(events, [health, "healthy" if recovers else health])
    lifecycle.health_check_interval_seconds = 0.08
    _patch_track_runtime(monkeypatch, events)
    launches = []
    fences = []

    def launch(*args, **kwargs):
        sample_count = events.count("owner.health")
        launches.append(sample_count)
        events.append("track.start")
        return SimpleNamespace(
            pid=43210,
            # The first track exits just after the first uncertain sample.
            poll=lambda: 0 if sample_count == 0 and "owner.health" in events else None,
        )

    def fence(*args, **kwargs):
        fences.append(events.count("owner.health"))
        return True, (43210,)

    monkeypatch.setattr(runner, "start_track", launch)
    monkeypatch.setattr(runner, "_terminate_managed_process", fence)
    monkeypatch.setattr(
        runner, "_restarting_track_must_preserve_extra_gate_grok", lambda *a, **k: False
    )
    result = runner.run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=(),
        duration_seconds=0.21,
        heartbeat_interval_seconds=0.05,
        managed_quack_owner=lifecycle,
        output=lambda line: None,
    )
    assert result["completed"] is True
    assert events.count("owner.health") == 2
    assert launches == ([0, 2] if recovers else [0])
    assert fences == ([2] if recovers else [])
    assert events[-2:] == ["track.stop", "owner.shutdown"]
    assert "owner.recover" not in events


@pytest.mark.parametrize("health", ["healthy", "unknown", "dead"])
def test_retained_track_custody_never_stops_required_owner(
    tmp_path, monkeypatch, health
):
    result, events, _ = run(tmp_path, monkeypatch, [health] * 100, fenced=False)
    assert result["completed"] is False
    assert result["all_trees_fenced"] is False
    assert "owner.shutdown" not in events and "owner.recover" not in events
    assert events.count("track.start") == 1
    assert result["managed_quack_owner"]["shutdown"] == {
        "stopped": False,
        "deferred": True,
        "reason": "track_custody_not_closed",
    }


def test_unverified_tree_is_not_reported_as_whole_tree_fencing(tmp_path, monkeypatch):
    track = _track(tmp_path)
    process = SimpleNamespace(pid=54321)
    monkeypatch.setattr(runner, "_terminate_managed_process", lambda *a, **k: (False, ()))
    monkeypatch.setattr(runner, "_remove_track_runtime_markers", lambda *a, **k: pytest.fail("unknown tree cannot retire markers"), raising=False)
    result = runner.stop_tracks(
        [track], {"lane": process}, repo_root=tmp_path, output=lambda message: None
    )
    assert result["all_trees_fenced"] is False
    assert result["stopped_pids"] == [] and result["removed_runtime_markers"] == []


@pytest.mark.parametrize(
    "phase", ["status_identity", "process_birth", "authenticated_readiness"]
)
def test_native_health_records_redacted_failure_boundary(monkeypatch, phase):
    lifecycle = object.__new__(runner.ManagedLocalQuackOwnerLifecycle)
    birth = births.current_process_birth()
    binding = SimpleNamespace(to_dict=lambda: {"generation": 7})
    monkeypatch.setattr(lifecycle, "_status_identity", lambda **kwargs: ({}, {}))
    monkeypatch.setattr(lifecycle, "_birth_from_identity", lambda value: birth)
    monkeypatch.setattr(lifecycle, "_binding_from_identity", lambda value: binding)
    monkeypatch.setattr(
        lifecycle,
        "_authenticated_readiness_once",
        lambda owner: SimpleNamespace(binding=binding),
    )

    def failed(*a, **k):
        raise OSError("private token and path must not escape")

    if phase == "status_identity":
        monkeypatch.setattr(lifecycle, "_status_identity", failed)
    elif phase == "process_birth":
        monkeypatch.setattr(births, "read_process_birth", failed)
    else:
        monkeypatch.setattr(lifecycle, "_authenticated_readiness_once", failed)
    observed = lifecycle.health()
    assert observed["health"] == (
        "unhealthy" if phase == "authenticated_readiness" else "unknown"
    )
    assert observed["probe_diagnostic"] == {"phase": phase, "error_class": "OSError"}
    assert "private" not in json.dumps(observed) and "token" not in json.dumps(observed)


def test_healthy_recheck_clears_only_prior_probe_error(monkeypatch):
    lifecycle = object.__new__(runner.ManagedLocalQuackOwnerLifecycle)
    birth = births.current_process_birth()
    binding = SimpleNamespace(to_dict=lambda: {"generation": 7})
    monkeypatch.setattr(lifecycle, "_status_identity", lambda **kwargs: ({}, {}))
    monkeypatch.setattr(lifecycle, "_birth_from_identity", lambda value: birth)
    monkeypatch.setattr(lifecycle, "_binding_from_identity", lambda value: binding)

    def failed(owner):
        raise TimeoutError("do not record raw probe details")

    monkeypatch.setattr(lifecycle, "_authenticated_readiness_once", failed)
    assert lifecycle.health()["probe_diagnostic"]["error_class"] == "TimeoutError"
    monkeypatch.setattr(
        lifecycle,
        "_authenticated_readiness_once",
        lambda owner: SimpleNamespace(binding=binding),
    )
    healthy = lifecycle.health()
    assert healthy["health"] == "healthy" and healthy["probe_diagnostic"] == {}
