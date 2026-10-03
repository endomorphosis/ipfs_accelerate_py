"""Deterministic polling boundaries plus an actual signed native lifecycle."""
import json
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
    ProcessIdentity, ProcessTreeSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.entrypoints import isolated_benchmark_runtime as subject


def identity(pid, parent, *, started=1):
    return ProcessIdentity(pid=pid, parent_pid=parent, start_time_ticks=started,
        process_group_id=100, session_id=100, boot_id="boot", argv=("python",),
        cwd="/repo", executable="/python", run_id="run", profile_id="profile",
        target_id="repo", repository_root="/repo", state_root="/state",
        run_root="/state/run", fencing_epoch=1, configuration_root="config")


ROOT, OWNER, CHILD = identity(100, 1), identity(101, 100), identity(102, 101)


def observation(*, members=(ROOT, OWNER), heartbeat=1000, healthy=True, daemon_pid=101):
    tree = ProcessTreeSnapshot(profile_id="profile", run_id="run", members=members)
    return dict(process_tree=tree.to_dict(), healthy=healthy,
        native_heartbeat=None if heartbeat is None else dict(updated_at_ms=heartbeat,
            supervisor_pid=100, daemon_pid=daemon_pid, status="running", authority=False))


def poll(monkeypatch, select, *, deadline=3., observation_delay=0., initial=None, root_cid=None,
         validation_delay=False):
    clock = SimpleNamespace(value=0.)
    def sleep(seconds): clock.value += seconds
    def observe():
        value = select(clock.value)
        clock.value += observation_delay
        return value
    monkeypatch.setattr(subject.time, "monotonic", lambda: clock.value)
    monkeypatch.setattr(subject.time, "sleep", sleep)
    if validation_delay:
        decode = ProcessTreeSnapshot.from_dict
        def slow_decode(value):
            result = decode(value)
            if clock.value >= .75:
                clock.value += .4
            return result
        monkeypatch.setattr(subject.ProcessTreeSnapshot, "from_dict", slow_decode)
    evidence = {}
    passed = subject._wait_for_stable_empty_supervisor(SimpleNamespace(observe=observe),
        initial=initial or observation(), root_cid=root_cid or cid_for_dag_json(ROOT.to_dict(), for_identity=False),
        native_start_tree=observation()["process_tree"],
        deadline=deadline, evidence=evidence)
    return passed, evidence, clock.value


def test_waits_for_delayed_heartbeat_without_extending_deadline(monkeypatch):
    passed, evidence, elapsed = poll(monkeypatch,
        lambda now: observation(heartbeat=1001 if now >= 1.3 else 1000))
    assert passed and 1.3 <= elapsed < 1.5
    assert evidence["stable_seconds"] >= .75
    assert evidence["samples"][-1]["heartbeat_ms"] > evidence["initial_heartbeat_ms"]


def test_transient_descendant_resets_window_but_keeps_owner_anchors(monkeypatch):
    def select(now):
        return observation(members=(ROOT, OWNER, CHILD) if .2 <= now < .6 else (ROOT, OWNER),
            heartbeat=1000+int(now*1000))
    passed, evidence, elapsed = poll(monkeypatch, select)
    assert passed and 1.3 <= elapsed < 1.6
    assert evidence["daemon_identity_id"] == OWNER.identity_id
    assert any(CHILD.identity_id in row["added_identity_ids"] for row in evidence["samples"])
    assert any(CHILD.identity_id in row["removed_identity_ids"] for row in evidence["samples"])


def test_unhealthy_sample_requires_an_entire_new_healthy_window(monkeypatch):
    passed, evidence, elapsed = poll(monkeypatch, lambda now: observation(
        healthy=not (.3 <= now < .6), heartbeat=None if .3 <= now < .6 else 1000+int(now*1000)))
    assert passed and elapsed >= 1.3
    assert any(not row["healthy"] for row in evidence["samples"])


@pytest.mark.parametrize("mode", ["heartbeat_stuck", "unhealthy", "churning"])
def test_nonqualifying_observations_stop_at_original_bound(monkeypatch, mode):
    def select(now):
        return observation(healthy=mode != "unhealthy", heartbeat=1000 if mode == "heartbeat_stuck" else 1000+int(now*1000),
            members=(ROOT, OWNER, CHILD) if mode == "churning" and int(now*10) % 2 else (ROOT, OWNER))
    passed, evidence, elapsed = poll(monkeypatch, select, deadline=1.5)
    assert not passed and elapsed == pytest.approx(1.5)
    assert evidence["reason"] == "deadline_exceeded" and not evidence["passed"]


@pytest.mark.parametrize("mode", ["root_reuse", "owner_reuse", "owner_restart", "owner_absent", "extra_root"])
def test_identity_drift_is_never_reanchored_even_if_later_healthy(monkeypatch, mode):
    changed = dict(root_reuse=(identity(100, 1, started=2), OWNER),
        owner_reuse=(ROOT, identity(101, 100, started=2)),
        owner_restart=(ROOT, identity(103, 100)), owner_absent=(ROOT,),
        extra_root=(ROOT, OWNER, identity(104, 1)))[mode]
    def select(now):
        return observation(members=changed if .2 <= now < .4 else (ROOT, OWNER), heartbeat=1000+int(now*1000))
    passed, evidence, elapsed = poll(monkeypatch, select)
    assert not passed and elapsed < .4
    assert evidence["reason"] == "root_or_owner_identity_changed"


@pytest.mark.parametrize("mode", ["regressed", "foreign_owner"])
def test_native_heartbeat_drift_is_not_retried_as_transience(monkeypatch, mode):
    def select(now):
        return observation(heartbeat=999 if now >= .2 and mode == "regressed" else 1000,
            daemon_pid=102 if now >= .2 and mode == "foreign_owner" else 101)
    passed, evidence, elapsed = poll(monkeypatch, select)
    assert not passed and elapsed < .4
    assert evidence["reason"] == ("native_heartbeat_regressed" if mode == "regressed" else "native_heartbeat_owner_changed")


def test_slow_observation_cannot_finish_after_deadline(monkeypatch):
    passed, evidence, elapsed = poll(monkeypatch, lambda now: observation(heartbeat=1000+int(now*1000)),
        deadline=1., observation_delay=.6)
    assert not passed and elapsed > 1.
    assert evidence["reason"] == "deadline_exceeded"


def test_start_anchor_must_match_original_root_receipt(monkeypatch):
    foreign = cid_for_dag_json(identity(100, 1, started=2).to_dict(), for_identity=False)
    with pytest.raises(ValueError, match="START root identity differs"):
        poll(monkeypatch, lambda now: observation(), root_cid=foreign)


def test_owner_restart_between_native_start_and_wrapper_observation_is_refused(monkeypatch):
    changed = observation(members=(ROOT, identity(101, 100, started=2)))
    with pytest.raises(ValueError, match="changed after the native START transition"):
        poll(monkeypatch, lambda now: changed, initial=changed)


def test_identity_decoding_cannot_cross_deadline_and_still_pass(monkeypatch):
    passed, evidence, elapsed = poll(monkeypatch, lambda now: observation(heartbeat=1000+int(now*1000)),
        deadline=1., validation_delay=True)
    assert not passed and elapsed > 1.
    assert evidence["reason"] == "deadline_exceeded"


@pytest.mark.parametrize("failure", ["record", "registry"])
def test_qualification_cleanup_releases_owners_after_receipt_failure(tmp_path, monkeypatch, failure):
    state = tmp_path / "state"; state.mkdir(); records = state / "receipts"; records.mkdir()
    initial = observation(); initial_cid = cid_for_dag_json(initial, for_identity=False)
    native = {"data": {"transition": {"new_tree": initial["process_tree"]}}}
    start_cid = cid_for_dag_json(native, for_identity=False)
    for cid, value in ((initial_cid, initial), (start_cid, native)):
        (records / (cid + ".json")).write_text(json.dumps(value))
    calls = []
    def invoke(name):
        calls.append(name)
        return SimpleNamespace(receipt_cid=start_cid if name == "start" else "stop",
            values={"old_tree_fenced": True, "health_revision_cid": initial_cid,
                "process_cid": cid_for_dag_json(ROOT.to_dict(), for_identity=False)})
    def record(*args):
        calls.append("record")
        if failure == "record": raise OSError("receipt persistence failed")
    def registry_close():
        calls.append("registry_close")
        if failure == "registry": raise OSError("registry persistence failed")
    factory = SimpleNamespace(invoke=invoke, handler_manifest=lambda: {},
        registry=SimpleNamespace(close=registry_close))
    runtime = SimpleNamespace(repository=tmp_path, state=state, repository_id="repository",
        manifest_id="configuration", local_profile=SimpleNamespace(profile_id="profile"),
        lease=SimpleNamespace(lease_id="lease"), profile=None, _record=record,
        process=SimpleNamespace(snapshot=lambda profile: SimpleNamespace(members=())),
        runtime_factory=lambda: factory, close=lambda: calls.append("runtime_close"))
    monkeypatch.setattr(subject.IsolatedBenchmarkRuntime, "create", lambda *a, **k: runtime)
    monkeypatch.setattr(subject, "_wait_for_stable_empty_supervisor", lambda *a, **k: True)
    with pytest.raises(OSError, match="persistence failed"):
        subject.qualify_empty_supervisor(tmp_path / "qualification", timeout_ms=2000)
    assert calls == ["start", "stop", "record", "registry_close", "runtime_close"]


def test_real_empty_native_qualification_keeps_start_stop_and_heartbeat_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "account-root")
    result = subject.qualify_empty_supervisor(tmp_path / "qualification", timeout_ms=20_000)
    assert result["passed"], result
    assert result["start_receipt_cid"] and result["stop_receipt_cid"]
    assert result["healthy_stable_process_tree"] and result["process_tree_absent_after_stop"]
    evidence = result["stability_observation"]
    assert evidence["passed"] and evidence["stable_seconds"] >= .75
    assert evidence["samples"][-1]["heartbeat_ms"] > evidence["initial_heartbeat_ms"]
    assert evidence["root_identity_id"] in result["observed_process_identity_ids"]
    assert evidence["daemon_identity_id"] in result["observed_process_identity_ids"]
    assert not any(result[key] for key in ("production_activation", "provider_dispatch_allowed",
        "task_admission_authority", "completion_authority"))
