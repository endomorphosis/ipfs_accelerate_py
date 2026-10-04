from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation
from ipfs_accelerate_py.agent_supervisor.entrypoints.isolated_benchmark_runtime import (
    IsolatedBenchmarkRuntime, empty_supervisor_argv,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.runtime_factory import MissingRuntimeHandlerError


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    # A real account lifecycle registry and Ed25519 chain in an isolated test
    # root; no fake signatures, profile trust booleans, permits or process API.
    monkeypatch.setattr(profile_authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "account-root")
    value = IsolatedBenchmarkRuntime.create(tmp_path / "qualification", timeout_ms=20_000)
    yield value
    if value.process.snapshot(value.profile).members:
        assert value.stop().succeeded
    value.close()


def test_real_signed_native_start_observe_stop(runtime):
    factory = runtime.runtime_factory()
    assert {k for k, v in factory.handler_manifest().items() if v} == {"start", "observe", "stop"}
    assert not factory.production
    with pytest.raises(MissingRuntimeHandlerError):
        factory.invoke("materialize")
    started = factory.invoke("start")
    assert started.values["process_cid"]
    assert started.values["lease_id"] == runtime.lease.lease_id
    observation = factory.invoke("observe").values
    assert observation["healthy"]
    members = observation["process_tree"]["members"]
    assert len(members) >= 2
    assert all(item["start_time_ticks"] > 0 and item["boot_id"] for item in members)
    assert all(item["fencing_epoch"] == runtime.lease.fence_epoch for item in members)
    assert observation["completion_authority"] is False
    assert observation["task_admission_authority"] is False
    assert observation["provider_dispatch_allowed"] is False
    assert observation["production_activation"] is False
    assert observation["process_effect_applied"] is False
    status = json.loads((runtime.state / "run" / "isolated_supervisor_status.json").read_text())
    assert status["status"] == "running"
    assert status["supervisor_pid"] in {item["pid"] for item in members}
    assert status["daemon_pid"] in {item["pid"] for item in members}
    assert status["log_path"].split("/")[-1].startswith("isolated_managed_daemon_")
    assert factory.invoke("stop").values["old_tree_fenced"] is True
    assert runtime.observe()["process_tree"]["members"] == []
    assert not runtime.observe()["healthy"]
    assert (runtime.state / "control-audit.jsonl").is_file()
    assert (runtime.state / "lifecycle-transitions.jsonl").is_file()
    factory.registry.close()


def test_foreign_run_request_cannot_borrow_local_permit(runtime):
    request = runtime.request(Operation.START)
    foreign = replace(request, parameters={**request.parameters, "run_id": "foreign-run"})
    result = runtime.service.execute(foreign)
    assert not result.succeeded
    assert "exact locally issued" in result.error.message
    assert runtime.process.snapshot(runtime.profile).members == ()


def test_stale_native_token_is_refused_before_launch(runtime):
    original = runtime.lease
    request = runtime.request(Operation.START)
    runtime.lease = replace(original, fencing_token=original.fencing_token + 1)
    try:
        result = runtime.service.execute(request)
        assert not result.succeeded
        assert "stale fencing" in result.error.message
        assert runtime.process.snapshot(runtime.profile).members == ()
    finally:
        runtime.lease = original


def test_changed_empty_queue_does_not_gain_task_authority(runtime):
    source = runtime.repository / "tasks.todo.md"
    original = source.read_text()
    source.write_text(original + "\n## ISOLATED-1\nStatus: ready\n")
    try:
        with pytest.raises(ValueError, match="empty qualification queue changed"):
            runtime.start()
        assert not runtime.process.snapshot(runtime.profile).members
    finally:
        source.write_text(original)


def test_revoked_local_profile_is_refused_before_launch(runtime):
    marker = runtime.profile_dir / profile_authority.REVOKE_MARKER
    marker.write_text("revoked for qualification\n")
    marker.chmod(0o600)
    try:
        with pytest.raises(profile_authority.LocalProfileRevoked):
            runtime.start()
        assert not runtime.process.snapshot(runtime.profile).members
    finally:
        marker.unlink()


def test_health_rejects_stale_or_foreign_native_status(runtime, monkeypatch):
    assert runtime.start().succeeded
    tree = runtime.process.snapshot(runtime.profile)
    path = runtime.state / "run" / "isolated_supervisor_status.json"
    status = json.loads(path.read_text())
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    # Only the observational file read is replaced; the exact process tree,
    # root/child birth identities and identity_alive probes remain real.
    real_read = type(path).read_text
    value = dict(status)

    def read(path_self, *args, **kwargs):
        return json.dumps(value) if path_self == path else real_read(path_self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(type(path), "read_text", read)
        value["supervisor_pid"] = 1
        assert not runtime.process.healthy(runtime.profile, tree, fencing_epoch=runtime.lease.fence_epoch, now_ms=now)
        value.clear(); value.update(status)
        value["updated_at"] = "2000-01-01T00:00:00+00:00"
        assert not runtime.process.healthy(runtime.profile, tree, fencing_epoch=runtime.lease.fence_epoch, now_ms=now)
        value.clear(); value.update(status)
        value["daemon_pid"] = status["supervisor_pid"]
        assert not runtime.process.healthy(runtime.profile, tree, fencing_epoch=runtime.lease.fence_epoch, now_ms=now)
    assert runtime.stop().succeeded


def test_launch_uses_parser_supported_exact_nonimplementing_flags(runtime):
    argv = empty_supervisor_argv(runtime.repository, runtime.state)
    assert argv == runtime.profile.argv
    assert "--no-implement" in argv
    assert "--implement" not in argv
    assert "--profile" not in argv
    assert "--run-namespace" not in argv
    assert argv[argv.index("--max-restarts") + 1] == "1"


def test_existing_directory_is_never_adopted(tmp_path):
    with pytest.raises(ValueError, match="new directory"):
        IsolatedBenchmarkRuntime.create(tmp_path)
