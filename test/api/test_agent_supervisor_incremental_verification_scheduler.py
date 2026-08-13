from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
    ProcessTreeSnapshot,
)
from scripts.ops.agent_supervisor import (
    incremental_verification_planner_scheduler as scheduler,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "config/agent_supervisor_incremental_verification_planner_scheduler.json"
)


def _write_config(tmp_path: Path, mutate=None) -> Path:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(payload)
    path = tmp_path / "scheduler.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_ivp_launch_plan_binds_strict_shards_controls_and_quota_only_route():
    board = scheduler.load_board(CONFIG_PATH, repo_root=REPO_ROOT)

    plan = scheduler.launch_plan(
        board,
        implement=True,
        foreground=False,
        duration_seconds=3600,
        stamp="20260811T170000Z",
    )

    assert plan["lanes"] == 3
    assert plan["strict_task_sharding"] is True
    assert len(plan["source_head"]) == 40
    assert plan["environment"] == {
        scheduler.PROVIDER_ENV: "grok_cli",
        scheduler.FALLBACK_PROVIDER_ENV: "codex",
        scheduler.FALLBACK_TRIGGER_ENV: "primary_quota_exhausted",
        scheduler.GROK_MODEL_ENV: "grok-4.5",
        scheduler.CODEX_MODEL_ENV: "gpt-5.6-terra",
        scheduler.CODEX_REASONING_EFFORT_ENV: "high",
        scheduler.PROVIDER_FALLBACK_POLICY_ENV: "grok_quota_only",
    }
    argv = plan["argv"]
    assert argv.count("--implementation-supervisor-lanes-per-track") == 1
    lane_index = argv.index("--implementation-supervisor-lanes-per-track")
    assert argv[lane_index + 1] == "3"
    assert "--common-arg=--strict-task-sharding" in argv
    assert "--common-arg=--implement" in argv
    assert "--common-arg=## IVP-" in argv
    assert "--common-arg=--no-objective-task-janitor" in argv
    for relative in board.protected_paths:
        index = argv.index("--common-arg=--implementation-protected-path")
        assert f"--common-arg={relative}" in argv[index + 1 :]
    for relative in board.worktree_submodule_paths:
        assert f"--common-arg={relative}" in argv


def test_ivp_launch_requires_explicit_implementation_authority():
    board = scheduler.load_board(CONFIG_PATH, repo_root=REPO_ROOT)
    with pytest.raises(scheduler.IVPSchedulerError, match="explicit --implement"):
        scheduler.launch_plan(
            board,
            implement=False,
            foreground=True,
            duration_seconds=60,
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload["provider"].__setitem__(
            "fallback_trigger", "primary_quota_or_auth_unavailable"
        ),
        lambda payload: payload["provider"].__setitem__(
            "fallback_reasoning_effort", "medium"
        ),
        lambda payload: payload.__setitem__("strict_task_sharding", False),
        lambda payload: payload["source_binding"].__setitem__(
            "bootstrap_task_source", "duckdb"
        ),
    ),
)
def test_ivp_loader_rejects_route_or_authority_drift(tmp_path: Path, mutate):
    path = _write_config(tmp_path, mutate)
    with pytest.raises(scheduler.IVPSchedulerError):
        scheduler.load_board(path, repo_root=tmp_path)


def test_ivp_config_has_latest_pr_ancestor_and_launcher_fence():
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["source_binding"]["accelerator_required_ancestor"] == (
        "c1e33e8f443253e106c464d7c5b5c341c3095876"
    )
    assert payload["source_binding"]["accelerator_required_branch"] == (
        "integration/incremental-verification-planner-main-20260811"
    )
    assert (
        "scripts/ops/agent_supervisor/incremental_verification_planner_scheduler.py"
        in payload["protected_paths"]
    )


def _identity(profile, pid: int, *, parent_pid: int = 1):
    return scheduler.ProcessIdentity(
        pid=pid,
        start_time_ticks=pid,
        parent_pid=parent_pid,
        process_group_id=pid,
        session_id=pid,
        boot_id="test-boot",
        argv=profile.argv,
        cwd=profile.cwd,
        executable=str(Path(sys.executable).resolve()),
        run_id=profile.run_id,
        profile_id=profile.profile_id,
        target_id=profile.target_id,
        repository_root=profile.repository_root,
        state_root=profile.state_root,
        run_root=profile.run_root,
        fencing_epoch=0,
        configuration_root=profile.configuration_root,
    )


class _FakeProcessAdapter:
    def __init__(self, *, launch_parent_pid: int = 1):
        self.trees = {}
        self.next_pid = 999_991
        self.launch_parent_pid = launch_parent_pid

    def snapshot(self, profile):
        members = tuple(self.trees.get(profile.profile_id, ()))
        return ProcessTreeSnapshot(
            profile_id=profile.profile_id,
            run_id=profile.run_id,
            members=members,
            captured_at_ms=int(time.time() * 1000),
        )

    def launch(self, profile, *, fencing_epoch):
        assert fencing_epoch == 0
        identity = _identity(
            profile,
            self.next_pid,
            parent_pid=self.launch_parent_pid,
        )
        self.trees[profile.profile_id] = (identity,)
        return identity

    def identity_alive(self, identity):
        return any(
            member.identity_id == identity.identity_id
            for members in self.trees.values()
            for member in members
        )

    def terminate(self, tree, *, grace_seconds, deadline_ms):
        assert grace_seconds > 0
        assert deadline_ms > 0
        self.trees.pop(tree.profile_id, None)


def _tmp_board(tmp_path: Path):
    path = _write_config(tmp_path)
    subprocess.run(
        [
            "git",
            "init",
            "-q",
            "-b",
            "integration/incremental-verification-planner-main-20260811",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=IVP Test",
            "-c",
            "user.email=ivp@example.invalid",
            "add",
            "scheduler.json",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=IVP Test",
            "-c",
            "user.email=ivp@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    return scheduler.load_board(path, repo_root=tmp_path)


def _terminal_lane_files(board, plan, *, count=20):
    for track, _profile in scheduler._lane_profiles(board, plan):
        status_path = track.supervisor_status_path
        assert status_path is not None
        prefix = status_path.name.removesuffix("_supervisor_status.json")
        task_path = status_path.parent / f"{prefix}_task_state.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps({"status": "stopped"}), encoding="utf-8")
        task_path.write_text(
            json.dumps(
                {
                    "task_count": count,
                    "completed_count": count,
                    "ready_count": 0,
                    "selectable_ready_count": 0,
                    "eligible_ready_count": 0,
                    "external_reserved_count": 0,
                    "waiting_count": 0,
                    "blocked_count": 0,
                    "active_task_id": "",
                    "implementation_in_progress": False,
                    "heartbeat_at": scheduler._utc_now(),
                    "last_progress_at": scheduler._utc_now(),
                    "active_phase": "",
                    "active_phase_started_at": "",
                    "active_phase_detail": "",
                    "last_implementation_returncode": 0,
                    "last_implementation_commit": "deadbeef",
                    "last_merge_returncode": 0,
                    "last_merge_commit": "cafebabe",
                    "last_merge_error": "",
                    "selection_idle_reason": "no_tasks_found",
                }
            ),
            encoding="utf-8",
        )


def test_launch_persists_exact_identity_and_status_stop_use_it(tmp_path, monkeypatch):
    board = _tmp_board(tmp_path)
    plan = scheduler.launch_plan(
        board,
        implement=True,
        foreground=False,
        duration_seconds=60,
        stamp="20260811T180000Z",
        expected_task_count=20,
    )
    adapter = _FakeProcessAdapter()
    for name in plan["environment"]:
        monkeypatch.delenv(name, raising=False)

    assert scheduler.run_launch(board, plan, adapter=adapter) == 0
    record = json.loads(board.lifecycle_path.read_text(encoding="utf-8"))
    profile = scheduler.LifecycleProfile.from_dict(record["profile"])
    identity = scheduler.ProcessIdentity.from_dict(record["identity"])
    assert identity.profile_id == profile.profile_id
    assert identity.run_id == profile.run_id
    board.master_pid_path.write_text(f"{identity.pid}\n", encoding="ascii")

    running = scheduler.status(board, adapter=adapter)
    assert running["lifecycle"] == "running"
    assert running["master_root_pids"] == [identity.pid]

    lock_fd = os.open(board.lifecycle_lock_path, os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(scheduler.IVPSchedulerError, match="launch/stop"):
            scheduler.stop(board, grace_seconds=1, adapter=adapter)
    finally:
        os.close(lock_fd)
    stopped = scheduler.stop(board, grace_seconds=1, adapter=adapter)
    assert stopped["fenced"] is True
    assert scheduler.status(board, adapter=adapter)["lifecycle"] == "stopped"


def _identity_with(identity, **changes):
    payload = identity.to_dict()
    payload.update(changes)
    payload.pop("identity_id")
    return scheduler.ProcessIdentity.from_dict(payload)


def test_status_accepts_exact_master_after_launcher_reparenting(
    tmp_path, monkeypatch
):
    board = _tmp_board(tmp_path)
    plan = scheduler.launch_plan(
        board,
        implement=True,
        foreground=False,
        duration_seconds=60,
        stamp="20260811T180050Z",
        expected_task_count=20,
    )
    adapter = _FakeProcessAdapter(launch_parent_pid=43210)
    for name in plan["environment"]:
        monkeypatch.delenv(name, raising=False)

    assert scheduler.run_launch(board, plan, adapter=adapter) == 0
    record = json.loads(board.lifecycle_path.read_text(encoding="utf-8"))
    profile = scheduler.LifecycleProfile.from_dict(record["profile"])
    launched = scheduler.ProcessIdentity.from_dict(record["identity"])
    adopted = _identity_with(launched, parent_pid=1)
    assert adopted.identity_id != launched.identity_id
    adapter.trees[profile.profile_id] = (adopted,)
    board.master_pid_path.write_text(f"{launched.pid}\n", encoding="ascii")

    running = scheduler.status(board, adapter=adapter)
    assert running["lifecycle"] == "running"
    assert running["healthy"] is True
    assert running["issues"] == []


@pytest.mark.parametrize(
    ("changes", "case"),
    (
        ({"parent_pid": 1, "start_time_ticks": 999_992}, "PID reuse"),
        ({"parent_pid": 1, "fencing_epoch": 1}, "lifecycle fence drift"),
    ),
)
def test_status_rejects_non_parent_master_identity_drift(
    tmp_path, monkeypatch, changes, case
):
    board = _tmp_board(tmp_path)
    plan = scheduler.launch_plan(
        board,
        implement=True,
        foreground=False,
        duration_seconds=60,
        stamp="20260811T180055Z",
        expected_task_count=20,
    )
    adapter = _FakeProcessAdapter(launch_parent_pid=43210)
    for name in plan["environment"]:
        monkeypatch.delenv(name, raising=False)

    assert scheduler.run_launch(board, plan, adapter=adapter) == 0
    record = json.loads(board.lifecycle_path.read_text(encoding="utf-8"))
    profile = scheduler.LifecycleProfile.from_dict(record["profile"])
    launched = scheduler.ProcessIdentity.from_dict(record["identity"])
    drifted = _identity_with(launched, **changes)
    adapter.trees[profile.profile_id] = (drifted,)
    board.master_pid_path.write_text(f"{launched.pid}\n", encoding="ascii")

    report = scheduler.status(board, adapter=adapter)
    assert report["lifecycle"] == "unhealthy", case
    assert report["healthy"] is False
    assert "recorded master identity is not live" in report["issues"]


def test_terminal_receipt_requires_fresh_complete_lane_evidence(tmp_path, monkeypatch):
    board = _tmp_board(tmp_path)
    plan = scheduler.launch_plan(
        board,
        implement=True,
        foreground=False,
        duration_seconds=60,
        stamp="20260811T180100Z",
        expected_task_count=20,
    )
    adapter = _FakeProcessAdapter()
    for name in plan["environment"]:
        monkeypatch.delenv(name, raising=False)
    scheduler.run_launch(board, plan, adapter=adapter)
    record = json.loads(board.lifecycle_path.read_text(encoding="utf-8"))
    profile = scheduler.LifecycleProfile.from_dict(record["profile"])
    identity = scheduler.ProcessIdentity.from_dict(record["identity"])
    board.master_pid_path.write_text(f"{identity.pid}\n", encoding="ascii")
    _terminal_lane_files(board, plan)
    launched_at = scheduler._parse_time(record["launched_at"])
    assert launched_at is not None
    terminal, lanes = scheduler._terminal_evidence(
        board,
        plan,
        adapter=adapter,
        launched_at=launched_at,
        require_live_lanes=False,
    )
    assert terminal is True
    assert len(lanes) == 3
    assert lanes[0]["last_progress_at"]
    assert lanes[0]["last_implementation_commit"] == "deadbeef"
    assert lanes[0]["last_merge_returncode"] == 0
    assert lanes[0]["active_progress_hard_timeout_exceeded"] is False

    adapter.trees.pop(profile.profile_id)
    board.master_pid_path.unlink()
    scheduler._atomic_json(
        board.terminal_path,
        {
            "schema": scheduler.TERMINAL_RECEIPT_SCHEMA,
            "run_id": profile.run_id,
            "profile_id": profile.profile_id,
            "configuration_root": profile.configuration_root,
            "drained": True,
        },
    )
    assert scheduler.status(board, adapter=adapter)["lifecycle"] == "completed"

    first_task = Path(lanes[0]["task_state_path"])
    payload = json.loads(first_task.read_text(encoding="utf-8"))
    payload["completed_count"] = 19
    first_task.write_text(json.dumps(payload), encoding="utf-8")
    assert scheduler.status(board, adapter=adapter)["lifecycle"] != "completed"


def test_launch_rejects_source_head_race(tmp_path, monkeypatch):
    board = _tmp_board(tmp_path)
    plan = scheduler.launch_plan(
        board,
        implement=True,
        foreground=False,
        duration_seconds=60,
        stamp="20260811T180200Z",
        expected_task_count=20,
    )
    for name in plan["environment"]:
        monkeypatch.delenv(name, raising=False)
    plan["source_head"] = "0" * 40
    plan["configuration_root"] = scheduler._plan_configuration_root(plan)
    with pytest.raises(scheduler.IVPSchedulerError, match="source HEAD changed"):
        scheduler.run_launch(board, plan, adapter=_FakeProcessAdapter())


def test_provider_preflight_rejects_ambient_drift_and_proves_quota_only(
    tmp_path, monkeypatch
):
    board = _tmp_board(tmp_path)
    monkeypatch.setenv(scheduler.PROVIDER_ENV, "codex")
    with pytest.raises(scheduler.IVPSchedulerError, match="ambient provider policy"):
        scheduler._provider_preflight(board)

    for name in scheduler.PROVIDER_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("IMPLEMENTATION_DAEMON_COMMAND", "unauthorized --override")
    with pytest.raises(scheduler.IVPSchedulerError, match="command override"):
        scheduler._provider_preflight(board)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND")
    monkeypatch.setattr(
        scheduler,
        "_probe_provider_readiness",
        lambda _environment: SimpleNamespace(
            grok_ready=True,
            codex_ready=True,
            effective_provider="grok",
            reason_code="grok_ready",
            grok_model="grok-4.5",
            codex_model="gpt-5.6-terra",
            codex_reasoning_effort="high",
        ),
    )
    expected_runner = (
        board.repo_root
        / "ipfs_accelerate_py/agent_supervisor/provider_fallback_runner.py"
    ).resolve()
    monkeypatch.setattr(
        scheduler,
        "_build_provider_route_command",
        lambda _board, _expected: [
            sys.executable,
            str(expected_runner),
            "--fallback-policy",
            "grok_quota_only",
        ],
    )
    report = scheduler._provider_preflight(board)
    assert report["fallback_policy"] == "grok_quota_only"
    assert report["grok_ready"] is report["codex_ready"] is True
