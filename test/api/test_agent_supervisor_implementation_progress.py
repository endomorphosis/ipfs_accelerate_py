from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import pid_alive
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    IMPLEMENTATION_CHECKPOINT_DIR_ENV,
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    run_process_group_stream,
)


def _task(*, metadata: dict[str, str] | None = None) -> PortalTask:
    return PortalTask(
        task_id="SRT-014",
        title="Run a resumable provider-backed matrix",
        status="todo",
        completion="manual",
        priority="P0",
        track="benchmark",
        outputs=["report.json"],
        validation=["python validate.py"],
        acceptance="Every frozen coordinate is represented exactly once.",
        metadata=metadata or {},
    )


def _daemon(tmp_path: Path, *, timeout: float = 1800.0) -> PortalImplementationDaemon:
    return PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        implementation_timeout=timeout,
        implementation_log_dir=tmp_path / "state" / "logs",
    )


def _seed_repository(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Supervisor Test"],
        cwd=path,
        check=True,
    )
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture"],
        cwd=path,
        check=True,
    )


def test_provider_task_gets_bounded_progress_aware_timeout(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path)

    ordinary = daemon._implementation_timeout_policy(_task())
    provider = daemon._implementation_timeout_policy(
        _task(metadata={"requires provider": "true"})
    )
    extended = daemon._implementation_timeout_policy(
        _task(metadata={"implementation timeout seconds": "7200"})
    )
    explicit = daemon._implementation_timeout_policy(
        _task(
            metadata={
                "requires provider": "true",
                "implementation timeout seconds": "3600",
                "implementation progress timeout seconds": "300",
                "implementation max timeout seconds": "5400",
            }
        )
    )

    assert ordinary.progress_aware is False
    assert ordinary.max_timeout_seconds == 1800.0
    assert provider.progress_aware is True
    assert provider.progress_timeout_seconds == 1800.0
    assert provider.max_timeout_seconds == 7200.0
    assert provider.source == "provider_task_progress"
    assert extended.progress_timeout_seconds == 1800.0
    assert extended.max_timeout_seconds == 7200.0
    assert extended.source == "task_metadata"
    assert explicit.progress_timeout_seconds == 300.0
    assert explicit.max_timeout_seconds == 5400.0
    assert explicit.source == "task_metadata"


def test_progress_output_renews_idle_deadline_but_not_hard_cap(
    tmp_path: Path,
) -> None:
    completed_log = tmp_path / "completed.log"
    completed_script = (
        "import sys, time\n"
        "print(sys.stdin.read(), flush=True)\n"
        "for index in range(8):\n"
        " print(index, flush=True)\n"
        " time.sleep(0.08)\n"
    )
    progress_events: list[dict[str, object]] = []
    with completed_log.open("w", encoding="utf-8") as log_fh:
        completed = run_process_group_stream(
            [sys.executable, "-c", completed_script],
            cwd=tmp_path,
            stdout=log_fh,
            input_text="checkpoint prompt",
            timeout_seconds=2.0,
            progress_timeout_seconds=0.15,
            max_timeout_seconds=2.0,
            progress_poll_seconds=0.02,
            on_progress=lambda value: progress_events.append(dict(value)),
        )
    assert completed.returncode == 0
    assert progress_events
    assert "checkpoint prompt" in completed_log.read_text(encoding="utf-8")

    hard_cap_log = tmp_path / "hard-cap.log"
    hard_cap_script = (
        "import time\n"
        "for index in range(100):\n"
        " print(index, flush=True)\n"
        " time.sleep(0.03)\n"
    )
    with hard_cap_log.open("w", encoding="utf-8") as log_fh:
        with pytest.raises(subprocess.TimeoutExpired) as raised:
            run_process_group_stream(
                [sys.executable, "-c", hard_cap_script],
                cwd=tmp_path,
                stdout=log_fh,
                timeout_seconds=0.45,
                progress_timeout_seconds=0.15,
                max_timeout_seconds=0.45,
                progress_poll_seconds=0.02,
                termination_grace_seconds=0.05,
            )
    assert getattr(raised.value, "timeout_reason") == "hard_timeout"
    assert getattr(raised.value, "progress_events") > 0


def test_absolute_timeout_output_emits_progress_without_extending_deadline(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "absolute.log"
    script = (
        "import time\n"
        "for index in range(100):\n"
        " print(index, flush=True)\n"
        " time.sleep(0.03)\n"
    )
    progress_events: list[dict[str, object]] = []
    with log_path.open("w", encoding="utf-8") as log_fh:
        with pytest.raises(subprocess.TimeoutExpired) as raised:
            run_process_group_stream(
                [sys.executable, "-c", script],
                cwd=tmp_path,
                stdout=log_fh,
                timeout_seconds=0.25,
                on_progress=lambda value: progress_events.append(dict(value)),
                progress_poll_seconds=0.02,
                termination_grace_seconds=0.05,
            )

    assert getattr(raised.value, "timeout_reason") == "absolute_timeout"
    assert getattr(raised.value, "progress_events") > 0
    assert progress_events


def test_silent_process_hits_progress_idle_timeout(tmp_path: Path) -> None:
    log_path = tmp_path / "silent.log"
    with log_path.open("w", encoding="utf-8") as log_fh:
        with pytest.raises(subprocess.TimeoutExpired) as raised:
            run_process_group_stream(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                cwd=tmp_path,
                stdout=log_fh,
                timeout_seconds=2.0,
                progress_timeout_seconds=0.15,
                max_timeout_seconds=2.0,
                progress_poll_seconds=0.02,
                termination_grace_seconds=0.05,
            )
    assert getattr(raised.value, "timeout_reason") == "progress_idle_timeout"


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").is_dir(),
    reason="process-tree interruption regression requires Linux process sessions",
)
def test_streamed_runner_fences_child_tree_when_owner_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_pid_path = tmp_path / "provider-child.pid"
    script = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(60)'], "
        "start_new_session=True"
        "); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )
    launched: list[subprocess.Popen[str]] = []
    real_launch = supervisor_runtime.launch_process_child

    def launch_and_interrupt(*args: object, **kwargs: object) -> subprocess.Popen[str]:
        process = real_launch(*args, **kwargs)
        launched.append(process)

        def interrupt_communicate(*_args: object, **_kwargs: object) -> object:
            deadline = time.monotonic() + 3.0
            while not child_pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            raise KeyboardInterrupt

        process.communicate = interrupt_communicate  # type: ignore[method-assign]
        return process

    monkeypatch.setattr(
        supervisor_runtime,
        "launch_process_child",
        launch_and_interrupt,
    )
    log_path = tmp_path / "interrupted.log"
    child_pid = 0
    try:
        with log_path.open("w", encoding="utf-8") as log_fh:
            with pytest.raises(KeyboardInterrupt):
                run_process_group_stream(
                    [sys.executable, "-c", script],
                    cwd=tmp_path,
                    stdout=log_fh,
                    timeout_seconds=60.0,
                    termination_grace_seconds=0.1,
                )
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert launched[0].poll() is not None
        assert not pid_alive(child_pid)
    finally:
        if child_pid and pid_alive(child_pid):
            os.kill(child_pid, 9)
        for process in launched:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=1.0)


def test_progress_observer_refreshes_durable_supervisor_heartbeat(
    tmp_path: Path,
) -> None:
    daemon = _daemon(tmp_path)
    task = _task(metadata={"requires provider": "true"})
    state = PortalTaskState(
        active_task_id=task.task_id,
        active_attempt=2,
        implementation_in_progress=True,
    )
    state.save(daemon.state_path)

    observer = daemon._implementation_progress_observer(
        state,
        task,
        attempt=2,
    )
    observer({"progress_events": 1})

    persisted = PortalTaskState.load(daemon.state_path)
    assert persisted.heartbeat_at
    assert persisted.last_progress_at == persisted.heartbeat_at


def test_progress_observer_preserves_concurrent_projection_fields(
    tmp_path: Path,
) -> None:
    daemon = _daemon(tmp_path)
    task = _task(metadata={"requires provider": "true"})
    state = PortalTaskState(
        active_task_id=task.task_id,
        active_attempt=2,
        implementation_in_progress=True,
        completed_task_ids=["SRT-001"],
        completed_count=1,
    )
    state.save(daemon.state_path)
    observer = daemon._implementation_progress_observer(
        state,
        task,
        attempt=2,
    )

    concurrent = PortalTaskState.load(daemon.state_path)
    concurrent.completed_task_ids = ["SRT-001", "SRT-002"]
    concurrent.completed_count = 2
    concurrent.blocked_task_ids = ["SRT-003"]
    concurrent.blocked_count = 1
    concurrent.save(daemon.state_path)

    observer({"progress_events": 1})

    persisted = PortalTaskState.load(daemon.state_path)
    assert persisted.completed_task_ids == ["SRT-001", "SRT-002"]
    assert persisted.completed_count == 2
    assert persisted.blocked_task_ids == ["SRT-003"]
    assert persisted.blocked_count == 1
    assert persisted.heartbeat_at
    assert persisted.last_progress_at == persisted.heartbeat_at
    assert state.heartbeat_at == persisted.heartbeat_at


def test_checkpoint_manifest_is_cid_bound_and_propagated_to_retry(
    tmp_path: Path,
) -> None:
    _seed_repository(tmp_path)
    task = _task(metadata={"requires provider": "true"})
    daemon = _daemon(tmp_path)
    checkpoint_dir = daemon._ensure_implementation_checkpoint_dir(task)
    (checkpoint_dir / "raw.jsonl").write_text(
        '{"coordinate":"a"}\n{"coordinate":"b"}\n',
        encoding="utf-8",
    )
    (checkpoint_dir / "meta.json").write_text(
        json.dumps({"expected": 670, "completed": 2}),
        encoding="utf-8",
    )

    manifest = daemon._implementation_checkpoint_manifest(task)
    assert manifest["file_count"] == 2
    assert manifest["manifest_cid"].startswith("b")
    assert {item["path"] for item in manifest["files"]} == {
        "meta.json",
        "raw.jsonl",
    }
    assert all(item["cid"].startswith("b") for item in manifest["files"])

    first_prompt = daemon._build_implementation_prompt(task, attempt=1)
    daemon._persist_implementation_context_receipt(task, attempt=1)
    diagnostic = daemon._record_failed_attempt_retry_context(
        task,
        returncode=124,
        timeout_result={
            "timeout_reason": "hard_timeout",
            "timeout_policy": (
                daemon._implementation_timeout_policy(task).to_dict()
            ),
        },
    )
    assert diagnostic is not None
    assert diagnostic.failure["checkpoint_manifest"]["manifest_cid"] == (
        manifest["manifest_cid"]
    )

    restarted = _daemon(tmp_path)
    retry_prompt = restarted._build_implementation_prompt(task, attempt=2)
    assert str(checkpoint_dir) in first_prompt
    assert str(checkpoint_dir) in retry_prompt
    assert manifest["manifest_cid"] in retry_prompt
    for prompt in (first_prompt, retry_prompt):
        assert "Authoritative validation environment" in prompt
        assert "inherited `PATH` is ignored" in prompt
        assert "ipfs-accelerate-validation-home-" in prompt
        assert "$HOME/.config" in prompt
    environment = restarted._implementation_process_environment(
        task,
        attempt=2,
        checkpoint_dir=checkpoint_dir,
    )
    assert environment[IMPLEMENTATION_CHECKPOINT_DIR_ENV] == str(
        checkpoint_dir
    )
