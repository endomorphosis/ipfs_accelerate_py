from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    IMPLEMENTATION_CHECKPOINT_DIR_ENV,
    IMPLEMENTATION_TASK_ID_ENV,
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    run_process_group_stream,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    PROOF_REUSE_STATE_ROOT_ENV,
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


def test_provider_process_uses_exact_sanitized_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        DATABASE_PROGRAM_JSON_ENV,
        DatabaseProgramConfig,
    )

    custom_secret_name = "APMC_CUSTOM_CONTROL_CREDENTIAL"
    monkeypatch.setenv(custom_secret_name, "private-control-token")
    monkeypatch.setenv("QUACK_TOKEN", "private-quack-token")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
        "private-canonical-quack-token",
    )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
        "/private/quack-mutations",
    )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_RUNTIME_REGISTRY_PATH",
        "/private/runtime-registry",
    )
    monkeypatch.setenv("APMC_PROVIDER_ALLOWED", "allowed")
    program = DatabaseProgramConfig(
        authority_mode="quack",
        task_source_kind="duckdb",
        endpoint_secret_handle=f"env://{custom_secret_name}",
        quack_endpoint="quack:127.0.0.1:45231",
        store_id="apmc-control",
        store_generation="1",
        schema_revision="1",
    )
    monkeypatch.setenv(
        DATABASE_PROGRAM_JSON_ENV,
        json.dumps(program.to_dict(), separators=(",", ":"), sort_keys=True),
    )
    daemon = _daemon(tmp_path)
    environment = daemon._implementation_process_environment(
        _task(),
        attempt=2,
        checkpoint_dir=tmp_path / "checkpoint",
    )
    result_path = tmp_path / "provider-environment.json"
    script = (
        "import json, os, pathlib; "
        "pathlib.Path('provider-environment.json').write_text(json.dumps({"
        f"'custom_secret': bool(os.environ.get('{custom_secret_name}')), "
        "'quack_token': bool(os.environ.get('QUACK_TOKEN')), "
        "'canonical_quack_token': bool(os.environ.get("
        "'IPFS_ACCELERATE_AGENT_QUACK_TOKEN')), "
        "'mutation_inbox': bool(os.environ.get("
        "'IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR')), "
        "'runtime_registry': bool(os.environ.get("
        "'IPFS_ACCELERATE_AGENT_RUNTIME_REGISTRY_PATH')), "
        "'database_program': bool(os.environ.get("
        "'IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON')), "
        "'allowed': os.environ.get('APMC_PROVIDER_ALLOWED'), "
        "'task_id': os.environ.get('IPFS_ACCELERATE_AGENT_TASK_ID')}))"
    )
    with (tmp_path / "provider.log").open("w", encoding="utf-8") as log_fh:
        completed = run_process_group_stream(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            stdout=log_fh,
            env=environment,
            inherit_environment=False,
            timeout_seconds=5.0,
        )

    assert completed.returncode == 0
    observed = json.loads(result_path.read_text(encoding="utf-8"))
    assert observed == {
        "allowed": "allowed",
        "canonical_quack_token": False,
        "custom_secret": False,
        "database_program": False,
        "mutation_inbox": False,
        "quack_token": False,
        "runtime_registry": False,
        "task_id": "SRT-014",
    }
    assert environment[IMPLEMENTATION_TASK_ID_ENV] == "SRT-014"


def test_process_group_stream_preserves_default_environment_inheritance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APMC_AMBIENT_INHERITANCE_PROBE", "visible")
    result_path = tmp_path / "ambient.txt"
    script = (
        "import os, pathlib; pathlib.Path('ambient.txt').write_text("
        "os.environ.get('APMC_AMBIENT_INHERITANCE_PROBE', 'missing'))"
    )
    with (tmp_path / "ambient.log").open("w", encoding="utf-8") as log_fh:
        completed = run_process_group_stream(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            stdout=log_fh,
            timeout_seconds=5.0,
        )

    assert completed.returncode == 0
    assert result_path.read_text(encoding="utf-8") == "visible"


def test_deterministic_writer_uses_sanitized_exact_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUACK_TOKEN", "private-quack-token")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
        "private-canonical-quack-token",
    )
    monkeypatch.setenv("APMC_WRITER_ALLOWED", "allowed")
    writer_path = tmp_path / "writer.py"
    result_path = tmp_path / "writer-environment.json"
    writer_path.write_text(
        "import json, os, pathlib\n"
        "pathlib.Path('writer-environment.json').write_text(json.dumps({\n"
        "    'quack_token': bool(os.environ.get('QUACK_TOKEN')),\n"
        "    'canonical_quack_token': bool(os.environ.get(\n"
        "        'IPFS_ACCELERATE_AGENT_QUACK_TOKEN')),\n"
        "    'allowed': os.environ.get('APMC_WRITER_ALLOWED'),\n"
        "    'task_id': os.environ.get('IPFS_ACCELERATE_AGENT_TASK_ID'),\n"
        "}))\n",
        encoding="utf-8",
    )
    daemon = _daemon(tmp_path)

    completed = daemon._run_lgswf_writer(
        writer_path,
        workspace_path=tmp_path,
        task=_task(),
        attempt=3,
        checkpoint_dir=tmp_path / "checkpoint",
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "allowed": "allowed",
        "canonical_quack_token": False,
        "quack_token": False,
        "task_id": "SRT-014",
    }


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
    # Retry evidence retains the content identity, not the private host path.
    assert str(checkpoint_dir) not in retry_prompt
    assert manifest["manifest_cid"] in retry_prompt
    environment = restarted._implementation_process_environment(
        task,
        attempt=2,
        checkpoint_dir=checkpoint_dir,
    )
    assert environment[IMPLEMENTATION_CHECKPOINT_DIR_ENV] == str(
        checkpoint_dir
    )
    assert environment[PROOF_REUSE_STATE_ROOT_ENV] == ""
