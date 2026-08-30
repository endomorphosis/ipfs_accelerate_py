"""Managed-local Quack lifecycle integration for configured supervisors."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    read_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    configured_board_scheduler as scheduler,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as runner,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_owner_watchdog import (
    AuthenticatedReadiness,
    OwnerHealth,
    QuackOwnerBinding,
    QuackOwnerObservation,
    QuackOwnerStartAbsentError,
    SpawnedQuackOwner,
    WatchdogDisposition,
    process_birth_id,
)


def _owner_policy(state_dir: Path) -> dict[str, object]:
    return {
        "mode": "managed_local",
        "owner_state_dir": str(state_dir),
        "startup_timeout_seconds": 5,
        "health_check_interval_seconds": 0.01,
        "max_restart_attempts": 3,
        "initial_backoff_seconds": 0,
        "max_backoff_seconds": 1,
        "termination_grace_seconds": 2,
    }


def _managed_program(
    state_dir: Path,
    *,
    port: int = 25123,
) -> runner.DatabaseProgramConfig:
    return runner.DatabaseProgramConfig.from_mapping(
        {
            "authority_mode": "quack",
            "task_source_kind": "duckdb",
            "endpoint_secret_handle": "env://TEST_QUACK_OWNER_TOKEN",
            "quack_endpoint": f"quack:127.0.0.1:{port}",
            "store_id": "state/control.duckdb",
            "store_generation": "logical-g1",
            "schema_revision": "1",
            "failover_policy": "fail_closed",
            "owner_management": _owner_policy(state_dir),
        }
    )


def _write_owner_status(
    *,
    repo: Path,
    program: runner.DatabaseProgramConfig,
    birth: ProcessBirthIdentity,
    generation: int,
    lifecycle: str,
    repository_id: str = "repository:sealed-authority",
) -> dict[str, object]:
    assert program.owner_management is not None
    state_dir = Path(program.owner_management.owner_state_dir)
    identity: dict[str, object] = {
        "server_id": f"server-{generation}",
        "store_id": program.store_id,
        "database_uuid": "database-uuid-1",
        "schema_revision": int(program.schema_revision),
        "schema_fingerprint": "sha256:" + "1" * 64,
        "generation": generation,
        "process_birth": birth.to_dict(),
        "process_birth_id": process_birth_id(birth),
        "listen_uri": program.quack_endpoint,
        "secret_handle": program.endpoint_secret_handle,
        "repository_id": repository_id,
        "status": lifecycle,
    }
    storage_schema_fingerprint = "baguqeera" + "2" * 56
    payload: dict[str, object] = {
        "schema": "ipfs_accelerate_py/agent-supervisor/quack-state-server@1",
        "interface": "QuackStateServer@1",
        "lifecycle": lifecycle,
        "database_path": str(repo / program.store_id),
        "state_dir": str(state_dir),
        "store_id": program.store_id,
        "secret_handle": program.endpoint_secret_handle,
        "storage_schema_fingerprint": storage_schema_fingerprint,
        "identity": identity,
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "quack-state-server.status.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    return identity


def _concrete_lifecycle(
    tmp_path: Path,
) -> tuple[runner.ManagedLocalQuackOwnerLifecycle, runner.DatabaseProgramConfig]:
    repo = tmp_path.resolve()
    database = repo / "state" / "control.duckdb"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"exact-store-file")
    state_dir = database.parent / "quack-owner"
    program = _managed_program(state_dir)
    dead_birth = ProcessBirthIdentity(
        pid=999_999_999,
        start_time_ticks=1,
        boot_id="test-boot",
    )
    _write_owner_status(
        repo=repo,
        program=program,
        birth=dead_birth,
        generation=7,
        lifecycle="stopped",
    )
    entry = repo / "quack-owner-entry.py"
    entry.write_text("raise SystemExit(0)\n", encoding="utf-8")
    lifecycle = runner.ManagedLocalQuackOwnerLifecycle(
        program=program,
        repo_root=repo,
        python_executable=sys.executable,
        owner_entry_path=entry,
    )
    return lifecycle, program


def test_managed_owner_policy_is_closed_and_explicit(tmp_path: Path) -> None:
    program = _managed_program(tmp_path / "owner")
    assert program.owner_management is not None
    assert program.owner_management.mode == "managed_local"
    assert runner.DatabaseProgramConfig.from_mapping(program.to_dict()) == program

    invalid = _owner_policy(tmp_path / "owner")
    invalid["surprise"] = True
    with pytest.raises(runner.DatabaseProgramConfigError, match="closed object"):
        runner.QuackOwnerManagementConfig.from_mapping(invalid)
    with pytest.raises(
        runner.DatabaseProgramConfigError,
        match="only for quack",
    ):
        runner.DatabaseProgramConfig(
            authority_mode="embedded",
            task_source_kind="duckdb",
            owner_management=runner.QuackOwnerManagementConfig.from_mapping(
                _owner_policy(tmp_path / "owner")
            ),
        )


def test_configured_board_confines_managed_owner_to_exact_store_directory(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "state").mkdir(parents=True)
    exact = _managed_program(repo / "state" / "quack-owner")
    scheduler._validate_managed_quack_owner_confinement(repo, exact)

    outside = _managed_program(tmp_path / "outside" / "quack-owner")
    with pytest.raises(scheduler.ConfiguredBoardError, match="outside"):
        scheduler._validate_managed_quack_owner_confinement(repo, outside)

    target = tmp_path / "linked-state"
    target.mkdir()
    (repo / "linked").symlink_to(target, target_is_directory=True)
    linked = runner.DatabaseProgramConfig.from_mapping(
        {
            **exact.to_dict(),
            "store_id": "linked/control.duckdb",
            "owner_management": _owner_policy(repo / "linked" / "quack-owner"),
        }
    )
    with pytest.raises(scheduler.ConfiguredBoardError, match="linked component"):
        scheduler._validate_managed_quack_owner_confinement(repo, linked)


def test_cli_runner_propagates_managed_program_only_when_explicit(
    tmp_path: Path,
) -> None:
    program = _managed_program(tmp_path / "owner")
    configured = runner.build_configured_multi_supervisor_cli_runner(
        repo_root=tmp_path,
        implementation_tracks=("lane|script|state|prefix",),
        database_program=program,
    )
    args = configured.args()
    assert "--database-program-json" in args
    restored = runner.parse_database_program_config(
        __import__("json").loads(
            args[args.index("--database-program-json") + 1]
        )
    )
    assert restored == program

    external = runner.build_configured_multi_supervisor_cli_runner(
        repo_root=tmp_path,
        implementation_tracks=("lane|script|state|prefix",),
        database_program=runner.DatabaseProgramConfig(
            authority_mode="quack",
            task_source_kind="duckdb",
            endpoint_secret_handle="env://TEST_QUACK_OWNER_TOKEN",
            quack_endpoint="quack:127.0.0.1:45123",
            store_id="state/control.duckdb",
            store_generation="logical-g1",
            schema_revision="1",
        ),
    ).args()
    assert "--database-program-json" not in external


def test_configured_board_rejects_unreserved_ephemeral_owner_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    state_dir = repo / "state" / "quack-owner"
    state_dir.mkdir(parents=True)
    program = _managed_program(state_dir, port=45123)
    monkeypatch.setattr(
        scheduler,
        "_linux_unreserved_ephemeral_tcp_port",
        lambda port: port == 45123,
    )

    with pytest.raises(
        scheduler.ConfiguredBoardError,
        match="ephemeral client-port range",
    ):
        scheduler._validate_managed_quack_owner_confinement(repo, program)


def test_concrete_owner_authenticates_by_handle_and_preserves_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, program = _concrete_lifecycle(tmp_path)
    birth = read_process_birth(os.getpid())
    assert birth is not None
    identity = _write_owner_status(
        repo=tmp_path.resolve(),
        program=program,
        birth=birth,
        generation=8,
        lifecycle="ready",
    )
    captured: dict[str, object] = {}

    transport_binding = {
        **identity,
        "schema_fingerprint": "baguqeera" + "2" * 56,
    }

    class _Connection:
        _quack_mutation_binding = transport_binding

        def close(self) -> None:
            captured["closed"] = True

    def open_connection(uri: str) -> _Connection:
        captured["uri"] = uri
        captured["handle"] = os.environ.get(
            runner.STATE_ENDPOINT_SECRET_HANDLE_ENV
        )
        captured["raw_handle_target"] = os.environ.get(
            "TEST_QUACK_OWNER_TOKEN"
        )
        captured["raw_state_token"] = os.environ.get(
            "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
        )
        return _Connection()

    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state

    monkeypatch.setattr(
        duckdb_state,
        "open_quack_transport_connection",
        open_connection,
    )
    monkeypatch.setenv("TEST_QUACK_OWNER_TOKEN", "stale-handle-token")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "stale-token")
    owner = SpawnedQuackOwner(
        process=None,
        process_birth=birth,
        isolated_process_group=True,
        adopted=True,
    )

    readiness = lifecycle._authenticated_readiness_once(owner)

    assert readiness.admitted is True
    assert readiness.binding.generation == 8
    assert readiness.binding.schema_fingerprint == identity["schema_fingerprint"]
    assert (
        transport_binding["schema_fingerprint"]
        != readiness.binding.schema_fingerprint
    )
    assert lifecycle._repository_id == "repository:sealed-authority"
    assert captured == {
        "uri": program.quack_endpoint,
        "handle": program.endpoint_secret_handle,
        "raw_handle_target": None,
        "raw_state_token": None,
        "closed": True,
    }


def test_proven_dead_owner_status_can_seed_a_new_sealed_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path.resolve()
    database = repo / "state" / "control.duckdb"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"exact-store-file")
    state_dir = database.parent / "quack-owner"
    old_program = _managed_program(state_dir, port=25123)
    _write_owner_status(
        repo=repo,
        program=old_program,
        birth=ProcessBirthIdentity(
            pid=999_999_999,
            start_time_ticks=1,
            boot_id="test-boot",
        ),
        generation=7,
        lifecycle="failed",
    )
    entry = repo / "quack-owner-entry.py"
    entry.write_text("raise SystemExit(0)\n", encoding="utf-8")
    new_program = _managed_program(state_dir, port=25124)

    lifecycle = runner.ManagedLocalQuackOwnerLifecycle(
        program=new_program,
        repo_root=repo,
        python_executable=sys.executable,
        owner_entry_path=entry,
    )

    assert lifecycle.program.quack_endpoint == "quack:127.0.0.1:25124"
    assert lifecycle._startup_endpoint_migration_pending is True
    with pytest.raises(
        runner._StableArtifactReadError,
        match="differs from the configured authority",
    ):
        lifecycle._status_identity()
    assert lifecycle._read_owner_observation(
        authenticate_alive=False
    ).liveness is OwnerLiveness.UNKNOWN
    dead = lifecycle._read_owner_observation(
        authenticate_alive=False,
        allow_proven_dead_endpoint_migration=True,
    )
    assert dead.provably_dead is True
    assert dead.binding is not None

    birth = ProcessBirthIdentity(
        pid=65432,
        start_time_ticks=2345,
        boot_id="test-boot",
    )
    owner = SpawnedQuackOwner(
        process=SimpleNamespace(pid=birth.pid, poll=lambda: None),
        process_birth=birth,
        isolated_process_group=True,
    )
    later = QuackOwnerBinding(
        store_id=dead.binding.store_id,
        schema_revision=dead.binding.schema_revision,
        database_uuid=dead.binding.database_uuid,
        schema_fingerprint=dead.binding.schema_fingerprint,
        generation=dead.binding.generation + 1,
        server_id="server-recovered-on-new-endpoint",
    )
    monkeypatch.setattr(lifecycle._watchdog, "_start_owner", lambda: owner)
    monkeypatch.setattr(
        lifecycle._watchdog,
        "_readiness_probe",
        lambda selected: AuthenticatedReadiness(
            authenticated=True,
            ready=True,
            process_birth_id=selected.birth_id,
            binding=later,
        ),
    )

    startup = lifecycle.ensure_before_tracks()

    assert startup["ready"] is True
    assert startup["recovered"] is True
    assert lifecycle._startup_endpoint_migration_pending is False


def test_live_owner_status_cannot_migrate_to_another_endpoint(
    tmp_path: Path,
) -> None:
    repo = tmp_path.resolve()
    database = repo / "state" / "control.duckdb"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"exact-store-file")
    state_dir = database.parent / "quack-owner"
    old_program = _managed_program(state_dir, port=25123)
    birth = read_process_birth(os.getpid())
    assert birth is not None
    _write_owner_status(
        repo=repo,
        program=old_program,
        birth=birth,
        generation=7,
        lifecycle="ready",
    )
    entry = repo / "quack-owner-entry.py"
    entry.write_text("raise SystemExit(0)\n", encoding="utf-8")

    with pytest.raises(
        runner._StableArtifactReadError,
        match="differs from the configured authority",
    ):
        runner.ManagedLocalQuackOwnerLifecycle(
            program=_managed_program(state_dir, port=25124),
            repo_root=repo,
            python_executable=sys.executable,
            owner_entry_path=entry,
        )


def test_managed_owner_watchdog_fences_and_does_not_swallow_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, _program = _concrete_lifecycle(tmp_path)
    dead = lifecycle._read_owner_observation(authenticate_alive=False)
    assert dead.provably_dead is True
    birth = read_process_birth(os.getpid())
    assert birth is not None
    owner = SpawnedQuackOwner(
        process=None,
        process_birth=birth,
        isolated_process_group=True,
        adopted=True,
    )
    calls = {"readiness": 0, "termination": 0}

    def interrupt(_owner: SpawnedQuackOwner) -> AuthenticatedReadiness:
        calls["readiness"] += 1
        raise runner.SupervisorRunInterrupted("operator shutdown")

    def terminate(_owner: SpawnedQuackOwner) -> SimpleNamespace:
        calls["termination"] += 1
        return SimpleNamespace(termination_confirmed=True)

    monkeypatch.setattr(lifecycle._watchdog, "_start_owner", lambda: owner)
    monkeypatch.setattr(lifecycle._watchdog, "_readiness_probe", interrupt)
    monkeypatch.setattr(lifecycle._watchdog, "_terminate_owner", terminate)

    with pytest.raises(runner.SupervisorRunInterrupted, match="operator shutdown"):
        lifecycle._watchdog.ensure(dead)
    assert calls == {"readiness": 1, "termination": 1}


def test_concrete_owner_spawn_reuses_repository_id_and_positive_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, program = _concrete_lifecycle(tmp_path)
    captured: dict[str, object] = {}
    birth = ProcessBirthIdentity(
        pid=54321,
        start_time_ticks=1234,
        boot_id="test-boot",
    )

    class _Process:
        pid = birth.pid

        def poll(self) -> None:
            return None

    def popen(command: list[str], **kwargs: object) -> _Process:
        captured["command"] = command
        captured["environment"] = dict(kwargs["env"])
        return _Process()

    from ipfs_accelerate_py.agent_supervisor.merge import worktree_lifecycle

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(
        worktree_lifecycle,
        "read_process_birth",
        lambda _pid: birth,
    )
    monkeypatch.setattr(
        lifecycle,
        "_wait_for_endpoint_bindability",
        lambda host, port: captured.setdefault(
            "bind_preflight", (host, port)
        ),
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "stale-token")
    monkeypatch.setenv("TEST_QUACK_OWNER_TOKEN", "stale-handle-token")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")

    owner = lifecycle._start_owner()

    assert owner.process_birth == birth
    assert captured["bind_preflight"] == ("127.0.0.1", 25123)
    command = captured["command"]
    assert isinstance(command, list)
    repository_index = command.index("--repository-id")
    assert command[repository_index + 1] == "repository:sealed-authority"
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert "OPENAI_API_KEY" not in environment
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in environment
    assert "TEST_QUACK_OWNER_TOKEN" not in environment
    assert environment[runner.STATE_ENDPOINT_SECRET_HANDLE_ENV] == (
        program.endpoint_secret_handle
    )


def test_managed_owner_waits_through_transient_endpoint_bind_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, _program = _concrete_lifecycle(tmp_path)
    states = iter(
        (
            runner._MANAGED_QUACK_ENDPOINT_TRANSIENT,
            runner._MANAGED_QUACK_ENDPOINT_TRANSIENT,
            runner._MANAGED_QUACK_ENDPOINT_BINDABLE,
        )
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        lifecycle,
        "_probe_endpoint_bindability",
        lambda _host, _port: next(states),
    )
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)

    lifecycle._wait_for_endpoint_bindability("127.0.0.1", 45123)

    assert sleeps == [0.1, 0.1]


def test_managed_owner_refuses_unauthenticated_live_endpoint_listener(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, _program = _concrete_lifecycle(tmp_path)
    monkeypatch.setattr(
        lifecycle,
        "_probe_endpoint_bindability",
        lambda _host, _port: runner._MANAGED_QUACK_ENDPOINT_LIVE,
    )

    with pytest.raises(
        runner.DatabaseProgramConfigError,
        match="unauthenticated live listener",
    ):
        lifecycle._wait_for_endpoint_bindability("127.0.0.1", 45123)


def test_uncaptured_owner_birth_is_exactly_fenced_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, _program = _concrete_lifecycle(tmp_path)
    signals: list[int] = []

    class _Process:
        pid = 76543
        exited = False

        def poll(self) -> int | None:
            return 0 if self.exited else None

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            self.exited = True
            return 0

    process = _Process()
    group_alive = {"value": True}
    monkeypatch.setattr(runner.os, "getpgid", lambda _pid: process.pid)

    def signal_group(_pid: int, sent: int) -> None:
        if sent == 0:
            if not group_alive["value"]:
                raise ProcessLookupError
            return
        signals.append(sent)
        group_alive["value"] = False

    monkeypatch.setattr(
        runner.os,
        "killpg",
        signal_group,
    )

    assert lifecycle._terminate_uncaptured_owner(process) is True
    assert signals == [runner.signal.SIGTERM]


def test_concrete_recovery_retries_transient_confirmed_absence_internally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, _program = _concrete_lifecycle(tmp_path)
    dead = lifecycle._read_owner_observation(authenticate_alive=False)
    assert dead.binding is not None
    attempts = {"count": 0}
    birth = ProcessBirthIdentity(
        pid=65432,
        start_time_ticks=2345,
        boot_id="test-boot",
    )
    owner = SpawnedQuackOwner(
        process=SimpleNamespace(pid=birth.pid, poll=lambda: None),
        process_birth=birth,
        isolated_process_group=True,
    )
    later = QuackOwnerBinding(
        store_id=dead.binding.store_id,
        schema_revision=dead.binding.schema_revision,
        database_uuid=dead.binding.database_uuid,
        schema_fingerprint=dead.binding.schema_fingerprint,
        generation=dead.binding.generation + 1,
        server_id="server-recovered",
    )

    def start_owner() -> SpawnedQuackOwner:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise QuackOwnerStartAbsentError("confirmed absent transient")
        return owner

    monkeypatch.setattr(lifecycle, "_read_owner_observation", lambda **_kwargs: dead)
    monkeypatch.setattr(lifecycle._watchdog, "_observe_owner", lambda: dead)
    monkeypatch.setattr(lifecycle._watchdog, "_start_owner", start_owner)
    monkeypatch.setattr(
        lifecycle._watchdog,
        "_readiness_probe",
        lambda selected: AuthenticatedReadiness(
            authenticated=True,
            ready=True,
            process_birth_id=selected.birth_id,
            binding=later,
        ),
    )

    result = lifecycle.recover_after_fence()

    assert result["recovered"] is True
    assert attempts["count"] == 2


def test_lock_contender_waits_through_missing_status_for_exact_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, _program = _concrete_lifecycle(tmp_path)
    dead = lifecycle._read_owner_observation(authenticate_alive=False)
    assert dead.binding is not None
    later = QuackOwnerBinding(
        store_id=dead.binding.store_id,
        schema_revision=dead.binding.schema_revision,
        database_uuid=dead.binding.database_uuid,
        schema_fingerprint=dead.binding.schema_fingerprint,
        generation=dead.binding.generation + 1,
        server_id="concurrent-winner",
    )
    winner_birth = ProcessBirthIdentity(
        pid=87654,
        start_time_ticks=3456,
        boot_id="test-boot",
    )
    missing_status = QuackOwnerObservation(
        process_birth=None,
        liveness=OwnerLiveness.UNKNOWN,
        health=OwnerHealth.UNKNOWN,
    )
    winner = QuackOwnerObservation(
        process_birth=winner_birth,
        liveness=OwnerLiveness.ALIVE,
        health=OwnerHealth.HEALTHY,
        authenticated_ready=True,
        binding=later,
    )
    observations = iter((dead, missing_status, winner))
    monkeypatch.setattr(
        lifecycle,
        "_read_owner_observation",
        lambda **_kwargs: next(observations),
    )
    monkeypatch.setattr(runner.os, "getpgid", lambda pid: pid)
    contender = SimpleNamespace(
        recovered=False,
        operational_ready=False,
        disposition=WatchdogDisposition.LOCK_CONTENDED,
        to_dict=lambda: {
            "disposition": WatchdogDisposition.LOCK_CONTENDED.value
        },
    )
    ensure_calls = {"count": 0}

    def ensure(_observation: QuackOwnerObservation) -> object:
        ensure_calls["count"] += 1
        return contender

    monkeypatch.setattr(lifecycle._watchdog, "ensure", ensure)

    result = lifecycle.recover_after_fence()

    assert result["recovered"] is True
    assert result["recovery_source"] == "concurrent_exact_winner"
    assert ensure_calls["count"] == 1


def test_start_track_scrubs_rotated_raw_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = _managed_program(tmp_path / "state" / "quack-owner")
    script = tmp_path / "lane.py"
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    track = runner.SupervisorTrack(
        name="lane",
        script_path=script,
        log_path=tmp_path / "lane.log",
        supervisor_pid_path=tmp_path / "lane.pid",
        daemon_pid_path=tmp_path / "daemon.pid",
        database_program=program,
    )
    captured: dict[str, object] = {}

    def popen(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured["environment"] = dict(kwargs["env"])
        return SimpleNamespace(pid=os.getpid())

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "old-generation")
    monkeypatch.setenv("TEST_QUACK_OWNER_TOKEN", "old-handle-generation")

    runner.start_track(
        track,
        repo_root=tmp_path,
        common_args=(),
        python_executable=sys.executable,
        output=lambda _message: None,
    )

    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in environment
    assert "TEST_QUACK_OWNER_TOKEN" not in environment
    assert environment[runner.STATE_ENDPOINT_SECRET_HANDLE_ENV] == (
        program.endpoint_secret_handle
    )
    assert json.loads(environment[runner.DATABASE_PROGRAM_JSON_ENV]) == (
        program.to_dict()
    )


class _FakeProcess:
    pid = 43210

    def poll(self) -> None:
        return None


class _Lifecycle:
    health_check_interval_seconds = 0.005

    def __init__(self, events: list[str], health: list[str]) -> None:
        self.events = events
        self.health_values = health
        self.recovery_count = 0

    def ensure_before_tracks(self) -> dict[str, object]:
        self.events.append("owner.ensure")
        return {"ready": True, "generation": 1}

    def health(self) -> dict[str, object]:
        self.events.append("owner.health")
        value = self.health_values.pop(0) if self.health_values else "healthy"
        return {"health": value}

    def recover_after_fence(self) -> dict[str, object]:
        self.events.append("owner.recover")
        self.recovery_count += 1
        return {"recovered": True, "generation": 2}

    def shutdown(self) -> dict[str, object]:
        self.events.append("owner.shutdown")
        return {"stopped": True, "desired_state": "stopped"}


class _ShutdownFailureLifecycle(_Lifecycle):
    def shutdown(self) -> dict[str, object]:
        self.events.append("owner.shutdown")
        return {"stopped": False, "desired_state": "stopped"}


def _track(tmp_path: Path) -> runner.SupervisorTrack:
    return runner.SupervisorTrack(
        name="lane",
        script_path=tmp_path / "lane.py",
        log_path=tmp_path / "lane.log",
        supervisor_pid_path=tmp_path / "lane.pid",
        daemon_pid_path=tmp_path / "daemon.pid",
    )


def test_status_projection_is_bound_to_current_lane_birth(tmp_path: Path) -> None:
    status_path = tmp_path / "lane-status.json"
    status_path.write_text(
        json.dumps(
            {
                "supervisor_pid": 41,
                "updated_at": "2000-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    track = runner.SupervisorTrack(
        name="lane",
        script_path=tmp_path / "lane.py",
        log_path=tmp_path / "lane.log",
        supervisor_pid_path=tmp_path / "lane.pid",
        daemon_pid_path=tmp_path / "daemon.pid",
        supervisor_status_path=status_path,
    )

    starting = runner.supervisor_status_health_fields(
        track.resolve(tmp_path),
        repo_root=tmp_path,
        stale_seconds=1,
        expected_supervisor_pid=42,
        fresh_after_epoch_seconds=1_800_000_000,
        startup_grace_remaining_seconds=5,
    )
    assert starting["supervisor_status"] == "starting"
    assert starting["expected_supervisor_pid"] == 42
    assert starting["observed_supervisor_pid"] == 41
    assert "restart_supervisor" not in starting

    stalled = runner.supervisor_status_health_fields(
        track.resolve(tmp_path),
        repo_root=tmp_path,
        stale_seconds=1,
        expected_supervisor_pid=42,
        fresh_after_epoch_seconds=1_800_000_000,
        startup_grace_remaining_seconds=0,
    )
    assert stalled["supervisor_status"] == "stale"
    assert stalled["restart_supervisor"] is True


def test_live_lane_is_not_restarted_from_prior_birth_status_during_grace(
    tmp_path: Path,
) -> None:
    script = tmp_path / "lane.py"
    script.write_text(
        "\n".join(
            [
                "import signal",
                "import sys",
                "import time",
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))",
                "while True:",
                "    time.sleep(0.05)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    status_path = tmp_path / "lane-status.json"
    status_path.write_text(
        json.dumps(
            {
                "supervisor_pid": 999_999_999,
                "status": "stopped",
                "updated_at": "2000-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    track = runner.SupervisorTrack(
        name="lane",
        script_path=script,
        log_path=tmp_path / "lane.log",
        supervisor_pid_path=tmp_path / "lane.pid",
        daemon_pid_path=tmp_path / "daemon.pid",
        supervisor_status_path=status_path,
    )
    output: list[str] = []

    result = runner.run_supervisor_tracks(
        (track,),
        repo_root=tmp_path,
        common_args=(),
        duration_seconds=0.2,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=0.01,
        supervisor_startup_grace_seconds=1,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        output=output.append,
    )

    assert result["completed"] is True
    assert sum("started lane supervisor" in line for line in output) == 1
    assert any("supervisor_status=starting" in line for line in output)
    assert not any("restarting stale lane supervisor" in line for line in output)


def _patch_track_runtime(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> None:
    def start_track(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        events.append("track.start")
        return _FakeProcess()

    def stop_tracks(*_args: Any, **_kwargs: Any) -> dict[str, object]:
        events.append("track.stop")
        return {
            "all_trees_fenced": True,
            "stopped_count": 1,
            "removed_runtime_markers": [],
        }

    monkeypatch.setattr(runner, "start_track", start_track)
    monkeypatch.setattr(runner, "stop_tracks", stop_tracks)
    monkeypatch.setattr(runner, "pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        runner,
        "daemon_pid_health_fields",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        runner,
        "supervisor_status_health_fields",
        lambda *_args, **_kwargs: {"restart_supervisor": False},
    )


def test_proven_owner_death_fences_then_recovers_then_restarts_tracks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    lifecycle = _Lifecycle(events, ["dead", "healthy"])
    _patch_track_runtime(monkeypatch, events)

    result = runner.run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=(),
        duration_seconds=0.12,
        heartbeat_interval_seconds=0.5,
        managed_quack_owner=lifecycle,
        output=lambda _message: None,
    )

    assert result["completed"] is True
    assert lifecycle.recovery_count == 1
    assert events.index("owner.ensure") < events.index("track.start")
    first_stop = events.index("track.stop")
    assert first_stop < events.index("owner.recover")
    starts = [index for index, event in enumerate(events) if event == "track.start"]
    assert len(starts) == 2
    assert events.index("owner.recover") < starts[1]
    assert events[-1] == "owner.shutdown"


@pytest.mark.parametrize("health", ["unknown", "unhealthy"])
def test_unknown_or_live_unhealthy_owner_fences_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    health: str,
) -> None:
    events: list[str] = []
    lifecycle = _Lifecycle(events, [health])
    _patch_track_runtime(monkeypatch, events)

    result = runner.run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=(),
        duration_seconds=0.08,
        heartbeat_interval_seconds=0.005,
        managed_quack_owner=lifecycle,
        output=lambda _message: None,
    )

    assert result["completed"] is False
    assert "refusing a competing owner" in str(result["blocked"])
    assert "owner.recover" not in events
    assert events[-1] == "owner.shutdown"


def test_intentional_shutdown_cannot_bounce_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    lifecycle = _Lifecycle(events, [])
    _patch_track_runtime(monkeypatch, events)

    result = runner.run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=(),
        duration_seconds=0,
        managed_quack_owner=lifecycle,
        output=lambda _message: None,
    )

    assert result["completed"] is True
    assert "owner.health" not in events
    assert events[-1] == "owner.shutdown"
    shutdown = result["managed_quack_owner"]["shutdown"]
    assert shutdown["desired_state"] == "stopped"


def test_unverified_intentional_shutdown_blocks_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    lifecycle = _ShutdownFailureLifecycle(events, [])
    _patch_track_runtime(monkeypatch, events)

    result = runner.run_supervisor_tracks(
        [_track(tmp_path)],
        repo_root=tmp_path,
        common_args=(),
        duration_seconds=0,
        managed_quack_owner=lifecycle,
        output=lambda _message: None,
    )

    assert result["completed"] is False
    assert "shutdown could not be verified" in str(result["blocked"])


@pytest.mark.parametrize(
    ("completed", "all_trees_fenced", "expected"),
    ((True, True, 0), (False, True, 2), (True, False, 2)),
)
def test_main_wires_managed_lifecycle_and_propagates_failure_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed: bool,
    all_trees_fenced: bool,
    expected: int,
) -> None:
    program = _managed_program(tmp_path / "state" / "quack-owner")
    lifecycle = object()
    captured: dict[str, object] = {}

    def build_lifecycle(**kwargs: object) -> object:
        captured["build"] = kwargs
        return lifecycle

    def run_tracks(tracks: list[runner.SupervisorTrack], **kwargs: object):
        captured["tracks"] = tracks
        captured["run"] = kwargs
        return {
            "completed": completed,
            "all_trees_fenced": all_trees_fenced,
        }

    monkeypatch.setattr(
        runner,
        "build_managed_quack_owner_lifecycle",
        build_lifecycle,
    )
    monkeypatch.setattr(runner, "run_supervisor_tracks", run_tracks)
    track_spec = "lane|lane.py|lane.log|lane.pid|daemon.pid"

    result = runner.main(
        [
            "--repo-root",
            str(tmp_path),
            "--master-dir",
            str(tmp_path / "master"),
            "--duration-seconds",
            "0",
            "--track",
            track_spec,
            "--database-program-json",
            json.dumps(program.to_dict(), sort_keys=True),
        ]
    )

    assert result == expected
    assert captured["build"]["program"] == program
    assert captured["run"]["managed_quack_owner"] is lifecycle
    tracks = captured["tracks"]
    assert isinstance(tracks, list)
    assert tracks[0].database_program == program


def test_detached_master_keeps_program_handle_but_scrubs_raw_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = _managed_program(tmp_path / "state" / "quack-owner")
    encoded = json.dumps(program.to_dict(), sort_keys=True)
    argv = [
        "--repo-root",
        str(tmp_path),
        "--master-dir",
        str(tmp_path / "master"),
        "--track",
        "lane|lane.py|lane.log|lane.pid|daemon.pid",
        "--database-program-json",
        encoded,
        "--detach",
    ]
    args = runner.build_arg_parser().parse_args(argv)
    _master_log, master_pid_path = runner._master_paths(args)
    master_pid_path.parent.mkdir(parents=True, mode=0o700)
    os.chmod(master_pid_path.parent, 0o700)
    master_pid_path.write_text("424242\n", encoding="ascii")
    os.chmod(master_pid_path, 0o664)
    captured: dict[str, object] = {}

    class _DetachedProcess:
        pid = 55555

        def poll(self) -> None:
            return None

    def popen(command: list[str], **kwargs: object) -> _DetachedProcess:
        captured["spawn_count"] = int(captured.get("spawn_count", 0)) + 1
        captured["command"] = command
        captured["environment"] = dict(kwargs["env"])
        return _DetachedProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner, "pid_alive", lambda pid: pid == 55555)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "old-generation")
    monkeypatch.setenv("TEST_QUACK_OWNER_TOKEN", "old-handle-generation")
    monkeypatch.setenv("PYTHONPATH", "/tmp/untrusted-ambient-import-root")

    result = runner.launch_detached(args, argv)

    assert result["master_pid"] == 55555
    command = captured["command"]
    assert isinstance(command, list)
    assert "--database-program-json" in command
    assert encoded in command
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in environment
    assert "TEST_QUACK_OWNER_TOKEN" not in environment
    assert environment[runner.STATE_ENDPOINT_SECRET_HANDLE_ENV] == (
        program.endpoint_secret_handle
    )
    assert environment["PYTHONPATH"] == str(
        Path(runner.__file__).resolve().parents[3]
    )
    assert master_pid_path.read_text(encoding="ascii") == "55555\n"
    assert stat.S_IMODE(master_pid_path.stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="names a live process"):
        runner.launch_detached(args, argv)
    assert captured["spawn_count"] == 1
