"""Focused ASEH bootstrap tests for the existing Quack supervisor handoff."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    current_process_birth,
    read_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    configured_board_scheduler as configured_scheduler,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as multi_runner,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
    provider_subprocess_environment,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    QuackStateServerControlError,
    QuackStateServerReadyError,
    TypedStateOwnerGrantBroker,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    typed_state_owner as typed_state_owner_module,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
    TaskSourceBoundsError,
    TaskSourceUnknownOutcomeError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS,
    QuackOwnerCommandRemoteError,
    reset_quack_transport_cache,
    submit_quack_owner_command,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    task_authority_spec_cid,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    QuackCapabilityStatus,
    probe_quack_capabilities,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    DATABASE_TASK_COMMANDS,
    TYPED_STATE_OWNER_GRANT_BROKER_SCHEMA,
    TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
    TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
    TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME,
    TYPED_STATE_OWNER_SOCKET_ENV,
    TypedStateOwnerAuthorizationError,
    TypedStateOwnerConnection,
    TypedStateOwnerError,
    TypedStateOwnerRemoteError,
    kernel_process_birth_id,
    request_database_task_command_credential,
    request_quack_attach_credential,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    supervisor as todo_supervisor,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    supervisor_loop as supervisor_loop_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import ManagedDaemonSpec
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import (
    SupervisorLoop,
    SupervisorLoopConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    SUPERVISED_CHILD_IDENTITY_PATH_ENV,
    SUPERVISED_CHILD_OWNER_SCOPE_ENV,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    build_validation_environment,
)

from scripts import run_agent_supervisor_efficiency_state_hardening as aseh_operator

_CWD_OWNER_DIR = Path("/proc/self/cwd/quack-owner")


def test_foreground_master_pid_recovers_only_a_proven_dead_owner(
    tmp_path: Path,
) -> None:
    exited = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    dead_pid = int(exited.pid)
    assert exited.wait(timeout=5) == 0

    pid_path = tmp_path / "state" / "configured-board-master.pid"
    pid_path.parent.mkdir()
    pid_path.write_text(f"{dead_pid}\n", encoding="ascii")
    pid_path.chmod(0o600)

    multi_runner._adopt_or_create_current_master_pid_projection(  # noqa: SLF001
        pid_path
    )

    assert pid_path.read_text(encoding="ascii") == f"{os.getpid()}\n"
    current_projection = pid_path.stat()
    assert current_projection.st_nlink == 1
    assert current_projection.st_mode & 0o777 == 0o600
    quarantines = tuple(pid_path.parent.glob(f".{pid_path.name}.stale-*.quarantine"))
    decisions = tuple(pid_path.parent.glob(f".{pid_path.name}.stale-*.decision.json"))
    receipts = tuple(pid_path.parent.glob(f".{pid_path.name}.stale-*.receipt.json"))
    assert len(quarantines) == len(decisions) == len(receipts) == 1
    assert quarantines[0].read_text(encoding="ascii") == f"{dead_pid}\n"
    decision = json.loads(decisions[0].read_text(encoding="utf-8"))
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert decision["schema"] == multi_runner.STALE_DETACHED_MASTER_PID_DECISION_SCHEMA
    assert decision["decision"] == "quarantine_authorized"
    assert decision["legacy_pid"] == dead_pid
    assert decision["liveness_evidence"]["errno"] == "ESRCH"
    assert receipt["schema"] == multi_runner.STALE_DETACHED_MASTER_PID_RECEIPT_SCHEMA
    assert receipt["outcome"] == "quarantined"
    assert receipt["legacy_pid"] == dead_pid


def test_foreground_master_pid_refuses_a_live_owner(tmp_path: Path) -> None:
    live = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_path = tmp_path / "state" / "configured-board-master.pid"
    pid_path.parent.mkdir()
    pid_path.write_text(f"{live.pid}\n", encoding="ascii")
    pid_path.chmod(0o600)
    try:
        with pytest.raises(
            ValueError,
            match="master PID projection names a live process",
        ):
            multi_runner._adopt_or_create_current_master_pid_projection(  # noqa: SLF001
                pid_path
            )
        assert pid_path.read_text(encoding="ascii") == f"{live.pid}\n"
        assert not tuple(pid_path.parent.glob(f".{pid_path.name}.stale-*"))
    finally:
        live.terminate()
        live.wait(timeout=5)


def test_aseh_task_authority_spec_excludes_typed_lifecycle_fields_only() -> None:
    task = {
        "task_cid": "task:aseh-authority-spec",
        "task_alias": "ASEH-001",
        "goal_cid": "goal:aseh-authority-spec",
        "objective_id": "objective:aseh-authority-spec",
        "ordinal": 1,
        "priority": "P0",
        "identity": {
            "task_cid": "task:aseh-authority-spec",
            "task_alias": "ASEH-001",
            "repository_tree_id": "tree:aseh-authority-spec",
        },
        "body": {"acceptance_conditions": "sealed acceptance"},
        "extension_schema": "",
        "extension": {},
        "dependencies": [],
        "outputs": [],
        "acceptance": [],
        "validations": [],
    }
    sealed = task_authority_spec_cid(task)
    lifecycle = {
        **task,
        "status": "in_progress",
        "revision": 4,
        "body": {
            **task["body"],
            "completion_receipt": {
                "operation": "database_claim",
                "claim_id": "claim:aseh-authority-spec",
            },
            "unknown_callback_reopen_count": 1,
        },
    }
    assert task_authority_spec_cid(lifecycle) == sealed

    forged = {**lifecycle, "body": dict(lifecycle["body"])}
    forged["body"]["acceptance_conditions"] = "reduced acceptance"
    assert task_authority_spec_cid(forged) != sealed


def _cwd_owner_socket(name: str) -> Path:
    """Bind Unix sockets through the same AF_UNIX-safe cwd alias as production."""

    return _CWD_OWNER_DIR / name


def test_aseh_parallel_quack_lanes_require_strict_deterministic_sharding() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (
            repository_root
            / "config/agent_supervisor_efficiency_state_hardening_scheduler.json"
        ).read_text(encoding="utf-8")
    )

    assert config["database_program"]["authority_mode"] == "quack"
    assert config["max_lanes"] == 4
    assert config["strict_task_sharding"] is True
    assert config["idle_lane_work_stealing"] == ""


def _materialize_one_task(path: Path) -> None:
    with DatabaseTaskSource(path) as source:
        source.materialize(
            {
                "repository_tree_id": "tree:aseh-bootstrap-test",
                "objectives": [
                    {
                        "goal_id": "ASEH-G000",
                        "goal_cid": "goal:aseh-bootstrap-test",
                        "objective_id": "objective:aseh-bootstrap-test",
                        "title": "Prove the canonical handoff",
                    }
                ],
                "taskboard": [
                    {
                        "task_id": "ASEH-000",
                        "task_cid": "task:aseh-bootstrap-test",
                        "goal_cid": "goal:aseh-bootstrap-test",
                        "status": "ready",
                    }
                ],
            }
        )


def test_aseh_offline_continuity_replay_never_mutates_authoritative_db(
    tmp_path: Path,
) -> None:
    database = tmp_path / "control.duckdb"
    _materialize_one_task(database)
    before = database.read_bytes()
    before_stat = database.stat()

    assert (
        aseh_operator._projection_matches_events_on_disposable_copy(database)
        is True
    )
    with aseh_operator._read_only_database_task_source(
        database,
        owner_id="aseh-test-read-only",
        repository_tree_id="tree:aseh-bootstrap-test",
        plan_root_cid="",
    ) as source:
        assert source.intent.uses_bound_connection is True
        assert source.snapshot().task_count == 1

    after_stat = database.stat()
    assert database.read_bytes() == before
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns


def _sealed_memfd(value: str) -> int:
    flags = int(getattr(os, "MFD_CLOEXEC", 0x0001)) | int(
        getattr(os, "MFD_ALLOW_SEALING", 0x0002)
    )
    descriptor = os.memfd_create("aseh-test-secret", flags=flags)
    os.write(descriptor, value.encode("ascii"))
    seals = (
        int(getattr(fcntl, "F_SEAL_SEAL", 0x0001))
        | int(getattr(fcntl, "F_SEAL_SHRINK", 0x0002))
        | int(getattr(fcntl, "F_SEAL_GROW", 0x0004))
        | int(getattr(fcntl, "F_SEAL_WRITE", 0x0008))
    )
    fcntl.fcntl(
        descriptor,
        int(getattr(fcntl, "F_ADD_SEALS", 1033)),
        seals,
    )
    return descriptor


def test_grant_broker_recovers_only_same_uid_stale_socket(
    tmp_path: Path,
) -> None:
    assert DATABASE_TASK_COMMANDS == frozenset(
        {
            "compare_and_set_status",
            "rearm_blocked_task",
            "record_queue_backoff",
            "record_queue_retry",
            "record_evidence",
            "record_validation_result",
        }
    )
    broker_path = tmp_path / "owner" / "grants.sock"
    broker_path.parent.mkdir()

    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(broker_path))
    stale.close()

    credential_calls: list[str] = []

    def credential(
        _kind: str,
        _client: str,
        _birth: str,
        _pid: int,
    ) -> str:
        credential_calls.append(_kind)
        return "fixed_test_credential"

    broker = TypedStateOwnerGrantBroker(
        socket_path=broker_path,
        bootstrap_secret="0" * 64,
        store_id="store:test",
        resolve_credential=credential,
    )
    broker.start()
    try:
        assert broker.alive() is True
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        channel.settimeout(1)
        try:
            channel.connect(str(broker_path))
            channel.sendall(
                (
                    json.dumps(
                        {
                            "schema": TYPED_STATE_OWNER_GRANT_BROKER_SCHEMA,
                            "credential_kind": "caller_selected_mutation_scope",
                            "bootstrap_secret": "0" * 64,
                            "client_id": "client:unknown-kind",
                            "process_birth_id": kernel_process_birth_id(),
                            "store_id": "store:test",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            with channel.makefile("rb") as stream:
                response = json.loads(stream.readline())
        finally:
            channel.close()
        assert response["ok"] is False
        assert response["error_code"] == "grant_denied"
        assert credential_calls == []
        assert broker.alive() is True
    finally:
        broker.stop()
    assert not broker_path.exists()

    live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    live.bind(str(broker_path))
    live.listen(1)
    blocked = TypedStateOwnerGrantBroker(
        socket_path=broker_path,
        bootstrap_secret="1" * 64,
        store_id="store:test",
        resolve_credential=credential,
    )
    try:
        with pytest.raises(QuackStateServerControlError, match="live listener"):
            blocked.start()
        assert broker_path.is_socket()
    finally:
        live.close()
        broker_path.unlink()

    broker_path.write_text("not a socket", encoding="utf-8")
    unsafe = TypedStateOwnerGrantBroker(
        socket_path=broker_path,
        bootstrap_secret="2" * 64,
        store_id="store:test",
        resolve_credential=credential,
    )
    with pytest.raises(QuackStateServerControlError, match="same-UID socket"):
        unsafe.start()
    assert broker_path.read_text(encoding="utf-8") == "not a socket"


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_grant_broker_reclaims_only_a_proved_stale_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip(f"reviewed preinstalled Quack unavailable: {capability.status.value}")

    database = tmp_path / "control.duckdb"
    owner_dir = tmp_path / "quack-owner"
    monkeypatch.chdir(tmp_path)
    broker_socket = _cwd_owner_socket(TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME)
    _materialize_one_task(database)
    owner_dir.mkdir(parents=True, exist_ok=True)

    stale_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale_listener.bind(str(broker_socket))
    stale_listener.close()
    assert broker_socket.is_socket()

    server = build_server(
        database_path=database,
        state_dir=owner_dir,
        repository_root=tmp_path,
        port=0,
        store_id=str(database),
        secret_handle="handle:aseh-stale-broker-test",
        typed_command_socket_path=_cwd_owner_socket("typed-owner.sock"),
    )
    server.start()
    try:
        handoff = dict(server.start_supervisor_grant_broker())
        assert Path(
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV]
        ).is_socket()
        assert server.ready()["live"] is True
    finally:
        server.stop()
        reset_quack_transport_cache()
    assert not broker_socket.exists()

    live_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    live_listener.bind(str(broker_socket))
    live_listener.listen(1)
    second = build_server(
        database_path=database,
        state_dir=owner_dir,
        repository_root=tmp_path,
        port=0,
        store_id=str(database),
        secret_handle="handle:aseh-live-broker-test",
        typed_command_socket_path=_cwd_owner_socket("typed-owner.sock"),
    )
    second.start()
    try:
        with pytest.raises(
            QuackStateServerControlError,
            match="already serves a live listener",
        ):
            second.start_supervisor_grant_broker()
        assert broker_socket.is_socket()
    finally:
        second.stop()
        live_listener.close()
        broker_socket.unlink(missing_ok=True)
        reset_quack_transport_cache()


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_real_configured_supervisor_handoff_reads_and_mutates_via_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip(f"reviewed preinstalled Quack unavailable: {capability.status.value}")

    database = tmp_path / "control.duckdb"
    owner_dir = tmp_path / "quack-owner"
    monkeypatch.chdir(tmp_path)
    owner_socket = Path("/proc/self/cwd/quack-owner/custom-typed-owner.sock")
    _materialize_one_task(database)
    server = build_server(
        database_path=database,
        state_dir=owner_dir,
        repository_root=tmp_path,
        host="127.0.0.1",
        port=0,
        repository_id="repository:aseh-bootstrap-test",
        store_id=str(database),
        secret_handle="handle:aseh-bootstrap-test",
        typed_command_socket_path=owner_socket,
    )
    identity = server.start()
    try:
        handoff = dict(server.start_supervisor_grant_broker())
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV],
        )
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV],
        )
        monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(owner_socket))
        monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(database))
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION",
            str(identity.generation),
        )
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
            str(owner_dir / "mutations"),
        )
        # A stale legacy value must never shadow the live broker exchange.
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "stale_legacy_token"
        )

        assert not (owner_dir / "handle_aseh-bootstrap-test.quack-token").exists()
        assert server.status()["configured_supervisor_credential_broker"] == {
            "available": True,
            "server_owned": True,
            "socket_path": "/proc/self/cwd/quack-owner/typed-state-owner-grants.sock",
            "credential_published": False,
            "task_mutation_path": "typed_state_owner_database_task_command",
            "last_error_type": "",
        }
        assert handoff[TYPED_STATE_OWNER_SOCKET_ENV] == str(owner_socket)
        assert server.status_path().is_file()
        assert not any("ControlPlaneBoundsError" in item for item in server.logs())

        def forbidden_legacy_inbox(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("filesystem mutation servicing was invoked")

        for service_name in (
            "process_mutation_inbox",
            "service_mutation_inbox",
            "service_database_task_command_inbox",
        ):
            monkeypatch.setattr(server, service_name, forbidden_legacy_inbox)
        refresh_before = int(
            server.status()["read_replica"]["refresh_sequence"]
        )
        with DatabaseTaskSource(
            identity.listen_uri,
            owner_id="aseh-test-supervisor",
            install_schema=False,
        ) as source:
            snapshot = source.snapshot()
            assert snapshot.task_count == 1
            ready = source.ready_tasks(limit=10).tasks
            assert [item.task_alias for item in ready] == ["ASEH-000"]
            sealed_projection = source.plan_projection(
                task_cids=[ready[0].task_cid]
            )
            sealed_authority = aseh_operator._task_authority_spec_cids(  # noqa: SLF001
                sealed_projection
            )
            sealed_task = sealed_projection["tasks"][0]
            sealed_projection_spec_cid = str(sealed_task["spec_cid"])
            changed = source.compare_and_set_status(
                ready[0],
                ready[0].revision,
                "in_progress",
                receipt={
                    "operation": "database_claim",
                    "claim_id": "claim:aseh-bootstrap-test",
                    "attempt_id": "attempt:aseh-bootstrap-test",
                    "owner_session_id": "owner:aseh-bootstrap-test",
                    "lease_id": "lease:aseh-bootstrap-test",
                    "fencing_token": 1,
                    "fence_epoch": 1,
                    "claimed_from_revision": 1,
                },
            )
            assert changed.changed is True
            assert changed.task.status == "in_progress"
            requeued = source.compare_and_set_status(
                changed.task,
                changed.revision,
                "todo",
                receipt={
                    "operation": "requeue_unimplemented_stale_attempt",
                    "attempt_id": "attempt:aseh-bootstrap-test",
                    "unknown_callback_reopen_count": 1,
                },
            )
            assert requeued.changed is True
            assert requeued.task.status == "todo"
            assert requeued.task.revision == 3
            reclaimed = source.compare_and_set_status(
                requeued.task,
                requeued.revision,
                "in_progress",
                receipt={
                    "operation": "database_claim",
                    "claim_id": "claim:aseh-bootstrap-test:retry-1",
                    "attempt_id": "attempt:aseh-bootstrap-test:retry-1",
                    "owner_session_id": "owner:aseh-bootstrap-test",
                    "lease_id": "lease:aseh-bootstrap-test:retry-1",
                    "fencing_token": 2,
                    "fence_epoch": 1,
                    "claimed_from_revision": 3,
                },
            )
            assert reclaimed.changed is True
            assert reclaimed.task.status == "in_progress"
            assert reclaimed.task.revision == 4
            # A successful owner acknowledgement is also a synchronous
            # Quack-publication barrier.  Strict sharding re-reads this exact
            # binding immediately after reclaim and must not see a stale
            # ``todo`` projection.
            observed = source.get("ASEH-000")
            assert observed is not None
            assert observed.status == "in_progress"
            assert observed.revision == reclaimed.revision
            assert observed.body["unknown_callback_reopen_count"] == 1
            assert observed.body["completion_receipt"] == {
                "operation": "database_claim",
                "claim_id": "claim:aseh-bootstrap-test:retry-1",
                "attempt_id": "attempt:aseh-bootstrap-test:retry-1",
                "owner_session_id": "owner:aseh-bootstrap-test",
                "lease_id": "lease:aseh-bootstrap-test:retry-1",
                "fencing_token": 2,
                "fence_epoch": 1,
                "claimed_from_revision": 3,
                "unknown_callback_reopen_count": 1,
            }
            lifecycle_projection = source.plan_projection(
                task_cids=[observed.task_cid]
            )
            lifecycle_task = lifecycle_projection["tasks"][0]
            assert lifecycle_task["task_cid"] == sealed_task["task_cid"]
            assert lifecycle_task["spec_cid"] != sealed_projection_spec_cid
            assert (
                aseh_operator._task_authority_spec_cids(  # noqa: SLF001
                    lifecycle_projection
                )
                == sealed_authority
            )

            forged_task = {
                **lifecycle_task,
                "body": dict(lifecycle_task["body"]),
            }
            forged_task["body"]["acceptance_conditions"] = "reduced acceptance"
            assert (
                aseh_operator._task_authority_spec_cids(  # noqa: SLF001
                    {"tasks": [forged_task]}
                )
                != sealed_authority
            )
            # The canonical repository classifies a stale lower revision as
            # a bounds failure; the typed gateway must preserve that code.
            with pytest.raises(TaskSourceBoundsError):
                source.compare_and_set_status(ready[0], 0, "failed")
        live_status = server.status()
        assert live_status["read_replica"]["live"] is True
        assert int(live_status["read_replica"]["refresh_sequence"]) > refresh_before
        assert int(live_status["read_replica"]["refresh_sequence"]) >= 2
        mutation_dir = owner_dir / "mutations"
        assert not mutation_dir.exists() or not tuple(mutation_dir.glob("*.json"))
        assert server._connection is not None  # noqa: SLF001
        idempotency = server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM idempotency_records "
            "WHERE command_kind = 'compare_and_set_status'"
        ).fetchone()
        assert idempotency is not None and int(idempotency[0]) == 3

        descriptor = int(handoff[TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV])
        for safe_environment in (
            provider_subprocess_environment(os.environ),
            build_validation_environment(os.environ),
        ):
            assert TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV not in safe_environment
            assert (
                TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV
                not in safe_environment
            )
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,sys; fd=int(sys.argv[1]); "
                        "\ntry: os.fstat(fd)"
                        "\nexcept OSError: raise SystemExit(0)"
                        "\nraise SystemExit(1)"
                    ),
                    str(descriptor),
                ],
                env=safe_environment,
                check=False,
            )
            assert child.returncode == 0

        raw_transport_token = request_quack_attach_credential(
            store_id=str(database),
            client_id="aseh-test-persistence-audit",
            process_birth_id=kernel_process_birth_id(),
            timeout_seconds=2,
        )
        with pytest.raises(TypedStateOwnerError):
            TypedStateOwnerConnection(
                socket_path=owner_socket,
                token=raw_transport_token,
                client_id="raw-read-token-cannot-mutate",
                process_birth_id=kernel_process_birth_id(),
                store_id=str(database),
                timeout_seconds=2,
            )

        command_client_id = f"grant-inspection:{os.getpid()}"
        process_birth_id = kernel_process_birth_id()
        command_grant = request_database_task_command_credential(
            store_id=str(database),
            client_id=command_client_id,
            process_birth_id=process_birth_id,
            timeout_seconds=2,
        )
        copied = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json,sys; from pathlib import Path; "
                    "from ipfs_accelerate_py.agent_supervisor.task_sources."
                    "typed_state_owner import TypedStateOwnerConnection; "
                    "p=json.loads(sys.stdin.read()); "
                    "\ntry: c=TypedStateOwnerConnection(socket_path=Path(p['socket']), "
                    "token=p['token'], client_id=p['client'], "
                    "process_birth_id=p['birth'], store_id=p['store'], "
                    "timeout_seconds=2)"
                    "\nexcept BaseException: raise SystemExit(0)"
                    "\nc.close(); raise SystemExit(7)"
                ),
            ],
            input=json.dumps(
                {
                    "socket": str(owner_socket.resolve()),
                    "token": command_grant,
                    "client": command_client_id,
                    "birth": process_birth_id,
                    "store": str(database),
                }
            ),
            text=True,
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=False,
        )
        assert copied.returncode == 0
        command_connection = TypedStateOwnerConnection(
            socket_path=owner_socket,
            token=command_grant,
            client_id=command_client_id,
            process_birth_id=process_birth_id,
            store_id=str(database),
            timeout_seconds=2,
        )
        try:
            with pytest.raises(TypedStateOwnerError):
                TypedStateOwnerConnection(
                    socket_path=owner_socket,
                    token=command_grant,
                    client_id=command_client_id,
                    process_birth_id=process_birth_id,
                    store_id=str(database),
                    timeout_seconds=2,
                )
            assert set(
                command_connection.grant["allowed_database_task_commands"]
            ) == set(DATABASE_TASK_COMMANDS)
            assert command_connection.grant["allowed_operations"] == []
            assert command_connection.grant["allowed_command_operations"] == []
            assert (
                int(command_connection.grant["expires_at"])
                - int(command_connection.grant["issued_at"])
                <= 60_000
            )
        finally:
            command_connection.close()
        deadline = time.monotonic() + 2
        while (
            server.status()["typed_command_gateway"]["active_grants"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        with pytest.raises(TypedStateOwnerError):
            TypedStateOwnerConnection(
                socket_path=owner_socket,
                token=command_grant,
                client_id=command_client_id,
                process_birth_id=process_birth_id,
                store_id=str(database),
                timeout_seconds=2,
            )

        expiring_client_id = f"expiring-task-grant:{os.getpid()}"
        expiring_grant = server.issue_typed_client_grant(
            client_id=expiring_client_id,
            process_birth_id=process_birth_id,
            allowed_database_task_commands=("record_queue_retry",),
            peer_pid=os.getpid(),
            ttl_seconds=1,
        )
        server.issue_typed_client_grant(
            client_id=f"orphaned-task-grant:{os.getpid()}",
            process_birth_id=process_birth_id,
            allowed_database_task_commands=("record_queue_retry",),
            peer_pid=os.getpid(),
            ttl_seconds=1,
        )
        expiring_connection = TypedStateOwnerConnection(
            socket_path=owner_socket,
            token=expiring_grant,
            client_id=expiring_client_id,
            process_birth_id=process_birth_id,
            store_id=str(database),
            timeout_seconds=2,
        )
        try:
            time.sleep(1.05)
            with pytest.raises(TypedStateOwnerRemoteError) as expired:
                expiring_connection.execute_database_task_command(
                    "record_queue_retry",
                    {"task_cid": "task:aseh-bootstrap-test"},
                    command_request_id="f" * 32,
                )
            assert expired.value.error_code == "authorization_denied"
        finally:
            expiring_connection.close()
        assert server.status()["typed_command_gateway"]["active_grants"] == 0

        # Drop the first post-commit success response at the typed gateway.
        # The client must surface an unknown outcome for reconciliation and
        # must not manufacture a second command/request behind the caller's
        # back.
        assert server._connection is not None  # noqa: SLF001
        before_drop = server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM idempotency_records "
            "WHERE command_kind = 'record_queue_backoff'"
        ).fetchone()
        assert before_drop is not None
        real_send_frame = typed_state_owner_module._send_frame  # noqa: SLF001
        dropped = threading.Event()

        def drop_post_commit_response(
            channel: socket.socket,
            payload: dict[str, object],
        ) -> None:
            if (
                not dropped.is_set()
                and payload.get("ok") is True
                and "result" in payload
            ):
                dropped.set()
                try:
                    channel.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                channel.close()
                raise OSError("injected post-commit response loss")
            real_send_frame(channel, payload)

        monkeypatch.setattr(
            typed_state_owner_module,
            "_send_frame",
            drop_post_commit_response,
        )
        try:
            with DatabaseTaskSource(
                identity.listen_uri,
                owner_id="aseh-test-unknown-outcome",
                install_schema=False,
            ) as source:
                with pytest.raises(TaskSourceUnknownOutcomeError):
                    source.record_queue_backoff(
                        task_cid="task:aseh-bootstrap-test",
                        delay_ms=2_000,
                        reason="dropped-response-test",
                    )
        finally:
            monkeypatch.setattr(
                typed_state_owner_module,
                "_send_frame",
                real_send_frame,
            )
        assert dropped.is_set()
        after_drop = server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM idempotency_records "
            "WHERE command_kind = 'record_queue_backoff'"
        ).fetchone()
        assert after_drop is not None
        assert int(after_drop[0]) == int(before_drop[0]) + 1
        with DatabaseTaskSource(
            identity.listen_uri,
            owner_id="aseh-test-reconcile-observation",
            install_schema=False,
        ) as source:
            entry = source.get_queue_entry("task:aseh-bootstrap-test")
            assert entry is not None
            assert entry.reason == "dropped-response-test"

        raw_bootstrap_secret = os.pread(descriptor, 257, 0)
        for path in owner_dir.rglob("*"):
            if not path.is_file():
                continue
            body = path.read_bytes()
            assert raw_transport_token.encode("ascii") not in body
            assert raw_bootstrap_secret not in body

        # Status publication is part of the acknowledgement barrier too. A
        # failure after the command commit must quarantine the owner and
        # surface an unknown outcome, never an ordinary retryable error.
        assert server._connection is not None  # noqa: SLF001
        before_publication_failure = server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM idempotency_records "
            "WHERE command_kind = 'record_queue_retry'"
        ).fetchone()
        assert before_publication_failure is not None

        def fail_status_publication() -> None:
            raise OSError("injected status publication failure")

        real_write_status = server._write_status  # noqa: SLF001
        monkeypatch.setattr(server, "_write_status", fail_status_publication)
        try:
            with DatabaseTaskSource(
                identity.listen_uri,
                owner_id="aseh-test-publication-failure",
                install_schema=False,
            ) as source:
                with pytest.raises(TaskSourceUnknownOutcomeError):
                    source.record_queue_retry(
                        task_cid="task:aseh-bootstrap-test"
                    )
        finally:
            monkeypatch.setattr(server, "_write_status", real_write_status)
        after_publication_failure = server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM idempotency_records "
            "WHERE command_kind = 'record_queue_retry'"
        ).fetchone()
        assert after_publication_failure is not None
        assert int(after_publication_failure[0]) == (
            int(before_publication_failure[0]) + 1
        )
        assert server.status()["lifecycle"] == "failed"
        assert server.status()["read_replica"]["live"] is False
    finally:
        # This process is both the real owner and its configured-supervisor
        # client.  Release the client-side Quack pool before asking the owner
        # to prove that its endpoint closed; production processes reach the
        # same ordering when supervisor children terminate before the owner.
        reset_quack_transport_cache()
        server.stop()
        assert not any("transport stop warning" in item for item in server.logs())


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_live_broker_status_replays_only_an_exact_disposable_replica(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip(f"reviewed preinstalled Quack unavailable: {capability.status.value}")

    database = tmp_path / "control.duckdb"
    owner_dir = tmp_path / "owner"
    tree_id = "tree:aseh-bootstrap-test"
    monkeypatch.chdir(tmp_path)
    _materialize_one_task(database)
    with DatabaseTaskSource(
        database,
        install_schema=False,
        repository_tree_id=tree_id,
    ) as source:
        sealed_snapshot = source.snapshot().to_dict()
        goal = source.get_goal("goal:aseh-bootstrap-test")
        assert goal is not None
    bootstrap = {
        "schema": aseh_operator.BOOTSTRAP_SCHEMA,
        "source_head": "commit:aseh-bootstrap-test",
        "repository_tree_id": tree_id,
        "plan_root_cid": sealed_snapshot["plan_root_cid"],
        "source_forest": {},
        "source_identities": {},
        "database_task_source_receipt": {},
        "snapshot": sealed_snapshot,
        "integrity": {
            "goal_records": {
                "ASEH-G000": aseh_operator._immutable_goal_record(goal)
            }
        },
        "initial_ready_task_ids": ["ASEH-000"],
        "bootstrap_validation": {},
        "recovered_after_interrupted_materialization": False,
        "authority": {},
        "ducklake_projection": {},
    }
    bootstrap["bootstrap_receipt_id"] = aseh_operator._identity(bootstrap)
    bootstrap_path = tmp_path / "bootstrap.json"
    aseh_operator._atomic_json(bootstrap_path, bootstrap)
    paths = {
        "runtime": tmp_path,
        "database": database,
        "owner": owner_dir,
        "bootstrap_receipt": bootstrap_path,
    }
    server = build_server(
        database_path=database,
        state_dir=owner_dir,
        repository_root=tmp_path,
        port=0,
        store_id=str(database),
        secret_handle="handle:aseh-live-status-test",
    )
    identity = server.start()
    try:
        handoff = dict(server.start_supervisor_grant_broker())
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV],
        )
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV],
        )
        monkeypatch.setenv(
            TYPED_STATE_OWNER_SOCKET_ENV,
            handoff[TYPED_STATE_OWNER_SOCKET_ENV],
        )
        monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(database))
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION",
            str(identity.generation),
        )
        monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", raising=False)
        board = SimpleNamespace(
            resolved_database_program=lambda: SimpleNamespace(
                quack_endpoint=identity.listen_uri
            )
        )
        real_replay = DatabaseTaskSource.projection_matches_events
        replay_transports: list[bool] = []

        def reject_live_rebuild(source: DatabaseTaskSource) -> bool:
            replay_transports.append(source.intent.uses_quack_transport)
            if source.intent.uses_quack_transport:
                raise AssertionError("live Quack projection rebuild was invoked")
            return real_replay(source)

        monkeypatch.setattr(
            DatabaseTaskSource,
            "projection_matches_events",
            reject_live_rebuild,
        )
        with aseh_operator._LIVE_REPLAY_CACHE_LOCK:
            aseh_operator._LIVE_REPLAY_CACHE.clear()
        event_cursor_before = int(
            server._connection.execute(  # noqa: SLF001
                "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
            ).fetchone()[0]
        )
        owner_status = server.status()
        real_retire = aseh_operator._retire_live_replay_directory

        def fail_retirement(directory: Path) -> None:
            (directory / "unexpected-artifact").write_text(
                "not admitted\n", encoding="utf-8"
            )
            real_retire(directory)

        monkeypatch.setattr(
            aseh_operator,
            "_retire_live_replay_directory",
            fail_retirement,
        )
        for _attempt in range(2):
            with pytest.raises(
                aseh_operator.OperatorError,
                match="unexpected artifacts: unexpected-artifact",
            ):
                aseh_operator._broker_status_query(
                    board, paths, owner_status=owner_status
                )
            with aseh_operator._LIVE_REPLAY_CACHE_LOCK:
                assert aseh_operator._LIVE_REPLAY_CACHE == {}
        assert replay_transports == [False, False]
        replay_directories = tuple(tmp_path.glob(".live-projection-replay*"))
        assert len(replay_directories) == 2
        monkeypatch.setattr(
            aseh_operator,
            "_retire_live_replay_directory",
            real_retire,
        )
        for replay_directory in replay_directories:
            (replay_directory / "unexpected-artifact").unlink()
            real_retire(replay_directory)

        report = aseh_operator._broker_status_query(
            board, paths, owner_status=owner_status
        )
        assert report["available"] is True
        owner_binding_fields = (
            "server_id", "store_id", "database_uuid", "schema_revision",
            "schema_fingerprint", "generation", "process_birth_id",
            "listen_uri", "extension_fingerprint",
        )
        assert owner_status["storage_schema_fingerprint"] != (
            owner_status["identity"]["schema_fingerprint"]
        )
        assert report["owner_binding"] == {
            field: owner_status["identity"][field]
            for field in owner_binding_fields
        }
        assert report["ready_task_ids"] == ["ASEH-000"]
        assert report["event_cursor"] == event_cursor_before
        assert report["projection_matches_events"] is True
        witness = report["projection_reconciliation"]
        assert witness["authoritative"] is False
        assert witness["mutation_authority"] is False
        assert replay_transports == [False, False, False]
        assert not tuple(tmp_path.glob(".live-projection-replay*"))
        assert int(
            server._connection.execute(  # noqa: SLF001
                "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
            ).fetchone()[0]
        ) == event_cursor_before

        mismatched_identity = json.loads(json.dumps(owner_status))
        mismatched_identity["identity"]["generation"] += 1
        with pytest.raises(
            aseh_operator.OperatorError,
            match="owner binding differs from published owner",
        ):
            aseh_operator._broker_status_query(
                board, paths, owner_status=mismatched_identity
            )
        mismatched_storage = json.loads(json.dumps(owner_status))
        mismatched_storage["storage_schema_fingerprint"] = "storage:stale"
        with pytest.raises(
            aseh_operator.OperatorError,
            match="storage schema differs from published owner",
        ):
            aseh_operator._broker_status_query(
                board, paths, owner_status=mismatched_storage
            )

        # An exact cache hit still reopens and re-hashes the owner-published
        # bytes.  Same-sized path tampering cannot reuse the prior witness.
        replica_path = Path(owner_status["read_replica"]["path"])
        descriptor = os.open(replica_path, os.O_RDWR | os.O_CLOEXEC)
        try:
            original = os.pread(descriptor, 1, 0)
            assert original
            changed = bytes([original[0] ^ 0xFF])
            assert os.pwrite(descriptor, changed, 0) == 1
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        with pytest.raises(
            aseh_operator.OperatorError,
            match="published replica bytes differ",
        ):
            aseh_operator._admit_live_projection_shadow_replay(
                paths=paths,
                owner_status=owner_status,
                expected_snapshot=report["snapshot"],
            )
        assert replay_transports == [False, False, False]

        tampered_status = json.loads(json.dumps(server.status()))
        tampered_status["read_replica"]["sha256"] = f"sha256:{'0' * 64}"
        with pytest.raises(aseh_operator.OperatorError):
            aseh_operator._admit_live_projection_shadow_replay(
                paths=paths,
                owner_status=tampered_status,
                expected_snapshot=report["snapshot"],
            )
        assert int(
            server._connection.execute(  # noqa: SLF001
                "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
            ).fetchone()[0]
        ) == event_cursor_before
    finally:
        with aseh_operator._LIVE_REPLAY_CACHE_LOCK:
            aseh_operator._LIVE_REPLAY_CACHE.clear()
        reset_quack_transport_cache()
        server.stop()


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_typed_owner_never_acknowledges_an_unpublished_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip(
            f"reviewed preinstalled Quack unavailable: {capability.status.value}"
        )

    database = tmp_path / "control.duckdb"
    owner_dir = tmp_path / "quack-owner"
    monkeypatch.chdir(tmp_path)
    owner_socket = _cwd_owner_socket("typed-owner.sock")
    _materialize_one_task(database)
    server = build_server(
        database_path=database,
        state_dir=owner_dir,
        repository_root=tmp_path,
        port=0,
        store_id=str(database),
        secret_handle="handle:aseh-publication-failure-test",
        typed_command_socket_path=owner_socket,
    )
    identity = server.start()
    try:
        handoff = dict(server.start_supervisor_grant_broker())
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV],
        )
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV],
        )
        monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(owner_socket))
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(database)
        )
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION",
            str(identity.generation),
        )
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
            str(owner_dir / "mutations"),
        )

        def fail_publication(*_args: object, **_kwargs: object) -> dict[str, object]:
            raise RuntimeError("injected read-replica publication failure")

        monkeypatch.setattr(server, "_refresh_read_replica", fail_publication)

        with pytest.raises(QuackOwnerCommandRemoteError) as unknown:
            submit_quack_owner_command(
                QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS,
                {
                    "task_cid_or_alias": "ASEH-000",
                    "expected_revision": 1,
                    "status": "in_progress",
                    "receipt": {"operation": "database_claim"},
                    "evidence_digests": None,
                },
                timeout_seconds=0.5,
            )
        assert unknown.value.code == "read_replica_refresh_unknown_outcome"
        mutation_dir = owner_dir / "mutations"
        assert not mutation_dir.exists() or not tuple(
            mutation_dir.glob("*.json")
        )
        assert server._connection is not None  # noqa: SLF001
        row = server._connection.execute(  # noqa: SLF001
            "SELECT status, revision FROM tasks WHERE task_alias = 'ASEH-000'"
        ).fetchone()
        assert row is not None
        assert (str(row[0]), int(row[1])) == ("in_progress", 2)
        idempotency = server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM idempotency_records "
            "WHERE command_kind = 'compare_and_set_status'"
        ).fetchone()
        assert idempotency is not None and int(idempotency[0]) == 1
        assert server.status()["lifecycle"] == "failed"
        assert server.status()["read_replica"]["live"] is False
    finally:
        reset_quack_transport_cache()
        server.stop()


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_broker_denial_and_slow_peer_do_not_break_later_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip(f"reviewed preinstalled Quack unavailable: {capability.status.value}")

    database = tmp_path / "control.duckdb"
    owner_dir = tmp_path / "quack-owner"
    monkeypatch.chdir(tmp_path)
    _materialize_one_task(database)
    server = build_server(
        database_path=database,
        state_dir=owner_dir,
        repository_root=tmp_path,
        port=0,
        store_id=str(database),
        secret_handle="handle:aseh-broker-denial-test",
        typed_command_socket_path=_cwd_owner_socket("typed-owner.sock"),
    )
    server.start()
    wrong_fd = -1
    slow: socket.socket | None = None
    try:
        handoff = dict(server.start_supervisor_grant_broker())
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV],
        )
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV],
        )
        monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(database))

        wrong_fd = _sealed_memfd("0" * 64)
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV, str(wrong_fd)
        )
        with pytest.raises(TypedStateOwnerAuthorizationError, match="denied"):
            request_quack_attach_credential(
                store_id=str(database),
                client_id="aseh-test-wrong-secret",
                process_birth_id=kernel_process_birth_id(),
                timeout_seconds=2,
            )
        assert server.status()["configured_supervisor_credential_broker"][
            "available"
        ] is True

        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV],
        )
        with pytest.raises(TypedStateOwnerAuthorizationError, match="denied"):
            request_quack_attach_credential(
                store_id=str(database),
                client_id="caller-selected-label-is-diagnostic-only",
                process_birth_id="birth:caller-selected-not-kernel-derived",
                timeout_seconds=2,
            )
        assert server.status()["configured_supervisor_credential_broker"][
            "available"
        ] is True

        slow = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        slow.connect(handoff[TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV])
        started = time.monotonic()
        token = request_quack_attach_credential(
            store_id=str(database),
            client_id="aseh-test-after-slow-peer",
            process_birth_id=kernel_process_birth_id(),
            timeout_seconds=3,
        )
        assert token
        assert time.monotonic() - started < 2
        assert server.status()["configured_supervisor_credential_broker"][
            "available"
        ] is True

        def fail_resolver(
            _kind: str,
            _client: str,
            _birth: str,
            _pid: int,
        ) -> str:
            raise RuntimeError("injected broker authority failure")

        assert server._grant_broker is not None  # noqa: SLF001
        monkeypatch.setattr(
            server._grant_broker,  # noqa: SLF001
            "_resolve_credential",
            fail_resolver,
        )
        with pytest.raises(TypedStateOwnerAuthorizationError, match="denied"):
            request_quack_attach_credential(
                store_id=str(database),
                client_id="aseh-test-owner-failure",
                process_birth_id=kernel_process_birth_id(),
                timeout_seconds=2,
            )
        broker_status = server.status()[
            "configured_supervisor_credential_broker"
        ]
        assert broker_status["available"] is False
        assert broker_status["last_error_type"] == "RuntimeError"
        with pytest.raises(QuackStateServerReadyError, match="credential broker"):
            server.ready()
    finally:
        if slow is not None:
            slow.close()
        if wrong_fd >= 0:
            os.close(wrong_fd)
        server.stop()
        reset_quack_transport_cache()


def _aseh_health_fixture(
    tmp_path: Path,
    *,
    status: str = "todo",
    revision: int = 1,
    event_cursor: int = 10,
    ready: bool = True,
    active: bool = False,
    worker_count: int = 0,
    observed_at: float,
    lane_mtime_ns: int,
    lane_stalled: bool = False,
    delayed_retry_not_before_ms: int = 0,
) -> tuple[SimpleNamespace, dict[str, Path], dict[str, object]]:
    task_cid = "task:aseh-health"
    goal_record = {
        "goal_cid": "goal:aseh-health",
        "goal_alias": "ASEH-G000",
        "objective_id": "objective:aseh-root",
        "parent_goal_cid": "",
        "ordinal": 1,
        "title": "ASEH health",
        "body": {"priority": "P0"},
    }
    plan_record = {
        "plan_cid": "plan:aseh-health",
        "goal_cid": "goal:aseh-health",
        "plan_alias": "ASEH-PLAN-R1",
        "body": {"plan_cid": "plan:aseh-health"},
    }
    objective_record = {
        "objective_id": "objective:aseh-root",
        "objective_alias": "ASEH-G000",
        "parent_objective_id": "",
        "title": "ASEH root objective",
        "priority": "P0",
        "body": {"program_id": aseh_operator.PROGRAM},
        "extension_schema": "",
        "extension": {},
    }
    bootstrap_snapshot = {
        "source_schema": "source@1",
        "schema_version": "1",
        "plan_root_cid": "plan:aseh-health",
        "repository_tree_id": "tree:aseh-health",
        "formal_plan_id": "formal:aseh-health",
        "source_identity": "",
        "projection_cid": "projection:aseh-health",
        "event_cursor": 10,
        "task_count": 1,
        "goal_count": 1,
        "dependency_count": 0,
        "objective_count": 1,
        "plan_count": 1,
    }
    bootstrap_snapshot["source_identity"] = aseh_operator.content_identity(
        {
            "plan_root_cid": bootstrap_snapshot["plan_root_cid"],
            "repository_tree_id": bootstrap_snapshot["repository_tree_id"],
            "projection_cid": bootstrap_snapshot["projection_cid"],
        }
    )
    integrity = {
        "schema": "ipfs_accelerate_py/agent-supervisor/aseh-integrity@1",
        "projection_matches_events": True,
        "projection_cid": "projection:aseh-health",
        "event_cursor": 10,
        "task_statuses": {"ASEH-000": "todo"},
        "task_revisions": {"ASEH-000": 1},
        "task_cids": {"ASEH-000": task_cid},
        "task_owner_bindings": {
            "ASEH-000": {
                "owning_repository": "ipfs_accelerate_py",
                "base_revision": "commit:aseh-health",
                "base_repository_tree_id": "tree:aseh-health",
                "source_forest_cid": "forest:aseh-health",
                "owner_source_identity": "source-owner:aseh-health",
            }
        },
        "task_dependencies": {"ASEH-000": []},
        "task_authority_spec_cids": {
            "ASEH-000": "sha256:aseh-health-authority-spec"
        },
        "goal_records": {"ASEH-G000": goal_record},
        "goal_edges": [],
        "plan_record": plan_record,
        "objective_record": objective_record,
        "task_count": 1,
        "goal_count": 1,
        "dependency_count": 0,
        "objective_count": 1,
        "plan_count": 1,
    }
    integrity["integrity_receipt_id"] = aseh_operator._identity(integrity)
    bootstrap = {
        "schema": aseh_operator.BOOTSTRAP_SCHEMA,
        "source_head": "commit:aseh-health",
        "repository_tree_id": "tree:aseh-health",
        "plan_root_cid": "plan:aseh-health",
        "source_forest": {"forest_cid": "forest:aseh-health"},
        "source_identities": {"operator": "source:operator"},
        "database_task_source_receipt": {},
        "snapshot": bootstrap_snapshot,
        "integrity": integrity,
        "initial_ready_task_ids": ["ASEH-000"],
        "bootstrap_validation": {},
        "recovered_after_interrupted_materialization": False,
        "authority": {},
        "ducklake_projection": {},
    }
    bootstrap["bootstrap_receipt_id"] = aseh_operator._identity(bootstrap)
    bootstrap_path = tmp_path / "bootstrap.json"
    aseh_operator._atomic_json(bootstrap_path, bootstrap)

    process_birth = current_process_birth().to_dict()
    binding = {
        "server_id": "server:aseh-health",
        "store_id": "store:aseh-health",
        "database_uuid": "database:aseh-health",
        "schema_revision": 3,
        "schema_fingerprint": "schema:aseh-health",
        "generation": 9,
        "process_birth_id": aseh_operator._state_owner_process_birth_id(
            process_birth
        ),
        "listen_uri": "quack:127.0.0.1:1",
        "extension_fingerprint": "extensions:aseh-health",
    }
    owner_identity = {
        **binding,
        "process_birth": process_birth,
        "status": "ready",
    }
    current_snapshot = {**bootstrap_snapshot, "event_cursor": event_cursor}
    database_path = tmp_path / "control.duckdb"
    replica_binding = {
        "path": str(tmp_path / "control.read-replica.duckdb"),
        "source_database_path": str(database_path),
        "server_id": binding["server_id"],
        "database_uuid": binding["database_uuid"],
        "generation": binding["generation"],
        "schema_revision": binding["schema_revision"],
        "schema_fingerprint": binding["schema_fingerprint"],
        "storage_schema_fingerprint": "storage-schema:aseh-health",
        "sha256": f"sha256:{'a' * 64}",
        "size_bytes": 1,
        "refresh_sequence": 1,
    }
    replay_witness = {
        "schema": aseh_operator.LIVE_REPLAY_SCHEMA,
        "method": "disposable_exact_owner_published_replica_replay",
        "authoritative": False,
        "mutation_authority": False,
        "projection_matches_events": True,
        "replica": replica_binding,
        "projection_cid": current_snapshot["projection_cid"],
        "event_cursor": current_snapshot["event_cursor"],
        "cache_key": aseh_operator._identity(
            {
                "replica": replica_binding,
                "projection_cid": current_snapshot["projection_cid"],
                "event_cursor": current_snapshot["event_cursor"],
                "plan_root_cid": current_snapshot["plan_root_cid"],
                "repository_tree_id": current_snapshot["repository_tree_id"],
            }
        ),
    }
    replay_witness["witness_cid"] = aseh_operator._identity(replay_witness)
    ready_task_ids = ["ASEH-000"] if ready else []
    authority = {
        "available": True,
        "transport": "quack",
        "credential_path": "sealed_memfd_broker",
        "projection_matches_events": True,
        "projection_reconciliation": replay_witness,
        "owner_binding": binding,
        "snapshot": current_snapshot,
        "task_statuses": {"ASEH-000": status},
        "task_revisions": {"ASEH-000": revision},
        "task_cids": {"ASEH-000": task_cid},
        "task_owner_bindings": integrity["task_owner_bindings"],
        "task_dependencies": integrity["task_dependencies"],
        "task_authority_spec_cids": integrity["task_authority_spec_cids"],
        "goal_records": integrity["goal_records"],
        "goal_edges": integrity["goal_edges"],
        "plan_record": integrity["plan_record"],
        "objective_record": integrity["objective_record"],
        "ready_task_ids": ready_task_ids,
        "ready_count": len(ready_task_ids),
        "queue_entries": {
            "ASEH-000": {
                "task_cid": task_cid,
                "retry_not_before_ms": delayed_retry_not_before_ms,
            }
        } if delayed_retry_not_before_ms else {},
        "query_started_at_ms": int(observed_at * 1_000),
        "delayed_ready_task_ids": (
            ["ASEH-000"] if delayed_retry_not_before_ms else []
        ),
        "active_count": int(active),
        "blocked_count": 0,
        "terminal_count": int(status in aseh_operator.TERMINAL_STATUSES),
        "event_cursor": event_cursor,
    }
    sample: dict[str, object] = {
        "observed_at": observed_at,
        "scheduler": {
            "pid": 4242,
            "process_group": 4242,
            "alive": True,
            "returncode": None,
        },
        "owner_status": {
            "lifecycle": "ready",
            "database_path": str(database_path),
            "storage_schema_fingerprint": "storage-schema:aseh-health",
            "identity": owner_identity,
            "read_replica": {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "read-replica-observation@1"
                ),
                "authority": "non_authoritative_read_replica",
                "live": True,
                **replica_binding,
            },
            "configured_supervisor_credential_broker": {
                "available": True,
                "last_error_type": "",
            },
        },
        "authority": authority,
        "lanes": [
            {
                "lane": 0,
                "fresh": True,
                "admissible": True,
                "watchdog_admissible": True,
                "mtime_ns": lane_mtime_ns,
                "worker_metrics_available": True,
                "worker_census_method": "linux-procfs-descendant-census@1",
                "worker_root_pid": 4321,
                "worker_root_start_time_ticks": 987654,
                "worker_root_boot_id": "boot-id",
                "worker_root_identity_source": "supervised_child_identity",
                "active_worker_count": worker_count,
                "active_worker_pids": list(range(5000, 5000 + worker_count)),
                "worker_descendant_count": worker_count,
                "worker_descendant_pids": list(
                    range(5000, 5000 + worker_count)
                ),
                "worker_phase_guarded": lane_stalled,
                "worker_phase_available": lane_stalled,
                "worker_phase_age_seconds": 0.5 if lane_stalled else None,
                "worker_stall_evidence_available": lane_stalled,
                "stalled_without_active_worker": (
                    True if lane_stalled else None
                ),
            }
        ],
    }
    board = SimpleNamespace(
        max_lanes=1,
        payload={
            "stale_seconds": 2.0,
            "watchdog_startup_grace_seconds": 1.0,
            "initial_projection": {
                "task_count": 1,
                "goal_count": 1,
                "task_dependency_count": 0,
            },
        },
    )
    return board, {
        "bootstrap_receipt": bootstrap_path,
        "database": database_path,
        "runtime": tmp_path,
    }, sample


def test_aseh_status_sample_rejects_replica_generation_change_during_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    board, paths, sample = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    before = json.loads(json.dumps(sample["owner_status"]))
    after = json.loads(json.dumps(before))
    after["read_replica"]["refresh_sequence"] += 1
    observations = iter((before, after) * 3)
    server = SimpleNamespace(status=lambda: next(observations))
    scheduler = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
    broker_calls = []
    monkeypatch.setattr(
        aseh_operator,
        "_broker_status_query",
        lambda *_args, **_kwargs: broker_calls.append(True)
        or {"available": True},
    )
    monkeypatch.setattr(
        aseh_operator,
        "_lane_status_observations",
        lambda *_args, **_kwargs: [],
    )

    observed = aseh_operator._status_sample(
        board, paths, server, scheduler
    )

    assert observed["authority"]["available"] is False
    assert observed["authority"]["error_type"] == "OperatorError"
    assert observed["owner_status"]["read_replica"]["refresh_sequence"] == (
        after["read_replica"]["refresh_sequence"]
    )
    assert len(broker_calls) == 3


def test_aseh_status_sample_retries_the_whole_query_until_replica_is_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    board, paths, sample = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    before = json.loads(json.dumps(sample["owner_status"]))
    after = json.loads(json.dumps(before))
    after["read_replica"]["refresh_sequence"] += 1
    observations = iter((before, after, after, after))
    server = SimpleNamespace(status=lambda: next(observations))
    scheduler = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
    broker_calls = []
    monkeypatch.setattr(
        aseh_operator,
        "_broker_status_query",
        lambda *_args, **_kwargs: broker_calls.append(True)
        or {"available": True},
    )
    monkeypatch.setattr(
        aseh_operator,
        "_lane_status_observations",
        lambda *_args, **_kwargs: [],
    )

    observed = aseh_operator._status_sample(
        board, paths, server, scheduler
    )

    assert observed["authority"]["available"] is True
    assert len(broker_calls) == 2
    assert observed["owner_status"]["read_replica"]["refresh_sequence"] == (
        after["read_replica"]["refresh_sequence"]
    )


def test_aseh_external_status_binds_receipt_to_current_owner_incarnation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now,
        failure={},
    )
    assert receipt["healthy"] is True
    owner_directory = tmp_path / "owner"
    status_receipt = tmp_path / "live-status.json"
    paths.update(
        {
            "owner": owner_directory,
            "status_receipt": status_receipt,
        }
    )
    owner_status_path = owner_directory / "quack-state-server.status.json"
    aseh_operator._atomic_json(owner_status_path, current["owner_status"])
    aseh_operator._atomic_json(status_receipt, receipt)
    monkeypatch.setattr(aseh_operator, "_load", lambda _path: (board, {}))
    monkeypatch.setattr(aseh_operator, "_paths", lambda _board: paths)

    exit_code, exact = aseh_operator.status(
        tmp_path / "unused-config.json", require_ready=True
    )
    assert exit_code == 0
    assert exact["healthy"] is True
    assert exact["broker_authenticated_receipt"] is True

    # A recent healthy receipt from the prior owner must not make a newly
    # ready incarnation healthy after restart.
    restarted = json.loads(json.dumps(current["owner_status"]))
    restarted_identity = restarted["identity"]
    restarted_identity["server_id"] = "server:aseh-health-restarted"
    restarted_identity["database_uuid"] = "database:aseh-health-restarted"
    restarted_identity["generation"] += 1
    restarted_identity["process_birth_id"] = "birth:aseh-health-restarted"
    restarted_identity["process_birth"] = {"pid": os.getpid() + 1}
    restarted_identity["listen_uri"] = "quack:127.0.0.1:2"
    restarted_replica = restarted["read_replica"]
    for field in (
        "server_id", "database_uuid", "generation", "schema_revision",
        "schema_fingerprint",
    ):
        restarted_replica[field] = restarted_identity[field]
    restarted_replica["refresh_sequence"] += 1
    restarted_replica["sha256"] = f"sha256:{'b' * 64}"
    aseh_operator._atomic_json(owner_status_path, restarted)

    exit_code, stale = aseh_operator.status(
        tmp_path / "unused-config.json", require_ready=True
    )
    assert exit_code == 1
    assert stale["owner_ready"] is True
    assert stale["healthy"] is False
    assert stale["broker_authenticated_receipt"] is False
    assert stale["receipt_error"]["reason"] == (
        "live_status_receipt_unavailable_or_invalid"
    )


def test_aseh_external_status_rejects_abruptly_dead_bound_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        child_birth = read_process_birth(child.pid)
        assert child_birth is not None
        now = time.time()
        board, paths, before = _aseh_health_fixture(
            tmp_path,
            observed_at=now - 0.25,
            lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
        )
        _board, _paths, current = _aseh_health_fixture(
            tmp_path,
            observed_at=now,
            lane_mtime_ns=int(now * 1_000_000_000),
        )
        receipt = aseh_operator._health_receipt(
            board,
            paths,
            samples=(before, current),
            launched_at=now - 0.5,
            last_progress_at=now,
            failure={},
        )
        assert receipt["healthy"] is True
        child_birth_payload = child_birth.to_dict()
        child_birth_id = aseh_operator._state_owner_process_birth_id(
            child_birth_payload
        )
        receipt = json.loads(json.dumps(receipt))
        for sample in receipt["samples"]:
            sample_identity = sample["owner_status"]["identity"]
            sample_identity["process_birth"] = child_birth_payload
            sample_identity["process_birth_id"] = child_birth_id
            sample["authority"]["owner_binding"][
                "process_birth_id"
            ] = child_birth_id
        unsigned_receipt = dict(receipt)
        unsigned_receipt.pop("receipt_cid")
        receipt["receipt_cid"] = aseh_operator._identity(unsigned_receipt)

        child_status = json.loads(json.dumps(current["owner_status"]))
        child_status["identity"]["process_birth"] = child_birth_payload
        child_status["identity"]["process_birth_id"] = child_birth_id
        owner_directory = tmp_path / "owner"
        status_receipt = tmp_path / "live-status.json"
        paths.update(
            {
                "owner": owner_directory,
                "status_receipt": status_receipt,
            }
        )
        reused_status = json.loads(json.dumps(child_status))
        reused_birth = reused_status["identity"]["process_birth"]
        reused_birth["start_time_ticks"] += 1
        reused_status["identity"][
            "process_birth_id"
        ] = aseh_operator._state_owner_process_birth_id(reused_birth)
        with pytest.raises(aseh_operator.OperatorError, match="not alive"):
            aseh_operator._owner_incarnation_binding(reused_status, paths)

        owner_status_path = owner_directory / "quack-state-server.status.json"
        aseh_operator._atomic_json(owner_status_path, child_status)
        aseh_operator._atomic_json(status_receipt, receipt)
        monkeypatch.setattr(aseh_operator, "_load", lambda _path: (board, {}))
        monkeypatch.setattr(aseh_operator, "_paths", lambda _board: paths)

        exit_code, alive = aseh_operator.status(
            tmp_path / "unused-config.json", require_ready=True
        )
        assert exit_code == 0
        assert alive["healthy"] is True

        child.kill()
        assert child.wait(timeout=5) == -signal.SIGKILL
        exit_code, dead = aseh_operator.status(
            tmp_path / "unused-config.json", require_ready=True
        )
        assert exit_code == 1
        assert dead["owner_ready"] is True
        assert dead["healthy"] is False
        assert dead["broker_authenticated_receipt"] is False
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_aseh_health_heartbeat_only_cannot_mask_stuck_board(
    tmp_path: Path,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        observed_at=now - 1.0,
        lane_mtime_ns=int((now - 1.0) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 100.0,
        last_progress_at=now - 100.0,
        failure={},
    )

    assert receipt["progress_evidence"] == []
    assert receipt["liveness_evidence"] == ["lane_heartbeat_advanced"]
    assert receipt["stuck"] is True
    assert receipt["healthy"] is False

    cursor_only = json.loads(json.dumps(current))
    cursor_only["authority"]["event_cursor"] = 11
    cursor_only["authority"]["snapshot"]["event_cursor"] = 11
    assert aseh_operator._authoritative_progress_between(
        current, cursor_only
    ) == []

    current["lanes"][0]["stalled_without_active_worker"] = True  # type: ignore[index]
    stalled = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now,
        failure={},
    )
    assert stalled["lane_stalled_without_active_worker"] is True
    assert stalled["stuck"] is True
    assert stalled["healthy"] is False


def test_aseh_health_requires_two_sample_semantic_authority_and_exact_terminal(
    tmp_path: Path,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        observed_at=now - 1.0,
        lane_mtime_ns=int((now - 1.0) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    forged = json.loads(json.dumps(current))
    forged["authority"].pop("projection_reconciliation")
    forged["authority"]["projection_matches_events"] = True
    forged_receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, forged),
        launched_at=now - 0.5,
        last_progress_at=now,
        failure={},
    )
    assert forged_receipt["source_identity_admitted"] is False
    assert forged_receipt["healthy"] is False

    before["authority"]["objective_record"]["title"] = "amended"  # type: ignore[index]
    repaired = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now,
        failure={},
    )
    assert repaired["semantic_corpus_admitted"] is False
    assert repaired["healthy"] is False
    assert aseh_operator._post_admission_health_action(
        repaired,
        prior_available=True,
        current_available=True,
        unhealthy_edges=0,
    )[:2] == ("fail", "authoritative_health_admission_lost")

    _board, _paths, terminal_before = _aseh_health_fixture(
        tmp_path,
        status="failed",
        revision=2,
        ready=False,
        observed_at=now - 1.0,
        lane_mtime_ns=int((now - 1.0) * 1_000_000_000),
    )
    _board, _paths, terminal_current = _aseh_health_fixture(
        tmp_path,
        status="failed",
        revision=2,
        ready=False,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    terminal_receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(terminal_before, terminal_current),
        launched_at=now - 10.0,
        last_progress_at=now - 1.0,
        failure={},
    )
    assert terminal_receipt["terminal"] is True
    assert terminal_receipt["healthy"] is True
    assert aseh_operator._post_admission_health_action(
        terminal_receipt,
        prior_available=True,
        current_available=True,
        unhealthy_edges=0,
    )[:2] == ("stop", "")

    terminal_current["authority"]["task_authority_spec_cids"][  # type: ignore[index]
        "ASEH-000"
    ] = "b" + ("a" * 60)
    rejected_terminal = aseh_operator._health_receipt(
        board,
        paths,
        samples=(terminal_before, terminal_current),
        launched_at=now - 10.0,
        last_progress_at=now - 1.0,
        failure={},
    )
    assert rejected_terminal["terminal"] is True
    assert rejected_terminal["healthy"] is False
    assert aseh_operator._post_admission_health_action(
        rejected_terminal,
        prior_available=True,
        current_available=True,
        unhealthy_edges=0,
    )[:2] == ("fail", "authoritative_terminal_not_admitted")


def test_aseh_restart_admits_only_monotonic_lifecycle_on_sealed_corpus(
    tmp_path: Path,
) -> None:
    now = time.time()
    _board, paths, sample = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    bootstrap = aseh_operator._secure_runtime_json(
        paths["bootstrap_receipt"],
        max_bytes=aseh_operator.STATUS_RECEIPT_MAX_BYTES,
    )
    snapshot = dict(bootstrap["snapshot"])
    snapshot["event_cursor"] = int(snapshot["event_cursor"]) + 1
    snapshot["projection_cid"] = "projection:advanced"
    snapshot["source_identity"] = aseh_operator.content_identity(
        {
            "plan_root_cid": snapshot["plan_root_cid"],
            "repository_tree_id": snapshot["repository_tree_id"],
            "projection_cid": snapshot["projection_cid"],
        }
    )
    integrity = json.loads(json.dumps(bootstrap["integrity"]))
    integrity["event_cursor"] = snapshot["event_cursor"]
    integrity["task_revisions"]["ASEH-000"] += 1
    integrity["task_statuses"]["ASEH-000"] = "in_progress"
    integrity["projection_cid"] = "projection:advanced"
    integrity["integrity_receipt_id"] = aseh_operator._identity(
        {
            key: value
            for key, value in integrity.items()
            if key != "integrity_receipt_id"
        }
    )
    aseh_operator._admit_current_projection_against_bootstrap(
        bootstrap, snapshot, integrity
    )

    integrity["objective_record"]["title"] = "amended"
    with pytest.raises(aseh_operator.OperatorError, match="immutable corpus"):
        aseh_operator._admit_current_projection_against_bootstrap(
            bootstrap, snapshot, integrity
        )


def test_aseh_lane_status_projects_worker_watchdog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aseh_operator, "ROOT", tmp_path)
    state_root = tmp_path / "state"
    status_path = state_root / "lane-0" / "aseh_lane_0_supervisor_status.json"
    status_path.parent.mkdir(parents=True)
    payload = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "todo_implementation_supervisor.supervisor"
        ),
        "repo_root": str(tmp_path),
        "task_prefix": "## ASEH-",
        "state_prefix": "aseh_lane_0",
        "status": "running",
        "run_id": "run-1",
        "daemon_pid": 4321,
        "worker_metrics_available": True,
        "worker_metrics_unavailable_reason": "",
        "worker_census_method": "linux-procfs-descendant-census@1",
        "worker_root_pid": 4321,
        "worker_root_start_time_ticks": 987654,
        "worker_root_boot_id": "boot-id",
        "worker_root_identity_source": "supervised_child_identity",
        "worker_observed_at_ns": time.time_ns(),
        "worker_observation_generation": (
            "run-1:4321:987654:boot-id"
        ),
        "active_worker_count": 0,
        "active_worker_pids": [],
        "worker_descendant_count": 0,
        "worker_descendant_pids": [],
        "worker_phase": "",
        "worker_phase_available": False,
        "worker_phase_known": True,
        "worker_phase_known_non_worktree": False,
        "worker_phase_guarded": False,
        "worker_phase_age_seconds": None,
        "worker_stall_evidence_available": False,
        "worker_stall_evidence_unavailable_reason": "phase_not_guarded",
        "stalled_without_active_worker": None,
    }
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    board = SimpleNamespace(
        max_lanes=1,
        task_prefix="ASEH",
        task_header_prefix="## ASEH-",
        repo_root=tmp_path,
        runtime_paths={"state": "state"},
        path=lambda value: tmp_path / Path(value),
    )

    observations = aseh_operator._lane_status_observations(
        board, now=time.time()
    )

    assert observations[0]["watchdog_admissible"] is True
    assert observations[0]["active_worker_count"] == 0
    assert observations[0]["worker_phase_age_seconds"] is None
    assert observations[0]["stalled_without_active_worker"] is None

    # A no-work maintenance pass publishes this fresh, quiescent state until
    # the next managed-daemon cycle.  It remains an admitted live lane when
    # the exact worker census and watchdog evidence above are current.
    payload["status"] = "agentic_maintenance_completed"
    payload["worker_observed_at_ns"] = time.time_ns()
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    maintenance_completed = aseh_operator._lane_status_observations(
        board, now=time.time()
    )
    assert maintenance_completed[0]["admissible"] is True
    assert maintenance_completed[0]["watchdog_admissible"] is True

    payload["status"] = "agentic_maintenance_failed"
    payload["worker_observed_at_ns"] = time.time_ns()
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    maintenance_failed = aseh_operator._lane_status_observations(
        board, now=time.time()
    )
    assert maintenance_failed[0]["admissible"] is False

    payload["status"] = "running"

    payload["worker_observed_at_ns"] = time.time_ns() + 250_000_000
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    future_obs = aseh_operator._lane_status_observations(
        board, now=time.time()
    )
    assert future_obs[0]["watchdog_admissible"] is False
    assert future_obs[0]["worker_observation_age_seconds"] < 0.0
    payload["worker_observed_at_ns"] = time.time_ns()

    payload.update(
        {
            "worker_phase": "validating_reconciled_candidate",
            "worker_phase_available": True,
            "worker_phase_known": True,
            "worker_phase_known_non_worktree": True,
        }
    )
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    known_non_worktree = aseh_operator._lane_status_observations(
        board,
        now=time.time(),
    )
    assert known_non_worktree[0]["watchdog_admissible"] is True

    payload.update(
        {
            "worker_phase": "implementng",
            "worker_phase_available": True,
            "worker_phase_known": False,
            "worker_phase_known_non_worktree": False,
            "worker_stall_evidence_unavailable_reason": "phase_unknown",
        }
    )
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    unknown_phase = aseh_operator._lane_status_observations(
        board,
        now=time.time(),
    )
    assert unknown_phase[0]["watchdog_admissible"] is False

    payload.update(
        {
            "worker_phase": "validating",
            "worker_phase_available": True,
            "worker_phase_known": True,
            "worker_phase_known_non_worktree": True,
            "worker_stall_evidence_unavailable_reason": "phase_not_guarded",
        }
    )
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    validating = aseh_operator._lane_status_observations(
        board,
        now=time.time(),
    )
    assert validating[0]["watchdog_admissible"] is True
    assert validating[0]["worker_phase_known_non_worktree"] is True

    payload.update(
        {
            "worker_phase": "implementng",
            "worker_phase_available": True,
            "worker_phase_known": True,
            "worker_phase_known_non_worktree": True,
        }
    )
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    forged_phase = aseh_operator._lane_status_observations(
        board,
        now=time.time(),
    )
    assert forged_phase[0]["watchdog_admissible"] is False

    payload.update(
        {
            "worker_phase": "",
            "worker_phase_available": False,
            "worker_phase_known": None,
            "worker_phase_known_non_worktree": None,
            "worker_phase_guarded": None,
            "worker_stall_evidence_unavailable_reason": "worker_metrics_unavailable",
            "worker_metrics_available": False,
            "worker_metrics_unavailable_reason": "procfs_unavailable",
            "active_worker_count": None,
            "active_worker_pids": None,
            "worker_descendant_count": None,
            "worker_descendant_pids": None,
        }
    )
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    unavailable = aseh_operator._lane_status_observations(
        board,
        now=time.time(),
    )
    assert unavailable[0]["watchdog_admissible"] is False
    assert unavailable[0]["active_worker_count"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("worker_root_boot_id", ""),
        ("worker_root_identity_source", "captured_before_census"),
        ("worker_root_pid", 4322),
        ("worker_observed_at_ns", 1),
        ("worker_observed_at_ns", time.time_ns() + 60_000_000_000),
        ("worker_observation_generation", "run-1:4321:1:boot-id"),
    ],
)
def test_aseh_lane_status_rejects_unsealed_worker_root_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    monkeypatch.setattr(aseh_operator, "ROOT", tmp_path)
    state_root = tmp_path / "state"
    status_path = state_root / "lane-0" / "aseh_lane_0_supervisor_status.json"
    status_path.parent.mkdir(parents=True)
    payload = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "todo_implementation_supervisor.supervisor"
        ),
        "repo_root": str(tmp_path),
        "task_prefix": "## ASEH-",
        "state_prefix": "aseh_lane_0",
        "status": "running",
        "run_id": "run-1",
        "daemon_pid": 4321,
        "worker_metrics_available": True,
        "worker_census_method": "linux-procfs-descendant-census@1",
        "worker_root_pid": 4321,
        "worker_root_start_time_ticks": 987654,
        "worker_root_boot_id": "boot-id",
        "worker_root_identity_source": "supervised_child_identity",
        "worker_observed_at_ns": time.time_ns(),
        "worker_observation_generation": (
            "run-1:4321:987654:boot-id"
        ),
        "active_worker_count": 0,
        "active_worker_pids": [],
        "worker_descendant_count": 0,
        "worker_descendant_pids": [],
        "worker_phase": "",
        "worker_phase_available": False,
        "worker_phase_known": True,
        "worker_phase_known_non_worktree": False,
        "worker_phase_guarded": False,
        "worker_stall_evidence_available": False,
        "worker_stall_evidence_unavailable_reason": "phase_not_guarded",
        "stalled_without_active_worker": None,
        field: value,
    }
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    board = SimpleNamespace(
        max_lanes=1,
        task_prefix="ASEH",
        task_header_prefix="## ASEH-",
        repo_root=tmp_path,
        runtime_paths={"state": "state"},
        path=lambda item: tmp_path / Path(item),
    )

    observations = aseh_operator._lane_status_observations(
        board,
        now=time.time(),
    )

    assert observations[0]["watchdog_admissible"] is False


def test_known_non_worktree_phase_allowlist_is_exact_and_not_guarded() -> None:
    expected = frozenset(
        {
            "merge_queue",
            "merge_reconciliation",
            "validating",
            "validating_reconciled_candidate",
        }
    )
    assert todo_supervisor.KNOWN_NON_WORKTREE_PHASES == expected

    for phase in sorted(expected):
        status = todo_supervisor.worktree_phase_worker_status(
            {"active_phase": phase},
            daemon_pid=1234,
            threshold_seconds=60,
            descendants=[],
        )
        assert status["required"] is False
        assert status["phase_known"] is True
        assert status["phase_known_non_worktree"] is True
        assert status["stall_evidence_unavailable_reason"] == "phase_not_guarded"
        assert status["stalled_without_active_worker"] is None

    unknown = todo_supervisor.worktree_phase_worker_status(
        {"active_phase": "implementng"},
        daemon_pid=1234,
        threshold_seconds=60,
        descendants=[],
    )
    assert unknown["phase_known"] is False
    assert unknown["phase_known_non_worktree"] is False
    assert unknown["stall_evidence_unavailable_reason"] == "phase_unknown"


def test_worker_observation_binds_exact_run_child_and_root_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = object.__new__(SupervisorLoop)
    loop.last_run_id = "run-7"
    loop._last_worker_status = {}
    monkeypatch.setattr(
        supervisor_loop_module.time,
        "time_ns",
        lambda: 1_700_000_000_000_000_000,
    )

    observed = loop._record_worker_observation(
        SimpleNamespace(pid=4321),
        {
            "worker_metrics_available": True,
            "worker_root_start_time_ticks": 987654,
            "worker_root_boot_id": "boot-id",
        },
    )

    assert observed["worker_observed_at_ns"] == 1_700_000_000_000_000_000
    assert observed["worker_observation_generation"] == (
        "run-7:4321:987654:boot-id"
    )
    assert loop._last_worker_status == observed

    unavailable = loop._record_worker_observation(
        SimpleNamespace(pid=4321),
        {"worker_metrics_available": False},
    )
    assert unavailable["worker_observed_at_ns"] == 1_700_000_000_000_000_000
    assert unavailable["worker_observation_generation"] == ""


def test_supervisor_loop_publishes_worker_census_before_startup_grace(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = repo / "state"
    state_dir.mkdir(parents=True)
    identity_path = state_dir / "child.identity.json"
    command = (
        sys.executable,
        "-c",
        "import time; time.sleep(0.2)",
    )
    spec = ManagedDaemonSpec(
        name="aseh-census-test",
        schema="test.aseh-census",
        repo_root=repo,
        daemon_dir=state_dir,
        runner=command,
        status_path=state_dir / "daemon_status.json",
        supervisor_status_path=state_dir / "supervisor_status.json",
        supervisor_pid_path=state_dir / "supervisor.pid",
        child_pid_path=state_dir / "child.pid",
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure_status.json",
        ensure_check_path=state_dir / "ensure_check.json",
    )
    watchdog_calls: list[bool] = []
    loop = SupervisorLoop(
        SupervisorLoopConfig(
            spec=spec,
            command=command,
            log_prefix="child",
            heartbeat_seconds=0.01,
            poll_seconds=0.01,
            watchdog_startup_grace_seconds=3600,
            max_restarts=1,
            child_env={
                SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(identity_path),
                SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps(
                    {"test": "aseh-pre-grace-census"},
                    sort_keys=True,
                ),
            },
        ),
        watchdog_hook=lambda *_args: watchdog_calls.append(True),
    )
    snapshots: list[dict[str, object]] = []
    original_write = loop._safe_write_status

    def record_status(*args, **kwargs) -> None:
        original_write(*args, **kwargs)
        snapshots.append(
            json.loads(
                (state_dir / "supervisor_status.json").read_text(
                    encoding="utf-8"
                )
            )
        )

    loop._safe_write_status = record_status  # type: ignore[method-assign]

    result = loop.run()

    live = [
        item
        for item in snapshots
        if item.get("status") in {"starting", "running"}
    ]
    assert result.status == "child_exited"
    assert watchdog_calls == []
    assert {item["status"] for item in live} == {"starting", "running"}
    available_live = [
        item for item in live if item["worker_metrics_available"] is True
    ]
    assert {item["status"] for item in available_live} == {
        "starting",
        "running",
    }
    assert all(
        item["active_worker_count"] is None
        for item in live
        if item["worker_metrics_available"] is False
    )
    assert all(
        item["worker_census_method"]
        == "linux-procfs-descendant-census@1"
        for item in available_live
    )
    assert all(
        item["worker_root_identity_source"] == "supervised_child_identity"
        for item in available_live
    )
    assert all(
        type(item["worker_root_start_time_ticks"]) is int
        for item in available_live
    )
    assert all(bool(item["worker_root_boot_id"]) for item in available_live)
    assert all(
        type(item["active_worker_count"]) is int for item in available_live
    )
    assert all(
        isinstance(item["active_worker_pids"], list)
        for item in available_live
    )
    assert all(item["worker_phase"] == "" for item in available_live)
    assert all(
        item["worker_phase_guarded"] is False for item in available_live
    )
    assert all(item["worker_phase_known"] is True for item in available_live)
    assert all(
        item["worker_phase_known_non_worktree"] is False
        for item in available_live
    )
    assert all(
        item["stalled_without_active_worker"] is None
        for item in available_live
    )
    assert all(
        type(item["worker_observed_at_ns"]) is int
        and item["worker_observed_at_ns"] > 0
        for item in available_live
    )
    assert all(
        item["worker_observation_generation"]
        == (
            f"{item['run_id']}:{item['worker_root_pid']}:"
            f"{item['worker_root_start_time_ticks']}:"
            f"{item['worker_root_boot_id']}"
        )
        for item in available_live
    )


def test_supervisor_loop_publishes_unavailable_census_without_false_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    state_dir = repo / "state"
    state_dir.mkdir(parents=True)
    spec = ManagedDaemonSpec(
        name="aseh-census-test",
        schema="test.aseh-census",
        repo_root=repo,
        daemon_dir=state_dir,
        runner=(sys.executable, "-c", "pass"),
        status_path=state_dir / "daemon_status.json",
        supervisor_status_path=state_dir / "supervisor_status.json",
        supervisor_pid_path=state_dir / "supervisor.pid",
        child_pid_path=state_dir / "child.pid",
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure_status.json",
        ensure_check_path=state_dir / "ensure_check.json",
    )
    loop = SupervisorLoop(
        SupervisorLoopConfig(
            spec=spec,
            command=(sys.executable, "-c", "pass"),
            log_prefix="child",
        )
    )
    birth = current_process_birth()
    child = SimpleNamespace(
        pid=os.getpid(),
        identity_process_birth=birth,
    )

    def unavailable_census(_pid: int) -> list[dict[str, object]]:
        raise OSError("procfs unavailable")

    monkeypatch.setattr(
        supervisor_loop_module,
        "procfs_descendant_processes",
        unavailable_census,
    )

    loop._observe_worker_status(child, {})
    loop._write_status("running", child=child)
    status = json.loads(
        (state_dir / "supervisor_status.json").read_text(encoding="utf-8")
    )

    assert status["worker_metrics_available"] is False
    assert status["worker_metrics_unavailable_reason"] == "worker_census_unavailable"
    assert status["active_worker_count"] is None
    assert status["active_worker_pids"] is None
    assert status["worker_descendant_count"] is None
    assert status["worker_descendant_pids"] is None
    assert status["stalled_without_active_worker"] is None


def test_cleanup_watchdog_cannot_extend_implementation_worker_lease() -> None:
    command = (
        "/usr/bin/python3 -m "
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner "
        "--internal-docker-cleanup-watchdog --container-id provider-1"
    )

    assert todo_supervisor._is_agent_worker_command(command) is False


@pytest.mark.parametrize("mutation", ["reparent", "pid_reuse_after_argv"])
def test_procfs_worker_census_rejects_ancestry_and_pid_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    proc_root = tmp_path / "proc"
    root_pid = 4100
    worker_pid = 4101

    def stat_record(pid: int, parent: int, start_ticks: int) -> str:
        fields = ["0"] * 50
        fields[0] = "S"
        fields[1] = str(parent)
        fields[19] = str(start_ticks)
        return f"{pid} (worker) {' '.join(fields)}\n"

    for pid, parent, start_ticks in (
        (root_pid, 1, 100),
        (worker_pid, root_pid, 101),
    ):
        process_dir = proc_root / str(pid)
        process_dir.mkdir(parents=True)
        (process_dir / "stat").write_text(
            stat_record(pid, parent, start_ticks),
            encoding="utf-8",
        )
        (process_dir / "cmdline").write_bytes(
            b"grok\x00--workspace\x00/tmp/task\x00"
        )

    original = todo_supervisor._strict_procfs_process_identity
    worker_reads = 0

    def raced_identity(path: Path):
        nonlocal worker_reads
        observed = original(path)
        if path == proc_root / str(worker_pid) / "stat":
            worker_reads += 1
            if mutation == "reparent" and worker_reads == 2:
                return (1, 101, "S")
            if mutation == "pid_reuse_after_argv" and worker_reads == 3:
                return (root_pid, 202, "S")
        return observed

    monkeypatch.setattr(
        todo_supervisor,
        "_strict_procfs_process_identity",
        raced_identity,
    )

    with pytest.raises(OSError, match="ancestry or identity changed"):
        todo_supervisor.procfs_descendant_processes(
            root_pid,
            proc_root=proc_root,
        )


def test_aseh_health_zero_frontier_dependency_deadlock_is_blocked_and_stuck(
    tmp_path: Path,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        ready=False,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        ready=False,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now,
        failure={},
    )

    assert receipt["dependency_deadlock"] is True
    assert receipt["blocked"] is True
    assert receipt["stuck"] is True
    assert receipt["healthy"] is False


def test_aseh_health_gives_exact_blocked_reconciliation_a_bounded_window(
    tmp_path: Path,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        status="blocked",
        revision=2,
        event_cursor=11,
        ready=False,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        status="blocked",
        revision=2,
        event_cursor=11,
        ready=False,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    before["authority"]["blocked_count"] = 1  # type: ignore[index]
    current["authority"]["blocked_count"] = 1  # type: ignore[index]

    receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now - 0.25,
        failure={},
    )

    assert receipt["blocked"] is True
    assert receipt["stuck"] is True
    assert receipt["healthy"] is False
    assert receipt["blocked_recovery_admitted"] is True
    edges = 0
    for _index in range(2):
        action, reason, edges = (
            aseh_operator._post_admission_health_action(
                receipt,
                prior_available=True,
                current_available=True,
                unhealthy_edges=edges,
            )
        )
        assert (action, reason) == ("continue", "")
    assert aseh_operator._post_admission_health_action(
        receipt,
        prior_available=True,
        current_available=True,
        unhealthy_edges=edges,
    ) == (
        "fail",
        "authoritative_blocked_recovery_grace_exhausted",
        3,
    )

    current["authority"]["objective_record"]["title"] = (  # type: ignore[index]
        "unsealed mutation"
    )
    rejected = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now - 0.25,
        failure={},
    )
    assert rejected["blocked_recovery_admitted"] is False
    assert aseh_operator._post_admission_health_action(
        rejected,
        prior_available=True,
        current_available=True,
        unhealthy_edges=0,
    )[:2] == ("fail", "authoritative_board_blocked")


def test_aseh_health_blocked_reconciliation_allows_startup_lane_refresh(
    tmp_path: Path,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        status="blocked",
        revision=2,
        event_cursor=11,
        ready=False,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 100.0) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        status="blocked",
        revision=2,
        event_cursor=11,
        ready=False,
        observed_at=now,
        lane_mtime_ns=int((now - 100.0) * 1_000_000_000),
    )
    for sample in (before, current):
        sample["authority"]["blocked_count"] = 1  # type: ignore[index]
        sample["lanes"][0]["fresh"] = False  # type: ignore[index]
        sample["lanes"][0]["watchdog_admissible"] = False  # type: ignore[index]

    startup = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now - 0.25,
        failure={},
    )
    assert startup["startup_grace_active"] is True
    assert startup["lane_heartbeat_fresh"] is False
    assert startup["blocked_recovery_admitted"] is True

    after_startup = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 2.0,
        last_progress_at=now - 0.25,
        failure={},
    )
    assert after_startup["startup_grace_active"] is False
    assert after_startup["blocked_recovery_admitted"] is False


def test_aseh_health_admits_exact_delayed_retry_frontier(
    tmp_path: Path,
) -> None:
    now = time.time()
    retry_at_ms = int((now + 30.0) * 1_000)
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        ready=False,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
        delayed_retry_not_before_ms=retry_at_ms,
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        ready=False,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
        delayed_retry_not_before_ms=retry_at_ms,
    )
    receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 100.0,
        last_progress_at=now - 100.0,
        failure={},
    )

    assert receipt["delayed_frontier_admitted"] is True
    assert receipt["delayed_ready_task_ids"] == ["ASEH-000"]
    assert receipt["dependency_deadlock"] is False
    assert receipt["blocked"] is False
    assert receipt["stuck"] is False
    assert receipt["healthy"] is True


def test_aseh_health_rejects_outage_progress_and_bounds_recovery_edges(
    tmp_path: Path,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        status="claimed",
        revision=2,
        event_cursor=11,
        ready=False,
        active=True,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    unavailable = json.loads(json.dumps(before))
    unavailable["authority"] = {
        "available": False,
        "task_statuses": {},
        "task_revisions": {},
        "event_cursor": 0,
    }
    assert aseh_operator._authoritative_progress_between(
        unavailable, current
    ) == []
    assert aseh_operator._authoritative_progress_between(
        before, unavailable
    ) == []

    rollback = json.loads(json.dumps(current))
    rollback["authority"]["task_revisions"]["ASEH-000"] = 0
    rollback["authority"]["event_cursor"] = 12
    rollback["authority"]["snapshot"]["event_cursor"] = 12
    rejected = aseh_operator._health_receipt(
        board,
        paths,
        samples=(current, rollback),
        launched_at=now - 1.0,
        last_progress_at=now,
        failure={},
    )
    assert rejected["task_authority_pair"]["revision_monotonic"] is False
    assert rejected["healthy"] is False

    unhealthy = {
        "healthy": False,
        "blocked": False,
        "stuck": False,
        "scheduler_alive": True,
        "owner_ready": True,
        "broker_ready": True,
    }
    edges = 0
    for prior_available, current_available in (
        (True, False),
        (False, True),
    ):
        action, reason, edges = aseh_operator._post_admission_health_action(
            unhealthy,
            prior_available=prior_available,
            current_available=current_available,
            unhealthy_edges=edges,
        )
        assert (action, reason) == ("continue", "")
    action, reason, edges = aseh_operator._post_admission_health_action(
        unhealthy,
        prior_available=True,
        current_available=False,
        unhealthy_edges=edges,
    )
    assert action == "fail"
    assert reason == "authoritative_status_recovery_grace_exhausted"
    assert edges == 3


def test_aseh_post_admission_grace_is_exclusive_to_typed_lane_loss() -> None:
    lane_only = {
        "healthy": False,
        "blocked": False,
        "stuck": False,
        "terminal": False,
        "scheduler_alive": True,
        "owner_ready": True,
        "broker_ready": True,
        "health_without_lane_admitted": True,
        "lane_heartbeat_fresh": False,
    }

    edges = 0
    for _index in range(2):
        action, reason, edges = aseh_operator._post_admission_health_action(
            lane_only,
            prior_available=True,
            current_available=True,
            unhealthy_edges=edges,
        )
        assert (action, reason) == ("continue", "")
    action, reason, edges = aseh_operator._post_admission_health_action(
        lane_only,
        prior_available=True,
        current_available=True,
        unhealthy_edges=edges,
    )
    assert (action, reason, edges) == (
        "fail",
        "authoritative_health_admission_lost",
        3,
    )

    healthy = {**lane_only, "healthy": True, "lane_heartbeat_fresh": True}
    assert aseh_operator._post_admission_health_action(
        healthy,
        prior_available=True,
        current_available=True,
        unhealthy_edges=2,
    ) == ("continue", "", 0)

    for field, value in (
        ("scheduler_alive", False),
        ("owner_ready", False),
        ("broker_ready", False),
        ("health_without_lane_admitted", False),
        ("lane_heartbeat_fresh", None),
    ):
        degraded = {**lane_only, field: value}
        action, _reason, next_edges = (
            aseh_operator._post_admission_health_action(
                degraded,
                prior_available=True,
                current_available=True,
                unhealthy_edges=0,
            )
        )
        assert action == "fail", field
        assert next_edges == 0

    scheduler_lost = {**lane_only, "scheduler_alive": False}
    assert aseh_operator._post_admission_health_action(
        scheduler_lost,
        prior_available=True,
        current_available=False,
        unhealthy_edges=0,
    )[:2] == ("fail", "authoritative_scheduler_not_live")


def test_aseh_startup_fails_after_two_unavailable_authority_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    board, fixture_paths, current = _aseh_health_fixture(
        tmp_path,
        status="claimed",
        revision=2,
        event_cursor=11,
        ready=False,
        active=True,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
    )
    paths = {
        **fixture_paths,
        "status_receipt": tmp_path / "live-status.json",
    }
    unavailable = json.loads(json.dumps(current))
    unavailable["authority"] = {
        "available": False,
        "error": "published replica file identity is unsafe",
        "error_type": "OperatorError",
        "ready_count": 0,
        "active_count": 0,
        "blocked_count": 0,
        "terminal_count": 0,
        "event_cursor": 0,
        "task_statuses": {},
        "task_revisions": {},
    }
    samples = [unavailable, json.loads(json.dumps(unavailable))]
    recorded_failure: dict[str, object] = {}

    def fake_sample(*_args: object, **_kwargs: object) -> dict[str, object]:
        assert samples, "startup admission sampled past the stable pair"
        return samples.pop(0)

    monkeypatch.setattr(aseh_operator, "_status_sample", fake_sample)
    monkeypatch.setattr(aseh_operator, "STATUS_SAMPLE_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(
        aseh_operator,
        "_record_control_failure",
        lambda _paths, _failure, _event, **fields: (
            recorded_failure.update(fields)
        ),
    )

    with pytest.raises(
        aseh_operator.OperatorError,
        match="two consecutive authoritative status samples unavailable",
    ):
        aseh_operator._await_initial_health(
            board,
            paths,
            server=SimpleNamespace(),
            scheduler=SimpleNamespace(pid=4242, poll=lambda: None),
            launched_at=now - 0.5,
            failure={},
            failure_event=threading.Event(),
            shutdown_requested=threading.Event(),
            received_signal={},
        )

    assert recorded_failure == {
        "reason_code": "authoritative_status_unavailable_two_samples",
        "error_type": "ASEHHealthQueryFailure",
    }
    assert samples == []


def test_aseh_health_accepts_fast_claim_before_first_sample(
    tmp_path: Path,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        status="claimed",
        revision=2,
        event_cursor=11,
        ready=False,
        active=True,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        status="claimed",
        revision=2,
        event_cursor=11,
        ready=False,
        active=True,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now - 0.5,
        failure={},
        require_authoritative_progress=True,
    )

    assert "authoritative_event_since_bootstrap" in receipt["progress_evidence"]
    assert "authoritative_status_since_bootstrap" in receipt["progress_evidence"]
    assert "authoritative_revision_since_bootstrap" in receipt["progress_evidence"]
    assert receipt["authoritative_progress_admitted"] is True
    assert receipt["healthy"] is True


def test_aseh_stale_bootstrap_is_rejected_before_owner_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    _board, fixture_paths, _sample = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    database = tmp_path / "control.duckdb"
    database.touch()
    paths = {**fixture_paths, "database": database}
    board = SimpleNamespace(config_path=tmp_path / "board.json")
    config: dict[str, object] = {}
    current_population = {
        "source_head": "commit:current",
        "repository_tree_id": "tree:current",
        "plan_root_cid": "plan:current",
        "source_forest": {"forest_cid": "forest:current"},
        "source_identities": {"operator": "source:current"},
    }
    owner_built = False

    def build_owner(_board: object, _paths: object) -> object:
        nonlocal owner_built
        owner_built = True
        return object()

    monkeypatch.setattr(aseh_operator, "_load", lambda _path: (board, config))
    monkeypatch.setattr(aseh_operator, "_paths", lambda _board: paths)
    monkeypatch.setattr(
        aseh_operator,
        "_population",
        lambda _board, _config: current_population,
    )
    monkeypatch.setattr(aseh_operator, "_build_server", build_owner)
    monkeypatch.setattr(
        configured_scheduler,
        "preflight_configured_board",
        lambda _board: {"valid": True},
    )

    with pytest.raises(aseh_operator.OperatorError, match="source forest"):
        aseh_operator.run_supervisor(board.config_path, implement=True, duration=1)
    assert owner_built is False


def test_aseh_canonical_merge_suffix_admits_only_exact_two_parent_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        checkout_repository_id,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeRequest

    def git(*args: str) -> str:
        result = subprocess.run(
            ("git", *args), cwd=tmp_path, text=True, capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    git("init", "-b", "main")
    git("config", "user.email", "aseh-continuity@example.invalid")
    git("config", "user.name", "ASEH Continuity")
    (tmp_path / "base.txt").write_text("base\n", encoding="utf-8")
    git("add", "base.txt")
    git("commit", "-m", "sealed base")
    base = git("rev-parse", "HEAD")
    git("checkout", "-b", "candidate")
    (tmp_path / "output.txt").write_text("admitted\n", encoding="utf-8")
    git("add", "output.txt")
    git("commit", "-m", "ASEH-000 exact output")
    candidate = git("rev-parse", "HEAD")
    candidate_tree = git("rev-parse", "HEAD^{tree}")
    git("checkout", "main")
    git("merge", "--no-ff", "--no-edit", "candidate")
    integrated = git("rev-parse", "HEAD")

    monkeypatch.setattr(aseh_operator, "ROOT", tmp_path)
    task_cid = "task:aseh-continuity"
    board = SimpleNamespace(
        protected_paths=("protected.py",),
        merge_target_branch="main",
    )
    bootstrap = {
        "integrity": {"task_revisions": {"ASEH-000": 1}}
    }
    integrity = {
        "task_statuses": {"ASEH-000": "completed"},
        "task_revisions": {"ASEH-000": 3},
        "task_cids": {"ASEH-000": task_cid},
    }
    metadata = {
        "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
        "target_binding_schema": (
            "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
        ),
        "target_repository_id": checkout_repository_id(tmp_path),
        "target_branch": "main",
        "candidate_tree": candidate_tree,
        "repository_tree_id": f"git-tree:{candidate_tree}",
        "baseline_ref": base,
        "changed_submodule_paths": [],
        "completion_task_cids": {"ASEH-000": task_cid},
        "task": {"outputs": ["output.txt"]},
        "validation_proof": {
            "passed": True,
            "target_commit": candidate,
            "target_tree": candidate_tree,
        },
    }
    request = MergeRequest(
        request_id="request:aseh-continuity",
        branch_name="candidate",
        task_id="ASEH-000",
        priority="P1",
        lane_id="lane-0",
        enqueued_at=1.0,
        metadata=metadata,
        commit_sha=candidate,
        canonical_task_id=task_cid,
        canonical_task_key=task_cid,
        status="completed",
    )
    proof = aseh_operator._admit_canonical_merge_suffix(
        board,
        base_head=base,
        target_head=integrated,
        bootstrap=bootstrap,
        integrity=integrity,
        task_outputs={"ASEH-000": ("output.txt",)},
        completed_requests=(request,),
    )
    assert proof["integrations"][0]["candidate_commit"] == candidate
    assert proof["integrations"][0]["changed_paths"] == ["output.txt"]

    monkeypatch.setattr(
        aseh_operator, "REPAIR_FOLLOWUP_TRANSITION_FIRST_PARENT", base
    )
    monkeypatch.setattr(
        aseh_operator, "REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD", integrated
    )
    monkeypatch.setattr(
        aseh_operator, "REPAIR_FOLLOWUP_TRANSITION_CANDIDATE", candidate
    )
    monkeypatch.setattr(
        aseh_operator, "REPAIR_FOLLOWUP_TRANSITION_TASK_ALIAS", "ASEH-000"
    )
    integrity["task_statuses"]["ASEH-000"] = "blocked"
    nonterminal = aseh_operator._admit_canonical_merge_suffix(
        board,
        base_head=base,
        target_head=integrated,
        bootstrap=bootstrap,
        integrity=integrity,
        task_outputs={"ASEH-000": ("output.txt",)},
        completed_requests=(request,),
        admission_mode="followup_repair_base",
    )
    assert nonterminal["schema"].endswith(
        "/aseh-nonterminal-integration-base@1"
    )
    assert nonterminal["task_completion_admitted"] is False
    assert nonterminal["completion_authoritative"] is False
    assert nonterminal["integrations"][0]["observed_task_status"] == "blocked"

    for restart_status in ("ready", "claimed", "in_progress", "running"):
        integrity["task_statuses"]["ASEH-000"] = restart_status
        restart_witness = aseh_operator._admit_canonical_merge_suffix(
            board,
            base_head=base,
            target_head=integrated,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs={"ASEH-000": ("output.txt",)},
            completed_requests=(request,),
            admission_mode="followup_repair_base",
        )
        assert restart_witness["task_completion_admitted"] is False
        assert restart_witness["integrations"][0][
            "observed_task_status"
        ] == restart_status
    integrity["task_statuses"]["ASEH-000"] = "completed"

    metadata["baseline_ref"] = "HEAD"
    with pytest.raises(aseh_operator.OperatorError, match="exact commit"):
        aseh_operator._admit_canonical_merge_suffix(
            board,
            base_head=base,
            target_head=integrated,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs={"ASEH-000": ("output.txt",)},
            completed_requests=(request,),
        )
    metadata["baseline_ref"] = base

    git("checkout", "-b", "omitted-candidate-output", base)
    git("merge", "--no-ff", "-s", "ours", "--no-edit", "candidate")
    omitted = git("rev-parse", "HEAD")
    with pytest.raises(aseh_operator.OperatorError, match="output differs"):
        aseh_operator._admit_canonical_merge_suffix(
            board,
            base_head=base,
            target_head=omitted,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs={"ASEH-000": ("output.txt",)},
            completed_requests=(request,),
        )
    git("checkout", "main")

    metadata["candidate_tree"] = "0" * 40
    with pytest.raises(aseh_operator.OperatorError, match="validation binding"):
        aseh_operator._admit_canonical_merge_suffix(
            board,
            base_head=base,
            target_head=integrated,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs={"ASEH-000": ("output.txt",)},
            completed_requests=(request,),
        )
    metadata["candidate_tree"] = candidate_tree

    protected_board = SimpleNamespace(
        protected_paths=("output.txt",),
        merge_target_branch="main",
    )
    with pytest.raises(aseh_operator.OperatorError, match="protected-path"):
        aseh_operator._admit_canonical_merge_suffix(
            protected_board,
            base_head=base,
            target_head=integrated,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs={"ASEH-000": ("output.txt",)},
            completed_requests=(request,),
        )

    (tmp_path / "arbitrary.txt").write_text("escape\n", encoding="utf-8")
    git("add", "arbitrary.txt")
    git("commit", "-m", "arbitrary child")
    arbitrary = git("rev-parse", "HEAD")
    with pytest.raises(aseh_operator.OperatorError, match="non-canonical"):
        aseh_operator._admit_canonical_merge_suffix(
            board,
            base_head=base,
            target_head=arbitrary,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs={"ASEH-000": ("output.txt",)},
            completed_requests=(request,),
        )


def test_aseh_repair_transition_receipt_is_closed_and_non_mutating() -> None:
    receipt = {
        "schema": aseh_operator.REPAIR_TRANSITION_SCHEMA,
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": "agent-supervisor-efficiency/ASEH-BOOTSTRAP-002",
        "program_id": aseh_operator.PROGRAM,
        "bootstrap_receipt_id": "sha256:" + ("1" * 64),
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": aseh_operator.REPAIR_TRANSITION_BASE_HEAD,
        "base_tree": "2" * 40,
        "repair_head": "3" * 40,
        "repair_tree": "4" * 40,
        "changed_paths": list(aseh_operator.REPAIR_TRANSITION_CHANGED_PATHS),
        "patch_digest": "sha256:" + ("5" * 64),
        "dependencies": ["ASEH-BOOTSTRAP-001", "ASEH-000"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": "explicit bootstrap repair authority",
        "validation_results": [],
        "terminal_success_criteria": "exact repair",
        "terminal_non_success_criteria": "all drift rejected",
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    assert aseh_operator._repair_transition_receipt_id(receipt) == (
        receipt["receipt_cid"]
    )
    receipt["database_mutated"] = True
    with pytest.raises(aseh_operator.OperatorError, match="schema"):
        aseh_operator._repair_transition_receipt_id(receipt)


def test_aseh_repair_transition_followup_receipt_is_closed_and_chained() -> None:
    witness = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "aseh-nonterminal-integration-witness@1"
        ),
        "base_head": "1" * 40,
        "base_tree": "2" * 40,
        "target_head": "3" * 40,
        "target_tree": "4" * 40,
        "request_id": "request:aseh-followup",
        "merge_request_cid": "sha256:" + ("5" * 64),
        "task_id": "ASEH-001",
        "task_cid": "task:aseh-followup",
        "candidate_commit": "6" * 40,
        "candidate_tree": "7" * 40,
        "integration_commit": "3" * 40,
        "integration_tree": "4" * 40,
        "baseline_ref": "8" * 40,
        "changed_paths": ["sealed.json"],
        "validation_proof_cid": "sha256:" + ("9" * 64),
        "completion_authoritative": False,
        "task_completion_admitted": False,
    }
    witness["receipt_cid"] = aseh_operator._identity(witness)
    receipt = {
        "schema": aseh_operator.REPAIR_FOLLOWUP_TRANSITION_SCHEMA,
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R2"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 2,
        "bootstrap_receipt_id": "sha256:" + ("a" * 64),
        "previous_receipt_cid": "sha256:" + ("b" * 64),
        "base_integration_witness": witness,
        "authorization_task_observation": {
            "task_id": "ASEH-001",
            "task_cid": "task:aseh-followup",
            "status": "blocked",
            "revision": 6,
            "completion_authoritative": False,
            "observed_at": 1.0,
        },
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": "3" * 40,
        "base_tree": "4" * 40,
        "repair_head": "c" * 40,
        "repair_tree": "d" * 40,
        "changed_paths": list(
            aseh_operator.REPAIR_FOLLOWUP_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": "sha256:" + ("e" * 64),
        "dependencies": [
            "ASEH-BOOTSTRAP-002@ASEH-PLAN-R1",
            "ASEH-001",
        ],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": "explicit follow-up repair authority",
        "validation_results": [],
        "terminal_success_criteria": "exact automatic recovery",
        "terminal_non_success_criteria": "all drift rejected",
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    assert aseh_operator._repair_followup_transition_receipt_id(receipt) == (
        receipt["receipt_cid"]
    )

    receipt["transition_revision"] = 3
    receipt["receipt_cid"] = aseh_operator._identity(
        {key: value for key, value in receipt.items() if key != "receipt_cid"}
    )
    with pytest.raises(aseh_operator.OperatorError, match="schema"):
        aseh_operator._repair_followup_transition_receipt_id(receipt)


def test_aseh_repair_authorization_replay_rejects_head_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap_path = tmp_path / "bootstrap.json"
    repair_path = tmp_path / "repair.json"
    bootstrap_path.touch()
    repair_path.touch()
    board = object()
    config: dict[str, object] = {}
    paths = {
        "bootstrap_receipt": bootstrap_path,
        "repair_transition_receipt": repair_path,
    }
    bootstrap = {"bootstrap_receipt_id": "bootstrap:sealed"}
    prior = {"repair_head": "a" * 40}
    launch_called = False

    monkeypatch.setattr(aseh_operator, "_load", lambda _path: (board, config))
    monkeypatch.setattr(aseh_operator, "_paths", lambda _board: paths)
    monkeypatch.setattr(
        aseh_operator,
        "_population",
        lambda _board, _config: {"source_head": "b" * 40},
    )
    monkeypatch.setattr(
        aseh_operator,
        "_secure_runtime_json",
        lambda path, **_kwargs: bootstrap if path == bootstrap_path else prior,
    )
    monkeypatch.setattr(
        aseh_operator, "_bootstrap_receipt_id", lambda _payload: "bootstrap:sealed"
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_transition",
        lambda *_args, **_kwargs: {},
    )

    def reject_regression(*args: str, **_kwargs: object) -> str:
        assert args[:2] == ("merge-base", "--is-ancestor")
        raise aseh_operator.OperatorError("repair is not an ancestor")

    def launch(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal launch_called
        launch_called = True
        return {}

    monkeypatch.setattr(aseh_operator, "_git", reject_regression)
    monkeypatch.setattr(aseh_operator, "_admit_materialized_launch", launch)

    with pytest.raises(aseh_operator.OperatorError, match="not an ancestor"):
        aseh_operator.authorize_repair_transition(tmp_path / "board.json")
    assert launch_called is False


def test_aseh_scheduler_uses_only_complete_live_owner_generation_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = DatabaseProgramConfig(
        authority_mode="quack",
        task_source_kind="duckdb",
        endpoint_secret_handle="handle:aseh-live-test",
        quack_endpoint="quack:127.0.0.1:41487",
        store_id="store:aseh-live-test",
        store_generation="1",
        schema_revision="1",
        runtime_registry_path="registry",
        failover_policy="fail_closed",
    )
    board = SimpleNamespace(
        database_program=program,
        resolved_database_program=lambda: program,
        path=lambda value: tmp_path / Path(value),
    )
    registry = tmp_path / "registry"
    registry.mkdir()
    descriptor = os.open("/dev/null", os.O_RDONLY)
    birth = current_process_birth()
    status_identity = {
        "store_id": program.store_id,
        "generation": 8,
        "schema_revision": 3,
        "process_birth_id": "birth:aseh-live-test",
        "process_birth": birth.to_dict(),
    }
    monkeypatch.setattr(
        configured_scheduler,
        "_read_stable_regular_json",
        lambda _path: (
            {"lifecycle": "ready", "identity": status_identity},
            {"state": "present"},
        ),
    )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION", "9"
    )
    with pytest.raises(configured_scheduler.ConfiguredBoardError, match="incomplete"):
        configured_scheduler._database_program_with_admitted_live_owner(board)

    inherited = program.to_dict()
    inherited["store_generation"] = "9"
    inherited["schema_revision"] = "3"
    environment = {
        "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION": "9",
        "IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION": "3",
        "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION": "9",
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION": "3",
        "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON": json.dumps(inherited),
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET": str(
            registry / "typed-state-owner-grants.sock"
        ),
        "IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET": str(
            registry / "typed-state-owner.sock"
        ),
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD": str(descriptor),
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    try:
        with pytest.raises(
            configured_scheduler.ConfiguredBoardError,
            match="exact owner status",
        ):
            configured_scheduler._database_program_with_admitted_live_owner(board)

        status_identity["generation"] = 9
        admitted = configured_scheduler._database_program_with_admitted_live_owner(
            board
        )
        assert admitted.store_generation == "9"
        assert admitted.schema_revision == "3"
        assert "--state-store-generation" in admitted.cli_args()
        assert admitted.cli_args()[
            admitted.cli_args().index("--state-store-generation") + 1
        ] == "9"
    finally:
        os.close(descriptor)


def test_aseh_stop_signal_handlers_request_cleanup_and_restore() -> None:
    requested = threading.Event()
    received: dict[str, int] = {}
    prior = signal.getsignal(signal.SIGTERM)

    with aseh_operator._stop_signal_handlers(requested, received):
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        assert requested.is_set()
        assert received == {"signum": signal.SIGTERM}

    assert signal.getsignal(signal.SIGTERM) == prior
