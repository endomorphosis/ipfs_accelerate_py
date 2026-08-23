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
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
    provider_subprocess_environment,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    configured_board_scheduler as configured_scheduler,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    current_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    QuackStateServerControlError,
    QuackStateServerMutationError,
    QuackStateServerReadyError,
    TypedStateOwnerGrantBroker,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
    TaskSourceBoundsError,
    TaskSourceUnknownOutcomeError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS,
    DuckDBConnectionPolicyError,
    reset_quack_transport_cache,
    submit_quack_owner_command,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    QuackCapabilityStatus,
    probe_quack_capabilities,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    DATABASE_TASK_COMMANDS,
    TYPED_STATE_OWNER_GRANT_BROKER_SCHEMA,
    TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME,
    TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
    TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
    TYPED_STATE_OWNER_SOCKET_ENV,
    TypedStateOwnerAuthorizationError,
    TypedStateOwnerConnection,
    TypedStateOwnerError,
    TypedStateOwnerRemoteError,
    kernel_process_birth_id,
    request_database_task_command_credential,
    request_quack_attach_credential,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    typed_state_owner as typed_state_owner_module,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    build_validation_environment,
)
from scripts import run_agent_supervisor_efficiency_state_hardening as aseh_operator


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
) -> None:
    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip(f"reviewed preinstalled Quack unavailable: {capability.status.value}")

    database = tmp_path / "control.duckdb"
    owner_dir = tmp_path / "quack-owner"
    broker_socket = owner_dir / TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME
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
            # A successful owner acknowledgement is also a synchronous
            # Quack-publication barrier.  Strict sharding re-reads this exact
            # binding immediately after claim and must not see the startup
            # replica's stale ``todo`` projection.
            observed = source.get("ASEH-000")
            assert observed is not None
            assert observed.status == "in_progress"
            assert observed.revision == changed.revision
            assert observed.body["completion_receipt"] == {
                "operation": "database_claim",
                "claim_id": "claim:aseh-bootstrap-test",
                "attempt_id": "attempt:aseh-bootstrap-test",
                "owner_session_id": "owner:aseh-bootstrap-test",
                "lease_id": "lease:aseh-bootstrap-test",
                "fencing_token": 1,
                "fence_epoch": 1,
                "claimed_from_revision": 1,
            }
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
        assert idempotency is not None and int(idempotency[0]) == 1

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
    _materialize_one_task(database)
    server = build_server(
        database_path=database,
        state_dir=owner_dir,
        repository_root=tmp_path,
        port=0,
        store_id=str(database),
        secret_handle="handle:aseh-publication-failure-test",
    )
    identity = server.start()
    client: threading.Thread | None = None
    outcome: dict[str, BaseException] = {}
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

        def submit() -> None:
            try:
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
            except BaseException as exc:  # noqa: BLE001 - assert exact boundary
                outcome["error"] = exc

        client = threading.Thread(target=submit, daemon=True)
        client.start()
        request_deadline = time.monotonic() + 2.0
        while (
            not tuple((owner_dir / "mutations").glob("*.request.json"))
            and time.monotonic() < request_deadline
        ):
            time.sleep(0.01)
        requests = tuple((owner_dir / "mutations").glob("*.request.json"))
        assert len(requests) == 1

        with pytest.raises(
            QuackStateServerMutationError,
            match="read_replica_refresh_unknown_outcome",
        ):
            server.service_database_task_command_inbox(
                expected_store_generation=str(identity.generation),
                max_requests=1,
            )
        client.join(timeout=2.0)
        assert not client.is_alive()
        assert isinstance(outcome.get("error"), DuckDBConnectionPolicyError)
        assert "unknown outcome" in str(outcome["error"])
        assert requests[0].is_file()
        assert not tuple((owner_dir / "mutations").glob("*.done.json"))
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
    finally:
        if client is not None:
            client.join(timeout=2.0)
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
    _materialize_one_task(database)
    server = build_server(
        database_path=database,
        state_dir=owner_dir,
        repository_root=tmp_path,
        port=0,
        store_id=str(database),
        secret_handle="handle:aseh-broker-denial-test",
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
    bootstrap_snapshot = {
        "source_schema": "source@1",
        "schema_version": "1",
        "plan_root_cid": "plan:aseh-health",
        "repository_tree_id": "tree:aseh-health",
        "formal_plan_id": "formal:aseh-health",
        "source_identity": "source:aseh-health",
        "projection_cid": "projection:aseh-health",
        "event_cursor": 10,
        "task_count": 1,
        "goal_count": 1,
        "dependency_count": 0,
        "objective_count": 1,
        "plan_count": 1,
    }
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

    binding = {
        "server_id": "server:aseh-health",
        "store_id": "store:aseh-health",
        "database_uuid": "database:aseh-health",
        "schema_revision": 3,
        "schema_fingerprint": "schema:aseh-health",
        "generation": 9,
        "process_birth_id": "birth:aseh-health",
        "listen_uri": "quack:127.0.0.1:1",
        "extension_fingerprint": "extensions:aseh-health",
    }
    owner_identity = {
        **binding,
        "process_birth": {"pid": os.getpid()},
    }
    current_snapshot = {**bootstrap_snapshot, "event_cursor": event_cursor}
    ready_task_ids = ["ASEH-000"] if ready else []
    authority = {
        "available": True,
        "transport": "quack",
        "credential_path": "sealed_memfd_broker",
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
            "identity": owner_identity,
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
                "active_worker_count": int(active),
                "worker_phase_age_seconds": 0.5,
                "stalled_without_active_worker": lane_stalled,
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
    return board, {"bootstrap_receipt": bootstrap_path}, sample


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


def test_aseh_lane_status_projects_worker_watchdog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aseh_operator, "ROOT", tmp_path)
    state_root = tmp_path / "state"
    status_path = state_root / "lane-0" / "aseh_lane_0_supervisor_status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "todo_implementation_supervisor.supervisor"
                ),
                "repo_root": str(tmp_path),
                "task_prefix": "## ASEH-",
                "state_prefix": "aseh_lane_0",
                "status": "running",
                "active_worker_count": 0,
                "worker_phase_age_seconds": 42.5,
                "stalled_without_active_worker": True,
            }
        ),
        encoding="utf-8",
    )
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
    assert observations[0]["worker_phase_age_seconds"] == 42.5
    assert observations[0]["stalled_without_active_worker"] is True


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

    unhealthy = {"healthy": False, "blocked": False, "stuck": False}
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
    assert reason == "broker_health_recovery_grace_exhausted"
    assert edges == 3


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
