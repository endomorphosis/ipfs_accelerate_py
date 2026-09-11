"""Native dispatch pauses must preserve admitted obligations and task authority."""

from __future__ import annotations

import pytest

from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
    _population,
)


class Boundary:
    def __init__(self, permitted=False):
        self.permitted = permitted
        self.calls = 0
        self.retained = []

    def before_claim(self):
        self.calls += 1
        return {
            "new_dispatch_permitted": self.permitted,
            "drain_epoch": 1,
            "reason": "native_dispatch_pause_requested",
        }

    def reconciliation_started(self):
        pass

    def retained_work(self, attempt):
        self.retained.append(attempt.attempt_id)


@pytest.mark.parametrize("public_claim", [False, True])
def test_native_pause_prevents_new_canonical_claim_and_provider(tmp_path, public_claim):
    calls = []
    daemon = _open_daemon(tmp_path, session="session:drain", provider_calls=calls)
    try:
        daemon.materialize_population(_population(1))
        before = daemon.task_source.get("task:cid:001")
        boundary = Boundary()
        daemon._native_dispatch_control = boundary
        if public_claim:
            assert daemon.claim_next() is None
        else:
            result = daemon.run_once()
            assert result["selection_idle_reason"] == "native_dispatch_pause_requested"
        after = daemon.task_source.get("task:cid:001")
        assert (after.status, after.revision) == (before.status, before.revision)
        assert calls == []
        assert boundary.calls == 1
    finally:
        daemon.close()


def test_native_pause_retains_existing_claim_and_resumes_its_obligation(tmp_path):
    calls = []
    daemon = _open_daemon(tmp_path, session="session:retained", provider_calls=calls)
    try:
        daemon.materialize_population(_population(2))
        attempt = daemon.claim_next()
        assert attempt is not None
        boundary = Boundary()
        daemon._native_dispatch_control = boundary
        daemon.run_once()
        assert calls == [attempt.task_cid]
        assert boundary.retained == [attempt.attempt_id]
        assert boundary.calls == 0
        untouched = daemon.task_source.get("task:cid:002")
        assert untouched.status == "ready"
    finally:
        daemon.close()


# Disposable native peer/custody qualification: a real DuckDB writer owns both
# native FLOCKs, and a separate master->supervisor->daemon tree uses real peers.
import fcntl
import json
import multiprocessing
import os
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from ipfs_accelerate_py.agent_supervisor.runtime import native_dispatch_drain as drain
from ipfs_accelerate_py.agent_supervisor.runtime import (
    owner_status_observation as observation,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    DuckDBConnection,
)

TOKEN = "disposable-test-owner-token-not-a-production-credential"


def _configuration():
    return {
        "board_namespace": drain.PROGRAM,
        "max_lanes": 1,
        "runtime_paths": {"state": "state"},
        "database_program": {"store_generation": 1},
        "quack_owner": {
            "database_path": "control.duckdb",
            "state_dir": ".",
            "store_id": "control.duckdb",
            "repository_id": "repository:test",
            "secret_handle": "env://TEST_DRAIN_TOKEN",
        },
    }


def _peer_owner(directory, pipe):
    import duckdb

    root = Path(directory)
    descriptors = []
    native = None
    listener = None
    try:
        for path in observation._locks(root / "control.duckdb"):
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            descriptors.append(fd)
        native = duckdb.connect(str(root / "control.duckdb"), config={"threads": 1})
        native.execute("CREATE TABLE retained_work (value BIGINT)")
        native.execute("INSERT INTO retained_work VALUES (17)")
        identity = {
            "server_id": "server:test",
            "store_id": "control.duckdb",
            "database_uuid": "database:test",
            "generation": 1,
            "fence_epoch": 1,
            "process_birth_id": "birth:test",
            "repository_id": "repository:test",
        }
        server = SimpleNamespace(
            config=SimpleNamespace(
                database_path=root / "control.duckdb", state_dir=root
            ),
            lifecycle=SimpleNamespace(value="ready"),
            identity=SimpleNamespace(to_dict=lambda: dict(identity)),
            secret_handle="env://TEST_DRAIN_TOKEN",
            _vault=SimpleNamespace(resolve=lambda _: TOKEN),
            _connection=DuckDBConnection.wrap(native),
            _lock=threading.RLock(),
        )

        def open_listener():
            return observation.OwnerStatusObservation(
                server,
                program_id=drain.PROGRAM,
                configuration=_configuration(),
                source_head="a" * 40,
                source_tree="b" * 40,
                task_registry={"task:one": "TEST-001"},
            )

        listener = open_listener()
        pipe.send({"ready": True})
        while True:
            if pipe.poll():
                command = pipe.recv()
                if command == "stop":
                    break
                if command == "restart_listener":
                    listener.close()
                    listener = open_listener()
                elif command == "release_lock":
                    fcntl.flock(descriptors[0], fcntl.LOCK_UN)
                elif command == "state":
                    pipe.send(
                        native.execute("SELECT value FROM retained_work").fetchall()
                    )
                    continue
                pipe.send({"ready": True})
            listener.poll()
            time.sleep(0.002)
    except BaseException as exc:  # noqa: BLE001 - isolated fixture reports only error type
        pipe.send({"error_type": type(exc).__name__})
    finally:
        if listener is not None:
            listener.close()
        if native is not None:
            native.close()
        for descriptor in descriptors:
            os.close(descriptor)


def _new_client(directory, token=TOKEN):
    root = Path(directory)
    return drain.NativeDispatchClient(
        database=root / "control.duckdb",
        state_dir=root,
        configuration=_configuration(),
        source_head="a" * 40,
        source_tree="b" * 40,
        token=token,
    )


def _peer_lane(directory, pipe):
    client = _new_client(directory)
    pipe.send(drain._birth(os.getpid()))
    while True:
        command = pipe.recv()
        if command == "stop":
            return
        if command == "preclaim":
            pipe.send(client.before_claim())
        elif command == "retained":
            client.retained_work(None)
            pipe.send({"retained": True})


def _peer_supervisor(directory, pipe):
    import sys

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
        SUPERVISED_CHILD_IDENTITY_PATH_ENV,
        SUPERVISED_CHILD_OWNER_SCOPE_ENV,
        SupervisedChildSpec,
        clear_child_pid_file,
        launch_supervised_child,
    )

    context = multiprocessing.get_context("fork")
    root = Path(directory)
    owner_scope = {"repo_root": str(root), "state_prefix": "disposable-lane"}

    def launch(suffix="current"):
        parent_pipe, child_pipe = context.Pipe()
        code = (
            "import sys; from multiprocessing.connection import Connection; "
            "from test.api.test_sawm_native_dispatch_drain import _peer_lane; "
            "_peer_lane(sys.argv[1], Connection(int(sys.argv[2])))"
        )
        command = (sys.executable, "-c", code, directory, str(child_pipe.fileno()))
        spec = SupervisedChildSpec(
            repo_root=Path(__file__).resolve().parents[2],
            command=command,
            log_path=root / f"{suffix}.log",
            child_pid_path=root / f"{suffix}.pid",
            pass_fds=(child_pipe.fileno(),),
            env={
                SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(
                    root / f"{suffix}.identity.json"
                ),
                SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps(owner_scope),
            },
        )
        child = launch_supervised_child(spec)
        assert parent_pipe.poll(10)
        birth = parent_pipe.recv()
        assert (
            child.identity_process_birth.start_time_ticks == birth["start_time_ticks"]
        )
        return child, parent_pipe, birth

    def close(child, child_pipe):
        child_pipe.send("stop")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            pid, status = os.waitpid(child.pid, os.WNOHANG)
            if pid:
                assert os.waitstatus_to_exitcode(status) == 0
                clear_child_pid_file(child)
                return
            time.sleep(0.01)
        raise AssertionError("disposable native child did not exit")

    child, child_pipe, birth = launch()
    native = object.__new__(PortalImplementationSupervisor)
    native.config = SimpleNamespace(
        repo_root=root, configured_board_live_admission=None
    )
    native._native_dispatch_control = _new_client(directory)
    native._managed_daemon_owner_scope = lambda: dict(owner_scope)
    native._build_daemon_command = lambda: list(child.command)

    def register():
        native._refresh_native_dispatch_child(child)

    pipe.send({"supervisor": drain._birth(os.getpid()), "daemon": birth})
    try:
        while True:
            command = pipe.recv()
            if command == "register":
                register()
                pipe.send({"registered": True})
            elif command == "replace":
                close(child, child_pipe)
                child, child_pipe, birth = launch()
                register()
                pipe.send({"daemon": birth})
            elif command == "ancillary":
                other, other_pipe, _ = launch("ancillary")
                other_pipe.send("preclaim")
                assert other_pipe.poll(5)
                result = other_pipe.recv()
                close(other, other_pipe)
                pipe.send(result)
            else:
                if command == "stop":
                    close(child, child_pipe)
                    child = None
                    return
                child_pipe.send(command)
                pipe.send(child_pipe.recv())
    finally:
        if child is not None:
            close(child, child_pipe)


@pytest.fixture
def peer_native(tmp_path):
    context = multiprocessing.get_context("fork")
    (tmp_path / "state").mkdir()
    (tmp_path / "state/configured-board-master.pid").write_text(str(os.getpid()))
    owner_pipe, owner_child = context.Pipe()
    owner = context.Process(target=_peer_owner, args=(str(tmp_path), owner_child))
    owner.start()
    assert owner_pipe.poll(10)
    assert owner_pipe.recv() == {"ready": True}
    supervisor_pipe, supervisor_child = context.Pipe()
    supervisor = context.Process(
        target=_peer_supervisor, args=(str(tmp_path), supervisor_child)
    )
    supervisor.start()
    assert supervisor_pipe.poll(10)
    births = supervisor_pipe.recv()
    client = _new_client(tmp_path)

    def roster():
        return client.exchange(
            "coordinator_boundary",
            {"lanes": [{"lane": "lane-0", "supervisor_birth": births["supervisor"]}]},
        )

    roster()
    supervisor_pipe.send("register")
    assert supervisor_pipe.poll(5)
    assert supervisor_pipe.recv() == {"registered": True}
    native = SimpleNamespace(
        client=client,
        public=_new_client(tmp_path, token=""),
        root=tmp_path,
        supervisor_pipe=supervisor_pipe,
        owner_pipe=owner_pipe,
        supervisor=supervisor,
        owner=owner,
        births=births,
        roster=roster,
    )
    try:
        yield native
    finally:
        if supervisor.is_alive():
            supervisor_pipe.send("stop")
            supervisor.join(5)
        if owner.is_alive():
            owner_pipe.send("stop")
            owner.join(5)
        assert not supervisor.is_alive()
        assert not owner.is_alive()


def _public(native, operation="status", request_id=""):
    return native.public.exchange(
        operation, {"master_birth": drain._birth(os.getpid()), "request_id": request_id}
    )


def _lane(native, command):
    native.supervisor_pipe.send(command)
    # Replacement and ancillary probes include a real child launch (10s),
    # native child shutdown (5s), and, for ancillary, a preclaim reply (5s).
    # The caller must allow those existing bounded operations to finish.
    timeout = 25 if command in {"replace", "ancillary"} else 5
    assert native.supervisor_pipe.poll(timeout)
    return native.supervisor_pipe.recv()


def test_real_owner_ack_is_not_pause_and_retained_work_is_not_closure(peer_native):
    native = peer_native
    assert _lane(native, "preclaim")["new_dispatch_permitted"] is True
    ack = _public(native, "request")
    assert ack["request_received"] is True
    assert ack["state"]["dispatch_pause_observed"] is False
    assert _lane(native, "retained") == {"retained": True}
    native.roster()
    state = _public(native)["state"]
    assert state["dispatch_pause_observed"] is False
    assert state["lanes"][0]["retained_work_reported"] is True
    assert _lane(native, "preclaim")["new_dispatch_permitted"] is False
    state = _public(native)["state"]
    assert state["dispatch_pause_observed"] is True
    assert state["callback_custody_known"] is False
    assert state["terminal_custody"] == "unknown"
    assert all(
        state[key] is False
        for key in [
            "task_authority",
            "completion_authority",
            "source_transition_authority",
            "signals_sent",
        ]
    )
    native.owner_pipe.send("state")
    assert native.owner_pipe.recv() == [(17,)]
    assert TOKEN not in json.dumps(state)
    _public(native, "release", ack["state"]["request_id"])
    assert _lane(native, "preclaim")["new_dispatch_permitted"] is True


def test_real_new_epoch_replacement_and_listener_reopen_invalidate_old_ack(peer_native):
    native = peer_native
    first = _public(native, "request")
    native.roster()
    _lane(native, "preclaim")
    assert _public(native)["state"]["dispatch_pause_observed"] is True
    replacement = _lane(native, "replace")
    assert replacement["daemon"] != native.births["daemon"]
    assert _public(native)["state"]["dispatch_pause_observed"] is False
    _lane(native, "preclaim")
    assert _public(native)["state"]["dispatch_pause_observed"] is True
    native.owner_pipe.send("restart_listener")
    assert native.owner_pipe.recv() == {"ready": True}
    assert _public(native)["state"]["request_id"] == first["state"]["request_id"]
    _public(native, "release", first["state"]["request_id"])
    second = _public(native, "request")
    assert second["state"]["drain_epoch"] > first["state"]["drain_epoch"]
    assert second["state"]["dispatch_pause_observed"] is False
    native.roster()
    assert _public(native)["state"]["dispatch_pause_observed"] is False
    _lane(native, "preclaim")
    assert _public(native)["state"]["dispatch_pause_observed"] is True


@pytest.mark.parametrize(
    "violation",
    [
        "wrong_token",
        "wrong_source",
        "wrong_generation",
        "wrong_master",
        "foreign_lane",
        "missing_lock",
    ],
)
def test_real_native_scope_and_custody_fail_closed(peer_native, violation):
    native = peer_native
    client = _new_client(native.root)
    if violation == "wrong_token":
        client._token = "wrong-token"
        operation, body = "coordinator_boundary", {"lanes": []}
    elif violation == "wrong_source":
        client.source_head = "c" * 40
        operation, body = (
            "status",
            {"master_birth": drain._birth(os.getpid()), "request_id": ""},
        )
    elif violation == "wrong_generation":
        client.configuration["database_program"]["store_generation"] = 2
        operation, body = (
            "status",
            {"master_birth": drain._birth(os.getpid()), "request_id": ""},
        )
    elif violation == "wrong_master":
        operation, body = (
            "request",
            {"master_birth": native.births["supervisor"], "request_id": ""},
        )
    elif violation == "foreign_lane":
        operation, body = "lane_boundary", {"phase": "preclaim"}
    else:
        native.owner_pipe.send("release_lock")
        assert native.owner_pipe.recv() == {"ready": True}
        operation, body = (
            "status",
            {"master_birth": drain._birth(os.getpid()), "request_id": ""},
        )
    with pytest.raises(drain.DispatchObservationUnavailable):
        client.exchange(operation, body)
    assert native.owner.is_alive()
    assert native.supervisor.is_alive()


def test_malformed_actual_socket_does_not_retire_owner_or_create_pause(peer_native):
    native = peer_native
    scope = native.client._scope()
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
        connection.settimeout(2)
        connection.connect(drain._address(scope))
        connection.sendall(b"[" * 1200 + b"0" + b"]" * 1200)
        reply = json.loads(connection.recv(16384))
    assert reply["request_received"] is False
    assert reply["new_dispatch_permitted"] is False
    assert _public(native)["state"]["drain_requested"] is False
    assert native.owner.is_alive()


def test_failed_final_master_check_does_not_clear_retained_pause(
    peer_native, monkeypatch
):
    import copy
    import uuid

    native = peer_native
    scope = native.client._scope()
    service = object.__new__(drain.NativeDrainService)
    service.scope = scope
    service.observer = SimpleNamespace(database=native.root / "control.duckdb")
    service.server = SimpleNamespace(
        lifecycle=SimpleNamespace(value="ready"),
        identity=SimpleNamespace(to_dict=lambda: dict(scope["owner_identity"])),
    )
    service.master_path = native.root / "state/configured-board-master.pid"
    service.state = drain.DrainState(1)
    service.state.request_id = "f" * 32
    service.state.epoch = 7
    before = copy.deepcopy(service.state.__dict__)
    birth = drain._birth(os.getpid())
    changed = {**birth, "start_time_ticks": birth["start_time_ticks"] + 1}
    observations = iter([birth, changed])
    monkeypatch.setattr(drain, "_master", lambda _: next(observations))
    packet = {
        "schema": drain.SCHEMA,
        "scope_cid": observation._digest(scope),
        "nonce": uuid.uuid4().hex,
        "birth": birth,
        "operation": "release",
        "sent_at": time.time(),
        "body": {"master_birth": birth, "request_id": "f" * 32},
        "proof": "",
    }
    with pytest.raises(drain.DispatchObservationUnavailable):
        service.handle(packet, os.getpid(), os.getuid())
    assert service.state.__dict__ == before


def test_replayed_native_boundary_packet_is_denied(peer_native):
    import uuid

    native = peer_native
    scope = native.client._scope()
    packet = {
        "schema": drain.SCHEMA,
        "scope_cid": observation._digest(scope),
        "nonce": uuid.uuid4().hex,
        "birth": drain._birth(os.getpid()),
        "operation": "coordinator_boundary",
        "sent_at": time.time(),
        "body": {
            "lanes": [
                {"lane": "lane-0", "supervisor_birth": native.births["supervisor"]}
            ]
        },
    }
    packet["proof"] = drain._proof(TOKEN, packet)

    def send():
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
            connection.settimeout(2)
            connection.connect(drain._address(scope))
            connection.sendall(drain._raw(packet))
            return json.loads(connection.recv(drain.MAX_PACKET))

    assert send()["request_received"] is True
    assert send()["request_received"] is False
    assert native.owner.is_alive()


@pytest.mark.parametrize("violation", ["uid", "birth"])
def test_client_rejects_wrong_kernel_receiver_identity(
    peer_native, monkeypatch, violation
):
    native = peer_native
    original = observation._peer

    def different(connection):
        pid, uid = original(connection)
        return (pid + 1, uid) if violation == "birth" else (pid, uid + 1)

    monkeypatch.setattr(observation, "_peer", different)
    with pytest.raises(drain.DispatchObservationUnavailable):
        _public(native)


def test_stale_lane_boundary_never_establishes_paused_state(peer_native, monkeypatch):
    native = peer_native
    scope = native.client._scope()
    state = drain.DrainState(1)
    state.request_id, state.epoch = "e" * 32, 1
    state.roster(
        drain._birth(os.getpid()),
        [{"lane": "lane-0", "supervisor_birth": native.births["supervisor"]}],
    )
    state.register_daemon(
        native.births["supervisor"],
        native.births["daemon"],
        drain._command_sha256(native.births["daemon"]["pid"]),
    )
    state.boundary(native.births["daemon"], "preclaim")
    assert state.projection()["dispatch_pause_observed"] is True
    clock = time.monotonic()
    monkeypatch.setattr(
        drain.time, "monotonic", lambda: clock + drain.FRESH_SECONDS + 1
    )
    assert state.projection()["dispatch_pause_observed"] is False
    assert scope["owner_identity"]["generation"] == 1


def test_ancillary_token_holding_child_cannot_ack_for_registered_daemon(peer_native):
    native = peer_native
    _public(native, "request")
    native.roster()
    assert _lane(native, "ancillary")["new_dispatch_permitted"] is False
    state = _public(native)["state"]
    assert state["lanes"][0]["daemon_birth"] == native.births["daemon"]
    assert state["dispatch_pause_observed"] is False
    _lane(native, "preclaim")
    assert _public(native)["state"]["dispatch_pause_observed"] is True


def test_actual_coordinator_waits_for_release_before_any_new_launch(
    peer_native, monkeypatch
):
    import sys

    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner as runner,
    )

    native = peer_native
    ack = _public(native, "request")
    released = threading.Event()
    errors = []

    def release():
        try:
            time.sleep(0.15)
            _public(native, "release", ack["state"]["request_id"])
            released.set()
        except BaseException as exc:  # noqa: BLE001 - isolated fixture reports only error type
            errors.append(type(exc).__name__)

    thread = threading.Thread(target=release)
    track = runner.parse_track_spec(
        "lane-0|unused.py|logs/{stamp}.log|state/super.pid|state/daemon.pid",
        stamp="TEST",
    )
    calls = []

    class Process:
        pid = native.supervisor.pid

        def poll(self):
            return None

    def start(*args, **kwargs):
        assert released.is_set()
        calls.append("launch")
        return Process()

    monkeypatch.setattr(drain, "from_native_admission", lambda **_: native.client)
    monkeypatch.setattr(runner, "start_track", start)
    monkeypatch.setattr(
        runner,
        "supervisor_status_health_fields",
        lambda *a, **k: {"restart_supervisor": False},
    )
    monkeypatch.setattr(
        runner,
        "daemon_pid_health_fields",
        lambda *a, **k: {
            "daemon_pid": native.births["daemon"]["pid"],
            "daemon_alive": True,
        },
    )
    # The existing finite test window still owns its teardown policy. This
    # observer test retains its fixture processes, and claims no closure.
    monkeypatch.setattr(
        runner,
        "stop_tracks",
        lambda *a, **k: {
            "all_trees_fenced": False,
            "stopped_pids": [],
            "stopped_count": 0,
            "removed_runtime_markers": [],
            "stop_failure_receipts": [],
        },
    )
    thread.start()
    try:
        runner.run_supervisor_tracks(
            [track],
            repo_root=native.root,
            common_args=[],
            duration_seconds=0.5,
            heartbeat_interval_seconds=0.05,
            python_executable=sys.executable,
            output=lambda _: None,
        )
    finally:
        thread.join(5)
    assert errors == []
    assert calls == ["launch"]
    assert native.supervisor.is_alive()
    assert native.owner.is_alive()


def test_actual_coordinator_pause_never_recycles_existing_lane(
    peer_native, monkeypatch
):
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner as runner,
    )

    native = peer_native
    requested = threading.Event()
    launched = threading.Event()
    errors = []

    def request():
        try:
            assert launched.wait(5)
            _public(native, "request")
            requested.set()
        except BaseException as exc:  # noqa: BLE001 - isolated fixture reports only error type
            errors.append(type(exc).__name__)

    class Process:
        pid = native.supervisor.pid

        def poll(self):
            return None

    def start(*args, **kwargs):
        launched.set()
        return Process()

    monkeypatch.setattr(drain, "from_native_admission", lambda **_: native.client)
    monkeypatch.setattr(runner, "start_track", start)
    monkeypatch.setattr(
        runner,
        "supervisor_status_health_fields",
        lambda *a, **k: {"restart_supervisor": requested.is_set()},
    )
    monkeypatch.setattr(
        runner,
        "daemon_pid_health_fields",
        lambda *a, **k: {
            "daemon_pid": native.births["daemon"]["pid"],
            "daemon_alive": True,
        },
    )
    fencing = []

    def forbidden(*args, **kwargs):
        fencing.append(True)
        return False, ()

    monkeypatch.setattr(runner, "_terminate_managed_process", forbidden)
    monkeypatch.setattr(
        runner,
        "stop_tracks",
        lambda *a, **k: {
            "all_trees_fenced": False,
            "stopped_pids": [],
            "stopped_count": 0,
            "removed_runtime_markers": [],
            "stop_failure_receipts": [],
        },
    )
    thread = threading.Thread(target=request)
    thread.start()
    try:
        result = runner.run_supervisor_tracks(
            [
                runner.parse_track_spec(
                    "lane-0|unused.py|logs/{stamp}.log|state/super.pid|state/daemon.pid",
                    stamp="TEST",
                )
            ],
            repo_root=native.root,
            common_args=[],
            duration_seconds=0.3,
            heartbeat_interval_seconds=0.05,
            output=lambda _: None,
        )
    finally:
        thread.join(5)
    assert errors == []
    assert requested.is_set()
    assert fencing == []
    assert result["all_trees_fenced"] is False
    assert native.supervisor.is_alive()
    assert native.owner.is_alive()


@pytest.mark.parametrize(
    "reason",
    ["native_dispatch_pause_requested", "native_dispatch_observation_unavailable"],
)
def test_initial_native_dispatch_wait_obeys_finite_run_without_birth_or_signal(
    tmp_path, monkeypatch, reason
):
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner as runner,
    )

    began = time.monotonic()
    launches, teardown_populations = [], []

    class Boundary:
        def coordinator_boundary(self, _processes):
            # The delayed permit bounds the original broken implementation
            # too; no real child or native service is involved in this test.
            return {
                "new_dispatch_permitted": time.monotonic() - began >= 0.45,
                "reason": reason,
            }

    class Process:
        pid = os.getpid()

        def poll(self):
            return None

    def start(*_args, **_kwargs):
        launches.append(True)
        return Process()

    def stopped(_tracks, processes, **_kwargs):
        teardown_populations.append(dict(processes))
        return {
            "all_trees_fenced": not processes,
            "stopped_pids": [],
            "stopped_count": 0,
            "removed_runtime_markers": [],
            "stop_failure_receipts": [],
        }

    monkeypatch.setattr(drain, "from_native_admission", lambda **_: Boundary())
    monkeypatch.setattr(runner, "start_track", start)
    monkeypatch.setattr(runner, "stop_tracks", stopped)
    monkeypatch.setattr(
        runner,
        "_terminate_managed_process",
        lambda *a, **k: pytest.fail("pause cannot signal a process"),
    )
    runner.run_supervisor_tracks(
        [
            runner.parse_track_spec(
                "lane-0|unused.py|logs/{stamp}.log|state/super.pid|state/daemon.pid",
                stamp="TEST",
            )
        ],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.1,
        heartbeat_interval_seconds=0.05,
        output=lambda _: None,
    )
    assert launches == []
    assert teardown_populations == [{}]
    assert time.monotonic() - began < 0.4


from test.api import (
    test_agent_supervisor_configured_board_live_capsule as capsule_fixtures,
)

quack_projection = capsule_fixtures.quack_projection


def _admitted_native_context(tmp_path, projection):
    import hashlib

    from ipfs_accelerate_py.agent_supervisor.runtime import (
        configured_board_live_capsule as capsule,
    )
    from test.api.test_agent_supervisor_configured_board_live_capsule import (
        _authority,
        _commit_controls,
        _native_authorization,
        _native_pin,
        _pin,
        _seed,
    )

    pin, extension_set, _ = projection
    root, paths = _seed(tmp_path, extension_set)
    config_path = root / "config/scheduler.json"
    config = json.loads(config_path.read_text())
    config.update(
        board_namespace=drain.PROGRAM,
        task_prefix="SAWM-",
        max_lanes=1,
        runtime_paths={"state": "state"},
    )
    config["quack_owner"].update(
        database_path="data/control.duckdb",
        state_dir="state/owner",
        store_id="data/control.duckdb",
        repository_id="repository:sawm",
        secret_handle="env://TEST_QUACK_TOKEN",
    )
    config_path.write_text(json.dumps(config, sort_keys=True))
    _commit_controls(root, "Bind disposable SAWM dispatch config")
    native = _native_pin()
    authorization = _native_authorization(native)
    admission = capsule.build_configured_board_live_capsule_admission(
        repo_root=root,
        board_namespace=drain.PROGRAM,
        plan_revision=config["plan_revision"],
        task_prefix="SAWM-",
        config_path="config/scheduler.json",
        configuration_root=capsule._cid(
            {"bytes_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
        ),
        control_paths=tuple(sorted(paths)),
        control_plane_pin=_pin(root),
        native_authorization_id=authorization["authorization_id"],
        native_dependency_id=native.dependency_id,
        native_python_executable_sha256=native.python_executable_sha256,
        quack_extension_projection=pin,
        extension_set_pin=extension_set,
        database_authority=_authority(),
        max_lanes=1,
        strict_task_sharding=True,
    )
    return root, admission


def test_native_factory_uses_admitted_source_config_and_own_inherited_scope(
    tmp_path, quack_projection, monkeypatch
):
    root, admission = _admitted_native_context(tmp_path, quack_projection)
    monkeypatch.setenv("TEST_QUACK_TOKEN", TOKEN)
    client = drain.from_native_admission(admission=admission, repo_root=root)
    assert client.source_head == admission.source_head
    assert client.source_tree == admission.source_tree
    assert client.configuration["board_namespace"] == drain.PROGRAM
    assert (
        client.before_claim()["new_dispatch_permitted"] is False
    )  # No owner, no permissive fallback.
    assert drain.from_native_admission(admission=None, repo_root=root) is None
    with pytest.raises(drain.DispatchObservationUnavailable):
        drain.from_native_admission(
            admission=SimpleNamespace(board_namespace=drain.PROGRAM), repo_root=root
        )
    assert TOKEN not in repr(client)


@pytest.mark.parametrize(
    "violation", ["changed_config", "missing_token", "wrong_configuration_root", "fifo"]
)
def test_native_factory_rejects_changed_or_unavailable_bootstrap_binding(
    tmp_path, quack_projection, monkeypatch, violation
):
    from dataclasses import replace

    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_live_capsule import (
        ConfiguredBoardLiveCapsuleError,
    )

    root, admission = _admitted_native_context(tmp_path, quack_projection)
    monkeypatch.setenv("TEST_QUACK_TOKEN", TOKEN)
    if violation == "changed_config":
        (root / admission.config_path).write_text("{}")
    elif violation == "missing_token":
        monkeypatch.delenv("TEST_QUACK_TOKEN")
    elif violation == "wrong_configuration_root":
        admission = replace(admission, configuration_root="sha256:" + "0" * 64)
    else:
        path = root / admission.config_path
        path.unlink()
        os.mkfifo(path)
    started = time.monotonic()
    with pytest.raises(
        (drain.DispatchObservationUnavailable, ConfiguredBoardLiveCapsuleError)
    ):
        drain.from_native_admission(admission=admission, repo_root=root)
    assert time.monotonic() - started < 2


def test_native_supervisor_loop_enables_exact_child_identity_for_admitted_sawm_only(
    tmp_path, monkeypatch
):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
        PortalSupervisorConfig,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
        SUPERVISED_CHILD_IDENTITY_PATH_ENV,
        SUPERVISED_CHILD_OWNER_SCOPE_ENV,
    )

    config = PortalSupervisorConfig(
        todo_path=tmp_path / "board.duckdb",
        state_path=tmp_path / "task.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        state_dir=tmp_path,
        repo_root=tmp_path,
    )
    native = object.__new__(PortalImplementationSupervisor)
    native.config = config
    native._build_daemon_command = lambda: ["python", "-m", "native-daemon-fixture"]
    native._proof_rollout_status_fields = dict
    native._autonomous_unstall_status = dict
    native._control_plane_status_projection = dict
    native._implementation_watchdog_timeout_seconds = lambda: 1
    native._watchdog_startup_grace_seconds = lambda: 1
    ordinary = native.build_supervisor_loop_config()
    assert ordinary.child_env == {}
    config.configured_board_live_admission = SimpleNamespace(
        board_namespace=drain.PROGRAM
    )
    scoped = native.build_supervisor_loop_config()
    assert scoped.child_env[SUPERVISED_CHILD_IDENTITY_PATH_ENV] == str(
        native._managed_daemon_identity_path()
    )
    assert (
        json.loads(scoped.child_env[SUPERVISED_CHILD_OWNER_SCOPE_ENV])
        == native._managed_daemon_owner_scope()
    )
    assert scoped.spec.pass_fds == ordinary.spec.pass_fds
    assert scoped.command == ordinary.command
