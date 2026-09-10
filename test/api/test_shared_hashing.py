"""File observations reuse one real owner's reads without trusting mutable paths.

All databases, socket grants, and admission locks in these tests are disposable.
The process test gives each client a kernel-bound grant; no client opens DuckDB.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import shared_hashing as hashing
from ipfs_accelerate_py.agent_supervisor.task_sources import hash_observations
from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed_owner
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    HASH_OBSERVATION_SERVICE_OPERATION,
    TypedStateOwnerConnection,
    kernel_process_birth_id,
)
from test.api.test_agent_supervisor_hash_observation_transport import _real_database_server
from test.api.test_agent_supervisor_hash_observation_transport import owner as owner

from ipfs_accelerate_py import _hash_resources as resources

duckdb = pytest.importorskip("duckdb")
pytestmark = pytest.mark.skipif(not Path("/proc/self/fdinfo").exists(), reason="Linux file identity")
OWNER_ENV = (
    "IPFS_ACCELERATE_AGENT_STATE_STORE_ID",
    "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET",
    "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD",
)


class Clock:
    def __init__(self):
        self.wall = 1_000_000
        self.mono = 1_000

    def advance(self, milliseconds):
        self.wall += milliseconds
        self.mono += milliseconds


class LockedOwner:
    """The same transaction boundary as the service, without socket machinery."""

    def __init__(self, database, clock):
        self.database = database
        self.clock = clock
        self.lock = threading.Lock()
        self.requests = []
        self.store = hash_observations.HashObservationStore(
            database, generation="client-test-owner", clock_ms=lambda: clock.wall,
            monotonic_ms=lambda: clock.mono,
        )

    def hash_observation(self, request):
        with self.lock:
            self.requests.append(dict(request))
            return self.store.handle(request, principal="test-hash-client")


@pytest.fixture(autouse=True)
def isolated_admission(tmp_path, monkeypatch):
    path = tmp_path / "hash-admission.lock"
    monkeypatch.setattr(resources, "hash_lock_path", lambda kind="file-hash": path)
    monkeypatch.setattr(resources, "host_hash_pressure", lambda: (4, "test"))
    monkeypatch.setenv("IPFS_HASH_MAX_WORKERS", "2")
    monkeypatch.delenv("IPFS_HASH_CACHE_TTL_SECONDS", raising=False)
    for name in OWNER_ENV:
        monkeypatch.delenv(name, raising=False)
    # Spawned clients must not create large native pools while importing DuckDB.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        monkeypatch.setenv(name, "1")
    yield path
    assert not resources._open_fds


@pytest.fixture
def shared_owner():
    database = duckdb.connect(":memory:", config={"threads": 1})
    migration = Path(hash_observations.__file__).parent / "sql" / "0004_hash_observations.sql"
    database.execute(migration.read_text(encoding="utf-8"))
    adapter = LockedOwner(database, Clock())
    try:
        yield adapter
    finally:
        database.close()


@pytest.fixture
def payload(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"observed content\n" * 128)
    return path


@pytest.fixture
def reads(monkeypatch, tmp_path):
    calls = []
    original = os.pread

    def count(descriptor, length, offset):
        result = original(descriptor, length, offset)
        # A real broker also preads its sealed credential descriptor. Count
        # only disposable filesystem payloads, never credential memfd traffic.
        target = os.readlink(f"/proc/self/fd/{descriptor}")
        if target.startswith(str(tmp_path) + os.sep):
            calls.append((offset, len(result)))
        return result

    monkeypatch.setattr(hashing.os, "pread", count)
    return calls


def test_default_day_ttl_hits_without_rereading_or_renewing(shared_owner, payload, reads):
    first = hashing.hash_file(payload, connection=shared_owner)
    assert first == hashing.FileHashObservation(
        hashlib.sha256(payload.read_bytes()).hexdigest(), False, "fresh-bytes", payload.stat().st_size,
    )
    original_expiry = shared_owner.database.execute(
        "SELECT expires_at_ms FROM hash_observations"
    ).fetchone()[0]
    assert original_expiry == shared_owner.clock.wall + 86_400_000
    shared_owner.clock.advance(86_399_999)
    hit = hashing.hash_file(payload, connection=shared_owner)
    assert hit == hashing.FileHashObservation(first.sha256, True, "metadata-and-ttl", 0)
    assert sum(length for _, length in reads) == payload.stat().st_size
    assert shared_owner.database.execute(
        "SELECT expires_at_ms FROM hash_observations"
    ).fetchone()[0] == original_expiry
    shared_owner.clock.advance(1)
    refresh = hashing.hash_file(payload, connection=shared_owner)
    assert not refresh.cache_hit and refresh.sha256 == first.sha256
    assert sum(length for _, length in reads) == 2 * payload.stat().st_size


@pytest.mark.parametrize("options", [{"strict": True}, {"ttl_seconds": 0}])
def test_strict_and_zero_ttl_always_read_even_with_completed_observation(
    shared_owner, payload, reads, options,
):
    first = hashing.hash_file(payload, connection=shared_owner)
    requests = len(shared_owner.requests)
    for _ in range(2):
        fresh = hashing.hash_file(payload, connection=shared_owner, **options)
        assert fresh.sha256 == first.sha256 and fresh.freshness == "fresh-bytes"
        assert not fresh.cache_hit and fresh.bytes_read == payload.stat().st_size
    assert len(shared_owner.requests) == requests  # Strict validation needs no owner.
    assert sum(length for _, length in reads) == 3 * payload.stat().st_size


def test_environment_ttl_controls_expiration(shared_owner, payload, reads, monkeypatch):
    monkeypatch.setenv("IPFS_HASH_CACHE_TTL_SECONDS", "0.5")
    hashing.hash_file(payload, connection=shared_owner)
    shared_owner.clock.advance(499)
    assert hashing.hash_file(payload, connection=shared_owner).cache_hit
    shared_owner.clock.advance(1)
    assert not hashing.hash_file(payload, connection=shared_owner).cache_hit
    assert sum(length for _, length in reads) == 2 * payload.stat().st_size


def test_preserved_mtime_same_length_mutation_invalidates_by_ctime(shared_owner, payload, reads):
    old = hashing.hash_file(payload, connection=shared_owner)
    before = payload.stat()
    payload.write_bytes(b"x" * before.st_size)
    os.utime(payload, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = payload.stat()
    assert after.st_mtime_ns == before.st_mtime_ns and after.st_size == before.st_size
    assert after.st_ctime_ns != before.st_ctime_ns
    changed = hashing.hash_file(payload, connection=shared_owner)
    assert not changed.cache_hit and changed.sha256 != old.sha256
    assert changed.sha256 == hashlib.sha256(payload.read_bytes()).hexdigest()
    assert sum(length for _, length in reads) == 2 * before.st_size


def test_existing_hardlink_aliases_share_inode_observation(shared_owner, payload, reads):
    alias = payload.with_name("alias.bin")
    os.link(payload, alias)  # Establish nlink before either observation.
    first = hashing.hash_file(payload, connection=shared_owner)
    second = hashing.hash_file(alias, connection=shared_owner)
    assert second.cache_hit and second.sha256 == first.sha256
    assert sum(length for _, length in reads) == payload.stat().st_size
    assert shared_owner.database.execute("SELECT count(*) FROM hash_observations").fetchone()[0] == 1


def test_new_hardlink_invalidates_old_metadata_observation(shared_owner, payload, reads):
    first = hashing.hash_file(payload, connection=shared_owner)
    alias = payload.with_name("new-alias.bin")
    os.link(payload, alias)
    second = hashing.hash_file(alias, connection=shared_owner)
    assert not second.cache_hit and second.sha256 == first.sha256
    assert hashing.hash_file(payload, connection=shared_owner).cache_hit
    assert sum(length for _, length in reads) == 2 * payload.stat().st_size


def test_equal_content_distinct_inodes_require_distinct_measurements(shared_owner, payload, reads):
    copy = payload.with_name("copy.bin")
    copy.write_bytes(payload.read_bytes())
    first = hashing.hash_file(payload, connection=shared_owner)
    second = hashing.hash_file(copy, connection=shared_owner)
    assert first.sha256 == second.sha256 and not second.cache_hit
    assert sum(length for _, length in reads) == 2 * payload.stat().st_size


def test_changed_during_read_aborts_claim_and_never_publishes(shared_owner, payload, monkeypatch):
    original = os.pread
    mutated = False

    def mutate(descriptor, length, offset):
        nonlocal mutated
        content = original(descriptor, length, offset)
        if not mutated:
            mutated = True
            with payload.open("r+b") as stream:
                stream.write(b"!")
        return content

    monkeypatch.setattr(hashing.os, "pread", mutate)
    with pytest.raises(hashing.SharedHashError, match="file changed"):
        hashing.hash_file(payload, connection=shared_owner)
    assert [request["action"] for request in shared_owner.requests] == ["claim", "abort"]
    assert shared_owner.database.execute(
        "SELECT state, sha256 FROM hash_observations"
    ).fetchone() == ("aborted", "")
    monkeypatch.setattr(hashing.os, "pread", original)
    result = hashing.hash_file(payload, connection=shared_owner)
    assert not result.cache_hit and result.sha256 == hashlib.sha256(payload.read_bytes()).hexdigest()


def test_path_replacement_during_owner_hit_is_rejected(shared_owner, payload, monkeypatch):
    hashing.hash_file(payload, connection=shared_owner)
    replacement = payload.with_name("replacement.bin")
    replacement.write_bytes(b"different object")
    original = shared_owner.hash_observation

    def replace(request):
        result = original(request)
        if result["status"] == "hit":
            os.replace(replacement, payload)
        return result

    monkeypatch.setattr(shared_owner, "hash_observation", replace)
    with pytest.raises(hashing.SharedHashError, match="changed|replaced"):
        hashing.hash_file(payload, connection=shared_owner)


def test_descriptor_hash_preserves_caller_offset(shared_owner, payload):
    descriptor = os.open(payload, os.O_RDONLY)
    try:
        os.lseek(descriptor, 7, os.SEEK_SET)
        result = hashing.hash_descriptor(descriptor, connection=shared_owner)
        assert result.sha256 == hashlib.sha256(payload.read_bytes()).hexdigest()
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 7
        assert hashing.hash_descriptor(descriptor, connection=shared_owner).cache_hit
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 7
    finally:
        os.close(descriptor)


def test_maximum_bytes_applies_before_hits_and_reads(shared_owner, payload, reads):
    hashing.hash_file(payload, connection=shared_owner)
    count = len(reads)
    requests = len(shared_owner.requests)
    with pytest.raises(hashing.SharedHashError, match="byte limit"):
        hashing.hash_file(payload, connection=shared_owner, max_bytes=payload.stat().st_size - 1)
    assert len(reads) == count and len(shared_owner.requests) == requests


def test_empty_regular_file_has_standard_digest(shared_owner, tmp_path, reads):
    path = tmp_path / "empty"
    path.touch()
    first = hashing.hash_file(path, connection=shared_owner)
    assert first.sha256 == hashlib.sha256(b"").hexdigest() and first.bytes_read == 0
    assert not first.cache_hit and hashing.hash_file(path, connection=shared_owner).cache_hit
    assert reads == []


def test_symlink_and_directory_do_not_become_observations(shared_owner, payload):
    alias = payload.with_name("symlink")
    alias.symlink_to(payload)
    with pytest.raises(OSError):
        hashing.hash_file(alias, connection=shared_owner)
    with pytest.raises(hashing.SharedHashError, match="regular"):
        hashing.hash_file(payload.parent, connection=shared_owner)
    assert shared_owner.requests == []


def test_absent_owner_configuration_uses_bounded_fresh_local_reads(payload, reads, monkeypatch):
    admissions = []
    real_slot = hashing.hashing_worker_slot

    @contextmanager
    def slot(**kwargs):
        admissions.append(kwargs)
        with real_slot(**kwargs) as count:
            yield count

    monkeypatch.setattr(hashing, "hashing_worker_slot", slot)
    assert hashing.default_hash_connection() is None
    for _ in range(2):
        result = hashing.hash_file(payload)
        assert not result.cache_hit and result.freshness == "fresh-bytes"
    assert len(admissions) == 2 and all(0 < item["timeout"] <= 60 for item in admissions)
    assert sum(length for _, length in reads) == 2 * payload.stat().st_size


def test_configured_owner_failure_never_falls_back_to_independent_rehash(payload, reads, monkeypatch):
    for name in OWNER_ENV:
        monkeypatch.setenv(name, "configured-test-owner")

    def unavailable():
        raise ConnectionError("owner unavailable")

    monkeypatch.setattr(hashing, "_default_hash_connection_unlocked", unavailable)
    assert hashing.default_hash_connection() is not None
    with pytest.raises(ConnectionError, match="owner unavailable"):
        hashing.hash_file(payload)
    assert reads == []


@pytest.mark.parametrize("configured", [
    (OWNER_ENV[1],), (OWNER_ENV[2],), (OWNER_ENV[0], OWNER_ENV[1]),
    (OWNER_ENV[0], OWNER_ENV[2]), (OWNER_ENV[1], OWNER_ENV[2]),
])
def test_incomplete_owner_handoff_fails_closed(payload, reads, monkeypatch, configured):
    for name in configured:
        monkeypatch.setenv(name, "configured-test-owner")
    with pytest.raises(hashing.SharedHashError, match="incomplete"):
        hashing.hash_file(payload)
    assert reads == []


def test_store_identity_without_broker_is_not_an_authority_handoff(payload, reads, monkeypatch):
    monkeypatch.setenv(OWNER_ENV[0], "non-authority-read-only-store")
    assert hashing.default_hash_connection() is None
    assert hashing.hash_file(payload).freshness == "fresh-bytes"
    assert sum(length for _, length in reads) == payload.stat().st_size


@pytest.mark.parametrize("options", [{"strict": True}, {"ttl_seconds": 0}])
def test_strict_read_does_not_depend_on_configured_owner_availability(payload, reads, monkeypatch, options):
    for name in OWNER_ENV:
        monkeypatch.setenv(name, "configured-test-owner")

    def unavailable():
        pytest.fail("strict byte verification must not request cached observations")

    monkeypatch.setattr(hashing, "_default_hash_connection_unlocked", unavailable)
    result = hashing.hash_file(payload, **options)
    assert result.freshness == "fresh-bytes" and not result.cache_hit
    assert sum(length for _, length in reads) == payload.stat().st_size


def test_default_proxy_uses_sealed_broker_grant_renews_and_evicts_failed_session(
    tmp_path, payload, reads, monkeypatch,
):
    server = _real_database_server(tmp_path)
    monkeypatch.setattr(hashing, "_connections", {})
    issued = []
    real_credential = typed_owner.request_hash_observation_credential

    def credential(**kwargs):
        issued.append(kwargs)
        return real_credential(**kwargs)

    monkeypatch.setattr(typed_owner, "request_hash_observation_credential", credential)
    try:
        server.start()
        handoff = server.start_supervisor_grant_broker()
        monkeypatch.setenv(OWNER_ENV[0], server.config.store_id)
        for name in (*OWNER_ENV[1:], typed_owner.TYPED_STATE_OWNER_SOCKET_ENV):
            monkeypatch.setenv(name, handoff[name])
        fresh = hashing.hash_file(payload)
        assert not fresh.cache_hit and hashing.hash_file(payload).cache_hit
        assert len(issued) == 1 and len(hashing._connections) == 1
        first = next(iter(hashing._connections.values()))
        assert first.grant["allowed_operations"] == [HASH_OBSERVATION_SERVICE_OPERATION]
        assert issued[0]["client_id"] == f"hash-observer:{os.getpid()}"
        assert issued[0]["process_birth_id"] == kernel_process_birth_id()
        assert issued[0]["store_id"] == server.config.store_id

        # The cached authority must be renewed, not used after its expiry.
        first.grant = {**first.grant, "expires_at": 0}
        assert hashing.hash_file(payload).cache_hit
        assert len(issued) == 2 and len(hashing._connections) == 1
        second = next(iter(hashing._connections.values()))
        assert second is not first and first._closed
        attempts = []

        def lost_reply(request):
            attempts.append(request)
            raise ConnectionError("uncertain owner reply")

        monkeypatch.setattr(second, "hash_observation", lost_reply)
        with pytest.raises(ConnectionError, match="uncertain owner reply"):
            hashing.hash_file(payload)
        assert len(attempts) == 1 and len(issued) == 2  # No implicit replay.
        assert not hashing._connections and second._closed and second._socket.fileno() == -1
        assert hashing.hash_file(payload).cache_hit  # A new explicit call may reconnect.
        assert len(issued) == 3
        assert sum(length for _, length in reads) == payload.stat().st_size
    finally:
        hashing._close_connections()
        server.stop()


def test_explicit_owner_failure_never_falls_back_to_independent_rehash(payload, reads):
    class FailedOwner:
        def hash_observation(self, request):
            raise ConnectionError("owner disconnected")

    with pytest.raises(ConnectionError, match="owner disconnected"):
        hashing.hash_file(payload, connection=FailedOwner())
    assert reads == []


def test_busy_owner_wait_has_no_payload_reads_or_worker_slot(payload, reads, monkeypatch):
    requests = []

    class BusyOwner:
        def hash_observation(self, request):
            requests.append(request)
            return {"status": "busy"}

    def forbidden_slot(**kwargs):
        pytest.fail("waiting on another producer must not hold a hashing slot")

    monkeypatch.setattr(hashing, "hashing_worker_slot", forbidden_slot)
    with pytest.raises(TimeoutError, match="deadline"):
        hashing.hash_file(payload, connection=BusyOwner(), timeout_seconds=0.02)
    assert requests and reads == []


def test_owner_transactions_and_requests_are_outside_streaming_slot(
    shared_owner, payload, monkeypatch,
):
    original_read = os.pread
    original_request = shared_owner.hash_observation
    active_slot = False
    streamed = False

    @contextmanager
    def slot(**kwargs):
        nonlocal active_slot
        assert not active_slot and not shared_owner.lock.locked()
        active_slot = True
        try:
            yield 1
        finally:
            active_slot = False

    def read(descriptor, length, offset):
        nonlocal streamed
        assert active_slot and not shared_owner.lock.locked()
        # This would fail if the owner left its transaction open across a read.
        with shared_owner.lock:
            shared_owner.database.execute("BEGIN TRANSACTION")
            shared_owner.database.execute("ROLLBACK")
        streamed = True
        return original_read(descriptor, length, offset)

    def request(fields):
        assert not active_slot
        return original_request(fields)

    monkeypatch.setattr(hashing, "hashing_worker_slot", slot)
    monkeypatch.setattr(hashing.os, "pread", read)
    monkeypatch.setattr(shared_owner, "hash_observation", request)
    hashing.hash_file(payload, connection=shared_owner)
    assert streamed


def test_distinct_files_stream_in_parallel_but_never_exceed_shared_slots(
    shared_owner, payload, monkeypatch,
):
    paths = [payload]
    for index in range(2):
        path = payload.with_name(f"distinct-{index}.bin")
        path.write_bytes(bytes([index]) * 1024)
        paths.append(path)
    original_read = os.pread
    original_request = shared_owner.hash_observation
    state_lock = threading.Lock()
    release = threading.Event()
    two_streaming = threading.Event()
    three_claimed = threading.Event()
    state = {"active": 0, "maximum": 0, "reads": 0, "claims": 0}

    def read(descriptor, length, offset):
        with state_lock:
            state["active"] += 1
            state["reads"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
            if state["active"] == 2:
                two_streaming.set()
        try:
            assert release.wait(5), "parent did not release file reads"
            return original_read(descriptor, length, offset)
        finally:
            with state_lock:
                state["active"] -= 1

    def request(fields):
        response = original_request(fields)
        if response["status"] == "claimed":
            with state_lock:
                state["claims"] += 1
                if state["claims"] == 3:
                    three_claimed.set()
        return response

    monkeypatch.setattr(hashing.os, "pread", read)
    monkeypatch.setattr(shared_owner, "hash_observation", request)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(hashing.hash_file, path, connection=shared_owner)
                   for path in paths]
        try:
            assert two_streaming.wait(3), "distinct files did not overlap"
            assert three_claimed.wait(3), "owner request blocked behind payload reads"
            with state_lock:
                assert state["reads"] == 2 and state["active"] == 2
        finally:
            release.set()
        results = [future.result(timeout=5) for future in futures]
    assert state["maximum"] == 2 and state["reads"] == 3 and state["active"] == 0
    assert all(not result.cache_hit for result in results)
    assert [result.sha256 for result in results] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    ]


def test_subsecond_ttl_works_through_real_typed_owner(owner, payload, reads):
    _, _, connect = owner
    client, _ = connect("hash-subsecond-client")
    first = hashing.hash_file(payload, connection=client, ttl_seconds=0.75)
    second = hashing.hash_file(payload, connection=client, ttl_seconds=0.75)
    assert second.cache_hit and second.sha256 == first.sha256
    assert sum(length for _, length in reads) == payload.stat().st_size


@pytest.mark.parametrize("digest", [None, "a" * 63, "A" * 64, "z" * 64, 1])
def test_malformed_owner_hit_is_rejected_without_reading(payload, reads, digest):
    class BadOwner:
        def hash_observation(self, request):
            return {"status": "hit", "sha256": digest}

    with pytest.raises(hashing.SharedHashError, match="invalid SHA-256"):
        hashing.hash_file(payload, connection=BadOwner())
    assert reads == []


@pytest.mark.parametrize("ttl", [-1, 0.0001, float("nan"), float("inf"), 604801])
def test_invalid_ttl_rejected_before_owner_or_reads(shared_owner, payload, reads, ttl):
    with pytest.raises(ValueError, match="TTL"):
        hashing.hash_file(payload, connection=shared_owner, ttl_seconds=ttl)
    assert reads == [] and shared_owner.requests == []


def _process_observer(pipe, start, path, lock_path, socket_path, store_id):
    """A fresh process with only a hash-service grant, not database authority."""
    client = None
    try:
        resources.hash_lock_path = lambda kind="file-hash": Path(lock_path)
        resources.host_hash_pressure = lambda: (4, "test")
        client_id = f"test-hash-process:{os.getpid()}"
        birth = kernel_process_birth_id()
        pipe.send((os.getpid(), birth, client_id))
        token = pipe.recv()
        client = TypedStateOwnerConnection(
            socket_path=Path(socket_path), token=token, client_id=client_id,
            process_birth_id=birth, store_id=store_id, timeout_seconds=10,
        )
        original_read = os.pread
        actual_bytes = 0

        def count(descriptor, length, offset):
            nonlocal actual_bytes
            result = original_read(descriptor, length, offset)
            actual_bytes += len(result)
            time.sleep(0.10)  # Give the other clients time to observe the live lease.
            return result

        hashing.os.pread = count
        pipe.send("connected")
        if not start.wait(10):
            raise TimeoutError("parent did not start hash clients")
        result = hashing.hash_file(Path(path), connection=client, timeout_seconds=10)
        pipe.send(("ok", result.sha256, result.cache_hit, result.bytes_read, actual_bytes))
    except BaseException as error:
        pipe.send(("error", type(error).__name__, str(error)))
    finally:
        if client is not None:
            client.close()
        pipe.close()


def test_three_real_owner_process_clients_read_payload_only_once(owner, payload, isolated_admission):
    gateway, database, _ = owner
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    children = []
    pipes = []
    try:
        for _ in range(3):
            parent_pipe, child_pipe = context.Pipe()
            child = context.Process(
                target=_process_observer,
                args=(child_pipe, start, str(payload), str(isolated_admission),
                      str(gateway.socket_path), gateway.store_id),
            )
            child.start()
            child_pipe.close()
            children.append(child)
            pipes.append(parent_pipe)
        for pipe, child in zip(pipes, children, strict=True):
            assert pipe.poll(20), "spawned hash client did not identify itself"
            pid, birth, client_id = pipe.recv()
            assert pid == child.pid and birth == kernel_process_birth_id(pid)
            token, _ = gateway.issue_grant(
                client_id=client_id, process_birth_id=birth, peer_pid=pid,
                allowed_operations=(HASH_OBSERVATION_SERVICE_OPERATION,),
            )
            pipe.send(token)
        for pipe in pipes:
            assert pipe.poll(20), "hash client did not attach to owner"
            assert pipe.recv() == "connected"
        start.set()
        results = []
        for pipe in pipes:
            assert pipe.poll(20), "shared hash client did not finish"
            results.append(pipe.recv())
        assert all(result[0] == "ok" for result in results), results
        expected = hashlib.sha256(payload.read_bytes()).hexdigest()
        assert all(result[1] == expected for result in results)
        assert sum(result[2] for result in results) == 2
        assert sum(result[3] for result in results) == payload.stat().st_size
        assert sum(result[4] for result in results) == payload.stat().st_size
        for child in children:
            child.join(timeout=5)
            assert child.exitcode == 0
        assert database.execute(
            "SELECT count(*) FROM hash_observations WHERE state = 'complete'"
        ).fetchone()[0] == 1
    finally:
        start.set()
        for child in children:
            child.join(timeout=1)
            if child.is_alive():
                child.terminate()
                child.join(timeout=3)
        for pipe in pipes:
            pipe.close()
