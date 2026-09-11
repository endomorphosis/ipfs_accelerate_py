"""Bounded local observations from the native exclusive writer.

The descriptor is only an endpoint locator. A reply requires a fresh challenge,
kernel peer credentials, exact process birth, and both native writer locks on
the configured canonical store. No credentials, SQL or completion grants cross
this channel. The native owner calls poll between its ordinary transactions.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import socket
import stat
import struct
import threading
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..merge.worktree_lifecycle import read_process_birth

SCHEMA = "ipfs_accelerate_py/agent-supervisor/owner-status-observation@1"
DESCRIPTOR = "owner-status-observation.json"
MAX_PACKET = 65536
MAX_TASKS = 256
MAX_AGE_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 30
QUERY_BUDGET_SECONDS = 0.25
_OWNER_FIELDS = {"server_id", "store_id", "database_uuid", "generation", "fence_epoch",
                 "process_birth_id", "repository_id"}
_SCOPE_FIELDS = {"schema", "program_id", "configuration_cid", "source_head", "source_tree",
                 "owner_identity", "owner_birth", "uid", "store_identity", "lock_identities",
                 "task_registry_cid"}


class OwnerObservationUnavailable(RuntimeError):
    def __init__(self):
        super().__init__("native owner observation unavailable")


def _encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _birth(pid: int) -> dict[str, Any]:
    birth = read_process_birth(pid)
    if birth is None or birth.start_time_ticks <= 0 or not birth.boot_id:
        raise OwnerObservationUnavailable()
    return {"pid": pid, "start_time_ticks": birth.start_time_ticks, "boot_id": birth.boot_id}


def _regular_identity(path: Path) -> dict[str, int]:
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode) or observed.st_uid != os.getuid():
        raise OwnerObservationUnavailable()
    return {"device": observed.st_dev, "inode": observed.st_ino}


def _locks(database: Path) -> tuple[Path, Path]:
    return (database.with_name(f".{database.name}.lock"),
            database.with_name(f".{database.name}.state-owner.lock"))


def _read_bounded(path: Path, maximum: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode) or observed.st_size > maximum:
            raise OwnerObservationUnavailable()
        chunks = []
        size = 0
        while size <= maximum:
            raw = os.read(fd, maximum + 1 - size)
            if not raw:
                return b"".join(chunks)
            chunks.append(raw)
            size += len(raw)
        raise OwnerObservationUnavailable()
    finally:
        os.close(fd)


def _decode(raw: bytes) -> dict[str, Any]:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise OwnerObservationUnavailable()
            result[key] = value
        return result
    result = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(result, dict):
        raise OwnerObservationUnavailable()
    return result


def _validated_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    if (set(scope) != _SCOPE_FIELDS or scope.get("schema") != SCHEMA
            or type(scope.get("uid")) is not int or scope["uid"] != os.getuid()):
        raise OwnerObservationUnavailable()
    for field in ("configuration_cid", "task_registry_cid"):
        if not isinstance(scope[field], str) or not re.fullmatch(r"[0-9a-f]{64}", scope[field]):
            raise OwnerObservationUnavailable()
    for field in ("source_head", "source_tree"):
        if not isinstance(scope[field], str) or not re.fullmatch(r"[0-9a-f]{40}", scope[field]):
            raise OwnerObservationUnavailable()
    if (not isinstance(scope["program_id"], str) or not 1 <= len(scope["program_id"]) <= 256
            or not isinstance(scope["owner_identity"], dict)
            or set(scope["owner_identity"]) != _OWNER_FIELDS):
        raise OwnerObservationUnavailable()
    owner = scope["owner_identity"]
    for field in _OWNER_FIELDS:
        if field in {"generation", "fence_epoch"}:
            valid = type(owner[field]) is int and owner[field] > 0
        else:
            valid = isinstance(owner[field], str) and 0 < len(owner[field]) <= 512
        if not valid:
            raise OwnerObservationUnavailable()
    birth = scope["owner_birth"]
    if (not isinstance(birth, dict) or set(birth) != {"pid", "start_time_ticks", "boot_id"}
            or any(type(birth[k]) is not int or birth[k] <= 0 for k in ("pid", "start_time_ticks"))
            or not isinstance(birth["boot_id"], str) or not birth["boot_id"]):
        raise OwnerObservationUnavailable()
    if not isinstance(scope["lock_identities"], list) or len(scope["lock_identities"]) != 2:
        raise OwnerObservationUnavailable()
    for identity in [scope["store_identity"], *scope["lock_identities"]]:
        if (not isinstance(identity, dict) or set(identity) != {"device", "inode"}
                or any(type(identity[k]) is not int or identity[k] <= 0 for k in identity)):
            raise OwnerObservationUnavailable()
    raw = _encoded(scope)
    if len(raw) > 8192:
        raise OwnerObservationUnavailable()
    return _decode(raw)


def _custody(scope: Mapping[str, Any], database: Path) -> None:
    """Check actual writer custody; a mutable descriptor alone proves nothing."""
    pid = scope["owner_birth"]["pid"]
    if _birth(pid) != scope["owner_birth"] or os.stat(f"/proc/{pid}").st_uid != scope["uid"]:
        raise OwnerObservationUnavailable()
    if _regular_identity(database) != scope["store_identity"]:
        raise OwnerObservationUnavailable()
    identities = [_regular_identity(path) for path in _locks(database)]
    if identities != scope["lock_identities"]:
        raise OwnerObservationUnavailable()
    wanted = {(os.major(i["device"]), os.minor(i["device"]), i["inode"]) for i in identities}
    held = set()
    writer = False
    store = scope["store_identity"]
    store_lock = (os.major(store["device"]), os.minor(store["device"]), store["inode"])
    for line in _read_bounded(Path("/proc/locks"), 1024 * 1024).decode().splitlines():
        parts = line.split()
        if (len(parts) == 8 and parts[1:5] == ["FLOCK", "ADVISORY", "WRITE", str(pid)]
                and parts[6:] == ["0", "EOF"]):
            major, minor, inode = parts[5].split(":")
            held.add((int(major, 16), int(minor, 16), int(inode)))
        if (len(parts) == 8 and parts[1:5] == ["POSIX", "ADVISORY", "WRITE", str(pid)]
                and parts[6:] == ["0", "EOF"]):
            major, minor, inode = parts[5].split(":")
            writer = (int(major, 16), int(minor, 16), int(inode)) == store_lock or writer
    if not wanted <= held or not writer:
        raise OwnerObservationUnavailable()
    # The peer must also hold the exact canonical database, not only lock files.
    found = False
    entries = list(itertools.islice(Path(f"/proc/{pid}/fd").iterdir(), 4097))
    if len(entries) > 4096:
        raise OwnerObservationUnavailable()
    for entry in entries:
        try:
            observed = entry.stat()
        except FileNotFoundError:
            continue
        if {"device": observed.st_dev, "inode": observed.st_ino} == scope["store_identity"]:
            found = True
            break
    if not found or _birth(pid) != scope["owner_birth"]:
        raise OwnerObservationUnavailable()


def _address(scope: Mapping[str, Any]) -> str:
    return "\0ipfs-owner-status-" + _digest(scope)


def _peer(connection: socket.socket) -> tuple[int, int]:
    pid, uid, _ = struct.unpack("3i", connection.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    return pid, uid


def _receive(connection: socket.socket) -> dict[str, Any]:
    raw, _, flags, _ = connection.recvmsg(MAX_PACKET)
    if not raw or flags & socket.MSG_TRUNC:
        raise OwnerObservationUnavailable()
    return _decode(raw)


def _validate_facts(facts: Mapping[str, Any]) -> None:
    if (set(facts) != {"task_count", "task_counts", "task_statuses", "task_revisions", "event_cursor",
                      "authenticated_query", "transport", "completion_authority", "source_transition_authority"}
            or type(facts["task_count"]) is not int or not 1 <= facts["task_count"] <= MAX_TASKS
            or type(facts["event_cursor"]) is not int or facts["event_cursor"] < 0
            or facts["authenticated_query"] is not True
            or facts["transport"] != "native_exclusive_owner_peer_observation"
            or facts["completion_authority"] is not False or facts["source_transition_authority"] is not False):
        raise OwnerObservationUnavailable()
    counts, statuses, revisions = (facts[key] for key in ("task_counts", "task_statuses", "task_revisions"))
    if (not all(isinstance(value, dict) for value in (counts, statuses, revisions))
            or len(statuses) != facts["task_count"] or set(revisions) != set(statuses)
            or any(not isinstance(alias, str) or not 1 <= len(alias) <= 256 for alias in statuses)
            or any(not isinstance(status, str) or not re.fullmatch(r"[a-z_]{1,40}", status) for status in statuses.values())
            or any(type(revision) is not int or revision < 1 for revision in revisions.values())
            or any(type(count) is not int or count < 1 for count in counts.values())
            or counts != dict(Counter(statuses.values()))):
        raise OwnerObservationUnavailable()


@contextmanager
def _query_budget(connection: Any):
    """Interrupt only this owner's observation; retire the timer before reuse.

    DuckDB's native interrupt cooperatively cancels query execution. This is
    not a kernel I/O deadline. No connection operation runs on the timer thread
    except interrupt, and no later mutation can overlap a surviving timer.
    """
    import duckdb
    from ..task_sources.duckdb_state import DuckDBConnection

    if type(connection) is not DuckDBConnection:
        raise OwnerObservationUnavailable()
    # Keep typed transaction dispatch out until cancellation has been retired.
    # These closed native SELECTs may be interrupted without poisoning the
    # wrapper as an unknown mutation outcome or replacing its writer handle.
    lock = connection._execution_lock
    if not lock.acquire(blocking=False):
        raise OwnerObservationUnavailable()
    try:
        if (connection.in_transaction or connection._closed or connection._poisoned
                or connection._default_catalog is not None
                or type(connection._connection) is not duckdb.DuckDBPyConnection):
            raise OwnerObservationUnavailable()
        native = connection._connection
        stopped = threading.Event()
        expired = threading.Event()

        def deadline():
            if not stopped.wait(QUERY_BUDGET_SECONDS):
                expired.set()
                # A deadline can land between two read statements. Keep cancellation
                # armed until the owner leaves this budget, so a later statement
                # cannot miss a one-shot interrupt issued before execution began.
                while not stopped.is_set():
                    native.interrupt()
                    stopped.wait(0.01)

        timer = threading.Thread(target=deadline, name="native-owner-observation-deadline", daemon=True)
        timer.start()
        try:
            yield native
            if expired.is_set():
                raise OwnerObservationUnavailable()
        finally:
            stopped.set()
            timer.join()
    finally:
        lock.release()


class OwnerStatusObservation:
    """Read only from the already admitted owner's retained connection."""

    def __init__(self, server: Any, *, program_id: str, configuration: Mapping[str, Any],
                 source_head: str, source_tree: str, task_registry: Mapping[str, str]):
        self.server = server
        self.database = Path(server.config.database_path).resolve()
        self.registry = dict(task_registry)
        if not self.registry or len(self.registry) > MAX_TASKS or len(set(self.registry.values())) != len(self.registry):
            raise OwnerObservationUnavailable()
        identity = server.identity.to_dict()
        self.scope = _validated_scope({"schema": SCHEMA, "program_id": program_id,
            "configuration_cid": _digest(configuration), "source_head": source_head, "source_tree": source_tree,
            "owner_identity": {k: identity[k] for k in _OWNER_FIELDS}, "owner_birth": _birth(os.getpid()),
            "uid": os.getuid(), "store_identity": _regular_identity(self.database),
            "lock_identities": [_regular_identity(path) for path in _locks(self.database)],
            "task_registry_cid": _digest(self.registry)})
        _custody(self.scope, self.database)
        self.next_sample = 0.0
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        temporary = None
        temporary_identity = None
        try:
            self.listener.bind(_address(self.scope))
            self.listener.listen(4)
            self.listener.setblocking(False)
            descriptor = Path(server.config.state_dir) / DESCRIPTOR
            temporary = descriptor.with_name(f".{DESCRIPTOR}.{uuid.uuid4().hex}.tmp")
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            opened = os.fstat(fd)
            temporary_identity = (opened.st_dev, opened.st_ino)
            try:
                raw = _encoded(self.scope)
                if os.write(fd, raw) != len(raw):
                    raise OwnerObservationUnavailable()
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temporary, descriptor)
        except BaseException:
            self.close()
            raise
        finally:
            if temporary is not None and temporary_identity is not None:
                try:
                    observed = temporary.lstat()
                    if (observed.st_dev, observed.st_ino) == temporary_identity:
                        temporary.unlink()
                except OSError:
                    pass

    def close(self) -> None:
        try:
            self.listener.close()
        except OSError:
            pass

    def _snapshot(self) -> dict[str, Any]:
        lock = self.server._lock
        if not lock.acquire(blocking=False):
            raise OwnerObservationUnavailable()
        try:
            with _query_budget(self.server._connection) as connection:
                return self._bounded_snapshot(connection)
        finally:
            lock.release()

    def _bounded_snapshot(self, connection: Any) -> dict[str, Any]:
        _custody(self.scope, self.database)
        if self.server.lifecycle.value != "ready":
            raise OwnerObservationUnavailable()
        identity = self.server.identity.to_dict()
        if _encoded({k: identity[k] for k in _OWNER_FIELDS}) != _encoded(self.scope["owner_identity"]):
            raise OwnerObservationUnavailable()
        owner = self.scope["owner_identity"]
        # Scope validation requires an exact positive int (not bool or text).
        # Binding even this scalar can initialize DuckDB's optional Python
        # conversion imports on the first observation and exhaust its deadline.
        # Keep that initialization out of the native owner's bounded read path.
        generation = owner["generation"]
        if type(generation) is not int or generation <= 0:
            raise OwnerObservationUnavailable()
        row = connection.execute(
            "SELECT database_uuid, birth_id, fence_epoch FROM store_generations "
            f"WHERE generation = {generation}"
        ).fetchall()
        if len(row) != 1 or tuple(row[0][i] for i in range(3)) != (owner["database_uuid"], owner["process_birth_id"], owner["fence_epoch"]):
            raise OwnerObservationUnavailable()
        rows = connection.execute(
            "SELECT task_cid, task_alias, status, revision FROM tasks ORDER BY task_cid LIMIT 257"
        ).fetchall()
        if len(rows) != len(self.registry) or {r[0]: r[1] for r in rows} != self.registry:
            raise OwnerObservationUnavailable()
        if any(not isinstance(r[2], str) or not re.fullmatch(r"[a-z_]{1,40}", r[2])
               or type(r[3]) is not int or r[3] < 1 for r in rows):
            raise OwnerObservationUnavailable()
        cursor = connection.execute("SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events").fetchone()[0]
        if type(cursor) is not int or cursor < 0:
            raise OwnerObservationUnavailable()
        observed = {"task_count": len(rows), "task_counts": dict(Counter(r[2] for r in rows)),
            "task_statuses": {r[1]: r[2] for r in rows}, "task_revisions": {r[1]: r[3] for r in rows},
            "event_cursor": cursor, "authenticated_query": True,
            "transport": "native_exclusive_owner_peer_observation", "completion_authority": False,
            "source_transition_authority": False}
        _validate_facts(observed)
        _custody(self.scope, self.database)
        return observed

    def poll(self) -> None:
        # This optional diagnostic channel cannot shut down the native owner.
        if self.listener.fileno() < 0:
            # The owner loop can recreate a closed listener from its retained
            # launch scope; it cannot reconstruct task or completion authority.
            raise OwnerObservationUnavailable()
        try:
            connection, _ = self.listener.accept()
            with connection:
                connection.settimeout(0.02)
                nonce = None
                status = None
                observed_at = time.time()
                try:
                    pid, uid = _peer(connection)
                    packet = _receive(connection)
                    if (uid != self.scope["uid"] or set(packet) != {"schema", "scope_cid", "requester_birth", "nonce"}
                            or packet["schema"] != SCHEMA or packet["scope_cid"] != _digest(self.scope)
                            or _encoded(packet["requester_birth"]) != _encoded(_birth(pid))
                            or not isinstance(packet["nonce"], str) or not re.fullmatch(r"[0-9a-f]{32}", packet["nonce"])):
                        raise OwnerObservationUnavailable()
                    nonce = packet["nonce"]
                    now = time.monotonic()
                    if now >= self.next_sample:
                        self.next_sample = now + 1.0
                        # Do not hide a slow canonical query behind a new
                        # response timestamp. The reader admits sample age.
                        observed_at = time.time()
                        status = self._snapshot()
                    if _birth(pid) != packet["requester_birth"]:
                        status = None
                except Exception:
                    status = None
                response = {"schema": SCHEMA, "scope_cid": _digest(self.scope), "nonce": nonce,
                    "observed_at": observed_at, "task_authority": status,
                    "completion_authority": False, "source_transition_authority": False}
                raw = _encoded(response)
                if len(raw) <= MAX_PACKET:
                    connection.sendall(raw)
        except Exception:
            return


def read_owner_status(*, database: Path, state_dir: Path, program_id: str,
                      configuration: Mapping[str, Any], expected_owner: Mapping[str, Any],
                      expected_task_registry: Mapping[str, str]) -> dict[str, Any]:
    """Read fresh facts; descriptors and old replies never establish authority."""
    try:
        if (not {"store_id", "repository_id", "generation"} <= set(expected_owner)
                or not set(expected_owner) <= _OWNER_FIELDS):
            raise OwnerObservationUnavailable()
        scope = _validated_scope(_decode(_read_bounded(state_dir / DESCRIPTOR, 8192)))
        if (scope["program_id"] != program_id or scope["configuration_cid"] != _digest(configuration)
                or scope["task_registry_cid"] != _digest(expected_task_registry)
                or _encoded({k: scope["owner_identity"].get(k) for k in expected_owner}) != _encoded(expected_owner)):
            raise OwnerObservationUnavailable()
        _custody(scope, database)
        nonce = uuid.uuid4().hex
        started = time.monotonic()
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
            connection.settimeout(REQUEST_TIMEOUT_SECONDS)
            connection.connect(_address(scope))
            if _peer(connection) != (scope["owner_birth"]["pid"], scope["uid"]):
                raise OwnerObservationUnavailable()
            connection.sendall(_encoded({"schema": SCHEMA, "scope_cid": _digest(scope),
                "requester_birth": _birth(os.getpid()), "nonce": nonce}))
            reply = _receive(connection)
            if _peer(connection) != (scope["owner_birth"]["pid"], scope["uid"]):
                raise OwnerObservationUnavailable()
        _custody(scope, database)
        if (set(reply) != {"schema", "scope_cid", "nonce", "observed_at", "task_authority",
                          "completion_authority", "source_transition_authority"}
                or reply["schema"] != SCHEMA or reply["scope_cid"] != _digest(scope) or reply["nonce"] != nonce
                or reply["completion_authority"] is not False or reply["source_transition_authority"] is not False
                or type(reply["observed_at"]) not in (int, float)
                or not 0 <= time.time() - reply["observed_at"] <= MAX_AGE_SECONDS
                or time.monotonic() - started > REQUEST_TIMEOUT_SECONDS + 1
                or not isinstance(reply["task_authority"], dict)):
            raise OwnerObservationUnavailable()
        _validate_facts(reply["task_authority"])
        return {**reply, "owner_ready": True, "peer_authenticated_observation": True,
                "owner_identity": scope["owner_identity"], "source_head": scope["source_head"], "source_tree": scope["source_tree"],
                "source_context_only": True, "source_verified": False}
    except Exception:
        raise OwnerObservationUnavailable() from None
