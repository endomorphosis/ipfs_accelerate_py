"""Native CLI adapter for the existing private, PID-bound owner bootstrap."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading
import time

from ..merge.worktree_lifecycle import read_process_birth
from ..merge.database_worktree_registry import process_birth_id

from ..task_sources.state_owner_bootstrap import request_state_owner_bootstrap
from ..task_sources.quack_state_client import QuackStateClient
from ..task_sources.typed_database_task_source import TypedDatabaseTaskSource

STARTUP_WAIT_ENV = "IPFS_ACCELERATE_STATE_OWNER_BOOTSTRAP_TIMEOUT_MS"


def _startup_wait_seconds():
    value = os.environ.get(STARTUP_WAIT_ENV)
    if value is None:
        return 30.0
    if (not re.fullmatch(r"[0-9]{1,6}", value)
            or not 2_000 <= int(value) <= 120_000):
        raise ValueError("native bootstrap wait must be in 2000..120000 milliseconds")
    return int(value) / 1000


class NativeOwnerHeartbeat:
    """Live owner read evidence, independent of completed daemon passes.

    Every sequence requires a successful native typed owner generation read.
    This observation grants no task progress, completion or dispatch authority.
    A separate native client avoids blocking the main client's request lock.
    """

    def __init__(self, *, credentials, state_dir: Path, state_prefix: str,
                 interval_seconds: float = 0.25):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", state_prefix):
            raise ValueError("invalid live owner heartbeat prefix")
        if not 0.1 <= interval_seconds <= 1.0:
            raise ValueError("live owner heartbeat interval must be bounded")
        self.credentials = credentials
        self.path = Path(state_dir) / f"{state_prefix}_native_owner_heartbeat.json"
        self.interval = interval_seconds
        self.stop_event = threading.Event()
        self.thread = None
        self.sequence = 0
        self.last_error = ""
        self.client = QuackStateClient(
            owner_id=credentials.client_id, store_id=credentials.store_id,
            process_birth_id=credentials.process_birth_id,
        )

    def start(self):
        self.client.attach(self.credentials.endpoint, server_id=self.credentials.server_id)
        self.source = TypedDatabaseTaskSource(
            self.client, execution_route_policy=self.credentials.execution_route_policy,
        )
        self._publish()
        self.thread = threading.Thread(target=self._run, name="native-owner-heartbeat", daemon=True)
        self.thread.start()
        return self

    def _publish(self):
        binding = self.source.require_quack_authority_binding(
            expected_endpoint=self.credentials.endpoint,
            expected_process_instance_id=self.credentials.process_birth_id,
            bootstrap_credentials=self.credentials,
        )
        birth = read_process_birth(os.getpid())
        if birth is None or process_birth_id(birth) != self.credentials.process_birth_id:
            raise ValueError("live owner heartbeat process birth changed")
        self.sequence += 1
        payload = {
            "schema": "native-typed-owner-live-observation@1",
            "observed_at_ms": time.time_ns() // 1_000_000, "sequence": self.sequence,
            "process_birth": birth.to_dict(), "process_birth_id": process_birth_id(birth),
            **{key: binding[key] for key in ("store_id", "server_id", "generation", "fence_epoch", "route_policy_id")},
            "client_id": self.credentials.client_id, "owner_read_succeeded": True,
            "completion_authority": False, "task_progress_authority": False,
            "dispatch_authority": False,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".tmp-{os.getpid()}")
        with temporary.open("w") as stream:
            os.chmod(temporary, 0o600)
            json.dump(payload, stream, sort_keys=True)
        os.replace(temporary, self.path)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                self._publish()
            except Exception as exc:
                # Preserve the last real observation; its age fails health.
                self.last_error = type(exc).__name__

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.client.detach()


def database_owner_bootstrap_kwargs(args, program) -> dict:
    descriptor = getattr(args, "state_owner_bootstrap_fd", -1)
    client_id = str(getattr(args, "state_owner_client_id", "") or "")
    if descriptor == -1 and not client_id:
        return {}
    if (type(descriptor) is not int or descriptor < 3 or not client_id
            or program is None or program.authority_mode != "quack"
            or program.task_source_kind != "duckdb"):
        raise ValueError("native owner bootstrap requires its descriptor, client and Quack program")
    credentials = request_state_owner_bootstrap(
        descriptor, client_id=client_id, store_id=program.store_id,
        timeout_seconds=_startup_wait_seconds(),
    )
    if credentials.endpoint != program.quack_endpoint:
        raise ValueError("bootstrap owner endpoint differs from the launch program")
    credentials.install_environment()
    client = QuackStateClient(
        owner_id=credentials.client_id, store_id=credentials.store_id,
        process_birth_id=credentials.process_birth_id,
    )
    try:
        client.attach(credentials.endpoint, server_id=credentials.server_id)
        source = TypedDatabaseTaskSource(client, execution_route_policy=credentials.execution_route_policy)
        source.require_quack_authority_binding(
            expected_endpoint=program.quack_endpoint,
            expected_process_instance_id=credentials.process_birth_id,
            bootstrap_credentials=credentials,
        )
    except BaseException:
        client.detach()
        raise
    return {
        "task_source": source, "close_task_source": True,
        "process_instance_id": credentials.process_birth_id,
        "state_owner_bootstrap_credentials": credentials,
    }


def observe_typed_database_without_dispatch(daemon) -> dict:
    """Read actual queue state without creating a noop execution attempt."""
    if type(daemon.task_source) is not TypedDatabaseTaskSource:
        raise ValueError("observation mode requires the native typed owner")
    daemon.open()
    tasks = daemon.task_source.ready_tasks(limit=256)
    return {
        "schema": "native-typed-daemon-observation@1",
        "ready_task_count": len(tasks.tasks), "selection_idle_reason": "implementation_disabled",
        "unchanged": True, "write_count": 0, "provider_dispatched": False,
        "attempt_consumed": False, "completion_authority": False,
    }


# The embedded observation is deliberately separate from the typed owner path.
_EMBEDDED_OBSERVATION_BINDING_KEYS = frozenset({
    "schema", "database_preimage", "task_scope", "plan_root_cid",
    "repository_tree_id", "observer_source", "source_schema",
})


def _embedded_observation_file(path, maximum):
    """Read a bounded owner file with before/open/close/path identity fences."""
    import hashlib
    import stat
    path = Path(path)
    if (not path.is_absolute() or any(item.is_symlink() for item in (path, *path.parents))):
        raise ValueError("embedded observation requires an unaliased absolute file")
    before = path.lstat()
    def identity(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink,
                info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
            or before.st_nlink != 1 or not 0 < before.st_size <= maximum):
        raise ValueError("embedded observation file exceeds owner bounds")
    digest = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        opened = os.fstat(stream.fileno())
        if identity(before) != identity(opened):
            raise ValueError("embedded observation file changed before open")
        count = 0
        while True:
            chunk = stream.read(min(65536, maximum + 1 - count))
            if not chunk:
                break
            count += len(chunk)
            if count > maximum:
                raise ValueError("embedded observation file grew beyond its cap")
            digest.update(chunk)
        closed = os.fstat(stream.fileno())
    if (count != before.st_size or identity(before) != identity(closed)
            or identity(before) != identity(path.lstat())):
        raise ValueError("embedded observation file changed during read")
    return {"path": str(path), "bytes": count, "sha256": digest.hexdigest()}


def observe_embedded_database_without_dispatch(daemon, *, binding) -> dict:
    """Observe a finite embedded scope through the original read-session API.

    This does not activate the daemon: its public task_source property and open()
    create daemon metadata, so the exact real daemon's already supplied source is
    inspected only after provenance/type checks. No schemas, claims, attempts,
    providers or validation hooks are installed or invoked. read_session rejects
    repository writes in this thread; it is not a frozen SQL snapshot or kernel
    read-only handle. A changing cursor/projection/scope fails this observation.
    """
    import inspect
    import stat
    from .implementation_daemon import DatabaseImplementationDaemon
    from ..task_sources.database_task_source import DatabaseTaskSource
    from ..task_sources.intent_repository import IntentRepository

    def require(value, message):
        if not value:
            raise ValueError(message)
    def descriptor(value, maximum):
        require(type(value) is dict and set(value) == {"path", "bytes", "sha256"},
                "closed embedded observation descriptor required")
        require(type(value["path"]) is str and type(value["bytes"]) is int
                and 0 < value["bytes"] <= maximum and type(value["sha256"]) is str
                and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None,
                "bounded embedded observation descriptor required")
        return value

    require(type(binding) is dict and set(binding) == _EMBEDDED_OBSERVATION_BINDING_KEYS
            and binding["schema"] == "native-embedded-daemon-observation-binding@1",
            "closed embedded observation binding required")
    scope = binding["task_scope"]
    require(type(scope) is list and 1 <= len(scope) <= 256
            and all(type(item) is str and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:-]{0,255}", item)
                    for item in scope) and len(set(scope)) == len(scope),
            "finite unique exact task CID scope required")
    require(all(type(binding[name]) is str and 0 < len(binding[name]) <= 4096
                for name in ("plan_root_cid", "repository_tree_id", "source_schema")),
            "sealed catalogue root/tree required")
    prior = descriptor(binding["database_preimage"], 128 * 1024**2)
    observer = descriptor(binding["observer_source"], 4 * 1024**2)
    require(observer["path"] == str(Path(__file__).absolute())
            and _embedded_observation_file(observer["path"], 4 * 1024**2) == observer,
            "current canonical observer source descriptor differs")
    require(type(daemon) is DatabaseImplementationDaemon
            and Path(inspect.getfile(type(daemon))).absolute()
                == Path(__file__).with_name("implementation_daemon.py").absolute(),
            "exact original database daemon required")
    require(daemon.authority_mode == "embedded_exclusive" and daemon.task_source_kind == "duckdb"
            and daemon._connection is None and daemon._coordinator is None
            and daemon._embedded_writer_lock_handle is None and daemon._closed is False
            and daemon._typed_quack_authority_binding is None
            and daemon._provider_fn is None and daemon._effect_fn is None
            and daemon._validation_fn is None and daemon.max_task_attempts == 0
            and daemon.execution_slice_task_cids == frozenset(scope)
            and not daemon.execution_slice_task_ids,
            "observation requires an unactivated embedded daemon without dispatch hooks")
    # Do not access daemon.task_source: that property activates the daemon.
    source = daemon._task_source
    require(type(source) is DatabaseTaskSource
            and Path(inspect.getfile(type(source))).absolute()
                == Path(__file__).parent.parent / "task_sources/database_task_source.py",
            "exact original embedded database task source required")
    intent = source.intent
    require(type(intent) is IntentRepository
            and Path(inspect.getfile(type(intent))).absolute()
                == Path(__file__).parent.parent / "task_sources/intent_repository.py"
            and intent._bound_connection is None and intent.uses_quack_transport is False
            and intent.is_open and getattr(intent._read_session_state, "connection", None) is None,
            "original unbound embedded read-session repository required")
    database = Path(prior["path"])
    require(Path(daemon.database_path) == database and Path(source.database_path) == database
            and Path(intent.database_path) == database and Path(intent._open_target) == database,
            "actual embedded database path differs")
    require(source.plan_root_cid == binding["plan_root_cid"]
            and source.repository_tree_id == binding["repository_tree_id"],
            "source catalogue binding differs")
    require(_embedded_observation_file(database, 128 * 1024**2) == prior,
            "embedded database preimage changed before observation")
    db_before = database.lstat()
    def database_identity(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink)
    def scoped_rows():
        rows = []
        for cid in scope:
            task = source.get_task(cid)
            require(task is not None and task.task_cid == cid
                    and task.plan_cid == binding["plan_root_cid"]
                    and type(task.revision) is int and task.revision >= 1
                    and type(task.status) is str and 0 < len(task.status) <= 128,
                    "scoped task catalogue identity is absent or mismatched")
            rows.append({"task_cid": cid, "status": task.status, "revision": task.revision})
        return rows
    with intent.read_session(read_only=True):
        before = source.snapshot()
        rows_before = scoped_rows()
        # The original selector filters exact CIDs BEFORE the limit. A global
        # first page cannot establish that a finite scope has no ready task.
        page = source.ready_tasks(limit=len(scope), task_cids=tuple(scope))
        ready = [task.task_cid for task in page.tasks]
        require(len(set(ready)) == len(ready) and set(ready) <= set(scope)
                and not page.next_cursor, "scoped ready selection exceeds its bound")
        rows_after = scoped_rows()
        after = source.snapshot()
        require(before == after and rows_before == rows_after
                and page.revision == max(1, before.event_cursor)
                and before.plan_root_cid == binding["plan_root_cid"]
                and before.repository_tree_id == binding["repository_tree_id"]
                and before.source_schema == binding["source_schema"],
                "embedded catalogue changed during bounded observation; retry a fresh cycle")
    require(database_identity(db_before) == database_identity(database.lstat())
            and _embedded_observation_file(observer["path"], 4 * 1024**2) == observer,
            "embedded database or observer custody changed")
    require(daemon._connection is None and daemon._coordinator is None
            and daemon._embedded_writer_lock_handle is None,
            "observation unexpectedly activated the daemon")
    return {
        "schema": "native-embedded-daemon-observation@1",
        "ready_task_count": len(ready),
        "selection_idle_reason": "scoped_ready" if ready else "no_scoped_ready_tasks",
        "unchanged": True, "write_count": 0, "provider_dispatched": False,
        "attempt_consumed": False, "completion_authority": False,
        "backend": "embedded", "database_path": str(database), "task_scope": list(scope),
        "observer_source": observer,
        "observer_callable": "observe_embedded_database_without_dispatch",
        "catalogue_binding": {
            "database_preimage": prior, "plan_root_cid": before.plan_root_cid,
            "repository_tree_id": before.repository_tree_id,
            "source_schema": before.source_schema, "event_cursor": before.event_cursor,
            "task_statuses": rows_before,
        },
    }
