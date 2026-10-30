"""Portal task state on the DuckDB/Quack control plane.

Current state is a DuckDB row, opened through the Quack transport when the
target is a ``quack:`` URI. DuckLake is a post-commit history projection and
is not authority. JSON task-state files are not written while this plane is
configured. The row cannot complete a task.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

ENV_QUACK = "IPFS_ACCELERATE_AGENT_QUACK_ENDPOINT"
ENV_DUCKDB = "IPFS_ACCELERATE_AGENT_CONTROL_DUCKDB"
ENV_DUCKLAKE = "IPFS_ACCELERATE_AGENT_CONTROL_DUCKLAKE"
TABLE = "portal_task_state"
HANDOFF_TABLE = "spar_merge_handoff"
DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    lane_key VARCHAR PRIMARY KEY,
    body_json VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    completion_authority BOOLEAN NOT NULL,
    updated_at VARCHAR NOT NULL
)
"""
HANDOFF_DDL = f"""
CREATE TABLE IF NOT EXISTS {HANDOFF_TABLE} (
    record_key VARCHAR PRIMARY KEY,
    body_json VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    completion_authority BOOLEAN NOT NULL,
    updated_at VARCHAR NOT NULL
)
"""


class PortalTaskStateControlPlaneError(ValueError):
    """Task state cannot be stored on the control plane."""


# This fragment is inserted after the original control-plane exception class.
# Its projections and registry are process-local correspondence, never grants.
_BORROWED_REGISTRY_LOCK = threading.RLock()
_BORROWED_REGISTRY_PID = os.getpid()
_BORROWED_BY_STATE: dict[Path, Any] = {}
_BORROWED_BY_TARGET: dict[Path, Any] = {}


def _borrow_need(condition: bool, reason: str) -> None:
    if not condition:
        raise PortalTaskStateControlPlaneError(reason)


def _borrow_path(value: Path | str, *, existing: bool = False) -> Path:
    _borrow_need(isinstance(value, (Path, str)), "borrowed_state_path_required")
    path = Path(value).expanduser()
    _borrow_need(path.is_absolute() and str(path) == os.path.abspath(path)
                 and not any(part.is_symlink() for part in (path, *path.parents)),
                 "borrowed_state_unaliased_absolute_path_required")
    resolved = path.resolve(strict=existing)
    _borrow_need(resolved == path, "borrowed_state_canonical_path_required")
    return path


def _borrow_file_identity(path: Path) -> tuple[int, int]:
    metadata = path.stat()
    _borrow_need(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
                 and metadata.st_uid in {0, os.getuid()} and not metadata.st_mode & 0o002,
                 "borrowed_state_original_local_regular_database_required")
    return metadata.st_dev, metadata.st_ino


def _borrow_thread_pid(pid: int, thread_id: int, thread: Any) -> None:
    """Necessary identity check only; this function cannot validate a handle."""
    _borrow_need(pid == os.getpid() and thread_id == threading.get_ident()
                 and thread is threading.current_thread(),
                 "borrowed_state_original_process_and_thread_required")


class _BorrowedStateBinding:
    def __init__(self, state_path: Path, target: Path, source: str, connection: Any):
        self.state_path, self.target, self.source = state_path, target, source
        self.connection = connection
        self.pid, self.thread_id = os.getpid(), threading.get_ident()
        self.thread = threading.current_thread()
        self.condition, self.execution_lock = connection._execution_condition, connection._execution_lock
        self.raw, self.lock_context = connection._connection, connection._lock_context
        self.epoch = connection._reconnect_epoch
        self.file_identity = _borrow_file_identity(target)

    def __reduce_ex__(self, _protocol):
        raise TypeError("borrowed state bindings cannot be serialized")


def _borrowed_binding(state_path: Path | str | None, target: str | None):
    if not _BORROWED_BY_STATE:
        return None
    # Refuse a fork before touching an inherited lock or native connection.
    _borrow_need(os.getpid() == _BORROWED_REGISTRY_PID, "borrowed_state_fork_refused")
    key = _borrow_path(state_path) if state_path is not None else None
    with _BORROWED_REGISTRY_LOCK:
        binding = _BORROWED_BY_STATE.get(key)
        if binding is not None:
            _borrow_thread_pid(binding.pid, binding.thread_id, binding.thread)
            _borrow_need(type(target) is str and target == str(binding.target),
                         "borrowed_state_original_explicit_target_changed")
            _borrow_need(_BORROWED_BY_TARGET.get(binding.target) is binding,
                         "borrowed_state_original_registry_identity_required")
            return binding
        if target is not None and not str(target).startswith("quack:"):
            candidate = Path(target).expanduser().resolve()
            _borrow_need(candidate not in _BORROWED_BY_TARGET,
                         "borrowed_state_exact_original_state_path_required")
        return None


def _check_borrowed_locked(binding: _BorrowedStateBinding) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import DuckDBConnection
    import duckdb

    connection = binding.connection
    _borrow_thread_pid(binding.pid, binding.thread_id, binding.thread)
    _borrow_need(type(connection) is DuckDBConnection
                 and type(connection._connection) is duckdb.DuckDBPyConnection
                 and connection._connection is binding.raw
                 and connection._execution_condition is binding.condition
                 and connection._execution_lock is binding.execution_lock
                 and binding.condition._lock is binding.execution_lock
                 and connection._lock_context is binding.lock_context and binding.lock_context is not None
                 and type(binding.epoch) is int and connection._reconnect_epoch == binding.epoch
                 and _borrow_path(connection.path, existing=True) == binding.target
                 and _borrow_file_identity(binding.target) == binding.file_identity,
                 "borrowed_state_original_native_handle_and_target_required")
    _borrow_need(connection._closed is False and connection._poisoned is False
                 and connection._owner_binding_retired is False and connection._transaction_active is False
                 and connection._context_finalizing is False and connection._closing_owner == 0
                 and connection._transaction_lock_owner == 0
                 and (connection._context_depth == 0 or connection._context_owner == binding.thread_id)
                 and connection._default_catalog is None and not connection._quack_uri
                 and not connection._quack_pending_mutations and connection._pooled is False,
                 "borrowed_state_usable_nontransactional_local_owner_required")
    # execute() may reconnect poison; the original condition and the checks
    # above fence that branch. These are SELECT-only actual catalog witnesses.
    databases = connection.execute(
        "SELECT database_name, path FROM duckdb_databases() WHERE internal = FALSE"
    ).fetchall()
    current = connection.execute("SELECT current_database(), current_schema()").fetchone()
    _borrow_need(len(databases) == 1 and len(databases[0]) == 2
                 and type(databases[0][1]) is str
                 and _borrow_path(databases[0][1], existing=True) == binding.target
                 and current is not None and len(current) == 2
                 and current[0] == databases[0][0] and current[1] == "main"
                 and connection._connection is binding.raw and connection._reconnect_epoch == binding.epoch
                 and _borrow_file_identity(binding.target) == binding.file_identity,
                 "borrowed_state_original_current_database_and_schema_required")


@contextmanager
def borrow_task_state_connection(
    state_path: Path | str, *, connection: Any, target: Path | str,
):
    """Borrow the caller's existing local handle; never acquire or close it.

    The trusted host owns opening/provenance/schema and final close. It must
    keep this scope entered throughout its original attempt and any unresolved
    custody. Current state routing must already have an explicit bound/env
    target; migration-capable implicit routing is unsupported in this mode.
    """
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import DuckDBConnection

    _borrow_need(os.getpid() == _BORROWED_REGISTRY_PID, "borrowed_state_fork_refused")
    _borrow_need(type(connection) is DuckDBConnection, "borrowed_state_actual_existing_connection_required")
    state_key, target_path = _borrow_path(state_path), _borrow_path(target, existing=True)
    selected = control_plane_target(state_key)
    _borrow_need(type(selected) is str and selected == str(target_path),
                 "borrowed_state_original_explicit_target_required")
    source = "bound" if _BOUND_STATE_TARGETS.get(str(state_key)) == selected else "env"
    # Only entry/registration takes registry then condition. Operations copy the
    # binding before acquiring its condition; no condition->registry lock edge.
    with _BORROWED_REGISTRY_LOCK:
        _borrow_need(state_key not in _BORROWED_BY_STATE and target_path not in _BORROWED_BY_TARGET,
                     "borrowed_state_nested_or_shared_scope_refused")
        with connection._execution_condition:
            binding = _BorrowedStateBinding(state_key, target_path, source, connection)
            _check_borrowed_locked(binding)
            _BORROWED_BY_STATE[state_key] = binding
            _BORROWED_BY_TARGET[target_path] = binding
    try:
        yield connection
    finally:
        _borrow_thread_pid(binding.pid, binding.thread_id, binding.thread)
        with _BORROWED_REGISTRY_LOCK:
            _borrow_need(_BORROWED_BY_STATE.get(state_key) is binding
                         and _BORROWED_BY_TARGET.get(target_path) is binding,
                         "borrowed_state_scope_identity_changed")
            del _BORROWED_BY_STATE[state_key]
            del _BORROWED_BY_TARGET[target_path]
        # No execute/commit/rollback/reconnect/close and no ownership transfer.


def require_borrowed_task_state_connection(
    state_path: Path | str, *, connection: Any = None,
):
    """Return the identical active original handle; metadata is not authority."""
    selected = control_plane_target(state_path)
    binding = _borrowed_binding(state_path, selected)
    _borrow_need(binding is not None, "borrowed_state_active_original_scope_required")
    _borrow_need(connection is None or connection is binding.connection,
                 "borrowed_state_identical_original_connection_required")
    with binding.condition:
        _check_borrowed_locked(binding)
    return binding.connection


def task_state_connection_is_borrowed(state_path: Path | str) -> bool:
    binding = _borrowed_binding(state_path, control_plane_target(state_path))
    if binding is None:
        return False
    require_borrowed_task_state_connection(state_path, connection=binding.connection)
    return True


def require_borrowed_task_state_source_profile(
    state_path: Path | str, *, task_source: Any, plan_revision_store: Any,
    require_active_plan_revision: bool, configured_parallel_execution_plan: Any,
    active_plan_binding: Any,
) -> None:
    """Only the original legacy Markdown/no-plan profile is supported in V01."""
    require_borrowed_task_state_connection(state_path)
    _borrow_need(task_source is None and plan_revision_store is None
                 and require_active_plan_revision is False
                 and configured_parallel_execution_plan is None and active_plan_binding is None,
                 "borrowed_state_task_source_or_plan_reopen_not_supported")


@contextmanager
def _state_connection(target: str, *, state_path: Path | str | None):
    binding = _borrowed_binding(state_path, target)
    if binding is not None:
        with binding.condition:
            _check_borrowed_locked(binding)
            yield binding.connection
        return
    connection = _connect(target)
    try:
        yield connection
    finally:
        connection.close()


def _refuse_borrowed_target_open(target: str) -> None:
    if not _BORROWED_BY_STATE:
        return
    _borrow_need(os.getpid() == _BORROWED_REGISTRY_PID, "borrowed_state_fork_refused")
    if str(target).startswith("quack:"):
        return
    candidate = Path(target).expanduser().resolve()
    with _BORROWED_REGISTRY_LOCK:
        _borrow_need(candidate not in _BORROWED_BY_TARGET,
                     "borrowed_state_same_target_reopen_refused")


def _state_ducklake_projection(
    state_path: Path | str | None, target: str, record_key: str,
    body_json: str, revision: int, updated_at: str,
):
    binding = _borrowed_binding(state_path, target)
    if binding is not None:
        require_borrowed_task_state_connection(state_path, connection=binding.connection)
        return {"status": "skipped_borrowed_state_connection", "authoritative": False,
                "completion_authority": False}
    return _project_ducklake(record_key, body_json, revision, updated_at, target=target)


_BOUND_STATE_TARGETS: dict[str, str] = {}


def bind_task_state_control_plane(state_path: Path | str, target: str) -> None:
    """Remember the Quack or DuckDB target for one task-state path."""

    text = str(target or "").strip()
    if not text:
        return
    if _BORROWED_BY_STATE:
        key = _borrow_path(state_path)
        with _BORROWED_REGISTRY_LOCK:
            _borrow_need(key not in _BORROWED_BY_STATE, "borrowed_state_binding_mutation_refused")
        _refuse_borrowed_target_open(text)
    _BOUND_STATE_TARGETS[str(Path(state_path).expanduser().resolve())] = text


def control_plane_target(state_path: Path | str | None = None) -> str | None:
    """Return the Quack URI or DuckDB path, or None when JSON remains.

    A target bound to ``state_path`` wins over the process environment so one
    daemon does not redirect another lane's JSON file.
    """

    if state_path is not None:
        bound = _BOUND_STATE_TARGETS.get(str(Path(state_path).expanduser().resolve()))
        if bound:
            return bound
    quack = str(os.environ.get(ENV_QUACK) or "").strip()
    if quack:
        return quack
    path = str(os.environ.get(ENV_DUCKDB) or "").strip()
    return path or None


def implicit_portal_database(state_path: Path | str) -> str:
    """Lane DuckDB under the account home, not beside the worktree or ``.git``.

    The directory is ``<home>/.ipfs_accelerate/agent_supervisor/orchestration``:
    ``/home/<user>`` on Linux, ``/Users/<user>`` on macOS, and
    ``C:\\Users\\<user>`` on Windows.
    """

    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        orchestration_scope_dir,
    )

    path = Path(state_path).expanduser().resolve()
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in path.stem)
    return str(orchestration_scope_dir(path) / "lanes" / f"{safe or 'lane'}.duckdb")


def _board_control_database(state_path: Path | str) -> Path | None:
    """Return the repo board-control ``control.duckdb``, if this path is in a repo."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        CONTROL_DATABASE_NAME,
        default_control_plane_root,
    )

    current = Path(state_path).expanduser()
    if current.is_file() or current.suffix:
        current = current.parent
    for parent in (current, *current.parents):
        if not (parent / ".git").exists():
            continue
        try:
            return default_control_plane_root(parent) / CONTROL_DATABASE_NAME
        except OSError:
            return None
    return None


def _is_board_control_database(target: str) -> bool:
    path = Path(target)
    return (
        path.name == "control.duckdb"
        and path.parent.name == "board-control-plane"
    )


def resolve_control_plane(state_path: Path | str | None = None) -> tuple[str | None, str]:
    """Return ``(target, source)`` where source is bound, env, board, implicit, or none."""

    selected = control_plane_target(state_path)
    binding = _borrowed_binding(state_path, selected)
    if binding is not None:
        require_borrowed_task_state_connection(state_path, connection=binding.connection)
        return str(binding.target), binding.source
    # Implicit legacy resolution can adopt/move catalogs. No such resolution
    # may occur incidentally inside an active borrowed attempt.
    _borrow_need(not _BORROWED_BY_STATE or selected is not None,
                 "borrowed_state_implicit_resolution_during_scope_refused")
    if state_path is not None:
        bound = _BOUND_STATE_TARGETS.get(str(Path(state_path).expanduser().resolve()))
        if bound and selected == bound:
            return bound, "bound"
    if selected is not None:
        return selected, "env"
    if state_path is not None:
        board = _board_control_database(state_path)
        if board is not None:
            return str(board), "board"
        return implicit_portal_database(state_path), "implicit"
    return None, "none"


def control_plane_target_for_coordination(directory: Path | str | None) -> str | None:
    """Return the control plane for a lane directory, including a bound state file.

    Environment configuration wins. Otherwise a task-state file bound beside
    ``semantic-world``, or one directory below it, selects that lane's database.
    """

    configured = control_plane_target()
    if configured is not None:
        return configured
    if directory is None:
        return None
    candidate = Path(directory).expanduser().resolve()
    for state_path, target in _BOUND_STATE_TARGETS.items():
        parent = Path(state_path).parent
        if candidate in {parent / "semantic-world", parent.parent / "semantic-world"}:
            return target
    return None


def resolve_handoff_plane(directory: Path | str | None) -> tuple[str | None, str]:
    """Return ``(target, source)`` for a merge handoff directory.

    Bound and environment targets are used as given. A directory with neither
    gets a handoff database under the account home
    (``<home>/.ipfs_accelerate/agent_supervisor``: Linux ``/home/<user>``,
    macOS ``/Users/<user>``, Windows ``C:\\Users\\<user>``), not inside the
    worktree.
    """

    if directory is not None and control_plane_target() is None:
        candidate = Path(directory).expanduser().resolve()
        for state_path, target in _BOUND_STATE_TARGETS.items():
            parent = Path(state_path).parent
            if candidate in {parent / "semantic-world", parent.parent / "semantic-world"}:
                return target, "bound"
    selected = control_plane_target_for_coordination(directory)
    if selected is not None:
        return selected, "env" if control_plane_target() is not None else "bound"
    if directory is None:
        return None, "none"
    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        orchestration_scope_dir,
    )

    root = Path(directory).expanduser().resolve()
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:20]
    return str(orchestration_scope_dir(root) / "handoffs" / f"{digest}.duckdb"), "implicit"


def target_from_task_source(source: Any) -> str | None:
    """Return a task source's Quack URI or DuckDB file, never a Markdown path."""

    if source is None:
        return None
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        is_quack_transport_target,
    )

    database = getattr(source, "database_path", None)
    text = str(database or "").strip()
    if not text:
        return None
    if is_quack_transport_target(text):
        return text
    if text.endswith((".duckdb", ".ddb")):
        return text
    return None


def _ducklake_root(target: str | None = None) -> Path | None:
    """History directory. Explicit env wins; a file target gets a sibling lake."""

    raw = str(os.environ.get(ENV_DUCKLAKE) or "").strip()
    if raw:
        return Path(raw)
    if not target:
        return None
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        is_quack_transport_target,
    )

    if is_quack_transport_target(target):
        return None
    return Path(target).expanduser().resolve().parent / "ducklake"


def _connect(target: str):
    _refuse_borrowed_target_open(target)
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        is_quack_transport_target,
        open_duckdb_connection,
    )

    bound = target in _BOUND_STATE_TARGETS.values()
    if not is_quack_transport_target(target):
        path = Path(target)
        if (
            path.name == "control.duckdb"
            and not bound
            and not _is_board_control_database(target)
        ):
            raise PortalTaskStateControlPlaneError(
                "portal task state refuses the exclusive extra-gate control.duckdb name"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = open_duckdb_connection(
        target,
        prefer_quack=is_quack_transport_target(target),
    )
    if _is_board_control_database(target):
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS artefact_generic (
                board_namespace VARCHAR,
                kind VARCHAR,
                artefact_id VARCHAR,
                payload_json VARCHAR,
                updated_at_ms BIGINT,
                PRIMARY KEY (board_namespace, kind, artefact_id)
            )
            """
        )
    else:
        connection.execute(DDL)
        connection.execute(HANDOFF_DDL)
    return connection


def _project_ducklake(
    record_key: str,
    body_json: str,
    revision: int,
    updated_at: str,
    *,
    kind: str = "portal_task_state",
    target: str | None = None,
) -> dict[str, Any]:
    root = _ducklake_root(target)
    if root is None:
        return {
            "status": "unconfigured",
            "authoritative": False,
            "completion_authority": False,
        }
    if not root.is_absolute():
        return {
            "status": "unavailable",
            "reason_code": "ducklake_root_must_be_absolute",
            "authoritative": False,
            "completion_authority": False,
        }
    try:
        import duckdb

        catalog = root / "metadata.ducklake"
        data = root / "parquet"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        data.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = duckdb.connect(
            ":memory:", config={"threads": 1, "memory_limit": "256MB"}
        )
        try:
            connection.execute("SET autoinstall_known_extensions=false")
            connection.execute("SET autoload_known_extensions=false")
            connection.execute("LOAD ducklake")
            catalog_sql = str(catalog).replace("'", "''")
            data_sql = str(data).replace("'", "''")
            connection.execute(
                "ATTACH 'ducklake:"
                + catalog_sql
                + "' AS task_state_lake (DATA_PATH '"
                + data_sql
                + "', DATA_INLINING_ROW_LIMIT 0)"
            )
            if kind == "spar_merge_handoff":
                relation = "spar_merge_handoff"
                key_column = "record_key"
            else:
                relation = "portal_task_state"
                key_column = "lane_key"
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS task_state_lake.{relation} (
                    {key_column} VARCHAR,
                    body_json VARCHAR,
                    revision BIGINT,
                    updated_at VARCHAR,
                    completion_authority BOOLEAN
                )
                """
            )
            connection.execute(
                f"""
                INSERT INTO task_state_lake.{relation}
                VALUES (?, ?, ?, ?, FALSE)
                """,
                [record_key, body_json, int(revision), updated_at],
            )
        finally:
            connection.close()
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason_code": type(exc).__name__,
            "authoritative": False,
            "completion_authority": False,
        }
    return {
        "status": "projected",
        "authoritative": False,
        "ducklake_authoritative": False,
        "completion_authority": False,
    }


def read_task_state_payload(path: Path | str) -> dict[str, Any] | None:
    """Return the DuckDB task-state row when it exists, otherwise the JSON mirror.

    Reading does not create a database. DuckLake is not consulted.
    """

    state_path = Path(path)
    routed = load_portal_task_state(state_path.stem, state_path=state_path, create=False)
    if routed:
        return routed
    if task_state_connection_is_borrowed(state_path):
        raise PortalTaskStateControlPlaneError("borrowed_state_legacy_json_fallback_refused")
    if not state_path.is_file():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def task_state_authority_file(path: Path | str) -> Path | None:
    """Path of the DuckDB file when it exists, else a legacy JSON file."""

    state_path = Path(path)
    target, _source = resolve_control_plane(state_path)
    if target:
        from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
            is_quack_transport_target,
        )

        database = Path(target)
        if not is_quack_transport_target(target) and database.is_file():
            return database
    if state_path.is_file():
        return state_path
    return None


def task_state_authority_mtime(path: Path | str) -> float | None:
    """Modification time of the DuckDB file, else the JSON mirror."""

    authority = task_state_authority_file(path)
    if authority is None:
        return None
    try:
        return authority.stat().st_mtime
    except OSError:
        return None


def read_ducklake_rows(
    root: Path | str,
    *,
    kind: str = "portal_task_state",
) -> list[dict[str, Any]]:
    """Read DuckLake history. This does not grant authority."""

    import duckdb

    lake = Path(root)
    relation = "spar_merge_handoff" if kind == "spar_merge_handoff" else "portal_task_state"
    key_column = "record_key" if relation == "spar_merge_handoff" else "lane_key"
    catalog = lake / "metadata.ducklake"
    data = lake / "parquet"
    connection = duckdb.connect(":memory:", config={"threads": 1, "memory_limit": "256MB"})
    try:
        connection.execute("SET autoinstall_known_extensions=false")
        connection.execute("SET autoload_known_extensions=false")
        connection.execute("LOAD ducklake")
        catalog_sql = str(catalog).replace("'", "''")
        data_sql = str(data).replace("'", "''")
        connection.execute(
            "ATTACH 'ducklake:"
            + catalog_sql
            + "' AS task_state_lake (DATA_PATH '"
            + data_sql
            + "', DATA_INLINING_ROW_LIMIT 0)"
        )
        rows = connection.execute(
            f"""
            SELECT {key_column}, body_json, revision, completion_authority
            FROM task_state_lake.{relation}
            """
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "key": row[0],
            "body_json": row[1],
            "revision": int(row[2]),
            "completion_authority": bool(row[3]),
            "authoritative": False,
        }
        for row in rows
    ]


_BOARD_NAMESPACE = "portal-task-state"
_BOARD_KIND = "portal_task_state"


def _save_board_task_state(
    target: str,
    key: str,
    body_json: str,
    updated_at: str,
    source: str,
    *, state_path: Path | str | None = None,
) -> dict[str, Any]:
    updated_ms = int(datetime.now(UTC).timestamp() * 1000)
    with _state_connection(target, state_path=state_path) as connection:
        current = connection.execute(
            """
            SELECT payload_json FROM artefact_generic
            WHERE board_namespace = ? AND kind = ? AND artefact_id = ?
            """,
            [_BOARD_NAMESPACE, _BOARD_KIND, key],
        ).fetchone()
        changed = current is None or str(current[0]) != body_json
        if changed:
            connection.execute(
                """
                INSERT INTO artefact_generic (
                    board_namespace, kind, artefact_id, payload_json, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (board_namespace, kind, artefact_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at_ms = excluded.updated_at_ms
                """,
                [_BOARD_NAMESPACE, _BOARD_KIND, key, body_json, updated_ms],
            )
    return {
        "changed": changed,
        "lane_key": key,
        "revision": updated_ms,
        "completion_authority": False,
        "ducklake": _state_ducklake_projection(
            state_path, target, key, body_json, updated_ms, updated_at
        ),
        "json_authority": False,
        "source": source,
        "store": "board-control-plane",
    }


def _load_board_task_state(target: str, key: str, *, create: bool, state_path: Path | str | None = None) -> dict[str, Any]:
    if not create and not Path(target).is_file():
        return {}
    with _state_connection(target, state_path=state_path) as connection:
        row = connection.execute(
            """
            SELECT payload_json FROM artefact_generic
            WHERE board_namespace = ? AND kind = ? AND artefact_id = ?
            """,
            [_BOARD_NAMESPACE, _BOARD_KIND, key],
        ).fetchone()
    if row is None:
        if state_path is not None and task_state_connection_is_borrowed(state_path):
            raise PortalTaskStateControlPlaneError("borrowed_state_original_authoritative_row_required")
        return {}
    payload = json.loads(str(row[0]))
    if not isinstance(payload, dict):
        raise PortalTaskStateControlPlaneError("stored task state is not an object")
    payload["completion_authority"] = False
    return payload


def save_portal_task_state(
    lane_key: str,
    body: Mapping[str, Any],
    *,
    state_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Upsert one lane. None means the control plane is not configured."""

    target, source = resolve_control_plane(state_path)
    if target is None:
        return None
    if not isinstance(body, Mapping):
        raise PortalTaskStateControlPlaneError("task state must be an object")
    if body.get("completion_authority") is True:
        raise PortalTaskStateControlPlaneError("task state cannot be completion authority")
    key = str(lane_key or "").strip() or "default"
    payload = dict(body)
    payload["completion_authority"] = False
    body_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    updated_at = datetime.now(UTC).isoformat()
    if _is_board_control_database(target):
        return _save_board_task_state(target, key, body_json, updated_at, source, state_path=state_path)
    with _state_connection(target, state_path=state_path) as connection:
        current = connection.execute(
            f"SELECT body_json, revision FROM {TABLE} WHERE lane_key = ?",
            [key],
        ).fetchone()
        if current is not None and str(current[0]) == body_json:
            revision = int(current[1])
            changed = False
        else:
            revision = 1 if current is None else int(current[1]) + 1
            connection.execute(
                f"""
                INSERT INTO {TABLE} (
                    lane_key, body_json, revision, completion_authority, updated_at
                ) VALUES (?, ?, ?, FALSE, ?)
                ON CONFLICT (lane_key) DO UPDATE SET
                    body_json = excluded.body_json,
                    revision = excluded.revision,
                    completion_authority = FALSE,
                    updated_at = excluded.updated_at
                """,
                [key, body_json, revision, updated_at],
            )
            changed = True
    return {
        "changed": changed,
        "lane_key": key,
        "revision": revision,
        "completion_authority": False,
        "ducklake": _state_ducklake_projection(
            state_path, target, key, body_json, revision, updated_at
        ),
        "json_authority": False,
        "source": source,
    }


def load_portal_task_state(
    lane_key: str,
    *,
    state_path: Path | str | None = None,
    create: bool = True,
) -> dict[str, Any] | None:
    """Return the lane body, {} if configured but empty, None if unconfigured."""

    target, _source = resolve_control_plane(state_path)
    if target is None:
        return None
    if not create:
        from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
            is_quack_transport_target,
        )

        if not is_quack_transport_target(target) and not Path(target).is_file():
            return {}
    key = str(lane_key or "").strip() or "default"
    if _is_board_control_database(target):
        return _load_board_task_state(target, key, create=create, state_path=state_path)
    with _state_connection(target, state_path=state_path) as connection:
        row = connection.execute(
            f"SELECT body_json FROM {TABLE} WHERE lane_key = ?",
            [key],
        ).fetchone()
    if row is None:
        if state_path is not None and task_state_connection_is_borrowed(state_path):
            raise PortalTaskStateControlPlaneError("borrowed_state_original_authoritative_row_required")
        return {}
    payload = json.loads(str(row[0]))
    if not isinstance(payload, dict):
        raise PortalTaskStateControlPlaneError("stored task state is not an object")
    payload["completion_authority"] = False
    return payload


def save_merge_handoff(
    record_key: str,
    body: Mapping[str, Any],
    *,
    target: str | None = None,
) -> dict[str, Any] | None:
    """Upsert a SPAR merge handoff. None means JSON files remain."""

    selected = target if target is not None else control_plane_target()
    if selected is None:
        return None
    if not isinstance(body, Mapping):
        raise PortalTaskStateControlPlaneError("merge handoff must be an object")
    if body.get("merged") is not False or body.get("writes_repository") is not False:
        raise PortalTaskStateControlPlaneError("merge handoff cannot claim a merge")
    if body.get("completion_authority") is True:
        raise PortalTaskStateControlPlaneError("merge handoff cannot be completion authority")
    key = str(record_key or "").strip()
    if not key:
        raise PortalTaskStateControlPlaneError("merge handoff key is required")
    payload = dict(body)
    payload["merged"] = False
    payload["writes_repository"] = False
    payload["completion_authority"] = False
    body_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    updated_at = datetime.now(UTC).isoformat()
    connection = _connect(selected)
    try:
        current = connection.execute(
            f"SELECT body_json, revision FROM {HANDOFF_TABLE} WHERE record_key = ?",
            [key],
        ).fetchone()
        if current is not None and str(current[0]) == body_json:
            revision = int(current[1])
            changed = False
        else:
            revision = 1 if current is None else int(current[1]) + 1
            connection.execute(
                f"""
                INSERT INTO {HANDOFF_TABLE} (
                    record_key, body_json, revision, completion_authority, updated_at
                ) VALUES (?, ?, ?, FALSE, ?)
                ON CONFLICT (record_key) DO UPDATE SET
                    body_json = excluded.body_json,
                    revision = excluded.revision,
                    completion_authority = FALSE,
                    updated_at = excluded.updated_at
                """,
                [key, body_json, revision, updated_at],
            )
            changed = True
    finally:
        connection.close()
    return {
        "changed": changed,
        "record_key": key,
        "revision": revision,
        "completion_authority": False,
        "ducklake": _project_ducklake(
            key,
            body_json,
            revision,
            updated_at,
            kind="spar_merge_handoff",
            target=selected,
        ),
        "json_authority": False,
    }


def load_merge_handoff(
    record_key: str,
    *,
    target: str | None = None,
) -> dict[str, Any] | None:
    """Return the handoff, {} if configured but absent, None if unconfigured."""

    selected = target if target is not None else control_plane_target()
    if selected is None:
        return None
    key = str(record_key or "").strip()
    if not key:
        return {}
    connection = _connect(selected)
    try:
        row = connection.execute(
            f"SELECT body_json FROM {HANDOFF_TABLE} WHERE record_key = ?",
            [key],
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return {}
    payload = json.loads(str(row[0]))
    if not isinstance(payload, dict):
        raise PortalTaskStateControlPlaneError("stored merge handoff is not an object")
    payload["completion_authority"] = False
    return payload
