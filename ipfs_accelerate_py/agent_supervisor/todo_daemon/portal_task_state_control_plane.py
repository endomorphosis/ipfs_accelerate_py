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


_BOUND_STATE_TARGETS: dict[str, str] = {}


def bind_task_state_control_plane(state_path: Path | str, target: str) -> None:
    """Remember the Quack or DuckDB target for one task-state path."""

    text = str(target or "").strip()
    if not text:
        return
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
) -> dict[str, Any]:
    updated_ms = int(datetime.now(UTC).timestamp() * 1000)
    connection = _connect(target)
    try:
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
    finally:
        connection.close()
    return {
        "changed": changed,
        "lane_key": key,
        "revision": updated_ms,
        "completion_authority": False,
        "ducklake": _project_ducklake(
            key, body_json, updated_ms, updated_at, target=target
        ),
        "json_authority": False,
        "source": source,
        "store": "board-control-plane",
    }


def _load_board_task_state(target: str, key: str, *, create: bool) -> dict[str, Any]:
    if not create and not Path(target).is_file():
        return {}
    connection = _connect(target)
    try:
        row = connection.execute(
            """
            SELECT payload_json FROM artefact_generic
            WHERE board_namespace = ? AND kind = ? AND artefact_id = ?
            """,
            [_BOARD_NAMESPACE, _BOARD_KIND, key],
        ).fetchone()
    finally:
        connection.close()
    if row is None:
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
        return _save_board_task_state(target, key, body_json, updated_at, source)
    connection = _connect(target)
    try:
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
    finally:
        connection.close()
    return {
        "changed": changed,
        "lane_key": key,
        "revision": revision,
        "completion_authority": False,
        "ducklake": _project_ducklake(
            key, body_json, revision, updated_at, target=target
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
        return _load_board_task_state(target, key, create=create)
    connection = _connect(target)
    try:
        row = connection.execute(
            f"SELECT body_json FROM {TABLE} WHERE lane_key = ?",
            [key],
        ).fetchone()
    finally:
        connection.close()
    if row is None:
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
