"""Local DuckLake history fed by the authenticated DuckDB + Quack control plane.

One bounded exporter owns this observational catalog. Supervisors continue to
use their native Quack owners; unavailable history never blocks their writes.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .fleet_history import project_view, read_native_view

SCHEMA = "ipfs_accelerate_py/ducklake-fleet-history@1"


def _literal(value: str) -> str:
    if "\x00" in value:
        raise ValueError("invalid DuckLake path")
    return "'" + value.replace("'", "''") + "'"


@contextmanager
def open_history(root: Path, *, create: bool = True):
    """Serialize local catalog access without ever deleting a lock or WAL.

    The private history directory is dedicated to this exporter. All callers,
    including inspection, take the same nonblocking lifetime lock before LOAD
    or ATTACH. Process exit releases it; the next timer retries by observation CID.
    """
    if not root.is_absolute():
        raise ValueError("absolute dedicated DuckLake history directory required")
    if create:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = root.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o077):
        raise ValueError("DuckLake history directory must be private and owned")
    catalog, data = root / "metadata.ducklake", root / "parquet"
    if catalog.is_symlink() or data.is_symlink():
        raise ValueError("DuckLake paths must not be symlinks")
    if not create and not catalog.is_file():
        raise FileNotFoundError("DuckLake history has not been initialized")
    fd = os.open(root / "history.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    connection = None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("DuckLake lock must be private, regular and owned")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import duckdb

        connection = duckdb.connect(":memory:")
        connection.execute("SET threads=1")
        connection.execute("SET memory_limit='256MB'")
        connection.execute("SET autoinstall_known_extensions=false")
        connection.execute("SET autoload_known_extensions=false")
        connection.execute("LOAD ducklake")
        options = "DATA_PATH " + _literal(str(data)) + ", DATA_INLINING_ROW_LIMIT 0"
        if not create:
            options += ", READ_ONLY"
        connection.execute("ATTACH " + _literal("ducklake:" + str(catalog)) + " AS fleet_lake (" + options + ")")
        yield connection
    finally:
        try:
            if connection is not None:
                connection.close()
        finally:
            os.close(fd)


def inspect_history(connection) -> dict:
    rows = connection.execute("""
        SELECT source_id, count(*), max(observed_at)
        FROM fleet_lake.fleet_source_observations
        GROUP BY source_id ORDER BY source_id
    """).fetchall()
    snapshot = connection.execute("SELECT max(snapshot_id) FROM ducklake_snapshots('fleet_lake')").fetchone()[0]
    return {"snapshot_id": snapshot, "stored_observations": sum(row[1] for row in rows),
            "sources": {row[0]: {"stored_observations": row[1], "latest_observed_at": row[2]} for row in rows}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inspect", action="store_true", help="Read history under the same catalog lock")
    parser.add_argument("--status-file", type=Path, help="Atomic diagnostic output; never an authority input")
    args = parser.parse_args()
    result = {"schema": SCHEMA, "completion_authority": False,
              "observed_at": datetime.now(timezone.utc).isoformat()}
    code = 0
    try:
        if args.inspect:
            with open_history(args.history_root, create=False) as connection:
                result.update(inspect_history(connection), status="inspected")
        else:
            if args.deployment is None or args.inventory is None:
                raise ValueError("native deployment and inventory are required for export")
            # No local JSON observation fallback when the native owner is denied.
            view = read_native_view(args.deployment, args.inventory)
            with open_history(args.history_root) as connection:
                count = project_view(connection, view)
                result.update(inspect_history(connection), status="projected", source_observations=count)
    except Exception as error:
        result.update(status="unavailable", error_type=type(error).__name__)
        code = 1
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.status_file:
        # Unit execution is serialized; PID suffix also separates manual probes.
        temporary = args.status_file.with_name(args.status_file.name + f".{os.getpid()}.tmp")
        temporary.write_text(encoded)
        os.replace(temporary, args.status_file)
    print(encoded, end="")
    return code
