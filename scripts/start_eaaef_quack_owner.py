#!/usr/bin/env python3
"""Start the EAAEF run-vN Quack state-owner on the admitted loopback port.

The sealed ``open_duckdb_connection`` policy denies dynamic extension bytes, so
the exclusive owner opens DuckDB, LOAD quack, then wraps the handle — the same
pattern as the legal-boards owner.  Does not mount a Docker socket.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[1]
DATA = (
    ROOT
    / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
)
CURSOR_PATH = DATA / "generation-cursor.json"
HOST = "127.0.0.1"
PORT = 19495
SECRET_HANDLE = "secret-handle:eaaef-quack-owner-v1"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the intentionally argument-free legacy launch contract.

    Parsing must remain the first operation in :func:`main`: ``--help`` and
    malformed invocations are operator inspection paths, not launch requests.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def _active_generation() -> str:
    generation = "eaaef-run-v14"
    if CURSOR_PATH.is_file():
        try:
            cursor = json.loads(CURSOR_PATH.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            cursor = {}
        active = str((cursor or {}).get("active_generation") or "").strip()
        if active.startswith("eaaef-run-v"):
            generation = active
    return generation


def main(argv: list[str] | None = None) -> int:
    _parse_args(argv)
    generation = _active_generation()
    run_dir = DATA / generation.removeprefix("eaaef-")
    database = run_dir / "control.duckdb"
    state_dir = run_dir / "live/state/quack-owner"
    identity_path = state_dir / "owner-identity.json"
    mutation_dir = state_dir / "mutations"
    store_id = f"eaaef-control-{generation.removeprefix('eaaef-')}"

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    state_dir.mkdir(parents=True, exist_ok=True)
    mutation_dir.mkdir(parents=True, exist_ok=True)

    import duckdb
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        InProcessQuackTransport,
        build_server,
        listen_uri,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
        MigrationDowngradeError,
        MigrationRunReport,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        install_control_plane_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBConnection,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
        probe_quack_capabilities,
    )
    from ipfs_accelerate_py.agent_supervisor.validation.control_plane_identity_recovery import (
        OWNER_IDENTITY_RECOVERY_SQL,
    )

    def _probe():
        return probe_quack_capabilities(
            allow_local_load=True,
            allow_network_install=False,
        )

    def _migrate(path: Path) -> MigrationRunReport:
        raw = duckdb.connect(str(path), read_only=True)
        try:
            def _meta(key: str) -> str:
                row = raw.execute(
                    "SELECT value FROM control_plane_metadata WHERE key = ?",
                    [key],
                ).fetchone()
                return str(row[0] if row else "")

            version = int(_meta("schema_version") or 0)
            schema_fp = _meta("schema_fingerprint")
            catalog_fp = _meta("catalog_fingerprint")
        finally:
            raw.close()
        if version >= 2 and schema_fp:
            return MigrationRunReport(
                from_version=version,
                to_version=version,
                receipts=(),
                schema_fingerprint=schema_fp,
                catalog_fingerprint=catalog_fp,
                changed=False,
            )
        try:
            return install_control_plane_schema(
                path,
                application_version="0.0.45",
                tool_version="1.5.5",
                owner_id="quack-state-server:eaaef",
            )
        except MigrationDowngradeError:
            return MigrationRunReport(
                from_version=version,
                to_version=version,
                receipts=(),
                schema_fingerprint=schema_fp,
                catalog_fingerprint=catalog_fp,
                changed=False,
            )

    def _connection(path: Path) -> DuckDBConnection:
        raw = duckdb.connect(str(path))
        raw.execute("LOAD quack")
        try:
            raw.execute("LOAD ducklake")
        except Exception:
            pass
        return DuckDBConnection.wrap(raw)

    class _OwnerTransport(InProcessQuackTransport):
        def start(self, connection, *, host, port, token, identity):
            uri = listen_uri(host, port)
            last_error = None
            attempts = (
                (
                    "SELECT * FROM quack_serve(?, token := ?, "
                    "allow_other_hostname := false, disable_ssl := true)",
                    [uri, token],
                ),
                ("SELECT quack_serve(?, ?)", [uri, token]),
                ("SELECT quack_serve(?, ?, ?)", [host, int(port), token]),
            )
            for sql, params in attempts:
                try:
                    connection.execute(sql, params)
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
            if last_error is not None:
                raise last_error
            self._started = True
            self._listen_uri = uri
            self._server_identity = {
                "server_id": identity.server_id,
                "store_id": identity.store_id,
                "database_uuid": identity.database_uuid,
                "schema_revision": identity.schema_revision,
                "schema_fingerprint": identity.schema_fingerprint,
                "generation": identity.generation,
                "process_birth_id": identity.process_birth_id,
                "listen_uri": uri,
            }
            return MappingProxyType(dict(self._server_identity))

    server = build_server(
        database_path=database,
        state_dir=state_dir,
        host=HOST,
        port=PORT,
        repository_id="external-agent-autonomous-execution-fabric-v1",
        store_id=store_id,
        secret_handle=SECRET_HANDLE,
        transport=_OwnerTransport(),
        capability_probe=_probe,
        migrate=_migrate,
        connection_factory=_connection,
    )
    identity = server.start()
    payload = identity.to_dict() if hasattr(identity, "to_dict") else dict(identity)
    identity_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    identity_path.chmod(0o600)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)

    stop_requested = {"value": False}

    def _handle(_signum, _frame):
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    control_path = server.stop_control_path()
    os.environ["IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR"] = str(mutation_dir)

    def _apply_owner_mutations() -> None:
        owner_conn = getattr(server, "_connection", None)
        if owner_conn is None:
            return
        inner = getattr(owner_conn, "_connection", owner_conn)
        for request in sorted(mutation_dir.glob("*.request.json")):
            done = request.with_name(request.name.replace(".request.json", ".done.json"))
            try:
                payload = json.loads(request.read_text(encoding="utf-8"))
                sql = " ".join(str(payload.get("sql") or "").split())
                parameters = payload.get("parameters")
                if sql not in OWNER_IDENTITY_RECOVERY_SQL:
                    raise RuntimeError("owner mutation SQL is not allowlisted")
                if parameters is None:
                    result = inner.execute(sql)
                else:
                    result = inner.execute(sql, parameters)
                rowcount = 0
                try:
                    rows = result.fetchall()
                    if rows:
                        try:
                            rowcount = int(rows[0][0])
                        except (TypeError, ValueError):
                            rowcount = len(rows)
                except Exception:
                    rowcount = 0
                done.write_text(
                    json.dumps({"ok": True, "rowcount": rowcount}) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:
                done.write_text(
                    json.dumps(
                        {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    )
                    + "\n",
                    encoding="utf-8",
                )
            try:
                request.unlink()
            except OSError:
                pass

    while server.lifecycle.value == "ready" and not stop_requested["value"]:
        if control_path.is_file():
            break
        _apply_owner_mutations()
        time.sleep(0.05)
    result = server.stop()
    print(json.dumps(result if isinstance(result, dict) else {"stopped": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
