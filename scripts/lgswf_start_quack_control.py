#!/usr/bin/env python3
"""Start the LGSWF DuckDB + Quack state-owner and print supervisor args.

DuckLake is not started here. It remains an optional non-authoritative
history projection downstream of this control plane.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/barberb/lift_coding/.worktrees/ipfs-accelerate-lgswf-actual-head")
DB = ROOT / "data/agent_supervisor/logic_governed_semantic_work_fabric/run-actual-v6/control.duckdb"
OWNER_STATE = ROOT / "data/agent_supervisor/logic_governed_semantic_work_fabric/run-actual-v6/quack-owner"
HANDLE = "handle:lgswf-actual-v6"


def main() -> int:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        build_server,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBConnection,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
        MigrationRunReport,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_datasets_authoritative_operational_schema,
    )

    def _verify_existing_control_plane(path):
        verification = verify_datasets_authoritative_operational_schema(path)
        if verification.get("valid") is not True:
            raise RuntimeError(
                "existing control plane failed datasets-authoritative verification"
            )
        fingerprint = str(verification.get("schema_fingerprint") or "")
        return MigrationRunReport(
            from_version=1,
            to_version=1,
            receipts=(),
            schema_fingerprint=fingerprint,
            catalog_fingerprint=fingerprint,
            changed=False,
        )

    def _owner_connection(path):
        import duckdb

        connection = duckdb.connect(str(path))
        connection.execute("LOAD quack")
        return DuckDBConnection.wrap(connection)

    class _LiveQuackTransport:
        def __init__(self) -> None:
            self._listen_uri = ""

        def start(self, connection, *, host, port, token, identity):
            from types import MappingProxyType
            from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
                listen_uri,
            )

            uri = listen_uri(host, port)
            connection.execute(
                "SELECT * FROM quack_serve(?, token := ?, "
                "allow_other_hostname := false, disable_ssl := true)",
                [uri, token],
            )
            self._listen_uri = uri
            return MappingProxyType(
                {
                    "server_id": identity.server_id,
                    "store_id": identity.store_id,
                    "database_uuid": identity.database_uuid,
                    "schema_revision": identity.schema_revision,
                    "schema_fingerprint": identity.schema_fingerprint,
                    "generation": identity.generation,
                    "process_birth_id": identity.process_birth_id,
                    "listen_uri": uri,
                }
            )

        def live_query(self, connection, *, identity, token):
            return {"listen_uri": self._listen_uri, "server_id": identity.server_id}

        def stop(self, connection=None) -> None:
            if connection is None:
                return
            try:
                connection.execute("SELECT quack_stop()")
            except Exception:
                pass

    OWNER_STATE.mkdir(parents=True, exist_ok=True)
    server = build_server(
        database_path=DB,
        state_dir=OWNER_STATE,
        host="127.0.0.1",
        port=41307,
        store_id="control.duckdb",
        secret_handle=HANDLE,
        allow_experimental=False,
        migrate=_verify_existing_control_plane,
        connection_factory=_owner_connection,
        transport=_LiveQuackTransport(),
    )
    identity = server.start()
    payload = identity.to_dict()
    status_path = OWNER_STATE / "quack-state-server.status.json"
    status_path.write_text(
        json.dumps(
            {"lifecycle": "ready", "identity": payload, "ready": True},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    sys.stdout.flush()
    import signal
    import time

    stop = {"value": False}

    def _handle(signum, _frame):
        del signum
        stop["value"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    control = server.stop_control_path()
    mutation_dir = OWNER_STATE / "mutations"
    mutation_dir.mkdir(parents=True, exist_ok=True)
    while server.lifecycle.value == "ready" and not stop["value"]:
        if control.is_file():
            break
        for request in sorted(mutation_dir.glob("*.request.json")):
            done = request.with_name(
                request.name.replace(".request.json", ".done.json")
            )
            try:
                payload = json.loads(request.read_text(encoding="utf-8"))
                sql = str(payload.get("sql") or "")
                parameters = payload.get("parameters")
                owner_conn = getattr(server, "_connection", None)
                if owner_conn is None:
                    raise RuntimeError("owner connection is not available")
                if parameters is None:
                    result = owner_conn.execute(sql)
                else:
                    result = owner_conn.execute(sql, parameters)
                rowcount = -1
                try:
                    description = getattr(result, "description", None)
                    if description:
                        result.fetchall()
                    elif hasattr(result, "rowcount"):
                        rowcount = int(result.rowcount)
                except Exception:
                    pass
                done.write_text(
                    json.dumps({"ok": True, "rowcount": rowcount}) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:
                done.write_text(
                    json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                    + "\n",
                    encoding="utf-8",
                )
            try:
                request.unlink()
            except OSError:
                pass
        time.sleep(0.05)
    result = server.stop()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
