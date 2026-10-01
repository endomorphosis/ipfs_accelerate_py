"""Qualify the actual Quack listener and typed owner without model dispatch.

The disposable task is deliberately unfinished. This qualifies native owner
read/reservation/admission, not prompt planning, repair, or task completion.
No transport, capability report, migration, grant, or claim is mocked.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    InProcessQuackTransport,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import QuackStateClient
from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import StateOwnerBootstrapCredentials
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
    TypedDatabaseTaskSource,
    daemon_required_owner_command_operations,
    daemon_required_owner_operations,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA,
    TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon


@dataclass(repr=False)
class ExistingNativeOwnerSession:
    """Live resources, including private credentials; never serialize this object."""

    checkout: Path
    database: Path
    server: Any
    identity: Any
    client: QuackStateClient
    source: TypedDatabaseTaskSource
    credentials: StateOwnerBootstrapCredentials


@dataclass(repr=False)
class NativeOwnerSession(ExistingNativeOwnerSession):
    output: Path
    task_cid: str
    tree_id: str
    baseline_returncode: int


def _json(path: Path, value: Any) -> None:
    def plain(item):
        if isinstance(item, Mapping):
            return {key: plain(member) for key, member in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(member) for member in item]
        return item

    path.write_text(json.dumps(plain(value), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _prepare(output: Path) -> tuple[Path, Path, str, str, int]:
    output.mkdir(mode=0o700)
    checkout = output / "checkout"
    checkout.mkdir()
    (checkout / "value.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    (checkout / "validate.py").write_text(
        "from value import answer\nassert answer() == 2\n", encoding="utf-8"
    )
    for arguments in (
        ("init", "-q"), ("config", "user.email", "qualification@example.invalid"),
        ("config", "user.name", "Native Quack Qualification"), ("add", "."),
        ("commit", "-qm", "Authored incomplete qualification task"),
    ):
        subprocess.run(["git", "-C", str(checkout), *arguments], check=True, capture_output=True)
    tree_id = "git-tree:" + subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    baseline = subprocess.run(
        [sys.executable, "validate.py"], cwd=checkout, capture_output=True, text=True,
        timeout=10, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if baseline.returncode == 0:
        raise RuntimeError("qualification fixture must begin with failing acceptance")
    _json(output / "baseline-validation.json", {
        "argv": [sys.executable, "validate.py"], "cwd": str(checkout),
        "returncode": baseline.returncode, "repository_tree_id": tree_id,
        "stderr": baseline.stderr[-2048:], "acceptance_passed": False,
    })
    goal_cid = content_identity({"namespace": "native-quack-qualification-goal", "tree": tree_id})
    task_cid = content_identity({"namespace": "native-quack-qualification-task", "tree": tree_id})
    database = output / "control.duckdb"
    initial = DatabaseTaskSource(database)
    try:
        initial.materialize({
            "repository_tree_id": tree_id,
            "objectives": [{
                "objective_id": "objective:native-quack-qualification",
                "objective_alias": "NQQ-O001", "goal_cid": goal_cid,
                "goal_alias": "NQQ-G001", "title": "Qualify actual owner admission",
                "status": "open",
            }],
            "tasks": [{
                "task_cid": task_cid, "task_id": "NQQ-T001", "goal_cid": goal_cid,
                "status": "ready", "priority": "P0", "ordinal": 1,
                "title": "Repair answer to satisfy the authored acceptance test",
                "completion": "auto", "outputs": [{"path": "value.py"}],
                "acceptance": [{"criterion": "answer() returns two"}],
                "validations": [{"argv": [sys.executable, "validate.py"], "cwd": str(checkout)}],
            }],
        })
        assert initial.get_task(task_cid).status == "ready"
    finally:
        initial.close()
    return checkout, database, task_cid, tree_id, baseline.returncode


@contextmanager
def open_existing_native_owner(
    *, database: Path, checkout: Path, state_dir: Path,
    repository_id: str, execution_routes: Mapping[str, str],
    store_id: str = "native-quack-qualification-v1",
):
    """Open an existing admitted database without creating or updating tasks.

    Repository identity and routes belong to the caller's independent manifest.
    The native owner validates/seals the exact existing task route policy.
    Returned credentials belong only to this process; children need new grants.
    """
    database, checkout, state_dir = (Path(path).resolve() for path in (
        database, checkout, state_dir,
    ))
    if not database.is_file() or not (checkout / ".git").exists():
        raise ValueError("existing native owner requires a database and Git checkout")
    if not isinstance(repository_id, str) or not repository_id.strip():
        raise ValueError("existing native owner requires the manifest repository identity")
    if not isinstance(execution_routes, Mapping) or not execution_routes:
        raise ValueError("existing native owner requires explicit task routes")
    server = build_server(
        database_path=database, state_dir=state_dir,
        repository_id=repository_id, store_id=store_id,
    )
    client = None
    source = None
    saved_env = {name: os.environ.get(name) for name in (
        TYPED_STATE_OWNER_SOCKET_ENV, TYPED_STATE_OWNER_TOKEN_ENV,
    )}
    try:
        identity = server.start()
        if type(server.transport) is not InProcessQuackTransport:
            raise RuntimeError("qualification requires the actual native Quack transport")
        client_id = "database-implementation-daemon:native-quack-qualification"
        token, _grant = server.issue_typed_client_grant_record(
            client_id=client_id, process_birth_id=identity.process_birth_id,
            allowed_operations=daemon_required_owner_operations(),
            allowed_command_operations=daemon_required_owner_command_operations(),
            peer_pid=os.getpid(),
        )
        os.environ[TYPED_STATE_OWNER_SOCKET_ENV] = str(server.typed_command_socket_path())
        os.environ[TYPED_STATE_OWNER_TOKEN_ENV] = token
        client = QuackStateClient(
            owner_id=client_id, store_id=identity.store_id,
            process_birth_id=identity.process_birth_id,
        )
        client.attach(identity.listen_uri, server_id=identity.server_id)
        unsealed = TypedDatabaseTaskSource(client, owns_client=False)
        try:
            policy = unsealed.seal_execution_route_policy(dict(execution_routes))
        finally:
            unsealed.close()
        source = TypedDatabaseTaskSource(client, execution_route_policy=policy)
        credentials = StateOwnerBootstrapCredentials(
            endpoint=identity.listen_uri, socket_path=str(server.typed_command_socket_path()),
            store_id=identity.store_id, server_id=identity.server_id,
            client_id=client_id, process_birth_id=identity.process_birth_id,
            token=token, execution_route_policy=policy,
        )
        yield ExistingNativeOwnerSession(
            checkout, database, server, identity, client, source, credentials,
        )
    finally:
        try:
            if source is not None:
                source.close()
            elif client is not None:
                client.detach()
        finally:
            try:
                server.stop()
            finally:
                for name, value in saved_env.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value


@contextmanager
def native_owner_session(output: Path):
    """Materialize a disposable unfinished task and open its actual owner."""
    output = Path(output).resolve()
    checkout, database, task_cid, tree_id, baseline_returncode = _prepare(output)
    with open_existing_native_owner(
        database=database, checkout=checkout, state_dir=output / "owner",
        repository_id=content_identity({"repository_root": str(checkout)}),
        execution_routes={"NQQ-T001": GROK_CODEX_EXECUTION_MODE},
    ) as session:
        yield NativeOwnerSession(
            checkout=session.checkout, database=session.database,
            server=session.server, identity=session.identity, client=session.client,
            source=session.source, credentials=session.credentials,
            output=output, task_cid=task_cid, tree_id=tree_id,
            baseline_returncode=baseline_returncode,
        )


def qualify(output: Path) -> dict[str, Any]:
    started = time.monotonic()
    with native_owner_session(output) as session:
        # Distinct from the typed Unix-socket protocol: prove the actual Quack
        # TCP listener answers a read-only query from a new DuckDB connection.
        # The token stays in this owner process and is parameter-bound only.
        import duckdb

        remote = duckdb.connect()
        token = None
        try:
            remote.execute("SET autoinstall_known_extensions=false")
            remote.execute("SET autoload_known_extensions=false")
            remote.execute("LOAD httpfs")
            remote.execute("LOAD quack")
            token = session.server._vault.resolve(session.identity.secret_handle)
            observed_count = remote.execute(
                "SELECT * FROM quack_query(?, ?, token := ?, disable_ssl := true)",
                [session.identity.listen_uri, "SELECT COUNT(*) AS task_count FROM tasks", token],
            ).fetchone()
            token = None
            if observed_count is None or observed_count[0] != 1:
                raise RuntimeError("native Quack TCP task observation differs")
        except Exception as exc:
            # Do not export a driver exception that could include credentials.
            detail = str(exc)
            if token:
                detail = detail.replace(token, "[redacted]")
            raise RuntimeError(
                f"native Quack TCP probe failed ({type(exc).__name__}): {detail[:2048]}"
            ) from None
        finally:
            remote.close()
        daemon = DatabaseImplementationDaemon(
            database_path=session.database,
            coordination_path=session.output / "coordination.duckdb",
            execution_path=session.output / "execution.duckdb",
            authority_mode="quack", task_source_kind="duckdb",
            owner_session_id="session:native-quack-qualification",
            process_instance_id=session.identity.process_birth_id,
            quack_uri=session.identity.listen_uri,
            task_source=session.source, close_task_source=False,
            state_owner_bootstrap_credentials=session.credentials,
            strict_task_sharding=True, max_task_attempts=2,
            lease_ms=60_000, require_real_execution=True,
        ).open()
        try:
            before = session.source.get_task(session.task_cid)
            if before is None or before.status != "ready":
                raise RuntimeError("native owner did not expose the materialized ready task")
            attempt = daemon.claim_next()
            if attempt is None or attempt.task_cid != session.task_cid:
                raise RuntimeError("native owner did not admit the exact ready task")
            admitted = session.source.get_task(session.task_cid)
            receipt = dict(admitted.body.get("completion_receipt") or {})
            if (
                admitted.status != "in_progress"
                or receipt.get("operation") != "database_attempt_admitted"
                or receipt.get("claim_phase_schema") != TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
                or receipt.get("attempt_id") != attempt.attempt_id
                or receipt.get("claim_id") != attempt.claim_id
                or receipt.get("lease_id") != attempt.lease_id
                or admitted.revision != before.revision + 2
            ):
                raise RuntimeError("native reservation/admission differs from the exact claim")
            history = session.source.task_revision_diagnostic_window(
                session.task_cid, current_revision=admitted.revision
            )
            reservation = next(
                row for row in history["revisions"]
                if row["revision"] == admitted.revision - 1
            )
            reservation_receipt = reservation["body"]["completion_receipt"]
            if (
                reservation_receipt.get("claim_phase_schema") != TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA
                or reservation_receipt.get("operation") != "database_claim"
                or reservation_receipt.get("claim_id") != attempt.claim_id
            ):
                raise RuntimeError("native reservation evidence is missing")
            if daemon.provider_invocation_recorded(
                attempt.attempt_id, idempotency_key=f"provider:{attempt.attempt_id}"
            ) is not None:
                raise RuntimeError("qualification unexpectedly dispatched a provider")
            if daemon.claim_next() is not None:
                raise RuntimeError("native owner allowed a duplicate ready claim")
            ready = session.server.ready()
            result = {
                "schema": "native-quack-owner-qualification@1", "status": "passed",
                "transport": type(session.server.transport).__name__,
                "fake_transport": False, "native_owner_ready": ready.get("ready") is True,
                "quack_tcp_read_verified": True,
                "duckdb_version": duckdb.__version__,
                "extension_fingerprint": session.identity.extension_fingerprint,
                "typed_client_authenticated": True,
                "server_id": session.identity.server_id,
                "store_id": session.identity.store_id,
                "generation": session.identity.generation,
                "repository_tree_id": session.tree_id,
                "task_cid": session.task_cid, "attempt_id": attempt.attempt_id,
                "claim_id": attempt.claim_id, "lease_id": attempt.lease_id,
                "ready_revision": before.revision,
                "reservation_revision": reservation["revision"],
                "admission_revision": admitted.revision,
                "final_task_status": admitted.status,
                "failing_acceptance_returncode": session.baseline_returncode,
                "provider_calls": 0, "completion_authority_exercised": False,
                "duplicate_claim_refused": True,
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
            _json(session.output / "claim-history.json", history)
            _json(session.output / "qualification.json", result)
        finally:
            daemon.close()
    if session.server.status().get("lifecycle") != "stopped":
        raise RuntimeError("native owner did not stop after qualification")
    result["native_owner_stopped"] = True
    _json(Path(output) / "qualification.json", result)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: native_quack_qualification.py FRESH_OUTPUT_DIRECTORY")
    print(json.dumps(qualify(Path(sys.argv[1])), sort_keys=True))
