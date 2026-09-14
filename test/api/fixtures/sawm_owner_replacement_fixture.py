"""Actual native owner/daemon storage methods with a fake extension transport.

All files and credentials belong to a newly created disposable test directory.
This qualifies native storage closure/reuse, not sealed extension admission.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, default=str))


def rows(connection):
    tables = [str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()]
    return {
        name: sorted(
            [
                list(row)
                for row in connection.execute(
                    'SELECT * FROM "' + name.replace('"', '""') + '"'
                ).fetchall()
            ],
            key=repr,
        )
        for name in tables
    }


def writer_held(database):
    value = database.stat()
    wanted = (os.major(value.st_dev), os.minor(value.st_dev), value.st_ino)
    for line in Path("/proc/locks").read_text().splitlines():
        parts = line.split()
        if len(parts) == 8 and parts[1:5] == [
            "POSIX",
            "ADVISORY",
            "WRITE",
            str(os.getpid()),
        ]:
            a, b, c = parts[5].split(":")
            if (int(a, 16), int(b, 16), int(c)) == wanted:
                return True
    return False


def prepare(root):
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "README").write_text("Disposable native recovery qualification.\n")
    subprocess.run(["git", "-C", str(root), "add", "README"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "disposable",
        ],
        check=True,
    )
    from test.api.test_agent_supervisor_database_implementation_daemon import (
        _open_daemon,
        _population,
    )

    def unknown(_):
        raise RuntimeError("disposable unreturned native callback")

    daemon = _open_daemon(
        root, session="session:unresolved-native", provider_fn=unknown, lease_ms=3600000
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        try:
            daemon.run_provider(attempt)
        except RuntimeError:
            pass
        else:
            raise AssertionError("callback did not interrupt")
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert claim.state.value == "accepted"
        assert daemon.get_attempt(attempt.attempt_id).status == "running"
        assert (
            daemon.provider_invocation_recorded(
                attempt.attempt_id, idempotency_key="provider:" + attempt.attempt_id
            )
            is None
        )
        write(
            root / "retained.json",
            {
                "attempt_id": attempt.attempt_id,
                "claim_id": claim.claim_id,
                "execution": rows(daemon._connection),
                "coordination": rows(daemon.coordinator._connection),
            },
        )
    finally:
        daemon.close()
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.worktrees import WorktreePool

    pool = WorktreePool(repo_root=root, worktree_root=root / "worktrees")
    lease = pool.acquire(cache_key="unknown-callback", branch_name="retained-candidate")
    (lease.path / "unsettled.txt").write_text(
        "Unresolved callback candidate must be retained.\n"
    )
    write(root / "pool-before.json", pool_snapshot(root))


def pool_snapshot(root):
    return {
        str(p.relative_to(root)): [
            p.stat().st_ino,
            p.stat().st_mode,
            hashlib.sha256(p.read_bytes()).hexdigest(),
        ]
        for base in [root / "worktrees", root / ".git/worktrees"]
        if base.exists()
        for p in base.rglob("*")
        if p.is_file() and not p.is_symlink()
    }


def wrapper(root):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
        PortalSupervisorConfig,
    )

    state = root / "portal-state"
    state.mkdir()
    (root / "todo.md").write_text("# Disposable SAWM\n")
    task_state = state / "task_state.json"
    task_state.write_text('{"implementation_in_progress":false,"active_task_id":""}\n')
    before = task_state.read_bytes()
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=root / "todo.md",
            state_path=task_state,
            strategy_path=state / "strategy.json",
            events_path=state / "supervisor_events.jsonl",
            state_dir=state,
            repo_root=root,
            state_prefix="sawm_pause",
            worktree_root=root / "worktrees",
        )
    )

    def idle():
        (root / "wrapper-ready").touch()
        while True:
            time.sleep(0.1)

    supervisor._run_forever_loop = idle
    try:
        supervisor.run_forever()
    except SystemExit as e:
        assert e.code == 143
        assert task_state.read_bytes() == before
        assert pool_snapshot(root) == json.loads(
            (root / "pool-before.json").read_text()
        )
        (root / "wrapper-closed").touch()


def owner(root, source, reuse):
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        FakeQuackTransport,
        build_server,
    )
    from test.api.test_agent_supervisor_quack_state_server import _compatible_report

    spec = importlib.util.spec_from_file_location(
        "disposable_sawm_operator",
        source / "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
    )
    operator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = operator
    spec.loader.exec_module(operator)
    expected = json.loads((root / "owner-before.json").read_text()) if reuse else None
    server = build_server(
        database_path=root / "control.duckdb",
        state_dir=root / "owner",
        store_id="disposable-sawm",
        repository_id="repository:disposable-sawm",
        secret_handle="env://DISPOSABLE_SAWM_TOKEN",
        transport=FakeQuackTransport(),
        capability_probe=lambda **_: _compatible_report(),
        expected_generation=expected["generation"] if expected else None,
        expected_database_uuid=expected["database_uuid"] if expected else None,
        reuse_expected_generation=reuse,
    )
    try:
        identity = server.start()
        assert writer_held(root / "control.duckdb")
        with server._lock:
            protected = {
                name: value
                for name, value in rows(server._connection).items()
                if name in {"tasks", "task_revisions", "domain_events"}
            }
            server._connection.execute("CHECKPOINT")
            # Reproduce the actual old25f source copier's canonical-close bug.
            operator._SawmQuackTransport._copy_replica(
                root / "control.duckdb", root / "replica.duckdb"
            )
        held = writer_held(root / "control.duckdb")
        assert held is reuse, (
            "old source loses writer fence; successor copier must retain it"
        )
        value = {
            "generation": identity.generation,
            "database_uuid": identity.database_uuid,
            "server_id": identity.server_id,
            "protected": protected,
            "writer_held_after_copy": held,
            "pid": os.getpid(),
            "identity": identity.to_dict(),
        }
        write(root / ("owner-after.json" if reuse else "owner-before.json"), value)
        operator._serve_sawm_owner(server)
        assert not writer_held(root / "control.duckdb")
        write(
            root / ("closed-after.json" if reuse else "closed-before.json"),
            {"closed": True, "canonical_writer_released": True},
        )
    finally:
        server.stop()


def verify(root):
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_duckdb_connection,
    )

    retained = json.loads((root / "retained.json").read_text())
    result = {}
    for key, name in [
        ("execution", "execution.duckdb"),
        ("coordination", "coordination.duckdb"),
    ]:
        with open_duckdb_connection(root / name) as connection:
            result[key] = rows(connection)
    # JSON normalizes native tuple/datetime row wrappers on both sides.
    result = json.loads(json.dumps(result, default=str))
    assert result["execution"] == retained["execution"]
    assert result["coordination"] == retained["coordination"]
    write(
        root / "verified.json", {"native_unresolved_attempt_and_claim_unchanged": True}
    )


if __name__ == "__main__":
    source, root, action = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    sys.path.insert(0, str(source))
    if action == "owner":
        owner(root, source, sys.argv[4] == "reuse")
    else:
        globals()[action](root)
