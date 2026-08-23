from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path
from types import ModuleType, SimpleNamespace

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[3]
OPERATOR_PATH = ROOT / "scripts/run_agent_supervisor_architecture_refactorer.py"


def _operator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pcar_bootstrap_operator", OPERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_owner_mutation_vocabulary_is_closed() -> None:
    operator = _operator()

    assert operator._normalized_owner_dml(" update tasks set status = ? ").startswith(
        "UPDATE "
    )
    with pytest.raises(operator.OperatorError, match="closed owner-DML"):
        operator._normalized_owner_dml("SELECT * FROM tasks")
    with pytest.raises(operator.OperatorError, match="exactly one SQL statement"):
        operator._normalized_owner_dml("UPDATE tasks SET status='ready'; DELETE FROM tasks")


def test_atomic_json_is_private_and_canonical(tmp_path: Path) -> None:
    operator = _operator()
    target = tmp_path / "receipt.json"

    operator._atomic_json(target, {"z": 2, "a": 1})

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "z": 2}
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_task_status_distinguishes_dependency_waiting_from_blocked() -> None:
    operator = _operator()
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(
            "CREATE TABLE tasks (task_cid VARCHAR, task_alias VARCHAR, ordinal BIGINT, "
            "status VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE task_dependencies (task_cid VARCHAR, dependency_task_cid VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE task_blocks (task_cid VARCHAR, state VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO tasks VALUES (?, ?, ?, ?)",
            [
                ("cid:000", "PCAR-000", 1, "todo"),
                ("cid:001", "PCAR-001", 2, "todo"),
                ("cid:002", "PCAR-002", 3, "blocked"),
            ],
        )
        connection.execute(
            "INSERT INTO task_dependencies VALUES ('cid:001', 'cid:000')"
        )

        observed = operator._task_status(connection)
    finally:
        connection.close()

    assert observed["dependency_ready_task_ids"] == ["PCAR-000"]
    assert observed["blocked_count"] == 1
    assert observed["active_task_ids"] == []
    assert observed["task_count"] == 3


def test_token_path_uses_only_the_opaque_handle(tmp_path: Path) -> None:
    operator = _operator()

    path = operator._token_path(tmp_path, "handle:pcar-v1")

    assert path == tmp_path / "handle_pcar-v1.quack-token"
    assert "token-value" not in str(path)


def test_state_owner_verifies_the_canonical_full_control_plane(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    operator = _operator()
    database = tmp_path / "control.duckdb"
    with DatabaseTaskSource(database, owner_id="pcar-full-schema-test"):
        pass

    report = operator._verify_control_plane(database)

    assert report.from_version == 1
    assert report.to_version == 1
    assert report.changed is False
    assert report.schema_fingerprint
    assert report.catalog_fingerprint


def test_managed_daemon_receives_source_checkout_environment(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_supervisor as supervisor_module,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import (
        SupervisorLoop,
    )

    todo_path = tmp_path / "tasks.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    args = supervisor_module.parse_args(
        [
            "--todo-path",
            str(todo_path),
            "--state-dir",
            str(tmp_path / "state"),
            "--worktree-root",
            str(tmp_path / "worktrees"),
        ]
    )
    supervisor = supervisor_module.PortalImplementationSupervisor(
        supervisor_module.supervisor_config_from_args(
            args,
            repo_root=supervisor_module.REPO_ROOT,
        )
    )

    loop_config = supervisor.build_supervisor_loop_config()
    expected_root = str(supervisor_module.REPO_ROOT)

    assert loop_config.child_env == loop_config.spec.launch_env
    assert expected_root in loop_config.child_env["PYTHONPATH"].split(os.pathsep)
    assert SupervisorLoop(loop_config)._child_spec("restart").env == (
        loop_config.child_env
    )


def test_quack_database_execution_paths_are_lane_local(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        parse_args,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
        resolve_database_implementation_paths,
    )

    state_dir = tmp_path / "lane-2"
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "quack",
            "--database-path",
            str(tmp_path / "shared-control.duckdb"),
            "--state-dir",
            str(state_dir),
        ]
    )

    paths = resolve_database_implementation_paths(args, authority_mode="quack")

    assert paths["database_path"] == state_dir / "quack-lane-control.duckdb"
    assert paths["database_path"] != tmp_path / "shared-control.duckdb"


def test_database_daemon_builder_preserves_attempt_and_shard_bounds(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        parse_args,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
        build_portal_implementation_daemon_from_args,
    )

    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            str(tmp_path / "control.duckdb"),
            "--state-dir",
            str(tmp_path / "lane-1"),
            "--max-task-attempts",
            "4",
            "--task-shard-count",
            "3",
            "--task-shard-index",
            "1",
            "--strict-task-sharding",
        ]
    )
    daemon, _context = build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert daemon.max_task_attempts == 4
        assert daemon.task_shard_count == 3
        assert daemon.task_shard_index == 1
        assert daemon.strict_task_sharding is True
    finally:
        daemon.close()


def test_database_daemon_strict_shard_excludes_other_alias_homes(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon,
    )

    tasks = [
        SimpleNamespace(
            task_cid=f"task:cid:{index}",
            task_alias=f"PCAR-{index:03d}",
            status="ready",
            revision=1,
            dependencies=(),
            body={},
        )
        for index in range(8)
    ]
    task_source = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            projection_cid="projection:strict-shard",
            task_count=len(tasks),
        ),
        list_tasks=lambda **_kwargs: SimpleNamespace(
            tasks=tasks,
            next_cursor="",
        ),
        ready_tasks=lambda **_kwargs: SimpleNamespace(tasks=tasks),
    )
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "lane-control.duckdb",
        authority_mode="embedded",
        task_source=task_source,
        coordinator=object(),
        install_schema=False,
        task_shard_count=3,
        task_shard_index=1,
        strict_task_sharding=True,
    )

    excluded = daemon._automatic_claim_exclusions()
    expected = {
        task.task_cid
        for task in tasks
        if int(
            hashlib.sha256(task.task_alias.encode("utf-8")).hexdigest()[:8],
            16,
        )
        % 3
        != 1
    }

    assert excluded == expected


def test_database_claim_releases_shared_cas_loser_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        TaskSourceConflictError,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon,
    )

    tasks = [
        SimpleNamespace(
            task_cid=f"task:cid:{index}",
            task_alias=f"PCAR-{index:03d}",
            status="ready",
            revision=1,
            dependencies=(),
            body={},
        )
        for index in range(2)
    ]

    class TaskSource:
        def snapshot(self) -> SimpleNamespace:
            return SimpleNamespace(
                projection_cid="projection:cas-loser",
                task_count=len(tasks),
            )

        def list_tasks(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(tasks=tasks, next_cursor="")

        def ready_tasks(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(tasks=tasks)

        def get(self, task_cid: str) -> SimpleNamespace | None:
            return next(
                (task for task in tasks if task.task_cid == task_cid),
                None,
            )

    class Coordinator:
        def __init__(self) -> None:
            self.released: list[tuple[str, str]] = []

        def register_task(self, **_kwargs: object) -> None:
            return None

        def synchronize_authoritative_task(self, **_kwargs: object) -> None:
            return None

        def coordination_registry_projection(self) -> dict[str, object]:
            return {
                "tasks": [
                    {
                        "task_cid": task.task_cid,
                        "task_id": task.task_alias,
                        "body": {
                            "authority": "task_source",
                            "authoritative_revision": task.revision,
                            "authoritative_status": task.status,
                        },
                    }
                    for task in tasks
                ]
            }

        def claim_ready_task(
            self,
            *,
            exclude_task_cids: set[str],
            **_kwargs: object,
        ) -> SimpleNamespace | None:
            candidate = next(
                (
                    task
                    for task in tasks
                    if task.task_cid not in exclude_task_cids
                ),
                None,
            )
            if candidate is None:
                return None
            index = tasks.index(candidate)
            return SimpleNamespace(
                task_cid=candidate.task_cid,
                claim_id=f"claim:{index}",
                attempt_id=f"attempt:{index}",
                attempt_number=1,
                owner_session_id="session:lane",
                lease_id=f"lease:{index}",
                fencing_token=1,
                fence_epoch=1,
                claimed_at_ms=1,
                worktree_id="",
            )

        def get_lease(self, lease_id: str) -> SimpleNamespace:
            return SimpleNamespace(lease_id=lease_id)

        def release(
            self,
            lease: SimpleNamespace,
            *,
            reason: str,
            **_kwargs: object,
        ) -> None:
            self.released.append((lease.lease_id, reason))

    coordinator = Coordinator()
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "lane.duckdb",
        authority_mode="embedded",
        task_source=TaskSource(),
        coordinator=coordinator,
        install_schema=False,
        require_real_execution=True,
    )
    cas_calls: list[str] = []

    def cas(task_cid: str, **_kwargs: object) -> None:
        cas_calls.append(task_cid)
        if task_cid == tasks[0].task_cid:
            raise TaskSourceConflictError("injected shared CAS loser")

    monkeypatch.setattr(daemon, "_cas_task_status_database", cas)
    monkeypatch.setattr(daemon, "_protect_new_claim", lambda _claim: None)
    monkeypatch.setattr(
        daemon,
        "_insert_attempt_from_claim",
        lambda claim, **_kwargs: SimpleNamespace(
            attempt_id=claim.attempt_id,
            task_cid=claim.task_cid,
            to_dict=lambda: {},
        ),
    )
    monkeypatch.setattr(daemon, "_record_event", lambda *_args, **_kwargs: None)

    attempt = daemon.claim_next()

    assert attempt is not None
    assert attempt.task_cid == tasks[1].task_cid
    assert cas_calls == [tasks[0].task_cid, tasks[1].task_cid]
    assert coordinator.released == [
        ("lease:0", "shared_board_claim_conflict")
    ]


def test_database_claim_rechecks_alias_home_after_local_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon,
    )

    initial = SimpleNamespace(
        task_cid="task:cid:003",
        task_alias="PCAR-003",
        status="ready",
        revision=1,
        dependencies=(),
        body={},
    )
    assert int(hashlib.sha256(b"PCAR-003").hexdigest()[:8], 16) % 3 == 2
    raced = SimpleNamespace(**{**vars(initial), "task_alias": "PCAR-002"})
    assert int(hashlib.sha256(b"PCAR-002").hexdigest()[:8], 16) % 3 == 0

    class TaskSource:
        population_reads = 0

        def snapshot(self) -> SimpleNamespace:
            return SimpleNamespace(
                projection_cid="projection:alias-race",
                task_count=1,
            )

        def list_tasks(self, **_kwargs: object) -> SimpleNamespace:
            self.population_reads += 1
            task = initial if self.population_reads <= 3 else raced
            return SimpleNamespace(tasks=[task], next_cursor="")

        def ready_tasks(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(tasks=[initial])

        def get(self, _task_cid: str) -> SimpleNamespace:
            return raced

    class Coordinator:
        released: list[str] = []
        claimed = False

        def register_task(self, **_kwargs: object) -> None:
            return None

        def synchronize_authoritative_task(self, **_kwargs: object) -> None:
            return None

        def coordination_registry_projection(self) -> dict[str, object]:
            return {
                "tasks": [
                    {
                        "task_cid": initial.task_cid,
                        "task_id": initial.task_alias,
                        "body": {
                            "authority": "task_source",
                            "authoritative_revision": initial.revision,
                            "authoritative_status": initial.status,
                        },
                    }
                ]
            }

        def claim_ready_task(self, **_kwargs: object) -> SimpleNamespace | None:
            if self.claimed:
                return None
            self.claimed = True
            return SimpleNamespace(
                task_cid=initial.task_cid,
                claim_id="claim:race",
                attempt_id="attempt:race",
                attempt_number=1,
                owner_session_id="session:lane-2",
                lease_id="lease:race",
                fencing_token=1,
                fence_epoch=1,
                claimed_at_ms=1,
                worktree_id="",
            )

        def get_lease(self, lease_id: str) -> SimpleNamespace:
            return SimpleNamespace(lease_id=lease_id)

        def release(
            self,
            lease: SimpleNamespace,
            *,
            reason: str,
            **_kwargs: object,
        ) -> None:
            self.released.append(f"{lease.lease_id}:{reason}")

    coordinator = Coordinator()
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "lane.duckdb",
        authority_mode="embedded",
        task_source=TaskSource(),
        coordinator=coordinator,
        install_schema=False,
        require_real_execution=True,
        task_shard_count=3,
        task_shard_index=2,
        strict_task_sharding=True,
    )
    monkeypatch.setattr(
        daemon,
        "_cas_task_status_database",
        lambda *_args, **_kwargs: pytest.fail("wrong-shard task reached shared CAS"),
    )

    assert daemon.claim_next() is None
    assert coordinator.released == [
        "lease:race:shared_board_task_out_of_strict_shard"
    ]
