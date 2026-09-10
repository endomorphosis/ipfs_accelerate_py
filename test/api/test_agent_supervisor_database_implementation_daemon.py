"""Tests for DatabaseImplementationDaemon@1 cutover (DQP-018).

Evidence subset: ready selection, strict shards, lost response, provider
capacity, hard quota, timeout, cancellation, crash, restart, stale worker,
status parity.

Acceptance: Four daemon processes claim distinct work; no task status is
updated in Markdown under database authority; JSON queue/status/events/PID
projections can be absent; crash/restart resumes from committed phase and does
not duplicate provider/effect work.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Callable

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinationError,
    DatabaseCoordinationConflictError,
    LeaseState,
    ProcessSerializedDatabaseCoordinator,
    open_database_coordinator,
    open_process_serialized_database_coordinator,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    ProcessBirthIdentity,
    WorktreeLifecycleStore,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
    TaskRecord,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
    install_datasets_authoritative_operational_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    exclusive_file_lock,
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    database_portal_bridge as database_portal_bridge_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ATTEMPT_PHASE_COMPLETE,
    ATTEMPT_PHASE_EFFECT,
    ATTEMPT_PHASE_PROVIDER,
    DATABASE_CONTROL_CLAIM_BINDING_SCHEMA,
    DATABASE_CONTROL_CLAIM_BINDING_SCHEMA_V1,
    DATABASE_DAEMON_PASS_HEARTBEAT_SCHEMA,
    DATABASE_IMPLEMENTATION_DAEMON_INTERFACE,
    DATABASE_TASK_ATTEMPT_INTERFACE,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    PortalImplementationDaemon,
    PortalTaskState,
    database_daemon_pass_heartbeat_path,
    is_database_authority_mode,
    open_database_implementation_daemon,
    parse_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
    DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1,
    DatabasePortalBridgeDeferred,
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_database_implementation_daemon_from_args,
    build_portal_implementation_daemon_from_args,
    resolve_database_implementation_paths,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for database implementation daemon tests",
)


def _population(task_count: int = 4) -> dict[str, object]:
    tasks = []
    for index in range(1, task_count + 1):
        tasks.append(
            {
                "task_cid": f"task:cid:{index:03d}",
                "task_id": f"DQP-T{index:03d}",
                "goal_cid": "goal:cid:root",
                "status": "ready",
                "priority": "P0",
                "ordinal": index,
                "title": f"Task {index}",
            }
        )
    return {
        "repository_tree_id": "tree:dqp-018",
        "objectives": [
            {
                "objective_id": "objective:dqp-018",
                "objective_alias": "DQP-O018",
                "title": "Daemon cutover",
                "goal_cid": "goal:cid:root",
                "goal_alias": "DQP-G030",
                "status": "open",
            }
        ],
        "tasks": tasks,
    }


def _open_daemon(
    tmp_path: Path,
    *,
    session: str = "",
    provider_calls: list[str] | None = None,
    effect_calls: list[str] | None = None,
    markdown_path: Path | None = None,
    provider_fn: Callable[[DatabaseTaskAttempt], dict[str, object]] | None = None,
    lease_ms: int = 60_000,
    clock_ms: Callable[[], int] | None = None,
    task_shard_count: int = 1,
    task_shard_index: int = 0,
    strict_task_sharding: bool = False,
    task_prefix: str = "",
) -> DatabaseImplementationDaemon:
    database_path = tmp_path / "control.duckdb"
    coordination_path = tmp_path / "coordination.duckdb"
    execution_path = (
        tmp_path / "execution.duckdb" if task_shard_count == 1 else None
    )
    state_dir = (
        tmp_path / "state" / f"lane-{task_shard_index}"
        if task_shard_count > 1
        else None
    )
    state_prefix = (
        f"dqp-lane-{task_shard_index}" if task_shard_count > 1 else ""
    )

    def default_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        if provider_calls is not None:
            provider_calls.append(attempt.task_cid)
        return {"status": "ok", "task_cid": attempt.task_cid}

    def effect(
        attempt: DatabaseTaskAttempt, provider_result: dict[str, object]
    ) -> dict[str, object]:
        if effect_calls is not None:
            effect_calls.append(attempt.task_cid)
        return {
            "status": "applied",
            "task_cid": attempt.task_cid,
            "provider_result": dict(provider_result),
        }

    return DatabaseImplementationDaemon(
        database_path=database_path,
        coordination_path=coordination_path,
        execution_path=execution_path,
        state_dir=state_dir,
        state_prefix=state_prefix,
        owner_session_id=session,
        authority_mode="embedded",
        task_source_kind="duckdb",
        markdown_path=markdown_path,
        # Projections intentionally absent.
        state_path=None,
        strategy_path=None,
        events_path=None,
        pid_path=None,
        queue_path=None,
        lease_ms=lease_ms,
        provider_fn=provider_fn or default_provider,
        effect_fn=effect,
        clock_ms=clock_ms,
        task_shard_count=task_shard_count,
        task_shard_index=task_shard_index,
        strict_task_sharding=strict_task_sharding,
        task_prefix=task_prefix,
    )


def _authority_task_record(*, revision: int) -> TaskRecord:
    return TaskRecord(
        task_cid="task:cid:authority",
        task_alias="AUTH-001",
        goal_cid="goal:cid:authority",
        plan_cid="plan:cid:authority",
        ordinal=1,
        status="in_progress",
        revision=revision,
        priority="P0",
        body={
            "objective": "Recover one exact superseded attempt",
            "completion": "auto",
            "track": "implementation",
        },
    )


def _legacy_control_binding(
    control: dict[str, object],
) -> dict[str, object]:
    payload = dict(control)
    payload.pop("binding_id")
    payload.pop("database_portal_binding_basis")
    payload.pop("database_portal_binding_basis_cid")
    payload["schema"] = DATABASE_CONTROL_CLAIM_BINDING_SCHEMA_V1
    payload["binding_id"] = content_identity(payload)
    return payload


def _rehash_control_binding(
    control: dict[str, object],
) -> dict[str, object]:
    payload = dict(control)
    payload.pop("binding_id", None)
    payload["binding_id"] = content_identity(payload)
    return payload


def _rehash_portal_binding(
    binding: dict[str, object],
) -> dict[str, object]:
    payload = dict(binding)
    payload.pop("binding_id", None)
    payload["binding_id"] = database_portal_bridge_module._sha256_bytes(
        database_portal_bridge_module._canonical_json(payload)
    )
    return payload


def _authority_bound_attempt(
    bridge: DatabasePortalExecutionBridge,
    *,
    attempt_id: str,
    claim_id: str,
    attempt_number: int,
    owner_session_id: str,
    fencing_token: int,
    fence_epoch: int,
    lease_id: str,
    revision: int,
    status: str,
    control_schema: str,
    portal_schema: str = "",
) -> tuple[DatabaseTaskAttempt, dict[str, object], dict[str, object]]:
    attempt = DatabaseTaskAttempt(
        attempt_id=attempt_id,
        claim_id=claim_id,
        task_cid="task:cid:authority",
        task_alias="AUTH-001",
        attempt_number=attempt_number,
        owner_session_id=owner_session_id,
        fencing_token=fencing_token,
        fence_epoch=fence_epoch,
        lease_id=lease_id,
        committed_phase="claimed" if status == "running" else status,
        status=status,
        started_at_ms=1,
    )
    record = _authority_task_record(revision=revision)
    claim = SimpleNamespace(
        task_cid=attempt.task_cid,
        claim_id=attempt.claim_id,
        attempt_id=attempt.attempt_id,
        attempt_number=attempt.attempt_number,
        lease_id=attempt.lease_id,
        owner_session_id=attempt.owner_session_id,
        fencing_token=attempt.fencing_token,
        fence_epoch=attempt.fence_epoch,
    )
    control = DatabaseImplementationDaemon._control_claim_binding(claim, record)
    if control_schema == DATABASE_CONTROL_CLAIM_BINDING_SCHEMA_V1:
        control = _legacy_control_binding(control)
    else:
        assert control_schema == DATABASE_CONTROL_CLAIM_BINDING_SCHEMA
    attempt = replace(attempt, body={"control_binding": control})
    selected_portal_schema = portal_schema or (
        DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA
        if control_schema == DATABASE_CONTROL_CLAIM_BINDING_SCHEMA
        else DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1
    )
    projection = bridge._render_projection(attempt, record)
    portal = bridge._binding(
        attempt,
        record,
        projection,
        schema=selected_portal_schema,
    )
    return attempt, control, portal


class _SupersededAuthorityCoordinator:
    def __init__(self, prior: DatabaseTaskAttempt) -> None:
        self.prior_claim = SimpleNamespace(
            claim_id=prior.claim_id,
            task_cid=prior.task_cid,
            attempt_id=prior.attempt_id,
            attempt_number=prior.attempt_number,
            owner_session_id=prior.owner_session_id,
            fencing_token=prior.fencing_token,
            fence_epoch=prior.fence_epoch,
            lease_id=prior.lease_id,
            state=SimpleNamespace(value="expired"),
        )
        self.prior_attempt = SimpleNamespace(
            attempt_id=prior.attempt_id,
            task_cid=prior.task_cid,
            attempt_number=prior.attempt_number,
            owner_session_id=prior.owner_session_id,
            fencing_token=prior.fencing_token,
            fence_epoch=prior.fence_epoch,
            status=SimpleNamespace(value="failed"),
        )

    def get_task_claim(self, _claim_id: str) -> object:
        return self.prior_claim

    def get_task_attempt(self, _attempt_id: str) -> object:
        return self.prior_attempt


def _superseded_authority_fake(
    current: DatabaseTaskAttempt,
    prior: DatabaseTaskAttempt,
) -> tuple[
    DatabaseImplementationDaemon,
    _SupersededAuthorityCoordinator,
    list[str],
]:
    daemon = object.__new__(DatabaseImplementationDaemon)
    attempts = {
        current.attempt_id: current,
        prior.attempt_id: prior,
    }
    protected: list[str] = []
    coordinator = _SupersededAuthorityCoordinator(prior)
    daemon.get_attempt = lambda attempt_id: attempts.get(attempt_id)  # type: ignore[method-assign]
    daemon._protect_attempt_write = (  # type: ignore[method-assign]
        lambda attempt: protected.append(attempt.attempt_id)
    )
    daemon.open = lambda: daemon  # type: ignore[method-assign]
    daemon._coordinator = coordinator
    return daemon, coordinator, protected


def _superseded_authority_fixture(
    tmp_path: Path,
    *,
    current_control_schema: str = DATABASE_CONTROL_CLAIM_BINDING_SCHEMA,
    prior_control_schema: str = DATABASE_CONTROL_CLAIM_BINDING_SCHEMA,
    current_portal_schema: str = DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
    prior_portal_schema: str = DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
    current_revision: int = 11,
    prior_revision: int = 11,
) -> SimpleNamespace:
    bridge = DatabasePortalExecutionBridge(
        task_source=SimpleNamespace(),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: object(),
    )
    current, current_control, current_portal = _authority_bound_attempt(
        bridge,
        attempt_id="attempt:current",
        claim_id="claim:current",
        attempt_number=5,
        owner_session_id="session:current",
        fencing_token=7,
        fence_epoch=3,
        lease_id="lease:current",
        revision=current_revision,
        status="running",
        control_schema=current_control_schema,
        portal_schema=current_portal_schema,
    )
    prior, prior_control, prior_portal = _authority_bound_attempt(
        bridge,
        attempt_id="attempt:prior",
        claim_id="claim:prior",
        attempt_number=4,
        owner_session_id="session:prior",
        fencing_token=6,
        fence_epoch=2,
        lease_id="lease:prior",
        revision=prior_revision,
        status="failed",
        control_schema=prior_control_schema,
        portal_schema=prior_portal_schema,
    )
    daemon, coordinator, protected = _superseded_authority_fake(current, prior)
    return SimpleNamespace(
        bridge=bridge,
        current=current,
        prior=prior,
        current_control=current_control,
        prior_control=prior_control,
        current_portal=current_portal,
        prior_portal=prior_portal,
        daemon=daemon,
        coordinator=coordinator,
        protected=protected,
    )


def test_interface_identities() -> None:
    assert DATABASE_IMPLEMENTATION_DAEMON_INTERFACE == (
        "DatabaseImplementationDaemon@1"
    )
    assert DATABASE_TASK_ATTEMPT_INTERFACE == "DatabaseTaskAttempt@1"
    assert DatabaseImplementationDaemon.INTERFACE == (
        DATABASE_IMPLEMENTATION_DAEMON_INTERFACE
    )
    assert DatabaseTaskAttempt.INTERFACE == DATABASE_TASK_ATTEMPT_INTERFACE
    assert is_database_authority_mode(authority_mode="embedded")
    assert is_database_authority_mode(task_source_kind="duckdb")
    assert not is_database_authority_mode(
        authority_mode="legacy_markdown", task_source_kind="legacy-markdown"
    )


def test_four_daemon_processes_claim_distinct_work(tmp_path: Path) -> None:
    markdown = tmp_path / "board.md"
    markdown.write_text(
        "# Board\n\n## DQP-T001 Sample\n\n- Status: todo\n",
        encoding="utf-8",
    )
    original_markdown = markdown.read_text(encoding="utf-8")

    seed = _open_daemon(tmp_path, session="session:seed", markdown_path=markdown)
    try:
        seed.materialize_population(_population(4))
    finally:
        seed.close()

    claimed: list[str] = []
    for index in range(1, 5):
        daemon = _open_daemon(
            tmp_path,
            session=f"session:{index}",
            markdown_path=markdown,
        )
        try:
            attempt = daemon.claim_next()
            assert attempt is not None, f"session {index} failed to claim"
            claimed.append(attempt.task_cid)
            assert attempt.owner_session_id == f"session:{index}"
            assert attempt.committed_phase == "claimed"
        finally:
            daemon.close()

    assert len(claimed) == 4
    assert len(set(claimed)) == 4

    idle = _open_daemon(tmp_path, session="session:extra", markdown_path=markdown)
    try:
        assert idle.claim_next() is None
        assert idle.markdown_status_write_count == 0
        assert markdown.read_text(encoding="utf-8") == original_markdown
    finally:
        idle.close()


def test_strict_shards_claim_only_home_lane_tasks(tmp_path: Path) -> None:
    seed = _open_daemon(tmp_path, session="session:seed")
    try:
        seed.materialize_population(_population(8))
    finally:
        seed.close()

    claimed: dict[int, str] = {}
    for index in range(4):
        daemon = _open_daemon(
            tmp_path,
            session=f"session:shard-{index}",
            task_shard_count=4,
            task_shard_index=index,
            strict_task_sharding=True,
            task_prefix="DQP-T",
        )
        try:
            attempt = daemon.claim_next()
            assert attempt is not None, f"shard {index} found no home-lane work"
            alias = str(attempt.task_alias or "")
            home = daemon._task_home_shard_index(alias)
            assert home == index, f"{alias} home={home} claimed by shard {index}"
            claimed[index] = alias
        finally:
            daemon.close()

    assert len(set(claimed.values())) == 4


def test_no_markdown_status_update_under_database_authority(tmp_path: Path) -> None:
    markdown = tmp_path / "tasks.md"
    markdown.write_text(
        "# Tasks\n\n## DQP-T001 Work\n\n- Status: todo\n",
        encoding="utf-8",
    )
    before = markdown.read_text(encoding="utf-8")
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:md",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        markdown_path=markdown,
    )
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        assert result["unchanged"] is False
        assert result["markdown_status_writes"] == 0
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "completed"
        with pytest.raises(DatabaseImplementationAuthorityError, match="Markdown"):
            daemon.write_markdown_task_status("DQP-T001", "completed")
        assert markdown.read_text(encoding="utf-8") == before
        assert "- Status: completed" not in markdown.read_text(encoding="utf-8")
    finally:
        daemon.close()


def test_json_projections_can_be_absent(tmp_path: Path) -> None:
    daemon = open_database_implementation_daemon(
        tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:proj",
        authority_mode="embedded",
        task_source_kind="duckdb",
    )
    try:
        assert daemon.projections_required() is False
        assert daemon.state_path is None
        assert daemon.strategy_path is None
        assert daemon.events_path is None
        assert daemon.pid_path is None
        assert daemon.queue_path is None
        # No projection files created by open/materialize/run.
        daemon.materialize_population(_population(1))
        daemon.run_once()
        assert not (tmp_path / "task_state.json").exists()
        assert not (tmp_path / "events.jsonl").exists()
        assert not (tmp_path / "task_queue.json").exists()
        assert not list(tmp_path.glob("*.pid"))
    finally:
        daemon.close()


def test_datasets_authoritative_open_requires_preinstalled_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    control_path = tmp_path / "missing-control.duckdb"
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="preinstalled by the trusted materializer",
    ):
        DatabaseImplementationDaemon(
            database_path=control_path,
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            authority_mode="embedded",
            task_source_kind="duckdb",
        )
    assert not control_path.exists()
    assert not (tmp_path / "execution.duckdb").exists()


def test_datasets_authoritative_open_rejects_full_control_plane_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "full-control.duckdb"
    install_control_plane_schema(control_path)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="not the verified datasets-authoritative operational profile",
    ):
        DatabaseImplementationDaemon(
            database_path=control_path,
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            authority_mode="embedded",
            task_source_kind="duckdb",
        )
    with open_duckdb_connection(control_path) as connection:
        names = {str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()}
    assert "proof_obligations" in names
    assert not (tmp_path / "execution.duckdb").exists()


def test_datasets_authoritative_open_rejects_tampered_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "tampered-control.duckdb"
    install_datasets_authoritative_operational_schema(control_path)
    with open_duckdb_connection(control_path) as connection:
        connection.execute(
            "UPDATE schema_migrations SET checksum = 'sha256:tampered'"
        )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="not the verified datasets-authoritative operational profile",
    ):
        DatabaseImplementationDaemon(
            database_path=control_path,
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            authority_mode="embedded",
            task_source_kind="duckdb",
        )
    assert not (tmp_path / "execution.duckdb").exists()


def test_datasets_authoritative_open_verifies_existing_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "operational-control.duckdb"
    install_datasets_authoritative_operational_schema(control_path)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    daemon = DatabaseImplementationDaemon(
        database_path=control_path,
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        authority_mode="embedded",
        task_source_kind="duckdb",
    )
    try:
        evidence = dict(daemon.control_schema_evidence)
        assert evidence["state_schema_revision"] == (
            "datasets-authoritative-operational-v1"
        )
        assert evidence["verified"] is True
        assert evidence["profile_id"]
        assert evidence["schema_fingerprint"]
        daemon.materialize_population(_population(1))
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "ready"
    finally:
        daemon.close()


def test_crash_restart_resumes_without_duplicating_provider_or_effect(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []

    first = _open_daemon(
        tmp_path,
        session="session:resume",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, "context")
        attempt, provider_result, duplicated = first.run_provider(attempt)
        assert duplicated is False
        assert provider_calls == ["task:cid:001"]
        assert attempt.committed_phase == ATTEMPT_PHASE_PROVIDER
        # Crash boundary: process dies after provider commits, before effect.
        assert effect_calls == []
        attempt_id = attempt.attempt_id
    finally:
        first.close()

    second = _open_daemon(
        tmp_path,
        session="session:resume",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        running = second.list_running_attempts()
        assert len(running) == 1
        assert running[0].attempt_id == attempt_id
        assert running[0].committed_phase == ATTEMPT_PHASE_PROVIDER
        result = second.resume_attempt(running[0])
        assert result["resumed"] is True
        assert result["provider_duplicated"] is True
        assert result["effect_duplicated"] is False
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        assert result["committed_phase"] == ATTEMPT_PHASE_COMPLETE
        assert result["status"] == "succeeded"
        task = second.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "completed"

        # Second resume of a finished attempt is a no-op for provider/effect.
        finished = second.get_attempt(attempt_id)
        assert finished is not None
        again = second.resume_attempt(finished)
        assert again["resumed"] is False
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
    finally:
        second.close()


def test_implicit_embedded_owner_is_store_scoped_and_restart_stable(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        first_owner = first.owner_session_id
        assert first_owner.startswith("embedded-store:")
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, "context")
        attempt, _, duplicated = first.run_provider(attempt)
        assert duplicated is False
        assert provider_calls == [attempt.task_cid]
    finally:
        first.close()

    second = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        assert second.owner_session_id == first_owner
        result = second.run_once()
        assert result["implementation_result"]["provider_duplicated"] is True
        assert result["implementation_result"]["status"] == "succeeded"
        assert provider_calls == [attempt.task_cid]
        assert effect_calls == [attempt.task_cid]
    finally:
        second.close()


def test_implicit_embedded_owner_is_distinct_for_different_stores(
    tmp_path: Path,
) -> None:
    first = _open_daemon(tmp_path / "first")
    second = _open_daemon(tmp_path / "second")
    try:
        assert first.owner_session_id.startswith("embedded-store:")
        assert second.owner_session_id.startswith("embedded-store:")
        assert first.owner_session_id != second.owner_session_id
    finally:
        second.close()
        first.close()


def test_embedded_writer_lock_rejects_a_concurrent_same_store_opener(
    tmp_path: Path,
) -> None:
    first = _open_daemon(tmp_path)
    try:
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="active database writer",
        ):
            _open_daemon(tmp_path)
        first.materialize_population(_population(1))
        assert first.claim_next() is not None
    finally:
        first.close()

    replacement = _open_daemon(tmp_path)
    try:
        assert replacement.owner_session_id == first.owner_session_id
    finally:
        replacement.close()


def test_strict_lanes_use_distinct_sidecars_and_duplicate_lane_still_fences(
    tmp_path: Path,
) -> None:
    task_source = DatabaseTaskSource(tmp_path / "control.duckdb")

    def open_lane(index: int) -> DatabaseImplementationDaemon:
        return DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            coordination_path=tmp_path / "coordination.duckdb",
            state_dir=tmp_path / "state" / f"lane-{index}",
            state_prefix=f"dqp-lane-{index}",
            authority_mode="embedded",
            task_source_kind="duckdb",
            task_source=task_source,
            task_shard_count=2,
            task_shard_index=index,
            strict_task_sharding=True,
        )

    first = open_lane(0)
    second: DatabaseImplementationDaemon | None = None
    try:
        second = open_lane(1)
        assert isinstance(
            first.coordinator, ProcessSerializedDatabaseCoordinator
        )
        assert isinstance(
            second.coordinator, ProcessSerializedDatabaseCoordinator
        )
        assert first.execution_path == (
            tmp_path
            / "state"
            / "lane-0"
            / "dqp-lane-0.execution.shard-0000-of-0002.duckdb"
        )
        assert second.execution_path == (
            tmp_path
            / "state"
            / "lane-1"
            / "dqp-lane-1.execution.shard-0001-of-0002.duckdb"
        )
        assert first.execution_path != second.execution_path
        assert first._embedded_writer_lock_path != second._embedded_writer_lock_path
        assert first._embedded_writer_lock_handle is not None
        assert second._embedded_writer_lock_handle is not None

        # A direct coordinator can open while both logical adapters are live:
        # neither lane retains the shared DuckDB file lock between methods.
        direct = open_database_coordinator(tmp_path / "coordination.duckdb")
        direct.close()

        first.materialize_population(_population(8))
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(first.claim_next)
            second_future = pool.submit(second.claim_next)
            first_claim = first_future.result(timeout=30)
            second_claim = second_future.result(timeout=30)
        assert first_claim is not None
        assert second_claim is not None
        assert first_claim.task_cid != second_claim.task_cid
        assert first._task_home_shard_index(first_claim.task_alias) == 0
        assert second._task_home_shard_index(second_claim.task_alias) == 1

        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="active database writer",
        ):
            open_lane(0)
    finally:
        if second is not None:
            second.close()
        first.close()
        task_source.close()


def test_strict_lane_reopens_its_exact_execution_sidecar(tmp_path: Path) -> None:
    first = _open_daemon(
        tmp_path,
        task_shard_count=4,
        task_shard_index=2,
        strict_task_sharding=True,
    )
    execution_path = first.execution_path
    first_process_id = str(
        first._connection.execute(
            """
            SELECT value FROM daemon_execution_metadata
            WHERE key = 'process_instance_id'
            """
        ).fetchone()["value"]
    )
    first.close()

    replacement = _open_daemon(
        tmp_path,
        task_shard_count=4,
        task_shard_index=2,
        strict_task_sharding=True,
    )
    try:
        assert replacement.execution_path == execution_path
        rows = replacement._connection.execute(
            """
            SELECT key, value FROM daemon_execution_metadata
            WHERE key IN (
                'execution_state_dir',
                'execution_state_prefix',
                'task_shard_count',
                'task_shard_index',
                'strict_task_sharding'
            )
            """
        ).fetchall()
        assert {str(row["key"]): str(row["value"]) for row in rows} == {
            "execution_state_dir": str(tmp_path / "state" / "lane-2"),
            "execution_state_prefix": "dqp-lane-2",
            "task_shard_count": "4",
            "task_shard_index": "2",
            "strict_task_sharding": "true",
        }
        replacement_process_id = str(
            replacement._connection.execute(
                """
                SELECT value FROM daemon_execution_metadata
                WHERE key = 'process_instance_id'
                """
            ).fetchone()["value"]
        )
        assert replacement_process_id != first_process_id
    finally:
        replacement.close()


def test_strict_lane_reopen_rejects_missing_binding_rows_with_evidence(
    tmp_path: Path,
) -> None:
    first = _open_daemon(
        tmp_path,
        task_shard_count=4,
        task_shard_index=2,
        strict_task_sharding=True,
    )
    first._connection.execute(
        """
        DELETE FROM daemon_execution_metadata
        WHERE key IN (
            'execution_state_dir',
            'execution_state_prefix',
            'task_shard_count',
            'task_shard_index',
            'strict_task_sharding'
        )
        """
    )
    first.close()

    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="execution sidecar lane metadata does not match",
    ):
        _open_daemon(
            tmp_path,
            task_shard_count=4,
            task_shard_index=2,
            strict_task_sharding=True,
        )


def test_process_serialized_coordinator_keeps_fenced_callback_atomic(
    tmp_path: Path,
) -> None:
    coordinator = open_process_serialized_database_coordinator(
        tmp_path / "coordination.duckdb"
    )
    try:
        coordinator.register_task(task_cid="task:one", task_id="ONE")
        claim = coordinator.claim_task(
            task_cid="task:one",
            owner_session_id="session:one",
        )
        writer = coordinator.claim_resource(
            resource_kind="database_writer",
            resource_id="shared-control",
            owner_session_id="session:one",
            task_cid="task:one",
        )
        before = coordinator.lease_events()

        def swallowed_reentry() -> dict[str, object]:
            with pytest.raises(
                DatabaseCoordinationConflictError,
                match="must not re-enter",
            ):
                coordinator.release(writer.as_fenced_lease())
            return {"status": "swallowed"}

        with pytest.raises(
            DatabaseCoordinationConflictError,
            match="attempted to re-enter",
        ):
            coordinator.execute_with_task_and_resource_fences(
                claim,
                writer,
                swallowed_reentry,
            )
        assert coordinator.lease_events() == before
        observed = coordinator.get_lease(writer.lease_id)
        assert observed is not None
        assert observed.state is LeaseState.ACCEPTED
    finally:
        coordinator.close()


def test_process_serialized_coordinator_types_lock_timeout(
    tmp_path: Path,
) -> None:
    coordinator = open_process_serialized_database_coordinator(
        tmp_path / "coordination.duckdb",
        lock_timeout_seconds=0.05,
    )
    entered = Event()
    release = Event()

    def hold_serialization_lock() -> None:
        with exclusive_file_lock(coordinator.serialization_lock_path):
            entered.set()
            assert release.wait(timeout=5)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            held = pool.submit(hold_serialization_lock)
            assert entered.wait(timeout=5)
            with pytest.raises(
                DatabaseCoordinationConflictError,
                match="process-serialized coordination authority",
            ):
                coordinator.lease_events()
            release.set()
            held.result(timeout=5)
        assert coordinator.lease_events() == []
    finally:
        release.set()
        coordinator.close()


def test_process_serialized_coordinator_guards_claim_filter_and_close(
    tmp_path: Path,
) -> None:
    coordinator = open_process_serialized_database_coordinator(
        tmp_path / "coordination.duckdb"
    )
    coordinator.register_task(task_cid="task:one", task_id="ONE")
    callback_entered = Event()
    release_callback = Event()
    close_started = Event()
    close_finished = Event()

    def accept_task_cid(_task_cid: str) -> bool:
        callback_entered.set()
        assert release_callback.wait(timeout=5)
        with pytest.raises(
            DatabaseCoordinationConflictError,
            match="must not re-enter",
        ):
            coordinator.lease_events()
        return True

    def close_coordinator() -> None:
        close_started.set()
        coordinator.close()
        close_finished.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            claim_future = pool.submit(
                coordinator.claim_ready_task,
                owner_session_id="session:one",
                accept_task_cid=accept_task_cid,
            )
            assert callback_entered.wait(timeout=5)
            close_future = pool.submit(close_coordinator)
            assert close_started.wait(timeout=5)
            assert not close_finished.wait(timeout=0.1)
            release_callback.set()
            with pytest.raises(
                DatabaseCoordinationConflictError,
                match="attempted to re-enter",
            ):
                claim_future.result(timeout=5)
            close_future.result(timeout=5)
        assert close_finished.is_set()
        assert coordinator.is_open is False
    finally:
        release_callback.set()
        coordinator.close()


def test_multi_lane_sidecar_binding_rejects_unsafe_or_inconsistent_inputs(
    tmp_path: Path,
) -> None:
    base = {
        "database_path": tmp_path / "control.duckdb",
        "coordination_path": tmp_path / "coordination.duckdb",
        "authority_mode": "embedded",
        "task_source_kind": "duckdb",
        "install_schema": False,
        "task_shard_count": 2,
        "task_shard_index": 0,
    }
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="strict task sharding",
    ):
        DatabaseImplementationDaemon(**base)
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="sealed state_dir/state_prefix",
    ):
        DatabaseImplementationDaemon(**base, strict_task_sharding=True)
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="safe path component",
    ):
        DatabaseImplementationDaemon(
            **base,
            strict_task_sharding=True,
            state_dir=tmp_path / "state",
            state_prefix="../lane-0",
        )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="does not match sealed lane binding",
    ):
        DatabaseImplementationDaemon(
            **base,
            strict_task_sharding=True,
            state_dir=tmp_path / "state",
            state_prefix="lane-0",
            execution_path=tmp_path / "wrong.duckdb",
        )
    assert not (tmp_path / "wrong.duckdb").exists()


def test_single_lane_preserves_control_adjacent_execution_sidecar(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.duckdb"
    daemon = DatabaseImplementationDaemon(
        database_path=control_path,
        coordination_path=tmp_path / "coordination.duckdb",
        state_dir=tmp_path / "state",
        state_prefix="legacy-lane",
        authority_mode="embedded",
        task_source_kind="duckdb",
        install_schema=False,
    )
    try:
        assert daemon.execution_path == tmp_path / "control.execution.duckdb"
        assert not daemon.execution_path.exists()
    finally:
        daemon.close()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task_shard_count", 2.5, "positive integer"),
        ("task_shard_count", "2", "positive integer"),
        ("task_shard_index", 0.5, "must be an integer"),
        ("task_shard_index", "0", "must be an integer"),
    ],
)
def test_shard_binding_rejects_non_integral_values_without_truncation(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "database_path": tmp_path / "control.duckdb",
        "authority_mode": "embedded",
        "task_source_kind": "duckdb",
        "install_schema": False,
        "task_shard_count": 1,
        "task_shard_index": 0,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        DatabaseImplementationDaemon(**kwargs)


def test_effect_phase_resume_skips_both_provider_and_effect(tmp_path: Path) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        session="session:effect",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, "context")
        attempt, provider_result, _ = first.run_provider(attempt)
        attempt, effect_result, _ = first.run_effect(attempt, provider_result)
        assert attempt.committed_phase == ATTEMPT_PHASE_EFFECT
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        attempt_id = attempt.attempt_id
    finally:
        first.close()

    second = _open_daemon(
        tmp_path,
        session="session:effect",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        attempt = second.get_attempt(attempt_id)
        assert attempt is not None
        result = second.resume_attempt(attempt)
        assert result["provider_duplicated"] is True
        assert result["effect_duplicated"] is True
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        assert result["status"] == "succeeded"
    finally:
        second.close()


def test_provider_heartbeat_renews_exact_task_claim(tmp_path: Path) -> None:
    holder: dict[str, DatabaseImplementationDaemon] = {}
    observed_revisions: list[int] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        daemon = holder["daemon"]
        initial = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert initial is not None
        observed_revisions.append(int(initial.revision))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            current = daemon.coordinator.get_task_claim(attempt.claim_id)
            assert current is not None
            if int(current.revision) > int(initial.revision):
                observed_revisions.append(int(current.revision))
                break
            time.sleep(0.005)
        assert len(observed_revisions) == 2, "background lease renewal did not run"
        return {"status": "ok", "task_cid": attempt.task_cid}

    daemon = _open_daemon(
        tmp_path,
        session="session:heartbeat",
        provider_fn=provider,
        lease_ms=5_000,
    )
    holder["daemon"] = daemon
    daemon._lease_heartbeat_interval_seconds = 0.01
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        attempt = daemon.commit_phase(attempt, "context")
        updated, _, duplicated = daemon.run_provider(attempt)
        assert duplicated is False
        assert updated.committed_phase == ATTEMPT_PHASE_PROVIDER
        assert observed_revisions[1] > observed_revisions[0]
    finally:
        daemon.close()


def test_provider_result_is_rejected_after_fenced_takeover(tmp_path: Path) -> None:
    now = {"ms": 1_000}
    holder: dict[str, DatabaseImplementationDaemon] = {}
    replacement_claim_ids: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        daemon = holder["daemon"]
        # Cross the renewed deadline and let another session claim the same
        # ready coordination task before this provider result is returned.
        now["ms"] = 7_000
        replacement = daemon.coordinator.claim_ready_task(
            owner_session_id="session:replacement",
            lease_ms=5_000,
            now_ms=now["ms"],
        )
        assert replacement is not None
        assert replacement.task_cid == attempt.task_cid
        replacement_claim_ids.append(replacement.claim_id)
        return {"status": "ok", "task_cid": attempt.task_cid}

    daemon = _open_daemon(
        tmp_path,
        session="session:stale-provider",
        provider_fn=provider,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    holder["daemon"] = daemon
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        attempt = daemon.commit_phase(attempt, "context")
        with pytest.raises(DatabaseCoordinationError):
            daemon.run_provider(attempt)
        assert replacement_claim_ids
        assert (
            daemon.provider_invocation_recorded(
                attempt.attempt_id,
                idempotency_key=f"provider:{attempt.attempt_id}",
            )
            is None
        )
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.committed_phase == "context"
        assert stored.status == "running"
    finally:
        daemon.close()


def test_expired_attempt_cannot_commit_logical_completion(tmp_path: Path) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:expired-completion",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)
        now["ms"] = 6_000
        with pytest.raises(DatabaseCoordinationError):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "a" * 64,
                    "argv": ["focused-validation"],
                },
            )
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "in_progress"
        assert daemon.coordinator.claimability(attempt.task_cid)["claimable"] is True
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.committed_phase == "validation"
        assert stored.status == "running"
    finally:
        daemon.close()


def test_restart_retires_prepared_absent_expired_attempt_then_refences_retry(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        old_attempt = first.claim_next()
        assert old_attempt is not None
        old_attempt = first.commit_phase(old_attempt, "context")
        old_attempt, _, duplicated = first.run_provider(old_attempt)
        assert duplicated is False
        old_owner = first.owner_session_id
    finally:
        first.close()

    # No intervening coordinator mutation performs an expiry sweep.
    now["ms"] = 7_000
    replacement = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        assert replacement.owner_session_id == old_owner
        result = replacement.run_once()
        reconciliations = result["expired_attempt_reconciliations"]
        assert len(reconciliations) == 1
        assert reconciliations[0]["status"] == "expired"
        assert reconciliations[0]["provider_evidence_reused"] is False
        assert reconciliations[0]["effect_evidence_reused"] is False
        assert result["attempt_id"] != old_attempt.attempt_id
        assert result["implementation_result"]["status"] == "succeeded"
        assert provider_calls == [old_attempt.task_cid, old_attempt.task_cid]
        assert effect_calls == [old_attempt.task_cid]
        retired = replacement.get_attempt(old_attempt.attempt_id)
        assert retired is not None
        assert retired.status == "failed"
        assert retired.committed_phase == "failed"
        replacement_claim = replacement.coordinator.get_task_claim(
            result["claim_id"]
        )
        assert replacement_claim is not None
        assert replacement_claim.fencing_token > old_attempt.fencing_token
    finally:
        replacement.close()


def test_restart_replays_failed_expired_attempt_control_requeue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    session = "session:recover-failed-expired-requeue"
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        session=session,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        old_attempt = first.claim_next()
        assert old_attempt is not None
        old_attempt = first.commit_phase(old_attempt, "context")
        old_attempt, _, duplicated = first.run_provider(old_attempt)
        assert duplicated is False
        now["ms"] = 7_000

        def crash_before_control_requeue(*args: object, **kwargs: object) -> object:
            raise SystemExit("simulated death before control requeue")

        monkeypatch.setattr(
            first,
            "_requeue_control_after_expired_attempt",
            crash_before_control_requeue,
        )
        with pytest.raises(SystemExit, match="before control requeue"):
            first.run_once()
        retired = first.get_attempt(old_attempt.attempt_id)
        assert retired is not None and retired.status == "failed"
        task = first.task_source.get(old_attempt.task_cid)
        assert task is not None and task.status == "in_progress"
        claim = first.coordinator.get_task_claim(old_attempt.claim_id)
        assert claim is not None and claim.state.value == "expired"
    finally:
        first.close()

    replacement = _open_daemon(
        tmp_path,
        session=session,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        result = replacement.run_once()
        assert result["implementation_result"]["status"] == "succeeded"
        assert result["attempt_id"] != old_attempt.attempt_id
        assert len(result["orphan_claim_reconciliations"]) == 1
        replay = result["orphan_claim_reconciliations"][0]
        assert replay["operation"] == (
            "automatic_failed_attempt_requeue_replay"
        )
        assert replay["attempt_id"] == old_attempt.attempt_id
        assert replay["claim_state"] == "expired"
        assert replay["prior_execution_evidence_reused"] is False
        assert result["write_count"] == 2
        assert provider_calls == [old_attempt.task_cid, old_attempt.task_cid]
        assert effect_calls == [old_attempt.task_cid]
    finally:
        replacement.close()


def test_claim_persists_exact_post_cas_control_revision_binding(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:claim-control-binding")
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "in_progress"
        binding = attempt.body["control_binding"]
        assert binding["task_cid"] == attempt.task_cid
        assert binding["claim_id"] == attempt.claim_id
        assert binding["attempt_id"] == attempt.attempt_id
        assert binding["attempt_number"] == attempt.attempt_number
        assert binding["lease_id"] == attempt.lease_id
        assert binding["fencing_token"] == attempt.fencing_token
        assert binding["fence_epoch"] == attempt.fence_epoch
        assert binding["owner_session_id"] == attempt.owner_session_id
        assert binding["schema"] == DATABASE_CONTROL_CLAIM_BINDING_SCHEMA
        assert binding["control_expected_status"] == "in_progress"
        assert binding["control_expected_revision"] == task.revision == 2
        assert binding["control_task_projection_cid"] == content_identity(
            task.to_dict()
        )
        basis = binding["database_portal_binding_basis"]
        assert set(basis) == {
            "schema",
            "task_alias",
            "task_revision",
            "goal_cid",
            "plan_cid",
            "task_body_digest",
            "control_task_projection_cid",
        }
        assert basis["task_alias"] == attempt.task_alias
        assert basis["task_revision"] == task.revision
        assert basis["control_task_projection_cid"] == binding[
            "control_task_projection_cid"
        ]
        assert binding["database_portal_binding_basis_cid"] == content_identity(
            basis
        )
        assert str(binding["binding_id"]).startswith("bagu")
        assert daemon._control_binding_for_attempt(attempt) == binding
    finally:
        daemon.close()


@pytest.mark.parametrize(
    (
        "current_control_schema",
        "current_portal_schema",
        "prior_control_schema",
        "prior_portal_schema",
        "current_revision",
        "prior_revision",
    ),
    (
        (
            DATABASE_CONTROL_CLAIM_BINDING_SCHEMA,
            DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
            DATABASE_CONTROL_CLAIM_BINDING_SCHEMA,
            DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
            11,
            11,
        ),
        (
            DATABASE_CONTROL_CLAIM_BINDING_SCHEMA,
            DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
            DATABASE_CONTROL_CLAIM_BINDING_SCHEMA_V1,
            DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1,
            11,
            11,
        ),
        (
            DATABASE_CONTROL_CLAIM_BINDING_SCHEMA_V1,
            DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1,
            DATABASE_CONTROL_CLAIM_BINDING_SCHEMA_V1,
            DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1,
            11,
            11,
        ),
        (
            DATABASE_CONTROL_CLAIM_BINDING_SCHEMA,
            DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1,
            DATABASE_CONTROL_CLAIM_BINDING_SCHEMA,
            DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1,
            11,
            11,
        ),
        (
            DATABASE_CONTROL_CLAIM_BINDING_SCHEMA,
            DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
            DATABASE_CONTROL_CLAIM_BINDING_SCHEMA,
            DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
            12,
            11,
        ),
    ),
)
def test_superseded_portal_authority_closes_control_and_legacy_contracts(
    tmp_path: Path,
    current_control_schema: str,
    current_portal_schema: str,
    prior_control_schema: str,
    prior_portal_schema: str,
    current_revision: int,
    prior_revision: int,
) -> None:
    fixture = _superseded_authority_fixture(
        tmp_path,
        current_control_schema=current_control_schema,
        prior_control_schema=prior_control_schema,
        current_portal_schema=current_portal_schema,
        prior_portal_schema=prior_portal_schema,
        current_revision=current_revision,
        prior_revision=prior_revision,
    )

    authority = fixture.daemon.authorize_superseded_portal_attempt_binding(
        fixture.current,
        fixture.current_portal,
        fixture.prior_portal,
    )

    assert authority == {
        "schema": database_portal_bridge_module.CROSS_ATTEMPT_LIFECYCLE_AUTHORITY_SCHEMA,
        "authorized": True,
        "task_cid": fixture.current.task_cid,
        "task_alias": fixture.current.task_alias,
        "current_attempt_id": fixture.current.attempt_id,
        "prior_attempt_id": fixture.prior.attempt_id,
        "current_attempt_number": fixture.current.attempt_number,
        "prior_attempt_number": fixture.prior.attempt_number,
        "current_binding_id": fixture.current_portal["binding_id"],
        "prior_binding_id": fixture.prior_portal["binding_id"],
        "current_fencing_token": fixture.current.fencing_token,
        "prior_fencing_token": fixture.prior.fencing_token,
        "current_control_binding_id": fixture.current_control["binding_id"],
        "prior_control_binding_id": fixture.prior_control["binding_id"],
        "current_control_task_projection_cid": fixture.current_control[
            "control_task_projection_cid"
        ],
        "prior_control_task_projection_cid": fixture.prior_control[
            "control_task_projection_cid"
        ],
        "current_control_expected_revision": current_revision,
        "prior_control_expected_revision": prior_revision,
        "prior_execution_status": "failed",
        "prior_claim_state": "expired",
        "prior_coordination_status": "failed",
        "legacy_current_binding": current_portal_schema
        == DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1,
        "legacy_prior_binding": prior_portal_schema
        == DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1,
        "mutation_authority": False,
        "completion_authority": False,
    }
    assert set(authority) == database_portal_bridge_module._PRIOR_AUTHORITY_FIELDS
    assert DatabasePortalExecutionBridge._validated_prior_authority(
        authority,
        current_binding=fixture.current_portal,
        prior_binding=fixture.prior_portal,
    ) == authority
    assert fixture.protected == [
        fixture.current.attempt_id,
        fixture.current.attempt_id,
    ]
    serialized = json.dumps(authority)
    assert "lease_id" not in serialized
    assert "owner_session_id" not in serialized


@pytest.mark.parametrize(
    ("side", "field"),
    (
        ("current", "control_binding_id"),
        ("prior", "control_task_projection_cid"),
        ("current", "control_portal_binding_basis_cid"),
        ("prior", "goal_cid"),
        ("current", "plan_cid"),
        ("prior", "task_body_digest"),
        ("current", "revision_pair"),
    ),
)
def test_superseded_portal_authority_rejects_rehashed_portal_mismatch(
    tmp_path: Path,
    side: str,
    field: str,
) -> None:
    fixture = _superseded_authority_fixture(tmp_path)
    selected = dict(getattr(fixture, f"{side}_portal"))
    if field == "task_body_digest":
        selected[field] = "sha256:" + ("0" * 64)
    elif field == "revision_pair":
        selected["task_revision"] = 12
        selected["control_expected_revision"] = 12
    else:
        selected[field] = f"forged:{field}"
    selected = _rehash_portal_binding(selected)
    current_portal = (
        selected if side == "current" else fixture.current_portal
    )
    prior_portal = selected if side == "prior" else fixture.prior_portal

    with pytest.raises(DatabaseImplementationAuthorityError):
        fixture.daemon.authorize_superseded_portal_attempt_binding(
            fixture.current,
            current_portal,
            prior_portal,
        )


@pytest.mark.parametrize(
    ("control_schema", "mutation"),
    (
        (DATABASE_CONTROL_CLAIM_BINDING_SCHEMA, "basis_alias"),
        (DATABASE_CONTROL_CLAIM_BINDING_SCHEMA, "basis_open_record"),
        (DATABASE_CONTROL_CLAIM_BINDING_SCHEMA, "attempt_number"),
        (DATABASE_CONTROL_CLAIM_BINDING_SCHEMA_V1, "owner_session_id"),
    ),
)
def test_superseded_portal_authority_rejects_rehashed_control_mismatch(
    tmp_path: Path,
    control_schema: str,
    mutation: str,
) -> None:
    prior_portal_schema = (
        DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA
        if control_schema == DATABASE_CONTROL_CLAIM_BINDING_SCHEMA
        else DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1
    )
    fixture = _superseded_authority_fixture(
        tmp_path,
        prior_control_schema=control_schema,
        prior_portal_schema=prior_portal_schema,
    )
    control = dict(fixture.prior_control)
    if mutation.startswith("basis_"):
        basis = dict(control["database_portal_binding_basis"])
        if mutation == "basis_alias":
            basis["task_alias"] = "AUTH-FORGED"
        else:
            basis["unexpected"] = "forged"
        control["database_portal_binding_basis"] = basis
        control["database_portal_binding_basis_cid"] = content_identity(basis)
    elif mutation == "attempt_number":
        control["attempt_number"] = fixture.prior.attempt_number + 1
    else:
        control["owner_session_id"] = "session:forged"
    control = _rehash_control_binding(control)
    prior = replace(fixture.prior, body={"control_binding": control})
    daemon, _coordinator, _protected = _superseded_authority_fake(
        fixture.current,
        prior,
    )

    with pytest.raises(DatabaseImplementationAuthorityError):
        daemon.authorize_superseded_portal_attempt_binding(
            fixture.current,
            fixture.current_portal,
            fixture.prior_portal,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("attempt_number", 6),
        ("owner_session_id", "session:forged"),
    ),
)
def test_superseded_portal_authority_rejects_stale_current_identity(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    fixture = _superseded_authority_fixture(tmp_path)
    passed_current = replace(fixture.current, **{field: value})

    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="stored identity",
    ):
        fixture.daemon.authorize_superseded_portal_attempt_binding(
            passed_current,
            fixture.current_portal,
            fixture.prior_portal,
        )


@pytest.mark.parametrize(
    ("target", "field", "value"),
    (
        ("prior_claim", "claim_id", "claim:forged"),
        ("prior_claim", "task_cid", "task:cid:forged"),
        ("prior_claim", "attempt_id", "attempt:forged"),
        ("prior_claim", "attempt_number", True),
        ("prior_claim", "owner_session_id", "session:forged"),
        ("prior_claim", "fencing_token", 5),
        ("prior_claim", "fence_epoch", 1),
        ("prior_claim", "lease_id", "lease:forged"),
        ("prior_claim", "state", SimpleNamespace(value="accepted")),
        ("prior_attempt", "attempt_id", "attempt:forged"),
        ("prior_attempt", "task_cid", "task:cid:forged"),
        ("prior_attempt", "attempt_number", True),
        ("prior_attempt", "owner_session_id", "session:forged"),
        ("prior_attempt", "fencing_token", 5),
        ("prior_attempt", "fence_epoch", 1),
        ("prior_attempt", "status", SimpleNamespace(value="running")),
    ),
)
def test_superseded_portal_authority_requires_exact_prior_coordination_tuple(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
) -> None:
    fixture = _superseded_authority_fixture(tmp_path)
    setattr(getattr(fixture.coordinator, target), field, value)

    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="remains live or mismatched",
    ):
        fixture.daemon.authorize_superseded_portal_attempt_binding(
            fixture.current,
            fixture.current_portal,
            fixture.prior_portal,
        )


@pytest.mark.parametrize(
    ("prior_attempt_number", "current_revision", "prior_revision"),
    (
        (5, 11, 11),
        (4, 11, 12),
    ),
)
def test_superseded_portal_authority_rejects_non_predecessor_order(
    tmp_path: Path,
    prior_attempt_number: int,
    current_revision: int,
    prior_revision: int,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=SimpleNamespace(),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: object(),
    )
    current, _current_control, current_portal = _authority_bound_attempt(
        bridge,
        attempt_id="attempt:current",
        claim_id="claim:current",
        attempt_number=5,
        owner_session_id="session:current",
        fencing_token=7,
        fence_epoch=3,
        lease_id="lease:current",
        revision=current_revision,
        status="running",
        control_schema=DATABASE_CONTROL_CLAIM_BINDING_SCHEMA,
    )
    prior, _prior_control, prior_portal = _authority_bound_attempt(
        bridge,
        attempt_id="attempt:prior",
        claim_id="claim:prior",
        attempt_number=prior_attempt_number,
        owner_session_id="session:prior",
        fencing_token=6,
        fence_epoch=2,
        lease_id="lease:prior",
        revision=prior_revision,
        status="failed",
        control_schema=DATABASE_CONTROL_CLAIM_BINDING_SCHEMA,
    )
    daemon, _coordinator, _protected = _superseded_authority_fake(
        current,
        prior,
    )

    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="not authoritatively superseded",
    ):
        daemon.authorize_superseded_portal_attempt_binding(
            current,
            current_portal,
            prior_portal,
        )


def test_restart_recovers_owned_claim_after_crash_before_control_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = "session:recover-before-control-cas"
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        session=session,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        first.materialize_population(_population(1))

        def crash_before_control_cas(*args: object, **kwargs: object) -> object:
            raise SystemExit("simulated process death before control CAS")

        monkeypatch.setattr(
            first,
            "_cas_task_status_database",
            crash_before_control_cas,
        )
        with pytest.raises(SystemExit, match="before control CAS"):
            first.claim_next()
        leases = first.coordinator.list_active_leases(
            lease_kind="task",
            owner_session_id=session,
        )
        assert len(leases) == 1
        original_claim = first.coordinator.get_task_claim(leases[0].claim_id)
        assert original_claim is not None
        assert first.get_attempt(original_claim.attempt_id) is None
        task = first.task_source.get(original_claim.task_cid)
        assert task is not None and task.status == "ready"
    finally:
        first.close()

    replacement = _open_daemon(
        tmp_path,
        session=session,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        result = replacement.run_once()
        assert result["implementation_result"]["status"] == "succeeded"
        assert result["claim_id"] == original_claim.claim_id
        assert result["attempt_id"] == original_claim.attempt_id
        assert provider_calls == [original_claim.task_cid]
        assert effect_calls == [original_claim.task_cid]
    finally:
        replacement.close()


def test_restart_recovers_owned_claim_after_control_cas_before_attempt_insert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = "session:recover-after-control-cas"
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        session=session,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        first.materialize_population(_population(1))

        def crash_before_attempt_insert(*args: object, **kwargs: object) -> object:
            raise SystemExit("simulated process death before attempt insert")

        monkeypatch.setattr(
            first,
            "_insert_attempt_from_claim",
            crash_before_attempt_insert,
        )
        with pytest.raises(SystemExit, match="before attempt insert"):
            first.claim_next()
        leases = first.coordinator.list_active_leases(
            lease_kind="task",
            owner_session_id=session,
        )
        assert len(leases) == 1
        original_claim = first.coordinator.get_task_claim(leases[0].claim_id)
        assert original_claim is not None
        assert first.get_attempt(original_claim.attempt_id) is None
        task = first.task_source.get(original_claim.task_cid)
        assert task is not None and task.status == "in_progress"
        assert task.body["completion_receipt"] == (
            first._database_claim_receipt(original_claim)
        )
    finally:
        first.close()

    replacement = _open_daemon(
        tmp_path,
        session=session,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        result = replacement.run_once()
        assert result["implementation_result"]["status"] == "succeeded"
        assert result["claim_id"] == original_claim.claim_id
        assert result["attempt_id"] == original_claim.attempt_id
        assert provider_calls == [original_claim.task_cid]
        assert effect_calls == [original_claim.task_cid]
    finally:
        replacement.close()


def test_expired_unadmitted_claim_is_exactly_requeued_and_reclaimed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    session = "session:recover-expired-unadmitted"
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        session=session,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))

        def crash_before_attempt_insert(*args: object, **kwargs: object) -> object:
            raise SystemExit("simulated process death before attempt insert")

        monkeypatch.setattr(
            first,
            "_insert_attempt_from_claim",
            crash_before_attempt_insert,
        )
        with pytest.raises(SystemExit, match="before attempt insert"):
            first.claim_next()
        leases = first.coordinator.list_active_leases(
            lease_kind="task",
            owner_session_id=session,
        )
        assert len(leases) == 1
        original_claim = first.coordinator.get_task_claim(leases[0].claim_id)
        assert original_claim is not None
    finally:
        first.close()

    now["ms"] = 7_000
    replacement = _open_daemon(
        tmp_path,
        session=session,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        result = replacement.run_once()
        assert result["implementation_result"]["status"] == "succeeded"
        assert result["claim_id"] != original_claim.claim_id
        assert result["attempt_id"] != original_claim.attempt_id
        assert len(result["orphan_claim_reconciliations"]) == 1
        assert result["write_count"] == 2
        recovery = result["orphan_claim_reconciliations"][0]
        assert recovery["claim_id"] == original_claim.claim_id
        assert recovery["claim_state"] == "expired"
        assert recovery["provider_execution_admitted"] is False
        assert recovery["effect_execution_admitted"] is False
        replacement_claim = replacement.coordinator.get_task_claim(
            result["claim_id"]
        )
        assert replacement_claim is not None
        assert replacement_claim.attempt_number == original_claim.attempt_number + 1
        assert replacement_claim.fencing_token > original_claim.fencing_token
        assert provider_calls == [original_claim.task_cid]
        assert effect_calls == [original_claim.task_cid]
    finally:
        replacement.close()


@pytest.mark.parametrize(
    ("successor_session", "claim_visibility"),
    [
        ("session:successor-wave", "accepted"),
        ("session:successor-wave", "released"),
        ("session:successor-wave", "missing"),
        ("session:prior-wave", "missing"),
    ],
)
def test_orphan_requeue_requires_owned_exact_claim_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    successor_session: str,
    claim_visibility: str,
) -> None:
    """No local attempt or visible worker proves nothing about foreign effects."""
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(tmp_path, session="session:prior-wave")
    try:
        first.materialize_population(_population(1))

        def crash_before_attempt_insert(*args: object, **kwargs: object) -> object:
            raise SystemExit("prior-wave crash")

        monkeypatch.setattr(first, "_insert_attempt_from_claim", crash_before_attempt_insert)
        with pytest.raises(SystemExit, match="prior-wave crash"):
            first.claim_next()
        lease = first.coordinator.list_active_leases(
            lease_kind="task", owner_session_id="session:prior-wave",
        )[0]
        original_claim = first.coordinator.get_task_claim(lease.claim_id)
        assert original_claim is not None
        if claim_visibility in {"released", "missing"}:
            first.coordinator.release(lease, reason="test-closed-lease")
        original_task = first.task_source.get(original_claim.task_cid)
        assert original_task is not None
    finally:
        first.close()

    successor = _open_daemon(
        tmp_path, session=successor_session,
        provider_calls=provider_calls, effect_calls=effect_calls,
    )
    try:
        original_get = successor.coordinator.get_task_claim
        before_claim = original_get(original_claim.claim_id)
        if claim_visibility == "missing":
            monkeypatch.setattr(
                successor.coordinator, "get_task_claim",
                lambda claim_id: None if claim_id == original_claim.claim_id else original_get(claim_id),
            )
        # A scan can miss an unlabelled callback or a detached provider. The
        # closed owner/claim proof must deny even when the scan sees nothing.
        monkeypatch.setattr(successor, "_list_implementation_process_commands", lambda: [], raising=False)
        assert successor._requeue_expired_owned_claims() == []
        after = successor.task_source.get(original_claim.task_cid)
        assert after is not None
        assert after.status == "in_progress"
        assert after.revision == original_task.revision
        assert after.body == original_task.body
        assert original_get(original_claim.claim_id) == before_claim
        assert provider_calls == []
        assert effect_calls == []
    finally:
        successor.close()


def test_automatic_claim_exclusions_re_resolve_legacy_on_hold_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:on-hold-exclusion")
    try:
        daemon.materialize_population(_population(1))
        task = daemon.task_source.ready_tasks(limit=1).tasks[0]
        original_get = daemon.task_source.get

        def get_with_operator_hold(task_cid: str) -> object:
            if task_cid == task.task_cid:
                return SimpleNamespace(
                    task_cid=task.task_cid,
                    status="on_hold",
                )
            return original_get(task_cid)

        monkeypatch.setattr(daemon.task_source, "get", get_with_operator_hold)

        assert daemon._automatic_claim_exclusions() == {task.task_cid}
    finally:
        daemon.close()


def test_post_claim_blocked_race_withdraws_without_provider_and_reports_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:post-claim-block-race",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        daemon.materialize_population(_population(2))
        original_claim = daemon.coordinator.claim_ready_task
        raced: dict[str, object] = {}

        def block_after_coordination_claim(*args: object, **kwargs: object) -> object:
            claim = original_claim(*args, **kwargs)
            if claim is not None and not raced:
                task = daemon.task_source.get(claim.task_cid)
                assert task is not None and task.status == "ready"
                daemon.task_source.compare_and_set_status(
                    task.task_cid,
                    expected_revision=int(task.revision),
                    status="blocked",
                    receipt={"operation": "independent_operator_block"},
                )
                raced.update(
                    {
                        "task_cid": claim.task_cid,
                        "claim_id": claim.claim_id,
                        "attempt_id": claim.attempt_id,
                    }
                )
            return claim

        monkeypatch.setattr(
            daemon.coordinator,
            "claim_ready_task",
            block_after_coordination_claim,
        )
        first = daemon.run_once()
        assert first["selection_idle_reason"] == (
            "claim_withdrawn_control_not_dispatchable"
        )
        assert first["unchanged"] is False
        assert first["write_count"] >= 1
        assert first["implementation_result"] is None
        assert first["claim_withdrawal"]["task_cid"] == raced["task_cid"]
        assert provider_calls == []
        assert effect_calls == []
        assert daemon.get_attempt(str(raced["attempt_id"])) is None
        released = daemon.coordinator.get_task_claim(str(raced["claim_id"]))
        assert released is not None and released.state.value == "released"
        released_attempt = daemon.coordinator.get_task_attempt(
            str(raced["attempt_id"])
        )
        assert released_attempt is not None
        assert released_attempt.status.value == "released"

        second = daemon.run_once()
        assert second["implementation_result"]["status"] == "succeeded"
        assert second["claimed_task_cid"] != raced["task_cid"]
        assert provider_calls == [second["claimed_task_cid"]]
        assert effect_calls == [second["claimed_task_cid"]]
    finally:
        daemon.close()


def test_terminal_portal_failure_blocks_and_releases_exact_claim_for_operator_retry(
    tmp_path: Path,
) -> None:
    fail = {"enabled": True}
    provider_calls: list[str] = []
    effect_calls: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        if fail["enabled"]:
            raise DatabasePortalBridgeError("terminal pre-provider failure")
        return {"status": "ok", "task_cid": attempt.task_cid}

    daemon = _open_daemon(
        tmp_path,
        session="session:portal-terminal",
        provider_fn=provider,
        effect_calls=effect_calls,
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        first_result = first["implementation_result"]
        assert first_result["portal_terminal_failure"] is True
        assert first_result["status"] == "blocked"
        first_attempt = daemon.get_attempt(first["attempt_id"])
        assert first_attempt is not None
        assert first_attempt.status == "failed"
        assert first_attempt.committed_phase == "failed"
        assert first_result["settlement"]["provider_invocation_count"] == 0
        assert first_result["settlement"]["effect_claim_count"] == 0

        task = daemon.task_source.get(first_attempt.task_cid)
        assert task is not None
        assert task.status == "blocked"
        receipt = task.body["completion_receipt"]
        assert receipt["attempt_id"] == first_attempt.attempt_id
        assert receipt["automatic_retry_admitted"] is False

        old_claim = daemon.coordinator.get_task_claim(first_attempt.claim_id)
        assert old_claim is not None
        assert old_claim.state.value == "released"
        old_coordination_attempt = daemon.coordinator.get_task_attempt(
            first_attempt.attempt_id
        )
        assert old_coordination_attempt is not None
        assert old_coordination_attempt.status.value == "failed"
        idle = daemon.run_once()
        assert idle["selection_idle_reason"] == "no_ready_tasks"
        assert idle.get("portal_failure_rearms") == []

        # A distinct trusted recovery receipt is required to requeue.  The
        # terminal failure path never retries itself.
        fail["enabled"] = False
        daemon.task_source.compare_and_set_status(
            task.task_cid,
            expected_revision=int(task.revision),
            status="retrying",
            receipt={
                "operation": "operator_control_plane_repair",
                "settlement_id": receipt["settlement_id"],
            },
        )
        second = daemon.run_once()
        assert second["implementation_result"]["status"] == "succeeded"
        second_attempt = daemon.get_attempt(second["attempt_id"])
        assert second_attempt is not None
        assert second_attempt.attempt_number == first_attempt.attempt_number + 1
        assert second_attempt.fencing_token > first_attempt.fencing_token
        assert provider_calls == [
            first_attempt.attempt_id,
            second_attempt.attempt_id,
        ]
        assert effect_calls == [first_attempt.task_cid]
    finally:
        daemon.close()


def test_recoverable_accepted_source_portal_failure_auto_rearms_blocked_task(
    tmp_path: Path,
) -> None:
    fail = {"enabled": True}
    provider_calls: list[str] = []
    effect_calls: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        if fail["enabled"]:
            raise DatabasePortalBridgeError(
                "Portal accepted-source transition is not the exact Git merge"
            )
        return {"status": "ok", "task_cid": attempt.task_cid}

    daemon = _open_daemon(
        tmp_path,
        session="session:portal-gitlink-rearm",
        provider_fn=provider,
        effect_calls=effect_calls,
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        first_result = first["implementation_result"]
        assert first_result["portal_terminal_failure"] is True
        assert first_result["status"] == "blocked"
        first_attempt = daemon.get_attempt(first["attempt_id"])
        assert first_attempt is not None
        task = daemon.task_source.get(first_attempt.task_cid)
        assert task is not None
        assert task.status == "blocked"
        assert task.body["completion_receipt"]["automatic_retry_admitted"] is False

        fail["enabled"] = False
        second = daemon.run_once()
        rearms = second.get("portal_failure_rearms") or []
        assert len(rearms) == 1
        assert rearms[0]["from_status"] == "blocked"
        assert rearms[0]["to_status"] == "retrying"
        assert rearms[0]["attempt_id"] == first_attempt.attempt_id
        assert second["implementation_result"]["status"] == "succeeded"
        second_attempt = daemon.get_attempt(second["attempt_id"])
        assert second_attempt is not None
        assert second_attempt.attempt_number == first_attempt.attempt_number + 1
        assert provider_calls == [
            first_attempt.attempt_id,
            second_attempt.attempt_id,
        ]
        assert effect_calls == [first_attempt.task_cid]
        third = daemon.run_once()
        assert third["selection_idle_reason"] == "no_ready_tasks"
        assert third.get("portal_failure_rearms") == []
    finally:
        daemon.close()


def test_zero_provider_portal_provider_failed_auto_rearms_blocked_task(
    tmp_path: Path,
) -> None:
    fail = {"enabled": True}
    provider_calls: list[str] = []
    effect_calls: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        if fail["enabled"]:
            raise DatabasePortalBridgeError("portal_provider_failed")
        return {"status": "ok", "task_cid": attempt.task_cid}

    daemon = _open_daemon(
        tmp_path,
        session="session:portal-provider-failed-rearm",
        provider_fn=provider,
        effect_calls=effect_calls,
    )
    _bind_explicit_zero_provider_callback(daemon)
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        first_result = first["implementation_result"]
        assert first_result["portal_terminal_failure"] is True
        assert first_result["status"] == "blocked"
        assert first_result["settlement"]["provider_invocation_count"] == 0
        assert first_result["settlement"]["effect_claim_count"] == 0
        first_attempt = daemon.get_attempt(first["attempt_id"])
        assert first_attempt is not None
        task = daemon.task_source.get(first_attempt.task_cid)
        assert task is not None
        assert task.status == "blocked"

        fail["enabled"] = False
        second = daemon.run_once()
        rearms = second.get("portal_failure_rearms") or []
        assert len(rearms) == 1
        assert rearms[0]["from_status"] == "blocked"
        assert rearms[0]["to_status"] == "retrying"
        assert second["implementation_result"]["status"] == "succeeded"
        second_attempt = daemon.get_attempt(second["attempt_id"])
        assert second_attempt is not None
        assert second_attempt.attempt_number == first_attempt.attempt_number + 1
        assert provider_calls == [
            first_attempt.attempt_id,
            second_attempt.attempt_id,
        ]
        assert effect_calls == [first_attempt.task_cid]
    finally:
        daemon.close()


@pytest.mark.parametrize("reason", [
    "cross_attempt_lifecycle_worktree_process_active",
    "cross_attempt_lifecycle_process_inventory_unavailable",
    "cross_attempt_lifecycle_absent_database_effect_admitted",
    "new_native_deferred_reason",
    "Portal task projection is not complete",
])
def test_unknown_portal_deferral_preserves_exact_running_claim(
    tmp_path: Path, reason: str,
) -> None:
    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeDeferred(reason)

    daemon = _open_daemon(
        tmp_path, session="session:deferred-claim", provider_fn=provider,
    )
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        assert result["implementation_result"]["deferred"] is True
        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt is not None and attempt.status == "running"
        assert daemon.task_source.get(attempt.task_cid).status == "in_progress"
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None and str(getattr(claim.state, "value", claim.state)) == "accepted"
        assert not daemon.reconcile_recoverable_portal_failure_rearms()
    finally:
        daemon.close()


def test_protected_path_failure_with_provider_evidence_does_not_auto_rearm(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    daemon = _open_daemon(tmp_path, provider_calls=calls)

    def deferred_effect(attempt: DatabaseTaskAttempt, result: dict[str, object]) -> dict[str, object]:
        raise DatabasePortalBridgeDeferred("implementation_protected_path_mutated")

    daemon._effect_fn = deferred_effect
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        attempt = daemon.get_attempt(first["attempt_id"])
        assert attempt is not None
        counts = daemon._attempt_execution_evidence_counts(attempt.attempt_id)
        assert counts["provider_invocation_count"] == 1
        assert counts["effect_claim_count"] == 0
        assert first["implementation_result"]["status"] == "blocked"
        assert daemon.reconcile_recoverable_portal_failure_rearms() == []
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"
        assert len(calls) == 1
    finally:
        daemon.close()


def test_protected_path_deferral_settles_instead_of_pinning_running_claim(
    tmp_path: Path,
) -> None:
    fail = {"enabled": True}

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        if fail["enabled"]:
            raise DatabasePortalBridgeDeferred(
                "implementation_protected_path_mutated"
            )
        return {"status": "ok", "task_cid": attempt.task_cid}

    daemon = _open_daemon(
        tmp_path,
        session="session:portal-protected-path",
        provider_fn=provider,
    )
    _bind_explicit_zero_provider_callback(daemon)
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        first_result = first["implementation_result"]
        assert first_result["portal_terminal_failure"] is True
        assert first_result["status"] == "blocked"
        assert first_result["reason"] == "implementation_protected_path_mutated"
        first_attempt = daemon.get_attempt(first["attempt_id"])
        assert first_attempt is not None
        assert first_attempt.status == "failed"
        task = daemon.task_source.get(first_attempt.task_cid)
        assert task is not None
        assert task.status == "blocked"

        fail["enabled"] = False
        second = daemon.run_once()
        rearms = second.get("portal_failure_rearms") or []
        assert len(rearms) == 1
        assert rearms[0]["reason"] == "implementation_protected_path_mutated"
        assert second["implementation_result"]["status"] == "succeeded"
    finally:
        daemon.close()


def _bind_explicit_zero_provider_callback(daemon):
    """Supply the native non-dispatch proof in these execution-store tests.

    The separate bridge tests exercise the real immutable callback verifier.
    Raising from this provider double alone is deliberately not that proof.
    """
    original = daemon._provider_fn

    class VerifiedCallbackProvider:
        def run(self, attempt):
            return original(attempt)

        def zero_provider_failure_rearm_ready(self, attempt):
            return attempt.status == "failed" and attempt.committed_phase == "failed"

    daemon._provider_fn = VerifiedCallbackProvider().run


def test_live_owner_auto_rearms_zero_provider_portal_claim_failure(
    tmp_path: Path,
) -> None:
    fail = {"enabled": True}
    provider_calls: list[str] = []
    effect_calls: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        if fail["enabled"]:
            raise DatabasePortalBridgeError("embedded-store claim without live owner")
        return {"status": "ok", "task_cid": attempt.task_cid}

    daemon = _open_daemon(
        tmp_path,
        session="session:portal-live-owner-rearm",
        provider_fn=provider,
        effect_calls=effect_calls,
    )
    _bind_explicit_zero_provider_callback(daemon)
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        first_result = first["implementation_result"]
        assert first_result["status"] == "blocked"
        first_attempt = daemon.get_attempt(first["attempt_id"])
        assert first_attempt is not None
        blocked = daemon.task_source.get(first_attempt.task_cid)
        assert blocked is not None and blocked.status == "blocked"
        assert blocked.body["completion_receipt"]["provider_invocation_count"] == 0
        assert blocked.body["completion_receipt"]["automatic_retry_admitted"] is False

        idle = daemon.run_once()
        assert idle["selection_idle_reason"] == "no_ready_tasks"
        assert idle.get("portal_failure_rearms") == []

        daemon.authority_mode = "quack"
        fail["enabled"] = False
        second = daemon.run_once()
        rearms = second.get("portal_failure_rearms") or []
        assert len(rearms) == 1
        assert rearms[0]["from_status"] == "blocked"
        assert rearms[0]["to_status"] == "retrying"
        assert rearms[0]["reason"] == "live_owner_zero_provider_portal_claim_failure"
        assert second["implementation_result"]["status"] == "succeeded"
        second_attempt = daemon.get_attempt(second["attempt_id"])
        assert second_attempt is not None
        assert second_attempt.attempt_number == first_attempt.attempt_number + 1
        assert provider_calls == [
            first_attempt.attempt_id,
            second_attempt.attempt_id,
        ]
        assert effect_calls == [first_attempt.task_cid]
        third = daemon.run_once()
        assert third["selection_idle_reason"] == "no_ready_tasks"
        assert third.get("portal_failure_rearms") == []
    finally:
        daemon.close()


def test_verified_source_rearms_new_zero_provider_settlement_after_lifetime_budget(
    tmp_path: Path,
) -> None:
    source = {"source_head": "a" * 40, "source_tree": "b" * 40}

    daemon = _open_daemon(
        tmp_path,
        session="session:portal-source-qualified-rearm",
        provider_fn=lambda _attempt: (_ for _ in ()).throw(
            DatabasePortalBridgeError("embedded-store claim without live owner")
        ),
    )
    _bind_explicit_zero_provider_callback(daemon)
    try:
        daemon.materialize_population(_population(1))
        daemon.authority_mode = "quack"
        first = daemon.run_once()
        assert first["implementation_result"]["status"] == "blocked"
        task_cid = daemon.get_attempt(first["attempt_id"]).task_cid
        first_settlement = daemon.task_source.get(task_cid).body[
            "completion_receipt"
        ]["settlement_id"]
        second = daemon.run_once()
        assert second.get("portal_failure_rearms")
        task = daemon.task_source.get(task_cid)
        if str(task.status or "") != "blocked":
            daemon.run_once()
            task = daemon.task_source.get(task_cid)
        assert task is not None and task.status == "blocked"
        second_settlement = task.body["completion_receipt"]["settlement_id"]
        assert second_settlement != first_settlement
        idle = daemon.run_once()
        assert idle["selection_idle_reason"] == "no_ready_tasks"
        assert idle.get("portal_failure_rearms") == []
        rearms = daemon.reconcile_recoverable_portal_failure_rearms(
            recovery_source_validator=lambda: dict(source),
        )
        assert len(rearms) == 1
        assert rearms[0]["reason"] == "live_owner_zero_provider_portal_claim_failure"
        assert rearms[0]["settlement_id"] == second_settlement
        retried = daemon.task_source.get(rearms[0]["task_cid"])
        assert retried is not None and retried.status == "retrying"
    finally:
        daemon.close()


def test_terminal_portal_failure_coordination_response_loss_replays_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:portal-settlement-replay",
        provider_fn=lambda _attempt: (_ for _ in ()).throw(
            DatabasePortalBridgeError("terminal bridge failure")
        ),
    )
    try:
        daemon.materialize_population(_population(1))
        original_settle = daemon.coordinator.fail_task_claim
        calls = {"count": 0}

        def lose_first_settlement(*args: object, **kwargs: object) -> object:
            calls["count"] += 1
            result = original_settle(*args, **kwargs)
            if calls["count"] == 1:
                raise RuntimeError("simulated post-commit response loss")
            return result

        monkeypatch.setattr(
            daemon.coordinator,
            "fail_task_claim",
            lose_first_settlement,
        )
        first = daemon.run_once()
        assert first["implementation_result"]["status"] == "blocked"
        assert calls["count"] == 2
        attempt = daemon.get_attempt(first["attempt_id"])
        assert attempt is not None and attempt.status == "failed"
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "blocked"
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None and claim.state.value == "released"
        coordination_attempt = daemon.coordinator.get_task_attempt(
            attempt.attempt_id
        )
        assert coordination_attempt is not None
        assert coordination_attempt.status.value == "failed"

        second = daemon.run_once()
        assert second["selection_idle_reason"] == "no_ready_tasks"
        assert second["portal_failure_reconciliations"] == []
    finally:
        daemon.close()


def test_terminal_portal_failure_control_cas_response_loss_never_repeats_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("terminal control-CAS failure")

    daemon = _open_daemon(
        tmp_path,
        session="session:portal-control-cas-loss",
        provider_fn=provider,
    )
    try:
        daemon.materialize_population(_population(1))
        original_cas = daemon._cas_task_status_database
        blocked_calls = {"count": 0}

        def lose_block_response(*args: object, **kwargs: object) -> object:
            result = original_cas(*args, **kwargs)
            if kwargs.get("new_status") == "blocked":
                blocked_calls["count"] += 1
                if blocked_calls["count"] == 1:
                    raise RuntimeError("simulated control CAS response loss")
            return result

        monkeypatch.setattr(
            daemon,
            "_cas_task_status_database",
            lose_block_response,
        )
        first = daemon.run_once()
        assert first["implementation_result"]["status"] == "blocked"
        assert blocked_calls["count"] == 1
        assert len(provider_calls) == 1
        attempt = daemon.get_attempt(first["attempt_id"])
        assert attempt is not None and attempt.status == "failed"

        second = daemon.run_once()
        assert second["selection_idle_reason"] == "no_ready_tasks"
        assert second["portal_failure_reconciliations"] == []
        assert len(provider_calls) == 1
    finally:
        daemon.close()


def test_terminal_portal_failure_settles_after_exact_lease_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("terminal expiry-window failure")

    daemon = _open_daemon(
        tmp_path,
        session="session:portal-expiry-window",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
        provider_fn=provider,
    )
    try:
        daemon.materialize_population(_population(1))
        original_settle = daemon.coordinator.fail_task_claim

        def expire_before_settlement(*args: object, **kwargs: object) -> object:
            now["ms"] = 7_000
            kwargs["now_ms"] = now["ms"]
            return original_settle(*args, **kwargs)

        monkeypatch.setattr(
            daemon.coordinator,
            "fail_task_claim",
            expire_before_settlement,
        )
        result = daemon.run_once()
        assert result["implementation_result"]["status"] == "blocked"
        assert len(provider_calls) == 1
        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt is not None and attempt.status == "failed"
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None and claim.state.value == "released"
        coordination_attempt = daemon.coordinator.get_task_attempt(
            attempt.attempt_id
        )
        assert coordination_attempt is not None
        assert coordination_attempt.status.value == "failed"
    finally:
        daemon.close()


def test_legacy_attempt_without_control_binding_quarantines_provider_replay(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    now = {"ms": 1_000}

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("legacy binding unavailable")

    daemon = _open_daemon(
        tmp_path,
        session="session:legacy-binding-quarantine",
        provider_fn=provider,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        # Rehearse a pre-repair DatabaseTaskAttempt@1 row.  Its original
        # control revision cannot be reconstructed after the fact.
        daemon._require_connection().execute(
            "UPDATE database_task_attempts SET body_json = '{}' "
            "WHERE attempt_id = ?",
            [attempt.attempt_id],
        )

        first = daemon.run_once()
        result = first["implementation_result"]
        assert result["settlement_failed"] is True
        assert result["provider_replay_quarantined"] is True
        assert len(provider_calls) == 1
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None and stored.status == "failed"
        failure_phase = [
            item
            for item in daemon.phase_history(attempt.attempt_id)
            if item["phase"] == "failed"
        ][-1]
        assert failure_phase["body"]["operation"] == (
            "quarantine_unsettled_portal_failure"
        )
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "in_progress"
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None and claim.state.value == "accepted"
        assert daemon._automatic_claim_exclusions() == {attempt.task_cid}

        second = daemon.run_once()
        assert second["selection_idle_reason"] == (
            "unsettled_portal_failure_quarantine"
        )
        assert second["unsettled_quarantine_task_cids"] == [attempt.task_cid]
        assert len(provider_calls) == 1

        # Lease expiry is not new evidence and cannot admit an identical
        # provider invocation. The quarantined negative-memory barrier remains
        # explicit until a separate operator control-plane repair supersedes it.
        now["ms"] = 7_000
        after_expiry = daemon.run_once()
        assert after_expiry["selection_idle_reason"] == (
            "unsettled_portal_failure_quarantine"
        )
        assert after_expiry["unsettled_quarantine_task_cids"] == [
            attempt.task_cid
        ]
        assert len(provider_calls) == 1

        daemon.close()
        replacement = _open_daemon(
            tmp_path,
            session="session:legacy-binding-quarantine-restart",
            provider_fn=provider,
            lease_ms=5_000,
            clock_ms=lambda: now["ms"],
        )
        try:
            after_restart = replacement.run_once()
            assert after_restart["selection_idle_reason"] == (
                "unsettled_portal_failure_quarantine"
            )
            assert after_restart["unsettled_quarantine_task_cids"] == [
                attempt.task_cid
            ]
            assert len(provider_calls) == 1
        finally:
            replacement.close()
    finally:
        daemon.close()


def test_completed_control_cas_is_recovered_from_prepared_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expiry after control CAS cannot expose an uncoordinated completion."""

    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:prepared-recovery",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)

        original_complete = daemon.coordinator.complete_task_claim

        def expire_at_promotion(*args: object, **kwargs: object) -> object:
            now["ms"] = 7_000
            kwargs["now_ms"] = now["ms"]
            return original_complete(*args, **kwargs)

        monkeypatch.setattr(
            daemon.coordinator,
            "complete_task_claim",
            expire_at_promotion,
        )
        with pytest.raises(DatabaseCoordinationError):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "b" * 64,
                    "argv": ["focused-validation"],
                },
            )

        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "completed"
        assert task.revision == 3
        readiness = daemon.coordinator.claimability(attempt.task_cid)
        assert readiness["claimable"] is False
        assert readiness["completion_status"] == "prepared"
        prepared = daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        )
        assert prepared is not None
        assert prepared["attempt_id"] == attempt.attempt_id
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.status == "running"
        assert stored.committed_phase == "validation"

        # Restore the ordinary method.  The next pass proves the exact control
        # receipt, promotes and settles the expired preparation, and repairs
        # the execution projection without rerunning provider/effect work.
        monkeypatch.setattr(
            daemon.coordinator,
            "complete_task_claim",
            original_complete,
        )
        result = daemon.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["recovered"] is True
        recovered = daemon.get_attempt(attempt.attempt_id)
        assert recovered is not None
        assert recovered.status == "succeeded"
        assert recovered.committed_phase == ATTEMPT_PHASE_COMPLETE
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        assert claim.state.value == "completed"
    finally:
        daemon.close()


def test_restart_recovers_prepared_control_completion_without_prior_expiry_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = first.commit_phase(attempt, phase)

        def crash_before_promotion(*args: object, **kwargs: object) -> object:
            now["ms"] = 7_000
            raise RuntimeError("simulated crash before coordination promotion")

        monkeypatch.setattr(
            first.coordinator,
            "complete_task_claim",
            crash_before_promotion,
        )
        with pytest.raises(RuntimeError, match="before coordination promotion"):
            first.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "e" * 64,
                    "argv": ["focused-validation"],
                },
            )
        task = first.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "completed"
        unswept = first.coordinator.get_task_claim(attempt.claim_id)
        assert unswept is not None
        assert unswept.state.value == "accepted"
    finally:
        first.close()

    replacement = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        result = replacement.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["recovered"] is True
        recovered = replacement.get_attempt(attempt.attempt_id)
        assert recovered is not None
        assert recovered.status == "succeeded"
        claim = replacement.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        assert claim.state.value == "completed"
        assert provider_calls == []
        assert effect_calls == []
    finally:
        replacement.close()


def test_promoted_completion_replays_after_local_phase_response_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:promotion-replay",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)

        original_commit_phase = daemon.commit_phase

        def lose_local_complete(
            current: DatabaseTaskAttempt | str,
            phase: str,
            **kwargs: object,
        ) -> DatabaseTaskAttempt:
            if phase == ATTEMPT_PHASE_COMPLETE:
                raise RuntimeError("simulated local COMPLETE outage")
            return original_commit_phase(current, phase, **kwargs)

        monkeypatch.setattr(daemon, "commit_phase", lose_local_complete)
        with pytest.raises(RuntimeError, match="simulated local COMPLETE outage"):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "d" * 64,
                    "argv": ["focused-validation"],
                },
            )
        promoted = daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        )
        assert promoted is not None
        assert promoted["status"] == "succeeded"
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        assert claim.state.value == "accepted"
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.status == "running"
        assert stored.committed_phase == "validation"

        monkeypatch.setattr(daemon, "commit_phase", original_commit_phase)
        result = daemon.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert result["implementation_result"] is None
        assert len(result["completion_reconciliations"]) == 1
        repaired = daemon.get_attempt(attempt.attempt_id)
        assert repaired is not None
        assert repaired.status == "succeeded"
        assert repaired.committed_phase == ATTEMPT_PHASE_COMPLETE
        assert provider_calls == []
        assert effect_calls == []
        settled = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert settled is not None
        assert settled.state.value == "released"
    finally:
        daemon.close()


def test_expired_preparation_without_control_cas_is_aborted_and_requeued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:prepared-abort",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)

        original_cas = daemon._cas_task_status_database

        def reject_control_completion(*args: object, **kwargs: object) -> object:
            raise RuntimeError("simulated control CAS outage")

        monkeypatch.setattr(
            daemon,
            "_cas_task_status_database",
            reject_control_completion,
        )
        with pytest.raises(RuntimeError, match="simulated control CAS outage"):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "c" * 64,
                    "argv": ["focused-validation"],
                },
            )
        assert daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        ) is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "in_progress"

        monkeypatch.setattr(daemon, "_cas_task_status_database", original_cas)
        now["ms"] = 7_000
        result = daemon.run_once()
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["status"] == "aborted"
        assert result["implementation_result"]["status"] == "succeeded"
        assert result["attempt_id"] != attempt.attempt_id
        old_attempt = daemon.get_attempt(attempt.attempt_id)
        assert old_attempt is not None
        assert old_attempt.status == "failed"
        final_completion = daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        )
        assert final_completion is not None
        assert final_completion["status"] == "succeeded"
        assert final_completion["attempt_id"] == result["attempt_id"]
        completed = daemon.task_source.get(attempt.task_cid)
        assert completed is not None
        assert completed.status == "completed"
    finally:
        daemon.close()


def test_task_claim_settlement_authority_loss_is_not_suppressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:settlement-loss")
    try:
        daemon.materialize_population(_population(1))

        def reject_settlement(*args: object, **kwargs: object) -> object:
            raise DatabaseCoordinationError("simulated settlement authority loss")

        monkeypatch.setattr(
            daemon.coordinator,
            "settle_task_claim",
            reject_settlement,
        )
        with pytest.raises(
            DatabaseCoordinationError,
            match="simulated settlement authority loss",
        ):
            daemon.run_once()
    finally:
        daemon.close()


@pytest.mark.parametrize(
    ("restart_ms", "expected_claim_state"),
    ((2_000, "released"), (7_000, "completed")),
)
def test_restart_settles_promoted_completion_after_local_complete_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restart_ms: int,
    expected_claim_state: str,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))

        def crash_before_settlement(*args: object, **kwargs: object) -> object:
            raise RuntimeError("simulated crash before claim settlement")

        monkeypatch.setattr(
            first.coordinator,
            "settle_task_claim",
            crash_before_settlement,
        )
        with pytest.raises(RuntimeError, match="before claim settlement"):
            first.run_once()
        row = first._require_connection().execute(
            """
            SELECT attempt_id, claim_id FROM database_task_attempts
            WHERE status = 'succeeded'
            """
        ).fetchone()
        assert row is not None
        attempt_id, claim_id = str(row[0]), str(row[1])
        unsettled = first.coordinator.get_task_claim(claim_id)
        assert unsettled is not None
        assert unsettled.state.value == "accepted"
    finally:
        first.close()

    now["ms"] = restart_ms
    replacement = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        result = replacement.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["status"] == "succeeded"
        settled = replacement.coordinator.get_task_claim(claim_id)
        assert settled is not None
        assert settled.state.value == expected_claim_state
        local = replacement.get_attempt(attempt_id)
        assert local is not None
        assert local.status == "succeeded"
        assert replacement.coordinator.list_unsettled_task_completions() == []
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
    finally:
        replacement.close()


def test_automatic_run_once_never_claims_manual_or_review_only_task(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path)
    try:
        population = _population(2)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["completion"] = "manual"
        tasks[1]["review_only"] = True
        daemon.materialize_population(population)
        result = daemon.run_once()
        assert result["unchanged"] is True
        assert result["selection_idle_reason"] == "no_ready_tasks"
        assert daemon.list_running_attempts() == []
        assert daemon.coordinator.get_task_claim("claim:missing") is None
        for task_cid in ("task:cid:001", "task:cid:002"):
            task = daemon.task_source.get(task_cid)
            assert task is not None
            assert task.status == "ready"

        # The coordinator still exposes the task to a separately authorized
        # trusted manual-seal path; only automatic daemon dispatch is excluded.
        direct = daemon.coordinator.claim_task(
            task_cid="task:cid:001",
            owner_session_id="session:trusted-manual-seal",
            now_ms=daemon._now_ms(),
        )
        assert direct.task_cid == "task:cid:001"
    finally:
        daemon.close()


def test_parse_args_accepts_database_authority_flags() -> None:
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            "/tmp/control.duckdb",
            "--owner-session-id",
            "session:cli",
            "--once",
        ]
    )
    assert args.task_source_kind == "duckdb"
    assert args.authority_mode == "embedded"
    assert Path(args.database_path) == Path("/tmp/control.duckdb")
    assert args.owner_session_id == "session:cli"
    paths = resolve_database_implementation_paths(args)
    assert paths["database_path"] == Path("/tmp/control.duckdb")


def test_runner_builds_database_daemon_without_json_projections(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "control.duckdb"
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            str(database_path),
            "--todo-path",
            str(tmp_path / "unused.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "dqp",
            "--task-prefix",
            "DQP-",
            "--once",
        ]
    )
    daemon = build_database_implementation_daemon_from_args(
        args,
        owner_session_id="session:runner",
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert daemon.state_path is None
        assert daemon.events_path is None
        assert daemon.projections_required() is False
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        assert result["authority_mode"] == "embedded"
        assert result["markdown_status_writes"] == 0
    finally:
        daemon.close()


def test_database_runner_passes_exact_strict_lane_binding(tmp_path: Path) -> None:
    state_dir = tmp_path / "state" / "lane-1"
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            str(tmp_path / "control.duckdb"),
            "--state-dir",
            str(state_dir),
            "--state-prefix",
            "dqp-lane-1",
            "--task-shard-count",
            "3",
            "--task-shard-index",
            "1",
            "--strict-task-sharding",
            "--once",
        ]
    )
    daemon = build_database_implementation_daemon_from_args(args)
    try:
        assert daemon.task_shard_count == 3
        assert daemon.task_shard_index == 1
        assert daemon.strict_task_sharding is True
        assert daemon.execution_state_dir == state_dir
        assert daemon.execution_state_prefix == "dqp-lane-1"
        assert daemon.execution_path == (
            state_dir / "dqp-lane-1.execution.shard-0001-of-0003.duckdb"
        )
    finally:
        daemon.close()


def test_portal_runner_passes_exact_strict_lane_binding(tmp_path: Path) -> None:
    state_dir = tmp_path / "state" / "lane-2"
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            str(tmp_path / "control.duckdb"),
            "--state-dir",
            str(state_dir),
            "--state-prefix",
            "dqp-lane-2",
            "--task-shard-count",
            "4",
            "--task-shard-index",
            "2",
            "--strict-task-sharding",
            "--once",
        ]
    )
    daemon, _context = build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert daemon.task_shard_count == 4
        assert daemon.task_shard_index == 2
        assert daemon.strict_task_sharding is True
        assert daemon.execution_state_dir == state_dir
        assert daemon.execution_state_prefix == "dqp-lane-2"
        assert daemon.execution_path == (
            state_dir / "dqp-lane-2.execution.shard-0002-of-0004.duckdb"
        )
    finally:
        daemon.close()


def test_direct_cli_passes_exact_strict_lane_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_daemon as daemon_module,
    )

    captured: dict[str, object] = {}

    class RecordingDatabaseDaemon:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run_once(self) -> dict[str, object]:
            return {"unchanged": True, "selection_idle_reason": "test"}

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        daemon_module,
        "DatabaseImplementationDaemon",
        RecordingDatabaseDaemon,
    )
    state_dir = tmp_path / "state" / "lane-3"
    daemon_module.main(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            str(tmp_path / "control.duckdb"),
            "--state-dir",
            str(state_dir),
            "--state-prefix",
            "dqp-lane-3",
            "--task-shard-count",
            "4",
            "--task-shard-index",
            "3",
            "--strict-task-sharding",
            "--once",
        ]
    )
    assert captured["state_dir"] == state_dir
    assert captured["state_prefix"] == "dqp-lane-3"
    assert captured["task_shard_count"] == 4
    assert captured["task_shard_index"] == 3
    assert captured["strict_task_sharding"] is True
    heartbeat_path = database_daemon_pass_heartbeat_path(
        state_dir=state_dir,
        state_prefix="dqp-lane-3",
    )
    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat["schema"] == DATABASE_DAEMON_PASS_HEARTBEAT_SCHEMA
    assert heartbeat["sequence"] == 1
    assert heartbeat["process_birth"]["pid"] > 0
    assert heartbeat["task_shard_count"] == 4
    assert heartbeat["task_shard_index"] == 3
    assert heartbeat["strict_task_sharding"] is True
    assert heartbeat["selection_idle_reason"] == "test"
    assert "implementation_result" not in heartbeat


def test_runner_portal_builder_selects_database_daemon(tmp_path: Path) -> None:
    database_path = tmp_path / "control.duckdb"
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            str(database_path),
            "--todo-path",
            str(tmp_path / "board.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "dqp",
            "--task-prefix",
            "DQP-",
            "--once",
        ]
    )
    daemon, context = build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert context.state_path.name.startswith("dqp_")
        daemon.materialize_population(_population(2))
        first = daemon.claim_next()
        second = daemon.claim_next()
        # Single session claims one at a time via claim_ready; second claim is
        # a different task while the first remains leased.
        assert first is not None
        assert second is not None
        assert first.task_cid != second.task_cid
    finally:
        daemon.close()


def _database_bound_portal_daemon(
    tmp_path: Path,
    *,
    database_attempt_number: object = 10,
) -> tuple[PortalImplementationDaemon, object]:
    task_path = tmp_path / "database-attempt.runtime.todo.md"
    task_path.write_text(
        "\n".join(
            (
                "# Database attempt projection (non-authoritative)",
                "",
                "## LGSWF-004 Preserve database lifecycle identity",
                "",
                "- Status: ready",
                "- Completion: auto",
                "- Priority: P0",
                "- Track: implementation",
                "- Depends on:",
                "- Outputs: inventory/result.json",
                "- Validation: python3 -m pytest focused.py",
                "- Acceptance: Focused validation passes",
                "- Database task CID: task:cid:004",
                "- Database attempt ID: attempt:010",
                "- Database claim ID: claim:010",
                f"- Database attempt number: {database_attempt_number}",
                "- Projection authority: false",
                "",
            )
        ),
        encoding="utf-8",
    )
    daemon = PortalImplementationDaemon(
        todo_path=task_path,
        state_path=tmp_path / "task-state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## LGSWF-",
        max_task_attempts=4,
        worktree_pool_enabled=False,
    )
    [task] = daemon._load_tasks()
    return daemon, task


def _bind_database_attempt_authority(
    daemon: PortalImplementationDaemon,
    **overrides: object,
) -> object:
    arguments: dict[str, object] = {
        "task_id": "LGSWF-004",
        "database_task_cid": "task:cid:004",
        "database_attempt_id": "attempt:010",
        "database_claim_id": "claim:010",
        "database_attempt_number": 10,
        "database_binding_id": "sha256:" + "b" * 64,
    }
    arguments.update(overrides)
    return daemon.bind_database_attempt_authority(**arguments)


def test_portal_database_attempt_binder_accepts_only_exact_projection(
    tmp_path: Path,
) -> None:
    daemon, task = _database_bound_portal_daemon(tmp_path)

    receipt = _bind_database_attempt_authority(daemon)

    assert receipt["task_id"] == task.task_id
    assert receipt["database_task_cid"] == "task:cid:004"
    assert receipt["database_attempt_id"] == "attempt:010"
    assert receipt["database_claim_id"] == "claim:010"
    assert receipt["database_attempt_number"] == 10
    assert receipt["database_binding_id"] == "sha256:" + "b" * 64
    assert receipt["canonical_task_cid"] == daemon._canonical_ref(task)
    assert receipt["worktree_lifecycle_attempt_prefix"] == (
        (1 << 52) | (10 << 16)
    )
    assert receipt["worktree_lifecycle_attempt_prefix_only"] is True
    assert receipt["portal_attempt_authority"] is False
    assert receipt["completion_authority"] is False
    assert receipt["quarantine_authority"] is False


def test_database_attempt_ordinal_namespaces_lifecycle_not_portal_retries(
    tmp_path: Path,
) -> None:
    daemon, task = _database_bound_portal_daemon(tmp_path)
    _bind_database_attempt_authority(daemon)
    state = PortalTaskState()

    first_portal_attempt = daemon._task_attempt(state, task)
    daemon._record_task_attempt(state, task, first_portal_attempt)
    second_portal_attempt = daemon._task_attempt(state, task)

    assert daemon.max_task_attempts == 4
    assert first_portal_attempt == 1
    assert second_portal_attempt == 2
    assert daemon._task_attempt_count(state, task) == 1
    first_lifecycle_attempt = daemon._worktree_lifecycle_attempt(
        task,
        first_portal_attempt,
    )
    second_lifecycle_attempt = daemon._worktree_lifecycle_attempt(
        task,
        second_portal_attempt,
    )
    assert first_lifecycle_attempt == (1 << 52) | (10 << 16) | 1
    assert second_lifecycle_attempt == (1 << 52) | (10 << 16) | 2
    assert first_lifecycle_attempt != second_lifecycle_attempt
    assert first_lifecycle_attempt not in {
        first_portal_attempt,
        second_portal_attempt,
    }
    with pytest.raises(
        RuntimeError,
        match="does not bind the lifecycle task",
    ):
        daemon._worktree_lifecycle_attempt_for_identity(
            task_id=task.task_id,
            canonical_task_cid="sha256:" + "f" * 64,
            portal_attempt=first_portal_attempt,
        )


def test_database_lifecycle_pair_allows_retry_after_prior_quarantine(
    tmp_path: Path,
) -> None:
    daemon, task = _database_bound_portal_daemon(tmp_path)
    _bind_database_attempt_authority(daemon)
    first_attempt = daemon._worktree_lifecycle_attempt(task, 1)
    second_attempt = daemon._worktree_lifecycle_attempt(task, 2)
    proc_root = tmp_path / "empty-proc"
    proc_root.mkdir()
    store = WorktreeLifecycleStore(
        repo_root=tmp_path,
        store_dir=tmp_path / "lifecycle",
        startup_grace_seconds=0.0,
        proc_root=proc_root,
    )
    first_workspace = tmp_path / "worktrees" / "first"
    first = store.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=daemon._canonical_ref(task),
        attempt=first_attempt,
        lane_id="database-attempt-10",
        workspace_path=first_workspace,
        branch="implementation/database-attempt-10-local-1",
        merge_target="main",
        state_dir=str(tmp_path / "attempt-10"),
        owner=ProcessBirthIdentity(
            pid=2**30 - 17,
            start_time_ticks=1,
            boot_id="dead-database-attempt",
            parent_pid=1,
        ),
    )
    exact = {
        "expected_record_id": first.record_id,
        "expected_fence": first.fence,
        "expected_lease_id": first.lease_id,
        "expected_task_id": first.task_id,
        "expected_canonical_task_cid": first.canonical_task_cid,
        "expected_attempt": first.attempt,
        "expected_branch": first.branch,
        "expected_merge_target": first.merge_target,
        "expected_repo_root": first.repo_root,
        "expected_state_dir": first.state_dir,
    }
    store.quarantine_exact_dead_owner(
        first_workspace,
        fence_authority={
            "schema": "test-database-attempt-fence@1",
            "database_attempt_number": 10,
            "portal_attempt": 1,
            "provider_dispatched": False,
        },
        **exact,
    )

    second = store.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=daemon._canonical_ref(task),
        attempt=second_attempt,
        lane_id="database-attempt-10",
        workspace_path=tmp_path / "worktrees" / "second",
        branch="implementation/database-attempt-10-local-2",
        merge_target="main",
        state_dir=str(tmp_path / "attempt-10"),
    )

    assert store.load_quarantine(first_workspace) is not None
    assert second.attempt == second_attempt
    assert second.attempt != first.attempt


def test_database_lifecycle_pair_preserves_unbound_attempts_and_safe_bounds(
    tmp_path: Path,
) -> None:
    unbound_root = tmp_path / "unbound"
    unbound_root.mkdir()
    unbound, unbound_task = _database_bound_portal_daemon(unbound_root)
    assert unbound._worktree_lifecycle_attempt_for_identity(
        task_id=unbound_task.task_id,
        canonical_task_cid=unbound._canonical_ref(unbound_task),
        portal_attempt=1 << 16,
    ) == (1 << 16)

    bound_root = tmp_path / "bound"
    bound_root.mkdir()
    bound, bound_task = _database_bound_portal_daemon(bound_root)
    _bind_database_attempt_authority(bound)
    maximum = bound._worktree_lifecycle_attempt(bound_task, (1 << 16) - 1)
    assert maximum < 1 << 53
    with pytest.raises(RuntimeError, match="must fit unsigned 16-bit"):
        bound._worktree_lifecycle_attempt(bound_task, 1 << 16)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"task_id": "LGSWF-999"}, "one exact projected task"),
        (
            {"database_task_cid": "task:cid:wrong"},
            "disagrees with the projected task",
        ),
        (
            {"database_attempt_id": "attempt:wrong"},
            "disagrees with the projected task",
        ),
        (
            {"database_claim_id": "claim:wrong"},
            "disagrees with the projected task",
        ),
        (
            {"database_attempt_number": 11},
            "disagrees with the projected task",
        ),
    ),
)
def test_portal_database_attempt_binder_rejects_identity_or_metadata_mismatch(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    daemon, _task = _database_bound_portal_daemon(tmp_path)

    with pytest.raises(RuntimeError, match=message):
        _bind_database_attempt_authority(daemon, **override)

    assert daemon._database_attempt_authority is None


@pytest.mark.parametrize(
    "invalid_ordinal",
    (0, -1, 1 << 32, True, 1.0, "10", None),
)
def test_portal_database_attempt_binder_rejects_invalid_ordinal(
    tmp_path: Path,
    invalid_ordinal: object,
) -> None:
    daemon, _task = _database_bound_portal_daemon(tmp_path)

    with pytest.raises(
        ValueError,
        match="ordinal must be a positive u32 integer",
    ):
        _bind_database_attempt_authority(
            daemon,
            database_attempt_number=invalid_ordinal,
        )

    assert daemon._database_attempt_authority is None


def test_portal_rearm_new_source_retains_once_per_source_and_settlement_budget(tmp_path):
    def provider(_attempt):
        raise DatabasePortalBridgeError("closed bridge failure")
    daemon = _open_daemon(tmp_path, session="session:source-recovery", provider_fn=provider)
    source = {"source_head": "a" * 40, "source_tree": "b" * 40}
    _bind_explicit_zero_provider_callback(daemon)
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        daemon.authority_mode = "quack"
        second = daemon.run_once()
        assert len(second["portal_failure_rearms"]) == 1
        assert second["implementation_result"]["status"] == "blocked"
        assert daemon.reconcile_recoverable_portal_failure_rearms() == []
        task = daemon.task_source.get(daemon.get_attempt(first["attempt_id"]).task_cid)
        def rejected_source():
            raise ValueError("sealed admission rejected")
        with pytest.raises(ValueError, match="sealed admission rejected"):
            daemon.reconcile_recoverable_portal_failure_rearms(recovery_source_validator=rejected_source)
        assert daemon.task_source.get(task.task_cid).revision == task.revision
        rearmed = daemon.reconcile_recoverable_portal_failure_rearms(recovery_source_validator=lambda: source)
        assert len(rearmed) == 1
        assert rearmed[0]["accepted_recovery_source"] == source
        third = daemon.run_once()
        assert third["implementation_result"]["status"] == "blocked"
        assert daemon.reconcile_recoverable_portal_failure_rearms(recovery_source_validator=lambda: source) == []
        other = {"source_head": "c" * 40, "source_tree": "d" * 40}
        assert len(daemon.reconcile_recoverable_portal_failure_rearms(recovery_source_validator=lambda: other)) == 1
    finally:
        daemon.close()


@pytest.mark.parametrize("later_callback_proof", [True, False, None, "true"])
def test_portal_rearm_same_source_admits_later_settlement_within_attempt_budget(
    tmp_path: Path, later_callback_proof,
) -> None:
    class _BoundedProvider:
        max_task_attempts = 4
        callback_proof = True

        def provider(self, _attempt: DatabaseTaskAttempt) -> dict[str, object]:
            raise DatabasePortalBridgeError("closed bridge failure")

        def zero_provider_failure_rearm_ready(self, attempt):
            assert attempt.status == "failed"
            assert attempt.committed_phase == "failed"
            return self.callback_proof

    holder = _BoundedProvider()
    daemon = _open_daemon(
        tmp_path,
        session="session:attempt-budget-recovery",
        provider_fn=holder.provider,
    )
    source = {"source_head": "a" * 40, "source_tree": "b" * 40}
    try:
        daemon.materialize_population(_population(1))
        daemon.run_once()
        daemon.authority_mode = "quack"
        first = daemon.reconcile_recoverable_portal_failure_rearms(
            recovery_source_validator=lambda: source
        )
        assert len(first) == 1
        second_pass = daemon.run_once()
        assert second_pass["implementation_result"]["status"] == "blocked"
        attempt = daemon.get_attempt(second_pass["attempt_id"])
        before = daemon.task_source.get(attempt.task_cid)
        holder.callback_proof = later_callback_proof
        later = daemon.reconcile_recoverable_portal_failure_rearms(
            recovery_source_validator=lambda: source
        )
        after = daemon.task_source.get(attempt.task_cid)
        if later_callback_proof is True:
            assert len(later) == 1
            assert later[0]["accepted_recovery_source"] == source
            assert later[0]["settlement_id"] != first[0]["settlement_id"]
            assert after.status == "retrying"
            assert after.revision == before.revision + 1
        else:
            assert later == []
            assert after.status == "blocked"
            assert after.revision == before.revision
            assert after.body == before.body
        assert daemon.get_attempt(attempt.attempt_id).status == "failed"
    finally:
        daemon.close()


def test_production_protected_rearm_requires_native_fence_precondition(tmp_path):
    def provider(attempt):
        raise DatabasePortalBridgeDeferred("implementation_protected_path_mutated")
    daemon = _open_daemon(tmp_path, session="session:protected-fence", provider_fn=provider)
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        attempt = daemon.get_attempt(first["attempt_id"])
        daemon.require_real_execution = True
        assert daemon.reconcile_recoverable_portal_failure_rearms() == []
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"
        assert daemon.get_attempt(attempt.attempt_id).status == "failed"
    finally:
        daemon.close()
