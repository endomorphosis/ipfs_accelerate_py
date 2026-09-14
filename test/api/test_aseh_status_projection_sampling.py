"""Status sampling preserves canonical evidence without per-task round trips."""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import build_server
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import reset_quack_transport_cache
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    QuackCapabilityStatus, probe_quack_capabilities,
)
from scripts import run_agent_supervisor_efficiency_state_hardening as operator


@pytest.fixture(scope="module")
def portfolio(tmp_path_factory):
    return _create_portfolio(tmp_path_factory.mktemp("portfolio") / "control.duckdb")


def _create_portfolio(database, *, task_count=40, frontier="terminal"):
    with DatabaseTaskSource(database) as source:
        source.materialize({
            "repository_tree_id": "tree:status-sampling-fixture",
            "objectives": [
                {"goal_id": f"ASEH-G{index:03d}", "goal_cid": f"goal:sample:{index}",
                 "objective_id": "objective:sample", "title": f"Goal {index}"}
                for index in range(9)
            ],
            "taskboard": [
                {"task_id": f"ASEH-{index:03d}", "task_cid": f"task:sample:{index:03d}",
                 "goal_cid": f"goal:sample:{index % 9}",
                 "status": ("completed" if frontier == "terminal" else
                            "in_progress" if frontier == "active" and index == 0 else "ready"),
                 "dependencies": [f"task:sample:{index - 1:03d}"] if index else [],
                 "owning_repository": "repository:sample", "base_revision": "a" * 40,
                 "source_forest_cid": "forest:sample"}
                for index in range(task_count)
            ],
        })
        if frontier == "cooldown":
            source.record_queue_backoff(task_cid="task:sample:000", delay_ms=600_000, reason="private-fixture")
        snapshot = source.snapshot().to_dict()
        tasks = source.list_tasks(limit=100).tasks
        projection = dict(source.plan_projection())
        status = source.intent.snapshot()
        semantics = {row["goal_alias"]: operator._immutable_goal_record(row)
                     for row in projection["goals"]}
    return database, snapshot, tasks, projection, status, semantics


def project(fixture, *, projection=None, status=None):
    _database, snapshot, _tasks, original, fence, _semantics = fixture
    return operator._status_portfolio_from_plan_projection(
        original if projection is None else projection, fence if status is None else status,
        repository_tree_id=snapshot["repository_tree_id"], plan_root_cid=snapshot["plan_root_cid"],
    )


def test_full_projection_matches_existing_snapshot_and_task_records(portfolio):
    snapshot, tasks = project(portfolio)
    assert snapshot == portfolio[1]
    assert tasks == portfolio[2]
    assert len(tasks) == 40 and len(portfolio[3]["goals"]) == 9
    assert snapshot["terminal"] is True
    assert all(task.status == "completed" for task in tasks)


@pytest.mark.parametrize("corruption", [
    "task_missing", "task_status", "task_revision", "dependency_missing", "goal_missing",
    "plan_missing", "objective_missing", "watermark", "digest", "duplicate_task",
    "oversized", "unordered", "boolean_revision", "extra_field",
])
def test_mixed_truncated_or_malformed_projection_never_supplies_status(portfolio, corruption):
    projection = copy.deepcopy(portfolio[3])
    if corruption == "task_missing": projection["tasks"].pop()
    elif corruption == "task_status": projection["tasks"][0]["status"] = "ready"
    elif corruption == "task_revision": projection["tasks"][0]["revision"] += 1
    elif corruption == "dependency_missing": projection["tasks"][-1]["dependencies"] = []
    elif corruption == "goal_missing": projection["goals"].pop()
    elif corruption == "plan_missing": projection["plans"].pop()
    elif corruption == "objective_missing": projection["objectives"].pop()
    elif corruption == "watermark": projection["event_watermark"] += 1
    elif corruption == "duplicate_task": projection["tasks"].append(copy.deepcopy(projection["tasks"][0]))
    elif corruption == "oversized": projection["tasks"] *= 3
    elif corruption == "unordered": projection["tasks"].reverse()
    elif corruption == "boolean_revision": projection["tasks"][0]["revision"] = True
    elif corruption == "extra_field": projection["extra"] = "not canonical"
    projection.pop("projection_cid")
    projection["projection_cid"] = operator.content_identity(projection)
    if corruption == "digest": projection["projection_cid"] = "forged"
    with pytest.raises(operator.OperatorError): project(portfolio, projection=projection)


def test_independent_snapshot_cannot_be_replaced_by_projection_self_attestation(portfolio):
    with pytest.raises(operator.OperatorError, match="canonical snapshot"):
        project(portfolio, status=replace(portfolio[4], event_watermark=portfolio[4].event_watermark + 1))


def test_real_quack_complete_portfolio_equivalence_and_bounded_round_trips(portfolio, tmp_path, monkeypatch):
    _assert_real_quack_portfolio(portfolio, tmp_path, monkeypatch, frontier="terminal")


@pytest.mark.parametrize("frontier", ["ready", "cooldown", "active"])
def test_real_quack_ready_dependency_and_cooldown_semantics(frontier, tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("frontier")
    fixture = _create_portfolio(root / "control.duckdb", task_count=4, frontier=frontier)
    _assert_real_quack_portfolio(fixture, root, monkeypatch, frontier=frontier)


def _assert_real_quack_portfolio(portfolio, tmp_path, monkeypatch, *, frontier):
    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip(f"reviewed preinstalled Quack unavailable: {capability.status.value}")
    database, snapshot, expected_tasks, projection, _status, semantics = portfolio
    # Keep the real AF_UNIX broker path below the kernel limit, independent of
    # the descriptive pytest function name.
    tmp_path = database.parent
    owner_dir = tmp_path / "owner"
    bootstrap = {
        "schema": operator.BOOTSTRAP_SCHEMA, "source_head": "commit:status-sampling-fixture",
        "repository_tree_id": snapshot["repository_tree_id"], "plan_root_cid": snapshot["plan_root_cid"],
        "source_forest": {}, "source_identities": {}, "database_task_source_receipt": {},
        "snapshot": snapshot, "integrity": {"goal_records": semantics},
        "initial_ready_task_ids": [], "bootstrap_validation": {},
        "recovered_after_interrupted_materialization": False, "authority": {}, "ducklake_projection": {},
    }
    bootstrap["bootstrap_receipt_id"] = operator._identity(bootstrap)
    paths = {"runtime": tmp_path, "database": database, "owner": owner_dir,
             "bootstrap_receipt": tmp_path / "bootstrap.json"}
    operator._atomic_json(paths["bootstrap_receipt"], bootstrap)
    server = build_server(database_path=database, state_dir=owner_dir,
                          repository_root=tmp_path, port=0, store_id=str(database),
                          secret_handle="handle:status-sampling-test")
    identity = server.start()
    try:
        for key, value in dict(server.start_supervisor_grant_broker()).items():
            monkeypatch.setenv(key, str(value))
        monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(database))
        monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", str(identity.generation))
        monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", raising=False)
        board = SimpleNamespace(resolved_database_program=lambda: SimpleNamespace(quack_endpoint=identity.listen_uri))
        with DatabaseTaskSource(identity.listen_uri, install_schema=False,
                                repository_tree_id=snapshot["repository_tree_id"],
                                plan_root_cid=snapshot["plan_root_cid"]) as source:
            with source.intent._connection(write=False) as connection:
                connection_type = type(connection)
            original_execute = connection_type.execute
            statements = []
            def counted(connection, sql, *args, **kwargs):
                if getattr(connection, "_quack_mutation_binding", None):
                    statements.append(str(sql))
                return original_execute(connection, sql, *args, **kwargs)
            monkeypatch.setattr(connection_type, "execute", counted)
            started = time.monotonic()
            # The prior sampler performs these existing public operations,
            # including both complete task enumerations, on the same owner.
            expected_snapshot = source.snapshot().to_dict()
            expected_page = source.list_tasks(limit=100)
            expected_ready = [task.task_alias for task in source.ready_tasks(limit=100).tasks]
            expected_goals = {alias: operator._immutable_goal_record(source.get_goal(record["goal_cid"]))
                              for alias, record in semantics.items()}
            expected_edges = sorted((dict(row) for row in source.list_goal_edges(limit=100)), key=operator._goal_edge_sort_key)
            expected_plan = operator._immutable_plan_record(source.get_plan(snapshot["plan_root_cid"]))
            expected_projection = source.plan_projection(task_cids=[task.task_cid for task in expected_page.tasks])
            expected_queues = {}
            if not expected_ready and not any(task.status in operator.ACTIVE_STATUSES for task in expected_tasks):
                for task in expected_tasks:
                    if task.status in operator.READY_STATUSES:
                        entry = source.get_queue_entry(task.task_cid)
                        if entry and entry.retry_not_before_ms > int(time.time() * 1000):
                            expected_queues[task.task_alias] = entry.to_dict()
            baseline_elapsed = time.monotonic() - started
            baseline_count = len(statements)
        with operator._LIVE_REPLAY_CACHE_LOCK:
            operator._LIVE_REPLAY_CACHE.clear()
        statements.clear()
        started = time.monotonic()
        result = operator._broker_status_query(board, paths, owner_status=server.status())
        candidate_elapsed = time.monotonic() - started
        candidate_count = len(statements)
        assert result["snapshot"] == expected_snapshot == snapshot
        assert result["goal_records"] == expected_goals == semantics
        assert result["goal_edges"] == expected_edges
        assert result["plan_record"] == expected_plan
        assert result["ready_task_ids"] == expected_ready
        assert expected_ready == (["ASEH-000"] if frontier == "ready" else [])
        assert result["task_statuses"] == {task.task_alias: task.status for task in expected_tasks}
        assert result["task_revisions"] == {task.task_alias: task.revision for task in expected_tasks}
        assert result["task_dependencies"] == {task.task_alias: list(task.dependencies) for task in expected_tasks}
        assert result["task_cids"] == {task.task_alias: task.task_cid for task in expected_tasks}
        assert result["task_authority_spec_cids"] == operator._task_authority_spec_cids(expected_projection)
        assert result["objective_record"] == operator._objective_record_from_projection(expected_projection)
        assert result["queue_entries"] == expected_queues
        assert result["delayed_ready_task_ids"] == (["ASEH-000"] if frontier == "cooldown" else [])
        assert result["active_count"] == (1 if frontier == "active" else 0)
        assert result["task_owner_bindings"] == {
            task.task_alias: {key: task.body.get(key) for key in (
                "owning_repository", "base_revision", "base_repository_tree_id", "source_forest_cid", "owner_source_identity"
            )} for task in expected_tasks
        }
        assert result["projection_matches_events"] is True
        assert result["goal_lifecycle"]["available"] is True
        assert baseline_count > candidate_count and candidate_count < 65
        if frontier == "terminal": assert baseline_count >= 400
        # Preserve the existing ready selector's full-record reads; only the
        # duplicate complete-portfolio enumerations are removed.
        assert sum("WHERE task_cid = ? OR task_alias = ?" in sql for sql in statements) == len(expected_ready)
        assert server._connection.execute("SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events").fetchone()[0] == snapshot["event_cursor"]
        verified_replicas = []
        original_verify = operator._published_replica_bytes_still_match
        def verified(binding):
            verified_replicas.append(dict(binding))
            return original_verify(binding)
        monkeypatch.setattr(operator, "_published_replica_bytes_still_match", verified)
        statements.clear()
        started = time.monotonic()
        warmed = operator._broker_status_query(board, paths, owner_status=server.status())
        warmed_elapsed = time.monotonic() - started
        assert len(verified_replicas) == 1
        assert {key: value for key, value in warmed.items() if key != "query_started_at_ms"} == {
            key: value for key, value in result.items() if key != "query_started_at_ms"
        }
        assert len(statements) == candidate_count
        assert server._connection.execute("SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events").fetchone()[0] == snapshot["event_cursor"]
        print(json.dumps({"frontier": frontier, "baseline_quack_statements": baseline_count, "candidate_quack_statements": candidate_count,
                          "baseline_seconds": baseline_elapsed, "candidate_seconds_including_first_shadow_replay": candidate_elapsed,
                          "candidate_seconds_with_reverified_cached_replay": warmed_elapsed}, sort_keys=True))
    finally:
        reset_quack_transport_cache()
        server.stop()


@pytest.mark.parametrize("failure_kind", ["one_race", "persistent_race", "malformed"])
def test_projection_race_reobserves_whole_query_with_original_freshness_clock(
    portfolio, tmp_path, monkeypatch, failure_kind,
):
    from test.api.test_agent_supervisor_configured_typed_grant_handoff import _aseh_health_fixture

    board, paths, prior = _aseh_health_fixture(tmp_path, observed_at=100.0, lane_mtime_ns=100_000_000_000)
    board.resolved_database_program = lambda: SimpleNamespace(quack_endpoint="quack:127.0.0.1:45123")
    clock, calls = [100.0], []
    def query(*_args, **_kwargs):
        calls.append(clock[0])
        clock[0] += 40.0
        if failure_kind == "malformed":
            broken = copy.deepcopy(portfolio[3]);broken["projection_cid"] = "forged"
            project(portfolio, projection=broken)
        if failure_kind == "persistent_race" or len(calls) == 1:
            project(portfolio, status=replace(portfolio[4], event_watermark=portfolio[4].event_watermark + 1))
        return {"available": True}
    monkeypatch.setattr(operator, "time", SimpleNamespace(
        time=lambda: clock[0], monotonic_ns=lambda: int(clock[0] * 1e9), sleep=lambda _seconds: None,
    ))
    monkeypatch.setattr(operator, "_broker_status_query", query)
    monkeypatch.setattr(operator, "_lane_status_observations", lambda *_args, **_kwargs: [])
    sample = operator._status_sample(
        board, paths, SimpleNamespace(status=lambda: prior["owner_status"]),
        SimpleNamespace(pid=os.getpid(), poll=lambda: None),
    )
    expected_calls = 2 if failure_kind == "one_race" else operator.STATUS_REPLICA_STABILITY_ATTEMPTS if failure_kind == "persistent_race" else 1
    assert len(calls) == expected_calls
    assert sample["authority"]["available"] is (failure_kind == "one_race")
    # Even the successful query took 40 seconds. Its publication must remain
    # older than the unchanged 30-second TTL, never redated to completion.
    assert sample["observed_at"] == calls[-1] == clock[0] - 40.0


def test_fresh_publication_does_not_extend_thirty_second_receipt_ttl(tmp_path, monkeypatch):
    from test.api.test_agent_supervisor_configured_typed_grant_handoff import _aseh_health_fixture

    board, paths, _sample = _aseh_health_fixture(tmp_path, observed_at=100.0, lane_mtime_ns=100_000_000_000)
    board.payload["check_interval_seconds"] = 10
    bootstrap = operator._secure_runtime_json(paths["bootstrap_receipt"], max_bytes=operator.STATUS_RECEIPT_MAX_BYTES)
    receipt = {"schema": operator.LIVE_STATUS_SCHEMA, "program_id": operator.PROGRAM,
               "samples": [{"observed_at": 90.0}, {"observed_at": 100.0}], "observed_at": 100.0,
               **{key: bootstrap[key] for key in ("source_head", "repository_tree_id", "plan_root_cid", "bootstrap_receipt_id")}}
    receipt["receipt_cid"] = operator._identity(receipt)
    paths["status_receipt"] = tmp_path / "just-published.json"
    operator._atomic_json(paths["status_receipt"], receipt)
    monkeypatch.setattr(operator, "time", SimpleNamespace(time=lambda: 130.0))
    assert operator._read_live_status_receipt(board, paths)[1] == 30.0
    monkeypatch.setattr(operator, "time", SimpleNamespace(time=lambda: 130.001))
    with pytest.raises(operator.OperatorError, match="receipt is stale"):
        operator._read_live_status_receipt(board, paths)
