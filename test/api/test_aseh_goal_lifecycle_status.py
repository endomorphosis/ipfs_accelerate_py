"""Goal lifecycle observations use owner state, never sealed source status."""

from __future__ import annotations

import copy
import json
import os
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import build_server
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    reset_quack_transport_cache,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    QuackCapabilityStatus,
    probe_quack_capabilities,
)
from scripts import run_agent_supervisor_efficiency_state_hardening as operator
from test.api.test_agent_supervisor_configured_typed_grant_handoff import (
    _aseh_health_fixture,
    _materialize_one_task,
)


def _seal_projection(projection):
    projection.pop("projection_cid", None)
    projection["projection_cid"] = operator.content_identity(projection)


def _bind_replay(sample):
    """Seal the unit fixture witness to its exact synthetic snapshot."""
    authority = sample["authority"]
    snapshot = authority["snapshot"]
    witness = authority["projection_reconciliation"]
    witness.update(
        projection_cid=snapshot["projection_cid"], event_cursor=snapshot["event_cursor"]
    )
    witness["cache_key"] = operator._identity(
        {
            "replica": witness["replica"],
            **{
                key: snapshot[key]
                for key in (
                    "projection_cid",
                    "event_cursor",
                    "plan_root_cid",
                    "repository_tree_id",
                )
            },
        }
    )
    witness.pop("witness_cid", None)
    witness["witness_cid"] = operator._identity(witness)


@pytest.fixture(scope="module")
def canonical_capture(tmp_path_factory):
    """Capture actual disposable owner records before starting any server."""
    database = tmp_path_factory.mktemp("aseh-goal-capture") / "control.duckdb"
    _materialize_one_task(database)
    with DatabaseTaskSource(database, install_schema=False) as source:
        goal = dict(source.get_goal("goal:aseh-bootstrap-test"))
        body = {**goal["body"], "body": {"status": "active"}}
        source.intent.upsert_goal(
            goal_cid=goal["goal_cid"],
            goal_alias=goal["goal_alias"],
            title=goal["title"],
            objective_id=goal["objective_id"],
            ordinal=goal["ordinal"],
            body=body,
            status="completed",
            expected_revision=goal["revision"],
        )
        goal = dict(source.get_goal(goal["goal_cid"]))
        snapshot = source.snapshot().to_dict()
        projection = dict(source.plan_projection())
    return (
        database,
        snapshot,
        projection,
        {goal["goal_alias"]: operator._immutable_goal_record(goal)},
    )


def test_canonical_lifecycle_is_separate_from_immutable_source_status(
    canonical_capture,
):
    _database, snapshot, projection, semantics = canonical_capture
    before = copy.deepcopy(semantics)
    observed = operator._goal_lifecycle_from_plan_projection(
        projection, snapshot, semantics
    )
    assert observed["available"] is True
    assert observed["records"]["ASEH-G000"] == {
        "goal_cid": "goal:aseh-bootstrap-test",
        "status": "completed",
        "revision": 2,
    }
    assert semantics == before
    assert semantics["ASEH-G000"]["body"]["body"]["status"] == "active"
    assert observed["snapshot_projection_cid"] == snapshot["projection_cid"]
    assert observed["snapshot_material"]["event_watermark"] == snapshot["event_cursor"]


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_goal",
        "duplicate_goal",
        "foreign_alias",
        "missing_revision",
        "boolean_revision",
        "zero_revision",
        "missing_status",
        "foreign_semantics",
        "truncated_tasks",
        "missing_plan",
        "malformed_goals",
        "watermark",
        "digest",
    ],
)
def test_incomplete_or_mixed_projection_is_unavailable(canonical_capture, corruption):
    _database, snapshot, captured, semantics = canonical_capture
    projection = copy.deepcopy(captured)
    goal = projection["goals"][0]
    if corruption == "missing_goal":
        projection["goals"] = []
    elif corruption == "duplicate_goal":
        projection["goals"].append(copy.deepcopy(goal))
    elif corruption == "foreign_alias":
        goal["goal_alias"] = "ASEH-G999"
    elif corruption == "missing_revision":
        goal.pop("revision")
    elif corruption == "boolean_revision":
        goal["revision"] = True
    elif corruption == "zero_revision":
        goal["revision"] = 0
    elif corruption == "missing_status":
        goal.pop("status")
    elif corruption == "foreign_semantics":
        goal["body"] = {"status": "completed"}
    elif corruption == "truncated_tasks":
        projection["tasks"] = []
    elif corruption == "missing_plan":
        projection["plans"] = []
    elif corruption == "malformed_goals":
        projection["goals"] = None
    elif corruption == "watermark":
        projection["event_watermark"] += 1
    _seal_projection(projection)
    if corruption == "digest":
        projection["projection_cid"] = "forged"
    observed = operator._goal_lifecycle_from_plan_projection(
        projection, snapshot, semantics
    )
    assert observed["available"] is False
    assert observed["records"] is None
    assert observed["status_counts"] is None


@pytest.fixture
def external_status(tmp_path, monkeypatch, canonical_capture):
    _database, snapshot, projection, semantics = canonical_capture
    now = time.time()
    board, paths, current = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    previous = copy.deepcopy(current)
    previous["observed_at"] = now - 0.25
    lifecycle = operator._goal_lifecycle_from_plan_projection(
        projection, snapshot, semantics
    )
    for sample in (previous, current):
        sample["authority"].update(
            {
                "snapshot": copy.deepcopy(snapshot),
                "event_cursor": snapshot["event_cursor"],
                "goal_records": copy.deepcopy(semantics),
                "goal_lifecycle": copy.deepcopy(lifecycle),
            }
        )
        _bind_replay(sample)
    bootstrap = json.loads(paths["bootstrap_receipt"].read_text())
    bootstrap["integrity"]["goal_records"] = copy.deepcopy(semantics)
    bootstrap.pop("bootstrap_receipt_id")
    bootstrap["bootstrap_receipt_id"] = operator._identity(bootstrap)
    operator._atomic_json(paths["bootstrap_receipt"], bootstrap)
    receipt = {
        "schema": operator.LIVE_STATUS_SCHEMA,
        "program_id": operator.PROGRAM,
        **{
            key: bootstrap[key]
            for key in (
                "source_head",
                "repository_tree_id",
                "plan_root_cid",
                "bootstrap_receipt_id",
            )
        },
        "samples": [previous, current],
        "observed_at": now,
        "healthy": True,
        "broker_authenticated": True,
        "goal_records": copy.deepcopy(semantics),
    }
    paths.update(
        {"owner": tmp_path / "owner", "status_receipt": tmp_path / "live-status.json"}
    )
    owner_path = paths["owner"] / "quack-state-server.status.json"
    operator._atomic_json(owner_path, current["owner_status"])
    monkeypatch.setattr(operator, "_load", lambda _path: (board, {}))
    monkeypatch.setattr(operator, "_paths", lambda _board: paths)

    def publish():
        receipt.pop("receipt_cid", None)
        receipt["receipt_cid"] = operator._identity(receipt)
        operator._atomic_json(paths["status_receipt"], receipt)
        return operator.status(tmp_path / "unused.json", require_ready=True)

    return receipt, publish, owner_path


def test_external_status_returns_canonical_revision_and_observation_fence(
    external_status,
):
    receipt, publish, _owner_path = external_status
    code, report = publish()
    assert code == 0
    observed = report["goal_lifecycle"]
    assert observed["available"] is True
    assert observed["records"]["ASEH-G000"]["status"] == "completed"
    assert observed["records"]["ASEH-G000"]["revision"] == 2
    assert observed["status_counts"] == {"completed": 1}
    assert observed["observed_at"] == receipt["samples"][-1]["observed_at"]
    assert (
        observed["event_watermark"]
        == receipt["samples"][-1]["authority"]["event_cursor"]
    )
    assert observed["receipt_cid"] == report["receipt"]["receipt_cid"]
    assert (
        observed["owner_binding"]
        == receipt["samples"][-1]["authority"]["owner_binding"]
    )
    assert (
        report["receipt"]["goal_records"]["ASEH-G000"]["body"]["body"]["status"]
        == "active"
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "legacy",
        "one_missing",
        "unavailable",
        "partial",
        "row_boolean",
        "snapshot_digest",
        "cursor_mismatch",
        "status_without_revision",
        "revision_regression",
        "cursor_regression",
        "same_cursor_change",
        "unbound_replay",
        "wrong_transport",
        "wrong_credential",
        "unsealed_semantics",
        "redated_receipt",
        "unordered_samples",
        "boolean_timestamp",
        "huge_timestamp",
        "future_timestamp",
    ],
)
def test_missing_or_inconsistent_lifecycle_never_infers_status(
    external_status, corruption
):
    receipt, publish, _owner_path = external_status
    samples = receipt["samples"]
    authority = samples[-1]["authority"]
    lifecycle = authority["goal_lifecycle"]
    if corruption == "legacy":
        for sample in samples:
            sample["authority"].pop("goal_lifecycle")
    elif corruption == "one_missing":
        samples[0]["authority"].pop("goal_lifecycle")
    elif corruption == "unavailable":
        authority["available"] = False
    elif corruption == "partial":
        lifecycle["records"] = {}
    elif corruption == "row_boolean":
        lifecycle["records"]["ASEH-G000"]["revision"] = True
    elif corruption == "snapshot_digest":
        lifecycle["snapshot_material"]["goals"][0]["status"] = "active"
    elif corruption == "cursor_mismatch":
        authority["event_cursor"] += 1
    elif corruption == "unbound_replay":
        authority["projection_reconciliation"]["projection_cid"] = "stale"
    elif corruption == "wrong_transport":
        authority["transport"] = "embedded"
    elif corruption == "wrong_credential":
        authority["credential_path"] = "ambient_token"
    elif corruption == "unsealed_semantics":
        authority["goal_records"]["ASEH-G000"]["body"]["body"]["status"] = "completed"
    elif corruption == "redated_receipt":
        for sample in samples:
            sample["observed_at"] -= 100
    elif corruption == "unordered_samples":
        samples[0]["observed_at"] += 1
    elif corruption == "boolean_timestamp":
        samples[0]["observed_at"] = True
    elif corruption == "huge_timestamp":
        samples[0]["observed_at"] = 10**1000
    elif corruption == "future_timestamp":
        samples[-1]["observed_at"] += 100
        receipt["observed_at"] = samples[-1]["observed_at"]
    else:
        material = lifecycle["snapshot_material"]
        if corruption == "status_without_revision":
            lifecycle["records"]["ASEH-G000"]["status"] = "active"
            material["goals"][0]["status"] = "active"
            material["event_watermark"] += 1
        elif corruption == "revision_regression":
            lifecycle["records"]["ASEH-G000"]["revision"] = 1
            material["goals"][0]["revision"] = 1
            material["event_watermark"] += 1
        elif corruption == "cursor_regression":
            material["event_watermark"] -= 1
        else:
            lifecycle["records"]["ASEH-G000"]["revision"] += 1
            material["goals"][0]["revision"] += 1
        authority["event_cursor"] = material["event_watermark"]
        authority["snapshot"]["event_cursor"] = material["event_watermark"]
        authority["snapshot"]["projection_cid"] = operator.content_identity(material)
        lifecycle["snapshot_projection_cid"] = authority["snapshot"]["projection_cid"]
        _bind_replay(samples[-1])
    code, report = publish()
    assert code == 0  # This additive observation does not reinterpret health.
    assert report["healthy"] is True
    assert report["goal_lifecycle"]["available"] is False
    assert report["goal_lifecycle"]["records"] is None
    assert report["goal_lifecycle"]["status_counts"] is None


@pytest.mark.parametrize(
    "corruption", ["expired", "owner_generation", "wrong_receipt_digest"]
)
def test_existing_freshness_and_owner_gates_also_fence_lifecycle(
    external_status, corruption
):
    receipt, publish, owner_path = external_status
    if corruption == "expired":
        receipt["observed_at"] -= 61
    elif corruption == "owner_generation":
        owner = json.loads(owner_path.read_text())
        owner["identity"]["generation"] += 1
        operator._atomic_json(owner_path, owner)
    else:
        receipt["bootstrap_receipt_id"] = "stale-bootstrap"
    code, report = publish()
    assert code == 1
    assert report["broker_authenticated_receipt"] is False
    assert report["goal_lifecycle"]["available"] is False
    assert report["goal_lifecycle"]["records"] is None


@pytest.mark.parametrize("goal_change", [True, False])
def test_lifecycle_or_unrelated_events_can_advance_monotonically(
    external_status, goal_change
):
    receipt, publish, _owner_path = external_status
    authority = receipt["samples"][-1]["authority"]
    lifecycle = authority["goal_lifecycle"]
    material = lifecycle["snapshot_material"]
    material["event_watermark"] += 1
    if goal_change:
        lifecycle["records"]["ASEH-G000"].update(status="reopened", revision=3)
        material["goals"][0].update(status="reopened", revision=3)
    else:
        material["tasks"][0]["revision"] += 1
    authority["event_cursor"] = material["event_watermark"]
    authority["snapshot"]["event_cursor"] = material["event_watermark"]
    authority["snapshot"]["projection_cid"] = operator.content_identity(material)
    lifecycle["snapshot_projection_cid"] = authority["snapshot"]["projection_cid"]
    _bind_replay(receipt["samples"][-1])
    code, report = publish()
    assert code == 0
    assert report["goal_lifecycle"]["available"] is True
    assert report["goal_lifecycle"]["records"]["ASEH-G000"]["revision"] == (
        3 if goal_change else 2
    )


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"), reason="sealed memfd requires Linux"
)
def test_real_quack_owner_reports_lifecycle_without_direct_database_reads(
    tmp_path,
    monkeypatch,
    canonical_capture,
):
    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip(
            f"reviewed preinstalled Quack unavailable: {capability.status.value}"
        )
    captured_database, snapshot, _projection, semantics = canonical_capture
    database = tmp_path / "control.duckdb"
    database.write_bytes(captured_database.read_bytes())
    monkeypatch.chdir(tmp_path)
    owner_dir = tmp_path / "owner"
    bootstrap = {
        "schema": operator.BOOTSTRAP_SCHEMA,
        "source_head": "commit:lifecycle-test",
        "repository_tree_id": "tree:aseh-bootstrap-test",
        "plan_root_cid": snapshot["plan_root_cid"],
        "source_forest": {},
        "source_identities": {},
        "database_task_source_receipt": {},
        "snapshot": snapshot,
        "integrity": {"goal_records": semantics},
        "initial_ready_task_ids": ["ASEH-000"],
        "bootstrap_validation": {},
        "recovered_after_interrupted_materialization": False,
        "authority": {},
        "ducklake_projection": {},
    }
    bootstrap["bootstrap_receipt_id"] = operator._identity(bootstrap)
    paths = {
        "runtime": tmp_path,
        "database": database,
        "owner": owner_dir,
        "bootstrap_receipt": tmp_path / "bootstrap.json",
    }
    operator._atomic_json(paths["bootstrap_receipt"], bootstrap)
    server = build_server(
        database_path=database,
        state_dir=owner_dir,
        repository_root=tmp_path,
        port=0,
        store_id=str(database),
        secret_handle="handle:goal-lifecycle-test",
    )
    identity = server.start()
    try:
        for key, value in dict(server.start_supervisor_grant_broker()).items():
            monkeypatch.setenv(key, str(value))
        monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(database))
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", str(identity.generation)
        )
        monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", raising=False)
        board = SimpleNamespace(
            resolved_database_program=lambda: SimpleNamespace(
                quack_endpoint=identity.listen_uri
            ),
            payload={"check_interval_seconds": 10},
        )
        with operator._LIVE_REPLAY_CACHE_LOCK:
            operator._LIVE_REPLAY_CACHE.clear()
        original = DatabaseTaskSource.plan_projection
        reads = []

        def only_quack(source, **kwargs):
            reads.append(source.intent.uses_quack_transport)
            assert source.intent.uses_quack_transport is True
            return original(source, **kwargs)

        monkeypatch.setattr(DatabaseTaskSource, "plan_projection", only_quack)
        observed = operator._broker_status_query(
            board, paths, owner_status=server.status()
        )
        assert reads == [True]
        assert observed["projection_matches_events"] is True
        assert observed["goal_lifecycle"]["available"] is True
        assert observed["goal_lifecycle"]["records"]["ASEH-G000"] == {
            "goal_cid": "goal:aseh-bootstrap-test",
            "status": "completed",
            "revision": 2,
        }
        assert observed["goal_records"] == semantics
        assert observed["event_cursor"] == snapshot["event_cursor"]
        sample = {
            "observed_at": time.time(),
            "owner_status": server.status(),
            "authority": observed,
        }
        receipt = {
            "schema": operator.LIVE_STATUS_SCHEMA,
            "program_id": operator.PROGRAM,
            **{
                key: bootstrap[key]
                for key in (
                    "source_head",
                    "repository_tree_id",
                    "plan_root_cid",
                    "bootstrap_receipt_id",
                )
            },
            "samples": [copy.deepcopy(sample), sample],
            "observed_at": sample["observed_at"],
            "healthy": True,
            "broker_authenticated": True,
            "goal_records": semantics,
        }
        receipt["receipt_cid"] = operator._identity(receipt)
        paths["status_receipt"] = tmp_path / "lifecycle-status.json"
        operator._atomic_json(paths["status_receipt"], receipt)
        monkeypatch.setattr(operator, "_load", lambda _path: (board, {}))
        monkeypatch.setattr(operator, "_paths", lambda _board: paths)
        code, report = operator.status(
            tmp_path / "unused-config.json", require_ready=True
        )
        assert code == 0
        assert report["goal_lifecycle"]["available"] is True
        assert (
            report["goal_lifecycle"]["records"] == observed["goal_lifecycle"]["records"]
        )
    finally:
        reset_quack_transport_cache()
        server.stop()
