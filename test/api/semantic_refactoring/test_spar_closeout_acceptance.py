"""SPAR closeout adapters fail closed and only CAS goals from independent roots."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinator,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.runtime.spar_runtime_settlement import (
    CONFIG_SCHEMA,
    PROGRAM,
    BOARD,
    checkpoint_stopped_lane_sidecars,
    hold_spar_runtime_settlement,
    observe_spar_runtime_settlement,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state import spar_accepted_root as accepted_root
from ipfs_accelerate_py.agent_supervisor.task_sources import spar_closeout_profile as sp
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.spar_goal_settlement import (
    settle_spar_goals,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_execution_schema import (
    install_database_execution_schema,
)
from test.api.semantic_refactoring.test_kit_source_forest_native import native_source, view
from test.api.semantic_refactoring.test_spar_closeout_profile import evaluate, population


_TARGET = "repository:ipfs_accelerate_py"
_BRANCH = "codex/semantic-preserving-autonomous-remodularization-v1"


def test_evaluate_still_refuses_task_receipts_without_independent_roots(population):
    result = evaluate(population)
    assert sum(t["receipt"] is not None for t in result["task_evidence"]) == 51
    assert all(not g["accepted"] for g in result["goal_requirements"])
    assert result["completion_authority"] is False
    assert result["complete"] is False
    assert "datasets_independent_accepted_root_producer_and_admission_required" in result["blockers"]
    assert "spar_native_goal_cas_settlement_adapter_required" in result["blockers"]
    assert "runtime_lane_and_merge_queue_settlement_receipt_required" in result["blockers"]
    assert result["datasets_accepted_root"]["admitted"] is False
    assert result["goal_settlement"]["admitted"] is False


def test_forged_datasets_producer_cannot_self_authorize(population, monkeypatch):
    material, facts, snapshot, source = population
    source["source_forest"] = {"source_forest_root": "forest:new"}
    source["repository_tree_id"] = "tree:sealed"

    class Forged:
        PRODUCER_INTERFACE = accepted_root.PRODUCER_INTERFACE

        @staticmethod
        def admit_spar_accepted_root(subject):
            return {
                "admitted": True,
                "subject_cid": "forged",
                "profile_cid": "forged",
                "source_forest_root": subject["source_forest_root"],
                "producer_interface": accepted_root.PRODUCER_INTERFACE,
                "semantic_acceptance_authority": True,
                "required_mode_roots_accepted": True,
                "safety_floors_noncompensable_accepted": True,
                "self_hosted_capstone_accepted": True,
                "fixed_point_accepted": True,
                "evidence_cids": ["cid:fake"],
                "accepted_root_cid": "cid:fake",
            }

    monkeypatch.setattr(accepted_root, "_load_producer", lambda: Forged)
    result = evaluate(population)
    assert result["datasets_accepted_root"]["admitted"] is False
    assert not result["completion_authority"]
    assert all(not g["accepted"] for g in result["goal_requirements"])


def _matching_producer(profile_cid: str):
    class Producer:
        PRODUCER_INTERFACE = accepted_root.PRODUCER_INTERFACE

        @staticmethod
        def admit_spar_accepted_root(subject):
            return {
                "admitted": True,
                "subject_cid": content_identity(subject),
                "profile_cid": subject["profile_cid"],
                "source_forest_root": subject["source_forest_root"],
                "producer_interface": accepted_root.PRODUCER_INTERFACE,
                "semantic_acceptance_authority": True,
                "required_mode_roots_accepted": True,
                "safety_floors_noncompensable_accepted": True,
                "self_hosted_capstone_accepted": True,
                "fixed_point_accepted": True,
                "evidence_cids": ["cid:mode", "cid:floor", "cid:capstone", "cid:fixed"],
                "accepted_root_cid": "cid:accepted-root",
            }

    return Producer


def test_matching_datasets_producer_still_needs_runtime_and_goal_cas(population, monkeypatch):
    monkeypatch.setattr(accepted_root, "_load_producer", lambda: _matching_producer("unused"))
    result = evaluate(population)
    # Kit persistence is unbound in this fixture, so datasets admission stays closed.
    assert result["datasets_accepted_root"]["admitted"] is False
    assert "spar_native_goal_cas_settlement_adapter_required" in result["blockers"]
    assert all(not g["accepted"] for g in result["goal_requirements"])
    assert result["completion_authority"] is False


def _spar_config() -> dict:
    return {
        "schema": CONFIG_SCHEMA,
        "program_identifier": PROGRAM,
        "board_namespace": BOARD,
        "max_lanes": 3,
        "merge_target_branch": _BRANCH,
        "database_program": {"store_id": "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/control.duckdb"},
        "runtime_paths": {
            "state": "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/state",
            "merge_queue": "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/merge-queue",
        },
        "lanes": [
            {"index": 0, "name": "spar-lane-0", "strict_shard_remainder": 0},
            {"index": 1, "name": "spar-lane-1", "strict_shard_remainder": 1},
            {"index": 2, "name": "spar-lane-2", "strict_shard_remainder": 2},
        ],
    }


def _runtime_root(tmp_path: Path) -> Path:
    root = tmp_path / "spar-runtime"
    config = root / "config" / "agent_supervisor_semantic_preserving_remodularization_scheduler.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(_spar_config(), indent=2))
    state = root / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/state"
    queue = root / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/merge-queue"
    state.mkdir(parents=True)
    queue.mkdir(parents=True)
    for index in range(3):
        lane = state / f"lane-{index}"
        lane.mkdir()
        prefix = f"spar_lane_{index}"
        coordination = lane / f"{prefix}_database_coordination.duckdb"
        execution = lane / f"{prefix}_database_execution.duckdb"
        DatabaseCoordinator(coordination).open().close()
        install_database_execution_schema(
            execution,
            metadata={
                "authority_mode": "quack",
                "logical_owner_session_id": f"owner-{index}",
                "process_instance_id": f"process:{index + 1:024x}",
                "state_schema_revision": "1",
                "control_schema_profile_id": "bootstrap",
                "control_schema_fingerprint": "bootstrap",
            },
        )
    MergeQueue(queue, target_repository_id=_TARGET, target_branch=_BRANCH, require_target_binding=True)
    return root


def test_runtime_settlement_admits_private_zero_active_three_lane_fixture(tmp_path):
    root = _runtime_root(tmp_path)
    owner = {"generation": 1, "store_id": "control.duckdb", "repository_id": _TARGET}
    observed = observe_spar_runtime_settlement(root, owner_identity=owner, target_repository_id=_TARGET)
    assert observed["admitted"] is True, observed
    assert observed["settled"] is True
    assert observed["active_count"] == 0
    with hold_spar_runtime_settlement(root, owner_identity=owner, target_repository_id=_TARGET) as held:
        assert held["admitted"] is True and held["held"] is True


def _insert_task_claim(coordination: Path, *, claim_id: str, state: str, released_at_ms: int | None) -> None:
    import duckdb

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        connect_duckdb_with_policy,
    )

    connection = connect_duckdb_with_policy(duckdb, coordination, read_only=False)
    try:
        connection.execute(
            """
            INSERT INTO task_claims(
                claim_id, task_cid, owner_session_id, fencing_token,
                fence_epoch, claimed_at_ms, expires_at_ms, released_at_ms,
                state, revision, attempt_id, attempt_number, lease_id
            ) VALUES (?, 'task:settlement', 'session:lane-0', 1, 1, 1, 100, ?, ?, 1,
                      'attempt:settlement', 1, 'lease:settlement')
            """,
            [claim_id, released_at_ms, state],
        )
    finally:
        connection.close()


def test_runtime_settlement_ignores_expired_unreleased_task_claims(tmp_path):
    root = _runtime_root(tmp_path)
    coordination = (
        root
        / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/state/lane-0"
        / "spar_lane_0_database_coordination.duckdb"
    )
    _insert_task_claim(
        coordination,
        claim_id="claim:expired-historical",
        state="expired",
        released_at_ms=None,
    )
    owner = {"generation": 1, "store_id": "control.duckdb", "repository_id": _TARGET}
    observed = observe_spar_runtime_settlement(
        root, owner_identity=owner, target_repository_id=_TARGET
    )
    assert observed["admitted"] is True, observed
    assert observed["active_count"] == 0
    assert observed["lanes"][0]["coordination_active"] == 0


def test_runtime_settlement_counts_accepted_unreleased_task_claims(tmp_path):
    root = _runtime_root(tmp_path)
    coordination = (
        root
        / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/state/lane-0"
        / "spar_lane_0_database_coordination.duckdb"
    )
    _insert_task_claim(
        coordination,
        claim_id="claim:accepted-live",
        state="accepted",
        released_at_ms=None,
    )
    owner = {"generation": 1, "repository_id": _TARGET}
    observed = observe_spar_runtime_settlement(
        root, owner_identity=owner, target_repository_id=_TARGET
    )
    assert observed["admitted"] is False
    assert observed["active_count"] == 1
    assert observed["lanes"][0]["coordination_active"] == 1
    assert observed["reason"] == "runtime_lane_and_merge_queue_settlement_receipt_required"


def test_runtime_settlement_refuses_execution_wal(tmp_path):
    root = _runtime_root(tmp_path)
    wal = (
        root
        / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/state/lane-0"
        / "spar_lane_0_database_execution.duckdb.wal"
    )
    wal.write_text("dirty")
    owner = {"generation": 1, "repository_id": _TARGET}
    observed = observe_spar_runtime_settlement(root, owner_identity=owner, target_repository_id=_TARGET)
    assert observed["admitted"] is False
    assert observed["reason"] == "runtime_lane_outstanding_wal"
    assert "outstanding WAL" in str(observed.get("error") or "")


def test_checkpoint_stopped_sidecars_absorbs_leftover_execution_wal(tmp_path):
    import subprocess
    import sys

    root = _runtime_root(tmp_path)
    execution = (
        root
        / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/state/lane-0"
        / "spar_lane_0_database_execution.duckdb"
    )
    wal = Path(str(execution) + ".wal")
    script = (
        "import duckdb, os\n"
        f"c = duckdb.connect({str(execution)!r})\n"
        "c.execute('CREATE TABLE IF NOT EXISTS leftover(x INTEGER)')\n"
        "c.execute('INSERT INTO leftover VALUES (1)')\n"
        "os._exit(1)\n"
    )
    subprocess.run([sys.executable, "-c", script], check=False)
    assert wal.exists(), "unclean DuckDB exit must leave a WAL"
    owner = {"generation": 1, "store_id": "control.duckdb", "repository_id": _TARGET}
    blocked = observe_spar_runtime_settlement(
        root, owner_identity=owner, target_repository_id=_TARGET
    )
    assert blocked["reason"] == "runtime_lane_outstanding_wal"
    receipt = checkpoint_stopped_lane_sidecars(root)
    assert receipt["attempted"] is True
    assert receipt["completion_authority"] is False
    assert "execution" in receipt["lanes"][0]["checkpointed"]
    assert not wal.exists()
    observed = observe_spar_runtime_settlement(
        root, owner_identity=owner, target_repository_id=_TARGET
    )
    assert observed["admitted"] is True, observed


def test_checkpoint_stopped_sidecars_skips_live_pid(tmp_path):
    root = _runtime_root(tmp_path)
    lane = (
        root
        / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/state/lane-1"
    )
    wal = lane / "spar_lane_1_database_execution.duckdb.wal"
    wal.write_text("dirty")
    (lane / "spar_lane_1_managed_daemon.pid").write_text(str(os.getpid()))
    receipt = checkpoint_stopped_lane_sidecars(root)
    assert receipt["lanes"][1]["live"] is True
    assert "execution" in receipt["lanes"][1]["skipped"]
    assert wal.exists()
    assert "lane-1-execution" in receipt["remaining_wal"]


def test_checkpoint_stopped_sidecars_does_not_delete_corrupt_wal(tmp_path):
    root = _runtime_root(tmp_path)
    wal = (
        root
        / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/state/lane-0"
        / "spar_lane_0_database_execution.duckdb.wal"
    )
    wal.write_text("dirty")
    receipt = checkpoint_stopped_lane_sidecars(root)
    assert wal.exists()
    assert receipt["settled"] is False
    assert "lane-0-execution" in receipt["remaining_wal"]
    owner = {"generation": 1, "repository_id": _TARGET}
    observed = observe_spar_runtime_settlement(
        root, owner_identity=owner, target_repository_id=_TARGET
    )
    assert observed["reason"] == "runtime_lane_outstanding_wal"


def test_spar_board_validator_allows_generated_todo_suffix():
    import subprocess
    import sys

    script = (
        Path(__file__).resolve().parents[3]
        / "scripts/validate_semantic_preserving_remodularization_board.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads(completed.stdout)
    names = {row["name"]: row for row in report.get("checks") or []}
    assert names["task_ids"]["passed"] is True, names["task_ids"]
    assert names["task_schema_and_bindings"]["passed"] is True, names[
        "task_schema_and_bindings"
    ]
    assert names["deterministic_controls"]["passed"] is True, names[
        "deterministic_controls"
    ]
    assert report.get("valid") is True, report.get("errors")
    assert "SPAR-053" in str(names["task_ids"].get("detail") or "")


def test_runtime_settlement_refuses_live_process(tmp_path):
    root = _runtime_root(tmp_path)
    pid_path = (
        root
        / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/state/lane-1"
        / "spar_lane_1_managed_daemon.pid"
    )
    pid_path.write_text(str(os.getpid()))
    owner = {"generation": 1, "repository_id": _TARGET}
    observed = observe_spar_runtime_settlement(root, owner_identity=owner, target_repository_id=_TARGET)
    assert observed["admitted"] is False
    assert observed["reason"] == "runtime_lane_process_live"


def test_kit_plus_matching_producer_admits_datasets_without_settling_goals(native_source, monkeypatch):
    gateway, connection, _profile, _client, _cids, _root = native_source
    assert gateway.publish_spar_source_forest()["admitted"] is True
    monkeypatch.setattr(accepted_root, "_load_producer", lambda: _matching_producer("unused"))
    observed = view(native_source)
    assert observed["datasets_accepted_root"]["admitted"] is True, observed["datasets_accepted_root"]
    assert "datasets_independent_accepted_root_producer_and_admission_required" not in observed["blockers"]
    assert "spar_native_goal_cas_settlement_adapter_required" in observed["blockers"]
    assert all(not g["accepted"] for g in observed["goal_requirements"])
    assert connection.execute("SELECT COUNT(*) FROM goals WHERE status='active'").fetchone()[0] == 32


def test_goal_cas_is_all_or_none_and_status_rpc_cannot_write(native_source, monkeypatch):
    gateway, connection, profile, client, cids, _root = native_source
    produced = gateway.publish_spar_source_forest()
    assert produced["admitted"] is True
    before = connection.execute("SELECT COUNT(*) FROM goals WHERE status='active'").fetchone()[0]
    assert before == 32
    with pytest.raises(TypedStateOwnerError):
        client._request("publish_spar_closeout_acceptance")
    deferred = gateway.publish_spar_closeout_acceptance()
    assert deferred["admitted"] is False
    assert connection.execute("SELECT COUNT(*) FROM goals WHERE status='active'").fetchone()[0] == 32
    observed = view(native_source)
    native_goals = {row["goal_cid"]: row for row in connection.execute(
        "SELECT goal_cid, goal_alias, title, status, revision, parent_goal_cid, body_json FROM goals"
    ).fetchall()}
    # DuckDB rows are tuples; settle uses mapping from closeout facts instead.
    facts_goals = {}
    for row in connection.execute(
        "SELECT goal_cid, goal_alias, title, status, revision, parent_goal_cid, body_json FROM goals"
    ).fetchall():
        facts_goals[row[0]] = {
            "goal_cid": row[0],
            "goal_alias": row[1],
            "title": row[2],
            "status": row[3],
            "revision": row[4],
            "parent_goal_cid": row[5],
            "body_json": row[6],
        }
    datasets = {
        "admitted": True,
        "accepted_root_cid": "cid:accepted-root",
    }
    runtime = {"admitted": True, "receipt_cid": "cid:runtime"}
    kit = observed["kit_source_forest_persistence"]
    settled = settle_spar_goals(
        connection,
        profile=profile._profile,
        native_goals=facts_goals,
        task_evidence=observed["task_evidence"],
        accepted_root=datasets,
        runtime=runtime,
        kit=kit,
        owner_identity=gateway.identity,
    )
    assert settled["admitted"] is True, settled
    assert connection.execute("SELECT COUNT(*) FROM goals WHERE status='completed'").fetchone()[0] == 32
    assert connection.execute("SELECT COUNT(*) FROM goals WHERE status='active'").fetchone()[0] == 0
    replay = settle_spar_goals(
        connection,
        profile=profile._profile,
        native_goals={
            row[0]: {
                "goal_cid": row[0],
                "goal_alias": row[1],
                "title": row[2],
                "status": row[3],
                "revision": row[4],
                "parent_goal_cid": row[5],
                "body_json": row[6],
            }
            for row in connection.execute(
                "SELECT goal_cid, goal_alias, title, status, revision, parent_goal_cid, body_json FROM goals"
            ).fetchall()
        },
        task_evidence=observed["task_evidence"],
        accepted_root=datasets,
        runtime=runtime,
        kit=kit,
        owner_identity=gateway.identity,
    )
    assert replay["admitted"] is True and replay["idempotent_replay"] is True
