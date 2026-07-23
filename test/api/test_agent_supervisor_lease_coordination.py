from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.bundle_supervisor import (
    launch_bundle_lanes,
    plan_bundle_lanes,
)
from ipfs_accelerate_py.agent_supervisor.lease_coordination import (
    DependencyNotReadyError,
    ExecutionScopeConflictError,
    LeaseConflictError,
    LeaseCoordinator,
    LeaseExpiredError,
    LeaseQueueBridge,
    StaleFencingTokenError,
    adapt_goal_bundle,
    migrate_sqlite_coordination_store,
    profile_g_cid,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    build_bundle_task_payloads,
    submit_bundle_tasks,
)


def _bundle() -> dict[str, object]:
    return {
        "bundle_key": "objective/accelerator/leases",
        "parallel_lane": "accelerator-a",
        "todo_path": "tasks.todo.md",
        "source_todo": "all.todo.md",
        "tasks": [{"task_id": "SVD-085"}],
    }


def _named_bundle(name: str) -> dict[str, object]:
    return {
        "bundle_key": f"objective/dependency/{name.lower()}",
        "source_todo": "dependency.todo.md",
        "tasks": [{"task_id": name}],
    }


def test_claim_waits_for_latest_successful_prerequisite_receipt(tmp_path: Path) -> None:
    prerequisite = _named_bundle("PRE-1")
    prerequisite_cid = adapt_goal_bundle(prerequisite, created_at_ms=1)["canonical_task_cid"]
    dependent = {
        **_named_bundle("DEP-1"),
        "dependency_task_cids": [prerequisite_cid],
    }

    with LeaseCoordinator(tmp_path / "leases.sqlite3") as coordinator:
        dependent_task = coordinator.register_bundle(dependent, created_at_ms=1)
        with pytest.raises(DependencyNotReadyError) as missing_error:
            coordinator.claim(dependent_task["task_cid"], "did:web:dependent.example")
        missing = missing_error.value.evidence
        assert missing_error.value.code == "G_DEPENDENCY_NOT_READY"
        assert missing["missing_dependency_task_cids"] == [prerequisite_cid]
        assert missing["repair_evidence"][0]["kind"] == "missing_dependency"

        prerequisite_task = coordinator.register_bundle(prerequisite, created_at_ms=1)
        failed_grant = coordinator.claim(prerequisite_task["task_cid"], "did:web:prerequisite.example")
        coordinator.receipt(failed_grant, status="failed", failure_class="validation")
        failed = coordinator.claimability(dependent_task["task_cid"])
        assert failed["claimable"] is False
        assert failed["repair_evidence"][0]["latest_status"] == "failed"

        successful_grant = coordinator.claim(prerequisite_task["task_cid"], "did:web:prerequisite.example")
        coordinator.receipt(successful_grant, status="succeeded", output={"merge_commit": "abc123"})
        ready = coordinator.claimability(dependent_task["task_cid"])
        assert ready["claimable"] is True
        assert ready["satisfied_dependency_task_cids"] == [prerequisite_cid]
        assert coordinator.claim(dependent_task["task_cid"], "did:web:dependent.example")


def test_claimability_evidence_is_bounded_and_aliases_resolve_to_bundle_receipts(tmp_path: Path) -> None:
    prerequisite = _named_bundle("PRE-ALIAS")
    member_cid = profile_g_cid({"member": "PRE-ALIAS"})
    prerequisite["tasks"][0]["canonical_task_cid"] = member_cid  # type: ignore[index]
    dependent = {**_named_bundle("DEP-ALIAS"), "dependency_task_cids": [member_cid]}
    missing = {
        **_named_bundle("DEP-MISSING"),
        "dependency_task_cids": ["missing-a", "missing-b", "missing-c"],
    }

    with LeaseCoordinator(tmp_path / "leases.sqlite3") as coordinator:
        prerequisite_task = coordinator.register_bundle(prerequisite, created_at_ms=1)
        dependent_task = coordinator.register_bundle(dependent, created_at_ms=1)
        missing_task = coordinator.register_bundle(missing, created_at_ms=1)
        bounded = coordinator.claimability(missing_task["task_cid"], max_evidence=1)
        assert len(bounded["repair_evidence"]) == 1
        assert bounded["evidence_truncated"] is True

        grant = coordinator.claim(prerequisite_task["task_cid"], "did:web:prerequisite.example")
        coordinator.receipt(grant, status="succeeded", output={"merge_commit": "abc123"})
        assert coordinator.claimability(dependent_task["task_cid"])["claimable"] is True


def test_dependency_cycle_produces_repair_evidence_instead_of_claiming(tmp_path: Path) -> None:
    first = _named_bundle("CYCLE-A")
    second = _named_bundle("CYCLE-B")
    first_cid = adapt_goal_bundle(first, created_at_ms=1)["canonical_task_cid"]
    second_cid = adapt_goal_bundle(second, created_at_ms=1)["canonical_task_cid"]
    first["dependency_task_cids"] = [second_cid]
    second["dependency_task_cids"] = [first_cid]

    with LeaseCoordinator(tmp_path / "leases.sqlite3") as coordinator:
        first_task = coordinator.register_bundle(first, created_at_ms=1)
        coordinator.register_bundle(second, created_at_ms=1)
        readiness = coordinator.claimability(first_task["task_cid"])
        assert readiness["claimable"] is False
        assert readiness["dependency_cycles"] == [[first_cid, second_cid, first_cid]]
        assert any(item["kind"] == "dependency_cycle" for item in readiness["repair_evidence"])
        with pytest.raises(DependencyNotReadyError):
            coordinator.claim(first_task["task_cid"], "did:web:worker.example")


def test_empty_bundle_dependency_projection_absorbs_internal_member_edges(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "source_todo": "tasks.todo.md",
                "bundles": {
                    "objective/internal": {
                        "shard_path": "internal.todo.md",
                        "tasks": [
                            {"task_id": "INTERNAL-A", "task_cid": "cid-a"},
                            {
                                "task_id": "INTERNAL-B",
                                "task_cid": "cid-b",
                                "depends_on": ["INTERNAL-A"],
                            },
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    payload = build_bundle_task_payloads(index)[0]
    assert payload["dependency_task_cids"] == []
    assert payload["profile_g"]["task"]["dependency_task_cids"] == []

    with LeaseCoordinator(tmp_path / "leases.sqlite3") as coordinator:
        registered = coordinator.register_bundle(payload, created_at_ms=1)
        assert registered["dependency_task_cids"] == []
        assert coordinator.claimability(registered["task_cid"])["claimable"] is True
        assert coordinator.claim(registered["task_cid"], "did:web:internal.example")


def test_explicit_empty_projection_overrides_stale_embedded_dependencies(tmp_path: Path) -> None:
    member_cid = profile_g_cid({"member": "LEGACY-INTERNAL"})
    legacy = _named_bundle("LEGACY-INTERNAL")
    legacy["tasks"][0]["dependency_task_cids"] = [member_cid]  # type: ignore[index]
    legacy["profile_g"] = adapt_goal_bundle(legacy, created_at_ms=1)
    legacy["dependency_task_cids"] = []

    with LeaseCoordinator(tmp_path / "leases.sqlite3") as coordinator:
        registered = coordinator.register_bundle(legacy, created_at_ms=1)
        assert registered["dependency_task_cids"] == []
        assert coordinator.claimability(registered["task_cid"])["claimable"] is True


def test_planner_structural_repairs_block_claims_without_resolved_dependency_cids(tmp_path: Path) -> None:
    blocked = {
        **_named_bundle("PLANNER-BLOCKED"),
        "dependency_repair_evidence": [
            {
                "kind": "missing_dependency",
                "task_id": "PLANNER-BLOCKED",
                "reference": "UNKNOWN-1",
                "message": "dependency reference cannot be resolved",
            },
            {
                "kind": "dependency_cycle",
                "task_id": "PLANNER-BLOCKED",
                "reference": "A -> B -> A",
                "message": "task participates in a dependency cycle",
            },
        ],
    }
    clean = _named_bundle("PLANNER-CLEAN")

    with LeaseCoordinator(tmp_path / "leases.sqlite3") as coordinator:
        blocked_task = coordinator.register_bundle(blocked, created_at_ms=1)
        clean_task = coordinator.register_bundle(clean, created_at_ms=1)
        readiness = coordinator.claimability(blocked_task["task_cid"], max_evidence=1)
        assert readiness["claimable"] is False
        assert readiness["dependency_task_cids"] == []
        assert readiness["planner_repair_evidence_count"] == 2
        assert len(readiness["repair_evidence"]) == 1
        assert readiness["evidence_truncated"] is True
        with pytest.raises(DependencyNotReadyError):
            coordinator.claim(blocked_task["task_cid"], "did:web:blocked.example")
        assert coordinator.claim(clean_task["task_cid"], "did:web:clean.example")


def test_goal_bundle_adapter_emits_immutable_linked_profile_g_artifacts() -> None:
    adapted = adapt_goal_bundle(_bundle(), created_at_ms=1_783_872_000_000)

    assert adapted["goal"]["schema"] == "mcp++/profile-g/goal@1"
    assert adapted["subgoal"]["goal_cid"] == adapted["goal_cid"]
    assert adapted["selection"]["plan_branch_cid"] == adapted["plan_branch_cid"]
    assert adapted["task"]["subgoal_cid"] == adapted["subgoal_cid"]
    assert adapted["task"]["selection_cid"] == adapted["selection_cid"]
    assert adapted["task"]["tool"] == "codex.todo_bundle"
    for name in ("goal", "subgoal", "plan_branch", "selection", "task"):
        assert profile_g_cid(adapted[name]) in adapted["artifacts"]


def test_regenerated_bundle_keeps_one_canonical_lease_identity(tmp_path: Path) -> None:
    first = adapt_goal_bundle(_bundle(), created_at_ms=1_783_872_000_000)
    second = adapt_goal_bundle(_bundle(), created_at_ms=1_783_872_001_000)

    assert first["task_spec_cid"] != second["task_spec_cid"]
    assert first["canonical_task_cid"] == second["canonical_task_cid"]
    assert first["task"]["canonical_task_cid"] == first["canonical_task_cid"]

    with LeaseCoordinator(tmp_path / "leases.sqlite3") as coordinator:
        registered_first = coordinator.register_bundle(_bundle(), created_at_ms=1_783_872_000_000)
        grant = coordinator.claim(
            registered_first["task_cid"],
            "did:web:lane-a.example",
            requested_lease_ms=10_000,
        )
        registered_second = coordinator.register_bundle(_bundle(), created_at_ms=1_783_872_001_000)

        assert registered_first["task_spec_cid"] != registered_second["task_spec_cid"]
        assert registered_first["task_cid"] == registered_second["task_cid"] == grant.task_cid
        assert coordinator.claim(
            registered_second["task_cid"],
            "did:web:lane-a.example",
            requested_lease_ms=10_000,
        ).goal_cid == grant.goal_cid
        with pytest.raises(LeaseConflictError):
            coordinator.claim(
                registered_second["task_cid"],
                "did:web:lane-b.example",
                requested_lease_ms=10_000,
            )


def test_identical_registration_is_a_duckdb_noop(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "leases.duckdb"
    now = [10]
    payload = _bundle()
    payload["profile_g"] = adapt_goal_bundle(payload, created_at_ms=1)

    with LeaseCoordinator(path, clock_ms=lambda: now[0]) as coordinator:
        registered = coordinator.register_bundle(payload)
        first = coordinator.task_state(registered["task_cid"])
        now[0] = 20
        coordinator.register_bundle(payload)
        second = coordinator.task_state(registered["task_cid"])

    connection = duckdb.connect(str(path), read_only=True)
    try:
        artifact_count = connection.execute(
            "SELECT count(*) FROM artifacts"
        ).fetchone()[0]
        alias_count = connection.execute(
            "SELECT count(*) FROM task_aliases"
        ).fetchone()[0]
    finally:
        connection.close()

    assert artifact_count == 5
    assert alias_count == 2
    assert first is not None and second is not None
    assert first["updated_at_ms"] == second["updated_at_ms"] == 10


def test_legacy_sqlite_coordination_store_migrates_to_duckdb(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.sqlite3"
    target = tmp_path / "leases.duckdb"
    connection = sqlite3.connect(source)
    try:
        connection.execute(
            """CREATE TABLE artifacts(
                 cid TEXT PRIMARY KEY, kind TEXT NOT NULL,
                 payload_json TEXT NOT NULL, created_at_ms INTEGER NOT NULL
               )"""
        )
        connection.execute(
            "INSERT INTO artifacts VALUES(?,?,?,?)",
            ("cid-one", "test", json.dumps({"migrated": True}), 1),
        )
        connection.execute(
            """CREATE TABLE tasks(
                 task_cid TEXT PRIMARY KEY, goal_cid TEXT NOT NULL,
                 subgoal_cid TEXT NOT NULL, task_id TEXT NOT NULL,
                 bundle_json TEXT NOT NULL, registered_at_ms INTEGER NOT NULL,
                 updated_at_ms INTEGER NOT NULL
               )"""
        )
        connection.execute(
            "INSERT INTO tasks VALUES(?,?,?,?,?,?,?)",
            (
                "task-one",
                "goal-one",
                "subgoal-one",
                "T-1",
                json.dumps({"bundle_key": "one", "tasks": [{"task_id": "T-1"}]}),
                1,
                1,
            ),
        )
        connection.commit()
    finally:
        connection.close()

    migrated = migrate_sqlite_coordination_store(source, target)

    assert migrated["row_counts"]["artifacts"] == 1
    assert migrated["row_counts"]["tasks"] == 1
    with LeaseCoordinator(target) as coordinator:
        assert coordinator.get_artifact("cid-one") == {"migrated": True}
        assert coordinator.task_state("task-one")["bundle_key"] == "one"


def test_reopened_blocked_bundle_resets_exhausted_attempt_budget(tmp_path: Path) -> None:
    blocked_bundle = {
        **_bundle(),
        "max_attempts": 1,
        "tasks": [{"task_id": "SVD-085", "status": "blocked"}],
    }
    reopened_bundle = {
        **blocked_bundle,
        "tasks": [{"task_id": "SVD-085", "status": "todo"}],
    }

    with LeaseCoordinator(tmp_path / "leases.sqlite3") as coordinator:
        registered = coordinator.register_bundle(blocked_bundle, created_at_ms=1)
        failed = coordinator.claim(registered["task_cid"], "did:web:lane-a.example")
        coordinator.receipt(failed, status="failed", failure_class="blocked")
        assert coordinator.task_state(registered["task_cid"])["state"] == "blocked"

        reopened = coordinator.register_bundle(reopened_bundle, created_at_ms=2)
        reopened_state = coordinator.task_state(reopened["task_cid"])

        assert reopened["task_cid"] == registered["task_cid"]
        assert reopened["attempt_budget_reset"] is True
        assert reopened_state["state"] == "ready"
        assert reopened_state["attempt"] == 0
        assert reopened_state["release_reason"] == "requeued:bundle_status_reopened"
        replacement = coordinator.claim(reopened["task_cid"], "did:web:lane-b.example")
        assert replacement.attempt == 1
        assert replacement.fencing_token > failed.fencing_token


def test_changed_bundle_revision_cannot_overlap_its_active_execution_scope(
    tmp_path: Path,
) -> None:
    first_bundle = _bundle()
    first_bundle["tasks"] = [{"task_id": "SVD-085", "title": "First revision"}]
    second_bundle = _bundle()
    second_bundle["tasks"] = [{"task_id": "SVD-086", "title": "Second revision"}]

    with LeaseCoordinator(tmp_path / "leases.sqlite3") as coordinator:
        first = coordinator.register_bundle(first_bundle, created_at_ms=1)
        second = coordinator.register_bundle(second_bundle, created_at_ms=2)
        assert first["task_cid"] != second["task_cid"]

        first_grant = coordinator.claim(
            first["task_cid"],
            "did:web:lane-a.example",
            requested_lease_ms=10_000,
        )
        with pytest.raises(ExecutionScopeConflictError) as raised:
            coordinator.claim(
                second["task_cid"],
                "did:web:lane-b.example",
                requested_lease_ms=10_000,
            )
        assert raised.value.code == "G_EXECUTION_SCOPE_CONFLICT"
        assert coordinator.claim_ready(
            "did:web:lane-b.example",
            requested_lease_ms=10_000,
            eligible_task_cids=(second["task_cid"],),
        ) is None

        coordinator.release(first_grant)
        assert coordinator.claim(
            second["task_cid"],
            "did:web:lane-b.example",
            requested_lease_ms=10_000,
        ).task_cid == second["task_cid"]


def test_claim_renew_heartbeat_release_and_receipt_are_fenced(tmp_path: Path) -> None:
    now = 1_783_872_000_000
    with LeaseCoordinator(tmp_path / "leases.sqlite3", clock_ms=lambda: now) as coordinator:
        adapted = coordinator.register_bundle(_bundle(), created_at_ms=now)
        grant = coordinator.claim(adapted["task_cid"], "did:web:lane-a.example", requested_lease_ms=10_000)

        with pytest.raises(LeaseConflictError):
            coordinator.claim(adapted["task_cid"], "did:web:lane-b.example", requested_lease_ms=10_000)

        renewed = coordinator.renew(grant, requested_lease_ms=20_000, now_ms=now + 1_000)
        assert renewed.fencing_token == grant.fencing_token == 1
        assert renewed.lease_expires_at_ms == now + 21_000
        heartbeat = coordinator.heartbeat(renewed, capacity_millionths=750_000, now_ms=now + 2_000)
        assert heartbeat["fencing_token"] == 1
        assert heartbeat["capacity_millionths"] == 750_000

        result = coordinator.receipt(
            renewed,
            status="succeeded",
            output={"commit": "abc123"},
            started_at_ms=now,
            now_ms=now + 3_000,
        )
        assert result["goal_cid"] == adapted["goal_cid"]
        assert result["subgoal_cid"] == adapted["subgoal_cid"]
        assert result["receipt"]["claim_cid"] == grant.claim_cid
        assert result["receipt"]["fencing_token"] == 1
        assert coordinator.active_lease(adapted["task_cid"], now_ms=now + 3_001) is None
        with pytest.raises(LeaseExpiredError):
            coordinator.heartbeat(renewed, capacity_millionths=1, now_ms=now + 3_001)


def test_expired_lane_is_recovered_with_higher_epoch_and_token(tmp_path: Path) -> None:
    now = 1_783_872_000_000
    with LeaseCoordinator(tmp_path / "leases.sqlite3", clock_ms=lambda: now) as coordinator:
        adapted = coordinator.register_bundle(_bundle(), created_at_ms=now)
        old = coordinator.claim(adapted["task_cid"], "did:web:lane-a.example", requested_lease_ms=5_000)
        replacement = coordinator.claim(
            adapted["task_cid"],
            "did:web:lane-b.example",
            requested_lease_ms=5_000,
            now_ms=now + 5_001,
        )

        assert replacement.logical_epoch == old.logical_epoch + 1
        assert replacement.fencing_token == old.fencing_token + 1
        with pytest.raises((LeaseExpiredError, StaleFencingTokenError)):
            coordinator.renew(old, requested_lease_ms=5_000, now_ms=now + 5_002)
        with pytest.raises((LeaseExpiredError, StaleFencingTokenError)):
            coordinator.receipt(old, status="failed", failure_class="fenced", now_ms=now + 5_002)
        assert coordinator.active_lease(adapted["task_cid"], now_ms=now + 5_002) == replacement


def test_release_allows_a_fenced_takeover(tmp_path: Path) -> None:
    now = 1_783_872_000_000
    with LeaseCoordinator(tmp_path / "leases.sqlite3", clock_ms=lambda: now) as coordinator:
        adapted = coordinator.register_bundle(_bundle(), created_at_ms=now)
        old = coordinator.claim(adapted["task_cid"], "did:web:lane-a.example", requested_lease_ms=5_000)
        resolution_cid = coordinator.release(old, now_ms=now + 100)
        assert coordinator.get_artifact(resolution_cid)["outcome"] == "released"

        replacement = coordinator.claim(
            adapted["task_cid"], "did:web:lane-b.example", requested_lease_ms=5_000, now_ms=now + 101
        )
        assert replacement.fencing_token == old.fencing_token + 1
        with pytest.raises((LeaseExpiredError, StaleFencingTokenError)):
            coordinator.release(old, now_ms=now + 102)


def test_queue_bridge_carries_goal_task_adapters(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps({"source_todo": "tasks.todo.md", "bundles": {"objective/test": {
            "shard_path": "test.todo.md", "parallel_lane": "test", "tasks": [{"task_id": "T-1"}]
        }}}),
        encoding="utf-8",
    )
    payload = build_bundle_task_payloads(index)[0]
    assert payload["profile_g"]["task"]["subgoal_cid"] == payload["profile_g"]["subgoal_cid"]

    submitted: list[dict[str, object]] = []

    class Queue:
        def submit(self, **kwargs):
            submitted.append(kwargs)
            return "queue-id"

    assert submit_bundle_tasks(index, queue=Queue()) == ["queue-id"]
    assert submitted[0]["payload"]["profile_g"]["goal_cid"]
    assert submitted[0]["payload"]["profile_g"]["task_cid"]


def test_queue_bridge_requires_lease_and_links_terminal_receipt(tmp_path: Path) -> None:
    from types import SimpleNamespace

    payload = _bundle()
    payload["profile_g"] = adapt_goal_bundle(payload, created_at_ms=1_783_872_000_000)

    class Queue:
        released: list[str] = []
        completed: list[dict[str, object]] = []

        def claim_next(self, **_kwargs):
            return SimpleNamespace(task_id="queue-1", payload=payload)

        def release(self, *, reason, **_kwargs):
            self.released.append(reason)
            return True

        def complete(self, **kwargs):
            self.completed.append(kwargs)
            return True

    queue = Queue()
    with LeaseCoordinator(tmp_path / "leases.sqlite3") as coordinator:
        bridge = LeaseQueueBridge(
            queue,
            coordinator,
            worker_id="worker-a",
            claimant_did="did:web:worker-a.example",
            lease_ms=5_000,
        )
        leased = bridge.claim_next(supported_task_types=["codex.todo_bundle"])
        assert leased is not None
        assert coordinator.active_lease(leased.grant.task_cid) == leased.grant
        receipt = bridge.complete(leased, status="succeeded", output={"commit": "abc"})

    assert receipt["goal_cid"] == payload["profile_g"]["goal_cid"]
    assert queue.completed[0]["status"] == "completed"
    assert queue.completed[0]["result"]["profile_g"]["receipt"]["fencing_token"] == 1


def test_bundle_launcher_runs_only_an_accepted_lease(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    index = repo / "index.json"
    index.write_text(json.dumps({"bundles": {"objective/test": {
        "shard_path": "test.todo.md", "parallel_lane": "test", "tasks": [{"task_id": "T-1"}]
    }}}), encoding="utf-8")
    lanes = plan_bundle_lanes(
        bundle_index_path=index, repo_root=repo, state_root=repo / "state",
        worktree_root=repo / "worktrees", log_dir=repo / "logs",
    )
    starts: list[list[str]] = []

    class Process:
        pid = 1234

    def fake_popen(command, **_kwargs):
        starts.append(list(command))
        return Process()

    monkeypatch.setattr("ipfs_accelerate_py.agent_supervisor.bundle_supervisor.subprocess.Popen", fake_popen)
    coordination = repo / "coordination.sqlite3"
    first = launch_bundle_lanes(lanes, repo_root=repo, coordination_path=coordination, claimant_did="did:web:lane-a.example")
    second = launch_bundle_lanes(lanes, repo_root=repo, coordination_path=coordination, claimant_did="did:web:lane-b.example")

    assert first[0]["accepted"] is True
    assert "ipfs_accelerate_py.agent_supervisor.leased_lane" in first[0]["command"]
    assert second[0]["accepted"] is False
    assert second[0]["code"] == "G_CLAIM_CONFLICT"
    assert len(starts) == 1
