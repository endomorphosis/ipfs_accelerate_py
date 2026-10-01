"""Actual DuckDB intent/coordination and datasets producer world integration."""

import json
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.intent_world_snapshot import (
    IntentWorldSnapshotError,
    capture_intent_world_snapshot,
    generate_prompt_goal_graph_with_world,
    load_intent_world_context,
    persist_intent_world_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


def seed(tmp_path):
    intent = IntentRepository(tmp_path / "intent.duckdb")
    intent.upsert_objective(
        objective_id="objective:1", objective_alias="OBJ-1", title="Repair addition"
    )
    intent.upsert_goal(
        goal_cid="goal:root", goal_alias="GOAL-1", title="Addition", objective_id="objective:1"
    )
    intent.upsert_goal(
        goal_cid="goal:child",
        goal_alias="GOAL-2",
        title="Regression",
        objective_id="objective:1",
        parent_goal_cid="goal:root",
    )
    intent.link_goal_edge(
        parent_goal_cid="goal:root", child_goal_cid="goal:child", edge_kind="depends_on"
    )
    intent.upsert_plan(
        plan_cid="plan:1",
        goal_cid="goal:root",
        plan_alias="PLAN-1",
        status="active",
        body={"steps": ["fix", "test"]},
    )
    intent.upsert_task(
        task_cid="task:1",
        task_alias="TEST-1",
        goal_cid="goal:root",
        plan_cid="plan:1",
        objective_id="objective:1",
        status="ready",
        outputs=[{"path": "add.py", "effect": "modify"}],
        acceptance=[{"criterion": "Addition holds", "evidence_kind": "validation"}],
        validations=[["python", "-m", "pytest"]],
    )
    return intent


def test_real_intent_graph_and_empty_completion_are_evidence_not_authority(tmp_path):
    intent = seed(tmp_path)
    result = capture_intent_world_snapshot(intent, repository_id="repo:1")
    assert result == capture_intent_world_snapshot(intent, repository_id="repo:1")
    context = result["planning_context"]
    assert context["tasks"][0]["acceptance"][0]["criterion"] == "Addition holds"
    assert context["tasks"][0]["outputs"][0]["path"] == "add.py"
    assert len(context["goals"]) == 2
    assert context["active_plan_heads"][0]["plan_cid"] == "plan:1"
    assert context["event_watermark"] == intent.event_watermark()
    assert result["completion_projection"]["completion_receipts"] == []
    assert context["component_status"]["completion_root"] == "current"
    assert context["component_status"]["task_population"] == "current"
    assert context["component_status"]["claims"] == "unavailable"
    assert context["component_status"]["repository_tree"] == "unavailable"
    assert not context["schedulable"]
    assert not result["execution_authority"]
    assert not result["completion_authority"]
    intent.upsert_task(task_cid="task:2", task_alias="TEST-2", goal_cid="goal:root", status="ready")
    changed = capture_intent_world_snapshot(intent, repository_id="repo:1")
    assert changed["capture_cid"] != result["capture_cid"]
    scoped = capture_intent_world_snapshot(intent, repository_id="repo:1", task_cids=["task:1"])
    assert len(scoped["planning_context"]["tasks"]) == 1
    assert len(scoped["planning_context"]["goals"]) == 2


def test_caller_transaction_is_never_committed_or_rolled_back(tmp_path):
    intent = seed(tmp_path)
    with intent._connection(write=False) as connection:
        connection.execute("BEGIN TRANSACTION")
        bound = IntentRepository(bound_connection=connection)
        with pytest.raises(IntentWorldSnapshotError, match="caller transaction"):
            capture_intent_world_snapshot(bound, repository_id="repo:1")
        result = capture_intent_world_snapshot(
            bound, repository_id="repo:1", transaction_owned_by_caller=True
        )
        assert result["planning_context"]["tasks"][0]["task_cid"] == "task:1"
        connection.execute("ROLLBACK")
    with pytest.raises(IntentWorldSnapshotError, match="bound intent"):
        capture_intent_world_snapshot(
            intent, repository_id="repo:1", transaction_owned_by_caller=True
        )


def test_intent_capture_is_one_mvcc_snapshot_during_owner_update(tmp_path, monkeypatch):
    import duckdb

    intent = seed(tmp_path)
    previous_watermark = intent.event_watermark()
    original = IntentRepository.plan_projection
    # Exercise the existing typed owner's borrowed-connection contract. Native
    # file-mode connections exclusively serialize access before this point.
    reading = duckdb.connect(str(intent.database_path), config={"threads": 1})
    writing = reading.cursor()
    writer = IntentRepository(bound_connection=writing)

    def update_after_plan(reader, **kwargs):
        plan = original(reader, **kwargs)
        if reader.uses_bound_connection:
            writing.execute("BEGIN TRANSACTION")
            writer.upsert_task(
                task_cid="task:1", task_alias="TEST-1", goal_cid="goal:root", status="in_progress"
            )
            writing.execute("COMMIT")
        return plan

    monkeypatch.setattr(IntentRepository, "plan_projection", update_after_plan)
    try:
        reading.execute("BEGIN TRANSACTION")
        captured = capture_intent_world_snapshot(
            IntentRepository(bound_connection=reading),
            repository_id="repo:1",
            transaction_owned_by_caller=True,
        )
        reading.execute("ROLLBACK")
    finally:
        writing.close()
        reading.close()
    assert captured["planning_context"]["event_watermark"] == previous_watermark
    assert captured["completion_projection"]["task_states"][0]["status"] == "ready"
    assert captured["planning_context"]["tasks"][0]["status"] == "ready"
    assert intent.get_task("task:1")["status"] == "in_progress"
    assert intent.event_watermark() > previous_watermark


def test_real_world_evidence_reaches_native_planner_without_plan_admission(tmp_path):
    # Provider output is a deterministic test stub; there is no proof receipt or
    # admitted plan fixture. The context and its freshness come from actual DBs.
    from test.api.test_agent_supervisor_prompt_goal_planner import (
        _request,
        _scan,
        _encoded_proposal,
    )

    intent = seed(tmp_path)
    capture = capture_intent_world_snapshot(intent, repository_id="repo:1")
    persisted = persist_intent_world_snapshot(capture, output=tmp_path / "world", task_id="TEST-1")
    request = _request()
    scan = _scan(request)
    seen = []
    arguments = {
        "intent": intent,
        "artifact": tmp_path / "world/intent-world.json",
        "expected_sha256": persisted["artifact_sha256"],
        "task_id": "TEST-1",
        "repository_id": "repo:1",
        "router": lambda prompt: seen.append(json.loads(prompt)) or _encoded_proposal(scan),
    }
    result = generate_prompt_goal_graph_with_world(request, scan, **arguments)
    assert result.provider_succeeded
    world = seen[0]["constraints"]["constraint_summaries"][0]
    assert world["world_snapshot_cid"] == capture["snapshot"]["snapshot_cid"]
    assert world["plan_projection_cid"] == capture["plan_projection"]["projection_cid"]
    assert world["tasks"][0]["task_cid"] == "task:1"
    assert len(world["goals"]) == 2
    assert world["intent_freshness_checked"]
    assert "policy_root" in world["unavailable_components"]
    assert world["execution_authority"] is False
    assert world["completion_authority"] is False
    seen.clear()
    with pytest.raises(IntentWorldSnapshotError, match="byte bound"):
        generate_prompt_goal_graph_with_world(request, scan, **arguments, max_world_context_bytes=1)
    assert not seen
    intent.upsert_task(
        task_cid="task:1", task_alias="TEST-1", goal_cid="goal:root", status="in_progress"
    )
    with pytest.raises(IntentWorldSnapshotError, match="stale"):
        generate_prompt_goal_graph_with_world(request, scan, **arguments)
    assert not seen


def test_world_graph_is_required_in_actual_daemon_prompt(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalTask,
        TodoImplementationDaemon,
    )

    intent = seed(tmp_path)
    capture = capture_intent_world_snapshot(intent, repository_id="repo:1")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "add.py").write_text("def add(a,b): return a+b\n")
    (repo / "tasks.todo.md").write_text("# World integration\n")
    for args in [
        ("init", "-q"),
        ("add", "."),
        (
            "-c",
            "user.name=Integration",
            "-c",
            "user.email=local@example.invalid",
            "commit",
            "-qm",
            "seed",
        ),
    ]:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    persisted = persist_intent_world_snapshot(capture, output=repo / "world", task_id="TEST-1")
    daemon = TodoImplementationDaemon(
        todo_path=repo / "tasks.todo.md",
        state_path=tmp_path / "state/tasks.json",
        strategy_path=tmp_path / "state/strategy.json",
        events_path=tmp_path / "state/events.jsonl",
        repo_root=repo,
        task_header_prefix="## TEST-",
        merge_queue_dir=tmp_path / "merge-queue",
        worktree_root=tmp_path / "worktrees",
        validation_cache_dir=tmp_path / "validation-cache",
    )
    daemon._world_intent_repository = intent
    task = PortalTask(
        task_id="TEST-1",
        title="Preserve addition",
        status="ready",
        completion="manual",
        priority="P0",
        track="context",
        outputs=["add.py"],
        validation=["python -m pytest"],
        acceptance="Addition holds",
        metadata={
            "World context artifact": "world/intent-world.json",
            "World context sha256": persisted["artifact_sha256"],
            "World context repository": "repo:1",
        },
    )
    prompt = daemon._build_implementation_prompt(task, attempt=1)
    assert capture["snapshot"]["snapshot_cid"] in prompt
    assert "supervisor-intent-world-planning-context@1" in prompt
    assert "goal:root" in prompt and "goal:child" in prompt and "task:1" in prompt
    references = [
        row for row in json.loads(prompt)["evidence"] if row["kind"] == "intent-world-context"
    ]
    world = json.loads(
        "".join(row["summary"] for row in sorted(references, key=lambda row: row["reference_id"]))
    )
    assert "policy_root" in world["unavailable_components"]
    assert world["completion_authority"] is False
    assert world["execution_authority"] is False
    intent.upsert_task(
        task_cid="task:1", task_alias="TEST-1", goal_cid="goal:root", status="in_progress"
    )
    with pytest.raises(IntentWorldSnapshotError, match="stale"):
        daemon._build_implementation_prompt(task, attempt=1)


def test_native_claim_owner_records_are_bound_and_drift_is_refused(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        open_database_coordinator,
    )

    intent = seed(tmp_path)
    with open_database_coordinator(tmp_path / "coordination.duckdb") as coordinator:
        coordinator.register_task(task_cid="task:1", task_id="TEST-1")
        claim = coordinator.claim_task(task_cid="task:1", owner_session_id="session:1")
        result = capture_intent_world_snapshot(
            intent, repository_id="repo:1", coordinator=coordinator
        )
        component = result["snapshot"]["components"]["claims"]
        assert component["status"] == "current"
        payload = result["component_payloads"][component["cid"]]["material"]
        assert payload["task_claims"][0]["claim_id"] == claim.claim_id
        assert payload["lease_freshness_verified"] is False
        original = coordinator.coordination_registry_projection
        calls = 0

        def drift():
            nonlocal calls
            calls += 1
            if calls == 2:
                coordinator.register_task(task_cid="task:new", task_id="TEST-NEW")
            return original()

        monkeypatch.setattr(coordinator, "coordination_registry_projection", drift)
        with pytest.raises(IntentWorldSnapshotError, match="changed during capture"):
            capture_intent_world_snapshot(intent, repository_id="repo:1", coordinator=coordinator)


def test_persist_and_hydrate_native_ducklake(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_database import (
        ProgramWorldDatabase,
    )

    capture = capture_intent_world_snapshot(seed(tmp_path), repository_id="repo:1")
    result = persist_intent_world_snapshot(capture, output=tmp_path / "world", task_id="TEST-1")
    assert result["metadata"]["status"] == "projected"
    assert result["world_record"]["ducklake"]["status"] == "projected"
    assert json.loads((tmp_path / "world/intent-world.json").read_text()) == capture
    reopened = ProgramWorldDatabase(tmp_path / "world/world.duckdb", tmp_path / "world/world-lake")
    records = reopened.records_for_decision(task_id="TEST-1")
    assert records["records"][0]["payload"]["planning_context"] == capture["planning_context"]
    assert not records["decision_authority"]
    with pytest.raises(IntentWorldSnapshotError, match="captured task"):
        persist_intent_world_snapshot(capture, output=tmp_path / "other", task_id="OTHER")
    capture["planning_context"]["event_watermark"] += 1
    with pytest.raises(IntentWorldSnapshotError, match="content identity"):
        persist_intent_world_snapshot(capture, output=tmp_path / "tampered", task_id="TEST-1")


def test_loader_binds_live_intent_repository_task_and_artifact(tmp_path):
    intent = seed(tmp_path)
    capture = capture_intent_world_snapshot(intent, repository_id="repo:1")
    persisted = persist_intent_world_snapshot(capture, output=tmp_path / "world", task_id="TEST-1")
    arguments = {
        "artifact": tmp_path / "world/intent-world.json",
        "expected_sha256": persisted["artifact_sha256"],
        "task_id": "TEST-1",
        "repository_id": "repo:1",
    }
    context = load_intent_world_context(**arguments, intent=intent)
    assert context["intent_freshness_checked"]
    assert context["world_snapshot_cid"] == capture["snapshot"]["snapshot_cid"]
    assert not load_intent_world_context(**arguments)["intent_freshness_checked"]
    with pytest.raises(IntentWorldSnapshotError, match="task identity"):
        load_intent_world_context(**{**arguments, "task_id": "OTHER"})
    with pytest.raises(IntentWorldSnapshotError, match="repository identity"):
        load_intent_world_context(**{**arguments, "repository_id": "repo:other"})
    with pytest.raises(IntentWorldSnapshotError, match="digest mismatch"):
        load_intent_world_context(**{**arguments, "expected_sha256": "0" * 64})
    intent.upsert_task(
        task_cid="task:1", task_alias="TEST-1", goal_cid="goal:root", status="in_progress"
    )
    with pytest.raises(IntentWorldSnapshotError, match="stale"):
        load_intent_world_context(**arguments, intent=intent)


def test_live_datasets_root_is_verified_and_keeps_operational_owners_unavailable(tmp_path):
    pytest.importorskip("ipfs_datasets_py.logic.software_contracts.semantic_state")
    from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import (
        IpfsDatasetsSemanticStateProvider,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_index.scanner import RepositoryScanner

    source = tmp_path / "source"
    source.mkdir()
    (source / "add.py").write_text("def add(a, b):\n    return a + b\n")
    state = RepositoryScanner(repository_id="repo:1", namespace="world-test").scan(source)
    provider = IpfsDatasetsSemanticStateProvider()
    bundle = provider.build_semantic_state(state)
    view = provider.view_semantic_state_bundle(bundle)
    intent = seed(tmp_path)
    arguments = {
        "repository_id": "repo:1",
        "semantic_root_cid": view.root.root_cid,
        "get_semantic_block": bundle.blocks.__getitem__,
    }
    capture = capture_intent_world_snapshot(intent, **arguments)
    context = capture["planning_context"]
    assert context["semantic_root_cid"] == view.root.root_cid
    assert context["component_status"]["datasets_semantic_state_root"] == "current"
    assert context["component_status"]["capsule_index"] == "current"
    assert context["component_status"]["contract_root"] == "unavailable"
    assert context["component_status"]["policy_root"] == "unavailable"
    assert not context["schedulable"]
    with pytest.raises(IntentWorldSnapshotError, match="repository identity"):
        capture_intent_world_snapshot(intent, **{**arguments, "repository_id": "repo:other"})
    tampered = dict(bundle.blocks)
    tampered[view.root.capsule_index_cid] = b"{}"
    with pytest.raises(ValueError):
        capture_intent_world_snapshot(
            intent, **{**arguments, "get_semantic_block": tampered.__getitem__}
        )
