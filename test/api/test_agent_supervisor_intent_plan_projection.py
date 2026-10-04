"""Real-storage regression checks for the prompt materialization projection."""
import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import intent_repository as module


def seed(repo):
    repo.upsert_objective(objective_id="objective:test", objective_alias="O1", title="Repair")
    repo.upsert_goal(goal_cid="goal:root", goal_alias="G1", title="Repair", objective_id="objective:test")
    repo.upsert_goal(goal_cid="goal:child", goal_alias="G2", title="Verify", parent_goal_cid="goal:root")
    repo.link_goal_edge(parent_goal_cid="goal:root", child_goal_cid="goal:child", edge_kind="depends_on")
    repo.upsert_plan(plan_cid="plan:test", goal_cid="goal:root", plan_alias="P1", status="active")
    for number in (1, 2):
        repo.upsert_task(
            task_cid=f"task:{number}", task_alias=f"T{number}", goal_cid="goal:root",
            plan_cid="plan:test", objective_id="objective:test", ordinal=number,
            body={"title": f"Step {number}"},
            dependencies=["task:1"] if number == 2 else [],
            outputs=[{"path": "example.py", "effect": "modify"}],
            acceptance=[{"criterion": "public tests pass", "evidence_kind": "validation"}],
            validations=[["python", "-m", "pytest"]],
        )


def test_full_projection_reopens_without_mutation_and_preserves_specs(tmp_path):
    path = tmp_path / "intent.duckdb"
    with module.open_intent_repository(path) as repo:
        seed(repo)
        before = repo.snapshot().event_watermark
        projected = repo.plan_projection()
        assert projected["schema"] == module.INTENT_PLAN_PROJECTION_SCHEMA
        assert [len(projected[name]) for name in ("objectives", "goals", "goal_edges", "plans", "tasks")] == [1, 2, 1, 1, 2]
        task = projected["tasks"][1]
        assert task["dependencies"] == [{"dependency_task_cid": "task:1", "kind": "depends_on"}]
        assert task["outputs"][0]["effect"] == {"path": "example.py", "effect": "modify"}
        assert task["acceptance"][0]["criterion"] == "public tests pass"
        assert task["validations"][0]["argv"] == ["python", "-m", "pytest"]
        assert task["spec_cid"] == module.task_projection_spec_cid(task)
        assert repo.snapshot().event_watermark == before
        with repo.read_session():
            assert repo.plan_projection() == projected
    with module.open_intent_repository(path) as repo:
        assert repo.plan_projection() == projected
        with repo._connection(write=True) as connection:
            connection.execute("UPDATE task_validations SET argv_json = ? WHERE task_cid = ?", ['["python", "-m", "pytest", "-x"]', "task:2"])
        changed = repo.plan_projection()
        assert changed["projection_cid"] != projected["projection_cid"]
        assert changed["tasks"][1]["spec_cid"] != task["spec_cid"]


def test_selection_is_exact_and_retains_dependency_context(tmp_path):
    with module.open_intent_repository(tmp_path / "intent.duckdb") as repo:
        seed(repo)
        result = repo.plan_projection(task_cids=["task:2"])
        assert [task["task_cid"] for task in result["tasks"]] == ["task:2"]
        assert result["tasks"][0]["dependencies"][0]["dependency_task_cid"] == "task:1"
        with pytest.raises(KeyError):
            repo.plan_projection(task_cids=["T2"])
        with pytest.raises(module.IntentRepositoryBoundsError):
            repo.plan_projection(task_cids=["task:2"] * 1001)
        with pytest.raises(module.IntentRepositoryError):
            repo.plan_projection(task_cids="task:2")


@pytest.mark.parametrize("raw", ['', '{"x":1,"x":2}', '{"x":{"a":1,"a":2}}', '{"x":NaN}', 'null', '[]'])
def test_corrupt_persisted_json_is_rejected(tmp_path, raw):
    with module.open_intent_repository(tmp_path / "intent.duckdb") as repo:
        seed(repo)
        with repo._connection(write=True) as connection:
            connection.execute("UPDATE tasks SET extension_json = ? WHERE task_cid = 'task:1'", [raw])
        with pytest.raises(module.IntentRepositoryIntegrityError):
            repo.plan_projection()


def test_bounds_fail_instead_of_returning_truncated_plan(tmp_path, monkeypatch):
    with module.open_intent_repository(tmp_path / "intent.duckdb") as repo:
        seed(repo)
        with monkeypatch.context() as patch:
            patch.setattr(module, "MAX_PROJECTION_RECORDS", 1)
            with pytest.raises(module.IntentRepositoryBoundsError, match="record count"):
                repo.plan_projection()
        monkeypatch.setattr(module, "MAX_PLAN_PROJECTION_BYTES", 10)
        with pytest.raises(module.IntentRepositoryBoundsError, match="byte bound"):
            repo.plan_projection()


def test_empty_projection_is_stable_and_unknown_task_rejected(tmp_path):
    with module.open_intent_repository(tmp_path / "intent.duckdb") as repo:
        first = repo.plan_projection()
        assert first["tasks"] == []
        assert first == repo.plan_projection()
        with pytest.raises(KeyError):
            repo.plan_projection(task_cids=["task:missing"])


def test_borrowed_transaction_is_not_committed(tmp_path):
    with module.open_intent_repository(tmp_path / "intent.duckdb") as repo:
        seed(repo)
        with repo._connection(write=True) as connection:
            connection.execute("UPDATE tasks SET priority = 'P0' WHERE task_cid = 'task:1'")
            with module.IntentRepository(bound_connection=connection) as bound:
                assert bound.plan_projection()["tasks"][0]["priority"] == "P0"
            # A caller-owned transaction must still be active.
            connection.execute("UPDATE tasks SET priority = 'P1' WHERE task_cid = 'task:1'")
        assert repo.plan_projection()["tasks"][0]["priority"] == "P1"
