"""Pinned continuation diagnostics preserve old task origins and no authority.

Every database is a private temporary fixture. No native constructor, worker,
provider, service command or host database may be reached by this cohort.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import native_catalogue_probe as probe

NS = "fixture-legal-ir"
ROOT = "plan:original"
NEXT = "plan:continuation"
TIP = "plan:tip"
GOAL = "goal:programme"
CHILD_GOAL = "goal:representation"
_REAL_RUN = subprocess.run
_REAL_POPEN = subprocess.Popen


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def fixture(tmp_path, *, plan_count=2, old_count=2):
    import duckdb

    plans = []
    bodies = {}
    roots = [ROOT, NEXT, TIP][:plan_count]
    for index, cid in enumerate(roots):
        body = ({"plan_cid": ROOT, "goal_cid": GOAL, "plan_alias": "original",
                 "constraints": ["Retain source and review gates"]} if index == 0 else {
            "repository_tree_id": "tree:" + cid.split(":")[1],
            "continuation_of": roots[index - 1], "revision_cid": "revision:" + str(index),
            "delta_cid": "delta:" + str(index), "idempotency_key": "steer:" + str(index)})
        if index + 1 < len(roots):
            body["continued_by"] = roots[index + 1]
        raw = encoded(body)
        bodies[cid] = raw
        plans.append({"plan_cid": cid, "goal_cid": GOAL,
            "status": "continued" if index + 1 < len(roots) else "active",
            "revision": 2 if index + 1 < len(roots) else 1,
            "body_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            **{key: body.get(key) for key in probe._PLAN_BODY_BINDINGS}})
    tasks = [{"task_cid": "task:old:" + str(i), "task_alias": "LIR-" + str(9000 + i),
              "goal_cid": GOAL, "plan_cid": ROOT, "objective_id": "objective:law",
              "dependencies": [] if i == 0 else ["task:old:0"]} for i in range(old_count)]
    tasks.append({"task_cid": "task:new:4096", "task_alias": "LIR-4096",
        "goal_cid": CHILD_GOAL, "plan_cid": NEXT, "objective_id": "objective:law",
        "dependencies": ["task:old:0"]})
    goals = [{"goal_cid": GOAL, "goal_alias": "LIR-G000", "objective_id": "objective:law",
              "parent_goal_cid": ""}, {"goal_cid": CHILD_GOAL, "goal_alias": "LIR-G119",
              "objective_id": "", "parent_goal_cid": GOAL}]
    population = {"schema": probe.LINEAGE_POPULATION_SCHEMA, "board_namespace": NS,
        "plan_root_cid": roots[-1], "repository_tree_id": plans[-1]["repository_tree_id"],
        "plans": plans, "tasks": tasks, "goals": goals}
    path = tmp_path / "population.json"
    database = tmp_path / "fixture.duckdb"
    con = duckdb.connect(str(database))
    try:
        statements = [
            "CREATE TABLE tasks (task_cid VARCHAR, task_alias VARCHAR, goal_cid VARCHAR, plan_cid VARCHAR, objective_id VARCHAR, status VARCHAR, revision BIGINT, body_json VARCHAR)",
            "CREATE TABLE goals (goal_cid VARCHAR, goal_alias VARCHAR, objective_id VARCHAR, parent_goal_cid VARCHAR, status VARCHAR, revision BIGINT)",
            "CREATE TABLE plans (plan_cid VARCHAR, goal_cid VARCHAR, status VARCHAR, revision BIGINT, body_json VARCHAR)",
            "CREATE TABLE domain_events (global_sequence BIGINT)",
            "CREATE TABLE task_attempts (task_cid VARCHAR, attempt_id VARCHAR, attempt_number BIGINT, status VARCHAR)",
            "CREATE TABLE provider_invocations (task_cid VARCHAR, attempt_id VARCHAR, status VARCHAR)",
            "CREATE TABLE task_claims (task_cid VARCHAR, state VARCHAR)",
            "CREATE TABLE task_dependencies (task_cid VARCHAR, dependency_task_cid VARCHAR)",
            "CREATE TABLE store_generations (generation BIGINT, schema_revision BIGINT, database_uuid VARCHAR)"]
        for statement in statements:
            con.execute(statement)
        con.execute("INSERT INTO domain_events VALUES (214)")
        con.execute("INSERT INTO store_generations VALUES (19,5,'fixture-database')")
        for goal in goals:
            con.execute("INSERT INTO goals VALUES (?,?,?,?,?,?)", [*(goal[k] for k in
                ("goal_cid", "goal_alias", "objective_id", "parent_goal_cid")), "open", 1])
        for plan in plans:
            con.execute("INSERT INTO plans VALUES (?,?,?,?,?)", [plan["plan_cid"], GOAL,
                plan["status"], plan["revision"], bodies[plan["plan_cid"]]])
        for index, task in enumerate(tasks):
            body = {"board_namespace": NS}
            if index == 0:
                body["completion_receipt"] = {"attempt_number": 2, "max_task_attempts": 2,
                    "attempt_budget_exhausted": True, "automatic_retry_admitted": False,
                    "failure_kind": "administrative_operator_stop_outcome_unknown"}
            con.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?)", [*(task[k] for k in
                ("task_cid", "task_alias", "goal_cid", "plan_cid", "objective_id")),
                "blocked", 13 if index == 0 else 1, encoded(body)])
            for dep in task["dependencies"]:
                con.execute("INSERT INTO task_dependencies VALUES (?,?)", [task["task_cid"], dep])
    finally:
        con.close()
    result = SimpleNamespace(path=path, database=database, population=population,
        bodies=bodies, roots=roots, tmp_path=tmp_path, connect=duckdb.connect)
    save(result)
    return result


def save(case):
    case.path.write_text(encoded(case.population))


def arguments(case):
    return dict(database_path=str(case.database), population_path=str(case.path),
        board_namespace=NS, plan_root_cid=case.population["plan_root_cid"],
        repository_tree_id=case.population["repository_tree_id"],
        population_sha256=hashlib.sha256(case.path.read_bytes()).hexdigest())


def mutate(case, sql, parameters=()):
    assert case.database.parent == case.tmp_path
    con = case.connect(str(case.database))
    try:
        con.execute(sql, parameters)
    finally:
        con.close()


def read(case, monkeypatch):
    import duckdb

    def private_connect(path, **kwargs):
        assert Path(path) == case.database
        assert kwargs["read_only"] is True
        assert kwargs["config"]["enable_external_access"] is False
        return case.connect(path, **kwargs)
    with monkeypatch.context() as guard:
        guard.setattr(duckdb, "connect", private_connect)
        return probe._read_catalogue(arguments(case))


@pytest.fixture(autouse=True)
def no_worker_or_host_command(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("unexpected worker, service or host command")
    monkeypatch.setattr(probe.subprocess, "run", forbidden)
    monkeypatch.setattr(probe.subprocess, "Popen", forbidden)


def test_two_root_addition_keeps_old_task_origin_and_unknown_history(tmp_path, monkeypatch):
    case = fixture(tmp_path)
    before = case.database.read_bytes()
    result = read(case, monkeypatch)
    assert [t["plan_cid"] for t in result["tasks"]].count(ROOT) == 2
    assert next(t for t in result["tasks"] if t["task_alias"] == "LIR-4096")["plan_cid"] == NEXT
    assert result["population_schema"] == probe.LINEAGE_POPULATION_SCHEMA
    assert result["plan_lineage"] == case.population["plans"]
    assert result["tasks"][1]["completion_receipt_attempt_number"] == 2
    assert result["attempt_rows"] == result["provider_status_rows"] == result["claim_state_rows"] == []
    assert result["database_and_wal_metadata_unchanged"] is True
    assert case.database.read_bytes() == before and not Path(str(case.database) + ".wal").exists()


def test_three_root_chain_allows_tip_without_own_tasks(tmp_path, monkeypatch):
    case = fixture(tmp_path, plan_count=3)
    result = read(case, monkeypatch)
    assert result["plan_lineage"][-1]["plan_cid"] == TIP
    assert not any(t["plan_cid"] == TIP for t in result["tasks"])


def test_actual_original64_plus_one_child_population(tmp_path, monkeypatch):
    case = fixture(tmp_path, old_count=64)
    result = read(case, monkeypatch)
    assert len(result["tasks"]) == 65 and result["dependency_count"] == 64
    assert sum(t["plan_cid"] == ROOT for t in result["tasks"]) == 64


def test_observer_remains_diagnostic_after_exact_lineage_read(tmp_path, monkeypatch):
    case = fixture(tmp_path)
    def worker(argv, **kwargs):
        assert argv[-2] == "--worker"
        assert json.loads(argv[-1]) == arguments(case)
        result = read(case, monkeypatch)
        return SimpleNamespace(returncode=0, stdout=encoded(result).encode())
    monkeypatch.setattr(probe.subprocess, "run", worker)
    args = arguments(case)
    result = probe.observe_native_catalogue(case.database, case.path, board_id="fixture-board",
        **{k:v for k,v in args.items() if k not in {"database_path", "population_path"}})
    diagnostic = result["details"]["native_catalogue"]
    assert result["health"] == "blocked" and result["progress_token"]
    assert diagnostic["activation_state"] == "not_activated"
    assert diagnostic["executor_health"] == diagnostic["provider_outcome_knowledge"] == "not_established"
    assert diagnostic["external_outcome_knowledge"] == "unknown_observed"
    assert diagnostic["attempt_history_complete"] is False
    assert diagnostic["completion_authority"] is result["complete"] is result["completion_candidate"] is result["busy"] is False
    assert "recovery_action" not in result


def test_actual_isolated_v2_worker_reads_only_the_private_fixture(tmp_path, monkeypatch):
    case = fixture(tmp_path, plan_count=3)
    expected = arguments(case)
    before = case.database.read_bytes()
    observed_calls = []

    def check_command(argv):
        assert type(argv) is list and len(argv) == 6
        assert argv[:5] == [sys.executable, "-I", "-B",
                           str(Path(probe.__file__).resolve()), "--worker"]
        supplied = json.loads(argv[-1])
        assert supplied == expected
        assert Path(supplied["database_path"]).resolve() == case.database.resolve()
        assert Path(supplied["population_path"]).resolve() == case.path.resolve()
        assert case.database.parent.resolve() == case.path.parent.resolve() == tmp_path.resolve()

    def private_popen(argv, **kwargs):
        check_command(argv)
        assert kwargs.get("shell", False) is False
        assert kwargs["preexec_fn"] is probe._limits
        return _REAL_POPEN(argv, **kwargs)

    def private_worker(argv, **kwargs):
        check_command(argv)
        assert set(kwargs) == {"capture_output", "timeout", "env", "preexec_fn"}
        assert kwargs["capture_output"] is True and kwargs["timeout"] == 15
        assert kwargs["preexec_fn"] is probe._limits
        assert not ({"HOME", "PYTHONPATH", "PYTHONHOME"} & set(kwargs["env"]))
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        observed_calls.append(tuple(argv[:5]))
        # The original autouse gate remains in force for every other argv;
        # even the real run's lower-level Popen is restricted to this worker.
        with monkeypatch.context() as guard:
            guard.setattr(probe.subprocess, "Popen", private_popen)
            return _REAL_RUN(argv, **kwargs)

    monkeypatch.setattr(probe.subprocess, "run", private_worker)
    result = probe.observe_native_catalogue(case.database, case.path, board_id="fixture-board",
        **{key: value for key, value in expected.items()
           if key not in {"database_path", "population_path"}})
    diagnostic = result["details"]["native_catalogue"]
    assert len(observed_calls) == 1 and diagnostic["fresh_read"] is True
    assert diagnostic["population_schema"] == probe.LINEAGE_POPULATION_SCHEMA
    assert diagnostic["plan_lineage"] == case.population["plans"]
    assert diagnostic["activation_state"] == "not_activated"
    assert diagnostic["executor_health"] == "not_established"
    assert diagnostic["attempt_history_complete"] is False
    assert diagnostic["external_outcome_knowledge"] == "unknown_observed"
    assert result["health"] == "blocked" and result["complete"] is False
    assert result["completion_candidate"] is result["busy"] is diagnostic["completion_authority"] is False
    assert case.database.read_bytes() == before
    assert not Path(str(case.database) + ".wal").exists()


@pytest.mark.parametrize("damage", ["unknown_top", "unknown_task_field", "unknown_goal_field",
    "unknown_plan_field", "duplicate_task", "duplicate_alias", "duplicate_goal", "duplicate_plan",
    "unknown_task_root", "wrong_tip", "wrong_tree", "cycle", "unrelated", "unordered", "fork",
    "wrong_plan_goal", "unrelated_goal", "goal_cycle", "missing_continuation_tree",
    "missing_delta", "bad_revision", "bad_digest", "wrong_status", "too_many_plans"])
def test_closed_lineage_manifest_refuses_before_any_worker_or_database(tmp_path, damage):
    case = fixture(tmp_path, plan_count=3)
    p = case.population
    if damage == "unknown_top": p["automatic_retry_admitted"] = True
    elif damage == "unknown_task_field": p["tasks"][0]["origin_wildcard"] = True
    elif damage == "unknown_goal_field": p["goals"][0]["wildcard"] = True
    elif damage == "unknown_plan_field": p["plans"][0]["wildcard"] = True
    elif damage == "duplicate_task": p["tasks"].append(copy.deepcopy(p["tasks"][0]))
    elif damage == "duplicate_alias": p["tasks"][1]["task_alias"] = p["tasks"][0]["task_alias"]
    elif damage == "duplicate_goal": p["goals"].append(copy.deepcopy(p["goals"][0]))
    elif damage == "duplicate_plan": p["plans"].append(copy.deepcopy(p["plans"][0]))
    elif damage == "unknown_task_root": p["tasks"][0]["plan_cid"] = "plan:unknown"
    elif damage == "wrong_tip": p["plan_root_cid"] = ROOT
    elif damage == "wrong_tree": p["repository_tree_id"] = "tree:other"
    elif damage == "cycle": p["plans"][0]["continuation_of"] = TIP
    elif damage == "unrelated": p["plans"][1]["continuation_of"] = "plan:unrelated"
    elif damage == "unordered": p["plans"].reverse()
    elif damage == "fork": p["plans"][0]["continued_by"] = TIP
    elif damage == "wrong_plan_goal": p["plans"][1]["goal_cid"] = CHILD_GOAL
    elif damage == "unrelated_goal": p["goals"][1]["parent_goal_cid"] = "goal:unrelated"
    elif damage == "goal_cycle": p["goals"][1]["parent_goal_cid"] = CHILD_GOAL
    elif damage == "missing_continuation_tree": p["plans"][1]["repository_tree_id"] = None
    elif damage == "missing_delta": p["plans"][1]["delta_cid"] = None
    elif damage == "bad_revision": p["plans"][1]["revision"] = True
    elif damage == "bad_digest": p["plans"][0]["body_sha256"] = "wildcard"
    elif damage == "wrong_status": p["plans"][0]["status"] = "active"
    elif damage == "too_many_plans": p["plans"] = p["plans"] * (probe.MAX_PLAN_ROOTS + 1)
    save(case)
    args = arguments(case)
    result = probe.observe_native_catalogue(case.database, case.path, board_id="fixture-board",
        **{k:v for k,v in args.items() if k not in {"database_path", "population_path"}})
    assert result["health"] == "unknown" and result["progress_token"] == ""
    assert result["details"]["native_catalogue"]["fresh_read"] is False


@pytest.mark.parametrize("damage", ["global_plan_rewrite", "missing_task", "extra_task",
    "wrong_namespace", "wrong_goal", "wrong_objective", "goal_alias", "goal_parent", "extra_goal",
    "missing_plan", "extra_plan", "extra_child_plan", "plan_status", "plan_revision", "plan_goal",
    "plan_body_hash", "plan_link", "plan_tree", "genesis_null", "genesis_empty", "dependencies"])
def test_native_scope_or_lineage_drift_cannot_become_a_matching_read(tmp_path, monkeypatch, damage):
    case = fixture(tmp_path)
    if damage == "global_plan_rewrite": mutate(case, "UPDATE tasks SET plan_cid=?", [NEXT])
    elif damage == "missing_task": mutate(case, "DELETE FROM tasks WHERE task_cid='task:new:4096'")
    elif damage == "extra_task": mutate(case, "INSERT INTO tasks SELECT 'task:extra','LIR-X',goal_cid,plan_cid,objective_id,status,revision,body_json FROM tasks LIMIT 1")
    elif damage == "wrong_namespace": mutate(case, "UPDATE tasks SET body_json=? WHERE task_cid='task:new:4096'", [encoded({"board_namespace":"other"})])
    elif damage == "wrong_goal": mutate(case, "UPDATE tasks SET goal_cid=? WHERE task_cid='task:new:4096'", [GOAL])
    elif damage == "wrong_objective": mutate(case, "UPDATE tasks SET objective_id='objective:other'")
    elif damage == "goal_alias": mutate(case, "UPDATE goals SET goal_alias='OTHER' WHERE goal_cid=?", [CHILD_GOAL])
    elif damage == "goal_parent": mutate(case, "UPDATE goals SET parent_goal_cid='' WHERE goal_cid=?", [CHILD_GOAL])
    elif damage == "extra_goal": mutate(case, "INSERT INTO goals VALUES ('goal:extra','OTHER','',?,'open',1)", [GOAL])
    elif damage == "missing_plan": mutate(case, "DELETE FROM plans WHERE plan_cid=?", [ROOT])
    elif damage in {"extra_plan", "extra_child_plan"}:
        mutate(case, "INSERT INTO plans VALUES ('plan:extra',?,'active',1,'{}')", [GOAL if damage == "extra_plan" else CHILD_GOAL])
    elif damage == "plan_status": mutate(case, "UPDATE plans SET status='active' WHERE plan_cid=?", [ROOT])
    elif damage == "plan_revision": mutate(case, "UPDATE plans SET revision=revision+1 WHERE plan_cid=?", [ROOT])
    elif damage == "plan_goal": mutate(case, "UPDATE plans SET goal_cid=? WHERE plan_cid=?", [CHILD_GOAL, NEXT])
    elif damage in {"plan_body_hash", "plan_link", "plan_tree", "genesis_null", "genesis_empty"}:
        cid = ROOT if damage.startswith("genesis") else NEXT
        body = json.loads(case.bodies[cid])
        if damage == "plan_body_hash": body["extra"] = "changed"
        elif damage == "plan_link": body["continuation_of"] = "plan:unrelated"
        elif damage == "plan_tree": body["repository_tree_id"] = "tree:other"
        else: body["repository_tree_id"] = None if damage == "genesis_null" else ""
        raw = encoded(body)
        mutate(case, "UPDATE plans SET body_json=? WHERE plan_cid=?", [raw, cid])
        if damage != "plan_body_hash":
            # Rehashing the body cannot disguise a wrong declared binding or
            # turn literal genesis absence into an explicit null/empty key.
            next(p for p in case.population["plans"] if p["plan_cid"] == cid)["body_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
            save(case)
    elif damage == "dependencies": mutate(case, "DELETE FROM task_dependencies")
    with pytest.raises(probe.InvalidCatalogue):
        read(case, monkeypatch)


def test_v1_still_refuses_mixed_origins_before_worker(tmp_path):
    case = fixture(tmp_path)
    case.population["schema"] = probe.POPULATION_SCHEMA
    save(case)
    with pytest.raises(probe.InvalidCatalogue, match="population_task_invalid"):
        probe._population(arguments(case))


@pytest.mark.parametrize("metadata", [{}, {"lineage": "retained legacy note"},
    {"lineage": [{"plan_cid": "plan:forged", "automatic_retry_admitted": True}]}])
def test_v1_single_root_contract_and_legacy_plan_columns_are_unchanged(tmp_path, monkeypatch, metadata):
    case = fixture(tmp_path)
    mutate(case, "DELETE FROM tasks WHERE task_cid='task:new:4096'")
    mutate(case, "DELETE FROM task_dependencies WHERE task_cid='task:new:4096'")
    case.population.update(schema=probe.POPULATION_SCHEMA, plan_root_cid=ROOT,
        repository_tree_id="tree:original", tasks=case.population["tasks"][:-1],
        plans=[{"plan_cid": ROOT, "goal_cid": GOAL, **metadata}])
    save(case)
    def forbidden(*a, **kw):
        pytest.fail("legacy JSON metadata must not select the v2 lineage profile")
    monkeypatch.setattr(probe, "_read_plan_lineage", forbidden)
    # The old profile never depended on plan status/body/revision or the extra
    # continuation row. Preserve its original two-column exact root check.
    result = read(case, monkeypatch)
    assert len(result["tasks"]) == 2
    assert "population_schema" not in result and "plan_lineage" not in result
