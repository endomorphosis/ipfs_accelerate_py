"""Direct catalogue observations retain wait/authority and read-only boundaries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import native_catalogue_probe as probe

NAMESPACE = "legal-ir-span-graph-lean-refinement-20261016"
ROOT = "baguqeeraexpectedplan"
TREE = "baguqeeraexpectedtree"


@pytest.fixture
def catalogue(tmp_path):
    import duckdb

    tasks = [{"task_cid": f"baguqeeratask{letter}", "task_alias": f"LIR-900{number}",
              "goal_cid": "baguqeeragoal", "plan_cid": ROOT, "objective_id": "objective:one"}
             for number, letter in enumerate("ab", 1)]
    population = {"schema": probe.POPULATION_SCHEMA, "board_namespace": NAMESPACE,
                  "plan_root_cid": ROOT, "repository_tree_id": TREE, "tasks": tasks,
                  "plans": [{"plan_cid": ROOT, "goal_cid": "baguqeeragoal"}],
                  "goals": [{"goal_cid": "baguqeeragoal", "goal_alias": "LIR-G090",
                             "objective_id": "objective:one", "parent_goal_cid": ""}]}
    pop = tmp_path / "population.json"
    pop.write_text(json.dumps(population))
    database = tmp_path / "control.duckdb"
    connection = duckdb.connect(str(database))
    try:
        connection.execute("CREATE TABLE tasks (task_cid VARCHAR, task_alias VARCHAR, goal_cid VARCHAR, plan_cid VARCHAR, objective_id VARCHAR, status VARCHAR, revision BIGINT, body_json VARCHAR)")
        connection.execute("CREATE TABLE plans (plan_cid VARCHAR, goal_cid VARCHAR)")
        connection.execute("INSERT INTO plans VALUES (?,?)", [ROOT, "baguqeeragoal"])
        connection.execute("CREATE TABLE goals (goal_cid VARCHAR, goal_alias VARCHAR, objective_id VARCHAR, parent_goal_cid VARCHAR, status VARCHAR, revision BIGINT)")
        connection.execute("CREATE TABLE domain_events (global_sequence BIGINT)")
        connection.execute("CREATE TABLE task_attempts (task_cid VARCHAR, attempt_id VARCHAR, attempt_number BIGINT, status VARCHAR)")
        connection.execute("CREATE TABLE provider_invocations (task_cid VARCHAR, attempt_id VARCHAR, status VARCHAR)")
        connection.execute("CREATE TABLE task_claims (task_cid VARCHAR, state VARCHAR)")
        connection.execute("CREATE TABLE task_dependencies (task_cid VARCHAR, dependency_task_cid VARCHAR, kind VARCHAR)")
        connection.execute("CREATE TABLE store_generations (generation BIGINT, schema_revision BIGINT, database_uuid VARCHAR)")
        connection.execute("INSERT INTO goals VALUES ('baguqeeragoal','LIR-G090','objective:one','','open',2)")
        connection.execute("INSERT INTO domain_events VALUES (7)")
        connection.execute("INSERT INTO store_generations VALUES (3,5,'test-database-uuid')")
        for number, task in enumerate(tasks):
            body = {"board_namespace": NAMESPACE}
            if number == 0:
                body["completion_receipt"] = {"attempt_number": 2, "automatic_retry_admitted": False,
                    "max_task_attempts": 2, "attempt_budget_exhausted": True,
                    "failure_kind": "provider_outcome_unknown"}
            connection.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?)",
                [*(task[key] for key in ("task_cid", "task_alias", "goal_cid", "plan_cid", "objective_id")),
                 "blocked", 13 if number == 0 else 2, json.dumps(body)])
    finally:
        connection.close()
    return database, pop


def observe(catalogue, **changes):
    database, population = catalogue
    arguments = dict(board_id="legal-ir-scientific", board_namespace=NAMESPACE,
                     plan_root_cid=ROOT, repository_tree_id=TREE,
                     population_sha256=(changes["population_sha256"] if "population_sha256" in changes
                                        else hashlib.sha256(population.read_bytes()).hexdigest()))
    arguments.update(changes)
    return probe.observe_native_catalogue(database, population, **arguments)


def change_database(catalogue, statement, parameters=()):
    import duckdb
    connection = duckdb.connect(str(catalogue[0]))
    try:
        connection.execute(statement, parameters)
    finally:
        connection.close()


def test_current_blocked_catalogue_is_observed_without_live_executor_or_authority(catalogue):
    result = observe(catalogue)
    diagnostic = result["details"]["native_catalogue"]
    assert result["health"] == "blocked"
    assert "native_catalogue_expected_blocked" in result["reason_codes"]
    assert diagnostic["fresh_read"] is True and diagnostic["counts"] == {"blocked": 2}
    assert diagnostic["activation_state"] == "not_activated"
    assert diagnostic["executor_health"] == "not_established"
    assert diagnostic["completion_authority"] is result["complete"] is result["completion_candidate"] is result["busy"] is False
    assert result["details"]["wait"]["kind"] == "dependency_or_review"
    assert "recovery_action" not in result


def test_unknown_provider_and_exhausted_receipt_survive_empty_normalized_tables(catalogue):
    result = observe(catalogue)
    diagnostic = result["details"]["native_catalogue"]
    assert diagnostic["normalized_attempt_rows"] == diagnostic["normalized_provider_status_rows"] == []
    assert diagnostic["tasks"][0]["completion_receipt_attempt_number"] == 2
    assert diagnostic["tasks"][0]["automatic_retry_admitted"] is False
    assert diagnostic["attempt_history_complete"] is False
    assert diagnostic["empty_normalized_rows_establish_zero_attempts_or_provider_success"] is False
    assert diagnostic["provider_outcome_knowledge"] == "unknown_observed"
    assert {"native_catalogue_attempt_budget_exhausted", "native_catalogue_provider_outcome_unknown",
            "native_catalogue_retry_not_admitted"} <= set(result["reason_codes"])


def test_absent_budget_and_outcome_fields_are_unknown_not_default_zero_or_success(catalogue):
    change_database(catalogue, "UPDATE tasks SET body_json=?", [json.dumps({"board_namespace": NAMESPACE})])
    result = observe(catalogue)
    diagnostic = result["details"]["native_catalogue"]
    assert diagnostic["provider_outcome_knowledge"] == "not_established"
    assert all(task["receipt_max_task_attempts"] is None for task in diagnostic["tasks"])
    assert all(task["completion_receipt_attempt_number"] is None for task in diagnostic["tasks"])
    assert diagnostic["exhausted_receipt_task_cids"] == []
    assert "native_catalogue_attempt_budget_exhausted" not in result["reason_codes"]


def test_direct_read_only_connection_changes_no_database_or_wal(catalogue):
    database, _ = catalogue
    before = database.read_bytes()
    assert not Path(str(database) + ".wal").exists()
    result = observe(catalogue)
    assert result["details"]["native_catalogue"]["database_and_wal_metadata_unchanged"] is True
    assert database.read_bytes() == before and not Path(str(database) + ".wal").exists()


@pytest.mark.parametrize("change", ["namespace", "missing_task", "extra_task", "task_identity", "goal_identity"])
def test_current_scope_and_population_drift_are_unknown(catalogue, change):
    if change == "namespace":
        change_database(catalogue, "UPDATE tasks SET body_json=? WHERE task_alias='LIR-9002'",
                        [json.dumps({"board_namespace": "different-board"})])
    elif change == "missing_task": change_database(catalogue, "DELETE FROM tasks WHERE task_alias='LIR-9002'")
    elif change == "extra_task":
        change_database(catalogue, "INSERT INTO tasks SELECT 'baguqeeraextra','LIR-X',goal_cid,plan_cid,objective_id,status,revision,body_json FROM tasks LIMIT 1")
    elif change == "task_identity": change_database(catalogue, "UPDATE tasks SET plan_cid='other-plan'")
    elif change == "goal_identity": change_database(catalogue, "UPDATE goals SET objective_id='other-objective'")
    result = observe(catalogue)
    assert result["health"] == "unknown" and not result["progress_token"]
    assert result["details"]["native_catalogue"]["fresh_read"] is False
    assert "recovery_action" not in result


def test_writer_lock_is_unavailable_not_a_reason_to_repair_or_migrate(catalogue):
    import duckdb
    writer = duckdb.connect(str(catalogue[0]))
    try:
        result = observe(catalogue)
        assert result["health"] == "unknown"
        assert result["reason_codes"] == ["native_catalogue_read_unavailable"]
        assert "recovery_action" not in result
    finally:
        writer.close()


def test_ready_status_still_does_not_establish_executor_health(catalogue):
    change_database(catalogue, "UPDATE tasks SET status='todo' WHERE task_alias='LIR-9002'")
    result = observe(catalogue)
    assert result["health"] == "blocked" and result["busy"] is False
    assert result["details"]["native_catalogue"]["ready_status_count"] == 1
    assert "native_catalogue_executor_unverified" in result["reason_codes"]


def test_progress_token_ignores_read_clock_but_tracks_native_revision(catalogue):
    first = observe(catalogue)
    assert observe(catalogue)["progress_token"] == first["progress_token"]
    change_database(catalogue, "UPDATE tasks SET revision=revision+1 WHERE task_alias='LIR-9002'")
    assert observe(catalogue)["progress_token"] != first["progress_token"]


@pytest.mark.parametrize("kind", ["missing", "symlink", "fifo", "oversized", "bad_digest", "duplicate_json_key"])
def test_population_file_boundary_refuses_without_worker(catalogue, monkeypatch, kind):
    population = catalogue[1]
    changes = {}
    if kind == "missing": population.unlink()
    elif kind == "symlink":
        target = population.with_name("target.json")
        population.rename(target)
        population.symlink_to(target)
    elif kind == "fifo":
        population.unlink()
        os.mkfifo(population)
    elif kind == "oversized":
        with population.open("wb") as handle: handle.truncate(probe.MAX_POPULATION_BYTES + 1)
    elif kind == "bad_digest": changes["population_sha256"] = "0" * 64
    elif kind == "duplicate_json_key": population.write_text('{"schema":1,"schema":2}')
    digest = "0" * 64 if kind in {"missing", "fifo"} else hashlib.sha256(population.read_bytes()).hexdigest()
    changes.setdefault("population_sha256", digest)
    def forbidden(*args, **kwargs): pytest.fail("invalid population started worker")
    monkeypatch.setattr(probe.subprocess, "run", forbidden)
    result = observe(catalogue, **changes)
    assert result["health"] == "unknown" and not result["progress_token"]


def test_bounded_worker_timeout_and_oversized_response_are_unknown(catalogue, monkeypatch):
    def timeout(*args, **kwargs): raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(probe.subprocess, "run", timeout)
    result = observe(catalogue)
    assert result["reason_codes"] == ["native_catalogue_read_timeout"]
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs:
        subprocess.CompletedProcess(args[0], 0, stdout=b"x" * (probe.MAX_REPORT_BYTES + 1), stderr=b""))
    assert observe(catalogue)["health"] == "unknown"


def test_stale_worker_output_cannot_become_live_health(catalogue, monkeypatch):
    data = probe._read_catalogue({"database_path": str(catalogue[0]), "population_path": str(catalogue[1]),
        "board_namespace": NAMESPACE, "plan_root_cid": ROOT, "repository_tree_id": TREE,
        "population_sha256": hashlib.sha256(catalogue[1].read_bytes()).hexdigest()})
    data["read_finished_at_epoch"] = 1.0
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs:
        subprocess.CompletedProcess(args[0], 0, stdout=json.dumps(data).encode(), stderr=b""))
    assert observe(catalogue)["reason_codes"] == ["native_catalogue_native_read_stale"]


def test_cli_preserves_binding_and_false_authority(catalogue, capsys):
    database, population = catalogue
    assert probe.main(["--board", "legal-ir-scientific", "--database", str(database),
        "--population", str(population), "--namespace", NAMESPACE, "--plan-root-cid", ROOT,
        "--repository-tree-id", TREE, "--population-sha256", hashlib.sha256(population.read_bytes()).hexdigest()]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["health"] == "blocked" and result["complete"] is False


def test_administrative_unknown_receipt_does_not_claim_verified_provider_outcome(catalogue):
    change_database(catalogue, "UPDATE tasks SET body_json=? WHERE task_alias='LIR-9001'",
        [json.dumps({"board_namespace": NAMESPACE, "completion_receipt": {
            "attempt_number": 2, "automatic_retry_admitted": False,
            "failure_kind": "administrative_operator_stop_outcome_unknown"}})])
    result = observe(catalogue)
    diagnostic = result["details"]["native_catalogue"]
    assert diagnostic["tasks"][0]["completion_receipt_attempt_number"] == 2
    assert diagnostic["external_outcome_knowledge"] == "unknown_observed"
    assert diagnostic["provider_outcome_knowledge"] == "not_established"
    assert "native_catalogue_outcome_unknown" in result["reason_codes"]
    assert "native_catalogue_provider_outcome_unknown" not in result["reason_codes"]


def test_fresh_error_clock_is_not_fresh_catalogue_facts(catalogue):
    import duckdb
    import time

    before = time.time()
    writer = duckdb.connect(str(catalogue[0]))
    try:
        result = observe(catalogue)
    finally:
        writer.close()
    diagnostic = result["details"]["native_catalogue"]
    assert result["health"] == "unknown"
    assert before <= diagnostic["probe_observed_at_epoch"] <= time.time()
    assert diagnostic["checked_at_epoch"] is None and diagnostic["fresh_read"] is False
    assert "counts" not in diagnostic


def test_actual_catalogue_report_retains_wait_across_watchdog_and_repair_boundaries(catalogue, tmp_path, monkeypatch):
    import time
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_watchdog as fleet
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as repair

    observation = observe(catalogue)
    digest = hashlib.sha256(catalogue[1].read_bytes()).hexdigest()
    board = {"id": "legal-ir-scientific", "cwd": str(tmp_path), "hold_files": [],
        "probe_profile": "native_catalogue", "probe": {"argv": ["probe"]},
        "max_repair_attempts": 3, "failure_grace_seconds": 0,
        "native_catalogue_binding": {"board_namespace": NAMESPACE, "plan_root_cid": ROOT,
            "repository_tree_id": TREE, "population_sha256": digest, "max_age_seconds": 180},
        "recovery_policy": {"profile": "infrastructure-only/v1",
            "repairable_stalls": ["native_catalogue_read_unavailable"],
            "max_actions_per_window": 3, "window_seconds": 1000, "circuit_breaker_seconds": 100}}
    def forbidden(*args, **kwargs): pytest.fail("no native recovery command authorized")
    monkeypatch.setattr(fleet, "command", forbidden)
    monkeypatch.setattr(repair, "command", forbidden)
    state = fleet.assess(observation, {}, board, time.time())
    assert state["health"] == "blocked"
    assert fleet.select_action(state, board, time.time()) == ""
    assert state["observation"]["details"]["native_catalogue"]["counts"] == {"blocked": 2}
    incident = tmp_path / "incident.json"
    incident.write_text(json.dumps({"board_id": board["id"], "observation": observation}))
    config = {"state_dir": str(tmp_path / "watchdog-state"), "boards": [board],
              "repair_worker": {"cwd": str(tmp_path)}}
    assert repair.enqueue(config, incident)["status"] == "policy_wait"
    assert not Path(config["state_dir"]).exists()
    verified = repair.verify_job_recovery(board, {}, observation, tmp_path)
    assert verified["observer_verified"] is True and verified["wait"] is True
    assert verified["task_health"] == "blocked"
    assert all(verified[key] is False for key in ("native_admission", "completion_authority",
        "task_completion_authority", "goal_completion_authority", "complete"))


def test_child_goal_inherited_objective_keeps_exact_native_binding(catalogue):
    population = json.loads(catalogue[1].read_text())
    population["goals"].append({"goal_cid": "baguqeerachildgoal", "goal_alias": "LIR-G091",
        "objective_id": "", "parent_goal_cid": "baguqeeragoal"})
    for task in population["tasks"]:
        task["goal_cid"] = "baguqeerachildgoal"
    catalogue[1].write_text(json.dumps(population))
    change_database(catalogue, "INSERT INTO goals VALUES ('baguqeerachildgoal','LIR-G091','','baguqeeragoal','open',2)")
    change_database(catalogue, "UPDATE tasks SET goal_cid='baguqeerachildgoal'")
    result = observe(catalogue)
    assert result["health"] == "blocked"
    diagnostic = result["details"]["native_catalogue"]
    assert diagnostic["fresh_read"] is True and diagnostic["goal_count"] == 2
    assert all(task["goal_cid"] == "baguqeerachildgoal" for task in diagnostic["tasks"])
    change_database(catalogue, "UPDATE goals SET objective_id='other-objective' WHERE goal_cid='baguqeerachildgoal'")
    assert observe(catalogue)["health"] == "unknown"


@pytest.mark.parametrize("invalid", [None, "invalid objective!"])
def test_optional_goal_objective_rejects_nonstring_or_invalid_nonempty_identifier(catalogue, monkeypatch, invalid):
    population = json.loads(catalogue[1].read_text())
    population["goals"][0]["objective_id"] = invalid
    catalogue[1].write_text(json.dumps(population))
    def forbidden(*args, **kwargs): pytest.fail("invalid goal objective started worker")
    monkeypatch.setattr(probe.subprocess, "run", forbidden)
    result = observe(catalogue)
    assert result["health"] == "unknown" and not result["progress_token"]
    assert result["reason_codes"] == ["native_catalogue_invalid_binding"]
