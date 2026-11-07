"""Direct catalogue observations retain wait/authority and read-only boundaries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import copy
import runpy
import stat

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


@pytest.fixture
def population_special_file_cleanup(tmp_path):
    """Close only the exact private negative object created by this test."""
    registered = []
    def identity(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink,
                info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    def register(path):
        assert path == tmp_path / "population.json" and not registered
        info = path.lstat()
        assert info.st_uid == os.geteuid() and info.st_nlink == 1
        assert stat.S_ISFIFO(info.st_mode) or stat.S_ISLNK(info.st_mode)
        target = os.readlink(path) if stat.S_ISLNK(info.st_mode) else None
        registered.append((path, identity(info), target))
    try:
        yield register
    finally:
        for path, original, target in registered:
            assert identity(path.lstat()) == original
            if target is not None:
                assert os.readlink(path) == target
            path.unlink()


@pytest.mark.parametrize("kind", ["missing", "symlink", "fifo", "oversized", "bad_digest", "duplicate_json_key"])
def test_population_file_boundary_refuses_without_worker(catalogue, monkeypatch, kind, population_special_file_cleanup):
    population = catalogue[1]
    changes = {}
    if kind == "missing": population.unlink()
    elif kind == "symlink":
        target = population.with_name("target.json")
        population.rename(target)
        population.symlink_to(target)
        population_special_file_cleanup(population)
    elif kind == "fifo":
        population.unlink()
        os.mkfifo(population)
        population_special_file_cleanup(population)
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


# A current independently verified task completion must not hide a prior UNKNOWN.
def _complete_with_retained_outcome(catalogue, *, mutate=None, exhausted=False):
    receipt = {"task_cid": "baguqeerataska", "attempt_number": 2,
               "automatic_retry_admitted": False,
               "failure_kind": "administrative_operator_stop_outcome_unknown",
               "provider_invocation_count": 1}
    if exhausted:
        receipt.update(max_task_attempts=2, attempt_budget_exhausted=True)
    entry = {"task_cid": "baguqeerataska", "task_revision": 13,
             "completion_receipt": receipt,
             "sha256": hashlib.sha256(json.dumps(receipt, sort_keys=True,
                 separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()}
    if mutate is not None:
        mutate(entry)
    body = {"board_namespace": NAMESPACE,
            "completion_receipt": {"operation": "independent_metadata_acceptance"},
            "retained_execution_outcomes": [entry]}
    change_database(catalogue,
        "UPDATE tasks SET body_json=?, status='completed', revision=15 WHERE task_alias='LIR-9001'",
        [json.dumps(body)])


def test_completed_factual_work_retains_prior_unknown_without_retry_authority(catalogue):
    _complete_with_retained_outcome(catalogue)
    result = observe(catalogue)
    diagnostic = result["details"]["native_catalogue"]
    task = diagnostic["tasks"][0]
    assert task["status"] == "completed" and task["revision"] == 15
    assert task["receipt_failure_kind"] is None
    assert task["retained_execution_outcomes"][0]["attempt_number"] == 2
    assert task["retained_execution_outcomes"][0]["max_task_attempts"] is None
    assert diagnostic["retained_unknown_outcome_task_cids"] == ["baguqeerataska"]
    assert diagnostic["retry_not_admitted_task_cids"] == ["baguqeerataska"]
    assert diagnostic["provider_outcome_knowledge"] == "unknown_observed"
    assert diagnostic["external_outcome_knowledge"] == "unknown_observed"
    assert diagnostic["historical_outcomes_are_separate_from_current_completion"] is True
    assert result["complete"] is result["completion_candidate"] is False
    assert diagnostic["completion_authority"] is False
    assert "recovery_action" not in result


def _bound_scientific_board(catalogue, tmp_path):
    return {"id": "legal-ir-scientific", "cwd": str(tmp_path), "hold_files": [],
        "probe_profile": "native_catalogue", "probe": {"argv": ["probe"]},
        "max_repair_attempts": 3, "failure_grace_seconds": 0,
        "native_catalogue_binding": {"board_namespace": NAMESPACE, "plan_root_cid": ROOT,
            "repository_tree_id": TREE,
            "population_sha256": hashlib.sha256(catalogue[1].read_bytes()).hexdigest(),
            "max_age_seconds": 180},
        "recovery_policy": {"profile": "infrastructure-only/v1",
            "repairable_stalls": ["native_catalogue_read_unavailable"],
            "max_actions_per_window": 3, "window_seconds": 1000, "circuit_breaker_seconds": 100}}


@pytest.mark.parametrize("peer_status", ["blocked", "todo"])
def test_completed_history_exposes_current_activation_wait_and_keeps_recovery_budget(catalogue, tmp_path, monkeypatch, peer_status):
    import time
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_watchdog as fleet

    _complete_with_retained_outcome(catalogue, exhausted=True)
    change_database(catalogue, "UPDATE tasks SET status=? WHERE task_alias='LIR-9002'", [peer_status])
    change_database(catalogue, "INSERT INTO provider_invocations VALUES ('baguqeerataska','old-attempt-a','unknown')")
    result = observe(catalogue)
    diagnostic = result["details"]["native_catalogue"]
    assert diagnostic["provider_outcome_knowledge"] == diagnostic["external_outcome_knowledge"] == "unknown_observed"
    assert diagnostic["exhausted_receipt_task_cids"] == diagnostic["retry_not_admitted_task_cids"] == ["baguqeerataska"]
    assert diagnostic["tasks"][0]["retained_execution_outcomes"][0]["attempt_budget_exhausted"] is True
    assert all(not cids for cids in diagnostic["current_nonterminal_blockers"].values())
    assert diagnostic["ready_status_count"] == (1 if peer_status == "todo" else 0)
    assert diagnostic["normalized_provider_status_rows"] == [["baguqeerataska", "old-attempt-a", "unknown"]]
    assert set(result["reason_codes"]) == {"native_catalogue_activation_missing", "native_catalogue_executor_unverified"}
    board = _bound_scientific_board(catalogue, tmp_path)
    previous = {"attempts": 2, "ensure_attempts": 1,
                "infrastructure_budget": {"status": "open", "total_actions": 3}}
    monkeypatch.setattr(fleet, "command", lambda *a, **k: pytest.fail("history cannot authorize recovery"))
    now = time.time()
    state = fleet.assess(result, previous, board, now)
    assert state["health"] == "blocked" and state["stall_class"] == "native_catalogue_activation_missing"
    assert fleet.select_action(state, board, now) == ""
    for field in ("attempts", "ensure_attempts", "infrastructure_budget"):
        assert state[field] == previous[field]
    assert diagnostic["activation_state"] == "not_activated" and diagnostic["executor_health"] == "not_established"
    assert diagnostic["completion_authority"] is result["complete"] is result["completion_candidate"] is False
    assert "recovery_action" not in result


@pytest.mark.parametrize("source", ["current_receipt", "normalized_provider_row"])
def test_completed_history_does_not_mask_an_unresolved_current_peer(catalogue, tmp_path, source):
    import time
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_watchdog as fleet

    _complete_with_retained_outcome(catalogue)
    if source == "current_receipt":
        change_database(catalogue, "UPDATE tasks SET body_json=? WHERE task_alias='LIR-9002'",
            [json.dumps({"board_namespace": NAMESPACE, "completion_receipt": {
                "attempt_number": 1, "max_task_attempts": 1,
                "automatic_retry_admitted": False, "attempt_budget_exhausted": True,
                "failure_kind": "provider_outcome_unknown", "provider_invocation_count": 1}})])
    else:
        change_database(catalogue, "INSERT INTO provider_invocations VALUES ('baguqeerataskb','attempt-b','unknown')")
    result = observe(catalogue)
    diagnostic = result["details"]["native_catalogue"]
    current = diagnostic["current_nonterminal_blockers"]
    assert current["provider_outcome_unknown_task_cids"] == ["baguqeerataskb"]
    assert diagnostic["retained_unknown_outcome_task_cids"] == ["baguqeerataska"]
    assert diagnostic["tasks"][0]["retained_execution_outcomes"][0]["automatic_retry_admitted"] is False
    if source == "current_receipt":
        assert current["retry_not_admitted_task_cids"] == current["exhausted_receipt_task_cids"] == ["baguqeerataskb"]
        assert set(diagnostic["retry_not_admitted_task_cids"]) == {"baguqeerataska", "baguqeerataskb"}
    board = _bound_scientific_board(catalogue, tmp_path)
    now = time.time()
    state = fleet.assess(result, {}, board, now)
    assert state["stall_class"] == "native_catalogue_provider_outcome_unknown"
    assert fleet.select_action(state, board, now) == ""
    assert diagnostic["completion_authority"] is result["complete"] is result["completion_candidate"] is False


@pytest.mark.parametrize("status", ["blocked", "todo", "cancelled", "failed", "complete", "done"])
def test_only_exact_native_completion_separates_retained_unknown_from_current_wait(catalogue, status):
    _complete_with_retained_outcome(catalogue, exhausted=True)
    change_database(catalogue, "UPDATE tasks SET status=? WHERE task_alias='LIR-9001'", [status])
    result = observe(catalogue)
    diagnostic = result["details"]["native_catalogue"]
    assert all(cids == ["baguqeerataska"] for cids in diagnostic["current_nonterminal_blockers"].values())
    assert {"native_catalogue_provider_outcome_unknown", "native_catalogue_outcome_unknown",
            "native_catalogue_retry_not_admitted", "native_catalogue_attempt_budget_exhausted"} <= set(result["reason_codes"])
    assert result["complete"] is diagnostic["completion_authority"] is False


@pytest.mark.parametrize("damage", ["digest", "wrong_task", "future_revision", "boolean_revision", "receipt_identity", "malformed"])
def test_invalid_retained_outcome_is_unknown_instead_of_silently_discarded(catalogue, damage):
    def mutate(entry):
        if damage == "digest": entry["sha256"] = "0" * 64
        elif damage == "wrong_task": entry["task_cid"] = "baguqeeraother"
        elif damage == "future_revision": entry["task_revision"] = 15
        elif damage == "boolean_revision": entry["task_revision"] = True
        elif damage == "malformed": entry["extra"] = "unexpected"
        else:
            entry["completion_receipt"]["task_cid"] = "baguqeeraother"
            entry["sha256"] = hashlib.sha256(json.dumps(entry["completion_receipt"],
                sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    _complete_with_retained_outcome(catalogue, mutate=mutate)
    result = observe(catalogue)
    assert result["health"] == "unknown"
    assert "native_catalogue_native_retained_outcome_invalid" in result["reason_codes"]
    assert result["details"]["native_catalogue"]["fresh_read"] is False


def test_current_administrative_unknown_retains_provider_invocation_knowledge(catalogue):
    receipt = {"attempt_number": 2, "automatic_retry_admitted": False,
               "failure_kind": "administrative_operator_stop_outcome_unknown",
               "provider_invocation_count": 1}
    change_database(catalogue, "UPDATE tasks SET body_json=? WHERE task_alias='LIR-9001'",
                    [json.dumps({"board_namespace": NAMESPACE, "completion_receipt": receipt})])
    result = observe(catalogue)
    assert result["details"]["native_catalogue"]["provider_outcome_knowledge"] == "unknown_observed"
    assert result["details"]["native_catalogue"]["external_outcome_knowledge"] == "unknown_observed"


def test_absent_executor_profile_never_reads_health_evidence(catalogue, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import native_executor_health as health
    monkeypatch.setattr(health, "verify_bound_executor", lambda *a, **k: pytest.fail("legacy observer read executor evidence"))
    report = observe(catalogue)
    native = report["details"]["native_catalogue"]
    assert native["activation_state"] == "not_activated"
    assert native["executor_health"] == "not_established"
    assert "executor_verification" not in native
    assert "native_catalogue_activation_missing" in report["reason_codes"]


@pytest.mark.parametrize("verified", [False, True])
def test_opt_in_executor_join_preserves_native_unknown_and_task_counts(catalogue, monkeypatch, verified):
    from ipfs_accelerate_py.agent_supervisor.rescue import native_executor_health as health
    profile = {"path": "/private/operator/executor.json", "bytes": 10, "sha256": "a" * 64}
    seen = []
    def independent(ref, native, *, now):
        assert native["fresh_read"] and native["database_and_wal_metadata_unchanged"]
        seen.append((ref, native["counts"], now))
        return {"verified": verified, "scope": "genuine_scoped_idle", "reason": "fixture_join"}
    monkeypatch.setattr(health, "verify_bound_executor", independent)
    report = observe(catalogue, executor_profile_ref=profile)
    native = report["details"]["native_catalogue"]
    assert len(seen) == 1 and seen[0][0] == profile and seen[0][1] == {"blocked": 2}
    assert native["executor_verification"]["verified"] is verified
    assert native["activation_state"] == ("active" if verified else "not_activated")
    assert native["executor_health"] == ("healthy" if verified else "not_established")
    assert ("native_catalogue_activation_missing" in report["reason_codes"]) is not verified
    assert "native_catalogue_provider_outcome_unknown" in report["reason_codes"]
    assert "native_catalogue_attempt_budget_exhausted" in report["reason_codes"]
    assert report["complete"] is report["completion_candidate"] is report["busy"] is False
    assert report["health"] == "blocked"
    assert native["counts"] == {"blocked": 2} and native["completion_authority"] is False


@pytest.mark.parametrize("selected", [["--executor-profile", "/missing/profile"],
    ["--executor-profile-bytes", "1"], ["--executor-profile-sha256", "a" * 64]])
def test_executor_cli_requires_complete_descriptor_before_native_read(monkeypatch, selected):
    monkeypatch.setattr(probe, "observe_native_catalogue", lambda *a, **k: pytest.fail("partial health binding reached native read"))
    with pytest.raises(SystemExit) as error:
        probe.main(["--board", "legal-ir-scientific", "--database", "/missing/db", "--population", "/missing/pop",
            "--namespace", NAMESPACE, "--plan-root-cid", ROOT, "--repository-tree-id", TREE,
            "--population-sha256", "a" * 64, *selected])
    assert error.value.code == 2


@pytest.fixture
def authored_dependency_graph(tmp_path, monkeypatch):
    """Synthetic authored64/34/115 input; no retired boundary schema emitted."""
    goals = [{"goal_cid": f"baguqeeragoal{i}", "goal_alias": f"LIR-G{i:03}",
              "objective_id": "objective:one", "parent_goal_cid": ""} for i in range(34)]
    tasks = [{"task_cid": f"baguqeeratask{i}", "task_alias": f"LIR-{9000+i}",
              "goal_cid": goals[min(i, 33)]["goal_cid"], "plan_cid": ROOT,
              "objective_id": "objective:one", "dependencies": []} for i in range(64)]
    for i in range(1, 64): tasks[i]["dependencies"].append(tasks[0]["task_cid"])
    for i in range(2, 54): tasks[i]["dependencies"].append(tasks[1]["task_cid"])
    authored = {"schema": probe.POPULATION_SCHEMA, "board_namespace": NAMESPACE,
                "plan_root_cid": ROOT, "repository_tree_id": TREE, "tasks": tasks, "goals": goals,
                "plans": [{"plan_cid": ROOT, "goal_cid": goals[0]["goal_cid"]}]}
    def store(name, value):
        path = tmp_path / name; raw = json.dumps(value, sort_keys=True).encode()
        path.write_bytes(raw)
        return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    authored_ref = store("authored.json", authored)
    monkeypatch.setattr(probe, "AUTHORED_BOUNDARY_POPULATION_SHA256", authored_ref["sha256"])
    def envelope(document):
        boundary_ref = store("boundary.json", document)
        return store("envelope.json", {"schema": probe.BOUNDARY_POPULATION_SCHEMA,
                     "authored_population_ref": authored_ref, "dependency_boundary_ref": boundary_ref})
    return {"authored": authored, "authored_ref": authored_ref, "envelope": envelope,
            "edges": [(task["task_cid"], dep, "depends_on") for task in tasks for dep in task["dependencies"]]}


@pytest.fixture
def observational_graph(authored_dependency_graph):
    """Synthetic closed graph: eight roots, two grandchildren, one foreign prerequisite."""
    base = authored_dependency_graph
    nodes, roots = [], []
    for i in range(8):
        parent = base["authored"]["tasks"][62 if i < 5 else 63]
        cid, alias = parent["task_cid"] + f"--obligation-{i}", parent["task_alias"] + f"--obligation-{i}"
        projection = {"completion": "manual", "resource_stage": "analysis",
            "parent_task_cid": parent["task_cid"], "parent_task_alias": parent["task_alias"],
            "resolution_subgoal": True, "obligation": f"Store the independently recorded fact {i}",
            "obligation_kind": "recorded_fact", "spawn_decision": "missing_evidence",
            "execution_packet_required": True, "admitted": False, "formalized": False,
            "gold_or_teacher_admission_granted": False, "constitution_roundtrip_ok": False,
            "completion_authority": False}
        nodes.append({"task_cid": cid, "task_alias": alias,
            **{key: parent[key] for key in ("goal_cid", "plan_cid", "objective_id")},
            "minimum_revision": 1, "namespace_binding": {"present": False, "value": None},
            "observation_role": "root_dependency",
            "body_binding": {"profile": "resolution-obligation/v1", "projection": projection},
            "dependencies": [{"dependency_task_cid": dep, "kind": "depends_on"} for dep in parent["dependencies"]]})
        roots.append({"task_cid": parent["task_cid"], "dependency_task_cid": cid, "kind": "depends_on"})
    foreign = {"task_cid": "baguqeeraforeignprerequisite", "task_alias": "OTHER-42",
        "goal_cid": "baguqeeraforeigngoal", "plan_cid": "baguqeeraforeignplan", "objective_id": None,
        "minimum_revision": 2, "namespace_binding": {"present": True, "value": "separate-board"},
        "observation_role": "transitive_dependency", "body_binding": {"profile": "literal-prerequisite/v1"},
        "dependencies": [{"dependency_task_cid": "baguqeeratask0", "kind": "reviewed_prerequisite"}]}
    parent = nodes[0]
    for i in range(2):
        child = copy.deepcopy(parent)
        child["task_cid"] += f"--nested-{i}"; child["task_alias"] += f"--nested-{i}"
        child["observation_role"] = "transitive_dependency"
        child["body_binding"]["projection"].update(parent_task_cid=parent["task_cid"], parent_task_alias=parent["task_alias"])
        child["dependencies"] = [{"dependency_task_cid": foreign["task_cid"], "kind": "depends_on"}]
        parent["dependencies"].append({"dependency_task_cid": child["task_cid"], "kind": "depends_on"})
        nodes.append(child)
    nodes.append(foreign)
    document = {"schema": probe.GRAPH_DEPENDENCY_BOUNDARY_SCHEMA, "board_namespace": NAMESPACE,
        "plan_root_cid": ROOT, "repository_tree_id": TREE,
        "authored_population_sha256": base["authored_ref"]["sha256"], "authored_task_count": 64,
        "authored_goal_count": 34, "authored_dependency_count": 115,
        "root_dependencies": roots, "observed_tasks": nodes, "completion_authority": False}
    base["edges"] += [(row["task_cid"], row["dependency_task_cid"], row["kind"]) for row in roots]
    ref = base["envelope"](document)
    return {"base": base, "document": document, "foreign": foreign,
            "arguments": {"population_path": ref["path"], "population_sha256": ref["sha256"],
                          "board_namespace": NAMESPACE, "plan_root_cid": ROOT, "repository_tree_id": TREE}}


def graph_population(case, document=None):
    if document is None:
        return probe._population(case["arguments"])
    ref = case["base"]["envelope"](document)
    return probe._population(dict(case["arguments"], population_path=ref["path"], population_sha256=ref["sha256"]))


def graph_rows(document):
    rows = []
    for node in document["observed_tasks"]:
        binding = node["body_binding"]
        body = (copy.deepcopy(binding["projection"]) if binding["profile"] == "resolution-obligation/v1"
                else {"arbitrary_support_semantics": "not qualified", "admitted": True})
        if node["namespace_binding"]["present"]:
            body["board_namespace"] = node["namespace_binding"]["value"]
        rows.append([*(node[key] for key in ("task_cid", "task_alias", "goal_cid", "plan_cid", "objective_id")),
                     "blocked", node["minimum_revision"], json.dumps(body)])
    return rows


def graph_observation(case, *, document=None, rows=None, primary_edges=None, observed_edges=None):
    document = case["document"] if document is None else document
    selected, _, plan = graph_population(case, document)
    edges = [(cid, dep, "depends_on") for cid, task in selected.items() for dep in task["dependencies"]]
    edges += [(r["task_cid"], r["dependency_task_cid"], r["kind"]) for r in document["root_dependencies"]]
    noncore = [(n["task_cid"], r["dependency_task_cid"], r["kind"])
               for n in document["observed_tasks"] for r in n["dependencies"]]
    return probe._validate_graph_observation(plan, selected,
        edges if primary_edges is None else primary_edges,
        graph_rows(document) if rows is None else rows,
        noncore if observed_edges is None else observed_edges)


def test_closed_graph_keeps_grandchildren_and_foreign_prerequisite_outside_primary_authority(observational_graph):
    case = observational_graph
    selected, goals, plan = graph_population(case)
    assert isinstance(plan, probe._GraphBoundaryPlan) and (len(selected), len(goals)) == (64, 34)
    result = graph_observation(case)
    assert result["authored_dependency_count"] == 115 and result["boundary_dependency_count"] == 8
    assert result["native_outgoing_dependency_count"] == 123
    assert (result["node_count"], result["edge_count"], result["observed_task_count"],
            result["observed_dependency_count"], result["maximum_outward_depth"]) == (75, 136, 11, 13, 3)
    support = next(n for n in result["observed_tasks"] if n["task_cid"] == case["foreign"]["task_cid"])
    assert support["goal_cid"] == "baguqeeraforeigngoal" and support["objective_id"] is None
    assert support["namespace_binding"] == {"present": True, "value": "separate-board"}
    assert result["completion_authority"] is result["dispatch_authority"] is False
    assert result["observed_task_completion_established"] is result["prerequisite_body_semantics_qualified"] is False
    unsupported = copy.deepcopy(case["document"])
    unsupported["schema"] = "legal-ir-native-dependency-boundary/v1"
    with pytest.raises(probe.InvalidCatalogue, match="dependency_boundary_schema_unsupported"):
        graph_population(case, unsupported)


def test_new_ninth_root_requires_an_explicit_closed_certificate_without_a_source_change(observational_graph):
    case = observational_graph
    selected, _, old_plan = graph_population(case)
    document = copy.deepcopy(case["document"])
    extra = copy.deepcopy(document["observed_tasks"][-1])
    extra.update(task_cid="baguqeeraninthroot", task_alias="OTHER-43", observation_role="root_dependency")
    document["observed_tasks"].append(extra)
    edge = ("baguqeeratask63", extra["task_cid"], "depends_on")
    document["root_dependencies"].append(dict(zip(("task_cid", "dependency_task_cid", "kind"), edge)))
    primary = case["base"]["edges"] + [edge]
    with pytest.raises(probe.InvalidCatalogue, match="native_graph_dependencies_changed"):
        probe._validate_graph_observation(old_plan, selected, primary, [], [])
    result = graph_observation(case, document=document)
    assert result["boundary_dependency_count"] == 9 and result["observed_task_count"] == 12
    assert result["authored_dependency_count"] == 115 and result["native_outgoing_dependency_count"] == 124
    assert result["completion_authority"] is False


@pytest.mark.parametrize("damage", ["missing", "unreachable", "role", "cycle", "target", "duplicate_node",
                                    "duplicate_edge", "core_edge", "core_count", "authority", "semantic_profile"])
def test_closed_graph_declaration_refuses_scope_and_topology_damage(observational_graph, damage):
    case = observational_graph
    document = copy.deepcopy(case["document"])
    nodes = document["observed_tasks"]
    if damage == "missing": nodes.pop()
    elif damage == "unreachable":
        extra = copy.deepcopy(nodes[-1]); extra.update(task_cid="baguqeeraunreachable", task_alias="UNREACHABLE")
        nodes.append(extra)
    elif damage == "role": nodes[0]["observation_role"] = "transitive_dependency"
    elif damage == "cycle": nodes[-1]["dependencies"] = [{"dependency_task_cid": nodes[0]["task_cid"], "kind": "depends_on"}]
    elif damage == "target": nodes[-1]["dependencies"][0]["dependency_task_cid"] = "baguqeeraunknown"
    elif damage == "duplicate_node": nodes.append(copy.deepcopy(nodes[-1]))
    elif damage == "duplicate_edge": document["root_dependencies"].append(copy.deepcopy(document["root_dependencies"][0]))
    elif damage == "core_edge": document["root_dependencies"][0]["dependency_task_cid"] = "baguqeeratask0"
    elif damage == "core_count": document["authored_dependency_count"] = 114
    elif damage == "authority": document["completion_authority"] = True
    else: nodes[0]["body_binding"]["projection"]["admitted"] = True
    with pytest.raises(probe.InvalidCatalogue):
        graph_population(case, document)


@pytest.mark.parametrize("damage", ["foreign_goal", "namespace", "revision", "obligation", "new_optional_semantics",
                                    "missing_node", "missing_edge", "edge_kind"])
def test_closed_graph_actual_rows_cannot_hide_identity_semantic_or_edge_drift(observational_graph, damage):
    case = observational_graph
    rows = graph_rows(case["document"])
    noncore = [(n["task_cid"], r["dependency_task_cid"], r["kind"])
               for n in case["document"]["observed_tasks"] for r in n["dependencies"]]
    if damage == "foreign_goal": rows[-1][2] = "baguqeeradifferentgoal"
    elif damage == "namespace":
        body = json.loads(rows[-1][7]); body["board_namespace"] = NAMESPACE; rows[-1][7] = json.dumps(body)
    elif damage == "revision": rows[-1][6] -= 1
    elif damage in {"obligation", "new_optional_semantics"}:
        body = json.loads(rows[0][7])
        body["obligation" if damage == "obligation" else "missing_record_subgoal"] = "changed" if damage == "obligation" else False
        rows[0][7] = json.dumps(body)
    elif damage == "missing_node": rows.pop()
    elif damage == "missing_edge": noncore.pop()
    else: noncore[-1] = (*noncore[-1][:2], "different_kind")
    with pytest.raises(probe.InvalidCatalogue):
        graph_observation(case, rows=rows, observed_edges=noncore)


@pytest.mark.parametrize("literal", [None, True, 1, 1.0, {"foreign": [True, 1, 1.0]}, NAMESPACE])
def test_foreign_namespace_is_a_typed_literal_without_inheritance(observational_graph, literal):
    case = observational_graph
    document = copy.deepcopy(case["document"])
    document["observed_tasks"][-1]["namespace_binding"]["value"] = literal
    result = graph_observation(case, document=document)
    support = next(n for n in result["observed_tasks"] if n["task_cid"] == case["foreign"]["task_cid"])
    assert type(support["namespace_binding"]["value"]) is type(literal)
    assert support["namespace_binding"]["value"] == literal
    rows = graph_rows(document)
    body = json.loads(rows[-1][7])
    body["board_namespace"] = 1 if literal is True else True
    rows[-1][7] = json.dumps(body)
    with pytest.raises(probe.InvalidCatalogue, match="native_graph_namespace_changed"):
        graph_observation(case, document=document, rows=rows)


def test_graph_progress_does_not_rebind_semantics_or_establish_parent_completion(observational_graph):
    case = observational_graph
    rows = graph_rows(case["document"])
    rows[0][5:7] = ["completed", 5]
    body = json.loads(rows[0][7]); body.update(resolution_decomposed=True, completion_receipt={"operation": "new fact"})
    rows[0][7] = json.dumps(body)
    rows[-1][5:7] = ["skipped", 7]
    body = json.loads(rows[-1][7]); body["arbitrary_support_semantics"] = "not qualified by this observer"
    rows[-1][7] = json.dumps(body)
    result = graph_observation(case, rows=rows)
    assert result["observed_status_counts"] == {"completed": 1, "blocked": 9, "skipped": 1}
    assert result["observed_task_completion_established"] is result["completion_authority"] is False


def test_resolution_marker_presence_cannot_downgrade_to_identity_only_prerequisite(observational_graph):
    case = observational_graph
    rows = graph_rows(case["document"])
    body = json.loads(rows[-1][7]); body["resolution_work"] = False
    rows[-1][7] = json.dumps(body)
    with pytest.raises(probe.InvalidCatalogue, match="native_graph_resolution_profile_required"):
        graph_observation(case, rows=rows)


@pytest.mark.parametrize("damage", ["orphan_parent", "parent_alias", "goal", "alias_prefix", "cid_prefix", "origin_edge_kind"])
def test_resolution_profile_joins_declared_native_parent_and_constructor_lineage(observational_graph, damage):
    case = observational_graph
    document = copy.deepcopy(case["document"])
    node = document["observed_tasks"][0]
    projection = node["body_binding"]["projection"]
    if damage == "orphan_parent": projection["parent_task_cid"] = "baguqeeraunobservedparent"
    elif damage == "parent_alias": projection["parent_task_alias"] = "WRONG-PARENT"
    elif damage == "goal": node["goal_cid"] = "baguqeeraforeigngoal"
    elif damage == "alias_prefix": node["task_alias"] = "NO-CONSTRUCTOR-PREFIX"
    elif damage == "origin_edge_kind": document["root_dependencies"][0]["kind"] = "related_only"
    else:
        old = node["task_cid"]; node["task_cid"] = "baguqeeranonsuffixchild"
        document["root_dependencies"][0]["dependency_task_cid"] = node["task_cid"]
        for child in document["observed_tasks"]:
            for row in child["dependencies"]:
                if row["dependency_task_cid"] == old: row["dependency_task_cid"] = node["task_cid"]
    with pytest.raises(probe.InvalidCatalogue, match="graph_resolution_lineage_invalid"):
        graph_population(case, document)


def _literal_graph_node(index):
    return {"task_cid": f"baguqeeraliteral{index}", "task_alias": f"OTHER-{index}",
            "goal_cid": "baguqeeraforeigngoal", "plan_cid": None, "objective_id": "",
            "minimum_revision": 1, "namespace_binding": {"present": False, "value": None},
            "observation_role": "transitive_dependency", "body_binding": {"profile": "literal-prerequisite/v1"},
            "dependencies": []}


def test_outward_discovery_depth_is_bounded_separately_from_whole_graph_acyclicity(observational_graph):
    case = observational_graph
    document = copy.deepcopy(case["document"])
    previous = document["observed_tasks"][0]
    for i in range(15):
        node = _literal_graph_node(i)
        previous["dependencies"].append({"dependency_task_cid": node["task_cid"], "kind": "depends_on"})
        document["observed_tasks"].append(node); previous = node
    _, _, plan = graph_population(case, document)
    assert plan.graph_topology["maximum_outward_depth"] == 16
    node = _literal_graph_node(15)
    previous["dependencies"].append({"dependency_task_cid": node["task_cid"], "kind": "depends_on"})
    document["observed_tasks"].append(node)
    with pytest.raises(probe.InvalidCatalogue, match="graph_depth_bound"):
        graph_population(case, document)


def test_iterative_graph_validation_handles_the_total_node_bound_without_recursion(observational_graph):
    case = observational_graph
    document = copy.deepcopy(case["document"])
    previous = None
    for i in range(925):
        node = _literal_graph_node(i)
        node["observation_role"] = "root_dependency"
        document["root_dependencies"].append({"task_cid": "baguqeeratask63",
            "dependency_task_cid": node["task_cid"], "kind": "depends_on"})
        if previous is not None:
            previous["dependencies"].append({"dependency_task_cid": node["task_cid"], "kind": "depends_on"})
        document["observed_tasks"].append(node); previous = node
    _, _, plan = graph_population(case, document)
    assert plan.graph_topology["node_count"] == 1000
    document["observed_tasks"].append(_literal_graph_node(925))
    with pytest.raises(probe.InvalidCatalogue, match="graph_bounds"):
        graph_population(case, document)


def test_graph_edge_budget_and_actual_read_budget_refuse_without_partial_observation(observational_graph, monkeypatch):
    case = observational_graph
    selected, _, plan = graph_population(case)
    monkeypatch.setattr(probe, "MAX_GRAPH_EDGES", 136)
    assert graph_observation(case)["edge_count"] == 136
    document = copy.deepcopy(case["document"])
    document["observed_tasks"][0]["dependencies"].append({"dependency_task_cid": "baguqeeratask1", "kind": "depends_on"})
    with pytest.raises(probe.InvalidCatalogue, match="graph_edge_bound"):
        graph_population(case, document)
    monkeypatch.setattr(probe, "MAX_GRAPH_BOUNDARY_BYTES", 128)
    with pytest.raises(probe.InvalidCatalogue, match="native_graph_read_bound"):
        probe._validate_graph_observation(plan, selected, case["base"]["edges"], graph_rows(case["document"]),
            [(n["task_cid"], d["dependency_task_cid"], d["kind"])
             for n in case["document"]["observed_tasks"] for d in n["dependencies"]])


def test_graph_read_byte_budget_is_shared_across_queries_and_refuses_incrementally():
    class Cursor:
        def __init__(self, rows):
            self.rows = iter(rows)
            self.fetched = 0
        def fetchone(self):
            self.fetched += 1
            return next(self.rows, None)
    class Connection:
        def execute(self, sql, parameters):
            self.cursor = Cursor([("x" * 32,)] if sql == "first" else [("y" * 32,)] * 10)
            return self.cursor
    connection = Connection()
    budget = [96]
    assert probe._bounded_rows(connection, "first", byte_budget=budget) == [("x" * 32,)]
    with pytest.raises(probe.InvalidCatalogue, match="native_graph_read_bound"):
        probe._bounded_rows(connection, "next", byte_budget=budget)
    assert connection.cursor.fetched == 2


@pytest.mark.parametrize("same_namespace", [False, True])
def test_private_graph_SQL_preserves_authored_scope_and_rejects_unrepresented_namespace_rows(catalogue, observational_graph, same_namespace):
    import duckdb
    case = observational_graph
    document = copy.deepcopy(case["document"])
    if same_namespace:
        document["observed_tasks"][-1]["namespace_binding"]["value"] = NAMESPACE
    ref = case["base"]["envelope"](document)
    arguments = dict(case["arguments"], population_path=ref["path"], population_sha256=ref["sha256"],
                     database_path=str(catalogue[0]))
    connection = duckdb.connect(str(catalogue[0]))
    try:
        for table in ("tasks", "goals", "plans", "task_dependencies"):
            connection.execute("DELETE FROM " + table)
        connection.execute("INSERT INTO plans VALUES (?,?)", [ROOT, "baguqeeragoal0"])
        for goal in case["base"]["authored"]["goals"]:
            connection.execute("INSERT INTO goals VALUES (?,?,?,?,?,?)", [
                goal["goal_cid"], goal["goal_alias"], goal["objective_id"], goal["parent_goal_cid"], "open", 1])
        for task in case["base"]["authored"]["tasks"]:
            connection.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?)", [
                *(task[k] for k in ("task_cid", "task_alias", "goal_cid", "plan_cid", "objective_id")),
                "blocked", 3, json.dumps({"board_namespace": NAMESPACE})])
        for row in graph_rows(document):
            connection.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?)", row)
        edges = list(case["base"]["edges"]) + [(n["task_cid"], d["dependency_task_cid"], d["kind"])
                for n in document["observed_tasks"] for d in n["dependencies"]]
        for edge in edges:
            connection.execute("INSERT INTO task_dependencies VALUES (?,?,?)", list(edge))
    finally:
        connection.close()
    before = catalogue[0].read_bytes()
    result = probe._read_catalogue(arguments)
    assert len(result["tasks"]) == 64 and result["goal_count"] == 34 and result["dependency_count"] == 115
    assert result["dependency_boundary"]["observed_task_count"] == 11
    assert result["dependency_boundary"]["node_count"] == 75
    assert catalogue[0].read_bytes() == before and not Path(str(catalogue[0]) + ".wal").exists()
    change_database(catalogue, "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?)", [
        "baguqeeraunrepresented", "UNREPRESENTED", "baguqeeraforeigngoal", None, None,
        "blocked", 1, json.dumps({"board_namespace": NAMESPACE})])
    with pytest.raises(probe.InvalidCatalogue, match="native_population_mismatch"):
        probe._read_catalogue(arguments)


def test_legacy_v2_lineage_population_keeps_its_original_three_value_contract(tmp_path):
    goal = {"goal_cid": "baguqeeragoal", "goal_alias": "LIR-G090",
            "objective_id": "objective:one", "parent_goal_cid": ""}
    task = {"task_cid": "baguqeeratask", "task_alias": "LIR-9001",
            "goal_cid": goal["goal_cid"], "plan_cid": ROOT,
            "objective_id": "objective:one", "dependencies": []}
    plan = {"plan_cid": ROOT, "goal_cid": goal["goal_cid"], "status": "active", "revision": 1,
            "body_sha256": "f" * 64, "repository_tree_id": TREE, "continuation_of": None,
            "continued_by": None, "revision_cid": None, "delta_cid": None, "idempotency_key": None}
    population = {"schema": probe.LINEAGE_POPULATION_SCHEMA, "board_namespace": NAMESPACE,
                  "plan_root_cid": ROOT, "repository_tree_id": TREE,
                  "tasks": [task], "goals": [goal], "plans": [plan]}
    path = tmp_path / "legacy-lineage.json"; path.write_text(json.dumps(population))
    selected, goals, bound_plan = probe._population({"population_path": str(path),
        "population_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "board_namespace": NAMESPACE, "plan_root_cid": ROOT, "repository_tree_id": TREE})
    assert selected == {task["task_cid"]: task} and goals == {goal["goal_cid"]: goal}
    assert isinstance(bound_plan, probe._LineagePlan) and not isinstance(bound_plan, probe._GraphBoundaryPlan)
    assert bound_plan["lineage"] == [plan]


def test_selected_script_observer_uses_canonical_health_owner_without_a_fake_package(tmp_path, monkeypatch):
    """Actual file execution/import; worker and verifier facts explicitly modeled."""
    import time
    from ipfs_accelerate_py.agent_supervisor.rescue import native_executor_health as health
    selected = runpy.run_path(probe.__file__, run_name="_selected_catalogue_observer")
    assert selected["__package__"] in (None, "") and selected["__file__"] == probe.__file__
    task = {"task_cid": "baguqeeramodeledtask", "task_alias": "MODELED-1",
            "goal_cid": "baguqeeramodeledgoal", "plan_cid": ROOT, "objective_id": "objective:one"}
    population = {"schema": probe.POPULATION_SCHEMA, "board_namespace": NAMESPACE,
        "plan_root_cid": ROOT, "repository_tree_id": TREE, "tasks": [task],
        "goals": [{"goal_cid": task["goal_cid"], "goal_alias": "MODELED-G1",
                   "objective_id": task["objective_id"], "parent_goal_cid": ""}],
        "plans": [{"plan_cid": ROOT, "goal_cid": task["goal_cid"]}]}
    path = tmp_path / "modeled-population.json"; path.write_text(json.dumps(population))
    data = {"tasks": [{**task, "status": "blocked", "revision": 1,
        "completion_receipt_attempt_number": None, "automatic_retry_admitted": None,
        "receipt_max_task_attempts": None, "receipt_attempt_budget_exhausted": None,
        "receipt_failure_kind": None, "receipt_provider_invocation_count": None,
        "retained_execution_outcomes": []}], "goal_count": 1, "dependency_count": 0,
        "attempt_rows": [], "provider_status_rows": [], "claim_state_rows": [],
        "event_cursor": 1, "store_generation": 1, "store_schema_revision": 1,
        "database_uuid": "modeled-database", "database_and_wal_metadata_unchanged": True,
        "read_finished_at_epoch": time.time()}
    seen = []
    def verifier(reference, native, *, now):
        seen.append((reference, native["task_count"]))
        return {"verified": False, "reason": "modeled_verifier_called"}
    monkeypatch.setattr(health, "verify_bound_executor", verifier)
    monkeypatch.setattr(selected["subprocess"], "run", lambda *a, **k:
        subprocess.CompletedProcess(a[0], 0, stdout=json.dumps(data).encode(), stderr=b""))
    profile = {"path": "/modeled/profile.json", "bytes": 1, "sha256": "a" * 64}
    result = selected["observe_native_catalogue"](Path("/modeled/unopened.duckdb"), path,
        board_id="legal-ir-scientific", board_namespace=NAMESPACE, plan_root_cid=ROOT,
        repository_tree_id=TREE, population_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        executor_profile_ref=profile)
    assert seen == [(profile, 1)]
    assert result["details"]["native_catalogue"]["executor_verification"]["reason"] == "modeled_verifier_called"
    assert result["completion_candidate"] is result["details"]["completion_authority"] is False
