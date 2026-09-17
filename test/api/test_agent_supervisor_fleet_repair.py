"""Durable repair scheduling must not duplicate jobs or starve another board."""
import time
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue.fleet_repair import enqueue, next_job, repair_prompt
from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as repair
from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import read_json, write_json


def config(tmp_path):
    return {"state_dir": str(tmp_path), "boards": [
        {"id": "spar", "cwd": str(tmp_path), "hold_files": []},
        {"id": "sawm", "cwd": str(tmp_path), "hold_files": []}],
        "repair_worker": {"cwd": str(tmp_path)}}


def test_duplicate_incident_preserves_running_job_and_cooldown(tmp_path):
    cfg = config(tmp_path)
    incident = tmp_path / "incident.json"
    write_json(incident, {"board_id": "spar", "signature": "one"})
    enqueue(cfg, incident)
    path = tmp_path / "repairs/spar/job.json"
    job = read_json(path)
    job.update(status="running", attempts=2, next_attempt_at=9000)
    write_json(path, job)
    write_json(incident, {"board_id": "spar", "signature": "changed"})
    result = enqueue(cfg, incident)
    assert result["status"] == "running"
    assert read_json(path)["next_attempt_at"] == 9000
    assert read_json(path)["attempts"] == 2
    assert read_json(path)["latest_incident"]["signature"] == "changed"


def test_wait_stall_does_not_launch_coding_repair(tmp_path):
    cfg = config(tmp_path)
    write_json(tmp_path / "sawm/state.json", {
        "stall_class": "in_progress_awaiting_effect",
        "observation": {"health": "stalled", "reason_codes": ["no_task_progress"],
                        "details": {"task_counts": {"in_progress": 2}}},
    })
    write_json(tmp_path / "repairs/sawm/job.json", {
        "status": "queued", "queued_at": 1, "last_started_at": 1, "next_attempt_at": 0,
    })
    write_json(tmp_path / "repairs/spar/job.json", {
        "status": "queued", "queued_at": 2, "last_started_at": 2, "next_attempt_at": 0,
    })
    selected = next_job(cfg, 100)
    assert selected is not None and selected[0]["id"] == "spar"


def test_complete_board_does_not_block_stalled_repair(tmp_path):
    cfg = config(tmp_path)
    write_json(tmp_path / "spar/state.json", {
        "health": "healthy", "observation": {"complete": True},
        "last_action": "publish", "last_action_result": {"status": "published"},
    })
    write_json(tmp_path / "repairs/spar/job.json", {
        "status": "queued", "queued_at": 1, "last_started_at": 1, "next_attempt_at": 0,
    })
    write_json(tmp_path / "repairs/sawm/job.json", {
        "status": "queued", "queued_at": 2, "last_started_at": 2, "next_attempt_at": 50,
    })
    assert next_job(cfg, 100)[0]["id"] == "sawm"


def test_stale_codex_route_is_due_for_llm_router(tmp_path):
    cfg = config(tmp_path)
    write_json(tmp_path / "repairs/spar/job.json", {
        "status": "queued", "queued_at": 1, "last_started_at": 10, "attempts": 3,
        "next_attempt_at": 9_999_999, "repair_route": "codex_exec",
    })
    selected = next_job(cfg, 100)
    assert selected is not None and selected[0]["id"] == "spar"


def test_enqueue_retires_complete_board(tmp_path):
    cfg = config(tmp_path)
    write_json(tmp_path / "spar/state.json", {"observation": {"complete": True}})
    incident = tmp_path / "incident.json"
    write_json(incident, {"board_id": "spar"})
    result = enqueue(cfg, incident)
    assert result["status"] == "retired_complete"
    assert read_json(tmp_path / "repairs/spar/job.json")["status"] == "retired_complete"


def test_short_failed_llm_router_job_retries_after_five_minutes(tmp_path):
    cfg = config(tmp_path)
    write_json(tmp_path / "repairs/sawm/job.json", {
        "status": "queued", "attempts": 64, "queued_at": 1, "last_started_at": 1000,
        "finished_at": 1010, "returncode": 1, "next_attempt_at": 9_999_999,
        "repair_route": repair.LLM_ROUTER_ROUTE, "repair_workspace": str(tmp_path),
    })
    assert next_job(cfg, 1200) is None
    selected = next_job(cfg, 1311)
    assert selected is not None and selected[0]["id"] == "sawm"


def test_enqueue_resets_stale_route_backoff(tmp_path):
    cfg = config(tmp_path)
    path = tmp_path / "repairs/sawm/job.json"
    write_json(path, {"status": "queued", "attempts": 40, "next_attempt_at": 9_999_999,
                      "repair_route": "codex_exec"})
    incident = tmp_path / "incident.json"
    write_json(incident, {"board_id": "sawm"})
    now = time.time()
    result = enqueue(cfg, incident)
    job = read_json(path)
    assert result["status"] == "queued"
    assert job["attempts"] == 40
    assert job["next_attempt_at"] <= now + 1


def test_repair_fairness_and_backoff(tmp_path):
    cfg = config(tmp_path)
    for identifier in ("spar", "sawm"):
        write_json(tmp_path / f"repairs/{identifier}/job.json", {
            "board_id": identifier, "status": "queued", "queued_at": 1,
            "last_started_at": 10 if identifier == "spar" else 0})
    assert next_job(cfg, 100)[0]["id"] == "sawm"
    write_json(tmp_path / "repairs/sawm/job.json", {"status": "queued", "next_attempt_at": 200})
    assert next_job(cfg, 100)[0]["id"] == "spar"


def test_hold_prevents_coding_repair(tmp_path):
    cfg = config(tmp_path)
    hold = tmp_path / "OPERATOR_STOP"
    hold.touch()
    cfg["boards"][0]["hold_files"] = [str(hold)]
    write_json(tmp_path / "repairs/spar/job.json", {"status": "queued"})
    assert next_job(cfg, time.time()) is None


def test_hold_arriving_during_job_prevents_publication(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_repair import verify_job_recovery
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_completion
    hold = tmp_path / "HOLD"
    hold.touch()
    calls = []
    monkeypatch.setattr(fleet_completion, "publish_completed_board", lambda *args: calls.append(args))
    result = verify_job_recovery({"hold_files": [str(hold)], "publication": {"board_id": "spar"}},
                                 {}, {"completion_candidate": True}, tmp_path)
    assert result == {"verified": False, "reason": "operator_hold"}
    assert not calls


def test_unknown_board_cannot_create_job(tmp_path):
    import pytest
    incident = tmp_path / "incident.json"
    write_json(incident, {"board_id": "../../unrelated"})
    with pytest.raises(ValueError, match="not authorized"):
        enqueue(config(tmp_path), incident)


def test_repair_queue_preserves_explicit_cron_launch_custody(tmp_path):
    cfg = config(tmp_path)
    hold = tmp_path / "watchdog.hold"
    hold.write_text("cron owns relaunch")
    board = cfg["boards"][0]
    board.update(hold_files=[str(hold)], launch_only_hold_files=[str(hold)])
    write_json(tmp_path / "repairs/spar/job.json", {"status": "queued"})
    assert next_job(cfg, time.time())[0]["id"] == "spar"
    prompt = repair_prompt(board, {}, cfg, tmp_path / "report.json")
    assert str(hold) in prompt
    assert "Do not start an alternative service" in prompt
    assert hold.exists()


def test_repair_prompt_preserves_authority_and_llama_stop(tmp_path):
    cfg = config(tmp_path)
    prompt = repair_prompt(cfg["boards"][0], {"board_id": "spar"}, cfg, tmp_path / "report.json")
    assert "Never start llama-server" in prompt
    assert "regression test" in prompt
    assert "UNTRUSTED DIAGNOSTIC DATA" in prompt
    assert "active claims zero" in prompt
    assert "never force push" in prompt
    assert "Production supervisors default to DuckDB + Quack" in prompt
    assert "AST/hash/state in a separate DuckDB + Quack instance" in prompt


def test_repair_prompt_requires_hosted_checks_even_for_bypass_capable_accounts(tmp_path):
    cfg = config(tmp_path)
    prompt = repair_prompt(cfg["boards"][0], {}, cfg, tmp_path / "report.json")
    assert "Never push directly to GitHub\nmain" in prompt
    assert "--match-head-commit" in prompt
    assert "all required checks\nhave succeeded" in prompt
    assert "required reviews are satisfied" in prompt
    assert "Local tests do not replace required hosted checks" in prompt
    assert "a ruleset bypass, or a branch-protection bypass" in prompt
    assert "billing, quota, or an outage" in prompt
    assert "if it still pushes directly to main, repair and test that path" in prompt


@pytest.mark.parametrize("health,busy,token,verified", [
    ("healthy", False, "old", False),
    ("healthy", False, None, False),
    ("healthy", True, "old", True),
    ("healthy", False, "new", False),
    ("blocked", True, "new", False),
])
def test_stall_recovery_requires_task_progress(tmp_path, health, busy, token, verified):
    incident = {"observation": {"health": "stalled", "progress_token": "old",
                               "reason_codes": ["no_task_progress"]}}
    result = repair.verify_job_recovery({}, incident,
        {"health": health, "busy": busy, "progress_token": token}, tmp_path)
    assert result["verified"] is verified


def test_raw_completion_does_not_verify_publication(tmp_path):
    result = repair.verify_job_recovery({}, {}, {"health": "complete"}, tmp_path)
    assert result == {"verified": False, "reason": "publication_configuration_required"}


@pytest.mark.parametrize("status,verified", [("held", False), ("published", True)])
def test_completion_runs_current_publication_gate(tmp_path, monkeypatch, status, verified):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_completion
    calls = []
    def publish(manifest, directory):
        calls.append((manifest, directory))
        return {"status": status}
    monkeypatch.setattr(fleet_completion, "publish_completed_board", publish)
    manifest = {"board_id": "spar"}
    result = repair.verify_job_recovery({"publication": manifest}, {},
        {"health": "healthy", "completion_candidate": True}, tmp_path)
    assert result["verified"] is verified
    assert calls == [(manifest, tmp_path)]


def test_verify_independent_work_and_in_progress_are_not_board_not_healthy():
    board = {"id": "doep", "hold_files": []}
    blocked = {
        "health": "blocked", "reason_codes": ["board_has_blocked_or_quarantined_tasks"],
        "details": {"task_counts": {"todo": 23, "blocked": 2, "in_progress": 0},
                    "lanes": [{"daemon": {"pid": 1}}]},
    }
    assert repair.verify_job_recovery(board, {}, blocked, Path("/tmp"))["reason"] == "independent_work_retained"
    awaiting = {
        "health": "stalled", "reason_codes": ["no_task_progress"],
        "details": {"task_counts": {"in_progress": 2, "completed": 20, "todo": 23}},
    }
    assert repair.verify_job_recovery(board, {}, awaiting, Path("/tmp"))["reason"] == "in_progress_awaiting_effect"


def test_supervisor_heals_wait_on_typed_native_stalls():
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import apply_supervisor_heal
    board = {"id": "doep", "cwd": "/absent"}
    unclaimed = apply_supervisor_heal(board, {
        "stall_class": "independent_todos_unclaimed",
        "observation": {"details": {"task_counts": {"todo": 23}, "lanes": [{"daemon": {"pid": 1}}]}},
    })
    assert unclaimed["status"] == "wait"
    assert unclaimed["recipe"] == "native_lanes_own_independent_todos"
    rearmed = apply_supervisor_heal(board, {
        "stall_class": "stalled_no_progress",
        "observation": {
            "reason_codes": ["no_task_progress"],
            "details": {
                "task_counts": {"todo": 28, "retrying": 2, "in_progress": 0},
                "lanes": [{"daemon": {"pid": 1}}],
            },
        },
    })
    assert rearmed["status"] == "wait"
    assert rearmed["recipe"] == "native_lanes_own_independent_todos"
    missing = apply_supervisor_heal(board, {
        "stall_class": "board_checkout_missing",
        "observation": {"reason_codes": ["board_checkout_missing"]},
    })
    assert missing["status"] == "wait"
    assert missing["recipe"] == "deleted_checkout_not_rematerialized"
    dstate = apply_supervisor_heal(board, {
        "stall_class": "kernel_uninterruptible_wait",
        "observation": {"reason_codes": ["lane_0_daemon_process_uninterruptible"]},
    })
    assert dstate["status"] == "wait"
    assert dstate["recipe"] == "kernel_uninterruptible_wait"
    mixed = apply_supervisor_heal(board, {
        "stall_class": "in_progress_awaiting_effect",
        "observation": {
            "reason_codes": ["lane_0_daemon_process_uninterruptible", "no_task_progress"],
            "details": {"task_counts": {"in_progress": 2, "todo": 23}},
        },
    })
    assert mixed["status"] == "wait"
    assert mixed["recipe"] == "kernel_uninterruptible_wait"
    flush = apply_supervisor_heal(board, {
        "stall_class": "in_progress_awaiting_effect",
        "observation": {
            "reason_codes": ["no_task_progress"],
            "details": {
                "task_counts": {"completed": 20, "in_progress": 2, "todo": 23},
                "lanes": [
                    {"lane": 0, "daemon": {"pid": 1, "process_state": "D", "wait_channel": "__flush_work"}},
                    {"lane": 1, "daemon": {"pid": 2, "process_state": "D", "wait_channel": "__flush_work"}},
                    {"lane": 3, "daemon": {"pid": 3, "process_state": "R"}},
                ],
            },
        },
    })
    assert flush["status"] == "wait"
    assert flush["recipe"] == "kernel_uninterruptible_wait"
    blocked = apply_supervisor_heal(board, {
        "stall_class": "blocked_without_independent_work",
        "observation": {"reason_codes": ["no_ready_independent_tasks"]},
    })
    assert blocked["status"] == "wait"
    assert blocked["recipe"] == "todos_waiting_on_blocked_dependencies"
    goals = apply_supervisor_heal(board, {
        "stall_class": "closeout_waiting_on_unsettled_goals",
        "observation": {"reason_codes": ["board_has_unsettled_goals"]},
    })
    assert goals["status"] == "wait"
    assert goals["recipe"] == "native_goals_still_active"
    disabled = apply_supervisor_heal(board, {
        "stall_class": "closeout_waiting_on_unsettled_goals",
        "observation": {
            "reason_codes": ["board_has_unsettled_goals", "goal_closeout_disabled_on_launch"],
            "details": {"task_counts": {"completed": 40}},
        },
    })
    assert disabled["completion_authority"] is False
    assert disabled["recipe"] in {"native_goals_still_active", "provisionally_complete_terminal_goals"}
    assert disabled.get("reason") in {
        "quack_endpoint_absent", "objective_path_missing", "no_active_goals",
        "extra_gate_closeout_not_disabled",
        "active goals moved to provisionally_complete; verification still required",
    }


def test_provisional_goal_heal_uses_quack_and_does_not_verify(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import apply_supervisor_heal
    objectives = tmp_path / "docs" / "objectives.md"
    objectives.parent.mkdir(parents=True)
    objectives.write_text("# Objectives\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"objectives_path": "docs/objectives.md"}))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"boards": [{
        "id": "aseh", "cwd": str(tmp_path),
        "quack_endpoint": "quack:127.0.0.1:41487",
        "config_path": str(config),
        "database_path": str(tmp_path / "control.duckdb"),
        "runtime_root": str(tmp_path),
        "owner_status_path": str(tmp_path / "q" / "quack-state-server.status.json"),
    }]}))
    board = {"id": "aseh", "cwd": str(tmp_path), "probe": {"argv": [
        "probe", "--inventory", str(inventory), "--board", "aseh",
    ]}}
    class Goal:
        goal_id = "ASEH-G000"
    class Source:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def get_goal(self, key):
            return {"status": "active", "revision": 1}
        def compare_and_set_goal_status(self, *args, **kwargs):
            return {"ok": True}
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.objectives.objective_graph.parse_goal_heap",
        lambda text: [Goal()],
    )
    monkeypatch.setattr(fleet_heals, "_open_provisional_goal_source", lambda endpoint: Source())
    result = apply_supervisor_heal(board, {
        "stall_class": "closeout_waiting_on_unsettled_goals",
        "observation": {
            "reason_codes": ["board_has_unsettled_goals", "goal_closeout_disabled_on_launch"],
            "details": {"task_counts": {"completed": 40}, "unsettled_goal_count": 9},
        },
    })
    assert result["status"] == "applied"
    assert result["completion_authority"] is False
    assert result["changed_goal_ids"] == ["ASEH-G000"]
    assert result["recipe"] == "provisionally_complete_terminal_goals"


def test_provisional_goal_heal_maps_missing_quack_attach_token(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import apply_supervisor_heal
    objectives = tmp_path / "docs" / "objectives.md"
    objectives.parent.mkdir(parents=True)
    objectives.write_text("# Objectives\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"objectives_path": "docs/objectives.md"}))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"boards": [{
        "id": "aseh", "cwd": str(tmp_path),
        "quack_endpoint": "quack:127.0.0.1:41487",
        "config_path": str(config),
        "database_path": str(tmp_path / "control.duckdb"),
        "runtime_root": str(tmp_path),
    }]}))
    board = {"id": "aseh", "cwd": str(tmp_path), "probe": {"argv": [
        "probe", "--inventory", str(inventory), "--board", "aseh",
    ]}}
    class Boom:
        def __enter__(self):
            raise RuntimeError("Invalid Input Error: Could not find a Quack authentication token")
        def __exit__(self, *_):
            return False
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.objectives.objective_graph.parse_goal_heap",
        lambda text: [],
    )
    monkeypatch.setattr(fleet_heals, "_open_provisional_goal_source", lambda endpoint: Boom())
    result = apply_supervisor_heal(board, {
        "stall_class": "closeout_waiting_on_unsettled_goals",
        "observation": {
            "reason_codes": ["board_has_unsettled_goals", "goal_closeout_disabled_on_launch"],
            "details": {"task_counts": {"completed": 40}, "unsettled_goal_count": 9},
        },
    })
    assert result["completion_authority"] is False
    assert result["reason"] == "quack_attach_token_absent"
    assert result["status"] == "wait"


def test_goal_status_cas_sql_is_allowlisted_for_extra_gate_drain():
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_owner_mutation import (
        MUTATION_SQL_TO_TEMPLATE,
        QUACK_MUTATION_DOMAIN_EVENT_INSERT,
        QUACK_MUTATION_GOAL_STATUS_CAS,
        QUACK_MUTATION_GOAL_STATUS_TRANSITION,
        mutation_operation,
        normalize_mutation_sql,
    )

    sql = (
        "UPDATE goals SET status = ?, updated_at = ?, revision = ?, "
        "body_json = ? WHERE goal_cid = ? AND revision = ?"
    )
    assert MUTATION_SQL_TO_TEMPLATE[normalize_mutation_sql(sql)] == QUACK_MUTATION_GOAL_STATUS_CAS
    assert mutation_operation([
        {"template_id": QUACK_MUTATION_GOAL_STATUS_CAS,
         "parameters": ["provisionally_complete", "t", 2, "{}", "g", 1]},
        {"template_id": QUACK_MUTATION_DOMAIN_EVENT_INSERT,
         "parameters": ["e", "s", 1, 1, "t", "", "", "sess", "t", "{}"]},
    ]) == QUACK_MUTATION_GOAL_STATUS_TRANSITION


def test_database_task_source_imports_owner_command_contract():
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        FALSE_TERMINAL_BLOCKED_REASON_MARKERS,
        QUACK_OWNER_COMMAND_COMPARE_AND_SET_GOAL_STATUS,
        STALE_IN_PROGRESS_UNSTALL_SECONDS,
        QuackOwnerCommandRemoteError,
        submit_quack_owner_command,
        validate_quack_owner_command,
    )

    assert DatabaseTaskSource.INTERFACE == "DatabaseTaskSource@1"
    assert "quack_transport_unavailable" in FALSE_TERMINAL_BLOCKED_REASON_MARKERS
    assert "callback_authority_incomplete_blocked" in FALSE_TERMINAL_BLOCKED_REASON_MARKERS
    assert "database_unknown_outcome_blocked" in FALSE_TERMINAL_BLOCKED_REASON_MARKERS
    assert "typed_portal_deferral_budget_exhausted" in FALSE_TERMINAL_BLOCKED_REASON_MARKERS
    assert "identity changed" in FALSE_TERMINAL_BLOCKED_REASON_MARKERS
    assert "database_portal_terminal_failure" in FALSE_TERMINAL_BLOCKED_REASON_MARKERS
    assert QUACK_OWNER_COMMAND_COMPARE_AND_SET_GOAL_STATUS == "compare_and_set_goal_status"
    assert STALE_IN_PROGRESS_UNSTALL_SECONDS == 16_200
    assert callable(submit_quack_owner_command)
    assert callable(validate_quack_owner_command)
    assert issubclass(QuackOwnerCommandRemoteError, Exception)


def test_provisional_goal_heal_binds_owner_transport(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import _owner_transport_env

    owner = tmp_path / "q"
    owner.mkdir()
    (owner / "mutations").mkdir()
    (owner / "typed-state-owner.token").write_text("owner-token-value")
    (owner / "quack-state-server.status.json").write_text(json.dumps({
        "store_id": "data/aseh/control.duckdb",
        "identity": {"store_id": "data/aseh/control.duckdb", "generation": 165},
    }))
    env = _owner_transport_env({
        "owner_status_path": str(owner / "quack-state-server.status.json"),
        "runtime_root": str(tmp_path),
        "database_path": str(tmp_path / "control.duckdb"),
    })
    assert env["IPFS_ACCELERATE_AGENT_STATE_STORE_ID"] == "data/aseh/control.duckdb"
    assert env["IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION"] == "165"
    assert env["IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR"] == str(owner / "mutations")
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in env


def test_open_provisional_goal_source_does_not_raise_import_error():
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        _open_provisional_goal_source,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    assert _open_provisional_goal_source.__defaults__ is None
    assert DatabaseTaskSource.__name__ == "DatabaseTaskSource"


def test_unstall_uses_typed_owner_when_quack_attach_token_absent(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals

    monkeypatch.setattr(
        fleet_heals, "_owner_transport_env",
        lambda inventory: {},
    )
    monkeypatch.setattr(
        fleet_heals, "_inventory_board",
        lambda board: {
            "quack_endpoint": "quack:127.0.0.1:27942",
            "owner_status_path": str(tmp_path / "quack-owner/quack-state-server.status.json"),
            "database_path": str(tmp_path / "control.duckdb"),
        },
    )
    (tmp_path / "quack-owner").mkdir()
    (tmp_path / "quack-owner" / "typed-state-owner.token").write_text("a" * 32)
    (tmp_path / "quack-owner" / "quack-state-server.status.json").write_text(json.dumps({
        "lifecycle": "ready",
        "identity": {"store_id": "doep-v1-r5"},
    }))
    sock = tmp_path / "typed.sock"
    sock.write_text("")
    monkeypatch.setattr(fleet_heals, "_typed_owner_socket_path", lambda database: sock)
    recvs = iter([
        {"ok": True, "schema": fleet_heals._TYPED_OWNER_SCHEMA},
        {"ok": True, "result": {"changed": True, "completion_authority": False}},
    ])
    sent = []
    monkeypatch.setattr(fleet_heals, "_typed_owner_send", lambda *a, **k: sent.append(a[1] if a else k))
    monkeypatch.setattr(fleet_heals, "_typed_owner_recv", lambda *a, **k: next(recvs))

    class Sock:
        def settimeout(self, *_):
            return None
        def connect(self, *_):
            return None
        def close(self):
            return None

    monkeypatch.setattr(fleet_heals.socket, "socket", lambda *a, **k: Sock())
    result = fleet_heals.unstall_stale_native_work(
        {"id": "doep", "cwd": str(tmp_path)},
        {"details": {"blocked_task_ids": ["DOEP-044"]}},
    )
    assert result["status"] == "applied"
    assert result["completion_authority"] is False
    assert result["unstalled"][0]["task_alias"] == "DOEP-044"
    assert sent[0]["action"] == "open_status"
    assert sent[0]["client_id"] == "casf-bootstrap-operator:typed-status"
    assert sent[0]["store_id"] == "doep-v1-r5"


def test_board_local_repair_argv_runs_after_local_pass_without_token(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        rearm_locally_validated_blocked_tasks,
    )

    script = tmp_path / "repair.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o700)
    monkeypatch.setattr(
        fleet_heals, "_inventory_board",
        lambda board: {
            "cwd": str(tmp_path),
            "quack_endpoint": "quack:127.0.0.1:27942",
            "repair_argv": [str(script)],
        },
    )
    monkeypatch.setattr(fleet_heals, "_owner_transport_env", lambda inventory: {})
    result = rearm_locally_validated_blocked_tasks(
        {"id": "doep", "cwd": str(tmp_path)},
        {"details": {"blocked_task_ids": ["DOEP-044"]}},
        [{"task_id": "DOEP-044", "status": "passed", "completion_authoritative": False}],
    )
    assert result["status"] == "applied"
    assert result["completion_authority"] is False
    assert result["completion_authoritative"] is False
    assert result["unstalled"][0]["reason"] == "board_local_repair_argv"


def test_claim_verification_recover_uses_live_status_then_board_command(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        rearm_locally_validated_blocked_tasks,
    )

    handoff = tmp_path / "handoff.py"
    handoff.write_text("print('unused')\n")
    calls = []
    monkeypatch.setattr(
        fleet_heals, "_inventory_board",
        lambda board: {
            "id": "doep",
            "cwd": str(tmp_path),
            "quack_endpoint": "quack:127.0.0.1:27942",
            "existing_service": "agent-supervisor-doep-v1.service",
            "ensure_argv": ["systemctl", "--user", "start", "agent-supervisor-doep-v1.service"],
            "status_argv": ["/usr/bin/python3", str(handoff), "status"],
            "repair_argv": ["/usr/bin/python3", str(handoff), "recover-blocked-lock-timeout"],
        },
    )
    monkeypatch.setattr(fleet_heals, "_owner_transport_env", lambda inventory: {})
    copied = tmp_path / "external/ipfs_accelerate/test/api/doep/test_doep_063.py"
    copied.parent.mkdir(parents=True)
    copied.write_text("def test_ok():\n    assert True\n")
    seen_stop = []

    def fake_run(argv, cwd=None, **kwargs):
        calls.append(list(argv))
        if argv[:3] == ["systemctl", "--user", "stop"]:
            seen_stop.append(copied.exists())
        if "authoritative-status" in argv:
            payload = {
                "completion_authority": False,
                "tasks": [
                    {"task_alias": "DOEP-044", "status": "blocked", "revision": 7},
                    {"task_alias": "DOEP-063", "status": "blocked", "revision": 3},
                ],
            }
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload), stderr="")
        if "recover-claim-verification" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="{}", stderr="")
        if argv[:2] == ["systemctl", "--user"]:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="no")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = rearm_locally_validated_blocked_tasks(
        {"id": "doep", "cwd": str(tmp_path)},
        {"details": {"blocked_task_ids": ["DOEP-044", "DOEP-063"]}},
        [
            {"task_id": "DOEP-044", "status": "passed", "completion_authoritative": False},
            {
                "task_id": "DOEP-063",
                "status": "passed",
                "completion_authoritative": False,
                "repaired": ["external/ipfs_accelerate/test/api/doep/test_doep_063.py"],
            },
        ],
    )
    assert copied.is_file()
    assert seen_stop == [False]
    assert result["status"] == "applied"
    assert result["completion_authority"] is False
    assert result["completion_authoritative"] is False
    assert {item["task_alias"] for item in result["unstalled"]} == {"DOEP-044", "DOEP-063"}
    assert any("authoritative-status" in item for call in calls for item in call)
    assert any("recover-claim-verification" in item for call in calls for item in call)
    assert ["systemctl", "--user", "stop", "agent-supervisor-doep-v1.service"] in calls
    assert ["systemctl", "--user", "start", "agent-supervisor-doep-v1.service"] in calls
    spar = rearm_locally_validated_blocked_tasks
    monkeypatch.setattr(
        fleet_heals, "_inventory_board",
        lambda board: {
            "id": "spar",
            "cwd": str(tmp_path),
            "quack_endpoint": "quack:127.0.0.1:1",
            "existing_service": "ipfs-taskboard-spar-supervisor.service",
            "status_argv": ["/usr/bin/python3", str(handoff), "authoritative-status"],
            "repair_argv": ["/usr/bin/python3", str(handoff), "recover-blocked-lock-timeout"],
        },
    )
    calls.clear()
    spar_result = spar(
        {"id": "spar", "cwd": str(tmp_path)},
        {"details": {"blocked_task_ids": ["SPAR-001"]}},
        [{"task_id": "SPAR-001", "status": "passed", "completion_authoritative": False}],
    )
    assert spar_result["completion_authority"] is False
    assert not any(call[:4] == ["systemctl", "--user", "stop", "ipfs-taskboard-spar-supervisor.service"] for call in calls)


def test_live_doep_owner_is_not_stopped_for_claim_verification(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        rearm_locally_validated_blocked_tasks,
    )

    handoff = tmp_path / "handoff.py"
    handoff.write_text("print('unused')\n")
    calls = []

    def fake_run(argv, cwd=None, **kwargs):
        calls.append(list(argv))
        if argv[:3] == ["systemctl", "--user", "is-active"]:
            return subprocess.CompletedProcess(argv, 0, stdout="active\n", stderr="")
        if "authoritative-status" in argv:
            payload = {
                "completion_authority": False,
                "tasks": [{"task_alias": "DOEP-044", "status": "blocked", "revision": 7}],
            }
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload), stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="no")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(fleet_heals, "_owner_transport_env", lambda inventory: {})
    monkeypatch.setattr(
        fleet_heals, "_unstall_via_typed_owner",
        lambda *a, **k: {
            "status": "applied",
            "recipe": "unstall_stale_native_work",
            "completion_authority": False,
            "unstalled": [{"task_alias": "DOEP-044", "reason": "false_terminal_blocked_supervisor_bug"}],
        },
    )
    monkeypatch.setattr(
        fleet_heals, "_inventory_board",
        lambda board: {
            "id": "doep",
            "cwd": str(tmp_path),
            "quack_endpoint": "quack:127.0.0.1:27942",
            "existing_service": "agent-supervisor-doep-v1.service",
            "ensure_argv": ["systemctl", "--user", "start", "agent-supervisor-doep-v1.service"],
            "status_argv": ["/usr/bin/python3", str(handoff), "status"],
            "repair_argv": ["/usr/bin/python3", str(handoff), "recover-blocked-lock-timeout"],
        },
    )
    result = rearm_locally_validated_blocked_tasks(
        {"id": "doep", "cwd": str(tmp_path)},
        {"details": {"blocked_task_ids": ["DOEP-044"], "owner_ready": True}},
        [{"task_id": "DOEP-044", "status": "passed", "completion_authoritative": False}],
    )
    assert result["completion_authority"] is False
    assert not any(call[:3] == ["systemctl", "--user", "stop"] for call in calls)
    assert not any("recover-claim-verification" in item for call in calls for item in call)
    assert result["status"] == "applied"
    assert result["recipe"] == "rearm_locally_validated_blocked_tasks"


def test_unstall_heal_rearms_false_terminal_blocked_without_forging(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import apply_supervisor_heal
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"boards": [{
        "id": "pctdd", "cwd": str(tmp_path),
        "quack_endpoint": "quack:127.0.0.1:27278",
        "database_path": str(tmp_path / "control.duckdb"),
        "runtime_root": str(tmp_path),
        "owner_status_path": str(tmp_path / "quack-owner" / "quack-state-server.status.json"),
    }]}))
    board = {"id": "pctdd", "cwd": str(tmp_path), "probe": {"argv": [
        "probe", "--inventory", str(inventory), "--board", "pctdd",
    ]}}
    class Source:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def unstall_stale_in_progress_tasks(self, **kwargs):
            return {"unstalled": [{
                "task_alias": "PCTDD-006", "task_cid": "bag:one",
                "revision": 5, "reason": "false_terminal_blocked_supervisor_bug",
            }]}
    monkeypatch.setattr(fleet_heals, "_owner_transport_env", lambda inventory: {
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN": "test-token-value",
        "IPFS_ACCELERATE_AGENT_STATE_STORE_ID": "data/control.duckdb",
        "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION": "1",
    })
    monkeypatch.setattr(fleet_heals, "_open_task_source", lambda endpoint, owner_id: Source())
    result = apply_supervisor_heal(board, {
        "stall_class": "independent_work_beside_blocked_peer",
        "observation": {
            "reason_codes": ["board_has_blocked_or_quarantined_tasks"],
            "details": {
                "task_counts": {"blocked": 2, "in_progress": 2, "todo": 28},
                "blocked_task_ids": ["PCTDD-006", "PCTDD-035"],
                "lanes": [{"daemon": {"pid": 9}}],
            },
        },
    })
    assert result["status"] == "applied"
    assert result["completion_authority"] is False
    assert result["recipe"] == "unstall_stale_native_work"
    assert result["unstalled"][0]["task_alias"] == "PCTDD-006"
    recorded = apply_supervisor_heal(board, {
        "stall_class": "independent_work_beside_blocked_peer",
        "observation": {
            "details": {"lanes": [{"daemon": {"pid": 9}}]},
        },
        "last_action_result": result,
    })
    assert recorded.get("completion_authority") is False
    assert recorded["recipe"] in {
        "unstall_stale_native_work", "independent_work_has_live_workers",
    }


def test_retire_settled_mutation_inbox_drops_old_dones_and_typed_commands(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_owner_mutation import (
        retire_settled_mutation_inbox,
    )
    inbox = tmp_path / "mutations"
    inbox.mkdir()
    old_done = inbox / "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.done.json"
    live_req = inbox / "baguqeerabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.request.json"
    typed = inbox / ("a" * 32 + ".request.json")
    old_done.write_text("{}")
    live_req.write_text("{}")
    typed.write_text("{}")
    os.utime(old_done, (0, 0))
    os.utime(typed, (0, 0))
    removed = retire_settled_mutation_inbox(inbox, now_ms=10_000_000, limit=16)
    assert removed == 2
    assert live_req.is_file()
    assert not old_done.exists()
    assert not typed.exists()


def test_owner_transport_binding_uses_storage_schema_fingerprint(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        _mutation_binding_from_owner_status,
    )
    binding = _mutation_binding_from_owner_status({
        "store_id": "data/control.duckdb",
        "storage_schema_fingerprint": "baguqeerah2s7odhlvt7hjaaxzfkax6dqcydviq7uztbtg5xmbilz5vlw4gia",
        "identity": {
            "server_id": "server:e4cfbc28-4f8f-4283-b714-d9623a0cfa1d",
            "store_id": "data/control.duckdb",
            "database_uuid": "496924b1-85df-439c-afcf-cb39a6ed0efa",
            "schema_revision": 1,
            "schema_fingerprint": "sha256:3ea5f70cebacfe748017c9540bf87016075443f4ccc33376ec0a179ed576e190",
            "generation": 117,
            "process_birth_id": "birth:838811802d62085a88f8bee7a4862262",
            "listen_uri": "quack:127.0.0.1:27278",
            "extension_fingerprint": "sha256:b77954ae50ecc06e10c6e20fc6fd421d73b5c31cf72bb60ae3f29b1f8a85f20b",
        },
    })
    assert binding is not None
    assert binding["schema_fingerprint"].startswith("baguqeera")
    assert binding["generation"] == 117


def test_compare_and_set_uses_intent_when_mutation_binding_ready(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.task_sources import database_task_source as dts
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
        _mutation_transport_ready,
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_BINDING", '{"server_id":"s"}')
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", "/tmp/mutations")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "token-value")
    assert _mutation_transport_ready() is True
    calls = []
    class Intent:
        uses_quack_transport = True
        def cas_task_status(self, **kwargs):
            calls.append(kwargs)
            from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentReceipt
            return IntentReceipt(
                event_id="bag:event", event_type="intent.task_status_changed",
                global_sequence=9, recorded_at="t", subject_id="task:one",
                revision=2, changed=True, details={"previous_status": "blocked"},
            )
        def get_task(self, key):
            return None
    source = DatabaseTaskSource.__new__(DatabaseTaskSource)
    source._intent = Intent()
    record = dts.TaskRecord(
        task_cid="task:one", task_alias="PCTDD-006", goal_cid="goal:one",
        ordinal=1, status="retrying", revision=2,
    )
    monkeypatch.setattr(source, "get_task", lambda key: record)
    result = source._cas_via_intent_repository("PCTDD-006", 1, "retrying", {"operation": "x"})
    assert result.changed is True
    assert result.previous_status == "blocked"
    assert calls[0]["new_status"] == "retrying"
    assert getattr(result, "completion_authority", False) is False


def test_local_validation_of_blocked_candidate_does_not_admit_completion(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        apply_supervisor_heal, run_local_blocked_candidate_validation,
    )
    receipts = tmp_path / "external/ipfs_accelerate/artifacts/doep/receipts"
    receipts.mkdir(parents=True)
    (receipts / "DOEP-044.json").write_text(json.dumps({
        "task_id": "DOEP-044",
        "completion_authoritative": False,
        "validation": {"commands": [{
            "argv": ["python3", "-m", "pytest", "test/api/doep/test_doep_044.py", "-q"],
            "cwd": "external/ipfs_accelerate",
        }]},
    }))
    accel = tmp_path / "external/ipfs_accelerate"
    accel.mkdir(parents=True, exist_ok=True)
    (accel / "test/api/doep").mkdir(parents=True)
    (accel / "test/api/doep/test_doep_044.py").write_text("def test_ok():\n    assert True\n")
    calls = []
    def fake_run(argv, cwd=None, **kwargs):
        calls.append((list(argv), cwd))
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    observation = {
        "details": {"blocked_task_ids": ["DOEP-044", "DOEP-063"],
                    "task_counts": {"todo": 23, "blocked": 2},
                    "owner_ready": True},
    }
    result = run_local_blocked_candidate_validation({"cwd": str(tmp_path)}, observation)
    assert result["completion_authoritative"] is False
    assert result["recipe"] == "local_validation_pending_native_admission"
    assert result["results"][0]["status"] == "passed"
    assert result["results"][0]["completion_authoritative"] is False
    assert result["results"][1]["status"] == "receipt_missing"
    assert calls and calls[0][0][0] == "python3"
    healed = apply_supervisor_heal(
        {"cwd": str(tmp_path)},
        {"stall_class": "blocked_without_independent_work", "observation": observation},
    )
    assert healed["completion_authoritative"] is False
    assert healed["recipe"] == "local_validation_pending_native_admission"
    recorded = apply_supervisor_heal(
        {"cwd": str(tmp_path)},
        {"stall_class": "blocked_without_independent_work", "observation": observation,
         "last_action_result": healed},
    )
    assert recorded["completion_authoritative"] is False
    assert recorded["recipe"] == "local_validation_pending_native_admission"
    assert recorded["results"][0]["status"] == "passed"
    assert recorded["results"][1]["status"] == "receipt_missing"
    again = apply_supervisor_heal(
        {"cwd": str(tmp_path)},
        {"stall_class": "blocked_without_independent_work", "observation": observation,
         "last_action_result": recorded},
    )
    assert again["completion_authoritative"] is False
    assert again["results"][1]["status"] == "receipt_missing"
    assert len(calls) == 4


def test_missing_receipt_is_materialized_from_validation_profile(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        run_local_blocked_candidate_validation,
    )

    config = tmp_path / "config"
    config.mkdir()
    (config / "board_validation_profiles.json").write_text(json.dumps({
        "DOEP-063": {
            "task_id": "DOEP-063",
            "plan_revision": "DOEP-PLAN-V5",
            "profile_id": "doep-validation/DOEP-PLAN-V5/DOEP-063@1",
            "receipt": "external/ipfs_accelerate/artifacts/doep/receipts/DOEP-063.json",
            "commands": [{
                "argv": [
                    "python3", "-m", "pytest",
                    "external/ipfs_accelerate/test/api/doep/test_doep_063.py", "-q",
                ],
            }],
        },
    }))
    test_src = tmp_path / "external/ipfs_accelerate/test/api/doep"
    test_src.mkdir(parents=True)
    (test_src / "test_doep_063.py").write_text("def test_ok():\n    assert True\n")
    calls = []
    def fake_run(argv, cwd=None, **kwargs):
        calls.append((list(argv), cwd))
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_local_blocked_candidate_validation(
        {"cwd": str(tmp_path)},
        {"details": {"blocked_task_ids": ["DOEP-063"], "owner_ready": True}},
    )
    receipt = tmp_path / "external/ipfs_accelerate/artifacts/doep/receipts/DOEP-063.json"
    payload = json.loads(receipt.read_text())
    assert payload["completion_authoritative"] is False
    assert payload["candidate_status"] == "receipt_materialized"
    assert payload["supervisor_acceptance"]["completion_authoritative"] is False
    assert result["completion_authoritative"] is False
    assert result["recipe"] == "local_validation_pending_native_admission"
    assert result["results"][0]["status"] == "passed"
    assert result["results"][0]["completion_authoritative"] is False
    assert calls and "test_doep_063.py" in calls[0][0][-2]


def test_missing_pytest_source_does_not_fail_sibling_local_pass(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        run_local_blocked_candidate_validation,
    )

    receipts = tmp_path / "external/ipfs_accelerate/artifacts/doep/receipts"
    receipts.mkdir(parents=True)
    (receipts / "DOEP-044.json").write_text(json.dumps({
        "task_id": "DOEP-044",
        "completion_authoritative": False,
        "validation": {"commands": [{
            "argv": ["python3", "-m", "pytest", "test/api/doep/test_doep_044.py", "-q"],
            "cwd": "external/ipfs_accelerate",
        }]},
    }))
    (receipts / "DOEP-063.json").write_text(json.dumps({
        "task_id": "DOEP-063",
        "completion_authoritative": False,
        "validation": {"commands": [{
            "argv": [
                "python3", "-m", "pytest",
                "external/ipfs_accelerate/test/api/doep/test_doep_063.py", "-q",
            ],
        }]},
    }))
    accel = tmp_path / "external/ipfs_accelerate"
    accel.mkdir(parents=True, exist_ok=True)
    (accel / "test/api/doep").mkdir(parents=True)
    (accel / "test/api/doep/test_doep_044.py").write_text("def test_ok():\n    assert True\n")
    calls = []
    def fake_run(argv, cwd=None, **kwargs):
        calls.append((list(argv), cwd))
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_local_blocked_candidate_validation(
        {"cwd": str(tmp_path)},
        {"details": {"blocked_task_ids": ["DOEP-044", "DOEP-063"], "owner_ready": True}},
    )
    assert result["completion_authoritative"] is False
    assert result["status"] == "applied"
    by_id = {item["task_id"]: item for item in result["results"]}
    assert by_id["DOEP-044"]["status"] == "passed"
    assert by_id["DOEP-063"]["status"] == "validation_source_missing"
    assert "test_doep_063.py" in by_id["DOEP-063"]["missing"]
    assert len(calls) == 1
    assert "overlay copy work" in result["reason"]


def test_overlay_copy_is_skipped_until_owner_is_ready(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        run_local_blocked_candidate_validation,
    )

    overlay = tmp_path / "overlay"
    source = overlay / "test/api/doep/test_doep_063.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_ok():\n    assert True\n")
    monkeypatch.setattr(fleet_heals, "supervisor_overlay_root", lambda: str(overlay))
    config = tmp_path / "config"
    config.mkdir()
    (config / "board_validation_profiles.json").write_text(json.dumps({
        "DOEP-063": {
            "task_id": "DOEP-063",
            "receipt": "external/ipfs_accelerate/artifacts/doep/receipts/DOEP-063.json",
            "required_outputs": [
                "external/ipfs_accelerate/test/api/doep/test_doep_063.py",
            ],
            "commands": [{
                "argv": [
                    "python3", "-m", "pytest",
                    "external/ipfs_accelerate/test/api/doep/test_doep_063.py",
                    "-q",
                ],
            }],
        },
    }))
    receipts = tmp_path / "external/ipfs_accelerate/artifacts/doep/receipts"
    receipts.mkdir(parents=True)
    (receipts / "DOEP-063.json").write_text(json.dumps({
        "task_id": "DOEP-063",
        "completion_authoritative": False,
        "validation": {"commands": [{
            "argv": [
                "python3", "-m", "pytest",
                "external/ipfs_accelerate/test/api/doep/test_doep_063.py",
                "-q",
            ],
        }]},
    }))
    dest = tmp_path / "external/ipfs_accelerate/test/api/doep/test_doep_063.py"
    result = run_local_blocked_candidate_validation(
        {"cwd": str(tmp_path)},
        {"details": {"blocked_task_ids": ["DOEP-063"], "owner_ready": False}},
    )
    assert not dest.exists()
    assert result["results"][0]["status"] == "validation_source_missing"
    assert result["completion_authoritative"] is False


def test_owner_missing_heal_removes_overlay_copies(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        apply_supervisor_heal,
    )
    nested = tmp_path / "external/ipfs_accelerate"
    copied = nested / "test/api/doep/test_doep_063_implement_accelerate_freshness_and_selection.py"
    copied.parent.mkdir(parents=True)
    copied.write_text("def test_ok():\n    assert True\n")
    subprocess.run(["git", "init"], cwd=str(nested), check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(nested), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(nested), "config", "user.name", "t"], check=True)
    result = apply_supervisor_heal(
        {"id": "doep", "cwd": str(tmp_path)},
        {"stall_class": "owner_missing", "observation": {"details": {"owner_ready": False}}},
    )
    assert result["recipe"] == "clear_overlay_copies_for_owner_start"
    assert result["completion_authority"] is False
    assert not copied.exists()


def test_missing_pytest_source_is_copied_from_overlay(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        run_local_blocked_candidate_validation,
    )

    overlay = tmp_path / "overlay"
    source = overlay / "test/api/doep/test_doep_063_implement_accelerate_freshness_and_selection.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_ok():\n    assert True\n")
    output = overlay / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-063.json"
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps({
        "task_id": "DOEP-063",
        "completion_authoritative": False,
    }))
    monkeypatch.setattr(fleet_heals, "supervisor_overlay_root", lambda: str(overlay))

    config = tmp_path / "config"
    config.mkdir()
    (config / "board_validation_profiles.json").write_text(json.dumps({
        "DOEP-063": {
            "task_id": "DOEP-063",
            "receipt": "external/ipfs_accelerate/artifacts/doep/receipts/DOEP-063.json",
            "required_outputs": [
                "external/ipfs_accelerate/test/api/doep/test_doep_063_implement_accelerate_freshness_and_selection.py",
                "external/ipfs_accelerate/artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-063.json",
            ],
            "commands": [{
                "argv": [
                    "python3", "-m", "pytest",
                    "external/ipfs_accelerate/test/api/doep/test_doep_063_implement_accelerate_freshness_and_selection.py",
                    "-q",
                ],
            }],
        },
    }))
    receipts = tmp_path / "external/ipfs_accelerate/artifacts/doep/receipts"
    receipts.mkdir(parents=True)
    (receipts / "DOEP-063.json").write_text(json.dumps({
        "task_id": "DOEP-063",
        "completion_authoritative": False,
        "validation": {"commands": [{
            "argv": [
                "python3", "-m", "pytest",
                "external/ipfs_accelerate/test/api/doep/test_doep_063_implement_accelerate_freshness_and_selection.py",
                "-q",
            ],
        }]},
    }))
    calls = []
    def fake_run(argv, cwd=None, **kwargs):
        calls.append((list(argv), cwd))
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_local_blocked_candidate_validation(
        {"cwd": str(tmp_path)},
        {"details": {"blocked_task_ids": ["DOEP-063"], "owner_ready": True}},
    )
    dest = tmp_path / (
        "external/ipfs_accelerate/test/api/doep/"
        "test_doep_063_implement_accelerate_freshness_and_selection.py"
    )
    copied_output = tmp_path / (
        "external/ipfs_accelerate/artifacts/"
        "agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-063.json"
    )
    assert dest.is_file()
    assert dest.read_text() == source.read_text()
    assert copied_output.is_file()
    assert json.loads(copied_output.read_text())["completion_authoritative"] is False
    assert result["completion_authoritative"] is False
    assert result["results"][0]["status"] == "passed"
    assert any(
        "test_doep_063_implement_accelerate_freshness_and_selection.py" in item
        for item in result["results"][0]["repaired"]
    )
    assert calls and "test_doep_063_implement_accelerate_freshness_and_selection.py" in calls[0][0][-2]

    dest.write_text("def test_keep():\n    assert True\n")
    again = run_local_blocked_candidate_validation(
        {"cwd": str(tmp_path)},
        {"details": {"blocked_task_ids": ["DOEP-063"], "owner_ready": True}},
    )
    assert dest.read_text() == "def test_keep():\n    assert True\n"
    assert again["completion_authoritative"] is False
    assert json.loads((receipts / "DOEP-063.json").read_text())["completion_authoritative"] is False


def test_source_missing_retries_once_overlay_has_the_file(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        local_validation_already_recorded,
    )

    overlay = tmp_path / "overlay"
    source = overlay / "test/api/doep/test_doep_063.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_ok():\n    assert True\n")
    monkeypatch.setattr(fleet_heals, "supervisor_overlay_root", lambda: str(overlay))
    state = {
        "stall_class": "blocked_without_independent_work",
        "observation": {"details": {"blocked_task_ids": ["DOEP-063"]}},
        "last_action_result": {
            "recipe": "local_validation_pending_native_admission",
            "results": [{
                "task_id": "DOEP-063",
                "status": "validation_source_missing",
                "missing": "external/ipfs_accelerate/test/api/doep/test_doep_063.py",
                "completion_authoritative": False,
            }],
        },
    }
    assert local_validation_already_recorded(state) is False
    source.unlink()
    assert local_validation_already_recorded(state) is True


def test_recorded_local_pass_retries_if_copied_source_disappeared(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        local_validation_already_recorded,
    )

    relative = (
        "external/ipfs_accelerate/test/api/doep/"
        "test_doep_063_implement_accelerate_freshness_and_selection.py"
    )
    state = {
        "observation": {"details": {"blocked_task_ids": ["DOEP-063"]}},
        "last_action_result": {
            "recipe": "local_validation_pending_native_admission",
            "results": [{
                "task_id": "DOEP-063",
                "status": "passed",
                "repaired": [relative],
                "completion_authoritative": False,
            }],
        },
    }
    board = {"cwd": str(tmp_path)}
    assert local_validation_already_recorded(state, board) is False
    dest = tmp_path / relative
    dest.parent.mkdir(parents=True)
    dest.write_text("def test_ok():\n    assert True\n")
    assert local_validation_already_recorded(state, board) is True


def test_candidate_receipt_admits_owner_rearm_only_with_pytest_source(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        candidate_receipt_admits_owner_rearm,
    )

    receipts = tmp_path / "external/ipfs_accelerate/artifacts/doep/receipts"
    receipts.mkdir(parents=True)
    test_rel = "external/ipfs_accelerate/test/api/doep/test_doep_063.py"
    (receipts / "DOEP-063.json").write_text(json.dumps({
        "task_id": "DOEP-063",
        "completion_authoritative": False,
        "validation": {"commands": [{
            "argv": ["python3", "-m", "pytest", test_rel, "-q"],
        }]},
    }))
    assert candidate_receipt_admits_owner_rearm("DOEP-063", cwd=tmp_path) is False
    dest = tmp_path / test_rel
    dest.parent.mkdir(parents=True)
    dest.write_text("def test_ok():\n    assert True\n")
    assert candidate_receipt_admits_owner_rearm("DOEP-063", cwd=tmp_path) is True
    (receipts / "DOEP-063.json").write_text(json.dumps({
        "task_id": "DOEP-063",
        "completion_authoritative": True,
        "validation": {"commands": [{
            "argv": ["python3", "-m", "pytest", test_rel, "-q"],
        }]},
    }))
    assert candidate_receipt_admits_owner_rearm("DOEP-063", cwd=tmp_path) is False


def test_materialized_receipt_does_not_overwrite_existing_receipt(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        _materialize_candidate_receipt,
    )

    receipts = tmp_path / "external/ipfs_accelerate/artifacts/doep/receipts"
    receipts.mkdir(parents=True)
    existing = receipts / "DOEP-044.json"
    existing.write_text(json.dumps({"task_id": "DOEP-044", "keep": True}))
    config = tmp_path / "config"
    config.mkdir()
    (config / "board_validation_profiles.json").write_text(json.dumps({
        "DOEP-044": {
            "task_id": "DOEP-044",
            "receipt": "external/ipfs_accelerate/artifacts/doep/receipts/DOEP-044.json",
            "commands": [{"argv": ["python3", "-m", "pytest", "x.py", "-q"]}],
        },
    }))
    path = _materialize_candidate_receipt(
        tmp_path, {"cwd": str(tmp_path)}, "DOEP-044",
    )
    assert path == existing
    assert json.loads(existing.read_text())["keep"] is True


def test_restore_dirty_control_plane_checkouts_configured_paths(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import restore_dirty_control_plane
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    tracked = repo / "ipfs_accelerate_py" / "agent_supervisor"
    tracked.mkdir(parents=True)
    (tracked / "runner.py").write_text("clean\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    (tracked / "runner.py").write_text("dirty\n")
    result = restore_dirty_control_plane({}, {
        "details": {"source_integrity": {
            "reason": "configured_control_plane_dirty",
            "checked": [{"repository": str(repo), "paths": ["ipfs_accelerate_py/agent_supervisor"]}],
        }},
    })
    assert result["status"] == "applied"
    assert (tracked / "runner.py").read_text() == "clean\n"


def test_llm_router_repair_argv_uses_provider_fallback_not_codex_only(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_repair import llm_router_repair_argv
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("fix it")
    last = tmp_path / "last.txt"
    argv = llm_router_repair_argv(
        {"argv": ["codex", "exec"], "cwd": str(tmp_path)}, prompt, last,
    )
    assert argv[2:4] == ["-m", "ipfs_accelerate_py.agent_supervisor.provider_fallback_runner"]
    assert "--primary-provider" in argv and "grok" in argv
    assert "--fallback-provider" in argv and "codex" in argv
    assert "--probe-route-readiness" in argv
    primary = json.loads(argv[argv.index("--primary-command-json") + 1])
    assert primary[-2:] == ["--workspace", str(tmp_path)]
    assert "grok_cli_runner.py" in primary[2]
    assert "ipfs-accelerate-provider-isolated" not in primary
    fallback = json.loads(argv[argv.index("--fallback-command-json") + 1])
    assert fallback[:2] == ["codex", "exec"]


def test_should_reclaim_orphan_sigterm_and_stale_workspace(tmp_path):
    cfg = config(tmp_path)
    write_json(tmp_path / "repairs/sawm/job.json", {
        "status": "queued", "attempts": 1, "last_started_at": 10, "finished_at": 20,
        "returncode": -15, "repair_route": repair.LLM_ROUTER_ROUTE,
    })
    assert repair.should_reclaim_repair_unit(cfg) is True
    write_json(tmp_path / "repairs/sawm/job.json", {
        "status": "running", "repair_route": repair.LLM_ROUTER_ROUTE,
        "repair_workspace": str(tmp_path),
    })
    assert repair.should_reclaim_repair_unit(cfg) is False
    write_json(tmp_path / "sawm/state.json", {
        "stall_class": "in_progress_awaiting_effect",
        "observation": {"health": "stalled", "reason_codes": ["no_task_progress"],
                        "details": {"task_counts": {"in_progress": 2}}},
    })
    assert repair.should_reclaim_repair_unit(cfg) is True
    write_json(tmp_path / "sawm/state.json", {})
    write_json(tmp_path / "repairs/sawm/job.json", {
        "status": "running", "repair_route": repair.LLM_ROUTER_ROUTE,
        "repair_workspace": str(tmp_path / "maintenance"),
    })
    assert repair.should_reclaim_repair_unit(cfg) is True


def test_stale_maintenance_workspace_is_due_for_board_checkout(tmp_path):
    cfg = config(tmp_path)
    board = tmp_path / "board"
    board.mkdir()
    cfg["boards"][0]["cwd"] = str(board)
    write_json(tmp_path / "repairs/spar/job.json", {
        "status": "queued", "queued_at": 1, "last_started_at": 10, "attempts": 1,
        "next_attempt_at": 9_999_999, "repair_route": repair.LLM_ROUTER_ROUTE,
        "repair_workspace": str(tmp_path / "maintenance"),
    })
    selected = next_job(cfg, 100)
    assert selected is not None and selected[0]["id"] == "spar"


def test_llm_router_repair_argv_binds_board_checkout(tmp_path):
    board = tmp_path / "board"
    board.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("fix it")
    argv = repair.llm_router_repair_argv(
        {"argv": ["codex", "exec"], "cwd": str(tmp_path)},
        prompt, tmp_path / "last.txt", workspace=board,
    )
    workspace = str(board.resolve())
    assert argv[argv.index("--workspace") + 1] == workspace
    assert json.loads(argv[argv.index("--primary-command-json") + 1])[-1] == workspace
    fallback = json.loads(argv[argv.index("--fallback-command-json") + 1])
    assert workspace in fallback
    assert repair.repair_workspace({"cwd": str(board)}, {"cwd": str(tmp_path)}) == board.resolve()


def test_run_job_keeps_backoff_for_idle_healthy_probe(tmp_path, monkeypatch):
    # This case exercises launch continuity; capacity failures have separate tests.
    monkeypatch.setattr(repair.os, "fstatvfs", lambda _fd: SimpleNamespace(
        f_frsize=4096, f_bavail=1 << 30, f_favail=1 << 20))
    cfg = config(tmp_path)
    cfg["repair_worker"].update(argv=["codex", "exec"], retry_seconds=100)
    board = dict(cfg["boards"][0], probe={"argv": ["probe"]})
    directory = tmp_path / "repairs/spar"
    prior_report = directory / "report-previous.json"
    write_json(prior_report, {"status": "blocked", "next_action": "continue regression fix"})
    path = directory / "job.json"
    write_json(path, {"status": "queued", "attempts": 2,
        "report_path": str(prior_report), "latest_incident": {
            "observation": {"health": "stalled", "progress_token": "old",
                            "reason_codes": ["no_task_progress"]}}})
    def command(spec, **kwargs):
        if spec["argv"][0] == "systemctl":
            return {"returncode": 3, "stdout": "inactive"}
        return {"returncode": 0, "stdout": json.dumps({
            "board_id": "spar", "health": "healthy", "busy": False,
            "progress_token": "old", "reason_codes": []})}
    class Process:
        def __init__(self, *args, **kwargs):
            self.prompt = kwargs["stdin"].read().decode()
            assert "continue regression fix" in self.prompt
            active = read_json(path)
            assert active["status"] == "running"
            assert active["log_path"] == kwargs["stdout"].name
            assert Path(active["log_path"]).exists()
        def wait(self, timeout):
            return 0
    monkeypatch.setattr(repair, "command", command)
    monkeypatch.setattr(repair.shutil, "which", lambda executable: "/test/" + executable)
    monkeypatch.setattr(repair.subprocess, "Popen", Process)
    result = repair.run_job(cfg, board, path)
    durable = read_json(path)
    assert result["status"] == "queued"
    assert durable["attempts"] == 3
    assert durable["next_attempt_at"] > time.time()
    assert durable["verification"]["reason"] == "task_progress_not_verified"
    assert durable["report_path"] != str(prior_report)
    assert durable["last_valid_report_path"] == str(prior_report)
    # This worker returned zero without writing its promised report. The next
    # attempt must retain the actual previous instructions and retry budget.
    result = repair.run_job(cfg, board, path)
    assert result["status"] == "queued"
    durable = read_json(path)
    assert durable["attempts"] == 4
    assert durable["last_valid_report_path"] == str(prior_report)


@pytest.mark.parametrize("latest", [None, "{incomplete", "{}", "[]"])
def test_reportless_worker_keeps_last_usable_context(tmp_path, latest):
    old, new = tmp_path / "report-old.json", tmp_path / "report-new.json"
    write_json(old, {"status": "blocked", "next_action": "recover exact callback provenance"})
    if latest is not None:
        new.write_text(latest)
    job = {"report_path": str(new), "last_valid_report_path": str(old)}
    context = repair._continuation_context(job, tmp_path)
    assert context["path"] == str(old)
    assert context["continuation"]["next_action"] == "recover exact callback provenance"
    assert "latest_report_unavailable" in context
    write_json(new, {"status": "blocked", "next_action": "new independently observed incident"})
    assert repair._continuation_context(job, tmp_path)["path"] == str(new)


def test_remembered_report_does_not_escape_its_board(tmp_path):
    other = tmp_path / "another-board.json"
    write_json(other, {"next_action": "foreign context"})
    board = tmp_path / "board"
    board.mkdir()
    job = {"report_path": str(board / "missing.json"), "last_valid_report_path": str(other)}
    assert "continuation" not in repair._continuation_context(job, board)
    link = board / "report-link.json"
    link.symlink_to(other)
    job["last_valid_report_path"] = str(link)
    assert "continuation" not in repair._continuation_context(job, board)


def test_prior_report_continuation_is_bounded_and_confined(tmp_path):
    directory = tmp_path / "reports"
    directory.mkdir()
    prior = directory / "prior.json"
    prior.write_text("x" * 20000)
    context = repair._prior_report_context(str(prior), directory)
    assert len(context["content"]) == 16000
    assert context["truncated"] is True
    assert context["path"] == str(prior)
    foreign = tmp_path / "foreign.json"
    foreign.write_text("secret")
    assert "content" not in repair._prior_report_context(str(foreign), directory)
    link = directory / "linked.json"
    link.symlink_to(foreign)
    assert "content" not in repair._prior_report_context(str(link), directory)


def test_prior_report_keeps_next_action_after_large_deployment_log(tmp_path):
    prior = tmp_path / "report.json"
    write_json(prior, {
        "deployment": {"stdout": "already deployed\n" * 20000},
        "next_action": "Implement the missing native goal CAS consumer; do not repeat the readiness cutover.",
        "remaining_blockers": ["independent_current_source_acceptance_required"],
        "status": "blocked",
    })
    context = repair._prior_report_context(str(prior), tmp_path)
    assert context["continuation"]["next_action"].startswith("Implement the missing native goal CAS")
    assert context["continuation"]["remaining_blockers"] == ["independent_current_source_acceptance_required"]
    assert context["continuation"]["deployment"]["truncated"] is True
    assert len(json.dumps(context)) < 16000
    assert context["truncated"] is True


def test_prior_report_caps_oversized_or_non_object_input(tmp_path):
    prior = tmp_path / "report.json"
    prior.write_text('{"deployment":"' + "x" * (2 * 1024 * 1024) + '"}')
    context = repair._prior_report_context(str(prior), tmp_path)
    assert context["error"] == "prior_report_too_large"
    assert len(context["content"]) == 16000
    assert "continuation" not in context
    prior.write_text('["not a report"]')
    assert repair._prior_report_context(str(prior), tmp_path)["error"] == "prior_report_not_object"


def test_queue_reports_future_and_held_work_instead_of_idle(tmp_path):
    cfg = config(tmp_path)
    hold = tmp_path / "HOLD"
    hold.touch()
    cfg["boards"][1]["hold_files"] = [str(hold)]
    for identifier, due in (("spar", 900), ("sawm", 50)):
        write_json(tmp_path / f"repairs/{identifier}/job.json", {
            "status": "queued", "next_attempt_at": due})
    assert next_job(cfg, 100) is None
    result = repair.queue_status(cfg, 100)
    assert result["status"] == "waiting"
    assert result["next_attempt_at"] == 900
    assert result["waiting"] == [{"board_id": "spar", "next_attempt_at": 900}]
    assert result["held"] == [{"board_id": "sawm", "next_attempt_at": 50}]


def test_pythonpath_overlay_is_not_a_pending_runtime_update(tmp_path, monkeypatch):
    release = tmp_path / "release"
    release.mkdir()
    loaded = Path(repair.__file__).resolve().parents[3]
    monkeypatch.setenv("PYTHONPATH", f"{loaded}:{release}")
    assert not repair.runtime_update_pending({"runtime_release": str(release)})
    assert repair.runtime_update_pending({"runtime_release": str(tmp_path / "next")})


def test_dispatcher_finishes_job_before_adopting_staged_release(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    for board in cfg["boards"]:
        board["probe"] = {"argv": ["probe"]}
    cfg["runtime_release"] = str(Path(repair.__file__).resolve().parents[3])
    assert not repair.runtime_update_pending(cfg)
    assert not repair.runtime_update_pending({})
    config_path = tmp_path / "config.json"
    write_json(config_path, cfg)
    write_json(tmp_path / "repairs/spar/job.json", {"status": "queued"})
    calls = []
    def run_job(current, board, path):
        calls.append(board["id"])
        assert read_json(tmp_path / "repair-worker.json")["status"] == "running"
        changed = dict(cfg, runtime_release=str(tmp_path / "new-release"))
        write_json(config_path, changed)
        # Updating the on-disk config cannot interrupt the active job.
        assert not repair.runtime_update_pending(current)
        return {"status": "verified_healthy"}
    class Event:
        def is_set(self):
            return False
        def wait(self, seconds):
            pass
        def set(self):
            pass
    monkeypatch.setattr(repair, "run_job", run_job)
    monkeypatch.setattr(repair.threading, "Event", Event)
    monkeypatch.setattr(repair.signal, "signal", lambda *args: None)
    assert repair.main(["run", "--config", str(config_path)]) == 0
    assert calls == ["spar"]
    assert read_json(tmp_path / "repair-worker.json")["status"] == "runtime_update_ready"


def test_doep_local_pass_rearms_blocked_tasks_without_wrapping_owner(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import apply_supervisor_heal

    user_dir = tmp_path / "systemd"
    restarts = []
    real_admit = fleet_heals.admit_native_owner_overlay
    monkeypatch.setattr(
        fleet_heals, "unstall_stale_native_work",
        lambda *a, **k: {
            "status": "skip", "recipe": "unstall_stale_native_work",
            "completion_authority": False, "reason": "quack_attach_token_absent",
        },
    )
    observation = {
        "board_id": "doep",
        "reason_codes": ["board_has_blocked_or_quarantined_tasks"],
        "details": {
            "blocked_task_ids": ["DOEP-044", "DOEP-063"],
            "task_counts": {"blocked": 2, "todo": 23},
            "extra_gate": {"live_owner_unit": "agent-supervisor-doep-v1.service"},
        },
    }
    prior = {
        "recipe": "local_validation_pending_native_admission",
        "results": [
            {"task_id": "DOEP-044", "status": "passed", "completion_authoritative": False},
            {"task_id": "DOEP-063", "status": "passed", "completion_authoritative": False},
        ],
    }
    monkeypatch.setattr(
        fleet_heals, "rearm_locally_validated_blocked_tasks",
        lambda *a, **k: {
            "status": "applied",
            "recipe": "rearm_locally_validated_blocked_tasks",
            "completion_authority": False,
            "completion_authoritative": False,
            "unstalled": [
                {"task_alias": "DOEP-044", "reason": "local_validation_pending_native_admission"},
                {"task_alias": "DOEP-063", "reason": "local_validation_pending_native_admission"},
            ],
            "results": prior["results"],
            "reason": "locally validated blocked tasks rearmed to retrying",
        },
    )
    result = apply_supervisor_heal(
        {"id": "doep", "cwd": str(tmp_path / "board")},
        {"stall_class": "blocked_without_independent_work",
         "observation": observation, "last_action_result": prior},
    )
    assert result["status"] == "applied"
    assert result["completion_authority"] is False
    assert result["recipe"] == "rearm_locally_validated_blocked_tasks"
    assert result["unstalled"][0]["task_alias"] == "DOEP-044"

    written = real_admit(
        {"id": "doep", "cwd": str(tmp_path / "board")},
        observation,
        restart_unit=lambda unit: restarts.append(unit),
        systemd_user_dir=user_dir,
    )
    assert written["status"] == "skip"
    assert written["completion_authority"] is False
    assert written["reason"] == "overlay_first_wrap_disabled_mixed_package"
    assert restarts == []
    skipped = real_admit(
        {"id": "spar", "cwd": str(tmp_path)},
        {"details": {"extra_gate": {"live_owner_unit": "ipfs-taskboard-spar-supervisor.service"}}},
        systemd_user_dir=user_dir,
    )
    assert skipped["reason"] == "retain_owner_not_rewrapped"


def test_overlay_sys_path_stays_first_after_sealed_insert(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue.overlay_sys_path import (
        OverlayFirstPath, pin_overlay_sys_path,
    )
    overlay = tmp_path / "overlay"
    sealed = tmp_path / "sealed"
    overlay.mkdir()
    sealed.mkdir()
    path = OverlayFirstPath(["/usr/lib/python3", str(sealed)], overlay=str(overlay))
    path.insert(0, str(sealed / "external" / "ipfs_accelerate"))
    path.insert(0, str(sealed))
    assert path[0] == str(overlay.resolve())
    monkeypatch.setattr("sys.path", ["/usr/lib/python3"])
    pinned = pin_overlay_sys_path(str(overlay))
    import sys
    assert sys.path[0] == pinned
    sys.path.insert(0, str(sealed))
    assert sys.path[0] == pinned


def test_rearm_locally_validated_blocked_does_not_admit_completion(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        rearm_locally_validated_blocked_tasks,
    )

    class Result:
        changed = True
        completion_authority = False

    class Source:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def rearm_blocked_task(self, task_id, receipt=None):
            assert receipt["completion_authoritative"] is False
            assert receipt["operation"] == "local_validation_pending_native_admission"
            return Result()

    monkeypatch.setattr(
        fleet_heals, "_inventory_board",
        lambda board: {"quack_endpoint": "quack:127.0.0.1:27942"},
    )
    monkeypatch.setattr(
        fleet_heals, "_owner_transport_env",
        lambda inventory: {
            "IPFS_ACCELERATE_AGENT_QUACK_TOKEN": "test-token-value",
            "IPFS_ACCELERATE_AGENT_STATE_STORE_ID": "doep-v1-r5",
        },
    )
    monkeypatch.setattr(fleet_heals, "_open_task_source", lambda endpoint, owner_id: Source())
    result = rearm_locally_validated_blocked_tasks(
        {"id": "doep", "cwd": str(tmp_path)},
        {"details": {"blocked_task_ids": ["DOEP-044", "DOEP-063"]}},
        [
            {"task_id": "DOEP-044", "status": "passed", "completion_authoritative": False},
            {"task_id": "DOEP-063", "status": "passed", "completion_authoritative": False},
        ],
    )
    assert result["status"] == "applied"
    assert result["completion_authority"] is False
    assert result["completion_authoritative"] is False
    assert {item["task_alias"] for item in result["unstalled"]} == {"DOEP-044", "DOEP-063"}


def test_build_server_accepts_sealed_extra_gate_kwargs():
    import inspect
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        QuackStateServer,
        build_server,
    )
    params = inspect.signature(build_server).parameters
    assert "repository_root" in params
    assert "allow_legacy_board_unstall" in params
    assert hasattr(QuackStateServer, "configure_database_status_before_start")
    assert hasattr(QuackStateServer, "issue_typed_client_grant_record")
    assert hasattr(QuackStateServer, "bind_database_status_scope")


def test_extra_gate_recursion_heal_unstalls_for_native_admission(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import apply_supervisor_heal

    monkeypatch.setattr(
        fleet_heals, "collapse_extra_gate_recursion",
        lambda *a, **k: {
            "status": "skip", "recipe": "collapse_extra_gate_recursion",
            "completion_authority": False, "reason": "heal_overlay_pythonpath_already_bound",
        },
    )
    monkeypatch.setattr(
        fleet_heals, "unstall_stale_native_work",
        lambda *a, **k: {
            "status": "applied", "recipe": "unstall_stale_native_work",
            "completion_authority": False,
            "unstalled": [{"task_alias": "DOEP-044", "reason": "false_terminal_blocked_supervisor_bug"}],
            "reason": "stale in_progress or false-terminal blocked tasks rearmed; native extra-gate lanes admit",
        },
    )
    result = apply_supervisor_heal(
        {"id": "doep", "cwd": str(tmp_path)},
        {
            "stall_class": "extra_gate_recursion",
            "observation": {
                "reason_codes": ["extra_gate_recursion_sealed_package"],
                "details": {"task_counts": {"blocked": 2}, "blocked_task_ids": ["DOEP-044"]},
            },
        },
    )
    assert result["status"] == "applied"
    assert result["completion_authority"] is False
    assert result["recipe"] == "unstall_stale_native_work"
    assert "native extra-gate lanes admit" in result["reason"]


def test_sawm_in_progress_binds_overlay_pythonpath_without_wrapping(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import apply_supervisor_heal

    monkeypatch.setattr(
        fleet_heals, "unstall_stale_native_work",
        lambda *a, **k: {
            "status": "skip", "recipe": "unstall_stale_native_work",
            "completion_authority": False, "reason": "quack_attach_token_absent",
        },
    )
    monkeypatch.setattr(
        fleet_heals, "collapse_extra_gate_recursion",
        lambda *a, **k: {
            "status": "applied",
            "recipe": "collapse_extra_gate_recursion",
            "completion_authority": False,
            "restarted": False,
            "reason": "one exclusive owner; overlay PYTHONPATH bound; competing extra-gate not started",
        },
    )
    result = apply_supervisor_heal(
        {"id": "sawm", "cwd": str(tmp_path)},
        {
            "stall_class": "in_progress_awaiting_effect",
            "observation": {
                "reason_codes": ["extra_gate_recursion_sealed_package"],
                "details": {
                    "task_counts": {"in_progress": 2, "todo": 23, "completed": 20},
                    "extra_gate": {"live_owner_unit": "ipfs-taskboard-sawm-supervisor.service"},
                },
            },
        },
    )
    assert result["status"] == "applied"
    assert result["completion_authority"] is False
    assert result["recipe"] == "collapse_extra_gate_recursion"
    assert result.get("restarted") is False


def test_wrap_python_execstart_injects_heal_overlay_once():
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import wrap_python_execstart

    overlay = "/overlay"
    wrapped = wrap_python_execstart(
        ["/usr/bin/python3", "-P", "state-owner.py", "state-owner"],
        overlay=overlay, source_root="/board",
    )
    assert wrapped[2].endswith("sealed_board_supervisor_launch.py")
    assert "--overlay" in wrapped and overlay in wrapped
    assert wrapped[-2:] == ["state-owner.py", "state-owner"]
    again = wrap_python_execstart(wrapped, overlay=overlay, source_root="/board")
    assert again == wrapped


def test_collapse_extra_gate_recursion_binds_live_unit_not_inventory(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_heals import (
        apply_supervisor_heal,
        collapse_extra_gate_recursion,
    )

    user_dir = tmp_path / "systemd"
    dropin_dir = user_dir / "pctdd-g9-quack-owner.service.d"
    dropin_dir.mkdir(parents=True)
    (dropin_dir / "81-supervisor-heal-overlay.conf").write_text("ExecStart=/broken\n")
    reloads = []
    result = collapse_extra_gate_recursion(
        {"id": "pctdd", "cwd": str(tmp_path / "board")},
        {
            "board_id": "pctdd",
            "details": {
                "extra_gate": {
                    "live_owner_unit": "pctdd-g9-quack-owner.service",
                    "inventory_owner_unit": "ipfs-accelerate-pctdd-g9-watchdog.service",
                    "heal_overlay": False,
                }
            },
        },
        daemon_reload=lambda: reloads.append(True),
        systemd_user_dir=user_dir,
    )
    assert result["status"] == "applied"
    assert result["completion_authority"] is False
    assert result["restarted"] is False
    assert result["live_owner_unit"] == "pctdd-g9-quack-owner.service"
    text = (dropin_dir / "80-overlay-pythonpath.conf").read_text()
    assert "PYTHONPATH=" in text
    assert not (dropin_dir / "81-supervisor-heal-overlay.conf").exists()
    assert reloads == [True]

    skipped = apply_supervisor_heal(
        {"id": "spar", "cwd": str(tmp_path)},
        {
            "stall_class": "extra_gate_recursion",
            "observation": {
                "board_id": "spar",
                "details": {"extra_gate": {"live_owner_unit": "ipfs-taskboard-spar-supervisor.service"}},
            },
        },
    )
    assert skipped["status"] == "wait"
    assert skipped["completion_authority"] is False
    assert skipped["reason"] == "retain_owner_not_rewrapped"
