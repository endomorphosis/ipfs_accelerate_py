"""Generic fleet repair denials cannot mutate a retained native owner or attempt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals as heals


POLICY = {"schema": "agent-supervisor/owner-retention@1", "mode": "retain",
          "profile_id": "private-native-profile", "reason": "original native owner must remain"}


def _case(tmp_path, mode):
    inventory = tmp_path / "inventory.json"
    row = {"id": "private-owner", "cwd": str(tmp_path),
           "database_path": str(tmp_path / "never-open.duckdb")}
    inventory.write_text(json.dumps({"boards": [row]}))
    board = {"id": "private-owner", "cwd": str(tmp_path), "probe": {"argv": [
        "private-probe", "--inventory", str(inventory), "--board", "private-owner"]}}
    observation = {"board_id": "private-owner", "health": "blocked", "busy": False,
        "details": {"owner_ready": True, "authenticated_task_observation": True,
                    "task_counts": {"todo": 3, "retrying": 2},
                    "blocked_task_ids": ["T-1"], "selection_idle_reason": "no_ready_tasks",
                    "lanes": [{"supervisor": None, "daemon": None}]}}
    if mode == "infrastructure":
        board["max_repair_attempts"] = 3
        board["recovery_policy"] = {"profile": "infrastructure-only/v1",
            "repairable_stalls": ["probe_failed"], "max_actions_per_window": 3,
            "window_seconds": 86400, "circuit_breaker_seconds": 3600}
        reason = "infrastructure_only_recovery"
    elif mode == "policy":
        board["recovery_policy"] = dict(POLICY)
        reason = "native_owner_retention_required"
    elif mode == "invalid":
        board["recovery_policy"] = {**POLICY, "mode": "restart"}
        reason = "invalid_owner_retention_policy"
    else:
        observation["details"]["selection_idle_reason"] = "expired_attempt_settlement_unavailable"
        reason = "retained_attempt_settlement_wait"
    return board, observation, reason


def _forbid_effects(monkeypatch, board):
    # The native entrypoint first verifies its explicit private binding. Exercise
    # that real read before asserting the independent retention-policy denial.
    original_inventory_board = heals._inventory_board
    expected = {"id": board["id"], "cwd": board["cwd"],
                "database_path": str(Path(board["cwd"]) / "never-open.duckdb")}

    def private_inventory_read(selected):
        assert selected is board
        assert selected["probe"]["argv"] == ["private-probe", "--inventory",
            str(Path(board["cwd"]) / "inventory.json"), "--board", board["id"]]
        found = original_inventory_board(selected)
        assert found == expected
        return found

    monkeypatch.setattr(heals, "_inventory_board", private_inventory_read)
    def forbidden(*args, **kwargs):
        pytest.fail("retention denial must precede native, process and source effects")

    for name in ("restore_native_owner_execstart", "_systemd_daemon_reload",
                 "_systemd_restart_unit", "_mirror_native_owner_overlay"):
        monkeypatch.setattr(heals, name, forbidden)
    monkeypatch.setattr(heals.subprocess, "run", forbidden)
    monkeypatch.setattr(heals.subprocess, "Popen", forbidden)
    monkeypatch.setattr(heals.os, "kill", forbidden)
    monkeypatch.setattr(heals.os, "killpg", forbidden)
    monkeypatch.setattr(heals.shutil, "copy2", forbidden)
    for name in ("home", "mkdir", "unlink", "write_text", "write_bytes"):
        monkeypatch.setattr(Path, name, forbidden)


def _invoke(name, board, observation):
    function = getattr(heals, name)
    state = {"observation": observation, "stall_class": "blocked_without_independent_work"}
    if name == "apply_supervisor_heal":
        return function(board, state)
    if name in {"restart_native_owner_and_attach", "relaunch_native_lanes_on_ready_owner",
                "run_board_dump_stop_repair_import_start"}:
        return function(board, observation, state)
    if name in {"run_board_claim_verification_recover", "run_board_local_repair"}:
        return function(board, observation, ["T-1"], [{"task_id": "T-1", "status": "passed"}])
    if name == "rearm_locally_validated_blocked_tasks":
        return function(board, observation, [{"task_id": "T-1", "status": "passed"}])
    return function(board, observation)


@pytest.mark.parametrize("mode", ["policy", "invalid", "settlement", "infrastructure"])
@pytest.mark.parametrize("entrypoint", [
    "apply_supervisor_heal", "run_board_dump_stop_repair_import_start",
    "restart_native_owner_and_attach", "relaunch_native_lanes_on_ready_owner",
    "collapse_extra_gate_recursion", "bind_schema_serve_in_place", "admit_native_owner_overlay",
    "restore_dirty_control_plane", "run_board_claim_verification_recover", "run_board_local_repair",
    "attach_native_lanes", "unstall_stale_native_work", "rearm_locally_validated_blocked_tasks",
    "provisionally_complete_disabled_extra_gate_goals",
])
def test_direct_native_repair_denies_before_all_effects(tmp_path, monkeypatch, mode, entrypoint):
    board, observation, reason = _case(tmp_path, mode)
    _forbid_effects(monkeypatch, board)
    result = _invoke(entrypoint, board, observation)
    assert result["status"] in {"skip", "wait"}
    assert result["reason"] == reason
    assert result["completion_authority"] is False


@pytest.mark.parametrize("mode", ["policy", "invalid", "infrastructure"])
@pytest.mark.parametrize("entrypoint", ["dump_board_before_owner_stop",
                                        "clear_overlay_copies_blocking_owner_start"])
def test_board_only_owner_preparation_denies_without_reading_or_writing(
        tmp_path, monkeypatch, mode, entrypoint):
    board, _, reason = _case(tmp_path, mode)
    _forbid_effects(monkeypatch, board)
    result = getattr(heals, entrypoint)(board)
    assert result["status"] == "skip" and result["reason"] == reason
    assert result["completion_authority"] is False


@pytest.mark.parametrize("mode", ["policy", "invalid", "settlement", "infrastructure"])
def test_retained_owner_cannot_select_unit_or_dump_stop_recipe(tmp_path, monkeypatch, mode):
    board, observation, _ = _case(tmp_path, mode)
    observation["details"]["extra_gate"] = {"live_owner_unit": "foreign.service"}
    _forbid_effects(monkeypatch, board)
    assert heals._exclusive_owner_unit(board, observation) == ""
    assert heals.should_run_dump_stop_repair_import_start(
        board, observation, {"last_action_result": {"recipe": "unstall_stale_native_work"}},
        "blocked_without_independent_work") is False


@pytest.mark.parametrize("facts", [
    {}, {"owner_ready": False}, {"authenticated_task_observation": False},
    {"observed_at": "1970-01-01T00:00:00+00:00"},
    {"owner_ready": True, "authenticated_task_observation": True, "ready_count": 3,
     "eligible_ready_count": 3, "recovery_policy": {"mode": "restart"}},
])
def test_probe_readiness_or_staleness_cannot_override_operator_retention(tmp_path, monkeypatch, facts):
    board, observation, reason = _case(tmp_path, "policy")
    observation["details"] = facts
    _forbid_effects(monkeypatch, board)
    result = heals.apply_supervisor_heal(board, {"observation": observation,
                                               "stall_class": "owner_missing"})
    assert result["status"] == "wait" and result["reason"] == reason


@pytest.mark.parametrize("marker", [
    {"selection_idle_reason": "expired_attempt_settlement_unavailable"},
    {"lanes": [{"summary": {"reason": "production_attempt_settlement_required"}}]},
    {"custody_observation": {"schema": "ipfs_accelerate_py/attempt-custody-observation@1",
                             "callback_outcome": "unknown"}},
])
def test_unknown_attempt_diagnosis_has_no_generic_mutation_targets(marker):
    observation = {"details": {"task_counts": {"retrying": 3, "in_progress": 1},
        "blocked_task_ids": ["T-1"], "selection_idle_reason": "no_ready_tasks",
        "authenticated_task_observation": {"retrying_task_ids": ["T-2"]},
        "lanes": [{"stalled_without_active_worker": True}], **marker}}
    result = heals.diagnose_board_repair_problems(observation)
    assert result["recovery_denial"] == "retained_attempt_settlement_wait"
    assert result["rearm_retrying"] is result["unstall_all_in_progress"] is False
    assert result["leftover_retrying"] is False and result["problems"] == []
    assert result["blocked_aliases"] == result["in_progress_aliases"] == result["retrying_aliases"] == []


def test_unmarked_legacy_diagnosis_is_unchanged():
    result = heals.diagnose_board_repair_problems({"details": {
        "task_counts": {"retrying": 2}, "selection_idle_reason": "no_ready_tasks"}})
    assert result["rearm_retrying"] is True
    assert "retrying_without_typed_cooldown" in result["problems"]
    assert "recovery_denial" not in result


def test_ordinary_private_inventory_and_noop_keep_original_behavior(tmp_path, monkeypatch):
    inventory = tmp_path / "inventory.json"
    row = {"id": "private-owner", "cwd": str(tmp_path),
           "database_path": str(tmp_path / "never-open.duckdb"),
           "existing_service": "private-owner.service"}
    inventory.write_text(json.dumps({"boards": [row]}))
    board = {"id": "private-owner", "cwd": str(tmp_path), "probe": {"argv": [
        "private-probe", "--inventory", str(inventory), "--board", "private-owner"]}}
    monkeypatch.setattr(Path, "home", lambda: pytest.fail("no implicit home inventory"))
    monkeypatch.setattr(heals.subprocess, "run", lambda *a, **k: pytest.fail("no host commands"))
    assert heals._exclusive_owner_unit(board, {}) == "private-owner.service"
    assert heals.restore_dirty_control_plane(board, {}) == {"status": "skip"}
    assert not (tmp_path / "never-open.duckdb").exists()


def test_original_explicit_inventory_guard_remains_before_heal_effects(tmp_path, monkeypatch):
    board = {"id": "private-owner", "cwd": str(tmp_path), "probe": {"argv": ["private-probe"]}}
    monkeypatch.setattr(Path, "home", lambda: pytest.fail("unbound board cannot read home inventory"))
    monkeypatch.setattr(heals.subprocess, "run", lambda *a, **k: pytest.fail("no host commands"))
    result = heals.apply_supervisor_heal(board, {"observation": {}, "stall_class": "owner_missing"})
    assert result["status"] == "skip" and result["reason"] == "explicit_inventory_required"


def test_missing_runtime_assets_preserve_existing_dropins(tmp_path, monkeypatch):
    inventory = tmp_path / "inventory.json"
    row = {"id": "private-owner", "cwd": str(tmp_path),
           "database_path": str(tmp_path / "never-open.duckdb"),
           "existing_service": "private-owner.service"}
    inventory.write_text(json.dumps({"boards": [row]}))
    board = {"id": "private-owner", "cwd": str(tmp_path), "probe": {"argv": [
        "private-probe", "--inventory", str(inventory), "--board", "private-owner"]}}
    user = tmp_path / "units"
    directory = user / "private-owner.service.d"
    directory.mkdir(parents=True)
    paths = [directory / "81-schema-serve-in-place.conf", directory / "90-existing-runtime.conf"]
    for path in paths:
        path.write_text("retained original " + path.name)
    before = {path: path.read_bytes() for path in paths}
    monkeypatch.setattr(heals, "supervisor_overlay_root", lambda: str(tmp_path / "missing-runtime"))
    monkeypatch.setattr(heals.subprocess, "run", lambda *a, **k: pytest.fail("no host commands"))
    monkeypatch.setattr(Path, "home", lambda: pytest.fail("explicit private unit directory only"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: pytest.fail("missing assets cannot create drop-in directories"))
    result = heals.bind_schema_serve_in_place(
        board, {}, systemd_user_dir=user,
        daemon_reload=lambda: pytest.fail("missing runtime cannot reload systemd"))
    assert result["status"] == "skip" and result["missing_assets"]
    assert all(path.read_bytes() == raw for path, raw in before.items())
