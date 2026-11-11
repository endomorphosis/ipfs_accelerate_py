"""Regression SOURCE for normal-package integration; not executed in this handoff.

Fixture authority is synthetic unit-test data. No fixture grants live authority.
The original authorization parser is separately owned; these tests isolate the
new metadata admission fence and verify no queue/provider effects on refusal.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as fleet
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_runtime as runtime


def _document(root, name, value):
    path = root / name
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(raw)
    path.chmod(0o400)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _metadata(tmp_path, *, extra_board=False, second_owner_board=None, malformed_projection=False):
    epoch = {"board_id": "engineering", "board_namespace": "test_repair",
             "candidate_policy_sha256": "a" * 64, "epoch_id": "unit-epoch"}
    board = {"id": "engineering", "repair_authorization_epoch": epoch}
    rows = [copy.deepcopy(board), {"id": "science"}]
    if extra_board:
        rows.append({"id": "other", "repair_authorization_epoch": {
            "board_id": "other", "board_namespace": "test_repair",
            "candidate_policy_sha256": "b" * 64, "epoch_id": "other-unit-epoch"}})
    commands = []
    for number in range(2):
        selected = copy.deepcopy(rows)
        if number == 1 and second_owner_board is not None:
            selected[0]["id"] = second_owner_board
            selected[0]["repair_authorization_epoch"]["board_id"] = second_owner_board
        anchor = _document(tmp_path, f"anchor-{number}.json", {
            "schema": "agent-supervisor/repair-initial-profile-bootstrap@1",
            # Deliberately unrelated fleet layout; never a repair-board grant.
            "original_contract": {"profile": "fleet-layout"},
            "expected_config_projection": [] if malformed_projection else {"boards": selected}})
        policy = _document(tmp_path, f"owner-{number}.json", {
            "schema": "agent-supervisor/repair-public-owner-policy@2", "adoption": anchor})
        argv = ("/usr/bin/python3", "/unused/owner.py", "--policy", policy["path"],
                "--policy-sha256", policy["sha256"])
        commands.append(runtime.OwnerCommand(argv, argv, (policy,), 1))
    config = {"boards": [board, {"id": "science"}], "repair_worker": {},
              "state_dir": str(tmp_path / "state")}
    return config, board, epoch, SimpleNamespace(namespace="test_repair"), tuple(commands)


def test_scope_accepts_exact_epoch_in_both_pinned_owner_anchors(tmp_path):
    config, board, epoch, policy, commands = _metadata(tmp_path)
    with patch.object(runtime, "require_current_authorization", return_value=SimpleNamespace(value=epoch)):
        assert runtime._candidate_board_scope(config, board, policy, commands) is None


def test_extra_explicitly_enrolled_board_does_not_remove_current_coverage(tmp_path):
    config, board, epoch, policy, commands = _metadata(tmp_path, extra_board=True)
    with patch.object(runtime, "require_current_authorization", return_value=SimpleNamespace(value=epoch)):
        assert runtime._candidate_board_scope(config, board, policy, commands) is None


def test_board_without_epoch_cannot_inherit_global_candidate_owners(tmp_path):
    config, board, _, policy, commands = _metadata(tmp_path)
    with patch.object(runtime, "require_current_authorization", return_value=None):
        assert runtime._candidate_board_scope(config, config["boards"][1], policy, commands) == "repair_candidate_board_unbound"


def test_disagreeing_validation_and_adoption_board_coverage_refuses(tmp_path):
    config, board, epoch, policy, commands = _metadata(tmp_path, second_owner_board="other")
    with patch.object(runtime, "require_current_authorization", return_value=SimpleNamespace(value=epoch)):
        assert runtime._candidate_board_scope(config, board, policy, commands) == "repair_candidate_board_unbound"


def test_live_epoch_cannot_replace_the_immutable_bootstrap_epoch(tmp_path):
    config, board, epoch, policy, commands = _metadata(tmp_path)
    changed = {**epoch, "epoch_id": "different-unit-epoch"}
    with patch.object(runtime, "require_current_authorization", return_value=SimpleNamespace(value=changed)):
        with pytest.raises(ValueError, match="live board epoch differs"):
            runtime._candidate_board_scope(config, board, policy, commands)


def test_changed_pinned_owner_policy_refuses_before_authority_is_used(tmp_path):
    config, board, epoch, policy, commands = _metadata(tmp_path)
    path = Path(commands[0].source_descriptors[0]["path"])
    path.chmod(0o600)
    path.write_text("{}\n")
    path.chmod(0o400)
    with patch.object(runtime, "require_current_authorization", return_value=SimpleNamespace(value=epoch)):
        with pytest.raises(ValueError, match="descriptor changed"):
            runtime._candidate_board_scope(config, board, policy, commands)


def test_legacy_route_does_not_acquire_a_productive_epoch_or_parse_new_owners():
    with patch.object(runtime, "require_current_authorization", side_effect=AssertionError("legacy authority touched")), \
         patch.object(runtime.candidate, "parse_candidate_policy", side_effect=AssertionError("legacy policy touched")):
        assert runtime.candidate_board_scope_preflight({"repair_worker": {}}, {"id": "legacy"}) is None
        assert fleet._candidate_scope_denial({"repair_worker": {}}, {"id": "legacy"}) is None


def test_enqueue_unbound_board_does_not_create_queue_or_attempt(tmp_path):
    board = {"id": "science"}
    config = {"boards": [board], "repair_worker": {"candidate_policy": {}}, "state_dir": str(tmp_path)}
    incident = {"board_id": "science", "observation": {"health": "failed"}}
    fail = AssertionError("unbound enqueue attempted an authority or queue effect")
    with patch.object(fleet, "_repair_limits", return_value={"science": 3}), \
         patch.object(fleet, "read_json", return_value=incident), \
         patch.object(fleet, "observation_only", return_value=False), \
         patch.object(fleet, "_recovery_denial", return_value=None), \
         patch.object(fleet, "infrastructure_policy", return_value=None), \
         patch.object(runtime, "candidate_board_scope_preflight", return_value="repair_candidate_board_unbound"), \
         patch.object(fleet, "require_current_authorization", side_effect=fail), \
         patch.object(fleet, "repair_job_directory", side_effect=fail), \
         patch.object(fleet, "lock", side_effect=fail), \
         patch.object(fleet, "write_json", side_effect=fail):
        result = fleet.enqueue(config, tmp_path / "incident.json")
    assert result == {"status": "repair_scope_wait", "board_id": "science",
                      "reason": "repair_candidate_board_unbound", "coding_attempts_incremented": False}


def test_queued_unbound_job_refuses_before_liveness_budget_or_provider(tmp_path):
    board = {"id": "science"}
    config = {"boards": [board], "repair_worker": {"candidate_policy": {}}, "state_dir": str(tmp_path)}
    path = tmp_path / "job.json"
    fail = AssertionError("unbound job attempted liveness, coding charge or provider")
    with patch.object(fleet, "_repair_limits", return_value={"science": 3}), \
         patch.object(fleet, "_repair_job_path", return_value=path), \
         patch.object(fleet, "require_current_authorization", return_value=None), \
         patch.object(fleet, "read_json", return_value={"status": "queued"}), \
         patch.object(fleet, "observation_only", return_value=False), \
         patch.object(fleet, "_recovery_denial", return_value=None), \
         patch.object(fleet, "infrastructure_policy", return_value=None), \
         patch.object(fleet, "hold_paths", return_value=[]), \
         patch.object(runtime, "candidate_board_scope_preflight", return_value="repair_candidate_board_unbound"), \
         patch.object(fleet, "command", side_effect=fail), \
         patch.object(fleet, "_check_repair_budget", side_effect=fail), \
         patch.object(fleet, "write_json", side_effect=fail), \
         patch.object(fleet.subprocess, "Popen", side_effect=fail):
        result = fleet.run_job(config, board, path)
    assert result["status"] == "repair_scope_wait"
    assert result["coding_attempts_incremented"] is False


def test_unbound_queued_board_cannot_starve_independent_admitted_job(tmp_path):
    science, engineering = {"id": "science"}, {"id": "engineering"}
    config = {"boards": [science, engineering], "repair_worker": {}, "state_dir": str(tmp_path)}
    jobs = {str(tmp_path / "science.json"): {"status": "queued", "next_attempt_at": 0},
            str(tmp_path / "engineering.json"): {"status": "queued", "next_attempt_at": 0}}
    denial = {"status": "repair_scope_wait", "board_id": "science", "reason": "repair_candidate_board_unbound"}
    with patch.object(fleet, "_repair_limits", return_value={"science": 3, "engineering": 3}), \
         patch.object(fleet, "_repair_job_path", side_effect=lambda cfg, b: tmp_path / (b["id"] + ".json")), \
         patch.object(fleet, "read_json", side_effect=lambda p: jobs[str(p)]), \
         patch.object(fleet, "observation_only", return_value=False), \
         patch.object(fleet, "_recovery_denial", return_value=None), \
         patch.object(fleet, "_candidate_scope_denial", side_effect=lambda cfg, b: denial if b["id"] == "science" else None), \
         patch.object(fleet, "hold_paths", return_value=[]), \
         patch.object(fleet, "board_is_complete", return_value=False), \
         patch.object(fleet, "board_wait_stall", return_value=None), \
         patch.object(fleet, "_check_repair_budget", return_value=None):
        assert fleet.next_job(config, 10) == (engineering, tmp_path / "engineering.json")
        status = fleet.queue_status(config, 10)
    assert status["held"] == [denial]
    assert [row["board_id"] for row in status["waiting"]] == ["engineering"]


def test_unbound_pending_candidate_cannot_starve_independent_candidate(tmp_path):
    science, engineering = {"id": "science"}, {"id": "engineering"}
    config = {"boards": [science, engineering], "repair_worker": {}, "state_dir": str(tmp_path)}
    jobs = {}
    for board, token, due in ((science, "a" * 32, 0), (engineering, "b" * 32, 1)):
        jobs[str(tmp_path / (board["id"] + ".json"))] = {
            "board_id": board["id"], "status": "validation_pending", "attempts": 1,
            "repair_unit_token": token, "candidate_next_resume_at": due,
            "candidate_runtime": {"attempt_token": token, "attempt_number": 1}}
    denial = {"status": "repair_scope_wait", "board_id": "science", "reason": "repair_candidate_board_unbound"}
    def admitted_only(cfg, board, job):
        assert board["id"] == "engineering", "unbound freeze readiness was scanned"
        return True
    with patch.object(fleet, "_repair_job_path", side_effect=lambda cfg, b: tmp_path / (b["id"] + ".json")), \
         patch.object(fleet, "read_json", side_effect=lambda p: jobs[str(p)]), \
         patch.object(fleet, "observation_only", return_value=False), \
         patch.object(fleet, "_recovery_denial", return_value=None), \
         patch.object(fleet, "hold_paths", return_value=[]), \
         patch.object(fleet, "_candidate_scope_denial", side_effect=lambda cfg, b: denial if b["id"] == "science" else None), \
         patch.object(fleet, "_candidate_resume_ready", side_effect=admitted_only):
        assert fleet.next_pending_candidate(config, 10) == (engineering, tmp_path / "engineering.json")


def test_inner_candidate_constructor_refuses_unbound_board_before_effects(tmp_path):
    board = {"id": "science"}
    config = {"boards": [board], "repair_worker": {}, "state_dir": str(tmp_path)}
    policy = SimpleNamespace(state_root=tmp_path)
    owners = SimpleNamespace(validation_owner=object(), adoption_owner=object())
    fail = AssertionError("unbound inner runtime attempted queue, process, storage or metadata")
    with patch.object(runtime.candidate, "parse_candidate_policy", return_value=policy), \
         patch.object(runtime, "parse_runtime_policy", return_value=owners), \
         patch.object(runtime, "require_current_authorization", return_value=None), \
         patch.object(runtime, "repair_job_directory", side_effect=fail), \
         patch.object(runtime, "_pin", side_effect=fail), \
         patch.object(runtime, "_flock_owner", side_effect=fail), \
         patch.object(runtime, "_unit_snapshot", side_effect=fail), \
         patch.object(fleet, "_storage_preflight", side_effect=fail), \
         patch.object(runtime.Path, "mkdir", side_effect=fail):
        with pytest.raises(ValueError, match="repair_candidate_board_unbound"):
            runtime.CandidateRuntime(config, board, tmp_path / "job.json", "a" * 32)
        with patch.object(runtime, "_read", return_value=b'{"board_id":"science"}'):
            result = runtime.run_service_candidate(config, tmp_path / "job.json", "a" * 32, mode="run")
    assert result["status"] == "candidate_pending"
    assert result["provider_started"] is False
    assert result["coding_attempts_incremented"] is False
    assert result["detail"] == "repair_candidate_board_unbound"


def test_malformed_bootstrap_projection_has_typed_refusal_before_inner_effects(tmp_path):
    config, board, epoch, policy, commands = _metadata(tmp_path, malformed_projection=True)
    with patch.object(runtime, "require_current_authorization", return_value=SimpleNamespace(value=epoch)):
        with pytest.raises(ValueError, match="closed bootstrap config projection required"):
            runtime._candidate_board_scope(config, board, policy, commands)
