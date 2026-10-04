"""Gate actual retained native observations; never promote an incomplete run.

The small fixture contains selected fields copied from host qualifications 02
and 03, with original artifact hashes. It contains no synthetic positive run.
Filesystem tests serialize that projection only to exercise the loader.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding.run_supervision_docker import (
    SOURCE, TESTS, _verify_native_doctor_result, verified_native_doctor_lifecycle,
)


@pytest.fixture
def observations():
    path = Path(__file__).with_name("fixtures") / "native_doctor_gate_observations.json"
    return json.loads(path.read_text())


def test_actual_complete_run_passes_and_incomplete_retrieval_run_refuses(observations):
    complete = observations["native_doctor_03"]
    assert _verify_native_doctor_result(complete["result"], complete["refresh"])["task"] == {
        "status": "completed", "revision": 4}
    incomplete = observations["native_doctor_02"]
    assert incomplete["result"]["qualified"] is True
    assert incomplete["refresh"]["retrieval"]["status"] == "unavailable"
    with pytest.raises(ValueError, match="pinned retrieval refresh is incomplete"):
        _verify_native_doctor_result(incomplete["result"], incomplete["refresh"])


@pytest.mark.parametrize("mutation", [
    "pending_task", "wrong_revision", "failed_check", "provider_call", "unfenced_stop",
    "remaining_process", "missing_proof", "foreign_task", "stale_semantic", "stale_world",
    "retrieval_unavailable", "old_index", "changed_model", "unprojected_ducklake",
    "expanded_scope", "completion_authority", "foreign_publication",
])
def test_complete_observation_cannot_hide_a_missing_required_stage(observations, mutation):
    record = observations["native_doctor_03"]
    result, refresh = record["result"], record["refresh"]
    if mutation == "pending_task":
        result["task"]["status"] = "in_progress"
    elif mutation == "wrong_revision":
        result["task"]["revision"] = 3
    elif mutation == "failed_check":
        result["final_public_check_exit_code"] = 1
    elif mutation == "provider_call":
        result["provider_calls"] = 1
    elif mutation == "unfenced_stop":
        result["stop"]["data"]["old_tree_fenced"] = False
    elif mutation == "remaining_process":
        result["after_stop"]["process_tree"]["members"] = [{"pid": 1}]
    elif mutation == "missing_proof":
        del result["doctor_stages"]["proof"]
    elif mutation == "foreign_task":
        refresh["task_cid"] = "foreign-task"
    elif mutation in {"stale_semantic", "stale_world"}:
        field = "semantic_root_cid" if mutation == "stale_semantic" else "world_snapshot_cid"
        refresh[field] = refresh["predecessor_" + field]
        result["after_stop"]["published_context"][0][field] = refresh[field]
    elif mutation == "retrieval_unavailable":
        refresh["retrieval"]["status"] = "unavailable"
    elif mutation == "old_index":
        refresh["retrieval"]["index_id"] = refresh["retrieval"]["previous_index_id"]
    elif mutation == "changed_model":
        refresh["retrieval"]["config_id"] = "different-config"
    elif mutation == "unprojected_ducklake":
        refresh["retrieval"]["ducklake"]["status"] = "unavailable"
    elif mutation == "expanded_scope":
        refresh["source_scope"]["scope_expanded"] = True
        result["after_stop"]["published_context"][0]["source_scope"] = refresh["source_scope"]
    elif mutation == "completion_authority":
        refresh["completion_authority"] = True
    else:
        refresh["published_commit"] = result["baseline_commit"]
    with pytest.raises(ValueError):
        _verify_native_doctor_result(result, refresh)


def _write_projection(tmp_path, record):
    result = record["result"]
    row = result["after_stop"]["published_context"][0]
    artifact = tmp_path / "repository" / row["refresh_artifact"]
    artifact.parent.mkdir(parents=True)
    raw = json.dumps(record["refresh"], sort_keys=True).encode()
    artifact.write_bytes(raw)
    # This digest binds the serialized test projection. The fixture separately
    # retains the original runtime artifact digest; neither replaces the other.
    row["refresh_sha256"] = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    return path, artifact


def test_loader_checks_exact_projected_receipt_bytes(observations, tmp_path):
    result, artifact = _write_projection(tmp_path, observations["native_doctor_03"])
    assert verified_native_doctor_lifecycle(result)["qualified"] is True
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest differs"):
        verified_native_doctor_lifecycle(result)


@pytest.mark.parametrize("replacement", ["missing", "symlink", "parent_symlink", "fifo", "escape"])
def test_loader_rejects_missing_or_replaced_receipt(observations, tmp_path, replacement):
    result, artifact = _write_projection(tmp_path, observations["native_doctor_03"])
    if replacement == "missing":
        artifact.unlink()
    elif replacement == "symlink":
        original = artifact.with_suffix(".retained")
        artifact.rename(original)
        artifact.symlink_to(original)
    elif replacement == "parent_symlink":
        parent = artifact.parent
        original = parent.with_name(parent.name + "-retained")
        parent.rename(original)
        parent.symlink_to(original, target_is_directory=True)
    elif replacement == "fifo":
        artifact.unlink()
        os.mkfifo(artifact)
    else:
        data = json.loads(result.read_text())
        data["after_stop"]["published_context"][0]["refresh_artifact"] = "../foreign.json"
        result.write_text(json.dumps(data))
    with pytest.raises((OSError, ValueError)):
        verified_native_doctor_lifecycle(result)


def test_current_integration_tests_are_registered_and_exist():
    required = {
        "test/api/test_semantic_router_translation.py", "test/api/test_semantic_router_integration.py",
        "test/api/test_doctor_task_workflow.py", "test/api/test_doctor_candidate_runner.py",
        "test/api/test_terminal_doctor_dispatch.py", "test/api/test_native_doctor_interruption.py",
        "test/api/test_local_validation_python_launcher.py",
        "test/api/semantic_state/test_published_task_context.py",
        "test/api/semantic_state/test_published_retrieval.py",
        "test/integration/test_admitted_context_refresh.py",
    }
    assert required <= set(TESTS)
    assert len(TESTS) == len(set(TESTS))
    assert all((SOURCE / path).is_file() for path in TESTS)
