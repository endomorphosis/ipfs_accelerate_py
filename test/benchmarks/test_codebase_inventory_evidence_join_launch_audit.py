"""The retained real launcher must keep native limits and the exact worker."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding.qualify_codebase_inventory_evidence_join import (
    JoinProcessAudit, _inventory_launch)


def launch():
    root = Path(__file__).resolve().parents[4]
    path = root / "artifacts/codebase_ir_terminal_bench/inventory-evidence-join-qualification-20261003-01/result.json"
    raw = path.read_bytes()
    assert len(raw) <= 2 * 1024 * 1024
    result = json.loads(raw)
    event = next(row for row in result["parent_process_audit"]["events"]
                 if row["stage"] == "complete_scan_join" and row["kind"] == "other")
    return deepcopy(event)


def check(event):
    return _inventory_launch(event["executable"], event["argv"], python="/home/barberb/.local/bin/python",
        worker=Path(__file__).resolve().parents[3] / "ipfs_datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/codebase_inventory_feature_worker.py",
        max_workspace_bytes=48 * 1024 * 1024)


def test_retained_native_prlimit_launch_is_accepted_without_removing_limits():
    event = launch()
    before = deepcopy(event)
    result = check(event)
    assert result["expected_inventory_worker"] is True
    assert result["native_limits_wrapper"] == "prlimit"
    assert result["unwrapped_argv"] == event["argv"][4:]
    assert result["max_workspace_bytes"] == 48 * 1024 * 1024
    assert event == before


@pytest.mark.parametrize("fault", (
    "bare_python", "different_limiter", "core_unlimited", "file_unlimited",
    "larger_file_bound", "missing_separator", "different_python", "different_worker", "different_libraries", "extra_flag",
))
def test_modified_native_limits_or_worker_launch_is_refused(fault):
    event = launch()
    values = event["argv"]
    if fault == "bare_python":
        event["executable"], event["argv"] = values[4], values[4:]
    elif fault == "different_limiter":
        event["executable"] = values[0] = "/usr/bin/true"
    elif fault == "core_unlimited":
        values[1] = "--core=unlimited:unlimited"
    elif fault == "file_unlimited":
        values[2] = "--fsize=unlimited:unlimited"
    elif fault == "larger_file_bound":
        values[2] = "--fsize=67108864:67108864"
    elif fault == "missing_separator":
        values[3] = "-I"
    elif fault == "different_python":
        values[4] = "/usr/bin/true"
    elif fault == "different_worker":
        values[7] = "/tmp/unreviewed_worker.py"
    elif fault == "different_libraries":
        values[8] = json.dumps(["/tmp/unreviewed_libraries"])
    else:
        values.append("--unsafe")
    with pytest.raises(AssertionError):
        check(event)


def test_event_count_overflow_fails_with_complete_retained_prefix():
    audit = JoinProcessAudit(max_events=1)
    args = ("git", ["git", "status"], "/tmp/fixture", {})
    audit.observe("subprocess.Popen", args)
    with pytest.raises(AssertionError, match="retention exceeded"):
        audit.observe("subprocess.Popen", args)
    result = audit.to_dict()
    assert result["retention_overflow"] is True and result["overflow_aborts_qualification"] is True
    assert result["retained_events"] == 1 and result["counts"]["startup:git"] == 2
    assert result["retained_serialized_event_bytes"] == len(json.dumps(result["events"],
        sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


def test_serialized_event_overflow_fails_without_silent_truncation():
    audit = JoinProcessAudit(max_event_bytes=256)
    audit.observe("subprocess.Popen", ("git", ["git", "status"], "/tmp/fixture", {}))
    with pytest.raises(AssertionError, match="retention exceeded"):
        audit.observe("subprocess.Popen", ("git", ["git", "x" * 512], "/tmp/fixture", {}))
    result = audit.to_dict()
    assert result["retention_overflow"] is True and result["retained_events"] == 1
    assert result["event_bytes_high_water"] == result["retained_serialized_event_bytes"] <= 256


def test_bad_scan_launcher_retains_raw_event_and_sticky_policy_failure():
    event = launch()
    audit = JoinProcessAudit(worker="/tmp/wrong_worker.py")
    with audit.scope("complete_scan_join"), pytest.raises(AssertionError):
        audit.observe("subprocess.Popen", (event["executable"], event["argv"], event["cwd"], {}))
    result = audit.to_dict()
    assert result["launch_policy_failure"] is True and result["launch_policy_failure_aborts_qualification"] is True
    assert result["retained_events"] == 1 and result["events"][0]["argv"] == event["argv"]
    assert "expected_inventory_worker" not in result["events"][0]


def test_lookup_non_git_launch_sets_sticky_failure_before_runner_can_mask_error():
    audit = JoinProcessAudit()
    with audit.scope("receiving_validation", only_git=True), pytest.raises(AssertionError):
        audit.observe("subprocess.Popen", ("/usr/bin/true", ["/usr/bin/true"], "/tmp/fixture", {}))
    assert audit.to_dict()["launch_policy_failure"] is True
