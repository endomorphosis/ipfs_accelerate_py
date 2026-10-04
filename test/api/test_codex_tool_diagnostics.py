"""Native rollout diagnostics expose bounded counts, never tool contents."""
import json
import os
from pathlib import Path
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.codex_usage_receipt import recover_codex_usage

THREAD = "diagnostic-thread-123456"


def rollout(tmp_path, rows, *, completed=True):
    home = tmp_path / "codex"
    folder = home / "sessions/2026/09/29"
    folder.mkdir(parents=True, exist_ok=True)
    workspace = tmp_path / "work"
    workspace.mkdir(exist_ok=True)
    path = folder / ("rollout-" + THREAD + ".jsonl")
    records = [{"type": "session_meta", "payload": {"id": THREAD, "cwd": str(workspace)}}]
    records += [{"type": "response_item", "payload": row} for row in rows]
    if completed:
        records += [{"type": "event_msg", "payload": {"type": "task_complete"}}]
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    return home, workspace, path


def recover(home, workspace):
    return recover_codex_usage(home=home, thread_id=THREAD, workspace=workspace)


def test_counts_bind_exact_thread_and_workspace_and_never_export_contents(tmp_path):
    marker = "PRIVATE_SOURCE_AND_CREDENTIAL_MUST_NEVER_APPEAR"
    home, workspace, path = rollout(tmp_path, [
        {"type": "function_call", "arguments": json.dumps({"cmd": "read /app/bottle.py " + marker})},
        {"type": "function_call_output", "output": "Permission denied: /app/bottle.py " + marker},
        {"type": "custom_tool_call", "input": "await tools.exec_command({cmd: 'compile relative.py'})"},
        {"type": "custom_tool_call_output", "output": "bwrap: sandbox setup failed: Operation not permitted"},
        {"type": "function_call", "arguments": {"command": "relative.py"}},
        {"type": "function_call_output", "output": {"content": [{"text": "Read-only file system; TypeError: tools.write is not a function"}]}},
        # Assistant prose is not a tool result and must not inflate counts.
        {"type": "message", "role": "assistant", "content": "Permission denied " + marker},
    ])
    result = recover(home, workspace)
    assert result["usage"] == {} and result["usage_available"] is False
    d = result["tool_diagnostics"]
    assert d["observed_tool_call_events"] == d["observed_tool_output_events"] == 3
    assert d["absolute_canonical_app_reference_observed"] is True
    assert d["observed_calls_with_absolute_canonical_app_reference"] == 1
    assert d["complete_output_indicator_counts"] == {
        "permission_denied": 1, "read_only_filesystem": 1,
        "operation_not_permitted": 1, "sandbox_failure": 1, "tool_protocol_error": 1,
        "tool_host_unavailable": 0}
    assert d["root_cause_established"] is False and d["raw_tool_data_exported"] is False
    assert marker not in json.dumps(result) and "bottle.py" not in json.dumps(result)
    assert recover_codex_usage(home=home, thread_id=THREAD, workspace=tmp_path) is None
    assert recover_codex_usage(home=home, thread_id="other-thread-123456", workspace=workspace) is None
    # A second matching rollout is ambiguous; do not silently choose one.
    (path.parent / ("duplicate-" + THREAD + ".jsonl")).write_bytes(path.read_bytes())
    assert recover(home, workspace) is None


@pytest.mark.parametrize("argument,expected", [
    ("cat /app/bottle.py", True), (json.dumps({"cwd": "/app"}), True),
    ('{"cmd":"cat \\/app\\/bottle.py"}', True),
    ("cat /apple/bottle.py", False), ("cat relative/app/bottle.py", False),
    ("cat /opt/app/bottle.py", False),
])
def test_absolute_app_indicator_does_not_claim_write_or_prefix_match(tmp_path, argument, expected):
    home, workspace, _ = rollout(tmp_path, [
        {"type": "custom_tool_call", "input": argument},
        {"type": "custom_tool_call_output", "output": "success"}])
    d = recover(home, workspace)["tool_diagnostics"]
    assert d["absolute_canonical_app_reference_observed"] is expected
    assert d["root_cause_established"] is False


def test_incomplete_unrecognized_missing_outputs_stay_unknown(tmp_path):
    home, workspace, path = rollout(tmp_path, [
        {"type": "custom_tool_call", "input": None},
        {"type": "custom_tool_call_output", "output": None},
    ], completed=False)
    path.write_text(path.read_text() + "{partial")
    d = recover(home, workspace)["tool_diagnostics"]
    assert d["observed_tool_call_events"] == d["observed_tool_output_events"] == 1
    assert d["unclassified_tool_calls"] == d["unclassified_tool_outputs"] == 1
    assert d["malformed_json_records"] == 1
    assert d["complete_output_indicator_counts"] is None
    assert d["absolute_canonical_app_reference_observed"] is None
    assert d["tool_event_counts_complete"] is False
    # Even a task_complete marker cannot invent an unrecorded tool response.
    home, workspace, _ = rollout(tmp_path, [{"type": "custom_tool_call", "input": "true"}])
    assert recover(home, workspace)["tool_diagnostics"]["complete_output_indicator_counts"] is None


def test_fifo_symlink_oversized_and_foreign_metadata_are_unavailable(tmp_path):
    home, workspace, path = rollout(tmp_path, [])
    saved = path.read_bytes()
    path.unlink()
    os.mkfifo(path)
    started = time.monotonic()
    assert recover(home, workspace) is None
    assert time.monotonic() - started < 1
    path.unlink()
    target = tmp_path / "outside.jsonl"
    target.write_bytes(saved)
    path.symlink_to(target)
    assert recover(home, workspace) is None
    path.unlink()
    with path.open("wb") as stream:
        stream.truncate(32_000_001)
    assert recover(home, workspace) is None
    path.write_text(json.dumps({"type": "session_meta", "payload": {"id": "foreign", "cwd": str(workspace)}}))
    assert recover(home, workspace) is None


def test_valid_complete_session_without_tools_reports_observed_zero(tmp_path):
    home, workspace, _ = rollout(tmp_path, [])
    d = recover(home, workspace)["tool_diagnostics"]
    assert d["observed_tool_call_events"] == d["observed_tool_output_events"] == 0
    assert d["complete_output_indicator_counts"] == dict.fromkeys(d["observed_output_indicator_counts"], 0)
    assert d["absolute_canonical_app_reference_observed"] is False


def test_missing_native_code_mode_host_is_distinct_from_permission_or_path_errors(tmp_path):
    error = ("failed to spawn code-mode host "
        "/opt/ipfs-supervisor/provider-bin/codex-code-mode-host: "
        "No such file or directory (os error 2)")
    home, workspace, _ = rollout(tmp_path, [
        {"type": "custom_tool_call", "input": "await tools.exec_command({cmd: 'true'})"},
        {"type": "custom_tool_call_output", "output": error},
        {"type": "custom_tool_call", "input": "await tools.exec_command({cmd: 'pwd'})"},
        {"type": "custom_tool_call_output", "output": error},
    ])
    result = recover(home, workspace)
    diagnostics = result["tool_diagnostics"]
    indicators = diagnostics["complete_output_indicator_counts"]
    assert indicators["tool_host_unavailable"] == 2
    assert all(value == 0 for name, value in indicators.items() if name != "tool_host_unavailable")
    assert diagnostics["absolute_canonical_app_reference_observed"] is False
    assert diagnostics["root_cause_established"] is False
    assert error not in json.dumps(result) and "/opt/ipfs-supervisor" not in json.dumps(result)
