"""Real shared Grok adapter timeout custody with an authored executable, no model.

The local runner uses its planning purpose to preserve the real coding Docker
identity gate. The authored executable deliberately edits its fixture checkout
and forks; this tests subprocess custody, not native tool-policy enforcement or
live coding-container qualification. Only discovery/catalog and binary location
are substituted. generate_text and the Grok provider implementation are real.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _identity(pid: int):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return int(fields[19]), int(fields[1]), fields[0]
    except (FileNotFoundError, ProcessLookupError):
        return None


def _authored_cli(path: Path, directory: Path, case: str):
    marker = directory / "owned-descendant.json"
    calls = directory / "cli-calls.jsonl"
    fork_code = f'''import json, os, time
from pathlib import Path
if os.fork():
    os._exit(0)
os.setsid()
if os.fork():
    os._exit(0)
os.chdir("/")
raw = Path(f"/proc/{{os.getpid()}}/stat").read_text()
fields = raw[raw.rfind(")") + 2:].split()
Path({str(marker)!r}).write_text(json.dumps({{"pid": os.getpid(), "birth": int(fields[19])}}))
time.sleep(120)
'''
    path.write_text(f'''#!{sys.executable}
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
args = sys.argv[1:]
assert args[args.index("--model") + 1] == "grok-4.7"
assert args[args.index("--output-format") + 1] == "json"
assert Path(args[args.index("--prompt-file") + 1]).read_text().strip()
assert "--no-subagents" in args
Path("result.py").write_text("VALUE = 1\\n")
with open({str(calls)!r}, "a") as stream:
    stream.write(json.dumps({{"provider": "authored_executable", "pid": os.getpid(),
        "edited_sha256": hashlib.sha256(Path("result.py").read_bytes()).hexdigest()}}) + "\\n")
case = {case!r}
if case != "no_child":
    child = subprocess.Popen([sys.executable, "-c", {fork_code!r} if case == "double_fork" else "import time; time.sleep(120)"],
        start_new_session=(case == "detached"), cwd="/", stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if case == "double_fork":
        child.wait(timeout=2)
        deadline = time.monotonic() + 2
        while not Path({str(marker)!r}).exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
    else:
        raw = Path(f"/proc/{{child.pid}}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        Path({str(marker)!r}).write_text(json.dumps({{"pid": child.pid, "birth": int(fields[19])}}))
time.sleep(120)
''')
    path.chmod(0o700)


def _router_launcher(path: Path, executable: Path, *, confined: bool):
    path.write_text(f'''import sys
from types import SimpleNamespace
sys.path.insert(0, {str(ROOT)!r})
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.llm_allocation import intelligence_index
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
llm_router.find_grok_cli = lambda: {str(executable)!r}
intelligence_index.discover_available_providers = lambda: ["grok_cli"]
intelligence_index.select_efficient_route = lambda **kwargs: SimpleNamespace(provider="grok_cli", model_name="grok-4.7", reasoning_effort="high", catalog_revision="authored-fixture")
def main():
    runner.run(prompt="Authored process custody fixture", provider="grok_cli", model="grok-4.7",
        purpose="planning", timeout=1, max_output_tokens=128)
    return 0
if {confined!r}:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_cli_subreaper import run_with_child_custody
    raise SystemExit(run_with_child_custody(main))
raise SystemExit(main())
''')


def _cleanup_owned_descendant(marker: Path):
    """Reap only the exact authored child, inside this isolated fixture owner."""
    if not marker.exists():
        return False
    owned = json.loads(marker.read_text())
    identity = _identity(owned["pid"])
    if identity is None:
        return False
    assert identity[0] == owned["birth"], "authored descendant identity changed"
    assert identity[1] == os.getpid(), "authored descendant was not adopted by this owner"
    if identity[2] != "Z":
        os.kill(owned["pid"], signal.SIGKILL)
    assert os.waitpid(owned["pid"], 0)[0] == owned["pid"]
    assert _identity(owned["pid"]) is None
    return True


def _run_isolated_case(directory: Path, case: str):
    from test.api.test_native_provider_failure_settlement import _native_chain
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_cli_subreaper import _enable_child_subreaper, _direct_child_pids
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_provider_custody_observation import validate_native_provider_custody_observation

    assert _enable_child_subreaper() and not _direct_child_pids()
    directory.mkdir(parents=True, exist_ok=True)
    daemon, bridge, portals = _native_chain(directory, None)
    executable = directory / "grok"
    authored_case = {"confined_detached": "detached", "confined_double_fork": "double_fork"}.get(case, case)
    _authored_cli(executable, directory, authored_case)
    _router_launcher(directory / "router_timeout.py", executable, confined=case != "detached")
    marker = directory / "owned-descendant.json"
    report = None
    try:
        result = daemon.run_once()
        attempt = daemon.get_attempt(result["attempt_id"])
        receipt = daemon.provider_invocation_recorded(attempt.attempt_id,
            idempotency_key=f"provider:{attempt.attempt_id}")
        logs = list(bridge._paths(attempt).root.rglob("*.log"))
        invocations = [json.loads(line) for path in logs for line in path.read_text().splitlines()
            if line.startswith('{"') and 'router-implementation-invocation@1' in line]
        calls = [json.loads(line) for line in (directory / "cli-calls.jsonl").read_text().splitlines()]
        assert len(calls) == len(invocations) == len(portals) == 1
        assert calls[0]["edited_sha256"] == hashlib.sha256(b"VALUE = 1\n").hexdigest()
        invocation = invocations[0]
        assert invocation["provider"] == "grok_cli" and invocation["model"] == "grok-4.7"
        assert invocation["error_type"] == "TimeoutExpired"
        assert invocation["timeout_seconds"] == 1 and invocation["purpose"] == "planning"
        assert (directory / "repository/result.py").read_text() == "VALUE = 0\n"
        assert subprocess.check_output(["git", "status", "--porcelain"],
            cwd=directory / "repository", text=True) == ""
        edited = list((directory / "worktrees").glob("*/result.py"))
        # Native exception cleanup may remove the scratch worktree. Preserve
        # the authored write witness and report that disposition separately.
        assert len(edited) <= 1
        if edited:
            assert edited[0].read_text() == "VALUE = 1\n"
        diagnostic = result["implementation_result"].get("bridge_failure_diagnostic", {})
        custody = validate_native_provider_custody_observation(
            diagnostic.get("native_provider_custody_observation"))
        assert custody is not None
        assert marker.is_file() is (authored_case != "no_child")
        child = json.loads(marker.read_text()) if marker.exists() else None
        identity = _identity(child["pid"]) if child else None
        if identity:
            assert identity[0] == child["birth"] and identity[1] == os.getpid()
        report = {
            "case": case, "attempt_status": attempt.status,
            "task_status": daemon.task_source.get(attempt.task_cid).status,
            "claim_state": daemon.coordinator.get_task_claim(attempt.claim_id).state.value,
            "callback_state": receipt.get("callback_state"),
            "native_exit_present": isinstance(receipt.get("native_exit"), dict),
            "descendant_present_before_fixture_cleanup": identity is not None,
            "descendant_state_before_fixture_cleanup": identity[2] if identity else None,
            "partial_edit_verified_before_timeout": True, "target_repository_unchanged": True,
            "scratch_worktree_disposition": "retained" if edited else "removed",
            "native_custody_observation": custody,
            "actual_shared_grok_adapter": True, "router_purpose": invocation["purpose"],
            "dedicated_worker_custody_enabled": case != "detached",
            "router_error_type": invocation["error_type"], "paid_model_calls": 0,
            "shared_grok_adapter_calls": 1,
            "authored_cli_invocations": len(calls),
        }
        if receipt.get("native_exit"):
            report["native_receipt_automatic_retry_admitted"] = receipt["native_exit"]["automatic_retry_admitted"]
            report["native_receipt_completion_authority"] = receipt["native_exit"]["completion_authority"]
        _cleanup_owned_descendant(marker)
        if receipt.get("callback_state") == "started_outcome_unknown":
            resumed = daemon._resume_attempt_without_process_crash(attempt)
            assert resumed["reason"] == "provider_callback_outcome_unknown"
            assert daemon.provider_invocation_recorded(attempt.attempt_id,
                idempotency_key=f"provider:{attempt.attempt_id}") == receipt
            assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
            assert len((directory / "cli-calls.jsonl").read_text().splitlines()) == 1
            report["later_process_cleanup_did_not_settle_or_retry"] = True
        else:
            assert daemon.claim_next() is None
        assert not _direct_child_pids()
        report["fixture_owned_children_absent_after_cleanup"] = True
    finally:
        _cleanup_owned_descendant(marker)
        daemon.close()
    assert report is not None
    (directory / "closed-result.json").write_text(json.dumps(report, sort_keys=True))


@pytest.fixture
def isolated_grok_case(tmp_path):
    def run(case):
        directory = tmp_path / case
        wrapper = ROOT / "ipfs_accelerate_py/agent_supervisor/todo_daemon/native_cli_subreaper.py"
        # The existing outer wrapper owns only this fixture process tree. If
        # the fixture fails, it closes adopted children before returning.
        process = subprocess.Popen([sys.executable, "-I", str(wrapper), "--",
            sys.executable, "-B", str(Path(__file__).resolve()), "--isolated-case", str(directory), case],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        try:
            output, _ = process.communicate(timeout=75)
        except subprocess.TimeoutExpired:
            # Signal the known wrapper only; its verified subreaper owns
            # descendant termination/reaping, including detached sessions.
            process.terminate()
            output, _ = process.communicate(timeout=15)
            pytest.fail("isolated authored Grok custody fixture timed out")
        assert process.returncode == 0, output[-6000:]
        report = json.loads((directory / "closed-result.json").read_text())
        print("CLOSED_GROK_CUSTODY_RESULT " + json.dumps(report, sort_keys=True))
        return report
    return run


@pytest.mark.parametrize("case", ["no_child", "same_group", "confined_detached", "confined_double_fork"])
def test_real_grok_timeout_with_partial_edit_settles_after_owned_cleanup(isolated_grok_case, case):
    report = isolated_grok_case(case)
    assert report["callback_state"] == "failed_outcome_settled", report
    assert report["attempt_status"] == "failed" and report["task_status"] == "blocked"
    assert report["claim_state"] == "released"
    assert report["native_exit_present"]
    assert report["native_receipt_automatic_retry_admitted"] is False
    assert report["native_receipt_completion_authority"] is False
    assert report["descendant_present_before_fixture_cleanup"] is False
    assert len(report["native_custody_observation"]["checks"]) == 4
    assert all(check["status"] == "passed" for check in report["native_custody_observation"]["checks"])


def test_real_grok_detached_survivor_refuses_settlement_after_partial_edit(isolated_grok_case):
    report = isolated_grok_case("detached")
    assert report["descendant_present_before_fixture_cleanup"] is True, report
    assert report["descendant_state_before_fixture_cleanup"] != "Z"
    assert report["callback_state"] == "started_outcome_unknown"
    assert report["attempt_status"] == "running" and report["task_status"] == "in_progress"
    assert report["claim_state"] == "accepted" and report["native_exit_present"] is False
    assert report["later_process_cleanup_did_not_settle_or_retry"] is True
    assert {"stage": "native_exit", "status": "denied",
            "reason_code": "child_census_not_empty_or_unstable"} in report["native_custody_observation"]["checks"]


if __name__ == "__main__":
    assert len(sys.argv) == 4 and sys.argv[1] == "--isolated-case"
    assert sys.argv[3] in {"no_child", "same_group", "detached", "confined_detached", "confined_double_fork"}
    sys.path.insert(0, str(ROOT))
    _run_isolated_case(Path(sys.argv[2]), sys.argv[3])
