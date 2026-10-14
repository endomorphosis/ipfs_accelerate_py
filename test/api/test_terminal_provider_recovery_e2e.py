"""Authored local CLI -> real router -> persisted cooldown -> fresh recovery.

Only the CLI's provider events are fixtures. The adapter, receipt recovery,
stop projection, filesystem-backed admission and process execution are real.
No credential files, network services, containers or external models are used.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import measured_usage
from benchmarks.agent_supervisor.container_coding.terminal_provider_backoff import ProviderBackoff
from benchmarks.agent_supervisor.container_coding.terminal_provider_stop import provider_stop_observation
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from test.api.test_supervisor_router_execution_contract import native_router  # noqa: F401


def _authored_cli(executable: Path, dispatches: Path) -> None:
    executable.write_text("#!" + sys.executable + "\n" + f'''
import json, os, pathlib, sys
args = sys.argv[1:]
assert args.count('--output-last-message') == 1
sys.stdin.read()
cwd = pathlib.Path.cwd()
case = os.environ['AUTHORED_RECOVERY_CASE']
assert case in ('first-failure', 'fresh-recovery')
with pathlib.Path({str(dispatches)!r}).open('a') as stream:
    stream.write(json.dumps({{"case": case, "workspace": str(cwd)}}) + '\\n')
thread_id = 'authored-rate-limit-0001' if case == 'first-failure' else 'authored-recovery-0002'
session = pathlib.Path(os.environ['CODEX_HOME']) / 'sessions/2026/10/12' / ('rollout-' + thread_id + '.jsonl')
session.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
records = [{{"type": "session_meta", "payload": {{"id": thread_id, "cwd": str(cwd)}}}}]
if case == 'fresh-recovery':
    records.append({{"type": "event_msg", "payload": {{"type": "token_count", "info": {{
        "total_token_usage": {{"input_tokens": 15, "cached_input_tokens": 5,
            "cache_write_input_tokens": 0, "output_tokens": 4, "reasoning_output_tokens": 2, "total_tokens": 19}}
    }}}}}})
records.append({{"type": "event_msg", "payload": {{"type": "task_complete"}}}})
session.write_text(''.join(json.dumps(record) + '\\n' for record in records))
print(json.dumps({{"type": "thread.started", "thread_id": thread_id}}))
if case == 'first-failure':
    (cwd / 'partial-effect.txt').write_text('retained authored candidate edit')
    print(json.dumps({{"type": "error", "status": 429, "error": {{
        "type": "rate_limit_exceeded", "message": "private authored provider message",
        "retry_after_seconds": 90}}
    }}))
    raise SystemExit(1)
pathlib.Path(args[args.index('--output-last-message') + 1]).write_text('authored recovery complete')
print(json.dumps({{"type": "turn.completed", "usage": {{
    "input_tokens": 15, "cached_tokens": 5, "output_tokens": 4}}
}}))
''')


def _captured_receipt(capsys) -> dict:
    output = capsys.readouterr().out
    rows = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    receipts = [row for row in rows if row.get("schema") == "router-implementation-invocation@1"]
    assert len(receipts) == 1
    assert "private authored provider message" not in output
    assert "private authored task instruction" not in output
    return receipts[0]


@pytest.mark.parametrize("purpose", ["planning", "coding"])
def test_rate_limit_defers_fresh_work_then_admits_one_real_recovery(
        native_router, tmp_path, monkeypatch, capsys, purpose):
    failed_workspace, cache, _ = native_router
    dispatches = tmp_path / "physical-dispatches.jsonl"
    _authored_cli(Path(llm_router.shutil.which("codex")), dispatches)
    monkeypatch.setenv("AUTHORED_RECOVERY_CASE", "first-failure")
    clock = {"wall": 10000.0, "mono": 100.0}
    def open_gate():
        return ProviderBackoff(tmp_path / "provider-state", wall_clock=lambda: clock["wall"],
            monotonic=lambda: clock["mono"], jitter=lambda: 0, boot_id="authored-e2e-boot")
    def run_candidate():
        return runner.run(prompt="private authored task instruction", provider="codex_cli",
            model="gpt-6.1-sol", timeout=15, max_output_tokens=128, purpose=purpose)
    def physical_calls():
        return [json.loads(line) for line in dispatches.read_text().splitlines()]

    gate = open_gate()
    initial = gate.acquire("codex_cli")
    assert initial.allowed and not initial.recovery_probe
    try:
        with pytest.raises(llm_router.LLMRouterError):
            run_candidate()
        failed = _captured_receipt(capsys)
        failed_report = {"provider_invocations": [{"phase": purpose, **failed}]}
        stop = provider_stop_observation(failed_report)
        assert stop["state"] == "rate_limited"
        assert stop["retry_after_seconds"] == 90
        settled_failure = gate.settle(initial, stop=stop)
    finally:
        initial.close()
    assert failed["status"] == "failed" and failed["usage"]["exit_code"] == 1
    assert failed["completion_authority"] is False
    native_failure = failed["native_rollout_usage"]
    assert native_failure["task_complete_event_observed"] is True
    assert native_failure["task_complete_observed"] is True
    assert native_failure["usage_available"] is False
    assert native_failure["usage"] == {}
    assert native_failure["observed_token_count_records"] == 0
    usage = measured_usage(failed_report)
    assert usage["provider_calls"] == 1
    assert usage["usage_accounting_complete"] is False
    assert all(usage[key] is None for key in (
        "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"))
    retained_failure = tmp_path / "retained-failed-attempt.json"
    retained_failure.write_text(json.dumps(failed_report, sort_keys=True))
    original_failure = retained_failure.read_bytes()
    assert settled_failure["recovery_required"] is True
    assert settled_failure["retry_after_seconds"] == 90
    assert settled_failure["automatic_replay_allowed"] is False
    assert physical_calls() == [{"case": "first-failure", "workspace": str(failed_workspace)}]

    # A different launcher object sees the persisted stop. It has a fresh
    # attempt available but receives no permission to execute its candidate.
    fresh_gate = open_gate()
    deferred = fresh_gate.acquire("codex_cli")
    try:
        if deferred.allowed:
            run_candidate()
        assert deferred.allowed is False
        assert deferred.reason == "cooldown" and deferred.retry_after_seconds == 90
    finally:
        deferred.close()
    assert len(physical_calls()) == 1

    clock["wall"] += 89.5
    clock["mono"] += 89.5
    early = open_gate().acquire("codex_cli")
    try:
        assert not early.allowed and early.retry_after_seconds == .5
    finally:
        early.close()
    clock["wall"] += .5
    clock["mono"] += .5
    probe = fresh_gate.acquire("codex_cli")
    assert probe.allowed and probe.recovery_probe
    try:
        contender = open_gate().acquire("codex_cli")
        try:
            if contender.allowed:
                run_candidate()
            assert not contender.allowed and contender.reason == "recovery_probe_in_progress"
        finally:
            contender.close()
        assert len(physical_calls()) == 1

        # Recovery is a separately requested candidate in a fresh checkout;
        # the failed attempt and its partial edit are never reset or replayed.
        recovered_workspace = tmp_path / "fresh-recovery-workspace"
        recovered_workspace.mkdir()
        subprocess.run(["git", "init", "-q", str(recovered_workspace)], check=True)
        monkeypatch.chdir(recovered_workspace)
        monkeypatch.setenv("AUTHORED_RECOVERY_CASE", "fresh-recovery")
        output, recovered = run_candidate()
        assert _captured_receipt(capsys) == recovered
        assert output == "authored recovery complete"
        assert recovered["status"] == "provider_returned" and recovered["usage"]["exit_code"] == 0
        assert recovered["provider"] == probe.provider == "codex_cli"
        assert recovered["model"] == "gpt-6.1-sol"
        assert recovered["invocation_id"] != failed["invocation_id"]
        recovered_report = {"provider_invocations": [{"phase": purpose, **recovered}]}
        assert provider_stop_observation(recovered_report) is None
        assert measured_usage(recovered_report)["usage_accounting_complete"] is True
        assert measured_usage(recovered_report)["total_tokens"] == 19
        settled_recovery = fresh_gate.settle(probe, provider_succeeded=(
            recovered["status"] == "provider_returned" and recovered["usage"]["exit_code"] == 0))
    finally:
        probe.close()
    assert settled_recovery["recovery_required"] is False
    assert settled_recovery["blocked"] is False and settled_recovery["reason"] == "ready"
    ready = open_gate().acquire("codex_cli")
    try:
        assert ready.allowed and not ready.recovery_probe
    finally:
        ready.close()
    assert physical_calls() == [
        {"case": "first-failure", "workspace": str(failed_workspace)},
        {"case": "fresh-recovery", "workspace": str(recovered_workspace)},
    ]
    assert retained_failure.read_bytes() == original_failure
    assert (failed_workspace / "partial-effect.txt").read_text() == "retained authored candidate edit"
    assert not (recovered_workspace / "partial-effect.txt").exists()
    assert (cache.reads, cache.writes) == (0, 0)
