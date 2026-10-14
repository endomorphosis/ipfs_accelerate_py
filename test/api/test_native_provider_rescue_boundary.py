"""An actual first child cannot authorize an unbound database repair child."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from test.api.test_native_provider_failure_settlement import _native_chain, _assert_failed_checkpoint
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    _accepted_source_events,
)


@pytest.mark.parametrize("pooled", [False, True])
def test_failed_validation_does_not_spawn_unbound_database_repair(tmp_path, monkeypatch, pooled):
    database, bridge, portals = _native_chain(tmp_path, monkeypatch, pooled=pooled)
    dispatches = tmp_path / "authored-dispatches.jsonl"
    # The native provider child succeeds as a process but leaves invalid Python.
    # A second invocation would repair it; the old inline path must not obtain
    # that effect without fresh owner round admission and child custody.
    (tmp_path / "router_timeout.py").write_text(
        "import json\nfrom pathlib import Path\n"
        f"dispatches = Path({str(dispatches)!r})\n"
        "prior = dispatches.read_text().splitlines() if dispatches.exists() else []\n"
        "with dispatches.open('a') as stream:\n"
        "    stream.write(json.dumps({'workspace': str(Path.cwd()), 'round': len(prior)}) + '\\n')\n"
        "Path('result.py').write_text('VALUE = (\\n' if not prior else 'VALUE = 1\\n')\n"
    )
    try:
        summary = database.run_once()
        attempt = database.get_attempt(summary["attempt_id"])
        events, _ = _accepted_source_events(bridge._paths(attempt).events)
        finished = [row for row in events if row.get("type") == "implementation_finished"]
        assert len(finished) == 1
        result = finished[0]
        assert result["provider_dispatched"] is True
        assert result["returncode"] != 0
        assert result["validation_result"]["passed"] is False
        assert result["board_completion"]["complete"] is False
        assert result["merge_result"]["merged"] is False
        assert result["validation_result"]["auto_rescue"]["provider_passes"] == 0
        assert result["validation_result"]["auto_rescue"]["provider_rescue_blocked_reason"] == (
            "database_provider_rescue_requires_owner_dispatch")
        assert any(row.get("type") == "implementation_auto_rescue_provider_blocked"
                   and row.get("reason") == "database_provider_rescue_requires_owner_dispatch"
                   for row in events)
        assert not any(row.get("type") == "implementation_auto_rescue_provider_started" for row in events)
        calls = [json.loads(row) for row in dispatches.read_text().splitlines()]
        assert len(calls) == 1 and calls[0]["round"] == 0
        assert Path(calls[0]["workspace"]) != tmp_path / "repository"
        assert (tmp_path / "repository/result.py").read_text() == "VALUE = 0\n"
        assert len(portals) == 1
        # This test does not grant failure settlement or retry authority to a
        # successful first child whose later validation failed.
        assert database.task_source.get(attempt.task_cid).status != "completed"
    finally:
        database.close()


@pytest.mark.parametrize("pooled", [False, True])
def test_native_grok_turn_exhaustion_preserves_edits_and_usage_without_completion(tmp_path, monkeypatch, pooled):
    """Real router/Git settlement; authored CLI and container boundary, no Docker."""
    database, bridge, portals = _native_chain(tmp_path, monkeypatch, pooled=pooled)
    executable = tmp_path / "grok"
    dispatches = tmp_path / "grok-dispatches.jsonl"
    executable.write_text(f"#!{sys.executable}\n" +
        "import json,sys\nfrom pathlib import Path\n"
        "argv = sys.argv[1:]\n"
        "assert argv.count('--max-turns') == 1\n"
        "assert argv[argv.index('--max-turns')+1] == '16'\n"
        "assert argv[argv.index('--output-format')+1] == 'streaming-json'\n"
        f"with Path({str(dispatches)!r}).open('a') as stream:\n"
        "    stream.write(json.dumps({'max_turns': 16, 'workspace': str(Path.cwd())})+'\\n')\n"
        "Path('result.py').write_text('VALUE = 1\\n')\n"
        "print(json.dumps({'type':'text','data':'Unfinished candidate'}), flush=True)\n"
        "print(json.dumps({'type':'end','stopReason':'max_turn_requests','num_turns':16,"
        "'usage':{'input_tokens':120,'output_tokens':40}}), flush=True)\n")
    executable.chmod(0o700)
    checkout = Path(__file__).resolve().parents[2]
    (tmp_path / "router_timeout.py").write_text(f'''import io,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, {str(checkout)!r})
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.llm_allocation import intelligence_index,duckdb_store
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner,container_worker_boundary
intelligence_index.discover_available_providers=lambda: ['grok_cli']
intelligence_index.select_efficient_route=lambda **kwargs: SimpleNamespace(provider='grok_cli',model_name='grok-4.7',reasoning_effort='high',catalog_revision='authored-turn-bound')
duckdb_store._DEFAULT_STORE=duckdb_store.AllocationStore(Path({str(tmp_path / 'allocation.duckdb')!r}))
llm_router._grok_cli_command=lambda: {str(executable)!r}
llm_router._cli_available=lambda command: True
provider=llm_router._get_grok_cli_provider()
llm_router.get_llm_provider=lambda *args,**kwargs: provider
# The fixture qualifies process/outcome/retention handling, not container isolation.
container_worker_boundary.verify_container_worker_boundary=lambda **kwargs: {{'fixture':True}}
sys.argv=['authored-router','--provider','grok_cli','--model','grok-4.7','--timeout','10','--max-output-tokens','128','--grok-coding-max-turns','16','--container-boundary','/authored/boundary','--container-boundary-sha256','{'a' * 64}']
raise SystemExit(runner.main())
''')
    try:
        summary = database.run_once()
        assert summary["implementation_result"].get("portal_terminal_failure") is True, summary
        attempt = database.get_attempt(summary["attempt_id"])
        assert attempt.status == "failed"
        assert database.task_source.get(attempt.task_cid).status == "blocked"
        assert database.coordinator.get_task_claim(attempt.claim_id).state.value == "released"
        receipt = database.provider_invocation_recorded(attempt.attempt_id,
            idempotency_key=f"provider:{attempt.attempt_id}")
        assert receipt["callback_state"] == "failed_outcome_settled"
        assert receipt["native_exit"]["automatic_retry_admitted"] is False
        assert receipt["native_exit"]["subreaper_children_absent"] is True
        calls = [json.loads(row) for row in dispatches.read_text().splitlines()]
        assert len(calls) == 1 and calls[0]["max_turns"] == 16
        events, _ = _accepted_source_events(bridge._paths(attempt).events)
        finished = [row for row in events if row.get("type") == "implementation_finished"]
        assert len(finished) == len(portals) == 1
        _assert_failed_checkpoint(bridge, attempt, portals[0], finished[0], events)
        logs = list(bridge._paths(attempt).root.rglob("*.log"))
        invocations = [json.loads(line) for log in logs for line in log.read_text().splitlines()
            if line.startswith('{"') and 'router-implementation-invocation@1' in line]
        assert len(invocations) == 1
        invocation = invocations[0]
        assert invocation["status"] == "failed"
        assert invocation["failure_phase"] == "provider_invocation"
        assert invocation["provider_invocation_policy"]["max_turns"] == 16
        assert invocation["usage"]["exit_code"] == 0
        assert invocation["native_provider_outcome"]["reason_code"] == "max_turns"
        assert invocation["native_rollout_usage"]["usage"] == {"input_tokens": 120, "output_tokens": 40}
        assert invocation["native_rollout_usage"]["billing_total_verified"] is False
        assert invocation["completion_authority"] is False
        assert (tmp_path / "repository/result.py").read_text() == "VALUE = 0\n"
        assert database.claim_next() is None
    finally:
        database.close()
