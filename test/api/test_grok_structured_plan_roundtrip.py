"""Authored native-process fixture through router and strict plan admission.

This exercises actual subprocess transport, not a live Grok model or proof of
task correctness. Schema-valid output remains subject to source/evidence checks.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test.api.test_agent_supervisor_prompt_goal_planner import _proposal, _request, _scan
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
    PromptGoalProposalError, build_prompt_goal_provider_request, parse_prompt_goal_graph,
)


@pytest.mark.parametrize("case", ["valid", "foreign_output", "foreign_evidence"])
def test_native_structured_result_reaches_independent_plan_checks(tmp_path, monkeypatch, case):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    from ipfs_accelerate_py.llm_allocation import duckdb_store, intelligence_index

    request = _request()
    scan = _scan(request)
    prompt = build_prompt_goal_provider_request(request, scan)
    proposal = _proposal(scan)
    if case == "foreign_output":
        proposal["tasks"][0]["outputs"][0]["path"] = "outside/unauthorized.py"
    elif case == "foreign_evidence":
        proposal["tasks"][0]["evidence_cids"] = [request.policy_root]

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    monkeypatch.chdir(workspace)
    capture = tmp_path / "native-observation.json"
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(proposal))
    executable = tmp_path / "grok"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "argv = sys.argv[1:]\n"
        "def option(name):\n"
        "    assert argv.count(name) == 1, name\n"
        "    return argv[argv.index(name) + 1]\n"
        "prompt_path = pathlib.Path(option('--prompt-file'))\n"
        "prompt = json.loads(prompt_path.read_text())\n"
        "schema = json.loads(option('--json-schema'))\n"
        "assert schema == prompt['response_schema']\n"
        "assert option('--tools') == 'read_file'\n"
        "assert option('--disallowed-tools') == 'read_file,search_tool,use_tool'\n"
        "assert option('--max-turns') == '2'\n"
        "assert option('--permission-mode') == 'dontAsk'\n"
        f"pathlib.Path({str(capture)!r}).write_text(json.dumps({{'prompt_file':str(prompt_path), 'schema_matched':True}}))\n"
        f"proposal = json.loads(pathlib.Path({str(fixture)!r}).read_text())\n"
        "print(json.dumps({'text':'Prose is not the structured plan.', 'structuredOutput':proposal,\n"
        "    'stopReason':'end_turn', 'num_turns':1, 'usage':{'input_tokens':10,\n"
        "    'cache_read_input_tokens':2, 'cache_creation_input_tokens':0,\n"
        "    'output_tokens':3, 'total_tokens':15}}))\n"
    )
    executable.chmod(0o700)
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: str(executable))
    provider = llm_router._get_grok_cli_provider()
    assert provider is not None
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["grok_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="grok_cli", model_name="grok-4.7", catalog_revision="authored-fixture"))
    monkeypatch.setattr(duckdb_store, "_DEFAULT_STORE", duckdb_store.AllocationStore(tmp_path / "allocation.duckdb"))

    text, receipt = runner.run(prompt=prompt, provider="grok_cli", model="grok-4.7",
        timeout=20, max_output_tokens=8192, purpose="planning")
    assert json.loads(text) == proposal
    assert "Prose is not" not in text
    observed = json.loads(capture.read_text())
    assert observed["schema_matched"] is True
    assert not Path(observed["prompt_file"]).exists()
    assert receipt["completion_authority"] is False
    assert receipt["native_rollout_usage"]["usage"]["total_tokens"] == 15
    assert receipt["provider_invocation_policy"]["effective_toolset_verified"] is False
    if case == "valid":
        graph = parse_prompt_goal_graph(text, request, scan)
        assert graph.request_cid == request.request_cid
        assert graph.scan_cid == scan.scan_cid
        assert graph.tasks[0].goal_cid == graph.root_goal.goal_cid
        assert graph.tasks[0].outputs[0].path == "pkg/retry_planner.py"
    else:
        with pytest.raises(PromptGoalProposalError):
            parse_prompt_goal_graph(text, request, scan)
