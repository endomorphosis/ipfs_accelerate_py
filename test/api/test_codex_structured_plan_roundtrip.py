"""Native process schema custody and independent graph checks, without a model."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test.api.test_agent_supervisor_prompt_goal_planner import _request, _scan, _proposal
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
    PromptGoalProposalError, build_prompt_goal_provider_request, parse_prompt_goal_graph,
)


@pytest.mark.parametrize("case", ["valid", "task_title", "foreign_output", "foreign_evidence"])
def test_native_schema_custody_keeps_independent_plan_admission(tmp_path, monkeypatch, capsys, case):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    from ipfs_accelerate_py.llm_allocation import duckdb_store, intelligence_index

    request = _request()
    scan = _scan(request)
    prompt = build_prompt_goal_provider_request(request, scan)
    proposal = _proposal(scan)
    if case == "task_title":
        proposal["tasks"][0]["title"] = "Extra display label"
    elif case == "foreign_output":
        proposal["tasks"][0]["outputs"][0]["path"] = "outside/unauthorized.py"
    elif case == "foreign_evidence":
        proposal["tasks"][0]["evidence_cids"] = [request.policy_root]

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-codex-home"))
    capture = tmp_path / "native-observation.json"
    fixture = tmp_path / "proposal.json"
    fixture.write_text(json.dumps(proposal))
    executable = tmp_path / "codex"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import hashlib,json,pathlib,stat,sys\n"
        "argv=sys.argv[1:]\n"
        "def option(name):\n"
        " assert argv.count(name)==1,name\n"
        " return argv[argv.index(name)+1]\n"
        "schema_path=pathlib.Path(option('--output-schema'))\n"
        "raw=schema_path.read_bytes(); schema=json.loads(raw)\n"
        "prompt=sys.stdin.read()\n"
        "assert schema['additionalProperties'] is False\n"
        "assert 'title' not in schema['$defs']['task']['properties']\n"
        "assert 'title' in schema['$defs']['goal']['properties']\n"
        "assert stat.S_IMODE(schema_path.stat().st_mode)==0o600\n"
        f"pathlib.Path({str(capture)!r}).write_text(json.dumps({{'schema_path':str(schema_path),'schema_sha256':hashlib.sha256(raw).hexdigest(),'schema_bytes':len(raw),'prompt':prompt}}))\n"
        f"text=pathlib.Path({str(fixture)!r}).read_text()\n"
        "pathlib.Path(option('--output-last-message')).write_text(text)\n"
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':15,'output_tokens':4}}))\n"
    )
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: llm_router._get_codex_cli_provider())
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="codex_cli", model_name="pinned", catalog_revision="authored-fixture"))
    monkeypatch.setattr(duckdb_store, "_DEFAULT_STORE", duckdb_store.AllocationStore(tmp_path / "allocation.duckdb"))

    if case == "task_title":
        with pytest.raises(ValueError, match="planning response"):
            runner.run(prompt=prompt, provider="codex_cli", model="pinned", timeout=20,
                       max_output_tokens=4096, purpose="planning")
        rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
        receipt = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
        assert receipt["status"] == "failed"
        assert receipt["failure_phase"] == "provider_result_validation"
        assert receipt["usage"]["prompt_tokens"] == 15
        assert receipt["usage"]["completion_tokens"] == 4
    else:
        text, receipt = runner.run(prompt=prompt, provider="codex_cli", model="pinned", timeout=20,
                                   max_output_tokens=4096, purpose="planning")
        assert json.loads(text) == proposal
        if case == "valid":
            assert parse_prompt_goal_graph(text, request, scan).request_cid == request.request_cid
        else:
            with pytest.raises(PromptGoalProposalError):
                parse_prompt_goal_graph(text, request, scan)

    observed = json.loads(capture.read_text())
    assert observed["prompt"] == prompt
    assert not Path(observed["schema_path"]).exists()
    policy = receipt["provider_invocation_policy"]["structured_output"]
    assert policy["native_schema_requested"] is True
    assert policy["response_schema_validated"] is (case != "task_title")
    assert policy["plan_admitted"] is False
    assert receipt["usage"]["codex_output_schema_sha256"] == observed["schema_sha256"]
    assert receipt["usage"]["codex_output_schema_bytes"] == observed["schema_bytes"]
    assert receipt["native_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert receipt["completion_authority"] is False
