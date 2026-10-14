"""Default prompt planner crosses the actual isolated child and native CLI.

The executable is a local transport fixture; no model, proof, or admission is
simulated. The real router, projection, native schema validators, and graph
parser process its response.
"""
import hashlib
import json
import os
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
    PromptGoalPlannerConfig, _default_router, build_prompt_goal_provider_request,
    generate_prompt_goal_graph,
)
from ipfs_accelerate_py.agent_supervisor.runtime.codex_planning_schema import planning_schema
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import (
    LlmRouterInvocation, call_llm_router, last_llm_child_result,
)
from test.api.test_agent_supervisor_prompt_goal_planner import _proposal, _request, _scan


@pytest.fixture
def native_codex(tmp_path, monkeypatch):
    executable = tmp_path / "codex"
    executable.write_text("#!" + sys.executable + "\n" + '''
import json,os,pathlib,sys
args=sys.argv[1:]
prompt=sys.stdin.read()
if "--output-schema" not in args:
    raise SystemExit("the default planner omitted the native output schema")
schema=json.loads(pathlib.Path(args[args.index("--output-schema")+1]).read_text())
pathlib.Path(os.environ["CODEX_PLANNER_FIXTURE_CALL"]).write_text(json.dumps({"argv":args,"schema":schema,"prompt":prompt}))
response=pathlib.Path(os.environ["CODEX_PLANNER_FIXTURE_RESPONSE"]).read_text()
pathlib.Path(args[args.index("--output-last-message")+1]).write_text(response)
print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":response}}))
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":13,"output_tokens":2,"total_tokens":15}}))
''', encoding="utf-8")
    executable.chmod(0o755)
    response_file = tmp_path / "proposal.json"
    call_file = tmp_path / "native-call.json"
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("CODEX_PLANNER_FIXTURE_CALL", str(call_file))
    monkeypatch.setenv("CODEX_PLANNER_FIXTURE_RESPONSE", str(response_file))
    monkeypatch.setenv("IPFS_ACCELERATE_SUPERVISOR_USAGE_MODE", "off")
    monkeypatch.delenv("ipfs_accelerate_py_CODEX_SANDBOX", raising=False)
    return response_file, call_file


@pytest.mark.parametrize("provider", ["codex", "codex-cli", "codex_cli"])
def test_default_planner_preserves_formal_strings_through_real_child_native_schema(
    native_codex, tmp_path, provider
):
    response_file, call_file = native_codex
    request = _request()
    scan = _scan(request)
    proposal = _proposal(scan)
    formal = "Preserve ∀x. O(retain<File>(x)) ∧ t ≤ 20; lake build <Lib>; &amp; &lt;Lib&gt;."
    proposal["goals"][0]["objective"] = formal
    proposal["tasks"][0]["objective"] = formal
    raw = json.dumps(proposal, ensure_ascii=False, indent=2)
    response_file.write_text(raw, encoding="utf-8")
    result = generate_prompt_goal_graph(request, scan, config=PromptGoalPlannerConfig(
        repo_root=tmp_path, provider=provider, model="fixture-model", timeout_seconds=20,
    ))
    assert result.provider_succeeded and not result.used_fallback
    assert result.graph.goals[0].objective == formal
    assert result.graph.tasks[0].objective == formal
    assert result.receipt.provider.response_sha256 == "sha256:" + hashlib.sha256(raw.encode()).hexdigest()
    assert result.receipt.provider.response_bytes == len(raw.encode())
    call = json.loads(call_file.read_text())
    wire, _, _ = planning_schema(call["prompt"])
    assert call["schema"] == wire
    assert "--output-schema" in call["argv"]
    child = last_llm_child_result()
    assert child is not None and child.effective_provider == "codex_cli"
    assert child.text_chars == len(raw)


def test_opted_in_child_rejects_noncanonical_request_before_native_execution(native_codex, tmp_path):
    _, call_file = native_codex
    config = LlmRouterInvocation(repo_root=tmp_path, model_name="fixture-model", provider="codex_cli",
                                 python_executable=sys.executable, temperature=0.0,
                                 timeout_seconds=20, codex_planning_schema=True)
    with pytest.raises(RuntimeError, match="canonical prompt-goal request"):
        call_llm_router("ordinary prompt without a planning contract", config)
    assert not call_file.exists()


def test_default_planner_native_schema_rejects_invalid_response_without_repair(native_codex, tmp_path):
    response_file, _ = native_codex
    request = _request()
    scan = _scan(request)
    proposal = _proposal(scan)
    proposal["tasks"][0]["title"] = "PRIVATE INVALID PROVIDER FIELD"
    response_file.write_text(json.dumps(proposal), encoding="utf-8")
    config = PromptGoalPlannerConfig(repo_root=tmp_path, provider="codex_cli", model="fixture-model",
                                    timeout_seconds=20)
    with pytest.raises(RuntimeError, match="llm_router child exited with code 1") as caught:
        _default_router(build_prompt_goal_provider_request(request, scan), request, config)
    assert "PRIVATE INVALID PROVIDER FIELD" not in str(caught.value)
