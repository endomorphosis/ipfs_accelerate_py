"""Signed owner declarations constrain generation, never replace admission."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator, ValidationError

from ipfs_accelerate_py.agent_supervisor.runtime import terminal_planner_contract as contract
from ipfs_accelerate_py.cli_runtime.grok_structured_output import planning_response_format, native_schema_projection


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
    root = tmp_path_factory.mktemp("typed-terminal-contract")
    repository = root / "app"
    repository.mkdir()
    (repository / "bottle.py").write_text("def application():\n    return 'fixture'\n")
    for argv in (["init", "-q"], ["add", "."], ["-c", "user.name=Fixture", "-c",
            "user.email=fixture@example.invalid", "commit", "-qm", "authored input"]):
        subprocess.run(["git", "-C", str(repository), *argv], check=True, capture_output=True)
    instruction = root / "instruction.md"
    instruction.write_text("Inspect bottle.py, repair it and write report.jsonl containing file_path and cwe_id fields.")
    return prep.prepare(repository=repository, instruction=instruction, state=root / "state",
        provider_profile="grok-4.7-cli-1.0.46@1")


def request_text(prepared):
    return (Path(prepared["state"]) / "provider-request.json").read_text()


def proposal(prepared):
    from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import _proposal_json
    return json.loads(_proposal_json(prepared))


def graph(prepared, text):
    from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import parse_prompt_goal_graph
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptWorkflowRequest, DirectoryScanReceipt
    return parse_prompt_goal_graph(text, PromptWorkflowRequest.from_dict(prepared["request"]),
        DirectoryScanReceipt.from_dict(prepared["scan"]), config=prep._config(Path(prepared["repository"])),
        constraint_summaries=prepared["constraints"])


def resign_hint(value):
    value["contract_sha256"] = hashlib.sha256(contract._json(
        {key: row for key, row in value.items() if key != "contract_sha256"}).encode()).hexdigest()


def test_verified_preparation_projects_every_signed_task_field(prepared):
    prompt = request_text(prepared)
    value = contract.contract_from_prompt(prompt)
    signed = prepared["manifest"]["payload"]["tasks"][0]
    task = value["tasks"][0]
    for key in ("task_key", "scope_paths", "outputs", "acceptance"):
        assert task[key] == signed[key]
    assert task["dependency_task_keys"] == signed["dependencies"]
    assert task["validations"] == [{key: item for key, item in row.items() if key != "policy_cid"}
                                   for row in signed["validations"]]
    assert value["execution_authority"] is value["completion_authority"] is False
    fmt = planning_response_format(prompt)
    canonical = deepcopy(fmt["json_schema"]["schema"])
    wire, encoded, receipt = native_schema_projection(canonical, task_contract=value)
    Draft202012Validator.check_schema(wire)
    Draft202012Validator(wire).validate(proposal(prepared))
    assert fmt["json_schema"]["schema"] == canonical
    assert receipt["projection_id"] == "canonical-prompt-goal-task-contract@1"
    assert receipt["task_contract_sha256"] == value["contract_sha256"]
    assert receipt["task_count"] == 1 and receipt["task_contract_authority"] is False
    assert receipt["canonical_schema_sha256"] != receipt["native_wire_schema_sha256"]
    assert hashlib.sha256(encoded.encode()).hexdigest() == receipt["native_wire_schema_sha256"]


@pytest.mark.parametrize("mutation", ["task_key", "scope", "output", "validation", "acceptance", "dependency",
    "assumption", "goal_acceptance", "goal_assumption", "goal_evidence", "goal_scope", "population"])
def test_schema_valid_contract_drift_is_rejected_by_native_intersection(prepared, mutation):
    value = proposal(prepared)
    task = value["tasks"][0]
    if mutation == "task_key": task["task_key"] = "OTHER-TASK"
    elif mutation == "scope": task["scope_paths"] = ["bottle.py"]
    elif mutation == "output": task["outputs"][0]["effect"] = "create"
    elif mutation == "validation": task["validations"][0]["cwd"] = "bottle.py"
    elif mutation == "acceptance": task["acceptance"][0]["criterion"] = "A paraphrase"
    elif mutation == "dependency": task["dependency_task_keys"] = ["OTHER-TASK"]
    elif mutation == "assumption": task["assumptions"] = ["An assumption"]
    elif mutation == "goal_acceptance": value["goals"][0]["acceptance"][0]["criterion"] = "A paraphrase"
    elif mutation == "goal_assumption": value["goals"][0]["assumptions"] = ["An assumption"]
    elif mutation == "goal_evidence": value["goals"][0]["evidence_cids"] = ["foreign-evidence"]
    elif mutation == "goal_scope": value["goals"][0]["scope_paths"] = ["outside.py"]
    else: value["goals"] = value["goals"][:1]
    prompt = request_text(prepared)
    canonical = planning_response_format(prompt)["json_schema"]["schema"]
    Draft202012Validator(canonical).validate(value)
    wire, _, _ = native_schema_projection(canonical, task_contract=contract.contract_from_prompt(prompt))
    with pytest.raises(ValidationError):
        Draft202012Validator(wire).validate(value)


@pytest.mark.parametrize("mutation", ["hash", "request", "scope", "command", "evidence", "authority"])
def test_mutated_typed_hint_rejected_before_provider_call(prepared, mutation):
    prompt = json.loads(request_text(prepared))
    hint = prompt[contract.FIELD]
    if mutation == "hash": hint["tasks"][0]["acceptance"][0]["criterion"] = "Tampered"
    elif mutation == "request": hint["request_cid"] = "another-request"
    elif mutation == "scope": hint["tasks"][0]["scope_paths"].append("outside.py")
    elif mutation == "command": hint["tasks"][0]["validations"][0]["argv"] = ["python3", "-B", "different.py"]
    elif mutation == "evidence": hint["evidence_cids"].append("another-evidence")
    else: hint["completion_authority"] = True
    if mutation != "hash": resign_hint(hint)
    with pytest.raises(ValueError):
        contract.contract_from_prompt(json.dumps(prompt))


def test_forged_preparation_cannot_author_a_decoding_hint(prepared):
    forged = deepcopy(prepared)
    forged["spec"]["acceptance"][0]["criterion"] = "Tampered"
    prompt = json.loads(request_text(prepared))
    del prompt[contract.FIELD]
    with pytest.raises(ValueError):
        contract.bind_terminal_task_contract(json.dumps(prompt), prepared=forged)


@pytest.mark.parametrize("field", ["request_core", "constraints"])
@pytest.mark.parametrize("value", [None, []])
def test_malformed_binding_maps_fail_with_closed_contract_error(prepared, field, value):
    payload = json.loads(request_text(prepared))
    payload[field] = value
    with pytest.raises(ValueError, match="terminal task contract differs"):
        contract.contract_from_prompt(json.dumps(payload))


@pytest.mark.parametrize("prepared", [
    {"schema": "terminal-indexed-public-preparation@2", "planning_strategy": "intent_symbolic"},
    {"schema": "terminal-indexed-public-preparation@1", "planning_strategy": "intent_coverage"},
    {"schema": "terminal-indexed-public-preparation@1", "planning_strategy": "intent_symbolic"},
])
def test_non_direct_or_multitask_routes_are_unchanged(prepared):
    assert contract.bind_terminal_task_contract("original wrapped request", prepared=prepared) == "original wrapped request"


@pytest.mark.parametrize("budget_field", ["max_tasks", "max_goals"])
def test_unsupported_task_budget_fails_before_native_dispatch(prepared, budget_field):
    prompt = request_text(prepared)
    schema = planning_response_format(prompt)["json_schema"]["schema"]
    schema["bounds"][budget_field] += 1
    array = "tasks" if budget_field == "max_tasks" else "goals"
    schema["properties"][array]["maxItems"] += 1
    with pytest.raises(ValueError, match="singleton planner grammar"):
        native_schema_projection(schema, task_contract=contract.contract_from_prompt(prompt))


def test_rehashed_decoding_hint_still_cannot_authorize_a_different_task(prepared):
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    prompt = json.loads(request_text(prepared))
    hint = prompt[contract.FIELD]
    hint["tasks"][0]["acceptance"][0]["criterion"] = "A forged acceptance meaning"
    resign_hint(hint)
    validated = contract.contract_from_prompt(json.dumps(prompt))
    value = proposal(prepared)
    for goal in value["goals"]: goal["acceptance"] = deepcopy(validated["tasks"][0]["acceptance"])
    value["tasks"][0]["acceptance"] = deepcopy(validated["tasks"][0]["acceptance"])
    wire, _, _ = native_schema_projection(prompt["response_schema"], task_contract=validated)
    Draft202012Validator(wire).validate(value)
    parsed = graph(prepared, json.dumps(value))
    with pytest.raises(local.LocalPlanningError, match="signed scope, acceptance"):
        local.admit_local_benchmark_plan(graph=parsed, manifest=prepared["manifest"])


def _invoke_authored_cli(prepared, tmp_path, monkeypatch, value):
    """Exercise the real process adapter with authored output, never a provider."""
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    from ipfs_accelerate_py.llm_allocation import duckdb_store, intelligence_index
    fixture = tmp_path / "proposal.json"
    fixture.write_text(json.dumps(value))
    executable = tmp_path / "grok"
    executable.write_text(f"#!{sys.executable}\n" +
        "import json,pathlib,sys\n"
        "from jsonschema import Draft202012Validator\n"
        "argv=sys.argv[1:]\n"
        "def option(name):\n assert argv.count(name)==1\n return argv[argv.index(name)+1]\n"
        "prompt=json.loads(pathlib.Path(option('--prompt-file')).read_text())\n"
        "wire=json.loads(option('--json-schema'))\n"
        "Draft202012Validator.check_schema(wire)\n"
        "assert '$id' not in wire\n"
        "assert wire['definitions']['task']['allOf'][0]['properties']['acceptance']['const']==prompt['terminal_planner_task_contract']['tasks'][0]['acceptance']\n"
        "assert wire['definitions']['task']['allOf'][0]['properties']['outputs']['const']==prompt['terminal_planner_task_contract']['tasks'][0]['outputs']\n"
        "assert option('--tools')=='read_file' and option('--disallowed-tools')=='read_file,search_tool,use_tool'\n"
        "assert option('--max-turns')=='2' and option('--permission-mode')=='dontAsk'\n"
        f"proposal=json.loads(pathlib.Path({str(fixture)!r}).read_text())\n"
        "print(json.dumps({'structuredOutput':proposal,'stopReason':'end_turn','num_turns':1}))\n")
    executable.chmod(0o700)
    monkeypatch.chdir(prepared["repository"])
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: str(executable))
    provider = llm_router._get_grok_cli_provider()
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *a, **kw: provider)
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["grok_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kw:
        SimpleNamespace(provider="grok_cli", model_name="grok-4.7", catalog_revision="authored-fixture"))
    monkeypatch.setattr(duckdb_store, "_DEFAULT_STORE", duckdb_store.AllocationStore(tmp_path / "allocation.duckdb"))
    return runner.run(prompt=request_text(prepared), provider="grok_cli", model="grok-4.7", timeout=20,
        max_output_tokens=4096, purpose="planning")


@pytest.mark.parametrize("drift", [False, True])
def test_actual_cli_router_parser_and_signed_admission(prepared, tmp_path, monkeypatch, drift):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    value = proposal(prepared)
    if drift: value["tasks"][0]["acceptance"][0]["criterion"] = "Schema-valid signed-contract drift"
    if drift:
        with pytest.raises(llm_router.LLMRouterError, match="structured response failed validation"):
            _invoke_authored_cli(prepared, tmp_path, monkeypatch, value)
    else:
        text, receipt = _invoke_authored_cli(prepared, tmp_path, monkeypatch, value)
        assert json.loads(text) == value
        typed = receipt["provider_invocation_policy"]["structured_output"]["native_schema_projection"]
        assert typed["projection_id"] == "canonical-prompt-goal-task-contract@1"
        assert typed["task_contract_authority"] is False
        admission = local.admit_local_benchmark_plan(graph=graph(prepared, text), manifest=prepared["manifest"])
        verified = local.verify_local_benchmark_admission(admission)
        assert verified["receipt"]["completion_authority"] is False
        assert all(row["phase"] == "post_execution" for row in verified["receipt"]["pending_requirements"])
