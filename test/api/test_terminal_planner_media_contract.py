"""Catalog output formats traverse the signed planner without changing contracts.

Only public catalog declarations are consumed. Input bytes and instructions are
authored fixtures, never benchmark data, solutions, or provider responses.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest
from jsonschema import Draft202012Validator, ValidationError

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.terminal_profile_catalog import CATALOG
from ipfs_accelerate_py.agent_supervisor.prompt import prompt_goal_planner as planner
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import terminal_planner_contract as contract
from ipfs_accelerate_py.agent_supervisor.runtime import terminal_task_profile as profiles
from ipfs_accelerate_py.cli_runtime.grok_structured_output import planning_response_format, native_schema_projection
from test.api.test_terminal_planner_task_contract import (
    _invoke_authored_cli, graph, proposal, request_text,
)


CATALOG_TASKS = json.loads(CATALOG.read_text())["tasks"]
AUTHORED_INPUTS = {
    None: b"def fixture():\n    return 'authored'\n",
    "application/json": b'{"fixture":true}\n',
    "application/x-ndjson": b'{"fixture":true}\n',
    "application/xml": b'<mujoco model="authored"><worldbody/></mujoco>\n',
    "text/calendar": b"BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:authored\nEND:VCALENDAR\n",
}


def _prepare_catalog_shape(root, task_name, *, output_media=None):
    row = CATALOG_TASKS[task_name]
    repository = root / "app"
    repository.mkdir()
    for item in row["inputs"]:
        path = repository / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(AUTHORED_INPUTS[item["media_type"]])
    for argv in (["init", "-q"], ["add", "."], ["-c", "user.name=Fixture", "-c",
            "user.email=fixture@example.invalid", "commit", "--allow-empty", "-qm", "authored public shape"]):
        subprocess.run(["git", "-C", str(repository), *argv], check=True, capture_output=True)
    instruction = root / "instruction.md"
    instruction.write_text("Create the declared outputs from the authored public inputs.\n")
    outputs = deepcopy(row["outputs"])
    if output_media is not None:
        outputs[0]["media_type"] = output_media
    profile = profiles.validate_task_profile({
        "schema": profiles.DATA_SCHEMA,
        "instruction_sha256": profiles.instruction_sha256(instruction.read_text()),
        "input_paths": [item["path"] for item in row["inputs"]],
        "data_inputs": [{key: item[key] for key in ("path", "media_type")}
                        for item in row["inputs"] if item["media_type"] is not None],
        "outputs": outputs,
    })
    return prep.prepare(repository=repository, instruction=instruction, state=root / "state",
        task_profile=profile, disable_intent_autoencoder=True,
        provider_profile="grok-4.7-cli-1.0.46@1")


@pytest.fixture(scope="module", params=sorted(CATALOG_TASKS))
def catalog_prepared(request, tmp_path_factory):
    return _prepare_catalog_shape(tmp_path_factory.mktemp(request.param), request.param)


def _proposal(prepared):
    value = proposal(prepared)
    # The shared fixture's original Bottle predictions are unrelated to these
    # catalog declarations. Author exact declared paths before any router call.
    value["tasks"][0]["predicted_files"] = [row["path"] for row in prepared["spec"]["outputs"]]
    return value


def test_catalog_and_closed_data_validators_have_canonical_planner_coverage():
    catalog_media = {row["media_type"] for task in CATALOG_TASKS.values() for row in task["outputs"]}
    assert catalog_media <= planner._SAFE_MEDIA_TYPES
    assert set(profiles.DATA_MEDIA_SUFFIXES) <= planner._SAFE_MEDIA_TYPES


def test_catalog_shape_reaches_router_parser_and_independent_signed_admission(catalog_prepared, tmp_path, monkeypatch):
    prepared = catalog_prepared
    original_outputs = deepcopy(prepared["spec"]["outputs"])
    prompt = request_text(prepared)
    canonical = planning_response_format(prompt)["json_schema"]["schema"]
    hint = contract.contract_from_prompt(prompt)
    assert hint["tasks"][0]["outputs"] == original_outputs
    wire, _, projection = native_schema_projection(canonical, task_contract=hint)
    Draft202012Validator.check_schema(wire)
    value = _proposal(prepared)
    Draft202012Validator(canonical).validate(value)
    Draft202012Validator(wire).validate(value)
    text, receipt = _invoke_authored_cli(prepared, tmp_path, monkeypatch, value)
    assert json.loads(text) == value
    assert prepared["spec"]["outputs"] == original_outputs
    assert receipt["provider_invocation_policy"]["structured_output"]["native_schema_projection"] == projection
    parsed = graph(prepared, text)
    # Native records use content-ID ordering; admission compares output rows
    # without order, preserving every path, effect and MIME value.
    parsed_outputs = [{key: getattr(row, key) for key in ("path", "effect", "media_type")}
                      for row in parsed.tasks[0].outputs]
    assert sorted(parsed_outputs, key=lambda row: row["path"]) == sorted(original_outputs, key=lambda row: row["path"])
    verified = local.verify_local_benchmark_admission(local.admit_local_benchmark_plan(
        graph=parsed, manifest=prepared["manifest"]))
    assert verified["receipt"]["completion_authority"] is False
    assert all(row["phase"] == "post_execution" for row in verified["receipt"]["pending_requirements"])
    assert all(not (Path(prepared["repository"]) / row["path"]).exists() for row in original_outputs)


@pytest.mark.parametrize("replacement", ["text/plain", "other_reviewed_data"])
def test_allowed_but_unsigned_media_is_rejected_by_generation_and_admission(catalog_prepared, tmp_path, monkeypatch, replacement):
    from ipfs_accelerate_py import llm_router
    prepared = catalog_prepared
    value = _proposal(prepared)
    output = value["tasks"][0]["outputs"][0]
    if replacement == "other_reviewed_data":
        replacement = "text/calendar" if output["media_type"] != "text/calendar" else "application/xml"
    output["media_type"] = replacement
    canonical = planning_response_format(request_text(prepared))["json_schema"]["schema"]
    Draft202012Validator(canonical).validate(value)
    parsed = graph(prepared, json.dumps(value))
    with pytest.raises(local.LocalPlanningError, match="signed scope, acceptance"):
        local.admit_local_benchmark_plan(graph=parsed, manifest=prepared["manifest"])
    with pytest.raises(llm_router.LLMRouterError, match="structured response failed validation"):
        _invoke_authored_cli(prepared, tmp_path, monkeypatch, value)


@pytest.mark.parametrize("media", ["application/x-unreviewed", "text/html", "application/xml; charset=utf-8"])
def test_unknown_media_remains_rejected_by_schema_parser_and_router(catalog_prepared, tmp_path, monkeypatch, media):
    from ipfs_accelerate_py import llm_router
    prepared = catalog_prepared
    value = _proposal(prepared)
    value["tasks"][0]["outputs"][0]["media_type"] = media
    canonical = planning_response_format(request_text(prepared))["json_schema"]["schema"]
    with pytest.raises(ValidationError):
        Draft202012Validator(canonical).validate(value)
    with pytest.raises(planner.PromptGoalProposalError, match="media type is not admitted"):
        graph(prepared, json.dumps(value))
    with pytest.raises(llm_router.LLMRouterError, match="structured response failed validation"):
        _invoke_authored_cli(prepared, tmp_path, monkeypatch, value)


def test_signed_unsupported_media_refuses_before_provider_request(tmp_path):
    with pytest.raises(ValueError, match="canonical grammar"):
        _prepare_catalog_shape(tmp_path, "tune-mjcf", output_media="application/x-unreviewed")
    assert not (tmp_path / "state/provider-request.json").exists()
