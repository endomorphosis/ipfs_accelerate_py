"""Codex generation constraints preserve the native planner's strict contract."""
from copy import deepcopy
import hashlib
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from ipfs_accelerate_py.agent_supervisor.runtime import codex_planning_schema as adapter
from ipfs_accelerate_py.cli_runtime.grok_structured_output import _bounded_json, MAX_SCHEMA_BYTES
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
    PromptGoalProposalError, build_prompt_goal_provider_request, parse_prompt_goal_graph,
)
from test.api.test_agent_supervisor_prompt_goal_planner import _proposal, _request, _scan


@pytest.fixture
def canonical():
    request = _request()
    scan = _scan(request)
    prompt = build_prompt_goal_provider_request(request, scan)
    return request, scan, prompt, json.loads(prompt), _proposal(scan)


def test_real_canonical_schema_and_original_response_preserved(canonical):
    request, scan, prompt, original, proposal = canonical
    before = deepcopy(original)
    wire, receipt, validator = adapter.planning_schema(prompt)
    Draft202012Validator.check_schema(wire)
    Draft202012Validator(wire).validate(proposal)
    text = json.dumps(proposal, indent=2)
    assert adapter.validate_planning_response(text, validator) == text
    assert parse_prompt_goal_graph(text, request, scan).request_cid == request.request_cid
    assert original == before
    assert validator.schema == original["response_schema"]
    assert "$id" not in wire and "bounds" not in wire and "definitions" not in wire
    assert receipt["canonical_schema_sha256"] == hashlib.sha256(
        _bounded_json(original["response_schema"], MAX_SCHEMA_BYTES, schema=True).encode()).hexdigest()
    assert receipt["native_wire_schema_sha256"] == hashlib.sha256(
        _bounded_json(wire, MAX_SCHEMA_BYTES, schema=True).encode()).hexdigest()
    assert receipt["shape_only"] is receipt["canonical_validation_preserved"] is True
    for key in ("execution_authority", "proof_authority", "completion_authority",
                "publication_authority", "scope_expansion_authority"):
        assert receipt[key] is False
    wire["$defs"]["task"]["properties"]["title"] = {"type": "string"}
    assert validator.schema == original["response_schema"] == before["response_schema"]


def test_projection_retains_assertions_and_resolves_every_reference(canonical):
    _, _, prompt, original, _ = canonical
    wire, _, _ = adapter.planning_schema(prompt)
    expected = deepcopy(original["response_schema"])
    del expected["$id"]
    del expected["bounds"]
    expected["$defs"] = expected.pop("definitions")
    changes = []

    def rewrite(value, path="$", check=False):
        if isinstance(value, dict):
            if "$ref" in value:
                if check:
                    pieces = value["$ref"].removeprefix("#/").split("/")
                    resolved = wire
                    for piece in pieces:
                        resolved = resolved[piece]
                    assert isinstance(resolved, dict)
                else:
                    value["$ref"] = value["$ref"].replace("#/definitions/", "#/$defs/", 1)
            if "const" in value:
                value.update(type="string", enum=[value.pop("const")])
                changes.append(path)
            elif "enum" in value and "type" not in value:
                value["type"] = "string"
                changes.append(path)
            if value.get("type") == "object":
                assert value["additionalProperties"] is False
                assert set(value["required"]) == set(value["properties"])
            for key, child in value.items():
                rewrite(child, path + "." + key, check)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                rewrite(child, path + f"[{index}]", check)

    rewrite(expected)
    assert len(changes) == 5
    assert wire == expected
    rewrite(wire, check=True)
    assert wire["$defs"]["validation"]["properties"]["expected_exit_codes"]["items"] == {
        "type": "integer", "minimum": 0, "maximum": 255}
    assert wire["$defs"]["validation"]["properties"]["argv"]["maxItems"] == 256
    assert wire["properties"]["tasks"]["maxItems"] == original["response_schema"]["bounds"]["max_tasks"]
    assert wire["$defs"]["task"]["properties"]["task_key"]["minLength"] == 1


@pytest.mark.parametrize("change", ["task_title", "missing_task_field", "extra_nested", "empty_key",
                                    "noninteger_exit", "oversize_argv", "empty_outputs"])
def test_invalid_original_response_rejects_without_repair(canonical, change):
    _, _, prompt, _, proposal = canonical
    wire, _, validator = adapter.planning_schema(prompt)
    task = proposal["tasks"][0]
    if change == "task_title":
        task["title"] = "PRIVATE MODEL CONTENT"
    elif change == "missing_task_field":
        del task["rationale"]
    elif change == "extra_nested":
        task["outputs"][0]["extra"] = "PRIVATE MODEL CONTENT"
    elif change == "empty_key":
        task["task_key"] = ""
    elif change == "noninteger_exit":
        task["validations"][0]["expected_exit_codes"] = [True]
    elif change == "oversize_argv":
        task["validations"][0]["argv"] = ["python"] * 257
    else:
        task["outputs"] = []
    before = deepcopy(proposal)
    with pytest.raises(ValidationError):
        Draft202012Validator(wire).validate(proposal)
    with pytest.raises(ValueError, match="^Codex planning response does not match the canonical schema$"):
        adapter.validate_planning_response(json.dumps(proposal), validator)
    assert proposal == before


def test_goal_title_allowed_and_task_title_rejected_by_native_parser(canonical):
    request, scan, prompt, _, proposal = canonical
    _, _, validator = adapter.planning_schema(prompt)
    proposal["goals"][0]["title"] = "An arbitrary valid goal title"
    text = json.dumps(proposal)
    assert adapter.validate_planning_response(text, validator) == text
    parse_prompt_goal_graph(text, request, scan)
    proposal["tasks"][0]["title"] = "A task has no title field"
    with pytest.raises(PromptGoalProposalError, match="unknown title"):
        parse_prompt_goal_graph(json.dumps(proposal), request, scan)


@pytest.mark.parametrize("change", ["task_title", "nested_required", "nested_open", "bounds",
                                    "extra_annotation", "external_ref", "nested_id"])
def test_altered_canonical_schema_rejects_before_provider(canonical, change):
    _, _, _, value, _ = canonical
    schema = value["response_schema"]
    if change == "task_title":
        schema["definitions"]["task"]["properties"]["title"] = {"type": "string"}
        schema["definitions"]["task"]["required"].append("title")
    elif change == "nested_required":
        schema["definitions"]["task"]["required"].remove("rationale")
    elif change == "nested_open":
        schema["definitions"]["task"]["additionalProperties"] = True
    elif change == "bounds":
        schema["bounds"]["max_tasks"] += 1
    elif change == "extra_annotation":
        schema["description"] = "A lookalike schema"
    elif change == "external_ref":
        schema["properties"]["tasks"]["items"]["$ref"] = "https://example.invalid/private"
    else:
        schema["definitions"]["task"]["$id"] = "https://example.invalid/nested"
    with pytest.raises(ValueError):
        adapter.planning_schema(json.dumps(value))


def test_noncanonical_and_coverage_requests_are_untouched(canonical):
    _, _, _, value, _ = canonical
    assert adapter.planning_schema("Return READY") is None
    assert adapter.planning_schema('{"unrelated":"JSON"}') is None
    coverage = {"schema": "intent-plan-provider-request@1", "graph_request": value,
                "response_contract": {"schema": "intent-plan-proposal@1"}}
    assert adapter.planning_schema(json.dumps(coverage)) is None


def test_duplicate_nonfinite_deep_and_oversize_inputs_reject(canonical):
    _, _, prompt, value, proposal = canonical
    marker = value["schema"]
    with pytest.raises(ValueError):
        adapter.planning_schema('{"schema":' + json.dumps(marker) + ',"schema":"other"}')
    with pytest.raises(ValueError):
        adapter.planning_schema(prompt[:-1] + ',"extra":NaN}')
    value["extra"] = "leaf"
    for _ in range(35):
        value["extra"] = [value["extra"]]
    with pytest.raises(ValueError):
        adapter.planning_schema(json.dumps(value))
    with pytest.raises(ValueError):
        adapter.planning_schema("x" * (adapter.MAX_PROMPT_BYTES + 1))
    _, _, validator = adapter.planning_schema(prompt)
    text = json.dumps(proposal)
    bad = [" " + text, text + "\n", text + " trailing", text.replace('"schema":', '"schema":"other","schema":', 1),
           text[:-1] + ',"extra":Infinity}']
    for output in bad:
        with pytest.raises(ValueError, match="^Codex planning response does not match the canonical schema$"):
            adapter.validate_planning_response(output, validator)
    oversized_original = "{" + " " * validator.schema["bounds"]["max_serialized_bytes"] + text[1:]
    assert json.loads(oversized_original) == proposal
    with pytest.raises(ValueError, match="^Codex planning response does not match the canonical schema$"):
        adapter.validate_planning_response(oversized_original, validator)


def test_transport_does_not_admit_schema_valid_foreign_scope(canonical):
    request, scan, prompt, _, proposal = canonical
    _, _, validator = adapter.planning_schema(prompt)
    proposal["tasks"][0]["outputs"][0]["path"] = "outside/unauthorized.py"
    text = json.dumps(proposal)
    assert adapter.validate_planning_response(text, validator) == text
    with pytest.raises(PromptGoalProposalError):
        parse_prompt_goal_graph(text, request, scan)
