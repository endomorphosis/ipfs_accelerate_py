"""Native schema transport must preserve exact output and independent admission."""
import json
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.cli_runtime import grok_structured_output as structured


def response_format():
    return {"type": "json_schema", "json_schema": {"name": "ready_probe", "strict": True,
        "schema": {"type": "object", "additionalProperties": False, "required": ["status"],
                   "properties": {"status": {"const": "READY"}}}}}


@pytest.fixture
def adapter(monkeypatch):
    from ipfs_accelerate_py import llm_router
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/fixture/grok")
    monkeypatch.setattr(llm_router, "_cli_available", lambda _: True)
    return llm_router


@pytest.mark.parametrize("field", ["structuredOutput", "structured_output", "text", "direct"])
def test_native_structured_variants_validate_and_preserve_usage(adapter, monkeypatch, field):
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation
    expected = {"status": "READY"}
    envelope = {"stopReason": "end_turn", "usage": {"input_tokens": 12, "output_tokens": 4},
                "text": "Human-readable envelope message"}
    if field == "direct":
        envelope = expected
    else:
        envelope[field] = json.dumps(expected) if field == "text" else expected
    calls = []
    def execute(argv, **kwargs):
        calls.append(argv)
        assert json.loads(argv[argv.index("--json-schema") + 1]) == response_format()["json_schema"]["schema"]
        assert Path(argv[argv.index("--prompt-file") + 1]).read_text() == "canonical prompt"
        assert kwargs["timeout"] == 19
        return SimpleNamespace(returncode=0, stdout=json.dumps(envelope), stderr="")
    monkeypatch.setattr(adapter.subprocess, "run", execute)
    result = adapter._get_grok_cli_provider().generate("canonical prompt", model_name="grok-4.7",
        timeout=19, response_format=response_format(), grok_tools="read_file",
        grok_disallowed_tools="read_file,search_tool,use_tool", grok_max_turns=2)
    assert result == '{"status":"READY"}'
    assert len(calls) == 1
    assert calls[0][calls[0].index("--max-turns") + 1] == "2"
    assert not Path(calls[0][calls[0].index("--prompt-file") + 1]).exists()
    if field != "direct":
        assert get_last_cli_observation("grok_cli")["native_grok_input_tokens"] == 12


@pytest.mark.parametrize("payload", [
    {"structuredOutput": {"status": "WRONG"}},
    {"structuredOutput": {"status": "READY", "extra": True}},
    {"structuredOutput": {"status": "READY"}, "structuredOutputError": "private failure"},
    {"structured_output": {"status": "READY"}, "structured_output_error": "private failure"},
    {"structuredOutput": {"status": "READY"}, "structured_output": {"status": "WRONG"}},
    {"structuredOutput": None, "text": '{"status":"READY"}'},
    {"structuredOutput": [], "text": '{"status":"READY"}'},
    {"text": 'Here is the result: {"status":"READY"}'},
    {"text": '{"status":"READY"} trailing prose'},
    {"text": '{"status":"WRONG","status":"READY"}'},
    {"text": '{"status":NaN}'},
    {"text": '{"status":"READY"}', "is_error": True},
    {"text": '{"status":"READY"}', "subtype": "error_max_structured_output_retries"},
    {"structuredOutput": {"status": "READY"}, "stopReason": "cancelled"},
    {"structuredOutput": {"status": "READY"}, "stopReason": "max_tokens"},
    {"structuredOutput": {"status": "READY"}, "stopReason": "max_turn_requests"},
    {"structuredOutput": {"status": "READY"}, "type": "max_turns_reached"},
    {"structuredOutput": {"status": "READY"}, "stopReason": "end_turn", "stop_reason": "stop"},
    {"stopReason": "end_turn"},
])
def test_bad_native_result_is_rejected_without_prose_repair(adapter, monkeypatch, payload):
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **kw:
        SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr=""))
    with pytest.raises(adapter.LLMRouterError, match="structured response failed validation"):
        adapter._get_grok_cli_provider().generate("canonical prompt", response_format=response_format())


def test_native_duplicate_envelope_fields_are_not_discarded():
    _, _, validator = structured.validate_response_format(response_format())
    with pytest.raises(ValueError, match="duplicate"):
        structured.decode_response('{"structuredOutput":{"status":"WRONG"},"structuredOutput":{"status":"READY"}}', validator)


def test_ndjson_uses_only_final_envelope_and_matching_aliases():
    _, _, validator = structured.validate_response_format(response_format())
    payload = {"structuredOutput": {"status": "READY"}, "structured_output": {"status": "READY"}}
    assert structured.decode_response('{"event":"progress"}\n' + json.dumps(payload), validator) == '{"status":"READY"}'
    with pytest.raises(ValueError):
        structured.decode_response(json.dumps(payload) + '\n{"type":"error"}', validator)


@pytest.mark.parametrize("change", [
    {"type": "json_object"}, {"type": "text"}, {},
    {"type": "json_schema", "json_schema": {}},
    {"type": "json_schema", "json_schema": {"name": "fixture", "strict": False, "schema": {"type": "object"}}},
])
def test_unsupported_format_fails_before_native_call(adapter, monkeypatch, change):
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **kw: pytest.fail("native call reached"))
    with pytest.raises(ValueError, match="response_format"):
        adapter._get_grok_cli_provider().generate("canonical prompt", response_format=change)


@pytest.mark.parametrize("reference", ["$ref", "$dynamicRef", "$recursiveRef"])
def test_external_schema_reference_is_rejected_before_native_call(adapter, monkeypatch, reference):
    selected = response_format()
    selected["json_schema"]["schema"]["properties"]["status"] = {reference: "https://invalid.example/schema"}
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **kw: pytest.fail("native call reached"))
    with pytest.raises(ValueError, match="only local references"):
        adapter._get_grok_cli_provider().generate("canonical prompt", response_format=selected)


def test_internal_definition_reference_retained_and_validated():
    selected = response_format()
    schema = selected["json_schema"]["schema"]
    schema["definitions"] = {"status": schema["properties"]["status"]}
    schema["properties"]["status"] = {"$ref": "#/definitions/status"}
    actual, encoded, validator = structured.validate_response_format(selected)
    assert actual == json.loads(encoded) == schema
    assert structured.decode_response('{"text":"{\\"status\\":\\"READY\\"}"}', validator) == '{"status":"READY"}'


def test_unknown_dialect_cannot_silently_select_a_different_validator():
    selected = response_format()
    selected["json_schema"]["schema"]["$schema"] = "https://invalid.example/unknown-draft"
    with pytest.raises(ValueError, match="unsupported dialect"):
        structured.validate_response_format(selected)


def test_validator_binds_the_same_schema_snapshot_as_native_argv():
    selected = response_format()
    schema, encoded, validator = structured.validate_response_format(selected)
    selected["json_schema"]["schema"]["properties"]["status"]["const"] = "MUTATED"
    assert schema == json.loads(encoded)
    assert structured.decode_response('{"structuredOutput":{"status":"READY"}}', validator) == '{"status":"READY"}'


@pytest.mark.parametrize("kind", ["depth", "bytes", "envelope"])
def test_provider_output_is_bounded_before_schema_validation(kind):
    _, _, validator = structured.validate_response_format(response_format())
    if kind == "depth":
        child = {}
        result = child
        for _ in range(34):
            child["nested"] = {}
            child = child["nested"]
        raw = json.dumps({"structuredOutput": result})
    elif kind == "bytes":
        raw = json.dumps({"structuredOutput": {"status": "x" * structured.MAX_RESPONSE_BYTES}})
    else:
        raw = " " * (structured.MAX_ENVELOPE_BYTES + 1)
    with pytest.raises(ValueError, match="bound"):
        structured.decode_response(raw, validator)


@pytest.mark.parametrize("kind", ["cycle", "depth", "bytes", "nonfinite"])
def test_schema_bounds_fail_before_validation_or_execution(kind):
    selected = response_format()
    schema = selected["json_schema"]["schema"]
    if kind == "cycle":
        schema["cycle"] = schema
    elif kind == "depth":
        child = schema
        for _ in range(34):
            child["child"] = {}
            child = child["child"]
    elif kind == "bytes":
        schema["description"] = "x" * structured.MAX_SCHEMA_BYTES
    else:
        schema["maximum"] = float("nan")
    with pytest.raises(ValueError):
        structured.validate_response_format(selected)


@pytest.mark.parametrize("command", [
    ["grok", "--json-schema", "{}"], ["grok", "--json-schema={}"],
    ["grok", "--json-schema"], ["grok", "--output-format", "plain"],
    ["grok", "--output-format=plain"], ["grok", "--"],
    ["grok", "--output-format", "json", "--output-format", "json"],
])
def test_custom_command_conflicts_fail_before_native_call(adapter, monkeypatch, command):
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **kw: pytest.fail("native call reached"))
    with pytest.raises(ValueError):
        adapter._get_grok_cli_provider().generate("prompt", response_format=response_format(), grok_cli_cmd=command)


def test_matching_custom_schema_is_not_duplicated(adapter, monkeypatch):
    schema = json.dumps(response_format()["json_schema"]["schema"], indent=2)
    def execute(argv, **kwargs):
        assert sum(part.startswith("--json-schema") for part in argv) == 1
        assert sum(part.startswith("--output-format") for part in argv) == 1
        return SimpleNamespace(returncode=0, stdout='{"structuredOutput":{"status":"READY"}}', stderr="")
    monkeypatch.setattr(adapter.subprocess, "run", execute)
    assert adapter._get_grok_cli_provider().generate("prompt", response_format=response_format(),
        grok_cli_cmd=["grok", "--json-schema=" + schema, "--output-format=json"]) == '{"status":"READY"}'


def test_custom_shell_route_does_not_silently_ignore_schema(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_run_cli_command", lambda *a, **kw: pytest.fail("native call reached"))
    with pytest.raises(ValueError, match="structured Grok CLI command"):
        adapter._get_grok_cli_provider().generate("prompt", response_format=response_format(),
            grok_cli_cmd="custom-wrapper --prompt {prompt}")


def test_canonical_planning_request_retains_exact_schema():
    from test.api.test_agent_supervisor_prompt_goal_planner import _request, _scan
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import build_prompt_goal_provider_request
    request = _request()
    prompt = build_prompt_goal_provider_request(request, _scan(request))
    selected = structured.planning_response_format(prompt)
    assert selected["json_schema"]["schema"] == json.loads(prompt)["response_schema"]
    assert selected["json_schema"]["strict"] is True
    assert structured.planning_response_format("Return READY") is None
    assert structured.planning_response_format('{"unrelated":"JSON"}') is None


@pytest.mark.parametrize("change", ["malformed", "stage", "missing", "extra", "required"])
def test_recognized_malformed_planner_request_never_downgrades_to_text(change):
    from test.api.test_agent_supervisor_prompt_goal_planner import _request, _scan
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import build_prompt_goal_provider_request
    request = _request()
    prompt = build_prompt_goal_provider_request(request, _scan(request))
    if change == "malformed":
        prompt = prompt[:-1]
    else:
        value = json.loads(prompt)
        if change == "stage":
            value["stage"] = "other"
        elif change == "missing":
            del value["response_schema"]
        elif change == "extra":
            value["response_schema"]["properties"]["authority"] = {"type": "boolean"}
        else:
            value["response_schema"]["required"] = [{"not": "a string"}]
        prompt = json.dumps(value)
    with pytest.raises(ValueError):
        structured.planning_response_format(prompt)


def canonical_response_format():
    from test.api.test_agent_supervisor_prompt_goal_planner import _request, _scan
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import build_prompt_goal_provider_request
    request = _request()
    return structured.planning_response_format(build_prompt_goal_provider_request(request, _scan(request)))


def test_native_projection_omits_only_canonical_top_level_id_without_mutation():
    selected = canonical_response_format()
    schema, canonical, validator = structured.validate_response_format(selected)
    before = copy.deepcopy(schema)
    wire, encoded, receipt = structured.native_schema_projection(schema)
    expected = dict(before)
    del expected["$id"]
    assert wire == expected
    assert schema == before == selected["json_schema"]["schema"]
    assert wire["definitions"] == before["definitions"]
    assert wire["bounds"] == before["bounds"]
    assert receipt == {
        "projection_id": "canonical-prompt-goal-native-id-omission@1",
        "top_level_id_omitted": True,
        "canonical_schema_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "canonical_schema_bytes": len(canonical.encode()),
        "native_wire_schema_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "native_wire_schema_bytes": len(encoded.encode()),
        "canonical_validation_preserved": True,
    }
    assert json.loads(encoded) == wire
    assert validator.schema == before


@pytest.mark.parametrize("change", ["generic", "id", "required", "nested_assertion", "budget_inconsistent", "extra", "nested_id"])
def test_native_projection_leaves_other_schemas_and_planner_lookalikes_unchanged(change):
    schema = canonical_response_format()["json_schema"]["schema"]
    if change == "generic":
        schema = response_format()["json_schema"]["schema"]
        schema["$id"] = "https://example.invalid/probe"
    elif change == "id":
        schema["$id"] = "https://example.invalid/other-planner"
    elif change == "required":
        schema["required"] = ["schema"]
    elif change == "nested_assertion":
        schema["definitions"]["task"]["additionalProperties"] = True
    elif change == "budget_inconsistent":
        schema["bounds"]["max_tasks"] += 1
    elif change == "extra":
        schema["description"] = "Another schema"
    else:
        schema["definitions"]["task"]["$id"] = "https://example.invalid/nested"
    before = copy.deepcopy(schema)
    wire, encoded, receipt = structured.native_schema_projection(schema)
    assert wire == before == schema == json.loads(encoded)
    assert receipt["projection_id"] == "identity@1"
    assert receipt["top_level_id_omitted"] is False
    assert receipt["canonical_schema_sha256"] == receipt["native_wire_schema_sha256"]


@pytest.mark.parametrize("valid", [True, False])
def test_projected_native_call_validates_original_canonical_schema(adapter, monkeypatch, valid):
    from test.api.test_agent_supervisor_prompt_goal_planner import _request, _scan, _proposal
    selected = canonical_response_format()
    original = copy.deepcopy(selected)
    proposal = _proposal(_scan(_request()))
    if not valid:
        proposal["schema"] = "provider-created-schema"
    def execute(argv, **kwargs):
        wire = json.loads(argv[argv.index("--json-schema") + 1])
        canonical = dict(selected["json_schema"]["schema"])
        del canonical["$id"]
        assert wire == canonical
        return SimpleNamespace(returncode=0, stdout=json.dumps({"structuredOutput": proposal, "stopReason": "end_turn"}), stderr="")
    monkeypatch.setattr(adapter.subprocess, "run", execute)
    if valid:
        text = adapter._get_grok_cli_provider().generate("authored proposal", response_format=selected)
        assert json.loads(text) == proposal
    else:
        with pytest.raises(adapter.LLMRouterError, match="structured response failed validation"):
            adapter._get_grok_cli_provider().generate("authored proposal", response_format=selected)
    assert selected == original


@pytest.mark.parametrize("command_schema", ["original", "assertion_changed", "matching_wire"])
def test_cli_override_compares_actual_projected_schema(adapter, monkeypatch, command_schema):
    from test.api.test_agent_supervisor_prompt_goal_planner import _request, _scan, _proposal
    selected = canonical_response_format()
    schema = copy.deepcopy(selected["json_schema"]["schema"])
    if command_schema != "original":
        del schema["$id"]
    if command_schema == "assertion_changed":
        schema["additionalProperties"] = True
    command = ["grok", "--json-schema", json.dumps(schema)]
    def execute(argv, **kwargs):
        assert command_schema == "matching_wire", "conflicting override reached native CLI"
        assert argv.count("--json-schema") == 1
        return SimpleNamespace(returncode=0, stdout=json.dumps({"structuredOutput": _proposal(_scan(_request()))}), stderr="")
    monkeypatch.setattr(adapter.subprocess, "run", execute)
    if command_schema == "matching_wire":
        adapter._get_grok_cli_provider().generate("authored proposal", response_format=selected, grok_cli_cmd=command)
    else:
        with pytest.raises(ValueError, match="schema conflicts"):
            adapter._get_grok_cli_provider().generate("authored proposal", response_format=selected, grok_cli_cmd=command)
