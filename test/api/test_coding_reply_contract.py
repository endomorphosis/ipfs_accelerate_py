"""Owner-selected coding schema preserves native decoding and input custody."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import coding_reply_contract as contract
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from test.api.test_semantic_router_integration import allocated
from test.api.test_semantic_router_translation import native, _reply


ACKNOWLEDGMENT = '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'
TRANSPORTS = [codec.TRANSPORT_SCHEMA, codec.COMPACT_TRANSPORT_SCHEMA]


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def test_legacy_contract_preserves_literal_bytes_and_requests_no_schema():
    text = '\n Literal $(private) {"$semantic_ref":"s0"} — unchanged.\n'
    assert contract.apply_coding_reply_contract(model_prompt=text, mode="legacy",
        purpose="planning", provider="grok_cli") == (text, None)


def test_coding_schema_is_closed_and_returns_an_independent_copy():
    schema = contract.coding_completion_schema()
    assert schema == {"type": "object", "properties": {
        "schema": {"type": "string", "enum": ["supervisor-coding-completion@1"]},
        "status": {"type": "string", "enum": ["candidate_ready"]}},
        "required": ["schema", "status"], "additionalProperties": False}
    schema["properties"]["status"]["enum"][0] = "completed"
    assert contract.coding_completion_schema()["properties"]["status"]["enum"] == ["candidate_ready"]


@pytest.mark.parametrize("reply", [ACKNOWLEDGMENT, '\n ' + ACKNOWLEDGMENT + '\t',
    '{"status":"candidate_ready", "schema":"supervisor-coding-completion@1"}'])
def test_closed_acknowledgment_is_validated_without_rewriting(reply):
    assert contract.validate_coding_completion_reply(reply) is reply


@pytest.mark.parametrize("reply", [
    "done", "", "null", "[]", '"candidate_ready"',
    '{"schema":"supervisor-coding-completion@1"}',
    '{"schema":"supervisor-coding-completion@1","status":"completed"}',
    '{"schema":"foreign","status":"candidate_ready"}',
    '{"schema":"supervisor-coding-completion@1","status":"candidate_ready","extra":"private-canary"}',
    '{"schema":"supervisor-coding-completion@1","schema":"supervisor-coding-completion@1","status":"candidate_ready"}',
    '{"schema":"supervisor-coding-completion@1","status":NaN}',
    '{"schema":"supervisor-coding-completion@1","status":Infinity}',
    ACKNOWLEDGMENT + ACKNOWLEDGMENT,
    ACKNOWLEDGMENT + " " * contract.MAX_COMPLETION_REPLY_BYTES,
    {"schema": "supervisor-coding-completion@1", "status": "candidate_ready"},
])
def test_invalid_acknowledgment_refuses_without_exporting_values(reply):
    with pytest.raises(ValueError, match="selected contract") as error:
        contract.validate_coding_completion_reply(reply)
    assert "private-canary" not in str(error.value)


def test_contract_attests_fixed_suffix_and_complete_final_input():
    original = "task source — literal"
    final, receipt = contract.apply_coding_reply_contract(model_prompt=original,
        mode="ordinary-completion@1", purpose="coding", provider="codex_cli")
    suffix = final[len(original):]
    assert final.startswith(original) and ACKNOWLEDGMENT in suffix
    assert receipt["model_prompt_before_sha256"] == _sha(original)
    assert receipt["model_prompt_after_sha256"] == _sha(final)
    assert receipt["instruction_sha256"] == _sha(suffix)
    assert receipt["model_prompt_after_bytes"] == len(final.encode())
    assert receipt["model_prompt_before_bytes"] + receipt["instruction_bytes"] == len(final.encode())
    assert receipt["response_validated"] is False
    assert receipt["native_output_schema_requested"] is True
    assert receipt["residual_task_family_selected"] is False
    for key in ("execution_authority", "completion_authority", "settlement_authority", "automatic_retry_admitted"):
        assert receipt[key] is False
    wire = json.dumps(contract.coding_completion_schema(), ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":"))
    assert receipt["native_output_schema_sha256"] == _sha(wire)
    assert receipt["native_output_schema_bytes"] == len(wire.encode())
    with pytest.raises(ValueError, match="byte bound"):
        contract.apply_coding_reply_contract(model_prompt="x" * contract.MAX_MODEL_PROMPT_BYTES,
            mode="ordinary-completion@1", purpose="coding", provider="codex_cli")


@pytest.fixture
def observed_provider(monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index

    state = {"reply": ACKNOWLEDGMENT, "schema_custody": "matched"}
    observed = []
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="codex_cli", model_name="pinned", catalog_revision="authored-fixture"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: object())

    def generate(prompt, **kwargs):
        observed.append((prompt, kwargs))
        observation = {"exit_code": 0, "prompt_tokens": 123, "completion_tokens": 7}
        if "codex_output_schema" in kwargs and state["schema_custody"] != "missing":
            wire = json.dumps(kwargs["codex_output_schema"], ensure_ascii=False, allow_nan=False,
                              sort_keys=True, separators=(",", ":"))
            observation.update(codex_output_schema_sha256=_sha(wire), codex_output_schema_bytes=len(wire.encode()))
            if state["schema_custody"] == "wrong_digest":
                observation["codex_output_schema_sha256"] = "0" * 64
            elif state["schema_custody"] == "wrong_bytes":
                observation["codex_output_schema_bytes"] += 1
        set_last_cli_observation("codex_cli", observation)
        if state.get("after_generate"):
            state["after_generate"]()
        return state["reply"]

    monkeypatch.setattr(llm_router, "generate_text", generate)
    return state, observed


def _run(repository, prompt, **kwargs):
    return runner.run(prompt=prompt, provider="codex_cli", model="pinned", timeout=10,
        max_output_tokens=256, semantic_repository=repository, **kwargs)


def _failure_receipt(capsys):
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    return next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")


@pytest.mark.parametrize("transport_schema", TRANSPORTS)
def test_default_route_preserves_existing_input_and_receipt(allocated, observed_provider, transport_schema):
    repository, workspace, prompt, _ = allocated
    state, observed = observed_provider
    state["reply"] = "existing literal summary"
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository,
                                                  transport_schema=transport_schema)
    output, receipt = _run(repository, prompt, semantic_transport_schema=transport_schema)
    expected, _ = runner.render_model_prompt(prompt=encoded.provider_prompt, purpose="coding",
                                             workspace=workspace, semantic_transport=True)
    assert output == state["reply"] and observed[0][0] == expected
    assert "coding_reply_contract" not in receipt
    assert "provider_invocation_policy" not in receipt
    assert "codex_output_schema" not in observed[0][1]
    assert receipt["semantic_translation"] == encoded.receipt


@pytest.mark.parametrize("transport_schema", TRANSPORTS)
def test_opt_in_contract_is_final_and_fully_accounted(allocated, observed_provider, transport_schema):
    repository, workspace, prompt, _ = allocated
    state, observed = observed_provider
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository,
                                                  transport_schema=transport_schema)
    before, _ = runner.render_model_prompt(prompt=encoded.provider_prompt, purpose="coding",
                                          workspace=workspace, semantic_transport=True)
    expected, contract_receipt = contract.apply_coding_reply_contract(model_prompt=before,
        mode="ordinary-completion@1", purpose="coding", provider="codex_cli")
    output, receipt = _run(repository, prompt, semantic_transport_schema=transport_schema,
                           coding_reply_mode="ordinary-completion@1")
    assert output == ACKNOWLEDGMENT
    assert observed[0][0] == expected
    assert observed[0][1]["codex_output_schema"] == contract.coding_completion_schema()
    assert receipt["model_prompt_sha256"] == contract_receipt["model_prompt_after_sha256"] == _sha(expected)
    assert receipt["model_prompt_bytes"] == len(expected.encode())
    assert receipt["native_prompt_sha256"] == _sha(prompt)
    assert receipt["router_prompt_sha256"] == _sha(encoded.provider_prompt)
    assert receipt["semantic_translation"] == encoded.receipt
    assert receipt["semantic_response_translation"]["structured_translation"] is False
    assert receipt["coding_reply_contract"] == {**contract_receipt, "response_validated": True}
    assert receipt["provider_invocation_policy"]["structured_output"]["response_schema_validated"] is True
    assert receipt["usage"]["codex_output_schema_sha256"] == contract_receipt["native_output_schema_sha256"]
    assert receipt["usage"]["codex_output_schema_bytes"] == contract_receipt["native_output_schema_bytes"]
    assert receipt["completion_authority"] is False


@pytest.mark.parametrize("mode,purpose,provider", [
    ("unsupported", "coding", "codex_cli"),
    ("ordinary-completion@1", "planning", "codex_cli"),
    ("ordinary-completion@1", "coding", "grok_cli"),
])
def test_mode_is_selected_by_owner_and_invalid_routes_do_not_dispatch(
        allocated, observed_provider, mode, purpose, provider):
    repository, _, prompt, _ = allocated
    _, observed = observed_provider
    with pytest.raises(ValueError, match="reply mode"):
        runner.run(prompt=prompt, provider=provider, model="grok-4.7" if provider == "grok_cli" else "pinned",
            timeout=10, max_output_tokens=256, purpose=purpose, semantic_repository=repository,
            coding_reply_mode=mode)
    assert observed == []


@pytest.mark.parametrize("custody", ["missing", "wrong_digest", "wrong_bytes"])
def test_schema_custody_is_required_and_failed_attempt_usage_is_retained(
        allocated, observed_provider, capsys, custody):
    repository, _, prompt, _ = allocated
    state, observed = observed_provider
    state["schema_custody"] = custody
    with pytest.raises(RuntimeError, match="schema invocation metadata"):
        _run(repository, prompt, coding_reply_mode="ordinary-completion@1")
    receipt = _failure_receipt(capsys)
    assert len(observed) == 1 and receipt["failure_phase"] == "coding_reply_validation"
    assert receipt["usage"]["prompt_tokens"] == 123 and receipt["usage"]["completion_tokens"] == 7
    assert receipt["coding_reply_contract"]["response_validated"] is False
    assert receipt["completion_authority"] is False


@pytest.mark.parametrize("transport_schema", TRANSPORTS)
@pytest.mark.parametrize("key", ["translation_cid", "task_id", "semantic_root_cid", "scope_cid", "extra", "missing"])
def test_bad_reserved_reply_still_fails_in_native_decoder_before_completion_contract(
        allocated, observed_provider, capsys, transport_schema, key):
    repository, _, prompt, _ = allocated
    state, observed = observed_provider
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository,
                                                  transport_schema=transport_schema)
    reply, _ = _reply(encoded)
    if key == "missing":
        del reply["task_family"]
    else:
        reply[key] = "private-returned-value-canary"
    state["reply"] = json.dumps(reply)
    with pytest.raises(codec.SemanticTranslationError) as error:
        _run(repository, prompt, semantic_transport_schema=transport_schema,
             coding_reply_mode="ordinary-completion@1")
    assert error.value.reason_code == "response_envelope_binding_mismatch"
    captured = capsys.readouterr().out
    assert "private-returned-value-canary" not in captured
    receipt = next(json.loads(line) for line in captured.splitlines()
        if line.startswith("{") and json.loads(line).get("schema") == "router-implementation-invocation@1")
    assert len(observed) == 1 and receipt["failure_phase"] == "semantic_response_decode"
    assert receipt["semantic_response_failure"]["reason_code"] == "response_envelope_binding_mismatch"
    assert receipt["coding_reply_contract"]["response_validated"] is False
    assert receipt["usage"]["prompt_tokens"] == 123 and receipt["usage"]["completion_tokens"] == 7


def test_valid_reserved_candidate_cannot_substitute_for_ordinary_editing_acknowledgment(
        allocated, observed_provider, capsys):
    repository, _, prompt, _ = allocated
    state, _ = observed_provider
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository)
    reply, _ = _reply(encoded)
    state["reply"] = json.dumps(reply)
    with pytest.raises(ValueError, match="selected contract"):
        _run(repository, prompt, coding_reply_mode="ordinary-completion@1")
    receipt = _failure_receipt(capsys)
    assert receipt["semantic_response_translation"]["native_grammar_validated"] is True
    assert receipt["failure_phase"] == "coding_reply_validation"
    assert receipt["coding_reply_contract"]["response_validated"] is False


@pytest.mark.parametrize("reply", ["done", '{"schema":"supervisor-coding-completion@1","status":"candidate_ready","private-key":"private-canary"}'])
def test_unconstrained_provider_cannot_bypass_reply_validation(allocated, observed_provider, capsys, reply):
    repository, _, prompt, _ = allocated
    state, _ = observed_provider
    state["reply"] = reply
    with pytest.raises(ValueError, match="selected contract"):
        _run(repository, prompt, coding_reply_mode="ordinary-completion@1")
    captured = capsys.readouterr().out
    assert "private-canary" not in captured and "private-key" not in captured
    receipt = next(json.loads(line) for line in captured.splitlines()
        if line.startswith("{") and json.loads(line).get("schema") == "router-implementation-invocation@1")
    assert receipt["failure_phase"] == "coding_reply_validation"
    assert receipt["coding_reply_contract"]["response_validated"] is False


def test_actual_cli_receives_closed_coding_schema_and_final_prompt_without_provider_call(
        allocated, tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.llm_allocation import duckdb_store, intelligence_index

    repository, workspace, prompt, _ = allocated
    capture = tmp_path / "authored-native-capture.json"
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
        "assert schema['additionalProperties'] is False\n"
        "assert schema['required']==['schema','status']\n"
        "assert schema['properties']['status']['enum']==['candidate_ready']\n"
        "assert stat.S_IMODE(schema_path.stat().st_mode)==0o600\n"
        "prompt=sys.stdin.read()\n"
        f"pathlib.Path({str(capture)!r}).write_text(json.dumps({{'schema_path':str(schema_path),'schema_sha256':hashlib.sha256(raw).hexdigest(),'schema_bytes':len(raw),'prompt':prompt}}))\n"
        f"pathlib.Path(option('--output-last-message')).write_text({ACKNOWLEDGMENT!r})\n"
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':15,'output_tokens':4}}))\n"
    )
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: llm_router._get_codex_cli_provider())
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="codex_cli", model_name="pinned", catalog_revision="authored-fixture"))
    monkeypatch.setattr(duckdb_store, "_DEFAULT_STORE", duckdb_store.AllocationStore(tmp_path / "allocation.duckdb"))
    output, receipt = _run(repository, prompt, semantic_transport_schema=codec.COMPACT_TRANSPORT_SCHEMA,
                          coding_reply_mode="ordinary-completion@1")
    observed = json.loads(capture.read_text())
    assert output == ACKNOWLEDGMENT
    assert receipt["model_prompt_sha256"] == _sha(observed["prompt"])
    assert receipt["model_prompt_bytes"] == len(observed["prompt"].encode())
    assert receipt["coding_reply_contract"]["native_output_schema_sha256"] == observed["schema_sha256"]
    assert receipt["coding_reply_contract"]["native_output_schema_bytes"] == observed["schema_bytes"]
    assert receipt["usage"]["codex_output_schema_sha256"] == observed["schema_sha256"]
    assert receipt["usage"]["codex_output_schema_bytes"] == observed["schema_bytes"]
    assert receipt["usage"]["prompt_tokens"] == 15 and receipt["usage"]["completion_tokens"] == 4
    assert receipt["coding_reply_contract"]["response_validated"] is True
    assert not Path(observed["schema_path"]).exists()
    assert receipt["completion_authority"] is False
