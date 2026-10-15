"""Literal Codex transport with authored CLI output, never a real model call.

Native Git/worktree and router bindings run normally in the integration cases.
The subprocess replacement produces only the authored final file and usage.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_dialogue import (
    SemanticDecisionCall,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_provider import SharedRouterDecisionProvider
from ipfs_accelerate_py.cli_runtime import cli_metadata


@pytest.fixture
def authored_codex(monkeypatch, tmp_path):
    original_run = subprocess.run
    original_which = llm_router.shutil.which
    state = {"final": b"authored final", "stdout": "", "returncode": 0, "calls": [],
        "files": [], "schema_files": [], "schemas": [], "error": None}
    monkeypatch.setattr(llm_router.shutil, "which", lambda name, *args, **kwargs:
        "/authored-fixture/codex" if name == "codex" else original_which(name, *args, **kwargs))
    monkeypatch.setattr(llm_router, "_response_cache_enabled", lambda: False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-authored-codex-home"))
    monkeypatch.setenv("ipfs_accelerate_py_CODEX_SANDBOX", "workspace-write")

    def cli(command, **kwargs):
        if command[0] != "codex":
            return original_run(command, **kwargs)
        state["calls"].append((list(command), dict(kwargs)))
        final = Path(command[command.index("--output-last-message") + 1])
        state["files"].append(final)
        if "--output-schema" in command:
            path = Path(command[command.index("--output-schema") + 1])
            raw = path.read_bytes()
            state["schema_files"].append(path)
            state["schemas"].append({"sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw), "mode": path.stat().st_mode & 0o777,
                "value": json.loads(raw)})
        if state["error"] is not None:
            raise state["error"]
        if state["final"] is None:
            final.unlink()
        else:
            final.write_bytes(state["final"])
        return SimpleNamespace(returncode=state["returncode"], stdout=state["stdout"], stderr="")

    def remember(provider, stdout, stderr, *, extra):
        assert provider == "codex_cli"
        # Authored accounting metadata; this is not actual provider usage.
        cli_metadata.set_last_cli_observation(provider, {
            "exit_code": extra["exit_code"], "model_id": extra["model_id"],
            "prompt_tokens": 11, "completion_tokens": 5,
            **{key: extra[key] for key in ("codex_output_schema_sha256", "codex_output_schema_bytes")
                if key in extra},
        })

    monkeypatch.setattr(llm_router.subprocess, "run", cli)
    monkeypatch.setattr(cli_metadata, "remember_cli_run", remember)
    return state


@pytest.mark.parametrize("literal", [
    '  {"source":"<script>safe</script>","entity":"&quot;&amp;&lt;"}\r\n',
    '{"code":"if (left < right && right > 0):\\n    return \\\"&amp;\\\""}\n',
    '```json\n{"source":"<p>literal markup</p>"}\n```\n',
    '"quoted JSON string with <tag> and &quot;"\n',
    '{"unicode":"λ雪","escaped":"\\u003cnode\\u003e"}\n',
])
def test_literal_cli_final_file_preserves_all_utf8_bytes(authored_codex, literal):
    authored_codex["final"] = literal.encode("utf-8")
    authored_codex["stdout"] = json.dumps({"type": "agent_message", "text": "different JSONL message"}) + "\n"
    provider = llm_router._get_codex_cli_provider()
    result = llm_router.generate_text(
        "authored coding input", provider="codex_cli", model_name="fixture-model",
        provider_instance=provider, allow_local_fallback=False,
        allow_cross_provider_fallback=False, codex_cli_literal_response=True,
        timeout=2, reasoning_effort="high",
    )
    assert result.encode("utf-8") == authored_codex["final"]
    assert len(authored_codex["calls"]) == 1
    command, settings = authored_codex["calls"][0]
    assert command[:2] == ["codex", "exec"]
    assert command[-1] == "-" and "--json" in command
    assert command[command.index("-m") + 1] == "fixture-model"
    assert settings["input"] == "authored coding input"
    assert all(not path.exists() for path in authored_codex["files"])


@pytest.mark.parametrize("opt_in", [{}, {"codex_cli_literal_response": False}])
def test_default_codex_prose_cleanup_and_jsonl_precedence_remain(authored_codex, opt_in):
    authored_codex["final"] = b"unused final"
    authored_codex["stdout"] = json.dumps({"type": "agent_message", "text": " <b>Hello &amp; goodbye</b> "})
    result = llm_router._get_codex_cli_provider().generate("input", model_name="fixture-model", **opt_in)
    assert result == "Hello & goodbye"
    assert len(authored_codex["calls"]) == 1


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false"])
def test_literal_opt_in_refuses_non_boolean_before_cli_dispatch(authored_codex, value):
    with pytest.raises(ValueError, match="explicit Boolean"):
        llm_router._get_codex_cli_provider().generate("input", codex_cli_literal_response=value)
    with pytest.raises(ValueError, match="explicit Boolean"):
        llm_router.generate_text("input", provider="codex_cli", codex_cli_literal_response=value)
    with pytest.raises(ValueError, match="explicit Boolean"):
        runner.run(prompt="input", provider="codex_cli", model="fixture-model", timeout=2,
            max_output_tokens=128, codex_cli_literal_response=value)
    assert authored_codex["calls"] == []


@pytest.mark.parametrize("final", [None, b"\xff"])
def test_unavailable_literal_final_file_refuses_jsonl_fallback_and_retains_exit(authored_codex, final):
    authored_codex["final"] = final
    authored_codex["stdout"] = json.dumps({"type": "agent_message", "text": "do not substitute this"})
    with pytest.raises(llm_router.LLMRouterError, match="literal final response unavailable"):
        llm_router.generate_text("input", provider="codex_cli", model_name="fixture-model",
            provider_instance=llm_router._get_codex_cli_provider(), allow_local_fallback=False,
            allow_cross_provider_fallback=False, codex_cli_literal_response=True)
    assert len(authored_codex["calls"]) == 1
    assert cli_metadata.get_last_cli_observation("codex_cli")["exit_code"] == 0
    assert all(not path.exists() for path in authored_codex["files"])


def test_empty_literal_final_file_is_not_replaced_from_jsonl(authored_codex):
    authored_codex["final"] = b""
    authored_codex["stdout"] = json.dumps({"type": "agent_message", "text": "other reply"})
    assert llm_router._get_codex_cli_provider().generate("input", codex_cli_literal_response=True) == ""
    assert len(authored_codex["calls"]) == 1


def test_literal_codex_opt_in_does_not_enter_grok_route(authored_codex):
    with pytest.raises(ValueError, match="requires the Codex CLI provider"):
        llm_router.generate_text("input", provider="grok_cli", model_name="fixture-model",
            codex_cli_literal_response=True)
    assert authored_codex["calls"] == []


@pytest.fixture
def allocated_router(authored_codex, monkeypatch, tmp_path):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        DATABASE_PROGRAM_ENV_NAMES, STATE_CREDENTIAL_ENV_NAMES,
    )
    for name in (*DATABASE_PROGRAM_ENV_NAMES, *STATE_CREDENTIAL_ENV_NAMES,
                 "IPFS_ACCELERATE_AGENT_STATE_OWNER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / "mod.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(repository), "add", "mod.py"], check=True)
    subprocess.run(["git", "-C", str(repository), "-c", "user.name=AuthoredFixture",
        "-c", "user.email=authored@example.invalid", "commit", "-qm", "authored baseline"], check=True)
    workspace = tmp_path / "allocated"
    subprocess.run(["git", "-C", str(repository), "worktree", "add", "--detach", "-q", str(workspace)], check=True)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="codex_cli", model_name="fixture-model", catalog_revision="authored-fixture"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: llm_router._get_codex_cli_provider())
    counter = runner.PromptTokenCounter(provider="codex_cli", model="fixture-model",
        tokenizer_id="authored:byte-estimate", tokenizer_revision="fixture-v1",
        special_token_policy="authored-only", count_tokens=lambda text: len(text.encode()),
        count_kind="estimate", compatibility_basis="authored_fixture_only")
    return repository, workspace, counter


@pytest.mark.parametrize("literal", [False, True])
def test_actual_runner_propagates_per_call_mode_without_changing_native_input(allocated_router, authored_codex, literal):
    _, workspace, counter = allocated_router
    final = ' {"code":"<tag> &amp;"}\r\n'
    authored_codex["final"] = final.encode()
    output, receipt = runner.run(prompt="original authored task", provider="codex_cli",
        model="fixture-model", timeout=2, max_output_tokens=128, prompt_token_counter=counter,
        codex_cli_literal_response=literal)
    assert output == (final if literal else llm_router._clean_codex_output(final))
    model_prompt, _ = runner.render_model_prompt(prompt="original authored task", purpose="coding", workspace=workspace)
    assert authored_codex["calls"][0][1]["input"] == model_prompt
    assert receipt["model_prompt_sha256"] == hashlib.sha256(model_prompt.encode()).hexdigest()
    assert receipt["router_calls"] == 1 and receipt["completion_authority"] is False
    assert receipt["provider_output_token_cap_enforced"] is False
    assert len(authored_codex["calls"]) == 1


def test_actual_semantic_codex_callback_preserves_code_and_receipt_binding(allocated_router, authored_codex):
    repository, workspace, counter = allocated_router
    prompt = "authored bound decision request"
    rendered, _ = runner.render_model_prompt(prompt=prompt, purpose="coding", workspace=workspace)
    call = SemanticDecisionCall(call_id="authored-call", session_id="authored-session", generation=1,
        binding_sha256="a" * 64, provider="codex_cli", model="fixture-model", prompt=prompt,
        timeout=2, max_output_tokens=128, model_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest())
    final = ' {"code":"<script>literal</script>","escaped":"&quot;"}\r\n'
    authored_codex["final"] = final.encode()
    provider = SharedRouterDecisionProvider(provider="codex_cli", model="fixture-model",
        repository=repository, workspace=workspace, prompt_token_counter=counter)
    reply = provider(call)
    assert reply.text.encode() == authored_codex["final"]
    assert reply.receipt["model_prompt_sha256"] == call.model_prompt_sha256
    assert reply.receipt["native_invocation"]["model_prompt_sha256"] == call.model_prompt_sha256
    assert reply.receipt["provider_called"] is True and reply.receipt["completion_authority"] is False
    assert len(authored_codex["calls"]) == 1


def test_semantic_rendered_input_mismatch_still_refuses_before_dispatch(allocated_router, authored_codex):
    repository, workspace, counter = allocated_router
    call = SemanticDecisionCall(call_id="authored-call", session_id="authored-session", generation=1,
        binding_sha256="a" * 64, provider="codex_cli", model="fixture-model", prompt="authored input",
        timeout=2, max_output_tokens=128, model_prompt_sha256="f" * 64)
    provider = SharedRouterDecisionProvider(provider="codex_cli", model="fixture-model",
        repository=repository, workspace=workspace, prompt_token_counter=counter)
    with pytest.raises(ValueError, match="complete rendered model prompt"):
        provider(call)
    assert authored_codex["calls"] == []


def literal_schema():
    return {"type": "object", "additionalProperties": False, "required": ["code", "entity"],
        "properties": {"code": {"type": "string"}, "entity": {"type": "string"}}}


def test_literal_final_schema_validation_preserves_bytes_and_ignores_different_valid_jsonl(authored_codex):
    final = '  {"code":"<script>λ雪</script>","entity":"&quot;&amp;\\u003c"}\r\n'
    authored_codex["final"] = final.encode()
    authored_codex["stdout"] = json.dumps({"type": "agent_message",
        "text": '{"code":"different valid JSONL","entity":"other"}'})
    result = llm_router.generate_text("authored schema input", provider="codex_cli",
        model_name="fixture-model", provider_instance=llm_router._get_codex_cli_provider(),
        allow_local_fallback=False, allow_cross_provider_fallback=False,
        codex_cli_literal_response=True, codex_output_schema=literal_schema())
    assert result.encode() == authored_codex["final"]
    assert len(authored_codex["calls"]) == len(authored_codex["schemas"]) == 1
    schema = authored_codex["schemas"][0]
    assert schema["mode"] == 0o600 and schema["value"] == literal_schema()
    observed = cli_metadata.get_last_cli_observation("codex_cli")
    assert observed["codex_output_schema_sha256"] == schema["sha256"]
    assert observed["codex_output_schema_bytes"] == schema["bytes"]
    assert observed["exit_code"] == 0
    assert all(not p.exists() for p in authored_codex["files"] + authored_codex["schema_files"])


@pytest.mark.parametrize("final", [None, b"\xff", b"", b"{bad json}", b"{}",
    b'{"code":1,"entity":"valid"}', b'{"code":"a","code":"b","entity":"valid"}'])
def test_literal_schema_refuses_bad_final_even_when_jsonl_is_valid(authored_codex, final):
    authored_codex["final"] = final
    authored_codex["stdout"] = json.dumps({"type": "agent_message",
        "text": '{"code":"valid fallback must not be used","entity":"valid"}'})
    with pytest.raises(llm_router.LLMRouterError, match="^Codex CLI structured response failed validation$"):
        llm_router.generate_text("authored schema refusal", provider="codex_cli", model_name="fixture-model",
            provider_instance=llm_router._get_codex_cli_provider(), allow_local_fallback=False,
            allow_cross_provider_fallback=False, codex_cli_literal_response=True,
            codex_output_schema=literal_schema())
    assert len(authored_codex["calls"]) == 1
    observed = cli_metadata.get_last_cli_observation("codex_cli")
    assert observed["exit_code"] == 0 and observed["prompt_tokens"] == 11
    assert observed["codex_output_schema_sha256"] == authored_codex["schemas"][0]["sha256"]
    assert all(not p.exists() for p in authored_codex["files"] + authored_codex["schema_files"])


@pytest.mark.parametrize("exception_class", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("prior_environment", [None, "owner-original"])
def test_callback_interruption_runs_after_exact_environment_restoration(
    allocated_router, authored_codex, monkeypatch, exception_class, prior_environment,
):
    _, _, counter = allocated_router
    keys = ("ipfs_accelerate_py_CODEX_SANDBOX", "PYTHONDONTWRITEBYTECODE")
    for name in keys:
        if prior_environment is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, prior_environment)
    interruption = exception_class("authored callback interruption")
    receipts = []
    def observe(receipt):
        assert all(os.environ.get(name) == prior_environment for name in keys)
        receipts.append(receipt)
        raise interruption
    with pytest.raises(exception_class) as caught:
        runner.run(prompt="authored interrupted observer", provider="codex_cli", model="fixture-model",
            timeout=2, max_output_tokens=128, prompt_token_counter=counter, receipt_observer=observe,
            codex_cli_literal_response=True)
    assert caught.value is interruption
    assert len(receipts) == len(authored_codex["calls"]) == 1
    assert receipts[0]["status"] == "provider_returned"
    assert receipts[0]["usage"]["prompt_tokens"] == 11
    assert all(os.environ.get(name) == prior_environment for name in keys)
    assert all(not p.exists() for p in authored_codex["files"])


@pytest.mark.parametrize("provider_exception", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("callback_exception", [KeyboardInterrupt, SystemExit])
def test_callback_cannot_replace_original_pending_provider_interruption(
    allocated_router, authored_codex, monkeypatch, provider_exception, callback_exception,
):
    _, _, counter = allocated_router
    monkeypatch.setenv("ipfs_accelerate_py_CODEX_SANDBOX", "owner-sandbox")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "owner-bytecode")
    original = provider_exception("authored provider interruption")
    authored_codex["error"] = original
    receipts = []
    def observe(receipt):
        assert os.environ["ipfs_accelerate_py_CODEX_SANDBOX"] == "owner-sandbox"
        assert os.environ["PYTHONDONTWRITEBYTECODE"] == "owner-bytecode"
        receipts.append(receipt)
        raise callback_exception("secondary authored callback interruption")
    with pytest.raises(provider_exception) as caught:
        runner.run(prompt="authored interrupted provider", provider="codex_cli", model="fixture-model",
            timeout=2, max_output_tokens=128, prompt_token_counter=counter, receipt_observer=observe,
            codex_cli_literal_response=True)
    assert caught.value is original
    assert len(receipts) == len(authored_codex["calls"]) == 1
    assert receipts[0]["status"] == "failed" and receipts[0]["error_type"] == provider_exception.__name__
    assert "exit_code" not in receipts[0]["usage"]
    assert os.environ["ipfs_accelerate_py_CODEX_SANDBOX"] == "owner-sandbox"
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "owner-bytecode"
    assert all(not p.exists() for p in authored_codex["files"])


@pytest.mark.parametrize("exception_class", [KeyboardInterrupt, SystemExit])
def test_successful_run_inside_handled_exception_still_propagates_callback_interruption(
    allocated_router, authored_codex, monkeypatch, exception_class,
):
    _, _, counter = allocated_router
    monkeypatch.setenv("ipfs_accelerate_py_CODEX_SANDBOX", "owner-sandbox")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "owner-bytecode")
    interruption = exception_class("authored fresh callback interruption")
    receipts = []
    def observe(receipt):
        assert os.environ["ipfs_accelerate_py_CODEX_SANDBOX"] == "owner-sandbox"
        assert os.environ["PYTHONDONTWRITEBYTECODE"] == "owner-bytecode"
        receipts.append(receipt)
        raise interruption
    try:
        raise ValueError("authored outer handled exception")
    except ValueError:
        with pytest.raises(exception_class) as caught:
            runner.run(prompt="successful task inside handled exception", provider="codex_cli",
                model="fixture-model", timeout=2, max_output_tokens=128,
                prompt_token_counter=counter, receipt_observer=observe, codex_cli_literal_response=True)
    assert caught.value is interruption
    assert len(receipts) == len(authored_codex["calls"]) == 1
    assert receipts[0]["status"] == "provider_returned"
    assert receipts[0]["usage"]["prompt_tokens"] == 11
    assert os.environ["ipfs_accelerate_py_CODEX_SANDBOX"] == "owner-sandbox"
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "owner-bytecode"


def test_automatic_offline_factory_uses_restored_explicit_estimate_contract(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer as automatic
    def unavailable():
        raise RuntimeError("authored unavailable installed tokenizer")
    monkeypatch.setattr(automatic, "_installed_tiktoken", unavailable)
    counter, provenance = automatic.resolve_prompt_tokenizer(provider="codex_cli", model="unknown-authored-model")
    assert type(counter) is runner.PromptTokenCounter
    assert counter.count_kind == provenance["count_kind"] == "estimate"
    assert counter.compatibility_basis == provenance["compatibility_basis"] == "utf8_byte_estimate"
    assert counter.count_field == "estimated_input_tokens"
    assert counter.count("λ雪") == 2 and counter.count("") == 1
    assert provenance["provider_usage"] is False
    assert provenance["remote_tokenizer_compatibility_verified"] is False


@pytest.mark.parametrize("kind", ["tokenizer", "estimate"])
def test_schema_counts_and_world_projection_preserve_count_quality_labels(kind):
    counter = runner.PromptTokenCounter(provider="codex_cli", model="fixture-model",
        tokenizer_id="authored:byte-counter", tokenizer_revision="fixture-v1",
        special_token_policy="authored-only", count_tokens=lambda text: len(text.encode()),
        count_kind=kind, compatibility_basis="authored_fixture_only")
    assert set(counter.binding()) == {"provider", "model", "tokenizer_id", "tokenizer_revision", "special_token_policy"}
    native, candidate, schema = "long native input", "small", literal_schema()
    schema_bytes = json.dumps(schema, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()
    total = len(candidate.encode()) + len(schema_bytes)
    selected, receipt = runner.select_model_prompt(native_complete_prompt=native,
        candidate_complete_prompt=candidate, provider="codex_cli", model="fixture-model",
        token_counter=counter, native_output_schema=schema, max_input_tokens=total)
    assert selected == candidate
    assert receipt["provider_framing_measured"] is False
    assert receipt["provider_tokenizer_equivalence_verified"] is False
    assert receipt["total_session_token_savings"] is None
    world = runner._measure_world_packet_prompt(model_prompt=candidate, provider="codex_cli",
        model="fixture-model", token_counter=counter, max_input_tokens=total, native_output_schema=schema)
    if kind == "estimate":
        assert receipt["measurement_status"] == world["measurement_status"] == "estimated"
        assert receipt["selected_input_tokens"] is world["selected_input_tokens"] is None
        assert receipt["selected_content_tokens"] is world["selected_content_tokens"] is None
        assert receipt["native_output_schema_tokens"] is world["native_output_schema_tokens"] is None
        assert receipt["selected_estimated_input_tokens"] == world["selected_estimated_input_tokens"] == total
        assert receipt["estimated_native_output_schema_tokens"] == world["estimated_native_output_schema_tokens"] == len(schema_bytes)
        assert world["measurement_kind"] == "declared_local_estimate"
        assert receipt["token_budget_guaranteed"] is world["token_budget_guaranteed"] is False
    else:
        assert receipt["measurement_status"] == world["measurement_status"] == "measured"
        assert receipt["selected_input_tokens"] == world["selected_input_tokens"] == total
        assert receipt["native_output_schema_tokens"] == world["native_output_schema_tokens"] == len(schema_bytes)
        assert "selected_estimated_input_tokens" not in receipt
        assert world["measurement_kind"] == "declared_local_tokenizer"
    with pytest.raises(ValueError, match="exceeds its token budget"):
        runner.select_model_prompt(native_complete_prompt=candidate, candidate_complete_prompt=None,
            provider="codex_cli", model="fixture-model", token_counter=counter,
            native_output_schema=schema, max_input_tokens=total - 1)


@pytest.mark.parametrize("option,value", [("count_kind", None), ("count_kind", True),
    ("count_kind", "unknown"), ("compatibility_basis", None), ("compatibility_basis", " "),
    ("compatibility_basis", " padded ")])
def test_counter_compatibility_fields_cannot_weaken_existing_binding_checks(option, value):
    values = {"provider":"codex_cli", "model":"fixture-model", "tokenizer_id":"authored:counter",
        "tokenizer_revision":"fixture-v1", "special_token_policy":"authored-only",
        "count_tokens":lambda text: len(text.encode()), option:value}
    with pytest.raises(ValueError):
        runner.PromptTokenCounter(**values)


@pytest.mark.parametrize("foreign_argument,value", [
    ("native_prompt", "different task"), ("suffix", "mandatory instruction"),
    ("semantic_prompt", None),
])
def test_prompt_selector_rejects_mixed_forms_before_counting(foreign_argument, value):
    counted = []
    counter = runner.PromptTokenCounter(provider="codex_cli", model="fixture-model",
        tokenizer_id="authored:counter", tokenizer_revision="fixture-v1",
        special_token_policy="authored-only", count_tokens=lambda text: counted.append(text) or 1)
    with pytest.raises(TypeError, match="mixed complete and legacy"):
        runner.select_model_prompt(native_complete_prompt="original complete task",
            candidate_complete_prompt=None, provider="codex_cli", model="fixture-model",
            token_counter=counter, **{foreign_argument: value})
    assert counted == []
