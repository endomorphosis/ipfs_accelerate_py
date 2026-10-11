"""Offline complete-input selection; authored tokenizer, no remote model calls."""
import hashlib
import io
import json
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import coding_reply_contract as reply
from ipfs_accelerate_py.agent_supervisor.runtime.local_prompt_tokenizer import load_owner_prompt_tokenizer
from test.api.test_local_prompt_tokenizer import make_binding, MODEL
from test.api.test_semantic_router_translation import native  # noqa: F401
from test.api.test_semantic_metadata_dispatch import dispatch_case  # noqa: F401


@pytest.fixture
def tokenizer(tmp_path):
    artifact, digest, _ = make_binding(tmp_path)
    local = load_owner_prompt_tokenizer(artifact=artifact, expected_sha256=digest,
                                        provider="codex_cli", model=MODEL)
    return runner.PromptTokenCounter(**local.binding(), count_tokens=local.count)


def select(tokenizer, native, candidate, **kwargs):
    return runner.select_model_prompt(native_complete_prompt=native,
        candidate_complete_prompt=candidate, provider="codex_cli", model=MODEL,
        token_counter=tokenizer, **kwargs)


def complete(body, workspace):
    rendered, _ = runner.render_model_prompt(prompt=body, purpose="coding", workspace=workspace)
    return reply.apply_coding_reply_contract(model_prompt=rendered,
        mode="ordinary-completion@1", purpose="coding", provider="codex_cli")[0]


def test_shrinking_counts_whole_prompt_and_separate_schema(tokenizer, tmp_path):
    suffix = "\nLiteral task: preserve café 法律 /app/report.txt."
    native = complete("hello world " * 80 + suffix, tmp_path)
    candidate = complete('{"schema":"capsule@1","body":"hello"}' + suffix, tmp_path)
    schema = reply.coding_completion_schema()
    schema_wire = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    limit = tokenizer.count(candidate) + tokenizer.count(schema_wire)
    selected, receipt = select(tokenizer, native, candidate, native_output_schema=schema,
                               max_input_tokens=limit)
    assert selected == candidate and suffix in selected
    assert receipt["selected_representation"] == "semantic"
    assert receipt["selected_content_tokens"] == tokenizer.count(candidate)
    assert receipt["selected_input_tokens"] == limit
    assert receipt["native_output_schema_tokens"] == tokenizer.count(schema_wire)
    assert receipt["candidates"]["native"]["input_tokens"] > limit
    assert receipt["total_session_token_savings"] is None
    assert receipt["completion_authority"] is False
    assert receipt["provider_tokenizer_equivalence_verified"] is False
    with pytest.raises(ValueError, match="token budget"):
        select(tokenizer, native, candidate, native_output_schema=schema, max_input_tokens=limit - 1)


def test_translation_dictionary_growth_falls_back_to_literal_native(tokenizer, tmp_path):
    native = complete("hello world café 法律", tmp_path)
    # The abbreviated body is smaller, but its complete transport is larger.
    candidate = complete(json.dumps({"body": "hello", "dictionary": ["hello world"] * 70}), tmp_path)
    selected, receipt = select(tokenizer, native, candidate)
    assert selected == native
    assert receipt["selection_reason"] == "native_not_larger"
    assert receipt["candidates"]["semantic"]["content_tokens"] > receipt["selected_content_tokens"]
    assert receipt["candidates"]["semantic"]["model_prompt_sha256"] == hashlib.sha256(candidate.encode()).hexdigest()


def test_equal_counts_retain_native(tokenizer):
    selected, receipt = select(tokenizer, "hello world", "café 法律")
    assert selected == "hello world" and receipt["selection_reason"] == "native_not_larger"


def test_metadata_view_competes_with_original_semantic_transport(tokenizer):
    selected, receipt = select(tokenizer, "hello " * 30, "hello " * 10,
        semantic_complete_prompt="hello world", candidate_representation="semantic_metadata")
    assert selected == "hello world" and receipt["selected_representation"] == "semantic"
    assert set(receipt["candidates"]) == {"native", "semantic", "semantic_metadata"}


def test_unknown_or_failed_count_keeps_native_but_cannot_admit_budget(tokenizer):
    selected, receipt = select(None, "hello world", "hello")
    assert selected == "hello world" and receipt["measurement_status"] == "unavailable"
    with pytest.raises(ValueError, match="working pinned local counter"):
        select(None, "hello world", "hello", max_input_tokens=100)

    def broken(_):
        raise RuntimeError("private tokenizer exception details")

    failed = runner.PromptTokenCounter(**tokenizer.binding(), count_tokens=broken)
    selected, receipt = select(failed, "hello world", "hello")
    assert selected == "hello world" and receipt["measurement_status"] == "failed"
    assert receipt["measurement_error_type"] == "RuntimeError"
    assert "private" not in json.dumps(receipt)
    with pytest.raises(ValueError, match="working pinned local counter"):
        select(failed, "hello world", "hello", max_input_tokens=100)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_budget_refuses(tokenizer, limit):
    with pytest.raises(ValueError, match="positive integer"):
        select(tokenizer, "hello world", "hello", max_input_tokens=limit)


def test_wrong_provider_model_binding_refuses(tokenizer):
    with pytest.raises(ValueError, match="pinned provider/model"):
        runner.select_model_prompt(native_complete_prompt="hello", candidate_complete_prompt=None,
            provider="codex_cli", model="other", token_counter=tokenizer)


@pytest.fixture
def bound_dispatch(dispatch_case, tmp_path):
    assets = tmp_path / "tokenizer-assets"
    assets.mkdir()
    artifact, digest, _ = make_binding(assets, model="authored-model")
    loaded = load_owner_prompt_tokenizer(artifact=artifact, expected_sha256=digest,
        provider="codex_cli", model="authored-model")
    return dispatch_case, artifact, digest, loaded


def invoke(bound_dispatch, **kwargs):
    case, artifact, digest, _ = bound_dispatch
    return runner.run(prompt=case["prompt"], provider="codex_cli", model="authored-model",
        timeout=30, max_output_tokens=4096, semantic_repository=case["repository"],
        coding_reply_mode="ordinary-completion@1", prompt_tokenizer_artifact=artifact,
        prompt_tokenizer_sha256=digest, **kwargs)


@pytest.mark.parametrize("metadata_view", ["legacy", "common-bindings@1"])
def test_real_runner_selection_retains_reply_schema_and_actual_input_receipt(bound_dispatch, metadata_view):
    case, _, _, loaded = bound_dispatch
    _, receipt = invoke(bound_dispatch, semantic_metadata_view=metadata_view)
    delivered, options = case["calls"][-1]
    selection = receipt["prompt_selection"]
    schema = options["codex_output_schema"]
    assert schema == reply.coding_completion_schema()
    assert selection["selected_content_tokens"] == loaded.count(delivered)
    assert receipt["model_prompt_sha256"] == hashlib.sha256(delivered.encode()).hexdigest()
    assert receipt["coding_reply_contract"]["model_prompt_after_sha256"] == receipt["model_prompt_sha256"]
    assert receipt["coding_reply_contract"]["response_validated"] is True
    assert selection["selected_input_tokens"] == loaded.count(delivered) + loaded.count(
        json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    assert receipt["prompt_tokenizer_source"] == loaded.provenance()
    if selection["selected_representation"] == "native":
        assert case["prompt"] in delivered
        assert receipt["semantic_translation"] is None
        assert "semantic_metadata_view" not in receipt
        assert "semantic_response_translation" not in receipt
    assert receipt["completion_authority"] is False


def test_budget_failure_precedes_provider_dispatch(bound_dispatch):
    with pytest.raises(ValueError, match="token budget"):
        invoke(bound_dispatch, max_input_tokens=1)
    assert bound_dispatch[0]["calls"] == []


def test_grok_budget_counts_projected_native_schema(bound_dispatch, tmp_path, monkeypatch):
    from ipfs_accelerate_py.cli_runtime import grok_structured_output as grok
    artifact, digest, _ = make_binding(tmp_path, provider="grok_cli", model="grok-4.7")
    loaded = load_owner_prompt_tokenizer(artifact=artifact, expected_sha256=digest,
        provider="grok_cli", model="grok-4.7")
    canonical, wire = {"canonical": "hello"}, {"wire": "hello world"}
    monkeypatch.setattr(grok, "planning_response_format", lambda _: {"json_schema": {"schema": canonical}})
    def projection(schema, **_):
        assert schema == canonical
        return wire, json.dumps(wire), {}
    monkeypatch.setattr(grok, "native_schema_projection", projection)
    measured = []
    def count(text):
        measured.append(text)
        return loaded.count(text)
    counter = runner.PromptTokenCounter(**loaded.binding(), count_tokens=count)
    with pytest.raises(ValueError, match="token budget"):
        runner.run(prompt="hello world", purpose="planning", provider="grok_cli", model="grok-4.7",
            timeout=30, max_output_tokens=512, prompt_token_counter=counter, max_input_tokens=1)
    assert measured == ["hello world", json.dumps(wire, sort_keys=True, separators=(",", ":"))]
    assert bound_dispatch[0]["calls"] == []


@pytest.mark.parametrize("raw", [b"{", b'{"schema":"a","schema":"b"}', b'{"model":NaN}'])
def test_malformed_hash_bound_artifact_refuses_before_dispatch(bound_dispatch, raw):
    case, artifact, _, _ = bound_dispatch
    artifact.write_bytes(raw)
    altered = (case, artifact, hashlib.sha256(raw).hexdigest(), None)
    with pytest.raises(ValueError):
        invoke(altered)
    assert case["calls"] == []


def test_cli_passes_owner_tokenizer_and_budget(bound_dispatch, monkeypatch, capsys):
    case, artifact, digest, _ = bound_dispatch
    monkeypatch.setattr(sys, "stdin", type("Input", (), {"buffer": io.BytesIO(case["prompt"].encode())})())
    monkeypatch.setattr(sys, "argv", ["runner", "--model", "authored-model",
        "--semantic-repository", str(case["repository"]), "--coding-reply-mode", "ordinary-completion@1",
        "--prompt-tokenizer-artifact", str(artifact), "--prompt-tokenizer-sha256", digest,
        "--max-input-tokens", "200000"])
    assert runner.main() == 0 and len(case["calls"]) == 1
    assert '"prompt_selection"' in capsys.readouterr().out


@pytest.mark.parametrize("arguments", [
    ["--prompt-tokenizer-artifact", "/unused"], ["--prompt-tokenizer-sha256", ""],
    ["--prompt-tokenizer-sha256", "0" * 64], ["--max-input-tokens", "0"],
])
def test_cli_malformed_binding_or_limit_refuses_before_read(monkeypatch, arguments):
    class Unreadable:
        def read(self, *_):
            pytest.fail("invalid launch should refuse before reading task input")
    monkeypatch.setattr(sys, "stdin", type("Input", (), {"buffer": Unreadable()})())
    monkeypatch.setattr(sys, "argv", ["runner", "--model", MODEL, *arguments])
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == 2
