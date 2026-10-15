"""Offline tokenizer wiring through the real runner and CLI with a stub provider.

Tokenizer assets, provider output and usage are authored fixtures; no remote
model-token equivalence, deployment, benchmark or semantic admission follows.
"""

import hashlib
import io
import json
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.local_prompt_tokenizer import (
    load_owner_prompt_tokenizer,
)
from test.api.test_local_prompt_tokenizer import MODEL, make_binding


@pytest.fixture
def route(tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index

    assets = tmp_path / "owner-assets"
    assets.mkdir()
    artifact, digest, descriptor = make_binding(assets)
    workspace = tmp_path / "allocated"
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
                        SimpleNamespace(provider="codex_cli", model_name=MODEL, catalog_revision="fixture"))
    instances, prompts = [], []

    def instance(*args, **kwargs):
        instances.append(True)
        return object()

    def generate(prompt, **kwargs):
        prompts.append(prompt)
        assert kwargs["temperature"] == 0
        assert kwargs["allow_local_fallback"] is False
        assert kwargs["allow_cross_provider_fallback"] is False
        set_last_cli_observation("codex_cli", {"exit_code": 0})
        return "authored fixture output"

    monkeypatch.setattr(llm_router, "get_llm_provider", instance)
    monkeypatch.setattr(llm_router, "generate_text", generate)
    return artifact, digest, descriptor, workspace, instances, prompts


def invoke(route, **overrides):
    artifact, digest, *_ = route
    return runner.run(**{
        "prompt": "hello world café 法律", "provider": "codex_cli", "model": MODEL,
        "timeout": 30, "max_output_tokens": 512,
        "prompt_tokenizer_artifact": artifact, "prompt_tokenizer_sha256": digest,
        **overrides,
    })


def test_runner_counts_complete_wire_and_preserves_source_provenance(route):
    artifact, digest, descriptor, workspace, instances, prompts = route
    loaded = load_owner_prompt_tokenizer(
        artifact=artifact, expected_sha256=digest, provider="codex_cli", model=MODEL)
    model_prompt, advisory = runner.render_model_prompt(
        prompt="hello world café 法律", purpose="coding", workspace=workspace)
    limit = loaded.count(model_prompt)
    assert limit > loaded.count("hello world café 法律")
    output, receipt = invoke(route, max_input_tokens=limit)
    assert output == "authored fixture output" and instances == [True]
    assert prompts == [model_prompt]
    assert receipt["workspace_advisory_bytes"] == len(advisory.encode())
    assert receipt["prompt_selection"]["measurement_status"] == "measured"
    assert receipt["prompt_selection"]["selected_content_tokens"] == limit
    assert receipt["prompt_selection"]["tokenizer_binding"] == loaded.binding()
    assert receipt["prompt_selection"]["total_session_token_savings"] is None
    assert receipt["prompt_tokenizer_source"] == loaded.provenance()
    assert receipt["prompt_tokenizer_source"]["asset_sha256"] == descriptor["asset_sha256"]
    assert receipt["model_prompt_sha256"] == hashlib.sha256(model_prompt.encode()).hexdigest()


def test_bound_counter_budget_overflow_refuses_before_provider(route):
    artifact, digest, _, workspace, instances, prompts = route
    loaded = load_owner_prompt_tokenizer(
        artifact=artifact, expected_sha256=digest, provider="codex_cli", model=MODEL)
    model_prompt, _ = runner.render_model_prompt(
        prompt="hello world café 法律", purpose="coding", workspace=workspace)
    with pytest.raises(ValueError, match="token budget"):
        invoke(route, max_input_tokens=loaded.count(model_prompt) - 1)
    assert instances == prompts == []


@pytest.mark.parametrize("fault", ["orphan_artifact", "orphan_digest", "wrong_hash", "wrong_model"])
def test_invalid_owner_binding_refuses_before_provider(route, fault):
    _, _, _, _, instances, prompts = route
    overrides = {
        "orphan_artifact": {"prompt_tokenizer_sha256": ""},
        "orphan_digest": {"prompt_tokenizer_artifact": None},
        "wrong_hash": {"prompt_tokenizer_sha256": "0" * 64},
        "wrong_model": {"model": "different-owner-model"},
    }[fault]
    with pytest.raises(ValueError):
        invoke(route, **overrides)
    assert instances == prompts == []


def test_file_and_injected_counters_cannot_compete(route):
    counter = runner.PromptTokenCounter(
        provider="codex_cli", model=MODEL, tokenizer_id="fixture", tokenizer_revision="fixture",
        special_token_policy="fixture", count_tokens=lambda text: 1)
    with pytest.raises(ValueError, match="choose one"):
        invoke(route, prompt_token_counter=counter)
    assert route[4] == route[5] == []


def test_cli_forwards_exact_owner_binding_and_budget(route, monkeypatch, capsys):
    artifact, digest, *_ = route
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"hello world")))
    monkeypatch.setattr(sys, "argv", [
        "runner", "--model", MODEL, "--prompt-tokenizer-artifact", str(artifact),
        "--prompt-tokenizer-sha256", digest, "--max-input-tokens", "10000",
    ])
    assert runner.main() == 0
    assert route[4] == [True] and len(route[5]) == 1
    printed = capsys.readouterr().out
    assert '"prompt_tokenizer_source"' in printed
    assert '"measurement_status": "measured"' in printed
    assert '"max_input_tokens": 10000' in printed


def test_cli_unbound_budget_uses_labelled_automatic_estimate(route, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"hello world")))
    monkeypatch.setattr(sys, "argv", ["runner", "--model", MODEL, "--max-input-tokens", "10000"])
    assert runner.main() == 0
    assert route[4] == [True] and len(route[5]) == 1
    receipts = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    selection = receipts[0]["prompt_selection"]
    assert selection["measurement_status"] == "estimated"
    assert selection["selected_content_tokens"] is None
    assert selection["selected_estimated_content_tokens"] <= 10000
    assert selection["token_budget_guaranteed"] is False


@pytest.mark.parametrize("fault", [
    "artifact_only", "empty_digest_only", "digest_only", "paired_empty_digest", "empty_digest_equals",
])
def test_cli_requested_binding_refuses_incomplete_or_empty_values_before_stdin(
        route, monkeypatch, capsys, fault):
    artifact, digest, *_ = route
    arguments = {
        "artifact_only": ["--prompt-tokenizer-artifact", str(artifact)],
        "empty_digest_only": ["--prompt-tokenizer-sha256", ""],
        "digest_only": ["--prompt-tokenizer-sha256", digest],
        "paired_empty_digest": ["--prompt-tokenizer-artifact", str(artifact),
                                "--prompt-tokenizer-sha256", ""],
        "empty_digest_equals": ["--prompt-tokenizer-sha256="],
    }[fault]

    def forbidden(*args, **kwargs):
        raise AssertionError("malformed CLI bindings must refuse before stdin")

    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=SimpleNamespace(read=forbidden)))
    monkeypatch.setattr(sys, "argv", ["runner", "--model", MODEL, *arguments])
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == 2
    assert route[4] == route[5] == []
    assert "both tokenizer binding arguments require nonempty values" in capsys.readouterr().err


def test_cli_without_binding_resolves_automatic_estimate(route, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"hello world")))
    monkeypatch.setattr(sys, "argv", ["runner", "--model", MODEL])
    assert runner.main() == 0
    receipts = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert len(receipts) == 1
    assert receipts[0]["prompt_selection"]["measurement_status"] == "estimated"
    assert receipts[0]["prompt_selection"]["selected_content_tokens"] is None
    assert receipts[0]["prompt_selection"]["selected_estimated_content_tokens"] > 0
    assert receipts[0]["prompt_tokenizer_source"]["count_kind"] == "estimate"
    assert route[4] == [True] and len(route[5]) == 1


def test_module_entry_uses_its_own_counter_class_without_duplicate_import(route, monkeypatch):
    artifact, digest, *_ = route
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"hello world")))
    monkeypatch.setattr(sys, "argv", [
        "runner", "--model", MODEL, "--prompt-tokenizer-artifact", str(artifact),
        "--prompt-tokenizer-sha256", digest, "--max-input-tokens", "10000",
    ])
    # The imported runner above is deliberately already present. Executing its
    # actual source as __main__ creates a distinct PromptTokenCounter class.
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as exit_status:
            runpy.run_module(runner.__name__, run_name="__main__", alter_sys=True)
    assert exit_status.value.code == 0
    assert route[4] == [True] and len(route[5]) == 1
