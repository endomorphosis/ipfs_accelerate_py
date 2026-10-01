"""Real native capsules reach the shared router through a reversible boundary.

Only the model/provider is replaced: the semantic producer, native context
compiler, Git worktree relation, encoder and response validators run normally.
"""
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from test.api.test_semantic_router_translation import native, _reply


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def allocated(native, monkeypatch):
    repository, artifact, prompt, source = native
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "mod.py"], check=True)
    subprocess.run(["git", "-C", str(repository), "-c", "user.name=Qualification",
                    "-c", "user.email=qualification@example.invalid", "commit", "-qm", "fixture"], check=True)
    workspace = repository.parent / "allocated-worktree"
    subprocess.run(["git", "-C", str(repository), "worktree", "add", "--detach", "-q", str(workspace)], check=True)
    monkeypatch.chdir(workspace)
    # No host credential or rollout file is needed or read by this qualification.
    monkeypatch.setenv("CODEX_HOME", str(repository.parent / "empty-codex-home"))
    return repository, workspace, prompt, source


@pytest.fixture
def provider(monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index

    observed = []
    state = {"reply": "literal summary"}
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="codex_cli", model_name="pinned", catalog_revision="fixture"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: object())

    def generate(prompt, **kwargs):
        observed.append((prompt, kwargs))
        assert kwargs["allow_local_fallback"] is False
        assert kwargs["allow_cross_provider_fallback"] is False
        assert kwargs["model_name"] == "pinned"
        assert kwargs["reasoning_effort"] == "high"
        set_last_cli_observation("codex_cli", {"exit_code": 0, "prompt_tokens": 123,
            "completion_tokens": 7, "cached_tokens": 19, "reasoning_tokens": 3})
        if state.get("after_generate") is not None:
            state["after_generate"]()
        return state["reply"]

    monkeypatch.setattr(llm_router, "generate_text", generate)
    return state, observed


def _run(repository, prompt, **kwargs):
    return runner.run(prompt=prompt, provider="codex_cli", model="pinned", timeout=1,
        max_output_tokens=128, semantic_repository=repository, **kwargs)


def test_actual_router_transports_capsules_and_expands_native_candidate(allocated, provider):
    repository, workspace, prompt, _ = allocated
    state, observed = provider
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository)
    reply, symbol = _reply(encoded)
    state["reply"] = json.dumps(reply)

    output, receipt = _run(repository, prompt)
    assert len(observed) == 1
    actual, _ = observed[0]
    expected, advisory = runner.render_model_prompt(prompt=encoded.provider_prompt,
        purpose="coding", workspace=workspace, semantic_transport=True)
    assert actual == expected == advisory + encoded.provider_prompt
    assert "Do not write to canonical /app" in advisory
    assert "does not expand the task's allowed files or grant completion authority" in advisory
    assert receipt["workspace"] == str(workspace)
    assert receipt["native_prompt_sha256"] == receipt["prompt_sha256"] == _sha(prompt)
    assert receipt["router_prompt_sha256"] == _sha(encoded.provider_prompt)
    assert receipt["model_prompt_sha256"] == _sha(actual)
    assert receipt["workspace_advisory_sha256"] == _sha(advisory)
    assert receipt["native_prompt_bytes"] == len(prompt.encode())
    assert receipt["router_prompt_bytes"] == len(encoded.provider_prompt.encode())
    assert receipt["model_prompt_bytes"] == len(actual.encode())
    assert receipt["semantic_translation"] == encoded.receipt
    assert receipt["semantic_response_translation"]["native_grammar_validated"] is True
    assert receipt["semantic_response_translation"]["completion_authority"] is False
    assert receipt["completion_authority"] is False
    assert receipt["usage"]["prompt_tokens"] == 123
    assert json.loads(output)["structured_payload"]["symbol_ids"] == [symbol]
    assert codec.restore_semantic_router_prompt(provider_prompt=actual[len(advisory):],
        table=encoded.table, repository=repository) == prompt


@pytest.mark.parametrize("refusal", ["stale", "foreign", "canonical_workspace"])
def test_dispatch_refuses_stale_or_unrelated_semantic_repository(allocated, provider, monkeypatch, refusal):
    repository, workspace, prompt, source = allocated
    _, observed = provider
    if refusal == "stale":
        (repository / "mod.py").write_text(source + "# changed canonical source\n")
        match = "stale"
    elif refusal == "foreign":
        repository = repository.parent / "foreign-repository"
        repository.mkdir()
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        match = "differs from allocated worktree"
    else:
        monkeypatch.chdir(repository)
        match = "separate allocated worktree"
    with pytest.raises(ValueError, match=match):
        _run(repository, prompt)
    assert observed == []


def test_decode_refusal_retains_observed_provider_usage(allocated, provider, capsys):
    repository, workspace, prompt, _ = allocated
    state, observed = provider
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository)
    reply, _ = _reply(encoded)
    reply["response"]["structured_payload"]["symbol_ids"] = [{codec.REF_KEY: "unmapped"}]
    state["reply"] = json.dumps(reply)
    with pytest.raises(codec.SemanticTranslationError):
        _run(repository, prompt)
    receipts = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    receipt = next(row for row in receipts if row.get("schema") == "router-implementation-invocation@1")
    assert len(observed) == 1
    assert receipt["status"] == "failed"
    assert receipt["error_type"] == "SemanticTranslationError"
    assert receipt["usage"] == {"exit_code": 0, "prompt_tokens": 123, "completion_tokens": 7,
        "cached_tokens": 19, "reasoning_tokens": 3}
    assert receipt["model_prompt_sha256"] == _sha(observed[0][0])
    assert receipt["semantic_translation"] == encoded.receipt
    assert "semantic_response_translation" not in receipt
    assert receipt["completion_authority"] is False


def test_plain_summary_remains_literal_after_actual_translated_dispatch(allocated, provider):
    repository, workspace, prompt, _ = allocated
    state, observed = provider
    state["reply"] = '  Literal s0 and {"$semantic_ref":"s99999"}, not structured evidence.\n'
    output, receipt = _run(repository, prompt)
    assert output == state["reply"]
    assert len(observed) == 1
    assert receipt["semantic_response_translation"]["structured_translation"] is False
    assert receipt["status"] == "provider_returned"


def test_structured_reply_refuses_source_change_during_provider_call(allocated, provider, capsys):
    repository, workspace, prompt, source = allocated
    state, observed = provider
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository)
    reply, _ = _reply(encoded)
    state["reply"] = json.dumps(reply)
    state["after_generate"] = lambda: (repository / "mod.py").write_text(source + "# concurrent update\n")
    with pytest.raises(codec.SemanticTranslationError, match="stale"):
        _run(repository, prompt)
    receipts = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    receipt = next(row for row in receipts if row.get("schema") == "router-implementation-invocation@1")
    assert len(observed) == 1
    assert receipt["status"] == "failed"
    assert receipt["usage"]["prompt_tokens"] == 123
    assert receipt["usage"]["completion_tokens"] == 7
    assert "semantic_response_translation" not in receipt
