"""Actual local fixture tokenization across semantic dispatch and replay.

The semantic producer, source bindings, Git worktree, local tokenizer, selection,
decoder, historical receipt validation and model projection execute normally.
The provider and tokenizer
compatibility are authored fixtures; these counts establish no remote-model
token savings, semantic admission or completion.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import tokenizers
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from ipfs_accelerate_py.agent_supervisor.runtime import local_prompt_tokenizer as local
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from test.api.test_local_prompt_tokenizer import make_binding
from test.api.test_semantic_router_integration import (  # noqa: F401 -- register fixtures
    allocated,
    provider,
)
from test.api.test_semantic_router_translation import (  # noqa: F401 -- allocated depends on native
    _reply,
    native,
)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prepare(allocation, tmp_path):
    repository, workspace, prompt, _ = allocation
    assets = tmp_path / "owner-tokenizer-assets"
    assets.mkdir()
    artifact, descriptor_hash, descriptor = make_binding(assets, provider="codex_cli", model="pinned")
    loaded = local.load_owner_prompt_tokenizer(
        artifact=artifact, expected_sha256=descriptor_hash, provider="codex_cli", model="pinned")
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository)
    native_model, _ = runner.render_model_prompt(prompt=prompt, purpose="coding", workspace=workspace)
    semantic_model, _ = runner.render_model_prompt(
        prompt=encoded.provider_prompt, purpose="coding", workspace=workspace, semantic_transport=True)
    counts = {"native": loaded.count(native_model), "semantic": loaded.count(semantic_model)}
    runtime = {
        "version": tokenizers.__version__,
        "python_wrapper_sha256": hashlib.sha256(Path(tokenizers.__file__).read_bytes()).hexdigest(),
        "native_extension_sha256": hashlib.sha256(
            Path(sys.modules["tokenizers.tokenizers"].__file__).read_bytes()).hexdigest(),
    }
    # This difference is observed with the actual authored tokenizer, rather
    # than a byte callback, selected-response override or forced token count.
    assert counts["native"] > counts["semantic"] > 1
    return artifact, descriptor_hash, descriptor, loaded, encoded, native_model, semantic_model, counts, runtime


def invoke(allocation, *, artifact, descriptor_hash, limit):
    repository, _, prompt, _ = allocation
    return runner.run(
        prompt=prompt, provider="codex_cli", model="pinned", timeout=1, max_output_tokens=128,
        semantic_repository=repository, prompt_tokenizer_artifact=artifact,
        prompt_tokenizer_sha256=descriptor_hash, max_input_tokens=limit)


def test_local_semantic_winner_decodes_and_historical_audit_survives_asset_removal(
        allocated, provider, tmp_path, monkeypatch):  # noqa: F811 -- pytest injects imported fixtures
    repository, workspace, prompt, source = allocated
    state, observed = provider
    artifact, descriptor_hash, descriptor, loaded, encoded, native_model, semantic_model, counts, runtime = prepare(
        allocated, tmp_path)
    reply, symbol = _reply(encoded)
    state["reply"] = json.dumps(reply)
    output, receipt = invoke(
        allocated, artifact=artifact, descriptor_hash=descriptor_hash, limit=counts["semantic"])

    assert len(observed) == 1 and observed[0][0] == semantic_model
    assert observed[0][1]["temperature"] == 0
    assert json.loads(output)["structured_payload"]["symbol_ids"] == [symbol]
    assert receipt["semantic_response_translation"]["native_grammar_validated"] is True
    assert receipt["semantic_response_translation"]["completion_authority"] is False
    assert receipt["semantic_translation"] == receipt["semantic_encoding_candidate"] == encoded.receipt
    assert receipt["prompt_tokenizer_source"] == loaded.provenance()
    selection = receipt["prompt_selection"]
    assert selection["selected_representation"] == "semantic"
    assert selection["measurement_status"] == "measured"
    assert selection["selected_content_tokens"] == selection["max_input_tokens"] == counts["semantic"]
    assert selection["tokenizer_binding"] == loaded.binding()
    assert selection["total_session_token_savings"] is None
    for name, model in (("native", native_model), ("semantic", semantic_model)):
        assert selection["candidates"][name] == {
            "model_prompt_sha256": digest(model), "model_prompt_bytes": len(model.encode()),
            "input_tokens": counts[name],
        }
    assert codec.restore_semantic_router_prompt(
        provider_prompt=encoded.provider_prompt, table=encoded.table, repository=repository) == prompt

    # Preserve the actual receipt while removing its tokenizer assets and
    # advancing canonical source. Historical replay must use captured semantic
    # artifacts and recorded counts without reasserting currentness.
    artifact.unlink()
    (artifact.parent / descriptor["asset_path"]).unlink()
    (repository / "mod.py").write_text(source + "# subsequent canonical revision\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("historical audit must not reread or count tokenizer assets")

    monkeypatch.setattr(local, "load_owner_prompt_tokenizer", forbidden)
    monkeypatch.setattr(local, "_bounded_local_bytes", forbidden)
    monkeypatch.setattr(local, "_runtime", forbidden)
    monkeypatch.setattr(local.OwnerPromptTokenizer, "count", forbidden)
    parsed = audit._receipt({**receipt, "phase": "coding"}, workspace.parent)
    replayed, _, checks = audit._model_projection(rendered=prompt, receipt=parsed, repository=repository)
    assert replayed == observed[0][0] and all(checks.values()), checks
    assert len(observed) == 1
    historical = audit._historical_tokenizer_source(
        parsed["prompt_tokenizer_source"], parsed["prompt_selection"])
    assert historical["schema"] == "terminal-historical-prompt-tokenizer-provenance@1"
    assert historical["descriptor_sha256"] == descriptor_hash
    assert historical["asset_sha256"] == descriptor["asset_sha256"]
    assert historical["descriptor_path_sha256"] == digest(str(artifact))
    assert historical["asset_path_sha256"] == digest(str(artifact.parent / descriptor["asset_path"]))
    assert historical["source_currentness_revalidated"] is False
    assert historical["counts_recomputed"] is False
    assert historical["provider_tokenizer_compatibility_verified"] is False
    assert historical["completion_authority"] is False
    assert str(artifact) not in json.dumps(historical)
    observation = {
        "schema": "authored-offline-semantic-roundtrip-observation@1",
        "scope": "authored local tokenizer and stub provider; historical projection audit core",
        "tokenizer_binding": loaded.binding(), "tokenizer_source": historical,
        "observed_runtime_fingerprints": runtime,
        "candidates": selection["candidates"], "selected_representation": selection["selected_representation"],
        "exact_complete_prompt_ceiling": selection["max_input_tokens"],
        "prompt_roundtrip_exact": True, "native_candidate_grammar_validated": True,
        "historical_model_prompt_reconstruction_exact": True,
        "stub_provider_calls": len(observed), "real_provider_calls": 0,
        "compatibility_basis": "authored_fixture_owner_declaration",
        "provider_tokenizer_compatibility_verified": False, "actual_remote_token_counts": None,
        "total_session_token_savings": None, "semantic_qualification": False,
        "lean_admit": False, "completion_authority": False,
    }
    (tmp_path / "offline-semantic-observation.json").write_text(json.dumps(observation, indent=2, sort_keys=True) + "\n")


def test_local_selected_wire_overflow_refuses_before_provider_construction(
        allocated, provider, tmp_path, monkeypatch):  # noqa: F811 -- pytest injects imported fixtures
    from ipfs_accelerate_py import llm_router

    _, observed = provider
    artifact, descriptor_hash, _, _, _, _, _, counts, _ = prepare(allocated, tmp_path)
    constructions = []

    def forbidden(*args, **kwargs):
        constructions.append(True)
        raise AssertionError("provider construction must follow the complete-prompt budget gate")

    monkeypatch.setattr(llm_router, "get_llm_provider", forbidden)
    with pytest.raises(ValueError, match="token budget"):
        invoke(allocated, artifact=artifact, descriptor_hash=descriptor_hash, limit=counts["semantic"] - 1)
    assert constructions == observed == []
