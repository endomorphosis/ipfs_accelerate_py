"""Estimate quality survives real routing and observer-based comparisons.

Provider output/usage and counters are authored fixtures. The final comparison
fixture also substitutes codec admission; it establishes accounting behavior,
not production tokenizer compatibility, codec freshness or token savings.
"""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer as automatic
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.context_projection_comparison import (
    measure_context_candidates,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_candidate_selection import (
    select_semantic_candidate,
)
from test.api import test_semantic_selection_comparison as comparison_fixtures

MODEL = "authored-automatic-model"
PROMPT = "hello world café 法律"
authored = comparison_fixtures.authored


def counter(callback=None, *, kind="estimate", basis="authored_proxy", **overrides):
    return runner.PromptTokenCounter(**{
        "provider": "codex_cli", "model": MODEL, "tokenizer_id": "authored:estimate",
        "tokenizer_revision": "fixture-v1", "special_token_policy": "fixture-literal-only",
        "count_kind": kind, "compatibility_basis": basis, **overrides,
    }, count_tokens=callback or (lambda text: len(text.encode())))


@pytest.fixture
def automatic_route(tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index

    workspace = tmp_path / "allocated"
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    monkeypatch.chdir(workspace)
    state = SimpleNamespace(
        workspace=workspace, events=[], instances=[], prompts=[], requests=[],
        route=SimpleNamespace(provider="codex_cli", model_name=MODEL, catalog_revision="authored"),
        usage={"exit_code": 0, "prompt_tokens": 1001, "completion_tokens": 17,
               "cached_tokens": 99, "reasoning_tokens": 7, "model_id": MODEL},
    )

    def count(text):
        state.events.append(("count", text))
        return len(text.encode())

    state.counter = counter(count)
    state.provenance = {"schema": automatic.SCHEMA, "backend": "authored-fixture",
                        "count_kind": "estimate", "compatibility_basis": "authored_proxy",
                        "remote_tokenizer_compatibility_verified": False, "provider_usage": False}

    def resolve(*, provider, model):
        state.events.append(("resolve", provider, model))
        assert (provider, model) == ("codex_cli", MODEL)
        return state.counter, state.provenance

    def instance(provider, **kwargs):
        state.events.append(("instance", provider))
        state.instances.append(provider)
        return object()

    def generate(prompt, **kwargs):
        state.events.append(("generate", prompt))
        state.prompts.append(prompt)
        state.requests.append(kwargs)
        assert kwargs["provider"] == "codex_cli" and kwargs["model_name"] == MODEL
        assert kwargs["allow_local_fallback"] is kwargs["allow_cross_provider_fallback"] is False
        assert kwargs["temperature"] == 0
        set_last_cli_observation("codex_cli", state.usage)
        return "authored provider output"

    monkeypatch.setattr(automatic, "resolve_prompt_tokenizer", resolve)
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs: state.route)
    monkeypatch.setattr(llm_router, "get_llm_provider", instance)
    monkeypatch.setattr(llm_router, "generate_text", generate)
    return state


def run_route(**kwargs):
    return runner.run(**{
        "prompt": PROMPT, "provider": "codex_cli", "model": MODEL,
        "timeout": 30, "max_output_tokens": 512, **kwargs,
    })


def test_automatic_estimate_precedes_provider_and_never_enters_native_usage(automatic_route):
    state = automatic_route
    output, receipt = run_route()
    expected, advisory = runner.render_model_prompt(
        prompt=PROMPT, purpose="coding", workspace=state.workspace)
    selection = receipt["prompt_selection"]
    assert output == "authored provider output"
    assert [event[0] for event in state.events] == ["resolve", "count", "instance", "generate"]
    assert state.prompts == [expected]
    assert selection["measurement_status"] == "estimated"
    assert selection["candidates"]["native"]["input_tokens"] is None
    assert selection["candidates"]["native"]["estimated_input_tokens"] == len(expected.encode())
    assert selection["selected_content_tokens"] is None
    assert selection["selected_estimated_content_tokens"] == len(expected.encode())
    assert selection["token_count_kind"] == "estimate"
    assert selection["tokenizer_compatibility_basis"] == "authored_proxy"
    assert selection["total_session_token_savings"] is None
    assert receipt["workspace_advisory_bytes"] == len(advisory.encode())
    assert receipt["model_prompt_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
    assert receipt["prompt_tokenizer_source"] == state.provenance
    assert receipt["usage"] == state.usage


def test_estimate_budget_counts_workspace_and_required_suffix_before_provider(
        automatic_route, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction

    state = automatic_route
    suffix = "\nRequired literal /app/report.txt; preserve café 法律 acceptance criteria.\n"
    monkeypatch.setattr(router_public_instruction, "load_public_instruction",
                        lambda **kwargs: (suffix, {"schema": "authored-instruction"}))
    expected, _ = runner.render_model_prompt(
        prompt=PROMPT + suffix, purpose="coding", workspace=state.workspace)
    limit = len(expected.encode())
    _, receipt = run_route(
        public_instruction_artifact=state.workspace / "authored-owner-instruction",
        public_instruction_sha256="a" * 64, public_instruction_task_cid="authored-task",
        max_input_tokens=limit)
    assert limit > len((PROMPT + suffix).encode())
    assert state.prompts == [expected]
    assert expected.count(suffix) == 1
    assert [event[1] for event in state.events if event[0] == "count"] == [expected]
    selection = receipt["prompt_selection"]
    assert selection["selected_estimated_content_tokens"] == limit
    assert selection["selected_content_tokens"] is None
    assert selection["max_input_tokens"] == limit
    assert selection["token_budget_basis"] == "estimated_content_tokens"
    assert selection["token_budget_guaranteed"] is False
    assert receipt["usage"] == state.usage


def test_estimate_budget_overflow_refuses_before_provider_construction(automatic_route):
    state = automatic_route
    expected, _ = runner.render_model_prompt(prompt=PROMPT, purpose="coding", workspace=state.workspace)
    with pytest.raises(ValueError, match="token budget"):
        run_route(max_input_tokens=len(expected.encode()) - 1)
    assert [event[0] for event in state.events] == ["resolve", "count"]
    assert state.instances == state.prompts == []


@pytest.mark.parametrize("fault", [
    "orphan_artifact", "orphan_digest", "wrong_hash", "malformed", "wrong_model", "wrong_provider",
])
def test_invalid_explicit_descriptor_never_falls_back_to_automatic_or_provider(
        automatic_route, fault):
    state = automatic_route
    artifact = state.workspace / "owner-tokenizer.json"
    descriptor = {
        "schema": "supervisor-local-prompt-tokenizer@1", "provider": "codex_cli", "model": MODEL,
        "tokenizer_id": "authored", "tokenizer_revision": "authored",
        "special_token_policy": "tokenizers.encode:add_special_tokens=false",
        "backend": "tokenizers-json@1", "tokenizers_version": "authored",
        "asset_path": "unread-tokenizer.json", "asset_sha256": "b" * 64,
    }
    if fault == "wrong_model":
        descriptor["model"] = "other-model"
    if fault == "wrong_provider":
        descriptor["provider"] = "grok_cli"
    raw = b"{" if fault == "malformed" else json.dumps(descriptor).encode()
    artifact.write_bytes(raw)
    kwargs = {"prompt_tokenizer_artifact": artifact,
              "prompt_tokenizer_sha256": hashlib.sha256(raw).hexdigest()}
    if fault == "orphan_artifact":
        kwargs["prompt_tokenizer_sha256"] = ""
    elif fault == "orphan_digest":
        kwargs["prompt_tokenizer_artifact"] = None
    elif fault == "wrong_hash":
        kwargs["prompt_tokenizer_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        run_route(**kwargs)
    assert state.events == state.instances == state.prompts == []


def test_injected_counter_precedes_automatic_and_retains_caller_declaration(automatic_route):
    state = automatic_route
    supplied = counter(kind="tokenizer", basis="caller_declaration")
    _, receipt = run_route(prompt_token_counter=supplied)
    assert all(event[0] != "resolve" for event in state.events)
    assert receipt["prompt_selection"]["measurement_status"] == "measured"
    assert receipt["prompt_selection"]["tokenizer_compatibility_basis"] == "caller_declaration"
    assert receipt["prompt_selection"]["selected_content_tokens"] == len(state.prompts[0].encode())
    assert receipt["prompt_selection"]["selected_estimated_content_tokens"] is None
    assert receipt["usage"] == state.usage


def test_valid_owner_descriptor_retains_owner_basis_without_automatic_resolution(
        automatic_route, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import local_prompt_tokenizer

    state = automatic_route
    owner = counter(kind="tokenizer", basis="owner_declaration")
    provenance = {"compatibility_basis": "owner_declaration", "authored_loader_substitution": True}
    monkeypatch.setattr(local_prompt_tokenizer, "load_owner_prompt_tokenizer", lambda **kwargs:
                        SimpleNamespace(binding=owner.binding, count=owner.count, provenance=lambda: provenance))
    _, receipt = run_route(prompt_tokenizer_artifact=state.workspace / "authored-owner",
                           prompt_tokenizer_sha256="c" * 64)
    assert all(event[0] != "resolve" for event in state.events)
    assert receipt["prompt_tokenizer_source"] == provenance
    assert receipt["prompt_selection"]["measurement_status"] == "measured"
    assert receipt["prompt_selection"]["tokenizer_compatibility_basis"] == "owner_declaration"


@pytest.mark.parametrize("changed", ["provider", "model"])
def test_automatic_resolution_does_not_relax_the_explicit_provider_pin(automatic_route, changed):
    state = automatic_route
    if changed == "provider":
        state.route.provider = "grok_cli"
    else:
        state.route.model_name = "other-model"
    with pytest.raises(ValueError, match="changed the explicit benchmark route"):
        run_route()
    assert state.instances == state.prompts == []


@pytest.mark.parametrize("semantic_count,selected", [(99, "semantic"), (100, "native"), (101, "native")])
def test_estimate_selection_requires_a_strictly_smaller_complete_count(semantic_count, selected):
    seen = []

    def count(text):
        seen.append(text)
        return 100 if "N" * 2000 in text else semantic_count

    _, _, _, receipt = runner.select_model_prompt(
        native_prompt="N" * 2000, semantic_prompt="S" * 100, suffix=" required literal\n",
        purpose="coding", workspace=Path("/allocated/task"), provider="codex_cli", model=MODEL,
        token_counter=counter(count))
    assert len(seen) == 2 and all(text.endswith(" required literal\n") for text in seen)
    assert receipt["selected_representation"] == selected
    assert receipt["measurement_status"] == "estimated"
    assert receipt["selected_content_tokens"] is None
    assert all(row["input_tokens"] is None for row in receipt["candidates"].values())
    assert receipt["selected_estimated_content_tokens"] == (semantic_count if selected == "semantic" else 100)


def select_candidates(**overrides):
    return select_semantic_candidate(**{
        "native_prompt": "N" * 2000, "candidates": [("legacy", "L" * 500), ("v2", "V" * 100)],
        "suffix": "\nRequired literal acceptance.\n", "purpose": "coding", "workspace": Path("/allocated/task"),
        "provider": "codex_cli", "model": MODEL, "token_counter": counter(), **overrides,
    })


def test_v2_observer_preserves_estimate_quality_and_exact_budget_boundary():
    result = select_candidates()
    selected, _, _, receipt, label = result
    limit = len(selected.encode())
    assert label == "v2"
    assert receipt["measurement_status"] == "estimated"
    assert receipt["token_count_kind"] == "estimate"
    assert receipt["tokenizer_compatibility_basis"] == "authored_proxy"
    assert receipt["selected_content_tokens"] is None
    assert receipt["selected_estimated_content_tokens"] == limit
    assert all(row["input_tokens"] is None and isinstance(row["estimated_input_tokens"], int)
               for row in receipt["comparison_candidates"].values())
    bounded = select_candidates(max_input_tokens=limit)[3]
    assert bounded["token_budget_basis"] == "estimated_content_tokens"
    assert bounded["token_budget_guaranteed"] is False
    with pytest.raises(ValueError, match="token budget"):
        select_candidates(max_input_tokens=limit - 1)


@pytest.mark.parametrize("fault", ["late_failure", "drift"])
def test_v2_failure_or_drift_discards_all_partial_actual_and_estimated_counts(fault):
    observations = {}

    def count(text):
        observations[text] = observations.get(text, 0) + 1
        if fault == "late_failure" and "V" * 100 in text:
            raise RuntimeError("authored late candidate failure")
        return len(text.encode()) + (observations[text] if fault == "drift" else 0)

    _, _, _, receipt, label = select_candidates(token_counter=counter(count))
    assert label is None and receipt["selected_representation"] == "native"
    assert receipt["measurement_status"] == "failed"
    assert receipt["measurement_error_type"] == ("InconsistentTokenCount" if fault == "drift" else "RuntimeError")
    assert receipt["selected_content_tokens"] is receipt["selected_estimated_content_tokens"] is None
    assert receipt["token_measurements_consistent"] is False
    for row in [*receipt["candidates"].values(), *receipt["comparison_candidates"].values()]:
        assert row["input_tokens"] is row["estimated_input_tokens"] is None
    with pytest.raises(ValueError, match="consistent pinned counter"):
        select_candidates(token_counter=counter(count), max_input_tokens=100_000)


def test_oversized_native_pair_keeps_estimates_until_a_later_fitting_candidate():
    def count(text):
        return 100 if "N" * 100 in text else 200 if "legacy-marker" in text else 50

    _, _, _, receipt, label = select_candidates(
        native_prompt="N" * runner.MAX_MODEL_PROMPT_BYTES,
        candidates=[("legacy", "legacy-marker"), ("v2", "v2-marker")],
        token_counter=counter(count), max_input_tokens=50)
    assert label == "v2" and receipt["measurement_status"] == "estimated"
    assert receipt["comparison_candidates"]["legacy"]["input_tokens"] is None
    assert receipt["comparison_candidates"]["legacy"]["estimated_input_tokens"] == 200
    assert receipt["selected_content_tokens"] is None
    assert receipt["selected_estimated_content_tokens"] == 50
    assert receipt["token_budget_guaranteed"] is False


@pytest.mark.parametrize("with_alias", [False, True])
def test_context_projection_comparison_emits_only_estimated_token_deltas(with_alias):
    report = measure_context_candidates(
        native_prompt="native " * 1000, projected_prompt="projected " * 100,
        alias_prompt="alias " * 10 if with_alias else None, suffix="\nLiteral criteria.\n",
        purpose="coding", workspace=Path("/allocated/task"), provider="codex_cli", model=MODEL,
        token_counter=counter(), measurement_basis="automatic_offline_tokenizer")
    assert report["token_count_kind"] == "estimate"
    assert report["tokenizer_compatibility_basis"] == "authored_proxy"
    assert set(report["measurement_status"].values()) == {"estimated"}
    assert all(row["input_tokens"] is None and isinstance(row["estimated_input_tokens"], int)
               for row in report["candidates"].values())
    native = report["candidates"]["native"]["estimated_input_tokens"]
    for name, comparison in report["comparisons"].items():
        assert comparison["rendered_user_content_token_delta"] is None
        assert comparison["rendered_user_estimated_content_token_delta"] == (
            native - report["candidates"][name]["estimated_input_tokens"])
    assert report["provider_invoked"] is report["provider_compatibility_verified"] is False
    assert report["whole_session_token_savings"] is None


def test_authored_semantic_comparison_preserves_estimates_without_actual_savings(authored):
    # The borrowed fixture substitutes encode/restore; this is an accounting control.
    result = comparison_fixtures._compare(authored, token_counter=comparison_fixtures._counter(
        count_kind="estimate", compatibility_basis="authored_proxy"))
    assert result["measurement_status"] == "estimated"
    assert result["token_count_kind"] == "estimate"
    assert result["tokenizer_compatibility_basis"] == "authored_proxy"
    assert all(row["input_tokens"] is None and row["estimated_input_tokens"] == row["model_prompt_bytes"]
               for row in result["representations"].values())
    baseline = result["representations"]["full_native"]["estimated_input_tokens"]
    for name, comparison in result["comparisons_to_full_native"].items():
        assert comparison["rendered_user_content_token_delta"] is None
        assert comparison["rendered_user_estimated_content_token_delta"] == (
            baseline - result["representations"][name]["estimated_input_tokens"])
    assert result["whole_session_token_savings"] is None
    assert result["provider_calls"] == 0


def test_authored_semantic_comparison_drift_clears_every_estimate(authored):
    observations = {}

    def count(text):
        observations[text] = observations.get(text, 0) + 1
        return len(text.encode()) + observations[text]

    result = comparison_fixtures._compare(authored, token_counter=comparison_fixtures._counter(
        count, count_kind="estimate", compatibility_basis="authored_proxy"))
    assert result["measurement_status"] == "failed"
    assert result["measurement_error_type"] == "InconsistentTokenCount"
    for row in result["representations"].values():
        assert row["input_tokens"] is row["estimated_input_tokens"] is None
    for comparison in result["comparisons_to_full_native"].values():
        assert comparison["rendered_user_content_token_delta"] is None
        assert comparison["rendered_user_estimated_content_token_delta"] is None
    assert result["provider_calls"] == 0
