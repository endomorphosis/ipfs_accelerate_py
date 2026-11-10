"""Real local JSON counts and codec checks; provider observations are authored."""
import hashlib
import io
import json
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import local_prompt_tokenizer as local_codec
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_metadata_view as metadata
from test.api.test_local_prompt_tokenizer import make_binding
from test.api.test_semantic_router_translation import native  # noqa: F401
from test.api.test_semantic_metadata_dispatch import dispatch_case, token_counter  # noqa: F401
from test.api.test_repository_world_dispatch import (
    dispatch_case as strict_dispatch_case, world_session, required, scheduler, TASK,
)  # noqa: F401


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def local_binding(dispatch_case, tmp_path):
    folder = tmp_path / "local-tokenizer"
    folder.mkdir()
    artifact, pin, descriptor = make_binding(folder, model="authored-model")
    return dispatch_case, artifact, pin, descriptor


def invoke(binding, **changes):
    case, artifact, pin, _ = binding
    options = dict(prompt=case["prompt"], provider="codex_cli", model="authored-model",
        timeout=30, max_output_tokens=4096, semantic_repository=case["repository"],
        coding_reply_mode="ordinary-completion@1", semantic_metadata_view="common-bindings@1",
        semantic_metadata_tokenizer_artifact=artifact, semantic_metadata_tokenizer_sha256=pin)
    options.update(changes)
    return runner.run(**options)


def baseline(binding):
    return invoke(binding, semantic_metadata_view="legacy",
        semantic_metadata_tokenizer_artifact=None, semantic_metadata_tokenizer_sha256="")


def test_real_local_counts_select_complete_prompt_and_bind_exact_inverse(local_binding, monkeypatch):
    case, artifact, pin, descriptor = local_binding
    baseline(local_binding)
    original = case["calls"][-1][0]
    observed = []
    actual_selector = metadata.select_semantic_metadata_view

    def selector(**options):
        selected = actual_selector(**options)
        observed.append((options, selected))
        return selected

    monkeypatch.setattr(metadata, "select_semantic_metadata_view", selector)
    _, receipt = invoke(local_binding)
    (options, selected), = observed
    local, counter, provenance = runner.load_semantic_metadata_tokenizer(
        artifact=artifact, expected_sha256=pin, provider="codex_cli", model="authored-model")
    assert options["native_complete_prompt"] == original
    assert metadata.restore_semantic_metadata_view(provider_prompt=options["view"].provider_prompt,
        receipt=options["view"].receipt) == options["view"].native_prompt
    assert receipt["semantic_metadata_view"] == selected.receipt
    assert selected.receipt["native_complete_tokens"] == local.count(original)
    assert selected.receipt["candidate_complete_tokens"] == local.count(options["candidate_complete_prompt"])
    assert case["calls"][-1][0] == selected.selected_prompt
    assert receipt["model_prompt_sha256"] == _sha(selected.selected_prompt.encode())
    assert receipt["coding_reply_contract"]["model_prompt_after_sha256"] == receipt["model_prompt_sha256"]
    assert receipt["semantic_metadata_tokenizer_source"] == provenance
    assert receipt["semantic_metadata_tokenizer_dispatch_rechecked"] is True
    assert receipt["semantic_metadata_view"]["token_counter_identity"]["qualification"] == "estimate"
    assert counter.tokenizer_revision == descriptor["tokenizer_revision"] + "/special-policy-sha256:" + _sha(
        descriptor["special_token_policy"].encode())
    assert "prompt_selection" not in receipt
    assert "prompt_tokenizer_source" not in receipt


def test_content_free_provenance_has_asset_policy_pins_without_remote_claims(local_binding):
    _, artifact, pin, descriptor = local_binding
    _, counter, provenance = runner.load_semantic_metadata_tokenizer(artifact=artifact,
        expected_sha256=pin, provider="codex_cli", model="authored-model")
    assert provenance["descriptor_sha256"] == pin
    assert provenance["asset_sha256"] == descriptor["asset_sha256"]
    assert provenance["special_token_policy"] == descriptor["special_token_policy"]
    assert provenance["counter_tokenizer_revision"] == counter.tokenizer_revision
    assert provenance["measurement_scope"] == "complete-rendered-prompt-text@1"
    assert provenance["compatibility_basis"] == "owner_declaration"
    assert str(artifact.parent) not in json.dumps(provenance)
    assert "asset_path" not in provenance and "descriptor_path" not in provenance
    for key in ("provider_tokenizer_equivalence_verified", "provider_usage_verified",
                "provider_ingestion_verified", "complete_run_token_savings_claimed", "completion_authority"):
        assert provenance[key] is False


def test_special_token_policy_is_part_of_counter_identity(local_binding, tmp_path):
    _, artifact, pin, _ = local_binding
    second = tmp_path / "special-token-policy"
    second.mkdir()
    other, other_pin, _ = make_binding(second, model="authored-model", add_special_tokens=True)
    _, left, _ = runner.load_semantic_metadata_tokenizer(artifact=artifact, expected_sha256=pin,
        provider="codex_cli", model="authored-model")
    _, right, _ = runner.load_semantic_metadata_tokenizer(artifact=other, expected_sha256=other_pin,
        provider="codex_cli", model="authored-model")
    assert left.tokenizer_revision != right.tokenizer_revision
    assert right.count("hello") == left.count("hello") + 2


def test_explicit_none_preserves_existing_default_receipt(local_binding):
    _, receipt = baseline(local_binding)
    assert "semantic_metadata_tokenizer_source" not in receipt
    assert "semantic_metadata_tokenizer_dispatch_rechecked" not in receipt
    assert "semantic_metadata_view" not in receipt


def test_general_prompt_tokenizer_is_not_silently_reused_for_metadata(local_binding):
    _, artifact, pin, _ = local_binding
    _, receipt = invoke(local_binding, semantic_metadata_tokenizer_artifact=None,
        semantic_metadata_tokenizer_sha256="", prompt_tokenizer_artifact=artifact, prompt_tokenizer_sha256=pin)
    selection = receipt.get("semantic_metadata_view", receipt.get("semantic_metadata_candidate"))
    assert selection["schema"] == "supervisor-semantic-metadata-selection@1"
    assert "semantic_metadata_tokenizer_source" not in receipt
    assert "prompt_selection" in receipt


def test_counter_failure_selects_original_without_proxy_rescue_or_error_text(local_binding, monkeypatch):
    case = local_binding[0]
    baseline(local_binding)
    original = case["calls"][-1][0]

    def broken(_self, text):
        raise RuntimeError("PRIVATE_TOKENIZER_FAILURE " + text)

    monkeypatch.setattr(local_codec.OwnerPromptTokenizer, "count", broken)
    _, receipt = invoke(local_binding)
    assert case["calls"][-1][0] == original
    selection = receipt["semantic_metadata_view"]
    assert selection["selected_mode"] == "legacy"
    assert selection["token_counter_status"] == "failed"
    assert selection["native_complete_tokens"] is selection["candidate_complete_tokens"] is None
    assert selection["fallback_reason"] == "complete_input_token_counter_failed"
    assert receipt["semantic_metadata_tokenizer_dispatch_rechecked"] is True
    assert "PRIVATE_TOKENIZER_FAILURE" not in json.dumps(receipt)


def test_token_expansion_refuses_smaller_byte_candidate(local_binding, monkeypatch):
    case = local_binding[0]
    baseline(local_binding)
    original = case["calls"][-1][0]
    measured, values = [], iter((100, 101))

    def count(_self, text):
        measured.append(text)
        return next(values)

    monkeypatch.setattr(local_codec.OwnerPromptTokenizer, "count", count)
    _, receipt = invoke(local_binding)
    selection = receipt["semantic_metadata_view"]
    assert measured[0] == original and len(measured) == 2
    assert selection["candidate_complete_bytes"] < selection["native_complete_bytes"]
    assert selection["candidate_complete_tokens"] > selection["native_complete_tokens"]
    assert selection["selected_mode"] == "legacy"
    assert case["calls"][-1][0] == original


@pytest.mark.parametrize("changes", [
    {"semantic_metadata_tokenizer_artifact": None}, {"semantic_metadata_tokenizer_sha256": ""},
    {"semantic_metadata_tokenizer_sha256": None}, {"semantic_metadata_tokenizer_sha256": "A" * 64},
    {"semantic_metadata_tokenizer_sha256": "0" * 63}, {"semantic_metadata_tokenizer_sha256": "0" * 64},
])
def test_invalid_or_missing_descriptor_pin_refuses_before_provider(local_binding, changes):
    with pytest.raises(ValueError):
        invoke(local_binding, **changes)
    assert local_binding[0]["calls"] == []


@pytest.mark.parametrize("changes", [
    {"semantic_metadata_view": "legacy"}, {"purpose": "planning"}, {"semantic_repository": None},
    {"semantic_transport_schema": "supervisor-semantic-router-input@2"},
])
def test_descriptor_requires_explicit_original_semantic_coding_route(local_binding, monkeypatch, changes):
    def forbidden(**_):
        raise AssertionError("unsupported route must refuse before loading any tokenizer")
    monkeypatch.setattr(runner, "load_semantic_metadata_tokenizer", forbidden)
    with pytest.raises(ValueError):
        invoke(local_binding, **changes)
    assert local_binding[0]["calls"] == []


def test_descriptor_and_typed_metadata_counter_are_mutually_exclusive(local_binding, monkeypatch):
    monkeypatch.setattr(runner, "load_semantic_metadata_tokenizer", lambda **_: pytest.fail("loaded conflicting binding"))
    with pytest.raises(ValueError, match="choose one metadata"):
        invoke(local_binding, semantic_metadata_token_counter=token_counter(len))
    assert local_binding[0]["calls"] == []


@pytest.mark.parametrize("kind", ["counter", "artifact", "pin", "budget"])
def test_descriptor_cannot_be_overwritten_by_general_prompt_selection(local_binding, monkeypatch, kind):
    _, artifact, pin, _ = local_binding
    changes = {"counter": {"prompt_token_counter": object()},
        "artifact": {"prompt_tokenizer_artifact": artifact},
        "pin": {"prompt_tokenizer_sha256": pin}, "budget": {"max_input_tokens": 100}}[kind]
    monkeypatch.setattr(runner, "load_semantic_metadata_tokenizer", lambda **_: pytest.fail("loaded conflicting binding"))
    with pytest.raises(ValueError, match="independent metadata selection"):
        invoke(local_binding, **changes)
    assert local_binding[0]["calls"] == []


@pytest.mark.parametrize("field", ["provider", "model"])
def test_owner_declared_route_mismatch_refuses(local_binding, field):
    _, artifact, _, descriptor = local_binding
    descriptor[field] = "different"
    raw = json.dumps(descriptor, sort_keys=True).encode()
    artifact.write_bytes(raw)
    with pytest.raises(ValueError, match="pinned provider/model"):
        invoke(local_binding, semantic_metadata_tokenizer_sha256=_sha(raw))
    assert local_binding[0]["calls"] == []


@pytest.mark.parametrize("raw", [b"{", b'{"schema":"a","schema":"b"}', b'{"model":NaN}'])
def test_hash_bound_malformed_descriptor_cannot_reach_provider(local_binding, raw):
    local_binding[1].write_bytes(raw)
    with pytest.raises(ValueError):
        invoke(local_binding, semantic_metadata_tokenizer_sha256=_sha(raw))
    assert local_binding[0]["calls"] == []


@pytest.mark.parametrize("which", ["descriptor", "asset"])
def test_provider_initialization_asset_drift_refuses_dispatch(local_binding, monkeypatch, capsys, which):
    from ipfs_accelerate_py import llm_router
    case, artifact, _, descriptor = local_binding
    changed = artifact if which == "descriptor" else artifact.parent / descriptor["asset_path"]

    def initialize(*_args, **_kwargs):
        changed.write_bytes(changed.read_bytes() + b" ")
        return object()

    monkeypatch.setattr(llm_router, "get_llm_provider", initialize)
    with pytest.raises(ValueError):
        invoke(local_binding)
    assert case["calls"] == []
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    failed = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
    assert failed["failure_phase"] == "semantic_metadata_tokenizer"
    assert failed["semantic_metadata_tokenizer_dispatch_rechecked"] is False


def test_drift_during_count_is_not_rescued_by_original_transport(local_binding, monkeypatch):
    case, artifact, _, descriptor = local_binding
    asset = artifact.parent / descriptor["asset_path"]
    original = local_codec.OwnerPromptTokenizer.count
    seen = []

    def change_after_count(self, text):
        result = original(self, text)
        seen.append(text)
        asset.write_bytes(asset.read_bytes() + b" ")
        return result

    monkeypatch.setattr(local_codec.OwnerPromptTokenizer, "count", change_after_count)
    with pytest.raises(ValueError):
        invoke(local_binding)
    assert len(seen) == 1
    assert case["calls"] == []


def test_strict_world_packet_route_refuses_metadata_descriptor_before_loading(strict_dispatch_case, tmp_path, monkeypatch):
    artifact, pin, _ = make_binding(tmp_path, model="pinned")
    monkeypatch.setattr(runner, "load_semantic_metadata_tokenizer", lambda **_: pytest.fail("loaded on world route"))
    with pytest.raises(ValueError, match="independently selected Codex"):
        runner.run(prompt=TASK, provider="codex_cli", model="pinned", timeout=30,
            max_output_tokens=128, world_task_packet=strict_dispatch_case[2],
            semantic_metadata_tokenizer_artifact=artifact, semantic_metadata_tokenizer_sha256=pin)
    assert strict_dispatch_case[3]["seen"] == []


def test_direct_census_records_local_estimate_without_discovering_provider(local_binding, monkeypatch):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: pytest.fail("census discovered provider"))
    _, receipt = invoke(local_binding, census_only=True)
    assert receipt["schema"] == "router-semantic-metadata-census@2"
    assert receipt["semantic_metadata_tokenizer_source"]["qualification"] == "estimate"
    assert receipt["semantic_metadata_tokenizer_dispatch_rechecked"] is False
    assert receipt["provider_calls"] == 0
    assert local_binding[0]["calls"] == []


def test_cli_forwards_only_explicit_metadata_descriptor_pair(tmp_path, monkeypatch):
    artifact = tmp_path / "descriptor.json"
    pin = "a" * 64
    seen = []
    monkeypatch.setattr(sys, "argv", ["router", "--model", "authored-model", "--semantic-repository", str(tmp_path),
        "--semantic-metadata-view", "common-bindings@1", "--semantic-metadata-tokenizer-artifact", str(artifact),
        "--semantic-metadata-tokenizer-sha256", pin])
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"hello")))
    monkeypatch.setattr(runner, "run", lambda **kwargs: (seen.append(kwargs) or "authored", {}))
    assert runner.main() == 0
    assert seen[0]["semantic_metadata_tokenizer_artifact"] == artifact
    assert seen[0]["semantic_metadata_tokenizer_sha256"] == pin
    assert seen[0]["prompt_tokenizer_artifact"] is None
    assert seen[0]["prompt_tokenizer_sha256"] == ""


@pytest.mark.parametrize("flags", [["--semantic-metadata-tokenizer-artifact", "/descriptor.json"],
    ["--semantic-metadata-tokenizer-sha256", "a" * 64],
    ["--semantic-metadata-tokenizer-artifact", "/descriptor.json", "--semantic-metadata-tokenizer-sha256", ""]])
def test_cli_requires_nonempty_exact_pair_before_reading_prompt(monkeypatch, flags):
    class ForbiddenInput:
        @property
        def buffer(self):
            pytest.fail("invalid binding read prompt")
    monkeypatch.setattr(sys, "argv", ["router", "--model", "authored-model", *flags])
    monkeypatch.setattr(sys, "stdin", ForbiddenInput())
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
