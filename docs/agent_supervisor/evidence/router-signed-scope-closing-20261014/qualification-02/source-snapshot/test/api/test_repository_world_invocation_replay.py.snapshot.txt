"""Actual runner inputs replay after dispatch without asserting live authority.

Source indexing and prompt rendering are real. Provider behavior and byte-based
token counts are authored fixtures; these tests do not measure session savings.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_invocation_replay as replay
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet_replay import (
    freeze_repository_world_packet,
)
from test.api.test_repository_world_dispatch import (
    TASK, counter, dispatch_case, invoke, required, scheduler, world_session,
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _join(packet, receipt):
    return replay.freeze_repository_world_invocation(packet_artifact=packet,
        expected_packet_sha256=_sha(packet), invocation_receipt=receipt)


def _restore(data):
    return replay.replay_repository_world_invocation(data, expected_sha256=_sha(data))


@pytest.fixture
def captured(dispatch_case):
    # Freeze must precede a worker's possible edits or owner shutdown.
    packet = freeze_repository_world_packet(dispatch_case[2])
    _output, receipt = invoke(dispatch_case)
    return dict(case=dispatch_case, packet=packet, receipt=receipt,
                artifact=_join(packet, receipt))


@pytest.mark.parametrize("mode", ["plain", "measured", "helpers"])
def test_actual_runner_input_replays_after_source_owner_closes(dispatch_case, mode):
    source, session, _packet, state = dispatch_case
    frozen = freeze_repository_world_packet(dispatch_case[2])
    changes = {}
    if mode == "measured":
        changes["prompt_token_counter"] = counter()
    elif mode == "helpers":
        changes.update(tool_output_policy="artifact-preview@2",
                       repository_source_policy="repository-source-compact@1")
    _output, receipt = invoke(dispatch_case, **changes)
    assert len(state["seen"]) == 1
    expected = state["seen"][0][0]
    session.close()
    (source / "calc.py").write_text("def add(value):\n    return value + 100\n")
    artifact = _join(frozen, receipt)
    restored = _restore(artifact)
    assert restored.model_prompt == expected
    assert restored.artifact_sha256 == _sha(artifact)
    diagnostic = restored.receipt()
    assert diagnostic["historical_input_reconstructed"] is True
    assert diagnostic["recorded_token_measurement_present"] is (mode == "measured")
    assert diagnostic["token_counts_replayed"] is False
    assert diagnostic["provider_delivery_verified"] is False
    assert diagnostic["model_prompt_sha256"] == receipt["model_prompt_sha256"]


def test_ordinary_completion_replays_exact_separate_native_schema(dispatch_case, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import (
        get_last_cli_observation, set_last_cli_observation,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.coding_reply_contract import coding_completion_schema

    original = llm_router.generate_text

    def generate(prompt, **kwargs):
        original(prompt, **kwargs)
        schema = _encoded(kwargs["codex_output_schema"])
        observation = dict(get_last_cli_observation("codex_cli"))
        observation.update(codex_output_schema_sha256=_sha(schema), codex_output_schema_bytes=len(schema))
        set_last_cli_observation("codex_cli", observation)
        return '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'

    monkeypatch.setattr(llm_router, "generate_text", generate)
    packet = freeze_repository_world_packet(dispatch_case[2])
    _output, receipt = invoke(dispatch_case, coding_reply_mode="ordinary-completion@1",
                              prompt_token_counter=counter())
    assert receipt["coding_reply_contract"]["response_validated"] is True
    raw = _join(packet, receipt)
    restored = _restore(raw)
    assert restored.model_prompt == dispatch_case[3]["seen"][0][0]
    diagnostic = restored.receipt()
    schema = _encoded(coding_completion_schema())
    assert diagnostic["native_output_schema_sha256"] == _sha(schema)
    assert diagnostic["native_output_schema_bytes"] == len(schema)
    assert diagnostic["provider_response_replayed"] is False
    assert diagnostic["token_counts_replayed"] is False
    # Rehashed edits cannot alter the reconstructed reply or its schema.
    for field in ("instruction_sha256", "native_output_schema_sha256", "mode"):
        changed = json.loads(raw)
        changed["invocation"]["coding_reply_contract"][field] = "altered"
        with pytest.raises(ValueError):
            _restore(_encoded(changed))
    changed = json.loads(raw)
    changed["invocation"]["world_task_packet"]["prompt_measurement"]["native_output_schema_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        _restore(_encoded(changed))


def test_diagnostics_are_immutable_copies_and_export_no_content_or_authority(captured):
    restored = _restore(captured["artifact"])
    with pytest.raises(FrozenInstanceError):
        restored.model_prompt = "changed"
    diagnostic = restored.receipt()
    for field in ("source_currentness_verified", "owner_currentness_verified", "owner_signature_replayed",
                  "compiler_token_counts_replayed", "provider_delivery_verified", "provider_response_replayed",
                  "token_counts_replayed", "proof_authority", "execution_authority", "completion_authority",
                  "live_packet_issued", "canonical_state_mutated"):
        assert diagnostic[field] is False
    encoded = json.dumps(diagnostic)
    assert TASK not in encoded
    assert str(captured["case"][0]) not in encoded
    assert captured["case"][2].prompt not in encoded
    diagnostic["completion_authority"] = True
    assert restored.receipt()["completion_authority"] is False


@pytest.mark.parametrize("mutation", ["prompt_hash", "native_hash", "router_hash", "model_hash",
    "advisory_hash", "workspace", "packet_link", "packet_receipt", "route", "schema",
    "unknown_artifact_field", "unknown_invocation_field", "bool_byte_count", "invocation_id", "dispatch_claim"])
def test_rehashed_invocation_tampering_is_refused(captured, mutation):
    value = json.loads(captured["artifact"])
    invocation = value["invocation"]
    hashes = {"prompt_hash": "prompt_sha256", "native_hash": "native_prompt_sha256",
              "router_hash": "router_prompt_sha256", "model_hash": "model_prompt_sha256",
              "advisory_hash": "workspace_advisory_sha256"}
    if mutation in hashes:
        invocation[hashes[mutation]] = "0" * 64
    elif mutation == "workspace":
        invocation["workspace"] += "-foreign"
    elif mutation == "packet_link":
        value["packet_artifact_sha256"] = "0" * 64
    elif mutation == "packet_receipt":
        invocation["world_task_packet"]["context_cid"] = "sha256:" + "0" * 64
    elif mutation == "route":
        invocation["provider"] = "foreign"
    elif mutation == "schema":
        value["schema"] = "repository-world-invocation-artifact@999"
    elif mutation == "unknown_artifact_field":
        value["allow_dispatch"] = True
    elif mutation == "unknown_invocation_field":
        invocation["world_packet_override"] = "unbound"
    elif mutation == "bool_byte_count":
        invocation["model_prompt_bytes"] = True
    elif mutation == "invocation_id":
        invocation["invocation_id"] = "invalid-invocation"
    elif mutation == "dispatch_claim":
        invocation["world_task_packet"].update(provider_attempted=True, dispatch_rechecked=False)
    with pytest.raises(ValueError):
        _restore(_encoded(value))


def test_rehashed_helper_policy_changes_are_refused(dispatch_case):
    packet = freeze_repository_world_packet(dispatch_case[2])
    _output, receipt = invoke(dispatch_case, tool_output_policy="artifact-preview@2",
                              repository_source_policy="repository-source-compact@1")
    raw = _join(packet, receipt)
    for policy in ("tool_output_policy", "repository_source_policy"):
        changed = json.loads(raw)
        changed["invocation"][policy]["instruction_sha256"] = "0" * 64
        with pytest.raises(ValueError):
            _restore(_encoded(changed))


def test_freeze_refuses_incompatible_input_transformations(captured):
    for field in ("semantic_translation", "source_input_view", "query_reference_view",
                  "source_view_advisory_policy", "public_instruction", "source_lookup_retrieval",
                  "numerical_obligation_retrieval", "source_view_retrieval", "representation_displacement",
                  "semantic_metadata_candidate", "prompt_selection"):
        changed = deepcopy(captured["receipt"])
        changed[field] = {}
        with pytest.raises(ValueError, match="incompatible"):
            _join(captured["packet"], changed)


def test_recorded_measurement_remains_structurally_bound_without_claiming_recount(dispatch_case):
    packet = freeze_repository_world_packet(dispatch_case[2])
    _output, receipt = invoke(dispatch_case, prompt_token_counter=counter())
    raw = _join(packet, receipt)
    for mutation in ("missing_field", "unknown_field", "authority", "provider_framing", "count_scope",
                     "bool_count", "sum", "foreign_model", "foreign_provider", "schema_tokens", "over_budget"):
        changed = json.loads(raw)
        measured = changed["invocation"]["world_task_packet"]["prompt_measurement"]
        if mutation == "missing_field":
            del measured["selected_content_tokens"]
        elif mutation == "unknown_field":
            measured["unbound_counter_claim"] = True
        elif mutation == "authority":
            measured["completion_authority"] = True
        elif mutation == "provider_framing":
            measured["provider_framing_measured"] = True
        elif mutation == "count_scope":
            measured["count_scope"] = "all_provider_input"
        elif mutation == "bool_count":
            measured["selected_content_tokens"] = True
        elif mutation == "sum":
            measured["selected_input_tokens"] += 1
        elif mutation in {"foreign_model", "foreign_provider"}:
            measured["tokenizer_binding"][mutation.removeprefix("foreign_")] = "foreign"
        elif mutation == "schema_tokens":
            measured["native_output_schema_tokens"] = 1
            measured["selected_input_tokens"] += 1
        elif mutation == "over_budget":
            measured["max_input_tokens"] = measured["selected_input_tokens"] - 1
        with pytest.raises(ValueError):
            _restore(_encoded(changed))


def test_failed_recorded_counter_replays_prompt_without_recount(dispatch_case):
    def failed(_text):
        raise RuntimeError("private counter failure")
    packet = freeze_repository_world_packet(dispatch_case[2])
    _output, receipt = invoke(dispatch_case, prompt_token_counter=counter(count_tokens=failed))
    assert receipt["world_task_packet"]["prompt_measurement"]["measurement_status"] == "failed"
    restored = _restore(_join(packet, receipt))
    assert restored.model_prompt == dispatch_case[3]["seen"][0][0]
    assert restored.receipt()["recorded_token_measurement_present"] is True
    assert restored.receipt()["token_counts_replayed"] is False
    assert "private counter failure" not in json.dumps(restored.receipt())


def test_external_packet_and_invocation_pins_are_required(captured):
    for digest in ("", "0" * 64, "G" * 64, True, None):
        with pytest.raises(ValueError):
            replay.replay_repository_world_invocation(captured["artifact"], expected_sha256=digest)
        with pytest.raises(ValueError):
            replay.freeze_repository_world_invocation(packet_artifact=captured["packet"],
                expected_packet_sha256=digest, invocation_receipt=captured["receipt"])
