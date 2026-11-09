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
    TASK, counter, dispatch_case, invoke, required, richer_world_case, scheduler, world_session,
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
def compacted_case(dispatch_case):
    return richer_world_case(dispatch_case)


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
    assert diagnostic["recorded_token_measurement_present"] is (mode != "helpers")
    if mode != "helpers":
        measurement = receipt["world_task_packet"]["prompt_measurement"]
        assert measurement["measurement_status"] == ("estimated" if mode == "plain" else "measured")
    assert diagnostic["token_counts_replayed"] is False
    assert diagnostic["provider_delivery_verified"] is False
    assert diagnostic["model_prompt_sha256"] == receipt["model_prompt_sha256"]


@pytest.mark.parametrize("count_kind", ["tokenizer", "estimate"])
def test_ordinary_completion_replays_exact_separate_native_schema(dispatch_case, monkeypatch, count_kind):
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
                              prompt_token_counter=counter(count_kind=count_kind), max_input_tokens=100_000)
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
    measurement = receipt["world_task_packet"]["prompt_measurement"]
    assert measurement["measurement_status"] == ("estimated" if count_kind == "estimate" else "measured")
    if count_kind == "estimate":
        assert measurement["selected_input_tokens"] is None
        assert measurement["estimated_native_output_schema_tokens"] == len(schema)
        assert measurement["tokenizer_compatibility_basis"] == "caller_declaration"
        assert measurement["token_budget_basis"] == "estimated_content_tokens"
        assert measurement["token_budget_guaranteed"] is False
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


def test_default_offline_estimate_replays_complete_bounded_input_without_recount(dispatch_case, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner

    source, session, packet, state = dispatch_case
    frozen = freeze_repository_world_packet(packet)
    complete, _ = runner.render_model_prompt(prompt=packet.prompt, purpose="coding", workspace=source, world_packet=True)
    local, _ = automatic_prompt_tokenizer.resolve_prompt_tokenizer(provider="codex_cli", model="pinned")
    limit = local.count(complete)
    _output, receipt = invoke(dispatch_case, max_input_tokens=limit, compact_world_prompt=False)
    measured = receipt["world_task_packet"]["prompt_measurement"]
    assert measured["measurement_kind"] == "declared_local_estimate"
    assert measured["measurement_status"] == "estimated"
    assert measured["selected_estimated_input_tokens"] == limit
    assert measured["selected_input_tokens"] is None
    assert receipt["prompt_tokenizer_source"]["offline"] is True
    assert receipt["prompt_tokenizer_source"]["provider_usage"] is False
    assert state["seen"][0][0] == complete
    session.close()
    (source / "calc.py").write_text("def add(value):\n    return value + 100\n")

    def denied(**kwargs):
        raise AssertionError("historical replay must not resolve or recount a tokenizer")
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", denied)
    restored = _restore(_join(frozen, receipt))
    assert restored.model_prompt == complete
    assert restored.receipt()["recorded_token_measurement_present"] is True
    assert restored.receipt()["token_counts_replayed"] is False
    assert restored.receipt()["provider_delivery_verified"] is False
    assert restored.receipt()["completion_authority"] is False


def test_rehashed_estimated_measurement_tampering_is_refused(captured):
    mutations = {
        "missing_field": "measurement_binding", "unknown_field": "measurement_binding",
        "exact_status": "measurement_binding", "exact_kind": "measurement_binding",
        "exact_count_kind": "measurement_estimate", "bool_content": "measurement_counts",
        "bool_schema": "measurement_counts", "bool_total": "measurement_counts",
        "exact_content": "measurement_estimate", "exact_schema": "measurement_estimate",
        "sum": "measurement_counts", "foreign_provider": "measurement_tokenizer",
        "foreign_model": "measurement_tokenizer", "schema_tokens": "measurement_counts",
        "schema_hash": "measurement_binding", "prompt_hash": "model_prompt_mismatch",
        "over_budget": "measurement_counts", "bool_limit": "measurement_limit",
        "budget_basis": "measurement_estimate", "guaranteed_budget": "measurement_estimate",
        "blank_basis": "measurement_estimate", "invalid_basis_type": "measurement_estimate",
        "error": "measurement_counts", "completion": "measurement_authority",
        "provider_framing": "measurement_authority", "missing_binding": "measurement_counts",
        "provider_claim": "measurement_binding",
    }
    for mutation, reason in mutations.items():
        changed = json.loads(captured["artifact"])
        measured = changed["invocation"]["world_task_packet"]["prompt_measurement"]
        assert measured["measurement_status"] == "estimated"
        if mutation == "missing_field":
            del measured["selected_estimated_input_tokens"]
        elif mutation == "unknown_field":
            measured["unbound_counter_claim"] = True
        elif mutation == "exact_status":
            measured["measurement_status"] = "measured"
        elif mutation == "exact_kind":
            measured["measurement_kind"] = "declared_local_tokenizer"
        elif mutation == "exact_count_kind":
            measured["token_count_kind"] = "tokenizer"
        elif mutation.startswith("bool_") and mutation != "bool_limit":
            field = {"bool_content": "selected_estimated_content_tokens",
                     "bool_schema": "estimated_native_output_schema_tokens",
                     "bool_total": "selected_estimated_input_tokens"}[mutation]
            measured[field] = True
        elif mutation == "exact_content":
            measured["selected_content_tokens"] = 1
        elif mutation == "exact_schema":
            measured["native_output_schema_tokens"] = 0
        elif mutation == "sum":
            measured["selected_estimated_input_tokens"] += 1
        elif mutation in {"foreign_provider", "foreign_model"}:
            measured["tokenizer_binding"][mutation.removeprefix("foreign_")] = "foreign"
        elif mutation == "schema_tokens":
            measured["estimated_native_output_schema_tokens"] = 1
            measured["selected_estimated_input_tokens"] += 1
        elif mutation == "schema_hash":
            measured["native_output_schema_sha256"] = "0" * 64
        elif mutation == "prompt_hash":
            measured["model_prompt_sha256"] = "0" * 64
        elif mutation == "over_budget":
            measured["max_input_tokens"] = measured["selected_estimated_input_tokens"] - 1
            measured["token_budget_basis"] = "estimated_content_tokens"
        elif mutation == "bool_limit":
            measured["max_input_tokens"] = True
            measured["token_budget_basis"] = "estimated_content_tokens"
        elif mutation == "budget_basis":
            measured["token_budget_basis"] = "estimated_content_tokens"
        elif mutation == "guaranteed_budget":
            measured["token_budget_guaranteed"] = True
        elif mutation == "blank_basis":
            measured["tokenizer_compatibility_basis"] = ""
        elif mutation == "invalid_basis_type":
            measured["tokenizer_compatibility_basis"] = True
        elif mutation == "error":
            measured["measurement_error_type"] = "RuntimeError"
        elif mutation == "completion":
            measured["completion_authority"] = True
        elif mutation == "provider_framing":
            measured["provider_framing_measured"] = True
        elif mutation == "missing_binding":
            measured["tokenizer_binding"] = None
        elif mutation == "provider_claim":
            measured["provider_token_usage_verified"] = True
        with pytest.raises(replay.RepositoryWorldInvocationReplayError) as rejected:
            _restore(_encoded(changed))
        assert rejected.value.code == "world_replay_" + reason, mutation


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


@pytest.mark.parametrize("count_kind", ["tokenizer", "estimate"])
@pytest.mark.parametrize("mode", ["plain", "helpers_reply"])
def test_factored_world_input_replays_after_close_without_tokenizer(compacted_case, monkeypatch, count_kind, mode):
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
        project_repository_world_prompt, restore_repository_world_prompt,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.coding_reply_contract import (
        apply_coding_reply_contract, coding_completion_schema,
    )

    source, session, packet, state = compacted_case
    frozen = freeze_repository_world_packet(packet)
    options = {"prompt_token_counter": counter(count_kind=count_kind)}
    if mode == "helpers_reply":
        from ipfs_accelerate_py import llm_router
        from ipfs_accelerate_py.cli_runtime.cli_metadata import (
            get_last_cli_observation, set_last_cli_observation,
        )

        original = llm_router.generate_text

        def generate(prompt, **kwargs):
            original(prompt, **kwargs)
            schema = _encoded(kwargs["codex_output_schema"])
            observation = dict(get_last_cli_observation("codex_cli"))
            observation.update(codex_output_schema_sha256=_sha(schema), codex_output_schema_bytes=len(schema))
            set_last_cli_observation("codex_cli", observation)
            return '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'

        monkeypatch.setattr(llm_router, "generate_text", generate)
        options.update(tool_output_policy="artifact-preview@2",
                       repository_source_policy="repository-source-compact@1",
                       coding_reply_mode="ordinary-completion@1")
    _output, receipt = invoke(compacted_case, **options)
    marker = receipt["world_task_packet"]["prompt_compaction"]
    assert marker["enabled"] is True and marker["selected_mode"] == "compact"
    view = project_repository_world_prompt(packet.prompt)
    assert marker["projection"] == view.receipt
    assert restore_repository_world_prompt(view.provider_prompt, view.receipt) == packet.prompt
    native_model, _ = runner.render_model_prompt(prompt=packet.prompt, purpose="coding",
        workspace=source, world_packet=True, tool_output_binding=receipt.get("tool_output_policy"),
        repository_source_binding=receipt.get("repository_source_policy"))
    native_model, _ = apply_coding_reply_contract(model_prompt=native_model,
        mode=options.get("coding_reply_mode", "legacy"), purpose="coding", provider="codex_cli")
    expected = state["seen"][0][0]
    assert len(expected.encode()) < len(native_model.encode())
    assert receipt["router_prompt_sha256"] == _sha(packet.prompt.encode())
    assert marker["selection"]["selected_representation"] == "semantic_metadata"
    session.close()
    (source / "calc.py").write_text("def add(value):\n    return value + 100\n")

    def denied(*args, **kwargs):
        raise AssertionError("historical replay must not resolve or execute a tokenizer")

    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", denied)
    monkeypatch.setattr(runner.PromptTokenCounter, "count", denied)
    restored = _restore(_join(frozen, receipt))
    assert restored.model_prompt == expected
    diagnostic = restored.receipt()
    assert diagnostic["recorded_prompt_compaction_present"] is True
    assert diagnostic["recorded_prompt_compaction_sha256"] == _sha(_encoded(marker))
    assert diagnostic["token_counts_replayed"] is diagnostic["provider_delivery_verified"] is False
    if mode == "helpers_reply":
        schema = _encoded(coding_completion_schema())
        assert diagnostic["native_output_schema_sha256"] == _sha(schema)
        assert diagnostic["native_output_schema_bytes"] == len(schema)
        assert receipt["coding_reply_contract"]["response_validated"] is True
        assert diagnostic["provider_response_replayed"] is False
        assert receipt["tool_output_policy"] and receipt["repository_source_policy"]


def test_explicit_world_compaction_optout_and_absent_legacy_marker_replay(dispatch_case):
    packet = freeze_repository_world_packet(dispatch_case[2])
    _output, receipt = invoke(dispatch_case, prompt_token_counter=counter(), compact_world_prompt=False)
    marker = receipt["world_task_packet"]["prompt_compaction"]
    assert marker == {"schema": "supervisor-world-prompt-compaction@1", "enabled": False,
                      "projection": None, "selection": None, "selected_mode": "native"}
    expected = dispatch_case[3]["seen"][0][0]
    assert _restore(_join(packet, receipt)).model_prompt == expected
    legacy = deepcopy(receipt)
    del legacy["world_task_packet"]["prompt_compaction"]
    restored = _restore(_join(packet, legacy))
    assert restored.model_prompt == expected
    assert restored.receipt()["recorded_prompt_compaction_present"] is False


def test_rehashed_world_compaction_and_selection_tampering_is_refused(compacted_case):
    packet = freeze_repository_world_packet(compacted_case[2])
    _output, receipt = invoke(compacted_case, prompt_token_counter=counter())
    raw = _join(packet, receipt)
    mutations = ("marker_field", "marker_type", "selected_mode", "projection", "missing_projection",
        "projection_field", "missing_selection", "selection_field", "candidate_hash", "candidate_bytes",
        "candidate_bool_count", "candidate_sum", "chosen_count", "selection_winner", "selection_reason",
        "tokenizer_provider", "tokenizer_model", "tokenizer_id", "count_kind", "count_scope",
        "schema_hash", "completion", "unknown_candidate", "coherent_count_change", "missing_measurement")
    for mutation in mutations:
        changed = json.loads(raw)
        marker = changed["invocation"]["world_task_packet"]["prompt_compaction"]
        assert marker["selected_mode"] == "compact"
        selected = marker["selection"]
        candidate = selected["candidates"]["semantic_metadata"]
        if mutation == "marker_field":
            marker["unbound_authority"] = True
        elif mutation == "marker_type":
            marker["enabled"] = 1
        elif mutation == "selected_mode":
            marker["selected_mode"] = "native"
        elif mutation == "projection":
            marker["projection"]["native_prompt_sha256"] = "0" * 64
        elif mutation == "missing_projection":
            marker["projection"] = None
        elif mutation == "projection_field":
            marker["projection"]["unbound_common_fields"] = True
        elif mutation == "missing_selection":
            marker["selection"] = None
        elif mutation == "selection_field":
            selected["unbound_counter_authority"] = True
        elif mutation == "candidate_hash":
            candidate["model_prompt_sha256"] = "0" * 64
        elif mutation == "candidate_bytes":
            candidate["model_prompt_bytes"] += 1
        elif mutation == "candidate_bool_count":
            candidate["content_tokens"] = True
        elif mutation == "candidate_sum":
            candidate["input_tokens"] += 1
        elif mutation == "chosen_count":
            selected["selected_content_tokens"] += 1
        elif mutation == "selection_winner":
            selected["selected_representation"] = "native"
        elif mutation == "selection_reason":
            selected["selection_reason"] = "native_not_larger"
        elif mutation.startswith("tokenizer_"):
            key = "tokenizer_id" if mutation == "tokenizer_id" else mutation.removeprefix("tokenizer_")
            selected["tokenizer_binding"][key] = "foreign"
        elif mutation == "count_kind":
            selected["token_count_kind"] = "estimate"
        elif mutation == "count_scope":
            selected["count_scope"] = "all_provider_tokens"
        elif mutation == "schema_hash":
            selected["native_output_schema_sha256"] = "0" * 64
        elif mutation == "completion":
            selected["completion_authority"] = True
        elif mutation == "unknown_candidate":
            selected["candidates"]["unbound"] = deepcopy(candidate)
        elif mutation == "coherent_count_change":
            candidate["content_tokens"] += 1
            candidate["input_tokens"] += 1
            selected["selected_content_tokens"] += 1
            selected["selected_input_tokens"] += 1
        elif mutation == "missing_measurement":
            changed["invocation"]["world_task_packet"]["prompt_measurement"] = None
        with pytest.raises(ValueError):
            _restore(_encoded(changed))
