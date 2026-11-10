"""Historical, pinned audit of matched full/focused residual model inputs.

The full original invocation is retained, including failed/partial observations.
Hashes bind historical bytes, not owner authenticity, freshness, delivery, usage,
or the relevance of omitted optional evidence. A caller may explicitly supply a
trusted offline counter; no counter or tokenizer path is loaded from receipts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

from .repository_world_packet_replay import _decode_canonical, replay_repository_world_packet
from .repository_world_presentation_invocation import _wire as _bounded_wire
from .supervisor_meta_index import public_replay_without_metadata

SCHEMA = "repository-world-residual-selection-artifact@1"
RECEIPT_SCHEMA = "repository-world-residual-selection-replay@1"
MAX_BYTES = 6 * 1024 * 1024
MAX_INVOCATION_BYTES = 1024 * 1024
SELECTION_FIELD = "world_task_residual_selection"
_ARTIFACT_FIELDS = frozenset({"schema", "mode", "full_packet_artifact",
    "full_packet_artifact_sha256", "focused_packet_artifact", "focused_packet_artifact_sha256",
    "invocation", "invocation_sha256"})
FALSE_FIELDS = frozenset({"source_currentness_verified", "owner_currentness_verified",
    "owner_signature_replayed", "compiler_token_counts_replayed", "provider_delivery_verified",
    "provider_response_replayed", "provider_usage_verified", "billing_verified",
    "provider_tokenizer_equivalence_verified", "model_consumption_verified", "proof_authority",
    "execution_authority", "completion_authority", "live_packet_issued", "canonical_state_mutated",
    "semantic_equivalence_verified", "omission_authority", "total_session_token_savings_verified"})


class RepositoryWorldResidualSelectionReplayError(ValueError):
    """Closed codes without task/source text or callback exception contents."""


def _need(condition, code):
    if not condition:
        raise RepositoryWorldResidualSelectionReplayError("world_residual_selection_replay_" + code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pin(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _wire(value, *, maximum=MAX_BYTES):
    return _bounded_wire(value, maximum=maximum)


def invocation_receipt_sha256(receipt):
    """Digest the complete canonical receipt; pretty-file hashes differ."""
    return _sha(_wire(receipt, maximum=MAX_INVOCATION_BYTES))


def _invocation_input(receipt, expected_sha256):
    _need(_pin(expected_sha256), "original_invocation_pin_required")
    if type(receipt) is bytes:
        # Bytes must have the existing canonical archival spelling. Their
        # independently supplied pin is preserved without normalizing bytes.
        value = _decode_canonical(receipt, expected_sha256, max_bytes=MAX_INVOCATION_BYTES)
    else:
        _need(type(receipt) is dict, "original_invocation_object_required")
        raw = _wire(receipt, maximum=MAX_INVOCATION_BYTES)
        _need(_sha(raw) == expected_sha256, "original_invocation_pin")
        value = json.loads(raw)
    _need(invocation_receipt_sha256(value) == expected_sha256, "original_invocation_pin")
    return value


def _packet(raw, expected_sha256):
    from .repository_world_packet_replay import MAX_BYTES as PACKET_MAX_BYTES
    material = _decode_canonical(raw, expected_sha256, max_bytes=PACKET_MAX_BYTES)
    packet = replay_repository_world_packet(raw, expected_sha256=expected_sha256)
    _need("validation_profile" not in material["recipe"]
        and type(packet.recorded_packet_receipt().get("validation_count")) is int
        and packet.recorded_packet_receipt()["validation_count"] > 0,
        "strict_residual_packet_required")
    return material, packet


def _historical_pair_receipt(full, focused):
    """Rebuild recorded joins, including the original failure-free baseline.

    These data constructors never verify an owner signature or compile/admit a
    new live capsule. Optional retrieval provenance is allowed to differ.
    """
    from ..context.context_contracts import ContextCapsule
    from ..context.context_compiler import ContextCompilationReceipt, ContextCompileResult
    from .repository_world_packet import _render_repository_world_packet
    from .repository_world_residual_selection import (
        _arguments, _population, validate_repository_world_residual_alternatives_receipt,
    )
    before, after = full["recipe"], focused["recipe"]
    _need("residual_context" in before and _wire(before) == _wire(after), "matched_recipe_and_residual")
    capsules = [ContextCapsule.from_dict(value["capsule"]) for value in (full, focused)]
    first, second = capsules
    for name in ("repository_id", "tree_id", "objective_id", "objective_revision", "policy_id",
            "policy_revision", "goal", "scope", "acceptance", "caller", "stage", "budget"):
        values = [getattr(capsule, name) for capsule in capsules]
        if name == "budget":
            values = [value.to_record() for value in values]
        # Contract mappings may be immutable proxies. to_record returns the
        # already validated JSON representation without coercing caller data.
        else:
            values = [value["capsule"][name] for value in (full, focused)]
        _need(_wire(values[0]) == _wire(values[1]), "matched_task_core")
    _need(_wire(full["capsule"]["authority"]["task"])
        == _wire(focused["capsule"]["authority"]["task"]), "matched_task_authority")
    advisory = {"query", "ranking", "population_count", "matching_count", "selected_count", "omitted_count", "truncated"}
    provenance = [{key: item for key, item in value["capsule"]["authority"]["repository_context"].items()
        if key not in advisory} for value in (full, focused)]
    for name in ("required_reference_ids", "required_coverage_ids"):
        for record in provenance:
            values = record[name]
            _need(type(values) is list and all(type(value) is str for value in values)
                and len(values) == len(set(values)), "source_required_population")
            record[name] = sorted(values)
    _need(_wire(provenance[0]) == _wire(provenance[1]), "matched_source_provenance")
    required = [{ref.reference_id: ref.canonical_bytes() for ref in capsule.evidence if ref.required}
        for capsule in capsules]
    _need(required[0] == required[1], "matched_required_references")
    full_receipt, focused_receipt = full["packet_receipt"], focused["packet_receipt"]
    _need(_wire(full_receipt["source_binding"]) == _wire(focused_receipt["source_binding"])
        and _wire(full_receipt.get("intent_world")) == _wire(focused_receipt.get("intent_world")),
        "matched_source_and_intent")
    bare_recipe = {key: value for key, value in before.items() if key != "residual_context"}
    compilation_receipt = ContextCompilationReceipt.from_dict(full["compilation_receipt"])
    compilation = ContextCompileResult(first, compilation_receipt, compilation_receipt.decisions)
    with public_replay_without_metadata():
        _bare_prompt, bare_receipt, _native = _render_repository_world_packet(
            compilation=compilation, intent_world_receipt=full_receipt.get("intent_world"),
            **_arguments(bare_recipe))
    lineage = before["residual_context"]
    receipt = dict(schema="repository-world-residual-alternatives@1",
        baseline_prompt_sha256=bare_receipt["prompt_sha256"], baseline_prompt_bytes=bare_receipt["prompt_bytes"],
        full_prompt_sha256=full_receipt["prompt_sha256"], full_prompt_bytes=full_receipt["prompt_bytes"],
        focused_prompt_sha256=focused_receipt["prompt_sha256"], focused_prompt_bytes=focused_receipt["prompt_bytes"],
        baseline_context_cid=first.capsule_id, full_context_cid=first.capsule_id,
        focused_context_cid=second.capsule_id, residual_packet_id=lineage["packet_id"],
        residual_context_id=lineage["context_id"], residual_context_sha256=lineage["context_sha256"],
        source_binding=full_receipt["source_binding"], task_text_sha256=bare_receipt["task_text_sha256"],
        required_population_sha256=_sha(_wire(_population(first, True))),
        full_optional_population_sha256=_sha(_wire(_population(first, False))),
        focused_optional_population_sha256=_sha(_wire(_population(second, False))),
        required_reference_count=len(required[0]), full_optional_count=len(_population(first, False)),
        focused_optional_count=len(_population(second, False)),
        full_baseline_optional_preserved=True, full_task_invariants_preserved=True, identical_residual_context=True,
        provider_calls=0, provider_input_tokens=None,
        **dict.fromkeys(("provider_retained_parent", "optional_omission_equivalence_verified",
            "token_savings_established", "default_adoption_authorized", "proof_authority",
            "execution_authority", "completion_authority"), False))
    return validate_repository_world_residual_alternatives_receipt(receipt)


def _render_alternative(packet, invocation):
    from .router_implementation_runner import render_model_prompt
    from .coding_reply_contract import apply_coding_reply_contract, coding_completion_schema
    # The selected input is first checked by ordinary replay, including its
    # exact workspace and complete reply contract. Reuse only those bindings.
    recorded_reply = invocation.get("coding_reply_contract")
    with public_replay_without_metadata():
        model_prompt, _advisory = render_model_prompt(prompt=packet.prompt,
            purpose="coding", workspace=Path(invocation["workspace"]), world_packet=True,
            tool_output_binding=invocation.get("tool_output_policy"),
            repository_source_binding=invocation.get("repository_source_policy"))
        model_prompt, reply = apply_coding_reply_contract(model_prompt=model_prompt,
            mode="legacy" if recorded_reply is None else recorded_reply["mode"],
            purpose="coding", provider=invocation["provider"])
    _need(len(model_prompt.encode("utf-8")) <= 512000, "alternative_model_byte_bound")
    schema = None if reply is None else coding_completion_schema()
    return model_prompt, schema


@dataclass(frozen=True, eq=False)
class ReplayedRepositoryWorldResidualSelection:
    """Historical bytes, with no source owner or live dispatch capability."""

    model_prompt: str = field(repr=False)
    artifact_sha256: str
    _invocation_json: str = field(repr=False)
    _receipt_json: str = field(repr=False)

    def recorded_invocation_receipt(self):
        return json.loads(self._invocation_json)

    def receipt(self):
        return json.loads(self._receipt_json)


def _replay(data, *, expected_sha256, token_counter):
    from .repository_world_invocation_replay import (
        freeze_repository_world_invocation, replay_repository_world_invocation,
    )
    from .repository_world_residual_selection import (
        validate_repository_world_residual_selection_receipt,
        repository_world_residual_selected_measurement,
        select_repository_world_residual_prompt,
    )
    artifact = _decode_canonical(data, expected_sha256, max_bytes=MAX_BYTES)
    _need(set(artifact) == _ARTIFACT_FIELDS and artifact["schema"] == SCHEMA
        and artifact["mode"] == "historical_only", "artifact_schema")
    materials, packets, packet_raws = {}, {}, {}
    for name in ("full", "focused"):
        raw = _wire(artifact[name + "_packet_artifact"])
        material, packet = _packet(raw, artifact[name + "_packet_artifact_sha256"])
        materials[name], packets[name], packet_raws[name] = material, packet, raw
    pair_receipt = _historical_pair_receipt(materials["full"], materials["focused"])
    invocation = _invocation_input(artifact["invocation"], artifact["invocation_sha256"])
    selection = invocation.get(SELECTION_FIELD)
    _need(type(selection) is dict and type(selection.get("dispatch_rechecked")) is bool,
        "selection_dispatch_shape")
    dispatch = selection["dispatch_rechecked"]
    selection = validate_repository_world_residual_selection_receipt(
        {key: value for key, value in selection.items() if key != "dispatch_rechecked"})
    _need(selection["alternatives_receipt_sha256"] == _sha(_wire(pair_receipt))
        and selection["provider"] == invocation["provider"]
        and selection["model"] == invocation["model"], "selection_binding")
    _need(dispatch is invocation["world_task_packet"].get("dispatch_rechecked"),
        "selection_dispatch_join")
    _need(invocation.get("world_task_presentation") is None, "incompatible_input")
    if invocation.get("world_packet_artifacts") is not None:
        from .repository_world_artifact_retention import unavailable_world_packet_artifacts
        unavailable = unavailable_world_packet_artifacts(invocation_id=invocation["invocation_id"],
            reason="retention_not_requested")
        _need(_wire(invocation["world_packet_artifacts"]) == _wire(unavailable), "incompatible_retention")

    selected = selection["selected_representation"]
    # Remove only the fully checked selection envelope. The ordinary legacy
    # adapter makes its documented closed projection of this DERIVED mapping;
    # the independently pinned original above remains complete and unchanged.
    derived = dict(invocation)
    del derived[SELECTION_FIELD]
    derived_artifact = freeze_repository_world_invocation(
        packet_artifact=packet_raws[selected],
        expected_packet_sha256=artifact[selected + "_packet_artifact_sha256"],
        invocation_receipt=derived)
    ordinary = replay_repository_world_invocation(derived_artifact,
        expected_sha256=_sha(derived_artifact))
    prompts, schemas = {}, {}
    for name in ("full", "focused"):
        prompts[name], schemas[name] = _render_alternative(packets[name], invocation)
        candidate = selection["candidates"][name]
        raw = prompts[name].encode("utf-8")
        _need(candidate["model_prompt_sha256"] == _sha(raw)
            and candidate["model_prompt_bytes"] == len(raw), "candidate_model_binding")
    _need(prompts[selected] == ordinary.model_prompt
        and _wire(schemas["full"]) == _wire(schemas["focused"]), "matched_complete_inputs")
    schema = schemas[selected]
    schema_sha = None if schema is None else _sha(_wire(schema))
    _need(selection["native_output_schema_sha256"] == schema_sha, "native_reply_schema")
    measurement = repository_world_residual_selected_measurement(selection)
    _need(_wire(invocation["world_task_packet"]["prompt_measurement"]) == _wire(measurement),
        "selected_measurement_join")

    counter_verified = False
    if token_counter is not None:
        # An explicit trusted caller supplies this object, never an archival
        # identity/path. Its callbacks cannot issue any native/live authority.
        from .router_implementation_runner import PromptTokenCounter
        fields = ("provider", "model", "tokenizer_id", "tokenizer_revision", "special_token_policy")
        _need(type(token_counter) is PromptTokenCounter
            and {name: getattr(token_counter, name) for name in fields} == selection["tokenizer_binding"],
            "trusted_counter_binding")
        _choice, reproduced, _measurement = select_repository_world_residual_prompt(
            alternatives_receipt=pair_receipt,
            full_complete_prompt=prompts["full"], focused_complete_prompt=prompts["focused"],
            provider=invocation["provider"], model=invocation["model"], token_counter=token_counter,
            max_input_tokens=selection["max_input_tokens"], native_output_schema=schema)
        _need(_wire(reproduced) == _wire(selection), "trusted_counter_reproduction")
        counter_verified = selection["measurement_status"] == "measured"
    observed = dict(schema=RECEIPT_SCHEMA, replay_mode="historical_only",
        artifact_sha256=expected_sha256,
        full_packet_artifact_sha256=artifact["full_packet_artifact_sha256"],
        focused_packet_artifact_sha256=artifact["focused_packet_artifact_sha256"],
        invocation_sha256=artifact["invocation_sha256"], original_invocation_retained=True,
        original_invocation_bytes=len(_wire(invocation, maximum=MAX_INVOCATION_BYTES)),
        invocation_id=invocation["invocation_id"], provider=invocation["provider"], model=invocation["model"],
        selected_representation=selected, recorded_selection_sha256=_sha(_wire(selection)),
        alternatives_receipt_sha256=_sha(_wire(pair_receipt)), matched_residual_record_verified=True,
        matched_required_references_verified=True, matched_task_invariants_verified=True,
        both_complete_model_inputs_reconstructed=True,
        candidates={name: dict(model_prompt_sha256=_sha(prompts[name].encode()),
            model_prompt_bytes=len(prompts[name].encode())) for name in ("full", "focused")},
        model_prompt_sha256=invocation["model_prompt_sha256"], model_prompt_bytes=invocation["model_prompt_bytes"],
        native_output_schema_sha256=schema_sha,
        recorded_dispatch_rechecked=dispatch,
        recorded_provider_attempted=invocation["world_task_packet"]["provider_attempted"],
        recorded_status=invocation.get("status"),
        recorded_counts_arithmetic_checked=True,
        trusted_counter_supplied=token_counter is not None,
        local_counter_verification_status="verified" if counter_verified else "unknown",
        local_counter_verified=counter_verified,
        counter_verification_scope="caller_supplied_local_counter_on_complete_retained_inputs",
        derived_ordinary_artifact_sha256=_sha(derived_artifact),
        derived_ordinary_invocation_projection_sha256=ordinary.receipt()["invocation_projection_sha256"],
        derived_projection_independently_retained=False,
        derived_projection_authority="internal_checked_projection_only",
        freshness="historical_unchecked", total_session_token_savings=None,
        **dict.fromkeys(FALSE_FIELDS, False))
    return ReplayedRepositoryWorldResidualSelection(ordinary.model_prompt, expected_sha256,
        _wire(invocation, maximum=MAX_INVOCATION_BYTES).decode(), _wire(observed).decode())


def replay_repository_world_residual_selection(data, *, expected_sha256, token_counter=None):
    """Reconstruct matched historical input; optionally repeat a trusted count."""
    try:
        return _replay(data, expected_sha256=expected_sha256, token_counter=token_counter)
    except RepositoryWorldResidualSelectionReplayError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError,
            OverflowError, OSError) as error:
        raise RepositoryWorldResidualSelectionReplayError("world_residual_selection_replay_contract_mismatch") from error


def freeze_repository_world_residual_selection(*, full_packet_artifact, expected_full_packet_sha256,
        focused_packet_artifact, expected_focused_packet_sha256, invocation_receipt,
        expected_invocation_sha256):
    """Join independently supplied pins, preserving the complete original receipt.

    Capture both live packet artifacts before dispatch can edit source. This
    later historical join needs no owner or source access. Retain the resulting
    outer digest independently; its component pins remain historical byte pins.
    """
    try:
        full, _ = _packet(full_packet_artifact, expected_full_packet_sha256)
        focused, _ = _packet(focused_packet_artifact, expected_focused_packet_sha256)
        invocation = _invocation_input(invocation_receipt, expected_invocation_sha256)
        data = _wire(dict(schema=SCHEMA, mode="historical_only", full_packet_artifact=full,
            full_packet_artifact_sha256=expected_full_packet_sha256, focused_packet_artifact=focused,
            focused_packet_artifact_sha256=expected_focused_packet_sha256,
            invocation=invocation, invocation_sha256=expected_invocation_sha256))
        replay_repository_world_residual_selection(data, expected_sha256=_sha(data))
        return data
    except RepositoryWorldResidualSelectionReplayError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError,
            OverflowError, OSError) as error:
        raise RepositoryWorldResidualSelectionReplayError("world_residual_selection_replay_contract_mismatch") from error
