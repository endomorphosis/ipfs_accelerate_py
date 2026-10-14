"""Reconstruct historical world-packet model input without dispatch authority.

Artifacts retain task/context content and a closed invocation projection. An
independently retained digest binds these bytes; it is not a provider attestation.
Replay uses installed renderers but never opens source or intent owners, restores
live packet seals, or independently verifies the retained native signatures.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

from .repository_world_packet_replay import (
    _canonical_bytes, _decode_canonical, replay_repository_world_packet,
)
from .supervisor_meta_index import public_replay_without_metadata


SCHEMA = "repository-world-invocation-artifact@1"
RECEIPT_SCHEMA = "repository-world-invocation-replay@1"
MAX_BYTES = 3 * 1024 * 1024
MAX_INVOCATION_BYTES = 131072
_REQUIRED = frozenset({
    "schema", "invocation_id", "provider", "model", "purpose", "workspace",
    "prompt_sha256", "prompt_bytes", "native_prompt_sha256", "native_prompt_bytes",
    "router_prompt_sha256", "router_prompt_bytes", "model_prompt_sha256", "model_prompt_bytes",
    "workspace_advisory_sha256", "workspace_advisory_bytes", "world_task_packet", "completion_authority",
})
_OPTIONAL = frozenset({"tool_output_policy", "repository_source_policy", "coding_reply_contract"})
_INCOMPATIBLE = frozenset({
    "semantic_translation", "semantic_prompt_components", "semantic_metadata_view",
    "source_input_view", "query_reference_view", "source_view_advisory_policy",
    "source_formalization_context", "doctor_residual_context", "public_instruction",
    "source_lookup", "source_view", "semantic_metadata_native_coding_reply_contract",
    "source_lookup_retrieval", "numerical_obligation_retrieval", "source_view_retrieval",
    "source_input_policy", "source_input_dispatched", "query_reference_policy",
    "query_reference_selected", "query_reference_dispatched", "query_reference_provider_attempted",
    "representation_displacement", "semantic_metadata_candidate", "prompt_selection",
    "terminal_semantic_scope_dispatch_rechecked",
})
_FALSE = dict(source_currentness_verified=False, owner_currentness_verified=False,
    owner_signature_replayed=False, compiler_token_counts_replayed=False,
    provider_delivery_verified=False, provider_response_replayed=False,
    token_counts_replayed=False, proof_authority=False, execution_authority=False,
    completion_authority=False, live_packet_issued=False, canonical_state_mutated=False)


class RepositoryWorldInvocationReplayError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(value, code):
    if not value:
        raise RepositoryWorldInvocationReplayError(code)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(member) for key, member in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(member) for member in value]
    return value


def _projection(receipt):
    _require(isinstance(receipt, Mapping), "world_replay_invocation_required")
    _require(_REQUIRED <= set(receipt), "world_replay_invocation_fields")
    _require(not any(receipt.get(key) is not None for key in _INCOMPATIBLE),
             "world_replay_incompatible_input")
    selected = _plain({key: receipt[key] for key in _REQUIRED | _OPTIONAL if key in receipt})
    _require(len(_canonical_bytes(selected)) <= MAX_INVOCATION_BYTES,
             "world_replay_invocation_byte_limit")
    return selected


def _identity(value, text, prefix):
    raw = text.encode("utf-8")
    _require(type(value.get(prefix + "_bytes")) is int
             and value[prefix + "_bytes"] == len(raw)
             and value.get(prefix + "_sha256") == _sha(raw),
             "world_replay_" + prefix + "_mismatch")


def _measurement(value, *, invocation, model_prompt, schema_sha):
    """Check a recorded measurement's closed shape and arithmetic, not counts."""
    fields = {"schema", "measurement_kind", "model_prompt_sha256", "model_prompt_bytes",
        "measurement_status", "measurement_error_type", "tokenizer_binding", "count_scope",
        "native_output_schema_tokens", "native_output_schema_sha256", "selected_content_tokens",
        "selected_input_tokens", "max_input_tokens", "max_prompt_bytes", "provider_framing_measured",
        "provider_tokenizer_equivalence_verified", "total_session_token_savings",
        "completion_authority", "provider_system_prompt_measured", "provider_tool_schemas_measured",
        "provider_internal_history_measured"}
    _require(type(value) is dict and set(value) == fields
             and value["schema"] == "supervisor-world-task-prompt-measurement@1"
             and value["measurement_kind"] == "declared_local_tokenizer"
             and value["count_scope"] == "rendered_user_prompt_and_separate_native_output_schema"
             and type(value["max_prompt_bytes"]) is int and value["max_prompt_bytes"] == 256000
             and value["native_output_schema_sha256"] == schema_sha,
             "world_replay_measurement_binding")
    _identity(value, model_prompt, "model_prompt")
    _require(all(value[key] is False for key in (
        "provider_framing_measured", "provider_tokenizer_equivalence_verified", "completion_authority",
        "provider_system_prompt_measured", "provider_tool_schemas_measured",
        "provider_internal_history_measured")) and value["total_session_token_savings"] is None,
        "world_replay_measurement_authority")
    status = value["measurement_status"]
    _require(type(status) is str and status in {"measured", "failed", "unavailable"},
             "world_replay_measurement_status")
    binding = value["tokenizer_binding"]
    if binding is not None:
        _require(type(binding) is dict and set(binding) == {
            "provider", "model", "tokenizer_id", "tokenizer_revision", "special_token_policy"}
            and all(type(item) is str and item and item.strip() == item for item in binding.values())
            and binding["provider"] == invocation["provider"] and binding["model"] == invocation["model"],
            "world_replay_measurement_tokenizer")
    limit = value["max_input_tokens"]
    _require(limit is None or type(limit) is int and limit > 0, "world_replay_measurement_limit")
    content, schema_tokens, total = (value[key] for key in (
        "selected_content_tokens", "native_output_schema_tokens", "selected_input_tokens"))
    if status == "measured":
        _require(binding is not None and value["measurement_error_type"] is None
                 and type(content) is int and content > 0
                 and type(schema_tokens) is int and schema_tokens >= 0
                 and (schema_tokens > 0 if schema_sha is not None else schema_tokens == 0)
                 and type(total) is int and total == content + schema_tokens
                 and (limit is None or total <= limit), "world_replay_measurement_counts")
    else:
        _require(content is None and schema_tokens is None and total is None and limit is None,
                 "world_replay_measurement_unavailable_counts")
        if status == "failed":
            from .router_implementation_runner import RUNNER_FAILURE_TYPES
            _require(binding is not None and type(value["measurement_error_type"]) is str
                     and value["measurement_error_type"] in RUNNER_FAILURE_TYPES | {"other"},
                     "world_replay_measurement_failure")
        else:
            _require(binding is None and value["measurement_error_type"] is None,
                     "world_replay_measurement_unavailable")


@dataclass(frozen=True)
class ReplayedRepositoryWorldInvocation:
    """Historical rendered bytes, deliberately not a Prepared packet."""

    model_prompt: str = field(repr=False)
    artifact_sha256: str
    _receipt_json: str = field(repr=False)

    def receipt(self):
        return json.loads(self._receipt_json)


def _replay_repository_world_invocation(data: bytes, *, expected_sha256: str):
    """Rebuild exact recorded user input and separate reply-schema identity.

    Token counts, provider-internal input and actual delivery are not replayed.
    Historical readiness/dispatch flags remain claims in the retained receipt.
    """
    artifact = _decode_canonical(data, expected_sha256, max_bytes=MAX_BYTES)
    _require(type(artifact) is dict and set(artifact) == {
        "schema", "packet_artifact", "packet_artifact_sha256", "invocation",
    } and artifact["schema"] == SCHEMA, "world_replay_invocation_artifact_schema")
    packet_bytes = _canonical_bytes(artifact["packet_artifact"])
    packet = replay_repository_world_packet(packet_bytes,
        expected_sha256=artifact["packet_artifact_sha256"])
    invocation = artifact["invocation"]
    _require(type(invocation) is dict and _REQUIRED <= set(invocation)
             and set(invocation) <= _REQUIRED | _OPTIONAL,
             "world_replay_invocation_fields")
    _require(_canonical_bytes(_projection(invocation)) == _canonical_bytes(invocation),
             "world_replay_invocation_projection")
    _require(invocation["schema"] == "router-implementation-invocation@1"
             and invocation["purpose"] == "coding" and invocation["provider"] == "codex_cli"
             and invocation["completion_authority"] is False,
             "world_replay_invocation_route")
    _require(type(invocation["invocation_id"]) is str
             and re.fullmatch(r"[0-9a-f]{32}", invocation["invocation_id"]) is not None,
             "world_replay_invocation_identity")
    _require(type(invocation["model"]) is str and invocation["model"].strip()
             and len(invocation["model"].encode("utf-8")) <= 512,
             "world_replay_model_identity")
    location = invocation["workspace"]
    _require(type(location) is str and len(location.encode("utf-8")) <= 4096
             and "\x00" not in location, "world_replay_workspace_identity")
    workspace = Path(location)
    _require(workspace.is_absolute() and str(workspace) == location
             and ".." not in workspace.parts, "world_replay_workspace_identity")
    recorded = invocation["world_task_packet"]
    _require(type(recorded) is dict, "world_replay_packet_receipt")
    dispatch_fields = {"dispatch_rechecked", "provider_attempted", "prompt_measurement"}
    packet_receipt = packet.recorded_packet_receipt()
    _require(set(recorded) == set(packet_receipt) | dispatch_fields
             and _canonical_bytes({key: recorded[key] for key in packet_receipt})
                 == _canonical_bytes(packet_receipt), "world_replay_packet_receipt")
    _require(type(recorded["dispatch_rechecked"]) is bool
             and type(recorded["provider_attempted"]) is bool
             and (not recorded["provider_attempted"] or recorded["dispatch_rechecked"]),
             "world_replay_dispatch_claims")
    _identity(invocation, packet.task_text, "prompt")
    _identity(invocation, packet.task_text, "native_prompt")
    _identity(invocation, packet.prompt, "router_prompt")
    from .router_implementation_runner import render_model_prompt
    from .coding_reply_contract import apply_coding_reply_contract
    with public_replay_without_metadata():
        model_prompt, advisory = render_model_prompt(prompt=packet.prompt,
            purpose="coding", workspace=workspace, world_packet=True,
            tool_output_binding=invocation.get("tool_output_policy"),
            repository_source_binding=invocation.get("repository_source_policy"))
        recorded_reply = invocation.get("coding_reply_contract")
        _require(recorded_reply is None or type(recorded_reply) is dict,
                 "world_replay_reply_contract")
        mode = "legacy" if recorded_reply is None else recorded_reply.get("mode")
        model_prompt, reply = apply_coding_reply_contract(model_prompt=model_prompt,
            mode=mode, purpose="coding", provider="codex_cli")
        if reply is not None:
            _require(type(recorded_reply.get("response_validated")) is bool,
                     "world_replay_reply_contract")
            _require(not recorded_reply["response_validated"] or recorded["provider_attempted"],
                     "world_replay_reply_dispatch_claims")
            # Reconstruct input/schema only, never reattest a provider response.
            reply["response_validated"] = recorded_reply["response_validated"]
        _require(_canonical_bytes(reply) == _canonical_bytes(recorded_reply),
                 "world_replay_reply_contract")
    _require(len(model_prompt.encode("utf-8")) <= 256000, "world_replay_model_byte_limit")
    _identity(invocation, advisory, "workspace_advisory")
    _identity(invocation, model_prompt, "model_prompt")
    schema_sha = None if reply is None else reply["native_output_schema_sha256"]
    measurement = recorded["prompt_measurement"]
    if measurement is not None:
        _measurement(measurement, invocation=invocation, model_prompt=model_prompt, schema_sha=schema_sha)
    diagnostic = dict(schema=RECEIPT_SCHEMA, artifact_sha256=expected_sha256,
        packet_artifact_sha256=artifact["packet_artifact_sha256"],
        invocation_projection_sha256=_sha(_canonical_bytes(invocation)),
        invocation_id=invocation["invocation_id"],
        native_prompt_sha256=invocation["native_prompt_sha256"],
        native_prompt_bytes=invocation["native_prompt_bytes"],
        router_prompt_sha256=invocation["router_prompt_sha256"],
        router_prompt_bytes=invocation["router_prompt_bytes"],
        model_prompt_sha256=invocation["model_prompt_sha256"],
        model_prompt_bytes=invocation["model_prompt_bytes"],
        native_output_schema_sha256=schema_sha,
        native_output_schema_bytes=None if reply is None else reply["native_output_schema_bytes"],
        recorded_token_measurement_present=measurement is not None,
        recorded_token_measurement_sha256=None if measurement is None else _sha(_canonical_bytes(measurement)),
        historical_input_reconstructed=True, packet_builder_replayed=True,
        guidance_replayed=True, freshness="historical_unchecked", **_FALSE)
    return ReplayedRepositoryWorldInvocation(model_prompt, expected_sha256,
                                             _canonical_bytes(diagnostic).decode("utf-8"))


def replay_repository_world_invocation(data: bytes, *, expected_sha256: str):
    """Verify bounded historical input, normalizing malformed contracts to ValueError."""
    try:
        return _replay_repository_world_invocation(data, expected_sha256=expected_sha256)
    except RepositoryWorldInvocationReplayError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError,
            OverflowError, OSError) as exc:
        raise RepositoryWorldInvocationReplayError("world_replay_invocation_contract_mismatch") from exc


def _freeze_repository_world_invocation(*, packet_artifact: bytes,
        expected_packet_sha256: str, invocation_receipt: Mapping) -> bytes:
    """Join a previously frozen packet to its reconstructable invocation input.

    Freeze the live packet before dispatch. This join can run after an edit or
    owner shutdown because it asserts historical byte consistency only. Caller
    storage must retain the returned digest independently of the artifact.
    """
    packet = _decode_canonical(packet_artifact, expected_packet_sha256)
    encoded = _canonical_bytes(dict(schema=SCHEMA, packet_artifact=packet,
        packet_artifact_sha256=expected_packet_sha256, invocation=_projection(invocation_receipt)))
    _require(len(encoded) <= MAX_BYTES, "world_replay_invocation_artifact_byte_limit")
    replay_repository_world_invocation(encoded, expected_sha256=_sha(encoded))
    return encoded


def freeze_repository_world_invocation(*, packet_artifact: bytes,
        expected_packet_sha256: str, invocation_receipt: Mapping) -> bytes:
    """Export a historical invocation projection without replaying live owners."""
    try:
        return _freeze_repository_world_invocation(packet_artifact=packet_artifact,
            expected_packet_sha256=expected_packet_sha256, invocation_receipt=invocation_receipt)
    except RepositoryWorldInvocationReplayError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError,
            OverflowError, OSError) as exc:
        raise RepositoryWorldInvocationReplayError("world_replay_invocation_contract_mismatch") from exc
