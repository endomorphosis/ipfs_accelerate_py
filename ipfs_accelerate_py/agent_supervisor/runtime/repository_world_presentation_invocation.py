"""Pinned historical replay of a presented world-packet router invocation.

The original packet, presentation custody and full invocation receipt are
private archival data. Replay reconstructs the rendered user input, not a live
packet, current source owner, provider response, token count or billing proof.
Recorded failure and usage fields remain observations without normalization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

SCHEMA = "repository-world-presentation-invocation-artifact@1"
RECEIPT_SCHEMA = "repository-world-presentation-invocation-replay@1"
MAX_BYTES = 4 * 1024 * 1024
MAX_INVOCATION_BYTES = 1024 * 1024
MAX_NODES = 100000
MAX_DEPTH = 64
MAX_MODEL_PROMPT_BYTES = 256000
_HEX = re.compile(r"[0-9a-f]{64}")
_INVOCATION_ID = re.compile(r"[0-9a-f]{32}")
_REQUIRED = frozenset({"schema", "invocation_id", "provider", "model", "purpose", "workspace",
    "prompt_sha256", "prompt_bytes", "native_prompt_sha256", "native_prompt_bytes",
    "router_prompt_sha256", "router_prompt_bytes", "model_prompt_sha256", "model_prompt_bytes",
    "workspace_advisory_sha256", "workspace_advisory_bytes", "world_task_packet", "world_task_presentation",
    "completion_authority"})
_ARTIFACT_FIELDS = frozenset({"schema", "mode", "packet_artifact", "packet_artifact_sha256",
    "presentation", "invocation", "invocation_sha256"})
_PRESENTATION_FIELDS = frozenset({"view_prompt", "sidecar", "sidecar_sha256", "codec_receipt_sha256"})
_PRESENTATION_FALSE = frozenset({"proof_authority", "execution_authority", "completion_authority",
    "omission_authority", "provider_ingestion_verified", "total_token_savings_measured", "source_freshness_verified"})
FALSE_FIELDS = frozenset({"source_currentness_verified", "owner_currentness_verified", "owner_signature_replayed",
    "compiler_token_counts_replayed", "provider_delivery_verified", "provider_response_replayed",
    "token_counts_replayed", "billing_verified", "proof_authority", "execution_authority", "completion_authority",
    "live_packet_issued", "canonical_state_mutated", "model_consumption_verified", "total_token_savings_measured"})


class RepositoryWorldPresentationInvocationError(ValueError):
    """Closed refusal codes; never include task, source or exception contents."""


def _need(condition, code):
    if not condition:
        raise RepositoryWorldPresentationInvocationError("world_presentation_invocation_" + code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pin(value):
    return type(value) is str and _HEX.fullmatch(value) is not None


def _wire(value, *, maximum=MAX_BYTES):
    from .repository_world_packet_replay import _canonical_bytes, _preflight_json
    # Bound the complete caller population before serializer traversal. Values
    # stay JSON data; no mapping/sequence callbacks or coercion are accepted.
    pending, count, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        count += 1
        _need(count <= MAX_NODES and depth <= MAX_DEPTH, "structure_bound")
        if type(item) is str:
            _need(len(item) <= maximum, "text_bound")
            text_bytes += len(item.encode("utf-8", "strict"))
            _need(text_bytes <= maximum, "text_bound")
        elif type(item) is dict:
            _need(len(item) <= MAX_NODES and all(type(key) is str for key in item), "json_type")
            _need(count + len(pending) + 2 * len(item) <= MAX_NODES, "structure_bound")
            pending.extend((child, depth + 1) for pair in item.items() for child in pair)
        elif type(item) is list:
            _need(len(item) <= MAX_NODES, "structure_bound")
            _need(count + len(pending) + len(item) <= MAX_NODES, "structure_bound")
            pending.extend((child, depth + 1) for child in item)
        else:
            _need(item is None or type(item) in (bool, float)
                  or type(item) is int and item.bit_length() <= 4096, "json_type")
    raw = _canonical_bytes(value)
    _need(len(raw) <= maximum, "byte_bound")
    _preflight_json(raw.decode("utf-8", "strict"))
    return raw


def _decode(raw, pin, *, maximum):
    from .repository_world_packet_replay import _decode_canonical
    _need(type(raw) is bytes and 0 < len(raw) <= maximum and _pin(pin) and _sha(raw) == pin, "input_pin")
    return _decode_canonical(raw, pin, max_bytes=maximum)


def invocation_receipt_sha256(receipt):
    """FULL finite canonical-JSON digest, distinct from a pretty file's hash."""
    return _sha(_wire(receipt, maximum=MAX_INVOCATION_BYTES))


def _identity(value, text, prefix):
    raw = text.encode("utf-8", "strict")
    _need(type(value.get(prefix + "_bytes")) is int and value[prefix + "_bytes"] == len(raw)
          and value.get(prefix + "_sha256") == _sha(raw), prefix + "_pin")


def _strict_packet(raw, expected_sha256):
    from .repository_world_packet_replay import MAX_BYTES as PACKET_MAX_BYTES, replay_repository_world_packet
    artifact = _decode(raw, expected_sha256, maximum=PACKET_MAX_BYTES)
    packet = replay_repository_world_packet(raw, expected_sha256=expected_sha256)
    recorded = packet.recorded_packet_receipt()
    _need("validation_profile" not in artifact["recipe"] and "validation_profile" not in recorded
          and type(recorded.get("validation_count")) is int and recorded["validation_count"] > 0,
          "strict_provider_packet_required")
    return artifact, packet


def _presentation_material(packet, *, presentation, view_prompt, sidecar_bytes, expected_sidecar_sha256):
    from .repository_world_packet_presentation import (PreparedRepositoryWorldPacketPresentation, _build,
        MAX_BYTES as VIEW_MAX_BYTES, MAX_SIDECAR_BYTES, replay_repository_world_packet_presentation)
    if presentation is not None:
        _need(view_prompt is None and sidecar_bytes is None
              and type(presentation) is PreparedRepositoryWorldPacketPresentation, "exclusive_presentation_inputs")
        PreparedRepositoryWorldPacketPresentation._retained(presentation)
        issued = PreparedRepositoryWorldPacketPresentation.validate_original(presentation, packet.prompt)
        view_prompt, sidecar_bytes = presentation.prompt, presentation.sidecar_bytes
        expected_sidecar_sha256 = issued["sidecar_sha256"] if expected_sidecar_sha256 is None else expected_sidecar_sha256
    else:
        _need(type(view_prompt) is str and type(sidecar_bytes) is bytes and _pin(expected_sidecar_sha256),
              "pinned_presentation_inputs_required")
    _need(type(view_prompt) is str and 0 < len(view_prompt) <= VIEW_MAX_BYTES
          and len(view_prompt.encode("utf-8", "strict")) <= VIEW_MAX_BYTES, "view_bound")
    sidecar = _decode(sidecar_bytes, expected_sidecar_sha256, maximum=MAX_SIDECAR_BYTES)
    original = replay_repository_world_packet_presentation(view_prompt=view_prompt, sidecar_bytes=sidecar_bytes,
        expected_sidecar_sha256=expected_sidecar_sha256, expected_original_sha256=_sha(packet.prompt.encode()))
    _need(original == packet.prompt, "inverse_original")
    expected_view, expected_sidecar, receipt_raw = _build(packet.prompt)
    _need(expected_view == view_prompt and expected_sidecar == sidecar_bytes, "fixed_presentation")
    codec = _decode(receipt_raw, _sha(receipt_raw), maximum=MAX_INVOCATION_BYTES)
    from .repository_world_packet_presentation import MODE
    _need(codec["codec_supported"] is True and codec["model_view_mode"] == MODE, "supported_presentation_required")
    return dict(view_prompt=view_prompt, sidecar=sidecar, sidecar_sha256=expected_sidecar_sha256,
        codec_receipt_sha256=_sha(receipt_raw)), codec


def _render(packet, presentation, codec, invocation):
    from .repository_world_invocation_replay import _INCOMPATIBLE, _measurement
    from .repository_world_packet_presentation import MODE, RECEIPT_SCHEMA as CODEC_RECEIPT_SCHEMA
    from .supervisor_meta_index import public_replay_without_metadata
    _need(type(invocation) is dict and _REQUIRED <= set(invocation), "invocation_fields")
    _need(not any(invocation.get(key) is not None for key in _INCOMPATIBLE), "incompatible_input")
    _need(invocation["schema"] == "router-implementation-invocation@1" and invocation["purpose"] == "coding"
          and invocation["provider"] == "codex_cli" and invocation["completion_authority"] is False,
          "invocation_route")
    _need(type(invocation["invocation_id"]) is str and _INVOCATION_ID.fullmatch(invocation["invocation_id"]), "invocation_identity")
    _need(type(invocation["model"]) is str and invocation["model"].strip() == invocation["model"]
          and 0 < len(invocation["model"].encode()) <= 512, "model_identity")
    _need(type(invocation.get("status")) is str
          and invocation["status"] in {"provider_returned", "failed"}, "invocation_status")
    location = invocation["workspace"]
    _need(type(location) is str and 0 < len(location.encode()) <= 4096 and "\0" not in location, "workspace_identity")
    workspace = Path(location)
    _need(workspace.is_absolute() and str(workspace) == location and ".." not in workspace.parts, "workspace_identity")
    original = packet.recorded_packet_receipt()
    recorded = invocation["world_task_packet"]
    extra = {"dispatch_rechecked", "provider_attempted", "provider_retained_parent", "prompt_measurement"}
    _need(type(recorded) is dict and set(recorded) == set(original) | extra
          and _wire({key: recorded[key] for key in original}) == _wire(original), "original_packet_receipt")
    _need(type(recorded["dispatch_rechecked"]) is bool and type(recorded["provider_attempted"]) is bool
          and recorded["provider_retained_parent"] is False
          and (not recorded["provider_attempted"] or recorded["dispatch_rechecked"]), "dispatch_observation")
    expected = dict(schema="router-world-task-presentation@1", profile=MODE, codec_receipt_schema=CODEC_RECEIPT_SCHEMA,
        model_view_mode=MODE, codec_supported=True,
        **{key: codec[key] for key in ("original_prompt_sha256", "original_prompt_bytes", "presented_prompt_sha256",
            "presented_prompt_bytes", "sidecar_sha256", "sidecar_bytes")},
        codec_receipt_sha256=presentation["codec_receipt_sha256"], codec_receipt_bytes=len(_wire(codec)),
        inverse_verified=True, dispatch_rechecked=recorded["dispatch_rechecked"], **dict.fromkeys(_PRESENTATION_FALSE, False))
    _need(type(invocation["world_task_presentation"]) is dict
          and _wire(invocation["world_task_presentation"]) == _wire(expected), "presentation_receipt")
    _identity(invocation, packet.task_text, "prompt")
    _identity(invocation, packet.task_text, "native_prompt")
    _identity(invocation, presentation["view_prompt"], "router_prompt")
    artifacts = invocation.get("world_packet_artifacts")
    if artifacts is not None:
        from .repository_world_artifact_retention import validate_world_packet_artifacts
        retained = validate_world_packet_artifacts(artifacts, invocation_id=invocation["invocation_id"])
        _need(retained["status"] == "unavailable" and retained["packet"] is None and retained["invocation"] is None,
              "mixed_old_retention")
    from .router_implementation_runner import render_model_prompt
    from .coding_reply_contract import apply_coding_reply_contract
    with public_replay_without_metadata():
        model_prompt, advisory = render_model_prompt(prompt=presentation["view_prompt"], purpose="coding", workspace=workspace,
            world_packet=True, tool_output_binding=invocation.get("tool_output_policy"),
            repository_source_binding=invocation.get("repository_source_policy"))
        reply = invocation.get("coding_reply_contract")
        _need(reply is None or type(reply) is dict, "coding_reply")
        model_prompt, expected_reply = apply_coding_reply_contract(model_prompt=model_prompt,
            mode="legacy" if reply is None else reply.get("mode"), purpose="coding", provider="codex_cli")
        if expected_reply is not None:
            _need(type(reply.get("response_validated")) is bool
                  and (not reply["response_validated"] or recorded["provider_attempted"]), "coding_reply_observation")
            expected_reply["response_validated"] = reply["response_validated"]
        _need(_wire(reply) == _wire(expected_reply), "coding_reply")
    _need(len(model_prompt.encode()) <= MAX_MODEL_PROMPT_BYTES, "model_prompt_bound")
    _identity(invocation, advisory, "workspace_advisory")
    _identity(invocation, model_prompt, "model_prompt")
    schema_sha = None if expected_reply is None else expected_reply["native_output_schema_sha256"]
    if recorded["prompt_measurement"] is not None:
        _measurement(recorded["prompt_measurement"], invocation=invocation, model_prompt=model_prompt, schema_sha=schema_sha)
    return model_prompt, schema_sha


@dataclass(frozen=True, eq=False)
class ReplayedRepositoryWorldPresentationInvocation:
    """Historical bytes only; no live owner/currentness/dispatch methods."""
    model_prompt: str = field(repr=False)
    artifact_sha256: str
    _invocation_json: str = field(repr=False)
    _receipt_json: str = field(repr=False)

    def recorded_invocation_receipt(self):
        return json.loads(self._invocation_json)

    def receipt(self):
        return json.loads(self._receipt_json)


def _replay(data, *, expected_sha256, expected_packet_sha256, expected_invocation_sha256, expected_sidecar_sha256):
    artifact = _decode(data, expected_sha256, maximum=MAX_BYTES)
    _need(type(artifact) is dict and set(artifact) == _ARTIFACT_FIELDS and artifact["schema"] == SCHEMA
          and artifact["mode"] == "historical_only", "artifact_schema")
    _need(_pin(expected_packet_sha256) and _pin(expected_invocation_sha256) and _pin(expected_sidecar_sha256)
          and artifact["packet_artifact_sha256"] == expected_packet_sha256
          and artifact["invocation_sha256"] == expected_invocation_sha256, "external_component_pins")
    packet_raw = _wire(artifact["packet_artifact"])
    _original, packet = _strict_packet(packet_raw, expected_packet_sha256)
    body = artifact["presentation"]
    _need(type(body) is dict and set(body) == _PRESENTATION_FIELDS and body["sidecar_sha256"] == expected_sidecar_sha256,
          "presentation_fields")
    from .repository_world_packet_presentation import MAX_SIDECAR_BYTES
    presentation, codec = _presentation_material(packet, presentation=None, view_prompt=body["view_prompt"],
        sidecar_bytes=_wire(body["sidecar"], maximum=MAX_SIDECAR_BYTES), expected_sidecar_sha256=expected_sidecar_sha256)
    _need(_wire(body) == _wire(presentation), "presentation_projection")
    invocation = artifact["invocation"]
    _need(invocation_receipt_sha256(invocation) == expected_invocation_sha256, "invocation_pin")
    model_prompt, schema_sha = _render(packet, presentation, codec, invocation)
    observed = dict(schema=RECEIPT_SCHEMA, replay_mode="historical_only", artifact_sha256=expected_sha256,
        packet_artifact_sha256=expected_packet_sha256, invocation_sha256=expected_invocation_sha256,
        sidecar_sha256=expected_sidecar_sha256, codec_receipt_sha256=presentation["codec_receipt_sha256"],
        invocation_id=invocation["invocation_id"], provider=invocation["provider"], model=invocation["model"],
        native_prompt_sha256=invocation["native_prompt_sha256"], native_prompt_bytes=invocation["native_prompt_bytes"],
        original_packet_prompt_sha256=codec["original_prompt_sha256"], original_packet_prompt_bytes=codec["original_prompt_bytes"],
        router_prompt_sha256=invocation["router_prompt_sha256"], router_prompt_bytes=invocation["router_prompt_bytes"],
        model_prompt_sha256=invocation["model_prompt_sha256"], model_prompt_bytes=invocation["model_prompt_bytes"],
        native_output_schema_sha256=schema_sha, recorded_provider_attempted=invocation["world_task_packet"]["provider_attempted"],
        recorded_dispatch_rechecked=invocation["world_task_packet"]["dispatch_rechecked"],
        recorded_status=invocation.get("status"), recorded_usage_present="usage" in invocation,
        recorded_usage_sha256=_sha(_wire(invocation["usage"])) if "usage" in invocation else None,
        recorded_native_usage_present="native_rollout_usage" in invocation,
        recorded_native_usage_sha256=_sha(_wire(invocation["native_rollout_usage"])) if "native_rollout_usage" in invocation else None,
        usage_scope="recorded_fields_without_recount_or_billing_validation", freshness="historical_unchecked",
        historical_model_input_reconstructed=True, exact_presentation_inverse_verified=True, **dict.fromkeys(FALSE_FIELDS, False))
    return ReplayedRepositoryWorldPresentationInvocation(model_prompt, expected_sha256,
        _wire(invocation, maximum=MAX_INVOCATION_BYTES).decode(), _wire(observed).decode())


def replay_repository_world_presentation_invocation(data, *, expected_sha256, expected_packet_sha256,
        expected_invocation_sha256, expected_sidecar_sha256):
    try:
        return _replay(data, expected_sha256=expected_sha256, expected_packet_sha256=expected_packet_sha256,
            expected_invocation_sha256=expected_invocation_sha256, expected_sidecar_sha256=expected_sidecar_sha256)
    except RepositoryWorldPresentationInvocationError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError, OverflowError, OSError) as error:
        raise RepositoryWorldPresentationInvocationError("world_presentation_invocation_contract_mismatch") from error


def freeze_repository_world_presentation_invocation(*, packet_artifact, expected_packet_sha256,
        invocation_receipt, expected_invocation_sha256, presentation=None, view_prompt=None, sidecar_bytes=None,
        expected_sidecar_sha256=None):
    """Join independent original/presentation/invocation pins without live IO.

    Either an exact issued presentation or a view plus independently pinned
    sidecar is required. Store the returned bundle and its digest independently;
    a complete receipt hash is not a provider or native-owner attestation.
    """
    try:
        packet_record, packet = _strict_packet(packet_artifact, expected_packet_sha256)
        _need(_pin(expected_invocation_sha256)
              and invocation_receipt_sha256(invocation_receipt) == expected_invocation_sha256, "invocation_pin")
        material, codec = _presentation_material(packet, presentation=presentation, view_prompt=view_prompt,
            sidecar_bytes=sidecar_bytes, expected_sidecar_sha256=expected_sidecar_sha256)
        _render(packet, material, codec, invocation_receipt)
        data = _wire(dict(schema=SCHEMA, mode="historical_only", packet_artifact=packet_record,
            packet_artifact_sha256=expected_packet_sha256, presentation=material,
            invocation=invocation_receipt, invocation_sha256=expected_invocation_sha256))
        replay_repository_world_presentation_invocation(data, expected_sha256=_sha(data),
            expected_packet_sha256=expected_packet_sha256, expected_invocation_sha256=expected_invocation_sha256,
            expected_sidecar_sha256=material["sidecar_sha256"])
        return data
    except RepositoryWorldPresentationInvocationError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError, OverflowError, OSError) as error:
        raise RepositoryWorldPresentationInvocationError("world_presentation_invocation_contract_mismatch") from error
