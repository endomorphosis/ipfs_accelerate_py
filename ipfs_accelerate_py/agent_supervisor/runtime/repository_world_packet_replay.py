"""Bounded historical reconstruction of an exported repository task packet.

Artifacts are portable data, not live owner capabilities. Replay reconstructs
the exact native packet and delivered bytes without opening source, intent or
Quack owners. Historical freshness assertions remain recorded assertions;
dispatch requires a newly admitted live packet from the current native owners.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re

from ..context.context_compiler import ContextCompilationReceipt, ContextCompileResult
from ..context.context_contracts import ContextCapsule
from ..procedure_compiler.world_model import RepositoryWorldState
from ..proof.formal_verification_contracts import content_identity
from .repository_world_packet import (
    ARTIFACT_SCHEMA, REPLAY_MAX_BYTES, PreparedRepositoryWorldPacket,
    _render_repository_world_packet, _residual_binding_arguments, _residual_recipe,
)
from .supervisor_meta_index import public_replay_without_metadata


MAX_BYTES = REPLAY_MAX_BYTES
MAX_DEPTH = 64
MAX_NODES = 100_000
RECEIPT_SCHEMA = "repository-world-packet-replay-receipt@1"
_ARTIFACT_FIELDS = frozenset({"schema", "mode", "capsule", "compilation_receipt",
    "recipe", "native_packet", "prompt", "packet_receipt"})
_RECIPE_FIELDS = frozenset({"generation", "task_id", "task_text",
    "unresolved_obligations", "validation_commands", "proof_obligations",
    "acceptance_ids", "allowed_effects", "write_paths", "current_world", "max_bytes"})
_SEQUENCES = ("unresolved_obligations", "validation_commands", "proof_obligations",
              "acceptance_ids", "allowed_effects", "write_paths")
_OPTIONAL_RECIPE_FIELDS = frozenset({"validation_profile", "residual_context"})


class RepositoryWorldPacketReplayError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _require(value, code):
    if not value:
        raise RepositoryWorldPacketReplayError("world_packet_replay_" + code)


def _canonical_bytes(value) -> bytes:
    """One strict JSON spelling, shared with the complete-invocation replay."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_key")
        result[key] = value
    return result


def _nonfinite(_value):
    raise RepositoryWorldPacketReplayError("world_packet_replay_nonfinite")


def _preflight_json(text):
    # Bound nesting and lexical nodes before the JSON parser allocates nested
    # containers. Quotes and escapes inside strings do not affect either bound.
    depth, nodes = 0, 0
    quoted = escaped = token = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted, token = True, False
            nodes += 1
        elif char in "[{":
            depth += 1
            nodes += 1
            token = False
            _require(depth <= MAX_DEPTH, "depth_limit")
        elif char in "]}":
            depth -= 1
            token = False
            _require(depth >= 0, "invalid_json")
        elif char in " \r\n\t,:":
            token = False
        elif not token:
            token = True
            nodes += 1
        _require(nodes <= MAX_NODES, "nodes_limit")
    _require(not quoted and depth == 0, "invalid_json")


def _decode_canonical(data: bytes, expected_sha256: str, *, max_bytes: int = MAX_BYTES):
    """Decode bounded, pinned canonical JSON without external IO or defaults."""
    _require(type(data) is bytes and type(max_bytes) is int
             and 0 < len(data) <= max_bytes, "bytes_limit")
    _require(type(expected_sha256) is str
             and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
             "expected_sha256_required")
    _require(hashlib.sha256(data).hexdigest() == expected_sha256, "sha256_mismatch")
    try:
        text = data.decode("utf-8", errors="strict")
        _preflight_json(text)
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_nonfinite)
        _require(type(value) is dict, "object_required")
        _require(_canonical_bytes(value) == data, "noncanonical_json")
        return value
    except RepositoryWorldPacketReplayError:
        raise
    except (UnicodeError, ValueError, TypeError, RecursionError, OverflowError) as exc:
        raise RepositoryWorldPacketReplayError("world_packet_replay_invalid_json") from exc


def _exact_record(record, raw, name):
    _require(type(raw) is dict and _canonical_bytes(record.to_record()) == _canonical_bytes(raw),
             name + "_record_mismatch")


def _recorded_intent_receipt(capsule, recipe, receipt):
    """Join recorded public evidence without claiming signature revalidation."""
    reserved = tuple(ref for ref in (*capsule.evidence, *capsule.expansion_references)
                     if ref.kind == "repository-intent-world")
    if receipt is None:
        _require(not reserved, "intent_receipt_missing")
        return
    _require(type(receipt) is dict and len(reserved) == 1 and reserved[0].required
             and reserved[0] in capsule.evidence, "intent_reference_mismatch")
    ref = reserved[0]
    summary = ref.summary.encode("utf-8")
    body = _decode_canonical(summary, hashlib.sha256(summary).hexdigest(), max_bytes=65536)
    base_fields = {"schema", "task_id", "source_binding", "intent_binding", "task_spec",
        "observed_task", "requirements", "requirement_observation_cid", "intent_world",
        "identity_join", "measurement_scope", "freshness_scope", "missing_authority",
        "proof_authority", "execution_authority", "completion_authority",
        "canonical_state_mutated", "semantic_alignment_verified", "source_semantics_verified"}
    native_fields = {"native_owner_binding", "native_owner_access"}
    _require(set(body) in (base_fields, base_fields | native_fields)
             and body["schema"] == "repository-intent-world-observation@1", "intent_schema")
    authority = {key: False for key in ("proof_authority", "execution_authority",
        "completion_authority", "canonical_state_mutated", "semantic_alignment_verified",
        "source_semantics_verified")}
    _require(all(body[key] is False for key in authority), "intent_authority")
    binding, source = body["intent_binding"], body["source_binding"]
    world, observed = body["intent_world"], body["observed_task"]
    _require(body["task_id"] == recipe["task_id"]
             and type(source["generation"]) is int and source["generation"] == recipe["generation"]
             and source["datasets_head"]["repository_id"] == capsule.repository_id
             and "sha256:" + source["source_tree_sha256"] == capsule.tree_id
             and source["session_id"] == capsule.authority["repository_context"]["session_id"]
             and source["semantic_state_cid"] == capsule.authority["repository_context"]["semantic_state_cid"],
             "intent_source_binding_mismatch")
    observation_cid = content_identity(body)
    _require(ref.reference_id == "repository-intent-world:" + observation_cid
             and ref.referenced_content_id == observation_cid
             and ref.repository_id == capsule.repository_id and ref.tree_id == capsule.tree_id,
             "intent_observation_identity_mismatch")
    expected = dict(schema="repository-intent-world-observation-receipt@1",
        observation_cid=observation_cid, reference_id=ref.reference_id,
        reference_content_id=ref.content_id, summary_sha256=hashlib.sha256(summary).hexdigest(),
        summary_bytes=len(summary), reference_bytes=len(ref.canonical_bytes()),
        task_id=body["task_id"], task_revision=observed["revision"],
        intent_owner_id=binding["intent_owner_id"],
        native_event_watermark=binding["native_event_watermark"],
        intent_repository_id=binding["repository_id"],
        intent_source_tree_id=binding["current_source_tree_id"],
        source_repository_id=source["datasets_head"]["repository_id"],
        source_tree_sha256=source["source_tree_sha256"], session_id=source["session_id"],
        generation=source["generation"], world_snapshot_cid=world["snapshot"]["snapshot_cid"],
        world_capture_cid=world["capture_cid"],
        requirement_observation_cid=body["requirement_observation_cid"],
        requirement_count=len(body["requirements"]), validation_count=len(observed["validations"]),
        world_schedulable=world["schedulable"], native_intent_freshness_checked=True,
        source_freshness_checked=True, resource_limits=dict(admission_memory_mb=256,
            cooperative_deadline_seconds=30, parser_hard_rss_limit=False), **authority)
    if native_fields <= set(body):
        expected.update(native_owner_binding=body["native_owner_binding"],
                        native_owner_access=body["native_owner_access"])
    _require(_canonical_bytes(receipt) == _canonical_bytes(expected), "intent_receipt_mismatch")


@dataclass(frozen=True, eq=False)
class ReplayedRepositoryWorldPacket:
    """Read-only historical bytes; deliberately no live validation capability."""

    prompt: str
    task_text: str
    artifact_sha256: str
    _recorded_receipt_json: str = field(repr=False)
    _receipt_json: str = field(repr=False)

    def recorded_packet_receipt(self) -> dict:
        """Copy historical assertions; these are not current owner attestations."""
        return json.loads(self._recorded_receipt_json)

    def receipt(self) -> dict:
        return json.loads(self._receipt_json)


def freeze_repository_world_packet(packet: PreparedRepositoryWorldPacket) -> bytes:
    """Export only a still-current issued packet; perform no filesystem IO."""
    _require(type(packet) is PreparedRepositoryWorldPacket, "live_packet_required")
    # This call checks the retained immutable replay bytes as part of issuance
    # before this function parses them. The task is recovered only afterwards.
    with public_replay_without_metadata():
        from .repository_world_packet import _ISSUANCE_LOCK, _ISSUED
        with _ISSUANCE_LOCK:
            issued = _ISSUED.get(packet)
        _require(issued is not None and issued == packet._seal(), "live_packet_not_issued")
        data = packet._replay_material_json.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        artifact = _decode_canonical(data, digest)
        packet.validate_current(workspace=packet._workspace,
                                task_text=artifact["recipe"]["task_text"])
    return data


def replay_repository_world_packet(data: bytes, *, expected_sha256: str) -> ReplayedRepositoryWorldPacket:
    """Reconstruct exact recorded packet bytes with no live owner or dispatch."""
    artifact = _decode_canonical(data, expected_sha256)
    try:
        _require(set(artifact) == _ARTIFACT_FIELDS and artifact["schema"] == ARTIFACT_SCHEMA
                 and artifact["mode"] == "historical_only", "artifact_schema")
        recipe = artifact["recipe"]
        _require(type(recipe) is dict and _RECIPE_FIELDS <= set(recipe)
                 and set(recipe) <= _RECIPE_FIELDS | _OPTIONAL_RECIPE_FIELDS, "recipe_schema")
        if "validation_profile" in recipe:
            _require(type(recipe["validation_profile"]) is str
                     and recipe["validation_profile"] == "read_only_observation@1",
                     "validation_profile")
        _require(type(artifact["packet_receipt"]) is dict, "packet_receipt_schema")
        with public_replay_without_metadata():
            capsule = ContextCapsule.from_dict(artifact["capsule"])
            _exact_record(capsule, artifact["capsule"], "capsule")
            compile_receipt = ContextCompilationReceipt.from_dict(artifact["compilation_receipt"])
            _exact_record(compile_receipt, artifact["compilation_receipt"], "compilation_receipt")
            compilation = ContextCompileResult(capsule, compile_receipt, compile_receipt.decisions)
            arguments = dict(recipe)
            for key in _SEQUENCES:
                _require(type(arguments[key]) is list, "recipe_sequence")
                arguments[key] = tuple(arguments[key])
            if arguments["current_world"] is not None:
                world = RepositoryWorldState.from_dict(arguments["current_world"])
                _exact_record(world, arguments["current_world"], "world")
                arguments["current_world"] = world
            intent_receipt = artifact["packet_receipt"].get("intent_world")
            _recorded_intent_receipt(capsule, recipe, intent_receipt)
            residual_context = None
            if "residual_context" in arguments:
                from .repository_residual_context import replay_repository_residual_context
                recorded = arguments.pop("residual_context")
                _require(type(recorded) is dict and set(recorded) == {
                    "record", "record_sha256", "packet_id", "context_id",
                    "packet_sha256", "context_sha256"}, "residual_context_schema")
                _require(type(recorded["record_sha256"]) is str
                         and hashlib.sha256(_canonical_bytes(recorded["record"])).hexdigest()
                         == recorded["record_sha256"], "residual_record_pin")
                residual_context = replay_repository_residual_context(recorded["record"],
                    **_residual_binding_arguments(capsule, task_id=arguments["task_id"],
                        write_paths=arguments["write_paths"],
                        validation_commands=arguments["validation_commands"],
                        acceptance_ids=arguments["acceptance_ids"]))
                _require(_canonical_bytes(recorded) == _canonical_bytes(_residual_recipe(residual_context)),
                         "residual_component_pins")
            prompt, receipt, native_packet = _render_repository_world_packet(
                compilation=compilation, intent_world_receipt=intent_receipt,
                **({"_residual_context": residual_context} if residual_context is not None else {}),
                **arguments)
        _require(type(artifact["prompt"]) is str and prompt == artifact["prompt"], "prompt_mismatch")
        _require(_canonical_bytes(native_packet) == _canonical_bytes(artifact["native_packet"]),
                 "native_packet_mismatch")
        _require(_canonical_bytes(receipt) == _canonical_bytes(artifact["packet_receipt"]),
                 "packet_receipt_mismatch")
        historical = dict(schema=RECEIPT_SCHEMA, replay_mode="historical_only",
            artifact_sha256=expected_sha256, artifact_bytes=len(data),
            prompt_sha256=receipt["prompt_sha256"], prompt_bytes=receipt["prompt_bytes"],
            task_text_sha256=receipt["task_text_sha256"], packet_id=receipt["packet_id"],
            context_cid=receipt["context_cid"], repository_id=receipt["repository_id"],
            tree_id=receipt["tree_id"], exact_prompt_reconstructed=True,
            native_packet_reconstructed=True, compiled_context_structure_checked=True,
            source_owner_reopened=False, intent_owner_reopened=False,
            live_source_freshness_checked=False, live_intent_freshness_checked=False,
            native_signatures_reverified=False, compiler_token_recounted=False,
            provider_dispatch_verified=False, provider_input_tokens=None,
            full_session_token_savings=None, proof_authority=False,
            execution_authority=False, completion_authority=False)
        if "residual_context" in receipt:
            historical["residual_context"] = receipt["residual_context"]
        return ReplayedRepositoryWorldPacket(prompt, recipe["task_text"], expected_sha256,
            _canonical_bytes(receipt).decode("utf-8"), _canonical_bytes(historical).decode("utf-8"))
    except RepositoryWorldPacketReplayError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError,
            OverflowError) as exc:
        raise RepositoryWorldPacketReplayError("world_packet_replay_contract_mismatch") from exc
