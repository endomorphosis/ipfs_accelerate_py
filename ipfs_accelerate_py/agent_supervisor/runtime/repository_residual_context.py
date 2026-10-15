"""Detach existing sealed residual evidence from a full admitted task's scope.

This binding adds no provider, world schema, renderer, source owner or authority.
The existing residual invocation preparer supplies the bounded, self-contained
nomination context.  Callers retain the full task's obligations and independently
revalidate its native source and policy before dispatch.  Replay checks recorded
bytes and structural bindings; it never restores live admission or parent memory.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math

from ..planning.residual_llm_packet import ResidualLlmPacket
from ..todo_daemon.residual_provider_invocation import PathLease, ResidualProviderInvocation


SCHEMA = "repository-residual-context@1"
RECEIPT_SCHEMA = "repository-residual-context-receipt@1"
MAX_PACKET_BYTES = 65_536
MAX_RECORD_BYTES = 131_072
MAX_NODES = 8_192
MAX_DEPTH = 20


class RepositoryResidualContextError(ValueError):
    """Content-free refusal; input or exception bodies are never interpolated."""


def _need(condition, code):
    if not condition:
        raise RepositoryResidualContextError(code)


def _wire(value, *, maximum=MAX_RECORD_BYTES):
    """Bound literal JSON before encoding, including cycles and nonfinite values."""
    pending, nodes, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _need(nodes <= MAX_NODES and depth <= MAX_DEPTH, "residual_structure_bound")
        if type(item) is str:
            _need(len(item) <= maximum, "residual_text_bound")
            try:
                text_bytes += len(item.encode("utf-8", "strict"))
            except UnicodeError:
                raise RepositoryResidualContextError("residual_invalid_text") from None
            _need(text_bytes <= maximum, "residual_text_bound")
        elif type(item) is dict:
            _need(len(item) <= 256 and all(type(key) is str for key in item),
                  "residual_mapping_invalid")
            pending.extend((part, depth + 1) for pair in item.items() for part in pair)
        elif type(item) in (list, tuple):
            _need(len(item) <= 256, "residual_sequence_bound")
            pending.extend((part, depth + 1) for part in item)
        else:
            _need(item is None or type(item) is bool
                  or type(item) is int and -(1 << 63) <= item < (1 << 63)
                  or type(item) is float and math.isfinite(item), "residual_value_invalid")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise RepositoryResidualContextError("residual_json_invalid") from None
    _need(len(encoded) <= maximum, "residual_wire_bound")
    return encoded


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _text(value, name):
    _need(type(value) is str and value and value.strip() == value
          and "\x00" not in value and len(value) <= 1_024,
          "residual_expected_" + name + "_invalid")
    return value


def _population(value, name, *, required, maximum):
    _need(type(value) in (tuple, list) and len(value) <= maximum
          and (not required or bool(value)), "residual_expected_" + name + "_invalid")
    result = tuple(_text(part, name) for part in value)
    _need(len(set(result)) == len(result), "residual_expected_" + name + "_duplicate")
    return result


def _expected(*, expected_task_id, repository_id, tree_id, policy_id, policy_revision,
              write_paths, validation_commands, acceptance_ids):
    result = dict(task_id=_text(expected_task_id, "task_id"),
                  repository_id=_text(repository_id, "repository_id"),
                  tree_id=_text(tree_id, "tree_id"),
                  policy_id=_text(policy_id, "policy_id"),
                  policy_revision=_text(policy_revision, "policy_revision"),
                  write_paths=_population(write_paths, "write_paths", required=True, maximum=64),
                  validation_commands=_population(validation_commands, "validation_commands",
                                                  required=True, maximum=64),
                  acceptance_ids=_population(acceptance_ids, "acceptance_ids", required=False,
                                             maximum=128))
    _wire(result, maximum=MAX_PACKET_BYTES)
    return result


def _packet(residual_packet):
    """Reconstruct every typed field; the old decoder otherwise trusts/coerces some."""
    try:
        if type(residual_packet) is ResidualLlmPacket:
            original_fields = {field.name: getattr(residual_packet, field.name)
                               for field in fields(ResidualLlmPacket)}
            detached_fields = json.loads(_wire(original_fields, maximum=MAX_PACKET_BYTES))
            # A typed instance's to_dict hard-codes authority flags. Reconstruct
            # actual attributes instead, so forged flags and changed nested data
            # cannot hide behind that projection or a stale packet_id.
            _need(type(detached_fields.get("packet_id")) is str
                  and bool(detached_fields["packet_id"]), "residual_packet_identity_missing")
            sealed = ResidualLlmPacket(**detached_fields)
            reconstructed_fields = {field.name: getattr(sealed, field.name)
                                    for field in fields(ResidualLlmPacket)}
            _need(_wire(reconstructed_fields, maximum=MAX_PACKET_BYTES)
                  == _wire(detached_fields, maximum=MAX_PACKET_BYTES),
                  "residual_packet_normalized")
        else:
            _need(type(residual_packet) is dict, "residual_packet_type_invalid")
            record = json.loads(_wire(residual_packet, maximum=MAX_PACKET_BYTES))
            has_content_id = "content_id" in record
            claimed_content_id = record.pop("content_id", None)
            _need(type(record.get("packet_id")) is str and bool(record["packet_id"]),
                  "residual_packet_identity_missing")
            sealed = ResidualLlmPacket.from_dict(record)
            # Require the full canonical sealed payload. Missing/unknown fields,
            # falsy malformed authority flags and normalized values cannot drop.
            _need(_wire(sealed.to_dict(), maximum=MAX_PACKET_BYTES)
                  == _wire(record, maximum=MAX_PACKET_BYTES), "residual_packet_not_canonical")
            _need(not has_content_id or type(claimed_content_id) is str
                  and claimed_content_id == sealed.packet_id,
                  "residual_packet_content_identity_mismatch")
        return sealed
    except RepositoryResidualContextError:
        raise
    except Exception:
        raise RepositoryResidualContextError("residual_packet_invalid") from None


@dataclass(frozen=True)
class PreparedRepositoryResidualContext:
    """Detached structural evidence, with no live issuance or execution grant."""

    _packet_json: str
    _context_json: str
    _receipt_json: str

    @property
    def packet(self):
        return json.loads(self._packet_json)

    @property
    def context(self):
        return json.loads(self._context_json)

    @property
    def receipt(self):
        return json.loads(self._receipt_json)

    @property
    def record(self):
        return dict(schema=SCHEMA, packet=self.packet, context=self.context, receipt=self.receipt)

    @property
    def prompt(self):
        return self._context_json

    @property
    def packet_id(self):
        return self.receipt["packet_id"]

    @property
    def context_id(self):
        return self.receipt["context_id"]


def prepare_repository_residual_context(
    *, residual_packet, expected_task_id, repository_id, tree_id, policy_id,
    policy_revision, write_paths, validation_commands, acceptance_ids,
):
    """Bind a complete sealed packet to exact independently supplied task scope.

    Populations and their order must match exactly. Original task obligations
    remain with the caller; residual obligations are additional candidate work.
    Authority roots are retained as inert references, never admitted as grants.
    """
    expected = _expected(expected_task_id=expected_task_id, repository_id=repository_id,
        tree_id=tree_id, policy_id=policy_id, policy_revision=policy_revision,
        write_paths=write_paths, validation_commands=validation_commands,
        acceptance_ids=acceptance_ids)
    sealed = _packet(residual_packet)
    for name, value in expected.items():
        _need(getattr(sealed, name) == value, "residual_" + name + "_mismatch")
    try:
        # This is the preparer's structural path allow-list, not a newly issued
        # native/resource lease. Empty environment prevents ambient-value drift.
        prepared = ResidualProviderInvocation(require_path_lease=True).prepare(
            sealed, path_lease=PathLease(permitted_write_paths=expected["write_paths"]),
            base_env={})
        context = json.loads(prepared.prompt_body)
        context_bytes = _wire(context, maximum=MAX_PACKET_BYTES)
        _need(context_bytes.decode("utf-8") == prepared.prompt_body,
              "residual_context_not_canonical")
        packet_bytes = _wire(sealed.to_dict(), maximum=MAX_PACKET_BYTES)
        receipt = dict(schema=RECEIPT_SCHEMA, packet_id=sealed.packet_id,
            context_id=prepared.content_id, packet_sha256=_sha(packet_bytes),
            packet_bytes=len(packet_bytes), context_sha256=_sha(context_bytes),
            context_bytes=len(context_bytes), binding_sha256=_sha(_wire(expected)),
            write_path_count=len(sealed.write_paths), validation_count=len(sealed.validation_commands),
            acceptance_count=len(sealed.acceptance_ids), residual_obligation_count=len(sealed.obligation_ids),
            scope="bound_residual_nomination_evidence_only", delivery_mode="self_contained",
            provider_retained_parent=False, nomination_only=True,
            provider_input_tokens=None, token_estimate=prepared.prompt_tokens,
            token_estimate_method="sealed_context_utf8_bytes_divided_by_3_ceil",
            token_estimate_scope="sealed_residual_context_only_excludes_full_task_prompt_and_wrappers",
            source_owner_freshness="not_revalidated_by_this_binding", provider_invoked=False,
            path_scope_checked=True, proof_authority=False, semantic_authority=False,
            write_authority=False, execution_authority=False, completion_authority=False,
            omission_authority=False, source_semantics_verified=False, world_proof_verified=False)
        result = PreparedRepositoryResidualContext(packet_bytes.decode("utf-8"),
            prepared.prompt_body, _wire(receipt).decode("utf-8"))
        _wire(result.record)
        return result
    except RepositoryResidualContextError:
        raise
    except Exception:
        raise RepositoryResidualContextError("residual_context_preparation_refused") from None


def replay_repository_residual_context(record, *, expected_task_id, repository_id,
    tree_id, policy_id, policy_revision, write_paths, validation_commands, acceptance_ids):
    """Recompute all detached bytes against caller-supplied historical bindings.

    The enclosing retained artifact must independently pin this record. A
    self-consistent record alone proves neither custody nor live source state.
    """
    _need(type(record) is dict and set(record) == {"schema", "packet", "context", "receipt"}
          and record["schema"] == SCHEMA, "residual_record_invalid")
    original = _wire(record)
    captured = json.loads(original)
    reconstructed = prepare_repository_residual_context(residual_packet=captured["packet"],
        expected_task_id=expected_task_id, repository_id=repository_id, tree_id=tree_id,
        policy_id=policy_id, policy_revision=policy_revision, write_paths=write_paths,
        validation_commands=validation_commands, acceptance_ids=acceptance_ids)
    _need(_wire(reconstructed.record) == original, "residual_record_reconstruction_mismatch")
    return reconstructed


def validate_repository_residual_context(binding, *, expected_task_id, repository_id,
    tree_id, policy_id, policy_revision, write_paths, validation_commands, acceptance_ids):
    """Revalidate a frozen value; direct construction conveys no extra privilege."""
    _need(type(binding) is PreparedRepositoryResidualContext, "residual_binding_type_invalid")
    try:
        # Check bounds before getters decode potentially forged private strings.
        for value in (binding._packet_json, binding._context_json, binding._receipt_json):
            _need(type(value) is str and len(value) <= MAX_RECORD_BYTES,
                  "residual_binding_storage_invalid")
        reconstructed = replay_repository_residual_context(binding.record,
            expected_task_id=expected_task_id, repository_id=repository_id, tree_id=tree_id,
            policy_id=policy_id, policy_revision=policy_revision, write_paths=write_paths,
            validation_commands=validation_commands, acceptance_ids=acceptance_ids)
        _need((binding._packet_json, binding._context_json, binding._receipt_json)
              == (reconstructed._packet_json, reconstructed._context_json,
                  reconstructed._receipt_json), "residual_binding_storage_not_canonical")
        return reconstructed.receipt
    except RepositoryResidualContextError:
        raise
    except Exception:
        raise RepositoryResidualContextError("residual_binding_invalid") from None


__all__ = ["PreparedRepositoryResidualContext", "RepositoryResidualContextError",
    "prepare_repository_residual_context", "replay_repository_residual_context",
    "validate_repository_residual_context"]
