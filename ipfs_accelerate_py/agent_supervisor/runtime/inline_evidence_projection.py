"""Lossless inline objects for admitted, canonical evidence-artifact summaries.

This pure codec changes only representation. It performs no source, provider,
file or model operations and grants no freshness, omission or dispatch authority.
The owner retains admission and source checks; all artifact bodies remain inline.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

from ..context.context_compiler import build_text_context_references
from ..context.context_contracts import ContextReference
from . import semantic_router_translation as semantic

MODE = "inline-evidence-objects@1"
VIEW_SCHEMA = "supervisor-inline-evidence-view@1"
RECEIPT_SCHEMA = "supervisor-inline-evidence-projection@1"
SELECTION_SCHEMA = "supervisor-inline-evidence-selection@1"
REFERENCE_KEY = "$inline_evidence"
MAX_BYTES = 2_000_000
MAX_DEPTH = 40
MAX_NODES = 100_000
MAX_ARTIFACTS = 32
MAX_ROWS = 4096
CHUNK_BYTES = 6144
KINDS = {
    "intent-world-context": "intent-world",
    "source-formalization-context": "source-formalization",
    "code-retrieval-context": "code-retrieval",
}
INSTRUCTIONS = (
    "A native_context.evidence summary with $inline_evidence identifies the complete "
    "canonical JSON object at that index in inline_evidence_artifacts. Read that entire "
    "object as the row's required evidence; its original chunk_index and byte_count "
    "identify the exact UTF-8 slice. Every other row field is unchanged. All evidence "
    "stays inline. This representation grants no proof, freshness, omission or authority."
)
_TRANSPORT_FIELDS = frozenset({"schema", "translation_cid", "translation_table", "native_context",
    "translated_semantic", "native_suffix", "instructions", "execution_authority", "completion_authority"})
_VIEW_FIELDS = _TRANSPORT_FIELDS | {"source_transport_schema", "inline_evidence_artifacts",
    "inline_evidence_instructions"}
_FALSE_FIELDS = frozenset({"proof_authority", "execution_authority", "completion_authority",
    "omission_authority", "source_freshness_verified", "program_semantics_proved",
    "provider_token_usage_verified", "complete_run_token_savings_claimed", "raw_content_exported"})
_TRUE_FIELDS = frozenset({"representation_lossless", "evidence_bodies_preserved", "unknown_fields_preserved"})
_COUNT_FIELDS = frozenset({"artifact_count", "evidence_row_count", "summary_utf8_bytes",
    "summary_json_string_bytes", "removed_string_overhead_bytes"})
_DIGEST_FIELDS = frozenset({"native_prompt_sha256", "translation_table_sha256",
    "inline_artifacts_sha256", "group_bindings_sha256"})
_RECEIPT_FIELDS = frozenset({"schema", "view_schema", "source_schema", "provider_token_usage",
    "native_transport_sha256", "native_transport_bytes", "candidate_view_sha256", "candidate_view_bytes",
    *_COUNT_FIELDS, *_DIGEST_FIELDS, *_FALSE_FIELDS, *_TRUE_FIELDS})
_SELECTION_FIELDS = frozenset({"schema", "requested_mode", "selected_mode", "fallback_reason",
    "selection_basis", "view_receipt", "view_receipt_sha256", "provider_token_usage", *_FALSE_FIELDS,
    *(name + suffix for name in ("native_complete", "candidate_complete", "selected_complete")
        for suffix in ("_sha256", "_bytes"))})


class InlineEvidenceProjectionError(ValueError):
    """Closed representation failure; errors never include input content."""


def _need(value, reason):
    if not value:
        raise InlineEvidenceProjectionError(reason)


def _bytes(text):
    try:
        _need(type(text) is str, "bounded UTF-8 text required")
        result = len(text.encode("utf-8"))
        _need(result <= MAX_BYTES, "inline evidence text exceeds bound")
        return result
    except UnicodeError:
        raise InlineEvidenceProjectionError("bounded UTF-8 text required") from None


def _sha(text):
    _bytes(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json(value):
    try:
        # Retrieval scores may be finite floats. This is a representation codec,
        # not the formal-IR value normalizer, which intentionally rejects floats.
        text = json.dumps(value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False)
        _bytes(text)
        return text
    except (ValueError, TypeError, RecursionError, UnicodeError, OverflowError):
        raise InlineEvidenceProjectionError("invalid bounded inline evidence JSON") from None


def _parse(text):
    _bytes(text)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result, "duplicate inline evidence JSON key")
            result[key] = value
        return result
    def no_constant(_):
        raise InlineEvidenceProjectionError("nonfinite inline evidence number")
    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=no_constant)
        nodes = 0
        def bounded(item, depth):
            nonlocal nodes
            nodes += 1
            _need(depth <= MAX_DEPTH and nodes <= MAX_NODES,
                "inline evidence JSON exceeds structure bound")
            _need(type(item) is not float or math.isfinite(item), "nonfinite inline evidence number")
            if type(item) is dict:
                for child in item.values(): bounded(child, depth + 1)
            elif type(item) is list:
                for child in item: bounded(child, depth + 1)
        bounded(value, 0)
        _need(_json(value) == text, "canonical inline evidence JSON required")
        return value
    except (ValueError, TypeError, RecursionError, UnicodeError, OverflowError):
        raise InlineEvidenceProjectionError("invalid bounded canonical inline evidence JSON") from None


def _measure(name, text):
    return {name + "_sha256": _sha(text), name + "_bytes": _bytes(text)}


def _transport(text):
    wire = _parse(text)
    _need(type(wire) is dict and set(wire) == _TRANSPORT_FIELDS
        and wire.get("schema") == semantic.TRANSPORT_SCHEMA
        and wire.get("execution_authority") is False and wire.get("completion_authority") is False
        and type(wire.get("native_context")) is dict
        and type(wire["native_context"].get("evidence")) is list
        and len(wire["native_context"]["evidence"]) <= MAX_ROWS,
        "original semantic transport required")
    return wire


def _validate_group(rows, text):
    """Reuse the native producer, retaining all additional fields verbatim."""
    first = rows[0]
    kind = first["kind"]
    meta = first.get("metadata")
    _need(type(meta) is dict and meta.get("required") is True
        and type(meta.get("priority")) is int and type(meta.get("coverage_ids")) is list,
        "required evidence group metadata differs")
    try:
        expected = build_text_context_references(text, reference_prefix=KINDS[kind], kind=kind,
            path=first["path"], repository_id=first["repository_id"], tree_id=first["tree_id"],
            required=True, priority=meta["priority"], coverage_ids=meta["coverage_ids"],
            chunk_bytes=CHUNK_BYTES)
        _need(len(rows) == len(expected), "evidence group chunk count differs")
        for row, reference in zip(rows, expected):
            # Native string enums must be compared in their serialized wire form.
            known = _parse(_json(reference.to_record()))
            metadata = row.get("metadata")
            _need(type(metadata) is dict
                and all(key in metadata and type(metadata[key]) is type(value)
                    and _json(metadata[key]) == _json(value) for key, value in known["metadata"].items()),
                "evidence group chunk metadata differs")
            _need(all(key in row and type(row[key]) is type(value) and _json(row[key]) == _json(value)
                for key, value in known.items() if key not in {"metadata", "content_id"}),
                "evidence group native chunk differs")
            # The native reference validates its complete metadata identity.
            # Future outer reference fields are retained but not reinterpreted.
            ContextReference.from_dict({key: row[key] for key in known})
    except InlineEvidenceProjectionError:
        raise
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise InlineEvidenceProjectionError("evidence group native contract differs") from None


def _groups(evidence):
    grouped = {}
    for position, row in enumerate(evidence):
        if type(row) is not dict or type(row.get("kind")) is not str or row["kind"] not in KINDS:
            continue
        summary = row.get("summary")
        _need(not (type(summary) is dict and REFERENCE_KEY in summary),
            "inline evidence reference marker collision")
        if type(summary) is not str:
            continue
        meta = row.get("metadata")
        _need(type(meta) is dict, "evidence group metadata unavailable")
        if meta.get("required") is not True:
            continue
        fields = (row.get("kind"), row.get("path"), row.get("repository_id"), row.get("tree_id"),
                  meta.get("artifact_content_id"))
        _need(all(type(value) is str and value for value in fields),
            "evidence group identity unavailable")
        grouped.setdefault(fields, []).append((position, row))
    _need(len(grouped) <= MAX_ARTIFACTS, "inline evidence artifact count exceeds bound")
    selected = []
    for pairs in grouped.values():
        positions, rows = [p for p, _ in pairs], [row for _, row in pairs]
        text = "".join(row["summary"] for row in rows)
        _validate_group(rows, text)
        try:
            payload = _parse(text)
        except InlineEvidenceProjectionError:
            # Opaque or noncanonical summaries keep their existing string form.
            continue
        if type(payload) is dict:
            selected.append((positions, rows, text, payload))
    return selected


def _project(original_prompt):
    wire = _transport(original_prompt)
    groups = _groups(wire["native_context"]["evidence"])
    artifacts, bindings = [], []
    string_bytes = text_bytes = row_count = 0
    for index, (positions, rows, text, payload) in enumerate(groups):
        artifacts.append(payload)
        bindings.append({"positions": positions, "artifact_sha256": _sha(text),
            "rows_sha256": _sha(_json(rows))})
        text_bytes += _bytes(text)
        row_count += len(rows)
        for row in rows:
            string_bytes += _bytes(_json(row["summary"]))
            row["summary"] = {REFERENCE_KEY: index}
    wire.update(schema=VIEW_SCHEMA, source_transport_schema=semantic.TRANSPORT_SCHEMA,
        inline_evidence_artifacts=artifacts, inline_evidence_instructions=INSTRUCTIONS)
    return _json(wire), dict(artifact_count=len(artifacts), evidence_row_count=row_count,
        summary_utf8_bytes=text_bytes, summary_json_string_bytes=string_bytes,
        removed_string_overhead_bytes=string_bytes - text_bytes,
        inline_artifacts_sha256=_sha(_json(artifacts)), group_bindings_sha256=_sha(_json(bindings)))


@dataclass(frozen=True)
class InlineEvidenceView:
    original_prompt: str
    provider_prompt: str
    receipt_json: str

    @property
    def receipt(self):
        return _parse(self.receipt_json)


@dataclass(frozen=True)
class InlineEvidenceSelection:
    selected_prompt: str
    selected_mode: str
    receipt_json: str

    @property
    def receipt(self):
        return _parse(self.receipt_json)


def project_inline_evidence(encoded: semantic.EncodedSemanticPrompt) -> InlineEvidenceView:
    """Project current owner-verified input, without independently granting freshness."""
    _need(type(encoded) is semantic.EncodedSemanticPrompt and encoded.freshness_checked is True,
        "current typed semantic prompt required")
    try:
        _need(semantic._restore(encoded.provider_prompt, encoded.table) == encoded.native_prompt,
            "native semantic inverse differs")
        _transport(encoded.provider_prompt)
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError):
        raise InlineEvidenceProjectionError("verified original semantic transport required") from None
    candidate, counts = _project(encoded.provider_prompt)
    receipt = dict(schema=RECEIPT_SCHEMA, view_schema=VIEW_SCHEMA, source_schema=semantic.TRANSPORT_SCHEMA,
        native_prompt_sha256=_sha(encoded.native_prompt), translation_table_sha256=_sha(encoded.table.payload_json),
        provider_token_usage=None, **counts, **{key: False for key in _FALSE_FIELDS},
        **{key: True for key in _TRUE_FIELDS}, **_measure("native_transport", encoded.provider_prompt),
        **_measure("candidate_view", candidate))
    result = InlineEvidenceView(encoded.provider_prompt, candidate, _json(receipt))
    _need(restore_inline_evidence(candidate, receipt) == encoded.provider_prompt,
        "inline evidence projection did not round-trip")
    return result


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _counts(receipt, names):
    _need(all(_digest(receipt[name + "_sha256"])
        and type(receipt[name + "_bytes"]) is int and 0 <= receipt[name + "_bytes"] <= MAX_BYTES
        for name in names), "inline evidence digest or byte count differs")


def validate_projection_receipt(receipt):
    _need(type(receipt) is dict, "inline evidence projection receipt required")
    value = _parse(_json(receipt))
    _need(set(value) == _RECEIPT_FIELDS and value["schema"] == RECEIPT_SCHEMA
        and value["view_schema"] == VIEW_SCHEMA and value["source_schema"] == semantic.TRANSPORT_SCHEMA
        and all(value[key] is False for key in _FALSE_FIELDS)
        and all(value[key] is True for key in _TRUE_FIELDS) and value["provider_token_usage"] is None,
        "inline evidence receipt schema or authority differs")
    _counts(value, ("native_transport", "candidate_view"))
    _need(all(_digest(value[key]) for key in _DIGEST_FIELDS)
        and all(type(value[key]) is int and 0 <= value[key] <= MAX_BYTES for key in _COUNT_FIELDS)
        and value["artifact_count"] <= MAX_ARTIFACTS and value["evidence_row_count"] <= MAX_ROWS
        and value["artifact_count"] <= value["evidence_row_count"]
        and value["summary_json_string_bytes"] - value["summary_utf8_bytes"]
            == value["removed_string_overhead_bytes"], "inline evidence count or binding differs")
    return value


def restore_inline_evidence(provider_prompt: str, receipt: dict) -> str:
    """Restore exact transport bytes against an independently retained receipt."""
    retained = validate_projection_receipt(receipt)
    wire = _parse(provider_prompt)
    _need(type(wire) is dict and set(wire) == _VIEW_FIELDS and wire["schema"] == VIEW_SCHEMA
        and wire["source_transport_schema"] == semantic.TRANSPORT_SCHEMA
        and wire["inline_evidence_instructions"] == INSTRUCTIONS
        and type(wire["native_context"]) is dict and type(wire["native_context"].get("evidence")) is list
        and type(wire["inline_evidence_artifacts"]) is list
        and len(wire["inline_evidence_artifacts"]) <= MAX_ARTIFACTS
        and all(type(item) is dict for item in wire["inline_evidence_artifacts"])
        and all(retained[key] == value for key, value in _measure("candidate_view", provider_prompt).items()),
        "inline evidence view differs from retained receipt")
    artifacts = wire["inline_evidence_artifacts"]
    groups = [[] for _ in artifacts]
    for row in wire["native_context"]["evidence"]:
        if type(row) is not dict or type(row.get("kind")) is not str or row["kind"] not in KINDS:
            continue
        summary = row.get("summary")
        if type(summary) is not dict or REFERENCE_KEY not in summary:
            continue
        index = summary[REFERENCE_KEY]
        _need(set(summary) == {REFERENCE_KEY} and type(index) is int and 0 <= index < len(artifacts),
            "inline evidence artifact reference differs")
        groups[index].append(row)
    for artifact, rows in zip(artifacts, groups):
        _need(bool(rows), "inline evidence artifact is unreferenced")
        text = _json(artifact)
        raw, offset = text.encode("utf-8"), 0
        for row in rows:
            size = row.get("byte_count")
            _need(type(size) is int and 0 < size <= CHUNK_BYTES and offset + size <= len(raw),
                "inline evidence chunk byte count differs")
            try:
                row["summary"] = raw[offset:offset + size].decode("utf-8")
            except UnicodeError:
                raise InlineEvidenceProjectionError("inline evidence UTF-8 chunk boundary differs") from None
            offset += size
        _need(offset == len(raw), "inline evidence artifact coverage differs")
        _validate_group(rows, text)
    wire["schema"] = wire.pop("source_transport_schema")
    del wire["inline_evidence_artifacts"]
    del wire["inline_evidence_instructions"]
    original = _json(wire)
    _need(all(retained[key] == value for key, value in _measure("native_transport", original).items()),
        "inline evidence native transport differs")
    reproduced, counts = _project(original)
    _need(reproduced == provider_prompt and all(retained[key] == value for key, value in counts.items()),
        "inline evidence projection replay differs")
    return original


def validate_selection_receipt(receipt):
    _need(type(receipt) is dict, "inline evidence selection receipt required")
    value = _parse(_json(receipt))
    _need(set(value) == _SELECTION_FIELDS and value["schema"] == SELECTION_SCHEMA
        and value["requested_mode"] == MODE and value["selection_basis"] == "complete_prompt_utf8_bytes"
        and all(value[key] is False for key in _FALSE_FIELDS) and value["provider_token_usage"] is None,
        "inline evidence selection schema or authority differs")
    view = validate_projection_receipt(value["view_receipt"])
    _need(value["view_receipt_sha256"] == _sha(_json(view)), "inline evidence selection view binding differs")
    _counts(value, ("native_complete", "candidate_complete", "selected_complete"))
    _need(value["native_complete_bytes"] >= view["native_transport_bytes"]
        and value["candidate_complete_bytes"] >= view["candidate_view_bytes"]
        and value["native_complete_bytes"] - view["native_transport_bytes"]
            == value["candidate_complete_bytes"] - view["candidate_view_bytes"],
        "inline evidence complete wrapper byte counts differ")
    smaller = value["candidate_complete_bytes"] < value["native_complete_bytes"]
    _need(value["selected_mode"] == (MODE if smaller else "legacy")
        and value["fallback_reason"] == (None if smaller else "complete_input_not_smaller"),
        "inline evidence selection differs from strict byte reduction")
    chosen = "candidate_complete" if smaller else "native_complete"
    _need(all(value["selected_complete" + suffix] == value[chosen + suffix]
        for suffix in ("_sha256", "_bytes")), "inline evidence selected input differs")
    return value


def select_inline_evidence(view: InlineEvidenceView, native_complete_prompt: str,
        candidate_complete_prompt: str) -> InlineEvidenceSelection:
    """Select only a smaller complete input with identical literal wrappers."""
    _need(type(view) is InlineEvidenceView, "typed inline evidence view required")
    _need(restore_inline_evidence(view.provider_prompt, view.receipt) == view.original_prompt,
        "inline evidence original input differs")
    _bytes(native_complete_prompt); _bytes(candidate_complete_prompt)
    _need(native_complete_prompt.count(view.original_prompt) == 1,
        "complete native transport occurrence differs")
    prefix, suffix = native_complete_prompt.split(view.original_prompt)
    _need(candidate_complete_prompt == prefix + view.provider_prompt + suffix
        and candidate_complete_prompt.count(view.provider_prompt) == 1,
        "inline evidence complete input changed literal wrappers")
    smaller = _bytes(candidate_complete_prompt) < _bytes(native_complete_prompt)
    selected = candidate_complete_prompt if smaller else native_complete_prompt
    receipt = dict(schema=SELECTION_SCHEMA, requested_mode=MODE,
        selected_mode=MODE if smaller else "legacy", fallback_reason=None if smaller else "complete_input_not_smaller",
        selection_basis="complete_prompt_utf8_bytes", view_receipt=view.receipt,
        view_receipt_sha256=_sha(view.receipt_json), provider_token_usage=None,
        **{key: False for key in _FALSE_FIELDS}, **_measure("native_complete", native_complete_prompt),
        **_measure("candidate_complete", candidate_complete_prompt), **_measure("selected_complete", selected))
    validate_selection_receipt(receipt)
    return InlineEvidenceSelection(selected, receipt["selected_mode"], _json(receipt))


__all__ = ["InlineEvidenceProjectionError", "InlineEvidenceView", "InlineEvidenceSelection",
    "project_inline_evidence", "restore_inline_evidence", "select_inline_evidence",
    "validate_projection_receipt", "validate_selection_receipt"]
