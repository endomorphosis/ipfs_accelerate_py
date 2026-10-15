"""Lossless provider representation of an already admitted candidate prompt.

The owner retains its exact candidate envelope. This codec only moves its
canonical native JSON out of a JSON string and preserves the literal suffix.
Every evidence body, unknown field and public requirement stays inline. The
caller owns admission and freshness; a receipt proves neither of those.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re

from . import semantic_metadata_view as metadata


MODE = "native-context-object@1"
VIEW_SCHEMA = "supervisor-candidate-provider-view@1"
RECEIPT_SCHEMA = "supervisor-candidate-prompt-projection@1"
SELECTION_SCHEMA = "supervisor-candidate-prompt-selection@1"
SOURCE_SCHEMAS = frozenset("supervisor-owner-candidate-router-input@" + str(i) for i in (1, 2, 3))
MAX_BYTES = 256_000
INSTRUCTIONS = (
    "original_candidate retains every original candidate field except native_prompt. "
    "Its exact native_prompt is the canonical JSON of native_context followed by native_suffix. "
    "All native requirements, evidence and public instructions remain inline with their original requirement status. "
    "This representation changes no source freshness, proof status or authority."
)
_FALSE_FIELDS = frozenset({"proof_authority", "execution_authority", "completion_authority",
    "omission_authority", "source_freshness_verified", "program_semantics_proved",
    "provider_token_usage_verified", "complete_run_token_savings_claimed", "raw_content_exported"})
_BYTE_FIELDS = ("original_input", "native_prompt", "native_context", "native_suffix",
    "public_instruction_block", "candidate_view")
_RECEIPT_FIELDS = frozenset({"schema", "view_schema", "source_schema", "representation_lossless",
    "evidence_bodies_preserved", "unknown_fields_preserved", *_FALSE_FIELDS,
    *(name + suffix for name in _BYTE_FIELDS for suffix in ("_sha256", "_bytes"))})
_VIEW_FIELDS = frozenset({"schema", "instructions", "original_candidate", "native_context", "native_suffix"})
_SELECTION_FIELDS = frozenset({"schema", "requested_mode", "selected_mode", "fallback_reason",
    "selection_basis", "view_receipt", "view_receipt_sha256", *_FALSE_FIELDS,
    *(name + suffix for name in ("native_complete", "candidate_complete", "selected_complete")
      for suffix in ("_sha256", "_bytes"))})


class CandidatePromptProjectionError(ValueError):
    """Closed representation, size or identity failure; no input text exposed."""


def _need(condition, reason):
    if not condition:
        raise CandidatePromptProjectionError(reason)


def _bytes(text):
    try:
        size = metadata._byte_count(text)
    except (ValueError, TypeError, AttributeError, UnicodeError):
        raise CandidatePromptProjectionError("valid bounded UTF-8 text required") from None
    _need(type(text) is str and size <= MAX_BYTES, "candidate projection text exceeds bound")
    return size


def _json(value):
    try:
        # Owner candidate envelopes use JSON numbers, including finite public
        # timing measurements. Formal-IR canonicalization rejects all floats
        # and therefore is not the serialization contract for this boundary.
        text = json.dumps(value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False)
        _bytes(text)
        return text
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise CandidatePromptProjectionError("invalid bounded canonical candidate JSON") from None


def _parse(text):
    _bytes(text)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result, "duplicate candidate JSON key")
            result[key] = value
        return result

    def no_constant(_value):
        raise CandidatePromptProjectionError("nonfinite candidate JSON number")

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=no_constant)
        nodes = 0
        def bounded(item, depth):
            nonlocal nodes
            nodes += 1
            _need(depth <= metadata.MAX_DEPTH and nodes <= metadata.MAX_NODES,
                "candidate JSON exceeds structure bound")
            _need(not isinstance(item, float) or math.isfinite(item),
                "nonfinite candidate JSON number")
            if isinstance(item, dict):
                for child in item.values():
                    bounded(child, depth + 1)
            elif isinstance(item, list):
                for child in item:
                    bounded(child, depth + 1)
        bounded(value, 0)
        _need(_json(value) == text, "canonical candidate JSON required")
        return value
    except (ValueError, TypeError, RecursionError, UnicodeError, OverflowError):
        raise CandidatePromptProjectionError("invalid bounded canonical candidate JSON") from None


def _measure(name, text):
    return {name + "_sha256": metadata._sha(text), name + "_bytes": _bytes(text)}


def _native_parts(text):
    _bytes(text)
    try:
        _wire, end = json.JSONDecoder().raw_decode(text)
    except (ValueError, TypeError, RecursionError, OverflowError):
        raise CandidatePromptProjectionError("canonical native context prefix required") from None
    prefix = text[:end]
    wire = _parse(prefix)
    _need(type(wire) is dict and type(wire.get("evidence")) is list
        and type(wire.get("contract_version")) is int and wire["contract_version"] == 1
        and all(key in wire for key in ("goal", "scope", "acceptance", "authority")),
        "supported native context shape required")
    return wire, prefix, text[end:]


def _original(text):
    wire = _parse(text)
    _need(type(wire) is dict and type(wire.get("schema")) is str
        and wire["schema"] in SOURCE_SCHEMAS and type(wire.get("native_prompt")) is str
        and type(wire.get("original_public_instruction_block")) is str,
        "supported original candidate envelope required")
    native, prefix, suffix = _native_parts(wire["native_prompt"])
    return wire, native, prefix, suffix


def _validated_counts(receipt, names):
    for name in names:
        value, size = receipt[name + "_sha256"], receipt[name + "_bytes"]
        _need(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None
            and type(size) is int and 0 <= size <= MAX_BYTES,
            "candidate projection digest or byte count differs")


def _receipt(value):
    _need(type(value) is dict, "candidate projection receipt required")
    value = _parse(_json(value))
    _need(set(value) == _RECEIPT_FIELDS and value["schema"] == RECEIPT_SCHEMA
        and value["view_schema"] == VIEW_SCHEMA and type(value["source_schema"]) is str
        and value["source_schema"] in SOURCE_SCHEMAS
        and all(value[key] is True for key in ("representation_lossless", "evidence_bodies_preserved",
            "unknown_fields_preserved")) and all(value[key] is False for key in _FALSE_FIELDS),
        "candidate projection receipt schema or flags differ")
    _validated_counts(value, _BYTE_FIELDS)
    _need(value["native_prompt_bytes"] == value["native_context_bytes"] + value["native_suffix_bytes"],
        "candidate native prefix and suffix byte counts differ")
    return value


@dataclass(frozen=True)
class CandidatePromptView:
    original_prompt: str
    provider_prompt: str
    receipt_json: str

    @property
    def receipt(self):
        return _parse(self.receipt_json)


@dataclass(frozen=True)
class CandidatePromptSelection:
    selected_prompt: str
    selected_mode: str
    receipt_json: str

    @property
    def receipt(self):
        return _parse(self.receipt_json)


def project_candidate_prompt(prompt: str) -> CandidatePromptView:
    """Build a lossless candidate; the complete-input selector decides use."""
    wire, native, prefix, suffix = _original(prompt)
    candidate = _json(dict(schema=VIEW_SCHEMA, instructions=INSTRUCTIONS,
        original_candidate={key: value for key, value in wire.items() if key != "native_prompt"},
        native_context=native, native_suffix=suffix))
    receipt = dict(schema=RECEIPT_SCHEMA, view_schema=VIEW_SCHEMA, source_schema=wire["schema"],
        representation_lossless=True, evidence_bodies_preserved=True, unknown_fields_preserved=True,
        **{key: False for key in _FALSE_FIELDS})
    for name, text in (("original_input", prompt), ("native_prompt", wire["native_prompt"]),
            ("native_context", prefix), ("native_suffix", suffix),
            ("public_instruction_block", wire["original_public_instruction_block"]),
            ("candidate_view", candidate)):
        receipt.update(_measure(name, text))
    result = CandidatePromptView(prompt, candidate, _json(receipt))
    _need(restore_candidate_prompt(candidate, receipt) == prompt,
        "candidate projection did not round-trip exactly")
    return result


def restore_candidate_prompt(provider_prompt: str, receipt: dict) -> str:
    """Restore exact original bytes against an independently retained receipt."""
    retained = _receipt(receipt)
    wire = _parse(provider_prompt)
    _need(type(wire) is dict and set(wire) == _VIEW_FIELDS and wire["schema"] == VIEW_SCHEMA
        and wire["instructions"] == INSTRUCTIONS and type(wire["original_candidate"]) is dict
        and "native_prompt" not in wire["original_candidate"] and type(wire["native_suffix"]) is str,
        "candidate provider view shape differs")
    prefix = _json(wire["native_context"])
    native_prompt = prefix + wire["native_suffix"]
    original = _json({**wire["original_candidate"], "native_prompt": native_prompt})
    original_wire, native, parsed_prefix, suffix = _original(original)
    _need(prefix == parsed_prefix and suffix == wire["native_suffix"]
        and original_wire["schema"] == retained["source_schema"],
        "candidate restored native context differs")
    for name, text in (("original_input", original), ("native_prompt", native_prompt),
            ("native_context", prefix), ("native_suffix", suffix),
            ("public_instruction_block", original_wire["original_public_instruction_block"]),
            ("candidate_view", provider_prompt)):
        _need(all(retained[key] == value for key, value in _measure(name, text).items()),
            "candidate projection differs from retained binding")
    return original


def validate_selection_receipt(receipt: dict) -> dict:
    """Check a closed observation, without granting admission or byte custody."""
    _need(type(receipt) is dict, "candidate projection selection receipt required")
    value = _parse(_json(receipt))
    _need(set(value) == _SELECTION_FIELDS and value["schema"] == SELECTION_SCHEMA
        and value["requested_mode"] == MODE and value["selection_basis"] == "complete_utf8_bytes"
        and all(value[key] is False for key in _FALSE_FIELDS),
        "candidate selection schema or flags differ")
    view = _receipt(value["view_receipt"])
    _need(value["view_receipt_sha256"] == metadata._sha(_json(view)),
        "candidate selection projection receipt differs")
    _validated_counts(value, ("native_complete", "candidate_complete", "selected_complete"))
    _need(value["native_complete_bytes"] >= view["original_input_bytes"]
        and value["candidate_complete_bytes"] >= view["candidate_view_bytes"]
        and value["native_complete_bytes"] - view["original_input_bytes"]
            == value["candidate_complete_bytes"] - view["candidate_view_bytes"],
        "candidate complete input wrapper byte counts differ")
    smaller = value["candidate_complete_bytes"] < value["native_complete_bytes"]
    _need(value["selected_mode"] == (MODE if smaller else "legacy")
        and value["fallback_reason"] == (None if smaller else "complete_input_not_smaller"),
        "candidate selection does not match strict byte reduction")
    chosen = "candidate_complete" if smaller else "native_complete"
    _need(all(value["selected_complete" + suffix] == value[chosen + suffix]
        for suffix in ("_sha256", "_bytes")), "candidate selected input binding differs")
    return value


def select_candidate_prompt(view: CandidatePromptView, native_complete_prompt: str,
                            candidate_complete_prompt: str) -> CandidatePromptSelection:
    """Select only a strictly smaller full input with identical literal wrappers.

    UTF-8 size is a representation measurement, not a tokenizer measurement or
    evidence of total provider usage, cached input, output or billing savings.
    """
    _need(type(view) is CandidatePromptView, "typed candidate provider view required")
    _need(restore_candidate_prompt(view.provider_prompt, view.receipt) == view.original_prompt,
        "candidate view original input differs")
    _bytes(native_complete_prompt)
    _bytes(candidate_complete_prompt)
    _need(native_complete_prompt.count(view.original_prompt) == 1,
        "complete original candidate occurrence differs")
    prefix, suffix = native_complete_prompt.split(view.original_prompt)
    _need(candidate_complete_prompt == prefix + view.provider_prompt + suffix
        and candidate_complete_prompt.count(view.provider_prompt) == 1,
        "candidate complete input changed literal wrappers")
    smaller = _bytes(candidate_complete_prompt) < _bytes(native_complete_prompt)
    chosen = candidate_complete_prompt if smaller else native_complete_prompt
    receipt = dict(schema=SELECTION_SCHEMA, requested_mode=MODE,
        selected_mode=MODE if smaller else "legacy",
        fallback_reason=None if smaller else "complete_input_not_smaller",
        selection_basis="complete_utf8_bytes", view_receipt=view.receipt,
        view_receipt_sha256=metadata._sha(view.receipt_json), **{key: False for key in _FALSE_FIELDS})
    for name, text in (("native_complete", native_complete_prompt),
            ("candidate_complete", candidate_complete_prompt), ("selected_complete", chosen)):
        receipt.update(_measure(name, text))
    validate_selection_receipt(receipt)
    return CandidatePromptSelection(chosen, receipt["selected_mode"], _json(receipt))


__all__ = ["CandidatePromptProjectionError", "CandidatePromptView", "CandidatePromptSelection",
    "project_candidate_prompt", "restore_candidate_prompt", "select_candidate_prompt",
    "validate_selection_receipt"]
