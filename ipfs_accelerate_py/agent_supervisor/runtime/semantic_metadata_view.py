"""Lossless readable factoring of an upstream-verified semantic @1 transport.

Common binding values stay inline and apply to every row in their named group.
Source bodies, paths, signatures, statuses and unknown fields are untouched.
This codec verifies representation equality only. Source freshness, producer
admission and the trusted receipt's custody belong to the existing owner.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

from ..context.context_contracts import canonical_context_json_bytes


TRANSPORT_SCHEMA = "supervisor-semantic-router-input@1"
VIEW_SCHEMA = "supervisor-semantic-metadata-view@1"
RECEIPT_SCHEMA = "supervisor-semantic-metadata-view-receipt@1"
SELECTION_SCHEMA = "supervisor-semantic-metadata-selection@1"
TOKEN_PROXY = "utf8-bytes-ceil-div4@1"
MAX_BYTES = 2_000_000
MAX_ROWS = 4096
MAX_DEPTH = 40
MAX_NODES = 100_000

CAPSULE_FIELDS = frozenset({"schema", "capsule_schema", "semantic_index_schema",
    "capsule_compiler_version", "extractor_version", "source_cid"})
ADMISSION_FIELDS = frozenset({"schema", "interface", "freshness"})
ADMISSION_REF_FIELDS = frozenset({"source_cid", "semantic_state_root_cid", "raw_source_required"})
GROUP_FIELDS = {"capsules": CAPSULE_FIELDS, "admissions": ADMISSION_FIELDS,
                "admission_refs": ADMISSION_REF_FIELDS}

METADATA_INSTRUCTIONS = (
    "Each translated_semantic.capsules row inherits common_bindings.capsules; "
    "each admissions row inherits common_bindings.admissions, and its ref inherits "
    "common_bindings.admission_refs. Shared fields have identical values in every "
    "row and no overrides. All values remain inline. This factoring grants no "
    "proof, source omission or authority."
)
_TRANSPORT_FIELDS = frozenset({"schema", "translation_cid", "translation_table", "native_context",
    "translated_semantic", "native_suffix", "instructions", "execution_authority", "completion_authority"})
_VIEW_FIELDS = _TRANSPORT_FIELDS | {"source_transport_schema", "common_bindings", "metadata_instructions"}
_BINDING_FIELDS = frozenset({"translation_cid", "task_id", "scope_cid", "semantic_root_cid",
    "source_manifest_sha256", "native_core_sha256", "native_context_sha256", "native_suffix_sha256",
    "original_translated_semantic_sha256"})
_FALSE_FIELDS = frozenset({"source_freshness_verified", "program_semantics_proved", "omission_authority",
    "proof_authority", "execution_authority", "completion_authority", "publication_authority"})
_RECEIPT_FIELDS = (_BINDING_FIELDS | _FALSE_FIELDS | {"schema", "view_schema", "source_transport_schema",
    "native_transport_sha256", "native_transport_bytes", "candidate_view_sha256", "candidate_view_bytes",
    "common_bindings_sha256", "shared_field_paths", "capsule_count", "admission_count",
    "token_proxy", "native_transport_proxy_tokens", "candidate_view_proxy_tokens", "candidate_only"})


class SemanticMetadataViewError(ValueError):
    """A bounded shape, binding or exact reconstruction check failed."""


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(value) -> str:
    return canonical_context_json_bytes(value).decode("utf-8")


def _byte_count(value: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except (AttributeError, UnicodeEncodeError) as error:
        raise SemanticMetadataViewError("valid UTF-8 text required") from error


def _parse(text: str):
    if type(text) is not str or _byte_count(text) > MAX_BYTES:
        raise SemanticMetadataViewError("metadata input exceeds text bound")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SemanticMetadataViewError("duplicate metadata JSON key")
            result[key] = value
        return result
    def no_constant(_value):
        raise SemanticMetadataViewError("nonfinite metadata JSON number")
    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=no_constant)
    except SemanticMetadataViewError:
        raise
    except (ValueError, RecursionError, OverflowError) as error:
        raise SemanticMetadataViewError("invalid bounded metadata JSON") from error
    count = 0
    def check(item, depth):
        nonlocal count
        count += 1
        if depth > MAX_DEPTH or count > MAX_NODES:
            raise SemanticMetadataViewError("metadata JSON exceeds structure bound")
        if isinstance(item, float) and not math.isfinite(item):
            raise SemanticMetadataViewError("nonfinite metadata JSON number")
        if isinstance(item, dict):
            for child in item.values():
                check(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                check(child, depth + 1)
    check(value, 0)
    try:
        canonical = _json(value)
    except (TypeError, ValueError, RecursionError, UnicodeEncodeError) as error:
        raise SemanticMetadataViewError("invalid canonical metadata JSON") from error
    if canonical != text:
        raise SemanticMetadataViewError("canonical metadata JSON required")
    return value


def _text(value, name):
    if type(value) is not str or not value or _byte_count(value) > 4096:
        raise SemanticMetadataViewError("invalid metadata binding: " + name)
    return value


def _semantic_rows(wire):
    semantic = wire.get("translated_semantic")
    if not isinstance(semantic, dict):
        raise SemanticMetadataViewError("typed translated semantic metadata required")
    capsules, admissions = semantic.get("capsules"), semantic.get("admissions")
    if (not isinstance(capsules, list) or not isinstance(admissions, list)
            or len(capsules) != len(admissions) or len(capsules) > MAX_ROWS
            or any(not isinstance(row, dict) for row in capsules + admissions)
            or any(not isinstance(row.get("ref"), dict) for row in admissions)):
        raise SemanticMetadataViewError("bounded capsule and admission rows required")
    return semantic, {"capsules": capsules, "admissions": admissions,
                      "admission_refs": [row["ref"] for row in admissions]}


def _bindings(wire):
    semantic, _groups = _semantic_rows(wire)
    native = wire.get("native_context")
    if not isinstance(native, dict) or not isinstance(native.get("evidence"), list):
        raise SemanticMetadataViewError("native context core required")
    task = _text(semantic.get("task_id"), "task_id")
    if native.get("objective_id") != task:
        raise SemanticMetadataViewError("metadata task binding differs")
    if not isinstance(semantic.get("manifest"), dict) or not semantic["manifest"]:
        raise SemanticMetadataViewError("metadata source manifest required")
    if type(wire.get("native_suffix")) is not str:
        raise SemanticMetadataViewError("literal native suffix required")
    return {
        "translation_cid": _text(wire.get("translation_cid"), "translation_cid"),
        "task_id": task, "scope_cid": _text(semantic.get("scope_cid"), "scope_cid"),
        "semantic_root_cid": _text(semantic.get("semantic_root_cid"), "semantic_root_cid"),
        "source_manifest_sha256": _sha(_json(semantic["manifest"])),
        "native_core_sha256": _sha(_json({key: value for key, value in native.items() if key != "evidence"})),
        "native_context_sha256": _sha(_json(native)), "native_suffix_sha256": _sha(wire["native_suffix"]),
        "original_translated_semantic_sha256": _sha(_json(semantic)),
    }


def _transport(wire):
    if not isinstance(wire, dict) or set(wire) != _TRANSPORT_FIELDS or wire.get("schema") != TRANSPORT_SCHEMA:
        raise SemanticMetadataViewError("only the exact semantic transport @1 is supported")
    if (wire.get("execution_authority") is not False or wire.get("completion_authority") is not False
            or not isinstance(wire.get("translation_table"), dict) or type(wire.get("instructions")) is not str):
        raise SemanticMetadataViewError("native transport authority or dictionary differs")
    _bindings(wire)


def _binding_value(field, value, aliases):
    if field == "raw_source_required":
        if type(value) is not bool:
            raise SemanticMetadataViewError("raw source metadata must remain Boolean")
    elif field in {"source_cid", "semantic_state_root_cid"}:
        if isinstance(value, dict):
            if (set(value) != {"$semantic_ref"} or type(value["$semantic_ref"]) is not str
                    or value["$semantic_ref"] not in aliases):
                raise SemanticMetadataViewError("unknown native semantic identifier reference")
        else:
            _text(value, field)
    else:
        _text(value, field)


def _common(rows, fields, aliases):
    if len(rows) < 2:
        return {}
    common = {}
    for field in sorted(fields):
        if field not in rows[0]:
            continue
        original = rows[0][field]
        _binding_value(field, original, aliases)
        if all(field in row and _json(row[field]) == _json(original) for row in rows):
            common[field] = original
    return common


def _proxy(text: str) -> int:
    return (_byte_count(text) + 3) // 4


@dataclass(frozen=True)
class SemanticMetadataView:
    native_prompt: str
    provider_prompt: str
    receipt_json: str

    @property
    def receipt(self) -> dict:
        return _parse(self.receipt_json)


def project_semantic_metadata_view(provider_prompt: str) -> SemanticMetadataView:
    """Create a candidate view; the final complete-input gate selects it later.

    The input must already have passed the existing semantic transport owner.
    This pure codec neither performs that validation nor establishes freshness.
    """
    wire = _parse(provider_prompt)
    _transport(wire)
    bindings = _bindings(wire)
    _semantic, groups = _semantic_rows(wire)
    common = {name: _common(rows, GROUP_FIELDS[name], wire["translation_table"])
              for name, rows in groups.items()}
    for name, rows in groups.items():
        for row in rows:
            for field in common[name]:
                del row[field]
    wire.update(schema=VIEW_SCHEMA, source_transport_schema=TRANSPORT_SCHEMA,
                common_bindings=common, metadata_instructions=METADATA_INSTRUCTIONS)
    candidate = _json(wire)
    if _byte_count(candidate) > MAX_BYTES:
        raise SemanticMetadataViewError("metadata candidate exceeds text bound")
    receipt = {
        "schema": RECEIPT_SCHEMA, "view_schema": VIEW_SCHEMA, "source_transport_schema": TRANSPORT_SCHEMA,
        **bindings, "native_transport_sha256": _sha(provider_prompt), "native_transport_bytes": _byte_count(provider_prompt),
        "candidate_view_sha256": _sha(candidate), "candidate_view_bytes": _byte_count(candidate),
        "common_bindings_sha256": _sha(_json(common)),
        "shared_field_paths": [[name, field] for name in sorted(common) for field in sorted(common[name])],
        "capsule_count": len(groups["capsules"]), "admission_count": len(groups["admissions"]),
        "token_proxy": TOKEN_PROXY, "native_transport_proxy_tokens": _proxy(provider_prompt),
        "candidate_view_proxy_tokens": _proxy(candidate), "candidate_only": True,
        **{field: False for field in _FALSE_FIELDS},
    }
    result = SemanticMetadataView(provider_prompt, candidate, _json(receipt))
    if restore_semantic_metadata_view(provider_prompt=candidate, receipt=receipt) != provider_prompt:
        raise SemanticMetadataViewError("metadata transport did not round-trip exactly")
    return result


def _receipt(receipt):
    if not isinstance(receipt, dict):
        raise SemanticMetadataViewError("trusted metadata receipt required")
    try:
        receipt = _parse(_json(receipt))
    except (TypeError, ValueError, RecursionError, UnicodeEncodeError) as error:
        raise SemanticMetadataViewError("invalid bounded metadata receipt") from error
    if (set(receipt) != _RECEIPT_FIELDS or receipt.get("schema") != RECEIPT_SCHEMA
            or receipt.get("view_schema") != VIEW_SCHEMA or receipt.get("source_transport_schema") != TRANSPORT_SCHEMA
            or receipt.get("token_proxy") != TOKEN_PROXY or receipt.get("candidate_only") is not True
            or any(receipt.get(field) is not False for field in _FALSE_FIELDS)):
        raise SemanticMetadataViewError("metadata receipt schema or authority differs")
    for key, value in receipt.items():
        if key.endswith("sha256") and (type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None):
            raise SemanticMetadataViewError("metadata receipt digest invalid")
    for key in ("native_transport_bytes", "candidate_view_bytes", "native_transport_proxy_tokens",
                "candidate_view_proxy_tokens", "capsule_count", "admission_count"):
        if type(receipt[key]) is not int or not 0 <= receipt[key] <= MAX_BYTES:
            raise SemanticMetadataViewError("metadata receipt count invalid")
    return receipt


def restore_semantic_metadata_view(*, provider_prompt: str, receipt: dict) -> str:
    """Restore exact @1 bytes against an independently retained trusted receipt."""
    retained = _receipt(receipt)
    wire = _parse(provider_prompt)
    if (not isinstance(wire, dict) or set(wire) != _VIEW_FIELDS or wire.get("schema") != VIEW_SCHEMA
            or wire.get("source_transport_schema") != TRANSPORT_SCHEMA
            or wire.get("metadata_instructions") != METADATA_INSTRUCTIONS
            or not isinstance(wire.get("translation_table"), dict)
            or _sha(provider_prompt) != retained["candidate_view_sha256"]
            or _byte_count(provider_prompt) != retained["candidate_view_bytes"]
            or _proxy(provider_prompt) != retained["candidate_view_proxy_tokens"]):
        raise SemanticMetadataViewError("metadata candidate differs from retained receipt")
    common = wire["common_bindings"]
    if (not isinstance(common, dict) or set(common) != set(GROUP_FIELDS)
            or _sha(_json(common)) != retained["common_bindings_sha256"]):
        raise SemanticMetadataViewError("metadata common bindings differ")
    _semantic, groups = _semantic_rows(wire)
    for name, rows in groups.items():
        values = common[name]
        if not isinstance(values, dict) or not set(values) <= GROUP_FIELDS[name] or (values and len(rows) < 2):
            raise SemanticMetadataViewError("metadata shared field allowlist differs")
        for field, value in values.items():
            _binding_value(field, value, wire.get("translation_table", {}))
        for row in rows:
            if set(row) & set(values):
                raise SemanticMetadataViewError("metadata row overrides a shared binding")
            row.update(values)
    paths = [[name, field] for name in sorted(common) for field in sorted(common[name])]
    if (retained["shared_field_paths"] != paths or len(groups["capsules"]) != retained["capsule_count"]
            or len(groups["admissions"]) != retained["admission_count"]):
        raise SemanticMetadataViewError("metadata row or shared field population differs")
    for field in ("common_bindings", "metadata_instructions", "source_transport_schema"):
        del wire[field]
    wire["schema"] = TRANSPORT_SCHEMA
    _transport(wire)
    if any(_bindings(wire)[key] != retained[key] for key in _BINDING_FIELDS):
        raise SemanticMetadataViewError("metadata native source/task/root/core binding differs")
    native = _json(wire)
    if (_sha(native) != retained["native_transport_sha256"] or _byte_count(native) != retained["native_transport_bytes"]
            or _proxy(native) != retained["native_transport_proxy_tokens"]):
        raise SemanticMetadataViewError("metadata restored transport differs")
    return native


@dataclass(frozen=True)
class SemanticMetadataSelection:
    selected_prompt: str
    selected_mode: str
    receipt_json: str

    @property
    def receipt(self) -> dict:
        return _parse(self.receipt_json)


def select_semantic_metadata_view(*, view: SemanticMetadataView, native_complete_prompt: str,
                                  candidate_complete_prompt: str) -> SemanticMetadataSelection:
    """Select only a smaller complete input under an explicitly named proxy."""
    if type(view) is not SemanticMetadataView:
        raise SemanticMetadataViewError("native metadata candidate required")
    if restore_semantic_metadata_view(provider_prompt=view.provider_prompt, receipt=view.receipt) != view.native_prompt:
        raise SemanticMetadataViewError("metadata candidate identity differs")
    for text in (native_complete_prompt, candidate_complete_prompt):
        if type(text) is not str or _byte_count(text) > MAX_BYTES:
            raise SemanticMetadataViewError("complete metadata input exceeds bound")
    if native_complete_prompt.count(view.native_prompt) != 1:
        raise SemanticMetadataViewError("complete native transport occurrence differs")
    prefix, suffix = native_complete_prompt.split(view.native_prompt)
    if (candidate_complete_prompt != prefix + view.provider_prompt + suffix
            or candidate_complete_prompt.count(view.provider_prompt) != 1):
        raise SemanticMetadataViewError("complete metadata input changed a prefix or suffix")
    selected = (_byte_count(candidate_complete_prompt) < _byte_count(native_complete_prompt)
                and _proxy(candidate_complete_prompt) < _proxy(native_complete_prompt))
    chosen = candidate_complete_prompt if selected else native_complete_prompt
    receipt = {
        "schema": SELECTION_SCHEMA, "view_receipt_sha256": _sha(view.receipt_json),
        "selected_mode": "common-bindings@1" if selected else "legacy",
        "fallback_reason": None if selected else "complete_input_not_smaller_under_bytes_and_proxy",
        "token_proxy": TOKEN_PROXY, "candidate_only": True,
        "native_complete_sha256": _sha(native_complete_prompt), "native_complete_bytes": _byte_count(native_complete_prompt),
        "native_complete_proxy_tokens": _proxy(native_complete_prompt),
        "candidate_complete_sha256": _sha(candidate_complete_prompt), "candidate_complete_bytes": _byte_count(candidate_complete_prompt),
        "candidate_complete_proxy_tokens": _proxy(candidate_complete_prompt),
        "selected_complete_sha256": _sha(chosen), "selected_complete_bytes": _byte_count(chosen),
        "selected_complete_proxy_tokens": _proxy(chosen),
        **{field: False for field in _FALSE_FIELDS},
    }
    return SemanticMetadataSelection(chosen, receipt["selected_mode"], _json(receipt))
