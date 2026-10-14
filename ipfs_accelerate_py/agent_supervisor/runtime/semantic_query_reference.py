"""Self-contained, exact query references over upstream-verified semantic input.

Only worker_projection.query may change in this provider presentation. Its exact
literal value remains in one required inline raw source. Native producer bytes,
source identities, admission, proof and authority remain owned by existing
validators. This pure codec performs no filesystem/network access and confers no
freshness. The independently retained receipt binds exact original wire bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

TRANSPORT_SCHEMA = "supervisor-semantic-router-input@1"
COMPACT_TRANSPORT_SCHEMA = "supervisor-semantic-router-input@2"
METADATA_SCHEMA = "supervisor-semantic-metadata-view@1"
VIEW_SCHEMA = "supervisor-semantic-query-reference@1"
RECEIPT_SCHEMA = "supervisor-semantic-query-reference-receipt@1"
CODEC_VERSION = "required-inline-query-reference@1"
REF_KEY = "$required_raw_source"
SLOT = ("translated_semantic", "worker_projection", "query")
MAX_BYTES = 2_000_000
MAX_QUERY_BYTES = 32768
MAX_RECEIPT_BYTES = 16384
MAX_DEPTH = 44
MAX_NODES = 100000
MAX_SOURCES = 64
INSTRUCTIONS = (
    "Only worker_projection.query uses '$required_raw_source': read its exact string "
    "from transport.translated_semantic.raw_sources at that path. The required source "
    "stays literal and unchanged. This reference grants no proof, source omission or authority."
)
_BASE_FIELDS = {"schema", "translation_cid", "translation_table", "native_context", "translated_semantic",
                "native_suffix", "instructions", "execution_authority", "completion_authority"}
_METADATA_FIELDS = _BASE_FIELDS | {"source_transport_schema", "common_bindings", "metadata_instructions"}
_COMPACT_FIELDS = (_BASE_FIELDS - {"translation_table"}) | {"task_id", "scope_cid", "semantic_root_cid"}
_VIEW_FIELDS = {"schema", "transport", "instructions", "execution_authority", "completion_authority"}
_FALSE_FIELDS = {"source_freshness_verified", "program_semantics_proved", "omission_authority", "proof_authority",
                 "execution_authority", "completion_authority", "publication_authority"}
_BINDINGS = {"translation_cid", "task_id", "scope_cid", "semantic_root_cid", "source_manifest_sha256",
             "native_context_sha256", "native_suffix_sha256", "translated_semantic_sha256"}
_RECEIPT_FIELDS = _BINDINGS | _FALSE_FIELDS | {
    "schema", "view_schema", "source_transport_schema", "codec_version", "codec_sha256", "slot",
    "reference_count", "source_path", "source_sha256", "source_cid", "query_sha256", "query_bytes",
    "query_json_bytes", "native_transport_sha256", "native_transport_bytes", "candidate_view_sha256",
    "candidate_view_bytes", "byte_savings", "candidate_only"}


class SemanticQueryReferenceError(ValueError):
    """Closed content-free diagnostic, including explicit experiment ineligibility."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code):
    raise SemanticQueryReferenceError(code)


def _json(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail("invalid_json_value")


def _bytes(text):
    if type(text) is not str:
        _fail("invalid_text")
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        _fail("invalid_utf8")


def _sha(text):
    return hashlib.sha256(_bytes(text)).hexdigest()


# This is a digest of the public codec contract, not a file/source-code hash.
# Campaign source seals own implementation-file identity independently.
CODEC_SHA256 = _sha(_json({"version": CODEC_VERSION, "view_schema": VIEW_SCHEMA,
    "reference_key": REF_KEY, "allowed_slot": list(SLOT), "instructions": INSTRUCTIONS,
    "source_schemas": [TRANSPORT_SCHEMA, COMPACT_TRANSPORT_SCHEMA, METADATA_SCHEMA],
    "matching": "one required inline source; exact UTF-8 equality; manifest SHA256",
    "restore": "resolve only allowed slot; exact independently bound original transport"}))


def _parse(text, *, maximum=MAX_BYTES, canonical=True):
    if len(_bytes(text)) > maximum:
        _fail("input_bytes_exceeded")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("duplicate_json_key")
            result[key] = value
        return result
    def invalid_constant(_value):
        _fail("nonfinite_json_number")
    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)
    except SemanticQueryReferenceError:
        raise
    except (ValueError, RecursionError, OverflowError):
        _fail("invalid_json")
    count = 0
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        count += 1
        if depth > MAX_DEPTH or count > MAX_NODES:
            _fail("json_structure_exceeded")
        if type(item) is float and not math.isfinite(item):
            _fail("nonfinite_json_number")
        if type(item) is dict:
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in item)
    if canonical and _json(value) != text:
        _fail("noncanonical_json")
    return value


def _text(value, *, maximum=512):
    if type(value) is not str or not 1 <= len(_bytes(value)) <= maximum:
        _fail("invalid_identity")
    return value


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        _fail("invalid_digest")
    return value


def _path(value):
    value = _text(value, maximum=1024)
    if (value.startswith("/") or "\\" in value or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        _fail("invalid_source_path")
    return value


def _markers(value):
    found = []
    stack = [(value, ())]
    while stack:
        item, path = stack.pop()
        if type(item) is dict:
            if REF_KEY in item:
                found.append(path)
            stack.extend((child, path + (key,)) for key, child in item.items())
        elif type(item) is list:
            stack.extend((child, path + (index,)) for index, child in enumerate(item))
    return found


def _transport(wire, *, referenced=False):
    if type(wire) is not dict:
        _fail("unsupported_transport")
    schema = wire.get("schema")
    if type(schema) is not str or schema not in {TRANSPORT_SCHEMA, COMPACT_TRANSPORT_SCHEMA, METADATA_SCHEMA}:
        _fail("unsupported_transport")
    expected = _BASE_FIELDS if schema == TRANSPORT_SCHEMA else _COMPACT_FIELDS if schema == COMPACT_TRANSPORT_SCHEMA else _METADATA_FIELDS
    if (set(wire) != expected or wire.get("execution_authority") is not False
            or wire.get("completion_authority") is not False or type(wire.get("instructions")) is not str
            or type(wire.get("native_suffix")) is not str):
        _fail("invalid_transport")
    if schema != COMPACT_TRANSPORT_SCHEMA and type(wire.get("translation_table")) is not dict:
        _fail("invalid_translation_table")
    if schema == METADATA_SCHEMA and (wire.get("source_transport_schema") != TRANSPORT_SCHEMA
            or type(wire.get("common_bindings")) is not dict
            or set(wire["common_bindings"]) != {"capsules", "admissions", "admission_refs"}
            or any(type(row) is not dict for row in wire["common_bindings"].values())
            or type(wire.get("metadata_instructions")) is not str):
        _fail("invalid_metadata_transport")
    semantic, context = wire.get("translated_semantic"), wire.get("native_context")
    if (type(semantic) is not dict or type(context) is not dict or type(context.get("evidence")) is not list
            or semantic.get("schema") not in ("supervisor-semantic-worker-context@1", "supervisor-semantic-worker-context@2")
            or semantic.get("completion_authority") is not False):
        _fail("invalid_semantic_payload")
    task, scope, root = (_text(semantic.get(key)) for key in ("task_id", "scope_cid", "semantic_root_cid"))
    if context.get("objective_id") != task:
        _fail("task_binding_mismatch")
    if schema == COMPACT_TRANSPORT_SCHEMA and any(wire.get(key) != semantic[key] for key in ("task_id", "scope_cid", "semantic_root_cid")):
        _fail("compact_binding_mismatch")
    manifest, raw, required = (semantic.get(key) for key in ("manifest", "raw_sources", "required_raw_paths"))
    if (type(manifest) is not dict or not 1 <= len(manifest) <= MAX_SOURCES
            or type(raw) is not dict or not set(raw) <= set(manifest)
            or type(required) is not list or not 1 <= len(required) <= MAX_SOURCES
            or any(type(path) is not str for path in required)
            or len(set(required)) != len(required) or not set(required) <= set(raw)):
        _fail("invalid_source_population")
    for path, binding in manifest.items():
        _path(path)
        if type(binding) is not dict or set(binding) != {"sha256", "source_cid"}:
            _fail("invalid_source_binding")
        _digest(binding["sha256"])
        _text(binding["source_cid"])
    for path, text in raw.items():
        if _sha(text) != manifest[path]["sha256"]:
            _fail("inline_source_hash_mismatch")
    projection = semantic.get("worker_projection")
    if type(projection) is not dict or projection.get("schema") != "supervisor-semantic-worker-projection@1":
        _fail("unsupported_worker_projection")
    query = projection.get("query")
    if not referenced and (type(query) is not str or not 1 <= len(_bytes(query)) <= MAX_QUERY_BYTES):
        _fail("invalid_query")
    if referenced and (type(query) is not dict or set(query) != {REF_KEY} or type(query.get(REF_KEY)) is not str):
        _fail("invalid_query_reference")
    return semantic, {"translation_cid": _text(wire.get("translation_cid")), "task_id": task,
        "scope_cid": scope, "semantic_root_cid": root, "source_manifest_sha256": _sha(_json(manifest)),
        "native_context_sha256": _sha(_json(context)), "native_suffix_sha256": _sha(wire["native_suffix"]),
        "translated_semantic_sha256": _sha(_json(semantic))}


@dataclass(frozen=True)
class SemanticQueryReferenceView:
    native_prompt: str
    provider_prompt: str
    receipt_json: str

    @property
    def receipt(self):
        return json.loads(self.receipt_json)


def validate_semantic_query_reference_receipt(receipt: dict) -> dict:
    if type(receipt) is not dict:
        _fail("invalid_receipt")
    retained = _parse(_json(receipt), maximum=MAX_RECEIPT_BYTES)
    if (set(retained) != _RECEIPT_FIELDS or retained.get("schema") != RECEIPT_SCHEMA
            or retained.get("view_schema") != VIEW_SCHEMA or retained.get("codec_version") != CODEC_VERSION
            or retained.get("codec_sha256") != CODEC_SHA256
            or type(retained.get("source_transport_schema")) is not str
            or retained["source_transport_schema"] not in {TRANSPORT_SCHEMA, COMPACT_TRANSPORT_SCHEMA, METADATA_SCHEMA}
            or retained.get("slot") != list(SLOT) or type(retained.get("reference_count")) is not int
            or retained["reference_count"] != 1 or retained.get("candidate_only") is not True
            or any(retained.get(key) is not False for key in _FALSE_FIELDS)):
        _fail("invalid_receipt")
    for key in retained:
        if key.endswith("sha256"):
            _digest(retained[key])
    for key in ("translation_cid", "task_id", "scope_cid", "semantic_root_cid", "source_cid"):
        _text(retained[key])
    _path(retained["source_path"])
    for key, maximum in (("native_transport_bytes", MAX_BYTES), ("candidate_view_bytes", MAX_BYTES),
                         ("byte_savings", MAX_BYTES), ("query_bytes", MAX_QUERY_BYTES),
                         ("query_json_bytes", MAX_BYTES)):
        if type(retained[key]) is not int or not 1 <= retained[key] <= maximum:
            _fail("invalid_receipt_count")
    if (retained["native_transport_bytes"] - retained["candidate_view_bytes"] != retained["byte_savings"]
            or retained["query_sha256"] != retained["source_sha256"]):
        _fail("invalid_receipt_accounting")
    return retained


def project_semantic_query_reference(provider_prompt: str) -> SemanticQueryReferenceView:
    """Project a verified semantic wire; reject absent, ambiguous or larger views."""
    wire = _parse(provider_prompt)
    semantic, bindings = _transport(wire)
    if _markers(wire):
        _fail("reserved_marker_collision")
    query = semantic["worker_projection"]["query"]
    matches = [path for path in semantic["required_raw_paths"] if semantic["raw_sources"][path] == query]
    if not matches:
        _fail("no_exact_required_source")
    if len(matches) != 1:
        _fail("ambiguous_required_source")
    source = matches[0]
    semantic["worker_projection"]["query"] = {REF_KEY: source}
    candidate = _json({"schema": VIEW_SCHEMA, "transport": wire, "instructions": INSTRUCTIONS,
                       "execution_authority": False, "completion_authority": False})
    original_size, candidate_size = len(_bytes(provider_prompt)), len(_bytes(candidate))
    if candidate_size >= original_size:
        _fail("unprofitable_query_reference")
    if candidate_size > MAX_BYTES:
        _fail("input_bytes_exceeded")
    receipt = {"schema": RECEIPT_SCHEMA, "view_schema": VIEW_SCHEMA, "source_transport_schema": wire["schema"],
        "codec_version": CODEC_VERSION, "codec_sha256": CODEC_SHA256, "slot": list(SLOT), "reference_count": 1,
        **bindings, "source_path": source, "source_sha256": semantic["manifest"][source]["sha256"],
        "source_cid": semantic["manifest"][source]["source_cid"], "query_sha256": _sha(query),
        "query_bytes": len(_bytes(query)), "query_json_bytes": len(_bytes(_json(query))),
        "native_transport_sha256": _sha(provider_prompt), "native_transport_bytes": original_size,
        "candidate_view_sha256": _sha(candidate), "candidate_view_bytes": candidate_size,
        "byte_savings": original_size - candidate_size, "candidate_only": True,
        **{key: False for key in _FALSE_FIELDS}}
    validate_semantic_query_reference_receipt(receipt)
    result = SemanticQueryReferenceView(provider_prompt, candidate, _json(receipt))
    if restore_semantic_query_reference(provider_prompt=candidate, receipt=receipt) != provider_prompt:
        _fail("round_trip_mismatch")
    return result


def restore_semantic_query_reference(*, provider_prompt: str, receipt: dict) -> str:
    """Resolve from still-inline required source; verify exact original bytes."""
    retained = validate_semantic_query_reference_receipt(receipt)
    candidate = _parse(provider_prompt)
    if (type(candidate) is not dict or set(candidate) != _VIEW_FIELDS or candidate.get("schema") != VIEW_SCHEMA
            or candidate.get("instructions") != INSTRUCTIONS or candidate.get("execution_authority") is not False
            or candidate.get("completion_authority") is not False
            or _sha(provider_prompt) != retained["candidate_view_sha256"]
            or len(_bytes(provider_prompt)) != retained["candidate_view_bytes"]):
        _fail("candidate_receipt_mismatch")
    wire = candidate["transport"]
    semantic, _ = _transport(wire, referenced=True)
    if _markers(wire) != [SLOT]:
        _fail("invalid_reference_locations")
    path = semantic["worker_projection"]["query"][REF_KEY]
    if (path != retained["source_path"] or path not in semantic["required_raw_paths"]
            or path not in semantic["raw_sources"] or wire["schema"] != retained["source_transport_schema"]
            or semantic["manifest"][path] != {"sha256": retained["source_sha256"], "source_cid": retained["source_cid"]}):
        _fail("reference_binding_mismatch")
    query = semantic["raw_sources"][path]
    if (_sha(query) != retained["query_sha256"] or len(_bytes(query)) != retained["query_bytes"]
            or len(_bytes(_json(query))) != retained["query_json_bytes"]
            or sum(semantic["raw_sources"][name] == query for name in semantic["required_raw_paths"]) != 1):
        _fail("reference_source_mismatch")
    semantic["worker_projection"]["query"] = query
    _, bindings = _transport(wire)
    if bindings != {key: retained[key] for key in _BINDINGS}:
        _fail("restored_binding_mismatch")
    original = _json(wire)
    if _sha(original) != retained["native_transport_sha256"] or len(_bytes(original)) != retained["native_transport_bytes"]:
        _fail("restored_transport_mismatch")
    return original
