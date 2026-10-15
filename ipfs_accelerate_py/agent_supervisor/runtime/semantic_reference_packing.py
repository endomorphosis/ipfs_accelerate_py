"""Lossless column packing at two closed semantic reference-list locations.

This pure representation boundary does not validate producer admission, load
source, or grant substitution, execution, completion, or proof authority.
"""
from __future__ import annotations

import json

from ..context.context_contracts import (
    CONTEXT_CONTRACT_VERSION,
    CONTEXT_REFERENCE_SCHEMA,
    canonical_context_json_bytes,
)

PACKING_REVISION = "context-reference-columns@1"
PACKED_SCHEMA = "supervisor-context-reference-columns@1"
MAX_BYTES = 2_000_000
MAX_NODES = 100_000
MAX_DEPTH = 40
MAX_ROWS = 4096
REFERENCE_FIELDS = frozenset({
    "schema", "contract_version", "reference_id", "kind", "tier",
    "referenced_content_id", "repository_id", "tree_id", "path", "summary",
    "byte_count", "token_count", "metadata",
})
_PATHS = {
    "supervisor-semantic-worker-context@1": ("pack", "references"),
    "supervisor-semantic-worker-role-view@1": ("required_context", "pack", "references"),
}


class SemanticReferencePackingError(ValueError):
    """The closed representation or its bounded expansion is invalid."""


def _wire(value) -> bytes:
    try:
        return canonical_context_json_bytes(value)
    except (ValueError, TypeError, RecursionError, UnicodeError) as error:
        raise SemanticReferencePackingError("literal canonical JSON required") from error


def _profile(value, *, omitted_path=None) -> tuple[int, int, int]:
    """Bound literal input before serialization/copying; never interpret markers."""
    pending, nodes, text_bytes, maximum = [(value, 0, ())], 0, 0, 0
    while pending:
        item, depth, path = pending.pop()
        if omitted_path is not None and path == omitted_path:
            continue
        nodes += 1
        maximum = max(maximum, depth)
        if nodes > MAX_NODES or depth > MAX_DEPTH:
            raise SemanticReferencePackingError("reference packing structure exceeds bounds")
        if type(item) is dict:
            omitted_child = (omitted_path is not None and path == omitted_path[:-1]
                             and omitted_path[-1] in item)
            if len(item) - int(omitted_child) > MAX_NODES - nodes - len(pending):
                raise SemanticReferencePackingError("reference packing structure exceeds bounds")
            for key, child in item.items():
                if type(key) is not str:
                    raise SemanticReferencePackingError("literal JSON object keys must be strings")
                text_bytes += len(key.encode())
                child_path = (*path, key)
                if child_path != omitted_path:
                    pending.append((child, depth + 1, child_path))
        elif type(item) is list:
            if len(item) > MAX_NODES - nodes - len(pending):
                raise SemanticReferencePackingError("reference packing structure exceeds bounds")
            pending.extend((child, depth + 1, (*path, index)) for index, child in enumerate(item))
        elif type(item) is str:
            text_bytes += len(item.encode())
        elif item is not None and type(item) not in {bool, int}:
            raise SemanticReferencePackingError("literal canonical JSON values required")
        if text_bytes > MAX_BYTES:
            raise SemanticReferencePackingError("reference packing text exceeds byte bound")
    return nodes, text_bytes, maximum


def _bounded_wire(payload) -> bytes:
    try:
        _profile(payload)
        raw = _wire(payload)
    except UnicodeError as error:
        raise SemanticReferencePackingError("literal UTF-8 values required") from error
    if len(raw) > MAX_BYTES:
        raise SemanticReferencePackingError("reference packing input exceeds byte bound")
    return raw


def _location(payload):
    if type(payload) is not dict:
        return None, None
    path = _PATHS.get(payload.get("schema")) if type(payload.get("schema")) is str else None
    if path is None:
        return None, None
    item = payload
    for key in path:
        if type(item) is not dict or key not in item:
            return None, None
        item = item[key]
    return path, item


def _replace(payload, path, value):
    parent = payload
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value


def _row_preflight(references) -> None:
    """Apply the targeted population ceiling before canonical normalization."""
    rows = references.get("rows") if type(references) is dict else references
    if type(rows) is list and len(rows) > MAX_ROWS:
        raise SemanticReferencePackingError("reference row count exceeds bounds")


def _known_reference(row) -> bool:
    return (type(row) is dict and set(row) == REFERENCE_FIELDS
            and row["schema"] == CONTEXT_REFERENCE_SCHEMA
            and type(row["contract_version"]) is int
            and row["contract_version"] == CONTEXT_CONTRACT_VERSION)


def _descriptor(references: list) -> dict:
    shared = {}
    for key in sorted(REFERENCE_FIELDS):
        first = _wire(references[0][key])
        if all(_wire(row[key]) == first for row in references[1:]):
            shared[key] = references[0][key]
    columns = sorted(REFERENCE_FIELDS - shared.keys())
    return {"schema": PACKED_SCHEMA, "shared": shared, "columns": columns,
            "rows": [[row[key] for key in columns] for row in references]}


def _unpack_descriptor(descriptor, *, payload, path, input_raw: bytes) -> list:
    if (type(descriptor) is not dict or set(descriptor) != {"schema", "shared", "columns", "rows"}
            or descriptor["schema"] != PACKED_SCHEMA):
        raise SemanticReferencePackingError("unsupported reference column wrapper")
    shared, columns, rows = descriptor["shared"], descriptor["columns"], descriptor["rows"]
    if type(shared) is not dict or type(columns) is not list or type(rows) is not list:
        raise SemanticReferencePackingError("invalid reference column wrapper types")
    if (any(type(key) is not str for key in columns) or columns != sorted(set(columns))
            or set(shared) & set(columns) or set(shared) | set(columns) != REFERENCE_FIELDS):
        raise SemanticReferencePackingError("reference fields must be a disjoint complete ordered partition")
    if not 2 <= len(rows) <= MAX_ROWS:
        raise SemanticReferencePackingError("reference column row count exceeds bounds")
    if any(type(row) is not list or len(row) != len(columns) for row in rows):
        raise SemanticReferencePackingError("reference column row width differs")

    # Compute the expanded wire and structure without constructing expanded
    # dictionaries or repeating shared metadata into a copied payload.
    outside_nodes, outside_text, outside_depth = _profile(payload, omitted_path=path)
    nodes, text_bytes, relative_depth, expanded_bytes = 1, 0, 1, 2 + len(rows) - 1
    descriptor_raw = _wire(descriptor)
    outside_bytes = len(input_raw) - len(descriptor_raw)
    key_wire_lengths = {key: len(_wire(key)) for key in REFERENCE_FIELDS}
    shared_metrics = {key: (_profile(value), len(_wire(value))) for key, value in shared.items()}
    column_indices = {key: index for index, key in enumerate(columns)}
    for row in rows:
        nodes += 1
        expanded_bytes += 2 + len(REFERENCE_FIELDS) - 1
        for key in REFERENCE_FIELDS:
            value = shared[key] if key in shared else row[column_indices[key]]
            profile, byte_count = shared_metrics[key] if key in shared else (_profile(value), len(_wire(value)))
            child_nodes, child_text, child_depth = profile
            nodes += child_nodes
            text_bytes += len(key.encode()) + child_text
            relative_depth = max(relative_depth, child_depth + 2)
            expanded_bytes += key_wire_lengths[key] + 1 + byte_count
            if (outside_nodes + nodes > MAX_NODES or outside_text + text_bytes > MAX_BYTES
                    or max(outside_depth, len(path) + relative_depth) > MAX_DEPTH):
                raise SemanticReferencePackingError("expanded reference structure exceeds bounds")
        schema = shared.get("schema") if "schema" in shared else row[column_indices["schema"]]
        version = shared.get("contract_version") if "contract_version" in shared else row[column_indices["contract_version"]]
        if schema != CONTEXT_REFERENCE_SCHEMA or type(version) is not int or version != CONTEXT_CONTRACT_VERSION:
            raise SemanticReferencePackingError("unsupported expanded reference schema or version")
        if outside_bytes + expanded_bytes > MAX_BYTES:
            raise SemanticReferencePackingError("expanded reference payload exceeds byte bound")
    references = [{**shared, **dict(zip(columns, row, strict=True))} for row in rows]
    expected = _descriptor(references)
    if _wire(expected) != descriptor_raw or len(descriptor_raw) >= expanded_bytes:
        raise SemanticReferencePackingError("noncanonical or nonshrinking reference column encoding")
    return references


def pack_semantic_references(payload) -> tuple[object, dict]:
    """Copy and pack only a homogeneous supported reference list when smaller."""
    path, references = _location(payload)
    _row_preflight(references)
    raw = _bounded_wire(payload)
    stats = {"packing_revision": PACKING_REVISION, "packed": False,
             "reason": "not_packed",
             "reference_path": list(path) if path is not None else None,
             "row_count": len(references) if type(references) is list else 0,
             "shared_field_count": 0, "reference_bytes_before": None,
             "reference_bytes_after": None, "bytes_saved": 0}
    descriptor = None
    if type(references) is dict:
        restored = _unpack_descriptor(references, payload=payload, path=path, input_raw=raw)
        stats.update(reason="already_packed", row_count=len(restored))
    elif type(references) is list:
        if len(references) > MAX_ROWS:
            raise SemanticReferencePackingError("reference row count exceeds bounds")
        before = len(_wire(references))
        stats.update(reference_bytes_before=before, reference_bytes_after=before)
        if len(references) >= 2 and all(_known_reference(row) for row in references):
            candidate = _descriptor(references)
            after = len(_wire(candidate))
            if after < before:
                descriptor = candidate
                stats.update(packed=True, reason="canonical_bytes_smaller",
                             shared_field_count=len(candidate["shared"]),
                             reference_bytes_after=after, bytes_saved=before - after)
    copied = json.loads(raw)
    if descriptor is not None:
        _replace(copied, path, descriptor)
        try:
            compact_raw = _bounded_wire(copied)
        except SemanticReferencePackingError:
            copied = json.loads(raw)
            stats.update(packed=False, reason="candidate_bounds_exceeded", shared_field_count=0,
                         reference_bytes_after=stats["reference_bytes_before"], bytes_saved=0)
        else:
            copied = json.loads(compact_raw)
    return copied, stats


def unpack_semantic_references(payload):
    """Copy and strictly expand a descriptor only at the closed reference path."""
    path, references = _location(payload)
    _row_preflight(references)
    raw = _bounded_wire(payload)
    expanded = None
    if type(references) is dict:
        expanded = _unpack_descriptor(references, payload=payload, path=path, input_raw=raw)
    elif type(references) is list and len(references) > MAX_ROWS:
        raise SemanticReferencePackingError("reference row count exceeds bounds")
    copied = json.loads(raw)
    if expanded is not None:
        _replace(copied, path, expanded)
        copied = json.loads(_bounded_wire(copied))
    return copied
