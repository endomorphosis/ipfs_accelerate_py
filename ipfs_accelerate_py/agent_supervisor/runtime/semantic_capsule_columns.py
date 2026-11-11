"""Lossless typed columns for selected native capsules and admission records.

All values remain inline. Only three named row populations change presentation;
code, signatures, facts, constraints and unknown fields are never interpreted or
summarized. The upstream semantic owner retains producer validation, source
freshness and dispatch authority. Exact inverse reconstruction is not a proof of
program correctness, token savings, model comprehension or provider ingestion.
"""
from __future__ import annotations

import json
import math
import re

from . import semantic_metadata_view as metadata

MODE = "capsule-columns@1"
VIEW_SCHEMA = "supervisor-semantic-capsule-columns-view@1"
RECEIPT_SCHEMA = "supervisor-semantic-capsule-columns-receipt@1"
TABLE_SCHEMA = "supervisor-semantic-capsule-columns-table@1"
TRANSPORT_SCHEMA = metadata.TRANSPORT_SCHEMA
MAX_BYTES = metadata.MAX_BYTES
MAX_ROWS = metadata.MAX_ROWS
MAX_DEPTH = metadata.MAX_DEPTH
MAX_NODES = metadata.MAX_NODES
MAX_RECEIPT_BYTES = 128 * 1024
INSTRUCTIONS = (
    "Capsules and admissions use typed column tables. For each row, combine shared "
    "fields, values in columns order, and that row's extras; fields never overlap. "
    "Admission refs are reconstructed from the same row of refs. Extras preserve "
    "unknown or nonuniformly present fields; absent fields are not null or empty. "
    "Every literal value remains inline, including signatures, source requirements, "
    "constraints, caveats and obligations. No proof, omission or authority is added."
)
CAPSULE_FIELDS = frozenset({
    "schema", "stable_symbol_id", "version_cid", "semantic_index_schema", "extractor_version",
    "capsule_schema", "capsule_compiler_version", "source_slice_path", "source_cid",
    "symbol_fact_cid", "signature", "annotations", "defaults", "decorators", "contracts",
    "effects", "exception_behavior", "schema_relations", "serialization_relations", "test_refs",
    "fixture_refs", "proof_obligation_refs", "dependency_stable_ids", "dependency_version_cids",
    "dependency_fact_cids", "dependency_link_ids", "confidence", "relevant_binding_projection_cid",
    "docstring_hint", "metadata", "capsule_cid",
})
ADMISSION_FIELDS = frozenset({"schema", "interface", "admission", "freshness", "caveats", "assessment_cid"})
ADMISSION_REF_FIELDS = frozenset({"capsule_cid", "semantic_state_root_cid", "stable_symbol_id",
    "version_cid", "source_cid", "confidence", "validity_bindings", "raw_source_required"})
GROUP_FIELDS = dict(capsules=CAPSULE_FIELDS, admissions=ADMISSION_FIELDS, admission_refs=ADMISSION_REF_FIELDS)
_TABLE_FIELDS = frozenset({"schema", "kind", "shared", "columns", "rows", "extras"})
_VIEW_FIELDS = metadata._TRANSPORT_FIELDS | {"source_transport_schema", "capsule_columns_instructions"}
_STAT_FIELDS = frozenset({"rows", "shared_fields", "columns", "extra_fields", "original_bytes", "packed_bytes"})
_RECEIPT_FIELDS = metadata._BINDING_FIELDS | metadata._FALSE_FIELDS | {
    "schema", "view_schema", "source_transport_schema", "requested_mode", "native_transport_sha256",
    "native_transport_bytes", "candidate_view_sha256", "candidate_view_bytes", "columns_sha256",
    "group_counts", "capsule_count", "admission_count", "token_proxy", "native_transport_proxy_tokens",
    "candidate_view_proxy_tokens", "candidate_only",
}
CapsuleColumnsError = metadata.SemanticMetadataViewError


def _need(condition, reason):
    if not condition:
        raise CapsuleColumnsError(reason)


def _string_wire_bytes(value):
    """Count exact ensure_ascii=False JSON bytes before allocating escapes."""
    _need(type(value) is str and len(value) <= MAX_BYTES, "column_string_bound")
    total = 2
    for char in value:
        point = ord(char)
        _need(not 0xD800 <= point <= 0xDFFF, "column_invalid_unicode")
        total += (2 if char in '"\\\b\f\n\r\t' else 6 if point < 32 else
                  1 if point < 128 else 2 if point < 2048 else 3 if point < 65536 else 4)
        _need(total <= MAX_BYTES, "column_byte_bound")
    return total


def _profile(value):
    """Measure bounded expanded JSON without copying or serializing it."""
    pending = [(value, 0)]
    nodes = size = deepest = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        deepest = max(deepest, depth)
        _need(nodes <= MAX_NODES and depth <= MAX_DEPTH, "column_structure_bound")
        if type(item) is dict:
            _need(len(item) * 2 + len(pending) <= MAX_NODES - nodes,
                  "column_structure_bound")
            _need(all(type(key) is str for key in item), "column_non_json_value")
            size += 2 + max(0, len(item) - 1) + len(item)
            for key, child in item.items():
                pending.extend(((key, depth + 1), (child, depth + 1)))
        elif type(item) is list:
            _need(len(item) + len(pending) <= MAX_NODES - nodes, "column_structure_bound")
            size += 2 + max(0, len(item) - 1)
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            size += _string_wire_bytes(item)
        elif item is None:
            size += 4
        elif type(item) is bool:
            size += 4 if item else 5
        elif type(item) is int:
            _need(item.bit_length() <= 4096, "column_integer_bound")
            size += len(str(item))
        elif type(item) is float:
            _need(math.isfinite(item), "column_nonfinite_value")
            size += len(json.dumps(item, allow_nan=False))
        else:
            _need(False, "column_non_json_value")
        _need(size <= MAX_BYTES, "column_byte_bound")
    return nodes, size, deepest


def _json(value):
    _, predicted, _ = _profile(value)
    actual = metadata._json(value)
    _need(len(actual.encode("utf-8")) == predicted, "column_canonical_measurement_mismatch")
    return actual


def _parse(text):
    """Lexically cap nesting/node starts before materializing a JSON tree."""
    _need(type(text) is str and len(text) <= MAX_BYTES, "column_input_bound")
    try:
        raw = text.encode("utf-8")
    except UnicodeError as error:
        raise CapsuleColumnsError("column_invalid_unicode") from error
    _need(len(raw) <= MAX_BYTES, "column_input_bound")
    stack = []
    quoted = escaped = literal = False
    nodes = 0
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
            quoted = True
            literal = False
            nodes += 1
        elif char in "[{":
            nodes += 1
            stack.append(char)
            _need(len(stack) <= MAX_DEPTH, "column_structure_bound")
            literal = False
        elif char in "]}":
            _need(bool(stack) and stack.pop() == ("[" if char == "]" else "{"), "column_invalid_json")
            literal = False
        elif char in ",:" or char.isspace():
            literal = False
        elif not literal:
            nodes += 1
            literal = True
        _need(nodes <= MAX_NODES, "column_structure_bound")
    _need(not quoted and not escaped and not stack, "column_invalid_json")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result, "column_duplicate_key")
            result[key] = value
        return result

    def nonfinite(_value):
        _need(False, "column_nonfinite_value")

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)
    except (ValueError, TypeError, RecursionError) as error:
        if isinstance(error, CapsuleColumnsError):
            raise
        raise CapsuleColumnsError("column_invalid_json") from error
    _need(_json(value) == text, "column_noncanonical_json")
    return value


def _pack_group(rows, kind):
    known = GROUP_FIELDS[kind]
    _need(type(rows) is list and len(rows) <= MAX_ROWS and all(type(row) is dict for row in rows),
          "column_row_population_bound")
    present = known.intersection(*(set(row) for row in rows)) if rows else set()
    shared = {}
    if len(rows) >= 2:
        for key in sorted(present):
            original = _json(rows[0][key])
            if all(_json(row[key]) == original for row in rows[1:]):
                shared[key] = rows[0][key]
    columns = sorted(present - shared.keys())
    used = set(shared) | set(columns)
    descriptor = dict(schema=TABLE_SCHEMA, kind=kind, shared=shared, columns=columns,
        rows=[[row[key] for key in columns] for row in rows],
        extras=[{key: value for key, value in row.items() if key not in used} for row in rows])
    return descriptor


def _shape(descriptor, kind, *, refs=False):
    _need(type(descriptor) is dict and set(descriptor) == _TABLE_FIELDS | ({"refs"} if refs else set())
          and descriptor["schema"] == TABLE_SCHEMA and descriptor["kind"] == kind,
          "column_table_schema_mismatch")
    shared, columns, rows, extras = (descriptor[k] for k in ("shared", "columns", "rows", "extras"))
    _need(type(shared) is dict and type(columns) is list and type(rows) is list and type(extras) is list,
          "column_table_shape_mismatch")
    _need(len(rows) <= MAX_ROWS and len(extras) == len(rows)
          and (len(rows) >= 2 or not shared), "column_row_population_bound")
    _need(all(type(key) is str for key in columns) and columns == sorted(set(columns))
          and set(shared).issubset(GROUP_FIELDS[kind]) and set(columns).issubset(GROUP_FIELDS[kind])
          and not set(shared).intersection(columns), "column_header_partition_mismatch")
    for row, extra in zip(rows, extras, strict=True):
        _need(type(row) is list and len(row) == len(columns) and type(extra) is dict
              and not set(extra).intersection(set(shared) | set(columns)), "column_row_width_or_overlap")
        _need(kind != "admissions" or "ref" not in extra, "column_ref_override")


def _pairs(descriptor, index):
    yield from descriptor["shared"].items()
    yield from zip(descriptor["columns"], descriptor["rows"][index], strict=True)
    yield from descriptor["extras"][index].items()


def _row_metrics(descriptor, index, ref_metrics=None):
    nodes, size, depth, fields = 1, 2, 0, 0
    for key, value in _pairs(descriptor, index):
        value_nodes, value_size, value_depth = _profile(value)
        nodes += 1 + value_nodes
        size += _string_wire_bytes(key) + 1 + value_size + int(fields > 0)
        depth = max(depth, value_depth + 1)
        fields += 1
        _need(nodes <= MAX_NODES and size <= MAX_BYTES and depth <= MAX_DEPTH, "column_expansion_bound")
    if ref_metrics is not None:
        ref_nodes, ref_size, ref_depth = ref_metrics
        nodes += 1 + ref_nodes
        size += _string_wire_bytes("ref") + 1 + ref_size + int(fields > 0)
        depth = max(depth, ref_depth + 1)
    _need(nodes <= MAX_NODES and size <= MAX_BYTES and depth <= MAX_DEPTH, "column_expansion_bound")
    return nodes, size, depth


def _expanded_metrics(descriptor, *, refs=None):
    nodes, size, depth = 1, 2 + max(0, len(descriptor["rows"]) - 1), 0
    for index in range(len(descriptor["rows"])):
        ref = _row_metrics(refs, index) if refs is not None else None
        row_nodes, row_size, row_depth = _row_metrics(descriptor, index, ref)
        nodes += row_nodes
        size += row_size
        depth = max(depth, row_depth + 1)
        _need(nodes <= MAX_NODES and size <= MAX_BYTES and depth <= MAX_DEPTH, "column_expansion_bound")
    return nodes, size, depth


def _materialize(descriptor):
    """Called only after the complete three-group expansion preflight."""
    return [{key: value for key, value in _pairs(descriptor, index)}
            for index in range(len(descriptor["rows"]))]


def _columns(wire):
    semantic = wire["translated_semantic"]
    return {key: semantic[key] for key in ("capsules", "admissions")}


def _stats(rows, descriptor):
    return dict(rows=len(rows), shared_fields=len(descriptor["shared"]), columns=len(descriptor["columns"]),
        extra_fields=sum(len(row) for row in descriptor["extras"]),
        original_bytes=len(_json(rows).encode()), packed_bytes=len(_json(descriptor).encode()))


def _forward(wire):
    semantic, populations = metadata._semantic_rows(wire)
    original_admissions = [{k: v for k, v in row.items() if k != "ref"} for row in populations["admissions"]]
    original_refs = populations["admission_refs"]
    capsules = _pack_group(populations["capsules"], "capsules")
    admissions = _pack_group(original_admissions, "admissions")
    refs = _pack_group(original_refs, "admission_refs")
    stats = dict(capsules=_stats(populations["capsules"], capsules),
                 admissions=_stats(original_admissions, admissions), admission_refs=_stats(original_refs, refs))
    admissions["refs"] = refs
    semantic["capsules"], semantic["admissions"] = capsules, admissions
    wire.update(schema=VIEW_SCHEMA, source_transport_schema=TRANSPORT_SCHEMA,
                capsule_columns_instructions=INSTRUCTIONS)
    return stats


def _inverse(wire):
    _need(type(wire) is dict and set(wire) == _VIEW_FIELDS and wire.get("schema") == VIEW_SCHEMA
          and wire.get("source_transport_schema") == TRANSPORT_SCHEMA
          and wire.get("capsule_columns_instructions") == INSTRUCTIONS,
          "column_view_schema_mismatch")
    semantic = wire.get("translated_semantic")
    _need(type(semantic) is dict and "capsules" in semantic and "admissions" in semantic,
          "column_semantic_populations_missing")
    capsules, admissions = semantic["capsules"], semantic["admissions"]
    _shape(capsules, "capsules")
    _shape(admissions, "admissions", refs=True)
    refs = admissions["refs"]
    _shape(refs, "admission_refs")
    _need(len(capsules["rows"]) == len(admissions["rows"]) == len(refs["rows"]),
          "column_population_alignment_mismatch")
    # Replace the descriptors only in a shallow skeleton. No shared literals or
    # rows are expanded until total nodes, bytes and depth have been measured.
    skeleton = dict(wire)
    skeleton["translated_semantic"] = {**semantic, "capsules": [], "admissions": []}
    skeleton.pop("source_transport_schema")
    skeleton.pop("capsule_columns_instructions")
    skeleton["schema"] = TRANSPORT_SCHEMA
    outside_nodes, outside_bytes, outside_depth = _profile(skeleton)
    capsule_nodes, capsule_bytes, capsule_depth = _expanded_metrics(capsules)
    admission_nodes, admission_bytes, admission_depth = _expanded_metrics(admissions, refs=refs)
    _need(outside_nodes - 2 + capsule_nodes + admission_nodes <= MAX_NODES
          and outside_bytes - 4 + capsule_bytes + admission_bytes <= MAX_BYTES
          and max(outside_depth, capsule_depth + 2, admission_depth + 2) <= MAX_DEPTH,
          "column_complete_expansion_bound")
    original_capsules = _materialize(capsules)
    original_admissions = _materialize(admissions)
    original_refs = _materialize(refs)
    for row, ref in zip(original_admissions, original_refs, strict=True):
        row["ref"] = ref
    # The deterministic header/extras recipe must replay, independently of the
    # receipt's supplied hashes. No alternative sparse map or default inference.
    _need(_json(_pack_group(original_capsules, "capsules")) == _json(capsules), "column_capsule_recipe_mismatch")
    base_admissions = [{k: v for k, v in row.items() if k != "ref"} for row in original_admissions]
    actual_admissions = {k: v for k, v in admissions.items() if k != "refs"}
    _need(_json(_pack_group(base_admissions, "admissions")) == _json(actual_admissions)
          and _json(_pack_group(original_refs, "admission_refs")) == _json(refs), "column_admission_recipe_mismatch")
    skeleton["translated_semantic"]["capsules"] = original_capsules
    skeleton["translated_semantic"]["admissions"] = original_admissions
    metadata._transport(skeleton)
    return skeleton


def validate_capsule_columns_receipt(receipt):
    _need(type(receipt) is dict, "column_trusted_receipt_required")
    _need(_profile(receipt)[1] <= MAX_RECEIPT_BYTES, "column_receipt_bound")
    text = _json(receipt)
    _need(len(text.encode()) <= MAX_RECEIPT_BYTES, "column_receipt_bound")
    retained = _parse(text)
    _need(set(retained) == _RECEIPT_FIELDS and retained.get("schema") == RECEIPT_SCHEMA
          and retained.get("view_schema") == VIEW_SCHEMA and retained.get("source_transport_schema") == TRANSPORT_SCHEMA
          and retained.get("requested_mode") == MODE and retained.get("token_proxy") == metadata.TOKEN_PROXY
          and retained.get("candidate_only") is True
          and all(retained.get(field) is False for field in metadata._FALSE_FIELDS), "column_receipt_schema_or_authority")
    for key, value in retained.items():
        if key.endswith("sha256"):
            _need(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "column_receipt_digest")
    for key in ("translation_cid", "task_id", "scope_cid", "semantic_root_cid"):
        metadata._text(retained[key], key)
    for key in ("native_transport_bytes", "candidate_view_bytes", "native_transport_proxy_tokens", "candidate_view_proxy_tokens"):
        _need(type(retained[key]) is int and 0 <= retained[key] <= MAX_BYTES, "column_receipt_count")
    _need(retained["native_transport_proxy_tokens"] == (retained["native_transport_bytes"] + 3) // 4
          and retained["candidate_view_proxy_tokens"] == (retained["candidate_view_bytes"] + 3) // 4,
          "column_receipt_proxy_arithmetic")
    for key in ("capsule_count", "admission_count"):
        _need(type(retained[key]) is int and 0 <= retained[key] <= MAX_ROWS, "column_receipt_population_bound")
    groups = retained["group_counts"]
    _need(type(groups) is dict and set(groups) == set(GROUP_FIELDS), "column_receipt_group_counts")
    for kind, stats in groups.items():
        _need(type(stats) is dict and set(stats) == _STAT_FIELDS
              and all(type(v) is int and 0 <= v <= MAX_BYTES for v in stats.values()), "column_receipt_group_counts")
        _need(stats["rows"] == retained["capsule_count"] == retained["admission_count"]
              and stats["shared_fields"] + stats["columns"] <= len(GROUP_FIELDS[kind]), "column_receipt_population_alignment")
    return retained


def project_capsule_columns(provider_prompt):
    """Present an already verified canonical native @1 transport losslessly."""
    wire = _parse(provider_prompt)
    metadata._transport(wire)
    bindings = metadata._bindings(wire)
    stats = _forward(wire)
    candidate = _json(wire)
    receipt = dict(schema=RECEIPT_SCHEMA, view_schema=VIEW_SCHEMA, source_transport_schema=TRANSPORT_SCHEMA,
        requested_mode=MODE, **bindings, native_transport_sha256=metadata._sha(provider_prompt),
        native_transport_bytes=len(provider_prompt.encode()), candidate_view_sha256=metadata._sha(candidate),
        candidate_view_bytes=len(candidate.encode()), columns_sha256=metadata._sha(_json(_columns(wire))),
        group_counts=stats, capsule_count=stats["capsules"]["rows"], admission_count=stats["admissions"]["rows"],
        token_proxy=metadata.TOKEN_PROXY, native_transport_proxy_tokens=metadata._proxy(provider_prompt),
        candidate_view_proxy_tokens=metadata._proxy(candidate), candidate_only=True,
        **dict.fromkeys(metadata._FALSE_FIELDS, False))
    validate_capsule_columns_receipt(receipt)
    view = metadata.SemanticMetadataView(provider_prompt, candidate, _json(receipt))
    _need(restore_capsule_columns(provider_prompt=candidate, receipt=receipt) == provider_prompt,
          "column_native_round_trip_mismatch")
    return view


def restore_capsule_columns(*, provider_prompt, receipt):
    """Restore against the independently retained original/candidate bindings."""
    retained = validate_capsule_columns_receipt(receipt)
    wire = _parse(provider_prompt)
    _need(metadata._sha(provider_prompt) == retained["candidate_view_sha256"]
          and len(provider_prompt.encode()) == retained["candidate_view_bytes"], "column_candidate_pin_mismatch")
    _need(type(wire) is dict and type(wire.get("translated_semantic")) is dict
          and "capsules" in wire["translated_semantic"] and "admissions" in wire["translated_semantic"],
          "column_semantic_populations_missing")
    _need(metadata._sha(_json(_columns(wire))) == retained["columns_sha256"], "column_table_pin_mismatch")
    native = _inverse(wire)
    _need(all(metadata._bindings(native)[key] == retained[key] for key in metadata._BINDING_FIELDS),
          "column_native_binding_mismatch")
    original = _json(native)
    _need(metadata._sha(original) == retained["native_transport_sha256"]
          and len(original.encode()) == retained["native_transport_bytes"], "column_native_pin_mismatch")
    stats = _forward(native)
    _need(stats == retained["group_counts"] and _json(native) == provider_prompt,
          "column_fixed_projection_mismatch")
    return original


def select_capsule_columns(*, view, native_complete_prompt, candidate_complete_prompt,
                           token_counter=None, provider=None, model=None):
    """Count all replacement headers and literal wrappers before selection.

    A supplied counter needs the explicitly matching caller route. Missing,
    mismatched, invalid or failed counter measurements retain the original;
    they never fall back to a byte proxy. Tokenizer qualification remains a
    caller declaration and counts exclude hidden provider context/later turns.
    """
    _need(type(view) is metadata.SemanticMetadataView, "column_typed_view_required")
    _need(type(view.receipt_json) is str and len(view.receipt_json) <= MAX_RECEIPT_BYTES,
          "column_receipt_bound")
    _need(restore_capsule_columns(provider_prompt=view.provider_prompt, receipt=_parse(view.receipt_json)) == view.native_prompt,
          "column_view_identity_mismatch")
    for text in (native_complete_prompt, candidate_complete_prompt):
        _need(type(text) is str and len(text) <= MAX_BYTES and len(text.encode()) <= MAX_BYTES,
              "column_complete_input_bound")
    _need(native_complete_prompt.count(view.native_prompt) == 1, "column_native_transport_occurrence")
    prefix, suffix = native_complete_prompt.split(view.native_prompt)
    _need(candidate_complete_prompt == prefix + view.provider_prompt + suffix
          and candidate_complete_prompt.count(view.provider_prompt) == 1, "column_complete_wrapper_mismatch")
    smaller = len(candidate_complete_prompt.encode()) < len(native_complete_prompt.encode())
    measurement = None
    if token_counter is None:
        selected = smaller and metadata._proxy(candidate_complete_prompt) < metadata._proxy(native_complete_prompt)
        fallback = None if selected else "complete_input_not_smaller_under_bytes_and_proxy"
    else:
        bound = (type(token_counter) is metadata.SemanticMetadataTokenCounter
                 and type(provider) is str and type(model) is str
                 and 0 < len(provider) <= 512 and 0 < len(model) <= 512
                 and provider == provider.strip() and model == model.strip()
                 and token_counter.provider == provider and token_counter.model == model)
        measurement = metadata._counter_measurement(token_counter, native_complete_prompt, candidate_complete_prompt) if bound else dict(
            token_counter_status="invalid", token_counter_identity=None, token_counter_identity_sha256=None,
            native_complete_tokens=None, candidate_complete_tokens=None)
        selected = (measurement["token_counter_status"] == "counted" and smaller
                    and measurement["candidate_complete_tokens"] < measurement["native_complete_tokens"])
        fallback = (None if selected else "complete_input_not_smaller_under_bytes_and_token_counter"
                    if measurement["token_counter_status"] == "counted" else
                    "complete_input_token_counter_" + measurement["token_counter_status"])
    chosen = candidate_complete_prompt if selected else native_complete_prompt
    receipt = dict(schema=metadata.SELECTION_SCHEMA, requested_mode=MODE,
        view_receipt_sha256=metadata._sha(view.receipt_json), selected_mode=MODE if selected else "legacy",
        fallback_reason=fallback, token_proxy=metadata.TOKEN_PROXY, candidate_only=True,
        **{arm + "_complete_" + field: value for arm, text in
           (("native", native_complete_prompt), ("candidate", candidate_complete_prompt), ("selected", chosen))
           for field, value in (("sha256", metadata._sha(text)), ("bytes", len(text.encode())),
                                ("proxy_tokens", metadata._proxy(text)))},
        **dict.fromkeys(metadata._FALSE_FIELDS, False))
    if measurement is not None:
        receipt.update(schema=metadata.TOKEN_SELECTION_SCHEMA, **measurement, selection_basis="token_counter",
            actual_provider_usage_verified=False, provider_usage_reported=False, cli_scaffolding_measured=False,
            complete_run_token_savings_claimed=False,
            selected_complete_tokens=(measurement["candidate_complete_tokens"] if selected else measurement["native_complete_tokens"]))
    return metadata.SemanticMetadataSelection(chosen, receipt["selected_mode"], _json(receipt))


__all__ = ["MODE", "VIEW_SCHEMA", "RECEIPT_SCHEMA", "TABLE_SCHEMA", "INSTRUCTIONS",
    "CapsuleColumnsError", "project_capsule_columns", "restore_capsule_columns",
    "validate_capsule_columns_receipt", "select_capsule_columns"]
