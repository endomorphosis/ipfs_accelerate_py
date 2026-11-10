"""Closed byte accounting for the existing inline ContextPack representation.

This measures exact serialized values and repeated native identifiers in fixed
typed slots. Equal identifiers do not make their reference records redundant:
roles, caveats, requiredness and source obligations remain independently bound.
No provider input, pack policy, admission or source file is changed here.
"""
from __future__ import annotations

from collections import Counter
import hashlib

from . import semantic_router_translation as codec

SCHEMA = "supervisor-semantic-pack-census@1"
MAX_REPORT_BYTES = 16_384

# ContextPackResult.to_dict, ContextPack.to_dict and ContextReference._payload
# define these vocabularies. Unknown extensions remain an unnamed aggregate.
PACK_FIELDS = (
    "schema", "interface", "pack", "pack_cid", "references", "token_estimate",
    "coverage_satisfied", "production_slice_cid", "budget_exceeded", "decisions",
)
CONTRACT_FIELDS = (
    "objective", "target_source_cid", "surrounding_source_cid", "test_source_cid",
    "dependency_capsule_cids", "obligation_cids", "counterexample_cids", "delta_cid",
    "interface_cids", "assumptions", "exclusions", "token_totals", "estimator_version",
    "risk", "route", "escalation_recommendation",
)
REFERENCE_FIELDS = (
    "schema", "contract_version", "reference_id", "kind", "tier", "referenced_content_id",
    "repository_id", "tree_id", "path", "summary", "byte_count", "token_count", "metadata",
)
REFERENCE_KINDS = (
    "target_source", "surrounding_source", "test_source", "dependency_capsule",
    "raw_dependency_source", "raw_source_region", "obligation", "counterexample",
    "repository_delta", "interface_schema",
)
_SINGLE_IDENTIFIERS = ("target_source_cid", "surrounding_source_cid", "test_source_cid", "delta_cid")
_LIST_IDENTIFIERS = ("dependency_capsule_cids", "obligation_cids", "counterexample_cids", "interface_cids")


class SemanticPackCensusError(ValueError):
    """Closed observation failure without source or dynamic field contents."""


def _size(value):
    return len(codec._json(value).encode("utf-8"))


def _members():
    return dict(count=0, value_bytes=0, member_bytes=0)


def _partition(value, labels):
    if type(value) is not dict:
        raise SemanticPackCensusError("pack census requires object components")
    members = {label: _members() for label in (*labels, "other_fields")}
    for key, child in value.items():
        row = members[key if key in labels else "other_fields"]
        child_bytes = _size(child)
        row["count"] += 1
        row["value_bytes"] += child_bytes
        row["member_bytes"] += _size(key) + 1 + child_bytes
    punctuation = 2 + max(0, len(value) - 1)
    total = _size(value)
    if sum(row["member_bytes"] for row in members.values()) + punctuation != total:
        raise SemanticPackCensusError("pack census object byte accounting differs")
    return dict(object_bytes=total, outer_punctuation_bytes=punctuation, members=members)


def _references(rows):
    if type(rows) is not list or len(rows) > codec.MAX_ITEMS:
        raise SemanticPackCensusError("pack census requires bounded references")
    groups = {label: dict(count=0, row_bytes=0) for label in (*REFERENCE_KINDS, "other_references")}
    members = {label: _members() for label in (*REFERENCE_FIELDS, "other_fields")}
    object_bytes = object_punctuation = 0
    summary_utf8 = summary_json = 0
    for item in rows:
        partition = _partition(item, REFERENCE_FIELDS)
        kind = item.get("kind")
        label = kind if type(kind) is str and kind in REFERENCE_KINDS else "other_references"
        groups[label]["count"] += 1
        groups[label]["row_bytes"] += partition["object_bytes"]
        object_bytes += partition["object_bytes"]
        object_punctuation += partition["outer_punctuation_bytes"]
        for name, row in partition["members"].items():
            for metric, count in row.items():
                members[name][metric] += count
        summary = item.get("summary")
        if type(summary) is str:
            summary_utf8 += len(summary.encode("utf-8"))
            summary_json += _size(summary)
    punctuation = 2 + max(0, len(rows) - 1)
    total = _size(rows)
    if (object_bytes + punctuation != total
            or sum(row["row_bytes"] for row in groups.values()) != object_bytes
            or sum(row["member_bytes"] for row in members.values()) + object_punctuation != object_bytes):
        raise SemanticPackCensusError("pack census reference byte accounting differs")
    return dict(array_bytes=total, outer_punctuation_bytes=punctuation, groups=groups,
        object_rows=dict(count=len(rows), object_bytes=object_bytes,
            outer_punctuation_bytes=object_punctuation, members=members),
        summary_strings=dict(utf8_bytes=summary_utf8, json_string_bytes=summary_json,
            json_string_overhead_bytes=summary_json-summary_utf8))


def _identifier_repetition(native_pack, translated_pack):
    """Count exact identities only in known slots, not arbitrary repeated text.

    A group's byte measure includes every occurrence when its native identity
    appears more than once. It is neither removable bytes nor an encoding bound.
    Serialized bytes refer to the actual translated value in the same slot.
    """
    slots = []

    def slot(native, translated):
        if type(native) is str and native:
            slots.append((native, _size(translated)))

    native, translated = native_pack["pack"], translated_pack["pack"]
    for name in _SINGLE_IDENTIFIERS:
        if name in native:
            slot(native[name], translated[name])
    for name in _LIST_IDENTIFIERS:
        if name in native:
            if type(native[name]) is not list or len(native[name]) > codec.MAX_ITEMS:
                raise SemanticPackCensusError("pack census requires bounded typed identifier lists")
            for index, item in enumerate(native[name]):
                slot(item, translated[name][index])
    for index, reference in enumerate(native_pack["references"]):
        counterpart = translated_pack["references"][index]
        if "referenced_content_id" in reference:
            slot(reference["referenced_content_id"], counterpart["referenced_content_id"])
        metadata = reference.get("metadata")
        if type(metadata) is dict:
            for name in ("source_cid", "capsule_cid", "assessment_cid"):
                if name in metadata:
                    slot(metadata[name], counterpart["metadata"][name])
    counts = Counter(identity for identity, _ in slots)
    repeated = {identity for identity, count in counts.items() if count > 1}
    total = sum(size for _, size in slots)
    repeated_bytes = sum(size for identity, size in slots if identity in repeated)
    return dict(scope="known-native-pack-identifier-slots@1", slots=len(slots),
        distinct_native_identities=len(counts), repeated_identity_groups=len(repeated),
        occurrences_beyond_first=sum(count-1 for count in counts.values()),
        translated_value_bytes=total, repeated_groups_translated_value_bytes=repeated_bytes,
        singleton_translated_value_bytes=total-repeated_bytes,
        all_occurrences_in_repeated_groups_included=True,
        literal_embedded_identifiers_counted=False, removable_bytes_estimated=False)


def census_semantic_pack(*, encoded: codec.EncodedSemanticPrompt) -> dict:
    """Measure an already encoded current input without files or model calls.

    Currentness is supplied by the encoder and is not revalidated by this pure
    helper. All nested partitions describe parent bytes again. Identifier
    repetition overlaps these partitions and never authorizes removing a role.
    """
    if type(encoded) is not codec.EncodedSemanticPrompt or encoded.freshness_checked is not True:
        raise SemanticPackCensusError("current typed semantic encoding required for pack census")
    try:
        if codec._restore(encoded.provider_prompt, encoded.table) != encoded.native_prompt:
            raise SemanticPackCensusError("pack census native reconstruction differs")
        provider = codec._parse(encoded.provider_prompt)
        if codec._json(provider) != encoded.provider_prompt:
            raise SemanticPackCensusError("pack census requires canonical provider input")
        _, _, _, _, native_text, _ = codec._prompt_parts(encoded.native_prompt)
        native = codec._parse(native_text)["pack"]
        translated = provider["translated_semantic"]["pack"]
        output = dict(schema=SCHEMA, measurement_scope="canonical-translated-semantic-pack-utf8-bytes@1",
            partition_rule="member_bytes_plus_outer_punctuation_equals_object_bytes",
            nested_partitions_overlap=True, repetition_overlaps_partitions=True,
            typed_encoding_currentness_supplied=True, source_freshness_verified=False,
            exact_native_round_trip_verified=True,
            native_prompt_sha256=hashlib.sha256(encoded.native_prompt.encode()).hexdigest(),
            provider_prompt_sha256=hashlib.sha256(encoded.provider_prompt.encode()).hexdigest(),
            pack_sha256=hashlib.sha256(codec._json(translated).encode()).hexdigest(),
            pack_object=_partition(translated, PACK_FIELDS),
            contract_object=_partition(translated["pack"], CONTRACT_FIELDS),
            references=_references(translated["references"]),
            typed_identifier_repetition=_identifier_repetition(native, translated),
            provider_calls=0, provider_tokens=None, token_estimate=False,
            producer_token_estimates_reexported=False, model_prompt_wrappers_included=False,
            raw_content_exported=False, dynamic_labels_exported=False,
            proof_authority=False, omission_authority=False, execution_authority=False,
            completion_authority=False, semantic_equivalence_claimed=False,
            input_changed=False, token_savings_measured=False)
        if len(codec._json(output).encode()) > MAX_REPORT_BYTES:
            raise SemanticPackCensusError("closed pack census exceeds byte bound")
        return output
    except (ValueError, KeyError, IndexError, TypeError, RecursionError, AttributeError):
        raise SemanticPackCensusError("pack census input or accounting differs") from None
