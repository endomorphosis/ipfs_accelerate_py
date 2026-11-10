"""Closed pack accounting over actual producer fixtures and exact inverses."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from test.api.test_semantic_router_translation import native  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import build_text_context_references
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_pack_census as census


def _size(value):
    return len(codec._json(value).encode("utf-8"))


def _check_partition(value):
    assert value["outer_punctuation_bytes"] + sum(
        row["member_bytes"] for row in value["members"].values()) == value["object_bytes"]
    assert all(row["member_bytes"] >= row["value_bytes"] >= 0 for row in value["members"].values())


def _check(report, encoded):
    pack = json.loads(encoded.provider_prompt)["translated_semantic"]["pack"]
    _check_partition(report["pack_object"])
    _check_partition(report["contract_object"])
    refs = report["references"]
    _check_partition(refs["object_rows"])
    assert report["pack_object"]["object_bytes"] == _size(pack)
    assert report["pack_object"]["members"]["pack"]["value_bytes"] == report["contract_object"]["object_bytes"]
    assert report["pack_object"]["members"]["references"]["value_bytes"] == refs["array_bytes"]
    assert refs["array_bytes"] == refs["outer_punctuation_bytes"] + refs["object_rows"]["object_bytes"]
    assert refs["object_rows"]["object_bytes"] == sum(row["row_bytes"] for row in refs["groups"].values())
    assert refs["object_rows"]["count"] == sum(row["count"] for row in refs["groups"].values())
    summary = refs["summary_strings"]
    assert summary["utf8_bytes"] + summary["json_string_overhead_bytes"] == summary["json_string_bytes"]
    repetition = report["typed_identifier_repetition"]
    assert repetition["translated_value_bytes"] == (
        repetition["repeated_groups_translated_value_bytes"] + repetition["singleton_translated_value_bytes"])
    assert repetition["occurrences_beyond_first"] == repetition["slots"] - repetition["distinct_native_identities"]
    component = codec.census_semantic_router_prompt(encoded=encoded,
        census_schema=codec.DETAILED_COMPONENT_CENSUS_SCHEMA)
    assert component["translated_semantic"]["members"]["pack"]["value_bytes"] == report["pack_object"]["object_bytes"]


@pytest.mark.parametrize("transport", [codec.TRANSPORT_SCHEMA, codec.COMPACT_TRANSPORT_SCHEMA])
def test_actual_pack_partitions_and_current_exact_inverse(native, monkeypatch, transport):
    root, _, prompt, source = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root, transport_schema=transport)
    before = deepcopy(encoded)
    monkeypatch.setattr(codec, "_read", lambda *a, **kw: pytest.fail("pure census cannot read files"))
    report = census.census_semantic_pack(encoded=encoded)
    _check(report, encoded)
    assert encoded == before
    assert codec._restore(encoded.provider_prompt, encoded.table) == prompt
    assert report["schema"] == census.SCHEMA
    assert report["exact_native_round_trip_verified"] is True
    assert report["typed_encoding_currentness_supplied"] is True
    assert report["nested_partitions_overlap"] is report["repetition_overlaps_partitions"] is True
    assert report["provider_calls"] == 0 and report["provider_tokens"] is None
    for flag in ("source_freshness_verified", "token_estimate", "producer_token_estimates_reexported",
        "model_prompt_wrappers_included", "raw_content_exported", "dynamic_labels_exported",
        "proof_authority", "omission_authority", "execution_authority", "completion_authority",
        "semantic_equivalence_claimed", "input_changed", "token_savings_measured"):
        assert report[flag] is False
    output = codec._json(report)
    assert len(output.encode()) <= census.MAX_REPORT_BYTES
    for private in (source, str(root), "mod.py", "TASK-1", encoded.table.translation_cid,
                    "Inspect both functions."):
        assert private not in output


def test_native_identifier_repetition_keeps_distinct_required_roles(native):
    root, artifact, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    report = census.census_semantic_pack(encoded=encoded)
    pack = json.loads(artifact.read_text())["pack"]
    # The real producer emits exact source references for three separate roles.
    assert pack["pack"]["target_source_cid"] == pack["pack"]["test_source_cid"]
    groups = report["references"]["groups"]
    for kind in ("target_source", "surrounding_source", "test_source"):
        assert groups[kind]["count"] == 1
    repetition = report["typed_identifier_repetition"]
    assert repetition["repeated_identity_groups"] > 0
    assert repetition["occurrences_beyond_first"] > 0
    assert repetition["repeated_groups_translated_value_bytes"] > 0
    assert repetition["all_occurrences_in_repeated_groups_included"] is True
    assert repetition["literal_embedded_identifiers_counted"] is False
    assert repetition["removable_bytes_estimated"] is False
    assert report["omission_authority"] is False
    assert codec._restore(encoded.provider_prompt, encoded.table) == prompt


def _authored_extension(native, mutate):
    """Extend an authored v1 fixture; retain source/producer and native bindings."""
    root, artifact, prompt, _ = native
    payload = json.loads(artifact.read_text())
    mutate(payload["pack"])
    artifact.write_text(codec._json(payload))
    wire, end = json.JSONDecoder().raw_decode(prompt)
    refs = build_text_context_references(artifact.read_text(), reference_prefix="semantic-context",
        kind="semantic-context", path=artifact.relative_to(root).as_posix(),
        repository_id=wire["repository_id"], tree_id=wire["tree_id"], required=True, chunk_bytes=1201)
    wire["evidence"] = [row.to_record() for row in refs]
    return codec.encode_semantic_router_prompt(prompt=codec._json(wire)+prompt[end:], repository=root)


def test_unknown_fields_and_reference_kinds_stay_unnamed(native):
    def mutate(pack):
        pack["PRIVATE-TOP-KEY"] = {"PRIVATE-NESTED-KEY": "PRIVATE-CONTENT"}
        pack["pack"]["PRIVATE-CONTRACT-KEY"] = "PRIVATE-CONTRACT-VALUE"
        pack["references"][0].update(kind="PRIVATE-KIND", summary='PRIVATE-SUMMARY λ\\"\n',
            **{"PRIVATE-REFERENCE-KEY": {"PRIVATE-CHILD": 1234567890}})
        pack["references"][0]["metadata"]["PRIVATE-METADATA-KEY"] = "PRIVATE-METADATA-CONTENT"
        pack["pack"]["token_totals"]["PRIVATE-TOKEN-CATEGORY"] = 1234567890
    encoded = _authored_extension(native, mutate)
    report = census.census_semantic_pack(encoded=encoded)
    _check(report, encoded)
    assert set(report["pack_object"]["members"]) == {*census.PACK_FIELDS, "other_fields"}
    assert set(report["contract_object"]["members"]) == {*census.CONTRACT_FIELDS, "other_fields"}
    assert set(report["references"]["groups"]) == {*census.REFERENCE_KINDS, "other_references"}
    assert report["pack_object"]["members"]["other_fields"]["count"] == 1
    assert report["contract_object"]["members"]["other_fields"]["count"] == 1
    assert report["references"]["object_rows"]["members"]["other_fields"]["count"] == 1
    assert report["references"]["groups"]["other_references"]["count"] == 1
    assert report["references"]["summary_strings"]["json_string_overhead_bytes"] > 0
    output = codec._json(report)
    assert "PRIVATE" not in output and "1234567890" not in output


def test_literal_embedded_identifiers_do_not_inflate_typed_repetition(native):
    root, _, prompt, _ = native
    before = census.census_semantic_pack(encoded=codec.encode_semantic_router_prompt(prompt=prompt, repository=root))
    def mutate(pack):
        identity = pack["pack"]["target_source_cid"]
        pack["decisions"].extend([identity] * 12)
        pack["pack"]["assumptions"].extend([identity] * 12)
        pack["references"][0]["summary"] = identity * 12
        pack["references"][0]["metadata"]["PRIVATE-DYNAMIC-IDENTIFIER"] = identity
    encoded = _authored_extension(native, mutate)
    after = census.census_semantic_pack(encoded=encoded)
    _check(after, encoded)
    assert after["typed_identifier_repetition"] == before["typed_identifier_repetition"]
    assert after["pack_object"]["object_bytes"] > before["pack_object"]["object_bytes"]


def test_empty_reference_array_still_has_exact_punctuation(native):
    encoded = _authored_extension(native, lambda pack: pack.update(references=[]))
    report = census.census_semantic_pack(encoded=encoded)
    _check(report, encoded)
    assert report["references"]["array_bytes"] == report["references"]["outer_punctuation_bytes"] == 2
    assert report["references"]["object_rows"]["object_bytes"] == 0


@pytest.mark.parametrize("change", ["native", "provider", "noncanonical", "suffix", "historical", "freshness-int"])
def test_changed_or_historical_encodings_refused_without_private_error_text(native, change):
    root, _, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    if change == "native": encoded = replace(encoded, native_prompt="PRIVATE changed prompt")
    elif change == "historical": encoded = replace(encoded, freshness_checked=False)
    elif change == "freshness-int": encoded = replace(encoded, freshness_checked=1)
    else:
        value = json.loads(encoded.provider_prompt)
        if change == "provider": value["translated_semantic"]["pack"]["decisions"] = ["PRIVATE-CHANGE"]
        if change == "suffix": value["native_suffix"] += "PRIVATE-CHANGE"
        encoded = replace(encoded, provider_prompt=json.dumps(value, indent=2)
            if change == "noncanonical" else codec._json(value))
    with pytest.raises(census.SemanticPackCensusError) as error:
        census.census_semantic_pack(encoded=encoded)
    assert "PRIVATE" not in str(error.value)


@pytest.mark.parametrize("value", [None, {}, "PRIVATE", True])
def test_exact_typed_input_required(value):
    with pytest.raises(census.SemanticPackCensusError, match="current typed"):
        census.census_semantic_pack(encoded=value)
