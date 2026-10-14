"""Exact in-memory byte partitions, without model input or authority changes."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from test.api.test_semantic_router_translation import native  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import build_text_context_references
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec


def byte_count(value):
    return len(codec._json(value).encode("utf-8"))


def check_partitions(report):
    for name in ("native_object", "provider_object", "remaining_native_context", "translated_semantic"):
        partition = report[name]
        assert (sum(row["member_bytes"] for row in partition["members"].values())
                + partition["outer_punctuation_bytes"] == partition["object_bytes"])
        assert all(row["member_bytes"] >= row["value_bytes"] >= 0
                   for row in partition["members"].values())
    for name in ("native_evidence", "remaining_native_evidence"):
        partition = report[name]
        assert (sum(row["row_bytes"] for row in partition["groups"].values())
                + partition["outer_punctuation_bytes"] == partition["array_bytes"])
        for row in partition["groups"].values():
            assert row["row_bytes"] == row["row_envelope_bytes"] + row["summary_json_string_bytes"]
            assert row["summary_json_string_bytes"] == row["summary_utf8_bytes"] + row["summary_json_string_overhead_bytes"]
    assert report["native_prompt_bytes"] == report["native_object"]["object_bytes"] + report["native_suffix_bytes"]
    assert report["provider_prompt_bytes"] == report["provider_object"]["object_bytes"]
    assert report["provider_object"]["members"]["native_context"]["value_bytes"] == report["remaining_native_context"]["object_bytes"]
    assert report["provider_object"]["members"]["translated_semantic"]["value_bytes"] == report["translated_semantic"]["object_bytes"]


@pytest.mark.parametrize("schema", [codec.TRANSPORT_SCHEMA, codec.COMPACT_TRANSPORT_SCHEMA])
def test_census_reconciles_real_producer_bytes_and_preserves_all_inputs(native, schema, monkeypatch):
    root, _, prompt, source = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root, transport_schema=schema)
    before = deepcopy(encoded)
    monkeypatch.setattr(codec, "_read", lambda *a, **kw: pytest.fail("census must not read source"))
    report = codec.census_semantic_router_prompt(encoded=encoded)
    check_partitions(report)
    assert encoded == before
    assert report["native_prompt_sha256"] == codec._sha(prompt)
    assert report["provider_prompt_sha256"] == codec._sha(encoded.provider_prompt)
    assert report["native_prompt_bytes"] == len(prompt.encode())
    assert report["provider_prompt_bytes"] == len(encoded.provider_prompt.encode())
    provider = json.loads(encoded.provider_prompt)
    for key in ("capsules", "admissions", "raw_sources", "manifest"):
        assert report["translated_semantic"]["members"][key]["value_bytes"] == byte_count(provider["translated_semantic"][key])
    native_rows = json.JSONDecoder().raw_decode(prompt)[0]["evidence"]
    assert report["native_evidence"]["groups"]["semantic_context"]["count"] == len(native_rows)
    assert report["remaining_native_evidence"]["groups"]["semantic_context"]["count"] == 0
    assert report["provider_object"]["members"]["translation_table"]["count"] == (schema == codec.TRANSPORT_SCHEMA)
    assert report["exact_native_round_trip_verified"] and report["nested_partitions_overlap"]
    assert not report["source_freshness_verified"]
    assert report["provider_tokens"] is None and report["provider_calls"] == 0
    for flag in ("token_estimate", "model_prompt_wrappers_included", "raw_content_exported",
                 "dynamic_labels_exported", "proof_authority", "omission_authority",
                 "completion_authority", "execution_authority"):
        assert report[flag] is False
    output = json.dumps(report)
    for private in (source, str(root), "mod.py", encoded.table.translation_cid, "TASK-1", "Inspect both functions."):
        assert private not in output
    assert len(output.encode()) < 16384


def test_formalization_chunks_and_unknown_evidence_use_closed_labels(native):
    root, _, prompt, _ = native
    wire, end = json.JSONDecoder().raw_decode(prompt)
    body = codec._json({"schema": "authored-formalization-shape@1",
        "data": ('private marker \\" λ\n' * 40)})
    refs = build_text_context_references(body, reference_prefix="source-formalization",
        kind="source-formalization-context", path=".runtime/authored.json",
        repository_id=wire["repository_id"], tree_id=wire["tree_id"], required=True, chunk_bytes=300)
    rows = [row.to_record() for row in refs]
    wire["evidence"].extend(rows)
    wire["evidence"].extend([
        {"kind": "PRIVATE-DYNAMIC-LABEL", "summary": "PRIVATE-OTHER-TEXT λ\n"},
        {"kind": {"PRIVATE-DYNAMIC-KEY": True}, "summary": {"PRIVATE-SUMMARY": 1}},
    ])
    updated = codec._json(wire) + prompt[end:]
    encoded = codec.encode_semantic_router_prompt(prompt=updated, repository=root)
    report = codec.census_semantic_router_prompt(encoded=encoded)
    check_partitions(report)
    for name in ("native_evidence", "remaining_native_evidence"):
        groups = report[name]["groups"]
        assert set(groups) == {"semantic_context", "source_formalization_context", "other_evidence"}
        formal = groups["source_formalization_context"]
        assert formal["count"] == len(refs)
        assert formal["row_bytes"] == sum(byte_count(row) for row in rows)
        assert formal["summary_utf8_bytes"] == len(body.encode())
        assert formal["summary_json_string_bytes"] == sum(byte_count(row.summary) for row in refs)
        assert formal["summary_json_string_overhead_bytes"] > 2 * len(refs)
        assert groups["other_evidence"]["count"] == 2
    assert "PRIVATE" not in json.dumps(report) and "private marker" not in json.dumps(report)


@pytest.mark.parametrize("change", ["native", "provider", "provider_noncanonical", "native_suffix"])
def test_census_refuses_mismatched_or_noncanonical_inputs(native, change):
    root, _, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    if change == "native":
        encoded = replace(encoded, native_prompt="unrelated PRIVATE prompt")
    elif change == "native_suffix":
        value = json.loads(encoded.provider_prompt)
        value["native_suffix"] += "PRIVATE changed suffix"
        encoded = replace(encoded, provider_prompt=codec._json(value))
    elif change == "provider_noncanonical":
        encoded = replace(encoded, provider_prompt=json.dumps(json.loads(encoded.provider_prompt), indent=2))
    else:
        value = json.loads(encoded.provider_prompt)
        value["translated_semantic"]["raw_sources"]["mod.py"] = "PRIVATE changed source"
        encoded = replace(encoded, provider_prompt=codec._json(value))
    with pytest.raises(codec.SemanticTranslationError) as failure:
        codec.census_semantic_router_prompt(encoded=encoded)
    assert "PRIVATE" not in str(failure.value)


@pytest.mark.parametrize("value", [None, {}, "PRIVATE prompt"])
def test_census_requires_the_typed_in_memory_encoding(value):
    with pytest.raises(codec.SemanticTranslationError, match="typed encoded"):
        codec.census_semantic_router_prompt(encoded=value)
