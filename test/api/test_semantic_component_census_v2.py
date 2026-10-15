"""Closed, opt-in byte accounting for semantic preparation and retrieval."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from test.api.test_semantic_router_translation import native  # noqa: F401
from test.api.test_semantic_prompt_census import byte_count, check_partitions
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import build_text_context_references
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context


EXTRA_MEMBERS = frozenset({"reconstruction", "refresh_lineage", "preparation_bounds", "pack",
    "worker_projection", "required_raw_paths", "program_paths", "task_data_projection"})
EXTRA_EVIDENCE = frozenset({"intent_world_context", "code_retrieval_context"})


def _detailed(encoded):
    return codec.census_semantic_router_prompt(encoded=encoded,
        census_schema=codec.DETAILED_COMPONENT_CENSUS_SCHEMA)


def _aggregate_to_v1(report):
    """The richer census changes only labels, never the measured byte set."""
    result = deepcopy(report)
    result["schema"] = codec.COMPONENT_CENSUS_SCHEMA
    members = result["translated_semantic"]["members"]
    for label in EXTRA_MEMBERS:
        for metric, value in members.pop(label).items():
            members["other_fields"][metric] += value
    for name in ("native_evidence", "remaining_native_evidence"):
        groups = result[name]["groups"]
        for label in EXTRA_EVIDENCE:
            for metric, value in groups.pop(label).items():
                groups["other_evidence"][metric] += value
    return result


@pytest.mark.parametrize("transport_schema", [codec.TRANSPORT_SCHEMA, codec.COMPACT_TRANSPORT_SCHEMA])
def test_detailed_census_is_explicit_and_exactly_refines_the_unchanged_default(native, transport_schema,
        monkeypatch):
    root, _, prompt, source = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root,
        transport_schema=transport_schema)
    before = deepcopy(encoded)
    monkeypatch.setattr(codec, "_read", lambda *a, **kw: pytest.fail("census cannot read sources"))
    default = codec.census_semantic_router_prompt(encoded=encoded)
    explicit = codec.census_semantic_router_prompt(encoded=encoded,
        census_schema=codec.COMPONENT_CENSUS_SCHEMA)
    report = _detailed(encoded)
    assert codec._json(default) == codec._json(explicit)
    assert default["schema"] == codec.COMPONENT_CENSUS_SCHEMA
    assert report["schema"] == codec.DETAILED_COMPONENT_CENSUS_SCHEMA
    assert _aggregate_to_v1(report) == default
    check_partitions(report)
    assert encoded == before
    assert report["native_prompt_sha256"] == codec._sha(prompt)
    assert report["provider_prompt_sha256"] == codec._sha(encoded.provider_prompt)
    assert report["exact_native_round_trip_verified"] is True
    assert report["nested_partitions_overlap"] is True
    assert report["provider_tokens"] is None and report["provider_calls"] == 0
    for name in ("token_estimate", "model_prompt_wrappers_included", "raw_content_exported",
            "dynamic_labels_exported", "source_freshness_verified", "proof_authority",
            "omission_authority", "completion_authority", "execution_authority"):
        assert report[name] is False
    output = codec._json(report)
    assert len(output.encode()) < 16_384
    for private in (source, str(root), "mod.py", "TASK-1", encoded.table.translation_cid):
        assert private not in output


@pytest.mark.parametrize("projected", [False, True])
def test_detailed_semantic_members_measure_real_producer_fields(native, projected):
    root, artifact, prompt, _ = native
    if projected:
        # Produce the optional fields through their real constructor. This is
        # an authored test repository, not a live owner archive.
        artifact = artifact.parent.with_name("semantic-projected") / "worker-context.json"
        prepare_semantic_context(repository=root, paths=["mod.py"], required_raw_paths=["mod.py"],
            program_paths=["mod.py"], objective="Inspect both functions.", task_id="TASK-1",
            output=artifact.parent, worker_query="lengthy_first_function", worker_capsule_limit=2)
        wire, end = json.JSONDecoder().raw_decode(prompt)
        refs = build_text_context_references(artifact.read_text(), reference_prefix="semantic-context",
            kind="semantic-context", path=artifact.relative_to(root).as_posix(),
            repository_id=wire["repository_id"], tree_id=wire["tree_id"], required=True, chunk_bytes=1201)
        wire["evidence"] = [row.to_record() for row in refs]
        prompt = codec._json(wire) + prompt[end:]
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    report = _detailed(encoded)
    check_partitions(report)
    payload = json.loads(encoded.provider_prompt)["translated_semantic"]
    members = report["translated_semantic"]["members"]
    assert set(members) == {"capsules", "admissions", "raw_sources", "manifest", "other_fields",
                           *EXTRA_MEMBERS}
    for key in EXTRA_MEMBERS:
        assert members[key]["count"] == (key in payload)
        assert members[key]["value_bytes"] == (byte_count(payload[key]) if key in payload else 0)
        assert members[key]["member_bytes"] == (
            byte_count(key) + 1 + byte_count(payload[key]) if key in payload else 0)
    assert members["worker_projection"]["count"] == projected
    assert members["program_paths"]["count"] == projected
    assert members["reconstruction"]["value_bytes"] > 0
    # These measured fields are still fully inline after this observation.
    assert codec._restore(encoded.provider_prompt, encoded.table) == prompt


def test_world_retrieval_and_unknown_evidence_remain_closed_and_reconcile(native):
    root, _, prompt, _ = native
    wire, end = json.JSONDecoder().raw_decode(prompt)
    wire["PRIVATE-DYNAMIC-TOP-KEY"] = {"PRIVATE-NESTED-KEY": "PRIVATE-TOP-CONTENT"}
    known = {}
    for kind in ("intent-world-context", "code-retrieval-context", "source-formalization-context"):
        body = codec._json({"schema": "authored-input@1", "PRIVATE-KEY": 'PRIVATE-BODY λ\\"\n' * 60})
        refs = build_text_context_references(body, reference_prefix=kind, kind=kind,
            path=".runtime/PRIVATE-PATH.json", repository_id=wire["repository_id"],
            tree_id=wire["tree_id"], required=True, chunk_bytes=271)
        rows = [row.to_record() for row in refs]
        known[kind.replace("-", "_")] = (body, rows)
        wire["evidence"].extend(rows)
    wire["evidence"].extend([
        {"kind": "PRIVATE-KIND", "summary": "PRIVATE-UNKNOWN-TEXT λ\n"},
        {"kind": {"PRIVATE-DYNAMIC-LABEL": True}, "summary": {"PRIVATE-SUMMARY": 1}},
        "PRIVATE-NONOBJECT-ROW",
    ])
    prompt = codec._json(wire) + prompt[end:]
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    report = _detailed(encoded)
    check_partitions(report)
    assert _aggregate_to_v1(report) == codec.census_semantic_router_prompt(encoded=encoded)
    for name in ("native_evidence", "remaining_native_evidence"):
        groups = report[name]["groups"]
        assert set(groups) == {"semantic_context", "source_formalization_context", "other_evidence",
                               *EXTRA_EVIDENCE}
        for label, (body, rows) in known.items():
            measured = groups[label]
            assert measured["count"] == len(rows) > 1
            assert measured["row_bytes"] == sum(byte_count(row) for row in rows)
            assert measured["summary_utf8_bytes"] == len(body.encode())
            assert measured["summary_json_string_bytes"] == sum(byte_count(row["summary"]) for row in rows)
            assert measured["summary_json_string_overhead_bytes"] > 2 * len(rows)
        assert groups["other_evidence"]["count"] == 3
    assert "PRIVATE" not in codec._json(report)
    assert len(codec._json(report).encode()) < 16_384


@pytest.mark.parametrize("schema", [None, 2, True, {}, [], "", "PRIVATE-SCHEMA",
    "supervisor-semantic-prompt-component-census@3"])
def test_unknown_census_versions_refuse_before_inspecting_inputs(schema):
    with pytest.raises(codec.SemanticTranslationError, match="unsupported semantic component census schema") as failure:
        codec.census_semantic_router_prompt(encoded=None, census_schema=schema)
    assert "PRIVATE" not in str(failure.value)


@pytest.mark.parametrize("change", ["native", "provider", "provider_noncanonical", "native_suffix"])
def test_detailed_census_rejects_changed_input_bindings(native, change):
    root, _, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    if change == "native":
        encoded = replace(encoded, native_prompt="PRIVATE changed prompt")
    else:
        value = json.loads(encoded.provider_prompt)
        if change == "provider":
            value["translated_semantic"]["reconstruction"] = {"PRIVATE-CHANGED": 1}
        elif change == "native_suffix":
            value["native_suffix"] += "PRIVATE changed suffix"
        encoded = replace(encoded, provider_prompt=json.dumps(value, indent=2)
            if change == "provider_noncanonical" else codec._json(value))
    with pytest.raises(codec.SemanticTranslationError) as failure:
        _detailed(encoded)
    assert "PRIVATE" not in str(failure.value)
