"""Authored producer inputs and closed metadata-only export controls."""
from dataclasses import replace
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudget, canonical_context_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_metadata_census as census
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_metadata_view as view_codec
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as native_codec
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata


TASK_SENTINEL = "PRIVATE_TASK_SENTINEL"
BODY_SENTINEL = "PRIVATE_SOURCE_BODY_SENTINEL"
PREFIX = "PRIVATE_PROVIDER_OR_PLAN_TEXT_SENTINEL\n"
SUFFIX = "\nPRIVATE_EVALUATOR_OR_SIGNATURE_TEXT_SENTINEL"


def canonical(value):
    return canonical_context_json_bytes(value).decode()


@pytest.fixture(scope="module")
def native_cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("authored-census-producer")
    result = {}
    specifications = {
        "one_row": {"PRIVATE_PATH_SENTINEL.py": "# " + BODY_SENTINEL + "\n"},
        "zero_rows": {"PRIVATE_PATH_SENTINEL.py": "# " + BODY_SENTINEL + "\ndef omitted(value):\n    return value\n"},
        "many": {"PRIVATE_PATH_SENTINEL.py": "# " + BODY_SENTINEL + "\n" + "".join(
            f"def function_{index}(value):\n    return value + {index}\n\n" for index in range(6))},
        "heterogeneous": {
            "PRIVATE_FIRST_PATH.py": "# " + BODY_SENTINEL + "\ndef first(value):\n    return value + 1\n",
            "PRIVATE_SECOND_PATH.py": "# " + BODY_SENTINEL + "\ndef second(value):\n    return value - 2\n",
        },
        "same_content": {
            "PRIVATE_FIRST_PATH.py": "# " + BODY_SENTINEL + "\ndef common(value):\n    return value + 1\n",
            "PRIVATE_SECOND_PATH.py": "# " + BODY_SENTINEL + "\ndef common(value):\n    return value + 1\n",
        },
    }
    with public_replay_without_metadata():
        for case_name, sources in specifications.items():
            repository = root / case_name
            repository.mkdir()
            for name, text in sources.items():
                (repository / name).write_text(text)
            output = repository / ".runtime/semantic"
            projection = ({"worker_query": "authored-explicit-empty-capsule-projection", "worker_capsule_limit": 0}
                          if case_name == "zero_rows" else {})
            prepare_semantic_context(repository=repository, paths=sorted(sources),
                required_raw_paths=sorted(sources), objective="Inspect authored source only.",
                task_id=TASK_SENTINEL, output=output, **projection)
            artifact = output / "worker-context.json"
            references = build_text_context_references(artifact.read_text(),
                reference_prefix="semantic-context", kind="semantic-context",
                path=artifact.relative_to(repository).as_posix(), repository_id="repo:authored-census",
                tree_id="tree:authored-census", required=True, chunk_bytes=1201)
            compiled = ContextCompiler(ContextBudget(max_input_tokens=131072, max_items=128,
                max_item_bytes=16384, max_serialized_bytes=262144)).compile(
                repository_id="repo:authored-census", tree_id="tree:authored-census",
                objective_id=TASK_SENTINEL, objective_revision="sha256:task",
                policy_id="policy:authored", policy_revision="sha256:policy",
                caller="supervisor:authored-census", stage="implementation",
                goal={"id": TASK_SENTINEL}, authority={"mode": "candidate_only", "completion_authority": False},
                scope={"allowed_paths": sorted(sources)}, acceptance={"criteria": ["pending native validation"]},
                evidence=references)
            prompt = render_context_capsule(compiled.capsule) + "\nLiteral private owner guidance.\n"
            encoded = native_codec.encode_semantic_router_prompt(prompt=prompt, repository=repository)
            result[case_name] = (repository, encoded, sources)
    return result


def observe(encoded, prefix=PREFIX, suffix=SUFFIX):
    view = view_codec.project_semantic_metadata_view(encoded.provider_prompt)
    report = census.census_semantic_metadata_view(view=view,
        native_complete_prompt=prefix + view.native_prompt + suffix,
        candidate_complete_prompt=prefix + view.provider_prompt + suffix)
    return view, report


def test_real_native_positive_round_trip_closed_export_and_original_inputs(native_cases):
    repository, encoded, sources = native_cases["many"]
    snapshot = (encoded.native_prompt, encoded.provider_prompt, encoded.receipt_json, encoded.table)
    view, report = observe(encoded)
    assert report["schema"] == census.CENSUS_SCHEMA
    assert report["exact_restoration_verified"] is True
    assert report["selected_mode"] == "common-bindings@1"
    assert report["eligible_for_matched_input_comparison"] is True
    assert report["native_complete_within_bound"] and report["candidate_complete_within_bound"]
    assert report["candidate_complete_byte_reduction"] > 0
    assert report["selected_complete_byte_reduction"] == report["candidate_complete_byte_reduction"]
    assert report["token_proxy"] == "utf8-bytes-ceil-div4@1"
    assert report["actual_provider_tokens_measured"] is False
    assert report["total_token_savings_qualified"] is False
    assert report["task_id_sha256"] == hashlib.sha256(TASK_SENTINEL.encode()).hexdigest()
    wire = json.loads(encoded.provider_prompt)
    for field in ("translation_cid", "scope_cid", "semantic_root_cid"):
        value = wire[field] if field == "translation_cid" else wire["translated_semantic"][field]
        assert report[field + "_sha256"] == hashlib.sha256(value.encode()).hexdigest()
        assert value not in canonical(report)
    assert (encoded.native_prompt, encoded.provider_prompt, encoded.receipt_json, encoded.table) == snapshot
    restored = view_codec.restore_semantic_metadata_view(provider_prompt=view.provider_prompt, receipt=view.receipt)
    assert restored == encoded.provider_prompt
    assert native_codec.restore_semantic_router_prompt(provider_prompt=restored,
        table=encoded.table, repository=repository) == encoded.native_prompt
    assert all((repository / name).read_text() == text for name, text in sources.items())
    serialized = canonical(report)
    for marker in (TASK_SENTINEL, BODY_SENTINEL, PREFIX.strip(), SUFFIX.strip(), *sources):
        assert marker not in serialized


def test_explicit_false_authority_is_strict_boolean_and_no_provider_calls(native_cases):
    _view, report = observe(native_cases["many"][1])
    for field in ("source_freshness_verified", "program_semantics_proved", "proof_authority", "omission_authority",
                  "execution_authority", "dispatch_authority", "completion_authority", "publication_authority"):
        assert report[field] is False
        assert type(report[field]) is bool
    assert type(report["provider_calls"]) is int and report["provider_calls"] == 0
    assert report["representation_only"] is True and report["candidate_only"] is True


def test_real_one_row_fallback_with_measured_counts(native_cases):
    view, report = observe(native_cases["one_row"][1])
    assert report["capsule_count"] == report["admission_count"] == 1
    assert report["captured_source_count"] == report["captured_source_group_count"] == 1
    assert report["capsule_source_group_count"] == report["admission_source_group_count"] == 1
    assert report["selected_mode"] == "legacy"
    assert report["fallback_reason"] == "complete_input_not_smaller_under_bytes_and_proxy"
    assert report["candidate_complete_bytes"] > report["native_complete_bytes"]
    assert report["candidate_complete_byte_reduction"] < 0
    assert report["selected_complete_byte_reduction"] == report["selected_complete_proxy_reduction"] == 0
    assert not report["eligible_for_matched_input_comparison"]
    assert report["shared_field_count"] == 0
    assert all(not fields for fields in report["shared_field_names"].values())
    assert view_codec.restore_semantic_metadata_view(provider_prompt=view.provider_prompt,
        receipt=view.receipt) == view.native_prompt


def test_real_empty_capsule_projection_falls_back_with_raw_sources_preserved(native_cases):
    _repository, encoded, sources = native_cases["zero_rows"]
    view, report = observe(encoded)
    semantic = json.loads(encoded.provider_prompt)["translated_semantic"]
    assert semantic["capsules"] == semantic["admissions"] == []
    assert semantic["raw_sources"] == sources
    assert report["capsule_count"] == report["admission_count"] == 0
    assert report["captured_source_count"] == report["captured_source_group_count"] == 1
    assert report["capsule_source_group_count"] == report["admission_source_group_count"] == 0
    assert report["selected_mode"] == "legacy"
    assert not report["eligible_for_matched_input_comparison"]
    assert view_codec.restore_semantic_metadata_view(provider_prompt=view.provider_prompt,
        receipt=view.receipt) == encoded.provider_prompt


@pytest.mark.parametrize("case_name,groups", [("heterogeneous", 2), ("same_content", 1)])
def test_real_source_group_counts_distinguish_paths_from_content(native_cases, case_name, groups):
    view, report = observe(native_cases[case_name][1])
    assert report["captured_source_count"] == 2
    assert report["captured_source_group_count"] == groups
    assert report["capsule_source_group_count"] == groups
    assert report["admission_source_group_count"] == groups
    assert report["capsule_count"] == report["admission_count"] == 4
    common = json.loads(view.provider_prompt)["common_bindings"]
    assert report["shared_field_names"] == {group: sorted(fields) for group, fields in common.items()}
    assert report["shared_field_counts"] == {group: len(fields) for group, fields in common.items()}
    assert report["shared_field_count"] == sum(map(len, common.values()))
    if groups == 2:
        assert "source_cid" not in report["shared_field_names"]["capsules"]
        assert "source_cid" not in report["shared_field_names"]["admission_refs"]


def test_unknown_secret_bearing_fields_never_expand_the_export(native_cases):
    _repository, encoded, _sources = native_cases["many"]
    baseline_view, baseline = observe(encoded)
    wire = json.loads(encoded.provider_prompt)
    marker = "UNKNOWN_SECRET_FIELD_SENTINEL"
    wire["native_context"]["secret_plan"] = {"password": marker}
    wire["translated_semantic"]["unknown_evaluator_text"] = marker
    for row in wire["translated_semantic"]["capsules"]:
        row["unknown_sensitive_name_" + marker] = {"api_key": marker, "source_path": marker}
    for row in wire["translated_semantic"]["admissions"]:
        row["unknown_provider_body"] = marker
    view = view_codec.project_semantic_metadata_view(canonical(wire))
    report = census.census_semantic_metadata_view(view=view,
        native_complete_prompt=PREFIX + view.native_prompt + SUFFIX,
        candidate_complete_prompt=PREFIX + view.provider_prompt + SUFFIX)
    assert set(report) == set(baseline)
    assert report["shared_field_names"] == baseline["shared_field_names"]
    for text in ("secret_plan", "password", "api_key", "unknown_evaluator_text", marker, "source_path"):
        assert text not in canonical(report)
    assert view_codec.restore_semantic_metadata_view(provider_prompt=view.provider_prompt,
        receipt=view.receipt) == canonical(wire)
    assert baseline_view.native_prompt == encoded.provider_prompt


@pytest.mark.parametrize("mutation", ["view", "native", "receipt", "unknown_receipt", "authority_zero"])
def test_tampered_view_or_receipt_is_refused_without_text_export(native_cases, mutation):
    view, _report = observe(native_cases["many"][1])
    marker = "NEVER_EXPORT_TAMPER_SENTINEL"
    if mutation == "view":
        altered = replace(view, provider_prompt=view.provider_prompt + marker)
    elif mutation == "native":
        altered = replace(view, native_prompt=view.native_prompt + marker)
    else:
        receipt = view.receipt
        if mutation == "receipt":
            receipt["candidate_view_sha256"] = "0" * 64
        elif mutation == "unknown_receipt":
            receipt["private_password"] = marker
        else:
            receipt["proof_authority"] = 0
        altered = replace(view, receipt_json=canonical(receipt))
    with pytest.raises(census.SemanticMetadataCensusError) as error:
        census.census_semantic_metadata_view(view=altered,
            native_complete_prompt=PREFIX + altered.native_prompt + SUFFIX,
            candidate_complete_prompt=PREFIX + altered.provider_prompt + SUFFIX)
    assert marker not in str(error.value)


@pytest.mark.parametrize("boundary", ["prefix", "suffix", "duplicate", "missing"])
def test_complete_boundary_tampering_is_refused(native_cases, boundary):
    view, _report = observe(native_cases["many"][1])
    native, candidate = PREFIX + view.native_prompt + SUFFIX, PREFIX + view.provider_prompt + SUFFIX
    if boundary == "prefix":
        candidate = "changed:" + candidate
    elif boundary == "suffix":
        candidate += ":changed"
    elif boundary == "duplicate":
        native += view.native_prompt
    else:
        native = "missing body"
    with pytest.raises(census.SemanticMetadataCensusError):
        census.census_semantic_metadata_view(view=view,
            native_complete_prompt=native, candidate_complete_prompt=candidate)


@pytest.mark.parametrize("limit", [True, False, 0, 255999, 256001, 256000.0, "256000", None])
def test_bound_is_a_fixed_typed_runtime_observation(native_cases, limit):
    view, _report = observe(native_cases["many"][1])
    with pytest.raises(census.SemanticMetadataCensusError, match="fixed"):
        census.census_semantic_metadata_view(view=view, native_complete_prompt=view.native_prompt,
            candidate_complete_prompt=view.provider_prompt, fixed_max_input_bytes=limit)


def test_native_over_bound_candidate_within_is_not_eligible(native_cases):
    view, _report = observe(native_cases["many"][1], prefix="", suffix="")
    pad = "P" * (census.FIXED_MAX_INPUT_BYTES - len(view.native_prompt.encode()) + 1)
    report = census.census_semantic_metadata_view(view=view,
        native_complete_prompt=pad + view.native_prompt, candidate_complete_prompt=pad + view.provider_prompt)
    assert report["selected_mode"] == "common-bindings@1"
    assert report["native_complete_bytes"] == census.FIXED_MAX_INPUT_BYTES + 1
    assert report["native_complete_within_bound"] is False
    assert report["candidate_complete_within_bound"] is True
    assert report["selected_complete_within_bound"] is True
    assert report["eligible_for_matched_input_comparison"] is False
    assert report["dispatch_authority"] is False


def test_expanding_candidate_exceeding_bound_keeps_original_and_is_not_eligible(native_cases):
    view, _report = observe(native_cases["one_row"][1], prefix="", suffix="")
    pad = "P" * (census.FIXED_MAX_INPUT_BYTES - len(view.native_prompt.encode()))
    report = census.census_semantic_metadata_view(view=view,
        native_complete_prompt=pad + view.native_prompt, candidate_complete_prompt=pad + view.provider_prompt)
    assert report["native_complete_within_bound"] is True
    assert report["candidate_complete_within_bound"] is False
    assert report["selected_mode"] == "legacy" and report["selected_complete_within_bound"] is True
    assert report["eligible_for_matched_input_comparison"] is False


def test_foreign_source_group_cannot_be_nominated_as_captured(native_cases):
    _repository, encoded, _sources = native_cases["many"]
    wire = json.loads(encoded.provider_prompt)
    wire["translated_semantic"]["capsules"][0]["source_cid"] = "NOT_CAPTURED_PRIVATE_SOURCE_SENTINEL"
    view = view_codec.project_semantic_metadata_view(canonical(wire))
    with pytest.raises(census.SemanticMetadataCensusError, match="source groups"):
        census.census_semantic_metadata_view(view=view, native_complete_prompt=view.native_prompt,
            candidate_complete_prompt=view.provider_prompt)


def test_unknown_alias_source_group_is_refused(native_cases):
    wire = json.loads(native_cases["many"][1].provider_prompt)
    wire["translated_semantic"]["capsules"][-1]["source_cid"] = {"$semantic_ref": "UNKNOWN_PRIVATE_ALIAS"}
    view = view_codec.project_semantic_metadata_view(canonical(wire))
    with pytest.raises(census.SemanticMetadataCensusError, match="source reference"):
        census.census_semantic_metadata_view(view=view, native_complete_prompt=view.native_prompt,
            candidate_complete_prompt=view.provider_prompt)


def test_rehashed_lossless_alternate_factoring_is_not_the_fixed_view(native_cases):
    view, _report = observe(native_cases["many"][1])
    wire = json.loads(view.provider_prompt)
    value = wire["common_bindings"]["capsules"].pop("extractor_version")
    for row in wire["translated_semantic"]["capsules"]:
        row["extractor_version"] = value
    candidate = canonical(wire)
    receipt = view.receipt
    receipt["candidate_view_sha256"] = hashlib.sha256(candidate.encode()).hexdigest()
    receipt["candidate_view_bytes"] = len(candidate.encode())
    receipt["candidate_view_proxy_tokens"] = (len(candidate.encode()) + 3) // 4
    receipt["common_bindings_sha256"] = hashlib.sha256(canonical(wire["common_bindings"]).encode()).hexdigest()
    receipt["shared_field_paths"] = [[group, field] for group in sorted(wire["common_bindings"])
                                     for field in sorted(wire["common_bindings"][group])]
    alternate = replace(view, provider_prompt=candidate, receipt_json=canonical(receipt))
    assert view_codec.restore_semantic_metadata_view(provider_prompt=alternate.provider_prompt,
        receipt=alternate.receipt) == view.native_prompt
    with pytest.raises(census.SemanticMetadataCensusError, match="projected candidate"):
        census.census_semantic_metadata_view(view=alternate, native_complete_prompt=alternate.native_prompt,
            candidate_complete_prompt=alternate.provider_prompt)


def test_untyped_view_and_nonstring_complete_prompts_are_refused(native_cases):
    view, _report = observe(native_cases["many"][1])
    for value in (view.receipt, None, "view body", object()):
        with pytest.raises(census.SemanticMetadataCensusError, match="typed"):
            census.census_semantic_metadata_view(view=value, native_complete_prompt=view.native_prompt,
                candidate_complete_prompt=view.provider_prompt)
    for value in (None, {}, 256000):
        with pytest.raises(census.SemanticMetadataCensusError):
            census.census_semantic_metadata_view(view=view, native_complete_prompt=value,
                candidate_complete_prompt=view.provider_prompt)
