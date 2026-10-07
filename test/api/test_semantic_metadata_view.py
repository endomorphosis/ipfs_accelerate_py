"""Authored real-producer fixtures for an exact metadata representation view."""
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudget, canonical_context_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_metadata_view as view_codec
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as native_codec
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata


def canonical(value):
    return canonical_context_json_bytes(value).decode()


@pytest.fixture
def native_factory(tmp_path):
    def build(function_count=6):
        repository = tmp_path / ("authored-repository-" + str(function_count))
        repository.mkdir()
        source = "# Literal common_bindings and $semantic_ref stay source text.\n"
        if function_count:
            source += "ROOT_TEXT = 'error, cancelled, inconclusive, unsupported; café'\n"
        for index in range(function_count):
            source += (f"\ndef authored_function_{index}(value):\n"
                       "    '''The literal schema and source_cid are not metadata slots.'''\n"
                       f"    return value + {index}\n")
        (repository / "module.py").write_text(source)
        output = repository / ".runtime/semantic"
        with public_replay_without_metadata():
            prepare_semantic_context(repository=repository, paths=["module.py"],
                required_raw_paths=["module.py"], objective="Review authored functions.",
                task_id="AUTHORED-METADATA-TASK", output=output)
            artifact = output / "worker-context.json"
            references = build_text_context_references(artifact.read_text(),
                reference_prefix="semantic-context", kind="semantic-context",
                path=artifact.relative_to(repository).as_posix(), repository_id="repo:authored",
                tree_id="tree:authored", required=True, chunk_bytes=1201)
            compiled = ContextCompiler(ContextBudget(max_input_tokens=131072, max_items=128,
                max_item_bytes=16384, max_serialized_bytes=262144)).compile(
                repository_id="repo:authored", tree_id="tree:authored",
                objective_id="AUTHORED-METADATA-TASK", objective_revision="sha256:task",
                policy_id="policy:authored", policy_revision="sha256:policy",
                caller="supervisor:authored", stage="implementation",
                goal={"id": "AUTHORED-METADATA-TASK"},
                authority={"mode": "candidate_only", "completion_authority": False},
                scope={"allowed_paths": ["module.py"]},
                acceptance={"criteria": ["pending native validation"]}, evidence=references)
            prompt = render_context_capsule(compiled.capsule) + "\nLiteral authorized guidance: common_bindings, $semantic_ref.\n"
            encoded = native_codec.encode_semantic_router_prompt(prompt=prompt, repository=repository)
        return repository, encoded, source
    return build


@pytest.fixture
def native(native_factory):
    return native_factory(6)


def test_real_producer_full_transport_and_native_prompt_round_trip(native):
    repository, encoded, source = native
    candidate = view_codec.project_semantic_metadata_view(encoded.provider_prompt)
    restored = view_codec.restore_semantic_metadata_view(provider_prompt=candidate.provider_prompt,
                                                         receipt=candidate.receipt)
    assert restored == encoded.provider_prompt
    assert native_codec.restore_semantic_router_prompt(provider_prompt=restored,
        table=encoded.table, repository=repository) == encoded.native_prompt
    original, projected = json.loads(encoded.provider_prompt), json.loads(candidate.provider_prompt)
    assert projected["translated_semantic"]["raw_sources"]["module.py"] == source
    for key in ("native_context", "native_suffix", "translation_table", "translation_cid", "instructions",
                "execution_authority", "completion_authority"):
        assert projected[key] == original[key]
    assert projected["schema"] == view_codec.VIEW_SCHEMA
    assert projected["source_transport_schema"] == native_codec.TRANSPORT_SCHEMA
    assert projected["metadata_instructions"] == view_codec.METADATA_INSTRUCTIONS
    with pytest.raises(native_codec.SemanticTranslationError):
        native_codec.restore_semantic_router_prompt(provider_prompt=candidate.provider_prompt,
            table=encoded.table, repository=repository)


def test_actual_allowlist_and_every_value_remain_inline(native):
    _repository, encoded, _source = native
    original = json.loads(encoded.provider_prompt)["translated_semantic"]
    projected = json.loads(view_codec.project_semantic_metadata_view(encoded.provider_prompt).provider_prompt)
    semantic, common = projected["translated_semantic"], projected["common_bindings"]
    assert set(common["capsules"]) == view_codec.CAPSULE_FIELDS
    assert set(common["admissions"]) == view_codec.ADMISSION_FIELDS
    assert set(common["admission_refs"]) == view_codec.ADMISSION_REF_FIELDS
    for original_row, row in zip(original["capsules"], semantic["capsules"]):
        assert {**common["capsules"], **row} == original_row
        for field in ("signature", "source_slice_path", "metadata", "confidence", "docstring_hint"):
            assert row[field] == original_row[field]
    for original_row, row in zip(original["admissions"], semantic["admissions"]):
        assert {**common["admissions"], **row, "ref": {**common["admission_refs"], **row["ref"]}} == original_row
        for field in ("admission", "caveats", "assessment_cid"):
            assert row[field] == original_row[field]
        for field in ("confidence", "validity_bindings", "capsule_cid", "stable_symbol_id", "version_cid"):
            assert row["ref"][field] == original_row["ref"][field]


def test_receipt_binds_native_core_source_task_and_roots_without_authority(native):
    _repository, encoded, _source = native
    candidate = view_codec.project_semantic_metadata_view(encoded.provider_prompt)
    receipt, table = candidate.receipt, encoded.table.to_dict()
    for key in ("translation_cid", "task_id", "scope_cid", "semantic_root_cid", "native_core_sha256", "native_suffix_sha256"):
        assert receipt[key] == (encoded.table.translation_cid if key == "translation_cid" else table[key])
    assert receipt["source_manifest_sha256"] == hashlib.sha256(canonical(table["source_manifest"]).encode()).hexdigest()
    assert receipt["native_transport_sha256"] == hashlib.sha256(encoded.provider_prompt.encode()).hexdigest()
    assert receipt["candidate_view_sha256"] == hashlib.sha256(candidate.provider_prompt.encode()).hexdigest()
    assert receipt["candidate_only"]
    for field in ("source_freshness_verified", "program_semantics_proved", "omission_authority", "proof_authority",
                  "execution_authority", "completion_authority", "publication_authority"):
        assert receipt[field] is False
    detached = candidate.receipt
    detached["task_id"] = "foreign"
    assert candidate.receipt["task_id"] == table["task_id"]


def test_complete_input_selection_counts_all_literal_prefix_and_suffix(native):
    _repository, encoded, _source = native
    candidate = view_codec.project_semantic_metadata_view(encoded.provider_prompt)
    prefix = "Literal allocated workspace contract.\n"
    suffix = "\nDoctor uncertainty remains literal.\nPublic instruction café.\nReply contract."
    original, projected = prefix + encoded.provider_prompt + suffix, prefix + candidate.provider_prompt + suffix
    selected = view_codec.select_semantic_metadata_view(view=candidate,
        native_complete_prompt=original, candidate_complete_prompt=projected)
    assert selected.selected_mode == "common-bindings@1"
    assert selected.selected_prompt == projected
    receipt = selected.receipt
    assert receipt["native_complete_bytes"] == len(original.encode())
    assert receipt["candidate_complete_bytes"] == len(projected.encode())
    assert receipt["selected_complete_sha256"] == hashlib.sha256(projected.encode()).hexdigest()
    assert receipt["token_proxy"] == "utf8-bytes-ceil-div4@1"
    assert receipt["candidate_complete_proxy_tokens"] < receipt["native_complete_proxy_tokens"]
    assert receipt["fallback_reason"] is None


def test_real_small_fixture_falls_back_without_altering_native_input(native_factory):
    _repository, encoded, _source = native_factory(0)
    candidate = view_codec.project_semantic_metadata_view(encoded.provider_prompt)
    semantic = json.loads(encoded.provider_prompt)["translated_semantic"]
    assert len(semantic["capsules"]) == len(semantic["admissions"]) == 1
    assert len(candidate.provider_prompt.encode()) > len(encoded.provider_prompt.encode())
    selection = view_codec.select_semantic_metadata_view(view=candidate,
        native_complete_prompt=encoded.provider_prompt, candidate_complete_prompt=candidate.provider_prompt)
    assert selection.selected_mode == "legacy"
    assert selection.selected_prompt == encoded.provider_prompt
    assert selection.receipt["fallback_reason"] == "complete_input_not_smaller_under_bytes_and_proxy"
    assert selection.receipt["native_complete_bytes"] == len(encoded.provider_prompt.encode())
    assert selection.receipt["candidate_complete_bytes"] == len(candidate.provider_prompt.encode())
    assert view_codec.restore_semantic_metadata_view(provider_prompt=candidate.provider_prompt,
        receipt=candidate.receipt) == encoded.provider_prompt


@pytest.mark.parametrize("changed", ["prefix", "suffix", "duplicate", "missing"])
def test_complete_selection_rejects_changed_or_ambiguous_literal_boundaries(native, changed):
    _repository, encoded, _source = native
    candidate = view_codec.project_semantic_metadata_view(encoded.provider_prompt)
    original, projected = "prefix:" + encoded.provider_prompt + ":suffix", "prefix:" + candidate.provider_prompt + ":suffix"
    if changed == "prefix":
        projected = "foreign:" + candidate.provider_prompt + ":suffix"
    elif changed == "suffix":
        projected += ":changed"
    elif changed == "duplicate":
        original += encoded.provider_prompt
    else:
        original = "missing transport"
    with pytest.raises(view_codec.SemanticMetadataViewError):
        view_codec.select_semantic_metadata_view(view=candidate,
            native_complete_prompt=original, candidate_complete_prompt=projected)


def test_original_at1_at2_are_unchanged_and_at2_is_refused(native):
    repository, encoded, _source = native
    legacy_again = native_codec.encode_semantic_router_prompt(prompt=encoded.native_prompt, repository=repository)
    compact = native_codec.encode_semantic_router_prompt(prompt=encoded.native_prompt, repository=repository,
        transport_schema=native_codec.COMPACT_TRANSPORT_SCHEMA)
    snapshot = encoded.receipt_json
    view_codec.project_semantic_metadata_view(encoded.provider_prompt)
    assert encoded == legacy_again and encoded.receipt_json == snapshot
    assert compact.table == encoded.table
    with pytest.raises(view_codec.SemanticMetadataViewError, match="@1"):
        view_codec.project_semantic_metadata_view(compact.provider_prompt)


def test_unknown_fields_statuses_and_structured_literals_stay_row_local(native):
    _repository, encoded, _source = native
    wire = json.loads(encoded.provider_prompt)
    for row in wire["translated_semantic"]["capsules"]:
        row["unknown_literal_field"] = {"common_bindings": "literal café", "schema": "literal schema"}
        row["status"] = "cancelled"
    for index, row in enumerate(wire["translated_semantic"]["admissions"]):
        row["uncertainty_status"] = ["inconclusive", "error", "cancelled", "unsupported"][index % 4]
        row["ref"]["unknown_flag"] = True
    original = canonical(wire)
    candidate = view_codec.project_semantic_metadata_view(original)
    assert view_codec.restore_semantic_metadata_view(provider_prompt=candidate.provider_prompt,
        receipt=candidate.receipt) == original
    projected = json.loads(candidate.provider_prompt)["translated_semantic"]
    assert [row["status"] for row in projected["capsules"]] == ["cancelled"] * len(projected["capsules"])
    assert all("unknown_literal_field" in row for row in projected["capsules"])
    assert all("uncertainty_status" in row and "unknown_flag" in row["ref"] for row in projected["admissions"])


def test_nonidentical_or_missing_binding_fields_are_never_factored(native):
    _repository, encoded, _source = native
    wire = json.loads(encoded.provider_prompt)
    wire["translated_semantic"]["capsules"][0]["extractor_version"] = "different"
    del wire["translated_semantic"]["capsules"][1]["capsule_compiler_version"]
    wire["translated_semantic"]["admissions"][0]["freshness"] = "unknown"
    wire["translated_semantic"]["admissions"][0]["ref"]["raw_source_required"] = False
    original = canonical(wire)
    candidate = view_codec.project_semantic_metadata_view(original)
    common = json.loads(candidate.provider_prompt)["common_bindings"]
    assert "extractor_version" not in common["capsules"]
    assert "capsule_compiler_version" not in common["capsules"]
    assert "freshness" not in common["admissions"]
    assert "raw_source_required" not in common["admission_refs"]
    assert view_codec.restore_semantic_metadata_view(provider_prompt=candidate.provider_prompt,
        receipt=candidate.receipt) == original


@pytest.mark.parametrize("mutation", ["common", "row", "task", "scope", "root", "manifest", "core", "literal",
                                     "alias", "instructions", "authority", "unknown_top"])
def test_candidate_tampering_refuses_exact_restore(native, mutation):
    _repository, encoded, _source = native
    candidate = view_codec.project_semantic_metadata_view(encoded.provider_prompt)
    wire = json.loads(candidate.provider_prompt)
    if mutation == "common":
        wire["common_bindings"]["capsules"]["schema"] = "foreign"
    elif mutation == "row":
        wire["translated_semantic"]["capsules"][0]["signature"] = {"changed": True}
    elif mutation in {"task", "scope", "root"}:
        wire["translated_semantic"][{"task": "task_id", "scope": "scope_cid", "root": "semantic_root_cid"}[mutation]] = "foreign"
    elif mutation == "manifest":
        wire["translated_semantic"]["manifest"]["module.py"]["sha256"] = "0" * 64
    elif mutation == "core":
        wire["native_context"]["scope"] = {"allowed_paths": ["foreign.py"]}
    elif mutation == "literal":
        wire["translated_semantic"]["raw_sources"]["module.py"] += "# changed literal\n"
    elif mutation == "alias":
        wire["translation_table"]["s0"] = "foreign"
    elif mutation == "instructions":
        wire["metadata_instructions"] = "foreign"
    elif mutation == "authority":
        wire["completion_authority"] = True
    else:
        wire["common_bindings_reserved_collision"] = {}
    with pytest.raises(view_codec.SemanticMetadataViewError):
        view_codec.restore_semantic_metadata_view(provider_prompt=canonical(wire), receipt=candidate.receipt)


@pytest.mark.parametrize("field", ["task_id", "scope_cid", "semantic_root_cid", "translation_cid",
    "source_manifest_sha256", "native_core_sha256", "native_context_sha256", "native_suffix_sha256",
    "original_translated_semantic_sha256", "native_transport_sha256", "candidate_view_sha256",
    "common_bindings_sha256", "shared_field_paths", "capsule_count", "candidate_view_proxy_tokens",
    "proof_authority", "candidate_only", "unknown"])
def test_receipt_mutation_or_foreign_binding_refuses_restore(native, field):
    _repository, encoded, _source = native
    candidate = view_codec.project_semantic_metadata_view(encoded.provider_prompt)
    receipt = candidate.receipt
    if field.endswith("sha256"):
        receipt[field] = "0" * 64
    elif field in {"capsule_count", "candidate_view_proxy_tokens"}:
        receipt[field] += 1
    elif field == "shared_field_paths":
        receipt[field] = []
    elif field == "proof_authority":
        receipt[field] = True
    elif field == "candidate_only":
        receipt[field] = False
    else:
        receipt[field] = "foreign"
    with pytest.raises(view_codec.SemanticMetadataViewError):
        view_codec.restore_semantic_metadata_view(provider_prompt=candidate.provider_prompt, receipt=receipt)


def test_even_rehashed_receipt_cannot_allow_unknown_common_fields_or_row_overrides(native):
    _repository, encoded, _source = native
    candidate = view_codec.project_semantic_metadata_view(encoded.provider_prompt)
    for mutation in ("unknown_common", "row_override", "unknown_reference"):
        wire = json.loads(candidate.provider_prompt)
        if mutation == "unknown_common":
            wire["common_bindings"]["capsules"]["signature"] = {}
        elif mutation == "row_override":
            wire["translated_semantic"]["capsules"][0]["schema"] = wire["common_bindings"]["capsules"]["schema"]
        else:
            wire["common_bindings"]["capsules"]["source_cid"] = {"$semantic_ref": "unknown"}
        altered = canonical(wire)
        receipt = candidate.receipt
        receipt["candidate_view_sha256"] = hashlib.sha256(altered.encode()).hexdigest()
        receipt["candidate_view_bytes"] = len(altered.encode())
        receipt["candidate_view_proxy_tokens"] = (len(altered.encode()) + 3) // 4
        receipt["common_bindings_sha256"] = hashlib.sha256(canonical(wire["common_bindings"]).encode()).hexdigest()
        with pytest.raises(view_codec.SemanticMetadataViewError):
            view_codec.restore_semantic_metadata_view(provider_prompt=altered, receipt=receipt)


@pytest.mark.parametrize("value", ['{"schema":1,"schema":1}', '{"x":NaN}', '{"x":Infinity}',
                                  '{"x":1e999}', '{} ', '[]', '"literal"'])
def test_duplicate_nonfinite_noncanonical_and_nontransport_inputs_refused(value):
    with pytest.raises(view_codec.SemanticMetadataViewError):
        view_codec.project_semantic_metadata_view(value)


def test_oversized_deep_and_wrong_row_shapes_refused(native):
    _repository, encoded, _source = native
    with pytest.raises(view_codec.SemanticMetadataViewError, match="bound"):
        view_codec.project_semantic_metadata_view(" " * (view_codec.MAX_BYTES + 1))
    wire = json.loads(encoded.provider_prompt)
    nested = {}
    for _ in range(view_codec.MAX_DEPTH + 1):
        nested = {"nested": nested}
    wire["native_context"]["too_deep"] = nested
    with pytest.raises(view_codec.SemanticMetadataViewError, match="structure bound"):
        view_codec.project_semantic_metadata_view(canonical(wire))
    for mutation in ("row", "ref", "count", "raw_flag", "task"):
        wire = json.loads(encoded.provider_prompt)
        if mutation == "row":
            wire["translated_semantic"]["capsules"][0] = "not a row"
        elif mutation == "ref":
            wire["translated_semantic"]["admissions"][0]["ref"] = None
        elif mutation == "count":
            wire["translated_semantic"]["capsules"].pop()
        elif mutation == "raw_flag":
            for row in wire["translated_semantic"]["admissions"]:
                row["ref"]["raw_source_required"] = 1
        else:
            wire["native_context"]["objective_id"] = "foreign"
        with pytest.raises(view_codec.SemanticMetadataViewError):
            view_codec.project_semantic_metadata_view(canonical(wire))


def test_large_integer_and_node_population_fail_with_closed_errors():
    with pytest.raises(view_codec.SemanticMetadataViewError):
        view_codec.project_semantic_metadata_view('{"large":' + "1" * 5000 + "}")
    with pytest.raises(view_codec.SemanticMetadataViewError, match="structure bound"):
        view_codec.project_semantic_metadata_view(canonical({"many": [0] * view_codec.MAX_NODES}))


def test_nonfinite_or_noncanonical_receipt_data_is_refused(native):
    _repository, encoded, _source = native
    candidate = view_codec.project_semantic_metadata_view(encoded.provider_prompt)
    for value in (float("nan"), float("inf"), object()):
        receipt = candidate.receipt
        receipt["capsule_count"] = value
        with pytest.raises(view_codec.SemanticMetadataViewError):
            view_codec.restore_semantic_metadata_view(provider_prompt=candidate.provider_prompt, receipt=receipt)
