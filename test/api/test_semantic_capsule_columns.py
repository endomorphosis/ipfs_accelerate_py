"""Exact typed presentation, complete-input gates and hostile expansion bounds."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import semantic_capsule_columns as columns
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_metadata_view as metadata
from test.api.test_semantic_metadata_view import native_factory  # noqa: F401


def canonical(value):
    return metadata._json(value)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def transport(count=6):
    capsules, admissions = [], []
    for index in range(count):
        capsule = dict(schema="ipfs-datasets.software-contracts.semantic-capsule@1",
            stable_symbol_id=f"symbol:{index}", version_cid=f"version:{index}", semantic_index_schema="index@1",
            extractor_version="extractor@1", capsule_schema="capsule@1", capsule_compiler_version="compiler@1",
            source_slice_path="module.py", source_cid="source:module", symbol_fact_cid=f"fact:{index}",
            signature={"parameters": ["value"], "return": "int"}, annotations={}, defaults={}, decorators=[],
            contracts={}, effects=[], exception_behavior={}, schema_relations=[], serialization_relations=[],
            test_refs=[], fixture_refs=[], proof_obligation_refs=[f"pending:{index}"], dependency_stable_ids=[],
            dependency_version_cids=[], dependency_fact_cids=[], dependency_link_ids=[], confidence="exact",
            relevant_binding_projection_cid=None, docstring_hint=None,
            metadata={"facets": [{"version_evidence": {"literal": f"café Ω {index}",
                        "nested_markers": {"schema": "literal", "columns": ["shared", "$semantic_ref"]}}}]},
            capsule_cid=f"capsule:{index}")
        ref = dict(capsule_cid=f"capsule:{index}", semantic_state_root_cid="root:original",
            stable_symbol_id=f"symbol:{index}", version_cid=f"version:{index}", source_cid="source:module",
            confidence="exact", validity_bindings=[], raw_source_required=True)
        admission = dict(schema="ipfs-accelerate.semantic-capsule-admission@1", interface="SemanticCapsuleAdmission@1",
            ref=ref, admission="raw_source_required", freshness="fresh", caveats=["Read exact source before editing."],
            assessment_cid=f"assessment:{index}")
        capsules.append(capsule)
        admissions.append(admission)
    semantic = dict(task_id="task:columns", scope_cid="scope:original", semantic_root_cid="root:original",
        manifest={"module.py": {"sha256": "a" * 64, "source_cid": "source:module"}},
        capsules=capsules, admissions=admissions, required_raw_paths=["module.py"],
        raw_sources={"module.py": "# constraints, source whitespace and $semantic_ref stay exact\nVALUE = 'café'\n"},
        worker_projection={"query": "Keep all constraints.", "selected_symbols": list(range(count))},
        pack={"references": [], "assumptions": ["all obligations pending"]},
        unknown_native={"literal": "shared/columns/rows/extras are literal here"})
    return dict(schema=metadata.TRANSPORT_SCHEMA, translation_cid="translation:original", translation_table={},
        native_context={"objective_id": "task:columns", "evidence": [], "goal": {"constraint": "Never change source scope"},
                        "authority": {"completion_authority": False}}, translated_semantic=semantic,
        native_suffix="Original required instruction and validation contract remain literal Ω.\n",
        instructions="Resolve only native semantic identifiers.", execution_authority=False, completion_authority=False)


@pytest.fixture
def view():
    return columns.project_capsule_columns(canonical(transport()))


def _complete(view, *, suffix="\nRequired complete suffix Ω\n"):
    prefix = "Allocated workspace and exact public requirements:\n"
    return prefix + view.native_prompt + suffix, prefix + view.provider_prompt + suffix


def _counter(call, *, provider="codex_cli", model="qualified-local", qualification="estimate"):
    return metadata.SemanticMetadataTokenCounter(call, provider, model, "local-counter", "sha256:local-pinned",
                                                qualification)


def _reseal_candidate(view, wire):
    receipt = deepcopy(view.receipt)
    text = canonical(wire)
    receipt.update(candidate_view_sha256=digest(text), candidate_view_bytes=len(text.encode()),
        candidate_view_proxy_tokens=(len(text.encode()) + 3) // 4,
        columns_sha256=digest(canonical({k: wire["translated_semantic"][k] for k in ("capsules", "admissions")})))
    return text, receipt


def test_complete_authored_transport_round_trips_and_keeps_every_field(view):
    restored = columns.restore_capsule_columns(provider_prompt=view.provider_prompt, receipt=view.receipt)
    assert restored == canonical(transport()) == view.native_prompt
    assert json.loads(restored) == transport()
    assert view.receipt["capsule_count"] == view.receipt["admission_count"] == 6
    assert view.receipt["group_counts"]["admission_refs"]["rows"] == 6
    assert all(view.receipt[field] is False for field in metadata._FALSE_FIELDS)


def test_actual_upstream_producer_transport_and_native_prompt_round_trip(native_factory):
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as native
    repository, encoded, source = native_factory(4)
    view = columns.project_capsule_columns(encoded.provider_prompt)
    restored = columns.restore_capsule_columns(provider_prompt=view.provider_prompt, receipt=view.receipt)
    assert restored == encoded.provider_prompt
    assert native.restore_semantic_router_prompt(provider_prompt=restored, table=encoded.table,
                                                repository=repository) == encoded.native_prompt
    assert (repository / "module.py").read_text() == source
    assert view.receipt["native_transport_sha256"] == digest(encoded.provider_prompt)


def test_literal_sources_constraints_signatures_and_unknown_fields_stay_exact(view):
    original, candidate = json.loads(view.native_prompt), json.loads(view.provider_prompt)
    assert candidate["native_context"] == original["native_context"]
    assert candidate["native_suffix"] == original["native_suffix"]
    for field in ("raw_sources", "required_raw_paths", "manifest", "pack", "worker_projection", "unknown_native"):
        assert candidate["translated_semantic"][field] == original["translated_semantic"][field]
    capsule_table = candidate["translated_semantic"]["capsules"]
    assert capsule_table["shared"]["signature"] == original["translated_semantic"]["capsules"][0]["signature"]
    assert len(capsule_table["rows"]) == 6
    assert columns.restore_capsule_columns(provider_prompt=view.provider_prompt, receipt=view.receipt) == view.native_prompt


def test_unknown_fields_and_absent_null_empty_do_not_become_defaults():
    wire = transport(4)
    rows = wire["translated_semantic"]["capsules"]
    rows[0].pop("docstring_hint")
    rows[1]["docstring_hint"] = None
    rows[2]["docstring_hint"] = ""
    rows[3]["docstring_hint"] = []
    for index, row in enumerate(rows):
        row["unknown"] = {"schema": "literal", "number": index, "shared": [None, {}, [], "", False, 0]}
    admissions = wire["translated_semantic"]["admissions"]
    admissions[0]["ref"].pop("raw_source_required")
    admissions[1]["ref"]["raw_source_required"] = None
    admissions[2]["ref"]["unknown_flag"] = False
    admissions[3]["unknown_extra"] = {"rows": "literal café"}
    view = columns.project_capsule_columns(canonical(wire))
    candidate = json.loads(view.provider_prompt)["translated_semantic"]
    assert "docstring_hint" not in candidate["capsules"]["shared"]
    assert "docstring_hint" not in candidate["capsules"]["columns"]
    assert "docstring_hint" not in candidate["capsules"]["extras"][0]
    assert candidate["capsules"]["extras"][1]["docstring_hint"] is None
    assert columns.restore_capsule_columns(provider_prompt=view.provider_prompt, receipt=view.receipt) == canonical(wire)


def test_boolean_integer_and_null_have_distinct_canonical_common_values():
    wire = transport(3)
    for row, value in zip(wire["translated_semantic"]["capsules"], (False, 0, None), strict=True):
        row["defaults"] = value
    view = columns.project_capsule_columns(canonical(wire))
    table = json.loads(view.provider_prompt)["translated_semantic"]["capsules"]
    assert "defaults" in table["columns"]
    assert "defaults" not in table["shared"]
    restored = json.loads(columns.restore_capsule_columns(provider_prompt=view.provider_prompt, receipt=view.receipt))
    assert [type(r["defaults"]) for r in restored["translated_semantic"]["capsules"]] == [bool, int, type(None)]


def test_repeated_rows_and_order_are_retained_without_deduplication():
    wire = transport(3)
    wire["translated_semantic"]["capsules"].append(deepcopy(wire["translated_semantic"]["capsules"][0]))
    wire["translated_semantic"]["admissions"].append(deepcopy(wire["translated_semantic"]["admissions"][0]))
    view = columns.project_capsule_columns(canonical(wire))
    restored = json.loads(columns.restore_capsule_columns(provider_prompt=view.provider_prompt, receipt=view.receipt))
    assert [r["capsule_cid"] for r in restored["translated_semantic"]["capsules"]] == ["capsule:0", "capsule:1", "capsule:2", "capsule:0"]
    assert view.receipt["capsule_count"] == 4


@pytest.mark.parametrize("population", [0, 1])
def test_small_populations_round_trip_but_keep_original_when_overhead_is_larger(population):
    view = columns.project_capsule_columns(canonical(transport(population)))
    assert columns.restore_capsule_columns(provider_prompt=view.provider_prompt, receipt=view.receipt) == view.native_prompt
    native, candidate = _complete(view)
    result = columns.select_capsule_columns(view=view, native_complete_prompt=native, candidate_complete_prompt=candidate)
    assert result.selected_mode == "legacy"
    assert result.selected_prompt == native
    assert result.receipt["requested_mode"] == columns.MODE


def test_complete_byte_selector_counts_headers_and_literal_wrappers(view):
    native, candidate = _complete(view)
    selected = columns.select_capsule_columns(view=view, native_complete_prompt=native, candidate_complete_prompt=candidate)
    assert selected.selected_mode == columns.MODE
    assert selected.selected_prompt == candidate
    assert selected.receipt["schema"] == metadata.SELECTION_SCHEMA
    assert selected.receipt["requested_mode"] == columns.MODE
    assert selected.receipt["candidate_complete_bytes"] == len(candidate.encode())
    assert selected.receipt["native_complete_bytes"] == len(native.encode())
    assert columns.INSTRUCTIONS in candidate
    assert selected.receipt["source_freshness_verified"] is False


@pytest.mark.parametrize("qualification", ["estimate", "provider_tokenizer"])
def test_bound_counter_counts_each_complete_prompt_and_preserves_qualification(view, qualification):
    native, candidate = _complete(view)
    seen = []
    def count(text):
        seen.append(text)
        return len(text.encode())
    result = columns.select_capsule_columns(view=view, native_complete_prompt=native, candidate_complete_prompt=candidate,
        token_counter=_counter(count, qualification=qualification), provider="codex_cli", model="qualified-local")
    assert seen == [native, candidate]
    assert result.selected_mode == columns.MODE
    assert result.receipt["schema"] == metadata.TOKEN_SELECTION_SCHEMA
    assert result.receipt["token_counter_identity"]["qualification"] == qualification
    assert result.receipt["selected_complete_tokens"] == len(candidate.encode())
    assert all(result.receipt[k] is False for k in ("actual_provider_usage_verified", "provider_usage_reported",
                                                 "cli_scaffolding_measured", "complete_run_token_savings_claimed"))


def test_counter_reports_more_tokens_for_smaller_bytes_and_keeps_native(view):
    native, candidate = _complete(view)
    result = columns.select_capsule_columns(view=view, native_complete_prompt=native, candidate_complete_prompt=candidate,
        token_counter=_counter(lambda text: 100 if text == native else 101), provider="codex_cli", model="qualified-local")
    assert result.selected_mode == "legacy"
    assert result.selected_prompt == native
    assert result.receipt["selected_complete_tokens"] == 100


@pytest.mark.parametrize("counter_value", [None, 0, True, -1, 1 << 63, float("inf")])
def test_invalid_counter_counts_cannot_be_rescued_by_byte_proxy(view, counter_value):
    native, candidate = _complete(view)
    result = columns.select_capsule_columns(view=view, native_complete_prompt=native, candidate_complete_prompt=candidate,
        token_counter=_counter(lambda _text: counter_value), provider="codex_cli", model="qualified-local")
    assert result.selected_mode == "legacy"
    assert result.receipt["token_counter_status"] == "failed"
    assert result.receipt["native_complete_tokens"] is None
    assert result.receipt["candidate_complete_tokens"] is None


def test_counter_failure_is_atomic_and_does_not_export_private_error(view):
    native, candidate = _complete(view)
    def count(text):
        if text == candidate:
            raise RuntimeError("private prompt and credential")
        return 100
    result = columns.select_capsule_columns(view=view, native_complete_prompt=native, candidate_complete_prompt=candidate,
        token_counter=_counter(count), provider="codex_cli", model="qualified-local")
    assert result.selected_mode == "legacy"
    assert result.receipt["native_complete_tokens"] is None
    assert "private prompt" not in result.receipt_json


@pytest.mark.parametrize("provider,model", [(None, None), ("grok_cli", "qualified-local"), ("codex_cli", "foreign-model")])
def test_counter_route_mismatch_or_absence_refuses_measurement(view, provider, model):
    native, candidate = _complete(view)
    seen = []
    result = columns.select_capsule_columns(view=view, native_complete_prompt=native, candidate_complete_prompt=candidate,
        token_counter=_counter(lambda text: seen.append(text) or 100), provider=provider, model=model)
    assert result.selected_mode == "legacy"
    assert result.receipt["token_counter_status"] == "invalid"
    assert seen == []


@pytest.mark.parametrize("mutation", ["prefix", "suffix", "duplicate_native", "duplicate_candidate"])
def test_complete_prefix_suffix_and_transport_occurrences_are_exact(view, mutation):
    native, candidate = _complete(view)
    if mutation == "prefix":
        candidate = "changed" + candidate
    elif mutation == "suffix":
        candidate += "changed"
    elif mutation == "duplicate_native":
        native += view.native_prompt
    else:
        candidate += view.provider_prompt
    with pytest.raises(columns.CapsuleColumnsError):
        columns.select_capsule_columns(view=view, native_complete_prompt=native, candidate_complete_prompt=candidate)


@pytest.mark.parametrize("mutation", ["row_order", "row_drop", "ref_drop", "unknown_extra", "signature", "header", "authority", "native_context", "raw_source", "instructions"])
def test_independent_original_pin_rejects_resealed_candidate_changes(view, mutation):
    wire = json.loads(view.provider_prompt)
    capsule = wire["translated_semantic"]["capsules"]
    admission = wire["translated_semantic"]["admissions"]
    if mutation == "row_order":
        capsule["rows"].reverse()
    elif mutation == "row_drop":
        capsule["rows"].pop(); capsule["extras"].pop()
    elif mutation == "ref_drop":
        admission["refs"]["rows"].pop(); admission["refs"]["extras"].pop()
    elif mutation == "unknown_extra":
        capsule["extras"][0]["unknown"] = "changed"
    elif mutation == "signature":
        capsule["shared"]["signature"]["return"] = "foreign"
    elif mutation == "header":
        capsule["columns"].reverse()
    elif mutation == "authority":
        wire["execution_authority"] = True
    elif mutation == "native_context":
        wire["native_context"]["goal"]["constraint"] = "changed"
    elif mutation == "raw_source":
        wire["translated_semantic"]["raw_sources"]["module.py"] += "changed"
    else:
        wire["capsule_columns_instructions"] = "omit pending obligations"
    text, receipt = _reseal_candidate(view, wire)
    with pytest.raises(columns.CapsuleColumnsError):
        columns.restore_capsule_columns(provider_prompt=text, receipt=receipt)


def test_noncanonical_uniform_field_encoding_refuses_even_if_inverse_values_match(view):
    wire = json.loads(view.provider_prompt)
    table = wire["translated_semantic"]["capsules"]
    confidence = table["shared"].pop("confidence")
    old_columns = list(table["columns"])
    table["columns"] = sorted(old_columns + ["confidence"])
    for index, row in enumerate(table["rows"]):
        values = dict(zip(old_columns, row, strict=True)); values["confidence"] = confidence
        table["rows"][index] = [values[key] for key in table["columns"]]
    text, receipt = _reseal_candidate(view, wire)
    with pytest.raises(columns.CapsuleColumnsError, match="recipe"):
        columns.restore_capsule_columns(provider_prompt=text, receipt=receipt)


@pytest.mark.parametrize("mutation", ["shared_override", "ref_override", "unknown_column", "duplicate_column", "extra_table_key"])
def test_closed_table_boundaries_refuse_ambiguous_overrides(view, mutation):
    wire = json.loads(view.provider_prompt)
    table = wire["translated_semantic"]["capsules"]
    if mutation == "shared_override":
        table["extras"][0]["confidence"] = "exact"
    elif mutation == "ref_override":
        wire["translated_semantic"]["admissions"]["extras"][0]["ref"] = {"fake": True}
    elif mutation == "unknown_column":
        table["columns"].append("arbitrary_unknown")
    elif mutation == "duplicate_column":
        table["columns"].append(table["columns"][0])
    else:
        table["authority"] = False
    text, receipt = _reseal_candidate(view, wire)
    with pytest.raises(columns.CapsuleColumnsError):
        columns.restore_capsule_columns(provider_prompt=text, receipt=receipt)


@pytest.mark.parametrize("field", ["proof_authority", "source_freshness_verified", "execution_authority", "omission_authority", "candidate_only"])
def test_receipt_cannot_grant_authority_or_change_candidate_status(view, field):
    receipt = deepcopy(view.receipt)
    receipt[field] = not receipt[field]
    with pytest.raises(columns.CapsuleColumnsError):
        columns.restore_capsule_columns(provider_prompt=view.provider_prompt, receipt=receipt)


@pytest.mark.parametrize("mutation", ["noncanonical", "duplicate", "nan", "wrong_schema"])
def test_input_json_is_canonical_finite_unique_and_native_v1(mutation):
    text = canonical(transport())
    if mutation == "noncanonical":
        text = " " + text
    elif mutation == "duplicate":
        text = '{"schema":"duplicate",' + text[1:]
    elif mutation == "nan":
        text = text.replace('"confidence":"exact"', '"confidence":NaN', 1)
    else:
        wire = transport(); wire["schema"] = "supervisor-semantic-router-input@2"; text = canonical(wire)
    with pytest.raises(columns.CapsuleColumnsError):
        columns.project_capsule_columns(text)


def test_json_lexical_node_limit_precedes_materialization(monkeypatch):
    text = canonical(transport())
    monkeypatch.setattr(columns, "MAX_NODES", 10)
    def must_not_load(*_args, **_kwargs):
        pytest.fail("JSON materialization occurred before lexical bound")
    monkeypatch.setattr(columns.json, "loads", must_not_load)
    with pytest.raises(columns.CapsuleColumnsError, match="structure_bound"):
        columns.project_capsule_columns(text)


def test_json_scanner_ignores_quoted_escaped_delimiters(monkeypatch):
    monkeypatch.setattr(columns, "MAX_NODES", 4)
    value = {"literal": '[[{{\\\"}}]]'}
    assert columns._parse(canonical(value)) == value


def test_source_population_limit_refuses_complete_request(view, monkeypatch):
    monkeypatch.setattr(columns, "MAX_ROWS", 2)
    with pytest.raises(columns.CapsuleColumnsError, match="population_bound"):
        columns.project_capsule_columns(view.native_prompt)


def test_shared_value_expansion_bomb_refuses_before_any_row_materialization(view, monkeypatch):
    wire = json.loads(view.provider_prompt)
    capsule = wire["translated_semantic"]["capsules"]
    old_columns = list(capsule["columns"])
    metadata_index = old_columns.index("metadata")
    capsule["columns"].pop(metadata_index)
    for row in capsule["rows"]:
        row.pop(metadata_index)
    capsule["shared"]["metadata"] = {"big_literal": "X" * 5000}
    for table in (capsule, wire["translated_semantic"]["admissions"], wire["translated_semantic"]["admissions"]["refs"]):
        table["rows"] = [deepcopy(table["rows"][0]) for _ in range(1000)]
        table["extras"] = [deepcopy(table["extras"][0]) for _ in range(1000)]
    text, receipt = _reseal_candidate(view, wire)
    receipt["capsule_count"] = receipt["admission_count"] = 1000
    for stats in receipt["group_counts"].values():
        stats["rows"] = 1000
    def must_not_expand(*_args, **_kwargs):
        pytest.fail("Row expansion occurred before complete bound")
    monkeypatch.setattr(columns, "_materialize", must_not_expand)
    with pytest.raises(columns.CapsuleColumnsError, match="expansion_bound"):
        columns.restore_capsule_columns(provider_prompt=text, receipt=receipt)


def test_direct_receipt_structure_and_expansion_are_bounded_before_serialization(view, monkeypatch):
    receipt = deepcopy(view.receipt)
    receipt["unexpected"] = "X" * 1000
    monkeypatch.setattr(columns, "MAX_RECEIPT_BYTES", 100)
    def must_not_serialize(*_args, **_kwargs):
        pytest.fail("Receipt serialization occurred before bound")
    monkeypatch.setattr(metadata, "_json", must_not_serialize)
    with pytest.raises(columns.CapsuleColumnsError, match="receipt_bound"):
        columns.validate_capsule_columns_receipt(receipt)


def test_selector_refuses_fabricated_view_or_modified_original_pin(view):
    changed = metadata.SemanticMetadataView(view.native_prompt + "changed", view.provider_prompt, view.receipt_json)
    native, candidate = _complete(changed)
    with pytest.raises(columns.CapsuleColumnsError):
        columns.select_capsule_columns(view=changed, native_complete_prompt=native, candidate_complete_prompt=candidate)


def test_unicode_json_wire_measurement_includes_control_escaping():
    value = {"Ω": 'café\n\x00\\"😀', "typed": [False, 0, None, [], {}]}
    assert columns._profile(value)[1] == len(canonical(value).encode())
    assert columns._parse(canonical(value)) == value
