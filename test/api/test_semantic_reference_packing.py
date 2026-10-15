"""Pure lossless packing contracts, without producer or provider admission."""
from __future__ import annotations

from copy import deepcopy

import pytest
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    CONTEXT_REFERENCE_SCHEMA,
    canonical_context_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_reference_packing as packing


def reference(index=0):
    return {"schema": CONTEXT_REFERENCE_SCHEMA, "contract_version": 1,
            "reference_id": "reference-" + str(index), "kind": "dependency-capsule",
            "tier": "evidence", "referenced_content_id": {"$semantic_ref": "s0"},
            "repository_id": "repository:preserved", "tree_id": "tree:preserved",
            "path": "src/λ.py", "summary": "  λ\n# Literal source and `names` remain.  ",
            "byte_count": 48, "token_count": 19,
            "metadata": {"required": True, "aliases": [{"$semantic_ref": "s1"}],
                         "literal": {"schema": packing.PACKED_SCHEMA, "unknown": "untouched"}}}


def payload(rows=None, *, role=False):
    body = {"pack": {"references": [reference(), reference(1)] if rows is None else rows},
            "raw_sources": {"src/λ.py": "# Exact λ\nprint('unchanged')\n"},
            "completion_authority": False}
    if role:
        return {"schema": "supervisor-semantic-worker-role-view@1", "required_context": body,
                "optional_evidence": {"schema": packing.PACKED_SCHEMA, "unknown": "literal"}}
    return {"schema": "supervisor-semantic-worker-context@1", **body}


def references(value):
    body = value["required_context"] if value["schema"] == "supervisor-semantic-worker-role-view@1" else value
    return body["pack"]["references"]


def encoded():
    result, stats = packing.pack_semantic_references(payload())
    assert stats["packed"] is True
    return result


@pytest.mark.parametrize("role", [False, True])
def test_exact_unicode_literal_alias_metadata_and_order_roundtrip_without_mutation(role):
    original = payload(role=role)
    before = deepcopy(original)
    compact, stats = packing.pack_semantic_references(original)
    assert original == before
    assert stats["packing_revision"] == packing.PACKING_REVISION
    assert stats["packed"] is True and stats["row_count"] == 2
    assert stats["bytes_saved"] > 0
    assert len(canonical_context_json_bytes(compact)) < len(canonical_context_json_bytes(original))
    compact_before = deepcopy(compact)
    restored = packing.unpack_semantic_references(compact)
    assert canonical_context_json_bytes(restored) == canonical_context_json_bytes(original)
    assert references(restored)[0]["reference_id"] == "reference-0"
    assert references(restored)[1]["reference_id"] == "reference-1"
    assert compact == compact_before
    references(restored)[0]["metadata"]["aliases"][0]["$semantic_ref"] = "changed returned copy"
    assert original == before and compact == compact_before
    assert packing.pack_semantic_references(packing.unpack_semantic_references(compact))[0] == compact


def test_false_and_zero_are_distinct_shared_field_values():
    rows = [reference(), reference(1)]
    rows[0]["metadata"]["literal_value"] = False
    rows[1]["metadata"]["literal_value"] = 0
    compact, stats = packing.pack_semantic_references(payload(rows))
    assert stats["packed"] is True
    assert "metadata" in references(compact)["columns"]
    restored = references(packing.unpack_semantic_references(compact))
    assert type(restored[0]["metadata"]["literal_value"]) is bool
    assert type(restored[1]["metadata"]["literal_value"]) is int


@pytest.mark.parametrize("rows", [[], [reference()], [reference(), "heterogeneous"],
                                  [reference(), {**reference(1), "future_field": True}],
                                  [reference(), {**reference(1), "schema": "future-reference@2"}],
                                  [reference(), {**reference(1), "contract_version": True}]])
def test_empty_singleton_heterogeneous_and_unknown_reference_rows_stay_literal(rows):
    original = payload(rows)
    result, stats = packing.pack_semantic_references(original)
    assert result == original and result is not original
    assert stats["packed"] is False and stats["bytes_saved"] == 0
    assert packing.unpack_semantic_references(result) == original


def test_all_shared_fields_can_use_empty_ordered_cells():
    compact, stats = packing.pack_semantic_references(payload([reference(), reference()]))
    assert stats["packed"] is True and stats["shared_field_count"] == 13
    assert references(compact)["columns"] == []
    assert references(compact)["rows"] == [[], []]
    assert packing.unpack_semantic_references(compact) == payload([reference(), reference()])


def test_unknown_top_level_schema_and_markers_outside_closed_paths_remain_literal():
    original = {"schema": "unrelated@1", "pack": {"references": {"schema": packing.PACKED_SCHEMA}},
                "nested": [{"schema": packing.PACKED_SCHEMA, "rows": "literal source"}]}
    compact, stats = packing.pack_semantic_references(original)
    assert stats["packed"] is False and stats["reference_path"] is None
    assert compact == original and packing.unpack_semantic_references(compact) == original
    known = payload()
    known["other"] = original
    restored = packing.unpack_semantic_references(packing.pack_semantic_references(known)[0])
    assert restored["other"] == original


@pytest.mark.parametrize("mutation", ["schema", "unknown_wrapper", "shared_type", "columns_type",
                                      "rows_type", "duplicate_column", "unsorted_columns", "overlap",
                                      "missing_field", "unknown_field", "width", "row_type",
                                      "single_row", "row_schema", "row_version", "row_bool_version"])
def test_targeted_descriptor_validation_refuses_malformed_encodings(mutation):
    result = encoded()
    descriptor = references(result)
    if mutation == "schema":
        descriptor["schema"] = "supervisor-context-reference-columns@99"
    elif mutation == "unknown_wrapper":
        descriptor["future"] = True
    elif mutation == "shared_type":
        descriptor["shared"] = []
    elif mutation == "columns_type":
        descriptor["columns"] = {}
    elif mutation == "rows_type":
        descriptor["rows"] = {}
    elif mutation == "duplicate_column":
        descriptor["columns"] *= 2
    elif mutation == "unsorted_columns":
        field = "summary"
        value = descriptor["shared"].pop(field)
        descriptor["columns"].insert(0, field)
        for row in descriptor["rows"]:
            row.insert(0, value)
    elif mutation == "overlap":
        descriptor["shared"]["reference_id"] = "overlap"
    elif mutation == "missing_field":
        descriptor["shared"].pop("kind")
    elif mutation == "unknown_field":
        descriptor["shared"]["other"] = "unknown"
    elif mutation == "width":
        descriptor["rows"][0].append("extra")
    elif mutation == "row_type":
        descriptor["rows"][0] = {}
    elif mutation == "single_row":
        descriptor["rows"] = descriptor["rows"][:1]
    elif mutation == "row_schema":
        descriptor["shared"]["schema"] = "unsupported-reference@99"
    elif mutation == "row_version":
        descriptor["shared"]["contract_version"] = 2
    else:
        descriptor["shared"]["contract_version"] = True
    with pytest.raises(packing.SemanticReferencePackingError):
        packing.unpack_semantic_references(result)


def test_alternate_full_field_partition_is_rejected_by_deterministic_repacking():
    result = encoded()
    descriptor = references(result)
    field = "kind"
    value = descriptor["shared"].pop(field)
    descriptor["columns"] = sorted([*descriptor["columns"], field])
    position = descriptor["columns"].index(field)
    for row in descriptor["rows"]:
        row.insert(position, value)
    with pytest.raises(packing.SemanticReferencePackingError, match="noncanonical"):
        packing.unpack_semantic_references(result)


@pytest.mark.parametrize("operation", [packing.pack_semantic_references, packing.unpack_semantic_references])
def test_structure_depth_nodes_cycles_and_bytes_are_bounded_before_copy(operation):
    deep = "leaf"
    for _ in range(packing.MAX_DEPTH + 1):
        deep = [deep]
    cyclic = []
    cyclic.append(cyclic)
    for invalid in (deep, cyclic, [None] * packing.MAX_NODES,
                    {"text": "x" * packing.MAX_BYTES}, {1: "nonstring key"}):
        with pytest.raises(packing.SemanticReferencePackingError):
            operation(invalid)


def test_descriptor_row_population_is_bounded_before_expansion():
    result = encoded()
    references(result)["rows"] = [["id"] for _ in range(packing.MAX_ROWS + 1)]
    with pytest.raises(packing.SemanticReferencePackingError, match="row count"):
        packing.unpack_semantic_references(result)
    with pytest.raises(packing.SemanticReferencePackingError, match="row count"):
        packing.pack_semantic_references(payload([None] * (packing.MAX_ROWS + 1)))


def test_optional_wrapper_depth_cannot_block_a_valid_bounded_unpacked_payload():
    rows = [reference(), reference(1)]
    for index, row in enumerate(rows):
        nested = index
        for _ in range(packing.MAX_DEPTH - 5):
            nested = [nested]
        row["metadata"] = {"value": nested}
    original = payload(rows)
    compact, stats = packing.pack_semantic_references(original)
    assert canonical_context_json_bytes(compact) == canonical_context_json_bytes(original)
    assert stats["packed"] is False and stats["reason"] == "candidate_bounds_exceeded"
    assert stats["bytes_saved"] == 0
    assert packing.unpack_semantic_references(compact) == original


@pytest.mark.parametrize("bound", ["bytes", "nodes"])
def test_virtual_shared_expansion_is_bounded_before_reference_materialization(bound):
    row = reference()
    if bound == "bytes":
        row["summary"] = "λ" * 4000
        count = 300
    else:
        row["metadata"] = {"array": list(range(30))}
        count = 4096
    descriptor = {"schema": packing.PACKED_SCHEMA, "shared": row,
                  "columns": [], "rows": [[] for _ in range(count)]}
    original = payload([])
    original["pack"]["references"] = descriptor
    assert len(canonical_context_json_bytes(original)) < packing.MAX_BYTES
    with pytest.raises(packing.SemanticReferencePackingError, match="expanded reference"):
        packing.unpack_semantic_references(original)
