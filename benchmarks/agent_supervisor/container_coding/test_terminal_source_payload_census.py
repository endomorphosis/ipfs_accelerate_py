"""Real producer shape checks and disjoint, content-free census accounting."""
from copy import deepcopy
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_source_payload_census as census


@pytest.fixture(scope="module")
def authored(tmp_path_factory):
    output = tmp_path_factory.mktemp("source-payload") / "authored"
    result = census.run_authored_census(output=output)
    payload = json.loads((output / "repository/semantic/worker-context.json").read_bytes())
    return output, result, payload


def test_real_producer_retains_support_and_omits_optional_program(authored):
    _, result, payload = authored
    assert result["real_producer_replay_byte_exact"] is True
    assert result["original_sources_unchanged"] is True
    assert result["models_invoked"] == result["containers_started"] == 0
    assert result["inline_program_count"] == result["inline_program_source_utf8_bytes"] == 0
    assert result["omitted_optional_program_count"] == 1
    assert set(payload["raw_sources"]) == {"required.md", "support.json"}
    assert payload["raw_sources"]["required.md"] == census.AUTHORED_INSTRUCTION
    assert result["provider_usage_tokens"] is None


def test_real_analyzer_elides_child_bodies_but_preserves_own_implementation(authored):
    assert authored[1]["shape_qualification"]["all_checks_passed"] is True
    assert all(authored[1]["shape_qualification"]["checks"].values())
    capsules = authored[2]["capsules"]
    classes = [facet["version_evidence"] for capsule in capsules for facet in capsule["metadata"]["facets"]
        if facet["version_evidence"].get("_type") == "ClassDef"]
    assert classes
    assert b"837461" not in census.encoded(classes)
    assert b'"member":"calculate"' in census.encoded(classes)


def test_disjoint_byte_partition_reconciles_with_exact_serialization(authored):
    _, result, payload = authored
    assert result["payload_bytes"] == len(census.encoded(payload))
    assert sum(result["disjoint_byte_partition"].values()) == result["payload_bytes"]
    assert result["disjoint_byte_partition"]["pack"] > result["disjoint_byte_partition"]["capsule_facet_version_evidence"]
    changed = deepcopy(payload)
    count = 0
    for capsule in changed["capsules"]:
        for facet in capsule["metadata"]["facets"]:
            if "version_evidence" in facet:
                facet["version_evidence"] = None
                count += 1
    removed_value_bytes = len(census.encoded(payload)) - len(census.encoded(changed)) + 4 * count
    assert result["disjoint_byte_partition"]["capsule_facet_version_evidence"] == removed_value_bytes
    assert result["authored_wrapper_probe"]["byte_equation_verified"] is True
    assert result["authored_wrapper_probe"]["actual_provider_input"] is False


def test_diagnostic_exports_counts_and_hashes_without_source_or_mutation(authored):
    payload = deepcopy(authored[2])
    before = census.encoded(payload)
    result = census.census_worker_payload(payload)
    rendered = census.encoded(result)
    assert b"837461" not in rendered and b"718291" not in rendered
    assert b"Return a bounded nonnegative sum" not in rendered
    assert b"required.md" not in rendered and b"program.py" not in rendered
    assert census.encoded(payload) == before
    assert result["canonical_payload_modified"] is False
    assert result["strict_subtree_overlap"]["safe_to_remove_bytes"] is None


def test_overlap_excludes_descendants_of_already_counted_repeated_objects():
    repeated = {"nested": {"source": "AUTHORED_PRIVATE_CONTENT_" * 6}, "another": 5}
    result = census._overlap({"capsules": [repeated], "pack": [repeated]})
    # The two identical enclosing lists are counted once; their children are not added again.
    assert result["strict_identical_repeated_subtree_bytes"] == len(census.encoded([repeated]))
    assert len(result["groups"]) == 1
    assert b"AUTHORED_PRIVATE_CONTENT" not in census.encoded(result)


def test_exact_long_string_overlap_counts_literal_repetition_without_export():
    instruction = "AUTHORED_LITERAL_INSTRUCTION_" * 10
    result = census._long_string_overlap({"raw_sources": {"required.md": instruction},
        "worker_projection": {"query": instruction}, "short_repeated": ["identifier", "identifier"]})
    assert result["strict_identical_repeated_string_bytes"] == len(census.encoded(instruction))
    assert result["groups"][0]["top_level_groups"] == {"raw_sources": 1, "worker_projection": 1}
    assert result["additive_with_structured_subtree_overlap"] is False
    assert b"AUTHORED_LITERAL_INSTRUCTION" not in census.encoded(result)


@pytest.mark.parametrize("damage", ["raw_hash", "required_missing", "foreign_capsule", "duplicate_selection", "bad_facet"])
def test_shape_or_binding_drift_rejected(authored, damage):
    payload = deepcopy(authored[2])
    if damage == "raw_hash":
        payload["raw_sources"]["required.md"] += "changed"
    elif damage == "required_missing":
        payload["raw_sources"].pop("required.md")
    elif damage == "foreign_capsule":
        payload["capsules"][0]["stable_symbol_id"] = "foreign"
    elif damage == "duplicate_selection":
        payload["worker_projection"]["selected_symbols"][1] = payload["worker_projection"]["selected_symbols"][0]
    else:
        payload["capsules"][0]["metadata"]["facets"] = {}
    with pytest.raises(ValueError):
        census.census_worker_payload(payload)


def test_source_and_traversal_limits_fail_before_export():
    with pytest.raises(ValueError, match="bounded"):
        census.census_worker_payload({"too_large": "x" * (census.MAX_PAYLOAD_BYTES + 1)})
    with pytest.raises(ValueError, match="traversal"):
        census.census_worker_payload({"nodes": [None] * census.MAX_NODES})
    value = {}
    for _ in range(census.MAX_DEPTH + 1):
        value = {"nested": value}
    with pytest.raises(ValueError, match="traversal"):
        census.census_worker_payload(value)


def test_existing_census_artifact_never_overwritten(authored):
    with pytest.raises(ValueError, match="new canonical"):
        census.run_authored_census(output=authored[0])


def test_public_unknown_task_rejected_before_output(tmp_path):
    with pytest.raises(ValueError, match="reviewed public"):
        census.run_public_census(dataset=tmp_path, task_name="unreviewed", output=tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()
