"""Content-free census of actual rendered input, using authored stdlib fixtures."""
from __future__ import annotations

import hashlib
import json

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.rendered_input_census import (
    census_rendered_input,
)

PARTITIONS = {
    "evidence_summaries", "evidence_metadata", "evidence_other", "goal", "scope",
    "authority", "acceptance", "native_context", "translated_semantic",
    "translation_table", "instructions", "other_top_level_values", "keys_and_framing",
}
GROUP_FIELDS = {"value_sha256", "canonical_bytes", "occurrences", "repeated_value_bytes"}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _census(value, *, advisory="", suffix="", invocation="fixture-invocation"):
    return census_rendered_input(model_prompt=advisory + _canonical(value).decode() + suffix,
                                 workspace_advisory=advisory, invocation_id=invocation)


def _check(result):
    assert result["schema"] == "supervisor-rendered-input-census@1"
    assert result["content_tokens"] is None and result["provider_usage_tokens"] is None
    assert result["safe_to_remove_bytes"] is None
    for name in ("omission_authority", "execution_authority", "completion_authority",
                 "raw_content_exported", "prompt_modified"):
        assert result[name] is False
    assert set(result["canonical_byte_partition"]) == PARTITIONS
    assert result["byte_partition_reconciles"] is True
    assert result["canonical_partition_reconciles"] is True
    assert result["overlap_totals_are_additive"] is False
    assert result["token_savings_established"] is False
    assert result["nested_json_strings_inspected"] is False
    assert result["literal_duplicate_substrings_mined"] is False
    assert sum(result["canonical_byte_partition"].values()) == result["json_prefix"]["canonical_bytes"]
    assert set(result["disjoint_byte_partition"]) == {"workspace_advisory", "json_prefix", "router_suffix"}
    assert sum(result["disjoint_byte_partition"].values()) == result["model_prompt_bytes"]
    for name in ("repeated_json_subtrees", "repeated_long_strings"):
        report = result[name]
        assert {"groups", "selected_repeated_value_bytes", "groups_truncated"} <= set(report)
        assert report["selected_group_count"] >= len(report["groups"])
        assert report["minimum_canonical_bytes"] == (64 if name == "repeated_json_subtrees" else 128)
        assert report["literal_input_byte_coverage"] is None and report["safe_to_remove_bytes"] is None
        if name == "repeated_json_subtrees":
            assert report["selection"] == "largest_first_disjoint_including_representatives"
        else:
            assert report["additive_with_json_subtrees"] is False
        assert type(report["groups_truncated"]) is bool and len(report["groups"]) <= 32
        assert report["groups"] == sorted(report["groups"], key=lambda group: (-group["canonical_bytes"], group["value_sha256"]))
        for group in report["groups"]:
            assert set(group) == GROUP_FIELDS
            assert len(group["value_sha256"]) == 64 and group["occurrences"] >= 2
            assert group["repeated_value_bytes"] == group["canonical_bytes"] * (group["occurrences"] - 1)
        subtotal = sum(group["repeated_value_bytes"] for group in report["groups"])
        assert report["selected_repeated_value_bytes"] >= subtotal
        if not report["groups_truncated"]:
            assert report["selected_group_count"] == len(report["groups"])
            assert report["selected_repeated_value_bytes"] == subtotal
    assert len(_canonical(result)) <= 65536


def test_exact_rendered_bytes_and_canonical_categories_partition_without_content():
    private = "/private/owner/secret-source.py"
    values = {"evidence": [{"summary": {"private-summary-key": private},
                             "metadata": {"private-metadata-key": "λ secret"},
                             "unclassified-evidence-key": [1, 2]}],
              "goal": {"private-goal-key": "legal task"}, "scope": ["only native"],
              "authority": False, "acceptance": {"lake": "required"},
              "native_context": {"private-native-key": private},
              "translated_semantic": {"private-translated-key": [True, None]},
              "translation_table": {"private-alias": "source"},
              "instructions": "private literal instructions", "unknown-private-key": ["other"]}
    advisory, suffix, invocation = "workspace λ\n", "\nprivate router suffix", "private invocation λ"
    raw = _canonical(values)
    prompt = advisory + raw.decode() + suffix
    result = census_rendered_input(model_prompt=prompt, workspace_advisory=advisory,
                                  invocation_id=invocation)
    _check(result)
    assert result["model_prompt_sha256"] == _digest(prompt.encode())
    assert result["model_prompt_bytes"] == len(prompt.encode())
    assert result["invocation_id_sha256"] == _digest(invocation.encode())
    assert result["json_prefix"] == {"sha256": _digest(raw), "bytes": len(raw),
                                    "canonical_bytes": len(raw), "is_canonical": True}
    assert result["disjoint_byte_partition"] == {"workspace_advisory": len(advisory.encode()),
        "json_prefix": len(raw), "router_suffix": len(suffix.encode())}
    parts = result["canonical_byte_partition"]
    summary_bytes = len(_canonical(values["evidence"][0]["summary"]))
    metadata_bytes = len(_canonical(values["evidence"][0]["metadata"]))
    assert parts["evidence_summaries"] == summary_bytes
    assert parts["evidence_metadata"] == metadata_bytes
    assert parts["evidence_other"] == len(_canonical(values["evidence"])) - summary_bytes - metadata_bytes
    for category in ("goal", "scope", "authority", "acceptance", "native_context",
                     "translated_semantic", "translation_table", "instructions"):
        assert parts[category] == len(_canonical(values[category]))
    assert parts["other_top_level_values"] == len(_canonical(values["unknown-private-key"]))
    assert parts["keys_and_framing"] == len(raw) - sum(len(_canonical(value)) for value in values.values())
    exported = _canonical(result).decode()
    for secret in (private, "private-summary-key", "private-metadata-key", "private-goal-key",
                   "private-native-key", "unknown-private-key", suffix, invocation):
        assert secret not in exported
    assert prompt == advisory + raw.decode() + suffix


def test_unicode_escapes_and_noncanonical_spaces_use_actual_disjoint_byte_lengths():
    prefix = '{ "goal" : "\\u03bb法律", "evidence" : [] }'
    advisory, suffix = "π\n", "\nRULES λ"
    prompt = advisory + prefix + suffix
    result = census_rendered_input(model_prompt=prompt, workspace_advisory=advisory,
                                  invocation_id="unicode-fixture")
    _check(result)
    canonical = _canonical(json.loads(prefix))
    assert result["json_prefix"] == {"sha256": _digest(prefix.encode()), "bytes": len(prefix.encode()),
                                    "canonical_bytes": len(canonical), "is_canonical": False}
    assert result["disjoint_byte_partition"] == {"workspace_advisory": len(advisory.encode()),
        "json_prefix": len(prefix.encode()), "router_suffix": len(suffix.encode())}
    assert len(prompt) != result["model_prompt_bytes"]


def test_largest_disjoint_groups_reserve_representative_and_all_copies():
    inner = {"payload": "a" * 80}
    outer = {"child": inner, "padding": "b" * 80}
    result = _census({"first": outer, "second": outer, "outside": inner})
    _check(result)
    assert result["repeated_json_subtrees"]["groups"] == [{"value_sha256": _digest(_canonical(outer)),
        "canonical_bytes": len(_canonical(outer)), "occurrences": 2,
        "repeated_value_bytes": len(_canonical(outer))}]


def test_equal_size_group_ties_are_deterministic_by_digest_not_input_order():
    first, second = {"payload": "a" * 80}, {"payload": "b" * 80}
    before = _census({"a": first, "b": second, "c": first, "d": second})
    after = _census({"d": second, "c": first, "b": second, "a": first})
    _check(before)
    _check(after)
    assert before["repeated_json_subtrees"] == after["repeated_json_subtrees"]
    groups = before["repeated_json_subtrees"]["groups"]
    assert len(groups) == 2 and groups[0]["canonical_bytes"] == groups[1]["canonical_bytes"]
    assert [group["value_sha256"] for group in groups] == sorted(_digest(_canonical(value)) for value in (first, second))


def test_truncated_groups_keep_accurate_total_over_all_selected_groups():
    children = [{"number": index, "payload": "x" * 90} for index in range(40)]
    result = _census({f"pair{index}": [child, child] for index, child in enumerate(children)})
    _check(result)
    report = result["repeated_json_subtrees"]
    assert report["groups_truncated"] is True and len(report["groups"]) == 32
    assert report["selected_repeated_value_bytes"] == sum(len(_canonical(child)) for child in children)
    assert report["selected_repeated_value_bytes"] > sum(group["repeated_value_bytes"] for group in report["groups"])


def test_exact_long_string_repetitions_are_separate_from_nested_subtree_totals():
    literal = "private repeated λ " * 12
    child = {"literal": literal}
    result = _census({"first": child, "second": child, "outside": literal,
                      "not-an-exact-match": literal + "!"})
    _check(result)
    assert result["repeated_json_subtrees"]["selected_repeated_value_bytes"] == len(_canonical(child))
    strings = result["repeated_long_strings"]
    assert strings["groups"] == [{"value_sha256": _digest(_canonical(literal)),
        "canonical_bytes": len(_canonical(literal)), "occurrences": 3,
        "repeated_value_bytes": len(_canonical(literal)) * 2}]
    assert literal not in _canonical(result).decode()
    assert result["safe_to_remove_bytes"] is None


def test_non_list_evidence_is_fully_accounted_as_other_and_output_has_no_shared_state():
    value = {"evidence": {"summary": "not a list element", "metadata": [1, 2]}, "unknown": "λ"}
    result = _census(value)
    _check(result)
    assert result["canonical_byte_partition"]["evidence_summaries"] == 0
    assert result["canonical_byte_partition"]["evidence_metadata"] == 0
    assert result["canonical_byte_partition"]["evidence_other"] == len(_canonical(value["evidence"]))
    preserved = _canonical(value)
    result["canonical_byte_partition"]["evidence_other"] = -1
    result["repeated_json_subtrees"]["groups"].append({"not": "shared"})
    fresh = _census(value)
    _check(fresh)
    assert fresh["canonical_byte_partition"]["evidence_other"] >= 0
    assert not fresh["repeated_json_subtrees"]["groups"] and _canonical(value) == preserved


@pytest.mark.parametrize("advisory", ["not-the-prefix", "prefixλ-longer"])
def test_workspace_advisory_must_be_the_exact_actual_prefix(advisory):
    with pytest.raises(ValueError) as captured:
        census_rendered_input(model_prompt='prefixλ{"goal":"private-goal"}',
                              workspace_advisory=advisory, invocation_id="private-invocation")
    assert "private" not in str(captured.value) and advisory not in str(captured.value)


@pytest.mark.parametrize("body", [
    '{"private-duplicate":1,"private-duplicate":2}',
    '{"nested":{"private-duplicate":1,"private-duplicate":2}}',
    '{"private-number":NaN}',
    '[{"private-root":"array"}]',
])
def test_strict_dictionary_json_refuses_duplicates_nonfinite_and_wrong_root(body):
    with pytest.raises(ValueError) as captured:
        census_rendered_input(model_prompt=body, workspace_advisory="", invocation_id="private-invocation")
    assert "private" not in str(captured.value)


@pytest.mark.parametrize("bound", ["bytes", "nodes", "depth", "work"])
def test_all_independent_resource_bounds_refuse_without_exporting_input(bound):
    if bound == "bytes":
        value = {"private-large": "x" * 256000}
    elif bound == "nodes":
        value = {"private-many": [0] * 20001}
    elif bound == "depth":
        value = "private-deep"
        for _ in range(33):
            value = [value]
        value = {"depth": value}
    else:
        value = "x" * 255900
        for _ in range(31):
            value = [value]
        value = {"x": value}
        assert len(_canonical(value)) <= 256000, "work fixture must respect the independent input bound"
    with pytest.raises(ValueError) as captured:
        _census(value)
    assert "private" not in str(captured.value)


@pytest.mark.parametrize("arguments", [
    {"model_prompt": True, "workspace_advisory": "", "invocation_id": "private-id"},
    {"model_prompt": "{}", "workspace_advisory": "", "invocation_id": "λ" * 257},
    {"model_prompt": "{}", "workspace_advisory": "", "invocation_id": ""},
    {"model_prompt": '{"private":"\ud800"}', "workspace_advisory": "", "invocation_id": "id"},
])
def test_types_identity_and_utf8_are_strict_and_errors_do_not_echo_content(arguments):
    with pytest.raises(ValueError) as captured:
        census_rendered_input(**arguments)
    assert "private" not in str(captured.value) and "λ" not in str(captured.value)


def test_empty_dictionary_and_exact_total_input_byte_boundary_are_supported():
    empty = _census({})
    _check(empty)
    assert empty["canonical_byte_partition"]["keys_and_framing"] == 2
    assert not empty["repeated_json_subtrees"]["groups"]
    suffix = "x" * (256000 - 2)
    boundary = _census({}, suffix=suffix)
    _check(boundary)
    assert boundary["model_prompt_bytes"] == 256000
    assert boundary["disjoint_byte_partition"]["router_suffix"] == len(suffix)
