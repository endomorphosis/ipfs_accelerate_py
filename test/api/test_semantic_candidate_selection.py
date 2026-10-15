"""Authored full-wire counter controls; no producer or provider qualification."""
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    PromptTokenCounter,
    render_model_prompt,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_candidate_selection import (
    select_semantic_candidate,
)


def counter(callback=None, **bindings):
    return PromptTokenCounter(**{
        "provider": "codex_cli", "model": "fixture-model", "tokenizer_id": "fixture:utf8",
        "tokenizer_revision": "authored-v1", "special_token_policy": "fixture-only", **bindings,
    }, count_tokens=callback or (lambda value: len(value.encode())))


def select(**kwargs):
    return select_semantic_candidate(**{
        "native_prompt": "N" * 1000,
        "candidates": [("legacy", "L" * 200), ("v2", "V" * 100),
                       ("projected_v2", "R" * 300)],
        "suffix": "\nOriginal acceptance and /app/file.py remain literal.",
        "purpose": "coding", "workspace": Path("/allocated/task"),
        "provider": "codex_cli", "model": "fixture-model", "token_counter": counter(), **kwargs,
    })


def test_three_way_competition_selects_v2_when_projection_is_larger():
    model, router, advisory, receipt, label = select()
    assert label == receipt["selected_candidate"] == "v2"
    assert receipt["selected_representation"] == "semantic"
    assert model == advisory + router
    assert router.startswith("V" * 100)
    assert set(receipt["comparison_candidates"]) == {"legacy", "v2", "projected_v2"}
    assert receipt["candidates"]["semantic"] == {
        key: receipt["comparison_candidates"][label][key]
        for key in ("model_prompt_bytes", "model_prompt_sha256", "input_tokens")}
    assert receipt["provider_called"] is receipt["completion_authority"] is False
    assert receipt["total_session_token_savings"] is None


def test_all_counter_inputs_are_complete_with_same_suffix_and_correct_workspace_advisory():
    seen = []
    result = select(token_counter=counter(lambda value: seen.append(value) or len(value.encode())))
    assert len(seen) == 6
    assert seen[0] == seen[2] == seen[4]
    assert all(value.count("Original acceptance") == 1 for value in seen)
    assert all("/allocated/task" in value for value in seen)
    assert result[3]["selected_content_tokens"] == len(result[0].encode())
    assert "N" * 1000 not in str(result[3])


@pytest.mark.parametrize("candidates,expected", [
    ([("projected_v2", "R" * 100), ("v2", "V" * 100), ("legacy", "L" * 100)], "legacy"),
    ([("projected_v2", "R" * 100), ("v2", "V" * 100)], "v2"),
])
def test_ties_prefer_existing_variants_independently_of_caller_order(candidates, expected):
    assert select(candidates=candidates)[4] == expected


def test_equal_or_larger_counts_keep_full_native():
    result = select(token_counter=counter(lambda _: 100))
    assert result[4] is None and result[3]["selected_representation"] == "native"
    assert result[1].startswith("N" * 1000)
    assert result[3]["selected_content_tokens"] == 100


def test_missing_counter_keeps_full_native_and_every_count_unknown():
    result = select(token_counter=None)
    assert result[4] is None and result[1].startswith("N" * 1000)
    assert result[3]["measurement_status"] == "unavailable"
    assert result[3]["selected_content_tokens"] is None
    assert result[3]["token_measurements_consistent"] is None
    assert all(value["input_tokens"] is None for value in result[3]["comparison_candidates"].values())


def test_late_counter_failure_discards_earlier_measured_reduction():
    calls = []
    def count(value):
        calls.append(value)
        if len(calls) == 4:
            raise RuntimeError("authored late count failure")
        return len(value.encode())
    result = select(token_counter=counter(count))
    assert result[4] is None
    assert result[3]["measurement_status"] == "failed"
    assert result[3]["measurement_error_type"] == "RuntimeError"
    assert result[3]["selected_content_tokens"] is None
    assert result[3]["candidates"]["native"]["input_tokens"] is None
    assert all(value["input_tokens"] is None for value in result[3]["comparison_candidates"].values())


def test_repeated_native_count_drift_cannot_select_a_stochastic_winner():
    calls = []
    def count(value):
        calls.append(value)
        return len(value.encode()) + len(calls)
    result = select(token_counter=counter(count))
    assert result[4] is None and result[1].startswith("N" * 1000)
    assert result[3]["selection_reason"] == "tokenizer_inconsistent"
    assert result[3]["measurement_error_type"] == "InconsistentTokenCount"
    assert all(value["input_tokens"] is None for value in result[3]["comparison_candidates"].values())


def test_same_semantic_wire_with_conflicting_counts_discards_all_measurements():
    seen = {}
    def count(value):
        seen[value] = seen.get(value, 0) + 1
        return len(value.encode()) + (seen[value] if "V" * 100 in value else 0)
    result = select(candidates=[("legacy", "V" * 100), ("v2", "V" * 100)],
                    token_counter=counter(count))
    assert result[4] is None
    assert result[3]["selection_reason"] == "tokenizer_inconsistent"


@pytest.mark.parametrize("bad", [True, 0, -1, 1.5])
def test_bad_count_values_do_not_partially_promote_other_variants(bad):
    def count(value):
        return bad if "R" * 300 in value else len(value.encode())
    result = select(token_counter=counter(count))
    assert result[4] is None and result[3]["selected_content_tokens"] is None
    assert result[3]["measurement_status"] == "failed"


def test_invalid_and_oversized_strings_are_unavailable_without_losing_other_candidates():
    result = select(candidates=[("legacy", None), ("v2", "V" * 100),
                                ("projected_v2", "R" * 256001)])
    assert result[4] == "v2"
    assert result[3]["comparison_candidates"]["legacy"]["available"] is False
    assert result[3]["comparison_candidates"]["projected_v2"]["input_tokens"] is None


def test_complete_candidate_byte_limit_includes_semantic_advisory_and_suffix():
    value, _ = render_model_prompt(prompt="V" * 100 + "\nOriginal acceptance and /app/file.py remain literal.",
                                    purpose="coding", workspace=Path("/allocated/task"), semantic_transport=True)
    result = select(max_prompt_bytes=len(value.encode()))
    assert result[4] == "v2"
    assert result[3]["comparison_candidates"]["legacy"]["available"] is False
    assert result[3]["comparison_candidates"]["projected_v2"]["available"] is False


def test_token_ceiling_can_admit_smaller_candidate_when_native_does_not_fit():
    result = select()
    boundary = result[3]["selected_content_tokens"]
    assert select(max_input_tokens=boundary)[4] == "v2"
    with pytest.raises(ValueError, match="token budget"):
        select(max_input_tokens=boundary - 1)


@pytest.mark.parametrize("callback", [None, lambda _: False])
def test_strict_token_ceiling_refuses_missing_or_failed_counter(callback):
    with pytest.raises(ValueError, match="working consistent pinned"):
        select(token_counter=counter(callback) if callback else None, max_input_tokens=5000)


@pytest.mark.parametrize("field", ["provider", "model"])
def test_foreign_counter_binding_refuses(field):
    with pytest.raises(ValueError, match="pinned provider/model"):
        select(token_counter=counter(**{field: "other"}))


@pytest.mark.parametrize("candidates", [
    [("native", "text")], [("unknown", "text")], [("v2", "a"), ("v2", "b")],
    [["v2", "text"]], [(True, "text")], [("v2", "text", "extra")],
])
def test_unknown_duplicate_or_unclosed_labels_refuse(candidates):
    with pytest.raises(ValueError):
        select(candidates=candidates)


def test_no_requested_candidates_uses_existing_native_render_and_count():
    result = select(candidates=[])
    assert result[4] is None
    assert result[3]["comparison_candidates"] == {}
    assert result[3]["selection_reason"] == "semantic_not_requested"
    assert result[3]["selected_content_tokens"] == len(result[0].encode())


def test_hard_cap_oversized_native_pair_cannot_mask_later_fitting_candidate():
    seen = []
    def count(value):
        seen.append(value)
        return 100 if "N" * 1000 in value else 200 if "L" * 100 in value else 50
    result = select(native_prompt="N" * 256000,
                    candidates=[("legacy", "L" * 100), ("v2", "V" * 10)],
                    token_counter=counter(count))
    assert result[4] == "v2"
    assert result[3]["candidates"]["native"]["model_prompt_bytes"] > 256000
    assert result[3]["comparison_candidates"]["legacy"]["input_tokens"] == 200
    assert result[3]["selected_content_tokens"] == 50
    assert len(seen) == 4
    assert len(result[0].encode()) < 256000


def test_counter_failure_in_refused_native_pair_is_not_hidden_by_later_candidate():
    seen = []
    def count(value):
        seen.append(value)
        if "L" * 100 in value:
            raise RuntimeError("authored refused-pair counter failure")
        return 100 if "N" * 1000 in value else 50
    with pytest.raises(ValueError, match="selected prompt exceeds its byte bound"):
        select(native_prompt="N" * 256000,
               candidates=[("legacy", "L" * 100), ("v2", "V" * 10)],
               token_counter=counter(count))
    assert any("V" * 10 in value for value in seen)


def test_all_oversized_native_fallback_pairs_are_tried_before_final_refusal():
    seen = []
    def count(value):
        seen.append(value)
        return 100 if "N" * 1000 in value else 200
    with pytest.raises(ValueError, match="exceeds its byte bound"):
        select(native_prompt="N" * 256000,
               candidates=[("legacy", "L" * 100), ("v2", "V" * 10)],
               token_counter=counter(count))
    assert any("L" * 100 in value for value in seen)
    assert any("V" * 10 in value for value in seen)
