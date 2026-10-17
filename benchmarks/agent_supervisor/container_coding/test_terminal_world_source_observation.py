"""Actual selected world codec/census join; retained audit assertions authored."""
from copy import deepcopy

import pytest

from benchmarks.agent_supervisor.container_coding.test_terminal_inline_source_observation import (
    INVOCATION, _collect, _json, _read, _report, _sha,
)
from ipfs_accelerate_py.agent_supervisor.runtime.rendered_input_census import census_rendered_input
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import GUIDANCE
from test.api.test_world_inline_source_observation import semantic_members, world_case, wire


def bind(report, case):
    model = case["model"]
    census = census_rendered_input(model_prompt=model, workspace_advisory=case["advisory"],
        invocation_id=INVOCATION, input_representation="world_task_packet")
    report["provider_invocations"][0].update(model_prompt_sha256=_sha(model), model_prompt_bytes=len(model.encode()))
    report["context_input_audit"]["matches"][0].update(model_prompt_sha256=_sha(model), model_prompt_bytes=len(model.encode()))
    report["context_input_audit"]["rendered_input_census"]["observations"] = [census]
    return report


def test_selected_world_sources_join_only_same_invocation_original_reads():
    case = world_case()
    reads = [_read(i + 1, path=path, source=body) for i, (path, body) in enumerate(case["sources"].items())]
    report = bind(_report(*reads), case)
    before = deepcopy(report)
    result, overlap = _collect(report)
    assert overlap["status"] == "observed"
    assert overlap["initial_source_observation_schema"] == "inline-source-observation@3"
    assert overlap["initial_source_carrier_scope"] == "world_packet.selected_semantic_context.raw_sources"
    assert overlap["matched_reads"] == len(case["sources"])
    assert overlap["matched_logical_bytes"] == sum(len(body.encode()) for body in case["sources"].values())
    assert overlap["provider_tokens"] is overlap["token_savings"] is overlap["inline_overlap_bytes"] is None
    assert report == before
    census = report["context_input_audit"]["rendered_input_census"]["observations"][0]
    assert census["nested_json_strings_inspected"] is True
    assert census["repetition_census_nested_json_strings_inspected"] is False
    assert sum(census["disjoint_byte_partition"].values()) == len(case["model"].encode())
    assert "calc.py" not in _json(result) and "value =" not in _json(result)


def test_missing_selected_chunk_keeps_overlap_unknown_without_changing_native_usage():
    case = world_case()
    case["envelope"]["packet"]["members"].remove(semantic_members(case["envelope"])[0])
    case["model"] = case["advisory"] + GUIDANCE + "\n" + wire(case["envelope"])
    path, body = next(iter(case["sources"].items()))
    report = bind(_report(_read(1, path=path, source=body)), case)
    before = deepcopy(report)
    _, overlap = _collect(report)
    assert overlap["status"] == "unavailable"
    assert overlap["reason"] == "inline_sources_invalid_world_source_carrier"
    assert overlap["matched_reads"] is overlap["known_matched_reads"] is None
    assert report == before


@pytest.mark.parametrize("change", ["schema", "scope", "transport", "model", "invocation"])
def test_world_observation_cannot_be_relabelled_or_rebound(change):
    case = world_case()
    path, body = next(iter(case["sources"].items()))
    report = bind(_report(_read(1, path=path, source=body)), case)
    inline = report["context_input_audit"]["rendered_input_census"]["observations"][0]["inline_sources"]
    key, value = {
        "schema": ("schema", "inline-source-observation@2"),
        "scope": ("scope", "translated_semantic.raw_sources"),
        "transport": ("transport_schema", "supervisor-semantic-router-input@1"),
        "model": ("model_prompt_sha256", "0" * 64),
        "invocation": ("invocation_id_sha256", "0" * 64),
    }[change]
    inline[key] = value
    _, overlap = _collect(report)
    assert overlap["status"] == "unavailable"
    assert overlap["reason"] == "invalid_inline_sources"
    assert overlap["matched_reads"] is overlap["known_matched_reads"] is None


def test_changed_source_does_not_match_despite_same_selected_world_carrier():
    case = world_case()
    path, body = next(iter(case["sources"].items()))
    _, overlap = _collect(bind(_report(_read(1, path=path, source=body + "# successor\n")), case))
    assert overlap["status"] == "observed" and overlap["matched_reads"] == 0
    assert overlap["full_prompt_source_coverage"] is None
