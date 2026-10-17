"""Selected pure codecs through the collector; audit/native claims are authored."""
from copy import deepcopy
import json

import pytest

from benchmarks.agent_supervisor.container_coding.test_terminal_inline_source_observation import (
    INVOCATION, _collect, _json, _read, _report, _sha,
)
from ipfs_accelerate_py.agent_supervisor.runtime.rendered_input_census import census_rendered_input


def _selected(kind):
    if kind == "query":
        from test.api.test_inline_source_carriers import query_view
        model = query_view().provider_prompt
        sources = json.loads(model)["transport"]["translated_semantic"]["raw_sources"]
        scope = "transport.translated_semantic.raw_sources"
    elif kind == "source_input":
        from test.api.test_inline_source_carriers import source_wrapper
        wire = source_wrapper()
        model = _json(wire)
        sources = wire["semantic_transport"]["translated_semantic"]["raw_sources"]
        scope = "semantic_transport.translated_semantic.raw_sources"
    else:
        from test.api.test_inline_source_instruction import instruction_case
        native, view, block, _ = instruction_case()
        model = view.provider_prompt + block
        sources = json.loads(native)["translated_semantic"]["raw_sources"]
        scope = "translated_semantic.raw_sources_and_bound_public_instruction"
    return model, sources, scope


def _rebind(report, model):
    # These are authored retained-audit assertions, not a historical replay.
    # Exercise the real selected-text observer and its collector join only.
    census = census_rendered_input(model_prompt=model, workspace_advisory="", invocation_id=INVOCATION)
    report["provider_invocations"][0].update(model_prompt_sha256=_sha(model), model_prompt_bytes=len(model.encode()))
    audit = report["context_input_audit"]
    audit["matches"][0].update(model_prompt_sha256=_sha(model), model_prompt_bytes=len(model.encode()))
    audit["rendered_input_census"]["observations"] = [census]
    return report


@pytest.mark.parametrize("kind", ["query", "source_input", "instruction"])
def test_new_carrier_source_presence_flows_through_actual_census_and_collector(kind):
    model, sources, scope = _selected(kind)
    assert sources
    rows = [_read(i + 1, path=path, source=body) for i, (path, body) in enumerate(sources.items())]
    report = _rebind(_report(*rows), model)
    before = deepcopy(report)
    result, observed = _collect(report)
    assert observed["status"] == "observed"
    assert observed["matched_reads"] == len(sources)
    assert observed["matched_logical_bytes"] == sum(len(body.encode()) for body in sources.values())
    assert observed["initial_source_carrier_scope"] == scope
    assert observed["initial_source_observation_schema"] == "inline-source-observation@2"
    assert observed["provider_tokens"] is observed["token_savings"] is None
    assert report == before
    exported = _json(result)
    for path, body in sources.items():
        assert path not in exported and body not in exported


@pytest.mark.parametrize("kind", ["query", "source_input", "instruction"])
def test_changed_source_version_is_a_scoped_nonmatch_for_each_new_carrier(kind):
    model, sources, scope = _selected(kind)
    path, body = next(iter(sources.items()))
    _, observed = _collect(_rebind(_report(_read(1, path=path, source=body + "\n# changed")), model))
    assert observed["status"] == "observed"
    assert observed["matched_reads"] == observed["matched_logical_bytes"] == 0
    assert observed["initial_source_carrier_scope"] == scope
    assert observed["full_prompt_source_coverage"] is None


@pytest.mark.parametrize("mutation", ["old_schema", "wrong_scope", "stale_model"])
def test_new_carrier_cannot_relabel_or_rebind_imported_evidence(mutation):
    model, sources, _ = _selected("query")
    path, body = next(iter(sources.items()))
    report = _rebind(_report(_read(1, path=path, source=body)), model)
    value = report["context_input_audit"]["rendered_input_census"]["observations"][0]["inline_sources"]
    if mutation == "old_schema":
        value["schema"] = "inline-source-observation@1"
    elif mutation == "wrong_scope":
        value["scope"] = "translated_semantic.raw_sources"
    else:
        value["model_prompt_sha256"] = "0" * 64
    _, observed = _collect(report)
    assert observed["status"] == "unavailable"
    assert observed["matched_reads"] is observed["known_matched_reads"] is None
    assert observed["initial_source_carrier_scope"] is None


def test_native_fallback_does_not_count_unselected_query_candidate():
    candidate, sources, _ = _selected("query")
    path, body = next(iter(sources.items()))
    # Candidate-looking text in the opaque suffix cannot replace selected input.
    selected = '{"goal":"repair the current source"}\nRejected candidate:\n' + candidate
    _, observed = _collect(_rebind(_report(_read(1, path=path, source=body)), selected))
    assert observed["status"] == "unavailable"
    assert observed["known_matched_reads"] is observed["matched_reads"] is None


def test_instruction_with_preceding_guidance_keeps_overlap_unknown():
    from test.api.test_inline_source_instruction import instruction_case
    native, view, block, _ = instruction_case()
    sources = json.loads(native)["translated_semantic"]["raw_sources"]
    path, body = next(iter(sources.items()))
    selected = view.provider_prompt + "\nOpaque prior guidance\n" + block
    _, observed = _collect(_rebind(_report(_read(1, path=path, source=body)), selected))
    assert observed["status"] == "unavailable"
    assert observed["known_matched_reads"] is observed["matched_reads"] is None
