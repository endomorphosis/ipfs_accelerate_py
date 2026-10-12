"""Selective candidate paging preserves generation custody and bounded state."""
from copy import deepcopy
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_context as module
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_selection as selector
from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_views import canonical_json
from test.api.test_worker_candidate_context import case, decode, request, sha  # noqa: F401


def selection(path="code.py", **kwargs):
    return dict(path=path, symbol_id=None, function_id=None, kind=None) | kwargs


@pytest.fixture
def selected(case, monkeypatch):
    calls = []

    def select(**kwargs):
        calls.append(deepcopy(kwargs))
        return canonical_json(dict(schema="authored-selection@1", artifact=kwargs["artifact"],
            selection=kwargs["selection"],
            source_hashes={path: sha(raw) for path, raw in kwargs["sources"].items()},
            content="selected λ🙂\\\"\n" * 500))

    monkeypatch.setattr(selector, "select_candidate_artifact", select)
    case["selection_calls"] = calls
    case["selector"] = select
    return case


def page(case, **kwargs):
    return request(case, **dict(artifact="capsules", selection=selection()) | kwargs)


def assert_discarded(case):
    service = case["service"]
    assert service._generation is service._producers is None
    assert service._artifacts == service._continuations == {}
    assert service._selection_key is service._selected_artifact is None


@pytest.mark.parametrize("artifact,filters", [
    ("catalog", selection(None)), ("catalog", selection()),
    ("capsules", selection()), ("capsules", selection(symbol_id="symbol:increment")),
    ("obligations", selection()),
    ("obligations", selection(function_id="increment", kind="assertion")),
])
def test_selection_pages_reconstruct_exact_bound_payload(selected, artifact, filters):
    chunks, pages, cursor, offset = [], [], None, 0
    for _ in range(100):
        result = page(selected, artifact=artifact, selection=filters, cursor=cursor)
        assert result["schema"] == module.SELECTED_PAGE_SCHEMA
        assert result["selection"] == filters and result["selection"] is not filters
        assert result["offset"] == offset
        assert len(canonical_json(result)) <= 2400
        assert result["baseline_evidence_applicable"] is result["proof_authority"] is False
        assert result["completion_authority"] is False
        assert result["source_obligations_discharged"] == 0
        chunks.append(decode(result))
        pages.append(result)
        offset, cursor = result["end_offset"], result["next_cursor"]
        if cursor is None:
            break
    else:
        pytest.fail("selection pagination did not terminate")
    raw = b"".join(chunks)
    assert len(pages) > 1
    assert len(raw) == pages[0]["artifact_bytes"]
    assert sha(raw) == pages[0]["artifact_sha256"]
    assert raw == selected["service"]._selected_artifact
    assert json.loads(raw)["selection"] == filters
    assert len(selected["calls"]) == len(selected["selection_calls"]) == 1
    assert selected["selection_calls"][0]["sources"] == selected["sources"]
    assert selected["selection_calls"][0]["artifacts"] == selected["service"]._artifacts


def test_selection_switch_recomputes_one_cache_without_revoking_current_cursors(selected):
    first = page(selected)
    page(selected, selection=selection("other.py"))
    assert len(selected["calls"]) == 1
    assert len(selected["selection_calls"]) == 2
    assert len(selected["service"]._continuations) == 2
    resumed = page(selected, cursor=first["next_cursor"])
    assert resumed["offset"] == first["end_offset"]
    assert resumed["artifact_sha256"] == first["artifact_sha256"]
    assert len(selected["selection_calls"]) == 3
    assert json.loads(selected["service"]._selection_key)["selection"] == selection()
    assert len(selected["service"]._continuations) == 2
    page(selected, cursor=resumed["next_cursor"])
    assert len(selected["selection_calls"]) == 3


def test_full_artifact_read_does_not_evict_selected_payload(selected):
    first = page(selected)
    baseline = request(selected, artifact="capsules")
    assert baseline["schema"] == module.PAGE_SCHEMA and "selection" not in baseline
    page(selected, cursor=first["next_cursor"])
    assert len(selected["calls"]) == len(selected["selection_calls"]) == 1


@pytest.mark.parametrize("initial,changed,artifact", [
    (selection(), selection("other.py"), "capsules"),
    (selection(), selection(symbol_id="symbol:increment"), "capsules"),
    (selection(), selection(function_id="increment"), "obligations"),
    (selection(), selection(kind="assertion"), "obligations"),
    (selection(None), selection(), "catalog"),
])
def test_changed_selection_cannot_reuse_continuation(selected, initial, changed, artifact):
    first = page(selected, artifact=artifact, selection=initial)
    with pytest.raises(ValueError, match="continuation"):
        page(selected, artifact=artifact, selection=changed, cursor=first["next_cursor"])
    assert_discarded(selected)


@pytest.mark.parametrize("direction", ["selected_to_full", "full_to_selected"])
def test_cursor_cannot_cross_page_versions(selected, direction):
    if direction == "selected_to_full":
        cursor = page(selected)["next_cursor"]
        action = lambda: request(selected, artifact="capsules", cursor=cursor)
    else:
        cursor = request(selected, artifact="capsules")["next_cursor"]
        action = lambda: page(selected, cursor=cursor)
    with pytest.raises(ValueError, match="continuation"):
        action()
    assert_discarded(selected)


def test_changed_selected_payload_cannot_resume_same_filters(selected, monkeypatch):
    first = page(selected)
    page(selected, selection=selection("other.py"))
    original = selected["selector"]
    def changed(**kwargs):
        payload = json.loads(original(**kwargs))
        payload["content"] += "changed deterministic result"
        return canonical_json(payload)
    monkeypatch.setattr(selector, "select_candidate_artifact", changed)
    with pytest.raises(ValueError, match="continuation"):
        page(selected, cursor=first["next_cursor"])
    assert_discarded(selected)


@pytest.mark.parametrize("location", ["workspace", "root"])
@pytest.mark.parametrize("path", ["code.py", "other.py", "support.txt"])
def test_selection_cache_checks_complete_source_population(selected, location, path):
    first = page(selected)
    (selected[location] / path).write_bytes(b"changed source\n")
    with pytest.raises(ValueError):
        page(selected, cursor=first["next_cursor"])
    assert_discarded(selected)
    assert len(selected["selection_calls"]) == 1


def test_selection_build_checks_source_again_before_return(selected, monkeypatch):
    original = selected["selector"]
    def changed(**kwargs):
        raw = original(**kwargs)
        (selected["workspace"] / "support.txt").write_bytes(b"changed support\n")
        return raw
    monkeypatch.setattr(selector, "select_candidate_artifact", changed)
    with pytest.raises(ValueError):
        page(selected)
    assert_discarded(selected)


@pytest.mark.parametrize("when", ["before", "after"])
def test_selected_cache_checks_producers_before_and_after_retrieval(selected, monkeypatch, when):
    page(selected)
    calls = []
    def producers():
        calls.append(True)
        return b"changed native producer bindings" if when == "before" or len(calls) > 1 else None
    monkeypatch.setattr(selected["service"], "_native_producers", producers)
    with pytest.raises(ValueError, match="producer sources changed"):
        page(selected)
    assert_discarded(selected)
    assert len(selected["selection_calls"]) == 1


def test_selection_never_refreshes_stale_generation_and_rebuilds_only_after_explicit_refresh(selected):
    first = page(selected)
    old_counter = selected["views"]._working_counter
    (selected["workspace"] / "code.py").write_bytes(b"def increment(value):\n    return value + 2\n")
    with pytest.raises(ValueError):
        page(selected)
    assert_discarded(selected)
    assert selected["views"]._working_counter == old_counter
    generation = selected["views"].working_page(path="code.py")["workspace_generation"]
    current = page(selected, expected_workspace_generation=generation)
    assert current["workspace_generation"] != first["workspace_generation"]
    assert current["artifact_sha256"] != first["artifact_sha256"]
    assert len(selected["calls"]) == len(selected["selection_calls"]) == 2


@pytest.mark.parametrize("artifact,filters", [
    ("summary", selection()), ("formal", selection()), ("capsules", {}),
    ("capsules", []), ("capsules", selection(None)),
    ("capsules", selection(function_id="increment")),
    ("catalog", selection(symbol_id="symbol:increment")),
    ("obligations", selection(symbol_id="symbol:increment")),
    ("obligations", selection("../code.py")),
    ("capsules", selection() | {"extra": True}),
])
def test_selection_contract_rejected_before_native_work(selected, artifact, filters):
    with pytest.raises(ValueError):
        page(selected, artifact=artifact, selection=filters)
    assert_discarded(selected)
    assert selected["calls"] == selected["selection_calls"] == []


@pytest.mark.parametrize("bad", [b"", "{}", b"[]", b'{ "x": 1 }', b'{"x":1,"x":2}', b'{"x":NaN}'])
def test_selector_output_must_be_nonempty_canonical_object(selected, monkeypatch, bad):
    monkeypatch.setattr(selector, "select_candidate_artifact", lambda **kwargs: bad)
    with pytest.raises(ValueError):
        page(selected)
    assert_discarded(selected)


def test_selected_cache_has_independent_byte_bound(selected, monkeypatch):
    request(selected)
    monkeypatch.setattr(module, "MAX_ARTIFACT_BYTES", 256)
    with pytest.raises(ValueError, match="bounded selected candidate artifact"):
        page(selected)
    assert_discarded(selected)


def test_returned_selection_is_detached_from_request_and_cache(selected):
    filters = selection()
    first = page(selected, selection=filters)
    filters["path"] = "other.py"
    first["selection"]["path"] = "changed caller object"
    second = page(selected, selection=selection(), cursor=first["next_cursor"])
    assert second["selection"] == selection()
    assert len(selected["selection_calls"]) == 1


def test_clear_drops_full_and_selected_cache_and_cursors(selected):
    page(selected)
    selected["service"].clear()
    assert_discarded(selected)
