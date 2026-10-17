"""Selected-message observation and failure isolation using authored inputs."""
import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import rendered_input_census as census
from ipfs_accelerate_py.agent_supervisor.runtime import inline_source_observation as inline
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def source_prompt(sources):
    return json.dumps(dict(schema="supervisor-semantic-router-input@1", translated_semantic={
        "raw_sources": sources,
        "manifest": {path: {"sha256": sha(body)} for path, body in sources.items()},
    }), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def test_selected_source_slots_survive_real_rendering_without_candidate_or_suffix_mining():
    source = "x = '雪\\\"'\r\n"
    rejected_candidate = source_prompt({"rejected.py": "SECRET_REJECTED = 1\n"})
    selected = source_prompt({"chosen.py": source})
    suffix = "\nPrivate instructions mentioning an unselected candidate:\n" + rejected_candidate
    model, advisory = render_model_prompt(prompt=selected + suffix, purpose="coding",
        workspace=Path("/authored/removed-workspace"), semantic_transport=True)
    result = census.census_rendered_input(model_prompt=model, workspace_advisory=advisory,
        invocation_id="selected-census-call")
    observed = inline.validate_inline_sources(result["inline_sources"],
        model_prompt_sha256=result["model_prompt_sha256"], model_prompt_bytes=result["model_prompt_bytes"],
        invocation_id_sha256=result["invocation_id_sha256"])
    assert observed["status"] == "observed"
    assert observed["source_count"] == 1
    assert observed["logical_source_bytes"] == len(source.encode())
    assert observed["entries"][0]["content_sha256"] == sha(source)
    assert observed["wire_overlap_bytes"] is None
    assert result["byte_partition_reconciles"] is True
    exported = json.dumps(result)
    for secret in ("chosen.py", "rejected.py", "SECRET_REJECTED", "雪", "removed-workspace"):
        assert secret not in exported


@pytest.mark.parametrize("failure", ["exception", "oversized", "invalid_json"])
def test_optional_inline_failure_preserves_existing_census_and_contains_private_errors(monkeypatch, failure):
    prompt = source_prompt({"private.py": "private = 1\n"})
    expected = census.census_rendered_input(model_prompt=prompt, workspace_advisory="", invocation_id="call")
    expected.pop("inline_sources")

    def fail(**kwargs):
        if failure == "exception":
            raise ValueError("PRIVATE exception path/content")
        return {"PRIVATE": "x" * census.MAX_RESULT_BYTES} if failure == "oversized" else {"PRIVATE": float("nan")}

    monkeypatch.setattr(inline, "observe_inline_sources", fail)
    actual = census.census_rendered_input(model_prompt=prompt, workspace_advisory="", invocation_id="call")
    assert actual == expected
    assert "PRIVATE" not in json.dumps(actual)


def test_native_plain_context_is_unknown_not_empty():
    result = census.census_rendered_input(model_prompt='{"goal":"repair file"}', workspace_advisory="",
        invocation_id="native-call")
    observed = result["inline_sources"]
    assert observed["status"] == "unavailable"
    assert observed["entries"] is None
    assert observed["source_count"] is None
    assert observed["logical_source_bytes"] is None
    assert result["canonical_partition_reconciles"] is True
