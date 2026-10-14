"""Joint signed selection, compact edit guidance and historical input custody.

Git, signatures, native context, owner capture and linked allocation are real.
The reused provider and UID/descriptor route are explicitly authored seams. No
edit, socket request, coding child, model generation or task verifier runs here.
"""
from copy import deepcopy
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import git
from benchmarks.agent_supervisor.container_coding.test_terminal_worker_signed_scope import (
    invoke, sha, source_endpoint, worker,  # noqa: F401
)
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as source
from test.api.test_semantic_router_integration import provider  # noqa: F401


def mapped_endpoint(worker, monkeypatch):
    views, options = source_endpoint(worker, monkeypatch)
    original_loader = source.load_descriptor

    def load(path, digest, *, workspace):
        descriptor = original_loader(path, digest, workspace=workspace)
        descriptor.update(schema=source.DESCRIPTOR_SCHEMA_V3, source_edit_mode="mapped-span@2",
            source_scope_sha256=views.admission()["scope_sha256"], view_policy_binding_sha256=None)
        return descriptor

    monkeypatch.setattr(source, "load_descriptor", load)
    return views, {**options, "source_edit_mode": "mapped-span@2"}


@pytest.mark.parametrize("metadata_view", ["legacy", "common-bindings-instruction@1"])
def test_signed_scope_compact_edit_guidance_and_census_replay_together(
        worker, provider, monkeypatch, record_property, metadata_view):
    _, observed = provider
    views, options = mapped_endpoint(worker, monkeypatch)
    originals = {name: (worker.case.root / name).read_bytes()
        for name in worker.case.prepared["worker_inputs"]}
    text, receipt = invoke(worker, semantic_metadata_view=metadata_view, **options)
    assert text == "literal summary" and len(observed) == 1
    actual = observed[0][0]
    assert receipt["public_instruction"]["schema"] == "supervisor-public-instruction-inclusion@5"
    assert receipt["terminal_semantic_scope_dispatch_rechecked"] is True
    scope = receipt["public_instruction"]["terminal_semantic_scope"]
    assert (scope["selected_count"], scope["full_count"], scope["frontier_count"]) == (5, 73, 68)
    dispatch = receipt["source_view_retrieval"]
    assert dispatch["schema"] == "worker-source-dispatch-observation@3"
    assert dispatch["source_edit_mode"] == "mapped-span@2"
    assert dispatch["source_scope_sha256"] == views.admission()["scope_sha256"]
    assert dispatch["requests_observed"] is False
    advice = source.guidance(options["source_view_profile"], source_edit_mode="mapped-span@2")
    assert advice in actual
    assert dispatch["guidance_sha256"] == sha(advice.encode())
    assert dispatch["guidance_bytes"] == len(advice.encode())
    if metadata_view != "legacy":
        assert receipt["semantic_metadata_view"]["requested_mode"] == metadata_view
    census = receipt["semantic_prompt_components"]
    assert census["schema"] == "supervisor-semantic-prompt-component-census@1"
    assert census["native_prompt_sha256"] == sha(worker.prompt.encode())
    assert census["exact_native_round_trip_verified"] is True
    assert census["nested_partitions_overlap"] is True
    assert census["provider_tokens"] is None and census["provider_calls"] == 0
    assert census["raw_content_exported"] is False and census["dynamic_labels_exported"] is False
    assert census["model_prompt_wrappers_included"] is False and census["token_estimate"] is False
    wire = json.dumps(census, sort_keys=True)
    assert len(wire.encode()) < 16384
    for private in (str(worker.case.root), "module_069.py", worker.case.source[worker.case.selected[0]].decode()):
        assert private not in wire
    assert census["provider_prompt_bytes"] < receipt["model_prompt_bytes"]
    assert receipt["model_prompt_sha256"] == sha(actual.encode())
    assert {name: (worker.case.root / name).read_bytes() for name in originals} == originals
    assert {name: (worker.workspace / name).read_bytes() for name in originals} == originals
    parsed = audit._receipt({**receipt, "phase": "coding"}, worker.workspace.parent)
    assert parsed["semantic_prompt_components"] == census
    (worker.case.root / "module_069.py").write_bytes(b"def later_state(): return 11\n")
    monkeypatch.chdir(worker.case.root.parent)
    git(worker.case.root, "worktree", "remove", "--force", str(worker.workspace))
    replayed, _, checks = audit._model_projection(rendered=worker.prompt, receipt=parsed,
        repository=worker.case.root)
    assert replayed == actual and all(checks.values()), checks
    assert checks["semantic_prompt_components"] is True
    changed_census = deepcopy(parsed)
    changed_census["semantic_prompt_components"]["provider_prompt_sha256"] = "0" * 64
    _, _, census_checks = audit._model_projection(rendered=worker.prompt, receipt=changed_census,
        repository=worker.case.root)
    assert census_checks["semantic_prompt_components"] is False
    changed_scope = deepcopy(parsed)
    changed_scope["public_instruction"]["terminal_semantic_scope"]["profile_sha256"] = "0" * 64
    _, _, changed_checks = audit._model_projection(rendered=worker.prompt, receipt=changed_scope,
        repository=worker.case.root)
    assert not all(changed_checks.values())
    changed_mode = deepcopy(parsed)
    changed_mode["source_view_retrieval"]["source_edit_mode"] = "mapped-span@1"
    if metadata_view == "legacy":
        _, _, changed_checks = audit._model_projection(rendered=worker.prompt, receipt=changed_mode,
            repository=worker.case.root)
        assert not all(changed_checks.values())
    else:
        # Retained compact-selection accounting independently rejects changed
        # complete input before returning the final comparison dictionary.
        from ipfs_accelerate_py.agent_supervisor.runtime.representation_displacement import RepresentationDisplacementError
        with pytest.raises(RepresentationDisplacementError):
            audit._model_projection(rendered=worker.prompt, receipt=changed_mode,
                repository=worker.case.root)
    downgraded = deepcopy(parsed)
    downgraded["source_view_retrieval"]["schema"] = "worker-source-dispatch-observation@1"
    with pytest.raises(ValueError, match="dispatch schema"):
        audit._model_projection(rendered=worker.prompt, receipt=downgraded, repository=worker.case.root)
    record_property("authored_provider_fixture_calls", 1)
    record_property("external_provider_calls", 0)
    record_property("source_edit_performed", False)
    record_property("native_coding_child_launched", False)
    record_property("provider_token_savings_measured", False)


def test_joint_mapped_scope_retains_complete_omitted_source_closing_check(worker, provider, monkeypatch, capsys):
    from ipfs_accelerate_py import llm_router

    _, observed = provider
    _, options = mapped_endpoint(worker, monkeypatch)
    initialized = []

    def initialize(*args, **kwargs):
        initialized.append(True)
        (worker.case.root / "module_069.py").write_bytes(b"def changed_after_context(): return 11\n")
        return object()

    monkeypatch.setattr(llm_router, "get_llm_provider", initialize)
    capsys.readouterr()
    with pytest.raises(ValueError):
        invoke(worker, **options)
    receipt = json.loads(capsys.readouterr().out)
    assert initialized == [True] and observed == []
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "terminal_semantic_scope_currentness"
    assert receipt["terminal_semantic_scope_dispatch_rechecked"] is False
    assert receipt["source_view_retrieval"]["source_edit_mode"] == "mapped-span@2"
