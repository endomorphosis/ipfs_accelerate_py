"""Independent negative controls for instruction view audit and census plumbing."""
from copy import deepcopy
import subprocess

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_semantic_router_integration import provider  # noqa: F401
from test.api.test_semantic_instruction_dispatch import (
    expanded_selected, compose, invoke, counter, MODE,  # noqa: F401
)
from test.api.test_semantic_instruction_view import case, native_factory, project, prompts  # noqa: F401
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_metadata_census import (
    SemanticMetadataCensusError, census_semantic_metadata_view,
)


@pytest.mark.parametrize("compact_tokens,selected_mode", [(90, MODE), (200, "legacy")])
def test_historical_instruction_audit_without_counter_keeps_recorded_delivery_only(
        expanded_selected, provider, monkeypatch, compact_tokens, selected_mode):
    state = compose(expanded_selected, monkeypatch)
    _, receipt = invoke(state, semantic_metadata_view=MODE,
        semantic_metadata_token_counter=counter(compact_tokens))
    expected = provider[1][-1][0]
    parsed = audit._receipt({**receipt, "phase": "coding"}, state["request"]["workspace"].parent)
    assert parsed["semantic_metadata_view"]["selected_mode"] == selected_mode
    (state["root"] / "answer.py").write_text("def answer():\n    return 2\n")
    (state["root"] / "test_answer.py").write_text("changed after invocation")
    subprocess.run(["git", "-C", str(state["root"]), "worktree", "remove", "--force",
        str(state["request"]["workspace"])], check=True)
    monkeypatch.chdir(state["root"].parent)
    model, _, checks = audit._model_projection(rendered=state["request"]["prompt"],
        receipt=parsed, repository=state["root"])
    assert model == expected
    assert checks["semantic_metadata_recorded_prompt_identities"] is True
    assert checks["semantic_metadata_token_counter_verified"] is False
    assert checks["semantic_metadata_selection_receipt"] is False
    assert all(value for key, value in checks.items() if key not in {
        "semantic_metadata_token_counter_verified", "semantic_metadata_selection_receipt"})


@pytest.mark.parametrize("requested", [None, [], "legacy", "common-bindings@1", "foreign@1"])
def test_selected_instruction_view_cannot_be_relabelled_by_receipt(
        expanded_selected, provider, monkeypatch, requested):
    state = compose(expanded_selected, monkeypatch)
    _, receipt = invoke(state, semantic_metadata_view=MODE,
        semantic_metadata_token_counter=counter(90))
    if requested is None:
        receipt["semantic_metadata_view"].pop("requested_mode")
    else:
        receipt["semantic_metadata_view"]["requested_mode"] = requested
    with pytest.raises(ValueError):
        audit._receipt({**receipt, "phase": "coding"}, state["request"]["workspace"].parent)


def test_fallback_cannot_hide_instruction_treatment_by_deleting_requested_mode(
        expanded_selected, provider, monkeypatch):
    state = compose(expanded_selected, monkeypatch)
    _, receipt = invoke(state, semantic_metadata_view=MODE,
        semantic_metadata_token_counter=counter(200))
    assert receipt["semantic_metadata_view"]["selected_mode"] == "legacy"
    receipt["semantic_metadata_view"].pop("requested_mode")
    parsed = audit._receipt({**receipt, "phase": "coding"}, state["request"]["workspace"].parent)
    _model, _, checks = audit._model_projection(rendered=state["request"]["prompt"],
        receipt=parsed, repository=state["root"])
    assert checks["semantic_metadata_recorded_prompt_identities"] is False
    assert checks["semantic_metadata_selection_receipt"] is False


@pytest.mark.parametrize("mutation", ["missing-both", "missing-receipt", "foreign-context"])
def test_census_requires_independent_matching_instruction_binding(case, mutation):
    view, block, receipt = project(case), case[3], deepcopy(case[4])
    native, candidate = prompts(view, block)
    options = {"instruction_block": block, "instruction_receipt": receipt}
    if mutation == "missing-both":
        options = {}
    elif mutation == "missing-receipt":
        options.pop("instruction_receipt")
    else:
        receipt["context_cid"] = "cid:foreign-context"
    with pytest.raises(SemanticMetadataCensusError):
        census_semantic_metadata_view(view=view, native_complete_prompt=native,
            candidate_complete_prompt=candidate, **options)
