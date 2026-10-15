"""Current producer evidence reaches the opt-in projection and router path.

Only provider responses and local token counts are authored fixtures. Native
source preparation, context compilation, allocated worktrees, parent custody,
projection, identifier translation and response validation execute normally.
"""

import copy
import hashlib
import io
import json
import runpy
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    build_text_context_references,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextReference
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    PromptTokenCounter,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_projection_dispatch import (
    project_semantic_router_prompt,
)
from test.api.test_semantic_router_integration import _run
from test.api.test_semantic_router_translation import _reply

pytest_plugins = ("test.api.test_semantic_router_integration",)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _view(prompt):
    wire, _ = json.JSONDecoder().raw_decode(prompt)
    references = [row for row in wire["evidence"] if row["kind"] == "semantic-context"]
    return json.loads("".join(row["summary"] for row in references))


def test_projection_preserves_complete_decision_core_sources_and_original_artifact(native):
    repository, artifact, prompt, source = native
    original = artifact.read_bytes()
    projected = project_semantic_router_prompt(prompt=prompt, repository=repository)
    view = _view(projected.projected_prompt)
    old = json.loads(original)
    required = view["required_context"]
    assert required["raw_sources"]["mod.py"] == source
    for key in (
        "objective", "manifest", "required_raw_paths", "capsules", "admissions",
        "reconstruction", "refresh_lineage", "worker_projection",
    ):
        if key in old:
            assert required[key] == old[key]
    assert required["pack"]["coverage_satisfied"] == old["pack"]["coverage_satisfied"]
    for key in ("assumptions", "exclusions", "risk", "route", "obligation_cids", "counterexample_cids"):
        assert required["pack"]["pack"][key] == old["pack"]["pack"][key]
    native_wire, native_end = json.JSONDecoder().raw_decode(prompt)
    projected_wire, projected_end = json.JSONDecoder().raw_decode(projected.projected_prompt)
    assert {key: value for key, value in projected_wire.items() if key != "evidence"} == {
        key: value for key, value in native_wire.items() if key != "evidence"
    }
    assert projected.projected_prompt[projected_end:] == prompt[native_end:]
    assert artifact.read_bytes() == original
    assert projected.receipt["deferred_reference_count"] == 1
    assert projected.receipt["total_session_token_savings"] is None
    assert projected.receipt["provider_called"] is False


def test_original_required_coverage_and_unrelated_evidence_order_are_preserved(native):
    repository, _, prompt, _ = native
    wire, end = json.JSONDecoder().raw_decode(prompt)
    for index, row in enumerate(wire["evidence"]):
        row["metadata"]["coverage_ids"] = ["source:target", "acceptance:original"]
        row.pop("content_id", None)
        row.pop("reference_content_id", None)
        wire["evidence"][index] = ContextReference.from_dict(row).to_record()
    extra = build_text_context_references(
        "Exact current failure and original test requirement.", reference_prefix="failure",
        kind="failure-context", repository_id=wire["repository_id"], tree_id=wire["tree_id"], required=True,
    )[0].to_record()
    wire["evidence"].insert(1, extra)
    bound_prompt = _json(wire) + prompt[end:]
    projected = project_semantic_router_prompt(prompt=bound_prompt, repository=repository)
    actual, _ = json.JSONDecoder().raw_decode(projected.projected_prompt)
    assert [row for row in actual["evidence"] if row["kind"] != "semantic-context"] == [extra]
    for row in actual["evidence"]:
        if row["kind"] == "semantic-context":
            assert row["metadata"]["required"] is True
            assert set(row["metadata"]["coverage_ids"]) >= {"source:target", "acceptance:original"}


def test_versioned_codec_round_trips_projected_prompt_without_changing_native_parent(native):
    repository, artifact, prompt, _ = native
    before = artifact.read_bytes()
    projected = project_semantic_router_prompt(prompt=prompt, repository=repository)
    encoded = codec.encode_semantic_router_prompt(
        prompt=projected.projected_prompt, repository=repository,
        slot_policy_revision="closed-typed-identifiers@2",
    )
    assert encoded.table.slot_policy_revision == "closed-typed-identifiers@2"
    assert codec.restore_semantic_router_prompt(
        provider_prompt=encoded.provider_prompt, table=encoded.table, repository=repository,
    ) == projected.projected_prompt
    assert artifact.read_bytes() == before
    assert json.loads(encoded.table.payload_json)["semantic_artifact"] == artifact.relative_to(repository).as_posix()


@pytest.mark.parametrize("mutation", ["source", "capsule", "coverage", "parent", "extra"])
def test_repaired_role_view_chunk_digests_cannot_change_required_context_or_parent(native, mutation):
    repository, _, prompt, _ = native
    projected = project_semantic_router_prompt(prompt=prompt, repository=repository)
    wire, end = json.JSONDecoder().raw_decode(projected.projected_prompt)
    view = copy.deepcopy(_view(projected.projected_prompt))
    if mutation == "source":
        view["required_context"]["raw_sources"]["mod.py"] += "# forged source\n"
    elif mutation == "capsule":
        view["required_context"]["capsules"].pop()
    elif mutation == "coverage":
        view["required_context"]["pack"]["coverage_satisfied"] = False
    elif mutation == "parent":
        view["origin"]["expected_sha256"] = "0" * 64
    else:
        view["future_authority"] = True
    old_refs = [row for row in wire["evidence"] if row["kind"] == "semantic-context"]
    forged = build_text_context_references(
        _json(view), reference_prefix="semantic-context", kind="semantic-context", path=old_refs[0]["path"],
        repository_id=wire["repository_id"], tree_id=wire["tree_id"], required=True,
    )
    wire["evidence"] = [reference.to_record() for reference in forged]
    with pytest.raises(ValueError):
        codec.encode_semantic_router_prompt(
            prompt=_json(wire) + projected.projected_prompt[end:], repository=repository,
            slot_policy_revision="closed-typed-identifiers@2",
        )


def test_projection_requires_current_source(native):
    repository, _, prompt, source = native
    (repository / "mod.py").write_text(source + "# same-head concurrent source edit\n")
    with pytest.raises(ValueError, match="stale"):
        project_semantic_router_prompt(prompt=prompt, repository=repository)


def test_router_opt_in_uses_projection_and_native_symbol_candidate_validation(allocated, provider):
    repository, _, prompt, _ = allocated
    state, observed = provider
    projected = project_semantic_router_prompt(prompt=prompt, repository=repository)
    encoded = codec.encode_semantic_router_prompt(
        prompt=projected.projected_prompt, repository=repository,
        slot_policy_revision="closed-typed-identifiers@2",
    )
    reply, symbol = _reply(encoded)
    state["reply"] = json.dumps(reply)
    output, receipt = _run(
        repository, prompt, semantic_slot_policy_revision="closed-typed-identifiers@2",
        semantic_context_projection=True,
        prompt_token_counter=PromptTokenCounter(
            provider="codex_cli", model="pinned", tokenizer_id="fixture:projection-selection",
            tokenizer_revision="test-v1", special_token_policy="authored-fixture-only",
            count_tokens=lambda text: 100 if "supervisor-semantic-worker-role-view@1" in text else 1000,
        ),
    )
    assert len(observed) == 1
    assert receipt["semantic_slot_policy_revision"] == "closed-typed-identifiers@2"
    assert receipt["semantic_context_projection"] == projected.receipt
    assert receipt["semantic_context_projection_selected"] is True
    assert receipt["native_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert receipt["prompt_selection"]["selected_representation"] == "semantic"
    assert receipt["semantic_response_translation"]["native_grammar_validated"] is True
    assert json.loads(output)["structured_payload"]["symbol_ids"] == [symbol]
    assert receipt["completion_authority"] is False


def test_router_can_keep_full_context_aliases_when_role_view_overhead_is_larger(allocated, provider):
    repository, _, prompt, _ = allocated
    _, observed = provider
    _, receipt = _run(
        repository, prompt, semantic_slot_policy_revision="closed-typed-identifiers@2",
        semantic_context_projection=True,
    )
    assert len(observed) == 1
    assert receipt["prompt_selection"]["selected_representation"] == "semantic"
    assert receipt["prompt_selection"]["selected_candidate"] == "packed_v2"
    assert receipt["semantic_context_projection_selected"] is False
    assert receipt["semantic_slot_policy_selected"] == "closed-typed-identifiers@2"
    assert receipt["semantic_reference_packing_selected"] == "context-reference-columns@1"
    assert receipt["prompt_selection"]["comparison_candidates"]["v2"]["input_tokens"] < (
        receipt["prompt_selection"]["comparison_candidates"]["projected_v2"]["input_tokens"]
    )
    comparison = receipt["prompt_selection"]["comparison_candidates"]
    assert comparison["packed_v2"]["input_tokens"] < comparison["v2"]["input_tokens"]
    assert comparison["packed_v2"]["input_tokens"] == min(
        row["input_tokens"] for row in comparison.values() if row["available"])


def test_missing_explicit_counter_uses_estimates_for_projection_competition(allocated, provider):
    repository, _, prompt, _ = allocated
    _, observed = provider
    _, receipt = _run(
        repository, prompt, semantic_slot_policy_revision="closed-typed-identifiers@2",
        semantic_context_projection=True, prompt_token_counter=None,
    )
    assert len(observed) == 1
    selection = receipt["prompt_selection"]
    assert selection["measurement_status"] == "estimated"
    assert selection["selected_content_tokens"] is None
    assert selection["selected_estimated_content_tokens"] > 0
    rows = selection["comparison_candidates"].values()
    assert all(row["input_tokens"] is None for row in rows)
    assert selection["tokenizer_binding"]["model"] == "pinned"
    if selection["selected_candidate"] is not None:
        counts = [row["estimated_input_tokens"] for row in selection["comparison_candidates"].values()
                  if row["available"]]
        assert selection["selected_estimated_content_tokens"] == min(counts)
    else:
        assert receipt["router_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert receipt["semantic_context_projection"]["total_session_token_savings"] is None


def test_unknown_policy_and_projection_without_versioning_refuse_before_provider(allocated, provider):
    repository, _, prompt, _ = allocated
    _, observed = provider
    for options in (
        {"semantic_slot_policy_revision": "future@99"},
        {"semantic_context_projection": True},
        {"semantic_context_projection": "yes"},
    ):
        with pytest.raises(ValueError):
            _run(repository, prompt, **options)
    assert observed == []


def test_module_main_uses_canonical_bound_counter_for_candidate_selection(allocated, provider, monkeypatch, capsys):
    from ipfs_accelerate_py.agent_supervisor.runtime import local_prompt_tokenizer

    repository, _, prompt, _ = allocated
    _, observed = provider
    counter = PromptTokenCounter(
        provider="codex_cli", model="pinned", tokenizer_id="fixture:module-main-byte-counter",
        tokenizer_revision="test-v1", special_token_policy="authored-fixture-only",
        count_tokens=lambda text: len(text.encode()),
    )
    # The local artifact loader is an authored seam; canonical run constructs
    # its real PromptTokenCounter from this exact provider/model binding.
    local = SimpleNamespace(
        binding=counter.binding, count=counter.count,
        provenance=lambda: {"measurement_basis": "authored_fixture"},
    )
    monkeypatch.setattr(local_prompt_tokenizer, "load_owner_prompt_tokenizer", lambda **kwargs: local)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(prompt.encode())))
    monkeypatch.setattr(sys, "argv", [
        "router_implementation_runner", "--model", "pinned", "--timeout", "1",
        "--semantic-repository", str(repository),
        "--semantic-slot-policy-revision", "closed-typed-identifiers@2",
        "--semantic-context-projection", "--prompt-tokenizer-artifact", "authored-descriptor",
        "--prompt-tokenizer-sha256", "0" * 64,
    ])
    module = "ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner"
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as stopped:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
    assert stopped.value.code == 0
    assert len(observed) == 1
    receipts = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    receipt = next(row for row in receipts if row.get("schema") == "router-implementation-invocation@1")
    assert receipt["prompt_selection"]["selected_candidate"] == "packed_v2"
    assert receipt["prompt_selection"]["tokenizer_binding"] == counter.binding()
    assert receipt["prompt_tokenizer_source"] == {"measurement_basis": "authored_fixture"}
