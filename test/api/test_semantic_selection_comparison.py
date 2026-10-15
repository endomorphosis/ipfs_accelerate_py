"""Authored comparison controls; codec authority is explicitly substituted.

Native capsule/index records and reference chunking are real. Encode/restore
seams are fixture callbacks, so these controls establish comparison behavior,
not producer freshness, V2 compression, provider compatibility or acceptance.
The companion probe supplies the genuine native producer/codec qualification.
"""
from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    build_text_context_references,
)
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    PromptTokenCounter,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_selection_comparison import (
    compare_semantic_selection,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.capsules import admit_capsule
from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
    SemanticCapsule,
    SortedPairIndex,
    canonical_dag_json_bytes,
    cid_for_structured,
)


def _identity(value):
    return cid_for_structured({"authored_identity": value})


def _counter(callback=None, **kwargs):
    return PromptTokenCounter(**{
        "provider": "codex_cli", "model": "fixture-model", "tokenizer_id": "authored:utf8",
        "tokenizer_revision": "fixture-v1", "special_token_policy": "fixture-only", **kwargs,
    }, count_tokens=callback or (lambda text: len(text.encode())))


@pytest.fixture
def authored(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    source = "# Keep literal source s0 and $semantic_ref.\ndef target(value):\n    return helper(value)\n"
    source_cid = _identity("authored-source")
    target_id, helper_id = _identity("target"), _identity("helper")
    capsules = [SemanticCapsule(
        stable_symbol_id=identity, version_cid=_identity("version-" + name),
        semantic_index_schema="authored-semantic-index@1", extractor_version="authored-extractor@1",
        source_slice_path="code.py", source_cid=source_cid,
        signature={"name": name}, dependency_stable_ids=(helper_id,) if name == "target" else (),
    ) for name, identity in (("target", target_id), ("helper", helper_id))]
    capsule_rows = [capsule.to_dict() for capsule in capsules]
    index = SortedPairIndex(pairs=[(row["stable_symbol_id"], row["capsule_cid"]) for row in capsule_rows])
    root_identity = {"capsule_index_cid": index.index_cid, "authored_root_control": True}
    root_cid = cid_for_structured(root_identity)
    admissions = [admit_capsule(capsule, semantic_state_root_cid=root_cid, force_raw_source=True).to_dict()
                  for capsule in capsules]
    full = {
        "schema": "supervisor-semantic-worker-context@1", "task_id": "TASK-CONTROL",
        "objective": "Inspect target without changing helper.", "scope_cid": _identity("scope"),
        "semantic_root_cid": root_cid, "manifest": {"code.py": {
            "sha256": sha256(source.encode()).hexdigest(), "source_cid": source_cid}},
        "required_raw_paths": ["code.py"], "preparation_bounds": {"max_symbols": 32, "context_input_tokens": 32768},
        "reconstruction": {"authored_control": True, "semantic_root_cid": root_cid},
        "refresh_lineage": None, "doctor_snapshot_id": "authored:doctor", "doctor_manifest_cid": _identity("doctor"),
        "completion_authority": False, "capsules": capsule_rows, "admissions": admissions,
        "raw_sources": {"code.py": source},
        "pack": {"schema": "authored-pack-control@1", "decisions": ["inspect-target", "fetch-helper"],
                 "coverage_satisfied": True, "budget_exceeded": False,
                 "pack": {"assumptions": ["Both sources are present."], "risk": "high", "route": "review",
                          "obligation_cids": [_identity("helper-obligation")], "counterexample_cids": []}},
    }
    selected = deepcopy(full)
    selected["capsules"] = selected["capsules"][:1]
    selected["admissions"] = selected["admissions"][:1]
    selected["pack"]["decisions"] = ["inspect-target"]
    selected["pack"]["pack"]["obligation_cids"] = []
    selected["worker_projection"] = {
        "schema": "supervisor-semantic-worker-projection@1", "selector": "lexical-query-symbol-name@1",
        "query": "target", "max_capsules": 1, "max_bytes": 65536,
        "full_capsule_count": 2, "selected_capsule_count": 1, "omitted_capsule_count": 1,
        "selected_symbols": [{"path": "code.py", "qualified_name": "code.target", "stable_symbol_id": target_id}],
        "capsule_index_cid": index.index_cid, "raw_source_fetch_required": {},
        "omission_reason": "Authored capsule selection control.", "raw_required_caveat": "Read exact source before editing.",
        "semantic_equivalence_claimed": False, "completion_authority": False,
    }
    for arm in ("full", "selected"):
        blocks = repository / arm / "blocks"
        blocks.mkdir(parents=True)
        (blocks / root_cid).write_bytes(canonical_dag_json_bytes(root_identity))
        (blocks / index.index_cid).write_bytes(canonical_dag_json_bytes(index.identity_payload()))

    state = {"fresh": True, "encodes": [], "restores": 0}

    def encode(*, prompt, repository, slot_policy_revision):
        assert slot_policy_revision == codec.SLOT_POLICY_V2
        state["encodes"].append(prompt)
        return SimpleNamespace(provider_prompt=codec._json({"authored_alias_wire": sha256(prompt.encode()).hexdigest()}),
                               table=SimpleNamespace(native=prompt))

    def restore(*, provider_prompt, table, repository):
        state["restores"] += 1
        if not state["fresh"]:
            raise ValueError("authored source freshness refusal")
        return table.native

    monkeypatch.setattr(codec, "encode_semantic_router_prompt", encode)
    monkeypatch.setattr(codec, "restore_semantic_router_prompt", restore)
    return SimpleNamespace(repository=repository, full=full, selected=selected, state=state,
                           target_id=target_id, helper_id=helper_id, source=source)


def _prompt(control, payload, arm, *, core=None, suffix="\nLiteral acceptance remains unchanged.\n",
            chunk_bytes=6144, unrelated=None):
    refs = build_text_context_references(
        codec._json(payload), reference_prefix="semantic-context", kind="semantic-context",
        path=arm + "/worker-context.json", repository_id="repo:control", tree_id="tree:control",
        required=True, chunk_bytes=chunk_bytes, coverage_ids=("requirement:public",))
    wire = {"objective_id": "TASK-CONTROL", "repository_id": "repo:control", "tree_id": "tree:control",
            "goal": {"objective": "Inspect target"}, "scope": {"allowed_paths": ["code.py"]},
            "authority": {"completion_authority": False}, "acceptance": {"criteria": ["preserve helper"]},
            "evidence": [ref.to_dict() for ref in refs] + (unrelated or [])}
    if core is not None:
        wire.update(core)
    return codec._json(wire) + suffix


def _compare(control, *, full_prompt=None, selected_prompt=None, **kwargs):
    return compare_semantic_selection(**{
        "repository": control.repository, "workspace": control.repository, "purpose": "coding",
        "provider": "codex_cli", "model": "fixture-model", "measurement_basis": "authored_fixture",
        "full_prompt": full_prompt or _prompt(control, control.full, "full"),
        "selected_prompt": selected_prompt or _prompt(control, control.selected, "selected"), **kwargs,
    })


def test_authored_complete_measurement_retains_only_identities_and_decision_differences(authored):
    result = _compare(authored, token_counter=_counter())
    assert result["measurement_status"] == "measured"
    assert all(row["input_tokens"] == row["model_prompt_bytes"] for row in result["representations"].values())
    assert result["matched_source_inventory"]["all_scoped_raw_sources_identically_embedded"] is True
    assert result["selection"]["selected_capsules_and_admissions_exact_subsets"] is True
    assert "pack.decisions" in result["pack_decision_differences"]["changed_fields"]
    assert "pack.pack.obligation_cids" in result["pack_decision_differences"]["changed_fields"]
    assert result["dependency_audit"]["missing_selected_dependency_symbol_ids"] == [authored.helper_id]
    assert result["dependency_audit"]["declared_graph_closure_selected"] is False
    assert result["dependency_closure_verified"] is result["deployment_ready"] is False
    assert result["task_acceptance_verified"] is result["runtime_activation"] is False
    assert result["whole_session_token_savings"] is result["campaign_usage_complete"] is None
    assert result["provider_calls"] == 0
    assert authored.source not in json.dumps(result)
    assert "fetch-helper" not in json.dumps(result)
    assert authored.state["restores"] == 4


def test_absent_counter_reports_four_unknown_counts_and_no_local_token_delta(authored):
    result = _compare(authored)
    assert result["measurement_status"] == "unavailable"
    assert result["token_measurements_consistent"] is None
    assert all(row["input_tokens"] is None for row in result["representations"].values())
    assert all(row["rendered_user_content_token_delta"] is None
               for row in result["comparisons_to_full_native"].values())


@pytest.mark.parametrize("mismatch", ["core", "suffix", "unrelated"])
def test_changed_native_invariant_or_unrelated_evidence_refuses(authored, mismatch):
    kwargs = {"core": {"scope": {"allowed_paths": ["other.py"]}}} if mismatch == "core" else (
        {"suffix": "\nChanged acceptance.\n"} if mismatch == "suffix" else {"unrelated": [{"kind": "authored-other"}]})
    with pytest.raises(ValueError, match="core|suffix|unrelated"):
        _compare(authored, selected_prompt=_prompt(authored, authored.selected, "selected", **kwargs))


@pytest.mark.parametrize("field", ["manifest", "scope_cid", "semantic_root_cid"])
def test_changed_source_scope_or_original_producer_root_refuses(authored, field):
    if field == "manifest":
        authored.selected[field]["code.py"]["sha256"] = "0" * 64
    else:
        authored.selected[field] = _identity("changed-" + field)
    with pytest.raises(ValueError, match="bindings differ"):
        _compare(authored)


@pytest.mark.parametrize("mutation", ["drop", "change"])
def test_source_elision_or_changed_raw_bytes_is_outside_this_control(authored, mutation):
    if mutation == "drop":
        authored.selected["raw_sources"].clear()
    else:
        authored.selected["raw_sources"]["code.py"] += "# changed\n"
    with pytest.raises(ValueError, match="every scoped raw source"):
        _compare(authored)


@pytest.mark.parametrize("mutation", ["capsule", "admission"])
def test_changed_nominated_capsule_or_admission_refuses_exact_subset_claim(authored, mutation):
    if mutation == "capsule":
        authored.selected["capsules"][0]["signature"]["name"] = "changed"
    else:
        authored.selected["admissions"][0]["caveats"].append("changed caveat")
    with pytest.raises(ValueError, match="exact full-context subset"):
        _compare(authored)


def test_baseline_must_cover_complete_root_backed_capsule_index(authored):
    authored.full["capsules"] = authored.full["capsules"][:1]
    authored.full["admissions"] = authored.full["admissions"][:1]
    with pytest.raises(ValueError, match="complete producer index"):
        _compare(authored)


def test_changed_projection_denominator_or_index_binding_refuses(authored):
    authored.selected["worker_projection"]["full_capsule_count"] = 3
    with pytest.raises(ValueError, match="denominator differ"):
        _compare(authored)
    authored.selected["worker_projection"]["full_capsule_count"] = 2
    authored.selected["worker_projection"]["capsule_index_cid"] = _identity("foreign-index")
    with pytest.raises(ValueError, match="denominator differ"):
        _compare(authored)


def test_different_reference_packaging_refuses_before_measurement(authored):
    with pytest.raises(ValueError, match="6144-byte reference chunk policy"):
        _compare(authored, selected_prompt=_prompt(authored, authored.selected, "selected", chunk_bytes=500))


def test_partial_failed_counter_discards_all_four_counts(authored):
    seen = []

    def callback(text):
        seen.append(text)
        if len(seen) == 4:
            raise RuntimeError("authored counter failure")
        return len(text.encode())

    result = _compare(authored, token_counter=_counter(callback))
    assert result["measurement_status"] == "failed"
    assert result["measurement_error_type"] == "RuntimeError"
    assert all(row["input_tokens"] is None for row in result["representations"].values())
    assert all(row["rendered_user_content_token_delta"] is None
               for row in result["comparisons_to_full_native"].values())


def test_repeated_native_counter_drift_discards_all_counts_and_refuses_strict_budget(authored):
    observations = {}

    def callback(text):
        observations[text] = observations.get(text, 0) + 1
        return len(text.encode()) + observations[text]

    result = _compare(authored, token_counter=_counter(callback))
    assert result["measurement_status"] == "failed"
    assert result["measurement_error_type"] == "InconsistentTokenCount"
    assert result["token_measurements_consistent"] is False
    assert all(row["input_tokens"] is None for row in result["representations"].values())
    with pytest.raises(ValueError, match="consistent pinned counter"):
        _compare(authored, token_counter=_counter(callback), max_input_tokens=100000)


def test_bound_counter_and_complete_prompt_ceiling_are_not_relaxed(authored):
    with pytest.raises(ValueError, match="pinned provider/model"):
        _compare(authored, token_counter=_counter(model="other-model"))
    with pytest.raises(ValueError, match="working pinned"):
        _compare(authored, max_input_tokens=100000)
    with pytest.raises(ValueError, match="token budget"):
        _compare(authored, token_counter=_counter(), max_input_tokens=1)


def test_current_source_is_rechecked_after_external_counter_callback(authored):
    def callback(text):
        authored.state["fresh"] = False
        return len(text.encode())

    with pytest.raises(ValueError, match="freshness refusal"):
        _compare(authored, token_counter=_counter(callback))
