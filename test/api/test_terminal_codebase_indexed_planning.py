"""Native current-repository inputs bind the real administrative plan snapshot."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_indexed_planning as api
from benchmarks.agent_supervisor.container_coding.terminal_codebase_repository_experiment import load_completed_intent_experiment
from benchmarks.agent_supervisor.container_coding.terminal_codebase_catalog_capture import capture_frozen_codebase_learning
from benchmarks.agent_supervisor.container_coding.terminal_codebase_repository_index import (
    build_current_repository_metadata, persist_repository_index, query_repository_index,
)

ROOT = Path(__file__).resolve().parents[4]
PARENT = ROOT / "artifacts/codebase_ir_terminal_bench/intent-qualification-20261001-02"


def _pin(path, value):
    raw = api._wire(value) + b"\n"
    path.write_bytes(raw)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture(scope="module")
def indexed_case(tmp_path_factory):
    parent = load_completed_intent_experiment(PARENT)
    output = tmp_path_factory.mktemp("actual-indexed-planning")
    native = build_current_repository_metadata(source_bytes=parent["sources"],
        repository_id=parent["declared"]["repository_cid"], task_spec=parent["declared"]["tasks"][0])
    proof = parent["result"]["proof_index"]
    index = persist_repository_index(current=native, proof_index_manifest=proof, output=output / "index")
    focus = parent["result"]["intent_codebase_matches"]["conditional_model_context"]["query"]
    query = query_repository_index(source_bytes=parent["sources"], repository_id=parent["declared"]["repository_cid"],
        task_spec=parent["declared"]["tasks"][0], proof_index_manifest=proof,
        expected=index, output=output / "index", query={"source_path": focus["source_path"],
            "symbols": focus["symbols"], "property": focus["property"], "limit": 12})
    learning = capture_frozen_codebase_learning(intent_experiment=PARENT, current_source_bytes=parent["sources"])
    learning_artifact = _pin(output / "learning.json", learning)
    inputs = {"intent_experiment": PARENT, "manifest": parent["capture"]["manifest"],
        "workflow_request": parent["capture"]["workflow_request"], "control": parent["result"]["intent_control"],
        "proof_index_manifest": proof,
        "match_result": parent["result"]["intent_codebase_matches"]["conditional_model_context"],
        "repository_index_manifest": index, "repository_query": query,
        "frozen_learning": learning, "learning_artifact": learning_artifact}
    result = api.build_indexed_repository_planning_snapshot(**inputs)
    return {"inputs": inputs, "result": result, "parent": parent}


def test_actual_current_index_changes_native_planning_identity_and_preserves_graph(indexed_case):
    new = indexed_case["result"]["snapshot"]
    old = indexed_case["parent"]["result"]["repository_planning_snapshot"]
    assert new["baseline_planning_snapshot_id"] == old["snapshot_id"]
    assert new["native_graph_cid"] == old["native_graph_cid"]
    assert new["native_input_snapshot"]["snapshot_cid"] != old["native_input_snapshot"]["snapshot_cid"]
    assert new["native_input_snapshot"]["material_binding"]["field_digests"]["extra"] != old["native_input_snapshot"]["material_binding"]["field_digests"]["extra"]
    assert new["native_input_snapshot"]["budget"]["max_model_calls"] == 0
    assert new["native_input_snapshot"]["material_binding"]["reuse_supported"] is True
    assert new["current_source_inventory"] == indexed_case["parent"]["capture"]["source_snapshot"]
    assert new["complete_declared_task_population_preserved"] is True
    assert new["current_behavioral_facts"] == new["behavioral_satisfied_requirements"] == []
    assert all(new[key] is False for key in api.AUTHORITY)
    assert new["public_request_planned"] is new["repository_evidence_admitted"] is False
    assert new["native_symbolic_receipt"]["observed_facts_supplied"] == 0
    assert new["indexed_planner_executed"] is True
    assert new["native_symbolic_receipt"]["input_snapshot_cid"] == new["native_input_snapshot"]["snapshot_cid"]
    assert new["native_symbolic_receipt"]["input_snapshot_cid"] != new["baseline_native_symbolic_receipt"]["input_snapshot_cid"]
    assert new["native_symbolic_receipt"]["portfolio_id"] != new["baseline_native_symbolic_receipt"]["portfolio_id"]
    assert new["native_symbolic_receipt"]["indexed_materials_sha256"] == new["full_materials_sha256"]


def test_actual_indexed_snapshot_replays_exactly(indexed_case):
    replayed = api.replay_indexed_repository_planning_snapshot(
        expected=indexed_case["result"]["snapshot"], **indexed_case["inputs"])
    assert replayed["snapshot"] == indexed_case["result"]["snapshot"]


@pytest.mark.parametrize("key,value", [
    ("property", "arbitrary_unreviewed_property"), ("source_path", ".supervisor-public-smoke.py"),
])
def test_match_focus_change_cannot_silently_keep_repository_query(indexed_case, key, value):
    inputs = deepcopy(indexed_case["inputs"])
    inputs["match_result"]["query"][key] = value
    with pytest.raises(ValueError):
        api.build_indexed_repository_planning_snapshot(**inputs)


def test_query_report_cannot_be_forged_with_only_false_authority_flags(indexed_case):
    inputs = deepcopy(indexed_case["inputs"])
    inputs["repository_query"]["forged_context"] = "unchecked replacement"
    with pytest.raises(ValueError, match="repository query differs"):
        api.build_indexed_repository_planning_snapshot(**inputs)


def test_unsupported_intent_domain_cannot_freeze_broader_repository_nominations(indexed_case):
    from ipfs_accelerate_py.agent_supervisor.planning.intent_codebase_matching import match_intent_codebase
    inputs = deepcopy(indexed_case["inputs"])
    authored = inputs["control"]["authored_control"]
    native = authored["native_document"]
    ref = native["sources"][0]
    identity = {key: ref[key] for key in ("ref_id", "source_uri", "source_id", "source_revision", "content_sha256")}
    query = deepcopy(inputs["match_result"]["query"])
    query["domain"]["input_types"][0]["type"] = "int"
    inputs["match_result"] = match_intent_codebase(intent_document=native, source_text=authored["text"],
        source_identity=identity, query=query, evidence_rows=inputs["match_result"]["evidence_rows"],
        current_source_snapshot=inputs["proof_index_manifest"]["source_snapshot"])
    assert inputs["match_result"]["model_nominations"] == []
    with pytest.raises(ValueError, match="query nominations differ"):
        api.build_indexed_repository_planning_snapshot(**inputs)


def test_inflight_caller_mutation_cannot_replace_the_validated_intent_domain(indexed_case, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.planning.intent_codebase_matching import match_intent_codebase
    from benchmarks.agent_supervisor.container_coding import terminal_codebase_repository_index as repository_index
    inputs = deepcopy(indexed_case["inputs"])
    authored = inputs["control"]["authored_control"]
    native = authored["native_document"]
    source_ref = native["sources"][0]
    identity = {key: source_ref[key] for key in ("ref_id", "source_uri", "source_id", "source_revision", "content_sha256")}
    changed_query = deepcopy(inputs["match_result"]["query"])
    changed_query["domain"]["input_types"][0]["type"] = "int"
    inputs["match_result"] = match_intent_codebase(intent_document=native, source_text=authored["text"],
        source_identity=identity, query=changed_query,
        evidence_rows=inputs["match_result"]["evidence_rows"],
        current_source_snapshot=inputs["proof_index_manifest"]["source_snapshot"])
    original_query = repository_index.query_repository_index
    def mutate_caller_after_native_query(**kwargs):
        result = original_query(**kwargs)
        inputs["match_result"]["model_nominations"] = deepcopy(
            indexed_case["inputs"]["match_result"]["model_nominations"])
        return result
    monkeypatch.setattr(repository_index, "query_repository_index", mutate_caller_after_native_query)
    with pytest.raises(ValueError, match="query nominations differ"):
        api.build_indexed_repository_planning_snapshot(**inputs)


def test_caller_modified_latent_is_refused_even_with_an_updated_artifact_pin(indexed_case, tmp_path):
    inputs = deepcopy(indexed_case["inputs"])
    inputs["frozen_learning"]["records"]["vectors"][0]["latent"][0] += 0.5
    inputs["learning_artifact"] = _pin(tmp_path / "forged-learning.json", inputs["frozen_learning"])
    with pytest.raises(ValueError, match="frozen learning differs"):
        api.build_indexed_repository_planning_snapshot(**inputs)


def test_raw_learning_artifact_drift_invalidates_reuse(indexed_case, tmp_path):
    inputs = deepcopy(indexed_case["inputs"])
    path = tmp_path / "learning.json"
    inputs["learning_artifact"] = _pin(path, inputs["frozen_learning"])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact pin differs"):
        api.build_indexed_repository_planning_snapshot(**inputs)


def test_learning_artifact_symlink_is_refused(indexed_case, tmp_path):
    inputs = deepcopy(indexed_case["inputs"])
    target = tmp_path / "learning.json"
    _pin(target, inputs["frozen_learning"])
    link = tmp_path / "alias.json"
    link.symlink_to(target)
    inputs["learning_artifact"] = {"path": str(link), "bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="canonical independent"):
        api.build_indexed_repository_planning_snapshot(**inputs)


def test_stale_indexed_snapshot_receipt_is_refused(indexed_case):
    expected = deepcopy(indexed_case["result"]["snapshot"])
    expected["current_behavioral_facts"] = [{"unchecked_behavior": True}]
    with pytest.raises(ValueError, match="snapshot differs"):
        api.replay_indexed_repository_planning_snapshot(expected=expected, **indexed_case["inputs"])
