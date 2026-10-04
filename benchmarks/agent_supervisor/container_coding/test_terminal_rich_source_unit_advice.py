"""Real trained child inference, sidecar replay and bounded supervisor summary.

Set INTENT_RICH_TEST_CHECKPOINT to an installed rich child descriptor. These
integration cases never substitute an accepted candidate or neural backend.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_source_unit_advice as api

INSTRUCTION = "agent must inspect cache."


@pytest.fixture(scope="module")
def published_child():
    default = Path(__file__).resolve().parents[5] / "artifacts/intent-rich-decoder-20261002/inference-01/descriptor.json"
    selected = Path(os.environ.get("INTENT_RICH_TEST_CHECKPOINT", str(default)))
    if not selected.exists():
        pytest.skip("frozen rich child descriptor is not installed; set INTENT_RICH_TEST_CHECKPOINT")
    from ipfs_datasets_py.logic.intent_ir.formalize.rich_decoder import load_rich_intent_checkpoint
    value = json.loads(selected.read_text())
    load_rich_intent_checkpoint(value)
    return value


@pytest.fixture
def actual_advice(tmp_path, published_child):
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(json.dumps(published_child, sort_keys=True))
    advice = api.prepare_source_unit_advice(instruction=INSTRUCTION, enabled=True,
        intent_descriptor_path=descriptor, project_logic_families=True,
        requested_intent_families=["dcec", "higher_order"])
    assert advice["status"] == "source_unit_candidate_advice", advice
    assert advice["report"]["counts"]["intent_encoder_executions"] >= 1
    assert advice["report"]["counts"]["intent_decoder_executions"] >= 1
    return advice, descriptor


def write_advice(tmp_path, value):
    path = tmp_path / "advice.json"
    raw = api._wire(value)
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_real_rich_child_replays_and_preserves_ast_in_supervisor_summary(tmp_path, actual_advice):
    advice, descriptor = actual_advice
    path, digest = write_advice(tmp_path, advice)
    loaded, replay = api.load_source_unit_advice(path=path, expected_sha256=digest, instruction=INSTRUCTION)
    assert loaded == advice and replay["inference_replays"] == 1
    assert loaded["schema"] == api.FAMILY_SCHEMA
    assert loaded["descriptor_pins"]["intent"]["sha256"] == hashlib.sha256(descriptor.read_bytes()).hexdigest()
    summary = api.source_unit_planner_summary(loaded, maximum_bytes=8192)
    payload = json.loads(summary.split("\n", 1)[1])
    candidate = payload["candidates"][0]
    assert candidate["rich_ir"]["schema"] == "intent-rich-ir/v1"
    assert candidate["rich_ir"]["ast"] == {"kind": "atom", "actor": "agent", "action": "inspect", "object": "cache", "modality": "required"}
    assert candidate["logic_families"] == ["dcec", "higher_order"]
    assert candidate["logic_projection_sha256"]
    assert payload["counts"]["intent_candidates"] == 1
    assert "no proof" in payload["limitations"]
    assert not payload["execution_authority"] and loaded["raw_instruction_preserved"]
    assert loaded["before_goal_decomposition"] and loaded["continue_planning"]


def test_real_rich_descriptor_change_invalidates_sidecar_before_replay(tmp_path, actual_advice):
    advice, descriptor = actual_advice
    path, digest = write_advice(tmp_path, advice)
    changed = json.loads(descriptor.read_text())
    changed["sha256"] = "0" * 64
    descriptor.write_text(json.dumps(changed))
    loaded, replay = api.load_source_unit_advice(path=path, expected_sha256=digest, instruction=INSTRUCTION)
    assert loaded["status"] == "fail_open_invalid_sidecar" and replay["inference_replays"] == 0
    assert loaded["continue_planning"] and loaded["raw_instruction_preserved"]
    assert api.source_unit_planner_summary(loaded, maximum_bytes=8192) is None


def test_rehashed_rich_candidate_change_cannot_bypass_actual_inference_replay(tmp_path, actual_advice):
    advice, _ = actual_advice
    forged = deepcopy(advice)
    forged["report"]["candidates"][0]["rich_ir"]["ast"]["modality"] = "permitted"
    forged["report"]["report_sha256"] = api._sha(api._wire({key: value for key, value in forged["report"].items() if key != "report_sha256"}))
    forged["advice_sha256"] = api._sha(api._wire({key: value for key, value in forged.items() if key != "advice_sha256"}))
    path, digest = write_advice(tmp_path, forged)
    loaded, replay = api.load_source_unit_advice(path=path, expected_sha256=digest, instruction=INSTRUCTION)
    assert loaded["status"] == "fail_open_invalid_sidecar" and replay["inference_replays"] == 1
    assert loaded["continue_planning"] and api.source_unit_planner_summary(loaded, maximum_bytes=8192) is None


def test_real_rich_advice_does_not_relax_planner_token_bound(actual_advice):
    advice, _ = actual_advice
    with pytest.raises(ValueError, match="planner bound"):
        api.source_unit_planner_summary(advice, maximum_bytes=128)


def test_fixed_authored_conditional_reaches_consumer_without_dropping_guard(tmp_path, published_child):
    # A declared training control checks transport/inference, not held-out accuracy.
    corpus = json.loads((Path(published_child["path"]).parent / "corpus.json").read_text())
    example = next(row for row in corpus["samples"] if row["id"] == "authored-rich:train:if:0")
    instruction = example["instruction"]
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(json.dumps(published_child))
    advice = api.prepare_source_unit_advice(instruction=instruction, enabled=True,
        intent_descriptor_path=descriptor, project_logic_families=True,
        requested_intent_families=["higher_order", "dcec"])
    assert advice["status"] == "source_unit_candidate_advice", advice
    path, digest = write_advice(tmp_path, advice)
    loaded, replay = api.load_source_unit_advice(path=path, expected_sha256=digest, instruction=instruction)
    assert loaded == advice and replay["inference_replays"] == 1
    payload = json.loads(api.source_unit_planner_summary(loaded, maximum_bytes=8192).split("\n", 1)[1])
    ast = payload["candidates"][0]["rich_ir"]["ast"]
    assert ast == example["ast"] and ast["kind"] == "if"
    unit = next(row for row in loaded["report"]["rich_intent"]["units"] if row["accepted"])
    assert unit["inference"]["logic"]["formula"]["op"] == "implies"
    assert unit["inference"]["logic"]["native_intent_ir"] is None
    assert unit["atomic_family_projection"] is None and not unit["selected_native_targets"]


def test_fixed_held_out_conditional_discloses_grammar_composition_and_actual_leaf_inference(tmp_path, published_child):
    # This predeclared held-out control specifically exercises hybrid recovery;
    # it does not measure the accuracy of neural whole-instruction decoding.
    corpus = json.loads((Path(published_child["path"]).parent / "corpus.json").read_text())
    example = next(row for row in corpus["samples"] if row["id"] == "authored-rich:test:if:0")
    instruction = example["instruction"]
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(json.dumps(published_child))
    advice = api.prepare_source_unit_advice(instruction=instruction, enabled=True,
        intent_descriptor_path=descriptor, project_logic_families=True,
        requested_intent_families=["higher_order", "dcec"])
    assert advice["status"] == "source_unit_candidate_advice", advice
    unit = next(row for row in advice["report"]["rich_intent"]["units"] if row["accepted"])
    inference = unit["inference"]
    assert inference["schema"] == "intent-grammar-composed-roundtrip/v1"
    assert inference["direct_report"]["rich_ir"] is None
    assert inference["phase_counts"]["direct"]["encoder_executions"] >= 1
    assert len(inference["neural_leaves"]) == 1
    leaf = inference["neural_leaves"][0]
    assert leaf["inference"]["rich_ir"]["ast"] == example["ast"]["body"]
    assert leaf["inference"]["counts"]["encoder_executions"] >= 1
    assert leaf["inference"]["counts"]["decoder_executions"] >= 1
    assert inference["rich_ir"]["ast"] == example["ast"]
    assert inference["whole_AST_generated_by_neural_encoder"] is False
    parts = inference["composition"]["source_parts"]
    assert "".join(row["text"] for row in parts) == instruction
    for row in parts:
        assert instruction[row["start_char"]:row["end_char"]] == row["text"]
    assert unit["atomic_family_projection"] is None and not unit["selected_native_targets"]
    assert inference["logic"]["formula"]["op"] == "implies"
    assert inference["logic"]["formula"]["left"]["op"] == "not"
    assert inference["logic"]["formula"]["right"]["modality"] == "prohibited"
    path, digest = write_advice(tmp_path, advice)
    loaded, replay = api.load_source_unit_advice(path=path, expected_sha256=digest, instruction=instruction)
    assert loaded == advice and replay["inference_replays"] == 1
    payload = json.loads(api.source_unit_planner_summary(loaded, maximum_bytes=8192).split("\n", 1)[1])
    candidate = payload["candidates"][0]
    assert candidate["rich_ir"]["ast"] == example["ast"]
    assert candidate["decoding_method"] == "grammar_composition_with_neural_leaves"
    assert candidate["symbolic_structure"]["guard_origin"] == "explicit_source_grammar"
    assert not candidate["symbolic_structure"]["neural_structure_prediction_claimed"]
