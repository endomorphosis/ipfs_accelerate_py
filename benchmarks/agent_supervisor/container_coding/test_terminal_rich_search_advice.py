"""Real training-03 constrained recovery reaches the supervisor with replay.

This fixed development source exercises transport and provenance, not accuracy.
"""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_source_unit_advice as api

INSTRUCTION = "Check availability across date range"
EXPECTED_AST = {"kind": "atom", "actor": "unspecified", "action": "check",
                "object": "availability across date range", "modality": "intended"}


@pytest.fixture(scope="module")
def published_search_child():
    default = Path(__file__).resolve().parents[5] / "artifacts/intent-rich-decoder-20261002/training-03/descriptor.json"
    selected = Path(os.environ.get("INTENT_RICH_SEARCH_TEST_CHECKPOINT", str(default)))
    if not selected.exists():
        pytest.skip("training-03 search child is not installed; set INTENT_RICH_SEARCH_TEST_CHECKPOINT")
    from ipfs_datasets_py.logic.intent_ir.formalize.rich_decoder import load_rich_intent_checkpoint
    checkpoint = json.loads(selected.read_text())
    load_rich_intent_checkpoint(checkpoint)
    return checkpoint


@pytest.fixture(scope="module")
def actual_search_advice(tmp_path_factory, published_search_child):
    directory = tmp_path_factory.mktemp("actual-rich-search-advice")
    descriptor = directory / "descriptor.json"
    descriptor.write_text(json.dumps(published_search_child, sort_keys=True))
    advice = api.prepare_source_unit_advice(instruction=INSTRUCTION, enabled=True,
        intent_descriptor_path=descriptor, project_logic_families=True,
        requested_intent_families=["higher_order", "dcec"])
    assert advice["status"] == "source_unit_candidate_advice", advice
    return advice


def _write(tmp_path, advice):
    path = tmp_path / "advice.json"
    raw = api._wire(advice)
    path.write_bytes(raw)
    return path, api._sha(raw)


def _summary(advice):
    return json.loads(api.source_unit_planner_summary(advice, maximum_bytes=8192).split("\n", 1)[1])


def _rehash(value, key):
    value[key] = api._sha(api._wire({name: item for name, item in value.items() if name != key}))


def test_actual_recovery_sidecar_replays_with_policy_and_method_in_summary(tmp_path, actual_search_advice):
    from ipfs_datasets_py.logic.intent_ir.formalize.rich_search_policy import policy_identity
    advice = actual_search_advice
    path, digest = _write(tmp_path, advice)
    loaded, replay = api.load_source_unit_advice(path=path, expected_sha256=digest, instruction=INSTRUCTION)
    assert loaded == advice and replay["inference_replays"] == 1
    payload = _summary(loaded)
    candidate = payload["candidates"][0]
    assert candidate["rich_ir"]["ast"] == EXPECTED_AST
    assert candidate["decoding_method"] == "grammar_constrained_neural_roundtrip"
    assert candidate["syntax_constraint_policy"] == policy_identity()
    assert candidate["logic_families"] == ["dcec", "higher_order"]
    assert payload["counts"]["intent_candidates"] == 1
    assert not payload["execution_authority"]
    assert loaded["raw_instruction_preserved"] and loaded["continue_planning"] and loaded["before_goal_decomposition"]
    assert advice["report"]["counts"]["lake_attempts"] == 0


def test_rehashed_summary_method_tampering_is_refused_by_actual_replay(tmp_path, actual_search_advice):
    forged = deepcopy(actual_search_advice)
    rich = forged["report"]["rich_intent"]
    unit = next(unit for unit in rich["units"] if unit["accepted"])
    unit["decoding_method"] = "direct_neural_roundtrip"
    assert _summary(forged)["candidates"][0]["decoding_method"] == "direct_neural_roundtrip"
    _rehash(rich, "report_sha256")
    _rehash(forged["report"], "report_sha256")
    _rehash(forged, "advice_sha256")
    path, digest = _write(tmp_path, forged)
    loaded, replay = api.load_source_unit_advice(path=path, expected_sha256=digest, instruction=INSTRUCTION)
    assert loaded["status"] == "fail_open_invalid_sidecar"
    assert replay["inference_replays"] == 1
    assert loaded["continue_planning"] and loaded["raw_instruction_preserved"]
    assert api.source_unit_planner_summary(loaded, maximum_bytes=8192) is None
