"""Source-profile refusals and bounded captured-helper differential controls."""
from copy import deepcopy
import ast
import hashlib
import itertools
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_selector_semantics as api
from test.api.test_terminal_codebase_requirement_grounding import (
    public_inputs, _arguments, _build)


@pytest.fixture(scope="module")
def selector_arguments(public_inputs):
    grounding = _build(public_inputs)
    return {"grounding_receipt": grounding,
        "grounding_arguments": _arguments(public_inputs),
        "expected_grounding_sha256": grounding["grounding_sha256"],
        "current_source_records": [deepcopy(grounding["source_context"]["source_record"])]}


def test_exact_source_span_and_inert_authorities(selector_arguments):
    result = api.build_terminal_batching_selector_semantics(**selector_arguments)
    assert result["source_binding"]["start_line"] == 79
    assert result["source_binding"]["end_line"] == 83
    assert result["closed_profile_ast_lowering_verified"] is True
    assert result["conditional_source_model_proof_status"] == "not_run"
    assert result["source_runtime_equivalence_verified"] is False
    assert result["whole_program_proved"] is False
    assert result["generic_program_profile_widened"] is False
    assert result["full_task_satisfaction"] == "unknown"
    assert result["model_convergence"] == "unproved"
    assert result["training_calls"] == result["checker_calls"] == 0
    assert result["planning_handoff"] == "abstained"
    assert result["lean_source_sha256"] == hashlib.sha256(result["lean_source"].encode()).hexdigest()
    assert api.validate_terminal_batching_selector_semantics(result, **selector_arguments) == result


@pytest.mark.parametrize("change", ["comparator", "comment", "pin", "duplicate", "missing"])
def test_changed_present_source_refused(selector_arguments, change):
    arguments = deepcopy(selector_arguments)
    records = arguments["current_source_records"]
    if change in {"comparator", "comment"}:
        source = records[0]
        source["source_text"] = (source["source_text"].replace("rep >= s_val", "rep > s_val")
            if change == "comparator" else source["source_text"] + "\n# changed current checkout\n")
        source["source_sha256"] = hashlib.sha256(source["source_text"].encode()).hexdigest()
    elif change == "pin":
        records[0]["source_sha256"] = "0" * 64
    elif change == "duplicate":
        records.append(deepcopy(records[0]))
    else:
        records.clear()
    with pytest.raises(api.SelectorSemanticsError):
        api.build_terminal_batching_selector_semantics(**arguments)


def test_foreign_grounding_pin_refused(selector_arguments):
    arguments = deepcopy(selector_arguments)
    arguments["expected_grounding_sha256"] = "0" * 64
    with pytest.raises(api.SelectorSemanticsError):
        api.build_terminal_batching_selector_semantics(**arguments)


def test_resealed_foreign_lowering_refused(selector_arguments):
    result = api.build_terminal_batching_selector_semantics(**selector_arguments)
    result["ir"]["comparison"] = "greater_than"
    del result["selector_sha256"]
    result["selector_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    with pytest.raises(api.SelectorSemanticsError):
        api.validate_terminal_batching_selector_semantics(result, **selector_arguments)


def test_captured_ast_helper_matches_ir_on_1705_cases(selector_arguments):
    text = selector_arguments["current_source_records"][0]["source_text"]
    parent = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
        and n.name == "_plan_for_requests")
    helper = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == "assign_rep")
    # Only the independently pinned closed helper is compiled. No baseline
    # imports, I/O, callers, cost-model functions or benchmark oracle executes.
    code = compile(ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])),
        "<pinned-public-assign-rep>", "exec")
    checked = 0
    for length in range(5):
        for values in itertools.product((-64, 0, 64, 128), repeat=length):
            representatives = list(values)
            namespace = {"__builtins__": {}, "int": int, "reps": representatives}
            exec(code, namespace)
            for sequence in (-65, -64, 0, 65, 129):
                if not values:
                    with pytest.raises(IndexError): namespace["assign_rep"](sequence)
                    with pytest.raises(IndexError): api.evaluate_selector_model(representatives, sequence)
                else:
                    assert namespace["assign_rep"](sequence) == api.evaluate_selector_model(representatives, sequence)
                checked += 1
    assert checked == 1705


def test_order_fallback_and_unbounded_exact_integer_domain():
    assert api.evaluate_selector_model([128, 64], 1) == 128  # first, not sorted minimum
    assert api.evaluate_selector_model([64, 128], 129) == 128  # fallback does not cover
    assert api.evaluate_selector_model([64, 64, 128], 64) == 64
    large = 10 ** 1000
    assert api.evaluate_selector_model([-large, large], large - 1) == large
    with pytest.raises(IndexError): api.evaluate_selector_model([], 0)


@pytest.mark.parametrize("representatives,sequence", [([True], 0), ([1], False),
    ((1, 2), 1), ([1.0], 1), ([1], 1.0), (list(range(10001)), 0)])
def test_outside_closed_domain_refused(representatives, sequence):
    with pytest.raises(api.SelectorSemanticsError):
        api.evaluate_selector_model(representatives, sequence)
