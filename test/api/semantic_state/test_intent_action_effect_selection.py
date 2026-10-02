"""Generated associations use declared Intent semantics, never code outcomes.

These fixtures are authored typed candidates. Numerical inference is tested
separately with the published checkpoints; no fixture is a model prediction.
"""
from copy import deepcopy
import hashlib
import os

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import intent_code_effect_advisor as subject


def inputs(operator="+", precondition="left > 0", mapping=None):
    from ipfs_datasets_py.logic.intent_ir.formalize import action_contracts as codec
    from ipfs_datasets_py.logic.software_verification.program import ProgramExpression
    instruction = ("the agent must compute result; requires " + precondition
        + "; ensures result = old(left) " + operator + " old(right) and returned.")
    candidate = codec.bind_candidate_source(instruction, codec.source_to_target(instruction))["bound_candidate"]
    source = "def compute(capacity: int, threshold: int) -> int:\n    return capacity + threshold\n"
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    operands = ("expr:capacity", "expr:threshold")
    code = dict(kind="program_expression", document=ProgramExpression("expr:result", "binary", "integer",
        operand_ids=operands, evaluation_order=operands, operator="+", source_ref_ids=("source",)).to_dict())
    security_pin = "a" * 64
    return candidate, dict(instruction=instruction, intent_advice={"advice_sha256": "b" * 64},
        security_advice=dict(schema="supervisor-security-source-program-384-advice/v2",
            checkpoint_selection=dict(checkpoint_sha256=security_pin),
            inference=dict(domain_id="security_ir", checkpoint_sha256=security_pin,
                rows=[dict(id="input-0", source_sha256=source_hash, candidate_ir=code)]),
            source_hashes={"code.py": source_hash},
            input_bindings=[dict(source_id="code.py", inference_id="input-0", source_sha256=source_hash)],
            proof_authority=False, execution_authority=False, completion_authority=False, source_semantics_verified=False),
        source_rows=[dict(id="code.py", source_text=source, source_sha256=source_hash)],
        config=dict(schema=subject.ACTION_CONFIG_SCHEMA, lake=None, contracts=[dict(
            id="return-contract", source_id="code.py", action_id="action",
            input_parameter_mapping=mapping or {"left": "capacity", "right": "threshold"},
            input_domains={name: dict(lower=-1, upper=1) for name in ("capacity", "threshold")})]))


def authored_transport(monkeypatch, candidate):
    monkeypatch.setattr(subject, "_intent", lambda *args: (deepcopy(candidate), "c" * 64))


@pytest.mark.parametrize("operator,precondition,status", [
    ("+", "left > 0", "satisfied"), ("-", "left > 0", "refuted"),
    ("*", "left > 2", "no_enabled_cases")])
def test_generated_formulas_keep_positive_negative_and_disabled_cases(monkeypatch, operator, precondition, status):
    candidate, values = inputs(operator, precondition)
    authored_transport(monkeypatch, candidate)
    before = deepcopy(values)
    report = subject.prepare_intent_code_effect_advice(**values)
    assert report["native"] is not None, report
    assert report["association_replay_verified"] and values == before
    assert report["rows"][0]["effect_status"] == status
    assert not report["live_build_verified"] and not report["selected_bounded_effects_satisfied"]
    assert report["action_selections"] == values["config"]["contracts"]
    assert report["configuration_sha256"] == subject._sha(subject._wire(values["config"]))
    assert all(report[key] is False for key in subject.FALSE)


@pytest.mark.parametrize("change", ["aliased", "missing", "foreign", "action", "authored_formula", "source",
    "prediction", "instruction", "missing_effect", "no_prediction"])
def test_bad_selection_or_candidate_fails_open_without_manufacturing_effects(monkeypatch, change):
    candidate, values = inputs()
    selected = values["config"]["contracts"][0]
    if change == "aliased": selected["input_parameter_mapping"]["right"] = "capacity"
    elif change == "missing": selected["input_parameter_mapping"].pop("right")
    elif change == "foreign": selected["input_parameter_mapping"]["right"] = "absent"
    elif change == "action": selected["action_id"] = "unknown"
    elif change == "authored_formula": selected["association"] = {}
    elif change == "source": values["source_rows"][0]["source_text"] += "\n"
    elif change == "prediction": values["security_advice"]["inference"]["rows"][0]["candidate_ir"]["document"]["operator"] = "-"
    elif change == "instruction": values["instruction"] += " Also erase the file."
    elif change == "missing_effect": candidate["document"]["actions"][0]["effect_ids"].pop()
    else: values["security_advice"]["inference"]["rows"][0]["candidate_ir"] = None
    authored_transport(monkeypatch, candidate)
    original = deepcopy(values)
    report = subject.prepare_intent_code_effect_advice(**values)
    assert report["status"] == "fail_open_unavailable" and report["continue_planning"]
    assert report["native"] is None and not report["live_build_verified"] and values == original


def test_generated_formula_mutation_is_rejected_by_independent_rebuild(monkeypatch):
    from ipfs_datasets_py.logic.formalization.autoencoder import intent_action_association as builder
    candidate, values = inputs()
    authored_transport(monkeypatch, candidate)
    build = builder.build_intent_action_association
    calls = []
    def altered(*args, **kwargs):
        result = build(*args, **kwargs)
        # Corrupt only the initial build; the independent verifier rebuilds.
        if not calls:
            result["effect_bindings"][0]["expression_id"] = "contract:returned"
        calls.append(True)
        return result
    monkeypatch.setattr(builder, "build_intent_action_association", altered)
    report = subject.prepare_intent_code_effect_advice(**values)
    assert len(calls) == 2 and report["failure_stage"] == "datasets_action_association"
    assert report["native"] is None


@pytest.mark.parametrize("operator,precondition,status", [
    ("+", "left > 0", "satisfied"), ("-", "left > 0", "refuted"),
    ("*", "left > 2", "no_enabled_cases")])
def test_generated_selection_runs_live_lake(monkeypatch, operator, precondition, status):
    lake = os.environ.get("IR384_TEST_LAKE_EXECUTABLE")
    if not lake:
        pytest.skip("explicit installed Lake required; no download")
    candidate, values = inputs(operator, precondition)
    authored_transport(monkeypatch, candidate)
    values["config"]["lake"] = dict(executable=lake, timeout_seconds=60)
    report = subject.prepare_intent_code_effect_advice(**values)
    assert report["native"] is not None, report
    assert report["live_build_verified"] and report["all_selected_contracts_checked"]
    assert report["rows"][0]["effect_status"] == status
    assert report["selected_bounded_effects_satisfied"] is (status == "satisfied")
    assert report["rows"][0]["counterexample_kernel_checked"] is (status == "refuted")
