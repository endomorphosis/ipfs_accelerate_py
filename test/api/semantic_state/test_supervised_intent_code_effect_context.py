"""Optional cross-source advice preserves the native task and original advisors."""
from copy import deepcopy

import pytest

from test.api.semantic_state.test_supervised_task_context import inputs, runtime
from ipfs_accelerate_py.agent_supervisor.runtime import intent_code_effect_advisor as effects
from ipfs_accelerate_py.agent_supervisor.runtime import security_source_program_advisor_384 as security


def test_absent_cross_contract_config_keeps_the_existing_task_context(inputs, monkeypatch):
    monkeypatch.setattr(effects, "prepare_repository_intent_code_effect_advice",
        lambda **kwargs: pytest.fail("unselected optional contract consumer ran"))
    prepared = runtime.prepare_supervised_task_context(**inputs)
    assert "intent_code_effect_advice" not in prepared


def test_explicit_instruction_and_advice_are_forwarded_without_using_task_title(inputs, monkeypatch):
    source_advice = {"status": "source_candidate_advice", "candidate": "unchanged"}
    intent_advice = {"instruction": "independent decoded instruction"}
    instruction = "The operator must preserve the returned sum."
    selected = {"explicit": "effect association"}
    calls = []
    monkeypatch.setattr(security, "prepare_repository_source_program_advice", lambda **kwargs: source_advice)
    def prepare(**options):
        calls.append(deepcopy(options))
        return {"status": "contract_interpretation_advice", "bounded_effects_satisfied": False,
                "continue_planning": True, "proof_authority": False}
    monkeypatch.setattr(effects, "prepare_repository_intent_code_effect_advice", prepare)
    before = inputs["intent"].plan_projection()
    result = runtime.prepare_supervised_task_context(**inputs, security_source_program_config={"explicit": "security"},
        intent_code_effect_instruction=instruction, intent_code_effect_intent_advice=intent_advice,
        intent_code_effect_config=selected)
    assert calls == [dict(repository=inputs["repository"].resolve(), paths=inputs["paths"], instruction=instruction,
        intent_advice=intent_advice, security_advice=source_advice, config=selected)]
    assert result["task_title"] != instruction
    assert result["security_source_program_advice"] is source_advice
    assert result["intent_code_effect_advice"]["continue_planning"]
    assert inputs["intent"].plan_projection() == before and not result["canonical_task_mutated"]
    assert all("effect" not in key.lower() for key in result["metadata"])


def test_missing_optional_contract_dependency_preserves_security_advice(inputs, monkeypatch):
    source_advice = {"status": "source_candidate_advice", "candidate": "unchanged"}
    monkeypatch.setattr(security, "prepare_repository_source_program_advice", lambda **kwargs: source_advice)
    def absent(**options): raise ImportError("private machine information")
    monkeypatch.setattr(effects, "prepare_repository_intent_code_effect_advice", absent)
    result = runtime.prepare_supervised_task_context(**inputs, security_source_program_config={"explicit": "security"},
        intent_code_effect_config={"explicit": "effect association"})
    assert result["security_source_program_advice"] is source_advice
    assert result["intent_code_effect_advice"]["status"] == "fail_open_unavailable"
    assert "private machine information" not in str(result["intent_code_effect_advice"])


@pytest.mark.parametrize("changed_owner", ["source", "intent"])
def test_source_and_native_task_drift_still_reject_context_after_optional_contract(inputs, monkeypatch, changed_owner):
    def changed(**options):
        if changed_owner == "source":
            (inputs["repository"] / "dependency.py").write_text("def add(a,b): return a-b\n")
        else:
            inputs["intent"].upsert_task(task_cid="task:add", task_alias="CONTEXT-001", goal_cid="goal:add",
                plan_cid="plan:add", objective_id="objective:add", body={"title": "Changed task"})
        return {"status": "fail_open_unavailable", "continue_planning": True}
    monkeypatch.setattr(effects, "prepare_repository_intent_code_effect_advice", changed)
    with pytest.raises(ValueError):
        runtime.prepare_supervised_task_context(**inputs, intent_code_effect_config={"explicit": "effect association"})
    assert not (inputs["output"] / "result.json").exists()
