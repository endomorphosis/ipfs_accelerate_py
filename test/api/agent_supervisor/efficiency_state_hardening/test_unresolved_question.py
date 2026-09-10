"""ASEH-021 closed unresolved-question admission contract."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import HarnessError
from ipfs_accelerate_py.agent_supervisor.semantic_state.unresolved_question import (
    MAX_CONTEXT_BUDGET,
    MAX_COST_BUDGET_MICROUSD,
    UNRESOLVED_QUESTION_FIELDS,
    UNRESOLVED_QUESTION_SCHEMA,
    UNRESOLVED_QUESTION_SCHEMA_PATH,
    UnresolvedQuestion,
    build_unresolved_question,
    canonical_question_bytes,
    load_unresolved_question_schema,
    round_trip_unresolved_question,
)


def _question(**overrides: object) -> UnresolvedQuestion:
    fields: dict[str, object] = {
        "exact_question": "Which bounded route should resolve the unresolved contract?",
        "why_prior_deterministic_stages_could_not_resolve": "Static analysis found two valid contract interpretations.",
        "evidence_available": ["current schema", "selected test receipt"],
        "evidence_missing": ["authoritative contract interpretation"],
        "candidate_decisions_answer_could_change": ["small_local_model", "medium_model"],
        "minimum_model_capability": "local_small_specialist_model",
        "context_budget": 4096,
        "response_schema": {"type": "string", "enum": ["narrow", "broad"]},
        "deadline": "2027-01-02T03:04:05Z",
        "cost_budget": 250000,
    }
    fields.update(overrides)
    return build_unresolved_question(**fields)


def test_schema_is_present_closed_and_matches_the_closed_record() -> None:
    schema = load_unresolved_question_schema()
    assert UNRESOLVED_QUESTION_SCHEMA_PATH.is_file()
    assert schema["$id"] == UNRESOLVED_QUESTION_SCHEMA
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == UNRESOLVED_QUESTION_FIELDS
    assert set(schema["properties"]) == UNRESOLVED_QUESTION_FIELDS
    assert schema["$defs"]["responseSchema"]["additionalProperties"] is False


def test_canonical_round_trip_derives_identity_and_canonicalizes_lists() -> None:
    question = _question(
        evidence_available=["selected test receipt", "current schema"],
        response_schema={"type": "string", "enum": ["broad", "narrow"]},
        candidate_decisions_answer_could_change=["medium_model", "small_local_model"],
    )
    payload = question.to_dict()
    restored = UnresolvedQuestion.from_dict(payload)
    assert question.question_id.startswith("sha256:")
    assert payload == restored.to_dict() == round_trip_unresolved_question(payload)
    assert question.canonical_bytes() == canonical_question_bytes(payload)
    assert json.loads(question.canonical_bytes()) == payload


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_budget", 0),
        ("context_budget", MAX_CONTEXT_BUDGET + 1),
        ("cost_budget", 0),
        ("cost_budget", MAX_COST_BUDGET_MICROUSD + 1),
        ("deadline", "2027-01-02T03:04:05+00:00"),
        ("deadline", "2101-01-02T03:04:05Z"),
        ("response_schema", {"type": "string", "enum": ["only"]}),
    ],
)
def test_unbounded_or_invalid_escalation_limits_fail_closed(field: str, value: object) -> None:
    with pytest.raises(HarnessError):
        _question(**{field: value})


def test_identity_must_bind_the_complete_canonical_question() -> None:
    payload = _question().to_dict()
    payload["exact_question"] = "Could the route be changed?"
    with pytest.raises(HarnessError, match="question_id does not match"):
        UnresolvedQuestion.from_dict(payload)


def test_unknown_fields_and_overlapping_evidence_fail_closed() -> None:
    payload = _question().to_dict()
    payload["dispatch_now"] = True
    with pytest.raises(HarnessError, match="fields must be exactly"):
        UnresolvedQuestion.from_dict(payload)
    with pytest.raises(HarnessError, match="must not overlap"):
        _question(evidence_missing=["current schema"])


@pytest.mark.parametrize(
    "impacts",
    [[], ["small_local_model"], ["small_local_model", "small_local_model"]],
)
def test_questions_without_an_admissible_decision_change_are_rejected(
    impacts: list[str],
) -> None:
    with pytest.raises(HarnessError, match="admissible decisions|duplicates"):
        _question(candidate_decisions_answer_could_change=impacts)


def test_question_builder_rejects_caller_selected_identity() -> None:
    with pytest.raises(HarnessError, match="provided question_id"):
        _question(question_id="sha256:" + "f" * 64)
