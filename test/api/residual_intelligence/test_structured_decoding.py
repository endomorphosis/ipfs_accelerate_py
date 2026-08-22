from __future__ import annotations

import json

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualTaskFamily,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.structured_decoding import (
    DEFAULT_GRAMMARS,
    DecodeStatus,
    PayloadFieldContract,
    decode_structured_output,
    grammar_for,
)


def encoded(**overrides: object) -> str:
    payload: dict[str, object] = {
        "output_class": "FAILURE_ATTRIBUTION",
        "structured_payload": {
            "failure_class": "missing_dependency_edge",
            "recommended_action": "expand_context_reference",
            "reference_ids": ["dependency:1"],
        },
        "confidence_or_score": 900_000,
        "calibration_group": "failure:python:R2:fixture",
        "abstained": False,
        "reason_codes": [],
        "evidence_references": ["validator:1"],
        "candidate_only": True,
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_all_closed_task_families_have_a_grammar() -> None:
    assert set(DEFAULT_GRAMMARS) == set(ResidualTaskFamily)
    for grammar in DEFAULT_GRAMMARS.values():
        assert set(grammar.field_contracts) == set(grammar.payload_fields)
        assert all(
            isinstance(contract, PayloadFieldContract)
            for contract in grammar.field_contracts.values()
        )


def test_structured_decode_and_compact_output() -> None:
    result = decode_structured_output(
        encoded(), grammar_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    )
    assert result.status is DecodeStatus.VALID
    assert result.output is not None
    assert result.output.candidate_only is True


def test_parse_failure_is_invalid_output_without_best_effort() -> None:
    result = decode_structured_output(
        "failure class: timeout", grammar_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    )
    assert result.status is DecodeStatus.INVALID_OUTPUT
    assert result.output is None
    assert result.reason_codes == ("invalid_output",)


def test_unknown_outer_and_payload_fields_rejected() -> None:
    grammar = grammar_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    outer = decode_structured_output(encoded(explanation="long prose"), grammar)
    assert outer.status is DecodeStatus.INVALID_OUTPUT
    body = json.loads(encoded())
    body["structured_payload"]["arbitrary_shell"] = "rm -rf elsewhere"
    nested = decode_structured_output(json.dumps(body), grammar)
    assert nested.status is DecodeStatus.INVALID_OUTPUT


def test_model_created_completion_is_invalid_output() -> None:
    body = json.loads(encoded())
    body["structured_payload"] = {
        "failure_class": "missing_dependency_edge",
        "recommended_action": "expand_context_reference",
        "completed": True,
    }
    result = decode_structured_output(
        json.dumps(body), grammar_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    )
    assert result.status is DecodeStatus.INVALID_OUTPUT


def test_abstention_is_explicit_and_bounded() -> None:
    result = decode_structured_output(
        encoded(
            output_class="ABSTAIN",
            structured_payload={},
            confidence_or_score=0,
            abstained=True,
            reason_codes=["out_of_distribution"],
            evidence_references=[],
        ),
        grammar_for(ResidualTaskFamily.FAILURE_ATTRIBUTION),
    )
    assert result.status is DecodeStatus.VALID
    assert result.output is not None and result.output.abstained is True


def test_context_sufficiency_rejects_untyped_values_and_abstention_payloads() -> None:
    grammar = grammar_for(ResidualTaskFamily.CONTEXT_SUFFICIENCY)
    wrong_scalar = decode_structured_output(
        encoded(
            output_class="CONTEXT_SUFFICIENCY",
            structured_payload={
                "sufficient": "definitely",
                "missing_reference_ids": 123,
            },
        ),
        grammar,
    )
    assert wrong_scalar.status is DecodeStatus.INVALID_OUTPUT

    arbitrary_abstention = decode_structured_output(
        encoded(
            output_class="ABSTAIN",
            structured_payload={"sufficient": {"arbitrary": "payload"}},
            confidence_or_score=0,
            abstained=True,
            reason_codes=["out_of_distribution"],
            evidence_references=[],
        ),
        grammar,
    )
    assert arbitrary_abstention.status is DecodeStatus.INVALID_OUTPUT


def test_rankings_require_bounded_aligned_descending_vectors() -> None:
    grammar = grammar_for(ResidualTaskFamily.EVIDENCE_RANKING)

    def ranking_payload(reference_ids: object, scores: object) -> str:
        return encoded(
            output_class="EVIDENCE_RANKING",
            structured_payload={
                "ranked_reference_ids": reference_ids,
                "scores_ppm": scores,
            },
        )

    valid = decode_structured_output(
        ranking_payload(["evidence:1", "evidence:2"], [900_000, 700_000]),
        grammar,
    )
    assert valid.status is DecodeStatus.VALID
    for references, scores in (
        (["evidence:1"], [900_000, 700_000]),
        (["evidence:1", "evidence:2"], [700_000, 900_000]),
        (["evidence:1"], [1_000_001]),
        (["evidence:1", 2], [900_000, 800_000]),
        (["evidence:1", "evidence:1"], [900_000, 800_000]),
    ):
        result = decode_structured_output(ranking_payload(references, scores), grammar)
        assert result.status is DecodeStatus.INVALID_OUTPUT


def test_patch_sketch_rejects_unbounded_or_untyped_scope() -> None:
    grammar = grammar_for(ResidualTaskFamily.PATCH_SKETCH_GENERATION)

    def patch_payload(**overrides: object) -> str:
        payload: dict[str, object] = {
            "files": ["ipfs_accelerate_py/module.py"],
            "symbol_ids": ["module.symbol"],
            "operations": ["replace_function"],
            "maximum_changed_lines": 200,
            "validation_ids": ["pytest:focused"],
        }
        payload.update(overrides)
        return encoded(output_class="PATCH_SKETCH", structured_payload=payload)

    assert decode_structured_output(patch_payload(), grammar).status is DecodeStatus.VALID
    for mutant in (
        {"maximum_changed_lines": 0},
        {"maximum_changed_lines": 10_001},
        {"maximum_changed_lines": "200"},
        {"files": ["../outside.py"]},
        {"files": ["/absolute.py"]},
        {"operations": [{"arbitrary": "shell"}]},
        {"symbol_ids": []},
    ):
        result = decode_structured_output(patch_payload(**mutant), grammar)
        assert result.status is DecodeStatus.INVALID_OUTPUT


def test_outer_sequence_and_calibration_types_are_not_coerced() -> None:
    grammar = grammar_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    assert (
        decode_structured_output(encoded(calibration_group=123), grammar).status
        is DecodeStatus.INVALID_OUTPUT
    )
    assert (
        decode_structured_output(encoded(reason_codes="not-an-array"), grammar).status
        is DecodeStatus.INVALID_OUTPUT
    )


def test_duplicate_json_field_is_invalid_output() -> None:
    raw = encoded()[:-1] + ',"candidate_only":true}'
    result = decode_structured_output(raw, grammar_for(ResidualTaskFamily.FAILURE_ATTRIBUTION))
    assert result.status is DecodeStatus.INVALID_OUTPUT
