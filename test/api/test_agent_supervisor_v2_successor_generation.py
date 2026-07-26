from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
    SELF_IMPROVEMENT_SUCCESSOR_RECORD_SCHEMA,
    SELF_IMPROVEMENT_SUCCESSOR_RECORDS_KEY,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement_v2 import (
    ACTIONABLE_V2_RESIDUAL_KINDS,
    V2ResidualKind,
    V2ResidualSignal,
    V2SuccessorGenerationPolicy,
    V2SuccessorRejectionReason,
    generate_v2_successor_goals,
)


NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def _residual(
    residual_id: str = "residual:cache-regression",
    *,
    kind: V2ResidualKind | str = V2ResidualKind.REGRESSION,
    **overrides: object,
) -> V2ResidualSignal:
    slug = residual_id.rsplit(":", 1)[-1]
    payload: dict[str, object] = {
        "residual_id": residual_id,
        "kind": kind,
        "title": f"Repair measured {slug} residual",
        "detail": (
            f"The {slug} producer receipt measures a remaining failure. "
            "Reduce the measured residual without weakening correctness."
        ),
        "acceptance_criteria": (
            f"The {slug} benchmark meets its declared threshold",
            f"A focused regression test reproduces and closes {slug}",
        ),
        "evidence_ids": (f"receipt:{slug}@1",),
        "predicted_files": (
            f"ipfs_accelerate_py/agent_supervisor/{slug.replace('-', '_')}.py",
            f"test/api/test_{slug.replace('-', '_')}.py",
        ),
        "predicted_symbols": (f"repair_{slug.replace('-', '_')}",),
        "validation_commands": (
            f"python -m pytest test/api/test_{slug.replace('-', '_')}.py -q",
        ),
        "confidence": 0.9,
        "estimated_tokens": 1_000,
        "depth": 1,
        "task_count": 1,
        "changed": True,
        "completed": False,
        "source_receipt_id": f"receipt:{slug}@1",
    }
    payload.update(overrides)
    return V2ResidualSignal(**payload)


def _reason_values(result: object) -> list[str]:
    return [
        item.reason.value
        for item in result.rejected  # type: ignore[attr-defined]
    ]


@pytest.mark.parametrize("kind", sorted(ACTIONABLE_V2_RESIDUAL_KINDS, key=str))
def test_only_each_typed_actionable_residual_kind_can_generate_a_goal(
    kind: V2ResidualKind,
) -> None:
    residual = _residual(f"residual:{kind.value}", kind=kind)

    result = generate_v2_successor_goals((residual,), observed_at=NOW)

    assert not result.rejected
    assert result.generated_goal_count == 1
    assert result.generated_task_count == 1
    assert len(result.accepted) == 1
    candidate = result.accepted[0]
    assert candidate.source_residual_id == residual.residual_id
    assert candidate.proposal.source_id == residual.residual_id
    assert candidate.proposal.expected_evidence_delta == residual.evidence_ids
    assert candidate.proposal.predicted_files == residual.predicted_files
    assert candidate.proposal.validation_commands == residual.validation_commands
    assert 0.0 <= candidate.semantic_novelty <= 1.0


@pytest.mark.parametrize(
    ("residual", "reason"),
    [
        (
            _residual(
                "residual:generic",
                kind=V2ResidualKind.GENERIC_IMPROVEMENT,
            ),
            "generic-improvement",
        ),
        (
            _residual(
                "residual:completed-kind",
                kind=V2ResidualKind.COMPLETED_EVIDENCE,
            ),
            "completed-evidence",
        ),
        (
            _residual(
                "residual:completed-flag",
                completed=True,
            ),
            "completed-evidence",
        ),
        (
            _residual(
                "residual:delivery",
                kind=V2ResidualKind.DELIVERY_NOISE,
            ),
            "delivery-noise",
        ),
        (
            _residual(
                "residual:unchanged-kind",
                kind=V2ResidualKind.UNCHANGED_RESIDUAL,
            ),
            "unchanged-residual",
        ),
        (
            _residual(
                "residual:unchanged-flag",
                changed=False,
            ),
            "unchanged-residual",
        ),
    ],
)
def test_generic_completed_delivery_and_unchanged_inputs_create_no_proposal(
    residual: V2ResidualSignal,
    reason: str,
) -> None:
    result = generate_v2_successor_goals((residual,), observed_at=NOW)

    assert result.accepted == ()
    assert result.generated_goal_count == 0
    assert result.generated_task_count == 0
    assert _reason_values(result) == [reason]


def test_malformed_or_untyped_input_is_rejected_without_becoming_work() -> None:
    result = generate_v2_successor_goals(
        (
            {},
            {
                **_residual("residual:unknown-kind").to_dict(),
                "kind": "free-form-observation",
            },
        ),
        observed_at=NOW,
    )

    assert result.accepted == ()
    assert result.generated_goal_count == 0
    assert set(_reason_values(result)) <= {
        "malformed-residual",
        "ineligible-residual-kind",
    }
    assert len(result.rejected) == 2


def test_goal_quality_lint_and_unsupported_dependencies_fail_closed() -> None:
    lint_failure = _residual(
        "residual:missing-contract",
        acceptance_criteria=(),
        evidence_ids=(),
        predicted_files=(),
        predicted_symbols=(),
        validation_commands=(),
    )
    unsupported = _residual(
        "residual:unsupported-dependency",
        dependencies=("capability:gpu-profiler",),
    )

    result = generate_v2_successor_goals(
        (lint_failure, unsupported),
        supported_dependencies=("capability:cpu-profiler",),
        observed_at=NOW,
    )

    assert result.accepted == ()
    assert sorted(_reason_values(result)) == sorted([
        "goal-quality-lint",
        "unsupported-dependency",
    ])
    details = {item.reason.value: item.detail for item in result.rejected}
    assert "missing_" in details["goal-quality-lint"]
    assert "capability:gpu-profiler" in details["unsupported-dependency"]


def test_confidence_and_semantic_novelty_are_independent_finite_gates() -> None:
    low_confidence = _residual(
        "residual:low-confidence",
        confidence=0.49,
    )
    familiar = _residual("residual:familiar")
    semantic_text = {
        "title": familiar.title,
        "expected_evidence_delta": familiar.evidence_ids,
        "predicted_files": familiar.predicted_files,
        "predicted_symbols": familiar.predicted_symbols,
        "acceptance_subset": familiar.acceptance_criteria,
    }

    result = generate_v2_successor_goals(
        (low_confidence, familiar),
        strategy={"semantic_texts": (semantic_text,)},
        policy=V2SuccessorGenerationPolicy(
            min_confidence=0.5,
            min_semantic_novelty=0.35,
        ),
        observed_at=NOW,
    )

    assert result.accepted == ()
    assert sorted(_reason_values(result)) == sorted([
        "low-confidence",
        "low-semantic-novelty",
    ])


def test_exact_identity_historical_identity_and_cooldown_are_distinct() -> None:
    residual = _residual()
    initial = generate_v2_successor_goals((residual,), observed_at=NOW)
    assert len(initial.accepted) == 1
    candidate = initial.accepted[0]

    duplicate = generate_v2_successor_goals(
        (residual,),
        existing_goals=(candidate.proposal,),
        observed_at=NOW,
    )
    historical = generate_v2_successor_goals(
        (residual,),
        strategy={"historical_identities": (candidate.canonical_identity,)},
        observed_at=NOW,
    )
    cooldown = generate_v2_successor_goals(
        (residual,),
        strategy={"cooldown_identities": (candidate.canonical_identity,)},
        observed_at=NOW,
    )

    assert _reason_values(duplicate) == ["duplicate-identity"]
    assert _reason_values(historical) == ["historical-identity"]
    assert _reason_values(cooldown) == ["cooldown-active"]
    assert not duplicate.accepted
    assert not historical.accepted
    assert not cooldown.accepted


def test_durable_lifecycle_admission_maps_to_typed_historical_rejection() -> None:
    residual = _residual("residual:durable-history")
    initial = generate_v2_successor_goals((residual,), observed_at=NOW)
    proposal = initial.accepted[0].proposal
    record = {
        "schema": SELF_IMPROVEMENT_SUCCESSOR_RECORD_SCHEMA,
        "version": 1,
        "canonical_id": proposal.canonical_id,
        "semantic_key": proposal.semantic_key,
        "status": "admitted",
        "epoch_id": "epoch:durable-history",
        "transaction_id": "transaction:durable-history",
        "recorded_at": NOW.isoformat(),
        "cooldown_until": "",
        "reason_codes": [],
        "attempts": [],
    }

    replay = generate_v2_successor_goals(
        (residual,),
        strategy={
            SELF_IMPROVEMENT_SUCCESSOR_RECORDS_KEY: {
                proposal.canonical_id: record,
            }
        },
        observed_at=NOW,
    )

    assert not replay.accepted
    assert _reason_values(replay) == ["historical-identity"]


def test_one_residual_has_one_goal_and_cannot_duplicate_its_tasks() -> None:
    residual = _residual("residual:bounded-fanout", task_count=3)

    result = generate_v2_successor_goals(
        (residual, residual),
        policy=V2SuccessorGenerationPolicy(
            max_breadth_per_residual=3,
            max_goals=2,
            max_tasks=3,
        ),
        observed_at=NOW,
    )

    assert result.generated_goal_count == 1
    assert result.generated_task_count == 3
    assert len(result.accepted) == 1
    assert len(result.accepted[0].task_ids) == 3
    assert len(set(result.accepted[0].task_ids)) == 3
    assert _reason_values(result) == ["duplicate-residual"]


@pytest.mark.parametrize(
    ("policy", "residuals", "current_open_work", "reason"),
    [
        (
            V2SuccessorGenerationPolicy(max_depth=0),
            (_residual("residual:depth"),),
            0,
            "depth-budget",
        ),
        (
            V2SuccessorGenerationPolicy(max_breadth_per_residual=1),
            (_residual("residual:breadth", task_count=2),),
            0,
            "breadth-budget",
        ),
        (
            V2SuccessorGenerationPolicy(max_open_work=2),
            (_residual("residual:open-work"),),
            2,
            "open-work-budget",
        ),
        (
            V2SuccessorGenerationPolicy(max_tokens=999),
            (_residual("residual:tokens", estimated_tokens=1_000),),
            0,
            "token-budget",
        ),
        (
            V2SuccessorGenerationPolicy(max_goals=1),
            (
                _residual("residual:goal-a"),
                _residual("residual:goal-b"),
            ),
            0,
            "goal-budget",
        ),
        (
            V2SuccessorGenerationPolicy(max_tasks=1),
            (_residual("residual:tasks", task_count=2),),
            0,
            "task-budget",
        ),
    ],
)
def test_each_finite_generation_budget_rejects_with_a_typed_reason(
    policy: V2SuccessorGenerationPolicy,
    residuals: tuple[V2ResidualSignal, ...],
    current_open_work: int,
    reason: str,
) -> None:
    result = generate_v2_successor_goals(
        residuals,
        policy=policy,
        current_open_work=current_open_work,
        observed_at=NOW,
    )

    assert reason in _reason_values(result)
    assert result.generated_goal_count <= policy.max_goals
    assert result.generated_task_count <= policy.max_tasks
    assert result.consumed_tokens <= policy.max_tokens
    assert current_open_work + result.generated_task_count <= policy.max_open_work


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_confidence", math.nan),
        ("min_semantic_novelty", math.inf),
        ("max_depth", True),
        ("max_breadth_per_residual", -1),
        ("max_open_work", -1),
        ("max_tokens", -1),
        ("max_goals", -1),
        ("max_tasks", -1),
        ("max_rejections", -1),
        ("max_residuals", -1),
        ("cooldown_seconds", -1),
    ],
)
def test_policy_rejects_nonfinite_boolean_and_negative_limits(
    field: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        V2SuccessorGenerationPolicy(**{field: value})


def test_input_and_rejection_overflow_are_bounded_and_typed() -> None:
    residuals = tuple(
        _residual(
            f"residual:noise-{index}",
            kind=V2ResidualKind.DELIVERY_NOISE,
        )
        for index in range(8)
    )

    result = generate_v2_successor_goals(
        residuals,
        policy=V2SuccessorGenerationPolicy(
            max_residuals=4,
            max_rejections=3,
        ),
        observed_at=NOW,
    )

    assert not result.accepted
    assert len(result.rejected) <= 3
    assert "input-budget" in _reason_values(result)
    assert result.rejection_overflow_count > 0
    assert all(
        isinstance(item.reason, V2SuccessorRejectionReason)
        for item in result.rejected
    )
    assert all(len(item.detail.encode("utf-8")) <= 512 for item in result.rejected)
    payload = result.to_dict()
    assert payload["rejection_overflow_count"] == result.rejection_overflow_count
    assert len(payload["rejected"]) <= 3
