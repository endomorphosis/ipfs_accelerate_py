from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_policy import (
    ProofResultStatus,
    RolloutMode,
)
from ipfs_accelerate_py.agent_supervisor.proof_fallbacks import (
    DEFAULT_MAX_DIAGNOSTIC_BYTES,
    ProofFailureKind,
    ProofFallbackDeduplicator,
    ProofFallbackDiagnostic,
    ProofFallbackRouter,
    ProofFallbackValidationError,
    ProofRegressionFixture,
    RegressionExpectation,
    normalize_counterexample,
    normalize_unsat_core,
    route_proof_fallback,
    route_proof_fallbacks,
)
from ipfs_accelerate_py.agent_supervisor.validation_commands import (
    ValidationRequirementKind,
    ValidationStage,
    build_declared_validations,
    build_focused_validation_commands,
)


def _obligation(
    *,
    tree: str = "git-tree:candidate-one",
    checks: tuple[str, ...] = (
        "pytest:lease-stale-token",
        "static:lease-types",
        "manual:review-locking-boundary",
    ),
) -> CodeProofObligation:
    return CodeProofObligation(
        repository_id="repo:supervisor",
        repository_tree_id=tree,
        ast_scope_ids=("scope:lease.acquire",),
        statement="Every lease mutation presents the current fencing token.",
        premise_ids=("premise:monotonic-fencing-token",),
        template_id="lease-uniqueness-and-fencing",
        template_version="1.0.0",
        template_semantic_hash="sha256:" + "a" * 64,
        invariant_class="lease_safety",
        task_id="REF-256",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        fallback_checks=checks,
    )


def _disproved_result(counterexample: object) -> dict[str, object]:
    return {
        "verdict": "disproved",
        "attempts": [
            {
                "effective_outcome": "counterexample",
                "conclusive": True,
                "evidence": {"counterexample": counterexample},
            }
        ],
    }


def test_counterexample_becomes_bounded_redacted_diagnostic_and_fixture() -> None:
    counterexample = {
        "active_leases": [
            {"resource": "r1", "token": number, "padding": "x" * 4000}
            for number in range(80)
        ],
        "access_token": "must-not-enter-task-context",
        "observed_token": 3.5,
    }

    plan = route_proof_fallback(
        _obligation(),
        _disproved_result(counterexample),
        rollout_mode=RolloutMode.SHADOW,
    )

    assert plan.proof_status is ProofResultStatus.DISPROVED
    assert len(plan.diagnostics) == 1
    diagnostic = plan.diagnostics[0]
    assert diagnostic.kind is ProofFailureKind.COUNTEREXAMPLE
    assert diagnostic.truncated is True
    assert diagnostic.redacted is True
    assert "must-not-enter-task-context" not in diagnostic.to_json()
    assert len(diagnostic.to_json().encode("utf-8")) <= DEFAULT_MAX_DIAGNOSTIC_BYTES
    assert len(plan.regression_fixtures) == 1
    fixture = plan.regression_fixtures[0]
    assert fixture.counterexample_id == diagnostic.counterexample_id
    assert fixture.expected is RegressionExpectation.REJECT
    assert ProofFallbackDiagnostic.from_dict(diagnostic.to_dict()) == diagnostic
    assert ProofRegressionFixture.from_dict(fixture.to_dict()) == fixture


def test_counterexample_and_unsat_core_normalization_are_canonical() -> None:
    first = normalize_counterexample({"b": 2, "a": [1, 2]})
    second = normalize_counterexample({"a": [1, 2], "b": 2})
    assert first[0] == second[0]
    assert first[1] == second[1]

    core_one = normalize_unsat_core(["premise:z", "premise:a"])
    core_two = normalize_unsat_core(["premise:a", "premise:z"])
    assert core_one[0] == ["premise:a", "premise:z"]
    assert core_one[1] == core_two[1]

    plan = route_proof_fallback(
        _obligation(),
        {"status": "inconclusive", "diagnostics": {"unsat_core": core_one[0]}},
    )
    assert plan.diagnostics[0].kind is ProofFailureKind.UNSAT_CORE
    assert plan.regression_fixtures[0].expected is RegressionExpectation.UNSATISFIABLE


def test_unsupported_obligation_uses_only_declared_validation_routes() -> None:
    catalog = {
        "pytest:lease-stale-token": (
            "python -m pytest test/api/test_agent_supervisor_lease_coordination.py "
            "-q"
        ),
        "static:lease-types": (
            "python -m py_compile "
            "ipfs_accelerate_py/agent_supervisor/lease_coordination.py"
        ),
        # Explicit manual review stays manual even if a catalog is malicious.
        "manual:review-locking-boundary": "sh -c 'exit 0'",
    }
    plan = route_proof_fallback(
        _obligation(),
        {"status": "unsupported", "reason": "translator has no exact semantics"},
        command_catalog=catalog,
    )

    by_id = {item.validation_id: item for item in plan.validations}
    assert by_id["pytest:lease-stale-token"].kind is ValidationRequirementKind.FOCUSED_TEST
    assert by_id["pytest:lease-stale-token"].command.stage is ValidationStage.TARGETED
    assert by_id["static:lease-types"].kind is ValidationRequirementKind.STATIC_CHECK
    assert by_id["static:lease-types"].command.stage is ValidationStage.CHEAP
    assert by_id["manual:review-locking-boundary"].manual_review_required
    assert by_id["manual:review-locking-boundary"].command is None
    assert len(plan.executable_commands) == 2
    assert plan.manual_review_requirements == (
        by_id["manual:review-locking-boundary"],
    )


def test_unresolved_and_unknown_declarations_never_become_shell_commands() -> None:
    declarations = build_declared_validations(
        [
            "pytest:named-but-not-registered",
            "provider-output --pretend-safe",
            "ruff check src",
            "pytest tests/test_safe.py",
        ]
    )

    assert declarations[0].kind is ValidationRequirementKind.FOCUSED_TEST
    assert declarations[0].command is None
    assert declarations[1].kind is ValidationRequirementKind.MANUAL_REVIEW
    assert declarations[1].command is None
    assert declarations[2].kind is ValidationRequirementKind.STATIC_CHECK
    assert declarations[2].command is not None
    assert declarations[3].kind is ValidationRequirementKind.FOCUSED_TEST
    assert [item.command for item in build_focused_validation_commands(declarations)] == [
        "ruff check src",
        "pytest tests/test_safe.py",
    ]


def test_shadow_continues_but_enforcement_preserves_required_assurance() -> None:
    result = {"status": "unsupported", "reason_code": "no_supported_backend"}
    shadow = route_proof_fallback(
        _obligation(), result, rollout_mode=RolloutMode.SHADOW
    )
    enforcement = route_proof_fallback(
        _obligation(), result, rollout_mode=RolloutMode.ENFORCEMENT
    )
    canary = route_proof_fallback(
        _obligation(), result, rollout_mode=RolloutMode.CANARY
    )

    assert shadow.can_continue is True
    assert shadow.blocking is False
    assert "shadow_fallback_validation_continues" in shadow.reason_codes
    for plan in (enforcement, canary):
        assert plan.can_continue is False
        assert plan.blocking is True
        assert plan.required_assurance is AssuranceLevel.KERNEL_VERIFIED
        assert "required_assurance_not_satisfied" in plan.reason_codes


def test_equivalent_failures_deduplicate_by_obligation_tree_and_example() -> None:
    deduplicator = ProofFallbackDeduplicator()
    router = ProofFallbackRouter(deduplicator=deduplicator)
    example_a = {"resource": "r1", "tokens": [7, 6]}
    example_a_reordered = {"tokens": [7, 6], "resource": "r1"}

    first = router.route(_obligation(), _disproved_result(example_a))
    duplicate = router.route(
        _obligation(), _disproved_result(example_a_reordered)
    )
    other_tree = router.route(
        _obligation(tree="git-tree:candidate-two"),
        _disproved_result(example_a),
    )

    assert len(first.diagnostics) == 1
    assert duplicate.diagnostics == ()
    assert duplicate.regression_fixtures == ()
    assert duplicate.deduplicated_count == 1
    assert "equivalent_failure_deduplicated" in duplicate.reason_codes
    assert len(other_tree.diagnostics) == 1
    assert deduplicator.count == 2

    batch = route_proof_fallbacks(
        [
            (_obligation(), _disproved_result(example_a)),
            (_obligation(), _disproved_result(example_a_reordered)),
        ]
    )
    assert [len(item.diagnostics) for item in batch] == [1, 0]


def test_missing_declared_fallback_fails_safe_to_manual_review() -> None:
    plan = route_proof_fallback(
        _obligation(checks=()),
        {"status": "unsupported", "reason": "unknown shape"},
    )

    assert [item.validation_id for item in plan.validations] == [
        "manual:unsupported-proof-obligation"
    ]
    assert plan.manual_review_requirements == plan.validations
    assert "manual_review_required" in plan.reason_codes


def test_plan_serialization_is_deterministic_and_contains_no_raw_transcript() -> None:
    raw = {
        "status": "disproved",
        "stdout": "unbounded unrelated solver transcript" * 1000,
        "evidence": {"counterexample": {"state": "invalid"}},
    }
    plan = route_proof_fallback(_obligation(), raw)
    payload = json.loads(plan.to_json())

    assert payload["proof_status"] == "disproved"
    assert "unbounded unrelated solver transcript" not in plan.to_json()
    assert plan.plan_id == replace(plan).plan_id
    assert plan.from_dict(plan.to_dict()) == plan
    outcome = plan.to_proof_outcome("policy-requirement:lease")
    assert outcome.requirement_id == "policy-requirement:lease"
    assert outcome.status is ProofResultStatus.DISPROVED
    assert outcome.authoritative_assurance is AssuranceLevel.UNVERIFIED

    with pytest.raises(ProofFallbackValidationError, match="proved obligation"):
        route_proof_fallback(_obligation(), {"status": "proved"})
