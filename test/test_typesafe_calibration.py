from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    HIGH_CONFIDENCE,
    SmtTriageAction,
    triage_smt,
)
from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_calibration import (
    clear_samples,
    mismatch_rate,
    recommend_skip_policy,
    record_sample,
    should_trust_skip,
)


def setup_function() -> None:
    clear_samples()


def test_float32_miss_forces_z3_and_never_auto_applies() -> None:
    record_sample(
        family="float32_point_one_plus_point_two",
        predicted="unsat",
        actual="sat",
        confidence=0.48,
        trap_family=True,
        case_id="float32_point_one_plus_point_two",
    )
    assert mismatch_rate(family="float32_point_one_plus_point_two") == 1.0
    assert (
        should_trust_skip(
            family="float32_point_one_plus_point_two",
            confidence=HIGH_CONFIDENCE,
            trap_family=True,
        )
        is False
    )
    policy = recommend_skip_policy()
    assert policy["auto_apply"] is False
    assert policy["accepted_as_authority"] is False
    assert "float32_point_one_plus_point_two" in policy["never_skip_families"]


def test_high_confidence_non_trap_may_skip_without_samples() -> None:
    assert (
        should_trust_skip(family="fol_identity", confidence=0.92, trap_family=False)
        is True
    )


def test_mismatch_rate_forces_z3_on_otherwise_skippable_family(
    monkeypatch,
) -> None:
    record_sample(
        family="fol_identity_cal",
        predicted="sat",
        actual="unsat",
        confidence=0.9,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.typesafe_permitted",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_z3_benchmark.run_typesafe",
        lambda _case, timeout=30.0: type(
            "T",
            (),
            {"status": "unsat", "confidence": 0.92, "usage": {}},
        )(),
    )
    receipt = triage_smt(
        english="identity",
        smtlib="(set-logic UF)\n(check-sat)\n",
        case_id="fol_identity_cal",
        complexity="easy",
    )
    assert receipt.action == SmtTriageAction.RUN_Z3.value
    assert "calibration_force_z3" in receipt.reason_codes
