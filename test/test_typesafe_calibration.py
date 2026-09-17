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


def test_compare_timeout_is_hint_only_not_a_sample(monkeypatch) -> None:
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_calibration import (
        samples,
    )
    from ipfs_accelerate_py.typesafe_z3_benchmark import SmtCase, compare_case

    clear_samples()
    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_z3_benchmark.run_z3",
        lambda *_a, **_k: type(
            "Z",
            (),
            {
                "status": "timeout",
                "timeout": True,
                "seconds": 8.0,
                "to_dict": lambda self: {"status": "timeout", "seconds": 8, "timeout": True},
            },
        )(),
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_z3_benchmark.run_typesafe",
        lambda *_a, **_k: type(
            "T",
            (),
            {
                "status": "sat",
                "confidence": 0.9,
                "seconds": 0.1,
                "to_dict": lambda self: {"status": "sat", "seconds": 0.1, "confidence": 0.9},
            },
        )(),
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_z3_benchmark.typesafe_configured",
        lambda: True,
    )
    row = compare_case(
        SmtCase(
            case_id="hint-timeout",
            english="x",
            smtlib="(check-sat)",
            expected="sat",
            complexity="hard",
        ),
        call_typesafe=True,
        z3_timeout_seconds=1.0,
    )
    assert row["typesafe_hint_only"] is True
    assert all(item.family != "hint-timeout" for item in samples())


def test_spot_check_can_force_z3_without_changing_skip_policy(monkeypatch) -> None:
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_calibration import (
        spot_check_due,
    )

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_calibration.random.random",
        lambda: 0.0,
    )
    assert spot_check_due(rate=0.01) is True
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_calibration.random.random",
        lambda: 0.99,
    )
    assert spot_check_due(rate=0.01) is False
    monkeypatch.setenv("TYPESAFE_Z3_SPOT_CHECK_RATE", "0")
    assert spot_check_due() is False


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
