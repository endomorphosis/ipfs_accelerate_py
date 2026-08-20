"""FACP-046: Enforce protocol/runtime trace conformance.

Acceptance:
- Monitor accepts exactly normative vectors.
- Rejects stale fences / replay / incompatible idempotency / receipt arguments.
- Crash injection covers every persistent transition boundary in the harness.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.formal_transition_monitor import (
    ACTIONS,
    BUNDLE,
    CRASH_BOUNDARIES,
    EVIDENCE_SUBSET,
    GOAL_ID,
    HAPPY_PATH,
    MODEL_EVIDENCE,
    MONITOR_VERSION,
    REQUIRED_INVARIANTS,
    RUNTIME_MONITOR_SCHEMA,
    SCHEMA,
    TASK_ID,
    TYPESTATES,
    VECTOR_SCHEMA,
    VERDICT_SCHEMA,
    CrashInjectionHarness,
    FormalTransitionMonitor,
    MonitorErrorCode,
    TraceEvent,
    TransitionMonitorError,
    VectorAdapter,
    argument_cid_for,
    default_monitor,
    evaluate_all_normative_vectors,
    evaluate_normative_vector,
    happy_path_steps,
    load_normative_vectors,
    receipt_cid_for,
)


# ---------------------------------------------------------------------------
# Identity / hermetic import
# ---------------------------------------------------------------------------


def test_module_identity_and_evidence_envelope() -> None:
    assert TASK_ID == "FACP-046"
    assert GOAL_ID == "FACP-G510"
    assert SCHEMA == "facp/tep-monitor@1"
    assert RUNTIME_MONITOR_SCHEMA == "facp/runtime-monitor@1"
    assert BUNDLE == "facp/protocols/runtime"
    assert MODEL_EVIDENCE == "facp/tep-models@1"
    assert MONITOR_VERSION.startswith("formal-transition-monitor/")
    assert VERDICT_SCHEMA.startswith("facp/")
    assert VECTOR_SCHEMA.startswith("facp/")


def test_evidence_subset_and_invariants_are_closed() -> None:
    for token in (
        "prior_state",
        "next_state",
        "protocol",
        "instance",
        "operation",
        "actor",
        "fence",
        "idempotency",
        "observation",
        "time",
    ):
        assert token in EVIDENCE_SUBSET
    for name in (
        "NoDoubleEffect",
        "NoStaleFenceCompletion",
        "NoSuccessWithoutObservation",
        "NoConfirmationReuse",
        "NoBlindUnknownRetry",
    ):
        assert name in REQUIRED_INVARIANTS


def test_crash_boundaries_match_tep_model() -> None:
    expected = {
        "admission",
        "reservation",
        "started",
        "unknown",
        "observed",
        "receipt",
        "current",
        "lease",
        "fence",
        "retry",
        "idempotency",
        "crash",
        "settlement",
        "compensation",
        "proof_promotion",
    }
    assert set(CRASH_BOUNDARIES) == expected
    assert len(CRASH_BOUNDARIES) == len(expected)


def test_cold_import_is_hermetic() -> None:
    mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.runtime.formal_transition_monitor"
    )
    source = inspect.getsource(mod)
    for banned in ("socket.", "urllib.", "requests.", "subprocess.", "http.client"):
        assert banned not in source
    assert "TODO" not in source
    assert "FIXME" not in source


# ---------------------------------------------------------------------------
# Normative vectors — accept exactly
# ---------------------------------------------------------------------------


def test_normative_corpus_non_empty_and_partitioned() -> None:
    vectors = load_normative_vectors()
    assert len(vectors) >= 8
    accepts = [v for v in vectors if v.expect_accept]
    rejects = [v for v in vectors if not v.expect_accept]
    assert accepts and rejects
    ids = [v.vector_id for v in vectors]
    assert len(ids) == len(set(ids))


def test_monitor_accepts_exactly_normative_vectors() -> None:
    summary = evaluate_all_normative_vectors()
    assert summary["exact_match"] is True, summary["failures"]
    assert summary["failures"] == []
    assert summary["accepted_ok"] + summary["rejected_ok"] == summary["vector_count"]
    assert summary["task_id"] == TASK_ID
    assert summary["schema"] == SCHEMA


def test_each_normative_vector_individually() -> None:
    for vector in load_normative_vectors():
        verdict = evaluate_normative_vector(vector)
        if vector.expect_accept:
            assert verdict.accepted, (vector.vector_id, verdict.to_dict())
        else:
            assert not verdict.accepted, (vector.vector_id, verdict.to_dict())


def test_happy_path_trace_carries_evidence_subset() -> None:
    monitor = default_monitor()
    adapter = VectorAdapter()
    steps = adapter.adapt_model_trace(
        happy_path_steps("o1"),
        argument_cid=argument_cid_for({"k": "v"}),
        idempotency_key="idem:happy",
    )
    adapter.run_adapted(monitor, steps, expect_ok=True)
    st = monitor.instances["o1"]
    assert st.typestate == "ReceiptSealed"
    assert st.observed and st.receipt_sealed and st.proof_promoted
    assert st.effect_count == 1
    assert st.current_pointer == 1
    assert monitor.trace
    sample = monitor.trace[-1]
    for key in EVIDENCE_SUBSET:
        assert key in sample
    boundaries = {ev["boundary"] for ev in monitor.trace}
    for required in (
        "admission",
        "lease",
        "reservation",
        "started",
        "idempotency",
        "observed",
        "receipt",
        "current",
        "proof_promotion",
    ):
        assert required in boundaries


# ---------------------------------------------------------------------------
# Explicit rejection classes
# ---------------------------------------------------------------------------


def test_rejects_stale_fence_completion() -> None:
    monitor = default_monitor()
    o = "o1"
    monitor.run_steps(
        [
            *[{"action": "AdvanceAdmission", "instance_id": o} for _ in range(HAPPY_PATH.index("ObligationsSatisfied"))],
            {"action": "SatisfyConfirmation", "instance_id": o},
            {"action": "AcquireLease", "instance_id": o},
            {"action": "Reserve", "instance_id": o, "idempotency_key": "idem:1", "argument_cid": argument_cid_for({"a": 1})},
            {"action": "Start", "instance_id": o, "idempotency_key": "idem:1"},
            {"action": "ApplyEffect", "instance_id": o, "idempotency_key": "idem:1"},
            {"action": "Observe", "instance_id": o},
            {"action": "BumpFence", "instance_id": o},
        ],
        expect_ok=True,
    )
    verdict = monitor.apply_action("SealReceipt", instance_id=o)
    assert verdict.accepted is False
    assert verdict.code == MonitorErrorCode.STALE_FENCE.value


def test_rejects_event_replay() -> None:
    monitor = default_monitor()
    o = "o1"
    arg = argument_cid_for({"x": 1})
    monitor.run_steps(
        [
            *[{"action": "AdvanceAdmission", "instance_id": o} for _ in range(HAPPY_PATH.index("ObligationsSatisfied"))],
            {"action": "SatisfyConfirmation", "instance_id": o, "confirmation_cid": "confirm:1"},
            {"action": "AcquireLease", "instance_id": o},
            {
                "action": "Reserve",
                "instance_id": o,
                "idempotency_key": "idem:1",
                "argument_cid": arg,
                "event_id": "evt-1",
            },
        ],
        expect_ok=True,
    )
    verdict = monitor.apply_action(
        "Start",
        instance_id=o,
        idempotency_key="idem:1",
        argument_cid=arg,
        event_id="evt-1",
    )
    assert verdict.accepted is False
    assert verdict.code == MonitorErrorCode.REPLAY.value


def test_rejects_incompatible_idempotency_key() -> None:
    monitor = default_monitor()
    o = "o1"
    arg = argument_cid_for({"x": 1})
    monitor.run_steps(
        [
            *[{"action": "AdvanceAdmission", "instance_id": o} for _ in range(HAPPY_PATH.index("ObligationsSatisfied"))],
            {"action": "SatisfyConfirmation", "instance_id": o},
            {"action": "AcquireLease", "instance_id": o},
            {
                "action": "Reserve",
                "instance_id": o,
                "idempotency_key": "idem:alpha",
                "argument_cid": arg,
            },
        ],
        expect_ok=True,
    )
    verdict = monitor.apply_action(
        "Start",
        instance_id=o,
        argument_cid=arg,
        idempotency_key="idem:beta",
    )
    assert verdict.accepted is False
    assert verdict.code == MonitorErrorCode.INCOMPATIBLE_IDEMPOTENCY.value


def test_rejects_incompatible_receipt_arguments() -> None:
    monitor = default_monitor()
    o = "o1"
    arg = argument_cid_for({"x": 1})
    idem = "idem:receipt"
    monitor.run_steps(
        [
            *[{"action": "AdvanceAdmission", "instance_id": o} for _ in range(HAPPY_PATH.index("ObligationsSatisfied"))],
            {"action": "SatisfyConfirmation", "instance_id": o},
            {"action": "AcquireLease", "instance_id": o},
            {"action": "Reserve", "instance_id": o, "idempotency_key": idem, "argument_cid": arg},
            {"action": "Start", "instance_id": o, "idempotency_key": idem},
            {"action": "ApplyEffect", "instance_id": o, "idempotency_key": idem},
            {"action": "Observe", "instance_id": o},
        ],
        expect_ok=True,
    )
    verdict = monitor.apply_action(
        "SealReceipt",
        instance_id=o,
        argument_cid=arg,
        idempotency_key=idem,
        receipt_cid="receipt:sha256:forged",
    )
    assert verdict.accepted is False
    assert verdict.code == MonitorErrorCode.INCOMPATIBLE_RECEIPT.value


def test_compatible_receipt_arguments_accepted() -> None:
    monitor = default_monitor()
    o = "o1"
    arg = argument_cid_for({"x": 1})
    idem = "idem:ok"
    monitor.run_steps(
        [
            *[{"action": "AdvanceAdmission", "instance_id": o} for _ in range(HAPPY_PATH.index("ObligationsSatisfied"))],
            {"action": "SatisfyConfirmation", "instance_id": o},
            {"action": "AcquireLease", "instance_id": o},
            {"action": "Reserve", "instance_id": o, "idempotency_key": idem, "argument_cid": arg},
            {"action": "Start", "instance_id": o, "idempotency_key": idem},
            {"action": "ApplyEffect", "instance_id": o, "idempotency_key": idem},
            {"action": "Observe", "instance_id": o},
        ],
        expect_ok=True,
    )
    st = monitor.instances[o]
    expected = receipt_cid_for(
        instance_id=o,
        argument_cid=st.argument_cid,
        idempotency_key=st.idempotency_key,
        observation_cid=st.observation_cid,
    )
    verdict = monitor.apply_action(
        "SealReceipt",
        instance_id=o,
        argument_cid=arg,
        idempotency_key=idem,
        receipt_cid=expected,
    )
    assert verdict.accepted is True


def test_rejects_double_effect() -> None:
    monitor = default_monitor()
    o = "o1"
    monitor.run_steps(
        [
            *[{"action": "AdvanceAdmission", "instance_id": o} for _ in range(HAPPY_PATH.index("ObligationsSatisfied"))],
            {"action": "SatisfyConfirmation", "instance_id": o},
            {"action": "AcquireLease", "instance_id": o},
            {"action": "Reserve", "instance_id": o, "idempotency_key": "idem:1"},
            {"action": "Start", "instance_id": o, "idempotency_key": "idem:1"},
            {"action": "ApplyEffect", "instance_id": o, "idempotency_key": "idem:1"},
        ],
        expect_ok=True,
    )
    verdict = monitor.apply_action("ApplyEffect", instance_id=o, idempotency_key="idem:1")
    assert verdict.accepted is False
    assert verdict.code == MonitorErrorCode.NO_DOUBLE_EFFECT.value


def test_rejects_blind_unknown_retry_for_irreversible() -> None:
    monitor = default_monitor()
    o = "o1"
    monitor.run_steps(
        [
            *[{"action": "AdvanceAdmission", "instance_id": o, "reversibility": "irreversible"} for _ in range(HAPPY_PATH.index("ObligationsSatisfied"))],
            {"action": "SatisfyConfirmation", "instance_id": o, "reversibility": "irreversible"},
            {"action": "AcquireLease", "instance_id": o, "reversibility": "irreversible"},
            {"action": "Reserve", "instance_id": o, "reversibility": "irreversible", "idempotency_key": "idem:1"},
            {"action": "Start", "instance_id": o, "reversibility": "irreversible", "idempotency_key": "idem:1"},
            {"action": "ApplyEffect", "instance_id": o, "reversibility": "irreversible", "idempotency_key": "idem:1"},
            {"action": "EnterUnknown", "instance_id": o, "reversibility": "irreversible"},
            {"action": "Fail", "instance_id": o, "reversibility": "irreversible"},
        ],
        expect_ok=True,
    )
    verdict = monitor.apply_action("Retry", instance_id=o, reversibility="irreversible")
    assert verdict.accepted is False
    assert verdict.code == MonitorErrorCode.NO_BLIND_UNKNOWN_RETRY.value


def test_rejects_confirmation_reuse() -> None:
    monitor = default_monitor()
    o = "o1"
    monitor.run_steps(
        [
            *[{"action": "AdvanceAdmission", "instance_id": o} for _ in range(HAPPY_PATH.index("ObligationsSatisfied"))],
            {"action": "SatisfyConfirmation", "instance_id": o, "confirmation_cid": "confirm:1"},
        ],
        expect_ok=True,
    )
    st = monitor.instances[o]
    st.typestate = "ObligationsSatisfied"
    verdict = monitor.apply_action(
        "SatisfyConfirmation",
        instance_id=o,
        confirmation_cid="confirm:1",
    )
    assert verdict.accepted is False
    assert verdict.code == MonitorErrorCode.NO_CONFIRMATION_REUSE.value


def test_rejects_unknown_action() -> None:
    monitor = default_monitor()
    verdict = monitor.apply_action("NotARealAction", instance_id="o1")
    assert verdict.accepted is False
    assert verdict.code == MonitorErrorCode.UNKNOWN_ACTION.value


def test_rejects_actor_or_operation_identity_drift() -> None:
    monitor = default_monitor()
    monitor.apply_action_or_raise(
        "AdvanceAdmission",
        instance_id="o1",
        operation="op.a",
        actor="actor.a",
    )
    with pytest.raises(TransitionMonitorError) as exc:
        monitor.apply_action_or_raise(
            "AdvanceAdmission",
            instance_id="o1",
            operation="op.b",
            actor="actor.a",
        )
    assert exc.value.code == MonitorErrorCode.REPLAY


# ---------------------------------------------------------------------------
# Crash injection harness
# ---------------------------------------------------------------------------


def test_crash_injection_covers_every_persistent_boundary() -> None:
    harness = CrashInjectionHarness()
    report = harness.cover_all_boundaries()
    assert report["complete"] is True, report["missing"]
    assert report["missing"] == []
    assert set(report["covered"]) == set(CRASH_BOUNDARIES)
    for boundary in CRASH_BOUNDARIES:
        assert boundary in report["details"]
        assert all(report["details"][boundary]["invariants"].values())


@pytest.mark.parametrize("boundary", list(CRASH_BOUNDARIES))
def test_crash_injection_per_boundary(boundary: str) -> None:
    harness = CrashInjectionHarness()
    monitor = harness.inject_at_boundary(boundary)
    assert monitor.crashed is False
    crash_events = [
        ev
        for ev in monitor.trace
        if ev.get("action") == "Crash" and ev.get("named_crash_boundary") == boundary
    ]
    assert crash_events
    recover = [ev for ev in monitor.trace if ev.get("action") == "Recover"]
    assert recover


# ---------------------------------------------------------------------------
# Vector adapter
# ---------------------------------------------------------------------------


def test_vector_adapter_model_and_runtime_events() -> None:
    adapter = VectorAdapter()
    model_steps = adapter.adapt_model_trace(
        [
            ("AdvanceAdmission", "o1"),
            ("Crash:admission", None),
            ("Recover", None),
        ]
    )
    assert model_steps[1].action == "Crash"
    assert model_steps[1].kwargs["named_crash_boundary"] == "admission"

    runtime = adapter.adapt_runtime_event(
        {
            "event": "Reserve",
            "instance_id": "o1",
            "operation_id": "accelerate.inference",
            "actor_cid": "actor:1",
            "idempotency_key": "idem:rt",
            "argument_cid": argument_cid_for({"p": 1}),
        }
    )
    assert runtime.action == "Reserve"
    assert runtime.instance_id == "o1"
    assert runtime.kwargs["operation"] == "accelerate.inference"
    assert runtime.kwargs["idempotency_key"] == "idem:rt"

    monitor = default_monitor()
    # Drive a short adapted path.
    short = adapter.adapt_model_trace(
        [
            *[( "AdvanceAdmission", "o1") for _ in range(HAPPY_PATH.index("ObligationsSatisfied"))],
            ("SatisfyConfirmation", "o1"),
            ("AcquireLease", "o1"),
            ("Reserve", "o1"),
        ],
        argument_cid=argument_cid_for({"p": 1}),
        idempotency_key="idem:rt",
    )
    verdicts = adapter.run_adapted(monitor, short, expect_ok=True)
    assert verdicts[-1].accepted is True
    assert monitor.instances["o1"].typestate == "Reserved"


def test_apply_event_round_trip_with_trace_event() -> None:
    monitor = default_monitor()
    monitor.apply_action_or_raise(
        "AdvanceAdmission",
        instance_id="o1",
        operation="op.default",
        actor="actor.default",
    )
    st = monitor.instances["o1"]
    event = TraceEvent(
        action="AdvanceAdmission",
        prior_state=st.typestate,
        next_state="ActorAuthenticated",  # may not match exact; monitor applies live
        protocol=st.protocol,
        instance=st.instance_id,
        operation=st.operation,
        actor=st.actor,
        fence=st.fence_gen,
        idempotency=st.idempotency_key,
        observation=st.observed,
        time=st.time + 1,
        boundary="admission",
    )
    # Prior in event is ContractResolved after first advance; build correctly.
    event = TraceEvent(
        action="AdvanceAdmission",
        prior_state="ContractResolved",
        next_state="ActorAuthenticated",
        protocol=st.protocol,
        instance="o1",
        operation="op.default",
        actor="actor.default",
        fence=st.fence_gen,
        idempotency="",
        observation=False,
        time=2,
        boundary="admission",
    )
    verdict = monitor.apply_event(event)
    assert verdict.accepted is True


def test_apply_action_or_raise_surfaces_codes() -> None:
    monitor = default_monitor()
    with pytest.raises(TransitionMonitorError) as exc:
        monitor.apply_action_or_raise("Recover")
    assert exc.value.code == MonitorErrorCode.NOT_CRASHED


def test_typestate_and_action_vocabularies_are_closed() -> None:
    assert "Proposed" in TYPESTATES
    assert "ReceiptSealed" in TYPESTATES
    assert "ApplyEffect" in ACTIONS
    assert "Crash" in ACTIONS
    assert "Recover" in ACTIONS
    assert len(HAPPY_PATH) == 12


def test_monitor_verdict_to_dict_schema() -> None:
    monitor = default_monitor()
    verdict = monitor.apply_action("Reject", instance_id="o1")
    payload = verdict.to_dict()
    assert payload["schema"] == VERDICT_SCHEMA
    assert payload["task_id"] == TASK_ID
    assert payload["accepted"] is True
    assert "invariants" in payload


def test_no_success_without_observation_on_settle() -> None:
    """Forged state cannot settle current without observation."""

    monitor = default_monitor()
    o = "o1"
    monitor.run_steps(
        [
            *[{"action": "AdvanceAdmission", "instance_id": o} for _ in range(HAPPY_PATH.index("ObligationsSatisfied"))],
            {"action": "SatisfyConfirmation", "instance_id": o},
            {"action": "AcquireLease", "instance_id": o},
            {"action": "Reserve", "instance_id": o, "idempotency_key": "idem:1"},
            {"action": "Start", "instance_id": o, "idempotency_key": "idem:1"},
            {"action": "Fail", "instance_id": o},
            {"action": "SealReceipt", "instance_id": o},
        ],
        expect_ok=True,
    )
    # Failure receipts may seal without observation, but must not settle current.
    st = monitor.instances[o]
    st.pending_current = st.current_pointer + 1
    verdict = monitor.apply_action("SettleCurrent", instance_id=o)
    assert verdict.accepted is False
    assert verdict.code == MonitorErrorCode.NO_SUCCESS_WITHOUT_OBSERVATION.value
