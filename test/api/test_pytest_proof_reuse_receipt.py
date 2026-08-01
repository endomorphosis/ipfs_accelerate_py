"""Tests for complete-pass receipt capture (PTR-052)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    PhaseOutcome,
    TestExecutionKey,
    TestLocatorKey,
    TestPassReceipt,
)
from ipfs_accelerate_py.testing.proof_reuse.receipt import (
    DISQUALIFIER_INCOMPLETE_PHASES,
    DISQUALIFIER_INCOMPLETE_TRACE,
    DISQUALIFIER_INTERRUPTED,
    DISQUALIFIER_LEAKED_RESOURCES,
    DISQUALIFIER_RERUN,
    DISQUALIFIER_SKIP,
    DISQUALIFIER_TEARDOWN_FAILURE,
    DISQUALIFIER_TIMEOUT,
    DISQUALIFIER_XFAIL,
    DISQUALIFIER_XPASS,
    ITEM_COLLECTOR_ATTRIBUTE,
    ITEM_RECEIPT_RESULT_ATTRIBUTE,
    ReceiptCaptureResult,
    TestPassReceiptCollector,
    attach_collector,
    clear_collectors,
    evaluate_complete_pass,
    finalize_test_pass_receipt,
    get_collector,
    map_report_to_phase_outcome,
    pytest_runtest_logreport,
    register_collector,
)


@dataclass
class _Report:
    nodeid: str = "test_example.py::test_example"
    when: str = "call"
    outcome: str = "passed"
    duration: float = 0.01
    wasxfail: Any = None
    keywords: dict[str, Any] = field(default_factory=dict)
    longreprtext: str = ""
    execution_count: int | None = None


@dataclass
class _Trace:
    complete: bool = True
    cid: str = "cid:runtime-trace"
    completeness_reasons: tuple[str, ...] = ()
    leaked_resources: bool = False

    @property
    def trace_cid(self) -> str:
        return self.cid

    @property
    def root_cid(self) -> str:
        return self.cid


class _Store:
    def __init__(
        self,
        *,
        fail: bool = False,
        reject: bool = False,
        wrong_cid: bool = False,
    ) -> None:
        self.fail = fail
        self.reject = reject
        self.wrong_cid = wrong_cid
        self.calls: list[Any] = []

    def put_receipt(self, receipt: Any) -> Any:
        self.calls.append(receipt)
        if self.fail:
            raise OSError("disk full")
        if self.reject:
            return SimpleNamespace(stored=False, cid="", reason_code="malformed")
        if self.wrong_cid:
            return SimpleNamespace(
                stored=True,
                cid="cid:not-the-receipt",
                reason_code="ok",
            )
        return SimpleNamespace(
            stored=True,
            cid=receipt.receipt_id,
            reason_code="ok",
        )


class _Issuer:
    def __init__(self, *, fail: bool = False, status: str = "certificate_deferred") -> None:
        self.fail = fail
        self.status = status
        self.calls: list[Any] = []

    def issue(self, request: Any) -> Any:
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("prover boom")
        return SimpleNamespace(status=self.status, reason="certificate_deferred")


def _locator() -> TestLocatorKey:
    return TestLocatorKey(
        repository_id="repository:example",
        package_identity="package:example",
        node_id="test_example.py::test_example",
    )


def _execution_key(locator: TestLocatorKey) -> TestExecutionKey:
    return TestExecutionKey(
        locator_cid=locator.locator_id,
        repository_forest_cid="cid:repository-forest",
        static_trace_root_cid="cid:static-trace",
        runtime_trace_root_cid="cid:runtime-trace",
        runtime_completeness_policy="complete-v1",
        policy_cid="cid:policy",
    )


def _complete_collector(
    *,
    setup: str = "passed",
    call: str = "passed",
    teardown: str = "passed",
    nodeid: str = "test_example.py::test_example",
    **report_kwargs: Any,
) -> TestPassReceiptCollector:
    collector = TestPassReceiptCollector(nodeid=nodeid)
    for when, outcome in (
        ("setup", setup),
        ("call", call),
        ("teardown", teardown),
    ):
        collector.record_report(
            _Report(nodeid=nodeid, when=when, outcome=outcome, **report_kwargs)
        )
    return collector


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    clear_collectors()
    yield
    clear_collectors()


def test_complete_pass_creates_reusable_admitted_receipt() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    store = _Store()
    collector = _complete_collector()
    trace = _Trace()

    result = finalize_test_pass_receipt(
        collector,
        locator=locator,
        execution_key=execution_key,
        runtime_trace=trace,
        store=store,
        issuer_key_id="key:issuer",
    )

    assert result.reusable is True
    assert result.admitted is True
    assert isinstance(result.receipt, TestPassReceipt)
    assert result.receipt.admitted is True
    assert result.receipt.all_phases_pass is True
    assert result.receipt.disqualifying_states == ()
    assert result.receipt.locator_cid == locator.locator_id
    assert result.receipt.execution_key_cid == execution_key.execution_key_id
    assert result.receipt.completeness_receipt_cid == trace.cid
    assert result.stored is True
    assert result.store_reason == "ok"
    assert result.receipt_cid == result.receipt.receipt_id
    assert len(store.calls) == 1
    assert store.calls[0].receipt_id == result.receipt_cid


def test_skip_does_not_create_reusable_receipt() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    collector = _complete_collector(call="skipped")

    result = finalize_test_pass_receipt(
        collector,
        locator=locator,
        execution_key=execution_key,
        store=_Store(),
    )

    assert result.reusable is False
    assert result.receipt is None
    assert DISQUALIFIER_SKIP in result.disqualifying_states


@pytest.mark.parametrize(
    ("outcome", "wasxfail", "expected"),
    [
        ("skipped", "reason: expected fail", DISQUALIFIER_XFAIL),
        ("passed", "reason: unexpected pass", DISQUALIFIER_XPASS),
    ],
)
def test_xfail_and_xpass_do_not_create_reusable_receipt(
    outcome: str,
    wasxfail: str,
    expected: str,
) -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    collector = TestPassReceiptCollector(nodeid="n")
    collector.record_report(_Report(when="setup", outcome="passed"))
    collector.record_report(
        _Report(when="call", outcome=outcome, wasxfail=wasxfail)
    )
    collector.record_report(_Report(when="teardown", outcome="passed"))

    result = finalize_test_pass_receipt(
        collector,
        locator=locator,
        execution_key=execution_key,
        store=_Store(),
    )

    assert result.reusable is False
    assert expected in result.disqualifying_states


def test_rerun_does_not_create_reusable_receipt() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    collector = TestPassReceiptCollector(nodeid="n")
    collector.record_report(_Report(when="setup", outcome="passed"))
    collector.record_report(
        _Report(
            when="call",
            outcome="passed",
            execution_count=2,
            keywords={"rerun": 1},
        )
    )
    collector.record_report(_Report(when="teardown", outcome="passed"))

    result = finalize_test_pass_receipt(
        collector,
        locator=locator,
        execution_key=execution_key,
    )

    assert result.reusable is False
    assert DISQUALIFIER_RERUN in result.disqualifying_states


def test_interruption_does_not_create_reusable_receipt() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    collector = _complete_collector()
    collector.mark_interrupted()

    result = finalize_test_pass_receipt(
        collector,
        locator=locator,
        execution_key=execution_key,
    )

    assert result.reusable is False
    assert DISQUALIFIER_INTERRUPTED in result.disqualifying_states


def test_timeout_does_not_create_reusable_receipt() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    collector = TestPassReceiptCollector(nodeid="n")
    collector.record_report(_Report(when="setup", outcome="passed"))
    collector.record_report(
        _Report(
            when="call",
            outcome="failed",
            keywords={"timeout": 1},
            longreprtext="Failed: Timeout >1.0s",
        )
    )
    collector.record_report(_Report(when="teardown", outcome="passed"))

    result = finalize_test_pass_receipt(
        collector,
        locator=locator,
        execution_key=execution_key,
    )

    assert result.reusable is False
    assert DISQUALIFIER_TIMEOUT in result.disqualifying_states


def test_teardown_failure_does_not_create_reusable_receipt() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    collector = _complete_collector(teardown="failed")

    result = finalize_test_pass_receipt(
        collector,
        locator=locator,
        execution_key=execution_key,
        store=_Store(),
    )

    assert result.reusable is False
    assert DISQUALIFIER_TEARDOWN_FAILURE in result.disqualifying_states
    assert result.receipt is None


def test_incomplete_trace_does_not_create_reusable_receipt() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    collector = _complete_collector()
    trace = _Trace(complete=False, completeness_reasons=("overflow",))

    result = finalize_test_pass_receipt(
        collector,
        locator=locator,
        execution_key=execution_key,
        runtime_trace=trace,
        store=_Store(),
    )

    assert result.reusable is False
    assert DISQUALIFIER_INCOMPLETE_TRACE in result.disqualifying_states


def test_leaked_resources_do_not_create_reusable_receipt() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    collector = _complete_collector()
    trace = _Trace(
        complete=True,
        completeness_reasons=("leaked_resources",),
        leaked_resources=True,
    )

    result = finalize_test_pass_receipt(
        collector,
        locator=locator,
        execution_key=execution_key,
        runtime_trace=trace,
    )

    assert result.reusable is False
    assert DISQUALIFIER_LEAKED_RESOURCES in result.disqualifying_states


def test_incomplete_phases_do_not_create_reusable_receipt() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    collector = TestPassReceiptCollector(nodeid="n")
    collector.record_report(_Report(when="setup", outcome="passed"))
    collector.record_report(_Report(when="call", outcome="passed"))
    # teardown missing

    result = finalize_test_pass_receipt(
        collector,
        locator=locator,
        execution_key=execution_key,
    )

    assert result.reusable is False
    assert DISQUALIFIER_INCOMPLETE_PHASES in result.disqualifying_states


def test_store_error_does_not_change_reusable_receipt_or_raise() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    store = _Store(fail=True)
    collector = _complete_collector()

    result = finalize_test_pass_receipt(
        collector,
        locator=locator,
        execution_key=execution_key,
        runtime_trace=_Trace(),
        store=store,
    )

    assert result.reusable is True
    assert result.admitted is True
    assert isinstance(result.receipt, TestPassReceipt)
    assert result.stored is False
    assert "store_error" in result.store_reason


def test_store_reject_and_cid_mismatch_are_non_fatal() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)

    rejected = finalize_test_pass_receipt(
        _complete_collector(),
        locator=locator,
        execution_key=execution_key,
        runtime_trace=_Trace(),
        store=_Store(reject=True),
    )
    assert rejected.reusable is True
    assert rejected.stored is False
    assert rejected.store_reason == "malformed"

    mismatched = finalize_test_pass_receipt(
        _complete_collector(),
        locator=locator,
        execution_key=execution_key,
        runtime_trace=_Trace(),
        store=_Store(wrong_cid=True),
    )
    assert mismatched.reusable is True
    assert mismatched.stored is False
    assert mismatched.store_reason == "store_cid_mismatch"


def test_prover_error_never_changes_test_result_fields() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    store = _Store()
    issuer = _Issuer(fail=True)
    collector = _complete_collector()

    result = finalize_test_pass_receipt(
        collector,
        locator=locator,
        execution_key=execution_key,
        runtime_trace=_Trace(),
        store=store,
        issuer=issuer,
        deferred_request={"receipt": "synthetic"},
    )

    assert result.reusable is True
    assert result.stored is True
    assert result.deferred_proving_status == "error"
    assert "prover_error" in result.deferred_proving_reason
    assert len(issuer.calls) == 1


def test_deferred_proving_after_store_success() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    issuer = _Issuer(status="certificate_deferred")

    result = finalize_test_pass_receipt(
        _complete_collector(),
        locator=locator,
        execution_key=execution_key,
        runtime_trace=_Trace(),
        store=_Store(),
        issuer=issuer,
    )

    assert result.reusable is True
    assert result.deferred_proving_status == "deferred"
    assert result.deferred_proving_reason in {
        "certificate_deferred",
        "deferred",
    }


def test_writes_receipts_false_skips_store_but_still_builds_receipt() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    store = _Store()

    result = finalize_test_pass_receipt(
        _complete_collector(),
        locator=locator,
        execution_key=execution_key,
        runtime_trace=_Trace(),
        store=store,
        writes_receipts=False,
    )

    assert result.reusable is True
    assert result.stored is False
    assert result.store_reason == "write_disabled"
    assert store.calls == []


def test_pytest_runtest_logreport_records_registered_collector() -> None:
    collector = TestPassReceiptCollector(nodeid="test_example.py::test_example")
    register_collector(collector)

    for when in ("setup", "call", "teardown"):
        pytest_runtest_logreport(
            _Report(
                nodeid="test_example.py::test_example",
                when=when,
                outcome="passed",
                duration=0.002,
            )
        )

    assert set(collector.phases) == {"setup", "call", "teardown"}
    eligible, disqualifiers = collector.evaluate()
    assert eligible is True
    assert disqualifiers == ()


def test_pytest_runtest_logreport_never_raises_on_bad_report() -> None:
    register_collector(TestPassReceiptCollector(nodeid="n"))
    # Missing attributes should be ignored rather than failing the session.
    pytest_runtest_logreport(object())
    pytest_runtest_logreport(SimpleNamespace(nodeid="n", when="call"))


def test_attach_collector_and_item_result_attribute() -> None:
    item = SimpleNamespace(nodeid="test_example.py::test_item", user_properties=[])
    collector = attach_collector(item)
    assert isinstance(getattr(item, ITEM_COLLECTOR_ATTRIBUTE), TestPassReceiptCollector)
    assert get_collector(item.nodeid) is collector

    locator = _locator()
    execution_key = _execution_key(locator)
    collector.record_phase("setup", PhaseOutcome.PASS)
    collector.record_phase("call", PhaseOutcome.PASS)
    collector.record_phase("teardown", PhaseOutcome.PASS)

    result = finalize_test_pass_receipt(
        item=item,
        locator=locator,
        execution_key=execution_key,
        runtime_trace=_Trace(),
        store=_Store(),
    )
    assert result.reusable is True
    assert getattr(item, ITEM_RECEIPT_RESULT_ATTRIBUTE) is result


def test_map_report_to_phase_outcome_covers_closed_set() -> None:
    assert map_report_to_phase_outcome(_Report(outcome="passed")) is PhaseOutcome.PASS
    assert map_report_to_phase_outcome(_Report(outcome="failed")) is PhaseOutcome.FAIL
    assert map_report_to_phase_outcome(_Report(outcome="skipped")) is PhaseOutcome.SKIP
    assert (
        map_report_to_phase_outcome(_Report(outcome="skipped", wasxfail="x"))
        is PhaseOutcome.XFAIL
    )
    assert (
        map_report_to_phase_outcome(_Report(outcome="passed", wasxfail="x"))
        is PhaseOutcome.XPASS
    )
    assert (
        map_report_to_phase_outcome(
            _Report(outcome="failed", keywords={"timeout": True})
        )
        is PhaseOutcome.ERROR
    )


def test_evaluate_complete_pass_direct_mapping() -> None:
    eligible, disqualifiers = evaluate_complete_pass(
        {
            "setup": PhaseOutcome.PASS,
            "call": PhaseOutcome.PASS,
            "teardown": PhaseOutcome.PASS,
        }
    )
    assert eligible is True
    assert disqualifiers == ()

    eligible, disqualifiers = evaluate_complete_pass(
        {
            "setup": PhaseOutcome.PASS,
            "call": PhaseOutcome.FAIL,
            "teardown": PhaseOutcome.PASS,
        }
    )
    assert eligible is False
    assert "fail" in disqualifiers


def test_receipt_capture_result_truthiness_is_reusable_only() -> None:
    assert bool(ReceiptCaptureResult(reusable=True)) is True
    assert bool(ReceiptCaptureResult(reusable=False)) is False


def test_finalize_swallows_unexpected_internal_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    collector = _complete_collector()

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(
        "ipfs_accelerate_py.testing.proof_reuse.receipt.TestPassReceipt",
        _boom,
    )
    result = finalize_test_pass_receipt(
        collector,
        locator=locator,
        execution_key=execution_key,
        runtime_trace=_Trace(),
    )
    assert result.reusable is False
    assert result.diagnostics.get("stage") in {"build_receipt", "finalize"}
