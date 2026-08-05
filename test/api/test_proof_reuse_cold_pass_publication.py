"""Cold runtime-trace capture and final pass-candidate assembly (PTR-146).

Acceptance covered:

* Tracer starts immediately before setup and stops only after teardown.
* Setup, call, and teardown each pass exactly once; body is never re-invoked.
* Complete observed trace is canonical and bound into a newly compiled final
  execution key.
* Receipt binds that final key and trace.
* Candidate descriptor and every required canonical component are retained.
* Skipped, xfailed, failed, incomplete, uncontrolled, overflowed, or
  exceptional traces publish nothing authoritative.
* Tracing faults never alter pytest's real outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.test_execution_identity import (
    mint_content_identity,
)
from ipfs_accelerate_py.agent_supervisor.analysis.test_runtime_dependency_trace import (
    RuntimeTestDependencyTracer,
    RuntimeTraceCompleteness,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_candidate_context_store import (
    REQUIRED_COMPONENT_KEYS,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    PhaseOutcome,
    TestExecutionKey,
    TestLocatorKey,
    TestPassReceipt,
)
from ipfs_accelerate_py.testing.proof_reuse.candidate_publication import (
    CANDIDATE_PUBLICATION_ENVELOPE_INTERFACE,
    COMPLETED_EXECUTION_IDENTITY_INTERFACE,
    CandidatePublicationEnvelope,
    CompletedExecutionIdentity,
    assemble_candidate_publication,
    build_completed_execution_identity,
    finalize_cold_pass_publication,
    publication_is_authoritative,
)
from ipfs_accelerate_py.testing.proof_reuse.receipt import (
    DISQUALIFIER_EXCEPTIONAL_TRACE,
    DISQUALIFIER_FAIL,
    DISQUALIFIER_INCOMPLETE_TRACE,
    DISQUALIFIER_OVERFLOWED_TRACE,
    DISQUALIFIER_SKIP,
    DISQUALIFIER_UNCONTROLLED_TRACE,
    DISQUALIFIER_XFAIL,
    TestPassReceiptCollector,
    evaluate_complete_pass,
    finalize_test_pass_receipt,
)
from ipfs_accelerate_py.testing.proof_reuse.runtime_trace_lifecycle import (
    ITEM_RUNTIME_LIFECYCLE_ATTRIBUTE,
    LifecycleAuthority,
    PYTEST_RUNTIME_TRACE_LIFECYCLE_INTERFACE,
    PytestRuntimeTraceLifecycle,
    attach_runtime_lifecycle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    identity = mint_content_identity(
        {
            "schema": "ipfs_accelerate_py/test/cold-pass-label@1",
            "label": label,
        }
    )
    return str(getattr(identity, "cid", identity))


@dataclass
class _Report:
    nodeid: str = "test_cold.py::test_example"
    when: str = "call"
    outcome: str = "passed"
    duration: float = 0.01
    wasxfail: Any = None
    keywords: dict[str, Any] = field(default_factory=dict)
    longreprtext: str = ""


def _complete_collector(
    *,
    setup: str = "passed",
    call: str = "passed",
    teardown: str = "passed",
    nodeid: str = "test_cold.py::test_example",
) -> TestPassReceiptCollector:
    collector = TestPassReceiptCollector(nodeid=nodeid)
    for when, outcome in (
        ("setup", setup),
        ("call", call),
        ("teardown", teardown),
    ):
        collector.record_report(_Report(nodeid=nodeid, when=when, outcome=outcome))
    return collector


def _live_complete_trace(tmp_path: Path) -> Any:
    tracer = RuntimeTestDependencyTracer(
        allowed_roots={"repo": tmp_path},
        capture_code_objects=False,
    )
    with tracer:
        # Observe a harmless environment-allowlisted touch is optional; an
        # empty bounded session is already complete when instrumentation is
        # healthy.
        pass
    trace = tracer.result
    assert trace is not None
    assert trace.complete is True
    return trace


class _FakeIncompleteTrace:
    complete = False
    completeness_reasons = ("private_event",)
    trace_cid = ""
    cid = ""
    retained_canonical_bytes = b"{}"

    @property
    def root_cid(self) -> str:
        return ""


class _FakeOverflowTrace:
    complete = False
    completeness_reasons = ("overflow",)
    trace_cid = "cid:overflow"
    cid = "cid:overflow"
    retained_canonical_bytes = b"{}"

    @property
    def root_cid(self) -> str:
        return self.cid


class _FakeExceptionalTrace:
    complete = False
    completeness_reasons = ("instrumentation_failure",)
    trace_cid = "cid:exceptional"
    cid = "cid:exceptional"
    retained_canonical_bytes = b"{}"
    health = {"internal_failure_kinds": ["audit_callback"]}

    @property
    def root_cid(self) -> str:
        return self.cid


class _FakeUncontrolledTrace:
    complete = False
    completeness_reasons = ("unsupported_event",)
    trace_cid = "cid:uncontrolled"
    cid = "cid:uncontrolled"
    retained_canonical_bytes = b"{}"
    health = {"unsupported_event_kinds": ["network"]}

    @property
    def root_cid(self) -> str:
        return self.cid


class _RaisingTracer:
    """Tracer whose start/stop raise; must not alter pytest outcomes."""

    def start(self) -> "_RaisingTracer":
        raise RuntimeError("tracer start boom")

    def stop(self) -> None:
        raise RuntimeError("tracer stop boom")


# ---------------------------------------------------------------------------
# Lifecycle: start before setup, stop after teardown, body never re-entered
# ---------------------------------------------------------------------------


def test_lifecycle_starts_before_setup_and_stops_after_teardown(tmp_path: Path) -> None:
    events: list[str] = []

    class _RecordingTracer:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False
            self._result = None

        def start(self) -> "_RecordingTracer":
            events.append("tracer_start")
            self.started = True
            return self

        def stop(self) -> Any:
            events.append("tracer_stop")
            self.stopped = True
            # Produce a real complete empty trace for authority evaluation.
            real = RuntimeTestDependencyTracer(
                allowed_roots={"repo": tmp_path},
                capture_code_objects=False,
            )
            real.start()
            self._result = real.stop()
            return self._result

    lifecycle = PytestRuntimeTraceLifecycle(
        nodeid="test_cold.py::test_example",
        locator_cid=_cid("locator"),
        tracer_factory=_RecordingTracer,
    )

    events.append("before_start")
    assert lifecycle.start() is True
    events.append("after_start_before_setup")

    assert lifecycle.note_phase("setup", "passed")
    events.append("setup_done")
    assert lifecycle.note_phase("call", "passed")
    events.append("call_done")
    assert lifecycle.note_phase("teardown", "passed")
    events.append("teardown_done")

    trace = lifecycle.stop()
    events.append("after_stop")

    assert events[0] == "before_start"
    assert events[1] == "tracer_start"
    assert events[2] == "after_start_before_setup"
    assert "setup_done" in events
    assert "call_done" in events
    assert "teardown_done" in events
    # Stop only after teardown.
    assert events.index("teardown_done") < events.index("tracer_stop")
    assert events[-1] == "after_stop"

    assert lifecycle.setup_count == 1
    assert lifecycle.call_count == 1
    assert lifecycle.teardown_count == 1
    assert lifecycle.body_invocations == 0
    assert lifecycle.lifecycle_complete is True
    assert lifecycle.phases_each_once is True
    assert lifecycle.interface == PYTEST_RUNTIME_TRACE_LIFECYCLE_INTERFACE
    assert lifecycle.may_authorize_skip is False
    assert trace is not None
    assert lifecycle.publishes_authoritatively is True
    assert lifecycle.authority is LifecycleAuthority.COMPLETE_PASS


def test_lifecycle_never_invokes_body_twice(tmp_path: Path) -> None:
    body_calls = {"n": 0}

    def body() -> str:
        body_calls["n"] += 1
        return "ok"

    lifecycle = PytestRuntimeTraceLifecycle(
        nodeid="test_cold.py::test_example",
        tracer_factory=lambda: RuntimeTestDependencyTracer(
            allowed_roots={"repo": tmp_path},
            capture_code_objects=False,
        ),
    )
    lifecycle.start()
    # Ordinary pytest path: setup/call/teardown noted; body runs once outside.
    lifecycle.note_phase("setup", PhaseOutcome.PASS)
    result = body()
    lifecycle.note_phase("call", PhaseOutcome.PASS)
    lifecycle.note_phase("teardown", PhaseOutcome.PASS)
    lifecycle.stop()

    assert result == "ok"
    assert body_calls["n"] == 1
    assert lifecycle.body_invocations == 0
    assert lifecycle.test_call_count == 1
    # Lifecycle has no run_body / execute entry that could re-enter.
    assert not hasattr(lifecycle, "run_body")
    assert not hasattr(lifecycle, "execute_call")


def test_duplicate_phase_notes_disqualify_authority(tmp_path: Path) -> None:
    lifecycle = PytestRuntimeTraceLifecycle(
        nodeid="test_cold.py::test_example",
        tracer_factory=lambda: RuntimeTestDependencyTracer(
            allowed_roots={"repo": tmp_path},
            capture_code_objects=False,
        ),
    )
    lifecycle.start()
    lifecycle.note_phase("setup", "passed")
    lifecycle.note_phase("call", "passed")
    # Duplicate call note is a lifecycle fault.
    assert lifecycle.note_phase("call", "passed") is False
    lifecycle.note_phase("teardown", "passed")
    lifecycle.stop()
    assert lifecycle.publishes_authoritatively is False
    assert lifecycle.fault


def test_tracing_faults_never_raise_into_caller(tmp_path: Path) -> None:
    lifecycle = PytestRuntimeTraceLifecycle(
        nodeid="test_cold.py::test_example",
        tracer_factory=_RaisingTracer,
    )
    # start() swallows the tracer exception.
    assert lifecycle.start() is False
    assert lifecycle.fault
    # note/stop also fail open.
    assert lifecycle.note_phase("setup", "passed") is True
    assert lifecycle.stop() is None
    assert lifecycle.publishes_authoritatively is False


def test_attach_runtime_lifecycle_is_idempotent(tmp_path: Path) -> None:
    item = SimpleNamespace(nodeid="test_cold.py::test_example")
    first = attach_runtime_lifecycle(
        item,
        allowed_roots={"repo": tmp_path},
        tracer_factory=lambda: RuntimeTestDependencyTracer(
            allowed_roots={"repo": tmp_path},
            capture_code_objects=False,
        ),
    )
    second = attach_runtime_lifecycle(item)
    assert first is second
    assert getattr(item, ITEM_RUNTIME_LIFECYCLE_ATTRIBUTE) is first


# ---------------------------------------------------------------------------
# Final execution key + receipt + candidate envelope
# ---------------------------------------------------------------------------


def test_complete_trace_bound_into_newly_compiled_final_key(tmp_path: Path) -> None:
    trace = _live_complete_trace(tmp_path)
    locator_cid = _cid("locator-final")
    seed_key = TestExecutionKey(
        locator_cid=locator_cid,
        repository_forest_cid=_cid("forest"),
        static_trace_root_cid=_cid("static"),
        runtime_trace_root_cid=_cid("placeholder-runtime"),
        policy_cid=_cid("policy"),
        environment_cid=_cid("env"),
        test_ast_cid=_cid("ast"),
    )

    completed = build_completed_execution_identity(
        locator_cid=locator_cid,
        runtime_trace=trace,
        seed_execution_key=seed_key,
    )
    assert completed is not None
    assert isinstance(completed, CompletedExecutionIdentity)
    assert completed.interface == COMPLETED_EXECUTION_IDENTITY_INTERFACE
    assert completed.may_authorize_skip is False
    assert completed.runtime_trace_root_cid == trace.trace_cid
    # Newly compiled: not the collection-time placeholder runtime CID.
    assert completed.execution_key.runtime_trace_root_cid == trace.trace_cid
    assert completed.execution_key.runtime_trace_root_cid != seed_key.runtime_trace_root_cid
    assert completed.execution_key_cid == completed.execution_key.execution_key_id
    assert completed.execution_key_cid != seed_key.execution_key_id
    assert completed.retained_execution_key_bytes == completed.execution_key.canonical_bytes()
    assert completed.retained_runtime_trace_bytes == trace.retained_canonical_bytes


def test_receipt_binds_final_key_and_trace(tmp_path: Path) -> None:
    trace = _live_complete_trace(tmp_path)
    locator_cid = _cid("locator-receipt")
    collector = _complete_collector()

    result, completed, publication = finalize_cold_pass_publication(
        collector=collector,
        runtime_trace=trace,
        locator_cid=locator_cid,
        repository_forest_cid=_cid("forest"),
        static_trace_root_cid=_cid("static"),
        policy_cid=_cid("policy"),
        environment_cid=_cid("env"),
        test_ast_cid=_cid("ast"),
    )

    assert completed is not None
    assert result.admitted is True
    assert result.receipt is not None
    assert isinstance(result.receipt, TestPassReceipt)
    assert result.receipt.execution_key_cid == completed.execution_key_cid
    assert result.receipt.runtime_trace_root_cid == trace.trace_cid
    assert result.receipt.locator_cid == locator_cid
    assert result.receipt.admitted is True
    assert publication is not None
    assert publication.receipt_cid == result.receipt.receipt_id
    assert publication.execution_key_cid == completed.execution_key_cid


def test_candidate_retains_descriptor_and_required_components(tmp_path: Path) -> None:
    trace = _live_complete_trace(tmp_path)
    locator_cid = _cid("locator-candidate")
    collector = _complete_collector()

    result, completed, publication = finalize_cold_pass_publication(
        collector=collector,
        runtime_trace=trace,
        locator_cid=locator_cid,
        repository_forest_cid=_cid("forest"),
        static_trace_root_cid=_cid("static"),
        policy_cid=_cid("policy"),
        environment_cid=_cid("env"),
        test_ast_cid=_cid("ast"),
    )

    assert result.admitted is True
    assert completed is not None
    assert publication is not None
    assert isinstance(publication, CandidatePublicationEnvelope)
    assert publication.interface == CANDIDATE_PUBLICATION_ENVELOPE_INTERFACE
    assert publication.may_authorize_skip is False
    assert publication.authoritative is True
    assert publication.required_components_present() is True
    for name in REQUIRED_COMPONENT_KEYS:
        assert name in publication.component_bytes
        assert publication.component_bytes[name]
        assert name in publication.component_cids
        assert publication.component_cids[name]
    assert publication.descriptor.execution_key_cid == completed.execution_key_cid
    assert publication.descriptor.runtime_trace_root_cid == trace.trace_cid
    assert publication.descriptor.pass_receipt_cid == result.receipt.receipt_id
    assert publication.descriptor.may_authorize_skip is False
    assert publication.retained_descriptor_bytes == publication.descriptor.canonical_bytes()
    assert publication_is_authoritative(publication) is True


# ---------------------------------------------------------------------------
# Non-authoritative paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setup", "call", "teardown", "expected_marker"),
    [
        ("skipped", "passed", "passed", DISQUALIFIER_SKIP),
        ("passed", "skipped", "passed", DISQUALIFIER_SKIP),
        ("passed", "failed", "passed", DISQUALIFIER_FAIL),
        ("passed", "passed", "failed", "teardown_failure"),
    ],
)
def test_non_pass_phases_publish_nothing_authoritative(
    tmp_path: Path,
    setup: str,
    call: str,
    teardown: str,
    expected_marker: str,
) -> None:
    trace = _live_complete_trace(tmp_path)
    collector = _complete_collector(setup=setup, call=call, teardown=teardown)
    result, completed, publication = finalize_cold_pass_publication(
        collector=collector,
        runtime_trace=trace,
        locator_cid=_cid("locator-fail"),
        repository_forest_cid=_cid("forest"),
        static_trace_root_cid=_cid("static"),
        policy_cid=_cid("policy"),
    )
    assert result.admitted is False
    assert publication is None
    # completed identity may still be built from the trace, but without an
    # admitted receipt it cannot form an authoritative publication.
    if completed is not None:
        assert publication_is_authoritative(None) is False
    assert any(expected_marker in d for d in result.disqualifying_states)


def test_xfailed_trace_publishes_nothing(tmp_path: Path) -> None:
    trace = _live_complete_trace(tmp_path)
    collector = TestPassReceiptCollector(nodeid="test_cold.py::test_xfail")
    collector.record_report(
        _Report(when="setup", outcome="passed")
    )
    report = _Report(when="call", outcome="skipped", wasxfail="expected fail")
    collector.record_report(report)
    collector.record_report(_Report(when="teardown", outcome="passed"))

    result, _completed, publication = finalize_cold_pass_publication(
        collector=collector,
        runtime_trace=trace,
        locator_cid=_cid("locator-xfail"),
        repository_forest_cid=_cid("forest"),
        static_trace_root_cid=_cid("static"),
        policy_cid=_cid("policy"),
    )
    assert result.admitted is False
    assert publication is None
    assert DISQUALIFIER_XFAIL in result.disqualifying_states or DISQUALIFIER_SKIP in result.disqualifying_states


@pytest.mark.parametrize(
    ("trace_factory", "expected"),
    [
        (lambda: _FakeIncompleteTrace(), DISQUALIFIER_INCOMPLETE_TRACE),
        (lambda: _FakeOverflowTrace(), DISQUALIFIER_OVERFLOWED_TRACE),
        (lambda: _FakeExceptionalTrace(), DISQUALIFIER_EXCEPTIONAL_TRACE),
        (lambda: _FakeUncontrolledTrace(), DISQUALIFIER_UNCONTROLLED_TRACE),
    ],
)
def test_bad_traces_publish_nothing_authoritative(
    trace_factory: Any,
    expected: str,
) -> None:
    trace = trace_factory()
    collector = _complete_collector()
    result, completed, publication = finalize_cold_pass_publication(
        collector=collector,
        runtime_trace=trace,
        locator_cid=_cid("locator-bad-trace"),
        repository_forest_cid=_cid("forest"),
        static_trace_root_cid=_cid("static"),
        policy_cid=_cid("policy"),
    )
    assert result.admitted is False
    assert completed is None
    assert publication is None
    assert expected in result.disqualifying_states or DISQUALIFIER_INCOMPLETE_TRACE in result.disqualifying_states


def test_evaluate_complete_pass_rejects_overflow_and_exceptional() -> None:
    phases = {
        "setup": PhaseOutcome.PASS,
        "call": PhaseOutcome.PASS,
        "teardown": PhaseOutcome.PASS,
    }
    eligible, disqualifiers = evaluate_complete_pass(
        phases,
        runtime_trace=_FakeOverflowTrace(),
        require_runtime_trace=True,
    )
    assert eligible is False
    assert DISQUALIFIER_OVERFLOWED_TRACE in disqualifiers or DISQUALIFIER_INCOMPLETE_TRACE in disqualifiers

    eligible, disqualifiers = evaluate_complete_pass(
        phases,
        runtime_trace=_FakeExceptionalTrace(),
        require_runtime_trace=True,
    )
    assert eligible is False
    assert (
        DISQUALIFIER_EXCEPTIONAL_TRACE in disqualifiers
        or DISQUALIFIER_INCOMPLETE_TRACE in disqualifiers
    )


def test_incomplete_phases_publish_nothing(tmp_path: Path) -> None:
    trace = _live_complete_trace(tmp_path)
    collector = TestPassReceiptCollector(nodeid="test_cold.py::test_partial")
    collector.record_report(_Report(when="setup", outcome="passed"))
    # Missing call + teardown.
    result, completed, publication = finalize_cold_pass_publication(
        collector=collector,
        runtime_trace=trace,
        locator_cid=_cid("locator-partial"),
    )
    assert result.admitted is False
    assert publication is None


def test_assemble_rejects_receipt_key_mismatch(tmp_path: Path) -> None:
    trace = _live_complete_trace(tmp_path)
    completed = build_completed_execution_identity(
        locator_cid=_cid("locator"),
        runtime_trace=trace,
        repository_forest_cid=_cid("forest"),
        static_trace_root_cid=_cid("static"),
        policy_cid=_cid("policy"),
    )
    assert completed is not None
    # Receipt bound to a different execution key.
    foreign_key = TestExecutionKey(
        locator_cid=completed.locator_cid,
        repository_forest_cid=_cid("other-forest"),
        runtime_trace_root_cid=completed.runtime_trace_root_cid,
        policy_cid=_cid("policy"),
    )
    receipt = TestPassReceipt(
        execution_key_cid=foreign_key.execution_key_id,
        locator_cid=completed.locator_cid,
        runtime_trace_root_cid=completed.runtime_trace_root_cid,
        admitted=True,
    )
    envelope = assemble_candidate_publication(
        completed_identity=completed,
        receipt=receipt,
    )
    assert envelope is None


def test_build_completed_identity_rejects_incomplete_trace() -> None:
    assert (
        build_completed_execution_identity(
            locator_cid=_cid("locator"),
            runtime_trace=_FakeIncompleteTrace(),
        )
        is None
    )


# ---------------------------------------------------------------------------
# Plugin composition smoke (write mode hooks attach lifecycle)
# ---------------------------------------------------------------------------


def test_plugin_composition_attaches_lifecycle_for_write_mode(tmp_path: Path) -> None:
    from ipfs_accelerate_py.testing.proof_reuse.config import ProofReuseConfig
    from ipfs_accelerate_py.testing.proof_reuse.plugin import (
        COMPOSITION_ATTRIBUTE,
        CONFIG_ATTRIBUTE,
        ProofReuseRuntimeComposition,
    )

    config = SimpleNamespace()
    setattr(config, CONFIG_ATTRIBUTE, ProofReuseConfig.resolve(environ={"IPFS_TEST_PROOF_REUSE_MODE": "write"}))
    setattr(config, "rootpath", tmp_path)
    composition = ProofReuseRuntimeComposition(config=config)
    setattr(config, COMPOSITION_ATTRIBUTE, composition)

    item = SimpleNamespace(nodeid="test_cold.py::test_example")
    item._ipfs_proof_reuse_locator = TestLocatorKey(  # type: ignore[attr-defined]
        repository_id="repository:example",
        package_identity="package:example",
        node_id="test_cold.py::test_example",
    )
    lifecycle = composition.attach_runtime_lifecycle(item)
    assert lifecycle is not None
    assert getattr(item, ITEM_RUNTIME_LIFECYCLE_ATTRIBUTE) is lifecycle

    # Start/stop around a synthetic single lifecycle.
    composition.start_runtime_lifecycle(item)
    lifecycle.note_phase("setup", "passed")
    lifecycle.note_phase("call", "passed")
    lifecycle.note_phase("teardown", "passed")
    trace = composition.stop_runtime_lifecycle(item)
    assert trace is not None
    assert getattr(item, "_ipfs_proof_reuse_runtime_trace", None) is trace


def test_finalize_receipt_requires_runtime_trace_when_requested() -> None:
    collector = _complete_collector()
    result = finalize_test_pass_receipt(
        collector,
        locator_cid=_cid("locator"),
        execution_key_cid=_cid("execution"),
        runtime_trace=None,
        require_runtime_trace=True,
        writes_receipts=False,
    )
    assert result.admitted is False
    assert DISQUALIFIER_INCOMPLETE_TRACE in result.disqualifying_states
