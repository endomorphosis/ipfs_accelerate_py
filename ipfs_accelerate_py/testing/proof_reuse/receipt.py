"""Complete-pass receipt capture for proof-backed pytest reuse (PTR-052).

Aggregates setup/call/teardown reports and eligible trace/context into a
canonical :class:`TestPassReceipt`.  Only a full three-phase pass with clear
disqualifying bits yields a reusable admitted receipt.  Storage and deferred
proving run after the outcome and never change pytest status.

This module does not import pytest at module import time.  The optional
``pytest_runtest_logreport`` hook is a duck-typed recorder; composition layers
register collectors and call :func:`finalize_test_pass_receipt` after teardown.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Optional, Tuple

from ...agent_supervisor.proof.test_execution_contracts import (
    PhaseOutcome,
    TestExecutionKey,
    TestPassReceipt,
)

PROOF_REUSE_RECEIPT_INTERFACE: Final = "ProofReuseReceiptCapture@1"
TEST_PASS_RECEIPT_COLLECTOR_INTERFACE: Final = "TestPassReceiptCollector@1"
DEFERRED_ISSUANCE_ENVELOPE_INTERFACE: Final = "DeferredIssuanceEnvelope@1"

ITEM_COLLECTOR_ATTRIBUTE: Final = "_ipfs_proof_reuse_receipt_collector"
ITEM_RECEIPT_RESULT_ATTRIBUTE: Final = "_ipfs_proof_reuse_receipt_result"
CONFIG_COLLECTORS_ATTRIBUTE: Final = "_ipfs_proof_reuse_receipt_collectors"

# Fields workers may publish for controller-side deferred reconstruction.
# Witness, private secrets, and raw statement bodies are intentionally absent.
_PUBLIC_DEFERRED_FIELDS: Final = frozenset(
    {
        "interface",
        "request_id",
        "receipt_cid",
        "execution_key_cid",
        "candidate_context_cid",
        "policy_cid",
        "statement_cid",
        "circuit_cid",
        "verifying_key_cid",
        "issuer_id",
        "epoch",
        "locator_cid",
        "backend_id",
        "proof_system_id",
        "content_profile",
        "statement_version",
        "statement_interface",
        "statement_digest",
        "retained_receipt_bytes_hex",
        "retained_candidate_context_bytes_hex",
    }
)
_PRIVATE_DEFERRED_MARKERS: Final = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "hidden_witness",
        "password",
        "private",
        "private_key",
        "private_premise",
        "private_witness",
        "refresh_token",
        "secret",
        "session_token",
        "witness",
        "witness_bytes",
        "witness_hex",
        "witness_json",
    }
)

DEFAULT_OUTCOME_POLICY_ID: Final = "pytest-complete-pass-v1"
PHASES: Final = ("setup", "call", "teardown")

# Closed disqualifier vocabulary used by eligibility policy and diagnostics.
DISQUALIFIER_SKIP: Final = "skip"
DISQUALIFIER_XFAIL: Final = "xfail"
DISQUALIFIER_XPASS: Final = "xpass"
DISQUALIFIER_RERUN: Final = "rerun"
DISQUALIFIER_INTERRUPTED: Final = "interrupted"
DISQUALIFIER_TIMEOUT: Final = "timeout"
DISQUALIFIER_TEARDOWN_FAILURE: Final = "teardown_failure"
DISQUALIFIER_INCOMPLETE_TRACE: Final = "incomplete_trace"
DISQUALIFIER_LEAKED_RESOURCES: Final = "leaked_resources"
DISQUALIFIER_FAIL: Final = "fail"
DISQUALIFIER_ERROR: Final = "error"
DISQUALIFIER_INCOMPLETE_PHASES: Final = "incomplete_phases"
DISQUALIFIER_NOT_RUN: Final = "not_run"
DISQUALIFIER_DISABLED: Final = "reuse_disabled"

_MAX_DIAGNOSTIC_CHARS: Final = 128
_MAX_DIAGNOSTIC_KEYS: Final = 16
_MAX_NODE_ID_CHARS: Final = 2048
_MAX_REASON_CHARS: Final = 512

_LEAK_REASON_MARKERS: Final = frozenset(
    {
        "leaked_resources",
        "leaked_resource",
        "resource_leak",
        "unclosed_resource",
        "unclosed_resources",
        "resource_not_released",
    }
)
_TIMEOUT_TEXT_MARKERS: Final = frozenset(
    {
        "timeout",
        "timed out",
        "timedout",
        "pytest-timeout",
        "failed: timeout",
    }
)
_RERUN_KEYWORD_MARKERS: Final = frozenset(
    {
        "rerun",
        "flaky",
        "rerunfailures",
    }
)

_REGISTRY_LOCK = threading.RLock()
_COLLECTORS: dict[str, "TestPassReceiptCollector"] = {}


def _bounded_text(value: Any, *, max_chars: int = _MAX_DIAGNOSTIC_CHARS) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            return ""
    text = value.strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _bounded_diagnostics(
    values: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not values:
        return {}
    result: dict[str, Any] = {}
    for index, (key, raw) in enumerate(values.items()):
        if index >= _MAX_DIAGNOSTIC_KEYS:
            break
        name = _bounded_text(key, max_chars=64)
        if not name:
            continue
        if isinstance(raw, bool) or raw is None:
            result[name] = raw
        elif isinstance(raw, int) and not isinstance(raw, bool):
            result[name] = raw
        elif isinstance(raw, float):
            # Public diagnostics reject non-finite floats by construction.
            if raw != raw or raw in (float("inf"), float("-inf")):
                continue
            result[name] = raw
        else:
            result[name] = _bounded_text(raw, max_chars=_MAX_DIAGNOSTIC_CHARS)
    return result


def _duration_ms(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return 0
    try:
        if isinstance(value, int):
            return max(0, value)
        seconds = float(value)
    except (TypeError, ValueError):
        return 0
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return 0
    if seconds < 0:
        return 0
    # Cap to a day of milliseconds to keep contracts bounded.
    return min(int(seconds * 1000.0), 86_400_000)


def _keywords_of(report: Any) -> Tuple[str, ...]:
    raw = getattr(report, "keywords", None)
    if raw is None:
        return ()
    names: list[str] = []
    try:
        if isinstance(raw, Mapping):
            iterable = raw.keys()
        else:
            iterable = raw
        for item in iterable:
            text = _bounded_text(item, max_chars=64).lower()
            if text and text not in names:
                names.append(text)
            if len(names) >= 64:
                break
    except Exception:
        return ()
    return tuple(names)


def _longrepr_text(report: Any) -> str:
    for attr in ("longreprtext", "longrepr"):
        value = getattr(report, attr, None)
        if value is None:
            continue
        return _bounded_text(value, max_chars=_MAX_REASON_CHARS).lower()
    return ""


def _report_indicates_timeout(report: Any) -> bool:
    keywords = _keywords_of(report)
    if any(marker in keywords for marker in _TIMEOUT_TEXT_MARKERS):
        return True
    if any("timeout" in keyword for keyword in keywords):
        return True
    text = _longrepr_text(report)
    if not text:
        return False
    return any(marker in text for marker in _TIMEOUT_TEXT_MARKERS)


def _report_indicates_rerun(report: Any) -> bool:
    keywords = _keywords_of(report)
    if any(marker in keywords for marker in _RERUN_KEYWORD_MARKERS):
        return True
    execution_count = getattr(report, "execution_count", None)
    if (
        isinstance(execution_count, int)
        and not isinstance(execution_count, bool)
        and execution_count > 1
    ):
        return True
    reruns = getattr(report, "rerun", None)
    if isinstance(reruns, int) and not isinstance(reruns, bool) and reruns > 0:
        return True
    return bool(getattr(report, "wasrerun", False))


def map_report_to_phase_outcome(report: Any) -> PhaseOutcome:
    """Map a duck-typed pytest TestReport into a closed :class:`PhaseOutcome`."""

    when = _bounded_text(getattr(report, "when", ""), max_chars=32).lower()
    if when not in PHASES:
        return PhaseOutcome.NOT_RUN

    outcome = _bounded_text(getattr(report, "outcome", ""), max_chars=32).lower()
    wasxfail = getattr(report, "wasxfail", None)
    has_xfail = wasxfail is not None and wasxfail is not False and wasxfail != ""

    if outcome in ("interrupted", "interrupt"):
        return PhaseOutcome.INTERRUPTED
    if _report_indicates_timeout(report):
        # Timeout is a disqualifier distinct from plain fail; still map phase
        # outcome to ERROR so admitted receipts cannot form.
        return PhaseOutcome.ERROR
    if has_xfail:
        if outcome == "passed":
            return PhaseOutcome.XPASS
        return PhaseOutcome.XFAIL
    if outcome in ("passed", "pass"):
        return PhaseOutcome.PASS
    if outcome in ("skipped", "skip"):
        return PhaseOutcome.SKIP
    if outcome in ("failed", "fail"):
        return PhaseOutcome.FAIL
    if outcome in ("error",):
        return PhaseOutcome.ERROR
    if outcome in ("rerun",):
        return PhaseOutcome.RERUN
    if outcome in ("not_run", "notset", ""):
        return PhaseOutcome.NOT_RUN
    return PhaseOutcome.ERROR


@dataclass(frozen=True)
class PhaseCapture:
    """One recorded setup/call/teardown report."""

    __test__ = False

    when: str
    outcome: PhaseOutcome
    duration_ms: int = 0
    wasxfail: str = ""
    keywords: Tuple[str, ...] = ()
    timeout: bool = False
    rerun: bool = False

    def __post_init__(self) -> None:
        when = _bounded_text(self.when, max_chars=32).lower()
        object.__setattr__(self, "when", when)
        if not isinstance(self.outcome, PhaseOutcome):
            object.__setattr__(
                self,
                "outcome",
                PhaseOutcome(str(self.outcome)),
            )
        object.__setattr__(self, "duration_ms", _duration_ms(self.duration_ms))
        object.__setattr__(
            self, "wasxfail", _bounded_text(self.wasxfail, max_chars=_MAX_REASON_CHARS)
        )
        keywords = tuple(
            _bounded_text(item, max_chars=64)
            for item in (self.keywords or ())
            if _bounded_text(item, max_chars=64)
        )[:64]
        object.__setattr__(self, "keywords", keywords)
        object.__setattr__(self, "timeout", bool(self.timeout))
        object.__setattr__(self, "rerun", bool(self.rerun))


@dataclass(frozen=True)
class ReceiptCaptureResult:
    """Post-outcome capture result; never represents a pytest failure."""

    __test__ = False

    reusable: bool = False
    admitted: bool = False
    receipt: Optional[TestPassReceipt] = None
    receipt_cid: str = ""
    disqualifying_states: Tuple[str, ...] = ()
    phase_outcomes: Mapping[str, str] = field(default_factory=dict)
    stored: bool = False
    store_reason: str = ""
    deferred_proving_status: str = "not_requested"
    deferred_proving_reason: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.reusable

    @property
    def interface(self) -> str:
        return PROOF_REUSE_RECEIPT_INTERFACE


def _sorted_unique(values: Sequence[str]) -> Tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        text = _bounded_text(raw, max_chars=64)
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(sorted(ordered))


def _trace_is_complete(runtime_trace: Any) -> bool | None:
    """Return True/False when known; None when no trace was supplied."""

    if runtime_trace is None:
        return None
    complete_attr = getattr(runtime_trace, "complete", None)
    if isinstance(complete_attr, bool):
        return complete_attr
    completeness = getattr(runtime_trace, "completeness", None)
    if completeness is not None:
        complete = getattr(completeness, "complete", None)
        if isinstance(complete, bool):
            return complete
        if isinstance(completeness, str):
            return completeness.lower() == "complete"
    if isinstance(runtime_trace, Mapping):
        if "complete" in runtime_trace and isinstance(runtime_trace["complete"], bool):
            return runtime_trace["complete"]
        nested = runtime_trace.get("completeness")
        if isinstance(nested, Mapping) and isinstance(nested.get("complete"), bool):
            return nested["complete"]
    return False


def _trace_root_cid(runtime_trace: Any) -> str:
    if runtime_trace is None:
        return ""
    for attr in ("trace_cid", "root_cid", "cid", "content_id"):
        value = getattr(runtime_trace, attr, None)
        if isinstance(value, str) and value:
            return _bounded_text(value, max_chars=256)
    if isinstance(runtime_trace, Mapping):
        for key in ("trace_cid", "root_cid", "cid", "content_id"):
            value = runtime_trace.get(key)
            if isinstance(value, str) and value:
                return _bounded_text(value, max_chars=256)
    return ""


def _trace_completeness_reasons(runtime_trace: Any) -> Tuple[str, ...]:
    if runtime_trace is None:
        return ()
    reasons = getattr(runtime_trace, "completeness_reasons", None)
    if reasons is None and isinstance(runtime_trace, Mapping):
        reasons = runtime_trace.get("completeness_reasons")
        nested = runtime_trace.get("completeness")
        if reasons is None and isinstance(nested, Mapping):
            reasons = nested.get("reasons")
    if reasons is None:
        completeness = getattr(runtime_trace, "completeness", None)
        reasons = getattr(completeness, "reasons", None) if completeness is not None else None
    if not reasons:
        return ()
    result: list[str] = []
    try:
        for item in reasons:
            text = _bounded_text(item, max_chars=64)
            if text:
                result.append(text)
            if len(result) >= 64:
                break
    except Exception:
        return ()
    return tuple(result)


def _trace_indicates_leaks(runtime_trace: Any) -> bool:
    reasons = {reason.lower() for reason in _trace_completeness_reasons(runtime_trace)}
    if reasons & _LEAK_REASON_MARKERS:
        return True
    for reason in reasons:
        if "leak" in reason or "unclosed" in reason:
            return True
    leaked = getattr(runtime_trace, "leaked_resources", None)
    if isinstance(leaked, bool):
        return leaked
    if isinstance(runtime_trace, Mapping) and isinstance(
        runtime_trace.get("leaked_resources"), bool
    ):
        return bool(runtime_trace.get("leaked_resources"))
    return False


def evaluate_complete_pass(
    phases: Mapping[str, PhaseCapture] | Mapping[str, PhaseOutcome] | None,
    *,
    runtime_trace: Any = None,
    interrupted: bool = False,
    timeout: bool = False,
    rerun: bool = False,
    leaked_resources: bool = False,
    disabled: bool = False,
    extra_disqualifiers: Sequence[str] = (),
    require_runtime_trace: bool = False,
) -> Tuple[bool, Tuple[str, ...]]:
    """Return ``(eligible, disqualifying_states)`` for one terminal outcome.

    Eligibility requires setup+call+teardown all ``pass`` and an empty
    disqualifier set.  Missing phases, skips, xfail/xpass, reruns, interrupts,
    timeouts, teardown failures, incomplete traces, and leaked resources all
    disqualify.
    """

    disqualifiers: list[str] = []
    if disabled:
        disqualifiers.append(DISQUALIFIER_DISABLED)
    if interrupted:
        disqualifiers.append(DISQUALIFIER_INTERRUPTED)
    if timeout:
        disqualifiers.append(DISQUALIFIER_TIMEOUT)
    if rerun:
        disqualifiers.append(DISQUALIFIER_RERUN)
    if leaked_resources:
        disqualifiers.append(DISQUALIFIER_LEAKED_RESOURCES)
    for extra in extra_disqualifiers:
        text = _bounded_text(extra, max_chars=64)
        if text:
            disqualifiers.append(text)

    normalized: dict[str, PhaseOutcome] = {}
    if phases:
        for name in PHASES:
            raw = phases.get(name)  # type: ignore[arg-type]
            if raw is None:
                continue
            if isinstance(raw, PhaseCapture):
                normalized[name] = raw.outcome
                if raw.timeout:
                    disqualifiers.append(DISQUALIFIER_TIMEOUT)
                if raw.rerun:
                    disqualifiers.append(DISQUALIFIER_RERUN)
                if raw.outcome is PhaseOutcome.XFAIL:
                    disqualifiers.append(DISQUALIFIER_XFAIL)
                elif raw.outcome is PhaseOutcome.XPASS:
                    disqualifiers.append(DISQUALIFIER_XPASS)
                elif raw.outcome is PhaseOutcome.SKIP:
                    disqualifiers.append(DISQUALIFIER_SKIP)
                elif raw.outcome is PhaseOutcome.INTERRUPTED:
                    disqualifiers.append(DISQUALIFIER_INTERRUPTED)
                elif raw.outcome is PhaseOutcome.RERUN:
                    disqualifiers.append(DISQUALIFIER_RERUN)
                elif raw.outcome is PhaseOutcome.FAIL:
                    disqualifiers.append(DISQUALIFIER_FAIL)
                elif raw.outcome is PhaseOutcome.ERROR:
                    disqualifiers.append(DISQUALIFIER_ERROR)
                elif raw.outcome is PhaseOutcome.NOT_RUN:
                    disqualifiers.append(DISQUALIFIER_NOT_RUN)
            elif isinstance(raw, PhaseOutcome):
                normalized[name] = raw
            else:
                try:
                    normalized[name] = PhaseOutcome(str(raw))
                except Exception:
                    normalized[name] = PhaseOutcome.ERROR
                    disqualifiers.append(DISQUALIFIER_ERROR)

    missing = [name for name in PHASES if name not in normalized]
    if missing:
        disqualifiers.append(DISQUALIFIER_INCOMPLETE_PHASES)

    teardown = normalized.get("teardown")
    if teardown is not None and teardown is not PhaseOutcome.PASS:
        disqualifiers.append(DISQUALIFIER_TEARDOWN_FAILURE)

    for name, outcome in normalized.items():
        if outcome is PhaseOutcome.SKIP:
            disqualifiers.append(DISQUALIFIER_SKIP)
        elif outcome is PhaseOutcome.XFAIL:
            disqualifiers.append(DISQUALIFIER_XFAIL)
        elif outcome is PhaseOutcome.XPASS:
            disqualifiers.append(DISQUALIFIER_XPASS)
        elif outcome is PhaseOutcome.INTERRUPTED:
            disqualifiers.append(DISQUALIFIER_INTERRUPTED)
        elif outcome is PhaseOutcome.RERUN:
            disqualifiers.append(DISQUALIFIER_RERUN)
        elif outcome is PhaseOutcome.FAIL and name != "teardown":
            disqualifiers.append(DISQUALIFIER_FAIL)
        elif outcome is PhaseOutcome.ERROR and name != "teardown":
            disqualifiers.append(DISQUALIFIER_ERROR)
        elif outcome is PhaseOutcome.NOT_RUN:
            disqualifiers.append(DISQUALIFIER_NOT_RUN)

    if require_runtime_trace and runtime_trace is None:
        disqualifiers.append(DISQUALIFIER_INCOMPLETE_TRACE)
    else:
        completeness = _trace_is_complete(runtime_trace)
        if completeness is False:
            disqualifiers.append(DISQUALIFIER_INCOMPLETE_TRACE)
        if _trace_indicates_leaks(runtime_trace):
            disqualifiers.append(DISQUALIFIER_LEAKED_RESOURCES)

    unique = _sorted_unique(disqualifiers)
    all_pass = (
        not missing
        and all(normalized.get(name) is PhaseOutcome.PASS for name in PHASES)
        and not unique
    )
    return all_pass, unique


class TestPassReceiptCollector:
    """Aggregates per-item phase reports until terminal finalization."""

    __test__ = False
    interface = TEST_PASS_RECEIPT_COLLECTOR_INTERFACE

    def __init__(self, nodeid: str = "") -> None:
        self.nodeid = _bounded_text(nodeid, max_chars=_MAX_NODE_ID_CHARS)
        self._phases: dict[str, PhaseCapture] = {}
        self._interrupted = False
        self._timeout = False
        self._rerun = False
        self._leaked_resources = False
        self._disabled = False
        self._extra_disqualifiers: list[str] = []
        self._finalized = False
        self._lock = threading.RLock()

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def phases(self) -> Mapping[str, PhaseCapture]:
        with self._lock:
            return dict(self._phases)

    def mark_interrupted(self) -> None:
        with self._lock:
            self._interrupted = True

    def mark_timeout(self) -> None:
        with self._lock:
            self._timeout = True

    def mark_rerun(self) -> None:
        with self._lock:
            self._rerun = True

    def mark_leaked_resources(self) -> None:
        with self._lock:
            self._leaked_resources = True

    def mark_disabled(self, reason: str = "") -> None:
        with self._lock:
            self._disabled = True
            text = _bounded_text(reason, max_chars=64)
            if text:
                self._extra_disqualifiers.append(text)

    def add_disqualifier(self, state: str) -> None:
        text = _bounded_text(state, max_chars=64)
        if not text:
            return
        with self._lock:
            self._extra_disqualifiers.append(text)

    def record_report(self, report: Any) -> Optional[PhaseCapture]:
        """Record one pytest phase report. Never raises into the runner."""

        try:
            when = _bounded_text(getattr(report, "when", ""), max_chars=32).lower()
            if when not in PHASES:
                return None
            outcome = map_report_to_phase_outcome(report)
            wasxfail = getattr(report, "wasxfail", None)
            capture = PhaseCapture(
                when=when,
                outcome=outcome,
                duration_ms=_duration_ms(getattr(report, "duration", 0)),
                wasxfail="" if wasxfail in (None, False) else _bounded_text(wasxfail),
                keywords=_keywords_of(report),
                timeout=_report_indicates_timeout(report),
                rerun=_report_indicates_rerun(report),
            )
            with self._lock:
                if self._finalized:
                    return capture
                # Later reports for the same phase win (rerun plugins re-emit).
                previous = self._phases.get(when)
                if previous is not None and previous.outcome is PhaseOutcome.PASS:
                    if capture.outcome is not PhaseOutcome.PASS:
                        # A previously passed phase re-reported as non-pass is a
                        # rerun/instability signal.
                        self._rerun = True
                if capture.timeout:
                    self._timeout = True
                if capture.rerun:
                    self._rerun = True
                if capture.outcome is PhaseOutcome.INTERRUPTED:
                    self._interrupted = True
                if capture.outcome is PhaseOutcome.RERUN:
                    self._rerun = True
                nodeid = _bounded_text(
                    getattr(report, "nodeid", ""), max_chars=_MAX_NODE_ID_CHARS
                )
                if nodeid and not self.nodeid:
                    self.nodeid = nodeid
                self._phases[when] = capture
            return capture
        except Exception:
            return None

    def record_phase(
        self,
        when: str,
        outcome: PhaseOutcome | str,
        *,
        duration_ms: int = 0,
        wasxfail: str = "",
        timeout: bool = False,
        rerun: bool = False,
    ) -> Optional[PhaseCapture]:
        """Direct phase injection for unit tests and non-pytest callers."""

        class _Report:
            pass

        report = _Report()
        report.when = when  # type: ignore[attr-defined]
        report.outcome = (
            outcome.value if isinstance(outcome, PhaseOutcome) else str(outcome)
        )  # type: ignore[attr-defined]
        report.duration = float(duration_ms) / 1000.0  # type: ignore[attr-defined]
        if wasxfail:
            report.wasxfail = wasxfail  # type: ignore[attr-defined]
        if timeout:
            report.keywords = {"timeout": 1}  # type: ignore[attr-defined]
        if rerun:
            report.execution_count = 2  # type: ignore[attr-defined]
            report.keywords = getattr(report, "keywords", {})  # type: ignore[attr-defined]
            if isinstance(report.keywords, dict):  # type: ignore[attr-defined]
                report.keywords = {**report.keywords, "rerun": 1}  # type: ignore[attr-defined]
        return self.record_report(report)

    def evaluate(
        self,
        *,
        runtime_trace: Any = None,
        require_runtime_trace: bool = False,
    ) -> Tuple[bool, Tuple[str, ...]]:
        with self._lock:
            phases = dict(self._phases)
            interrupted = self._interrupted
            timeout = self._timeout
            rerun = self._rerun
            leaked = self._leaked_resources
            disabled = self._disabled
            extra = list(self._extra_disqualifiers)
        return evaluate_complete_pass(
            phases,
            runtime_trace=runtime_trace,
            interrupted=interrupted,
            timeout=timeout,
            rerun=rerun,
            leaked_resources=leaked,
            disabled=disabled,
            extra_disqualifiers=extra,
            require_runtime_trace=require_runtime_trace,
        )

    def phase_outcomes(self) -> dict[str, str]:
        with self._lock:
            return {
                name: capture.outcome.value for name, capture in self._phases.items()
            }

    def phase_durations_ms(self) -> dict[str, int]:
        with self._lock:
            return {name: capture.duration_ms for name, capture in self._phases.items()}

    def mark_finalized(self) -> None:
        with self._lock:
            self._finalized = True


def register_collector(
    collector: TestPassReceiptCollector,
    *,
    nodeid: str | None = None,
) -> TestPassReceiptCollector:
    """Register a collector for hook-driven phase recording."""

    key = _bounded_text(nodeid if nodeid is not None else collector.nodeid, max_chars=_MAX_NODE_ID_CHARS)
    if not key:
        raise ValueError("collector nodeid is required for registration")
    if not collector.nodeid:
        collector.nodeid = key
    with _REGISTRY_LOCK:
        _COLLECTORS[key] = collector
    return collector


def get_collector(nodeid: str) -> Optional[TestPassReceiptCollector]:
    key = _bounded_text(nodeid, max_chars=_MAX_NODE_ID_CHARS)
    if not key:
        return None
    with _REGISTRY_LOCK:
        return _COLLECTORS.get(key)


def get_or_create_collector(nodeid: str) -> TestPassReceiptCollector:
    key = _bounded_text(nodeid, max_chars=_MAX_NODE_ID_CHARS)
    if not key:
        raise ValueError("nodeid is required")
    with _REGISTRY_LOCK:
        existing = _COLLECTORS.get(key)
        if existing is not None:
            return existing
        collector = TestPassReceiptCollector(nodeid=key)
        _COLLECTORS[key] = collector
        return collector


def attach_collector(item: Any) -> TestPassReceiptCollector:
    """Attach (or return) a collector on a pytest item."""

    existing = getattr(item, ITEM_COLLECTOR_ATTRIBUTE, None)
    if isinstance(existing, TestPassReceiptCollector):
        if existing.nodeid:
            register_collector(existing)
        return existing
    nodeid = _bounded_text(getattr(item, "nodeid", ""), max_chars=_MAX_NODE_ID_CHARS)
    collector = get_or_create_collector(nodeid) if nodeid else TestPassReceiptCollector()
    try:
        setattr(item, ITEM_COLLECTOR_ATTRIBUTE, collector)
    except Exception:
        pass
    if collector.nodeid:
        register_collector(collector)
    return collector


def clear_collectors() -> None:
    with _REGISTRY_LOCK:
        _COLLECTORS.clear()


def pytest_runtest_logreport(report: Any) -> None:
    """Pytest hook: record setup/call/teardown without affecting outcomes.

    Collectors must be registered (via :func:`attach_collector` /
    :func:`register_collector`) for the report's nodeid.  Failures here are
    swallowed so receipt capture can never change the test result.
    """

    try:
        nodeid = _bounded_text(getattr(report, "nodeid", ""), max_chars=_MAX_NODE_ID_CHARS)
        if not nodeid:
            return
        collector = get_collector(nodeid)
        if collector is None:
            return
        collector.record_report(report)
    except Exception:
        return


def _cid_from(value: Any, *, attr_names: Sequence[str]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _bounded_text(value, max_chars=256)
    for name in attr_names:
        attr = getattr(value, name, None)
        if isinstance(attr, str) and attr:
            return _bounded_text(attr, max_chars=256)
    if isinstance(value, Mapping):
        for name in attr_names:
            attr = value.get(name)
            if isinstance(attr, str) and attr:
                return _bounded_text(attr, max_chars=256)
    return ""


def _store_receipt(store: Any, receipt: TestPassReceipt) -> tuple[bool, str, str]:
    """Persist receipt bytes. Returns ``(stored, cid, reason)``."""

    if store is None:
        return False, "", "store_not_configured"
    try:
        put = getattr(store, "put_receipt", None)
        if not callable(put):
            put = getattr(store, "put", None)
        if not callable(put):
            return False, "", "store_unsupported"
        result = put(receipt)
    except Exception as exc:
        return (
            False,
            "",
            f"store_error:{type(exc).__name__}"[:_MAX_DIAGNOSTIC_CHARS],
        )

    stored = False
    cid = ""
    reason = "store_rejected"
    try:
        if result is True:
            return True, receipt.receipt_id, "ok"
        if result is False or result is None:
            return False, "", "store_rejected"
        if isinstance(result, Mapping):
            stored = bool(result.get("stored", result.get("ok", False)))
            cid = _bounded_text(result.get("cid") or result.get("receipt_cid") or "")
            reason = _bounded_text(
                result.get("reason_code") or result.get("reason") or ("ok" if stored else "store_rejected")
            )
        else:
            stored_attr = getattr(result, "stored", None)
            if stored_attr is None:
                stored_attr = getattr(result, "ok", None)
            if isinstance(stored_attr, bool):
                stored = stored_attr
            cid = _bounded_text(
                getattr(result, "cid", None)
                or getattr(result, "receipt_cid", None)
                or ""
            )
            reason_attr = getattr(result, "reason_code", None)
            if reason_attr is None:
                reason_attr = getattr(result, "reason", None)
            if reason_attr is not None:
                reason = _bounded_text(
                    getattr(reason_attr, "value", reason_attr) or ("ok" if stored else "store_rejected")
                )
            elif stored:
                reason = "ok"
        if stored and not cid:
            cid = receipt.receipt_id
        if stored and cid and cid != receipt.receipt_id:
            # Identity mismatch: treat as non-stored for reuse safety.
            return False, cid, "store_cid_mismatch"
        return stored, cid if stored else cid, reason if reason else ("ok" if stored else "store_rejected")
    except Exception as exc:
        return (
            False,
            "",
            f"store_result_error:{type(exc).__name__}"[:_MAX_DIAGNOSTIC_CHARS],
        )


def _request_deferred_proving(
    issuer: Any,
    request: Any,
    *,
    receipt: TestPassReceipt,
) -> tuple[str, str]:
    """Invoke optional deferred issuer. Never raises; never affects pytest."""

    if issuer is None:
        return "not_requested", ""
    if request is None:
        # Allow issuer duck-types that accept the receipt alone.
        request = receipt
    try:
        method = None
        for name in (
            "issue_deferred",
            "issue",
            "schedule",
            "prove_deferred",
            "request_certificate",
        ):
            candidate = getattr(issuer, name, None)
            if callable(candidate):
                method = candidate
                break
        if method is None and callable(issuer):
            method = issuer
        if method is None:
            return "error", "issuer_unsupported"
        result = method(request)
    except Exception as exc:
        return "error", f"prover_error:{type(exc).__name__}"[:_MAX_DIAGNOSTIC_CHARS]

    try:
        if result is None:
            return "deferred", "certificate_deferred"
        status = getattr(result, "status", None)
        reason = getattr(result, "reason", None)
        if status is not None:
            status_text = _bounded_text(getattr(status, "value", status), max_chars=64)
            reason_text = _bounded_text(
                getattr(reason, "value", reason) if reason is not None else "",
                max_chars=64,
            )
            if "issued" in status_text:
                return "issued", reason_text or "issued"
            if "reject" in status_text:
                return "rejected", reason_text or "rejected"
            if "defer" in status_text:
                return "deferred", reason_text or "certificate_deferred"
            return status_text or "deferred", reason_text
        if isinstance(result, Mapping):
            status_text = _bounded_text(result.get("status"), max_chars=64)
            reason_text = _bounded_text(result.get("reason"), max_chars=64)
            if status_text:
                return status_text, reason_text
        if result is True:
            return "issued", "issued"
        if result is False:
            return "deferred", "certificate_deferred"
        return "deferred", "certificate_deferred"
    except Exception as exc:
        return "error", f"prover_result_error:{type(exc).__name__}"[:_MAX_DIAGNOSTIC_CHARS]


def finalize_test_pass_receipt(
    collector: TestPassReceiptCollector | Mapping[str, Any] | None = None,
    *,
    locator: Any = None,
    execution_key: Any = None,
    locator_cid: str = "",
    execution_key_cid: str = "",
    runtime_trace: Any = None,
    static_trace_root_cid: str = "",
    runtime_trace_root_cid: str = "",
    completeness_receipt_cid: str = "",
    dependency_forest_cid: str = "",
    capability_root_cid: str = "",
    policy_cid: str = "",
    outcome_policy_id: str = DEFAULT_OUTCOME_POLICY_ID,
    runner_identity: str = "",
    trust_domain: str = "",
    issuer_key_id: str = "",
    nonce: str = "",
    epoch_policy_id: str = "",
    schema_cid: str = "",
    store: Any = None,
    issuer: Any = None,
    deferred_request: Any = None,
    writes_receipts: bool = True,
    require_runtime_trace: bool = False,
    metadata: Mapping[str, Any] | None = None,
    item: Any = None,
) -> ReceiptCaptureResult:
    """Build and optionally store a reusable receipt after terminal teardown.

    Store and prover failures are recorded in the result and never re-raised, so
    they cannot change the pytest pass/fail outcome of the captured test.
    """

    diagnostics: dict[str, Any] = {}
    try:
        active = collector
        if active is None and item is not None:
            attached = getattr(item, ITEM_COLLECTOR_ATTRIBUTE, None)
            if isinstance(attached, TestPassReceiptCollector):
                active = attached
            else:
                nodeid = _bounded_text(
                    getattr(item, "nodeid", ""), max_chars=_MAX_NODE_ID_CHARS
                )
                if nodeid:
                    active = get_collector(nodeid)

        if isinstance(active, TestPassReceiptCollector):
            eligible, disqualifiers = active.evaluate(
                runtime_trace=runtime_trace,
                require_runtime_trace=require_runtime_trace,
            )
            phase_outcomes = active.phase_outcomes()
            durations = active.phase_durations_ms()
        elif isinstance(active, Mapping):
            # Allow a plain phase mapping for hermetic unit tests.
            rebuilt: dict[str, PhaseCapture] = {}
            for name in PHASES:
                raw = active.get(name)
                if raw is None:
                    continue
                if isinstance(raw, PhaseCapture):
                    rebuilt[name] = raw
                else:
                    rebuilt[name] = PhaseCapture(
                        when=name,
                        outcome=(
                            raw
                            if isinstance(raw, PhaseOutcome)
                            else PhaseOutcome(str(raw))
                        ),
                    )
            eligible, disqualifiers = evaluate_complete_pass(
                rebuilt,
                runtime_trace=runtime_trace,
                require_runtime_trace=require_runtime_trace,
            )
            phase_outcomes = {k: v.outcome.value for k, v in rebuilt.items()}
            durations = {k: v.duration_ms for k, v in rebuilt.items()}
        else:
            return ReceiptCaptureResult(
                reusable=False,
                admitted=False,
                disqualifying_states=(DISQUALIFIER_INCOMPLETE_PHASES,),
                diagnostics={"stage": "finalize", "reason": "missing_collector"},
            )

        if not eligible:
            if isinstance(active, TestPassReceiptCollector):
                active.mark_finalized()
            result = ReceiptCaptureResult(
                reusable=False,
                admitted=False,
                disqualifying_states=disqualifiers,
                phase_outcomes=phase_outcomes,
                store_reason="not_eligible",
                deferred_proving_status="not_requested",
                diagnostics=_bounded_diagnostics(
                    {
                        "stage": "eligibility",
                        "disqualifier_count": len(disqualifiers),
                    }
                ),
            )
            if item is not None:
                try:
                    setattr(item, ITEM_RECEIPT_RESULT_ATTRIBUTE, result)
                except Exception:
                    pass
            return result

        resolved_locator_cid = locator_cid or _cid_from(
            locator, attr_names=("locator_id", "content_id", "cid")
        )
        resolved_execution_key_cid = execution_key_cid or _cid_from(
            execution_key,
            attr_names=("execution_key_id", "content_id", "cid"),
        )
        if not resolved_locator_cid or not resolved_execution_key_cid:
            if isinstance(active, TestPassReceiptCollector):
                active.mark_finalized()
            return ReceiptCaptureResult(
                reusable=False,
                admitted=False,
                disqualifying_states=disqualifiers,
                phase_outcomes=phase_outcomes,
                store_reason="missing_identity",
                diagnostics=_bounded_diagnostics(
                    {
                        "stage": "identity",
                        "has_locator": bool(resolved_locator_cid),
                        "has_execution_key": bool(resolved_execution_key_cid),
                    }
                ),
            )

        if isinstance(execution_key, TestExecutionKey):
            static_cid = static_trace_root_cid or execution_key.static_trace_root_cid
            runtime_cid = (
                runtime_trace_root_cid
                or _trace_root_cid(runtime_trace)
                or execution_key.runtime_trace_root_cid
            )
            forest_cid = dependency_forest_cid or execution_key.repository_forest_cid
            resolved_policy = policy_cid or execution_key.policy_cid
        else:
            static_cid = static_trace_root_cid
            runtime_cid = runtime_trace_root_cid or _trace_root_cid(runtime_trace)
            forest_cid = dependency_forest_cid
            resolved_policy = policy_cid

        completeness_cid = completeness_receipt_cid
        if not completeness_cid:
            # A complete runtime trace is itself the completeness receipt when
            # the caller did not supply a separate completeness CID.
            if _trace_is_complete(runtime_trace) is True:
                completeness_cid = _trace_root_cid(runtime_trace)

        try:
            receipt = TestPassReceipt(
                execution_key_cid=resolved_execution_key_cid,
                locator_cid=resolved_locator_cid,
                setup_outcome=PhaseOutcome.PASS,
                call_outcome=PhaseOutcome.PASS,
                teardown_outcome=PhaseOutcome.PASS,
                setup_duration_ms=int(durations.get("setup", 0)),
                call_duration_ms=int(durations.get("call", 0)),
                teardown_duration_ms=int(durations.get("teardown", 0)),
                outcome_policy_id=outcome_policy_id or DEFAULT_OUTCOME_POLICY_ID,
                disqualifying_states=(),
                static_trace_root_cid=static_cid,
                runtime_trace_root_cid=runtime_cid,
                completeness_receipt_cid=completeness_cid,
                runner_identity=runner_identity,
                trust_domain=trust_domain,
                issuer_key_id=issuer_key_id,
                nonce=nonce,
                epoch_policy_id=epoch_policy_id,
                dependency_forest_cid=forest_cid,
                capability_root_cid=capability_root_cid,
                schema_cid=schema_cid,
                policy_cid=resolved_policy,
                admitted=True,
                metadata=dict(metadata or {}),
            )
        except Exception as exc:
            if isinstance(active, TestPassReceiptCollector):
                active.mark_finalized()
            return ReceiptCaptureResult(
                reusable=False,
                admitted=False,
                disqualifying_states=disqualifiers,
                phase_outcomes=phase_outcomes,
                store_reason="receipt_build_error",
                diagnostics=_bounded_diagnostics(
                    {
                        "stage": "build_receipt",
                        "error_type": type(exc).__name__,
                    }
                ),
            )

        stored = False
        store_reason = "write_disabled"
        stored_cid = ""
        if writes_receipts:
            stored, stored_cid, store_reason = _store_receipt(store, receipt)
            diagnostics["store_attempted"] = True
        else:
            diagnostics["store_attempted"] = False

        deferred_status = "not_requested"
        deferred_reason = ""
        # Proving is only attempted after an admitted receipt; store failure
        # still must not raise, and may skip issuance when nothing was retained.
        if issuer is not None:
            if writes_receipts and not stored:
                deferred_status = "deferred"
                deferred_reason = "receipt_store_rejected"
            else:
                deferred_status, deferred_reason = _request_deferred_proving(
                    issuer,
                    deferred_request,
                    receipt=receipt,
                )

        if isinstance(active, TestPassReceiptCollector):
            active.mark_finalized()

        result = ReceiptCaptureResult(
            reusable=True,
            admitted=True,
            receipt=receipt,
            receipt_cid=receipt.receipt_id,
            disqualifying_states=(),
            phase_outcomes=phase_outcomes,
            stored=stored,
            store_reason=store_reason,
            deferred_proving_status=deferred_status,
            deferred_proving_reason=deferred_reason,
            diagnostics=_bounded_diagnostics(
                {
                    **diagnostics,
                    "stage": "complete_pass",
                    "stored_cid": stored_cid,
                    "nodeid": (
                        active.nodeid
                        if isinstance(active, TestPassReceiptCollector)
                        else ""
                    ),
                }
            ),
        )
        if item is not None:
            try:
                setattr(item, ITEM_RECEIPT_RESULT_ATTRIBUTE, result)
            except Exception:
                pass
        return result
    except Exception as exc:
        # Absolute fail-open for the capture path: never change the test result.
        return ReceiptCaptureResult(
            reusable=False,
            admitted=False,
            store_reason="finalize_error",
            diagnostics=_bounded_diagnostics(
                {
                    "stage": "finalize",
                    "error_type": type(exc).__name__,
                }
            ),
        )


def capture_complete_pass_from_reports(
    reports: Sequence[Any],
    **finalize_kwargs: Any,
) -> ReceiptCaptureResult:
    """Convenience helper: record reports then finalize in one call."""

    collector = TestPassReceiptCollector(
        nodeid=_bounded_text(
            getattr(reports[0], "nodeid", "") if reports else "",
            max_chars=_MAX_NODE_ID_CHARS,
        )
    )
    for report in reports:
        collector.record_report(report)
    return finalize_test_pass_receipt(collector, **finalize_kwargs)


def _field_is_private(name: str) -> bool:
    lowered = name.strip().lower()
    if not lowered:
        return True
    if lowered in _PRIVATE_DEFERRED_MARKERS:
        return True
    return any(marker in lowered for marker in _PRIVATE_DEFERRED_MARKERS)


def public_deferred_mapping(value: Any) -> dict[str, Any] | None:
    """Return only public deferred fields; reject private/witness keys."""

    if value is None:
        return None
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            value = value.to_dict()
        except Exception:
            return None
    if not isinstance(value, Mapping):
        return None
    public: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if _field_is_private(key):
            continue
        if key not in _PUBLIC_DEFERRED_FIELDS:
            continue
        if isinstance(raw_value, (str, int, float, bool)) or raw_value is None:
            public[key] = raw_value
        elif isinstance(raw_value, Mapping):
            # Nested maps are not accepted on the public envelope.
            continue
        else:
            text = _bounded_text(raw_value, max_chars=_MAX_REASON_CHARS)
            if text:
                public[key] = text
    if "interface" not in public:
        public["interface"] = DEFERRED_ISSUANCE_ENVELOPE_INTERFACE
    return public


@dataclass(frozen=True, slots=True)
class DeferredIssuanceEnvelope:
    """Public-only deferred issuance transport for workers and controllers.

    Workers may serialize this envelope.  Controllers reconstruct typed
    issuance requests from retained public receipt/candidate bytes instead of
    trusting worker-supplied private material.
    """

    receipt_cid: str
    execution_key_cid: str = ""
    candidate_context_cid: str = ""
    policy_cid: str = ""
    statement_cid: str = ""
    circuit_cid: str = ""
    verifying_key_cid: str = ""
    issuer_id: str = ""
    epoch: str = ""
    locator_cid: str = ""
    backend_id: str = ""
    proof_system_id: str = ""
    retained_receipt_bytes_hex: str = ""
    retained_candidate_context_bytes_hex: str = ""
    statement_digest: str = ""
    content_profile: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def interface(self) -> str:
        return DEFERRED_ISSUANCE_ENVELOPE_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "interface": self.interface,
            "receipt_cid": self.receipt_cid,
            "execution_key_cid": self.execution_key_cid,
            "candidate_context_cid": self.candidate_context_cid,
            "policy_cid": self.policy_cid,
            "statement_cid": self.statement_cid,
            "circuit_cid": self.circuit_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "issuer_id": self.issuer_id,
            "epoch": self.epoch,
            "locator_cid": self.locator_cid,
            "backend_id": self.backend_id,
            "proof_system_id": self.proof_system_id,
            "retained_receipt_bytes_hex": self.retained_receipt_bytes_hex,
            "retained_candidate_context_bytes_hex": (
                self.retained_candidate_context_bytes_hex
            ),
            "statement_digest": self.statement_digest,
            "content_profile": self.content_profile,
        }
        return {key: value for key, value in payload.items() if value not in ("", None)}

    @classmethod
    def from_mapping(cls, value: Any) -> "DeferredIssuanceEnvelope | None":
        public = public_deferred_mapping(value)
        if public is None:
            return None
        receipt_cid = str(public.get("receipt_cid") or "")
        if not receipt_cid:
            return None
        return cls(
            receipt_cid=receipt_cid,
            execution_key_cid=str(public.get("execution_key_cid") or ""),
            candidate_context_cid=str(public.get("candidate_context_cid") or ""),
            policy_cid=str(public.get("policy_cid") or ""),
            statement_cid=str(public.get("statement_cid") or ""),
            circuit_cid=str(public.get("circuit_cid") or ""),
            verifying_key_cid=str(public.get("verifying_key_cid") or ""),
            issuer_id=str(public.get("issuer_id") or ""),
            epoch=str(public.get("epoch") or ""),
            locator_cid=str(public.get("locator_cid") or ""),
            backend_id=str(public.get("backend_id") or ""),
            proof_system_id=str(public.get("proof_system_id") or ""),
            retained_receipt_bytes_hex=str(
                public.get("retained_receipt_bytes_hex") or ""
            ),
            retained_candidate_context_bytes_hex=str(
                public.get("retained_candidate_context_bytes_hex") or ""
            ),
            statement_digest=str(public.get("statement_digest") or ""),
            content_profile=str(public.get("content_profile") or ""),
        )

    @classmethod
    def from_admitted_receipt(
        cls,
        receipt: Any,
        *,
        locator_cid: str = "",
        candidate_context_cid: str = "",
        backend_id: str = "",
        proof_system_id: str = "",
        retained_receipt_bytes: bytes | bytearray | None = None,
        retained_candidate_context_bytes: bytes | bytearray | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> "DeferredIssuanceEnvelope | None":
        """Build a public envelope from an admitted pass receipt only."""

        if receipt is None:
            return None
        try:
            receipt_cid = str(
                getattr(receipt, "receipt_id", None)
                or getattr(receipt, "receipt_cid", None)
                or ""
            )
            if not receipt_cid and isinstance(receipt, Mapping):
                receipt_cid = str(
                    receipt.get("receipt_id") or receipt.get("receipt_cid") or ""
                )
            if not receipt_cid:
                return None
            if hasattr(receipt, "to_dict") and callable(receipt.to_dict):
                payload = receipt.to_dict()
            elif isinstance(receipt, Mapping):
                payload = dict(receipt)
            else:
                payload = {}
            admitted = getattr(receipt, "admitted", payload.get("admitted"))
            if admitted is False:
                return None
            execution_key_cid = str(
                getattr(receipt, "execution_key_cid", None)
                or payload.get("execution_key_cid")
                or ""
            )
            policy_cid = str(
                getattr(receipt, "policy_cid", None) or payload.get("policy_cid") or ""
            )
            resolved_locator = locator_cid or str(
                getattr(receipt, "locator_cid", None)
                or payload.get("locator_cid")
                or ""
            )
            retained_receipt_hex = ""
            if isinstance(retained_receipt_bytes, (bytes, bytearray)):
                retained_receipt_hex = bytes(retained_receipt_bytes).hex()
            retained_candidate_hex = ""
            if isinstance(retained_candidate_context_bytes, (bytes, bytearray)):
                retained_candidate_hex = bytes(retained_candidate_context_bytes).hex()
            extra = dict(extras or {})
            return cls(
                receipt_cid=receipt_cid,
                execution_key_cid=execution_key_cid,
                candidate_context_cid=candidate_context_cid
                or str(extra.get("candidate_context_cid") or ""),
                policy_cid=policy_cid,
                statement_cid=str(extra.get("statement_cid") or ""),
                circuit_cid=str(extra.get("circuit_cid") or ""),
                verifying_key_cid=str(extra.get("verifying_key_cid") or ""),
                issuer_id=str(
                    getattr(receipt, "issuer_key_id", None)
                    or extra.get("issuer_id")
                    or ""
                ),
                epoch=str(
                    getattr(receipt, "epoch_policy_id", None) or extra.get("epoch") or ""
                ),
                locator_cid=resolved_locator,
                backend_id=backend_id or str(extra.get("backend_id") or ""),
                proof_system_id=proof_system_id
                or str(extra.get("proof_system_id") or ""),
                retained_receipt_bytes_hex=retained_receipt_hex,
                retained_candidate_context_bytes_hex=retained_candidate_hex,
                statement_digest=str(extra.get("statement_digest") or ""),
                content_profile=str(extra.get("content_profile") or ""),
            )
        except Exception:
            return None


def reconstruct_deferred_request_from_public(
    envelope: Any,
    *,
    retained_receipt_bytes: bytes | bytearray | None = None,
    retained_candidate_context_bytes: bytes | bytearray | None = None,
) -> dict[str, Any] | None:
    """Controller-side reconstruction of a public deferred request.

    Workers are never trusted for private witness data.  The controller rebuilds
    the issuance envelope from public retained bytes plus identity bindings.
    """

    try:
        if isinstance(envelope, DeferredIssuanceEnvelope):
            public = envelope.to_dict()
        else:
            public = public_deferred_mapping(envelope)
        if not public:
            return None
        if retained_receipt_bytes is not None:
            public["retained_receipt_bytes_hex"] = bytes(
                retained_receipt_bytes
            ).hex()
        if retained_candidate_context_bytes is not None:
            public["retained_candidate_context_bytes_hex"] = bytes(
                retained_candidate_context_bytes
            ).hex()
        # Prefer datasets typed reconstruction when available; fall back to the
        # public envelope itself so issuance remains deferred rather than lost.
        try:
            from ipfs_datasets_py.logic.zkp.test_certificate_issuer import (
                DeferredTestCertificateRequest,
            )

            typed = DeferredTestCertificateRequest.from_public_mapping(public)
            return typed.to_dict()
        except Exception:
            public.setdefault("interface", DEFERRED_ISSUANCE_ENVELOPE_INTERFACE)
            return public
    except Exception:
        return None


__all__ = [
    "CONFIG_COLLECTORS_ATTRIBUTE",
    "DEFAULT_OUTCOME_POLICY_ID",
    "DEFERRED_ISSUANCE_ENVELOPE_INTERFACE",
    "DISQUALIFIER_DISABLED",
    "DISQUALIFIER_ERROR",
    "DISQUALIFIER_FAIL",
    "DISQUALIFIER_INCOMPLETE_PHASES",
    "DISQUALIFIER_INCOMPLETE_TRACE",
    "DISQUALIFIER_INTERRUPTED",
    "DISQUALIFIER_LEAKED_RESOURCES",
    "DISQUALIFIER_NOT_RUN",
    "DISQUALIFIER_RERUN",
    "DISQUALIFIER_SKIP",
    "DISQUALIFIER_TEARDOWN_FAILURE",
    "DISQUALIFIER_TIMEOUT",
    "DISQUALIFIER_XFAIL",
    "DISQUALIFIER_XPASS",
    "DeferredIssuanceEnvelope",
    "ITEM_COLLECTOR_ATTRIBUTE",
    "ITEM_RECEIPT_RESULT_ATTRIBUTE",
    "PHASES",
    "PROOF_REUSE_RECEIPT_INTERFACE",
    "PhaseCapture",
    "ReceiptCaptureResult",
    "TEST_PASS_RECEIPT_COLLECTOR_INTERFACE",
    "TestPassReceiptCollector",
    "attach_collector",
    "capture_complete_pass_from_reports",
    "clear_collectors",
    "evaluate_complete_pass",
    "finalize_test_pass_receipt",
    "get_collector",
    "get_or_create_collector",
    "map_report_to_phase_outcome",
    "public_deferred_mapping",
    "pytest_runtest_logreport",
    "reconstruct_deferred_request_from_public",
    "register_collector",
]
