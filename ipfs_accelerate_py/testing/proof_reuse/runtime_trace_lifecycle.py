"""Cold pytest lifecycle binding for ``RuntimeTestDependencyTracer@1`` (PTR-146).

``PytestRuntimeTraceLifecycle`` starts the production tracer immediately before
setup and stops it only after teardown.  It never re-invokes the test body:
observation is a side channel around the single ordinary pytest protocol
execution.

Authority doctrine (fail-closed for publication, fail-open for pytest):

* setup, call, and teardown must each pass exactly once before a complete
  observed trace may contribute to a receipt or candidate;
* incomplete, uncontrolled, overflowed, or exceptional traces retain no
  publication authority;
* tracing faults are swallowed and recorded as diagnostics — they never alter
  pytest's real outcome.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final, Optional

from ...agent_supervisor.proof.test_execution_contracts import PhaseOutcome


PYTEST_RUNTIME_TRACE_LIFECYCLE_INTERFACE: Final = "PytestRuntimeTraceLifecycle@1"
PYTEST_RUNTIME_TRACE_LIFECYCLE_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/pytest-runtime-trace-lifecycle@1"
)

ITEM_RUNTIME_LIFECYCLE_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_runtime_trace_lifecycle"
)

PHASES: Final = ("setup", "call", "teardown")

# Completeness-reason markers that permanently disqualify publication authority.
_NON_AUTHORITATIVE_REASON_MARKERS: Final = frozenset(
    {
        "overflow",
        "instrumentation_failure",
        "unsupported_event",
        "private_event",
        "concurrent_trace",
        "not_started",
        "invalid_lifecycle",
        "uncontrolled",
        "exceptional",
        "exception",
    }
)


class LifecyclePhase(str, Enum):
    """Ordered phases of the single cold pytest protocol execution."""

    IDLE = "idle"
    TRACING = "tracing"
    SETUP = "setup"
    CALL = "call"
    TEARDOWN = "teardown"
    STOPPED = "stopped"
    FAILED = "failed"

    __test__ = False


class LifecycleAuthority(str, Enum):
    """Whether the observed lifecycle may contribute to publication."""

    NONE = "none"
    COMPLETE_PASS = "complete_pass"

    __test__ = False


def _bounded_text(value: Any, *, max_chars: int = 128) -> str:
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


def _phase_outcome_from_report(report: Any) -> PhaseOutcome:
    """Map a duck-typed pytest report onto :class:`PhaseOutcome`."""

    try:
        from .receipt import map_report_to_phase_outcome

        return map_report_to_phase_outcome(report)
    except Exception:
        outcome = str(getattr(report, "outcome", "") or "").lower()
        if outcome == "passed":
            return PhaseOutcome.PASS
        if outcome == "skipped":
            return PhaseOutcome.SKIP
        if outcome in {"failed", "fail"}:
            return PhaseOutcome.FAIL
        if outcome in {"error", "errored"}:
            return PhaseOutcome.ERROR
        return PhaseOutcome.ERROR


def _trace_reasons(trace: Any) -> tuple[str, ...]:
    if trace is None:
        return ()
    reasons = getattr(trace, "completeness_reasons", None)
    if reasons is None and isinstance(trace, Mapping):
        reasons = trace.get("completeness_reasons")
        nested = trace.get("completeness")
        if reasons is None and isinstance(nested, Mapping):
            reasons = nested.get("reasons")
    if reasons is None:
        completeness = getattr(trace, "completeness", None)
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


def _trace_is_complete(trace: Any) -> bool:
    if trace is None:
        return False
    complete = getattr(trace, "complete", None)
    if isinstance(complete, bool):
        return complete
    completeness = getattr(trace, "completeness", None)
    if completeness is not None:
        nested = getattr(completeness, "complete", None)
        if isinstance(nested, bool):
            return nested
        if isinstance(completeness, str):
            return completeness.lower() == "complete"
    if isinstance(trace, Mapping):
        if isinstance(trace.get("complete"), bool):
            return bool(trace["complete"])
        nested_map = trace.get("completeness")
        if isinstance(nested_map, Mapping) and isinstance(
            nested_map.get("complete"), bool
        ):
            return bool(nested_map["complete"])
    return False


def _reasons_disqualify_authority(reasons: tuple[str, ...]) -> bool:
    lowered = {reason.lower() for reason in reasons}
    if lowered & _NON_AUTHORITATIVE_REASON_MARKERS:
        return True
    for reason in lowered:
        if any(marker in reason for marker in _NON_AUTHORITATIVE_REASON_MARKERS):
            return True
    return False


@dataclass
class PytestRuntimeTraceLifecycle:
    """One-shot cold-pass tracer lifecycle around pytest setup/call/teardown.

    The lifecycle owns a single :class:`RuntimeTestDependencyTracer` session.
    Callers (the plugin) start observation immediately before setup, note each
    phase report, and stop only after teardown.  ``run_body`` is deliberately
    absent: the test body is invoked by pytest exactly once; this object never
    re-enters it.
    """

    __test__: ClassVar[bool] = False

    nodeid: str = ""
    locator_cid: str = ""
    allowed_roots: Mapping[str, Any] | None = None
    tracer_factory: Callable[..., Any] | None = None
    capture_code_objects: bool = False

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._phase = LifecyclePhase.IDLE
        self._setup_count = 0
        self._call_count = 0
        self._teardown_count = 0
        self._setup_outcome: PhaseOutcome | None = None
        self._call_outcome: PhaseOutcome | None = None
        self._teardown_outcome: PhaseOutcome | None = None
        self._tracer: Any = None
        self._trace: Any = None
        self._started = False
        self._stopped = False
        self._fault: str = ""
        self._diagnostics: dict[str, Any] = {}
        self._body_invocations = 0  # always 0: body is owned by pytest
        self._authority = LifecycleAuthority.NONE

    @property
    def interface(self) -> str:
        return PYTEST_RUNTIME_TRACE_LIFECYCLE_INTERFACE

    @property
    def phase(self) -> LifecyclePhase:
        return self._phase

    @property
    def setup_count(self) -> int:
        return self._setup_count

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def teardown_count(self) -> int:
        return self._teardown_count

    @property
    def setup_call_count(self) -> int:
        return self._setup_count

    @property
    def test_call_count(self) -> int:
        return self._call_count

    @property
    def teardown_call_count(self) -> int:
        return self._teardown_count

    @property
    def body_invocations(self) -> int:
        """Always zero: this lifecycle never invokes the test body."""

        return self._body_invocations

    @property
    def started(self) -> bool:
        return self._started

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def fault(self) -> str:
        return self._fault

    @property
    def trace(self) -> Any:
        return self._trace

    @property
    def runtime_trace(self) -> Any:
        return self._trace

    @property
    def tracer(self) -> Any:
        return self._tracer

    @property
    def authority(self) -> LifecycleAuthority:
        return self._authority

    @property
    def may_authorize_skip(self) -> bool:
        return False

    @property
    def lifecycle_complete(self) -> bool:
        return (
            self._setup_count == 1
            and self._call_count == 1
            and self._teardown_count == 1
            and self._setup_outcome is PhaseOutcome.PASS
            and self._call_outcome is PhaseOutcome.PASS
            and self._teardown_outcome is PhaseOutcome.PASS
        )

    @property
    def phases_each_once(self) -> bool:
        return (
            self._setup_count == 1
            and self._call_count == 1
            and self._teardown_count == 1
        )

    @property
    def publishes_authoritatively(self) -> bool:
        """True only for a complete three-phase pass with a complete trace."""

        return self._authority is LifecycleAuthority.COMPLETE_PASS

    def start(self) -> bool:
        """Start the tracer immediately before setup. Never raises into pytest."""

        with self._lock:
            if self._started:
                self._record_fault("duplicate_start")
                return False
            if self._stopped:
                self._record_fault("start_after_stop")
                return False
            self._started = True
            self._phase = LifecyclePhase.TRACING
        try:
            tracer = self._build_tracer()
            if tracer is None:
                self._record_fault("tracer_unavailable")
                return False
            start = getattr(tracer, "start", None)
            if callable(start):
                start()
            elif hasattr(tracer, "__enter__"):
                tracer.__enter__()
            with self._lock:
                self._tracer = tracer
            return True
        except BaseException as exc:
            # Tracing faults never alter pytest's real outcome.
            self._record_fault(f"start_failed:{type(exc).__name__}")
            return False

    def note_phase(self, when: str, outcome: PhaseOutcome | str | Any = "") -> bool:
        """Record one setup/call/teardown observation. Never raises."""

        try:
            phase_name = _bounded_text(when, max_chars=32).lower()
            if phase_name not in PHASES:
                return False
            if isinstance(outcome, PhaseOutcome):
                resolved = outcome
            elif isinstance(outcome, str) and outcome:
                try:
                    # Accept both PhaseOutcome values and pytest report outcomes.
                    normalized = outcome.strip().lower()
                    if normalized == "passed":
                        resolved = PhaseOutcome.PASS
                    elif normalized == "skipped":
                        resolved = PhaseOutcome.SKIP
                    elif normalized in {"failed", "fail"}:
                        resolved = PhaseOutcome.FAIL
                    elif normalized in {"error", "errored"}:
                        resolved = PhaseOutcome.ERROR
                    else:
                        resolved = PhaseOutcome(normalized)
                except Exception:
                    resolved = PhaseOutcome.ERROR
            else:
                resolved = PhaseOutcome.ERROR

            with self._lock:
                if phase_name == "setup":
                    if self._setup_count != 0:
                        self._phase = LifecyclePhase.FAILED
                        self._record_fault("duplicate_setup")
                        return False
                    self._setup_count = 1
                    self._setup_outcome = resolved
                    self._phase = LifecyclePhase.SETUP
                elif phase_name == "call":
                    if self._setup_count != 1 or self._call_count != 0:
                        self._phase = LifecyclePhase.FAILED
                        self._record_fault("duplicate_or_out_of_order_call")
                        return False
                    self._call_count = 1
                    self._call_outcome = resolved
                    self._phase = LifecyclePhase.CALL
                else:  # teardown
                    if self._call_count != 1 or self._teardown_count != 0:
                        # Allow teardown after setup-only failure paths so the
                        # ordinary pytest protocol can still finish cleanly.
                        if self._setup_count == 1 and self._call_count == 0:
                            self._call_count = 0
                        elif self._teardown_count != 0:
                            self._phase = LifecyclePhase.FAILED
                            self._record_fault("duplicate_teardown")
                            return False
                        else:
                            self._phase = LifecyclePhase.FAILED
                            self._record_fault("teardown_before_call")
                            return False
                    self._teardown_count = 1
                    self._teardown_outcome = resolved
                    self._phase = LifecyclePhase.TEARDOWN
            return True
        except BaseException as exc:
            self._record_fault(f"note_phase_failed:{type(exc).__name__}")
            return False

    def note_report(self, report: Any) -> bool:
        """Record a duck-typed pytest report for setup/call/teardown."""

        try:
            when = _bounded_text(getattr(report, "when", ""), max_chars=32).lower()
            if when not in PHASES:
                return False
            return self.note_phase(when, _phase_outcome_from_report(report))
        except BaseException as exc:
            self._record_fault(f"note_report_failed:{type(exc).__name__}")
            return False

    def stop(self) -> Any:
        """Stop the tracer after teardown and evaluate publication authority.

        Never raises.  Incomplete or exceptional traces yield ``None`` authority
        while still returning the observed (possibly incomplete) trace object.
        """

        with self._lock:
            if self._stopped:
                return self._trace
            self._stopped = True

        trace: Any = None
        try:
            tracer = self._tracer
            if tracer is not None:
                stop = getattr(tracer, "stop", None)
                if not callable(stop):
                    stop = getattr(tracer, "finish", None)
                if callable(stop):
                    trace = stop()
                else:
                    result = getattr(tracer, "result", None)
                    if result is not None and not callable(result):
                        trace = result
                    elif callable(getattr(tracer, "__exit__", None)):
                        tracer.__exit__(None, None, None)
                        trace = getattr(tracer, "result", None)
        except BaseException as exc:
            self._record_fault(f"stop_failed:{type(exc).__name__}")
            trace = None

        with self._lock:
            self._trace = trace
            self._phase = LifecyclePhase.STOPPED
            self._authority = self._evaluate_authority(trace)
            self._diagnostics["authority"] = self._authority.value
            self._diagnostics["lifecycle_complete"] = self.lifecycle_complete
            self._diagnostics["phases_each_once"] = self.phases_each_once
            self._diagnostics["body_invocations"] = self._body_invocations
        return trace

    def _evaluate_authority(self, trace: Any) -> LifecycleAuthority:
        if self._fault:
            return LifecycleAuthority.NONE
        if not self.lifecycle_complete:
            return LifecycleAuthority.NONE
        if not _trace_is_complete(trace):
            return LifecycleAuthority.NONE
        reasons = _trace_reasons(trace)
        if _reasons_disqualify_authority(reasons):
            return LifecycleAuthority.NONE
        # A complete empty-or-observed trace with three PASS phases is the only
        # path that may contribute to receipt / candidate publication.
        return LifecycleAuthority.COMPLETE_PASS

    def _build_tracer(self) -> Any:
        if self.tracer_factory is not None:
            return self.tracer_factory()
        from ...agent_supervisor.analysis.test_runtime_dependency_trace import (
            RuntimeTestDependencyTracer,
        )

        kwargs: dict[str, Any] = {
            "capture_code_objects": bool(self.capture_code_objects),
        }
        if self.allowed_roots is not None:
            kwargs["allowed_roots"] = dict(self.allowed_roots)
        return RuntimeTestDependencyTracer(**kwargs)

    def _record_fault(self, reason: str) -> None:
        text = _bounded_text(reason, max_chars=128)
        if not text:
            return
        if not self._fault:
            self._fault = text
        self._diagnostics.setdefault("faults", [])
        faults = self._diagnostics["faults"]
        if isinstance(faults, list) and text not in faults and len(faults) < 16:
            faults.append(text)
        self._authority = LifecycleAuthority.NONE

    def phase_outcomes(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if self._setup_outcome is not None:
            result["setup"] = self._setup_outcome.value
        if self._call_outcome is not None:
            result["call"] = self._call_outcome.value
        if self._teardown_outcome is not None:
            result["teardown"] = self._teardown_outcome.value
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PYTEST_RUNTIME_TRACE_LIFECYCLE_SCHEMA,
            "interface": PYTEST_RUNTIME_TRACE_LIFECYCLE_INTERFACE,
            "nodeid": self.nodeid,
            "locator_cid": self.locator_cid,
            "phase": self._phase.value,
            "started": self._started,
            "stopped": self._stopped,
            "setup_count": self._setup_count,
            "call_count": self._call_count,
            "teardown_count": self._teardown_count,
            "body_invocations": self._body_invocations,
            "lifecycle_complete": self.lifecycle_complete,
            "phases_each_once": self.phases_each_once,
            "authority": self._authority.value,
            "publishes_authoritatively": self.publishes_authoritatively,
            "may_authorize_skip": False,
            "fault": self._fault,
            "phase_outcomes": self.phase_outcomes(),
            "runtime_trace_root_cid": _trace_root_cid(self._trace),
            "diagnostics": dict(self._diagnostics),
        }


def _trace_root_cid(trace: Any) -> str:
    if trace is None:
        return ""
    for attr in ("trace_cid", "root_cid", "cid", "content_id"):
        value = getattr(trace, attr, None)
        if isinstance(value, str) and value:
            return _bounded_text(value, max_chars=256)
    if isinstance(trace, Mapping):
        for key in ("trace_cid", "root_cid", "cid", "content_id"):
            value = trace.get(key)
            if isinstance(value, str) and value:
                return _bounded_text(value, max_chars=256)
    return ""


def attach_runtime_lifecycle(
    item: Any,
    *,
    lifecycle: PytestRuntimeTraceLifecycle | None = None,
    allowed_roots: Mapping[str, Any] | None = None,
    tracer_factory: Callable[..., Any] | None = None,
    capture_code_objects: bool = False,
) -> PytestRuntimeTraceLifecycle:
    """Attach (or return) a lifecycle on a pytest item. Never raises."""

    existing = getattr(item, ITEM_RUNTIME_LIFECYCLE_ATTRIBUTE, None)
    if isinstance(existing, PytestRuntimeTraceLifecycle):
        return existing
    nodeid = _bounded_text(getattr(item, "nodeid", ""), max_chars=2048)
    locator = getattr(item, "_ipfs_proof_reuse_locator", None)
    locator_cid = ""
    if locator is not None:
        for attr in ("locator_id", "content_id", "cid"):
            value = getattr(locator, attr, None)
            if isinstance(value, str) and value:
                locator_cid = value
                break
    created = lifecycle or PytestRuntimeTraceLifecycle(
        nodeid=nodeid,
        locator_cid=locator_cid,
        allowed_roots=allowed_roots,
        tracer_factory=tracer_factory,
        capture_code_objects=capture_code_objects,
    )
    try:
        setattr(item, ITEM_RUNTIME_LIFECYCLE_ATTRIBUTE, created)
    except Exception:
        pass
    return created


def get_runtime_lifecycle(item: Any) -> Optional[PytestRuntimeTraceLifecycle]:
    existing = getattr(item, ITEM_RUNTIME_LIFECYCLE_ATTRIBUTE, None)
    if isinstance(existing, PytestRuntimeTraceLifecycle):
        return existing
    return None


__all__ = [
    "ITEM_RUNTIME_LIFECYCLE_ATTRIBUTE",
    "LifecycleAuthority",
    "LifecyclePhase",
    "PHASES",
    "PYTEST_RUNTIME_TRACE_LIFECYCLE_INTERFACE",
    "PYTEST_RUNTIME_TRACE_LIFECYCLE_SCHEMA",
    "PytestRuntimeTraceLifecycle",
    "attach_runtime_lifecycle",
    "get_runtime_lifecycle",
]
