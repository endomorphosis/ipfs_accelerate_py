"""Deterministic counterexample minimization for incremental verification.

IVP-011 / ``ivp/counterexample@1``
=================================

On a failed selected check the planner must not ship full logs into model
context.  This module turns already-bounded local artifacts into a compact
:class:`~.contracts.CounterexampleReceipt` that:

* retains the failing selector, relevant symbols, assertion, input/expected/
  observed diagnostics, source spans, environment and dependency-lock
  identities, reproduction argv (as a list), and artifact references;
* slices tracebacks and prunes log noise by a deterministic semantic cone;
* redacts private keys and marks inapplicable values with typed diagnostic
  states (``present`` / ``redacted`` / ``unavailable`` / ``not_applicable``);
* **reruns every candidate argv under a separate bounded admission lease**
  and only claims ``minimized=True`` when the rerun preserves the same
  failure identity;
* on minimization failure still emits an explicit receipt that references
  bounded artifact CIDs instead of embedding whole logs.

No model calls are made.  Completeness of the original process logs is never
required: only the captured bounded previews and content-addressed artifact
references are consulted.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

from .contracts import (
    MAX_COUNTEREXAMPLE_BYTES,
    CounterexampleReceipt,
    DiagnosticValueState,
    TerminalStatus,
    TestReceipt,
    TypeCheckReceipt,
    VerificationBoundsError,
    VerificationContractError,
    VerificationReceipt,
)

# Optional process-runner dependency is imported lazily-friendly for type
# checkers; runtime import stays local to avoid circular import pressure.
from .process_runner import (  # noqa: F401 — re-exported for callers/tests
    VerificationCommand,
    VerificationProcessRunner,
    VerificationRunDisposition,
    VerificationRunResult,
)

# ---------------------------------------------------------------------------
# Schema / evidence constants
# ---------------------------------------------------------------------------

COUNTEREXAMPLE_MINIMIZER_INTERFACE: Final[str] = "CounterexampleMinimizer@1"
COUNTEREXAMPLE_MINIMIZER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/counterexample-minimizer@1"
)
COUNTEREXAMPLE_MINIMIZATION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/counterexample-minimization-result@1"
)
FAILURE_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-failure-identity@1"
)
MINIMIZATION_QUALITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/counterexample-minimization-quality@1"
)
COUNTEREXAMPLE_EVIDENCE: Final[str] = "ivp/counterexample@1"
ALGORITHM_VERSION: Final[str] = "ivp-counterexample-minimizer/1.0.0"

# ---------------------------------------------------------------------------
# Bounds (closed; never upgraded silently)
# ---------------------------------------------------------------------------

DEFAULT_MAX_TRACEBACK_FRAMES: Final[int] = 16
DEFAULT_MAX_TRACEBACK_LINE_CHARS: Final[int] = 512
DEFAULT_MAX_LOG_LINES: Final[int] = 32
DEFAULT_MAX_LOG_LINE_CHARS: Final[int] = 512
DEFAULT_MAX_ASSERTION_CHARS: Final[int] = 4_096
DEFAULT_MAX_INPUT_KEYS: Final[int] = 32
DEFAULT_MAX_SOURCE_SPANS: Final[int] = 16
DEFAULT_MAX_CANDIDATE_ARGV: Final[int] = 16
DEFAULT_MAX_ORACLE_RERUNS: Final[int] = 16
DEFAULT_MAX_RECEIPT_BYTES: Final[int] = MAX_COUNTEREXAMPLE_BYTES

# Pytest / stdlib frames that never contribute semantic cone evidence.
_IRRELEVANT_FRAME_MARKERS: Final[tuple[str, ...]] = (
    "site-packages/",
    "dist-packages/",
    "/_pytest/",
    "\\_pytest\\",
    "/pluggy/",
    "\\pluggy\\",
    "/importlib/",
    "\\importlib\\",
    "/pytest/",
    "\\pytest\\",
    "pytest_runtest",
    "py/source.py",
    "<frozen importlib",
    "unittest/case.py",
    "unittest\\case.py",
)

_OPTIONAL_ARGV_FLAGS: Final[frozenset[str]] = frozenset(
    {
        "-v",
        "-vv",
        "-vvv",
        "--verbose",
        "-s",
        "--capture=no",
        "--capture=sys",
        "--capture=fd",
        "-q",
        "--quiet",
        "--tb=long",
        "--tb=auto",
        "--tb=line",
        "--tb=native",
        "--full-trace",
        "-x",
        "--exitfirst",
        "--lf",
        "--last-failed",
        "--ff",
        "--failed-first",
        "--color=yes",
        "--color=no",
        "--color=auto",
        "-p",
        "no:cacheprovider",
    }
)

# Flags that take a following argument and may be dropped as a pair when the
# flag itself is optional for reproduction.
_OPTIONAL_ARGV_FLAG_WITH_VALUE: Final[frozenset[str]] = frozenset(
    {
        "--tb",
        "--maxfail",
        "--durations",
        "-p",
    }
)

_PRIVATE_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "hidden_witness",
        "password",
        "private_key",
        "private_premise",
        "private_witness",
        "refresh_token",
        "secret",
        "session_token",
        "token",
        "witness",
    }
)

_LOG_KEEP_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bFAILED\b"),
    re.compile(r"\bERROR\b"),
    re.compile(r"\bAssertionError\b"),
    re.compile(r"\bassert\b", re.IGNORECASE),
    re.compile(r"^E\s+"),
    re.compile(r"^>\s+"),
    re.compile(r"^[A-Za-z0-9_./\\:-]+\.py:\d+"),
    re.compile(r"^=+ FAILURES =+$"),
    re.compile(r"^_+ .+ _+$"),
)

_FRAME_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<path>[^\s:]+\.py):(?P<line>\d+)(?::\s*(?P<body>.*))?$"
)
_NODE_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"(?P<node>[^\s:]+\.py(?:::[^\s]+)+)"
)
_ASSERTION_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:E\s+)?(?:AssertionError:\s*)?(?P<body>.+)$"
)


# ---------------------------------------------------------------------------
# Errors / guarantees
# ---------------------------------------------------------------------------


class CounterexampleMinimizationError(VerificationContractError):
    """Raised when a minimization request is malformed before any receipt."""


class MinimizationGuarantee(str, Enum):
    """Truthful reduction ladder; never upgraded without a successful lease rerun."""

    NONE = "none"
    NORMALIZED = "normalized"
    BOUNDED = "bounded"
    RERUN_VALIDATED = "rerun_validated"


# Callable oracle used by tests and production adapters.
# Returns a :class:`RerunObservation` for the candidate argv.
RerunOracle = Callable[[Sequence[str]], "RerunObservation"]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RerunObservation:
    """Bounded observation of one candidate argv rerun under a separate lease."""

    terminal_status: TerminalStatus
    exit_code: int | None
    lease_id: str
    command_argv: tuple[str, ...]
    stdout_preview: str = ""
    stderr_preview: str = ""
    stdout_artifact_cid: str = ""
    stderr_artifact_cid: str = ""
    process_started: bool = True
    publication_allowed: bool = True
    timed_out: bool = False
    cancelled: bool = False
    unavailable: bool = False
    reason_codes: tuple[str, ...] = ()
    combined_output: str = ""

    def __post_init__(self) -> None:
        status = self.terminal_status
        if not isinstance(status, TerminalStatus):
            object.__setattr__(self, "terminal_status", TerminalStatus(status))
        object.__setattr__(
            self,
            "command_argv",
            tuple(str(item) for item in self.command_argv),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(str(item) for item in self.reason_codes if str(item).strip()),
        )
        object.__setattr__(self, "lease_id", str(self.lease_id or "").strip())
        combined = self.combined_output
        if not combined:
            combined = "\n".join(
                part
                for part in (self.stdout_preview, self.stderr_preview)
                if part
            )
            object.__setattr__(self, "combined_output", combined)


@dataclass(frozen=True)
class FailureMaterial:
    """Already-bounded failure materials extracted from a selected check.

    Complete process logs are **not** stored here.  Callers must pass only
    bounded previews (or empty strings) plus content-addressed artifact CIDs.
    """

    node_id: str = ""
    exception_type: str = ""
    assertion_message: str = ""
    traceback_lines: tuple[str, ...] = ()
    log_lines: tuple[str, ...] = ()
    relevant_input: Mapping[str, Any] | None = None
    expected_output: Mapping[str, Any] | Any = None
    observed_output: Mapping[str, Any] | Any = None
    source_spans: tuple[Mapping[str, Any], ...] = ()
    relevant_paths: tuple[str, ...] = ()
    relevant_symbols: tuple[str, ...] = ()
    bounded_stdout_artifact_cid: str = ""
    bounded_stderr_artifact_cid: str = ""
    extra_artifact_cids: tuple[str, ...] = ()
    raw_stdout_preview: str = ""
    raw_stderr_preview: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", str(self.node_id or "").strip())
        object.__setattr__(
            self, "exception_type", str(self.exception_type or "").strip()
        )
        object.__setattr__(
            self,
            "assertion_message",
            _clip(str(self.assertion_message or ""), DEFAULT_MAX_ASSERTION_CHARS),
        )
        object.__setattr__(
            self,
            "traceback_lines",
            tuple(
                _clip(str(line), DEFAULT_MAX_TRACEBACK_LINE_CHARS)
                for line in (self.traceback_lines or ())
                if str(line).strip()
            ),
        )
        object.__setattr__(
            self,
            "log_lines",
            tuple(
                _clip(str(line), DEFAULT_MAX_LOG_LINE_CHARS)
                for line in (self.log_lines or ())
                if str(line).strip()
            ),
        )
        object.__setattr__(
            self,
            "relevant_paths",
            tuple(
                str(item).strip().replace("\\", "/")
                for item in (self.relevant_paths or ())
                if str(item).strip()
            ),
        )
        object.__setattr__(
            self,
            "relevant_symbols",
            tuple(
                str(item).strip()
                for item in (self.relevant_symbols or ())
                if str(item).strip()
            ),
        )
        object.__setattr__(
            self,
            "extra_artifact_cids",
            tuple(
                str(item).strip()
                for item in (self.extra_artifact_cids or ())
                if str(item).strip()
            ),
        )
        spans: list[Mapping[str, Any]] = []
        for span in self.source_spans or ():
            if isinstance(span, Mapping):
                spans.append(dict(span))
        object.__setattr__(self, "source_spans", tuple(spans))
        object.__setattr__(
            self,
            "bounded_stdout_artifact_cid",
            str(self.bounded_stdout_artifact_cid or "").strip(),
        )
        object.__setattr__(
            self,
            "bounded_stderr_artifact_cid",
            str(self.bounded_stderr_artifact_cid or "").strip(),
        )
        object.__setattr__(
            self, "raw_stdout_preview", str(self.raw_stdout_preview or "")
        )
        object.__setattr__(
            self, "raw_stderr_preview", str(self.raw_stderr_preview or "")
        )


@dataclass(frozen=True)
class MinimizationBudget:
    """Hard bounds for deterministic minimization work."""

    max_traceback_frames: int = DEFAULT_MAX_TRACEBACK_FRAMES
    max_traceback_line_chars: int = DEFAULT_MAX_TRACEBACK_LINE_CHARS
    max_log_lines: int = DEFAULT_MAX_LOG_LINES
    max_log_line_chars: int = DEFAULT_MAX_LOG_LINE_CHARS
    max_assertion_chars: int = DEFAULT_MAX_ASSERTION_CHARS
    max_input_keys: int = DEFAULT_MAX_INPUT_KEYS
    max_source_spans: int = DEFAULT_MAX_SOURCE_SPANS
    max_candidate_argv: int = DEFAULT_MAX_CANDIDATE_ARGV
    max_oracle_reruns: int = DEFAULT_MAX_ORACLE_RERUNS
    max_receipt_bytes: int = DEFAULT_MAX_RECEIPT_BYTES

    def __post_init__(self) -> None:
        for name in (
            "max_traceback_frames",
            "max_traceback_line_chars",
            "max_log_lines",
            "max_log_line_chars",
            "max_assertion_chars",
            "max_input_keys",
            "max_source_spans",
            "max_candidate_argv",
            "max_oracle_reruns",
            "max_receipt_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CounterexampleMinimizationError(
                    f"{name} must be a positive integer"
                )


@dataclass(frozen=True)
class MinimizationQuality:
    """Recorded minimization quality for routing and diagnostics."""

    SCHEMA: ClassVar[str] = MINIMIZATION_QUALITY_SCHEMA

    guarantee: MinimizationGuarantee
    frames_before: int
    frames_after: int
    log_lines_before: int
    log_lines_after: int
    input_keys_before: int
    input_keys_after: int
    oracle_reruns: int
    candidate_count: int
    score: float
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        guarantee = self.guarantee
        if not isinstance(guarantee, MinimizationGuarantee):
            object.__setattr__(
                self, "guarantee", MinimizationGuarantee(str(guarantee))
            )
        score = float(self.score)
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        object.__setattr__(self, "score", score)
        object.__setattr__(
            self,
            "reason_codes",
            tuple(str(item) for item in self.reason_codes if str(item).strip()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MINIMIZATION_QUALITY_SCHEMA,
            "guarantee": self.guarantee.value,
            "frames_before": self.frames_before,
            "frames_after": self.frames_after,
            "log_lines_before": self.log_lines_before,
            "log_lines_after": self.log_lines_after,
            "input_keys_before": self.input_keys_before,
            "input_keys_after": self.input_keys_after,
            "oracle_reruns": self.oracle_reruns,
            "candidate_count": self.candidate_count,
            "score": self.score,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class CounterexampleMinimizationResult:
    """Outcome of one deterministic minimization attempt."""

    SCHEMA: ClassVar[str] = COUNTEREXAMPLE_MINIMIZATION_RESULT_SCHEMA

    receipt: CounterexampleReceipt
    quality: MinimizationQuality
    failure_identity_cid: str
    lease_ids: tuple[str, ...]
    accepted_argv: tuple[str, ...]
    evidence: tuple[str, ...] = (COUNTEREXAMPLE_EVIDENCE,)
    algorithm_version: str = ALGORITHM_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COUNTEREXAMPLE_MINIMIZATION_RESULT_SCHEMA,
            "interface": COUNTEREXAMPLE_MINIMIZER_INTERFACE,
            "algorithm_version": self.algorithm_version,
            "evidence": list(self.evidence),
            "failure_identity_cid": self.failure_identity_cid,
            "lease_ids": list(self.lease_ids),
            "accepted_argv": list(self.accepted_argv),
            "quality": self.quality.to_dict(),
            "receipt": self.receipt.to_record(),
        }


@dataclass(frozen=True)
class MinimizationRequest:
    """One minimization request for a failed selected check.

    Exactly one of ``rerun_oracle`` or ``(process_runner, command_template)``
    should be supplied to claim ``minimized=True``.  Without a lease-backed
    rerun path the receipt is still produced but ``minimized`` is forced
    false with an explicit reason code.
    """

    failed_receipt: VerificationReceipt
    material: FailureMaterial
    reproduction_argv: Sequence[str] | None = None
    process_runner: VerificationProcessRunner | None = None
    command_template: VerificationCommand | None = None
    rerun_oracle: RerunOracle | None = None
    budget: MinimizationBudget = field(default_factory=MinimizationBudget)
    failed_obligation_cid: str = ""
    semantic_cone_paths: Sequence[str] = ()
    semantic_cone_symbols: Sequence[str] = ()


# ---------------------------------------------------------------------------
# Public helpers — diagnostic values, identity, extraction
# ---------------------------------------------------------------------------


def diagnostic_present(value: Any) -> Mapping[str, Any]:
    """Build a present-state diagnostic mapping."""

    return MappingProxyType({"state": DiagnosticValueState.PRESENT.value, "value": value})


def diagnostic_redacted() -> Mapping[str, Any]:
    return MappingProxyType({"state": DiagnosticValueState.REDACTED.value})


def diagnostic_unavailable() -> Mapping[str, Any]:
    return MappingProxyType({"state": DiagnosticValueState.UNAVAILABLE.value})


def diagnostic_not_applicable() -> Mapping[str, Any]:
    return MappingProxyType({"state": DiagnosticValueState.NOT_APPLICABLE.value})


def is_private_field_name(name: str) -> bool:
    """Return True when *name* looks like private/witness material."""

    normalized = str(name or "").strip().lower().replace("-", "_")
    if not normalized:
        return False
    return any(
        normalized == marker
        or normalized.endswith("_" + marker)
        or marker in normalized
        for marker in _PRIVATE_FIELD_MARKERS
    )


def sanitize_diagnostic_value(
    value: Any,
    *,
    field_name: str = "diagnostic",
    max_keys: int = DEFAULT_MAX_INPUT_KEYS,
    allow_missing: bool = True,
) -> Mapping[str, Any]:
    """Project a diagnostic value into a typed, secret-safe mapping.

    * Already-typed ``{"state": ...}`` mappings are normalized.
    * Mappings with private keys become ``redacted`` (no embedded value).
    * ``None`` becomes ``unavailable`` (or ``not_applicable`` when
      ``allow_missing`` is false for expected/observed in non-test contexts).
    * Plain values become ``present`` with a public-safe projection.
    """

    if isinstance(value, Mapping) and "state" in value:
        state_raw = value.get("state")
        try:
            state = DiagnosticValueState(str(state_raw))
        except (TypeError, ValueError) as exc:
            raise CounterexampleMinimizationError(
                f"{field_name}.state is not a DiagnosticValueState"
            ) from exc
        if state is DiagnosticValueState.PRESENT:
            if "value" not in value:
                raise CounterexampleMinimizationError(
                    f"{field_name} present state requires value"
                )
            projected = _project_public_value(
                value["value"], field_name=field_name, max_keys=max_keys
            )
            if projected is _REDACT_SENTINEL:
                return diagnostic_redacted()
            return diagnostic_present(projected)
        if "value" in value:
            # Non-present states never embed a value.
            return MappingProxyType({"state": state.value})
        return MappingProxyType({"state": state.value})

    if value is None:
        return diagnostic_unavailable() if allow_missing else diagnostic_not_applicable()

    if isinstance(value, Mapping):
        projected = _project_public_value(
            value, field_name=field_name, max_keys=max_keys
        )
        if projected is _REDACT_SENTINEL:
            return diagnostic_redacted()
        if projected is None or (isinstance(projected, Mapping) and not projected):
            return diagnostic_unavailable()
        return diagnostic_present(projected)

    projected = _project_public_value(value, field_name=field_name, max_keys=max_keys)
    if projected is _REDACT_SENTINEL:
        return diagnostic_redacted()
    return diagnostic_present(projected)


_REDACT_SENTINEL: Final[object] = object()


def _project_public_value(
    value: Any,
    *,
    field_name: str,
    max_keys: int,
    depth: int = 0,
) -> Any:
    if depth > 8:
        return "<truncated-depth>"
    if value is None or type(value) in {bool, int}:
        return value
    if isinstance(value, str):
        return _clip(value, DEFAULT_MAX_LOG_LINE_CHARS)
    if isinstance(value, float):
        # Contracts forbid floats in public records; stringify conservatively.
        return _clip(repr(value), 64)
    if isinstance(value, Mapping):
        if any(is_private_field_name(str(key)) for key in value):
            return _REDACT_SENTINEL
        items = sorted(
            ((str(k), value[k]) for k in value if str(k).strip()),
            key=lambda item: item[0],
        )
        if len(items) > max_keys:
            items = items[:max_keys]
        result: dict[str, Any] = {}
        for key, item in items:
            if is_private_field_name(key):
                return _REDACT_SENTINEL
            projected = _project_public_value(
                item,
                field_name=f"{field_name}.{key}",
                max_keys=max_keys,
                depth=depth + 1,
            )
            if projected is _REDACT_SENTINEL:
                return _REDACT_SENTINEL
            result[key] = projected
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)[:max_keys]
        projected_items = []
        for index, item in enumerate(items):
            projected = _project_public_value(
                item,
                field_name=f"{field_name}[{index}]",
                max_keys=max_keys,
                depth=depth + 1,
            )
            if projected is _REDACT_SENTINEL:
                return _REDACT_SENTINEL
            projected_items.append(projected)
        return projected_items
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _REDACT_SENTINEL
    return _clip(str(value), DEFAULT_MAX_LOG_LINE_CHARS)


def compute_failure_identity_cid(
    *,
    failed_selector: str,
    node_id: str = "",
    exception_type: str = "",
    assertion_message: str = "",
    primary_source: str = "",
    terminal_status: TerminalStatus | str = TerminalStatus.FAILED,
    environment_cid: str = "",
    dependency_lock_cid: str = "",
) -> str:
    """Compute a content-addressed failure identity for identity preservation.

    The identity binds the *class* of failure (selector, exception, assertion
    digest, primary source location, terminal status) plus environment and
    dependency-lock identities so a cross-environment flake cannot claim the
    same minimized counterexample.
    """

    status = (
        terminal_status.value
        if isinstance(terminal_status, TerminalStatus)
        else str(terminal_status)
    )
    assertion = _clip(str(assertion_message or "").strip(), DEFAULT_MAX_ASSERTION_CHARS)
    return content_identity(
        {
            "schema": FAILURE_IDENTITY_SCHEMA,
            "failed_selector": str(failed_selector or "").strip(),
            "node_id": str(node_id or "").strip(),
            "exception_type": str(exception_type or "").strip(),
            "assertion_digest": content_identity(
                {"schema": "assertion-digest@1", "text": assertion}
            ),
            "primary_source": str(primary_source or "").strip().replace("\\", "/"),
            "terminal_status": status,
            "environment_cid": str(environment_cid or "").strip(),
            "dependency_lock_cid": str(dependency_lock_cid or "").strip(),
        }
    )


def extract_failure_material_from_pytest_output(
    text: str,
    *,
    stdout_artifact_cid: str = "",
    stderr_artifact_cid: str = "",
    extra_artifact_cids: Sequence[str] = (),
    relevant_paths: Sequence[str] = (),
    relevant_symbols: Sequence[str] = (),
    relevant_input: Mapping[str, Any] | None = None,
    expected_output: Any = None,
    observed_output: Any = None,
    source_spans: Sequence[Mapping[str, Any]] = (),
) -> FailureMaterial:
    """Parse bounded pytest output into failure materials (no full-log retention)."""

    raw = str(text or "")
    lines = tuple(raw.splitlines())
    node_id = _detect_node_id(raw)
    exception_type, assertion = _detect_assertion(lines)
    traceback_lines = tuple(
        line
        for line in lines
        if _FRAME_RE.match(line.strip())
        or line.strip().startswith("E ")
        or line.strip().startswith("> ")
        or "Error" in line
        or "assert" in line.lower()
    )
    log_lines = lines
    primary_path = ""
    primary_line = 0
    for line in traceback_lines:
        match = _FRAME_RE.match(line.strip())
        if match and not _is_irrelevant_frame(match.group("path")):
            primary_path = match.group("path").replace("\\", "/")
            primary_line = int(match.group("line"))
            break
    spans = list(source_spans)
    if primary_path and not spans:
        spans.append(
            {
                "path": primary_path.lstrip("./"),
                "start_line": max(1, primary_line),
                "end_line": max(1, primary_line),
                "artifact_cid": stderr_artifact_cid
                or stdout_artifact_cid
                or content_identity(
                    {
                        "schema": "synthetic-source-span-artifact@1",
                        "path": primary_path,
                        "line": primary_line,
                    }
                ),
                "symbol": (
                    relevant_symbols[0]
                    if relevant_symbols
                    else (node_id.split("::")[-1] if node_id else "")
                ),
            }
        )
    return FailureMaterial(
        node_id=node_id,
        exception_type=exception_type,
        assertion_message=assertion,
        traceback_lines=traceback_lines,
        log_lines=log_lines,
        relevant_input=relevant_input,
        expected_output=expected_output,
        observed_output=observed_output,
        source_spans=tuple(spans),
        relevant_paths=tuple(relevant_paths),
        relevant_symbols=tuple(relevant_symbols),
        bounded_stdout_artifact_cid=stdout_artifact_cid,
        bounded_stderr_artifact_cid=stderr_artifact_cid,
        extra_artifact_cids=tuple(extra_artifact_cids),
        raw_stdout_preview=_clip(raw, 8_192),
        raw_stderr_preview="",
    )


# ---------------------------------------------------------------------------
# Traceback / log pruning
# ---------------------------------------------------------------------------


def slice_traceback(
    lines: Sequence[str],
    *,
    cone_paths: Sequence[str] = (),
    max_frames: int = DEFAULT_MAX_TRACEBACK_FRAMES,
    max_line_chars: int = DEFAULT_MAX_TRACEBACK_LINE_CHARS,
) -> tuple[str, ...]:
    """Deterministically slice a traceback to the relevant semantic cone."""

    cone = {
        str(path).strip().replace("\\", "/").lstrip("./")
        for path in cone_paths
        if str(path).strip()
    }
    kept: list[str] = []
    seen: set[str] = set()

    def _add(line: str) -> None:
        clipped = _clip(line.rstrip(), max_line_chars)
        if not clipped or clipped in seen:
            return
        seen.add(clipped)
        kept.append(clipped)

    frame_bodies: list[str] = []
    for raw in lines:
        line = str(raw).rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        match = _FRAME_RE.match(stripped)
        if match:
            path = match.group("path").replace("\\", "/")
            if _is_irrelevant_frame(path):
                continue
            rel = _normalize_repo_path(path, cone)
            body = match.group("body") or ""
            normalized_frame = f"{rel}:{match.group('line')}"
            if body:
                normalized_frame = f"{normalized_frame}: {body}"
            if cone and not _path_in_cone(rel, cone):
                # Keep the frame only when no cone match exists yet — final
                # pass below will drop pure-noise if cone frames appear.
                frame_bodies.append(normalized_frame)
                continue
            _add(normalized_frame)
            continue
        if stripped.startswith("E ") or stripped.startswith("> "):
            _add(stripped)
            continue
        if "Error" in stripped or stripped.lower().startswith("assert"):
            _add(stripped)

    # If the cone produced nothing, fall back to non-irrelevant frames.
    if not any(_FRAME_RE.match(item) for item in kept):
        for item in frame_bodies:
            if len([x for x in kept if _FRAME_RE.match(x)]) >= max_frames:
                break
            _add(item)

    # Bound frames: keep last max_frames frame lines plus assertion lines.
    frame_indices = [i for i, line in enumerate(kept) if _FRAME_RE.match(line)]
    if len(frame_indices) > max_frames:
        drop = set(frame_indices[: len(frame_indices) - max_frames])
        kept = [line for i, line in enumerate(kept) if i not in drop]

    return tuple(kept[: max_frames * 3])


def prune_log_lines(
    lines: Sequence[str],
    *,
    cone_paths: Sequence[str] = (),
    max_lines: int = DEFAULT_MAX_LOG_LINES,
    max_line_chars: int = DEFAULT_MAX_LOG_LINE_CHARS,
) -> tuple[str, ...]:
    """Drop irrelevant log noise; retain failure-bearing lines only."""

    cone = {
        str(path).strip().replace("\\", "/").lstrip("./")
        for path in cone_paths
        if str(path).strip()
    }
    kept: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = str(raw).rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        # Drop pure progress / noise.
        if stripped.startswith("=") and "FAILURES" not in stripped:
            if stripped.startswith("=") and stripped.endswith("="):
                if "short test summary" in stripped.lower():
                    continue
                if "FAILURES" not in stripped and "ERRORS" not in stripped:
                    continue
        if re.fullmatch(r"\.+|F+|E+|s+|x+|X+|p+", stripped):
            continue
        if stripped.startswith("INFO ") or stripped.startswith("DEBUG "):
            continue
        if _is_irrelevant_frame(stripped):
            # Full path noise lines (e.g. site-packages stack) skip.
            if any(marker in stripped for marker in _IRRELEVANT_FRAME_MARKERS):
                continue
        keep = False
        for pattern in _LOG_KEEP_PATTERNS:
            if pattern.search(stripped):
                keep = True
                break
        if not keep and cone:
            for path in cone:
                if path and path in stripped.replace("\\", "/"):
                    keep = True
                    break
        if not keep:
            continue
        clipped = _clip(stripped, max_line_chars)
        if clipped in seen:
            continue
        seen.add(clipped)
        kept.append(clipped)
        if len(kept) >= max_lines:
            break
    return tuple(kept)


def minimize_source_spans(
    spans: Sequence[Mapping[str, Any]],
    *,
    cone_paths: Sequence[str] = (),
    max_spans: int = DEFAULT_MAX_SOURCE_SPANS,
) -> tuple[dict[str, Any], ...]:
    """Keep cone-relevant source spans; drop duplicates and out-of-cone noise."""

    cone = {
        str(path).strip().replace("\\", "/").lstrip("./")
        for path in cone_paths
        if str(path).strip()
    }
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for span in spans or ():
        if not isinstance(span, Mapping):
            continue
        path = str(span.get("path") or "").strip().replace("\\", "/").lstrip("./")
        if not path or path.startswith("/") or ".." in path.split("/"):
            continue
        if cone and not _path_in_cone(path, cone):
            continue
        try:
            start = int(span.get("start_line") or 0)
            end = int(span.get("end_line") or start)
        except (TypeError, ValueError):
            continue
        if start < 1:
            continue
        if end < start:
            end = start
        artifact_cid = str(span.get("artifact_cid") or "").strip()
        if not artifact_cid:
            artifact_cid = content_identity(
                {
                    "schema": "synthetic-source-span-artifact@1",
                    "path": path,
                    "start_line": start,
                    "end_line": end,
                }
            )
        symbol = str(span.get("symbol") or "").strip()
        identity = f"{path}:{start}:{end}:{artifact_cid}:{symbol}"
        if identity in seen:
            continue
        seen.add(identity)
        result.append(
            {
                "path": path,
                "start_line": start,
                "end_line": end,
                "artifact_cid": artifact_cid,
                "symbol": symbol,
            }
        )
        if len(result) >= max_spans:
            break
    # Stable order by path then start line.
    result.sort(key=lambda item: (item["path"], item["start_line"], item["end_line"]))
    return tuple(result)


# ---------------------------------------------------------------------------
# Argv candidates
# ---------------------------------------------------------------------------


def build_reproduction_argv_candidates(
    argv: Sequence[str],
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATE_ARGV,
) -> tuple[tuple[str, ...], ...]:
    """Build a deterministic sequence of argv candidates (list form each).

    The original argv is always first.  Subsequent candidates drop optional
    verbosity / traceback flags that do not change selection.  Selection
    tokens (pytest node ids, suite paths) are never removed.
    """

    base = tuple(str(item) for item in argv)
    if not base:
        raise CounterexampleMinimizationError("reproduction_argv must not be empty")
    candidates: list[tuple[str, ...]] = [base]
    seen: set[tuple[str, ...]] = {base}

    # Prefer a short traceback form when a longer form is present.
    tb_normalized = _normalize_tb_flag(base)
    if tb_normalized not in seen:
        candidates.append(tb_normalized)
        seen.add(tb_normalized)

    # Greedy left-to-right optional-flag drops (deterministic).
    working = list(tb_normalized)
    index = 1  # never drop executable
    while index < len(working) and len(candidates) < max_candidates:
        token = working[index]
        if token in _OPTIONAL_ARGV_FLAGS or token.startswith("--tb="):
            trial = working[:index] + working[index + 1 :]
            # Drop paired value for -p no:cacheprovider etc. when value is optional.
            if (
                token in _OPTIONAL_ARGV_FLAG_WITH_VALUE
                and index < len(working)
                and index < len(trial) + 1
            ):
                # already dropped only the flag; if next token looks like a
                # value (no leading path/node), drop it too on a separate trial.
                pass
            candidate = tuple(trial)
            if candidate and candidate not in seen and candidate[0].strip():
                candidates.append(candidate)
                seen.add(candidate)
                working = list(candidate)
                # Do not advance index: next token shifted into place.
                continue
        if (
            token in _OPTIONAL_ARGV_FLAG_WITH_VALUE
            and index + 1 < len(working)
            and not working[index + 1].startswith("-")
            and not working[index + 1].endswith(".py")
            and "::" not in working[index + 1]
        ):
            trial = working[:index] + working[index + 2 :]
            candidate = tuple(trial)
            if candidate and candidate not in seen and candidate[0].strip():
                candidates.append(candidate)
                seen.add(candidate)
                working = list(candidate)
                continue
        index += 1

    return tuple(candidates[:max_candidates])


def _normalize_tb_flag(argv: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    replaced = False
    for item in argv:
        if item in {"--tb=long", "--tb=auto", "--tb=native", "--full-trace"}:
            if not replaced:
                result.append("--tb=short")
                replaced = True
            continue
        if item == "--tb":
            # skip flag; value handled below
            continue
        if result and result[-1] == "--tb":
            # unreachable due to continue above — keep structure defensive
            pass
        result.append(item)
    # Clean any orphaned --tb that lost its value handling
    cleaned: list[str] = []
    skip_next = False
    for i, item in enumerate(result):
        if skip_next:
            skip_next = False
            continue
        if item == "--tb":
            cleaned.append("--tb=short")
            if i + 1 < len(result) and result[i + 1] in {
                "long",
                "auto",
                "native",
                "line",
                "short",
            }:
                skip_next = True
            continue
        cleaned.append(item)
    return tuple(cleaned)


# ---------------------------------------------------------------------------
# Minimizer
# ---------------------------------------------------------------------------


class CounterexampleMinimizer:
    """Deterministic, lease-rerun-validated counterexample minimizer.

    The minimizer never embeds whole logs.  Successful minimization requires
    a separate bounded-lease rerun that preserves failure identity.
    """

    interface: str = COUNTEREXAMPLE_MINIMIZER_INTERFACE
    schema: str = COUNTEREXAMPLE_MINIMIZER_SCHEMA
    evidence: str = COUNTEREXAMPLE_EVIDENCE
    algorithm_version: str = ALGORITHM_VERSION

    def minimize(
        self, request: MinimizationRequest
    ) -> CounterexampleMinimizationResult:
        """Minimize a failed selected check into a compact CounterexampleReceipt."""

        if not isinstance(request, MinimizationRequest):
            raise CounterexampleMinimizationError(
                "request must be a MinimizationRequest"
            )
        receipt = request.failed_receipt
        if not isinstance(receipt, (TestReceipt, TypeCheckReceipt)):
            # Accept other VerificationReceipt forms structurally via duck typing
            # of .key / .execution / .receipt_id / .artifact_cids.
            if not hasattr(receipt, "key") or not hasattr(receipt, "execution"):
                raise CounterexampleMinimizationError(
                    "failed_receipt must be a verification receipt with key and execution"
                )

        key = receipt.key
        execution = receipt.execution
        if execution.terminal_status not in {
            TerminalStatus.FAILED,
            TerminalStatus.DISPROVED,
            TerminalStatus.INVALID,
        }:
            raise CounterexampleMinimizationError(
                "counterexample minimization requires a failed terminal status"
            )

        budget = request.budget or MinimizationBudget()
        material = request.material
        # Semantic cone is caller/material declared paths only.  Source spans are
        # filtered *against* the cone and must not expand it (otherwise pytest
        # internals listed as spans would keep themselves).
        cone_paths = tuple(
            dict.fromkeys(
                [
                    *(material.relevant_paths or ()),
                    *(request.semantic_cone_paths or ()),
                ]
            )
        )
        cone_paths = tuple(
            _normalize_repo_path(p, cone_paths) for p in cone_paths if str(p).strip()
        )
        cone_paths = tuple(p for p in cone_paths if p and not _is_irrelevant_frame(p))
        original_argv = tuple(
            str(item)
            for item in (
                request.reproduction_argv
                if request.reproduction_argv is not None
                else execution.command_argv
            )
        )
        if not original_argv:
            raise CounterexampleMinimizationError(
                "reproduction_argv must be a non-empty list"
            )

        frames_before = len(material.traceback_lines)
        log_before = len(material.log_lines)
        input_before = _count_input_keys(material.relevant_input)

        minimized_tb = slice_traceback(
            material.traceback_lines or material.log_lines,
            cone_paths=cone_paths,
            max_frames=budget.max_traceback_frames,
            max_line_chars=budget.max_traceback_line_chars,
        )
        # Prefer assertion-bearing log slice when traceback empty.
        pruned_logs = prune_log_lines(
            material.log_lines or material.traceback_lines,
            cone_paths=cone_paths,
            max_lines=budget.max_log_lines,
            max_line_chars=budget.max_log_line_chars,
        )
        if not minimized_tb and pruned_logs:
            minimized_tb = pruned_logs[: budget.max_traceback_frames]

        # Always retain at least one compact failure line.
        if not minimized_tb:
            fallback = material.assertion_message or material.exception_type or "failed"
            minimized_tb = (_clip(f"failure: {fallback}", budget.max_traceback_line_chars),)

        assertion = _clip(
            material.assertion_message
            or _assertion_from_lines(minimized_tb)
            or material.exception_type
            or "failure",
            budget.max_assertion_chars,
        )

        relevant_input = sanitize_diagnostic_value(
            material.relevant_input,
            field_name="relevant_input",
            max_keys=budget.max_input_keys,
            allow_missing=True,
        )
        expected_output = sanitize_diagnostic_value(
            material.expected_output,
            field_name="expected_output",
            max_keys=budget.max_input_keys,
            allow_missing=True,
        )
        observed_output = sanitize_diagnostic_value(
            material.observed_output,
            field_name="observed_output",
            max_keys=budget.max_input_keys,
            allow_missing=True,
        )

        # If inputs fully redacted due to secrets, keep redacted state.
        if material.relevant_input is None and relevant_input.get("state") == "unavailable":
            # Explicit not_applicable when the check family has no fixture input.
            if not isinstance(receipt, TestReceipt):
                relevant_input = diagnostic_not_applicable()

        source_spans = minimize_source_spans(
            material.source_spans,
            cone_paths=cone_paths,
            max_spans=budget.max_source_spans,
        )

        primary_source = ""
        if source_spans:
            primary_source = (
                f"{_normalize_repo_path(str(source_spans[0]['path']), cone_paths)}"
                f":{source_spans[0]['start_line']}"
            )
        else:
            for line in minimized_tb:
                match = _FRAME_RE.match(line.strip())
                if match:
                    primary_source = (
                        f"{_normalize_repo_path(match.group('path'), cone_paths)}"
                        f":{match.group('line')}"
                    )
                    break

        original_identity = compute_failure_identity_cid(
            failed_selector=key.selector_cid,
            node_id=material.node_id,
            exception_type=material.exception_type,
            assertion_message=assertion,
            primary_source=primary_source,
            terminal_status=execution.terminal_status,
            environment_cid=key.environment_cid,
            dependency_lock_cid=key.dependency_lock_cid,
        )

        artifact_cids = _collect_artifact_cids(
            receipt=receipt,
            material=material,
            extra=(),
        )

        candidates = build_reproduction_argv_candidates(
            original_argv, max_candidates=budget.max_candidate_argv
        )
        oracle = self._resolve_oracle(request)
        lease_ids: list[str] = []
        accepted_argv: tuple[str, ...] | None = None
        rerun_count = 0
        reason_codes: list[str] = []
        last_rerun: RerunObservation | None = None

        if oracle is None:
            reason_codes.append("minimization_failed_no_lease_rerun")
        else:
            for candidate in candidates:
                if rerun_count >= budget.max_oracle_reruns:
                    reason_codes.append("minimization_budget_exhausted")
                    break
                observation = oracle(candidate)
                rerun_count += 1
                if observation.lease_id:
                    lease_ids.append(observation.lease_id)
                last_rerun = observation

                if observation.unavailable or observation.cancelled or observation.timed_out:
                    reason_codes.append(
                        f"rerun_{observation.terminal_status.value}"
                    )
                    continue
                if not observation.process_started:
                    reason_codes.append("rerun_process_not_started")
                    continue
                if observation.terminal_status not in {
                    TerminalStatus.FAILED,
                    TerminalStatus.DISPROVED,
                    TerminalStatus.INVALID,
                }:
                    reason_codes.append("rerun_did_not_fail")
                    continue

                rerun_material = extract_failure_material_from_pytest_output(
                    observation.combined_output,
                    stdout_artifact_cid=observation.stdout_artifact_cid,
                    stderr_artifact_cid=observation.stderr_artifact_cid,
                    relevant_paths=cone_paths,
                    relevant_symbols=material.relevant_symbols,
                )
                # Prefer original assertion/exception when rerun text is sparse
                # but status failed (e.g. mocked oracle with identity-only payload).
                rerun_exception = (
                    rerun_material.exception_type or material.exception_type
                )
                rerun_assertion = (
                    rerun_material.assertion_message
                    or assertion
                    or material.assertion_message
                )
                rerun_node = rerun_material.node_id or material.node_id
                rerun_primary = primary_source
                for line in slice_traceback(
                    rerun_material.traceback_lines or rerun_material.log_lines,
                    cone_paths=cone_paths,
                    max_frames=budget.max_traceback_frames,
                ):
                    match = _FRAME_RE.match(line.strip())
                    if match:
                        rerun_primary = (
                            f"{_normalize_repo_path(match.group('path'), cone_paths)}"
                            f":{match.group('line')}"
                        )
                        break
                # Prefer the original assertion text when the rerun parser only
                # recovered a weaker summary form of the same exception.
                if (
                    material.assertion_message
                    and rerun_assertion
                    and material.assertion_message in rerun_assertion
                ):
                    rerun_assertion = material.assertion_message
                if not rerun_exception:
                    rerun_exception = material.exception_type

                rerun_identity = compute_failure_identity_cid(
                    failed_selector=key.selector_cid,
                    node_id=rerun_node,
                    exception_type=rerun_exception,
                    assertion_message=rerun_assertion,
                    primary_source=rerun_primary or primary_source,
                    terminal_status=observation.terminal_status,
                    environment_cid=key.environment_cid,
                    dependency_lock_cid=key.dependency_lock_cid,
                )
                if rerun_identity != original_identity:
                    reason_codes.append("failure_identity_not_preserved")
                    # Keep going — a later smaller argv might still match if
                    # the mismatch was noise; for identity drift on same
                    # selector this usually fails all candidates.
                    continue

                # Success: preserve identity under a separate lease.
                accepted_argv = tuple(candidate)
                # Attach rerun stream artifacts (bounded references only).
                if observation.stdout_artifact_cid:
                    artifact_cids.append(observation.stdout_artifact_cid)
                if observation.stderr_artifact_cid:
                    artifact_cids.append(observation.stderr_artifact_cid)
                reason_codes.append("deterministic_slice_preserved_failure")
                reason_codes.append("lease_rerun_validated")
                break
            else:
                if "failure_identity_not_preserved" not in reason_codes:
                    reason_codes.append("minimization_failed_no_candidate")

        minimized = accepted_argv is not None
        if not minimized:
            if "minimization_failed_no_lease_rerun" not in reason_codes:
                reason_codes.append("minimization_failed")
            # Reference bounded original artifacts; never embed whole logs.
            if material.bounded_stdout_artifact_cid:
                artifact_cids.append(material.bounded_stdout_artifact_cid)
            if material.bounded_stderr_artifact_cid:
                artifact_cids.append(material.bounded_stderr_artifact_cid)
            if last_rerun is not None:
                if last_rerun.stdout_artifact_cid:
                    artifact_cids.append(last_rerun.stdout_artifact_cid)
                if last_rerun.stderr_artifact_cid:
                    artifact_cids.append(last_rerun.stderr_artifact_cid)
            reproduction = original_argv
        else:
            reproduction = accepted_argv or original_argv

        # Deduplicate artifact CIDs while preserving order.
        deduped_artifacts: list[str] = []
        seen_art: set[str] = set()
        for cid in artifact_cids:
            text = str(cid or "").strip()
            if not text or text in seen_art:
                continue
            seen_art.add(text)
            deduped_artifacts.append(text)

        # Relevant symbol versions from the sealed receipt key (cone binding).
        symbol_cids = tuple(key.affected_symbol_version_cids or ())

        # Build receipt; if it exceeds the byte budget, shrink traceback further.
        receipt_obj = self._build_receipt(
            key=key,
            failed_receipt=receipt,
            failure_identity_cid=original_identity,
            minimized_traceback=minimized_tb,
            assertion=assertion,
            relevant_input=relevant_input,
            expected_output=expected_output,
            observed_output=observed_output,
            source_spans=source_spans,
            reproduction_argv=reproduction,
            artifact_cids=tuple(deduped_artifacts),
            minimized=minimized,
            failed_obligation_cid=request.failed_obligation_cid
            or getattr(key, "proof_obligation_cid", "")
            or "",
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            symbol_cids=symbol_cids,
            max_bytes=budget.max_receipt_bytes,
        )

        frames_after = len(receipt_obj.minimized_traceback)
        log_after = len(pruned_logs)
        input_after = _count_input_keys(receipt_obj.relevant_input)

        guarantee = (
            MinimizationGuarantee.RERUN_VALIDATED
            if minimized
            else (
                MinimizationGuarantee.NORMALIZED
                if oracle is None
                else MinimizationGuarantee.BOUNDED
            )
        )
        score = _quality_score(
            guarantee=guarantee,
            frames_before=frames_before,
            frames_after=frames_after,
            log_before=log_before,
            log_after=log_after,
            input_before=input_before,
            input_after=input_after,
            minimized=minimized,
        )
        quality = MinimizationQuality(
            guarantee=guarantee,
            frames_before=frames_before,
            frames_after=frames_after,
            log_lines_before=log_before,
            log_lines_after=log_after,
            input_keys_before=input_before,
            input_keys_after=input_after,
            oracle_reruns=rerun_count,
            candidate_count=len(candidates),
            score=score,
            reason_codes=tuple(dict.fromkeys(reason_codes)),
        )
        return CounterexampleMinimizationResult(
            receipt=receipt_obj,
            quality=quality,
            failure_identity_cid=original_identity,
            lease_ids=tuple(lease_ids),
            accepted_argv=tuple(reproduction) if minimized else (),
            evidence=(COUNTEREXAMPLE_EVIDENCE,),
            algorithm_version=self.algorithm_version,
        )

    # -- internals ---------------------------------------------------------

    def _resolve_oracle(self, request: MinimizationRequest) -> RerunOracle | None:
        if request.rerun_oracle is not None:
            return request.rerun_oracle
        if request.process_runner is not None and request.command_template is not None:
            runner = request.process_runner
            template = request.command_template

            def _oracle(argv: Sequence[str]) -> RerunObservation:
                # Each call builds a fresh command so the runner acquires a
                # *separate* bounded admission lease (lane_id uniqueness).
                command = VerificationCommand(
                    argv=list(argv),
                    cwd=template.cwd,
                    environment=dict(template.environment),
                    timeout_seconds=template.timeout_seconds,
                    sandbox=template.sandbox,
                    network_policy=template.network_policy,
                    max_stdout_bytes=template.max_stdout_bytes,
                    max_stderr_bytes=template.max_stderr_bytes,
                    lane_id="",  # force a fresh lane / lease identity
                    resource_class=template.resource_class,
                    stage=template.stage,
                    metadata={
                        **dict(template.metadata),
                        "purpose": "counterexample-minimization-rerun",
                    },
                    stdin=template.stdin,
                )
                result = runner.run(command)
                return rerun_observation_from_run_result(result)

            return _oracle
        return None

    def _build_receipt(
        self,
        *,
        key: Any,
        failed_receipt: Any,
        failure_identity_cid: str,
        minimized_traceback: Sequence[str],
        assertion: str,
        relevant_input: Mapping[str, Any],
        expected_output: Mapping[str, Any],
        observed_output: Mapping[str, Any],
        source_spans: Sequence[Mapping[str, Any]],
        reproduction_argv: Sequence[str],
        artifact_cids: Sequence[str],
        minimized: bool,
        failed_obligation_cid: str,
        reason_codes: Sequence[str],
        symbol_cids: Sequence[str],
        max_bytes: int,
    ) -> CounterexampleReceipt:
        frames = tuple(minimized_traceback)
        reasons = tuple(reason_codes)
        while True:
            try:
                receipt = CounterexampleReceipt(
                    failed_key_cid=key.key_id,
                    failed_receipt_cid=failed_receipt.receipt_id,
                    failed_selector=key.selector_cid,
                    failure_identity_cid=failure_identity_cid,
                    relevant_symbol_version_cids=tuple(symbol_cids),
                    minimized_traceback=frames,
                    relevant_assertion=assertion,
                    relevant_input=dict(relevant_input),
                    expected_output=dict(expected_output),
                    observed_output=dict(observed_output),
                    source_spans=tuple(dict(span) for span in source_spans),
                    environment_cid=key.environment_cid,
                    dependency_lock_cid=key.dependency_lock_cid,
                    reproduction_argv=tuple(reproduction_argv),
                    artifact_cids=tuple(artifact_cids),
                    minimized=minimized,
                    failed_obligation_cid=failed_obligation_cid or "",
                    reason_codes=reasons,
                )
            except VerificationBoundsError:
                if len(frames) > 1:
                    frames = frames[: max(1, len(frames) // 2)]
                    if "receipt_byte_budget_shrink" not in reasons:
                        reasons = reasons + ("receipt_byte_budget_shrink",)
                    continue
                if len(assertion) > 64:
                    assertion = _clip(assertion, max(64, len(assertion) // 2))
                    continue
                raise
            if len(receipt.canonical_bytes()) > max_bytes:
                if len(frames) > 1:
                    frames = frames[: max(1, len(frames) // 2)]
                    if "receipt_byte_budget_shrink" not in reasons:
                        reasons = reasons + ("receipt_byte_budget_shrink",)
                    continue
                raise VerificationBoundsError(
                    f"counterexample receipt exceeds {max_bytes} bytes after shrink"
                )
            return receipt


def minimize_counterexample(
    failed_receipt: VerificationReceipt,
    material: FailureMaterial | str | None = None,
    *,
    reproduction_argv: Sequence[str] | None = None,
    process_runner: VerificationProcessRunner | None = None,
    command_template: VerificationCommand | None = None,
    rerun_oracle: RerunOracle | None = None,
    budget: MinimizationBudget | None = None,
    failed_obligation_cid: str = "",
    semantic_cone_paths: Sequence[str] = (),
    semantic_cone_symbols: Sequence[str] = (),
    relevant_input: Mapping[str, Any] | None = None,
    expected_output: Any = None,
    observed_output: Any = None,
    source_spans: Sequence[Mapping[str, Any]] = (),
    stdout_artifact_cid: str = "",
    stderr_artifact_cid: str = "",
) -> CounterexampleMinimizationResult:
    """Module-level entry point for IVP-011 counterexample minimization.

    ``material`` may be a :class:`FailureMaterial`, a bounded pytest output
    string, or omitted (in which case materials are taken from the receipt's
    execution artifact CIDs with empty previews — useful when only identity
    and argv need to be bound and the caller supplies diagnostics separately).
    """

    if material is None:
        execution = failed_receipt.execution
        failure_material = FailureMaterial(
            bounded_stdout_artifact_cid=stdout_artifact_cid
            or execution.stdout_artifact_cid,
            bounded_stderr_artifact_cid=stderr_artifact_cid
            or execution.stderr_artifact_cid,
            extra_artifact_cids=tuple(execution.artifact_cids or ()),
            relevant_input=relevant_input,
            expected_output=expected_output,
            observed_output=observed_output,
            source_spans=tuple(source_spans),
            assertion_message="failure",
            exception_type="Failure",
            traceback_lines=("failure: selected check failed",),
            log_lines=("failure: selected check failed",),
        )
    elif isinstance(material, str):
        execution = failed_receipt.execution
        failure_material = extract_failure_material_from_pytest_output(
            material,
            stdout_artifact_cid=stdout_artifact_cid
            or execution.stdout_artifact_cid,
            stderr_artifact_cid=stderr_artifact_cid
            or execution.stderr_artifact_cid,
            extra_artifact_cids=tuple(execution.artifact_cids or ()),
            relevant_paths=semantic_cone_paths,
            relevant_symbols=semantic_cone_symbols,
            relevant_input=relevant_input,
            expected_output=expected_output,
            observed_output=observed_output,
            source_spans=source_spans,
        )
    elif isinstance(material, FailureMaterial):
        failure_material = material
    else:
        raise CounterexampleMinimizationError(
            "material must be FailureMaterial, str, or None"
        )

    request = MinimizationRequest(
        failed_receipt=failed_receipt,
        material=failure_material,
        reproduction_argv=reproduction_argv,
        process_runner=process_runner,
        command_template=command_template,
        rerun_oracle=rerun_oracle,
        budget=budget or MinimizationBudget(),
        failed_obligation_cid=failed_obligation_cid,
        semantic_cone_paths=semantic_cone_paths,
        semantic_cone_symbols=semantic_cone_symbols,
    )
    return CounterexampleMinimizer().minimize(request)


def rerun_observation_from_run_result(
    result: VerificationRunResult,
) -> RerunObservation:
    """Project a process-runner result into a bounded rerun observation."""

    return RerunObservation(
        terminal_status=result.terminal_status,
        exit_code=result.exit_code,
        lease_id=str(result.lease_id or ""),
        command_argv=tuple(result.command_argv),
        stdout_preview=str(result.stdout.preview or ""),
        stderr_preview=str(result.stderr.preview or ""),
        stdout_artifact_cid=str(result.stdout.cid or ""),
        stderr_artifact_cid=str(result.stderr.cid or ""),
        process_started=bool(result.process_started),
        publication_allowed=bool(result.publication_allowed),
        timed_out=bool(result.timed_out),
        cancelled=bool(result.cancelled),
        unavailable=bool(result.unavailable),
        reason_codes=tuple(result.reason_codes),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clip(text: str, maximum: int) -> str:
    raw = str(text or "")
    if len(raw) <= maximum:
        return raw
    if maximum <= 3:
        return raw[:maximum]
    return raw[: maximum - 3] + "..."


def _is_irrelevant_frame(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return any(marker in normalized for marker in _IRRELEVANT_FRAME_MARKERS)


def _normalize_repo_path(path: str, cone: Sequence[str] | set[str] = ()) -> str:
    """Project an absolute or noisy path onto a repository-relative form."""

    rel = str(path or "").replace("\\", "/").strip()
    if not rel:
        return ""
    # Drop leading noise while preserving relative segments.
    for marker in ("/site-packages/", "/dist-packages/"):
        if marker in rel:
            return rel.split(marker, 1)[-1]
    for item in cone:
        candidate = str(item or "").replace("\\", "/").lstrip("./")
        if not candidate:
            continue
        if rel.endswith("/" + candidate) or rel.endswith(candidate):
            return candidate
        idx = rel.find("/" + candidate)
        if idx >= 0:
            return rel[idx + 1 :]
        if candidate in rel:
            idx = rel.find(candidate)
            if idx >= 0:
                return rel[idx:]
    # Absolute paths without a cone match: keep the last meaningful segment chain.
    if rel.startswith("/"):
        parts = [part for part in rel.split("/") if part]
        for i, part in enumerate(parts):
            if part in {"src", "lib", "tests", "test", "ipfs_accelerate_py"}:
                return "/".join(parts[i:])
        if parts:
            return parts[-1]
    return rel.lstrip("./")


def _path_in_cone(path: str, cone: set[str]) -> bool:
    rel = _normalize_repo_path(path, cone)
    for item in cone:
        candidate = _normalize_repo_path(str(item), cone)
        if not candidate:
            continue
        if rel == candidate or rel.endswith("/" + candidate) or candidate.endswith(
            "/" + rel
        ):
            return True
        if candidate in rel or rel in candidate:
            return True
    return False


def _detect_node_id(text: str) -> str:
    # Prefer FAILED summary lines.
    for line in str(text or "").splitlines():
        if "FAILED" in line:
            match = _NODE_ID_RE.search(line)
            if match:
                return match.group("node")
    match = _NODE_ID_RE.search(str(text or ""))
    return match.group("node") if match else ""


def _detect_assertion(lines: Sequence[str]) -> tuple[str, str]:
    exception_type = ""
    assertion = ""
    # Prefer pytest "E " detail lines over FAILED summary lines so the
    # assertion text stays stable across short/long and summary formats.
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("E "):
            continue
        body = stripped[2:].strip()
        if "Error" in body or "Exception" in body:
            if ":" in body:
                head, _, rest = body.partition(":")
                head = head.strip()
                if head.endswith("Error") or head.endswith("Exception"):
                    exception_type = head
                    assertion = rest.strip() or body
                    break
            exception_type = body.split()[0] if body else ""
            assertion = body
            break
        if "assert" in body.lower() and not assertion:
            assertion = body
            if not exception_type:
                exception_type = "AssertionError"
    if assertion:
        return exception_type or "AssertionError", assertion

    for line in lines:
        stripped = line.strip()
        # Skip session summary lines that embed the exception type.
        if stripped.startswith("FAILED ") or stripped.startswith("ERROR "):
            # Recover exception type only when detail lines were absent.
            if "AssertionError" in stripped and not exception_type:
                exception_type = "AssertionError"
                _, _, tail = stripped.partition("AssertionError")
                assertion = tail.lstrip(": -").strip() or "AssertionError"
            continue
        if "AssertionError" in stripped:
            exception_type = "AssertionError"
            match = _ASSERTION_RE.match(stripped)
            assertion = match.group("body") if match else stripped
            assertion = assertion.replace("AssertionError:", "").strip()
            break
    if not assertion:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("E ") and "assert" in stripped.lower():
                assertion = stripped[2:].strip()
                if not exception_type:
                    exception_type = "AssertionError"
                break
    return exception_type, assertion


def _assertion_from_lines(lines: Sequence[str]) -> str:
    _exc, assertion = _detect_assertion(lines)
    return assertion


def _count_input_keys(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, Mapping):
        if "state" in value:
            inner = value.get("value")
            if isinstance(inner, Mapping):
                return len(inner)
            return 1 if value.get("state") == DiagnosticValueState.PRESENT.value else 0
        return len(value)
    return 0


def _collect_artifact_cids(
    *,
    receipt: Any,
    material: FailureMaterial,
    extra: Sequence[str],
) -> list[str]:
    result: list[str] = []
    for cid in (
        *tuple(getattr(receipt, "artifact_cids", ()) or ()),
        material.bounded_stdout_artifact_cid,
        material.bounded_stderr_artifact_cid,
        *material.extra_artifact_cids,
        *extra,
    ):
        text = str(cid or "").strip()
        if text:
            result.append(text)
    return result


def _quality_score(
    *,
    guarantee: MinimizationGuarantee,
    frames_before: int,
    frames_after: int,
    log_before: int,
    log_after: int,
    input_before: int,
    input_after: int,
    minimized: bool,
) -> float:
    base = {
        MinimizationGuarantee.NONE: 0.0,
        MinimizationGuarantee.NORMALIZED: 0.25,
        MinimizationGuarantee.BOUNDED: 0.45,
        MinimizationGuarantee.RERUN_VALIDATED: 0.75,
    }[guarantee]
    reduction = 0.0
    if frames_before > 0:
        reduction += 0.1 * max(0.0, 1.0 - (frames_after / frames_before))
    if log_before > 0:
        reduction += 0.1 * max(0.0, 1.0 - (log_after / log_before))
    if input_before > 0:
        reduction += 0.05 * max(0.0, 1.0 - (input_after / input_before))
    bonus = 0.1 if minimized else 0.0
    score = base + reduction + bonus
    if score > 1.0:
        return 1.0
    if score < 0.0:
        return 0.0
    return round(score, 4)


__all__ = [
    "ALGORITHM_VERSION",
    "COUNTEREXAMPLE_EVIDENCE",
    "COUNTEREXAMPLE_MINIMIZER_INTERFACE",
    "COUNTEREXAMPLE_MINIMIZER_SCHEMA",
    "COUNTEREXAMPLE_MINIMIZATION_RESULT_SCHEMA",
    "FAILURE_IDENTITY_SCHEMA",
    "MINIMIZATION_QUALITY_SCHEMA",
    "CounterexampleMinimizationError",
    "CounterexampleMinimizationResult",
    "CounterexampleMinimizer",
    "FailureMaterial",
    "MinimizationBudget",
    "MinimizationGuarantee",
    "MinimizationQuality",
    "MinimizationRequest",
    "RerunObservation",
    "RerunOracle",
    "build_reproduction_argv_candidates",
    "compute_failure_identity_cid",
    "diagnostic_not_applicable",
    "diagnostic_present",
    "diagnostic_redacted",
    "diagnostic_unavailable",
    "extract_failure_material_from_pytest_output",
    "is_private_field_name",
    "minimize_counterexample",
    "minimize_source_spans",
    "prune_log_lines",
    "rerun_observation_from_run_result",
    "sanitize_diagnostic_value",
    "slice_traceback",
]
