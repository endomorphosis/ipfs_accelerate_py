"""Pytest verification adapter for incremental verification (IVP-005).

``PytestVerificationAdapter`` executes exact selected node IDs or an explicit
full-suite oracle mode through the admitted :class:`VerificationProcessRunner`.
It never interpolates a shell string, never installs packages, and never
upgrades a timeout, cancellation, or unavailability into a pass.

Authority rules
---------------
* Explicit ``python -m pytest`` argv is the only execution form.
* Selector, config, observed environment, and fixture CIDs must bind the
  supplied :class:`VerificationReceiptKey`.
* Setup / call / teardown phase outcomes are accounted and project terminal
  status; empty collection, usage errors, and malformed reports are
  ``invalid``.
* Required skip/xfail is ``not_modeled`` unless predeclared advisory.
* Unexpected xpass or collection/setup/teardown failure cannot pass.
* Existing :class:`TestPassReceipt` / :class:`TestExecutionKey` projection is
  authoritative only when the direct observation is a true pass.
* Simulated mode is retained as ``simulated`` and never satisfies production.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Optional

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_bytes,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    PhaseOutcome,
    TestExecutionKey,
    TestPassReceipt,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    DirectExecutionObservation,
    TerminalStatus,
    TestReceipt,
    VerificationContractError,
    VerificationIdentityError,
    VerificationReceiptKey,
    VerificationReceiptKind,
)
from ipfs_accelerate_py.agent_supervisor.verification.process_runner import (
    NETWORK_POLICY_DENY_ALL,
    PROCESS_RUNNER_EVIDENCE,
    PROCESS_TREE_CANCELLATION_EVIDENCE,
    VerificationCancellation,
    VerificationCommand,
    VerificationProcessRunner,
    VerificationProcessRunnerError,
    VerificationRunDisposition,
    VerificationRunResult,
    VerificationSandboxIdentity,
    build_hermetic_environment,
)

# ---------------------------------------------------------------------------
# Interface / schema constants
# ---------------------------------------------------------------------------

PYTEST_VERIFICATION_ADAPTER_INTERFACE: Final[str] = "PytestVerificationAdapter@1"
PYTEST_VERIFICATION_ADAPTER_SCHEMA: Final[str] = "pytest-verification-adapter@1"
PYTEST_ADAPTER_EVIDENCE: Final[str] = "ivp/pytest-adapter@1"
PYTEST_PHASE_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/pytest-phase-report@1"
)
PYTEST_PHASE_ACCOUNTING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/pytest-phase-accounting@1"
)
DEFAULT_PHASE_REPORT_RELPATH: Final[str] = "pytest-phase-report.json"

_PYTEST_NODE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[^\x00\r\n:]+\.py(?:::[^\x00\r\n]+)+$"
)
_MAX_NODE_IDS: Final[int] = 10_000
_MAX_NODE_ID_CHARS: Final[int] = 2_048
_MAX_CONFIG_ARGS: Final[int] = 256
_MAX_REASON_CODES: Final[int] = 64

_PHASE_OUTCOME_ALIASES: Final[Mapping[str, PhaseOutcome]] = MappingProxyType(
    {
        "pass": PhaseOutcome.PASS,
        "passed": PhaseOutcome.PASS,
        "fail": PhaseOutcome.FAIL,
        "failed": PhaseOutcome.FAIL,
        "skip": PhaseOutcome.SKIP,
        "skipped": PhaseOutcome.SKIP,
        "xfail": PhaseOutcome.XFAIL,
        "xfailed": PhaseOutcome.XFAIL,
        "xpass": PhaseOutcome.XPASS,
        "xpassed": PhaseOutcome.XPASS,
        "error": PhaseOutcome.ERROR,
        "not_run": PhaseOutcome.NOT_RUN,
        "notset": PhaseOutcome.NOT_RUN,
        "interrupted": PhaseOutcome.INTERRUPTED,
        "interrupt": PhaseOutcome.INTERRUPTED,
        "rerun": PhaseOutcome.RERUN,
    }
)

_FAILING_PHASES: Final[frozenset[PhaseOutcome]] = frozenset(
    {
        PhaseOutcome.FAIL,
        PhaseOutcome.ERROR,
        PhaseOutcome.INTERRUPTED,
        PhaseOutcome.RERUN,
        PhaseOutcome.NOT_RUN,
    }
)
_SKIP_LIKE: Final[frozenset[PhaseOutcome]] = frozenset(
    {PhaseOutcome.SKIP, PhaseOutcome.XFAIL}
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PytestVerificationAdapterError(ValueError):
    """Fail-closed adapter contract violation (pre-execution)."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "invalid_request",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


# ---------------------------------------------------------------------------
# Request / result types
# ---------------------------------------------------------------------------


class PytestRunMode(str, Enum):
    """Closed execution modes for the pytest adapter."""

    SELECTED_NODES = "selected_nodes"
    FULL_SUITE_ORACLE = "full_suite_oracle"


@dataclass(frozen=True)
class PytestAdvisoryPolicy:
    """Predeclared advisory skip/xfail allowances (never upgrades failures).

    A required skip/xfail remains ``not_modeled`` unless the node id or one of
    its markers is listed here *before* execution.
    """

    node_ids: tuple[str, ...] = ()
    markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        nodes = _normalize_node_ids(self.node_ids, field_name="advisory.node_ids")
        markers = tuple(
            str(item).strip()
            for item in (self.markers or ())
            if str(item).strip()
        )
        if len(markers) > _MAX_NODE_IDS:
            raise PytestVerificationAdapterError(
                "advisory markers exceed bound",
                reason_code="bounds_exceeded",
            )
        object.__setattr__(self, "node_ids", nodes)
        object.__setattr__(self, "markers", markers)

    def allows(self, nodeid: str, markers: Sequence[str]) -> bool:
        if nodeid in self.node_ids:
            return True
        if not self.markers:
            return False
        present = {str(item).strip() for item in markers if str(item).strip()}
        return bool(present.intersection(self.markers))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_ids": list(self.node_ids),
            "markers": list(self.markers),
        }


@dataclass(frozen=True)
class PytestNodePhaseAccounting:
    """Setup/call/teardown accounting for one collected node."""

    __test__ = False

    nodeid: str
    setup: PhaseOutcome
    call: PhaseOutcome
    teardown: PhaseOutcome
    markers: tuple[str, ...] = ()
    wasxfail: bool = False
    advisory: bool = False
    duration_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodeid", str(self.nodeid).strip())
        object.__setattr__(self, "setup", _coerce_phase(self.setup, "setup"))
        object.__setattr__(self, "call", _coerce_phase(self.call, "call"))
        object.__setattr__(
            self, "teardown", _coerce_phase(self.teardown, "teardown")
        )
        object.__setattr__(
            self,
            "markers",
            tuple(str(item).strip() for item in (self.markers or ()) if str(item).strip()),
        )
        object.__setattr__(self, "wasxfail", bool(self.wasxfail))
        object.__setattr__(self, "advisory", bool(self.advisory))
        duration = self.duration_ms
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
            duration = 0
        object.__setattr__(self, "duration_ms", duration)

    @property
    def all_phases_pass(self) -> bool:
        return (
            self.setup is PhaseOutcome.PASS
            and self.call is PhaseOutcome.PASS
            and self.teardown is PhaseOutcome.PASS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodeid": self.nodeid,
            "setup": self.setup.value,
            "call": self.call.value,
            "teardown": self.teardown.value,
            "markers": list(self.markers),
            "wasxfail": self.wasxfail,
            "advisory": self.advisory,
            "duration_ms": self.duration_ms,
            "all_phases_pass": self.all_phases_pass,
        }


@dataclass(frozen=True)
class PytestVerificationRequest:
    """One admitted pytest verification request.

    *receipt_key* must already bind the selector argv that
    :meth:`PytestVerificationAdapter.build_argv` will emit for this request
    (tool name ``pytest``, matching configuration and fixture CIDs).
    """

    __test__ = False

    receipt_key: VerificationReceiptKey
    mode: PytestRunMode
    python_executable: str
    sandbox: VerificationSandboxIdentity
    cwd: str
    timeout_seconds: float
    node_ids: Sequence[str] = ()
    suite_paths: Sequence[str] = ()
    config_args: Sequence[str] = ()
    extra_pytest_args: Sequence[str] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    advisory: PytestAdvisoryPolicy = field(default_factory=PytestAdvisoryPolicy)
    existing_test_pass_receipt: TestPassReceipt | None = None
    existing_test_execution_key: TestExecutionKey | None = None
    simulated: bool = False
    network_policy: str = NETWORK_POLICY_DENY_ALL
    max_stdout_bytes: int = 256 * 1024
    max_stderr_bytes: int = 256 * 1024
    phase_report_relpath: str = DEFAULT_PHASE_REPORT_RELPATH
    lane_id: str = ""
    resource_class: str = "cpu-validation"
    stage: str = "validation"
    metadata: Mapping[str, str] = field(default_factory=dict)
    # When set, used instead of reading the artifact/stdout report (tests inject).
    injected_phase_report: Mapping[str, Any] | bytes | str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.receipt_key, VerificationReceiptKey):
            raise PytestVerificationAdapterError(
                "receipt_key must be a VerificationReceiptKey",
                reason_code="invalid_receipt_key",
            )
        if self.receipt_key.receipt_kind is not VerificationReceiptKind.TEST:
            raise PytestVerificationAdapterError(
                "receipt_key must be receipt_kind=test",
                reason_code="invalid_receipt_kind",
            )
        if self.receipt_key.tool_name != "pytest":
            raise PytestVerificationAdapterError(
                "receipt_key.tool_name must be pytest",
                reason_code="invalid_tool",
            )
        if self.receipt_key.adapter_schema != PYTEST_VERIFICATION_ADAPTER_SCHEMA:
            raise PytestVerificationAdapterError(
                "receipt_key.adapter_schema must be pytest-verification-adapter@1",
                reason_code="invalid_adapter_schema",
            )
        mode = (
            self.mode
            if isinstance(self.mode, PytestRunMode)
            else PytestRunMode(str(self.mode))
        )
        object.__setattr__(self, "mode", mode)
        python = str(self.python_executable or "").strip()
        if not python:
            raise PytestVerificationAdapterError(
                "python_executable is required",
                reason_code="invalid_python",
            )
        if not PurePosixPath(python).is_absolute() and not Path(python).is_absolute():
            # Allow non-posix absolute (Windows) via Path.
            if not Path(python).expanduser().is_absolute():
                raise PytestVerificationAdapterError(
                    "python_executable must be absolute",
                    reason_code="invalid_python",
                )
        object.__setattr__(self, "python_executable", python)
        if not isinstance(self.sandbox, VerificationSandboxIdentity):
            raise PytestVerificationAdapterError(
                "sandbox must be a VerificationSandboxIdentity",
                reason_code="sandbox_unavailable",
            )
        cwd = str(self.cwd or "").strip()
        if not cwd:
            raise PytestVerificationAdapterError(
                "cwd is required",
                reason_code="invalid_cwd",
            )
        object.__setattr__(self, "cwd", cwd)
        timeout = float(self.timeout_seconds)
        if not (timeout > 0.0):
            raise PytestVerificationAdapterError(
                "timeout_seconds must be positive",
                reason_code="invalid_timeout",
            )
        object.__setattr__(self, "timeout_seconds", timeout)
        nodes = _normalize_node_ids(self.node_ids, field_name="node_ids")
        object.__setattr__(self, "node_ids", nodes)
        suites = tuple(
            str(item).strip()
            for item in (self.suite_paths or ())
            if str(item).strip()
        )
        object.__setattr__(self, "suite_paths", suites)
        config_args = _normalize_arg_sequence(
            self.config_args, field_name="config_args"
        )
        object.__setattr__(self, "config_args", config_args)
        extra = _normalize_arg_sequence(
            self.extra_pytest_args, field_name="extra_pytest_args"
        )
        object.__setattr__(self, "extra_pytest_args", extra)
        env = {
            str(key): str(value)
            for key, value in dict(self.environment or {}).items()
        }
        object.__setattr__(self, "environment", MappingProxyType(env))
        advisory = self.advisory
        if not isinstance(advisory, PytestAdvisoryPolicy):
            raise PytestVerificationAdapterError(
                "advisory must be a PytestAdvisoryPolicy",
                reason_code="invalid_advisory",
            )
        if (self.existing_test_pass_receipt is None) != (
            self.existing_test_execution_key is None
        ):
            raise PytestVerificationAdapterError(
                "existing test bridge requires both TestPassReceipt and TestExecutionKey",
                reason_code="invalid_existing_bridge",
            )
        object.__setattr__(self, "simulated", bool(self.simulated))
        network = str(self.network_policy or "").strip() or NETWORK_POLICY_DENY_ALL
        if network != NETWORK_POLICY_DENY_ALL:
            raise PytestVerificationAdapterError(
                "network policy must be deny_all",
                reason_code="network_policy_denied",
            )
        object.__setattr__(self, "network_policy", network)
        report_path = str(self.phase_report_relpath or DEFAULT_PHASE_REPORT_RELPATH).strip()
        if (
            not report_path
            or report_path.startswith("/")
            or ".." in PurePosixPath(report_path).parts
            or "\x00" in report_path
        ):
            raise PytestVerificationAdapterError(
                "phase_report_relpath must be a sandbox-relative path",
                reason_code="invalid_phase_report_path",
            )
        object.__setattr__(self, "phase_report_relpath", report_path)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                {str(k): str(v) for k, v in dict(self.metadata or {}).items()}
            ),
        )
        if mode is PytestRunMode.SELECTED_NODES and not nodes:
            raise PytestVerificationAdapterError(
                "selected_nodes mode requires at least one node id",
                reason_code="empty_selector",
            )


@dataclass(frozen=True)
class PytestVerificationResult:
    """Observed pytest verification outcome with retained argv and artifacts."""

    __test__ = False

    terminal_status: TerminalStatus
    receipt: TestReceipt | None
    command_argv: tuple[str, ...]
    mode: PytestRunMode
    phase_accounting: tuple[PytestNodePhaseAccounting, ...]
    collected_count: int
    artifact_cids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    production_admissible: bool
    simulated: bool
    run_result: VerificationRunResult | None
    phase_report_cid: str
    evidence: tuple[str, ...] = (
        PYTEST_ADAPTER_EVIDENCE,
        PROCESS_RUNNER_EVIDENCE,
        PROCESS_TREE_CANCELLATION_EVIDENCE,
    )
    duration_ms: int = 0
    exit_code: int | None = None
    publication_allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terminal_status",
            TerminalStatus(self.terminal_status),
        )
        object.__setattr__(
            self,
            "mode",
            self.mode
            if isinstance(self.mode, PytestRunMode)
            else PytestRunMode(str(self.mode)),
        )
        object.__setattr__(
            self,
            "command_argv",
            tuple(str(item) for item in self.command_argv),
        )
        object.__setattr__(
            self,
            "phase_accounting",
            tuple(self.phase_accounting),
        )
        object.__setattr__(
            self,
            "artifact_cids",
            tuple(str(item) for item in self.artifact_cids if str(item).strip()),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(str(item) for item in self.reason_codes if str(item).strip()),
        )
        object.__setattr__(self, "production_admissible", bool(self.production_admissible))
        object.__setattr__(self, "simulated", bool(self.simulated))
        object.__setattr__(self, "publication_allowed", bool(self.publication_allowed))
        object.__setattr__(
            self,
            "evidence",
            tuple(str(item) for item in self.evidence),
        )

    @property
    def ok(self) -> bool:
        return (
            self.production_admissible
            and self.terminal_status is TerminalStatus.PASSED
            and self.receipt is not None
            and self.receipt.terminal_success
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PYTEST_VERIFICATION_ADAPTER_SCHEMA,
            "interface": PYTEST_VERIFICATION_ADAPTER_INTERFACE,
            "evidence": list(self.evidence),
            "terminal_status": self.terminal_status.value,
            "receipt": self.receipt.to_record() if self.receipt is not None else None,
            "command_argv": list(self.command_argv),
            "mode": self.mode.value,
            "phase_accounting": [item.to_dict() for item in self.phase_accounting],
            "collected_count": self.collected_count,
            "artifact_cids": list(self.artifact_cids),
            "reason_codes": list(self.reason_codes),
            "production_admissible": self.production_admissible,
            "simulated": self.simulated,
            "phase_report_cid": self.phase_report_cid,
            "duration_ms": self.duration_ms,
            "exit_code": self.exit_code,
            "publication_allowed": self.publication_allowed,
            "ok": self.ok,
            "run_result": self.run_result.to_dict() if self.run_result else None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_node_ids(values: Sequence[str] | None, *, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PytestVerificationAdapterError(
            f"{field_name} must be a sequence of node ids",
            reason_code="invalid_node_ids",
        )
    if len(values) > _MAX_NODE_IDS:
        raise PytestVerificationAdapterError(
            f"{field_name} exceeds {_MAX_NODE_IDS} items",
            reason_code="bounds_exceeded",
        )
    ordered: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        if not isinstance(raw, str):
            raise PytestVerificationAdapterError(
                f"{field_name}[{index}] must be a string",
                reason_code="invalid_node_ids",
            )
        node = raw.strip()
        if not node:
            raise PytestVerificationAdapterError(
                f"{field_name}[{index}] must not be empty",
                reason_code="invalid_node_ids",
            )
        if "\x00" in node or "\n" in node or "\r" in node:
            raise PytestVerificationAdapterError(
                f"{field_name}[{index}] contains control characters",
                reason_code="invalid_node_ids",
            )
        if len(node) > _MAX_NODE_ID_CHARS:
            raise PytestVerificationAdapterError(
                f"{field_name}[{index}] exceeds {_MAX_NODE_ID_CHARS} characters",
                reason_code="bounds_exceeded",
            )
        if not _PYTEST_NODE_RE.fullmatch(node):
            raise PytestVerificationAdapterError(
                f"{field_name}[{index}] is not a pytest node id",
                reason_code="invalid_node_ids",
                details={"nodeid": node},
            )
        if node in seen:
            continue
        seen.add(node)
        ordered.append(node)
    return tuple(ordered)


def _normalize_arg_sequence(
    values: Sequence[str] | None, *, field_name: str
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PytestVerificationAdapterError(
            f"{field_name} must be a non-string sequence of strings",
            reason_code="invalid_args",
        )
    if len(values) > _MAX_CONFIG_ARGS:
        raise PytestVerificationAdapterError(
            f"{field_name} exceeds {_MAX_CONFIG_ARGS} items",
            reason_code="bounds_exceeded",
        )
    result: list[str] = []
    for index, raw in enumerate(values):
        if not isinstance(raw, str):
            raise PytestVerificationAdapterError(
                f"{field_name}[{index}] must be a string",
                reason_code="invalid_args",
            )
        if "\x00" in raw:
            raise PytestVerificationAdapterError(
                f"{field_name}[{index}] must not contain NUL",
                reason_code="invalid_args",
            )
        result.append(raw)
    return tuple(result)


def _coerce_phase(value: Any, field_name: str) -> PhaseOutcome:
    if isinstance(value, PhaseOutcome):
        return value
    text = str(value or "").strip().lower()
    if text in _PHASE_OUTCOME_ALIASES:
        return _PHASE_OUTCOME_ALIASES[text]
    raise PytestVerificationAdapterError(
        f"unknown phase outcome for {field_name}: {value!r}",
        reason_code="malformed_phase_report",
    )


def build_pytest_argv(
    *,
    python_executable: str,
    mode: PytestRunMode,
    node_ids: Sequence[str] = (),
    suite_paths: Sequence[str] = (),
    config_args: Sequence[str] = (),
    extra_pytest_args: Sequence[str] = (),
    phase_report_relpath: str = DEFAULT_PHASE_REPORT_RELPATH,
) -> tuple[str, ...]:
    """Build the explicit reproducible ``python -m pytest`` argv sequence."""

    python = str(python_executable or "").strip()
    if not python:
        raise PytestVerificationAdapterError(
            "python_executable is required",
            reason_code="invalid_python",
        )
    if isinstance(python, str) and (" " in python and not Path(python).exists()):
        # Spaces are legal in absolute paths; shell metacharacters are single argv items.
        pass
    mode = mode if isinstance(mode, PytestRunMode) else PytestRunMode(str(mode))
    nodes = _normalize_node_ids(node_ids, field_name="node_ids") if node_ids else ()
    suites = tuple(str(item).strip() for item in suite_paths if str(item).strip())
    config = _normalize_arg_sequence(config_args, field_name="config_args")
    extra = _normalize_arg_sequence(extra_pytest_args, field_name="extra_pytest_args")
    report = str(phase_report_relpath or DEFAULT_PHASE_REPORT_RELPATH).strip()

    argv: list[str] = [python, "-m", "pytest"]
    argv.extend(config)
    # Reproducible defaults that keep collection bounded and reports local.
    # Phase-report path is bound on the request (artifact_root-relative) and is
    # not a pytest CLI flag, so real pytest invocations remain valid.
    default_flags = (
        "-q",
        "--tb=short",
        "-p",
        "no:cacheprovider",
    )
    present = set(config) | set(extra)
    for flag in default_flags:
        if flag not in present:
            argv.append(flag)
    # Preserve the phase-report relative path in argv as a reproducible comment
    # token only when the caller opts in via extra args; otherwise metadata
    # alone retains the binding on the request/result artifacts.
    _ = report  # bound by request; artifact path is not shell-interpolated
    argv.extend(extra)
    if mode is PytestRunMode.SELECTED_NODES:
        if not nodes:
            raise PytestVerificationAdapterError(
                "selected_nodes mode requires node ids",
                reason_code="empty_selector",
            )
        argv.extend(nodes)
    else:
        argv.extend(suites)
    return tuple(argv)


def encode_phase_report(
    *,
    items: Sequence[Mapping[str, Any]] | Sequence[PytestNodePhaseAccounting],
    collected: int | None = None,
    collection_errors: Sequence[str] = (),
    usage_error: bool = False,
    malformed: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical phase-report mapping for injection or persistence."""

    serialized_items: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, PytestNodePhaseAccounting):
            serialized_items.append(
                {
                    "nodeid": item.nodeid,
                    "setup": item.setup.value,
                    "call": item.call.value,
                    "teardown": item.teardown.value,
                    "markers": list(item.markers),
                    "wasxfail": item.wasxfail,
                    "duration_ms": item.duration_ms,
                }
            )
        else:
            serialized_items.append(dict(item))
    payload: dict[str, Any] = {
        "schema": PYTEST_PHASE_REPORT_SCHEMA,
        "collected": int(collected if collected is not None else len(serialized_items)),
        "items": serialized_items,
        "collection_errors": list(collection_errors),
        "usage_error": bool(usage_error),
        "malformed": bool(malformed),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def parse_phase_report(
    source: Mapping[str, Any] | bytes | str | None,
    *,
    advisory: PytestAdvisoryPolicy | None = None,
) -> tuple[tuple[PytestNodePhaseAccounting, ...], int, tuple[str, ...], bool, bool]:
    """Parse a phase report into accounting rows.

    Returns ``(items, collected, collection_errors, usage_error, malformed)``.
    """

    reasons: list[str] = []
    if source is None:
        return (), 0, (), False, True
    payload: Any
    if isinstance(source, (bytes, bytearray)):
        try:
            text = bytes(source).decode("utf-8")
        except UnicodeDecodeError:
            return (), 0, (), False, True
        source = text
    if isinstance(source, str):
        text = source.strip()
        if not text:
            return (), 0, (), False, True
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # Attempt to locate a trailing JSON object in mixed stdout.
            start = text.rfind("{")
            if start < 0:
                return (), 0, (), False, True
            try:
                payload = json.loads(text[start:])
            except json.JSONDecodeError:
                return (), 0, (), False, True
    elif isinstance(source, Mapping):
        payload = dict(source)
    else:
        return (), 0, (), False, True

    if not isinstance(payload, Mapping):
        return (), 0, (), False, True
    schema = str(payload.get("schema") or "").strip()
    if schema and schema != PYTEST_PHASE_REPORT_SCHEMA:
        return (), 0, (), False, True
    if payload.get("malformed") is True:
        return (), 0, (), False, True
    usage_error = bool(payload.get("usage_error"))
    try:
        collected = int(payload.get("collected", 0))
    except (TypeError, ValueError):
        return (), 0, (), False, True
    if collected < 0:
        return (), 0, (), False, True
    raw_errors = payload.get("collection_errors") or ()
    if isinstance(raw_errors, str):
        collection_errors = (raw_errors,) if raw_errors.strip() else ()
    elif isinstance(raw_errors, Sequence):
        collection_errors = tuple(str(item) for item in raw_errors if str(item).strip())
    else:
        return (), 0, (), False, True
    raw_items = payload.get("items")
    if raw_items is None:
        raw_items = ()
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return (), 0, (), False, True

    policy = advisory or PytestAdvisoryPolicy()
    items: list[PytestNodePhaseAccounting] = []
    try:
        for index, raw in enumerate(raw_items):
            if not isinstance(raw, Mapping):
                return (), collected, collection_errors, usage_error, True
            nodeid = str(raw.get("nodeid") or "").strip()
            if not nodeid:
                return (), collected, collection_errors, usage_error, True
            markers = tuple(
                str(item).strip()
                for item in (raw.get("markers") or ())
                if str(item).strip()
            )
            wasxfail = bool(raw.get("wasxfail"))
            setup = _coerce_phase(raw.get("setup", "not_run"), f"items[{index}].setup")
            call = _coerce_phase(raw.get("call", "not_run"), f"items[{index}].call")
            teardown = _coerce_phase(
                raw.get("teardown", "not_run"), f"items[{index}].teardown"
            )
            # XPass/xfail may also be encoded via wasxfail + outcome.
            if wasxfail and call is PhaseOutcome.PASS:
                call = PhaseOutcome.XPASS
            elif wasxfail and call is PhaseOutcome.FAIL:
                call = PhaseOutcome.XFAIL
            duration = raw.get("duration_ms", 0)
            try:
                duration_ms = int(duration)
            except (TypeError, ValueError):
                duration_ms = 0
            items.append(
                PytestNodePhaseAccounting(
                    nodeid=nodeid,
                    setup=setup,
                    call=call,
                    teardown=teardown,
                    markers=markers,
                    wasxfail=wasxfail,
                    advisory=policy.allows(nodeid, markers),
                    duration_ms=max(0, duration_ms),
                )
            )
    except PytestVerificationAdapterError:
        return (), collected, collection_errors, usage_error, True

    return tuple(items), collected, collection_errors, usage_error, False


def project_terminal_status(
    *,
    run_result: VerificationRunResult | None,
    phase_items: Sequence[PytestNodePhaseAccounting],
    collected: int,
    collection_errors: Sequence[str],
    usage_error: bool,
    malformed: bool,
    mode: PytestRunMode,
    selected_node_ids: Sequence[str],
    simulated: bool,
) -> tuple[TerminalStatus, tuple[str, ...]]:
    """Project closed terminal status from runner + phase accounting."""

    reasons: list[str] = []

    if simulated:
        reasons.append("simulated_mode")
        return TerminalStatus.SIMULATED, tuple(reasons)

    if run_result is not None:
        if run_result.timed_out or run_result.disposition is VerificationRunDisposition.TIMEOUT:
            reasons.append("timeout")
            reasons.extend(run_result.reason_codes)
            return TerminalStatus.TIMEOUT, _unique_reasons(reasons)
        if run_result.cancelled or run_result.disposition is VerificationRunDisposition.CANCELLED:
            reasons.append("cancelled")
            reasons.extend(run_result.reason_codes)
            return TerminalStatus.CANCELLED, _unique_reasons(reasons)
        if (
            run_result.unavailable
            or run_result.disposition is VerificationRunDisposition.UNAVAILABLE
        ):
            reasons.append("unavailable")
            reasons.extend(run_result.reason_codes)
            return TerminalStatus.UNAVAILABLE, _unique_reasons(reasons)

    if usage_error:
        reasons.append("usage_error")
        return TerminalStatus.INVALID, tuple(reasons)
    if malformed:
        reasons.append("malformed_output")
        return TerminalStatus.INVALID, tuple(reasons)
    if collection_errors:
        reasons.append("collection_failure")
        reasons.extend(
            f"collection_error:{_node_reason_token(err)}"
            for err in collection_errors[:8]
        )
        return TerminalStatus.FAILED, _unique_reasons(reasons)
    if collected == 0 or not phase_items:
        reasons.append("empty_collection")
        return TerminalStatus.INVALID, tuple(reasons)

    if mode is PytestRunMode.SELECTED_NODES:
        observed = {item.nodeid for item in phase_items}
        missing = [node for node in selected_node_ids if node not in observed]
        if missing:
            reasons.append("missing_selected_nodes")
            reasons.extend(
                f"missing:{_node_reason_token(node)}" for node in missing[:8]
            )
            return TerminalStatus.INVALID, _unique_reasons(reasons)

    has_failure = False
    has_not_modeled = False
    has_required_pass = False

    for item in phase_items:
        node_token = _node_reason_token(item.nodeid)
        # Skip/xfail short-circuit: setup skip or call skip/xfail.  Downstream
        # phases are expected to be not_run and must not project as failures.
        if item.setup is PhaseOutcome.SKIP or item.call in _SKIP_LIKE:
            if item.advisory:
                reasons.append(f"advisory_skip_or_xfail:{node_token}")
                continue
            has_not_modeled = True
            reasons.append(f"required_skip_or_xfail:{node_token}")
            continue
        # Collection/setup/teardown failures cannot pass.
        if item.setup in _FAILING_PHASES:
            has_failure = True
            reasons.append(f"setup_failure:{node_token}")
            continue
        if item.call is PhaseOutcome.XPASS:
            has_failure = True
            reasons.append(f"unexpected_xpass:{node_token}")
            continue
        if item.call in {PhaseOutcome.FAIL, PhaseOutcome.ERROR, PhaseOutcome.INTERRUPTED}:
            has_failure = True
            reasons.append(f"call_failure:{node_token}")
            continue
        if item.teardown in _FAILING_PHASES:
            has_failure = True
            reasons.append(f"teardown_failure:{node_token}")
            continue
        if item.all_phases_pass:
            if not item.advisory:
                has_required_pass = True
            continue
        # Any other non-pass combination is a failure.
        has_failure = True
        reasons.append(f"phase_incomplete:{node_token}")

    if has_failure:
        reasons.insert(0, "failed")
        return TerminalStatus.FAILED, _unique_reasons(reasons)
    if has_not_modeled:
        reasons.insert(0, "not_modeled")
        return TerminalStatus.NOT_MODELED, _unique_reasons(reasons)

    # Require at least one non-advisory full pass for production success.
    if not has_required_pass:
        # Full suite that only hit advisory skips is not a production pass.
        reasons.append("no_required_pass")
        return TerminalStatus.NOT_MODELED, _unique_reasons(reasons)

    if run_result is not None and run_result.exit_code not in (0, None):
        # Exit non-zero without phase failure evidence is still a failure.
        if run_result.exit_code != 0:
            reasons.append("nonzero_exit")
            return TerminalStatus.FAILED, _unique_reasons(reasons)

    reasons.append("all_required_phases_passed")
    return TerminalStatus.PASSED, _unique_reasons(reasons)


_REASON_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z][a-z0-9_.:/+-]{0,127}$"
)


def _node_reason_token(nodeid: str) -> str:
    """Map free-form node ids / messages into a closed reason-token fragment."""

    text = str(nodeid or "").strip().lower()
    # Replace characters outside the closed token alphabet.
    cleaned = re.sub(r"[^a-z0-9_.:/+-]+", "_", text)
    cleaned = cleaned.strip("._:-/+") or "unknown"
    if not cleaned[0].isalpha():
        cleaned = "n_" + cleaned
    return cleaned[:96]


def _sanitize_reason_token(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    if _REASON_TOKEN_RE.fullmatch(text) and len(text) <= 128:
        return text
    cleaned = re.sub(r"[^a-z0-9_.:/+-]+", "_", text)
    cleaned = cleaned.strip("._:-/+") or "reason"
    if not cleaned[0].isalpha():
        cleaned = "r_" + cleaned
    return cleaned[:128]


def _unique_reasons(reasons: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in reasons:
        text = _sanitize_reason_token(str(raw or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
        if len(ordered) >= _MAX_REASON_CODES:
            break
    return tuple(ordered)


def _exit_code_for_status(status: TerminalStatus, run_result: VerificationRunResult | None) -> int | None:
    if run_result is not None and run_result.exit_code is not None:
        if status in {
            TerminalStatus.TIMEOUT,
            TerminalStatus.CANCELLED,
            TerminalStatus.UNAVAILABLE,
        }:
            return run_result.exit_code
        if status is TerminalStatus.PASSED:
            return 0
        return run_result.exit_code if run_result.exit_code != 0 else 1
    if status is TerminalStatus.PASSED:
        return 0
    if status in {
        TerminalStatus.TIMEOUT,
        TerminalStatus.CANCELLED,
        TerminalStatus.UNAVAILABLE,
        TerminalStatus.SIMULATED,
    }:
        return None
    return 1


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class PytestVerificationAdapter:
    """Execute selected pytest nodes or a full-suite oracle via the shared runner."""

    interface: Final[str] = PYTEST_VERIFICATION_ADAPTER_INTERFACE
    schema: Final[str] = PYTEST_VERIFICATION_ADAPTER_SCHEMA
    evidence: Final[str] = PYTEST_ADAPTER_EVIDENCE

    def __init__(
        self,
        process_runner: VerificationProcessRunner | None = None,
        *,
        require_production: bool = True,
    ) -> None:
        self._runner = process_runner or VerificationProcessRunner()
        self._require_production = bool(require_production)

    @property
    def process_runner(self) -> VerificationProcessRunner:
        return self._runner

    def build_argv(self, request: PytestVerificationRequest) -> tuple[str, ...]:
        """Return the reproducible explicit ``python -m pytest`` argv list."""

        if not isinstance(request, PytestVerificationRequest):
            raise PytestVerificationAdapterError(
                "request must be a PytestVerificationRequest",
                reason_code="invalid_request",
            )
        return build_pytest_argv(
            python_executable=request.python_executable,
            mode=request.mode,
            node_ids=request.node_ids,
            suite_paths=request.suite_paths,
            config_args=request.config_args,
            extra_pytest_args=request.extra_pytest_args,
            phase_report_relpath=request.phase_report_relpath,
        )

    def execute(
        self,
        request: PytestVerificationRequest,
        *,
        cancellation: VerificationCancellation | None = None,
    ) -> PytestVerificationResult:
        """Run pytest (or project from injected report) and emit a TestReceipt."""

        if not isinstance(request, PytestVerificationRequest):
            raise PytestVerificationAdapterError(
                "request must be a PytestVerificationRequest",
                reason_code="invalid_request",
            )
        argv = self.build_argv(request)
        self._validate_selector_binding(request, argv)

        if request.simulated:
            return self._finalize(
                request=request,
                argv=argv,
                run_result=None,
                phase_items=(),
                collected=0,
                collection_errors=(),
                usage_error=False,
                malformed=False,
                forced_status=TerminalStatus.SIMULATED,
                extra_reasons=("simulated_mode",),
            )

        run_result: VerificationRunResult | None = None
        if request.injected_phase_report is None:
            command = self._build_command(request, argv)
            try:
                run_result = self._runner.run(command, cancellation=cancellation)
            except VerificationProcessRunnerError as exc:
                return self._finalize(
                    request=request,
                    argv=argv,
                    run_result=None,
                    phase_items=(),
                    collected=0,
                    collection_errors=(),
                    usage_error=False,
                    malformed=False,
                    forced_status=TerminalStatus.UNAVAILABLE,
                    extra_reasons=(
                        getattr(exc, "reason_code", None) or "runner_error",
                        "unavailable",
                    ),
                )
            # Preserve runner-terminal outcomes before report parsing.
            if (
                run_result.timed_out
                or run_result.cancelled
                or run_result.unavailable
                or run_result.disposition
                in {
                    VerificationRunDisposition.TIMEOUT,
                    VerificationRunDisposition.CANCELLED,
                    VerificationRunDisposition.UNAVAILABLE,
                }
            ):
                return self._finalize(
                    request=request,
                    argv=tuple(run_result.command_argv) or argv,
                    run_result=run_result,
                    phase_items=(),
                    collected=0,
                    collection_errors=(),
                    usage_error=False,
                    malformed=False,
                    forced_status=None,
                    extra_reasons=(),
                )
            observed_argv = tuple(run_result.command_argv) or argv
            report_source = self._load_phase_report(request, run_result)
        else:
            observed_argv = argv
            report_source = request.injected_phase_report

        phase_items, collected, collection_errors, usage_error, malformed = (
            parse_phase_report(report_source, advisory=request.advisory)
        )
        return self._finalize(
            request=request,
            argv=observed_argv,
            run_result=run_result,
            phase_items=phase_items,
            collected=collected,
            collection_errors=collection_errors,
            usage_error=usage_error,
            malformed=malformed,
            forced_status=None,
            extra_reasons=(),
        )

    # -- internals ---------------------------------------------------------

    def _validate_selector_binding(
        self,
        request: PytestVerificationRequest,
        argv: Sequence[str],
    ) -> None:
        key = request.receipt_key
        # Fixture and configuration identity bindings.
        if tuple(key.fixture_data_cids) != tuple(key.fixture_data_cids):
            raise PytestVerificationAdapterError(
                "fixture binding inconsistent",
                reason_code="fixture_binding_mismatch",
            )
        # Ensure argv form is python -m pytest.
        if len(argv) < 3 or argv[1] != "-m" or argv[2] != "pytest":
            raise PytestVerificationAdapterError(
                "argv must be explicit python -m pytest",
                reason_code="invalid_argv_form",
                details={"argv_preview": list(argv[:6])},
            )
        if argv[0] != request.python_executable:
            raise PytestVerificationAdapterError(
                "argv[0] must equal python_executable",
                reason_code="invalid_argv_form",
            )
        # Selector CID must match the receipt key.
        from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
            _SELECTOR_IDENTITY_INPUT_SCHEMA,
            _structured_cid,
        )

        try:
            observed_selector = _structured_cid(
                _SELECTOR_IDENTITY_INPUT_SCHEMA,
                {"argv": tuple(argv)},
                field_name="selector_argv",
            )
        except (VerificationContractError, VerificationIdentityError) as exc:
            raise PytestVerificationAdapterError(
                "selector argv cannot be identity-bound",
                reason_code="selector_binding_error",
                details={"error": str(exc)},
            ) from exc
        if observed_selector != key.selector_cid:
            raise PytestVerificationAdapterError(
                "built argv does not match receipt_key.selector_cid",
                reason_code="selector_binding_mismatch",
                details={
                    "expected_selector_cid": key.selector_cid,
                    "observed_selector_cid": observed_selector,
                },
            )
        env = key.environment_observation
        if env.get("tool_name") != "pytest":
            raise PytestVerificationAdapterError(
                "environment tool_name must be pytest",
                reason_code="environment_binding_mismatch",
            )
        if env.get("adapter_schema") != PYTEST_VERIFICATION_ADAPTER_SCHEMA:
            raise PytestVerificationAdapterError(
                "environment adapter_schema mismatch",
                reason_code="environment_binding_mismatch",
            )
        if env.get("network_policy") != request.network_policy:
            raise PytestVerificationAdapterError(
                "environment network_policy mismatch",
                reason_code="environment_binding_mismatch",
            )

    def _build_command(
        self,
        request: PytestVerificationRequest,
        argv: Sequence[str],
    ) -> VerificationCommand:
        env = dict(request.environment)
        if not env:
            env = build_hermetic_environment()
        return VerificationCommand(
            argv=list(argv),
            cwd=request.cwd,
            environment=env,
            timeout_seconds=request.timeout_seconds,
            sandbox=request.sandbox,
            network_policy=request.network_policy,
            max_stdout_bytes=request.max_stdout_bytes,
            max_stderr_bytes=request.max_stderr_bytes,
            lane_id=request.lane_id,
            resource_class=request.resource_class,
            stage=request.stage,
            metadata={
                **dict(request.metadata),
                "adapter": PYTEST_VERIFICATION_ADAPTER_SCHEMA,
                "mode": request.mode.value,
            },
        )

    def _load_phase_report(
        self,
        request: PytestVerificationRequest,
        run_result: VerificationRunResult,
    ) -> Any:
        artifact_root = Path(request.sandbox.artifact_root)
        report_path = artifact_root / request.phase_report_relpath
        if report_path.is_file():
            try:
                return report_path.read_bytes()
            except OSError:
                pass
        # Fall back to stdout capture preview (bounded).
        preview = run_result.stdout.preview if run_result.stdout else ""
        if preview.strip():
            return preview
        return None

    def _finalize(
        self,
        *,
        request: PytestVerificationRequest,
        argv: Sequence[str],
        run_result: VerificationRunResult | None,
        phase_items: Sequence[PytestNodePhaseAccounting],
        collected: int,
        collection_errors: Sequence[str],
        usage_error: bool,
        malformed: bool,
        forced_status: TerminalStatus | None,
        extra_reasons: Sequence[str],
    ) -> PytestVerificationResult:
        if forced_status is not None:
            status = forced_status
            reasons = _unique_reasons(extra_reasons)
        else:
            status, reasons = project_terminal_status(
                run_result=run_result,
                phase_items=phase_items,
                collected=collected,
                collection_errors=collection_errors,
                usage_error=usage_error,
                malformed=malformed,
                mode=request.mode,
                selected_node_ids=request.node_ids,
                simulated=request.simulated,
            )
            if extra_reasons:
                reasons = _unique_reasons((*extra_reasons, *reasons))

        duration_ms = int(run_result.duration_ms) if run_result is not None else 0
        exit_code = _exit_code_for_status(status, run_result)

        stdout_cid = ""
        stderr_cid = ""
        artifact_cids: list[str] = []
        if run_result is not None:
            if run_result.stdout and run_result.stdout.cid:
                stdout_cid = run_result.stdout.cid
                artifact_cids.append(stdout_cid)
            if run_result.stderr and run_result.stderr.cid:
                stderr_cid = run_result.stderr.cid
                artifact_cids.append(stderr_cid)

        phase_report_payload = encode_phase_report(
            items=phase_items,
            collected=collected,
            collection_errors=collection_errors,
            usage_error=usage_error,
            malformed=malformed,
            extra={
                "terminal_status": status.value,
                "mode": request.mode.value,
                "reason_codes": list(reasons),
            },
        )
        phase_report_bytes = json.dumps(
            phase_report_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        phase_report_cid = cid_for_bytes(phase_report_bytes)
        artifact_cids.append(phase_report_cid)

        accounting_payload = {
            "schema": PYTEST_PHASE_ACCOUNTING_SCHEMA,
            "items": [item.to_dict() for item in phase_items],
            "collected": collected,
        }
        accounting_cid = cid_for_bytes(
            json.dumps(
                accounting_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
        artifact_cids.append(accounting_cid)

        # Existing-test bridge artifacts must be named on the observation.
        bridge_receipt = request.existing_test_pass_receipt
        bridge_key = request.existing_test_execution_key
        attach_bridge = False
        if (
            bridge_receipt is not None
            and bridge_key is not None
            and status is TerminalStatus.PASSED
        ):
            artifact_cids.append(bridge_key.execution_key_id)
            artifact_cids.append(bridge_receipt.receipt_id)
            attach_bridge = True

        # DirectExecutionObservation exit/artifact rules for conclusive statuses.
        observation_exit = exit_code
        if status in {
            TerminalStatus.TIMEOUT,
            TerminalStatus.CANCELLED,
            TerminalStatus.UNAVAILABLE,
            TerminalStatus.SIMULATED,
        }:
            # Non-conclusive: exit may be None; avoid requiring stdout when never started.
            if run_result is None or not run_result.process_started:
                observation_exit = None
                if not stdout_cid:
                    # Provide empty stream digests so incomplete runs remain bounded.
                    empty = cid_for_bytes(b"")
                    stdout_cid = stdout_cid or empty
                    stderr_cid = stderr_cid or empty
                    if empty not in artifact_cids:
                        artifact_cids.extend([stdout_cid, stderr_cid])
            elif not (stdout_cid and stderr_cid):
                empty = cid_for_bytes(b"")
                stdout_cid = stdout_cid or empty
                stderr_cid = stderr_cid or empty
        else:
            if observation_exit is None:
                observation_exit = 0 if status is TerminalStatus.PASSED else 1
            if status is TerminalStatus.PASSED:
                observation_exit = 0
            empty = cid_for_bytes(b"")
            stdout_cid = stdout_cid or empty
            stderr_cid = stderr_cid or empty
            for cid in (stdout_cid, stderr_cid):
                if cid not in artifact_cids:
                    artifact_cids.append(cid)

        # Deduplicate artifact CIDs while preserving order.
        deduped: list[str] = []
        seen_cids: set[str] = set()
        for cid in artifact_cids:
            if cid and cid not in seen_cids:
                seen_cids.add(cid)
                deduped.append(cid)

        key = request.receipt_key
        execution = DirectExecutionObservation(
            receipt_key_cid=key.key_id,
            repository_tree_cid=key.repository_tree_cid,
            environment_cid=key.environment_cid,
            repository_tree_observation=key.repository_tree_observation,
            environment_observation=key.environment_observation,
            terminal_status=status,
            command_argv=tuple(argv),
            duration_ms=duration_ms,
            exit_code=observation_exit,
            stdout_artifact_cid=stdout_cid,
            stderr_artifact_cid=stderr_cid,
            artifact_cids=tuple(deduped),
            reason_codes=reasons,
        )

        receipt: TestReceipt | None
        try:
            receipt = TestReceipt(
                key=key,
                execution=execution,
                test_pass_receipt=bridge_receipt if attach_bridge else None,
                test_execution_key=bridge_key if attach_bridge else None,
                artifact_cids=tuple(deduped),
                reason_codes=reasons,
            )
        except (VerificationContractError, VerificationIdentityError) as exc:
            # Bridge identity failure cannot manufacture a pass.
            if attach_bridge and status is TerminalStatus.PASSED:
                status = TerminalStatus.INVALID
                reasons = _unique_reasons(
                    (*reasons, "existing_receipt_projection_invalid", type(exc).__name__)
                )
                execution = DirectExecutionObservation(
                    receipt_key_cid=key.key_id,
                    repository_tree_cid=key.repository_tree_cid,
                    environment_cid=key.environment_cid,
                    repository_tree_observation=key.repository_tree_observation,
                    environment_observation=key.environment_observation,
                    terminal_status=status,
                    command_argv=tuple(argv),
                    duration_ms=duration_ms,
                    exit_code=1,
                    stdout_artifact_cid=stdout_cid,
                    stderr_artifact_cid=stderr_cid,
                    artifact_cids=tuple(deduped),
                    reason_codes=reasons,
                )
                receipt = TestReceipt(
                    key=key,
                    execution=execution,
                    artifact_cids=tuple(deduped),
                    reason_codes=reasons,
                )
            else:
                raise PytestVerificationAdapterError(
                    f"failed to project TestReceipt: {exc}",
                    reason_code="receipt_projection_failed",
                    details={"error": str(exc)},
                ) from exc

        production_admissible = (
            not request.simulated
            and status is TerminalStatus.PASSED
            and receipt is not None
            and receipt.terminal_success
            and receipt.status is TerminalStatus.PASSED
        )
        if self._require_production and request.simulated:
            production_admissible = False

        publication_allowed = True
        if run_result is not None:
            publication_allowed = bool(run_result.publication_allowed)
        if status in {
            TerminalStatus.CANCELLED,
            TerminalStatus.TIMEOUT,
            TerminalStatus.UNAVAILABLE,
            TerminalStatus.SIMULATED,
        }:
            if status is TerminalStatus.SIMULATED:
                publication_allowed = False
            elif run_result is not None:
                publication_allowed = bool(run_result.publication_allowed)
            else:
                publication_allowed = False

        return PytestVerificationResult(
            terminal_status=status,
            receipt=receipt,
            command_argv=tuple(argv),
            mode=request.mode,
            phase_accounting=tuple(phase_items),
            collected_count=collected,
            artifact_cids=tuple(deduped),
            reason_codes=reasons,
            production_admissible=production_admissible,
            simulated=request.simulated,
            run_result=run_result,
            phase_report_cid=phase_report_cid,
            duration_ms=duration_ms,
            exit_code=exit_code,
            publication_allowed=publication_allowed,
        )


def create_pytest_verification_adapter(
    process_runner: VerificationProcessRunner | None = None,
    *,
    require_production: bool = True,
) -> PytestVerificationAdapter:
    """Factory for the production pytest verification adapter."""

    return PytestVerificationAdapter(
        process_runner=process_runner,
        require_production=require_production,
    )


__all__ = [
    "DEFAULT_PHASE_REPORT_RELPATH",
    "PYTEST_ADAPTER_EVIDENCE",
    "PYTEST_PHASE_ACCOUNTING_SCHEMA",
    "PYTEST_PHASE_REPORT_SCHEMA",
    "PYTEST_VERIFICATION_ADAPTER_INTERFACE",
    "PYTEST_VERIFICATION_ADAPTER_SCHEMA",
    "PytestAdvisoryPolicy",
    "PytestNodePhaseAccounting",
    "PytestRunMode",
    "PytestVerificationAdapter",
    "PytestVerificationAdapterError",
    "PytestVerificationRequest",
    "PytestVerificationResult",
    "build_pytest_argv",
    "create_pytest_verification_adapter",
    "encode_phase_report",
    "parse_phase_report",
    "project_terminal_status",
]
