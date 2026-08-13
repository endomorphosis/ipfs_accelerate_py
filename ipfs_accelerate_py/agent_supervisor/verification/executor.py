"""Execute sealed verification plans and recompute production acceptance.

``VerificationExecutor`` / :func:`execute_verification_plan` are the
orchestration surface for IVP-014 / ``ivp/execution-bundle@1``.

Normative pipeline:

1. Validate the sealed :class:`VerificationPlan` and revalidate observed
   target-tree / sandbox / tool identities against it (pre-execution).
2. Acquire a typed plan-level resource lease bounded by
   ``expected_processes``; rejection is explicit and never silent.
3. Reuse only plan-approved exact cache hits; never treat cache presence or
   provider text as authority.
4. Execute required misses under the sealed dependency DAG with bounded
   parallelism, per-step deadlines, and a global wall-time budget.
5. Cancellation / timeout terminates process trees (including grandchildren
   and escaped sessions via the shared runner fence) and discards late
   success receipts.
6. Unavailable tools remain ``unavailable``; failures receive compact
   counterexamples when minimization succeeds.
7. Publish scoped-staleness tombstones only through a post-plan CAS
   transaction after identity revalidation.
8. Bundle results, choose a provider-neutral model route, emit a compact
   summary and structural commitment, and recompute production acceptance.

Production acceptance is true only when every required non-advisory current
leaf is a successful production terminal (``passed`` / ``proved``), no
mandatory full-suite fallback remains, and human review is false.  Explicitly
advisory obligations may remain unresolved and can never be upgraded.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Protocol

from ..runtime.resource_scheduler import (
    HostResourceSnapshot,
    LaneResourceRequirements,
    ResourceAdmissionLease,
    ResourceScheduler,
)
from .bundle import (
    build_verification_bundle,
    build_verification_commitment,
    build_verification_summary,
)
from .contracts import (
    CacheReuseDisposition,
    CounterexampleReceipt,
    DirectExecutionObservation,
    ModelRoute,
    ModelRouteDecision,
    ProofReceipt,
    StaticAnalysisReceipt,
    TerminalStatus,
    TestReceipt,
    TypeCheckReceipt,
    VerificationBundle,
    VerificationCommitment,
    VerificationContractError,
    VerificationIdentityError,
    VerificationPlan,
    VerificationReceipt,
    VerificationReceiptKey,
    VerificationReceiptKind,
    VerificationSummary,
)
from .counterexamples import minimize_counterexample
from .model_route import (
    ModelRouteError,
    decide_model_route,
    default_inventory,
    derive_model_route_facts,
    policy_cid_for,
)
from .process_runner import (
    VerificationCancellation,
    VerificationProcessRunner,
    fence_process_tree,
)
from .receipt_cache import (
    ReceiptCacheError,
    VerificationReceiptCache,
    production_eligible,
)
from .receipt_store import ReceiptStoreError

# ---------------------------------------------------------------------------
# Schema / interface / evidence
# ---------------------------------------------------------------------------

VERIFICATION_EXECUTOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-executor@1"
)
VERIFICATION_EXECUTOR_INTERFACE: Final[str] = "VerificationExecutor@1"
EXECUTION_BUNDLE_EVIDENCE: Final[str] = "ivp/execution-bundle@1"
EXECUTION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-execution-result@1"
)

DEFAULT_RESOURCE_CLASS: Final[str] = "cpu-validation"
DEFAULT_STAGE: Final[str] = "validation"
DEFAULT_LANE_PREFIX: Final[str] = "verification-plan"

_PRODUCTION_SUCCESS: Final[frozenset[TerminalStatus]] = frozenset(
    {TerminalStatus.PASSED, TerminalStatus.PROVED}
)
_FAILURE_STATUSES: Final[frozenset[TerminalStatus]] = frozenset(
    {TerminalStatus.FAILED, TerminalStatus.DISPROVED}
)
_KIND_STEP_PREFIX: Final[Mapping[VerificationReceiptKind, str]] = MappingProxyType(
    {
        VerificationReceiptKind.STATIC_ANALYSIS: "static",
        VerificationReceiptKind.TYPE_CHECK: "type",
        VerificationReceiptKind.TEST: "test",
        VerificationReceiptKind.PROOF: "proof",
    }
)
_KIND_ORDER: Final[tuple[VerificationReceiptKind, ...]] = (
    VerificationReceiptKind.STATIC_ANALYSIS,
    VerificationReceiptKind.TYPE_CHECK,
    VerificationReceiptKind.PROOF,
    VerificationReceiptKind.TEST,
)

Clock = Callable[[], float]
Sleep = Callable[[float], None]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class VerificationExecutorError(VerificationContractError):
    """Fail-closed executor contract or orchestration violation."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "executor_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


class VerificationExecutorIdentityError(
    VerificationExecutorError, VerificationIdentityError
):
    """Pre/post observed identities disagree with the sealed plan."""


class VerificationResourceRejectionError(VerificationExecutorError):
    """Typed resource admission rejection (never silent)."""


# ---------------------------------------------------------------------------
# Resource rejection
# ---------------------------------------------------------------------------


class ResourceRejectionKind(str, Enum):
    """Closed vocabulary for plan/process resource rejection."""

    PLAN_LEASE_DENIED = "plan_resource_lease_denied"
    PROCESS_LEASE_DENIED = "resource_lease_denied"
    CAPACITY_EXHAUSTED = "capacity_exhausted"
    LEASE_REVOKED = "lease_revoked"
    PARALLELISM_CAP = "bounded_parallelism_cap"


@dataclass(frozen=True, slots=True)
class ResourceRejection:
    """Typed resource rejection record (fail-closed, never upgraded)."""

    kind: ResourceRejectionKind
    reason_codes: tuple[str, ...]
    step_id: str = ""
    key_cid: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ResourceRejectionKind(self.kind))
        reasons = tuple(
            str(item).strip()
            for item in (self.reason_codes or ())
            if str(item).strip()
        )
        if not reasons:
            reasons = (self.kind.value,)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "step_id", str(self.step_id or "").strip())
        object.__setattr__(self, "key_cid", str(self.key_cid or "").strip())
        object.__setattr__(
            self,
            "details",
            MappingProxyType(dict(self.details or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "reason_codes": list(self.reason_codes),
            "step_id": self.step_id,
            "key_cid": self.key_cid,
            "details": dict(self.details),
        }


# ---------------------------------------------------------------------------
# Observed identities
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ObservedPlanIdentities:
    """Observed identities that must match the sealed plan pre and post.

    Callers supply the currently observed tree / semantic / environment /
    lock roots (and optional sandbox / tool identities).  The executor
    refuses to execute when any sealed plan root disagrees.
    """

    repository_tree_cid: str
    semantic_state_root_cid: str
    environment_cid: str
    dependency_lock_cid: str
    sandbox_id: str = ""
    tool_identity_cids: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "repository_tree_cid",
            "semantic_state_root_cid",
            "environment_cid",
            "dependency_lock_cid",
        ):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise VerificationExecutorIdentityError(
                    f"{name} is required for observed plan identities",
                    reason_code="identity_missing",
                    details={"field": name},
                )
            object.__setattr__(self, name, value)
        object.__setattr__(self, "sandbox_id", str(self.sandbox_id or "").strip())
        tools = {
            str(key): str(value)
            for key, value in dict(self.tool_identity_cids or {}).items()
            if str(key).strip() and str(value).strip()
        }
        object.__setattr__(self, "tool_identity_cids", MappingProxyType(tools))

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_tree_cid": self.repository_tree_cid,
            "semantic_state_root_cid": self.semantic_state_root_cid,
            "environment_cid": self.environment_cid,
            "dependency_lock_cid": self.dependency_lock_cid,
            "sandbox_id": self.sandbox_id,
            "tool_identity_cids": dict(self.tool_identity_cids),
        }

    @classmethod
    def from_plan(
        cls,
        plan: VerificationPlan,
        *,
        sandbox_id: str = "",
        tool_identity_cids: Mapping[str, str] | None = None,
    ) -> ObservedPlanIdentities:
        """Build an identity observation that matches *plan* exactly."""

        return cls(
            repository_tree_cid=plan.repository_tree_cid,
            semantic_state_root_cid=plan.semantic_state_root_cid,
            environment_cid=plan.environment_cid,
            dependency_lock_cid=plan.dependency_lock_cid,
            sandbox_id=sandbox_id,
            tool_identity_cids=dict(tool_identity_cids or {}),
        )


@dataclass(frozen=True, slots=True)
class IdentityRevalidation:
    """Pre/post identity match result against the sealed plan."""

    pre_matched: bool
    post_matched: bool
    pre_mismatches: tuple[str, ...]
    post_mismatches: tuple[str, ...]
    pre: Mapping[str, Any]
    post: Mapping[str, Any]

    @property
    def matched(self) -> bool:
        return self.pre_matched and self.post_matched

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_matched": self.pre_matched,
            "post_matched": self.post_matched,
            "matched": self.matched,
            "pre_mismatches": list(self.pre_mismatches),
            "post_mismatches": list(self.post_mismatches),
            "pre": dict(self.pre),
            "post": dict(self.post),
        }


# ---------------------------------------------------------------------------
# Check runner protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckRunOutcome:
    """Outcome of one required-check execution attempt.

    ``publication_allowed`` is authoritative for cache/bundle admission of a
    successful terminal: late success after cancellation/timeout must set it
    false (and typically emit a cancelled/timeout receipt).
    """

    receipt: VerificationReceipt | None
    publication_allowed: bool
    cancelled: bool = False
    timed_out: bool = False
    unavailable: bool = False
    reason_codes: tuple[str, ...] = ()
    process_tree_fenced: bool = False
    process: Any = None
    resource_rejection: ResourceRejection | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": (
                self.receipt.receipt_id if self.receipt is not None else ""
            ),
            "status": (
                self.receipt.status.value if self.receipt is not None else ""
            ),
            "publication_allowed": self.publication_allowed,
            "cancelled": self.cancelled,
            "timed_out": self.timed_out,
            "unavailable": self.unavailable,
            "reason_codes": list(self.reason_codes),
            "process_tree_fenced": self.process_tree_fenced,
            "resource_rejection": (
                self.resource_rejection.to_dict()
                if self.resource_rejection is not None
                else None
            ),
        }


class CheckRunner(Protocol):
    """Callable that executes one sealed receipt key under a timeout fence."""

    def __call__(
        self,
        key: VerificationReceiptKey,
        *,
        step_id: str,
        timeout_ms: int,
        cancellation: VerificationCancellation | None,
        plan: VerificationPlan,
    ) -> CheckRunOutcome: ...


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerificationExecutionResult:
    """Complete plan-bound execution result with recomputed acceptance."""

    SCHEMA: ClassVar[str] = EXECUTION_RESULT_SCHEMA
    INTERFACE: ClassVar[str] = VERIFICATION_EXECUTOR_INTERFACE
    EVIDENCE: ClassVar[str] = EXECUTION_BUNDLE_EVIDENCE

    verification_plan: VerificationPlan
    bundle: VerificationBundle
    summary: VerificationSummary
    commitment: VerificationCommitment
    model_route_decision: ModelRouteDecision
    production_acceptance: bool
    reused_receipts: tuple[VerificationReceipt, ...]
    executed_receipts: tuple[VerificationReceipt, ...]
    counterexamples: tuple[CounterexampleReceipt, ...]
    resource_rejections: tuple[ResourceRejection, ...]
    cancelled: bool
    timed_out: bool
    cancellation_fenced: bool
    late_receipts_fenced: int
    identity_revalidation: IdentityRevalidation
    advisory_unresolved_key_cids: tuple[str, ...]
    step_outcomes: Mapping[str, Mapping[str, Any]]
    reason_codes: tuple[str, ...]
    wall_time_ms: int
    tombstones_published: tuple[str, ...]
    execution_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "production_acceptance",
            bool(self.production_acceptance),
        )
        object.__setattr__(self, "cancelled", bool(self.cancelled))
        object.__setattr__(self, "timed_out", bool(self.timed_out))
        object.__setattr__(
            self, "cancellation_fenced", bool(self.cancellation_fenced)
        )
        object.__setattr__(
            self, "late_receipts_fenced", int(self.late_receipts_fenced)
        )
        object.__setattr__(self, "wall_time_ms", max(0, int(self.wall_time_ms)))
        if not self.execution_id:
            object.__setattr__(
                self,
                "execution_id",
                f"exec:{uuid.uuid4().hex}",
            )
        object.__setattr__(
            self,
            "step_outcomes",
            MappingProxyType(
                {str(key): dict(value) for key, value in dict(self.step_outcomes).items()}
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(str(item) for item in self.reason_codes if str(item)),
        )
        object.__setattr__(
            self,
            "advisory_unresolved_key_cids",
            tuple(sorted(set(self.advisory_unresolved_key_cids))),
        )
        object.__setattr__(
            self,
            "tombstones_published",
            tuple(sorted(set(self.tombstones_published))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "evidence": self.EVIDENCE,
            "execution_id": self.execution_id,
            "plan_id": self.verification_plan.plan_id,
            "bundle_id": self.bundle.bundle_id,
            "summary_id": self.summary.summary_id,
            "commitment_id": self.commitment.commitment_id,
            "model_route_decision_id": self.model_route_decision.decision_id,
            "model_route": self.model_route_decision.route.value,
            "production_acceptance": self.production_acceptance,
            "reused_receipt_cids": [
                item.receipt_id for item in self.reused_receipts
            ],
            "executed_receipt_cids": [
                item.receipt_id for item in self.executed_receipts
            ],
            "counterexample_cids": [
                item.counterexample_id for item in self.counterexamples
            ],
            "resource_rejections": [
                item.to_dict() for item in self.resource_rejections
            ],
            "cancelled": self.cancelled,
            "timed_out": self.timed_out,
            "cancellation_fenced": self.cancellation_fenced,
            "late_receipts_fenced": self.late_receipts_fenced,
            "identity_revalidation": self.identity_revalidation.to_dict(),
            "advisory_unresolved_key_cids": list(
                self.advisory_unresolved_key_cids
            ),
            "step_outcomes": {
                key: dict(value) for key, value in self.step_outcomes.items()
            },
            "reason_codes": list(self.reason_codes),
            "wall_time_ms": self.wall_time_ms,
            "tombstones_published": list(self.tombstones_published),
            "aggregate_terminal_status": (
                self.summary.aggregate_terminal_status.value
            ),
            "structurally_complete": self.bundle.structurally_complete,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms(clock: Clock | None = None) -> int:
    if clock is not None:
        return int(max(0.0, float(clock()) * 1000.0))
    return int(time.time() * 1000)


def _monotonic(clock: Clock | None = None) -> float:
    if clock is not None:
        return float(clock())
    return time.monotonic()


def _require_plan(value: Any) -> VerificationPlan:
    if isinstance(value, VerificationPlan):
        # Re-validate via round-trip so caller-owned nested mappings detach.
        return VerificationPlan.from_dict(value.to_record())
    if isinstance(value, Mapping):
        return VerificationPlan.from_dict(value)
    raise VerificationExecutorError(
        "verification_plan must be a VerificationPlan",
        reason_code="invalid_plan",
    )


def _identity_mismatches(
    plan: VerificationPlan,
    observed: ObservedPlanIdentities,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if observed.repository_tree_cid != plan.repository_tree_cid:
        mismatches.append("repository_tree_cid")
    if observed.semantic_state_root_cid != plan.semantic_state_root_cid:
        mismatches.append("semantic_state_root_cid")
    if observed.environment_cid != plan.environment_cid:
        mismatches.append("environment_cid")
    if observed.dependency_lock_cid != plan.dependency_lock_cid:
        mismatches.append("dependency_lock_cid")
    return tuple(mismatches)


def _default_host_snapshot(*, process_slots: int = 8) -> HostResourceSnapshot:
    slots = max(1, int(process_slots))
    return HostResourceSnapshot(
        observed_at_ms=int(time.time() * 1000),
        cpu_percent=10,
        memory_percent=20,
        disk_percent=10,
        memory_total_bytes=16 * 1024 * 1024 * 1024,
        memory_available_bytes=8 * 1024 * 1024 * 1024,
        disk_total_bytes=100 * 1024 * 1024 * 1024,
        disk_available_bytes=50 * 1024 * 1024 * 1024,
        active_phase="verification_executor",
        active_workers=0,
        worker_limit=slots,
        available_worker_capacity=slots,
        capabilities=("cpu",),
        resource_classes=(
            "cpu-small",
            "cpu-validation",
            "cpu-proof",
            DEFAULT_RESOURCE_CLASS,
        ),
    )


def _receipt_for_status(
    key: VerificationReceiptKey,
    status: TerminalStatus,
    *,
    label: str,
    reason_codes: Sequence[str] = (),
    duration_ms: int = 0,
    command_argv: Sequence[str] | None = None,
) -> VerificationReceipt:
    """Project a direct-status receipt for non-proof and non-conclusive proof.

    Conclusive proof (``proved`` / ``disproved``) requires authoritative
    formal evidence and must be supplied by a check runner, not synthesized
    here.  Unavailable / timeout / cancelled / unknown / not_modeled /
    invalid / simulated / stale remain direct-observation statuses.
    """

    if key.receipt_kind is VerificationReceiptKind.PROOF and status in {
        TerminalStatus.PROVED,
        TerminalStatus.DISPROVED,
        TerminalStatus.PASSED,
        TerminalStatus.FAILED,
    }:
        raise VerificationExecutorError(
            "conclusive proof statuses cannot be synthesized by the executor",
            reason_code="proof_status_not_synthesized",
            details={"status": status.value},
        )
    if key.receipt_kind is not VerificationReceiptKind.PROOF and status in {
        TerminalStatus.PROVED,
        TerminalStatus.DISPROVED,
    }:
        raise VerificationExecutorError(
            "non-proof receipts cannot use proof terminal statuses",
            reason_code="invalid_status_for_kind",
            details={"status": status.value, "kind": key.receipt_kind.value},
        )

    default_argv = {
        VerificationReceiptKind.STATIC_ANALYSIS: (
            "/usr/bin/ruff",
            "check",
            "src/example.py",
        ),
        VerificationReceiptKind.TYPE_CHECK: (
            "/usr/bin/python3.12",
            "-m",
            "mypy",
            "src/example.py",
        ),
        VerificationReceiptKind.TEST: (
            "/usr/bin/python3.12",
            "-m",
            "pytest",
            "src/example.py",
        ),
        VerificationReceiptKind.PROOF: (
            "/usr/bin/z3",
            "-smt2",
            "obligation.smt2",
        ),
    }[key.receipt_kind]
    argv = tuple(command_argv) if command_argv is not None else default_argv

    conclusive = status in {
        TerminalStatus.PASSED,
        TerminalStatus.PROVED,
        TerminalStatus.DISPROVED,
    }
    exit_code: int | None
    if conclusive:
        exit_code = 0
    elif status in {
        TerminalStatus.UNAVAILABLE,
        TerminalStatus.CANCELLED,
        TerminalStatus.TIMEOUT,
        TerminalStatus.NOT_MODELED,
        TerminalStatus.UNKNOWN,
        TerminalStatus.STALE,
        TerminalStatus.SIMULATED,
        TerminalStatus.INVALID,
    }:
        exit_code = None
    else:
        exit_code = 1

    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
        content_identity,
    )

    stdout_cid = content_identity(
        {"schema": "verification-executor-stream@1", "stream": "stdout", "label": label}
    )
    stderr_cid = content_identity(
        {"schema": "verification-executor-stream@1", "stream": "stderr", "label": label}
    )
    # Non-completed observations may omit stream CIDs only when exit_code is
    # None; contracts require both streams whenever exit_code is set.
    if exit_code is None and status in {
        TerminalStatus.UNAVAILABLE,
        TerminalStatus.CANCELLED,
        TerminalStatus.TIMEOUT,
    }:
        # Keep streams for auditability on failure-like paths that still have
        # an exit code; unavailable/cancel/timeout may omit them.
        stdout_cid = ""
        stderr_cid = ""

    observation = DirectExecutionObservation(
        receipt_key_cid=key.key_id,
        repository_tree_cid=key.repository_tree_cid,
        environment_cid=key.environment_cid,
        repository_tree_observation=dict(key.repository_tree_observation),
        environment_observation=dict(key.environment_observation),
        terminal_status=status,
        command_argv=argv,
        duration_ms=max(0, int(duration_ms)),
        exit_code=exit_code,
        stdout_artifact_cid=stdout_cid,
        stderr_artifact_cid=stderr_cid,
        artifact_cids=(),
        reason_codes=tuple(str(item) for item in reason_codes if str(item)),
    )

    if key.receipt_kind is VerificationReceiptKind.STATIC_ANALYSIS:
        return StaticAnalysisReceipt(
            key,
            observation,
            reason_codes=tuple(reason_codes),
        )
    if key.receipt_kind is VerificationReceiptKind.TYPE_CHECK:
        return TypeCheckReceipt(
            key,
            observation,
            reason_codes=tuple(reason_codes),
        )
    if key.receipt_kind is VerificationReceiptKind.TEST:
        return TestReceipt(
            key,
            observation,
            reason_codes=tuple(reason_codes),
        )
    return ProofReceipt(
        key=key,
        execution=observation,
        reason_codes=tuple(reason_codes),
    )


def _bind_steps_to_keys(
    plan: VerificationPlan,
    *,
    step_bindings: Mapping[str, str] | None = None,
) -> dict[str, VerificationReceiptKey]:
    """Map sealed dependency-DAG step ids onto required receipt keys.

    Resolution order per step:

    1. explicit ``step_bindings`` (step_id → key_id)
    2. step_id equals a required ``key_id``
    3. unique remaining key of the kind implied by a ``kind:…`` prefix
    4. single remaining key when the plan has one key / one step
    """

    keys = list(plan.required_receipt_keys)
    keys_by_id = {key.key_id: key for key in keys}
    explicit = {
        str(step): str(key_id)
        for step, key_id in dict(step_bindings or {}).items()
        if str(step).strip() and str(key_id).strip()
    }
    steps = list(plan.dependency_dag)
    bound: dict[str, VerificationReceiptKey] = {}
    used_key_ids: set[str] = set()

    def _claim(step: str, key: VerificationReceiptKey) -> None:
        if key.key_id in used_key_ids and bound.get(step) is not key:
            raise VerificationExecutorError(
                "step binding assigns the same receipt key more than once",
                reason_code="step_binding_conflict",
                details={"step_id": step, "key_id": key.key_id},
            )
        bound[step] = key
        used_key_ids.add(key.key_id)

    # Pass 1: explicit + direct key_id equality.
    for step in steps:
        if step in explicit:
            key_id = explicit[step]
            key = keys_by_id.get(key_id)
            if key is None:
                raise VerificationExecutorError(
                    "step binding references a key outside the sealed plan",
                    reason_code="step_binding_unknown_key",
                    details={"step_id": step, "key_id": key_id},
                )
            _claim(step, key)
            continue
        if step in keys_by_id:
            _claim(step, keys_by_id[step])

    # Pass 2: kind-prefix uniqueness for remaining steps.
    remaining_steps = [step for step in steps if step not in bound]
    remaining_keys = [key for key in keys if key.key_id not in used_key_ids]
    for step in list(remaining_steps):
        prefix = step.split(":", 1)[0] if ":" in step else ""
        kind_for_prefix = {
            value: kind for kind, value in _KIND_STEP_PREFIX.items()
        }.get(prefix)
        if kind_for_prefix is None:
            continue
        candidates = [
            key for key in remaining_keys if key.receipt_kind is kind_for_prefix
        ]
        if len(candidates) == 1:
            key = candidates[0]
            _claim(step, key)
            remaining_keys = [
                item for item in remaining_keys if item.key_id != key.key_id
            ]
            remaining_steps.remove(step)

    # Pass 3: 1:1 leftover assignment in stable key_id order.
    remaining_steps = sorted(step for step in steps if step not in bound)
    remaining_keys = sorted(
        (key for key in keys if key.key_id not in used_key_ids),
        key=lambda item: ( _KIND_ORDER.index(item.receipt_kind), item.key_id ),
    )
    if len(remaining_steps) == len(remaining_keys):
        for step, key in zip(remaining_steps, remaining_keys):
            _claim(step, key)
    elif remaining_steps and not remaining_keys:
        raise VerificationExecutorError(
            "dependency DAG has steps with no remaining required keys",
            reason_code="step_binding_orphan_step",
            details={"steps": remaining_steps},
        )
    elif remaining_steps:
        raise VerificationExecutorError(
            "unable to bind dependency DAG steps to required receipt keys",
            reason_code="step_binding_ambiguous",
            details={
                "unbound_steps": remaining_steps,
                "unbound_key_ids": [key.key_id for key in remaining_keys],
            },
        )

    # Keys with no DAG step still need execution order.  Attach synthetic
    # independent steps keyed by key_id so they appear as roots.
    for key in keys:
        if key.key_id not in used_key_ids:
            step = key.key_id
            bound[step] = key
            used_key_ids.add(key.key_id)

    if set(used_key_ids) != {key.key_id for key in keys}:
        raise VerificationExecutorError(
            "step binding must cover every required receipt key",
            reason_code="step_binding_incomplete",
        )
    return bound


def _dependency_ready(
    step: str,
    dag: Mapping[str, tuple[str, ...]],
    completed: set[str],
) -> bool:
    deps = dag.get(step, ())
    return all(dep in completed for dep in deps)


def compute_production_acceptance(
    bundle: VerificationBundle,
    *,
    advisory_key_cids: Sequence[str] = (),
) -> bool:
    """Return True only for fully satisfied required production leaves.

    Advisory keys (explicitly declared) may remain unresolved or non-success
    without blocking acceptance, and are never upgraded by this function.
    """

    plan = bundle.verification_plan
    if plan.human_review_required or bundle.human_review_required:
        return False
    if bundle.mandatory_fallback_pending:
        return False

    advisory = frozenset(str(item) for item in advisory_key_cids if str(item))
    receipts_by_key = {receipt.key.key_id: receipt for receipt in bundle.receipts}

    for key in plan.required_receipt_keys:
        if key.key_id in advisory:
            # Advisory leaves may remain unresolved; never upgrade them here.
            continue
        receipt = receipts_by_key.get(key.key_id)
        if receipt is None:
            return False
        if receipt.status not in _PRODUCTION_SUCCESS:
            return False
        if not receipt.terminal_success:
            return False
        if not production_eligible(receipt):
            return False

    # Required proof obligations that are not advisory must be resolved.
    for obligation_cid in bundle.unresolved_proof_obligation_cids:
        # Map obligation → key; if that key is advisory, tolerate.
        related_keys = [
            key
            for key in plan.required_receipt_keys
            if key.receipt_kind is VerificationReceiptKind.PROOF
            and key.proof_obligation_cid == obligation_cid
        ]
        if related_keys and all(key.key_id in advisory for key in related_keys):
            continue
        return False

    # No unresolved non-advisory requirements.
    for key_id in bundle.unresolved_requirement_ids:
        if key_id not in advisory:
            return False

    return True


def _never_upgrade_advisory(
    receipt: VerificationReceipt | None,
    *,
    advisory: bool,
    prior_status: TerminalStatus | None = None,
) -> VerificationReceipt | None:
    """Refuse to upgrade an advisory leaf to production success."""

    if receipt is None or not advisory:
        return receipt
    if receipt.status in _PRODUCTION_SUCCESS:
        # Explicit policy: advisory obligations can never be upgraded to a
        # production-success terminal by the executor orchestration layer.
        # Collapse to not_modeled so acceptance does not treat them as
        # required current leaves while preserving failure/unavailable.
        return _receipt_for_status(
            receipt.key,
            TerminalStatus.NOT_MODELED,
            label="advisory-no-upgrade",
            reason_codes=(
                "advisory_obligation",
                "advisory_never_upgraded",
                f"collapsed_from:{receipt.status.value}",
            ),
            duration_ms=int(receipt.execution.duration_ms),
            command_argv=tuple(receipt.execution.command_argv),
        )
    return receipt


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class VerificationExecutor:
    """Execute a sealed :class:`VerificationPlan` under resource leases.

    Parameters
    ----------
    cache:
        Optional exact-key receipt cache for admitting fresh production
        successes and publishing post-plan staleness tombstones.
    process_runner:
        Shared admitted process runner (used by default adapters / fencing).
    resource_scheduler / host_snapshot:
        Plan-level resource admission.  Denial is typed and never silent.
    check_runner / check_runners:
        Injectable check execution.  Tests inject hermetic runners; production
        callers bind adapter-backed runners per receipt kind.
    model_route_decision:
        Optional precomputed provider-neutral route; when omitted a hermetic
        route is derived from the plan and available inventory.
    require_resource_lease:
        When false (tests only), plan-level lease acquisition is skipped.
    """

    SCHEMA: ClassVar[str] = VERIFICATION_EXECUTOR_SCHEMA
    INTERFACE: ClassVar[str] = VERIFICATION_EXECUTOR_INTERFACE
    EVIDENCE: ClassVar[str] = EXECUTION_BUNDLE_EVIDENCE

    def __init__(
        self,
        *,
        cache: VerificationReceiptCache | None = None,
        process_runner: VerificationProcessRunner | None = None,
        resource_scheduler: ResourceScheduler | None = None,
        host_snapshot: HostResourceSnapshot | Mapping[str, Any] | None = None,
        check_runner: CheckRunner | None = None,
        check_runners: Mapping[VerificationReceiptKind, CheckRunner] | None = None,
        model_route_decision: ModelRouteDecision | None = None,
        model_route_policy: Any = None,
        available_models: Sequence[Any] | None = None,
        context_pack: Any = None,
        prior_attempts: Sequence[Any] = (),
        require_resource_lease: bool = True,
        admit_successes: bool = True,
        minimize_failures: bool = True,
        clock: Clock | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        self._cache = cache
        self._runner = process_runner
        self._scheduler = resource_scheduler or ResourceScheduler()
        self._host_snapshot = host_snapshot
        self._check_runner = check_runner
        self._check_runners = dict(check_runners or {})
        self._model_route_decision = model_route_decision
        self._model_route_policy = model_route_policy
        self._available_models = (
            None if available_models is None else tuple(available_models)
        )
        self._context_pack = context_pack
        self._prior_attempts = tuple(prior_attempts)
        self._require_resource_lease = bool(require_resource_lease)
        self._admit_successes = bool(admit_successes)
        self._minimize_failures = bool(minimize_failures)
        self._clock: Clock = clock or time.monotonic
        self._sleep: Sleep = sleep or time.sleep

    # -- public API --------------------------------------------------------

    def execute(
        self,
        verification_plan: VerificationPlan | Mapping[str, Any],
        *,
        observed_identities: ObservedPlanIdentities | Mapping[str, Any] | None = None,
        post_observed_identities: (
            ObservedPlanIdentities | Mapping[str, Any] | None
        ) = None,
        cancellation: VerificationCancellation | None = None,
        step_bindings: Mapping[str, str] | None = None,
        advisory_key_cids: Sequence[str] = (),
        human_review_required: bool | None = None,
        model_route_decision: ModelRouteDecision | None = None,
        routing_hints: Mapping[str, Any] | None = None,
    ) -> VerificationExecutionResult:
        """Execute *verification_plan* and recompute production acceptance."""

        plan = _require_plan(verification_plan)
        started = _monotonic(self._clock)
        advisory = frozenset(str(item) for item in advisory_key_cids if str(item))
        reason_codes: list[str] = []
        resource_rejections: list[ResourceRejection] = []
        step_outcomes: dict[str, dict[str, Any]] = {}
        late_fenced = 0
        cancelled = False
        timed_out = False
        cancellation_fenced = False
        tombstones: list[str] = []

        cancel = cancellation
        if cancel is None:
            cancel = VerificationCancellation()

        # ---- identity revalidation (pre) ---------------------------------
        if observed_identities is None:
            pre_obs = ObservedPlanIdentities.from_plan(plan)
        elif isinstance(observed_identities, ObservedPlanIdentities):
            pre_obs = observed_identities
        elif isinstance(observed_identities, Mapping):
            pre_obs = ObservedPlanIdentities(
                repository_tree_cid=str(
                    observed_identities.get("repository_tree_cid") or ""
                ),
                semantic_state_root_cid=str(
                    observed_identities.get("semantic_state_root_cid") or ""
                ),
                environment_cid=str(
                    observed_identities.get("environment_cid") or ""
                ),
                dependency_lock_cid=str(
                    observed_identities.get("dependency_lock_cid") or ""
                ),
                sandbox_id=str(observed_identities.get("sandbox_id") or ""),
                tool_identity_cids=dict(
                    observed_identities.get("tool_identity_cids") or {}
                ),
            )
        else:
            raise VerificationExecutorError(
                "observed_identities must be ObservedPlanIdentities or mapping",
                reason_code="invalid_identities",
            )

        pre_mismatches = _identity_mismatches(plan, pre_obs)
        pre_matched = not pre_mismatches
        if not pre_matched:
            raise VerificationExecutorIdentityError(
                "pre-execution observed identities do not match the sealed plan",
                reason_code="pre_identity_mismatch",
                details={"mismatches": list(pre_mismatches)},
            )

        # ---- plan-level resource lease -----------------------------------
        plan_lease: ResourceAdmissionLease | None = None
        if self._require_resource_lease:
            plan_lease, rejection = self._acquire_plan_lease(plan)
            if plan_lease is None:
                assert rejection is not None
                resource_rejections.append(rejection)
                reason_codes.append(rejection.kind.value)
                result = self._terminal_resource_rejection(
                    plan=plan,
                    rejection=rejection,
                    pre_obs=pre_obs,
                    advisory=advisory,
                    cancel=cancel,
                    started=started,
                    reason_codes=reason_codes,
                    human_review_required=human_review_required,
                    model_route_decision=model_route_decision,
                    routing_hints=routing_hints,
                )
                return result

        try:
            # ---- partition reuse vs execute ------------------------------
            decisions = {
                item.key_cid: item for item in plan.cache_reuse_decisions
            }
            reused: list[VerificationReceipt] = []
            to_execute: list[VerificationReceiptKey] = []
            stale_keys: list[VerificationReceiptKey] = []

            for key in plan.required_receipt_keys:
                decision = decisions[key.key_id]
                if decision.disposition is CacheReuseDisposition.REUSED:
                    candidate = decision.candidate_receipt
                    if (
                        candidate is None
                        or candidate.key.key_id != key.key_id
                        or candidate.status not in _PRODUCTION_SUCCESS
                        or not candidate.terminal_success
                    ):
                        # Plan claimed reuse but candidate is not production-
                        # admissible; fall back to execution fail-closed.
                        reason_codes.append("reuse_candidate_rejected")
                        to_execute.append(key)
                        continue
                    # Re-validate key identity against sealed plan roots.
                    if (
                        candidate.key.repository_tree_cid != plan.repository_tree_cid
                        or candidate.key.environment_cid != plan.environment_cid
                        or candidate.key.dependency_lock_cid
                        != plan.dependency_lock_cid
                    ):
                        reason_codes.append("reuse_identity_mismatch")
                        to_execute.append(key)
                        continue
                    reused.append(candidate)
                else:
                    to_execute.append(key)
                    if decision.disposition is CacheReuseDisposition.STALE:
                        stale_keys.append(key)

            # ---- cancellation before work --------------------------------
            if cancel.is_cancelled():
                cancelled = True
                cancellation_fenced = True
                reason_codes.append("cancelled_before_execution")
                executed = [
                    _receipt_for_status(
                        key,
                        TerminalStatus.CANCELLED,
                        label="cancelled-before",
                        reason_codes=("cancelled_before_execution",),
                    )
                    for key in to_execute
                ]
                return self._finalize(
                    plan=plan,
                    reused=reused,
                    executed=executed,
                    counterexamples=(),
                    resource_rejections=resource_rejections,
                    cancelled=cancelled,
                    timed_out=False,
                    cancellation_fenced=cancellation_fenced,
                    late_receipts_fenced=0,
                    pre_obs=pre_obs,
                    post_obs=pre_obs,
                    pre_mismatches=pre_mismatches,
                    post_mismatches=(),
                    advisory=advisory,
                    step_outcomes=step_outcomes,
                    reason_codes=reason_codes,
                    tombstones=(),
                    started=started,
                    human_review_required=human_review_required,
                    model_route_decision=model_route_decision,
                    routing_hints=routing_hints,
                )

            # ---- post-plan stale tombstones (before replacement admits) ---
            # Scoped staleness is published after plan sealing and identity
            # revalidation, but before admitting replacement receipts so the
            # tombstone targets the prior live entry rather than the fresh one.
            if (
                self._cache is not None
                and stale_keys
                and not cancel.is_cancelled()
            ):
                for key in stale_keys:
                    decision = decisions[key.key_id]
                    prior = decision.receipt_cid or None
                    try:
                        cas = self._cache.mark_stale(
                            key,
                            reason="scoped_staleness_after_plan",
                            prior_receipt_cid=prior,
                            metadata={"plan_id": plan.plan_id},
                        )
                        if cas.success:
                            tombstones.append(key.key_id)
                    except (ReceiptCacheError, ReceiptStoreError):
                        reason_codes.append("tombstone_publish_failed")

            # ---- DAG + bounded parallel execution ------------------------
            step_to_key = _bind_steps_to_keys(plan, step_bindings=step_bindings)
            key_to_step = {key.key_id: step for step, key in step_to_key.items()}
            execute_key_ids = {key.key_id for key in to_execute}

            # Steps that need execution (or already completed via reuse).
            dag: dict[str, tuple[str, ...]] = {
                step: tuple(deps) for step, deps in plan.dependency_dag.items()
            }
            for step, key in step_to_key.items():
                dag.setdefault(step, ())

            completed_steps: set[str] = set()
            # Mark steps whose keys were reused as completed for dependency
            # purposes without re-running them.
            for receipt in reused:
                step = key_to_step.get(receipt.key.key_id)
                if step is not None:
                    completed_steps.add(step)
                    step_outcomes[step] = {
                        "disposition": "reused",
                        "key_cid": receipt.key.key_id,
                        "receipt_cid": receipt.receipt_id,
                        "status": receipt.status.value,
                    }

            executed_by_key: dict[str, VerificationReceipt] = {}
            counterexamples: list[CounterexampleReceipt] = []
            max_workers = max(1, int(plan.expected_processes))
            global_deadline = started + (plan.max_execution_time_ms / 1000.0)
            lock = threading.Lock()

            pending_steps = {
                step
                for step, key in step_to_key.items()
                if key.key_id in execute_key_ids
            }

            def _run_step(step: str) -> tuple[str, CheckRunOutcome, int]:
                key = step_to_key[step]
                timeout_ms = int(
                    plan.step_timeouts_ms.get(step, plan.max_execution_time_ms)
                )
                remaining_ms = int(
                    max(1.0, (global_deadline - _monotonic(self._clock)) * 1000.0)
                )
                timeout_ms = min(timeout_ms, remaining_ms)
                outcome = self._execute_one(
                    key,
                    step_id=step,
                    timeout_ms=timeout_ms,
                    cancellation=cancel,
                    plan=plan,
                    advisory=key.key_id in advisory,
                )
                return step, outcome, timeout_ms

            while pending_steps:
                now = _monotonic(self._clock)
                if now >= global_deadline:
                    timed_out = True
                    reason_codes.append("plan_deadline_exceeded")
                    break
                if cancel.is_cancelled():
                    cancelled = True
                    cancellation_fenced = True
                    reason_codes.append("cancelled_during_execution")
                    break

                ready = sorted(
                    step
                    for step in pending_steps
                    if _dependency_ready(step, dag, completed_steps)
                )
                if not ready:
                    # Deadlock should be impossible for a validated DAG; treat
                    # remaining as unavailable fail-closed.
                    reason_codes.append("dependency_deadlock")
                    for step in sorted(pending_steps):
                        key = step_to_key[step]
                        receipt = _receipt_for_status(
                            key,
                            TerminalStatus.UNAVAILABLE,
                            label="dependency-deadlock",
                            reason_codes=("dependency_deadlock",),
                        )
                        executed_by_key[key.key_id] = receipt
                        step_outcomes[step] = {
                            "disposition": "unavailable",
                            "key_cid": key.key_id,
                            "reason_codes": ["dependency_deadlock"],
                        }
                    pending_steps.clear()
                    break

                batch = ready[:max_workers]
                # Cap concurrency to remaining capacity under the plan lease.
                with ThreadPoolExecutor(max_workers=max(1, len(batch))) as pool:
                    futures: dict[Future[tuple[str, CheckRunOutcome, int]], str] = {
                        pool.submit(_run_step, step): step for step in batch
                    }
                    for future in as_completed(futures):
                        step = futures[future]
                        exc = (
                            CancelledError()
                            if future.cancelled()
                            else future.exception()
                        )
                        if exc is None:
                            step, outcome, _timeout_ms = future.result()
                        elif isinstance(exc, Exception):
                            key = step_to_key[step]
                            outcome = CheckRunOutcome(
                                receipt=_receipt_for_status(
                                    key,
                                    TerminalStatus.UNAVAILABLE,
                                    label="executor-error",
                                    reason_codes=("check_runner_error", type(exc).__name__),
                                ),
                                publication_allowed=False,
                                unavailable=True,
                                reason_codes=("check_runner_error",),
                            )
                        else:
                            raise exc

                        key = step_to_key[step]
                        with lock:
                            pending_steps.discard(step)
                            completed_steps.add(step)

                            # Late-success fence: cancellation wins.
                            if (
                                cancel.is_cancelled()
                                and outcome.receipt is not None
                                and (
                                    outcome.receipt.status in _PRODUCTION_SUCCESS
                                    or outcome.publication_allowed
                                )
                            ):
                                late_fenced += 1
                                cancellation_fenced = True
                                if outcome.process is not None:
                                    fence_process_tree(outcome.process)
                                outcome = CheckRunOutcome(
                                    receipt=_receipt_for_status(
                                        key,
                                        TerminalStatus.CANCELLED,
                                        label="late-fenced",
                                        reason_codes=(
                                            "late_receipt_fenced",
                                            "cancelled",
                                        ),
                                        duration_ms=int(
                                            outcome.receipt.execution.duration_ms
                                        ),
                                        command_argv=tuple(
                                            outcome.receipt.execution.command_argv
                                        ),
                                    ),
                                    publication_allowed=False,
                                    cancelled=True,
                                    reason_codes=(
                                        "late_receipt_fenced",
                                        "cancelled",
                                    ),
                                    process_tree_fenced=True,
                                )
                                reason_codes.append("late_receipt_fenced")

                            if outcome.resource_rejection is not None:
                                resource_rejections.append(outcome.resource_rejection)

                            if outcome.cancelled:
                                cancelled = True
                                cancellation_fenced = (
                                    cancellation_fenced or outcome.process_tree_fenced
                                )
                            if outcome.timed_out:
                                timed_out = True
                                if outcome.process_tree_fenced:
                                    cancellation_fenced = True

                            receipt = outcome.receipt
                            if receipt is None:
                                status = (
                                    TerminalStatus.CANCELLED
                                    if outcome.cancelled
                                    else TerminalStatus.TIMEOUT
                                    if outcome.timed_out
                                    else TerminalStatus.UNAVAILABLE
                                )
                                receipt = _receipt_for_status(
                                    key,
                                    status,
                                    label="missing-receipt",
                                    reason_codes=outcome.reason_codes
                                    or (status.value,),
                                )

                            # Advisory never upgrades.
                            receipt = _never_upgrade_advisory(
                                receipt, advisory=key.key_id in advisory
                            )
                            assert receipt is not None

                            # Identity binding must still match the sealed plan.
                            if (
                                receipt.key.repository_tree_cid
                                != plan.repository_tree_cid
                                or receipt.key.environment_cid
                                != plan.environment_cid
                            ):
                                receipt = _receipt_for_status(
                                    key,
                                    TerminalStatus.INVALID,
                                    label="identity-mismatch",
                                    reason_codes=("post_check_identity_mismatch",),
                                )
                                reason_codes.append("post_check_identity_mismatch")

                            # Publication fence for cancelled/timeout/unavailable.
                            if (
                                outcome.cancelled
                                or outcome.timed_out
                                or not outcome.publication_allowed
                            ) and receipt.status in _PRODUCTION_SUCCESS:
                                late_fenced += 1
                                status = (
                                    TerminalStatus.CANCELLED
                                    if outcome.cancelled
                                    else TerminalStatus.TIMEOUT
                                    if outcome.timed_out
                                    else TerminalStatus.UNAVAILABLE
                                )
                                receipt = _receipt_for_status(
                                    key,
                                    status,
                                    label="publication-fenced",
                                    reason_codes=(
                                        "publication_fenced",
                                        *outcome.reason_codes,
                                    ),
                                    duration_ms=int(
                                        receipt.execution.duration_ms
                                    ),
                                    command_argv=tuple(
                                        receipt.execution.command_argv
                                    ),
                                )

                            executed_by_key[key.key_id] = receipt
                            step_outcomes[step] = {
                                "disposition": (
                                    "cancelled"
                                    if outcome.cancelled
                                    else "timeout"
                                    if outcome.timed_out
                                    else "unavailable"
                                    if outcome.unavailable
                                    else "executed"
                                ),
                                "key_cid": key.key_id,
                                "receipt_cid": receipt.receipt_id,
                                "status": receipt.status.value,
                                "publication_allowed": outcome.publication_allowed
                                and receipt.status in _PRODUCTION_SUCCESS,
                                "reason_codes": list(outcome.reason_codes),
                                "process_tree_fenced": outcome.process_tree_fenced,
                            }

                            # Counterexamples for selected failures.
                            if (
                                self._minimize_failures
                                and receipt.status in _FAILURE_STATUSES
                                and key.key_id not in advisory
                            ):
                                cx = self._minimize_failure(receipt)
                                if cx is not None:
                                    counterexamples.append(cx)

                            # Admit production successes when allowed.
                            if (
                                self._admit_successes
                                and self._cache is not None
                                and receipt.status in _PRODUCTION_SUCCESS
                                and production_eligible(receipt)
                                and outcome.publication_allowed
                                and not outcome.cancelled
                                and not outcome.timed_out
                                and key.key_id not in advisory
                            ):
                                try:
                                    self._cache.admit(receipt, for_production=True)
                                except ReceiptCacheError:
                                    reason_codes.append("cache_admit_failed")

            # Remaining unfinished steps after cancel/timeout.
            for step in sorted(pending_steps):
                key = step_to_key[step]
                if key.key_id in executed_by_key:
                    continue
                status = (
                    TerminalStatus.CANCELLED
                    if cancelled
                    else TerminalStatus.TIMEOUT
                    if timed_out
                    else TerminalStatus.UNAVAILABLE
                )
                receipt = _receipt_for_status(
                    key,
                    status,
                    label="unfinished",
                    reason_codes=(status.value, "not_started_after_stop"),
                )
                executed_by_key[key.key_id] = receipt
                step_outcomes[step] = {
                    "disposition": status.value,
                    "key_cid": key.key_id,
                    "receipt_cid": receipt.receipt_id,
                    "status": status.value,
                    "reason_codes": [status.value, "not_started_after_stop"],
                }

            executed = [
                executed_by_key[key.key_id]
                for key in to_execute
                if key.key_id in executed_by_key
            ]

            # ---- post identity revalidation ------------------------------
            if post_observed_identities is None:
                post_obs = pre_obs
            elif isinstance(post_observed_identities, ObservedPlanIdentities):
                post_obs = post_observed_identities
            elif isinstance(post_observed_identities, Mapping):
                post_obs = ObservedPlanIdentities(
                    repository_tree_cid=str(
                        post_observed_identities.get("repository_tree_cid") or ""
                    ),
                    semantic_state_root_cid=str(
                        post_observed_identities.get("semantic_state_root_cid")
                        or ""
                    ),
                    environment_cid=str(
                        post_observed_identities.get("environment_cid") or ""
                    ),
                    dependency_lock_cid=str(
                        post_observed_identities.get("dependency_lock_cid") or ""
                    ),
                    sandbox_id=str(
                        post_observed_identities.get("sandbox_id") or ""
                    ),
                    tool_identity_cids=dict(
                        post_observed_identities.get("tool_identity_cids") or {}
                    ),
                )
            else:
                raise VerificationExecutorError(
                    "post_observed_identities must be ObservedPlanIdentities or mapping",
                    reason_code="invalid_identities",
                )

            post_mismatches = _identity_mismatches(plan, post_obs)
            if post_mismatches:
                reason_codes.append("post_identity_mismatch")
                # Fail closed: do not admit successes when identities drifted.
                downgraded: list[VerificationReceipt] = []
                for receipt in executed:
                    if receipt.status in _PRODUCTION_SUCCESS:
                        downgraded.append(
                            _receipt_for_status(
                                receipt.key,
                                TerminalStatus.INVALID,
                                label="post-identity",
                                reason_codes=("post_identity_mismatch",),
                                duration_ms=int(receipt.execution.duration_ms),
                                command_argv=tuple(receipt.execution.command_argv),
                            )
                        )
                    else:
                        downgraded.append(receipt)
                executed = downgraded
                # Also strip reused leaves from production authority by
                # reclassifying them as executed invalid — but bundle rules
                # require reused to be successful plan-approved hits.  Drop
                # reuse instead and leave unresolved if we cannot re-label.
                if reused:
                    reason_codes.append("reuse_invalidated_by_post_identity")
                    for receipt in reused:
                        executed.append(
                            _receipt_for_status(
                                receipt.key,
                                TerminalStatus.INVALID,
                                label="post-identity-reuse",
                                reason_codes=("post_identity_mismatch",),
                            )
                        )
                    reused = []

            return self._finalize(
                plan=plan,
                reused=reused,
                executed=executed,
                counterexamples=tuple(counterexamples),
                resource_rejections=resource_rejections,
                cancelled=cancelled,
                timed_out=timed_out,
                cancellation_fenced=cancellation_fenced,
                late_receipts_fenced=late_fenced,
                pre_obs=pre_obs,
                post_obs=post_obs,
                pre_mismatches=pre_mismatches,
                post_mismatches=post_mismatches,
                advisory=advisory,
                step_outcomes=step_outcomes,
                reason_codes=reason_codes,
                tombstones=tuple(tombstones),
                started=started,
                human_review_required=human_review_required,
                model_route_decision=model_route_decision,
                routing_hints=routing_hints,
            )
        finally:
            if plan_lease is not None:
                self._scheduler.release(
                    plan_lease, reason="verification_plan_complete"
                )

    # -- internals ---------------------------------------------------------

    def _acquire_plan_lease(
        self, plan: VerificationPlan
    ) -> tuple[ResourceAdmissionLease | None, ResourceRejection | None]:
        requirement = LaneResourceRequirements(
            lane_id=f"{DEFAULT_LANE_PREFIX}:{plan.plan_id[:24]}",
            stage=DEFAULT_STAGE,
            resource_class=DEFAULT_RESOURCE_CLASS,
            process_slots=max(1, int(plan.expected_processes)),
            requires_provider=False,
            memory_bytes=max(0, int(plan.expected_memory_bytes)),
            disk_bytes=max(0, int(plan.expected_artifact_bytes)),
        )
        host = self._host_snapshot
        if host is None:
            host = _default_host_snapshot(
                process_slots=max(8, int(plan.expected_processes) * 2)
            )
        elif not isinstance(host, HostResourceSnapshot):
            host = HostResourceSnapshot.from_mapping(host)
        decision, lease = self._scheduler.acquire(requirement, host=host)
        if lease is None or not decision.admitted:
            reasons = tuple(decision.reasons) or ("resource_admission_denied",)
            kind = ResourceRejectionKind.PLAN_LEASE_DENIED
            if any("capacity" in str(item).lower() for item in reasons):
                kind = ResourceRejectionKind.CAPACITY_EXHAUSTED
            rejection = ResourceRejection(
                kind=kind,
                reason_codes=reasons,
                details={
                    "lane_id": decision.lane_id,
                    "resource_class": decision.resource_class,
                    "host_available_slots": decision.host_available_slots,
                    "effective_slots": decision.effective_slots,
                },
            )
            return None, rejection
        return lease, None

    def _execute_one(
        self,
        key: VerificationReceiptKey,
        *,
        step_id: str,
        timeout_ms: int,
        cancellation: VerificationCancellation | None,
        plan: VerificationPlan,
        advisory: bool,
    ) -> CheckRunOutcome:
        if cancellation is not None and cancellation.is_cancelled():
            return CheckRunOutcome(
                receipt=_receipt_for_status(
                    key,
                    TerminalStatus.CANCELLED,
                    label="cancelled",
                    reason_codes=("cancelled",),
                ),
                publication_allowed=False,
                cancelled=True,
                reason_codes=("cancelled",),
                process_tree_fenced=True,
            )

        runner = self._resolve_runner(key.receipt_kind)
        if runner is None:
            # Fail-closed unavailable when no tool/adapter is bound.
            return CheckRunOutcome(
                receipt=_receipt_for_status(
                    key,
                    TerminalStatus.UNAVAILABLE,
                    label="tool-unavailable",
                    reason_codes=("tool_unavailable", "no_check_runner"),
                ),
                publication_allowed=False,
                unavailable=True,
                reason_codes=("tool_unavailable", "no_check_runner"),
            )

        outcome = runner(
            key,
            step_id=step_id,
            timeout_ms=timeout_ms,
            cancellation=cancellation,
            plan=plan,
        )

        if not isinstance(outcome, CheckRunOutcome):
            raise VerificationExecutorError(
                "check runner must return CheckRunOutcome",
                reason_code="invalid_check_outcome",
            )

        if outcome.receipt is not None and outcome.receipt.key.key_id != key.key_id:
            return CheckRunOutcome(
                receipt=_receipt_for_status(
                    key,
                    TerminalStatus.INVALID,
                    label="key-mismatch",
                    reason_codes=("check_runner_key_mismatch",),
                ),
                publication_allowed=False,
                reason_codes=("check_runner_key_mismatch",),
            )

        if advisory and outcome.receipt is not None:
            receipt = _never_upgrade_advisory(outcome.receipt, advisory=True)
            return CheckRunOutcome(
                receipt=receipt,
                publication_allowed=False,
                cancelled=outcome.cancelled,
                timed_out=outcome.timed_out,
                unavailable=outcome.unavailable,
                reason_codes=tuple(outcome.reason_codes) + ("advisory_obligation",),
                process_tree_fenced=outcome.process_tree_fenced,
                process=outcome.process,
                resource_rejection=outcome.resource_rejection,
            )
        return outcome

    def _resolve_runner(
        self, kind: VerificationReceiptKind
    ) -> CheckRunner | None:
        if kind in self._check_runners:
            return self._check_runners[kind]
        return self._check_runner

    def _minimize_failure(
        self, receipt: VerificationReceipt
    ) -> CounterexampleReceipt | None:
        try:
            obligation = (
                receipt.key.proof_obligation_cid
                if receipt.key.receipt_kind is VerificationReceiptKind.PROOF
                else ""
            )
            result = minimize_counterexample(
                receipt,
                failed_obligation_cid=obligation,
            )
            cx = result.receipt
            # Bundle contracts forbid non-proof counterexamples from naming any
            # obligation CID (including the canonical not-applicable sentinel).
            if (
                receipt.key.receipt_kind is not VerificationReceiptKind.PROOF
                and cx.failed_obligation_cid
            ):
                payload = dict(cx.to_record())
                payload["failed_obligation_cid"] = ""
                payload.pop("counterexample_id", None)
                payload.pop("content_id", None)
                cx = CounterexampleReceipt.from_dict(payload)
            return cx
        except (
            AttributeError,
            KeyError,
            VerificationContractError,
            TypeError,
            ValueError,
        ):
            return None

    def _choose_route(
        self,
        plan: VerificationPlan,
        *,
        model_route_decision: ModelRouteDecision | None,
        routing_hints: Mapping[str, Any] | None,
        force_human: bool = False,
    ) -> ModelRouteDecision:
        if model_route_decision is not None:
            return model_route_decision
        if self._model_route_decision is not None:
            return self._model_route_decision
        if force_human:
            return ModelRouteDecision(
                route=ModelRoute.HUMAN_REVIEW_REQUIRED,
                considered_routes=(
                    ModelRoute.SMALL_LOCAL_MODEL,
                    ModelRoute.HUMAN_REVIEW_REQUIRED,
                ),
                decisive_reason_codes=("human_review_required",),
                required_capabilities=("human_review",),
                context_token_estimate=0,
                policy_cid=plan.policy_cid,
            )
        policy = self._model_route_policy
        if policy is None:
            policy = {
                "policy_cid": policy_cid_for("verification-executor-default"),
                "max_context_tokens": 32_768,
            }
        inventory = self._available_models
        if inventory is None:
            inventory = default_inventory()
        context = self._context_pack if self._context_pack is not None else {}
        try:
            facts = derive_model_route_facts(
                context, plan, routing_hints=routing_hints
            )
            return decide_model_route(
                facts,
                prior_attempts=self._prior_attempts,
                available_models=inventory,
                policy=policy,
            )
        except (KeyError, ModelRouteError, TypeError, ValueError):
            # Fail closed to a deterministic small-local route when routing
            # inputs are incomplete; never invent provider identity.
            return ModelRouteDecision(
                route=ModelRoute.SMALL_LOCAL_MODEL,
                considered_routes=(ModelRoute.SMALL_LOCAL_MODEL,),
                decisive_reason_codes=("executor_default_localized_route",),
                required_capabilities=("bounded_context",),
                context_token_estimate=0,
                policy_cid=plan.policy_cid,
            )

    def _terminal_resource_rejection(
        self,
        *,
        plan: VerificationPlan,
        rejection: ResourceRejection,
        pre_obs: ObservedPlanIdentities,
        advisory: frozenset[str],
        cancel: VerificationCancellation,
        started: float,
        reason_codes: Sequence[str],
        human_review_required: bool | None,
        model_route_decision: ModelRouteDecision | None,
        routing_hints: Mapping[str, Any] | None,
    ) -> VerificationExecutionResult:
        executed = [
            _receipt_for_status(
                key,
                TerminalStatus.UNAVAILABLE,
                label="resource-rejected",
                reason_codes=(rejection.kind.value, *rejection.reason_codes),
            )
            for key in plan.required_receipt_keys
        ]
        # No reuse when the plan cannot even acquire its lease.
        return self._finalize(
            plan=plan,
            reused=(),
            executed=executed,
            counterexamples=(),
            resource_rejections=(rejection,),
            cancelled=False,
            timed_out=False,
            cancellation_fenced=False,
            late_receipts_fenced=0,
            pre_obs=pre_obs,
            post_obs=pre_obs,
            pre_mismatches=(),
            post_mismatches=(),
            advisory=advisory,
            step_outcomes={
                key.key_id: {
                    "disposition": "resource_rejected",
                    "key_cid": key.key_id,
                    "status": TerminalStatus.UNAVAILABLE.value,
                    "reason_codes": [rejection.kind.value],
                }
                for key in plan.required_receipt_keys
            },
            reason_codes=list(reason_codes),
            tombstones=(),
            started=started,
            human_review_required=human_review_required,
            model_route_decision=model_route_decision,
            routing_hints=routing_hints,
        )

    def _finalize(
        self,
        *,
        plan: VerificationPlan,
        reused: Sequence[VerificationReceipt],
        executed: Sequence[VerificationReceipt],
        counterexamples: Sequence[CounterexampleReceipt],
        resource_rejections: Sequence[ResourceRejection],
        cancelled: bool,
        timed_out: bool,
        cancellation_fenced: bool,
        late_receipts_fenced: int,
        pre_obs: ObservedPlanIdentities,
        post_obs: ObservedPlanIdentities,
        pre_mismatches: Sequence[str],
        post_mismatches: Sequence[str],
        advisory: frozenset[str],
        step_outcomes: Mapping[str, Mapping[str, Any]],
        reason_codes: Sequence[str],
        tombstones: Sequence[str],
        started: float,
        human_review_required: bool | None,
        model_route_decision: ModelRouteDecision | None,
        routing_hints: Mapping[str, Any] | None,
    ) -> VerificationExecutionResult:
        # Deduplicate by key (last write wins for executed; reused is authoritative
        # only when still present).
        reused_list = list(reused)
        executed_list = list(executed)
        reused_ids = {item.key.key_id for item in reused_list}
        executed_list = [
            item for item in executed_list if item.key.key_id not in reused_ids
        ]

        review_flag = human_review_required
        if review_flag is None:
            review_flag = bool(plan.human_review_required)
        elif plan.human_review_required and not review_flag:
            review_flag = True

        bundle = build_verification_bundle(
            plan,
            reused_receipts=reused_list,
            executed_receipts=executed_list,
            counterexamples=counterexamples,
            human_review_required=review_flag,
        )

        route = self._choose_route(
            plan,
            model_route_decision=model_route_decision,
            routing_hints=routing_hints,
            force_human=bool(review_flag),
        )
        # Route cannot downgrade plan/bundle human review.
        if bundle.human_review_required and not route.requires_human_review:
            route = ModelRouteDecision(
                route=ModelRoute.HUMAN_REVIEW_REQUIRED,
                considered_routes=tuple(
                    dict.fromkeys(
                        (
                            *route.considered_routes,
                            ModelRoute.HUMAN_REVIEW_REQUIRED,
                        )
                    )
                ),
                decisive_reason_codes=tuple(
                    dict.fromkeys(
                        (
                            *route.decisive_reason_codes,
                            "bundle_human_review_required",
                        )
                    )
                ),
                required_capabilities=tuple(
                    dict.fromkeys((*route.required_capabilities, "human_review"))
                ),
                context_token_estimate=route.context_token_estimate,
                policy_cid=route.policy_cid,
            )

        wall_ms = int(max(0.0, (_monotonic(self._clock) - started) * 1000.0))
        summary = build_verification_summary(
            bundle,
            route,
            verification_wall_time_ms=wall_ms,
        )
        commitment = build_verification_commitment(bundle)

        acceptance = compute_production_acceptance(
            bundle, advisory_key_cids=tuple(advisory)
        )
        # Hard gates that acceptance must never ignore.
        if cancelled or timed_out or resource_rejections or post_mismatches:
            acceptance = False
        if any(
            item.status
            in {
                TerminalStatus.TIMEOUT,
                TerminalStatus.UNAVAILABLE,
                TerminalStatus.CANCELLED,
                TerminalStatus.SIMULATED,
                TerminalStatus.STALE,
                TerminalStatus.INVALID,
            }
            and item.key.key_id not in advisory
            for item in bundle.receipts
        ):
            acceptance = False

        advisory_unresolved = tuple(
            sorted(
                key_id
                for key_id in advisory
                if key_id in set(bundle.unresolved_requirement_ids)
                or any(
                    receipt.key.key_id == key_id
                    and receipt.status not in _PRODUCTION_SUCCESS
                    for receipt in bundle.receipts
                )
            )
        )

        identity = IdentityRevalidation(
            pre_matched=not pre_mismatches,
            post_matched=not post_mismatches,
            pre_mismatches=tuple(pre_mismatches),
            post_mismatches=tuple(post_mismatches),
            pre=pre_obs.to_dict(),
            post=post_obs.to_dict(),
        )

        return VerificationExecutionResult(
            verification_plan=plan,
            bundle=bundle,
            summary=summary,
            commitment=commitment,
            model_route_decision=route,
            production_acceptance=acceptance,
            reused_receipts=tuple(reused_list),
            executed_receipts=tuple(executed_list),
            counterexamples=tuple(bundle.counterexamples),
            resource_rejections=tuple(resource_rejections),
            cancelled=cancelled,
            timed_out=timed_out,
            cancellation_fenced=cancellation_fenced,
            late_receipts_fenced=int(late_receipts_fenced),
            identity_revalidation=identity,
            advisory_unresolved_key_cids=advisory_unresolved,
            step_outcomes=step_outcomes,
            reason_codes=tuple(dict.fromkeys(str(item) for item in reason_codes)),
            wall_time_ms=wall_ms,
            tombstones_published=tuple(tombstones),
        )


def execute_verification_plan(
    verification_plan: VerificationPlan | Mapping[str, Any],
    *,
    observed_identities: ObservedPlanIdentities | Mapping[str, Any] | None = None,
    post_observed_identities: (
        ObservedPlanIdentities | Mapping[str, Any] | None
    ) = None,
    cancellation: VerificationCancellation | None = None,
    step_bindings: Mapping[str, str] | None = None,
    advisory_key_cids: Sequence[str] = (),
    human_review_required: bool | None = None,
    model_route_decision: ModelRouteDecision | None = None,
    routing_hints: Mapping[str, Any] | None = None,
    cache: VerificationReceiptCache | None = None,
    process_runner: VerificationProcessRunner | None = None,
    resource_scheduler: ResourceScheduler | None = None,
    host_snapshot: HostResourceSnapshot | Mapping[str, Any] | None = None,
    check_runner: CheckRunner | None = None,
    check_runners: Mapping[VerificationReceiptKind, CheckRunner] | None = None,
    require_resource_lease: bool = True,
    admit_successes: bool = True,
    minimize_failures: bool = True,
) -> VerificationExecutionResult:
    """Module-level entry point for IVP-014 plan execution."""

    executor = VerificationExecutor(
        cache=cache,
        process_runner=process_runner,
        resource_scheduler=resource_scheduler,
        host_snapshot=host_snapshot,
        check_runner=check_runner,
        check_runners=check_runners,
        model_route_decision=model_route_decision,
        require_resource_lease=require_resource_lease,
        admit_successes=admit_successes,
        minimize_failures=minimize_failures,
    )
    return executor.execute(
        verification_plan,
        observed_identities=observed_identities,
        post_observed_identities=post_observed_identities,
        cancellation=cancellation,
        step_bindings=step_bindings,
        advisory_key_cids=advisory_key_cids,
        human_review_required=human_review_required,
        model_route_decision=model_route_decision,
        routing_hints=routing_hints,
    )


def create_verification_executor(
    **kwargs: Any,
) -> VerificationExecutor:
    """Construct a :class:`VerificationExecutor` (factory for tests/wiring)."""

    return VerificationExecutor(**kwargs)


__all__ = [
    "EXECUTION_BUNDLE_EVIDENCE",
    "EXECUTION_RESULT_SCHEMA",
    "VERIFICATION_EXECUTOR_INTERFACE",
    "VERIFICATION_EXECUTOR_SCHEMA",
    "CheckRunOutcome",
    "CheckRunner",
    "IdentityRevalidation",
    "ObservedPlanIdentities",
    "ResourceRejection",
    "ResourceRejectionKind",
    "VerificationExecutionResult",
    "VerificationExecutor",
    "VerificationExecutorError",
    "VerificationExecutorIdentityError",
    "VerificationResourceRejectionError",
    "compute_production_acceptance",
    "create_verification_executor",
    "execute_verification_plan",
    "fence_process_tree",
]
