"""Paired compressed/expanded shadow execution in isolated worktrees (SCG-026).

``ShadowExecutor`` and :func:`execute_shadow_plan` run separately bound
compressed and expanded attempts under fenced evaluation worktrees, rechecking
budgets and disclosure **before every provider invocation**.

Normative fail-closed invariants:

* Expanded output never auto-accepts (oracle / candidate only).
* Budgets (wall time, model spend, expansion tokens) and disclosure policy are
  rechecked immediately before invocation — plan admission is not enough.
* Cancellation and timeouts leave the production checkout unchanged.
* No production checkout edits; evaluation mutations stay inside disposable
  worktrees with lease/fence identity.
* Simulated provenance cannot claim production acceptance.

Conflict policy: reuses ResourceScheduler admission seams, provider privacy
gates, semantic work scheduling concepts, and isolated worktree lifecycle
identity. Does not rebuild the harness loop or mint a new receipt hierarchy.

Importing this module performs no I/O, opens no sockets, and never invokes a
provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Final, Mapping, Protocol, Sequence
import threading
import time
import unicodedata

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
    validate_structured_value,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    ArtifactProvenance,
    AssumptionKind,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    GovernorArtifactHeader,
    GovernorAssumption,
    GovernorTerminalStatus,
    SemanticGovernorBaseError,
)

from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    AcceptanceDisposition,
    AttemptTerminalStatus,
    CostTimingProjection,
    PairedAttemptRecord,
    SHADOW_EXECUTION_RESULT_INTERFACE,
    ShadowAttemptRole,
    ShadowExecutionPlan,
    ShadowExecutionResult,
    ShadowSelectionReason,
    VerificationProjection,
    assert_expanded_never_accepted,
    verify_plan_identity,
    verify_result_identity,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.privacy import (
    DisclosureDisposition,
    DisclosureForbiddenError,
    ProviderLocality,
    ShadowDisclosurePolicy,
    WorktreePolicyError,
    assert_isolated_evaluation_worktree,
    classify_provider_locality,
    default_shadow_disclosure_policy,
    prepare_provider_invocation,
)

# ---------------------------------------------------------------------------
# Evidence / interface / schema constants
# ---------------------------------------------------------------------------

SCG_SHADOW_RUN_EVIDENCE: Final[str] = "scg/shadow-run@1"
SHADOW_EXECUTOR_INTERFACE: Final[str] = "ShadowExecutor@1"
EXECUTE_SHADOW_PLAN_INTERFACE: Final[str] = "execute_shadow_plan@1"

SHADOW_ATTEMPT_INVOCATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "shadow-attempt-invocation@1"
)
SHADOW_ATTEMPT_PROPOSAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "shadow-attempt-proposal@1"
)
EVALUATION_WORKTREE_HANDLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "evaluation-worktree-handle@1"
)

GENERATOR_ID: Final[str] = "semantic_governor_shadow"
GENERATOR_VERSION: Final[str] = "1.0.0"

DEFAULT_COMPRESSED_PROVIDER_ID: Final[str] = "local:shadow.compressed"
DEFAULT_EXPANDED_PROVIDER_ID: Final[str] = "local:shadow.expanded"
DEFAULT_EXTERNAL_PROVIDER_ID: Final[str] = "external.shadow.unapproved"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_REASON_CODES: Final[int] = 256
MAX_METADATA_KEYS: Final[int] = 64
MAX_WORKTREE_ID_CHARS: Final[int] = 128

_TOKEN_RE_SOURCE: Final[str] = r"^[a-z][a-z0-9_.:/+-]{0,127}$"
_WORKTREE_ID_RE_SOURCE: Final[str] = r"^[A-Za-z0-9][A-Za-z0-9_.:/+@-]{0,127}$"

import re as _re

_TOKEN_RE: Final[_re.Pattern[str]] = _re.compile(_TOKEN_RE_SOURCE)
_WORKTREE_ID_RE: Final[_re.Pattern[str]] = _re.compile(_WORKTREE_ID_RE_SOURCE)
_TASK_ID_RE: Final[_re.Pattern[str]] = _re.compile(
    r"^[A-Za-z][A-Za-z0-9_.:/+-]{0,127}$"
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SemanticGovernorShadowError(SemanticGovernorBaseError):
    """Raised when shadow execution is malformed or fail-closed."""


class PlanAdmissionError(SemanticGovernorShadowError):
    """Shadow plan is not admissible for paired execution."""


class BudgetExceededError(SemanticGovernorShadowError):
    """Wall-time, model-spend, or expansion-token budget exhausted."""


class DisclosureRecheckError(SemanticGovernorShadowError):
    """Disclosure recheck refused the intended provider invocation."""


class ProductionStateMutatedError(SemanticGovernorShadowError):
    """Production checkout changed during shadow evaluation."""


class ShadowCancellationError(SemanticGovernorShadowError):
    """Shadow run was cancelled before completion."""


class ShadowTimeoutError(SemanticGovernorShadowError):
    """Shadow run exceeded the plan wall-time budget."""


class WorktreeLifecycleError(SemanticGovernorShadowError):
    """Isolated evaluation worktree lifecycle failed."""


# ---------------------------------------------------------------------------
# Closed enums
# ---------------------------------------------------------------------------


class ShadowRunPhase(str, Enum):
    """Closed phases of a paired shadow execution."""

    ADMITTED = "admitted"
    COMPRESSED_RUNNING = "compressed_running"
    COMPRESSED_DONE = "compressed_done"
    EXPANDED_RUNNING = "expanded_running"
    EXPANDED_DONE = "expanded_done"
    EXPANDED_SKIPPED = "expanded_skipped"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    COMPLETE = "complete"
    FAILED = "failed"


class BudgetKind(str, Enum):
    """Closed budget dimensions rechecked before invocation."""

    WALL_TIME_MS = "wall_time_ms"
    MODEL_SPEND_MICROS = "model_spend_micros"
    EXPANSION_TOKEN_BUDGET = "expansion_token_budget"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise SemanticGovernorShadowError(f"{name} must be a nonempty string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise SemanticGovernorShadowError(f"{name} must be trimmed NFC text")
    if len(value) > MAX_TEXT_CHARS or any(not char.isprintable() for char in value):
        raise SemanticGovernorShadowError(f"{name} contains invalid text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise SemanticGovernorShadowError(
            f"{name} must be a lowercase token matching {_TOKEN_RE_SOURCE}"
        )
    return text


def _task_id(value: Any, name: str = "task_id") -> str:
    text = _text(value, name)
    if _TASK_ID_RE.fullmatch(text) is None:
        raise SemanticGovernorShadowError(f"{name} must match task id pattern")
    return text


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise SemanticGovernorShadowError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise SemanticGovernorShadowError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise SemanticGovernorShadowError(f"{name} must be a nonnegative integer")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        raise SemanticGovernorShadowError(
            f"{name} has unsupported value {value!r}"
        ) from exc


def _worktree_id(value: Any, name: str = "worktree_id") -> str:
    text = _text(value, name)
    if len(text) > MAX_WORKTREE_ID_CHARS:
        raise SemanticGovernorShadowError(f"{name} exceeds maximum length")
    if _WORKTREE_ID_RE.fullmatch(text) is None:
        raise SemanticGovernorShadowError(
            f"{name} must match managed worktree id pattern"
        )
    return text


def _optional_worktree_id(value: Any, name: str = "worktree_id") -> str | None:
    if value is None:
        return None
    return _worktree_id(value, name)


def _freeze_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_structured(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_structured(item) for item in value)
    return value


def _thaw_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_structured(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_structured(item) for item in value]
    return value


def _mapping(value: Any, name: str, *, frozen: bool = True) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticGovernorShadowError(f"{name} must be a mapping")
    if len(value) > MAX_METADATA_KEYS:
        raise SemanticGovernorShadowError(f"{name} exceeds maximum key count")
    try:
        validate_structured_value(dict(value), path=name)
    except Exception as exc:
        raise SemanticGovernorShadowError(
            f"{name} must be strict DAG-JSON without floats or host types"
        ) from exc
    return _freeze_structured(dict(value)) if frozen else dict(value)


def _unique_sorted_tokens(
    values: Sequence[Any], name: str, *, max_items: int
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise SemanticGovernorShadowError(f"{name} must be a list")
    ordered = tuple(sorted(_token(value, name) for value in values))
    if len(ordered) > max_items:
        raise SemanticGovernorShadowError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise SemanticGovernorShadowError(f"{name} must not contain duplicates")
    return ordered


# ---------------------------------------------------------------------------
# Cancellation / clock / production fence
# ---------------------------------------------------------------------------


class ShadowCancellationToken:
    """Thread-safe cancellation flag for a single shadow run."""

    def __init__(self, *, cancelled: bool = False) -> None:
        self._lock = threading.Lock()
        self._cancelled = bool(cancelled)
        self._reason: str | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            self._cancelled = True
            self._reason = _text(reason, "reason")

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def raise_if_cancelled(self) -> None:
        with self._lock:
            if self._cancelled:
                raise ShadowCancellationError(
                    self._reason or "shadow execution cancelled"
                )


class MonotonicClock:
    """Injectable wall clock (milliseconds)."""

    def now_ms(self) -> int:
        return int(time.monotonic() * 1000)


@dataclass
class ProductionCheckoutGuard:
    """Fences the production checkout against shadow-side mutation.

    Callers supply a content-addressed *fingerprint* of production refs (for
    example HEAD + index + protected path digests). The guard never writes to
    production; it only detects change.
    """

    fingerprint: str
    production_mutation_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fingerprint", _text(self.fingerprint, "fingerprint")
        )
        object.__setattr__(
            self,
            "production_mutation_count",
            _nonneg_int(self.production_mutation_count, "production_mutation_count"),
        )

    def record_production_mutation(self) -> None:
        """Test/diagnostic hook: marks that production was mutated (forbidden)."""

        with self._lock:
            self.production_mutation_count += 1

    def current_fingerprint(self) -> str:
        """Return the observed production fingerprint.

        Default implementation returns the bound fingerprint. Subclasses or
        wrappers may override via assignment of ``observe`` callable.
        """

        observe = getattr(self, "observe", None)
        if callable(observe):
            value = observe()
            return _text(value, "observed_fingerprint")
        return self.fingerprint

    def assert_unchanged(self, *, phase: str = "shadow") -> None:
        current = self.current_fingerprint()
        with self._lock:
            mutations = self.production_mutation_count
        if current != self.fingerprint or mutations != 0:
            raise ProductionStateMutatedError(
                f"production checkout changed during {phase}: "
                f"expected fingerprint {self.fingerprint!r}, got {current!r}, "
                f"mutations={mutations}"
            )


def production_fingerprint_from_refs(
    *,
    repository_state_cid: str,
    head_commit: str | None = None,
    protected_path_digests: Mapping[str, str] | None = None,
) -> str:
    """Build a deterministic production fingerprint from closed refs only."""

    payload = {
        "kind": "production_checkout_fingerprint",
        "repository_state_cid": _cid(repository_state_cid, "repository_state_cid"),
        "head_commit": _optional_text(head_commit, "head_commit"),
        "protected_path_digests": dict(protected_path_digests or {}),
    }
    return cid_for_structured(payload)


# ---------------------------------------------------------------------------
# Budget ledger (rechecked before every invocation)
# ---------------------------------------------------------------------------


@dataclass
class ShadowBudgetLedger:
    """Mutable remaining budgets for a paired shadow run.

    Zero remaining on a dimension fails closed before provider invocation.
    """

    max_wall_time_ms: int
    max_model_spend_micros: int
    max_expansion_token_budget: int
    spent_wall_time_ms: int = 0
    spent_model_spend_micros: int = 0
    spent_expansion_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.max_wall_time_ms = _nonneg_int(self.max_wall_time_ms, "max_wall_time_ms")
        self.max_model_spend_micros = _nonneg_int(
            self.max_model_spend_micros, "max_model_spend_micros"
        )
        self.max_expansion_token_budget = _nonneg_int(
            self.max_expansion_token_budget, "max_expansion_token_budget"
        )

    @classmethod
    def from_plan(cls, plan: ShadowExecutionPlan) -> "ShadowBudgetLedger":
        return cls(
            max_wall_time_ms=int(plan.max_wall_time_ms),
            max_model_spend_micros=int(plan.max_model_spend_micros),
            max_expansion_token_budget=int(plan.max_expansion_token_budget),
        )

    @property
    def remaining_wall_time_ms(self) -> int:
        with self._lock:
            return max(0, self.max_wall_time_ms - self.spent_wall_time_ms)

    @property
    def remaining_model_spend_micros(self) -> int:
        with self._lock:
            return max(0, self.max_model_spend_micros - self.spent_model_spend_micros)

    @property
    def remaining_expansion_tokens(self) -> int:
        with self._lock:
            return max(
                0, self.max_expansion_token_budget - self.spent_expansion_tokens
            )

    def snapshot(self) -> Mapping[str, int]:
        with self._lock:
            return {
                "max_wall_time_ms": self.max_wall_time_ms,
                "max_model_spend_micros": self.max_model_spend_micros,
                "max_expansion_token_budget": self.max_expansion_token_budget,
                "spent_wall_time_ms": self.spent_wall_time_ms,
                "spent_model_spend_micros": self.spent_model_spend_micros,
                "spent_expansion_tokens": self.spent_expansion_tokens,
                "remaining_wall_time_ms": max(
                    0, self.max_wall_time_ms - self.spent_wall_time_ms
                ),
                "remaining_model_spend_micros": max(
                    0, self.max_model_spend_micros - self.spent_model_spend_micros
                ),
                "remaining_expansion_tokens": max(
                    0, self.max_expansion_token_budget - self.spent_expansion_tokens
                ),
            }

    def recheck_before_invocation(
        self,
        *,
        role: str,
        estimated_input_tokens: int = 0,
        estimated_model_spend_micros: int = 0,
        estimated_wall_time_ms: int = 0,
    ) -> None:
        """Fail closed when remaining budgets cannot cover the next call."""

        role_value = _enum(role, ShadowAttemptRole, "role")
        est_tokens = _nonneg_int(estimated_input_tokens, "estimated_input_tokens")
        est_spend = _nonneg_int(
            estimated_model_spend_micros, "estimated_model_spend_micros"
        )
        est_wall = _nonneg_int(estimated_wall_time_ms, "estimated_wall_time_ms")

        with self._lock:
            if self.max_wall_time_ms == 0:
                raise BudgetExceededError(
                    f"{BudgetKind.WALL_TIME_MS.value} budget is zero"
                )
            remaining_wall = self.max_wall_time_ms - self.spent_wall_time_ms
            if remaining_wall <= 0 or (est_wall and est_wall > remaining_wall):
                raise BudgetExceededError(
                    f"{BudgetKind.WALL_TIME_MS.value} budget exhausted "
                    f"(remaining={max(0, remaining_wall)})"
                )

            remaining_spend = (
                self.max_model_spend_micros - self.spent_model_spend_micros
            )
            # Zero max spend is allowed for pure local/static evaluation, but
            # a positive estimate against a zero remaining spend fails closed.
            if est_spend > 0 and est_spend > remaining_spend:
                raise BudgetExceededError(
                    f"{BudgetKind.MODEL_SPEND_MICROS.value} budget exhausted "
                    f"(remaining={max(0, remaining_spend)})"
                )

            if role_value == ShadowAttemptRole.EXPANDED.value:
                if self.max_expansion_token_budget == 0:
                    raise BudgetExceededError(
                        f"{BudgetKind.EXPANSION_TOKEN_BUDGET.value} budget is zero"
                    )
                remaining_tokens = (
                    self.max_expansion_token_budget - self.spent_expansion_tokens
                )
                if remaining_tokens <= 0 or (
                    est_tokens and est_tokens > remaining_tokens
                ):
                    raise BudgetExceededError(
                        f"{BudgetKind.EXPANSION_TOKEN_BUDGET.value} budget "
                        f"exhausted (remaining={max(0, remaining_tokens)})"
                    )

    def record_cost(
        self,
        cost: CostTimingProjection,
        *,
        role: str,
        count_expansion_tokens: bool = True,
    ) -> None:
        role_value = _enum(role, ShadowAttemptRole, "role")
        if not isinstance(cost, CostTimingProjection):
            raise SemanticGovernorShadowError("cost must be CostTimingProjection")
        with self._lock:
            self.spent_wall_time_ms += int(cost.wall_time_ms)
            self.spent_model_spend_micros += int(cost.model_spend_micros)
            if (
                count_expansion_tokens
                and role_value == ShadowAttemptRole.EXPANDED.value
            ):
                self.spent_expansion_tokens += int(cost.input_tokens) + int(
                    cost.output_tokens
                )


# ---------------------------------------------------------------------------
# Isolated evaluation worktree lifecycle (managed ids; no production edits)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvaluationWorktreeHandle:
    """Managed evaluation worktree identity (never a production checkout)."""

    worktree_id: str
    role: str
    task_id: str
    attempt_index: int
    lease_id: str
    fence: int
    isolated: bool = True
    released: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "worktree_id", _worktree_id(self.worktree_id, "worktree_id")
        )
        object.__setattr__(
            self, "role", _enum(self.role, ShadowAttemptRole, "role")
        )
        object.__setattr__(self, "task_id", _task_id(self.task_id))
        object.__setattr__(
            self,
            "attempt_index",
            _nonneg_int(self.attempt_index, "attempt_index"),
        )
        object.__setattr__(self, "lease_id", _token(self.lease_id, "lease_id"))
        object.__setattr__(self, "fence", _nonneg_int(self.fence, "fence"))
        object.__setattr__(self, "isolated", _bool(self.isolated, "isolated"))
        object.__setattr__(self, "released", _bool(self.released, "released"))
        if not self.isolated:
            raise WorktreeLifecycleError(
                "evaluation worktree must be isolated from production"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EVALUATION_WORKTREE_HANDLE_SCHEMA,
            "worktree_id": self.worktree_id,
            "role": self.role,
            "task_id": self.task_id,
            "attempt_index": self.attempt_index,
            "lease_id": self.lease_id,
            "fence": self.fence,
            "isolated": self.isolated,
            "released": self.released,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()


class EvaluationWorktreeLifecycle(Protocol):
    """Protocol for disposable evaluation worktree create/release."""

    def create(
        self,
        *,
        role: str,
        task_id: str,
        attempt_index: int,
        plan_cid: str,
    ) -> EvaluationWorktreeHandle: ...

    def release(self, handle: EvaluationWorktreeHandle) -> EvaluationWorktreeHandle: ...

    def active_ids(self) -> tuple[str, ...]: ...


class InMemoryEvaluationWorktreeLifecycle:
    """Hermetic worktree lifecycle using managed ids only.

    Does not invoke git, touch the production checkout, or open host paths on
    the privacy surface. Suitable for unit tests and as the default executor
    backend when a real IsolatedWorktree adapter is not injected.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, EvaluationWorktreeHandle] = {}
        self._fence = 0
        self._create_count = 0
        self._release_count = 0
        self._created_ids: list[str] = []

    @property
    def create_count(self) -> int:
        return self._create_count

    @property
    def release_count(self) -> int:
        return self._release_count

    def create(
        self,
        *,
        role: str,
        task_id: str,
        attempt_index: int,
        plan_cid: str,
    ) -> EvaluationWorktreeHandle:
        role_value = _enum(role, ShadowAttemptRole, "role")
        tid = _task_id(task_id)
        plan = _cid(plan_cid, "plan_cid")
        with self._lock:
            self._fence += 1
            self._create_count += 1
            digest = cid_for_structured(
                {
                    "kind": "evaluation_worktree",
                    "role": role_value,
                    "task_id": tid,
                    "attempt_index": int(attempt_index),
                    "plan_cid": plan,
                    "fence": self._fence,
                }
            )
            # Managed id: alphanumeric + separators only (no host path).
            short = digest.replace("baguqeera", "").replace("bafkrei", "")[:20]
            if not short:
                short = f"{self._fence:08d}"
            worktree_id = f"eval-{role_value}-{short}"
            # Ensure uniqueness under concurrent creates.
            if worktree_id in self._active:
                worktree_id = f"{worktree_id}-{self._fence}"
            handle = EvaluationWorktreeHandle(
                worktree_id=worktree_id,
                role=role_value,
                task_id=tid,
                attempt_index=int(attempt_index),
                lease_id=f"lease.{role_value}.{self._fence}",
                fence=self._fence,
                isolated=True,
                released=False,
            )
            self._active[worktree_id] = handle
            self._created_ids.append(worktree_id)
            return handle

    def release(self, handle: EvaluationWorktreeHandle) -> EvaluationWorktreeHandle:
        if not isinstance(handle, EvaluationWorktreeHandle):
            raise WorktreeLifecycleError("handle must be EvaluationWorktreeHandle")
        with self._lock:
            current = self._active.get(handle.worktree_id)
            if current is None:
                raise WorktreeLifecycleError(
                    f"unknown evaluation worktree {handle.worktree_id!r}"
                )
            if (
                current.lease_id != handle.lease_id
                or current.fence != handle.fence
            ):
                raise WorktreeLifecycleError(
                    "stale evaluation worktree lease/fence"
                )
            released = EvaluationWorktreeHandle(
                worktree_id=handle.worktree_id,
                role=handle.role,
                task_id=handle.task_id,
                attempt_index=handle.attempt_index,
                lease_id=handle.lease_id,
                fence=handle.fence,
                isolated=True,
                released=True,
            )
            del self._active[handle.worktree_id]
            self._release_count += 1
            return released

    def active_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._active))

    def created_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._created_ids)


# ---------------------------------------------------------------------------
# Resource admission seam (wraps ResourceScheduler-like acquire/release)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShadowResourceLease:
    """Opaque resource lease for one shadow attempt."""

    lease_id: str
    role: str
    admitted: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _token(self.lease_id, "lease_id"))
        object.__setattr__(
            self, "role", _enum(self.role, ShadowAttemptRole, "role")
        )
        object.__setattr__(self, "admitted", _bool(self.admitted, "admitted"))


class ShadowResourceGate(Protocol):
    def admit(
        self, *, role: str, plan: ShadowExecutionPlan
    ) -> ShadowResourceLease: ...

    def release(self, lease: ShadowResourceLease) -> None: ...


class AlwaysAdmitResourceGate:
    """Default gate: always admits (budget ledger remains authoritative)."""

    def __init__(self) -> None:
        self._counter = 0
        self._lock = threading.Lock()
        self.admissions: list[str] = []
        self.releases: list[str] = []

    def admit(
        self, *, role: str, plan: ShadowExecutionPlan
    ) -> ShadowResourceLease:
        if not isinstance(plan, ShadowExecutionPlan):
            raise PlanAdmissionError("plan must be ShadowExecutionPlan")
        role_value = _enum(role, ShadowAttemptRole, "role")
        with self._lock:
            self._counter += 1
            lease_id = f"shadow.resource.{role_value}.{self._counter}"
            self.admissions.append(lease_id)
        return ShadowResourceLease(
            lease_id=lease_id, role=role_value, admitted=True
        )

    def release(self, lease: ShadowResourceLease) -> None:
        if not isinstance(lease, ShadowResourceLease):
            raise SemanticGovernorShadowError(
                "lease must be ShadowResourceLease"
            )
        with self._lock:
            self.releases.append(lease.lease_id)


class ResourceSchedulerGate:
    """Optional adapter over a ResourceScheduler-like object.

    Expects ``acquire(requirement) -> (decision, lease|None)`` and
    ``release(lease)``. Admission failure fails closed.
    """

    def __init__(self, scheduler: Any) -> None:
        if scheduler is None:
            raise SemanticGovernorShadowError("scheduler is required")
        self._scheduler = scheduler

    def admit(
        self, *, role: str, plan: ShadowExecutionPlan
    ) -> ShadowResourceLease:
        role_value = _enum(role, ShadowAttemptRole, "role")
        if not isinstance(plan, ShadowExecutionPlan):
            raise PlanAdmissionError("plan must be ShadowExecutionPlan")
        requirement = {
            "lane_id": f"shadow-{role_value}",
            "task_id": plan.task_id,
            "token_budget": int(plan.max_expansion_token_budget)
            if role_value == ShadowAttemptRole.EXPANDED.value
            else 0,
            "quota_units": 1,
            "resource_pool": "model",
            "metadata": {
                "plan_cid": plan.plan_cid,
                "role": role_value,
                "evidence": SCG_SHADOW_RUN_EVIDENCE,
            },
        }
        acquire = getattr(self._scheduler, "acquire", None)
        if not callable(acquire):
            raise SemanticGovernorShadowError(
                "scheduler must provide acquire()"
            )
        result = acquire(requirement)
        lease_obj = None
        admitted = False
        if isinstance(result, tuple) and len(result) >= 2:
            decision, lease_obj = result[0], result[1]
            admitted = bool(
                getattr(decision, "admitted", None)
                if not isinstance(decision, Mapping)
                else decision.get("admitted")
            )
            if not admitted and hasattr(decision, "status"):
                admitted = str(getattr(decision, "status", "")).lower() in {
                    "admitted",
                    "accepted",
                    "ok",
                }
        elif result is not None:
            lease_obj = result
            admitted = True
        if not admitted or lease_obj is None:
            raise BudgetExceededError(
                f"resource admission denied for role {role_value}"
            )
        lease_id = str(
            getattr(lease_obj, "lease_id", None)
            or getattr(lease_obj, "id", None)
            or f"scheduler.{role_value}"
        )
        # Normalize to token form.
        safe = "".join(
            ch if ch.isalnum() or ch in "._:+-" else "-" for ch in lease_id.lower()
        )
        if not safe or not safe[0].isalpha():
            safe = f"lease.{safe}" if safe else f"lease.{role_value}"
        return ShadowResourceLease(
            lease_id=safe[:128], role=role_value, admitted=True
        )

    def release(self, lease: ShadowResourceLease) -> None:
        release = getattr(self._scheduler, "release", None)
        if callable(release):
            try:
                release(lease.lease_id)
            except TypeError:
                release(lease)


# ---------------------------------------------------------------------------
# Attempt invocation / proposal (runner boundary)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShadowAttemptInvocation:
    """Pre-checked, privacy-filtered invocation request for one attempt.

    Constructed only after budget and disclosure rechecks succeed.
    """

    role: str
    plan_cid: str
    task_id: str
    context_pack_cid: str
    route_id: str
    provider_id: str
    provider_locality: str
    worktree_id: str
    worktree_lease_id: str
    worktree_fence: int
    execution_mode: str
    disclosure_disposition: str
    authorization_decision_cid: str | None
    redacted_context: Mapping[str, Any] | Any
    estimated_input_tokens: int
    estimated_model_spend_micros: int
    remaining_wall_time_ms: int
    remaining_model_spend_micros: int
    remaining_expansion_tokens: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "role", _enum(self.role, ShadowAttemptRole, "role")
        )
        object.__setattr__(self, "plan_cid", _cid(self.plan_cid, "plan_cid"))
        object.__setattr__(self, "task_id", _task_id(self.task_id))
        object.__setattr__(
            self,
            "context_pack_cid",
            _cid(self.context_pack_cid, "context_pack_cid"),
        )
        object.__setattr__(self, "route_id", _token(self.route_id, "route_id"))
        object.__setattr__(
            self, "provider_id", _text(self.provider_id, "provider_id")
        )
        object.__setattr__(
            self,
            "provider_locality",
            _enum(self.provider_locality, ProviderLocality, "provider_locality"),
        )
        object.__setattr__(
            self, "worktree_id", _worktree_id(self.worktree_id, "worktree_id")
        )
        object.__setattr__(
            self,
            "worktree_lease_id",
            _token(self.worktree_lease_id, "worktree_lease_id"),
        )
        object.__setattr__(
            self,
            "worktree_fence",
            _nonneg_int(self.worktree_fence, "worktree_fence"),
        )
        object.__setattr__(
            self,
            "execution_mode",
            _enum(self.execution_mode, ExecutionMode, "execution_mode"),
        )
        object.__setattr__(
            self,
            "disclosure_disposition",
            _enum(
                self.disclosure_disposition,
                DisclosureDisposition,
                "disclosure_disposition",
            ),
        )
        object.__setattr__(
            self,
            "authorization_decision_cid",
            _optional_cid(
                self.authorization_decision_cid, "authorization_decision_cid"
            ),
        )
        object.__setattr__(
            self,
            "estimated_input_tokens",
            _nonneg_int(self.estimated_input_tokens, "estimated_input_tokens"),
        )
        object.__setattr__(
            self,
            "estimated_model_spend_micros",
            _nonneg_int(
                self.estimated_model_spend_micros, "estimated_model_spend_micros"
            ),
        )
        object.__setattr__(
            self,
            "remaining_wall_time_ms",
            _nonneg_int(self.remaining_wall_time_ms, "remaining_wall_time_ms"),
        )
        object.__setattr__(
            self,
            "remaining_model_spend_micros",
            _nonneg_int(
                self.remaining_model_spend_micros, "remaining_model_spend_micros"
            ),
        )
        object.__setattr__(
            self,
            "remaining_expansion_tokens",
            _nonneg_int(
                self.remaining_expansion_tokens, "remaining_expansion_tokens"
            ),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        if self.disclosure_disposition == DisclosureDisposition.FORBIDDEN.value:
            raise DisclosureRecheckError(
                "cannot build invocation with forbidden disclosure disposition"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SHADOW_ATTEMPT_INVOCATION_SCHEMA,
            "role": self.role,
            "plan_cid": self.plan_cid,
            "task_id": self.task_id,
            "context_pack_cid": self.context_pack_cid,
            "route_id": self.route_id,
            "provider_id": self.provider_id,
            "provider_locality": self.provider_locality,
            "worktree_id": self.worktree_id,
            "worktree_lease_id": self.worktree_lease_id,
            "worktree_fence": self.worktree_fence,
            "execution_mode": self.execution_mode,
            "disclosure_disposition": self.disclosure_disposition,
            "authorization_decision_cid": self.authorization_decision_cid,
            "redacted_context": _thaw_structured(self.redacted_context)
            if isinstance(self.redacted_context, (Mapping, list, tuple))
            else self.redacted_context,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_model_spend_micros": self.estimated_model_spend_micros,
            "remaining_wall_time_ms": self.remaining_wall_time_ms,
            "remaining_model_spend_micros": self.remaining_model_spend_micros,
            "remaining_expansion_tokens": self.remaining_expansion_tokens,
            "metadata": _thaw_structured(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ShadowAttemptProposal:
    """Raw attempt outcome from a runner — acceptance is NOT trusted for expanded."""

    attempt_status: str
    cost_timing: CostTimingProjection
    verification: VerificationProjection
    patch_cid: str | None = None
    failure_reason_codes: Sequence[str] = ()
    notes: str | None = None
    # Optional claimed disposition — executor overrides for expanded.
    claimed_acceptance_disposition: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attempt_status",
            _enum(self.attempt_status, AttemptTerminalStatus, "attempt_status"),
        )
        if not isinstance(self.cost_timing, CostTimingProjection):
            raise SemanticGovernorShadowError(
                "cost_timing must be CostTimingProjection"
            )
        if not isinstance(self.verification, VerificationProjection):
            raise SemanticGovernorShadowError(
                "verification must be VerificationProjection"
            )
        object.__setattr__(
            self, "patch_cid", _optional_cid(self.patch_cid, "patch_cid")
        )
        object.__setattr__(
            self,
            "failure_reason_codes",
            _unique_sorted_tokens(
                list(self.failure_reason_codes),
                "failure_reason_codes",
                max_items=MAX_REASON_CODES,
            ),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        if self.claimed_acceptance_disposition is not None:
            object.__setattr__(
                self,
                "claimed_acceptance_disposition",
                _enum(
                    self.claimed_acceptance_disposition,
                    AcceptanceDisposition,
                    "claimed_acceptance_disposition",
                ),
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        if (
            self.attempt_status == AttemptTerminalStatus.FAILED.value
            and not self.failure_reason_codes
        ):
            raise SemanticGovernorShadowError(
                "failed proposal requires failure_reason_codes"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SHADOW_ATTEMPT_PROPOSAL_SCHEMA,
            "attempt_status": self.attempt_status,
            "cost_timing": self.cost_timing.to_dict(),
            "verification": self.verification.to_dict(),
            "patch_cid": self.patch_cid,
            "failure_reason_codes": list(self.failure_reason_codes),
            "notes": self.notes,
            "claimed_acceptance_disposition": self.claimed_acceptance_disposition,
            "metadata": _thaw_structured(self.metadata),
        }


class ShadowAttemptRunner(Protocol):
    """Provider / harness attempt runner bound to one evaluation worktree."""

    def run_attempt(
        self, invocation: ShadowAttemptInvocation
    ) -> ShadowAttemptProposal: ...


class SimulatedShadowAttemptRunner:
    """Deterministic simulated runner for hermetic tests and dry runs.

    Never touches production. Outcomes are content-addressed from the
    invocation and never production-eligible.
    """

    def __init__(
        self,
        *,
        compressed_status: str = AttemptTerminalStatus.SUCCEEDED.value,
        expanded_status: str = AttemptTerminalStatus.SUCCEEDED.value,
        claim_expanded_accepted: bool = False,
        compressed_cost: CostTimingProjection | None = None,
        expanded_cost: CostTimingProjection | None = None,
        on_invoke: Callable[[ShadowAttemptInvocation], None] | None = None,
        raise_on_role: Mapping[str, BaseException] | None = None,
    ) -> None:
        self.compressed_status = compressed_status
        self.expanded_status = expanded_status
        self.claim_expanded_accepted = claim_expanded_accepted
        self.compressed_cost = compressed_cost or CostTimingProjection(
            input_tokens=100,
            output_tokens=20,
            wall_time_ms=50,
            model_spend_micros=1000,
            verification_time_ms=10,
        )
        self.expanded_cost = expanded_cost or CostTimingProjection(
            input_tokens=400,
            output_tokens=40,
            wall_time_ms=80,
            model_spend_micros=4000,
            verification_time_ms=20,
        )
        self.on_invoke = on_invoke
        self.raise_on_role = dict(raise_on_role or {})
        self.invocations: list[ShadowAttemptInvocation] = []

    def run_attempt(
        self, invocation: ShadowAttemptInvocation
    ) -> ShadowAttemptProposal:
        if not isinstance(invocation, ShadowAttemptInvocation):
            raise SemanticGovernorShadowError(
                "invocation must be ShadowAttemptInvocation"
            )
        self.invocations.append(invocation)
        if self.on_invoke is not None:
            self.on_invoke(invocation)
        if invocation.role in self.raise_on_role:
            raise self.raise_on_role[invocation.role]

        status = (
            self.compressed_status
            if invocation.role == ShadowAttemptRole.COMPRESSED.value
            else self.expanded_status
        )
        cost = (
            self.compressed_cost
            if invocation.role == ShadowAttemptRole.COMPRESSED.value
            else self.expanded_cost
        )
        patch_cid = cid_for_structured(
            {
                "kind": "shadow_attempt_patch",
                "role": invocation.role,
                "plan_cid": invocation.plan_cid,
                "context_pack_cid": invocation.context_pack_cid,
                "worktree_id": invocation.worktree_id,
            }
        )
        verification_bundle_cid = cid_for_structured(
            {
                "kind": "shadow_attempt_verification",
                "role": invocation.role,
                "plan_cid": invocation.plan_cid,
                "patch_cid": patch_cid,
            }
        )
        succeeded = status == AttemptTerminalStatus.SUCCEEDED.value
        verification = VerificationProjection(
            verification_bundle_cid=verification_bundle_cid,
            selected_tests_passed=True if succeeded else False,
            full_suite_passed=True if succeeded else False,
            proofs_passed=True if succeeded else None,
            static_checks_passed=True if succeeded else False,
            counterexample_present=not succeeded,
            acceptance_matrix_satisfied=False,
            production_eligible=False,
        )
        claimed: str | None = None
        if (
            invocation.role == ShadowAttemptRole.EXPANDED.value
            and self.claim_expanded_accepted
        ):
            # Malicious / buggy runner claim — executor must reject/override.
            claimed = AcceptanceDisposition.ACCEPTED.value
        elif succeeded:
            claimed = (
                AcceptanceDisposition.CANDIDATE_ONLY.value
                if invocation.role == ShadowAttemptRole.EXPANDED.value
                else AcceptanceDisposition.NOT_ACCEPTED.value
            )
        else:
            claimed = AcceptanceDisposition.NOT_ACCEPTED.value

        reasons: tuple[str, ...] = ()
        if not succeeded:
            reasons = ("simulated_attempt_failed",)

        return ShadowAttemptProposal(
            attempt_status=status,
            cost_timing=cost,
            verification=verification,
            patch_cid=patch_cid if succeeded else None,
            failure_reason_codes=reasons,
            notes="simulated_shadow_attempt",
            claimed_acceptance_disposition=claimed,
            metadata={"runner": "simulated"},
        )


class CallableShadowAttemptRunner:
    """Adapter that forwards to a caller-supplied callable."""

    def __init__(
        self, runner: Callable[[ShadowAttemptInvocation], ShadowAttemptProposal]
    ) -> None:
        if not callable(runner):
            raise SemanticGovernorShadowError("runner must be callable")
        self._runner = runner
        self.invocations: list[ShadowAttemptInvocation] = []

    def run_attempt(
        self, invocation: ShadowAttemptInvocation
    ) -> ShadowAttemptProposal:
        self.invocations.append(invocation)
        proposal = self._runner(invocation)
        if not isinstance(proposal, ShadowAttemptProposal):
            raise SemanticGovernorShadowError(
                "runner must return ShadowAttemptProposal"
            )
        return proposal


# ---------------------------------------------------------------------------
# Acceptance disposition enforcement
# ---------------------------------------------------------------------------


def resolve_acceptance_disposition(
    *,
    role: str,
    attempt_status: str,
    execution_mode: str,
    verification: VerificationProjection,
    claimed: str | None,
) -> str:
    """Resolve acceptance disposition with expanded never-accepted invariant.

    Expanded success → candidate_only (never accepted).
    Expanded non-success → not_accepted.
    Compressed may only be accepted when live + production_eligible + succeeded
    and the runner claimed accepted; shadow evaluation default is not_accepted.
    """

    role_value = _enum(role, ShadowAttemptRole, "role")
    status = _enum(attempt_status, AttemptTerminalStatus, "attempt_status")
    mode = _enum(execution_mode, ExecutionMode, "execution_mode")
    if not isinstance(verification, VerificationProjection):
        raise SemanticGovernorShadowError(
            "verification must be VerificationProjection"
        )

    if role_value == ShadowAttemptRole.EXPANDED.value:
        if status == AttemptTerminalStatus.SUCCEEDED.value:
            disposition = AcceptanceDisposition.CANDIDATE_ONLY.value
        elif status == AttemptTerminalStatus.CANCELLED.value:
            disposition = AcceptanceDisposition.NOT_ACCEPTED.value
        else:
            disposition = AcceptanceDisposition.NOT_ACCEPTED.value
        # Force non-production on expanded even if runner claimed otherwise.
        assert_expanded_never_accepted(disposition, role=role_value)
        if verification.production_eligible:
            raise SemanticGovernorShadowError(
                "expanded verification cannot be production_eligible"
            )
        return disposition

    # Compressed path.
    if (
        status == AttemptTerminalStatus.SUCCEEDED.value
        and mode == ExecutionMode.LIVE.value
        and verification.production_eligible
        and claimed == AcceptanceDisposition.ACCEPTED.value
    ):
        return AcceptanceDisposition.ACCEPTED.value
    if status == AttemptTerminalStatus.SUCCEEDED.value:
        if claimed == AcceptanceDisposition.HUMAN_REVIEW_REQUIRED.value:
            return AcceptanceDisposition.HUMAN_REVIEW_REQUIRED.value
        return AcceptanceDisposition.NOT_ACCEPTED.value
    if claimed == AcceptanceDisposition.HUMAN_REVIEW_REQUIRED.value:
        return AcceptanceDisposition.HUMAN_REVIEW_REQUIRED.value
    return AcceptanceDisposition.NOT_ACCEPTED.value


def force_expanded_verification(
    verification: VerificationProjection,
) -> VerificationProjection:
    """Return a verification projection that cannot be production-eligible."""

    if not isinstance(verification, VerificationProjection):
        raise SemanticGovernorShadowError(
            "verification must be VerificationProjection"
        )
    if not verification.production_eligible:
        return verification
    return VerificationProjection(
        verification_bundle_cid=verification.verification_bundle_cid,
        selected_tests_passed=verification.selected_tests_passed,
        full_suite_passed=verification.full_suite_passed,
        proofs_passed=verification.proofs_passed,
        static_checks_passed=verification.static_checks_passed,
        counterexample_present=verification.counterexample_present,
        acceptance_matrix_satisfied=False,
        production_eligible=False,
    )


def build_cancelled_verification(*, plan_cid: str, role: str) -> VerificationProjection:
    bundle = cid_for_structured(
        {
            "kind": "shadow_cancelled_verification",
            "plan_cid": plan_cid,
            "role": role,
        }
    )
    return VerificationProjection(
        verification_bundle_cid=bundle,
        selected_tests_passed=None,
        full_suite_passed=None,
        proofs_passed=None,
        static_checks_passed=None,
        counterexample_present=False,
        acceptance_matrix_satisfied=False,
        production_eligible=False,
    )


def empty_cost() -> CostTimingProjection:
    return CostTimingProjection(
        input_tokens=0,
        output_tokens=0,
        wall_time_ms=0,
        model_spend_micros=0,
        verification_time_ms=0,
    )


# ---------------------------------------------------------------------------
# Plan admission
# ---------------------------------------------------------------------------


def admit_shadow_plan(plan: ShadowExecutionPlan | Mapping[str, Any]) -> ShadowExecutionPlan:
    """Validate and admit a sealed shadow plan for paired execution."""

    if isinstance(plan, Mapping):
        plan = ShadowExecutionPlan.from_dict(plan)
    if not isinstance(plan, ShadowExecutionPlan):
        raise PlanAdmissionError("plan must be ShadowExecutionPlan")
    # Re-seal identity.
    verify_plan_identity(plan)
    if not plan.isolated_evaluation_worktree_required:
        raise PlanAdmissionError(
            "isolated_evaluation_worktree_required must be true"
        )
    if not plan.expanded_is_oracle_candidate_only:
        raise PlanAdmissionError(
            "expanded_is_oracle_candidate_only must be true"
        )
    if plan.max_wall_time_ms == 0:
        raise PlanAdmissionError("max_wall_time_ms must be positive for execution")
    return plan


# ---------------------------------------------------------------------------
# Result header
# ---------------------------------------------------------------------------


def _build_result_header(
    *,
    plan: ShadowExecutionPlan,
    execution_mode: str,
    terminal_status: str = GovernorTerminalStatus.COMPLETE.value,
) -> GovernorArtifactHeader:
    mode = _enum(execution_mode, ExecutionMode, "execution_mode")
    generator = GeneratorIdentity(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        interface_id=EXECUTE_SHADOW_PLAN_INTERFACE,
    )
    provenance = ArtifactProvenance(
        producer_id="semantic_governor",
        producer_version="1",
        execution_mode=mode,
        authority_source=AuthoritySource.DETERMINISTIC
        if mode == ExecutionMode.SIMULATED.value
        else AuthoritySource.DETERMINISTIC,
        input_cids=(plan.plan_cid, plan.compressed_context_pack_cid),
        tool_ids=("shadow.v1", "shadow_executor.v1"),
        policy_cid=plan.audit_policy_cid,
        notes=None,
    )
    return GovernorArtifactHeader(
        artifact_kind="shadow_execution_result",
        repository_state_cid=plan.header.repository_state_cid,
        context_pack_cid=plan.compressed_context_pack_cid,
        verification_bundle_cid=plan.header.verification_bundle_cid,
        generator=generator,
        provenance=provenance,
        terminal_status=terminal_status,
        assumptions=(
            GovernorAssumption(
                assumption_id="isolated_worktree",
                kind=AssumptionKind.ENVIRONMENT,
                statement=(
                    "Paired shadow attempts run in disposable evaluation worktrees"
                ),
                supporting_cids=(plan.plan_cid,),
            ),
            GovernorAssumption(
                assumption_id="expanded_oracle_only",
                kind=AssumptionKind.VERIFICATION,
                statement=(
                    "Expanded shadow output is oracle/candidate only and never "
                    "auto-accepts"
                ),
                supporting_cids=(plan.plan_cid,),
            ),
            GovernorAssumption(
                assumption_id="production_unchanged",
                kind=AssumptionKind.ENVIRONMENT,
                statement=(
                    "Cancellation and timeouts leave production checkout unchanged"
                ),
                supporting_cids=(plan.plan_cid,),
            ),
        ),
        metadata={
            "task_id": plan.task_id,
            "evidence": SCG_SHADOW_RUN_EVIDENCE,
            "plan_cid": plan.plan_cid,
        },
    )


# ---------------------------------------------------------------------------
# ShadowExecutor
# ---------------------------------------------------------------------------


@dataclass
class ShadowExecutor:
    """Execute paired compressed/expanded shadow attempts in isolated worktrees.

    Parameters are injectable for hermetic tests. Production wiring may supply
    a ResourceScheduler-backed gate and a real worktree lifecycle adapter; the
    defaults never mutate the production checkout.
    """

    disclosure_policy: ShadowDisclosurePolicy = field(
        default_factory=default_shadow_disclosure_policy
    )
    attempt_runner: ShadowAttemptRunner = field(
        default_factory=SimulatedShadowAttemptRunner
    )
    worktree_lifecycle: EvaluationWorktreeLifecycle = field(
        default_factory=InMemoryEvaluationWorktreeLifecycle
    )
    resource_gate: ShadowResourceGate = field(
        default_factory=AlwaysAdmitResourceGate
    )
    production_guard: ProductionCheckoutGuard | None = None
    cancellation_token: ShadowCancellationToken | None = None
    clock: MonotonicClock = field(default_factory=MonotonicClock)
    compressed_provider_id: str = DEFAULT_COMPRESSED_PROVIDER_ID
    expanded_provider_id: str = DEFAULT_EXPANDED_PROVIDER_ID
    local_expanded_provider_id: str = DEFAULT_EXPANDED_PROVIDER_ID
    execution_mode: str = ExecutionMode.SIMULATED.value
    # When True, external disclosure failure falls back to local expanded.
    fallback_expanded_to_local: bool = True

    # Observability (not part of durable identity).
    phase: str = field(default=ShadowRunPhase.ADMITTED.value, init=False)
    last_budget_snapshot: Mapping[str, int] | None = field(
        default=None, init=False, repr=False
    )
    invocation_count: int = field(default=0, init=False)
    disclosure_recheck_count: int = field(default=0, init=False)
    budget_recheck_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.disclosure_policy, ShadowDisclosurePolicy):
            if isinstance(self.disclosure_policy, Mapping):
                self.disclosure_policy = ShadowDisclosurePolicy.from_dict(
                    self.disclosure_policy
                )
            else:
                raise SemanticGovernorShadowError(
                    "disclosure_policy must be ShadowDisclosurePolicy"
                )
        self.execution_mode = _enum(
            self.execution_mode, ExecutionMode, "execution_mode"
        )
        self.compressed_provider_id = _text(
            self.compressed_provider_id, "compressed_provider_id"
        )
        self.expanded_provider_id = _text(
            self.expanded_provider_id, "expanded_provider_id"
        )
        self.local_expanded_provider_id = _text(
            self.local_expanded_provider_id, "local_expanded_provider_id"
        )
        self.fallback_expanded_to_local = _bool(
            self.fallback_expanded_to_local, "fallback_expanded_to_local"
        )
        if self.cancellation_token is None:
            self.cancellation_token = ShadowCancellationToken()
        if self.production_guard is None:
            # Neutral fingerprint when caller does not supply production refs.
            self.production_guard = ProductionCheckoutGuard(
                fingerprint=cid_for_structured(
                    {
                        "kind": "production_checkout_fingerprint",
                        "unbound": True,
                    }
                )
            )

    # -- public entrypoint ---------------------------------------------------

    def execute(
        self,
        plan: ShadowExecutionPlan | Mapping[str, Any],
        *,
        compressed_context: Any = None,
        expanded_context: Any = None,
        compressed_provider_id: str | None = None,
        expanded_provider_id: str | None = None,
        execution_mode: str | None = None,
        estimated_compressed_input_tokens: int = 0,
        estimated_expanded_input_tokens: int = 0,
        estimated_compressed_spend_micros: int = 0,
        estimated_expanded_spend_micros: int = 0,
        raise_on_cancel: bool = False,
    ) -> ShadowExecutionResult:
        """Execute a sealed :class:`ShadowExecutionPlan` as paired attempts.

        Expanded results remain candidate-only. Cancellation/timeouts return a
        sealed result (or raise when *raise_on_cancel*) without production
        mutations.
        """

        admitted = admit_shadow_plan(plan)
        mode = _enum(
            execution_mode if execution_mode is not None else self.execution_mode,
            ExecutionMode,
            "execution_mode",
        )
        c_provider = (
            compressed_provider_id
            if compressed_provider_id is not None
            else self.compressed_provider_id
        )
        e_provider = (
            expanded_provider_id
            if expanded_provider_id is not None
            else self.expanded_provider_id
        )

        budget = ShadowBudgetLedger.from_plan(admitted)
        start_ms = self.clock.now_ms()
        deadline_ms = start_ms + int(admitted.max_wall_time_ms)

        self.production_guard.assert_unchanged(phase="pre_execute")
        self.phase = ShadowRunPhase.ADMITTED.value

        compressed_attempt: PairedAttemptRecord | None = None
        expanded_attempt: PairedAttemptRecord | None = None
        expanded_skipped_reason: str | None = None
        both_isolated = False
        run_meta: dict[str, Any] = {
            "evidence": SCG_SHADOW_RUN_EVIDENCE,
            "executor_interface": SHADOW_EXECUTOR_INTERFACE,
            "budget_recheck_count": 0,
            "disclosure_recheck_count": 0,
            "invocation_count": 0,
        }

        try:
            self._check_cancel_or_timeout(deadline_ms)

            # ---- compressed ------------------------------------------------
            self.phase = ShadowRunPhase.COMPRESSED_RUNNING.value
            compressed_attempt = self._run_role(
                plan=admitted,
                role=ShadowAttemptRole.COMPRESSED.value,
                context_pack_cid=admitted.compressed_context_pack_cid,
                route_id=admitted.compressed_route_id,
                provider_id=c_provider,
                context=compressed_context,
                budget=budget,
                deadline_ms=deadline_ms,
                execution_mode=mode,
                attempt_index=1,
                estimated_input_tokens=estimated_compressed_input_tokens,
                estimated_model_spend_micros=estimated_compressed_spend_micros,
                allow_external=True,  # compressed path uses its own provider
                force_local_on_forbidden=False,
            )
            self.phase = ShadowRunPhase.COMPRESSED_DONE.value
            self.production_guard.assert_unchanged(phase="post_compressed")

            self._check_cancel_or_timeout(deadline_ms)

            # ---- expanded --------------------------------------------------
            skip_reason = self._should_skip_expanded(admitted, e_provider)
            if skip_reason is not None:
                expanded_skipped_reason = skip_reason
                expanded_attempt = None
                both_isolated = False
                self.phase = ShadowRunPhase.EXPANDED_SKIPPED.value
            else:
                self.phase = ShadowRunPhase.EXPANDED_RUNNING.value
                try:
                    expanded_attempt = self._run_role(
                        plan=admitted,
                        role=ShadowAttemptRole.EXPANDED.value,
                        context_pack_cid=admitted.expanded_context_pack_cid,
                        route_id=admitted.expanded_route_id,
                        provider_id=e_provider,
                        context=expanded_context,
                        budget=budget,
                        deadline_ms=deadline_ms,
                        execution_mode=mode,
                        attempt_index=2,
                        estimated_input_tokens=estimated_expanded_input_tokens,
                        estimated_model_spend_micros=estimated_expanded_spend_micros,
                        allow_external=bool(
                            admitted.allow_external_expanded_disclosure
                        ),
                        force_local_on_forbidden=self.fallback_expanded_to_local,
                    )
                    both_isolated = True
                    self.phase = ShadowRunPhase.EXPANDED_DONE.value
                except DisclosureRecheckError:
                    if (
                        ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
                        in admitted.selection_reasons
                        or not admitted.allow_external_expanded_disclosure
                    ):
                        expanded_skipped_reason = (
                            ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
                        )
                        expanded_attempt = None
                        both_isolated = False
                        self.phase = ShadowRunPhase.EXPANDED_SKIPPED.value
                    else:
                        raise

            self.production_guard.assert_unchanged(phase="post_expanded")
            self.phase = ShadowRunPhase.COMPLETE.value

        except ShadowCancellationError as exc:
            self.phase = ShadowRunPhase.CANCELLED.value
            self.production_guard.assert_unchanged(phase="cancelled")
            if raise_on_cancel:
                raise
            compressed_attempt = compressed_attempt or self._cancelled_attempt(
                plan=admitted,
                role=ShadowAttemptRole.COMPRESSED.value,
                context_pack_cid=admitted.compressed_context_pack_cid,
                route_id=admitted.compressed_route_id,
                execution_mode=mode,
                reason=str(exc),
            )
            if expanded_attempt is None and expanded_skipped_reason is None:
                # Compressed finished, expanded not started — record expanded cancel.
                if compressed_attempt.attempt_status != AttemptTerminalStatus.CANCELLED.value:
                    expanded_attempt = self._cancelled_attempt(
                        plan=admitted,
                        role=ShadowAttemptRole.EXPANDED.value,
                        context_pack_cid=admitted.expanded_context_pack_cid,
                        route_id=admitted.expanded_route_id,
                        execution_mode=mode,
                        reason=str(exc),
                    )
                    both_isolated = True
                else:
                    # Neither ran fully; still require an expanded skip or attempt.
                    expanded_skipped_reason = (
                        ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
                        if not admitted.allow_external_expanded_disclosure
                        else None
                    )
                    if expanded_skipped_reason is None:
                        expanded_attempt = self._cancelled_attempt(
                            plan=admitted,
                            role=ShadowAttemptRole.EXPANDED.value,
                            context_pack_cid=admitted.expanded_context_pack_cid,
                            route_id=admitted.expanded_route_id,
                            execution_mode=mode,
                            reason=str(exc),
                        )
                        both_isolated = True
            run_meta["cancel_reason"] = str(exc)

        except ShadowTimeoutError as exc:
            self.phase = ShadowRunPhase.TIMED_OUT.value
            self.production_guard.assert_unchanged(phase="timeout")
            if raise_on_cancel:
                raise
            compressed_attempt = compressed_attempt or self._timeout_attempt(
                plan=admitted,
                role=ShadowAttemptRole.COMPRESSED.value,
                context_pack_cid=admitted.compressed_context_pack_cid,
                route_id=admitted.compressed_route_id,
                execution_mode=mode,
                reason=str(exc),
            )
            if expanded_attempt is None and expanded_skipped_reason is None:
                expanded_attempt = self._timeout_attempt(
                    plan=admitted,
                    role=ShadowAttemptRole.EXPANDED.value,
                    context_pack_cid=admitted.expanded_context_pack_cid,
                    route_id=admitted.expanded_route_id,
                    execution_mode=mode,
                    reason=str(exc),
                )
                both_isolated = True
            run_meta["timeout_reason"] = str(exc)

        except BudgetExceededError as exc:
            self.phase = ShadowRunPhase.FAILED.value
            self.production_guard.assert_unchanged(phase="budget_exceeded")
            if compressed_attempt is None:
                compressed_attempt = self._failed_attempt(
                    plan=admitted,
                    role=ShadowAttemptRole.COMPRESSED.value,
                    context_pack_cid=admitted.compressed_context_pack_cid,
                    route_id=admitted.compressed_route_id,
                    execution_mode=mode,
                    reason_codes=("budget_exceeded",),
                    notes=str(exc),
                )
            elif expanded_attempt is None and expanded_skipped_reason is None:
                expanded_attempt = self._failed_attempt(
                    plan=admitted,
                    role=ShadowAttemptRole.EXPANDED.value,
                    context_pack_cid=admitted.expanded_context_pack_cid,
                    route_id=admitted.expanded_route_id,
                    execution_mode=mode,
                    reason_codes=("budget_exceeded",),
                    notes=str(exc),
                )
                both_isolated = True
            run_meta["budget_error"] = str(exc)

        # Final production fence.
        self.production_guard.assert_unchanged(phase="finalize")

        if compressed_attempt is None:
            raise SemanticGovernorShadowError(
                "compressed attempt missing after shadow execution"
            )

        # Contract: expanded present requires both_attempts_isolated.
        if expanded_attempt is not None:
            both_isolated = True
            # Enforce never-accepted at the boundary.
            assert_expanded_never_accepted(
                expanded_attempt.acceptance_disposition,
                role=expanded_attempt.role,
            )
            if expanded_attempt.verification.production_eligible:
                raise SemanticGovernorShadowError(
                    "expanded attempt cannot be production_eligible"
                )
        elif expanded_skipped_reason is None:
            # Must have either expanded attempt or skip reason.
            expanded_skipped_reason = (
                ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
            )

        run_meta["budget_recheck_count"] = self.budget_recheck_count
        run_meta["disclosure_recheck_count"] = self.disclosure_recheck_count
        run_meta["invocation_count"] = self.invocation_count
        run_meta["phase"] = self.phase
        run_meta["budget"] = dict(budget.snapshot())
        self.last_budget_snapshot = budget.snapshot()

        if mode == ExecutionMode.SIMULATED.value:
            terminal = GovernorTerminalStatus.SIMULATED.value
        elif self.phase in {
            ShadowRunPhase.COMPLETE.value,
            ShadowRunPhase.EXPANDED_SKIPPED.value,
            ShadowRunPhase.EXPANDED_DONE.value,
        }:
            terminal = GovernorTerminalStatus.COMPLETE.value
        elif self.phase == ShadowRunPhase.FAILED.value:
            terminal = GovernorTerminalStatus.EVALUATION_FAILED.value
        elif self.phase in {
            ShadowRunPhase.CANCELLED.value,
            ShadowRunPhase.TIMED_OUT.value,
        }:
            terminal = GovernorTerminalStatus.CANCELLED.value
        else:
            terminal = GovernorTerminalStatus.INCONCLUSIVE.value

        header = _build_result_header(
            plan=admitted,
            execution_mode=mode,
            terminal_status=terminal,
        )

        result = ShadowExecutionResult(
            header=header,
            plan_cid=admitted.plan_cid,
            compressed_attempt=compressed_attempt,
            expanded_attempt=expanded_attempt,
            both_attempts_isolated=both_isolated
            if expanded_attempt is not None
            else False,
            expanded_skipped_reason=expanded_skipped_reason,
            metadata=run_meta,
        )
        verify_result_identity(result)
        return result

    # -- internal ------------------------------------------------------------

    def _check_cancel_or_timeout(self, deadline_ms: int) -> None:
        assert self.cancellation_token is not None
        self.cancellation_token.raise_if_cancelled()
        now = self.clock.now_ms()
        if now >= deadline_ms:
            raise ShadowTimeoutError("shadow execution wall-time budget exceeded")

    def _should_skip_expanded(
        self, plan: ShadowExecutionPlan, provider_id: str
    ) -> str | None:
        """Return skip reason when expanded must not be invoked externally."""

        if (
            ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
            in plan.selection_reasons
            and not plan.allow_external_expanded_disclosure
            and not self.fallback_expanded_to_local
        ):
            return ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
        return None

    def _recheck_disclosure(
        self,
        *,
        plan: ShadowExecutionPlan,
        role: str,
        provider_id: str,
        context: Any,
        worktree_id: str,
        allow_external: bool,
        force_local_on_forbidden: bool,
    ) -> tuple[str, str, str, str | None, Any]:
        """Recheck disclosure; return provider, locality, disposition, auth, ctx.

        May rewrite provider to local when external is forbidden and fallback
        is enabled.
        """

        self.disclosure_recheck_count += 1
        policy = self.disclosure_policy
        pid = _text(provider_id, "provider_id")
        locality = classify_provider_locality(pid, policy)

        # Plan-level external gate for expanded.
        if (
            role == ShadowAttemptRole.EXPANDED.value
            and locality
            in {
                ProviderLocality.APPROVED_EXTERNAL.value,
                ProviderLocality.UNAPPROVED_EXTERNAL.value,
            }
            and not allow_external
            and not plan.allow_external_expanded_disclosure
        ):
            if force_local_on_forbidden:
                pid = self.local_expanded_provider_id
                locality = classify_provider_locality(pid, policy)
            else:
                raise DisclosureRecheckError(
                    "plan forbids external expanded disclosure"
                )

        try:
            assert_isolated_evaluation_worktree(
                isolated_evaluation_worktree_required=True,
                worktree_id=worktree_id,
            )
            inv = prepare_provider_invocation(
                context if context is not None else {
                    "context_pack_cid": (
                        plan.expanded_context_pack_cid
                        if role == ShadowAttemptRole.EXPANDED.value
                        else plan.compressed_context_pack_cid
                    ),
                    "role": role,
                },
                policy,
                provider_id=pid,
                isolated_evaluation_worktree=True,
                worktree_id=worktree_id,
            )
            return (
                inv.provider_id,
                inv.provider_locality,
                inv.disposition,
                inv.authorization_decision_cid,
                inv.redacted_context,
            )
        except (DisclosureForbiddenError, WorktreePolicyError) as exc:
            if force_local_on_forbidden and role == ShadowAttemptRole.EXPANDED.value:
                # Retry local-only.
                local_pid = self.local_expanded_provider_id
                inv = prepare_provider_invocation(
                    context if context is not None else {
                        "context_pack_cid": plan.expanded_context_pack_cid,
                        "role": role,
                    },
                    policy,
                    provider_id=local_pid,
                    isolated_evaluation_worktree=True,
                    worktree_id=worktree_id,
                )
                return (
                    inv.provider_id,
                    inv.provider_locality,
                    inv.disposition,
                    inv.authorization_decision_cid,
                    inv.redacted_context,
                )
            raise DisclosureRecheckError(str(exc)) from exc

    def _run_role(
        self,
        *,
        plan: ShadowExecutionPlan,
        role: str,
        context_pack_cid: str,
        route_id: str,
        provider_id: str,
        context: Any,
        budget: ShadowBudgetLedger,
        deadline_ms: int,
        execution_mode: str,
        attempt_index: int,
        estimated_input_tokens: int,
        estimated_model_spend_micros: int,
        allow_external: bool,
        force_local_on_forbidden: bool,
    ) -> PairedAttemptRecord:
        self._check_cancel_or_timeout(deadline_ms)

        # Budget recheck BEFORE any provider work.
        self.budget_recheck_count += 1
        budget.recheck_before_invocation(
            role=role,
            estimated_input_tokens=estimated_input_tokens,
            estimated_model_spend_micros=estimated_model_spend_micros,
        )

        resource_lease: ShadowResourceLease | None = None
        worktree: EvaluationWorktreeHandle | None = None
        try:
            resource_lease = self.resource_gate.admit(role=role, plan=plan)
            worktree = self.worktree_lifecycle.create(
                role=role,
                task_id=plan.task_id,
                attempt_index=attempt_index,
                plan_cid=plan.plan_cid,
            )
            if not worktree.isolated:
                raise WorktreeLifecycleError(
                    "evaluation worktree is not isolated"
                )

            # Disclosure recheck BEFORE invocation.
            pid, locality, disposition, auth_cid, redacted = self._recheck_disclosure(
                plan=plan,
                role=role,
                provider_id=provider_id,
                context=context,
                worktree_id=worktree.worktree_id,
                allow_external=allow_external,
                force_local_on_forbidden=force_local_on_forbidden,
            )
            if disposition == DisclosureDisposition.FORBIDDEN.value:
                raise DisclosureRecheckError(
                    "disclosure recheck produced forbidden disposition"
                )

            # Second budget recheck immediately before runner call.
            self.budget_recheck_count += 1
            budget.recheck_before_invocation(
                role=role,
                estimated_input_tokens=estimated_input_tokens,
                estimated_model_spend_micros=estimated_model_spend_micros,
            )
            self._check_cancel_or_timeout(deadline_ms)

            snap = budget.snapshot()
            invocation = ShadowAttemptInvocation(
                role=role,
                plan_cid=plan.plan_cid,
                task_id=plan.task_id,
                context_pack_cid=context_pack_cid,
                route_id=route_id,
                provider_id=pid,
                provider_locality=locality,
                worktree_id=worktree.worktree_id,
                worktree_lease_id=worktree.lease_id,
                worktree_fence=worktree.fence,
                execution_mode=execution_mode,
                disclosure_disposition=disposition,
                authorization_decision_cid=auth_cid,
                redacted_context=redacted,
                estimated_input_tokens=estimated_input_tokens,
                estimated_model_spend_micros=estimated_model_spend_micros,
                remaining_wall_time_ms=int(snap["remaining_wall_time_ms"]),
                remaining_model_spend_micros=int(
                    snap["remaining_model_spend_micros"]
                ),
                remaining_expansion_tokens=int(
                    snap["remaining_expansion_tokens"]
                ),
                metadata={
                    "resource_lease_id": resource_lease.lease_id,
                    "evidence": SCG_SHADOW_RUN_EVIDENCE,
                },
            )

            self.invocation_count += 1
            try:
                proposal = self.attempt_runner.run_attempt(invocation)
            except ShadowCancellationError:
                raise
            except ShadowTimeoutError:
                raise
            except Exception as exc:
                # Runner failures become evaluation_failed attempts.
                proposal = ShadowAttemptProposal(
                    attempt_status=AttemptTerminalStatus.EVALUATION_FAILED.value,
                    cost_timing=empty_cost(),
                    verification=build_cancelled_verification(
                        plan_cid=plan.plan_cid, role=role
                    ),
                    patch_cid=None,
                    failure_reason_codes=("attempt_runner_error",),
                    notes=str(exc)[:MAX_TEXT_CHARS],
                    claimed_acceptance_disposition=(
                        AcceptanceDisposition.NOT_ACCEPTED.value
                    ),
                )

            if not isinstance(proposal, ShadowAttemptProposal):
                raise SemanticGovernorShadowError(
                    "attempt_runner must return ShadowAttemptProposal"
                )

            verification = proposal.verification
            if role == ShadowAttemptRole.EXPANDED.value:
                verification = force_expanded_verification(verification)

            disposition = resolve_acceptance_disposition(
                role=role,
                attempt_status=proposal.attempt_status,
                execution_mode=execution_mode,
                verification=verification,
                claimed=proposal.claimed_acceptance_disposition,
            )
            if role == ShadowAttemptRole.EXPANDED.value:
                assert_expanded_never_accepted(disposition, role=role)

            budget.record_cost(proposal.cost_timing, role=role)

            return PairedAttemptRecord(
                role=role,
                execution_mode=execution_mode,
                context_pack_cid=context_pack_cid,
                route_id=route_id,
                attempt_status=proposal.attempt_status,
                acceptance_disposition=disposition,
                cost_timing=proposal.cost_timing,
                verification=verification,
                patch_cid=proposal.patch_cid,
                worktree_id=worktree.worktree_id,
                failure_reason_codes=proposal.failure_reason_codes,
                notes=proposal.notes,
            )
        finally:
            if worktree is not None and not worktree.released:
                try:
                    self.worktree_lifecycle.release(worktree)
                except Exception:
                    pass
            if resource_lease is not None:
                try:
                    self.resource_gate.release(resource_lease)
                except Exception:
                    pass
            # Production must remain unchanged after every attempt.
            self.production_guard.assert_unchanged(
                phase=f"after_{role}_attempt"
            )

    def _cancelled_attempt(
        self,
        *,
        plan: ShadowExecutionPlan,
        role: str,
        context_pack_cid: str,
        route_id: str,
        execution_mode: str,
        reason: str,
    ) -> PairedAttemptRecord:
        verification = build_cancelled_verification(
            plan_cid=plan.plan_cid, role=role
        )
        if role == ShadowAttemptRole.EXPANDED.value:
            verification = force_expanded_verification(verification)
        disposition = resolve_acceptance_disposition(
            role=role,
            attempt_status=AttemptTerminalStatus.CANCELLED.value,
            execution_mode=execution_mode,
            verification=verification,
            claimed=AcceptanceDisposition.NOT_ACCEPTED.value,
        )
        return PairedAttemptRecord(
            role=role,
            execution_mode=execution_mode,
            context_pack_cid=context_pack_cid,
            route_id=route_id,
            attempt_status=AttemptTerminalStatus.CANCELLED.value,
            acceptance_disposition=disposition,
            cost_timing=empty_cost(),
            verification=verification,
            patch_cid=None,
            worktree_id=None,
            failure_reason_codes=("cancelled",),
            notes=reason[:MAX_TEXT_CHARS],
        )

    def _timeout_attempt(
        self,
        *,
        plan: ShadowExecutionPlan,
        role: str,
        context_pack_cid: str,
        route_id: str,
        execution_mode: str,
        reason: str,
    ) -> PairedAttemptRecord:
        verification = build_cancelled_verification(
            plan_cid=plan.plan_cid, role=role
        )
        if role == ShadowAttemptRole.EXPANDED.value:
            verification = force_expanded_verification(verification)
        disposition = resolve_acceptance_disposition(
            role=role,
            attempt_status=AttemptTerminalStatus.CANCELLED.value,
            execution_mode=execution_mode,
            verification=verification,
            claimed=AcceptanceDisposition.NOT_ACCEPTED.value,
        )
        return PairedAttemptRecord(
            role=role,
            execution_mode=execution_mode,
            context_pack_cid=context_pack_cid,
            route_id=route_id,
            attempt_status=AttemptTerminalStatus.CANCELLED.value,
            acceptance_disposition=disposition,
            cost_timing=empty_cost(),
            verification=verification,
            patch_cid=None,
            worktree_id=None,
            failure_reason_codes=("timeout",),
            notes=reason[:MAX_TEXT_CHARS],
        )

    def _failed_attempt(
        self,
        *,
        plan: ShadowExecutionPlan,
        role: str,
        context_pack_cid: str,
        route_id: str,
        execution_mode: str,
        reason_codes: Sequence[str],
        notes: str | None = None,
    ) -> PairedAttemptRecord:
        verification = build_cancelled_verification(
            plan_cid=plan.plan_cid, role=role
        )
        if role == ShadowAttemptRole.EXPANDED.value:
            verification = force_expanded_verification(verification)
        disposition = resolve_acceptance_disposition(
            role=role,
            attempt_status=AttemptTerminalStatus.FAILED.value,
            execution_mode=execution_mode,
            verification=verification,
            claimed=AcceptanceDisposition.NOT_ACCEPTED.value,
        )
        return PairedAttemptRecord(
            role=role,
            execution_mode=execution_mode,
            context_pack_cid=context_pack_cid,
            route_id=route_id,
            attempt_status=AttemptTerminalStatus.FAILED.value,
            acceptance_disposition=disposition,
            cost_timing=empty_cost(),
            verification=verification,
            patch_cid=None,
            worktree_id=None,
            failure_reason_codes=tuple(reason_codes),
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Public function API
# ---------------------------------------------------------------------------


def execute_shadow_plan(
    plan: ShadowExecutionPlan | Mapping[str, Any],
    *,
    compressed_context: Any = None,
    expanded_context: Any = None,
    disclosure_policy: ShadowDisclosurePolicy | Mapping[str, Any] | None = None,
    attempt_runner: ShadowAttemptRunner | None = None,
    worktree_lifecycle: EvaluationWorktreeLifecycle | None = None,
    resource_gate: ShadowResourceGate | None = None,
    resource_scheduler: Any | None = None,
    production_guard: ProductionCheckoutGuard | None = None,
    cancellation_token: ShadowCancellationToken | None = None,
    compressed_provider_id: str | None = None,
    expanded_provider_id: str | None = None,
    execution_mode: str | None = None,
    estimated_compressed_input_tokens: int = 0,
    estimated_expanded_input_tokens: int = 0,
    estimated_compressed_spend_micros: int = 0,
    estimated_expanded_spend_micros: int = 0,
    fallback_expanded_to_local: bool = True,
    raise_on_cancel: bool = False,
    executor: ShadowExecutor | None = None,
) -> ShadowExecutionResult:
    """Execute paired compressed and expanded shadow attempts.

    Release-surface equivalent used by the governor runtime::

        execute_shadow_plan(plan, ...)

    Preconditions: *plan* is a valid, admitted :class:`ShadowExecutionPlan`.
    Expanded output is never auto-accepted. Budgets and disclosure are
    rechecked before each provider invocation. Cancellation/timeouts leave
    production state unchanged.
    """

    if executor is not None:
        return executor.execute(
            plan,
            compressed_context=compressed_context,
            expanded_context=expanded_context,
            compressed_provider_id=compressed_provider_id,
            expanded_provider_id=expanded_provider_id,
            execution_mode=execution_mode,
            estimated_compressed_input_tokens=estimated_compressed_input_tokens,
            estimated_expanded_input_tokens=estimated_expanded_input_tokens,
            estimated_compressed_spend_micros=estimated_compressed_spend_micros,
            estimated_expanded_spend_micros=estimated_expanded_spend_micros,
            raise_on_cancel=raise_on_cancel,
        )

    disc = disclosure_policy
    if disc is None:
        disc_policy = default_shadow_disclosure_policy()
    elif isinstance(disc, ShadowDisclosurePolicy):
        disc_policy = disc
    elif isinstance(disc, Mapping):
        disc_policy = ShadowDisclosurePolicy.from_dict(disc)
    else:
        raise SemanticGovernorShadowError(
            "disclosure_policy must be ShadowDisclosurePolicy or mapping"
        )

    gate: ShadowResourceGate
    if resource_gate is not None:
        gate = resource_gate
    elif resource_scheduler is not None:
        gate = ResourceSchedulerGate(resource_scheduler)
    else:
        gate = AlwaysAdmitResourceGate()

    runner: ShadowAttemptRunner = (
        attempt_runner
        if attempt_runner is not None
        else SimulatedShadowAttemptRunner()
    )
    lifecycle: EvaluationWorktreeLifecycle = (
        worktree_lifecycle
        if worktree_lifecycle is not None
        else InMemoryEvaluationWorktreeLifecycle()
    )

    built = ShadowExecutor(
        disclosure_policy=disc_policy,
        attempt_runner=runner,
        worktree_lifecycle=lifecycle,
        resource_gate=gate,
        production_guard=production_guard,
        cancellation_token=cancellation_token,
        compressed_provider_id=(
            compressed_provider_id
            if compressed_provider_id is not None
            else DEFAULT_COMPRESSED_PROVIDER_ID
        ),
        expanded_provider_id=(
            expanded_provider_id
            if expanded_provider_id is not None
            else DEFAULT_EXPANDED_PROVIDER_ID
        ),
        execution_mode=(
            execution_mode
            if execution_mode is not None
            else ExecutionMode.SIMULATED.value
        ),
        fallback_expanded_to_local=fallback_expanded_to_local,
    )
    return built.execute(
        plan,
        compressed_context=compressed_context,
        expanded_context=expanded_context,
        compressed_provider_id=compressed_provider_id,
        expanded_provider_id=expanded_provider_id,
        execution_mode=execution_mode,
        estimated_compressed_input_tokens=estimated_compressed_input_tokens,
        estimated_expanded_input_tokens=estimated_expanded_input_tokens,
        estimated_compressed_spend_micros=estimated_compressed_spend_micros,
        estimated_expanded_spend_micros=estimated_expanded_spend_micros,
        raise_on_cancel=raise_on_cancel,
    )


def expanded_never_auto_accepts(result: ShadowExecutionResult) -> bool:
    """Return True when expanded is absent or non-accepted (invariant check)."""

    if not isinstance(result, ShadowExecutionResult):
        raise SemanticGovernorShadowError(
            "result must be ShadowExecutionResult"
        )
    if result.expanded_attempt is None:
        return True
    assert_expanded_never_accepted(
        result.expanded_attempt.acceptance_disposition,
        role=result.expanded_attempt.role,
    )
    return (
        result.expanded_attempt.acceptance_disposition
        != AcceptanceDisposition.ACCEPTED.value
        and not result.expanded_attempt.verification.production_eligible
    )


def production_state_unchanged(guard: ProductionCheckoutGuard) -> bool:
    """Return True when the production fence still matches the baseline."""

    try:
        guard.assert_unchanged(phase="check")
        return True
    except ProductionStateMutatedError:
        return False


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = (
    "AlwaysAdmitResourceGate",
    "BudgetExceededError",
    "BudgetKind",
    "CallableShadowAttemptRunner",
    "DEFAULT_COMPRESSED_PROVIDER_ID",
    "DEFAULT_EXPANDED_PROVIDER_ID",
    "DEFAULT_EXTERNAL_PROVIDER_ID",
    "DisclosureRecheckError",
    "EXECUTE_SHADOW_PLAN_INTERFACE",
    "EvaluationWorktreeHandle",
    "EvaluationWorktreeLifecycle",
    "InMemoryEvaluationWorktreeLifecycle",
    "MonotonicClock",
    "PlanAdmissionError",
    "ProductionCheckoutGuard",
    "ProductionStateMutatedError",
    "ResourceSchedulerGate",
    "SCG_SHADOW_RUN_EVIDENCE",
    "SHADOW_EXECUTOR_INTERFACE",
    "SemanticGovernorShadowError",
    "ShadowAttemptInvocation",
    "ShadowAttemptProposal",
    "ShadowAttemptRunner",
    "ShadowBudgetLedger",
    "ShadowCancellationError",
    "ShadowCancellationToken",
    "ShadowExecutor",
    "ShadowResourceGate",
    "ShadowResourceLease",
    "ShadowRunPhase",
    "ShadowTimeoutError",
    "SimulatedShadowAttemptRunner",
    "WorktreeLifecycleError",
    "admit_shadow_plan",
    "build_cancelled_verification",
    "empty_cost",
    "execute_shadow_plan",
    "expanded_never_auto_accepts",
    "force_expanded_verification",
    "production_fingerprint_from_refs",
    "production_state_unchanged",
    "resolve_acceptance_disposition",
)
