"""Frozen governor runtime composition and shadow/expansion resilience (SCG-032).

Joins recovery, idempotency, budgets, privacy, audit persistence, differential
comparison, and decision publication into one resumable composition path over
existing harness / verification / store / resource / provider / worktree
authorities.

Interfaces: :class:`GovernorRuntime`, :func:`audit_task`, :func:`shadow_task`,
:func:`expand_audit`.

Fail-closed resilience invariants:

* Interrupted audits recover from durable checkpoints without resetting spend.
* Duplicate inputs preserve content-addressed identities (plans, results,
  decisions, audit records).
* Private expanded source to unapproved external shadow providers is rejected.
* Unbounded expansion plans are rejected before the expansion loop runs.
* Suppressed verification failures are rejected (failures stay visible).
* Simulated attempts cannot claim live quality / production acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
)
import json
import re
import unicodedata

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.audit_contracts import (
    ContextExpansionPlan,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    ExecutionMode,
)

from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    AcceptanceDisposition,
    AttemptTerminalStatus,
    PairedAttemptRecord,
    SemanticGovernorExecutionError,
    ShadowExecutionResult,
    assert_expanded_never_accepted,
    verify_result_identity,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.differential import (
    COMPARE_SHADOW_RESULTS_INTERFACE,
    compare_shadow_results,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.expansion_loop import (
    EXECUTE_EXPANSION_LOOP_INTERFACE,
    ExpansionCheckpointStore,
    ExpansionLoopCheckpoint,
    ExpansionModelPolicy,
    ExpansionStepRunner,
    ExpansionVerificationPolicy,
    InMemoryExpansionCheckpointStore,
    default_model_policy,
    default_verification_policy,
    execute_expansion_loop,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.privacy import (
    DisclosureDisposition,
    DisclosureForbiddenError,
    ProviderLocality,
    ShadowDisclosurePolicy,
    authorize_shadow_disclosure,
    classify_provider_locality,
    contains_private_source,
    default_shadow_disclosure_policy,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.routes import (
    ModelRouteCalibrationState,
    RouteCalibrationUpdateResult,
    RouteRunObservation,
    update_model_route_calibration,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.shadow import (
    EXECUTE_SHADOW_PLAN_INTERFACE,
    EvaluationWorktreeLifecycle,
    InMemoryEvaluationWorktreeLifecycle,
    ProductionCheckoutGuard,
    ShadowAttemptRunner,
    ShadowCancellationToken,
    ShadowResourceGate,
    SimulatedShadowAttemptRunner,
    admit_shadow_plan,
    execute_shadow_plan,
    expanded_never_auto_accepts,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.shadow_plan import (
    CREATE_SHADOW_PLAN_INTERFACE,
    CompressedContextView,
    RepositoryStateSignals,
    ShadowSamplingPolicy,
    ShadowTaskView,
    create_shadow_plan,
    default_shadow_sampling_policy,
    development_shadow_sampling_policy,
)

# ---------------------------------------------------------------------------
# Evidence / interface / schema constants
# ---------------------------------------------------------------------------

SCG_RUNTIME_CONFORMANCE_EVIDENCE: Final[str] = "scg/runtime-conformance@1"

GOVERNOR_RUNTIME_INTERFACE: Final[str] = "GovernorRuntime@1"
AUDIT_TASK_INTERFACE: Final[str] = "audit_task@1"
SHADOW_TASK_INTERFACE: Final[str] = "shadow_task@1"
EXPAND_AUDIT_INTERFACE: Final[str] = "expand_audit@1"

AUDIT_CHECKPOINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/audit-checkpoint@1"
)
AUDIT_TASK_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/audit-task-result@1"
)
SHADOW_TASK_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/shadow-task-result@1"
)
EXPAND_AUDIT_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/expand-audit-result@1"
)

GENERATOR_ID: Final[str] = "semantic_governor_runtime"
GENERATOR_VERSION: Final[str] = "1.0.0"
PRODUCER_ID: Final[str] = "semantic_governor"
PRODUCER_VERSION: Final[str] = "1"
TOOL_ID: Final[str] = "governor_runtime.v1"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_METADATA_KEYS: Final[int] = 64
MAX_REASON_CODES: Final[int] = 256
MAX_IDS: Final[int] = 256

# Hard expansion ceilings enforced by the runtime before the loop is entered.
# These are composition-level bounds; the expansion loop may apply stricter
# plan-local limits.
MAX_RUNTIME_EXPANSION_STEPS: Final[int] = 64
MAX_RUNTIME_TOKEN_GROWTH: Final[int] = 2_000_000
MAX_RUNTIME_RETRIES: Final[int] = 32
MAX_RUNTIME_ESCALATIONS: Final[int] = 8
MAX_RUNTIME_WALL_TIME_MS: Final[int] = 3_600_000
MAX_RUNTIME_SPEND_MICROS: Final[int] = 500_000_000

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_TASK_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z][A-Za-z0-9_.:/+-]{0,127}$"
)

# Metadata / claim keys that attempt to hide verification failures.
_SUPPRESS_FAILURE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "suppress_failure",
        "suppress_failures",
        "suppressed_failure",
        "hide_failure",
        "hide_failures",
        "ignore_verification_failure",
        "ignore_verification_failures",
        "force_accept_despite_failure",
        "mask_failure",
        "failure_suppressed",
    }
)

# Metadata / claim keys that attempt to treat simulated work as live quality.
_LIVE_QUALITY_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "live_quality",
        "live_quality_claim",
        "claims_live_quality",
        "count_as_live_quality",
        "production_quality",
        "promote_as_live",
        "as_live",
        "live_metrics",
    }
)

_UNBOUNDED_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "unbounded",
        "unbounded_expansion",
        "unlimited",
        "no_limit",
        "no_limits",
        "infinite",
        "max_steps_unbounded",
        "token_budget_unbounded",
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SemanticGovernorRuntimeError(SemanticGovernorExecutionError):
    """Closed runtime composition / admission failure."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class AuditRecoveryError(SemanticGovernorRuntimeError):
    """Interrupted audit recovery failed (checkpoint mismatch / corruption)."""


class RuntimeAdmissionError(SemanticGovernorRuntimeError):
    """Input rejected at the runtime admission gate."""


class PrivateExternalShadowError(RuntimeAdmissionError):
    """Private expanded source cannot be sent to an unapproved external provider."""


class UnboundedExpansionError(RuntimeAdmissionError):
    """Expansion plan lacks hard bounds or claims unbounded growth."""


class SuppressedFailureError(RuntimeAdmissionError):
    """Verification or attempt failure was suppressed or made invisible."""


class SimulatedLiveQualityError(RuntimeAdmissionError):
    """Simulated work attempted to claim live quality or production acceptance."""


# ---------------------------------------------------------------------------
# Closed enumerations
# ---------------------------------------------------------------------------


class AuditPhase(str, Enum):
    """Resumable audit pipeline phases."""

    ADMITTED = "admitted"
    PLANNED = "planned"
    SHADOWED = "shadowed"
    COMPARED = "compared"
    EXPANDED = "expanded"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"
    REJECTED = "rejected"


class AuditDisposition(str, Enum):
    """Closed terminal dispositions for an audit task."""

    COMPLETE = "complete"
    INTERRUPTED = "interrupted"
    RECOVERED = "recovered"
    REJECTED = "rejected"
    NOT_SELECTED = "not_selected"
    IDEMPOTENT_HIT = "idempotent_hit"


class ShadowTaskDisposition(str, Enum):
    """Closed dispositions for shadow_task."""

    COMPLETE = "complete"
    REJECTED = "rejected"
    NOT_SELECTED = "not_selected"
    IDEMPOTENT_HIT = "idempotent_hit"


class ExpandAuditDisposition(str, Enum):
    """Closed dispositions for expand_audit."""

    COMPLETE = "complete"
    REJECTED = "rejected"
    IDEMPOTENT_HIT = "idempotent_hit"
    RECOVERED = "recovered"


# ---------------------------------------------------------------------------
# Normalization helpers (match sibling modules)
# ---------------------------------------------------------------------------


def _normalize_token(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SemanticGovernorRuntimeError(f"{name} must be a string")
    text = _normalize_token(value)
    if not text and not empty:
        raise SemanticGovernorRuntimeError(f"{name} must be non-empty")
    if len(text) > MAX_TEXT_CHARS:
        raise SemanticGovernorRuntimeError(f"{name} exceeds maximum length")
    return text


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name, empty=True) or None


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _TOKEN_RE.match(text.casefold().replace("-", "_")):
        # Allow mixed-case tokens used by task/route ids by checking a looser form.
        if not _TASK_ID_RE.match(text):
            raise SemanticGovernorRuntimeError(f"{name} is not a valid token")
    return text


def _task_id(value: Any, name: str = "task_id") -> str:
    text = _text(value, name)
    if not _TASK_ID_RE.match(text):
        raise SemanticGovernorRuntimeError(f"{name} is not a valid task id")
    return text


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        validate_cid(text)
    except Exception as exc:  # pragma: no cover - validate_cid message is enough
        raise SemanticGovernorRuntimeError(f"{name} is not a valid CID: {exc}") from exc
    return text


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SemanticGovernorRuntimeError(f"{name} must be a bool")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticGovernorRuntimeError(f"{name} must be a non-negative int")
    if value < 0:
        raise SemanticGovernorRuntimeError(f"{name} must be non-negative")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    if isinstance(value, str):
        text = _normalize_token(value)
        for item in enum_type:
            if item.value == text or item.name.casefold() == text.casefold():
                return item.value
    raise SemanticGovernorRuntimeError(f"{name} is not a valid {enum_type.__name__}")


def _freeze_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(k): _freeze_structured(v) for k, v in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_structured(v) for v in value)
    return value


def _thaw_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw_structured(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_structured(v) for v in value]
    return value


def _mapping(value: Any, name: str, *, max_keys: int = MAX_METADATA_KEYS) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise SemanticGovernorRuntimeError(f"{name} must be a mapping")
    if len(value) > max_keys:
        raise SemanticGovernorRuntimeError(f"{name} exceeds maximum keys")
    return MappingProxyType({str(k): _freeze_structured(v) for k, v in value.items()})


def _unique_sorted_tokens(
    values: Iterable[Any],
    name: str,
    *,
    max_items: int = MAX_REASON_CODES,
) -> tuple[str, ...]:
    items: list[str] = []
    for raw in values or ():
        items.append(_token(raw, name) if isinstance(raw, str) else _text(raw, name))
    unique = tuple(sorted(set(items)))
    if len(unique) > max_items:
        raise SemanticGovernorRuntimeError(f"{name} exceeds maximum length")
    return unique


def _unique_sorted_reason_codes(values: Iterable[Any], name: str) -> tuple[str, ...]:
    items: list[str] = []
    for raw in values or ():
        text = _text(raw, name)
        if not text:
            continue
        # Reason codes use looser token rules than route ids.
        items.append(text.casefold().replace(" ", "_"))
    unique = tuple(sorted(set(items)))
    if len(unique) > MAX_REASON_CODES:
        raise SemanticGovernorRuntimeError(f"{name} exceeds maximum length")
    return unique


def _stable_cid(label: str) -> str:
    return cid_for_structured({"kind": "scg-runtime-stable", "label": label})


def _truthy_claim(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, (int, float)) and value != 0:
        return True
    if isinstance(value, str) and value.strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
        "live",
        "unbounded",
        "infinite",
        "unlimited",
    }:
        return True
    return False


def _scan_metadata_claims(
    metadata: Mapping[str, Any] | None,
    keys: frozenset[str],
) -> list[str]:
    if not metadata:
        return []
    hits: list[str] = []
    for key, value in metadata.items():
        normalized = str(key).casefold().replace("-", "_")
        if normalized in keys and _truthy_claim(value):
            hits.append(normalized)
        if isinstance(value, Mapping):
            for nested in _scan_metadata_claims(value, keys):
                hits.append(f"{normalized}.{nested}")
    return hits


# ---------------------------------------------------------------------------
# Fail-closed admission guards
# ---------------------------------------------------------------------------


def reject_private_external_shadow(
    *,
    disclosure_policy: ShadowDisclosurePolicy | Mapping[str, Any] | None = None,
    provider_id: str | None,
    context: Any = None,
    includes_private_source: bool | None = None,
    allow_external_expanded_disclosure: bool = False,
    isolated_evaluation_worktree: bool = True,
    worktree_id: str | None = "worktree-eval-runtime",
    raise_on_forbidden: bool = True,
) -> Mapping[str, Any]:
    """Reject private expanded source on unapproved external shadow providers.

    Returns a closed decision mapping when allowed or when
    ``raise_on_forbidden=False``. Raises :class:`PrivateExternalShadowError`
    for forbidden private external disclosure when ``raise_on_forbidden=True``.
    """

    policy = (
        default_shadow_disclosure_policy()
        if disclosure_policy is None
        else (
            disclosure_policy
            if isinstance(disclosure_policy, ShadowDisclosurePolicy)
            else ShadowDisclosurePolicy.from_dict(disclosure_policy)
        )
    )
    private = (
        bool(includes_private_source)
        if includes_private_source is not None
        else contains_private_source(context)
    )
    if provider_id is None:
        return MappingProxyType(
            {
                "allowed": True,
                "disposition": DisclosureDisposition.LOCAL_ONLY.value,
                "reason_codes": ("no_provider_bound", "local_only_default"),
                "includes_private_source": private,
                "provider_id": None,
                "provider_locality": ProviderLocality.LOCAL.value,
            }
        )

    locality = classify_provider_locality(provider_id, policy)
    try:
        auth = authorize_shadow_disclosure(
            policy,
            provider_id=provider_id,
            context=context,
            includes_private_source=private,
            isolated_evaluation_worktree=isolated_evaluation_worktree,
            worktree_id=worktree_id,
            raise_on_forbidden=False,
        )
    except DisclosureForbiddenError as exc:
        if raise_on_forbidden:
            raise PrivateExternalShadowError(
                str(exc),
                reason_code="private_external_shadow_forbidden",
                details={"provider_id": provider_id},
            ) from exc
        return MappingProxyType(
            {
                "allowed": False,
                "disposition": DisclosureDisposition.FORBIDDEN.value,
                "reason_codes": ("private_external_shadow_forbidden",),
                "includes_private_source": private,
                "provider_id": provider_id,
                "provider_locality": locality.value,
            }
        )

    disposition = str(auth.disposition)
    reasons = list(auth.reason_codes)

    # Unapproved external + private is always forbidden at the runtime gate.
    if private and locality is ProviderLocality.UNAPPROVED_EXTERNAL:
        reasons.append("private_external_shadow_forbidden")
        if raise_on_forbidden:
            raise PrivateExternalShadowError(
                "private expanded source cannot be disclosed to unapproved "
                f"external provider {provider_id!r}",
                reason_code="private_external_shadow_forbidden",
                details={
                    "provider_id": provider_id,
                    "provider_locality": locality.value,
                    "disposition": disposition,
                },
            )
        return MappingProxyType(
            {
                "allowed": False,
                "disposition": DisclosureDisposition.FORBIDDEN.value,
                "reason_codes": tuple(sorted(set(reasons))),
                "includes_private_source": private,
                "provider_id": provider_id,
                "provider_locality": locality.value,
            }
        )

    # Explicit external flag without privacy authorization is also rejected.
    if (
        private
        and allow_external_expanded_disclosure
        and disposition == DisclosureDisposition.FORBIDDEN.value
    ):
        reasons.append("private_external_shadow_forbidden")
        if raise_on_forbidden:
            raise PrivateExternalShadowError(
                "allow_external_expanded_disclosure is true but privacy "
                "authorization forbids private external disclosure",
                reason_code="private_external_shadow_forbidden",
                details={
                    "provider_id": provider_id,
                    "disposition": disposition,
                },
            )
        return MappingProxyType(
            {
                "allowed": False,
                "disposition": DisclosureDisposition.FORBIDDEN.value,
                "reason_codes": tuple(sorted(set(reasons))),
                "includes_private_source": private,
                "provider_id": provider_id,
                "provider_locality": locality.value,
            }
        )

    allowed = disposition != DisclosureDisposition.FORBIDDEN.value
    if not allowed and raise_on_forbidden and private:
        raise PrivateExternalShadowError(
            "private external shadow disclosure forbidden",
            reason_code="private_external_shadow_forbidden",
            details={"provider_id": provider_id, "disposition": disposition},
        )
    return MappingProxyType(
        {
            "allowed": allowed,
            "disposition": disposition,
            "reason_codes": tuple(sorted(set(reasons))),
            "includes_private_source": private,
            "provider_id": provider_id,
            "provider_locality": locality.value,
        }
    )


def reject_unbounded_expansion(
    plan: ContextExpansionPlan | Mapping[str, Any],
) -> ContextExpansionPlan:
    """Admit only expansion plans with hard, finite bounds.

    Rejects plans that claim unbounded growth via metadata or that exceed
    composition-level ceilings. Zero ``max_steps`` with a non-empty step list
    is also rejected (would silently drop required expansion work).
    """

    if isinstance(plan, Mapping):
        resolved = ContextExpansionPlan.from_dict(plan)
    elif isinstance(plan, ContextExpansionPlan):
        resolved = plan
    else:
        raise UnboundedExpansionError(
            "expansion plan must be ContextExpansionPlan or mapping",
            reason_code="unbounded_expansion_rejected",
        )

    meta_hits = _scan_metadata_claims(
        getattr(resolved, "metadata", None) or {},
        _UNBOUNDED_CLAIM_KEYS,
    )
    if meta_hits:
        raise UnboundedExpansionError(
            "expansion plan metadata claims unbounded growth: "
            + ", ".join(meta_hits),
            reason_code="unbounded_expansion_rejected",
            details={"claim_keys": meta_hits},
        )

    max_steps = int(resolved.max_steps)
    max_token_growth = int(resolved.max_token_growth)
    max_retries = int(resolved.max_retries)
    max_escalations = int(resolved.max_escalations)
    max_wall_time_ms = int(resolved.max_wall_time_ms)
    max_spend_micros = int(resolved.max_spend_micros)
    step_count = len(tuple(resolved.steps))

    if max_steps <= 0 and step_count > 0:
        raise UnboundedExpansionError(
            "expansion plan has steps but max_steps is zero (unbounded/invalid)",
            reason_code="unbounded_expansion_rejected",
            details={"max_steps": max_steps, "step_count": step_count},
        )
    if max_steps > MAX_RUNTIME_EXPANSION_STEPS:
        raise UnboundedExpansionError(
            f"max_steps {max_steps} exceeds runtime ceiling "
            f"{MAX_RUNTIME_EXPANSION_STEPS}",
            reason_code="unbounded_expansion_rejected",
            details={"max_steps": max_steps},
        )
    if max_token_growth > MAX_RUNTIME_TOKEN_GROWTH:
        raise UnboundedExpansionError(
            f"max_token_growth {max_token_growth} exceeds runtime ceiling "
            f"{MAX_RUNTIME_TOKEN_GROWTH}",
            reason_code="unbounded_expansion_rejected",
            details={"max_token_growth": max_token_growth},
        )
    if max_retries > MAX_RUNTIME_RETRIES:
        raise UnboundedExpansionError(
            f"max_retries {max_retries} exceeds runtime ceiling "
            f"{MAX_RUNTIME_RETRIES}",
            reason_code="unbounded_expansion_rejected",
        )
    if max_escalations > MAX_RUNTIME_ESCALATIONS:
        raise UnboundedExpansionError(
            f"max_escalations {max_escalations} exceeds runtime ceiling "
            f"{MAX_RUNTIME_ESCALATIONS}",
            reason_code="unbounded_expansion_rejected",
        )
    if max_wall_time_ms <= 0:
        raise UnboundedExpansionError(
            "max_wall_time_ms must be positive (expansion must be time-bounded)",
            reason_code="unbounded_expansion_rejected",
        )
    if max_wall_time_ms > MAX_RUNTIME_WALL_TIME_MS:
        raise UnboundedExpansionError(
            f"max_wall_time_ms {max_wall_time_ms} exceeds runtime ceiling "
            f"{MAX_RUNTIME_WALL_TIME_MS}",
            reason_code="unbounded_expansion_rejected",
        )
    if max_spend_micros > MAX_RUNTIME_SPEND_MICROS:
        raise UnboundedExpansionError(
            f"max_spend_micros {max_spend_micros} exceeds runtime ceiling "
            f"{MAX_RUNTIME_SPEND_MICROS}",
            reason_code="unbounded_expansion_rejected",
        )
    # Token growth must be finite when any step increases tokens.
    total_increase = int(resolved.total_token_increase)
    if total_increase > 0 and max_token_growth <= 0:
        raise UnboundedExpansionError(
            "plan has positive token increase but max_token_growth is zero",
            reason_code="unbounded_expansion_rejected",
            details={
                "total_token_increase": total_increase,
                "max_token_growth": max_token_growth,
            },
        )
    return resolved


def reject_suppressed_failure(
    *,
    attempt: PairedAttemptRecord | Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    verification_passed: bool | None = None,
    acceptance_disposition: str | None = None,
    attempt_status: str | None = None,
    failure_reason_codes: Sequence[str] | None = None,
    production_eligible: bool | None = None,
    role: str | None = None,
) -> None:
    """Reject claims that hide or suppress verification / attempt failures."""

    meta = dict(metadata or {})
    hits = _scan_metadata_claims(meta, _SUPPRESS_FAILURE_KEYS)
    if hits:
        raise SuppressedFailureError(
            "suppressed failure claim is rejected: " + ", ".join(hits),
            reason_code="suppressed_failure_rejected",
            details={"claim_keys": hits},
        )

    status = attempt_status
    disposition = acceptance_disposition
    reasons = list(failure_reason_codes or ())
    prod = production_eligible
    ver_pass = verification_passed
    resolved_role = role

    if attempt is not None:
        if isinstance(attempt, Mapping):
            attempt = PairedAttemptRecord.from_dict(attempt)
        if not isinstance(attempt, PairedAttemptRecord):
            raise SuppressedFailureError(
                "attempt must be PairedAttemptRecord or mapping",
                reason_code="suppressed_failure_rejected",
            )
        status = attempt.attempt_status
        disposition = attempt.acceptance_disposition
        reasons = list(attempt.failure_reason_codes)
        prod = attempt.verification.production_eligible
        resolved_role = attempt.role
        # Treat partial verification matrix as failure signal when any required
        # check is explicitly False while acceptance is claimed.
        checks = (
            attempt.verification.selected_tests_passed,
            attempt.verification.full_suite_passed,
            attempt.verification.proofs_passed,
            attempt.verification.static_checks_passed,
        )
        if any(c is False for c in checks):
            ver_pass = False
        elif all(c is True for c in checks if c is not None) and checks:
            ver_pass = True if ver_pass is None else ver_pass
        if attempt.verification.counterexample_present:
            ver_pass = False
        if not attempt.verification.acceptance_matrix_satisfied and disposition in {
            AcceptanceDisposition.ACCEPTED.value,
        }:
            ver_pass = False

    if status == AttemptTerminalStatus.FAILED.value and not reasons:
        raise SuppressedFailureError(
            "failed attempt must expose at least one failure_reason_code",
            reason_code="suppressed_failure_rejected",
            details={"attempt_status": status},
        )

    if (
        disposition == AcceptanceDisposition.ACCEPTED.value
        and ver_pass is False
    ):
        raise SuppressedFailureError(
            "acceptance cannot suppress a verification failure",
            reason_code="suppressed_failure_rejected",
            details={
                "acceptance_disposition": disposition,
                "verification_passed": ver_pass,
            },
        )

    if prod is True and ver_pass is False:
        raise SuppressedFailureError(
            "production_eligible cannot be true when verification failed",
            reason_code="suppressed_failure_rejected",
        )

    if (
        status == AttemptTerminalStatus.SUCCEEDED.value
        and ver_pass is False
        and disposition
        in {
            AcceptanceDisposition.ACCEPTED.value,
        }
    ):
        raise SuppressedFailureError(
            "succeeded+accepted claim cannot suppress verification failure",
            reason_code="suppressed_failure_rejected",
        )

    if resolved_role is not None and disposition is not None:
        assert_expanded_never_accepted(disposition, role=resolved_role)


def reject_simulated_live_quality_claim(
    *,
    execution_mode: str | ExecutionMode | None,
    acceptance_disposition: str | None = None,
    production_eligible: bool | None = None,
    simulated: bool | None = None,
    metadata: Mapping[str, Any] | None = None,
    observation: RouteRunObservation | Mapping[str, Any] | None = None,
    quality_claim_live: bool = False,
) -> None:
    """Reject simulated work that claims live quality or production acceptance."""

    mode: str | None
    if execution_mode is None:
        mode = None
    elif isinstance(execution_mode, ExecutionMode):
        mode = execution_mode.value
    else:
        mode = _enum(execution_mode, ExecutionMode, "execution_mode")

    is_simulated = bool(simulated)
    if mode == ExecutionMode.SIMULATED.value:
        is_simulated = True
    resolved_observation: RouteRunObservation | None = None
    if observation is not None:
        if isinstance(observation, Mapping):
            observation = RouteRunObservation.from_value(observation)
        if not isinstance(observation, RouteRunObservation):
            raise SimulatedLiveQualityError(
                "observation must be RouteRunObservation or mapping",
                reason_code="simulated_live_quality_rejected",
            )
        resolved_observation = observation
        if observation.simulated:
            is_simulated = True
        # Fold observation metadata into the claim scan.
        merged_meta = {**(dict(metadata or {})), **dict(observation.metadata or {})}
        metadata = merged_meta

    meta_hits = _scan_metadata_claims(metadata or {}, _LIVE_QUALITY_CLAIM_KEYS)
    claims_live = quality_claim_live or bool(meta_hits)

    if not is_simulated:
        return

    if claims_live or meta_hits:
        raise SimulatedLiveQualityError(
            "simulated execution cannot claim live quality",
            reason_code="simulated_live_quality_rejected",
            details={
                "execution_mode": mode,
                "claim_keys": meta_hits,
            },
        )

    # Simulated *attempts* cannot claim production acceptance. Calibration
    # observations with accepted=True are skipped (not applied as live quality)
    # unless an explicit live-quality claim is present (handled above).
    if resolved_observation is None:
        if acceptance_disposition == AcceptanceDisposition.ACCEPTED.value:
            raise SimulatedLiveQualityError(
                "simulated execution cannot claim acceptance_disposition=accepted",
                reason_code="simulated_live_quality_rejected",
                details={"execution_mode": mode},
            )
        if production_eligible is True:
            raise SimulatedLiveQualityError(
                "simulated execution cannot be production_eligible",
                reason_code="simulated_live_quality_rejected",
                details={"execution_mode": mode},
            )


def reject_simulated_calibration_as_live(
    state: ModelRouteCalibrationState | Mapping[str, Any] | None,
    observations: Sequence[RouteRunObservation | Mapping[str, Any]] | None,
) -> RouteCalibrationUpdateResult:
    """Apply route calibration while rejecting simulated-as-live quality claims.

    Simulated observations are never applied to live counters (existing route
    calibration behavior). Explicit live-quality metadata on a simulated
    observation fails closed before update.
    """

    for raw in observations or ():
        obs = (
            raw
            if isinstance(raw, RouteRunObservation)
            else RouteRunObservation.from_value(raw)
        )
        # Simulated observations may record accepted=True for offline analysis,
        # but explicit live-quality claims fail closed before calibration update.
        reject_simulated_live_quality_claim(
            execution_mode=(
                ExecutionMode.SIMULATED.value
                if obs.simulated
                else ExecutionMode.LIVE.value
            ),
            simulated=obs.simulated,
            metadata=obs.metadata,
            observation=obs,
        )
    return update_model_route_calibration(state, observations)


# ---------------------------------------------------------------------------
# Durable audit checkpoint / result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditCheckpoint:
    """Restart-safe durable state for an in-progress or completed audit."""

    audit_id: str
    task_id: str
    phase: AuditPhase | str
    input_identity_cid: str
    generation: int = 0
    plan_cid: str | None = None
    shadow_result_cid: str | None = None
    differential_cid: str | None = None
    expansion_result_cid: str | None = None
    expansion_checkpoint_cid: str | None = None
    comparative_outcome: str | None = None
    disposition: AuditDisposition | str = AuditDisposition.INTERRUPTED.value
    reason_codes: Sequence[str] = ()
    plan: Mapping[str, Any] | None = None
    shadow_result: Mapping[str, Any] | None = None
    differential: Mapping[str, Any] | None = None
    expansion_result: Mapping[str, Any] | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "audit_id",
            "task_id",
            "phase",
            "input_identity_cid",
            "generation",
            "plan_cid",
            "shadow_result_cid",
            "differential_cid",
            "expansion_result_cid",
            "expansion_checkpoint_cid",
            "comparative_outcome",
            "disposition",
            "reason_codes",
            "plan",
            "shadow_result",
            "differential",
            "expansion_result",
            "notes",
            "metadata",
            "checkpoint_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "audit_id", _token(self.audit_id, "audit_id"))
        object.__setattr__(self, "task_id", _task_id(self.task_id))
        object.__setattr__(self, "phase", _enum(self.phase, AuditPhase, "phase"))
        object.__setattr__(
            self,
            "input_identity_cid",
            _cid(self.input_identity_cid, "input_identity_cid"),
        )
        object.__setattr__(
            self, "generation", _nonneg_int(self.generation, "generation")
        )
        object.__setattr__(self, "plan_cid", _optional_cid(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self,
            "shadow_result_cid",
            _optional_cid(self.shadow_result_cid, "shadow_result_cid"),
        )
        object.__setattr__(
            self,
            "differential_cid",
            _optional_cid(self.differential_cid, "differential_cid"),
        )
        object.__setattr__(
            self,
            "expansion_result_cid",
            _optional_cid(self.expansion_result_cid, "expansion_result_cid"),
        )
        object.__setattr__(
            self,
            "expansion_checkpoint_cid",
            _optional_cid(self.expansion_checkpoint_cid, "expansion_checkpoint_cid"),
        )
        if self.comparative_outcome is not None:
            object.__setattr__(
                self,
                "comparative_outcome",
                _text(self.comparative_outcome, "comparative_outcome"),
            )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, AuditDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_reason_codes(self.reason_codes, "reason_codes"),
        )
        for field_name in (
            "plan",
            "shadow_result",
            "differential",
            "expansion_result",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise AuditRecoveryError(f"{field_name} must be a mapping or None")
            object.__setattr__(
                self, field_name, MappingProxyType(_thaw_structured(dict(value)))
            )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": AUDIT_CHECKPOINT_SCHEMA,
            "audit_id": self.audit_id,
            "task_id": self.task_id,
            "phase": self.phase,
            "input_identity_cid": self.input_identity_cid,
            "generation": self.generation,
            "plan_cid": self.plan_cid,
            "shadow_result_cid": self.shadow_result_cid,
            "differential_cid": self.differential_cid,
            "expansion_result_cid": self.expansion_result_cid,
            "expansion_checkpoint_cid": self.expansion_checkpoint_cid,
            "comparative_outcome": self.comparative_outcome,
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
            "plan": _thaw_structured(self.plan) if self.plan is not None else None,
            "shadow_result": (
                _thaw_structured(self.shadow_result)
                if self.shadow_result is not None
                else None
            ),
            "differential": (
                _thaw_structured(self.differential)
                if self.differential is not None
                else None
            ),
            "expansion_result": (
                _thaw_structured(self.expansion_result)
                if self.expansion_result is not None
                else None
            ),
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def checkpoint_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "checkpoint_cid": self.checkpoint_cid}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditCheckpoint":
        if not isinstance(data, Mapping):
            raise AuditRecoveryError("AuditCheckpoint must be a mapping")
        payload = dict(data)
        payload.pop("checkpoint_cid", None)
        schema = payload.pop("schema", None)
        if schema is not None and schema != AUDIT_CHECKPOINT_SCHEMA:
            raise AuditRecoveryError("unsupported AuditCheckpoint schema")
        unknown = set(payload) - cls._FIELDS
        # Allow only declared fields (schema/checkpoint_cid already popped).
        allowed = {
            "audit_id",
            "task_id",
            "phase",
            "input_identity_cid",
            "generation",
            "plan_cid",
            "shadow_result_cid",
            "differential_cid",
            "expansion_result_cid",
            "expansion_checkpoint_cid",
            "comparative_outcome",
            "disposition",
            "reason_codes",
            "plan",
            "shadow_result",
            "differential",
            "expansion_result",
            "notes",
            "metadata",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise AuditRecoveryError(
                f"unknown AuditCheckpoint fields: {sorted(unknown)}"
            )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class AuditTaskResult:
    """Published result of :func:`audit_task` / :meth:`GovernorRuntime.audit_task`."""

    audit_id: str
    task_id: str
    disposition: AuditDisposition | str
    phase: AuditPhase | str
    input_identity_cid: str
    recovered: bool = False
    idempotent_hit: bool = False
    plan_cid: str | None = None
    shadow_result_cid: str | None = None
    differential_cid: str | None = None
    expansion_result_cid: str | None = None
    comparative_outcome: str | None = None
    decision_action: str | None = None
    reason_codes: Sequence[str] = ()
    plan: Mapping[str, Any] | None = None
    shadow_result: Mapping[str, Any] | None = None
    differential: Mapping[str, Any] | None = None
    expansion_result: Mapping[str, Any] | None = None
    checkpoint_cid: str | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "audit_id", _token(self.audit_id, "audit_id"))
        object.__setattr__(self, "task_id", _task_id(self.task_id))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, AuditDisposition, "disposition"),
        )
        object.__setattr__(self, "phase", _enum(self.phase, AuditPhase, "phase"))
        object.__setattr__(
            self,
            "input_identity_cid",
            _cid(self.input_identity_cid, "input_identity_cid"),
        )
        object.__setattr__(self, "recovered", _bool(self.recovered, "recovered"))
        object.__setattr__(
            self, "idempotent_hit", _bool(self.idempotent_hit, "idempotent_hit")
        )
        object.__setattr__(self, "plan_cid", _optional_cid(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self,
            "shadow_result_cid",
            _optional_cid(self.shadow_result_cid, "shadow_result_cid"),
        )
        object.__setattr__(
            self,
            "differential_cid",
            _optional_cid(self.differential_cid, "differential_cid"),
        )
        object.__setattr__(
            self,
            "expansion_result_cid",
            _optional_cid(self.expansion_result_cid, "expansion_result_cid"),
        )
        if self.comparative_outcome is not None:
            object.__setattr__(
                self,
                "comparative_outcome",
                _text(self.comparative_outcome, "comparative_outcome"),
            )
        object.__setattr__(
            self, "decision_action", _optional_text(self.decision_action, "decision_action")
        )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_reason_codes(self.reason_codes, "reason_codes"),
        )
        for field_name in (
            "plan",
            "shadow_result",
            "differential",
            "expansion_result",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise SemanticGovernorRuntimeError(
                    f"{field_name} must be a mapping or None"
                )
            object.__setattr__(
                self, field_name, MappingProxyType(_thaw_structured(dict(value)))
            )
        object.__setattr__(
            self, "checkpoint_cid", _optional_cid(self.checkpoint_cid, "checkpoint_cid")
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": AUDIT_TASK_RESULT_SCHEMA,
            "interface_id": AUDIT_TASK_INTERFACE,
            "evidence_id": SCG_RUNTIME_CONFORMANCE_EVIDENCE,
            "audit_id": self.audit_id,
            "task_id": self.task_id,
            "disposition": self.disposition,
            "phase": self.phase,
            "input_identity_cid": self.input_identity_cid,
            "recovered": self.recovered,
            "idempotent_hit": self.idempotent_hit,
            "plan_cid": self.plan_cid,
            "shadow_result_cid": self.shadow_result_cid,
            "differential_cid": self.differential_cid,
            "expansion_result_cid": self.expansion_result_cid,
            "comparative_outcome": self.comparative_outcome,
            "decision_action": self.decision_action,
            "reason_codes": list(self.reason_codes),
            "plan": _thaw_structured(self.plan) if self.plan is not None else None,
            "shadow_result": (
                _thaw_structured(self.shadow_result)
                if self.shadow_result is not None
                else None
            ),
            "differential": (
                _thaw_structured(self.differential)
                if self.differential is not None
                else None
            ),
            "expansion_result": (
                _thaw_structured(self.expansion_result)
                if self.expansion_result is not None
                else None
            ),
            "checkpoint_cid": self.checkpoint_cid,
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def result_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "result_cid": self.result_cid}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditTaskResult":
        if not isinstance(data, Mapping):
            raise SemanticGovernorRuntimeError("AuditTaskResult must be a mapping")
        payload = dict(data)
        payload.pop("result_cid", None)
        payload.pop("schema", None)
        payload.pop("interface_id", None)
        payload.pop("evidence_id", None)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ShadowTaskResult:
    """Published result of :func:`shadow_task`."""

    task_id: str
    disposition: ShadowTaskDisposition | str
    input_identity_cid: str
    plan_cid: str | None = None
    shadow_result_cid: str | None = None
    differential_cid: str | None = None
    comparative_outcome: str | None = None
    idempotent_hit: bool = False
    reason_codes: Sequence[str] = ()
    plan_decision: Mapping[str, Any] | None = None
    plan: Mapping[str, Any] | None = None
    shadow_result: Mapping[str, Any] | None = None
    differential: Mapping[str, Any] | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _task_id(self.task_id))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ShadowTaskDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "input_identity_cid",
            _cid(self.input_identity_cid, "input_identity_cid"),
        )
        object.__setattr__(self, "plan_cid", _optional_cid(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self,
            "shadow_result_cid",
            _optional_cid(self.shadow_result_cid, "shadow_result_cid"),
        )
        object.__setattr__(
            self,
            "differential_cid",
            _optional_cid(self.differential_cid, "differential_cid"),
        )
        if self.comparative_outcome is not None:
            object.__setattr__(
                self,
                "comparative_outcome",
                _text(self.comparative_outcome, "comparative_outcome"),
            )
        object.__setattr__(
            self, "idempotent_hit", _bool(self.idempotent_hit, "idempotent_hit")
        )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_reason_codes(self.reason_codes, "reason_codes"),
        )
        for field_name in ("plan_decision", "plan", "shadow_result", "differential"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise SemanticGovernorRuntimeError(
                    f"{field_name} must be a mapping or None"
                )
            object.__setattr__(
                self, field_name, MappingProxyType(_thaw_structured(dict(value)))
            )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SHADOW_TASK_RESULT_SCHEMA,
            "interface_id": SHADOW_TASK_INTERFACE,
            "evidence_id": SCG_RUNTIME_CONFORMANCE_EVIDENCE,
            "task_id": self.task_id,
            "disposition": self.disposition,
            "input_identity_cid": self.input_identity_cid,
            "plan_cid": self.plan_cid,
            "shadow_result_cid": self.shadow_result_cid,
            "differential_cid": self.differential_cid,
            "comparative_outcome": self.comparative_outcome,
            "idempotent_hit": self.idempotent_hit,
            "reason_codes": list(self.reason_codes),
            "plan_decision": (
                _thaw_structured(self.plan_decision)
                if self.plan_decision is not None
                else None
            ),
            "plan": _thaw_structured(self.plan) if self.plan is not None else None,
            "shadow_result": (
                _thaw_structured(self.shadow_result)
                if self.shadow_result is not None
                else None
            ),
            "differential": (
                _thaw_structured(self.differential)
                if self.differential is not None
                else None
            ),
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def result_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "result_cid": self.result_cid}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShadowTaskResult":
        if not isinstance(data, Mapping):
            raise SemanticGovernorRuntimeError("ShadowTaskResult must be a mapping")
        payload = dict(data)
        for key in ("result_cid", "schema", "interface_id", "evidence_id"):
            payload.pop(key, None)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ExpandAuditResult:
    """Published result of :func:`expand_audit`."""

    audit_case_cid: str
    disposition: ExpandAuditDisposition | str
    input_identity_cid: str
    expansion_result_cid: str | None = None
    plan_cid: str | None = None
    recovered: bool = False
    idempotent_hit: bool = False
    decision_action: str | None = None
    comparative_outcome: str | None = None
    reason_codes: Sequence[str] = ()
    expansion_result: Mapping[str, Any] | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "audit_case_cid", _cid(self.audit_case_cid, "audit_case_cid")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ExpandAuditDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "input_identity_cid",
            _cid(self.input_identity_cid, "input_identity_cid"),
        )
        object.__setattr__(
            self,
            "expansion_result_cid",
            _optional_cid(self.expansion_result_cid, "expansion_result_cid"),
        )
        object.__setattr__(self, "plan_cid", _optional_cid(self.plan_cid, "plan_cid"))
        object.__setattr__(self, "recovered", _bool(self.recovered, "recovered"))
        object.__setattr__(
            self, "idempotent_hit", _bool(self.idempotent_hit, "idempotent_hit")
        )
        object.__setattr__(
            self, "decision_action", _optional_text(self.decision_action, "decision_action")
        )
        if self.comparative_outcome is not None:
            object.__setattr__(
                self,
                "comparative_outcome",
                _text(self.comparative_outcome, "comparative_outcome"),
            )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_reason_codes(self.reason_codes, "reason_codes"),
        )
        if self.expansion_result is not None:
            if not isinstance(self.expansion_result, Mapping):
                raise SemanticGovernorRuntimeError(
                    "expansion_result must be a mapping or None"
                )
            object.__setattr__(
                self,
                "expansion_result",
                MappingProxyType(_thaw_structured(dict(self.expansion_result))),
            )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EXPAND_AUDIT_RESULT_SCHEMA,
            "interface_id": EXPAND_AUDIT_INTERFACE,
            "evidence_id": SCG_RUNTIME_CONFORMANCE_EVIDENCE,
            "audit_case_cid": self.audit_case_cid,
            "disposition": self.disposition,
            "input_identity_cid": self.input_identity_cid,
            "expansion_result_cid": self.expansion_result_cid,
            "plan_cid": self.plan_cid,
            "recovered": self.recovered,
            "idempotent_hit": self.idempotent_hit,
            "decision_action": self.decision_action,
            "comparative_outcome": self.comparative_outcome,
            "reason_codes": list(self.reason_codes),
            "expansion_result": (
                _thaw_structured(self.expansion_result)
                if self.expansion_result is not None
                else None
            ),
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def result_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "result_cid": self.result_cid}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpandAuditResult":
        if not isinstance(data, Mapping):
            raise SemanticGovernorRuntimeError("ExpandAuditResult must be a mapping")
        payload = dict(data)
        for key in ("result_cid", "schema", "interface_id", "evidence_id"):
            payload.pop(key, None)
        return cls(**payload)


# ---------------------------------------------------------------------------
# Checkpoint stores
# ---------------------------------------------------------------------------


class AuditCheckpointStore(Protocol):
    def load(self, audit_id: str) -> AuditCheckpoint | None: ...

    def save(self, checkpoint: AuditCheckpoint) -> None: ...

    def load_by_input_identity(
        self, input_identity_cid: str
    ) -> AuditCheckpoint | None: ...


class InMemoryAuditCheckpointStore:
    """Process-local durable audit checkpoint store for tests and single-process runs."""

    def __init__(self) -> None:
        self._by_audit: dict[str, AuditCheckpoint] = {}
        self._by_input: dict[str, str] = {}

    def load(self, audit_id: str) -> AuditCheckpoint | None:
        return self._by_audit.get(audit_id)

    def save(self, checkpoint: AuditCheckpoint) -> None:
        if not isinstance(checkpoint, AuditCheckpoint):
            raise AuditRecoveryError("checkpoint must be AuditCheckpoint")
        self._by_audit[checkpoint.audit_id] = checkpoint
        self._by_input[checkpoint.input_identity_cid] = checkpoint.audit_id

    def load_by_input_identity(
        self, input_identity_cid: str
    ) -> AuditCheckpoint | None:
        audit_id = self._by_input.get(input_identity_cid)
        if audit_id is None:
            return None
        return self._by_audit.get(audit_id)


class FilesystemAuditCheckpointStore:
    """Filesystem-backed audit checkpoint store (atomic write)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "input_index.json"
        self._index: dict[str, str] = {}
        if self._index_path.is_file():
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._index = {str(k): str(v) for k, v in raw.items()}

    def _path(self, audit_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._+-]+", "_", audit_id)
        return self.root / f"{safe}.json"

    def load(self, audit_id: str) -> AuditCheckpoint | None:
        path = self._path(audit_id)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return AuditCheckpoint.from_dict(data)

    def save(self, checkpoint: AuditCheckpoint) -> None:
        if not isinstance(checkpoint, AuditCheckpoint):
            raise AuditRecoveryError("checkpoint must be AuditCheckpoint")
        path = self._path(checkpoint.audit_id)
        tmp = path.with_suffix(".tmp")
        payload = checkpoint.to_dict()
        tmp.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
            encoding="utf-8",
        )
        tmp.replace(path)
        self._index[checkpoint.input_identity_cid] = checkpoint.audit_id
        idx_tmp = self._index_path.with_suffix(".tmp")
        idx_tmp.write_text(
            json.dumps(self._index, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        idx_tmp.replace(self._index_path)

    def load_by_input_identity(
        self, input_identity_cid: str
    ) -> AuditCheckpoint | None:
        audit_id = self._index.get(input_identity_cid)
        if audit_id is None:
            return None
        return self.load(audit_id)


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


def compute_audit_input_identity(
    *,
    task: ShadowTaskView | Mapping[str, Any] | str,
    compressed_context: CompressedContextView | Mapping[str, Any] | str,
    repository_state: RepositoryStateSignals | Mapping[str, Any] | str,
    audit_policy_cid: str | None = None,
    disclosure_policy_cid: str | None = None,
    execution_mode: str = ExecutionMode.SIMULATED.value,
    expanded_provider_id: str | None = None,
    run_expansion: bool = False,
    expansion_plan_cid: str | None = None,
) -> str:
    """Content-addressed identity of audit_task inputs (duplicate-preserving)."""

    task_payload: Any
    if isinstance(task, ShadowTaskView):
        task_payload = task.to_dict() if hasattr(task, "to_dict") else {
            "task_id": task.task_id,
            "task_class": task.task_class,
            "risk_class": task.risk_class,
            "environment": task.environment,
            "route_id": task.route_id,
            "expanded_route_id": task.expanded_route_id,
            "promotion_evaluation": task.promotion_evaluation,
            "new_task_class": task.new_task_class,
            "new_analyzer": task.new_analyzer,
            "new_route": task.new_route,
        }
    elif isinstance(task, Mapping):
        task_payload = {
            k: task[k]
            for k in (
                "task_id",
                "id",
                "task_class",
                "risk_class",
                "environment",
                "route_id",
                "expanded_route_id",
                "promotion_evaluation",
                "new_task_class",
                "new_analyzer",
                "new_route",
            )
            if k in task
        }
    else:
        task_payload = {"task_id": str(task)}

    if isinstance(compressed_context, CompressedContextView):
        ctx_payload = {
            "context_pack_cid": compressed_context.context_pack_cid,
            "capsule_uncertainty": compressed_context.capsule_uncertainty,
            "token_savings_eligible": compressed_context.token_savings_eligible,
            "proof_cache_reuse": compressed_context.proof_cache_reuse,
            "includes_private_source": compressed_context.includes_private_source,
            "expanded_context_pack_cid": compressed_context.expanded_context_pack_cid,
        }
    elif isinstance(compressed_context, Mapping):
        ctx_payload = {
            k: compressed_context[k]
            for k in (
                "context_pack_cid",
                "compressed_context_pack_cid",
                "cid",
                "capsule_uncertainty",
                "token_savings_eligible",
                "proof_cache_reuse",
                "includes_private_source",
                "expanded_context_pack_cid",
            )
            if k in compressed_context
        }
    else:
        ctx_payload = {"context_pack_cid": str(compressed_context)}

    if isinstance(repository_state, RepositoryStateSignals):
        repo_payload = {
            "repository_state_cid": repository_state.repository_state_cid,
            "recent_omission": repository_state.recent_omission,
            "recent_failure": repository_state.recent_failure,
            "verification_bundle_cid": repository_state.verification_bundle_cid,
        }
    elif isinstance(repository_state, Mapping):
        repo_payload = {
            k: repository_state[k]
            for k in (
                "repository_state_cid",
                "repo_state_cid",
                "cid",
                "recent_omission",
                "recent_failure",
                "verification_bundle_cid",
            )
            if k in repository_state
        }
    else:
        repo_payload = {"repository_state_cid": str(repository_state)}

    return cid_for_structured(
        {
            "kind": "scg-audit-input-identity@1",
            "task": task_payload,
            "compressed_context": ctx_payload,
            "repository_state": repo_payload,
            "audit_policy_cid": audit_policy_cid,
            "disclosure_policy_cid": disclosure_policy_cid,
            "execution_mode": execution_mode,
            "expanded_provider_id": expanded_provider_id,
            "run_expansion": bool(run_expansion),
            "expansion_plan_cid": expansion_plan_cid,
        }
    )


def compute_shadow_input_identity(
    *,
    task: Any,
    compressed_context: Any,
    repository_state: Any,
    audit_policy_cid: str | None = None,
    execution_mode: str = ExecutionMode.SIMULATED.value,
    expanded_provider_id: str | None = None,
) -> str:
    return compute_audit_input_identity(
        task=task,
        compressed_context=compressed_context,
        repository_state=repository_state,
        audit_policy_cid=audit_policy_cid,
        execution_mode=execution_mode,
        expanded_provider_id=expanded_provider_id,
        run_expansion=False,
    )


def compute_expand_input_identity(
    plan: ContextExpansionPlan,
    *,
    model_policy_cid: str,
    verification_policy_cid: str,
    comparative_outcome: str | None = None,
) -> str:
    return cid_for_structured(
        {
            "kind": "scg-expand-input-identity@1",
            "plan_cid": plan.plan_cid,
            "audit_case_cid": plan.audit_case_cid,
            "model_policy_cid": model_policy_cid,
            "verification_policy_cid": verification_policy_cid,
            "comparative_outcome": comparative_outcome,
            "max_steps": plan.max_steps,
            "max_token_growth": plan.max_token_growth,
            "step_count": plan.step_count,
        }
    )


def _coerce_task(task: ShadowTaskView | Mapping[str, Any] | str) -> ShadowTaskView:
    if isinstance(task, ShadowTaskView):
        return task
    if isinstance(task, str):
        return ShadowTaskView(task_id=task)
    if not isinstance(task, Mapping):
        raise SemanticGovernorRuntimeError(
            "task must be ShadowTaskView, mapping, or task_id string"
        )
    return ShadowTaskView(
        task_id=task.get("task_id", task.get("id", "task.unknown")),
        task_class=task.get("task_class", task.get("class", "default")),
        risk_class=task.get("risk_class", task.get("risk", "low")),
        environment=task.get("environment", task.get("env")),
        route_id=task.get("route_id", task.get("compressed_route_id", "route.compressed")),
        expanded_route_id=task.get("expanded_route_id", "route.expanded"),
        promotion_evaluation=bool(task.get("promotion_evaluation", False)),
        new_task_class=bool(task.get("new_task_class", False)),
        new_analyzer=bool(task.get("new_analyzer", False)),
        new_route=bool(task.get("new_route", False)),
        notes=task.get("notes"),
        metadata=dict(task.get("metadata") or {}),
    )


def _result_from_checkpoint(
    checkpoint: AuditCheckpoint,
    *,
    recovered: bool = False,
    idempotent_hit: bool = False,
    extra_reasons: Sequence[str] = (),
) -> AuditTaskResult:
    reasons = list(checkpoint.reason_codes) + list(extra_reasons)
    if recovered:
        reasons.append("audit_recovered_from_checkpoint")
    if idempotent_hit:
        reasons.append("idempotent_input_identity_hit")
    disposition = checkpoint.disposition
    if recovered and checkpoint.phase == AuditPhase.COMPLETE.value:
        disposition = AuditDisposition.RECOVERED.value
    elif idempotent_hit and checkpoint.phase == AuditPhase.COMPLETE.value:
        disposition = AuditDisposition.IDEMPOTENT_HIT.value
    decision_action = None
    if checkpoint.expansion_result is not None:
        decision_action = checkpoint.expansion_result.get("decision_action")
    elif checkpoint.differential is not None:
        decision_action = checkpoint.differential.get("decision_action")
    return AuditTaskResult(
        audit_id=checkpoint.audit_id,
        task_id=checkpoint.task_id,
        disposition=disposition,
        phase=checkpoint.phase,
        input_identity_cid=checkpoint.input_identity_cid,
        recovered=recovered,
        idempotent_hit=idempotent_hit,
        plan_cid=checkpoint.plan_cid,
        shadow_result_cid=checkpoint.shadow_result_cid,
        differential_cid=checkpoint.differential_cid,
        expansion_result_cid=checkpoint.expansion_result_cid,
        comparative_outcome=checkpoint.comparative_outcome,
        decision_action=decision_action,
        reason_codes=tuple(reasons),
        plan=checkpoint.plan,
        shadow_result=checkpoint.shadow_result,
        differential=checkpoint.differential,
        expansion_result=checkpoint.expansion_result,
        checkpoint_cid=checkpoint.checkpoint_cid,
        notes=checkpoint.notes,
        metadata=_thaw_structured(checkpoint.metadata),
    )


def _inspect_shadow_result_failures(result: ShadowExecutionResult) -> None:
    """Apply suppressed-failure and simulated-live gates to a shadow result."""

    for attempt in (result.compressed_attempt, result.expanded_attempt):
        if attempt is None:
            continue
        reject_suppressed_failure(attempt=attempt, metadata=result.metadata)
        reject_simulated_live_quality_claim(
            execution_mode=attempt.execution_mode,
            acceptance_disposition=attempt.acceptance_disposition,
            production_eligible=attempt.verification.production_eligible,
            metadata=result.metadata,
        )
    if not expanded_never_auto_accepts(result):
        raise SuppressedFailureError(
            "expanded attempt must never auto-accept",
            reason_code="suppressed_failure_rejected",
        )


# ---------------------------------------------------------------------------
# GovernorRuntime composition
# ---------------------------------------------------------------------------


@dataclass
class GovernorRuntime:
    """One composition path for resumable shadow/expansion audits.

    Coordinates planning, privacy admission, paired shadow execution,
    differential comparison, bounded expansion, and durable checkpoint
    recovery without inventing parallel authority paths.
    """

    audit_store: AuditCheckpointStore = field(
        default_factory=InMemoryAuditCheckpointStore
    )
    expansion_store: ExpansionCheckpointStore = field(
        default_factory=InMemoryExpansionCheckpointStore
    )
    audit_policy: ShadowSamplingPolicy | None = None
    disclosure_policy: ShadowDisclosurePolicy | None = None
    model_policy: ExpansionModelPolicy | None = None
    verification_policy: ExpansionVerificationPolicy | None = None
    attempt_runner: ShadowAttemptRunner | None = None
    expansion_runner: ExpansionStepRunner | None = None
    worktree_lifecycle: EvaluationWorktreeLifecycle | None = None
    resource_gate: ShadowResourceGate | None = None
    production_guard: ProductionCheckoutGuard | None = None
    default_execution_mode: str = ExecutionMode.SIMULATED.value
    fallback_expanded_to_local: bool = True

    def __post_init__(self) -> None:
        if self.audit_policy is None:
            self.audit_policy = default_shadow_sampling_policy(random_seed=7)
        if self.disclosure_policy is None:
            self.disclosure_policy = default_shadow_disclosure_policy()
        if self.model_policy is None:
            self.model_policy = default_model_policy()
        if self.verification_policy is None:
            self.verification_policy = default_verification_policy()
        if self.attempt_runner is None:
            self.attempt_runner = SimulatedShadowAttemptRunner()
        if self.worktree_lifecycle is None:
            self.worktree_lifecycle = InMemoryEvaluationWorktreeLifecycle()
        self.default_execution_mode = _enum(
            self.default_execution_mode, ExecutionMode, "default_execution_mode"
        )

    def _audit_id_for(self, input_identity_cid: str, task_id: str) -> str:
        # Deterministic audit id from input identity so recovery keys align.
        digest = input_identity_cid[-12:] if len(input_identity_cid) >= 12 else input_identity_cid
        safe_task = re.sub(r"[^A-Za-z0-9._+-]+", "_", task_id)[:48]
        return f"audit.{safe_task}.{digest}"

    def _persist(self, checkpoint: AuditCheckpoint) -> AuditCheckpoint:
        self.audit_store.save(checkpoint)
        return checkpoint

    def recover_audit(self, audit_id: str) -> AuditTaskResult:
        """Load a durable audit checkpoint and return its published view."""

        checkpoint = self.audit_store.load(audit_id)
        if checkpoint is None:
            raise AuditRecoveryError(
                f"no checkpoint for audit_id={audit_id!r}",
                reason_code="checkpoint_missing",
            )
        recovered = checkpoint.phase != AuditPhase.COMPLETE.value
        return _result_from_checkpoint(
            checkpoint,
            recovered=True,
            extra_reasons=("explicit_recover_audit",),
        )

    def shadow_task(
        self,
        task: ShadowTaskView | Mapping[str, Any] | str,
        compressed_context: CompressedContextView | Mapping[str, Any] | str,
        repository_state: RepositoryStateSignals | Mapping[str, Any] | str,
        audit_policy: ShadowSamplingPolicy | Mapping[str, Any] | None = None,
        *,
        disclosure_policy: ShadowDisclosurePolicy | Mapping[str, Any] | None = None,
        expanded_provider_id: str | None = None,
        expanded_context: Any = None,
        compressed_context_payload: Any = None,
        execution_mode: str | None = None,
        sample_roll: int | None = None,
        require_selected: bool = False,
        cancellation_token: ShadowCancellationToken | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ShadowTaskResult:
        """Plan, admit privacy, execute paired shadow, compare, and publish."""

        resolved_task = _coerce_task(task)
        policy = (
            self.audit_policy
            if audit_policy is None
            else (
                audit_policy
                if isinstance(audit_policy, ShadowSamplingPolicy)
                else ShadowSamplingPolicy.from_dict(audit_policy)
            )
        )
        disc = (
            self.disclosure_policy
            if disclosure_policy is None
            else (
                disclosure_policy
                if isinstance(disclosure_policy, ShadowDisclosurePolicy)
                else ShadowDisclosurePolicy.from_dict(disclosure_policy)
            )
        )
        mode = _enum(
            execution_mode if execution_mode is not None else self.default_execution_mode,
            ExecutionMode,
            "execution_mode",
        )
        reject_simulated_live_quality_claim(
            execution_mode=mode,
            metadata=metadata,
        )

        input_identity = compute_shadow_input_identity(
            task=resolved_task,
            compressed_context=compressed_context,
            repository_state=repository_state,
            audit_policy_cid=policy.policy_cid,
            execution_mode=mode,
            expanded_provider_id=expanded_provider_id,
        )

        # Idempotent hit on completed prior work with same shadow input identity.
        prior = self.audit_store.load_by_input_identity(input_identity)
        if (
            prior is not None
            and prior.phase == AuditPhase.COMPLETE.value
            and prior.shadow_result_cid is not None
        ):
            return ShadowTaskResult(
                task_id=resolved_task.task_id,
                disposition=ShadowTaskDisposition.IDEMPOTENT_HIT.value,
                input_identity_cid=input_identity,
                plan_cid=prior.plan_cid,
                shadow_result_cid=prior.shadow_result_cid,
                differential_cid=prior.differential_cid,
                comparative_outcome=prior.comparative_outcome,
                idempotent_hit=True,
                reason_codes=(
                    "idempotent_input_identity_hit",
                    "duplicate_inputs_preserve_identities",
                ),
                plan=prior.plan,
                shadow_result=prior.shadow_result,
                differential=prior.differential,
                notes="Duplicate shadow inputs returned sealed prior identities",
                metadata=dict(metadata or {}),
            )

        includes_private = None
        if expanded_context is not None:
            includes_private = contains_private_source(expanded_context)
        elif isinstance(compressed_context, Mapping):
            includes_private = bool(
                compressed_context.get("includes_private_source")
            ) or contains_private_source(compressed_context)
        elif isinstance(compressed_context, CompressedContextView):
            includes_private = compressed_context.includes_private_source

        # Fail closed on private → unapproved external before planning executes.
        if expanded_provider_id is not None:
            reject_private_external_shadow(
                disclosure_policy=disc,
                provider_id=expanded_provider_id,
                context=expanded_context
                if expanded_context is not None
                else compressed_context,
                includes_private_source=includes_private,
                allow_external_expanded_disclosure=bool(
                    getattr(policy, "allow_external_expanded_disclosure", False)
                ),
                raise_on_forbidden=True,
            )

        decision = create_shadow_plan(
            resolved_task,
            compressed_context,
            repository_state,
            policy,
            disclosure_policy=disc,
            expanded_provider_id=expanded_provider_id,
            expanded_context=expanded_context,
            require_selected=require_selected,
            sample_roll=sample_roll,
        )

        if not decision.selected or decision.plan is None:
            return ShadowTaskResult(
                task_id=resolved_task.task_id,
                disposition=ShadowTaskDisposition.NOT_SELECTED.value,
                input_identity_cid=input_identity,
                reason_codes=tuple(decision.reason_codes)
                + ("shadow_not_selected",),
                plan_decision=decision.to_dict(),
                notes="Shadow sampling did not select this task",
                metadata=dict(metadata or {}),
            )

        plan = admit_shadow_plan(decision.plan)
        # Re-check disclosure posture from the sealed plan.
        if plan.allow_external_expanded_disclosure and expanded_provider_id:
            reject_private_external_shadow(
                disclosure_policy=disc,
                provider_id=expanded_provider_id,
                context=expanded_context,
                includes_private_source=includes_private,
                allow_external_expanded_disclosure=True,
                raise_on_forbidden=True,
            )

        shadow_result = execute_shadow_plan(
            plan,
            compressed_context=compressed_context_payload
            if compressed_context_payload is not None
            else compressed_context,
            expanded_context=expanded_context,
            disclosure_policy=disc,
            attempt_runner=self.attempt_runner,
            worktree_lifecycle=self.worktree_lifecycle,
            resource_gate=self.resource_gate,
            production_guard=self.production_guard,
            cancellation_token=cancellation_token,
            expanded_provider_id=expanded_provider_id,
            execution_mode=mode,
            fallback_expanded_to_local=self.fallback_expanded_to_local,
        )
        verify_result_identity(shadow_result)
        _inspect_shadow_result_failures(shadow_result)

        outcome = compare_shadow_results(shadow_result=shadow_result)
        differential_map = outcome.to_dict()
        differential_cid = (
            outcome.outcome_cid
            if hasattr(outcome, "outcome_cid")
            else cid_for_structured(differential_map)
        )

        result = ShadowTaskResult(
            task_id=resolved_task.task_id,
            disposition=ShadowTaskDisposition.COMPLETE.value,
            input_identity_cid=input_identity,
            plan_cid=plan.plan_cid,
            shadow_result_cid=shadow_result.result_cid,
            differential_cid=differential_cid,
            comparative_outcome=str(outcome.comparative_outcome),
            idempotent_hit=False,
            reason_codes=(
                "shadow_task_complete",
                "differential_published",
                *tuple(decision.selection_reasons),
            ),
            plan_decision=decision.to_dict(),
            plan=plan.to_dict(),
            shadow_result=shadow_result.to_dict(),
            differential=differential_map,
            notes=None,
            metadata=dict(metadata or {}),
        )

        # Persist sealed identities so duplicate shadow_task inputs hit cache.
        shadow_audit_id = self._audit_id_for(input_identity, resolved_task.task_id)
        self._persist(
            AuditCheckpoint(
                audit_id=shadow_audit_id,
                task_id=resolved_task.task_id,
                phase=AuditPhase.COMPLETE.value,
                input_identity_cid=input_identity,
                generation=0,
                plan_cid=plan.plan_cid,
                shadow_result_cid=shadow_result.result_cid,
                differential_cid=differential_cid,
                comparative_outcome=str(outcome.comparative_outcome),
                disposition=AuditDisposition.COMPLETE.value,
                reason_codes=("shadow_task_persisted", "duplicate_inputs_preserve_identities"),
                plan=plan.to_dict(),
                shadow_result=shadow_result.to_dict(),
                differential=differential_map,
                metadata=dict(metadata or {}),
            )
        )
        return result

    def expand_audit(
        self,
        plan: ContextExpansionPlan | Mapping[str, Any],
        model_policy: ExpansionModelPolicy | Mapping[str, Any] | None = None,
        verification_policy: ExpansionVerificationPolicy | Mapping[str, Any] | None = None,
        *,
        runner: ExpansionStepRunner | None = None,
        checkpoint: ExpansionLoopCheckpoint | Mapping[str, Any] | None = None,
        comparative_outcome: str | None = None,
        counterexample_cids: Sequence[str] = (),
        cancel_requested: Callable[[], bool] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExpandAuditResult:
        """Admit bounds, execute (or resume) expansion, and publish a decision."""

        resolved_plan = reject_unbounded_expansion(plan)
        resolved_model = (
            self.model_policy
            if model_policy is None
            else (
                model_policy
                if isinstance(model_policy, ExpansionModelPolicy)
                else ExpansionModelPolicy.from_dict(model_policy)
            )
        )
        resolved_verify = (
            self.verification_policy
            if verification_policy is None
            else (
                verification_policy
                if isinstance(verification_policy, ExpansionVerificationPolicy)
                else ExpansionVerificationPolicy.from_dict(verification_policy)
            )
        )
        reject_simulated_live_quality_claim(
            execution_mode=self.default_execution_mode,
            metadata=metadata,
        )
        # Expansion metadata must not suppress failures.
        reject_suppressed_failure(metadata=metadata)

        input_identity = compute_expand_input_identity(
            resolved_plan,
            model_policy_cid=resolved_model.policy_cid,
            verification_policy_cid=resolved_verify.policy_cid,
            comparative_outcome=comparative_outcome,
        )

        # Resume detection via expansion checkpoint store.
        prior_ckpt: ExpansionLoopCheckpoint | None = None
        if checkpoint is not None:
            prior_ckpt = (
                checkpoint
                if isinstance(checkpoint, ExpansionLoopCheckpoint)
                else ExpansionLoopCheckpoint.from_dict(checkpoint)
            )
        elif self.expansion_store is not None:
            prior_ckpt = self.expansion_store.load(resolved_plan.plan_cid)

        recovered = prior_ckpt is not None

        loop_result = execute_expansion_loop(
            resolved_plan,
            resolved_model,
            resolved_verify,
            runner=runner if runner is not None else self.expansion_runner,
            checkpoint_store=self.expansion_store,
            checkpoint=prior_ckpt,
            comparative_outcome=comparative_outcome,
            counterexample_cids=counterexample_cids,
            cancel_requested=cancel_requested,
            metadata=metadata,
        )

        # Expansion outcomes must not suppress failures or claim production accept.
        if loop_result.compression_blamed:
            raise SuppressedFailureError(
                "expansion result cannot suppress failure by blaming compression "
                "when both contexts failed",
                reason_code="suppressed_failure_rejected",
            )
        meta = dict(loop_result.metadata) if loop_result.metadata else {}
        reject_suppressed_failure(metadata=meta)

        disposition = ExpandAuditDisposition.COMPLETE.value
        reasons = list(loop_result.reason_codes)
        if recovered:
            disposition = ExpandAuditDisposition.RECOVERED.value
            reasons.append("expansion_recovered_from_checkpoint")
        reasons.append("expand_audit_complete")
        reasons.append("bounded_expansion_enforced")

        return ExpandAuditResult(
            audit_case_cid=resolved_plan.audit_case_cid,
            disposition=disposition,
            input_identity_cid=input_identity,
            expansion_result_cid=loop_result.result_cid,
            plan_cid=resolved_plan.plan_cid,
            recovered=recovered,
            idempotent_hit=False,
            decision_action=loop_result.decision_action,
            comparative_outcome=loop_result.comparative_outcome,
            reason_codes=tuple(reasons),
            expansion_result=loop_result.to_dict(),
            notes=loop_result.notes,
            metadata=dict(metadata or {}),
        )

    def audit_task(
        self,
        task: ShadowTaskView | Mapping[str, Any] | str,
        compressed_context: CompressedContextView | Mapping[str, Any] | str,
        repository_state: RepositoryStateSignals | Mapping[str, Any] | str,
        audit_policy: ShadowSamplingPolicy | Mapping[str, Any] | None = None,
        *,
        disclosure_policy: ShadowDisclosurePolicy | Mapping[str, Any] | None = None,
        expanded_provider_id: str | None = None,
        expanded_context: Any = None,
        compressed_context_payload: Any = None,
        execution_mode: str | None = None,
        sample_roll: int | None = None,
        require_selected: bool = False,
        run_expansion: bool = False,
        expansion_plan: ContextExpansionPlan | Mapping[str, Any] | None = None,
        cancellation_token: ShadowCancellationToken | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        interrupt_after_phase: AuditPhase | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AuditTaskResult:
        """Full resumable audit: plan → shadow → compare → optional expand.

        When a prior checkpoint for the same input identity exists:

        * complete → return idempotent hit with sealed identities
        * interrupted → resume remaining phases (recovered=True)
        """

        resolved_task = _coerce_task(task)
        policy = (
            self.audit_policy
            if audit_policy is None
            else (
                audit_policy
                if isinstance(audit_policy, ShadowSamplingPolicy)
                else ShadowSamplingPolicy.from_dict(audit_policy)
            )
        )
        disc = (
            self.disclosure_policy
            if disclosure_policy is None
            else (
                disclosure_policy
                if isinstance(disclosure_policy, ShadowDisclosurePolicy)
                else ShadowDisclosurePolicy.from_dict(disclosure_policy)
            )
        )
        mode = _enum(
            execution_mode if execution_mode is not None else self.default_execution_mode,
            ExecutionMode,
            "execution_mode",
        )
        reject_simulated_live_quality_claim(execution_mode=mode, metadata=metadata)
        reject_suppressed_failure(metadata=metadata)

        expansion_plan_cid = None
        if expansion_plan is not None:
            admitted_plan = reject_unbounded_expansion(expansion_plan)
            expansion_plan_cid = admitted_plan.plan_cid
            expansion_plan = admitted_plan

        input_identity = compute_audit_input_identity(
            task=resolved_task,
            compressed_context=compressed_context,
            repository_state=repository_state,
            audit_policy_cid=policy.policy_cid,
            disclosure_policy_cid=getattr(disc, "policy_cid", None),
            execution_mode=mode,
            expanded_provider_id=expanded_provider_id,
            run_expansion=run_expansion,
            expansion_plan_cid=expansion_plan_cid,
        )
        audit_id = self._audit_id_for(input_identity, resolved_task.task_id)

        prior = self.audit_store.load(audit_id)
        if prior is None:
            prior = self.audit_store.load_by_input_identity(input_identity)

        if prior is not None and prior.input_identity_cid != input_identity:
            raise AuditRecoveryError(
                "checkpoint input identity does not match request",
                reason_code="checkpoint_identity_mismatch",
                details={
                    "expected": input_identity,
                    "found": prior.input_identity_cid,
                },
            )

        # Complete prior with same inputs → preserve identities.
        if prior is not None and prior.phase == AuditPhase.COMPLETE.value:
            return _result_from_checkpoint(
                prior,
                recovered=False,
                idempotent_hit=True,
                extra_reasons=("duplicate_inputs_preserve_identities",),
            )

        recovered = prior is not None and prior.phase in {
            AuditPhase.INTERRUPTED.value,
            AuditPhase.PLANNED.value,
            AuditPhase.SHADOWED.value,
            AuditPhase.COMPARED.value,
            AuditPhase.EXPANDED.value,
            AuditPhase.ADMITTED.value,
        }
        generation = int(prior.generation) + 1 if prior is not None else 0

        interrupt_phase = (
            None
            if interrupt_after_phase is None
            else _enum(interrupt_after_phase, AuditPhase, "interrupt_after_phase")
        )

        plan_map = prior.plan if prior is not None else None
        plan_cid = prior.plan_cid if prior is not None else None
        shadow_map = prior.shadow_result if prior is not None else None
        shadow_cid = prior.shadow_result_cid if prior is not None else None
        diff_map = prior.differential if prior is not None else None
        diff_cid = prior.differential_cid if prior is not None else None
        expansion_map = prior.expansion_result if prior is not None else None
        expansion_cid = prior.expansion_result_cid if prior is not None else None
        comparative_outcome = prior.comparative_outcome if prior is not None else None
        reasons: list[str] = list(prior.reason_codes) if prior is not None else []
        if recovered:
            reasons.append("resuming_interrupted_audit")

        phase = prior.phase if prior is not None else AuditPhase.ADMITTED.value

        def _checkpoint(
            current_phase: str,
            disposition: str,
            **fields: Any,
        ) -> AuditCheckpoint:
            base = {
                "audit_id": audit_id,
                "task_id": resolved_task.task_id,
                "phase": current_phase,
                "input_identity_cid": input_identity,
                "generation": generation,
                "plan_cid": plan_cid,
                "shadow_result_cid": shadow_cid,
                "differential_cid": diff_cid,
                "expansion_result_cid": expansion_cid,
                "comparative_outcome": comparative_outcome,
                "disposition": disposition,
                "reason_codes": tuple(sorted(set(reasons))),
                "plan": plan_map,
                "shadow_result": shadow_map,
                "differential": diff_map,
                "expansion_result": expansion_map,
                "metadata": dict(metadata or {}),
            }
            base.update(fields)
            return self._persist(AuditCheckpoint(**base))

        # --- ADMITTED -------------------------------------------------------
        if phase == AuditPhase.ADMITTED.value or prior is None:
            ckpt = _checkpoint(
                AuditPhase.ADMITTED.value,
                AuditDisposition.INTERRUPTED.value,
            )
            phase = AuditPhase.ADMITTED.value
            if interrupt_phase == AuditPhase.ADMITTED.value:
                reasons.append("interrupted_after_admitted")
                ckpt = _checkpoint(
                    AuditPhase.INTERRUPTED.value,
                    AuditDisposition.INTERRUPTED.value,
                    reason_codes=tuple(sorted(set(reasons))),
                )
                return _result_from_checkpoint(ckpt, recovered=recovered)

        # --- PLANNED / SHADOWED via shadow_task composition -----------------
        need_shadow = phase in {
            AuditPhase.ADMITTED.value,
            AuditPhase.PLANNED.value,
            AuditPhase.INTERRUPTED.value,
        } and shadow_cid is None

        if need_shadow:
            try:
                shadow = self.shadow_task(
                    resolved_task,
                    compressed_context,
                    repository_state,
                    policy,
                    disclosure_policy=disc,
                    expanded_provider_id=expanded_provider_id,
                    expanded_context=expanded_context,
                    compressed_context_payload=compressed_context_payload,
                    execution_mode=mode,
                    sample_roll=sample_roll,
                    require_selected=require_selected,
                    cancellation_token=cancellation_token,
                    metadata=metadata,
                )
            except PrivateExternalShadowError as exc:
                reasons.extend(
                    [
                        exc.reason_code or "private_external_shadow_forbidden",
                        "audit_rejected",
                    ]
                )
                ckpt = _checkpoint(
                    AuditPhase.REJECTED.value,
                    AuditDisposition.REJECTED.value,
                    reason_codes=tuple(sorted(set(reasons))),
                    notes=str(exc),
                )
                return _result_from_checkpoint(ckpt, recovered=recovered)

            if shadow.disposition == ShadowTaskDisposition.NOT_SELECTED.value:
                reasons.append("shadow_not_selected")
                ckpt = _checkpoint(
                    AuditPhase.COMPLETE.value,
                    AuditDisposition.NOT_SELECTED.value,
                    reason_codes=tuple(sorted(set(reasons))),
                    plan=shadow.plan_decision,
                )
                return _result_from_checkpoint(ckpt, recovered=recovered)

            plan_map = shadow.plan
            plan_cid = shadow.plan_cid
            shadow_map = shadow.shadow_result
            shadow_cid = shadow.shadow_result_cid
            diff_map = shadow.differential
            diff_cid = shadow.differential_cid
            comparative_outcome = shadow.comparative_outcome
            reasons.extend(shadow.reason_codes)
            reasons.append("shadow_and_differential_complete")

            ckpt = _checkpoint(
                AuditPhase.COMPARED.value,
                AuditDisposition.INTERRUPTED.value,
            )
            phase = AuditPhase.COMPARED.value

            if interrupt_phase in {
                AuditPhase.PLANNED.value,
                AuditPhase.SHADOWED.value,
                AuditPhase.COMPARED.value,
            }:
                reasons.append(f"interrupted_after_{interrupt_phase}")
                ckpt = _checkpoint(
                    AuditPhase.INTERRUPTED.value,
                    AuditDisposition.INTERRUPTED.value,
                    reason_codes=tuple(sorted(set(reasons))),
                )
                return _result_from_checkpoint(ckpt, recovered=recovered)

        # --- OPTIONAL EXPANSION --------------------------------------------
        if run_expansion and expansion_plan is not None and expansion_cid is None:
            if cancel_requested is not None and cancel_requested():
                reasons.append("cancelled_before_expansion")
                ckpt = _checkpoint(
                    AuditPhase.INTERRUPTED.value,
                    AuditDisposition.INTERRUPTED.value,
                    reason_codes=tuple(sorted(set(reasons))),
                )
                return _result_from_checkpoint(ckpt, recovered=recovered)

            try:
                expand = self.expand_audit(
                    expansion_plan,
                    comparative_outcome=comparative_outcome,
                    cancel_requested=cancel_requested,
                    metadata=metadata,
                )
            except UnboundedExpansionError as exc:
                reasons.extend(
                    [
                        exc.reason_code or "unbounded_expansion_rejected",
                        "audit_rejected",
                    ]
                )
                ckpt = _checkpoint(
                    AuditPhase.REJECTED.value,
                    AuditDisposition.REJECTED.value,
                    reason_codes=tuple(sorted(set(reasons))),
                    notes=str(exc),
                )
                return _result_from_checkpoint(ckpt, recovered=recovered)
            except SuppressedFailureError as exc:
                reasons.extend(
                    [
                        exc.reason_code or "suppressed_failure_rejected",
                        "audit_rejected",
                    ]
                )
                ckpt = _checkpoint(
                    AuditPhase.REJECTED.value,
                    AuditDisposition.REJECTED.value,
                    reason_codes=tuple(sorted(set(reasons))),
                    notes=str(exc),
                )
                return _result_from_checkpoint(ckpt, recovered=recovered)

            expansion_map = expand.expansion_result
            expansion_cid = expand.expansion_result_cid
            reasons.extend(expand.reason_codes)
            if expand.recovered:
                reasons.append("expansion_phase_recovered")
            phase = AuditPhase.EXPANDED.value
            ckpt = _checkpoint(
                AuditPhase.EXPANDED.value,
                AuditDisposition.INTERRUPTED.value,
            )
            if interrupt_phase == AuditPhase.EXPANDED.value:
                reasons.append("interrupted_after_expanded")
                ckpt = _checkpoint(
                    AuditPhase.INTERRUPTED.value,
                    AuditDisposition.INTERRUPTED.value,
                    reason_codes=tuple(sorted(set(reasons))),
                )
                return _result_from_checkpoint(ckpt, recovered=recovered)

        # --- COMPLETE -------------------------------------------------------
        reasons.append("audit_task_complete")
        if recovered:
            reasons.append("interrupted_audit_recovered")
        disposition = (
            AuditDisposition.RECOVERED.value
            if recovered
            else AuditDisposition.COMPLETE.value
        )
        ckpt = _checkpoint(
            AuditPhase.COMPLETE.value,
            disposition,
            reason_codes=tuple(sorted(set(reasons))),
        )
        return _result_from_checkpoint(
            ckpt,
            recovered=recovered,
            idempotent_hit=False,
        )


# ---------------------------------------------------------------------------
# Module-level frozen interfaces
# ---------------------------------------------------------------------------


def audit_task(
    task: ShadowTaskView | Mapping[str, Any] | str,
    compressed_context: CompressedContextView | Mapping[str, Any] | str,
    repository_state: RepositoryStateSignals | Mapping[str, Any] | str,
    audit_policy: ShadowSamplingPolicy | Mapping[str, Any] | None = None,
    *,
    runtime: GovernorRuntime | None = None,
    **kwargs: Any,
) -> AuditTaskResult:
    """Frozen ``audit_task@1`` entry: full resumable audit composition."""

    rt = runtime if runtime is not None else GovernorRuntime()
    return rt.audit_task(
        task, compressed_context, repository_state, audit_policy, **kwargs
    )


def shadow_task(
    task: ShadowTaskView | Mapping[str, Any] | str,
    compressed_context: CompressedContextView | Mapping[str, Any] | str,
    repository_state: RepositoryStateSignals | Mapping[str, Any] | str,
    audit_policy: ShadowSamplingPolicy | Mapping[str, Any] | None = None,
    *,
    runtime: GovernorRuntime | None = None,
    **kwargs: Any,
) -> ShadowTaskResult:
    """Frozen ``shadow_task@1`` entry: plan + execute + compare."""

    rt = runtime if runtime is not None else GovernorRuntime()
    return rt.shadow_task(
        task, compressed_context, repository_state, audit_policy, **kwargs
    )


def expand_audit(
    plan: ContextExpansionPlan | Mapping[str, Any],
    model_policy: ExpansionModelPolicy | Mapping[str, Any] | None = None,
    verification_policy: ExpansionVerificationPolicy | Mapping[str, Any] | None = None,
    *,
    runtime: GovernorRuntime | None = None,
    **kwargs: Any,
) -> ExpandAuditResult:
    """Frozen ``expand_audit@1`` entry: bounded expansion with recovery."""

    rt = runtime if runtime is not None else GovernorRuntime()
    return rt.expand_audit(plan, model_policy, verification_policy, **kwargs)


def governor_runtime_interface_id() -> str:
    return GOVERNOR_RUNTIME_INTERFACE


def audit_task_interface_id() -> str:
    return AUDIT_TASK_INTERFACE


def shadow_task_interface_id() -> str:
    return SHADOW_TASK_INTERFACE


def expand_audit_interface_id() -> str:
    return EXPAND_AUDIT_INTERFACE


def runtime_conformance_evidence_id() -> str:
    return SCG_RUNTIME_CONFORMANCE_EVIDENCE


__all__ = [
    "SCG_RUNTIME_CONFORMANCE_EVIDENCE",
    "GOVERNOR_RUNTIME_INTERFACE",
    "AUDIT_TASK_INTERFACE",
    "SHADOW_TASK_INTERFACE",
    "EXPAND_AUDIT_INTERFACE",
    "AUDIT_CHECKPOINT_SCHEMA",
    "AUDIT_TASK_RESULT_SCHEMA",
    "SHADOW_TASK_RESULT_SCHEMA",
    "EXPAND_AUDIT_RESULT_SCHEMA",
    "MAX_RUNTIME_EXPANSION_STEPS",
    "MAX_RUNTIME_TOKEN_GROWTH",
    "SemanticGovernorRuntimeError",
    "AuditRecoveryError",
    "RuntimeAdmissionError",
    "PrivateExternalShadowError",
    "UnboundedExpansionError",
    "SuppressedFailureError",
    "SimulatedLiveQualityError",
    "AuditPhase",
    "AuditDisposition",
    "ShadowTaskDisposition",
    "ExpandAuditDisposition",
    "AuditCheckpoint",
    "AuditTaskResult",
    "ShadowTaskResult",
    "ExpandAuditResult",
    "AuditCheckpointStore",
    "InMemoryAuditCheckpointStore",
    "FilesystemAuditCheckpointStore",
    "GovernorRuntime",
    "audit_task",
    "shadow_task",
    "expand_audit",
    "compute_audit_input_identity",
    "compute_shadow_input_identity",
    "compute_expand_input_identity",
    "reject_private_external_shadow",
    "reject_unbounded_expansion",
    "reject_suppressed_failure",
    "reject_simulated_live_quality_claim",
    "reject_simulated_calibration_as_live",
    "governor_runtime_interface_id",
    "audit_task_interface_id",
    "shadow_task_interface_id",
    "expand_audit_interface_id",
    "runtime_conformance_evidence_id",
    # Re-export composition dependencies used by callers/tests.
    "CREATE_SHADOW_PLAN_INTERFACE",
    "EXECUTE_SHADOW_PLAN_INTERFACE",
    "EXECUTE_EXPANSION_LOOP_INTERFACE",
    "COMPARE_SHADOW_RESULTS_INTERFACE",
    "development_shadow_sampling_policy",
    "default_shadow_sampling_policy",
    "default_shadow_disclosure_policy",
]
