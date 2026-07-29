"""Supervisor-side pre-dispatch enforcement (SupervisorPreInvocationEnforcement@1).

LIG-037 / LIG-G110: thin accelerate adapter that revalidates exact-context
authorization receipts and one-time capabilities immediately before the
supervisor delegates a side effect.

Design invariants
-----------------
* **Lazy imports** — ``ipfs_datasets_py`` receipt / service modules are
  imported only when verification or evaluation runs.  Importing this module
  (or ``agent_supervisor`` more broadly) never requires optional heavy provers
  or the datasets package.
* **Explicit modes** — ``off`` / ``audit`` / ``shadow`` / ``enforce``.  Only
  ``enforce`` blocks dispatch; audit and shadow record observations without
  manufacturing theorem-proof authority.
* **Injected store / service** — consumption store and optional authorization
  service are constructor-injected; no global mutable default store is required
  for production wiring.
* **Exact binding** — supervisor actor, delegation, audience, task, plan,
  tool, arguments, effects, and environment must match the receipt context
  (and current pinned roots) before consumption.
* **Atomic one-shot consumption** — a successful enforce-path call consumes
  the capability (or receipt nonce) exactly once before the delegate runs;
  replayed tokens never reach the delegate.
* **Observation ≠ proof** — emitted decision/runtime observations are
  operational receipts, never treated as theorem-prover evidence.

Environment / flags (documented)
--------------------------------
* ``IPFS_ACCELERATE_ADMISSIBILITY_ENFORCEMENT_MODE`` — default mode when
  constructing via :meth:`SupervisorPreInvocationEnforcement.from_env`
  (``off`` | ``audit`` | ``shadow`` | ``enforce``; default ``off``).
"""

from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any, Final, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Interface / schema constants
# ---------------------------------------------------------------------------

SUPERVISOR_PRE_INVOCATION_ENFORCEMENT_INTERFACE: Final = (
    "SupervisorPreInvocationEnforcement@1"
)
SUPERVISOR_PRE_INVOCATION_ENFORCEMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/pre-invocation-enforcement@1"
)
SUPERVISOR_PRE_INVOCATION_ENFORCEMENT_VERSION: Final[int] = 1
SUPERVISOR_ENFORCEMENT_OBSERVATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/enforcement-observation@1"
)

ENV_ENFORCEMENT_MODE: Final = "IPFS_ACCELERATE_ADMISSIBILITY_ENFORCEMENT_MODE"

# Lazy-import targets (never imported at module load).
_DATASETS_RECEIPT_MODULE: Final = "ipfs_datasets_py.logic.admissibility.receipt"
_DATASETS_SERVICE_MODULE: Final = "ipfs_datasets_py.logic.admissibility.service"
_DATASETS_COMPOSE_MODULE: Final = "ipfs_datasets_py.logic.admissibility.compose"
_DATASETS_REASONS_MODULE: Final = "ipfs_datasets_py.logic.admissibility.reasons"

_BANNED_HEAVY_IMPORT_MARKERS: Final[tuple[str, ...]] = (
    "z3",
    "cvc5",
    "vampire",
    "lean_dojo",
    "shadowprover",
)


class AdmissibilityEnforcementError(ValueError):
    """Raised when the enforcement contract is violated (malformed config / args)."""


class EnforcementMode(str, Enum):
    """Explicit pre-dispatch enforcement modes (LIG rollout vocabulary)."""

    OFF = "off"
    AUDIT = "audit"
    SHADOW = "shadow"
    ENFORCE = "enforce"

    @property
    def blocks(self) -> bool:
        """True when non-allow / invalid receipts prevent the delegate call."""

        return self is EnforcementMode.ENFORCE

    @property
    def evaluates(self) -> bool:
        """True when receipts are verified and observations are emitted."""

        return self is not EnforcementMode.OFF


class EnforcementDisposition(str, Enum):
    """Normalized disposition of one pre-dispatch attempt."""

    OFF = "off"
    ALLOWED = "allowed"
    DENIED = "denied"
    AUDITED = "audited"
    SHADOW_ALLOWED = "shadow_allowed"
    SHADOW_WOULD_BLOCK = "shadow_would_block"
    ERROR = "error"


class EnforcementDenialReason(str, Enum):
    """Closed denial vocabulary for non-dispatch outcomes."""

    ABSTAIN = "abstain"
    REJECT = "reject"
    ERROR = "error"
    EXPIRED = "expired"
    REPLAYED = "replayed"
    ROOT_CHANGED = "root-changed"
    ENVIRONMENT_CHANGED = "environment-changed"
    CONTEXT_MISMATCH = "context-mismatch"
    MISSING_RECEIPT = "missing-receipt"
    MISSING_CAPABILITY = "missing-capability"
    NOT_ALLOW = "not-allow"
    CONSUMPTION_FAILED = "consumption-failed"
    SERVICE_UNAVAILABLE = "service-unavailable"
    DISABLED = "disabled"


def _truthy_env(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        if required:
            raise AdmissibilityEnforcementError(f"{name} is required; fail closed")
        return ""
    if not isinstance(value, str):
        raise AdmissibilityEnforcementError(f"{name} must be a string; fail closed")
    text = value.strip()
    if required and not text:
        raise AdmissibilityEnforcementError(f"{name} must be non-empty; fail closed")
    return text


def _parse_mode(value: Any) -> EnforcementMode:
    if isinstance(value, EnforcementMode):
        return value
    if not isinstance(value, str) or not value.strip():
        raise AdmissibilityEnforcementError(
            "enforcement mode must be one of: off, audit, shadow, enforce; fail closed"
        )
    try:
        return EnforcementMode(value.strip().lower())
    except ValueError as exc:
        raise AdmissibilityEnforcementError(
            f"unknown enforcement mode {value!r}; fail closed"
        ) from exc


# ---------------------------------------------------------------------------
# Lazy datasets dependency surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _DatasetsAuthSurface:
    """Bound modules for one successful datasets auth import."""

    receipt: ModuleType
    service: ModuleType
    compose: ModuleType
    reasons: ModuleType


_datasets_auth_surface: _DatasetsAuthSurface | None = None
_datasets_auth_import_error: str | None = None


def datasets_auth_available() -> bool:
    """Return True when datasets receipt/service packages can be imported."""

    try:
        _load_datasets_auth_surface()
        return True
    except AdmissibilityEnforcementError:
        return False


def _load_datasets_auth_surface() -> _DatasetsAuthSurface:
    """Import datasets receipt / service APIs once; fail closed on ImportError."""

    global _datasets_auth_surface, _datasets_auth_import_error
    if _datasets_auth_surface is not None:
        return _datasets_auth_surface
    if _datasets_auth_import_error is not None:
        raise AdmissibilityEnforcementError(
            f"ipfs_datasets_py authorization surface unavailable: "
            f"{_datasets_auth_import_error}; fail closed"
        )
    try:
        receipt = importlib.import_module(_DATASETS_RECEIPT_MODULE)
        service = importlib.import_module(_DATASETS_SERVICE_MODULE)
        compose = importlib.import_module(_DATASETS_COMPOSE_MODULE)
        reasons = importlib.import_module(_DATASETS_REASONS_MODULE)
    except Exception as exc:  # noqa: BLE001 — surface any import failure closed
        _datasets_auth_import_error = f"{type(exc).__name__}: {exc}"
        raise AdmissibilityEnforcementError(
            f"ipfs_datasets_py authorization surface unavailable: "
            f"{_datasets_auth_import_error}; fail closed"
        ) from exc
    _datasets_auth_surface = _DatasetsAuthSurface(
        receipt=receipt,
        service=service,
        compose=compose,
        reasons=reasons,
    )
    return _datasets_auth_surface


def reset_datasets_auth_surface_cache() -> None:
    """Test helper: clear the lazy import cache."""

    global _datasets_auth_surface, _datasets_auth_import_error
    _datasets_auth_surface = None
    _datasets_auth_import_error = None


# ---------------------------------------------------------------------------
# Consumption store (atomic one-shot)
# ---------------------------------------------------------------------------


@runtime_checkable
class CapabilityConsumptionStore(Protocol):
    """Minimal store contract for one-time capability / nonce consumption."""

    def try_consume(self, token_key: str, *, meta: Mapping[str, Any] | None = None) -> bool:
        """Atomically mark *token_key* consumed.

        Returns True on first successful consumption, False if already consumed
        or permanently rejected.
        """


class InMemoryCapabilityConsumptionStore:
    """Thread-safe in-memory compare-and-consume store for tests and local use."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._consumed: dict[str, dict[str, Any]] = {}

    def try_consume(
        self, token_key: str, *, meta: Mapping[str, Any] | None = None
    ) -> bool:
        key = _text(token_key, "token_key")
        with self._lock:
            if key in self._consumed:
                return False
            self._consumed[key] = dict(meta or {})
            return True

    def is_consumed(self, token_key: str) -> bool:
        key = _text(token_key, "token_key", required=True)
        with self._lock:
            return key in self._consumed

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {key: dict(value) for key, value in self._consumed.items()}

    def clear(self) -> None:
        with self._lock:
            self._consumed.clear()


# ---------------------------------------------------------------------------
# Supervisor invocation context binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SupervisorInvocationContext:
    """Exact supervisor binding revalidated before side-effect delegation.

    Fields mirror the security-relevant axes of ``BoundContext`` plus
    supervisor task/plan identifiers required by LIG-037.
    """

    actor_id: str
    audience_id: str
    tool_id: str
    request_digest: str
    arguments_digest: str
    environment_digest: str
    effect_ids: tuple[str, ...]
    task_id: str = ""
    plan_id: str = ""
    tool_version: str = ""
    environment_id: str = ""
    delegation_ids: tuple[str, ...] = ()
    delegation_digest: str = ""
    nonce: str = ""
    resource_ids: tuple[str, ...] = ()
    capability_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", _text(self.actor_id, "actor_id"))
        object.__setattr__(
            self, "audience_id", _text(self.audience_id, "audience_id")
        )
        object.__setattr__(self, "tool_id", _text(self.tool_id, "tool_id"))
        object.__setattr__(
            self, "request_digest", _text(self.request_digest, "request_digest")
        )
        object.__setattr__(
            self,
            "arguments_digest",
            _text(self.arguments_digest, "arguments_digest"),
        )
        object.__setattr__(
            self,
            "environment_digest",
            _text(self.environment_digest, "environment_digest"),
        )
        effects = self.effect_ids
        if isinstance(effects, (str, bytes, bytearray)) or not isinstance(
            effects, Sequence
        ):
            raise AdmissibilityEnforcementError(
                "effect_ids must be a sequence of strings; fail closed"
            )
        normalized_effects = tuple(
            _text(item, f"effect_ids[{index}]") for index, item in enumerate(effects)
        )
        object.__setattr__(self, "effect_ids", normalized_effects)
        object.__setattr__(
            self, "task_id", _text(self.task_id, "task_id", required=False)
        )
        object.__setattr__(
            self, "plan_id", _text(self.plan_id, "plan_id", required=False)
        )
        object.__setattr__(
            self,
            "tool_version",
            _text(self.tool_version, "tool_version", required=False),
        )
        object.__setattr__(
            self,
            "environment_id",
            _text(self.environment_id, "environment_id", required=False),
        )
        object.__setattr__(
            self,
            "delegation_digest",
            _text(self.delegation_digest, "delegation_digest", required=False),
        )
        object.__setattr__(
            self, "nonce", _text(self.nonce, "nonce", required=False)
        )
        if isinstance(self.delegation_ids, (str, bytes, bytearray)) or not isinstance(
            self.delegation_ids, Sequence
        ):
            raise AdmissibilityEnforcementError(
                "delegation_ids must be a sequence of strings; fail closed"
            )
        object.__setattr__(
            self,
            "delegation_ids",
            tuple(
                _text(item, f"delegation_ids[{index}]")
                for index, item in enumerate(self.delegation_ids)
            ),
        )
        if isinstance(self.resource_ids, (str, bytes, bytearray)) or not isinstance(
            self.resource_ids, Sequence
        ):
            raise AdmissibilityEnforcementError(
                "resource_ids must be a sequence of strings; fail closed"
            )
        object.__setattr__(
            self,
            "resource_ids",
            tuple(
                _text(item, f"resource_ids[{index}]")
                for index, item in enumerate(self.resource_ids)
            ),
        )
        if isinstance(self.capability_ids, (str, bytes, bytearray)) or not isinstance(
            self.capability_ids, Sequence
        ):
            raise AdmissibilityEnforcementError(
                "capability_ids must be a sequence of strings; fail closed"
            )
        object.__setattr__(
            self,
            "capability_ids",
            tuple(
                _text(item, f"capability_ids[{index}]")
                for index, item in enumerate(self.capability_ids)
            ),
        )
        if not isinstance(self.metadata, Mapping):
            raise AdmissibilityEnforcementError(
                "metadata must be a mapping; fail closed"
            )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "arguments_digest": self.arguments_digest,
            "audience_id": self.audience_id,
            "capability_ids": list(self.capability_ids),
            "delegation_digest": self.delegation_digest,
            "delegation_ids": list(self.delegation_ids),
            "effect_ids": list(self.effect_ids),
            "environment_digest": self.environment_digest,
            "environment_id": self.environment_id,
            "metadata": dict(self.metadata),
            "nonce": self.nonce,
            "plan_id": self.plan_id,
            "request_digest": self.request_digest,
            "resource_ids": list(self.resource_ids),
            "task_id": self.task_id,
            "tool_id": self.tool_id,
            "tool_version": self.tool_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SupervisorInvocationContext":
        if not isinstance(value, Mapping):
            raise AdmissibilityEnforcementError(
                "supervisor invocation context must be a mapping; fail closed"
            )
        return cls(
            actor_id=value.get("actor_id", ""),
            audience_id=value.get("audience_id", ""),
            tool_id=value.get("tool_id", ""),
            request_digest=value.get("request_digest", ""),
            arguments_digest=value.get("arguments_digest", ""),
            environment_digest=value.get("environment_digest", ""),
            effect_ids=tuple(value.get("effect_ids", ())),
            task_id=value.get("task_id", ""),
            plan_id=value.get("plan_id", ""),
            tool_version=value.get("tool_version", ""),
            environment_id=value.get("environment_id", ""),
            delegation_ids=tuple(value.get("delegation_ids", ())),
            delegation_digest=value.get("delegation_digest", ""),
            nonce=value.get("nonce", ""),
            resource_ids=tuple(value.get("resource_ids", ())),
            capability_ids=tuple(value.get("capability_ids", ())),
            metadata=dict(value.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# Observation (decision-runtime friendly; not theorem proof)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnforcementObservation:
    """Serializable pre-dispatch observation for decision-runtime / audit logs.

    Explicitly **not** a theorem-prover receipt.  ``theorem_proof`` is always
    False so consumers cannot treat this observation as formal proof authority.
    """

    disposition: EnforcementDisposition
    mode: EnforcementMode
    allowed: bool
    delegated: bool
    reason_codes: tuple[str, ...]
    receipt_id: str = ""
    capability_id: str = ""
    nonce: str = ""
    actor_id: str = ""
    audience_id: str = ""
    task_id: str = ""
    plan_id: str = ""
    tool_id: str = ""
    error: str = ""
    denial_reason: str = ""
    consumed: bool = False
    theorem_proof: bool = False
    schema: str = SUPERVISOR_ENFORCEMENT_OBSERVATION_SCHEMA
    interface: str = SUPERVISOR_PRE_INVOCATION_ENFORCEMENT_INTERFACE

    def __post_init__(self) -> None:
        # Hard invariant: runtime observation is never theorem proof evidence.
        object.__setattr__(self, "theorem_proof", False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "allowed": self.allowed,
            "audience_id": self.audience_id,
            "capability_id": self.capability_id,
            "consumed": self.consumed,
            "delegated": self.delegated,
            "denial_reason": self.denial_reason,
            "disposition": self.disposition.value,
            "error": self.error,
            "interface": self.interface,
            "mode": self.mode.value,
            "nonce": self.nonce,
            "plan_id": self.plan_id,
            "reason_codes": list(self.reason_codes),
            "receipt_id": self.receipt_id,
            "schema": self.schema,
            "task_id": self.task_id,
            "theorem_proof": False,
            "tool_id": self.tool_id,
        }


@dataclass(frozen=True, slots=True)
class EnforcementResult:
    """Outcome of one authorize-and-delegate attempt."""

    observation: EnforcementObservation
    delegate_result: Any = None
    delegate_called: bool = False
    receipt: Any | None = None
    capability: Any | None = None

    @property
    def allowed(self) -> bool:
        return self.observation.allowed

    @property
    def disposition(self) -> EnforcementDisposition:
        return self.observation.disposition

    def to_dict(self) -> dict[str, Any]:
        return {
            "delegate_called": self.delegate_called,
            "observation": self.observation.to_dict(),
            "has_receipt": self.receipt is not None,
            "has_capability": self.capability is not None,
        }


# ---------------------------------------------------------------------------
# Optional authorization service protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AuthorizationServiceLike(Protocol):
    """Minimal injected service surface (LIG-035 IntentAuthorizationService)."""

    def evaluate(self, *args: Any, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Enforcement core
# ---------------------------------------------------------------------------


def _consumption_key(
    *,
    capability: Any | None,
    receipt: Any | None,
    context: SupervisorInvocationContext,
) -> str:
    if capability is not None:
        cap_id = str(getattr(capability, "capability_id", "") or "")
        nonce = str(getattr(capability, "nonce", "") or "")
        digest = str(
            getattr(capability, "digest", None)
            or getattr(capability, "content_digest", "")
            or ""
        )
        if cap_id:
            return f"capability:{cap_id}:{nonce}:{digest}"
    if receipt is not None:
        receipt_id = str(getattr(receipt, "receipt_id", "") or "")
        nonce = str(getattr(receipt, "nonce", "") or context.nonce)
        digest = str(
            getattr(receipt, "digest", None)
            or getattr(receipt, "content_digest", "")
            or ""
        )
        if receipt_id:
            return f"receipt:{receipt_id}:{nonce}:{digest}"
    if context.nonce:
        return f"nonce:{context.nonce}:{context.request_digest}"
    raise AdmissibilityEnforcementError(
        "cannot derive consumption key without capability, receipt, or nonce; "
        "fail closed"
    )


def _coerce_receipt(surface: _DatasetsAuthSurface, receipt: Any) -> Any:
    DecisionReceipt = surface.receipt.DecisionReceipt
    if isinstance(receipt, DecisionReceipt):
        return receipt
    if isinstance(receipt, Mapping):
        return DecisionReceipt.from_dict(receipt)
    raise AdmissibilityEnforcementError(
        "receipt must be a DecisionReceipt or mapping; fail closed"
    )


def _coerce_capability(surface: _DatasetsAuthSurface, capability: Any) -> Any:
    AuthorizationCapability = surface.receipt.AuthorizationCapability
    if isinstance(capability, AuthorizationCapability):
        return capability
    if isinstance(capability, Mapping):
        return AuthorizationCapability.from_dict(capability)
    raise AdmissibilityEnforcementError(
        "capability must be an AuthorizationCapability or mapping; fail closed"
    )


def _normalize_digest(value: str) -> str:
    text = value.strip()
    if text.startswith("sha256:"):
        return text[len("sha256:") :]
    return text


def _effects_match(expected: Sequence[str], actual: Sequence[str]) -> bool:
    return tuple(sorted(expected)) == tuple(sorted(actual))


def _delegation_match(
    context: SupervisorInvocationContext, receipt_ctx: Any
) -> bool:
    receipt_ids = tuple(getattr(receipt_ctx, "delegation_ids", ()) or ())
    if context.delegation_ids and tuple(context.delegation_ids) != tuple(receipt_ids):
        return False
    receipt_digest = str(getattr(receipt_ctx, "delegation_digest", "") or "")
    if context.delegation_digest and context.delegation_digest != receipt_digest:
        return False
    return True


@dataclass
class SupervisorPreInvocationEnforcement:
    """SupervisorPreInvocationEnforcement@1 — lazy, fail-closed pre-dispatch gate.

    Parameters
    ----------
    mode:
        Explicit ``off`` / ``audit`` / ``shadow`` / ``enforce``.
    store:
        Injected :class:`CapabilityConsumptionStore` (defaults to in-memory).
    service:
        Optional injected authorization service used when no receipt is
        supplied and evaluation is required.
    expected_roots:
        Pinned policy/corpus/revocation roots revalidated on every enforce /
        shadow / audit check (``BoundRoots`` or mapping).
    clock:
        Optional ``() -> ISO-8601`` callable for expiry checks.
    """

    mode: EnforcementMode | str = EnforcementMode.OFF
    store: CapabilityConsumptionStore | None = None
    service: Any | None = None
    expected_roots: Any | None = None
    clock: Callable[[], str] | None = None
    _store: CapabilityConsumptionStore = field(init=False, repr=False)
    _mode: EnforcementMode = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_mode", _parse_mode(self.mode))
        object.__setattr__(self, "mode", self._mode)
        if self.store is None:
            object.__setattr__(self, "_store", InMemoryCapabilityConsumptionStore())
        else:
            object.__setattr__(self, "_store", self.store)

    # -- interface -----------------------------------------------------------

    @property
    def interface(self) -> str:
        return SUPERVISOR_PRE_INVOCATION_ENFORCEMENT_INTERFACE

    @property
    def schema(self) -> str:
        return SUPERVISOR_PRE_INVOCATION_ENFORCEMENT_SCHEMA

    @property
    def version(self) -> int:
        return SUPERVISOR_PRE_INVOCATION_ENFORCEMENT_VERSION

    @property
    def active_mode(self) -> EnforcementMode:
        return self._mode

    @property
    def consumption_store(self) -> CapabilityConsumptionStore:
        return self._store

    # -- factories -----------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        store: CapabilityConsumptionStore | None = None,
        service: Any | None = None,
        expected_roots: Any | None = None,
        clock: Callable[[], str] | None = None,
    ) -> "SupervisorPreInvocationEnforcement":
        raw = os.environ.get(ENV_ENFORCEMENT_MODE, EnforcementMode.OFF.value)
        return cls(
            mode=raw,
            store=store,
            service=service,
            expected_roots=expected_roots,
            clock=clock,
        )

    # -- capabilities / introspection ----------------------------------------

    def capabilities(self) -> dict[str, Any]:
        """Report surface without evaluating receipts or loading provers."""

        return {
            "interface": self.interface,
            "schema": self.schema,
            "version": self.version,
            "mode": self._mode.value,
            "modes": [item.value for item in EnforcementMode],
            "datasets_auth_available": datasets_auth_available(),
            "store_injected": self.store is not None,
            "service_injected": self.service is not None,
            "expected_roots_pinned": self.expected_roots is not None,
            "executed": False,
            "provers_imported": False,
            "theorem_proof": False,
            "env_flags": {"mode": ENV_ENFORCEMENT_MODE},
            "lazy_import_targets": [
                _DATASETS_RECEIPT_MODULE,
                _DATASETS_SERVICE_MODULE,
            ],
            "banned_heavy_import_markers": list(_BANNED_HEAVY_IMPORT_MARKERS),
        }

    # -- verification helpers ------------------------------------------------

    def _now(self) -> str | None:
        if self.clock is None:
            return None
        return _text(self.clock(), "clock()")

    def _verify_against_context(
        self,
        surface: _DatasetsAuthSurface,
        *,
        context: SupervisorInvocationContext,
        receipt: Any,
        capability: Any | None,
    ) -> tuple[str, ...]:
        """Return denial reason codes (empty when verification succeeds)."""

        reasons: list[str] = []
        verify_decision_receipt = surface.receipt.verify_decision_receipt
        verify_capability = surface.receipt.verify_capability
        ReceiptVerificationError = surface.receipt.ReceiptVerificationError
        InternalDecisionStatus = surface.compose.InternalDecisionStatus
        AdmissibilityStatus = surface.reasons.AdmissibilityStatus

        now = self._now()
        expected_roots = self.expected_roots

        try:
            receipt = verify_decision_receipt(
                receipt,
                now=now,
                expected_roots=expected_roots,
                expected_audience=context.audience_id,
                expected_request_digest=context.request_digest,
                expected_actor=context.actor_id,
                expected_nonce=context.nonce or None,
                require_not_expired=now is not None,
            )
        except ReceiptVerificationError as exc:
            message = str(exc).lower()
            if "expir" in message or "deadline" in message:
                reasons.append(EnforcementDenialReason.EXPIRED.value)
            elif "root" in message or "stale" in message:
                reasons.append(EnforcementDenialReason.ROOT_CHANGED.value)
            elif "audience" in message or "actor" in message or "request" in message:
                reasons.append(EnforcementDenialReason.CONTEXT_MISMATCH.value)
            else:
                reasons.append(EnforcementDenialReason.ERROR.value)
            return tuple(reasons)
        except Exception as exc:  # noqa: BLE001
            reasons.append(EnforcementDenialReason.ERROR.value)
            reasons.append(f"verify:{type(exc).__name__}")
            return tuple(reasons)

        # Outcome gate: only exact allow may proceed under enforce.
        outcome = getattr(receipt, "outcome", None)
        wire = getattr(receipt, "wire_status", None)
        if outcome is not InternalDecisionStatus.ALLOW:
            if outcome is InternalDecisionStatus.DENY or (
                wire is AdmissibilityStatus.REJECT
            ):
                reasons.append(EnforcementDenialReason.REJECT.value)
            elif wire is AdmissibilityStatus.ABSTAIN or str(
                getattr(outcome, "value", outcome)
            ) in {"abstain", "unknown", "error", "cancelled"}:
                # Map non-allow non-deny to abstain/error as appropriate.
                status_value = str(getattr(outcome, "value", outcome) or "")
                if status_value in {"error", "cancelled"}:
                    reasons.append(EnforcementDenialReason.ERROR.value)
                else:
                    reasons.append(EnforcementDenialReason.ABSTAIN.value)
            else:
                reasons.append(EnforcementDenialReason.NOT_ALLOW.value)
            return tuple(reasons)
        if wire is not AdmissibilityStatus.ALLOW:
            reasons.append(EnforcementDenialReason.NOT_ALLOW.value)
            return tuple(reasons)

        receipt_ctx = receipt.context
        if context.tool_id and context.tool_id != str(
            getattr(receipt_ctx, "tool_id", "") or ""
        ):
            reasons.append(EnforcementDenialReason.CONTEXT_MISMATCH.value)
        if context.tool_version and context.tool_version != str(
            getattr(receipt_ctx, "tool_version", "") or ""
        ):
            reasons.append(EnforcementDenialReason.CONTEXT_MISMATCH.value)
        if context.arguments_digest and _normalize_digest(
            context.arguments_digest
        ) != _normalize_digest(str(getattr(receipt_ctx, "arguments_digest", "") or "")):
            reasons.append(EnforcementDenialReason.CONTEXT_MISMATCH.value)
        if context.effect_ids and not _effects_match(
            context.effect_ids, getattr(receipt_ctx, "effect_ids", ()) or ()
        ):
            # Allow capability attenuation: context effects must be a subset of
            # receipt effects when both declare effects.
            receipt_effects = set(getattr(receipt_ctx, "effect_ids", ()) or ())
            if not set(context.effect_ids).issubset(receipt_effects):
                reasons.append(EnforcementDenialReason.CONTEXT_MISMATCH.value)
        if not _delegation_match(context, receipt_ctx):
            reasons.append(EnforcementDenialReason.CONTEXT_MISMATCH.value)

        # Environment binding (TOCTOU-sensitive).
        receipt_env = _normalize_digest(
            str(getattr(receipt_ctx, "environment_digest", "") or "")
        )
        if context.environment_digest and receipt_env and (
            _normalize_digest(context.environment_digest) != receipt_env
        ):
            reasons.append(EnforcementDenialReason.ENVIRONMENT_CHANGED.value)
        if context.environment_id and str(
            getattr(receipt_ctx, "environment_id", "") or ""
        ) not in {"", context.environment_id}:
            reasons.append(EnforcementDenialReason.ENVIRONMENT_CHANGED.value)

        # Root revalidation when expected roots were supplied at construction.
        if expected_roots is not None:
            try:
                BoundRoots = surface.receipt.BoundRoots
                if isinstance(expected_roots, Mapping):
                    pinned = BoundRoots.from_dict(expected_roots)
                elif isinstance(expected_roots, BoundRoots):
                    pinned = expected_roots
                else:
                    pinned = expected_roots
                if not receipt.roots.matches(pinned):
                    reasons.append(EnforcementDenialReason.ROOT_CHANGED.value)
            except Exception:  # noqa: BLE001
                reasons.append(EnforcementDenialReason.ROOT_CHANGED.value)

        if capability is not None:
            try:
                verify_capability(
                    capability,
                    receipt,
                    now=now,
                    expected_audience=context.audience_id,
                    expected_roots=expected_roots,
                    expected_request_digest=context.request_digest,
                    require_not_expired=now is not None,
                )
            except ReceiptVerificationError as exc:
                message = str(exc).lower()
                if "expir" in message:
                    reasons.append(EnforcementDenialReason.EXPIRED.value)
                elif "root" in message or "stale" in message:
                    reasons.append(EnforcementDenialReason.ROOT_CHANGED.value)
                elif "one-time" in message or "replay" in message:
                    reasons.append(EnforcementDenialReason.REPLAYED.value)
                else:
                    reasons.append(EnforcementDenialReason.ERROR.value)
            except Exception:  # noqa: BLE001
                reasons.append(EnforcementDenialReason.ERROR.value)

        # Deduplicate while preserving order.
        seen: set[str] = set()
        ordered: list[str] = []
        for item in reasons:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return tuple(ordered)

    def _resolve_receipt_capability(
        self,
        surface: _DatasetsAuthSurface,
        *,
        context: SupervisorInvocationContext,
        receipt: Any | None,
        capability: Any | None,
    ) -> tuple[Any | None, Any | None, tuple[str, ...]]:
        if receipt is None and self.service is not None:
            try:
                if hasattr(self.service, "evaluate"):
                    result = self.service.evaluate(context=context)
                elif callable(self.service):
                    result = self.service(context)
                else:
                    return None, None, (EnforcementDenialReason.SERVICE_UNAVAILABLE.value,)
                if isinstance(result, tuple) and len(result) == 2:
                    receipt, capability = result
                elif hasattr(result, "receipt"):
                    receipt = result.receipt
                    capability = getattr(result, "capability", capability)
                else:
                    receipt = result
            except Exception:  # noqa: BLE001
                return None, None, (EnforcementDenialReason.SERVICE_UNAVAILABLE.value,)

        if receipt is None:
            return None, None, (EnforcementDenialReason.MISSING_RECEIPT.value,)

        try:
            receipt = _coerce_receipt(surface, receipt)
        except AdmissibilityEnforcementError:
            return None, None, (EnforcementDenialReason.ERROR.value,)

        if capability is None and getattr(receipt, "permits_capability_derivation", False):
            # Derive a one-time capability from the allow receipt for consumption.
            try:
                derive_capability = surface.receipt.derive_capability
                effects = context.effect_ids or None
                capability = derive_capability(
                    receipt,
                    capability_id=(
                        f"capability:supervisor:{context.task_id or 'anon'}:"
                        f"{receipt.receipt_id}"
                    ),
                    allowed_effects=effects,
                    audience_id=context.audience_id,
                    tool_id=context.tool_id or None,
                    require_strict_subset=bool(
                        getattr(receipt, "effect_ids", ())
                        and len(getattr(receipt, "effect_ids", ())) > 1
                        and effects
                        and len(effects) < len(getattr(receipt, "effect_ids", ()))
                    ),
                )
            except Exception:  # noqa: BLE001 — leave capability unset
                capability = None

        if capability is not None:
            try:
                capability = _coerce_capability(surface, capability)
            except AdmissibilityEnforcementError:
                return receipt, None, (EnforcementDenialReason.ERROR.value,)

        return receipt, capability, ()

    def _observation(
        self,
        *,
        disposition: EnforcementDisposition,
        allowed: bool,
        delegated: bool,
        reason_codes: Sequence[str],
        context: SupervisorInvocationContext | None = None,
        receipt: Any | None = None,
        capability: Any | None = None,
        error: str = "",
        denial_reason: str = "",
        consumed: bool = False,
    ) -> EnforcementObservation:
        return EnforcementObservation(
            disposition=disposition,
            mode=self._mode,
            allowed=allowed,
            delegated=delegated,
            reason_codes=tuple(reason_codes),
            receipt_id=str(getattr(receipt, "receipt_id", "") or ""),
            capability_id=str(getattr(capability, "capability_id", "") or ""),
            nonce=str(
                getattr(capability, "nonce", None)
                or getattr(receipt, "nonce", None)
                or (context.nonce if context is not None else "")
                or ""
            ),
            actor_id=context.actor_id if context is not None else "",
            audience_id=context.audience_id if context is not None else "",
            task_id=context.task_id if context is not None else "",
            plan_id=context.plan_id if context is not None else "",
            tool_id=context.tool_id if context is not None else "",
            error=error,
            denial_reason=denial_reason,
            consumed=consumed,
            theorem_proof=False,
        )

    # -- main entry points ---------------------------------------------------

    def authorize_and_delegate(
        self,
        context: SupervisorInvocationContext | Mapping[str, Any],
        delegate: Callable[[], Any],
        *,
        receipt: Any | None = None,
        capability: Any | None = None,
    ) -> EnforcementResult:
        """Revalidate context-bound receipt, consume once, then call *delegate*.

        Mode behaviour
        --------------
        * ``off`` — call *delegate* once without receipt verification or
          consumption (compatibility path; observation records mode off).
        * ``audit`` — verify and record; always call *delegate* (non-blocking).
        * ``shadow`` — verify and record would-block/allow; always call
          *delegate* without consuming the one-time token.
        * ``enforce`` — verify, atomically consume, call *delegate* **once**
          only on success; otherwise deny with zero calls.
        """

        if not callable(delegate):
            raise AdmissibilityEnforcementError(
                "delegate must be callable; fail closed"
            )
        if isinstance(context, Mapping):
            context = SupervisorInvocationContext.from_dict(context)
        elif not isinstance(context, SupervisorInvocationContext):
            raise AdmissibilityEnforcementError(
                "context must be SupervisorInvocationContext or mapping; fail closed"
            )

        # OFF: pass-through, single call, no auth surface import required.
        if self._mode is EnforcementMode.OFF:
            result = delegate()
            observation = self._observation(
                disposition=EnforcementDisposition.OFF,
                allowed=True,
                delegated=True,
                reason_codes=("mode.off",),
                context=context,
                receipt=receipt,
                capability=capability,
            )
            return EnforcementResult(
                observation=observation,
                delegate_result=result,
                delegate_called=True,
                receipt=receipt,
                capability=capability,
            )

        try:
            surface = _load_datasets_auth_surface()
        except AdmissibilityEnforcementError as exc:
            # Fail closed under enforce; audit/shadow still allow dispatch with
            # an error observation so operators can detect backend outage.
            if self._mode is EnforcementMode.ENFORCE:
                observation = self._observation(
                    disposition=EnforcementDisposition.DENIED,
                    allowed=False,
                    delegated=False,
                    reason_codes=(EnforcementDenialReason.SERVICE_UNAVAILABLE.value,),
                    context=context,
                    error=str(exc),
                    denial_reason=EnforcementDenialReason.SERVICE_UNAVAILABLE.value,
                )
                return EnforcementResult(
                    observation=observation,
                    delegate_called=False,
                )
            result = delegate()
            disposition = (
                EnforcementDisposition.AUDITED
                if self._mode is EnforcementMode.AUDIT
                else EnforcementDisposition.SHADOW_WOULD_BLOCK
            )
            observation = self._observation(
                disposition=disposition,
                allowed=True,
                delegated=True,
                reason_codes=(EnforcementDenialReason.SERVICE_UNAVAILABLE.value,),
                context=context,
                error=str(exc),
                denial_reason=EnforcementDenialReason.SERVICE_UNAVAILABLE.value,
            )
            return EnforcementResult(
                observation=observation,
                delegate_result=result,
                delegate_called=True,
            )

        resolved_receipt, resolved_capability, resolve_errors = (
            self._resolve_receipt_capability(
                surface,
                context=context,
                receipt=receipt,
                capability=capability,
            )
        )
        denial_reasons: list[str] = list(resolve_errors)
        if not denial_reasons and resolved_receipt is not None:
            denial_reasons.extend(
                self._verify_against_context(
                    surface,
                    context=context,
                    receipt=resolved_receipt,
                    capability=resolved_capability,
                )
            )

        would_allow = not denial_reasons
        primary_denial = denial_reasons[0] if denial_reasons else ""

        # AUDIT / SHADOW: never block, never consume.
        if self._mode is EnforcementMode.AUDIT:
            result = delegate()
            observation = self._observation(
                disposition=EnforcementDisposition.AUDITED,
                allowed=True,
                delegated=True,
                reason_codes=tuple(denial_reasons) or ("audit.pass",),
                context=context,
                receipt=resolved_receipt,
                capability=resolved_capability,
                denial_reason=primary_denial,
            )
            return EnforcementResult(
                observation=observation,
                delegate_result=result,
                delegate_called=True,
                receipt=resolved_receipt,
                capability=resolved_capability,
            )

        if self._mode is EnforcementMode.SHADOW:
            result = delegate()
            disposition = (
                EnforcementDisposition.SHADOW_ALLOWED
                if would_allow
                else EnforcementDisposition.SHADOW_WOULD_BLOCK
            )
            observation = self._observation(
                disposition=disposition,
                allowed=True,
                delegated=True,
                reason_codes=tuple(denial_reasons)
                or ("shadow.allow",),
                context=context,
                receipt=resolved_receipt,
                capability=resolved_capability,
                denial_reason=primary_denial,
            )
            return EnforcementResult(
                observation=observation,
                delegate_result=result,
                delegate_called=True,
                receipt=resolved_receipt,
                capability=resolved_capability,
            )

        # ENFORCE: fail closed, consume once, call once.
        assert self._mode is EnforcementMode.ENFORCE
        if not would_allow:
            observation = self._observation(
                disposition=EnforcementDisposition.DENIED,
                allowed=False,
                delegated=False,
                reason_codes=tuple(denial_reasons),
                context=context,
                receipt=resolved_receipt,
                capability=resolved_capability,
                denial_reason=primary_denial,
            )
            return EnforcementResult(
                observation=observation,
                delegate_called=False,
                receipt=resolved_receipt,
                capability=resolved_capability,
            )

        try:
            token_key = _consumption_key(
                capability=resolved_capability,
                receipt=resolved_receipt,
                context=context,
            )
        except AdmissibilityEnforcementError as exc:
            observation = self._observation(
                disposition=EnforcementDisposition.DENIED,
                allowed=False,
                delegated=False,
                reason_codes=(EnforcementDenialReason.ERROR.value,),
                context=context,
                receipt=resolved_receipt,
                capability=resolved_capability,
                error=str(exc),
                denial_reason=EnforcementDenialReason.ERROR.value,
            )
            return EnforcementResult(
                observation=observation,
                delegate_called=False,
                receipt=resolved_receipt,
                capability=resolved_capability,
            )

        consumed = self._store.try_consume(
            token_key,
            meta={
                "receipt_id": str(
                    getattr(resolved_receipt, "receipt_id", "") or ""
                ),
                "capability_id": str(
                    getattr(resolved_capability, "capability_id", "") or ""
                ),
                "actor_id": context.actor_id,
                "audience_id": context.audience_id,
                "task_id": context.task_id,
                "tool_id": context.tool_id,
            },
        )
        if not consumed:
            observation = self._observation(
                disposition=EnforcementDisposition.DENIED,
                allowed=False,
                delegated=False,
                reason_codes=(EnforcementDenialReason.REPLAYED.value,),
                context=context,
                receipt=resolved_receipt,
                capability=resolved_capability,
                denial_reason=EnforcementDenialReason.REPLAYED.value,
                consumed=False,
            )
            return EnforcementResult(
                observation=observation,
                delegate_called=False,
                receipt=resolved_receipt,
                capability=resolved_capability,
            )

        # Atomic consumption succeeded → single delegate invocation.
        result = delegate()
        observation = self._observation(
            disposition=EnforcementDisposition.ALLOWED,
            allowed=True,
            delegated=True,
            reason_codes=("enforce.allow", "consumed"),
            context=context,
            receipt=resolved_receipt,
            capability=resolved_capability,
            consumed=True,
        )
        return EnforcementResult(
            observation=observation,
            delegate_result=result,
            delegate_called=True,
            receipt=resolved_receipt,
            capability=resolved_capability,
        )

    def dispatch(
        self,
        context: SupervisorInvocationContext | Mapping[str, Any],
        delegate: Callable[[], Any],
        *,
        receipt: Any | None = None,
        capability: Any | None = None,
    ) -> EnforcementResult:
        """Alias for :meth:`authorize_and_delegate` (supervisor vocabulary)."""

        return self.authorize_and_delegate(
            context,
            delegate,
            receipt=receipt,
            capability=capability,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def create_supervisor_enforcement(
    *,
    mode: EnforcementMode | str = EnforcementMode.OFF,
    store: CapabilityConsumptionStore | None = None,
    service: Any | None = None,
    expected_roots: Any | None = None,
    clock: Callable[[], str] | None = None,
) -> SupervisorPreInvocationEnforcement:
    """Factory for an explicit enforcement configuration."""

    return SupervisorPreInvocationEnforcement(
        mode=mode,
        store=store,
        service=service,
        expected_roots=expected_roots,
        clock=clock,
    )


def authorize_and_delegate(
    context: SupervisorInvocationContext | Mapping[str, Any],
    delegate: Callable[[], Any],
    *,
    mode: EnforcementMode | str = EnforcementMode.ENFORCE,
    store: CapabilityConsumptionStore | None = None,
    service: Any | None = None,
    expected_roots: Any | None = None,
    clock: Callable[[], str] | None = None,
    receipt: Any | None = None,
    capability: Any | None = None,
) -> EnforcementResult:
    """One-shot helper: construct enforcement and authorize a single delegate."""

    enforcer = create_supervisor_enforcement(
        mode=mode,
        store=store,
        service=service,
        expected_roots=expected_roots,
        clock=clock,
    )
    return enforcer.authorize_and_delegate(
        context,
        delegate,
        receipt=receipt,
        capability=capability,
    )


__all__ = [
    "ENV_ENFORCEMENT_MODE",
    "SUPERVISOR_ENFORCEMENT_OBSERVATION_SCHEMA",
    "SUPERVISOR_PRE_INVOCATION_ENFORCEMENT_INTERFACE",
    "SUPERVISOR_PRE_INVOCATION_ENFORCEMENT_SCHEMA",
    "SUPERVISOR_PRE_INVOCATION_ENFORCEMENT_VERSION",
    "AdmissibilityEnforcementError",
    "AuthorizationServiceLike",
    "CapabilityConsumptionStore",
    "EnforcementDenialReason",
    "EnforcementDisposition",
    "EnforcementMode",
    "EnforcementObservation",
    "EnforcementResult",
    "InMemoryCapabilityConsumptionStore",
    "SupervisorInvocationContext",
    "SupervisorPreInvocationEnforcement",
    "authorize_and_delegate",
    "create_supervisor_enforcement",
    "datasets_auth_available",
    "reset_datasets_auth_surface_cache",
]
