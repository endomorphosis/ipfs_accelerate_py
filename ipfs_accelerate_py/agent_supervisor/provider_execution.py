"""Reservation-aware supervisor provider execution gateway (ASI-166).

Every supervisor provider call that charges endpoint usage must go through this
gateway: estimate → exact reserve → invoke → settle/release. The gateway is
operational only; it never becomes proof or completion authority.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, MutableMapping, Optional, Protocol, Sequence

from ipfs_accelerate_py.agent_supervisor.provider_usage import (
    SupervisorToEndpointRequest,
    SupervisorUsageEnvelope,
    SupervisorUsageFinalStatus,
    SupervisorUsageReceipt,
)
from ipfs_accelerate_py.endpoint_usage import UsageVector
from ipfs_accelerate_py.endpoint_usage.coordinator import (
    ReserveDecision,
    SettlementResult,
    UsageCoordinator,
)


PROVIDER_EXECUTION_CONTRACT_VERSION = "supervisor-provider-execution/v1"
PROVIDER_EXECUTION_REQUEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/provider-execution-request@1"
)
PROVIDER_EXECUTION_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/provider-execution-result@1"
)


class ProviderExecutionError(RuntimeError):
    """Fail-closed gateway error with stable reason codes."""

    def __init__(
        self,
        message: str,
        *,
        reason_codes: Sequence[str] = (),
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.reason_codes = tuple(str(code) for code in reason_codes if str(code))
        self.retryable = bool(retryable)


class ProviderExecutionPhase(str, Enum):
    ACCEPTED = "accepted"
    RESERVED = "reserved"
    DISPATCHED = "dispatched"
    SETTLED = "settled"
    RELEASED = "released"
    DENIED = "denied"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ProviderInvoker(Protocol):
    """Typed adapter that performs the provider side-effect."""

    def __call__(self, request: "ProviderExecutionRequest") -> Mapping[str, Any]:
        ...


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _content_id(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ProviderExecutionError(
            f"{name} is required",
            reason_codes=("missing_field", name),
        )
    return text


@dataclass(frozen=True)
class ProviderExecutionRequest:
    """Exact, idempotent supervisor provider execution request."""

    bridge: SupervisorToEndpointRequest
    envelope: SupervisorUsageEnvelope
    provider_id: str
    modality: str
    side_effect: str
    expected_output_kind: str
    cancelled: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _text(self.provider_id, "provider_id"))
        object.__setattr__(self, "modality", _text(self.modality, "modality"))
        object.__setattr__(self, "side_effect", _text(self.side_effect, "side_effect"))
        object.__setattr__(
            self,
            "expected_output_kind",
            _text(self.expected_output_kind, "expected_output_kind"),
        )
        if not isinstance(self.bridge, SupervisorToEndpointRequest):
            raise ProviderExecutionError(
                "bridge must be SupervisorToEndpointRequest",
                reason_codes=("invalid_bridge",),
            )
        if not isinstance(self.envelope, SupervisorUsageEnvelope):
            raise ProviderExecutionError(
                "envelope must be SupervisorUsageEnvelope",
                reason_codes=("invalid_envelope",),
            )
        if self.envelope.envelope_id != self.bridge.envelope_id:
            raise ProviderExecutionError(
                "envelope_id foreign to bridge request",
                reason_codes=("envelope_mismatch",),
            )
        if self.envelope.scope.request_id != self.bridge.request_id:
            raise ProviderExecutionError(
                "request_id foreign to envelope scope",
                reason_codes=("scope_mismatch",),
            )
        meta = self.metadata if isinstance(self.metadata, Mapping) else {}
        object.__setattr__(self, "metadata", dict(meta))
        object.__setattr__(self, "cancelled", bool(self.cancelled))

    @property
    def request_key(self) -> str:
        return self.bridge.idempotency_key

    @property
    def attempt_key(self) -> str:
        return f"{self.bridge.request_id}#{self.bridge.attempt}"

    def to_record(self) -> dict[str, Any]:
        payload = {
            "schema": PROVIDER_EXECUTION_REQUEST_SCHEMA,
            "contract_version": PROVIDER_EXECUTION_CONTRACT_VERSION,
            "bridge": self.bridge.to_record(),
            "envelope_id": self.envelope.envelope_id,
            "provider_id": self.provider_id,
            "modality": self.modality,
            "side_effect": self.side_effect,
            "expected_output_kind": self.expected_output_kind,
            "cancelled": self.cancelled,
            "metadata": dict(self.metadata),
            "request_key": self.request_key,
            "attempt_key": self.attempt_key,
        }
        payload["content_id"] = _content_id(payload)
        return payload


@dataclass(frozen=True)
class ProviderExecutionResult:
    """Normalized gateway outcome with operational receipt only."""

    schema: str = PROVIDER_EXECUTION_RESULT_SCHEMA
    phase: ProviderExecutionPhase = ProviderExecutionPhase.FAILED
    granted: bool = False
    reservation_id: str = ""
    usage_revision: str = ""
    provider_id: str = ""
    redacted_endpoint: str = ""
    attempt_key: str = ""
    request_key: str = ""
    reason_codes: tuple[str, ...] = ()
    observation: Mapping[str, Any] = field(default_factory=dict)
    receipt: Optional[SupervisorUsageReceipt] = None
    replayed: bool = False
    is_completion_evidence: bool = False
    is_correctness_evidence: bool = False

    def __post_init__(self) -> None:
        if self.is_completion_evidence or self.is_correctness_evidence:
            raise ProviderExecutionError(
                "provider execution results cannot claim proof/completion authority",
                reason_codes=("authority_boundary",),
            )
        object.__setattr__(self, "phase", ProviderExecutionPhase(self.phase))
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(
            self,
            "observation",
            dict(self.observation) if isinstance(self.observation, Mapping) else {},
        )

    def to_record(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "contract_version": PROVIDER_EXECUTION_CONTRACT_VERSION,
            "phase": self.phase.value,
            "granted": self.granted,
            "reservation_id": self.reservation_id,
            "usage_revision": self.usage_revision,
            "provider_id": self.provider_id,
            "redacted_endpoint": self.redacted_endpoint,
            "attempt_key": self.attempt_key,
            "request_key": self.request_key,
            "reason_codes": list(self.reason_codes),
            "observation": dict(self.observation),
            "receipt": self.receipt.to_record() if self.receipt is not None else None,
            "replayed": self.replayed,
            "is_completion_evidence": False,
            "is_correctness_evidence": False,
        }
        payload["content_id"] = _content_id(payload)
        return payload


@dataclass
class _TerminalRecord:
    result: ProviderExecutionResult
    finished_at: float


class ProviderExecutionGateway:
    """Atomic reserve → invoke → settle gateway with exact replay protection."""

    def __init__(
        self,
        *,
        coordinator: Optional[UsageCoordinator] = None,
        invoker: Optional[ProviderInvoker] = None,
        owner_id: str = "supervisor-provider-execution",
        reservation_ttl_ms: int = 60_000,
    ) -> None:
        self._coordinator = coordinator
        self._invoker = invoker
        self._owner_id = _text(owner_id, "owner_id")
        self._reservation_ttl_ms = max(1, int(reservation_ttl_ms))
        self._lock = threading.RLock()
        self._terminals: MutableMapping[str, _TerminalRecord] = {}
        self._in_flight: set[str] = set()

    def execute(self, request: ProviderExecutionRequest) -> ProviderExecutionResult:
        if not isinstance(request, ProviderExecutionRequest):
            raise ProviderExecutionError(
                "request must be ProviderExecutionRequest",
                reason_codes=("invalid_request",),
            )
        attempt_key = request.attempt_key
        with self._lock:
            prior = self._terminals.get(attempt_key)
            if prior is not None:
                replayed = ProviderExecutionResult(
                    phase=prior.result.phase,
                    granted=prior.result.granted,
                    reservation_id=prior.result.reservation_id,
                    usage_revision=prior.result.usage_revision,
                    provider_id=prior.result.provider_id,
                    redacted_endpoint=prior.result.redacted_endpoint,
                    attempt_key=prior.result.attempt_key,
                    request_key=prior.result.request_key,
                    reason_codes=prior.result.reason_codes + ("exact_replay",),
                    observation=prior.result.observation,
                    receipt=prior.result.receipt,
                    replayed=True,
                )
                return replayed
            if attempt_key in self._in_flight:
                raise ProviderExecutionError(
                    "attempt already in flight",
                    reason_codes=("attempt_in_flight",),
                    retryable=True,
                )
            self._in_flight.add(attempt_key)

        try:
            if request.cancelled:
                result = self._terminal(
                    request,
                    phase=ProviderExecutionPhase.CANCELLED,
                    granted=False,
                    reason_codes=("pre_dispatch_cancelled",),
                )
                return result

            decision = self._reserve(request)
            if not decision["granted"]:
                result = self._terminal(
                    request,
                    phase=ProviderExecutionPhase.DENIED,
                    granted=False,
                    reservation_id=str(decision.get("reservation_id") or ""),
                    usage_revision=str(decision.get("usage_revision") or ""),
                    reason_codes=tuple(
                        decision.get("reason_codes") or ("capacity_denied",)
                    ),
                )
                return result

            reservation_id = str(decision.get("reservation_id") or "")
            usage_revision = str(decision.get("usage_revision") or "")
            if (
                self._coordinator is not None
                and reservation_id
                and not reservation_id.startswith("sim:")
            ):
                self._coordinator.mark_dispatched(reservation_id)

            try:
                observation = self._invoke(request)
            except Exception as exc:
                if (
                    self._coordinator is not None
                    and reservation_id
                    and not reservation_id.startswith("sim:")
                ):
                    self._coordinator.cancel(reservation_id, reason="invoke_failed")
                result = self._terminal(
                    request,
                    phase=ProviderExecutionPhase.FAILED,
                    granted=True,
                    reservation_id=reservation_id,
                    usage_revision=usage_revision,
                    reason_codes=("invoke_failed", type(exc).__name__),
                    observation={"error": str(exc)[:500]},
                )
                return result

            settled_units = self._observation_units(observation, request)
            receipt = self._settle(
                request,
                reservation_id=reservation_id,
                units=settled_units,
                final_status=SupervisorUsageFinalStatus.COMMITTED,
            )

            redacted = self._redact_endpoint(
                str(observation.get("endpoint") or request.bridge.endpoint_scope_id)
            )
            result = self._terminal(
                request,
                phase=ProviderExecutionPhase.SETTLED,
                granted=True,
                reservation_id=reservation_id,
                usage_revision=usage_revision,
                reason_codes=("settled",),
                observation=dict(observation),
                receipt=receipt,
                redacted_endpoint=redacted,
            )
            return result
        finally:
            with self._lock:
                self._in_flight.discard(attempt_key)

    def _reserve(self, request: ProviderExecutionRequest) -> dict[str, Any]:
        if self._coordinator is None:
            # Simulation path for hermetic tests without a live ledger.
            return {
                "granted": True,
                "reservation_id": f"sim:{request.attempt_key}",
                "usage_revision": request.bridge.usage_revision,
                "reason_codes": ("simulated_reserve",),
            }
        try:
            decision: ReserveDecision = self._coordinator.reserve(
                request.bridge.endpoint_scope_id,
                request.bridge.estimated,
                request_id=request.bridge.request_id,
                attempt_id=str(request.bridge.attempt),
                idempotency_key=request.bridge.idempotency_key,
                owner_id=self._owner_id,
                lease_id=request.bridge.lease_id or None,
                expected_usage_revision=request.bridge.usage_revision,
                ttl_ms=self._reservation_ttl_ms,
            )
            return {
                "granted": bool(decision.granted),
                "reservation_id": str(decision.reservation_id or ""),
                "usage_revision": str(decision.usage_revision or ""),
                "reason_codes": tuple(decision.reason_codes or ()),
            }
        except Exception as exc:
            raise ProviderExecutionError(
                f"reserve failed: {exc}",
                reason_codes=("reserve_failed", type(exc).__name__),
                retryable=True,
            ) from exc

    def _invoke(self, request: ProviderExecutionRequest) -> dict[str, Any]:
        if self._invoker is None:
            return {
                "provider_id": request.provider_id,
                "endpoint": request.bridge.endpoint_scope_id,
                "status": "simulated_ok",
                "output_kind": request.expected_output_kind,
                "units": request.bridge.estimated.to_dict(),
            }
        raw = self._invoker(request)
        if not isinstance(raw, Mapping):
            raise ProviderExecutionError(
                "invoker must return a mapping observation",
                reason_codes=("invalid_observation",),
            )
        return dict(raw)

    def _observation_units(
        self,
        observation: Mapping[str, Any],
        request: ProviderExecutionRequest,
    ) -> UsageVector:
        units = observation.get("units")
        if units is None:
            return request.bridge.estimated
        if isinstance(units, UsageVector):
            return units
        return UsageVector.from_dict(units)

    def _settle(
        self,
        request: ProviderExecutionRequest,
        *,
        reservation_id: str,
        units: UsageVector,
        final_status: SupervisorUsageFinalStatus,
    ) -> Optional[SupervisorUsageReceipt]:
        event_ids: tuple[str, ...] = ()
        if self._coordinator is not None and reservation_id and not reservation_id.startswith("sim:"):
            try:
                settlement: SettlementResult = self._coordinator.commit(
                    reservation_id,
                    actual=units,
                )
                event_ids = (settlement.event_id,) if settlement.event_id else ()
                usage_revision = settlement.usage_revision
            except Exception:
                # Fall back to release so residual hold does not leak.
                try:
                    self._coordinator.release(reservation_id, reason="settle_fallback")
                except Exception:
                    pass
                usage_revision = request.bridge.usage_revision
                event_ids = ()
        else:
            usage_revision = request.bridge.usage_revision
            event_ids = (f"sim-event:{request.attempt_key}",)

        return SupervisorUsageReceipt(
            scope=request.bridge.scope,
            envelope_id=request.bridge.envelope_id,
            request_id=request.bridge.request_id,
            endpoint_scope_id=request.bridge.endpoint_scope_id,
            catalog_revision=request.bridge.catalog_revision,
            usage_revision=usage_revision or request.bridge.usage_revision,
            reservation_id=reservation_id or f"none:{request.attempt_key}",
            endpoint_event_ids=event_ids,
            settled=units,
            final_status=final_status,
        )

    def _terminal(
        self,
        request: ProviderExecutionRequest,
        *,
        phase: ProviderExecutionPhase,
        granted: bool,
        reason_codes: Sequence[str],
        reservation_id: str = "",
        usage_revision: str = "",
        observation: Optional[Mapping[str, Any]] = None,
        receipt: Optional[SupervisorUsageReceipt] = None,
        redacted_endpoint: str = "",
    ) -> ProviderExecutionResult:
        result = ProviderExecutionResult(
            phase=phase,
            granted=granted,
            reservation_id=reservation_id,
            usage_revision=usage_revision or request.bridge.usage_revision,
            provider_id=request.provider_id,
            redacted_endpoint=redacted_endpoint
            or self._redact_endpoint(request.bridge.endpoint_scope_id),
            attempt_key=request.attempt_key,
            request_key=request.request_key,
            reason_codes=tuple(reason_codes),
            observation=dict(observation or {}),
            receipt=receipt,
            replayed=False,
        )
        with self._lock:
            self._terminals[request.attempt_key] = _TerminalRecord(
                result=result,
                finished_at=time.time(),
            )
        return result

    @staticmethod
    def _redact_endpoint(endpoint: str) -> str:
        text = str(endpoint or "").strip()
        if not text:
            return "endpoint:redacted"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"endpoint:{digest}"


def new_attempt_idempotency_key(base_key: str, attempt: int) -> str:
    """Derive a fresh idempotency key for retry/fallback attempts."""

    base = _text(base_key, "base_key")
    return f"{base}#attempt-{int(attempt)}"


def build_execution_request(
    *,
    bridge: SupervisorToEndpointRequest,
    envelope: SupervisorUsageEnvelope,
    provider_id: str,
    modality: str = "text",
    side_effect: str = "generate_text",
    expected_output_kind: str = "text",
    cancelled: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ProviderExecutionRequest:
    return ProviderExecutionRequest(
        bridge=bridge,
        envelope=envelope,
        provider_id=provider_id,
        modality=modality,
        side_effect=side_effect,
        expected_output_kind=expected_output_kind,
        cancelled=cancelled,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "PROVIDER_EXECUTION_CONTRACT_VERSION",
    "PROVIDER_EXECUTION_REQUEST_SCHEMA",
    "PROVIDER_EXECUTION_RESULT_SCHEMA",
    "ProviderExecutionError",
    "ProviderExecutionGateway",
    "ProviderExecutionPhase",
    "ProviderExecutionRequest",
    "ProviderExecutionResult",
    "build_execution_request",
    "new_attempt_idempotency_key",
]
