"""Closed scheduling and execution contracts for the semantic-compression harness.

This module is a narrow projection over existing runtime resource, lease, and
provider contracts. It does not introduce a scheduler, queue authority, provider
registry, mock coordinator, or unbounded observation payload.

Scheduler observations never carry secrets, prompts, raw model output, or
repository source bodies. Terminal statuses distinguish unavailable, cancelled,
failed, and simulated outcomes. Scheduling success is never verification
success: only a separate, acceptance-eligible non-simulated verification
receipt may certify correctness.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    BOARD_NAMESPACE,
    HarnessError,
    HarnessMode,
    UnavailableResult,
    WorkKind,
    _bool,
    _closed,
    _enum,
    _nonneg_int,
    _optional_cid,
    _text,
    _unique_sorted_cids,
    _unique_sorted_texts,
)

SCHEDULING_CONTRACTS_SCHEMA = "semantic-state-scheduling@1"
SEMANTIC_WORK_SCHEDULING_INTERFACE = "SemanticWorkScheduling@1"

# Hard bounds for small scheduler observations and diagnostics.
_MAX_DIAGNOSTIC_CHARS = 512
_MAX_REASON_CODES = 32
_MAX_OBSERVATION_KEYS = 16
_MAX_TEXT_FIELD_CHARS = 256
_MAX_ARTIFACT_REFS = 64

# Keys that must never appear in scheduler observations.
_FORBIDDEN_OBSERVATION_KEYS = frozenset(
    {
        "secret",
        "secrets",
        "api_key",
        "api_keys",
        "password",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "credential",
        "credentials",
        "prompt",
        "prompts",
        "system_prompt",
        "user_prompt",
        "messages",
        "source_body",
        "source_text",
        "raw_source",
        "source_code",
        "file_body",
        "model_output",
        "raw_output",
        "raw_model_output",
        "completion",
        "completions",
        "response_text",
        "private_key",
    }
)

_SIMULATED_RESERVATION_PREFIXES = ("sim:", "degraded:")

_WORK_KIND_STAGE: Mapping[str, str] = {
    WorkKind.TASK_PARSING.value: "analysis",
    WorkKind.SCAN.value: "analysis",
    WorkKind.CAPSULE_COMPILATION.value: "analysis",
    WorkKind.TEST_SELECTION.value: "analysis",
    WorkKind.CONTEXT_PACKING.value: "analysis",
    WorkKind.MODEL_INVOCATION.value: "inference",
    WorkKind.STATIC_CHECK.value: "validation",
    WorkKind.PYTEST.value: "validation",
    WorkKind.PROVER.value: "proof",
    WorkKind.PERSISTENCE.value: "persistence",
}

_WORK_KIND_RESOURCE_CLASS: Mapping[str, str] = {
    WorkKind.TASK_PARSING.value: "cpu-small",
    WorkKind.SCAN.value: "cpu-medium",
    WorkKind.CAPSULE_COMPILATION.value: "cpu-medium",
    WorkKind.TEST_SELECTION.value: "cpu-small",
    WorkKind.CONTEXT_PACKING.value: "cpu-small",
    WorkKind.MODEL_INVOCATION.value: "llm-proof-draft",
    WorkKind.STATIC_CHECK.value: "cpu-validation",
    WorkKind.PYTEST.value: "cpu-validation",
    WorkKind.PROVER.value: "cpu-proof-solver",
    WorkKind.PERSISTENCE.value: "io-artifact",
}


class SemanticWorkStatus(str, Enum):
    """Terminal and admission outcomes for one scheduled semantic work unit.

    These are scheduling/execution dispositions only. None of them certify
    verification or root acceptance.
    """

    ADMITTED = "admitted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    SIMULATED = "simulated"


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 1:
        raise HarnessError(f"{name} must be a positive integer")
    return value


def _bounded_text(value: Any, name: str, *, max_chars: int) -> str:
    text = _text(value, name)
    if len(text) > max_chars:
        raise HarnessError(f"{name} must be at most {max_chars} characters")
    return text


def _optional_bounded_text(
    value: Any, name: str, *, max_chars: int
) -> str | None:
    if value is None:
        return None
    return _bounded_text(value, name, max_chars=max_chars)


def _bounded_reason_codes(values: Any, name: str = "reason_codes") -> tuple[str, ...]:
    ordered = _unique_sorted_texts(values, name)
    if len(ordered) > _MAX_REASON_CODES:
        raise HarnessError(f"{name} must contain at most {_MAX_REASON_CODES} entries")
    for item in ordered:
        if len(item) > _MAX_TEXT_FIELD_CHARS:
            raise HarnessError(
                f"{name} entries must be at most {_MAX_TEXT_FIELD_CHARS} characters"
            )
    return ordered


def _bounded_artifact_cids(values: Any, name: str) -> tuple[str, ...]:
    ordered = _unique_sorted_cids(values, name)
    if len(ordered) > _MAX_ARTIFACT_REFS:
        raise HarnessError(f"{name} must contain at most {_MAX_ARTIFACT_REFS} entries")
    return ordered


def _work_kind_value(value: Any) -> str:
    return _enum(value, WorkKind, "work_kind")


def _mode_value(value: Any) -> str:
    return _enum(value, HarnessMode, "mode")


def _status_value(value: Any) -> str:
    return _enum(value, SemanticWorkStatus, "status")


def stage_for_work_kind(work_kind: str | WorkKind) -> str:
    """Return the adaptive resource stage for a harness work kind."""

    kind = _work_kind_value(work_kind if isinstance(work_kind, str) else work_kind.value)
    return _WORK_KIND_STAGE[kind]


def resource_class_for_work_kind(work_kind: str | WorkKind) -> str:
    """Return the default resource class for a harness work kind."""

    kind = _work_kind_value(work_kind if isinstance(work_kind, str) else work_kind.value)
    return _WORK_KIND_RESOURCE_CLASS[kind]


def work_product_is_heuristic(work_kind: str | WorkKind) -> bool:
    """Optional model summaries are always heuristic work products."""

    kind = _work_kind_value(work_kind if isinstance(work_kind, str) else work_kind.value)
    return kind == WorkKind.MODEL_INVOCATION.value


def requires_provider(work_kind: str | WorkKind) -> bool:
    kind = _work_kind_value(work_kind if isinstance(work_kind, str) else work_kind.value)
    return kind == WorkKind.MODEL_INVOCATION.value


def _canonical_identity_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def compute_semantic_work_identity(
    *,
    work_kind: str | WorkKind,
    repository_id: str,
    attempt_id: str,
    input_artifact_cids: Sequence[str] = (),
    base_root_cid: str | None = None,
) -> str:
    """Return a deterministic, idempotent work identity for one attempt.

    The identity is a content digest of closed scheduling fields. It is not a
    Kubo CID and never embeds source bodies or secrets.
    """

    kind = _work_kind_value(work_kind if isinstance(work_kind, str) else work_kind.value)
    repo = _bounded_text(repository_id, "repository_id", max_chars=_MAX_TEXT_FIELD_CHARS)
    attempt = _bounded_text(attempt_id, "attempt_id", max_chars=_MAX_TEXT_FIELD_CHARS)
    inputs = _bounded_artifact_cids(list(input_artifact_cids), "input_artifact_cids")
    root = _optional_cid(base_root_cid, "base_root_cid")
    digest = hashlib.sha256(
        _canonical_identity_payload(
            {
                "schema": SCHEDULING_CONTRACTS_SCHEMA,
                "work_kind": kind,
                "repository_id": repo,
                "attempt_id": attempt,
                "input_artifact_cids": list(inputs),
                "base_root_cid": root,
            }
        )
    ).hexdigest()
    return f"sch-work:{digest}"


def _reject_simulated_reservation(reservation_id: str, *, mode: str) -> None:
    lowered = reservation_id.lower()
    if any(lowered.startswith(prefix) for prefix in _SIMULATED_RESERVATION_PREFIXES):
        if mode == HarnessMode.PRODUCTION.value:
            raise HarnessError(
                "production provider reservation must not use sim: or degraded: identity"
            )


@dataclass(frozen=True)
class ResourceBinding:
    """Projection of resource admission requirements for one work unit."""

    resource_class: str
    stage: str
    process_slots: int
    memory_bytes: int
    disk_bytes: int
    quota_units: int

    _FIELDS = frozenset(
        {
            "resource_class",
            "stage",
            "process_slots",
            "memory_bytes",
            "disk_bytes",
            "quota_units",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_class": self.resource_class,
            "stage": self.stage,
            "process_slots": self.process_slots,
            "memory_bytes": self.memory_bytes,
            "disk_bytes": self.disk_bytes,
            "quota_units": self.quota_units,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResourceBinding":
        payload = _closed(data, cls._FIELDS, "ResourceBinding")
        return cls(
            resource_class=_bounded_text(
                payload["resource_class"], "resource_class", max_chars=_MAX_TEXT_FIELD_CHARS
            ),
            stage=_bounded_text(payload["stage"], "stage", max_chars=_MAX_TEXT_FIELD_CHARS),
            process_slots=_positive_int(payload["process_slots"], "process_slots"),
            memory_bytes=_nonneg_int(payload["memory_bytes"], "memory_bytes"),
            disk_bytes=_nonneg_int(payload["disk_bytes"], "disk_bytes"),
            quota_units=_nonneg_int(payload["quota_units"], "quota_units"),
        )

    @classmethod
    def for_work_kind(
        cls,
        work_kind: str | WorkKind,
        *,
        process_slots: int = 1,
        memory_bytes: int = 0,
        disk_bytes: int = 0,
        quota_units: int = 1,
    ) -> "ResourceBinding":
        kind = _work_kind_value(work_kind if isinstance(work_kind, str) else work_kind.value)
        return cls(
            resource_class=resource_class_for_work_kind(kind),
            stage=stage_for_work_kind(kind),
            process_slots=process_slots,
            memory_bytes=memory_bytes,
            disk_bytes=disk_bytes,
            quota_units=quota_units,
        )


@dataclass(frozen=True)
class ProviderBinding:
    """Typed provider reservation projection for model-invocation work."""

    provider_id: str
    reservation_id: str
    mode: str
    simulated: bool

    _FIELDS = frozenset({"provider_id", "reservation_id", "mode", "simulated"})

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "reservation_id": self.reservation_id,
            "mode": self.mode,
            "simulated": self.simulated,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderBinding":
        payload = _closed(data, cls._FIELDS, "ProviderBinding")
        mode = _mode_value(payload["mode"])
        reservation_id = _bounded_text(
            payload["reservation_id"], "reservation_id", max_chars=_MAX_TEXT_FIELD_CHARS
        )
        simulated = _bool(payload["simulated"], "simulated")
        _reject_simulated_reservation(reservation_id, mode=mode)
        if mode == HarnessMode.PRODUCTION.value and simulated:
            raise HarnessError("production provider binding cannot be simulated")
        if simulated and not any(
            reservation_id.lower().startswith(prefix)
            for prefix in _SIMULATED_RESERVATION_PREFIXES
        ):
            # Development simulation must remain explicitly labeled.
            pass
        return cls(
            provider_id=_bounded_text(
                payload["provider_id"], "provider_id", max_chars=_MAX_TEXT_FIELD_CHARS
            ),
            reservation_id=reservation_id,
            mode=mode,
            simulated=simulated,
        )


@dataclass(frozen=True)
class LeaseBinding:
    """Attempt identity and fencing token for one admitted work unit."""

    attempt_id: str
    fencing_token: int
    lease_id: str
    logical_epoch: int

    _FIELDS = frozenset(
        {"attempt_id", "fencing_token", "lease_id", "logical_epoch"}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "fencing_token": self.fencing_token,
            "lease_id": self.lease_id,
            "logical_epoch": self.logical_epoch,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LeaseBinding":
        payload = _closed(data, cls._FIELDS, "LeaseBinding")
        return cls(
            attempt_id=_bounded_text(
                payload["attempt_id"], "attempt_id", max_chars=_MAX_TEXT_FIELD_CHARS
            ),
            fencing_token=_positive_int(payload["fencing_token"], "fencing_token"),
            lease_id=_bounded_text(
                payload["lease_id"], "lease_id", max_chars=_MAX_TEXT_FIELD_CHARS
            ),
            logical_epoch=_nonneg_int(payload["logical_epoch"], "logical_epoch"),
        )


class CancellationToken:
    """Thread-safe cooperative cancellation fenced by a stable identity.

    Only a caller presenting the exact ``cancellation_id`` may cancel. The
    serializable snapshot is secret- and body-free.
    """

    def __init__(self, cancellation_id: str) -> None:
        identity = _bounded_text(
            cancellation_id, "cancellation_id", max_chars=_MAX_TEXT_FIELD_CHARS
        )
        self._cancellation_id = identity
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""

    @property
    def cancellation_id(self) -> str:
        return self._cancellation_id

    def cancel(self, *, cancellation_id: str, reason: str = "cancelled") -> bool:
        """Cancel only when the caller presents the exact fencing identity."""

        if _text(cancellation_id, "cancellation_id") != self._cancellation_id:
            return False
        normalized = _bounded_text(
            reason or "cancelled", "reason", max_chars=_MAX_DIAGNOSTIC_CHARS
        )
        with self._lock:
            if self._event.is_set():
                return True
            self._reason = normalized
            self._event.set()
            return True

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def is_cancelled(self) -> bool:
        return self.cancelled

    def is_set(self) -> bool:
        return self.cancelled

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise HarnessError(
                f"work cancelled: {self.reason or 'cancelled'}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cancellation_id": self._cancellation_id,
            "cancelled": self.cancelled,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CancellationToken":
        payload = _closed(
            data,
            frozenset({"cancellation_id", "cancelled", "reason"}),
            "CancellationToken",
        )
        token = cls(_text(payload["cancellation_id"], "cancellation_id"))
        cancelled = _bool(payload["cancelled"], "cancelled")
        reason = payload["reason"]
        if cancelled:
            if reason in (None, ""):
                cancel_reason = "cancelled"
            else:
                cancel_reason = _bounded_text(
                    reason, "reason", max_chars=_MAX_DIAGNOSTIC_CHARS
                )
            token.cancel(
                cancellation_id=token.cancellation_id,
                reason=cancel_reason,
            )
        elif reason not in (None, ""):
            raise HarnessError(
                "CancellationToken reason must be empty when not cancelled"
            )
        return token


@dataclass(frozen=True)
class SemanticWorkRequest:
    """Idempotent request for one harness scheduling stage.

    Carries only references and bounded metadata. Source bodies, prompts, and
    secrets are forbidden.
    """

    work_id: str
    work_kind: str
    attempt_id: str
    idempotency_key: str
    mode: str
    repository_id: str
    input_artifact_cids: tuple[str, ...]
    base_root_cid: str | None
    resource: ResourceBinding
    provider: ProviderBinding | None
    cancellation_id: str

    _FIELDS = frozenset(
        {
            "work_id",
            "work_kind",
            "attempt_id",
            "idempotency_key",
            "mode",
            "repository_id",
            "input_artifact_cids",
            "base_root_cid",
            "resource",
            "provider",
            "cancellation_id",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "work_kind": self.work_kind,
            "attempt_id": self.attempt_id,
            "idempotency_key": self.idempotency_key,
            "mode": self.mode,
            "repository_id": self.repository_id,
            "input_artifact_cids": list(self.input_artifact_cids),
            "base_root_cid": self.base_root_cid,
            "resource": self.resource.to_dict(),
            "provider": None if self.provider is None else self.provider.to_dict(),
            "cancellation_id": self.cancellation_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticWorkRequest":
        payload = _closed(data, cls._FIELDS, "SemanticWorkRequest")
        work_kind = _work_kind_value(payload["work_kind"])
        mode = _mode_value(payload["mode"])
        resource = payload["resource"]
        if not isinstance(resource, Mapping):
            raise HarnessError("resource must be an object")
        provider_raw = payload["provider"]
        provider: ProviderBinding | None
        if provider_raw is None:
            provider = None
        else:
            if not isinstance(provider_raw, Mapping):
                raise HarnessError("provider must be an object or null")
            provider = ProviderBinding.from_dict(provider_raw)
        if requires_provider(work_kind) and provider is None:
            raise HarnessError(
                "model_invocation work requires a provider binding"
            )
        if not requires_provider(work_kind) and provider is not None:
            raise HarnessError(
                f"{work_kind} work must not bind a provider"
            )
        if provider is not None and provider.mode != mode:
            raise HarnessError("provider.mode must match request.mode")
        if mode == HarnessMode.PRODUCTION.value and provider is not None and provider.simulated:
            raise HarnessError("production requests cannot use simulated providers")
        return cls(
            work_id=_bounded_text(
                payload["work_id"], "work_id", max_chars=_MAX_TEXT_FIELD_CHARS
            ),
            work_kind=work_kind,
            attempt_id=_bounded_text(
                payload["attempt_id"], "attempt_id", max_chars=_MAX_TEXT_FIELD_CHARS
            ),
            idempotency_key=_bounded_text(
                payload["idempotency_key"],
                "idempotency_key",
                max_chars=_MAX_TEXT_FIELD_CHARS,
            ),
            mode=mode,
            repository_id=_bounded_text(
                payload["repository_id"],
                "repository_id",
                max_chars=_MAX_TEXT_FIELD_CHARS,
            ),
            input_artifact_cids=_bounded_artifact_cids(
                payload["input_artifact_cids"], "input_artifact_cids"
            ),
            base_root_cid=_optional_cid(payload["base_root_cid"], "base_root_cid"),
            resource=ResourceBinding.from_dict(resource),
            provider=provider,
            cancellation_id=_bounded_text(
                payload["cancellation_id"],
                "cancellation_id",
                max_chars=_MAX_TEXT_FIELD_CHARS,
            ),
        )

    @classmethod
    def build(
        cls,
        *,
        work_kind: str | WorkKind,
        attempt_id: str,
        repository_id: str,
        mode: str | HarnessMode = HarnessMode.DEVELOPMENT,
        input_artifact_cids: Sequence[str] = (),
        base_root_cid: str | None = None,
        resource: ResourceBinding | None = None,
        provider: ProviderBinding | None = None,
        cancellation_id: str | None = None,
        idempotency_key: str | None = None,
        work_id: str | None = None,
    ) -> "SemanticWorkRequest":
        """Construct a closed request with a deterministic work identity."""

        kind = _work_kind_value(work_kind if isinstance(work_kind, str) else work_kind.value)
        mode_value = _mode_value(mode if isinstance(mode, str) else mode.value)
        attempt = _bounded_text(attempt_id, "attempt_id", max_chars=_MAX_TEXT_FIELD_CHARS)
        inputs = tuple(input_artifact_cids)
        identity = work_id or compute_semantic_work_identity(
            work_kind=kind,
            repository_id=repository_id,
            attempt_id=attempt,
            input_artifact_cids=inputs,
            base_root_cid=base_root_cid,
        )
        cancel_id = cancellation_id or f"cancel:{attempt}"
        idem = idempotency_key or identity
        binding = resource or ResourceBinding.for_work_kind(kind)
        return cls.from_dict(
            {
                "work_id": identity,
                "work_kind": kind,
                "attempt_id": attempt,
                "idempotency_key": idem,
                "mode": mode_value,
                "repository_id": repository_id,
                "input_artifact_cids": list(inputs),
                "base_root_cid": base_root_cid,
                "resource": binding.to_dict(),
                "provider": None if provider is None else provider.to_dict(),
                "cancellation_id": cancel_id,
            }
        )

    @property
    def is_heuristic_work_product(self) -> bool:
        return work_product_is_heuristic(self.work_kind)


@dataclass(frozen=True)
class SemanticWorkResult:
    """Typed terminal result for one scheduled work unit.

    ``scheduling_success`` reports admission/execution disposition only.
    ``verification_success`` is always false: scheduler outcomes never certify
    verification or root acceptance.
    """

    request: SemanticWorkRequest
    status: str
    lease: LeaseBinding | None
    provider: ProviderBinding | None
    unavailable: UnavailableResult | None
    reason_codes: tuple[str, ...]
    output_artifact_cids: tuple[str, ...]
    diagnostic: str
    simulated: bool

    _FIELDS = frozenset(
        {
            "request",
            "status",
            "lease",
            "provider",
            "unavailable",
            "reason_codes",
            "output_artifact_cids",
            "diagnostic",
            "simulated",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "status": self.status,
            "lease": None if self.lease is None else self.lease.to_dict(),
            "provider": None if self.provider is None else self.provider.to_dict(),
            "unavailable": None
            if self.unavailable is None
            else self.unavailable.to_dict(),
            "reason_codes": list(self.reason_codes),
            "output_artifact_cids": list(self.output_artifact_cids),
            "diagnostic": self.diagnostic,
            "simulated": self.simulated,
            # Explicit derived flags so observers cannot confuse authorities.
            "scheduling_success": self.scheduling_success,
            "verification_success": self.verification_success,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticWorkResult":
        # Derived flags may be present for observation but are recomputed.
        if not isinstance(data, Mapping):
            raise HarnessError("SemanticWorkResult must be an object")
        raw = dict(data)
        raw.pop("scheduling_success", None)
        raw.pop("verification_success", None)
        payload = _closed(raw, cls._FIELDS, "SemanticWorkResult")
        request_raw = payload["request"]
        if not isinstance(request_raw, Mapping):
            raise HarnessError("request must be an object")
        request = SemanticWorkRequest.from_dict(request_raw)
        status = _status_value(payload["status"])
        lease_raw = payload["lease"]
        lease: LeaseBinding | None
        if lease_raw is None:
            lease = None
        else:
            if not isinstance(lease_raw, Mapping):
                raise HarnessError("lease must be an object or null")
            lease = LeaseBinding.from_dict(lease_raw)
        provider_raw = payload["provider"]
        provider: ProviderBinding | None
        if provider_raw is None:
            provider = None
        else:
            if not isinstance(provider_raw, Mapping):
                raise HarnessError("provider must be an object or null")
            provider = ProviderBinding.from_dict(provider_raw)
        unavailable_raw = payload["unavailable"]
        unavailable: UnavailableResult | None
        if unavailable_raw is None:
            unavailable = None
        else:
            if not isinstance(unavailable_raw, Mapping):
                raise HarnessError("unavailable must be an object or null")
            unavailable = UnavailableResult.from_dict(unavailable_raw)
        simulated = _bool(payload["simulated"], "simulated")
        diagnostic = _bounded_text(
            payload["diagnostic"], "diagnostic", max_chars=_MAX_DIAGNOSTIC_CHARS
        )
        reason_codes = _bounded_reason_codes(payload["reason_codes"])
        outputs = _bounded_artifact_cids(
            payload["output_artifact_cids"], "output_artifact_cids"
        )
        cls._validate_invariants(
            request=request,
            status=status,
            lease=lease,
            provider=provider,
            unavailable=unavailable,
            simulated=simulated,
            reason_codes=reason_codes,
        )
        return cls(
            request=request,
            status=status,
            lease=lease,
            provider=provider,
            unavailable=unavailable,
            reason_codes=reason_codes,
            output_artifact_cids=outputs,
            diagnostic=diagnostic,
            simulated=simulated,
        )

    @staticmethod
    def _validate_invariants(
        *,
        request: SemanticWorkRequest,
        status: str,
        lease: LeaseBinding | None,
        provider: ProviderBinding | None,
        unavailable: UnavailableResult | None,
        simulated: bool,
        reason_codes: tuple[str, ...],
    ) -> None:
        if status == SemanticWorkStatus.UNAVAILABLE.value:
            if unavailable is None:
                raise HarnessError(
                    "unavailable status requires UnavailableResult"
                )
        elif unavailable is not None:
            raise HarnessError(
                "UnavailableResult is only valid with unavailable status"
            )

        if status == SemanticWorkStatus.SIMULATED.value:
            if not simulated:
                raise HarnessError("simulated status requires simulated=true")
            if request.mode == HarnessMode.PRODUCTION.value:
                raise HarnessError(
                    "production mode cannot emit simulated work results"
                )
        if simulated and status not in {
            SemanticWorkStatus.SIMULATED.value,
            SemanticWorkStatus.ADMITTED.value,
            SemanticWorkStatus.SUCCEEDED.value,
            SemanticWorkStatus.FAILED.value,
            SemanticWorkStatus.CANCELLED.value,
        }:
            # Simulated flag is allowed on development outcomes except pure unavailable
            # without simulation label; unavailable stays non-simulated.
            if status == SemanticWorkStatus.UNAVAILABLE.value:
                raise HarnessError(
                    "unavailable results must not be labeled simulated"
                )

        if status == SemanticWorkStatus.ADMITTED.value and lease is None:
            raise HarnessError("admitted status requires a lease binding")

        if status == SemanticWorkStatus.SUCCEEDED.value and lease is None:
            raise HarnessError("succeeded status requires a lease binding")

        if status == SemanticWorkStatus.CANCELLED.value:
            if "cancelled" not in reason_codes and not any(
                code.startswith("cancel") for code in reason_codes
            ):
                raise HarnessError(
                    "cancelled status requires a cancellation reason code"
                )

        if (
            request.mode == HarnessMode.PRODUCTION.value
            and simulated
        ):
            raise HarnessError("production results cannot be simulated")

        if provider is not None and provider.simulated != simulated and status == SemanticWorkStatus.SIMULATED.value:
            raise HarnessError(
                "simulated status requires a simulated provider binding when present"
            )

        if (
            request.mode == HarnessMode.PRODUCTION.value
            and provider is not None
            and provider.simulated
        ):
            raise HarnessError("production results cannot bind simulated providers")

    @property
    def scheduling_success(self) -> bool:
        """Whether scheduling/admission/execution completed without failure.

        This never implies verification success or root acceptance.
        """

        return self.status in {
            SemanticWorkStatus.ADMITTED.value,
            SemanticWorkStatus.SUCCEEDED.value,
            SemanticWorkStatus.SIMULATED.value,
        }

    @property
    def verification_success(self) -> bool:
        """Scheduler records never certify verification.

        Only a separate ``VerificationReceipt`` that is fresh, non-simulated,
        and acceptance-eligible may prove verification.
        """

        return False

    @property
    def execution_completed(self) -> bool:
        return self.status in {
            SemanticWorkStatus.SUCCEEDED.value,
            SemanticWorkStatus.SIMULATED.value,
        }

    def as_scheduler_observation(self) -> "SchedulerObservation":
        return SchedulerObservation.from_result(self)


@dataclass(frozen=True)
class SchedulerObservation:
    """Bounded, secret- and source-body-free projection of work progress.

    Observations are admission metadata only. They cannot carry prompts, model
    output, repository source, credentials, or unbounded diagnostics.
    """

    work_id: str
    attempt_id: str
    work_kind: str
    status: str
    reason_codes: tuple[str, ...]
    fencing_token: int | None
    simulated: bool
    mode: str
    scheduling_success: bool
    verification_success: bool

    _FIELDS = frozenset(
        {
            "work_id",
            "attempt_id",
            "work_kind",
            "status",
            "reason_codes",
            "fencing_token",
            "simulated",
            "mode",
            "scheduling_success",
            "verification_success",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "attempt_id": self.attempt_id,
            "work_kind": self.work_kind,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "fencing_token": self.fencing_token,
            "simulated": self.simulated,
            "mode": self.mode,
            "scheduling_success": self.scheduling_success,
            "verification_success": self.verification_success,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SchedulerObservation":
        if not isinstance(data, Mapping):
            raise HarnessError("SchedulerObservation must be an object")
        forbidden = sorted(set(data) & _FORBIDDEN_OBSERVATION_KEYS)
        if forbidden:
            raise HarnessError(
                f"scheduler observation forbids secret/source fields: {forbidden}"
            )
        if len(data) > _MAX_OBSERVATION_KEYS:
            raise HarnessError(
                f"scheduler observation must have at most {_MAX_OBSERVATION_KEYS} keys"
            )
        payload = _closed(data, cls._FIELDS, "SchedulerObservation")
        fencing = payload["fencing_token"]
        if fencing is not None:
            fencing = _positive_int(fencing, "fencing_token")
        scheduling_success = _bool(payload["scheduling_success"], "scheduling_success")
        verification_success = _bool(
            payload["verification_success"], "verification_success"
        )
        if verification_success:
            raise HarnessError(
                "scheduler observation cannot claim verification_success"
            )
        status = _status_value(payload["status"])
        expected_scheduling = status in {
            SemanticWorkStatus.ADMITTED.value,
            SemanticWorkStatus.SUCCEEDED.value,
            SemanticWorkStatus.SIMULATED.value,
        }
        if scheduling_success != expected_scheduling:
            raise HarnessError(
                "scheduling_success does not match status"
            )
        return cls(
            work_id=_bounded_text(
                payload["work_id"], "work_id", max_chars=_MAX_TEXT_FIELD_CHARS
            ),
            attempt_id=_bounded_text(
                payload["attempt_id"], "attempt_id", max_chars=_MAX_TEXT_FIELD_CHARS
            ),
            work_kind=_work_kind_value(payload["work_kind"]),
            status=status,
            reason_codes=_bounded_reason_codes(payload["reason_codes"]),
            fencing_token=fencing,
            simulated=_bool(payload["simulated"], "simulated"),
            mode=_mode_value(payload["mode"]),
            scheduling_success=scheduling_success,
            verification_success=False,
        )

    @classmethod
    def from_result(cls, result: SemanticWorkResult) -> "SchedulerObservation":
        fencing = None if result.lease is None else result.lease.fencing_token
        return cls.from_dict(
            {
                "work_id": result.request.work_id,
                "attempt_id": result.request.attempt_id,
                "work_kind": result.request.work_kind,
                "status": result.status,
                "reason_codes": list(result.reason_codes),
                "fencing_token": fencing,
                "simulated": result.simulated,
                "mode": result.request.mode,
                "scheduling_success": result.scheduling_success,
                "verification_success": result.verification_success,
            }
        )


def semantic_work_scheduling_descriptor() -> dict[str, Any]:
    """Closed interface metadata for SemanticWorkScheduling@1."""

    return {
        "interface": SEMANTIC_WORK_SCHEDULING_INTERFACE,
        "schema": SCHEDULING_CONTRACTS_SCHEMA,
        "board_namespace": BOARD_NAMESPACE,
        "work_kinds": [item.value for item in WorkKind],
        "statuses": [item.value for item in SemanticWorkStatus],
        "records": [
            "SemanticWorkRequest",
            "SemanticWorkResult",
            "SemanticWorkStatus",
            "CancellationToken",
            "LeaseBinding",
            "ResourceBinding",
            "ProviderBinding",
            "SchedulerObservation",
        ],
        "invariants": [
            "scheduler_observations_secret_and_source_body_free",
            "scheduling_success_is_not_verification_success",
            "unavailable_cancelled_failed_simulated_are_distinct",
            "model_invocation_work_products_are_heuristic",
            "production_rejects_simulated_reservations",
        ],
    }


__all__ = [
    "BOARD_NAMESPACE",
    "SCHEDULING_CONTRACTS_SCHEMA",
    "SEMANTIC_WORK_SCHEDULING_INTERFACE",
    "CancellationToken",
    "HarnessError",
    "HarnessMode",
    "LeaseBinding",
    "ProviderBinding",
    "ResourceBinding",
    "SchedulerObservation",
    "SemanticWorkRequest",
    "SemanticWorkResult",
    "SemanticWorkStatus",
    "UnavailableResult",
    "WorkKind",
    "compute_semantic_work_identity",
    "requires_provider",
    "resource_class_for_work_kind",
    "semantic_work_scheduling_descriptor",
    "stage_for_work_kind",
    "work_product_is_heuristic",
]
