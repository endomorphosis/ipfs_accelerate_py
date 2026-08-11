"""Thin orchestration surface for deterministic-doctor operations (LPR-039).

:class:`DeterministicDoctorService` is the control-plane owner of
``inspect`` / ``explain`` / ``plan`` / ``repair`` / ``replay`` / ``rollback``
plus operator ``status`` and ``verify`` projections.  It:

* evaluates every request against :class:`DeterministicDoctorPolicy` first;
* keeps ``inspect`` / ``explain`` / ``plan`` strictly read-only;
* admits ``repair`` only after explicit operation elevation, enabled policy,
  an exact clean target (evidence snapshot with rebuild equivalence), a
  writer lease, and an eligible admitted plan;
* stores and reloads run receipts by incident CID so ``replay`` is
  identity-equivalent and incident-idempotent;
* returns actionable abstentions when optional stage backends are missing
  without making construction or cold import unhealthy; and
* hard-fails any LLM / remote model-provider / network surface with no
  automatic fallback.

This module owns orchestration and receipt projection only.  It does not
re-implement analysis, proof search, rendering, transaction engines, or
sandbox mutation logic — those remain in their stage modules and may be
injected as optional backends.  Optional datasets / prover / embedding
providers are never imported at module load.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final, Protocol

from ..analysis.deterministic_doctor_contracts import (
    ALL_DOCTOR_OPERATIONS,
    DEFAULT_DOCTOR_MODE,
    DEFAULT_DOCTOR_OPERATION,
    DETERMINISTIC_DOCTOR_RUN_RECEIPT_SCHEMA,
    DETERMINISTIC_DOCTOR_VERSION,
    DeterministicDoctorAuthorityError,
    DeterministicDoctorError,
    DeterministicDoctorPlan,
    DeterministicDoctorRunReceipt,
    DeterministicDoctorSafetyError,
    DoctorAuthorityRoots,
    DoctorEvidenceSnapshot,
    DoctorMode,
    DoctorOperation,
    DoctorPlanDisposition,
    DoctorRejectionReason,
    DoctorRepairDisposition,
    DoctorResourceBounds,
    ForgedDeterministicDoctorIdentityError,
    operation_is_read_only,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)
from ..validation.deterministic_doctor_policy import (
    DeterministicDoctorPolicy,
    DoctorPolicyDecision,
    PolicyVerdict,
    assert_run_receipt_policy,
    evaluate_doctor_operation,
    load_deterministic_doctor_policy,
)

# ---------------------------------------------------------------------------
# Schemas / interface
# ---------------------------------------------------------------------------

DETERMINISTIC_DOCTOR_SERVICE_INTERFACE: Final[str] = "DeterministicDoctorService@1"
DOCTOR_OPERATION_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/operation-request@1"
)
DOCTOR_OPERATION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/operation-result@1"
)
DOCTOR_SERVICE_STATUS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/service-status@1"
)
DOCTOR_SERVICE_DISCOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/service-discovery@1"
)

MAX_REQUEST_BYTES: Final[int] = 262_144
MAX_RESULT_BYTES: Final[int] = 262_144
MAX_TEXT_BYTES: Final[int] = 4_096

# Closed operator surface: DoctorOperation values plus meta projections.
SERVICE_META_OPERATIONS: Final[tuple[str, ...]] = ("status", "verify")
ALL_SERVICE_OPERATIONS: Final[tuple[str, ...]] = tuple(
    op.value for op in ALL_DOCTOR_OPERATIONS
) + SERVICE_META_OPERATIONS

# Body/secret markers forbidden on request surfaces and argv projection.
_FORBIDDEN_PAYLOAD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "source",
        "source_text",
        "source_body",
        "file_body",
        "raw_source",
        "secret",
        "secrets",
        "password",
        "passwd",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "private_key",
        "credential",
        "credentials",
        "cookie",
        "session",
    }
)

# Optional provider module roots that must never load on cold import.
_OPTIONAL_PROVIDER_ROOTS: Final[tuple[str, ...]] = (
    "torch",
    "transformers",
    "openai",
    "anthropic",
    "neo4j",
    "duckdb",
    "httpx",
    "aiohttp",
    "requests",
    "llm_router",
)


class DoctorServiceError(DeterministicDoctorError):
    """Malformed service request, result, or orchestration failure."""


class DoctorServiceSafetyError(DeterministicDoctorSafetyError):
    """A deterministic-doctor service safety invariant was violated."""


class DoctorServiceCapabilityCode(str, Enum):
    """Actionable codes when a stage backend is unavailable or unsupported."""

    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    CAPABILITY_UNSUPPORTED = "capability_unsupported"
    CAPABILITY_INCOMPATIBLE = "capability_incompatible"
    STAGE_BACKEND_MISSING = "stage_backend_missing"
    INCIDENT_NOT_FOUND = "incident_not_found"
    EXACT_CLEAN_TARGET_REQUIRED = "exact_clean_target_required"
    REPLAY_IDENTITY_MISMATCH = "replay_identity_mismatch"
    POLICY_REJECTED = "policy_rejected"
    POLICY_ABSTAINED = "policy_abstained"
    POLICY_APPROVAL_REQUIRED = "policy_approval_required"
    VERIFY_FAILED = "verify_failed"
    LLM_INVOCATION_FORBIDDEN = "llm_invocation_forbidden"
    NETWORK_ACCESS_FORBIDDEN = "network_access_forbidden"
    BODY_OR_SECRET_FORBIDDEN = "body_or_secret_forbidden"
    CONTROL_DEPENDENCY_INVALID = "control_dependency_invalid"
    CONTROL_PERMIT_REQUIRED = "control_permit_required"
    CONTROL_PERMIT_REJECTED = "control_permit_rejected"
    CONTROL_EFFECT_MISMATCH = "control_effect_mismatch"


# ---------------------------------------------------------------------------
# Normalization helpers (body-free, fail-closed)
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = False,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise DoctorServiceError(f"{field_name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise DoctorServiceError(f"{field_name} is required")
    if len(text.encode("utf-8")) > limit:
        raise DoctorServiceError(f"{field_name} exceeds its byte bound")
    if "\0" in text:
        raise DoctorServiceError(f"{field_name} must not contain NUL")
    return text


def _optional_text(value: Any, field_name: str) -> str:
    if value in (None, ""):
        return ""
    return _text(value, field_name, required=True)


def _identifier(value: Any, field_name: str) -> str:
    value = _text(value, field_name, required=True)
    if any(char.isspace() for char in value):
        raise DoctorServiceError(f"{field_name} must be an opaque compact identifier")
    return value


def _optional_identifier(value: Any, field_name: str) -> str:
    if value in (None, ""):
        return ""
    return _identifier(value, field_name)


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise DoctorServiceError(f"{field_name} must be a boolean")
    return value


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise DoctorServiceError(f"{field_name} must be one of: {allowed}") from exc


def _ids(
    values: Any,
    field_name: str,
    *,
    required: bool = False,
    limit: int = 256,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str) or not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray)
    ):
        raise DoctorServiceError(f"{field_name} must be a sequence of identifiers")
    else:
        raw = values
    if len(raw) > limit:
        raise DoctorServiceError(f"{field_name} exceeds its item bound")
    out = tuple(sorted({_identifier(item, field_name) for item in raw}))
    if required and not out:
        raise DoctorServiceError(f"{field_name} must not be empty")
    return out


def _paths(values: Any, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str) or not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray)
    ):
        raise DoctorServiceError(f"{field_name} must be a sequence of paths")
    out: list[str] = []
    for item in values:
        path = _text(item, field_name, required=True, limit=1_024)
        if path.startswith("/") or ".." in path.split("/"):
            raise DeterministicDoctorAuthorityError(
                f"{field_name} must be a relative repository path"
            )
        if path not in out:
            out.append(path)
    return tuple(sorted(out))


def _is_forbidden_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_").strip()
    if normalized in _FORBIDDEN_PAYLOAD_MARKERS:
        return True
    for marker in _FORBIDDEN_PAYLOAD_MARKERS:
        if normalized == marker or normalized.endswith("_" + marker):
            return True
    return False


def assert_body_free(value: Any, field_name: str = "payload") -> None:
    """Reject source bodies and secrets even when smuggled through a mapping."""

    if isinstance(value, float):
        raise DoctorServiceError(f"{field_name} may not contain floating-point values")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DoctorServiceError(f"{field_name} has a non-string key")
            if _is_forbidden_key(key):
                raise DoctorServiceError(
                    f"{field_name} may not contain source bodies or secrets"
                )
            assert_body_free(item, field_name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            assert_body_free(item, field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise DoctorServiceError(f"{field_name} may not contain binary bodies")


def _bounded(record: CanonicalContract, name: str, limit: int = MAX_RESULT_BYTES) -> None:
    assert_body_free(record.to_dict(), name)
    if len(canonical_json_bytes(record.to_dict())) > limit:
        raise DoctorServiceError(f"{name} exceeds its serialized byte bound")


def _decode_roots(value: Any) -> DoctorAuthorityRoots | None:
    if value is None or value == "":
        return None
    if isinstance(value, DoctorAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return DoctorAuthorityRoots.from_dict(value)
    raise DoctorServiceError("roots must be DoctorAuthorityRoots or mapping")


def _decode_snapshot(value: Any) -> DoctorEvidenceSnapshot | None:
    if value is None or value == "":
        return None
    if isinstance(value, DoctorEvidenceSnapshot):
        return value
    if isinstance(value, Mapping):
        return DoctorEvidenceSnapshot.from_dict(value)
    raise DoctorServiceError("snapshot must be DoctorEvidenceSnapshot or mapping")


def _decode_plan(value: Any) -> DeterministicDoctorPlan | None:
    if value is None or value == "":
        return None
    if isinstance(value, DeterministicDoctorPlan):
        return value
    if isinstance(value, Mapping):
        return DeterministicDoctorPlan.from_dict(value)
    raise DoctorServiceError("plan must be DeterministicDoctorPlan or mapping")


def _decode_receipt(value: Any) -> DeterministicDoctorRunReceipt | None:
    if value is None or value == "":
        return None
    if isinstance(value, DeterministicDoctorRunReceipt):
        return value
    if isinstance(value, Mapping):
        return DeterministicDoctorRunReceipt.from_dict(value)
    raise DoctorServiceError("receipt must be DeterministicDoctorRunReceipt or mapping")


def _operation_token(value: Any) -> str:
    if isinstance(value, DoctorOperation):
        return value.value
    text = _text(value, "operation", required=True)
    if text not in ALL_SERVICE_OPERATIONS:
        raise DoctorServiceError(
            f"operation must be one of: {', '.join(ALL_SERVICE_OPERATIONS)}"
        )
    return text


def is_meta_operation(operation: str | DoctorOperation) -> bool:
    token = operation.value if isinstance(operation, DoctorOperation) else str(operation)
    return token in SERVICE_META_OPERATIONS


def is_doctor_operation(operation: str | DoctorOperation) -> bool:
    token = operation.value if isinstance(operation, DoctorOperation) else str(operation)
    try:
        DoctorOperation(token)
    except ValueError:
        return False
    return True


def optional_providers_loaded() -> tuple[str, ...]:
    """Return optional provider roots currently present in ``sys.modules``."""

    import sys

    found: list[str] = []
    for root in _OPTIONAL_PROVIDER_ROOTS:
        if root in sys.modules or any(
            name == root or name.startswith(root + ".") for name in sys.modules
        ):
            found.append(root)
    return tuple(sorted(set(found)))


# True remote / generative LLM client roots.  These must not be present when
# constructing a deterministic Doctor service.
_LLM_CLIENT_ROOTS = frozenset(
    {
        "openai",
        "anthropic",
        "llm_router",
    }
)
# Local ML frameworks that often load transitively via accelerate/pytest
# plugins.  They are not generative LLM clients; rejecting ambient presence
# permanently stalls factory construction in monorepo supervisors.
_ML_FRAMEWORK_ROOTS = frozenset(
    {
        "torch",
        "transformers",
    }
)


def assert_no_llm_surface_loaded(
    *,
    include_ml_frameworks: bool = False,
    baseline_modules: frozenset[str] | None = None,
) -> None:
    """Fail closed if a generative LLM client surface is loaded.

    By default only true LLM/remote-client roots (``openai``, ``anthropic``,
    ``llm_router``) are rejected.  Ambient ``torch`` / ``transformers``
    presence from the host process is allowed so production factory builds and
    sealed pytest runs do not permanently stall when those packages were
    imported by unrelated plugins.

    Pass ``include_ml_frameworks=True`` for hermetic cold-import suites that
    require a process free of local ML frameworks as well.

    When ``baseline_modules`` is supplied, only **new** module roots loaded
    since that snapshot are considered (delta check during factory build).
    """

    import sys

    loaded = optional_providers_loaded()
    if baseline_modules is not None:
        current = frozenset(sys.modules)
        new_names = current - baseline_modules
        loaded = tuple(
            sorted(
                {
                    root
                    for root in loaded
                    if root in new_names
                    or any(
                        name == root or name.startswith(root + ".")
                        for name in new_names
                    )
                }
            )
        )
    forbidden_roots = set(_LLM_CLIENT_ROOTS)
    if include_ml_frameworks:
        forbidden_roots |= set(_ML_FRAMEWORK_ROOTS)
    forbidden = {name for name in loaded if name in forbidden_roots}
    if forbidden:
        raise DoctorServiceSafetyError(
            f"optional model/provider surface already loaded: {sorted(forbidden)}"
        )


# ---------------------------------------------------------------------------
# Request / result contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorOperationRequest(CanonicalContract):
    """Body-free, content-addressed doctor operation request.

    Requests carry only opaque identifiers, authority roots, and optional
    plan/snapshot records.  Source bodies and secrets are rejected.
    """

    SCHEMA: ClassVar[str] = DOCTOR_OPERATION_REQUEST_SCHEMA

    operation: str
    mode: DoctorMode = DEFAULT_DOCTOR_MODE
    request_id: str = ""
    incident_id: str = ""
    roots: DoctorAuthorityRoots | None = None
    snapshot: DoctorEvidenceSnapshot | None = None
    plan: DeterministicDoctorPlan | None = None
    prior_receipt: DeterministicDoctorRunReceipt | None = None
    lease_id: str = ""
    checkpoint_ref: str = ""
    rollback_ref: str = ""
    target_tree_cid: str = ""
    exact_clean_target: bool = False
    write_paths: tuple[str, ...] = ()
    finding_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    # Explicit safety observation flags (default fail-closed false).
    llm_router_invoked: bool = False
    remote_model_provider_invoked: bool = False
    model_invocation_count: int = 0
    provider_invocation_count: int = 0
    network_access: bool = False
    target_code_imported: bool = False
    semantic_authority_flags: Mapping[str, Any] = field(default_factory=dict)
    open_required_frontiers: tuple[str, ...] = ()
    approval_classes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        op = _operation_token(self.operation)
        object.__setattr__(self, "operation", op)
        object.__setattr__(self, "mode", _enum(self.mode, DoctorMode, "mode"))
        object.__setattr__(
            self, "request_id", _optional_identifier(self.request_id, "request_id")
        )
        object.__setattr__(
            self, "incident_id", _optional_identifier(self.incident_id, "incident_id")
        )
        object.__setattr__(self, "roots", _decode_roots(self.roots))
        object.__setattr__(self, "snapshot", _decode_snapshot(self.snapshot))
        object.__setattr__(self, "plan", _decode_plan(self.plan))
        object.__setattr__(self, "prior_receipt", _decode_receipt(self.prior_receipt))
        for name in (
            "lease_id",
            "checkpoint_ref",
            "rollback_ref",
            "target_tree_cid",
        ):
            object.__setattr__(
                self, name, _optional_identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "exact_clean_target",
            _bool(self.exact_clean_target, "exact_clean_target"),
        )
        object.__setattr__(self, "write_paths", _paths(self.write_paths, "write_paths"))
        object.__setattr__(
            self, "finding_ids", _ids(self.finding_ids, "finding_ids")
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes")
        )
        object.__setattr__(
            self,
            "llm_router_invoked",
            _bool(self.llm_router_invoked, "llm_router_invoked"),
        )
        object.__setattr__(
            self,
            "remote_model_provider_invoked",
            _bool(self.remote_model_provider_invoked, "remote_model_provider_invoked"),
        )
        for name in ("model_invocation_count", "provider_invocation_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DoctorServiceError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "network_access", _bool(self.network_access, "network_access")
        )
        object.__setattr__(
            self,
            "target_code_imported",
            _bool(self.target_code_imported, "target_code_imported"),
        )
        flags = self.semantic_authority_flags or {}
        if not isinstance(flags, Mapping):
            raise DoctorServiceError("semantic_authority_flags must be a mapping")
        assert_body_free(flags, "semantic_authority_flags")
        object.__setattr__(self, "semantic_authority_flags", dict(flags))
        object.__setattr__(
            self,
            "open_required_frontiers",
            _ids(self.open_required_frontiers, "open_required_frontiers"),
        )
        object.__setattr__(
            self, "approval_classes", _ids(self.approval_classes, "approval_classes")
        )
        if not self.request_id:
            object.__setattr__(
                self,
                "request_id",
                content_identity(
                    {
                        "schema": "doctor-request-id@1",
                        "operation": self.operation,
                        "incident_id": self.incident_id,
                        "mode": self.mode.value
                        if isinstance(self.mode, DoctorMode)
                        else str(self.mode),
                    }
                ),
            )
        _bounded(self, "doctor operation request", MAX_REQUEST_BYTES)

    @property
    def is_read_only(self) -> bool:
        if is_meta_operation(self.operation):
            return True
        return operation_is_read_only(self.operation)

    @property
    def doctor_operation(self) -> DoctorOperation | None:
        if not is_doctor_operation(self.operation):
            return None
        return DoctorOperation(self.operation)

    def effective_roots(self) -> DoctorAuthorityRoots | None:
        if self.roots is not None:
            return self.roots
        if self.snapshot is not None:
            return self.snapshot.roots
        if self.plan is not None:
            return self.plan.roots
        if self.prior_receipt is not None:
            return self.prior_receipt.roots
        return None

    def effective_snapshot_id(self) -> str:
        if self.snapshot is not None:
            return self.snapshot.snapshot_id
        if self.plan is not None:
            return self.plan.snapshot_id
        if self.prior_receipt is not None:
            return self.prior_receipt.snapshot_id
        return ""

    def effective_lease_id(self) -> str:
        if self.lease_id:
            return self.lease_id
        if self.plan is not None:
            return self.plan.lease_id or self.plan.roots.lease_id
        roots = self.effective_roots()
        if roots is not None:
            return roots.lease_id
        return ""

    def effective_checkpoint_ref(self) -> str:
        if self.checkpoint_ref:
            return self.checkpoint_ref
        if self.plan is not None:
            return self.plan.checkpoint_ref
        if self.prior_receipt is not None:
            return self.prior_receipt.checkpoint_ref
        return ""

    def effective_rollback_ref(self) -> str:
        if self.rollback_ref:
            return self.rollback_ref
        if self.plan is not None:
            return self.plan.rollback_ref
        if self.prior_receipt is not None:
            return self.prior_receipt.rollback_ref
        return ""

    def incident_cid(self) -> str:
        """Stable incident identity for store lookup and replay idempotency."""

        if self.incident_id:
            return self.incident_id
        roots = self.effective_roots()
        return content_identity(
            {
                "schema": "doctor-incident@1",
                "operation": self.operation,
                "snapshot_id": self.effective_snapshot_id(),
                "plan_id": self.plan.plan_id if self.plan is not None else "",
                "roots_id": roots.content_id if roots is not None else "",
                "request_id": self.request_id,
            }
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            "operation": self.operation,
            "mode": self.mode.value
            if isinstance(self.mode, DoctorMode)
            else str(self.mode),
            "request_id": self.request_id,
            "incident_id": self.incident_id,
            "roots": self.roots.to_dict() if self.roots is not None else None,
            "snapshot": self.snapshot.to_dict() if self.snapshot is not None else None,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "prior_receipt": (
                self.prior_receipt.to_dict() if self.prior_receipt is not None else None
            ),
            "lease_id": self.lease_id,
            "checkpoint_ref": self.checkpoint_ref,
            "rollback_ref": self.rollback_ref,
            "target_tree_cid": self.target_tree_cid,
            "exact_clean_target": self.exact_clean_target,
            "write_paths": list(self.write_paths),
            "finding_ids": list(self.finding_ids),
            "reason_codes": list(self.reason_codes),
            "llm_router_invoked": False,
            "remote_model_provider_invoked": False,
            "model_invocation_count": 0,
            "provider_invocation_count": 0,
            "network_access": False,
            "target_code_imported": False,
            "semantic_authority_flags": dict(self.semantic_authority_flags),
            "open_required_frontiers": list(self.open_required_frontiers),
            "approval_classes": list(self.approval_classes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorOperationRequest":
        if not isinstance(payload, Mapping):
            raise DoctorServiceError("operation request must be a mapping")
        assert_body_free(payload, "operation request")
        schema = payload.get("schema")
        if schema not in (None, "", cls.SCHEMA):
            raise DoctorServiceError(
                f"unsupported operation request schema; use {cls.SCHEMA}"
            )
        data = {
            key: payload[key]
            for key in (
                "operation",
                "mode",
                "request_id",
                "incident_id",
                "roots",
                "snapshot",
                "plan",
                "prior_receipt",
                "lease_id",
                "checkpoint_ref",
                "rollback_ref",
                "target_tree_cid",
                "exact_clean_target",
                "write_paths",
                "finding_ids",
                "reason_codes",
                "llm_router_invoked",
                "remote_model_provider_invoked",
                "model_invocation_count",
                "provider_invocation_count",
                "network_access",
                "target_code_imported",
                "semantic_authority_flags",
                "open_required_frontiers",
                "approval_classes",
            )
            if key in payload
        }
        return cls(**data)


@dataclass(frozen=True)
class DoctorOperationResult(CanonicalContract):
    """Bounded, machine-readable outcome of one doctor service operation."""

    SCHEMA: ClassVar[str] = DOCTOR_OPERATION_RESULT_SCHEMA

    request_id: str
    operation: str
    mode: DoctorMode
    disposition: DoctorRepairDisposition
    incident_id: str
    read_only: bool
    policy_decision: DoctorPolicyDecision | None = None
    run_receipt: DeterministicDoctorRunReceipt | None = None
    reason_codes: tuple[str, ...] = ()
    explanation: str = ""
    changed: bool = False
    replayed: bool = False
    status: Mapping[str, Any] = field(default_factory=dict)
    stage_refs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_id", _identifier(self.request_id, "request_id")
        )
        object.__setattr__(self, "operation", _operation_token(self.operation))
        object.__setattr__(self, "mode", _enum(self.mode, DoctorMode, "mode"))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorRepairDisposition, "disposition"),
        )
        object.__setattr__(
            self, "incident_id", _optional_identifier(self.incident_id, "incident_id")
        )
        object.__setattr__(self, "read_only", _bool(self.read_only, "read_only"))
        if self.policy_decision is not None and not isinstance(
            self.policy_decision, DoctorPolicyDecision
        ):
            if isinstance(self.policy_decision, Mapping):
                object.__setattr__(
                    self,
                    "policy_decision",
                    DoctorPolicyDecision.from_dict(self.policy_decision),
                )
            else:
                raise DoctorServiceError(
                    "policy_decision must be DoctorPolicyDecision or mapping"
                )
        if self.run_receipt is not None and not isinstance(
            self.run_receipt, DeterministicDoctorRunReceipt
        ):
            if isinstance(self.run_receipt, Mapping):
                object.__setattr__(
                    self,
                    "run_receipt",
                    DeterministicDoctorRunReceipt.from_dict(self.run_receipt),
                )
            else:
                raise DoctorServiceError(
                    "run_receipt must be DeterministicDoctorRunReceipt or mapping"
                )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes")
        )
        object.__setattr__(
            self,
            "explanation",
            _text(self.explanation, "explanation", required=False, limit=MAX_TEXT_BYTES),
        )
        object.__setattr__(self, "changed", _bool(self.changed, "changed"))
        object.__setattr__(self, "replayed", _bool(self.replayed, "replayed"))
        status = self.status or {}
        stages = self.stage_refs or {}
        if not isinstance(status, Mapping) or not isinstance(stages, Mapping):
            raise DoctorServiceError("status and stage_refs must be mappings")
        assert_body_free(status, "status")
        assert_body_free(stages, "stage_refs")
        object.__setattr__(self, "status", dict(status))
        object.__setattr__(
            self,
            "stage_refs",
            {str(k): str(v) for k, v in stages.items()},
        )
        if self.read_only and self.changed:
            raise DoctorServiceError("read-only results cannot report changed=True")
        if (
            self.run_receipt is not None
            and self.run_receipt.operation is DoctorOperation.REPAIR
            and self.run_receipt.disposition is DoctorRepairDisposition.SUPPORTED
            and self.read_only
        ):
            raise DoctorServiceError("supported repair cannot be marked read-only")
        _bounded(self, "doctor operation result", MAX_RESULT_BYTES)

    @property
    def succeeded(self) -> bool:
        return self.disposition is DoctorRepairDisposition.SUPPORTED

    @property
    def abstained(self) -> bool:
        return self.disposition is DoctorRepairDisposition.ABSTAIN

    @property
    def exit_code(self) -> int:
        """CLI-oriented exit code (0 success, 1 failure, 3 abstain, 4 approval)."""

        if self.disposition is DoctorRepairDisposition.SUPPORTED:
            return 0
        if self.disposition is DoctorRepairDisposition.ABSTAIN:
            return 3
        if self.disposition is DoctorRepairDisposition.APPROVAL_REQUIRED:
            return 4
        if self.disposition in (
            DoctorRepairDisposition.ROLLED_BACK,
            DoctorRepairDisposition.QUARANTINED,
        ):
            return 1
        return 1

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            "request_id": self.request_id,
            "operation": self.operation,
            "mode": self.mode.value
            if isinstance(self.mode, DoctorMode)
            else str(self.mode),
            "disposition": self.disposition.value
            if isinstance(self.disposition, DoctorRepairDisposition)
            else str(self.disposition),
            "incident_id": self.incident_id,
            "read_only": self.read_only,
            "policy_decision": (
                self.policy_decision.to_dict()
                if self.policy_decision is not None
                else None
            ),
            "run_receipt": (
                self.run_receipt.to_dict() if self.run_receipt is not None else None
            ),
            "reason_codes": list(self.reason_codes),
            "explanation": self.explanation,
            "changed": self.changed,
            "replayed": self.replayed,
            "status": dict(self.status),
            "stage_refs": dict(self.stage_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorOperationResult":
        if not isinstance(payload, Mapping):
            raise DoctorServiceError("operation result must be a mapping")
        assert_body_free(payload, "operation result")
        schema = payload.get("schema")
        if schema not in (None, "", cls.SCHEMA):
            raise DoctorServiceError(
                f"unsupported operation result schema; use {cls.SCHEMA}"
            )
        data = {
            key: payload[key]
            for key in (
                "request_id",
                "operation",
                "mode",
                "disposition",
                "incident_id",
                "read_only",
                "policy_decision",
                "run_receipt",
                "reason_codes",
                "explanation",
                "changed",
                "replayed",
                "status",
                "stage_refs",
            )
            if key in payload
        }
        return cls(**data)


# ---------------------------------------------------------------------------
# Receipt store (incident-CID index)
# ---------------------------------------------------------------------------


class DoctorReceiptStore(Protocol):
    """Minimal store for incident-CID indexed run receipts."""

    def get(self, incident_id: str) -> DeterministicDoctorRunReceipt | None: ...

    def put(self, incident_id: str, receipt: DeterministicDoctorRunReceipt) -> None: ...

    def list_incident_ids(self) -> tuple[str, ...]: ...


class InMemoryDoctorReceiptStore:
    """Process-local, body-free receipt index (default; no database)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_incident: dict[str, DeterministicDoctorRunReceipt] = {}

    def get(self, incident_id: str) -> DeterministicDoctorRunReceipt | None:
        with self._lock:
            return self._by_incident.get(incident_id)

    def put(self, incident_id: str, receipt: DeterministicDoctorRunReceipt) -> None:
        if not isinstance(receipt, DeterministicDoctorRunReceipt):
            raise DoctorServiceError("store requires DeterministicDoctorRunReceipt")
        with self._lock:
            existing = self._by_incident.get(incident_id)
            if existing is not None and existing.content_id != receipt.content_id:
                # Incident-CID idempotency: identical payloads are fine; drift fails.
                if existing.to_dict() != receipt.to_dict():
                    raise DoctorServiceError(
                        DoctorServiceCapabilityCode.REPLAY_IDENTITY_MISMATCH.value
                    )
            self._by_incident[incident_id] = receipt

    def list_incident_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._by_incident))


class RuntimeCASDoctorReceiptStore:
    """Optional RuntimeCAS-backed receipt index (lazy; never opened on import)."""

    def __init__(self, cas: Any, *, namespace: str = "deterministic-doctor") -> None:
        self._cas = cas
        self._namespace = _identifier(namespace, "namespace")
        self._lock = threading.RLock()

    def _key(self, incident_id: str) -> str:
        return f"{self._namespace}/incident/{incident_id}"

    def get(self, incident_id: str) -> DeterministicDoctorRunReceipt | None:
        key = self._key(_identifier(incident_id, "incident_id"))
        with self._lock:
            getter = getattr(self._cas, "get", None)
            if getter is None:
                return None
            raw = getter(key)
            if raw is None:
                return None
            if isinstance(raw, bytes):
                payload = json.loads(raw.decode("utf-8"))
            elif isinstance(raw, Mapping):
                payload = raw
            else:
                raise DoctorServiceError("CAS returned an unsupported receipt payload")
            return DeterministicDoctorRunReceipt.from_dict(payload)

    def put(self, incident_id: str, receipt: DeterministicDoctorRunReceipt) -> None:
        key = self._key(_identifier(incident_id, "incident_id"))
        payload = receipt.to_record()
        assert_body_free(payload, "cas receipt")
        blob = canonical_json_bytes(payload)
        with self._lock:
            existing = self.get(incident_id)
            if existing is not None and existing.content_id != receipt.content_id:
                if existing.to_dict() != receipt.to_dict():
                    raise DoctorServiceError(
                        DoctorServiceCapabilityCode.REPLAY_IDENTITY_MISMATCH.value
                    )
            putter = getattr(self._cas, "put", None)
            if putter is None:
                raise DoctorServiceError("CAS does not support put")
            putter(key, blob)

    def list_incident_ids(self) -> tuple[str, ...]:
        # RuntimeCAS may not support listing; empty is a safe abstention surface.
        lister = getattr(self._cas, "list_keys", None) or getattr(
            self._cas, "keys", None
        )
        if lister is None:
            return ()
        prefix = f"{self._namespace}/incident/"
        keys = lister()
        out: list[str] = []
        for key in keys:
            text = str(key)
            if text.startswith(prefix):
                out.append(text[len(prefix) :])
        return tuple(sorted(out))


# ---------------------------------------------------------------------------
# Optional stage backends (injected; never eager-imported)
# ---------------------------------------------------------------------------


class DoctorStageBackend(Protocol):
    """Optional stage handler.  Missing backends yield actionable abstentions."""

    def __call__(
        self,
        request: DoctorOperationRequest,
        *,
        policy: DeterministicDoctorPolicy,
        policy_decision: DoctorPolicyDecision,
    ) -> DoctorOperationResult | DeterministicDoctorRunReceipt | Mapping[str, Any]: ...


@dataclass(frozen=True)
class DoctorControlRequest:
    """Provider-free exact permit/effect binding for a Doctor mutation.

    The shared control catalog does not define a Doctor-specific operation.
    A deployment therefore injects a narrow adapter that translates this
    record to its canonical control operation.  The Doctor service never maps
    repair onto an unrelated catalog operation and never treats dependency
    presence as authority.
    """

    operation: str
    request_id: str
    incident_id: str
    roots_id: str
    tree_id: str
    snapshot_id: str
    plan_id: str
    lease_id: str
    checkpoint_ref: str
    rollback_ref: str
    write_paths: tuple[str, ...]
    expected_effect_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.operation != DoctorOperation.REPAIR.value:
            raise DoctorServiceError("Doctor control requests only authorize repair")
        for name in (
            "request_id",
            "incident_id",
            "roots_id",
            "tree_id",
            "snapshot_id",
            "plan_id",
            "lease_id",
            "checkpoint_ref",
            "rollback_ref",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "write_paths", _paths(self.write_paths, "write_paths"))
        object.__setattr__(
            self,
            "expected_effect_ids",
            _ids(
                self.expected_effect_ids,
                "expected_effect_ids",
                required=True,
                limit=256,
            ),
        )

    @property
    def control_request_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "deterministic-doctor/control-request@1"
            ),
            "operation": self.operation,
            "request_id": self.request_id,
            "incident_id": self.incident_id,
            "roots_id": self.roots_id,
            "tree_id": self.tree_id,
            "snapshot_id": self.snapshot_id,
            "plan_id": self.plan_id,
            "lease_id": self.lease_id,
            "checkpoint_ref": self.checkpoint_ref,
            "rollback_ref": self.rollback_ref,
            "write_paths": list(self.write_paths),
            "expected_effect_ids": list(self.expected_effect_ids),
        }


class DoctorControlDependency(Protocol):
    """Narrow adapter for immediate permit consumption and effect auditing."""

    def authorize_doctor_operation(self, request: DoctorControlRequest) -> Any: ...

    def record_doctor_effects(
        self,
        request: DoctorControlRequest,
        *,
        permit: Any,
        applied_effect_ids: Sequence[str],
        changed: bool,
    ) -> Any: ...


@dataclass
class DoctorStageBackends:
    """Optional injectables for diagnose / plan / synthesis / impact / txn / FP."""

    diagnose: DoctorStageBackend | None = None
    plan: DoctorStageBackend | None = None
    synthesis: DoctorStageBackend | None = None
    impact: DoctorStageBackend | None = None
    transaction: DoctorStageBackend | None = None
    fixed_point: DoctorStageBackend | None = None
    explain: DoctorStageBackend | None = None
    retrieve: DoctorStageBackend | None = None
    tactician: DoctorStageBackend | None = None
    proof: DoctorStageBackend | None = None

    def available(self) -> tuple[str, ...]:
        names: list[str] = []
        for name in (
            "diagnose",
            "plan",
            "synthesis",
            "impact",
            "transaction",
            "fixed_point",
            "explain",
            "retrieve",
            "tactician",
            "proof",
        ):
            if getattr(self, name) is not None:
                names.append(name)
        return tuple(names)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DeterministicDoctorService:
    """Transport-neutral deterministic-doctor control service.

    Construction and :meth:`discovery` are side-effect free: no process is
    started, no database opened, no network used, and no optional provider is
    imported.  Stage backends are optional injectables; when absent the service
    still answers with policy-gated receipts or actionable abstentions.
    """

    INTERFACE: Final[str] = DETERMINISTIC_DOCTOR_SERVICE_INTERFACE

    def __init__(
        self,
        *,
        policy: DeterministicDoctorPolicy | Mapping[str, Any] | None = None,
        receipt_store: DoctorReceiptStore | None = None,
        backends: DoctorStageBackends | None = None,
        control_service: Any | None = None,
        cas: Any | None = None,
    ) -> None:
        self._policy = load_deterministic_doctor_policy(policy)
        if receipt_store is not None:
            self._store: DoctorReceiptStore = receipt_store
        elif cas is not None:
            self._store = RuntimeCASDoctorReceiptStore(cas)
        else:
            self._store = InMemoryDoctorReceiptStore()
        self._backends = backends or DoctorStageBackends()
        self._control_service = control_service
        self._lock = threading.RLock()
        self._invocation_guard = {
            "llm_router_invoked": False,
            "remote_model_provider_invoked": False,
            "model_invocation_count": 0,
            "provider_invocation_count": 0,
        }

    # -- static discovery --------------------------------------------------

    @staticmethod
    def discovery(
        policy: DeterministicDoctorPolicy | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Static discovery; does not construct providers or inspect the host."""

        resolved = load_deterministic_doctor_policy(policy)
        return {
            "schema": DOCTOR_SERVICE_DISCOVERY_SCHEMA,
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            "interface": DETERMINISTIC_DOCTOR_SERVICE_INTERFACE,
            "operations": list(ALL_SERVICE_OPERATIONS),
            "doctor_operations": [op.value for op in ALL_DOCTOR_OPERATIONS],
            "meta_operations": list(SERVICE_META_OPERATIONS),
            "modes": [mode.value for mode in DoctorMode],
            "default_mode": resolved.default_mode.value
            if isinstance(resolved.default_mode, DoctorMode)
            else str(resolved.default_mode),
            "default_operation": DEFAULT_DOCTOR_OPERATION.value,
            "read_only_operations": [
                op.value
                for op in ALL_DOCTOR_OPERATIONS
                if op.is_read_only
            ]
            + list(SERVICE_META_OPERATIONS),
            "write_operations": [
                op.value for op in ALL_DOCTOR_OPERATIONS if op.may_write
            ],
            "policy_enabled": bool(resolved.enabled),
            "llm_router_enabled": False,
            "remote_model_provider_calls_allowed": False,
            "network_access_allowed": False,
            "optional_providers_loaded": False,
            "processes_started": False,
            "database_opened": False,
            "automatic_fallback": False,
            "run_receipt_schema": DETERMINISTIC_DOCTOR_RUN_RECEIPT_SCHEMA,
        }

    @property
    def policy(self) -> DeterministicDoctorPolicy:
        return self._policy

    @property
    def backends_available(self) -> tuple[str, ...]:
        return self._backends.available()

    # -- public entry points -----------------------------------------------

    def execute(
        self,
        request: DoctorOperationRequest | Mapping[str, Any] | str,
        **kwargs: Any,
    ) -> DoctorOperationResult:
        """Execute one doctor operation request (policy-first, fail-closed)."""

        selected = self._decode_request(request, **kwargs)
        with self._lock:
            self._reject_live_llm_surface(selected)
            if selected.operation == "status":
                return self._status(selected)
            if selected.operation == "verify":
                return self._verify(selected)

            decision = self._evaluate_policy(selected)
            if decision.verdict is PolicyVerdict.REJECT:
                return self._terminal(
                    selected,
                    decision,
                    disposition=DoctorRepairDisposition.ABSTAIN
                    if DoctorRejectionReason.UNSUPPORTED_OPERATION.value
                    in decision.reason_codes
                    else DoctorRepairDisposition.QUARANTINED
                    if DoctorRejectionReason.LLM_INVOCATION.value
                    in decision.reason_codes
                    or DoctorRejectionReason.REMOTE_MODEL_PROVIDER.value
                    in decision.reason_codes
                    else DoctorRepairDisposition.ABSTAIN,
                    reason_codes=decision.reason_codes
                    + (DoctorServiceCapabilityCode.POLICY_REJECTED.value,),
                    explanation="policy rejected the requested doctor operation",
                )
            if decision.verdict is PolicyVerdict.ABSTAIN:
                return self._terminal(
                    selected,
                    decision,
                    disposition=DoctorRepairDisposition.ABSTAIN,
                    reason_codes=decision.reason_codes
                    + (DoctorServiceCapabilityCode.POLICY_ABSTAINED.value,),
                    explanation="policy abstained for the requested doctor operation",
                )
            if decision.verdict is PolicyVerdict.APPROVAL_REQUIRED:
                return self._terminal(
                    selected,
                    decision,
                    disposition=DoctorRepairDisposition.APPROVAL_REQUIRED,
                    reason_codes=decision.reason_codes
                    + (DoctorServiceCapabilityCode.POLICY_APPROVAL_REQUIRED.value,),
                    explanation="approval is required before this doctor operation",
                )

            op = selected.operation
            if op == DoctorOperation.INSPECT.value:
                return self._inspect(selected, decision)
            if op == DoctorOperation.EXPLAIN.value:
                return self._explain(selected, decision)
            if op == DoctorOperation.PLAN.value:
                return self._plan(selected, decision)
            if op == DoctorOperation.REPAIR.value:
                return self._repair(selected, decision)
            if op == DoctorOperation.REPLAY.value:
                return self._replay(selected, decision)
            if op == DoctorOperation.ROLLBACK.value:
                return self._rollback(selected, decision)
            return self._terminal(
                selected,
                decision,
                disposition=DoctorRepairDisposition.ABSTAIN,
                reason_codes=(
                    DoctorRejectionReason.UNSUPPORTED_OPERATION.value,
                    DoctorServiceCapabilityCode.CAPABILITY_UNSUPPORTED.value,
                ),
                explanation="unsupported doctor operation",
            )

    def inspect(self, **kwargs: Any) -> DoctorOperationResult:
        return self.execute({"operation": DoctorOperation.INSPECT.value, **kwargs})

    def explain(self, **kwargs: Any) -> DoctorOperationResult:
        return self.execute({"operation": DoctorOperation.EXPLAIN.value, **kwargs})

    def plan(self, **kwargs: Any) -> DoctorOperationResult:
        return self.execute({"operation": DoctorOperation.PLAN.value, **kwargs})

    def repair(self, **kwargs: Any) -> DoctorOperationResult:
        return self.execute({"operation": DoctorOperation.REPAIR.value, **kwargs})

    def replay(self, **kwargs: Any) -> DoctorOperationResult:
        return self.execute({"operation": DoctorOperation.REPLAY.value, **kwargs})

    def rollback(self, **kwargs: Any) -> DoctorOperationResult:
        return self.execute({"operation": DoctorOperation.ROLLBACK.value, **kwargs})

    def status(self, **kwargs: Any) -> DoctorOperationResult:
        return self.execute({"operation": "status", **kwargs})

    def verify(self, **kwargs: Any) -> DoctorOperationResult:
        return self.execute({"operation": "verify", **kwargs})

    # -- request decoding --------------------------------------------------

    def _decode_request(
        self,
        request: DoctorOperationRequest | Mapping[str, Any] | str,
        **kwargs: Any,
    ) -> DoctorOperationRequest:
        if isinstance(request, DoctorOperationRequest):
            if kwargs:
                raise DoctorServiceError(
                    "kwargs are not accepted with a typed DoctorOperationRequest"
                )
            return request
        if isinstance(request, str):
            payload: dict[str, Any] = {"operation": request, **kwargs}
        elif isinstance(request, Mapping):
            payload = {**dict(request), **kwargs}
        else:
            raise DoctorServiceError("invalid doctor operation request")
        assert_body_free(payload, "operation request")
        return DoctorOperationRequest.from_dict(payload)

    def _evaluate_policy(
        self, request: DoctorOperationRequest
    ) -> DoctorPolicyDecision:
        op = request.doctor_operation
        if op is None:
            # Meta ops do not pass through DoctorOperation policy evaluate.
            return DoctorPolicyDecision(
                verdict=PolicyVerdict.ALLOW,
                operation=DoctorOperation.INSPECT,
                mode=request.mode,
                reason_codes=("meta_operation",),
                policy_id=self._policy.policy_id,
                read_only=True,
            )
        write_paths = request.write_paths
        if not write_paths and request.plan is not None:
            write_paths = request.plan.permitted_write_paths
        return evaluate_doctor_operation(
            op,
            policy=self._policy,
            mode=request.mode,
            plan=request.plan,
            write_paths=write_paths,
            approval_classes=request.approval_classes,
            open_required_frontiers=request.open_required_frontiers,
            lease_id=request.effective_lease_id(),
            checkpoint_ref=request.effective_checkpoint_ref(),
            rollback_ref=request.effective_rollback_ref(),
            llm_router_invoked=request.llm_router_invoked
            or self._invocation_guard["llm_router_invoked"],
            remote_model_provider_invoked=request.remote_model_provider_invoked
            or self._invocation_guard["remote_model_provider_invoked"],
            model_invocation_count=request.model_invocation_count
            + int(self._invocation_guard["model_invocation_count"]),
            provider_invocation_count=request.provider_invocation_count
            + int(self._invocation_guard["provider_invocation_count"]),
            network_access=request.network_access,
            target_code_imported=request.target_code_imported,
            semantic_authority_flags=request.semantic_authority_flags,
        )

    def _reject_live_llm_surface(self, request: DoctorOperationRequest) -> None:
        """Hard-fail intercepted LLM / model-provider observations.

        There is intentionally no automatic fallback path.
        """

        if (
            request.llm_router_invoked
            or request.remote_model_provider_invoked
            or request.model_invocation_count
            or request.provider_invocation_count
            or self._invocation_guard["llm_router_invoked"]
            or self._invocation_guard["remote_model_provider_invoked"]
            or self._invocation_guard["model_invocation_count"]
            or self._invocation_guard["provider_invocation_count"]
        ):
            # Surface as a safety error for direct callers / tests that intercept
            # the provider surface; execute maps this into a quarantined result
            # only when observed via request flags after policy evaluation.
            # Raising keeps "no automatic fallback" fail-closed.
            raise DoctorServiceSafetyError(
                DoctorServiceCapabilityCode.LLM_INVOCATION_FORBIDDEN.value
            )
        if request.network_access:
            raise DoctorServiceSafetyError(
                DoctorServiceCapabilityCode.NETWORK_ACCESS_FORBIDDEN.value
            )

    # -- operation handlers ------------------------------------------------

    def _inspect(
        self,
        request: DoctorOperationRequest,
        decision: DoctorPolicyDecision,
    ) -> DoctorOperationResult:
        if self._backends.diagnose is not None:
            return self._delegate(
                self._backends.diagnose, request, decision, read_only=True
            )
        roots = request.effective_roots()
        snapshot = request.snapshot
        reason_codes: list[str] = ["inspect_report"]
        disposition = DoctorRepairDisposition.SUPPORTED
        if snapshot is None and roots is None:
            reason_codes = [
                DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value,
                DoctorServiceCapabilityCode.STAGE_BACKEND_MISSING.value,
                "inspect_without_snapshot",
            ]
            disposition = DoctorRepairDisposition.ABSTAIN
            explanation = (
                "inspect abstained: bind a DoctorEvidenceSnapshot or inject a "
                "diagnose backend; no optional provider was started"
            )
            return self._terminal(
                request,
                decision,
                disposition=disposition,
                reason_codes=tuple(reason_codes),
                explanation=explanation,
            )
        if roots is None and snapshot is not None:
            roots = snapshot.roots
        assert roots is not None
        receipt = self._build_run_receipt(
            request,
            operation=DoctorOperation.INSPECT,
            disposition=DoctorRepairDisposition.SUPPORTED,
            roots=roots,
            reason_codes=tuple(reason_codes),
        )
        return self._finish(
            request,
            decision,
            receipt=receipt,
            disposition=DoctorRepairDisposition.SUPPORTED,
            reason_codes=tuple(reason_codes),
            explanation="inspect completed as a read-only evidence report",
            read_only=True,
            store=True,
        )

    def _explain(
        self,
        request: DoctorOperationRequest,
        decision: DoctorPolicyDecision,
    ) -> DoctorOperationResult:
        if self._backends.explain is not None:
            return self._delegate(
                self._backends.explain, request, decision, read_only=True
            )
        roots = request.effective_roots()
        if roots is None:
            return self._terminal(
                request,
                decision,
                disposition=DoctorRepairDisposition.ABSTAIN,
                reason_codes=(
                    DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value,
                    "explain_without_roots",
                ),
                explanation=(
                    "explain abstained: bind authority roots/snapshot/plan or "
                    "inject an explain backend"
                ),
            )
        reasons = ["explain_report"]
        if request.plan is not None:
            reasons.append(f"plan_disposition:{request.plan.disposition.value}")
        if request.finding_ids:
            reasons.append(f"finding_count:{len(request.finding_ids)}")
        receipt = self._build_run_receipt(
            request,
            operation=DoctorOperation.EXPLAIN,
            disposition=DoctorRepairDisposition.SUPPORTED,
            roots=roots,
            reason_codes=tuple(reasons),
            plan_id=request.plan.plan_id if request.plan is not None else "",
        )
        return self._finish(
            request,
            decision,
            receipt=receipt,
            disposition=DoctorRepairDisposition.SUPPORTED,
            reason_codes=tuple(reasons),
            explanation="explain completed as a read-only disposition report",
            read_only=True,
            store=True,
        )

    def _plan(
        self,
        request: DoctorOperationRequest,
        decision: DoctorPolicyDecision,
    ) -> DoctorOperationResult:
        if self._backends.plan is not None:
            return self._delegate(
                self._backends.plan, request, decision, read_only=True
            )
        roots = request.effective_roots()
        if request.plan is not None and roots is None:
            roots = request.plan.roots
        if roots is None:
            return self._terminal(
                request,
                decision,
                disposition=DoctorRepairDisposition.ABSTAIN,
                reason_codes=(
                    DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value,
                    DoctorServiceCapabilityCode.STAGE_BACKEND_MISSING.value,
                    "plan_backend_or_plan_required",
                ),
                explanation=(
                    "plan abstained: inject a plan backend or supply an existing "
                    "DeterministicDoctorPlan; optional tactician/provider not started"
                ),
            )
        plan = request.plan
        if plan is None:
            return self._terminal(
                request,
                decision,
                disposition=DoctorRepairDisposition.ABSTAIN,
                reason_codes=(
                    DoctorServiceCapabilityCode.STAGE_BACKEND_MISSING.value,
                    "plan_materialization_deferred",
                ),
                explanation=(
                    "plan abstained: no planner backend is bound; provide a plan "
                    "record or inject DeterministicDoctorTactician/impact stages"
                ),
            )
        disposition = (
            DoctorRepairDisposition.SUPPORTED
            if plan.disposition is DoctorPlanDisposition.ADMITTED
            else DoctorRepairDisposition.ABSTAIN
            if plan.disposition is DoctorPlanDisposition.ABSTAINED
            else DoctorRepairDisposition.APPROVAL_REQUIRED
            if plan.disposition is DoctorPlanDisposition.APPROVAL_REQUIRED
            else DoctorRepairDisposition.ABSTAIN
        )
        reasons = [
            "plan_report",
            f"plan_disposition:{plan.disposition.value}",
        ]
        receipt = self._build_run_receipt(
            request,
            operation=DoctorOperation.PLAN,
            disposition=disposition,
            roots=roots,
            reason_codes=tuple(reasons),
            plan_id=plan.plan_id,
            impact_closure_ref=plan.impact_closure_id,
        )
        return self._finish(
            request,
            decision,
            receipt=receipt,
            disposition=disposition,
            reason_codes=tuple(reasons),
            explanation="plan completed as a read-only analytical report",
            read_only=True,
            store=True,
            stage_refs={"plan_id": plan.plan_id, "impact_closure": plan.impact_closure_id},
        )

    def _repair(
        self,
        request: DoctorOperationRequest,
        decision: DoctorPolicyDecision,
    ) -> DoctorOperationResult:
        # Policy already enforced admitted plan / lease / checkpoint / rollback.
        if not self._policy.enabled:
            return self._terminal(
                request,
                decision,
                disposition=DoctorRepairDisposition.ABSTAIN,
                reason_codes=(
                    DoctorRejectionReason.MODE_FORBIDS_OPERATION.value,
                    DoctorServiceCapabilityCode.POLICY_REJECTED.value,
                ),
                explanation="repair requires an enabled deterministic-doctor policy",
            )
        if not request.exact_clean_target and not self._has_exact_clean_target(request):
            return self._terminal(
                request,
                decision,
                disposition=DoctorRepairDisposition.ABSTAIN,
                reason_codes=(
                    DoctorServiceCapabilityCode.EXACT_CLEAN_TARGET_REQUIRED.value,
                ),
                explanation=(
                    "repair requires an exact clean target: evidence snapshot with "
                    "clean-rebuild equivalence and matching tree identity"
                ),
            )
        if self._backends.transaction is not None:
            control_request: DoctorControlRequest | None = None
            control_permit: Any = None
            if self._control_service is not None:
                try:
                    control_request = self._build_control_request(request)
                    control_permit = self._authorize_control(control_request)
                except DoctorServiceSafetyError:
                    raise
                except Exception as exc:  # noqa: BLE001 - actionable abstention
                    reason = (
                        DoctorServiceCapabilityCode.CONTROL_PERMIT_REJECTED.value
                        if str(exc)
                        == DoctorServiceCapabilityCode.CONTROL_PERMIT_REJECTED.value
                        else DoctorServiceCapabilityCode.CONTROL_DEPENDENCY_INVALID.value
                    )
                    return self._terminal(
                        request,
                        decision,
                        disposition=DoctorRepairDisposition.ABSTAIN,
                        reason_codes=(
                            reason,
                            type(exc).__name__,
                        ),
                        explanation=(
                            "repair abstained before transaction: the configured "
                            f"control dependency rejected the exact permit ({exc})"
                        ),
                    )
            result = self._delegate(
                self._backends.transaction, request, decision, read_only=False
            )
            if control_request is not None:
                try:
                    result = self._record_control_effects(
                        control_request,
                        control_permit,
                        result,
                    )
                except DoctorServiceSafetyError:
                    raise
                except Exception as exc:  # noqa: BLE001 - effect audit fail closed
                    return DoctorOperationResult(
                        request_id=request.request_id,
                        operation=request.operation,
                        mode=request.mode,
                        disposition=DoctorRepairDisposition.QUARANTINED,
                        incident_id=request.incident_cid(),
                        read_only=False,
                        policy_decision=decision,
                        run_receipt=result.run_receipt,
                        reason_codes=(
                            DoctorServiceCapabilityCode.CONTROL_EFFECT_MISMATCH.value,
                            type(exc).__name__,
                        ),
                        explanation=(
                            "transaction effect audit failed closed; operator "
                            f"reconciliation/rollback is required ({exc})"
                        ),
                        changed=result.changed,
                        status={
                            **dict(result.status),
                            "control_effects_verified": False,
                        },
                        stage_refs=dict(result.stage_refs),
                    )
            if self._backends.fixed_point is not None and result.succeeded:
                # Optional post-commit fixed-point stage; failures abstain/rollback
                # without silent model fallback.
                try:
                    fp = self._backends.fixed_point(
                        request, policy=self._policy, policy_decision=decision
                    )
                    if isinstance(fp, DoctorOperationResult):
                        if control_request is not None:
                            fp = DoctorOperationResult(
                                request_id=fp.request_id,
                                operation=fp.operation,
                                mode=fp.mode,
                                disposition=fp.disposition,
                                incident_id=fp.incident_id,
                                read_only=fp.read_only,
                                policy_decision=fp.policy_decision,
                                run_receipt=fp.run_receipt,
                                reason_codes=tuple(result.reason_codes)
                                + tuple(fp.reason_codes),
                                explanation=fp.explanation,
                                changed=fp.changed,
                                replayed=fp.replayed,
                                status={
                                    **dict(result.status),
                                    **dict(fp.status),
                                },
                                stage_refs={
                                    **dict(result.stage_refs),
                                    **dict(fp.stage_refs),
                                },
                            )
                        return fp
                except DoctorServiceSafetyError:
                    raise
                except Exception as exc:  # noqa: BLE001 - map to abstention
                    return self._terminal(
                        request,
                        decision,
                        disposition=DoctorRepairDisposition.ABSTAIN,
                        reason_codes=(
                            DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value,
                            "fixed_point_stage_failed",
                        ),
                        explanation=f"fixed-point stage failed closed: {exc}",
                    )
            return result

        # Without a transaction backend, admit only a non-writing repair receipt
        # is forbidden: repair is write-capable.  Abstain with actionable code.
        return self._terminal(
            request,
            decision,
            disposition=DoctorRepairDisposition.ABSTAIN,
            reason_codes=(
                DoctorServiceCapabilityCode.STAGE_BACKEND_MISSING.value,
                DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value,
                "transaction_backend_required",
            ),
            explanation=(
                "repair abstained: inject DeterministicDoctorTransaction (and "
                "optionally fixed-point) backends; the service never mutates "
                "targets itself and starts no optional provider"
            ),
        )

    def _replay(
        self,
        request: DoctorOperationRequest,
        decision: DoctorPolicyDecision,
    ) -> DoctorOperationResult:
        incident = request.incident_cid()
        stored = self._store.get(incident)
        prior = request.prior_receipt or stored
        if prior is None:
            return self._terminal(
                request,
                decision,
                disposition=DoctorRepairDisposition.ABSTAIN,
                reason_codes=(
                    DoctorServiceCapabilityCode.INCIDENT_NOT_FOUND.value,
                ),
                explanation=(
                    f"replay abstained: no receipt for incident {incident}; "
                    "supply prior_receipt or run a prior operation first"
                ),
            )
        # Identity-equivalent replay: re-emit the same receipt payload.
        assert_run_receipt_policy(prior, self._policy)
        if stored is not None and stored.content_id != prior.content_id:
            if stored.to_dict() != prior.to_dict():
                return self._terminal(
                    request,
                    decision,
                    disposition=DoctorRepairDisposition.ABSTAIN,
                    reason_codes=(
                        DoctorServiceCapabilityCode.REPLAY_IDENTITY_MISMATCH.value,
                    ),
                    explanation="replay identity mismatch for incident CID",
                )
        # Re-store for idempotency (same content_id).
        self._store.put(incident, prior)
        again = self._store.get(incident)
        if again is None or again.content_id != prior.content_id:
            raise DoctorServiceError(
                DoctorServiceCapabilityCode.REPLAY_IDENTITY_MISMATCH.value
            )
        return DoctorOperationResult(
            request_id=request.request_id,
            operation=DoctorOperation.REPLAY.value,
            mode=request.mode,
            disposition=prior.disposition,
            incident_id=incident,
            read_only=True,
            policy_decision=decision,
            run_receipt=prior,
            reason_codes=("replay_identity_equivalent", "incident_cid_idempotent"),
            explanation="replay returned the identity-equivalent prior run receipt",
            changed=False,
            replayed=True,
            status={
                "schema": DOCTOR_SERVICE_STATUS_SCHEMA,
                "incident_id": incident,
                "receipt_id": prior.receipt_id,
                "receipt_content_id": prior.content_id,
                "operation": prior.operation.value,
            },
            stage_refs={"receipt_id": prior.receipt_id},
        )

    def _rollback(
        self,
        request: DoctorOperationRequest,
        decision: DoctorPolicyDecision,
    ) -> DoctorOperationResult:
        roots = request.effective_roots()
        checkpoint = request.effective_checkpoint_ref()
        rollback_ref = request.effective_rollback_ref()
        if roots is None:
            return self._terminal(
                request,
                decision,
                disposition=DoctorRepairDisposition.ABSTAIN,
                reason_codes=(
                    DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value,
                    "rollback_without_roots",
                ),
                explanation="rollback abstained: bind roots or a prior receipt",
            )
        if not checkpoint and not rollback_ref:
            return self._terminal(
                request,
                decision,
                disposition=DoctorRepairDisposition.ABSTAIN,
                reason_codes=(
                    DoctorRejectionReason.REPAIR_WITHOUT_ROLLBACK.value,
                    "rollback_evidence_required",
                ),
                explanation="rollback requires checkpoint or rollback evidence",
            )
        if self._backends.transaction is not None:
            return self._delegate(
                self._backends.transaction, request, decision, read_only=True
            )
        receipt = self._build_run_receipt(
            request,
            operation=DoctorOperation.ROLLBACK,
            disposition=DoctorRepairDisposition.ROLLED_BACK,
            roots=roots,
            reason_codes=("rollback_recorded",),
            checkpoint_ref=checkpoint,
            rollback_ref=rollback_ref or checkpoint,
            plan_id=request.plan.plan_id if request.plan is not None else "",
        )
        return self._finish(
            request,
            decision,
            receipt=receipt,
            disposition=DoctorRepairDisposition.ROLLED_BACK,
            reason_codes=("rollback_recorded",),
            explanation="rollback recorded against checkpoint evidence (no new repair)",
            read_only=True,
            store=True,
        )

    def _status(self, request: DoctorOperationRequest) -> DoctorOperationResult:
        incident_id = request.incident_id
        stored = self._store.get(incident_id) if incident_id else None
        incidents = self._store.list_incident_ids()
        status_payload = {
            "schema": DOCTOR_SERVICE_STATUS_SCHEMA,
            "contract_version": DETERMINISTIC_DOCTOR_VERSION,
            "interface": self.INTERFACE,
            "policy_id": self._policy.policy_id,
            "policy_enabled": bool(self._policy.enabled),
            "default_mode": self._policy.default_mode.value
            if isinstance(self._policy.default_mode, DoctorMode)
            else str(self._policy.default_mode),
            "backends_available": list(self.backends_available),
            "incident_id": incident_id,
            "incident_known": stored is not None,
            "incident_count": len(incidents),
            "receipt_id": stored.receipt_id if stored is not None else "",
            "receipt_content_id": stored.content_id if stored is not None else "",
            "last_operation": (
                stored.operation.value if stored is not None else ""
            ),
            "last_disposition": (
                stored.disposition.value if stored is not None else ""
            ),
            "optional_providers_loaded": list(optional_providers_loaded()),
            "llm_router_enabled": False,
            "automatic_fallback": False,
            "processes_started": False,
            "database_opened": False,
            "network_access_allowed": False,
            "control_service_bound": self._control_service is not None,
        }
        disposition = (
            DoctorRepairDisposition.SUPPORTED
            if stored is not None or not incident_id
            else DoctorRepairDisposition.ABSTAIN
        )
        reasons: tuple[str, ...] = ("status_report",)
        if incident_id and stored is None:
            reasons = (
                DoctorServiceCapabilityCode.INCIDENT_NOT_FOUND.value,
                "status_report",
            )
        return DoctorOperationResult(
            request_id=request.request_id,
            operation="status",
            mode=request.mode,
            disposition=disposition,
            incident_id=incident_id,
            read_only=True,
            policy_decision=None,
            run_receipt=stored,
            reason_codes=reasons,
            explanation="status projection of deterministic-doctor service state",
            changed=False,
            replayed=False,
            status=status_payload,
            stage_refs={},
        )

    def _verify(self, request: DoctorOperationRequest) -> DoctorOperationResult:
        receipt = request.prior_receipt
        if receipt is None and request.incident_id:
            receipt = self._store.get(request.incident_id)
        if receipt is None:
            return DoctorOperationResult(
                request_id=request.request_id,
                operation="verify",
                mode=request.mode,
                disposition=DoctorRepairDisposition.ABSTAIN,
                incident_id=request.incident_id,
                read_only=True,
                reason_codes=(
                    DoctorServiceCapabilityCode.INCIDENT_NOT_FOUND.value,
                    DoctorServiceCapabilityCode.VERIFY_FAILED.value,
                ),
                explanation="verify abstained: no run receipt bound for verification",
                changed=False,
                status={
                    "schema": DOCTOR_SERVICE_STATUS_SCHEMA,
                    "verified": False,
                    "llm_router_enabled": False,
                    "automatic_fallback": False,
                },
            )
        try:
            assert_run_receipt_policy(receipt, self._policy)
            if receipt.llm_router_invoked or receipt.remote_model_provider_invoked:
                raise DoctorServiceSafetyError(
                    DoctorServiceCapabilityCode.LLM_INVOCATION_FORBIDDEN.value
                )
            if (
                receipt.model_invocation_count
                or receipt.provider_invocation_count
            ):
                raise DoctorServiceSafetyError(
                    DoctorRejectionReason.NONZERO_MODEL_INVOCATION.value
                )
        except (
            DeterministicDoctorSafetyError,
            DeterministicDoctorAuthorityError,
            DeterministicDoctorError,
            DoctorServiceError,
            ContractValidationError,
        ) as exc:
            return DoctorOperationResult(
                request_id=request.request_id,
                operation="verify",
                mode=request.mode,
                disposition=DoctorRepairDisposition.ABSTAIN,
                incident_id=request.incident_id or receipt.incident_id,
                read_only=True,
                run_receipt=receipt,
                reason_codes=(
                    DoctorServiceCapabilityCode.VERIFY_FAILED.value,
                    str(exc),
                ),
                explanation=f"verify failed closed: {exc}",
                changed=False,
                status={
                    "schema": DOCTOR_SERVICE_STATUS_SCHEMA,
                    "verified": False,
                    "receipt_id": receipt.receipt_id,
                    "llm_router_enabled": False,
                },
            )
        return DoctorOperationResult(
            request_id=request.request_id,
            operation="verify",
            mode=request.mode,
            disposition=DoctorRepairDisposition.SUPPORTED,
            incident_id=request.incident_id or receipt.incident_id,
            read_only=True,
            run_receipt=receipt,
            reason_codes=("verify_passed", "zero_model_invocations"),
            explanation="run receipt verified under deterministic-doctor policy",
            changed=False,
            status={
                "schema": DOCTOR_SERVICE_STATUS_SCHEMA,
                "verified": True,
                "receipt_id": receipt.receipt_id,
                "receipt_content_id": receipt.content_id,
                "llm_router_enabled": False,
                "automatic_fallback": False,
            },
            stage_refs={"receipt_id": receipt.receipt_id},
        )

    # -- helpers -----------------------------------------------------------

    def _build_control_request(
        self, request: DoctorOperationRequest
    ) -> DoctorControlRequest:
        roots = request.effective_roots()
        plan = request.plan
        if roots is None or plan is None:
            raise DoctorServiceError(
                DoctorServiceCapabilityCode.CONTROL_PERMIT_REQUIRED.value
            )
        write_paths = request.write_paths or plan.permitted_write_paths
        if not write_paths:
            raise DoctorServiceError("control permit requires exact write paths")
        effects = tuple(
            content_identity(
                {
                    "schema": "deterministic-doctor-write-effect@1",
                    "operation": request.operation,
                    "tree_id": roots.tree_id,
                    "plan_id": plan.plan_id,
                    "path": path,
                }
            )
            for path in write_paths
        )
        return DoctorControlRequest(
            operation=request.operation,
            request_id=request.request_id,
            incident_id=request.incident_cid(),
            roots_id=roots.content_id,
            tree_id=roots.tree_id,
            snapshot_id=request.effective_snapshot_id(),
            plan_id=plan.plan_id,
            lease_id=request.effective_lease_id(),
            checkpoint_ref=request.effective_checkpoint_ref(),
            rollback_ref=request.effective_rollback_ref(),
            write_paths=write_paths,
            expected_effect_ids=effects,
        )

    @staticmethod
    def _control_value(value: Any, *names: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            for name in names:
                if name in value:
                    return value[name]
            return default
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
        return default

    @classmethod
    def _control_effect_ids(cls, value: Any, *names: str) -> tuple[str, ...]:
        raw = cls._control_value(value, *names, default=None)
        if raw is None:
            return ()
        if isinstance(raw, str) or not isinstance(raw, Sequence):
            raise DoctorServiceError("control effect IDs must be a sequence")
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                effect_id = item
            elif isinstance(item, Mapping):
                effect_id = str(
                    item.get("effect_id")
                    or item.get("id")
                    or item.get("content_id")
                    or ""
                )
            else:
                effect_id = str(
                    getattr(item, "effect_id", "")
                    or getattr(item, "id", "")
                    or getattr(item, "content_id", "")
                )
            if effect_id:
                out.append(_identifier(effect_id, "effect_id"))
        return tuple(sorted(set(out)))

    def _authorize_control(self, request: DoctorControlRequest) -> Any:
        authorizer = getattr(
            self._control_service, "authorize_doctor_operation", None
        )
        if not callable(authorizer):
            raise DoctorServiceError(
                "control dependency must implement authorize_doctor_operation"
            )
        permit = authorizer(request)
        permitted = self._control_value(
            permit, "permitted", "succeeded", "allowed", default=False
        )
        if permitted is not True:
            raise DoctorServiceError(
                DoctorServiceCapabilityCode.CONTROL_PERMIT_REJECTED.value
            )
        permit_id = self._control_value(
            permit,
            "permit_id",
            "decision_id",
            "authorization_id",
            default="",
        )
        if not permit_id:
            raise DoctorServiceError("control dependency returned no permit identity")
        authorized = self._control_effect_ids(
            permit,
            "authorized_effect_ids",
            "expected_effect_ids",
            "effect_ids",
            "effects",
        )
        if authorized != request.expected_effect_ids:
            raise DoctorServiceError(
                "control permit effects differ from the exact requested effects"
            )
        return permit

    def _record_control_effects(
        self,
        request: DoctorControlRequest,
        permit: Any,
        result: DoctorOperationResult,
    ) -> DoctorOperationResult:
        recorder = getattr(self._control_service, "record_doctor_effects", None)
        if not callable(recorder):
            raise DoctorServiceError(
                "control dependency must implement record_doctor_effects"
            )
        applied = request.expected_effect_ids if result.changed else ()
        audit = recorder(
            request,
            permit=permit,
            applied_effect_ids=applied,
            changed=result.changed,
        )
        succeeded = self._control_value(
            audit, "succeeded", "recorded", "verified", default=False
        )
        if succeeded is not True:
            raise DoctorServiceError("control dependency rejected the effect audit")
        observed = self._control_effect_ids(
            audit,
            "applied_effect_ids",
            "effect_ids",
            "effects",
        )
        if observed != applied:
            raise DoctorServiceError(
                "control audit effects differ from transaction effects"
            )
        audit_id = str(
            self._control_value(
                audit,
                "audit_receipt_id",
                "receipt_id",
                "result_id",
                default="",
            )
            or ""
        )
        if not audit_id:
            raise DoctorServiceError("control effect audit returned no receipt identity")
        permit_id = str(
            self._control_value(
                permit,
                "permit_id",
                "decision_id",
                "authorization_id",
                default="",
            )
        )
        return DoctorOperationResult(
            request_id=result.request_id,
            operation=result.operation,
            mode=result.mode,
            disposition=result.disposition,
            incident_id=result.incident_id,
            read_only=result.read_only,
            policy_decision=result.policy_decision,
            run_receipt=result.run_receipt,
            reason_codes=tuple(result.reason_codes) + ("control_effects_verified",),
            explanation=result.explanation,
            changed=result.changed,
            replayed=result.replayed,
            status={
                **dict(result.status),
                "control_effects_verified": True,
                "control_request_id": request.control_request_id,
                "control_permit_id": permit_id,
                "control_audit_receipt_id": audit_id,
                "control_applied_effect_ids": list(applied),
            },
            stage_refs={
                **dict(result.stage_refs),
                "control_permit_id": permit_id,
                "control_audit_receipt_id": audit_id,
            },
        )

    def _has_exact_clean_target(self, request: DoctorOperationRequest) -> bool:
        if request.exact_clean_target:
            return True
        snapshot = request.snapshot
        if snapshot is None:
            return False
        if not snapshot.clean_rebuild_equivalence_receipt_id:
            return False
        if snapshot.completeness not in ("complete", "full", "closed"):
            # Allow the closed vocabulary used by contracts ("complete").
            if snapshot.completeness != "complete":
                return False
        roots = snapshot.roots
        if request.target_tree_cid and request.target_tree_cid != roots.tree_id:
            return False
        return True

    def _build_run_receipt(
        self,
        request: DoctorOperationRequest,
        *,
        operation: DoctorOperation,
        disposition: DoctorRepairDisposition,
        roots: DoctorAuthorityRoots,
        reason_codes: Sequence[str],
        plan_id: str = "",
        checkpoint_ref: str = "",
        rollback_ref: str = "",
        lease_id: str = "",
        candidate_tree_cid: str = "",
        committed_tree_cid: str = "",
        impact_closure_ref: str = "",
        fixed_point_ref: str = "",
        transaction_ref: str = "",
    ) -> DeterministicDoctorRunReceipt:
        snapshot_id = request.effective_snapshot_id() or "snapshot:unbound"
        incident = request.incident_cid()
        invalidation = ()
        if request.snapshot is not None and request.snapshot.invalidation_refs:
            invalidation = request.snapshot.invalidation_refs
        elif request.plan is not None and request.plan.invalidation_refs:
            invalidation = request.plan.invalidation_refs
        else:
            invalidation = (roots.tree_id,)
        receipt_id = content_identity(
            {
                "schema": "doctor-run-receipt-id@1",
                "operation": operation.value,
                "incident_id": incident,
                "snapshot_id": snapshot_id,
                "plan_id": plan_id,
                "disposition": disposition.value,
                "request_id": request.request_id,
            }
        )
        return DeterministicDoctorRunReceipt(
            roots=roots,
            receipt_id=receipt_id,
            operation=operation,
            mode=request.mode,
            disposition=disposition,
            snapshot_id=snapshot_id,
            incident_id=incident,
            plan_id=plan_id,
            candidate_tree_cid=candidate_tree_cid,
            committed_tree_cid=committed_tree_cid,
            network_denied=True,
            secrets_inherited=False,
            lease_id=lease_id or request.effective_lease_id(),
            checkpoint_ref=checkpoint_ref or request.effective_checkpoint_ref(),
            transaction_ref=transaction_ref,
            rollback_ref=rollback_ref or request.effective_rollback_ref(),
            impact_closure_ref=impact_closure_ref,
            fixed_point_ref=fixed_point_ref,
            provider_invocation_count=0,
            model_invocation_count=0,
            llm_router_invoked=False,
            remote_model_provider_invoked=False,
            target_code_imported=False,
            reason_codes=tuple(reason_codes),
            invalidation_refs=invalidation,
            resource_bounds=self._policy.resource_bounds,
        )

    def _delegate(
        self,
        backend: DoctorStageBackend,
        request: DoctorOperationRequest,
        decision: DoctorPolicyDecision,
        *,
        read_only: bool,
    ) -> DoctorOperationResult:
        try:
            raw = backend(
                request, policy=self._policy, policy_decision=decision
            )
        except DoctorServiceSafetyError:
            raise
        except (
            DeterministicDoctorSafetyError,
            DeterministicDoctorAuthorityError,
        ):
            raise
        except Exception as exc:  # noqa: BLE001 - map missing/broken backends
            return self._terminal(
                request,
                decision,
                disposition=DoctorRepairDisposition.ABSTAIN,
                reason_codes=(
                    DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value,
                    "stage_backend_error",
                ),
                explanation=f"stage backend failed closed without fallback: {exc}",
            )
        if isinstance(raw, DoctorOperationResult):
            if read_only and raw.changed:
                raise DoctorServiceError(
                    "read-only stage backend reported a mutation"
                )
            if raw.run_receipt is not None:
                self._store.put(request.incident_cid(), raw.run_receipt)
            return raw
        if isinstance(raw, DeterministicDoctorRunReceipt):
            receipt = raw
        elif isinstance(raw, Mapping):
            receipt = DeterministicDoctorRunReceipt.from_dict(raw)
        else:
            raise DoctorServiceError(
                "stage backend must return DoctorOperationResult, "
                "DeterministicDoctorRunReceipt, or mapping"
            )
        assert_run_receipt_policy(receipt, self._policy)
        return self._finish(
            request,
            decision,
            receipt=receipt,
            disposition=receipt.disposition,
            reason_codes=receipt.reason_codes or ("stage_backend",),
            explanation="stage backend completed without model fallback",
            read_only=read_only,
            store=True,
            changed=bool(receipt.committed_tree_cid) and not read_only,
        )

    def _finish(
        self,
        request: DoctorOperationRequest,
        decision: DoctorPolicyDecision,
        *,
        receipt: DeterministicDoctorRunReceipt,
        disposition: DoctorRepairDisposition,
        reason_codes: Sequence[str],
        explanation: str,
        read_only: bool,
        store: bool,
        changed: bool = False,
        stage_refs: Mapping[str, str] | None = None,
        replayed: bool = False,
    ) -> DoctorOperationResult:
        assert_run_receipt_policy(receipt, self._policy)
        incident = request.incident_cid()
        if store:
            self._store.put(incident, receipt)
        return DoctorOperationResult(
            request_id=request.request_id,
            operation=request.operation,
            mode=request.mode,
            disposition=disposition,
            incident_id=incident,
            read_only=read_only,
            policy_decision=decision,
            run_receipt=receipt,
            reason_codes=tuple(reason_codes),
            explanation=explanation,
            changed=changed,
            replayed=replayed,
            status={
                "schema": DOCTOR_SERVICE_STATUS_SCHEMA,
                "incident_id": incident,
                "receipt_id": receipt.receipt_id,
                "receipt_content_id": receipt.content_id,
                "operation": receipt.operation.value,
                "disposition": receipt.disposition.value,
                "llm_router_enabled": False,
                "automatic_fallback": False,
            },
            stage_refs=dict(stage_refs or {}),
        )

    def _terminal(
        self,
        request: DoctorOperationRequest,
        decision: DoctorPolicyDecision | None,
        *,
        disposition: DoctorRepairDisposition,
        reason_codes: Sequence[str],
        explanation: str,
    ) -> DoctorOperationResult:
        return DoctorOperationResult(
            request_id=request.request_id,
            operation=request.operation,
            mode=request.mode,
            disposition=disposition,
            incident_id=request.incident_id or request.incident_cid(),
            read_only=request.is_read_only
            or request.operation != DoctorOperation.REPAIR.value,
            policy_decision=decision,
            run_receipt=None,
            reason_codes=tuple(reason_codes),
            explanation=explanation,
            changed=False,
            replayed=False,
            status={
                "schema": DOCTOR_SERVICE_STATUS_SCHEMA,
                "incident_id": request.incident_id,
                "llm_router_enabled": False,
                "automatic_fallback": False,
                "optional_providers_loaded": list(optional_providers_loaded()),
            },
            stage_refs={},
        )

    # -- test / operator hooks (explicit only) -----------------------------

    def note_provider_invocation(
        self,
        *,
        llm_router: bool = False,
        remote_model_provider: bool = False,
        model_count: int = 0,
        provider_count: int = 0,
    ) -> None:
        """Record an observed provider invocation (fails subsequent execute).

        Used by adversarial tests that intercept llm_router / model providers.
        There is no automatic fallback once any count or flag is raised.
        """

        if llm_router:
            self._invocation_guard["llm_router_invoked"] = True
        if remote_model_provider:
            self._invocation_guard["remote_model_provider_invoked"] = True
        self._invocation_guard["model_invocation_count"] = int(
            self._invocation_guard["model_invocation_count"]
        ) + int(model_count)
        self._invocation_guard["provider_invocation_count"] = int(
            self._invocation_guard["provider_invocation_count"]
        ) + int(provider_count)


def build_doctor_operation_request(
    operation: str | DoctorOperation,
    **kwargs: Any,
) -> DoctorOperationRequest:
    """Convenience builder for body-free operation requests."""

    payload: dict[str, Any] = {
        "operation": operation.value
        if isinstance(operation, DoctorOperation)
        else operation,
        **kwargs,
    }
    assert_body_free(payload, "operation request")
    return DoctorOperationRequest.from_dict(payload)


def create_deterministic_doctor_service(
    *,
    policy: DeterministicDoctorPolicy | Mapping[str, Any] | None = None,
    receipt_store: DoctorReceiptStore | None = None,
    backends: DoctorStageBackends | None = None,
    control_service: Any | None = None,
    cas: Any | None = None,
) -> DeterministicDoctorService:
    """Factory that never imports optional providers."""

    return DeterministicDoctorService(
        policy=policy,
        receipt_store=receipt_store,
        backends=backends,
        control_service=control_service,
        cas=cas,
    )


__all__ = [
    "ALL_SERVICE_OPERATIONS",
    "DETERMINISTIC_DOCTOR_SERVICE_INTERFACE",
    "DOCTOR_OPERATION_REQUEST_SCHEMA",
    "DOCTOR_OPERATION_RESULT_SCHEMA",
    "DOCTOR_SERVICE_DISCOVERY_SCHEMA",
    "DOCTOR_SERVICE_STATUS_SCHEMA",
    "DoctorOperation",
    "DoctorOperationRequest",
    "DoctorOperationResult",
    "DoctorControlDependency",
    "DoctorControlRequest",
    "DoctorReceiptStore",
    "DoctorServiceCapabilityCode",
    "DoctorServiceError",
    "DoctorServiceSafetyError",
    "DoctorStageBackend",
    "DoctorStageBackends",
    "DeterministicDoctorService",
    "InMemoryDoctorReceiptStore",
    "RuntimeCASDoctorReceiptStore",
    "assert_body_free",
    "assert_no_llm_surface_loaded",
    "build_doctor_operation_request",
    "create_deterministic_doctor_service",
    "is_doctor_operation",
    "is_meta_operation",
    "optional_providers_loaded",
]
