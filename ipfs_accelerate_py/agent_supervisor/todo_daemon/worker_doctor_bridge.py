"""Worker Doctor bridge from validation failures (WPD-030).

Interface: ``WorkerDoctorBridge@1``

Maps typed worker failures (validation, scope, proof, merge admission, path
escape, exact-roots mismatch, contract gap) onto body-free
:class:`~ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_service.DoctorOperationRequest`
records with exact authority roots.

Fail-closed rules:

* Known failure classes produce a read-only Doctor ``inspect`` (or, when
  explicitly plan-eligible, ``plan``) request bound to exact roots.
* Unknown failure classes yield disposition ``abstain_review`` and never
  invent a Doctor request.
* The bridge never opens an LLM / remote model-provider / network client.
* Source bodies and secrets are rejected on the failure surface.
* Mapping is pure: optional Doctor service execution is nomination-only and
  still hard-rejects any LLM surface observation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from ..analysis.deterministic_doctor_contracts import (
    AUTHORITY_ROOT_FIELDS,
    DoctorAuthorityRoots,
    DoctorMode,
    DoctorOperation,
)
from ..control.deterministic_doctor_service import DoctorOperationRequest
from ..proof.formal_verification_contracts import content_identity
from .implementation_disposition import ImplementationDisposition


# ---------------------------------------------------------------------------
# Interface identity
# ---------------------------------------------------------------------------

WORKER_DOCTOR_BRIDGE_INTERFACE: Final[str] = "WorkerDoctorBridge@1"
WORKER_DOCTOR_BRIDGE_VERSION: Final[int] = 1
WORKER_DOCTOR_BRIDGE_EVIDENCE: Final[str] = "wpd/worker-doctor-bridge@1"
WORKER_DOCTOR_BRIDGE_PRODUCER: Final[str] = "worker-doctor-bridge@1"

WORKER_FAILURE_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worker-failure-record@1"
)
WORKER_DOCTOR_BRIDGE_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worker-doctor-bridge-result@1"
)
WORKER_DOCTOR_BRIDGE_DISCOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worker-doctor-bridge/discovery@1"
)

# Reason codes
REASON_KNOWN_FAILURE_INSPECT: Final[str] = "known_failure_doctor_inspect"
REASON_KNOWN_FAILURE_PLAN: Final[str] = "known_failure_doctor_plan"
REASON_UNKNOWN_FAILURE_ABSTAIN: Final[str] = "unknown_failure_abstain_review"
REASON_MISSING_ROOTS: Final[str] = "missing_exact_roots"
REASON_MISSING_FAILURE_CLASS: Final[str] = "missing_failure_class"


# ---------------------------------------------------------------------------
# Closed failure vocabulary + mapping table
# ---------------------------------------------------------------------------


class WorkerFailureClass(str, Enum):
    """Closed worker failure classes admitted onto the Doctor bridge.

    Every member maps to a Doctor operation (see
    :data:`KNOWN_FAILURE_CLASS_MAP`).  Tokens outside this vocabulary yield
    typed ``abstain_review`` rather than a free-form re-prompt.
    """

    VALIDATION_FAILURE = "validation_failure"
    SCOPE_FAILURE = "scope_failure"
    PROOF_FAILURE = "proof_failure"
    MERGE_FAILURE = "merge_failure"
    MERGE_ADMISSION_FAILURE = "merge_admission_failure"
    PATH_ESCAPE = "path_escape"
    EXACT_ROOTS_MISMATCH = "exact_roots_mismatch"
    CONTRACT_GAP = "contract_gap"


# Primary mapping: known failure class → Doctor operation.
# All known classes produce inspect by default (acceptance: "Doctor inspect").
# Contract-gap may elevate to plan when the caller asks for plan and the class
# is plan-eligible; the default map entry remains inspect.
KNOWN_FAILURE_CLASS_MAP: Final[Mapping[WorkerFailureClass, DoctorOperation]] = {
    WorkerFailureClass.VALIDATION_FAILURE: DoctorOperation.INSPECT,
    WorkerFailureClass.SCOPE_FAILURE: DoctorOperation.INSPECT,
    WorkerFailureClass.PROOF_FAILURE: DoctorOperation.INSPECT,
    WorkerFailureClass.MERGE_FAILURE: DoctorOperation.INSPECT,
    WorkerFailureClass.MERGE_ADMISSION_FAILURE: DoctorOperation.INSPECT,
    WorkerFailureClass.PATH_ESCAPE: DoctorOperation.INSPECT,
    WorkerFailureClass.EXACT_ROOTS_MISMATCH: DoctorOperation.INSPECT,
    WorkerFailureClass.CONTRACT_GAP: DoctorOperation.INSPECT,
}

# Classes that may produce a plan request when the caller asks for plan.
PLAN_ELIGIBLE_FAILURE_CLASSES: Final[frozenset[WorkerFailureClass]] = frozenset(
    {
        WorkerFailureClass.VALIDATION_FAILURE,
        WorkerFailureClass.SCOPE_FAILURE,
        WorkerFailureClass.PROOF_FAILURE,
        WorkerFailureClass.CONTRACT_GAP,
        WorkerFailureClass.MERGE_ADMISSION_FAILURE,
    }
)

# Wire-token aliases normalized into the closed vocabulary.
_FAILURE_CLASS_ALIASES: Final[Mapping[str, WorkerFailureClass]] = {
    "validation": WorkerFailureClass.VALIDATION_FAILURE,
    "validation_failure": WorkerFailureClass.VALIDATION_FAILURE,
    "validation_failed": WorkerFailureClass.VALIDATION_FAILURE,
    "validation_signature": WorkerFailureClass.VALIDATION_FAILURE,
    "scope": WorkerFailureClass.SCOPE_FAILURE,
    "scope_failure": WorkerFailureClass.SCOPE_FAILURE,
    "scope_escape": WorkerFailureClass.SCOPE_FAILURE,
    "path_escape": WorkerFailureClass.PATH_ESCAPE,
    "path_escape_to_write": WorkerFailureClass.PATH_ESCAPE,
    "proof": WorkerFailureClass.PROOF_FAILURE,
    "proof_failure": WorkerFailureClass.PROOF_FAILURE,
    "merge": WorkerFailureClass.MERGE_FAILURE,
    "merge_failure": WorkerFailureClass.MERGE_FAILURE,
    "merge_admission": WorkerFailureClass.MERGE_ADMISSION_FAILURE,
    "merge_admission_failure": WorkerFailureClass.MERGE_ADMISSION_FAILURE,
    "admission_failure": WorkerFailureClass.MERGE_ADMISSION_FAILURE,
    "exact_roots": WorkerFailureClass.EXACT_ROOTS_MISMATCH,
    "exact_roots_mismatch": WorkerFailureClass.EXACT_ROOTS_MISMATCH,
    "roots_mismatch": WorkerFailureClass.EXACT_ROOTS_MISMATCH,
    "contract_gap": WorkerFailureClass.CONTRACT_GAP,
    "contract_mismatch": WorkerFailureClass.CONTRACT_GAP,
}

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "ast",
        "ast_body",
        "body",
        "code",
        "content",
        "contents",
        "file_text",
        "prompt",
        "prompt_body",
        "prompt_text",
        "raw_ast",
        "raw_log",
        "snippet",
        "source",
        "source_body",
        "source_text",
        "transcript",
    }
)

_SECRET_KEYS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "session_token",
        "token",
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WorkerDoctorBridgeError(RuntimeError):
    """Fail-closed rejection for an unsafe or incomplete bridge mapping."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "worker_doctor_bridge_error",
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "worker_doctor_bridge_error")


class WorkerDoctorBridgeInputError(WorkerDoctorBridgeError, ValueError):
    """Caller supplied a malformed failure record."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "invalid_failure_input",
    ) -> None:
        super().__init__(message, reason_code=reason_code)


# ---------------------------------------------------------------------------
# Failure record
# ---------------------------------------------------------------------------


def _assert_body_free(value: Any, field_name: str = "record") -> None:
    if isinstance(value, float):
        raise WorkerDoctorBridgeInputError(
            f"{field_name} may not contain floating-point values",
            reason_code="body_or_secret_rejected",
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise WorkerDoctorBridgeInputError(
                    f"{field_name} has a non-string key",
                    reason_code="body_or_secret_rejected",
                )
            normalized = key.lower().replace("-", "_").strip()
            if normalized in _BODY_MARKERS or normalized in _SECRET_KEYS:
                raise WorkerDoctorBridgeInputError(
                    f"{field_name} may not contain secrets or source bodies",
                    reason_code="body_or_secret_rejected",
                )
            if any(marker in normalized for marker in ("password", "private_key", "api_key")):
                raise WorkerDoctorBridgeInputError(
                    f"{field_name} may not contain secrets or source bodies",
                    reason_code="body_or_secret_rejected",
                )
            _assert_body_free(item, field_name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_body_free(item, field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise WorkerDoctorBridgeInputError(
            f"{field_name} may not contain binary bodies",
            reason_code="body_or_secret_rejected",
        )


def _normalize_path(path: str) -> str:
    text = str(path or "").strip().replace("\\", "/")
    if not text:
        raise WorkerDoctorBridgeInputError(
            "write path must be non-empty",
            reason_code="invalid_write_path",
        )
    if text.startswith("/") or ".." in text.split("/"):
        raise WorkerDoctorBridgeInputError(
            f"write path must be a relative repository path: {path!r}",
            reason_code="path_escape",
        )
    return text


def _normalize_ids(values: Sequence[str] | None, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise WorkerDoctorBridgeInputError(
            f"{field_name} must be a sequence of identifiers",
            reason_code="invalid_identifiers",
        )
    out: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        if any(ch.isspace() for ch in text):
            raise WorkerDoctorBridgeInputError(
                f"{field_name} entries must be compact identifiers",
                reason_code="invalid_identifiers",
            )
        if text not in out:
            out.append(text)
    return tuple(out)


def normalize_failure_class(value: Any) -> WorkerFailureClass | None:
    """Return a known :class:`WorkerFailureClass` or ``None`` if unknown.

    Empty / missing tokens raise; unknown non-empty tokens return ``None`` so
    the bridge can abstain without inventing a Doctor request.
    """

    if value is None:
        raise WorkerDoctorBridgeInputError(
            "failure_class is required",
            reason_code=REASON_MISSING_FAILURE_CLASS,
        )
    if isinstance(value, WorkerFailureClass):
        return value
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        raise WorkerDoctorBridgeInputError(
            "failure_class is required",
            reason_code=REASON_MISSING_FAILURE_CLASS,
        )
    if text in _FAILURE_CLASS_ALIASES:
        return _FAILURE_CLASS_ALIASES[text]
    try:
        return WorkerFailureClass(text)
    except ValueError:
        return None


def decode_authority_roots(
    roots: DoctorAuthorityRoots | Mapping[str, Any] | None,
) -> DoctorAuthorityRoots | None:
    """Decode exact Doctor authority roots or return ``None`` when absent."""

    if roots is None:
        return None
    if isinstance(roots, DoctorAuthorityRoots):
        return roots
    if not isinstance(roots, Mapping):
        raise WorkerDoctorBridgeInputError(
            "roots must be DoctorAuthorityRoots or a mapping",
            reason_code="invalid_roots",
        )
    _assert_body_free(roots, "roots")
    try:
        if roots.get("schema"):
            return DoctorAuthorityRoots.from_dict(roots)
        return DoctorAuthorityRoots(
            **{key: roots[key] for key in AUTHORITY_ROOT_FIELDS if key in roots}
        )
    except WorkerDoctorBridgeInputError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize contract errors
        raise WorkerDoctorBridgeInputError(
            f"invalid doctor authority roots: {exc}",
            reason_code="invalid_roots",
        ) from exc


@dataclass(frozen=True)
class WorkerFailureRecord:
    """Body-free typed failure presented to the Doctor bridge.

    Carries only opaque identifiers, exact roots, and compact reason codes.
    Source bodies and free-form task prose are rejected.
    """

    failure_class: str
    task_cid: str = ""
    incident_id: str = ""
    roots: DoctorAuthorityRoots | None = None
    write_paths: tuple[str, ...] = ()
    finding_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    evidence_cids: tuple[str, ...] = ()
    lease_id: str = ""
    checkpoint_ref: str = ""
    target_tree_cid: str = ""
    attempt: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        failure = str(self.failure_class or "").strip()
        if not failure:
            raise WorkerDoctorBridgeInputError(
                "failure_class is required",
                reason_code=REASON_MISSING_FAILURE_CLASS,
            )
        object.__setattr__(self, "failure_class", failure)
        object.__setattr__(self, "task_cid", str(self.task_cid or "").strip())
        object.__setattr__(self, "incident_id", str(self.incident_id or "").strip())
        object.__setattr__(self, "lease_id", str(self.lease_id or "").strip())
        object.__setattr__(
            self, "checkpoint_ref", str(self.checkpoint_ref or "").strip()
        )
        object.__setattr__(
            self, "target_tree_cid", str(self.target_tree_cid or "").strip()
        )
        if int(self.attempt) < 1:
            raise WorkerDoctorBridgeInputError(
                "attempt must be >= 1",
                reason_code="invalid_attempt",
            )
        object.__setattr__(self, "attempt", int(self.attempt))
        roots = self.roots
        if roots is not None and not isinstance(roots, DoctorAuthorityRoots):
            roots = decode_authority_roots(roots)  # type: ignore[arg-type]
        object.__setattr__(self, "roots", roots)
        object.__setattr__(
            self,
            "write_paths",
            tuple(_normalize_path(path) for path in (self.write_paths or ())),
        )
        object.__setattr__(
            self, "finding_ids", _normalize_ids(self.finding_ids, "finding_ids")
        )
        object.__setattr__(
            self, "reason_codes", _normalize_ids(self.reason_codes, "reason_codes")
        )
        object.__setattr__(
            self, "evidence_cids", _normalize_ids(self.evidence_cids, "evidence_cids")
        )
        metadata = dict(self.metadata or {})
        _assert_body_free(metadata, "metadata")
        object.__setattr__(self, "metadata", metadata)

    @property
    def known_class(self) -> WorkerFailureClass | None:
        return normalize_failure_class(self.failure_class)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": WORKER_FAILURE_RECORD_SCHEMA,
            "failure_class": self.failure_class,
            "task_cid": self.task_cid,
            "incident_id": self.incident_id,
            "roots": self.roots.to_dict() if self.roots is not None else None,
            "write_paths": list(self.write_paths),
            "finding_ids": list(self.finding_ids),
            "reason_codes": list(self.reason_codes),
            "evidence_cids": list(self.evidence_cids),
            "lease_id": self.lease_id,
            "checkpoint_ref": self.checkpoint_ref,
            "target_tree_cid": self.target_tree_cid,
            "attempt": self.attempt,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkerFailureRecord":
        if not isinstance(payload, Mapping):
            raise WorkerDoctorBridgeInputError(
                "failure record must be a mapping",
                reason_code="invalid_failure_input",
            )
        _assert_body_free(payload, "failure record")
        data = {
            key: payload[key]
            for key in (
                "failure_class",
                "task_cid",
                "incident_id",
                "roots",
                "write_paths",
                "finding_ids",
                "reason_codes",
                "evidence_cids",
                "lease_id",
                "checkpoint_ref",
                "target_tree_cid",
                "attempt",
                "metadata",
            )
            if key in payload
        }
        if "roots" in data:
            data["roots"] = decode_authority_roots(data["roots"])
        if "write_paths" in data:
            data["write_paths"] = tuple(data["write_paths"] or ())
        for key in ("finding_ids", "reason_codes", "evidence_cids"):
            if key in data:
                data[key] = tuple(data[key] or ())
        return cls(**data)


# ---------------------------------------------------------------------------
# Bridge result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerDoctorBridgeResult:
    """Outcome of :meth:`WorkerDoctorBridge.map_failure`.

    Known failures carry a sealed :class:`DoctorOperationRequest` (inspect by
    default).  Unknown failures carry ``abstain_review`` and no Doctor request.
    ``provider_hook_count`` is always zero — the bridge never opens an LLM.
    """

    known: bool
    failure_class: str
    reason_code: str
    doctor_request: DoctorOperationRequest | None = None
    disposition: ImplementationDisposition | None = None
    doctor_operation: str = ""
    provider_hook_count: int = 0
    mapping_table_hit: bool = False
    evidence_cids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "failure_class", str(self.failure_class or "").strip())
        object.__setattr__(self, "reason_code", str(self.reason_code or "").strip())
        object.__setattr__(
            self, "doctor_operation", str(self.doctor_operation or "").strip()
        )
        object.__setattr__(self, "provider_hook_count", int(self.provider_hook_count))
        object.__setattr__(
            self, "evidence_cids", tuple(str(item) for item in self.evidence_cids)
        )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))
        if self.provider_hook_count != 0:
            raise WorkerDoctorBridgeError(
                "worker doctor bridge must never open a provider/LLM hook",
                reason_code="provider_hook_forbidden",
            )
        if self.known:
            if self.doctor_request is None:
                raise WorkerDoctorBridgeError(
                    "known failure mapping requires a DoctorOperationRequest",
                    reason_code="missing_doctor_request",
                )
            if self.disposition is ImplementationDisposition.ABSTAIN_REVIEW:
                raise WorkerDoctorBridgeError(
                    "known failure must not abstain_review",
                    reason_code="known_failure_disposition",
                )
        else:
            if self.doctor_request is not None:
                raise WorkerDoctorBridgeError(
                    "unknown failure must not invent a DoctorOperationRequest",
                    reason_code="unknown_must_abstain",
                )
            if self.disposition is not ImplementationDisposition.ABSTAIN_REVIEW:
                raise WorkerDoctorBridgeError(
                    "unknown failure must yield abstain_review",
                    reason_code="unknown_must_abstain",
                )

    @property
    def authorizes_provider(self) -> bool:
        return False

    @property
    def abstained(self) -> bool:
        return self.disposition is ImplementationDisposition.ABSTAIN_REVIEW

    @property
    def produces_doctor_inspect(self) -> bool:
        return (
            self.known
            and self.doctor_request is not None
            and self.doctor_request.operation == DoctorOperation.INSPECT.value
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": WORKER_DOCTOR_BRIDGE_RESULT_SCHEMA,
            "interface": WORKER_DOCTOR_BRIDGE_INTERFACE,
            "known": self.known,
            "failure_class": self.failure_class,
            "reason_code": self.reason_code,
            "doctor_request": (
                self.doctor_request.to_dict()
                if self.doctor_request is not None
                else None
            ),
            "disposition": (
                self.disposition.value if self.disposition is not None else None
            ),
            "doctor_operation": self.doctor_operation,
            "provider_hook_count": 0,
            "mapping_table_hit": self.mapping_table_hit,
            "evidence_cids": list(self.evidence_cids),
            "notes": list(self.notes),
            "authorizes_provider": False,
            "llm_router_invoked": False,
            "network_access": False,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


def failure_class_mapping_table() -> dict[str, str]:
    """Return the closed known-failure → Doctor-operation mapping table."""

    return {
        failure.value: operation.value
        for failure, operation in KNOWN_FAILURE_CLASS_MAP.items()
    }


def known_failure_classes() -> frozenset[str]:
    """Return the closed set of known failure-class wire tokens."""

    return frozenset(item.value for item in WorkerFailureClass)


@dataclass
class WorkerDoctorBridge:
    """Map typed worker failures onto Doctor inspect/plan operation records.

    Pure mapping surface: never loads LLM clients, never re-prompts free-form
    task bodies, and never invents Doctor operation records for unknown classes.
    """

    require_exact_roots_for_known: bool = True
    default_mode: DoctorMode = DoctorMode.REPORT_ONLY

    @classmethod
    def discovery(cls) -> dict[str, Any]:
        return {
            "schema": WORKER_DOCTOR_BRIDGE_DISCOVERY_SCHEMA,
            "interface": WORKER_DOCTOR_BRIDGE_INTERFACE,
            "version": WORKER_DOCTOR_BRIDGE_VERSION,
            "evidence_key": WORKER_DOCTOR_BRIDGE_EVIDENCE,
            "known_failure_classes": sorted(known_failure_classes()),
            "mapping_table": failure_class_mapping_table(),
            "plan_eligible_failure_classes": sorted(
                item.value for item in PLAN_ELIGIBLE_FAILURE_CLASSES
            ),
            "unknown_disposition": ImplementationDisposition.ABSTAIN_REVIEW.value,
            "llm_router_enabled": False,
            "automatic_fallback": False,
            "network_access": False,
            "provider_hooks": 0,
        }

    def map_failure(
        self,
        failure: WorkerFailureRecord | Mapping[str, Any],
        *,
        operation: str | DoctorOperation | None = None,
    ) -> WorkerDoctorBridgeResult:
        """Map one typed failure to a Doctor request or abstain_review.

        Parameters
        ----------
        failure:
            Body-free failure record (or mapping).
        operation:
            Optional Doctor operation override.  Defaults to the mapping-table
            entry (``inspect``).  ``plan`` is admitted only for plan-eligible
            known classes.
        """

        record = self._normalize(failure)
        known = record.known_class

        if known is None:
            return WorkerDoctorBridgeResult(
                known=False,
                failure_class=record.failure_class,
                reason_code=REASON_UNKNOWN_FAILURE_ABSTAIN,
                doctor_request=None,
                disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                doctor_operation="",
                provider_hook_count=0,
                mapping_table_hit=False,
                evidence_cids=record.evidence_cids,
                notes=("unknown_failure_class", "typed_abstention"),
            )

        if self.require_exact_roots_for_known and record.roots is None:
            # Missing exact roots cannot produce a legitimate Doctor request;
            # fail closed to abstain rather than invent unbound authority.
            return WorkerDoctorBridgeResult(
                known=False,
                failure_class=record.failure_class,
                reason_code=REASON_MISSING_ROOTS,
                doctor_request=None,
                disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                doctor_operation="",
                provider_hook_count=0,
                mapping_table_hit=True,
                evidence_cids=record.evidence_cids,
                notes=("known_class_without_exact_roots", "typed_abstention"),
            )

        doctor_op = self._resolve_operation(known, operation)
        mode = (
            DoctorMode.PLAN
            if doctor_op is DoctorOperation.PLAN
            else self.default_mode
        )
        reason_codes = self._compose_reason_codes(record, known)
        incident_id = record.incident_id or self._default_incident_id(record, known)
        request = DoctorOperationRequest(
            operation=doctor_op.value,
            mode=mode,
            incident_id=incident_id,
            roots=record.roots,
            lease_id=record.lease_id
            or (record.roots.lease_id if record.roots is not None else ""),
            checkpoint_ref=record.checkpoint_ref,
            target_tree_cid=record.target_tree_cid,
            write_paths=record.write_paths,
            finding_ids=record.finding_ids,
            reason_codes=reason_codes,
            llm_router_invoked=False,
            remote_model_provider_invoked=False,
            model_invocation_count=0,
            provider_invocation_count=0,
            network_access=False,
            target_code_imported=False,
            semantic_authority_flags={
                "worker_doctor_bridge": True,
                "failure_class": known.value,
                "task_cid": record.task_cid or "",
            },
        )
        # Safety observation: request must stay hard-off for LLM surfaces.
        if (
            request.llm_router_invoked
            or request.remote_model_provider_invoked
            or request.network_access
            or request.model_invocation_count
            or request.provider_invocation_count
        ):
            raise WorkerDoctorBridgeError(
                "DoctorOperationRequest must not observe LLM/network surfaces",
                reason_code="llm_surface_forbidden",
            )

        reason = (
            REASON_KNOWN_FAILURE_PLAN
            if doctor_op is DoctorOperation.PLAN
            else REASON_KNOWN_FAILURE_INSPECT
        )
        return WorkerDoctorBridgeResult(
            known=True,
            failure_class=known.value,
            reason_code=reason,
            doctor_request=request,
            disposition=None,
            doctor_operation=doctor_op.value,
            provider_hook_count=0,
            mapping_table_hit=True,
            evidence_cids=record.evidence_cids,
            notes=("mapping_table_hit", f"operation:{doctor_op.value}"),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize(
        self,
        failure: WorkerFailureRecord | Mapping[str, Any],
    ) -> WorkerFailureRecord:
        if isinstance(failure, WorkerFailureRecord):
            return failure
        if not isinstance(failure, Mapping):
            raise WorkerDoctorBridgeInputError(
                "failure must be WorkerFailureRecord or mapping",
                reason_code="invalid_failure_input",
            )
        return WorkerFailureRecord.from_dict(failure)

    def _resolve_operation(
        self,
        known: WorkerFailureClass,
        operation: str | DoctorOperation | None,
    ) -> DoctorOperation:
        default = KNOWN_FAILURE_CLASS_MAP[known]
        if operation is None:
            return default
        if isinstance(operation, DoctorOperation):
            resolved = operation
        else:
            token = str(operation).strip().lower()
            try:
                resolved = DoctorOperation(token)
            except ValueError as exc:
                raise WorkerDoctorBridgeInputError(
                    f"unsupported doctor operation override: {operation!r}",
                    reason_code="invalid_operation",
                ) from exc
        if resolved is DoctorOperation.INSPECT:
            return DoctorOperation.INSPECT
        if resolved is DoctorOperation.PLAN:
            if known not in PLAN_ELIGIBLE_FAILURE_CLASSES:
                raise WorkerDoctorBridgeInputError(
                    f"failure class {known.value!r} is not plan-eligible",
                    reason_code="plan_not_eligible",
                )
            return DoctorOperation.PLAN
        # Bridge only admits inspect/plan; repair would require a permit path.
        raise WorkerDoctorBridgeInputError(
            "worker doctor bridge only admits inspect or plan operations",
            reason_code="operation_not_admitted",
        )

    def _compose_reason_codes(
        self,
        record: WorkerFailureRecord,
        known: WorkerFailureClass,
    ) -> tuple[str, ...]:
        codes: list[str] = [known.value, "worker_doctor_bridge"]
        for code in record.reason_codes:
            if code not in codes:
                codes.append(code)
        return tuple(codes)

    def _default_incident_id(
        self,
        record: WorkerFailureRecord,
        known: WorkerFailureClass,
    ) -> str:
        return content_identity(
            {
                "schema": "worker-doctor-bridge-incident@1",
                "failure_class": known.value,
                "task_cid": record.task_cid,
                "attempt": record.attempt,
                "roots_id": record.roots.content_id if record.roots is not None else "",
            }
        )


def build_worker_doctor_bridge(
    *,
    require_exact_roots_for_known: bool = True,
) -> WorkerDoctorBridge:
    """Construct the production-default worker Doctor bridge."""

    return WorkerDoctorBridge(
        require_exact_roots_for_known=require_exact_roots_for_known,
    )


def map_worker_failure(
    failure: WorkerFailureRecord | Mapping[str, Any],
    *,
    operation: str | DoctorOperation | None = None,
    require_exact_roots_for_known: bool = True,
) -> WorkerDoctorBridgeResult:
    """Module-level convenience wrapper around :meth:`WorkerDoctorBridge.map_failure`."""

    return build_worker_doctor_bridge(
        require_exact_roots_for_known=require_exact_roots_for_known,
    ).map_failure(failure, operation=operation)


__all__ = [
    "KNOWN_FAILURE_CLASS_MAP",
    "PLAN_ELIGIBLE_FAILURE_CLASSES",
    "REASON_KNOWN_FAILURE_INSPECT",
    "REASON_KNOWN_FAILURE_PLAN",
    "REASON_MISSING_ROOTS",
    "REASON_UNKNOWN_FAILURE_ABSTAIN",
    "WORKER_DOCTOR_BRIDGE_EVIDENCE",
    "WORKER_DOCTOR_BRIDGE_INTERFACE",
    "WORKER_DOCTOR_BRIDGE_PRODUCER",
    "WORKER_DOCTOR_BRIDGE_VERSION",
    "WORKER_FAILURE_RECORD_SCHEMA",
    "WorkerDoctorBridge",
    "WorkerDoctorBridgeError",
    "WorkerDoctorBridgeInputError",
    "WorkerDoctorBridgeResult",
    "WorkerFailureClass",
    "WorkerFailureRecord",
    "build_worker_doctor_bridge",
    "decode_authority_roots",
    "failure_class_mapping_table",
    "known_failure_classes",
    "map_worker_failure",
    "normalize_failure_class",
]
