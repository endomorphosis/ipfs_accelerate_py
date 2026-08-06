"""Implementation disposition and dual-view pre-implementation kernel contracts.

WPD-001 / ``ImplementationDisposition@1`` / ``PreImplementationKernelReceipt@1``

This module is a leaf serialization boundary shared by workers and the
supervisor.  It defines:

* the closed :class:`ImplementationDisposition` vocabulary;
* forest-root bindings for the exact repository observation authority;
* the dual-view kernel contract (planner and doctor as views of one
  obligation graph); and
* the durable :class:`PreImplementationKernelReceipt` that workers must emit
  before any provider call.

Authority rules (fail-closed):

* Only ``residual_llm_authorized`` may authorize a provider / LLM invocation.
* Receipts bind ``task_cid``, forest roots, and plan/doctor CIDs exactly.
* Planner and doctor CIDs are dual views of one shared obligation graph.
* Residual authorization requires a sealed residual-packet CID; other
  dispositions forbid residual packets.
* Unknown fields, forged content identities, unknown dispositions, floats,
  source bodies, and root/view mismatches reject at construction.
* No scanner, model provider, daemon runtime, or network client is imported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


# ---------------------------------------------------------------------------
# Version, schemas, bounds
# ---------------------------------------------------------------------------

IMPLEMENTATION_DISPOSITION_VERSION: Final[int] = 1
CONTRACT_VERSION: Final[int] = IMPLEMENTATION_DISPOSITION_VERSION
SCHEMA_VERSION: Final[int] = IMPLEMENTATION_DISPOSITION_VERSION

IMPLEMENTATION_DISPOSITION_INTERFACE: Final[str] = "ImplementationDisposition@1"
PRE_IMPLEMENTATION_KERNEL_RECEIPT_INTERFACE: Final[str] = (
    "PreImplementationKernelReceipt@1"
)
DUAL_VIEW_KERNEL_INTERFACE: Final[str] = "DualViewKernelContract@1"

SCHEMA_PREFIX: Final[str] = "ipfs_accelerate_py/agent-supervisor"
IMPLEMENTATION_DISPOSITION_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/implementation-disposition@1"
)
IMPLEMENTATION_FOREST_ROOTS_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/implementation-forest-roots@1"
)
DUAL_VIEW_KERNEL_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/dual-view-kernel-contract@1"
)
PRE_IMPLEMENTATION_KERNEL_RECEIPT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/pre-implementation-kernel-receipt@1"
)

MAX_RECORD_BYTES: Final[int] = 262_144
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_REFERENCE_COUNT: Final[int] = 256
MAX_ATTEMPT: Final[int] = 2**31 - 1

# Evidence key published with durable disposition receipts.
IMPLEMENTATION_DISPOSITION_EVIDENCE: Final[str] = "wpd/implementation-disposition@1"


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ImplementationDisposition(str, Enum):
    """Closed pre-implementation outcomes for a claimed worker task.

    * ``closed_deterministic`` — analytical / Doctor / planner path closed the
      work without any model provider call.
    * ``residual_llm_authorized`` — only sealed residual-packet provider use
      is authorized; free-form rediscovery is still forbidden.
    * ``abstain_review`` — typed residual for an operator; no provider call.
    * ``defer_capability`` — a required optional backend is unavailable; no
      provider call and no silent success.
    """

    CLOSED_DETERMINISTIC = "closed_deterministic"
    RESIDUAL_LLM_AUTHORIZED = "residual_llm_authorized"
    ABSTAIN_REVIEW = "abstain_review"
    DEFER_CAPABILITY = "defer_capability"

    @property
    def authorizes_provider(self) -> bool:
        """Whether this disposition may invoke a model provider."""

        return self is ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED

    @property
    def is_terminal_without_provider(self) -> bool:
        """Whether the attempt records a result without a provider call."""

        return self is not ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED


class KernelViewKind(str, Enum):
    """Dual views of one pre-implementation obligation kernel."""

    PLANNER = "planner"
    DOCTOR = "doctor"


class ResidualRequirement(str, Enum):
    """Whether a residual packet is required, forbidden, or optional."""

    REQUIRED = "required"
    FORBIDDEN = "forbidden"
    OPTIONAL = "optional"


_CLOSED_DISPOSITIONS: Final[frozenset[ImplementationDisposition]] = frozenset(
    ImplementationDisposition
)

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "source",
        "source_body",
        "source_text",
        "contents",
        "content",
        "snippet",
        "code",
        "file_text",
        "raw_ast",
        "ast_body",
        "prompt",
        "prompt_body",
        "prompt_text",
        "transcript",
        "raw_log",
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


class ImplementationDispositionError(ContractValidationError):
    """Malformed disposition / kernel-receipt contract."""


class ImplementationDispositionBoundsError(ImplementationDispositionError):
    """A count, byte, or integer limit exceeds a hard bound."""


class ForgedImplementationDispositionIdentityError(ImplementationDispositionError):
    """A stored content identity did not match the canonical preimage."""


class ImplementationDispositionAuthorityError(ImplementationDispositionError):
    """Task, forest, plan/doctor, or residual bindings did not match exactly."""


class UnauthorizedProviderInvocationError(ImplementationDispositionAuthorityError):
    """A provider call was requested under a non-authorizing disposition."""


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = False,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        normalized = ""
    elif not isinstance(value, str):
        raise ImplementationDispositionError(f"{field_name} must be a string")
    else:
        normalized = value.strip()
    if required and not normalized:
        raise ImplementationDispositionError(f"{field_name} is required")
    if len(normalized.encode("utf-8")) > limit:
        raise ImplementationDispositionBoundsError(
            f"{field_name} exceeds its byte bound"
        )
    return normalized


def _identifier(value: Any, field_name: str, *, required: bool = True) -> str:
    result = _text(value, field_name, required=required)
    if not result and not required:
        return ""
    if any(char.isspace() for char in result):
        raise ImplementationDispositionError(
            f"{field_name} must be an opaque compact identifier"
        )
    return result


def _bounded_int(
    value: Any,
    field_name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_ATTEMPT,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ImplementationDispositionError(
            f"{field_name} must be a finite integer"
        )
    if value < minimum or value > maximum:
        raise ImplementationDispositionBoundsError(
            f"{field_name} is outside the supported bound"
        )
    return value


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise ImplementationDispositionError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _ids(
    values: Any,
    field_name: str,
    *,
    required: bool = False,
    limit: int = MAX_REFERENCE_COUNT,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str) or not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray)
    ):
        raise ImplementationDispositionError(
            f"{field_name} must be a sequence of identifiers"
        )
    else:
        raw = values
    if len(raw) > limit:
        raise ImplementationDispositionBoundsError(
            f"{field_name} exceeds its item bound"
        )
    result = tuple(sorted({_identifier(item, field_name) for item in raw}))
    if required and not result:
        raise ImplementationDispositionError(f"{field_name} must not be empty")
    return result


def _secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _SECRET_KEYS or any(
        marker in normalized
        for marker in ("password", "private_key", "access_token", "api_key")
    )


def _assert_body_free(value: Any, field_name: str = "record") -> None:
    """Reject source bodies and secrets even when smuggled through mappings."""

    if isinstance(value, float):
        raise ImplementationDispositionError(
            f"{field_name} may not contain floating-point values"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ImplementationDispositionError(
                    f"{field_name} has a non-string key"
                )
            normalized = key.lower().replace("-", "_").strip()
            if normalized in _BODY_MARKERS or _secret_key(normalized):
                raise ImplementationDispositionError(
                    f"{field_name} may not contain secrets or source bodies"
                )
            _assert_body_free(item, field_name)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _assert_body_free(item, field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise ImplementationDispositionError(
            f"{field_name} may not contain binary bodies"
        )


def _bounded(record: CanonicalContract, name: str) -> None:
    payload = record.to_dict()
    _assert_body_free(payload, name)
    if len(canonical_json_bytes(payload)) > MAX_RECORD_BYTES:
        raise ImplementationDispositionBoundsError(
            f"{name} exceeds its serialized byte bound"
        )


def _verify_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    """Fail closed when any supplied identity key disagrees with the record.

    Both ``content_id`` and ``cid`` are accepted wire spellings.  Either key,
    when present and non-empty, must equal the canonical identity.  Checking
    only the first present key would let a forged alternate spelling pass.
    """

    expected = record.content_id
    for key in ("content_id", "cid"):
        if key not in payload:
            continue
        supplied = payload[key]
        if supplied in (None, ""):
            continue
        if not isinstance(supplied, str) or supplied != expected:
            raise ForgedImplementationDispositionIdentityError(
                "stored content identity does not match the canonical record"
            )


def _decode_fields(
    payload: Mapping[str, Any],
    schema: str,
    fields: Sequence[str],
    name: str,
) -> dict[str, Any]:
    """Fail-closed decoder shared by every externally supplied record."""

    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise ImplementationDispositionError(f"{name} has an unsupported schema")
    if payload.get("contract_version") not in (
        None,
        IMPLEMENTATION_DISPOSITION_VERSION,
    ):
        raise ImplementationDispositionError(
            f"{name} has an unsupported contract version"
        )
    # Body/secret markers must fail with their dedicated message even when the
    # key is also outside the allowed field set (e.g. smuggled prompt_body).
    _assert_body_free(payload, name)
    allowed = set(fields) | {"schema", "contract_version", "content_id", "cid"}
    unknown = set(payload).difference(allowed)
    if unknown:
        raise ImplementationDispositionError(
            f"{name} contains unsupported fields: "
            + ", ".join(sorted(unknown))
        )
    return {
        field_name: payload[field_name]
        for field_name in fields
        if field_name in payload
    }


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def closed_dispositions() -> frozenset[ImplementationDisposition]:
    """Return the closed disposition vocabulary."""

    return _CLOSED_DISPOSITIONS


def closed_disposition_values() -> frozenset[str]:
    """Return the closed disposition wire values."""

    return frozenset(item.value for item in ImplementationDisposition)


def parse_implementation_disposition(
    value: Any,
) -> ImplementationDisposition:
    """Parse a disposition value or fail closed on unknown tokens."""

    return _enum(  # type: ignore[return-value]
        value, ImplementationDisposition, "disposition"
    )


def residual_requirement_for(
    disposition: ImplementationDisposition | str,
) -> ResidualRequirement:
    """Return whether a residual packet is required under ``disposition``."""

    normalized = parse_implementation_disposition(disposition)
    if normalized is ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED:
        return ResidualRequirement.REQUIRED
    return ResidualRequirement.FORBIDDEN


def provider_invocation_authorized(
    disposition: ImplementationDisposition | str,
) -> bool:
    """Return whether a provider call is authorized under ``disposition``."""

    return parse_implementation_disposition(disposition).authorizes_provider


def assert_provider_invocation_allowed(
    disposition: ImplementationDisposition | str,
    *,
    residual_packet_cid: str = "",
) -> ImplementationDisposition:
    """Fail closed unless disposition and residual packet authorize a provider.

    Returns the normalized disposition on success.
    """

    normalized = parse_implementation_disposition(disposition)
    if not normalized.authorizes_provider:
        raise UnauthorizedProviderInvocationError(
            "provider invocation requires disposition residual_llm_authorized; "
            f"got {normalized.value}"
        )
    packet = _identifier(
        residual_packet_cid, "residual_packet_cid", required=False
    )
    if not packet:
        raise ImplementationDispositionAuthorityError(
            "residual_llm_authorized requires a sealed residual_packet_cid"
        )
    return normalized


def expected_provider_call_count(
    disposition: ImplementationDisposition | str,
) -> int:
    """Return the metric floor for provider calls under ``disposition``.

    Non-residual dispositions attribute zero provider calls.  Residual
    authorization does not invent a call count; callers record observed usage.
    """

    if provider_invocation_authorized(disposition):
        return -1  # authorized but not yet observed; never pretend zero-success
    return 0


def disposition_metric_labels(
    disposition: ImplementationDisposition | str,
) -> dict[str, str]:
    """Stable metric labels for closed disposition attribution."""

    normalized = parse_implementation_disposition(disposition)
    return {
        "disposition": normalized.value,
        "provider_authorized": (
            "true" if normalized.authorizes_provider else "false"
        ),
        "residual_requirement": residual_requirement_for(normalized).value,
    }


def implementation_disposition_cid(value: Any) -> str:
    """Return a CIDv1 identity for an arbitrary JSON-compatible value."""

    return content_identity(value)


# ---------------------------------------------------------------------------
# Nested contract records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImplementationForestRoots(CanonicalContract):
    """Exact forest observation roots bound into a kernel receipt.

    Drift of any bound root invalidates the disposition.  Roots are opaque
    compact identifiers (typically CIDv1 or namespaced digests); this module
    does not re-open repository scanning.
    """

    SCHEMA: ClassVar[str] = IMPLEMENTATION_FOREST_ROOTS_SCHEMA

    repository_id: str
    repository_forest_cid: str
    git_tree_id: str
    policy_root: str
    dirty_overlay_cid: str = ""
    capability_catalog_root: str = ""
    configuration_root: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _identifier(self.repository_id, "repository_id")
        )
        object.__setattr__(
            self,
            "repository_forest_cid",
            _identifier(self.repository_forest_cid, "repository_forest_cid"),
        )
        object.__setattr__(
            self, "git_tree_id", _identifier(self.git_tree_id, "git_tree_id")
        )
        object.__setattr__(
            self, "policy_root", _identifier(self.policy_root, "policy_root")
        )
        object.__setattr__(
            self,
            "dirty_overlay_cid",
            _identifier(
                self.dirty_overlay_cid, "dirty_overlay_cid", required=False
            ),
        )
        object.__setattr__(
            self,
            "capability_catalog_root",
            _identifier(
                self.capability_catalog_root,
                "capability_catalog_root",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "configuration_root",
            _identifier(
                self.configuration_root, "configuration_root", required=False
            ),
        )
        _bounded(self, "implementation forest roots")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": IMPLEMENTATION_DISPOSITION_VERSION,
            "repository_id": self.repository_id,
            "repository_forest_cid": self.repository_forest_cid,
            "git_tree_id": self.git_tree_id,
            "policy_root": self.policy_root,
            "dirty_overlay_cid": self.dirty_overlay_cid,
            "capability_catalog_root": self.capability_catalog_root,
            "configuration_root": self.configuration_root,
        }

    def matches(self, other: "ImplementationForestRoots") -> bool:
        return self.content_id == other.content_id

    def require_current(self, expected: "ImplementationForestRoots") -> None:
        if not self.matches(expected):
            raise ImplementationDispositionAuthorityError(
                "forest roots are stale relative to the expected snapshot"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImplementationForestRoots":
        names = (
            "repository_id",
            "repository_forest_cid",
            "git_tree_id",
            "policy_root",
            "dirty_overlay_cid",
            "capability_catalog_root",
            "configuration_root",
        )
        value = cls(
            **_decode_fields(
                payload, cls.SCHEMA, names, "implementation forest roots"
            )
        )
        _verify_identity(payload, value)
        return value


def _forest_roots(value: Any) -> ImplementationForestRoots:
    if isinstance(value, ImplementationForestRoots):
        return value
    if isinstance(value, Mapping):
        if "schema" in value:
            return ImplementationForestRoots.from_dict(value)
        return ImplementationForestRoots(**value)
    raise ImplementationDispositionError(
        "forest_roots must be ImplementationForestRoots"
    )


@dataclass(frozen=True)
class DualViewKernelContract(CanonicalContract):
    """Planner and Doctor as dual views of one obligation kernel.

    Both views must bind the same ``obligation_graph_cid``.  The planner view
    compiles desired behavior into obligations; the doctor view diagnoses
    observed mismatch against the same obligation graph.  Neither view grants
    mutation or completion authority by itself.
    """

    SCHEMA: ClassVar[str] = DUAL_VIEW_KERNEL_SCHEMA

    obligation_graph_cid: str
    plan_cid: str
    doctor_cid: str
    planner_view_kind: KernelViewKind = KernelViewKind.PLANNER
    doctor_view_kind: KernelViewKind = KernelViewKind.DOCTOR
    shared_validation_command_cids: tuple[str, ...] = ()
    shared_edit_packet_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "obligation_graph_cid",
            _identifier(self.obligation_graph_cid, "obligation_graph_cid"),
        )
        object.__setattr__(self, "plan_cid", _identifier(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self, "doctor_cid", _identifier(self.doctor_cid, "doctor_cid")
        )
        object.__setattr__(
            self,
            "planner_view_kind",
            _enum(self.planner_view_kind, KernelViewKind, "planner_view_kind"),
        )
        object.__setattr__(
            self,
            "doctor_view_kind",
            _enum(self.doctor_view_kind, KernelViewKind, "doctor_view_kind"),
        )
        if self.planner_view_kind is not KernelViewKind.PLANNER:
            raise ImplementationDispositionAuthorityError(
                "planner_view_kind must be planner"
            )
        if self.doctor_view_kind is not KernelViewKind.DOCTOR:
            raise ImplementationDispositionAuthorityError(
                "doctor_view_kind must be doctor"
            )
        if self.plan_cid == self.doctor_cid:
            raise ImplementationDispositionAuthorityError(
                "plan_cid and doctor_cid must be distinct dual-view identities"
            )
        object.__setattr__(
            self,
            "shared_validation_command_cids",
            _ids(
                self.shared_validation_command_cids,
                "shared_validation_command_cids",
            ),
        )
        object.__setattr__(
            self,
            "shared_edit_packet_cids",
            _ids(self.shared_edit_packet_cids, "shared_edit_packet_cids"),
        )
        _bounded(self, "dual-view kernel contract")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": IMPLEMENTATION_DISPOSITION_VERSION,
            "obligation_graph_cid": self.obligation_graph_cid,
            "plan_cid": self.plan_cid,
            "doctor_cid": self.doctor_cid,
            "planner_view_kind": self.planner_view_kind.value,
            "doctor_view_kind": self.doctor_view_kind.value,
            "shared_validation_command_cids": list(
                self.shared_validation_command_cids
            ),
            "shared_edit_packet_cids": list(self.shared_edit_packet_cids),
        }

    def binds_plan(self, plan_cid: str) -> bool:
        return self.plan_cid == _identifier(plan_cid, "plan_cid")

    def binds_doctor(self, doctor_cid: str) -> bool:
        return self.doctor_cid == _identifier(doctor_cid, "doctor_cid")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DualViewKernelContract":
        names = (
            "obligation_graph_cid",
            "plan_cid",
            "doctor_cid",
            "planner_view_kind",
            "doctor_view_kind",
            "shared_validation_command_cids",
            "shared_edit_packet_cids",
        )
        value = cls(
            **_decode_fields(payload, cls.SCHEMA, names, "dual-view kernel contract")
        )
        _verify_identity(payload, value)
        return value


def _dual_view(value: Any) -> DualViewKernelContract:
    if isinstance(value, DualViewKernelContract):
        return value
    if isinstance(value, Mapping):
        if "schema" in value:
            return DualViewKernelContract.from_dict(value)
        return DualViewKernelContract(**value)
    raise ImplementationDispositionError(
        "dual_view must be DualViewKernelContract"
    )


@dataclass(frozen=True)
class PreImplementationKernelReceipt(CanonicalContract):
    """Durable content-addressed result of pre-implementation kernel evaluation.

    Workers and the supervisor share this record.  It binds the claimed task,
    exact forest roots, planner/doctor dual-view CIDs, and the closed
    disposition that gates provider invocation.
    """

    SCHEMA: ClassVar[str] = PRE_IMPLEMENTATION_KERNEL_RECEIPT_SCHEMA

    task_cid: str
    disposition: ImplementationDisposition
    forest_roots: ImplementationForestRoots
    plan_cid: str
    doctor_cid: str
    dual_view: DualViewKernelContract
    attempt: int = 1
    residual_packet_cid: str = ""
    reason_code: str = ""
    evidence_cids: tuple[str, ...] = ()
    policy_revision: str = ""
    producer_id: str = "pre-implementation-kernel@1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "task_cid", _identifier(self.task_cid, "task_cid")
        )
        object.__setattr__(
            self,
            "disposition",
            parse_implementation_disposition(self.disposition),
        )
        if not isinstance(self.forest_roots, ImplementationForestRoots):
            object.__setattr__(self, "forest_roots", _forest_roots(self.forest_roots))
        if not isinstance(self.dual_view, DualViewKernelContract):
            object.__setattr__(self, "dual_view", _dual_view(self.dual_view))
        object.__setattr__(self, "plan_cid", _identifier(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self, "doctor_cid", _identifier(self.doctor_cid, "doctor_cid")
        )
        object.__setattr__(
            self, "attempt", _bounded_int(self.attempt, "attempt", minimum=1)
        )
        object.__setattr__(
            self,
            "residual_packet_cid",
            _identifier(
                self.residual_packet_cid, "residual_packet_cid", required=False
            ),
        )
        object.__setattr__(
            self,
            "reason_code",
            _identifier(self.reason_code, "reason_code", required=False),
        )
        object.__setattr__(
            self, "evidence_cids", _ids(self.evidence_cids, "evidence_cids")
        )
        object.__setattr__(
            self,
            "policy_revision",
            _identifier(self.policy_revision, "policy_revision", required=False),
        )
        object.__setattr__(
            self,
            "producer_id",
            _identifier(self.producer_id, "producer_id"),
        )

        # Dual-view identity must match the receipt's plan/doctor bindings.
        if self.dual_view.plan_cid != self.plan_cid:
            raise ImplementationDispositionAuthorityError(
                "plan_cid must match dual_view.plan_cid"
            )
        if self.dual_view.doctor_cid != self.doctor_cid:
            raise ImplementationDispositionAuthorityError(
                "doctor_cid must match dual_view.doctor_cid"
            )

        # Residual packet gating is fail-closed by disposition.
        requirement = residual_requirement_for(self.disposition)
        if requirement is ResidualRequirement.REQUIRED and not self.residual_packet_cid:
            raise ImplementationDispositionAuthorityError(
                "residual_llm_authorized requires residual_packet_cid"
            )
        if (
            requirement is ResidualRequirement.FORBIDDEN
            and self.residual_packet_cid
        ):
            raise ImplementationDispositionAuthorityError(
                f"{self.disposition.value} forbids residual_packet_cid"
            )

        _bounded(self, "pre-implementation kernel receipt")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": IMPLEMENTATION_DISPOSITION_VERSION,
            "task_cid": self.task_cid,
            "disposition": self.disposition.value,
            "forest_roots": self.forest_roots.to_dict(),
            "plan_cid": self.plan_cid,
            "doctor_cid": self.doctor_cid,
            "dual_view": self.dual_view.to_dict(),
            "attempt": self.attempt,
            "residual_packet_cid": self.residual_packet_cid,
            "reason_code": self.reason_code,
            "evidence_cids": list(self.evidence_cids),
            "policy_revision": self.policy_revision,
            "producer_id": self.producer_id,
        }

    @property
    def authorizes_provider(self) -> bool:
        return self.disposition.authorizes_provider

    @property
    def repository_forest_cid(self) -> str:
        return self.forest_roots.repository_forest_cid

    def require_provider_gate(self) -> str:
        """Return residual_packet_cid when provider invocation is authorized."""

        assert_provider_invocation_allowed(
            self.disposition,
            residual_packet_cid=self.residual_packet_cid,
        )
        return self.residual_packet_cid

    def metric_labels(self) -> dict[str, str]:
        labels = disposition_metric_labels(self.disposition)
        labels["task_cid"] = self.task_cid
        labels["repository_forest_cid"] = self.repository_forest_cid
        return labels

    def matches_task(self, task_cid: str) -> bool:
        return self.task_cid == _identifier(task_cid, "task_cid")

    def matches_forest(self, roots: ImplementationForestRoots) -> bool:
        return self.forest_roots.matches(roots)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PreImplementationKernelReceipt":
        names = (
            "task_cid",
            "disposition",
            "forest_roots",
            "plan_cid",
            "doctor_cid",
            "dual_view",
            "attempt",
            "residual_packet_cid",
            "reason_code",
            "evidence_cids",
            "policy_revision",
            "producer_id",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, names, "pre-implementation kernel receipt"
        )
        if "forest_roots" in values:
            values["forest_roots"] = _forest_roots(values["forest_roots"])
        if "dual_view" in values:
            values["dual_view"] = _dual_view(values["dual_view"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


def seal_pre_implementation_kernel_receipt(
    *,
    task_cid: str,
    disposition: ImplementationDisposition | str,
    forest_roots: ImplementationForestRoots | Mapping[str, Any],
    plan_cid: str,
    doctor_cid: str,
    obligation_graph_cid: str,
    attempt: int = 1,
    residual_packet_cid: str = "",
    reason_code: str = "",
    evidence_cids: Sequence[str] | None = None,
    policy_revision: str = "",
    shared_validation_command_cids: Sequence[str] | None = None,
    shared_edit_packet_cids: Sequence[str] | None = None,
    producer_id: str = "pre-implementation-kernel@1",
) -> PreImplementationKernelReceipt:
    """Construct a sealed kernel receipt from dual-view plan/doctor identities."""

    roots = _forest_roots(forest_roots)
    dual_view = DualViewKernelContract(
        obligation_graph_cid=obligation_graph_cid,
        plan_cid=plan_cid,
        doctor_cid=doctor_cid,
        shared_validation_command_cids=tuple(shared_validation_command_cids or ()),
        shared_edit_packet_cids=tuple(shared_edit_packet_cids or ()),
    )
    return PreImplementationKernelReceipt(
        task_cid=task_cid,
        disposition=disposition,  # type: ignore[arg-type]
        forest_roots=roots,
        plan_cid=plan_cid,
        doctor_cid=doctor_cid,
        dual_view=dual_view,
        attempt=attempt,
        residual_packet_cid=residual_packet_cid,
        reason_code=reason_code,
        evidence_cids=tuple(evidence_cids or ()),
        policy_revision=policy_revision,
        producer_id=producer_id,
    )


def verify_pre_implementation_kernel_receipt(
    payload: Mapping[str, Any] | PreImplementationKernelReceipt,
    *,
    expected_task_cid: str | None = None,
    expected_forest_roots: ImplementationForestRoots | None = None,
    require_provider: bool = False,
) -> PreImplementationKernelReceipt:
    """Decode and optionally re-bind a receipt against expected authority.

    Forged content identities, unknown fields, and root/task mismatches fail
    closed.
    """

    if isinstance(payload, PreImplementationKernelReceipt):
        receipt = payload
    else:
        receipt = PreImplementationKernelReceipt.from_dict(payload)

    if expected_task_cid is not None and not receipt.matches_task(expected_task_cid):
        raise ImplementationDispositionAuthorityError(
            "receipt task_cid does not match the expected task"
        )
    if expected_forest_roots is not None:
        receipt.forest_roots.require_current(expected_forest_roots)
    if require_provider:
        receipt.require_provider_gate()
    return receipt


__all__ = [
    "CONTRACT_VERSION",
    "DUAL_VIEW_KERNEL_INTERFACE",
    "DUAL_VIEW_KERNEL_SCHEMA",
    "DualViewKernelContract",
    "ForgedImplementationDispositionIdentityError",
    "IMPLEMENTATION_DISPOSITION_EVIDENCE",
    "IMPLEMENTATION_DISPOSITION_INTERFACE",
    "IMPLEMENTATION_DISPOSITION_SCHEMA",
    "IMPLEMENTATION_DISPOSITION_VERSION",
    "IMPLEMENTATION_FOREST_ROOTS_SCHEMA",
    "ImplementationDisposition",
    "ImplementationDispositionAuthorityError",
    "ImplementationDispositionBoundsError",
    "ImplementationDispositionError",
    "ImplementationForestRoots",
    "KernelViewKind",
    "MAX_RECORD_BYTES",
    "PRE_IMPLEMENTATION_KERNEL_RECEIPT_INTERFACE",
    "PRE_IMPLEMENTATION_KERNEL_RECEIPT_SCHEMA",
    "PreImplementationKernelReceipt",
    "ResidualRequirement",
    "SCHEMA_VERSION",
    "UnauthorizedProviderInvocationError",
    "assert_provider_invocation_allowed",
    "closed_disposition_values",
    "closed_dispositions",
    "disposition_metric_labels",
    "expected_provider_call_count",
    "implementation_disposition_cid",
    "parse_implementation_disposition",
    "provider_invocation_authorized",
    "residual_requirement_for",
    "seal_pre_implementation_kernel_receipt",
    "verify_pre_implementation_kernel_receipt",
]
