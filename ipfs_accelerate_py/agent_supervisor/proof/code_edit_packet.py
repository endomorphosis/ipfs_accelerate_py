"""CodeEditPacket@1 — content-addressed supervisor edit tasks (CBP-080).

A :class:`CodeEditPacket` is the bounded projection from open claims /
obligations into a supervisor implementable unit.  It binds the source tree,
claim and obligation identities, assumptions, invalidation reasons, predicted
files, and acceptance criteria without embedding full proof bodies or gold IR.

Normative rules:

* Packet identity is content-addressed from the canonical payload.
* ``implementable=false`` for reject, timeout, unsupported, not_measured, and
  stale-required-input dispositions (also satisfied and missing-tree).
* Every prover binding carries ``semantic_authority=false``.
* Cache and claim status are compact records (ids, disposition, assurance) —
  never full receipt or proof bodies.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .code_claim_contracts import ClaimStatus
from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    ContractValidationError,
    content_identity,
)


CODE_EDIT_PACKET_INTERFACE: Final = "CodeEditPacket@1"
CODE_EDIT_PACKET_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-edit-packet@1"
)
CODE_EDIT_PACKET_VERSION: Final = "1"
CODE_EDIT_PACKET_CONTRACT_VERSION: Final = 1

CACHE_STATUS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-edit-cache-status@1"
)
CLAIM_STATUS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-edit-claim-status@1"
)
PROVER_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-edit-prover-binding@1"
)

# Compact status fields that must never appear (proof bodies / gold IR).
_FORBIDDEN_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "proof_body",
        "proof_text",
        "lean_source",
        "gold_ir",
        "gold_ir_body",
        "full_receipt",
        "receipt_body",
        "kernel_proof_body",
        "solver_trace",
        "private_witness",
        "witness",
    }
)


class CodeEditPacketError(ContractValidationError):
    """Raised when a CodeEditPacket is malformed or fails closed."""


class NonImplementableReason(str, Enum):
    """Closed vocabulary of reasons a packet is not implementable."""

    REJECT = "reject"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    NOT_MEASURED = "not_measured"
    STALE_REQUIRED_INPUT = "stale_required_input"
    SATISFIED = "satisfied"
    MISSING_SOURCE_TREE = "missing_source_tree"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


# Acceptance requires these five to force implementable=false.
REQUIRED_NON_IMPLEMENTABLE: Final[frozenset[str]] = frozenset(
    {
        NonImplementableReason.REJECT.value,
        NonImplementableReason.TIMEOUT.value,
        NonImplementableReason.UNSUPPORTED.value,
        NonImplementableReason.NOT_MEASURED.value,
        NonImplementableReason.STALE_REQUIRED_INPUT.value,
    }
)


class CacheDisposition(str, Enum):
    """Compact cache outcome recorded on a packet (no proof body)."""

    UNKNOWN = "unknown"
    HIT = "hit"
    MISS = "miss"
    STALE = "stale"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    NOT_QUERIED = "not_queried"


def _sorted_unique(values: Iterable[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        values = (values,)
    out: list[str] = []
    seen: set[str] = set()
    for raw in values or ():
        text = str(raw or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(sorted(out))


def _norm_text(value: Any, *, field_name: str, required: bool = False) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise CodeEditPacketError(f"{field_name} must be a string")
    if required and not text:
        raise CodeEditPacketError(f"{field_name} is required")
    return text


def _norm_ids(
    values: Any,
    *,
    field_name: str,
    required: bool = False,
) -> tuple[str, ...]:
    if values is None:
        items: Iterable[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise CodeEditPacketError(f"{field_name} must be a sequence of strings")
    result = _sorted_unique(items)
    if required and not result:
        raise CodeEditPacketError(f"{field_name} must not be empty")
    return result


def _norm_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise CodeEditPacketError(f"{field_name} must be a mapping")
    if not all(isinstance(k, str) for k in value):
        raise CodeEditPacketError(f"{field_name} keys must be strings")
    for key in value:
        lowered = key.lower().replace("-", "_")
        # Exact key match only — flags like gold_ir_excluded are allowed.
        if lowered in _FORBIDDEN_BODY_KEYS:
            raise CodeEditPacketError(
                f"{field_name} must not embed full proof bodies ({key})"
            )
        # Nested body payloads are also rejected.
        nested = value[key]
        if isinstance(nested, Mapping):
            for nested_key in nested:
                nested_l = str(nested_key).lower().replace("-", "_")
                if nested_l in _FORBIDDEN_BODY_KEYS:
                    raise CodeEditPacketError(
                        f"{field_name} must not embed full proof bodies "
                        f"({key}.{nested_key})"
                    )
    return MappingProxyType(dict(value))


def _norm_assurance(value: Any) -> AssuranceLevel:
    if isinstance(value, AssuranceLevel):
        return value
    text = str(value or AssuranceLevel.KERNEL_VERIFIED.value).strip()
    try:
        return AssuranceLevel(text)
    except ValueError as exc:
        raise CodeEditPacketError(
            f"required_assurance must be a valid AssuranceLevel, got {text!r}"
        ) from exc


def _reject_body_fields(payload: Mapping[str, Any], *, field_name: str) -> None:
    for key in payload:
        lowered = str(key).lower()
        if lowered in _FORBIDDEN_BODY_KEYS:
            raise CodeEditPacketError(
                f"{field_name} must not embed full proof bodies ({key})"
            )


@dataclass(frozen=True)
class CacheStatusRecord:
    """Compact cache disposition without a proof body."""

    disposition: CacheDisposition = CacheDisposition.NOT_QUERIED
    cache_key_id: str = ""
    receipt_id: str = ""
    reason_codes: tuple[str, ...] = ()
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED

    def __post_init__(self) -> None:
        disposition = self.disposition
        if not isinstance(disposition, CacheDisposition):
            try:
                disposition = CacheDisposition(str(disposition))
            except ValueError as exc:
                raise CodeEditPacketError(
                    f"unknown cache disposition: {self.disposition!r}"
                ) from exc
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self, "cache_key_id", _norm_text(self.cache_key_id, field_name="cache_key_id")
        )
        object.__setattr__(
            self, "receipt_id", _norm_text(self.receipt_id, field_name="receipt_id")
        )
        object.__setattr__(
            self, "reason_codes", _norm_ids(self.reason_codes, field_name="reason_codes")
        )
        object.__setattr__(
            self, "required_assurance", _norm_assurance(self.required_assurance)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CACHE_STATUS_SCHEMA,
            "disposition": self.disposition.value,
            "cache_key_id": self.cache_key_id,
            "receipt_id": self.receipt_id,
            "reason_codes": list(self.reason_codes),
            "required_assurance": self.required_assurance.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "CacheStatusRecord":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise CodeEditPacketError("cache_status must be an object")
        _reject_body_fields(payload, field_name="cache_status")
        schema = payload.get("schema")
        if schema not in (None, "", CACHE_STATUS_SCHEMA):
            raise CodeEditPacketError("unsupported cache_status schema")
        return cls(
            disposition=str(payload.get("disposition") or CacheDisposition.NOT_QUERIED.value),
            cache_key_id=str(payload.get("cache_key_id") or ""),
            receipt_id=str(payload.get("receipt_id") or ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.KERNEL_VERIFIED
            ),
        )


@dataclass(frozen=True)
class ClaimStatusRecord:
    """Compact claim lifecycle handle without evidence bodies."""

    claim_id: str = ""
    property_id: str = ""
    status: str = ClaimStatus.UNKNOWN.value
    obligation_id: str = ""
    evidence_ids: tuple[str, ...] = ()
    evidence_tiers: tuple[str, ...] = ()
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED
    derived_assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("claim_id", "property_id", "obligation_id"):
            object.__setattr__(
                self, name, _norm_text(getattr(self, name), field_name=name)
            )
        status = self.status
        if isinstance(status, ClaimStatus):
            status_s = status.value
        else:
            status_s = str(status or ClaimStatus.UNKNOWN.value).strip()
        # Accept known ClaimStatus values; allow pass-through for extensions.
        object.__setattr__(self, "status", status_s)
        object.__setattr__(
            self, "evidence_ids", _norm_ids(self.evidence_ids, field_name="evidence_ids")
        )
        object.__setattr__(
            self,
            "evidence_tiers",
            _norm_ids(self.evidence_tiers, field_name="evidence_tiers"),
        )
        object.__setattr__(
            self, "reason_codes", _norm_ids(self.reason_codes, field_name="reason_codes")
        )
        object.__setattr__(
            self, "required_assurance", _norm_assurance(self.required_assurance)
        )
        object.__setattr__(
            self, "derived_assurance", _norm_assurance(self.derived_assurance)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CLAIM_STATUS_SCHEMA,
            "claim_id": self.claim_id,
            "property_id": self.property_id,
            "status": self.status,
            "obligation_id": self.obligation_id,
            "evidence_ids": list(self.evidence_ids),
            "evidence_tiers": list(self.evidence_tiers),
            "required_assurance": self.required_assurance.value,
            "derived_assurance": self.derived_assurance.value,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ClaimStatusRecord":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise CodeEditPacketError("claim_status must be an object")
        _reject_body_fields(payload, field_name="claim_status")
        schema = payload.get("schema")
        if schema not in (None, "", CLAIM_STATUS_SCHEMA):
            raise CodeEditPacketError("unsupported claim_status schema")
        return cls(
            claim_id=str(payload.get("claim_id") or ""),
            property_id=str(payload.get("property_id") or ""),
            status=str(payload.get("status") or ClaimStatus.UNKNOWN.value),
            obligation_id=str(payload.get("obligation_id") or ""),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
            evidence_tiers=tuple(payload.get("evidence_tiers") or ()),
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.KERNEL_VERIFIED
            ),
            derived_assurance=payload.get(
                "derived_assurance", AssuranceLevel.UNVERIFIED
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )


@dataclass(frozen=True)
class ProverBinding:
    """Prover/solver identity with mandatory non-semantic authority."""

    prover_id: str = ""
    solver_id: str = ""
    kernel_id: str = ""
    toolchain_id: str = ""
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        for name in ("prover_id", "solver_id", "kernel_id", "toolchain_id"):
            object.__setattr__(
                self, name, _norm_text(getattr(self, name), field_name=name)
            )
        # Hard rule: prover fields never carry semantic authority.
        object.__setattr__(self, "semantic_authority", False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVER_BINDING_SCHEMA,
            "prover_id": self.prover_id,
            "solver_id": self.solver_id,
            "kernel_id": self.kernel_id,
            "toolchain_id": self.toolchain_id,
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ProverBinding":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise CodeEditPacketError("prover must be an object")
        # Caller-supplied semantic_authority=true is ignored / forced false.
        return cls(
            prover_id=str(payload.get("prover_id") or ""),
            solver_id=str(payload.get("solver_id") or ""),
            kernel_id=str(payload.get("kernel_id") or ""),
            toolchain_id=str(payload.get("toolchain_id") or ""),
            semantic_authority=False,
        )


def compute_implementable(
    *,
    repository_tree_id: str,
    claim_status: str = "",
    compile_status: str = "",
    cache_disposition: CacheDisposition | str = CacheDisposition.NOT_QUERIED,
    reason_codes: Sequence[str] = (),
    explicit_reasons: Sequence[str] = (),
) -> tuple[bool, tuple[str, ...]]:
    """Decide implementability and return (implementable, non_implementable_reasons).

    ``implementable`` is false for reject / timeout / unsupported /
    not_measured / stale_required_input, as well as satisfied claims and a
    missing source tree.
    """

    reasons: list[str] = []
    reasons.extend(str(r).strip() for r in explicit_reasons if str(r).strip())

    if not str(repository_tree_id or "").strip():
        reasons.append(NonImplementableReason.MISSING_SOURCE_TREE.value)

    status = str(claim_status or "").strip().lower()
    compile = str(compile_status or "").strip().lower()
    disposition = (
        cache_disposition
        if isinstance(cache_disposition, CacheDisposition)
        else CacheDisposition(str(cache_disposition or CacheDisposition.NOT_QUERIED.value))
    )
    codes = {str(c).strip().lower() for c in reason_codes if str(c).strip()}
    codes.update(r.lower() for r in reasons)

    def _has(*tokens: str) -> bool:
        for token in tokens:
            if token in codes:
                return True
            if any(token in c for c in codes):
                return True
        return False

    if status in {ClaimStatus.UNSUPPORTED.value, "unsupported"} or compile == "unsupported":
        reasons.append(NonImplementableReason.UNSUPPORTED.value)
    if status in {ClaimStatus.NOT_MEASURED.value, "not_measured"} or compile == "not_measured":
        reasons.append(NonImplementableReason.NOT_MEASURED.value)
    if status in {ClaimStatus.STALE.value, "stale"} or _has(
        "stale_required_input", "stale_required", "required_input_stale"
    ):
        reasons.append(NonImplementableReason.STALE_REQUIRED_INPUT.value)
    if status in {ClaimStatus.SATISFIED.value, "satisfied"}:
        reasons.append(NonImplementableReason.SATISFIED.value)
    if status in {ClaimStatus.REFUTED.value, "refuted"}:
        reasons.append(NonImplementableReason.REFUTED.value)
    if disposition is CacheDisposition.REJECTED or _has("reject", "rejected", "cache_rejected"):
        reasons.append(NonImplementableReason.REJECT.value)
    if disposition is CacheDisposition.TIMEOUT or _has("timeout", "timed_out", "single_flight_timeout"):
        reasons.append(NonImplementableReason.TIMEOUT.value)

    # Stable unique ordered reasons.
    ordered = _sorted_unique(reasons)
    # Any non-empty reason set means not implementable; open claims with a tree
    # and no blocking reason remain implementable.
    implementable = len(ordered) == 0
    return implementable, ordered


@dataclass(frozen=True)
class CodeEditPacket(CanonicalContract):
    """Content-addressed implementable (or blocked) supervisor edit packet.

    Interface: ``CodeEditPacket@1``
    """

    SCHEMA: ClassVar[str] = CODE_EDIT_PACKET_SCHEMA

    repository_tree_id: str
    claim_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    assumption_ids: tuple[str, ...] = ()
    invalidation_reasons: tuple[str, ...] = ()
    predicted_files: tuple[str, ...] = ()
    acceptance_ids: tuple[str, ...] = ()
    property_ids: tuple[str, ...] = ()
    implementable: bool = True
    non_implementable_reasons: tuple[str, ...] = ()
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED
    cache_status: CacheStatusRecord = field(default_factory=CacheStatusRecord)
    claim_status: ClaimStatusRecord = field(default_factory=ClaimStatusRecord)
    prover: ProverBinding = field(default_factory=ProverBinding)
    repository_id: str = ""
    task_id: str = ""
    residual_ref_ids: tuple[str, ...] = ()
    plateau_packet_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_tree_id",
            _norm_text(self.repository_tree_id, field_name="repository_tree_id"),
        )
        for name in ("repository_id", "task_id", "plateau_packet_id"):
            object.__setattr__(
                self, name, _norm_text(getattr(self, name), field_name=name)
            )
        for name in (
            "claim_ids",
            "obligation_ids",
            "assumption_ids",
            "invalidation_reasons",
            "predicted_files",
            "acceptance_ids",
            "property_ids",
            "non_implementable_reasons",
            "residual_ref_ids",
        ):
            object.__setattr__(
                self, name, _norm_ids(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self, "required_assurance", _norm_assurance(self.required_assurance)
        )

        cache_status = self.cache_status
        if isinstance(cache_status, Mapping):
            cache_status = CacheStatusRecord.from_dict(cache_status)
        elif not isinstance(cache_status, CacheStatusRecord):
            cache_status = CacheStatusRecord()
        object.__setattr__(self, "cache_status", cache_status)

        claim_status = self.claim_status
        if isinstance(claim_status, Mapping):
            claim_status = ClaimStatusRecord.from_dict(claim_status)
        elif not isinstance(claim_status, ClaimStatusRecord):
            claim_status = ClaimStatusRecord()
        object.__setattr__(self, "claim_status", claim_status)

        prover = self.prover
        if isinstance(prover, Mapping):
            prover = ProverBinding.from_dict(prover)
        elif not isinstance(prover, ProverBinding):
            prover = ProverBinding()
        # Force semantic_authority=false even if a subclass slips through.
        if prover.semantic_authority is not False:
            prover = ProverBinding(
                prover_id=prover.prover_id,
                solver_id=prover.solver_id,
                kernel_id=prover.kernel_id,
                toolchain_id=prover.toolchain_id,
                semantic_authority=False,
            )
        object.__setattr__(self, "prover", prover)

        object.__setattr__(
            self, "metadata", _norm_mapping(self.metadata, field_name="metadata")
        )

        # Re-derive implementability from claim/cache signals and explicit reasons.
        derived_impl, derived_reasons = compute_implementable(
            repository_tree_id=self.repository_tree_id,
            claim_status=claim_status.status,
            cache_disposition=cache_status.disposition,
            reason_codes=tuple(claim_status.reason_codes)
            + tuple(cache_status.reason_codes),
            explicit_reasons=self.non_implementable_reasons,
        )
        # Caller may only *restrict* implementability (never widen past derived).
        implementable = bool(self.implementable) and derived_impl
        reasons = list(derived_reasons)
        if set(reasons) & REQUIRED_NON_IMPLEMENTABLE:
            implementable = False
        if not self.repository_tree_id:
            implementable = False
            if NonImplementableReason.MISSING_SOURCE_TREE.value not in reasons:
                reasons.append(NonImplementableReason.MISSING_SOURCE_TREE.value)
        object.__setattr__(self, "implementable", implementable)
        object.__setattr__(self, "non_implementable_reasons", _sorted_unique(reasons))

    @property
    def packet_id(self) -> str:
        return self.content_id

    @property
    def interface(self) -> str:
        return CODE_EDIT_PACKET_INTERFACE

    @property
    def source_tree_id(self) -> str:
        return self.repository_tree_id

    @property
    def assumptions(self) -> tuple[str, ...]:
        """Alias for assumption_ids (acceptance vocabulary)."""

        return self.assumption_ids

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CODE_EDIT_PACKET_CONTRACT_VERSION,
            "interface": CODE_EDIT_PACKET_INTERFACE,
            "version": CODE_EDIT_PACKET_VERSION,
            "repository_tree_id": self.repository_tree_id,
            "repository_id": self.repository_id,
            "claim_ids": list(self.claim_ids),
            "obligation_ids": list(self.obligation_ids),
            "assumption_ids": list(self.assumption_ids),
            "invalidation_reasons": list(self.invalidation_reasons),
            "predicted_files": list(self.predicted_files),
            "acceptance_ids": list(self.acceptance_ids),
            "property_ids": list(self.property_ids),
            "implementable": bool(self.implementable),
            "non_implementable_reasons": list(self.non_implementable_reasons),
            "required_assurance": self.required_assurance.value
            if isinstance(self.required_assurance, AssuranceLevel)
            else str(self.required_assurance),
            "cache_status": self.cache_status.to_dict(),
            "claim_status": self.claim_status.to_dict(),
            "prover": self.prover.to_dict(),
            "task_id": self.task_id,
            "residual_ref_ids": list(self.residual_ref_ids),
            "plateau_packet_id": self.plateau_packet_id,
            "metadata": dict(self.metadata),
        }

    def to_dict(self, *, include_id: bool = False) -> dict[str, Any]:  # type: ignore[override]
        payload = super().to_dict()
        if include_id:
            payload = {**payload, "packet_id": self.packet_id, "content_id": self.content_id}
        return payload

    def to_record(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "packet_id": self.packet_id,
            "content_id": self.content_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeEditPacket":
        if not isinstance(payload, Mapping):
            raise CodeEditPacketError("code edit packet must be an object")
        _reject_body_fields(payload, field_name="packet")
        schema = payload.get("schema")
        if schema not in (None, "", CODE_EDIT_PACKET_SCHEMA):
            raise CodeEditPacketError(
                f"unsupported code-edit-packet schema; use {CODE_EDIT_PACKET_SCHEMA}"
            )
        version = payload.get("contract_version")
        if version not in (None, CODE_EDIT_PACKET_CONTRACT_VERSION):
            raise CodeEditPacketError(
                "unsupported code-edit-packet contract version; rebuild with current contract"
            )
        interface = payload.get("interface")
        if interface not in (None, "", CODE_EDIT_PACKET_INTERFACE):
            raise CodeEditPacketError(
                f"unsupported interface; use {CODE_EDIT_PACKET_INTERFACE}"
            )

        tree = str(
            payload.get("repository_tree_id")
            or payload.get("source_tree_id")
            or payload.get("tree_id")
            or ""
        )
        assumptions = payload.get("assumption_ids")
        if assumptions is None:
            assumptions = payload.get("assumptions") or ()

        return cls(
            repository_tree_id=tree,
            claim_ids=tuple(payload.get("claim_ids") or ()),
            obligation_ids=tuple(payload.get("obligation_ids") or ()),
            assumption_ids=tuple(assumptions or ()),
            invalidation_reasons=tuple(payload.get("invalidation_reasons") or ()),
            predicted_files=tuple(payload.get("predicted_files") or ()),
            acceptance_ids=tuple(payload.get("acceptance_ids") or ()),
            property_ids=tuple(payload.get("property_ids") or ()),
            implementable=bool(payload.get("implementable", True)),
            non_implementable_reasons=tuple(
                payload.get("non_implementable_reasons") or ()
            ),
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.KERNEL_VERIFIED
            ),
            cache_status=CacheStatusRecord.from_dict(payload.get("cache_status")),
            claim_status=ClaimStatusRecord.from_dict(payload.get("claim_status")),
            prover=ProverBinding.from_dict(payload.get("prover")),
            repository_id=str(payload.get("repository_id") or ""),
            task_id=str(payload.get("task_id") or ""),
            residual_ref_ids=tuple(payload.get("residual_ref_ids") or ()),
            plateau_packet_id=str(
                payload.get("plateau_packet_id")
                or payload.get("plateau_codex_packet_id")
                or ""
            ),
            metadata=dict(payload.get("metadata") or {}),
        )

    def with_updates(self, **changes: Any) -> "CodeEditPacket":
        """Return a new packet with selected fields replaced."""

        base = {
            "repository_tree_id": self.repository_tree_id,
            "claim_ids": self.claim_ids,
            "obligation_ids": self.obligation_ids,
            "assumption_ids": self.assumption_ids,
            "invalidation_reasons": self.invalidation_reasons,
            "predicted_files": self.predicted_files,
            "acceptance_ids": self.acceptance_ids,
            "property_ids": self.property_ids,
            "implementable": self.implementable,
            "non_implementable_reasons": self.non_implementable_reasons,
            "required_assurance": self.required_assurance,
            "cache_status": self.cache_status,
            "claim_status": self.claim_status,
            "prover": self.prover,
            "repository_id": self.repository_id,
            "task_id": self.task_id,
            "residual_ref_ids": self.residual_ref_ids,
            "plateau_packet_id": self.plateau_packet_id,
            "metadata": dict(self.metadata),
        }
        base.update(changes)
        return CodeEditPacket(**base)


def build_code_edit_packet(
    *,
    repository_tree_id: str,
    claim_ids: Sequence[str] = (),
    obligation_ids: Sequence[str] = (),
    assumption_ids: Sequence[str] = (),
    invalidation_reasons: Sequence[str] = (),
    predicted_files: Sequence[str] = (),
    acceptance_ids: Sequence[str] = (),
    property_ids: Sequence[str] = (),
    required_assurance: AssuranceLevel | str = AssuranceLevel.KERNEL_VERIFIED,
    cache_status: CacheStatusRecord | Mapping[str, Any] | None = None,
    claim_status: ClaimStatusRecord | Mapping[str, Any] | None = None,
    prover: ProverBinding | Mapping[str, Any] | None = None,
    repository_id: str = "",
    task_id: str = "",
    residual_ref_ids: Sequence[str] = (),
    plateau_packet_id: str = "",
    claim_lifecycle: str = "",
    compile_status: str = "",
    reason_codes: Sequence[str] = (),
    force_non_implementable: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> CodeEditPacket:
    """Construct a packet and re-derive implementability from status signals."""

    if isinstance(cache_status, Mapping):
        cache_rec = CacheStatusRecord.from_dict(cache_status)
    elif isinstance(cache_status, CacheStatusRecord):
        cache_rec = cache_status
    else:
        cache_rec = CacheStatusRecord()

    if isinstance(claim_status, Mapping):
        claim_rec = ClaimStatusRecord.from_dict(claim_status)
    elif isinstance(claim_status, ClaimStatusRecord):
        claim_rec = claim_status
    else:
        claim_rec = ClaimStatusRecord()

    if isinstance(prover, Mapping):
        prover_rec = ProverBinding.from_dict(prover)
    elif isinstance(prover, ProverBinding):
        prover_rec = prover
    else:
        prover_rec = ProverBinding()

    lifecycle = claim_lifecycle or claim_rec.status
    implementable, non_impl = compute_implementable(
        repository_tree_id=repository_tree_id,
        claim_status=lifecycle,
        compile_status=compile_status,
        cache_disposition=cache_rec.disposition,
        reason_codes=tuple(reason_codes) + tuple(claim_rec.reason_codes),
        explicit_reasons=force_non_implementable,
    )

    return CodeEditPacket(
        repository_tree_id=repository_tree_id,
        claim_ids=tuple(claim_ids),
        obligation_ids=tuple(obligation_ids),
        assumption_ids=tuple(assumption_ids),
        invalidation_reasons=tuple(invalidation_reasons),
        predicted_files=tuple(predicted_files),
        acceptance_ids=tuple(acceptance_ids),
        property_ids=tuple(property_ids),
        implementable=implementable,
        non_implementable_reasons=non_impl,
        required_assurance=required_assurance,
        cache_status=cache_rec,
        claim_status=claim_rec,
        prover=prover_rec,
        repository_id=repository_id,
        task_id=task_id,
        residual_ref_ids=tuple(residual_ref_ids),
        plateau_packet_id=plateau_packet_id,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "CACHE_STATUS_SCHEMA",
    "CLAIM_STATUS_SCHEMA",
    "CODE_EDIT_PACKET_CONTRACT_VERSION",
    "CODE_EDIT_PACKET_INTERFACE",
    "CODE_EDIT_PACKET_SCHEMA",
    "CODE_EDIT_PACKET_VERSION",
    "PROVER_BINDING_SCHEMA",
    "REQUIRED_NON_IMPLEMENTABLE",
    "CacheDisposition",
    "CacheStatusRecord",
    "ClaimStatusRecord",
    "CodeEditPacket",
    "CodeEditPacketError",
    "NonImplementableReason",
    "ProverBinding",
    "build_code_edit_packet",
    "compute_implementable",
    "content_identity",
]
