"""SupervisorLogicPlatformReceiptAdmission@1 — ten-point receipt gate (LPC-111).

A proof result may influence completion or merge only when every check in the
human-plan ten-point floor holds. Importing this module never loads
``ipfs_datasets_py`` and never upgrades provider success into proof authority
(LPC-032).

The admission surface is intentionally separate from
:mod:`logic_platform_client` (LPC-110). Clients project untrusted receipt
envelopes; this module is the fail-closed gate that decides whether those
envelopes may affect completion or merge.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Iterable

from .formal_verification_contracts import (
    AssuranceLevel,
    ContractValidationError,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    assess_assurance,
    content_identity,
)


SUPERVISOR_LOGIC_PLATFORM_RECEIPT_ADMISSION_INTERFACE: Final = (
    "SupervisorLogicPlatformReceiptAdmission@1"
)
SUPERVISOR_LOGIC_PLATFORM_RECEIPT_ADMISSION_VERSION: Final = "1.0.0"
ADMISSION_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-receipt-admission@1"
)
ADMISSION_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-receipt-admission-result@1"
)
ADMISSION_CONTEXT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-receipt-admission-context@1"
)
ADMISSION_TASK_ID: Final = "LPC-111"
ADMISSION_GOAL_ID: Final = "LPC-G110"

# Ordered ten-point floor from plan §8 / LPC-111 acceptance.
TEN_POINT_CHECKS: Final[tuple[str, ...]] = (
    "structural_validity",
    "content_identity",
    "source_tree_environment_policy_binding",
    "translation_chain",
    "evidence_kind",
    "authority_ceiling",
    "required_reconstruction",
    "freshness",
    "non_simulation",
    "policy_admission",
)

_CONTENT_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:b[a-z2-7]{50,}|sha256:[0-9a-f]{64}|[a-z]+:[A-Za-z0-9._:/-]+)$"
)

_AUTHORITY_RANK: Final[Mapping[str, int]] = MappingProxyType(
    {
        AssuranceLevel.UNVERIFIED.value: 0,
        "unknown": 0,
        "none": 0,
        AssuranceLevel.CANDIDATE.value: 1,
        "advisory": 1,
        "simulated": 1,
        AssuranceLevel.SOLVER_CHECKED.value: 2,
        AssuranceLevel.KERNEL_VERIFIED.value: 3,
        AssuranceLevel.ATTESTED.value: 4,
    }
)

# Evidence kinds that cannot alone support a proved / kernel-required claim.
_NON_KERNEL_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        EvidenceKind.UNKNOWN.value,
        EvidenceKind.LLM_OUTPUT.value,
        EvidenceKind.ATP_CANDIDATE.value,
        EvidenceKind.SMT_CANDIDATE.value,
        EvidenceKind.SOLVER_RESULT.value,
        EvidenceKind.TEST_RESULT.value,
        EvidenceKind.STATIC_ANALYSIS.value,
        EvidenceKind.CACHE_ENTRY.value,
        "simulated",
        "advisory",
        "candidate",
    }
)

_CONCLUSIVE_VERDICTS: Final[frozenset[str]] = frozenset(
    {
        ProofVerdict.PROVED.value,
        ProofVerdict.DISPROVED.value,
    }
)

_VALID_TRANSLATION_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "exact",
        "equisatisfiable",
        "bounded_abstraction",
        "conservative_approximation",
        # Heuristic translations never support kernel/completion influence.
    }
)

_KERNEL_REQUIRED_TRANSLATION: Final[frozenset[str]] = frozenset(
    {
        "exact",
        "equisatisfiable",
    }
)


class LogicPlatformAdmissionError(ValueError):
    """Raised when the admission context or request is structurally invalid."""


class AdmissionCheck(str, Enum):
    """Closed vocabulary for the ten-point receipt admission floor."""

    STRUCTURAL_VALIDITY = "structural_validity"
    CONTENT_IDENTITY = "content_identity"
    BINDINGS = "source_tree_environment_policy_binding"
    TRANSLATION_CHAIN = "translation_chain"
    EVIDENCE_KIND = "evidence_kind"
    AUTHORITY_CEILING = "authority_ceiling"
    REQUIRED_RECONSTRUCTION = "required_reconstruction"
    FRESHNESS = "freshness"
    NON_SIMULATION = "non_simulation"
    POLICY_ADMISSION = "policy_admission"


class AdmissionDisposition(str, Enum):
    """Outcome of a single check or overall admission decision."""

    ADMITTED = "admitted"
    REJECTED = "rejected"
    NOT_EVALUATED = "not_evaluated"


def _token(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise LogicPlatformAdmissionError(f"{field_name} must be a non-empty string")
    token = value.strip()
    if not token:
        raise LogicPlatformAdmissionError(f"{field_name} must be a non-empty string")
    return token


def _optional_token(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LogicPlatformAdmissionError(f"{field_name} must be a string or null")
    token = value.strip()
    return token or None


def _soft_optional_token(value: Any) -> str | None:
    """Best-effort optional token; non-strings become None (structural path)."""

    if value is None:
        return None
    if not isinstance(value, str):
        return None
    token = value.strip()
    return token or None


def _authority_rank(value: Any) -> int:
    token = str(getattr(value, "value", value) or "unknown").strip().lower()
    if token not in _AUTHORITY_RANK:
        raise LogicPlatformAdmissionError(f"unknown authority ceiling: {token!r}")
    return _AUTHORITY_RANK[token]


def _normalize_authority(value: Any) -> str:
    token = str(getattr(value, "value", value) or "unknown").strip().lower()
    if not token or token in {"unknown", "none"}:
        return AssuranceLevel.UNVERIFIED.value
    if token in {"advisory", "simulated"}:
        return AssuranceLevel.CANDIDATE.value
    if token in _AUTHORITY_RANK:
        return token
    try:
        return AssuranceLevel(token).value
    except ValueError as error:
        raise LogicPlatformAdmissionError(
            f"unknown authority ceiling: {token!r}"
        ) from error


def _normalize_freshness(value: Any) -> str:
    token = str(getattr(value, "value", value) or "unknown").strip().lower()
    if token in {"", "unknown"}:
        return EvidenceFreshness.UNKNOWN.value
    if token in {"current", "fresh"}:
        return EvidenceFreshness.CURRENT.value
    if token in {"stale"}:
        return EvidenceFreshness.STALE.value
    try:
        return EvidenceFreshness(token).value
    except ValueError:
        return EvidenceFreshness.UNKNOWN.value


def _normalize_evidence_kind(value: Any | None) -> str | None:
    if value is None:
        return None
    token = str(getattr(value, "value", value)).strip().lower()
    if not token:
        return None
    try:
        return EvidenceKind(token).value
    except ValueError:
        return token


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _boolish(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "1"}:
            return True
        if token in {"false", "no", "0", ""}:
            return False
    return default


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of one ordered admission check."""

    check: AdmissionCheck
    passed: bool
    reason_code: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check.value,
            "passed": self.passed,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class AdmissionContext:
    """Bindings against which an untrusted receipt is evaluated.

    Missing required bindings fail closed at construction; they never soft-succeed.
    """

    task_id: str
    repository_tree_id: str
    policy_id: str
    operation: str
    required_authority: str = AssuranceLevel.KERNEL_VERIFIED.value
    repository_id: str | None = None
    environment_id: str | None = None
    source_id: str | None = None
    policy_revision: str | None = None
    plan_id: str | None = None
    obligation_id: str | None = None
    expected_content_id: str | None = None
    require_reconstruction: bool = True
    require_kernel: bool = True
    network_allowed: bool = False
    schema_version: str = ADMISSION_CONTEXT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _token(self.task_id, field_name="task_id"))
        object.__setattr__(
            self,
            "repository_tree_id",
            _token(self.repository_tree_id, field_name="repository_tree_id"),
        )
        object.__setattr__(
            self, "policy_id", _token(self.policy_id, field_name="policy_id")
        )
        object.__setattr__(
            self, "operation", _token(self.operation, field_name="operation")
        )
        object.__setattr__(
            self,
            "required_authority",
            _normalize_authority(self.required_authority),
        )
        object.__setattr__(
            self,
            "repository_id",
            _optional_token(self.repository_id, field_name="repository_id"),
        )
        object.__setattr__(
            self,
            "environment_id",
            _optional_token(self.environment_id, field_name="environment_id"),
        )
        object.__setattr__(
            self,
            "source_id",
            _optional_token(self.source_id, field_name="source_id"),
        )
        object.__setattr__(
            self,
            "policy_revision",
            _optional_token(self.policy_revision, field_name="policy_revision"),
        )
        object.__setattr__(
            self, "plan_id", _optional_token(self.plan_id, field_name="plan_id")
        )
        object.__setattr__(
            self,
            "obligation_id",
            _optional_token(self.obligation_id, field_name="obligation_id"),
        )
        object.__setattr__(
            self,
            "expected_content_id",
            _optional_token(
                self.expected_content_id, field_name="expected_content_id"
            ),
        )
        if not isinstance(self.require_reconstruction, bool):
            raise LogicPlatformAdmissionError(
                "require_reconstruction must be a boolean"
            )
        if not isinstance(self.require_kernel, bool):
            raise LogicPlatformAdmissionError("require_kernel must be a boolean")
        if not isinstance(self.network_allowed, bool):
            raise LogicPlatformAdmissionError("network_allowed must be a boolean")
        if self.schema_version != ADMISSION_CONTEXT_SCHEMA:
            raise LogicPlatformAdmissionError(
                f"unsupported admission context schema: {self.schema_version!r}"
            )
        # Kernel-required policies always require reconstruction.
        if (
            _authority_rank(self.required_authority)
            >= _authority_rank(AssuranceLevel.KERNEL_VERIFIED.value)
            and not self.require_reconstruction
        ):
            raise LogicPlatformAdmissionError(
                "kernel-required authority requires reconstruction"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema_version,
            "task_id": self.task_id,
            "repository_tree_id": self.repository_tree_id,
            "policy_id": self.policy_id,
            "operation": self.operation,
            "required_authority": self.required_authority,
            "repository_id": self.repository_id,
            "environment_id": self.environment_id,
            "source_id": self.source_id,
            "policy_revision": self.policy_revision,
            "plan_id": self.plan_id,
            "obligation_id": self.obligation_id,
            "expected_content_id": self.expected_content_id,
            "require_reconstruction": self.require_reconstruction,
            "require_kernel": self.require_kernel,
            "network_allowed": self.network_allowed,
        }


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """Fail-closed result of the ten-point admission gate.

    ``admitted`` is True only when every check in :data:`TEN_POINT_CHECKS`
    passed. ``may_affect_completion`` and ``may_affect_merge`` are locked to
    ``admitted`` so callers cannot promote partial passes.
    """

    admitted: bool
    disposition: AdmissionDisposition
    checks: tuple[CheckResult, ...]
    context: AdmissionContext
    receipt_content_id: str | None = None
    authority_ceiling: str = AssuranceLevel.UNVERIFIED.value
    simulated: bool = False
    freshness: str = EvidenceFreshness.UNKNOWN.value
    semantic_verdict: str = "unknown"
    reasons: tuple[str, ...] = ()
    schema_version: str = ADMISSION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if len(self.checks) != len(TEN_POINT_CHECKS):
            raise LogicPlatformAdmissionError(
                "admission result must record every ten-point check"
            )
        for index, expected in enumerate(TEN_POINT_CHECKS):
            if self.checks[index].check.value != expected:
                raise LogicPlatformAdmissionError(
                    f"admission checks must follow plan order; expected {expected}"
                )
        all_passed = all(item.passed for item in self.checks)
        if self.admitted != all_passed:
            raise LogicPlatformAdmissionError(
                "admitted flag must equal conjunction of all ten checks"
            )
        if self.admitted and self.disposition is not AdmissionDisposition.ADMITTED:
            raise LogicPlatformAdmissionError(
                "admitted results must use admitted disposition"
            )
        if (not self.admitted) and self.disposition is AdmissionDisposition.ADMITTED:
            raise LogicPlatformAdmissionError(
                "rejected results cannot use admitted disposition"
            )
        if self.schema_version != ADMISSION_RESULT_SCHEMA:
            raise LogicPlatformAdmissionError(
                f"unsupported admission result schema: {self.schema_version!r}"
            )

    @property
    def may_affect_completion(self) -> bool:
        """True only after full ten-point admission (never partial)."""

        return self.admitted

    @property
    def may_affect_merge(self) -> bool:
        """True only after full ten-point admission (never partial)."""

        return self.admitted

    def check_map(self) -> Mapping[str, CheckResult]:
        return MappingProxyType({item.check.value: item for item in self.checks})

    def failed_checks(self) -> tuple[CheckResult, ...]:
        return tuple(item for item in self.checks if not item.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema_version,
            "interface": SUPERVISOR_LOGIC_PLATFORM_RECEIPT_ADMISSION_INTERFACE,
            "task_id": ADMISSION_TASK_ID,
            "goal_id": ADMISSION_GOAL_ID,
            "admitted": self.admitted,
            "disposition": self.disposition.value,
            "may_affect_completion": self.may_affect_completion,
            "may_affect_merge": self.may_affect_merge,
            "receipt_content_id": self.receipt_content_id,
            "authority_ceiling": self.authority_ceiling,
            "simulated": self.simulated,
            "freshness": self.freshness,
            "semantic_verdict": self.semantic_verdict,
            "reasons": list(self.reasons),
            "checks": [item.to_dict() for item in self.checks],
            "context": self.context.to_dict(),
            "ten_point_checks": list(TEN_POINT_CHECKS),
        }


@dataclass(frozen=True, slots=True)
class _NormalizedReceipt:
    """Internal projection of an untrusted receipt envelope."""

    raw: Mapping[str, Any]
    content_id: str | None
    claimed_content_id: str | None
    recomputed_content_id: str | None
    repository_id: str | None
    repository_tree_id: str | None
    policy_id: str | None
    policy_revision: str | None
    environment_id: str | None
    source_id: str | None
    plan_id: str | None
    obligation_id: str | None
    operation: str | None
    semantic_verdict: str
    evidence_kind: str | None
    authority_ceiling: str
    freshness: str
    simulated: bool
    reconstruction_passed: bool | None
    kernel_checked: bool | None
    translation_valid: bool | None
    translation_class: str | None
    translation_chain: tuple[Mapping[str, Any], ...]
    evidence: tuple[ProofEvidence, ...]
    proof_receipt: ProofReceipt | None
    structural_errors: tuple[str, ...]


def _unwrap_receipt_payload(value: Any) -> Mapping[str, Any] | None:
    """Accept raw receipts or LPC-110 projected envelopes ``{receipt: ...}``."""

    mapping = _as_mapping(value)
    if mapping is None:
        return None
    nested = mapping.get("receipt")
    nested_map = _as_mapping(nested)
    if nested_map is not None and (
        "obligation_id" in nested_map
        or "schema" in nested_map
        or "verdict" in nested_map
        or "evidence" in nested_map
        or "content_id" in nested_map
        or "receipt_id" in nested_map
    ):
        # Prefer the nested body when the outer envelope is a client projection.
        merged = dict(nested_map)
        for key in (
            "simulated",
            "authority_ceiling",
            "admitted",
            "trusted",
            "freshness",
        ):
            if key not in merged and key in mapping:
                merged[key] = mapping[key]
        return merged
    return mapping


def _parse_proof_receipt(payload: Mapping[str, Any]) -> ProofReceipt | None:
    schema = str(payload.get("schema") or "")
    if schema and schema != ProofReceipt.SCHEMA:
        return None
    # Require a minimum ProofReceipt field set before attempting parse.
    required = (
        "obligation_id",
        "plan_id",
        "attempt_id",
        "repository_id",
        "repository_tree_id",
        "policy_id",
        "verdict",
    )
    if not all(str(payload.get(name) or "").strip() for name in required):
        return None
    try:
        return ProofReceipt.from_dict(payload)
    except (ContractValidationError, TypeError, ValueError, KeyError):
        return None


def _parse_evidence_items(value: Any) -> tuple[ProofEvidence, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    items: list[ProofEvidence] = []
    for entry in value:
        if isinstance(entry, ProofEvidence):
            items.append(entry)
            continue
        mapping = _as_mapping(entry)
        if mapping is None:
            continue
        try:
            items.append(ProofEvidence.from_dict(mapping))
        except (ContractValidationError, TypeError, ValueError, KeyError):
            continue
    return tuple(items)


def _normalize_receipt(value: Any) -> _NormalizedReceipt:
    structural_errors: list[str] = []
    if isinstance(value, ProofReceipt):
        proof = value
        payload = proof.to_dict()
        evidence = proof.evidence
        content_id = proof.receipt_id
        metadata = proof.metadata if isinstance(proof.metadata, Mapping) else {}
        environment_id = None
        source_id = None
        try:
            environment_id = _optional_token(
                metadata.get("environment_id"),
                field_name="environment_id",
            )
            source_id = _optional_token(
                metadata.get("source_id"),
                field_name="source_id",
            )
        except LogicPlatformAdmissionError:
            environment_id = None
            source_id = None
        has_kernel = any(
            item.kind is EvidenceKind.KERNEL_VERIFICATION
            and item.verdict is EvidenceVerdict.ACCEPTED
            and item.independent
            and not item.simulated
            for item in evidence
        )
        return _NormalizedReceipt(
            raw=payload,
            content_id=content_id,
            claimed_content_id=content_id,
            recomputed_content_id=content_id,
            repository_id=proof.repository_id,
            repository_tree_id=proof.repository_tree_id,
            policy_id=proof.policy_id,
            policy_revision=None,
            environment_id=environment_id,
            source_id=source_id,
            plan_id=proof.plan_id,
            obligation_id=proof.obligation_id,
            operation=None,
            semantic_verdict=str(
                getattr(proof.authoritative_verdict, "value", proof.verdict.value)
            ),
            evidence_kind=(
                evidence[0].kind.value if evidence else EvidenceKind.UNKNOWN.value
            ),
            authority_ceiling=proof.authoritative_assurance.value,
            freshness=proof.freshness.value,
            simulated=any(item.simulated for item in evidence),
            reconstruction_passed=bool(proof.kernel_receipt_id or has_kernel),
            kernel_checked=has_kernel,
            translation_valid=None,
            translation_class=None,
            translation_chain=(),
            evidence=evidence,
            proof_receipt=proof,
            structural_errors=(),
        )

    payload = _unwrap_receipt_payload(value)
    if payload is None:
        return _NormalizedReceipt(
            raw={},
            content_id=None,
            claimed_content_id=None,
            recomputed_content_id=None,
            repository_id=None,
            repository_tree_id=None,
            policy_id=None,
            policy_revision=None,
            environment_id=None,
            source_id=None,
            plan_id=None,
            obligation_id=None,
            operation=None,
            semantic_verdict="unknown",
            evidence_kind=None,
            authority_ceiling=AssuranceLevel.UNVERIFIED.value,
            freshness=EvidenceFreshness.UNKNOWN.value,
            simulated=False,
            reconstruction_passed=None,
            kernel_checked=None,
            translation_valid=None,
            translation_class=None,
            translation_chain=(),
            evidence=(),
            proof_receipt=None,
            structural_errors=("receipt_not_object",),
        )

    if not all(isinstance(key, str) for key in payload):
        structural_errors.append("receipt_keys_must_be_strings")

    proof = _parse_proof_receipt(payload)
    evidence = _parse_evidence_items(payload.get("evidence"))
    if proof is not None and not evidence:
        evidence = proof.evidence

    claimed_content_id = _soft_optional_token(
        _first_present(payload, "content_id", "receipt_id", "receipt_content_id")
    )
    recomputed: str | None = None
    if proof is not None:
        recomputed = proof.receipt_id
    else:
        try:
            # Content identity of the public payload excluding mutable projections.
            identity_payload = {
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "content_id",
                    "receipt_id",
                    "receipt_content_id",
                    "admitted",
                    "trusted",
                    "ten_point_gate",
                }
            }
            recomputed = content_identity(identity_payload)
        except (ContractValidationError, TypeError, ValueError):
            recomputed = None
            structural_errors.append("content_identity_not_computable")

    content_id = claimed_content_id or recomputed

    verdict_raw = _first_present(
        payload,
        "semantic_verdict",
        "verdict",
        "authoritative_verdict",
    )
    if proof is not None and verdict_raw is None:
        semantic_verdict = proof.authoritative_verdict.value
    else:
        semantic_verdict = str(
            getattr(verdict_raw, "value", verdict_raw) or "unknown"
        ).strip().lower() or "unknown"

    evidence_kind = _normalize_evidence_kind(
        _first_present(payload, "evidence_kind", "kind")
    )
    if evidence_kind is None and evidence:
        evidence_kind = evidence[0].kind.value

    authority_raw = _first_present(
        payload,
        "authority_ceiling",
        "authoritative_assurance",
        "assurance",
        "evidence_authority",
    )
    if authority_raw is None and proof is not None:
        authority_ceiling = proof.authoritative_assurance.value
    elif authority_raw is None and evidence:
        try:
            authority_ceiling = assess_assurance(evidence).level.value
        except (ContractValidationError, TypeError, ValueError):
            authority_ceiling = AssuranceLevel.UNVERIFIED.value
    else:
        try:
            authority_ceiling = _normalize_authority(authority_raw)
        except LogicPlatformAdmissionError:
            authority_ceiling = AssuranceLevel.UNVERIFIED.value
            structural_errors.append("authority_ceiling_unrecognized")

    freshness = _normalize_freshness(
        _first_present(payload, "freshness", "freshness_status", "cache_freshness")
    )
    if proof is not None and _first_present(payload, "freshness") is None:
        freshness = proof.freshness.value

    simulated = _boolish(
        _first_present(payload, "simulated", "is_simulated"),
        default=False,
    )
    if evidence and any(item.simulated for item in evidence):
        simulated = True

    reconstruction_raw = _first_present(
        payload,
        "reconstruction_passed",
        "reconstruction_ok",
        "reconstructed",
        "kernel_reconstruction_passed",
    )
    reconstruction_passed: bool | None
    if reconstruction_raw is None:
        reconstruction_passed = None
    else:
        reconstruction_passed = _boolish(reconstruction_raw)

    kernel_raw = _first_present(
        payload,
        "kernel_checked",
        "kernel_verified",
        "kernel_ok",
    )
    kernel_checked: bool | None
    if kernel_raw is None:
        kernel_checked = None
    else:
        kernel_checked = _boolish(kernel_raw)

    if proof is not None:
        has_kernel = any(
            item.kind is EvidenceKind.KERNEL_VERIFICATION
            and item.verdict is EvidenceVerdict.ACCEPTED
            and item.independent
            and not item.simulated
            for item in proof.evidence
        )
        if reconstruction_passed is None:
            reconstruction_passed = bool(proof.kernel_receipt_id or has_kernel)
        if kernel_checked is None:
            kernel_checked = has_kernel

    translation = _as_mapping(
        _first_present(
            payload,
            "translation",
            "translation_chain",
            "translation_receipt",
        )
    )
    translation_chain: list[Mapping[str, Any]] = []
    translation_valid: bool | None = None
    translation_class: str | None = None
    chain_raw = payload.get("translation_chain")
    if isinstance(chain_raw, Sequence) and not isinstance(
        chain_raw, (str, bytes, bytearray)
    ):
        for entry in chain_raw:
            entry_map = _as_mapping(entry)
            if entry_map is not None:
                translation_chain.append(entry_map)
    elif translation is not None:
        translation_chain.append(translation)

    if translation is not None:
        if "valid" in translation:
            translation_valid = _boolish(translation.get("valid"))
        elif "accepted" in translation:
            translation_valid = _boolish(translation.get("accepted"))
        translation_class = _soft_optional_token(
            translation.get("translation_class") or translation.get("class")
        )
    elif translation_chain:
        # Conjunction over chain steps when present.
        flags = []
        classes = []
        for step in translation_chain:
            if "valid" in step:
                flags.append(_boolish(step.get("valid")))
            elif "accepted" in step:
                flags.append(_boolish(step.get("accepted")))
            cls = step.get("translation_class") or step.get("class")
            if isinstance(cls, str) and cls.strip():
                classes.append(cls.strip().lower())
        if flags:
            translation_valid = all(flags)
        if classes:
            translation_class = classes[-1]

    # Structural floors for generic envelopes.
    if not (
        payload.get("obligation_id")
        or payload.get("receipt_id")
        or payload.get("content_id")
        or payload.get("verdict")
        or payload.get("semantic_verdict")
        or payload.get("evidence")
        or proof is not None
    ):
        structural_errors.append("receipt_missing_identity_fields")

    return _NormalizedReceipt(
        raw=dict(payload),
        content_id=content_id,
        claimed_content_id=claimed_content_id,
        recomputed_content_id=recomputed,
        repository_id=_soft_optional_token(
            _first_present(payload, "repository_id")
        )
        or (proof.repository_id if proof is not None else None),
        repository_tree_id=_soft_optional_token(
            _first_present(
                payload,
                "repository_tree_id",
                "tree_id",
                "repository_tree",
            )
        )
        or (proof.repository_tree_id if proof is not None else None),
        policy_id=_soft_optional_token(_first_present(payload, "policy_id"))
        or (proof.policy_id if proof is not None else None),
        policy_revision=_soft_optional_token(
            _first_present(payload, "policy_revision")
        ),
        environment_id=_soft_optional_token(
            _first_present(
                payload,
                "environment_id",
                "environment",
                "validation_environment_id",
            )
        ),
        source_id=_soft_optional_token(
            _first_present(payload, "source_id", "source", "source_digest")
        ),
        plan_id=_soft_optional_token(_first_present(payload, "plan_id"))
        or (proof.plan_id if proof is not None else None),
        obligation_id=_soft_optional_token(
            _first_present(payload, "obligation_id")
        )
        or (proof.obligation_id if proof is not None else None),
        operation=_soft_optional_token(
            _first_present(payload, "operation", "requested_operation")
        ),
        semantic_verdict=semantic_verdict,
        evidence_kind=evidence_kind,
        authority_ceiling=authority_ceiling,
        freshness=freshness,
        simulated=simulated,
        reconstruction_passed=reconstruction_passed,
        kernel_checked=kernel_checked,
        translation_valid=translation_valid,
        translation_class=translation_class,
        translation_chain=tuple(translation_chain),
        evidence=evidence,
        proof_receipt=proof,
        structural_errors=tuple(structural_errors),
    )


def _pass(check: AdmissionCheck, reason_code: str, detail: str = "") -> CheckResult:
    return CheckResult(
        check=check, passed=True, reason_code=reason_code, detail=detail
    )


def _fail(check: AdmissionCheck, reason_code: str, detail: str = "") -> CheckResult:
    return CheckResult(
        check=check, passed=False, reason_code=reason_code, detail=detail
    )


def _check_structural(receipt: _NormalizedReceipt) -> CheckResult:
    if receipt.structural_errors:
        return _fail(
            AdmissionCheck.STRUCTURAL_VALIDITY,
            receipt.structural_errors[0],
            ";".join(receipt.structural_errors),
        )
    if not receipt.raw and receipt.proof_receipt is None:
        return _fail(
            AdmissionCheck.STRUCTURAL_VALIDITY,
            "receipt_empty",
        )
    # Require a stable identity handle.
    if not (
        receipt.content_id
        or receipt.obligation_id
        or receipt.proof_receipt is not None
    ):
        return _fail(
            AdmissionCheck.STRUCTURAL_VALIDITY,
            "receipt_missing_identity",
        )
    return _pass(AdmissionCheck.STRUCTURAL_VALIDITY, "structurally_valid")


def _check_content_identity(
    receipt: _NormalizedReceipt,
    context: AdmissionContext,
) -> CheckResult:
    if receipt.claimed_content_id and receipt.recomputed_content_id:
        if receipt.claimed_content_id != receipt.recomputed_content_id:
            # Allow claim when the claim is a well-formed external id and the
            # recomputed digest is only an envelope projection hash — but only
            # when the claim itself is well-formed and matches expected.
            if not _CONTENT_ID_RE.match(receipt.claimed_content_id):
                return _fail(
                    AdmissionCheck.CONTENT_IDENTITY,
                    "content_identity_malformed",
                    receipt.claimed_content_id,
                )
            if (
                context.expected_content_id
                and receipt.claimed_content_id != context.expected_content_id
            ):
                return _fail(
                    AdmissionCheck.CONTENT_IDENTITY,
                    "content_identity_mismatch",
                )
            # For ProofReceipt objects, claim must equal recomputed exactly.
            if receipt.proof_receipt is not None:
                return _fail(
                    AdmissionCheck.CONTENT_IDENTITY,
                    "content_identity_mismatch",
                )
    content_id = receipt.content_id
    if not content_id:
        return _fail(
            AdmissionCheck.CONTENT_IDENTITY,
            "content_identity_missing",
        )
    if not _CONTENT_ID_RE.match(content_id):
        return _fail(
            AdmissionCheck.CONTENT_IDENTITY,
            "content_identity_malformed",
            content_id,
        )
    if (
        context.expected_content_id
        and content_id != context.expected_content_id
    ):
        return _fail(
            AdmissionCheck.CONTENT_IDENTITY,
            "content_identity_mismatch",
        )
    return _pass(
        AdmissionCheck.CONTENT_IDENTITY,
        "content_identity_valid",
        content_id,
    )


def _check_bindings(
    receipt: _NormalizedReceipt,
    context: AdmissionContext,
) -> CheckResult:
    if not receipt.repository_tree_id:
        return _fail(
            AdmissionCheck.BINDINGS,
            "repository_tree_id_missing",
        )
    if receipt.repository_tree_id != context.repository_tree_id:
        return _fail(
            AdmissionCheck.BINDINGS,
            "repository_tree_id_mismatch",
        )
    if not receipt.policy_id:
        return _fail(AdmissionCheck.BINDINGS, "policy_id_missing")
    if receipt.policy_id != context.policy_id:
        return _fail(AdmissionCheck.BINDINGS, "policy_id_mismatch")
    if (
        context.repository_id
        and receipt.repository_id
        and receipt.repository_id != context.repository_id
    ):
        return _fail(AdmissionCheck.BINDINGS, "repository_id_mismatch")
    if (
        context.environment_id
        and receipt.environment_id
        and receipt.environment_id != context.environment_id
    ):
        return _fail(AdmissionCheck.BINDINGS, "environment_id_mismatch")
    if (
        context.source_id
        and receipt.source_id
        and receipt.source_id != context.source_id
    ):
        return _fail(AdmissionCheck.BINDINGS, "source_id_mismatch")
    if (
        context.policy_revision
        and receipt.policy_revision
        and receipt.policy_revision != context.policy_revision
    ):
        return _fail(AdmissionCheck.BINDINGS, "policy_revision_mismatch")
    if (
        context.plan_id
        and receipt.plan_id
        and receipt.plan_id != context.plan_id
    ):
        return _fail(AdmissionCheck.BINDINGS, "plan_id_mismatch")
    if (
        context.obligation_id
        and receipt.obligation_id
        and receipt.obligation_id != context.obligation_id
    ):
        return _fail(AdmissionCheck.BINDINGS, "obligation_id_mismatch")
    return _pass(AdmissionCheck.BINDINGS, "bindings_match")


def _check_translation_chain(
    receipt: _NormalizedReceipt,
    context: AdmissionContext,
) -> CheckResult:
    # Translation is required when the receipt claims a translation or when
    # kernel-required completion influence is sought and a chain is present.
    if not receipt.translation_chain and receipt.translation_valid is None:
        # No translation claim: acceptable for pure kernel receipts that bind
        # translator_id on ProofReceipt, or for non-translated evidence.
        if receipt.proof_receipt is not None:
            if not str(receipt.proof_receipt.translator_id or "").strip():
                if (
                    _authority_rank(context.required_authority)
                    >= _authority_rank(AssuranceLevel.KERNEL_VERIFIED.value)
                    and receipt.semantic_verdict in _CONCLUSIVE_VERDICTS
                ):
                    return _fail(
                        AdmissionCheck.TRANSLATION_CHAIN,
                        "translator_id_missing",
                    )
            return _pass(
                AdmissionCheck.TRANSLATION_CHAIN,
                "translation_not_required_or_bound",
            )
        # Generic envelopes without translation claims pass only when not
        # claiming proved under kernel policy.
        if (
            receipt.semantic_verdict in _CONCLUSIVE_VERDICTS
            and _authority_rank(context.required_authority)
            >= _authority_rank(AssuranceLevel.KERNEL_VERIFIED.value)
        ):
            return _fail(
                AdmissionCheck.TRANSLATION_CHAIN,
                "translation_chain_missing",
            )
        return _pass(
            AdmissionCheck.TRANSLATION_CHAIN,
            "translation_not_claimed",
        )

    if receipt.translation_valid is False:
        return _fail(
            AdmissionCheck.TRANSLATION_CHAIN,
            "translation_chain_invalid",
        )
    if receipt.translation_class:
        cls = receipt.translation_class.lower()
        if cls == "heuristic":
            return _fail(
                AdmissionCheck.TRANSLATION_CHAIN,
                "translation_class_heuristic",
            )
        if (
            _authority_rank(context.required_authority)
            >= _authority_rank(AssuranceLevel.KERNEL_VERIFIED.value)
            and cls not in _KERNEL_REQUIRED_TRANSLATION
        ):
            return _fail(
                AdmissionCheck.TRANSLATION_CHAIN,
                "translation_class_insufficient",
                cls,
            )
        if cls not in _VALID_TRANSLATION_CLASSES and cls not in {
            "exact",
            "equisatisfiable",
            "bounded_abstraction",
            "conservative_approximation",
        }:
            return _fail(
                AdmissionCheck.TRANSLATION_CHAIN,
                "translation_class_unknown",
                cls,
            )
    if receipt.translation_valid is None and receipt.translation_chain:
        # Chain present without validity flag: fail closed.
        return _fail(
            AdmissionCheck.TRANSLATION_CHAIN,
            "translation_validity_unspecified",
        )
    return _pass(AdmissionCheck.TRANSLATION_CHAIN, "translation_chain_valid")


def _check_evidence_kind(
    receipt: _NormalizedReceipt,
    context: AdmissionContext,
) -> CheckResult:
    kind = receipt.evidence_kind or EvidenceKind.UNKNOWN.value
    verdict = receipt.semantic_verdict
    if verdict in _CONCLUSIVE_VERDICTS:
        if kind in _NON_KERNEL_EVIDENCE:
            return _fail(
                AdmissionCheck.EVIDENCE_KIND,
                "evidence_kind_does_not_support_verdict",
                f"{kind}:{verdict}",
            )
        if (
            _authority_rank(context.required_authority)
            >= _authority_rank(AssuranceLevel.KERNEL_VERIFIED.value)
            and kind
            not in {
                EvidenceKind.KERNEL_VERIFICATION.value,
                EvidenceKind.CRYPTOGRAPHIC_ATTESTATION.value,
            }
        ):
            return _fail(
                AdmissionCheck.EVIDENCE_KIND,
                "evidence_kind_below_kernel",
                kind,
            )
    # Success-only envelopes with unknown kind never support completion.
    if kind in {EvidenceKind.UNKNOWN.value, None, ""} and verdict in _CONCLUSIVE_VERDICTS:
        return _fail(
            AdmissionCheck.EVIDENCE_KIND,
            "evidence_kind_unknown",
        )
    return _pass(
        AdmissionCheck.EVIDENCE_KIND,
        "evidence_kind_supports_verdict",
        kind,
    )


def _check_authority_ceiling(
    receipt: _NormalizedReceipt,
    context: AdmissionContext,
) -> CheckResult:
    try:
        actual_rank = _authority_rank(receipt.authority_ceiling)
        required_rank = _authority_rank(context.required_authority)
    except LogicPlatformAdmissionError as error:
        return _fail(
            AdmissionCheck.AUTHORITY_CEILING,
            "authority_ceiling_unrecognized",
            str(error),
        )
    if actual_rank < required_rank:
        return _fail(
            AdmissionCheck.AUTHORITY_CEILING,
            "authority_ceiling_insufficient",
            f"{receipt.authority_ceiling}<{context.required_authority}",
        )
    # Provider success never upgrades authority: if envelope only claims
    # operation success without semantic authority, reject.
    op_status = str(receipt.raw.get("operation_status") or "").strip().lower()
    if (
        op_status == "succeeded"
        and receipt.semantic_verdict in {"unknown", "inconclusive", ""}
        and actual_rank
        >= _authority_rank(AssuranceLevel.KERNEL_VERIFIED.value)
    ):
        return _fail(
            AdmissionCheck.AUTHORITY_CEILING,
            "success_does_not_imply_authority",
        )
    return _pass(
        AdmissionCheck.AUTHORITY_CEILING,
        "authority_ceiling_adequate",
        receipt.authority_ceiling,
    )


def _check_reconstruction(
    receipt: _NormalizedReceipt,
    context: AdmissionContext,
) -> CheckResult:
    if not context.require_reconstruction and not context.require_kernel:
        return _pass(
            AdmissionCheck.REQUIRED_RECONSTRUCTION,
            "reconstruction_not_required",
        )
    if context.require_reconstruction:
        if receipt.reconstruction_passed is not True:
            return _fail(
                AdmissionCheck.REQUIRED_RECONSTRUCTION,
                "reconstruction_not_passed",
            )
    if context.require_kernel:
        if receipt.kernel_checked is not True:
            # Fall back to typed evidence inspection.
            has_kernel = any(
                item.kind is EvidenceKind.KERNEL_VERIFICATION
                and item.verdict is EvidenceVerdict.ACCEPTED
                and item.independent
                and not item.simulated
                for item in receipt.evidence
            )
            if not has_kernel:
                return _fail(
                    AdmissionCheck.REQUIRED_RECONSTRUCTION,
                    "kernel_check_not_passed",
                )
    return _pass(
        AdmissionCheck.REQUIRED_RECONSTRUCTION,
        "reconstruction_and_kernel_passed",
    )


def _check_freshness(receipt: _NormalizedReceipt) -> CheckResult:
    if receipt.freshness != EvidenceFreshness.CURRENT.value:
        return _fail(
            AdmissionCheck.FRESHNESS,
            "receipt_stale_or_unknown",
            receipt.freshness,
        )
    if any(
        item.freshness is not EvidenceFreshness.CURRENT for item in receipt.evidence
    ):
        return _fail(
            AdmissionCheck.FRESHNESS,
            "evidence_stale_or_unknown",
        )
    return _pass(AdmissionCheck.FRESHNESS, "fresh")


def _check_non_simulation(receipt: _NormalizedReceipt) -> CheckResult:
    if receipt.simulated:
        return _fail(
            AdmissionCheck.NON_SIMULATION,
            "receipt_simulated",
        )
    if any(item.simulated for item in receipt.evidence):
        return _fail(
            AdmissionCheck.NON_SIMULATION,
            "evidence_simulated",
        )
    return _pass(AdmissionCheck.NON_SIMULATION, "not_simulated")


def _check_policy_admission(
    receipt: _NormalizedReceipt,
    context: AdmissionContext,
) -> CheckResult:
    # Policy admits authority for the requested operation only when:
    # 1. operation matches (when receipt declares one),
    # 2. authority meets required ceiling,
    # 3. network policy is not overclaimed,
    # 4. simulated / candidate evidence cannot satisfy kernel-required policy.
    if receipt.operation and receipt.operation != context.operation:
        return _fail(
            AdmissionCheck.POLICY_ADMISSION,
            "operation_mismatch",
            f"{receipt.operation}!={context.operation}",
        )
    try:
        if _authority_rank(receipt.authority_ceiling) < _authority_rank(
            context.required_authority
        ):
            return _fail(
                AdmissionCheck.POLICY_ADMISSION,
                "policy_authority_insufficient",
            )
    except LogicPlatformAdmissionError:
        return _fail(
            AdmissionCheck.POLICY_ADMISSION,
            "policy_authority_unrecognized",
        )

    receipt_network = _boolish(
        _first_present(receipt.raw, "network_allowed", "network"),
        default=False,
    )
    if receipt_network and not context.network_allowed:
        return _fail(
            AdmissionCheck.POLICY_ADMISSION,
            "network_policy_denied",
        )

    if (
        _authority_rank(context.required_authority)
        >= _authority_rank(AssuranceLevel.KERNEL_VERIFIED.value)
        and (
            receipt.simulated
            or (receipt.evidence_kind or "") in _NON_KERNEL_EVIDENCE
            or _authority_rank(receipt.authority_ceiling)
            < _authority_rank(AssuranceLevel.KERNEL_VERIFIED.value)
        )
    ):
        return _fail(
            AdmissionCheck.POLICY_ADMISSION,
            "policy_rejects_authority_for_operation",
        )

    # Explicit policy denial flags on the envelope.
    if _boolish(receipt.raw.get("policy_admitted"), default=True) is False:
        return _fail(
            AdmissionCheck.POLICY_ADMISSION,
            "policy_explicitly_denied",
        )
    return _pass(
        AdmissionCheck.POLICY_ADMISSION,
        "policy_admits_authority",
        context.operation,
    )


def admit_receipt(
    receipt: Any,
    context: AdmissionContext,
) -> AdmissionResult:
    """Evaluate the ten-point floor against an untrusted receipt.

    Returns an :class:`AdmissionResult` whose ``admitted`` flag is True only
    when every check passes. Partial success never sets
    ``may_affect_completion`` or ``may_affect_merge``.
    """

    if not isinstance(context, AdmissionContext):
        raise LogicPlatformAdmissionError("context must be an AdmissionContext")

    normalized = _normalize_receipt(receipt)
    checks = (
        _check_structural(normalized),
        _check_content_identity(normalized, context),
        _check_bindings(normalized, context),
        _check_translation_chain(normalized, context),
        _check_evidence_kind(normalized, context),
        _check_authority_ceiling(normalized, context),
        _check_reconstruction(normalized, context),
        _check_freshness(normalized),
        _check_non_simulation(normalized),
        _check_policy_admission(normalized, context),
    )
    admitted = all(item.passed for item in checks)
    reasons = tuple(
        item.reason_code for item in checks if not item.passed
    ) or (("all_ten_points_passed",) if admitted else ())
    return AdmissionResult(
        admitted=admitted,
        disposition=(
            AdmissionDisposition.ADMITTED
            if admitted
            else AdmissionDisposition.REJECTED
        ),
        checks=checks,
        context=context,
        receipt_content_id=normalized.content_id,
        authority_ceiling=normalized.authority_ceiling,
        simulated=normalized.simulated,
        freshness=normalized.freshness,
        semantic_verdict=normalized.semantic_verdict,
        reasons=reasons,
    )


def may_affect_completion_or_merge(
    receipt: Any,
    context: AdmissionContext,
) -> bool:
    """Return True only after full ten-point admission."""

    return admit_receipt(receipt, context).admitted


def admit_receipts(
    receipts: Iterable[Any],
    context: AdmissionContext,
) -> tuple[AdmissionResult, ...]:
    """Admit a sequence of receipts against the same context."""

    return tuple(admit_receipt(item, context) for item in receipts)


@dataclass(frozen=True, slots=True)
class SupervisorLogicPlatformReceiptAdmission:
    """Stable interface object for supervisor receipt admission."""

    interface: str = SUPERVISOR_LOGIC_PLATFORM_RECEIPT_ADMISSION_INTERFACE
    version: str = SUPERVISOR_LOGIC_PLATFORM_RECEIPT_ADMISSION_VERSION
    schema_version: str = ADMISSION_SCHEMA_VERSION
    task_id: str = ADMISSION_TASK_ID
    goal_id: str = ADMISSION_GOAL_ID

    def admit(
        self,
        receipt: Any,
        context: AdmissionContext,
    ) -> AdmissionResult:
        return admit_receipt(receipt, context)

    def may_affect_completion_or_merge(
        self,
        receipt: Any,
        context: AdmissionContext,
    ) -> bool:
        return may_affect_completion_or_merge(receipt, context)

    def ten_point_checks(self) -> tuple[str, ...]:
        return TEN_POINT_CHECKS

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "version": self.version,
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "ten_point_checks": list(TEN_POINT_CHECKS),
        }


def get_receipt_admission() -> SupervisorLogicPlatformReceiptAdmission:
    """Return a process-local admission helper instance."""

    return SupervisorLogicPlatformReceiptAdmission()


__all__ = (
    "ADMISSION_CONTEXT_SCHEMA",
    "ADMISSION_GOAL_ID",
    "ADMISSION_RESULT_SCHEMA",
    "ADMISSION_SCHEMA_VERSION",
    "ADMISSION_TASK_ID",
    "AdmissionCheck",
    "AdmissionContext",
    "AdmissionDisposition",
    "AdmissionResult",
    "CheckResult",
    "LogicPlatformAdmissionError",
    "SUPERVISOR_LOGIC_PLATFORM_RECEIPT_ADMISSION_INTERFACE",
    "SUPERVISOR_LOGIC_PLATFORM_RECEIPT_ADMISSION_VERSION",
    "SupervisorLogicPlatformReceiptAdmission",
    "TEN_POINT_CHECKS",
    "admit_receipt",
    "admit_receipts",
    "get_receipt_admission",
    "may_affect_completion_or_merge",
)
