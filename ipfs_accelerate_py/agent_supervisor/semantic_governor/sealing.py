"""Gate promotion on release qualification and bind incremental seals (SCG-035).

``SemanticGovernorSealAdapter``, :func:`qualify_policy_candidate`,
:func:`seal_governor_run`, and :func:`verify_governor_seal` enforce the
normative sealing and promotion-qualification invariants:

* A missing released sealer is typed ``unavailable`` and is **never**
  satisfied by an IVP ``VerificationCommitment`` (structural non-ZK only).
* Promotion stays blocked unless either current released incremental-seal
  evidence is present, or an independently authorized
  ``VerificationBundle``-backed release-qualification path passes.
* Signed/sealed artifacts bind evaluated policy and evaluation identities and
  encode only the closed bounded claim set — never semantic sufficiency,
  universal completeness, or ZK/execution proof by substitution.

Conflict policy: use released ``IncrementalProofSealer`` only; otherwise
typed unavailable. Post-decision sealing binds but never upgrades evidence
semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Mapping, Sequence
import re
import unicodedata

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    SemanticGovernorBaseError,
    reject_private_and_model_authority,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.policy_contracts import (
    EvaluationVerdict,
    PolicyContractError,
    RuleEvaluationReport,
)

from ipfs_accelerate_py.agent_supervisor.semantic_governor.adapters import (
    EXPECTED_VERIFICATION_BUNDLE_INTERFACE,
    EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
    EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
    IncrementalSealerCapability,
    SealStatus,
    probe_incremental_sealer_capability,
    reject_ivp_commitment_as_sealer,
    sealer_capability_from_evidence,
)

# ---------------------------------------------------------------------------
# Evidence / interface / schema constants
# ---------------------------------------------------------------------------

SCG_SEAL_BINDING_EVIDENCE: Final[str] = "scg/seal-binding@1"
SCG_RELEASE_QUALIFICATION_EVIDENCE: Final[str] = "scg/release-qualification@1"

SEMANTIC_GOVERNOR_SEAL_ADAPTER_INTERFACE: Final[str] = (
    "SemanticGovernorSealAdapter@1"
)
QUALIFY_POLICY_CANDIDATE_INTERFACE: Final[str] = "qualify_policy_candidate@1"
SEAL_GOVERNOR_RUN_INTERFACE: Final[str] = "seal_governor_run@1"
VERIFY_GOVERNOR_SEAL_INTERFACE: Final[str] = "verify_governor_seal@1"

RELEASE_QUALIFICATION_INTERFACE: Final[str] = "ReleaseQualification@1"
RELEASE_QUALIFICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "release-qualification@1"
)

GOVERNOR_SEAL_INTERFACE: Final[str] = "GovernorSeal@1"
GOVERNOR_SEAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/governor-seal@1"
)

BOUNDED_CLAIM_SET_INTERFACE: Final[str] = "BoundedSealClaimSet@1"
BOUNDED_CLAIM_SET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "bounded-seal-claim-set@1"
)

ARTIFACT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "seal-artifact-binding@1"
)

GENERATOR_ID: Final[str] = "semantic_governor_sealer"
GENERATOR_VERSION: Final[str] = "1.0.0"
PRODUCER_ID: Final[str] = "semantic_governor"
PRODUCER_VERSION: Final[str] = "1"
TOOL_ID: Final[str] = "sealing.v1"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_CID_LIST: Final[int] = 4_096
MAX_CLAIMS: Final[int] = 64
MAX_BLOCKING_REASONS: Final[int] = 256
MAX_METADATA_KEYS: Final[int] = 64
MAX_ARTIFACT_BINDINGS: Final[int] = 1_024
MAX_DIAGNOSTIC: Final[int] = 512

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")

# Closed reason codes for qualification / sealing fail-closed paths.
REASON_SEALER_UNAVAILABLE: Final[str] = "sealer_unavailable"
REASON_IVP_COMMITMENT_NOT_SEALER: Final[str] = "ivp_commitment_not_sealer"
REASON_MISSING_RELEASE_QUALIFICATION: Final[str] = "missing_release_qualification"
REASON_MISSING_QUALIFICATION_AUTHORIZATION: Final[str] = (
    "missing_release_qualification_authorization"
)
REASON_SELF_AUTHORIZATION: Final[str] = "self_authorization_forbidden"
REASON_EVALUATION_NOT_PASS: Final[str] = "evaluation_verdict_not_pass"
REASON_MISSING_EVALUATION: Final[str] = "missing_evaluation_report"
REASON_MISSING_VERIFICATION_BUNDLE: Final[str] = (
    "missing_release_qualification_verification_bundle"
)
REASON_BUNDLE_IS_COMMITMENT: Final[str] = (
    "verification_commitment_cannot_back_release_qualification"
)
REASON_IDENTITY_MISMATCH: Final[str] = "policy_or_evaluation_identity_mismatch"
REASON_OVERCLAIM: Final[str] = "seal_overclaim_rejected"
REASON_SEAL_STATUS_MISMATCH: Final[str] = "seal_status_capability_mismatch"
REASON_STALE_SEAL: Final[str] = "stale_or_tampered_seal"
REASON_MISSING_INCREMENTAL_SEAL_EVIDENCE: Final[str] = (
    "missing_incremental_seal_evidence"
)
REASON_PROMOTION_BLOCKED: Final[str] = "promotion_blocked"
REASON_SCHEMA_INTEGRITY: Final[str] = "schema_or_integrity_failure"
REASON_UNAUTHORIZED_CLAIM: Final[str] = "unauthorized_claim_kind"
REASON_BINDING_MISMATCH: Final[str] = "artifact_binding_mismatch"

# Artifact roles that may be bound into a governor seal (plan §14).
ARTIFACT_ROLE_BENCHMARK: Final[str] = "benchmark"
ARTIFACT_ROLE_CONTEXT_PACK: Final[str] = "context_pack"
ARTIFACT_ROLE_VERIFICATION_BUNDLE: Final[str] = "verification_bundle"
ARTIFACT_ROLE_DIFFERENTIAL_REPORT: Final[str] = "differential_report"
ARTIFACT_ROLE_CALIBRATION_PROFILE: Final[str] = "calibration_profile"
ARTIFACT_ROLE_CANDIDATE: Final[str] = "candidate"
ARTIFACT_ROLE_PROMOTION_DECISION: Final[str] = "promotion_decision"
ARTIFACT_ROLE_EVALUATION_REPORT: Final[str] = "evaluation_report"
ARTIFACT_ROLE_POLICY: Final[str] = "policy"
ARTIFACT_ROLE_INCREMENTAL_SEAL: Final[str] = "incremental_seal"

KNOWN_ARTIFACT_ROLES: Final[frozenset[str]] = frozenset(
    {
        ARTIFACT_ROLE_BENCHMARK,
        ARTIFACT_ROLE_CONTEXT_PACK,
        ARTIFACT_ROLE_VERIFICATION_BUNDLE,
        ARTIFACT_ROLE_DIFFERENTIAL_REPORT,
        ARTIFACT_ROLE_CALIBRATION_PROFILE,
        ARTIFACT_ROLE_CANDIDATE,
        ARTIFACT_ROLE_PROMOTION_DECISION,
        ARTIFACT_ROLE_EVALUATION_REPORT,
        ARTIFACT_ROLE_POLICY,
        ARTIFACT_ROLE_INCREMENTAL_SEAL,
    }
)

# Explicit non-claims — never encoded unless separately proven.
FORBIDDEN_CLAIM_KINDS: Final[frozenset[str]] = frozenset(
    {
        "semantic_sufficiency",
        "universal_semantic_completeness",
        "zk_proof",
        "zero_knowledge_proof",
        "execution_proof",
        "proof_of_test_execution",
        "ivp_commitment_is_sealer",
        "full_suite_implied",
        "model_agreement_is_equivalence",
    }
)


class SealingError(SemanticGovernorBaseError):
    """Raised when sealing or release-qualification inputs are malformed/unsafe."""


class QualificationPath(str, Enum):
    """Closed paths that may satisfy the pre-promotion release gate."""

    INCREMENTAL_SEAL = "incremental_seal"
    AUTHORIZED_RELEASE_QUALIFICATION = "authorized_release_qualification"
    BLOCKED = "blocked"


class BoundedClaimKind(str, Enum):
    """Closed bounded claims a governor seal may encode (plan §14).

    These establish only that exact artifacts were evaluated, required
    evaluations completed, declared thresholds applied, no blocking status was
    omitted, and the promoted policy equals the evaluated candidate. They do
    **not** prove semantic sufficiency.
    """

    EXACT_ARTIFACTS_EVALUATED = "exact_artifacts_evaluated"
    REQUIRED_EVALUATIONS_COMPLETED = "required_evaluations_completed"
    DECLARED_THRESHOLDS_APPLIED = "declared_thresholds_applied"
    NO_BLOCKING_STATUS_OMITTED = "no_blocking_status_omitted"
    PROMOTED_POLICY_EQUALS_EVALUATED_CANDIDATE = (
        "promoted_policy_equals_evaluated_candidate"
    )


DEFAULT_BOUNDED_CLAIMS: Final[tuple[str, ...]] = tuple(
    item.value for item in BoundedClaimKind
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise SealingError(f"{name} must be a nonempty string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise SealingError(f"{name} must be trimmed NFC text")
    if len(value) > MAX_TEXT_CHARS or any(not char.isprintable() for char in value):
        raise SealingError(f"{name} contains invalid text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise SealingError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise SealingError(
            f"{name} must be a lowercase token matching {_TOKEN_RE.pattern}"
        )
    return text


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise SealingError(f"{name} must be a boolean")
    return value


def _enum_value(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    if type(value) is str:
        try:
            return enum_type(value).value
        except ValueError as exc:
            raise SealingError(
                f"{name} has unsupported value {value!r}"
            ) from exc
    raise SealingError(f"{name} must be a {enum_type.__name__} or string")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise SealingError(f"{name} must be a mapping")
    if len(value) > MAX_METADATA_KEYS:
        raise SealingError(f"{name} exceeds {MAX_METADATA_KEYS} keys")
    try:
        reject_private_and_model_authority(value, path=name)
    except SemanticGovernorBaseError as exc:
        raise SealingError(str(exc)) from exc
    frozen: dict[str, Any] = {}
    for key in value:
        token = _token(key, f"{name}.key")
        item = value[key]
        frozen[token] = dict(item) if isinstance(item, Mapping) else item
    return MappingProxyType(frozen)


def _cid_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise SealingError(f"{name} must be a sequence of CIDs")
    if len(value) > MAX_CID_LIST:
        raise SealingError(f"{name} exceeds {MAX_CID_LIST} entries")
    out: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        cid = _cid(item, f"{name}[{index}]")
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return tuple(out)


def _looks_like_ivp_commitment(evidence: Any) -> bool:
    """Mirror adapter IVP detection; commitments never satisfy sealer paths."""

    if evidence is None:
        return False
    if type(evidence) is str:
        text = evidence.strip()
        return text in {
            "VerificationCommitment",
            "build_verification_commitment",
            EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
            EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
            "VerificationCommitment@1",
        }
    name = getattr(evidence, "__name__", None)
    if type(name) is str and name in {
        "VerificationCommitment",
        "build_verification_commitment",
    }:
        return True
    cls_name = type(evidence).__name__
    if cls_name in {"VerificationCommitment", "build_verification_commitment"}:
        return True
    if isinstance(evidence, Mapping):
        schema = evidence.get("schema") or evidence.get("interface_id")
        if schema in {
            EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
            EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
        }:
            return True
        if evidence.get("kind") == "verification_commitment":
            return True
        if evidence.get("is_zero_knowledge_proof") is True and evidence.get(
            "schema"
        ) in {EXPECTED_VERIFICATION_COMMITMENT_SCHEMA, None}:
            # Commitments that forge ZK remain non-sealers.
            if "commitment" in str(schema or evidence.get("kind") or "").lower():
                return True
    iface = getattr(evidence, "interface_id", None) or getattr(
        evidence, "schema", None
    )
    if iface in {
        EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
        EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
    }:
        return True
    if getattr(evidence, "IS_ZERO_KNOWLEDGE_PROOF", None) is False and cls_name == (
        "VerificationCommitment"
    ):
        return True
    return False


def _looks_like_verification_bundle(evidence: Any) -> bool:
    if evidence is None:
        return False
    if _looks_like_ivp_commitment(evidence):
        return False
    if type(evidence) is str:
        text = evidence.strip()
        return text in {
            "VerificationBundle",
            EXPECTED_VERIFICATION_BUNDLE_INTERFACE,
            "VerificationBundle@1",
        }
    name = getattr(evidence, "__name__", None)
    if type(name) is str and name == "VerificationBundle":
        return True
    if type(evidence).__name__ == "VerificationBundle":
        return True
    if isinstance(evidence, Mapping):
        schema = evidence.get("schema") or evidence.get("interface_id")
        if schema in {
            EXPECTED_VERIFICATION_BUNDLE_INTERFACE,
            "VerificationBundle@1",
            "ipfs_accelerate_py/agent-supervisor/verification-bundle@1",
        }:
            return True
        if evidence.get("kind") == "verification_bundle":
            return True
        # Structural shape: plan-bound receipts.
        if "verification_plan" in evidence and "receipts" in evidence:
            return True
    iface = getattr(evidence, "interface_id", None) or getattr(
        evidence, "INTERFACE", None
    )
    if iface in {
        EXPECTED_VERIFICATION_BUNDLE_INTERFACE,
        "VerificationBundle@1",
    }:
        return True
    if hasattr(evidence, "verification_plan") and hasattr(evidence, "receipts"):
        return True
    return False


def _jsonable(value: Any) -> Any:
    """Convert nested structures to CID-safe JSON types (no tuples/sets)."""

    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    # Unsupported host objects are not admitted into seal identity.
    raise SealingError(
        f"unsupported structured value type {type(value).__name__} for seal identity"
    )


def _bundle_identity_cid(bundle: Any) -> str:
    """Content-address a VerificationBundle-like object without upgrading it."""

    if isinstance(bundle, Mapping):
        payload = _jsonable(dict(bundle))
        if isinstance(payload, dict):
            payload.pop("bundle_cid", None)
            payload.pop("receipt_cid", None)
        return cid_for_structured(
            {
                "kind": "verification_bundle_identity",
                "payload": payload,
            }
        )
    if hasattr(bundle, "to_record") and callable(bundle.to_record):
        record = bundle.to_record()
        if isinstance(record, Mapping):
            payload = _jsonable(dict(record))
            if isinstance(payload, dict):
                payload.pop("bundle_cid", None)
            return cid_for_structured(
                {
                    "kind": "verification_bundle_identity",
                    "payload": payload,
                }
            )
    if hasattr(bundle, "to_dict") and callable(bundle.to_dict):
        record = bundle.to_dict()
        if isinstance(record, Mapping):
            payload = _jsonable(dict(record))
            if isinstance(payload, dict):
                payload.pop("bundle_cid", None)
            return cid_for_structured(
                {
                    "kind": "verification_bundle_identity",
                    "payload": payload,
                }
            )
    # Last resort: identity by type + repr is forbidden; require CID field.
    for attr in ("bundle_cid", "content_id", "cid"):
        value = getattr(bundle, attr, None)
        if type(value) is str and value:
            return _cid(value, attr)
    raise SealingError(
        "release qualification VerificationBundle must expose to_record, "
        "to_dict, a mapping body, or a content CID"
    )


def _extract_report_fields(
    evaluation: RuleEvaluationReport | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if evaluation is None:
        raise SealingError(REASON_MISSING_EVALUATION)
    if isinstance(evaluation, RuleEvaluationReport):
        verdict = evaluation.verdict
        if isinstance(verdict, EvaluationVerdict):
            verdict_value = verdict.value
        else:
            verdict_value = str(verdict)
        return {
            "report_cid": evaluation.report_cid,
            "candidate_cid": evaluation.candidate_cid,
            "held_out_benchmark_cid": evaluation.held_out_benchmark_cid,
            "baseline_policy_cid": evaluation.baseline_policy_cid,
            "verdict": verdict_value,
            "declared_thresholds_applied": bool(
                evaluation.declared_thresholds_applied
            ),
            "blocking_reasons": tuple(evaluation.blocking_reasons),
            "high_risk_assurance_reduced": bool(
                evaluation.high_risk_assurance_reduced
            ),
        }
    if isinstance(evaluation, Mapping):
        try:
            if "report_cid" in evaluation and "schema" in evaluation:
                restored = RuleEvaluationReport.from_dict(evaluation)
                return _extract_report_fields(restored)
        except (PolicyContractError, SemanticGovernorBaseError, KeyError, TypeError):
            pass
        report_cid = evaluation.get("report_cid") or evaluation.get(
            "evaluation_report_cid"
        )
        candidate_cid = evaluation.get("candidate_cid")
        benchmark_cid = evaluation.get("held_out_benchmark_cid")
        policy_cid = evaluation.get("baseline_policy_cid") or evaluation.get(
            "policy_cid"
        )
        verdict = evaluation.get("verdict")
        if report_cid is None or candidate_cid is None or verdict is None:
            raise SealingError(
                "evaluation mapping requires report_cid, candidate_cid, and verdict"
            )
        return {
            "report_cid": _cid(report_cid, "evaluation.report_cid"),
            "candidate_cid": _cid(candidate_cid, "evaluation.candidate_cid"),
            "held_out_benchmark_cid": (
                _cid(benchmark_cid, "evaluation.held_out_benchmark_cid")
                if benchmark_cid is not None
                else None
            ),
            "baseline_policy_cid": (
                _cid(policy_cid, "evaluation.baseline_policy_cid")
                if policy_cid is not None
                else None
            ),
            "verdict": _enum_value(verdict, EvaluationVerdict, "evaluation.verdict"),
            "declared_thresholds_applied": _bool(
                evaluation.get("declared_thresholds_applied", True),
                "evaluation.declared_thresholds_applied",
            ),
            "blocking_reasons": tuple(evaluation.get("blocking_reasons") or ()),
            "high_risk_assurance_reduced": _bool(
                evaluation.get("high_risk_assurance_reduced", False),
                "evaluation.high_risk_assurance_reduced",
            ),
        }
    raise SealingError(
        "evaluation must be RuleEvaluationReport or mapping"
    )


def _normalize_claims(claims: Any) -> tuple[str, ...]:
    if claims is None:
        return DEFAULT_BOUNDED_CLAIMS
    if not isinstance(claims, (list, tuple)):
        raise SealingError("claims must be a sequence of claim kinds")
    if len(claims) > MAX_CLAIMS:
        raise SealingError(f"claims exceeds {MAX_CLAIMS} entries")
    allowed = {item.value for item in BoundedClaimKind}
    out: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(claims):
        kind = _token(raw, f"claims[{index}]")
        if kind in FORBIDDEN_CLAIM_KINDS:
            raise SealingError(
                f"{REASON_OVERCLAIM}: claim {kind!r} is forbidden"
            )
        if kind not in allowed:
            raise SealingError(
                f"{REASON_UNAUTHORIZED_CLAIM}: claim {kind!r} is not in the "
                "closed bounded claim set"
            )
        if kind not in seen:
            seen.add(kind)
            out.append(kind)
    if not out:
        raise SealingError("claims must not be empty")
    return tuple(out)


# ---------------------------------------------------------------------------
# Artifact binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SealArtifactBinding:
    """One content-addressed artifact bound into a governor seal."""

    role: str
    artifact_cid: str
    policy_cid: str | None = None
    evaluation_report_cid: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        role = _token(self.role, "role")
        if role not in KNOWN_ARTIFACT_ROLES:
            raise SealingError(
                f"role {role!r} is not a known seal artifact role"
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self, "artifact_cid", _cid(self.artifact_cid, "artifact_cid")
        )
        object.__setattr__(
            self, "policy_cid", _optional_cid(self.policy_cid, "policy_cid")
        )
        object.__setattr__(
            self,
            "evaluation_report_cid",
            _optional_cid(self.evaluation_report_cid, "evaluation_report_cid"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": ARTIFACT_BINDING_SCHEMA,
            "role": self.role,
            "artifact_cid": self.artifact_cid,
            "policy_cid": self.policy_cid,
            "evaluation_report_cid": self.evaluation_report_cid,
            "notes": self.notes,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SealArtifactBinding":
        if not isinstance(data, Mapping):
            raise SealingError("SealArtifactBinding requires a mapping")
        return cls(
            role=data["role"],
            artifact_cid=data["artifact_cid"],
            policy_cid=data.get("policy_cid"),
            evaluation_report_cid=data.get("evaluation_report_cid"),
            notes=data.get("notes"),
        )


def _normalize_bindings(
    bindings: Any,
    *,
    policy_cid: str | None,
    evaluation_report_cid: str | None,
) -> tuple[SealArtifactBinding, ...]:
    if bindings is None:
        return ()
    if not isinstance(bindings, (list, tuple)):
        raise SealingError("bindings must be a sequence")
    if len(bindings) > MAX_ARTIFACT_BINDINGS:
        raise SealingError(
            f"bindings exceeds {MAX_ARTIFACT_BINDINGS} entries"
        )
    out: list[SealArtifactBinding] = []
    seen_keys: set[tuple[str, str]] = set()
    for index, item in enumerate(bindings):
        if isinstance(item, SealArtifactBinding):
            binding = item
        elif isinstance(item, Mapping):
            role = item.get("role")
            artifact_cid = item.get("artifact_cid") or item.get("cid")
            if role is None or artifact_cid is None:
                raise SealingError(
                    f"bindings[{index}] requires role and artifact_cid"
                )
            binding = SealArtifactBinding(
                role=role,
                artifact_cid=artifact_cid,
                policy_cid=item.get("policy_cid", policy_cid),
                evaluation_report_cid=item.get(
                    "evaluation_report_cid", evaluation_report_cid
                ),
                notes=item.get("notes"),
            )
        else:
            raise SealingError(
                f"bindings[{index}] must be SealArtifactBinding or mapping"
            )
        # Bindings must not drift from the evaluated policy/report identities.
        if (
            policy_cid is not None
            and binding.policy_cid is not None
            and binding.policy_cid != policy_cid
        ):
            raise SealingError(
                f"{REASON_BINDING_MISMATCH}: binding role {binding.role} "
                "policy_cid does not match evaluated policy"
            )
        if (
            evaluation_report_cid is not None
            and binding.evaluation_report_cid is not None
            and binding.evaluation_report_cid != evaluation_report_cid
        ):
            raise SealingError(
                f"{REASON_BINDING_MISMATCH}: binding role {binding.role} "
                "evaluation_report_cid does not match evaluation"
            )
        # Default-fill policy/evaluation when omitted so seals always bind.
        if binding.policy_cid is None and policy_cid is not None:
            binding = SealArtifactBinding(
                role=binding.role,
                artifact_cid=binding.artifact_cid,
                policy_cid=policy_cid,
                evaluation_report_cid=binding.evaluation_report_cid
                or evaluation_report_cid,
                notes=binding.notes,
            )
        elif (
            binding.evaluation_report_cid is None
            and evaluation_report_cid is not None
        ):
            binding = SealArtifactBinding(
                role=binding.role,
                artifact_cid=binding.artifact_cid,
                policy_cid=binding.policy_cid,
                evaluation_report_cid=evaluation_report_cid,
                notes=binding.notes,
            )
        key = (binding.role, binding.artifact_cid)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(binding)
    return tuple(out)


# ---------------------------------------------------------------------------
# Release qualification result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReleaseQualification:
    """Pre-promotion release qualification decision (emit-only; no mutation).

    ``promotion_allowed`` is true only when an authorized path is current:
    released incremental-seal evidence, or a separately authorized
    VerificationBundle-backed release qualification while the sealer remains
    typed unavailable.
    """

    qualification_id: str
    path: str
    promotion_allowed: bool
    seal_status: str
    sealer_available: bool
    sealer_capability: Mapping[str, Any]
    evaluation_report_cid: str
    candidate_cid: str
    baseline_policy_cid: str | None
    held_out_benchmark_cid: str | None
    authorization_cid: str | None
    verification_bundle_cid: str | None
    incremental_seal_cid: str | None
    blocking_reasons: tuple[str, ...]
    claims: tuple[str, ...]
    diagnostic: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "qualification_id",
            "path",
            "promotion_allowed",
            "seal_status",
            "sealer_available",
            "sealer_capability",
            "evaluation_report_cid",
            "candidate_cid",
            "baseline_policy_cid",
            "held_out_benchmark_cid",
            "authorization_cid",
            "verification_bundle_cid",
            "incremental_seal_cid",
            "blocking_reasons",
            "claims",
            "diagnostic",
            "metadata",
            "qualification_cid",
            "evidence",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "qualification_id",
            _token(self.qualification_id, "qualification_id"),
        )
        path = _enum_value(self.path, QualificationPath, "path")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self, "promotion_allowed", _bool(self.promotion_allowed, "promotion_allowed")
        )
        seal_status = _enum_value(self.seal_status, SealStatus, "seal_status")
        object.__setattr__(self, "seal_status", seal_status)
        object.__setattr__(
            self, "sealer_available", _bool(self.sealer_available, "sealer_available")
        )
        if not isinstance(self.sealer_capability, Mapping):
            raise SealingError("sealer_capability must be a mapping")
        # Force IVP non-substitution invariant on the capability snapshot.
        cap = dict(self.sealer_capability)
        cap["can_be_satisfied_by_ivp_commitment"] = False
        object.__setattr__(self, "sealer_capability", MappingProxyType(cap))
        object.__setattr__(
            self,
            "evaluation_report_cid",
            _cid(self.evaluation_report_cid, "evaluation_report_cid"),
        )
        object.__setattr__(
            self, "candidate_cid", _cid(self.candidate_cid, "candidate_cid")
        )
        object.__setattr__(
            self,
            "baseline_policy_cid",
            _optional_cid(self.baseline_policy_cid, "baseline_policy_cid"),
        )
        object.__setattr__(
            self,
            "held_out_benchmark_cid",
            _optional_cid(self.held_out_benchmark_cid, "held_out_benchmark_cid"),
        )
        object.__setattr__(
            self,
            "authorization_cid",
            _optional_cid(self.authorization_cid, "authorization_cid"),
        )
        object.__setattr__(
            self,
            "verification_bundle_cid",
            _optional_cid(self.verification_bundle_cid, "verification_bundle_cid"),
        )
        object.__setattr__(
            self,
            "incremental_seal_cid",
            _optional_cid(self.incremental_seal_cid, "incremental_seal_cid"),
        )
        reasons = tuple(
            _token(item, f"blocking_reasons[{i}]")
            for i, item in enumerate(self.blocking_reasons or ())
        )
        if len(reasons) > MAX_BLOCKING_REASONS:
            reasons = reasons[:MAX_BLOCKING_REASONS]
        object.__setattr__(self, "blocking_reasons", reasons)
        object.__setattr__(self, "claims", _normalize_claims(self.claims))
        object.__setattr__(
            self, "diagnostic", _optional_text(self.diagnostic, "diagnostic")
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

        # Fail-closed consistency between path and promotion flag.
        if self.promotion_allowed and path == QualificationPath.BLOCKED.value:
            raise SealingError(
                "promotion_allowed cannot be true when path is blocked"
            )
        if (
            self.promotion_allowed
            and path == QualificationPath.INCREMENTAL_SEAL.value
            and not self.sealer_available
        ):
            raise SealingError(
                "incremental_seal path requires sealer_available=True"
            )
        if (
            self.promotion_allowed
            and path == QualificationPath.AUTHORIZED_RELEASE_QUALIFICATION.value
        ):
            if self.authorization_cid is None:
                raise SealingError(
                    "authorized_release_qualification requires authorization_cid"
                )
            if self.verification_bundle_cid is None:
                raise SealingError(
                    "authorized_release_qualification requires "
                    "verification_bundle_cid"
                )
        if self.promotion_allowed and self.blocking_reasons:
            raise SealingError(
                "promotion_allowed cannot be true when blocking_reasons is nonempty"
            )
        # Sealer unavailable must never claim AVAILABLE seal status.
        if (
            not self.sealer_available
            and seal_status == SealStatus.AVAILABLE.value
            and path != QualificationPath.AUTHORIZED_RELEASE_QUALIFICATION.value
        ):
            raise SealingError(REASON_SEAL_STATUS_MISMATCH)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": RELEASE_QUALIFICATION_SCHEMA,
            "interface_id": RELEASE_QUALIFICATION_INTERFACE,
            "evidence": SCG_RELEASE_QUALIFICATION_EVIDENCE,
            "qualification_id": self.qualification_id,
            "path": self.path,
            "promotion_allowed": self.promotion_allowed,
            "seal_status": self.seal_status,
            "sealer_available": self.sealer_available,
            "sealer_capability": dict(self.sealer_capability),
            "evaluation_report_cid": self.evaluation_report_cid,
            "candidate_cid": self.candidate_cid,
            "baseline_policy_cid": self.baseline_policy_cid,
            "held_out_benchmark_cid": self.held_out_benchmark_cid,
            "authorization_cid": self.authorization_cid,
            "verification_bundle_cid": self.verification_bundle_cid,
            "incremental_seal_cid": self.incremental_seal_cid,
            "blocking_reasons": list(self.blocking_reasons),
            "claims": list(self.claims),
            "diagnostic": self.diagnostic,
            "metadata": dict(self.metadata),
        }

    @property
    def qualification_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["qualification_cid"] = self.qualification_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReleaseQualification":
        if not isinstance(data, Mapping):
            raise SealingError("ReleaseQualification requires a mapping")
        payload = dict(data)
        claimed = payload.pop("qualification_cid", None)
        unknown = set(payload) - cls._FIELDS
        if unknown:
            raise SealingError(
                f"ReleaseQualification has unknown fields: {sorted(unknown)}"
            )
        result = cls(
            qualification_id=payload["qualification_id"],
            path=payload["path"],
            promotion_allowed=payload["promotion_allowed"],
            seal_status=payload["seal_status"],
            sealer_available=payload["sealer_available"],
            sealer_capability=payload.get("sealer_capability") or {},
            evaluation_report_cid=payload["evaluation_report_cid"],
            candidate_cid=payload["candidate_cid"],
            baseline_policy_cid=payload.get("baseline_policy_cid"),
            held_out_benchmark_cid=payload.get("held_out_benchmark_cid"),
            authorization_cid=payload.get("authorization_cid"),
            verification_bundle_cid=payload.get("verification_bundle_cid"),
            incremental_seal_cid=payload.get("incremental_seal_cid"),
            blocking_reasons=tuple(payload.get("blocking_reasons") or ()),
            claims=tuple(payload.get("claims") or DEFAULT_BOUNDED_CLAIMS),
            diagnostic=payload.get("diagnostic"),
            metadata=payload.get("metadata") or {},
        )
        if claimed is not None and claimed != result.qualification_cid:
            raise SealingError(
                f"{REASON_STALE_SEAL}: qualification_cid does not verify"
            )
        return result

    def require_promotion_allowed(self) -> None:
        if not self.promotion_allowed:
            reasons = ", ".join(self.blocking_reasons) or REASON_PROMOTION_BLOCKED
            raise SealingError(
                f"{REASON_PROMOTION_BLOCKED}: {reasons}"
            )


# ---------------------------------------------------------------------------
# Governor seal artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GovernorSeal:
    """Content-addressed seal binding governor artifacts to evaluated policy.

    When the released sealer is present the seal may carry ``AVAILABLE`` status
    and optional sealer-produced evidence. When the sealer is absent the seal
    remains content-addressed with ``seal_status=unavailable`` and still binds
    identities — it never upgrades to a ZK or execution proof via IVP
    commitment substitution.
    """

    seal_id: str
    seal_status: str
    claims: tuple[str, ...]
    evaluation_report_cid: str
    candidate_cid: str
    baseline_policy_cid: str | None
    qualification_cid: str | None
    qualification_path: str
    sealer_available: bool
    is_zk: bool
    bindings: tuple[SealArtifactBinding, ...]
    authorization_cid: str | None = None
    verification_bundle_cid: str | None = None
    incremental_seal_cid: str | None = None
    sealer_public_module: str | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "evidence",
            "seal_id",
            "seal_status",
            "claims",
            "evaluation_report_cid",
            "candidate_cid",
            "baseline_policy_cid",
            "qualification_cid",
            "qualification_path",
            "sealer_available",
            "is_zk",
            "bindings",
            "authorization_cid",
            "verification_bundle_cid",
            "incremental_seal_cid",
            "sealer_public_module",
            "notes",
            "metadata",
            "seal_cid",
            "non_claims",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "seal_id", _token(self.seal_id, "seal_id"))
        seal_status = _enum_value(self.seal_status, SealStatus, "seal_status")
        object.__setattr__(self, "seal_status", seal_status)
        object.__setattr__(self, "claims", _normalize_claims(self.claims))
        object.__setattr__(
            self,
            "evaluation_report_cid",
            _cid(self.evaluation_report_cid, "evaluation_report_cid"),
        )
        object.__setattr__(
            self, "candidate_cid", _cid(self.candidate_cid, "candidate_cid")
        )
        object.__setattr__(
            self,
            "baseline_policy_cid",
            _optional_cid(self.baseline_policy_cid, "baseline_policy_cid"),
        )
        object.__setattr__(
            self,
            "qualification_cid",
            _optional_cid(self.qualification_cid, "qualification_cid"),
        )
        object.__setattr__(
            self,
            "qualification_path",
            _enum_value(
                self.qualification_path, QualificationPath, "qualification_path"
            ),
        )
        object.__setattr__(
            self, "sealer_available", _bool(self.sealer_available, "sealer_available")
        )
        object.__setattr__(self, "is_zk", _bool(self.is_zk, "is_zk"))
        if not isinstance(self.bindings, (list, tuple)):
            raise SealingError("bindings must be a sequence")
        normalized_bindings = tuple(
            item
            if isinstance(item, SealArtifactBinding)
            else SealArtifactBinding.from_dict(item)
            for item in self.bindings
        )
        object.__setattr__(self, "bindings", normalized_bindings)
        object.__setattr__(
            self,
            "authorization_cid",
            _optional_cid(self.authorization_cid, "authorization_cid"),
        )
        object.__setattr__(
            self,
            "verification_bundle_cid",
            _optional_cid(self.verification_bundle_cid, "verification_bundle_cid"),
        )
        object.__setattr__(
            self,
            "incremental_seal_cid",
            _optional_cid(self.incremental_seal_cid, "incremental_seal_cid"),
        )
        object.__setattr__(
            self,
            "sealer_public_module",
            _optional_text(self.sealer_public_module, "sealer_public_module"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

        # Normative: ZK only when a released sealer is available and claims ZK.
        if self.is_zk and not self.sealer_available:
            raise SealingError(
                "is_zk cannot be true when the released sealer is unavailable"
            )
        if (
            not self.sealer_available
            and seal_status == SealStatus.AVAILABLE.value
        ):
            raise SealingError(
                f"{REASON_SEAL_STATUS_MISMATCH}: unavailable sealer cannot "
                "produce AVAILABLE seal status"
            )
        # Forbidden claim kinds are rejected in _normalize_claims; re-check
        # metadata for overclaim smuggling.
        meta_claims = self.metadata.get("extra_claims")
        if meta_claims:
            raise SealingError(
                f"{REASON_OVERCLAIM}: metadata must not smuggle extra_claims"
            )

    def non_claims(self) -> tuple[str, ...]:
        """Explicit non-claims bound into every seal for claim-boundary clarity."""

        return tuple(sorted(FORBIDDEN_CLAIM_KINDS))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": GOVERNOR_SEAL_SCHEMA,
            "interface_id": GOVERNOR_SEAL_INTERFACE,
            "evidence": SCG_SEAL_BINDING_EVIDENCE,
            "seal_id": self.seal_id,
            "seal_status": self.seal_status,
            "claims": list(self.claims),
            "non_claims": list(self.non_claims()),
            "evaluation_report_cid": self.evaluation_report_cid,
            "candidate_cid": self.candidate_cid,
            "baseline_policy_cid": self.baseline_policy_cid,
            "qualification_cid": self.qualification_cid,
            "qualification_path": self.qualification_path,
            "sealer_available": self.sealer_available,
            "is_zk": self.is_zk,
            "bindings": [item.to_dict() for item in self.bindings],
            "authorization_cid": self.authorization_cid,
            "verification_bundle_cid": self.verification_bundle_cid,
            "incremental_seal_cid": self.incremental_seal_cid,
            "sealer_public_module": self.sealer_public_module,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @property
    def seal_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["seal_cid"] = self.seal_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GovernorSeal":
        if not isinstance(data, Mapping):
            raise SealingError("GovernorSeal requires a mapping")
        payload = dict(data)
        claimed = payload.pop("seal_cid", None)
        payload.pop("non_claims", None)
        unknown = set(payload) - cls._FIELDS
        if unknown:
            raise SealingError(
                f"GovernorSeal has unknown fields: {sorted(unknown)}"
            )
        bindings_raw = payload.get("bindings") or ()
        bindings = tuple(
            item
            if isinstance(item, SealArtifactBinding)
            else SealArtifactBinding.from_dict(item)
            for item in bindings_raw
        )
        result = cls(
            seal_id=payload["seal_id"],
            seal_status=payload["seal_status"],
            claims=tuple(payload.get("claims") or DEFAULT_BOUNDED_CLAIMS),
            evaluation_report_cid=payload["evaluation_report_cid"],
            candidate_cid=payload["candidate_cid"],
            baseline_policy_cid=payload.get("baseline_policy_cid"),
            qualification_cid=payload.get("qualification_cid"),
            qualification_path=payload["qualification_path"],
            sealer_available=payload["sealer_available"],
            is_zk=payload.get("is_zk", False),
            bindings=bindings,
            authorization_cid=payload.get("authorization_cid"),
            verification_bundle_cid=payload.get("verification_bundle_cid"),
            incremental_seal_cid=payload.get("incremental_seal_cid"),
            sealer_public_module=payload.get("sealer_public_module"),
            notes=payload.get("notes"),
            metadata=payload.get("metadata") or {},
        )
        if claimed is not None and claimed != result.seal_cid:
            raise SealingError(
                f"{REASON_STALE_SEAL}: seal_cid does not verify"
            )
        return result


# ---------------------------------------------------------------------------
# Qualification helpers
# ---------------------------------------------------------------------------


def _stable_qualification_id(
    evaluation_report_cid: str,
    path: str,
) -> str:
    digest = cid_for_structured(
        {
            "kind": "release_qualification_id",
            "evaluation_report_cid": evaluation_report_cid,
            "path": path,
        }
    )
    suffix = digest.replace("baguqeera", "").replace("bafkrei", "")[:16]
    return f"rq_{suffix}"


def _stable_seal_id(
    evaluation_report_cid: str,
    candidate_cid: str,
) -> str:
    digest = cid_for_structured(
        {
            "kind": "governor_seal_id",
            "evaluation_report_cid": evaluation_report_cid,
            "candidate_cid": candidate_cid,
        }
    )
    suffix = digest.replace("baguqeera", "").replace("bafkrei", "")[:16]
    return f"seal_{suffix}"


def _resolve_sealer_capability(
    sealer: Any | None,
    *,
    sealer_surface: Any | None = None,
) -> IncrementalSealerCapability:
    """Resolve sealer capability; IVP commitments always yield unavailable."""

    if sealer is not None:
        if _looks_like_ivp_commitment(sealer):
            # Typed unavailable — never raise into a substitute success path.
            return sealer_capability_from_evidence(sealer)
        if isinstance(sealer, IncrementalSealerCapability):
            return sealer_capability_from_evidence(sealer)
        return sealer_capability_from_evidence(sealer)
    if sealer_surface is not None:
        if _looks_like_ivp_commitment(sealer_surface):
            return sealer_capability_from_evidence(sealer_surface)
        return probe_incremental_sealer_capability(surface=sealer_surface)
    return probe_incremental_sealer_capability()


def _authorization_is_self(
    authorization_cid: str | None,
    *,
    evaluation_report_cid: str,
    candidate_cid: str,
    baseline_policy_cid: str | None,
    verification_bundle_cid: str | None,
    incremental_seal_cid: str | None,
) -> bool:
    if authorization_cid is None:
        return False
    forbidden = {
        evaluation_report_cid,
        candidate_cid,
    }
    if baseline_policy_cid is not None:
        forbidden.add(baseline_policy_cid)
    if verification_bundle_cid is not None:
        forbidden.add(verification_bundle_cid)
    if incremental_seal_cid is not None:
        forbidden.add(incremental_seal_cid)
    return authorization_cid in forbidden


# ---------------------------------------------------------------------------
# Public API: qualify_policy_candidate
# ---------------------------------------------------------------------------


def qualify_policy_candidate(
    evaluation: RuleEvaluationReport | Mapping[str, Any],
    *,
    sealer: Any | None = None,
    sealer_surface: Any | None = None,
    incremental_seal_evidence: Any | None = None,
    release_qualification_authorization_cid: str | None = None,
    release_qualification_bundle: Any | None = None,
    claims: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ReleaseQualification:
    """Gate promotion on release qualification (SCG-035).

    Pure function: evaluates whether a held-out evaluation report may proceed
    to authorized promotion given either:

    1. a released ``IncrementalProofSealer`` with current incremental-seal
       evidence bound to the evaluation, or
    2. a separately authorized ``VerificationBundle``-backed release
       qualification while the sealer remains typed unavailable.

    A bare ``VerificationCommitment`` never satisfies either path.
    """

    report = _extract_report_fields(evaluation)
    evaluation_report_cid = report["report_cid"]
    candidate_cid = report["candidate_cid"]
    baseline_policy_cid = report.get("baseline_policy_cid")
    held_out_benchmark_cid = report.get("held_out_benchmark_cid")
    verdict = report["verdict"]
    normalized_claims = _normalize_claims(claims)
    meta = _mapping(metadata, "metadata")

    # Reject IVP commitment as sealer evidence up front (typed unavailable).
    if incremental_seal_evidence is not None and _looks_like_ivp_commitment(
        incremental_seal_evidence
    ):
        capability = sealer_capability_from_evidence(incremental_seal_evidence)
        return ReleaseQualification(
            qualification_id=_stable_qualification_id(
                evaluation_report_cid, QualificationPath.BLOCKED.value
            ),
            path=QualificationPath.BLOCKED.value,
            promotion_allowed=False,
            seal_status=SealStatus.UNAVAILABLE.value,
            sealer_available=False,
            sealer_capability=capability.to_mapping(),
            evaluation_report_cid=evaluation_report_cid,
            candidate_cid=candidate_cid,
            baseline_policy_cid=baseline_policy_cid,
            held_out_benchmark_cid=held_out_benchmark_cid,
            authorization_cid=None,
            verification_bundle_cid=None,
            incremental_seal_cid=None,
            blocking_reasons=(
                REASON_IVP_COMMITMENT_NOT_SEALER,
                REASON_PROMOTION_BLOCKED,
            ),
            claims=normalized_claims,
            diagnostic=(
                "VerificationCommitment is structural non-ZK evidence and "
                "cannot satisfy IncrementalSealerCapability or release "
                "qualification"
            ),
            metadata=meta,
        )

    capability = _resolve_sealer_capability(
        sealer, sealer_surface=sealer_surface
    )

    blocking: list[str] = []
    if verdict != EvaluationVerdict.PASS.value:
        blocking.append(REASON_EVALUATION_NOT_PASS)

    # Path A: released incremental sealer + seal evidence.
    incremental_seal_cid: str | None = None
    if capability.available:
        if incremental_seal_evidence is None:
            blocking.append(REASON_MISSING_INCREMENTAL_SEAL_EVIDENCE)
        elif _looks_like_ivp_commitment(incremental_seal_evidence):
            blocking.append(REASON_IVP_COMMITMENT_NOT_SEALER)
        else:
            # Accept mapping/object evidence with a CID, or content-address it.
            if isinstance(incremental_seal_evidence, Mapping):
                incremental_seal_cid = _optional_cid(
                    incremental_seal_evidence.get("seal_cid")
                    or incremental_seal_evidence.get("cid"),
                    "incremental_seal_evidence.cid",
                )
                if incremental_seal_cid is None:
                    incremental_seal_cid = cid_for_structured(
                        {
                            "kind": "incremental_seal_evidence",
                            "payload": dict(incremental_seal_evidence),
                        }
                    )
            elif type(incremental_seal_evidence) is str:
                incremental_seal_cid = _cid(
                    incremental_seal_evidence, "incremental_seal_evidence"
                )
            else:
                for attr in ("seal_cid", "cid", "content_id"):
                    value = getattr(incremental_seal_evidence, attr, None)
                    if type(value) is str and value:
                        incremental_seal_cid = _cid(
                            value, f"incremental_seal_evidence.{attr}"
                        )
                        break
                if incremental_seal_cid is None:
                    incremental_seal_cid = cid_for_structured(
                        {
                            "kind": "incremental_seal_evidence",
                            "type": type(incremental_seal_evidence).__name__,
                        }
                    )
        if not blocking:
            return ReleaseQualification(
                qualification_id=_stable_qualification_id(
                    evaluation_report_cid,
                    QualificationPath.INCREMENTAL_SEAL.value,
                ),
                path=QualificationPath.INCREMENTAL_SEAL.value,
                promotion_allowed=True,
                seal_status=SealStatus.AVAILABLE.value,
                sealer_available=True,
                sealer_capability=capability.to_mapping(),
                evaluation_report_cid=evaluation_report_cid,
                candidate_cid=candidate_cid,
                baseline_policy_cid=baseline_policy_cid,
                held_out_benchmark_cid=held_out_benchmark_cid,
                authorization_cid=None,
                verification_bundle_cid=None,
                incremental_seal_cid=incremental_seal_cid,
                blocking_reasons=(),
                claims=normalized_claims,
                diagnostic=None,
                metadata=meta,
            )
        # Sealer present but path incomplete — still blocked.
        return ReleaseQualification(
            qualification_id=_stable_qualification_id(
                evaluation_report_cid, QualificationPath.BLOCKED.value
            ),
            path=QualificationPath.BLOCKED.value,
            promotion_allowed=False,
            seal_status=SealStatus.UNAVAILABLE.value
            if not capability.available
            else SealStatus.INCONCLUSIVE.value,
            sealer_available=capability.available,
            sealer_capability=capability.to_mapping(),
            evaluation_report_cid=evaluation_report_cid,
            candidate_cid=candidate_cid,
            baseline_policy_cid=baseline_policy_cid,
            held_out_benchmark_cid=held_out_benchmark_cid,
            authorization_cid=None,
            verification_bundle_cid=None,
            incremental_seal_cid=incremental_seal_cid,
            blocking_reasons=tuple(blocking) + (REASON_PROMOTION_BLOCKED,),
            claims=normalized_claims,
            diagnostic=capability.diagnostic,
            metadata=meta,
        )

    # Path B: sealer unavailable → authorized VerificationBundle release qual.
    # Capability is typed unavailable; never treat commitment as success.
    assert capability.available is False
    assert capability.can_be_satisfied_by_ivp_commitment is False

    auth_cid = _optional_cid(
        release_qualification_authorization_cid,
        "release_qualification_authorization_cid",
    )
    bundle = release_qualification_bundle
    bundle_cid: str | None = None

    if auth_cid is None:
        blocking.append(REASON_MISSING_QUALIFICATION_AUTHORIZATION)
    if bundle is None:
        blocking.append(REASON_MISSING_VERIFICATION_BUNDLE)
        blocking.append(REASON_MISSING_RELEASE_QUALIFICATION)
    elif _looks_like_ivp_commitment(bundle):
        blocking.append(REASON_BUNDLE_IS_COMMITMENT)
        blocking.append(REASON_IVP_COMMITMENT_NOT_SEALER)
    elif not _looks_like_verification_bundle(bundle):
        # Also accept a bare CID string as a bundle reference only when paired
        # with explicit authorization — still not a commitment.
        if type(bundle) is str:
            try:
                bundle_cid = _cid(bundle, "release_qualification_bundle")
            except SealingError:
                blocking.append(REASON_MISSING_VERIFICATION_BUNDLE)
        else:
            blocking.append(REASON_MISSING_VERIFICATION_BUNDLE)
    else:
        try:
            bundle_cid = _bundle_identity_cid(bundle)
        except SealingError:
            blocking.append(REASON_SCHEMA_INTEGRITY)

    if auth_cid is not None and _authorization_is_self(
        auth_cid,
        evaluation_report_cid=evaluation_report_cid,
        candidate_cid=candidate_cid,
        baseline_policy_cid=baseline_policy_cid,
        verification_bundle_cid=bundle_cid,
        incremental_seal_cid=None,
    ):
        blocking.append(REASON_SELF_AUTHORIZATION)

    # Sealer unavailability is always recorded but is not itself a hard block
    # when the authorized release-qualification path is complete.
    sealer_unavailable_reason = (
        capability.reason_code or REASON_SEALER_UNAVAILABLE
    )

    if not blocking:
        return ReleaseQualification(
            qualification_id=_stable_qualification_id(
                evaluation_report_cid,
                QualificationPath.AUTHORIZED_RELEASE_QUALIFICATION.value,
            ),
            path=QualificationPath.AUTHORIZED_RELEASE_QUALIFICATION.value,
            promotion_allowed=True,
            # Sealer remains unavailable; qualification is structural bundle
            # evidence under independent authorization — not a sealer proof.
            seal_status=SealStatus.UNAVAILABLE.value,
            sealer_available=False,
            sealer_capability=capability.to_mapping(),
            evaluation_report_cid=evaluation_report_cid,
            candidate_cid=candidate_cid,
            baseline_policy_cid=baseline_policy_cid,
            held_out_benchmark_cid=held_out_benchmark_cid,
            authorization_cid=auth_cid,
            verification_bundle_cid=bundle_cid,
            incremental_seal_cid=None,
            blocking_reasons=(),
            claims=normalized_claims,
            diagnostic=(
                f"released sealer unavailable ({sealer_unavailable_reason}); "
                "promotion gated on authorized VerificationBundle release "
                "qualification only"
            ),
            metadata=meta,
        )

    # Deduplicate while preserving order.
    deduped: list[str] = []
    for reason in blocking + [
        sealer_unavailable_reason,
        REASON_PROMOTION_BLOCKED,
    ]:
        if reason not in deduped:
            deduped.append(reason)

    return ReleaseQualification(
        qualification_id=_stable_qualification_id(
            evaluation_report_cid, QualificationPath.BLOCKED.value
        ),
        path=QualificationPath.BLOCKED.value,
        promotion_allowed=False,
        seal_status=SealStatus.UNAVAILABLE.value,
        sealer_available=False,
        sealer_capability=capability.to_mapping(),
        evaluation_report_cid=evaluation_report_cid,
        candidate_cid=candidate_cid,
        baseline_policy_cid=baseline_policy_cid,
        held_out_benchmark_cid=held_out_benchmark_cid,
        authorization_cid=auth_cid,
        verification_bundle_cid=bundle_cid,
        incremental_seal_cid=None,
        blocking_reasons=tuple(deduped),
        claims=normalized_claims,
        diagnostic=capability.diagnostic
        or "promotion blocked: release qualification incomplete",
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Public API: seal_governor_run / verify_governor_seal
# ---------------------------------------------------------------------------


def seal_governor_run(
    evaluation: RuleEvaluationReport | Mapping[str, Any],
    *,
    qualification: ReleaseQualification | Mapping[str, Any] | None = None,
    bindings: Sequence[SealArtifactBinding | Mapping[str, Any]] | None = None,
    sealer: Any | None = None,
    sealer_surface: Any | None = None,
    claims: Sequence[str] | None = None,
    require_promotion_allowed: bool = False,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GovernorSeal:
    """Bind governor-run artifacts to evaluated policy with bounded claims.

    Sealing is post-decision binding: it never upgrades evidence semantics.
    When the released sealer is missing the returned seal has
    ``seal_status=unavailable`` and remains content-addressed. Overclaims
    outside :class:`BoundedClaimKind` are rejected.
    """

    report = _extract_report_fields(evaluation)
    evaluation_report_cid = report["report_cid"]
    candidate_cid = report["candidate_cid"]
    baseline_policy_cid = report.get("baseline_policy_cid")
    held_out_benchmark_cid = report.get("held_out_benchmark_cid")

    if qualification is None:
        qual = qualify_policy_candidate(
            evaluation,
            sealer=sealer,
            sealer_surface=sealer_surface,
            claims=claims,
            metadata=metadata,
        )
    elif isinstance(qualification, ReleaseQualification):
        qual = qualification
    elif isinstance(qualification, Mapping):
        qual = ReleaseQualification.from_dict(qualification)
    else:
        raise SealingError(
            "qualification must be ReleaseQualification, mapping, or None"
        )

    # Qualification must bind the same evaluation/candidate identities.
    if qual.evaluation_report_cid != evaluation_report_cid:
        raise SealingError(
            f"{REASON_IDENTITY_MISMATCH}: qualification evaluation_report_cid "
            "does not match evaluation"
        )
    if qual.candidate_cid != candidate_cid:
        raise SealingError(
            f"{REASON_IDENTITY_MISMATCH}: qualification candidate_cid does "
            "not match evaluation"
        )
    if (
        baseline_policy_cid is not None
        and qual.baseline_policy_cid is not None
        and qual.baseline_policy_cid != baseline_policy_cid
    ):
        raise SealingError(
            f"{REASON_IDENTITY_MISMATCH}: qualification baseline_policy_cid "
            "does not match evaluation"
        )

    if require_promotion_allowed:
        qual.require_promotion_allowed()

    capability = _resolve_sealer_capability(
        sealer, sealer_surface=sealer_surface
    )
    # Prefer qualification snapshot when it already resolved capability.
    sealer_available = bool(qual.sealer_available and capability.available)
    if qual.sealer_available and not capability.available:
        # Explicit injected unavailable after a prior available qualification
        # is treated as stale.
        sealer_available = False

    is_zk = bool(sealer_available and capability.is_zk)
    if sealer_available and qual.path == QualificationPath.INCREMENTAL_SEAL.value:
        seal_status = SealStatus.AVAILABLE.value
    elif sealer_available:
        seal_status = SealStatus.AVAILABLE.value
    else:
        seal_status = SealStatus.UNAVAILABLE.value

    normalized_claims = _normalize_claims(claims or qual.claims)

    # Build default bindings from evaluation identities when caller omits them.
    caller_bindings = list(bindings or ())
    if held_out_benchmark_cid is not None:
        caller_bindings.append(
            {
                "role": ARTIFACT_ROLE_BENCHMARK,
                "artifact_cid": held_out_benchmark_cid,
            }
        )
    caller_bindings.append(
        {
            "role": ARTIFACT_ROLE_EVALUATION_REPORT,
            "artifact_cid": evaluation_report_cid,
        }
    )
    caller_bindings.append(
        {
            "role": ARTIFACT_ROLE_CANDIDATE,
            "artifact_cid": candidate_cid,
        }
    )
    if baseline_policy_cid is not None:
        caller_bindings.append(
            {
                "role": ARTIFACT_ROLE_POLICY,
                "artifact_cid": baseline_policy_cid,
            }
        )
    if qual.verification_bundle_cid is not None:
        caller_bindings.append(
            {
                "role": ARTIFACT_ROLE_VERIFICATION_BUNDLE,
                "artifact_cid": qual.verification_bundle_cid,
            }
        )
    if qual.incremental_seal_cid is not None:
        caller_bindings.append(
            {
                "role": ARTIFACT_ROLE_INCREMENTAL_SEAL,
                "artifact_cid": qual.incremental_seal_cid,
            }
        )

    normalized_bindings = _normalize_bindings(
        caller_bindings,
        policy_cid=baseline_policy_cid,
        evaluation_report_cid=evaluation_report_cid,
    )

    meta_payload = dict(_mapping(metadata, "metadata"))
    meta_payload.update(
        {
            "evidence": SCG_SEAL_BINDING_EVIDENCE,
            "promotion_allowed": qual.promotion_allowed,
            "qualification_blocking_reasons": list(qual.blocking_reasons),
            "declared_thresholds_applied": report.get(
                "declared_thresholds_applied", True
            ),
        }
    )
    seal = GovernorSeal(
        seal_id=_stable_seal_id(evaluation_report_cid, candidate_cid),
        seal_status=seal_status,
        claims=normalized_claims,
        evaluation_report_cid=evaluation_report_cid,
        candidate_cid=candidate_cid,
        baseline_policy_cid=baseline_policy_cid,
        qualification_cid=qual.qualification_cid,
        qualification_path=qual.path,
        sealer_available=sealer_available,
        is_zk=is_zk,
        bindings=normalized_bindings,
        authorization_cid=qual.authorization_cid,
        verification_bundle_cid=qual.verification_bundle_cid,
        incremental_seal_cid=qual.incremental_seal_cid,
        sealer_public_module=capability.public_module,
        notes=_optional_text(notes, "notes"),
        metadata=meta_payload,
    )
    return seal


def verify_governor_seal(
    seal: GovernorSeal | Mapping[str, Any],
    *,
    expected_evaluation_report_cid: str | None = None,
    expected_candidate_cid: str | None = None,
    expected_policy_cid: str | None = None,
    allow_unavailable: bool = True,
) -> str:
    """Verify seal identity and claim-boundary invariants; return ``seal_cid``.

    Rejects overclaims, IVP-as-sealer forgeries, identity drift, and
    unavailable seals when ``allow_unavailable`` is false.
    """

    if isinstance(seal, GovernorSeal):
        artifact = seal
    elif isinstance(seal, Mapping):
        artifact = GovernorSeal.from_dict(seal)
    else:
        raise SealingError("seal must be GovernorSeal or mapping")

    # Recompute identity.
    recomputed = artifact.seal_cid
    restored = GovernorSeal.from_dict(artifact.to_dict())
    if restored.seal_cid != recomputed:
        raise SealingError(f"{REASON_STALE_SEAL}: seal_cid does not recompute")

    # Claim boundary.
    allowed = {item.value for item in BoundedClaimKind}
    for claim in artifact.claims:
        if claim in FORBIDDEN_CLAIM_KINDS or claim not in allowed:
            raise SealingError(
                f"{REASON_OVERCLAIM}: claim {claim!r} is not permitted"
            )

    if artifact.is_zk and not artifact.sealer_available:
        raise SealingError(
            f"{REASON_OVERCLAIM}: unavailable sealer cannot claim is_zk"
        )

    if (
        not artifact.sealer_available
        and artifact.seal_status == SealStatus.AVAILABLE.value
    ):
        raise SealingError(REASON_SEAL_STATUS_MISMATCH)

    if (
        not allow_unavailable
        and artifact.seal_status == SealStatus.UNAVAILABLE.value
    ):
        raise SealingError(REASON_SEALER_UNAVAILABLE)

    if (
        expected_evaluation_report_cid is not None
        and artifact.evaluation_report_cid
        != _cid(expected_evaluation_report_cid, "expected_evaluation_report_cid")
    ):
        raise SealingError(REASON_IDENTITY_MISMATCH)

    if (
        expected_candidate_cid is not None
        and artifact.candidate_cid
        != _cid(expected_candidate_cid, "expected_candidate_cid")
    ):
        raise SealingError(REASON_IDENTITY_MISMATCH)

    if (
        expected_policy_cid is not None
        and artifact.baseline_policy_cid is not None
        and artifact.baseline_policy_cid
        != _cid(expected_policy_cid, "expected_policy_cid")
    ):
        raise SealingError(REASON_IDENTITY_MISMATCH)

    # Bindings must not drift from seal-level identities.
    for binding in artifact.bindings:
        if (
            binding.evaluation_report_cid is not None
            and binding.evaluation_report_cid != artifact.evaluation_report_cid
        ):
            raise SealingError(REASON_BINDING_MISMATCH)
        if (
            binding.policy_cid is not None
            and artifact.baseline_policy_cid is not None
            and binding.policy_cid != artifact.baseline_policy_cid
        ):
            raise SealingError(REASON_BINDING_MISMATCH)

    return recomputed


# ---------------------------------------------------------------------------
# SemanticGovernorSealAdapter
# ---------------------------------------------------------------------------


class SemanticGovernorSealAdapter:
    """Runtime adapter for release qualification and incremental seal binding.

    Capability detection is lazy and fail-closed. Missing sealer → typed
    unavailable. IVP ``VerificationCommitment`` never satisfies sealer
    capability.
    """

    __slots__ = ("_sealer_surface", "_capability")

    def __init__(
        self,
        sealer_surface: Any | None = None,
        *,
        capability: IncrementalSealerCapability | None = None,
    ) -> None:
        self._sealer_surface = sealer_surface
        self._capability = capability

    @property
    def interface_id(self) -> str:
        return SEMANTIC_GOVERNOR_SEAL_ADAPTER_INTERFACE

    @property
    def capability(self) -> IncrementalSealerCapability:
        if self._capability is None:
            self._capability = probe_incremental_sealer_capability(
                surface=self._sealer_surface
            )
        # Re-assert IVP non-substitution on every access.
        assert self._capability.can_be_satisfied_by_ivp_commitment is False
        return self._capability

    def probe(self) -> IncrementalSealerCapability:
        """Re-probe sealer capability (never treats IVP commitment as success)."""

        if self._sealer_surface is not None and _looks_like_ivp_commitment(
            self._sealer_surface
        ):
            self._capability = sealer_capability_from_evidence(
                self._sealer_surface
            )
            return self._capability
        self._capability = probe_incremental_sealer_capability(
            surface=self._sealer_surface
        )
        return self._capability

    def sealer_is_available(self) -> bool:
        return bool(self.capability.available)

    def sealer_status(self) -> str:
        return self.capability.seal_status

    def reject_commitment_as_sealer(self, evidence: Any) -> None:
        """Fail closed when IVP commitment is offered as sealer capability."""

        reject_ivp_commitment_as_sealer(evidence)

    def require_sealer(self, operation: str = "seal") -> IncrementalSealerCapability:
        cap = self.capability
        cap.require_available(operation)
        return cap

    def qualify_policy_candidate(
        self,
        evaluation: RuleEvaluationReport | Mapping[str, Any],
        *,
        incremental_seal_evidence: Any | None = None,
        release_qualification_authorization_cid: str | None = None,
        release_qualification_bundle: Any | None = None,
        claims: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ReleaseQualification:
        return qualify_policy_candidate(
            evaluation,
            sealer=self.capability,
            sealer_surface=self._sealer_surface,
            incremental_seal_evidence=incremental_seal_evidence,
            release_qualification_authorization_cid=(
                release_qualification_authorization_cid
            ),
            release_qualification_bundle=release_qualification_bundle,
            claims=claims,
            metadata=metadata,
        )

    def seal_governor_run(
        self,
        evaluation: RuleEvaluationReport | Mapping[str, Any],
        *,
        qualification: ReleaseQualification | Mapping[str, Any] | None = None,
        bindings: Sequence[SealArtifactBinding | Mapping[str, Any]] | None = None,
        claims: Sequence[str] | None = None,
        require_promotion_allowed: bool = False,
        notes: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GovernorSeal:
        return seal_governor_run(
            evaluation,
            qualification=qualification,
            bindings=bindings,
            sealer=self.capability,
            sealer_surface=self._sealer_surface,
            claims=claims,
            require_promotion_allowed=require_promotion_allowed,
            notes=notes,
            metadata=metadata,
        )

    def verify_governor_seal(
        self,
        seal: GovernorSeal | Mapping[str, Any],
        *,
        expected_evaluation_report_cid: str | None = None,
        expected_candidate_cid: str | None = None,
        expected_policy_cid: str | None = None,
        allow_unavailable: bool = True,
    ) -> str:
        return verify_governor_seal(
            seal,
            expected_evaluation_report_cid=expected_evaluation_report_cid,
            expected_candidate_cid=expected_candidate_cid,
            expected_policy_cid=expected_policy_cid,
            allow_unavailable=allow_unavailable,
        )

    def runtime_view(self) -> Mapping[str, Any]:
        cap = self.capability
        return MappingProxyType(
            {
                "interface_id": SEMANTIC_GOVERNOR_SEAL_ADAPTER_INTERFACE,
                "evidence": SCG_SEAL_BINDING_EVIDENCE,
                "sealer": cap.to_mapping(),
                "can_be_satisfied_by_ivp_commitment": False,
                "bounded_claims": list(DEFAULT_BOUNDED_CLAIMS),
                "forbidden_claims": sorted(FORBIDDEN_CLAIM_KINDS),
                "qualification_paths": [item.value for item in QualificationPath],
            }
        )


def load_seal_adapter(
    sealer_surface: Any | None = None,
    *,
    require_sealer: bool = False,
) -> SemanticGovernorSealAdapter:
    """Load the seal adapter; sealer remains optional unless required."""

    adapter = SemanticGovernorSealAdapter(sealer_surface=sealer_surface)
    if require_sealer:
        adapter.require_sealer("load")
    return adapter


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


__all__ = [
    "ARTIFACT_BINDING_SCHEMA",
    "ARTIFACT_ROLE_BENCHMARK",
    "ARTIFACT_ROLE_CALIBRATION_PROFILE",
    "ARTIFACT_ROLE_CANDIDATE",
    "ARTIFACT_ROLE_CONTEXT_PACK",
    "ARTIFACT_ROLE_DIFFERENTIAL_REPORT",
    "ARTIFACT_ROLE_EVALUATION_REPORT",
    "ARTIFACT_ROLE_INCREMENTAL_SEAL",
    "ARTIFACT_ROLE_POLICY",
    "ARTIFACT_ROLE_PROMOTION_DECISION",
    "ARTIFACT_ROLE_VERIFICATION_BUNDLE",
    "BOUNDED_CLAIM_SET_INTERFACE",
    "BOUNDED_CLAIM_SET_SCHEMA",
    "BoundedClaimKind",
    "DEFAULT_BOUNDED_CLAIMS",
    "FORBIDDEN_CLAIM_KINDS",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "GOVERNOR_SEAL_INTERFACE",
    "GOVERNOR_SEAL_SCHEMA",
    "GovernorSeal",
    "KNOWN_ARTIFACT_ROLES",
    "QUALIFY_POLICY_CANDIDATE_INTERFACE",
    "QualificationPath",
    "REASON_BINDING_MISMATCH",
    "REASON_BUNDLE_IS_COMMITMENT",
    "REASON_EVALUATION_NOT_PASS",
    "REASON_IDENTITY_MISMATCH",
    "REASON_IVP_COMMITMENT_NOT_SEALER",
    "REASON_MISSING_EVALUATION",
    "REASON_MISSING_INCREMENTAL_SEAL_EVIDENCE",
    "REASON_MISSING_QUALIFICATION_AUTHORIZATION",
    "REASON_MISSING_RELEASE_QUALIFICATION",
    "REASON_MISSING_VERIFICATION_BUNDLE",
    "REASON_OVERCLAIM",
    "REASON_PROMOTION_BLOCKED",
    "REASON_SCHEMA_INTEGRITY",
    "REASON_SEALER_UNAVAILABLE",
    "REASON_SEAL_STATUS_MISMATCH",
    "REASON_SELF_AUTHORIZATION",
    "REASON_STALE_SEAL",
    "REASON_UNAUTHORIZED_CLAIM",
    "RELEASE_QUALIFICATION_INTERFACE",
    "RELEASE_QUALIFICATION_SCHEMA",
    "ReleaseQualification",
    "SCG_RELEASE_QUALIFICATION_EVIDENCE",
    "SCG_SEAL_BINDING_EVIDENCE",
    "SEAL_GOVERNOR_RUN_INTERFACE",
    "SEMANTIC_GOVERNOR_SEAL_ADAPTER_INTERFACE",
    "SealArtifactBinding",
    "SealingError",
    "SemanticGovernorSealAdapter",
    "VERIFY_GOVERNOR_SEAL_INTERFACE",
    "load_seal_adapter",
    "qualify_policy_candidate",
    "seal_governor_run",
    "verify_governor_seal",
]
