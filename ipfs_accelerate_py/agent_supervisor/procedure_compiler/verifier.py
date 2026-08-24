"""Independent procedure verification obligations (PCPC-017).

``ProcedureVerifier`` checks structural, authority, effect, dataflow,
temporal, semantic, and validation obligations against independently
admitted evidence.  It does not sign, promote, or execute a candidate.
A model, the candidate, or the procedure itself cannot discharge any
layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .contracts import (
    MAX_ITEMS,
    ArtifactBindings,
    ArtifactState,
    ProcedureCandidate,
    ProcedureContractError,
    ProcedureSpec,
    RiskClass,
    StepOperation,
    TaskFamily,
    _enum,
    _identifier,
    _nested,
    _nonnegative_int,
    _positive_int,
    _strings,
    _text,
)
from .contracts import (
    ProcedureVerificationResult as ProcedureVerificationResultArtifact,
)
from .procedure_ir import (
    ProcedureDataflowError,
    ProcedureEffectError,
    ProcedureGraphError,
    ProcedureIRValidationError,
    ProcedureScopeError,
    ProcedureValidationRetentionError,
    validate_procedure_spec,
)

VERIFIER_REVISION: Final[str] = "ProcedureVerifier@1"
VERIFICATION_POLICY_SCHEMA: Final[str] = "procedure-compiler/verification-policy@1"
INDEPENDENT_EVIDENCE_SCHEMA: Final[str] = "procedure-compiler/independent-evidence@1"

REQUIRED_VERIFICATION_LAYERS: Final[tuple[str, ...]] = (
    "structural",
    "authority",
    "effect",
    "dataflow",
    "temporal",
    "semantic",
    "validation",
)
REQUIRED_EVIDENCE_KINDS: Final[tuple[str, ...]] = (
    "proof",
    "test",
    "adversarial",
    "held_out",
    "shadow",
    "specification",
    "source_episode",
    "counterexample_set",
)
FORBIDDEN_SELF_PRODUCERS: Final[frozenset[str]] = frozenset(
    {
        "self",
        "candidate",
        "procedure",
        "model",
        "llm",
        "self-issued",
        "self-certified",
    }
)
_RISK_ORDER: Final[dict[RiskClass, int]] = {
    RiskClass.OBSERVATION_ONLY: 0,
    RiskClass.REVERSIBLE_LOCAL: 1,
    RiskClass.REPOSITORY_WRITE: 2,
    RiskClass.PUBLIC_CONTRACT: 3,
    RiskClass.AUTHORITY_OR_SECURITY: 4,
}


class ProcedureVerificationError(ProcedureContractError):
    """A verification request, policy, or evidence bundle is malformed."""


class VerificationLayer(str, Enum):
    STRUCTURAL = "structural"
    AUTHORITY = "authority"
    EFFECT = "effect"
    DATAFLOW = "dataflow"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    VALIDATION = "validation"


class VerificationStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class VerificationReasonCode(str, Enum):
    ACCEPTED = "accepted"
    MALFORMED_REQUEST = "malformed-request"
    CANDIDATE_REJECTED = "candidate-rejected"
    SELF_CERTIFICATION = "self-certification"
    MISSING_INDEPENDENT_EVIDENCE = "missing-independent-evidence"
    STALE_BINDINGS = "stale-bindings"
    STALE_POLICY = "stale-policy"
    STRUCTURAL_UNSAFE = "structural-unsafe"
    AUTHORITY_UNSAFE = "authority-unsafe"
    EFFECT_UNSAFE = "effect-unsafe"
    DATAFLOW_UNSAFE = "dataflow-unsafe"
    TEMPORAL_UNSAFE = "temporal-unsafe"
    SEMANTIC_UNSAFE = "semantic-unsafe"
    VALIDATION_WEAKENED = "validation-weakened"
    VALIDATION_INCOMPLETE = "validation-incomplete"
    LAYER_INCOMPLETE = "layer-incomplete"


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ProcedureVerificationError("{} must be a boolean".format(field_name))
    return value


def _risk(value: Any, field_name: str) -> RiskClass:
    return _enum(value, RiskClass, field_name)


def _unique(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ProcedureVerificationError("{} must be a sequence".format(field_name))
    raw = tuple(values)
    items = _strings(raw, field_name, identifiers=True, required=True)
    if len(raw) != len(items) or len(items) != len(set(items)):
        raise ProcedureVerificationError("{} contains duplicate identities".format(field_name))
    return items


def _optional_receipts(
    values: Any, field_name: str
) -> tuple["AdmittedReceipt", ...]:
    if values is None:
        return ()
    if isinstance(values, AdmittedReceipt):
        values = (values,)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ProcedureVerificationError("{} must be a sequence".format(field_name))
    if len(values) > MAX_ITEMS:
        raise ProcedureVerificationError("{} exceeds its item bound".format(field_name))
    items: list[AdmittedReceipt] = []
    seen: set[str] = set()
    for item in values:
        receipt = item if isinstance(item, AdmittedReceipt) else AdmittedReceipt.from_mapping(item)
        if receipt.receipt_cid in seen:
            raise ProcedureVerificationError("{} contains duplicate receipts".format(field_name))
        seen.add(receipt.receipt_cid)
        items.append(receipt)
    return tuple(items)


@dataclass(frozen=True)
class AdmittedReceipt:
    """One independently admitted proof, test, or evaluation receipt."""

    receipt_cid: str
    kind: str
    producer_id: str
    bindings: ArtifactBindings
    observed_at_ms: int
    expires_at_ms: int
    contract_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_cid", _identifier(self.receipt_cid, "receipt_cid"))
        object.__setattr__(self, "kind", _identifier(self.kind, "kind"))
        if self.kind not in REQUIRED_EVIDENCE_KINDS:
            raise ProcedureVerificationError("receipt kind is not a required evidence kind")
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))
        if self.producer_id.lower() in FORBIDDEN_SELF_PRODUCERS:
            raise ProcedureVerificationError("receipt producer is not independent")
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        object.__setattr__(
            self, "observed_at_ms", _nonnegative_int(self.observed_at_ms, "observed_at_ms")
        )
        object.__setattr__(
            self, "expires_at_ms", _positive_int(self.expires_at_ms, "expires_at_ms")
        )
        object.__setattr__(
            self, "contract_id", _identifier(self.contract_id, "contract_id", required=False)
        )
        if self.expires_at_ms <= self.observed_at_ms:
            raise ProcedureVerificationError("receipt expiry must follow observation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_cid": self.receipt_cid,
            "kind": self.kind,
            "producer_id": self.producer_id,
            "bindings": self.bindings.to_dict(),
            "observed_at_ms": self.observed_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "contract_id": self.contract_id,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AdmittedReceipt:
        if not isinstance(payload, Mapping):
            raise ProcedureVerificationError("receipt must be a mapping")
        return cls(
            receipt_cid=payload.get("receipt_cid", ""),
            kind=payload.get("kind", ""),
            producer_id=payload.get("producer_id", ""),
            bindings=payload.get("bindings"),  # type: ignore[arg-type]
            observed_at_ms=payload.get("observed_at_ms", 0),
            expires_at_ms=payload.get("expires_at_ms", 0),
            contract_id=payload.get("contract_id", ""),
        )


@dataclass(frozen=True)
class VerificationPolicy:
    """Current verification policy the candidate must meet or exceed."""

    revision: str
    bindings: ArtifactBindings
    operation_catalog_revision: str
    effect_policy_revision: str
    authority_policy_revision: str
    required_test_contracts: tuple[str, ...]
    required_proof_contracts: tuple[str, ...]
    require_adversarial: bool = True
    require_held_out: bool = True
    require_shadow: bool = True
    confirmation_required: bool = False
    max_risk_ceiling: RiskClass = RiskClass.REPOSITORY_WRITE
    review_horizon_ms: int = 86_400_000
    required_layers: tuple[str, ...] = REQUIRED_VERIFICATION_LAYERS
    required_evidence_kinds: tuple[str, ...] = REQUIRED_EVIDENCE_KINDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "revision", _identifier(self.revision, "revision"))
        object.__setattr__(self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings"))
        for name in (
            "operation_catalog_revision",
            "effect_policy_revision",
            "authority_policy_revision",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "required_test_contracts",
            _strings(self.required_test_contracts, "required_test_contracts", identifiers=True),
        )
        object.__setattr__(
            self,
            "required_proof_contracts",
            _strings(self.required_proof_contracts, "required_proof_contracts", identifiers=True),
        )
        for name in (
            "require_adversarial",
            "require_held_out",
            "require_shadow",
            "confirmation_required",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(
            self, "max_risk_ceiling", _risk(self.max_risk_ceiling, "max_risk_ceiling")
        )
        object.__setattr__(
            self,
            "review_horizon_ms",
            _positive_int(self.review_horizon_ms, "review_horizon_ms", maximum=31_536_000_000),
        )
        layers = _strings(self.required_layers, "required_layers", identifiers=True, required=True)
        unknown = [item for item in layers if item not in set(REQUIRED_VERIFICATION_LAYERS)]
        if unknown:
            raise ProcedureVerificationError("verification policy names an unknown layer")
        if tuple(REQUIRED_VERIFICATION_LAYERS) != layers and set(layers) != set(
            REQUIRED_VERIFICATION_LAYERS
        ):
            missing = [item for item in REQUIRED_VERIFICATION_LAYERS if item not in set(layers)]
            if missing:
                raise ProcedureVerificationError("verification policy omits a required layer")
        object.__setattr__(self, "required_layers", tuple(REQUIRED_VERIFICATION_LAYERS))
        kinds = _strings(
            self.required_evidence_kinds,
            "required_evidence_kinds",
            identifiers=True,
            required=True,
        )
        if any(item not in set(REQUIRED_EVIDENCE_KINDS) for item in kinds):
            raise ProcedureVerificationError("verification policy names an unknown evidence kind")
        missing_kinds = [item for item in REQUIRED_EVIDENCE_KINDS if item not in set(kinds)]
        if missing_kinds:
            raise ProcedureVerificationError("verification policy omits a required evidence kind")
        object.__setattr__(self, "required_evidence_kinds", tuple(REQUIRED_EVIDENCE_KINDS))
        if self.authority_policy_revision != self.bindings.policy_revision:
            raise ProcedureVerificationError("authority policy is not exact-binding current")

    @property
    def required_validation_contracts(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys((*self.required_test_contracts, *self.required_proof_contracts))
        )


@dataclass(frozen=True)
class IndependentEvidence:
    """Evidence that is not the candidate and is not procedure-asserted."""

    producer_id: str
    task_family: TaskFamily
    source_episode_cids: tuple[str, ...]
    specification_cids: tuple[str, ...]
    counterexample_set_cid: str
    proof_receipt_cids: tuple[str, ...]
    test_receipt_cids: tuple[str, ...]
    adversarial_assurance_cids: tuple[str, ...]
    held_out_evaluation_cid: str
    shadow_evaluation_cid: str
    repository_families: tuple[str, ...]
    supported_language_classes: tuple[str, ...]
    supported_framework_classes: tuple[str, ...]
    known_limitations: tuple[str, ...] = ()
    observed_at_ms: int = 0
    receipts: tuple[AdmittedReceipt, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))
        object.__setattr__(
            self, "task_family", _nested(self.task_family, TaskFamily, "task_family")
        )
        for name in (
            "source_episode_cids",
            "specification_cids",
            "proof_receipt_cids",
            "test_receipt_cids",
            "adversarial_assurance_cids",
            "repository_families",
            "supported_language_classes",
            "supported_framework_classes",
        ):
            object.__setattr__(self, name, _unique(getattr(self, name), name))
        object.__setattr__(
            self,
            "counterexample_set_cid",
            _identifier(self.counterexample_set_cid, "counterexample_set_cid"),
        )
        object.__setattr__(
            self,
            "held_out_evaluation_cid",
            _identifier(self.held_out_evaluation_cid, "held_out_evaluation_cid"),
        )
        object.__setattr__(
            self,
            "shadow_evaluation_cid",
            _identifier(self.shadow_evaluation_cid, "shadow_evaluation_cid"),
        )
        object.__setattr__(
            self,
            "known_limitations",
            _strings(self.known_limitations, "known_limitations", limit=64),
        )
        object.__setattr__(
            self, "observed_at_ms", _nonnegative_int(self.observed_at_ms, "observed_at_ms")
        )
        object.__setattr__(self, "receipts", _optional_receipts(self.receipts, "receipts"))
        if self.producer_id.lower() in FORBIDDEN_SELF_PRODUCERS:
            raise ProcedureVerificationError("evidence producer is not independent")
        for receipt in self.receipts:
            if receipt.producer_id.lower() in FORBIDDEN_SELF_PRODUCERS:
                raise ProcedureVerificationError("receipt producer is not independent")
            if receipt.kind not in REQUIRED_EVIDENCE_KINDS:
                raise ProcedureVerificationError("receipt kind is not a required evidence kind")

    @property
    def evidence_cids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.source_episode_cids,
                    *self.specification_cids,
                    self.counterexample_set_cid,
                    *self.proof_receipt_cids,
                    *self.test_receipt_cids,
                    *self.adversarial_assurance_cids,
                    self.held_out_evaluation_cid,
                    self.shadow_evaluation_cid,
                )
            )
        )

    def cids_for_kind(self, kind: str) -> tuple[str, ...]:
        mapping = {
            "proof": self.proof_receipt_cids,
            "test": self.test_receipt_cids,
            "adversarial": self.adversarial_assurance_cids,
            "held_out": (self.held_out_evaluation_cid,),
            "shadow": (self.shadow_evaluation_cid,),
            "specification": self.specification_cids,
            "source_episode": self.source_episode_cids,
            "counterexample_set": (self.counterexample_set_cid,),
        }
        return mapping[kind]


@dataclass(frozen=True)
class LayerOutcome:
    layer: VerificationLayer
    accepted: bool
    reason_code: str
    message: str
    evidence_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", _enum(self.layer, VerificationLayer, "layer"))
        object.__setattr__(self, "accepted", _bool(self.accepted, "accepted"))
        object.__setattr__(self, "reason_code", _identifier(self.reason_code, "reason_code"))
        object.__setattr__(self, "message", _text(self.message, "message", required=False))
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(self.evidence_cids, "evidence_cids", identifiers=True),
        )

    def to_facts(self) -> dict[str, Any]:
        return {
            "layer": self.layer.value,
            "accepted": self.accepted,
            "reason_code": self.reason_code,
            "message": self.message,
            "evidence_cids": self.evidence_cids,
        }


@dataclass(frozen=True)
class ProcedureVerification:
    """Independent verification result.  Never a certificate or promotion."""

    status: VerificationStatus
    reason_code: VerificationReasonCode
    candidate_cid: str
    procedure_cid: str
    producer_id: str
    policy_revision: str
    layers: tuple[LayerOutcome, ...]
    artifact: ProcedureVerificationResultArtifact
    evidence_cids: tuple[str, ...] = ()
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _enum(self.status, VerificationStatus, "status"))
        object.__setattr__(
            self, "reason_code", _enum(self.reason_code, VerificationReasonCode, "reason_code")
        )
        for name in ("candidate_cid", "procedure_cid", "producer_id", "policy_revision"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if not isinstance(self.layers, tuple) or len(self.layers) != len(
            REQUIRED_VERIFICATION_LAYERS
        ):
            raise ProcedureVerificationError("verification must report every required layer")
        reported = tuple(item.layer.value for item in self.layers)
        if reported != REQUIRED_VERIFICATION_LAYERS:
            raise ProcedureVerificationError("verification layers are incomplete or reordered")
        if not isinstance(self.artifact, ProcedureVerificationResultArtifact):
            raise ProcedureVerificationError("verification artifact is untyped")
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(self.evidence_cids, "evidence_cids", identifiers=True),
        )
        object.__setattr__(self, "message", _text(self.message, "message", required=False))
        accepted_layers = all(item.accepted for item in self.layers)
        if self.accepted and not accepted_layers:
            raise ProcedureVerificationError("accepted verification omitted a failed layer")
        if self.accepted and self.artifact.state is not ArtifactState.VERIFIED:
            raise ProcedureVerificationError("accepted verification must mint a verified artifact")
        if not self.accepted and self.artifact.state is ArtifactState.VERIFIED:
            raise ProcedureVerificationError("rejected verification cannot claim verified state")

    @property
    def accepted(self) -> bool:
        return self.status is VerificationStatus.ACCEPTED

    def outcome(self, layer: VerificationLayer | str) -> LayerOutcome:
        name = layer.value if isinstance(layer, VerificationLayer) else layer
        for item in self.layers:
            if item.layer.value == name:
                return item
        raise ProcedureVerificationError("unknown verification layer")


def _self_identities(candidate: ProcedureCandidate) -> frozenset[str]:
    """Candidate and procedure identities that cannot discharge verification."""

    procedure = candidate.procedure
    identities = {
        candidate.content_id,
        procedure.content_id,
        procedure.name,
        candidate.synthesis_plan_cid,
    }
    identities.update(procedure.provenance_cids)
    return frozenset(item.lower() for item in identities if item)


def _bound_input_identities(candidate: ProcedureCandidate) -> frozenset[str]:
    """Source-episode and counterexample identities evidence must repeat."""

    identities = set(candidate.source_episode_cids)
    if candidate.counterexample_set_cid:
        identities.add(candidate.counterexample_set_cid)
    return frozenset(item.lower() for item in identities if item)


def _is_self_producer(producer_id: str, candidate: ProcedureCandidate) -> bool:
    normalized = producer_id.lower()
    return (
        normalized in FORBIDDEN_SELF_PRODUCERS
        or normalized in _self_identities(candidate)
        or normalized in _bound_input_identities(candidate)
    )


def _layer_from_ir_error(exc: Exception) -> tuple[VerificationLayer, VerificationReasonCode]:
    if isinstance(exc, ProcedureDataflowError):
        return VerificationLayer.DATAFLOW, VerificationReasonCode.DATAFLOW_UNSAFE
    if isinstance(exc, ProcedureScopeError):
        return VerificationLayer.EFFECT, VerificationReasonCode.EFFECT_UNSAFE
    if isinstance(exc, ProcedureEffectError):
        message = str(exc).lower()
        if "authorit" in message or "stale" in message or "policy" in message:
            return VerificationLayer.AUTHORITY, VerificationReasonCode.AUTHORITY_UNSAFE
        return VerificationLayer.EFFECT, VerificationReasonCode.EFFECT_UNSAFE
    if isinstance(exc, ProcedureValidationRetentionError):
        return VerificationLayer.VALIDATION, VerificationReasonCode.VALIDATION_WEAKENED
    if isinstance(exc, (ProcedureGraphError, ProcedureIRValidationError)):
        return VerificationLayer.STRUCTURAL, VerificationReasonCode.STRUCTURAL_UNSAFE
    return VerificationLayer.STRUCTURAL, VerificationReasonCode.STRUCTURAL_UNSAFE


def _pass(layer: VerificationLayer, *, evidence_cids: Sequence[str] = ()) -> LayerOutcome:
    return LayerOutcome(
        layer=layer,
        accepted=True,
        reason_code=VerificationReasonCode.ACCEPTED.value,
        message="",
        evidence_cids=tuple(evidence_cids),
    )


def _fail(
    layer: VerificationLayer,
    reason: VerificationReasonCode,
    message: str,
    *,
    evidence_cids: Sequence[str] = (),
) -> LayerOutcome:
    return LayerOutcome(
        layer=layer,
        accepted=False,
        reason_code=reason.value,
        message=message,
        evidence_cids=tuple(evidence_cids),
    )


def _family_matches(procedure: ProcedureSpec, family: TaskFamily) -> bool:
    return procedure.task_family_id in {family.name, family.content_id}


class ProcedureVerifier:
    """Fail-closed independent verifier for synthesized procedure candidates."""

    revision: Final[str] = VERIFIER_REVISION

    def verify(
        self,
        candidate: ProcedureCandidate,
        evidence: IndependentEvidence,
        policy: VerificationPolicy,
        *,
        now_ms: int = 0,
    ) -> ProcedureVerification:
        if not isinstance(candidate, ProcedureCandidate):
            raise ProcedureVerificationError("candidate must be a ProcedureCandidate")
        if not isinstance(evidence, IndependentEvidence):
            raise ProcedureVerificationError("evidence must be IndependentEvidence")
        if not isinstance(policy, VerificationPolicy):
            raise ProcedureVerificationError("policy must be a VerificationPolicy")
        now = _nonnegative_int(now_ms, "now_ms")
        procedure = candidate.procedure
        layers = self._check_layers(candidate, evidence, policy, now)
        failed = tuple(item for item in layers if not item.accepted)
        if failed:
            status = VerificationStatus.REJECTED
            reason = VerificationReasonCode(failed[0].reason_code)
            message = failed[0].message
            state = ArtifactState.REJECTED
        else:
            status = VerificationStatus.ACCEPTED
            reason = VerificationReasonCode.ACCEPTED
            message = ""
            state = ArtifactState.VERIFIED
        artifact = ProcedureVerificationResultArtifact(
            bindings=candidate.bindings,
            state=state,
            subject_cid=candidate.content_id,
            reference_cids=evidence.evidence_cids,
            labels=tuple(item.layer.value for item in layers if item.accepted),
            facts={
                "status": status.value,
                "reason_code": reason.value,
                "verifier_revision": VERIFIER_REVISION,
                "policy_revision": policy.revision,
                "producer_id": evidence.producer_id,
                "procedure_cid": procedure.content_id,
                "layers": {item.layer.value: item.accepted for item in layers},
                "layer_reason_codes": {item.layer.value: item.reason_code for item in layers},
            },
            created_at_ms=now,
        )
        return ProcedureVerification(
            status=status,
            reason_code=reason,
            candidate_cid=candidate.content_id,
            procedure_cid=procedure.content_id,
            producer_id=evidence.producer_id,
            policy_revision=policy.revision,
            layers=layers,
            artifact=artifact,
            evidence_cids=evidence.evidence_cids,
            message=message,
        )

    def _check_layers(
        self,
        candidate: ProcedureCandidate,
        evidence: IndependentEvidence,
        policy: VerificationPolicy,
        now_ms: int,
    ) -> tuple[LayerOutcome, ...]:
        outcomes: dict[VerificationLayer, LayerOutcome] = {}
        if candidate.state is ArtifactState.REJECTED:
            rejection = _fail(
                VerificationLayer.STRUCTURAL,
                VerificationReasonCode.CANDIDATE_REJECTED,
                "rejected candidates cannot be verified",
            )
            return tuple(
                rejection
                if layer is VerificationLayer.STRUCTURAL
                else _fail(
                    layer,
                    VerificationReasonCode.CANDIDATE_REJECTED,
                    "rejected candidates cannot be verified",
                )
                for layer in VerificationLayer
            )
        if _is_self_producer(evidence.producer_id, candidate):
            rejection = _fail(
                VerificationLayer.SEMANTIC,
                VerificationReasonCode.SELF_CERTIFICATION,
                "evidence producer is the candidate or procedure",
            )
            return tuple(
                rejection
                if layer is VerificationLayer.SEMANTIC
                else _fail(
                    layer,
                    VerificationReasonCode.SELF_CERTIFICATION,
                    "evidence producer is the candidate or procedure",
                )
                for layer in VerificationLayer
            )

        ir_error: Exception | None = None
        try:
            validate_procedure_spec(candidate.procedure)
        except ProcedureContractError as exc:
            ir_error = exc

        outcomes[VerificationLayer.STRUCTURAL] = self._structural(candidate, ir_error)
        outcomes[VerificationLayer.AUTHORITY] = self._authority(
            candidate, evidence, policy, ir_error
        )
        outcomes[VerificationLayer.EFFECT] = self._effect(candidate, evidence, policy, ir_error)
        outcomes[VerificationLayer.DATAFLOW] = self._dataflow(ir_error)
        outcomes[VerificationLayer.TEMPORAL] = self._temporal(
            candidate, evidence, policy, now_ms, ir_error
        )
        outcomes[VerificationLayer.SEMANTIC] = self._semantic(candidate, evidence, policy)
        outcomes[VerificationLayer.VALIDATION] = self._validation(
            candidate, evidence, policy, now_ms, ir_error
        )
        return tuple(outcomes[VerificationLayer(name)] for name in REQUIRED_VERIFICATION_LAYERS)

    def _structural(
        self, candidate: ProcedureCandidate, ir_error: Exception | None
    ) -> LayerOutcome:
        if ir_error is not None:
            layer, reason = _layer_from_ir_error(ir_error)
            if layer is VerificationLayer.STRUCTURAL:
                return _fail(VerificationLayer.STRUCTURAL, reason, str(ir_error))
        procedure = candidate.procedure
        if procedure.state in {
            ArtifactState.VERIFIED,
            ArtifactState.PROMOTED,
            ArtifactState.REVOKED,
            ArtifactState.STALE,
        }:
            return _fail(
                VerificationLayer.STRUCTURAL,
                VerificationReasonCode.STRUCTURAL_UNSAFE,
                "procedure asserted a non-candidate lifecycle state",
            )
        if not procedure.steps or not procedure.terminal_step_ids:
            return _fail(
                VerificationLayer.STRUCTURAL,
                VerificationReasonCode.STRUCTURAL_UNSAFE,
                "procedure is structurally incomplete",
            )
        return _pass(VerificationLayer.STRUCTURAL)

    def _authority(
        self,
        candidate: ProcedureCandidate,
        evidence: IndependentEvidence,
        policy: VerificationPolicy,
        ir_error: Exception | None,
    ) -> LayerOutcome:
        if ir_error is not None:
            layer, reason = _layer_from_ir_error(ir_error)
            if layer is VerificationLayer.AUTHORITY:
                return _fail(VerificationLayer.AUTHORITY, reason, str(ir_error))
        procedure = candidate.procedure
        envelope = procedure.authority
        if envelope.authority_policy_revision != policy.authority_policy_revision:
            return _fail(
                VerificationLayer.AUTHORITY,
                VerificationReasonCode.STALE_POLICY,
                "authority policy revision is not current",
            )
        if envelope.authority_policy_revision != procedure.bindings.policy_revision:
            return _fail(
                VerificationLayer.AUTHORITY,
                VerificationReasonCode.STALE_POLICY,
                "authority envelope is stale for exact bindings",
            )
        if procedure.bindings != policy.bindings or candidate.bindings != policy.bindings:
            return _fail(
                VerificationLayer.AUTHORITY,
                VerificationReasonCode.STALE_BINDINGS,
                "candidate bindings are not current",
            )
        if _RISK_ORDER[envelope.risk_ceiling] > _RISK_ORDER[policy.max_risk_ceiling]:
            return _fail(
                VerificationLayer.AUTHORITY,
                VerificationReasonCode.AUTHORITY_UNSAFE,
                "procedure risk exceeds the current verification policy ceiling",
            )
        if _RISK_ORDER[envelope.risk_ceiling] > _RISK_ORDER[
            evidence.task_family.boundary.risk_ceiling
        ]:
            return _fail(
                VerificationLayer.AUTHORITY,
                VerificationReasonCode.AUTHORITY_UNSAFE,
                "procedure risk exceeds the task-family ceiling",
            )
        if policy.confirmation_required and not envelope.confirmation_required:
            return _fail(
                VerificationLayer.AUTHORITY,
                VerificationReasonCode.AUTHORITY_UNSAFE,
                "verification policy requires confirmation the procedure omitted",
            )
        allowed = set(envelope.requirement_ids)
        for step in procedure.steps:
            if not set(step.required_authority_ids).issubset(allowed):
                return _fail(
                    VerificationLayer.AUTHORITY,
                    VerificationReasonCode.AUTHORITY_UNSAFE,
                    "step authority exceeds the procedure envelope",
                )
        return _pass(VerificationLayer.AUTHORITY)

    def _effect(
        self,
        candidate: ProcedureCandidate,
        evidence: IndependentEvidence,
        policy: VerificationPolicy,
        ir_error: Exception | None,
    ) -> LayerOutcome:
        if ir_error is not None:
            layer, reason = _layer_from_ir_error(ir_error)
            if layer is VerificationLayer.EFFECT:
                return _fail(VerificationLayer.EFFECT, reason, str(ir_error))
        procedure = candidate.procedure
        family = evidence.task_family
        declared = {effect.effect_class for effect in procedure.declared_effects}
        permitted = set(family.boundary.permitted_effect_classes) | set(family.effect_classes)
        if not declared.issubset(permitted):
            return _fail(
                VerificationLayer.EFFECT,
                VerificationReasonCode.EFFECT_UNSAFE,
                "procedure effects exceed the task-family boundary",
            )
        if policy.effect_policy_revision == "":
            return _fail(
                VerificationLayer.EFFECT,
                VerificationReasonCode.STALE_POLICY,
                "effect policy revision is missing",
            )
        return _pass(VerificationLayer.EFFECT, evidence_cids=())

    def _dataflow(self, ir_error: Exception | None) -> LayerOutcome:
        if ir_error is not None:
            layer, reason = _layer_from_ir_error(ir_error)
            if layer is VerificationLayer.DATAFLOW:
                return _fail(VerificationLayer.DATAFLOW, reason, str(ir_error))
        return _pass(VerificationLayer.DATAFLOW)

    def _temporal(
        self,
        candidate: ProcedureCandidate,
        evidence: IndependentEvidence,
        policy: VerificationPolicy,
        now_ms: int,
        ir_error: Exception | None,
    ) -> LayerOutcome:
        if ir_error is not None:
            layer, reason = _layer_from_ir_error(ir_error)
            if layer is VerificationLayer.TEMPORAL:
                return _fail(VerificationLayer.TEMPORAL, reason, str(ir_error))
        procedure = candidate.procedure
        if evidence.observed_at_ms > now_ms:
            return _fail(
                VerificationLayer.TEMPORAL,
                VerificationReasonCode.TEMPORAL_UNSAFE,
                "evidence observation is in the future",
            )
        for step in procedure.steps:
            if step.retry_policy.max_attempts > 1 and not step.retry_policy.requires_new_evidence:
                return _fail(
                    VerificationLayer.TEMPORAL,
                    VerificationReasonCode.TEMPORAL_UNSAFE,
                    "retries cannot proceed without new evidence",
                )
            if step.timeout_ms > procedure.resources.wall_time_ms:
                return _fail(
                    VerificationLayer.TEMPORAL,
                    VerificationReasonCode.TEMPORAL_UNSAFE,
                    "step timeout exceeds the resource envelope",
                )
        if policy.review_horizon_ms <= 0:
            return _fail(
                VerificationLayer.TEMPORAL,
                VerificationReasonCode.TEMPORAL_UNSAFE,
                "verification policy omits a review horizon",
            )
        for receipt in evidence.receipts:
            if receipt.observed_at_ms > now_ms:
                return _fail(
                    VerificationLayer.TEMPORAL,
                    VerificationReasonCode.TEMPORAL_UNSAFE,
                    "receipt observation is in the future",
                    evidence_cids=(receipt.receipt_cid,),
                )
            if receipt.expires_at_ms <= now_ms:
                return _fail(
                    VerificationLayer.TEMPORAL,
                    VerificationReasonCode.STALE_BINDINGS,
                    "evidence receipt expired",
                    evidence_cids=(receipt.receipt_cid,),
                )
        return _pass(VerificationLayer.TEMPORAL)

    def _semantic(
        self,
        candidate: ProcedureCandidate,
        evidence: IndependentEvidence,
        policy: VerificationPolicy,
    ) -> LayerOutcome:
        procedure = candidate.procedure
        family = evidence.task_family
        if not _family_matches(procedure, family):
            return _fail(
                VerificationLayer.SEMANTIC,
                VerificationReasonCode.SEMANTIC_UNSAFE,
                "task family does not bind the procedure",
            )
        if family.bindings != policy.bindings or family.bindings != candidate.bindings:
            return _fail(
                VerificationLayer.SEMANTIC,
                VerificationReasonCode.STALE_BINDINGS,
                "task family bindings are not current",
            )
        if procedure.bindings.repository_id not in family.boundary.permitted_repositories:
            return _fail(
                VerificationLayer.SEMANTIC,
                VerificationReasonCode.SEMANTIC_UNSAFE,
                "procedure repository is outside the family boundary",
            )
        if not set(evidence.repository_families).issubset(
            set(family.boundary.permitted_repositories)
        ):
            return _fail(
                VerificationLayer.SEMANTIC,
                VerificationReasonCode.SEMANTIC_UNSAFE,
                "repository families exceed the task-family boundary",
            )
        if family.boundary.permitted_languages and not set(
            evidence.supported_language_classes
        ).issubset(set(family.boundary.permitted_languages)):
            return _fail(
                VerificationLayer.SEMANTIC,
                VerificationReasonCode.SEMANTIC_UNSAFE,
                "language classes exceed the task-family boundary",
            )
        if family.boundary.permitted_frameworks and not set(
            evidence.supported_framework_classes
        ).issubset(set(family.boundary.permitted_frameworks)):
            return _fail(
                VerificationLayer.SEMANTIC,
                VerificationReasonCode.SEMANTIC_UNSAFE,
                "framework classes exceed the task-family boundary",
            )
        family_ops = set(family.required_operation_contracts)
        procedure_ops = {step.operation_contract for step in procedure.steps}
        if not family_ops.issubset(procedure_ops):
            return _fail(
                VerificationLayer.SEMANTIC,
                VerificationReasonCode.SEMANTIC_UNSAFE,
                "procedure omits a family-required operation contract",
            )
        if candidate.counterexample_set_cid != evidence.counterexample_set_cid:
            return _fail(
                VerificationLayer.SEMANTIC,
                VerificationReasonCode.SEMANTIC_UNSAFE,
                "counterexample-set identity disagrees with independent evidence",
            )
        if not set(candidate.source_episode_cids).issubset(set(evidence.source_episode_cids)):
            return _fail(
                VerificationLayer.SEMANTIC,
                VerificationReasonCode.SEMANTIC_UNSAFE,
                "candidate source episodes are not independently evidenced",
            )
        return _pass(
            VerificationLayer.SEMANTIC,
            evidence_cids=evidence.specification_cids,
        )

    def _validation(
        self,
        candidate: ProcedureCandidate,
        evidence: IndependentEvidence,
        policy: VerificationPolicy,
        now_ms: int,
        ir_error: Exception | None,
    ) -> LayerOutcome:
        if ir_error is not None:
            layer, reason = _layer_from_ir_error(ir_error)
            if layer is VerificationLayer.VALIDATION:
                return _fail(VerificationLayer.VALIDATION, reason, str(ir_error))
        procedure = candidate.procedure
        plan = procedure.validation
        if not set(policy.required_test_contracts).issubset(set(plan.required_test_contracts)):
            return _fail(
                VerificationLayer.VALIDATION,
                VerificationReasonCode.VALIDATION_WEAKENED,
                "procedure validation omits a required test contract",
            )
        if not set(policy.required_proof_contracts).issubset(set(plan.required_proof_contracts)):
            return _fail(
                VerificationLayer.VALIDATION,
                VerificationReasonCode.VALIDATION_WEAKENED,
                "procedure validation omits a required proof contract",
            )
        missing_kinds = []
        if policy.require_adversarial and not evidence.adversarial_assurance_cids:
            missing_kinds.append("adversarial")
        if policy.require_held_out and not evidence.held_out_evaluation_cid:
            missing_kinds.append("held_out")
        if policy.require_shadow and not evidence.shadow_evaluation_cid:
            missing_kinds.append("shadow")
        if not evidence.proof_receipt_cids:
            missing_kinds.append("proof")
        if not evidence.test_receipt_cids:
            missing_kinds.append("test")
        if missing_kinds:
            return _fail(
                VerificationLayer.VALIDATION,
                VerificationReasonCode.VALIDATION_INCOMPLETE,
                "independent evidence omitted required validation: "
                + ",".join(missing_kinds),
            )
        self_ids = _self_identities(candidate) | _bound_input_identities(candidate)
        discharging_cids = (
            *evidence.specification_cids,
            *evidence.proof_receipt_cids,
            *evidence.test_receipt_cids,
            *evidence.adversarial_assurance_cids,
            evidence.held_out_evaluation_cid,
            evidence.shadow_evaluation_cid,
        )
        for cid in discharging_cids:
            if cid.lower() in self_ids:
                return _fail(
                    VerificationLayer.VALIDATION,
                    VerificationReasonCode.SELF_CERTIFICATION,
                    "candidate used its own identity as validation evidence",
                    evidence_cids=(cid,),
                )
        kind_to_cids = {
            "proof": set(evidence.proof_receipt_cids),
            "test": set(evidence.test_receipt_cids),
            "adversarial": set(evidence.adversarial_assurance_cids),
            "held_out": {evidence.held_out_evaluation_cid},
            "shadow": {evidence.shadow_evaluation_cid},
            "specification": set(evidence.specification_cids),
            "source_episode": set(evidence.source_episode_cids),
            "counterexample_set": {evidence.counterexample_set_cid},
        }
        seen_discharging: dict[str, str] = {}
        for kind in ("proof", "test", "adversarial", "held_out", "shadow"):
            for cid in kind_to_cids[kind]:
                previous = seen_discharging.get(cid)
                if previous is not None and previous != kind:
                    return _fail(
                        VerificationLayer.VALIDATION,
                        VerificationReasonCode.VALIDATION_INCOMPLETE,
                        "validation evidence identities are reused across kinds",
                        evidence_cids=(cid,),
                    )
                seen_discharging[cid] = kind
        bound = set(seen_discharging)
        if not evidence.receipts:
            return _fail(
                VerificationLayer.VALIDATION,
                VerificationReasonCode.VALIDATION_INCOMPLETE,
                "independent evidence omitted admitted receipts",
            )
        receipt_cids = {item.receipt_cid for item in evidence.receipts}
        if not bound.issubset(receipt_cids):
            return _fail(
                VerificationLayer.VALIDATION,
                VerificationReasonCode.VALIDATION_INCOMPLETE,
                "validation receipts do not cover bound evidence identities",
            )
        for receipt in evidence.receipts:
            if receipt.kind not in kind_to_cids:
                return _fail(
                    VerificationLayer.VALIDATION,
                    VerificationReasonCode.VALIDATION_INCOMPLETE,
                    "receipt kind is not a required evidence kind",
                    evidence_cids=(receipt.receipt_cid,),
                )
            if receipt.receipt_cid not in kind_to_cids[receipt.kind]:
                return _fail(
                    VerificationLayer.VALIDATION,
                    VerificationReasonCode.VALIDATION_INCOMPLETE,
                    "receipt identity is not bound to its declared kind",
                    evidence_cids=(receipt.receipt_cid,),
                )
            if _is_self_producer(receipt.producer_id, candidate):
                return _fail(
                    VerificationLayer.VALIDATION,
                    VerificationReasonCode.SELF_CERTIFICATION,
                    "receipt producer is not independent",
                    evidence_cids=(receipt.receipt_cid,),
                )
            if receipt.bindings != policy.bindings:
                return _fail(
                    VerificationLayer.VALIDATION,
                    VerificationReasonCode.STALE_BINDINGS,
                    "evidence receipt bindings are stale",
                    evidence_cids=(receipt.receipt_cid,),
                )
            if receipt.expires_at_ms <= now_ms:
                return _fail(
                    VerificationLayer.VALIDATION,
                    VerificationReasonCode.STALE_BINDINGS,
                    "evidence receipt is expired",
                    evidence_cids=(receipt.receipt_cid,),
                )
        covered_tests = {
            receipt.contract_id
            for receipt in evidence.receipts
            if receipt.kind == "test" and receipt.contract_id
        }
        if not set(policy.required_test_contracts).issubset(covered_tests):
            return _fail(
                VerificationLayer.VALIDATION,
                VerificationReasonCode.VALIDATION_WEAKENED,
                "test receipts do not cover the current required test contracts",
            )
        covered_proofs = {
            receipt.contract_id
            for receipt in evidence.receipts
            if receipt.kind == "proof" and receipt.contract_id
        }
        if not set(policy.required_proof_contracts).issubset(covered_proofs):
            return _fail(
                VerificationLayer.VALIDATION,
                VerificationReasonCode.VALIDATION_WEAKENED,
                "proof receipts do not cover the current required proof contracts",
            )
        has_postcondition = any(
            step.operation is StepOperation.CHECK_POSTCONDITION for step in procedure.steps
        )
        if not has_postcondition:
            return _fail(
                VerificationLayer.VALIDATION,
                VerificationReasonCode.VALIDATION_WEAKENED,
                "procedure omits an independently observed postcondition check",
            )
        return _pass(VerificationLayer.VALIDATION, evidence_cids=evidence.evidence_cids)


def verify_procedure(
    candidate: ProcedureCandidate,
    evidence: IndependentEvidence,
    policy: VerificationPolicy,
    *,
    now_ms: int = 0,
) -> ProcedureVerification:
    return ProcedureVerifier().verify(candidate, evidence, policy, now_ms=now_ms)


__all__ = [
    "FORBIDDEN_SELF_PRODUCERS",
    "INDEPENDENT_EVIDENCE_SCHEMA",
    "REQUIRED_EVIDENCE_KINDS",
    "REQUIRED_VERIFICATION_LAYERS",
    "VERIFICATION_POLICY_SCHEMA",
    "VERIFIER_REVISION",
    "AdmittedReceipt",
    "IndependentEvidence",
    "LayerOutcome",
    "ProcedureVerification",
    "ProcedureVerificationError",
    "ProcedureVerifier",
    "VerificationLayer",
    "VerificationPolicy",
    "VerificationReasonCode",
    "VerificationStatus",
    "verify_procedure",
]
