"""DCR-052: select, prove, and bound Doctor transforms and impact.

Interfaces
----------
* ``DoctorTransformProposal@1`` — body-free operator proposal with impact cone.
* ``RepairOperator@1`` — registered operator identity only (no prose bodies).

Predicted symbols: :class:`DoctorTransformProposal`, :func:`synthesize_transform`,
:func:`prove_impact`.

Normative rules (fail-closed)
-----------------------------
* Select only registered operators (Doctor + autonomous-repair catalogue).
* Lose transform authority when logic, proof, source, or impact validation fails.
* Never emit prose source bodies; refs/hashes only.
* Unmodeled effects, cross-root semantic changes, or proof failure → abstain.
* Runtime model calls remain 0.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis.deterministic_doctor_contracts import (
    DoctorAuthorityRoots,
    DoctorOperatorKind,
    DoctorRepairDisposition,
)
from ..analysis.deterministic_doctor_impact import (
    DeterministicDoctorImpactAnalyzer,
    DoctorImpactClosureReceipt,
    DoctorImpactPlanDisposition,
    DoctorImpactRequest,
    create_deterministic_doctor_impact_analyzer,
)
from ..autonomous_repair.operators.registry import (
    OperatorKind,
    build_default_operator_registry,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from ..sca_doctor_bridge import (
    DoctorDiagnosis,
    DoctorDiagnosisDisposition,
    DoctorFinding,
    diagnose_contract_failure,
)
from .deterministic_doctor_transforms import (
    DoctorOperatorProposal,
    DoctorRepairOperatorRegistry,
    build_default_doctor_operator_registry,
    make_edit_site,
)


# ---------------------------------------------------------------------------
# Interfaces / evidence
# ---------------------------------------------------------------------------

DOCTOR_TRANSFORM_PROPOSAL_INTERFACE: Final[str] = "DoctorTransformProposal@1"
REPAIR_OPERATOR_INTERFACE: Final[str] = "RepairOperator@1"
DCR_DOCTOR_TRANSFORM_EVIDENCE: Final[str] = "dcr/doctor-plan@1"
DCR_DOCTOR_TRANSFORM_VERSION: Final[int] = 1

DOCTOR_TRANSFORM_PROPOSAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-transform-proposal@1"
)
DOCTOR_TRANSFORM_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-transform-receipt@1"
)
DOCTOR_TRANSFORMS_CATALOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-transforms-catalog@1"
)
DEFAULT_DOCTOR_TRANSFORMS_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/doctor-transforms.json"
)

# Map DCR-024/051 edge kinds onto closed Doctor analytical operator kinds.
_EDGE_TO_OPERATOR_KIND: Final[Mapping[str, DoctorOperatorKind]] = MappingProxyType(
    {
        "declaration_to_registration": DoctorOperatorKind.ADD_REGISTRATION,
        "registration_to_dispatcher": DoctorOperatorKind.ADD_FACTORY_ROUTE,
        "route_to_dispatcher": DoctorOperatorKind.ADD_CONSTRUCTOR_ROUTE,
        "dispatcher_to_handler": DoctorOperatorKind.ADD_FACTORY_ROUTE,
        "handler_to_effect": DoctorOperatorKind.FINITE_ADAPTER,
        "effect_to_response": DoctorOperatorKind.SCHEMA_PROJECTION,
        "schema": DoctorOperatorKind.SCHEMA_PROJECTION,
        "profile": DoctorOperatorKind.FINITE_ADAPTER,
    }
)


class DoctorTransformDisposition(str, Enum):  # noqa: UP042
    """Closed outcomes for DCR-052 transform selection."""

    PROPOSED = "proposed"
    ABSTAIN_REVIEW = "abstain_review"
    DEFER_CAPABILITY = "defer_capability"
    REJECTED = "rejected"


class DoctorTransformPipelineError(ContractValidationError):
    """Malformed transform input or closed-boundary violation."""


def _fixture_roots(**overrides: str) -> DoctorAuthorityRoots:
    base = {
        "repository_id": "repository:dcr052",
        "forest_id": "forest:dcr052",
        "tree_id": "tree:dcr052",
        "overlay_id": "overlay:dcr052",
        "file_root_id": "file-root:dcr052",
        "ast_root_id": "ast:dcr052",
        "graph_id": "graph:dcr052",
        "corpus_id": "corpus:dcr052",
        "index_id": "index:dcr052",
        "model_id": "model:dcr052",
        "cache_id": "cache:dcr052",
        "operator_registry_id": "operators:dcr052",
        "translator_id": "translator:dcr052",
        "solver_id": "solver:dcr052",
        "kernel_id": "kernel:dcr052",
        "toolchain_id": "toolchain:dcr052",
        "policy_id": "policy:dcr052",
        "sandbox_id": "sandbox:dcr052",
        "environment_id": "environment:dcr052",
        "lease_id": "lease:dcr052",
    }
    base.update(overrides)
    return DoctorAuthorityRoots(**base)


def _assert_body_free(*texts: str) -> None:
    markers = (
        "def ",
        "class ",
        "import ",
        "#!/",
        "private_key",
        "BEGIN ",
        "password=",
    )
    for text in texts:
        lowered = text.lower()
        for marker in markers:
            if marker.lower() in lowered and len(text) > 64:
                # Short refs may contain tokens; only reject long prose bodies.
                if "\n" in text or len(text) > 256:
                    raise DoctorTransformPipelineError(
                        f"prose or source body rejected near {marker!r}"
                    )


# ---------------------------------------------------------------------------
# RepairOperator@1 identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairOperatorRef(CanonicalContract):
    """Registered operator identity (RepairOperator@1)."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/repair-operator-ref@1"
    )
    INTERFACE: ClassVar[str] = REPAIR_OPERATOR_INTERFACE

    operator_id: str
    kind: str
    registry_id: str
    catalogue: str = "doctor"
    grants_write_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator_id", str(self.operator_id).strip())
        object.__setattr__(self, "kind", str(self.kind).strip())
        object.__setattr__(self, "registry_id", str(self.registry_id).strip())
        object.__setattr__(self, "catalogue", str(self.catalogue or "doctor").strip())
        object.__setattr__(self, "grants_write_authority", False)
        if not self.operator_id or not self.kind:
            raise DoctorTransformPipelineError("operator_id and kind are required")

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "operator_id": self.operator_id,
            "kind": self.kind,
            "registry_id": self.registry_id,
            "catalogue": self.catalogue,
            "grants_write_authority": False,
        }


# ---------------------------------------------------------------------------
# DoctorTransformProposal@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorTransformProposal(CanonicalContract):
    """Body-free transform proposal with applicability and impact evidence."""

    SCHEMA: ClassVar[str] = DOCTOR_TRANSFORM_PROPOSAL_SCHEMA
    INTERFACE: ClassVar[str] = DOCTOR_TRANSFORM_PROPOSAL_INTERFACE

    proposal_id: str
    finding_cid: str
    operator: RepairOperatorRef
    write_paths: tuple[str, ...]
    arguments: Mapping[str, str]
    before_hashes: Mapping[str, str]
    expected_after_hashes: Mapping[str, str]
    applicability_proof_cid: str
    impact_cone: tuple[str, ...]
    rollback_ref: str
    expected_proof_transition: str
    disposition: DoctorTransformDisposition = DoctorTransformDisposition.PROPOSED
    reason_codes: tuple[str, ...] = ()
    grants_transform_authority: bool = False
    grants_write_authority: bool = False
    semantic_authority: bool = False
    runtime_model_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposal_id", str(self.proposal_id).strip())
        object.__setattr__(self, "finding_cid", str(self.finding_cid).strip())
        if not isinstance(self.operator, RepairOperatorRef):
            if isinstance(self.operator, Mapping):
                object.__setattr__(
                    self, "operator", RepairOperatorRef.from_dict(self.operator)
                )
            else:
                raise DoctorTransformPipelineError("operator must be RepairOperatorRef")
        paths = tuple(
            str(p).strip()
            for p in (self.write_paths or ())
            if str(p).strip() and ".." not in str(p)
        )
        object.__setattr__(self, "write_paths", paths)
        args = {
            str(k): str(v)
            for k, v in dict(self.arguments or {}).items()
            if str(k).strip()
        }
        for value in args.values():
            _assert_body_free(value)
        object.__setattr__(self, "arguments", MappingProxyType(dict(sorted(args.items()))))
        before = {
            str(k): str(v)
            for k, v in dict(self.before_hashes or {}).items()
            if str(k).strip() and str(v).strip()
        }
        after = {
            str(k): str(v)
            for k, v in dict(self.expected_after_hashes or {}).items()
            if str(k).strip() and str(v).strip()
        }
        object.__setattr__(self, "before_hashes", MappingProxyType(dict(sorted(before.items()))))
        object.__setattr__(
            self, "expected_after_hashes", MappingProxyType(dict(sorted(after.items())))
        )
        object.__setattr__(
            self, "applicability_proof_cid", str(self.applicability_proof_cid or "").strip()
        )
        cone = tuple(str(x).strip() for x in (self.impact_cone or ()) if str(x).strip())
        object.__setattr__(self, "impact_cone", cone)
        object.__setattr__(self, "rollback_ref", str(self.rollback_ref or "").strip())
        object.__setattr__(
            self,
            "expected_proof_transition",
            str(self.expected_proof_transition or "").strip(),
        )
        try:
            disposition = DoctorTransformDisposition(
                str(getattr(self.disposition, "value", self.disposition))
            )
        except ValueError as exc:
            raise DoctorTransformPipelineError(
                f"unsupported disposition: {self.disposition!r}"
            ) from exc
        object.__setattr__(self, "disposition", disposition)
        codes: list[str] = []
        for raw in self.reason_codes or ():
            text = str(raw).strip()
            if text and text not in codes:
                codes.append(text)
        if not codes:
            codes.append(disposition.value)
        object.__setattr__(self, "reason_codes", tuple(codes))
        # Authority flags hard-fail closed unless explicitly proposed+proved later.
        object.__setattr__(self, "grants_transform_authority", False)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "runtime_model_calls", 0)

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "proposal_id": self.proposal_id,
            "finding_cid": self.finding_cid,
            "operator": self.operator.to_dict(),
            "write_paths": list(self.write_paths),
            "arguments": dict(self.arguments),
            "before_hashes": dict(self.before_hashes),
            "expected_after_hashes": dict(self.expected_after_hashes),
            "applicability_proof_cid": self.applicability_proof_cid,
            "impact_cone": list(self.impact_cone),
            "rollback_ref": self.rollback_ref,
            "expected_proof_transition": self.expected_proof_transition,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "grants_transform_authority": False,
            "grants_write_authority": False,
            "semantic_authority": False,
            "runtime_model_calls": 0,
            "evidence_id": DCR_DOCTOR_TRANSFORM_EVIDENCE,
            "version": DCR_DOCTOR_TRANSFORM_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorTransformProposal":
        if not isinstance(payload, Mapping):
            raise DoctorTransformPipelineError("proposal must be an object")
        return cls(
            proposal_id=str(payload.get("proposal_id") or ""),
            finding_cid=str(payload.get("finding_cid") or ""),
            operator=payload.get("operator") or {},
            write_paths=tuple(payload.get("write_paths") or ()),
            arguments=payload.get("arguments") or {},
            before_hashes=payload.get("before_hashes") or {},
            expected_after_hashes=payload.get("expected_after_hashes") or {},
            applicability_proof_cid=str(payload.get("applicability_proof_cid") or ""),
            impact_cone=tuple(payload.get("impact_cone") or ()),
            rollback_ref=str(payload.get("rollback_ref") or ""),
            expected_proof_transition=str(
                payload.get("expected_proof_transition") or ""
            ),
            disposition=payload.get("disposition")
            or DoctorTransformDisposition.ABSTAIN_REVIEW,
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )


@dataclass(frozen=True)
class DoctorTransformReceipt(CanonicalContract):
    """Closed receipt for synthesize_transform / prove_impact."""

    SCHEMA: ClassVar[str] = DOCTOR_TRANSFORM_RECEIPT_SCHEMA

    disposition: DoctorTransformDisposition
    proposal: DoctorTransformProposal | None = None
    impact: Mapping[str, Any] | None = None
    reason_codes: tuple[str, ...] = ()
    grants_transform_authority: bool = False
    runtime_model_calls: int = 0
    evidence_id: str = DCR_DOCTOR_TRANSFORM_EVIDENCE

    def __post_init__(self) -> None:
        try:
            disposition = DoctorTransformDisposition(
                str(getattr(self.disposition, "value", self.disposition))
            )
        except ValueError as exc:
            raise DoctorTransformPipelineError(
                f"unsupported disposition: {self.disposition!r}"
            ) from exc
        object.__setattr__(self, "disposition", disposition)
        proposal = self.proposal
        if proposal is not None and not isinstance(proposal, DoctorTransformProposal):
            proposal = DoctorTransformProposal.from_dict(proposal)
        object.__setattr__(self, "proposal", proposal)
        impact = self.impact
        if impact is not None:
            if not isinstance(impact, Mapping):
                raise DoctorTransformPipelineError("impact must be a mapping")
            object.__setattr__(self, "impact", MappingProxyType(dict(impact)))
        codes: list[str] = []
        for raw in self.reason_codes or ():
            text = str(raw).strip()
            if text and text not in codes:
                codes.append(text)
        if not codes:
            codes.append(disposition.value)
        object.__setattr__(self, "reason_codes", tuple(codes))
        object.__setattr__(self, "grants_transform_authority", False)
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "evidence_id", DCR_DOCTOR_TRANSFORM_EVIDENCE)

    def _payload(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "disposition": self.disposition.value,
            "proposal": None if self.proposal is None else self.proposal.to_dict(),
            "impact": None if self.impact is None else dict(self.impact),
            "reason_codes": list(self.reason_codes),
            "grants_transform_authority": False,
            "runtime_model_calls": 0,
            "version": DCR_DOCTOR_TRANSFORM_VERSION,
        }


# ---------------------------------------------------------------------------
# Selection + synthesis
# ---------------------------------------------------------------------------


def _registered_operator_kinds() -> set[str]:
    doctor_kinds = {item.value for item in DoctorOperatorKind}
    try:
        repair = build_default_operator_registry()
        repair_kinds = {item.value for item in repair.kinds()}
    except Exception:  # noqa: BLE001 - catalogue optional in unit tests
        repair_kinds = set()
    return doctor_kinds | repair_kinds


def _select_operator_kind(finding: DoctorFinding) -> DoctorOperatorKind | None:
    edge = finding.edge_key
    if edge in _EDGE_TO_OPERATOR_KIND:
        return _EDGE_TO_OPERATOR_KIND[edge]
    # Fall back by mismatch class token.
    enum = finding.finding_enum
    if enum in {"schema", "protocol"}:
        return DoctorOperatorKind.SCHEMA_PROJECTION
    if enum in {"implementation", "mediation"}:
        return DoctorOperatorKind.FINITE_ADAPTER
    if enum in {"ambiguous", "unobserved", "expected_only"}:
        return None
    return DoctorOperatorKind.FINITE_ADAPTER


def synthesize_transform(
    diagnosis: DoctorDiagnosis | Mapping[str, Any] | None = None,
    *,
    finding: DoctorFinding | Mapping[str, Any] | None = None,
    roots: DoctorAuthorityRoots | None = None,
    registry: DoctorRepairOperatorRegistry | None = None,
    allow_cross_root: bool = False,
) -> DoctorTransformReceipt:
    """Select a registered operator for a unique diagnosis (no source bodies).

    Returns a proposal receipt.  Transform authority is never granted here —
    :func:`prove_impact` must still pass, and write authority remains false.
    """

    if diagnosis is not None and not isinstance(diagnosis, DoctorDiagnosis):
        diagnosis = DoctorDiagnosis.from_dict(diagnosis)
    if finding is not None and not isinstance(finding, DoctorFinding):
        finding = DoctorFinding.from_dict(finding)

    if diagnosis is not None:
        if diagnosis.disposition is not DoctorDiagnosisDisposition.DIAGNOSED:
            return DoctorTransformReceipt(
                disposition=DoctorTransformDisposition.ABSTAIN_REVIEW,
                proposal=None,
                reason_codes=(
                    "diagnosis_not_actionable",
                    diagnosis.disposition.value,
                    "no_transform",
                ),
            )
        if finding is None:
            finding = diagnosis.earliest

    if finding is None:
        return DoctorTransformReceipt(
            disposition=DoctorTransformDisposition.ABSTAIN_REVIEW,
            proposal=None,
            reason_codes=("missing_finding", "no_transform"),
        )

    if finding.grants_transform_authority:
        # Diagnosis must never carry transform authority into selection.
        return DoctorTransformReceipt(
            disposition=DoctorTransformDisposition.REJECTED,
            proposal=None,
            reason_codes=("diagnosis_claimed_transform_authority", "no_transform"),
        )

    kind = _select_operator_kind(finding)
    if kind is None:
        return DoctorTransformReceipt(
            disposition=DoctorTransformDisposition.ABSTAIN_REVIEW,
            proposal=None,
            reason_codes=(
                "no_applicable_registered_operator",
                finding.finding_enum,
                finding.edge_key,
                "no_transform",
            ),
        )

    registered = _registered_operator_kinds()
    if kind.value not in registered:
        return DoctorTransformReceipt(
            disposition=DoctorTransformDisposition.REJECTED,
            proposal=None,
            reason_codes=("operator_not_registered", kind.value, "no_transform"),
        )

    roots = roots or _fixture_roots(
        graph_id=f"graph:{finding.epoch_cid[:24]}",
        operator_registry_id="operators:doctor-default",
    )
    if registry is None:
        registry = build_default_doctor_operator_registry(roots)

    # Build body-free write set from diagnosis spans/hashes only.
    write_paths = tuple(sorted(finding.source_hashes)) or tuple(
        str(span.get("path"))
        for span in finding.source_spans
        if isinstance(span, Mapping) and span.get("path")
    )
    if not allow_cross_root:
        # Reject paths that escape a single logical root token (cross-repo).
        roots_seen = {
            path.split("/", 1)[0]
            for path in write_paths
            if "/" in path
        }
        if len(roots_seen) > 1:
            return DoctorTransformReceipt(
                disposition=DoctorTransformDisposition.ABSTAIN_REVIEW,
                proposal=None,
                reason_codes=(
                    "cross_root_semantic_change",
                    ",".join(sorted(roots_seen)),
                    "no_transform",
                ),
            )

    before = dict(finding.source_hashes)
    # Expected-after hashes are nominated as content-id of (before, operator, edge)
    # — never fabricated file bodies.
    expected_after = {
        path: content_identity(
            {
                "before": digest,
                "operator_kind": kind.value,
                "edge": finding.edge_key,
                "finding": finding.content_id,
            }
        )
        for path, digest in before.items()
    }
    if not expected_after and write_paths:
        expected_after = {
            path: content_identity(
                {"path": path, "operator_kind": kind.value, "finding": finding.content_id}
            )
            for path in write_paths
        }

    operator = RepairOperatorRef(
        operator_id=f"doctor-operator:{kind.value}@1",
        kind=kind.value,
        registry_id=str(getattr(registry, "content_id", "operators:doctor-default")),
        catalogue="doctor",
    )
    applicability = content_identity(
        {
            "finding": finding.content_id,
            "operator": operator.content_id,
            "edge": finding.edge_key,
            "registry": operator.registry_id,
        }
    )
    proposal = DoctorTransformProposal(
        proposal_id=f"proposal:{finding.content_id[:24]}:{kind.value}",
        finding_cid=finding.content_id,
        operator=operator,
        write_paths=write_paths,
        arguments={
            "edge_key": finding.edge_key,
            "finding_enum": finding.finding_enum,
            "epoch_cid": finding.epoch_cid,
        },
        before_hashes=before,
        expected_after_hashes=expected_after,
        applicability_proof_cid=applicability,
        impact_cone=tuple(write_paths) or (finding.edge_key,),
        rollback_ref=f"rollback:{finding.content_id[:24]}",
        expected_proof_transition=f"proof:{finding.edge_key}->admitted",
        disposition=DoctorTransformDisposition.PROPOSED,
        reason_codes=(
            "registered_operator_selected",
            kind.value,
            "body_free",
            "authority_not_granted",
        ),
    )
    return DoctorTransformReceipt(
        disposition=DoctorTransformDisposition.PROPOSED,
        proposal=proposal,
        reason_codes=proposal.reason_codes,
    )


def prove_impact(
    proposal: DoctorTransformProposal | Mapping[str, Any] | DoctorTransformReceipt,
    *,
    roots: DoctorAuthorityRoots | None = None,
    analyzer: DeterministicDoctorImpactAnalyzer | None = None,
    require_authoritative_closure: bool = False,
) -> DoctorTransformReceipt:
    """Bound impact for a proposal; abstain when impact cannot be closed.

    Does not grant write/transform authority.  Unmodeled / open frontiers lose
    transform eligibility (abstain_review).
    """

    if isinstance(proposal, DoctorTransformReceipt):
        if proposal.proposal is None:
            return DoctorTransformReceipt(
                disposition=DoctorTransformDisposition.ABSTAIN_REVIEW,
                proposal=None,
                reason_codes=("missing_proposal", "no_transform"),
            )
        base_receipt = proposal
        proposal = proposal.proposal
    else:
        base_receipt = None
    if not isinstance(proposal, DoctorTransformProposal):
        proposal = DoctorTransformProposal.from_dict(proposal)

    if proposal.disposition is not DoctorTransformDisposition.PROPOSED:
        return DoctorTransformReceipt(
            disposition=DoctorTransformDisposition.ABSTAIN_REVIEW,
            proposal=proposal,
            reason_codes=("proposal_not_proposed", proposal.disposition.value, "no_transform"),
        )

    roots = roots or _fixture_roots()
    analyzer = analyzer or create_deterministic_doctor_impact_analyzer()

    # Compact observation-only impact request (no program graph required for
    # fixture-bounded DCR-052 acceptance).  Live authoritative closure remains
    # optional and fails closed when required but unavailable.
    try:
        request = DoctorImpactRequest(
            roots=roots,
            subject_symbol_id=f"symbol:{proposal.finding_cid[:20]}",
            change_set_id=f"changeset:{proposal.proposal_id}",
            overlay_id=roots.overlay_id,
            consumers=(),
            second_order_consumers=(),
            frontiers=(),
            edges=(),
            require_authoritative_closure=require_authoritative_closure,
        )
        impact_receipt = analyzer.analyze(request)
    except Exception as exc:  # noqa: BLE001 - map to abstain
        return DoctorTransformReceipt(
            disposition=DoctorTransformDisposition.ABSTAIN_REVIEW,
            proposal=proposal,
            reason_codes=(
                "impact_validation_failed",
                type(exc).__name__,
                "no_transform",
            ),
        )

    impact_dict = (
        impact_receipt.to_dict()
        if hasattr(impact_receipt, "to_dict")
        else {"receipt": str(impact_receipt)}
    )
    # Fail closed on open required frontiers when the receipt exposes them.
    plan_disp = getattr(impact_receipt, "plan_disposition", None)
    if plan_disp is not None and plan_disp is not DoctorImpactPlanDisposition.ADMITTED:
        return DoctorTransformReceipt(
            disposition=DoctorTransformDisposition.ABSTAIN_REVIEW,
            proposal=proposal,
            impact=impact_dict,
            reason_codes=(
                "impact_not_admitted",
                str(getattr(plan_disp, "value", plan_disp)),
                "no_transform",
            ),
        )

    # Still no write/transform authority — only impact-bounded nomination.
    return DoctorTransformReceipt(
        disposition=DoctorTransformDisposition.PROPOSED,
        proposal=proposal,
        impact=impact_dict,
        reason_codes=(
            "impact_bounded",
            "registered_operator_only",
            "authority_not_granted",
            "runtime_model_calls_0",
        ),
    )


def materialize_doctor_transforms(
    *,
    diagnosis: DoctorDiagnosis | None = None,
    destination: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize doctor-transforms.json evidence for DCR-052."""

    if diagnosis is None:
        diagnosis = diagnose_contract_failure(
            repo_root=repo_root, require_shared_epoch=False
        )
    synth = synthesize_transform(diagnosis)
    proved = (
        prove_impact(synth)
        if synth.proposal is not None
        else synth
    )
    payload = {
        "schema": DOCTOR_TRANSFORMS_CATALOG_SCHEMA,
        "interface": DOCTOR_TRANSFORM_PROPOSAL_INTERFACE,
        "evidence_id": DCR_DOCTOR_TRANSFORM_EVIDENCE,
        "version": DCR_DOCTOR_TRANSFORM_VERSION,
        "synthesis": synth.to_dict(),
        "impact": proved.to_dict(),
        "runtime_model_calls": 0,
    }
    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    path = (
        Path(destination)
        if destination is not None
        else root.joinpath(*PurePosixPath(DEFAULT_DOCTOR_TRANSFORMS_PATH).parts)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "DCR_DOCTOR_TRANSFORM_EVIDENCE",
    "DCR_DOCTOR_TRANSFORM_VERSION",
    "DEFAULT_DOCTOR_TRANSFORMS_PATH",
    "DOCTOR_TRANSFORM_PROPOSAL_INTERFACE",
    "DOCTOR_TRANSFORM_PROPOSAL_SCHEMA",
    "DOCTOR_TRANSFORM_RECEIPT_SCHEMA",
    "DOCTOR_TRANSFORMS_CATALOG_SCHEMA",
    "REPAIR_OPERATOR_INTERFACE",
    "DoctorTransformDisposition",
    "DoctorTransformPipelineError",
    "DoctorTransformProposal",
    "DoctorTransformReceipt",
    "RepairOperatorRef",
    "materialize_doctor_transforms",
    "prove_impact",
    "synthesize_transform",
]
