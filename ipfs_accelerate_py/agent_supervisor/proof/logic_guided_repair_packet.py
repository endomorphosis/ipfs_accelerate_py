"""Analytical-first materialization of LPR-001 context overlays (LPR-016).

``LogicGuidedRepairPacketMaterializer``:

1. Requires exact target / atomic plan admission *before* any packet or
   provider work.
2. Attempts deterministic :class:`AnalyticalChangeTransformer` first.
3. On analytical success returns a deterministic
   :class:`LogicGuidedRepairPacket` overlay and **never** invokes a provider.
4. On a behavior-complete syntax / implementation gap builds a
   ``model_required`` overlay via :class:`LogicRepairContextBuilder`,
   projecting through an existing :class:`ChangePropagationEditPacket` or
   :class:`ContractRepairEditPacket` — never a third write-authority packet.
5. Malformed, refused, timeout, or scope-escape proposals create **no write**.

The canonical :class:`LogicGuidedRepairPacket` record is imported from LPR-001
(:mod:`program_logic_prediction_contracts`) and is not redefined here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis.program_logic_prediction_contracts import (
    ContextOverlayDisposition,
    CountermodelDisposition,
    CountermodelValidationReceipt,
    LogicGuidedRepairPacket,
    LogicPredictionReceipt,
    ProgramLogicAuthorityRoots,
    ProgramLogicAuthorityError,
    ProgramLogicPredictionError,
)
from ..context.logic_repair_context import (
    LOGIC_REPAIR_CONTEXT_INTERFACE,
    MODEL_FORBIDDEN_CHOICES,
    LogicRepairContextBuilder,
    LogicRepairContextOverlay,
    LogicRepairContextRequest,
    LogicRepairExpansionHandle,
    LogicRepairPathSpan,
    LogicRepairValidationBinding,
    LogicRepairValidationKind,
    RprPacketInterfaceKind,
    redact_logic_repair_data,
)
from ..planning.analytical_change_transforms import (
    ANALYTICAL_CHANGE_TRANSFORMER_INTERFACE,
    AnalyticalChangeTransformer,
    TransformRenderReceipt,
    TransformSite,
)
from ..proof.change_propagation_edit_packet import (
    CHANGE_PROPAGATION_EDIT_PACKET_INTERFACE,
    ChangePropagationEditPacket,
    PropagationEditStep,
    PropagationEditStepKind,
)
from ..proof.contract_repair_edit_packet import (
    CONTRACT_REPAIR_EDIT_PACKET_INTERFACE,
    ContractRepairEditPacket,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
)
from ..proof.missing_input_synthesis import (
    ValueMappingProof,
)
from ..todo_daemon.change_propagation_provider_router import (
    AnalyticalNonSuccessReason,
    CHANGE_PROPAGATION_PROVIDER_ROUTER_INTERFACE,
    MODEL_FORBIDDEN_CHOICES as PROPAGATION_MODEL_FORBIDDEN_CHOICES,
    PropagationProviderReason,
    PropagationProviderRoutingError,
    WriterLease,
    assert_proposal_within_lease,
    normalize_analytical_non_success_reason,
    parse_proposal_paths,
)


# ---------------------------------------------------------------------------
# Schema / bounds
# ---------------------------------------------------------------------------

LOGIC_GUIDED_REPAIR_PACKET_MATERIALIZER_INTERFACE: Final[str] = (
    "LogicGuidedRepairPacketMaterializer@1"
)
LOGIC_GUIDED_REPAIR_MATERIALIZATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-guided-repair-materialization-receipt@1"
)
LOGIC_GUIDED_PROPOSAL_DISPOSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-guided-proposal-disposition@1"
)

PRODUCER_ID: Final[str] = "logic-guided-repair-packet@1"
CONTRACT_VERSION: Final[int] = 1

MAX_SITES: Final[int] = 256
MAX_REASON_CODES: Final[int] = 64
MAX_PROPOSAL_BYTES: Final[int] = 2_000_000


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class MaterializationDisposition(str, Enum):
    """Closed outcomes for one analytical-first materialization attempt."""

    DETERMINISTIC = "deterministic"
    MODEL_REQUIRED = "model_required"
    ABSTAINED = "abstained"
    REJECTED = "rejected"
    ADMISSION_REQUIRED = "admission_required"
    NO_WRITE = "no_write"


class ProposalFailureKind(str, Enum):
    """Closed proposal failure kinds that must never create a write."""

    MALFORMED = "malformed"
    REFUSED = "refused"
    TIMEOUT = "timeout"
    SCOPE_ESCAPE = "scope_escape"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_FAILURE = "provider_failure"
    UNTRUSTED = "untrusted"
    ADMISSION_REJECTED = "admission_rejected"


class MaterializationReason(str, Enum):
    """Stable machine-readable materialization reason codes."""

    ANALYTICAL_SUCCESS = "analytical_success_no_provider"
    MODEL_REQUIRED_GAP = "behavior_complete_syntax_or_implementation_gap"
    PLAN_NOT_ADMITTED = "exact_plan_admission_required"
    RPR_PACKET_REQUIRED = "existing_rpr_packet_required"
    STEP_NOT_FOUND = "plan_step_not_found"
    ANALYTICAL_UNSUPPORTED = "analytical_transform_unsupported"
    ANALYTICAL_REJECTED = "analytical_transform_rejected"
    BEHAVIOR_INCOMPLETE = "behavior_incomplete"
    VALUE_INCOMPLETE = "value_source_incomplete"
    ROOT_MISMATCH = "root_mismatch"
    LEASE_REQUIRED = "writer_lease_required"
    LEASE_MISMATCH = "writer_lease_path_mismatch"
    PROVIDER_NOT_INVOKED = "provider_not_invoked"
    NO_WRITE = "no_write"
    PROPOSAL_MALFORMED = "proposal_malformed"
    PROPOSAL_REFUSED = "proposal_refused"
    PROPOSAL_TIMEOUT = "proposal_timeout"
    PROPOSAL_SCOPE_ESCAPE = "proposal_scope_escape"
    PACKET_MALFORMED = "packet_malformed"
    OVERLAY_BUILT = "context_overlay_built"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LogicGuidedRepairPacketError(ContractValidationError):
    """Fail-closed error for LPR packet materialization."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: MaterializationReason | str = MaterializationReason.PACKET_MALFORMED,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(getattr(reason_code, "value", reason_code))


class LogicGuidedRepairPacketAuthorityError(LogicGuidedRepairPacketError):
    """Materialization would invent or broaden write/semantic authority."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        if required:
            raise LogicGuidedRepairPacketError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise LogicGuidedRepairPacketError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise LogicGuidedRepairPacketError(f"{name} is required")
    return text


def _identifier(value: Any, name: str) -> str:
    text = _text(value, name, required=True)
    if any(char.isspace() for char in text):
        raise LogicGuidedRepairPacketError(
            f"{name} must be an opaque compact identifier"
        )
    return text


def _ids(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values is None:
        if required:
            raise LogicGuidedRepairPacketError(f"{name} is required")
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise LogicGuidedRepairPacketError(f"{name} must be a sequence of identifiers")
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        ident = _identifier(item, name)
        if ident not in seen:
            seen.add(ident)
            result.append(ident)
    if required and not result:
        raise LogicGuidedRepairPacketError(f"{name} must not be empty")
    return tuple(sorted(result))


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise LogicGuidedRepairPacketError(f"{name} must be a boolean")
    return value


def _prediction_id(receipt: LogicPredictionReceipt | Mapping[str, Any] | str) -> str:
    if isinstance(receipt, str):
        return _identifier(receipt, "prediction_id")
    if isinstance(receipt, LogicPredictionReceipt):
        return receipt.receipt_id
    if isinstance(receipt, Mapping):
        rid = receipt.get("receipt_id") or receipt.get("prediction_id")
        return _identifier(rid, "prediction_id")
    raise LogicGuidedRepairPacketError("prediction receipt is malformed")


def _countermodel_id(
    receipt: CountermodelValidationReceipt | Mapping[str, Any] | str,
) -> str | None:
    if isinstance(receipt, str):
        return _identifier(receipt, "countermodel_id")
    if isinstance(receipt, CountermodelValidationReceipt):
        if receipt.disposition is not CountermodelDisposition.VALIDATED:
            return None
        return receipt.receipt_id
    if isinstance(receipt, Mapping):
        disposition = str(receipt.get("disposition") or "")
        if disposition and disposition != CountermodelDisposition.VALIDATED.value:
            return None
        rid = receipt.get("receipt_id")
        return _identifier(rid, "countermodel_id") if rid else None
    raise LogicGuidedRepairPacketError("countermodel receipt is malformed")


def _analytical_non_success_from_reasons(
    reasons: Sequence[str],
) -> AnalyticalNonSuccessReason | None:
    """Map analytical rejection reasons onto supported model-escalation reasons."""

    mapping = {
        "unsupported_syntax": AnalyticalNonSuccessReason.UNSUPPORTED_SYNTAX,
        "unsupported_kind": AnalyticalNonSuccessReason.UNSUPPORTED_KIND,
        "missing_deterministic_render": AnalyticalNonSuccessReason.MISSING_DETERMINISTIC_RENDER,
        "complex_implementation_required": AnalyticalNonSuccessReason.COMPLEX_IMPLEMENTATION_REQUIRED,
        "behavior_implementation_gap": AnalyticalNonSuccessReason.BEHAVIOR_IMPLEMENTATION_GAP,
        "unsupported": AnalyticalNonSuccessReason.UNSUPPORTED_SYNTAX,
    }
    for reason in reasons:
        key = str(reason).casefold().replace("-", "_")
        if key in mapping:
            return mapping[key]
    return None


# ---------------------------------------------------------------------------
# Request / receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicGuidedRepairMaterializationRequest:
    """Inputs for analytical-first LPR overlay materialization.

    ``plan_admitted`` must be true and an existing RPR packet must be supplied
    before any analytical or provider work proceeds.
    """

    roots: ProgramLogicAuthorityRoots
    plan_admitted: bool
    rpr_packet: ChangePropagationEditPacket | ContractRepairEditPacket | None
    rpr_plan_id: str
    rpr_plan_step_id: str
    writer_lease: WriterLease | None
    admitted_prediction_id: str
    transform_sites: tuple[TransformSite, ...] = ()
    value_mappings: Mapping[str, ValueMappingProof] | None = None
    prediction_receipts: tuple[LogicPredictionReceipt | Mapping[str, Any] | str, ...] = ()
    countermodel_receipts: tuple[
        CountermodelValidationReceipt | Mapping[str, Any] | str, ...
    ] = ()
    chosen_value_refs: tuple[str, ...] = ()
    construction_route_refs: tuple[str, ...] = ()
    admitted_behavior_ids: tuple[str, ...] = ()
    forbidden_path_refs: tuple[str, ...] = ()
    forbidden_semantic_change_refs: tuple[str, ...] = ()
    validation_refs: tuple[str, ...] = ()
    postcondition_refs: tuple[str, ...] = ()
    expansion_handles: tuple[LogicRepairExpansionHandle | Mapping[str, Any], ...] = ()
    untrusted_source_snippets: tuple[Mapping[str, Any], ...] = ()
    untrusted_comment_snippets: tuple[Mapping[str, Any], ...] = ()
    untrusted_issue_snippets: tuple[Mapping[str, Any], ...] = ()
    analytical_non_success_reason: AnalyticalNonSuccessReason | str | None = None
    provider_id: str = ""
    model_id: str = ""
    config_id: str = ""
    packet_id: str = ""
    objective_id: str = ""
    delta_id: str = ""
    change_set_id: str = ""
    consumer_ids: tuple[str, ...] = ()
    scc_group_id: str = ""
    # Optional hook used only for MODEL_REQUIRED escalation tests; analytical
    # success must never call this.
    provider_callable: Callable[[Mapping[str, Any]], Any] | None = None
    transformer: AnalyticalChangeTransformer | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan_admitted, bool):
            raise LogicGuidedRepairPacketError("plan_admitted must be a boolean")
        if self.transform_sites is None:
            object.__setattr__(self, "transform_sites", ())
        if isinstance(self.transform_sites, (str, bytes)) or not isinstance(
            self.transform_sites, Sequence
        ):
            raise LogicGuidedRepairPacketError("transform_sites must be a sequence")
        if len(self.transform_sites) > MAX_SITES:
            raise LogicGuidedRepairPacketError("transform_sites exceed bound")
        if self.transform_sites and not all(
            isinstance(item, TransformSite) for item in self.transform_sites
        ):
            raise LogicGuidedRepairPacketError(
                "transform_sites must contain TransformSite values"
            )


@dataclass(frozen=True)
class LogicGuidedRepairMaterializationReceipt(CanonicalContract):
    """Canonical receipt for one analytical-first materialization attempt."""

    SCHEMA: ClassVar[str] = LOGIC_GUIDED_REPAIR_MATERIALIZATION_RECEIPT_SCHEMA

    disposition: MaterializationDisposition
    reason_codes: tuple[str, ...]
    plan_admitted: bool
    provider_invoked: bool
    write_performed: bool
    analytical_success: bool
    overlay: LogicGuidedRepairPacket | None = None
    context_overlay: LogicRepairContextOverlay | None = None
    analytical_receipts: tuple[TransformRenderReceipt, ...] = ()
    analytical_non_success_reason: str = ""
    rpr_packet_id: str = ""
    rpr_packet_interface: str = ""
    rpr_plan_id: str = ""
    rpr_plan_step_id: str = ""
    writer_lease_id: str = ""
    model_must_not_choose: tuple[str, ...] = MODEL_FORBIDDEN_CHOICES
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        if isinstance(self.disposition, MaterializationDisposition):
            disposition = self.disposition
        else:
            disposition = MaterializationDisposition(
                _text(self.disposition, "disposition")
            )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes")
        )
        object.__setattr__(self, "plan_admitted", _bool(self.plan_admitted, "plan_admitted"))
        object.__setattr__(
            self, "provider_invoked", _bool(self.provider_invoked, "provider_invoked")
        )
        object.__setattr__(
            self, "write_performed", _bool(self.write_performed, "write_performed")
        )
        object.__setattr__(
            self,
            "analytical_success",
            _bool(self.analytical_success, "analytical_success"),
        )
        if self.overlay is not None and not isinstance(
            self.overlay, LogicGuidedRepairPacket
        ):
            raise LogicGuidedRepairPacketError(
                "overlay must be the LPR-001 LogicGuidedRepairPacket"
            )
        if self.context_overlay is not None and not isinstance(
            self.context_overlay, LogicRepairContextOverlay
        ):
            raise LogicGuidedRepairPacketError(
                "context_overlay must be LogicRepairContextOverlay"
            )
        if self.analytical_receipts is None:
            object.__setattr__(self, "analytical_receipts", ())
        if not isinstance(self.analytical_receipts, Sequence) or not all(
            isinstance(item, TransformRenderReceipt)
            for item in self.analytical_receipts
        ):
            raise LogicGuidedRepairPacketError(
                "analytical_receipts must be TransformRenderReceipt values"
            )
        # Authority invariants.
        if self.analytical_success and self.provider_invoked:
            raise LogicGuidedRepairPacketAuthorityError(
                "analytical success must not invoke a provider"
            )
        if self.write_performed and not self.plan_admitted:
            raise LogicGuidedRepairPacketAuthorityError(
                "writes require prior plan admission"
            )
        if (
            disposition is MaterializationDisposition.DETERMINISTIC
            and self.provider_invoked
        ):
            raise LogicGuidedRepairPacketAuthorityError(
                "deterministic disposition forbids provider invocation"
            )
        if disposition in {
            MaterializationDisposition.ABSTAINED,
            MaterializationDisposition.REJECTED,
            MaterializationDisposition.ADMISSION_REQUIRED,
            MaterializationDisposition.NO_WRITE,
        } and self.write_performed:
            raise LogicGuidedRepairPacketAuthorityError(
                "failed materialization cannot perform a write"
            )
        object.__setattr__(
            self,
            "analytical_non_success_reason",
            _text(
                self.analytical_non_success_reason,
                "analytical_non_success_reason",
                required=False,
            ),
        )
        for name in (
            "rpr_packet_id",
            "rpr_packet_interface",
            "rpr_plan_id",
            "rpr_plan_step_id",
            "writer_lease_id",
            "producer_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self,
            "model_must_not_choose",
            _ids(self.model_must_not_choose, "model_must_not_choose")
            or MODEL_FORBIDDEN_CHOICES,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "interface": LOGIC_GUIDED_REPAIR_PACKET_MATERIALIZER_INTERFACE,
            "producer_id": self.producer_id,
            "disposition": self.disposition.value
            if isinstance(self.disposition, MaterializationDisposition)
            else self.disposition,
            "reason_codes": list(self.reason_codes),
            "plan_admitted": self.plan_admitted,
            "provider_invoked": self.provider_invoked,
            "write_performed": self.write_performed,
            "analytical_success": self.analytical_success,
            "overlay_id": self.overlay.packet_id if self.overlay is not None else "",
            "overlay_content_id": (
                self.overlay.content_id if self.overlay is not None else ""
            ),
            "context_capsule_id": (
                self.context_overlay.capsule.capsule_id
                if self.context_overlay is not None
                else ""
            ),
            "analytical_receipt_site_ids": [
                item.site_id for item in self.analytical_receipts
            ],
            "analytical_non_success_reason": self.analytical_non_success_reason,
            "rpr_packet_id": self.rpr_packet_id,
            "rpr_packet_interface": self.rpr_packet_interface,
            "rpr_plan_id": self.rpr_plan_id,
            "rpr_plan_step_id": self.rpr_plan_step_id,
            "writer_lease_id": self.writer_lease_id,
            "model_must_not_choose": list(self.model_must_not_choose),
            "write_authority": False,
            "semantic_authority": False,
        }

    def to_audit_dict(self) -> dict[str, Any]:
        """Serialize including nested overlay records for audit (not identity)."""

        payload = self.to_dict()
        if self.overlay is not None:
            payload["overlay"] = self.overlay.to_record()
        if self.context_overlay is not None:
            payload["context_overlay"] = self.context_overlay.to_dict()
        payload["content_id"] = self.content_id
        return payload

    @property
    def admitted(self) -> bool:
        return self.disposition in {
            MaterializationDisposition.DETERMINISTIC,
            MaterializationDisposition.MODEL_REQUIRED,
        } and self.overlay is not None


@dataclass(frozen=True)
class LogicGuidedProposalDisposition(CanonicalContract):
    """Outcome of disposing an untrusted provider proposal against the overlay.

    Malformed / refused / timeout / scope-escape outcomes always report
    ``write_performed=False``.
    """

    SCHEMA: ClassVar[str] = LOGIC_GUIDED_PROPOSAL_DISPOSITION_SCHEMA

    disposition: MaterializationDisposition
    failure_kind: ProposalFailureKind | str | None
    write_performed: bool
    provider_invoked: bool
    reason_codes: tuple[str, ...]
    proposal_paths: tuple[str, ...] = ()
    writer_lease_id: str = ""
    overlay_packet_id: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.disposition, MaterializationDisposition):
            disposition = self.disposition
        else:
            disposition = MaterializationDisposition(
                _text(self.disposition, "disposition")
            )
        object.__setattr__(self, "disposition", disposition)
        if self.failure_kind is None:
            object.__setattr__(self, "failure_kind", None)
        elif isinstance(self.failure_kind, ProposalFailureKind):
            pass
        else:
            object.__setattr__(
                self,
                "failure_kind",
                ProposalFailureKind(_text(self.failure_kind, "failure_kind")),
            )
        object.__setattr__(
            self, "write_performed", _bool(self.write_performed, "write_performed")
        )
        object.__setattr__(
            self, "provider_invoked", _bool(self.provider_invoked, "provider_invoked")
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes")
        )
        object.__setattr__(
            self, "proposal_paths", _ids(self.proposal_paths, "proposal_paths")
        )
        object.__setattr__(
            self,
            "writer_lease_id",
            _text(self.writer_lease_id, "writer_lease_id", required=False),
        )
        object.__setattr__(
            self,
            "overlay_packet_id",
            _text(self.overlay_packet_id, "overlay_packet_id", required=False),
        )
        if self.failure_kind is not None and self.write_performed:
            raise LogicGuidedRepairPacketAuthorityError(
                "failed proposals must not create a write"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "interface": LOGIC_GUIDED_REPAIR_PACKET_MATERIALIZER_INTERFACE,
            "disposition": self.disposition.value
            if isinstance(self.disposition, MaterializationDisposition)
            else self.disposition,
            "failure_kind": (
                self.failure_kind.value
                if isinstance(self.failure_kind, ProposalFailureKind)
                else self.failure_kind
            ),
            "write_performed": self.write_performed,
            "provider_invoked": self.provider_invoked,
            "reason_codes": list(self.reason_codes),
            "proposal_paths": list(self.proposal_paths),
            "writer_lease_id": self.writer_lease_id if self.write_performed else "",
            "overlay_packet_id": self.overlay_packet_id,
        }

# ---------------------------------------------------------------------------
# Materializer
# ---------------------------------------------------------------------------


class LogicGuidedRepairPacketMaterializer:
    """Analytical-first materializer for LPR-001 context overlays.

    Existing router type checks and RPR packet write authority remain intact:
    this type never issues a writer lease, never broadens permitted paths, and
    never claims write or semantic authority on the overlay.
    """

    INTERFACE: ClassVar[str] = LOGIC_GUIDED_REPAIR_PACKET_MATERIALIZER_INTERFACE

    def __init__(
        self,
        *,
        transformer: AnalyticalChangeTransformer | None = None,
        context_builder: LogicRepairContextBuilder | None = None,
    ) -> None:
        self._transformer = transformer or AnalyticalChangeTransformer()
        self._context_builder = context_builder or LogicRepairContextBuilder()

    def materialize(
        self,
        request: LogicGuidedRepairMaterializationRequest,
    ) -> LogicGuidedRepairMaterializationReceipt:
        if not isinstance(request, LogicGuidedRepairMaterializationRequest):
            raise LogicGuidedRepairPacketError(
                "request must be LogicGuidedRepairMaterializationRequest"
            )

        # 1) Exact plan admission precedes all packet/provider work.
        if not request.plan_admitted:
            return LogicGuidedRepairMaterializationReceipt(
                disposition=MaterializationDisposition.ADMISSION_REQUIRED,
                reason_codes=(MaterializationReason.PLAN_NOT_ADMITTED.value,),
                plan_admitted=False,
                provider_invoked=False,
                write_performed=False,
                analytical_success=False,
                rpr_plan_id=_text(request.rpr_plan_id, "rpr_plan_id", required=False),
                rpr_plan_step_id=_text(
                    request.rpr_plan_step_id, "rpr_plan_step_id", required=False
                ),
            )

        packet = request.rpr_packet
        if packet is None:
            return LogicGuidedRepairMaterializationReceipt(
                disposition=MaterializationDisposition.REJECTED,
                reason_codes=(MaterializationReason.RPR_PACKET_REQUIRED.value,),
                plan_admitted=True,
                provider_invoked=False,
                write_performed=False,
                analytical_success=False,
                rpr_plan_id=_identifier(request.rpr_plan_id, "rpr_plan_id"),
                rpr_plan_step_id=_identifier(
                    request.rpr_plan_step_id, "rpr_plan_step_id"
                ),
            )

        if isinstance(packet, ChangePropagationEditPacket):
            packet_interface = CHANGE_PROPAGATION_EDIT_PACKET_INTERFACE
            packet_id = packet.packet_id
            plan_id = packet.plan_id
            permitted_read = packet.permitted_read_paths
            permitted_write = packet.permitted_write_paths
            step = self._resolve_propagation_step(packet, request.rpr_plan_step_id)
            scc_group_id = request.scc_group_id or (step.scc_group_id if step else "")
            before_hashes = tuple(
                item.to_dict()
                for item in (
                    step.before_hashes if step is not None else packet.before_hashes
                )
            )
            behavior_ids = tuple(
                step.required_behavior_ids
                if step is not None and step.required_behavior_ids
                else packet.required_behavior_ids
            )
            value_sources = (
                step.selected_value_sources
                if step is not None and step.selected_value_sources
                else packet.selected_value_sources
            )
            value_refs = tuple(
                str(item.candidate_id or item.expression_ref)
                for item in value_sources
                if getattr(item, "candidate_id", "") or getattr(item, "expression_ref", "")
            )
            proof_refs = tuple(
                step.proof_refs if step is not None and step.proof_refs else packet.proof_refs
            )
            postconditions = tuple(
                step.postcondition_refs
                if step is not None and step.postcondition_refs
                else packet.per_edit_postcondition_refs
            )
            validation_commands = tuple(packet.validation_commands)
            read_paths = (
                step.read_paths if step is not None and step.read_paths else permitted_read
            )
            write_paths = (
                step.write_paths
                if step is not None and step.write_paths
                else permitted_write
            )
            delta_id = packet.delta_id
            change_set_id = packet.change_set_id
            fixed_point_ref = packet.fixed_point_obligation_ref
            if step is None:
                return LogicGuidedRepairMaterializationReceipt(
                    disposition=MaterializationDisposition.REJECTED,
                    reason_codes=(MaterializationReason.STEP_NOT_FOUND.value,),
                    plan_admitted=True,
                    provider_invoked=False,
                    write_performed=False,
                    analytical_success=False,
                    rpr_packet_id=packet_id,
                    rpr_packet_interface=packet_interface,
                    rpr_plan_id=plan_id,
                    rpr_plan_step_id=_identifier(
                        request.rpr_plan_step_id, "rpr_plan_step_id"
                    ),
                )
        elif isinstance(packet, ContractRepairEditPacket):
            packet_interface = CONTRACT_REPAIR_EDIT_PACKET_INTERFACE
            packet_id = packet.packet_id
            plan_id = request.rpr_plan_id or packet.decision_id
            permitted_read = packet.read_paths
            permitted_write = packet.write_paths
            step = None
            scc_group_id = request.scc_group_id
            before_hashes = ()
            behavior_ids = tuple(request.admitted_behavior_ids)
            value_refs = tuple(request.chosen_value_refs)
            proof_refs = tuple(item.content_id for item in packet.proof_refs)
            postconditions = tuple(packet.post_edit_obligation_ids)
            validation_commands = tuple(packet.validation_commands)
            read_paths = permitted_read
            write_paths = permitted_write
            delta_id = request.delta_id
            change_set_id = request.change_set_id
            fixed_point_ref = ""
        else:
            raise LogicGuidedRepairPacketError(
                "rpr_packet must be ChangePropagationEditPacket@1 or "
                "ContractRepairEditPacket@2",
                reason_code=MaterializationReason.RPR_PACKET_REQUIRED,
            )

        # Bind plan id consistency.
        if request.rpr_plan_id and request.rpr_plan_id != plan_id:
            # Allow request plan id when packet uses decision id as plan anchor.
            if not isinstance(packet, ContractRepairEditPacket):
                return LogicGuidedRepairMaterializationReceipt(
                    disposition=MaterializationDisposition.REJECTED,
                    reason_codes=(MaterializationReason.ROOT_MISMATCH.value,),
                    plan_admitted=True,
                    provider_invoked=False,
                    write_performed=False,
                    analytical_success=False,
                    rpr_packet_id=packet_id,
                    rpr_packet_interface=packet_interface,
                    rpr_plan_id=plan_id,
                    rpr_plan_step_id=request.rpr_plan_step_id,
                )

        lease = request.writer_lease
        if lease is None:
            return LogicGuidedRepairMaterializationReceipt(
                disposition=MaterializationDisposition.REJECTED,
                reason_codes=(MaterializationReason.LEASE_REQUIRED.value,),
                plan_admitted=True,
                provider_invoked=False,
                write_performed=False,
                analytical_success=False,
                rpr_packet_id=packet_id,
                rpr_packet_interface=packet_interface,
                rpr_plan_id=plan_id,
                rpr_plan_step_id=request.rpr_plan_step_id,
            )
        if not isinstance(lease, WriterLease):
            raise LogicGuidedRepairPacketError(
                "writer_lease must be WriterLease",
                reason_code=MaterializationReason.LEASE_REQUIRED,
            )
        if set(lease.permitted_write_paths) != set(write_paths):
            return LogicGuidedRepairMaterializationReceipt(
                disposition=MaterializationDisposition.REJECTED,
                reason_codes=(MaterializationReason.LEASE_MISMATCH.value,),
                plan_admitted=True,
                provider_invoked=False,
                write_performed=False,
                analytical_success=False,
                rpr_packet_id=packet_id,
                rpr_packet_interface=packet_interface,
                rpr_plan_id=plan_id,
                rpr_plan_step_id=request.rpr_plan_step_id,
                writer_lease_id=lease.lease_id,
            )

        # 2) Attempt deterministic AnalyticalChangeTransformer first.
        transformer = request.transformer or self._transformer
        analytical_receipts: list[TransformRenderReceipt] = []
        rejection_reasons: list[str] = []
        if request.transform_sites:
            mappings = request.value_mappings or {}
            for site in request.transform_sites:
                mapping = mappings.get(site.site_id)
                receipt = transformer.render(site, value_mapping=mapping)
                analytical_receipts.append(receipt)
                if not receipt.admitted:
                    rejection_reasons.extend(receipt.rejection_reasons)
                    rejection_reasons.extend(receipt.transform.rejection_reasons)

        analytical_success = bool(analytical_receipts) and all(
            item.admitted for item in analytical_receipts
        )
        # Also treat empty sites + explicit no model reason as non-analytical path.
        if not request.transform_sites:
            analytical_success = False

        prediction_id = _identifier(
            request.admitted_prediction_id, "admitted_prediction_id"
        )
        prediction_ids = [prediction_id]
        for item in request.prediction_receipts:
            prediction_ids.append(_prediction_id(item))

        countermodel_ids: list[str] = []
        for item in request.countermodel_receipts:
            cid = _countermodel_id(item)
            if cid:
                countermodel_ids.append(cid)

        behavior_ids = tuple(
            sorted(set(behavior_ids) | set(request.admitted_behavior_ids))
        )
        chosen_values = tuple(
            sorted(set(value_refs) | set(request.chosen_value_refs))
        )
        postcondition_refs = tuple(
            sorted(set(postconditions) | set(request.postcondition_refs))
        )
        if fixed_point_ref:
            postcondition_refs = tuple(
                sorted(set(postcondition_refs) | {fixed_point_ref})
            )

        # Build path spans from packet authority (never invent paths).
        read_spans = tuple(
            LogicRepairPathSpan(
                path=path,
                role="read",
                before_hash=self._before_hash_for_path(before_hashes, path),
            )
            for path in read_paths
        )
        write_spans = tuple(
            LogicRepairPathSpan(
                path=path,
                role="write",
                before_hash=self._before_hash_for_path(before_hashes, path),
            )
            for path in write_paths
        )

        validation_bindings = self._validation_bindings(
            validation_refs=request.validation_refs,
            validation_commands=validation_commands,
            fixed_point_ref=fixed_point_ref,
        )

        expansion_handles = tuple(
            item
            if isinstance(item, LogicRepairExpansionHandle)
            else LogicRepairExpansionHandle.from_dict(item)
            for item in request.expansion_handles
        )

        rpr_iface = (
            RprPacketInterfaceKind.CHANGE_PROPAGATION
            if packet_interface == CHANGE_PROPAGATION_EDIT_PACKET_INTERFACE
            else RprPacketInterfaceKind.CONTRACT_REPAIR
        )

        if analytical_success:
            # Deterministic path: never invoke provider_callable.
            if request.provider_callable is not None:
                # Explicitly do not call it — record that analytical success
                # short-circuits all provider work.
                pass

            context = self._context_builder.build(
                LogicRepairContextRequest(
                    roots=request.roots,
                    rpr_packet_interface=rpr_iface,
                    rpr_packet_id=packet_id,
                    rpr_plan_id=plan_id,
                    rpr_plan_step_id=request.rpr_plan_step_id,
                    writer_lease_id=lease.lease_id,
                    plan_admitted=True,
                    scc_group_id=scc_group_id,
                    admitted_prediction_ids=tuple(prediction_ids),
                    chosen_value_refs=chosen_values,
                    construction_route_refs=request.construction_route_refs,
                    admitted_behavior_ids=behavior_ids,
                    validated_countermodel_ids=tuple(countermodel_ids),
                    read_spans=read_spans,
                    write_spans=write_spans,
                    forbidden_path_refs=request.forbidden_path_refs,
                    forbidden_semantic_change_refs=request.forbidden_semantic_change_refs,
                    validations=validation_bindings,
                    postcondition_refs=postcondition_refs,
                    expansion_handles=expansion_handles,
                    provider_id="",
                    model_id="",
                    config_id=request.config_id,
                    untrusted_source_snippets=request.untrusted_source_snippets,
                    untrusted_comment_snippets=request.untrusted_comment_snippets,
                    untrusted_issue_snippets=request.untrusted_issue_snippets,
                    objective_id=request.objective_id or request.roots.objective_id,
                    delta_id=delta_id or request.delta_id,
                    change_set_id=change_set_id or request.change_set_id,
                    consumer_ids=request.consumer_ids,
                    proof_refs=proof_refs,
                    disposition=ContextOverlayDisposition.DETERMINISTIC,
                )
            )
            overlay = self._build_overlay(
                request=request,
                packet_id=packet_id,
                plan_id=plan_id,
                lease=lease,
                context=context,
                disposition=ContextOverlayDisposition.DETERMINISTIC,
                prediction_id=prediction_id,
                permitted_read=tuple(read_paths),
                permitted_write=tuple(write_paths),
                postcondition_refs=postcondition_refs,
            )
            return LogicGuidedRepairMaterializationReceipt(
                disposition=MaterializationDisposition.DETERMINISTIC,
                reason_codes=(
                    MaterializationReason.ANALYTICAL_SUCCESS.value,
                    MaterializationReason.PROVIDER_NOT_INVOKED.value,
                ),
                plan_admitted=True,
                provider_invoked=False,
                write_performed=False,
                analytical_success=True,
                overlay=overlay,
                context_overlay=context,
                analytical_receipts=tuple(analytical_receipts),
                rpr_packet_id=packet_id,
                rpr_packet_interface=packet_interface,
                rpr_plan_id=plan_id,
                rpr_plan_step_id=request.rpr_plan_step_id,
                writer_lease_id=lease.lease_id,
            )

        # 3) Model-required path only for supported analytical non-success.
        non_success = request.analytical_non_success_reason
        if non_success is None and rejection_reasons:
            non_success = _analytical_non_success_from_reasons(rejection_reasons)
        if non_success is None:
            return LogicGuidedRepairMaterializationReceipt(
                disposition=MaterializationDisposition.ABSTAINED,
                reason_codes=(
                    MaterializationReason.ANALYTICAL_UNSUPPORTED.value,
                    *tuple(rejection_reasons)[: MAX_REASON_CODES - 1],
                ),
                plan_admitted=True,
                provider_invoked=False,
                write_performed=False,
                analytical_success=False,
                analytical_receipts=tuple(analytical_receipts),
                rpr_packet_id=packet_id,
                rpr_packet_interface=packet_interface,
                rpr_plan_id=plan_id,
                rpr_plan_step_id=request.rpr_plan_step_id,
                writer_lease_id=lease.lease_id,
            )

        try:
            reason = normalize_analytical_non_success_reason(non_success)
        except PropagationProviderRoutingError as exc:
            return LogicGuidedRepairMaterializationReceipt(
                disposition=MaterializationDisposition.REJECTED,
                reason_codes=(exc.reason_code,),
                plan_admitted=True,
                provider_invoked=False,
                write_performed=False,
                analytical_success=False,
                analytical_receipts=tuple(analytical_receipts),
                analytical_non_success_reason=str(non_success),
                rpr_packet_id=packet_id,
                rpr_packet_interface=packet_interface,
                rpr_plan_id=plan_id,
                rpr_plan_step_id=request.rpr_plan_step_id,
                writer_lease_id=lease.lease_id,
            )

        # Behavior / value completeness gates (mirror provider router).
        if isinstance(packet, ChangePropagationEditPacket):
            if step is not None and step.kind is PropagationEditStepKind.ANALYTICAL:
                # Analytical-only steps never open a model path from this layer.
                return LogicGuidedRepairMaterializationReceipt(
                    disposition=MaterializationDisposition.REJECTED,
                    reason_codes=(
                        PropagationProviderReason.ANALYTICAL_ONLY.value,
                    ),
                    plan_admitted=True,
                    provider_invoked=False,
                    write_performed=False,
                    analytical_success=False,
                    analytical_receipts=tuple(analytical_receipts),
                    rpr_packet_id=packet_id,
                    rpr_packet_interface=packet_interface,
                    rpr_plan_id=plan_id,
                    rpr_plan_step_id=request.rpr_plan_step_id,
                    writer_lease_id=lease.lease_id,
                )
            if not behavior_ids:
                return LogicGuidedRepairMaterializationReceipt(
                    disposition=MaterializationDisposition.ABSTAINED,
                    reason_codes=(MaterializationReason.BEHAVIOR_INCOMPLETE.value,),
                    plan_admitted=True,
                    provider_invoked=False,
                    write_performed=False,
                    analytical_success=False,
                    analytical_receipts=tuple(analytical_receipts),
                    analytical_non_success_reason=reason.value,
                    rpr_packet_id=packet_id,
                    rpr_packet_interface=packet_interface,
                    rpr_plan_id=plan_id,
                    rpr_plan_step_id=request.rpr_plan_step_id,
                    writer_lease_id=lease.lease_id,
                )

        if not request.model_id:
            return LogicGuidedRepairMaterializationReceipt(
                disposition=MaterializationDisposition.REJECTED,
                reason_codes=(MaterializationReason.MODEL_REQUIRED_GAP.value,),
                plan_admitted=True,
                provider_invoked=False,
                write_performed=False,
                analytical_success=False,
                analytical_receipts=tuple(analytical_receipts),
                analytical_non_success_reason=reason.value,
                rpr_packet_id=packet_id,
                rpr_packet_interface=packet_interface,
                rpr_plan_id=plan_id,
                rpr_plan_step_id=request.rpr_plan_step_id,
                writer_lease_id=lease.lease_id,
            )

        context = self._context_builder.build(
            LogicRepairContextRequest(
                roots=request.roots,
                rpr_packet_interface=rpr_iface,
                rpr_packet_id=packet_id,
                rpr_plan_id=plan_id,
                rpr_plan_step_id=request.rpr_plan_step_id,
                writer_lease_id=lease.lease_id,
                plan_admitted=True,
                scc_group_id=scc_group_id,
                admitted_prediction_ids=tuple(prediction_ids),
                chosen_value_refs=chosen_values,
                construction_route_refs=request.construction_route_refs,
                admitted_behavior_ids=behavior_ids,
                validated_countermodel_ids=tuple(countermodel_ids),
                read_spans=read_spans,
                write_spans=write_spans,
                forbidden_path_refs=request.forbidden_path_refs,
                forbidden_semantic_change_refs=request.forbidden_semantic_change_refs,
                validations=validation_bindings,
                postcondition_refs=postcondition_refs,
                expansion_handles=expansion_handles,
                provider_id=request.provider_id,
                model_id=request.model_id,
                config_id=request.config_id,
                untrusted_source_snippets=request.untrusted_source_snippets,
                untrusted_comment_snippets=request.untrusted_comment_snippets,
                untrusted_issue_snippets=request.untrusted_issue_snippets,
                objective_id=request.objective_id or request.roots.objective_id,
                delta_id=delta_id or request.delta_id,
                change_set_id=change_set_id or request.change_set_id,
                consumer_ids=request.consumer_ids,
                proof_refs=proof_refs,
                disposition=ContextOverlayDisposition.MODEL_REQUIRED,
            )
        )
        overlay = self._build_overlay(
            request=request,
            packet_id=packet_id,
            plan_id=plan_id,
            lease=lease,
            context=context,
            disposition=ContextOverlayDisposition.MODEL_REQUIRED,
            prediction_id=prediction_id,
            permitted_read=tuple(read_paths),
            permitted_write=tuple(write_paths),
            postcondition_refs=postcondition_refs,
        )

        # Materialization itself never writes and does not invoke the provider.
        # Provider invocation is a separate explicit step (see
        # ``invoke_provider_for_overlay``) so analytical success is trivially
        # free of model calls.
        return LogicGuidedRepairMaterializationReceipt(
            disposition=MaterializationDisposition.MODEL_REQUIRED,
            reason_codes=(
                MaterializationReason.MODEL_REQUIRED_GAP.value,
                MaterializationReason.OVERLAY_BUILT.value,
                MaterializationReason.NO_WRITE.value,
            ),
            plan_admitted=True,
            provider_invoked=False,
            write_performed=False,
            analytical_success=False,
            overlay=overlay,
            context_overlay=context,
            analytical_receipts=tuple(analytical_receipts),
            analytical_non_success_reason=reason.value,
            rpr_packet_id=packet_id,
            rpr_packet_interface=packet_interface,
            rpr_plan_id=plan_id,
            rpr_plan_step_id=request.rpr_plan_step_id,
            writer_lease_id=lease.lease_id,
        )

    def invoke_provider_for_overlay(
        self,
        receipt: LogicGuidedRepairMaterializationReceipt,
        *,
        provider_callable: Callable[[Mapping[str, Any]], Any],
        request: LogicGuidedRepairMaterializationRequest | None = None,
    ) -> LogicGuidedProposalDisposition:
        """Optionally invoke a provider for a model-required overlay only.

        Analytical / non-model-required receipts return without calling the
        provider and without writing.
        """

        if not isinstance(receipt, LogicGuidedRepairMaterializationReceipt):
            raise LogicGuidedRepairPacketError(
                "receipt must be LogicGuidedRepairMaterializationReceipt"
            )
        if receipt.analytical_success or receipt.disposition is (
            MaterializationDisposition.DETERMINISTIC
        ):
            return LogicGuidedProposalDisposition(
                disposition=MaterializationDisposition.DETERMINISTIC,
                failure_kind=None,
                write_performed=False,
                provider_invoked=False,
                reason_codes=(
                    MaterializationReason.ANALYTICAL_SUCCESS.value,
                    MaterializationReason.PROVIDER_NOT_INVOKED.value,
                ),
                overlay_packet_id=(
                    receipt.overlay.packet_id if receipt.overlay is not None else ""
                ),
            )
        if receipt.disposition is not MaterializationDisposition.MODEL_REQUIRED:
            return LogicGuidedProposalDisposition(
                disposition=MaterializationDisposition.NO_WRITE,
                failure_kind=ProposalFailureKind.ADMISSION_REJECTED,
                write_performed=False,
                provider_invoked=False,
                reason_codes=(MaterializationReason.NO_WRITE.value,),
                overlay_packet_id=(
                    receipt.overlay.packet_id if receipt.overlay is not None else ""
                ),
            )
        if receipt.overlay is None or receipt.context_overlay is None:
            return LogicGuidedProposalDisposition(
                disposition=MaterializationDisposition.NO_WRITE,
                failure_kind=ProposalFailureKind.MALFORMED,
                write_performed=False,
                provider_invoked=False,
                reason_codes=(MaterializationReason.PACKET_MALFORMED.value,),
            )

        # Build a redacted provider payload from the context capsule only.
        capsule = receipt.context_overlay.capsule
        payload = redact_logic_repair_data(
            {
                "interface": LOGIC_GUIDED_REPAIR_PACKET_MATERIALIZER_INTERFACE,
                "router_interface": CHANGE_PROPAGATION_PROVIDER_ROUTER_INTERFACE,
                "context_interface": LOGIC_REPAIR_CONTEXT_INTERFACE,
                "transformer_interface": ANALYTICAL_CHANGE_TRANSFORMER_INTERFACE,
                "overlay": receipt.overlay.to_record(),
                "capsule": capsule.to_dict(),
                "authority": {
                    "write_authority": False,
                    "semantic_authority": False,
                    "model_must_not_choose": list(MODEL_FORBIDDEN_CHOICES),
                    "propagation_model_must_not_choose": list(
                        PROPAGATION_MODEL_FORBIDDEN_CHOICES
                    ),
                },
            }
        )
        try:
            raw = provider_callable(MappingProxyType(payload))
        except TimeoutError:
            return self.dispose_failed_proposal(
                ProposalFailureKind.TIMEOUT,
                overlay=receipt.overlay,
                provider_invoked=True,
            )
        except Exception as exc:  # noqa: BLE001 — map provider failures fail-closed
            message = str(exc).casefold()
            if "refus" in message:
                kind = ProposalFailureKind.REFUSED
            elif "timeout" in message:
                kind = ProposalFailureKind.TIMEOUT
            elif "unavailable" in message:
                kind = ProposalFailureKind.PROVIDER_UNAVAILABLE
            else:
                kind = ProposalFailureKind.PROVIDER_FAILURE
            return self.dispose_failed_proposal(
                kind,
                overlay=receipt.overlay,
                provider_invoked=True,
            )

        return self.admit_provider_proposal(
            raw,
            overlay=receipt.overlay,
            lease_write_paths=capsule.permitted_write_paths,
            writer_lease_id=receipt.writer_lease_id,
            provider_invoked=True,
        )

    def dispose_failed_proposal(
        self,
        failure_kind: ProposalFailureKind | str,
        *,
        overlay: LogicGuidedRepairPacket | None = None,
        provider_invoked: bool = True,
        proposal_paths: Sequence[str] = (),
    ) -> LogicGuidedProposalDisposition:
        """Map a failed proposal outcome to a no-write disposition."""

        if isinstance(failure_kind, ProposalFailureKind):
            kind = failure_kind
        else:
            kind = ProposalFailureKind(_text(failure_kind, "failure_kind"))
        reason_map = {
            ProposalFailureKind.MALFORMED: MaterializationReason.PROPOSAL_MALFORMED,
            ProposalFailureKind.REFUSED: MaterializationReason.PROPOSAL_REFUSED,
            ProposalFailureKind.TIMEOUT: MaterializationReason.PROPOSAL_TIMEOUT,
            ProposalFailureKind.SCOPE_ESCAPE: MaterializationReason.PROPOSAL_SCOPE_ESCAPE,
            ProposalFailureKind.PROVIDER_UNAVAILABLE: MaterializationReason.NO_WRITE,
            ProposalFailureKind.PROVIDER_FAILURE: MaterializationReason.NO_WRITE,
            ProposalFailureKind.UNTRUSTED: MaterializationReason.NO_WRITE,
            ProposalFailureKind.ADMISSION_REJECTED: MaterializationReason.NO_WRITE,
        }
        return LogicGuidedProposalDisposition(
            disposition=MaterializationDisposition.NO_WRITE,
            failure_kind=kind,
            write_performed=False,
            provider_invoked=provider_invoked,
            reason_codes=(
                reason_map[kind].value,
                MaterializationReason.NO_WRITE.value,
            ),
            proposal_paths=tuple(proposal_paths),
            writer_lease_id="",
            overlay_packet_id=overlay.packet_id if overlay is not None else "",
        )

    def admit_provider_proposal(
        self,
        proposal: Any,
        *,
        overlay: LogicGuidedRepairPacket,
        lease_write_paths: Sequence[str],
        writer_lease_id: str,
        provider_invoked: bool = True,
    ) -> LogicGuidedProposalDisposition:
        """Parse and scope-check an untrusted proposal; never write on failure."""

        if not isinstance(overlay, LogicGuidedRepairPacket):
            raise LogicGuidedRepairPacketError(
                "overlay must be LogicGuidedRepairPacket"
            )
        if overlay.write_authority is not False:
            raise LogicGuidedRepairPacketAuthorityError(
                "overlay cannot claim write authority"
            )

        # Malformed: non-mapping / missing structure.
        if proposal is None:
            return self.dispose_failed_proposal(
                ProposalFailureKind.MALFORMED,
                overlay=overlay,
                provider_invoked=provider_invoked,
            )
        if isinstance(proposal, (bytes, bytearray)):
            if len(proposal) > MAX_PROPOSAL_BYTES:
                return self.dispose_failed_proposal(
                    ProposalFailureKind.MALFORMED,
                    overlay=overlay,
                    provider_invoked=provider_invoked,
                )
            try:
                proposal = proposal.decode("utf-8")
            except UnicodeDecodeError:
                return self.dispose_failed_proposal(
                    ProposalFailureKind.MALFORMED,
                    overlay=overlay,
                    provider_invoked=provider_invoked,
                )
        if isinstance(proposal, str):
            # Raw patch text is treated as an untrusted proposal body.
            proposal = {"patch": proposal, "source": "provider"}
        if not isinstance(proposal, Mapping):
            return self.dispose_failed_proposal(
                ProposalFailureKind.MALFORMED,
                overlay=overlay,
                provider_invoked=provider_invoked,
            )
        if proposal.get("refused") is True or proposal.get("status") == "refused":
            return self.dispose_failed_proposal(
                ProposalFailureKind.REFUSED,
                overlay=overlay,
                provider_invoked=provider_invoked,
            )
        if proposal.get("timeout") is True or proposal.get("status") == "timeout":
            return self.dispose_failed_proposal(
                ProposalFailureKind.TIMEOUT,
                overlay=overlay,
                provider_invoked=provider_invoked,
            )

        try:
            paths = parse_proposal_paths(proposal)
        except PropagationProviderRoutingError:
            return self.dispose_failed_proposal(
                ProposalFailureKind.MALFORMED,
                overlay=overlay,
                provider_invoked=provider_invoked,
            )

        try:
            paths = assert_proposal_within_lease(
                proposal,
                allowed_write_paths=lease_write_paths,
            )
        except PropagationProviderRoutingError:
            return self.dispose_failed_proposal(
                ProposalFailureKind.SCOPE_ESCAPE,
                overlay=overlay,
                provider_invoked=provider_invoked,
                proposal_paths=paths,
            )

        # Successful parse/scope check still creates no write from this module —
        # write authority remains on the existing RPR lease/router path.
        return LogicGuidedProposalDisposition(
            disposition=MaterializationDisposition.MODEL_REQUIRED,
            failure_kind=None,
            write_performed=False,
            provider_invoked=provider_invoked,
            reason_codes=(
                MaterializationReason.OVERLAY_BUILT.value,
                MaterializationReason.NO_WRITE.value,
            ),
            proposal_paths=tuple(paths),
            writer_lease_id="",
            overlay_packet_id=overlay.packet_id,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_propagation_step(
        packet: ChangePropagationEditPacket,
        step_id: str,
    ) -> PropagationEditStep | None:
        want = _identifier(step_id, "rpr_plan_step_id")
        for step in packet.steps:
            if step.step_id == want:
                return step
        return None

    @staticmethod
    def _before_hash_for_path(
        before_hashes: Sequence[Mapping[str, Any]],
        path: str,
    ) -> str:
        for item in before_hashes:
            if not isinstance(item, Mapping):
                continue
            if item.get("path") == path:
                return str(item.get("before_hash") or "")
        return ""

    @staticmethod
    def _validation_bindings(
        *,
        validation_refs: Sequence[str],
        validation_commands: Sequence[str],
        fixed_point_ref: str,
    ) -> tuple[LogicRepairValidationBinding, ...]:
        bindings: list[LogicRepairValidationBinding] = []
        for ref in validation_refs:
            kind = LogicRepairValidationKind.TEST
            lower = ref.casefold()
            if "type" in lower:
                kind = LogicRepairValidationKind.TYPE
            elif "effect" in lower:
                kind = LogicRepairValidationKind.EFFECT
            elif "resource" in lower:
                kind = LogicRepairValidationKind.RESOURCE
            elif "fixed" in lower:
                kind = LogicRepairValidationKind.FIXED_POINT
            bindings.append(
                LogicRepairValidationBinding(validation_id=ref, kind=kind)
            )
        for index, command in enumerate(validation_commands):
            bindings.append(
                LogicRepairValidationBinding(
                    validation_id=f"validation:command:{index}",
                    kind=LogicRepairValidationKind.TEST,
                    command_ref=str(command)[:256],
                )
            )
        if fixed_point_ref:
            bindings.append(
                LogicRepairValidationBinding(
                    validation_id=fixed_point_ref
                    if fixed_point_ref.startswith("validation:")
                    or ":" in fixed_point_ref
                    else f"validation:{fixed_point_ref}",
                    kind=LogicRepairValidationKind.FIXED_POINT,
                    command_ref=fixed_point_ref,
                )
            )
        # Deduplicate by validation_id.
        seen: set[str] = set()
        unique: list[LogicRepairValidationBinding] = []
        for item in bindings:
            if item.validation_id in seen:
                continue
            seen.add(item.validation_id)
            unique.append(item)
        return tuple(unique)

    def _build_overlay(
        self,
        *,
        request: LogicGuidedRepairMaterializationRequest,
        packet_id: str,
        plan_id: str,
        lease: WriterLease,
        context: LogicRepairContextOverlay,
        disposition: ContextOverlayDisposition,
        prediction_id: str,
        permitted_read: Sequence[str],
        permitted_write: Sequence[str],
        postcondition_refs: Sequence[str],
    ) -> LogicGuidedRepairPacket:
        """Materialize the LPR-001 canonical overlay (never redefine the type)."""

        capsule = context.capsule
        packet_id_overlay = request.packet_id or f"overlay:{capsule.capsule_id}"
        invalidation = (
            request.roots.tree_id,
            lease.lease_id,
            packet_id,
            plan_id,
        )
        try:
            return LogicGuidedRepairPacket(
                roots=request.roots,
                packet_id=packet_id_overlay
                if str(packet_id_overlay).startswith("overlay:")
                else f"overlay:{packet_id_overlay}",
                admitted_prediction_id=prediction_id,
                rpr_packet_id=packet_id,
                rpr_plan_id=plan_id,
                rpr_plan_step_id=request.rpr_plan_step_id,
                writer_lease_id=lease.lease_id,
                disposition=disposition,
                context_capsule_id=capsule.capsule_id,
                scope_path_refs=tuple(
                    f"path:{path}" for path in (*permitted_read, *permitted_write)
                ),
                before_hash_refs=capsule.before_hash_refs,
                permitted_read_paths=tuple(permitted_read),
                permitted_write_paths=tuple(permitted_write)
                if disposition
                not in {
                    ContextOverlayDisposition.ABSTAINED,
                    ContextOverlayDisposition.REJECTED,
                }
                else (),
                forbidden_path_refs=request.forbidden_path_refs,
                forbidden_semantic_change_refs=request.forbidden_semantic_change_refs,
                postcondition_refs=tuple(postcondition_refs),
                validation_refs=capsule.validation_refs,
                rollback_policy_ref="",
                expansion_handle_refs=capsule.expansion_handle_refs,
                provider_id=request.provider_id
                if disposition is ContextOverlayDisposition.MODEL_REQUIRED
                else "",
                model_id=request.model_id
                if disposition is ContextOverlayDisposition.MODEL_REQUIRED
                else "",
                config_id=request.config_id,
                write_authority=False,
                semantic_authority=False,
                invalidation_refs=invalidation,
            )
        except (ProgramLogicPredictionError, ProgramLogicAuthorityError) as exc:
            raise LogicGuidedRepairPacketError(
                f"failed to materialize LogicGuidedRepairPacket: {exc}",
                reason_code=MaterializationReason.PACKET_MALFORMED,
            ) from exc


def materialize_logic_guided_repair_packet(
    request: LogicGuidedRepairMaterializationRequest,
    *,
    transformer: AnalyticalChangeTransformer | None = None,
    context_builder: LogicRepairContextBuilder | None = None,
) -> LogicGuidedRepairMaterializationReceipt:
    """Module-level convenience wrapper for analytical-first materialization."""

    return LogicGuidedRepairPacketMaterializer(
        transformer=transformer,
        context_builder=context_builder,
    ).materialize(request)


__all__ = [
    "LOGIC_GUIDED_REPAIR_PACKET_MATERIALIZER_INTERFACE",
    "LOGIC_GUIDED_REPAIR_MATERIALIZATION_RECEIPT_SCHEMA",
    "LOGIC_GUIDED_PROPOSAL_DISPOSITION_SCHEMA",
    "PRODUCER_ID",
    "CONTRACT_VERSION",
    "MAX_SITES",
    "MaterializationDisposition",
    "ProposalFailureKind",
    "MaterializationReason",
    "LogicGuidedRepairPacketError",
    "LogicGuidedRepairPacketAuthorityError",
    "LogicGuidedRepairMaterializationRequest",
    "LogicGuidedRepairMaterializationReceipt",
    "LogicGuidedProposalDisposition",
    "LogicGuidedRepairPacketMaterializer",
    "materialize_logic_guided_repair_packet",
    # Re-export canonical LPR-001 type for callers / tests.
    "LogicGuidedRepairPacket",
    "ContextOverlayDisposition",
]
