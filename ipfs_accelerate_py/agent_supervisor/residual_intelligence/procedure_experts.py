"""Narrow residual adapter for declared typed procedure holes.

The adapter nominates candidate ProcedureHoleResolution values.  The
procedure compiler remains the exclusive owner of IR, validation, and
authority.  When current compiler capability is missing, the adapter stays
inactive and abstains.  Successful or failed hole records are not training
data without an admitted TrainingCorpusAdmission.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ProcedureHole,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.runtime import (
    compiler_capabilities,
)

from .contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    bounded_int,
    canonical_id,
    reject_candidate_authority,
    required_text,
    strict_fields,
    text_tuple,
)
from .local_experts import IndependentValidationReceipt
from .structured_specialist import (
    ConstrainedStructuredExpert,
    StructuredDecodeRequest,
    StructuredSpecialistPrediction,
)

PROCEDURE_HOLE_ADAPTER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-procedure-hole-adapter@1"
)
PROCEDURE_HOLE_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-procedure-hole-capability@1"
)
PROCEDURE_HOLE_RESOLUTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-procedure-hole-resolution@1"
)
REASON_COMPILER_INACTIVE: Final = "compiler_capability_unavailable"
REASON_HOLE_UNDECLARED: Final = "typed_hole_undeclared"
REASON_PROCEDURE_ROOT_MISMATCH: Final = "exact_procedure_root_mismatch"
REASON_PRECONDITIONS_UNSATISFIED: Final = "procedure_preconditions_unsatisfied"
REASON_VALIDATOR_DECIDES: Final = "independent_validator_decides"
REASON_REPEATED_HOLE_RULE_NOMINATION: Final = "repeated_hole_rule_nomination"
REASON_AUTHORITY_MUTATION_FORBIDDEN: Final = "procedure_authority_mutation_forbidden"
NOMINATED_RULE_PREFIX: Final = "exact_lookup:"
REPEATED_HOLE_NOMINATION_ATTEMPTS: Final = 2


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


@dataclass(frozen=True)
class ProcedureHoleCapability:
    """Read-only snapshot of current compiler capability for this adapter."""

    parse_and_validate: bool
    deterministic_invoke: bool
    synthesize: bool = False
    issue_certificate: bool = False
    promote: bool = False
    modify_policy: bool = False
    program_id: str = ""
    principal_subsystem: str = ""
    schema: str = PROCEDURE_HOLE_CAPABILITY_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "capability_id",
            "available",
            "inactive_reason",
            "parse_and_validate",
            "deterministic_invoke",
            "synthesize",
            "issue_certificate",
            "promote",
            "modify_policy",
            "program_id",
            "principal_subsystem",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != PROCEDURE_HOLE_CAPABILITY_SCHEMA:
            raise ResidualIntelligenceError("unsupported procedure-hole capability schema")
        object.__setattr__(
            self,
            "parse_and_validate",
            _require_bool(self.parse_and_validate, "parse_and_validate"),
        )
        object.__setattr__(
            self,
            "deterministic_invoke",
            _require_bool(self.deterministic_invoke, "deterministic_invoke"),
        )
        for name in (
            "synthesize",
            "issue_certificate",
            "promote",
            "modify_policy",
        ):
            object.__setattr__(self, name, _require_bool(getattr(self, name), name))
            if getattr(self, name) is not False:
                raise ResidualIntelligenceError(REASON_AUTHORITY_MUTATION_FORBIDDEN)
        object.__setattr__(
            self,
            "program_id",
            "" if self.program_id in (None, "") else required_text(self.program_id, "program_id"),
        )
        object.__setattr__(
            self,
            "principal_subsystem",
            ""
            if self.principal_subsystem in (None, "")
            else required_text(self.principal_subsystem, "principal_subsystem"),
        )

    @property
    def available(self) -> bool:
        return self.parse_and_validate is True and self.deterministic_invoke is True

    @property
    def inactive_reason(self) -> str:
        return "" if self.available else REASON_COMPILER_INACTIVE

    @property
    def capability_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "available": self.available,
            "inactive_reason": self.inactive_reason,
            "parse_and_validate": self.parse_and_validate,
            "deterministic_invoke": self.deterministic_invoke,
            "synthesize": False,
            "issue_certificate": False,
            "promote": False,
            "modify_policy": False,
            "program_id": self.program_id,
            "principal_subsystem": self.principal_subsystem,
        }
        if include_id:
            payload["capability_id"] = self.capability_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureHoleCapability:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required={"parse_and_validate", "deterministic_invoke"},
            noun="procedure-hole capability",
        )
        result = cls(
            parse_and_validate=payload["parse_and_validate"],
            deterministic_invoke=payload["deterministic_invoke"],
            synthesize=payload.get("synthesize", False),
            issue_certificate=payload.get("issue_certificate", False),
            promote=payload.get("promote", False),
            modify_policy=payload.get("modify_policy", False),
            program_id=str(payload.get("program_id") or ""),
            principal_subsystem=str(payload.get("principal_subsystem") or ""),
            schema=str(payload.get("schema") or PROCEDURE_HOLE_CAPABILITY_SCHEMA),
        )
        claimed = str(payload.get("capability_id") or "")
        if claimed and claimed != result.capability_id:
            raise ResidualIntelligenceError("procedure-hole capability identity mismatch")
        return result

    @classmethod
    def current(cls, *, overlay: Mapping[str, Any] | None = None) -> ProcedureHoleCapability:
        snapshot = dict(compiler_capabilities())
        if overlay:
            snapshot.update(dict(overlay))
        return cls(
            parse_and_validate=bool(snapshot.get("parse_and_validate")),
            deterministic_invoke=bool(snapshot.get("deterministic_invoke")),
            synthesize=False,
            issue_certificate=False,
            promote=False,
            modify_policy=False,
            program_id=str(snapshot.get("program_id") or ""),
            principal_subsystem=str(snapshot.get("principal_subsystem") or ""),
        )


@dataclass(frozen=True)
class ProcedureHoleResolution:
    """Candidate-only hole fill; never procedure authority or completion."""

    hole_id: str
    procedure_root: str
    operator_id: str
    argument_reference_ids: tuple[str, ...]
    specialist_prediction: StructuredSpecialistPrediction | None
    independent_validation: IndependentValidationReceipt | None
    disposition: ExpertDisposition
    reason_codes: tuple[str, ...]
    nominated_rule: str = ""
    candidate_only: bool = True
    schema: str = PROCEDURE_HOLE_RESOLUTION_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "resolution_id",
            "hole_id",
            "procedure_root",
            "operator_id",
            "argument_reference_ids",
            "specialist_prediction",
            "independent_validation",
            "disposition",
            "reason_codes",
            "nominated_rule",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != PROCEDURE_HOLE_RESOLUTION_SCHEMA:
            raise ResidualIntelligenceError("unsupported procedure-hole resolution schema")
        object.__setattr__(self, "hole_id", required_text(self.hole_id, "hole_id"))
        object.__setattr__(
            self, "procedure_root", required_text(self.procedure_root, "procedure_root")
        )
        object.__setattr__(
            self,
            "operator_id",
            "" if self.operator_id in (None, "") else required_text(self.operator_id, "operator_id"),
        )
        object.__setattr__(
            self,
            "argument_reference_ids",
            text_tuple(self.argument_reference_ids, "argument_reference_ids", max_items=32),
        )
        if self.specialist_prediction is not None and not isinstance(
            self.specialist_prediction, StructuredSpecialistPrediction
        ):
            raise ResidualIntelligenceError(
                "specialist_prediction must be StructuredSpecialistPrediction"
            )
        if self.independent_validation is not None and not isinstance(
            self.independent_validation, IndependentValidationReceipt
        ):
            raise ResidualIntelligenceError(
                "independent_validation must be IndependentValidationReceipt"
            )
        object.__setattr__(self, "disposition", ExpertDisposition(self.disposition))
        object.__setattr__(
            self, "reason_codes", text_tuple(self.reason_codes, "reason_codes", max_items=32)
        )
        object.__setattr__(
            self,
            "nominated_rule",
            ""
            if self.nominated_rule in (None, "")
            else required_text(self.nominated_rule, "nominated_rule"),
        )
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("procedure-hole resolutions must remain candidate_only")
        reject_candidate_authority(
            {
                "hole_id": self.hole_id,
                "operator_id": self.operator_id,
                "nominated_rule": self.nominated_rule,
            }
        )

    @property
    def resolution_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "hole_id": self.hole_id,
            "procedure_root": self.procedure_root,
            "operator_id": self.operator_id,
            "argument_reference_ids": self.argument_reference_ids,
            "specialist_prediction": (
                None
                if self.specialist_prediction is None
                else self.specialist_prediction.to_dict()
            ),
            "independent_validation": (
                None
                if self.independent_validation is None
                else self.independent_validation.to_dict()
            ),
            "disposition": self.disposition.value,
            "reason_codes": self.reason_codes,
            "nominated_rule": self.nominated_rule,
            "candidate_only": True,
        }
        if include_id:
            payload["resolution_id"] = self.resolution_id
        return payload


@dataclass(frozen=True)
class ProcedureHoleExpertAdapter:
    """Connect residual specialists to declared holes without owning the compiler."""

    specialist: ConstrainedStructuredExpert
    procedure_root: str
    declared_holes: tuple[ProcedureHole, ...]
    capability: ProcedureHoleCapability
    schema: str = PROCEDURE_HOLE_ADAPTER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROCEDURE_HOLE_ADAPTER_SCHEMA:
            raise ResidualIntelligenceError("unsupported procedure-hole adapter schema")
        if not isinstance(self.specialist, ConstrainedStructuredExpert):
            raise ResidualIntelligenceError("adapter requires ConstrainedStructuredExpert")
        if self.specialist.task_family is not ResidualTaskFamily.PROCEDURE_HOLE_FILLING:
            raise ResidualIntelligenceError("adapter specialist must be procedure-hole filling")
        object.__setattr__(
            self, "procedure_root", required_text(self.procedure_root, "procedure_root")
        )
        holes = tuple(self.declared_holes)
        if not holes:
            raise ResidualIntelligenceError("adapter requires at least one declared typed hole")
        identities: list[str] = []
        for hole in holes:
            if not isinstance(hole, ProcedureHole):
                raise ResidualIntelligenceError("declared holes must be ProcedureHole contracts")
            identities.append(hole.hole_id)
        if len(set(identities)) != len(identities):
            raise ResidualIntelligenceError("declared holes must have unique hole_id values")
        object.__setattr__(self, "declared_holes", holes)
        if not isinstance(self.capability, ProcedureHoleCapability):
            raise ResidualIntelligenceError("adapter requires ProcedureHoleCapability")

    def hole(self, hole_id: str) -> ProcedureHole | None:
        wanted = required_text(hole_id, "hole_id")
        for hole in self.declared_holes:
            if hole.hole_id == wanted:
                return hole
        return None

    def resolve(
        self,
        request: StructuredDecodeRequest,
        *,
        prior_attempts: int = 0,
        capability: ProcedureHoleCapability | None = None,
    ) -> ProcedureHoleResolution:
        """Nominate one candidate fill for a declared hole.

        The adapter never mutates compiler IR, validators, or authority.
        """

        live = capability or self.capability
        if not isinstance(live, ProcedureHoleCapability):
            raise ResidualIntelligenceError("resolve requires ProcedureHoleCapability")
        attempts = bounded_int(prior_attempts, "prior_attempts", minimum=0, maximum=4)
        features = request.task_input.compact_features
        hole_id = str(features.get("hole_id") or "")
        procedure_root = str(features.get("procedure_root") or "")
        hole = self.hole(hole_id)
        if hole is None:
            return self._blocked(
                hole_id=hole_id or "hole:undeclared",
                procedure_root=procedure_root or self.procedure_root,
                reasons=(REASON_HOLE_UNDECLARED,),
                disposition=ExpertDisposition.REJECT_INPUT,
            )
        if procedure_root != self.procedure_root:
            return self._blocked(
                hole_id=hole.hole_id,
                procedure_root=procedure_root or self.procedure_root,
                reasons=(REASON_PROCEDURE_ROOT_MISMATCH,),
                disposition=ExpertDisposition.REJECT_INPUT,
            )
        if not live.available:
            return self._blocked(
                hole_id=hole.hole_id,
                procedure_root=self.procedure_root,
                reasons=(REASON_COMPILER_INACTIVE,),
                disposition=ExpertDisposition.CAPABILITY_UNAVAILABLE,
            )
        if features.get("procedure_preconditions_satisfied") is not True:
            return self._blocked(
                hole_id=hole.hole_id,
                procedure_root=self.procedure_root,
                reasons=(REASON_PRECONDITIONS_UNSATISFIED, "abstain_escalate"),
                disposition=ExpertDisposition.ABSTAIN,
            )
        nominated = ""
        if attempts >= min(REPEATED_HOLE_NOMINATION_ATTEMPTS, hole.maximum_attempts):
            nominated = f"{NOMINATED_RULE_PREFIX}{hole.hole_id}"
        prediction = self.specialist.predict(request)
        validation = request.independent_validation
        reasons = list(prediction.task_output.reason_codes)
        disposition = prediction.disposition
        operator_id = ""
        arguments: tuple[str, ...] = ()
        payload = prediction.task_output.structured_payload
        if isinstance(payload, Mapping):
            operator_id = str(payload.get("operator_id") or "")
            raw_arguments = payload.get("argument_reference_ids") or ()
            if isinstance(raw_arguments, Sequence) and not isinstance(
                raw_arguments, (str, bytes, bytearray)
            ):
                arguments = tuple(str(item) for item in raw_arguments)
        if nominated:
            reasons.append(REASON_REPEATED_HOLE_RULE_NOMINATION)
        if validation is None:
            reasons.append(REASON_VALIDATOR_DECIDES)
            if disposition is ExpertDisposition.ACCEPT:
                disposition = ExpertDisposition.VALIDATION_REQUIRED
        elif validation.accepted is not True:
            reasons.append(REASON_VALIDATOR_DECIDES)
            disposition = ExpertDisposition.REJECT_INPUT
        else:
            reasons.append(REASON_VALIDATOR_DECIDES)
        return ProcedureHoleResolution(
            hole_id=hole.hole_id,
            procedure_root=self.procedure_root,
            operator_id=operator_id,
            argument_reference_ids=arguments,
            specialist_prediction=prediction,
            independent_validation=validation,
            disposition=disposition,
            reason_codes=tuple(dict.fromkeys(reasons)),
            nominated_rule=nominated,
        )

    def synthesize(self, *_args: Any, **_kwargs: Any) -> None:
        raise ResidualIntelligenceError(REASON_AUTHORITY_MUTATION_FORBIDDEN)

    def promote(self, *_args: Any, **_kwargs: Any) -> None:
        raise ResidualIntelligenceError(REASON_AUTHORITY_MUTATION_FORBIDDEN)

    def modify_policy(self, *_args: Any, **_kwargs: Any) -> None:
        raise ResidualIntelligenceError(REASON_AUTHORITY_MUTATION_FORBIDDEN)

    def _blocked(
        self,
        *,
        hole_id: str,
        procedure_root: str,
        reasons: Sequence[str],
        disposition: ExpertDisposition,
    ) -> ProcedureHoleResolution:
        return ProcedureHoleResolution(
            hole_id=hole_id,
            procedure_root=procedure_root,
            operator_id="",
            argument_reference_ids=(),
            specialist_prediction=None,
            independent_validation=None,
            disposition=disposition,
            reason_codes=tuple(reasons),
        )
