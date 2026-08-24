"""Fail-closed procedure drift monitoring and exact registry recovery.

The services in this module do not own execution, compensation, promotion, or
completion authority.  They turn independently supplied observations into
content-addressed drift reports, prepare exact immutable rollback coordinates,
and delegate every mutation to :mod:`procedure_compiler.registry`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import CanonicalContract, content_identity
from .contracts import (
    PROCEDURE_CONTRACT_VERSION,
    ArtifactState,
    ProcedureDriftReport,
    _bounded,
    _decode_fields,
    _enum,
    _identifier,
    _nonnegative_int,
    _positive_int,
    _strings,
    _verify_identity,
)
from .registry import (
    DRIFT_ACTOR_ID,
    ProcedureRegistry,
    ProcedureRegistryError,
    RegistryAuthorization,
    RegistryAuthorizationError,
    RegistryCASError,
    RegistryCASOutcome,
    RegistryDriftCause,
    RegistryDriftDisposition,
    RegistryLifecycleState,
    RegistryMutation,
    RegistryNotFoundError,
    RegistryOperation,
    drift_disposition_for,
)

PROCEDURE_DRIFT_MONITOR_REVISION: Final[str] = "ProcedureDriftMonitor@1"
PROCEDURE_RECOVERY_PLANNER_REVISION: Final[str] = "ProcedureRecoveryPlanner@1"
PROCEDURE_ROLLBACK_SERVICE_REVISION: Final[str] = "ProcedureRollbackService@1"
PROCEDURE_DRIFT_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/procedure-compiler/"
    "procedure-drift-observation@1"
)
REGISTRY_RECOVERY_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/procedure-compiler/"
    "registry-recovery-plan@1"
)
ROLLBACK_VERIFICATION_FAILURE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/procedure-compiler/"
    "rollback-verification-failure@1"
)

# Public compatibility names use the one registry-owned closed vocabulary.
DriftDimension = RegistryDriftCause
DriftDisposition = RegistryDriftDisposition


class ProcedureRecoveryError(ProcedureRegistryError):
    """A drift or recovery request is not exact, current, or independently bound."""


class ProcedureRollbackPostconditionError(ProcedureRecoveryError):
    """The registry mutation committed but its externally visible head was invalid."""


class ProcedureRollbackFailure(ProcedureRecoveryError):
    """A rollback failed, was quarantined, and did not claim successful recovery."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = _identifier(reason_code, "reason_code")


@dataclass(frozen=True)
class ProcedureDriftObservation(CanonicalContract):
    """Independent observation bound to the exact registry head it examined."""

    SCHEMA: ClassVar[str] = PROCEDURE_DRIFT_OBSERVATION_SCHEMA

    procedure_id: str
    expected_revision_id: str
    dimension: RegistryDriftCause
    observer_id: str
    expected_cid: str
    observed_cid: str
    evidence_cids: tuple[str, ...]
    observed_at_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "procedure_id", _identifier(self.procedure_id, "procedure_id"))
        object.__setattr__(
            self,
            "expected_revision_id",
            _identifier(self.expected_revision_id, "expected_revision_id"),
        )
        object.__setattr__(
            self, "dimension", _enum(self.dimension, RegistryDriftCause, "dimension")
        )
        object.__setattr__(self, "observer_id", _identifier(self.observer_id, "observer_id"))
        object.__setattr__(self, "expected_cid", _identifier(self.expected_cid, "expected_cid"))
        object.__setattr__(self, "observed_cid", _identifier(self.observed_cid, "observed_cid"))
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(
                self.evidence_cids,
                "evidence_cids",
                identifiers=True,
                required=True,
                preserve_order=False,
            ),
        )
        object.__setattr__(
            self,
            "observed_at_ms",
            _nonnegative_int(self.observed_at_ms, "observed_at_ms"),
        )
        if self.expected_cid == self.observed_cid:
            raise ProcedureRecoveryError("drift observation must describe a changed value")
        if self.observer_id.lower() == DRIFT_ACTOR_ID:
            raise ProcedureRecoveryError("internal drift actor cannot produce observations")
        _bounded(self, self.__class__.__name__)

    @property
    def disposition(self) -> RegistryDriftDisposition:
        return drift_disposition_for(self.dimension)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "procedure_id": self.procedure_id,
            "expected_revision_id": self.expected_revision_id,
            "dimension": self.dimension.value,
            "observer_id": self.observer_id,
            "expected_cid": self.expected_cid,
            "observed_cid": self.observed_cid,
            "evidence_cids": self.evidence_cids,
            "observed_at_ms": self.observed_at_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureDriftObservation:
        fields = (
            "procedure_id",
            "expected_revision_id",
            "dimension",
            "observer_id",
            "expected_cid",
            "observed_cid",
            "evidence_cids",
            "observed_at_ms",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ProcedureDriftResult:
    """Typed result containing the canonical report and the registry CAS."""

    observation: ProcedureDriftObservation
    disposition: RegistryDriftDisposition
    report: ProcedureDriftReport
    mutation: RegistryMutation
    usable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ProcedureDriftObservation):
            raise ProcedureRecoveryError("drift result requires an exact observation")
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RegistryDriftDisposition, "disposition"),
        )
        if not isinstance(self.report, ProcedureDriftReport):
            raise ProcedureRecoveryError("drift result requires ProcedureDriftReport")
        if not isinstance(self.mutation, RegistryMutation):
            raise ProcedureRecoveryError("drift result requires a registry mutation")
        if type(self.usable) is not bool or self.usable:
            raise ProcedureRecoveryError("an admitted drift result cannot be usable")


@dataclass(frozen=True)
class RegistryRecoveryPlan(CanonicalContract):
    """Non-authoritative rollback coordinates tied to exact immutable rows."""

    SCHEMA: ClassVar[str] = REGISTRY_RECOVERY_PLAN_SCHEMA

    procedure_id: str
    expected_head_revision_id: str
    source_procedure_cid: str
    rollback_target_revision_id: str
    target_procedure_cid: str
    target_revision_generation: int

    def __post_init__(self) -> None:
        for name in (
            "procedure_id",
            "expected_head_revision_id",
            "source_procedure_cid",
            "rollback_target_revision_id",
            "target_procedure_cid",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "target_revision_generation",
            _positive_int(
                self.target_revision_generation,
                "target_revision_generation",
                maximum=1_000_000,
            ),
        )
        if self.expected_head_revision_id == self.rollback_target_revision_id:
            raise ProcedureRecoveryError("rollback target must differ from the current head")
        _bounded(self, self.__class__.__name__)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "procedure_id": self.procedure_id,
            "expected_head_revision_id": self.expected_head_revision_id,
            "source_procedure_cid": self.source_procedure_cid,
            "rollback_target_revision_id": self.rollback_target_revision_id,
            "target_procedure_cid": self.target_procedure_cid,
            "target_revision_generation": self.target_revision_generation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RegistryRecoveryPlan:
        fields = (
            "procedure_id",
            "expected_head_revision_id",
            "source_procedure_cid",
            "rollback_target_revision_id",
            "target_procedure_cid",
            "target_revision_generation",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


# Short name retained without shadowing contracts.ProcedureRecoveryPlan.
RecoveryPlan = RegistryRecoveryPlan


class ProcedureDriftMonitor:
    """Admit typed observations and make the exact registry head less usable."""

    revision: Final[str] = PROCEDURE_DRIFT_MONITOR_REVISION

    def __init__(self, registry: ProcedureRegistry) -> None:
        if not isinstance(registry, ProcedureRegistry):
            raise ProcedureRecoveryError("drift monitor requires a ProcedureRegistry")
        self._registry = registry

    def observe(self, observation: ProcedureDriftObservation) -> ProcedureDriftResult:
        if not isinstance(observation, ProcedureDriftObservation):
            raise ProcedureRecoveryError("drift monitor requires ProcedureDriftObservation")
        observed_head = self._registry.get(
            observation.procedure_id, demote_stale=False
        )
        if observed_head.revision_id == observation.expected_revision_id and (
            observation.observer_id.lower()
            in {
                observation.procedure_id.lower(),
                observed_head.procedure_cid.lower(),
            }
        ):
            raise ProcedureRecoveryError("a procedure cannot report its own drift")

        mutation = self._registry.apply_drift(
            procedure_id=observation.procedure_id,
            expected_old_revision_id=observation.expected_revision_id,
            cause=observation.dimension,
            drift_report_cid=observation.content_id,
            now_ms=observation.observed_at_ms,
        )
        actual_state = ArtifactState(mutation.revision.state.value)
        references = tuple(
            sorted(
                {
                    observation.content_id,
                    observation.expected_revision_id,
                    observation.expected_cid,
                    observation.observed_cid,
                    mutation.revision.revision_id,
                    mutation.cas.cas_id,
                    *observation.evidence_cids,
                }
            )
        )
        report = ProcedureDriftReport(
            bindings=mutation.revision.bindings,
            artifact_version=max(1, mutation.revision.generation),
            state=actual_state,
            subject_cid=mutation.revision.procedure_cid,
            reference_cids=references,
            labels=(
                observation.procedure_id,
                observation.dimension.value,
                observation.disposition.value,
            ),
            facts={
                "cas_id": mutation.cas.cas_id,
                "cas_outcome": mutation.cas.outcome.value,
                "drift_cause": observation.dimension.value,
                "expected_cid": observation.expected_cid,
                "expected_revision_id": observation.expected_revision_id,
                "no_op": mutation.cas.outcome is RegistryCASOutcome.NOOP,
                "observed_cid": observation.observed_cid,
                "observer_id": observation.observer_id,
                "registry_state": mutation.revision.state.value,
                "requested_disposition": observation.disposition.value,
            },
            created_at_ms=observation.observed_at_ms,
        )
        if self._registry.lookup_exact(mutation.revision.procedure_cid) is not None:
            raise ProcedureRecoveryError("admitted drift did not remove registry usability")
        return ProcedureDriftResult(
            observation=observation,
            disposition=observation.disposition,
            report=report,
            mutation=mutation,
        )

    report = observe


class ProcedureRecoveryPlanner:
    """Read immutable history and select only the exact recorded rollback row."""

    revision: Final[str] = PROCEDURE_RECOVERY_PLANNER_REVISION

    def __init__(self, registry: ProcedureRegistry) -> None:
        if not isinstance(registry, ProcedureRegistry):
            raise ProcedureRecoveryError("recovery planner requires a ProcedureRegistry")
        self._registry = registry

    def plan(self, procedure_id: str) -> RegistryRecoveryPlan:
        head = self._registry.get(procedure_id, demote_stale=False)
        target_id = head.rollback_target_revision_id
        if not target_id:
            raise ProcedureRecoveryError("current revision has no recorded rollback target")
        target = self._registry.get_revision(target_id)
        if target.procedure_id != head.procedure_id:
            raise ProcedureRecoveryError("recorded rollback target belongs to another procedure")
        if target.state is not RegistryLifecycleState.PROMOTED:
            raise ProcedureRecoveryError("recorded rollback target was not promoted")
        intact_ids = {item.revision_id for item in self._registry.history(head.procedure_id)}
        if target.revision_id not in intact_ids:
            raise ProcedureRecoveryError("recorded rollback target is not intact")
        return RegistryRecoveryPlan(
            procedure_id=head.procedure_id,
            expected_head_revision_id=head.revision_id,
            source_procedure_cid=head.procedure_cid,
            rollback_target_revision_id=target.revision_id,
            target_procedure_cid=target.procedure_cid,
            target_revision_generation=target.generation,
        )


class ProcedureRollbackService:
    """Delegate an exact authorized rollback and quarantine every failed attempt."""

    revision: Final[str] = PROCEDURE_ROLLBACK_SERVICE_REVISION

    def __init__(self, registry: ProcedureRegistry) -> None:
        if not isinstance(registry, ProcedureRegistry):
            raise ProcedureRecoveryError("rollback service requires a ProcedureRegistry")
        self._registry = registry

    def rollback(
        self,
        plan: RegistryRecoveryPlan,
        *,
        authorization: RegistryAuthorization,
        now_ms: int | None = None,
    ) -> RegistryMutation:
        if not isinstance(plan, RegistryRecoveryPlan):
            raise ProcedureRecoveryError("rollback requires RegistryRecoveryPlan")
        mutation: RegistryMutation | None = None
        containment_revision_id = ""
        try:
            if not isinstance(authorization, RegistryAuthorization):
                raise RegistryAuthorizationError(
                    "rollback requires RegistryAuthorization"
                )
            head = self._registry.get(plan.procedure_id, demote_stale=False)
            if head.revision_id != plan.expected_head_revision_id:
                raise ProcedureRecoveryError("recovery plan is stale")
            if head.procedure_cid != plan.source_procedure_cid:
                raise ProcedureRecoveryError("recovery plan source procedure drifted")
            if head.rollback_target_revision_id != plan.rollback_target_revision_id:
                raise ProcedureRecoveryError("recovery plan is not the exact recorded target")
            target = self._registry.get_revision(plan.rollback_target_revision_id)
            if (
                target.procedure_id != plan.procedure_id
                or target.procedure_cid != plan.target_procedure_cid
                or target.generation != plan.target_revision_generation
                or target.state is not RegistryLifecycleState.PROMOTED
            ):
                raise ProcedureRecoveryError("recovery plan target identity drifted")

            mutation = self._registry.rollback(
                procedure_id=plan.procedure_id,
                target_revision_id=plan.rollback_target_revision_id,
                authorization=authorization,
                expected_old_revision_id=plan.expected_head_revision_id,
                now_ms=now_ms,
            )
            if not self._verified_result(plan, mutation):
                failure_cid = content_identity(
                    {
                        "schema": ROLLBACK_VERIFICATION_FAILURE_SCHEMA,
                        "plan_cid": plan.content_id,
                        "new_revision_id": mutation.revision.revision_id,
                        "cas_id": mutation.cas.cas_id,
                    }
                )
                containment = self._registry.apply_drift(
                    procedure_id=plan.procedure_id,
                    expected_old_revision_id=mutation.revision.revision_id,
                    cause=RegistryDriftCause.OBSERVED_FAILURE,
                    drift_report_cid=failure_cid,
                    now_ms=now_ms,
                )
                containment_revision_id = containment.revision.revision_id
                raise ProcedureRollbackPostconditionError(
                    "rollback result failed verification and was revoked"
                )
            return mutation
        except ProcedureRegistryError as exc:
            reason_code = self._reason_code(exc)
            self._registry.store.quarantine(
                {
                    "kind": "rollback_failure",
                    "procedure_id": plan.procedure_id,
                    "expected_head_revision_id": plan.expected_head_revision_id,
                    "rollback_target_revision_id": plan.rollback_target_revision_id,
                    "recovery_plan_cid": plan.content_id,
                    "authorization_cid": (
                        authorization.authorization_cid
                        if isinstance(authorization, RegistryAuthorization)
                        else "invalid-authorization"
                    ),
                    "committed_revision_id": (
                        "" if mutation is None else mutation.revision.revision_id
                    ),
                    "containment_revision_id": containment_revision_id,
                    "reason_code": reason_code,
                }
            )
            raise ProcedureRollbackFailure(
                "rollback failed and was quarantined", reason_code=reason_code
            ) from exc

    def _verified_result(
        self, plan: RegistryRecoveryPlan, mutation: RegistryMutation
    ) -> bool:
        if (
            not mutation.cas.accepted
            or mutation.cas.operation is not RegistryOperation.ROLLBACK
            or mutation.revision.state is not RegistryLifecycleState.PROMOTED
            or mutation.revision.procedure_cid != plan.target_procedure_cid
            or mutation.revision.expected_old_revision_id
            != plan.expected_head_revision_id
            or mutation.revision.predecessor_revision_id
            != plan.expected_head_revision_id
            or mutation.revision.rollback_target_revision_id
            != plan.expected_head_revision_id
        ):
            return False
        current = self._registry.get(plan.procedure_id, demote_stale=False)
        if current.revision_id != mutation.revision.revision_id:
            return False
        usable = self._registry.lookup_exact(
            plan.target_procedure_cid,
            bindings=mutation.revision.bindings,
            usable_only=True,
        )
        return usable is not None and usable.revision_id == mutation.revision.revision_id

    @staticmethod
    def _reason_code(exc: ProcedureRegistryError) -> str:
        if isinstance(exc, ProcedureRollbackPostconditionError):
            return "rollback_verification_failed"
        if isinstance(exc, RegistryCASError):
            return "stale_expected_old"
        if isinstance(exc, RegistryAuthorizationError):
            return "authorization_refused"
        if isinstance(exc, RegistryNotFoundError):
            return "rollback_target_missing"
        if isinstance(exc, ProcedureRecoveryError):
            return "recovery_plan_drift"
        return "rollback_registry_failure"


__all__ = [
    "DriftDimension",
    "DriftDisposition",
    "PROCEDURE_DRIFT_MONITOR_REVISION",
    "PROCEDURE_DRIFT_OBSERVATION_SCHEMA",
    "PROCEDURE_RECOVERY_PLANNER_REVISION",
    "PROCEDURE_ROLLBACK_SERVICE_REVISION",
    "ProcedureDriftMonitor",
    "ProcedureDriftObservation",
    "ProcedureDriftReport",
    "ProcedureDriftResult",
    "ProcedureRecoveryError",
    "ProcedureRecoveryPlanner",
    "ProcedureRollbackFailure",
    "ProcedureRollbackPostconditionError",
    "ProcedureRollbackService",
    "REGISTRY_RECOVERY_PLAN_SCHEMA",
    "RecoveryPlan",
    "RegistryRecoveryPlan",
]
