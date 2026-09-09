"""Fail-closed PCPR canonical supervisor-contract freeze.

PCPR-002 freezes the public objective, task, event, ContextPack,
state-machine, and receipt contracts after live supervisor promotion, or
issues an honest non-promotion.  This module is not release authority: it
does not write DuckDB or Quack state and never emits a closed PCPR
release outcome.

A freeze requires ``supervisor_promoted`` live qualification from
PCPR-001, measured normative identities for every public surface, and a
live certification that competing authorities are absent.  Simulated,
hermetic, estimated, and unavailable evidence cannot mint a freeze.
Operator override may authorize campaign continuation; it cannot freeze
contracts or close a release.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from .direct_objective_event_driven_qualification import (
    CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
    qualify_current_head_without_live_campaign,
)
from .source_seal_and_supervisor_baseline import (
    CANONICAL_CONTRACT_CATALOG,
    SUPERVISOR_BASELINE_SURFACES,
)


CONTRACT_FREEZE_INTERFACE: Final = "CanonicalSupervisorContractFreeze@1"
CONTRACT_FREEZE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/canonical-supervisor-contract-freeze@1"
)
CONTRACT_FREEZE_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "canonical-supervisor-contract-freeze-verdict@1"
)

PCPR_002_TASK_ID: Final = "PCPR-002"
PCPR_002_GOAL_ID: Final = "PCPR-G130"
PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_001_GOAL_ID: Final = "PCPR-G120"
PCPR_003_TASK_ID: Final = "PCPR-003"
PCPR_040_TASK_ID: Final = "PCPR-040"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = "proof-carrying-platform-qualification-and-release-v1"

EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "measured",
        "measured_live",
        "measured_hermetic",
        "estimated",
        "simulated",
        "unavailable",
    }
)
LIVE_SATISFYING_KIND: Final = "measured_live"
PLACEHOLDER_SCHEMA: Final = "not_yet_normative"

PROMOTION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "supervisor_promoted",
        "supervisor_non_promoted",
        "rnd_non_promoted",
        "typed_unavailable",
        "typed_blocked",
    }
)
CLOSED_RELEASE_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        "release_candidate_qualified",
        "non_promoted_supervisor_unqualified",
        "non_promoted_import_or_false_success",
        "non_promoted_live_storage_gap",
        "non_promoted_live_compute_gap",
        "non_promoted_solver_gap",
        "non_promoted_packaging_gap",
        "non_promoted_dependency_reproducibility",
        "non_promoted_security_failure",
        "non_promoted_interoperability_gap",
        "non_promoted_reference_workflow_failure",
        "non_promoted_unmeasured",
        "non_promoted_operator_gate_required",
    }
)

PUBLIC_SUPERVISOR_SURFACES: Final[tuple[str, ...]] = SUPERVISOR_BASELINE_SURFACES
SURFACE_CONTRACT_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "objective": "SupervisorObjectiveIntent",
        "task": "TaskStateTransition",
        "event": "SupervisorEvent",
        "contextpack": "SupervisorContextPack",
        "state_machine": "TaskStateTransition",
        "receipt": "ObjectiveMaterializationReceipt",
    }
)

COMPETING_AUTHORITY_PROHIBITIONS: Final[tuple[str, ...]] = (
    "new_supervisor_family",
    "new_planner_family",
    "new_meta_controller",
    "new_task_database",
    "new_event_database",
    "direct_duckdb_writes",
    "direct_quack_writes",
    "self_authorized_promotion",
    "model_completion_authority",
    "markdown_as_live_authority",
)
COMPETING_AUTHORITY_SCOPES: Final[frozenset[str]] = frozenset({"this_task", "repository"})

REQUIRED_FREEZE_SURFACES: Final[tuple[str, ...]] = PUBLIC_SUPERVISOR_SURFACES
REQUIRED_CONTRACT_NAMES: Final[tuple[str, ...]] = tuple(
    item["name"] for item in CANONICAL_CONTRACT_CATALOG
)

# Current-head identities for this isolated implementation worktree.
CURRENT_HEAD_OUTER_COMMIT: Final = "fc16befc27c73b47d1d84a2a905fad5b3d987bb6"
CURRENT_HEAD_OUTER_TREE: Final = "ad536682d2404300da82dfc852a0ba3d78b23ab0"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "Merge commit '975361ed9a7ee70838283bd320bbb8f7ca98cfe8' into "
    "agent/proof-carrying-platform-qualification-and-release-v1"
)
CURRENT_HEAD_OUTER_BRANCH: Final = (
    "implementation/pcpr-002-44ec476ee53d-attempt-1-1788188665"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"
CURRENT_HEAD_FIRST_PARENT: Final = "41dd92e823ecc10be2925f92ef7ce07c494c78ce"
CURRENT_HEAD_LANDED_CANDIDATE: Final = "975361ed9a7ee70838283bd320bbb8f7ca98cfe8"
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "9a9fbfd89ffb32229e90254e0ff40f684446642a"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = "e1dc0f9002237d04f7b212f5c6ec72c3c8702b29"
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_DATASETS_COMMIT: Final = "f49afc579c22856849ca9f739435e5820003384f"
CURRENT_HEAD_DATASETS_TREE: Final = "47118a8e6d1b6b4e7ae04f9a2efda33aadebca1b"
CURRENT_HEAD_KIT_COMMIT: Final = "b6c65ba732733d7e33852713ba18aa3b12235668"
CURRENT_HEAD_KIT_TREE: Final = "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2"

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_source_seal_and_supervisor_baseline.py",
    "test/api/test_agent_supervisor_direct_objective_event_driven_qualification.py",
    "test/api/test_agent_supervisor_canonical_supervisor_contract_freeze.py",
)


class CanonicalSupervisorContractFreezeError(ValueError):
    """Malformed freeze evidence or a forbidden authority claim."""


@dataclass(frozen=True)
class QualificationPrerequisite:
    """PCPR-001 qualification identity consumed by the freeze evaluator."""

    task_id: str
    promotion_status: str
    supervisor_disposition: str
    live_qualification_evidence_kind: str
    verdict_cid: str
    live_campaign_identity: str = ""
    closed_release_outcome: None = None
    release_claim: bool = False
    completion_authoritative: bool = False
    missed_live_cohort_count: int = 0
    missed_target_count: int = 0
    reason: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal_id": PCPR_001_GOAL_ID,
            "promotion_status": self.promotion_status,
            "supervisor_disposition": self.supervisor_disposition,
            "live_qualification_evidence_kind": self.live_qualification_evidence_kind,
            "verdict_cid": self.verdict_cid,
            "live_campaign_identity": self.live_campaign_identity,
            "closed_release_outcome": self.closed_release_outcome,
            "release_claim": self.release_claim,
            "completion_authoritative": self.completion_authoritative,
            "missed_live_cohort_count": self.missed_live_cohort_count,
            "missed_target_count": self.missed_target_count,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SurfaceObservation:
    """One public supervisor surface with an explicit evidence kind."""

    surface: str
    contract_name: str
    schema: str
    authority: str
    evidence_kind: str
    freeze_claimed: bool = False
    reason: str = ""
    normative_task: str = PCPR_040_TASK_ID
    freeze_task: str = PCPR_002_TASK_ID

    def to_mapping(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "contract_name": self.contract_name,
            "schema": self.schema,
            "authority": self.authority,
            "evidence_kind": self.evidence_kind,
            "freeze_claimed": self.freeze_claimed,
            "reason": self.reason,
            "normative_task": self.normative_task,
            "freeze_task": self.freeze_task,
        }


@dataclass(frozen=True)
class ContractObservation:
    """One shared-catalog contract.  Freeze is not PCPR-040 stabilization."""

    name: str
    schema: str
    authority: str
    evidence_kind: str
    freeze_claimed: bool = False
    reason: str = ""
    normative_task: str = PCPR_040_TASK_ID
    freeze_task: str = PCPR_002_TASK_ID

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema": self.schema,
            "authority": self.authority,
            "evidence_kind": self.evidence_kind,
            "freeze_claimed": self.freeze_claimed,
            "reason": self.reason,
            "normative_task": self.normative_task,
            "freeze_task": self.freeze_task,
        }


@dataclass(frozen=True)
class CompetingAuthorityObservation:
    """Presence or typed unavailability of one prohibited competing authority."""

    name: str
    present: bool | None
    evidence_kind: str
    scope: str
    reason: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "present": self.present,
            "evidence_kind": self.evidence_kind,
            "scope": self.scope,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OperatorOverride:
    """Bounded operator approval.  Cannot freeze or close a release."""

    present: bool
    evidence_kind: str
    reason: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "evidence_kind": self.evidence_kind,
            "reason": self.reason,
            "can_freeze_contracts": False,
            "can_close_release": False,
        }


@dataclass(frozen=True)
class EvaluatedSurface:
    """Fail-closed evaluation of one public surface."""

    surface: str
    contract_name: str
    schema: str
    authority: str
    evidence_kind: str
    frozen: bool
    freeze_eligible: bool
    blockers: tuple[str, ...]
    reason: str
    normative_task: str
    freeze_task: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "contract_name": self.contract_name,
            "schema": self.schema,
            "authority": self.authority,
            "evidence_kind": self.evidence_kind,
            "frozen": self.frozen,
            "freeze_eligible": self.freeze_eligible,
            "blockers": list(self.blockers),
            "reason": self.reason,
            "normative_task": self.normative_task,
            "freeze_task": self.freeze_task,
        }


@dataclass(frozen=True)
class FreezeVerdict:
    """Fail-closed PCPR-002 freeze-or-non-promotion decision."""

    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    competing_authorities_prohibited_by_freeze: bool
    continuation_requires_bounded_operator_approval: bool
    duckdb_or_quack_state_written: bool
    qualification: QualificationPrerequisite
    surfaces: tuple[EvaluatedSurface, ...]
    contracts: tuple[ContractObservation, ...]
    competing_authorities: tuple[CompetingAuthorityObservation, ...]
    operator_override: OperatorOverride
    freeze_refused_reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    verdict_cid: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "promotion_status": self.promotion_status,
            "supervisor_disposition": self.supervisor_disposition,
            "closed_release_outcome": self.closed_release_outcome,
            "release_claim": self.release_claim,
            "completion_authoritative": self.completion_authoritative,
            "contracts_frozen": self.contracts_frozen,
            "competing_authorities_prohibited_by_freeze": (
                self.competing_authorities_prohibited_by_freeze
            ),
            "continuation_requires_bounded_operator_approval": (
                self.continuation_requires_bounded_operator_approval
            ),
            "duckdb_or_quack_state_written": self.duckdb_or_quack_state_written,
            "qualification": self.qualification.to_mapping(),
            "surfaces": [item.to_mapping() for item in self.surfaces],
            "contracts": [item.to_mapping() for item in self.contracts],
            "competing_authorities": [
                item.to_mapping() for item in self.competing_authorities
            ],
            "operator_override": self.operator_override.to_mapping(),
            "freeze_refused_reasons": list(self.freeze_refused_reasons),
            "blockers": list(self.blockers),
            "verdict_cid": self.verdict_cid,
        }


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanonicalSupervisorContractFreezeError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise CanonicalSupervisorContractFreezeError(f"{name} is not an admitted evidence kind")
    return kind


def _git_object_id(value: Any, name: str) -> str:
    text = _text(value, name).lower()
    if len(text) != 40 or any(char not in "0123456789abcdef" for char in text):
        raise CanonicalSupervisorContractFreezeError(
            f"{name} must be a lowercase 40-character git object id"
        )
    return text


def _require_ancestor(flag: Any, name: str) -> bool:
    if flag is not True:
        raise CanonicalSupervisorContractFreezeError(
            f"{name} must be true; a non-ancestor origin/main cannot bind current-head evidence"
        )
    return True


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise CanonicalSupervisorContractFreezeError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _require_unique(ids: Sequence[str], population: Sequence[str], name: str) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in ids:
        if item in seen:
            raise CanonicalSupervisorContractFreezeError(f"duplicate {name}: {item}")
        seen.add(item)
        ordered.append(item)
    extra = [item for item in ordered if item not in population]
    if extra:
        raise CanonicalSupervisorContractFreezeError(f"unknown {name}: {extra[0]}")
    missing = [item for item in population if item not in seen]
    if missing:
        raise CanonicalSupervisorContractFreezeError(f"missing {name}: {missing[0]}")
    return tuple(ordered)


def _schema_is_placeholder(schema: str) -> bool:
    return schema.strip() in {"", PLACEHOLDER_SCHEMA}


def _normalize_qualification(
    item: QualificationPrerequisite,
) -> QualificationPrerequisite:
    task_id = _text(item.task_id, "qualification.task_id")
    if task_id != PCPR_001_TASK_ID:
        raise CanonicalSupervisorContractFreezeError(
            "qualification.task_id must be PCPR-001"
        )
    promotion = _text(item.promotion_status, "qualification.promotion_status")
    if promotion not in PROMOTION_STATUSES:
        raise CanonicalSupervisorContractFreezeError(
            "qualification.promotion_status is not an admitted Phase-0 status"
        )
    _reject_closed_release_value(promotion, "qualification.promotion_status")
    if item.closed_release_outcome is not None:
        raise CanonicalSupervisorContractFreezeError(
            "qualification.closed_release_outcome must be null"
        )
    if item.release_claim:
        raise CanonicalSupervisorContractFreezeError(
            "qualification must not claim a PCPR release"
        )
    if item.completion_authoritative:
        raise CanonicalSupervisorContractFreezeError(
            "qualification completion is not authoritative"
        )
    live_kind = _kind(
        item.live_qualification_evidence_kind,
        "qualification.live_qualification_evidence_kind",
    )
    if live_kind == "simulated" and promotion == "supervisor_promoted":
        raise CanonicalSupervisorContractFreezeError(
            "simulated qualification cannot promote the supervisor"
        )
    disposition = _text(
        item.supervisor_disposition, "qualification.supervisor_disposition"
    )
    if disposition not in {"supervisor_promoted", "supervisor_non_promoted"}:
        raise CanonicalSupervisorContractFreezeError(
            "qualification.supervisor_disposition is not admitted"
        )
    missed_live = item.missed_live_cohort_count
    missed_targets = item.missed_target_count
    if isinstance(missed_live, bool) or not isinstance(missed_live, int) or missed_live < 0:
        raise CanonicalSupervisorContractFreezeError(
            "missed_live_cohort_count must be a non-negative integer"
        )
    if (
        isinstance(missed_targets, bool)
        or not isinstance(missed_targets, int)
        or missed_targets < 0
    ):
        raise CanonicalSupervisorContractFreezeError(
            "missed_target_count must be a non-negative integer"
        )
    campaign = item.live_campaign_identity
    if campaign is None or not isinstance(campaign, str):
        raise CanonicalSupervisorContractFreezeError(
            "live_campaign_identity must be a string"
        )
    if live_kind == LIVE_SATISFYING_KIND and promotion == "supervisor_promoted":
        if not campaign.strip():
            raise CanonicalSupervisorContractFreezeError(
                "measured_live supervisor promotion requires live_campaign_identity"
            )
        if missed_live != 0 or missed_targets != 0:
            raise CanonicalSupervisorContractFreezeError(
                "measured_live supervisor promotion cannot retain missed cohort or targets"
            )
    return QualificationPrerequisite(
        task_id=task_id,
        promotion_status=promotion,
        supervisor_disposition=disposition,
        live_qualification_evidence_kind=live_kind,
        verdict_cid=_text(item.verdict_cid, "qualification.verdict_cid"),
        live_campaign_identity=campaign.strip(),
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        missed_live_cohort_count=missed_live,
        missed_target_count=missed_targets,
        reason=str(item.reason or ""),
    )


def _normalize_surfaces(
    records: Sequence[SurfaceObservation],
) -> tuple[SurfaceObservation, ...]:
    normalized: list[SurfaceObservation] = []
    for record in records:
        surface = _text(record.surface, "surface")
        kind = _kind(record.evidence_kind, f"{surface}.evidence_kind")
        schema = _text(record.schema, f"{surface}.schema")
        if kind == "simulated" and record.freeze_claimed:
            raise CanonicalSupervisorContractFreezeError(
                "simulated freeze claims cannot mint a contract freeze"
            )
        expected_contract = SURFACE_CONTRACT_MAP.get(surface)
        contract_name = _text(record.contract_name, f"{surface}.contract_name")
        if expected_contract is not None and contract_name != expected_contract:
            raise CanonicalSupervisorContractFreezeError(
                f"{surface} must bind {expected_contract}"
            )
        normalized.append(
            SurfaceObservation(
                surface=surface,
                contract_name=contract_name,
                schema=schema,
                authority=_text(record.authority, f"{surface}.authority"),
                evidence_kind=kind,
                freeze_claimed=bool(record.freeze_claimed),
                reason=str(record.reason or ""),
                normative_task=_text(record.normative_task, f"{surface}.normative_task"),
                freeze_task=_text(record.freeze_task, f"{surface}.freeze_task"),
            )
        )
    _require_unique(
        [item.surface for item in normalized], REQUIRED_FREEZE_SURFACES, "surface"
    )
    by_id = {item.surface: item for item in normalized}
    return tuple(by_id[name] for name in REQUIRED_FREEZE_SURFACES)


def _normalize_contracts(
    records: Sequence[ContractObservation],
) -> tuple[ContractObservation, ...]:
    normalized: list[ContractObservation] = []
    catalog = {item["name"]: item for item in CANONICAL_CONTRACT_CATALOG}
    for record in records:
        name = _text(record.name, "contract.name")
        kind = _kind(record.evidence_kind, f"{name}.evidence_kind")
        if kind == "simulated" and record.freeze_claimed:
            raise CanonicalSupervisorContractFreezeError(
                "simulated freeze claims cannot mint a contract freeze"
            )
        expected = catalog.get(name)
        if expected is None:
            raise CanonicalSupervisorContractFreezeError(f"unknown contract: {name}")
        schema = _text(record.schema, f"{name}.schema")
        authority = _text(record.authority, f"{name}.authority")
        if authority != expected["authority"]:
            raise CanonicalSupervisorContractFreezeError(
                f"{name} authority must remain {expected['authority']}"
            )
        normalized.append(
            ContractObservation(
                name=name,
                schema=schema,
                authority=authority,
                evidence_kind=kind,
                freeze_claimed=bool(record.freeze_claimed),
                reason=str(record.reason or ""),
                normative_task=_text(record.normative_task, f"{name}.normative_task"),
                freeze_task=_text(record.freeze_task, f"{name}.freeze_task"),
            )
        )
    _require_unique(
        [item.name for item in normalized], REQUIRED_CONTRACT_NAMES, "contract"
    )
    by_id = {item.name: item for item in normalized}
    return tuple(by_id[name] for name in REQUIRED_CONTRACT_NAMES)


def _normalize_competing(
    records: Sequence[CompetingAuthorityObservation],
) -> tuple[CompetingAuthorityObservation, ...]:
    normalized: list[CompetingAuthorityObservation] = []
    for record in records:
        name = _text(record.name, "competing_authority.name")
        kind = _kind(record.evidence_kind, f"{name}.evidence_kind")
        scope = _text(record.scope, f"{name}.scope")
        if scope not in COMPETING_AUTHORITY_SCOPES:
            raise CanonicalSupervisorContractFreezeError(
                f"{name}.scope is not an admitted competing-authority scope"
            )
        present = record.present
        if present is not None and not isinstance(present, bool):
            raise CanonicalSupervisorContractFreezeError(
                f"{name}.present must be a boolean or null"
            )
        if kind == "simulated" and present is False:
            raise CanonicalSupervisorContractFreezeError(
                "simulated absence cannot certify that competing authorities are prohibited"
            )
        if kind == "unavailable" and present is not None:
            raise CanonicalSupervisorContractFreezeError(
                f"{name} unavailable competing-authority evidence must not record a boolean"
            )
        if kind == LIVE_SATISFYING_KIND and present is None:
            raise CanonicalSupervisorContractFreezeError(
                f"{name} measured_live competing-authority evidence requires present"
            )
        normalized.append(
            CompetingAuthorityObservation(
                name=name,
                present=present,
                evidence_kind=kind,
                scope=scope,
                reason=str(record.reason or ""),
            )
        )
    names = [item.name for item in normalized]
    seen: set[str] = set()
    for name in names:
        if name not in COMPETING_AUTHORITY_PROHIBITIONS:
            raise CanonicalSupervisorContractFreezeError(
                f"unknown competing authority: {name}"
            )
        key = name
        if key in seen:
            raise CanonicalSupervisorContractFreezeError(
                f"duplicate competing authority: {name}"
            )
        seen.add(key)
    missing = [name for name in COMPETING_AUTHORITY_PROHIBITIONS if name not in seen]
    if missing:
        raise CanonicalSupervisorContractFreezeError(
            f"missing competing authority: {missing[0]}"
        )
    by_id = {item.name: item for item in normalized}
    return tuple(by_id[name] for name in COMPETING_AUTHORITY_PROHIBITIONS)


def _normalize_override(item: OperatorOverride) -> OperatorOverride:
    kind = _kind(item.evidence_kind, "operator_override.evidence_kind")
    if kind == "simulated" and item.present:
        raise CanonicalSupervisorContractFreezeError(
            "simulated operator override cannot authorize freeze or continuation"
        )
    return OperatorOverride(
        present=bool(item.present),
        evidence_kind=kind,
        reason=str(item.reason or ""),
    )


def _surface_freeze_eligible(record: SurfaceObservation) -> tuple[bool, tuple[str, ...]]:
    blockers: list[str] = []
    if record.evidence_kind != LIVE_SATISFYING_KIND:
        blockers.append(f"{record.surface}:not_measured_live")
    if _schema_is_placeholder(record.schema):
        blockers.append(f"{record.surface}:not_yet_normative")
    return (not blockers, tuple(blockers))


def freeze_canonical_supervisor_contracts(
    *,
    qualification: QualificationPrerequisite,
    surfaces: Sequence[SurfaceObservation],
    contracts: Sequence[ContractObservation],
    competing_authorities: Sequence[CompetingAuthorityObservation],
    operator_override: OperatorOverride,
    duckdb_or_quack_state_written: bool = False,
) -> FreezeVerdict:
    """Evaluate PCPR-002.  Freeze is fail-closed; non-promotion is honest."""

    if duckdb_or_quack_state_written:
        raise CanonicalSupervisorContractFreezeError(
            "contract freeze must not write DuckDB or Quack state"
        )

    normalized_qualification = _normalize_qualification(qualification)
    normalized_surfaces = _normalize_surfaces(surfaces)
    normalized_contracts = _normalize_contracts(contracts)
    normalized_competing = _normalize_competing(competing_authorities)
    normalized_override = _normalize_override(operator_override)

    freeze_refused: list[str] = []
    blockers: list[str] = []

    qualification_promoted = (
        normalized_qualification.promotion_status == "supervisor_promoted"
        and normalized_qualification.supervisor_disposition == "supervisor_promoted"
        and normalized_qualification.live_qualification_evidence_kind == LIVE_SATISFYING_KIND
    )
    if not qualification_promoted:
        freeze_refused.append("qualification_not_supervisor_promoted")
        if normalized_qualification.live_qualification_evidence_kind != LIVE_SATISFYING_KIND:
            freeze_refused.append("live_qualification_unavailable")

    evaluated_surfaces: list[EvaluatedSurface] = []
    for record in normalized_surfaces:
        eligible, surface_blockers = _surface_freeze_eligible(record)
        if record.freeze_claimed and not qualification_promoted:
            blockers.append(f"{record.surface}:claimed_freeze_without_qualification")
        if record.freeze_claimed and not eligible:
            blockers.extend(surface_blockers)
        if not eligible:
            freeze_refused.extend(surface_blockers)
        evaluated_surfaces.append(
            EvaluatedSurface(
                surface=record.surface,
                contract_name=record.contract_name,
                schema=record.schema,
                authority=record.authority,
                evidence_kind=record.evidence_kind,
                frozen=False,
                freeze_eligible=eligible,
                blockers=surface_blockers,
                reason=record.reason,
                normative_task=record.normative_task,
                freeze_task=record.freeze_task,
            )
        )
    if any(not item.freeze_eligible for item in evaluated_surfaces):
        freeze_refused.append("public_surfaces_not_all_normative")

    repository_inventory = [
        item for item in normalized_competing if item.scope == "repository"
    ]
    live_absent = True
    if len(repository_inventory) != len(COMPETING_AUTHORITY_PROHIBITIONS):
        live_absent = False
        freeze_refused.append("competing_authority_inventory_incomplete")
    for item in repository_inventory:
        if item.present is True:
            live_absent = False
            blockers.append(f"{item.name}:present")
            freeze_refused.append("competing_authority_present")
        if item.evidence_kind != LIVE_SATISFYING_KIND or item.present is not False:
            live_absent = False
            if item.evidence_kind == "unavailable":
                freeze_refused.append("competing_authority_inventory_unavailable")
            elif item.evidence_kind != LIVE_SATISFYING_KIND:
                freeze_refused.append("competing_authority_not_measured_live")
    this_task_created = any(
        item.scope == "this_task" and item.present is True for item in normalized_competing
    )
    if this_task_created:
        live_absent = False
        blockers.append("this_task_created_competing_authority")
        freeze_refused.append("this_task_created_competing_authority")

    if normalized_override.present:
        freeze_refused.append("operator_override_cannot_freeze")

    unique_refused = tuple(dict.fromkeys(freeze_refused))
    unique_blockers = tuple(dict.fromkeys(blockers))
    can_freeze = (
        qualification_promoted
        and all(item.freeze_eligible for item in evaluated_surfaces)
        and live_absent
        and not normalized_override.present
        and not unique_blockers
    )

    if can_freeze:
        promotion_status = "supervisor_promoted"
        supervisor_disposition = "supervisor_promoted"
        contracts_frozen = True
        unique_refused = ()
        frozen_surfaces = tuple(
            EvaluatedSurface(
                surface=item.surface,
                contract_name=item.contract_name,
                schema=item.schema,
                authority=item.authority,
                evidence_kind=item.evidence_kind,
                frozen=True,
                freeze_eligible=True,
                blockers=(),
                reason=item.reason,
                normative_task=item.normative_task,
                freeze_task=item.freeze_task,
            )
            for item in evaluated_surfaces
        )
    else:
        contracts_frozen = False
        frozen_surfaces = tuple(evaluated_surfaces)
        if unique_blockers and (
            "competing_authority_present" in unique_refused
            or "this_task_created_competing_authority" in unique_refused
            or any("claimed_freeze_without_qualification" in item for item in unique_blockers)
        ):
            promotion_status = "typed_blocked"
            supervisor_disposition = "supervisor_non_promoted"
        elif normalized_qualification.promotion_status == "typed_blocked":
            promotion_status = "typed_blocked"
            supervisor_disposition = "supervisor_non_promoted"
        elif normalized_qualification.promotion_status == "typed_unavailable":
            promotion_status = "typed_unavailable"
            supervisor_disposition = "supervisor_non_promoted"
        else:
            promotion_status = "rnd_non_promoted"
            supervisor_disposition = "supervisor_non_promoted"

    if promotion_status not in PROMOTION_STATUSES:
        raise CanonicalSupervisorContractFreezeError(
            "internal promotion status is not admitted"
        )
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise CanonicalSupervisorContractFreezeError(
            "contract freeze must not mint a closed release outcome"
        )
    if contracts_frozen and promotion_status != "supervisor_promoted":
        raise CanonicalSupervisorContractFreezeError(
            "internal freeze without supervisor promotion is forbidden"
        )

    continuation_required = not contracts_frozen
    payload = {
        "schema": CONTRACT_FREEZE_VERDICT_SCHEMA,
        "interface": CONTRACT_FREEZE_INTERFACE,
        "task_id": PCPR_002_TASK_ID,
        "goal_id": PCPR_002_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": contracts_frozen,
        "competing_authorities_prohibited_by_freeze": contracts_frozen,
        "continuation_requires_bounded_operator_approval": continuation_required,
        "duckdb_or_quack_state_written": False,
        "qualification": normalized_qualification.to_mapping(),
        "surfaces": [item.to_mapping() for item in frozen_surfaces],
        "contracts": [item.to_mapping() for item in normalized_contracts],
        "competing_authorities": [item.to_mapping() for item in normalized_competing],
        "operator_override": normalized_override.to_mapping(),
        "freeze_refused_reasons": list(unique_refused),
        "blockers": list(unique_blockers),
    }
    verdict_cid = content_identity(payload)
    return FreezeVerdict(
        schema=CONTRACT_FREEZE_VERDICT_SCHEMA,
        interface=CONTRACT_FREEZE_INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition=supervisor_disposition,
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=contracts_frozen,
        competing_authorities_prohibited_by_freeze=contracts_frozen,
        continuation_requires_bounded_operator_approval=continuation_required,
        duckdb_or_quack_state_written=False,
        qualification=normalized_qualification,
        surfaces=frozen_surfaces,
        contracts=normalized_contracts,
        competing_authorities=normalized_competing,
        operator_override=normalized_override,
        freeze_refused_reasons=unique_refused,
        blockers=unique_blockers,
        verdict_cid=verdict_cid,
    )


def current_head_qualification_prerequisite() -> QualificationPrerequisite:
    """Bind the ordinary missing-live PCPR-001 verdict as freeze input."""

    verdict = qualify_current_head_without_live_campaign()
    return QualificationPrerequisite(
        task_id=PCPR_001_TASK_ID,
        promotion_status=verdict.promotion_status,
        supervisor_disposition=verdict.supervisor_disposition,
        live_qualification_evidence_kind="unavailable",
        verdict_cid=verdict.verdict_cid,
        live_campaign_identity="",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        missed_live_cohort_count=len(verdict.missed_live_cohort),
        missed_target_count=len(verdict.missed_targets),
        reason=(
            "PCPR-001 current-head qualification is rnd_non_promoted because the "
            "required live cohort and efficiency targets remain typed unavailable."
        ),
    )


def current_head_surface_observations() -> tuple[SurfaceObservation, ...]:
    """Catalog identities for the six public surfaces.  Not a freeze."""

    catalog = {item["name"]: item for item in CANONICAL_CONTRACT_CATALOG}
    observations: list[SurfaceObservation] = []
    for surface in REQUIRED_FREEZE_SURFACES:
        contract_name = SURFACE_CONTRACT_MAP[surface]
        item = catalog[contract_name]
        observations.append(
            SurfaceObservation(
                surface=surface,
                contract_name=contract_name,
                schema=item["schema"],
                authority=item["authority"],
                evidence_kind="measured",
                freeze_claimed=False,
                reason=(
                    "Observed from the PCPR-000 contract catalog. Schema is a "
                    "baseline identity, not a freeze, and not PCPR-040 stabilization."
                ),
            )
        )
    return tuple(observations)


def current_head_contract_observations() -> tuple[ContractObservation, ...]:
    """Fourteen shared contracts remain baseline_recorded_not_frozen."""

    return tuple(
        ContractObservation(
            name=item["name"],
            schema=item["schema"],
            authority=item["authority"],
            evidence_kind="measured",
            freeze_claimed=False,
            reason="Observed from the PCPR-000 catalog; freeze_task remains PCPR-002.",
            normative_task=item["normative_task"],
            freeze_task=item["freeze_task"],
        )
        for item in CANONICAL_CONTRACT_CATALOG
    )


def current_head_competing_authorities() -> tuple[CompetingAuthorityObservation, ...]:
    """This task created none; repository-wide inventory remains PCPR-003."""

    return tuple(
        CompetingAuthorityObservation(
            name=name,
            present=None,
            evidence_kind="unavailable",
            scope="repository",
            reason=(
                f"Repository-wide competing-authority inventory is {PCPR_003_TASK_ID}. "
                "This worker did not treat an empty local change as a live prohibition."
            ),
        )
        for name in COMPETING_AUTHORITY_PROHIBITIONS
    )


def current_head_operator_override() -> OperatorOverride:
    """No bounded operator freeze or continuation approval is present."""

    return OperatorOverride(
        present=False,
        evidence_kind="measured",
        reason=(
            "No bounded operator approval to freeze contracts or close a release "
            "was observed. Continuation after supervisor_non_promoted still requires "
            "that approval; this worker did not mint it."
        ),
    )


def qualify_current_head_without_supervisor_promotion() -> FreezeVerdict:
    """Ordinary current-head PCPR-002 evaluation: honest non-promotion."""

    return freeze_canonical_supervisor_contracts(
        qualification=current_head_qualification_prerequisite(),
        surfaces=current_head_surface_observations(),
        contracts=current_head_contract_observations(),
        competing_authorities=current_head_competing_authorities(),
        operator_override=current_head_operator_override(),
    )


# Pinned identity of the ordinary missing-live non-promotion verdict.  Drift
# means the default payload changed and the outer receipt must be regenerated.
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeerawqleelwwzlntopi6xiuhuqzg2l7c555nbmppi7nmf2w2adpxg6qa"
)


def pcpr_002_receipt_promotion(verdict: FreezeVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields.  Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise CanonicalSupervisorContractFreezeError(
            "contract freeze must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise CanonicalSupervisorContractFreezeError(
            "contract freeze must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise CanonicalSupervisorContractFreezeError(
            "contract-freeze completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise CanonicalSupervisorContractFreezeError(
            "contract freeze must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise CanonicalSupervisorContractFreezeError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise CanonicalSupervisorContractFreezeError(
            "promotion_status is not an admitted PCPR-002 status"
        )
    return {
        "schema": CONTRACT_FREEZE_VERDICT_SCHEMA,
        "interface": CONTRACT_FREEZE_INTERFACE,
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": verdict.promotion_status,
        "supervisor_disposition": verdict.supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": verdict.contracts_frozen,
        "competing_authorities_prohibited_by_freeze": (
            verdict.competing_authorities_prohibited_by_freeze
        ),
        "continuation_requires_bounded_operator_approval": (
            verdict.continuation_requires_bounded_operator_approval
        ),
        "duckdb_or_quack_state_written": False,
        "freeze_refused_reason_count": len(verdict.freeze_refused_reasons),
        "blocker_count": len(verdict.blockers),
        "blockers": list(verdict.blockers),
        "evidence_kind": "measured",
    }


def current_head_pcpr_002_receipt_promotion() -> dict[str, Any]:
    """Fail-closed promotion section for the ordinary missing-live case."""

    return pcpr_002_receipt_promotion(qualify_current_head_without_supervisor_promotion())


def pcpr_002_receipt_qualification(verdict: FreezeVerdict) -> dict[str, Any]:
    """Outer-receipt PCPR-001 prerequisite section."""

    payload = verdict.qualification.to_mapping()
    payload["evidence_kind"] = "measured"
    return payload


def pcpr_002_receipt_surfaces(verdict: FreezeVerdict) -> dict[str, Any]:
    """Outer-receipt public-surface freeze section."""

    return {
        "frozen": verdict.contracts_frozen,
        "required_surfaces": list(REQUIRED_FREEZE_SURFACES),
        "catalog": [item.to_mapping() for item in verdict.surfaces],
        "all_surfaces_freeze_eligible": all(item.freeze_eligible for item in verdict.surfaces),
        "evidence_kind": "measured",
    }


def pcpr_002_receipt_contracts(verdict: FreezeVerdict) -> dict[str, Any]:
    """Outer-receipt shared-contract catalog.  Stabilization remains PCPR-040."""

    return {
        "frozen": verdict.contracts_frozen,
        "freeze_task": PCPR_002_TASK_ID,
        "normative_stabilization_task": PCPR_040_TASK_ID,
        "catalog": [item.to_mapping() for item in verdict.contracts],
        "evidence_kind": "measured",
    }


def pcpr_002_receipt_competing_authorities(verdict: FreezeVerdict) -> dict[str, Any]:
    """Outer-receipt competing-authority section.  Inventory remains PCPR-003."""

    return {
        "prohibited_by_freeze": verdict.competing_authorities_prohibited_by_freeze,
        "inventory_task": PCPR_003_TASK_ID,
        "this_task_created_competing_authority": any(
            item.scope == "this_task" and item.present is True
            for item in verdict.competing_authorities
        ),
        "repository_live_inventory": False,
        "observations": [item.to_mapping() for item in verdict.competing_authorities],
        "evidence_kind": "measured",
    }


def pcpr_002_receipt_negative_results() -> dict[str, Any]:
    """Fixed negative results for the PCPR-002 R&D freeze."""

    return {
        "simulated_freeze_cannot_promote": True,
        "hermetic_pass_cannot_freeze": True,
        "non_promoted_qualification_cannot_freeze": True,
        "operator_override_cannot_freeze": True,
        "placeholder_schema_cannot_freeze": True,
        "unavailable_competing_authority_inventory_cannot_freeze": True,
        "closed_release_outcome_not_emitted": True,
        "direct_database_bypass_not_used": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_002_receipt_sections() -> dict[str, Any]:
    """Promotion, qualification, surface, contract, and negative sections."""

    verdict = qualify_current_head_without_supervisor_promotion()
    promotion = pcpr_002_receipt_promotion(verdict)
    return {
        "qualification_verdict": promotion,
        "qualification_prerequisite": pcpr_002_receipt_qualification(verdict),
        "public_surfaces": pcpr_002_receipt_surfaces(verdict),
        "shared_contracts": pcpr_002_receipt_contracts(verdict),
        "competing_authorities": pcpr_002_receipt_competing_authorities(verdict),
        "operator_override": verdict.operator_override.to_mapping(),
        "freeze_refused_reasons": list(verdict.freeze_refused_reasons),
        "negative_results": pcpr_002_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
    }


def pcpr_002_current_tree_binding(
    *,
    outer_commit: str,
    outer_tree: str,
    outer_subject: str,
    origin_main: str,
    origin_main_is_ancestor: bool,
    accelerator_pre_change_commit: str,
    accelerator_pre_change_tree: str,
    accelerator_gitlink: str,
    accelerator_origin_main: str,
    accelerator_origin_main_is_ancestor: bool,
    datasets_commit: str,
    datasets_tree: str,
    datasets_gitlink: str,
    kit_commit: str,
    kit_tree: str,
    kit_gitlink: str,
) -> dict[str, Any]:
    """Measured current-tree identities for a PCPR-002 outer receipt."""

    outer = _git_object_id(outer_commit, "outer_commit")
    tree = _git_object_id(outer_tree, "outer_tree")
    subject = _text(outer_subject, "outer_subject")
    origin = _git_object_id(origin_main, "origin_main")
    _require_ancestor(origin_main_is_ancestor, "origin_main_is_ancestor")
    accel = _git_object_id(accelerator_pre_change_commit, "accelerator_pre_change_commit")
    accel_tree = _git_object_id(accelerator_pre_change_tree, "accelerator_pre_change_tree")
    accel_link = _git_object_id(accelerator_gitlink, "accelerator_gitlink")
    accel_origin = _git_object_id(accelerator_origin_main, "accelerator_origin_main")
    _require_ancestor(
        accelerator_origin_main_is_ancestor, "accelerator_origin_main_is_ancestor"
    )
    if accel != accel_link:
        raise CanonicalSupervisorContractFreezeError(
            "accelerator_pre_change_commit must equal accelerator_gitlink"
        )
    datasets = _git_object_id(datasets_commit, "datasets_commit")
    datasets_tree_id = _git_object_id(datasets_tree, "datasets_tree")
    datasets_link = _git_object_id(datasets_gitlink, "datasets_gitlink")
    if datasets != datasets_link:
        raise CanonicalSupervisorContractFreezeError(
            "datasets_commit must equal datasets_gitlink"
        )
    kit = _git_object_id(kit_commit, "kit_commit")
    kit_tree_id = _git_object_id(kit_tree, "kit_tree")
    kit_link = _git_object_id(kit_gitlink, "kit_gitlink")
    if kit != kit_link:
        raise CanonicalSupervisorContractFreezeError("kit_commit must equal kit_gitlink")
    _reject_closed_release_value(subject, "outer_subject")
    return {
        "outer_repository": "endomorphosis/lift_coding",
        "owning_repository_for_receipts": "ipfs_accelerate_py",
        "outer_commit": outer,
        "outer_tree": tree,
        "outer_subject": subject,
        "origin_main": origin,
        "origin_main_is_ancestor": True,
        "accelerator_pre_change_commit": accel,
        "accelerator_pre_change_tree": accel_tree,
        "accelerator_gitlink": accel_link,
        "accelerator_origin_main": accel_origin,
        "accelerator_origin_main_is_ancestor": True,
        "accelerator_post_change_commit": "pending nested commit after admission",
        "accelerator_post_change_tree": (
            "dirty-worktree; exact CID after accepted nested commit"
        ),
        "datasets_commit": datasets,
        "datasets_tree": datasets_tree_id,
        "datasets_gitlink": datasets_link,
        "kit_commit": kit,
        "kit_tree": kit_tree_id,
        "kit_gitlink": kit_link,
        "evidence_kind": "measured",
    }


def current_head_pcpr_002_current_tree_binding() -> dict[str, Any]:
    """Measured current-tree binding for this isolated worktree."""

    return pcpr_002_current_tree_binding(
        outer_commit=CURRENT_HEAD_OUTER_COMMIT,
        outer_tree=CURRENT_HEAD_OUTER_TREE,
        outer_subject=CURRENT_HEAD_OUTER_SUBJECT,
        origin_main=CURRENT_HEAD_ORIGIN_MAIN,
        origin_main_is_ancestor=True,
        accelerator_pre_change_commit=CURRENT_HEAD_ACCELERATOR_COMMIT,
        accelerator_pre_change_tree=CURRENT_HEAD_ACCELERATOR_TREE,
        accelerator_gitlink=CURRENT_HEAD_ACCELERATOR_COMMIT,
        accelerator_origin_main=CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN,
        accelerator_origin_main_is_ancestor=True,
        datasets_commit=CURRENT_HEAD_DATASETS_COMMIT,
        datasets_tree=CURRENT_HEAD_DATASETS_TREE,
        datasets_gitlink=CURRENT_HEAD_DATASETS_COMMIT,
        kit_commit=CURRENT_HEAD_KIT_COMMIT,
        kit_tree=CURRENT_HEAD_KIT_TREE,
        kit_gitlink=CURRENT_HEAD_KIT_COMMIT,
    )


def validate_pcpr_002_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-002 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise CanonicalSupervisorContractFreezeError("outer receipt must be a mapping")
    if payload.get("task_id") != PCPR_002_TASK_ID:
        raise CanonicalSupervisorContractFreezeError("outer receipt task_id must be PCPR-002")
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise CanonicalSupervisorContractFreezeError("outer receipt must not claim a release")
    if payload.get("completion_authoritative") is True:
        raise CanonicalSupervisorContractFreezeError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise CanonicalSupervisorContractFreezeError(
            "qualification_verdict must be a mapping"
        )
    _reject_closed_release_value(
        verdict_section.get("promotion_status"),
        "qualification_verdict.promotion_status",
    )
    if verdict_section.get("closed_release_outcome") is not None:
        raise CanonicalSupervisorContractFreezeError(
            "qualification_verdict.closed_release_outcome must be null"
        )
    if verdict_section.get("release_claim") is True:
        raise CanonicalSupervisorContractFreezeError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise CanonicalSupervisorContractFreezeError(
            "contract freeze must not write DuckDB or Quack state"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise CanonicalSupervisorContractFreezeError(
            "qualification_verdict.promotion_status is not an admitted PCPR-002 status"
        )

    contracts_frozen = verdict_section.get("contracts_frozen")
    if contracts_frozen is True and promotion_status != "supervisor_promoted":
        raise CanonicalSupervisorContractFreezeError(
            "contracts cannot freeze without supervisor_promoted"
        )

    prerequisite = payload.get("qualification_prerequisite")
    if isinstance(prerequisite, Mapping):
        if prerequisite.get("task_id") not in {None, PCPR_001_TASK_ID}:
            raise CanonicalSupervisorContractFreezeError(
                "qualification_prerequisite.task_id must be PCPR-001"
            )
        _reject_closed_release_value(
            prerequisite.get("promotion_status"),
            "qualification_prerequisite.promotion_status",
        )
        if prerequisite.get("closed_release_outcome") is not None:
            raise CanonicalSupervisorContractFreezeError(
                "qualification_prerequisite.closed_release_outcome must be null"
            )
        if prerequisite.get("release_claim") is True:
            raise CanonicalSupervisorContractFreezeError(
                "qualification_prerequisite must not claim a release"
            )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise CanonicalSupervisorContractFreezeError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise CanonicalSupervisorContractFreezeError(
                "acceptance must not claim a release"
            )
        if acceptance.get("promotion_status") not in {None, promotion_status}:
            raise CanonicalSupervisorContractFreezeError(
                "acceptance.promotion_status must match qualification_verdict"
            )
        if acceptance.get("contracts_frozen") is True and contracts_frozen is not True:
            raise CanonicalSupervisorContractFreezeError(
                "acceptance.contracts_frozen must match qualification_verdict"
            )

    binding = payload.get("current_tree_binding")
    if isinstance(binding, Mapping):
        if binding.get("evidence_kind") != "measured":
            raise CanonicalSupervisorContractFreezeError(
                "current_tree_binding.evidence_kind must be measured"
            )
        if binding.get("origin_main_is_ancestor") is not True:
            raise CanonicalSupervisorContractFreezeError(
                "current_tree_binding.origin_main_is_ancestor must be true"
            )
        _reject_closed_release_value(binding.get("outer_subject"), "outer_subject")
        for name in (
            "outer_commit",
            "outer_tree",
            "origin_main",
            "accelerator_pre_change_commit",
            "accelerator_pre_change_tree",
            "accelerator_gitlink",
            "accelerator_origin_main",
        ):
            _git_object_id(binding.get(name), f"current_tree_binding.{name}")

    expected = current_head_pcpr_002_receipt_promotion()
    live_promoted = False
    if isinstance(prerequisite, Mapping):
        live_promoted = (
            prerequisite.get("promotion_status") == "supervisor_promoted"
            and prerequisite.get("live_qualification_evidence_kind") == LIVE_SATISFYING_KIND
        )
    if not live_promoted:
        if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
            raise CanonicalSupervisorContractFreezeError(
                "missing-live qualification_verdict.verdict_cid must match the evaluator"
            )
        if promotion_status != expected["promotion_status"]:
            raise CanonicalSupervisorContractFreezeError(
                "missing-live promotion_status must match the evaluator"
            )
        if contracts_frozen is True:
            raise CanonicalSupervisorContractFreezeError(
                "missing-live PCPR-002 must not freeze contracts"
            )
        if expected["promotion_status"] == "rnd_non_promoted" and promotion_status not in {
            "rnd_non_promoted",
            "typed_unavailable",
            "typed_blocked",
            "supervisor_non_promoted",
        }:
            raise CanonicalSupervisorContractFreezeError(
                "missing-live promotion_status must be an honest non-promotion"
            )
        if expected["verdict_cid"] != CURRENT_HEAD_NON_PROMOTION_VERDICT_CID:
            raise CanonicalSupervisorContractFreezeError(
                "pinned current-head non-promotion CID drifted from the evaluator"
            )
        if (
            isinstance(prerequisite, Mapping)
            and prerequisite.get("verdict_cid") not in {None, CURRENT_HEAD_UNAVAILABLE_VERDICT_CID}
        ):
            raise CanonicalSupervisorContractFreezeError(
                "missing-live qualification_prerequisite.verdict_cid must match PCPR-001"
            )
    return {
        "valid": True,
        "task_id": PCPR_002_TASK_ID,
        "promotion_status": promotion_status,
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": bool(contracts_frozen),
        "verdict_cid": verdict_section.get("verdict_cid"),
        "evidence_kind": "measured",
    }
