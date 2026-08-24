"""Hermetic adversarial and chaos qualification for the federation.

This module is deliberately an observation harness, not a second control
plane.  A caller supplies probes that exercise the canonical typed state-owner
boundary and returns compact evidence that the attempted attack was rejected
or blocked before an effect, authority, or completion claim escaped.  The
harness never opens DuckDB, contacts Quack, starts a process, or treats a
successful test as runtime authority.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..task_sources.control_plane_contracts import content_identity


CASF_CHAOS_TASK_ID: Final[str] = "CASF-037"
CASF_CHAOS_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/chaos-identity@1"
)
CASF_CHAOS_SCENARIO_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/chaos-scenario@1"
)
CASF_CHAOS_SUITE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/chaos-suite@1"
)
CASF_CHAOS_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/chaos-observation@1"
)
CASF_CHAOS_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/chaos-report@1"
)

_GIT_OID = re.compile(r"[0-9a-f]{40}")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{0,511}")
_MAX_EVIDENCE_REFS: Final[int] = 128
_IDENTITY_WIRE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema", "source_revision", "source_tree", "state_schema", "generation_id",
        "federation_id", "policy_id", "capability_ids", "task_id", "attempt_id",
        "lease_id", "fencing_epoch", "assignment_revision", "worktree_id", "identity",
    }
)


class FederationChaosError(ValueError):
    """A chaos-suite record is malformed or an unsafe observation was supplied."""


class ChaosVerificationError(FederationChaosError):
    """A probe did not prove that an attempted attack was contained."""


class ChaosAttack(str, Enum):
    """Closed attack catalog spanning every CASF-037 safety surface."""

    UNAUTHORIZED_MUTATION = "unauthorized_mutation"
    CROSS_TENANT_MUTATION = "cross_tenant_mutation"
    SECRET_SHAPED_INPUT = "secret_shaped_input"
    STALE_FENCE_MUTATION = "stale_fence_mutation"
    DUPLICATE_AUTHORITATIVE_EFFECT = "duplicate_authoritative_effect"
    EVENT_STORM = "event_storm"
    ILLEGAL_LIFECYCLE_TRANSITION = "illegal_lifecycle_transition"
    STALE_REBALANCE = "stale_rebalance"
    ORPHAN_CAUSAL_PROPAGATION = "orphan_causal_propagation"
    CRASH_RECOVERY_REPLAY = "crash_recovery_replay"


class ChaosDomain(str, Enum):
    AUTHORIZATION = "authorization"
    TENANCY = "tenancy"
    SECRETS = "secrets"
    LEASES_AND_FENCES = "leases_and_fences"
    EVENTS = "events"
    LIFECYCLE = "lifecycle"
    REBALANCE = "rebalance"
    CAUSAL_PROPAGATION = "causal_propagation"
    RECOVERY = "recovery"


class ChaosDisposition(str, Enum):
    """Only fail-closed attack dispositions are representable."""

    REJECTED = "rejected"
    BLOCKED = "blocked"


class ChaosReportStatus(str, Enum):
    QUALIFIED = "qualified"
    BLOCKED = "blocked"


def _token(value: Any, name: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise FederationChaosError(f"{name} must be nonempty exact text")
    if _TOKEN.fullmatch(value) is None:
        raise FederationChaosError(f"{name} is not a compact identity")
    return value


def _git_oid(value: Any, name: str) -> str:
    value = _token(value, name)
    if _GIT_OID.fullmatch(value) is None:
        raise FederationChaosError(f"{name} must be a lowercase 40-hex Git object id")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FederationChaosError(f"{name} must be a positive integer")
    return value


def _identity(value: Mapping[str, Any], *, prefix: str) -> str:
    return f"{prefix}:{content_identity(dict(value))}"


def _closed_mapping(value: Mapping[str, Any], allowed: frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise FederationChaosError(f"{label} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise FederationChaosError(
            f"{label} has unknown fields: " + repr(sorted(str(item) for item in unknown))
        )


@dataclass(frozen=True)
class FederationChaosIdentity:
    """Exact source and fence binding for a non-authoritative suite run."""

    SCHEMA: ClassVar[str] = CASF_CHAOS_IDENTITY_SCHEMA

    source_revision: str
    source_tree: str
    state_schema: str
    generation_id: str
    federation_id: str
    policy_id: str
    capability_ids: tuple[str, ...]
    task_id: str
    attempt_id: str
    lease_id: str
    fencing_epoch: int
    assignment_revision: int
    worktree_id: str
    schema: str = CASF_CHAOS_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != self.SCHEMA:
            raise FederationChaosError("unsupported chaos identity schema")
        object.__setattr__(self, "source_revision", _git_oid(self.source_revision, "source_revision"))
        object.__setattr__(self, "source_tree", _git_oid(self.source_tree, "source_tree"))
        for name in (
            "state_schema",
            "generation_id",
            "federation_id",
            "policy_id",
            "task_id",
            "attempt_id",
            "lease_id",
            "worktree_id",
        ):
            object.__setattr__(self, name, _token(getattr(self, name), name))
        if self.task_id != CASF_CHAOS_TASK_ID:
            raise FederationChaosError("task_id must be the exact CASF-037 identity")
        if not isinstance(self.capability_ids, tuple) or not self.capability_ids:
            raise FederationChaosError("capability_ids must be a nonempty identity tuple")
        capabilities = tuple(_token(item, "capability_id") for item in self.capability_ids)
        if len(capabilities) != len(set(capabilities)):
            raise FederationChaosError("capability_ids contains duplicates")
        object.__setattr__(self, "capability_ids", tuple(sorted(capabilities)))
        _positive_integer(self.fencing_epoch, "fencing_epoch")
        _positive_integer(self.assignment_revision, "assignment_revision")

    @property
    def identity(self) -> str:
        return _identity(self.to_dict(include_identity=False), prefix="chaos-identity")

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "source_revision": self.source_revision,
            "source_tree": self.source_tree,
            "state_schema": self.state_schema,
            "generation_id": self.generation_id,
            "federation_id": self.federation_id,
            "policy_id": self.policy_id,
            "capability_ids": list(self.capability_ids),
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "assignment_revision": self.assignment_revision,
            "worktree_id": self.worktree_id,
        }
        if include_identity:
            value["identity"] = self.identity
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FederationChaosIdentity:
        _closed_mapping(value, _IDENTITY_WIRE_FIELDS, "chaos identity")
        result = cls(
            source_revision=value.get("source_revision", ""),
            source_tree=value.get("source_tree", ""),
            state_schema=value.get("state_schema", ""),
            generation_id=value.get("generation_id", ""),
            federation_id=value.get("federation_id", ""),
            policy_id=value.get("policy_id", ""),
            capability_ids=tuple(value.get("capability_ids") or ()),
            task_id=value.get("task_id", ""),
            attempt_id=value.get("attempt_id", ""),
            lease_id=value.get("lease_id", ""),
            fencing_epoch=value.get("fencing_epoch", 0),
            assignment_revision=value.get("assignment_revision", 0),
            worktree_id=value.get("worktree_id", ""),
            schema=value.get("schema", ""),
        )
        if value.get("identity") not in (None, result.identity):
            raise FederationChaosError("claimed chaos identity does not match its content")
        return result


_ATTACK_DOMAINS: Final[Mapping[ChaosAttack, ChaosDomain]] = MappingProxyType(
    {
        ChaosAttack.UNAUTHORIZED_MUTATION: ChaosDomain.AUTHORIZATION,
        ChaosAttack.CROSS_TENANT_MUTATION: ChaosDomain.TENANCY,
        ChaosAttack.SECRET_SHAPED_INPUT: ChaosDomain.SECRETS,
        ChaosAttack.STALE_FENCE_MUTATION: ChaosDomain.LEASES_AND_FENCES,
        ChaosAttack.DUPLICATE_AUTHORITATIVE_EFFECT: ChaosDomain.EVENTS,
        ChaosAttack.EVENT_STORM: ChaosDomain.EVENTS,
        ChaosAttack.ILLEGAL_LIFECYCLE_TRANSITION: ChaosDomain.LIFECYCLE,
        ChaosAttack.STALE_REBALANCE: ChaosDomain.REBALANCE,
        ChaosAttack.ORPHAN_CAUSAL_PROPAGATION: ChaosDomain.CAUSAL_PROPAGATION,
        ChaosAttack.CRASH_RECOVERY_REPLAY: ChaosDomain.RECOVERY,
    }
)


@dataclass(frozen=True)
class ChaosScenario:
    """One deterministic attack recipe; it contains no raw attack payload."""

    scenario_id: str
    attack: ChaosAttack
    domain: ChaosDomain
    expected_dispositions: tuple[ChaosDisposition, ...] = (
        ChaosDisposition.REJECTED,
        ChaosDisposition.BLOCKED,
    )
    schema: str = CASF_CHAOS_SCENARIO_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CASF_CHAOS_SCENARIO_SCHEMA:
            raise FederationChaosError("unsupported chaos scenario schema")
        object.__setattr__(self, "scenario_id", _token(self.scenario_id, "scenario_id"))
        if not isinstance(self.attack, ChaosAttack) or not isinstance(self.domain, ChaosDomain):
            raise FederationChaosError("chaos scenario must use closed attack and domain values")
        if _ATTACK_DOMAINS[self.attack] is not self.domain:
            raise FederationChaosError("chaos scenario domain differs from its attack catalog")
        if not isinstance(self.expected_dispositions, tuple) or not self.expected_dispositions:
            raise FederationChaosError("scenario requires at least one expected disposition")
        if any(not isinstance(item, ChaosDisposition) for item in self.expected_dispositions):
            raise FederationChaosError("scenario disposition is outside the closed vocabulary")
        if len(set(self.expected_dispositions)) != len(self.expected_dispositions):
            raise FederationChaosError("scenario repeats an expected disposition")

    @property
    def identity(self) -> str:
        return _identity(self.to_dict(include_identity=False), prefix="chaos-scenario")

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "scenario_id": self.scenario_id,
            "attack": self.attack.value,
            "domain": self.domain.value,
            "expected_dispositions": [item.value for item in self.expected_dispositions],
        }
        if include_identity:
            value["identity"] = self.identity
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChaosScenario:
        _closed_mapping(
            value,
            frozenset(
                {"schema", "scenario_id", "attack", "domain", "expected_dispositions", "identity"}
            ),
            "chaos scenario",
        )
        try:
            result = cls(
                scenario_id=value.get("scenario_id", ""),
                attack=ChaosAttack(value.get("attack", "")),
                domain=ChaosDomain(value.get("domain", "")),
                expected_dispositions=tuple(
                    ChaosDisposition(item) for item in value.get("expected_dispositions", ())
                ),
                schema=value.get("schema", ""),
            )
        except ValueError as exc:
            raise FederationChaosError("chaos scenario has an unknown closed value") from exc
        if value.get("identity") not in (None, result.identity):
            raise FederationChaosError("claimed chaos scenario identity does not match its content")
        return result


@dataclass(frozen=True)
class FederationChaosSuite:
    """A complete, bounded catalog of CASF-037 attack probes."""

    identity: FederationChaosIdentity
    scenarios: tuple[ChaosScenario, ...]
    schema: str = CASF_CHAOS_SUITE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CASF_CHAOS_SUITE_SCHEMA:
            raise FederationChaosError("unsupported chaos suite schema")
        if not isinstance(self.identity, FederationChaosIdentity):
            raise FederationChaosError("chaos suite requires a typed identity")
        if not isinstance(self.scenarios, tuple) or any(
            not isinstance(item, ChaosScenario) for item in self.scenarios
        ):
            raise FederationChaosError("chaos suite must contain typed scenarios")
        attacks = tuple(item.attack for item in self.scenarios)
        if len(attacks) != len(set(attacks)):
            raise FederationChaosError("chaos suite repeats an attack")
        if set(attacks) != set(ChaosAttack):
            missing = sorted(item.value for item in set(ChaosAttack) - set(attacks))
            raise FederationChaosError("chaos suite is incomplete: " + repr(missing))

    @property
    def suite_id(self) -> str:
        return _identity(self.to_dict(include_identity=False), prefix="chaos-suite")

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "identity": self.identity.to_dict(),
            "scenarios": [item.to_dict() for item in self.scenarios],
            "authority_created": False,
            "completion_created": False,
        }
        if include_identity:
            value["suite_id"] = self.suite_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FederationChaosSuite:
        _closed_mapping(
            value,
            frozenset(
                {"schema", "identity", "scenarios", "authority_created", "completion_created", "suite_id"}
            ),
            "chaos suite",
        )
        if value.get("authority_created") is not False or value.get("completion_created") is not False:
            raise FederationChaosError("chaos suite cannot create authority or completion")
        try:
            identity_value = value.get("identity", {})
            scenarios_value = value.get("scenarios", ())
            result = cls(
                identity=FederationChaosIdentity.from_dict(identity_value),
                scenarios=tuple(ChaosScenario.from_dict(item) for item in scenarios_value),
                schema=value.get("schema", ""),
            )
        except (TypeError, FederationChaosError) as exc:
            raise FederationChaosError("chaos suite contains malformed nested records") from exc
        if value.get("suite_id") not in (None, result.suite_id):
            raise FederationChaosError("claimed chaos suite identity does not match its content")
        return result


@dataclass(frozen=True)
class ChaosObservation:
    """Compact probe evidence that an attack produced no unsafe escape."""

    scenario_id: str
    attack: ChaosAttack
    disposition: ChaosDisposition
    effect_observed: bool
    authority_created: bool
    completion_created: bool
    evidence_refs: tuple[str, ...]
    reason_code: str
    schema: str = CASF_CHAOS_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CASF_CHAOS_OBSERVATION_SCHEMA:
            raise FederationChaosError("unsupported chaos observation schema")
        object.__setattr__(self, "scenario_id", _token(self.scenario_id, "scenario_id"))
        if not isinstance(self.attack, ChaosAttack) or not isinstance(self.disposition, ChaosDisposition):
            raise FederationChaosError("observation must use closed attack and disposition values")
        for name in ("effect_observed", "authority_created", "completion_created"):
            if type(getattr(self, name)) is not bool:
                raise FederationChaosError(f"{name} must be boolean")
        if not isinstance(self.evidence_refs, tuple) or not self.evidence_refs:
            raise FederationChaosError("observation requires evidence references")
        if len(self.evidence_refs) > _MAX_EVIDENCE_REFS:
            raise FederationChaosError("observation exceeds its evidence bound")
        refs = tuple(_token(item, "evidence_ref") for item in self.evidence_refs)
        if len(refs) != len(set(refs)):
            raise FederationChaosError("observation repeats evidence references")
        object.__setattr__(self, "evidence_refs", tuple(sorted(refs)))
        object.__setattr__(self, "reason_code", _token(self.reason_code, "reason_code"))

    @property
    def observation_id(self) -> str:
        return _identity(self.to_dict(include_identity=False), prefix="chaos-observation")

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "scenario_id": self.scenario_id,
            "attack": self.attack.value,
            "disposition": self.disposition.value,
            "effect_observed": self.effect_observed,
            "authority_created": self.authority_created,
            "completion_created": self.completion_created,
            "evidence_refs": list(self.evidence_refs),
            "reason_code": self.reason_code,
        }
        if include_identity:
            value["observation_id"] = self.observation_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChaosObservation:
        _closed_mapping(
            value,
            frozenset(
                {
                    "schema", "scenario_id", "attack", "disposition", "effect_observed",
                    "authority_created", "completion_created", "evidence_refs", "reason_code",
                    "observation_id",
                }
            ),
            "chaos observation",
        )
        try:
            result = cls(
                scenario_id=value.get("scenario_id", ""),
                attack=ChaosAttack(value.get("attack", "")),
                disposition=ChaosDisposition(value.get("disposition", "")),
                effect_observed=value.get("effect_observed"),
                authority_created=value.get("authority_created"),
                completion_created=value.get("completion_created"),
                evidence_refs=tuple(value.get("evidence_refs") or ()),
                reason_code=value.get("reason_code", ""),
                schema=value.get("schema", ""),
            )
        except ValueError as exc:
            raise FederationChaosError("chaos observation has an unknown closed value") from exc
        if value.get("observation_id") not in (None, result.observation_id):
            raise FederationChaosError("claimed chaos observation identity does not match its content")
        return result


@dataclass(frozen=True)
class ChaosReport:
    """A qualification observation, explicitly not a completion authority."""

    suite_id: str
    status: ChaosReportStatus
    observations: tuple[ChaosObservation, ...]
    schema: str = CASF_CHAOS_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CASF_CHAOS_REPORT_SCHEMA:
            raise FederationChaosError("unsupported chaos report schema")
        object.__setattr__(self, "suite_id", _token(self.suite_id, "suite_id"))
        if not isinstance(self.status, ChaosReportStatus):
            raise FederationChaosError("report status is outside the closed vocabulary")
        if not self.observations or any(not isinstance(item, ChaosObservation) for item in self.observations):
            raise FederationChaosError("report requires typed observations")
        scenario_ids = tuple(item.scenario_id for item in self.observations)
        if len(scenario_ids) != len(set(scenario_ids)):
            raise FederationChaosError("report repeats a scenario observation")
        attacks = {item.attack for item in self.observations}
        if attacks != set(ChaosAttack):
            raise FederationChaosError("report does not cover the complete chaos attack catalog")
        if any(
            item.effect_observed or item.authority_created or item.completion_created
            for item in self.observations
        ):
            raise FederationChaosError("chaos report contains an unsafe attack escape")
        blocked = any(item.disposition is ChaosDisposition.BLOCKED for item in self.observations)
        expected_status = ChaosReportStatus.BLOCKED if blocked else ChaosReportStatus.QUALIFIED
        if self.status is not expected_status:
            raise FederationChaosError("chaos report status does not match contained observations")

    @property
    def qualified(self) -> bool:
        return self.status is ChaosReportStatus.QUALIFIED

    @property
    def report_id(self) -> str:
        return _identity(self.to_dict(include_identity=False), prefix="chaos-report")

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "suite_id": self.suite_id,
            "status": self.status.value,
            "observations": [item.to_dict() for item in self.observations],
            "authority_created": False,
            "completion_created": False,
            "bounded": True,
        }
        if include_identity:
            value["report_id"] = self.report_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChaosReport:
        _closed_mapping(
            value,
            frozenset(
                {
                    "schema", "suite_id", "status", "observations", "authority_created",
                    "completion_created", "bounded", "report_id",
                }
            ),
            "chaos report",
        )
        if (
            value.get("authority_created") is not False
            or value.get("completion_created") is not False
            or value.get("bounded") is not True
        ):
            raise FederationChaosError("chaos report cannot claim authority, completion, or unbounded proof")
        try:
            result = cls(
                suite_id=value.get("suite_id", ""),
                status=ChaosReportStatus(value.get("status", "")),
                observations=tuple(ChaosObservation.from_dict(item) for item in value.get("observations", ())),
                schema=value.get("schema", ""),
            )
        except (TypeError, ValueError, FederationChaosError) as exc:
            raise FederationChaosError("chaos report contains malformed nested records") from exc
        if value.get("report_id") not in (None, result.report_id):
            raise FederationChaosError("claimed chaos report identity does not match its content")
        return result


def build_federation_chaos_suite(identity: FederationChaosIdentity) -> FederationChaosSuite:
    """Build the complete deterministic attack catalog for one fenced run."""

    if not isinstance(identity, FederationChaosIdentity):
        raise FederationChaosError("chaos suite requires FederationChaosIdentity")
    scenarios = tuple(
        ChaosScenario(
            scenario_id=f"chaos:{attack.value}",
            attack=attack,
            domain=_ATTACK_DOMAINS[attack],
        )
        for attack in ChaosAttack
    )
    return FederationChaosSuite(identity=identity, scenarios=scenarios)


def run_federation_chaos_suite(
    suite: FederationChaosSuite,
    probe: Callable[[ChaosScenario], ChaosObservation],
) -> ChaosReport:
    """Run bounded probes and fail closed unless every attack is contained.

    A ``BLOCKED`` observation is safe but not a qualification pass: it signals
    that the required capability was unavailable and leaves the report blocked.
    Probe exceptions are intentionally propagated, because a missing
    observation cannot prove that no effect escaped.
    """

    if not isinstance(suite, FederationChaosSuite):
        raise FederationChaosError("suite must be FederationChaosSuite")
    if not callable(probe):
        raise FederationChaosError("probe must be callable")
    observations: list[ChaosObservation] = []
    blocked = False
    for scenario in suite.scenarios:
        observation = probe(scenario)
        if not isinstance(observation, ChaosObservation):
            raise ChaosVerificationError("probe returned no typed chaos observation")
        if observation.scenario_id != scenario.scenario_id or observation.attack is not scenario.attack:
            raise ChaosVerificationError("probe observation does not bind its scenario")
        if observation.disposition not in scenario.expected_dispositions:
            raise ChaosVerificationError("probe disposition is not permitted by its scenario")
        if observation.effect_observed:
            raise ChaosVerificationError("attack produced an observed effect")
        if observation.authority_created:
            raise ChaosVerificationError("attack created authority")
        if observation.completion_created:
            raise ChaosVerificationError("attack created a completion claim")
        blocked = blocked or observation.disposition is ChaosDisposition.BLOCKED
        observations.append(observation)
    return ChaosReport(
        suite_id=suite.suite_id,
        status=ChaosReportStatus.BLOCKED if blocked else ChaosReportStatus.QUALIFIED,
        observations=tuple(observations),
    )


__all__ = [
    "CASF_CHAOS_IDENTITY_SCHEMA",
    "CASF_CHAOS_OBSERVATION_SCHEMA",
    "CASF_CHAOS_REPORT_SCHEMA",
    "CASF_CHAOS_SCENARIO_SCHEMA",
    "CASF_CHAOS_SUITE_SCHEMA",
    "CASF_CHAOS_TASK_ID",
    "ChaosAttack",
    "ChaosDisposition",
    "ChaosDomain",
    "ChaosObservation",
    "ChaosReport",
    "ChaosReportStatus",
    "ChaosScenario",
    "ChaosVerificationError",
    "FederationChaosError",
    "FederationChaosIdentity",
    "FederationChaosSuite",
    "build_federation_chaos_suite",
    "run_federation_chaos_suite",
]
