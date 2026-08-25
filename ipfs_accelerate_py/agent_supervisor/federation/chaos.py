"""Bounded adversarial evidence for the causal supervisor federation.

Injected probes are diagnostics only and can never qualify CASF.  The closed
runner accepts no callback, binds an exact canonical observation population to
post-merge validation, proof, capability, and rollback evidence, and emits the
exact ``casf/adversarial-report@1`` record.  That report is deliberately not
runtime, completion, or promotion authority: a promotion consumer must fetch
and reverify every referenced receipt through its authoritative state owner.

This module never opens DuckDB, contacts Quack, starts a process, signs a
claim, or fabricates an external attestation.
"""

# Python 3.8 remains supported, so keep the compatible ``str, Enum`` spelling.
# ruff: noqa: UP042

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..task_sources.control_plane_contracts import content_identity
from ..todo_daemon.post_merge_validation import verify_post_merge_validation_evidence

CASF_CHAOS_TASK_ID: Final[str] = "CASF-037"
CASF_CHAOS_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/chaos-identity@2"
)
CASF_CHAOS_SCENARIO_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/chaos-scenario@2"
)
CASF_CHAOS_SUITE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/chaos-suite@2"
)
CASF_CHAOS_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/chaos-observation@2"
)
CASF_CHAOS_VALIDATION_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/chaos-validation-binding@1"
)
CASF_CHAOS_EVIDENCE_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/chaos-evidence-binding@1"
)
CASF_CHAOS_DIAGNOSTIC_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/chaos-diagnostic-report@1"
)
CASF_CHAOS_REPORT_SCHEMA: Final[str] = "casf/adversarial-report@1"

_GIT_OID = re.compile(r"[0-9a-f]{40}")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{0,511}")
_CONTENT_REF = re.compile(r"(?:sha256:[0-9a-f]{64}|b[a-z2-7]{20,})")
_SECRET_VALUE = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\bBearer\s+[A-Za-z0-9._~+/=-]{8,}|\bsk-[A-Za-z0-9_-]{12,})",
    re.IGNORECASE,
)
_MAX_EVIDENCE_ITEMS: Final[int] = 128


class FederationChaosError(ValueError):
    """A chaos record is malformed or an unsafe claim was supplied."""


class ChaosVerificationError(FederationChaosError):
    """Evidence did not prove that an attempted attack was contained."""


class ChaosAttack(str, Enum):
    """Closed catalog covering the required adversarial surfaces."""

    UNAUTHORIZED_MUTATION = "unauthorized_mutation"
    CROSS_TENANT_MUTATION = "cross_tenant_mutation"
    SECRET_SHAPED_INPUT = "secret_shaped_input"
    STALE_FENCE_MUTATION = "stale_fence_mutation"
    DUPLICATE_AUTHORITATIVE_EFFECT = "duplicate_authoritative_effect"
    EVENT_STORM = "event_storm"
    ILLEGAL_LIFECYCLE_TRANSITION = "illegal_lifecycle_transition"
    STALE_REBALANCE = "stale_rebalance"
    ORPHAN_CAUSAL_PROPAGATION = "orphan_causal_propagation"
    MISSED_CAUSAL_NOTIFICATION = "missed_causal_notification"
    CAUSAL_INDEPENDENCE_VIOLATION = "causal_independence_violation"
    STALE_ABSTRACTION_SUPPRESSION = "stale_abstraction_suppression"
    NON_AUTHORITATIVE_PROMOTION = "non_authoritative_promotion"
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
    CAUSAL_NOTIFICATION = "causal_notification"
    CAUSAL_INDEPENDENCE = "causal_independence"
    ABSTRACTION_FRESHNESS = "abstraction_freshness"
    NON_PROMOTION = "non_promotion"
    RECOVERY = "recovery"


class ChaosDisposition(str, Enum):
    REJECTED = "rejected"
    BLOCKED = "blocked"


class ChaosReportStatus(str, Enum):
    QUALIFIED = "qualified"
    BLOCKED = "blocked"


class ChaosDiagnosticStatus(str, Enum):
    DIAGNOSTIC = "diagnostic"
    BLOCKED = "blocked"


class ChaosCapabilityStatus(str, Enum):
    QUALIFIED = "qualified"
    MISSING = "missing"
    STALE = "stale"
    UNQUALIFIED = "unqualified"
    UNAVAILABLE = "unavailable"


class ChaosProofStatus(str, Enum):
    PASSED = "passed"
    BLOCKED = "blocked"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


def _token(value: Any, name: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise FederationChaosError(f"{name} must be nonempty exact text")
    if _SECRET_VALUE.search(value):
        raise FederationChaosError(f"{name} contains credential-shaped material")
    if _TOKEN.fullmatch(value) is None:
        raise FederationChaosError(f"{name} is not a compact identity")
    return value


def _content_ref(value: Any, name: str, *, required: bool = True) -> str:
    if not required and value in (None, ""):
        return ""
    value = _token(value, name)
    if _CONTENT_REF.fullmatch(value) is None:
        raise FederationChaosError(f"{name} must be a CID or sha256 content reference")
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


def _closed_mapping(value: Mapping[str, Any], fields: frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise FederationChaosError(f"{label} must be an object")
    unknown = set(value) - fields
    missing = fields - set(value)
    if unknown:
        raise FederationChaosError(
            f"{label} has unknown fields: " + repr(sorted(str(item) for item in unknown))
        )
    if missing:
        raise FederationChaosError(
            f"{label} is missing fields: " + repr(sorted(str(item) for item in missing))
        )


def _tokens(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
    content_refs: bool = False,
    allow_empty_content_refs: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise FederationChaosError(f"{name} must be an immutable tuple")
    if len(value) < minimum or len(value) > maximum:
        raise FederationChaosError(f"{name} exceeds its declared bounds")
    if content_refs:
        result = tuple(
            _content_ref(
                item,
                f"{name}[{index}]",
                required=not allow_empty_content_refs,
            )
            for index, item in enumerate(value)
        )
    else:
        result = tuple(_token(item, f"{name}[{index}]") for index, item in enumerate(value))
    nonempty = tuple(item for item in result if item)
    if len(nonempty) != len(set(nonempty)):
        raise FederationChaosError(f"{name} contains duplicates")
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
        ChaosAttack.MISSED_CAUSAL_NOTIFICATION: ChaosDomain.CAUSAL_NOTIFICATION,
        ChaosAttack.CAUSAL_INDEPENDENCE_VIOLATION: ChaosDomain.CAUSAL_INDEPENDENCE,
        ChaosAttack.STALE_ABSTRACTION_SUPPRESSION: ChaosDomain.ABSTRACTION_FRESHNESS,
        ChaosAttack.NON_AUTHORITATIVE_PROMOTION: ChaosDomain.NON_PROMOTION,
        ChaosAttack.CRASH_RECOVERY_REPLAY: ChaosDomain.RECOVERY,
    }
)
_CLOSED_PROBE_IDS: Final[Mapping[ChaosAttack, str]] = MappingProxyType(
    {attack: f"casf-037:closed-probe-slot:{attack.value}@1" for attack in ChaosAttack}
)
CASF_CHAOS_PROBE_CATALOG_ID: Final[str] = _identity(
    {
        "schema": CASF_CHAOS_SCENARIO_SCHEMA,
        "probes": [
            {
                "attack": attack.value,
                "domain": _ATTACK_DOMAINS[attack].value,
                "probe_id": _CLOSED_PROBE_IDS[attack],
            }
            for attack in ChaosAttack
        ],
    },
    prefix="chaos-probe-catalog",
)
CASF_CHAOS_CLOSED_RUNNER_ID: Final[str] = _identity(
    {
        "task_id": CASF_CHAOS_TASK_ID,
        "report_schema": CASF_CHAOS_REPORT_SCHEMA,
        "probe_catalog_id": CASF_CHAOS_PROBE_CATALOG_ID,
        "accepts_injected_probe": False,
        "creates_authority": False,
    },
    prefix="chaos-runner",
)
CASF_CHAOS_LOCAL_QUALIFICATION_AVAILABLE: Final[bool] = False


@dataclass(frozen=True)
class FederationChaosIdentity:
    """Exact source, policy, assignment, and fence binding for one suite."""

    SCHEMA: ClassVar[str] = CASF_CHAOS_IDENTITY_SCHEMA

    source_revision: str
    source_tree: str
    state_schema: str
    generation_id: str
    federation_id: str
    policy_id: str
    policy_revision: int
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
        object.__setattr__(
            self, "source_revision", _git_oid(self.source_revision, "source_revision")
        )
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
        _positive_integer(self.policy_revision, "policy_revision")
        _positive_integer(self.fencing_epoch, "fencing_epoch")
        _positive_integer(self.assignment_revision, "assignment_revision")
        capabilities = _tokens(
            self.capability_ids,
            "capability_ids",
            minimum=1,
            maximum=_MAX_EVIDENCE_ITEMS,
        )
        if capabilities != tuple(sorted(capabilities)):
            raise FederationChaosError("capability_ids is not canonical")
        object.__setattr__(self, "capability_ids", capabilities)

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
            "policy_revision": self.policy_revision,
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
        fields = frozenset(
            {
                "schema",
                "source_revision",
                "source_tree",
                "state_schema",
                "generation_id",
                "federation_id",
                "policy_id",
                "policy_revision",
                "capability_ids",
                "task_id",
                "attempt_id",
                "lease_id",
                "fencing_epoch",
                "assignment_revision",
                "worktree_id",
                "identity",
            }
        )
        _closed_mapping(value, fields, "chaos identity")
        try:
            result = cls(
                source_revision=value["source_revision"],
                source_tree=value["source_tree"],
                state_schema=value["state_schema"],
                generation_id=value["generation_id"],
                federation_id=value["federation_id"],
                policy_id=value["policy_id"],
                policy_revision=value["policy_revision"],
                capability_ids=tuple(value["capability_ids"]),
                task_id=value["task_id"],
                attempt_id=value["attempt_id"],
                lease_id=value["lease_id"],
                fencing_epoch=value["fencing_epoch"],
                assignment_revision=value["assignment_revision"],
                worktree_id=value["worktree_id"],
                schema=value["schema"],
            )
        except (KeyError, TypeError, FederationChaosError) as exc:
            raise FederationChaosError("chaos identity is malformed") from exc
        if value["identity"] != result.identity:
            raise FederationChaosError("claimed chaos identity does not match its content")
        return result


@dataclass(frozen=True)
class ChaosScenario:
    """One canonical attack recipe containing no raw attack payload."""

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
        if type(self.attack) is not ChaosAttack or type(self.domain) is not ChaosDomain:
            raise FederationChaosError("scenario attack and domain must be exact closed values")
        if _token(self.scenario_id, "scenario_id") != f"chaos:{self.attack.value}":
            raise FederationChaosError("scenario identity differs from its attack")
        if _ATTACK_DOMAINS[self.attack] is not self.domain:
            raise FederationChaosError("scenario domain differs from its attack catalog")
        if self.expected_dispositions != (
            ChaosDisposition.REJECTED,
            ChaosDisposition.BLOCKED,
        ):
            raise FederationChaosError("scenario dispositions differ from the closed catalog")

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
        fields = frozenset(
            {
                "schema",
                "scenario_id",
                "attack",
                "domain",
                "expected_dispositions",
                "identity",
            }
        )
        _closed_mapping(value, fields, "chaos scenario")
        try:
            result = cls(
                scenario_id=value["scenario_id"],
                attack=ChaosAttack(value["attack"]),
                domain=ChaosDomain(value["domain"]),
                expected_dispositions=tuple(
                    ChaosDisposition(item) for item in value["expected_dispositions"]
                ),
                schema=value["schema"],
            )
        except (KeyError, TypeError, ValueError, FederationChaosError) as exc:
            raise FederationChaosError("chaos scenario is malformed") from exc
        if value["identity"] != result.identity:
            raise FederationChaosError("claimed scenario identity does not match its content")
        return result


@dataclass(frozen=True)
class FederationChaosSuite:
    """Complete immutable canonically ordered CASF-037 attack suite."""

    identity: FederationChaosIdentity
    scenarios: tuple[ChaosScenario, ...]
    schema: str = CASF_CHAOS_SUITE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CASF_CHAOS_SUITE_SCHEMA:
            raise FederationChaosError("unsupported chaos suite schema")
        if type(self.identity) is not FederationChaosIdentity:
            raise FederationChaosError("suite requires exact FederationChaosIdentity")
        if not isinstance(self.scenarios, tuple) or any(
            type(item) is not ChaosScenario for item in self.scenarios
        ):
            raise FederationChaosError("suite scenarios must be exact immutable records")
        expected = tuple(
            ChaosScenario(
                scenario_id=f"chaos:{attack.value}",
                attack=attack,
                domain=_ATTACK_DOMAINS[attack],
            )
            for attack in ChaosAttack
        )
        if self.scenarios != expected:
            raise FederationChaosError("suite differs from the exact canonical catalog")

    @property
    def suite_id(self) -> str:
        return _identity(self.to_dict(include_identity=False), prefix="chaos-suite")

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "identity": self.identity.to_dict(),
            "probe_catalog_id": CASF_CHAOS_PROBE_CATALOG_ID,
            "scenarios": [item.to_dict() for item in self.scenarios],
            "authority_created": False,
            "completion_created": False,
        }
        if include_identity:
            value["suite_id"] = self.suite_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FederationChaosSuite:
        fields = frozenset(
            {
                "schema",
                "identity",
                "probe_catalog_id",
                "scenarios",
                "authority_created",
                "completion_created",
                "suite_id",
            }
        )
        _closed_mapping(value, fields, "chaos suite")
        if (
            value["probe_catalog_id"] != CASF_CHAOS_PROBE_CATALOG_ID
            or value["authority_created"] is not False
            or value["completion_created"] is not False
        ):
            raise FederationChaosError("suite has invalid catalog or authority flags")
        try:
            result = cls(
                identity=FederationChaosIdentity.from_dict(value["identity"]),
                scenarios=tuple(ChaosScenario.from_dict(item) for item in value["scenarios"]),
                schema=value["schema"],
            )
        except (KeyError, TypeError, FederationChaosError) as exc:
            raise FederationChaosError("chaos suite is malformed") from exc
        if value["suite_id"] != result.suite_id:
            raise FederationChaosError("claimed suite identity does not match its content")
        return result


@dataclass(frozen=True)
class ChaosObservation:
    """Content-bound observation for one closed local probe slot."""

    scenario_id: str
    attack: ChaosAttack
    probe_id: str
    disposition: ChaosDisposition
    unauthorized_effect_observed: bool
    authority_created: bool
    completion_created: bool
    evidence_refs: tuple[str, ...]
    reason_code: str
    schema: str = CASF_CHAOS_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CASF_CHAOS_OBSERVATION_SCHEMA:
            raise FederationChaosError("unsupported chaos observation schema")
        if type(self.attack) is not ChaosAttack or type(self.disposition) is not ChaosDisposition:
            raise FederationChaosError("observation uses a value outside the closed catalog")
        if _token(self.scenario_id, "scenario_id") != f"chaos:{self.attack.value}":
            raise FederationChaosError("observation does not bind its attack scenario")
        if _token(self.probe_id, "probe_id") != _CLOSED_PROBE_IDS[self.attack]:
            raise FederationChaosError("observation does not bind the closed probe")
        for name in (
            "unauthorized_effect_observed",
            "authority_created",
            "completion_created",
        ):
            if type(getattr(self, name)) is not bool:
                raise FederationChaosError(f"{name} must be boolean")
        refs = _tokens(
            self.evidence_refs,
            "evidence_refs",
            minimum=1,
            maximum=_MAX_EVIDENCE_ITEMS,
            content_refs=True,
        )
        if refs != tuple(sorted(refs)):
            raise FederationChaosError("evidence_refs is not canonical")
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "reason_code", _token(self.reason_code, "reason_code"))

    @property
    def observation_id(self) -> str:
        return _identity(self.to_dict(include_identity=False), prefix="chaos-observation")

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "scenario_id": self.scenario_id,
            "attack": self.attack.value,
            "probe_id": self.probe_id,
            "disposition": self.disposition.value,
            "unauthorized_effect_observed": self.unauthorized_effect_observed,
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
        fields = frozenset(
            {
                "schema",
                "scenario_id",
                "attack",
                "probe_id",
                "disposition",
                "unauthorized_effect_observed",
                "authority_created",
                "completion_created",
                "evidence_refs",
                "reason_code",
                "observation_id",
            }
        )
        _closed_mapping(value, fields, "chaos observation")
        try:
            result = cls(
                scenario_id=value["scenario_id"],
                attack=ChaosAttack(value["attack"]),
                probe_id=value["probe_id"],
                disposition=ChaosDisposition(value["disposition"]),
                unauthorized_effect_observed=value["unauthorized_effect_observed"],
                authority_created=value["authority_created"],
                completion_created=value["completion_created"],
                evidence_refs=tuple(value["evidence_refs"]),
                reason_code=value["reason_code"],
                schema=value["schema"],
            )
        except (KeyError, TypeError, ValueError, FederationChaosError) as exc:
            raise FederationChaosError("chaos observation is malformed") from exc
        if value["observation_id"] != result.observation_id:
            raise FederationChaosError("claimed observation identity mismatches")
        return result


@dataclass(frozen=True)
class ChaosValidationBinding:
    """Compact reference to a structurally verified post-merge receipt."""

    target_revision: str
    validated_revision: str
    target_tree: str
    receipt_id: str
    result_ref: str
    payload_ref: str
    attempted: bool
    passed: bool
    returncode: int
    stale: bool
    task_id: str = CASF_CHAOS_TASK_ID
    schema: str = CASF_CHAOS_VALIDATION_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CASF_CHAOS_VALIDATION_BINDING_SCHEMA:
            raise FederationChaosError("unsupported validation binding schema")
        if _token(self.task_id, "validation_task_id") != CASF_CHAOS_TASK_ID:
            raise FederationChaosError("validation receipt is for a different task")
        object.__setattr__(
            self, "target_revision", _git_oid(self.target_revision, "validation_target_revision")
        )
        object.__setattr__(
            self,
            "validated_revision",
            _git_oid(self.validated_revision, "validation_validated_revision"),
        )
        object.__setattr__(
            self, "target_tree", _git_oid(self.target_tree, "validation_target_tree")
        )
        for name in ("receipt_id", "result_ref", "payload_ref"):
            object.__setattr__(self, name, _content_ref(getattr(self, name), name))
        for name in ("attempted", "passed", "stale"):
            if type(getattr(self, name)) is not bool:
                raise FederationChaosError(f"validation {name} must be boolean")
        if isinstance(self.returncode, bool) or not isinstance(self.returncode, int):
            raise FederationChaosError("validation returncode must be an integer")
        if self.passed and (
            not self.attempted
            or self.returncode != 0
            or self.stale
            or self.validated_revision != self.target_revision
        ):
            raise FederationChaosError("validation pass is not fresh for the exact target")

    @property
    def binding_id(self) -> str:
        return _identity(self.to_dict(include_identity=False), prefix="chaos-validation")

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "task_id": self.task_id,
            "target_revision": self.target_revision,
            "validated_revision": self.validated_revision,
            "target_tree": self.target_tree,
            "receipt_id": self.receipt_id,
            "result_ref": self.result_ref,
            "payload_ref": self.payload_ref,
            "attempted": self.attempted,
            "passed": self.passed,
            "returncode": self.returncode,
            "stale": self.stale,
            "upstream_reverification_required": True,
        }
        if include_identity:
            value["binding_id"] = self.binding_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChaosValidationBinding:
        fields = frozenset(
            {
                "schema",
                "task_id",
                "target_revision",
                "validated_revision",
                "target_tree",
                "receipt_id",
                "result_ref",
                "payload_ref",
                "attempted",
                "passed",
                "returncode",
                "stale",
                "upstream_reverification_required",
                "binding_id",
            }
        )
        _closed_mapping(value, fields, "validation binding")
        if value["upstream_reverification_required"] is not True:
            raise FederationChaosError("validation binding bypasses upstream reverification")
        try:
            result = cls(
                task_id=value["task_id"],
                target_revision=value["target_revision"],
                validated_revision=value["validated_revision"],
                target_tree=value["target_tree"],
                receipt_id=value["receipt_id"],
                result_ref=value["result_ref"],
                payload_ref=value["payload_ref"],
                attempted=value["attempted"],
                passed=value["passed"],
                returncode=value["returncode"],
                stale=value["stale"],
                schema=value["schema"],
            )
        except (KeyError, TypeError, FederationChaosError) as exc:
            raise FederationChaosError("validation binding is malformed") from exc
        if value["binding_id"] != result.binding_id:
            raise FederationChaosError("validation binding identity mismatches")
        return result


def bind_post_merge_validation_evidence(
    value: Mapping[str, Any], *, identity: FederationChaosIdentity
) -> ChaosValidationBinding:
    """Bind a valid exact-tree receipt without claiming issuer authentication."""

    if type(identity) is not FederationChaosIdentity:
        raise FederationChaosError("validation binding requires exact chaos identity")
    valid, reasons = verify_post_merge_validation_evidence(
        value,
        expected_task_id=identity.task_id,
        expected_target_commit=identity.source_revision,
        expected_repository_tree_id=identity.source_tree,
    )
    if not valid:
        raise ChaosVerificationError(
            "post-merge validation evidence is invalid: " + ",".join(reasons)
        )
    try:
        return ChaosValidationBinding(
            task_id=value["task_id"],
            target_revision=value["target_commit"],
            validated_revision=value["validated_commit"],
            target_tree=value["repository_tree_id"],
            receipt_id=value["validation_receipt_id"],
            result_ref=value["validation_result_cid"],
            payload_ref=content_identity(dict(value)),
            attempted=value["attempted"],
            passed=value["passed"],
            returncode=value["returncode"],
            stale=value["stale"],
        )
    except (KeyError, TypeError, FederationChaosError) as exc:
        raise ChaosVerificationError("post-merge validation receipt cannot be bound") from exc


@dataclass(frozen=True)
class ChaosEvidenceBinding:
    """Exact evidence population consumed by the closed runner."""

    suite_id: str
    validation: ChaosValidationBinding
    rollback_revision: str
    rollback_tree: str
    rollback_generation_id: str
    capability_ids: tuple[str, ...]
    capability_statuses: tuple[ChaosCapabilityStatus, ...]
    capability_receipt_ids: tuple[str, ...]
    proof_property_ids: tuple[str, ...]
    proof_statuses: tuple[ChaosProofStatus, ...]
    proof_receipt_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    runner_id: str = CASF_CHAOS_CLOSED_RUNNER_ID
    schema: str = CASF_CHAOS_EVIDENCE_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CASF_CHAOS_EVIDENCE_BINDING_SCHEMA:
            raise FederationChaosError("unsupported evidence binding schema")
        object.__setattr__(self, "suite_id", _token(self.suite_id, "suite_id"))
        if _token(self.runner_id, "runner_id") != CASF_CHAOS_CLOSED_RUNNER_ID:
            raise FederationChaosError("evidence names an unknown closed runner")
        if type(self.validation) is not ChaosValidationBinding:
            raise FederationChaosError("evidence requires exact validation binding")
        object.__setattr__(
            self, "rollback_revision", _git_oid(self.rollback_revision, "rollback_revision")
        )
        object.__setattr__(self, "rollback_tree", _git_oid(self.rollback_tree, "rollback_tree"))
        object.__setattr__(
            self,
            "rollback_generation_id",
            _token(self.rollback_generation_id, "rollback_generation_id"),
        )
        capability_ids = _tokens(
            self.capability_ids,
            "capability_ids",
            minimum=1,
            maximum=_MAX_EVIDENCE_ITEMS,
        )
        if capability_ids != tuple(sorted(capability_ids)):
            raise FederationChaosError("capability evidence is not canonical")
        if not isinstance(self.capability_statuses, tuple) or any(
            type(item) is not ChaosCapabilityStatus for item in self.capability_statuses
        ):
            raise FederationChaosError("capability statuses must be exact immutable values")
        capability_receipts = _tokens(
            self.capability_receipt_ids,
            "capability_receipt_ids",
            minimum=len(capability_ids),
            maximum=len(capability_ids),
            content_refs=True,
            allow_empty_content_refs=True,
        )
        if not (len(capability_ids) == len(self.capability_statuses) == len(capability_receipts)):
            raise FederationChaosError("capability evidence populations differ")
        for index, status in enumerate(self.capability_statuses):
            receipt = capability_receipts[index]
            if status is ChaosCapabilityStatus.QUALIFIED and not receipt:
                raise FederationChaosError("qualified capability lacks a receipt")
        proof_properties = _tokens(
            self.proof_property_ids,
            "proof_property_ids",
            minimum=1,
            maximum=_MAX_EVIDENCE_ITEMS,
        )
        if proof_properties != tuple(sorted(proof_properties)):
            raise FederationChaosError("proof evidence is not canonical")
        if not isinstance(self.proof_statuses, tuple) or any(
            type(item) is not ChaosProofStatus for item in self.proof_statuses
        ):
            raise FederationChaosError("proof statuses must be exact immutable values")
        proof_receipts = _tokens(
            self.proof_receipt_ids,
            "proof_receipt_ids",
            minimum=len(proof_properties),
            maximum=len(proof_properties),
            content_refs=True,
            allow_empty_content_refs=True,
        )
        if not (len(proof_properties) == len(self.proof_statuses) == len(proof_receipts)):
            raise FederationChaosError("proof evidence populations differ")
        for index, status in enumerate(self.proof_statuses):
            receipt = proof_receipts[index]
            if status is ChaosProofStatus.PASSED and not receipt:
                raise FederationChaosError("passed proof lacks a receipt")
        observation_ids = _tokens(
            self.observation_ids,
            "observation_ids",
            minimum=0,
            maximum=len(ChaosAttack),
        )
        if observation_ids and len(observation_ids) != len(ChaosAttack):
            raise FederationChaosError("partial observation population is forbidden")
        object.__setattr__(self, "capability_ids", capability_ids)
        object.__setattr__(self, "capability_receipt_ids", capability_receipts)
        object.__setattr__(self, "proof_property_ids", proof_properties)
        object.__setattr__(self, "proof_receipt_ids", proof_receipts)
        object.__setattr__(self, "observation_ids", observation_ids)

    @property
    def binding_id(self) -> str:
        return _identity(self.to_dict(include_identity=False), prefix="chaos-evidence")

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "suite_id": self.suite_id,
            "runner_id": self.runner_id,
            "validation": self.validation.to_dict(),
            "rollback": {
                "revision": self.rollback_revision,
                "tree": self.rollback_tree,
                "generation_id": self.rollback_generation_id,
            },
            "capabilities": [
                {
                    "capability_id": capability_id,
                    "status": status.value,
                    "receipt_id": receipt_id,
                }
                for index, capability_id in enumerate(self.capability_ids)
                for status, receipt_id in (
                    (
                        self.capability_statuses[index],
                        self.capability_receipt_ids[index],
                    ),
                )
            ],
            "proofs": [
                {
                    "property_id": property_id,
                    "status": status.value,
                    "receipt_id": receipt_id,
                }
                for index, property_id in enumerate(self.proof_property_ids)
                for status, receipt_id in (
                    (self.proof_statuses[index], self.proof_receipt_ids[index]),
                )
            ],
            "observation_ids": list(self.observation_ids),
            "authority_created": False,
            "completion_created": False,
            "upstream_reverification_required": True,
        }
        if include_identity:
            value["binding_id"] = self.binding_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChaosEvidenceBinding:
        fields = frozenset(
            {
                "schema",
                "suite_id",
                "runner_id",
                "validation",
                "rollback",
                "capabilities",
                "proofs",
                "observation_ids",
                "authority_created",
                "completion_created",
                "upstream_reverification_required",
                "binding_id",
            }
        )
        _closed_mapping(value, fields, "chaos evidence binding")
        if (
            value["authority_created"] is not False
            or value["completion_created"] is not False
            or value["upstream_reverification_required"] is not True
        ):
            raise FederationChaosError("evidence binding has unsafe authority flags")
        rollback = value["rollback"]
        _closed_mapping(
            rollback,
            frozenset({"revision", "tree", "generation_id"}),
            "rollback target",
        )
        capabilities = value["capabilities"]
        proofs = value["proofs"]
        if not isinstance(capabilities, list) or not isinstance(proofs, list):
            raise FederationChaosError("evidence populations must be arrays")
        for item in capabilities:
            _closed_mapping(
                item,
                frozenset({"capability_id", "status", "receipt_id"}),
                "capability evidence",
            )
        for item in proofs:
            _closed_mapping(
                item,
                frozenset({"property_id", "status", "receipt_id"}),
                "proof evidence",
            )
        try:
            result = cls(
                suite_id=value["suite_id"],
                runner_id=value["runner_id"],
                validation=ChaosValidationBinding.from_dict(value["validation"]),
                rollback_revision=rollback["revision"],
                rollback_tree=rollback["tree"],
                rollback_generation_id=rollback["generation_id"],
                capability_ids=tuple(item["capability_id"] for item in capabilities),
                capability_statuses=tuple(
                    ChaosCapabilityStatus(item["status"]) for item in capabilities
                ),
                capability_receipt_ids=tuple(item["receipt_id"] for item in capabilities),
                proof_property_ids=tuple(item["property_id"] for item in proofs),
                proof_statuses=tuple(ChaosProofStatus(item["status"]) for item in proofs),
                proof_receipt_ids=tuple(item["receipt_id"] for item in proofs),
                observation_ids=tuple(value["observation_ids"]),
                schema=value["schema"],
            )
        except (KeyError, TypeError, ValueError, FederationChaosError) as exc:
            raise FederationChaosError("chaos evidence binding is malformed") from exc
        if value["binding_id"] != result.binding_id:
            raise FederationChaosError("evidence binding identity mismatches")
        return result


def _validate_observations(
    suite: FederationChaosSuite, observations: tuple[ChaosObservation, ...]
) -> None:
    if type(suite) is not FederationChaosSuite:
        raise FederationChaosError("runner requires exact FederationChaosSuite")
    if not isinstance(observations, tuple) or any(
        type(item) is not ChaosObservation for item in observations
    ):
        raise FederationChaosError("observations must be exact immutable records")
    if len(observations) != len(suite.scenarios):
        raise FederationChaosError("observation population is not exactly bounded")
    for index, scenario in enumerate(suite.scenarios):
        observation = observations[index]
        if (
            observation.scenario_id != scenario.scenario_id
            or observation.attack is not scenario.attack
            or observation.probe_id != _CLOSED_PROBE_IDS[scenario.attack]
        ):
            raise ChaosVerificationError("observation does not bind its scenario")
        if observation.disposition not in scenario.expected_dispositions:
            raise ChaosVerificationError("observation disposition is not permitted")
        if observation.unauthorized_effect_observed:
            raise ChaosVerificationError("attack produced an unauthorized effect")
        if observation.authority_created:
            raise ChaosVerificationError("attack created authority")
        if observation.completion_created:
            raise ChaosVerificationError("attack created a completion claim")


@dataclass(frozen=True)
class ChaosDiagnosticReport:
    """Result of arbitrary injected probes; categorically nonqualifying."""

    suite: FederationChaosSuite
    observations: tuple[ChaosObservation, ...]
    status: ChaosDiagnosticStatus = field(init=False)
    schema: str = CASF_CHAOS_DIAGNOSTIC_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CASF_CHAOS_DIAGNOSTIC_REPORT_SCHEMA:
            raise FederationChaosError("unsupported diagnostic report schema")
        _validate_observations(self.suite, self.observations)
        object.__setattr__(
            self,
            "status",
            ChaosDiagnosticStatus.BLOCKED
            if any(item.disposition is ChaosDisposition.BLOCKED for item in self.observations)
            else ChaosDiagnosticStatus.DIAGNOSTIC,
        )

    @property
    def qualified(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "suite_id": self.suite.suite_id,
            "status": self.status.value,
            "observation_ids": [item.observation_id for item in self.observations],
            "qualified": False,
            "authority_created": False,
            "completion_created": False,
            "promotion_eligible": False,
        }


@dataclass(frozen=True)
class ChaosReport:
    """Exact bounded evidence, never completion or promotion authority."""

    suite: FederationChaosSuite
    evidence: ChaosEvidenceBinding
    observations: tuple[ChaosObservation, ...]
    status: ChaosReportStatus = field(init=False)
    schema: str = CASF_CHAOS_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CASF_CHAOS_REPORT_SCHEMA:
            raise FederationChaosError("unsupported adversarial report schema")
        if type(self.evidence) is not ChaosEvidenceBinding:
            raise FederationChaosError("report requires exact evidence binding")
        _validate_observations(self.suite, self.observations)
        identity = self.suite.identity
        if self.evidence.suite_id != self.suite.suite_id:
            raise FederationChaosError("evidence binds a different suite")
        if (
            self.evidence.validation.target_revision != identity.source_revision
            or self.evidence.validation.target_tree != identity.source_tree
        ):
            raise FederationChaosError("validation binds a different source tree")
        if self.evidence.rollback_revision == identity.source_revision:
            raise FederationChaosError("rollback target cannot be the validated revision")
        if self.evidence.rollback_generation_id == identity.generation_id:
            raise FederationChaosError("rollback target cannot name the active generation")
        if self.evidence.capability_ids != identity.capability_ids:
            raise FederationChaosError("capability evidence does not cover the identity")
        if self.evidence.observation_ids != tuple(
            item.observation_id for item in self.observations
        ):
            raise FederationChaosError("evidence binds a different observation population")
        object.__setattr__(self, "status", ChaosReportStatus.BLOCKED)

    @property
    def qualified(self) -> bool:
        """Local evidence is never independently qualified by this boundary."""

        return False

    @property
    def promotion_eligible(self) -> bool:
        return False

    @property
    def report_id(self) -> str:
        return _identity(self.to_dict(include_identity=False), prefix="adversarial-report")

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "runner_id": CASF_CHAOS_CLOSED_RUNNER_ID,
            "suite": self.suite.to_dict(),
            "evidence": self.evidence.to_dict(),
            "status": self.status.value,
            "observations": [item.to_dict() for item in self.observations],
            "bounded": True,
            "authority_created": False,
            "completion_created": False,
            "promotion_eligible": False,
            "local_qualification_available": False,
            "upstream_reverification_required": True,
        }
        if include_identity:
            value["report_id"] = self.report_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChaosReport:
        fields = frozenset(
            {
                "schema",
                "runner_id",
                "suite",
                "evidence",
                "status",
                "observations",
                "bounded",
                "authority_created",
                "completion_created",
                "promotion_eligible",
                "local_qualification_available",
                "upstream_reverification_required",
                "report_id",
            }
        )
        _closed_mapping(value, fields, "adversarial report")
        if (
            value["runner_id"] != CASF_CHAOS_CLOSED_RUNNER_ID
            or value["bounded"] is not True
            or value["authority_created"] is not False
            or value["completion_created"] is not False
            or value["promotion_eligible"] is not False
            or value["local_qualification_available"] is not False
            or value["upstream_reverification_required"] is not True
        ):
            raise FederationChaosError("adversarial report has unsafe authority flags")
        try:
            result = cls(
                suite=FederationChaosSuite.from_dict(value["suite"]),
                evidence=ChaosEvidenceBinding.from_dict(value["evidence"]),
                observations=tuple(
                    ChaosObservation.from_dict(item) for item in value["observations"]
                ),
                schema=value["schema"],
            )
        except (KeyError, TypeError, FederationChaosError) as exc:
            raise FederationChaosError("adversarial report is malformed") from exc
        if value["status"] != result.status.value:
            raise FederationChaosError("adversarial report status is self-asserted")
        if value["report_id"] != result.report_id:
            raise FederationChaosError("adversarial report identity mismatches")
        return result


def build_federation_chaos_suite(
    identity: FederationChaosIdentity,
) -> FederationChaosSuite:
    """Build the complete deterministic attack catalog for one fenced run."""

    if type(identity) is not FederationChaosIdentity:
        raise FederationChaosError("suite requires exact FederationChaosIdentity")
    return FederationChaosSuite(
        identity=identity,
        scenarios=tuple(
            ChaosScenario(
                scenario_id=f"chaos:{attack.value}",
                attack=attack,
                domain=_ATTACK_DOMAINS[attack],
            )
            for attack in ChaosAttack
        ),
    )


def build_chaos_observation(
    scenario: ChaosScenario,
    *,
    disposition: ChaosDisposition,
    evidence_refs: tuple[str, ...],
    reason_code: str,
    unauthorized_effect_observed: bool = False,
    authority_created: bool = False,
    completion_created: bool = False,
) -> ChaosObservation:
    """Build one observation bound to the closed local probe-slot catalog."""

    if type(scenario) is not ChaosScenario:
        raise FederationChaosError("observation requires exact ChaosScenario")
    return ChaosObservation(
        scenario_id=scenario.scenario_id,
        attack=scenario.attack,
        probe_id=_CLOSED_PROBE_IDS[scenario.attack],
        disposition=disposition,
        unauthorized_effect_observed=unauthorized_effect_observed,
        authority_created=authority_created,
        completion_created=completion_created,
        evidence_refs=evidence_refs,
        reason_code=reason_code,
    )


def run_federation_chaos_suite(
    suite: FederationChaosSuite,
    probe: Callable[[ChaosScenario], ChaosObservation],
) -> ChaosDiagnosticReport:
    """Run injected probes as diagnostics that can never qualify CASF."""

    if type(suite) is not FederationChaosSuite:
        raise FederationChaosError("suite must be exact FederationChaosSuite")
    if not callable(probe):
        raise FederationChaosError("probe must be callable")
    return ChaosDiagnosticReport(
        suite=suite,
        observations=tuple(probe(scenario) for scenario in suite.scenarios),
    )


def run_closed_federation_chaos_suite(
    suite: FederationChaosSuite,
    evidence: ChaosEvidenceBinding,
) -> ChaosReport:
    """Emit exact blocked evidence without accepting a probe callback.

    This two-file hermetic boundary cannot authenticate the external receipt
    issuer or independently execute live capabilities.  It therefore creates
    its own canonical ``BLOCKED`` observation population and never reports a
    local qualification.  An upstream authority may re-run and reverify the
    referenced evidence, but cannot use this report itself as promotion
    authority.
    """

    if type(suite) is not FederationChaosSuite:
        raise FederationChaosError("suite must be exact FederationChaosSuite")
    if type(evidence) is not ChaosEvidenceBinding:
        raise FederationChaosError("evidence must be exact ChaosEvidenceBinding")
    evidence_refs = tuple(
        sorted(
            {
                evidence.validation.payload_ref,
                evidence.validation.result_ref,
                *(item for item in evidence.capability_receipt_ids if item),
                *(item for item in evidence.proof_receipt_ids if item),
            }
        )
    )
    observations = tuple(
        build_chaos_observation(
            scenario,
            disposition=ChaosDisposition.BLOCKED,
            evidence_refs=evidence_refs,
            reason_code="external_receipts_require_upstream_reverification",
        )
        for scenario in suite.scenarios
    )
    observation_ids = tuple(item.observation_id for item in observations)
    if evidence.observation_ids and evidence.observation_ids != observation_ids:
        raise FederationChaosError("evidence prebinds a different observation population")
    if not evidence.observation_ids:
        evidence = ChaosEvidenceBinding(
            suite_id=evidence.suite_id,
            validation=evidence.validation,
            rollback_revision=evidence.rollback_revision,
            rollback_tree=evidence.rollback_tree,
            rollback_generation_id=evidence.rollback_generation_id,
            capability_ids=evidence.capability_ids,
            capability_statuses=evidence.capability_statuses,
            capability_receipt_ids=evidence.capability_receipt_ids,
            proof_property_ids=evidence.proof_property_ids,
            proof_statuses=evidence.proof_statuses,
            proof_receipt_ids=evidence.proof_receipt_ids,
            observation_ids=observation_ids,
        )
    return ChaosReport(suite=suite, evidence=evidence, observations=observations)


__all__ = [
    "CASF_CHAOS_CLOSED_RUNNER_ID",
    "CASF_CHAOS_DIAGNOSTIC_REPORT_SCHEMA",
    "CASF_CHAOS_EVIDENCE_BINDING_SCHEMA",
    "CASF_CHAOS_IDENTITY_SCHEMA",
    "CASF_CHAOS_LOCAL_QUALIFICATION_AVAILABLE",
    "CASF_CHAOS_OBSERVATION_SCHEMA",
    "CASF_CHAOS_PROBE_CATALOG_ID",
    "CASF_CHAOS_REPORT_SCHEMA",
    "CASF_CHAOS_SCENARIO_SCHEMA",
    "CASF_CHAOS_SUITE_SCHEMA",
    "CASF_CHAOS_TASK_ID",
    "CASF_CHAOS_VALIDATION_BINDING_SCHEMA",
    "ChaosAttack",
    "ChaosCapabilityStatus",
    "ChaosDiagnosticReport",
    "ChaosDiagnosticStatus",
    "ChaosDisposition",
    "ChaosEvidenceBinding",
    "ChaosObservation",
    "ChaosProofStatus",
    "ChaosReport",
    "ChaosReportStatus",
    "ChaosScenario",
    "ChaosValidationBinding",
    "ChaosVerificationError",
    "FederationChaosError",
    "FederationChaosIdentity",
    "FederationChaosSuite",
    "bind_post_merge_validation_evidence",
    "build_chaos_observation",
    "build_federation_chaos_suite",
    "run_federation_chaos_suite",
    "run_closed_federation_chaos_suite",
]
