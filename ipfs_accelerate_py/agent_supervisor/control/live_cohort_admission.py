"""ASEH-015 live shadow/canary cohort admission.

The admission service seals one bounded UTC enrollment deadline, requires ten
distinct newly encountered live tasks before an evidence-qualified cohort,
proves live provenance, keeps shadow non-mutating and strictly before canary,
and rejects fixture or simulation substitution.  Honest shortfalls emit a
typed insufficiency or non-admission receipt.  Schema authority here cannot
attest a live measurement or authorize promotion.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity


LIVE_COHORT_MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-live-cohort-manifest@1"
)
LIVE_COHORT_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-live-cohort-admission-policy@1"
)
LIVE_COHORT_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-live-cohort-admission@1"
)
LIVE_COHORT_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-live-cohort-admission-receipt@1"
)
LIVE_TASK_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-live-task-evidence@1"
)
LIVE_TASK_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-live-task-admission-receipt@1"
)

PROGRAM_ID: Final[str] = "agent-supervisor-efficiency-and-state-hardening-v1"
OBJECTIVE_ID: Final[str] = "ASEH-G020"
TASK_ID: Final[str] = "ASEH-015"
POLICY_IDENTITY: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-supervisor-policy@1"
)
PLAN_ROOT_CID: Final[str] = (
    "baguqeeray5xunujqwmh4axot3sbeatuger45zdsfkrshdvwaqewv5qnw2xnq"
)
POPULATION_KIND: Final[str] = "new_live_shadow_canary"
LIVE_MINIMUM: Final[int] = 10
LIVE_ENROLLMENT_DEADLINE_MAXIMUM_DAYS: Final[int] = 30
SEALED_AT: Final[str] = "2026-09-02T00:00:00Z"
ENROLLMENT_DEADLINE: Final[str] = "2026-10-02T00:00:00Z"
PLANNING_COMMIT: Final[str] = "755f45475cc2d13dacd8b330036c1d597afeddde"
PLANNING_TREE: Final[str] = "729da9f8293ecfa046a0136381a3d3808f9ed140"

CID_RE: Final[re.Pattern[str]] = re.compile(r"^b[a-z2-7]{20,}$")
GIT_OID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
UTC_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
HERMETIC_FIXTURE_RE: Final[re.Pattern[str]] = re.compile(r"^aseh-h[0-9]{2,}$")

REQUIRED_PROVENANCE_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "task_cid",
    "objective_id",
    "objective_revision",
    "repository_commit",
    "repository_tree",
    "policy_identity",
    "validator_command",
    "validator_result_digest",
    "raw_log_reference",
    "receipt_cid",
    "truth_state",
)
LIVE_COHORT_GATES: Final[tuple[str, ...]] = (
    "enrollment_deadline",
    "distinct_new_tasks",
    "live_provenance",
    "shadow_before_canary",
    "reject_simulation",
    "reject_fixture",
)
LIVE_TRUTH_STATES: Final[frozenset[str]] = frozenset({"observed", "verified", "measured"})
FORBIDDEN_LIVE_TRUTH_STATES: Final[frozenset[str]] = frozenset(
    {"simulated", "unavailable", "estimated", "attempted"}
)
PROVIDER_USAGE_TRUTH_STATES: Final[frozenset[str]] = frozenset(
    {"measured", "observed", "verified"}
)
ALLOWED_SOURCE_KINDS: Final[frozenset[str]] = frozenset(
    {"live_supervisor_task", "newly_encountered_real_task"}
)
FORBIDDEN_SOURCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "fixture",
        "test_fixture",
        "golden",
        "hermetic_fixture",
        "hermetic_development",
        "historical_replay",
        "historical_exact_tree_replay",
        "simulation",
        "simulated",
    }
)
HISTORICAL_TASK_IDS: Final[frozenset[str]] = frozenset(
    {
        "ASEH-000",
        "ASEH-001",
        *[f"ASEH-{index:03d}" for index in range(10, 14)],
        *[f"ASEH-{index:03d}" for index in range(20, 25)],
        *[f"ASEH-{index:03d}" for index in range(30, 35)],
        *[f"ASEH-{index:03d}" for index in range(40, 46)],
        *[f"ASEH-{index:03d}" for index in range(50, 56)],
    }
)
CLOSED_DISPOSITIONS: Final[tuple[str, ...]] = (
    "evidence_qualified",
    "insufficient_evidence",
    "not_admitted",
    "deadline_elapsed",
)
CLOSED_TASK_DISPOSITIONS: Final[tuple[str, ...]] = (
    "admitted_shadow",
    "admitted_canary",
    "not_admitted",
    "insufficient_evidence",
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
MANIFEST_PATH: Final[Path] = (
    _REPO_ROOT
    / "benchmarks"
    / "agent_supervisor"
    / "efficiency_state_hardening"
    / "live_cohort_manifest.json"
)


class LiveCohortAdmissionError(ValueError):
    """Malformed live-cohort policy, evidence, or receipt."""


class ExecutionMode(str, Enum):
    SHADOW = "shadow"
    CANARY = "canary"


class CohortDisposition(str, Enum):
    EVIDENCE_QUALIFIED = "evidence_qualified"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_ADMITTED = "not_admitted"
    DEADLINE_ELAPSED = "deadline_elapsed"


class TaskDisposition(str, Enum):
    ADMITTED_SHADOW = "admitted_shadow"
    ADMITTED_CANARY = "admitted_canary"
    NOT_ADMITTED = "not_admitted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class LiveCohortReason(str, Enum):
    DEADLINE_ELAPSED = "deadline_elapsed"
    INSUFFICIENT_DISTINCT_LIVE_TASKS = "insufficient_distinct_live_tasks"
    DUPLICATE_TASK = "duplicate_task"
    NOT_NEWLY_ENCOUNTERED = "not_newly_encountered"
    HISTORICAL_SUBSTITUTION = "historical_substitution"
    HERMETIC_SUBSTITUTION = "hermetic_substitution"
    FIXTURE_SUBSTITUTION = "fixture_substitution"
    SIMULATION_SUBSTITUTION = "simulation_substitution"
    LIVE_PROVENANCE_MISSING = "live_provenance_missing"
    TRUTH_STATE_NOT_LIVE = "truth_state_not_live"
    SHADOW_BYPASS = "shadow_bypass"
    SHADOW_MUTATING = "shadow_mutating"
    CANARY_WITHOUT_SHADOW = "canary_without_shadow"
    CANARY_WITHOUT_SAFETY = "canary_without_safety"
    PROVIDER_EVIDENCE_MISSING = "provider_evidence_missing"
    VERIFIER_EVIDENCE_MISSING = "verifier_evidence_missing"
    NOT_LIVE = "not_live"
    POPULATION_NOT_LIVE = "population_not_live"
    NOT_YET_MEASURED = "not_yet_measured"
    SAFETY_NOT_ADMITTED = "safety_not_admitted"


def _text(value: Any, name: str, *, required: bool = True, maximum: int = 512) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise LiveCohortAdmissionError(f"{name} must be a string")
    else:
        text = value.strip()
    if "\x00" in text:
        raise LiveCohortAdmissionError(f"{name} must not contain NUL")
    if required and not text:
        raise LiveCohortAdmissionError(f"{name} must be a non-empty string")
    if len(text.encode("utf-8")) > maximum:
        raise LiveCohortAdmissionError(f"{name} is too large")
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise LiveCohortAdmissionError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str, *, minimum: int = 0, maximum: int = 10**18) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LiveCohortAdmissionError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise LiveCohortAdmissionError(f"{name} is out of bounds")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    if CID_RE.fullmatch(text) is None:
        raise LiveCohortAdmissionError(f"{name} must be a CIDv1")
    return text


def _digest(value: Any, name: str) -> str:
    text = _text(value, name, maximum=80)
    if DIGEST_RE.fullmatch(text) is None:
        raise LiveCohortAdmissionError(f"{name} must be a sha256 digest")
    return text


def _git_oid(value: Any, name: str) -> str:
    text = _text(value, name, maximum=40)
    if GIT_OID_RE.fullmatch(text) is None:
        raise LiveCohortAdmissionError(f"{name} must be a git object id")
    return text


def parse_utc(value: Any, *, name: str) -> datetime:
    text = _text(value, name)
    if UTC_RE.fullmatch(text) is None:
        raise LiveCohortAdmissionError(f"{name} must be a sealed UTC timestamp")
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise LiveCohortAdmissionError("timestamp must be timezone-aware")
    as_utc = value.astimezone(timezone.utc).replace(microsecond=0)
    return as_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def pretty_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _body(payload: Mapping[str, Any], *excluded: str) -> dict[str, Any]:
    skip = set(excluded)
    return {key: value for key, value in payload.items() if key not in skip}


def _identity_of(payload: Mapping[str, Any], *excluded: str) -> str:
    return content_identity(_body(payload, *excluded))


def seal_enrollment_deadline(*, sealed_at: str, enrollment_deadline: str) -> str:
    """Bind an immutable UTC deadline no later than 30 calendar days after seal."""

    start = parse_utc(sealed_at, name="sealed_at")
    deadline = parse_utc(enrollment_deadline, name="enrollment_deadline")
    if deadline <= start:
        raise LiveCohortAdmissionError("enrollment deadline must be after sealed_at")
    maximum = start + timedelta(days=LIVE_ENROLLMENT_DEADLINE_MAXIMUM_DAYS)
    if deadline > maximum:
        raise LiveCohortAdmissionError(
            "enrollment deadline exceeds the 30 calendar day bound"
        )
    return format_utc(deadline)


def _enum(value: Any, enum_cls: type[Enum], *, name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    text = _text(value, name)
    try:
        return enum_cls(text)
    except ValueError as exc:
        raise LiveCohortAdmissionError(f"{name} is not a closed {enum_cls.__name__} value") from exc


@dataclass(frozen=True)
class LiveCohortAdmissionPolicy:
    """Immutable live-cohort admission policy.  Gates cannot be weakened."""

    sealed_at: str
    enrollment_deadline: str
    enrollment_deadline_immutable: bool = True
    minimum: int = LIVE_MINIMUM
    enrollment_deadline_maximum_days: int = LIVE_ENROLLMENT_DEADLINE_MAXIMUM_DAYS
    shadow_before_canary: bool = True
    canary_requires_safety_admission: bool = True
    fixtures_satisfy_live: bool = False
    simulation_satisfies_live: bool = False
    schema: str = LIVE_COHORT_POLICY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "sealed_at", _text(self.sealed_at, "sealed_at"))
        object.__setattr__(
            self,
            "enrollment_deadline",
            seal_enrollment_deadline(
                sealed_at=self.sealed_at,
                enrollment_deadline=self.enrollment_deadline,
            ),
        )
        object.__setattr__(
            self,
            "enrollment_deadline_immutable",
            _bool(self.enrollment_deadline_immutable, "enrollment_deadline_immutable"),
        )
        if self.enrollment_deadline_immutable is not True:
            raise LiveCohortAdmissionError("enrollment deadline must be immutable once sealed")
        object.__setattr__(self, "minimum", _int(self.minimum, "minimum", minimum=LIVE_MINIMUM))
        object.__setattr__(
            self,
            "enrollment_deadline_maximum_days",
            _int(
                self.enrollment_deadline_maximum_days,
                "enrollment_deadline_maximum_days",
                minimum=1,
                maximum=LIVE_ENROLLMENT_DEADLINE_MAXIMUM_DAYS,
            ),
        )
        if self.enrollment_deadline_maximum_days != LIVE_ENROLLMENT_DEADLINE_MAXIMUM_DAYS:
            raise LiveCohortAdmissionError(
                "enrollment_deadline_maximum_days cannot differ from the sealed 30-day bound"
            )
        object.__setattr__(
            self, "shadow_before_canary", _bool(self.shadow_before_canary, "shadow_before_canary")
        )
        if self.shadow_before_canary is not True:
            raise LiveCohortAdmissionError("shadow_before_canary cannot be disabled")
        object.__setattr__(
            self,
            "canary_requires_safety_admission",
            _bool(self.canary_requires_safety_admission, "canary_requires_safety_admission"),
        )
        if self.canary_requires_safety_admission is not True:
            raise LiveCohortAdmissionError("canary_requires_safety_admission cannot be disabled")
        object.__setattr__(
            self, "fixtures_satisfy_live", _bool(self.fixtures_satisfy_live, "fixtures_satisfy_live")
        )
        if self.fixtures_satisfy_live is not False:
            raise LiveCohortAdmissionError("fixtures cannot satisfy live admission")
        object.__setattr__(
            self,
            "simulation_satisfies_live",
            _bool(self.simulation_satisfies_live, "simulation_satisfies_live"),
        )
        if self.simulation_satisfies_live is not False:
            raise LiveCohortAdmissionError("simulation cannot satisfy live admission")
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != LIVE_COHORT_POLICY_SCHEMA:
            raise LiveCohortAdmissionError("unsupported live cohort admission policy schema")

    @property
    def policy_identity(self) -> str:
        return content_identity(self.to_dict())

    @property
    def sealed_at_utc(self) -> datetime:
        return parse_utc(self.sealed_at, name="sealed_at")

    @property
    def enrollment_deadline_utc(self) -> datetime:
        return parse_utc(self.enrollment_deadline, name="enrollment_deadline")

    def to_dict(self) -> dict[str, Any]:
        return {
            "canary_requires_safety_admission": self.canary_requires_safety_admission,
            "enrollment_deadline": self.enrollment_deadline,
            "enrollment_deadline_immutable": self.enrollment_deadline_immutable,
            "enrollment_deadline_maximum_days": self.enrollment_deadline_maximum_days,
            "fixtures_satisfy_live": self.fixtures_satisfy_live,
            "minimum": self.minimum,
            "schema": self.schema,
            "sealed_at": self.sealed_at,
            "shadow_before_canary": self.shadow_before_canary,
            "simulation_satisfies_live": self.simulation_satisfies_live,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LiveCohortAdmissionPolicy":
        if not isinstance(payload, Mapping):
            raise LiveCohortAdmissionError("live cohort admission policy must be an object")
        return cls(
            sealed_at=payload.get("sealed_at", ""),
            enrollment_deadline=payload.get("enrollment_deadline", ""),
            enrollment_deadline_immutable=payload.get("enrollment_deadline_immutable", True),
            minimum=payload.get("minimum", LIVE_MINIMUM),
            enrollment_deadline_maximum_days=payload.get(
                "enrollment_deadline_maximum_days", LIVE_ENROLLMENT_DEADLINE_MAXIMUM_DAYS
            ),
            shadow_before_canary=payload.get("shadow_before_canary", True),
            canary_requires_safety_admission=payload.get(
                "canary_requires_safety_admission", True
            ),
            fixtures_satisfy_live=payload.get("fixtures_satisfy_live", False),
            simulation_satisfies_live=payload.get("simulation_satisfies_live", False),
            schema=payload.get("schema", LIVE_COHORT_POLICY_SCHEMA),
        )


def sealed_policy() -> LiveCohortAdmissionPolicy:
    return LiveCohortAdmissionPolicy(
        sealed_at=SEALED_AT,
        enrollment_deadline=ENROLLMENT_DEADLINE,
    )


@dataclass(frozen=True)
class LiveTaskEvidence:
    """One newly encountered live task with provider and verifier evidence."""

    task_id: str
    task_cid: str
    objective_id: str
    objective_revision: str
    repository_commit: str
    repository_tree: str
    policy_identity: str
    validator_command: str
    validator_result_digest: str
    raw_log_reference: str
    truth_state: str
    provider_id: str
    provider_request_id: str
    provider_usage_truth_state: str
    execution_mode: str
    source_kind: str
    live: bool = True
    newly_encountered: bool = True
    mutating: bool = False
    population_kind: str = POPULATION_KIND
    safety_admitted: bool = False
    fixture_id: str = ""
    simulated: bool = False
    historical_replay: bool = False
    hermetic: bool = False
    shadow_receipt_cid: str = ""
    schema: str = LIVE_TASK_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id", maximum=64))
        object.__setattr__(self, "task_cid", _cid(self.task_cid, "task_cid"))
        object.__setattr__(self, "objective_id", _text(self.objective_id, "objective_id", maximum=32))
        object.__setattr__(
            self, "objective_revision", _cid(self.objective_revision, "objective_revision")
        )
        object.__setattr__(
            self, "repository_commit", _git_oid(self.repository_commit, "repository_commit")
        )
        object.__setattr__(
            self, "repository_tree", _git_oid(self.repository_tree, "repository_tree")
        )
        object.__setattr__(
            self, "policy_identity", _text(self.policy_identity, "policy_identity")
        )
        object.__setattr__(
            self,
            "validator_command",
            _text(self.validator_command, "validator_command", maximum=512),
        )
        object.__setattr__(
            self,
            "validator_result_digest",
            _digest(self.validator_result_digest, "validator_result_digest"),
        )
        object.__setattr__(
            self, "raw_log_reference", _digest(self.raw_log_reference, "raw_log_reference")
        )
        object.__setattr__(self, "truth_state", _text(self.truth_state, "truth_state", maximum=32))
        object.__setattr__(self, "provider_id", _text(self.provider_id, "provider_id", maximum=128))
        object.__setattr__(
            self,
            "provider_request_id",
            _text(self.provider_request_id, "provider_request_id", maximum=256),
        )
        object.__setattr__(
            self,
            "provider_usage_truth_state",
            _text(self.provider_usage_truth_state, "provider_usage_truth_state", maximum=32),
        )
        mode = _enum(self.execution_mode, ExecutionMode, name="execution_mode")
        object.__setattr__(self, "execution_mode", mode.value)
        object.__setattr__(self, "source_kind", _text(self.source_kind, "source_kind", maximum=64))
        object.__setattr__(self, "live", _bool(self.live, "live"))
        object.__setattr__(
            self, "newly_encountered", _bool(self.newly_encountered, "newly_encountered")
        )
        object.__setattr__(self, "mutating", _bool(self.mutating, "mutating"))
        object.__setattr__(
            self, "population_kind", _text(self.population_kind, "population_kind", maximum=64)
        )
        object.__setattr__(self, "safety_admitted", _bool(self.safety_admitted, "safety_admitted"))
        object.__setattr__(
            self, "fixture_id", _text(self.fixture_id, "fixture_id", required=False, maximum=64)
        )
        object.__setattr__(self, "simulated", _bool(self.simulated, "simulated"))
        object.__setattr__(
            self, "historical_replay", _bool(self.historical_replay, "historical_replay")
        )
        object.__setattr__(self, "hermetic", _bool(self.hermetic, "hermetic"))
        object.__setattr__(
            self,
            "shadow_receipt_cid",
            _text(self.shadow_receipt_cid, "shadow_receipt_cid", required=False),
        )
        if self.shadow_receipt_cid and CID_RE.fullmatch(self.shadow_receipt_cid) is None:
            raise LiveCohortAdmissionError("shadow_receipt_cid must be a CIDv1")
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != LIVE_TASK_EVIDENCE_SCHEMA:
            raise LiveCohortAdmissionError("unsupported live task evidence schema")
        for field in REQUIRED_PROVENANCE_FIELDS:
            if field == "receipt_cid":
                continue
            if not getattr(self, field):
                raise LiveCohortAdmissionError(f"missing provenance field: {field}")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "execution_mode": self.execution_mode,
            "fixture_id": self.fixture_id,
            "hermetic": self.hermetic,
            "historical_replay": self.historical_replay,
            "live": self.live,
            "mutating": self.mutating,
            "newly_encountered": self.newly_encountered,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "policy_identity": self.policy_identity,
            "population_kind": self.population_kind,
            "provider_id": self.provider_id,
            "provider_request_id": self.provider_request_id,
            "provider_usage_truth_state": self.provider_usage_truth_state,
            "raw_log_reference": self.raw_log_reference,
            "repository_commit": self.repository_commit,
            "repository_tree": self.repository_tree,
            "safety_admitted": self.safety_admitted,
            "schema": self.schema,
            "shadow_receipt_cid": self.shadow_receipt_cid,
            "simulated": self.simulated,
            "source_kind": self.source_kind,
            "task_cid": self.task_cid,
            "task_id": self.task_id,
            "truth_state": self.truth_state,
            "validator_command": self.validator_command,
            "validator_result_digest": self.validator_result_digest,
        }
        identity = _identity_of(payload)
        payload["identity"] = identity
        payload["receipt_cid"] = identity
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LiveTaskEvidence":
        if not isinstance(payload, Mapping):
            raise LiveCohortAdmissionError("live task evidence must be an object")
        evidence = cls(
            task_id=payload.get("task_id", ""),
            task_cid=payload.get("task_cid", ""),
            objective_id=payload.get("objective_id", ""),
            objective_revision=payload.get("objective_revision", ""),
            repository_commit=payload.get("repository_commit", ""),
            repository_tree=payload.get("repository_tree", ""),
            policy_identity=payload.get("policy_identity", ""),
            validator_command=payload.get("validator_command", ""),
            validator_result_digest=payload.get("validator_result_digest", ""),
            raw_log_reference=payload.get("raw_log_reference", ""),
            truth_state=payload.get("truth_state", ""),
            provider_id=payload.get("provider_id", ""),
            provider_request_id=payload.get("provider_request_id", ""),
            provider_usage_truth_state=payload.get("provider_usage_truth_state", ""),
            execution_mode=payload.get("execution_mode", ""),
            source_kind=payload.get("source_kind", ""),
            live=payload.get("live", True),
            newly_encountered=payload.get("newly_encountered", True),
            mutating=payload.get("mutating", False),
            population_kind=payload.get("population_kind", POPULATION_KIND),
            safety_admitted=payload.get("safety_admitted", False),
            fixture_id=payload.get("fixture_id", ""),
            simulated=payload.get("simulated", False),
            historical_replay=payload.get("historical_replay", False),
            hermetic=payload.get("hermetic", False),
            shadow_receipt_cid=payload.get("shadow_receipt_cid", ""),
            schema=payload.get("schema", LIVE_TASK_EVIDENCE_SCHEMA),
        )
        encoded = evidence.to_dict()
        claimed_identity = payload.get("identity")
        claimed_receipt = payload.get("receipt_cid")
        if claimed_identity not in (None, "", encoded["identity"]):
            raise LiveCohortAdmissionError("forged live task evidence identity")
        if claimed_receipt not in (None, "", encoded["receipt_cid"]):
            raise LiveCohortAdmissionError("forged live task receipt_cid")
        return evidence


def live_task_evidence_payload(
    index: int,
    *,
    execution_mode: str = ExecutionMode.SHADOW.value,
    mutating: bool | None = None,
    safety_admitted: bool = False,
    shadow_receipt_cid: str = "",
    **overrides: Any,
) -> dict[str, Any]:
    """Build well-formed live evidence for admission tests.  Not a sealed fixture."""

    if index < 1:
        raise LiveCohortAdmissionError("live task index must be at least 1")
    mode = _enum(execution_mode, ExecutionMode, name="execution_mode").value
    if mutating is None:
        mutating = mode == ExecutionMode.CANARY.value
    task_id = f"ASEH-LIVE-{index:03d}"
    digest_seed = f"aseh-015-live-{index:03d}-{mode}"
    payload: dict[str, Any] = {
        "execution_mode": mode,
        "fixture_id": "",
        "hermetic": False,
        "historical_replay": False,
        "live": True,
        "mutating": mutating,
        "newly_encountered": True,
        "objective_id": OBJECTIVE_ID,
        "objective_revision": PLAN_ROOT_CID,
        "policy_identity": POLICY_IDENTITY,
        "population_kind": POPULATION_KIND,
        "provider_id": "grok_cli",
        "provider_request_id": f"live-provider-{index:03d}",
        "provider_usage_truth_state": "measured",
        "raw_log_reference": _sha256_label(digest_seed + ":log"),
        "repository_commit": PLANNING_COMMIT,
        "repository_tree": PLANNING_TREE,
        "safety_admitted": safety_admitted,
        "schema": LIVE_TASK_EVIDENCE_SCHEMA,
        "shadow_receipt_cid": shadow_receipt_cid,
        "simulated": False,
        "source_kind": "newly_encountered_real_task",
        "task_cid": content_identity({"aseh-015-live-task": task_id}),
        "task_id": task_id,
        "truth_state": "observed",
        "validator_command": (
            "python3 -m pytest -q "
            "test/api/agent_supervisor/efficiency_state_hardening/"
            "test_live_cohort_admission.py"
        ),
        "validator_result_digest": _sha256_label(digest_seed + ":validator"),
    }
    payload.update(overrides)
    return LiveTaskEvidence.from_dict(payload).to_dict()


def _sha256_label(label: str) -> str:
    identity = content_identity({"aseh-015": label})
    return "sha256:" + identity.encode("ascii").hex()[:64]


@dataclass(frozen=True)
class TaskAdmissionReceipt:
    """Typed per-task live admission or non-admission receipt."""

    task_id: str
    task_cid: str
    disposition: str
    admitted: bool
    live: bool
    execution_mode: str
    mutating: bool
    mutation_permitted: bool
    reasons: tuple[str, ...]
    shadow_receipt_cid: str = ""
    policy_identity: str = ""
    schema: str = LIVE_TASK_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id", maximum=64))
        object.__setattr__(self, "task_cid", _cid(self.task_cid, "task_cid"))
        disposition = _enum(self.disposition, TaskDisposition, name="disposition")
        object.__setattr__(self, "disposition", disposition.value)
        object.__setattr__(self, "admitted", _bool(self.admitted, "admitted"))
        object.__setattr__(self, "live", _bool(self.live, "live"))
        mode = _enum(self.execution_mode, ExecutionMode, name="execution_mode")
        object.__setattr__(self, "execution_mode", mode.value)
        object.__setattr__(self, "mutating", _bool(self.mutating, "mutating"))
        object.__setattr__(
            self, "mutation_permitted", _bool(self.mutation_permitted, "mutation_permitted")
        )
        reasons = tuple(_text(item, "reason", maximum=128) for item in self.reasons)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "shadow_receipt_cid",
            _text(self.shadow_receipt_cid, "shadow_receipt_cid", required=False),
        )
        object.__setattr__(
            self, "policy_identity", _text(self.policy_identity, "policy_identity", required=False)
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != LIVE_TASK_RECEIPT_SCHEMA:
            raise LiveCohortAdmissionError("unsupported live task admission receipt schema")
        expected_admitted = self.disposition in {
            TaskDisposition.ADMITTED_SHADOW.value,
            TaskDisposition.ADMITTED_CANARY.value,
        }
        if self.admitted != expected_admitted:
            raise LiveCohortAdmissionError("admitted is derived from the task disposition")
        if self.admitted and not self.live:
            raise LiveCohortAdmissionError("admitted live tasks must prove live provenance")
        if self.mutation_permitted and self.disposition != TaskDisposition.ADMITTED_CANARY.value:
            raise LiveCohortAdmissionError("mutation is permitted only for admitted canary tasks")
        if self.mutation_permitted and not self.mutating:
            raise LiveCohortAdmissionError("mutation_permitted requires mutating canary evidence")

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "disposition": self.disposition,
            "execution_mode": self.execution_mode,
            "live": self.live,
            "mutating": self.mutating,
            "mutation_permitted": self.mutation_permitted,
            "policy_identity": self.policy_identity,
            "reasons": list(self.reasons),
            "schema": self.schema,
            "shadow_receipt_cid": self.shadow_receipt_cid,
            "task_cid": self.task_cid,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskAdmissionReceipt":
        if not isinstance(payload, Mapping):
            raise LiveCohortAdmissionError("task admission receipt must be an object")
        receipt = cls(
            task_id=payload.get("task_id", ""),
            task_cid=payload.get("task_cid", ""),
            disposition=payload.get("disposition", ""),
            admitted=payload.get("admitted", False),
            live=payload.get("live", False),
            execution_mode=payload.get("execution_mode", ""),
            mutating=payload.get("mutating", False),
            mutation_permitted=payload.get("mutation_permitted", False),
            reasons=tuple(payload.get("reasons") or ()),
            shadow_receipt_cid=payload.get("shadow_receipt_cid", ""),
            policy_identity=payload.get("policy_identity", ""),
            schema=payload.get("schema", LIVE_TASK_RECEIPT_SCHEMA),
        )
        claimed = payload.get("receipt_id")
        if claimed not in (None, "", receipt.receipt_id):
            raise LiveCohortAdmissionError("forged task admission receipt_id")
        return receipt


@dataclass(frozen=True)
class CohortAdmissionReceipt:
    """Typed cohort qualification, insufficiency, or non-admission receipt."""

    disposition: str
    admitted: bool
    live: bool
    qualification: bool
    count: int
    minimum: int
    distinct_task_ids: tuple[str, ...]
    enrollment_deadline: str
    enrollment_deadline_immutable: bool
    shadow_admitted: bool
    canary_admitted: bool
    canary_mutation_permitted: bool
    population_status: str
    reasons: tuple[str, ...]
    task_receipts: tuple[Mapping[str, Any], ...]
    policy_identity: str
    sealed_at: str
    as_of: str
    schema: str = LIVE_COHORT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        disposition = _enum(self.disposition, CohortDisposition, name="disposition")
        object.__setattr__(self, "disposition", disposition.value)
        object.__setattr__(self, "admitted", _bool(self.admitted, "admitted"))
        object.__setattr__(self, "live", _bool(self.live, "live"))
        object.__setattr__(self, "qualification", _bool(self.qualification, "qualification"))
        object.__setattr__(self, "count", _int(self.count, "count"))
        object.__setattr__(self, "minimum", _int(self.minimum, "minimum", minimum=LIVE_MINIMUM))
        ids = tuple(_text(item, "task_id", maximum=64) for item in self.distinct_task_ids)
        if len(ids) != len(set(ids)):
            raise LiveCohortAdmissionError("distinct_task_ids values must be unique")
        object.__setattr__(self, "distinct_task_ids", ids)
        object.__setattr__(
            self,
            "enrollment_deadline",
            seal_enrollment_deadline(
                sealed_at=self.sealed_at,
                enrollment_deadline=self.enrollment_deadline,
            )
            if self.sealed_at
            else _text(self.enrollment_deadline, "enrollment_deadline"),
        )
        object.__setattr__(
            self,
            "enrollment_deadline_immutable",
            _bool(self.enrollment_deadline_immutable, "enrollment_deadline_immutable"),
        )
        if self.enrollment_deadline_immutable is not True:
            raise LiveCohortAdmissionError("cohort receipt cannot record a mutable deadline")
        object.__setattr__(self, "shadow_admitted", _bool(self.shadow_admitted, "shadow_admitted"))
        object.__setattr__(self, "canary_admitted", _bool(self.canary_admitted, "canary_admitted"))
        object.__setattr__(
            self,
            "canary_mutation_permitted",
            _bool(self.canary_mutation_permitted, "canary_mutation_permitted"),
        )
        status = _text(self.population_status, "population_status", maximum=32)
        if status not in {"unavailable", "sealed", "insufficient", "enrolling"}:
            raise LiveCohortAdmissionError("population_status is not a closed value")
        object.__setattr__(self, "population_status", status)
        object.__setattr__(
            self, "reasons", tuple(_text(item, "reason", maximum=128) for item in self.reasons)
        )
        receipts: list[Mapping[str, Any]] = []
        if not isinstance(self.task_receipts, Sequence) or isinstance(
            self.task_receipts, (str, bytes, bytearray)
        ):
            raise LiveCohortAdmissionError("task_receipts must be a sequence")
        for item in self.task_receipts:
            if isinstance(item, TaskAdmissionReceipt):
                receipts.append(MappingProxyType(item.to_dict()))
            elif isinstance(item, Mapping):
                receipts.append(MappingProxyType(dict(TaskAdmissionReceipt.from_dict(item).to_dict())))
            else:
                raise LiveCohortAdmissionError("task receipt must be an object")
        object.__setattr__(self, "task_receipts", tuple(receipts))
        object.__setattr__(
            self, "policy_identity", _text(self.policy_identity, "policy_identity")
        )
        if CID_RE.fullmatch(self.policy_identity) is None:
            raise LiveCohortAdmissionError("policy_identity must be a CIDv1")
        object.__setattr__(self, "sealed_at", _text(self.sealed_at, "sealed_at"))
        parse_utc(self.sealed_at, name="sealed_at")
        object.__setattr__(self, "as_of", _text(self.as_of, "as_of"))
        parse_utc(self.as_of, name="as_of")
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != LIVE_COHORT_RECEIPT_SCHEMA:
            raise LiveCohortAdmissionError("unsupported live cohort admission receipt schema")
        qualified = self.disposition == CohortDisposition.EVIDENCE_QUALIFIED.value
        if self.qualification != qualified:
            raise LiveCohortAdmissionError("qualification is derived from evidence_qualified")
        if self.admitted != qualified:
            raise LiveCohortAdmissionError("admitted is derived from evidence_qualified")
        if qualified and self.count < self.minimum:
            raise LiveCohortAdmissionError("evidence-qualified cohort requires 10 distinct live tasks")
        if qualified and not self.live:
            raise LiveCohortAdmissionError("evidence-qualified cohort must prove live provenance")
        if qualified and not self.shadow_admitted:
            raise LiveCohortAdmissionError("evidence-qualified cohort requires shadow admission")
        if self.canary_mutation_permitted and not qualified:
            raise LiveCohortAdmissionError("canary mutation requires an evidence-qualified shadow cohort")
        if self.canary_mutation_permitted and not self.canary_admitted:
            raise LiveCohortAdmissionError("canary mutation requires admitted canary evidence")
        if self.live and not qualified:
            raise LiveCohortAdmissionError("live cannot be claimed without an evidence-qualified cohort")
        if self.disposition not in CLOSED_DISPOSITIONS:
            raise LiveCohortAdmissionError("untyped cohort disposition")

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "as_of": self.as_of,
            "canary_admitted": self.canary_admitted,
            "canary_mutation_permitted": self.canary_mutation_permitted,
            "count": self.count,
            "disposition": self.disposition,
            "distinct_task_ids": list(self.distinct_task_ids),
            "enrollment_deadline": self.enrollment_deadline,
            "enrollment_deadline_immutable": self.enrollment_deadline_immutable,
            "live": self.live,
            "minimum": self.minimum,
            "policy_identity": self.policy_identity,
            "population_status": self.population_status,
            "qualification": self.qualification,
            "reasons": list(self.reasons),
            "schema": self.schema,
            "sealed_at": self.sealed_at,
            "shadow_admitted": self.shadow_admitted,
            "task_receipts": [dict(item) for item in self.task_receipts],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CohortAdmissionReceipt":
        if not isinstance(payload, Mapping):
            raise LiveCohortAdmissionError("cohort admission receipt must be an object")
        receipt = cls(
            disposition=payload.get("disposition", ""),
            admitted=payload.get("admitted", False),
            live=payload.get("live", False),
            qualification=payload.get("qualification", False),
            count=payload.get("count", 0),
            minimum=payload.get("minimum", LIVE_MINIMUM),
            distinct_task_ids=tuple(payload.get("distinct_task_ids") or ()),
            enrollment_deadline=payload.get("enrollment_deadline", ""),
            enrollment_deadline_immutable=payload.get("enrollment_deadline_immutable", True),
            shadow_admitted=payload.get("shadow_admitted", False),
            canary_admitted=payload.get("canary_admitted", False),
            canary_mutation_permitted=payload.get("canary_mutation_permitted", False),
            population_status=payload.get("population_status", "insufficient"),
            reasons=tuple(payload.get("reasons") or ()),
            task_receipts=tuple(payload.get("task_receipts") or ()),
            policy_identity=payload.get("policy_identity", ""),
            sealed_at=payload.get("sealed_at", ""),
            as_of=payload.get("as_of", ""),
            schema=payload.get("schema", LIVE_COHORT_RECEIPT_SCHEMA),
        )
        claimed = payload.get("receipt_id")
        if claimed not in (None, "", receipt.receipt_id):
            raise LiveCohortAdmissionError("forged cohort admission receipt_id")
        return receipt


def _substitution_reasons(evidence: LiveTaskEvidence) -> list[str]:
    reasons: list[str] = []
    source = evidence.source_kind
    if evidence.simulated or evidence.truth_state == "simulated" or source in {
        "simulation",
        "simulated",
    }:
        reasons.append(LiveCohortReason.SIMULATION_SUBSTITUTION.value)
    if evidence.truth_state in FORBIDDEN_LIVE_TRUTH_STATES and evidence.truth_state != "simulated":
        reasons.append(LiveCohortReason.TRUTH_STATE_NOT_LIVE.value)
    if evidence.provider_usage_truth_state == "simulated":
        reasons.append(LiveCohortReason.SIMULATION_SUBSTITUTION.value)
    fixture_claimed = bool(evidence.fixture_id) or evidence.hermetic or source in {
        "fixture",
        "test_fixture",
        "golden",
        "hermetic_fixture",
        "hermetic_development",
    }
    if fixture_claimed or HERMETIC_FIXTURE_RE.fullmatch(evidence.task_id):
        reasons.append(LiveCohortReason.FIXTURE_SUBSTITUTION.value)
    if evidence.hermetic or source in {"hermetic_fixture", "hermetic_development"}:
        reasons.append(LiveCohortReason.HERMETIC_SUBSTITUTION.value)
    historical_claimed = (
        evidence.historical_replay
        or source in {"historical_replay", "historical_exact_tree_replay"}
        or evidence.task_id in HISTORICAL_TASK_IDS
    )
    if historical_claimed:
        reasons.append(LiveCohortReason.HISTORICAL_SUBSTITUTION.value)
    return list(dict.fromkeys(reasons))


def _live_provenance_reasons(evidence: LiveTaskEvidence) -> list[str]:
    reasons: list[str] = []
    if evidence.live is not True:
        reasons.append(LiveCohortReason.NOT_LIVE.value)
    if evidence.population_kind != POPULATION_KIND:
        reasons.append(LiveCohortReason.POPULATION_NOT_LIVE.value)
    if evidence.newly_encountered is not True:
        reasons.append(LiveCohortReason.NOT_NEWLY_ENCOUNTERED.value)
    if evidence.source_kind not in ALLOWED_SOURCE_KINDS:
        if evidence.source_kind in FORBIDDEN_SOURCE_KINDS:
            pass
        else:
            reasons.append(LiveCohortReason.NOT_NEWLY_ENCOUNTERED.value)
    if evidence.truth_state not in LIVE_TRUTH_STATES:
        reasons.append(LiveCohortReason.TRUTH_STATE_NOT_LIVE.value)
    if evidence.provider_usage_truth_state not in PROVIDER_USAGE_TRUTH_STATES:
        reasons.append(LiveCohortReason.PROVIDER_EVIDENCE_MISSING.value)
    marker = f"{evidence.provider_id} {evidence.provider_request_id}".casefold()
    if "simulat" in marker or "fixture" in marker:
        reasons.append(LiveCohortReason.PROVIDER_EVIDENCE_MISSING.value)
    if not evidence.validator_command or not evidence.validator_result_digest:
        reasons.append(LiveCohortReason.VERIFIER_EVIDENCE_MISSING.value)
    return list(dict.fromkeys(reasons))


def _mode_reasons(
    evidence: LiveTaskEvidence,
    *,
    shadow_by_task: Mapping[str, TaskAdmissionReceipt],
) -> list[str]:
    reasons: list[str] = []
    if evidence.execution_mode == ExecutionMode.SHADOW.value:
        if evidence.mutating:
            reasons.append(LiveCohortReason.SHADOW_MUTATING.value)
        return reasons
    prior = shadow_by_task.get(evidence.task_id)
    if prior is None or prior.disposition != TaskDisposition.ADMITTED_SHADOW.value:
        reasons.append(LiveCohortReason.CANARY_WITHOUT_SHADOW.value)
        reasons.append(LiveCohortReason.SHADOW_BYPASS.value)
    elif evidence.shadow_receipt_cid and evidence.shadow_receipt_cid != prior.receipt_id:
        reasons.append(LiveCohortReason.CANARY_WITHOUT_SHADOW.value)
    if not evidence.safety_admitted:
        reasons.append(LiveCohortReason.CANARY_WITHOUT_SAFETY.value)
        reasons.append(LiveCohortReason.SAFETY_NOT_ADMITTED.value)
    return list(dict.fromkeys(reasons))


def admit_task(
    payload: LiveTaskEvidence | Mapping[str, Any],
    *,
    policy: LiveCohortAdmissionPolicy | Mapping[str, Any] | None = None,
    shadow_receipt: TaskAdmissionReceipt | Mapping[str, Any] | None = None,
) -> TaskAdmissionReceipt:
    """Admit one live shadow or canary task, or emit a typed non-admission."""

    evidence = (
        payload if isinstance(payload, LiveTaskEvidence) else LiveTaskEvidence.from_dict(payload)
    )
    admitted_policy = (
        policy
        if isinstance(policy, LiveCohortAdmissionPolicy)
        else LiveCohortAdmissionPolicy.from_dict(policy or sealed_policy().to_dict())
    )
    shadow_map: dict[str, TaskAdmissionReceipt] = {}
    if shadow_receipt is not None:
        prior = (
            shadow_receipt
            if isinstance(shadow_receipt, TaskAdmissionReceipt)
            else TaskAdmissionReceipt.from_dict(shadow_receipt)
        )
        shadow_map[prior.task_id] = prior
    reasons = [
        *_substitution_reasons(evidence),
        *_live_provenance_reasons(evidence),
        *_mode_reasons(evidence, shadow_by_task=shadow_map),
    ]
    reasons = list(dict.fromkeys(reasons))
    live_ok = not reasons and evidence.live is True
    if evidence.execution_mode == ExecutionMode.SHADOW.value and live_ok:
        disposition = TaskDisposition.ADMITTED_SHADOW
    elif evidence.execution_mode == ExecutionMode.CANARY.value and live_ok:
        disposition = TaskDisposition.ADMITTED_CANARY
    else:
        disposition = TaskDisposition.NOT_ADMITTED
    admitted = disposition in {TaskDisposition.ADMITTED_SHADOW, TaskDisposition.ADMITTED_CANARY}
    mutation_permitted = (
        admitted
        and disposition is TaskDisposition.ADMITTED_CANARY
        and evidence.mutating
        and evidence.safety_admitted
    )
    if not admitted and not reasons:
        reasons.append(LiveCohortReason.NOT_LIVE.value)
    bound_shadow = ""
    if disposition is TaskDisposition.ADMITTED_CANARY:
        prior = shadow_map.get(evidence.task_id)
        bound_shadow = evidence.shadow_receipt_cid or (prior.receipt_id if prior else "")
    return TaskAdmissionReceipt(
        task_id=evidence.task_id,
        task_cid=evidence.task_cid,
        disposition=disposition.value,
        admitted=admitted,
        live=admitted,
        execution_mode=evidence.execution_mode,
        mutating=evidence.mutating,
        mutation_permitted=mutation_permitted,
        reasons=tuple(reasons),
        shadow_receipt_cid=bound_shadow,
        policy_identity=admitted_policy.policy_identity,
    )


def _policy_from(
    policy: LiveCohortAdmissionPolicy | Mapping[str, Any] | None,
) -> LiveCohortAdmissionPolicy:
    if policy is None:
        return sealed_policy()
    if isinstance(policy, LiveCohortAdmissionPolicy):
        return policy
    return LiveCohortAdmissionPolicy.from_dict(policy)


_SUBSTITUTION_REASONS: Final[frozenset[str]] = frozenset(
    {
        LiveCohortReason.FIXTURE_SUBSTITUTION.value,
        LiveCohortReason.SIMULATION_SUBSTITUTION.value,
        LiveCohortReason.HERMETIC_SUBSTITUTION.value,
        LiveCohortReason.HISTORICAL_SUBSTITUTION.value,
    }
)


def _parse_as_of(
    now: datetime | str | None,
    *,
    policy: LiveCohortAdmissionPolicy,
) -> datetime:
    if now is None:
        return utc_now()
    if isinstance(now, datetime):
        return now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return parse_utc(now, name="now")


def _downgrade_duplicate(
    receipt: TaskAdmissionReceipt,
    *,
    policy_identity: str,
) -> TaskAdmissionReceipt:
    return TaskAdmissionReceipt(
        task_id=receipt.task_id,
        task_cid=receipt.task_cid,
        disposition=TaskDisposition.NOT_ADMITTED.value,
        admitted=False,
        live=False,
        execution_mode=receipt.execution_mode,
        mutating=receipt.mutating,
        mutation_permitted=False,
        reasons=tuple(dict.fromkeys((*receipt.reasons, LiveCohortReason.DUPLICATE_TASK.value))),
        shadow_receipt_cid=receipt.shadow_receipt_cid,
        policy_identity=policy_identity,
    )


def admit_cohort(
    tasks: Sequence[LiveTaskEvidence | Mapping[str, Any]] | None = None,
    *,
    policy: LiveCohortAdmissionPolicy | Mapping[str, Any] | None = None,
    shadow_receipts: Sequence[TaskAdmissionReceipt | Mapping[str, Any]] | None = None,
    now: datetime | str | None = None,
    request_canary: bool = False,
) -> CohortAdmissionReceipt:
    """Admit a live cohort or emit a typed insufficiency / non-admission."""

    admitted_policy = _policy_from(policy)
    as_of = _parse_as_of(now, policy=admitted_policy)
    as_of_text = format_utc(as_of)
    parsed: list[LiveTaskEvidence] = []
    seen_keys: set[tuple[str, str]] = set()
    duplicate_keys: set[tuple[str, str]] = set()
    for item in tasks or ():
        evidence = item if isinstance(item, LiveTaskEvidence) else LiveTaskEvidence.from_dict(item)
        key = (evidence.task_id, evidence.execution_mode)
        if key in seen_keys:
            duplicate_keys.add(key)
        else:
            seen_keys.add(key)
        parsed.append(evidence)

    shadow_by_task: dict[str, TaskAdmissionReceipt] = {}
    for item in shadow_receipts or ():
        receipt = item if isinstance(item, TaskAdmissionReceipt) else TaskAdmissionReceipt.from_dict(item)
        if receipt.task_id in shadow_by_task:
            raise LiveCohortAdmissionError("duplicated shadow receipt task_id")
        shadow_by_task[receipt.task_id] = receipt

    for evidence in parsed:
        if evidence.execution_mode != ExecutionMode.SHADOW.value:
            continue
        key = (evidence.task_id, evidence.execution_mode)
        if key in duplicate_keys and evidence.task_id in shadow_by_task:
            continue
        receipt = admit_task(evidence, policy=admitted_policy)
        if receipt.admitted and evidence.task_id not in shadow_by_task:
            shadow_by_task[evidence.task_id] = receipt

    task_receipts: list[TaskAdmissionReceipt] = []
    first_seen: set[tuple[str, str]] = set()
    substitution = False
    duplicate = bool(duplicate_keys)
    for evidence in parsed:
        prior = shadow_by_task.get(evidence.task_id)
        receipt = admit_task(evidence, policy=admitted_policy, shadow_receipt=prior)
        key = (evidence.task_id, evidence.execution_mode)
        if key in first_seen:
            receipt = _downgrade_duplicate(receipt, policy_identity=admitted_policy.policy_identity)
        else:
            first_seen.add(key)
        if _SUBSTITUTION_REASONS.intersection(receipt.reasons):
            substitution = True
        task_receipts.append(receipt)

    qualified_ids = tuple(
        dict.fromkeys(
            receipt.task_id
            for receipt in shadow_by_task.values()
            if receipt.disposition == TaskDisposition.ADMITTED_SHADOW.value
        )
    )
    canary_ids = tuple(
        dict.fromkeys(
            receipt.task_id
            for receipt in task_receipts
            if receipt.disposition == TaskDisposition.ADMITTED_CANARY.value
        )
    )
    bypass = any(
        LiveCohortReason.SHADOW_BYPASS.value in receipt.reasons
        or LiveCohortReason.CANARY_WITHOUT_SHADOW.value in receipt.reasons
        for receipt in task_receipts
    )
    elapsed = as_of > admitted_policy.enrollment_deadline_utc
    reasons: list[str] = []
    if substitution:
        disposition = CohortDisposition.NOT_ADMITTED
        for receipt in task_receipts:
            reasons.extend(code for code in receipt.reasons if code in _SUBSTITUTION_REASONS)
    elif request_canary and (bypass or len(qualified_ids) < admitted_policy.minimum):
        disposition = CohortDisposition.NOT_ADMITTED
        reasons.append(LiveCohortReason.SHADOW_BYPASS.value)
    elif len(qualified_ids) >= admitted_policy.minimum:
        disposition = CohortDisposition.EVIDENCE_QUALIFIED
    elif elapsed:
        disposition = CohortDisposition.DEADLINE_ELAPSED
        reasons.append(LiveCohortReason.DEADLINE_ELAPSED.value)
        reasons.append(LiveCohortReason.INSUFFICIENT_DISTINCT_LIVE_TASKS.value)
    else:
        disposition = CohortDisposition.INSUFFICIENT_EVIDENCE
        reasons.append(LiveCohortReason.INSUFFICIENT_DISTINCT_LIVE_TASKS.value)
        reasons.append(LiveCohortReason.NOT_YET_MEASURED.value)
    if duplicate and len(qualified_ids) < admitted_policy.minimum:
        reasons.append(LiveCohortReason.DUPLICATE_TASK.value)
    reasons = list(dict.fromkeys(reasons))
    qualified = disposition is CohortDisposition.EVIDENCE_QUALIFIED
    if qualified:
        population_status = "sealed"
    elif elapsed:
        population_status = "insufficient"
    else:
        population_status = "enrolling"
    canary_admitted = qualified and bool(canary_ids)
    canary_mutation_permitted = canary_admitted and all(
        receipt.mutation_permitted
        for receipt in task_receipts
        if receipt.disposition == TaskDisposition.ADMITTED_CANARY.value
    )
    if request_canary and qualified and not canary_admitted:
        disposition = CohortDisposition.NOT_ADMITTED
        qualified = False
        reasons.append(LiveCohortReason.CANARY_WITHOUT_SAFETY.value)
        population_status = "insufficient"
        canary_mutation_permitted = False
    return CohortAdmissionReceipt(
        disposition=disposition.value,
        admitted=qualified,
        live=qualified,
        qualification=qualified,
        count=len(qualified_ids),
        minimum=admitted_policy.minimum,
        distinct_task_ids=qualified_ids,
        enrollment_deadline=admitted_policy.enrollment_deadline,
        enrollment_deadline_immutable=True,
        shadow_admitted=qualified,
        canary_admitted=canary_admitted if qualified else False,
        canary_mutation_permitted=canary_mutation_permitted if qualified else False,
        population_status=population_status,
        reasons=tuple(reasons),
        task_receipts=tuple(receipt.to_dict() for receipt in task_receipts),
        policy_identity=admitted_policy.policy_identity,
        sealed_at=admitted_policy.sealed_at,
        as_of=as_of_text,
    )


def build_manifest(
    receipt: CohortAdmissionReceipt | None = None,
    *,
    policy: LiveCohortAdmissionPolicy | None = None,
) -> dict[str, Any]:
    """Seal the live-cohort admission contract and current honest disposition."""

    admitted_policy = policy or sealed_policy()
    snapshot = receipt or admit_cohort((), policy=admitted_policy, now=admitted_policy.sealed_at)
    payload = {
        "authority": False,
        "canary_admitted": snapshot.canary_admitted,
        "canary_mutation_permitted": snapshot.canary_mutation_permitted,
        "canary_requires_safety_admission": admitted_policy.canary_requires_safety_admission,
        "count": snapshot.count,
        "disposition": snapshot.disposition,
        "disposition_reason_code": (
            snapshot.reasons[0] if snapshot.reasons else LiveCohortReason.NOT_YET_MEASURED.value
        ),
        "enrollment_deadline": admitted_policy.enrollment_deadline,
        "enrollment_deadline_immutable": True,
        "enrollment_deadline_maximum_days": LIVE_ENROLLMENT_DEADLINE_MAXIMUM_DAYS,
        "fixtures_satisfy_live": False,
        "hermetic_sufficient_for_production_promotion": False,
        "historical_sufficient_for_production_promotion": False,
        "live": snapshot.live,
        "minimum": admitted_policy.minimum,
        "objective_id": OBJECTIVE_ID,
        "objective_revision": PLAN_ROOT_CID,
        "policy_identity": POLICY_IDENTITY,
        "population_kind": POPULATION_KIND,
        "population_status": snapshot.population_status,
        "program_id": PROGRAM_ID,
        "provenance_fields": list(REQUIRED_PROVENANCE_FIELDS),
        "qualification": snapshot.qualification,
        "reasons": list(snapshot.reasons),
        "required_gates": list(LIVE_COHORT_GATES),
        "schema": LIVE_COHORT_MANIFEST_SCHEMA,
        "schema_version": 1,
        "sealed_at": admitted_policy.sealed_at,
        "shadow_admitted": snapshot.shadow_admitted,
        "shadow_before_canary": True,
        "simulation_satisfies_live": False,
        "status": "sealed",
        "task_id": TASK_ID,
        "tasks": [
            {
                "disposition": item["disposition"],
                "task_cid": item["task_cid"],
                "task_id": item["task_id"],
            }
            for item in snapshot.task_receipts
        ],
    }
    payload["identity"] = content_identity(_body(payload, "identity"))
    return payload


def write_sealed_artifacts(path: Path | None = None) -> Path:
    target = path or MANIFEST_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_manifest()
    target.write_text(pretty_json(payload), encoding="utf-8")
    return target


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    target = path or MANIFEST_PATH
    raw = target.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise LiveCohortAdmissionError("live cohort manifest must be an object")
    expected = pretty_json(payload)
    if raw != expected:
        raise LiveCohortAdmissionError("live cohort manifest is not canonical JSON")
    return payload


def verify_sealed_artifacts(path: Path | None = None) -> dict[str, Any]:
    target = path or MANIFEST_PATH
    loaded = load_manifest(target)
    expected = build_manifest()
    if loaded != expected:
        raise LiveCohortAdmissionError("live cohort manifest does not match the sealed admission contract")
    if CID_RE.fullmatch(str(loaded.get("identity", ""))) is None:
        raise LiveCohortAdmissionError("live cohort manifest identity must be a CIDv1")
    deadline = parse_utc(loaded["enrollment_deadline"], name="enrollment_deadline")
    sealed_at = parse_utc(loaded["sealed_at"], name="sealed_at")
    if loaded["enrollment_deadline_immutable"] is not True:
        raise LiveCohortAdmissionError("sealed enrollment deadline is mutable")
    if deadline > sealed_at + timedelta(days=LIVE_ENROLLMENT_DEADLINE_MAXIMUM_DAYS):
        raise LiveCohortAdmissionError("sealed enrollment deadline exceeds the 30 calendar day bound")
    if loaded["minimum"] != LIVE_MINIMUM:
        raise LiveCohortAdmissionError("sealed live cohort minimum was lowered")
    if loaded["qualification"] is True and loaded["count"] < LIVE_MINIMUM:
        raise LiveCohortAdmissionError("sealed manifest claims qualification without 10 live tasks")
    if loaded["live"] is True and loaded["qualification"] is not True:
        raise LiveCohortAdmissionError("sealed manifest claims live evidence without qualification")
    if loaded["disposition"] not in CLOSED_DISPOSITIONS:
        raise LiveCohortAdmissionError("untyped sealed cohort disposition")
    return {
        "count": loaded["count"],
        "disposition": loaded["disposition"],
        "enrollment_deadline": loaded["enrollment_deadline"],
        "identity": loaded["identity"],
        "qualification": loaded["qualification"],
    }


class LiveCohortAdmissionService:
    """Fail-closed admission boundary for the live shadow/canary cohort."""

    def __init__(self, policy: LiveCohortAdmissionPolicy | None = None) -> None:
        self._policy = policy or sealed_policy()

    @property
    def policy(self) -> LiveCohortAdmissionPolicy:
        return self._policy

    def admit_task(
        self,
        payload: LiveTaskEvidence | Mapping[str, Any],
        *,
        shadow_receipt: TaskAdmissionReceipt | Mapping[str, Any] | None = None,
    ) -> TaskAdmissionReceipt:
        return admit_task(payload, policy=self._policy, shadow_receipt=shadow_receipt)

    def admit_cohort(
        self,
        tasks: Sequence[LiveTaskEvidence | Mapping[str, Any]] | None = None,
        *,
        shadow_receipts: Sequence[TaskAdmissionReceipt | Mapping[str, Any]] | None = None,
        now: datetime | str | None = None,
        request_canary: bool = False,
    ) -> CohortAdmissionReceipt:
        return admit_cohort(
            tasks,
            policy=self._policy,
            shadow_receipts=shadow_receipts,
            now=now,
            request_canary=request_canary,
        )

    def seal_manifest(self) -> dict[str, Any]:
        return build_manifest(policy=self._policy)


__all__ = (
    "CLOSED_DISPOSITIONS",
    "CohortAdmissionReceipt",
    "CohortDisposition",
    "ENROLLMENT_DEADLINE",
    "ExecutionMode",
    "LIVE_COHORT_GATES",
    "LIVE_ENROLLMENT_DEADLINE_MAXIMUM_DAYS",
    "LIVE_MINIMUM",
    "LiveCohortAdmissionError",
    "LiveCohortAdmissionPolicy",
    "LiveCohortAdmissionService",
    "LiveCohortReason",
    "LiveTaskEvidence",
    "REQUIRED_PROVENANCE_FIELDS",
    "SEALED_AT",
    "TaskAdmissionReceipt",
    "TaskDisposition",
    "admit_cohort",
    "admit_task",
    "build_manifest",
    "live_task_evidence_payload",
    "seal_enrollment_deadline",
    "sealed_policy",
    "verify_sealed_artifacts",
    "write_sealed_artifacts",
)


if __name__ == "__main__":
    written = write_sealed_artifacts()
    verify_sealed_artifacts(written)
