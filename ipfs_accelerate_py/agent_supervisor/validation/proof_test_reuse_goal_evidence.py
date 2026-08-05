"""Independent goal coverage, analyzer, and adversarial population receipts.

PTR-111 produces typed goal-assurance evidence for the proof-backed test reuse
program.  Requirement IDs are discovered from the objective heap (not a
hard-coded per-test registry).  Every receipt binds its producer channel,
canonical channel proof revision, current repository identities, observed /
fresh-until window, and retained validation bytes.

This module is deliberately a boundary adapter:

* heap labels and historical task status are discovery inputs only;
* unavailable Groth16 / ProveKit / cache / IPFS capabilities are typed and
  non-blocking, but they leave real-ZK and production-warm criteria
  unverified unless a reviewed, locally verifiable real certificate is
  present;
* synthetic ``_AlwaysVerify`` benchmark harness data is never deployment
  authority; and
* two exhaustion-quorum members must be independent, healthy, exhaustive,
  conclusive, fresh, and uncontradicted.
"""

from __future__ import annotations

import base64
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final

from ..objectives.objective_graph import ObjectiveGoal, parse_goal_heap
from ..proof.formal_verification_contracts import CanonicalContract, content_identity

# ---------------------------------------------------------------------------
# Interface / schema discriminators
# ---------------------------------------------------------------------------

PROOF_TEST_REUSE_GOAL_EVIDENCE_VERSION: Final = 1
PROOF_TEST_REUSE_GOAL_EVIDENCE_INTERFACE: Final = "ProofTestReuseGoalEvidence@1"
ACCEPTANCE_COVERAGE_INTERFACE: Final = "AcceptanceCoverage@1"
ACCEPTANCE_COVERAGE_RECEIPT_INTERFACE: Final = "AcceptanceCoverageReceipt@1"
PROOF_REUSE_ANALYZER_RECEIPT_INTERFACE: Final = "ProofReuseAnalyzerReceipt@1"
PROOF_REUSE_POPULATION_RECEIPT_INTERFACE: Final = "ProofReusePopulationReceipt@1"
GOAL_ASSURANCE_RESULT_INTERFACE: Final = "GoalAssuranceResult@1"
GOAL_EVIDENCE_GAP_INTERFACE: Final = "GoalEvidenceGap@1"
TEST_CERTIFICATE_ASSURANCE_RECEIPT_INTERFACE: Final = (
    "TestCertificateAssuranceReceipt@1"
)
ANALYZER_HEALTH_INTERFACE: Final = "AnalyzerHealth"
EXHAUSTION_QUORUM_INTERFACE: Final = "ExhaustionQuorum"
PROOF_REUSE_BENCHMARK_RECEIPT_INTERFACE: Final = "ProofReuseBenchmarkReceipt"
PROOF_REUSE_ROLLBACK_DECISION_INTERFACE: Final = "ProofReuseRollbackDecision"

PROOF_TEST_REUSE_GOAL_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-test-reuse-goal-evidence@1"
)
ACCEPTANCE_COVERAGE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/acceptance-coverage-receipt@1"
)
PROOF_REUSE_ANALYZER_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-reuse-analyzer-receipt@1"
)
PROOF_REUSE_POPULATION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-reuse-population-receipt@1"
)
GOAL_ASSURANCE_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-assurance-result@1"
)
GOAL_EVIDENCE_GAP_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-evidence-gap@1"
)

DEFAULT_PRODUCER_CHANNEL: Final = "goal-assurance"
DEFAULT_CHANNEL_PROOF_REVISION: Final = "channel:ptr-111@1"
DEFAULT_EVIDENCE_FRESHNESS_SECONDS: Final = 300.0
PRODUCING_TASK_ID: Final = "PTR-111"

REQUIRED_ANALYZER_CHANNELS: Final = frozenset(
    {"static-dependency", "runtime-dependency", "reuse-eligibility"}
)
REQUIRED_ADVERSARIAL_POPULATIONS: Final = frozenset(
    {"mutation", "storage-security-concurrency", "cross-repository"}
)
REQUIRED_QUORUM_MEMBERS: Final = 2

# Criteria that need a real, locally verifiable certificate when optional
# proof backends are absent.  Unavailable capability is never a hard failure;
# it simply withholds authority for these IDs.
REAL_ZK_REQUIREMENT_IDS: Final = frozenset(
    {
        "ptr/test-pass-statement@1",
        "ptr/real-zk-certificate-conformance@1",
        "ptr/deferred-certificate-issuance@1",
        "ptr/datasets-certificate-adapter@1",
        "ptr/datasets-test-certificate-provider@1",
    }
)
PRODUCTION_WARM_REQUIREMENT_IDS: Final = frozenset(
    {
        "ptr/warm-reuse-benchmark@1",
        "ptr/test-proof-cache-admission@1",
        "ptr/immutable-certificate-index@1",
    }
)
OPTIONAL_CAPABILITY_NAMES: Final = frozenset(
    {"groth16", "provekit", "cache", "ipfs"}
)

_AUTHORITATIVE: Final = "authoritative"
_NO_AUTHORITY: Final = "none"
_REQUIREMENT_ID_RE: Final = re.compile(r"ptr/[a-z0-9@._/-]+", re.IGNORECASE)
_ALWAYS_VERIFY_MARKERS: Final = (
    "_alwaysverify",
    "alwaysverify",
    "always_verify",
    "synthetic_benchmark",
    "benchmark_harness_only",
)

_MAX_DETAIL = 512
_MAX_RETAINED_BYTES = 1_048_576


class GoalEvidenceGapKind(str, Enum):
    """Closed reasons for withholding goal-assurance authority."""

    HEAP_MALFORMED = "heap_malformed"
    HEAP_REQUIREMENT_MISSING = "heap_requirement_missing"
    REQUIREMENT_REGISTRY_FORBIDDEN = "requirement_registry_forbidden"
    COVERAGE_MISSING = "coverage_missing"
    COVERAGE_FAILED = "coverage_failed"
    COVERAGE_STALE = "coverage_stale"
    COVERAGE_BINDING_MISMATCH = "coverage_binding_mismatch"
    RETAINED_BYTES_MISSING = "retained_bytes_missing"
    RETAINED_BYTES_MISMATCH = "retained_bytes_mismatch"
    PRODUCER_CHANNEL_MISSING = "producer_channel_missing"
    CHANNEL_PROOF_REVISION_MISSING = "channel_proof_revision_missing"
    ANALYZER_MISSING = "analyzer_missing"
    ANALYZER_UNHEALTHY = "analyzer_unhealthy"
    ANALYZER_INCOMPLETE = "analyzer_incomplete"
    POPULATION_MISSING = "population_missing"
    POPULATION_FAILED = "population_failed"
    FALSE_SKIP_DETECTED = "false_skip_detected"
    QUORUM_INSUFFICIENT = "quorum_insufficient"
    QUORUM_NOT_INDEPENDENT = "quorum_not_independent"
    QUORUM_UNHEALTHY = "quorum_unhealthy"
    QUORUM_NOT_EXHAUSTIVE = "quorum_not_exhaustive"
    QUORUM_INCONCLUSIVE = "quorum_inconclusive"
    QUORUM_STALE = "quorum_stale"
    QUORUM_CONTRADICTED = "quorum_contradicted"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    REAL_ZK_UNVERIFIED = "real_zk_unverified"
    PRODUCTION_WARM_UNVERIFIED = "production_warm_unverified"
    SYNTHETIC_BENCHMARK_AUTHORITY = "synthetic_benchmark_authority"
    BENCHMARK_NON_AUTHORITATIVE = "benchmark_non_authoritative"
    ROLLOUT_MISSING = "rollout_missing"
    UNEXPECTED_INPUT = "unexpected_input"
    MALFORMED = "malformed"


class CoverageStatus(str, Enum):
    """Per-requirement coverage conclusion."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    STALE = "stale"
    MISSING = "missing"
    BLOCKED_BY_CAPABILITY = "blocked_by_capability"
    REJECTED = "rejected"


class ProofTestReuseGoalEvidenceError(ValueError):
    """Raised only for invalid construction or contract authentication."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _record(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        candidate = to_dict()
        if isinstance(candidate, Mapping):
            return candidate
    to_record = getattr(value, "to_record", None)
    if callable(to_record):
        candidate = to_record()
        if isinstance(candidate, Mapping):
            return candidate
    fields = getattr(value, "__dataclass_fields__", None)
    if isinstance(fields, Mapping):
        return {name: getattr(value, name) for name in fields}
    return {}


def _value(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _boolean(record: Mapping[str, Any], *names: str) -> bool | None:
    value = _value(record, *names)
    return value if isinstance(value, bool) else None


def _integer(record: Mapping[str, Any], *names: str) -> int | None:
    value = _value(record, *names)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _clock_milliseconds(clock: Callable[[], float]) -> int:
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProofTestReuseGoalEvidenceError(
            "clock must return seconds since the Unix epoch"
        )
    return int(float(value) * 1_000)


def _split_requirement_field(raw: str) -> tuple[str, ...]:
    """Split a heap Evidence / Acceptance-criteria field into requirement IDs."""

    text = _text(raw)
    if not text:
        return ()
    # Prefer exact machine IDs (ptr/...) over free prose.
    found = tuple(dict.fromkeys(_REQUIREMENT_ID_RE.findall(text)))
    if found:
        return found
    parts: list[str] = []
    for chunk in re.split(r"[;,]", text):
        item = chunk.strip()
        if item:
            parts.append(item)
    return tuple(dict.fromkeys(parts))


def _normalize_capability_name(value: Any) -> str:
    return _text(value).lower().replace("-", "_").replace(" ", "_")


def _capability_available(facts: Mapping[str, Any], name: str) -> bool | None:
    """Return True/False when known, None when the capability was not reported."""

    key = _normalize_capability_name(name)
    if key not in facts and name not in facts:
        # Nested capabilities list/report form.
        items = facts.get("capabilities")
        if isinstance(items, Mapping):
            return _capability_available(items, name)
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
            for item in items:
                record = _record(item)
                item_name = _normalize_capability_name(
                    _value(record, "name", "capability", "capability_name", "id")
                )
                if item_name == key:
                    status = _text(
                        _value(record, "status", "state", "availability")
                    ).lower()
                    available = _boolean(record, "available")
                    if available is not None:
                        return available
                    if status in {"available", "ok", "ready"}:
                        return True
                    if status in {
                        "missing",
                        "unavailable",
                        "disabled",
                        "incompatible",
                        "unknown",
                    }:
                        return False
            return None
        return None
    raw = facts.get(key, facts.get(name))
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, Mapping):
        available = _boolean(raw, "available")
        if available is not None:
            return available
        status = _text(_value(raw, "status", "state")).lower()
        if status in {"available", "ok", "ready"}:
            return True
        if status in {"missing", "unavailable", "disabled", "incompatible", "unknown"}:
            return False
        return None
    status = _text(raw).lower()
    if status in {"available", "ok", "ready", "true", "1", "yes"}:
        return True
    if status in {
        "missing",
        "unavailable",
        "disabled",
        "incompatible",
        "unknown",
        "false",
        "0",
        "no",
    }:
        return False
    return None


def _encode_retained_bytes(payload: Mapping[str, Any] | bytes | str) -> tuple[str, str]:
    """Return (base64 retained bytes, content identity of those exact bytes)."""

    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        # CanonicalContract path: identity over the mapping itself, then encode
        # the sorted JSON-compatible dict form already used by content_identity.
        from ..proof.formal_verification_contracts import canonical_json_bytes

        raw = canonical_json_bytes(dict(payload))
    if not raw:
        raise ProofTestReuseGoalEvidenceError("retained validation bytes must be nonempty")
    if len(raw) > _MAX_RETAINED_BYTES:
        raise ProofTestReuseGoalEvidenceError("retained validation bytes exceed budget")
    encoded = base64.b64encode(raw).decode("ascii")
    # Content identity of the exact retained byte string (raw codec style via
    # the shared dag-json helper wrapping a bytes digest envelope).
    digest_payload = {
        "kind": "retained_validation_bytes",
        "sha256_b64": base64.b64encode(__import__("hashlib").sha256(raw).digest()).decode(
            "ascii"
        ),
        "size": len(raw),
    }
    return encoded, content_identity(digest_payload)


def _decode_retained_bytes(encoded: str) -> bytes:
    try:
        return base64.b64decode(encoded.encode("ascii"), validate=True)
    except Exception as exc:  # noqa: BLE001 - fail closed on any decode issue
        raise ProofTestReuseGoalEvidenceError(
            "retained validation bytes are not valid base64"
        ) from exc


def _contract_payload(
    payload: Mapping[str, Any],
    *,
    schema: str,
    interface: str,
    allowed: frozenset[str],
    artifact: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ProofTestReuseGoalEvidenceError(f"{artifact} must be a mapping")
    if set(payload).difference(allowed | {"content_id"}):
        raise ProofTestReuseGoalEvidenceError(
            f"{artifact} contains unsupported fields"
        )
    body = {str(key): value for key, value in payload.items() if key != "content_id"}
    if (
        body.get("schema") != schema
        or body.get("interface") != interface
        or body.get("contract_version") != PROOF_TEST_REUSE_GOAL_EVIDENCE_VERSION
    ):
        raise ProofTestReuseGoalEvidenceError(
            f"{artifact} has an unsupported contract discriminator"
        )
    claimed = _text(payload.get("content_id"))
    if claimed and claimed != content_identity(body):
        raise ProofTestReuseGoalEvidenceError(
            f"{artifact} content identity does not match its payload"
        )
    return body


def _is_synthetic_benchmark(record: Mapping[str, Any]) -> bool:
    """Return True when the benchmark payload is harness-only / AlwaysVerify."""

    blobs = [
        _text(record.get("verifier_id")),
        _text(record.get("verifier")),
        _text(record.get("corpus_id")),
        _text(record.get("authority")),
        _text(record.get("deployment_authority")),
        _text(record.get("benchmark_kind")),
        str(record.get("synthetic") or "").lower(),
        str(record.get("harness_only") or "").lower(),
    ]
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        blobs.extend(_text(value) for value in metadata.values())
        blobs.append(str(metadata.get("always_verify") or "").lower())
    joined = " ".join(blobs).lower().replace("-", "").replace(" ", "")
    if any(marker.replace("_", "") in joined for marker in _ALWAYS_VERIFY_MARKERS):
        return True
    if record.get("synthetic") is True or record.get("harness_only") is True:
        return True
    if record.get("deployment_authority") is False:
        return True
    # Explicit AlwaysVerify class name leakage.
    return "alwaysverify" in joined


def _is_real_certificate_assurance(record: Mapping[str, Any]) -> bool:
    """True only for reviewed, locally verified real-ZK assurance receipts."""

    if not record:
        return False
    interface = _text(_value(record, "interface", "schema"))
    status = _text(_value(record, "status", "assurance_status", "result")).lower()
    authority = _text(record.get("authority")).lower()
    backend = _text(
        _value(record, "backend", "proof_system_id", "backend_mode")
    ).lower()
    locally_verified = _boolean(
        record, "locally_verified", "verified", "locally_verifiable"
    )
    if status not in {"verified", "ok", "passed"} and record.get("verified") is not True:
        # RealZKConformanceReceipt uses interface + verified flag.
        if interface and "realzkconformance" not in interface.lower().replace(
            "-", ""
        ).replace("_", ""):
            if status not in {"verified"}:
                return False
    if authority and authority not in {_AUTHORITATIVE, "real", "cryptographic"}:
        return False
    if backend and backend not in {"groth16", "provekit", "cryptographic", "real"}:
        if "simulat" in backend or "demo" in backend or "mock" in backend:
            return False
    if locally_verified is False:
        return False
    # Require an explicit positive verification signal.
    if (
        status in {"verified", "ok", "passed"}
        or record.get("verified") is True
        or authority == _AUTHORITATIVE
    ):
        # Reject pure unavailable markers.
        if status == "unavailable" or record.get("unavailable") is True:
            return False
        return True
    return False


# ---------------------------------------------------------------------------
# Requirement discovery (objective heap — never a per-test registry)
# ---------------------------------------------------------------------------


def discover_requirement_ids_from_heap(
    heap: str | Path | Sequence[ObjectiveGoal] | Mapping[str, Any],
) -> tuple[str, ...]:
    """Discover machine acceptance requirement IDs from an objective heap.

    The heap is the sole population authority.  Callers must not substitute a
    per-test requirement registry; that path is rejected by
    :class:`GoalAssuranceRunner`.
    """

    goals = load_objective_goals(heap)
    discovered: list[str] = []
    for goal in goals:
        for requirement_id in goal_requirement_ids(goal):
            if requirement_id not in discovered:
                discovered.append(requirement_id)
    return tuple(discovered)


def goal_requirement_ids(goal: ObjectiveGoal | Mapping[str, Any]) -> tuple[str, ...]:
    """Return the ordered machine acceptance IDs declared on one goal."""

    if isinstance(goal, ObjectiveGoal):
        fields = goal.fields
        goal_id = goal.goal_id
    else:
        record = _record(goal)
        fields = _record(record.get("fields")) or {
            key: value
            for key, value in record.items()
            if key
            not in {
                "goal_id",
                "title",
                "fields",
            }
        }
        goal_id = _text(record.get("goal_id"))
    evidence = _split_requirement_field(
        _text(
            fields.get("acceptance_criteria")
            or fields.get("evidence")
            or fields.get("acceptance_criterion")
            or ""
        )
    )
    # Prefer acceptance_criteria when both are present and equal-length machine
    # IDs; otherwise fall back to evidence.  Board policy requires them to match.
    criteria = _split_requirement_field(_text(fields.get("acceptance_criteria") or ""))
    evidence_ids = _split_requirement_field(_text(fields.get("evidence") or ""))
    if criteria and evidence_ids and criteria != evidence_ids:
        # Fail closed at discovery time by returning the empty population; the
        # runner will emit a heap_malformed gap naming the goal.
        return ()
    selected = criteria or evidence_ids or evidence
    if not selected and goal_id:
        return ()
    return selected


def load_objective_goals(
    heap: str | Path | Sequence[ObjectiveGoal] | Mapping[str, Any],
) -> tuple[ObjectiveGoal, ...]:
    """Load objective goals from text, path, sequence, or structured mapping."""

    if isinstance(heap, ObjectiveGoal):
        return (heap,)
    if isinstance(heap, Path):
        return tuple(parse_goal_heap(heap.read_text(encoding="utf-8")))
    if isinstance(heap, str):
        path = Path(heap)
        if "\n" not in heap and path.is_file():
            return tuple(parse_goal_heap(path.read_text(encoding="utf-8")))
        return tuple(parse_goal_heap(heap))
    if isinstance(heap, Mapping):
        goals = heap.get("goals")
        if isinstance(goals, Sequence) and not isinstance(goals, (str, bytes)):
            return load_objective_goals(goals)
        if _text(heap.get("goal_id")):
            return (
                ObjectiveGoal(
                    goal_id=_text(heap.get("goal_id")),
                    title=_text(heap.get("title")),
                    fields={
                        str(key): str(value)
                        for key, value in _record(heap.get("fields") or heap).items()
                        if key not in {"goal_id", "title"}
                    },
                ),
            )
        raise ProofTestReuseGoalEvidenceError("objective heap mapping is malformed")
    if isinstance(heap, Sequence):
        goals: list[ObjectiveGoal] = []
        for item in heap:
            if isinstance(item, ObjectiveGoal):
                goals.append(item)
            else:
                goals.extend(load_objective_goals(item))
        return tuple(goals)
    raise ProofTestReuseGoalEvidenceError("unsupported objective heap input")


def goal_requirements_by_id(
    heap: str | Path | Sequence[ObjectiveGoal] | Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Map each goal ID to its ordered machine requirement population."""

    result: dict[str, tuple[str, ...]] = {}
    for goal in load_objective_goals(heap):
        result[goal.goal_id] = goal_requirement_ids(goal)
    return result


# ---------------------------------------------------------------------------
# Gap + receipt contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoalEvidenceGap(CanonicalContract):
    """Non-authoritative explanation of a missing or rejected premise."""

    SCHEMA: ClassVar[str] = GOAL_EVIDENCE_GAP_SCHEMA

    subject_id: str
    kind: GoalEvidenceGapKind | str
    detail: str
    input_cid: str = ""

    def __post_init__(self) -> None:
        subject_id = _text(self.subject_id) or "*"
        detail = _text(self.detail)
        if not detail:
            raise ProofTestReuseGoalEvidenceError("gap detail is required")
        try:
            kind = (
                self.kind
                if isinstance(self.kind, GoalEvidenceGapKind)
                else GoalEvidenceGapKind(_text(self.kind))
            )
        except ValueError as exc:
            raise ProofTestReuseGoalEvidenceError(
                "unsupported goal evidence gap kind"
            ) from exc
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "detail", detail[:_MAX_DETAIL])
        object.__setattr__(self, "input_cid", _text(self.input_cid))

    @property
    def interface(self) -> str:
        return GOAL_EVIDENCE_GAP_INTERFACE

    @property
    def authority(self) -> str:
        return _NO_AUTHORITY

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def reason_code(self) -> str:
        return self.kind.value

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROOF_TEST_REUSE_GOAL_EVIDENCE_VERSION,
            "interface": self.interface,
            "subject_id": self.subject_id,
            "kind": self.kind.value,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "input_cid": self.input_cid,
            "authority": self.authority,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoalEvidenceGap:
        body = _contract_payload(
            payload,
            schema=cls.SCHEMA,
            interface=GOAL_EVIDENCE_GAP_INTERFACE,
            allowed=frozenset(
                {
                    "schema",
                    "contract_version",
                    "interface",
                    "subject_id",
                    "kind",
                    "reason_code",
                    "detail",
                    "input_cid",
                    "authority",
                }
            ),
            artifact="goal evidence gap",
        )
        result = cls(
            subject_id=body.get("subject_id", ""),
            kind=body.get("kind", ""),
            detail=body.get("detail", ""),
            input_cid=body.get("input_cid", ""),
        )
        if (
            body.get("reason_code") != result.reason_code
            or body.get("authority") != result.authority
        ):
            raise ProofTestReuseGoalEvidenceError(
                "goal evidence gap carries contradictory derived fields"
            )
        return result


@dataclass(frozen=True, slots=True)
class AcceptanceCoverageReceipt(CanonicalContract):
    """Typed coverage receipt for one heap-discovered acceptance requirement."""

    SCHEMA: ClassVar[str] = ACCEPTANCE_COVERAGE_RECEIPT_SCHEMA

    requirement_id: str
    goal_id: str
    status: CoverageStatus | str
    producer_channel: str
    channel_proof_revision: str
    repository_id: str
    repository_state_cid: str
    git_commit_id: str
    git_tree_id: str
    gitlink_state_cid: str
    repository_forest_cid: str
    dirty: bool
    dirty_overlay_cid: str
    policy_cid: str
    capability_cid: str
    verifying_key_cid: str
    circuit_cid: str
    objective_revision: str
    observed_at_ms: int
    fresh_until_ms: int
    retained_validation_bytes_b64: str
    retained_validation_cid: str
    producing_task_id: str = PRODUCING_TASK_ID
    locally_verified: bool = True
    validation_passed: bool = True

    def __post_init__(self) -> None:
        for name in (
            "requirement_id",
            "goal_id",
            "producer_channel",
            "channel_proof_revision",
            "repository_id",
            "repository_state_cid",
            "git_commit_id",
            "git_tree_id",
            "gitlink_state_cid",
            "repository_forest_cid",
            "dirty_overlay_cid",
            "policy_cid",
            "capability_cid",
            "verifying_key_cid",
            "circuit_cid",
            "objective_revision",
            "retained_validation_bytes_b64",
            "retained_validation_cid",
            "producing_task_id",
        ):
            value = _text(getattr(self, name))
            if not value:
                raise ProofTestReuseGoalEvidenceError(f"{name} is required")
            object.__setattr__(self, name, value)
        try:
            status = (
                self.status
                if isinstance(self.status, CoverageStatus)
                else CoverageStatus(_text(self.status))
            )
        except ValueError as exc:
            raise ProofTestReuseGoalEvidenceError(
                "unsupported coverage status"
            ) from exc
        object.__setattr__(self, "status", status)
        if not isinstance(self.dirty, bool) or self.locally_verified is not True:
            raise ProofTestReuseGoalEvidenceError(
                "coverage must be locally verified with a boolean dirty flag"
            )
        if not isinstance(self.validation_passed, bool):
            raise ProofTestReuseGoalEvidenceError("validation_passed must be boolean")
        if (
            isinstance(self.observed_at_ms, bool)
            or isinstance(self.fresh_until_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or not isinstance(self.fresh_until_ms, int)
            or self.observed_at_ms < 0
            or self.fresh_until_ms <= self.observed_at_ms
        ):
            raise ProofTestReuseGoalEvidenceError("invalid coverage freshness window")
        # Re-check retained byte identity.
        raw = _decode_retained_bytes(self.retained_validation_bytes_b64)
        _, expected_cid = _encode_retained_bytes(raw)
        if expected_cid != self.retained_validation_cid:
            raise ProofTestReuseGoalEvidenceError(
                "retained validation CID does not match retained bytes"
            )

    @property
    def interface(self) -> str:
        return ACCEPTANCE_COVERAGE_RECEIPT_INTERFACE

    @property
    def verified(self) -> bool:
        return (
            self.status is CoverageStatus.VERIFIED
            and self.validation_passed
            and self.locally_verified
        )

    @property
    def authority(self) -> str:
        return _AUTHORITATIVE if self.verified else _NO_AUTHORITY

    @property
    def receipt_cid(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROOF_TEST_REUSE_GOAL_EVIDENCE_VERSION,
            "interface": self.interface,
            "acceptance_coverage_interface": ACCEPTANCE_COVERAGE_INTERFACE,
            "requirement_id": self.requirement_id,
            "goal_id": self.goal_id,
            "status": self.status.value,
            "verified": self.verified,
            "producer_channel": self.producer_channel,
            "channel_proof_revision": self.channel_proof_revision,
            "producing_task_id": self.producing_task_id,
            "repository_id": self.repository_id,
            "repository_state_cid": self.repository_state_cid,
            "git_commit_id": self.git_commit_id,
            "git_tree_id": self.git_tree_id,
            "gitlink_state_cid": self.gitlink_state_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "dirty": self.dirty,
            "dirty_overlay_cid": self.dirty_overlay_cid,
            "policy_cid": self.policy_cid,
            "capability_cid": self.capability_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "circuit_cid": self.circuit_cid,
            "objective_revision": self.objective_revision,
            "observed_at_ms": self.observed_at_ms,
            "fresh_until_ms": self.fresh_until_ms,
            "retained_validation_bytes_b64": self.retained_validation_bytes_b64,
            "retained_validation_cid": self.retained_validation_cid,
            "locally_verified": self.locally_verified,
            "validation_passed": self.validation_passed,
            "authority": self.authority,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AcceptanceCoverageReceipt:
        body = _contract_payload(
            payload,
            schema=cls.SCHEMA,
            interface=ACCEPTANCE_COVERAGE_RECEIPT_INTERFACE,
            allowed=frozenset(
                {
                    "schema",
                    "contract_version",
                    "interface",
                    "acceptance_coverage_interface",
                    "requirement_id",
                    "goal_id",
                    "status",
                    "verified",
                    "producer_channel",
                    "channel_proof_revision",
                    "producing_task_id",
                    "repository_id",
                    "repository_state_cid",
                    "git_commit_id",
                    "git_tree_id",
                    "gitlink_state_cid",
                    "repository_forest_cid",
                    "dirty",
                    "dirty_overlay_cid",
                    "policy_cid",
                    "capability_cid",
                    "verifying_key_cid",
                    "circuit_cid",
                    "objective_revision",
                    "observed_at_ms",
                    "fresh_until_ms",
                    "retained_validation_bytes_b64",
                    "retained_validation_cid",
                    "locally_verified",
                    "validation_passed",
                    "authority",
                }
            ),
            artifact="acceptance coverage receipt",
        )
        result = cls(
            requirement_id=body.get("requirement_id", ""),
            goal_id=body.get("goal_id", ""),
            status=body.get("status", ""),
            producer_channel=body.get("producer_channel", ""),
            channel_proof_revision=body.get("channel_proof_revision", ""),
            repository_id=body.get("repository_id", ""),
            repository_state_cid=body.get("repository_state_cid", ""),
            git_commit_id=body.get("git_commit_id", ""),
            git_tree_id=body.get("git_tree_id", ""),
            gitlink_state_cid=body.get("gitlink_state_cid", ""),
            repository_forest_cid=body.get("repository_forest_cid", ""),
            dirty=body.get("dirty"),
            dirty_overlay_cid=body.get("dirty_overlay_cid", ""),
            policy_cid=body.get("policy_cid", ""),
            capability_cid=body.get("capability_cid", ""),
            verifying_key_cid=body.get("verifying_key_cid", ""),
            circuit_cid=body.get("circuit_cid", ""),
            objective_revision=body.get("objective_revision", ""),
            observed_at_ms=body.get("observed_at_ms"),
            fresh_until_ms=body.get("fresh_until_ms"),
            retained_validation_bytes_b64=body.get(
                "retained_validation_bytes_b64", ""
            ),
            retained_validation_cid=body.get("retained_validation_cid", ""),
            producing_task_id=body.get("producing_task_id", PRODUCING_TASK_ID),
            locally_verified=body.get("locally_verified"),
            validation_passed=body.get("validation_passed"),
        )
        if (
            body.get("verified") != result.verified
            or body.get("authority") != result.authority
            or body.get("acceptance_coverage_interface")
            != ACCEPTANCE_COVERAGE_INTERFACE
        ):
            raise ProofTestReuseGoalEvidenceError(
                "acceptance coverage receipt carries contradictory derived fields"
            )
        return result


@dataclass(frozen=True, slots=True)
class ProofReuseAnalyzerReceipt(CanonicalContract):
    """Typed health/exhaustion receipt for one analyzer channel."""

    SCHEMA: ClassVar[str] = PROOF_REUSE_ANALYZER_RECEIPT_SCHEMA

    analyzer_id: str
    producer_channel: str
    channel_proof_revision: str
    repository_id: str
    git_tree_id: str
    repository_forest_cid: str
    objective_revision: str
    observed_at_ms: int
    fresh_until_ms: int
    retained_validation_bytes_b64: str
    retained_validation_cid: str
    healthy: bool = True
    exhaustive: bool = True
    conclusive: bool = True
    authority: str = _AUTHORITATIVE

    def __post_init__(self) -> None:
        for name in (
            "analyzer_id",
            "producer_channel",
            "channel_proof_revision",
            "repository_id",
            "git_tree_id",
            "repository_forest_cid",
            "objective_revision",
            "retained_validation_bytes_b64",
            "retained_validation_cid",
        ):
            value = _text(getattr(self, name))
            if not value:
                raise ProofTestReuseGoalEvidenceError(f"{name} is required")
            object.__setattr__(self, name, value)
        for name in ("healthy", "exhaustive", "conclusive"):
            if not isinstance(getattr(self, name), bool):
                raise ProofTestReuseGoalEvidenceError(f"{name} must be boolean")
        object.__setattr__(self, "authority", _text(self.authority) or _NO_AUTHORITY)
        if (
            isinstance(self.observed_at_ms, bool)
            or isinstance(self.fresh_until_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or not isinstance(self.fresh_until_ms, int)
            or self.observed_at_ms < 0
            or self.fresh_until_ms <= self.observed_at_ms
        ):
            raise ProofTestReuseGoalEvidenceError("invalid analyzer freshness window")
        raw = _decode_retained_bytes(self.retained_validation_bytes_b64)
        _, expected_cid = _encode_retained_bytes(raw)
        if expected_cid != self.retained_validation_cid:
            raise ProofTestReuseGoalEvidenceError(
                "analyzer retained validation CID does not match retained bytes"
            )

    @property
    def interface(self) -> str:
        return PROOF_REUSE_ANALYZER_RECEIPT_INTERFACE

    @property
    def analyzer_health_interface(self) -> str:
        return ANALYZER_HEALTH_INTERFACE

    @property
    def receipt_cid(self) -> str:
        return self.content_id

    @property
    def authoritative(self) -> bool:
        return (
            self.authority == _AUTHORITATIVE
            and self.healthy
            and self.exhaustive
            and self.conclusive
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROOF_TEST_REUSE_GOAL_EVIDENCE_VERSION,
            "interface": self.interface,
            "analyzer_health_interface": self.analyzer_health_interface,
            "analyzer_id": self.analyzer_id,
            "producer_channel": self.producer_channel,
            "channel_proof_revision": self.channel_proof_revision,
            "repository_id": self.repository_id,
            "git_tree_id": self.git_tree_id,
            "repository_forest_cid": self.repository_forest_cid,
            "objective_revision": self.objective_revision,
            "observed_at_ms": self.observed_at_ms,
            "fresh_until_ms": self.fresh_until_ms,
            "retained_validation_bytes_b64": self.retained_validation_bytes_b64,
            "retained_validation_cid": self.retained_validation_cid,
            "healthy": self.healthy,
            "exhaustive": self.exhaustive,
            "conclusive": self.conclusive,
            "authority": self.authority,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProofReuseAnalyzerReceipt:
        body = _contract_payload(
            payload,
            schema=cls.SCHEMA,
            interface=PROOF_REUSE_ANALYZER_RECEIPT_INTERFACE,
            allowed=frozenset(
                {
                    "schema",
                    "contract_version",
                    "interface",
                    "analyzer_health_interface",
                    "analyzer_id",
                    "producer_channel",
                    "channel_proof_revision",
                    "repository_id",
                    "git_tree_id",
                    "repository_forest_cid",
                    "objective_revision",
                    "observed_at_ms",
                    "fresh_until_ms",
                    "retained_validation_bytes_b64",
                    "retained_validation_cid",
                    "healthy",
                    "exhaustive",
                    "conclusive",
                    "authority",
                }
            ),
            artifact="proof reuse analyzer receipt",
        )
        return cls(
            analyzer_id=body.get("analyzer_id", ""),
            producer_channel=body.get("producer_channel", ""),
            channel_proof_revision=body.get("channel_proof_revision", ""),
            repository_id=body.get("repository_id", ""),
            git_tree_id=body.get("git_tree_id", ""),
            repository_forest_cid=body.get("repository_forest_cid", ""),
            objective_revision=body.get("objective_revision", ""),
            observed_at_ms=body.get("observed_at_ms"),
            fresh_until_ms=body.get("fresh_until_ms"),
            retained_validation_bytes_b64=body.get(
                "retained_validation_bytes_b64", ""
            ),
            retained_validation_cid=body.get("retained_validation_cid", ""),
            healthy=body.get("healthy"),
            exhaustive=body.get("exhaustive"),
            conclusive=body.get("conclusive"),
            authority=body.get("authority", _NO_AUTHORITY),
        )


@dataclass(frozen=True, slots=True)
class ProofReusePopulationReceipt(CanonicalContract):
    """Typed adversarial population receipt (mutation / security / cross-repo)."""

    SCHEMA: ClassVar[str] = PROOF_REUSE_POPULATION_RECEIPT_SCHEMA

    population_id: str
    producer_channel: str
    channel_proof_revision: str
    repository_id: str
    git_tree_id: str
    repository_forest_cid: str
    objective_revision: str
    observed_at_ms: int
    fresh_until_ms: int
    retained_validation_bytes_b64: str
    retained_validation_cid: str
    passed: bool = True
    false_skips: int = 0
    authority: str = _AUTHORITATIVE

    def __post_init__(self) -> None:
        for name in (
            "population_id",
            "producer_channel",
            "channel_proof_revision",
            "repository_id",
            "git_tree_id",
            "repository_forest_cid",
            "objective_revision",
            "retained_validation_bytes_b64",
            "retained_validation_cid",
        ):
            value = _text(getattr(self, name))
            if not value:
                raise ProofTestReuseGoalEvidenceError(f"{name} is required")
            object.__setattr__(self, name, value)
        if not isinstance(self.passed, bool):
            raise ProofTestReuseGoalEvidenceError("passed must be boolean")
        if isinstance(self.false_skips, bool) or not isinstance(self.false_skips, int):
            raise ProofTestReuseGoalEvidenceError("false_skips must be an integer")
        if self.false_skips < 0:
            raise ProofTestReuseGoalEvidenceError("false_skips must be non-negative")
        object.__setattr__(self, "authority", _text(self.authority) or _NO_AUTHORITY)
        if (
            isinstance(self.observed_at_ms, bool)
            or isinstance(self.fresh_until_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or not isinstance(self.fresh_until_ms, int)
            or self.observed_at_ms < 0
            or self.fresh_until_ms <= self.observed_at_ms
        ):
            raise ProofTestReuseGoalEvidenceError(
                "invalid population freshness window"
            )
        raw = _decode_retained_bytes(self.retained_validation_bytes_b64)
        _, expected_cid = _encode_retained_bytes(raw)
        if expected_cid != self.retained_validation_cid:
            raise ProofTestReuseGoalEvidenceError(
                "population retained validation CID does not match retained bytes"
            )

    @property
    def interface(self) -> str:
        return PROOF_REUSE_POPULATION_RECEIPT_INTERFACE

    @property
    def population(self) -> str:
        return self.population_id

    @property
    def receipt_cid(self) -> str:
        return self.content_id

    @property
    def authoritative(self) -> bool:
        return (
            self.authority == _AUTHORITATIVE
            and self.passed is True
            and self.false_skips == 0
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROOF_TEST_REUSE_GOAL_EVIDENCE_VERSION,
            "interface": self.interface,
            "population_id": self.population_id,
            "population": self.population_id,
            "producer_channel": self.producer_channel,
            "channel_proof_revision": self.channel_proof_revision,
            "repository_id": self.repository_id,
            "git_tree_id": self.git_tree_id,
            "repository_forest_cid": self.repository_forest_cid,
            "objective_revision": self.objective_revision,
            "observed_at_ms": self.observed_at_ms,
            "fresh_until_ms": self.fresh_until_ms,
            "retained_validation_bytes_b64": self.retained_validation_bytes_b64,
            "retained_validation_cid": self.retained_validation_cid,
            "passed": self.passed,
            "false_skips": self.false_skips,
            "authority": self.authority,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProofReusePopulationReceipt:
        body = _contract_payload(
            payload,
            schema=cls.SCHEMA,
            interface=PROOF_REUSE_POPULATION_RECEIPT_INTERFACE,
            allowed=frozenset(
                {
                    "schema",
                    "contract_version",
                    "interface",
                    "population_id",
                    "population",
                    "producer_channel",
                    "channel_proof_revision",
                    "repository_id",
                    "git_tree_id",
                    "repository_forest_cid",
                    "objective_revision",
                    "observed_at_ms",
                    "fresh_until_ms",
                    "retained_validation_bytes_b64",
                    "retained_validation_cid",
                    "passed",
                    "false_skips",
                    "authority",
                }
            ),
            artifact="proof reuse population receipt",
        )
        population_id = _text(
            body.get("population_id") or body.get("population") or ""
        )
        return cls(
            population_id=population_id,
            producer_channel=body.get("producer_channel", ""),
            channel_proof_revision=body.get("channel_proof_revision", ""),
            repository_id=body.get("repository_id", ""),
            git_tree_id=body.get("git_tree_id", ""),
            repository_forest_cid=body.get("repository_forest_cid", ""),
            objective_revision=body.get("objective_revision", ""),
            observed_at_ms=body.get("observed_at_ms"),
            fresh_until_ms=body.get("fresh_until_ms"),
            retained_validation_bytes_b64=body.get(
                "retained_validation_bytes_b64", ""
            ),
            retained_validation_cid=body.get("retained_validation_cid", ""),
            passed=body.get("passed"),
            false_skips=body.get("false_skips"),
            authority=body.get("authority", _NO_AUTHORITY),
        )


@dataclass(frozen=True, slots=True)
class GoalQuorumMember:
    """One independent exhaustion / audit quorum member for goal assurance."""

    member_id: str
    evidence_channel: str
    receipt_cid: str
    healthy: bool
    exhaustive: bool
    conclusive: bool
    fresh: bool
    uncontradicted: bool
    observed_at_ms: int
    fresh_until_ms: int

    def __post_init__(self) -> None:
        for name in ("member_id", "evidence_channel", "receipt_cid"):
            value = _text(getattr(self, name))
            if not value:
                raise ProofTestReuseGoalEvidenceError(f"{name} is required")
            object.__setattr__(self, name, value)
        for name in (
            "healthy",
            "exhaustive",
            "conclusive",
            "fresh",
            "uncontradicted",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ProofTestReuseGoalEvidenceError(f"{name} must be boolean")
        if (
            isinstance(self.observed_at_ms, bool)
            or isinstance(self.fresh_until_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or not isinstance(self.fresh_until_ms, int)
            or self.observed_at_ms < 0
            or self.fresh_until_ms <= self.observed_at_ms
        ):
            raise ProofTestReuseGoalEvidenceError("invalid quorum member freshness")

    @property
    def independent_key(self) -> str:
        return f"{self.member_id}|{self.evidence_channel}|{self.receipt_cid}"

    @property
    def admissible(self) -> bool:
        return (
            self.healthy
            and self.exhaustive
            and self.conclusive
            and self.fresh
            and self.uncontradicted
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "evidence_channel": self.evidence_channel,
            "receipt_cid": self.receipt_cid,
            "healthy": self.healthy,
            "exhaustive": self.exhaustive,
            "conclusive": self.conclusive,
            "fresh": self.fresh,
            "uncontradicted": self.uncontradicted,
            "observed_at_ms": self.observed_at_ms,
            "fresh_until_ms": self.fresh_until_ms,
            "admissible": self.admissible,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoalQuorumMember:
        record = _record(payload)
        return cls(
            member_id=_text(record.get("member_id")),
            evidence_channel=_text(
                _value(record, "evidence_channel", "channel", "independence_key")
            ),
            receipt_cid=_text(_value(record, "receipt_cid", "content_id")),
            healthy=record.get("healthy") is True,
            exhaustive=record.get("exhaustive") is True,
            conclusive=record.get("conclusive") is True,
            fresh=record.get("fresh") is True
            if "fresh" in record
            else True,
            uncontradicted=record.get("uncontradicted") is not False
            and record.get("contradicted") is not True,
            observed_at_ms=int(record.get("observed_at_ms") or 0),
            fresh_until_ms=int(record.get("fresh_until_ms") or 0),
        )


@dataclass(frozen=True, slots=True)
class ProofTestReuseGoalEvidence(CanonicalContract):
    """Replayable authority packet for one goal on the current tree."""

    SCHEMA: ClassVar[str] = PROOF_TEST_REUSE_GOAL_EVIDENCE_SCHEMA

    goal_id: str
    requirement_ids: tuple[str, ...]
    coverage_receipt_cids: tuple[str, ...]
    status: str
    producer_channel: str
    channel_proof_revision: str
    repository_id: str
    repository_state_cid: str
    git_commit_id: str
    git_tree_id: str
    gitlink_state_cid: str
    repository_forest_cid: str
    dirty: bool
    dirty_overlay_cid: str
    policy_cid: str
    capability_cid: str
    verifying_key_cid: str
    circuit_cid: str
    objective_revision: str
    observed_at_ms: int
    fresh_until_ms: int
    retained_validation_bytes_b64: str
    retained_validation_cid: str
    producing_task_id: str = PRODUCING_TASK_ID
    authority: str = _AUTHORITATIVE

    def __post_init__(self) -> None:
        for name in (
            "goal_id",
            "producer_channel",
            "channel_proof_revision",
            "repository_id",
            "repository_state_cid",
            "git_commit_id",
            "git_tree_id",
            "gitlink_state_cid",
            "repository_forest_cid",
            "dirty_overlay_cid",
            "policy_cid",
            "capability_cid",
            "verifying_key_cid",
            "circuit_cid",
            "objective_revision",
            "retained_validation_bytes_b64",
            "retained_validation_cid",
            "producing_task_id",
            "status",
        ):
            value = _text(getattr(self, name))
            if not value:
                raise ProofTestReuseGoalEvidenceError(f"{name} is required")
            object.__setattr__(self, name, value)
        reqs = tuple(_text(item) for item in self.requirement_ids)
        cids = tuple(_text(item) for item in self.coverage_receipt_cids)
        if any(not item for item in reqs) or len(reqs) != len(set(reqs)):
            raise ProofTestReuseGoalEvidenceError(
                "goal evidence requirement_ids must be unique and nonempty items"
            )
        if any(not item for item in cids):
            raise ProofTestReuseGoalEvidenceError(
                "coverage_receipt_cids must be nonempty strings"
            )
        if len(reqs) != len(cids):
            raise ProofTestReuseGoalEvidenceError(
                "coverage receipt population must match requirement population"
            )
        object.__setattr__(self, "requirement_ids", reqs)
        object.__setattr__(self, "coverage_receipt_cids", cids)
        if not isinstance(self.dirty, bool):
            raise ProofTestReuseGoalEvidenceError("dirty must be boolean")
        object.__setattr__(self, "authority", _text(self.authority) or _NO_AUTHORITY)
        if (
            isinstance(self.observed_at_ms, bool)
            or isinstance(self.fresh_until_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or not isinstance(self.fresh_until_ms, int)
            or self.observed_at_ms < 0
            or self.fresh_until_ms <= self.observed_at_ms
        ):
            raise ProofTestReuseGoalEvidenceError("invalid goal evidence freshness")
        raw = _decode_retained_bytes(self.retained_validation_bytes_b64)
        _, expected_cid = _encode_retained_bytes(raw)
        if expected_cid != self.retained_validation_cid:
            raise ProofTestReuseGoalEvidenceError(
                "goal evidence retained validation CID does not match retained bytes"
            )

    @property
    def interface(self) -> str:
        return PROOF_TEST_REUSE_GOAL_EVIDENCE_INTERFACE

    @property
    def authoritative(self) -> bool:
        return self.authority == _AUTHORITATIVE and self.status == "verified_complete"

    @property
    def evidence_cid(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROOF_TEST_REUSE_GOAL_EVIDENCE_VERSION,
            "interface": self.interface,
            "goal_id": self.goal_id,
            "requirement_ids": list(self.requirement_ids),
            "coverage_receipt_cids": list(self.coverage_receipt_cids),
            "status": self.status,
            "producer_channel": self.producer_channel,
            "channel_proof_revision": self.channel_proof_revision,
            "producing_task_id": self.producing_task_id,
            "repository_id": self.repository_id,
            "repository_state_cid": self.repository_state_cid,
            "git_commit_id": self.git_commit_id,
            "git_tree_id": self.git_tree_id,
            "gitlink_state_cid": self.gitlink_state_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "dirty": self.dirty,
            "dirty_overlay_cid": self.dirty_overlay_cid,
            "policy_cid": self.policy_cid,
            "capability_cid": self.capability_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "circuit_cid": self.circuit_cid,
            "objective_revision": self.objective_revision,
            "observed_at_ms": self.observed_at_ms,
            "fresh_until_ms": self.fresh_until_ms,
            "retained_validation_bytes_b64": self.retained_validation_bytes_b64,
            "retained_validation_cid": self.retained_validation_cid,
            "authority": self.authority,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProofTestReuseGoalEvidence:
        body = _contract_payload(
            payload,
            schema=cls.SCHEMA,
            interface=PROOF_TEST_REUSE_GOAL_EVIDENCE_INTERFACE,
            allowed=frozenset(
                {
                    "schema",
                    "contract_version",
                    "interface",
                    "goal_id",
                    "requirement_ids",
                    "coverage_receipt_cids",
                    "status",
                    "producer_channel",
                    "channel_proof_revision",
                    "producing_task_id",
                    "repository_id",
                    "repository_state_cid",
                    "git_commit_id",
                    "git_tree_id",
                    "gitlink_state_cid",
                    "repository_forest_cid",
                    "dirty",
                    "dirty_overlay_cid",
                    "policy_cid",
                    "capability_cid",
                    "verifying_key_cid",
                    "circuit_cid",
                    "objective_revision",
                    "observed_at_ms",
                    "fresh_until_ms",
                    "retained_validation_bytes_b64",
                    "retained_validation_cid",
                    "authority",
                }
            ),
            artifact="proof-test-reuse goal evidence",
        )
        reqs = body.get("requirement_ids")
        cids = body.get("coverage_receipt_cids")
        if not isinstance(reqs, list) or not isinstance(cids, list):
            raise ProofTestReuseGoalEvidenceError(
                "goal evidence populations must be lists"
            )
        return cls(
            goal_id=body.get("goal_id", ""),
            requirement_ids=tuple(reqs),
            coverage_receipt_cids=tuple(cids),
            status=body.get("status", ""),
            producer_channel=body.get("producer_channel", ""),
            channel_proof_revision=body.get("channel_proof_revision", ""),
            repository_id=body.get("repository_id", ""),
            repository_state_cid=body.get("repository_state_cid", ""),
            git_commit_id=body.get("git_commit_id", ""),
            git_tree_id=body.get("git_tree_id", ""),
            gitlink_state_cid=body.get("gitlink_state_cid", ""),
            repository_forest_cid=body.get("repository_forest_cid", ""),
            dirty=body.get("dirty"),
            dirty_overlay_cid=body.get("dirty_overlay_cid", ""),
            policy_cid=body.get("policy_cid", ""),
            capability_cid=body.get("capability_cid", ""),
            verifying_key_cid=body.get("verifying_key_cid", ""),
            circuit_cid=body.get("circuit_cid", ""),
            objective_revision=body.get("objective_revision", ""),
            observed_at_ms=body.get("observed_at_ms"),
            fresh_until_ms=body.get("fresh_until_ms"),
            retained_validation_bytes_b64=body.get(
                "retained_validation_bytes_b64", ""
            ),
            retained_validation_cid=body.get("retained_validation_cid", ""),
            producing_task_id=body.get("producing_task_id", PRODUCING_TASK_ID),
            authority=body.get("authority", _NO_AUTHORITY),
        )


@dataclass(frozen=True, slots=True)
class GoalAssuranceResult(CanonicalContract):
    """Atomic outcome of one goal-assurance evaluation on a discovered heap."""

    SCHEMA: ClassVar[str] = GOAL_ASSURANCE_RESULT_SCHEMA

    required_requirement_ids: tuple[str, ...]
    coverage_receipts: tuple[AcceptanceCoverageReceipt, ...]
    goal_evidence: tuple[ProofTestReuseGoalEvidence, ...]
    analyzer_receipts: tuple[ProofReuseAnalyzerReceipt, ...]
    population_receipts: tuple[ProofReusePopulationReceipt, ...]
    quorum_members: tuple[GoalQuorumMember, ...]
    gaps: tuple[GoalEvidenceGap, ...]
    unavailable_capabilities: tuple[str, ...]
    evaluated_at_ms: int
    objective_revision: str
    repository_forest_cid: str
    git_tree_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_requirement_ids",
            tuple(_text(item) for item in self.required_requirement_ids),
        )
        object.__setattr__(self, "coverage_receipts", tuple(self.coverage_receipts))
        object.__setattr__(self, "goal_evidence", tuple(self.goal_evidence))
        object.__setattr__(self, "analyzer_receipts", tuple(self.analyzer_receipts))
        object.__setattr__(
            self, "population_receipts", tuple(self.population_receipts)
        )
        object.__setattr__(self, "quorum_members", tuple(self.quorum_members))
        object.__setattr__(self, "gaps", tuple(self.gaps))
        object.__setattr__(
            self,
            "unavailable_capabilities",
            tuple(sorted({_normalize_capability_name(item) for item in self.unavailable_capabilities if _text(item)})),
        )
        object.__setattr__(self, "objective_revision", _text(self.objective_revision))
        object.__setattr__(
            self, "repository_forest_cid", _text(self.repository_forest_cid)
        )
        object.__setattr__(self, "git_tree_id", _text(self.git_tree_id))
        if (
            isinstance(self.evaluated_at_ms, bool)
            or not isinstance(self.evaluated_at_ms, int)
            or self.evaluated_at_ms < 0
        ):
            raise ProofTestReuseGoalEvidenceError(
                "evaluated_at_ms must be a nonnegative integer"
            )
        if not all(
            isinstance(item, AcceptanceCoverageReceipt)
            for item in self.coverage_receipts
        ):
            raise ProofTestReuseGoalEvidenceError("coverage receipts have wrong type")
        if not all(
            isinstance(item, ProofTestReuseGoalEvidence) for item in self.goal_evidence
        ):
            raise ProofTestReuseGoalEvidenceError("goal evidence has wrong type")
        if not all(
            isinstance(item, ProofReuseAnalyzerReceipt)
            for item in self.analyzer_receipts
        ):
            raise ProofTestReuseGoalEvidenceError("analyzer receipts have wrong type")
        if not all(
            isinstance(item, ProofReusePopulationReceipt)
            for item in self.population_receipts
        ):
            raise ProofTestReuseGoalEvidenceError("population receipts have wrong type")
        if not all(isinstance(item, GoalQuorumMember) for item in self.quorum_members):
            raise ProofTestReuseGoalEvidenceError("quorum members have wrong type")
        if not all(isinstance(item, GoalEvidenceGap) for item in self.gaps):
            raise ProofTestReuseGoalEvidenceError("gaps have wrong type")

    @property
    def interface(self) -> str:
        return GOAL_ASSURANCE_RESULT_INTERFACE

    @property
    def exhaustion_quorum_interface(self) -> str:
        return EXHAUSTION_QUORUM_INTERFACE

    @property
    def coverage_by_requirement(self) -> dict[str, AcceptanceCoverageReceipt]:
        return {item.requirement_id: item for item in self.coverage_receipts}

    @property
    def evidence_by_goal(self) -> dict[str, ProofTestReuseGoalEvidence]:
        return {item.goal_id: item for item in self.goal_evidence}

    @property
    def analyzer_by_id(self) -> dict[str, ProofReuseAnalyzerReceipt]:
        return {item.analyzer_id: item for item in self.analyzer_receipts}

    @property
    def population_by_id(self) -> dict[str, ProofReusePopulationReceipt]:
        return {item.population_id: item for item in self.population_receipts}

    @property
    def quorum_satisfied(self) -> bool:
        admissible = [member for member in self.quorum_members if member.admissible]
        if len(admissible) < REQUIRED_QUORUM_MEMBERS:
            return False
        member_ids = {item.member_id for item in admissible}
        channels = {item.evidence_channel for item in admissible}
        receipts = {item.receipt_cid for item in admissible}
        return (
            len(member_ids) >= REQUIRED_QUORUM_MEMBERS
            and len(channels) >= REQUIRED_QUORUM_MEMBERS
            and len(receipts) >= REQUIRED_QUORUM_MEMBERS
        )

    @property
    def populations_passed(self) -> bool:
        if set(self.population_by_id) != REQUIRED_ADVERSARIAL_POPULATIONS:
            return False
        return all(
            item.passed is True and item.false_skips == 0
            for item in self.population_receipts
        )

    @property
    def analyzers_healthy(self) -> bool:
        if set(self.analyzer_by_id) != REQUIRED_ANALYZER_CHANNELS:
            return False
        return all(
            item.healthy and item.exhaustive and item.conclusive
            for item in self.analyzer_receipts
        )

    @property
    def authoritative(self) -> bool:
        if self.gaps:
            return False
        if not self.required_requirement_ids:
            return False
        if set(self.coverage_by_requirement) != set(self.required_requirement_ids):
            return False
        if not all(item.verified for item in self.coverage_receipts):
            return False
        if not self.analyzers_healthy:
            return False
        if not self.populations_passed:
            return False
        if not self.quorum_satisfied:
            return False
        return all(item.authoritative for item in self.goal_evidence)

    @property
    def authority(self) -> str:
        return _AUTHORITATIVE if self.authoritative else _NO_AUTHORITY

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROOF_TEST_REUSE_GOAL_EVIDENCE_VERSION,
            "interface": self.interface,
            "exhaustion_quorum_interface": self.exhaustion_quorum_interface,
            "required_requirement_ids": list(self.required_requirement_ids),
            "coverage_receipts": [item.to_record() for item in self.coverage_receipts],
            "goal_evidence": [item.to_record() for item in self.goal_evidence],
            "analyzer_receipts": [item.to_record() for item in self.analyzer_receipts],
            "population_receipts": [
                item.to_record() for item in self.population_receipts
            ],
            "quorum_members": [item.to_dict() for item in self.quorum_members],
            "gaps": [item.to_record() for item in self.gaps],
            "unavailable_capabilities": list(self.unavailable_capabilities),
            "evaluated_at_ms": self.evaluated_at_ms,
            "objective_revision": self.objective_revision,
            "repository_forest_cid": self.repository_forest_cid,
            "git_tree_id": self.git_tree_id,
            "authority": self.authority,
            "quorum_satisfied": self.quorum_satisfied,
            "populations_passed": self.populations_passed,
            "analyzers_healthy": self.analyzers_healthy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoalAssuranceResult:
        body = _contract_payload(
            payload,
            schema=cls.SCHEMA,
            interface=GOAL_ASSURANCE_RESULT_INTERFACE,
            allowed=frozenset(
                {
                    "schema",
                    "contract_version",
                    "interface",
                    "exhaustion_quorum_interface",
                    "required_requirement_ids",
                    "coverage_receipts",
                    "goal_evidence",
                    "analyzer_receipts",
                    "population_receipts",
                    "quorum_members",
                    "gaps",
                    "unavailable_capabilities",
                    "evaluated_at_ms",
                    "objective_revision",
                    "repository_forest_cid",
                    "git_tree_id",
                    "authority",
                    "quorum_satisfied",
                    "populations_passed",
                    "analyzers_healthy",
                }
            ),
            artifact="goal assurance result",
        )
        required = body.get("required_requirement_ids")
        coverage = body.get("coverage_receipts")
        goals = body.get("goal_evidence")
        analyzers = body.get("analyzer_receipts")
        populations = body.get("population_receipts")
        quorum = body.get("quorum_members")
        gaps = body.get("gaps")
        unavailable = body.get("unavailable_capabilities")
        if not all(
            isinstance(item, list)
            for item in (
                required,
                coverage,
                goals,
                analyzers,
                populations,
                quorum,
                gaps,
                unavailable,
            )
        ):
            raise ProofTestReuseGoalEvidenceError(
                "goal assurance result populations must be lists"
            )
        result = cls(
            required_requirement_ids=tuple(required),
            coverage_receipts=tuple(
                AcceptanceCoverageReceipt.from_dict(item) for item in coverage
            ),
            goal_evidence=tuple(
                ProofTestReuseGoalEvidence.from_dict(item) for item in goals
            ),
            analyzer_receipts=tuple(
                ProofReuseAnalyzerReceipt.from_dict(item) for item in analyzers
            ),
            population_receipts=tuple(
                ProofReusePopulationReceipt.from_dict(item) for item in populations
            ),
            quorum_members=tuple(GoalQuorumMember.from_dict(item) for item in quorum),
            gaps=tuple(GoalEvidenceGap.from_dict(item) for item in gaps),
            unavailable_capabilities=tuple(unavailable),
            evaluated_at_ms=body.get("evaluated_at_ms"),
            objective_revision=body.get("objective_revision", ""),
            repository_forest_cid=body.get("repository_forest_cid", ""),
            git_tree_id=body.get("git_tree_id", ""),
        )
        for name in (
            "authority",
            "quorum_satisfied",
            "populations_passed",
            "analyzers_healthy",
        ):
            if name in body and body.get(name) != getattr(result, name):
                raise ProofTestReuseGoalEvidenceError(
                    f"goal assurance result {name} is contradictory"
                )
        if (
            body.get("exhaustion_quorum_interface")
            != EXHAUSTION_QUORUM_INTERFACE
        ):
            raise ProofTestReuseGoalEvidenceError(
                "goal assurance result quorum interface is contradictory"
            )
        return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoalAssuranceRunner:
    """Collect independent goal coverage and analyzer/population receipts."""

    repository_id: str
    repository_state_cid: str
    git_commit_id: str
    git_tree_id: str
    gitlink_state_cid: str
    repository_forest_cid: str
    dirty: bool
    dirty_overlay_cid: str
    objective_revision: str
    policy_cid: str
    capability_cid: str
    verifying_key_cid: str
    circuit_cid: str
    producer_channel: str = DEFAULT_PRODUCER_CHANNEL
    channel_proof_revision: str = DEFAULT_CHANNEL_PROOF_REVISION
    freshness_seconds: float = DEFAULT_EVIDENCE_FRESHNESS_SECONDS
    clock: Callable[[], float] = field(default=time.time, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "repository_state_cid",
            "git_commit_id",
            "git_tree_id",
            "gitlink_state_cid",
            "repository_forest_cid",
            "dirty_overlay_cid",
            "objective_revision",
            "policy_cid",
            "capability_cid",
            "verifying_key_cid",
            "circuit_cid",
            "producer_channel",
            "channel_proof_revision",
        ):
            value = _text(getattr(self, name))
            if not value:
                raise ProofTestReuseGoalEvidenceError(f"{name} is required")
            object.__setattr__(self, name, value)
        if not isinstance(self.dirty, bool):
            raise ProofTestReuseGoalEvidenceError("dirty must be boolean")
        if (
            isinstance(self.freshness_seconds, bool)
            or not isinstance(self.freshness_seconds, (int, float))
            or not 0 < float(self.freshness_seconds) <= 3_600
        ):
            raise ProofTestReuseGoalEvidenceError("freshness_seconds is invalid")

    def collect(
        self,
        objective_heap: str | Path | Sequence[ObjectiveGoal] | Mapping[str, Any],
        *,
        validation_by_requirement: Mapping[str, Any] | None = None,
        analyzer_inputs: Iterable[Any] = (),
        population_inputs: Iterable[Any] = (),
        quorum_inputs: Iterable[Any] = (),
        capability_facts: Mapping[str, Any] | None = None,
        certificate_assurance: Any = None,
        benchmark_receipt: Any = None,
        rollout_decision: Any = None,
        requirement_registry: Any = None,
    ) -> GoalAssuranceResult:
        """Evaluate heap-discovered requirements against retained premises.

        ``requirement_registry`` is intentionally rejected: population must
        come from the objective heap alone.
        """

        now_ms = _clock_milliseconds(self.clock)
        gaps: list[GoalEvidenceGap] = []

        if requirement_registry is not None:
            gaps.append(
                GoalEvidenceGap(
                    subject_id="*",
                    kind=GoalEvidenceGapKind.REQUIREMENT_REGISTRY_FORBIDDEN,
                    detail=(
                        "requirement IDs must be discovered from the objective "
                        "heap rather than a per-test registry"
                    ),
                )
            )

        try:
            goals = load_objective_goals(objective_heap)
        except (OSError, ProofTestReuseGoalEvidenceError, ValueError, TypeError) as exc:
            return self._terminal(
                required=(),
                gaps=[
                    GoalEvidenceGap(
                        subject_id="*",
                        kind=GoalEvidenceGapKind.HEAP_MALFORMED,
                        detail=f"objective heap could not be loaded: {exc}"[:_MAX_DETAIL],
                    ),
                    *gaps,
                ],
                now_ms=now_ms,
            )

        if not goals:
            return self._terminal(
                required=(),
                gaps=[
                    GoalEvidenceGap(
                        subject_id="*",
                        kind=GoalEvidenceGapKind.HEAP_MALFORMED,
                        detail="objective heap contains no goals",
                    ),
                    *gaps,
                ],
                now_ms=now_ms,
            )

        by_goal: dict[str, tuple[str, ...]] = {}
        required: list[str] = []
        for goal in goals:
            reqs = goal_requirement_ids(goal)
            if not reqs:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=goal.goal_id,
                        kind=GoalEvidenceGapKind.HEAP_REQUIREMENT_MISSING,
                        detail=(
                            "goal declares no machine acceptance criteria / "
                            "evidence requirement IDs (or criteria contradict evidence)"
                        ),
                    )
                )
                continue
            by_goal[goal.goal_id] = reqs
            for requirement_id in reqs:
                if requirement_id not in required:
                    required.append(requirement_id)

        if not required:
            return self._terminal(required=(), gaps=gaps, now_ms=now_ms)

        capability_facts = dict(capability_facts or {})
        unavailable = self._unavailable_capabilities(capability_facts)
        for name in unavailable:
            # Typed and non-blocking: recorded as a capability fact gap that
            # never alone fails non-ZK criteria.
            gaps.append(
                GoalEvidenceGap(
                    subject_id=f"capability:{name}",
                    kind=GoalEvidenceGapKind.CAPABILITY_UNAVAILABLE,
                    detail=(
                        f"optional capability {name!r} is unavailable; "
                        "tests continue, but real-ZK / production-warm criteria "
                        "remain unverified without a reviewed real certificate"
                    ),
                )
            )

        cert_record = _record(certificate_assurance)
        has_real_certificate = _is_real_certificate_assurance(cert_record)

        # Benchmark / rollout premises (deployment authority only when genuine).
        benchmark = _record(benchmark_receipt)
        if benchmark:
            if _is_synthetic_benchmark(benchmark):
                gaps.append(
                    GoalEvidenceGap(
                        subject_id="benchmark",
                        kind=GoalEvidenceGapKind.SYNTHETIC_BENCHMARK_AUTHORITY,
                        detail=(
                            "synthetic _AlwaysVerify benchmark data is never "
                            "deployment authority"
                        ),
                    )
                )
            else:
                authority = _text(benchmark.get("authority")).lower()
                if authority and authority != _AUTHORITATIVE:
                    gaps.append(
                        GoalEvidenceGap(
                            subject_id="benchmark",
                            kind=GoalEvidenceGapKind.BENCHMARK_NON_AUTHORITATIVE,
                            detail="benchmark receipt authority is not authoritative",
                        )
                    )

        rollout = _record(rollout_decision)
        # Rollout is optional at the runner boundary; missing rollout only
        # blocks G110 rollout criteria via coverage inputs.

        validations = {
            _text(key): _record(value)
            for key, value in dict(validation_by_requirement or {}).items()
            if _text(key)
        }

        coverage_receipts: list[AcceptanceCoverageReceipt] = []
        coverage_by_req: dict[str, AcceptanceCoverageReceipt] = {}
        goal_to_req_goal: dict[str, str] = {}
        for goal_id, reqs in by_goal.items():
            for requirement_id in reqs:
                goal_to_req_goal[requirement_id] = goal_id

        for requirement_id in required:
            goal_id = goal_to_req_goal[requirement_id]
            raw = validations.get(requirement_id, {})
            receipt, gap = self._coverage_receipt(
                requirement_id=requirement_id,
                goal_id=goal_id,
                raw=raw,
                now_ms=now_ms,
                unavailable=unavailable,
                has_real_certificate=has_real_certificate,
            )
            if gap is not None:
                gaps.append(gap)
            if receipt is not None:
                coverage_receipts.append(receipt)
                coverage_by_req[requirement_id] = receipt

        analyzer_receipts, analyzer_gaps = self._analyzer_receipts(
            analyzer_inputs, now_ms=now_ms
        )
        gaps.extend(analyzer_gaps)

        population_receipts, population_gaps = self._population_receipts(
            population_inputs, now_ms=now_ms
        )
        gaps.extend(population_gaps)

        quorum_members, quorum_gaps = self._quorum_members(
            quorum_inputs, now_ms=now_ms
        )
        gaps.extend(quorum_gaps)

        # Build per-goal evidence only when every declared requirement for that
        # goal has a verified coverage receipt.
        goal_evidence: list[ProofTestReuseGoalEvidence] = []
        for goal_id, reqs in sorted(by_goal.items()):
            receipts = [coverage_by_req[item] for item in reqs if item in coverage_by_req]
            if len(receipts) != len(reqs) or not all(item.verified for item in receipts):
                continue
            observed = min(item.observed_at_ms for item in receipts)
            fresh_until = min(item.fresh_until_ms for item in receipts)
            retained_payload = {
                "goal_id": goal_id,
                "requirement_ids": list(reqs),
                "coverage_receipt_cids": [item.receipt_cid for item in receipts],
                "objective_revision": self.objective_revision,
                "repository_forest_cid": self.repository_forest_cid,
                "git_tree_id": self.git_tree_id,
            }
            retained_b64, retained_cid = _encode_retained_bytes(retained_payload)
            goal_evidence.append(
                ProofTestReuseGoalEvidence(
                    goal_id=goal_id,
                    requirement_ids=reqs,
                    coverage_receipt_cids=tuple(item.receipt_cid for item in receipts),
                    status="verified_complete",
                    producer_channel=self.producer_channel,
                    channel_proof_revision=self.channel_proof_revision,
                    repository_id=self.repository_id,
                    repository_state_cid=self.repository_state_cid,
                    git_commit_id=self.git_commit_id,
                    git_tree_id=self.git_tree_id,
                    gitlink_state_cid=self.gitlink_state_cid,
                    repository_forest_cid=self.repository_forest_cid,
                    dirty=self.dirty,
                    dirty_overlay_cid=self.dirty_overlay_cid,
                    policy_cid=self.policy_cid,
                    capability_cid=self.capability_cid,
                    verifying_key_cid=self.verifying_key_cid,
                    circuit_cid=self.circuit_cid,
                    objective_revision=self.objective_revision,
                    observed_at_ms=observed,
                    fresh_until_ms=fresh_until,
                    retained_validation_bytes_b64=retained_b64,
                    retained_validation_cid=retained_cid,
                    authority=_AUTHORITATIVE,
                )
            )

        # Capability-unavailable gaps are informational when they do not leave
        # any real-ZK / warm criterion claimed-verified.  Filter them out of the
        # authoritative gap set only when every such criterion is either absent
        # from the required population or already covered by a real certificate
        # path (those criteria emit REAL_ZK_UNVERIFIED / PRODUCTION_WARM_UNVERIFIED
        # gaps instead).
        filtered_gaps = self._filter_informational_capability_gaps(gaps, required)

        return GoalAssuranceResult(
            required_requirement_ids=tuple(required),
            coverage_receipts=tuple(
                sorted(coverage_receipts, key=lambda item: item.requirement_id)
            ),
            goal_evidence=tuple(
                sorted(goal_evidence, key=lambda item: item.goal_id)
            ),
            analyzer_receipts=tuple(
                sorted(analyzer_receipts, key=lambda item: item.analyzer_id)
            ),
            population_receipts=tuple(
                sorted(population_receipts, key=lambda item: item.population_id)
            ),
            quorum_members=tuple(
                sorted(quorum_members, key=lambda item: item.member_id)
            ),
            gaps=tuple(filtered_gaps),
            unavailable_capabilities=tuple(sorted(unavailable)),
            evaluated_at_ms=now_ms,
            objective_revision=self.objective_revision,
            repository_forest_cid=self.repository_forest_cid,
            git_tree_id=self.git_tree_id,
        )

    def _terminal(
        self,
        *,
        required: Sequence[str],
        gaps: Sequence[GoalEvidenceGap],
        now_ms: int,
    ) -> GoalAssuranceResult:
        return GoalAssuranceResult(
            required_requirement_ids=tuple(required),
            coverage_receipts=(),
            goal_evidence=(),
            analyzer_receipts=(),
            population_receipts=(),
            quorum_members=(),
            gaps=tuple(gaps),
            unavailable_capabilities=(),
            evaluated_at_ms=now_ms,
            objective_revision=self.objective_revision,
            repository_forest_cid=self.repository_forest_cid,
            git_tree_id=self.git_tree_id,
        )

    def _unavailable_capabilities(
        self, facts: Mapping[str, Any]
    ) -> tuple[str, ...]:
        missing: list[str] = []
        for name in sorted(OPTIONAL_CAPABILITY_NAMES):
            available = _capability_available(facts, name)
            if available is False:
                missing.append(name)
        return tuple(missing)

    def _is_fresh(self, observed_at_ms: int, fresh_until_ms: int, now_ms: int) -> bool:
        max_age_ms = int(float(self.freshness_seconds) * 1_000)
        return (
            observed_at_ms <= now_ms <= fresh_until_ms
            and now_ms - observed_at_ms <= max_age_ms
        )

    def _bindings_match(self, raw: Mapping[str, Any]) -> bool:
        expected = {
            "repository_id": self.repository_id,
            "repository_state_cid": self.repository_state_cid,
            "git_commit_id": self.git_commit_id,
            "git_tree_id": self.git_tree_id,
            "gitlink_state_cid": self.gitlink_state_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "dirty_overlay_cid": self.dirty_overlay_cid,
            "objective_revision": self.objective_revision,
            "policy_cid": self.policy_cid,
            "capability_cid": self.capability_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "circuit_cid": self.circuit_cid,
        }
        for name, wanted in expected.items():
            actual = _text(_value(raw, name, "commit_id" if name == "git_commit_id" else ""))
            if actual and actual != wanted:
                return False
        dirty = _boolean(raw, "dirty")
        if dirty is not None and dirty is not self.dirty:
            return False
        return True

    def _coverage_receipt(
        self,
        *,
        requirement_id: str,
        goal_id: str,
        raw: Mapping[str, Any],
        now_ms: int,
        unavailable: Sequence[str],
        has_real_certificate: bool,
    ) -> tuple[AcceptanceCoverageReceipt | None, GoalEvidenceGap | None]:
        backends_down = bool(
            set(unavailable).intersection({"groth16", "provekit"})
        )
        store_down = bool(set(unavailable).intersection({"cache", "ipfs"}))

        # Unavailable proving backends leave real-ZK criteria unverified unless
        # a reviewed, locally verifiable real certificate is present.
        if (
            requirement_id in REAL_ZK_REQUIREMENT_IDS
            and backends_down
            and not has_real_certificate
        ):
            return None, GoalEvidenceGap(
                subject_id=requirement_id,
                kind=GoalEvidenceGapKind.REAL_ZK_UNVERIFIED,
                detail=(
                    "real-ZK criterion remains unverified: Groth16/ProveKit "
                    "unavailable and no reviewed locally verifiable real "
                    "certificate is present"
                ),
            )

        # Unavailable cache/IPFS leave production-warm criteria unverified
        # unless a reviewed real certificate covers the warm path.
        if (
            requirement_id in PRODUCTION_WARM_REQUIREMENT_IDS
            and store_down
            and not has_real_certificate
        ):
            return None, GoalEvidenceGap(
                subject_id=requirement_id,
                kind=GoalEvidenceGapKind.PRODUCTION_WARM_UNVERIFIED,
                detail=(
                    "production-warm criterion remains unverified while "
                    "cache/IPFS is unavailable and no reviewed real "
                    "certificate was supplied"
                ),
            )

        if not raw:
            return None, GoalEvidenceGap(
                subject_id=requirement_id,
                kind=GoalEvidenceGapKind.COVERAGE_MISSING,
                detail="no retained validation was supplied for this requirement",
            )

        producer_channel = _text(
            _value(raw, "producer_channel", "channel") or self.producer_channel
        )
        channel_revision = _text(
            _value(raw, "channel_proof_revision", "channel_revision")
            or self.channel_proof_revision
        )
        if not producer_channel:
            return None, GoalEvidenceGap(
                subject_id=requirement_id,
                kind=GoalEvidenceGapKind.PRODUCER_CHANNEL_MISSING,
                detail="coverage input lacks producer_channel",
            )
        if not channel_revision:
            return None, GoalEvidenceGap(
                subject_id=requirement_id,
                kind=GoalEvidenceGapKind.CHANNEL_PROOF_REVISION_MISSING,
                detail="coverage input lacks channel_proof_revision",
            )
        if not self._bindings_match(raw):
            return None, GoalEvidenceGap(
                subject_id=requirement_id,
                kind=GoalEvidenceGapKind.COVERAGE_BINDING_MISMATCH,
                detail="coverage input is not bound to the current identities",
            )

        observed = _integer(raw, "observed_at_ms")
        fresh_until = _integer(raw, "fresh_until_ms")
        if observed is None or fresh_until is None:
            return None, GoalEvidenceGap(
                subject_id=requirement_id,
                kind=GoalEvidenceGapKind.MALFORMED,
                detail="coverage input lacks observed/fresh-until window",
            )
        if not self._is_fresh(observed, fresh_until, now_ms):
            return None, GoalEvidenceGap(
                subject_id=requirement_id,
                kind=GoalEvidenceGapKind.COVERAGE_STALE,
                detail="coverage input is outside the freshness window",
            )

        passed = _boolean(raw, "passed", "validation_passed")
        if passed is False or (
            _text(_value(raw, "status", "disposition")).lower()
            in {"failed", "error", "rejected"}
        ):
            return None, GoalEvidenceGap(
                subject_id=requirement_id,
                kind=GoalEvidenceGapKind.COVERAGE_FAILED,
                detail="coverage validation did not pass",
            )
        if passed is not True and raw.get("verified") is not True:
            # Allow explicit status=passed spelling.
            if _text(raw.get("status")).lower() not in {"passed", "verified", "ok"}:
                return None, GoalEvidenceGap(
                    subject_id=requirement_id,
                    kind=GoalEvidenceGapKind.COVERAGE_FAILED,
                    detail="coverage validation did not explicitly pass",
                )

        retained_b64 = _text(
            _value(raw, "retained_validation_bytes_b64", "retained_bytes_b64")
        )
        retained_cid = _text(
            _value(raw, "retained_validation_cid", "retained_bytes_cid")
        )
        if not retained_b64:
            # Derive retained bytes from the validation payload itself.
            try:
                retained_b64, derived_cid = _encode_retained_bytes(dict(raw))
            except ProofTestReuseGoalEvidenceError:
                return None, GoalEvidenceGap(
                    subject_id=requirement_id,
                    kind=GoalEvidenceGapKind.RETAINED_BYTES_MISSING,
                    detail="coverage input has no retained validation bytes",
                )
            if retained_cid and retained_cid != derived_cid:
                return None, GoalEvidenceGap(
                    subject_id=requirement_id,
                    kind=GoalEvidenceGapKind.RETAINED_BYTES_MISMATCH,
                    detail="declared retained validation CID does not match bytes",
                )
            retained_cid = derived_cid
        else:
            try:
                raw_bytes = _decode_retained_bytes(retained_b64)
                _, derived_cid = _encode_retained_bytes(raw_bytes)
            except ProofTestReuseGoalEvidenceError:
                return None, GoalEvidenceGap(
                    subject_id=requirement_id,
                    kind=GoalEvidenceGapKind.RETAINED_BYTES_MISMATCH,
                    detail="retained validation bytes could not be authenticated",
                )
            if retained_cid and retained_cid != derived_cid:
                return None, GoalEvidenceGap(
                    subject_id=requirement_id,
                    kind=GoalEvidenceGapKind.RETAINED_BYTES_MISMATCH,
                    detail="declared retained validation CID does not match bytes",
                )
            retained_cid = retained_cid or derived_cid

        try:
            receipt = AcceptanceCoverageReceipt(
                requirement_id=requirement_id,
                goal_id=goal_id,
                status=CoverageStatus.VERIFIED,
                producer_channel=producer_channel,
                channel_proof_revision=channel_revision,
                repository_id=self.repository_id,
                repository_state_cid=self.repository_state_cid,
                git_commit_id=self.git_commit_id,
                git_tree_id=self.git_tree_id,
                gitlink_state_cid=self.gitlink_state_cid,
                repository_forest_cid=self.repository_forest_cid,
                dirty=self.dirty,
                dirty_overlay_cid=self.dirty_overlay_cid,
                policy_cid=self.policy_cid,
                capability_cid=self.capability_cid,
                verifying_key_cid=self.verifying_key_cid,
                circuit_cid=self.circuit_cid,
                objective_revision=self.objective_revision,
                observed_at_ms=observed,
                fresh_until_ms=fresh_until,
                retained_validation_bytes_b64=retained_b64,
                retained_validation_cid=retained_cid,
                validation_passed=True,
                locally_verified=True,
            )
        except ProofTestReuseGoalEvidenceError as exc:
            return None, GoalEvidenceGap(
                subject_id=requirement_id,
                kind=GoalEvidenceGapKind.MALFORMED,
                detail=str(exc)[:_MAX_DETAIL],
            )
        return receipt, None

    def _analyzer_receipts(
        self, inputs: Iterable[Any], *, now_ms: int
    ) -> tuple[list[ProofReuseAnalyzerReceipt], list[GoalEvidenceGap]]:
        gaps: list[GoalEvidenceGap] = []
        by_id: dict[str, ProofReuseAnalyzerReceipt] = {}
        for raw_value in inputs:
            raw = _record(raw_value)
            analyzer_id = _text(_value(raw, "analyzer_id", "id", "channel"))
            if not analyzer_id:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id="analyzer",
                        kind=GoalEvidenceGapKind.MALFORMED,
                        detail="analyzer input is missing analyzer_id",
                    )
                )
                continue
            if analyzer_id in by_id:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=analyzer_id,
                        kind=GoalEvidenceGapKind.UNEXPECTED_INPUT,
                        detail="duplicate analyzer receipt",
                    )
                )
                continue
            if analyzer_id not in REQUIRED_ANALYZER_CHANNELS:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=analyzer_id,
                        kind=GoalEvidenceGapKind.UNEXPECTED_INPUT,
                        detail="analyzer is outside the required channel population",
                    )
                )
                continue
            producer_channel = _text(
                raw.get("producer_channel") or f"analyzer:{analyzer_id}"
            )
            channel_revision = _text(
                raw.get("channel_proof_revision") or self.channel_proof_revision
            )
            if not self._bindings_match(raw):
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=analyzer_id,
                        kind=GoalEvidenceGapKind.COVERAGE_BINDING_MISMATCH,
                        detail="analyzer receipt is not bound to current identities",
                    )
                )
                continue
            observed = _integer(raw, "observed_at_ms")
            fresh_until = _integer(raw, "fresh_until_ms")
            if (
                observed is None
                or fresh_until is None
                or not self._is_fresh(observed, fresh_until, now_ms)
            ):
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=analyzer_id,
                        kind=GoalEvidenceGapKind.COVERAGE_STALE,
                        detail="analyzer receipt is stale or lacks a freshness window",
                    )
                )
                continue
            healthy = raw.get("healthy") is True
            exhaustive = raw.get("exhaustive") is not False
            conclusive = raw.get("conclusive") is not False
            if not healthy:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=analyzer_id,
                        kind=GoalEvidenceGapKind.ANALYZER_UNHEALTHY,
                        detail="analyzer is not healthy",
                    )
                )
                continue
            if not exhaustive or not conclusive:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=analyzer_id,
                        kind=GoalEvidenceGapKind.ANALYZER_INCOMPLETE,
                        detail="analyzer is not exhaustive and conclusive",
                    )
                )
                continue
            try:
                retained_b64 = _text(raw.get("retained_validation_bytes_b64"))
                retained_cid = _text(raw.get("retained_validation_cid"))
                if not retained_b64:
                    retained_b64, retained_cid = _encode_retained_bytes(dict(raw))
                else:
                    _, derived = _encode_retained_bytes(
                        _decode_retained_bytes(retained_b64)
                    )
                    retained_cid = retained_cid or derived
                receipt = ProofReuseAnalyzerReceipt(
                    analyzer_id=analyzer_id,
                    producer_channel=producer_channel,
                    channel_proof_revision=channel_revision,
                    repository_id=self.repository_id,
                    git_tree_id=self.git_tree_id,
                    repository_forest_cid=self.repository_forest_cid,
                    objective_revision=self.objective_revision,
                    observed_at_ms=observed,
                    fresh_until_ms=fresh_until,
                    retained_validation_bytes_b64=retained_b64,
                    retained_validation_cid=retained_cid,
                    healthy=True,
                    exhaustive=True,
                    conclusive=True,
                    authority=_AUTHORITATIVE,
                )
            except ProofTestReuseGoalEvidenceError as exc:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=analyzer_id,
                        kind=GoalEvidenceGapKind.MALFORMED,
                        detail=str(exc)[:_MAX_DETAIL],
                    )
                )
                continue
            by_id[analyzer_id] = receipt

        for analyzer_id in sorted(REQUIRED_ANALYZER_CHANNELS - set(by_id)):
            gaps.append(
                GoalEvidenceGap(
                    subject_id=analyzer_id,
                    kind=GoalEvidenceGapKind.ANALYZER_MISSING,
                    detail="required analyzer channel has no receipt",
                )
            )
        return list(by_id.values()), gaps

    def _population_receipts(
        self, inputs: Iterable[Any], *, now_ms: int
    ) -> tuple[list[ProofReusePopulationReceipt], list[GoalEvidenceGap]]:
        gaps: list[GoalEvidenceGap] = []
        by_id: dict[str, ProofReusePopulationReceipt] = {}
        for raw_value in inputs:
            raw = _record(raw_value)
            population_id = _text(
                _value(raw, "population_id", "population", "id")
            )
            if not population_id:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id="population",
                        kind=GoalEvidenceGapKind.MALFORMED,
                        detail="population input is missing population_id",
                    )
                )
                continue
            if population_id in by_id:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=population_id,
                        kind=GoalEvidenceGapKind.UNEXPECTED_INPUT,
                        detail="duplicate population receipt",
                    )
                )
                continue
            if population_id not in REQUIRED_ADVERSARIAL_POPULATIONS:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=population_id,
                        kind=GoalEvidenceGapKind.UNEXPECTED_INPUT,
                        detail="population is outside the required adversarial set",
                    )
                )
                continue
            producer_channel = _text(
                raw.get("producer_channel") or f"adversarial:{population_id}"
            )
            channel_revision = _text(
                raw.get("channel_proof_revision") or self.channel_proof_revision
            )
            if not self._bindings_match(raw):
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=population_id,
                        kind=GoalEvidenceGapKind.COVERAGE_BINDING_MISMATCH,
                        detail="population receipt is not bound to current identities",
                    )
                )
                continue
            observed = _integer(raw, "observed_at_ms")
            fresh_until = _integer(raw, "fresh_until_ms")
            if (
                observed is None
                or fresh_until is None
                or not self._is_fresh(observed, fresh_until, now_ms)
            ):
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=population_id,
                        kind=GoalEvidenceGapKind.COVERAGE_STALE,
                        detail="population receipt is stale or lacks a freshness window",
                    )
                )
                continue
            passed = raw.get("passed") is True
            false_skips = raw.get("false_skips")
            if isinstance(false_skips, bool) or not isinstance(false_skips, int):
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=population_id,
                        kind=GoalEvidenceGapKind.MALFORMED,
                        detail="false_skips must be an integer",
                    )
                )
                continue
            if not passed:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=population_id,
                        kind=GoalEvidenceGapKind.POPULATION_FAILED,
                        detail="adversarial population did not pass",
                    )
                )
                continue
            if false_skips != 0:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=population_id,
                        kind=GoalEvidenceGapKind.FALSE_SKIP_DETECTED,
                        detail="adversarial population reported nonzero false skips",
                    )
                )
                continue
            try:
                retained_b64 = _text(raw.get("retained_validation_bytes_b64"))
                retained_cid = _text(raw.get("retained_validation_cid"))
                if not retained_b64:
                    retained_b64, retained_cid = _encode_retained_bytes(dict(raw))
                else:
                    _, derived = _encode_retained_bytes(
                        _decode_retained_bytes(retained_b64)
                    )
                    retained_cid = retained_cid or derived
                receipt = ProofReusePopulationReceipt(
                    population_id=population_id,
                    producer_channel=producer_channel,
                    channel_proof_revision=channel_revision,
                    repository_id=self.repository_id,
                    git_tree_id=self.git_tree_id,
                    repository_forest_cid=self.repository_forest_cid,
                    objective_revision=self.objective_revision,
                    observed_at_ms=observed,
                    fresh_until_ms=fresh_until,
                    retained_validation_bytes_b64=retained_b64,
                    retained_validation_cid=retained_cid,
                    passed=True,
                    false_skips=0,
                    authority=_AUTHORITATIVE,
                )
            except ProofTestReuseGoalEvidenceError as exc:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=population_id,
                        kind=GoalEvidenceGapKind.MALFORMED,
                        detail=str(exc)[:_MAX_DETAIL],
                    )
                )
                continue
            by_id[population_id] = receipt

        for population_id in sorted(REQUIRED_ADVERSARIAL_POPULATIONS - set(by_id)):
            gaps.append(
                GoalEvidenceGap(
                    subject_id=population_id,
                    kind=GoalEvidenceGapKind.POPULATION_MISSING,
                    detail="required adversarial population has no receipt",
                )
            )
        return list(by_id.values()), gaps

    def _quorum_members(
        self, inputs: Iterable[Any], *, now_ms: int
    ) -> tuple[list[GoalQuorumMember], list[GoalEvidenceGap]]:
        gaps: list[GoalEvidenceGap] = []
        members: list[GoalQuorumMember] = []
        seen_ids: set[str] = set()
        seen_channels: set[str] = set()
        seen_receipts: set[str] = set()

        for raw_value in inputs:
            raw = _record(raw_value)
            try:
                member = GoalQuorumMember.from_dict(raw)
            except (ProofTestReuseGoalEvidenceError, TypeError, ValueError) as exc:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=_text(raw.get("member_id")) or "quorum",
                        kind=GoalEvidenceGapKind.MALFORMED,
                        detail=str(exc)[:_MAX_DETAIL],
                    )
                )
                continue
            # Re-evaluate freshness against the runner clock.
            fresh = self._is_fresh(
                member.observed_at_ms, member.fresh_until_ms, now_ms
            )
            if not fresh:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=member.member_id,
                        kind=GoalEvidenceGapKind.QUORUM_STALE,
                        detail="quorum member is outside the freshness window",
                    )
                )
                continue
            if not member.healthy:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=member.member_id,
                        kind=GoalEvidenceGapKind.QUORUM_UNHEALTHY,
                        detail="quorum member is not healthy",
                    )
                )
                continue
            if not member.exhaustive:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=member.member_id,
                        kind=GoalEvidenceGapKind.QUORUM_NOT_EXHAUSTIVE,
                        detail="quorum member is not exhaustive",
                    )
                )
                continue
            if not member.conclusive:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=member.member_id,
                        kind=GoalEvidenceGapKind.QUORUM_INCONCLUSIVE,
                        detail="quorum member is not conclusive",
                    )
                )
                continue
            if not member.uncontradicted:
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=member.member_id,
                        kind=GoalEvidenceGapKind.QUORUM_CONTRADICTED,
                        detail="quorum member is contradicted",
                    )
                )
                continue
            if (
                member.member_id in seen_ids
                or member.evidence_channel in seen_channels
                or member.receipt_cid in seen_receipts
            ):
                gaps.append(
                    GoalEvidenceGap(
                        subject_id=member.member_id,
                        kind=GoalEvidenceGapKind.QUORUM_NOT_INDEPENDENT,
                        detail=(
                            "quorum members must have distinct member_id, "
                            "evidence_channel, and receipt_cid values"
                        ),
                    )
                )
                continue
            seen_ids.add(member.member_id)
            seen_channels.add(member.evidence_channel)
            seen_receipts.add(member.receipt_cid)
            members.append(
                GoalQuorumMember(
                    member_id=member.member_id,
                    evidence_channel=member.evidence_channel,
                    receipt_cid=member.receipt_cid,
                    healthy=True,
                    exhaustive=True,
                    conclusive=True,
                    fresh=True,
                    uncontradicted=True,
                    observed_at_ms=member.observed_at_ms,
                    fresh_until_ms=member.fresh_until_ms,
                )
            )

        if len(members) < REQUIRED_QUORUM_MEMBERS:
            gaps.append(
                GoalEvidenceGap(
                    subject_id="quorum",
                    kind=GoalEvidenceGapKind.QUORUM_INSUFFICIENT,
                    detail=(
                        f"need {REQUIRED_QUORUM_MEMBERS} independent healthy "
                        f"exhaustive conclusive fresh uncontradicted quorum "
                        f"members; have {len(members)}"
                    ),
                )
            )
        return members, gaps

    @staticmethod
    def _filter_informational_capability_gaps(
        gaps: Sequence[GoalEvidenceGap],
        required: Sequence[str],
    ) -> list[GoalEvidenceGap]:
        """Drop pure capability-unavailable notices from the blocking gap set.

        Optional capability loss is typed on ``unavailable_capabilities`` and
        is never itself a completion failure.  Real-ZK and production-warm
        consequences are represented by their own gap kinds when a reviewed
        certificate is also absent.
        """

        del required  # population is consulted by the real-ZK / warm gates
        return [
            gap
            for gap in gaps
            if gap.kind is not GoalEvidenceGapKind.CAPABILITY_UNAVAILABLE
        ]


__all__ = [
    "ACCEPTANCE_COVERAGE_INTERFACE",
    "ACCEPTANCE_COVERAGE_RECEIPT_INTERFACE",
    "ANALYZER_HEALTH_INTERFACE",
    "CoverageStatus",
    "DEFAULT_CHANNEL_PROOF_REVISION",
    "DEFAULT_PRODUCER_CHANNEL",
    "EXHAUSTION_QUORUM_INTERFACE",
    "GoalAssuranceResult",
    "GoalAssuranceRunner",
    "GoalEvidenceGap",
    "GoalEvidenceGapKind",
    "GoalQuorumMember",
    "OPTIONAL_CAPABILITY_NAMES",
    "PRODUCTION_WARM_REQUIREMENT_IDS",
    "PRODUCING_TASK_ID",
    "PROOF_REUSE_ANALYZER_RECEIPT_INTERFACE",
    "PROOF_REUSE_BENCHMARK_RECEIPT_INTERFACE",
    "PROOF_REUSE_POPULATION_RECEIPT_INTERFACE",
    "PROOF_REUSE_ROLLBACK_DECISION_INTERFACE",
    "PROOF_TEST_REUSE_GOAL_EVIDENCE_INTERFACE",
    "ProofReuseAnalyzerReceipt",
    "ProofReusePopulationReceipt",
    "ProofTestReuseGoalEvidence",
    "ProofTestReuseGoalEvidenceError",
    "REAL_ZK_REQUIREMENT_IDS",
    "REQUIRED_ADVERSARIAL_POPULATIONS",
    "REQUIRED_ANALYZER_CHANNELS",
    "REQUIRED_QUORUM_MEMBERS",
    "TEST_CERTIFICATE_ASSURANCE_RECEIPT_INTERFACE",
    "AcceptanceCoverageReceipt",
    "discover_requirement_ids_from_heap",
    "goal_requirement_ids",
    "goal_requirements_by_id",
    "load_objective_goals",
]
