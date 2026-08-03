"""Fail-closed final authority gate for proof-backed test reuse.

The final gate is intentionally an aggregator, not another proof provider.  It
accepts only fresh, authoritative evidence already bound to the exact current
repository forest, Git tree, and objective-completion identity.  A pytest
``SKIP`` line, a rollout decision, or a historical benchmark is never authority
by itself.

PTR-122 repairs the final-gate self-reference: the gate no longer requires
``PTR-G110`` as its own child premise.  It verifies ``PTR-G010`` through
``PTR-G100``, validates G110 benchmark and rollout premises directly, checks a
fresh three-lane supervisor-health receipt, then emits separate exact evidence
for:

* ``ptr/final-current-tree-gate@1`` on ``PTR-G110``
* ``ptr/cross-repository-current-tree-gate@1`` on ``PTR-G000``

without claiming the other root requirements by implication.  The producing
task remains ``PTR-122``.

PTR-142 refreshes the sealed production population from 41 to the exact 53-task
board (adding the runtime-activation repair ``PTR-131`` … ``PTR-142``) and
required runtime-activation repair evidence.  A later current-tree audit found
that its injected/pseudo-certificate activation evidence was not production
authority.  PTR-149 therefore expands the population to 60 and requires fresh
evidence for the corrective ``PTR-143`` … ``PTR-149`` wave; historical PTR-142
evidence cannot satisfy that premise.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
from typing import Any, ClassVar, Final

from ...testing.proof_reuse.rollout import (
    ProofReusePromotionEvidence,
    ProofReuseRolloutDecision,
    ProofReuseRolloutPolicy,
    ProofReuseRolloutStage,
)
from ..objectives.goal_completion import (
    CHANNEL_PROOF_REVISION_NAMESPACE,
    CompletionEvidence,
)
from ..self_improvement.proof_reuse_benchmark import (
    ProofReuseBenchmarkReceipt,
    verify_benchmark_receipt,
)
from .proof_cached_test_validation import ProofCachedTestValidationReceipt
from .proof_test_reuse_objective_contracts import (
    cid_for_canonical_dag_json_bytes,
    cid_for_mapping,
    verify_retained_bytes,
)

PROOF_TEST_REUSE_CURRENT_TREE_GATE_INTERFACE: Final = (
    "ProofTestReuseCurrentTreeGate@1"
)
PROOF_TEST_REUSE_COMPLETION_EVIDENCE_INTERFACE: Final = (
    "ProofTestReuseCompletionEvidence@1"
)
PROOF_TEST_REUSE_COMPLETION_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "proof-test-reuse-completion-evidence@1"
)
PROOF_TEST_REUSE_GATE_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "proof-test-reuse-current-tree-gate-decision@1"
)
PROOF_TEST_REUSE_GATE_BUNDLE_INTERFACE: Final = "ProofTestReuseGateBundle@1"
PROOF_TEST_REUSE_PERSISTED_GATE_BUNDLE_INTERFACE: Final = (
    "ProofTestReusePersistedGateBundle@1"
)
PROOF_TEST_REUSE_PERSISTED_GATE_BUNDLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "proof-test-reuse-persisted-gate-bundle@1"
)
PROOF_TEST_REUSE_CURRENT_TREE_GATE_VERSION: Final = 1

ROOT_GOAL_ID: Final = "PTR-G000"
FINAL_GATE_GOAL_ID: Final = "PTR-G110"
FINAL_GATE_TASK_ID: Final = "PTR-122"
LEGACY_FINAL_GATE_TASK_ID: Final = "PTR-102"

FINAL_GATE_ACCEPTANCE_CRITERION: Final = "ptr/final-current-tree-gate@1"
ROOT_ACCEPTANCE_CRITERION: Final = "ptr/cross-repository-current-tree-gate@1"

# Exact satisfied requirements per emitted evidence — never the full root set.
FINAL_GATE_SATISFIED_REQUIREMENTS: Final = (FINAL_GATE_ACCEPTANCE_CRITERION,)
ROOT_SATISFIED_REQUIREMENTS: Final = (ROOT_ACCEPTANCE_CRITERION,)

# Historical name retained for importers that inspect root requirement IDs
# declared on G000.  The gate verifies the supporting premises but the root
# completion evidence claims only ``ROOT_ACCEPTANCE_CRITERION``.
ROOT_EVIDENCE_REQUIREMENTS: Final = (
    "ptr/cross-repository-current-tree-gate@1",
    "ptr/zero-false-authoritative-skip@1",
    "ptr/warm-reuse-benchmark@1",
    "ptr/supervisor-launch-health@1",
)

DEFAULT_PRODUCER_CHANNEL: Final = "current-tree-authority-gate"
DEFAULT_ANALYZER_REVISION: Final = "analyzer:current-tree-gate@1"
DEFAULT_CONFIGURATION_REVISION: Final = "configuration:current-tree-gate@1"

# Runtime-activation repair wave (PTR-131 … PTR-142).  These tasks expand the
# sealed board from 41 to 53 and must remain present in the production set.
RUNTIME_ACTIVATION_REPAIR_TASK_IDS: Final = frozenset(
    {
        "PTR-131",
        "PTR-132",
        "PTR-133",
        "PTR-134",
        "PTR-135",
        "PTR-136",
        "PTR-137",
        "PTR-138",
        "PTR-139",
        "PTR-140",
        "PTR-141",
        "PTR-142",
    }
)
RUNTIME_ACTIVATION_REPAIR_ID: Final = "runtime-activation"
RUNTIME_ACTIVATION_REPAIR_EVIDENCE_REQUIREMENT: Final = (
    "ptr/runtime-activation-repair-evidence@1"
)

# Production correction for activation claims that PTR-142 did not actually
# establish through the ordinary default two-process path.  Keep the historical
# constants above for replay/import compatibility; they are not the fresh
# repair population accepted by the current production gate.
PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS: Final = frozenset(
    {
        "PTR-143",
        "PTR-144",
        "PTR-145",
        "PTR-146",
        "PTR-147",
        "PTR-148",
        "PTR-149",
    }
)
PRODUCTION_RUNTIME_ACTIVATION_ID: Final = "production-runtime-activation"
PRODUCTION_RUNTIME_ACTIVATION_EVIDENCE_REQUIREMENT: Final = (
    "ptr/production-runtime-activation-evidence@1"
)
PRODUCTION_RUNTIME_ACTIVATION_PRODUCER_TASK_ID: Final = "PTR-149"

# Sealed production population: all 60 implementation tasks on the board.
REQUIRED_PTR_TASK_IDS: Final = frozenset(
    {
        "PTR-000",
        "PTR-001",
        "PTR-002",
        "PTR-003",
        "PTR-010",
        "PTR-011",
        "PTR-012",
        "PTR-020",
        "PTR-021",
        "PTR-022",
        "PTR-030",
        "PTR-031",
        "PTR-040",
        "PTR-041",
        "PTR-042",
        "PTR-043",
        "PTR-050",
        "PTR-051",
        "PTR-052",
        "PTR-053",
        "PTR-060",
        "PTR-061",
        "PTR-070",
        "PTR-080",
        "PTR-081",
        "PTR-090",
        "PTR-091",
        "PTR-092",
        "PTR-093",
        "PTR-100",
        "PTR-101",
        "PTR-102",
        "PTR-108",
        "PTR-109",
        "PTR-110",
        "PTR-111",
        "PTR-112",
        "PTR-120",
        "PTR-121",
        "PTR-122",
        "PTR-130",
        *RUNTIME_ACTIVATION_REPAIR_TASK_IDS,
        *PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS,
    }
)
SEALED_PRODUCTION_TASK_COUNT: Final = 60
assert len(REQUIRED_PTR_TASK_IDS) == SEALED_PRODUCTION_TASK_COUNT
# Verified child premises only — G110 is not a child premise of itself.
REQUIRED_CHILD_GOAL_IDS: Final = frozenset(
    {
        "PTR-G010",
        "PTR-G020",
        "PTR-G030",
        "PTR-G040",
        "PTR-G050",
        "PTR-G060",
        "PTR-G070",
        "PTR-G080",
        "PTR-G090",
        "PTR-G100",
    }
)
REQUIRED_GRAPH_GOAL_IDS: Final = REQUIRED_CHILD_GOAL_IDS | {
    ROOT_GOAL_ID,
    FINAL_GATE_GOAL_ID,
}
REQUIRED_ADVERSARIAL_POPULATIONS: Final = frozenset(
    {"mutation", "storage-security-concurrency", "cross-repository"}
)
REQUIRED_ANALYZERS: Final = frozenset(
    {"static-dependency", "runtime-dependency", "reuse-eligibility"}
)
REQUIRED_SUPERVISOR_LANE_IDS: Final = frozenset(
    {"ptr_lane_0", "ptr_lane_1", "ptr_lane_2"}
)
REQUIRED_SUPERVISOR_LANE_COUNT: Final = 3

_CLOSED_TASK_STATES = frozenset({"complete", "completed", "verified_complete"})
_CLOSED_GOAL_STATES = frozenset({"verified_complete"})
_QUARANTINE_STATES = frozenset({"quarantine", "quarantined"})
_AUTHORITATIVE = "authoritative"
_MAX_REASONS = 256
_ALLOWED_PRODUCER_KINDS = frozenset({"task", "scan"})


class TaskCompletionProvenanceKind(Enum):
    """Closed discriminator for the reviewed ways a task can be complete."""

    MANAGED_MERGE = "managed_merge"
    OPERATOR_PLANNING_SEAL = "operator_planning_seal"
    OPERATOR_REVIEWED_INTEGRATION = "operator_reviewed_integration"
    RETROSPECTIVE_INTEGRATION_VERIFICATION = (
        "retrospective_integration_verification"
    )


# Each member is a closed record.  Keeping the union closed is important:
# adding a new completion path requires an explicit gate review instead of
# silently treating a new queue disposition (especially quarantine) as proof
# of success.
_TASK_PROVENANCE_FIELDS: Final[
    Mapping[TaskCompletionProvenanceKind, tuple[frozenset[str], frozenset[str]]]
] = {
    TaskCompletionProvenanceKind.MANAGED_MERGE: (
        frozenset({"merge_receipt_cid", "merged_commit_id"}),
        frozenset({"merge_succeeded"}),
    ),
    TaskCompletionProvenanceKind.OPERATOR_PLANNING_SEAL: (
        frozenset(
            {
                "planning_seal_cid",
                "operator_approval_cid",
                "sealed_objective_revision",
            }
        ),
        frozenset({"planning_seal_accepted"}),
    ),
    TaskCompletionProvenanceKind.OPERATOR_REVIEWED_INTEGRATION: (
        frozenset(
            {
                "integration_receipt_cid",
                "integrated_commit_id",
                "integration_target_commit_id",
                "operator_review_cid",
            }
        ),
        frozenset({"integration_verified"}),
    ),
    TaskCompletionProvenanceKind.RETROSPECTIVE_INTEGRATION_VERIFICATION: (
        frozenset(
            {
                "integrated_commit_id",
                "ancestry_target_commit_id",
                "ancestry_receipt_cid",
                "current_tree_rerun_receipt_cid",
                "current_tree_rerun_repository_id",
                "current_tree_rerun_tree_id",
                "current_tree_rerun_commit_id",
                "current_tree_rerun_gitlink_state_cid",
                "current_tree_rerun_repository_forest_cid",
                "current_tree_rerun_policy_cid",
                "current_tree_rerun_capability_cid",
                "current_tree_rerun_verifying_key_cid",
                "current_tree_rerun_circuit_cid",
                "policy_approval_cid",
                "approved_policy_cid",
            }
        ),
        frozenset(
            {
                "ancestry_verified",
                "current_tree_rerun_passed",
                "policy_approved",
            }
        ),
    ),
}


class ProofTestReuseCurrentTreeGateError(ValueError):
    """Raised for invalid gate configuration, never for rejected evidence."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _content_id(value: Any) -> str:
    """Internal fingerprint for non-authoritative helper identity."""

    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _artifact_cid(value: Mapping[str, Any]) -> str:
    """Strict CIDv1/base32/dag-json/sha2-256 over retained canonical bytes."""

    return cid_for_mapping(dict(value))


def _namespaced_sha256_revision(value: Any, namespace: str) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = sha256()
    digest.update(str(namespace).encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical)
    return f"sha256:{digest.hexdigest()}"


def _text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _record(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        return result if isinstance(result, Mapping) else {}
    return {}


def _value(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _timestamp_ms(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        # Accept seconds for convenience, while persisted gate records use ms.
        return int(number * 1000) if abs(number) < 10_000_000_000 else int(number)
    isoformat = getattr(value, "timestamp", None)
    if callable(isoformat):
        try:
            return int(value.timestamp() * 1000)
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(value, str):
        try:
            from datetime import datetime as _dt

            normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
            return int(_dt.fromisoformat(normalized).timestamp() * 1000)
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def _reason(prefix: str, identifier: str = "") -> str:
    clean = _text(identifier).replace(":", "_")
    return f"{prefix}:{clean}" if clean else prefix


def _assert_identity_domains_distinct(
    git_tree_id: str,
    repository_forest_cid: str,
    objective_completion_tree_id: str,
) -> None:
    """Git tree, forest, and objective-completion identities are distinct."""

    domains = {
        "git_tree_id": _text(git_tree_id),
        "repository_forest_cid": _text(repository_forest_cid),
        "objective_completion_tree_id": _text(objective_completion_tree_id),
    }
    if any(not value for value in domains.values()):
        raise ProofTestReuseCurrentTreeGateError(
            "git_tree_id, repository_forest_cid, and "
            "objective_completion_tree_id are all required and distinct"
        )
    values = list(domains.values())
    if len(set(values)) != 3:
        raise ProofTestReuseCurrentTreeGateError(
            "git_tree_id, repository_forest_cid, and "
            "objective_completion_tree_id must be pairwise distinct domains"
        )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Enum):
        return value.value
    return str(value)


def _serialize_evaluate_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical JSON form of evaluate() kwargs for bundle persistence."""

    result: dict[str, Any] = {}
    for key, value in packet.items():
        if key in {
            "task_evidence",
            "child_goal_evidence",
            "adversarial_evidence",
            "analyzer_health",
        }:
            if isinstance(value, Mapping):
                result[key] = {
                    str(item_key): _jsonable(item_value)
                    for item_key, item_value in value.items()
                }
            else:
                result[key] = [_jsonable(item) for item in value]
        else:
            result[key] = _jsonable(value)
    return result


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        number = float(value)
        seconds = number / 1000.0 if abs(number) >= 10_000_000_000 else number
        return datetime.fromtimestamp(seconds, tz=UTC)
    return datetime.fromtimestamp(0, tz=UTC)


def _deserialize_evaluate_packet(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild evaluate() kwargs, restoring typed benchmark/rollout objects."""

    from ...testing.proof_reuse.rollout import RolloutDisposition

    result = dict(packet)
    benchmark = dict(_mapping(result.get("benchmark_evidence")))
    receipt = benchmark.get("receipt")
    if isinstance(receipt, Mapping):
        try:
            benchmark["receipt"] = ProofReuseBenchmarkReceipt.from_dict(receipt)
        except Exception:
            benchmark["receipt"] = ProofReuseBenchmarkReceipt(
                corpus_id=_text(receipt.get("corpus_id")),
                false_admissions=int(receipt.get("false_admissions") or 0),
                warm_eligible_count=int(receipt.get("warm_eligible_count") or 0),
                warm_verified_skips=int(receipt.get("warm_verified_skips") or 0),
                warm_skip_bps=int(receipt.get("warm_skip_bps") or 0),
                passed=bool(receipt.get("passed")),
            )
        result["benchmark_evidence"] = benchmark

    rollout = dict(_mapping(result.get("rollout_evidence")))
    decision_payload = rollout.get("decision")
    promotion_payload = rollout.get("promotion_evidence")
    if isinstance(promotion_payload, Mapping):
        gate_passed = promotion_payload.get("current_tree_gate_passed")
        if gate_passed is not None and not isinstance(gate_passed, bool):
            gate_passed = None
        promo_kwargs: dict[str, Any] = {
            "observed_at": _parse_datetime(promotion_payload.get("observed_at")),
            "repository_id": _text(promotion_payload.get("repository_id")),
            "tree_id": _text(promotion_payload.get("tree_id")),
            "policy_id": _text(promotion_payload.get("policy_id")),
            "policy_revision": _text(promotion_payload.get("policy_revision")),
            "current_stage": ProofReuseRolloutStage(
                _text(promotion_payload.get("current_stage"))
            ),
            "target_stage": ProofReuseRolloutStage(
                _text(promotion_payload.get("target_stage"))
            ),
            "mutation_false_skips": int(
                promotion_payload.get("mutation_false_skips") or 0
            ),
            "degradation_false_skips": int(
                promotion_payload.get("degradation_false_skips") or 0
            ),
            "authority_contradictions": int(
                promotion_payload.get("authority_contradictions") or 0
            ),
            "corruption_spike": bool(promotion_payload.get("corruption_spike")),
            "stale_keys": int(promotion_payload.get("stale_keys") or 0),
            "key_health_ok": bool(promotion_payload.get("key_health_ok")),
            "revocation_health_ok": bool(
                promotion_payload.get("revocation_health_ok")
            ),
            "controlled_issuer": bool(
                promotion_payload.get("controlled_issuer", True)
            ),
            "current_tree_gate_passed": gate_passed,
            "all_repositories_passed": bool(
                promotion_payload.get("all_repositories_passed", True)
            ),
        }
        if "forced_reruns" in promotion_payload and promotion_payload[
            "forced_reruns"
        ] is not None:
            promo_kwargs["forced_reruns"] = int(
                promotion_payload.get("forced_reruns") or 0
            )
        if _text(promotion_payload.get("operator_approval_id")):
            promo_kwargs["operator_approval_id"] = _text(
                promotion_payload.get("operator_approval_id")
            )
        reconstructed_promotion = ProofReusePromotionEvidence(**promo_kwargs)
        rollout["promotion_evidence"] = reconstructed_promotion
    else:
        reconstructed_promotion = None

    if isinstance(decision_payload, Mapping):
        evidence_id = _text(decision_payload.get("evidence_id"))
        if not evidence_id and reconstructed_promotion is not None:
            evidence_id = reconstructed_promotion.evidence_id
        gates_raw = decision_payload.get("gates") or ()
        rollout["decision"] = ProofReuseRolloutDecision(
            current_stage=ProofReuseRolloutStage(
                _text(decision_payload.get("current_stage"))
            ),
            requested_stage=ProofReuseRolloutStage(
                _text(decision_payload.get("requested_stage"))
            ),
            effective_stage=ProofReuseRolloutStage(
                _text(decision_payload.get("effective_stage"))
            ),
            disposition=RolloutDisposition(
                _text(decision_payload.get("disposition"))
            ),
            gates=tuple(gates_raw),
            evidence_id=evidence_id,
            policy_id=_text(decision_payload.get("policy_id")),
            policy_revision=_text(decision_payload.get("policy_revision")),
        )
        result["rollout_evidence"] = rollout
    return result


@dataclass(frozen=True, slots=True)
class ProofTestReuseCompletionEvidence:
    """One exact goal completion artifact emitted by the current-tree gate."""

    __test__: ClassVar[bool] = False

    repository_id: str
    tree_id: str
    commit_id: str
    gitlink_state_cid: str
    repository_forest_cid: str
    objective_completion_tree_id: str
    policy_cid: str
    capability_cid: str
    verifying_key_cid: str
    circuit_cid: str
    objective_revision: str
    task_evidence_cids: tuple[str, ...]
    child_goal_evidence_cids: tuple[str, ...]
    adversarial_evidence_cids: tuple[str, ...]
    analyzer_evidence_cids: tuple[str, ...]
    benchmark_receipt_cid: str
    rollout_decision_cid: str
    supervisor_health_cid: str
    observed_at_ms: int
    fresh_until_ms: int
    goal_id: str
    acceptance_criterion: str
    satisfied_requirements: tuple[str, ...]
    producing_task_id: str = FINAL_GATE_TASK_ID
    authority: str = _AUTHORITATIVE
    producer_kind: str = "task"
    producer_channel: str = DEFAULT_PRODUCER_CHANNEL
    analyzer_revision: str = DEFAULT_ANALYZER_REVISION
    configuration_revision: str = DEFAULT_CONFIGURATION_REVISION
    premise_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        goal_id = _text(self.goal_id)
        if goal_id not in {ROOT_GOAL_ID, FINAL_GATE_GOAL_ID}:
            raise ProofTestReuseCurrentTreeGateError(
                "completion evidence may target only PTR-G000 or PTR-G110"
            )
        object.__setattr__(self, "goal_id", goal_id)
        if self.producing_task_id != FINAL_GATE_TASK_ID:
            raise ProofTestReuseCurrentTreeGateError(
                "final completion evidence must be produced by PTR-122"
            )
        if self.authority != _AUTHORITATIVE:
            raise ProofTestReuseCurrentTreeGateError(
                "completion evidence authority must be authoritative"
            )
        if self.producer_kind not in _ALLOWED_PRODUCER_KINDS:
            raise ProofTestReuseCurrentTreeGateError(
                "producer_kind must be an allowed generic producer ('task' or 'scan')"
            )
        if not _text(self.producer_channel):
            raise ProofTestReuseCurrentTreeGateError(
                "producer_channel is required"
            )
        criterion = _text(self.acceptance_criterion)
        requirements = tuple(
            _text(item) for item in self.satisfied_requirements if _text(item)
        )
        if goal_id == FINAL_GATE_GOAL_ID:
            if criterion != FINAL_GATE_ACCEPTANCE_CRITERION:
                raise ProofTestReuseCurrentTreeGateError(
                    "G110 evidence must use ptr/final-current-tree-gate@1"
                )
            if requirements != FINAL_GATE_SATISFIED_REQUIREMENTS:
                raise ProofTestReuseCurrentTreeGateError(
                    "G110 evidence must claim only ptr/final-current-tree-gate@1"
                )
        else:
            if criterion != ROOT_ACCEPTANCE_CRITERION:
                raise ProofTestReuseCurrentTreeGateError(
                    "G000 evidence must use ptr/cross-repository-current-tree-gate@1"
                )
            if requirements != ROOT_SATISFIED_REQUIREMENTS:
                raise ProofTestReuseCurrentTreeGateError(
                    "G000 evidence must claim only "
                    "ptr/cross-repository-current-tree-gate@1"
                )
        object.__setattr__(self, "acceptance_criterion", criterion)
        object.__setattr__(self, "satisfied_requirements", requirements)
        _assert_identity_domains_distinct(
            self.tree_id,
            self.repository_forest_cid,
            self.objective_completion_tree_id,
        )
        for name in (
            "repository_id",
            "tree_id",
            "commit_id",
            "gitlink_state_cid",
            "repository_forest_cid",
            "objective_completion_tree_id",
            "policy_cid",
            "capability_cid",
            "verifying_key_cid",
            "circuit_cid",
            "objective_revision",
            "benchmark_receipt_cid",
            "rollout_decision_cid",
            "supervisor_health_cid",
            "analyzer_revision",
            "configuration_revision",
        ):
            if not _text(getattr(self, name)):
                raise ProofTestReuseCurrentTreeGateError(f"{name} is required")
        for name in (
            "task_evidence_cids",
            "child_goal_evidence_cids",
            "adversarial_evidence_cids",
            "analyzer_evidence_cids",
        ):
            values = tuple(_text(item) for item in getattr(self, name))
            if not values or any(not item for item in values):
                raise ProofTestReuseCurrentTreeGateError(f"{name} is incomplete")
            object.__setattr__(self, name, tuple(sorted(values)))
        premises = tuple(
            sorted({_text(item) for item in self.premise_cids if _text(item)})
        )
        object.__setattr__(self, "premise_cids", premises)
        if self.observed_at_ms < 0 or self.fresh_until_ms <= self.observed_at_ms:
            raise ProofTestReuseCurrentTreeGateError(
                "completion evidence freshness window is invalid"
            )

    def _identity_body(self) -> dict[str, Any]:
        return {
            "schema": PROOF_TEST_REUSE_COMPLETION_EVIDENCE_SCHEMA,
            "interface": PROOF_TEST_REUSE_COMPLETION_EVIDENCE_INTERFACE,
            "version": PROOF_TEST_REUSE_CURRENT_TREE_GATE_VERSION,
            "goal_id": self.goal_id,
            "producing_task_id": self.producing_task_id,
            "acceptance_criterion": self.acceptance_criterion,
            "authority": self.authority,
            "producer_kind": self.producer_kind,
            "producer_channel": self.producer_channel,
            "satisfied_requirements": list(self.satisfied_requirements),
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "git_tree_id": self.tree_id,
            "commit_id": self.commit_id,
            "gitlink_state_cid": self.gitlink_state_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "objective_completion_tree_id": self.objective_completion_tree_id,
            "policy_cid": self.policy_cid,
            "capability_cid": self.capability_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "circuit_cid": self.circuit_cid,
            "objective_revision": self.objective_revision,
            "analyzer_revision": self.analyzer_revision,
            "configuration_revision": self.configuration_revision,
            "task_evidence_cids": list(self.task_evidence_cids),
            "child_goal_evidence_cids": list(self.child_goal_evidence_cids),
            "adversarial_evidence_cids": list(self.adversarial_evidence_cids),
            "analyzer_evidence_cids": list(self.analyzer_evidence_cids),
            "benchmark_receipt_cid": self.benchmark_receipt_cid,
            "rollout_decision_cid": self.rollout_decision_cid,
            "supervisor_health_cid": self.supervisor_health_cid,
            "premise_cids": list(self.premise_cids),
            "observed_at_ms": self.observed_at_ms,
            "fresh_until_ms": self.fresh_until_ms,
        }

    @property
    def evidence_id(self) -> str:
        return _artifact_cid(self._identity_body())

    @property
    def provenance_cid(self) -> str:
        return self.evidence_id

    def to_dict(self) -> dict[str, Any]:
        body = self._identity_body()
        body["evidence_id"] = self.evidence_id
        body["provenance_cid"] = self.evidence_id
        body["channel_proof_revision"] = self.channel_proof_revision
        return body

    def build_channel_proof(self) -> dict[str, Any]:
        """Canonical channel proof for the generic completion adapter."""

        return {
            "channel": self.producer_channel,
            "producing_task_id": self.producing_task_id,
            "goal_id": self.goal_id,
            "acceptance_criterion": self.acceptance_criterion,
            "repository_id": self.repository_id,
            "git_tree_id": self.tree_id,
            "repository_forest_cid": self.repository_forest_cid,
            "objective_completion_tree_id": self.objective_completion_tree_id,
            "objective_revision": self.objective_revision,
            "policy_cid": self.policy_cid,
            "evidence_interface": PROOF_TEST_REUSE_COMPLETION_EVIDENCE_INTERFACE,
            "gate_interface": PROOF_TEST_REUSE_CURRENT_TREE_GATE_INTERFACE,
            "version": PROOF_TEST_REUSE_CURRENT_TREE_GATE_VERSION,
        }

    @property
    def channel_proof_revision(self) -> str:
        return _namespaced_sha256_revision(
            self.build_channel_proof(),
            CHANNEL_PROOF_REVISION_NAMESPACE,
        )

    def as_completion_evidence(self) -> CompletionEvidence:
        """Project into the shared objective ``CompletionEvidence`` surface.

        Uses allowed producer/source semantics (``producer_kind='task'``),
        exact per-goal objective revision, canonical channel proof, and
        explicit freshness.  Claims only this record's acceptance criterion.
        """

        channel_proof = self.build_channel_proof()
        channel_revision = self.channel_proof_revision
        validation_receipt = {
            "passed": True,
            "status": "passed",
            "producer_channel": self.producer_channel,
            "channel_proof_revision": channel_revision,
            "channel_proof": channel_proof,
            "goal_id": self.goal_id,
            "acceptance_criterion": self.acceptance_criterion,
            "satisfied_requirements": list(self.satisfied_requirements),
            "repository_id": self.repository_id,
            "git_tree_id": self.tree_id,
            "repository_forest_cid": self.repository_forest_cid,
            "objective_completion_tree_id": self.objective_completion_tree_id,
            "objective_revision": self.objective_revision,
            "policy_cid": self.policy_cid,
            "capability_cid": self.capability_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "circuit_cid": self.circuit_cid,
            "task_evidence_cids": list(self.task_evidence_cids),
            "child_goal_evidence_cids": list(self.child_goal_evidence_cids),
            "adversarial_evidence_cids": list(self.adversarial_evidence_cids),
            "analyzer_evidence_cids": list(self.analyzer_evidence_cids),
            "benchmark_receipt_cid": self.benchmark_receipt_cid,
            "rollout_decision_cid": self.rollout_decision_cid,
            "supervisor_health_cid": self.supervisor_health_cid,
            "premise_cids": list(self.premise_cids),
            "evidence_body": self._identity_body(),
            "producing_task_id": self.producing_task_id,
        }
        return CompletionEvidence(
            acceptance_criterion=self.acceptance_criterion,
            producing_task_or_scan=self.producing_task_id,
            producer_kind=self.producer_kind,
            producer_channel=self.producer_channel,
            channel_proof_revision=channel_revision,
            validation_receipt=validation_receipt,
            repository_tree=self.tree_id,
            tree_id=self.tree_id,
            repository_id=self.repository_id,
            objective_revision=self.objective_revision,
            analyzer_version=self.analyzer_revision,
            configuration_revision=self.configuration_revision,
            freshness={
                "observed_at_ms": self.observed_at_ms,
                "fresh_until_ms": self.fresh_until_ms,
            },
            provenance_cid=self.evidence_id,
            validation_passed=True,
            observed_at=datetime.fromtimestamp(
                self.observed_at_ms / 1000, tz=UTC
            ),
            fresh_until=datetime.fromtimestamp(
                self.fresh_until_ms / 1000, tz=UTC
            ),
            metadata={
                "goal_id": self.goal_id,
                "authority": self.authority,
                "repository_forest_cid": self.repository_forest_cid,
                "objective_completion_tree_id": (
                    self.objective_completion_tree_id
                ),
                "source_tier": "validation",
                "source_kind": "typed_receipt",
                "producer_channel": self.producer_channel,
                "channel_proof_revision": channel_revision,
                "satisfied_requirements": list(self.satisfied_requirements),
                "evidence_source_policy": {
                    "satisfies": True,
                    "reason_codes": (),
                    "match_kind": "typed_receipt",
                    "source_tier": "validation",
                },
            },
        )

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> ProofTestReuseCompletionEvidence:
        if not isinstance(payload, Mapping):
            raise ProofTestReuseCurrentTreeGateError(
                "completion evidence payload must be a mapping"
            )
        tree = _text(payload.get("tree_id") or payload.get("git_tree_id"))
        return cls(
            repository_id=_text(payload.get("repository_id")),
            tree_id=tree,
            commit_id=_text(payload.get("commit_id")),
            gitlink_state_cid=_text(payload.get("gitlink_state_cid")),
            repository_forest_cid=_text(payload.get("repository_forest_cid")),
            objective_completion_tree_id=_text(
                payload.get("objective_completion_tree_id")
            ),
            policy_cid=_text(payload.get("policy_cid")),
            capability_cid=_text(payload.get("capability_cid")),
            verifying_key_cid=_text(payload.get("verifying_key_cid")),
            circuit_cid=_text(payload.get("circuit_cid")),
            objective_revision=_text(payload.get("objective_revision")),
            task_evidence_cids=tuple(payload.get("task_evidence_cids") or ()),
            child_goal_evidence_cids=tuple(
                payload.get("child_goal_evidence_cids") or ()
            ),
            adversarial_evidence_cids=tuple(
                payload.get("adversarial_evidence_cids") or ()
            ),
            analyzer_evidence_cids=tuple(
                payload.get("analyzer_evidence_cids") or ()
            ),
            benchmark_receipt_cid=_text(payload.get("benchmark_receipt_cid")),
            rollout_decision_cid=_text(payload.get("rollout_decision_cid")),
            supervisor_health_cid=_text(payload.get("supervisor_health_cid")),
            observed_at_ms=int(payload.get("observed_at_ms") or 0),
            fresh_until_ms=int(payload.get("fresh_until_ms") or 0),
            goal_id=_text(payload.get("goal_id")),
            acceptance_criterion=_text(payload.get("acceptance_criterion")),
            satisfied_requirements=tuple(
                payload.get("satisfied_requirements") or ()
            ),
            producing_task_id=_text(payload.get("producing_task_id"))
            or FINAL_GATE_TASK_ID,
            authority=_text(payload.get("authority")) or _AUTHORITATIVE,
            producer_kind=_text(payload.get("producer_kind")) or "task",
            producer_channel=_text(payload.get("producer_channel"))
            or DEFAULT_PRODUCER_CHANNEL,
            analyzer_revision=_text(payload.get("analyzer_revision"))
            or DEFAULT_ANALYZER_REVISION,
            configuration_revision=_text(payload.get("configuration_revision"))
            or DEFAULT_CONFIGURATION_REVISION,
            premise_cids=tuple(payload.get("premise_cids") or ()),
        )


@dataclass(frozen=True, slots=True)
class ProofTestReuseCurrentTreeGateDecision:
    """Typed outcome.  Rejection can never carry completion evidence."""

    passed: bool
    reason_codes: tuple[str, ...]
    evaluated_at_ms: int
    final_gate_completion_evidence: ProofTestReuseCompletionEvidence | None = None
    root_completion_evidence: ProofTestReuseCompletionEvidence | None = None

    def __post_init__(self) -> None:
        reasons = tuple(dict.fromkeys(_text(item) for item in self.reason_codes))
        if any(not reason for reason in reasons) or len(reasons) > _MAX_REASONS:
            raise ProofTestReuseCurrentTreeGateError("invalid gate reason codes")
        object.__setattr__(self, "reason_codes", reasons)
        if self.passed:
            if reasons:
                raise ProofTestReuseCurrentTreeGateError(
                    "passing gate requires no rejection reasons"
                )
            if (
                self.final_gate_completion_evidence is None
                or self.root_completion_evidence is None
            ):
                raise ProofTestReuseCurrentTreeGateError(
                    "passing gate requires separate G110 and G000 evidence"
                )
            if self.final_gate_completion_evidence.goal_id != FINAL_GATE_GOAL_ID:
                raise ProofTestReuseCurrentTreeGateError(
                    "final_gate_completion_evidence must target PTR-G110"
                )
            if self.root_completion_evidence.goal_id != ROOT_GOAL_ID:
                raise ProofTestReuseCurrentTreeGateError(
                    "root_completion_evidence must target PTR-G000"
                )
            if (
                self.final_gate_completion_evidence.acceptance_criterion
                != FINAL_GATE_ACCEPTANCE_CRITERION
            ):
                raise ProofTestReuseCurrentTreeGateError(
                    "G110 evidence criterion mismatch"
                )
            if (
                self.root_completion_evidence.acceptance_criterion
                != ROOT_ACCEPTANCE_CRITERION
            ):
                raise ProofTestReuseCurrentTreeGateError(
                    "G000 evidence criterion mismatch"
                )
        elif (
            self.final_gate_completion_evidence is not None
            or self.root_completion_evidence is not None
        ):
            raise ProofTestReuseCurrentTreeGateError(
                "failed gate cannot emit completion evidence"
            )

    @property
    def completion_evidence(self) -> ProofTestReuseCompletionEvidence | None:
        """Compatibility alias for root (G000) completion evidence."""

        return self.root_completion_evidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_TEST_REUSE_GATE_DECISION_SCHEMA,
            "interface": PROOF_TEST_REUSE_CURRENT_TREE_GATE_INTERFACE,
            "version": PROOF_TEST_REUSE_CURRENT_TREE_GATE_VERSION,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "evaluated_at_ms": self.evaluated_at_ms,
            "final_gate_completion_evidence": (
                self.final_gate_completion_evidence.to_dict()
                if self.final_gate_completion_evidence is not None
                else None
            ),
            "root_completion_evidence": (
                self.root_completion_evidence.to_dict()
                if self.root_completion_evidence is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> ProofTestReuseCurrentTreeGateDecision:
        if not isinstance(payload, Mapping):
            raise ProofTestReuseCurrentTreeGateError(
                "gate decision payload must be a mapping"
            )
        final_raw = payload.get("final_gate_completion_evidence")
        root_raw = payload.get("root_completion_evidence")
        # Legacy single-evidence payloads are not admissible after PTR-122.
        if payload.get("passed") is True and (
            not isinstance(final_raw, Mapping) or not isinstance(root_raw, Mapping)
        ):
            raise ProofTestReuseCurrentTreeGateError(
                "passing decision requires both G110 and G000 evidence payloads"
            )
        final_ev = (
            ProofTestReuseCompletionEvidence.from_dict(final_raw)
            if isinstance(final_raw, Mapping)
            else None
        )
        root_ev = (
            ProofTestReuseCompletionEvidence.from_dict(root_raw)
            if isinstance(root_raw, Mapping)
            else None
        )
        return cls(
            passed=bool(payload.get("passed")),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            evaluated_at_ms=int(payload.get("evaluated_at_ms") or 0),
            final_gate_completion_evidence=final_ev,
            root_completion_evidence=root_ev,
        )


@dataclass(frozen=True, slots=True)
class ProofTestReusePersistedGateBundle:
    """Replayable persisted envelope for one gate evaluation."""

    SCHEMA: ClassVar[str] = PROOF_TEST_REUSE_PERSISTED_GATE_BUNDLE_SCHEMA

    repository_id: str
    git_tree_id: str
    repository_forest_cid: str
    objective_completion_tree_id: str
    objective_revision: str
    policy_cid: str
    capability_cid: str
    verifying_key_cid: str
    circuit_cid: str
    producing_task_id: str
    decision: ProofTestReuseCurrentTreeGateDecision
    gate_bindings: Mapping[str, Any]
    evaluate_packet: Mapping[str, Any]
    premise_cids: tuple[str, ...]
    retained_premise_bytes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_identity_domains_distinct(
            self.git_tree_id,
            self.repository_forest_cid,
            self.objective_completion_tree_id,
        )
        if self.producing_task_id != FINAL_GATE_TASK_ID:
            raise ProofTestReuseCurrentTreeGateError(
                "persisted gate bundle must be produced by PTR-122"
            )
        if not isinstance(self.decision, ProofTestReuseCurrentTreeGateDecision):
            raise ProofTestReuseCurrentTreeGateError(
                "decision must be ProofTestReuseCurrentTreeGateDecision"
            )
        object.__setattr__(self, "gate_bindings", dict(self.gate_bindings or {}))
        object.__setattr__(
            self, "evaluate_packet", dict(self.evaluate_packet or {})
        )
        object.__setattr__(
            self,
            "premise_cids",
            tuple(sorted({_text(item) for item in self.premise_cids if _text(item)})),
        )
        object.__setattr__(
            self,
            "retained_premise_bytes",
            {
                _text(key): str(value)
                for key, value in dict(self.retained_premise_bytes or {}).items()
                if _text(key) and isinstance(value, str)
            },
        )
        # Every retained premise CID must verify against its retained bytes.
        for cid, text in self.retained_premise_bytes.items():
            data = text.encode("utf-8")
            if not verify_retained_bytes(cid, data):
                # Also accept internal sha256 fingerprints for non-dag-json
                # helper identities retained during tests.
                if not (
                    cid.startswith("sha256:")
                    and cid == "sha256:" + hashlib.sha256(data).hexdigest()
                ):
                    raise ProofTestReuseCurrentTreeGateError(
                        f"retained premise CID failed verification: {cid}"
                    )

    @property
    def interface(self) -> str:
        return PROOF_TEST_REUSE_PERSISTED_GATE_BUNDLE_INTERFACE

    def _identity_body(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.interface,
            "version": PROOF_TEST_REUSE_CURRENT_TREE_GATE_VERSION,
            "repository_id": self.repository_id,
            "git_tree_id": self.git_tree_id,
            "tree_id": self.git_tree_id,
            "repository_forest_cid": self.repository_forest_cid,
            "objective_completion_tree_id": self.objective_completion_tree_id,
            "objective_revision": self.objective_revision,
            "policy_cid": self.policy_cid,
            "capability_cid": self.capability_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "circuit_cid": self.circuit_cid,
            "producing_task_id": self.producing_task_id,
            "decision": self.decision.to_dict(),
            "gate_bindings": dict(self.gate_bindings),
            "evaluate_packet": dict(self.evaluate_packet),
            "premise_cids": list(self.premise_cids),
            "retained_premise_bytes": dict(self.retained_premise_bytes),
        }

    def to_dict(self) -> dict[str, Any]:
        body = self._identity_body()
        body["bundle_cid"] = self.bundle_cid
        return body

    @property
    def bundle_cid(self) -> str:
        return cid_for_mapping(self._identity_body())

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> ProofTestReusePersistedGateBundle:
        if not isinstance(payload, Mapping):
            raise ProofTestReuseCurrentTreeGateError(
                "persisted gate bundle must be a mapping"
            )
        if payload.get("schema") not in {
            cls.SCHEMA,
            PROOF_TEST_REUSE_PERSISTED_GATE_BUNDLE_SCHEMA,
        }:
            raise ProofTestReuseCurrentTreeGateError(
                "unsupported persisted gate bundle schema"
            )
        decision_payload = payload.get("decision")
        if not isinstance(decision_payload, Mapping):
            raise ProofTestReuseCurrentTreeGateError(
                "persisted gate bundle missing decision"
            )
        return cls(
            repository_id=_text(payload.get("repository_id")),
            git_tree_id=_text(
                _value(payload, "git_tree_id", "tree_id")  # type: ignore[arg-type]
            ),
            repository_forest_cid=_text(payload.get("repository_forest_cid")),
            objective_completion_tree_id=_text(
                payload.get("objective_completion_tree_id")
            ),
            objective_revision=_text(payload.get("objective_revision")),
            policy_cid=_text(payload.get("policy_cid")),
            capability_cid=_text(payload.get("capability_cid")),
            verifying_key_cid=_text(payload.get("verifying_key_cid")),
            circuit_cid=_text(payload.get("circuit_cid")),
            producing_task_id=_text(payload.get("producing_task_id"))
            or FINAL_GATE_TASK_ID,
            decision=ProofTestReuseCurrentTreeGateDecision.from_dict(
                decision_payload
            ),
            gate_bindings=_mapping(payload.get("gate_bindings")),
            evaluate_packet=_mapping(payload.get("evaluate_packet")),
            premise_cids=tuple(payload.get("premise_cids") or ()),
            retained_premise_bytes=_mapping(
                payload.get("retained_premise_bytes")
            ),
        )

    def replay(
        self,
        *,
        rollout_policy: ProofReuseRolloutPolicy,
        clock: Callable[[], float] | None = None,
        required_task_ids: frozenset[str] | None = None,
        required_child_goal_ids: frozenset[str] | None = None,
    ) -> ProofTestReuseCurrentTreeGateDecision:
        """Deserialize retained premises and re-evaluate the gate strictly."""

        return verify_persisted_current_tree_gate_bundle(
            self,
            rollout_policy=rollout_policy,
            clock=clock,
            required_task_ids=required_task_ids,
            required_child_goal_ids=required_child_goal_ids,
        )


def verify_persisted_current_tree_gate_bundle(
    bundle: ProofTestReusePersistedGateBundle | Mapping[str, Any],
    *,
    rollout_policy: ProofReuseRolloutPolicy,
    clock: Callable[[], float] | None = None,
    required_task_ids: frozenset[str] | None = None,
    required_child_goal_ids: frozenset[str] | None = None,
    required_adversarial_populations: frozenset[str] | None = None,
    required_analyzers: frozenset[str] | None = None,
) -> ProofTestReuseCurrentTreeGateDecision:
    """Strictly deserialize a bundle and replay the gate against retained premises."""

    if not isinstance(bundle, ProofTestReusePersistedGateBundle):
        bundle = ProofTestReusePersistedGateBundle.from_dict(bundle)

    # Re-verify every retained premise CID before replaying.
    for cid, text in bundle.retained_premise_bytes.items():
        data = text.encode("utf-8")
        if verify_retained_bytes(cid, data):
            continue
        if cid.startswith("sha256:") and cid == (
            "sha256:" + hashlib.sha256(data).hexdigest()
        ):
            continue
        raise ProofTestReuseCurrentTreeGateError(
            f"premise CID verification failed during replay: {cid}"
        )

    bindings = dict(bundle.gate_bindings)
    gate = ProofTestReuseCurrentTreeGate(
        repository_id=bundle.repository_id,
        tree_id=bundle.git_tree_id,
        commit_id=_text(bindings.get("commit_id")),
        gitlink_state_cid=_text(bindings.get("gitlink_state_cid")),
        repository_forest_cid=bundle.repository_forest_cid,
        objective_completion_tree_id=bundle.objective_completion_tree_id,
        capability_cid=bundle.capability_cid,
        verifying_key_cid=bundle.verifying_key_cid,
        circuit_cid=bundle.circuit_cid,
        objective_revision=bundle.objective_revision,
        g110_objective_revision=_text(
            bindings.get("g110_objective_revision")
        )
        or bundle.objective_revision,
        root_objective_revision=_text(
            bindings.get("root_objective_revision")
        )
        or bundle.objective_revision,
        rollout_policy=rollout_policy,
        required_task_ids=required_task_ids or frozenset(
            bindings.get("required_task_ids") or REQUIRED_PTR_TASK_IDS
        ),
        required_child_goal_ids=required_child_goal_ids
        or frozenset(
            bindings.get("required_child_goal_ids") or REQUIRED_CHILD_GOAL_IDS
        ),
        required_adversarial_populations=required_adversarial_populations
        or frozenset(
            bindings.get("required_adversarial_populations")
            or REQUIRED_ADVERSARIAL_POPULATIONS
        ),
        required_analyzers=required_analyzers
        or frozenset(bindings.get("required_analyzers") or REQUIRED_ANALYZERS),
        clock=clock
        or (lambda: float(bundle.decision.evaluated_at_ms) / 1000.0),
    )
    if gate.policy_cid != bundle.policy_cid:
        raise ProofTestReuseCurrentTreeGateError(
            "persisted bundle policy_cid does not match rollout policy binding"
        )
    packet = _deserialize_evaluate_packet(bundle.evaluate_packet)
    replayed = gate.evaluate(**packet)
    if replayed.passed != bundle.decision.passed:
        raise ProofTestReuseCurrentTreeGateError(
            "persisted gate bundle replay outcome mismatch"
        )
    if replayed.passed:
        if (
            replayed.final_gate_completion_evidence is None
            or replayed.root_completion_evidence is None
            or bundle.decision.final_gate_completion_evidence is None
            or bundle.decision.root_completion_evidence is None
        ):
            raise ProofTestReuseCurrentTreeGateError(
                "persisted gate bundle missing completion evidence on replay"
            )
        if (
            replayed.final_gate_completion_evidence.evidence_id
            != bundle.decision.final_gate_completion_evidence.evidence_id
            or replayed.root_completion_evidence.evidence_id
            != bundle.decision.root_completion_evidence.evidence_id
        ):
            raise ProofTestReuseCurrentTreeGateError(
                "persisted gate bundle evidence identity mismatch on replay"
            )
    elif tuple(replayed.reason_codes) != tuple(bundle.decision.reason_codes):
        # Allow reason-code order differences only via set equality for
        # non-passing replays when both failed.
        if set(replayed.reason_codes) != set(bundle.decision.reason_codes):
            raise ProofTestReuseCurrentTreeGateError(
                "persisted gate bundle reason codes mismatch on replay"
            )
    return replayed


@dataclass(frozen=True, slots=True)
class ProofTestReuseCurrentTreeGate:
    """Evaluate the sealed PTR population against one current-tree identity."""

    repository_id: str
    tree_id: str
    commit_id: str
    gitlink_state_cid: str
    repository_forest_cid: str
    objective_completion_tree_id: str
    capability_cid: str
    verifying_key_cid: str
    circuit_cid: str
    objective_revision: str
    rollout_policy: ProofReuseRolloutPolicy
    g110_objective_revision: str = ""
    root_objective_revision: str = ""
    required_task_ids: frozenset[str] = REQUIRED_PTR_TASK_IDS
    required_child_goal_ids: frozenset[str] = REQUIRED_CHILD_GOAL_IDS
    required_adversarial_populations: frozenset[str] = (
        REQUIRED_ADVERSARIAL_POPULATIONS
    )
    required_analyzers: frozenset[str] = REQUIRED_ANALYZERS
    required_supervisor_lane_ids: frozenset[str] = REQUIRED_SUPERVISOR_LANE_IDS
    clock: Callable[[], float] = field(
        default=time.time, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "commit_id",
            "gitlink_state_cid",
            "repository_forest_cid",
            "objective_completion_tree_id",
            "capability_cid",
            "verifying_key_cid",
            "circuit_cid",
            "objective_revision",
        ):
            if not _text(getattr(self, name)):
                raise ProofTestReuseCurrentTreeGateError(f"{name} is required")
        _assert_identity_domains_distinct(
            self.tree_id,
            self.repository_forest_cid,
            self.objective_completion_tree_id,
        )
        if not isinstance(self.rollout_policy, ProofReuseRolloutPolicy):
            raise ProofTestReuseCurrentTreeGateError(
                "rollout_policy must be ProofReuseRolloutPolicy"
            )
        object.__setattr__(
            self,
            "g110_objective_revision",
            _text(self.g110_objective_revision) or self.objective_revision,
        )
        object.__setattr__(
            self,
            "root_objective_revision",
            _text(self.root_objective_revision) or self.objective_revision,
        )
        for name in (
            "required_task_ids",
            "required_child_goal_ids",
            "required_adversarial_populations",
            "required_analyzers",
            "required_supervisor_lane_ids",
        ):
            values = frozenset(_text(item) for item in getattr(self, name))
            if not values or any(not item for item in values):
                raise ProofTestReuseCurrentTreeGateError(f"{name} is invalid")
            object.__setattr__(self, name, values)
        # Production callers must never reintroduce G110 as a child premise.
        if FINAL_GATE_GOAL_ID in self.required_child_goal_ids:
            raise ProofTestReuseCurrentTreeGateError(
                "PTR-G110 must not be required as a child premise (self-reference)"
            )
        # Sealed expansion + runtime-activation repair must remain in production.
        sealed_expansion = {
            "PTR-108",
            "PTR-109",
            "PTR-110",
            "PTR-111",
            "PTR-112",
            "PTR-120",
            "PTR-121",
            "PTR-122",
            "PTR-130",
            *RUNTIME_ACTIVATION_REPAIR_TASK_IDS,
            *PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS,
        }
        if self.required_task_ids == REQUIRED_PTR_TASK_IDS and not sealed_expansion.issubset(
            self.required_task_ids
        ):
            raise ProofTestReuseCurrentTreeGateError(
                "required task population is missing sealed expansion tasks"
            )
        if self.required_task_ids == REQUIRED_PTR_TASK_IDS and (
            len(self.required_task_ids) != SEALED_PRODUCTION_TASK_COUNT
        ):
            raise ProofTestReuseCurrentTreeGateError(
                "production population must be the exact 60-task board"
            )
        if (
            self.required_task_ids == REQUIRED_PTR_TASK_IDS
            and not RUNTIME_ACTIVATION_REPAIR_TASK_IDS.issubset(
                self.required_task_ids
            )
        ):
            raise ProofTestReuseCurrentTreeGateError(
                "production population missing runtime-activation repair tasks"
            )
        if (
            self.required_task_ids == REQUIRED_PTR_TASK_IDS
            and not PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS.issubset(
                self.required_task_ids
            )
        ):
            raise ProofTestReuseCurrentTreeGateError(
                "production population missing production-activation correction tasks"
            )

    @property
    def policy_cid(self) -> str:
        return self.rollout_policy.policy_binding_id

    @property
    def git_tree_id(self) -> str:
        return self.tree_id

    def _bindings(self, record: Mapping[str, Any], prefix: str) -> list[str]:
        reasons: list[str] = []
        expected = {
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "commit_id": self.commit_id,
            "gitlink_state_cid": self.gitlink_state_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "policy_cid": self.policy_cid,
            "capability_cid": self.capability_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "circuit_cid": self.circuit_cid,
        }
        aliases = {
            "repository_forest_cid": ("repository_forest_cid", "forest_cid"),
            "verifying_key_cid": ("verifying_key_cid", "key_cid"),
            "tree_id": ("tree_id", "git_tree_id"),
            "commit_id": ("commit_id", "git_commit_id"),
        }
        for name, wanted in expected.items():
            actual = _text(_value(record, *aliases.get(name, (name,))))
            if actual != wanted:
                reasons.append(_reason(f"{name}_mismatch", prefix))
        # Objective-completion identity is optional on every record but, when
        # present, must match the gate's distinct domain.
        completion_id = _text(
            _value(
                record,
                "objective_completion_tree_id",
                "completion_tree_id",
            )
        )
        if completion_id and completion_id != self.objective_completion_tree_id:
            reasons.append(
                _reason("objective_completion_tree_id_mismatch", prefix)
            )
        if completion_id and completion_id in {
            self.tree_id,
            self.repository_forest_cid,
        }:
            reasons.append(
                _reason("identity_domain_collision", prefix)
            )
        if record.get("gitlink_closure_complete") is not True:
            reasons.append(_reason("recursive_gitlinks_incomplete", prefix))
        return reasons

    @staticmethod
    def _fresh(record: Mapping[str, Any], now_ms: int) -> bool:
        observed = _timestamp_ms(
            _value(record, "observed_at_ms", "observed_at", "verified_at_ms")
        )
        fresh_until = _timestamp_ms(
            _value(record, "fresh_until_ms", "fresh_until")
        )
        return (
            observed is not None
            and fresh_until is not None
            and 0 <= observed <= now_ms < fresh_until
        )

    @staticmethod
    def _evidence_cid(record: Mapping[str, Any]) -> str:
        supplied = _text(
            _value(
                record,
                "evidence_cid",
                "provenance_cid",
                "task_cid",
                "receipt_id",
            )
        )
        if supplied:
            return supplied
        return _content_id(dict(record))

    def _validate_premise_cid(
        self,
        record: Mapping[str, Any],
        *,
        prefix: str,
    ) -> list[str]:
        """Validate every retained premise CID on one record."""

        reasons: list[str] = []
        cid = self._evidence_cid(record)
        if not cid:
            reasons.append(_reason("missing_premise_cid", prefix))
            return reasons
        retained = _value(
            record,
            "retained_canonical_utf8",
            "canonical_utf8",
            "retained_bytes_utf8",
        )
        if isinstance(retained, str) and retained:
            data = retained.encode("utf-8")
            if cid.startswith("sha256:"):
                if cid != "sha256:" + hashlib.sha256(data).hexdigest():
                    reasons.append(_reason("premise_cid_mismatch", prefix))
            elif not verify_retained_bytes(cid, data):
                reasons.append(_reason("premise_cid_mismatch", prefix))
        elif cid.startswith("b"):
            # Authoritative dag-json CIDs require retained bytes for recheck.
            if not _text(
                _value(record, "evidence_cid", "provenance_cid", "task_cid")
            ):
                reasons.append(_reason("premise_cid_bytes_missing", prefix))
        return reasons

    def _task_provenance_reasons(
        self,
        record: Mapping[str, Any],
        task_id: str,
    ) -> list[str]:
        """Validate one member of the closed task-completion provenance union."""

        reasons: list[str] = []
        quarantine_fields = (
            "status",
            "state",
            "queue_status",
            "merge_status",
            "validation_disposition",
            "disposition",
            "outcome",
        )
        if record.get("quarantined") is True or any(
            _text(record.get(name)).lower() in _QUARANTINE_STATES
            for name in quarantine_fields
        ):
            reasons.append(_reason("quarantined_task", task_id))

        raw = record.get("task_provenance")
        if not isinstance(raw, Mapping):
            reasons.append(_reason("missing_task_provenance", task_id))
            return reasons
        provenance = dict(raw)
        kind_text = _text(provenance.get("kind")).lower()
        try:
            kind = TaskCompletionProvenanceKind(kind_text)
        except ValueError:
            reasons.append(_reason("unsupported_task_provenance", task_id))
            return reasons

        text_fields, true_fields = _TASK_PROVENANCE_FIELDS[kind]
        expected_fields = {"kind"} | set(text_fields) | set(true_fields)
        if set(provenance) != expected_fields:
            reasons.append(_reason("malformed_task_provenance", task_id))
            return reasons
        if any(
            not isinstance(provenance.get(name), str)
            or not provenance[name]
            or provenance[name] != provenance[name].strip()
            or "\x00" in provenance[name]
            for name in text_fields
        ):
            reasons.append(_reason("malformed_task_provenance", task_id))
        if any(provenance.get(name) is not True for name in true_fields):
            reasons.append(_reason("unsuccessful_task_provenance", task_id))

        if kind is TaskCompletionProvenanceKind.OPERATOR_PLANNING_SEAL:
            if (
                _text(provenance.get("sealed_objective_revision"))
                != self.objective_revision
            ):
                reasons.append(
                    _reason("planning_seal_revision_mismatch", task_id)
                )
        elif kind is TaskCompletionProvenanceKind.OPERATOR_REVIEWED_INTEGRATION:
            if (
                _text(provenance.get("integration_target_commit_id"))
                != self.commit_id
            ):
                reasons.append(
                    _reason("reviewed_integration_target_mismatch", task_id)
                )
        elif (
            kind
            is TaskCompletionProvenanceKind.RETROSPECTIVE_INTEGRATION_VERIFICATION
        ):
            if (
                _text(provenance.get("ancestry_target_commit_id"))
                != self.commit_id
            ):
                reasons.append(
                    _reason("retrospective_ancestry_target_mismatch", task_id)
                )
            rerun_bindings = {
                "current_tree_rerun_repository_id": self.repository_id,
                "current_tree_rerun_tree_id": self.tree_id,
                "current_tree_rerun_commit_id": self.commit_id,
                "current_tree_rerun_gitlink_state_cid": self.gitlink_state_cid,
                "current_tree_rerun_repository_forest_cid": (
                    self.repository_forest_cid
                ),
                "current_tree_rerun_policy_cid": self.policy_cid,
                "current_tree_rerun_capability_cid": self.capability_cid,
                "current_tree_rerun_verifying_key_cid": self.verifying_key_cid,
                "current_tree_rerun_circuit_cid": self.circuit_cid,
            }
            if any(
                _text(provenance.get(name)) != expected
                for name, expected in rerun_bindings.items()
            ):
                reasons.append(
                    _reason(
                        "retrospective_current_tree_rerun_binding_mismatch",
                        task_id,
                    )
                )
            if (
                _text(provenance.get("approved_policy_cid"))
                != self.policy_cid
            ):
                reasons.append(
                    _reason("retrospective_policy_approval_mismatch", task_id)
                )
        return reasons

    def _rollout_readiness_reasons(
        self,
        *,
        decision: ProofReuseRolloutDecision,
        promotion: ProofReusePromotionEvidence,
        now_ms: int,
    ) -> list[str]:
        """Validate fresh readiness without consuming this gate's own result."""

        reasons: list[str] = []
        observed_at_ms = _timestamp_ms(promotion.observed_at)
        max_age_ms = self.rollout_policy.max_evidence_age_seconds * 1000
        max_future_skew_ms = (
            self.rollout_policy.max_future_skew_seconds * 1000
        )
        if (
            observed_at_ms is None
            or observed_at_ms < now_ms - max_age_ms
            or observed_at_ms > now_ms + max_future_skew_ms
        ):
            reasons.append("rollout_readiness_stale")

        if (
            promotion.repository_id != self.repository_id
            or promotion.tree_id != self.tree_id
            or promotion.policy_id != self.rollout_policy.policy_id
            or promotion.policy_revision
            != self.rollout_policy.policy_revision
        ):
            reasons.append("rollout_readiness_binding_mismatch")

        if (
            decision.current_stage is not promotion.current_stage
            or decision.requested_stage is not promotion.target_stage
            or decision.effective_stage is not promotion.target_stage
        ):
            reasons.append("rollout_readiness_stage_mismatch")
        if (
            promotion.target_stage.rank
            >= ProofReuseRolloutStage.ELIGIBLE_DEFAULT.rank
            or decision.requested_stage.rank
            >= ProofReuseRolloutStage.ELIGIBLE_DEFAULT.rank
            or decision.effective_stage.rank
            >= ProofReuseRolloutStage.ELIGIBLE_DEFAULT.rank
        ):
            reasons.append("rollout_readiness_not_pre_default")

        # The result under evaluation cannot be supplied as one of its own
        # premises.  ``None`` is an unset precondition and ``False`` is an
        # explicit statement that the final gate has not run yet.
        if promotion.current_tree_gate_passed is True:
            reasons.append("rollout_current_tree_gate_preclaimed")
        return reasons

    def _supervisor_health_reasons(
        self,
        record: Mapping[str, Any],
        *,
        now_ms: int,
    ) -> tuple[list[str], str]:
        """Validate a fresh current-tree/config-bound three-lane health receipt."""

        reasons: list[str] = []
        if not record:
            return ["supervisor_health_missing"], ""
        reasons.extend(self._bindings(record, "supervisor_health"))
        if _text(record.get("authority")).lower() != _AUTHORITATIVE:
            reasons.append("supervisor_health_non_authoritative")
        if not self._fresh(record, now_ms):
            reasons.append("supervisor_health_stale")

        config_cid = _text(
            _value(
                record,
                "config_cid",
                "configuration_cid",
                "config_revision",
                "configuration_revision",
            )
        )
        if not config_cid:
            reasons.append("supervisor_health_config_missing")

        # Optional explicit objective-completion / config binding.
        completion_id = _text(
            _value(
                record,
                "objective_completion_tree_id",
                "completion_tree_id",
            )
        )
        if completion_id and completion_id != self.objective_completion_tree_id:
            reasons.append("supervisor_health_completion_identity_mismatch")

        lanes_raw = record.get("lanes")
        lane_records: dict[str, Mapping[str, Any]] = {}
        if isinstance(lanes_raw, Mapping):
            for lane_id, value in lanes_raw.items():
                body = dict(_record(value))
                body.setdefault("lane_id", lane_id)
                lane_records[_text(lane_id)] = body
        elif isinstance(lanes_raw, Sequence) and not isinstance(
            lanes_raw, (str, bytes)
        ):
            for raw in lanes_raw:
                body = _record(raw)
                lane_id = _text(_value(body, "lane_id", "id", "name"))
                if lane_id:
                    lane_records[lane_id] = body
        else:
            reasons.append("supervisor_health_lanes_missing")

        required = self.required_supervisor_lane_ids
        present = set(lane_records)
        for missing in sorted(required - present):
            reasons.append(_reason("missing_supervisor_lane", missing))
        for unexpected in sorted(present - required):
            reasons.append(_reason("unexpected_supervisor_lane", unexpected))
        if len(required) != REQUIRED_SUPERVISOR_LANE_COUNT:
            reasons.append("supervisor_health_lane_count_invalid")
        declared_count = record.get("lane_count")
        if declared_count is not None and declared_count != len(required):
            reasons.append("supervisor_health_lane_count_mismatch")
        if record.get("all_lanes_healthy") is not True:
            reasons.append("supervisor_health_not_all_lanes_healthy")

        for lane_id in sorted(required & present):
            lane = lane_records[lane_id]
            if lane.get("healthy") is not True:
                reasons.append(_reason("supervisor_lane_unhealthy", lane_id))
            if _text(lane.get("authority")).lower() not in {
                "",
                _AUTHORITATIVE,
            }:
                reasons.append(
                    _reason("supervisor_lane_non_authoritative", lane_id)
                )
            # Lane may carry current-tree bindings when present.
            for field_name, wanted in (
                ("repository_id", self.repository_id),
                ("tree_id", self.tree_id),
                ("repository_forest_cid", self.repository_forest_cid),
            ):
                actual = _text(
                    _value(
                        lane,
                        field_name,
                        "git_tree_id" if field_name == "tree_id" else field_name,
                        "forest_cid"
                        if field_name == "repository_forest_cid"
                        else field_name,
                    )
                )
                if actual and actual != wanted:
                    reasons.append(
                        _reason(
                            f"supervisor_lane_{field_name}_mismatch",
                            lane_id,
                        )
                    )

        reasons.extend(
            self._validate_premise_cid(record, prefix="supervisor_health")
        )
        return reasons, self._evidence_cid(record)

    def _requires_repair_evidence(self) -> bool:
        """The corrective production population needs fresh activation evidence."""

        return bool(
            PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS & self.required_task_ids
        )

    def _repair_evidence_reasons(
        self,
        record: Mapping[str, Any],
        *,
        now_ms: int,
    ) -> tuple[list[str], str]:
        """Validate fresh production-runtime activation evidence (PTR-149)."""

        reasons: list[str] = []
        if not self._requires_repair_evidence():
            return [], ""
        if not record:
            return ["repair_evidence_missing"], ""
        reasons.extend(self._bindings(record, "repair_evidence"))
        if _text(record.get("authority")).lower() != _AUTHORITATIVE:
            reasons.append("repair_evidence_non_authoritative")
        if not self._fresh(record, now_ms):
            reasons.append("repair_evidence_stale")

        repair_id = _text(
            _value(record, "repair_id", "repair", "population_id")
        )
        if repair_id != PRODUCTION_RUNTIME_ACTIVATION_ID:
            reasons.append("repair_evidence_id_mismatch")

        producer_task_id = _text(
            _value(record, "producer_task_id", "producing_task_id", "task_id")
        )
        if producer_task_id != PRODUCTION_RUNTIME_ACTIVATION_PRODUCER_TASK_ID:
            reasons.append("repair_evidence_producer_task_mismatch")

        covered = frozenset(
            _text(item)
            for item in (
                record.get("repair_task_ids")
                or record.get("task_ids")
                or ()
            )
        )
        missing_tasks = sorted(PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS - covered)
        if missing_tasks:
            reasons.extend(
                _reason("repair_evidence_missing_task", task_id)
                for task_id in missing_tasks
            )
        # Historical PTR-142 coverage cannot be mixed in to make the new
        # corrective evidence appear complete.
        unexpected = sorted(covered - PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS)
        if unexpected:
            reasons.extend(
                _reason("repair_evidence_unexpected_task", task_id)
                for task_id in unexpected
            )

        if record.get("passed") is not True:
            reasons.append("repair_evidence_failed")
        false_skips = record.get("false_skips")
        if isinstance(false_skips, bool) or false_skips not in (0, None):
            if false_skips != 0:
                reasons.append("repair_evidence_false_skips")
        if record.get("zero_false_skip_assurance") is not True:
            reasons.append("repair_evidence_zero_false_skip_missing")
        if record.get("activation_e2e_passed") is not True:
            reasons.append("repair_evidence_activation_e2e_missing")
        for field_name, reason_code in (
            ("zero_injection_default_path", "repair_evidence_default_path_missing"),
            ("three_repository_cold_warm", "repair_evidence_three_repo_missing"),
            ("real_groth16_certificate", "repair_evidence_real_groth16_missing"),
            ("measured_subprocess_benchmark", "repair_evidence_measured_benchmark_missing"),
            ("historical_activation_claims_superseded", "repair_evidence_supersession_missing"),
        ):
            if record.get(field_name) is not True:
                reasons.append(reason_code)
        # Explicitly fail closed on historical/synthetic claim markers.
        if record.get("injected") is True:
            reasons.append("repair_evidence_injected")
        if record.get("pseudo_certificate") is True:
            reasons.append("repair_evidence_pseudo_certificate")
        if record.get("synthetic_timing") is True:
            reasons.append("repair_evidence_synthetic_timing")
        if record.get("service_injection") is True:
            reasons.append("repair_evidence_service_injection")
        sealed = record.get("sealed_task_count")
        if sealed != SEALED_PRODUCTION_TASK_COUNT:
            reasons.append("repair_evidence_task_count_mismatch")
        # Historical 53-task packets cannot be re-admitted by omission tricks.
        if sealed == 53:
            reasons.append("repair_evidence_historical_53_task_population")
        if producer_task_id == "PTR-142" or repair_id == RUNTIME_ACTIVATION_REPAIR_ID:
            reasons.append("repair_evidence_historical_ptr142_inadmissible")

        requirement = _text(
            _value(
                record,
                "requirement_id",
                "acceptance_criterion",
                "satisfied_requirement",
            )
        )
        if requirement != PRODUCTION_RUNTIME_ACTIVATION_EVIDENCE_REQUIREMENT:
            reasons.append("repair_evidence_requirement_mismatch")

        reasons.extend(
            self._validate_premise_cid(record, prefix="repair_evidence")
        )
        return reasons, self._evidence_cid(record)

    def _validate_population(
        self,
        records: Iterable[Any],
        *,
        id_names: tuple[str, ...],
        required_ids: frozenset[str],
        kind: str,
        now_ms: int,
        allowed_states: frozenset[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        reasons: list[str] = []
        cids: list[str] = []
        by_id: dict[str, Mapping[str, Any]] = {}
        for raw in records:
            record = _record(raw)
            identifier = _text(_value(record, *id_names))
            if not identifier or identifier in by_id:
                reasons.append(
                    _reason(f"{kind}_duplicate_or_unidentified", identifier)
                )
                continue
            by_id[identifier] = record

        missing = sorted(required_ids - set(by_id))
        unexpected = sorted(set(by_id) - required_ids)
        reasons.extend(_reason(f"missing_{kind}", item) for item in missing)
        reasons.extend(
            _reason(f"unexpected_{kind}", item) for item in unexpected
        )
        for identifier in sorted(required_ids & set(by_id)):
            record = by_id[identifier]
            if allowed_states is not None:
                state = _text(_value(record, "status", "state")).lower()
                if state not in allowed_states:
                    reasons.append(_reason(f"open_{kind}", identifier))
            if _text(record.get("authority")).lower() != _AUTHORITATIVE:
                reasons.append(
                    _reason(f"non_authoritative_{kind}", identifier)
                )
            if not self._fresh(record, now_ms):
                reasons.append(_reason(f"stale_{kind}", identifier))
            reasons.extend(self._bindings(record, f"{kind}:{identifier}"))
            reasons.extend(
                self._validate_premise_cid(
                    record, prefix=f"{kind}:{identifier}"
                )
            )
            cids.append(self._evidence_cid(record))
        return reasons, cids

    def evaluate(
        self,
        *,
        objective_graph: Any,
        task_evidence: Iterable[Any],
        child_goal_evidence: Iterable[Any],
        adversarial_evidence: Iterable[Any],
        analyzer_health: Iterable[Any] | Mapping[str, Any],
        benchmark_evidence: Mapping[str, Any],
        rollout_evidence: Mapping[str, Any],
        supervisor_health_evidence: Mapping[str, Any] | None = None,
        repair_evidence: Mapping[str, Any] | None = None,
    ) -> ProofTestReuseCurrentTreeGateDecision:
        """Evaluate all current evidence; emit G110 + G000 evidence only on success."""

        now_ms = int(float(self.clock()) * 1000)
        reasons: list[str] = []
        task_evidence = tuple(task_evidence)
        child_goal_evidence = tuple(child_goal_evidence)
        adversarial_evidence = tuple(adversarial_evidence)
        graph = _record(objective_graph)
        graph_tasks = frozenset(
            _text(item) for item in graph.get("task_ids", ())
        )
        graph_goals = frozenset(
            _text(item) for item in graph.get("goal_ids", ())
        )
        if graph_tasks != self.required_task_ids:
            reasons.append("objective_graph_task_population_mismatch")
        expected_graph_goals = self.required_child_goal_ids | {
            ROOT_GOAL_ID,
            FINAL_GATE_GOAL_ID,
        }
        if graph_goals != expected_graph_goals:
            reasons.append("objective_graph_goal_population_mismatch")
        if _text(graph.get("objective_revision")) != self.objective_revision:
            reasons.append("objective_graph_revision_mismatch")
        reasons.extend(self._bindings(graph, "objective_graph"))

        task_reasons, task_cids = self._validate_population(
            task_evidence,
            id_names=("task_id",),
            required_ids=self.required_task_ids,
            kind="task",
            now_ms=now_ms,
            allowed_states=_CLOSED_TASK_STATES,
        )
        reasons.extend(task_reasons)

        # A skipped task has authority only through the supervisor's fresh,
        # reverified proof-cache receipt.  Executed tasks still require an
        # authoritative validation receipt CID and merge receipt.
        task_records = {
            _text(_record(item).get("task_id")): _record(item)
            for item in task_evidence
        }
        for task_id in sorted(self.required_task_ids & set(task_records)):
            record = task_records[task_id]
            if not _text(record.get("task_cid")):
                reasons.append(_reason("missing_task_cid", task_id))
            reasons.extend(self._task_provenance_reasons(record, task_id))
            disposition = _text(
                _value(
                    record,
                    "validation_disposition",
                    "disposition",
                    "outcome",
                )
            ).lower()
            receipt = record.get("validation_receipt")
            if disposition in {"skip", "skipped", "ordinary_skip"}:
                if not isinstance(receipt, ProofCachedTestValidationReceipt):
                    reasons.append(
                        _reason("ordinary_skip_not_authority", task_id)
                    )
                elif not receipt.is_completion_evidence(
                    now_ms=now_ms,
                    task_id=task_id,
                ):
                    reasons.append(
                        _reason("proof_skip_receipt_invalid", task_id)
                    )
                elif (
                    receipt.repository_forest_cid != self.repository_forest_cid
                    or receipt.git_commit_id != self.commit_id
                    or receipt.git_tree_id != self.tree_id
                    or receipt.gitlink_state_cid != self.gitlink_state_cid
                    or not receipt.gitlink_closure_complete
                    or receipt.policy_cid != self.policy_cid
                    or receipt.verifying_key_cid != self.verifying_key_cid
                    or receipt.circuit_cid != self.circuit_cid
                ):
                    reasons.append(
                        _reason("proof_skip_binding_mismatch", task_id)
                    )
            elif not _text(
                _value(record, "validation_receipt_cid", "receipt_cid")
            ):
                reasons.append(_reason("missing_validation_receipt", task_id))

        # G110 is deliberately excluded — validated via benchmark/rollout below.
        goal_reasons, goal_cids = self._validate_population(
            child_goal_evidence,
            id_names=("goal_id",),
            required_ids=self.required_child_goal_ids,
            kind="child_goal",
            now_ms=now_ms,
            allowed_states=_CLOSED_GOAL_STATES,
        )
        reasons.extend(goal_reasons)
        for raw in child_goal_evidence:
            record = _record(raw)
            if _text(record.get("goal_id")) == FINAL_GATE_GOAL_ID:
                reasons.append("child_goal_g110_self_reference")

        adversarial_reasons, adversarial_cids = self._validate_population(
            adversarial_evidence,
            id_names=("population_id", "population"),
            required_ids=self.required_adversarial_populations,
            kind="adversarial_population",
            now_ms=now_ms,
        )
        reasons.extend(adversarial_reasons)
        for raw in adversarial_evidence:
            record = _record(raw)
            identifier = _text(
                _value(record, "population_id", "population")
            )
            if record.get("passed") is not True:
                reasons.append(
                    _reason("adversarial_population_failed", identifier)
                )
            false_skips = record.get("false_skips")
            if isinstance(false_skips, bool) or false_skips != 0:
                reasons.append(_reason("false_skip_detected", identifier))

        if isinstance(analyzer_health, Mapping):
            analyzer_items = []
            for analyzer_id, value in analyzer_health.items():
                record = dict(_record(value))
                record.setdefault("analyzer_id", analyzer_id)
                analyzer_items.append(record)
        else:
            analyzer_items = list(analyzer_health)
        analyzer_reasons, analyzer_cids = self._validate_population(
            analyzer_items,
            id_names=("analyzer_id",),
            required_ids=self.required_analyzers,
            kind="analyzer",
            now_ms=now_ms,
        )
        reasons.extend(analyzer_reasons)
        for raw in analyzer_items:
            record = _record(raw)
            if record.get("healthy") is not True:
                reasons.append(
                    _reason(
                        "analyzer_unhealthy",
                        _text(record.get("analyzer_id")),
                    )
                )

        # Direct fresh G110 benchmark premise (not a G110 goal label).
        benchmark = _mapping(benchmark_evidence)
        reasons.extend(self._bindings(benchmark, "benchmark"))
        if _text(benchmark.get("authority")).lower() != _AUTHORITATIVE:
            reasons.append("benchmark_non_authoritative")
        if not self._fresh(benchmark, now_ms):
            reasons.append("benchmark_stale")
        reasons.extend(
            self._validate_premise_cid(benchmark, prefix="benchmark")
        )
        receipt = benchmark.get("receipt")
        benchmark_cid = ""
        if not isinstance(receipt, ProofReuseBenchmarkReceipt):
            reasons.append("benchmark_receipt_missing_or_malformed")
        else:
            benchmark_cid = receipt.receipt_id
            if not receipt.passed or receipt.false_admissions != 0:
                reasons.append("benchmark_failed_or_false_admission")
            try:
                if not verify_benchmark_receipt(receipt):
                    reasons.append("benchmark_not_reverified")
            except Exception:
                reasons.append("benchmark_not_reverified")

        # Direct fresh G110 rollout premise.
        rollout = _mapping(rollout_evidence)
        reasons.extend(self._bindings(rollout, "rollout"))
        if _text(rollout.get("authority")).lower() != _AUTHORITATIVE:
            reasons.append("rollout_non_authoritative")
        if not self._fresh(rollout, now_ms):
            reasons.append("rollout_stale")
        reasons.extend(self._validate_premise_cid(rollout, prefix="rollout"))
        decision = rollout.get("decision")
        promotion = rollout.get("promotion_evidence")
        rollout_cid = ""
        if not isinstance(decision, ProofReuseRolloutDecision):
            reasons.append("rollout_decision_missing_or_malformed")
        else:
            rollout_cid = decision.decision_id
            if not decision.promoted or decision.reason_codes:
                reasons.append("rollout_decision_not_promoted")
            if (
                decision.policy_id != self.rollout_policy.policy_id
                or decision.policy_revision
                != self.rollout_policy.policy_revision
            ):
                reasons.append("rollout_policy_mismatch")
        if not isinstance(promotion, ProofReusePromotionEvidence):
            reasons.append("rollout_promotion_evidence_missing_or_malformed")
        else:
            if promotion.evidence_id != _text(
                getattr(decision, "evidence_id", "")
            ):
                reasons.append("rollout_evidence_identity_mismatch")
            if (
                promotion.mutation_false_skips != 0
                or promotion.degradation_false_skips != 0
                or promotion.authority_contradictions != 0
            ):
                reasons.append("rollout_false_skip_or_authority_contradiction")
            if (
                promotion.key_health_ok is not True
                or promotion.revocation_health_ok is not True
                or promotion.all_repositories_passed is not True
            ):
                reasons.append("rollout_health_incomplete")
            if isinstance(decision, ProofReuseRolloutDecision):
                reasons.extend(
                    self._rollout_readiness_reasons(
                        decision=decision,
                        promotion=promotion,
                        now_ms=now_ms,
                    )
                )

        supervisor = _mapping(supervisor_health_evidence)
        health_reasons, supervisor_cid = self._supervisor_health_reasons(
            supervisor, now_ms=now_ms
        )
        reasons.extend(health_reasons)

        repair = _mapping(repair_evidence)
        repair_reasons, repair_cid = self._repair_evidence_reasons(
            repair, now_ms=now_ms
        )
        reasons.extend(repair_reasons)

        reasons = list(dict.fromkeys(reasons))[:_MAX_REASONS]
        if reasons:
            return ProofTestReuseCurrentTreeGateDecision(
                passed=False,
                reason_codes=tuple(reasons),
                evaluated_at_ms=now_ms,
            )

        freshness_ends: list[int] = []
        for collection in (
            task_records.values(),
            (_record(item) for item in child_goal_evidence),
            (_record(item) for item in adversarial_evidence),
            (_record(item) for item in analyzer_items),
            (benchmark, rollout, supervisor, repair),
        ):
            for record in collection:
                value = _timestamp_ms(
                    _value(record, "fresh_until_ms", "fresh_until")
                )
                if value is not None:
                    freshness_ends.append(value)

        premise_cids = tuple(
            sorted(
                {
                    *task_cids,
                    *goal_cids,
                    *adversarial_cids,
                    *analyzer_cids,
                    benchmark_cid,
                    rollout_cid,
                    supervisor_cid,
                    *( (repair_cid,) if repair_cid else () ),
                }
            )
        )
        # Validate every retained premise CID is present.
        if any(not item for item in premise_cids):
            return ProofTestReuseCurrentTreeGateDecision(
                passed=False,
                reason_codes=("incomplete_premise_cids",),
                evaluated_at_ms=now_ms,
            )

        shared_kwargs = {
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "commit_id": self.commit_id,
            "gitlink_state_cid": self.gitlink_state_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "objective_completion_tree_id": self.objective_completion_tree_id,
            "policy_cid": self.policy_cid,
            "capability_cid": self.capability_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "circuit_cid": self.circuit_cid,
            "task_evidence_cids": tuple(task_cids),
            "child_goal_evidence_cids": tuple(goal_cids),
            "adversarial_evidence_cids": tuple(adversarial_cids),
            "analyzer_evidence_cids": tuple(analyzer_cids),
            "benchmark_receipt_cid": benchmark_cid,
            "rollout_decision_cid": rollout_cid,
            "supervisor_health_cid": supervisor_cid,
            "observed_at_ms": now_ms,
            "fresh_until_ms": min(freshness_ends),
            "premise_cids": premise_cids,
            "producing_task_id": FINAL_GATE_TASK_ID,
            "producer_kind": "task",
            "producer_channel": DEFAULT_PRODUCER_CHANNEL,
        }
        final_gate_evidence = ProofTestReuseCompletionEvidence(
            **shared_kwargs,
            objective_revision=self.g110_objective_revision,
            goal_id=FINAL_GATE_GOAL_ID,
            acceptance_criterion=FINAL_GATE_ACCEPTANCE_CRITERION,
            satisfied_requirements=FINAL_GATE_SATISFIED_REQUIREMENTS,
        )
        root_evidence = ProofTestReuseCompletionEvidence(
            **shared_kwargs,
            objective_revision=self.root_objective_revision,
            goal_id=ROOT_GOAL_ID,
            acceptance_criterion=ROOT_ACCEPTANCE_CRITERION,
            satisfied_requirements=ROOT_SATISFIED_REQUIREMENTS,
        )
        return ProofTestReuseCurrentTreeGateDecision(
            passed=True,
            reason_codes=(),
            evaluated_at_ms=now_ms,
            final_gate_completion_evidence=final_gate_evidence,
            root_completion_evidence=root_evidence,
        )

    def persist_bundle(
        self,
        decision: ProofTestReuseCurrentTreeGateDecision,
        *,
        evaluate_packet: Mapping[str, Any],
        retained_premise_bytes: Mapping[str, str] | None = None,
    ) -> ProofTestReusePersistedGateBundle:
        """Build a strictly deserializable, replayable gate bundle."""

        premise_cids: list[str] = []
        if decision.passed and decision.root_completion_evidence is not None:
            premise_cids = list(decision.root_completion_evidence.premise_cids)
        retained = dict(retained_premise_bytes or {})
        for cid, text in list(retained.items()):
            data = text.encode("utf-8")
            if cid.startswith("b") and not verify_retained_bytes(cid, data):
                # Re-key retained dag-json premises under their true CID.
                try:
                    true_cid = cid_for_canonical_dag_json_bytes(data)
                except Exception:
                    continue
                retained[true_cid] = text
                if cid in retained and cid != true_cid:
                    del retained[cid]
        return ProofTestReusePersistedGateBundle(
            repository_id=self.repository_id,
            git_tree_id=self.tree_id,
            repository_forest_cid=self.repository_forest_cid,
            objective_completion_tree_id=self.objective_completion_tree_id,
            objective_revision=self.objective_revision,
            policy_cid=self.policy_cid,
            capability_cid=self.capability_cid,
            verifying_key_cid=self.verifying_key_cid,
            circuit_cid=self.circuit_cid,
            producing_task_id=FINAL_GATE_TASK_ID,
            decision=decision,
            gate_bindings={
                "commit_id": self.commit_id,
                "gitlink_state_cid": self.gitlink_state_cid,
                "g110_objective_revision": self.g110_objective_revision,
                "root_objective_revision": self.root_objective_revision,
                "required_task_ids": sorted(self.required_task_ids),
                "required_child_goal_ids": sorted(self.required_child_goal_ids),
                "required_adversarial_populations": sorted(
                    self.required_adversarial_populations
                ),
                "required_analyzers": sorted(self.required_analyzers),
                "required_supervisor_lane_ids": sorted(
                    self.required_supervisor_lane_ids
                ),
            },
            evaluate_packet=_serialize_evaluate_packet(evaluate_packet),
            premise_cids=tuple(premise_cids),
            retained_premise_bytes=retained,
        )


def build_production_runtime_activation_evidence(
    *,
    repository_id: str,
    tree_id: str,
    commit_id: str,
    gitlink_state_cid: str,
    repository_forest_cid: str,
    capability_cid: str,
    verifying_key_cid: str,
    circuit_cid: str,
    policy_cid: str,
    objective_completion_tree_id: str = "",
    observed_at_ms: int,
    fresh_until_ms: int,
    evidence_cid: str,
    false_skips: int = 0,
    activation_e2e_passed: bool = True,
    zero_injection_default_path: bool = True,
    three_repository_cold_warm: bool = True,
    real_groth16_certificate: bool = True,
    measured_subprocess_benchmark: bool = True,
    historical_activation_claims_superseded: bool = True,
    zero_false_skip_assurance: bool = True,
    passed: bool = True,
    supervisor_healthy: bool = True,
    injected: bool = False,
    pseudo_certificate: bool = False,
    synthetic_timing: bool = False,
    service_injection: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fresh PTR-149 production-runtime-activation evidence record.

    Callers must supply genuine current-tree bindings and only set the boolean
    activation claims after the corresponding no-injection e2e, real Groth16,
    and measured subprocess evidence has actually been observed.
    """

    if false_skips != 0:
        passed = False
        zero_false_skip_assurance = False
    if injected or pseudo_certificate or synthetic_timing or service_injection:
        passed = False
    record: dict[str, Any] = {
        "authority": _AUTHORITATIVE,
        "repair_id": PRODUCTION_RUNTIME_ACTIVATION_ID,
        "producer_task_id": PRODUCTION_RUNTIME_ACTIVATION_PRODUCER_TASK_ID,
        "repair_task_ids": sorted(PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS),
        "passed": bool(passed),
        "false_skips": int(false_skips),
        "zero_false_skip_assurance": bool(zero_false_skip_assurance),
        "activation_e2e_passed": bool(activation_e2e_passed),
        "zero_injection_default_path": bool(zero_injection_default_path),
        "three_repository_cold_warm": bool(three_repository_cold_warm),
        "real_groth16_certificate": bool(real_groth16_certificate),
        "measured_subprocess_benchmark": bool(measured_subprocess_benchmark),
        "historical_activation_claims_superseded": bool(
            historical_activation_claims_superseded
        ),
        "sealed_task_count": SEALED_PRODUCTION_TASK_COUNT,
        "requirement_id": PRODUCTION_RUNTIME_ACTIVATION_EVIDENCE_REQUIREMENT,
        "repository_id": repository_id,
        "tree_id": tree_id,
        "commit_id": commit_id,
        "gitlink_state_cid": gitlink_state_cid,
        "gitlink_closure_complete": True,
        "repository_forest_cid": repository_forest_cid,
        "objective_completion_tree_id": objective_completion_tree_id,
        "capability_cid": capability_cid,
        "verifying_key_cid": verifying_key_cid,
        "circuit_cid": circuit_cid,
        "policy_cid": policy_cid,
        "observed_at_ms": int(observed_at_ms),
        "fresh_until_ms": int(fresh_until_ms),
        "evidence_cid": evidence_cid,
        "supervisor_healthy": bool(supervisor_healthy),
        "injected": bool(injected),
        "pseudo_certificate": bool(pseudo_certificate),
        "synthetic_timing": bool(synthetic_timing),
        "service_injection": bool(service_injection),
    }
    if extra:
        for key, value in extra.items():
            name = str(key)
            if name in record:
                continue
            record[name] = value
    return record


__all__ = [
    "DEFAULT_PRODUCER_CHANNEL",
    "FINAL_GATE_ACCEPTANCE_CRITERION",
    "FINAL_GATE_GOAL_ID",
    "FINAL_GATE_SATISFIED_REQUIREMENTS",
    "FINAL_GATE_TASK_ID",
    "LEGACY_FINAL_GATE_TASK_ID",
    "PROOF_TEST_REUSE_COMPLETION_EVIDENCE_INTERFACE",
    "PROOF_TEST_REUSE_CURRENT_TREE_GATE_INTERFACE",
    "PROOF_TEST_REUSE_GATE_BUNDLE_INTERFACE",
    "PROOF_TEST_REUSE_PERSISTED_GATE_BUNDLE_INTERFACE",
    "PRODUCTION_RUNTIME_ACTIVATION_EVIDENCE_REQUIREMENT",
    "PRODUCTION_RUNTIME_ACTIVATION_ID",
    "PRODUCTION_RUNTIME_ACTIVATION_PRODUCER_TASK_ID",
    "PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS",
    "REQUIRED_ADVERSARIAL_POPULATIONS",
    "REQUIRED_ANALYZERS",
    "REQUIRED_CHILD_GOAL_IDS",
    "REQUIRED_GRAPH_GOAL_IDS",
    "REQUIRED_PTR_TASK_IDS",
    "REQUIRED_SUPERVISOR_LANE_COUNT",
    "REQUIRED_SUPERVISOR_LANE_IDS",
    "ROOT_ACCEPTANCE_CRITERION",
    "ROOT_EVIDENCE_REQUIREMENTS",
    "ROOT_GOAL_ID",
    "ROOT_SATISFIED_REQUIREMENTS",
    "RUNTIME_ACTIVATION_REPAIR_EVIDENCE_REQUIREMENT",
    "RUNTIME_ACTIVATION_REPAIR_ID",
    "RUNTIME_ACTIVATION_REPAIR_TASK_IDS",
    "SEALED_PRODUCTION_TASK_COUNT",
    "ProofTestReuseCompletionEvidence",
    "ProofTestReuseCurrentTreeGate",
    "ProofTestReuseCurrentTreeGateDecision",
    "ProofTestReuseCurrentTreeGateError",
    "ProofTestReusePersistedGateBundle",
    "TaskCompletionProvenanceKind",
    "build_production_runtime_activation_evidence",
    "verify_persisted_current_tree_gate_bundle",
]
