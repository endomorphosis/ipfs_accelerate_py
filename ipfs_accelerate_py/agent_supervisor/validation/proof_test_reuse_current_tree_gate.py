"""Fail-closed final authority gate for proof-backed test reuse.

The final gate is intentionally an aggregator, not another proof provider.  It
accepts only fresh, authoritative evidence already bound to the exact current
repository forest and cryptographic policy.  In particular, a pytest ``SKIP``
line, a rollout decision, or a historical benchmark is never authority by
itself.

Successful evaluation emits one content-addressed completion artifact for the
program root, ``PTR-G000``.  Failed evaluation emits reasons but no completion
evidence.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar, Final

from ...testing.proof_reuse.rollout import (
    ProofReusePromotionEvidence,
    ProofReuseRolloutDecision,
    ProofReuseRolloutPolicy,
)
from ..objectives.goal_completion import CompletionEvidence
from ..self_improvement.proof_reuse_benchmark import (
    ProofReuseBenchmarkReceipt,
    verify_benchmark_receipt,
)
from .proof_cached_test_validation import ProofCachedTestValidationReceipt

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
PROOF_TEST_REUSE_CURRENT_TREE_GATE_VERSION: Final = 1

ROOT_GOAL_ID: Final = "PTR-G000"
FINAL_GATE_GOAL_ID: Final = "PTR-G110"
FINAL_GATE_TASK_ID: Final = "PTR-102"
ROOT_ACCEPTANCE_CRITERION: Final = "ptr/cross-repository-current-tree-gate@1"
ROOT_EVIDENCE_REQUIREMENTS: Final = (
    "ptr/cross-repository-current-tree-gate@1",
    "ptr/zero-false-authoritative-skip@1",
    "ptr/warm-reuse-benchmark@1",
    "ptr/supervisor-launch-health@1",
)

# This is the sealed population in plan 46.  A caller may select a smaller
# population only by explicitly constructing a gate with ``required_task_ids``;
# production callers get this complete set.
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
    }
)
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
        "PTR-G110",
    }
)
REQUIRED_ADVERSARIAL_POPULATIONS: Final = frozenset(
    {"mutation", "storage-security-concurrency", "cross-repository"}
)
REQUIRED_ANALYZERS: Final = frozenset(
    {"static-dependency", "runtime-dependency", "reuse-eligibility"}
)

_CLOSED_TASK_STATES = frozenset({"complete", "completed", "verified_complete"})
_CLOSED_GOAL_STATES = frozenset({"verified_complete"})
_AUTHORITATIVE = "authoritative"
_MAX_REASONS = 256


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
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


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
            from datetime import datetime

            normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
            return int(datetime.fromisoformat(normalized).timestamp() * 1000)
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def _reason(prefix: str, identifier: str = "") -> str:
    clean = _text(identifier).replace(":", "_")
    return f"{prefix}:{clean}" if clean else prefix


@dataclass(frozen=True, slots=True)
class ProofTestReuseCompletionEvidence:
    """The sole authoritative completion artifact emitted by this gate."""

    __test__: ClassVar[bool] = False

    repository_id: str
    tree_id: str
    commit_id: str
    gitlink_state_cid: str
    repository_forest_cid: str
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
    observed_at_ms: int
    fresh_until_ms: int
    satisfied_requirements: tuple[str, ...] = ROOT_EVIDENCE_REQUIREMENTS
    goal_id: str = ROOT_GOAL_ID
    producing_task_id: str = FINAL_GATE_TASK_ID
    acceptance_criterion: str = ROOT_ACCEPTANCE_CRITERION
    authority: str = _AUTHORITATIVE

    def __post_init__(self) -> None:
        if self.goal_id != ROOT_GOAL_ID:
            raise ProofTestReuseCurrentTreeGateError(
                "final completion evidence may target only PTR-G000"
            )
        if self.producing_task_id != FINAL_GATE_TASK_ID:
            raise ProofTestReuseCurrentTreeGateError(
                "final completion evidence must be produced by PTR-102"
            )
        if self.authority != _AUTHORITATIVE:
            raise ProofTestReuseCurrentTreeGateError(
                "completion evidence authority must be authoritative"
            )
        if tuple(self.satisfied_requirements) != ROOT_EVIDENCE_REQUIREMENTS:
            raise ProofTestReuseCurrentTreeGateError(
                "root evidence requirements must be exact"
            )
        for name in (
            "repository_id",
            "tree_id",
            "commit_id",
            "gitlink_state_cid",
            "repository_forest_cid",
            "policy_cid",
            "capability_cid",
            "verifying_key_cid",
            "circuit_cid",
            "objective_revision",
            "benchmark_receipt_cid",
            "rollout_decision_cid",
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
        if self.observed_at_ms < 0 or self.fresh_until_ms <= self.observed_at_ms:
            raise ProofTestReuseCurrentTreeGateError(
                "completion evidence freshness window is invalid"
            )

    @property
    def evidence_id(self) -> str:
        return _content_id(self.to_dict())

    @property
    def provenance_cid(self) -> str:
        return self.evidence_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_TEST_REUSE_COMPLETION_EVIDENCE_SCHEMA,
            "interface": PROOF_TEST_REUSE_COMPLETION_EVIDENCE_INTERFACE,
            "version": PROOF_TEST_REUSE_CURRENT_TREE_GATE_VERSION,
            "goal_id": self.goal_id,
            "producing_task_id": self.producing_task_id,
            "acceptance_criterion": self.acceptance_criterion,
            "authority": self.authority,
            "satisfied_requirements": list(self.satisfied_requirements),
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "commit_id": self.commit_id,
            "gitlink_state_cid": self.gitlink_state_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "policy_cid": self.policy_cid,
            "capability_cid": self.capability_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "circuit_cid": self.circuit_cid,
            "objective_revision": self.objective_revision,
            "task_evidence_cids": list(self.task_evidence_cids),
            "child_goal_evidence_cids": list(self.child_goal_evidence_cids),
            "adversarial_evidence_cids": list(self.adversarial_evidence_cids),
            "analyzer_evidence_cids": list(self.analyzer_evidence_cids),
            "benchmark_receipt_cid": self.benchmark_receipt_cid,
            "rollout_decision_cid": self.rollout_decision_cid,
            "observed_at_ms": self.observed_at_ms,
            "fresh_until_ms": self.fresh_until_ms,
        }

    def as_completion_evidence(self) -> CompletionEvidence:
        """Project the sealed artifact into the shared objective interface."""

        return CompletionEvidence(
            acceptance_criterion=self.acceptance_criterion,
            producing_task_or_scan=self.producing_task_id,
            validation_receipt=self.to_dict(),
            repository_tree=self.tree_id,
            freshness={
                "observed_at_ms": self.observed_at_ms,
                "fresh_until_ms": self.fresh_until_ms,
            },
            provenance_cid=self.evidence_id,
            producer_kind="current_tree_authority_gate",
            repository_id=self.repository_id,
            tree_id=self.tree_id,
            objective_revision=self.objective_revision,
            validation_passed=True,
            observed_at=datetime.fromtimestamp(
                self.observed_at_ms / 1000, tz=UTC
            ),
            fresh_until=datetime.fromtimestamp(
                self.fresh_until_ms / 1000, tz=UTC
            ),
            metadata={"goal_id": ROOT_GOAL_ID, "authority": self.authority},
        )


@dataclass(frozen=True, slots=True)
class ProofTestReuseCurrentTreeGateDecision:
    """Typed outcome.  Rejection can never carry root completion evidence."""

    passed: bool
    reason_codes: tuple[str, ...]
    evaluated_at_ms: int
    completion_evidence: ProofTestReuseCompletionEvidence | None = None

    def __post_init__(self) -> None:
        reasons = tuple(dict.fromkeys(_text(item) for item in self.reason_codes))
        if any(not reason for reason in reasons) or len(reasons) > _MAX_REASONS:
            raise ProofTestReuseCurrentTreeGateError("invalid gate reason codes")
        object.__setattr__(self, "reason_codes", reasons)
        if self.passed:
            if reasons or self.completion_evidence is None:
                raise ProofTestReuseCurrentTreeGateError(
                    "passing gate requires evidence and no rejection reasons"
                )
        elif self.completion_evidence is not None:
            raise ProofTestReuseCurrentTreeGateError(
                "failed gate cannot emit completion evidence"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_TEST_REUSE_GATE_DECISION_SCHEMA,
            "interface": PROOF_TEST_REUSE_CURRENT_TREE_GATE_INTERFACE,
            "version": PROOF_TEST_REUSE_CURRENT_TREE_GATE_VERSION,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "evaluated_at_ms": self.evaluated_at_ms,
            "completion_evidence": (
                self.completion_evidence.to_dict()
                if self.completion_evidence is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ProofTestReuseCurrentTreeGate:
    """Evaluate the sealed PTR population against one current-tree identity."""

    repository_id: str
    tree_id: str
    commit_id: str
    gitlink_state_cid: str
    repository_forest_cid: str
    capability_cid: str
    verifying_key_cid: str
    circuit_cid: str
    objective_revision: str
    rollout_policy: ProofReuseRolloutPolicy
    required_task_ids: frozenset[str] = REQUIRED_PTR_TASK_IDS
    required_child_goal_ids: frozenset[str] = REQUIRED_CHILD_GOAL_IDS
    required_adversarial_populations: frozenset[str] = (
        REQUIRED_ADVERSARIAL_POPULATIONS
    )
    required_analyzers: frozenset[str] = REQUIRED_ANALYZERS
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
            "capability_cid",
            "verifying_key_cid",
            "circuit_cid",
            "objective_revision",
        ):
            if not _text(getattr(self, name)):
                raise ProofTestReuseCurrentTreeGateError(f"{name} is required")
        if not isinstance(self.rollout_policy, ProofReuseRolloutPolicy):
            raise ProofTestReuseCurrentTreeGateError(
                "rollout_policy must be ProofReuseRolloutPolicy"
            )
        for name in (
            "required_task_ids",
            "required_child_goal_ids",
            "required_adversarial_populations",
            "required_analyzers",
        ):
            values = frozenset(_text(item) for item in getattr(self, name))
            if not values or any(not item for item in values):
                raise ProofTestReuseCurrentTreeGateError(f"{name} is invalid")
            object.__setattr__(self, name, values)

    @property
    def policy_cid(self) -> str:
        return self.rollout_policy.policy_binding_id

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
        }
        for name, wanted in expected.items():
            field_aliases = {
                "tree_id": ("tree_id", "git_tree_id"),
                "commit_id": ("commit_id", "git_commit_id"),
                **aliases,
            }
            actual = _text(_value(record, *field_aliases.get(name, (name,))))
            if actual != wanted:
                reasons.append(_reason(f"{name}_mismatch", prefix))
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
        return supplied or _content_id(dict(record))

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
                reasons.append(_reason(f"{kind}_duplicate_or_unidentified", identifier))
                continue
            by_id[identifier] = record

        missing = sorted(required_ids - set(by_id))
        unexpected = sorted(set(by_id) - required_ids)
        reasons.extend(_reason(f"missing_{kind}", item) for item in missing)
        reasons.extend(_reason(f"unexpected_{kind}", item) for item in unexpected)
        for identifier in sorted(required_ids & set(by_id)):
            record = by_id[identifier]
            if allowed_states is not None:
                state = _text(_value(record, "status", "state")).lower()
                if state not in allowed_states:
                    reasons.append(_reason(f"open_{kind}", identifier))
            if _text(record.get("authority")).lower() != _AUTHORITATIVE:
                reasons.append(_reason(f"non_authoritative_{kind}", identifier))
            if not self._fresh(record, now_ms):
                reasons.append(_reason(f"stale_{kind}", identifier))
            reasons.extend(self._bindings(record, f"{kind}:{identifier}"))
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
    ) -> ProofTestReuseCurrentTreeGateDecision:
        """Evaluate all current evidence and emit root evidence only on success."""

        now_ms = int(float(self.clock()) * 1000)
        reasons: list[str] = []
        task_evidence = tuple(task_evidence)
        child_goal_evidence = tuple(child_goal_evidence)
        adversarial_evidence = tuple(adversarial_evidence)
        graph = _record(objective_graph)
        graph_tasks = frozenset(_text(item) for item in graph.get("task_ids", ()))
        graph_goals = frozenset(_text(item) for item in graph.get("goal_ids", ()))
        if graph_tasks != self.required_task_ids:
            reasons.append("objective_graph_task_population_mismatch")
        if graph_goals != self.required_child_goal_ids | {ROOT_GOAL_ID}:
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
            if not _text(record.get("merge_receipt_cid")):
                reasons.append(_reason("missing_merge_receipt", task_id))
            disposition = _text(
                _value(record, "validation_disposition", "disposition", "outcome")
            ).lower()
            receipt = record.get("validation_receipt")
            if disposition in {"skip", "skipped", "ordinary_skip"}:
                if not isinstance(receipt, ProofCachedTestValidationReceipt):
                    reasons.append(_reason("ordinary_skip_not_authority", task_id))
                elif not receipt.is_completion_evidence(
                    now_ms=now_ms,
                    task_id=task_id,
                ):
                    reasons.append(_reason("proof_skip_receipt_invalid", task_id))
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
                    reasons.append(_reason("proof_skip_binding_mismatch", task_id))
            elif not _text(_value(record, "validation_receipt_cid", "receipt_cid")):
                reasons.append(_reason("missing_validation_receipt", task_id))

        goal_reasons, goal_cids = self._validate_population(
            child_goal_evidence,
            id_names=("goal_id",),
            required_ids=self.required_child_goal_ids,
            kind="child_goal",
            now_ms=now_ms,
            allowed_states=_CLOSED_GOAL_STATES,
        )
        reasons.extend(goal_reasons)

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
            identifier = _text(_value(record, "population_id", "population"))
            if record.get("passed") is not True:
                reasons.append(_reason("adversarial_population_failed", identifier))
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
                    _reason("analyzer_unhealthy", _text(record.get("analyzer_id")))
                )

        benchmark = _mapping(benchmark_evidence)
        reasons.extend(self._bindings(benchmark, "benchmark"))
        if _text(benchmark.get("authority")).lower() != _AUTHORITATIVE:
            reasons.append("benchmark_non_authoritative")
        if not self._fresh(benchmark, now_ms):
            reasons.append("benchmark_stale")
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

        rollout = _mapping(rollout_evidence)
        reasons.extend(self._bindings(rollout, "rollout"))
        if _text(rollout.get("authority")).lower() != _AUTHORITATIVE:
            reasons.append("rollout_non_authoritative")
        if not self._fresh(rollout, now_ms):
            reasons.append("rollout_stale")
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
                or decision.policy_revision != self.rollout_policy.policy_revision
            ):
                reasons.append("rollout_policy_mismatch")
        if not isinstance(promotion, ProofReusePromotionEvidence):
            reasons.append("rollout_promotion_evidence_missing_or_malformed")
        else:
            if (
                promotion.evidence_id
                != _text(getattr(decision, "evidence_id", ""))
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
                or promotion.current_tree_gate_passed is not True
                or promotion.all_repositories_passed is not True
            ):
                reasons.append("rollout_health_incomplete")

        reasons = list(dict.fromkeys(reasons))[:_MAX_REASONS]
        if reasons:
            return ProofTestReuseCurrentTreeGateDecision(
                passed=False,
                reason_codes=tuple(reasons),
                evaluated_at_ms=now_ms,
            )

        freshness_ends = []
        for collection in (
            task_records.values(),
            (_record(item) for item in child_goal_evidence),
            (_record(item) for item in adversarial_evidence),
            (_record(item) for item in analyzer_items),
            (benchmark, rollout),
        ):
            for record in collection:
                value = _timestamp_ms(
                    _value(record, "fresh_until_ms", "fresh_until")
                )
                if value is not None:
                    freshness_ends.append(value)

        evidence = ProofTestReuseCompletionEvidence(
            repository_id=self.repository_id,
            tree_id=self.tree_id,
            commit_id=self.commit_id,
            gitlink_state_cid=self.gitlink_state_cid,
            repository_forest_cid=self.repository_forest_cid,
            policy_cid=self.policy_cid,
            capability_cid=self.capability_cid,
            verifying_key_cid=self.verifying_key_cid,
            circuit_cid=self.circuit_cid,
            objective_revision=self.objective_revision,
            task_evidence_cids=tuple(task_cids),
            child_goal_evidence_cids=tuple(goal_cids),
            adversarial_evidence_cids=tuple(adversarial_cids),
            analyzer_evidence_cids=tuple(analyzer_cids),
            benchmark_receipt_cid=benchmark_cid,
            rollout_decision_cid=rollout_cid,
            observed_at_ms=now_ms,
            fresh_until_ms=min(freshness_ends),
        )
        return ProofTestReuseCurrentTreeGateDecision(
            passed=True,
            reason_codes=(),
            evaluated_at_ms=now_ms,
            completion_evidence=evidence,
        )


__all__ = [
    "FINAL_GATE_GOAL_ID",
    "FINAL_GATE_TASK_ID",
    "PROOF_TEST_REUSE_COMPLETION_EVIDENCE_INTERFACE",
    "PROOF_TEST_REUSE_CURRENT_TREE_GATE_INTERFACE",
    "REQUIRED_ADVERSARIAL_POPULATIONS",
    "REQUIRED_ANALYZERS",
    "REQUIRED_CHILD_GOAL_IDS",
    "REQUIRED_PTR_TASK_IDS",
    "ROOT_EVIDENCE_REQUIREMENTS",
    "ROOT_GOAL_ID",
    "ProofTestReuseCompletionEvidence",
    "ProofTestReuseCurrentTreeGate",
    "ProofTestReuseCurrentTreeGateDecision",
    "ProofTestReuseCurrentTreeGateError",
]
