"""Single-consumer merge train for autonomous implementation lanes.

The queue is the hand-off boundary between implementation workers and the target
checkout.  This module deliberately owns the *only* dequeue-and-merge critical
section: requests are deduplicated, rebased on the target tip observed inside
that section, and either completed, retried a bounded number of times, or
quarantined by :class:`~.merge_queue.MergeQueue`.

The built-in merger rebases in a detached temporary worktree, then advances the
target with compare-and-swap semantics.  A checked-out target is fast-forwarded
only while its worktree is clean, so its index is never left stale.  A daemon
can supply ``merge_callback`` to reuse a more specialised merger (for example
one that coordinates nested submodules); the callback is still invoked under
the same repo-wide single-consumer lease.
"""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import marshal
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Final, Iterator, Mapping, Sequence

from ..proof.formal_verification_policy import (
    ChangedScope,
    FormalVerificationPolicy,
    InvariantClass,
    PolicySelection,
    RiskLevel,
    default_formal_verification_policy,
)
from .checkout_lock import checkout_repository_id
from .merge_queue import MergeQueue, MergeQueueFenceError, MergeRequest


MergeCallback = Callable[[MergeRequest], Mapping[str, Any]]
PreflightCallback = Callable[..., Mapping[str, Any] | bool]
PostMergeValidationCallback = Callable[..., Mapping[str, Any] | bool]
PostMergeEvidenceCallback = Callable[..., Any]

PARALLEL_ACCEPTANCE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/parallel-acceptance-receipt@1"
)
PARALLEL_ACCEPTANCE_THROUGHPUT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/acceptance-throughput@1"
)
PARALLEL_ACCEPTANCE_EVIDENCE_ID = (
    "185033715568272291470322170325431455647"
)
PARALLEL_EXECUTION_OBJECTIVE_ID: Final = "ASI-G060"
PARALLEL_EXECUTION_OBJECTIVE_REVISION: Final = "ASI-G060@asi-083"
PARALLEL_EXECUTION_COMPLETION_ANALYZER_VERSION: Final = (
    "parallel-execution-completion@1"
)
PARALLEL_EXECUTION_COMPLETION_CONFIGURATION_REVISION: Final = (
    "parallel-execution-completion-policy@1"
)
PARALLEL_EXECUTION_REQUIRED_EXHAUSTIVE_RECEIPTS: Final = 2
PARALLEL_EXECUTION_PRODUCING_TASK_IDS: Final[tuple[str, ...]] = (
    "ASI-015",
    "ASI-016",
    "ASI-017",
)
PARALLEL_EXECUTION_ACCEPTANCE_CRITERIA: Final[tuple[str, ...]] = (
    "Resource pools expose backpressure and fair admission",
    "compatible provider work shares model capacity",
    "independent validation and merge preflight run concurrently",
    "target-branch mutation remains fenced and serialized",
    (
        "paired independent fixtures achieve at least twice single-lane "
        "throughput without duplicate execution, stale acceptance, resource "
        "overcommit, or merge-conflict regression"
    ),
)
PARALLEL_GATE_CACHE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/parallel-gate-cache@1"
)
DISTRIBUTED_LANE_PUBLICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/distributed-lane-publication@1"
)
DISTRIBUTED_LANE_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/distributed-lane-admission@1"
)
_DISTRIBUTED_PUBLICATION_METADATA_KEY: Final = "distributed_publication"
_PARALLEL_ACCEPTANCE_RECEIPT_SEAL: Final = object()


@dataclass(frozen=True)
class ParallelAcceptanceReceipt:
    """Content-addressed proof of the fenced acceptance sequence.

    A successful receipt is constructed and persisted only after the
    synthesized merged tree has passed its post-merge validator and that exact
    commit has won the serialized target compare-and-swap.  The queue stores
    the receipt id when it transitions to ``completed``; therefore parallel
    preflight completion alone can never confer acceptance authority.
    """

    request_id: str
    canonical_task_id: str
    candidate_commit: str
    target_commit: str
    preflight: Mapping[str, Any]
    integration: Mapping[str, Any]
    post_merge_validation: Mapping[str, Any]
    mutation_fence_owner: str
    mutation_fence_generation: int
    mutation_fence_token_digest: str
    accepted: bool
    validation_receipt_ids: tuple[str, ...] = ()
    requirement_id: str = PARALLEL_ACCEPTANCE_EVIDENCE_ID
    schema: str = PARALLEL_ACCEPTANCE_RECEIPT_SCHEMA
    _producer_seal: object | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def _content(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "requirement_id": self.requirement_id,
            "request_id": self.request_id,
            "canonical_task_id": self.canonical_task_id,
            "candidate_commit": self.candidate_commit,
            "target_commit": self.target_commit,
            "preflight": dict(self.preflight),
            "integration": dict(self.integration),
            "post_merge_validation": dict(self.post_merge_validation),
            "validation_receipt_ids": list(self.validation_receipt_ids),
            "mutation_fence_owner": self.mutation_fence_owner,
            "mutation_fence_generation": self.mutation_fence_generation,
            "mutation_fence_token_digest": self.mutation_fence_token_digest,
            "accepted": self.accepted,
            "sequence": (
                "parallel_preflight",
                "synthesized_merged_tree",
                "post_merge_validation",
                "serialized_target_mutation",
                "queue_completion_authorized",
            ),
        }

    @property
    def receipt_id(self) -> str:
        content = json.dumps(
            self._content(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        return {"receipt_id": self.receipt_id, **self._content()}

    def verify_integrity(self) -> bool:
        """Re-derive all authority-bearing fields of this receipt."""

        validation = self.post_merge_validation
        validated_commit = str(
            validation.get("validated_commit")
            or validation.get("target_commit")
            or ""
        )
        return bool(
            self._producer_seal is _PARALLEL_ACCEPTANCE_RECEIPT_SEAL
            and self.schema == PARALLEL_ACCEPTANCE_RECEIPT_SCHEMA
            and self.requirement_id == PARALLEL_ACCEPTANCE_EVIDENCE_ID
            and self.request_id
            and self.canonical_task_id
            and self.candidate_commit
            and self.target_commit
            and self.accepted
            and validation.get("passed") is True
            and validated_commit == self.target_commit
            and self.mutation_fence_owner
            and self.mutation_fence_generation > 0
            and self.mutation_fence_token_digest.startswith("sha256:")
            and len(self.mutation_fence_token_digest) == len("sha256:") + 64
        )

    def proved_requirement_ids_for(
        self, repository_tree: str
    ) -> tuple[str, ...]:
        """Expose authority only for the exact accepted repository tree."""

        if (
            self.verify_integrity()
            and self.target_commit == str(repository_tree or "").strip()
        ):
            return (self.requirement_id,)
        return ()

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "ParallelAcceptanceReceipt":
        """Restore a diagnostic projection and reject identity tampering.

        Restored projections deliberately lack the private producer seal and
        therefore cannot expose completion authority through
        :meth:`proved_requirement_ids_for`.
        """

        if not isinstance(value, Mapping):
            raise TypeError("parallel acceptance receipt must be a mapping")
        sequence = value.get("sequence")
        expected_sequence = (
            "parallel_preflight",
            "synthesized_merged_tree",
            "post_merge_validation",
            "serialized_target_mutation",
            "queue_completion_authorized",
        )
        if sequence not in (expected_sequence, list(expected_sequence)):
            raise ValueError("parallel acceptance receipt sequence mismatch")
        receipt = cls(
            request_id=str(value.get("request_id") or ""),
            canonical_task_id=str(value.get("canonical_task_id") or ""),
            candidate_commit=str(value.get("candidate_commit") or ""),
            target_commit=str(value.get("target_commit") or ""),
            preflight=(
                dict(value["preflight"])
                if isinstance(value.get("preflight"), Mapping)
                else {}
            ),
            integration=(
                dict(value["integration"])
                if isinstance(value.get("integration"), Mapping)
                else {}
            ),
            post_merge_validation=(
                dict(value["post_merge_validation"])
                if isinstance(value.get("post_merge_validation"), Mapping)
                else {}
            ),
            validation_receipt_ids=tuple(
                str(item)
                for item in value.get("validation_receipt_ids", ())
            ),
            mutation_fence_owner=str(
                value.get("mutation_fence_owner") or ""
            ),
            mutation_fence_generation=int(
                value.get("mutation_fence_generation") or 0
            ),
            mutation_fence_token_digest=str(
                value.get("mutation_fence_token_digest") or ""
            ),
            accepted=value.get("accepted") is True,
            requirement_id=str(value.get("requirement_id") or ""),
            schema=str(value.get("schema") or ""),
        )
        if str(value.get("receipt_id") or "") != receipt.receipt_id:
            raise ValueError("parallel acceptance receipt identity mismatch")
        return receipt


def evaluate_parallel_execution_completion(
    *,
    repository_id: str,
    repository_tree: str,
    resource_policy: Any,
    operational_evidence: Sequence[Any] = (),
    producing_tasks: Sequence[Any] = (),
    current_state: Any = "active",
    evidence: Sequence[Any] = (),
    tasks_complete: bool = False,
    coverage: Any = None,
    analyzer_health: Any = None,
    exhaustion_quorum: Any = None,
    required_exhaustive_receipts: int = (
        PARALLEL_EXECUTION_REQUIRED_EXHAUSTIVE_RECEIPTS
    ),
    now: Any = None,
    freshness_seconds: float = 3600.0,
    clock_skew_seconds: float = 300.0,
    analysis_inconclusive: bool = False,
    blocked_reason: str = "",
) -> Any:
    """Evaluate the closed ASI-G060 completion boundary.

    Runtime counters and task status are deliberately insufficient.  The
    immutable parent criteria require all three live typed lane receipts, the
    exact successful producer population, one fresh passing current-tree
    validation per criterion, concrete coverage bound to those validation
    identities, an explicitly healthy completion-safe analyzer, and exactly
    the configured independent exhaustive quorum.
    """

    from ..objectives.goal_completion import evaluate_goal_completion
    from ..runtime.provider_batch_scheduler import (
        PARTIAL_CANCELLATION_REQUIREMENT_ID,
        ProviderBatchEvidenceReceipt,
    )
    from ..runtime.resource_scheduler import (
        ADAPTIVE_SCHEDULING_THROUGHPUT_REQUIREMENT_ID,
        AdaptiveThroughputBenchmarkReceipt,
    )

    if (
        isinstance(required_exhaustive_receipts, bool)
        or not isinstance(required_exhaustive_receipts, int)
        or required_exhaustive_receipts
        != PARALLEL_EXECUTION_REQUIRED_EXHAUSTIVE_RECEIPTS
    ):
        raise ValueError(
            "required_exhaustive_receipts must equal the configured "
            f"ASI-G060 count "
            f"{PARALLEL_EXECUTION_REQUIRED_EXHAUSTIVE_RECEIPTS}"
        )
    for name, value in (
        ("freshness_seconds", freshness_seconds),
        ("clock_skew_seconds", clock_skew_seconds),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) < 0
        ):
            raise ValueError(f"{name} must be a non-negative number")

    def payload(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        converter = getattr(value, "to_dict", None)
        if callable(converter):
            converted = converter()
            if isinstance(converted, Mapping):
                return dict(converted)
        return {}

    def normalized(value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def parsed_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            result = value
        elif isinstance(value, str) and value.strip():
            try:
                result = datetime.fromisoformat(
                    value.strip().replace("Z", "+00:00")
                )
            except ValueError:
                return None
        else:
            return None
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result.astimezone(timezone.utc)

    current = parsed_datetime(now) or datetime.now(timezone.utc)
    max_age = timedelta(seconds=float(freshness_seconds))
    clock_skew = timedelta(seconds=float(clock_skew_seconds))

    def fresh(value: Any) -> bool:
        observed = parsed_datetime(value)
        return bool(
            observed is not None
            and observed <= current + clock_skew
            and current - observed <= max_age
        )

    repository_id = str(repository_id or "").strip()
    repository_tree = str(repository_tree or "").strip()
    expected_binding = {
        "repository_id": repository_id,
        "tree_id": repository_tree,
        "objective_id": PARALLEL_EXECUTION_OBJECTIVE_ID,
        "objective_revision": PARALLEL_EXECUTION_OBJECTIVE_REVISION,
        "analyzer_version": (
            PARALLEL_EXECUTION_COMPLETION_ANALYZER_VERSION
        ),
        "configuration_revision": (
            PARALLEL_EXECUTION_COMPLETION_CONFIGURATION_REVISION
        ),
    }

    lane_receipts = tuple(operational_evidence)
    adaptive_receipts = tuple(
        item
        for item in lane_receipts
        if type(item) is AdaptiveThroughputBenchmarkReceipt
    )
    provider_receipts = tuple(
        item
        for item in lane_receipts
        if type(item) is ProviderBatchEvidenceReceipt
    )
    acceptance_receipts = tuple(
        item
        for item in lane_receipts
        if type(item) is ParallelAcceptanceReceipt
    )
    try:
        operational_complete = bool(
            len(lane_receipts) == 3
            and len(adaptive_receipts) == 1
            and len(provider_receipts) == 1
            and len(acceptance_receipts) == 1
            and adaptive_receipts[0].proved_requirement_ids_for(
                policy=resource_policy,
                repository_tree_id=repository_tree,
            )
            == (ADAPTIVE_SCHEDULING_THROUGHPUT_REQUIREMENT_ID,)
            and provider_receipts[0].proved_requirement_ids
            == (PARTIAL_CANCELLATION_REQUIREMENT_ID,)
            and acceptance_receipts[0].proved_requirement_ids_for(
                repository_tree
            )
            == (PARALLEL_ACCEPTANCE_EVIDENCE_ID,)
        )
    except (AttributeError, TypeError, ValueError):
        # Operational inputs are untrusted persisted evidence.  Malformed
        # policy/receipt data closes completion but does not crash evaluation.
        operational_complete = False

    successful_states = frozenset(
        {
            "complete",
            "completed",
            "passed",
            "success",
            "succeeded",
            "verified",
            "verified_complete",
        }
    )
    task_values = [payload(item) for item in producing_tasks]
    task_ids = [
        str(item.get("task_id", item.get("id", "")) or "").strip()
        for item in task_values
    ]
    producer_population_complete = bool(
        repository_id
        and repository_tree
        and len(task_ids) == len(set(task_ids))
        and tuple(sorted(task_ids))
        == tuple(sorted(PARALLEL_EXECUTION_PRODUCING_TASK_IDS))
        and all(
            normalized(item.get("status", item.get("state", "")))
            in successful_states
            for item in task_values
        )
    )

    expected_criteria = {
        normalized(item) for item in PARALLEL_EXECUTION_ACCEPTANCE_CRITERIA
    }
    receipt_ids_by_criterion: dict[str, set[str]] = {}
    evidence_criteria: list[str] = []
    for item in evidence:
        record = payload(item)
        source_value = record.get("evidence", record)
        source = (
            dict(source_value)
            if isinstance(source_value, Mapping)
            else record
        )
        criterion = normalized(
            source.get(
                "acceptance_criterion",
                source.get("criterion", source.get("acceptance", "")),
            )
        )
        evidence_criteria.append(criterion)
        receipt_id = str(
            source.get(
                "provenance_cid",
                source.get(
                    "receipt_id",
                    source.get(
                        "evidence_id", source.get("receipt_cid", "")
                    ),
                ),
            )
            or ""
        ).strip()
        if criterion and receipt_id:
            receipt_ids_by_criterion.setdefault(criterion, set()).add(
                receipt_id
            )
    evidence_population_complete = bool(
        len(evidence_criteria) == len(expected_criteria)
        and len(evidence_criteria) == len(set(evidence_criteria))
        and set(evidence_criteria) == expected_criteria
        and all(
            len(receipt_ids_by_criterion.get(criterion, set())) == 1
            for criterion in expected_criteria
        )
    )

    coverage_projection = getattr(coverage, "completion_gate_evidence", None)
    if callable(coverage_projection):
        try:
            projected = coverage_projection(PARALLEL_EXECUTION_OBJECTIVE_ID)
        except (TypeError, ValueError):
            projected = {}
        coverage_value = (
            dict(projected) if isinstance(projected, Mapping) else {}
        )
    else:
        coverage_value = payload(coverage)
    rows_value = coverage_value.get("criteria")
    rows = rows_value if isinstance(rows_value, list) else []

    def criterion_of(row: Mapping[str, Any]) -> str:
        return normalized(
            row.get(
                "criterion",
                row.get(
                    "acceptance_criterion",
                    row.get("acceptance", ""),
                ),
            )
        )

    def implementation_bound(row: Mapping[str, Any]) -> bool:
        for name in (
            "implementation",
            "implementation_binding",
            "changed_files",
            "predicted_files",
            "ast_symbols",
            "interfaces",
        ):
            value = row.get(name)
            if isinstance(value, str) and value.strip():
                return True
            if (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
                and any(str(item or "").strip() for item in value)
            ):
                return True
        return False

    def validation_ids(row: Mapping[str, Any]) -> set[str]:
        raw = row.get(
            "validation_receipt_ids",
            row.get("validation_receipt_id", ()),
        )
        if isinstance(raw, str):
            raw = (raw,)
        if not (
            isinstance(raw, Sequence)
            and not isinstance(raw, (str, bytes, bytearray))
        ):
            return set()
        return {
            str(item or "").strip()
            for item in raw
            if str(item or "").strip()
        }

    row_keys = [
        criterion_of(row) for row in rows if isinstance(row, Mapping)
    ]
    coverage_bound = bool(
        evidence_population_complete
        and coverage_value.get("verified") is True
        and coverage_value.get("repository_id") == repository_id
        and coverage_value.get("repository_tree") == repository_tree
        and len(row_keys) == len(expected_criteria)
        and len(row_keys) == len(set(row_keys))
        and set(row_keys) == expected_criteria
        and all(
            isinstance(row, Mapping)
            and implementation_bound(row)
            and len(validation_ids(row)) == 1
            and validation_ids(row)
            == receipt_ids_by_criterion.get(criterion_of(row), set())
            for row in rows
        )
    )
    if not coverage_bound:
        raw_reason_codes = coverage_value.get("reason_codes")
        prior_reason_codes = (
            list(raw_reason_codes)
            if isinstance(raw_reason_codes, (list, tuple))
            else []
        )
        coverage_value = {
            **coverage_value,
            "verified": False,
            "passed": False,
            "reason_codes": [
                *prior_reason_codes,
                (
                    "validation_evidence_population_incomplete"
                    if not evidence_population_complete
                    else "coverage_validation_receipt_unbound"
                ),
            ],
        }

    health_value = payload(analyzer_health)
    health_binding_value = health_value.get("binding")
    health_binding = (
        dict(health_binding_value)
        if isinstance(health_binding_value, Mapping)
        else {}
    )
    health_valid = bool(
        all(expected_binding.values())
        and health_binding == expected_binding
        and normalized(health_value.get("status")) == "healthy"
        and health_value.get("healthy") is True
        and health_value.get("safe_for_completion_reasoning") is True
    )
    if not health_valid:
        health_value = {
            **health_value,
            "healthy": False,
            "safe_for_completion_reasoning": False,
        }

    quorum_value = payload(exhaustion_quorum)
    members_value = quorum_value.get("members")
    members = (
        list(members_value)
        if isinstance(members_value, Sequence)
        and not isinstance(members_value, (str, bytes, bytearray))
        else []
    )
    quorum_binding_value = quorum_value.get("binding")
    quorum_binding = (
        dict(quorum_binding_value)
        if isinstance(quorum_binding_value, Mapping)
        else {}
    )

    def independent_member_field(name: str) -> bool:
        values = [
            str(member.get(name) or "").strip()
            for member in members
            if isinstance(member, Mapping)
        ]
        return bool(
            len(values) == len(members)
            and all(values)
            and len(values) == len(set(values))
        )

    quorum_valid = bool(
        quorum_value.get("required_members")
        == PARALLEL_EXECUTION_REQUIRED_EXHAUSTIVE_RECEIPTS
        and quorum_value.get("member_count") == len(members)
        and len(members)
        == PARALLEL_EXECUTION_REQUIRED_EXHAUSTIVE_RECEIPTS
        and quorum_value.get("satisfied") is True
        and quorum_value.get("quorum_met") is True
        and health_valid
        and quorum_binding == expected_binding
        and independent_member_field("member_id")
        and independent_member_field("evidence_channel")
        and independent_member_field("receipt_cid")
        and all(
            isinstance(member, Mapping)
            and member.get("healthy") is True
            and member.get("safe_for_completion_reasoning") is True
            and normalized(member.get("scan_mode")) == "exhaustive"
            and fresh(member.get("finished_at"))
            and isinstance(member.get("binding"), Mapping)
            and dict(member["binding"]) == expected_binding
            for member in members
        )
    )
    if not quorum_valid:
        quorum_value = {
            **quorum_value,
            "satisfied": False,
            "quorum_met": False,
        }

    return evaluate_goal_completion(
        current_state=current_state,
        acceptance_criteria=PARALLEL_EXECUTION_ACCEPTANCE_CRITERIA,
        evidence=evidence,
        tasks_complete=bool(
            tasks_complete
            and producer_population_complete
            and operational_complete
        ),
        repository_tree=repository_tree,
        repository_id=repository_id,
        now=current,
        freshness_seconds=freshness_seconds,
        clock_skew_seconds=clock_skew_seconds,
        coverage=coverage_value,
        analyzer_health=health_value,
        exhaustion_quorum=quorum_value,
        child_goals=(),
        analysis_result=None,
        analysis_inconclusive=analysis_inconclusive,
        blocked_reason=blocked_reason,
        require_completion_gate=True,
    )


def _request_value(request: MergeRequest, name: str, *metadata_names: str) -> str:
    """Read a request field while remaining compatible with v1 queue records."""

    value = getattr(request, name, "")
    if value:
        return str(value)
    metadata = getattr(request, "metadata", {})
    if isinstance(metadata, Mapping):
        for candidate in metadata_names:
            value = metadata.get(candidate)
            if value:
                return str(value)
    return ""


def conflict_fingerprint(
    *,
    canonical_task_id: str,
    candidate_commit: str,
    target_commit: str,
    conflict_index: str,
) -> str:
    """Return a stable identity for one concrete rebase conflict.

    ``git ls-files -u`` includes path, mode, object and stage.  Hashing it with
    both tips distinguishes genuinely changed conflicts without incorporating a
    volatile request id or retry counter.
    """

    payload = "\0".join(
        (canonical_task_id, candidate_commit, target_commit, conflict_index)
    )
    return hashlib.sha256(payload.encode("utf-8", errors="surrogateescape")).hexdigest()


class MergeTrain:
    """Consume queued merge candidates serially and durably.

    Args:
        repo_root: Any worktree belonging to the repository whose target ref is
            updated.
        queue: Persistent queue shared by all implementation lanes.
        target_branch: Local branch receiving candidates.
        resolver: Optional callable/object used for rebase conflicts.  Objects
            may expose ``resolve`` and, independently, ``acquire``/``release``
            methods compatible with :class:`MergeResolverRegistry`.
        max_attempts: Last failure count at which a request is quarantined.
        merge_callback: Optional specialised merger.  It receives the claimed
            request and returns a merge-result mapping.
        state_dir: Train receipts/lease/worktrees directory.  Defaults beneath
            the queue directory so independent processes converge on one lease.
        formal_verification_policy: Optional risk-selected policy applied before
            any callback or built-in target advancement.
        proof_gate: Evidence provider for selected proof requirements.  The
            ``proof_gate_callback`` spelling is retained as an adapter alias.
        proof_cache_dir: Durable exact-selection gate cache shared by retries.
        post_merge_evidence: Optional authoritative evidence assembler for the
            exact synthesized merged commit and Git tree.  When configured,
            its content-addressed receipt must independently grant merge and
            completion authority before the target CAS can run.
    """

    def __init__(
        self,
        repo_root: Path | str,
        queue: MergeQueue,
        *,
        target_branch: str = "main",
        resolver: Any = None,
        max_attempts: int = 3,
        merge_callback: MergeCallback | None = None,
        state_dir: Path | str | None = None,
        git_timeout_seconds: float = 600.0,
        owner_id: str | None = None,
        formal_verification_policy: FormalVerificationPolicy | Mapping[str, Any] | None = None,
        proof_gate: Callable[..., Any] | None = None,
        proof_gate_callback: Callable[..., Any] | None = None,
        proof_cache_dir: Path | str | None = None,
        preflight_callback: PreflightCallback | None = None,
        post_merge_validation: PostMergeValidationCallback | None = None,
        post_merge_validator: PostMergeValidationCallback | None = None,
        post_merge_evidence: PostMergeEvidenceCallback | None = None,
        post_merge_evidence_callback: PostMergeEvidenceCallback | None = None,
        preflight_workers: int = 1,
        parallel_workers: int | None = None,
        preflight_target_sensitive: bool = False,
        preflight_gate_id: str | None = None,
        post_merge_gate_id: str | None = None,
        reuse_gate_receipts: bool = True,
        max_worktree_disk_bytes: int = 8 * 1024 * 1024 * 1024,
        max_active_worktrees: int = 1,
        distributed_publication_required: bool = False,
        distributed_repository_id: str | None = None,
        repository_id: str | None = None,
        distributed_post_merge_evidence_required: bool = True,
        decision_runtime: Any = None,
        decision_runtime_cancellation: Any = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.queue = queue
        self.target_branch = str(target_branch or "main")
        queue_repository_id = str(
            getattr(queue, "target_repository_id", "") or ""
        ).strip()
        queue_target_branch = str(
            getattr(queue, "target_branch", "") or ""
        ).strip()
        if bool(queue_repository_id) != bool(queue_target_branch):
            raise ValueError("merge queue target binding is incomplete")
        if queue_repository_id:
            repository_id = checkout_repository_id(self.repo_root)
            if queue_repository_id != repository_id:
                raise ValueError(
                    "merge queue target repository differs from the train "
                    "repository"
                )
            if queue_target_branch != self.target_branch:
                raise ValueError(
                    "merge queue target branch differs from the train target"
                )
        self.resolver = resolver
        self.max_attempts = max(1, int(max_attempts))
        self.merge_callback = merge_callback
        self.decision_runtime = decision_runtime
        self.decision_runtime_cancellation = decision_runtime_cancellation
        self._last_merge_runtime_decision: Any = None
        self._last_merge_effect_observation: Any = None
        queue_dir = Path(getattr(queue, "queue_dir", self.repo_root / ".merge-queue"))
        self.state_dir = Path(state_dir) if state_dir is not None else queue_dir / "train"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.worktree_dir = self.state_dir / "worktrees"
        self.receipt_dir = self.state_dir / "receipts"
        self.worktree_dir.mkdir(parents=True, exist_ok=True)
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        self.consumer_lock_path = self.state_dir / "consumer.lock"
        self.distributed_publication_ledger_path = (
            self.state_dir / "distributed-publications.json"
        )
        self.git_timeout_seconds = max(1.0, float(git_timeout_seconds))
        self.owner_id = owner_id or f"merge-train:{os.getpid()}:{uuid.uuid4().hex}"
        if (
            distributed_repository_id is not None
            and repository_id is not None
            and str(distributed_repository_id).strip()
            != str(repository_id).strip()
        ):
            raise ValueError(
                "distributed_repository_id and repository_id must match"
            )
        self.distributed_publication_required = bool(
            distributed_publication_required
        )
        self.distributed_repository_id = str(
            distributed_repository_id
            if distributed_repository_id is not None
            else repository_id or ""
        ).strip()
        self.repository_id = self.distributed_repository_id
        self.distributed_post_merge_evidence_required = bool(
            distributed_post_merge_evidence_required
        )
        if (
            post_merge_validation is not None
            and post_merge_validator is not None
            and post_merge_validation is not post_merge_validator
        ):
            raise ValueError(
                "post_merge_validation and post_merge_validator must refer to "
                "the same callback"
            )
        if (
            post_merge_evidence is not None
            and post_merge_evidence_callback is not None
            and post_merge_evidence is not post_merge_evidence_callback
        ):
            raise ValueError(
                "post_merge_evidence and post_merge_evidence_callback must "
                "refer to the same callback"
            )
        workers = (
            parallel_workers
            if parallel_workers is not None
            else preflight_workers
        )
        if int(workers) <= 0:
            raise ValueError("preflight_workers must be positive")
        self.preflight_callback = preflight_callback
        self.post_merge_validation = (
            post_merge_validation or post_merge_validator
        )
        self.post_merge_validator = self.post_merge_validation
        self.post_merge_evidence = (
            post_merge_evidence or post_merge_evidence_callback
        )
        self.post_merge_evidence_callback = self.post_merge_evidence
        if (
            self.post_merge_evidence is not None
            and self.post_merge_validation is None
        ):
            raise ValueError(
                "post_merge_evidence requires post_merge_validation"
            )
        if (
            self.post_merge_evidence is not None
            and self.merge_callback is not None
        ):
            raise ValueError(
                "post_merge_evidence requires the built-in synthesized-commit "
                "CAS path; merge_callback may mutate the target before "
                "post-merge authority is established"
            )
        self.preflight_workers = int(workers)
        self.preflight_target_sensitive = bool(preflight_target_sensitive)
        self.preflight_gate_id = (
            str(preflight_gate_id).strip()
            if preflight_gate_id is not None
            else self._callback_identity(
                self.preflight_callback, fallback="git-merge-tree@1"
            )
        )
        self.post_merge_gate_id = (
            str(post_merge_gate_id).strip()
            if post_merge_gate_id is not None
            else self._callback_identity(
                self.post_merge_validation, fallback="none"
            )
        )
        if not self.preflight_gate_id:
            raise ValueError("preflight_gate_id must not be empty")
        if not self.post_merge_gate_id:
            raise ValueError("post_merge_gate_id must not be empty")
        self.reuse_gate_receipts = bool(reuse_gate_receipts)
        self.gate_cache_dir = self.state_dir / "gate-cache"
        self.gate_cache_dir.mkdir(parents=True, exist_ok=True)
        if int(max_worktree_disk_bytes) <= 0:
            raise ValueError("max_worktree_disk_bytes must be positive")
        if int(max_active_worktrees) <= 0:
            raise ValueError("max_active_worktrees must be positive")
        configured_worktree_bound = int(max_worktree_disk_bytes)
        queue_worktree_bound = getattr(
            self.queue, "max_worktree_bytes", None
        )
        self.max_worktree_disk_bytes = (
            min(configured_worktree_bound, int(queue_worktree_bound))
            if queue_worktree_bound is not None
            and int(queue_worktree_bound) > 0
            else configured_worktree_bound
        )
        self.max_active_worktrees = int(max_active_worktrees)
        self._last_throughput: dict[str, Any] = {
            "schema": PARALLEL_ACCEPTANCE_THROUGHPUT_SCHEMA,
            "lane": "validation-merge-acceptance",
            "attempted_count": 0,
            "accepted_count": 0,
            "elapsed_seconds": 0.0,
            "accepted_per_second": 0.0,
            "peak_preflight_parallelism": 0,
        }
        self._acceptance_evidence: deque[ParallelAcceptanceReceipt] = deque(
            maxlen=256
        )
        if (
            proof_gate is not None
            and proof_gate_callback is not None
            and proof_gate is not proof_gate_callback
        ):
            raise ValueError("proof_gate and proof_gate_callback must refer to the same callback")
        self.proof_gate = proof_gate or proof_gate_callback
        self.proof_gate_callback = self.proof_gate
        if formal_verification_policy is None:
            self.formal_verification_policy = (
                default_formal_verification_policy()
                if self.proof_gate is not None
                else None
            )
        elif isinstance(formal_verification_policy, FormalVerificationPolicy):
            self.formal_verification_policy = formal_verification_policy
        elif isinstance(formal_verification_policy, Mapping):
            self.formal_verification_policy = FormalVerificationPolicy.from_dict(
                formal_verification_policy
            )
        else:
            raise TypeError(
                "formal_verification_policy must be a FormalVerificationPolicy or mapping"
            )
        self.proof_cache_dir = Path(
            proof_cache_dir
            if proof_cache_dir is not None
            else self.state_dir / "proof-gate-cache"
        )
        self.proof_gate_state_dir = self.state_dir / "proof-gates"
        self.proof_gate_pin_dir = self.proof_gate_state_dir / "pins"
        self.proof_gate_attempt_dir = self.proof_gate_state_dir / "attempts"
        if self.formal_verification_policy is not None:
            self.proof_cache_dir.mkdir(parents=True, exist_ok=True)
            self.proof_gate_pin_dir.mkdir(parents=True, exist_ok=True)
            self.proof_gate_attempt_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _consumer_lease(self) -> Iterator[bool]:
        """Try to acquire the process-safe, crash-releasing consumer lease."""

        fd = os.open(self.consumer_lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        acquired = False
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                yield False
                return
            metadata = json.dumps(
                {"owner_id": self.owner_id, "pid": os.getpid(), "acquired_at": time.time()}
            ).encode("utf-8")
            os.ftruncate(fd, 0)
            os.write(fd, metadata)
            os.fsync(fd)
            yield True
        finally:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def run_once(self) -> dict[str, Any] | None:
        """Process at most one request.

        ``None`` means either the queue was empty or another train consumer owns
        the lease.  The latter can be distinguished using :meth:`status`.
        """

        with self._consumer_lease() as acquired:
            if not acquired:
                return None
            self._recover_abandoned_claims()
            self._cleanup_abandoned_worktrees()
            request = self._dequeue()
            if request is None:
                return None
            if (
                self.preflight_callback is not None
                or self.post_merge_validation is not None
                or self.post_merge_evidence is not None
            ):
                target = self._target_commit()
                preflight = self._run_preflight(request, target_commit=target)
                return self._process_after_preflight(request, preflight)
            return self._process_claimed(request)

    # Widely useful aliases for supervisors that phrase one iteration as a tick.
    process_next = run_once
    consume_once = run_once

    def drain(self, max_items: int | None = None) -> list[dict[str, Any]]:
        """Drain available work while retaining one consumer lease.

        A single lease for the full batch prevents another process from slipping
        between requests and preserves the queue's priority/age order.
        """

        if max_items is not None and int(max_items) <= 0:
            return []
        if self.preflight_workers > 1:
            return self.drain_parallel(max_items=max_items)
        results: list[dict[str, Any]] = []
        with self._consumer_lease() as acquired:
            if not acquired:
                return results
            self._recover_abandoned_claims()
            self._cleanup_abandoned_worktrees()
            while max_items is None or len(results) < int(max_items):
                request = self._dequeue()
                if request is None:
                    break
                if (
                    self.preflight_callback is not None
                    or self.post_merge_validation is not None
                    or self.post_merge_evidence is not None
                ):
                    preflight = self._run_preflight(
                        request, target_commit=self._target_commit()
                    )
                    results.append(
                        self._process_after_preflight(request, preflight)
                    )
                else:
                    results.append(self._process_claimed(request))
        return results

    run = drain

    def drain_parallel(
        self, max_items: int | None = None
    ) -> list[dict[str, Any]]:
        """Parallelize non-mutating preflight, then accept under one fence.

        Target mutation and post-merge validation intentionally remain in the
        deterministic queue order.  Parallel mode fails closed before claiming
        work unless a post-merge validator is configured.
        """

        if max_items is not None and int(max_items) <= 0:
            return []
        if self.post_merge_validation is None:
            raise RuntimeError(
                "parallel acceptance requires post_merge_validation"
            )

        batch_started = time.monotonic()
        results: list[dict[str, Any]] = []
        preflight_work_seconds = 0.0
        stale_preflight_count = 0
        cancelled_preflight_count = 0
        peak_parallelism = 0
        limit = int(max_items) if max_items is not None else None

        with self._consumer_lease() as acquired:
            if not acquired:
                return results
            self._recover_abandoned_claims()
            self._cleanup_abandoned_worktrees()
            while limit is None or len(results) < limit:
                remaining = (
                    self.preflight_workers
                    if limit is None
                    else min(self.preflight_workers, limit - len(results))
                )
                requests = self._dequeue_batch(remaining)
                if not requests:
                    break
                snapshot_target = self._target_commit()
                with ThreadPoolExecutor(
                    max_workers=self.preflight_workers,
                    thread_name_prefix="merge-preflight",
                ) as pool:
                    futures: list[
                        tuple[
                            MergeRequest,
                            Future[dict[str, Any]],
                            threading.Event,
                        ]
                    ] = []
                    for request in requests:
                        cancellation_event = threading.Event()
                        futures.append(
                            (
                                request,
                                pool.submit(
                                    self._run_preflight,
                                    request,
                                    target_commit=snapshot_target,
                                    cancellation_event=cancellation_event,
                                ),
                                cancellation_event,
                            )
                        )
                    peak_parallelism = max(
                        peak_parallelism, len(futures)
                    )
                    # Resolve in claim order so branch mutation remains
                    # deterministic even when a later preflight finishes first.
                    for index, (
                        request,
                        future,
                        cancellation_event,
                    ) in enumerate(futures):
                        try:
                            preflight = future.result()
                        except CancelledError:
                            cancelled_preflight_count += 1
                            preflight = self._run_preflight(
                                request,
                                target_commit=self._target_commit(),
                            )
                        preflight_work_seconds += float(
                            preflight.get("elapsed_seconds", 0.0) or 0.0
                        )
                        current_target = self._target_commit()
                        if (
                            bool(preflight.get("cancelled"))
                            or (
                                bool(preflight.get("target_sensitive"))
                                and str(
                                    preflight.get("target_commit") or ""
                                )
                                != current_target
                            )
                        ):
                            stale_preflight_count += 1
                            preflight = self._run_preflight(
                                request, target_commit=current_target
                            )
                            preflight_work_seconds += float(
                                preflight.get("elapsed_seconds", 0.0)
                                or 0.0
                            )
                            preflight["stale_preflight_replaced"] = True
                        result = self._process_after_preflight(
                            request, preflight
                        )
                        results.append(result)
                        if bool(result.get("accepted")):
                            # Advancing the target invalidates all speculative
                            # work in this batch.  Cooperative gates observe the
                            # token; not-yet-started futures are cancelled.  A
                            # target-insensitive completed receipt remains
                            # usable, while every target-sensitive receipt is
                            # repaired against the new tip above.
                            for (
                                _later_request,
                                later_future,
                                later_cancellation,
                            ) in futures[index + 1 :]:
                                later_cancellation.set()
                                later_future.cancel()

        elapsed = max(0.0, time.monotonic() - batch_started)
        accepted = sum(
            bool(result.get("accepted")) for result in results
        )
        self._last_throughput = {
            "schema": PARALLEL_ACCEPTANCE_THROUGHPUT_SCHEMA,
            "lane": "validation-merge-acceptance",
            "attempted_count": len(results),
            "accepted_count": accepted,
            "rejected_count": len(results) - accepted,
            "elapsed_seconds": elapsed,
            "accepted_per_second": (
                accepted / elapsed if elapsed > 0 else 0.0
            ),
            "preflight_work_seconds": preflight_work_seconds,
            "preflight_speedup": (
                preflight_work_seconds / elapsed if elapsed > 0 else 0.0
            ),
            "peak_preflight_parallelism": peak_parallelism,
            "stale_preflight_count": stale_preflight_count,
            "cancelled_preflight_count": cancelled_preflight_count,
            "mutation_parallelism": 1,
            "post_merge_gate_required": True,
            "requirement_id": PARALLEL_ACCEPTANCE_EVIDENCE_ID,
        }
        return results

    run_parallel = drain_parallel

    def _dequeue_batch(self, limit: int) -> tuple[MergeRequest, ...]:
        dequeue_many = getattr(self.queue, "dequeue_many", None)
        if callable(dequeue_many):
            return tuple(
                dequeue_many(limit, consumer_id=self.owner_id) or ()
            )
        claimed: list[MergeRequest] = []
        for _ in range(max(0, int(limit))):
            request = self._dequeue()
            if request is None:
                break
            claimed.append(request)
        return tuple(claimed)

    def _run_preflight(
        self,
        request: MergeRequest,
        *,
        target_commit: str,
        cancellation_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        candidate = _request_value(
            request,
            "commit_sha",
            "commit_sha",
            "implementation_commit",
            "commit",
        )
        binding = self._gate_cache_binding(
            kind="preflight",
            request=request,
            candidate_commit=candidate,
            target_commit=target_commit,
            gate_id=self.preflight_gate_id,
        )
        cached = self._read_gate_cache(binding)
        if cached is not None:
            return cached
        if cancellation_event is not None and cancellation_event.is_set():
            return {
                "passed": False,
                "reason": "stale_preflight_cancelled",
                "retryable": True,
                "cancelled": True,
                "target_sensitive": True,
                "kind": "cancelled",
                "target_commit": target_commit,
                "candidate_commit": candidate,
                "elapsed_seconds": max(
                    0.0, time.monotonic() - started
                ),
            }
        if self.preflight_callback is None:
            command = self._git(
                "merge-tree", "--write-tree", target_commit, candidate
            )
            payload: dict[str, Any] = {
                "passed": command.returncode == 0,
                "returncode": command.returncode,
                "merge_tree": command.stdout.strip().splitlines()[0]
                if command.returncode == 0 and command.stdout.strip()
                else "",
                "stderr": command.stderr[-4000:],
                "kind": "git_merge_tree",
                "target_sensitive": True,
            }
        else:
            try:
                raw = self._call_compatible(
                    self.preflight_callback,
                    request,
                    target_commit=target_commit,
                    candidate_commit=candidate,
                    repo_root=self.repo_root,
                    cancellation_event=cancellation_event,
                    cancel_event=cancellation_event,
                )
                payload = self._normalize_gate_result(
                    raw, default_reason="preflight_failed"
                )
            except Exception as exc:
                payload = {
                    "passed": False,
                    "reason": "preflight_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            payload.setdefault(
                "target_sensitive", self.preflight_target_sensitive
            )
            payload.setdefault("kind", "callback")
        payload.update(
            {
                "target_commit": target_commit,
                "candidate_commit": candidate,
                "elapsed_seconds": max(
                    0.0, time.monotonic() - started
                ),
            }
        )
        # A cooperative callback may finish after its base was invalidated.
        # Retaining the verdict as non-authoritative lets the ordered consumer
        # repair it on the new target without caching stale work.
        if (
            cancellation_event is not None
            and cancellation_event.is_set()
            and bool(payload.get("target_sensitive"))
        ):
            payload.update(
                {
                    "passed": False,
                    "reason": "stale_preflight_cancelled",
                    "retryable": True,
                    "cancelled": True,
                }
            )
        self._write_gate_cache(binding, payload)
        return payload

    @staticmethod
    def _normalize_gate_result(
        value: Mapping[str, Any] | bool | Any,
        *,
        default_reason: str,
    ) -> dict[str, Any]:
        if isinstance(value, bool):
            return {
                "passed": value,
                "reason": "" if value else default_reason,
            }
        if not isinstance(value, Mapping):
            return {
                "passed": False,
                "reason": default_reason,
                "error": "gate callback returned no structured verdict",
            }
        result = dict(value)
        if "passed" in result:
            passed = result.get("passed") is True
        elif "allowed" in result:
            passed = result.get("allowed") is True
        elif "succeeded" in result:
            passed = result.get("succeeded") is True
        elif "returncode" in result:
            try:
                passed = int(result.get("returncode", 1)) == 0
            except (TypeError, ValueError):
                passed = False
        else:
            passed = False
        result["passed"] = passed
        if not passed:
            result.setdefault("reason", default_reason)
        return result

    @staticmethod
    def _callback_identity(
        callback: Callable[..., Any] | None,
        *,
        fallback: str,
    ) -> str:
        """Return a deterministic gate identity suitable for exact reuse.

        An explicit ``*_gate_id`` remains the preferred production spelling
        when callback configuration can change independently of its code.  The
        derived identity intentionally includes executable bytecode, constants,
        defaults, module, and qualified name so a deployment which changes gate
        logic cannot silently reuse an older verdict.
        """

        if callback is None:
            return fallback
        underlying = getattr(callback, "__func__", callback)
        code = getattr(underlying, "__code__", None)
        payload: dict[str, Any] = {
            "module": getattr(underlying, "__module__", ""),
            "qualname": getattr(
                underlying,
                "__qualname__",
                getattr(underlying, "__name__", type(callback).__qualname__),
            ),
            "defaults": repr(getattr(underlying, "__defaults__", None)),
            "kwdefaults": repr(getattr(underlying, "__kwdefaults__", None)),
        }
        if code is not None:
            payload.update(
                {
                    "code_digest": hashlib.sha256(
                        marshal.dumps(code)
                    ).hexdigest(),
                }
            )
        else:
            payload["callable_type"] = (
                f"{type(callback).__module__}.{type(callback).__qualname__}"
            )
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        )
        return (
            f"{payload['module']}:{payload['qualname']}:"
            f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
        )

    @staticmethod
    def _validation_receipt_ids(value: Any) -> tuple[str, ...]:
        """Extract authoritative receipt identities without rewriting them."""

        found: set[str] = set()

        def visit(item: Any, *, depth: int) -> None:
            if depth > 5:
                return
            if isinstance(item, Mapping):
                receipt_id = item.get("receipt_id")
                if isinstance(receipt_id, str) and receipt_id.strip():
                    found.add(receipt_id.strip())
                direct = item.get("validation_receipt_id")
                if isinstance(direct, str) and direct.strip():
                    found.add(direct.strip())
                direct_many = item.get("validation_receipt_ids")
                if isinstance(direct_many, Sequence) and not isinstance(
                    direct_many, (str, bytes, bytearray)
                ):
                    found.update(
                        str(receipt).strip()
                        for receipt in direct_many
                        if str(receipt).strip()
                    )
                for key in (
                    "validation_dag_receipt",
                    "impact_validation_receipt",
                    "proposal_receipt",
                    "post_merge_evidence",
                    "post_merge_evidence_receipt",
                    "receipt",
                    "receipts",
                ):
                    if key in item:
                        visit(item[key], depth=depth + 1)
            elif isinstance(item, Sequence) and not isinstance(
                item, (str, bytes, bytearray)
            ):
                for nested in item:
                    visit(nested, depth=depth + 1)

        visit(value, depth=0)
        return tuple(sorted(found))

    def _repository_tree_id(
        self,
        commit: str,
        *,
        workspace: Path | None = None,
    ) -> str:
        """Return the canonical Git tree identity for one exact commit."""

        if not str(commit or "").strip():
            return ""
        tree = self._git(
            "rev-parse",
            "--verify",
            f"{str(commit).strip()}^{{tree}}",
            cwd=workspace,
        )
        if tree.returncode != 0 or not tree.stdout.strip():
            return ""
        return f"git-tree:{tree.stdout.strip()}"

    @staticmethod
    def _post_merge_evidence_payload(value: Any) -> dict[str, Any]:
        """Project a typed post-merge receipt without trusting its verdict."""

        if isinstance(value, Mapping):
            for key in (
                "post_merge_evidence_receipt",
                "post_merge_evidence",
                "receipt",
            ):
                nested = value.get(key)
                if nested is not None:
                    value = nested
                    break
        converter = getattr(value, "to_dict", None)
        if callable(converter):
            value = converter()
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _verified_post_merge_evidence(
        value: Any,
        *,
        candidate_tree_id: str,
        repository_tree_id: str,
        merge_commit: str,
        expected_repository_id: str = "",
        expected_task_id: str = "",
        expected_policy_id: str = "",
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Reconstruct and independently verify one authority receipt."""

        payload = MergeTrain._post_merge_evidence_payload(value)
        if not payload:
            return {}, ("post_merge_evidence_receipt_missing",)
        try:
            from ..analysis.code_evidence_graph import PostMergeEvidenceReceipt
            from ..planning.formal_plan_conformance import (
                evaluate_post_merge_completion_admission,
            )

            receipt = (
                value
                if isinstance(value, PostMergeEvidenceReceipt)
                else PostMergeEvidenceReceipt.from_dict(payload)
            )
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            return payload, (
                "post_merge_evidence_receipt_invalid",
                f"{type(exc).__name__}: {exc}",
            )

        # A typed object receives no special trust. Reconstruct its canonical
        # projection and independently replay completion admission.
        reconstructed = receipt.to_dict()
        canonical_reconstructed = json.dumps(
            reconstructed,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        canonical_payload = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        if canonical_reconstructed != canonical_payload:
            return payload, ("post_merge_evidence_receipt_noncanonical",)

        try:
            admission = evaluate_post_merge_completion_admission(
                receipt,
                current_repository_tree_id=repository_tree_id,
                expected_repository_id=expected_repository_id,
                expected_merge_commit_id=merge_commit,
            )
        except (TypeError, ValueError) as exc:
            return payload, (
                "post_merge_evidence_verification_failed",
                f"{type(exc).__name__}: {exc}",
            )

        failures = list(admission.reason_codes)
        if admission.admitted is not True and not failures:
            failures.append("post_merge_evidence_admission_rejected")
        if receipt.merge_authoritative is not True:
            failures.append("post_merge_evidence_merge_not_authoritative")
        if str(receipt.candidate_tree_id or "") != candidate_tree_id:
            failures.append("post_merge_evidence_candidate_tree_mismatch")
        if expected_task_id and str(receipt.task_id or "") != expected_task_id:
            failures.append("post_merge_evidence_task_mismatch")
        if (
            expected_policy_id
            and str(receipt.policy_id or "") != expected_policy_id
        ):
            failures.append("post_merge_evidence_policy_mismatch")
        return payload, tuple(dict.fromkeys(failures))

    def _assemble_post_merge_evidence(
        self,
        *,
        request: MergeRequest,
        workspace: Path,
        candidate_commit: str,
        synthesized_commit: str,
        target_commit_before: str,
        repository_tree_id: str,
        validation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build and verify evidence for the exact commit proposed to the CAS."""

        callback = self.post_merge_evidence
        if callback is None:
            return {}
        producer = callback if callable(callback) else getattr(
            callback, "assemble", None
        )
        if not callable(producer):
            return {
                "passed": False,
                "reason": "post_merge_evidence_callback_invalid",
            }
        candidate_tree_id = self._repository_tree_id(
            candidate_commit,
            workspace=workspace,
        )
        if not candidate_tree_id:
            return {
                "passed": False,
                "reason": "post_merge_candidate_tree_missing",
            }
        repository_id = _request_value(
            request, "repository_id", "repository_id"
        )
        task_id = _request_value(request, "task_id")
        policy_id = _request_value(request, "policy_id", "policy_id")
        try:
            raw = self._call_compatible(
                producer,
                request,
                workspace=workspace,
                repo_root=self.repo_root,
                repository_id=repository_id,
                task_id=task_id,
                policy_id=policy_id,
                candidate_commit=candidate_commit,
                candidate_tree_id=candidate_tree_id,
                merge_commit=synthesized_commit,
                merge_commit_id=synthesized_commit,
                merged_commit=synthesized_commit,
                synthesized_commit=synthesized_commit,
                target_commit=synthesized_commit,
                target_commit_before=target_commit_before,
                repository_tree_id=repository_tree_id,
                merged_tree_id=repository_tree_id,
                current_repository_tree_id=repository_tree_id,
                post_merge_validation=dict(validation),
                validation_report=dict(validation),
            )
        except Exception as exc:
            return {
                "passed": False,
                "reason": "post_merge_evidence_exception",
                "error": f"{type(exc).__name__}: {exc}",
            }
        payload, failures = self._verified_post_merge_evidence(
            raw,
            candidate_tree_id=candidate_tree_id,
            repository_tree_id=repository_tree_id,
            merge_commit=synthesized_commit,
            expected_repository_id=repository_id,
            expected_task_id=task_id,
            expected_policy_id=policy_id,
        )
        return {
            "passed": not failures,
            "reason": "" if not failures else failures[0],
            "reason_codes": list(failures),
            "receipt": payload,
            "receipt_id": str(payload.get("receipt_id") or ""),
            "repository_tree_id": repository_tree_id,
            "merge_commit": synthesized_commit,
        }

    def _gate_cache_binding(
        self,
        *,
        kind: str,
        request: MergeRequest,
        candidate_commit: str,
        target_commit: str,
        gate_id: str,
    ) -> dict[str, str]:
        return {
            "kind": kind,
            "request_id": request.request_id,
            "candidate_commit": candidate_commit,
            "target_commit": target_commit,
            "gate_id": gate_id,
        }

    @staticmethod
    def _gate_cache_digest(payload: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _gate_cache_path(self, binding: Mapping[str, str]) -> Path:
        return self.gate_cache_dir / (
            f"{binding['kind']}-{self._gate_cache_digest(binding)}.json"
        )

    def _read_gate_cache(
        self,
        binding: Mapping[str, str],
    ) -> dict[str, Any] | None:
        if not self.reuse_gate_receipts:
            return None
        try:
            record = json.loads(
                self._gate_cache_path(binding).read_text(encoding="utf-8")
            )
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        if not isinstance(record, Mapping):
            return None
        content = {
            "schema": record.get("schema"),
            "binding": record.get("binding"),
            "result": record.get("result"),
        }
        if (
            content["schema"] != PARALLEL_GATE_CACHE_SCHEMA
            or content["binding"] != dict(binding)
            or str(record.get("record_id") or "")
            != f"sha256:{self._gate_cache_digest(content)}"
            or not isinstance(content["result"], Mapping)
        ):
            return None
        result = dict(content["result"])
        # Negative, partial, and malformed receipts never acquire authority.
        if result.get("passed") is not True:
            return None
        return result

    def _write_gate_cache(
        self,
        binding: Mapping[str, str],
        result: Mapping[str, Any],
    ) -> None:
        if not self.reuse_gate_receipts or result.get("passed") is not True:
            return
        content = {
            "schema": PARALLEL_GATE_CACHE_SCHEMA,
            "binding": dict(binding),
            "result": dict(result),
        }
        self._atomic_json(
            self._gate_cache_path(binding),
            {
                **content,
                "record_id": f"sha256:{self._gate_cache_digest(content)}",
            },
        )

    @staticmethod
    def distributed_publication_digest(
        publication: Mapping[str, Any],
    ) -> str:
        """Return the canonical identity of an untrusted lane publication.

        The producer signs/seals the same projection: ``digest`` is omitted
        from the hashed content so the identity is stable when embedded in the
        envelope.  JSON's non-finite number extension is disabled because it
        is not portable across receipt implementations.
        """

        if not isinstance(publication, Mapping):
            raise TypeError("distributed publication must be a mapping")

        def validate(item: Any) -> None:
            if item is None or isinstance(item, (str, bool, int)):
                return
            if isinstance(item, float):
                raise ValueError(
                    "distributed publication cannot contain floats"
                )
            if isinstance(item, list):
                for child in item:
                    validate(child)
                return
            if isinstance(item, Mapping):
                if not all(isinstance(key, str) for key in item):
                    raise ValueError(
                        "distributed publication keys must be strings"
                    )
                for child in item.values():
                    validate(child)
                return
            raise ValueError(
                "unsupported distributed publication value: "
                f"{type(item).__name__}"
            )

        content = {
            key: value
            for key, value in publication.items()
            if key != "digest"
        }
        validate(content)
        canonical = json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return (
            "sha256:"
            + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        )

    def _read_distributed_publication_ledger(
        self,
    ) -> tuple[dict[str, Any], str]:
        try:
            value = json.loads(
                self.distributed_publication_ledger_path.read_text(
                    encoding="utf-8"
                )
            )
        except FileNotFoundError:
            return {
                "schema": DISTRIBUTED_LANE_ADMISSION_SCHEMA,
                "publications": {},
                "tasks": {},
            }, ""
        except (OSError, json.JSONDecodeError) as exc:
            return {}, f"{type(exc).__name__}: {exc}"
        if (
            not isinstance(value, Mapping)
            or value.get("schema") != DISTRIBUTED_LANE_ADMISSION_SCHEMA
            or not isinstance(value.get("publications"), Mapping)
            or not isinstance(value.get("tasks"), Mapping)
        ):
            return {}, "distributed publication ledger is malformed"
        return {
            "schema": DISTRIBUTED_LANE_ADMISSION_SCHEMA,
            "publications": dict(value["publications"]),
            "tasks": dict(value["tasks"]),
        }, ""

    @staticmethod
    def _positive_publication_integer(
        value: Any,
        *,
        name: str,
        required: bool = True,
    ) -> tuple[int, str]:
        if value in (None, "") and not required:
            return 0, ""
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            return 0, f"distributed_publication_{name}_invalid"
        return int(value), ""

    def _reject_distributed_publication(
        self,
        request: MergeRequest,
        *,
        reason: str,
        publication: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
        cancelled: bool = False,
    ) -> dict[str, Any]:
        """Persist one terminal admission decision without escaping a fence."""

        result: dict[str, Any] = {
            "schema": DISTRIBUTED_LANE_ADMISSION_SCHEMA,
            "status": "cancelled" if cancelled else "quarantined",
            "admitted": False,
            "accepted": False,
            "integrated": False,
            "merged": False,
            "request_id": request.request_id,
            "task_id": _request_value(request, "task_id"),
            "canonical_task_id": (
                str(getattr(request, "canonical_identity", "") or "")
                or _request_value(request, "canonical_task_id")
                or _request_value(request, "task_id")
            ),
            "commit_sha": _request_value(
                request,
                "commit_sha",
                "implementation_commit",
                "commit",
            ),
            "reason": str(reason),
            "distributed_publication": dict(publication or {}),
            "finished_at": time.time(),
            **dict(details or {}),
        }
        stored = getattr(self.queue, "get", lambda _request_id: None)(
            request.request_id
        )
        pending_unclaimed = bool(
            stored is not None
            and str(getattr(stored, "status", "")) == "pending"
            and not str(getattr(request, "claim_token", "") or "")
        )
        if not self._owns_claim(request) and not pending_unclaimed:
            if (
                stored is not None
                and str(getattr(stored, "status", "")) == "completed"
            ):
                result.update(
                    {
                        "status": "duplicate",
                        "duplicate": True,
                        "accepted": True,
                        "reason": "distributed_publication_already_completed",
                    }
                )
            else:
                result.update(
                    {
                        "status": "fenced_out",
                        "reason": "merge_queue_claim_fenced",
                        "fence_stage": "distributed_publication_admission",
                    }
                )
            self._write_receipt(
                f"distributed-{result['status']}-{request.request_id}",
                result,
            )
            return result
        try:
            if cancelled:
                cancel = getattr(self.queue, "cancel", None)
                if callable(cancel):
                    cancel(
                        request,
                        reason=reason,
                        metadata=result,
                    )
                else:
                    self._call_queue_failure(
                        self.queue.fail,
                        request,
                        reason,
                        result,
                        retryable=False,
                    )
            else:
                quarantine = getattr(self.queue, "quarantine", None)
                if callable(quarantine):
                    self._call_queue_failure(
                        quarantine,
                        request,
                        reason,
                        result,
                    )
                else:
                    self._call_queue_failure(
                        self.queue.fail,
                        request,
                        reason,
                        result,
                        retryable=False,
                    )
        except MergeQueueFenceError as exc:
            result.update(
                {
                    "status": "fenced_out",
                    "reason": "merge_queue_claim_fenced",
                    "fence_error": f"{type(exc).__name__}: {exc}",
                    "fence_stage": "distributed_publication_terminal",
                }
            )
        self._write_receipt(
            f"distributed-{result['status']}-{request.request_id}",
            result,
        )
        return result

    def admit_distributed_publication(
        self,
        request: MergeRequest,
    ) -> dict[str, Any]:
        """Validate and durably fence an optional remote result publication.

        Queue claim fencing protects the local consumer.  This second boundary
        protects it from the producer: repository/task/commit bindings,
        immutable input and environment receipts, publication content identity,
        and the remote lease epoch/token must all agree before any candidate is
        allowed to reach integration.
        """

        metadata = getattr(request, "metadata", {})
        metadata = metadata if isinstance(metadata, Mapping) else {}
        raw = metadata.get(_DISTRIBUTED_PUBLICATION_METADATA_KEY)
        if raw is None:
            if not self.distributed_publication_required:
                return {
                    "schema": DISTRIBUTED_LANE_ADMISSION_SCHEMA,
                    "status": "local",
                    "admitted": True,
                    "distributed": False,
                    "request_id": request.request_id,
                }
            return self._reject_distributed_publication(
                request,
                reason="distributed_publication_missing",
            )
        if not isinstance(raw, Mapping):
            return self._reject_distributed_publication(
                request,
                reason="distributed_publication_malformed",
            )
        publication = dict(raw)
        if publication.get("schema") != DISTRIBUTED_LANE_PUBLICATION_SCHEMA:
            return self._reject_distributed_publication(
                request,
                reason="distributed_publication_schema_invalid",
                publication=publication,
            )

        canonical = (
            str(getattr(request, "canonical_identity", "") or "")
            or _request_value(request, "canonical_task_id")
            or _request_value(request, "task_id")
        )
        candidate = _request_value(
            request,
            "commit_sha",
            "implementation_commit",
            "commit",
        )
        publication_id = str(
            publication.get("publication_id")
            or publication.get("request_id")
            or ""
        ).strip()
        repository_id = str(
            publication.get("repository_id") or ""
        ).strip()
        task_cid = str(
            publication.get("task_cid")
            or publication.get("canonical_task_id")
            or ""
        ).strip()
        artifact_id = str(
            publication.get("artifact_id")
            or publication.get("input_artifact_id")
            or ""
        ).strip()
        worker_id = str(publication.get("worker_id") or "").strip()
        claimant = str(publication.get("claimant") or "").strip()
        claim_cid = str(publication.get("claim_cid") or "").strip()
        published_candidate = str(
            publication.get("candidate_commit") or ""
        ).strip()
        capability_receipt_id = str(
            publication.get("capability_receipt_id")
            or publication.get("capability_receipt_cid")
            or ""
        ).strip()
        environment_receipt_id = str(
            publication.get("environment_receipt_id")
            or publication.get("environment_receipt_cid")
            or ""
        ).strip()

        required_text = {
            "publication_id": publication_id,
            "repository_id": repository_id,
            "task_cid": task_cid,
            "artifact_id": artifact_id,
            "worker_id": worker_id,
            "claim_cid": claim_cid,
            "candidate_commit": published_candidate,
            "capability_receipt_id": capability_receipt_id,
            "environment_receipt_id": environment_receipt_id,
        }
        missing = sorted(
            name for name, value in required_text.items() if not value
        )
        if missing:
            return self._reject_distributed_publication(
                request,
                reason="distributed_publication_malformed",
                publication=publication,
                details={"missing_fields": missing},
            )

        logical_epoch, epoch_error = self._positive_publication_integer(
            publication.get(
                "logical_epoch",
                publication.get("fencing_epoch"),
            ),
            name="logical_epoch",
        )
        fencing_epoch, fencing_epoch_error = (
            self._positive_publication_integer(
                publication.get("fencing_epoch"),
                name="fencing_epoch",
                required=False,
            )
        )
        fencing_token, token_error = self._positive_publication_integer(
            publication.get("fencing_token"),
            name="fencing_token",
        )
        integer_error = (
            epoch_error or fencing_epoch_error or token_error
        )
        if integer_error:
            return self._reject_distributed_publication(
                request,
                reason=integer_error,
                publication=publication,
            )

        expected_repository_id = (
            self.distributed_repository_id
            or str(metadata.get("repository_id") or "").strip()
        )
        binding_failures: list[str] = []
        if (
            expected_repository_id
            and repository_id != expected_repository_id
        ):
            binding_failures.append(
                "distributed_publication_repository_mismatch"
            )
        if task_cid != canonical:
            binding_failures.append(
                "distributed_publication_task_mismatch"
            )
        if published_candidate != candidate:
            binding_failures.append(
                "distributed_publication_candidate_mismatch"
            )
        if binding_failures:
            return self._reject_distributed_publication(
                request,
                reason=binding_failures[0],
                publication=publication,
                details={"reason_codes": binding_failures},
            )

        supplied_digest = str(publication.get("digest") or "").strip()
        try:
            derived_digest = self.distributed_publication_digest(publication)
        except (TypeError, ValueError) as exc:
            return self._reject_distributed_publication(
                request,
                reason="distributed_publication_malformed",
                publication=publication,
                details={"digest_error": f"{type(exc).__name__}: {exc}"},
            )
        if supplied_digest != derived_digest:
            return self._reject_distributed_publication(
                request,
                reason="distributed_publication_digest_mismatch",
                publication=publication,
                details={
                    "supplied_digest": supplied_digest,
                    "derived_digest": derived_digest,
                },
            )

        if publication.get("cancelled") is True:
            return self._reject_distributed_publication(
                request,
                reason="distributed_publication_cancelled",
                publication=publication,
                cancelled=True,
            )
        if publication.get("cancelled") not in (False, None):
            return self._reject_distributed_publication(
                request,
                reason="distributed_publication_cancelled_invalid",
                publication=publication,
            )
        if self.post_merge_validation is None:
            return self._reject_distributed_publication(
                request,
                reason="distributed_post_merge_validation_required",
                publication=publication,
            )
        if (
            self.distributed_post_merge_evidence_required
            and self.post_merge_evidence is None
        ):
            return self._reject_distributed_publication(
                request,
                reason="distributed_post_merge_evidence_required",
                publication=publication,
            )
        if not self._owns_claim(request):
            return self._reject_distributed_publication(
                request,
                reason="merge_queue_claim_fenced",
                publication=publication,
            )

        ledger, ledger_error = self._read_distributed_publication_ledger()
        if ledger_error:
            return self._reject_distributed_publication(
                request,
                reason="distributed_publication_ledger_malformed",
                publication=publication,
                details={"ledger_error": ledger_error},
            )
        publications = ledger["publications"]
        tasks = ledger["tasks"]
        prior_publication = publications.get(publication_id)
        if prior_publication is not None:
            if (
                not isinstance(prior_publication, Mapping)
                or str(prior_publication.get("digest") or "")
                != derived_digest
            ):
                return self._reject_distributed_publication(
                    request,
                    reason="distributed_publication_id_conflict",
                    publication=publication,
                )
            return {
                "schema": DISTRIBUTED_LANE_ADMISSION_SCHEMA,
                "status": "duplicate",
                "admitted": True,
                "distributed": True,
                "duplicate": True,
                "request_id": request.request_id,
                "publication_id": publication_id,
                "publication_digest": derived_digest,
                "fencing_token": fencing_token,
                "logical_epoch": logical_epoch,
                "fencing_epoch": fencing_epoch,
            }

        prior_task = tasks.get(task_cid)
        if prior_task is not None and not isinstance(
            prior_task, Mapping
        ):
            return self._reject_distributed_publication(
                request,
                reason="distributed_publication_ledger_malformed",
                publication=publication,
            )
        if isinstance(prior_task, Mapping):
            prior_token = int(prior_task.get("fencing_token") or 0)
            prior_epoch = int(prior_task.get("logical_epoch") or 0)
            prior_claim = str(prior_task.get("claim_cid") or "")
            if (
                fencing_token < prior_token
                or logical_epoch < prior_epoch
            ):
                return self._reject_distributed_publication(
                    request,
                    reason="distributed_publication_stale_fence",
                    publication=publication,
                    details={
                        "current_fencing_token": prior_token,
                        "current_logical_epoch": prior_epoch,
                    },
                )
            if (
                fencing_token == prior_token
                and logical_epoch == prior_epoch
                and (
                    prior_claim != claim_cid
                    or str(prior_task.get("publication_id") or "")
                    != publication_id
                )
            ):
                return self._reject_distributed_publication(
                    request,
                    reason="distributed_publication_foreign_claim",
                    publication=publication,
                    details={
                        "current_claim_cid": prior_claim,
                        "current_publication_id": str(
                            prior_task.get("publication_id") or ""
                        ),
                    },
                )

        admitted_at = time.time()
        publications[publication_id] = {
            "digest": derived_digest,
            "request_id": request.request_id,
            "task_cid": task_cid,
            "artifact_id": artifact_id,
            "candidate_commit": published_candidate,
            "worker_id": worker_id,
            "claimant": claimant,
            "claim_cid": claim_cid,
            "logical_epoch": logical_epoch,
            "fencing_epoch": fencing_epoch,
            "fencing_token": fencing_token,
            "admitted_at": admitted_at,
        }
        tasks[task_cid] = {
            "publication_id": publication_id,
            "digest": derived_digest,
            "claim_cid": claim_cid,
            "logical_epoch": logical_epoch,
            "fencing_epoch": fencing_epoch,
            "fencing_token": fencing_token,
        }
        self._atomic_json(
            self.distributed_publication_ledger_path,
            ledger,
        )
        admission = {
            "schema": DISTRIBUTED_LANE_ADMISSION_SCHEMA,
            "status": "admitted",
            "admitted": True,
            "distributed": True,
            "duplicate": False,
            "request_id": request.request_id,
            "publication_id": publication_id,
            "publication_digest": derived_digest,
            "repository_id": repository_id,
            "task_cid": task_cid,
            "artifact_id": artifact_id,
            "worker_id": worker_id,
            "claim_cid": claim_cid,
            "logical_epoch": logical_epoch,
            "fencing_epoch": fencing_epoch,
            "fencing_token": fencing_token,
            "admitted_at": admitted_at,
        }
        self._write_receipt(
            f"distributed-admission-{publication_id}",
            admission,
        )
        return admission

    # Adapter spelling used by remote dispatchers.
    submit_distributed_result = admit_distributed_publication

    def _process_after_preflight(
        self,
        request: MergeRequest,
        preflight: Mapping[str, Any],
    ) -> dict[str, Any]:
        started_at = time.time()
        publication_admission = self.admit_distributed_publication(request)
        if not bool(publication_admission.get("admitted")):
            return publication_admission
        if not bool(preflight.get("passed")):
            return self._finish_failure(
                request,
                reason=str(
                    preflight.get("reason") or "preflight_failed"
                ),
                details={"preflight": dict(preflight)},
                started_at=started_at,
                retryable=bool(preflight.get("retryable", True)),
            )

        integration = self._process_claimed(
            request,
            defer_completion=True,
            preflight_receipt=preflight,
            publication_admission=publication_admission,
        )
        if not bool(integration.get("integrated")):
            return integration
        # A specialised merge callback owns target mutation.  Once it reports
        # that mutation as landed, absent or rejected acceptance evidence is a
        # reconciliation concern, not permission to execute the mutation a
        # second time.  The callback path settles that integration unit under
        # its queue fence and returns this explicit terminal/pending state.
        if (
            integration.get("callback_owned_integration") is True
            and str(integration.get("status") or "").startswith(
                "integrated_pending_"
            )
        ):
            return integration
        return self._post_merge_accept(
            request,
            preflight=dict(preflight),
            integration=integration,
            started_at=started_at,
        )

    def _post_merge_accept(
        self,
        request: MergeRequest,
        *,
        preflight: Mapping[str, Any],
        integration: dict[str, Any],
        started_at: float,
    ) -> dict[str, Any]:
        canonical = str(integration.get("canonical_task_id") or "")
        candidate = str(integration.get("commit_sha") or "")
        target = str(integration.get("target_commit") or "")
        callback_owned = (
            integration.get("callback_owned_integration") is True
        )
        validation_value = integration.get("post_merge_validation")
        if not isinstance(validation_value, Mapping):
            merge_result = integration.get("merge_result")
            if isinstance(merge_result, Mapping):
                validation_value = merge_result.get(
                    "post_merge_validation"
                )
        if isinstance(validation_value, Mapping):
            validation = self._normalize_gate_result(
                validation_value,
                default_reason="post_merge_validation_failed",
            )
        elif str(integration.get("status") or "") in {
            "already_merged",
            "deduplicated",
        }:
            validation = self._validate_existing_integrated_commit(
                request,
                commit=target,
                candidate_commit=candidate,
            )
        else:
            validation = {
                "passed": False,
                "reason": "post_merge_validation_receipt_missing",
            }
        validated_commit = str(
            validation.get("validated_commit")
            or validation.get("target_commit")
            or ""
        )
        if validated_commit != target and (
            validation.get("passed") is True or not callback_owned
        ):
            validation.update(
                {
                    "passed": False,
                    "reason": "post_merge_validation_target_mismatch",
                    "validated_commit": validated_commit,
                    "synthesized_commit": target,
                }
            )
        live_target = self._target_commit()
        if live_target != target:
            validation.update(
                {
                    "passed": False,
                    "reason": "post_merge_target_changed",
                    "validated_commit": validated_commit,
                    "synthesized_commit": target,
                    "current_target_commit": live_target,
                    "retryable": True,
                }
            )
        if (
            validation.get("passed") is True
            and self.post_merge_evidence is not None
        ):
            current_tree_id = self._repository_tree_id(live_target)
            evidence_payload, evidence_failures = (
                self._verified_post_merge_evidence(
                    validation.get("post_merge_evidence_receipt"),
                    candidate_tree_id=self._repository_tree_id(candidate),
                    repository_tree_id=current_tree_id,
                    merge_commit=live_target,
                    expected_repository_id=_request_value(
                        request, "repository_id", "repository_id"
                    ),
                    expected_task_id=_request_value(request, "task_id"),
                    expected_policy_id=_request_value(
                        request, "policy_id", "policy_id"
                    ),
                )
            )
            validation["post_merge_evidence_receipt"] = evidence_payload
            if evidence_failures:
                validation.update(
                    {
                        "passed": False,
                        "reason": evidence_failures[0],
                        "reason_codes": list(evidence_failures),
                        "current_repository_tree_id": current_tree_id,
                        "current_target_commit": live_target,
                        "retryable": True,
                    }
                )
        claim_current = self._owns_claim(request)
        receipt = ParallelAcceptanceReceipt(
            request_id=request.request_id,
            canonical_task_id=canonical,
            candidate_commit=candidate,
            target_commit=target,
            preflight=preflight,
            integration={
                key: value
                for key, value in integration.items()
                if key not in {"preflight", "acceptance_receipt"}
            },
            post_merge_validation=validation,
            mutation_fence_owner=str(
                getattr(request, "consumer_id", "") or self.owner_id
            ),
            mutation_fence_generation=int(
                getattr(request, "claim_generation", 0) or 0
            ),
            mutation_fence_token_digest=(
                f"sha256:{hashlib.sha256(str(getattr(request, 'claim_token', '') or '').encode('utf-8')).hexdigest()}"
                if str(getattr(request, "claim_token", "") or "")
                else ""
            ),
            accepted=bool(validation.get("passed")) and claim_current,
            validation_receipt_ids=self._validation_receipt_ids(validation),
            _producer_seal=_PARALLEL_ACCEPTANCE_RECEIPT_SEAL,
        )
        receipt_payload = receipt.to_dict()
        self._write_acceptance_receipt(receipt_payload)

        if bool(validation.get("passed")) and not claim_current:
            fenced = {
                **integration,
                "status": (
                    "integrated_pending_acceptance"
                    if callback_owned
                    else "fenced_out"
                ),
                "accepted": False,
                "acceptance_pending": callback_owned,
                "reason": "merge_queue_claim_fenced",
                "fence_stage": "before_queue_completion",
                "post_merge_validation": validation,
                "acceptance_receipt": receipt_payload,
                "finished_at": time.time(),
            }
            if callback_owned:
                fenced.update(
                    {
                        "integrated": True,
                        "completion_authoritative": False,
                        "integration_terminal": True,
                        "queue_settlement": {
                            "status": "fenced_out",
                            "terminal": False,
                            "fence_stage": "before_queue_completion",
                        },
                    }
                )
            self._write_receipt(
                f"fenced-{request.request_id}", fenced
            )
            if callback_owned:
                self._write_receipt(
                    self._dedupe_key(canonical, candidate), fenced
                )
            return fenced

        if not bool(validation.get("passed")):
            if callback_owned:
                return self._finish_integrated_pending_validation(
                    request,
                    canonical=canonical,
                    candidate=candidate,
                    target=target,
                    started_at=started_at,
                    merged=bool(integration.get("merged")),
                    already_merged=bool(
                        integration.get("already_merged")
                    ),
                    validation_reason=str(
                        validation.get("reason")
                        or "post_merge_validation_failed"
                    ),
                    post_merge_validation=validation,
                    extra={
                        **integration,
                        "acceptance_receipt": receipt_payload,
                    },
                    preflight_receipt=preflight,
                )
            return self._finish_failure(
                request,
                reason=str(
                    validation.get("reason")
                    or "post_merge_validation_failed"
                ),
                details={
                    "preflight": dict(preflight),
                    "integration": dict(integration),
                    "post_merge_validation": validation,
                    "acceptance_receipt": receipt_payload,
                },
                started_at=started_at,
                retryable=bool(validation.get("retryable", False)),
            )

        # The evidence receipt is durable before queue completion.  A crash in
        # between leaves a recoverable processing claim, never a falsely
        # completed request.
        self._runtime_completion(
            request,
            target_commit=target,
            status="merged",
            evidence=validation,
        )
        try:
            post_merge_evidence_receipt = dict(
                validation.get("post_merge_evidence_receipt") or {}
            )
            self.queue.complete(
                request,
                metadata={
                    "acceptance_receipt_id": receipt.receipt_id,
                    "requirement_id": PARALLEL_ACCEPTANCE_EVIDENCE_ID,
                    "target_commit": target,
                    **(
                        {
                            "post_merge_evidence_receipt_id": str(
                                post_merge_evidence_receipt.get("receipt_id")
                                or ""
                            ),
                            "post_merge_evidence_requirement_id": str(
                                post_merge_evidence_receipt.get(
                                    "requirement_id"
                                )
                                or next(
                                    iter(
                                        post_merge_evidence_receipt.get(
                                            "proved_requirement_ids"
                                        )
                                        or ()
                                    ),
                                    "",
                                )
                                or ""
                            ),
                        }
                        if post_merge_evidence_receipt
                        else {}
                    ),
                },
            )
        except MergeQueueFenceError as exc:
            fenced = {
                **integration,
                "status": (
                    "integrated_pending_acceptance"
                    if callback_owned
                    else "fenced_out"
                ),
                "accepted": False,
                "acceptance_pending": callback_owned,
                "reason": "merge_queue_claim_fenced",
                "fence_error": f"{type(exc).__name__}: {exc}",
                "post_merge_validation": validation,
                "acceptance_receipt": receipt_payload,
                "finished_at": time.time(),
            }
            if callback_owned:
                fenced.update(
                    {
                        "integrated": True,
                        "completion_authoritative": False,
                        "integration_terminal": True,
                        "queue_settlement": {
                            "status": "fenced_out",
                            "terminal": False,
                            "fence_stage": "before_queue_completion",
                            "fence_error": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                        },
                    }
                )
            self._write_receipt(
                f"fenced-{request.request_id}", fenced
            )
            if callback_owned:
                self._write_receipt(
                    self._dedupe_key(canonical, candidate), fenced
                )
            return fenced
        integration.update(
            {
                "accepted": True,
                "acceptance_pending": False,
                "post_merge_validation": validation,
                **(
                    {
                        "post_merge_evidence_receipt": dict(
                            validation.get(
                                "post_merge_evidence_receipt"
                            )
                            or {}
                        )
                    }
                    if validation.get("post_merge_evidence_receipt")
                    else {}
                ),
                "validation_receipt_ids": list(
                    receipt.validation_receipt_ids
                ),
                "acceptance_receipt": receipt_payload,
                "finished_at": time.time(),
            }
        )
        self._write_receipt(
            self._dedupe_key(canonical, candidate), integration
        )
        # The live typed object retains producer authority in a bounded
        # in-memory ledger.  Its JSON projection is durable and
        # content-addressed, but restoration remains diagnostic so a caller
        # cannot manufacture completion authority from lookalike JSON.
        self._acceptance_evidence.append(receipt)
        return integration

    def _write_acceptance_receipt(
        self, payload: Mapping[str, Any]
    ) -> Path:
        receipt_id = str(payload.get("receipt_id") or "")
        digest = receipt_id.split(":", 1)[-1]
        path = self.receipt_dir / f"acceptance-{digest}.json"
        self._atomic_json(path, payload)
        return path

    def acceptance_evidence_receipts(
        self,
    ) -> tuple[ParallelAcceptanceReceipt, ...]:
        """Return live producer-sealed receipts retained by this train."""

        return tuple(self._acceptance_evidence)

    def _recover_abandoned_claims(self) -> int:
        recover = getattr(self.queue, "recover_abandoned_train_claims", None)
        if not callable(recover):
            return 0
        return int(recover() or 0)

    def _worktree_disk_usage(self) -> tuple[int, int]:
        """Return allocated bytes and child count beneath the train root."""

        total = 0
        children = 0
        try:
            roots = list(os.scandir(self.worktree_dir))
        except OSError:
            return 0, 0
        for root in roots:
            children += 1
            stack = [root]
            while stack:
                entry = stack.pop()
                try:
                    if entry.is_symlink():
                        total += entry.stat(follow_symlinks=False).st_size
                    elif entry.is_dir(follow_symlinks=False):
                        stack.extend(os.scandir(entry.path))
                    else:
                        stat = entry.stat(follow_symlinks=False)
                        # st_blocks measures actual disk pressure while
                        # remaining deterministic enough for a hard bound.
                        total += max(stat.st_size, stat.st_blocks * 512)
                except (FileNotFoundError, OSError):
                    continue
        return total, children

    def _cleanup_abandoned_worktrees(self) -> int:
        """Remove crash-left train worktrees while holding the consumer lease."""

        removed = 0
        root = self.worktree_dir.resolve()
        listing = self._git("worktree", "list", "--porcelain")
        if listing.returncode == 0:
            for line in listing.stdout.splitlines():
                if not line.startswith("worktree "):
                    continue
                candidate = Path(
                    line.removeprefix("worktree ").strip()
                ).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    continue
                self._git(
                    "worktree", "remove", "--force", str(candidate)
                )
                shutil.rmtree(candidate, ignore_errors=True)
                removed += 1
        try:
            children = list(self.worktree_dir.iterdir())
        except OSError:
            children = []
        for child in children:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    continue
            removed += 1
        self._git("worktree", "prune", "--expire", "now")
        return removed

    def _worktree_admission_failure(self) -> dict[str, Any] | None:
        observed_bytes, active = self._worktree_disk_usage()
        if active >= self.max_active_worktrees:
            return {
                "merged": False,
                "retryable": True,
                "reason": "worktree_count_limit_exceeded",
                "active_worktrees": active,
                "max_active_worktrees": self.max_active_worktrees,
                "worktree_disk_bytes": observed_bytes,
                "max_worktree_disk_bytes": self.max_worktree_disk_bytes,
            }
        if observed_bytes >= self.max_worktree_disk_bytes:
            return {
                "merged": False,
                "retryable": True,
                "reason": "worktree_disk_limit_exceeded",
                "active_worktrees": active,
                "max_active_worktrees": self.max_active_worktrees,
                "worktree_disk_bytes": observed_bytes,
                "max_worktree_disk_bytes": self.max_worktree_disk_bytes,
            }
        return None

    def _check_worktree_disk_bound(self) -> dict[str, Any] | None:
        observed_bytes, active = self._worktree_disk_usage()
        if observed_bytes <= self.max_worktree_disk_bytes:
            return None
        return {
            "merged": False,
            "retryable": True,
            "reason": "worktree_disk_limit_exceeded",
            "active_worktrees": active,
            "max_active_worktrees": self.max_active_worktrees,
            "worktree_disk_bytes": observed_bytes,
            "max_worktree_disk_bytes": self.max_worktree_disk_bytes,
        }

    def status(self) -> dict[str, Any]:
        queue_status = self.queue.status() if hasattr(self.queue, "status") else {}
        worktree_bytes, active_worktrees = self._worktree_disk_usage()
        return {
            "owner_id": self.owner_id,
            "target_branch": self.target_branch,
            "state_dir": str(self.state_dir),
            "consumer_lock_path": str(self.consumer_lock_path),
            "proof_gate_enabled": self.formal_verification_policy is not None,
            "proof_policy_id": (
                self.formal_verification_policy.policy_id
                if self.formal_verification_policy is not None
                else ""
            ),
            "proof_cache_dir": str(self.proof_cache_dir),
            "preflight_workers": self.preflight_workers,
            "post_merge_validation_required": (
                self.post_merge_validation is not None
            ),
            "post_merge_evidence_required": (
                self.post_merge_evidence is not None
            ),
            "distributed_publication_required": (
                self.distributed_publication_required
            ),
            "distributed_repository_id": self.distributed_repository_id,
            "distributed_post_merge_evidence_required": (
                self.distributed_post_merge_evidence_required
            ),
            "distributed_publication_ledger_path": str(
                self.distributed_publication_ledger_path
            ),
            "acceptance_requirement_id": PARALLEL_ACCEPTANCE_EVIDENCE_ID,
            "throughput": dict(self._last_throughput),
            "worktree_resources": {
                "disk_bytes": worktree_bytes,
                "max_disk_bytes": self.max_worktree_disk_bytes,
                "active": active_worktrees,
                "max_active": self.max_active_worktrees,
                "backpressure": (
                    worktree_bytes >= self.max_worktree_disk_bytes
                    or active_worktrees >= self.max_active_worktrees
                ),
            },
            "queue": queue_status,
        }

    def _process_claimed(
        self,
        request: MergeRequest,
        *,
        defer_completion: bool = False,
        preflight_receipt: Mapping[str, Any] | None = None,
        publication_admission: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        started_at = time.time()
        if publication_admission is None:
            publication_admission = self.admit_distributed_publication(
                request
            )
        if not bool(publication_admission.get("admitted")):
            return dict(publication_admission)
        canonical = str(getattr(request, "canonical_identity", "") or "") or _request_value(
            request,
            "canonical_task_id",
            "canonical_task_id",
            "canonical_task_key",
            "task_cid",
        ) or _request_value(request, "task_id")
        candidate = _request_value(
            request,
            "commit_sha",
            "commit_sha",
            "implementation_commit",
            "commit",
        )
        branch = _request_value(request, "branch_name", "branch")

        if not candidate and branch:
            candidate_result = self._git("rev-parse", "--verify", f"{branch}^{{commit}}")
            if candidate_result.returncode == 0:
                candidate = candidate_result.stdout.strip()
        if not canonical or not candidate:
            return self._finish_failure(
                request,
                reason="invalid_merge_request",
                details={"canonical_task_id": canonical, "commit_sha": candidate, "branch": branch},
                started_at=started_at,
                retryable=False,
            )

        verified = self._git("rev-parse", "--verify", f"{candidate}^{{commit}}")
        if verified.returncode != 0:
            return self._finish_failure(
                request,
                reason="candidate_commit_missing",
                details={"commit_sha": candidate, "stderr": verified.stderr[-2000:]},
                started_at=started_at,
                retryable=False,
            )
        candidate = verified.stdout.strip()
        target = self._target_commit()
        if not target:
            return self._finish_failure(
                request,
                reason="target_branch_missing",
                details={"target_branch": self.target_branch},
                started_at=started_at,
            )

        proof_gate_receipt: dict[str, Any] = {}
        proof_tree_id = ""
        try:
            proof_policy = self._proof_policy_for_request(request)
        except Exception as exc:
            return self._finish_failure(
                request,
                reason="proof_gate_identity_invalid",
                details={"proof_gate_error": f"{type(exc).__name__}: {exc}"},
                started_at=started_at,
                retryable=False,
            )
        if proof_policy is not None:
            tree_result = self._git("rev-parse", "--verify", f"{candidate}^{{tree}}")
            if tree_result.returncode != 0 or not tree_result.stdout.strip():
                return self._finish_failure(
                    request,
                    reason="candidate_tree_missing",
                    details={
                        "commit_sha": candidate,
                        "stderr": tree_result.stderr[-2000:],
                    },
                    started_at=started_at,
                    retryable=False,
                )
            proof_tree_id = f"git-tree:{tree_result.stdout.strip()}"
            gate = self._evaluate_proof_gate(
                request=request,
                candidate=candidate,
                target=target,
                repository_tree_id=proof_tree_id,
                policy=proof_policy,
            )
            proof_gate_receipt = dict(gate.get("receipt") or {})
            if not gate.get("allowed", False):
                return self._finish_failure(
                    request,
                    reason=str(gate.get("reason") or "proof_gate_blocked"),
                    details={
                        "proof_gate": proof_gate_receipt,
                        "proof_gate_cache_hit": bool(gate.get("cache_hit")),
                        "repository_tree_id": proof_tree_id,
                    },
                    started_at=started_at,
                    retryable=bool(gate.get("retryable", True)),
                )

        # A callback owns the complete integration lifecycle, including nested
        # repository handoff and taskboard completion.  Root-level receipts or
        # ancestry only prove that the parent commit landed; they cannot safely
        # short-circuit those callback side effects after a daemon restart.
        if self.merge_callback is None:
            dedupe_key = self._dedupe_key(canonical, candidate)
            previous = self._read_receipt(dedupe_key)
            if previous and str(previous.get("status")) in {
                "merged",
                "already_merged",
                "deduplicated",
            }:
                return self._finish_success(
                    request,
                    status="deduplicated",
                    canonical=canonical,
                    candidate=candidate,
                    target=target,
                    started_at=started_at,
                    extra={
                        "duplicate_of": previous.get("request_id", ""),
                        "distributed_publication_admission": dict(
                            publication_admission
                        ),
                        **(
                            {
                                "proof_gate": proof_gate_receipt,
                                "repository_tree_id": proof_tree_id,
                            }
                            if proof_gate_receipt
                            else {}
                        ),
                    },
                    defer_completion=defer_completion,
                    preflight_receipt=preflight_receipt,
                )

            if self._is_ancestor(candidate, target):
                return self._finish_success(
                    request,
                    status="already_merged",
                    canonical=canonical,
                    candidate=candidate,
                    target=target,
                    started_at=started_at,
                    extra=(
                        {
                            "distributed_publication_admission": dict(
                                publication_admission
                            ),
                            "proof_gate": proof_gate_receipt,
                            "repository_tree_id": proof_tree_id,
                        }
                        if proof_gate_receipt
                        else {
                            "distributed_publication_admission": dict(
                                publication_admission
                            )
                        }
                    ),
                    defer_completion=defer_completion,
                    preflight_receipt=preflight_receipt,
                )

        if self.merge_callback is not None:
            if not self._owns_claim(request):
                return self._finish_failure(
                    request,
                    reason="merge_queue_claim_fenced",
                    details={
                        "fence_stage": "before_merge_callback",
                    },
                    started_at=started_at,
                    retryable=True,
                )
            callback_receipt_key = self._dedupe_key(canonical, candidate)
            previous_callback_result = self._read_receipt(
                callback_receipt_key
            )
            previous_target = str(
                previous_callback_result.get("target_commit") or ""
            )
            callback_integration_recoverable = bool(
                previous_callback_result.get("callback_owned_integration")
                and previous_callback_result.get("integrated") is True
                and previous_callback_result.get("acceptance_pending") is True
                and str(
                    previous_callback_result.get("canonical_task_id") or ""
                )
                == canonical
                and str(previous_callback_result.get("commit_sha") or "")
                == candidate
                and previous_target
                and (
                    previous_target == target
                    or self._is_ancestor(previous_target, target)
                )
            )
            if callback_integration_recoverable:
                return self._settle_recovered_callback_integration(
                    request,
                    canonical=canonical,
                    candidate=candidate,
                    target=target,
                    previous=previous_callback_result,
                    receipt_key=callback_receipt_key,
                    preflight_receipt=preflight_receipt,
                )
            try:
                callback_result = dict(
                    self._runtime_mutation(
                        "merge",
                        {
                            "operation": "merge_callback",
                            "request_id": request.request_id,
                            "candidate_commit": candidate,
                            "target_commit": target,
                            "target_branch": self.target_branch,
                        },
                        lambda: dict(self.merge_callback(request) or {}),
                    )
                    or {}
                )
            except Exception as exc:  # callbacks are an isolation boundary
                return self._finish_failure(
                    request,
                    reason="merge_callback_exception",
                    details={"exception": f"{type(exc).__name__}: {exc}"},
                    started_at=started_at,
                )
            if callback_result.get("merged") or callback_result.get("already_merged"):
                callback_target = str(
                    self._target_commit()
                    or callback_result.get("target_commit")
                    or callback_result.get("merge_commit")
                    or target
                )
                callback_validation: dict[str, Any] = {}
                if self.post_merge_validation is not None:
                    raw_callback_validation = callback_result.get(
                        "post_merge_validation"
                    )
                    validation_missing = raw_callback_validation is None
                    callback_validation = self._normalize_gate_result(
                        raw_callback_validation,
                        default_reason=(
                            "callback_post_merge_validation_missing"
                            if validation_missing
                            else "callback_post_merge_validation_failed"
                        ),
                    )
                    validated_commit = str(
                        callback_validation.get("validated_commit")
                        or callback_validation.get("target_commit")
                        or ""
                    )
                    callback_validation["synthesized_commit"] = (
                        callback_target
                    )
                    if (
                        callback_validation.get("passed") is True
                        and validated_commit != callback_target
                    ):
                        callback_validation.update(
                            {
                                "passed": False,
                                "reason": (
                                    "callback_post_merge_validation_unbound"
                                ),
                                "validated_commit": validated_commit,
                                "synthesized_commit": callback_target,
                            }
                        )
                    if not bool(callback_validation.get("passed")):
                        return self._finish_integrated_pending_validation(
                            request,
                            canonical=canonical,
                            candidate=candidate,
                            target=callback_target,
                            started_at=started_at,
                            merged=bool(callback_result.get("merged")),
                            already_merged=bool(
                                callback_result.get("already_merged")
                            ),
                            validation_reason=str(
                                callback_validation.get("reason")
                                or "callback_post_merge_validation_missing"
                            ),
                            post_merge_validation=callback_validation,
                            extra={
                                "merge_result": callback_result,
                                "distributed_publication_admission": dict(
                                    publication_admission
                                ),
                                **(
                                    {
                                        "proof_gate": proof_gate_receipt,
                                        "repository_tree_id": proof_tree_id,
                                    }
                                    if proof_gate_receipt
                                    else {}
                                ),
                            },
                            preflight_receipt=preflight_receipt,
                        )
                    callback_result["post_merge_validation"] = (
                        callback_validation
                    )
                success_extra: dict[str, Any] = {
                    "merge_result": callback_result,
                    "distributed_publication_admission": dict(
                        publication_admission
                    ),
                    "accepted": True,
                    "acceptance_pending": False,
                    **(
                        {
                            "proof_gate": proof_gate_receipt,
                            "repository_tree_id": proof_tree_id,
                        }
                        if proof_gate_receipt
                        else {}
                    ),
                }
                if callback_validation:
                    # _finish_success passes this top-level evidence to the
                    # runtime completion decision.  Keeping it only beneath
                    # merge_result silently discarded the callback receipt.
                    success_extra["post_merge_validation"] = (
                        callback_validation
                    )
                return self._finish_success(
                    request,
                    status="merged" if callback_result.get("merged") else "already_merged",
                    canonical=canonical,
                    candidate=candidate,
                    target=callback_target,
                    started_at=started_at,
                    extra=success_extra,
                    defer_completion=defer_completion,
                    preflight_receipt=preflight_receipt,
                    callback_owned_integration=True,
                )
            callback_reason = str(callback_result.get("reason") or "merge_callback_failed")
            retryable = callback_reason not in {
                "invalid_merge_request",
                "candidate_commit_missing",
                "validation_failed",
                "branch_has_no_changes",
            }
            return self._finish_failure(
                request,
                reason=callback_reason,
                details={
                    "merge_result": callback_result,
                    **(
                        {
                            "proof_gate": proof_gate_receipt,
                            "repository_tree_id": proof_tree_id,
                        }
                        if proof_gate_receipt
                        else {}
                    ),
                },
                started_at=started_at,
                retryable=retryable,
            )

        result = self._rebase_and_integrate(
            request=request,
            canonical=canonical,
            candidate=candidate,
            target=target,
            proof_tree_id=proof_tree_id,
        )
        if result.get("merged"):
            return self._finish_success(
                request,
                status="merged",
                canonical=canonical,
                candidate=candidate,
                target=str(result.get("target_commit") or target),
                started_at=started_at,
                extra={
                    **result,
                    "distributed_publication_admission": dict(
                        publication_admission
                    ),
                    **(
                        {
                            "proof_gate": proof_gate_receipt,
                            "repository_tree_id": proof_tree_id,
                        }
                        if proof_gate_receipt
                        else {}
                    ),
                },
                defer_completion=defer_completion,
                preflight_receipt=preflight_receipt,
            )
        return self._finish_failure(
            request,
            reason=str(result.get("reason") or "merge_failed"),
            details={
                **result,
                **(
                    {
                        "proof_gate": proof_gate_receipt,
                        "repository_tree_id": proof_tree_id,
                    }
                    if proof_gate_receipt
                    else {}
                ),
            },
            started_at=started_at,
            retryable=bool(result.get("retryable", True)),
        )

    @staticmethod
    def _metadata_strings(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return tuple(
                item.strip()
                for item in value.replace("\n", ",").split(",")
                if item.strip()
            )
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return tuple(str(item).strip() for item in value if str(item).strip())
        return ()

    @staticmethod
    def _risk_for_priority(priority: str) -> RiskLevel:
        return {
            "P0": RiskLevel.CRITICAL,
            "P1": RiskLevel.HIGH,
            "P2": RiskLevel.MEDIUM,
            "P3": RiskLevel.LOW,
        }.get(str(priority or "").strip().upper(), RiskLevel.MEDIUM)

    @staticmethod
    def _modeled_invariant_hints(path: str) -> tuple[str, ...]:
        """Conservatively classify modeled supervisor surfaces by path.

        These hints are only used when a producer did not provide richer
        symbol-level scope information.  They are deliberately deterministic
        and additive: a broad hint can select a stricter rule, but cannot lower
        a requirement selected by another hint.
        """

        lowered = path.casefold()
        values: set[str] = set()
        if "agent_supervisor/" in lowered:
            values.update(
                {
                    InvariantClass.DATA_INTEGRITY.value,
                    InvariantClass.STATE_TRANSITION.value,
                }
            )
        if "merge" in lowered:
            values.add(InvariantClass.MERGE_IDEMPOTENCE.value)
        if any(token in lowered for token in ("queue", "scheduler", "goal", "task")):
            values.add(InvariantClass.STATE_TRANSITION.value)
        if any(token in lowered for token in ("dag", "graph", "dependency")):
            values.add(InvariantClass.DAG_ACYCLICITY.value)
        if any(token in lowered for token in ("lease", "lock")):
            values.add(InvariantClass.LEASE_SAFETY.value)
        if "cache" in lowered:
            values.add(InvariantClass.CACHE_KEY_COMPLETENESS.value)
        if any(token in lowered for token in ("proof", "evidence", "receipt")):
            values.add(InvariantClass.EVIDENCE_FRESHNESS.value)
        if any(token in lowered for token in ("auth", "permission", "override")):
            values.add(InvariantClass.AUTHORIZATION.value)
        if any(token in lowered for token in ("resource", "worktree", "sandbox")):
            values.add(InvariantClass.RESOURCE_ISOLATION.value)
        return tuple(sorted(values))

    def _changed_scopes(
        self,
        request: MergeRequest,
        *,
        candidate: str,
        target: str,
    ) -> tuple[ChangedScope, ...]:
        metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        supplied = metadata.get("proof_changed_scopes")
        if isinstance(supplied, Sequence) and not isinstance(
            supplied, (str, bytes, bytearray)
        ):
            if metadata.get("proof_changed_scopes_complete") is False:
                raise ValueError("proof changed-scope packet is incomplete")
            scopes = []
            for raw in supplied:
                if isinstance(raw, ChangedScope):
                    scopes.append(raw)
                    continue
                if not isinstance(raw, Mapping):
                    raise ValueError("proof_changed_scopes entries must be mappings")
                # Queue producers may use the canonical wire contract or a
                # compact, JSON-friendly spelling.
                if raw.get("schema"):
                    scopes.append(ChangedScope.from_dict(raw))
                else:
                    scopes.append(
                        ChangedScope(
                            path=raw.get("path", ""),
                            ast_scope_ids=tuple(raw.get("ast_scope_ids") or ()),
                            risk=raw.get(
                                "risk", self._risk_for_priority(request.priority)
                            ),
                            invariant_classes=tuple(
                                raw.get("invariant_classes") or ()
                            ),
                            change_kind=raw.get("change_kind", "modified"),
                            metadata=raw.get("metadata") or {},
                        )
                    )
            return tuple(sorted(scopes, key=lambda item: item.scope_id))

        baseline = str(metadata.get("baseline_ref") or "").strip()
        if not baseline:
            merge_base = self._git("merge-base", candidate, target)
            baseline = (
                merge_base.stdout.strip()
                if merge_base.returncode == 0 and merge_base.stdout.strip()
                else target
            )
        diff = self._git(
            "diff",
            "--name-status",
            "--find-renames",
            baseline,
            candidate,
        )
        if diff.returncode != 0:
            raise ValueError(f"could not derive proof changed scopes: {diff.stderr[-1000:]}")

        task_payload = metadata.get("task")
        task_metadata = (
            task_payload.get("metadata")
            if isinstance(task_payload, Mapping)
            and isinstance(task_payload.get("metadata"), Mapping)
            else {}
        )
        ast_scope_ids = self._metadata_strings(
            metadata.get("proof_ast_scope_ids")
            or task_metadata.get("ast symbols")
            or task_metadata.get("ast_symbols")
        )
        explicit_invariants = self._metadata_strings(
            metadata.get("proof_invariant_classes")
            or task_metadata.get("invariant classes")
            or task_metadata.get("invariant_classes")
        )
        risk = self._risk_for_priority(request.priority)
        scopes: list[ChangedScope] = []
        for line in diff.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            status = fields[0]
            path = fields[-1]
            kind = {
                "A": "added",
                "D": "deleted",
                "R": "renamed",
                "C": "copied",
            }.get(status[:1], "modified")
            invariants = tuple(
                sorted(
                    set(explicit_invariants)
                    | set(self._modeled_invariant_hints(path))
                )
            )
            scopes.append(
                ChangedScope(
                    path=path,
                    ast_scope_ids=ast_scope_ids,
                    risk=risk,
                    invariant_classes=invariants,
                    change_kind=kind,
                    metadata={"git_status": status},
                )
            )
        return tuple(sorted(scopes, key=lambda item: item.scope_id))

    @staticmethod
    def _proof_gate_cache_key(selection: PolicySelection) -> str:
        payload = "\0".join(
            (
                selection.policy_id,
                selection.selection_id,
                selection.repository_tree_id,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _proof_policy_for_request(
        self, request: MergeRequest
    ) -> FormalVerificationPolicy | None:
        """Resolve the immutable queue policy without allowing a retry downgrade.

        Queue producers persist the complete policy snapshot beside the
        candidate.  A consumer may additionally have a configured policy, but
        it must be the same snapshot.  This prevents a restarted consumer with
        a missing or weaker configuration from bypassing a gate that already
        timed out or observed an unavailable provider.
        """

        metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        raw_policy = metadata.get("formal_verification_policy")
        queued_policy: FormalVerificationPolicy | None = None
        if raw_policy:
            if not isinstance(raw_policy, Mapping):
                raise ValueError("queued formal-verification policy is malformed")
            queued_policy = FormalVerificationPolicy.from_dict(raw_policy)
        configured = self.formal_verification_policy
        if configured is not None and queued_policy is not None:
            if configured.policy_id != queued_policy.policy_id:
                raise ValueError(
                    "configured proof policy does not match the queued policy snapshot"
                )
        return queued_policy or configured

    @staticmethod
    def _default_proof_plan(
        *,
        policy: FormalVerificationPolicy,
        selection: PolicySelection,
    ) -> dict[str, Any]:
        """Describe the exact selected work even when a provider is unavailable."""

        payload: dict[str, Any] = {
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-proof-plan@1",
            "policy_id": policy.policy_id,
            "selection_id": selection.selection_id,
            "repository_tree_id": selection.repository_tree_id,
            "requirement_ids": [
                item.requirement_id for item in selection.requirements
            ],
            "requirements": [item.to_dict() for item in selection.requirements],
            "fallback_validations": list(selection.fallback_validations),
        }
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        )
        payload["plan_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _pin_proof_selection(
        self,
        request: MergeRequest,
        *,
        selection: PolicySelection,
        changes: Sequence[ChangedScope],
    ) -> dict[str, Any]:
        path = self.proof_gate_pin_dir / f"{request.request_id}.json"
        pin = {
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-proof-pin@1",
            "request_id": request.request_id,
            "policy_id": selection.policy_id,
            "selection_id": selection.selection_id,
            "repository_tree_id": selection.repository_tree_id,
            "rollout_mode": selection.rollout_mode.value,
            "changed_scope_ids": [item.scope_id for item in changes],
        }
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("proof gate pin is unreadable") from exc
            for name in (
                "request_id",
                "policy_id",
                "selection_id",
                "repository_tree_id",
                "rollout_mode",
                "changed_scope_ids",
            ):
                if existing.get(name) != pin.get(name):
                    raise ValueError(f"proof gate pin mismatch: {name}")
            return dict(existing)
        self._atomic_json(path, pin)
        return pin

    @staticmethod
    def _gate_receipt_type() -> Any:
        # Kept lazy so older installations which only use the ungated train
        # can still import this module during a rolling deployment.
        from . import formal_verification_policy as policy_module

        receipt_type = getattr(policy_module, "MergeProofGateReceipt", None)
        if receipt_type is None:
            raise RuntimeError("MergeProofGateReceipt is unavailable")
        return receipt_type

    def _read_cached_gate_receipt(
        self, selection: PolicySelection
    ) -> Any | None:
        path = self.proof_cache_dir / f"{self._proof_gate_cache_key(selection)}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            receipt = self._gate_receipt_type().from_dict(payload)
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        if (
            receipt.policy_id != selection.policy_id
            or receipt.selection_id != selection.selection_id
            or receipt.repository_tree_id != selection.repository_tree_id
            or not receipt.allowed
            or getattr(receipt, "override_receipt_id", "")
            or not all(
                result.requirement_satisfied
                for result in receipt.decision.results
            )
        ):
            return None
        return receipt

    def _persist_gate_receipt(
        self,
        request: MergeRequest,
        receipt: Any,
        *,
        cache: bool,
    ) -> dict[str, Any]:
        payload = dict(receipt.to_dict())
        attempt = max(1, int(getattr(request, "attempt", 1) or 1))
        attempt_path = (
            self.proof_gate_attempt_dir
            / f"{request.request_id}-attempt-{attempt}.json"
        )
        self._atomic_json(attempt_path, payload)
        if cache:
            selection = PolicySelection.from_dict(payload["selection"])
            cache_path = (
                self.proof_cache_dir
                / f"{self._proof_gate_cache_key(selection)}.json"
            )
            self._atomic_json(cache_path, payload)
        return payload

    def _evaluate_proof_gate(
        self,
        *,
        request: MergeRequest,
        candidate: str,
        target: str,
        repository_tree_id: str,
        policy: FormalVerificationPolicy,
    ) -> dict[str, Any]:
        try:
            changes = self._changed_scopes(
                request, candidate=candidate, target=target
            )
            selection = policy.select(
                changes, repository_tree_id=repository_tree_id
            )
            self._pin_proof_selection(
                request, selection=selection, changes=changes
            )
        except Exception as exc:
            return {
                "allowed": False,
                "retryable": False,
                "reason": "proof_gate_identity_invalid",
                "receipt": {
                    "policy_id": policy.policy_id,
                    "repository_tree_id": repository_tree_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            }

        cached = self._read_cached_gate_receipt(selection)
        if cached is not None:
            receipt_type = self._gate_receipt_type()
            cached_metadata = dict(getattr(cached, "metadata", {}) or {})
            cached_metadata["reused_receipt_id"] = cached.receipt_id
            receipt = receipt_type.build(
                policy=policy,
                selection=selection,
                repository_tree_id=repository_tree_id,
                proof_plan=getattr(cached, "proof_plan", None) or None,
                outcomes=getattr(cached, "proof_outcomes", ()),
                validations=getattr(cached, "validation_outcomes", ()),
                proof_receipts=getattr(cached, "proof_receipts", ()),
                proof_receipt_ids=getattr(cached, "proof_receipt_ids", ()),
                provider_status=getattr(cached, "provider_status", {}),
                provider_error=getattr(cached, "provider_error", ""),
                cache_status={
                    "status": "hit",
                    "cache_key": self._proof_gate_cache_key(selection),
                    "reused_receipt_id": cached.receipt_id,
                },
                metadata=cached_metadata,
            )
            payload = self._persist_gate_receipt(request, receipt, cache=False)
            return {
                "allowed": True,
                "retryable": False,
                "reason": "proof_gate_cache_hit",
                "receipt": payload,
                "cache_hit": True,
            }

        metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        evidence: Any = metadata.get("proof_gate") or metadata.get(
            "proof_gate_evidence"
        )
        provider_error = ""
        callback_failed = False
        if self.proof_gate is not None:
            try:
                evidence = self._call_compatible(
                    self.proof_gate,
                    request,
                    policy=policy,
                    selection=selection,
                    repository_tree_id=repository_tree_id,
                    changed_scopes=changes,
                    changes=changes,
                    cached_evidence=None,
                    candidate_commit=candidate,
                    target_commit=target,
                )
            except Exception as exc:
                callback_failed = True
                provider_error = f"{type(exc).__name__}: {exc}"
                evidence = {
                    "provider_status": {
                        "status": (
                            "timed_out"
                            if isinstance(exc, TimeoutError)
                            else "unavailable"
                        )
                    },
                    "provider_error": provider_error,
                }

        receipt_type = self._gate_receipt_type()
        try:
            if isinstance(evidence, receipt_type):
                receipt = evidence
            elif isinstance(evidence, Mapping) and str(
                evidence.get("schema") or ""
            ).endswith("merge-proof-gate-receipt@1"):
                receipt = receipt_type.from_dict(evidence)
            else:
                packet = dict(evidence) if isinstance(evidence, Mapping) else {}
                for claimed_name, actual in (
                    ("policy_id", policy.policy_id),
                    ("selection_id", selection.selection_id),
                    ("repository_tree_id", repository_tree_id),
                ):
                    claimed = str(packet.get(claimed_name) or "")
                    if claimed and claimed != actual:
                        raise ValueError(
                            f"proof gate {claimed_name} does not match pinned value"
                        )
                receipt = receipt_type.build(
                    policy=policy,
                    selection=selection,
                    repository_tree_id=repository_tree_id,
                    proof_plan=(
                        packet.get("proof_plan")
                        or self._default_proof_plan(
                            policy=policy,
                            selection=selection,
                        )
                    ),
                    outcomes=packet.get(
                        "proof_outcomes", packet.get("outcomes")
                    ),
                    validations=packet.get(
                        "validations", packet.get("validation_outcomes")
                    ),
                    proof_receipts=tuple(packet.get("proof_receipts") or ()),
                    proof_receipt_ids=tuple(
                        packet.get("proof_receipt_ids") or ()
                    ),
                    override=packet.get("override"),
                    provider_status=packet.get("provider_status"),
                    provider_error=str(
                        packet.get("provider_error") or provider_error
                    ),
                    cache_status={
                        "status": "miss",
                        "cache_key": self._proof_gate_cache_key(selection),
                    },
                )
            if (
                receipt.policy_id != policy.policy_id
                or receipt.selection_id != selection.selection_id
                or receipt.repository_tree_id != repository_tree_id
            ):
                raise ValueError("proof gate receipt identity does not match pinned selection")
        except Exception as exc:
            return {
                "allowed": False,
                "retryable": callback_failed,
                "reason": "proof_gate_provider_failed" if callback_failed else "proof_gate_identity_invalid",
                "receipt": {
                    "policy_id": policy.policy_id,
                    "selection_id": selection.selection_id,
                    "repository_tree_id": repository_tree_id,
                    "provider_error": provider_error
                    or f"{type(exc).__name__}: {exc}",
                },
            }

        payload = self._persist_gate_receipt(
            request,
            receipt,
            cache=bool(receipt.allowed)
            and not bool(getattr(receipt, "override_receipt_id", ""))
            and all(
                result.requirement_satisfied
                for result in receipt.decision.results
            ),
        )
        provider_status = payload.get("provider_status")
        status = (
            str(provider_status.get("status") or "")
            if isinstance(provider_status, Mapping)
            else str(provider_status or "")
        ).casefold()
        transient = callback_failed or status in {
            "timed_out",
            "timeout",
            "unavailable",
            "provider_unavailable",
            "error",
            "failed",
        }
        return {
            "allowed": bool(receipt.allowed),
            "retryable": transient or not bool(receipt.allowed),
            "reason": "proof_gate_allowed" if receipt.allowed else "proof_gate_blocked",
            "receipt": payload,
            "cache_hit": False,
        }

    def _validate_synthesized_tree(
        self,
        *,
        request: MergeRequest,
        workspace: Path,
        candidate_commit: str,
        synthesized_commit: str,
        target_commit_before: str,
    ) -> dict[str, Any]:
        """Validate the exact commit that would win the target CAS."""

        validator = self.post_merge_validation
        if validator is None:
            return {
                "passed": False,
                "reason": "post_merge_validation_not_configured",
            }
        binding = self._gate_cache_binding(
            kind="post-merge",
            request=request,
            candidate_commit=candidate_commit,
            target_commit=synthesized_commit,
            gate_id=self.post_merge_gate_id,
        )
        # Authoritative evidence is rebuilt for every actual synthesized tree:
        # freshness, revocation and source receipt changes must not be hidden
        # behind an earlier passing validation cache record.
        cached = (
            None
            if self.post_merge_evidence is not None
            else self._read_gate_cache(binding)
        )
        if cached is not None:
            return cached
        started = time.monotonic()
        try:
            raw = self._call_compatible(
                validator,
                request,
                workspace=workspace,
                target_commit=synthesized_commit,
                synthesized_commit=synthesized_commit,
                candidate_commit=candidate_commit,
                target_commit_before=target_commit_before,
                repo_root=self.repo_root,
            )
            validation = self._normalize_gate_result(
                raw, default_reason="post_merge_validation_failed"
            )
        except Exception as exc:
            validation = {
                "passed": False,
                "reason": "post_merge_validation_exception",
                "error": f"{type(exc).__name__}: {exc}",
            }
        validation["elapsed_seconds"] = max(
            0.0, time.monotonic() - started
        )
        claimed_commit = str(
            validation.get("validated_commit")
            or validation.get("target_commit")
            or synthesized_commit
        )
        validation["validated_commit"] = claimed_commit
        validation["synthesized_commit"] = synthesized_commit
        validation["target_commit_before"] = target_commit_before
        if claimed_commit != synthesized_commit:
            validation.update(
                {
                    "passed": False,
                    "reason": "post_merge_validation_target_mismatch",
                }
            )
        repository_tree_id = self._repository_tree_id(
            synthesized_commit,
            workspace=workspace,
        )
        validation["repository_tree_id"] = repository_tree_id
        if not repository_tree_id:
            validation.update(
                {
                    "passed": False,
                    "reason": "post_merge_repository_tree_missing",
                }
            )
        if (
            validation.get("passed") is True
            and self.post_merge_evidence is not None
        ):
            evidence = self._assemble_post_merge_evidence(
                request=request,
                workspace=workspace,
                candidate_commit=candidate_commit,
                synthesized_commit=synthesized_commit,
                target_commit_before=target_commit_before,
                repository_tree_id=repository_tree_id,
                validation=validation,
            )
            validation["post_merge_evidence"] = evidence
            validation["post_merge_evidence_receipt"] = dict(
                evidence.get("receipt") or {}
            )
            if evidence.get("passed") is not True:
                validation.update(
                    {
                        "passed": False,
                        "reason": str(
                            evidence.get("reason")
                            or "post_merge_evidence_failed"
                        ),
                        "retryable": False,
                    }
                )
        validation["validation_receipt_ids"] = list(
            self._validation_receipt_ids(validation)
        )
        if self.post_merge_evidence is None:
            self._write_gate_cache(binding, validation)
        return validation

    def _validate_existing_integrated_commit(
        self,
        request: MergeRequest,
        *,
        commit: str,
        candidate_commit: str,
    ) -> dict[str, Any]:
        """Validate an already-integrated/deduplicated commit in isolation."""

        admission = self._worktree_admission_failure()
        if admission is not None:
            return {
                "passed": False,
                **{
                    key: value
                    for key, value in admission.items()
                    if key != "merged"
                },
                "validated_commit": commit,
            }
        workspace = Path(
            tempfile.mkdtemp(prefix="validation-", dir=self.worktree_dir)
        )
        added = False
        try:
            add = self._git(
                "worktree", "add", "--detach", str(workspace), commit
            )
            if add.returncode != 0:
                return {
                    "passed": False,
                    "reason": "validation_worktree_add_failed",
                    "stderr": add.stderr[-4000:],
                    "validated_commit": commit,
                }
            added = True
            disk_failure = self._check_worktree_disk_bound()
            if disk_failure is not None:
                return {
                    "passed": False,
                    **{
                        key: value
                        for key, value in disk_failure.items()
                        if key != "merged"
                    },
                    "validated_commit": commit,
                }
            return self._validate_synthesized_tree(
                request=request,
                workspace=workspace,
                candidate_commit=candidate_commit,
                synthesized_commit=commit,
                target_commit_before=commit,
            )
        finally:
            if added:
                self._git(
                    "worktree", "remove", "--force", str(workspace)
                )
            shutil.rmtree(workspace, ignore_errors=True)

    def _rebase_and_integrate(
        self,
        *,
        request: MergeRequest,
        canonical: str,
        candidate: str,
        target: str,
        proof_tree_id: str = "",
    ) -> dict[str, Any]:
        admission = self._worktree_admission_failure()
        if admission is not None:
            return admission
        workspace = Path(tempfile.mkdtemp(prefix="candidate-", dir=self.worktree_dir))
        added = False
        try:
            add = self._git("worktree", "add", "--detach", str(workspace), candidate)
            if add.returncode != 0:
                return self._command_failure("worktree_add_failed", add)
            added = True
            disk_failure = self._check_worktree_disk_bound()
            if disk_failure is not None:
                return disk_failure

            rebase = self._git("rebase", target, cwd=workspace)
            resolver_result: dict[str, Any] = {}
            if rebase.returncode != 0:
                conflicts = self._git("ls-files", "-u", cwd=workspace).stdout
                fingerprint = conflict_fingerprint(
                    canonical_task_id=canonical,
                    candidate_commit=candidate,
                    target_commit=target,
                    conflict_index=conflicts,
                )
                resolver_result = self._resolve_conflict(
                    request=request,
                    workspace=workspace,
                    fingerprint=fingerprint,
                    candidate=candidate,
                    target=target,
                    conflicts=conflicts,
                )
                if not resolver_result.get("resolved"):
                    self._git("rebase", "--abort", cwd=workspace)
                    return {
                        "merged": False,
                        "retryable": resolver_result.get("retryable", True),
                        "reason": str(resolver_result.get("reason") or "rebase_conflict"),
                        "conflict_fingerprint": fingerprint,
                        "resolver": resolver_result,
                        "stderr": rebase.stderr[-4000:],
                    }
                # A resolver may complete the rebase itself.  Otherwise all
                # conflicts must be staged before the non-interactive continue.
                if self._git("rev-parse", "-q", "--verify", "REBASE_HEAD", cwd=workspace).returncode == 0:
                    continued = self._git(
                        "-c", "core.editor=true", "rebase", "--continue", cwd=workspace
                    )
                    if continued.returncode != 0:
                        self._git("rebase", "--abort", cwd=workspace)
                        return self._command_failure(
                            "resolver_rebase_continue_failed", continued, resolver=resolver_result
                        )

            rebased = self._git("rev-parse", "HEAD", cwd=workspace)
            if rebased.returncode != 0:
                return self._command_failure("rebased_commit_missing", rebased)
            rebased_commit = rebased.stdout.strip()
            if proof_tree_id:
                rebased_tree = self._git(
                    "rev-parse", "--verify", f"{rebased_commit}^{{tree}}", cwd=workspace
                )
                actual_tree_id = (
                    f"git-tree:{rebased_tree.stdout.strip()}"
                    if rebased_tree.returncode == 0 and rebased_tree.stdout.strip()
                    else ""
                )
                if actual_tree_id != proof_tree_id:
                    return {
                        "merged": False,
                        "retryable": True,
                        "reason": "proof_gate_tree_mismatch",
                        "proof_repository_tree_id": proof_tree_id,
                        "integration_repository_tree_id": actual_tree_id,
                        "rebased_commit": rebased_commit,
                        "stderr": rebased_tree.stderr[-2000:],
                    }
            post_merge_validation: dict[str, Any] = {}
            if self.post_merge_validation is not None:
                post_merge_validation = self._validate_synthesized_tree(
                    request=request,
                    workspace=workspace,
                    candidate_commit=candidate,
                    synthesized_commit=rebased_commit,
                    target_commit_before=target,
                )
                if not bool(post_merge_validation.get("passed")):
                    return {
                        "merged": False,
                        "retryable": bool(
                            post_merge_validation.get("retryable", False)
                        ),
                        "reason": str(
                            post_merge_validation.get("reason")
                            or "post_merge_validation_failed"
                        ),
                        "candidate_commit": candidate,
                        "rebased_commit": rebased_commit,
                        "target_commit_before": target,
                        "post_merge_validation": post_merge_validation,
                    }
            disk_failure = self._check_worktree_disk_bound()
            if disk_failure is not None:
                return {
                    **disk_failure,
                    "candidate_commit": candidate,
                    "rebased_commit": rebased_commit,
                    "target_commit_before": target,
                    **(
                        {
                            "post_merge_validation": (
                                post_merge_validation
                            )
                        }
                        if post_merge_validation
                        else {}
                    ),
                }
            # Compare-and-swap is important even under our lease: a human or a
            # different merge mechanism may legitimately advance the branch.
            if not self._owns_claim(request):
                return {
                    "merged": False,
                    "retryable": True,
                    "reason": "merge_queue_claim_fenced",
                    "fence_stage": "before_target_cas",
                    "rebased_commit": rebased_commit,
                }
            update = self._advance_target(rebased_commit, expected_target=target)
            if update.returncode != 0:
                update_reason = (
                    "target_worktree_dirty"
                    if "worktree is dirty" in str(update.stderr or "")
                    else "target_advanced"
                )
                return {
                    **self._command_failure(update_reason, update),
                    "retryable": True,
                    "rebased_commit": rebased_commit,
                }
            return {
                "merged": True,
                "rebased": rebased_commit != candidate,
                "candidate_commit": candidate,
                "rebased_commit": rebased_commit,
                "target_commit_before": target,
                "target_commit": rebased_commit,
                "merge_commit": rebased_commit,
                "resolver": resolver_result,
                **(
                    {"post_merge_validation": post_merge_validation}
                    if post_merge_validation
                    else {}
                ),
            }
        finally:
            if added:
                self._git("worktree", "remove", "--force", str(workspace))
            shutil.rmtree(workspace, ignore_errors=True)

    def _resolve_conflict(
        self,
        *,
        request: MergeRequest,
        workspace: Path,
        fingerprint: str,
        candidate: str,
        target: str,
        conflicts: str,
    ) -> dict[str, Any]:
        if self.resolver is None:
            return {"resolved": False, "reason": "rebase_conflict", "retryable": True}
        event = {
            "conflict_fingerprint": fingerprint,
            "request_id": request.request_id,
            "task_id": _request_value(request, "task_id"),
            "canonical_task_id": str(getattr(request, "canonical_identity", "") or "")
            or _request_value(request, "canonical_task_id")
            or _request_value(request, "task_id"),
            "branch": _request_value(request, "branch_name"),
            "candidate_commit": candidate,
            "source_commit": candidate,
            "commit_sha": candidate,
            "target_branch": self.target_branch,
            "target_commit": target,
            "workspace": str(workspace),
            "unmerged_index": conflicts,
            "unmerged_paths": sorted(
                {
                    line.split("\t", 1)[1]
                    for line in conflicts.splitlines()
                    if "\t" in line and line.split("\t", 1)[1]
                }
            ),
            "reason": "rebase_conflict",
        }
        claim: Any = None
        acquired = True
        outcome: dict[str, Any] = {
            "resolved": False,
            "reason": "resolver_did_not_run",
        }
        if hasattr(self.resolver, "acquire"):
            claim = self._call_compatible(
                self.resolver.acquire,
                event,
                owner_id=self.owner_id,
                fingerprint=fingerprint,
            )
            acquired = claim is not None and claim is not False
        if not acquired:
            return {"resolved": False, "reason": "resolver_already_active", "retryable": True}
        try:
            callback = getattr(self.resolver, "resolve", None)
            if callback is None and callable(self.resolver):
                callback = self.resolver
            if callback is None:
                outcome = {
                    "resolved": False,
                    "reason": "resolver_not_configured",
                    "retryable": True,
                }
                return outcome
            raw = self._call_compatible(
                callback,
                event,
                request=request,
                workspace=workspace,
                conflict_fingerprint=fingerprint,
                claim=claim,
            )
            if isinstance(raw, Mapping):
                outcome = dict(raw)
                outcome.setdefault(
                    "resolved", bool(outcome.get("applied") or outcome.get("completed"))
                )
                return outcome
            outcome = {"resolved": bool(raw)}
            return outcome
        except Exception as exc:
            outcome = {
                "resolved": False,
                "reason": "resolver_exception",
                "exception": f"{type(exc).__name__}: {exc}",
                "retryable": True,
            }
            return outcome
        finally:
            if claim is not None and hasattr(self.resolver, "release"):
                try:
                    self._call_compatible(
                        self.resolver.release,
                        claim,
                        event=event,
                        owner_id=self.owner_id,
                        succeeded=bool(outcome.get("resolved")),
                        outcome=outcome,
                        error=str(outcome.get("reason") or ""),
                    )
                except Exception:
                    pass

    @staticmethod
    def _call_compatible(callback: Callable[..., Any], positional: Any, **kwargs: Any) -> Any:
        """Invoke callbacks while filtering optional adapter keywords."""

        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            return callback(positional, **kwargs)
        accepts_kwargs = any(
            item.kind is inspect.Parameter.VAR_KEYWORD
            for item in signature.parameters.values()
        )
        filtered = kwargs if accepts_kwargs else {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
        positional_parameters = [
            item
            for item in signature.parameters.values()
            if item.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if positional_parameters:
            return callback(positional, **filtered)
        return callback(**filtered)

    def _settle_recovered_callback_integration(
        self,
        request: MergeRequest,
        *,
        canonical: str,
        candidate: str,
        target: str,
        previous: Mapping[str, Any],
        receipt_key: str,
        preflight_receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Settle a prior callback mutation without invoking it again."""

        prior_target = str(previous.get("target_commit") or "")
        result = {
            **dict(previous),
            "status": str(
                previous.get("status")
                or "integrated_pending_acceptance"
            ),
            "merged": bool(previous.get("merged")),
            "integrated": True,
            "accepted": False,
            "acceptance_pending": True,
            "completion_authoritative": False,
            "integration_terminal": True,
            "callback_owned_integration": True,
            "callback_recovered": True,
            "callback_reinvoked": False,
            "request_id": request.request_id,
            "task_id": _request_value(request, "task_id"),
            "canonical_task_id": canonical,
            "commit_sha": candidate,
            "target_branch": self.target_branch,
            "target_commit": target,
            "merge_commit": target,
            "previous_target_commit": prior_target,
            "finished_at": time.time(),
        }
        if prior_target != target:
            result.update(
                {
                    "status": "integrated_pending_acceptance",
                    "reason": "post_merge_target_changed",
                    "validation_reason": "post_merge_target_changed",
                }
            )
        if preflight_receipt is not None:
            result["preflight"] = dict(preflight_receipt)
        completion_metadata = {
            "status": str(result["status"]),
            "integrated": True,
            "accepted": False,
            "acceptance_pending": True,
            "completion_authoritative": False,
            "integration_terminal": True,
            "callback_owned_integration": True,
            "callback_recovered": True,
            "target_commit": target,
            "merge_commit": target,
            "previous_target_commit": prior_target,
            "validation_reason": str(
                result.get("validation_reason") or ""
            ),
            "post_merge_validation": dict(
                result.get("post_merge_validation") or {}
            ),
        }
        try:
            self._call_compatible(
                self.queue.complete,
                request,
                metadata=completion_metadata,
            )
            result["queue_settlement"] = {
                "status": "completed",
                "terminal": True,
            }
        except MergeQueueFenceError as exc:
            result["status"] = "integrated_pending_acceptance"
            result["reason"] = "merge_queue_claim_fenced"
            result["queue_settlement"] = {
                "status": "fenced_out",
                "terminal": False,
                "fence_stage": "recovered_callback_queue_completion",
                "fence_error": f"{type(exc).__name__}: {exc}",
            }
            self._write_receipt(
                f"fenced-{request.request_id}", result
            )
        self._write_receipt(receipt_key, result)
        return result

    def _finish_integrated_pending_validation(
        self,
        request: MergeRequest,
        *,
        canonical: str,
        candidate: str,
        target: str,
        started_at: float,
        merged: bool,
        already_merged: bool,
        validation_reason: str,
        post_merge_validation: Mapping[str, Any],
        extra: Mapping[str, Any] | None = None,
        preflight_receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Settle a callback-owned mutation while acceptance stays pending.

        A callback can mutate the target before its returned validation receipt
        is inspected.  Treating a missing, failed, or stale receipt as a merge
        failure would requeue an operation whose physical side effect already
        landed.  Persist the exact integrated state and terminally settle only
        the mutation unit; a reconciliation path can later produce fresh
        acceptance evidence without running the callback again.
        """

        result: dict[str, Any] = {
            "status": "integrated_pending_validation",
            "merged": bool(merged),
            "already_merged": bool(already_merged),
            "integrated": True,
            "accepted": False,
            "acceptance_pending": True,
            "completion_authoritative": False,
            "integration_terminal": True,
            "callback_owned_integration": True,
            "request_id": request.request_id,
            "task_id": _request_value(request, "task_id"),
            "canonical_task_id": canonical,
            "commit_sha": candidate,
            "target_branch": self.target_branch,
            "target_commit": target,
            "merge_commit": target,
            "reason": "post_merge_validation_pending",
            "validation_reason": str(validation_reason),
            "post_merge_validation": dict(post_merge_validation),
            "started_at": started_at,
            "finished_at": time.time(),
        }
        if extra:
            result.update(extra)
            # Callback internals are diagnostic only.  Preserve the stable
            # public statement that integration landed but is not accepted.
            result.update(
                {
                    "status": "integrated_pending_validation",
                    "merged": bool(merged),
                    "already_merged": bool(already_merged),
                    "integrated": True,
                    "accepted": False,
                    "acceptance_pending": True,
                    "completion_authoritative": False,
                    "integration_terminal": True,
                    "callback_owned_integration": True,
                    "target_commit": target,
                    "merge_commit": target,
                    "reason": "post_merge_validation_pending",
                    "validation_reason": str(validation_reason),
                    "post_merge_validation": dict(
                        post_merge_validation
                    ),
                }
            )
        if preflight_receipt is not None:
            result["preflight"] = dict(preflight_receipt)

        receipt_key = self._dedupe_key(canonical, candidate)
        self._write_receipt(receipt_key, result)
        completion_metadata = {
            "status": "integrated_pending_validation",
            "integrated": True,
            "accepted": False,
            "acceptance_pending": True,
            "completion_authoritative": False,
            "integration_terminal": True,
            "callback_owned_integration": True,
            "target_commit": target,
            "merge_commit": target,
            "validation_reason": str(validation_reason),
            "post_merge_validation": dict(post_merge_validation),
        }
        try:
            self._call_compatible(
                self.queue.complete,
                request,
                metadata=completion_metadata,
            )
            result["queue_settlement"] = {
                "status": "completed",
                "terminal": True,
            }
            self._write_receipt(receipt_key, result)
        except MergeQueueFenceError as exc:
            # The physical mutation remains integrated even if a newer queue
            # owner fenced this worker before settlement.  Never rewrite that
            # fact as a retryable merge failure.
            result["queue_settlement"] = {
                "status": "fenced_out",
                "terminal": False,
                "reason": "merge_queue_claim_fenced",
                "fence_error": f"{type(exc).__name__}: {exc}",
            }
            self._write_receipt(
                f"fenced-{request.request_id}", result
            )
        return result

    def _finish_success(
        self,
        request: MergeRequest,
        *,
        status: str,
        canonical: str,
        candidate: str,
        target: str,
        started_at: float,
        extra: Mapping[str, Any] | None = None,
        defer_completion: bool = False,
        preflight_receipt: Mapping[str, Any] | None = None,
        callback_owned_integration: bool = False,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": status,
            "merged": status == "merged",
            "integrated": True,
            "request_id": request.request_id,
            "task_id": _request_value(request, "task_id"),
            "canonical_task_id": canonical,
            "commit_sha": candidate,
            "target_branch": self.target_branch,
            "target_commit": target,
            "merge_commit": target,
            "started_at": started_at,
            "finished_at": time.time(),
        }
        if callback_owned_integration:
            result["callback_owned_integration"] = True
        if extra:
            result.update(extra)
            # Stable public semantics take precedence over callback internals.
            result.update({"status": status, "integrated": True})
            if callback_owned_integration:
                result["callback_owned_integration"] = True
        if preflight_receipt is not None:
            result["preflight"] = dict(preflight_receipt)
        if defer_completion:
            result.update(
                {
                    "accepted": False,
                    "acceptance_pending": True,
                }
            )
            return result
        self._runtime_completion(
            request,
            target_commit=target,
            status=status,
            evidence=(
                extra.get("post_merge_validation")
                if isinstance(extra, Mapping)
                and isinstance(extra.get("post_merge_validation"), Mapping)
                else {}
            ),
        )
        self._write_receipt(self._dedupe_key(canonical, candidate), result)
        try:
            self.queue.complete(request)
        except MergeQueueFenceError as exc:
            if callback_owned_integration:
                result.update(
                    {
                        "status": "integrated_pending_acceptance",
                        "integrated": True,
                        "accepted": False,
                        "acceptance_pending": True,
                        "completion_authoritative": False,
                        "integration_terminal": True,
                        "reason": "merge_queue_claim_fenced",
                        "queue_settlement": {
                            "status": "fenced_out",
                            "terminal": False,
                            "fence_stage": "before_queue_completion",
                            "fence_error": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                        },
                    }
                )
            else:
                result.update(
                    {
                        "status": "fenced_out",
                        "accepted": False,
                        "acceptance_pending": False,
                        "reason": "merge_queue_claim_fenced",
                        "fence_error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if callback_owned_integration:
                self._write_receipt(
                    self._dedupe_key(canonical, candidate), result
                )
            self._write_receipt(
                f"fenced-{request.request_id}", result
            )
        return result

    def _finish_failure(
        self,
        request: MergeRequest,
        *,
        reason: str,
        details: Mapping[str, Any],
        started_at: float,
        retryable: bool = True,
    ) -> dict[str, Any]:
        failures = int(getattr(request, "failure_count", 0) or 0) + 1
        exhausted = not retryable or failures >= self.max_attempts
        result = {
            "status": "quarantined" if exhausted else "retrying",
            "merged": False,
            "integrated": False,
            "accepted": False,
            "acceptance_pending": False,
            "request_id": request.request_id,
            "task_id": _request_value(request, "task_id"),
            "canonical_task_id": str(getattr(request, "canonical_identity", "") or "")
            or _request_value(request, "canonical_task_id")
            or _request_value(request, "task_id"),
            "commit_sha": _request_value(request, "commit_sha", "implementation_commit", "commit"),
            "target_branch": self.target_branch,
            "reason": reason,
            "failure_count": failures,
            "max_attempts": self.max_attempts,
            "retryable": retryable,
            "started_at": started_at,
            "finished_at": time.time(),
            **dict(details),
        }
        try:
            if exhausted:
                quarantine = getattr(self.queue, "quarantine", None)
                if quarantine is not None:
                    self._call_queue_failure(
                        quarantine, request, reason, result
                    )
                else:
                    self._call_queue_failure(
                        self.queue.fail,
                        request,
                        reason,
                        result,
                        retryable=False,
                    )
                self._write_receipt(
                    f"quarantine-{request.request_id}", result
                )
            else:
                requeue = getattr(self.queue, "requeue")
                self._call_queue_failure(
                    requeue, request, reason, result
                )
        except MergeQueueFenceError as exc:
            result.update(
                {
                    "status": "fenced_out",
                    "reason": "merge_queue_claim_fenced",
                    "fence_error": f"{type(exc).__name__}: {exc}",
                }
            )
            self._write_receipt(
                f"fenced-{request.request_id}", result
            )
        return result

    @staticmethod
    def _call_queue_failure(
        callback: Callable[..., Any],
        request: MergeRequest,
        reason: str,
        receipt: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        try:
            signature = inspect.signature(callback)
            supported = signature.parameters
            call_kwargs: dict[str, Any] = {}
            if "reason" in supported:
                call_kwargs["reason"] = reason
            if "receipt" in supported:
                call_kwargs["receipt"] = dict(receipt)
            if "details" in supported:
                call_kwargs["details"] = dict(receipt)
            call_kwargs.update({key: value for key, value in kwargs.items() if key in supported})
            callback(request, **call_kwargs)
        except (TypeError, ValueError):
            callback(request, reason)

    def _target_commit(self) -> str:
        result = self._git("rev-parse", "--verify", f"refs/heads/{self.target_branch}^{{commit}}")
        return result.stdout.strip() if result.returncode == 0 else ""

    def _decision_runtime_cancelled(self) -> bool:
        value = self.decision_runtime_cancellation
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if callable(value):
            return bool(value())
        checker = getattr(value, "is_set", None)
        if callable(checker):
            return bool(checker())
        raise TypeError(
            "decision_runtime_cancellation must be a boolean, predicate, "
            "event, or None"
        )

    def _runtime_decision(
        self, boundary: str, payload: Mapping[str, Any]
    ) -> Any:
        if self._decision_runtime_cancelled():
            from ..context.decision_runtime import DecisionRuntimeCancelled

            raise DecisionRuntimeCancelled(
                ("cancelled", f"cancelled_before_{boundary}")
            )
        if self.decision_runtime is None:
            return None
        route = getattr(self.decision_runtime, "route", None)
        if not callable(route):
            raise TypeError("decision_runtime must expose route()")
        return route(boundary, dict(payload))

    def _runtime_mutation(
        self,
        boundary: str,
        payload: Mapping[str, Any],
        callback: Callable[[], Any],
    ) -> Any:
        decision = self._runtime_decision(boundary, payload)
        if decision is None:
            return callback()
        authorize = getattr(
            self.decision_runtime, "authorize_mutation", None
        )
        if not callable(authorize):
            raise TypeError(
                "decision_runtime must expose authorize_mutation()"
            )

        def dispatch() -> dict[str, Any]:
            value = callback()
            request = getattr(decision, "decision_request", None)
            return {
                "value": value,
                "observed_effects": tuple(
                    getattr(request, "expected_effects", ())
                ),
            }

        execution = authorize(decision, dispatch)
        self._last_merge_runtime_decision = decision
        self._last_merge_effect_observation = getattr(
            execution, "effect_observation", None
        )
        wrapped = getattr(execution, "value", execution)
        return wrapped.get("value") if isinstance(wrapped, Mapping) else wrapped

    def _runtime_completion(
        self,
        request: MergeRequest,
        *,
        target_commit: str,
        status: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> Any:
        """Require a fresh merged-tree completion decision before queue state."""

        return self._runtime_decision(
            "completion",
            {
                "request_id": request.request_id,
                "task_id": _request_value(request, "task_id"),
                "target_branch": self.target_branch,
                "merged_commit": target_commit,
                "status": status,
                "merge_decision_id": str(
                    getattr(
                        getattr(
                            self._last_merge_runtime_decision, "receipt", None
                        ),
                        "receipt_id",
                        "",
                    )
                ),
                "effect_observation_receipt_id": str(
                    getattr(
                        self._last_merge_effect_observation, "receipt_id", ""
                    )
                ),
                "post_merge_evidence": dict(evidence or {}),
                "fresh_merged_tree_required": True,
            },
        )

    def _advance_target(
        self,
        rebased_commit: str,
        *,
        expected_target: str,
    ) -> subprocess.CompletedProcess[str]:
        """Fast-forward the target without leaving a checked-out tree stale."""

        def advance() -> subprocess.CompletedProcess[str]:
            target_worktree = self._target_worktree()
            if target_worktree is None:
                return self._git(
                    "update-ref",
                    f"refs/heads/{self.target_branch}",
                    rebased_commit,
                    expected_target,
                )
            status = self._git(
                "status",
                "--porcelain",
                "--untracked-files=normal",
                cwd=target_worktree,
            )
            if status.returncode != 0:
                return status
            if status.stdout.strip():
                return subprocess.CompletedProcess(
                    ["git", "status", "--porcelain"],
                    2,
                    stdout=status.stdout,
                    stderr=f"target worktree is dirty: {target_worktree}",
                )
            current = self._git("rev-parse", "HEAD", cwd=target_worktree)
            if current.returncode != 0:
                return current
            if current.stdout.strip() != expected_target:
                return subprocess.CompletedProcess(
                    ["git", "rev-parse", "HEAD"],
                    3,
                    stdout=current.stdout,
                    stderr="target advanced while candidate was rebased",
                )
            return self._git(
                "merge", "--ff-only", rebased_commit, cwd=target_worktree
            )

        return self._runtime_mutation(
            "merge",
            {
                "target_branch": self.target_branch,
                "expected_target": expected_target,
                "rebased_commit": rebased_commit,
                "operation": "advance_target",
            },
            advance,
        )

    def _target_worktree(self) -> Path | None:
        """Return the worktree currently holding the target branch, if any."""

        result = self._git("worktree", "list", "--porcelain")
        if result.returncode != 0:
            return None
        path: Path | None = None
        for line in [*result.stdout.splitlines(), ""]:
            if line.startswith("worktree "):
                path = Path(line.removeprefix("worktree ").strip())
            elif line == f"branch refs/heads/{self.target_branch}" and path is not None:
                return path
            elif not line:
                path = None
        return None

    def _dequeue(self) -> MergeRequest | None:
        """Claim with a consumer id when supported by the queue implementation."""

        try:
            signature = inspect.signature(self.queue.dequeue)
            if "consumer_id" in signature.parameters:
                return self.queue.dequeue(consumer_id=self.owner_id)
        except (TypeError, ValueError):
            pass
        return self.queue.dequeue()

    def _owns_claim(self, request: MergeRequest) -> bool:
        """Check the durable queue fence immediately before side effects."""

        owns_claim = getattr(self.queue, "owns_claim", None)
        if not callable(owns_claim):
            # Compatibility queues without fencing still remain protected by
            # the repo-wide consumer lease.
            return True
        try:
            signature = inspect.signature(owns_claim)
            if "consumer_id" in signature.parameters:
                return bool(
                    owns_claim(request, consumer_id=self.owner_id)
                )
        except (TypeError, ValueError):
            pass
        return bool(owns_claim(request))

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return self._git("merge-base", "--is-ancestor", ancestor, descendant).returncode == 0

    def _dedupe_key(self, canonical: str, commit: str) -> str:
        parts = [canonical, commit]
        queue_repository_id = str(
            getattr(self.queue, "target_repository_id", "") or ""
        ).strip()
        queue_target_branch = str(
            getattr(self.queue, "target_branch", "") or ""
        ).strip()
        if queue_repository_id and queue_target_branch:
            parts.extend((queue_repository_id, queue_target_branch))
        return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()

    def _receipt_path(self, key: str) -> Path:
        safe = "".join(character for character in key if character.isalnum() or character in "-_")
        return self.receipt_dir / f"{safe[:180]}.json"

    def _read_receipt(self, key: str) -> dict[str, Any]:
        try:
            payload = json.loads(self._receipt_path(key).read_text(encoding="utf-8"))
            return dict(payload) if isinstance(payload, dict) else {}
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}

    def _write_receipt(self, key: str, payload: Mapping[str, Any]) -> None:
        path = self._receipt_path(key)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(
                json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    def _git(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=cwd or self.repo_root,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.git_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                ["git", *args],
                124,
                stdout=str(exc.stdout or ""),
                stderr=f"git command timed out after {self.git_timeout_seconds}s: {exc.stderr or ''}",
            )

    @staticmethod
    def _command_failure(
        reason: str,
        command: subprocess.CompletedProcess[str],
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "merged": False,
            "retryable": True,
            "reason": reason,
            "returncode": command.returncode,
            "stdout": str(command.stdout or "")[-4000:],
            "stderr": str(command.stderr or "")[-4000:],
            **extra,
        }


__all__ = [
    "MergeCallback",
    "MergeTrain",
    "PARALLEL_ACCEPTANCE_EVIDENCE_ID",
    "PARALLEL_ACCEPTANCE_RECEIPT_SCHEMA",
    "PARALLEL_ACCEPTANCE_THROUGHPUT_SCHEMA",
    "PARALLEL_EXECUTION_ACCEPTANCE_CRITERIA",
    "PARALLEL_EXECUTION_COMPLETION_ANALYZER_VERSION",
    "PARALLEL_EXECUTION_COMPLETION_CONFIGURATION_REVISION",
    "PARALLEL_EXECUTION_OBJECTIVE_ID",
    "PARALLEL_EXECUTION_OBJECTIVE_REVISION",
    "PARALLEL_EXECUTION_PRODUCING_TASK_IDS",
    "PARALLEL_EXECUTION_REQUIRED_EXHAUSTIVE_RECEIPTS",
    "ParallelAcceptanceReceipt",
    "PostMergeEvidenceCallback",
    "PostMergeValidationCallback",
    "PreflightCallback",
    "conflict_fingerprint",
    "evaluate_parallel_execution_completion",
]
