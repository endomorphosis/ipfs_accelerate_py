"""Federation merge train over isolated worktrees and proof-release gates.

This coordinator does not open Git, does not open ``control.duckdb``, and does
not call the existing merge-queue DuckDB path. It binds isolated worktrees,
compiles an explicit ordinal train, and releases a merge only after proof,
test, and optional seal projections satisfy the release gate. DuckLake never
admits a merge. The current fence wins.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ..task_sources.control_plane_contracts import content_identity
from ..task_sources.quack_state_client import QuackStateClient, StatementKind
from .causal_graph import CausalGraphCommit, CausalGraphError
from .contracts import (
    FederationAuthorityError,
    FederationBinding,
    FederationContractError,
    _identifier,
    _integer,
    _strings,
)
from .deduplication import WRITE_EFFECT_CLASSES
from .events import EventEffectClass
from .proof_projection import (
    PROOF_STATUSES,
    TEST_STATUSES,
    projection_establishes_authority,
    projection_establishes_completion,
)
from .rebalancing import RebalancingStore
from .registry import _template
from .retrieval_projection import retrieval_establishes_authority
from .work_stealing import POLICY_GATES

CLOSED_EFFECT_CLASSES = frozenset(item.value for item in EventEffectClass)
WRITE_EFFECT_VALUES = frozenset(item.value for item in WRITE_EFFECT_CLASSES)
MERGE_OUTCOMES = frozenset({"merged", "failed", "blocked"})
MAX_MERGE_TRAIN = 1_024


class MergeError(CausalGraphError):
    """Base typed federation-merge failure."""


class MergeAuthorityError(FederationAuthorityError, MergeError):
    """An attempt to share exclusive worktrees, skip order, or bypass proof gates."""


def refuse_ducklake_merge_authority(receipt: Mapping[str, Any] | None) -> None:
    if not receipt:
        return
    if (
        receipt.get("authoritative") is True
        or receipt.get("schedules") is True
        or receipt.get("steals") is True
        or receipt.get("rebalances") is True
        or receipt.get("merges") is True
    ):
        raise MergeAuthorityError("DuckLake cannot admit a federation merge")


def _effect_class(value: EventEffectClass | str, name: str = "effect_class") -> str:
    if isinstance(value, EventEffectClass):
        return value.value
    text = _identifier(value, name)
    if text not in CLOSED_EFFECT_CLASSES:
        raise FederationContractError("effect_class is not closed")
    return text


def _write_like(effect_class: str) -> bool:
    return effect_class in WRITE_EFFECT_VALUES


def _reject_filesystem_path(value: str, name: str) -> None:
    if value.startswith(("/", "~")) or ".." in value.split("/"):
        raise MergeAuthorityError(
            f"{name} accepts server-resolved worktree identities, not filesystem paths"
        )


@dataclass(frozen=True)
class WorktreeBinding:
    """Exact isolated worktree occupancy bound to one repository tree."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/worktree-binding@1"
    )

    worktree_id: str
    repository_id: str
    tree_id: str
    owner_session_id: str
    head_commit_id: str
    branch_name: str
    fencing_epoch: int
    isolated: bool = True
    exclusive: bool = True

    def __post_init__(self) -> None:
        for name in (
            "worktree_id",
            "repository_id",
            "tree_id",
            "owner_session_id",
            "head_commit_id",
            "branch_name",
        ):
            _reject_filesystem_path(str(getattr(self, name)), name)
            _identifier(getattr(self, name), name)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        if type(self.isolated) is not bool or type(self.exclusive) is not bool:
            raise FederationContractError("isolated and exclusive must be boolean")
        if self.exclusive and self.isolated is not True:
            raise MergeAuthorityError("exclusive merge work requires an isolated worktree")


def bind_worktree(
    *,
    binding: FederationBinding,
    worktree_id: str,
    owner_session_id: str,
    head_commit_id: str,
    branch_name: str,
    fencing_epoch: int = 1,
    isolated: bool = True,
    exclusive: bool = True,
) -> WorktreeBinding:
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    refuse_ducklake_merge_authority(None)
    bound = WorktreeBinding(
        worktree_id=worktree_id,
        repository_id=binding.repository_ids[0],
        tree_id=binding.repository_tree_ids[0],
        owner_session_id=owner_session_id,
        head_commit_id=head_commit_id,
        branch_name=branch_name,
        fencing_epoch=fencing_epoch,
        isolated=isolated,
        exclusive=exclusive,
    )
    if bound.tree_id not in binding.repository_tree_ids:
        raise MergeAuthorityError("worktree tree identity mismatches")
    if bound.repository_id not in binding.repository_ids:
        raise MergeAuthorityError("worktree repository is not bound")
    return bound


@dataclass(frozen=True)
class MergeCandidate:
    """One worktree-bound task considered for the merge train."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/merge-candidate@1"
    )

    task_id: str
    worktree: WorktreeBinding
    merge_lane: str
    source_branch: str
    target_branch: str
    effect_class: str
    fencing_epoch: int
    predecessor_task_ids: tuple[str, ...] = ()
    requires_proof: bool = True
    requires_test: bool = True
    requires_seal: bool = False
    proof_status: str = "open"
    test_status: str = "pending"
    sealed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.worktree, WorktreeBinding):
            raise FederationContractError("merge candidate requires WorktreeBinding")
        for name in ("task_id", "merge_lane", "source_branch", "target_branch"):
            _identifier(getattr(self, name), name)
        object.__setattr__(self, "effect_class", _effect_class(self.effect_class))
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        _strings(self.predecessor_task_ids, "predecessor_task_ids", maximum=MAX_MERGE_TRAIN)
        for name in ("requires_proof", "requires_test", "requires_seal", "sealed"):
            if type(getattr(self, name)) is not bool:
                raise FederationContractError(f"{name} must be boolean")
        proof_status = _identifier(self.proof_status, "proof_status")
        if proof_status not in PROOF_STATUSES:
            raise FederationContractError("proof_status is not closed")
        object.__setattr__(self, "proof_status", proof_status)
        test_status = _identifier(self.test_status, "test_status")
        if test_status not in TEST_STATUSES:
            raise FederationContractError("test_status is not closed")
        object.__setattr__(self, "test_status", test_status)
        if self.fencing_epoch != self.worktree.fencing_epoch:
            raise MergeAuthorityError("candidate fencing epoch differs from its worktree")
        if _write_like(self.effect_class) and self.worktree.exclusive is not True:
            raise MergeAuthorityError("exclusive merge work requires an isolated worktree")


@dataclass(frozen=True)
class MergeTrainEntry:
    """One ordinal slot in the compiled merge train."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/merge-train-entry@1"
    )

    entry_id: str
    task_id: str
    worktree_id: str
    ordinal: int
    source_branch: str
    target_branch: str
    status: str

    def __post_init__(self) -> None:
        for name in (
            "entry_id",
            "task_id",
            "worktree_id",
            "source_branch",
            "target_branch",
            "status",
        ):
            _identifier(getattr(self, name), name)
        _integer(self.ordinal, "ordinal", minimum=1)


@dataclass(frozen=True)
class CompiledMergeTrain:
    """Explicit, isolated merge order for one lane and fencing epoch."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/compiled-merge-train@1"
    )

    lane_id: str
    fencing_epoch: int
    merge_order: tuple[str, ...]
    entries: tuple[MergeTrainEntry, ...]
    worktrees: tuple[WorktreeBinding, ...]
    candidates: tuple[MergeCandidate, ...]

    def __post_init__(self) -> None:
        _identifier(self.lane_id, "lane_id")
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        _strings(self.merge_order, "merge_order", maximum=MAX_MERGE_TRAIN, required=True)
        if not isinstance(self.entries, tuple) or not all(
            isinstance(item, MergeTrainEntry) for item in self.entries
        ):
            raise FederationContractError("entries must be MergeTrainEntry records")
        if not isinstance(self.worktrees, tuple) or not all(
            isinstance(item, WorktreeBinding) for item in self.worktrees
        ):
            raise FederationContractError("worktrees must be WorktreeBinding records")
        if not isinstance(self.candidates, tuple) or not all(
            isinstance(item, MergeCandidate) for item in self.candidates
        ):
            raise FederationContractError("candidates must be MergeCandidate records")
        if len(self.entries) != len(self.merge_order) or len(self.candidates) != len(self.entries):
            raise MergeAuthorityError("merge train order diverges from its entries")
        entry_tasks = tuple(item.task_id for item in self.entries)
        if entry_tasks != self.merge_order:
            raise MergeAuthorityError("merge train order diverges from its entries")

    @property
    def train_id(self) -> str:
        return "merge-train:" + self.cid

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "lane_id": self.lane_id,
                "fencing_epoch": self.fencing_epoch,
                "merge_order": list(self.merge_order),
                "worktree_ids": [item.worktree_id for item in self.worktrees],
            }
        )

    def candidate(self, task_id: str) -> MergeCandidate:
        for item in self.candidates:
            if item.task_id == task_id:
                return item
        raise MergeError("merge candidate is absent from the train")

    def entry(self, task_id: str) -> MergeTrainEntry:
        for item in self.entries:
            if item.task_id == task_id:
                return item
        raise MergeError("merge entry is absent from the train")


@dataclass(frozen=True)
class MergeReceipt:
    """Evidence that one train entry passed proof gates and occupied a merge slot."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/merge-receipt@1"
    )

    train_id: str
    entry_id: str
    task_id: str
    worktree_id: str
    outcome: str
    previous_fencing_epoch: int
    fencing_epoch: int
    result_commit_id: str
    proof_ref: str
    tree_id: str

    def __post_init__(self) -> None:
        for name in (
            "train_id",
            "entry_id",
            "task_id",
            "worktree_id",
            "result_commit_id",
            "proof_ref",
            "tree_id",
        ):
            _identifier(getattr(self, name), name)
        outcome = _identifier(self.outcome, "outcome")
        if outcome not in MERGE_OUTCOMES:
            raise MergeAuthorityError("merge outcome is outside its closed vocabulary")
        _integer(self.previous_fencing_epoch, "previous_fencing_epoch", minimum=1)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        if self.fencing_epoch <= self.previous_fencing_epoch:
            raise MergeAuthorityError("merge must increment the fencing epoch")

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "train_id": self.train_id,
                "entry_id": self.entry_id,
                "task_id": self.task_id,
                "outcome": self.outcome,
                "fencing_epoch": self.fencing_epoch,
                "result_commit_id": self.result_commit_id,
            }
        )


def _order_candidates(candidates: Sequence[MergeCandidate]) -> tuple[MergeCandidate, ...]:
    by_id: dict[str, MergeCandidate] = {}
    for item in candidates:
        if item.task_id in by_id:
            raise MergeAuthorityError("merge train contains duplicate task identities")
        by_id[item.task_id] = item
    remaining = {item.task_id: set(item.predecessor_task_ids) for item in candidates}
    for preds in remaining.values():
        unknown = preds - set(by_id)
        if unknown:
            raise MergeAuthorityError("merge predecessor is absent from the train")
    ordered: list[MergeCandidate] = []
    while remaining:
        ready = sorted(task_id for task_id, preds in remaining.items() if not preds)
        if not ready:
            raise MergeAuthorityError("merge train predecessors form a cycle")
        current = ready[0]
        ordered.append(by_id[current])
        del remaining[current]
        for preds in remaining.values():
            preds.discard(current)
    return tuple(ordered)


def _assert_isolated_worktrees(candidates: Sequence[MergeCandidate]) -> None:
    occupancy: dict[str, list[MergeCandidate]] = {}
    for item in candidates:
        occupancy.setdefault(item.worktree.worktree_id, []).append(item)
    for items in occupancy.values():
        exclusive = tuple(
            item
            for item in items
            if item.worktree.exclusive or _write_like(item.effect_class)
        )
        if exclusive and len(items) > 1:
            raise MergeAuthorityError("isolated worktrees cannot overlap exclusive work")


def _require_merge_capability(capability_refs: Sequence[str]) -> None:
    refs = _strings(capability_refs, "capability_refs", maximum=256, required=True)
    if POLICY_GATES["merge"] not in refs:
        raise MergeAuthorityError("merge capability is missing")


def _assert_proof_release(candidate: MergeCandidate) -> None:
    if projection_establishes_completion() is not False:
        raise MergeAuthorityError("proof projections cannot complete a merge")
    if projection_establishes_authority() is not False:
        raise MergeAuthorityError("proof projections cannot mint merge authority")
    if candidate.requires_proof and candidate.proof_status != "proved":
        raise MergeAuthorityError("proof-release gate requires a proved obligation")
    if candidate.requires_test and candidate.test_status != "passed":
        raise MergeAuthorityError("proof-release gate requires a passed test")
    if candidate.requires_seal and candidate.sealed is not True:
        raise MergeAuthorityError("proof-release gate requires a current seal")


def compile_merge_train(
    candidates: Sequence[MergeCandidate],
    *,
    binding: FederationBinding,
    capability_refs: Sequence[str],
    expected_fence: int,
    merge_slots: int = 1,
    lane_id: str = "merge-lane:default",
    ducklake_receipt: Mapping[str, Any] | None = None,
) -> CompiledMergeTrain:
    """Compile an explicit isolated merge order. Missing capability fails closed."""

    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    refuse_ducklake_merge_authority(ducklake_receipt)
    if retrieval_establishes_authority() is not False:
        raise MergeAuthorityError("retrieval cannot mint a federation merge")
    _require_merge_capability(capability_refs)
    _integer(merge_slots, "merge_slots", minimum=1)
    if not candidates:
        raise FederationContractError("merge train requires at least one candidate")
    if len(candidates) > MAX_MERGE_TRAIN:
        raise FederationContractError("merge train exceeds admitted size")
    for item in candidates:
        if not isinstance(item, MergeCandidate):
            raise FederationContractError("candidates must be MergeCandidate records")
        if item.worktree.tree_id not in binding.repository_tree_ids:
            raise MergeAuthorityError("merge requires current tree identity")
        if item.worktree.repository_id not in binding.repository_ids:
            raise MergeAuthorityError("merge worktree repository is not bound")
        if expected_fence != item.fencing_epoch:
            raise MergeAuthorityError("source fencing epoch is stale")
    _assert_isolated_worktrees(candidates)
    ordered = _order_candidates(candidates)
    worktrees = tuple(
        dict.fromkeys(item.worktree for item in ordered)
    )
    entries = tuple(
        MergeTrainEntry(
            entry_id="merge-entry:"
            + content_identity(
                {
                    "task_id": item.task_id,
                    "worktree_id": item.worktree.worktree_id,
                    "ordinal": ordinal,
                    "lane_id": lane_id,
                }
            ),
            task_id=item.task_id,
            worktree_id=item.worktree.worktree_id,
            ordinal=ordinal,
            source_branch=item.source_branch,
            target_branch=item.target_branch,
            status="queued",
        )
        for ordinal, item in enumerate(ordered, start=1)
    )
    return CompiledMergeTrain(
        lane_id=_identifier(lane_id, "lane_id"),
        fencing_epoch=expected_fence,
        merge_order=tuple(item.task_id for item in ordered),
        entries=entries,
        worktrees=worktrees,
        candidates=ordered,
    )


def release_merge(
    train: CompiledMergeTrain,
    task_id: str,
    *,
    binding: FederationBinding,
    expected_fence: int,
    result_commit_id: str,
    merged_task_ids: Sequence[str] = (),
    ducklake_receipt: Mapping[str, Any] | None = None,
) -> MergeReceipt:
    """Release the next explicit train entry after proof gates pass."""

    if not isinstance(train, CompiledMergeTrain):
        raise FederationContractError("compiled merge train is required")
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    refuse_ducklake_merge_authority(ducklake_receipt)
    if retrieval_establishes_authority() is not False:
        raise MergeAuthorityError("retrieval cannot mint a federation merge")
    task_id = _identifier(task_id, "task_id")
    candidate = train.candidate(task_id)
    entry = train.entry(task_id)
    if expected_fence != train.fencing_epoch or expected_fence != candidate.fencing_epoch:
        raise MergeAuthorityError("source fencing epoch is stale")
    if candidate.worktree.tree_id not in binding.repository_tree_ids:
        raise MergeAuthorityError("merge requires current tree identity")
    merged = set(_strings(merged_task_ids, "merged_task_ids", maximum=MAX_MERGE_TRAIN))
    if task_id in merged:
        raise MergeAuthorityError("merge train assigns a task to more than one owner")
    unfinished = tuple(
        predecessor
        for predecessor in candidate.predecessor_task_ids
        if predecessor not in merged
    )
    if unfinished:
        raise MergeAuthorityError("explicit merge order requires predecessors to merge first")
    position = train.merge_order.index(task_id)
    prior = train.merge_order[:position]
    if any(item not in merged for item in prior):
        raise MergeAuthorityError("explicit merge order requires predecessors to merge first")
    _assert_proof_release(candidate)
    result_commit_id = _identifier(result_commit_id, "result_commit_id")
    return MergeReceipt(
        train_id=train.train_id,
        entry_id=entry.entry_id,
        task_id=task_id,
        worktree_id=entry.worktree_id,
        outcome="merged",
        previous_fencing_epoch=candidate.fencing_epoch,
        fencing_epoch=candidate.fencing_epoch + 1,
        result_commit_id=result_commit_id,
        proof_ref=candidate.task_id if not candidate.requires_proof else "proof:" + candidate.task_id,
        tree_id=candidate.worktree.tree_id,
    )


class FederationMergeCoordinator:
    """Bind isolated worktrees, compile the train, and release proof-gated merges."""

    def bind_worktree(
        self,
        *,
        binding: FederationBinding,
        worktree_id: str,
        owner_session_id: str,
        head_commit_id: str,
        branch_name: str,
        fencing_epoch: int = 1,
        isolated: bool = True,
        exclusive: bool = True,
    ) -> WorktreeBinding:
        return bind_worktree(
            binding=binding,
            worktree_id=worktree_id,
            owner_session_id=owner_session_id,
            head_commit_id=head_commit_id,
            branch_name=branch_name,
            fencing_epoch=fencing_epoch,
            isolated=isolated,
            exclusive=exclusive,
        )

    def compile(
        self,
        candidates: Sequence[MergeCandidate],
        *,
        binding: FederationBinding,
        capability_refs: Sequence[str],
        expected_fence: int,
        merge_slots: int = 1,
        lane_id: str = "merge-lane:default",
        ducklake_receipt: Mapping[str, Any] | None = None,
    ) -> CompiledMergeTrain:
        return compile_merge_train(
            candidates,
            binding=binding,
            capability_refs=capability_refs,
            expected_fence=expected_fence,
            merge_slots=merge_slots,
            lane_id=lane_id,
            ducklake_receipt=ducklake_receipt,
        )

    def release(
        self,
        train: CompiledMergeTrain,
        task_id: str,
        *,
        binding: FederationBinding,
        expected_fence: int,
        result_commit_id: str,
        merged_task_ids: Sequence[str] = (),
        ducklake_receipt: Mapping[str, Any] | None = None,
    ) -> MergeReceipt:
        return release_merge(
            train,
            task_id,
            binding=binding,
            expected_fence=expected_fence,
            result_commit_id=result_commit_id,
            merged_task_ids=merged_task_ids,
            ducklake_receipt=ducklake_receipt,
        )


def _merge_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_insert_worktree",
            """
            INSERT INTO worktrees (
                worktree_id, repository_id, path, head_commit_id, branch_name,
                owner_session_id, status, created_at, updated_at, revision,
                fence_epoch
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "worktree_id",
                "repository_id",
                "path",
                "head_commit_id",
                "branch_name",
                "owner_session_id",
                "status",
                "created_at",
                "updated_at",
                "revision",
                "fence_epoch",
            ),
        ),
        _template(
            "casf_select_worktree",
            """
            SELECT worktree_id, repository_id, path, status, fence_epoch, revision
            FROM worktrees
            WHERE worktree_id = ?
            LIMIT 1
            """,
            ("worktree_id",),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_merge_queue_entry",
            """
            INSERT INTO merge_queue_entries (
                entry_id, repository_id, worktree_id, task_cid, source_branch,
                target_branch, status, ordinal, enqueued_at, updated_at,
                revision, fence_epoch
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "entry_id",
                "repository_id",
                "worktree_id",
                "task_cid",
                "source_branch",
                "target_branch",
                "status",
                "ordinal",
                "enqueued_at",
                "updated_at",
                "revision",
                "fence_epoch",
            ),
        ),
        _template(
            "casf_select_merge_queue_entry",
            """
            SELECT entry_id, worktree_id, task_cid, status, ordinal, fence_epoch
            FROM merge_queue_entries
            WHERE entry_id = ?
            LIMIT 1
            """,
            ("entry_id",),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_merge_attempt",
            """
            INSERT INTO merge_attempts (
                merge_attempt_id, entry_id, task_cid, worktree_id, started_at,
                finished_at, status, result_commit_id, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "merge_attempt_id",
                "entry_id",
                "task_cid",
                "worktree_id",
                "started_at",
                "finished_at",
                "status",
                "result_commit_id",
                "body_json",
            ),
        ),
        _template(
            "casf_select_merge_attempt",
            """
            SELECT merge_attempt_id, entry_id, task_cid, worktree_id, status,
                   result_commit_id
            FROM merge_attempts
            WHERE merge_attempt_id = ?
            LIMIT 1
            """,
            ("merge_attempt_id",),
            kind=StatementKind.QUERY,
        ),
    )


class MergeStore(RebalancingStore):
    """Persist worktree bindings, merge-queue entries, and merge attempts."""

    INTERFACE = "MergeStore@1"

    def __init__(
        self,
        client: QuackStateClient,
        *,
        event_notifier: Callable[[int], None] | None = None,
        outbox_notifier: Callable[[int], None] | None = None,
        test_failure_hook: Callable[[str], None] | None = None,
        require_quack_authority: bool = False,
    ) -> None:
        if isinstance(client, (str, bytes, Path)):
            raise MergeError("merge store never accepts a database path")
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise MergeError("merge store requires an already-attached typed state client")
        registered = set(client.list_templates())
        missing = [
            template.name for template in _merge_templates() if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise MergeError("merge templates are absent from the sealed catalog")
        else:
            for template in _merge_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    def record_train(
        self,
        train: CompiledMergeTrain,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
    ) -> CausalGraphCommit:
        if not isinstance(train, CompiledMergeTrain):
            raise FederationContractError("compiled merge train is required")
        return self._commit_fact(
            operation="federation.merge.train.record",
            fact_id=train.train_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(
                dict.fromkeys(
                    (train.train_id, *(item.entry_id for item in train.entries))
                )
            ),
            payload_ref=train.cid,
            prepare_fact=lambda: None,
            apply_fact=lambda revision, recorded_at: self._insert_train(
                train,
                federation_id=federation_id,
                repository_id=binding.repository_ids[0],
                graph_revision=revision,
                recorded_at=recorded_at,
            ),
        )

    def record_attempt(
        self,
        receipt: MergeReceipt,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
    ) -> CausalGraphCommit:
        if not isinstance(receipt, MergeReceipt):
            raise FederationContractError("merge receipt is required")
        attempt_id = "merge-attempt:" + receipt.cid
        return self._commit_fact(
            operation="federation.merge.attempt.record",
            fact_id=attempt_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(dict.fromkeys((attempt_id, receipt.entry_id, receipt.task_id))),
            payload_ref=receipt.cid,
            prepare_fact=lambda: None,
            apply_fact=lambda revision, recorded_at: self._insert_attempt(
                receipt,
                attempt_id=attempt_id,
                recorded_at=recorded_at,
            ),
        )

    def load_worktree(self, *, worktree_id: str) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_worktree",
            {"worktree_id": _identifier(worktree_id, "worktree_id")},
        )
        if len(rows) != 1:
            raise MergeError("worktree is absent")
        return dict(rows[0])

    def load_entry(self, *, entry_id: str) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_merge_queue_entry",
            {"entry_id": _identifier(entry_id, "entry_id")},
        )
        if len(rows) != 1:
            raise MergeError("merge queue entry is absent")
        return dict(rows[0])

    def load_attempt(self, *, attempt_id: str) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_merge_attempt",
            {"merge_attempt_id": _identifier(attempt_id, "attempt_id")},
        )
        if len(rows) != 1:
            raise MergeError("merge attempt is absent")
        return dict(rows[0])

    def _insert_train(
        self,
        train: CompiledMergeTrain,
        *,
        federation_id: str,
        repository_id: str,
        graph_revision: int,
        recorded_at: str,
    ) -> None:
        del federation_id
        for worktree in train.worktrees:
            self._client.execute(
                "casf_insert_worktree",
                {
                    "worktree_id": worktree.worktree_id,
                    "repository_id": worktree.repository_id,
                    "path": worktree.worktree_id,
                    "head_commit_id": worktree.head_commit_id,
                    "branch_name": worktree.branch_name,
                    "owner_session_id": worktree.owner_session_id,
                    "status": "isolated" if worktree.isolated else "shared",
                    "created_at": recorded_at,
                    "updated_at": recorded_at,
                    "revision": graph_revision,
                    "fence_epoch": worktree.fencing_epoch,
                },
            )
        for index, entry in enumerate(train.entries):
            candidate = train.candidates[index]
            self._client.execute(
                "casf_insert_merge_queue_entry",
                {
                    "entry_id": entry.entry_id,
                    "repository_id": repository_id,
                    "worktree_id": entry.worktree_id,
                    "task_cid": candidate.task_id,
                    "source_branch": entry.source_branch,
                    "target_branch": entry.target_branch,
                    "status": entry.status,
                    "ordinal": entry.ordinal,
                    "enqueued_at": recorded_at,
                    "updated_at": recorded_at,
                    "revision": graph_revision,
                    "fence_epoch": train.fencing_epoch,
                },
            )

    def _insert_attempt(
        self,
        receipt: MergeReceipt,
        *,
        attempt_id: str,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_merge_attempt",
            {
                "merge_attempt_id": attempt_id,
                "entry_id": receipt.entry_id,
                "task_cid": receipt.task_id,
                "worktree_id": receipt.worktree_id,
                "started_at": recorded_at,
                "finished_at": recorded_at,
                "status": receipt.outcome,
                "result_commit_id": receipt.result_commit_id,
                "body_json": json.dumps(
                    {"content_ref": receipt.cid},
                    separators=(",", ":"),
                ),
            },
        )


__all__ = (
    "CompiledMergeTrain",
    "FederationMergeCoordinator",
    "MERGE_OUTCOMES",
    "MergeAuthorityError",
    "MergeCandidate",
    "MergeError",
    "MergeReceipt",
    "MergeStore",
    "MergeTrainEntry",
    "WorktreeBinding",
    "bind_worktree",
    "compile_merge_train",
    "refuse_ducklake_merge_authority",
    "release_merge",
)
