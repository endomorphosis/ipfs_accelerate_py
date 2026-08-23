"""Fenced shard rebalancing with freeze, drain, transfer, and activate.

A rebalance freezes the source assignment, stops new claims, drains movable
work while preserving attempt, checkpoint, and cursor identities, increments
the fencing epoch, and activates the target assignment. Active irreversible
effects stay put. DuckLake, stale fences, missing capability, and policy
bypass fail closed. The current fence wins.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ..task_sources.control_plane_contracts import content_identity
from ..task_sources.quack_state_client import QuackStateClient, StatementKind
from .budgets import BudgetDimensionName, HierarchicalBudgetLedger
from .causal_graph import CausalGraphCommit, CausalGraphError
from .contracts import (
    FederationAuthorityError,
    FederationBinding,
    FederationContractError,
    _identifier,
    _integer,
    _strings,
)
from .parallel_frontier import IRREVERSIBLE_EFFECT_CLASSES
from .registry import _template
from .retrieval_projection import retrieval_establishes_authority
from .sharding import (
    ShardWork,
    SupervisorSpecializationBound,
)
from .work_stealing import POLICY_GATES, WorkStealingStore

IRREVERSIBLE_VALUES = frozenset(item.value for item in IRREVERSIBLE_EFFECT_CLASSES)
REBALANCE_OUTCOMES = frozenset({"failed", "rebalanced", "rolled_back"})
MAX_REBALANCE_TASKS = 10_000


class RebalancingError(CausalGraphError):
    """Base typed shard-rebalancing failure."""


class RebalancingAuthorityError(FederationAuthorityError, RebalancingError):
    """An attempt to move irreversible work, bypass policy, or use a stale fence."""


def refuse_ducklake_rebalance_authority(receipt: Mapping[str, Any] | None) -> None:
    if not receipt:
        return
    if (
        receipt.get("authoritative") is True
        or receipt.get("schedules") is True
        or receipt.get("steals") is True
        or receipt.get("rebalances") is True
    ):
        raise RebalancingAuthorityError("DuckLake cannot admit shard rebalancing")


@dataclass(frozen=True)
class RebalanceWork:
    """One shard-owned task considered for freeze, drain, and transfer."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/rebalance-work@1"
    )

    work: ShardWork
    claimed: bool = False
    in_flight: bool = False
    attempt_count: int = 0
    checkpoint_ref: str = ""
    cursor_ref: str = ""
    requires_human_review: bool = False
    requires_privacy_review: bool = False
    requires_proof: bool = False
    requires_merge: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.work, ShardWork):
            raise FederationContractError("rebalance work requires ShardWork")
        _integer(self.attempt_count, "attempt_count")
        _identifier(self.checkpoint_ref, "checkpoint_ref", required=False)
        _identifier(self.cursor_ref, "cursor_ref", required=False)
        for name in (
            "claimed",
            "in_flight",
            "requires_human_review",
            "requires_privacy_review",
            "requires_proof",
            "requires_merge",
        ):
            if type(getattr(self, name)) is not bool:
                raise FederationContractError(f"{name} must be boolean")

    @property
    def irreversible_active(self) -> bool:
        return self.in_flight is True and self.work.effect_class in IRREVERSIBLE_VALUES


@dataclass(frozen=True)
class ShardRebalanceRequest:
    """Exact source assignment and work snapshot for one rebalance cycle."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/shard-rebalance-request@1"
    )

    shard_id: str
    source_supervisor_id: str
    fencing_epoch: int
    assignment_revision: int
    semantic_root: str
    units: tuple[RebalanceWork, ...]
    claims_open: bool = True

    def __post_init__(self) -> None:
        _identifier(self.shard_id, "shard_id")
        _identifier(self.source_supervisor_id, "source_supervisor_id")
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        _integer(self.assignment_revision, "assignment_revision", minimum=1)
        _identifier(self.semantic_root, "semantic_root")
        if not isinstance(self.units, tuple) or not all(
            isinstance(item, RebalanceWork) for item in self.units
        ):
            raise FederationContractError("units must be RebalanceWork records")
        if type(self.claims_open) is not bool:
            raise FederationContractError("claims_open must be boolean")
        if len(self.units) > MAX_REBALANCE_TASKS:
            raise FederationContractError("rebalance request exceeds shard task ceiling")


@dataclass(frozen=True)
class CompiledRebalancePlan:
    """Frozen drain/transfer plan bound to the current fencing epoch."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/compiled-rebalance-plan@1"
    )

    shard_id: str
    source_supervisor_id: str
    target_supervisor_id: str
    source_revision: int
    target_revision: int
    previous_fencing_epoch: int
    fencing_epoch: int
    frozen: bool
    claims_stopped: bool
    transferred_task_ids: tuple[str, ...]
    preserved_checkpoint_refs: tuple[str, ...]
    preserved_cursor_refs: tuple[str, ...]
    preserved_attempt_counts: tuple[int, ...]
    tree_id: str
    semantic_root: str

    def __post_init__(self) -> None:
        for name in (
            "shard_id",
            "source_supervisor_id",
            "target_supervisor_id",
            "tree_id",
            "semantic_root",
        ):
            _identifier(getattr(self, name), name)
        _integer(self.source_revision, "source_revision", minimum=1)
        _integer(self.target_revision, "target_revision", minimum=1)
        if self.target_revision <= self.source_revision:
            raise RebalancingAuthorityError("rebalance must increment the assignment revision")
        _integer(self.previous_fencing_epoch, "previous_fencing_epoch", minimum=1)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        if self.fencing_epoch <= self.previous_fencing_epoch:
            raise RebalancingAuthorityError("rebalance must increment the fencing epoch")
        if type(self.frozen) is not bool or type(self.claims_stopped) is not bool:
            raise FederationContractError("frozen and claims_stopped must be boolean")
        _strings(self.transferred_task_ids, "transferred_task_ids", maximum=MAX_REBALANCE_TASKS)
        checkpoint_refs = tuple(
            _identifier(item, "preserved_checkpoint_refs", required=False)
            for item in self.preserved_checkpoint_refs
        )
        cursor_refs = tuple(
            _identifier(item, "preserved_cursor_refs", required=False)
            for item in self.preserved_cursor_refs
        )
        object.__setattr__(self, "preserved_checkpoint_refs", checkpoint_refs)
        object.__setattr__(self, "preserved_cursor_refs", cursor_refs)
        if not isinstance(self.preserved_attempt_counts, tuple) or not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in self.preserved_attempt_counts
        ):
            raise FederationContractError("preserved_attempt_counts must be integers")
        for item in self.preserved_attempt_counts:
            _integer(item, "preserved_attempt_counts")
        if not (
            len(self.transferred_task_ids)
            == len(self.preserved_checkpoint_refs)
            == len(self.preserved_cursor_refs)
            == len(self.preserved_attempt_counts)
        ):
            raise FederationContractError("preserved identities must align with transferred tasks")

    @property
    def plan_id(self) -> str:
        return "rebalance-plan:" + self.cid

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "shard_id": self.shard_id,
                "source_supervisor_id": self.source_supervisor_id,
                "target_supervisor_id": self.target_supervisor_id,
                "source_revision": self.source_revision,
                "target_revision": self.target_revision,
                "fencing_epoch": self.fencing_epoch,
                "transferred_task_ids": list(self.transferred_task_ids),
            }
        )


@dataclass(frozen=True)
class RebalanceReceipt:
    """Drain/transfer evidence after activation or fenced rollback."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/rebalance-receipt@1"
    )

    rebalance_plan_id: str
    shard_id: str
    source_supervisor_id: str
    target_supervisor_id: str
    source_revision: int
    target_revision: int
    previous_fencing_epoch: int
    fencing_epoch: int
    outcome: str
    transferred_task_ids: tuple[str, ...]
    preserved_checkpoint_refs: tuple[str, ...]
    preserved_cursor_refs: tuple[str, ...]
    preserved_attempt_counts: tuple[int, ...]
    budget_transferred: int
    tree_id: str
    activated: bool

    def __post_init__(self) -> None:
        for name in (
            "rebalance_plan_id",
            "shard_id",
            "source_supervisor_id",
            "target_supervisor_id",
            "tree_id",
        ):
            _identifier(getattr(self, name), name)
        _integer(self.source_revision, "source_revision", minimum=1)
        _integer(self.target_revision, "target_revision", minimum=1)
        _integer(self.previous_fencing_epoch, "previous_fencing_epoch", minimum=1)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        if self.fencing_epoch <= self.previous_fencing_epoch:
            raise RebalancingAuthorityError("rebalance must increment the fencing epoch")
        outcome = _identifier(self.outcome, "outcome")
        if outcome not in REBALANCE_OUTCOMES:
            raise RebalancingAuthorityError("rebalance outcome is outside its closed vocabulary")
        _strings(
            self.transferred_task_ids,
            "transferred_task_ids",
            maximum=MAX_REBALANCE_TASKS,
            required=False,
        )
        checkpoint_refs = tuple(
            _identifier(item, "preserved_checkpoint_refs", required=False)
            for item in self.preserved_checkpoint_refs
        )
        cursor_refs = tuple(
            _identifier(item, "preserved_cursor_refs", required=False)
            for item in self.preserved_cursor_refs
        )
        object.__setattr__(self, "preserved_checkpoint_refs", checkpoint_refs)
        object.__setattr__(self, "preserved_cursor_refs", cursor_refs)
        if not isinstance(self.preserved_attempt_counts, tuple) or not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in self.preserved_attempt_counts
        ):
            raise FederationContractError("preserved_attempt_counts must be integers")
        _integer(self.budget_transferred, "budget_transferred")
        if type(self.activated) is not bool:
            raise FederationContractError("activated must be boolean")
        if self.outcome == "rebalanced" and self.activated is not True:
            raise RebalancingAuthorityError("rebalanced receipts must activate the target")
        if self.outcome == "rolled_back" and self.activated is not False:
            raise RebalancingAuthorityError("rolled-back receipts cannot activate the target")

    @property
    def owner_supervisor_id(self) -> str:
        if self.outcome == "rebalanced":
            return self.target_supervisor_id
        return self.source_supervisor_id

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "rebalance_plan_id": self.rebalance_plan_id,
                "shard_id": self.shard_id,
                "outcome": self.outcome,
                "fencing_epoch": self.fencing_epoch,
                "target_revision": self.target_revision,
                "activated": self.activated,
            }
        )


def _require_policy_capability(
    units: Sequence[RebalanceWork],
    target: SupervisorSpecializationBound,
) -> None:
    required = (
        ("human_review", "requires_human_review"),
        ("privacy", "requires_privacy_review"),
        ("proof", "requires_proof"),
        ("merge", "requires_merge"),
    )
    for gate, attr in required:
        if any(getattr(unit, attr) for unit in units):
            if POLICY_GATES[gate] not in target.capability_refs:
                raise RebalancingAuthorityError(
                    "shard rebalancing cannot bypass policy, proof, merge, privacy, or human review"
                )


def _assert_live_plan(
    plan: CompiledRebalancePlan,
    *,
    binding: FederationBinding,
    current_tree_id: str,
    current_semantic_root: str,
    expected_source_fence: int,
    expected_assignment_revision: int,
    ducklake_receipt: Mapping[str, Any] | None,
) -> None:
    if not isinstance(plan, CompiledRebalancePlan):
        raise FederationContractError("compiled rebalance plan is required")
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    refuse_ducklake_rebalance_authority(ducklake_receipt)
    if retrieval_establishes_authority() is not False:
        raise RebalancingAuthorityError("retrieval cannot mint a shard rebalance")
    if plan.frozen is not True or plan.claims_stopped is not True:
        raise RebalancingAuthorityError("rebalance requires a frozen assignment with claims stopped")
    if expected_source_fence != plan.previous_fencing_epoch:
        raise RebalancingAuthorityError("source fencing epoch is stale")
    if expected_assignment_revision != plan.source_revision:
        raise RebalancingAuthorityError("assignment revision is stale")
    if current_tree_id != plan.tree_id:
        raise RebalancingAuthorityError("rebalance requires current tree identity")
    if current_semantic_root != plan.semantic_root:
        raise RebalancingAuthorityError("rebalance requires current semantic state")
    if current_tree_id not in binding.repository_tree_ids:
        raise RebalancingAuthorityError("rebalance tree is not bound to the federation")


def _receipt_from_plan(
    plan: CompiledRebalancePlan,
    *,
    outcome: str,
    transferred_task_ids: tuple[str, ...],
    budget_transferred: int,
    activated: bool,
) -> RebalanceReceipt:
    return RebalanceReceipt(
        rebalance_plan_id=plan.plan_id,
        shard_id=plan.shard_id,
        source_supervisor_id=plan.source_supervisor_id,
        target_supervisor_id=plan.target_supervisor_id,
        source_revision=plan.source_revision,
        target_revision=plan.target_revision,
        previous_fencing_epoch=plan.previous_fencing_epoch,
        fencing_epoch=plan.fencing_epoch,
        outcome=outcome,
        transferred_task_ids=transferred_task_ids,
        preserved_checkpoint_refs=plan.preserved_checkpoint_refs,
        preserved_cursor_refs=plan.preserved_cursor_refs,
        preserved_attempt_counts=plan.preserved_attempt_counts,
        budget_transferred=budget_transferred,
        tree_id=plan.tree_id,
        activated=activated,
    )


def _transfer_budget(
    *,
    ledger: HierarchicalBudgetLedger | None,
    source_budget_account_id: str,
    target_budget_account_id: str,
    budget_dimension: BudgetDimensionName | None,
    budget_amount: int,
    expected_source_budget_revision: int,
    expected_target_budget_revision: int,
) -> int:
    if not budget_amount:
        return 0
    if ledger is None or budget_dimension is None:
        raise RebalancingAuthorityError("budget transfer requires a hierarchical ledger")
    if not source_budget_account_id or not target_budget_account_id:
        raise RebalancingAuthorityError("budget transfer requires both supervisor accounts")
    ledger.transfer(
        source_account_id=source_budget_account_id,
        target_account_id=target_budget_account_id,
        dimension=budget_dimension,
        amount=budget_amount,
        expected_source_revision=expected_source_budget_revision,
        expected_target_revision=expected_target_budget_revision,
    )
    return budget_amount


def compile_shard_rebalance(
    request: ShardRebalanceRequest,
    *,
    target: SupervisorSpecializationBound,
    binding: FederationBinding,
    expected_source_fence: int,
    expected_assignment_revision: int,
    current_tree_id: str,
    current_semantic_root: str,
    target_existing_shard_count: int = 0,
    ducklake_receipt: Mapping[str, Any] | None = None,
) -> CompiledRebalancePlan:
    """Freeze the source assignment and compile a drain/transfer plan."""

    if not isinstance(request, ShardRebalanceRequest):
        raise FederationContractError("rebalance request is required")
    if not isinstance(target, SupervisorSpecializationBound):
        raise FederationContractError("target specialization is required")
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    refuse_ducklake_rebalance_authority(ducklake_receipt)
    if retrieval_establishes_authority() is not False:
        raise RebalancingAuthorityError("retrieval cannot mint a shard rebalance")
    if request.source_supervisor_id == target.supervisor_id:
        raise RebalancingAuthorityError("a supervisor cannot rebalance onto itself")
    if not request.units:
        raise FederationContractError("rebalance requires at least one work item")
    if expected_source_fence != request.fencing_epoch:
        raise RebalancingAuthorityError("source fencing epoch is stale")
    if expected_assignment_revision != request.assignment_revision:
        raise RebalancingAuthorityError("assignment revision is stale")
    if current_tree_id != request.units[0].work.tree_id:
        raise RebalancingAuthorityError("rebalance requires current tree identity")
    if current_semantic_root != request.semantic_root:
        raise RebalancingAuthorityError("rebalance requires current semantic state")
    if current_tree_id not in binding.repository_tree_ids:
        raise RebalancingAuthorityError("rebalance tree is not bound to the federation")
    _integer(target_existing_shard_count, "target_existing_shard_count")
    if target_existing_shard_count >= target.max_shards:
        raise RebalancingAuthorityError("target specialization cannot admit another shard")
    seen: set[str] = set()
    tree_id = ""
    for unit in request.units:
        if unit.irreversible_active:
            raise RebalancingAuthorityError("active irreversible effects cannot move")
        item = unit.work
        if item.task_id in seen:
            raise RebalancingAuthorityError("rebalance assigns a task to more than one owner")
        seen.add(item.task_id)
        if item.tree_id not in binding.repository_tree_ids:
            raise RebalancingAuthorityError("rebalance requires current tree identity")
        if item.repository_id not in binding.repository_ids:
            raise RebalancingAuthorityError("rebalance work repository is not bound")
        if current_tree_id != item.tree_id:
            raise RebalancingAuthorityError("rebalance requires current tree identity")
        if tree_id and tree_id != item.tree_id:
            raise RebalancingAuthorityError("rebalance work spans more than one tree")
        tree_id = item.tree_id
        if not target.admits(item):
            raise RebalancingAuthorityError("target specialization cannot admit the rebalanced work")
    _require_policy_capability(request.units, target)
    ordered = tuple(sorted(request.units, key=lambda unit: unit.work.task_id))
    return CompiledRebalancePlan(
        shard_id=request.shard_id,
        source_supervisor_id=request.source_supervisor_id,
        target_supervisor_id=target.supervisor_id,
        source_revision=request.assignment_revision,
        target_revision=request.assignment_revision + 1,
        previous_fencing_epoch=request.fencing_epoch,
        fencing_epoch=request.fencing_epoch + 1,
        frozen=True,
        claims_stopped=True,
        transferred_task_ids=tuple(unit.work.task_id for unit in ordered),
        preserved_checkpoint_refs=tuple(unit.checkpoint_ref for unit in ordered),
        preserved_cursor_refs=tuple(unit.cursor_ref for unit in ordered),
        preserved_attempt_counts=tuple(unit.attempt_count for unit in ordered),
        tree_id=tree_id,
        semantic_root=request.semantic_root,
    )


def execute_shard_rebalance(
    plan: CompiledRebalancePlan,
    *,
    binding: FederationBinding,
    current_tree_id: str,
    current_semantic_root: str,
    expected_source_fence: int,
    expected_assignment_revision: int,
    ledger: HierarchicalBudgetLedger | None = None,
    source_budget_account_id: str = "",
    target_budget_account_id: str = "",
    budget_dimension: BudgetDimensionName | None = None,
    budget_amount: int = 0,
    expected_source_budget_revision: int = 1,
    expected_target_budget_revision: int = 1,
    ducklake_receipt: Mapping[str, Any] | None = None,
) -> RebalanceReceipt:
    """Activate the frozen plan: drain/transfer work, increment fence, transfer budget."""

    _assert_live_plan(
        plan,
        binding=binding,
        current_tree_id=current_tree_id,
        current_semantic_root=current_semantic_root,
        expected_source_fence=expected_source_fence,
        expected_assignment_revision=expected_assignment_revision,
        ducklake_receipt=ducklake_receipt,
    )
    transferred = _transfer_budget(
        ledger=ledger,
        source_budget_account_id=source_budget_account_id,
        target_budget_account_id=target_budget_account_id,
        budget_dimension=budget_dimension,
        budget_amount=budget_amount,
        expected_source_budget_revision=expected_source_budget_revision,
        expected_target_budget_revision=expected_target_budget_revision,
    )
    return _receipt_from_plan(
        plan,
        outcome="rebalanced",
        transferred_task_ids=plan.transferred_task_ids,
        budget_transferred=transferred,
        activated=True,
    )


def rollback_shard_rebalance(
    plan: CompiledRebalancePlan,
    *,
    binding: FederationBinding,
    current_tree_id: str,
    current_semantic_root: str,
    expected_source_fence: int,
    expected_assignment_revision: int,
    ducklake_receipt: Mapping[str, Any] | None = None,
) -> RebalanceReceipt:
    """Abort a frozen plan: restore source ownership and increment the fence."""

    _assert_live_plan(
        plan,
        binding=binding,
        current_tree_id=current_tree_id,
        current_semantic_root=current_semantic_root,
        expected_source_fence=expected_source_fence,
        expected_assignment_revision=expected_assignment_revision,
        ducklake_receipt=ducklake_receipt,
    )
    return _receipt_from_plan(
        plan,
        outcome="rolled_back",
        transferred_task_ids=(),
        budget_transferred=0,
        activated=False,
    )


class ShardRebalancePlanner:
    """Compile then execute a fenced freeze-drain-transfer-activate cycle."""

    def compile(
        self,
        request: ShardRebalanceRequest,
        *,
        target: SupervisorSpecializationBound,
        binding: FederationBinding,
        expected_source_fence: int,
        expected_assignment_revision: int,
        current_tree_id: str,
        current_semantic_root: str,
        target_existing_shard_count: int = 0,
        ducklake_receipt: Mapping[str, Any] | None = None,
    ) -> CompiledRebalancePlan:
        return compile_shard_rebalance(
            request,
            target=target,
            binding=binding,
            expected_source_fence=expected_source_fence,
            expected_assignment_revision=expected_assignment_revision,
            current_tree_id=current_tree_id,
            current_semantic_root=current_semantic_root,
            target_existing_shard_count=target_existing_shard_count,
            ducklake_receipt=ducklake_receipt,
        )

    def execute(
        self,
        plan: CompiledRebalancePlan,
        *,
        binding: FederationBinding,
        current_tree_id: str,
        current_semantic_root: str,
        expected_source_fence: int,
        expected_assignment_revision: int,
        ledger: HierarchicalBudgetLedger | None = None,
        source_budget_account_id: str = "",
        target_budget_account_id: str = "",
        budget_dimension: BudgetDimensionName | None = None,
        budget_amount: int = 0,
        expected_source_budget_revision: int = 1,
        expected_target_budget_revision: int = 1,
        ducklake_receipt: Mapping[str, Any] | None = None,
    ) -> RebalanceReceipt:
        return execute_shard_rebalance(
            plan,
            binding=binding,
            current_tree_id=current_tree_id,
            current_semantic_root=current_semantic_root,
            expected_source_fence=expected_source_fence,
            expected_assignment_revision=expected_assignment_revision,
            ledger=ledger,
            source_budget_account_id=source_budget_account_id,
            target_budget_account_id=target_budget_account_id,
            budget_dimension=budget_dimension,
            budget_amount=budget_amount,
            expected_source_budget_revision=expected_source_budget_revision,
            expected_target_budget_revision=expected_target_budget_revision,
            ducklake_receipt=ducklake_receipt,
        )

    def rollback(
        self,
        plan: CompiledRebalancePlan,
        *,
        binding: FederationBinding,
        current_tree_id: str,
        current_semantic_root: str,
        expected_source_fence: int,
        expected_assignment_revision: int,
        ducklake_receipt: Mapping[str, Any] | None = None,
    ) -> RebalanceReceipt:
        return rollback_shard_rebalance(
            plan,
            binding=binding,
            current_tree_id=current_tree_id,
            current_semantic_root=current_semantic_root,
            expected_source_fence=expected_source_fence,
            expected_assignment_revision=expected_assignment_revision,
            ducklake_receipt=ducklake_receipt,
        )


def _rebalance_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_insert_shard_rebalance_plan",
            """
            INSERT INTO shard_rebalance_plans (
                rebalance_plan_id, tenant_id, federation_id, shard_id,
                source_revision, target_revision, source_supervisor_id,
                target_supervisor_id, state, content_ref, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "rebalance_plan_id",
                "tenant_id",
                "federation_id",
                "shard_id",
                "source_revision",
                "target_revision",
                "source_supervisor_id",
                "target_supervisor_id",
                "state",
                "content_ref",
                "created_at",
                "updated_at",
            ),
        ),
        _template(
            "casf_select_shard_rebalance_plan",
            """
            SELECT rebalance_plan_id, shard_id, source_revision, target_revision,
                   source_supervisor_id, target_supervisor_id, state, content_ref
            FROM shard_rebalance_plans
            WHERE rebalance_plan_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("rebalance_plan_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_shard_revision",
            """
            INSERT INTO shard_revisions (
                shard_id, revision, tenant_id, federation_id, previous_revision,
                fencing_epoch, boundary_population_ref, assignment_population_ref,
                state, content_ref, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "shard_id",
                "revision",
                "tenant_id",
                "federation_id",
                "previous_revision",
                "fencing_epoch",
                "boundary_population_ref",
                "assignment_population_ref",
                "state",
                "content_ref",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_shard_revision",
            """
            SELECT shard_id, revision, previous_revision, fencing_epoch, state, content_ref
            FROM shard_revisions
            WHERE shard_id = ? AND revision = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("shard_id", "revision", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_shard_rebalance_receipt",
            """
            INSERT INTO shard_rebalance_receipts (
                rebalance_receipt_id, tenant_id, federation_id, rebalance_plan_id,
                shard_id, source_revision, target_revision, fencing_epoch,
                disposition, content_ref, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "rebalance_receipt_id",
                "tenant_id",
                "federation_id",
                "rebalance_plan_id",
                "shard_id",
                "source_revision",
                "target_revision",
                "fencing_epoch",
                "disposition",
                "content_ref",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_shard_rebalance_receipt",
            """
            SELECT rebalance_receipt_id, rebalance_plan_id, shard_id,
                   source_revision, target_revision, fencing_epoch,
                   disposition, content_ref
            FROM shard_rebalance_receipts
            WHERE rebalance_receipt_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("rebalance_receipt_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
    )


class RebalancingStore(WorkStealingStore):
    """Persist rebalance plans, fencing revisions, and receipts through Quack."""

    INTERFACE = "RebalancingStore@1"

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
            raise RebalancingError("rebalancing store never accepts a database path")
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise RebalancingError(
                "rebalancing store requires an already-attached typed state client"
            )
        registered = set(client.list_templates())
        missing = [
            template.name for template in _rebalance_templates() if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise RebalancingError("rebalancing templates are absent from the sealed catalog")
        else:
            for template in _rebalance_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    def record_rebalance(
        self,
        plan: CompiledRebalancePlan,
        receipt: RebalanceReceipt,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
    ) -> CausalGraphCommit:
        if not isinstance(plan, CompiledRebalancePlan):
            raise FederationContractError("compiled rebalance plan is required")
        if not isinstance(receipt, RebalanceReceipt):
            raise FederationContractError("rebalance receipt is required")
        if receipt.rebalance_plan_id != plan.plan_id:
            raise RebalancingAuthorityError("receipt plan identity differs from the compiled plan")
        receipt_id = "rebalance-receipt:" + receipt.cid
        return self._commit_fact(
            operation="federation.rebalance.record",
            fact_id=receipt_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(
                dict.fromkeys((plan.plan_id, receipt_id, plan.shard_id, *receipt.transferred_task_ids))
            ),
            payload_ref=receipt.cid,
            prepare_fact=lambda: None,
            apply_fact=lambda revision, recorded_at: self._insert_rebalance(
                plan,
                receipt,
                receipt_id=receipt_id,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                graph_revision=revision,
                recorded_at=recorded_at,
            ),
        )

    def load_plan(
        self,
        *,
        plan_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_shard_rebalance_plan",
            {
                "rebalance_plan_id": _identifier(plan_id, "plan_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise RebalancingError("shard rebalance plan is absent")
        return dict(rows[0])

    def load_receipt(
        self,
        *,
        receipt_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_shard_rebalance_receipt",
            {
                "rebalance_receipt_id": _identifier(receipt_id, "receipt_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise RebalancingError("shard rebalance receipt is absent")
        return dict(rows[0])

    def load_revision(
        self,
        *,
        shard_id: str,
        revision: int,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_shard_revision",
            {
                "shard_id": _identifier(shard_id, "shard_id"),
                "revision": _integer(revision, "revision", minimum=1),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise RebalancingError("shard revision is absent")
        return dict(rows[0])

    def _insert_rebalance(
        self,
        plan: CompiledRebalancePlan,
        receipt: RebalanceReceipt,
        *,
        receipt_id: str,
        federation_id: str,
        tenant_id: str,
        graph_revision: int,
        recorded_at: str,
    ) -> None:
        del graph_revision
        self._client.execute(
            "casf_insert_shard_rebalance_plan",
            {
                "rebalance_plan_id": plan.plan_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "shard_id": plan.shard_id,
                "source_revision": plan.source_revision,
                "target_revision": plan.target_revision,
                "source_supervisor_id": plan.source_supervisor_id,
                "target_supervisor_id": plan.target_supervisor_id,
                "state": "frozen",
                "content_ref": plan.cid,
                "created_at": recorded_at,
                "updated_at": recorded_at,
            },
        )
        self._client.execute(
            "casf_insert_shard_revision",
            {
                "shard_id": plan.shard_id,
                "revision": plan.target_revision,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "previous_revision": plan.source_revision,
                "fencing_epoch": receipt.fencing_epoch,
                "boundary_population_ref": plan.cid,
                "assignment_population_ref": receipt.cid,
                "state": receipt.outcome,
                "content_ref": receipt.cid,
                "recorded_at": recorded_at,
            },
        )
        self._client.execute(
            "casf_insert_shard_rebalance_receipt",
            {
                "rebalance_receipt_id": receipt_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "rebalance_plan_id": plan.plan_id,
                "shard_id": plan.shard_id,
                "source_revision": plan.source_revision,
                "target_revision": plan.target_revision,
                "fencing_epoch": receipt.fencing_epoch,
                "disposition": receipt.outcome,
                "content_ref": receipt.cid,
                "recorded_at": recorded_at,
            },
        )


__all__ = (
    "CompiledRebalancePlan",
    "REBALANCE_OUTCOMES",
    "RebalanceReceipt",
    "RebalanceWork",
    "RebalancingAuthorityError",
    "RebalancingError",
    "RebalancingStore",
    "ShardRebalancePlanner",
    "ShardRebalanceRequest",
    "compile_shard_rebalance",
    "execute_shard_rebalance",
    "refuse_ducklake_rebalance_authority",
    "rollback_shard_rebalance",
)
