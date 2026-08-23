"""Virgin-only work stealing with atomic fence and budget transfer.

A steal is allowed only for unclaimed, never-attempted work that the thief can
admit under its specialization ceiling and current tree/semantic state. Active
irreversible effects stay put. Stealing cannot bypass repository, policy,
proof, merge, privacy, or human-review gates. The current fence wins.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
)
from .parallel_frontier import IRREVERSIBLE_EFFECT_CLASSES
from .registry import _template
from .retrieval_projection import retrieval_establishes_authority
from .sharding import (
    ShardingStore,
    ShardWork,
    SupervisorSpecializationBound,
    refuse_ducklake_shard_authority,
)

POLICY_GATES = {
    "human_review": "capability:human-review",
    "privacy": "capability:privacy",
    "proof": "capability:proof",
    "merge": "capability:merge",
}
IRREVERSIBLE_VALUES = frozenset(item.value for item in IRREVERSIBLE_EFFECT_CLASSES)


class WorkStealingError(CausalGraphError):
    """Base typed work-stealing failure."""


class WorkStealingAuthorityError(FederationAuthorityError, WorkStealingError):
    """An attempt to steal claimed, out-of-ceiling, or policy-gated work."""


def refuse_ducklake_steal_authority(receipt: Mapping[str, Any] | None) -> None:
    refuse_ducklake_shard_authority(receipt)
    if receipt and receipt.get("steals") is True:
        raise WorkStealingAuthorityError("DuckLake cannot admit work stealing")


@dataclass(frozen=True)
class StealCandidate:
    """One potentially stealable task bound to its current owner and gates."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/steal-candidate@1"
    )

    work: ShardWork
    source_supervisor_id: str
    fencing_epoch: int
    assignment_revision: int
    semantic_root: str
    claimed: bool = False
    attempt_count: int = 0
    in_flight: bool = False
    requires_human_review: bool = False
    requires_privacy_review: bool = False
    requires_proof: bool = False
    requires_merge: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.work, ShardWork):
            raise FederationContractError("steal candidate requires ShardWork")
        _identifier(self.source_supervisor_id, "source_supervisor_id")
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        _integer(self.assignment_revision, "assignment_revision", minimum=1)
        _identifier(self.semantic_root, "semantic_root")
        _integer(self.attempt_count, "attempt_count")
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
    def virgin(self) -> bool:
        return self.claimed is False and self.in_flight is False and self.attempt_count == 0

    @property
    def irreversible_active(self) -> bool:
        return self.in_flight is True and self.work.effect_class in IRREVERSIBLE_VALUES


@dataclass(frozen=True)
class StealReceipt:
    """Evidence that one virgin task transferred with fence and budget CAS."""

    SCHEMA: ClassVar[str] = "ipfs_accelerate_py/agent-supervisor/causal-federation/steal-receipt@1"

    task_id: str
    source_supervisor_id: str
    thief_supervisor_id: str
    previous_fencing_epoch: int
    fencing_epoch: int
    assignment_revision: int
    budget_transferred: int
    tree_id: str

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "source_supervisor_id",
            "thief_supervisor_id",
            "tree_id",
        ):
            _identifier(getattr(self, name), name)
        _integer(self.previous_fencing_epoch, "previous_fencing_epoch", minimum=1)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        if self.fencing_epoch <= self.previous_fencing_epoch:
            raise WorkStealingAuthorityError("steal must increment the fencing epoch")
        _integer(self.assignment_revision, "assignment_revision", minimum=1)
        _integer(self.budget_transferred, "budget_transferred")

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "task_id": self.task_id,
                "source_supervisor_id": self.source_supervisor_id,
                "thief_supervisor_id": self.thief_supervisor_id,
                "fencing_epoch": self.fencing_epoch,
                "assignment_revision": self.assignment_revision,
            }
        )


def _require_policy_capability(
    candidate: StealCandidate,
    thief: SupervisorSpecializationBound,
) -> None:
    required = (
        ("human_review", candidate.requires_human_review),
        ("privacy", candidate.requires_privacy_review),
        ("proof", candidate.requires_proof),
        ("merge", candidate.requires_merge),
    )
    for gate, needed in required:
        if needed and POLICY_GATES[gate] not in thief.capability_refs:
            raise WorkStealingAuthorityError(
                "work stealing cannot bypass policy, proof, merge, privacy, or human review"
            )


def steal_work(
    candidate: StealCandidate,
    *,
    thief: SupervisorSpecializationBound,
    binding: FederationBinding,
    current_tree_id: str,
    current_semantic_root: str,
    expected_source_fence: int,
    expected_assignment_revision: int,
    ledger: HierarchicalBudgetLedger | None = None,
    source_budget_account_id: str = "",
    thief_budget_account_id: str = "",
    budget_dimension: BudgetDimensionName | None = None,
    budget_amount: int = 0,
    expected_source_budget_revision: int = 1,
    expected_thief_budget_revision: int = 1,
    ducklake_receipt: Mapping[str, Any] | None = None,
) -> StealReceipt:
    """Transfer one virgin task to an idle capable thief. Current fence wins."""

    if not isinstance(candidate, StealCandidate):
        raise FederationContractError("steal candidate is required")
    if not isinstance(thief, SupervisorSpecializationBound):
        raise FederationContractError("thief specialization is required")
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    refuse_ducklake_steal_authority(ducklake_receipt)
    if retrieval_establishes_authority() is not False:
        raise WorkStealingAuthorityError("retrieval cannot mint a steal")
    if candidate.source_supervisor_id == thief.supervisor_id:
        raise WorkStealingAuthorityError("a supervisor cannot steal its own work")
    if candidate.irreversible_active:
        raise WorkStealingAuthorityError("active irreversible effects cannot move")
    if candidate.claimed or not candidate.virgin:
        raise WorkStealingAuthorityError("only unclaimed virgin work may be stolen")
    if expected_source_fence != candidate.fencing_epoch:
        raise WorkStealingAuthorityError("source fencing epoch is stale")
    if expected_assignment_revision != candidate.assignment_revision:
        raise WorkStealingAuthorityError("assignment revision is stale")
    if current_tree_id != candidate.work.tree_id:
        raise WorkStealingAuthorityError("steal requires current tree identity")
    if current_semantic_root != candidate.semantic_root:
        raise WorkStealingAuthorityError("steal requires current semantic state")
    if current_tree_id not in binding.repository_tree_ids:
        raise WorkStealingAuthorityError("steal tree is not bound to the federation")
    if not thief.admits(candidate.work):
        raise WorkStealingAuthorityError("thief specialization cannot admit the stolen work")
    _require_policy_capability(candidate, thief)
    transferred = 0
    if budget_amount:
        if ledger is None or budget_dimension is None:
            raise WorkStealingAuthorityError("budget transfer requires a hierarchical ledger")
        if not source_budget_account_id or not thief_budget_account_id:
            raise WorkStealingAuthorityError("budget transfer requires both supervisor accounts")
        ledger.transfer(
            source_account_id=source_budget_account_id,
            target_account_id=thief_budget_account_id,
            dimension=budget_dimension,
            amount=budget_amount,
            expected_source_revision=expected_source_budget_revision,
            expected_target_revision=expected_thief_budget_revision,
        )
        transferred = budget_amount
    return StealReceipt(
        task_id=candidate.work.task_id,
        source_supervisor_id=candidate.source_supervisor_id,
        thief_supervisor_id=thief.supervisor_id,
        previous_fencing_epoch=candidate.fencing_epoch,
        fencing_epoch=candidate.fencing_epoch + 1,
        assignment_revision=candidate.assignment_revision + 1,
        budget_transferred=transferred,
        tree_id=candidate.work.tree_id,
    )


def _steal_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_insert_work_steal_receipt",
            """
            INSERT INTO federation_receipts (
                federation_receipt_id, tenant_id, federation_id, receipt_kind,
                federation_revision, control_plane_generation, event_watermark,
                issuer_id, content_ref, recorded_at
            ) VALUES (?, ?, ?, 'work_steal', ?, ?, ?, ?, ?, ?)
            """,
            (
                "federation_receipt_id",
                "tenant_id",
                "federation_id",
                "federation_revision",
                "control_plane_generation",
                "event_watermark",
                "issuer_id",
                "content_ref",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_work_steal_receipt",
            """
            SELECT federation_receipt_id, receipt_kind, content_ref
            FROM federation_receipts
            WHERE federation_receipt_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("federation_receipt_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
    )


class WorkStealingStore(ShardingStore):
    """Persist steal receipts through the sealed state owner."""

    INTERFACE = "WorkStealingStore@1"

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
            raise WorkStealingError("work-stealing store never accepts a database path")
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise WorkStealingError(
                "work-stealing store requires an already-attached typed state client"
            )
        registered = set(client.list_templates())
        missing = [
            template.name for template in _steal_templates() if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise WorkStealingError(
                    "work-stealing templates are absent from the sealed catalog"
                )
        else:
            for template in _steal_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    def record_steal(
        self,
        receipt: StealReceipt,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
    ) -> CausalGraphCommit:
        if not isinstance(receipt, StealReceipt):
            raise FederationContractError("steal receipt is required")
        receipt_id = "federation-receipt:" + receipt.cid
        return self._commit_fact(
            operation="federation.steal.record",
            fact_id=receipt_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=(receipt_id, receipt.task_id),
            payload_ref=receipt.cid,
            prepare_fact=lambda: None,
            apply_fact=lambda revision, recorded_at: self._insert_steal(
                receipt,
                receipt_id=receipt_id,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                generation=binding.control_plane_generation,
                graph_revision=revision,
                recorded_at=recorded_at,
            ),
        )

    def load_steal(
        self,
        *,
        receipt_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_work_steal_receipt",
            {
                "federation_receipt_id": _identifier(receipt_id, "receipt_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise WorkStealingError("work-steal receipt is absent")
        return dict(rows[0])

    def _insert_steal(
        self,
        receipt: StealReceipt,
        *,
        receipt_id: str,
        federation_id: str,
        tenant_id: str,
        generation: int,
        graph_revision: int,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_work_steal_receipt",
            {
                "federation_receipt_id": receipt_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "federation_revision": graph_revision,
                "control_plane_generation": generation,
                "event_watermark": 0,
                "issuer_id": "work-stealing",
                "content_ref": receipt.cid,
                "recorded_at": recorded_at,
            },
        )


__all__ = (
    "POLICY_GATES",
    "StealCandidate",
    "StealReceipt",
    "WorkStealingAuthorityError",
    "WorkStealingError",
    "WorkStealingStore",
    "refuse_ducklake_steal_authority",
    "steal_work",
)
