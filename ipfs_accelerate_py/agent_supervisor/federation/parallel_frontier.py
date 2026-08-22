"""Conflict-free parallel frontier compilation for CASF.

The frontier admits only a pairwise-compatible wave. Shared reads may share a
worktree; authoritative writes serialize; disjoint worktrees run together only
with proved or policy-admitted independence. Unknown conflict, irreversible
effects, exhausted merge/proof slots, and ``do_not_wake`` supervisors reduce
concurrency. Every admitted task binds supervisor, subagent, worktree, lease,
fence, merge lane, and validation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ..task_sources.control_plane_contracts import content_identity
from ..task_sources.quack_state_client import QuackStateClient, StatementKind
from .causal_frontier import CompiledFrontier
from .causal_graph import CausalGraphCommit, CausalGraphError
from .contracts import (
    FederationAuthorityError,
    FederationBinding,
    FederationContractError,
    _identifier,
    _integer,
    _strings,
)
from .deduplication import (
    WRITE_EFFECT_CLASSES,
    DeduplicationStore,
    IntentDisposition,
    IntentIndependenceAdmission,
    TaskIntentIdentity,
    classify_intents,
    refuse_ducklake_dedup_authority,
)
from .events import EventEffectClass
from .registry import _template
from .retrieval_projection import retrieval_establishes_authority

IRREVERSIBLE_EFFECT_CLASSES = frozenset(
    {
        EventEffectClass.EXTERNAL_IRREVERSIBLE,
        EventEffectClass.SECURITY_OR_LEGAL,
        EventEffectClass.PAYMENT,
    }
)
ADMISSION_STATUSES = frozenset(
    {
        "admitted_parallel",
        "serialized",
        "suppressed",
        "blocked",
        "asleep",
    }
)
MAX_PARALLEL_TASKS = 1_024


class ParallelFrontierError(CausalGraphError):
    """Base typed parallel-frontier compilation failure."""


class ParallelFrontierAuthorityError(FederationAuthorityError, ParallelFrontierError):
    """An attempt to admit parallel work without assignment, independence, or capability."""


def _write_like(effect_class: EventEffectClass) -> bool:
    return effect_class in WRITE_EFFECT_CLASSES


def _irreversible(effect_class: EventEffectClass) -> bool:
    return effect_class in IRREVERSIBLE_EFFECT_CLASSES


def refuse_ducklake_parallel_authority(receipt: Mapping[str, Any] | None) -> None:
    refuse_ducklake_dedup_authority(receipt)
    if receipt and receipt.get("parallelizes") is True:
        raise ParallelFrontierAuthorityError("DuckLake cannot admit a parallel frontier")


@dataclass(frozen=True)
class FrontierCapacity:
    """Explicit merge, proof, and wave-size ceilings for one compiled wave."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/frontier-capacity@1"
    )

    merge_slots: int = 1
    proof_slots: int = 1
    max_parallel: int = MAX_PARALLEL_TASKS

    def __post_init__(self) -> None:
        _integer(self.merge_slots, "merge_slots")
        _integer(self.proof_slots, "proof_slots")
        _integer(self.max_parallel, "max_parallel", minimum=1, maximum=MAX_PARALLEL_TASKS)


@dataclass(frozen=True)
class ParallelTask:
    """One fully assigned task considered for the current parallel wave."""

    SCHEMA: ClassVar[str] = "ipfs_accelerate_py/agent-supervisor/causal-federation/parallel-task@1"

    intent: TaskIntentIdentity
    supervisor_id: str
    subagent_id: str
    worktree_id: str
    lease_id: str
    fencing_epoch: int
    merge_lane: str
    validation_plan_ref: str
    resource_reservation_ref: str
    token_reservation_ref: str
    requires_merge_slot: bool = False
    requires_proof_slot: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.intent, TaskIntentIdentity):
            raise FederationContractError("parallel task requires TaskIntentIdentity")
        for name in (
            "supervisor_id",
            "subagent_id",
            "worktree_id",
            "lease_id",
            "merge_lane",
            "validation_plan_ref",
            "resource_reservation_ref",
            "token_reservation_ref",
        ):
            _identifier(getattr(self, name), name)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        if type(self.requires_merge_slot) is not bool or type(self.requires_proof_slot) is not bool:
            raise FederationContractError("parallel task slot flags must be boolean")

    @property
    def task_id(self) -> str:
        return self.intent.task_id


def bind_parallel_task(
    *,
    binding: FederationBinding,
    intent: TaskIntentIdentity,
    supervisor_id: str,
    subagent_id: str,
    worktree_id: str,
    lease_id: str,
    fencing_epoch: int,
    merge_lane: str,
    validation_plan_ref: str,
    resource_reservation_ref: str,
    token_reservation_ref: str,
    requires_merge_slot: bool = False,
    requires_proof_slot: bool = False,
) -> ParallelTask:
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    if not isinstance(intent, TaskIntentIdentity):
        raise FederationContractError("intent must be a TaskIntentIdentity")
    if intent.tree_id not in binding.repository_tree_ids:
        raise ParallelFrontierAuthorityError("parallel task tree identity mismatches")
    if intent.validation_ref != validation_plan_ref:
        raise ParallelFrontierAuthorityError(
            "validation plan must match the task-intent validation identity"
        )
    return ParallelTask(
        intent=intent,
        supervisor_id=supervisor_id,
        subagent_id=subagent_id,
        worktree_id=worktree_id,
        lease_id=lease_id,
        fencing_epoch=fencing_epoch,
        merge_lane=merge_lane,
        validation_plan_ref=validation_plan_ref,
        resource_reservation_ref=resource_reservation_ref,
        token_reservation_ref=token_reservation_ref,
        requires_merge_slot=requires_merge_slot,
        requires_proof_slot=requires_proof_slot,
    )


@dataclass(frozen=True)
class CompiledParallelFrontier:
    """One conflict-free wave plus the serial remainder."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/compiled-parallel-frontier@1"
    )

    wave_id: str
    admitted: tuple[str, ...]
    serialized: tuple[str, ...]
    suppressed: tuple[str, ...]
    blocked: tuple[str, ...]
    asleep: tuple[str, ...]
    merge_order: tuple[str, ...]
    assignment_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.wave_id, "wave_id")
        for name in (
            "admitted",
            "serialized",
            "suppressed",
            "blocked",
            "asleep",
            "merge_order",
            "assignment_refs",
        ):
            _strings(getattr(self, name), name, maximum=MAX_PARALLEL_TASKS, required=False)
        groups = (
            set(self.admitted),
            set(self.serialized),
            set(self.suppressed),
            set(self.blocked),
            set(self.asleep),
        )
        overlap: set[str] = set()
        seen: set[str] = set()
        for group in groups:
            overlap |= seen & group
            seen |= group
        if overlap:
            raise FederationContractError("parallel frontier partitions overlap")

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "wave_id": self.wave_id,
                "admitted": list(self.admitted),
                "serialized": list(self.serialized),
                "suppressed": list(self.suppressed),
                "blocked": list(self.blocked),
                "asleep": list(self.asleep),
                "merge_order": list(self.merge_order),
            }
        )


def _incompatible_pairs(
    relations: Sequence[Any],
) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    for relation in relations:
        if relation.disposition is IntentDisposition.CONFLICT:
            pairs.add(frozenset({relation.left_task_id, relation.right_task_id}))
    return pairs


def compile_parallel_frontier(
    tasks: Sequence[ParallelTask],
    *,
    binding: FederationBinding,
    independence: Sequence[IntentIndependenceAdmission] = (),
    capacity: FrontierCapacity | None = None,
    causal_frontier: CompiledFrontier | None = None,
    ducklake_receipt: Mapping[str, Any] | None = None,
    force_parallel: Sequence[str] = (),
) -> CompiledParallelFrontier:
    """Admit one conflict-free wave. Unknown conflict reduces concurrency."""

    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    refuse_ducklake_parallel_authority(ducklake_receipt)
    if retrieval_establishes_authority() is not False:
        raise ParallelFrontierAuthorityError("retrieval cannot mint parallel admission")
    if force_parallel:
        raise ParallelFrontierAuthorityError(
            "parallel admission cannot be forced past the frontier"
        )
    if not tasks:
        raise FederationContractError("parallel frontier requires at least one task")
    if len(tasks) > MAX_PARALLEL_TASKS:
        raise ParallelFrontierError("parallel task set exceeds bound")
    limits = capacity or FrontierCapacity()
    if not isinstance(limits, FrontierCapacity):
        raise FederationContractError("capacity must be FrontierCapacity")
    by_id: dict[str, ParallelTask] = {}
    for task in tasks:
        if not isinstance(task, ParallelTask):
            raise FederationContractError("tasks must be ParallelTask records")
        if task.intent.tree_id not in binding.repository_tree_ids:
            raise ParallelFrontierAuthorityError("parallel task tree identity mismatches")
        if task.task_id in by_id:
            raise ParallelFrontierError("parallel task set contains duplicate task identities")
        by_id[task.task_id] = task
    asleep: tuple[str, ...] = ()
    if causal_frontier is not None:
        if not isinstance(causal_frontier, CompiledFrontier):
            raise FederationContractError("causal_frontier must be CompiledFrontier")
        asleep_ids = {
            task.task_id for task in tasks if task.supervisor_id in causal_frontier.do_not_wake
        }
        asleep = tuple(sorted(asleep_ids))
    active = [task for task in tasks if task.task_id not in set(asleep)]
    if not active:
        empty = CompiledParallelFrontier(
            wave_id="wave:empty",
            admitted=(),
            serialized=(),
            suppressed=(),
            blocked=(),
            asleep=asleep,
            merge_order=(),
            assignment_refs=(),
        )
        return CompiledParallelFrontier(
            wave_id="wave:" + empty.cid,
            admitted=(),
            serialized=(),
            suppressed=(),
            blocked=(),
            asleep=asleep,
            merge_order=(),
            assignment_refs=(),
        )
    report = classify_intents(
        tuple(task.intent for task in active),
        independence=independence,
        ducklake_receipt=ducklake_receipt,
    )
    canonical = dict(report.canonical_task_ids)
    grouped: dict[str, list[str]] = {}
    for task_id, canonical_id in canonical.items():
        grouped.setdefault(canonical_id, []).append(task_id)
    suppressed_set: set[str] = set()
    for members in grouped.values():
        ordered = tuple(sorted(set(members)))
        if len(ordered) > 1:
            suppressed_set.update(ordered[1:])
    blocked_set = {
        relation.left_task_id
        if relation.covering_task_id == relation.right_task_id
        else relation.right_task_id
        for relation in report.relations
        if relation.disposition is IntentDisposition.SUBSUMED
    }
    blocked_set -= suppressed_set
    candidates = [
        task
        for task in sorted(
            active,
            key=lambda item: (
                0 if _irreversible(item.intent.effect_class) else 1,
                item.task_id,
            ),
        )
        if task.task_id not in suppressed_set and task.task_id not in blocked_set
    ]
    incompatible = _incompatible_pairs(report.relations)
    admitted: list[ParallelTask] = []
    serialized: list[ParallelTask] = []
    merge_used = 0
    proof_used = 0
    for task in candidates:
        if _cannot_admit(
            task,
            admitted=admitted,
            incompatible=incompatible,
            merge_used=merge_used,
            proof_used=proof_used,
            limits=limits,
        ):
            serialized.append(task)
            continue
        admitted.append(task)
        merge_used += int(task.requires_merge_slot)
        proof_used += int(task.requires_proof_slot)
    merge_order = tuple(
        task.task_id for task in (*admitted, *serialized) if task.requires_merge_slot
    )
    admitted_ids = tuple(task.task_id for task in admitted)
    serialized_ids = tuple(task.task_id for task in serialized)
    suppressed_ids = tuple(sorted(suppressed_set))
    blocked_ids = tuple(sorted(blocked_set))
    assignment_refs = tuple(
        "assignment:"
        + content_identity(
            {
                "task_id": task.task_id,
                "supervisor_id": task.supervisor_id,
                "subagent_id": task.subagent_id,
                "worktree_id": task.worktree_id,
                "lease_id": task.lease_id,
                "fencing_epoch": task.fencing_epoch,
                "merge_lane": task.merge_lane,
            }
        )
        for task in admitted
    )
    provisional = CompiledParallelFrontier(
        wave_id="wave:provisional",
        admitted=admitted_ids,
        serialized=serialized_ids,
        suppressed=suppressed_ids,
        blocked=blocked_ids,
        asleep=asleep,
        merge_order=merge_order,
        assignment_refs=assignment_refs,
    )
    return CompiledParallelFrontier(
        wave_id="wave:" + provisional.cid,
        admitted=admitted_ids,
        serialized=serialized_ids,
        suppressed=suppressed_ids,
        blocked=blocked_ids,
        asleep=asleep,
        merge_order=merge_order,
        assignment_refs=assignment_refs,
    )


def _cannot_admit(
    task: ParallelTask,
    *,
    admitted: Sequence[ParallelTask],
    incompatible: set[frozenset[str]],
    merge_used: int,
    proof_used: int,
    limits: FrontierCapacity,
) -> bool:
    if len(admitted) >= limits.max_parallel:
        return True
    if task.requires_merge_slot and merge_used >= limits.merge_slots:
        return True
    if task.requires_proof_slot and proof_used >= limits.proof_slots:
        return True
    if _irreversible(task.intent.effect_class) and admitted:
        return True
    for current in admitted:
        if _irreversible(current.intent.effect_class):
            return True
        if frozenset({task.task_id, current.task_id}) in incompatible:
            return True
        if task.subagent_id == current.subagent_id or task.lease_id == current.lease_id:
            return True
        if task.worktree_id == current.worktree_id and (
            _write_like(task.intent.effect_class) or _write_like(current.intent.effect_class)
        ):
            return True
    return False


def _frontier_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_insert_federation_receipt",
            """
            INSERT INTO federation_receipts (
                federation_receipt_id, tenant_id, federation_id, receipt_kind,
                federation_revision, control_plane_generation, event_watermark,
                issuer_id, content_ref, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "federation_receipt_id",
                "tenant_id",
                "federation_id",
                "receipt_kind",
                "federation_revision",
                "control_plane_generation",
                "event_watermark",
                "issuer_id",
                "content_ref",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_federation_receipt",
            """
            SELECT federation_receipt_id, receipt_kind, federation_revision,
                   event_watermark, content_ref
            FROM federation_receipts
            WHERE federation_receipt_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("federation_receipt_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_federation_task_binding",
            """
            INSERT INTO federation_task_bindings (
                federation_task_binding_id, tenant_id, federation_id, task_cid,
                repository_id, tree_id, goal_cid, subgoal_id, plan_revision_id,
                assignment_revision, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "federation_task_binding_id",
                "tenant_id",
                "federation_id",
                "task_cid",
                "repository_id",
                "tree_id",
                "goal_cid",
                "subgoal_id",
                "plan_revision_id",
                "assignment_revision",
                "status",
                "created_at",
                "updated_at",
            ),
        ),
        _template(
            "casf_select_federation_task_binding",
            """
            SELECT federation_task_binding_id, task_cid, tree_id, status,
                   assignment_revision
            FROM federation_task_bindings
            WHERE federation_task_binding_id = ? AND tenant_id = ?
              AND federation_id = ?
            LIMIT 1
            """,
            ("federation_task_binding_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
    )


class ParallelFrontierStore(DeduplicationStore):
    """Persist compiled parallel waves through the sealed state owner."""

    INTERFACE = "ParallelFrontierStore@1"

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
            raise ParallelFrontierError("parallel frontier store never accepts a database path")
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise ParallelFrontierError(
                "parallel frontier store requires an already-attached typed state client"
            )
        registered = set(client.list_templates())
        missing = [
            template.name for template in _frontier_templates() if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise ParallelFrontierError(
                    "parallel frontier templates are absent from the sealed catalog"
                )
        else:
            for template in _frontier_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    def record_frontier(
        self,
        compiled: CompiledParallelFrontier,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        tasks: Sequence[ParallelTask],
        event_watermark: int = 0,
    ) -> CausalGraphCommit:
        if not isinstance(compiled, CompiledParallelFrontier):
            raise FederationContractError("compiled parallel frontier is required")
        receipt_id = "federation-receipt:" + compiled.cid
        by_id = {task.task_id: task for task in tasks}
        return self._commit_fact(
            operation="federation.parallel.frontier.record",
            fact_id=receipt_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(
                dict.fromkeys((receipt_id, compiled.wave_id, *compiled.admitted[:8]))
            ),
            payload_ref=compiled.cid,
            prepare_fact=lambda: self._prepare_frontier(
                receipt_id,
                tenant_id=binding.tenant_id,
                federation_id=federation_id,
            ),
            apply_fact=lambda revision, recorded_at: self._insert_frontier(
                compiled,
                receipt_id=receipt_id,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                binding=binding,
                tasks=by_id,
                graph_revision=revision,
                event_watermark=event_watermark,
                recorded_at=recorded_at,
            ),
        )

    def load_frontier(
        self,
        *,
        receipt_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_federation_receipt",
            {
                "federation_receipt_id": _identifier(receipt_id, "receipt_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise ParallelFrontierError("parallel frontier receipt is absent")
        return dict(rows[0])

    def load_task_binding(
        self,
        *,
        binding_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_federation_task_binding",
            {
                "federation_task_binding_id": _identifier(binding_id, "binding_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise ParallelFrontierError("parallel task binding is absent")
        return dict(rows[0])

    def _prepare_frontier(self, receipt_id: str, *, tenant_id: str, federation_id: str) -> None:
        existing = self._client.execute(
            "casf_select_federation_receipt",
            {
                "federation_receipt_id": receipt_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if existing:
            raise ParallelFrontierError("parallel frontier receipt identity is already bound")

    def _insert_frontier(
        self,
        compiled: CompiledParallelFrontier,
        *,
        receipt_id: str,
        federation_id: str,
        tenant_id: str,
        binding: FederationBinding,
        tasks: Mapping[str, ParallelTask],
        graph_revision: int,
        event_watermark: int,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_federation_receipt",
            {
                "federation_receipt_id": receipt_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "receipt_kind": "parallel_frontier",
                "federation_revision": graph_revision,
                "control_plane_generation": binding.control_plane_generation,
                "event_watermark": _integer(event_watermark, "event_watermark"),
                "issuer_id": "parallel-frontier",
                "content_ref": compiled.cid,
                "recorded_at": recorded_at,
            },
        )
        status_by_task = {
            **{task_id: "admitted_parallel" for task_id in compiled.admitted},
            **{task_id: "serialized" for task_id in compiled.serialized},
            **{task_id: "suppressed" for task_id in compiled.suppressed},
            **{task_id: "blocked" for task_id in compiled.blocked},
            **{task_id: "asleep" for task_id in compiled.asleep},
        }
        for task_id, status in sorted(status_by_task.items()):
            if status not in ADMISSION_STATUSES:
                raise ParallelFrontierError("admission status is not closed")
            task = tasks[task_id]
            binding_id = "task-binding:" + content_identity(
                {"wave_id": compiled.wave_id, "task_id": task_id, "status": status}
            )
            self._client.execute(
                "casf_insert_federation_task_binding",
                {
                    "federation_task_binding_id": binding_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "task_cid": task.intent.cid,
                    "repository_id": task.intent.repository_id,
                    "tree_id": task.intent.tree_id,
                    "goal_cid": task.intent.goal_id,
                    "subgoal_id": task.intent.subgoal_id,
                    "plan_revision_id": compiled.wave_id,
                    "assignment_revision": graph_revision,
                    "status": status,
                    "created_at": recorded_at,
                    "updated_at": recorded_at,
                },
            )


__all__ = (
    "CompiledParallelFrontier",
    "FrontierCapacity",
    "IRREVERSIBLE_EFFECT_CLASSES",
    "ParallelFrontierAuthorityError",
    "ParallelFrontierError",
    "ParallelFrontierStore",
    "ParallelTask",
    "bind_parallel_task",
    "compile_parallel_frontier",
    "refuse_ducklake_parallel_authority",
)
