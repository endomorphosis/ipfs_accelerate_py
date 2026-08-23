"""Conflict-free supervisor sharding and bounded specialization for CASF.

Each task is owned by exactly one shard. Exclusive write effects cannot be
claimed by two specializations. A supervisor may only own shards inside its
repository, goal, and effect-class ceiling. DuckLake and missing capability
fail closed.
"""

from __future__ import annotations

from collections import defaultdict
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
    utc_now,
)
from .deduplication import WRITE_EFFECT_CLASSES
from .events import EventEffectClass
from .parallel_frontier import ParallelFrontierStore
from .registry import _template
from .retrieval_projection import retrieval_establishes_authority

EXCLUSIVE_EFFECT_CLASSES = frozenset(item.value for item in WRITE_EFFECT_CLASSES)
CLOSED_EFFECT_CLASSES = frozenset(item.value for item in EventEffectClass)
MAX_SHARD_TASKS = 10_000


class ShardingError(CausalGraphError):
    """Base typed supervisor-sharding failure."""


class ShardingAuthorityError(FederationAuthorityError, ShardingError):
    """An attempt to overlap exclusive work or exceed specialization ceilings."""


def refuse_ducklake_shard_authority(receipt: Mapping[str, Any] | None) -> None:
    if not receipt:
        return
    if receipt.get("authoritative") is True or receipt.get("schedules") is True:
        raise ShardingAuthorityError("DuckLake cannot admit supervisor shards")


def _effect_class(value: EventEffectClass | str, name: str = "effect_class") -> str:
    if isinstance(value, EventEffectClass):
        return value.value
    text = _identifier(value, name)
    if text not in CLOSED_EFFECT_CLASSES:
        raise FederationContractError("effect_class is not closed")
    return text


def _exclusive(effect_class: str) -> bool:
    return effect_class in EXCLUSIVE_EFFECT_CLASSES


@dataclass(frozen=True)
class ShardWork:
    """One task considered for exact shard placement."""

    SCHEMA: ClassVar[str] = "ipfs_accelerate_py/agent-supervisor/causal-federation/shard-work@1"

    task_id: str
    repository_id: str
    tree_id: str
    goal_id: str
    effect_class: str
    symbol_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("task_id", "repository_id", "tree_id", "goal_id"):
            _identifier(getattr(self, name), name)
        object.__setattr__(self, "effect_class", _effect_class(self.effect_class))
        _strings(self.symbol_refs, "symbol_refs", maximum=1_024, required=False)


@dataclass(frozen=True)
class SupervisorSpecializationBound:
    """Capability and effect ceiling that bounds which shards a supervisor may own."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/supervisor-specialization-bound@1"
    )

    supervisor_id: str
    allowed_repository_ids: tuple[str, ...]
    allowed_goal_refs: tuple[str, ...]
    allowed_effect_classes: tuple[str, ...]
    capability_refs: tuple[str, ...]
    max_shards: int = 1

    def __post_init__(self) -> None:
        _identifier(self.supervisor_id, "supervisor_id")
        _strings(self.allowed_repository_ids, "allowed_repository_ids", maximum=256, required=True)
        _strings(self.allowed_goal_refs, "allowed_goal_refs", maximum=4_096, required=False)
        classes = tuple(
            _effect_class(item, "allowed_effect_classes") for item in self.allowed_effect_classes
        )
        if not classes:
            raise FederationContractError("specialization requires at least one effect class")
        if len(set(classes)) != len(classes):
            raise FederationContractError("allowed_effect_classes contains duplicates")
        object.__setattr__(self, "allowed_effect_classes", classes)
        _strings(self.capability_refs, "capability_refs", maximum=256, required=True)
        _integer(self.max_shards, "max_shards", minimum=1, maximum=64)

    def admits(self, work: ShardWork) -> bool:
        if work.repository_id not in self.allowed_repository_ids:
            return False
        if self.allowed_goal_refs and work.goal_id not in self.allowed_goal_refs:
            return False
        return work.effect_class in self.allowed_effect_classes


def bind_supervisor_specialization(
    *,
    binding: FederationBinding,
    supervisor_id: str,
    allowed_repository_ids: Sequence[str] | None = None,
    allowed_goal_refs: Sequence[str] = (),
    allowed_effect_classes: Sequence[str],
    capability_refs: Sequence[str],
    max_shards: int = 1,
) -> SupervisorSpecializationBound:
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    if retrieval_establishes_authority() is not False:
        raise ShardingAuthorityError("retrieval cannot mint shard specialization")
    repos = tuple(allowed_repository_ids or binding.repository_ids)
    unknown = set(repos) - set(binding.repository_ids)
    if unknown:
        raise ShardingAuthorityError("specialization repository is not bound")
    return SupervisorSpecializationBound(
        supervisor_id=supervisor_id,
        allowed_repository_ids=repos,
        allowed_goal_refs=tuple(allowed_goal_refs),
        allowed_effect_classes=tuple(allowed_effect_classes),
        capability_refs=tuple(capability_refs),
        max_shards=max_shards,
    )


@dataclass(frozen=True)
class CompiledShard:
    """One supervisor-owned shard with exact, non-overlapping exclusive work."""

    SCHEMA: ClassVar[str] = "ipfs_accelerate_py/agent-supervisor/causal-federation/compiled-shard@1"

    shard_id: str
    supervisor_id: str
    fencing_epoch: int
    revision: int
    repository_ids: tuple[str, ...]
    goal_refs: tuple[str, ...]
    task_ids: tuple[str, ...]
    symbol_refs: tuple[str, ...]
    effect_classes: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.shard_id, "shard_id")
        _identifier(self.supervisor_id, "supervisor_id")
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        _integer(self.revision, "revision", minimum=1)
        _strings(self.repository_ids, "repository_ids", maximum=256, required=True)
        _strings(self.goal_refs, "goal_refs", maximum=4_096, required=False)
        _strings(self.task_ids, "task_ids", maximum=MAX_SHARD_TASKS, required=True)
        _strings(self.symbol_refs, "symbol_refs", maximum=10_000, required=False)
        classes = tuple(_effect_class(item, "effect_classes") for item in self.effect_classes)
        object.__setattr__(self, "effect_classes", tuple(dict.fromkeys(classes)))

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "shard_id": self.shard_id,
                "supervisor_id": self.supervisor_id,
                "task_ids": list(self.task_ids),
                "repository_ids": list(self.repository_ids),
                "effect_classes": list(self.effect_classes),
            }
        )


@dataclass(frozen=True)
class CompiledShardPlan:
    """Complete conflict-free shard partition for one assignment revision."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/compiled-shard-plan@1"
    )

    shards: tuple[CompiledShard, ...]
    assignment_revision: int
    fencing_epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.shards, tuple) or not all(
            isinstance(item, CompiledShard) for item in self.shards
        ):
            raise FederationContractError("shards must be CompiledShard records")
        _integer(self.assignment_revision, "assignment_revision", minimum=1)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        task_ids = [task_id for shard in self.shards for task_id in shard.task_ids]
        if len(task_ids) != len(set(task_ids)):
            raise ShardingAuthorityError("shard plan assigns a task to more than one owner")

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "shards": [item.cid for item in self.shards],
                "assignment_revision": self.assignment_revision,
                "fencing_epoch": self.fencing_epoch,
            }
        )


def compile_supervisor_shards(
    work: Sequence[ShardWork],
    specializations: Sequence[SupervisorSpecializationBound],
    *,
    binding: FederationBinding,
    fencing_epoch: int = 1,
    assignment_revision: int = 1,
    ducklake_receipt: Mapping[str, Any] | None = None,
) -> CompiledShardPlan:
    """Partition work onto exact specialized shards. Exclusive overlap fails closed."""

    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    refuse_ducklake_shard_authority(ducklake_receipt)
    if retrieval_establishes_authority() is not False:
        raise ShardingAuthorityError("retrieval cannot mint shard placement")
    if not work:
        raise FederationContractError("sharding requires at least one work item")
    if not specializations:
        raise ShardingAuthorityError("supervisor specialization capability is missing")
    seen_work: set[str] = set()
    for item in work:
        if not isinstance(item, ShardWork):
            raise FederationContractError("work items must be ShardWork records")
        if item.tree_id not in binding.repository_tree_ids:
            raise ShardingAuthorityError("shard work tree identity mismatches")
        if item.repository_id not in binding.repository_ids:
            raise ShardingAuthorityError("shard work repository is not bound")
        if item.task_id in seen_work:
            raise ShardingError("shard work contains duplicate task identities")
        seen_work.add(item.task_id)
    seen_supervisors: set[str] = set()
    for spec in specializations:
        if not isinstance(spec, SupervisorSpecializationBound):
            raise FederationContractError("specializations must be SupervisorSpecializationBound")
        if spec.supervisor_id in seen_supervisors:
            raise ShardingError("specialization set contains duplicate supervisors")
        seen_supervisors.add(spec.supervisor_id)
    assigned: dict[str, list[ShardWork]] = defaultdict(list)
    for item in work:
        matches = tuple(spec for spec in specializations if spec.admits(item))
        if not matches:
            raise ShardingAuthorityError(f"no specialized supervisor can admit {item.task_id}")
        if _exclusive(item.effect_class) and len(matches) > 1:
            raise ShardingAuthorityError(
                "exclusive shard work cannot be claimed by two specializations"
            )
        owner = sorted(matches, key=lambda spec: spec.supervisor_id)[0]
        assigned[owner.supervisor_id].append(item)
    shards: list[CompiledShard] = []
    specs_by_id = {spec.supervisor_id: spec for spec in specializations}
    for supervisor_id, items in sorted(assigned.items()):
        spec = specs_by_id[supervisor_id]
        if spec.max_shards < 1:
            raise ShardingAuthorityError("specialization max_shards is below one")
        task_ids = tuple(item.task_id for item in sorted(items, key=lambda item: item.task_id))
        repos = tuple(dict.fromkeys(item.repository_id for item in items))
        goals = tuple(dict.fromkeys(item.goal_id for item in items))
        symbols = tuple(dict.fromkeys(symbol for item in items for symbol in item.symbol_refs))
        effects = tuple(dict.fromkeys(item.effect_class for item in items))
        if set(repos) - set(spec.allowed_repository_ids):
            raise ShardingAuthorityError("shard repository exceeds specialization ceiling")
        if spec.allowed_goal_refs and set(goals) - set(spec.allowed_goal_refs):
            raise ShardingAuthorityError("shard goal exceeds specialization ceiling")
        if set(effects) - set(spec.allowed_effect_classes):
            raise ShardingAuthorityError("shard effect class exceeds specialization ceiling")
        shards.append(
            CompiledShard(
                shard_id="shard:"
                + content_identity(
                    {
                        "supervisor_id": supervisor_id,
                        "task_ids": list(task_ids),
                        "assignment_revision": assignment_revision,
                    }
                ),
                supervisor_id=supervisor_id,
                fencing_epoch=fencing_epoch,
                revision=assignment_revision,
                repository_ids=repos,
                goal_refs=goals,
                task_ids=task_ids,
                symbol_refs=symbols,
                effect_classes=effects,
            )
        )
    return CompiledShardPlan(
        shards=tuple(shards),
        assignment_revision=assignment_revision,
        fencing_epoch=fencing_epoch,
    )


def _shard_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_insert_supervisor_shard",
            """
            INSERT INTO supervisor_shards (
                shard_id, tenant_id, federation_id, shard_kind, active_revision,
                fencing_epoch, state, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "shard_id",
                "tenant_id",
                "federation_id",
                "shard_kind",
                "active_revision",
                "fencing_epoch",
                "state",
                "created_at",
                "updated_at",
            ),
        ),
        _template(
            "casf_select_supervisor_shard",
            """
            SELECT shard_id, shard_kind, active_revision, fencing_epoch, state
            FROM supervisor_shards
            WHERE shard_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("shard_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_shard_boundary",
            """
            INSERT INTO shard_boundaries (
                shard_boundary_id, tenant_id, federation_id, shard_id,
                shard_revision, repository_id, tree_id, goal_ref, subgoal_ref,
                task_family, resource_class, boundary_kind, boundary_ref,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "shard_boundary_id",
                "tenant_id",
                "federation_id",
                "shard_id",
                "shard_revision",
                "repository_id",
                "tree_id",
                "goal_ref",
                "subgoal_ref",
                "task_family",
                "resource_class",
                "boundary_kind",
                "boundary_ref",
                "created_at",
            ),
        ),
        _template(
            "casf_select_shard_boundaries",
            """
            SELECT boundary_kind, boundary_ref
            FROM shard_boundaries
            WHERE shard_id = ? AND shard_revision = ?
            ORDER BY boundary_kind, boundary_ref
            """,
            ("shard_id", "shard_revision"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_shard_assignment",
            """
            INSERT INTO shard_assignments (
                shard_assignment_id, tenant_id, federation_id, shard_id,
                shard_revision, supervisor_id, assignment_revision,
                fencing_epoch, state, activated_at, retired_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "shard_assignment_id",
                "tenant_id",
                "federation_id",
                "shard_id",
                "shard_revision",
                "supervisor_id",
                "assignment_revision",
                "fencing_epoch",
                "state",
                "activated_at",
                "retired_at",
            ),
        ),
        _template(
            "casf_select_shard_assignment",
            """
            SELECT shard_assignment_id, supervisor_id, shard_revision, state
            FROM shard_assignments
            WHERE shard_assignment_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("shard_assignment_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
    )


class ShardingStore(ParallelFrontierStore):
    """Persist compiled shards through the sealed state owner."""

    INTERFACE = "ShardingStore@1"

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
            raise ShardingError("sharding store never accepts a database path")
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise ShardingError("sharding store requires an already-attached typed state client")
        registered = set(client.list_templates())
        missing = [
            template.name for template in _shard_templates() if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise ShardingError("sharding templates are absent from the sealed catalog")
        else:
            for template in _shard_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    def record_plan(
        self,
        plan: CompiledShardPlan,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
    ) -> CausalGraphCommit:
        if not isinstance(plan, CompiledShardPlan):
            raise FederationContractError("compiled shard plan is required")
        plan_id = "shard-plan:" + plan.cid
        return self._commit_fact(
            operation="federation.shard.plan.record",
            fact_id=plan_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(
                dict.fromkeys((plan_id, *(shard.shard_id for shard in plan.shards)))
            ),
            payload_ref=plan.cid,
            prepare_fact=lambda: None,
            apply_fact=lambda revision, recorded_at: self._insert_plan(
                plan,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                tree_id=binding.repository_tree_ids[0],
                graph_revision=revision,
                recorded_at=recorded_at or utc_now(),
            ),
        )

    def load_shard(
        self,
        *,
        shard_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_supervisor_shard",
            {
                "shard_id": _identifier(shard_id, "shard_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise ShardingError("supervisor shard is absent")
        return dict(rows[0])

    def load_assignment(
        self,
        *,
        assignment_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_shard_assignment",
            {
                "shard_assignment_id": _identifier(assignment_id, "assignment_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise ShardingError("shard assignment is absent")
        return dict(rows[0])

    def _insert_plan(
        self,
        plan: CompiledShardPlan,
        *,
        federation_id: str,
        tenant_id: str,
        tree_id: str,
        graph_revision: int,
        recorded_at: str,
    ) -> None:
        del graph_revision
        for shard in plan.shards:
            self._client.execute(
                "casf_insert_supervisor_shard",
                {
                    "shard_id": shard.shard_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "shard_kind": "specialized",
                    "active_revision": shard.revision,
                    "fencing_epoch": shard.fencing_epoch,
                    "state": "active",
                    "created_at": recorded_at,
                    "updated_at": recorded_at,
                },
            )
            assignment_id = "shard-assignment:" + content_identity(
                {
                    "shard_id": shard.shard_id,
                    "supervisor_id": shard.supervisor_id,
                    "revision": shard.revision,
                }
            )
            self._client.execute(
                "casf_insert_shard_assignment",
                {
                    "shard_assignment_id": assignment_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "shard_id": shard.shard_id,
                    "shard_revision": shard.revision,
                    "supervisor_id": shard.supervisor_id,
                    "assignment_revision": plan.assignment_revision,
                    "fencing_epoch": shard.fencing_epoch,
                    "state": "active",
                    "activated_at": recorded_at,
                    "retired_at": "",
                },
            )
            for kind, refs in (
                ("repository", shard.repository_ids),
                ("goal", shard.goal_refs),
                ("task", shard.task_ids),
                ("symbol", shard.symbol_refs),
                ("effect_class", shard.effect_classes),
            ):
                for ordinal, ref in enumerate(refs):
                    self._client.execute(
                        "casf_insert_shard_boundary",
                        {
                            "shard_boundary_id": "shard-boundary:"
                            + content_identity(
                                {
                                    "shard_id": shard.shard_id,
                                    "kind": kind,
                                    "ref": ref,
                                    "ordinal": ordinal,
                                }
                            ),
                            "tenant_id": tenant_id,
                            "federation_id": federation_id,
                            "shard_id": shard.shard_id,
                            "shard_revision": shard.revision,
                            "repository_id": shard.repository_ids[0],
                            "tree_id": tree_id,
                            "goal_ref": shard.goal_refs[0] if shard.goal_refs else "",
                            "subgoal_ref": "",
                            "task_family": "",
                            "resource_class": "",
                            "boundary_kind": kind,
                            "boundary_ref": ref,
                            "created_at": recorded_at,
                        },
                    )


__all__ = (
    "CompiledShard",
    "CompiledShardPlan",
    "EXCLUSIVE_EFFECT_CLASSES",
    "ShardWork",
    "ShardingAuthorityError",
    "ShardingError",
    "ShardingStore",
    "SupervisorSpecializationBound",
    "bind_supervisor_specialization",
    "compile_supervisor_shards",
    "refuse_ducklake_shard_authority",
)
