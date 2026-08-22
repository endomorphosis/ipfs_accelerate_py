"""Duplicate-work and task-subsumption detection for CASF.

Task-intent identity binds tree, goal/subgoal, operation, targets, acceptance,
effect class, and validation. Exact duplicates share one task and result;
subsumed work depends on its covering task; overlap receives explicit
boundaries; conflicts serialize. Only proved or policy-admitted independence
may run concurrently. Unknown overlap serializes. Retrieval and model
statements of independence remain nomination-only.
"""

# Python 3.8 compatibility requires ``str, Enum`` rather than ``StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
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
    _strings,
)
from .events import EventEffectClass
from .registry import _template
from .retrieval_projection import retrieval_establishes_authority
from .scheduler import FederationSchedulerStore

WRITE_EFFECT_CLASSES = frozenset(
    {
        EventEffectClass.AUTHORITATIVE_STATE,
        EventEffectClass.LEASE_OR_FENCE,
        EventEffectClass.EXTERNAL_IRREVERSIBLE,
        EventEffectClass.SECURITY_OR_LEGAL,
        EventEffectClass.PAYMENT,
        EventEffectClass.PROOF_LINEAGE,
    }
)
RESOLUTION_KINDS = frozenset(
    {
        "share_result",
        "depend",
        "bound",
        "serialize",
        "admit_parallel",
    }
)
MAX_INTENT_TARGETS = 4_096
MAX_INTENT_SET = 1_024


class DeduplicationError(CausalGraphError):
    """Base typed task-intent classification failure."""


class DeduplicationAuthorityError(FederationAuthorityError, DeduplicationError):
    """An attempt to mint independence, suppress conflict, or ignore tree identity."""


class IntentDisposition(str, Enum):
    DUPLICATE = "duplicate"
    SUBSUMED = "subsumed"
    OVERLAP = "overlap"
    CONFLICT = "conflict"
    INDEPENDENT = "independent"


def _write_like(effect_class: EventEffectClass) -> bool:
    return effect_class in WRITE_EFFECT_CLASSES


def refuse_ducklake_dedup_authority(receipt: Mapping[str, Any] | None) -> None:
    if not receipt:
        return
    if (
        receipt.get("authoritative") is True
        or receipt.get("schedules") is True
        or receipt.get("deduplicates") is True
    ):
        raise DeduplicationAuthorityError(
            "DuckLake cannot admit duplicate-work or independence authority"
        )


@dataclass(frozen=True)
class TaskIntentIdentity:
    """Canonical task-intent identity used to detect duplicate and subsumed work."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/task-intent-identity@1"
    )

    tree_id: str
    repository_id: str
    goal_id: str
    subgoal_id: str
    operation: str
    targets: tuple[str, ...]
    acceptance_ref: str
    effect_class: EventEffectClass
    validation_ref: str
    task_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "tree_id",
            "repository_id",
            "goal_id",
            "subgoal_id",
            "operation",
            "acceptance_ref",
            "validation_ref",
        ):
            _identifier(getattr(self, name), name)
        targets = _strings(self.targets, "targets", maximum=MAX_INTENT_TARGETS, required=True)
        object.__setattr__(self, "targets", tuple(sorted(targets)))
        if not isinstance(self.effect_class, EventEffectClass):
            raise FederationContractError("effect_class is not closed")
        task_id = self.task_id
        if not task_id:
            task_id = "task:" + self.cid
            object.__setattr__(self, "task_id", task_id)
        else:
            _identifier(task_id, "task_id")

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "tree_id": self.tree_id,
                "repository_id": self.repository_id,
                "goal_id": self.goal_id,
                "subgoal_id": self.subgoal_id,
                "operation": self.operation,
                "targets": list(self.targets),
                "acceptance_ref": self.acceptance_ref,
                "effect_class": self.effect_class.value,
                "validation_ref": self.validation_ref,
            }
        )

    @property
    def identity_core(self) -> tuple[str, ...]:
        return (
            self.tree_id,
            self.repository_id,
            self.goal_id,
            self.subgoal_id,
            self.operation,
            self.acceptance_ref,
            self.effect_class.value,
            self.validation_ref,
        )


@dataclass(frozen=True)
class IntentIndependenceAdmission:
    """Proved or policy-admitted independence used only for concurrent intents."""

    left_intent_cid: str
    right_intent_cid: str
    evidence_refs: tuple[str, ...]
    authoritative: bool
    policy_admitted: bool = False

    def __post_init__(self) -> None:
        left = _identifier(self.left_intent_cid, "left_intent_cid")
        right = _identifier(self.right_intent_cid, "right_intent_cid")
        if left == right:
            raise FederationContractError("independence cannot name one intent twice")
        if left > right:
            object.__setattr__(self, "left_intent_cid", right)
            object.__setattr__(self, "right_intent_cid", left)
        refs = _strings(self.evidence_refs, "evidence_refs", maximum=256, required=True)
        object.__setattr__(self, "evidence_refs", refs)
        if type(self.authoritative) is not bool or type(self.policy_admitted) is not bool:
            raise FederationContractError("independence flags must be boolean")
        if self.authoritative is False and self.policy_admitted is False:
            raise DeduplicationAuthorityError("retrieval nominations cannot prove independence")


@dataclass(frozen=True)
class IntentRelation:
    """Closed classification of one unordered pair of task intents."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/intent-relation@1"
    )

    left_task_id: str
    right_task_id: str
    left_intent_cid: str
    right_intent_cid: str
    disposition: IntentDisposition
    resolution_kind: str
    canonical_task_id: str
    covering_task_id: str = ""
    boundary_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    nomination_only: bool = False

    def __post_init__(self) -> None:
        for name in (
            "left_task_id",
            "right_task_id",
            "left_intent_cid",
            "right_intent_cid",
            "canonical_task_id",
        ):
            _identifier(getattr(self, name), name)
        _identifier(self.covering_task_id, "covering_task_id", required=False)
        if not isinstance(self.disposition, IntentDisposition):
            raise FederationContractError("intent disposition is not closed")
        if self.resolution_kind not in RESOLUTION_KINDS:
            raise FederationContractError("resolution kind is not closed")
        _strings(self.boundary_refs, "boundary_refs", maximum=MAX_INTENT_TARGETS, required=False)
        _strings(self.evidence_refs, "evidence_refs", maximum=256, required=True)
        if type(self.nomination_only) is not bool:
            raise FederationContractError("nomination_only must be boolean")
        if self.nomination_only and self.disposition is IntentDisposition.INDEPENDENT:
            raise DeduplicationAuthorityError(
                "nomination-only independence cannot admit parallel effects"
            )
        if (
            self.disposition is IntentDisposition.DUPLICATE
            and self.resolution_kind != "share_result"
        ):
            raise FederationContractError("duplicates must share one result")
        if (
            self.disposition is IntentDisposition.INDEPENDENT
            and self.resolution_kind != "admit_parallel"
        ):
            raise FederationContractError("independent intents must admit parallel resolution")

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "left_intent_cid": self.left_intent_cid,
                "right_intent_cid": self.right_intent_cid,
                "disposition": self.disposition.value,
                "resolution_kind": self.resolution_kind,
                "canonical_task_id": self.canonical_task_id,
                "covering_task_id": self.covering_task_id,
                "boundary_refs": list(self.boundary_refs),
            }
        )


@dataclass(frozen=True)
class DeduplicationReport:
    """Complete pairwise classification for one intent set."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/deduplication-report@1"
    )

    relations: tuple[IntentRelation, ...]
    canonical_task_ids: tuple[tuple[str, str], ...]
    duplicate_pairs: tuple[str, ...]
    serial_pairs: tuple[str, ...]
    parallel_pairs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.relations, tuple) or not all(
            isinstance(item, IntentRelation) for item in self.relations
        ):
            raise FederationContractError("relations must be IntentRelation records")
        for pair in self.canonical_task_ids:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise FederationContractError("canonical_task_ids must be identity pairs")
            _identifier(pair[0], "canonical_task_ids")
            _identifier(pair[1], "canonical_task_ids")

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "relations": [item.cid for item in self.relations],
                "canonical_task_ids": [list(item) for item in self.canonical_task_ids],
            }
        )


def bind_task_intent(
    *,
    binding: FederationBinding,
    goal_id: str,
    operation: str,
    targets: Sequence[str],
    acceptance_ref: str,
    effect_class: EventEffectClass | str,
    validation_ref: str,
    subgoal_id: str = "",
    task_id: str = "",
    tree_id: str = "",
    repository_id: str = "",
) -> TaskIntentIdentity:
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    if retrieval_establishes_authority() is not False:
        raise DeduplicationAuthorityError("retrieval cannot mint task-intent authority")
    if isinstance(effect_class, str):
        try:
            effect_class = EventEffectClass(effect_class)
        except ValueError as exc:
            raise FederationContractError("effect_class is not closed") from exc
    intent = TaskIntentIdentity(
        tree_id=tree_id or binding.repository_tree_ids[0],
        repository_id=repository_id or binding.repository_ids[0],
        goal_id=goal_id,
        subgoal_id=subgoal_id or "subgoal:default",
        operation=operation,
        targets=tuple(targets),
        acceptance_ref=acceptance_ref,
        effect_class=effect_class,
        validation_ref=validation_ref,
        task_id=task_id,
    )
    if intent.tree_id not in binding.repository_tree_ids:
        raise DeduplicationAuthorityError("task intent tree identity mismatches")
    if intent.repository_id not in binding.repository_ids:
        raise DeduplicationAuthorityError("task intent repository is not bound")
    return intent


def _pair_order(
    left: TaskIntentIdentity, right: TaskIntentIdentity
) -> tuple[TaskIntentIdentity, TaskIntentIdentity]:
    if left.cid <= right.cid:
        return left, right
    return right, left


def _independence_for(
    left: TaskIntentIdentity,
    right: TaskIntentIdentity,
    admissions: Sequence[IntentIndependenceAdmission],
) -> IntentIndependenceAdmission | None:
    key = tuple(sorted((left.cid, right.cid)))
    matched = [item for item in admissions if (item.left_intent_cid, item.right_intent_cid) == key]
    if len(matched) > 1:
        raise DeduplicationError("independence admission is duplicated")
    return matched[0] if matched else None


def classify_intent_pair(
    left: TaskIntentIdentity,
    right: TaskIntentIdentity,
    *,
    independence: IntentIndependenceAdmission | None = None,
    ducklake_receipt: Mapping[str, Any] | None = None,
) -> IntentRelation:
    """Classify one pair. Unknown or nominated independence never admits parallel work."""

    if not isinstance(left, TaskIntentIdentity) or not isinstance(right, TaskIntentIdentity):
        raise FederationContractError("intents must be TaskIntentIdentity records")
    refuse_ducklake_dedup_authority(ducklake_receipt)
    first, second = _pair_order(left, right)
    if independence is not None and not isinstance(independence, IntentIndependenceAdmission):
        raise FederationContractError("independence must be IntentIndependenceAdmission")
    left_targets = set(first.targets)
    right_targets = set(second.targets)
    intersection = left_targets & right_targets
    if independence is not None:
        expected = tuple(sorted((first.cid, second.cid)))
        observed = (independence.left_intent_cid, independence.right_intent_cid)
        if observed != expected:
            raise DeduplicationError("independence admission does not name this pair")
        if intersection:
            raise DeduplicationAuthorityError(
                "independence cannot suppress overlapping task targets"
            )
        return IntentRelation(
            left_task_id=first.task_id,
            right_task_id=second.task_id,
            left_intent_cid=first.cid,
            right_intent_cid=second.cid,
            disposition=IntentDisposition.INDEPENDENT,
            resolution_kind="admit_parallel",
            canonical_task_id=first.task_id,
            evidence_refs=independence.evidence_refs,
            nomination_only=False,
        )
    if first.cid == second.cid:
        canonical = "task:" + first.cid
        return IntentRelation(
            left_task_id=first.task_id,
            right_task_id=second.task_id,
            left_intent_cid=first.cid,
            right_intent_cid=second.cid,
            disposition=IntentDisposition.DUPLICATE,
            resolution_kind="share_result",
            canonical_task_id=canonical,
            evidence_refs=("dedup:exact-intent",),
        )
    same_core = first.identity_core == second.identity_core
    write_overlap = bool(intersection) and (
        _write_like(first.effect_class) or _write_like(second.effect_class)
    )
    if same_core and left_targets < right_targets:
        return IntentRelation(
            left_task_id=first.task_id,
            right_task_id=second.task_id,
            left_intent_cid=first.cid,
            right_intent_cid=second.cid,
            disposition=IntentDisposition.SUBSUMED,
            resolution_kind="depend",
            canonical_task_id=second.task_id,
            covering_task_id=second.task_id,
            evidence_refs=("dedup:strict-subset",),
        )
    if same_core and right_targets < left_targets:
        return IntentRelation(
            left_task_id=first.task_id,
            right_task_id=second.task_id,
            left_intent_cid=first.cid,
            right_intent_cid=second.cid,
            disposition=IntentDisposition.SUBSUMED,
            resolution_kind="depend",
            canonical_task_id=first.task_id,
            covering_task_id=first.task_id,
            evidence_refs=("dedup:strict-subset",),
        )
    if not intersection:
        return IntentRelation(
            left_task_id=first.task_id,
            right_task_id=second.task_id,
            left_intent_cid=first.cid,
            right_intent_cid=second.cid,
            disposition=IntentDisposition.CONFLICT,
            resolution_kind="serialize",
            canonical_task_id=first.task_id,
            evidence_refs=("dedup:unknown-serializes",),
        )
    if left_targets == right_targets or write_overlap:
        return IntentRelation(
            left_task_id=first.task_id,
            right_task_id=second.task_id,
            left_intent_cid=first.cid,
            right_intent_cid=second.cid,
            disposition=IntentDisposition.CONFLICT,
            resolution_kind="serialize",
            canonical_task_id=first.task_id,
            evidence_refs=("dedup:conflict-serialize",),
        )
    return IntentRelation(
        left_task_id=first.task_id,
        right_task_id=second.task_id,
        left_intent_cid=first.cid,
        right_intent_cid=second.cid,
        disposition=IntentDisposition.OVERLAP,
        resolution_kind="bound",
        canonical_task_id=first.task_id,
        boundary_refs=tuple(sorted(intersection)),
        evidence_refs=("dedup:overlap-boundary",),
    )


def classify_intents(
    intents: Sequence[TaskIntentIdentity],
    *,
    independence: Sequence[IntentIndependenceAdmission] = (),
    ducklake_receipt: Mapping[str, Any] | None = None,
) -> DeduplicationReport:
    """Classify every pair. Duplicate clusters share one canonical task identity."""

    refuse_ducklake_dedup_authority(ducklake_receipt)
    if not intents:
        raise FederationContractError("deduplication requires at least one intent")
    if len(intents) > MAX_INTENT_SET:
        raise DeduplicationError("intent set exceeds bound")
    unique: list[TaskIntentIdentity] = []
    seen_task_ids: set[str] = set()
    for intent in intents:
        if not isinstance(intent, TaskIntentIdentity):
            raise FederationContractError("intents must be TaskIntentIdentity records")
        if intent.task_id in seen_task_ids:
            raise DeduplicationError("intent set contains duplicate task identities")
        seen_task_ids.add(intent.task_id)
        unique.append(intent)
    relations: list[IntentRelation] = []
    for index, left in enumerate(unique):
        for right in unique[index + 1 :]:
            relations.append(
                classify_intent_pair(
                    left,
                    right,
                    independence=_independence_for(left, right, independence),
                    ducklake_receipt=ducklake_receipt,
                )
            )
    canonical: dict[str, str] = {}
    for intent in unique:
        canonical[intent.task_id] = intent.task_id
    for relation in relations:
        if relation.disposition is IntentDisposition.DUPLICATE:
            canonical[relation.left_task_id] = relation.canonical_task_id
            canonical[relation.right_task_id] = relation.canonical_task_id
    duplicate_pairs = tuple(
        item.cid for item in relations if item.disposition is IntentDisposition.DUPLICATE
    )
    serial_pairs = tuple(
        item.cid
        for item in relations
        if item.disposition in {IntentDisposition.CONFLICT, IntentDisposition.OVERLAP}
    )
    parallel_pairs = tuple(
        item.cid for item in relations if item.disposition is IntentDisposition.INDEPENDENT
    )
    return DeduplicationReport(
        relations=tuple(relations),
        canonical_task_ids=tuple(sorted(canonical.items())),
        duplicate_pairs=duplicate_pairs,
        serial_pairs=serial_pairs,
        parallel_pairs=parallel_pairs,
    )


def _dedup_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_insert_task_conflict",
            """
            INSERT INTO task_conflicts (
                task_conflict_id, tenant_id, federation_id, left_task_cid,
                right_task_cid, conflict_kind, effect_class, evidence_ref,
                status, revision, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "task_conflict_id",
                "tenant_id",
                "federation_id",
                "left_task_cid",
                "right_task_cid",
                "conflict_kind",
                "effect_class",
                "evidence_ref",
                "status",
                "revision",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_task_conflict",
            """
            SELECT task_conflict_id, left_task_cid, right_task_cid, conflict_kind,
                   effect_class, evidence_ref, status, revision
            FROM task_conflicts
            WHERE task_conflict_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("task_conflict_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_task_resolution",
            """
            INSERT INTO task_resolutions (
                task_resolution_id, tenant_id, federation_id, task_cid,
                conflict_id, resolution_kind, predecessor_task_cid, result_ref,
                evidence_ref, revision, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "task_resolution_id",
                "tenant_id",
                "federation_id",
                "task_cid",
                "conflict_id",
                "resolution_kind",
                "predecessor_task_cid",
                "result_ref",
                "evidence_ref",
                "revision",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_task_resolution",
            """
            SELECT task_resolution_id, task_cid, conflict_id, resolution_kind,
                   predecessor_task_cid, result_ref, evidence_ref
            FROM task_resolutions
            WHERE task_resolution_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("task_resolution_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
    )


class DeduplicationStore(FederationSchedulerStore):
    """Persist intent conflict and resolution records through the sealed state owner."""

    INTERFACE = "DeduplicationStore@1"

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
            raise DeduplicationError("deduplication store never accepts a database path")
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise DeduplicationError(
                "deduplication store requires an already-attached typed state client"
            )
        registered = set(client.list_templates())
        missing = [
            template.name for template in _dedup_templates() if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise DeduplicationError(
                    "deduplication templates are absent from the sealed catalog"
                )
        else:
            for template in _dedup_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    def record_relation(
        self,
        relation: IntentRelation,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        effect_class: EventEffectClass,
    ) -> CausalGraphCommit:
        if not isinstance(relation, IntentRelation):
            raise FederationContractError("intent relation is required")
        if relation.nomination_only:
            raise DeduplicationAuthorityError(
                "nomination-only relations cannot be recorded as authoritative"
            )
        conflict_id = "task-conflict:" + relation.cid
        return self._commit_fact(
            operation="federation.dedup.relation.record",
            fact_id=conflict_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(
                dict.fromkeys(
                    (
                        conflict_id,
                        relation.left_intent_cid,
                        relation.right_intent_cid,
                    )
                )
            ),
            payload_ref=relation.cid,
            prepare_fact=lambda: self._prepare_relation(
                conflict_id,
                tenant_id=binding.tenant_id,
                federation_id=federation_id,
            ),
            apply_fact=lambda revision, recorded_at: self._insert_relation(
                relation,
                conflict_id=conflict_id,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                effect_class=effect_class,
                graph_revision=revision,
                recorded_at=recorded_at,
            ),
        )

    def load_conflict(
        self,
        *,
        conflict_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_task_conflict",
            {
                "task_conflict_id": _identifier(conflict_id, "conflict_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise DeduplicationError("task conflict is absent")
        return dict(rows[0])

    def load_resolution(
        self,
        *,
        resolution_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_task_resolution",
            {
                "task_resolution_id": _identifier(resolution_id, "resolution_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise DeduplicationError("task resolution is absent")
        return dict(rows[0])

    def _prepare_relation(self, conflict_id: str, *, tenant_id: str, federation_id: str) -> None:
        existing = self._client.execute(
            "casf_select_task_conflict",
            {
                "task_conflict_id": conflict_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if existing:
            raise DeduplicationError("task conflict identity is already bound")

    def _insert_relation(
        self,
        relation: IntentRelation,
        *,
        conflict_id: str,
        federation_id: str,
        tenant_id: str,
        effect_class: EventEffectClass,
        graph_revision: int,
        recorded_at: str,
    ) -> None:
        if not isinstance(effect_class, EventEffectClass):
            raise FederationContractError("effect_class is not closed")
        self._client.execute(
            "casf_insert_task_conflict",
            {
                "task_conflict_id": conflict_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "left_task_cid": relation.left_intent_cid,
                "right_task_cid": relation.right_intent_cid,
                "conflict_kind": relation.disposition.value,
                "effect_class": effect_class.value,
                "evidence_ref": relation.evidence_refs[0],
                "status": "current",
                "revision": graph_revision,
                "recorded_at": recorded_at,
            },
        )
        resolution_id = "task-resolution:" + relation.cid
        self._client.execute(
            "casf_insert_task_resolution",
            {
                "task_resolution_id": resolution_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "task_cid": relation.canonical_task_id,
                "conflict_id": conflict_id,
                "resolution_kind": relation.resolution_kind,
                "predecessor_task_cid": relation.covering_task_id,
                "result_ref": (
                    "result:" + relation.left_intent_cid
                    if relation.disposition is IntentDisposition.DUPLICATE
                    else relation.canonical_task_id
                ),
                "evidence_ref": relation.evidence_refs[0],
                "revision": graph_revision,
                "recorded_at": recorded_at,
            },
        )


__all__ = (
    "DeduplicationAuthorityError",
    "DeduplicationError",
    "DeduplicationReport",
    "DeduplicationStore",
    "IntentDisposition",
    "IntentIndependenceAdmission",
    "IntentRelation",
    "TaskIntentIdentity",
    "WRITE_EFFECT_CLASSES",
    "bind_task_intent",
    "classify_intent_pair",
    "classify_intents",
    "refuse_ducklake_dedup_authority",
)
