"""Causal frontier compilation for event-driven supervisor wakeup.

Exact descendants of changed facts ``must_wake``. Nomination-only and unknown
dependencies widen to ``may_wake`` rather than suppress work. ``do_not_wake``
requires proved or policy-admitted independence and can never be manufactured
from retrieval, stale maps, or the mere absence of a path.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

from ..task_sources.control_plane_contracts import content_identity
from ..task_sources.quack_state_client import QuackStateClient, StatementKind
from .causal_abstraction import CausalAbstractionStore
from .causal_graph import CausalGraphCommit, CausalGraphError
from .contracts import (
    CausalEdge,
    CausalFrontierEntry,
    CausalNode,
    FederationAuthorityError,
    FederationBinding,
    FederationContractError,
    FrontierDisposition,
    _identifier,
    _integer,
)
from .registry import _template

_PROJECTION_EDGE_KINDS = frozenset({"ABSTRACTS", "REFINES", "IMPLEMENTS"})
_DISPOSITION_RANK = {
    FrontierDisposition.DO_NOT_WAKE: 0,
    FrontierDisposition.MAY_WAKE: 1,
    FrontierDisposition.MUST_WAKE: 2,
}


class CausalFrontierError(CausalGraphError):
    """Base typed frontier compilation failure."""


class CausalFrontierAuthorityError(FederationAuthorityError, CausalFrontierError):
    """An attempt to suppress wakeup without admitted independence."""


@dataclass(frozen=True)
class FrontierSubject:
    """One supervisor/node pair that must be classified on the frontier."""

    supervisor_id: str
    node_id: str

    def __post_init__(self) -> None:
        _identifier(self.supervisor_id, "supervisor_id")
        _identifier(self.node_id, "node_id")


@dataclass(frozen=True)
class IndependenceAdmission:
    """Proved or policy-admitted independence used only for do_not_wake."""

    subject: FrontierSubject
    evidence_refs: tuple[str, ...]
    authoritative: bool
    policy_admitted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.subject, FrontierSubject):
            raise FederationContractError("independence subject is invalid")
        refs = tuple(_identifier(item, "evidence_refs") for item in self.evidence_refs)
        if not refs:
            raise FederationContractError("independence requires evidence")
        if len(set(refs)) != len(refs):
            raise FederationContractError("independence evidence contains duplicates")
        object.__setattr__(self, "evidence_refs", refs)
        if type(self.authoritative) is not bool or type(self.policy_admitted) is not bool:
            raise FederationContractError("independence flags must be boolean")
        if self.authoritative is False and self.policy_admitted is False:
            raise CausalFrontierAuthorityError(
                "retrieval nominations cannot prove independence"
            )


@dataclass(frozen=True)
class CompiledFrontier:
    """Complete classified wakeup frontier for one event."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/compiled-frontier@1"
    )

    event_id: str
    graph_revision: int
    entries: tuple[CausalFrontierEntry, ...]
    must_wake: tuple[str, ...]
    may_wake: tuple[str, ...]
    do_not_wake: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.event_id, "event_id")
        _integer(self.graph_revision, "graph_revision")
        if not isinstance(self.entries, tuple) or not all(
            isinstance(item, CausalFrontierEntry) for item in self.entries
        ):
            raise FederationContractError("entries must be CausalFrontierEntry records")
        classified = (
            set(self.must_wake) | set(self.may_wake) | set(self.do_not_wake)
        )
        subjects = {item.supervisor_id for item in self.entries}
        if classified != subjects:
            raise FederationContractError("frontier classification is incomplete")
        overlap = set(self.must_wake) & set(self.may_wake)
        overlap |= set(self.must_wake) & set(self.do_not_wake)
        overlap |= set(self.may_wake) & set(self.do_not_wake)
        if overlap:
            raise FederationContractError("frontier dispositions overlap")

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "event_id": self.event_id,
                "graph_revision": self.graph_revision,
                "entries": [item.to_dict() for item in self.entries],
            }
        )


def _reachable(seeds: Sequence[str], adjacency: Mapping[str, Sequence[str]]) -> set[str]:
    seen: set[str] = set()
    queue = deque(seeds)
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        for nxt in adjacency.get(node, ()):
            if nxt not in seen:
                queue.append(nxt)
    return seen


def compile_frontier(
    *,
    event_id: str,
    binding: FederationBinding,
    graph_revision: int,
    nodes: Sequence[CausalNode],
    edges: Sequence[CausalEdge],
    changed_fact_refs: Sequence[str],
    subjects: Sequence[FrontierSubject],
    independence: Sequence[IndependenceAdmission] = (),
    admitted_projection_edge_ids: Sequence[str] = (),
) -> CompiledFrontier:
    """Classify every subject as must_wake, may_wake, or do_not_wake."""

    event_id = _identifier(event_id, "event_id")
    if not subjects:
        raise FederationContractError("frontier requires at least one subject")
    unique_subjects: list[FrontierSubject] = []
    seen_subjects: set[tuple[str, str]] = set()
    for subject in subjects:
        if not isinstance(subject, FrontierSubject):
            raise FederationContractError("subjects must be FrontierSubject records")
        key = (subject.supervisor_id, subject.node_id)
        if key in seen_subjects:
            continue
        seen_subjects.add(key)
        unique_subjects.append(subject)
    subjects = unique_subjects
    node_ids = {item.record_id for item in nodes}
    subject_refs = {item.subject_ref: item.record_id for item in nodes}
    facts = tuple(_identifier(item, "changed_fact_refs") for item in changed_fact_refs)
    seeds: list[str] = []
    for fact in facts:
        if fact in node_ids:
            seeds.append(fact)
        elif fact in subject_refs:
            seeds.append(subject_refs[fact])
    admitted_projections = {
        _identifier(item, "admitted_projection_edge_ids")
        for item in admitted_projection_edge_ids
    }
    exact_adj: dict[str, list[str]] = defaultdict(list)
    nomination_adj: dict[str, list[str]] = defaultdict(list)
    exact_evidence: dict[str, list[str]] = defaultdict(list)
    nomination_evidence: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if not isinstance(edge, CausalEdge):
            raise FederationContractError("edges must be CausalEdge records")
        projection = edge.edge_kind.value in _PROJECTION_EDGE_KINDS
        exact = not edge.nomination_only and (
            not projection or edge.record_id in admitted_projections
        )
        if exact:
            exact_adj[edge.source_node_id].append(edge.target_node_id)
            exact_evidence[edge.target_node_id].extend(edge.evidence_refs)
            exact_evidence[edge.source_node_id].extend(edge.evidence_refs)
        else:
            nomination_adj[edge.source_node_id].append(edge.target_node_id)
            nomination_evidence[edge.target_node_id].extend(edge.evidence_refs)
    must_nodes = _reachable(seeds, exact_adj)
    may_nodes = _reachable(tuple(must_nodes) + tuple(seeds), nomination_adj) - must_nodes
    independence_by_subject = {
        (item.subject.supervisor_id, item.subject.node_id): item for item in independence
    }
    entries: list[CausalFrontierEntry] = []
    strongest: dict[str, FrontierDisposition] = {}
    evidence_by_supervisor: dict[str, list[str]] = defaultdict(list)
    for subject in subjects:
        if not isinstance(subject, FrontierSubject):
            raise FederationContractError("subjects must be FrontierSubject records")
        if subject.node_id not in node_ids:
            raise CausalFrontierError("frontier subject node is absent from the graph")
        claimed = independence_by_subject.get((subject.supervisor_id, subject.node_id))
        if subject.node_id in must_nodes:
            if claimed is not None:
                raise CausalFrontierAuthorityError(
                    "independence cannot suppress an exact causal descendant"
                )
            disposition = FrontierDisposition.MUST_WAKE
            evidence = tuple(dict.fromkeys(exact_evidence.get(subject.node_id, (event_id,))))
            if not evidence:
                evidence = (event_id,)
        elif subject.node_id in may_nodes:
            if claimed is not None:
                raise CausalFrontierAuthorityError(
                    "nomination-only reachability cannot be suppressed as independence"
                )
            disposition = FrontierDisposition.MAY_WAKE
            evidence = tuple(
                dict.fromkeys(nomination_evidence.get(subject.node_id, (event_id,)))
            ) or (event_id,)
        elif claimed is not None:
            disposition = FrontierDisposition.DO_NOT_WAKE
            evidence = claimed.evidence_refs
        else:
            disposition = FrontierDisposition.MAY_WAKE
            evidence = ("frontier:unknown-widening",)
        current = strongest.get(subject.supervisor_id)
        if current is None or _DISPOSITION_RANK[disposition] > _DISPOSITION_RANK[current]:
            strongest[subject.supervisor_id] = disposition
        evidence_by_supervisor[subject.supervisor_id].extend(evidence)
        entries.append(
            CausalFrontierEntry(
                record_id="frontier-entry:"
                + content_identity(
                    {
                        "event_id": event_id,
                        "supervisor_id": subject.supervisor_id,
                        "node_id": subject.node_id,
                    }
                ),
                revision=max(graph_revision, 1),
                binding=replace(binding, causal_graph_revision=graph_revision),
                event_id=event_id,
                supervisor_id=subject.supervisor_id,
                node_id=subject.node_id,
                disposition=disposition,
                evidence_refs=evidence,
            )
        )
    must_wake = tuple(
        sorted(
            supervisor
            for supervisor, disposition in strongest.items()
            if disposition is FrontierDisposition.MUST_WAKE
        )
    )
    may_wake = tuple(
        sorted(
            supervisor
            for supervisor, disposition in strongest.items()
            if disposition is FrontierDisposition.MAY_WAKE
        )
    )
    do_not_wake = tuple(
        sorted(
            supervisor
            for supervisor, disposition in strongest.items()
            if disposition is FrontierDisposition.DO_NOT_WAKE
        )
    )
    ordered = tuple(
        sorted(entries, key=lambda item: (item.supervisor_id, item.node_id))
    )
    return CompiledFrontier(
        event_id=event_id,
        graph_revision=graph_revision,
        entries=ordered,
        must_wake=must_wake,
        may_wake=may_wake,
        do_not_wake=do_not_wake,
    )


def _frontier_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_insert_causal_frontier",
            """
            INSERT INTO causal_frontiers (
                causal_frontier_id, tenant_id, federation_id, event_id,
                graph_revision, abstraction_revision_ref, must_wake_count,
                may_wake_count, do_not_wake_count, content_ref, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "causal_frontier_id",
                "tenant_id",
                "federation_id",
                "event_id",
                "graph_revision",
                "abstraction_revision_ref",
                "must_wake_count",
                "may_wake_count",
                "do_not_wake_count",
                "content_ref",
                "created_at",
            ),
        ),
        _template(
            "casf_insert_causal_frontier_member",
            """
            INSERT INTO causal_frontier_members (
                causal_frontier_id, subject_kind, subject_ref, disposition,
                evidence_ref, ordinal
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "causal_frontier_id",
                "subject_kind",
                "subject_ref",
                "disposition",
                "evidence_ref",
                "ordinal",
            ),
        ),
        _template(
            "casf_select_causal_frontier",
            """
            SELECT causal_frontier_id, event_id, graph_revision,
                   must_wake_count, may_wake_count, do_not_wake_count,
                   content_ref
            FROM causal_frontiers
            WHERE causal_frontier_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("causal_frontier_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_causal_frontier_members",
            """
            SELECT subject_kind, subject_ref, disposition, evidence_ref, ordinal
            FROM causal_frontier_members
            WHERE causal_frontier_id = ?
            ORDER BY ordinal, subject_ref
            """,
            ("causal_frontier_id",),
            kind=StatementKind.QUERY,
        ),
    )


class CausalFrontierStore(CausalAbstractionStore):
    """Persist compiled wakeup frontiers through the sealed state owner."""

    INTERFACE = "CausalFrontierStore@1"

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
            raise CausalFrontierError("frontier store never accepts a database path")
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise CausalFrontierError(
                "frontier store requires an already-attached typed state client"
            )
        registered = set(client.list_templates())
        missing = [
            template.name
            for template in _frontier_templates()
            if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise CausalFrontierError(
                    "frontier templates are absent from the sealed catalog"
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
        compiled: CompiledFrontier,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        abstraction_revision_ref: str = "abstraction:none",
    ) -> CausalGraphCommit:
        if not isinstance(compiled, CompiledFrontier):
            raise FederationContractError("compiled frontier is required")
        frontier_id = "frontier:" + compiled.cid
        return self._commit_fact(
            operation="federation.causal.frontier.record",
            fact_id=frontier_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=(compiled.event_id, frontier_id),
            payload_ref=compiled.cid,
            prepare_fact=lambda: self._prepare_frontier(
                frontier_id,
                tenant_id=binding.tenant_id,
                federation_id=federation_id,
            ),
            apply_fact=lambda revision, recorded_at: self._insert_frontier(
                compiled,
                frontier_id=frontier_id,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                graph_revision=revision,
                recorded_at=recorded_at,
                abstraction_revision_ref=abstraction_revision_ref,
            ),
        )

    def load_frontier(
        self,
        *,
        frontier_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_causal_frontier",
            {
                "causal_frontier_id": _identifier(frontier_id, "frontier_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise CausalFrontierError("compiled frontier is absent")
        members = self._client.execute(
            "casf_select_causal_frontier_members",
            {"causal_frontier_id": frontier_id},
        )
        return {"frontier": rows[0], "members": members}

    def _prepare_frontier(
        self, frontier_id: str, *, tenant_id: str, federation_id: str
    ) -> None:
        existing = self._client.execute(
            "casf_select_causal_frontier",
            {
                "causal_frontier_id": frontier_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if existing:
            raise CausalFrontierError("compiled frontier identity is already bound")

    def _insert_frontier(
        self,
        compiled: CompiledFrontier,
        *,
        frontier_id: str,
        federation_id: str,
        tenant_id: str,
        graph_revision: int,
        recorded_at: str,
        abstraction_revision_ref: str,
    ) -> None:
        self._client.execute(
            "casf_insert_causal_frontier",
            {
                "causal_frontier_id": frontier_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "event_id": compiled.event_id,
                "graph_revision": graph_revision,
                "abstraction_revision_ref": _identifier(
                    abstraction_revision_ref, "abstraction_revision_ref"
                ),
                "must_wake_count": len(compiled.must_wake),
                "may_wake_count": len(compiled.may_wake),
                "do_not_wake_count": len(compiled.do_not_wake),
                "content_ref": compiled.cid,
                "created_at": recorded_at,
            },
        )
        evidence_by_supervisor = {
            entry.supervisor_id: entry.evidence_refs[0] for entry in compiled.entries
        }
        ordinal = 0
        for disposition, supervisors in (
            (FrontierDisposition.MUST_WAKE, compiled.must_wake),
            (FrontierDisposition.MAY_WAKE, compiled.may_wake),
            (FrontierDisposition.DO_NOT_WAKE, compiled.do_not_wake),
        ):
            for supervisor in supervisors:
                ordinal += 1
                self._client.execute(
                    "casf_insert_causal_frontier_member",
                    {
                        "causal_frontier_id": frontier_id,
                        "subject_kind": "supervisor",
                        "subject_ref": supervisor,
                        "disposition": disposition.value,
                        "evidence_ref": evidence_by_supervisor[supervisor],
                        "ordinal": ordinal,
                    },
                )


__all__ = (
    "CausalFrontierAuthorityError",
    "CausalFrontierError",
    "CausalFrontierStore",
    "CompiledFrontier",
    "FrontierSubject",
    "IndependenceAdmission",
    "compile_frontier",
)
