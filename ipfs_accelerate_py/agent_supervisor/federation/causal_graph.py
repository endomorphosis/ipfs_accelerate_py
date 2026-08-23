"""Multilevel causal graph store over the sealed Quack statement catalog.

This module never accepts a database path and never exposes arbitrary SQL.
Trusted static statement templates are registered once, then the client catalog
is sealed by :class:`FederationStateRepository`.  Mutations execute through
``StateTransaction.execute_command`` and append the domain event plus outbox
row before the generation/idempotency records commit.

Semantic meaning remains in ``ipfs_datasets_py``.  The store persists
operational references, exact-versus-nomination evidence admission, graph
revision CAS, and explicit fixed-point groups for cycles.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

from ..task_sources.control_plane_contracts import (
    CommandKind,
    content_identity,
)
from ..task_sources.quack_state_client import (
    QuackStateClient,
    StatementKind,
)
from .contracts import (
    CausalEdge,
    CausalEdgeKind,
    CausalEvidence,
    CausalEvidenceKind,
    CausalLevel,
    CausalNode,
    FederationAuthorityError,
    FederationBinding,
    FederationContractError,
    _identifier,
    _integer,
    utc_now,
)
from .events import EventClass, EventEffectClass
from .outbox import EventDraft
from .registry import (
    FederationRepositoryConflict,
    FederationRepositoryError,
    FederationRepositoryNotFound,
    FederationStateRepository,
    _template,
)

_LEVEL_RANK: Mapping[CausalLevel, int] = MappingProxyType(
    {
        CausalLevel.L0_RUNTIME: 0,
        CausalLevel.L1_CODE_ARTIFACT: 1,
        CausalLevel.L2_WORK: 2,
        CausalLevel.L3_INTENT: 3,
        CausalLevel.L4_FEDERATION: 4,
    }
)
_CROSS_LEVEL_EDGE_KINDS = frozenset(
    {
        CausalEdgeKind.ABSTRACTS,
        CausalEdgeKind.REFINES,
        CausalEdgeKind.IMPLEMENTS,
        CausalEdgeKind.DELEGATES_TO,
    }
)
_AUTHORITATIVE = "authoritative"
_NOMINATION_ONLY = "nomination_only"
_CURRENT = "current"


class CausalGraphError(FederationRepositoryError):
    """Base typed causal-graph failure."""


class CausalGraphConflict(FederationRepositoryConflict, CausalGraphError):
    """A graph revision, identity, or population invariant conflicted."""


class CausalGraphNotFound(FederationRepositoryNotFound, CausalGraphError):
    """A required causal graph record is absent."""


class CausalCycleError(FederationAuthorityError, CausalGraphError):
    """A cycle was nominated without an explicit fixed-point group."""


@dataclass(frozen=True)
class CausalGraphCommit:
    """Receipt for one transactional causal-graph mutation."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/causal-graph-commit@1"
    )

    graph_revision: int
    fact_id: str
    event_id: str
    outbox_id: str
    event_global_sequence: int

    def __post_init__(self) -> None:
        _integer(self.graph_revision, "graph_revision", minimum=1)
        _identifier(self.fact_id, "fact_id")
        _identifier(self.event_id, "event_id")
        _identifier(self.outbox_id, "outbox_id")
        _integer(self.event_global_sequence, "event_global_sequence", minimum=1)


@dataclass(frozen=True)
class CausalGraphSnapshot:
    """Exact current-revision population for one federation graph."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/causal-graph-snapshot@1"
    )

    tenant_id: str
    federation_id: str
    graph_revision: int
    nodes: tuple[CausalNode, ...]
    edges: tuple[CausalEdge, ...]
    evidence: tuple[CausalEvidence, ...]
    cycle_group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.tenant_id, "tenant_id")
        _identifier(self.federation_id, "federation_id")
        _integer(self.graph_revision, "graph_revision")
        if not isinstance(self.nodes, tuple) or not all(
            isinstance(item, CausalNode) for item in self.nodes
        ):
            raise FederationContractError("nodes must be CausalNode records")
        if not isinstance(self.edges, tuple) or not all(
            isinstance(item, CausalEdge) for item in self.edges
        ):
            raise FederationContractError("edges must be CausalEdge records")
        if not isinstance(self.evidence, tuple) or not all(
            isinstance(item, CausalEvidence) for item in self.evidence
        ):
            raise FederationContractError("evidence must be CausalEvidence records")
        seen_groups = []
        for item in self.cycle_group_ids:
            seen_groups.append(_identifier(item, "cycle_group_ids"))
        if len(set(seen_groups)) != len(seen_groups):
            raise FederationContractError("cycle_group_ids contains duplicate identities")


def _causal_templates() -> tuple[Any, ...]:
    """Return the closed causal-graph statement catalog (no caller SQL)."""

    return (
        _template(
            "casf_select_federation_graph_revision",
            """
            SELECT federation_id, tenant_id, causal_graph_revision, status
            FROM federations
            WHERE federation_id = ? AND tenant_id = ?
            LIMIT 1
            """,
            ("federation_id", "tenant_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_advance_federation_graph_revision",
            """
            UPDATE federations
            SET causal_graph_revision = causal_graph_revision + 1,
                updated_at = ?
            WHERE federation_id = ? AND tenant_id = ?
              AND causal_graph_revision = ?
            RETURNING causal_graph_revision
            """,
            (
                "updated_at",
                "federation_id",
                "tenant_id",
                "expected_revision",
            ),
        ),
        _template(
            "casf_select_causal_node",
            """
            SELECT causal_node_id, tenant_id, federation_id, causal_level,
                   node_kind, subject_ref, repository_id, tree_id, owner_id,
                   source_root, content_ref, graph_revision, freshness_state,
                   created_at
            FROM causal_nodes
            WHERE causal_node_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("causal_node_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_causal_node_by_subject",
            """
            SELECT causal_node_id, tenant_id, federation_id, causal_level,
                   node_kind, subject_ref, content_ref, graph_revision,
                   freshness_state
            FROM causal_nodes
            WHERE tenant_id = ? AND federation_id = ? AND causal_level = ?
              AND subject_ref = ? AND freshness_state = 'current'
            ORDER BY graph_revision DESC, causal_node_id
            """,
            ("tenant_id", "federation_id", "causal_level", "subject_ref"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_causal_node",
            """
            INSERT INTO causal_nodes (
                causal_node_id, tenant_id, federation_id, causal_level,
                node_kind, subject_ref, repository_id, tree_id, owner_id,
                source_root, content_ref, graph_revision, freshness_state,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'current', ?)
            """,
            (
                "causal_node_id",
                "tenant_id",
                "federation_id",
                "causal_level",
                "node_kind",
                "subject_ref",
                "repository_id",
                "tree_id",
                "owner_id",
                "source_root",
                "content_ref",
                "graph_revision",
                "created_at",
            ),
        ),
        _template(
            "casf_select_causal_evidence",
            """
            SELECT causal_evidence_id, tenant_id, federation_id, causal_edge_id,
                   evidence_kind, authority_disposition, repository_id, tree_id,
                   owner_id, source_root, content_ref, observed_at, expires_at
            FROM causal_evidence
            WHERE causal_evidence_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("causal_evidence_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_causal_evidence",
            """
            INSERT INTO causal_evidence (
                causal_evidence_id, tenant_id, federation_id, causal_edge_id,
                evidence_kind, authority_disposition, repository_id, tree_id,
                owner_id, source_root, content_ref, observed_at, expires_at
            ) VALUES (?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "causal_evidence_id",
                "tenant_id",
                "federation_id",
                "evidence_kind",
                "authority_disposition",
                "repository_id",
                "tree_id",
                "owner_id",
                "source_root",
                "content_ref",
                "observed_at",
                "expires_at",
            ),
        ),
        _template(
            "casf_attach_causal_evidence_edge",
            """
            UPDATE causal_evidence
            SET causal_edge_id = ?
            WHERE causal_evidence_id = ? AND tenant_id = ? AND federation_id = ?
              AND causal_edge_id = ''
            RETURNING causal_evidence_id
            """,
            (
                "causal_edge_id",
                "causal_evidence_id",
                "tenant_id",
                "federation_id",
            ),
        ),
        _template(
            "casf_select_live_causal_edges",
            """
            SELECT causal_edge_id, tenant_id, federation_id, source_node_id,
                   target_node_id, edge_kind, graph_revision,
                   authority_disposition, evidence_population_ref,
                   admitted_policy_ref, created_at, retired_at
            FROM causal_edges
            WHERE tenant_id = ? AND federation_id = ? AND retired_at IS NULL
            ORDER BY graph_revision, causal_edge_id
            """,
            ("tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_causal_edge",
            """
            SELECT causal_edge_id, tenant_id, federation_id, source_node_id,
                   target_node_id, edge_kind, graph_revision,
                   authority_disposition, evidence_population_ref,
                   admitted_policy_ref, created_at, retired_at
            FROM causal_edges
            WHERE causal_edge_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("causal_edge_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_causal_edge",
            """
            INSERT INTO causal_edges (
                causal_edge_id, tenant_id, federation_id, source_node_id,
                target_node_id, edge_kind, graph_revision,
                authority_disposition, evidence_population_ref,
                admitted_policy_ref, created_at, retired_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                "causal_edge_id",
                "tenant_id",
                "federation_id",
                "source_node_id",
                "target_node_id",
                "edge_kind",
                "graph_revision",
                "authority_disposition",
                "evidence_population_ref",
                "admitted_policy_ref",
                "created_at",
            ),
        ),
        _template(
            "casf_select_causal_nodes",
            """
            SELECT causal_node_id, tenant_id, federation_id, causal_level,
                   node_kind, subject_ref, repository_id, tree_id, owner_id,
                   source_root, content_ref, graph_revision, freshness_state,
                   created_at
            FROM causal_nodes
            WHERE tenant_id = ? AND federation_id = ?
            ORDER BY graph_revision, causal_node_id
            """,
            ("tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_causal_evidence_population",
            """
            SELECT causal_evidence_id, tenant_id, federation_id, causal_edge_id,
                   evidence_kind, authority_disposition, repository_id, tree_id,
                   owner_id, source_root, content_ref, observed_at, expires_at
            FROM causal_evidence
            WHERE tenant_id = ? AND federation_id = ?
            ORDER BY observed_at, causal_evidence_id
            """,
            ("tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_causal_cycle_slice",
            """
            INSERT INTO causal_slices (
                causal_slice_id, tenant_id, federation_id, graph_revision,
                root_event_id, root_fact_ref, node_population_ref,
                edge_population_ref, content_ref, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "causal_slice_id",
                "tenant_id",
                "federation_id",
                "graph_revision",
                "root_event_id",
                "root_fact_ref",
                "node_population_ref",
                "edge_population_ref",
                "content_ref",
                "created_at",
            ),
        ),
        _template(
            "casf_select_causal_cycle_slices",
            """
            SELECT causal_slice_id
            FROM causal_slices
            WHERE tenant_id = ? AND federation_id = ?
            ORDER BY graph_revision, causal_slice_id
            """,
            ("tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
    )


def _authority_disposition(*, authoritative: bool, nomination_only: bool) -> str:
    if nomination_only and authoritative:
        raise FederationAuthorityError(
            "nomination-only causal facts cannot be authoritative"
        )
    if nomination_only or not authoritative:
        return _NOMINATION_ONLY
    return _AUTHORITATIVE


def _level_pair_allowed(kind: CausalEdgeKind, source: CausalLevel, target: CausalLevel) -> bool:
    source_rank = _LEVEL_RANK[source]
    target_rank = _LEVEL_RANK[target]
    if source is target:
        return True
    if kind not in _CROSS_LEVEL_EDGE_KINDS:
        return False
    return abs(source_rank - target_rank) == 1


def _directed_cycle(edges: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    adjacency: dict[str, list[str]] = {}
    for source, target in edges:
        adjacency.setdefault(source, []).append(target)
        adjacency.setdefault(target, [])
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def dfs(node: str) -> tuple[str, ...] | None:
        if node in visiting:
            start = stack.index(node)
            return tuple(stack[start:] + [node])
        if node in visited:
            return None
        visiting.add(node)
        stack.append(node)
        for nxt in adjacency.get(node, ()):
            found = dfs(nxt)
            if found is not None:
                return found
        stack.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for node in adjacency:
        found = dfs(node)
        if found is not None:
            return found
    return ()


def _evidence_refs_identity(refs: Sequence[str]) -> str:
    return "evidence-population:" + content_identity(list(refs))


class CausalGraphStore(FederationStateRepository):
    """Sealed multilevel causal graph over one already-attached state client."""

    INTERFACE = "CausalGraphStore@1"

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
            raise CausalGraphError("causal graph store never accepts a database path")
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise CausalGraphError(
                "causal graph store requires an already-attached typed state client"
            )
        registered = set(client.list_templates())
        missing = [
            template.name
            for template in _causal_templates()
            if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise CausalGraphError(
                    "causal graph templates are absent from the sealed catalog"
                )
        else:
            for template in _causal_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    @property
    def _client(self) -> QuackStateClient:
        return self._FederationStateRepository__client

    def graph_revision(self, *, tenant_id: str, federation_id: str) -> int:
        row = self._federation_row(tenant_id=tenant_id, federation_id=federation_id)
        return int(row["causal_graph_revision"])

    def snapshot(self, *, tenant_id: str, federation_id: str) -> CausalGraphSnapshot:
        row = self._federation_row(tenant_id=tenant_id, federation_id=federation_id)
        binding = self._binding_for(
            tenant_id=tenant_id,
            federation_id=federation_id,
            graph_revision=int(row["causal_graph_revision"]),
        )
        node_rows = self._client.execute(
            "casf_select_causal_nodes",
            {"tenant_id": tenant_id, "federation_id": federation_id},
        )
        edge_rows = self._client.execute(
            "casf_select_live_causal_edges",
            {"tenant_id": tenant_id, "federation_id": federation_id},
        )
        evidence_rows = self._client.execute(
            "casf_select_causal_evidence_population",
            {"tenant_id": tenant_id, "federation_id": federation_id},
        )
        evidence_by_edge: dict[str, list[str]] = {}
        for item in evidence_rows:
            edge_id = str(item["causal_edge_id"] or "")
            if edge_id:
                evidence_by_edge.setdefault(edge_id, []).append(
                    str(item["causal_evidence_id"])
                )
        nodes = tuple(self._node_from_row(item, binding) for item in node_rows)
        edges = tuple(
            self._edge_from_row(
                item,
                binding,
                evidence_refs=tuple(evidence_by_edge.get(str(item["causal_edge_id"]), ())),
            )
            for item in edge_rows
        )
        evidence = tuple(self._evidence_from_row(item, binding) for item in evidence_rows)
        groups = tuple(
            str(item["causal_slice_id"])
            for item in self._client.execute(
                "casf_select_causal_cycle_slices",
                {"tenant_id": tenant_id, "federation_id": federation_id},
            )
        )
        return CausalGraphSnapshot(
            tenant_id=tenant_id,
            federation_id=federation_id,
            graph_revision=int(row["causal_graph_revision"]),
            nodes=nodes,
            edges=edges,
            evidence=evidence,
            cycle_group_ids=groups,
        )

    def record_node(
        self,
        node: CausalNode,
        *,
        federation_id: str,
        expected_graph_revision: int,
        owner_id: str,
        source_root: str,
        idempotency_key: str,
    ) -> CausalGraphCommit:
        if not isinstance(node, CausalNode):
            raise FederationContractError("node must be a CausalNode")
        self._assert_binding_scope(node.binding, federation_id=federation_id)
        return self._commit_fact(
            operation="federation.causal.node.record",
            fact_id=node.record_id,
            federation_id=federation_id,
            binding=node.binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=(node.record_id, node.subject_ref),
            payload_ref=node.cid,
            prepare_fact=lambda: self._prepare_node(node, federation_id=federation_id),
            apply_fact=lambda revision, recorded_at: self._insert_node(
                node,
                federation_id=federation_id,
                graph_revision=revision,
                owner_id=owner_id,
                source_root=source_root,
                recorded_at=recorded_at,
            ),
        )

    def record_evidence(
        self,
        evidence: CausalEvidence,
        *,
        federation_id: str,
        expected_graph_revision: int,
        owner_id: str,
        source_root: str,
        idempotency_key: str,
    ) -> CausalGraphCommit:
        if not isinstance(evidence, CausalEvidence):
            raise FederationContractError("evidence must be a CausalEvidence")
        self._assert_binding_scope(evidence.binding, federation_id=federation_id)
        if (
            evidence.evidence_kind is CausalEvidenceKind.RETRIEVAL_NOMINATION
            and evidence.authoritative
        ):
            raise FederationAuthorityError(
                "retrieval nomination cannot be authoritative causal evidence"
            )
        return self._commit_fact(
            operation="federation.causal.evidence.record",
            fact_id=evidence.record_id,
            federation_id=federation_id,
            binding=evidence.binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=(evidence.record_id, evidence.evidence_ref),
            payload_ref=evidence.cid,
            prepare_fact=lambda: self._prepare_evidence(
                evidence, federation_id=federation_id
            ),
            apply_fact=lambda revision, recorded_at: self._insert_evidence(
                evidence,
                federation_id=federation_id,
                owner_id=owner_id,
                source_root=source_root,
                recorded_at=recorded_at,
            ),
        )

    def record_edge(
        self,
        edge: CausalEdge,
        *,
        federation_id: str,
        expected_graph_revision: int,
        idempotency_key: str,
        fixed_point_group_id: str = "",
        admitted_policy_ref: str = "",
    ) -> CausalGraphCommit:
        if not isinstance(edge, CausalEdge):
            raise FederationContractError("edge must be a CausalEdge")
        self._assert_binding_scope(edge.binding, federation_id=federation_id)
        if edge.source_node_id == edge.target_node_id and not fixed_point_group_id:
            raise CausalCycleError(
                "causal self-cycles require an explicit fixed-point group"
            )
        group_id = _identifier(fixed_point_group_id, "fixed_point_group_id", required=False)
        policy_ref = _identifier(admitted_policy_ref, "admitted_policy_ref", required=False)
        return self._commit_fact(
            operation="federation.causal.edge.record",
            fact_id=edge.record_id,
            federation_id=federation_id,
            binding=edge.binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=(
                edge.record_id,
                edge.source_node_id,
                edge.target_node_id,
                *edge.evidence_refs,
            ),
            payload_ref=edge.cid,
            prepare_fact=lambda: self._prepare_edge(
                edge,
                federation_id=federation_id,
                fixed_point_group_id=group_id,
            ),
            apply_fact=lambda revision, recorded_at: self._insert_edge(
                edge,
                federation_id=federation_id,
                graph_revision=revision,
                recorded_at=recorded_at,
                fixed_point_group_id=group_id,
                admitted_policy_ref=policy_ref,
            ),
        )

    def _commit_fact(
        self,
        *,
        operation: str,
        fact_id: str,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        changed_fact_refs: Sequence[str],
        payload_ref: str,
        prepare_fact: Callable[[], None],
        apply_fact: Callable[[int, str], None],
    ) -> CausalGraphCommit:
        _integer(expected_graph_revision, "expected_graph_revision", minimum=0)
        recorded_at = utc_now()
        command = self._command(
            command_id=f"command:{operation}:{payload_ref}",
            idempotency_key=idempotency_key,
            command_kind=CommandKind.APPEND,
            parameters={
                "operation": operation,
                "federation_id": federation_id,
                "tenant_id": binding.tenant_id,
                "fact_id": fact_id,
            },
        )

        def apply(_txn: Any, _command: Any, live: Any) -> Mapping[str, Any]:
            if binding.control_plane_generation != live.generation:
                raise CausalGraphConflict("request control-plane generation is stale")
            row = self._federation_row(
                tenant_id=binding.tenant_id,
                federation_id=federation_id,
            )
            current_revision = int(row["causal_graph_revision"])
            if current_revision != expected_graph_revision:
                raise CausalGraphConflict(
                    "causal graph epoch does not match the expected value"
                )
            prepare_fact()
            advanced = self._client.execute(
                "casf_advance_federation_graph_revision",
                {
                    "updated_at": recorded_at,
                    "federation_id": federation_id,
                    "tenant_id": binding.tenant_id,
                    "expected_revision": expected_graph_revision,
                },
            )
            if len(advanced) != 1:
                raise CausalGraphConflict("causal graph revision CAS lost")
            next_revision = int(advanced[0]["causal_graph_revision"])
            apply_fact(next_revision, recorded_at)
            mutation_binding = replace(binding, causal_graph_revision=next_revision)
            draft = EventDraft(
                event_type=EventClass.CAUSAL_GRAPH_CHANGED,
                stream_id=federation_id,
                causal_parent_ids=(),
                correlation_id=f"correlation:{idempotency_key}",
                causation_id=f"causation:{payload_ref}",
                tenant_id=binding.tenant_id,
                federation_id=federation_id,
                repository_id=binding.repository_ids[0],
                tree_id=binding.repository_tree_ids[0],
                payload_ref=payload_ref,
                changed_fact_refs=tuple(changed_fact_refs),
                effect_class=EventEffectClass.AUTHORITATIVE_STATE,
                deduplication_key=f"{operation}:{idempotency_key}",
            )
            event, outbox = self._allocate_event(draft, recorded_at=recorded_at)
            self._insert_event_outbox(event, outbox, binding=mutation_binding)
            return {
                "graph_revision": next_revision,
                "fact_id": fact_id,
                "event_id": event.event_id,
                "outbox_id": outbox.outbox_id,
                "event_global_sequence": event.global_sequence,
            }

        result = self._submit(command, apply)
        payload = result.result
        return CausalGraphCommit(
            graph_revision=int(payload["graph_revision"]),
            fact_id=str(payload["fact_id"]),
            event_id=str(payload["event_id"]),
            outbox_id=str(payload["outbox_id"]),
            event_global_sequence=int(payload["event_global_sequence"]),
        )

    def _prepare_node(self, node: CausalNode, *, federation_id: str) -> None:
        existing = self._client.execute(
            "casf_select_causal_node",
            {
                "causal_node_id": node.record_id,
                "tenant_id": node.binding.tenant_id,
                "federation_id": federation_id,
            },
        )
        if existing:
            raise CausalGraphConflict("causal node identity is already bound")
        subject_rows = self._client.execute(
            "casf_select_causal_node_by_subject",
            {
                "tenant_id": node.binding.tenant_id,
                "federation_id": federation_id,
                "causal_level": node.level.value,
                "subject_ref": node.subject_ref,
            },
        )
        if subject_rows:
            raise CausalGraphConflict(
                "a current causal node already occupies this level and subject"
            )

    def _insert_node(
        self,
        node: CausalNode,
        *,
        federation_id: str,
        graph_revision: int,
        owner_id: str,
        source_root: str,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_causal_node",
            {
                "causal_node_id": node.record_id,
                "tenant_id": node.binding.tenant_id,
                "federation_id": federation_id,
                "causal_level": node.level.value,
                "node_kind": node.node_type,
                "subject_ref": node.subject_ref,
                "repository_id": node.binding.repository_ids[0],
                "tree_id": node.binding.repository_tree_ids[0],
                "owner_id": _identifier(owner_id, "owner_id"),
                "source_root": _identifier(source_root, "source_root"),
                "content_ref": node.cid,
                "graph_revision": graph_revision,
                "created_at": recorded_at,
            },
        )

    def _prepare_evidence(self, evidence: CausalEvidence, *, federation_id: str) -> None:
        existing = self._client.execute(
            "casf_select_causal_evidence",
            {
                "causal_evidence_id": evidence.record_id,
                "tenant_id": evidence.binding.tenant_id,
                "federation_id": federation_id,
            },
        )
        if existing:
            raise CausalGraphConflict("causal evidence identity is already bound")
        _authority_disposition(
            authoritative=evidence.authoritative,
            nomination_only=(
                evidence.evidence_kind is CausalEvidenceKind.RETRIEVAL_NOMINATION
            ),
        )

    def _insert_evidence(
        self,
        evidence: CausalEvidence,
        *,
        federation_id: str,
        owner_id: str,
        source_root: str,
        recorded_at: str,
    ) -> None:
        disposition = _authority_disposition(
            authoritative=evidence.authoritative,
            nomination_only=(
                evidence.evidence_kind is CausalEvidenceKind.RETRIEVAL_NOMINATION
            ),
        )
        self._client.execute(
            "casf_insert_causal_evidence",
            {
                "causal_evidence_id": evidence.record_id,
                "tenant_id": evidence.binding.tenant_id,
                "federation_id": federation_id,
                "evidence_kind": evidence.evidence_kind.value,
                "authority_disposition": disposition,
                "repository_id": evidence.binding.repository_ids[0],
                "tree_id": evidence.binding.repository_tree_ids[0],
                "owner_id": _identifier(owner_id, "owner_id"),
                "source_root": _identifier(source_root, "source_root"),
                "content_ref": evidence.evidence_ref,
                "observed_at": recorded_at,
                "expires_at": evidence.binding.expires_at,
            },
        )

    def _prepare_edge(
        self,
        edge: CausalEdge,
        *,
        federation_id: str,
        fixed_point_group_id: str,
    ) -> tuple[list[tuple[str, str]], tuple[str, ...]]:
        existing = self._client.execute(
            "casf_select_causal_edge",
            {
                "causal_edge_id": edge.record_id,
                "tenant_id": edge.binding.tenant_id,
                "federation_id": federation_id,
            },
        )
        if existing:
            raise CausalGraphConflict("causal edge identity is already bound")
        source = self._require_current_node(
            node_id=edge.source_node_id,
            tenant_id=edge.binding.tenant_id,
            federation_id=federation_id,
        )
        target = self._require_current_node(
            node_id=edge.target_node_id,
            tenant_id=edge.binding.tenant_id,
            federation_id=federation_id,
        )
        source_level = CausalLevel(str(source["causal_level"]))
        target_level = CausalLevel(str(target["causal_level"]))
        if not _level_pair_allowed(edge.edge_kind, source_level, target_level):
            raise FederationAuthorityError(
                "causal edge kind is not admitted for the nominated levels"
            )
        evidence_rows = [
            self._require_evidence(
                evidence_id=evidence_id,
                tenant_id=edge.binding.tenant_id,
                federation_id=federation_id,
            )
            for evidence_id in edge.evidence_refs
        ]
        authoritative_evidence = any(
            str(item["authority_disposition"]) == _AUTHORITATIVE for item in evidence_rows
        )
        if not edge.nomination_only and not authoritative_evidence:
            raise FederationAuthorityError(
                "authoritative causal edges require at least one exact evidence record"
            )
        if edge.nomination_only and authoritative_evidence:
            raise FederationAuthorityError(
                "nomination-only causal edges cannot carry authoritative evidence"
            )
        live_edges = [
            (str(item["source_node_id"]), str(item["target_node_id"]))
            for item in self._client.execute(
                "casf_select_live_causal_edges",
                {
                    "tenant_id": edge.binding.tenant_id,
                    "federation_id": federation_id,
                },
            )
        ]
        cycle = _directed_cycle(
            [*live_edges, (edge.source_node_id, edge.target_node_id)]
        )
        if cycle and not fixed_point_group_id:
            raise CausalCycleError(
                "causal cycles require an explicit fixed-point group"
            )
        return live_edges, cycle

    def _insert_edge(
        self,
        edge: CausalEdge,
        *,
        federation_id: str,
        graph_revision: int,
        recorded_at: str,
        fixed_point_group_id: str,
        admitted_policy_ref: str,
    ) -> None:
        live_edges, cycle = self._prepare_edge(
            edge,
            federation_id=federation_id,
            fixed_point_group_id=fixed_point_group_id,
        )
        population_ref = _evidence_refs_identity(edge.evidence_refs)
        disposition = _authority_disposition(
            authoritative=not edge.nomination_only,
            nomination_only=edge.nomination_only,
        )
        self._client.execute(
            "casf_insert_causal_edge",
            {
                "causal_edge_id": edge.record_id,
                "tenant_id": edge.binding.tenant_id,
                "federation_id": federation_id,
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "edge_kind": edge.edge_kind.value,
                "graph_revision": graph_revision,
                "authority_disposition": disposition,
                "evidence_population_ref": population_ref,
                "admitted_policy_ref": admitted_policy_ref,
                "created_at": recorded_at,
            },
        )
        for evidence_id in edge.evidence_refs:
            attached = self._client.execute(
                "casf_attach_causal_evidence_edge",
                {
                    "causal_edge_id": edge.record_id,
                    "causal_evidence_id": evidence_id,
                    "tenant_id": edge.binding.tenant_id,
                    "federation_id": federation_id,
                },
            )
            if not attached:
                raise CausalGraphConflict("causal evidence is already attached to an edge")
        if cycle:
            node_population = "node-population:" + content_identity(list(cycle))
            edge_population = "edge-population:" + content_identity(
                [item[0] + ">" + item[1] for item in live_edges]
                + [edge.source_node_id + ">" + edge.target_node_id]
            )
            self._client.execute(
                "casf_insert_causal_cycle_slice",
                {
                    "causal_slice_id": fixed_point_group_id,
                    "tenant_id": edge.binding.tenant_id,
                    "federation_id": federation_id,
                    "graph_revision": graph_revision,
                    "root_event_id": edge.record_id,
                    "root_fact_ref": edge.record_id,
                    "node_population_ref": node_population,
                    "edge_population_ref": edge_population,
                    "content_ref": edge.cid,
                    "created_at": recorded_at,
                },
            )

    def _federation_row(
        self, *, tenant_id: str, federation_id: str
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_federation_graph_revision",
            {
                "federation_id": _identifier(federation_id, "federation_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
            },
        )
        if len(rows) != 1:
            raise CausalGraphNotFound("federation is required for causal graph authority")
        return rows[0]

    def _require_current_node(
        self, *, node_id: str, tenant_id: str, federation_id: str
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_causal_node",
            {
                "causal_node_id": node_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if len(rows) != 1:
            raise CausalGraphNotFound("causal edge endpoint is absent")
        row = rows[0]
        if str(row["freshness_state"]) != _CURRENT:
            raise FederationAuthorityError("stale causal nodes cannot admit new edges")
        return row

    def _require_evidence(
        self, *, evidence_id: str, tenant_id: str, federation_id: str
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_causal_evidence",
            {
                "causal_evidence_id": evidence_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if len(rows) != 1:
            raise CausalGraphNotFound("causal edge evidence is absent")
        return rows[0]

    def _assert_binding_scope(
        self, binding: FederationBinding, *, federation_id: str
    ) -> None:
        if not isinstance(binding, FederationBinding):
            raise FederationContractError("binding must be a FederationBinding")
        _identifier(federation_id, "federation_id")

    def _binding_for(
        self,
        *,
        tenant_id: str,
        federation_id: str,
        graph_revision: int,
    ) -> FederationBinding:
        from .contracts import PROGRAM_ID, ROOT_OBJECTIVE

        return FederationBinding(
            tenant_id=tenant_id,
            repository_ids=("repo:casf-causal-graph",),
            repository_tree_ids=("tree:casf-causal-graph",),
            program_id=PROGRAM_ID,
            objective_ref=ROOT_OBJECTIVE,
            objective_revision=1,
            policy_ref="policy:casf-causal-graph",
            policy_revision=1,
            operation_catalog_ref="operations:casf-causal-graph",
            control_plane_generation=1,
            causal_graph_revision=graph_revision,
            semantic_state_roots=("semantic:casf-causal-graph",),
            supervisor_population=1,
            budget_ref="budget:casf-causal-graph",
            expires_at="2099-01-01T00:00:00Z",
            issuer="did:casf:causal-graph",
            authorization_evidence_ref="authz:casf-causal-graph",
        )

    def _node_from_row(self, row: Mapping[str, Any], binding: FederationBinding) -> CausalNode:
        return CausalNode(
            record_id=str(row["causal_node_id"]),
            revision=max(int(row["graph_revision"]), 1),
            binding=replace(binding, causal_graph_revision=int(row["graph_revision"]) or 0),
            level=CausalLevel(str(row["causal_level"])),
            node_type=str(row["node_kind"]),
            subject_ref=str(row["subject_ref"]),
        )

    def _edge_from_row(
        self,
        row: Mapping[str, Any],
        binding: FederationBinding,
        *,
        evidence_refs: tuple[str, ...],
    ) -> CausalEdge:
        return CausalEdge(
            record_id=str(row["causal_edge_id"]),
            revision=max(int(row["graph_revision"]), 1),
            binding=replace(binding, causal_graph_revision=int(row["graph_revision"]) or 0),
            source_node_id=str(row["source_node_id"]),
            target_node_id=str(row["target_node_id"]),
            edge_kind=CausalEdgeKind(str(row["edge_kind"])),
            evidence_refs=evidence_refs,
            nomination_only=str(row["authority_disposition"]) == _NOMINATION_ONLY,
        )

    def _evidence_from_row(
        self, row: Mapping[str, Any], binding: FederationBinding
    ) -> CausalEvidence:
        kind = CausalEvidenceKind(str(row["evidence_kind"]))
        return CausalEvidence(
            record_id=str(row["causal_evidence_id"]),
            revision=1,
            binding=binding,
            evidence_kind=kind,
            evidence_ref=str(row["content_ref"]),
            authoritative=str(row["authority_disposition"]) == _AUTHORITATIVE,
        )


__all__ = (
    "CausalCycleError",
    "CausalGraphCommit",
    "CausalGraphConflict",
    "CausalGraphError",
    "CausalGraphNotFound",
    "CausalGraphSnapshot",
    "CausalGraphStore",
    "_causal_templates",
)
