"""Tree-bound BM25, vector, and knowledge-graph retrieval projections.

Every hit binds index revision, source, tree, score, method, and partition.
Retrieval nominates only: it cannot establish cause, independence, authority,
policy, proof, or completion.  Incremental tree updates invalidate exactly the
affected index records.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ..task_sources.control_plane_contracts import content_identity
from ..task_sources.quack_state_client import QuackStateClient, StatementKind
from .causal_evidence import RetrievalNominationBinding
from .causal_graph import CausalGraphCommit, CausalGraphError
from .contracts import (
    FederationAuthorityError,
    FederationBinding,
    FederationContractError,
    _identifier,
    _integer,
    _text,
)
from .proof_projection import ProofProjectionStore
from .registry import _template

RETRIEVAL_METHODS = frozenset({"bm25", "vector", "kg", "lexical", "hybrid"})
AUTHORITY_NOMINATION = "nomination_only"
FRESHNESS_STATES = frozenset({"current", "stale", "invalidated"})


class RetrievalProjectionError(CausalGraphError):
    """Base typed retrieval-projection failure."""


class RetrievalProjectionAuthorityError(FederationAuthorityError, RetrievalProjectionError):
    """An attempt to mint authority from retrieval or ignore tree identity."""


def _reject_sibling_path(value: str, name: str) -> None:
    if value.startswith(("/", "~")) or ".." in value.split("/"):
        raise RetrievalProjectionAuthorityError(
            f"{name} is a sibling filesystem path, not an opaque retrieval identity"
        )


@dataclass(frozen=True)
class RetrievalIndexProjection:
    """One tree-bound BM25, vector, or knowledge-graph index root."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/retrieval-index@1"
    )

    record_id: str
    retrieval_method: str
    index_revision: int
    index_root: str
    repository_id: str
    tree_id: str
    content_ref: str
    privacy_scope_ref: str = "privacy:federation"

    def __post_init__(self) -> None:
        _reject_sibling_path(str(self.index_root), "index_root")
        _identifier(self.record_id, "record_id")
        method = _text(self.retrieval_method, "retrieval_method", maximum=64).casefold()
        if method not in RETRIEVAL_METHODS:
            raise FederationContractError("retrieval method is not closed")
        object.__setattr__(self, "retrieval_method", method)
        _integer(self.index_revision, "index_revision", minimum=1)
        _identifier(self.index_root, "index_root")
        _identifier(self.repository_id, "repository_id")
        _identifier(self.tree_id, "tree_id")
        _identifier(self.content_ref, "content_ref")
        _identifier(self.privacy_scope_ref, "privacy_scope_ref")


@dataclass(frozen=True)
class RetrievalNominationProjection:
    """One nomination-only retrieval hit with exact release bindings."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/retrieval-nomination@1"
    )

    record_id: str
    index_id: str
    subject_kind: str
    subject_ref: str
    score_millionths: int
    method: str
    source_cid: str
    tree_id: str
    index_revision: str
    partition_id: str = ""
    content_ref: str = ""

    def __post_init__(self) -> None:
        _identifier(self.record_id, "record_id")
        _identifier(self.index_id, "index_id")
        _identifier(self.subject_kind, "subject_kind")
        _identifier(self.subject_ref, "subject_ref")
        _integer(self.score_millionths, "score_millionths", maximum=1_000_000)
        method = _text(self.method, "method", maximum=64).casefold()
        if method not in RETRIEVAL_METHODS:
            raise FederationContractError("retrieval method is not closed")
        object.__setattr__(self, "method", method)
        _identifier(self.source_cid, "source_cid")
        _identifier(self.tree_id, "tree_id")
        _identifier(self.index_revision, "index_revision")
        _identifier(self.partition_id, "partition_id", required=False)
        _identifier(self.content_ref, "content_ref", required=False)

    @property
    def authority_disposition(self) -> str:
        return AUTHORITY_NOMINATION

    @property
    def nomination_binding(self) -> RetrievalNominationBinding:
        return RetrievalNominationBinding(
            index_revision=self.index_revision,
            source_cid=self.source_cid,
            tree_id=self.tree_id,
            method=self.method,
            score_millionths=self.score_millionths,
            partition_id=self.partition_id,
        )


@dataclass(frozen=True)
class KnowledgeGraphRelationProjection:
    """Bounded KG edge. Traversal nominates relationships; it does not prove cause."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/kg-relation@1"
    )

    record_id: str
    index_id: str
    index_revision: int
    source_node_id: str
    target_node_id: str
    relationship_kind: str
    content_ref: str
    tree_id: str

    def __post_init__(self) -> None:
        _identifier(self.record_id, "record_id")
        _identifier(self.index_id, "index_id")
        _integer(self.index_revision, "index_revision", minimum=1)
        _identifier(self.source_node_id, "source_node_id")
        _identifier(self.target_node_id, "target_node_id")
        _identifier(self.relationship_kind, "relationship_kind")
        _identifier(self.content_ref, "content_ref")
        _identifier(self.tree_id, "tree_id")

    @property
    def authority_disposition(self) -> str:
        return AUTHORITY_NOMINATION


def retrieval_establishes_authority() -> bool:
    """Retrieval cannot mint cause, independence, policy, proof, or completion."""

    return False


def bind_index(
    *,
    binding: FederationBinding,
    retrieval_method: str,
    index_revision: int,
    index_root: str,
    content_ref: str,
    record_id: str,
    privacy_scope_ref: str = "privacy:federation",
    tree_id: str = "",
    repository_id: str = "",
) -> RetrievalIndexProjection:
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    projection = RetrievalIndexProjection(
        record_id=record_id,
        retrieval_method=retrieval_method,
        index_revision=index_revision,
        index_root=index_root,
        repository_id=repository_id or binding.repository_ids[0],
        tree_id=tree_id or binding.repository_tree_ids[0],
        content_ref=content_ref,
        privacy_scope_ref=privacy_scope_ref,
    )
    if projection.tree_id != binding.repository_tree_ids[0]:
        raise RetrievalProjectionAuthorityError("retrieval index tree identity mismatches")
    if projection.repository_id != binding.repository_ids[0]:
        raise RetrievalProjectionAuthorityError("retrieval index repository is not bound")
    return projection


def bind_nomination(
    *,
    binding: FederationBinding,
    index_id: str,
    subject_kind: str,
    subject_ref: str,
    score_millionths: int,
    method: str,
    source_cid: str,
    index_revision: str,
    record_id: str,
    partition_id: str = "",
    content_ref: str = "",
    tree_id: str = "",
) -> RetrievalNominationProjection:
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    projection = RetrievalNominationProjection(
        record_id=record_id,
        index_id=index_id,
        subject_kind=subject_kind,
        subject_ref=subject_ref,
        score_millionths=score_millionths,
        method=method,
        source_cid=source_cid,
        tree_id=tree_id or binding.repository_tree_ids[0],
        index_revision=index_revision,
        partition_id=partition_id,
        content_ref=content_ref or source_cid,
    )
    if projection.tree_id != binding.repository_tree_ids[0]:
        raise RetrievalProjectionAuthorityError(
            "retrieval nomination tree identity mismatches"
        )
    return projection


def bind_nomination_from_hit(
    hit: RetrievalNominationBinding,
    *,
    binding: FederationBinding,
    index_id: str,
    subject_kind: str,
    subject_ref: str,
    record_id: str,
) -> RetrievalNominationProjection:
    if not isinstance(hit, RetrievalNominationBinding):
        raise FederationContractError("hit must be a RetrievalNominationBinding")
    return bind_nomination(
        binding=binding,
        index_id=index_id,
        subject_kind=subject_kind,
        subject_ref=subject_ref,
        score_millionths=hit.score_millionths,
        method=hit.method,
        source_cid=hit.source_cid,
        index_revision=hit.index_revision,
        record_id=record_id,
        partition_id=hit.partition_id,
        content_ref=hit.cid,
        tree_id=hit.tree_id,
    )


def bind_kg_relation(
    *,
    binding: FederationBinding,
    index_id: str,
    index_revision: int,
    source_node_id: str,
    target_node_id: str,
    relationship_kind: str,
    content_ref: str,
    record_id: str,
    tree_id: str = "",
) -> KnowledgeGraphRelationProjection:
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    projection = KnowledgeGraphRelationProjection(
        record_id=record_id,
        index_id=index_id,
        index_revision=index_revision,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        relationship_kind=relationship_kind,
        content_ref=content_ref,
        tree_id=tree_id or binding.repository_tree_ids[0],
    )
    if projection.tree_id != binding.repository_tree_ids[0]:
        raise RetrievalProjectionAuthorityError("knowledge-graph tree identity mismatches")
    return projection


def indexes_invalidated_by_tree_change(
    indexes: Sequence[RetrievalIndexProjection],
    *,
    tree_id: str,
) -> tuple[str, ...]:
    tree = _identifier(tree_id, "tree_id")
    affected: list[str] = []
    for index in indexes:
        if not isinstance(index, RetrievalIndexProjection):
            raise FederationContractError("indexes must be RetrievalIndexProjection records")
        if index.tree_id == tree:
            affected.append(index.record_id)
    return tuple(dict.fromkeys(affected))


def _retrieval_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_insert_retrieval_index",
            """
            INSERT INTO retrieval_indexes (
                retrieval_index_id, tenant_id, federation_id, repository_id,
                tree_id, retrieval_method, index_revision, index_root,
                owner_id, source_root, content_ref, freshness_state,
                privacy_scope_ref, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "retrieval_index_id",
                "tenant_id",
                "federation_id",
                "repository_id",
                "tree_id",
                "retrieval_method",
                "index_revision",
                "index_root",
                "owner_id",
                "source_root",
                "content_ref",
                "freshness_state",
                "privacy_scope_ref",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_retrieval_index",
            """
            SELECT retrieval_index_id, retrieval_method, index_revision,
                   index_root, tree_id, content_ref, freshness_state
            FROM retrieval_indexes
            WHERE retrieval_index_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("retrieval_index_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_retrieval_receipt",
            """
            INSERT INTO retrieval_receipts (
                retrieval_receipt_id, tenant_id, federation_id,
                retrieval_index_id, index_revision, repository_id, tree_id,
                retrieval_method, partition_population_ref, result_population_ref,
                owner_id, source_root, content_ref, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "retrieval_receipt_id",
                "tenant_id",
                "federation_id",
                "retrieval_index_id",
                "index_revision",
                "repository_id",
                "tree_id",
                "retrieval_method",
                "partition_population_ref",
                "result_population_ref",
                "owner_id",
                "source_root",
                "content_ref",
                "recorded_at",
            ),
        ),
        _template(
            "casf_insert_retrieval_nomination",
            """
            INSERT INTO retrieval_nominations (
                retrieval_nomination_id, tenant_id, federation_id,
                retrieval_receipt_id, subject_kind, subject_ref, score_micros,
                authority_disposition, owner_id, source_root, content_ref,
                recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'nomination_only', ?, ?, ?, ?)
            """,
            (
                "retrieval_nomination_id",
                "tenant_id",
                "federation_id",
                "retrieval_receipt_id",
                "subject_kind",
                "subject_ref",
                "score_micros",
                "owner_id",
                "source_root",
                "content_ref",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_retrieval_nomination",
            """
            SELECT retrieval_nomination_id, subject_kind, subject_ref,
                   score_micros, authority_disposition, content_ref
            FROM retrieval_nominations
            WHERE retrieval_nomination_id = ? AND tenant_id = ?
              AND federation_id = ?
            LIMIT 1
            """,
            ("retrieval_nomination_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_knowledge_graph_edge",
            """
            INSERT INTO knowledge_graph_edges (
                knowledge_graph_edge_id, retrieval_index_id, index_revision,
                source_node_id, target_node_id, relationship_kind, evidence_ref,
                authority_disposition, owner_id, source_root, provenance_ref,
                content_ref, revision, status, freshness_state, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'nomination_only', ?, ?, ?, ?, ?,
                      'current', 'current', ?)
            """,
            (
                "knowledge_graph_edge_id",
                "retrieval_index_id",
                "index_revision",
                "source_node_id",
                "target_node_id",
                "relationship_kind",
                "evidence_ref",
                "owner_id",
                "source_root",
                "provenance_ref",
                "content_ref",
                "revision",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_knowledge_graph_edge",
            """
            SELECT knowledge_graph_edge_id, source_node_id, target_node_id,
                   relationship_kind, authority_disposition, freshness_state
            FROM knowledge_graph_edges
            WHERE knowledge_graph_edge_id = ?
            LIMIT 1
            """,
            ("knowledge_graph_edge_id",),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_mark_retrieval_index_invalidated",
            """
            UPDATE retrieval_indexes
            SET freshness_state = 'invalidated'
            WHERE retrieval_index_id = ? AND tenant_id = ? AND federation_id = ?
              AND freshness_state = 'current'
            RETURNING retrieval_index_id
            """,
            ("retrieval_index_id", "tenant_id", "federation_id"),
        ),
    )


class RetrievalProjectionStore(ProofProjectionStore):
    """Persist nomination-only BM25, vector, and KG projections."""

    INTERFACE = "RetrievalProjectionStore@1"

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
            raise RetrievalProjectionError(
                "retrieval projection store never accepts a database path"
            )
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise RetrievalProjectionError(
                "retrieval projection store requires an already-attached typed state client"
            )
        registered = set(client.list_templates())
        missing = [
            template.name
            for template in _retrieval_templates()
            if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise RetrievalProjectionError(
                    "retrieval projection templates are absent from the sealed catalog"
                )
        else:
            for template in _retrieval_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    def record_index(
        self,
        projection: RetrievalIndexProjection,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        event_id: str,
    ) -> CausalGraphCommit:
        bound = bind_index(
            binding=binding,
            retrieval_method=projection.retrieval_method,
            index_revision=projection.index_revision,
            index_root=projection.index_root,
            content_ref=projection.content_ref,
            record_id=projection.record_id,
            privacy_scope_ref=projection.privacy_scope_ref,
            tree_id=projection.tree_id,
            repository_id=projection.repository_id,
        )
        return self._commit_fact(
            operation="federation.retrieval.index.record",
            fact_id=bound.record_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(
                dict.fromkeys((bound.record_id, bound.index_root, event_id))
            ),
            payload_ref=bound.content_ref,
            prepare_fact=lambda: self._prepare_index(
                bound.record_id,
                tenant_id=binding.tenant_id,
                federation_id=federation_id,
            ),
            apply_fact=lambda revision, recorded_at: self._insert_index(
                bound,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                recorded_at=recorded_at,
            ),
        )

    def record_nomination(
        self,
        projection: RetrievalNominationProjection,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        event_id: str,
        receipt_id: str,
    ) -> CausalGraphCommit:
        bound = bind_nomination(
            binding=binding,
            index_id=projection.index_id,
            subject_kind=projection.subject_kind,
            subject_ref=projection.subject_ref,
            score_millionths=projection.score_millionths,
            method=projection.method,
            source_cid=projection.source_cid,
            index_revision=projection.index_revision,
            record_id=projection.record_id,
            partition_id=projection.partition_id,
            content_ref=projection.content_ref,
            tree_id=projection.tree_id,
        )
        if bound.authority_disposition != AUTHORITY_NOMINATION:
            raise RetrievalProjectionAuthorityError(
                "retrieval hits cannot be recorded as authoritative"
            )
        return self._commit_fact(
            operation="federation.retrieval.nomination.record",
            fact_id=bound.record_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(
                dict.fromkeys((bound.record_id, bound.subject_ref, event_id))
            ),
            payload_ref=bound.content_ref or bound.source_cid,
            prepare_fact=lambda: None,
            apply_fact=lambda revision, recorded_at: self._insert_nomination(
                bound,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                repository_id=binding.repository_ids[0],
                receipt_id=receipt_id,
                recorded_at=recorded_at,
            ),
        )

    def record_kg_relation(
        self,
        projection: KnowledgeGraphRelationProjection,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        event_id: str,
    ) -> CausalGraphCommit:
        bound = bind_kg_relation(
            binding=binding,
            index_id=projection.index_id,
            index_revision=projection.index_revision,
            source_node_id=projection.source_node_id,
            target_node_id=projection.target_node_id,
            relationship_kind=projection.relationship_kind,
            content_ref=projection.content_ref,
            record_id=projection.record_id,
            tree_id=projection.tree_id,
        )
        return self._commit_fact(
            operation="federation.retrieval.kg.record",
            fact_id=bound.record_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(
                dict.fromkeys(
                    (bound.record_id, bound.source_node_id, bound.target_node_id, event_id)
                )
            ),
            payload_ref=bound.content_ref,
            prepare_fact=lambda: None,
            apply_fact=lambda revision, recorded_at: self._insert_kg_relation(
                bound,
                recorded_at=recorded_at,
            ),
        )

    def invalidate_indexes(
        self,
        indexes: Sequence[RetrievalIndexProjection],
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        event_id: str,
    ) -> CausalGraphCommit:
        affected = indexes_invalidated_by_tree_change(
            indexes, tree_id=binding.repository_tree_ids[0]
        )
        evidence = content_identity({"event_id": event_id, "indexes": list(affected)})
        return self._commit_fact(
            operation="federation.retrieval.index.invalidate",
            fact_id="invalidation:" + evidence,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(dict.fromkeys((event_id, *affected))),
            payload_ref=evidence,
            prepare_fact=lambda: None,
            apply_fact=lambda revision, recorded_at: self._apply_index_invalidations(
                affected,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
            ),
        )

    def load_index(self, *, record_id: str, tenant_id: str, federation_id: str) -> dict[str, Any]:
        rows = self._client.execute(
            "casf_select_retrieval_index",
            {
                "retrieval_index_id": record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if len(rows) != 1:
            raise RetrievalProjectionError("retrieval index is absent")
        return dict(rows[0])

    def load_nomination(
        self, *, record_id: str, tenant_id: str, federation_id: str
    ) -> dict[str, Any]:
        rows = self._client.execute(
            "casf_select_retrieval_nomination",
            {
                "retrieval_nomination_id": record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if len(rows) != 1:
            raise RetrievalProjectionError("retrieval nomination is absent")
        return dict(rows[0])

    def load_kg_relation(self, *, record_id: str) -> dict[str, Any]:
        rows = self._client.execute(
            "casf_select_knowledge_graph_edge",
            {"knowledge_graph_edge_id": record_id},
        )
        if len(rows) != 1:
            raise RetrievalProjectionError("knowledge-graph relation is absent")
        return dict(rows[0])

    def _prepare_index(self, record_id: str, *, tenant_id: str, federation_id: str) -> None:
        existing = self._client.execute(
            "casf_select_retrieval_index",
            {
                "retrieval_index_id": record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if existing:
            raise RetrievalProjectionError("retrieval index is already bound")

    def _insert_index(
        self,
        projection: RetrievalIndexProjection,
        *,
        federation_id: str,
        tenant_id: str,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_retrieval_index",
            {
                "retrieval_index_id": projection.record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "repository_id": projection.repository_id,
                "tree_id": projection.tree_id,
                "retrieval_method": projection.retrieval_method,
                "index_revision": projection.index_revision,
                "index_root": projection.index_root,
                "owner_id": "retrieval-projection",
                "source_root": projection.tree_id,
                "content_ref": projection.content_ref,
                "freshness_state": "current",
                "privacy_scope_ref": projection.privacy_scope_ref,
                "recorded_at": recorded_at,
            },
        )

    def _insert_nomination(
        self,
        projection: RetrievalNominationProjection,
        *,
        federation_id: str,
        tenant_id: str,
        repository_id: str,
        receipt_id: str,
        recorded_at: str,
    ) -> None:
        receipt = _identifier(receipt_id, "receipt_id")
        self._client.execute(
            "casf_insert_retrieval_receipt",
            {
                "retrieval_receipt_id": receipt,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "retrieval_index_id": projection.index_id,
                "index_revision": int(projection.index_revision)
                if str(projection.index_revision).isdigit()
                else 1,
                "repository_id": repository_id,
                "tree_id": projection.tree_id,
                "retrieval_method": projection.method,
                "partition_population_ref": projection.partition_id or "partition:default",
                "result_population_ref": projection.record_id,
                "owner_id": "retrieval-projection",
                "source_root": projection.tree_id,
                "content_ref": projection.content_ref or projection.source_cid,
                "recorded_at": recorded_at,
            },
        )
        self._client.execute(
            "casf_insert_retrieval_nomination",
            {
                "retrieval_nomination_id": projection.record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "retrieval_receipt_id": receipt,
                "subject_kind": projection.subject_kind,
                "subject_ref": projection.subject_ref,
                "score_micros": projection.score_millionths,
                "owner_id": "retrieval-projection",
                "source_root": projection.tree_id,
                "content_ref": projection.content_ref or projection.source_cid,
                "recorded_at": recorded_at,
            },
        )

    def _insert_kg_relation(
        self,
        projection: KnowledgeGraphRelationProjection,
        *,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_knowledge_graph_edge",
            {
                "knowledge_graph_edge_id": projection.record_id,
                "retrieval_index_id": projection.index_id,
                "index_revision": projection.index_revision,
                "source_node_id": projection.source_node_id,
                "target_node_id": projection.target_node_id,
                "relationship_kind": projection.relationship_kind,
                "evidence_ref": projection.content_ref,
                "owner_id": "retrieval-projection",
                "source_root": projection.tree_id,
                "provenance_ref": projection.content_ref,
                "content_ref": projection.content_ref,
                "revision": 1,
                "recorded_at": recorded_at,
            },
        )

    def _apply_index_invalidations(
        self,
        indexes: Sequence[str],
        *,
        federation_id: str,
        tenant_id: str,
    ) -> None:
        for index_id in indexes:
            self._client.execute(
                "casf_mark_retrieval_index_invalidated",
                {
                    "retrieval_index_id": index_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                },
            )


__all__ = (
    "KnowledgeGraphRelationProjection",
    "RetrievalIndexProjection",
    "RetrievalNominationProjection",
    "RetrievalProjectionAuthorityError",
    "RetrievalProjectionError",
    "RetrievalProjectionStore",
    "bind_index",
    "bind_kg_relation",
    "bind_nomination",
    "bind_nomination_from_hit",
    "indexes_invalidated_by_tree_change",
    "retrieval_establishes_authority",
)
