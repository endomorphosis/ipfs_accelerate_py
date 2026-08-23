"""Tree-bound AST, symbol, semantic-root, and capsule projections.

The accelerator records opaque ``ipfs_datasets_py`` identities and never
reinterprets their meaning.  Incremental tree updates invalidate exactly the
affected capsules.  Projections cannot establish authority, policy, proof, or
completion, and they cannot write a sibling repository.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ..semantic_state.contracts import SemanticCapsuleRef
from ..task_sources.control_plane_contracts import content_identity
from ..task_sources.quack_state_client import QuackStateClient, StatementKind
from .causal_graph import CausalGraphCommit, CausalGraphError
from .contracts import (
    FederationAuthorityError,
    FederationBinding,
    FederationContractError,
    _identifier,
)
from .registry import _template
from .world_snapshot import WorldSnapshotStore

SEMANTIC_KINDS = frozenset({"ast", "symbol", "semantic_state", "capsule_index"})
SUBJECT_KINDS = frozenset({"ast", "symbol", "file", "capsule"})
FRESHNESS_STATES = frozenset({"current", "stale", "invalidated"})
INVALIDATION_REASONS = frozenset(
    {"ast_change", "symbol_change", "semantic_root_change", "capsule_stale"}
)
SEMANTIC_OWNER = "ipfs_datasets_py"


class SemanticProjectionError(CausalGraphError):
    """Base typed semantic-projection failure."""


class SemanticProjectionAuthorityError(FederationAuthorityError, SemanticProjectionError):
    """An attempt to reinterpret datasets meaning or ignore tree identity."""


@dataclass(frozen=True)
class SemanticRootProjection:
    """Opaque tree-bound semantic root recorded by the accelerator."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/semantic-root-projection@1"
    )

    record_id: str
    semantic_kind: str
    semantic_root: str
    repository_id: str
    tree_id: str
    content_ref: str
    revision: int = 1

    def __post_init__(self) -> None:
        _identifier(self.record_id, "record_id")
        kind = _identifier(self.semantic_kind, "semantic_kind")
        if kind not in SEMANTIC_KINDS:
            raise FederationContractError("semantic_kind is not closed")
        object.__setattr__(self, "semantic_kind", kind)
        _identifier(self.semantic_root, "semantic_root")
        _identifier(self.repository_id, "repository_id")
        _identifier(self.tree_id, "tree_id")
        _identifier(self.content_ref, "content_ref")
        if self.revision < 1:
            raise FederationContractError("revision must be >= 1")
        _reject_sibling_path(self.semantic_root, "semantic_root")
        _reject_sibling_path(self.content_ref, "content_ref")


@dataclass(frozen=True)
class CapsuleProjection:
    """Opaque capsule reference bound to one tree and dependency root."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/capsule-projection@1"
    )

    record_id: str
    subject_kind: str
    subject_ref: str
    dependency_root: str
    repository_id: str
    tree_id: str
    content_ref: str
    revision: int = 1

    def __post_init__(self) -> None:
        _reject_sibling_path(str(self.subject_ref), "subject_ref")
        _reject_sibling_path(str(self.content_ref), "content_ref")
        _identifier(self.record_id, "record_id")
        kind = _identifier(self.subject_kind, "subject_kind")
        if kind not in SUBJECT_KINDS:
            raise FederationContractError("subject_kind is not closed")
        object.__setattr__(self, "subject_kind", kind)
        _identifier(self.subject_ref, "subject_ref")
        _identifier(self.dependency_root, "dependency_root")
        _identifier(self.repository_id, "repository_id")
        _identifier(self.tree_id, "tree_id")
        _identifier(self.content_ref, "content_ref")
        if self.revision < 1:
            raise FederationContractError("revision must be >= 1")


def _reject_sibling_path(value: str, name: str) -> None:
    if value.startswith(("/", "~")) or ".." in value.split("/"):
        raise SemanticProjectionAuthorityError(
            f"{name} is a sibling filesystem path, not an opaque semantic identity"
        )


def bind_semantic_root(
    *,
    binding: FederationBinding,
    semantic_kind: str,
    semantic_root: str,
    content_ref: str,
    record_id: str,
    revision: int = 1,
    repository_id: str = "",
    tree_id: str = "",
) -> SemanticRootProjection:
    """Admit one datasets semantic root against the federation tree binding."""

    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    projection = SemanticRootProjection(
        record_id=record_id,
        semantic_kind=semantic_kind,
        semantic_root=semantic_root,
        repository_id=repository_id or binding.repository_ids[0],
        tree_id=tree_id or binding.repository_tree_ids[0],
        content_ref=content_ref,
        revision=revision,
    )
    if projection.repository_id != binding.repository_ids[0]:
        raise SemanticProjectionAuthorityError("semantic root repository is not bound")
    if projection.tree_id != binding.repository_tree_ids[0]:
        raise SemanticProjectionAuthorityError("semantic root tree identity mismatches")
    if (
        projection.semantic_kind == "semantic_state"
        and projection.semantic_root not in binding.semantic_state_roots
    ):
        raise SemanticProjectionAuthorityError(
            "semantic state root is not tree-bound to this federation"
        )
    return projection


def bind_capsule(
    *,
    binding: FederationBinding,
    subject_kind: str,
    subject_ref: str,
    dependency_root: str,
    content_ref: str,
    record_id: str,
    revision: int = 1,
    repository_id: str = "",
    tree_id: str = "",
) -> CapsuleProjection:
    """Admit one capsule identity without reinterpreting datasets meaning."""

    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    projection = CapsuleProjection(
        record_id=record_id,
        subject_kind=subject_kind,
        subject_ref=subject_ref,
        dependency_root=dependency_root,
        repository_id=repository_id or binding.repository_ids[0],
        tree_id=tree_id or binding.repository_tree_ids[0],
        content_ref=content_ref,
        revision=revision,
    )
    if projection.tree_id != binding.repository_tree_ids[0]:
        raise SemanticProjectionAuthorityError("capsule tree identity mismatches")
    if (
        projection.dependency_root not in binding.semantic_state_roots
        and projection.dependency_root != binding.repository_tree_ids[0]
    ):
        raise SemanticProjectionAuthorityError(
            "capsule dependency root is not bound to this federation tree"
        )
    return projection


def bind_datasets_capsule_ref(
    capsule: SemanticCapsuleRef,
    *,
    binding: FederationBinding,
    record_id: str,
) -> CapsuleProjection:
    """Project a datasets capsule CID as an opaque federation reference."""

    if not isinstance(capsule, SemanticCapsuleRef):
        raise FederationContractError("capsule must be a SemanticCapsuleRef")
    if capsule.semantic_state_root_cid not in binding.semantic_state_roots:
        raise SemanticProjectionAuthorityError(
            "capsule semantic root is not tree-bound to this federation"
        )
    return bind_capsule(
        binding=binding,
        subject_kind="symbol",
        subject_ref=capsule.stable_symbol_id
        if _identifier_ok(capsule.stable_symbol_id)
        else capsule.capsule_cid,
        dependency_root=capsule.semantic_state_root_cid,
        content_ref=capsule.capsule_cid,
        record_id=record_id,
    )


def _identifier_ok(value: str) -> bool:
    try:
        _identifier(value, "subject_ref")
    except FederationContractError:
        return False
    return True


def capsules_invalidated_by_change(
    capsules: Sequence[CapsuleProjection],
    *,
    changed_subject_refs: Sequence[str] = (),
    changed_semantic_root: str = "",
    tree_id: str,
) -> tuple[str, ...]:
    """Return capsule identities affected by an incremental tree change."""

    changed = {
        _identifier(item, "changed_subject_refs") for item in changed_subject_refs
    }
    root = _identifier(changed_semantic_root, "changed_semantic_root", required=False)
    tree = _identifier(tree_id, "tree_id")
    affected: list[str] = []
    for capsule in capsules:
        if not isinstance(capsule, CapsuleProjection):
            raise FederationContractError("capsules must be CapsuleProjection records")
        if capsule.tree_id != tree:
            raise SemanticProjectionAuthorityError(
                "capsule tree identity mismatches the incremental update"
            )
        if capsule.subject_ref in changed:
            affected.append(capsule.record_id)
        elif root and capsule.dependency_root == root:
            affected.append(capsule.record_id)
    return tuple(dict.fromkeys(affected))


def federation_may_reinterpret_semantics() -> bool:
    """The accelerator never changes datasets semantic meaning."""

    return False


def _projection_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_insert_semantic_root_reference",
            """
            INSERT INTO semantic_root_references (
                semantic_root_reference_id, tenant_id, federation_id,
                repository_id, tree_id, semantic_root, semantic_kind,
                owner_id, source_root, content_ref, revision, status,
                freshness_state, provenance_ref, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "semantic_root_reference_id",
                "tenant_id",
                "federation_id",
                "repository_id",
                "tree_id",
                "semantic_root",
                "semantic_kind",
                "owner_id",
                "source_root",
                "content_ref",
                "revision",
                "status",
                "freshness_state",
                "provenance_ref",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_semantic_root_reference",
            """
            SELECT semantic_root_reference_id, semantic_kind, semantic_root,
                   tree_id, content_ref, freshness_state
            FROM semantic_root_references
            WHERE semantic_root_reference_id = ? AND tenant_id = ?
              AND federation_id = ?
            LIMIT 1
            """,
            (
                "semantic_root_reference_id",
                "tenant_id",
                "federation_id",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_semantic_capsule_reference",
            """
            INSERT INTO semantic_capsule_references (
                semantic_capsule_reference_id, tenant_id, federation_id,
                repository_id, tree_id, subject_kind, subject_ref,
                dependency_root, owner_id, source_root, content_ref, revision,
                status, freshness_state, provenance_ref, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "semantic_capsule_reference_id",
                "tenant_id",
                "federation_id",
                "repository_id",
                "tree_id",
                "subject_kind",
                "subject_ref",
                "dependency_root",
                "owner_id",
                "source_root",
                "content_ref",
                "revision",
                "status",
                "freshness_state",
                "provenance_ref",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_semantic_capsule_reference",
            """
            SELECT semantic_capsule_reference_id, subject_kind, subject_ref,
                   dependency_root, tree_id, content_ref, freshness_state
            FROM semantic_capsule_references
            WHERE semantic_capsule_reference_id = ? AND tenant_id = ?
              AND federation_id = ?
            LIMIT 1
            """,
            (
                "semantic_capsule_reference_id",
                "tenant_id",
                "federation_id",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_mark_capsule_invalidated",
            """
            UPDATE semantic_capsule_references
            SET freshness_state = 'invalidated', status = 'invalidated'
            WHERE semantic_capsule_reference_id = ? AND tenant_id = ?
              AND federation_id = ? AND freshness_state = 'current'
            RETURNING semantic_capsule_reference_id
            """,
            (
                "semantic_capsule_reference_id",
                "tenant_id",
                "federation_id",
            ),
        ),
        _template(
            "casf_insert_causal_invalidation",
            """
            INSERT INTO causal_invalidations (
                causal_invalidation_id, tenant_id, federation_id, event_id,
                graph_revision, subject_kind, subject_ref, reason_kind,
                evidence_ref, state, created_at, resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'issued', ?, NULL)
            """,
            (
                "causal_invalidation_id",
                "tenant_id",
                "federation_id",
                "event_id",
                "graph_revision",
                "subject_kind",
                "subject_ref",
                "reason_kind",
                "evidence_ref",
                "created_at",
            ),
        ),
    )


class SemanticProjectionStore(WorldSnapshotStore):
    """Persist opaque semantic projections through the exclusive state owner."""

    INTERFACE = "SemanticProjectionStore@1"

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
            raise SemanticProjectionError(
                "semantic projection store never accepts a database path"
            )
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise SemanticProjectionError(
                "semantic projection store requires an already-attached typed state client"
            )
        registered = set(client.list_templates())
        missing = [
            template.name
            for template in _projection_templates()
            if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise SemanticProjectionError(
                    "semantic projection templates are absent from the sealed catalog"
                )
        else:
            for template in _projection_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    def record_semantic_root(
        self,
        projection: SemanticRootProjection,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        event_id: str,
    ) -> CausalGraphCommit:
        bound = bind_semantic_root(
            binding=binding,
            semantic_kind=projection.semantic_kind,
            semantic_root=projection.semantic_root,
            content_ref=projection.content_ref,
            record_id=projection.record_id,
            revision=projection.revision,
        )
        return self._commit_fact(
            operation="federation.semantic.root.record",
            fact_id=bound.record_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=(bound.record_id, bound.semantic_root, event_id),
            payload_ref=bound.content_ref,
            prepare_fact=lambda: self._prepare_root(
                bound.record_id,
                tenant_id=binding.tenant_id,
                federation_id=federation_id,
            ),
            apply_fact=lambda revision, recorded_at: self._insert_root(
                bound,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                recorded_at=recorded_at,
            ),
        )

    def record_capsule(
        self,
        projection: CapsuleProjection,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        event_id: str,
    ) -> CausalGraphCommit:
        bound = bind_capsule(
            binding=binding,
            subject_kind=projection.subject_kind,
            subject_ref=projection.subject_ref,
            dependency_root=projection.dependency_root,
            content_ref=projection.content_ref,
            record_id=projection.record_id,
            revision=projection.revision,
        )
        return self._commit_fact(
            operation="federation.semantic.capsule.record",
            fact_id=bound.record_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=(bound.record_id, bound.subject_ref, event_id),
            payload_ref=bound.content_ref,
            prepare_fact=lambda: self._prepare_capsule(
                bound.record_id,
                tenant_id=binding.tenant_id,
                federation_id=federation_id,
            ),
            apply_fact=lambda revision, recorded_at: self._insert_capsule(
                bound,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                recorded_at=recorded_at,
            ),
        )

    def invalidate_capsules(
        self,
        capsules: Sequence[CapsuleProjection],
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        event_id: str,
        changed_subject_refs: Sequence[str] = (),
        changed_semantic_root: str = "",
        reason_kind: str = "symbol_change",
    ) -> CausalGraphCommit:
        if reason_kind not in INVALIDATION_REASONS:
            raise FederationContractError("invalidation reason is not closed")
        affected = capsules_invalidated_by_change(
            capsules,
            changed_subject_refs=changed_subject_refs,
            changed_semantic_root=changed_semantic_root,
            tree_id=binding.repository_tree_ids[0],
        )
        evidence = content_identity(
            {
                "event_id": event_id,
                "affected": list(affected),
                "reason_kind": reason_kind,
            }
        )
        return self._commit_fact(
            operation="federation.semantic.capsule.invalidate",
            fact_id="invalidation:" + evidence,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=(event_id, *affected) if affected else (event_id,),
            payload_ref=evidence,
            prepare_fact=lambda: None,
            apply_fact=lambda revision, recorded_at: self._apply_invalidations(
                affected,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                event_id=event_id,
                graph_revision=revision,
                recorded_at=recorded_at,
                reason_kind=reason_kind,
                evidence_ref=evidence,
            ),
        )

    def load_semantic_root(
        self, *, record_id: str, tenant_id: str, federation_id: str
    ) -> dict[str, Any]:
        rows = self._client.execute(
            "casf_select_semantic_root_reference",
            {
                "semantic_root_reference_id": record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if len(rows) != 1:
            raise SemanticProjectionError("semantic root projection is absent")
        return dict(rows[0])

    def load_capsule(
        self, *, record_id: str, tenant_id: str, federation_id: str
    ) -> dict[str, Any]:
        rows = self._client.execute(
            "casf_select_semantic_capsule_reference",
            {
                "semantic_capsule_reference_id": record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if len(rows) != 1:
            raise SemanticProjectionError("capsule projection is absent")
        return dict(rows[0])

    def _prepare_root(self, record_id: str, *, tenant_id: str, federation_id: str) -> None:
        existing = self._client.execute(
            "casf_select_semantic_root_reference",
            {
                "semantic_root_reference_id": record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if existing:
            raise SemanticProjectionError("semantic root projection is already bound")

    def _prepare_capsule(
        self, record_id: str, *, tenant_id: str, federation_id: str
    ) -> None:
        existing = self._client.execute(
            "casf_select_semantic_capsule_reference",
            {
                "semantic_capsule_reference_id": record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if existing:
            raise SemanticProjectionError("capsule projection is already bound")

    def _insert_root(
        self,
        projection: SemanticRootProjection,
        *,
        federation_id: str,
        tenant_id: str,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_semantic_root_reference",
            {
                "semantic_root_reference_id": projection.record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "repository_id": projection.repository_id,
                "tree_id": projection.tree_id,
                "semantic_root": projection.semantic_root,
                "semantic_kind": projection.semantic_kind,
                "owner_id": SEMANTIC_OWNER,
                "source_root": projection.tree_id,
                "content_ref": projection.content_ref,
                "revision": projection.revision,
                "status": "current",
                "freshness_state": "current",
                "provenance_ref": SEMANTIC_OWNER,
                "recorded_at": recorded_at,
            },
        )

    def _insert_capsule(
        self,
        projection: CapsuleProjection,
        *,
        federation_id: str,
        tenant_id: str,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_semantic_capsule_reference",
            {
                "semantic_capsule_reference_id": projection.record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "repository_id": projection.repository_id,
                "tree_id": projection.tree_id,
                "subject_kind": projection.subject_kind,
                "subject_ref": projection.subject_ref,
                "dependency_root": projection.dependency_root,
                "owner_id": SEMANTIC_OWNER,
                "source_root": projection.tree_id,
                "content_ref": projection.content_ref,
                "revision": projection.revision,
                "status": "current",
                "freshness_state": "current",
                "provenance_ref": SEMANTIC_OWNER,
                "recorded_at": recorded_at,
            },
        )

    def _apply_invalidations(
        self,
        affected: Sequence[str],
        *,
        federation_id: str,
        tenant_id: str,
        event_id: str,
        graph_revision: int,
        recorded_at: str,
        reason_kind: str,
        evidence_ref: str,
    ) -> None:
        for capsule_id in affected:
            self._client.execute(
                "casf_mark_capsule_invalidated",
                {
                    "semantic_capsule_reference_id": capsule_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                },
            )
            self._client.execute(
                "casf_insert_causal_invalidation",
                {
                    "causal_invalidation_id": "invalidation:"
                    + content_identity({"capsule": capsule_id, "event": event_id}),
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "event_id": event_id,
                    "graph_revision": graph_revision,
                    "subject_kind": "capsule",
                    "subject_ref": capsule_id,
                    "reason_kind": reason_kind,
                    "evidence_ref": evidence_ref,
                    "created_at": recorded_at,
                },
            )


__all__ = (
    "CapsuleProjection",
    "SEMANTIC_OWNER",
    "SemanticProjectionAuthorityError",
    "SemanticProjectionError",
    "SemanticProjectionStore",
    "SemanticRootProjection",
    "bind_capsule",
    "bind_datasets_capsule_ref",
    "bind_semantic_root",
    "capsules_invalidated_by_change",
    "federation_may_reinterpret_semantics",
)
