"""Tree-bound proof, test, cache, and seal projections.

These records are opaque references to existing proof/test/seal identities.
They never establish completion, policy permission, or scheduling authority.
Incremental tree updates invalidate exactly the affected proof and cache rows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
    _timestamp,
)
from .registry import _template
from .semantic_projection import SemanticProjectionStore

PROOF_KINDS = frozenset({"obligation", "unit", "receipt", "cache", "seal"})
PROOF_STATUSES = frozenset({"open", "proved", "failed", "stale", "invalidated"})
TEST_KINDS = frozenset({"unit", "integration", "property", "selection"})
TEST_STATUSES = frozenset({"pending", "passed", "failed", "stale", "invalidated"})
CACHE_STATES = frozenset({"current", "stale", "invalidated", "expired"})
SEAL_STATES = frozenset({"sealed", "stale", "invalidated"})
INVALIDATION_REASONS = frozenset(
    {"tree_change", "obligation_change", "test_change", "cache_stale", "seal_stale"}
)


class ProofProjectionError(CausalGraphError):
    """Base typed proof-projection failure."""


class ProofProjectionAuthorityError(FederationAuthorityError, ProofProjectionError):
    """An attempt to mint completion or ignore tree identity via a projection."""


def _reject_sibling_path(value: str, name: str) -> None:
    if value.startswith(("/", "~")) or ".." in value.split("/"):
        raise ProofProjectionAuthorityError(
            f"{name} is a sibling filesystem path, not an opaque proof identity"
        )


@dataclass(frozen=True)
class ProofProjection:
    """Opaque proof-obligation or receipt reference."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/proof-projection@1"
    )

    record_id: str
    proof_kind: str
    obligation_ref: str
    proof_status: str
    content_ref: str
    repository_id: str
    tree_id: str
    task_cid: str = ""
    revision: int = 1

    def __post_init__(self) -> None:
        _reject_sibling_path(str(self.obligation_ref), "obligation_ref")
        _reject_sibling_path(str(self.content_ref), "content_ref")
        _identifier(self.record_id, "record_id")
        kind = _identifier(self.proof_kind, "proof_kind")
        if kind not in PROOF_KINDS:
            raise FederationContractError("proof_kind is not closed")
        object.__setattr__(self, "proof_kind", kind)
        status = _identifier(self.proof_status, "proof_status")
        if status not in PROOF_STATUSES:
            raise FederationContractError("proof_status is not closed")
        object.__setattr__(self, "proof_status", status)
        _identifier(self.obligation_ref, "obligation_ref")
        _identifier(self.content_ref, "content_ref")
        _identifier(self.repository_id, "repository_id")
        _identifier(self.tree_id, "tree_id")
        _identifier(self.task_cid, "task_cid", required=False)
        if self.revision < 1:
            raise FederationContractError("revision must be >= 1")


@dataclass(frozen=True)
class TestProjection:
    """Opaque test identity bound to one tree."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/test-projection@1"
    )

    record_id: str
    test_kind: str
    test_ref: str
    test_status: str
    content_ref: str
    repository_id: str
    tree_id: str
    task_cid: str = ""
    revision: int = 1

    def __post_init__(self) -> None:
        _reject_sibling_path(str(self.test_ref), "test_ref")
        _identifier(self.record_id, "record_id")
        kind = _identifier(self.test_kind, "test_kind")
        if kind not in TEST_KINDS:
            raise FederationContractError("test_kind is not closed")
        object.__setattr__(self, "test_kind", kind)
        status = _identifier(self.test_status, "test_status")
        if status not in TEST_STATUSES:
            raise FederationContractError("test_status is not closed")
        object.__setattr__(self, "test_status", status)
        _identifier(self.test_ref, "test_ref")
        _identifier(self.content_ref, "content_ref")
        _identifier(self.repository_id, "repository_id")
        _identifier(self.tree_id, "tree_id")
        _identifier(self.task_cid, "task_cid", required=False)
        if self.revision < 1:
            raise FederationContractError("revision must be >= 1")


@dataclass(frozen=True)
class CacheProjection:
    """Opaque proof-cache entry; reuse still requires exact tree and policy."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/proof-cache-projection@1"
    )

    record_id: str
    obligation_ref: str
    dependency_root: str
    policy_ref: str
    provider_model_ref: str
    content_ref: str
    repository_id: str
    tree_id: str
    expires_at: str
    revision: int = 1

    def __post_init__(self) -> None:
        _identifier(self.record_id, "record_id")
        for name in (
            "obligation_ref",
            "dependency_root",
            "policy_ref",
            "provider_model_ref",
            "content_ref",
            "repository_id",
            "tree_id",
        ):
            _identifier(getattr(self, name), name)
        _timestamp(self.expires_at, "expires_at")
        if self.revision < 1:
            raise FederationContractError("revision must be >= 1")


@dataclass(frozen=True)
class SealProjection:
    """Opaque proof-seal reference; it does not complete a federation task."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/proof-seal-projection@1"
    )

    record_id: str
    proof_unit_id: str
    proof_receipt_id: str
    policy_ref: str
    content_ref: str
    repository_id: str
    tree_id: str
    revision: int = 1

    def __post_init__(self) -> None:
        _identifier(self.record_id, "record_id")
        for name in (
            "proof_unit_id",
            "proof_receipt_id",
            "policy_ref",
            "content_ref",
            "repository_id",
            "tree_id",
        ):
            _identifier(getattr(self, name), name)
        if self.revision < 1:
            raise FederationContractError("revision must be >= 1")


def projection_establishes_completion() -> bool:
    """Proof/test/cache/seal projections never complete federation work."""

    return False


def projection_establishes_authority() -> bool:
    """Projections cannot mint policy, scheduling, or proof authority."""

    return False


def bind_proof(
    *,
    binding: FederationBinding,
    proof_kind: str,
    obligation_ref: str,
    proof_status: str,
    content_ref: str,
    record_id: str,
    task_cid: str = "",
    revision: int = 1,
    tree_id: str = "",
    repository_id: str = "",
) -> ProofProjection:
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    projection = ProofProjection(
        record_id=record_id,
        proof_kind=proof_kind,
        obligation_ref=obligation_ref,
        proof_status=proof_status,
        content_ref=content_ref,
        repository_id=repository_id or binding.repository_ids[0],
        tree_id=tree_id or binding.repository_tree_ids[0],
        task_cid=task_cid,
        revision=revision,
    )
    if projection.tree_id != binding.repository_tree_ids[0]:
        raise ProofProjectionAuthorityError("proof tree identity mismatches")
    if projection.repository_id != binding.repository_ids[0]:
        raise ProofProjectionAuthorityError("proof repository is not bound")
    return projection


def bind_test(
    *,
    binding: FederationBinding,
    test_kind: str,
    test_ref: str,
    test_status: str,
    content_ref: str,
    record_id: str,
    task_cid: str = "",
    revision: int = 1,
    tree_id: str = "",
    repository_id: str = "",
) -> TestProjection:
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    projection = TestProjection(
        record_id=record_id,
        test_kind=test_kind,
        test_ref=test_ref,
        test_status=test_status,
        content_ref=content_ref,
        repository_id=repository_id or binding.repository_ids[0],
        tree_id=tree_id or binding.repository_tree_ids[0],
        task_cid=task_cid,
        revision=revision,
    )
    if projection.tree_id != binding.repository_tree_ids[0]:
        raise ProofProjectionAuthorityError("test tree identity mismatches")
    return projection


def bind_cache(
    *,
    binding: FederationBinding,
    obligation_ref: str,
    dependency_root: str,
    policy_ref: str,
    provider_model_ref: str,
    content_ref: str,
    expires_at: str,
    record_id: str,
    revision: int = 1,
    tree_id: str = "",
) -> CacheProjection:
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    projection = CacheProjection(
        record_id=record_id,
        obligation_ref=obligation_ref,
        dependency_root=dependency_root,
        policy_ref=policy_ref,
        provider_model_ref=provider_model_ref,
        content_ref=content_ref,
        repository_id=binding.repository_ids[0],
        tree_id=tree_id or binding.repository_tree_ids[0],
        expires_at=expires_at,
        revision=revision,
    )
    if projection.tree_id != binding.repository_tree_ids[0]:
        raise ProofProjectionAuthorityError("cache tree identity mismatches")
    if (
        projection.dependency_root not in binding.semantic_state_roots
        and projection.dependency_root != binding.repository_tree_ids[0]
    ):
        raise ProofProjectionAuthorityError(
            "cache dependency root is not bound to this federation tree"
        )
    return projection


def bind_seal(
    *,
    binding: FederationBinding,
    proof_unit_id: str,
    proof_receipt_id: str,
    policy_ref: str,
    content_ref: str,
    record_id: str,
    revision: int = 1,
    tree_id: str = "",
) -> SealProjection:
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    projection = SealProjection(
        record_id=record_id,
        proof_unit_id=proof_unit_id,
        proof_receipt_id=proof_receipt_id,
        policy_ref=policy_ref,
        content_ref=content_ref,
        repository_id=binding.repository_ids[0],
        tree_id=tree_id or binding.repository_tree_ids[0],
        revision=revision,
    )
    if projection.tree_id != binding.repository_tree_ids[0]:
        raise ProofProjectionAuthorityError("seal tree identity mismatches")
    return projection


def proofs_invalidated_by_change(
    proofs: Sequence[ProofProjection],
    *,
    changed_obligation_refs: Sequence[str] = (),
    tree_id: str,
) -> tuple[str, ...]:
    """Return proof identities affected by an incremental tree change."""

    changed = {
        _identifier(item, "changed_obligation_refs") for item in changed_obligation_refs
    }
    tree = _identifier(tree_id, "tree_id")
    affected: list[str] = []
    for proof in proofs:
        if not isinstance(proof, ProofProjection):
            raise FederationContractError("proofs must be ProofProjection records")
        if proof.tree_id != tree:
            raise ProofProjectionAuthorityError(
                "proof tree identity mismatches the incremental update"
            )
        if proof.obligation_ref in changed:
            affected.append(proof.record_id)
    return tuple(dict.fromkeys(affected))


def caches_invalidated_by_change(
    caches: Sequence[CacheProjection],
    *,
    changed_obligation_refs: Sequence[str] = (),
    changed_dependency_root: str = "",
    tree_id: str,
) -> tuple[str, ...]:
    changed = {
        _identifier(item, "changed_obligation_refs") for item in changed_obligation_refs
    }
    root = _identifier(changed_dependency_root, "changed_dependency_root", required=False)
    tree = _identifier(tree_id, "tree_id")
    affected: list[str] = []
    for cache in caches:
        if not isinstance(cache, CacheProjection):
            raise FederationContractError("caches must be CacheProjection records")
        if cache.tree_id != tree:
            raise ProofProjectionAuthorityError(
                "cache tree identity mismatches the incremental update"
            )
        if cache.obligation_ref in changed or (root and cache.dependency_root == root):
            affected.append(cache.record_id)
    return tuple(dict.fromkeys(affected))


def _proof_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_insert_proof_reference",
            """
            INSERT INTO proof_reference_projections (
                proof_reference_id, tenant_id, federation_id, repository_id,
                tree_id, task_cid, obligation_ref, proof_kind, proof_status,
                owner_id, source_root, content_ref, revision, status,
                freshness_state, provenance_ref, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "proof_reference_id",
                "tenant_id",
                "federation_id",
                "repository_id",
                "tree_id",
                "task_cid",
                "obligation_ref",
                "proof_kind",
                "proof_status",
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
            "casf_select_proof_reference",
            """
            SELECT proof_reference_id, obligation_ref, proof_kind, proof_status,
                   tree_id, content_ref, freshness_state
            FROM proof_reference_projections
            WHERE proof_reference_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("proof_reference_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_test_reference",
            """
            INSERT INTO test_reference_projections (
                test_reference_id, tenant_id, federation_id, repository_id,
                tree_id, task_cid, test_ref, test_kind, test_status,
                owner_id, source_root, content_ref, revision, status,
                freshness_state, provenance_ref, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "test_reference_id",
                "tenant_id",
                "federation_id",
                "repository_id",
                "tree_id",
                "task_cid",
                "test_ref",
                "test_kind",
                "test_status",
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
            "casf_select_test_reference",
            """
            SELECT test_reference_id, test_ref, test_kind, test_status,
                   tree_id, content_ref, freshness_state
            FROM test_reference_projections
            WHERE test_reference_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("test_reference_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_proof_cache_entry",
            """
            INSERT INTO proof_cache_entries (
                proof_cache_entry_id, tenant_id, repository_id, tree_id,
                obligation_ref, dependency_root, policy_ref, provider_model_ref,
                owner_id, source_root, provenance_ref, content_ref, revision,
                status, freshness_state, recorded_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "proof_cache_entry_id",
                "tenant_id",
                "repository_id",
                "tree_id",
                "obligation_ref",
                "dependency_root",
                "policy_ref",
                "provider_model_ref",
                "owner_id",
                "source_root",
                "provenance_ref",
                "content_ref",
                "revision",
                "status",
                "freshness_state",
                "recorded_at",
                "expires_at",
            ),
        ),
        _template(
            "casf_select_proof_cache_entry",
            """
            SELECT proof_cache_entry_id, obligation_ref, dependency_root,
                   tree_id, content_ref, freshness_state, expires_at
            FROM proof_cache_entries
            WHERE proof_cache_entry_id = ? AND tenant_id = ?
            LIMIT 1
            """,
            ("proof_cache_entry_id", "tenant_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_proof_seal",
            """
            INSERT INTO proof_seals (
                proof_seal_id, tenant_id, federation_id, proof_unit_id,
                proof_receipt_id, repository_id, tree_id, policy_ref,
                owner_id, source_root, provenance_ref, content_ref, revision,
                status, freshness_state, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "proof_seal_id",
                "tenant_id",
                "federation_id",
                "proof_unit_id",
                "proof_receipt_id",
                "repository_id",
                "tree_id",
                "policy_ref",
                "owner_id",
                "source_root",
                "provenance_ref",
                "content_ref",
                "revision",
                "status",
                "freshness_state",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_proof_seal",
            """
            SELECT proof_seal_id, proof_unit_id, proof_receipt_id, tree_id,
                   content_ref, freshness_state, status
            FROM proof_seals
            WHERE proof_seal_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("proof_seal_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_mark_proof_invalidated",
            """
            UPDATE proof_reference_projections
            SET freshness_state = 'invalidated', status = 'invalidated',
                proof_status = 'invalidated'
            WHERE proof_reference_id = ? AND tenant_id = ? AND federation_id = ?
              AND freshness_state = 'current'
            RETURNING proof_reference_id
            """,
            ("proof_reference_id", "tenant_id", "federation_id"),
        ),
        _template(
            "casf_mark_proof_cache_invalidated",
            """
            UPDATE proof_cache_entries
            SET freshness_state = 'invalidated', status = 'invalidated'
            WHERE proof_cache_entry_id = ? AND tenant_id = ?
              AND freshness_state = 'current'
            RETURNING proof_cache_entry_id
            """,
            ("proof_cache_entry_id", "tenant_id"),
        ),
    )


class ProofProjectionStore(SemanticProjectionStore):
    """Persist opaque proof/test/cache/seal projections."""

    INTERFACE = "ProofProjectionStore@1"

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
            raise ProofProjectionError("proof projection store never accepts a database path")
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise ProofProjectionError(
                "proof projection store requires an already-attached typed state client"
            )
        registered = set(client.list_templates())
        missing = [
            template.name
            for template in _proof_templates()
            if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise ProofProjectionError(
                    "proof projection templates are absent from the sealed catalog"
                )
        else:
            for template in _proof_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    def record_proof(
        self,
        projection: ProofProjection,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        event_id: str,
    ) -> CausalGraphCommit:
        bound = bind_proof(
            binding=binding,
            proof_kind=projection.proof_kind,
            obligation_ref=projection.obligation_ref,
            proof_status=projection.proof_status,
            content_ref=projection.content_ref,
            record_id=projection.record_id,
            task_cid=projection.task_cid,
            revision=projection.revision,
            tree_id=projection.tree_id,
            repository_id=projection.repository_id,
        )
        return self._commit_fact(
            operation="federation.proof.reference.record",
            fact_id=bound.record_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(
                dict.fromkeys((bound.record_id, bound.obligation_ref, event_id))
            ),
            payload_ref=bound.content_ref,
            prepare_fact=lambda: self._prepare_proof(
                bound.record_id,
                tenant_id=binding.tenant_id,
                federation_id=federation_id,
            ),
            apply_fact=lambda revision, recorded_at: self._insert_proof(
                bound,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                recorded_at=recorded_at,
            ),
        )

    def record_test(
        self,
        projection: TestProjection,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        event_id: str,
    ) -> CausalGraphCommit:
        bound = bind_test(
            binding=binding,
            test_kind=projection.test_kind,
            test_ref=projection.test_ref,
            test_status=projection.test_status,
            content_ref=projection.content_ref,
            record_id=projection.record_id,
            task_cid=projection.task_cid,
            revision=projection.revision,
            tree_id=projection.tree_id,
        )
        return self._commit_fact(
            operation="federation.test.reference.record",
            fact_id=bound.record_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(
                dict.fromkeys((bound.record_id, bound.test_ref, event_id))
            ),
            payload_ref=bound.content_ref,
            prepare_fact=lambda: self._prepare_test(
                bound.record_id,
                tenant_id=binding.tenant_id,
                federation_id=federation_id,
            ),
            apply_fact=lambda revision, recorded_at: self._insert_test(
                bound,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                recorded_at=recorded_at,
            ),
        )

    def record_cache(
        self,
        projection: CacheProjection,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        event_id: str,
    ) -> CausalGraphCommit:
        bound = bind_cache(
            binding=binding,
            obligation_ref=projection.obligation_ref,
            dependency_root=projection.dependency_root,
            policy_ref=projection.policy_ref,
            provider_model_ref=projection.provider_model_ref,
            content_ref=projection.content_ref,
            expires_at=projection.expires_at,
            record_id=projection.record_id,
            revision=projection.revision,
            tree_id=projection.tree_id,
        )
        return self._commit_fact(
            operation="federation.proof.cache.record",
            fact_id=bound.record_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(
                dict.fromkeys((bound.record_id, bound.obligation_ref, event_id))
            ),
            payload_ref=bound.content_ref,
            prepare_fact=lambda: None,
            apply_fact=lambda revision, recorded_at: self._insert_cache(
                bound,
                tenant_id=binding.tenant_id,
                recorded_at=recorded_at,
            ),
        )

    def record_seal(
        self,
        projection: SealProjection,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        event_id: str,
    ) -> CausalGraphCommit:
        bound = bind_seal(
            binding=binding,
            proof_unit_id=projection.proof_unit_id,
            proof_receipt_id=projection.proof_receipt_id,
            policy_ref=projection.policy_ref,
            content_ref=projection.content_ref,
            record_id=projection.record_id,
            revision=projection.revision,
            tree_id=projection.tree_id,
        )
        return self._commit_fact(
            operation="federation.proof.seal.record",
            fact_id=bound.record_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(
                dict.fromkeys((bound.record_id, bound.proof_receipt_id, event_id))
            ),
            payload_ref=bound.content_ref,
            prepare_fact=lambda: None,
            apply_fact=lambda revision, recorded_at: self._insert_seal(
                bound,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                recorded_at=recorded_at,
            ),
        )

    def invalidate_proofs(
        self,
        proofs: Sequence[ProofProjection],
        caches: Sequence[CacheProjection] = (),
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
        event_id: str,
        changed_obligation_refs: Sequence[str] = (),
        changed_dependency_root: str = "",
        reason_kind: str = "obligation_change",
    ) -> CausalGraphCommit:
        if reason_kind not in INVALIDATION_REASONS:
            raise FederationContractError("invalidation reason is not closed")
        affected_proofs = proofs_invalidated_by_change(
            proofs,
            changed_obligation_refs=changed_obligation_refs,
            tree_id=binding.repository_tree_ids[0],
        )
        affected_caches = caches_invalidated_by_change(
            caches,
            changed_obligation_refs=changed_obligation_refs,
            changed_dependency_root=changed_dependency_root,
            tree_id=binding.repository_tree_ids[0],
        )
        evidence = content_identity(
            {
                "event_id": event_id,
                "proofs": list(affected_proofs),
                "caches": list(affected_caches),
                "reason_kind": reason_kind,
            }
        )
        changed = tuple(dict.fromkeys((event_id, *affected_proofs, *affected_caches)))
        return self._commit_fact(
            operation="federation.proof.invalidate",
            fact_id="invalidation:" + evidence,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=changed,
            payload_ref=evidence,
            prepare_fact=lambda: None,
            apply_fact=lambda revision, recorded_at: self._apply_proof_invalidations(
                affected_proofs,
                affected_caches,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                recorded_at=recorded_at,
            ),
        )

    def load_proof(self, *, record_id: str, tenant_id: str, federation_id: str) -> dict[str, Any]:
        rows = self._client.execute(
            "casf_select_proof_reference",
            {
                "proof_reference_id": record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if len(rows) != 1:
            raise ProofProjectionError("proof projection is absent")
        return dict(rows[0])

    def load_test(self, *, record_id: str, tenant_id: str, federation_id: str) -> dict[str, Any]:
        rows = self._client.execute(
            "casf_select_test_reference",
            {
                "test_reference_id": record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if len(rows) != 1:
            raise ProofProjectionError("test projection is absent")
        return dict(rows[0])

    def load_cache(self, *, record_id: str, tenant_id: str) -> dict[str, Any]:
        rows = self._client.execute(
            "casf_select_proof_cache_entry",
            {"proof_cache_entry_id": record_id, "tenant_id": tenant_id},
        )
        if len(rows) != 1:
            raise ProofProjectionError("proof cache projection is absent")
        return dict(rows[0])

    def load_seal(self, *, record_id: str, tenant_id: str, federation_id: str) -> dict[str, Any]:
        rows = self._client.execute(
            "casf_select_proof_seal",
            {
                "proof_seal_id": record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if len(rows) != 1:
            raise ProofProjectionError("proof seal projection is absent")
        return dict(rows[0])

    def _prepare_proof(self, record_id: str, *, tenant_id: str, federation_id: str) -> None:
        existing = self._client.execute(
            "casf_select_proof_reference",
            {
                "proof_reference_id": record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if existing:
            raise ProofProjectionError("proof projection is already bound")

    def _prepare_test(self, record_id: str, *, tenant_id: str, federation_id: str) -> None:
        existing = self._client.execute(
            "casf_select_test_reference",
            {
                "test_reference_id": record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if existing:
            raise ProofProjectionError("test projection is already bound")

    def _insert_proof(
        self,
        projection: ProofProjection,
        *,
        federation_id: str,
        tenant_id: str,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_proof_reference",
            {
                "proof_reference_id": projection.record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "repository_id": projection.repository_id,
                "tree_id": projection.tree_id,
                "task_cid": projection.task_cid,
                "obligation_ref": projection.obligation_ref,
                "proof_kind": projection.proof_kind,
                "proof_status": projection.proof_status,
                "owner_id": "proof-projection",
                "source_root": projection.tree_id,
                "content_ref": projection.content_ref,
                "revision": projection.revision,
                "status": "current",
                "freshness_state": "current",
                "provenance_ref": projection.content_ref,
                "recorded_at": recorded_at,
            },
        )

    def _insert_test(
        self,
        projection: TestProjection,
        *,
        federation_id: str,
        tenant_id: str,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_test_reference",
            {
                "test_reference_id": projection.record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "repository_id": projection.repository_id,
                "tree_id": projection.tree_id,
                "task_cid": projection.task_cid,
                "test_ref": projection.test_ref,
                "test_kind": projection.test_kind,
                "test_status": projection.test_status,
                "owner_id": "proof-projection",
                "source_root": projection.tree_id,
                "content_ref": projection.content_ref,
                "revision": projection.revision,
                "status": "current",
                "freshness_state": "current",
                "provenance_ref": projection.content_ref,
                "recorded_at": recorded_at,
            },
        )

    def _insert_cache(
        self,
        projection: CacheProjection,
        *,
        tenant_id: str,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_proof_cache_entry",
            {
                "proof_cache_entry_id": projection.record_id,
                "tenant_id": tenant_id,
                "repository_id": projection.repository_id,
                "tree_id": projection.tree_id,
                "obligation_ref": projection.obligation_ref,
                "dependency_root": projection.dependency_root,
                "policy_ref": projection.policy_ref,
                "provider_model_ref": projection.provider_model_ref,
                "owner_id": "proof-projection",
                "source_root": projection.tree_id,
                "provenance_ref": projection.content_ref,
                "content_ref": projection.content_ref,
                "revision": projection.revision,
                "status": "current",
                "freshness_state": "current",
                "recorded_at": recorded_at,
                "expires_at": projection.expires_at,
            },
        )

    def _insert_seal(
        self,
        projection: SealProjection,
        *,
        federation_id: str,
        tenant_id: str,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_proof_seal",
            {
                "proof_seal_id": projection.record_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "proof_unit_id": projection.proof_unit_id,
                "proof_receipt_id": projection.proof_receipt_id,
                "repository_id": projection.repository_id,
                "tree_id": projection.tree_id,
                "policy_ref": projection.policy_ref,
                "owner_id": "proof-projection",
                "source_root": projection.tree_id,
                "provenance_ref": projection.content_ref,
                "content_ref": projection.content_ref,
                "revision": projection.revision,
                "status": "sealed",
                "freshness_state": "current",
                "recorded_at": recorded_at,
            },
        )

    def _apply_proof_invalidations(
        self,
        proofs: Sequence[str],
        caches: Sequence[str],
        *,
        federation_id: str,
        tenant_id: str,
        recorded_at: str,
    ) -> None:
        del recorded_at
        for proof_id in proofs:
            self._client.execute(
                "casf_mark_proof_invalidated",
                {
                    "proof_reference_id": proof_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                },
            )
        for cache_id in caches:
            self._client.execute(
                "casf_mark_proof_cache_invalidated",
                {"proof_cache_entry_id": cache_id, "tenant_id": tenant_id},
            )


__all__ = (
    "CacheProjection",
    "ProofProjection",
    "ProofProjectionAuthorityError",
    "ProofProjectionError",
    "ProofProjectionStore",
    "SealProjection",
    "TestProjection",
    "bind_cache",
    "bind_proof",
    "bind_seal",
    "bind_test",
    "caches_invalidated_by_change",
    "projection_establishes_authority",
    "projection_establishes_completion",
    "proofs_invalidated_by_change",
)
