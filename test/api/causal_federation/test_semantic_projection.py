"""Hermetic tests for CASF AST/symbol/capsule semantic projections."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.semantic_projection import (
    SEMANTIC_OWNER,
    SemanticProjectionAuthorityError,
    SemanticProjectionError,
    SemanticProjectionStore,
    bind_capsule,
    bind_datasets_capsule_ref,
    bind_semantic_root,
    capsules_invalidated_by_change,
    federation_may_reinterpret_semantics,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import SemanticCapsuleRef
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    open_embedded_client,
)
from test.api.causal_federation.test_contracts import sample_binding
from test.api.causal_federation.test_registry import _create
from test.api.causal_federation.test_trigger import sample_policy, sample_request

_CID = "b" + "a" * 51
_ROOT_CID = "b" + "c" * 51
_VERSION_CID = "b" + "d" * 51
_SOURCE_CID = "b" + "e" * 51


def test_federation_never_reinterprets_datasets_semantics() -> None:
    assert federation_may_reinterpret_semantics() is False
    assert SEMANTIC_OWNER == "ipfs_datasets_py"


def test_tree_mismatch_fails_closed() -> None:
    binding = sample_binding()
    with pytest.raises(SemanticProjectionAuthorityError, match="tree identity mismatches"):
        bind_capsule(
            binding=binding,
            subject_kind="symbol",
            subject_ref="symbol:dispatch",
            dependency_root=binding.semantic_state_roots[0],
            content_ref="capsule:content",
            record_id="capsule:one",
            tree_id="tree:other",
        )
    with pytest.raises(SemanticProjectionAuthorityError, match="tree-bound"):
        bind_semantic_root(
            binding=binding,
            semantic_kind="semantic_state",
            semantic_root="semantic:unbound",
            content_ref="semantic:content",
            record_id="root:unbound",
        )


def test_sibling_paths_are_rejected() -> None:
    binding = sample_binding()
    with pytest.raises(SemanticProjectionAuthorityError, match="filesystem path"):
        bind_capsule(
            binding=binding,
            subject_kind="file",
            subject_ref="file/../secret",
            dependency_root=binding.semantic_state_roots[0],
            content_ref="capsule:content",
            record_id="capsule:path",
        )


def test_incremental_symbol_change_invalidates_only_affected_capsules() -> None:
    binding = sample_binding()
    changed = bind_capsule(
        binding=binding,
        subject_kind="symbol",
        subject_ref="symbol:dispatch",
        dependency_root=binding.semantic_state_roots[0],
        content_ref="capsule:dispatch",
        record_id="capsule:dispatch",
    )
    dependent = bind_capsule(
        binding=binding,
        subject_kind="capsule",
        subject_ref="capsule:caller",
        dependency_root=binding.semantic_state_roots[0],
        content_ref="capsule:caller",
        record_id="capsule:caller",
    )
    with pytest.raises(SemanticProjectionAuthorityError, match="dependency root"):
        bind_capsule(
            binding=binding,
            subject_kind="symbol",
            subject_ref="symbol:unrelated",
            dependency_root="semantic:other-root",
            content_ref="capsule:unrelated",
            record_id="capsule:unrelated",
        )
    other = bind_capsule(
        binding=binding,
        subject_kind="symbol",
        subject_ref="symbol:other",
        dependency_root=binding.semantic_state_roots[0],
        content_ref="capsule:other",
        record_id="capsule:other",
    )
    affected = capsules_invalidated_by_change(
        (changed, dependent, other),
        changed_subject_refs=("symbol:dispatch",),
        changed_semantic_root="",
        tree_id=binding.repository_tree_ids[0],
    )
    assert affected == ("capsule:dispatch",)
    dependents = capsules_invalidated_by_change(
        (changed, dependent, other),
        changed_subject_refs=(),
        changed_semantic_root=binding.semantic_state_roots[0],
        tree_id=binding.repository_tree_ids[0],
    )
    assert set(dependents) == {"capsule:dispatch", "capsule:caller", "capsule:other"}


def test_datasets_capsule_ref_projects_opaquely() -> None:
    binding = sample_binding(semantic_state_roots=(_ROOT_CID,))
    capsule = SemanticCapsuleRef(
        capsule_cid=_CID,
        semantic_state_root_cid=_ROOT_CID,
        stable_symbol_id="symbol:dispatch",
        version_cid=_VERSION_CID,
        source_cid=_SOURCE_CID,
        confidence="exact",
        validity_bindings=(_SOURCE_CID,),
        raw_source_required=False,
    )
    projected = bind_datasets_capsule_ref(
        capsule, binding=binding, record_id="capsule:datasets"
    )
    assert projected.content_ref == _CID
    assert projected.subject_ref == "symbol:dispatch"
    assert projected.dependency_root == _ROOT_CID
    with pytest.raises(SemanticProjectionAuthorityError, match="not tree-bound"):
        bind_datasets_capsule_ref(
            capsule, binding=sample_binding(), record_id="capsule:unbound"
        )


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(SemanticProjectionError, match="database path"):
        SemanticProjectionStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for semantic projection")
def test_store_records_roots_and_invalidates_affected_capsules(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:semantic-projection")
    assert report.to_version == 3
    client = open_embedded_client(
        database,
        owner_id="owner:semantic-projection",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = SemanticProjectionStore(client)
    binding = sample_binding(
        control_plane_generation=generation.generation,
        supervisor_population=0,
        causal_graph_revision=1,
    )
    identity, _receipt = _create(
        store,
        request=sample_request(
            binding=binding, maximum_supervisors=2, maximum_subagents=2
        ),
        policy=sample_policy(
            binding,
            maximum_supervisors=2,
            maximum_subagents=2,
            maximum_concurrent_subagents=2,
        ),
    )
    root = bind_semantic_root(
        binding=binding,
        semantic_kind="ast",
        semantic_root="ast:tree-test",
        content_ref="ast:content",
        record_id="root:ast",
    )
    revision = store.record_semantic_root(
        root,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=store.graph_revision(
            tenant_id=binding.tenant_id, federation_id=identity.record_id
        ),
        idempotency_key="idempotency:ast-root",
        event_id="event:ast",
    ).graph_revision
    changed = bind_capsule(
        binding=binding,
        subject_kind="symbol",
        subject_ref="symbol:dispatch",
        dependency_root=binding.semantic_state_roots[0],
        content_ref="capsule:dispatch",
        record_id="capsule:dispatch",
    )
    other = bind_capsule(
        binding=binding,
        subject_kind="symbol",
        subject_ref="symbol:other",
        dependency_root=binding.semantic_state_roots[0],
        content_ref="capsule:other",
        record_id="capsule:other",
    )
    revision = store.record_capsule(
        changed,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:capsule-dispatch",
        event_id="event:capsule-dispatch",
    ).graph_revision
    revision = store.record_capsule(
        other,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:capsule-other",
        event_id="event:capsule-other",
    ).graph_revision
    store.invalidate_capsules(
        (changed, other),
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:invalidate",
        event_id="event:symbol-change",
        changed_subject_refs=("symbol:dispatch",),
        reason_kind="symbol_change",
    )
    loaded_changed = store.load_capsule(
        record_id="capsule:dispatch",
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    loaded_other = store.load_capsule(
        record_id="capsule:other",
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    loaded_root = store.load_semantic_root(
        record_id="root:ast",
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert loaded_changed["freshness_state"] == "invalidated"
    assert loaded_other["freshness_state"] == "current"
    assert loaded_root["semantic_kind"] == "ast"
    assert loaded_root["tree_id"] == binding.repository_tree_ids[0]
