"""Hermetic tests for CASF proof, test, cache, and seal projections."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.proof_projection import (
    ProofProjectionAuthorityError,
    ProofProjectionError,
    ProofProjectionStore,
    bind_cache,
    bind_proof,
    bind_seal,
    bind_test,
    caches_invalidated_by_change,
    projection_establishes_authority,
    projection_establishes_completion,
    proofs_invalidated_by_change,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    open_embedded_client,
)
from test.api.causal_federation.test_contracts import EXPIRY, sample_binding
from test.api.causal_federation.test_registry import _create
from test.api.causal_federation.test_trigger import sample_policy, sample_request


def test_projections_never_complete_or_mint_authority() -> None:
    assert projection_establishes_completion() is False
    assert projection_establishes_authority() is False


def test_tree_mismatch_fails_closed() -> None:
    binding = sample_binding()
    with pytest.raises(ProofProjectionAuthorityError, match="tree identity mismatches"):
        bind_proof(
            binding=binding,
            proof_kind="obligation",
            obligation_ref="obligation:one",
            proof_status="open",
            content_ref="proof:content",
            record_id="proof:one",
            tree_id="tree:other",
        )
    with pytest.raises(ProofProjectionAuthorityError, match="tree identity mismatches"):
        bind_test(
            binding=binding,
            test_kind="unit",
            test_ref="test:one",
            test_status="pending",
            content_ref="test:content",
            record_id="test:one",
            tree_id="tree:other",
        )
    with pytest.raises(ProofProjectionAuthorityError, match="tree identity mismatches"):
        bind_seal(
            binding=binding,
            proof_unit_id="unit:one",
            proof_receipt_id="receipt:one",
            policy_ref=binding.policy_ref,
            content_ref="seal:content",
            record_id="seal:one",
            tree_id="tree:other",
        )


def test_unbound_cache_dependency_root_fails_closed() -> None:
    binding = sample_binding()
    with pytest.raises(ProofProjectionAuthorityError, match="dependency root"):
        bind_cache(
            binding=binding,
            obligation_ref="obligation:one",
            dependency_root="semantic:unbound",
            policy_ref=binding.policy_ref,
            provider_model_ref="model:none",
            content_ref="cache:content",
            expires_at=EXPIRY,
            record_id="cache:one",
        )


def test_incremental_obligation_change_invalidates_only_affected_rows() -> None:
    binding = sample_binding()
    affected = bind_proof(
        binding=binding,
        proof_kind="obligation",
        obligation_ref="obligation:dispatch",
        proof_status="proved",
        content_ref="proof:dispatch",
        record_id="proof:dispatch",
    )
    other = bind_proof(
        binding=binding,
        proof_kind="obligation",
        obligation_ref="obligation:other",
        proof_status="open",
        content_ref="proof:other",
        record_id="proof:other",
    )
    cache_hit = bind_cache(
        binding=binding,
        obligation_ref="obligation:dispatch",
        dependency_root=binding.semantic_state_roots[0],
        policy_ref=binding.policy_ref,
        provider_model_ref="model:none",
        content_ref="cache:dispatch",
        expires_at=EXPIRY,
        record_id="cache:dispatch",
    )
    cache_other = bind_cache(
        binding=binding,
        obligation_ref="obligation:other",
        dependency_root=binding.semantic_state_roots[0],
        policy_ref=binding.policy_ref,
        provider_model_ref="model:none",
        content_ref="cache:other",
        expires_at=EXPIRY,
        record_id="cache:other",
    )
    assert proofs_invalidated_by_change(
        (affected, other),
        changed_obligation_refs=("obligation:dispatch",),
        tree_id=binding.repository_tree_ids[0],
    ) == ("proof:dispatch",)
    assert caches_invalidated_by_change(
        (cache_hit, cache_other),
        changed_obligation_refs=("obligation:dispatch",),
        tree_id=binding.repository_tree_ids[0],
    ) == ("cache:dispatch",)
    assert set(
        caches_invalidated_by_change(
            (cache_hit, cache_other),
            changed_dependency_root=binding.semantic_state_roots[0],
            tree_id=binding.repository_tree_ids[0],
        )
    ) == {"cache:dispatch", "cache:other"}


def test_sibling_paths_are_rejected() -> None:
    binding = sample_binding()
    with pytest.raises(ProofProjectionAuthorityError, match="filesystem path"):
        bind_proof(
            binding=binding,
            proof_kind="obligation",
            obligation_ref="file/../secret",
            proof_status="open",
            content_ref="proof:content",
            record_id="proof:path",
        )


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(ProofProjectionError, match="database path"):
        ProofProjectionStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for proof projection")
def test_store_records_and_invalidates_proof_and_cache_rows(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:proof-projection")
    assert report.to_version == 2
    client = open_embedded_client(
        database,
        owner_id="owner:proof-projection",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = ProofProjectionStore(client)
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
    proof = bind_proof(
        binding=binding,
        proof_kind="receipt",
        obligation_ref="obligation:dispatch",
        proof_status="proved",
        content_ref="proof:dispatch",
        record_id="proof:dispatch",
        task_cid="task:dispatch",
    )
    other = bind_proof(
        binding=binding,
        proof_kind="obligation",
        obligation_ref="obligation:other",
        proof_status="open",
        content_ref="proof:other",
        record_id="proof:other",
    )
    test_row = bind_test(
        binding=binding,
        test_kind="unit",
        test_ref="test:dispatch",
        test_status="passed",
        content_ref="test:dispatch",
        record_id="test:dispatch",
    )
    cache = bind_cache(
        binding=binding,
        obligation_ref="obligation:dispatch",
        dependency_root=binding.semantic_state_roots[0],
        policy_ref=binding.policy_ref,
        provider_model_ref="model:none",
        content_ref="cache:dispatch",
        expires_at=EXPIRY,
        record_id="cache:dispatch",
    )
    seal = bind_seal(
        binding=binding,
        proof_unit_id="unit:dispatch",
        proof_receipt_id="receipt:dispatch",
        policy_ref=binding.policy_ref,
        content_ref="seal:dispatch",
        record_id="seal:dispatch",
    )
    revision = store.graph_revision(
        tenant_id=binding.tenant_id, federation_id=identity.record_id
    )
    for item, method, key in (
        (proof, store.record_proof, "proof"),
        (other, store.record_proof, "other"),
        (test_row, store.record_test, "test"),
        (cache, store.record_cache, "cache"),
        (seal, store.record_seal, "seal"),
    ):
        revision = method(
            item,
            federation_id=identity.record_id,
            binding=binding,
            expected_graph_revision=revision,
            idempotency_key=f"idempotency:{key}",
            event_id=f"event:{key}",
        ).graph_revision
    store.invalidate_proofs(
        (proof, other),
        (cache,),
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:invalidate",
        event_id="event:obligation-change",
        changed_obligation_refs=("obligation:dispatch",),
        reason_kind="obligation_change",
    )
    loaded_proof = store.load_proof(
        record_id="proof:dispatch",
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    loaded_other = store.load_proof(
        record_id="proof:other",
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    loaded_cache = store.load_cache(
        record_id="cache:dispatch", tenant_id=binding.tenant_id
    )
    loaded_test = store.load_test(
        record_id="test:dispatch",
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    loaded_seal = store.load_seal(
        record_id="seal:dispatch",
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert loaded_proof["freshness_state"] == "invalidated"
    assert loaded_other["freshness_state"] == "current"
    assert loaded_cache["freshness_state"] == "invalidated"
    assert loaded_test["test_status"] == "passed"
    assert loaded_seal["status"] == "sealed"
    assert projection_establishes_completion() is False
