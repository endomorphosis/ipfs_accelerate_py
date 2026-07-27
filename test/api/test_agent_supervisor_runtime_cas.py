from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime_cas import (
    DEPENDENCY_CAS_REQUIREMENT_ID,
    ArtifactDependency,
    ArtifactIntegrityError,
    AuthorityIsolationError,
    CanonicalArtifactIdentity,
    DirectorySharedImmutableStore,
    ForgedDependencyError,
    ImmutableStoreError,
    RuntimeArtifactRecord,
    RuntimeAuthority,
    RuntimeCAS,
    RuntimeTier,
    artifact_key,
)
from ipfs_accelerate_py.agent_supervisor.supervisor_v2_contracts import (
    EvidenceFreshness,
    ResultBinding,
    SemanticDependencyIdentity,
)


TREE = "tree:sha256:current"


def _dependency(
    *,
    namespace: str = "repository",
    key: str = "source-tree",
    revision: str = TREE,
) -> SemanticDependencyIdentity:
    digest = hashlib.sha256(revision.encode("utf-8")).hexdigest()
    return SemanticDependencyIdentity(
        namespace=namespace,
        key=key,
        revision=revision,
        digest=f"sha256:{digest}",
    )


def _binding(
    *,
    tree_id: str = TREE,
    task_id: str = "ASI-100",
    producer_revision: str = "producer:runtime-cas@1",
    policy_revision: str = "policy:runtime-cas@1",
    capability_revision: str = "capability:runtime-cas@1",
    semantic_dependencies: tuple[SemanticDependencyIdentity, ...] | None = None,
) -> ResultBinding:
    return ResultBinding(
        repository_id="repository:fixture",
        tree_id=tree_id,
        objective_id="ASI-G250",
        objective_revision="objective:cache-v2@1",
        task_id=task_id,
        task_revision=f"task:{task_id}@1",
        policy_id="policy:runtime-cas",
        policy_revision=policy_revision,
        producer_id="producer:runtime-cas",
        producer_revision=producer_revision,
        capability_id="capability:runtime-cas",
        capability_revision=capability_revision,
        environment_id="environment:pytest",
        environment_revision="environment:pytest@1",
        semantic_dependencies=semantic_dependencies or (_dependency(),),
    )


def _object_path(root: Path, artifact_id: str) -> Path:
    digest = artifact_id.rsplit(":", 1)[1]
    return root / "objects" / digest[:2] / f"{digest}.json"


def test_canonical_identity_is_order_independent_and_binds_all_versions(
    tmp_path: Path,
) -> None:
    store = RuntimeCAS(tmp_path, clock=lambda: 1_000.0)
    binding = _binding()

    first = store.put(
        {"z": [3, 2, 1], "a": {"second": 2, "first": 1}},
        binding=binding,
        namespace="analysis",
        artifact_kind="analysis_receipt",
        authority=RuntimeAuthority.AUTHORITATIVE,
        payload_schema="fixture.analysis@1",
    )
    reordered = store.put(
        {"a": {"first": 1, "second": 2}, "z": [3, 2, 1]},
        binding=binding,
        namespace="analysis",
        artifact_kind="analysis_receipt",
        authority=RuntimeAuthority.AUTHORITATIVE,
        payload_schema="fixture.analysis@1",
    )

    assert reordered.artifact_id == first.artifact_id
    assert reordered.key.key_id == first.key.key_id
    assert (
        RuntimeArtifactRecord.from_dict(first.to_dict()).artifact_id
        == first.artifact_id
    )
    assert (
        CanonicalArtifactIdentity.from_dict(first.identity.to_dict())
        == first.identity
    )

    changed_payload = store.put(
        {"a": {"first": 1, "second": 2}, "z": [3, 2, 0]},
        binding=binding,
        namespace="analysis",
        artifact_kind="analysis_receipt",
        authority=RuntimeAuthority.AUTHORITATIVE,
        payload_schema="fixture.analysis@1",
    )
    changed_bindings = (
        _binding(producer_revision="producer:runtime-cas@2"),
        _binding(policy_revision="policy:runtime-cas@2"),
        _binding(capability_revision="capability:runtime-cas@2"),
        _binding(
            semantic_dependencies=(
                _dependency(revision="tree:sha256:changed"),
            )
        ),
    )
    versioned = tuple(
        store.put(
            first.payload,
            binding=item,
            namespace="analysis",
            artifact_kind="analysis_receipt",
            authority=RuntimeAuthority.AUTHORITATIVE,
            payload_schema="fixture.analysis@1",
        )
        for item in changed_bindings
    )
    assert len(
        {
            first.artifact_id,
            changed_payload.artifact_id,
            *(item.artifact_id for item in versioned),
        }
    ) == 6

    left = store.put(
        {"value": "left"},
        binding=_binding(task_id="ASI-100-left"),
        namespace="source",
        artifact_kind="source",
    )
    right = store.put(
        {"value": "right"},
        binding=_binding(task_id="ASI-100-right"),
        namespace="source",
        artifact_kind="source",
    )
    child = store.put(
        {"joined": True},
        binding=_binding(task_id="ASI-100-child"),
        namespace="planning",
        artifact_kind="plan",
        dependencies=(left, right),
    )
    reversed_child = store.put(
        {"joined": True},
        binding=_binding(task_id="ASI-100-child"),
        namespace="planning",
        artifact_kind="plan",
        dependencies=(right, left),
    )
    assert reversed_child.artifact_id == child.artifact_id
    assert child.identity.dependency_ids == tuple(
        sorted((left.artifact_id, right.artifact_id))
    )

    forged = first.identity.to_dict()
    forged["artifact_id"] = "runtime-artifact:sha256:" + "0" * 64
    with pytest.raises(ArtifactIntegrityError, match="identity mismatch"):
        CanonicalArtifactIdentity.from_dict(forged)


def test_process_host_shared_and_authoritative_projection_tiers(
    tmp_path: Path,
) -> None:
    shared = DirectorySharedImmutableStore(tmp_path / "shared")
    host_root = tmp_path / "host"
    store = RuntimeCAS(
        host_root,
        shared_store=shared,
        current_tree_id=TREE,
        clock=lambda: 1_000.0,
    )
    record = store.put(
        {"receipt": "current"},
        binding=_binding(),
        namespace="validation",
        artifact_kind="validation_receipt",
        authority=RuntimeAuthority.AUTHORITATIVE,
        tiers=(
            RuntimeTier.PROCESS_LOCAL,
            RuntimeTier.HOST_DURABLE,
            RuntimeTier.SHARED_IMMUTABLE,
            RuntimeTier.AUTHORITATIVE_PROJECTION,
        ),
        projection_key="current-validation",
    )

    process_hit = store.lookup(record.artifact_id)
    assert process_hit.hit
    assert process_hit.tier is RuntimeTier.PROCESS_LOCAL

    restarted = RuntimeCAS(
        host_root,
        shared_store=shared,
        current_tree_id=TREE,
        clock=lambda: 1_000.0,
    )
    host_hit = restarted.lookup(record.artifact_id)
    assert host_hit.hit
    assert host_hit.tier is RuntimeTier.HOST_DURABLE

    remote = RuntimeCAS(
        tmp_path / "other-host",
        shared_store=shared,
        current_tree_id=TREE,
        clock=lambda: 1_000.0,
    )
    shared_hit = remote.lookup(record.artifact_id)
    assert shared_hit.hit
    assert shared_hit.tier is RuntimeTier.SHARED_IMMUTABLE
    assert shared_hit.artifact == record

    projection_hit = restarted.lookup_projection(
        "current-validation",
        namespace="validation",
    )
    assert projection_hit.hit
    assert projection_hit.tier is RuntimeTier.AUTHORITATIVE_PROJECTION
    assert projection_hit.artifact == record

    with pytest.raises(ImmutableStoreError, match="cannot be overwritten"):
        shared.put(record.artifact_id, b'{"forged":true}')


def test_drafts_never_merge_with_authoritative_receipts(
    tmp_path: Path,
) -> None:
    store = RuntimeCAS(tmp_path, current_tree_id=TREE)
    binding = _binding()
    draft = store.put(
        {"result": "same bytes"},
        binding=binding,
        namespace="proof_draft",
        artifact_kind="proof",
        authority=RuntimeAuthority.DRAFT,
    )
    receipt = store.put(
        {"result": "same bytes"},
        binding=binding,
        namespace="proof",
        artifact_kind="proof",
        authority=RuntimeAuthority.AUTHORITATIVE,
    )

    assert draft.artifact_id != receipt.artifact_id
    assert draft.key.key_id != receipt.key.key_id
    assert not store.lookup(
        draft.artifact_id,
        expected_authority=RuntimeAuthority.AUTHORITATIVE,
    ).hit
    assert not store.lookup(
        receipt.artifact_id,
        expected_namespace="proof_draft",
    ).hit

    with pytest.raises(AuthorityIsolationError, match="depend on drafts"):
        store.put(
            {"receipt": "forged promotion"},
            binding=binding,
            namespace="proof",
            artifact_kind="proof",
            authority=RuntimeAuthority.AUTHORITATIVE,
            dependencies=(draft,),
        )
    with pytest.raises(
        AuthorityIsolationError, match="non-authoritative artifacts"
    ):
        store.project(
            "draft-proof",
            draft,
            namespace="proof_draft",
            tree_id=TREE,
        )


def test_authoritative_dependency_closure_cannot_launder_a_draft(
    tmp_path: Path,
) -> None:
    store = RuntimeCAS(tmp_path, current_tree_id=TREE)
    draft = store.put(
        {"candidate": "unverified"},
        binding=_binding(task_id="ASI-100-transitive-draft"),
        namespace="proof_draft",
        artifact_kind="proof_candidate",
        authority=RuntimeAuthority.DRAFT,
    )
    diagnostic = store.put(
        {"summary": "proposal-only"},
        binding=_binding(task_id="ASI-100-transitive-diagnostic"),
        namespace="analysis",
        artifact_kind="analysis",
        authority=RuntimeAuthority.DIAGNOSTIC,
        dependencies=(draft,),
    )

    with pytest.raises(
        AuthorityIsolationError, match="closure cannot contain drafts"
    ):
        store.put(
            {"receipt": "must-not-upgrade"},
            binding=_binding(task_id="ASI-100-transitive-receipt"),
            namespace="proof",
            artifact_kind="proof_receipt",
            authority=RuntimeAuthority.AUTHORITATIVE,
            dependencies=(diagnostic,),
        )


def test_forged_or_future_dependencies_are_rejected_and_corruption_repairs(
    tmp_path: Path,
) -> None:
    store = RuntimeCAS(tmp_path, clock=lambda: 1_000.0)
    dependency = store.put(
        {"source": 1},
        binding=_binding(task_id="ASI-100-source"),
        namespace="analysis",
        artifact_kind="source",
    )
    sibling = store.put(
        {"source": "unrelated"},
        binding=_binding(task_id="ASI-100-sibling"),
        namespace="analysis",
        artifact_kind="source",
    )

    forged = ArtifactDependency.from_artifact(dependency).to_dict()
    forged["payload_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ForgedDependencyError, match="missing or forged"):
        store.put(
            {"child": 1},
            binding=_binding(task_id="ASI-100-forged"),
            namespace="planning",
            artifact_kind="plan",
            dependencies=(forged,),
        )

    future = ArtifactDependency(
        artifact_id="runtime-artifact:sha256:" + "f" * 64,
        namespace="analysis",
        authority=RuntimeAuthority.DIAGNOSTIC,
        payload_digest="sha256:" + "e" * 64,
        binding_id=dependency.binding.binding_id,
    )
    with pytest.raises(ForgedDependencyError, match="missing or forged"):
        store.put(
            {"cycle": "future back-edge"},
            binding=_binding(task_id="ASI-100-future"),
            namespace="planning",
            artifact_kind="plan",
            dependencies=(future,),
        )
    # Requiring dependencies to pre-exist makes the immutable insertion graph
    # causal, so a future artifact cannot later close a dependency cycle.

    store.clear_process_cache()
    path = _object_path(tmp_path, dependency.artifact_id)
    path.write_text("{truncated", encoding="utf-8")
    corrupt = store.lookup(dependency.artifact_id)
    assert not corrupt.hit
    assert corrupt.reason_codes == ("artifact_miss",)
    assert not path.exists()
    assert store.metrics().corruption_recoveries >= 1

    repaired = store.put(
        {"source": 1},
        binding=_binding(task_id="ASI-100-source"),
        namespace="analysis",
        artifact_kind="source",
    )
    assert repaired.artifact_id == dependency.artifact_id
    assert store.lookup(repaired.key).artifact == repaired
    assert store.lookup(sibling.artifact_id).artifact == sibling


def test_authoritative_projections_require_fresh_current_tree_records(
    tmp_path: Path,
) -> None:
    now = 1_000.0
    store = RuntimeCAS(
        tmp_path,
        current_tree_id=TREE,
        clock=lambda: now,
    )
    current = store.put(
        {"validation": "passed"},
        binding=_binding(),
        namespace="validation",
        artifact_kind="validation_receipt",
        authority=RuntimeAuthority.AUTHORITATIVE,
        ttl_seconds=2,
        projection_key="validation/current",
    )
    assert (
        store.get_projection(
            "validation/current",
            namespace="validation",
        )
        == current
    )

    foreign_tree_store = RuntimeCAS(
        tmp_path,
        current_tree_id="tree:sha256:other",
        clock=lambda: now,
    )
    foreign = foreign_tree_store.lookup_projection(
        "validation/current",
        namespace="validation",
    )
    assert not foreign.hit
    assert foreign.reason_codes == ("projection_binding_mismatch",)

    with pytest.raises(AuthorityIsolationError, match="current tree"):
        store.project(
            "validation/foreign",
            current,
            namespace="validation",
            tree_id="tree:sha256:other",
        )

    now += 3
    stale = store.lookup_projection(
        "validation/current",
        namespace="validation",
    )
    assert not stale.hit
    assert stale.reason_codes == ("stale_or_forged_projection",)
    metrics = store.metrics()
    assert metrics.stale_rejections >= 1
    assert metrics.stale_authoritative_hits == 0

    explicitly_stale = store.put(
        {"validation": "old"},
        binding=_binding(task_id="ASI-100-stale"),
        namespace="validation",
        artifact_kind="validation_receipt",
        authority=RuntimeAuthority.AUTHORITATIVE,
        freshness=EvidenceFreshness.STALE,
    )
    assert not store.lookup(explicitly_stale.artifact_id).hit
    with pytest.raises(AuthorityIsolationError, match="fresh artifacts"):
        store.project(
            "validation/stale",
            explicitly_stale,
            namespace="validation",
        )


def test_expired_exact_result_can_be_recomputed_with_fresh_identity(
    tmp_path: Path,
) -> None:
    now = [1_000.0]
    store = RuntimeCAS(
        tmp_path,
        current_tree_id=TREE,
        clock=lambda: now[0],
    )
    key = artifact_key(
        namespace="analysis",
        artifact_kind="analysis_receipt",
        authority=RuntimeAuthority.AUTHORITATIVE,
        binding=_binding(task_id="ASI-100-expiry"),
        payload_schema="fixture.analysis@1",
    )
    calls = 0

    def produce() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"result": "semantically-identical"}

    first, produced = store.get_or_compute(
        key,
        produce,
        ttl_seconds=1,
    )
    assert produced
    warm, produced = store.get_or_compute(
        key,
        produce,
        ttl_seconds=1,
    )
    assert not produced
    assert warm.artifact_id == first.artifact_id

    now[0] += 2
    with pytest.raises(ForgedDependencyError, match="stale"):
        store.put(
            {"result": "must-not-use-stale-input"},
            binding=_binding(task_id="ASI-100-stale-dependent"),
            namespace="planning",
            artifact_kind="plan",
            authority=RuntimeAuthority.AUTHORITATIVE,
            dependencies=(first,),
        )
    replacement, produced = store.get_or_compute(
        key,
        produce,
        ttl_seconds=1,
    )

    assert produced
    assert calls == 2
    assert replacement.artifact_id != first.artifact_id
    assert not store.lookup(first.artifact_id).hit
    assert store.lookup(key).artifact == replacement
    assert replacement.identity.freshness is EvidenceFreshness.FRESH
    assert replacement.identity.created_at_ms == 1_002_000
    assert replacement.identity.expires_at_ms == 1_003_000


def test_exact_warm_reuse_and_semantic_change_invalidate_only_descendants(
    tmp_path: Path,
) -> None:
    store = RuntimeCAS(tmp_path, current_tree_id=TREE, clock=lambda: 1_000.0)
    changed_source = _dependency(
        namespace="repository",
        key="changed-source",
        revision="source@1",
    )
    unrelated_source = _dependency(
        namespace="repository",
        key="unrelated-source",
        revision="unrelated@1",
    )
    calls: list[str] = []

    def produce(label: str):
        def producer() -> dict[str, str]:
            calls.append(label)
            return {"result": label}

        return producer

    def key(
        label: str,
        semantic_dependency: SemanticDependencyIdentity,
        *dependencies,
    ):
        return artifact_key(
            namespace=label,
            artifact_kind=f"{label}_artifact",
            authority=RuntimeAuthority.AUTHORITATIVE,
            binding=_binding(
                task_id=f"ASI-100-{label}",
                semantic_dependencies=(semantic_dependency,),
            ),
            dependencies=dependencies,
            payload_schema=f"fixture.{label}@1",
        )

    analysis_key = key("analysis", changed_source)
    analysis, produced = store.get_or_compute(
        analysis_key, produce("analysis")
    )
    assert produced
    plan_key = key("planning", changed_source, analysis)
    plan, produced = store.get_or_compute(
        plan_key, produce("planning"), dependencies=(analysis,)
    )
    assert produced
    validation_key = key("validation", changed_source, plan)
    validation, produced = store.get_or_compute(
        validation_key,
        produce("validation"),
        dependencies=(plan,),
    )
    assert produced

    independent_key = key("independent", unrelated_source)
    independent, produced = store.get_or_compute(
        independent_key, produce("independent")
    )
    assert produced
    independent_child_key = key(
        "independent-child", unrelated_source, independent
    )
    independent_child, produced = store.get_or_compute(
        independent_child_key,
        produce("independent-child"),
        dependencies=(independent,),
    )
    assert produced
    assert calls == [
        "analysis",
        "planning",
        "validation",
        "independent",
        "independent-child",
    ]

    for exact_key, dependency_records in (
        (analysis_key, ()),
        (plan_key, (analysis,)),
        (validation_key, (plan,)),
        (independent_key, ()),
        (independent_child_key, (independent,)),
    ):
        warm, was_produced = store.get_or_compute(
            exact_key,
            lambda: pytest.fail("an exact warm hit invoked its producer"),
            dependencies=dependency_records,
        )
        assert not was_produced
        assert warm.key.key_id == exact_key.key_id
    assert len(calls) == 5

    replacement = _dependency(
        namespace=changed_source.namespace,
        key=changed_source.key,
        revision="source@2",
    )
    invalidation = store.invalidate_semantic_dependency(
        changed_source,
        replacement=replacement,
    )
    affected = {
        analysis.artifact_id,
        plan.artifact_id,
        validation.artifact_id,
    }
    preserved = {
        independent.artifact_id,
        independent_child.artifact_id,
    }
    assert invalidation.requirement_id == DEPENDENCY_CAS_REQUIREMENT_ID
    assert set(invalidation.invalidated_artifact_ids) == affected
    assert affected.isdisjoint(invalidation.preserved_artifact_ids)
    assert preserved.issubset(invalidation.preserved_artifact_ids)
    assert all(not store.lookup(item).hit for item in affected)
    assert all(store.lookup(item).hit for item in preserved)

    new_analysis_key = key("analysis", replacement)
    new_analysis, produced = store.get_or_compute(
        new_analysis_key, produce("analysis@2")
    )
    assert produced
    new_plan_key = key("planning", replacement, new_analysis)
    new_plan, produced = store.get_or_compute(
        new_plan_key,
        produce("planning@2"),
        dependencies=(new_analysis,),
    )
    assert produced
    new_validation_key = key("validation", replacement, new_plan)
    new_validation, produced = store.get_or_compute(
        new_validation_key,
        produce("validation@2"),
        dependencies=(new_plan,),
    )
    assert produced

    reused_independent, produced = store.get_or_compute(
        independent_key,
        lambda: pytest.fail("unaffected root was recomputed"),
    )
    assert not produced
    reused_child, produced = store.get_or_compute(
        independent_child_key,
        lambda: pytest.fail("unaffected descendant was recomputed"),
        dependencies=(independent,),
    )
    assert not produced
    assert reused_independent.artifact_id == independent.artifact_id
    assert reused_child.artifact_id == independent_child.artifact_id
    assert {
        new_analysis.artifact_id,
        new_plan.artifact_id,
        new_validation.artifact_id,
    }.isdisjoint(affected)
    assert calls == [
        "analysis",
        "planning",
        "validation",
        "independent",
        "independent-child",
        "analysis@2",
        "planning@2",
        "validation@2",
    ]
    metrics = store.metrics()
    assert metrics.invalidated == 3
    assert metrics.exact_reuses >= 7
    assert metrics.stale_authoritative_hits == 0
