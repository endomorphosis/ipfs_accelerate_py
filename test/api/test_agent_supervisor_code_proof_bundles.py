"""CBP-100: bundle optimizer locality for obligations and proof-cache keys.

Acceptance:
- Independent obligation sets remain parallel (same wave when file-safe).
- Conflicting predicted files serialize across waves.
- Optimizer prefers shared proof-cache key prefixes on the same tree.
- Shared prefixes never manufacture wrong-tree cache affinity hits.
"""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.bundle_optimizer import (
    BundleOptimizationPolicy,
    optimize_task_bundles,
)


TREE_A = "git-tree:cbp-100-a"
TREE_B = "git-tree:cbp-100-b"
CACHE_PREFIX_ALPHA = "proof-cache-ns/toolchain:nix/policy:formal/template:dep"
CACHE_PREFIX_BETA = "proof-cache-ns/toolchain:nix/policy:formal/template:sec"
OBLIGATION_SET_A = "obligation-set:alpha"
OBLIGATION_SET_B = "obligation-set:beta"


def _task(task_id: str, **overrides: object) -> dict[str, object]:
    suffix = task_id.casefold()
    payload: dict[str, object] = {
        "task_id": task_id,
        "canonical_task_cid": f"cid-{suffix}",
        "canonical_task_key": f"task/cbp100/{suffix}",
        "title": f"Prove and implement {task_id}",
        "goal_id": f"goal/{suffix}",
        "acceptance": [f"{task_id} accepted"],
        "effects": [f"{task_id} effect"],
        "outputs": [f"src/{suffix}.py"],
        "predicted_paths": [f"src/{suffix}.py"],
        "predicted_symbols": [f"{task_id}.run"],
        "context_paths": [f"context/{suffix}.md"],
        "immutable_context_keys": [f"context-cid/{suffix}"],
        "artifact_locality_keys": [f"artifact/{suffix}"],
        "validation_commands": [f"pytest test/{suffix}.py"],
        "resource_class": "cpu-medium",
        "provider_batch_key": "provider/batch/default",
        "estimated_context_tokens": 100,
        "estimated_tokens": 200,
        "estimated_validation_seconds": 5,
        "merge_fate": f"merge/{suffix}",
        "merge_pressure": 1,
        "dependencies": [],
        "conflicts": [],
        "status": "pending",
        "repository_tree_id": TREE_A,
        "obligation_ids": [f"obl/{suffix}"],
        "proof_cache_key_prefixes": [f"proof-cache-ns/{suffix}"],
        "proof_cache_key_ids": [f"proof-cache-key:sha256:{suffix}"],
    }
    payload.update(overrides)
    return payload


def _members(result: object) -> set[frozenset[str]]:
    return {frozenset(bundle.task_cids) for bundle in result.bundles}


def _wave_by_cid(result: object) -> dict[str, int]:
    return {
        cid: bundle.execution_wave
        for bundle in result.bundles
        for cid in bundle.task_cids
    }


def test_independent_obligation_sets_remain_parallel() -> None:
    """Distinct obligation sets with disjoint files stay in one execution wave."""

    tasks = [
        _task(
            "ALPHA-1",
            goal_id="goal/alpha",
            obligation_ids=["obl/alpha-1"],
            obligation_set_id=OBLIGATION_SET_A,
            predicted_paths=["src/alpha_one.py"],
            outputs=["src/alpha_one.py"],
            proof_cache_key_prefixes=[CACHE_PREFIX_ALPHA],
            proof_cache_key_ids=["proof-cache-key:sha256:alpha-1"],
        ),
        _task(
            "ALPHA-2",
            goal_id="goal/alpha",
            obligation_ids=["obl/alpha-2"],
            obligation_set_id=OBLIGATION_SET_A,
            predicted_paths=["src/alpha_two.py"],
            outputs=["src/alpha_two.py"],
            proof_cache_key_prefixes=[CACHE_PREFIX_ALPHA],
            proof_cache_key_ids=["proof-cache-key:sha256:alpha-2"],
            immutable_context_keys=["context-cid/alpha"],
            context_paths=["context/alpha.md"],
        ),
        _task(
            "BETA-1",
            goal_id="goal/beta",
            obligation_ids=["obl/beta-1"],
            obligation_set_id=OBLIGATION_SET_B,
            predicted_paths=["src/beta_one.py"],
            outputs=["src/beta_one.py"],
            proof_cache_key_prefixes=[CACHE_PREFIX_BETA],
            proof_cache_key_ids=["proof-cache-key:sha256:beta-1"],
        ),
        _task(
            "BETA-2",
            goal_id="goal/beta",
            obligation_ids=["obl/beta-2"],
            obligation_set_id=OBLIGATION_SET_B,
            predicted_paths=["src/beta_two.py"],
            outputs=["src/beta_two.py"],
            proof_cache_key_prefixes=[CACHE_PREFIX_BETA],
            proof_cache_key_ids=["proof-cache-key:sha256:beta-2"],
            immutable_context_keys=["context-cid/beta"],
            context_paths=["context/beta.md"],
        ),
    ]
    # Give ALPHA-1 the same immutable context as ALPHA-2 so set A co-locates.
    tasks[0] = {
        **tasks[0],
        "immutable_context_keys": ["context-cid/alpha"],
        "context_paths": ["context/alpha.md"],
    }
    tasks[2] = {
        **tasks[2],
        "immutable_context_keys": ["context-cid/beta"],
        "context_paths": ["context/beta.md"],
    }

    plan = optimize_task_bundles(
        tuple(reversed(tasks)),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=2),
    )

    waves = _wave_by_cid(plan)
    assert set(waves.values()) == {0}
    assert plan.metrics["critical_path_wave_count"] == 1
    assert plan.metrics["merge_conflict_rate_millionths"] == 0
    # Independent sets may occupy parallel lanes (width >= 2) without serializing.
    assert plan.execution_width_by_wave.get(0, 0) >= 2


def test_conflicting_predicted_files_serialize() -> None:
    """Shared predicted-file scopes still force sequential waves."""

    first = _task(
        "WRITER-A",
        goal_id="goal/shared-writer",
        predicted_paths=["src/shared_module.py"],
        outputs=["src/shared_module.py"],
        obligation_ids=["obl/writer-a"],
        obligation_set_id=OBLIGATION_SET_A,
        proof_cache_key_prefixes=[CACHE_PREFIX_ALPHA],
        immutable_context_keys=["context-cid/shared"],
        context_paths=["context/shared.md"],
        merge_fate="merge/shared",
    )
    second = _task(
        "WRITER-B",
        goal_id="goal/shared-writer",
        predicted_paths=["src/shared_module.py"],
        outputs=["src/shared_module.py"],
        obligation_ids=["obl/writer-b"],
        obligation_set_id=OBLIGATION_SET_A,
        proof_cache_key_prefixes=[CACHE_PREFIX_ALPHA],
        immutable_context_keys=["context-cid/shared"],
        context_paths=["context/shared.md"],
        merge_fate="merge/shared",
    )

    plan = optimize_task_bundles(
        (first, second),
        policy=BundleOptimizationPolicy(
            max_tasks_per_bundle=2,
            allow_internal_conflicts=False,
        ),
    )

    waves = _wave_by_cid(plan)
    assert waves["cid-writer-a"] != waves["cid-writer-b"]
    assert plan.metrics["critical_path_wave_count"] >= 2
    assert plan.metrics["blocking_conflict_count"] >= 1
    # High cache/obligation affinity must not collapse conflicting writers.
    assert len(plan.bundles) == 2


def test_optimizer_prefers_shared_proof_cache_key_prefixes_same_tree() -> None:
    """Same-tree tasks that share cache prefixes co-locate for single-flight."""

    tasks = [
        _task(
            "CACHE-A1",
            goal_id="goal/cache-a",
            predicted_paths=["src/cache_a1.py"],
            outputs=["src/cache_a1.py"],
            obligation_ids=["obl/cache-a1"],
            obligation_set_id=OBLIGATION_SET_A,
            repository_tree_id=TREE_A,
            proof_cache_key_prefixes=[CACHE_PREFIX_ALPHA],
            proof_cache_key_ids=["proof-cache-key:sha256:a1"],
            # Suppress incidental affinity so cache locality is the signal.
            immutable_context_keys=["context-cid/cache-a1"],
            context_paths=["context/cache_a1.md"],
            merge_fate="merge/cache-a1",
            validation_commands=["pytest test/cache_a1.py"],
        ),
        _task(
            "CACHE-A2",
            goal_id="goal/cache-a",
            predicted_paths=["src/cache_a2.py"],
            outputs=["src/cache_a2.py"],
            obligation_ids=["obl/cache-a2"],
            obligation_set_id=OBLIGATION_SET_A,
            repository_tree_id=TREE_A,
            proof_cache_key_prefixes=[CACHE_PREFIX_ALPHA],
            proof_cache_key_ids=["proof-cache-key:sha256:a2"],
            immutable_context_keys=["context-cid/cache-a2"],
            context_paths=["context/cache_a2.md"],
            merge_fate="merge/cache-a2",
            validation_commands=["pytest test/cache_a2.py"],
        ),
        _task(
            "CACHE-B1",
            goal_id="goal/cache-b",
            predicted_paths=["src/cache_b1.py"],
            outputs=["src/cache_b1.py"],
            obligation_ids=["obl/cache-b1"],
            obligation_set_id=OBLIGATION_SET_B,
            repository_tree_id=TREE_A,
            proof_cache_key_prefixes=[CACHE_PREFIX_BETA],
            proof_cache_key_ids=["proof-cache-key:sha256:b1"],
            immutable_context_keys=["context-cid/cache-b1"],
            context_paths=["context/cache_b1.md"],
            merge_fate="merge/cache-b1",
            validation_commands=["pytest test/cache_b1.py"],
        ),
        _task(
            "CACHE-B2",
            goal_id="goal/cache-b",
            predicted_paths=["src/cache_b2.py"],
            outputs=["src/cache_b2.py"],
            obligation_ids=["obl/cache-b2"],
            obligation_set_id=OBLIGATION_SET_B,
            repository_tree_id=TREE_A,
            proof_cache_key_prefixes=[CACHE_PREFIX_BETA],
            proof_cache_key_ids=["proof-cache-key:sha256:b2"],
            immutable_context_keys=["context-cid/cache-b2"],
            context_paths=["context/cache_b2.md"],
            merge_fate="merge/cache-b2",
            validation_commands=["pytest test/cache_b2.py"],
        ),
    ]

    plan = optimize_task_bundles(
        tuple(reversed(tasks)),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=2),
    )

    assert _members(plan) == {
        frozenset(("cid-cache-a1", "cid-cache-a2")),
        frozenset(("cid-cache-b1", "cid-cache-b2")),
    }
    assert plan.metrics["proof_cache_locality_reuse_millionths"] > 0
    assert plan.metrics["same_tree_proof_cache_co_located_pairs"] >= 2
    assert plan.metrics["wrong_tree_proof_cache_hits"] == 0
    for bundle in plan.bundles:
        assert CACHE_PREFIX_ALPHA in bundle.shared_proof_cache_key_prefixes or (
            CACHE_PREFIX_BETA in bundle.shared_proof_cache_key_prefixes
        )
        assert bundle.repository_tree_ids == (TREE_A,)


def test_shared_cache_prefixes_do_not_accept_wrong_tree_hits() -> None:
    """Identical prefixes across different trees never create cache affinity."""

    same_prefix = CACHE_PREFIX_ALPHA
    left = _task(
        "TREE-A",
        goal_id="goal/tree-a",
        predicted_paths=["src/tree_a.py"],
        outputs=["src/tree_a.py"],
        repository_tree_id=TREE_A,
        obligation_ids=["obl/tree-a"],
        obligation_set_id=OBLIGATION_SET_A,
        proof_cache_key_prefixes=[same_prefix],
        proof_cache_key_ids=["proof-cache-key:sha256:same-looking"],
        immutable_context_keys=["context-cid/tree-a"],
        context_paths=["context/tree_a.md"],
        merge_fate="merge/tree-a",
        validation_commands=["pytest test/tree_a.py"],
        # Isolate cache/obligation signals: no shared resource/provider/goal.
        resource_class="cpu-medium",
        provider_batch_key="provider/batch/tree-a",
    )
    right = _task(
        "TREE-B",
        goal_id="goal/tree-b",
        predicted_paths=["src/tree_b.py"],
        outputs=["src/tree_b.py"],
        repository_tree_id=TREE_B,
        obligation_ids=["obl/tree-b"],
        # Deliberately reuses the same obligation set id string to prove tree
        # binding still blocks cache-driven co-location.
        obligation_set_id=OBLIGATION_SET_A,
        proof_cache_key_prefixes=[same_prefix],
        proof_cache_key_ids=["proof-cache-key:sha256:same-looking"],
        immutable_context_keys=["context-cid/tree-b"],
        context_paths=["context/tree_b.md"],
        merge_fate="merge/tree-b",
        validation_commands=["pytest test/tree_b.py"],
        resource_class="gpu-large",
        provider_batch_key="provider/batch/tree-b",
    )

    plan = optimize_task_bundles(
        (left, right),
        policy=BundleOptimizationPolicy(
            max_tasks_per_bundle=2,
            require_affinity=True,
        ),
    )

    # Without same-tree cache affinity (and with distinct incidental keys),
    # require_affinity keeps the tasks in separate bundles.
    assert _members(plan) == {
        frozenset(("cid-tree-a",)),
        frozenset(("cid-tree-b",)),
    }
    assert plan.metrics["wrong_tree_proof_cache_affinity_rejections"] >= 1
    assert plan.metrics["wrong_tree_proof_cache_hits"] == 0
    assert plan.metrics["same_tree_proof_cache_co_located_pairs"] == 0
    for bundle in plan.bundles:
        assert not bundle.shared_proof_cache_key_prefixes
        assert not bundle.shared_proof_cache_key_ids


def test_unbound_tree_cannot_claim_proof_cache_locality() -> None:
    """In-memory / unbound trees never earn proof-cache co-location."""

    left = _task(
        "MEM-A",
        goal_id="goal/mem-a",
        repository_tree_id="in-memory",
        predicted_paths=["src/mem_a.py"],
        outputs=["src/mem_a.py"],
        proof_cache_key_prefixes=[CACHE_PREFIX_ALPHA],
        proof_cache_key_ids=["proof-cache-key:sha256:mem"],
        obligation_ids=["obl/mem-a"],
        immutable_context_keys=["context-cid/mem-a"],
        context_paths=["context/mem_a.md"],
        merge_fate="merge/mem-a",
        provider_batch_key="provider/batch/mem-a",
        resource_class="cpu-small",
    )
    right = _task(
        "MEM-B",
        goal_id="goal/mem-b",
        repository_tree_id="working-tree",
        predicted_paths=["src/mem_b.py"],
        outputs=["src/mem_b.py"],
        proof_cache_key_prefixes=[CACHE_PREFIX_ALPHA],
        proof_cache_key_ids=["proof-cache-key:sha256:mem"],
        obligation_ids=["obl/mem-b"],
        immutable_context_keys=["context-cid/mem-b"],
        context_paths=["context/mem_b.md"],
        merge_fate="merge/mem-b",
        provider_batch_key="provider/batch/mem-b",
        resource_class="cpu-large",
    )

    plan = optimize_task_bundles(
        (left, right),
        policy=BundleOptimizationPolicy(
            max_tasks_per_bundle=2,
            require_affinity=True,
        ),
    )

    assert _members(plan) == {
        frozenset(("cid-mem-a",)),
        frozenset(("cid-mem-b",)),
    }
    assert plan.metrics["wrong_tree_proof_cache_affinity_rejections"] >= 1
    assert plan.metrics["wrong_tree_proof_cache_hits"] == 0


def test_shared_obligation_set_co_locates_on_same_tree() -> None:
    """Open obligations within one set prefer a single bundle for single-flight."""

    tasks = [
        _task(
            "OBL-1",
            goal_id="goal/obligation-locality",
            predicted_paths=["src/obl_1.py"],
            outputs=["src/obl_1.py"],
            repository_tree_id=TREE_A,
            obligation_ids=["obl/shared-1", "obl/shared-core"],
            obligation_set_id=OBLIGATION_SET_A,
            proof_cache_key_prefixes=["proof-cache-ns/other-1"],
            proof_cache_key_ids=["proof-cache-key:sha256:obl-1"],
            immutable_context_keys=["context-cid/obl-1"],
            context_paths=["context/obl_1.md"],
            merge_fate="merge/obl-1",
            validation_commands=["pytest test/obl_1.py"],
        ),
        _task(
            "OBL-2",
            goal_id="goal/obligation-locality",
            predicted_paths=["src/obl_2.py"],
            outputs=["src/obl_2.py"],
            repository_tree_id=TREE_A,
            obligation_ids=["obl/shared-2", "obl/shared-core"],
            obligation_set_id=OBLIGATION_SET_A,
            proof_cache_key_prefixes=["proof-cache-ns/other-2"],
            proof_cache_key_ids=["proof-cache-key:sha256:obl-2"],
            immutable_context_keys=["context-cid/obl-2"],
            context_paths=["context/obl_2.md"],
            merge_fate="merge/obl-2",
            validation_commands=["pytest test/obl_2.py"],
        ),
    ]

    plan = optimize_task_bundles(
        tuple(reversed(tasks)),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=2),
    )

    assert _members(plan) == {frozenset(("cid-obl-1", "cid-obl-2"))}
    bundle = plan.bundles[0]
    assert "obl/shared-core" in bundle.shared_obligation_locality_keys
    assert OBLIGATION_SET_A in bundle.shared_obligation_locality_keys
    assert plan.metrics["obligation_locality_reuse_millionths"] > 0
    assert plan.metrics["wrong_tree_proof_cache_hits"] == 0


def test_bundle_projection_exposes_obligation_and_cache_locality() -> None:
    """Optimized bundles project locality keys for downstream schedulers."""

    plan = optimize_task_bundles(
        (
            _task(
                "PROJ-1",
                repository_tree_id=TREE_A,
                obligation_ids=["obl/proj"],
                obligation_set_id=OBLIGATION_SET_A,
                proof_cache_key_prefixes=[CACHE_PREFIX_ALPHA],
                proof_cache_key_ids=["proof-cache-key:sha256:proj"],
            ),
        )
    )
    bundle = plan.bundles[0]
    payload = bundle.to_dict()
    assert payload["repository_tree_ids"] == [TREE_A]
    assert "obl/proj" in payload["obligation_locality_keys"]
    assert OBLIGATION_SET_A in payload["obligation_locality_keys"]
    assert CACHE_PREFIX_ALPHA in payload["proof_cache_key_prefixes"]
    assert "proof-cache-key:sha256:proj" in payload["proof_cache_key_ids"]
    assert plan.metrics["wrong_tree_proof_cache_hits"] == 0
