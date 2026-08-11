"""IVP-010: create_verification_plan and planner policy.

Acceptance coverage:

* five-argument API for strict mappings and registered canonical objects
* patch base + RepositoryState/InvalidationPlan/ContextPack root cross-checks
* receipt keys bind the exact target patched tree
* relevant changes select/invalidate; unrelated avoid semantic over-selection
* cross-tree admission rejected; environment/lock/tool mismatches invalidate
* planning returns stale decisions without mutating tombstones
* uncertainty broadens; unbound sandbox / policy conflict / scope crossing
  require review
* resource and per-step/global timeout bounds are positive and capped
* acceptance requires production-admissible required success and no pending
  mandatory fallback
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.repository_forest import (
    AuthorityMode,
    LocalLocator,
    PortableGitClosure,
    RepositoryAuthority,
    RepositoryDescriptor,
    RepositoryForest,
    RepositoryIdentity,
)
from ipfs_accelerate_py.agent_supervisor.contract_analysis.execution_profile import (
    CapabilitySnapshot,
    LockIdentity,
    ToolIdentity,
)
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_bytes
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    CacheReuseDisposition,
    DirectExecutionObservation,
    TerminalStatus,
    TestReceipt,
    VerificationIdentityCompiler,
    VerificationReceiptKind,
)
from ipfs_accelerate_py.agent_supervisor.verification.datasets_adapter import (
    DATASETS_CONTEXT_PACK_SCHEMA,
    DATASETS_INVALIDATION_PLAN_SCHEMA,
    DATASETS_REPOSITORY_STATE_SCHEMA,
    DatasetsVerificationInputAdapter,
    InputKind,
    create_datasets_verification_input_adapter,
)
from ipfs_accelerate_py.agent_supervisor.verification.planner import (
    PLANNER_EVIDENCE,
    VERIFICATION_PLANNER_INTERFACE,
    VERIFICATION_PLANNER_SCHEMA,
    CheckToolSpec,
    IdentityBinding,
    IncrementalVerificationPlanner,
    PatchDelta,
    PlannerBoundsError,
    PlannerError,
    PlannerIdentityError,
    PlannerPolicy,
    REASON_CROSS_TREE_REJECTED,
    REASON_NO_PENDING_FALLBACK,
    REASON_POLICY_CONFLICT,
    REASON_PRODUCTION_SUCCESS_REQUIRED,
    REASON_SCOPE_CROSSING,
    REASON_UNBOUND_SANDBOX,
    compile_check_receipt_key,
    create_incremental_verification_planner,
    create_verification_plan,
)
from ipfs_accelerate_py.agent_supervisor.verification.receipt_cache import (
    REASON_TOMBSTONED,
    VerificationReceiptCache,
)
from ipfs_accelerate_py.agent_supervisor.verification.receipt_store import (
    HermeticVerificationReceiptStore,
)
from ipfs_accelerate_py.agent_supervisor.verification.selection import (
    SelectionPolicy,
    VerificationCatalog,
)


TREE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/observed-repository-tree@1"
SEMANTIC_SCHEMA = "ipfs_accelerate_py/agent-supervisor/observed-semantic-state@1"
ENVIRONMENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/effective-verification-environment@1"
)
TOOL_EXECUTABLE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/observed-tool-executable@1"
)

OPAQUE_TREE = "datasets-tree:planner-fixture-001"
TEST_A = "test/api/test_mod.py::test_fn"
TEST_B = "test/api/test_mod.py::test_helper"
TEST_C = "test/api/test_other.py::test_unrelated"
STATIC_A = "static:ruff:pkg/mod.py"
TYPE_A = "mypy:pkg.mod"


# ---------------------------------------------------------------------------
# Shared identity fixtures
# ---------------------------------------------------------------------------


def _artifact(label: str) -> str:
    return content_identity({"artifact": label, "schema": "fixture-artifact@1"})


def _structured_cid(schema: str, value: object) -> str:
    return content_identity({"schema": schema, "value": value})


def _repository_forest(
    *,
    commit: str = "abcdef0123456789abcdef0123456789abcdef01",
    tree: str = "0123456789abcdef0123456789abcdef01234567",
) -> RepositoryForest:
    alias = "ipfs_accelerate_py"
    descriptor = RepositoryDescriptor(
        identity=RepositoryIdentity(logical_name=alias),
        portable_closure=PortableGitClosure(commit=commit, tree=tree),
        local_locator=LocalLocator(
            alias=alias,
            root_path="/fixture/ipfs_accelerate_py",
            resolved_root_path="/fixture/ipfs_accelerate_py",
            local_repository_binding_id="fixture-binding:ipfs-accelerate",
        ),
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    return RepositoryForest(
        descriptors=(descriptor,),
        sole_write_alias=alias,
        policy_cid=_artifact("repository-forest-policy"),
    )


def _sandbox() -> dict[str, Any]:
    return {
        "sandbox_schema": "hermetic-sandbox@1",
        "sandbox_policy": {
            "schema": "hermetic-sandbox-policy@1",
            "network": "deny",
            "auto_install": "deny",
            "home_cache": "deny",
            "auth_material": "deny",
        },
        "filesystem_policy": {
            "schema": "verification-filesystem-policy@1",
            "source": "read_only",
            "artifacts": "private_writable",
        },
        "platform": {
            "schema": "verification-platform@1",
            "os": "linux",
            "architecture": "x86_64",
            "libc": "glibc-2.39",
        },
        "interpreter": {
            "schema": "verification-interpreter@1",
            "implementation": "cpython",
            "version": "3.12.3",
            "abi": "cp312",
        },
        "toolchain": {
            "schema": "verification-toolchain@1",
            "name": "locked-python",
            "revision": "fixture-1",
        },
        "dependency_distribution": {
            "schema": "verification-dependency-distribution@1",
            "entries": ("pytest==9.1.1",),
        },
        "environment_values": {
            "schema": "verification-environment-values@1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
    }


def _semantic() -> dict[str, Any]:
    return {
        "symbols": ["pkg.mod.fn@1"],
        "edge_root": "sha256:planner-semantic-edges",
    }


def _lock_bytes() -> bytes:
    return b"package==1.2.3 --hash=sha256:abcd\n"


def _identity_binding(
    forest: RepositoryForest | None = None,
    **overrides: Any,
) -> IdentityBinding:
    repository_forest = forest if forest is not None else _repository_forest()
    dependency_lock_bytes = _lock_bytes()
    dependency_lock_path = "requirements.lock"
    lock_sha = "sha256:" + hashlib.sha256(dependency_lock_bytes).hexdigest()
    # Capability snapshot starts with a placeholder tool; compile_check aligns
    # the selected tool before key construction.
    placeholder_bytes = b"reviewed-launcher:pytest"
    placeholder_sha = "sha256:" + hashlib.sha256(placeholder_bytes).hexdigest()
    capability_snapshot = CapabilitySnapshot(
        tool_identities={"verification-tool": placeholder_sha},
        lock_identities={dependency_lock_path: lock_sha},
        environment_names=("LANG", "LC_ALL"),
        read_paths=("/workspace/source",),
        write_paths=("/workspace/artifacts",),
    )
    semantic = _semantic()
    base: dict[str, Any] = {
        "patch_base_tree_id": OPAQUE_TREE,
        "observed_semantic_state": semantic,
        "sandbox_environment": _sandbox(),
        "capability_snapshot": capability_snapshot,
        "dependency_lock_path": dependency_lock_path,
        "dependency_lock_identity": LockIdentity(
            path=dependency_lock_path, identity=lock_sha
        ),
        "dependency_lock_bytes": dependency_lock_bytes,
        "network_policy": "deny_all",
        "receipt_schema_version": 1,
        "configuration_bytes": b"[tool]\nstrict = true\n",
        "fixture_data_bytes": (b"fixture-one\n",),
        "affected_symbol_versions": (
            {
                "symbol": "pkg.mod.fn",
                "version": 1,
                "source_cid": _artifact("source-fn-v1"),
            },
        ),
        "repository_forest": repository_forest,
        "repository_alias": repository_forest.sole_write_alias,
        "effective_sandbox_bound": True,
    }
    base.update(overrides)
    return IdentityBinding(**base)


def _target_tree_cid(forest: RepositoryForest, *, base: str = OPAQUE_TREE) -> str:
    descriptor = forest.write_descriptor()
    observation = {
        "repository_forest_cid": forest.forest_id,
        "git_commit_id": descriptor.commit,
        "git_tree_id": descriptor.tree,
        "gitlink_state_cid": descriptor.portable_closure.gitlink_closure_cid,
        "dirty_overlay_cid": descriptor.dirty_overlay_digest,
        "dirty": descriptor.dirty,
        "repository_alias": descriptor.alias,
        "repository_id": descriptor.repository_id,
        "descriptor_cid": descriptor.descriptor_cid,
        "base_repository_tree_id": base,
    }
    return _structured_cid(TREE_SCHEMA, observation)


def _pytest_tool(node_id: str = TEST_A) -> CheckToolSpec:
    return CheckToolSpec(
        tool_name="pytest",
        tool_version="9.1.1",
        adapter_schema="pytest-verification-adapter@1",
        selector_argv=("/usr/bin/python3.12", "-m", "pytest", node_id),
        resolved_tool_executable="/usr/bin/python3.12",
        tool_executable_bytes=b"reviewed-launcher:pytest",
    )


def _catalog(**overrides: Any) -> VerificationCatalog:
    base: dict[str, Any] = {
        "tests": [TEST_A, TEST_B, TEST_C],
        "static_checks": [STATIC_A],
        "type_checks": [TYPE_A],
        "proof_obligations": [],
        "static_check_targets": {STATIC_A: ["pkg.mod.fn", "pkg/mod.py"]},
        "type_check_targets": {TYPE_A: ["pkg.mod.fn", "pkg/mod.py"]},
        "proof_obligation_dependencies": {},
    }
    base.update(overrides)
    return VerificationCatalog(**base)


def _policy(
    *,
    forest: RepositoryForest | None = None,
    identity: IdentityBinding | None = None,
    **overrides: Any,
) -> PlannerPolicy:
    forest = forest if forest is not None else _repository_forest()
    kwargs: dict[str, Any] = {
        "identity": identity if identity is not None else _identity_binding(forest),
        "catalog": _catalog(),
        "selection_policy": SelectionPolicy(
            critical_uncertainty_requires_full_suite=False,
            broader_includes_sibling_tests=True,
        ),
        "tool_specs": {
            "test": _pytest_tool(),
        },
        "receipt_kinds": ("test",),
        "max_execution_time_ms": 120_000,
        "default_step_timeout_ms": 30_000,
        "expected_cpu_millis": 10_000,
        "expected_memory_bytes": 256 * 1024 * 1024,
        "expected_processes": 2,
        "expected_artifact_bytes": 8 * 1024 * 1024,
    }
    kwargs.update(overrides)
    return PlannerPolicy(**kwargs)


def _repository_state_mapping(**overrides: Any) -> dict[str, Any]:
    lock_cid = cid_for_bytes(_lock_bytes())
    payload = {
        "schema": DATASETS_REPOSITORY_STATE_SCHEMA,
        "repository_tree_id": OPAQUE_TREE,
        "semantic_state_root_cid": _structured_cid(SEMANTIC_SCHEMA, _semantic()),
        "environment_root_cid": "",
        "dependency_lock_root_cid": lock_cid,
    }
    payload.update(overrides)
    return payload


def _invalidation_plan_mapping(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema": DATASETS_INVALIDATION_PLAN_SCHEMA,
        "repository_tree_id": OPAQUE_TREE,
        "semantic_state_root_cid": _structured_cid(SEMANTIC_SCHEMA, _semantic()),
        "changed_symbols": ["pkg.mod.fn"],
        "changed_paths": ["pkg/mod.py"],
        "edges": [
            {
                "source": "pkg.mod.fn",
                "target": TEST_A,
                "kind": "tested_by",
            }
        ],
        "spans": [],
        "contracts": [],
        "uncertainty": {"frontier": "exact"},
        "uncovered_symbols": [],
        "uncovered_paths": [],
        "truncated": False,
        "uncovered_impact": False,
    }
    payload.update(overrides)
    return payload


def _context_pack_mapping(**overrides: Any) -> dict[str, Any]:
    lock_cid = cid_for_bytes(_lock_bytes())
    payload = {
        "schema": DATASETS_CONTEXT_PACK_SCHEMA,
        "repository_tree_id": OPAQUE_TREE,
        "semantic_state_root_cid": _structured_cid(SEMANTIC_SCHEMA, _semantic()),
        "environment_root_cid": "",
        "dependency_lock_root_cid": lock_cid,
        "token_estimate": 900,
        "fixture_task_references": ["fixture-task:planner-1"],
        "contracts": [],
    }
    payload.update(overrides)
    return payload


def _patch(
    forest: RepositoryForest | None = None,
    **overrides: Any,
) -> PatchDelta:
    forest = forest if forest is not None else _repository_forest()
    base: dict[str, Any] = {
        "base_tree_id": OPAQUE_TREE,
        "changed_paths": ["pkg/mod.py"],
        "changed_symbols": ["pkg.mod.fn"],
        "patch_paths": ["pkg/mod.py"],
        "declared_scope_paths": ["pkg/", "test/"],
        "target_tree_cid": _target_tree_cid(forest),
        "repository_forest": forest,
        "repository_alias": forest.sole_write_alias,
    }
    base.update(overrides)
    return PatchDelta(**base)


def _plan(**kwargs: Any):
    forest = kwargs.pop("forest", None) or _repository_forest()
    policy = kwargs.pop("policy", None) or _policy(forest=forest)
    patch = kwargs.pop("patch", None) or _patch(forest)
    rs = kwargs.pop("repository_state", None) or _repository_state_mapping()
    ip = kwargs.pop("invalidation_plan", None) or _invalidation_plan_mapping()
    cp = kwargs.pop("context_pack", None) or _context_pack_mapping()
    cache = kwargs.pop("cache", None)
    adapter = kwargs.pop("adapter", None)
    if kwargs:
        raise AssertionError(f"unexpected kwargs: {sorted(kwargs)}")
    return create_verification_plan(
        rs, ip, cp, patch, policy, cache=cache, adapter=adapter
    )


def _test_observation(key, status: TerminalStatus = TerminalStatus.PASSED):
    return DirectExecutionObservation(
        receipt_key_cid=key.key_id,
        repository_tree_cid=key.repository_tree_cid,
        environment_cid=key.environment_cid,
        repository_tree_observation=key.repository_tree_observation,
        environment_observation=dict(key.environment_observation),
        terminal_status=status,
        command_argv=("/usr/bin/python3.12", "-m", "pytest", TEST_A),
        duration_ms=40,
        exit_code=0 if status is TerminalStatus.PASSED else 1,
        stdout_artifact_cid=_artifact("stdout"),
        stderr_artifact_cid=_artifact("stderr"),
        artifact_cids=(_artifact("report"),),
        reason_codes=("fixture_run",),
    )


# ---------------------------------------------------------------------------
# Interface surface
# ---------------------------------------------------------------------------


def test_module_exports_planner_surface() -> None:
    assert VERIFICATION_PLANNER_INTERFACE == "IncrementalVerificationPlanner@1"
    assert VERIFICATION_PLANNER_SCHEMA.endswith("incremental-verification-planner@1")
    assert PLANNER_EVIDENCE == "ivp/verification-plan@1"
    planner = create_incremental_verification_planner()
    assert isinstance(planner, IncrementalVerificationPlanner)
    assert planner.INTERFACE == VERIFICATION_PLANNER_INTERFACE


def test_five_argument_api_strict_mappings() -> None:
    plan = _plan()
    assert plan.required_receipt_keys
    assert plan.affected_tests == (TEST_A,)
    assert plan.human_review_required is False
    assert plan.max_execution_time_ms == 120_000
    assert all(timeout > 0 for timeout in plan.step_timeouts_ms.values())
    assert all(timeout <= plan.max_execution_time_ms for timeout in plan.step_timeouts_ms.values())
    assert REASON_PRODUCTION_SUCCESS_REQUIRED in plan.acceptance_criteria
    assert REASON_NO_PENDING_FALLBACK in plan.acceptance_criteria
    # Every required key binds the target patched tree.
    forest = _repository_forest()
    target = _target_tree_cid(forest)
    for key in plan.required_receipt_keys:
        assert key.repository_tree_cid == target
        assert key.repository_tree_observation["base_repository_tree_id"] == OPAQUE_TREE


def test_registered_canonical_object_path() -> None:
    class FakeRepositoryState:
        def __init__(self, mapping: dict[str, Any]) -> None:
            self._mapping = mapping

    adapter = create_datasets_verification_input_adapter()
    adapter.register_upstream_type(
        FakeRepositoryState,
        input_kind=InputKind.REPOSITORY_STATE,
        converter=lambda obj: dict(obj._mapping),
        module_name="tests.fake_datasets",
        symbol_name="FakeRepositoryState",
    )
    plan = create_verification_plan(
        FakeRepositoryState(_repository_state_mapping()),
        _invalidation_plan_mapping(),
        _context_pack_mapping(),
        _patch(),
        _policy(),
        adapter=adapter,
    )
    assert plan.affected_tests == (TEST_A,)


# ---------------------------------------------------------------------------
# Root cross-checks
# ---------------------------------------------------------------------------


def test_patch_base_mismatch_fails_closed() -> None:
    with pytest.raises(PlannerIdentityError, match="patch_base_tree_mismatch"):
        create_verification_plan(
            _repository_state_mapping(repository_tree_id="other-tree"),
            _invalidation_plan_mapping(),
            _context_pack_mapping(),
            _patch(),
            _policy(),
        )


def test_semantic_root_mismatch_fails_closed() -> None:
    other = _structured_cid(SEMANTIC_SCHEMA, {"symbols": ["other@1"]})
    with pytest.raises(PlannerIdentityError, match="semantic_root_mismatch"):
        create_verification_plan(
            _repository_state_mapping(),
            _invalidation_plan_mapping(semantic_state_root_cid=other),
            _context_pack_mapping(),
            _patch(),
            _policy(),
        )


def test_cross_tree_receipt_cid_on_view_rejected() -> None:
    foreign = _artifact("foreign-tree")
    with pytest.raises(PlannerIdentityError, match=REASON_CROSS_TREE_REJECTED):
        create_verification_plan(
            _repository_state_mapping(repository_tree_cid=foreign),
            _invalidation_plan_mapping(),
            _context_pack_mapping(),
            _patch(),
            _policy(),
        )


# ---------------------------------------------------------------------------
# Selection + unrelated / uncertainty
# ---------------------------------------------------------------------------


def test_relevant_change_selects_and_requires_keys() -> None:
    plan = _plan()
    assert plan.affected_tests == (TEST_A,)
    assert TEST_C not in plan.affected_tests
    assert len(plan.required_receipt_keys) == 1
    assert len(plan.cache_reuse_decisions) == 1
    assert plan.cache_reuse_decisions[0].disposition is CacheReuseDisposition.MISSING


def test_unrelated_change_avoids_semantic_over_selection() -> None:
    plan = create_verification_plan(
        _repository_state_mapping(),
        _invalidation_plan_mapping(
            changed_symbols=["pkg.unrelated.symbol"],
            changed_paths=["docs/readme.md"],
            edges=[
                {"source": "pkg.mod.fn", "target": TEST_A, "kind": "tested_by"},
            ],
        ),
        _context_pack_mapping(),
        _patch(
            changed_paths=["docs/readme.md"],
            changed_symbols=["pkg.unrelated.symbol"],
            patch_paths=["docs/readme.md"],
            declared_scope_paths=["docs/", "pkg/", "test/"],
        ),
        _policy(),
    )
    # Semantic selection stays empty (no over-selection of TEST_A / TEST_C).
    assert plan.affected_tests == ()
    assert plan.fallback_tests == ()
    assert TEST_A not in plan.affected_tests
    assert TEST_C not in plan.affected_tests
    assert plan.full_suite_required is False
    # Contract requires a nonempty required key set; the planner admits only a
    # tree-rebind identity probe — not catalog application tests.
    assert plan.required_receipt_keys
    assert len(plan.required_receipt_keys) == 1
    assert TEST_A not in plan.fallback_tests
    assert TEST_C not in plan.fallback_tests


def test_uncertainty_broadens_selection() -> None:
    plan = create_verification_plan(
        _repository_state_mapping(),
        _invalidation_plan_mapping(
            edges=[
                {"source": "pkg.mod.fn", "target": TEST_A, "kind": "tested_by"},
                {
                    "source": "pkg.mod.fn",
                    "target": "runtime:plugin",
                    "kind": "opaque",
                    "opaque": True,
                    "disposition": "opaque",
                    "critical": True,
                },
            ],
        ),
        _context_pack_mapping(),
        _patch(),
        _policy(
            selection_policy=SelectionPolicy(
                critical_uncertainty_requires_full_suite=False,
                broader_includes_sibling_tests=True,
            )
        ),
    )
    assert TEST_A in plan.affected_tests
    # Broader sibling expansion and/or full-suite escalation under uncertainty.
    assert (
        plan.fallback_tests
        or plan.full_suite_required
        or TEST_B in plan.affected_tests
    )


def test_uncertainty_full_suite_when_policy_requires() -> None:
    plan = create_verification_plan(
        _repository_state_mapping(),
        _invalidation_plan_mapping(
            truncated=True,
            edges=[
                {"source": "pkg.mod.fn", "target": TEST_A, "kind": "tested_by"},
            ],
        ),
        _context_pack_mapping(),
        _patch(),
        _policy(
            selection_policy=SelectionPolicy(
                critical_uncertainty_requires_full_suite=True,
                broader_escalates_to_full_suite=True,
            )
        ),
    )
    assert plan.full_suite_required is True
    assert plan.full_suite_receipt_key_cids
    assert set(plan.full_suite_receipt_key_cids).issubset(
        {key.key_id for key in plan.required_receipt_keys}
    )
    assert REASON_PRODUCTION_SUCCESS_REQUIRED in plan.acceptance_criteria


# ---------------------------------------------------------------------------
# Cache: stale without tombstone mutation, cross-tree rejection
# ---------------------------------------------------------------------------


def test_planning_returns_stale_without_mutating_tombstones(tmp_path) -> None:
    forest = _repository_forest()
    policy = _policy(forest=forest)
    patch = _patch(forest)
    key = compile_check_receipt_key(
        policy=policy,
        kind=VerificationReceiptKind.TEST,
        check_id=TEST_A,
        patch=patch,
    )
    store = HermeticVerificationReceiptStore(tmp_path / "receipts")
    cache = VerificationReceiptCache(store)
    receipt = TestReceipt(key, _test_observation(key, TerminalStatus.PASSED))
    admit = cache.admit(receipt)
    assert admit.success

    # Mark stale via cache (executor path). Planning must not add further
    # tombstones when it only looks up.
    cache.mark_stale(key, reason="fixture_stale")
    index_before = store.current_index()
    tombstones_before = tuple(index_before.tombstones)

    plan = create_verification_plan(
        _repository_state_mapping(),
        _invalidation_plan_mapping(),
        _context_pack_mapping(),
        patch,
        policy,
        cache=cache,
    )
    assert plan.cache_reuse_decisions
    decision = plan.cache_reuse_decisions[0]
    assert decision.disposition is CacheReuseDisposition.STALE
    assert REASON_TOMBSTONED in decision.reason_codes or "stale" in " ".join(
        decision.reason_codes
    )

    index_after = store.current_index()
    assert tuple(index_after.tombstones) == tombstones_before


def test_cross_tree_old_receipt_not_admitted_for_new_tree(tmp_path) -> None:
    forest_old = _repository_forest(
        commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        tree="1111111111111111111111111111111111111111",
    )
    forest_new = _repository_forest(
        commit="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        tree="2222222222222222222222222222222222222222",
    )
    policy_old = _policy(forest=forest_old, identity=_identity_binding(forest_old))
    patch_old = _patch(forest_old)
    key_old = compile_check_receipt_key(
        policy=policy_old,
        kind=VerificationReceiptKind.TEST,
        check_id=TEST_A,
        patch=patch_old,
    )
    store = HermeticVerificationReceiptStore(tmp_path / "receipts-cross")
    cache = VerificationReceiptCache(store)
    receipt = TestReceipt(key_old, _test_observation(key_old))
    assert cache.admit(receipt).success

    policy_new = _policy(forest=forest_new, identity=_identity_binding(forest_new))
    patch_new = _patch(forest_new)
    plan = create_verification_plan(
        _repository_state_mapping(),
        _invalidation_plan_mapping(),
        _context_pack_mapping(),
        patch_new,
        policy_new,
        cache=cache,
    )
    assert plan.required_receipt_keys
    new_key = plan.required_receipt_keys[0]
    assert new_key.repository_tree_cid != key_old.repository_tree_cid
    assert new_key.key_id != key_old.key_id
    assert plan.cache_reuse_decisions[0].disposition is CacheReuseDisposition.MISSING
    # Old receipt remains immutable under its original key.
    old_decision = cache.lookup(key_old, for_production=True)
    assert old_decision.disposition is CacheReuseDisposition.REUSED


def test_environment_lock_tool_mismatch_invalidates() -> None:
    forest = _repository_forest()
    identity = _identity_binding(forest)
    # Claim a lock root that does not match the identity lock bytes.
    foreign_lock = cid_for_bytes(b"other-lock\n")
    with pytest.raises(PlannerIdentityError, match="dependency_lock_root_mismatch|lock"):
        create_verification_plan(
            _repository_state_mapping(dependency_lock_root_cid=foreign_lock),
            _invalidation_plan_mapping(),
            _context_pack_mapping(dependency_lock_root_cid=foreign_lock),
            _patch(forest),
            _policy(forest=forest, identity=identity),
        )


# ---------------------------------------------------------------------------
# Human review gates
# ---------------------------------------------------------------------------


def test_unbound_sandbox_requires_review() -> None:
    forest = _repository_forest()
    identity = _identity_binding(forest, effective_sandbox_bound=False)
    plan = create_verification_plan(
        _repository_state_mapping(),
        _invalidation_plan_mapping(),
        _context_pack_mapping(),
        _patch(forest),
        _policy(forest=forest, identity=identity),
    )
    assert plan.human_review_required is True
    assert REASON_UNBOUND_SANDBOX in plan.human_review_reason_codes


def test_policy_conflict_requires_review() -> None:
    plan = _plan(policy=_policy(policy_conflict=True))
    assert plan.human_review_required is True
    assert REASON_POLICY_CONFLICT in plan.human_review_reason_codes


def test_declared_scope_crossing_requires_review() -> None:
    forest = _repository_forest()
    plan = create_verification_plan(
        _repository_state_mapping(),
        _invalidation_plan_mapping(),
        _context_pack_mapping(),
        _patch(
            forest,
            patch_paths=["secrets/prod.env", "pkg/mod.py"],
            declared_scope_paths=["pkg/", "test/"],
        ),
        _policy(forest=forest),
    )
    assert plan.human_review_required is True
    assert REASON_SCOPE_CROSSING in plan.human_review_reason_codes


# ---------------------------------------------------------------------------
# Resources / timeouts / acceptance
# ---------------------------------------------------------------------------


def test_resource_and_timeout_bounds_positive_and_capped() -> None:
    plan = _plan()
    assert plan.expected_cpu_millis > 0
    assert plan.expected_memory_bytes > 0
    assert plan.expected_processes >= 1
    assert plan.expected_artifact_bytes > 0
    assert plan.max_execution_time_ms > 0
    assert plan.step_timeouts_ms
    assert set(plan.step_timeouts_ms) == set(plan.dependency_dag)
    for step, timeout in plan.step_timeouts_ms.items():
        assert timeout > 0
        assert timeout <= plan.max_execution_time_ms
    assert plan.execution_order


def test_policy_rejects_non_positive_timeouts() -> None:
    with pytest.raises(PlannerBoundsError):
        _policy(max_execution_time_ms=0)
    with pytest.raises(PlannerBoundsError):
        _policy(default_step_timeout_ms=0)
    with pytest.raises(PlannerBoundsError):
        _policy(max_execution_time_ms=10_000, default_step_timeout_ms=20_000)


def test_acceptance_criteria_require_success_and_no_pending_fallback() -> None:
    plan = _plan()
    assert REASON_PRODUCTION_SUCCESS_REQUIRED in plan.acceptance_criteria
    assert REASON_NO_PENDING_FALLBACK in plan.acceptance_criteria
    assert "every_required_receipt_current_production_admissible_success" in (
        plan.acceptance_criteria
    )
    assert plan.full_suite_required is False


def test_planner_object_matches_function() -> None:
    forest = _repository_forest()
    policy = _policy(forest=forest)
    patch = _patch(forest)
    rs = _repository_state_mapping()
    ip = _invalidation_plan_mapping()
    cp = _context_pack_mapping()
    planner = IncrementalVerificationPlanner()
    a = planner.create_plan(rs, ip, cp, patch, policy)
    b = create_verification_plan(rs, ip, cp, patch, policy)
    assert a.to_record() == b.to_record()
    c = planner(rs, ip, cp, patch, policy)
    assert c.to_record() == a.to_record()


def test_mapping_policy_and_patch_delta_accepted() -> None:
    forest = _repository_forest()
    policy = _policy(forest=forest)
    patch = _patch(forest)
    plan = create_verification_plan(
        _repository_state_mapping(),
        _invalidation_plan_mapping(),
        _context_pack_mapping(),
        patch.to_dict()
        | {
            "repository_forest": forest,
            "target_tree_cid": patch.target_tree_cid,
        },
        {
            "identity": {
                "patch_base_tree_id": policy.identity.patch_base_tree_id,
                "observed_semantic_state": dict(policy.identity.observed_semantic_state),
                "sandbox_environment": dict(policy.identity.sandbox_environment),
                "capability_snapshot": {
                    "tool_identities": dict(
                        policy.identity.capability_snapshot.tool_identities
                    ),
                    "lock_identities": dict(
                        policy.identity.capability_snapshot.lock_identities
                    ),
                    "environment_names": list(
                        policy.identity.capability_snapshot.environment_names
                    ),
                    "read_paths": list(policy.identity.capability_snapshot.read_paths),
                    "write_paths": list(
                        policy.identity.capability_snapshot.write_paths
                    ),
                },
                "dependency_lock_path": policy.identity.dependency_lock_path,
                "dependency_lock_identity": policy.identity.dependency_lock_identity.to_dict(),
                "dependency_lock_bytes": policy.identity.dependency_lock_bytes,
                "repository_forest": forest,
                "affected_symbol_versions": [
                    dict(item) for item in policy.identity.affected_symbol_versions
                ],
            },
            "catalog": policy.catalog.to_dict(),
            "selection_policy": policy.selection_policy.to_dict(),
            "tool_specs": {"test": policy.tool_specs["test"].to_dict() | {
                "tool_executable_bytes": policy.tool_specs["test"].tool_executable_bytes,
                "selector_argv": list(policy.tool_specs["test"].selector_argv),
                "resolved_tool_executable": policy.tool_specs["test"].resolved_tool_executable,
            }},
            "receipt_kinds": ["test"],
            "max_execution_time_ms": 120_000,
            "default_step_timeout_ms": 30_000,
        },
    )
    assert plan.affected_tests == (TEST_A,)


def test_malformed_repository_state_fails_closed() -> None:
    with pytest.raises(PlannerError, match="repository_state"):
        create_verification_plan(
            {"schema": "not-a-supported-schema@1"},
            _invalidation_plan_mapping(),
            _context_pack_mapping(),
            _patch(),
            _policy(),
        )
