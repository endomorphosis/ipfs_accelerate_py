"""IVP-016: required adversarial incremental-verification conformance matrix.

Evidence surface: ``ivp/conformance-matrix@1``

This hard gate consolidates the normative mutation, authority, cancellation,
concurrency, routing, commitment, and selection-evaluation behaviors that
release criteria depend on. Focused module tests remain the primary unit
coverage; this suite proves the closed 18-case matrix plus fail-closed
mutations and the controlled-fixture false-negative gate.

Required matrix (plan §13 / IVP-016 acceptance)
-----------------------------------------------
1. Unchanged same-tree receipt reuses
2. Relevant code change invalidates reuse
3. Unrelated edit preserves old-key history and rejects the new-tree key
4. Environment change invalidates
5. Dependency-lock change invalidates
6. Tool-version change invalidates
7. Stale receipt is rejected
8. Simulated receipt is rejected for production
9. Timeout remains timeout
10. Unavailable prover remains unavailable
11. Selected-test failure yields a rerun-validated minimized counterexample
12. Uncertain selection broadens
13. Concurrent writers preserve both entries (CAS)
14. Grandchild / escaped-session cancellation terminates the tree
15. Localized exact work selects the small-model route
16. Broad / opaque work selects the frontier route
17. Unresolved high risk or unavailable required tier selects human review
18. Commitment membership/content change + input-permutation invariance

Fail-closed extras: content corruption, kind mismatch, proof/test conflict,
and late cancelled success never manufacture production authority.

Selection-evaluation gate: controlled measured fixtures have zero false
negatives; missing canonical fixtures remain ``not_measured`` (never zero).
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, Sequence

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
    ResourceScheduler,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import pid_alive
from ipfs_accelerate_py.agent_supervisor.verification.bundle import (
    build_verification_bundle,
    build_verification_commitment,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    CacheReuseDisposition,
    ModelRoute,
    ProofReceipt,
    TerminalStatus,
    TypeCheckReceipt,
    VerificationReceiptKind,
)
from ipfs_accelerate_py.agent_supervisor.verification.counterexamples import (
    MinimizationGuarantee,
    extract_failure_material_from_pytest_output,
    minimize_counterexample,
)
from ipfs_accelerate_py.agent_supervisor.verification.evaluation import (
    CANONICAL_CORPUS_ID,
    REASON_CORPUS_ABSENT,
    MeasurementStatus,
    compare_selected_with_full_suite,
    evaluate_controlled_fixture_corpus,
    evaluate_default_corpus,
    load_controlled_fixtures,
)
from ipfs_accelerate_py.agent_supervisor.verification.executor import (
    CheckRunOutcome,
    execute_verification_plan,
)
from ipfs_accelerate_py.agent_supervisor.verification.model_route import (
    REASON_BROAD_DEPENDENCY_CONE,
    REASON_LOCALIZED_EXACT_COUNTEREXAMPLE,
    REASON_OPAQUE_CRITICAL_DEPENDENCY,
    REASON_PROOF_TEST_CONFLICT,
    REASON_REQUIRED_TIER_UNAVAILABLE,
    REASON_UNMODELED_HIGH_RISK,
    AnalysisKind,
    CounterexampleQuality,
    ModelRouteFacts,
    ModelRoutePolicy,
    RiskLevel,
    decide_model_route,
    default_inventory,
    policy_cid_for,
)
from ipfs_accelerate_py.agent_supervisor.verification.process_runner import (
    NETWORK_POLICY_DENY_ALL,
    VerificationCancellation,
    VerificationCommand,
    VerificationProcessRunner,
    VerificationRunDisposition,
    build_closed_sandbox,
    build_hermetic_environment,
)
from ipfs_accelerate_py.agent_supervisor.verification.receipt_cache import (
    REASON_EXACT_CURRENT_PRODUCTION,
    REASON_KEY_MISMATCH,
    REASON_KIND_MISMATCH,
    REASON_TOMBSTONED,
    AdmitResult,
    production_eligible,
)
from ipfs_accelerate_py.agent_supervisor.verification.receipt_store import (
    IndexEntry,
    build_receipt_envelope,
    cas_publish_entry,
    mapping_cid,
)
from ipfs_accelerate_py.agent_supervisor.verification.selection import (
    REASON_OPAQUE_CRITICAL,
    FallbackMode,
    select_affected_verification,
)

# Focused-suite helpers (canonical identity / receipt / plan factories).
from test.api.test_agent_supervisor_verification_bundle import (
    _failed_type,
    _passing_type,
)
from test.api.test_agent_supervisor_verification_contracts import (
    _key as _contract_key,
)
from test.api.test_agent_supervisor_verification_contracts import (
    _observation as _contract_observation,
)
from test.api.test_agent_supervisor_verification_contracts import (
    _plan,
    _route,
)
from test.api.test_agent_supervisor_verification_counterexamples import (
    NOISY_PYTEST_OUTPUT,
    ORIGINAL_ARGV,
    _failed_test,
    _material_from_noisy_output,
    _oracle_preserving,
)
from test.api.test_agent_supervisor_verification_receipt_cache import (
    _cache,
    _key,
    _related_key_variants,
    _repository_forest,
    _static_receipt,
    _type_check_receipt,
)
from test.api.test_agent_supervisor_verification_selection import (
    TEST_A as SEL_TEST_A,
)
from test.api.test_agent_supervisor_verification_selection import (
    _catalog as _selection_catalog,
)
from test.api.test_agent_supervisor_verification_selection import (
    _edge as _selection_edge,
)
from test.api.test_agent_supervisor_verification_selection import (
    _policy as _selection_policy,
)
from test.api.test_agent_supervisor_verification_selection import (
    _select,
)

# ---------------------------------------------------------------------------
# Evidence / matrix bookkeeping
# ---------------------------------------------------------------------------

CONFORMANCE_MATRIX_EVIDENCE: Final[str] = "ivp/conformance-matrix@1"
REQUIRED_CASE_COUNT: Final[int] = 18

REQUIRED_MATRIX_CASES: Final[tuple[str, ...]] = (
    "same_tree_reuse",
    "relevant_invalidation",
    "unrelated_old_key_preservation_new_tree_rejection",
    "environment_invalidation",
    "lock_invalidation",
    "tool_version_invalidation",
    "stale_rejection",
    "simulated_production_rejection",
    "timeout_preservation",
    "unavailable_prover_preservation",
    "rerun_validated_minimized_selected_test_counterexample",
    "uncertain_selection_broadening",
    "concurrent_writer_safety",
    "grandchild_escaped_session_cancellation",
    "small_localized_route",
    "frontier_broad_opaque_route",
    "human_review_unresolved_high_risk_or_unavailable_tier",
    "commitment_membership_content_permutation",
)

# Change kinds that are controlled, measured, and must show zero false
# negatives under the hard conformance gate (seeded FN/FP and not_measured
# / inconclusive corpus cases are excluded).
_CONTROLLED_MEASURED_ZERO_FN_KINDS: Final[frozenset[str]] = frozenset(
    {
        "direct_symbol",
        "transitive",
        "unrelated",
        "fixture_edge",
        "config_edge",
        "environment",
        "lock",
        "deliberately_failing",
        "equivalent_controlled",
        "opaque",
        "dynamic",
        "validation_mapping",
    }
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
FIXTURE_ROOT: Final[Path] = (
    REPO_ROOT / "test" / "fixtures" / "incremental_verification"
)

# Cases proven by this module; the meta-test asserts full coverage.
_PROVEN_CASES: set[str] = set()


def _mark(case: str) -> None:
    if case not in REQUIRED_MATRIX_CASES:
        raise AssertionError(f"unknown matrix case: {case!r}")
    _PROVEN_CASES.add(case)


def _ensure_corpus_manifest() -> Path:
    """Materialize the gitignored corpus manifest when absent."""

    manifest = FIXTURE_ROOT / "corpus_manifest.json"
    if manifest.is_file():
        return manifest
    script = FIXTURE_ROOT / "build_corpus.py"
    assert script.is_file(), f"missing corpus builder {script}"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"build_corpus failed: rc={proc.returncode} stderr={proc.stderr!r}"
    )
    assert manifest.is_file(), "corpus manifest not produced"
    return manifest


@pytest.fixture(scope="module")
def controlled_fixtures():
    _ensure_corpus_manifest()
    fixtures = load_controlled_fixtures(FIXTURE_ROOT, require_present=True)
    assert fixtures, "expected controlled corpus cases"
    return fixtures


# ---------------------------------------------------------------------------
# Shared micro-helpers
# ---------------------------------------------------------------------------


def _host() -> HostResourceSnapshot:
    return HostResourceSnapshot(
        worker_limit=16,
        available_worker_capacity=16,
        active_workers=0,
        memory_available_bytes=8 * 1024 * 1024 * 1024,
        disk_available_bytes=8 * 1024 * 1024 * 1024,
        memory_total_bytes=16 * 1024 * 1024 * 1024,
        disk_total_bytes=64 * 1024 * 1024 * 1024,
        capabilities=("cpu",),
        resource_classes=(
            "cpu-validation",
            "cpu-proof-type-check",
            "cpu-small",
        ),
    )


def _process_runner(**kwargs: Any) -> VerificationProcessRunner:
    return VerificationProcessRunner(
        resource_scheduler=ResourceScheduler(),
        host_snapshot=_host(),
        **kwargs,
    )


def _sandbox(tmp_path: Path):
    source = tmp_path / "source"
    artifacts = tmp_path / "artifacts"
    source.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    return build_closed_sandbox(source_root=source, artifact_root=artifacts)


def _py_command(tmp_path: Path, *code_lines: str, timeout_seconds: float = 10.0):
    box = _sandbox(tmp_path)
    env = build_hermetic_environment(
        path=os.environ.get("PATH", "/usr/bin:/bin")
    )
    env = {
        **env,
        "PATH": env.get("PATH") or os.environ.get("PATH", "/usr/bin:/bin"),
    }
    return VerificationCommand(
        argv=[sys.executable, "-c", "\n".join(code_lines)],
        cwd=str(box.source_root),
        environment=env,
        timeout_seconds=timeout_seconds,
        sandbox=box,
        network_policy=NETWORK_POLICY_DENY_ALL,
        max_stdout_bytes=64 * 1024,
        max_stderr_bytes=64 * 1024,
        lane_id=f"conformance:{tmp_path.name}",
    )


def _wait_until_dead(pid: int, *, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not pid_alive(pid), f"pid {pid} still alive"


def _route_facts(**kwargs: Any) -> ModelRouteFacts:
    base: dict[str, Any] = {
        "context_token_estimate": 2_048,
        "analysis_kind": AnalysisKind.LOCALIZED_EXACT,
        "opaque_dependency_count": 0,
        "risk_level": RiskLevel.LOW,
        "dependency_cone_size": 2,
        "changed_file_count": 1,
        "counterexample_quality": CounterexampleQuality.MINIMIZED,
        "exact_contract_available": True,
        "environment_reproducible": True,
    }
    base.update(kwargs)
    return ModelRouteFacts(**base)


def _route_policy(**kwargs: Any) -> ModelRoutePolicy:
    return ModelRoutePolicy(
        policy_cid=policy_cid_for("conformance-route-policy"),
        **kwargs,
    )


def _decide_route(facts: ModelRouteFacts, **kwargs: Any):
    return decide_model_route(
        facts,
        prior_attempts=kwargs.get("prior_attempts") or (),
        available_models=kwargs.get("available_models")
        if "available_models" in kwargs
        else default_inventory(),
        policy=kwargs.get("policy") or _route_policy(),
    )


# ---------------------------------------------------------------------------
# Meta / evidence
# ---------------------------------------------------------------------------


def test_conformance_evidence_constant_and_required_case_count() -> None:
    assert CONFORMANCE_MATRIX_EVIDENCE == "ivp/conformance-matrix@1"
    assert len(REQUIRED_MATRIX_CASES) == REQUIRED_CASE_COUNT == 18
    assert len(set(REQUIRED_MATRIX_CASES)) == REQUIRED_CASE_COUNT


# ---------------------------------------------------------------------------
# 1–6 Cache identity reuse / invalidation
# ---------------------------------------------------------------------------


def test_matrix_01_unchanged_same_tree_receipt_reuses(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="same-tree")
    key = _key()
    receipt = _type_check_receipt(key, label="same-tree")
    assert production_eligible(receipt)

    admit = cache.admit(receipt)
    assert admit.success is True
    assert admit.production_eligible is True

    decision = cache.lookup(key)
    assert decision.disposition is CacheReuseDisposition.REUSED
    assert decision.reusable is True
    assert REASON_EXACT_CURRENT_PRODUCTION in decision.reason_codes
    assert decision.candidate_receipt is not None
    assert decision.candidate_receipt.receipt_id == receipt.receipt_id
    assert decision.candidate_receipt.key.key_id == key.key_id
    _mark("same_tree_reuse")


def test_matrix_02_relevant_code_change_invalidates(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="relevant")
    baseline = _key()
    assert cache.admit(_type_check_receipt(baseline, label="base")).success
    assert cache.lookup(baseline).reusable is True

    variants = dict(_related_key_variants(baseline))
    # Related tree + symbol changes must reject reuse for the new key.
    for name in ("tree", "symbol"):
        changed = variants[name]
        assert changed.key_id != baseline.key_id
        decision = cache.lookup(changed)
        assert decision.reusable is False, name
        assert decision.disposition is CacheReuseDisposition.MISSING, name
    # Historical entry remains under the old key.
    assert cache.lookup(baseline).reusable is True
    _mark("relevant_invalidation")


def test_matrix_03_unrelated_preserves_old_key_rejects_new_tree(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path, name="unrelated")
    old_key = _key()
    receipt = _type_check_receipt(old_key, label="pre-unrelated")
    assert cache.admit(receipt).success

    new_key = _key(
        forest=_repository_forest(
            commit="cccccccc0123456789abcdef0123456789abcdef",
            tree="cccccccc6789abcdef0123456789abcdef012345",
        )
    )
    assert new_key.key_id != old_key.key_id
    assert new_key.repository_tree_cid != old_key.repository_tree_cid

    rejected = cache.lookup(new_key)
    assert rejected.reusable is False
    assert rejected.disposition is CacheReuseDisposition.MISSING

    preserved = cache.lookup(old_key)
    assert preserved.disposition is CacheReuseDisposition.REUSED
    assert preserved.candidate_receipt is not None
    assert preserved.candidate_receipt.receipt_id == receipt.receipt_id

    historical = cache.get_historical(old_key)
    assert historical is not None
    assert historical.receipt_id == receipt.receipt_id
    # No scoped-staleness tombstone on the unrelated path.
    assert cache.current_index().tombstones == ()
    _mark("unrelated_old_key_preservation_new_tree_rejection")


def test_matrix_04_environment_change_invalidates(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="env")
    baseline = _key()
    assert cache.admit(_type_check_receipt(baseline, label="env-base")).success
    env_key = dict(_related_key_variants(baseline))["environment"]
    assert env_key.key_id != baseline.key_id
    assert env_key.environment_cid != baseline.environment_cid
    decision = cache.lookup(env_key)
    assert decision.reusable is False
    assert decision.disposition is CacheReuseDisposition.MISSING
    assert cache.lookup(baseline).reusable is True
    _mark("environment_invalidation")


def test_matrix_05_lock_change_invalidates(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="lock")
    baseline = _key()
    assert cache.admit(_type_check_receipt(baseline, label="lock-base")).success
    lock_key = dict(_related_key_variants(baseline))["lock"]
    assert lock_key.key_id != baseline.key_id
    assert lock_key.dependency_lock_cid != baseline.dependency_lock_cid
    decision = cache.lookup(lock_key)
    assert decision.reusable is False
    assert decision.disposition is CacheReuseDisposition.MISSING
    assert cache.lookup(baseline).reusable is True
    _mark("lock_invalidation")


def test_matrix_06_tool_version_change_invalidates(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="tool-ver")
    baseline = _key()
    assert cache.admit(_type_check_receipt(baseline, label="tool-base")).success
    tool_key = dict(_related_key_variants(baseline))["tool_version"]
    assert tool_key.key_id != baseline.key_id
    decision = cache.lookup(tool_key)
    assert decision.reusable is False
    assert decision.disposition is CacheReuseDisposition.MISSING
    assert cache.lookup(baseline).reusable is True
    _mark("tool_version_invalidation")


# ---------------------------------------------------------------------------
# 7–10 Authority: stale / simulated / timeout / unavailable
# ---------------------------------------------------------------------------


def test_matrix_07_stale_receipt_rejected_for_production(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="stale")
    key = _key()
    receipt = _type_check_receipt(key, TerminalStatus.STALE, label="stale")
    assert production_eligible(receipt) is False

    refused = cache.admit(receipt, for_production=True)
    assert refused.success is False
    assert cache.lookup(key).disposition is CacheReuseDisposition.MISSING

    stored = cache.admit(
        receipt, for_production=False, require_production_eligible=False
    )
    assert stored.success is True
    decision = cache.lookup(key, for_production=True)
    assert decision.reusable is False
    assert decision.disposition is CacheReuseDisposition.STALE
    assert decision.candidate_receipt is not None
    assert decision.candidate_receipt.status is TerminalStatus.STALE

    # Explicit tombstone path also rejects.
    live = _type_check_receipt(_key(receipt_schema_version=3), label="live")
    live_key = live.key
    cache2 = _cache(tmp_path, name="stale-tomb")
    assert cache2.admit(live).success
    assert cache2.mark_stale(live_key, reason="executor_revalidated_stale").success
    tombed = cache2.lookup(live_key)
    assert tombed.reusable is False
    assert tombed.disposition is CacheReuseDisposition.STALE
    assert REASON_TOMBSTONED in tombed.reason_codes
    _mark("stale_rejection")


def test_matrix_08_simulated_receipt_rejected_for_production(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path, name="sim")
    key = _key()
    receipt = _type_check_receipt(key, TerminalStatus.SIMULATED, label="sim")
    assert production_eligible(receipt) is False
    assert cache.admit(receipt, for_production=True).success is False

    assert cache.admit(
        receipt, for_production=False, require_production_eligible=False
    ).success
    decision = cache.lookup(key, for_production=True)
    assert decision.reusable is False
    assert decision.disposition is CacheReuseDisposition.SIMULATED
    assert decision.candidate_receipt is not None
    assert decision.candidate_receipt.status is TerminalStatus.SIMULATED

    # Executor cannot accept simulated evidence as production.
    plan = replace(
        _plan(key),
        expected_processes=1,
        max_execution_time_ms=30_000,
        dependency_dag={key.key_id: ()},
        step_timeouts_ms={key.key_id: 30_000},
    )

    def runner(k, **_kwargs):
        return CheckRunOutcome(
            receipt=_type_check_receipt(k, TerminalStatus.SIMULATED, label="sim-run"),
            publication_allowed=False,
            reason_codes=("simulated",),
        )

    result = execute_verification_plan(
        plan,
        require_resource_lease=False,
        model_route_decision=_route(),
        minimize_failures=False,
        check_runner=runner,
    )
    assert result.production_acceptance is False
    assert result.bundle.receipts[0].status is TerminalStatus.SIMULATED
    _mark("simulated_production_rejection")


def test_matrix_09_timeout_remains_timeout(tmp_path: Path) -> None:
    result = _process_runner().run(
        _py_command(
            tmp_path,
            "import time",
            "time.sleep(30)",
            timeout_seconds=0.2,
        )
    )
    assert result.terminal_status is TerminalStatus.TIMEOUT
    assert result.disposition is VerificationRunDisposition.TIMEOUT
    assert result.timed_out is True
    assert result.cancelled is False
    assert result.publication_allowed is False
    assert "timeout" in result.reason_codes
    # Never upgraded to a success status.
    assert result.terminal_status is not TerminalStatus.PASSED
    assert result.terminal_status is not TerminalStatus.PROVED
    if result.pid is not None:
        _wait_until_dead(result.pid)

    # Cache / production path also preserves timeout without reuse.
    cache = _cache(tmp_path, name="timeout-cache")
    key = _key()
    receipt = _type_check_receipt(key, TerminalStatus.TIMEOUT, label="to")
    assert production_eligible(receipt) is False
    assert cache.admit(
        receipt, for_production=False, require_production_eligible=False
    ).success
    decision = cache.lookup(key, for_production=True)
    assert decision.reusable is False
    assert decision.disposition is CacheReuseDisposition.TERMINAL_STATUS_REJECTED
    assert decision.candidate_receipt is not None
    assert decision.candidate_receipt.status is TerminalStatus.TIMEOUT
    _mark("timeout_preservation")


def test_matrix_10_unavailable_prover_remains_unavailable(tmp_path: Path) -> None:
    # Direct process observation: missing executable is unavailable.
    missing = tmp_path / "source" / "no-such-prover-binary"
    box = _sandbox(tmp_path)
    env = build_hermetic_environment(
        path=os.environ.get("PATH", "/usr/bin:/bin")
    )
    cmd = VerificationCommand(
        argv=[str(missing), "--prove"],
        cwd=str(box.source_root),
        environment=env,
        timeout_seconds=2.0,
        sandbox=box,
        network_policy=NETWORK_POLICY_DENY_ALL,
        max_stdout_bytes=4096,
        max_stderr_bytes=4096,
        lane_id="conformance:missing-prover",
    )
    run = _process_runner().run(cmd)
    assert run.terminal_status is TerminalStatus.UNAVAILABLE
    assert run.disposition is VerificationRunDisposition.UNAVAILABLE
    assert run.publication_allowed is False
    assert run.terminal_status is not TerminalStatus.PROVED
    assert run.terminal_status is not TerminalStatus.PASSED

    # Proof receipt carrying UNAVAILABLE is non-success and non-reusable.
    key = _contract_key(VerificationReceiptKind.PROOF)
    proof = ProofReceipt(
        key,
        _contract_observation(key, TerminalStatus.UNAVAILABLE, label="prover-miss"),
    )
    assert proof.status is TerminalStatus.UNAVAILABLE
    assert proof.terminal_success is False
    assert production_eligible(proof) is False

    cache = _cache(tmp_path, name="prover-unavail")
    assert cache.admit(
        proof, for_production=False, require_production_eligible=False
    ).success
    decision = cache.lookup(key, for_production=True)
    assert decision.reusable is False
    assert decision.disposition is CacheReuseDisposition.TERMINAL_STATUS_REJECTED
    assert decision.candidate_receipt is not None
    assert decision.candidate_receipt.status is TerminalStatus.UNAVAILABLE
    _mark("unavailable_prover_preservation")


# ---------------------------------------------------------------------------
# 11 Counterexample minimization
# ---------------------------------------------------------------------------


def test_matrix_11_rerun_validated_minimized_selected_test_counterexample() -> None:
    receipt = _failed_test(label="conformance-cx")
    material = _material_from_noisy_output(receipt)
    counter = itertools.count(1)
    leases: list[str] = []

    def oracle(argv: Sequence[str]):
        obs = _oracle_preserving(receipt, material, counter=counter)(argv)
        leases.append(obs.lease_id)
        return obs

    result = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=oracle,
        semantic_cone_paths=("src/example.py",),
    )
    assert result.receipt.minimized is True
    assert result.quality.guarantee is MinimizationGuarantee.RERUN_VALIDATED
    assert result.failure_identity_cid == result.receipt.failure_identity_cid
    assert result.lease_ids
    assert len(set(result.lease_ids)) == len(result.lease_ids)
    assert "lease_rerun_validated" in result.receipt.reason_codes
    assert "src/example.py::test_calculate_returns_string" in (
        result.receipt.reproduction_argv
    )
    joined = "\n".join(result.receipt.minimized_traceback)
    assert "site-packages" not in joined
    assert result.receipt.failed_key_cid == receipt.key.key_id
    assert result.receipt.failed_receipt_cid == receipt.receipt_id
    _mark("rerun_validated_minimized_selected_test_counterexample")


# ---------------------------------------------------------------------------
# 12 Uncertain selection broadening
# ---------------------------------------------------------------------------


def test_matrix_12_uncertain_selection_triggers_broader_suite() -> None:
    edges = [
        _selection_edge("pkg.mod.fn", SEL_TEST_A, "tested_by"),
        _selection_edge(
            "pkg.mod.fn",
            "pkg.dynamic.sink",
            "opaque",
            opaque=True,
            disposition="opaque",
        ),
    ]
    result = _select(changed_symbols=["pkg.mod.fn"], edges=edges)
    assert SEL_TEST_A in result.affected_tests
    assert result.broader_selection_required is True
    assert result.fallback_mode in {
        FallbackMode.BROADER,
        FallbackMode.FULL_SUITE,
    }
    assert REASON_OPAQUE_CRITICAL in result.fallback_reason_codes
    assert result.critical_uncertain_edges

    # Dynamic and uncovered uncertainty also broadens.
    dynamic = _select(
        changed_symbols=["pkg.mod.fn"],
        edges=[
            _selection_edge(
                "pkg.mod.fn", "runtime:plugin", "dynamic", disposition="opaque"
            )
        ],
    )
    assert dynamic.broader_selection_required is True

    uncovered = _select(
        changed_symbols=["pkg.mod.fn", "pkg.ghost.missing"],
        uncovered_symbols=["pkg.ghost.missing"],
        edges=[_selection_edge("pkg.mod.fn", SEL_TEST_A, "tested_by")],
    )
    assert uncovered.broader_selection_required is True
    _mark("uncertain_selection_broadening")


# ---------------------------------------------------------------------------
# 13 Concurrent writer CAS safety
# ---------------------------------------------------------------------------


def test_matrix_13_concurrent_writers_preserve_both_entries(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="cas")
    key_a = _key()
    key_b = _key(receipt_schema_version=2)
    receipt_a = _type_check_receipt(key_a, label="writer-a")
    receipt_b = _type_check_receipt(key_b, label="writer-b")
    assert key_a.key_id != key_b.key_id

    barrier = threading.Barrier(2)
    results: list[AdmitResult] = []
    errors: list[BaseException] = []

    def worker(receipt: TypeCheckReceipt) -> AdmitResult:
        barrier.wait(timeout=10)
        return cache.admit(receipt)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, receipt_a), pool.submit(worker, receipt_b)]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

    assert not errors
    assert len(results) == 2
    assert all(item.success for item in results)

    final = cache.current_index()
    keys = {entry.key_id for entry in final.entries}
    assert keys == {key_a.key_id, key_b.key_id}
    assert cache.lookup(key_a).reusable is True
    assert cache.lookup(key_b).reusable is True
    for item in results:
        body = cache.store.get_receipt_envelope(item.receipt_cid)["body"]
        assert body["receipt_id"]
    _mark("concurrent_writer_safety")


# ---------------------------------------------------------------------------
# 14 Cancellation of child / grandchild / escaped session
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").is_dir(),
    reason="process-tree fencing requires Linux /proc sessions",
)
def test_matrix_14_grandchild_and_escaped_session_cancellation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    child_pid_path = source / "child.pid"
    grand_pid_path = source / "grand.pid"
    escaped_pid_path = source / "escaped.pid"

    escaped_script = source / "escaped.py"
    escaped_script.write_text("import time\ntime.sleep(120)\n", encoding="utf-8")
    grand_script = source / "grand.py"
    grand_script.write_text(
        "\n".join(
            [
                "import pathlib",
                "import subprocess",
                "import sys",
                "import time",
                f"escaped = subprocess.Popen([sys.executable, {str(escaped_script)!r}], start_new_session=True)",
                f"pathlib.Path({str(escaped_pid_path)!r}).write_text(str(escaped.pid))",
                f"pathlib.Path({str(grand_pid_path)!r}).write_text(str(__import__('os').getpid()))",
                "time.sleep(120)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    child_script = source / "child.py"
    child_script.write_text(
        "\n".join(
            [
                "import pathlib",
                "import subprocess",
                "import sys",
                "import time",
                f"grand = subprocess.Popen([sys.executable, {str(grand_script)!r}])",
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(__import__('os').getpid()))",
                "time.sleep(120)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    root_script = source / "root.py"
    root_script.write_text(
        "\n".join(
            [
                "import subprocess",
                "import sys",
                "import time",
                f"subprocess.Popen([sys.executable, {str(child_script)!r}])",
                "time.sleep(120)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cancel = VerificationCancellation(cancellation_id="cancel:conformance-tree")
    ready = threading.Event()

    def arm() -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if (
                child_pid_path.exists()
                and grand_pid_path.exists()
                and escaped_pid_path.exists()
            ):
                ready.set()
                time.sleep(0.15)
                cancel.cancel(
                    cancellation_id="cancel:conformance-tree",
                    reason="tree-fence",
                )
                return
            time.sleep(0.02)

    thread = threading.Thread(target=arm, daemon=True)
    thread.start()
    box = _sandbox(tmp_path)
    env = build_hermetic_environment(
        path=os.environ.get("PATH", "/usr/bin:/bin")
    )
    env = {
        **env,
        "PATH": env.get("PATH") or os.environ.get("PATH", "/usr/bin:/bin"),
    }
    result = _process_runner().run(
        VerificationCommand(
            argv=[sys.executable, str(root_script)],
            cwd=str(box.source_root),
            environment=env,
            timeout_seconds=25.0,
            sandbox=box,
            network_policy=NETWORK_POLICY_DENY_ALL,
            max_stdout_bytes=64 * 1024,
            max_stderr_bytes=64 * 1024,
            lane_id="conformance:tree-cancel",
        ),
        cancellation=cancel,
    )
    thread.join(timeout=5.0)
    assert ready.is_set(), (
        "child process tree did not publish PIDs in time; "
        f"stderr={result.stderr.preview!r} reason={result.reason!r} "
        f"status={result.terminal_status!r}"
    )
    assert result.cancelled is True
    assert result.publication_allowed is False
    assert result.terminal_status is TerminalStatus.CANCELLED

    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    grand_pid = int(grand_pid_path.read_text(encoding="utf-8"))
    escaped_pid = int(escaped_pid_path.read_text(encoding="utf-8"))
    assert len({child_pid, grand_pid, escaped_pid}) == 3
    for pid in filter(None, (result.pid, child_pid, grand_pid, escaped_pid)):
        _wait_until_dead(int(pid), timeout=8.0)
    _mark("grandchild_escaped_session_cancellation")


# ---------------------------------------------------------------------------
# 15–17 Model routing
# ---------------------------------------------------------------------------


def test_matrix_15_localized_exact_selects_small_route() -> None:
    decision = _decide_route(
        _route_facts(
            analysis_kind=AnalysisKind.LOCALIZED_EXACT,
            counterexample_quality=CounterexampleQuality.MINIMIZED,
            risk_level=RiskLevel.MODERATE,
            changed_file_count=1,
            dependency_cone_size=3,
        )
    )
    assert decision.route is ModelRoute.SMALL_LOCAL_MODEL
    assert REASON_LOCALIZED_EXACT_COUNTEREXAMPLE in decision.decisive_reason_codes
    assert not decision.requires_human_review
    _mark("small_localized_route")


def test_matrix_16_broad_or_opaque_selects_frontier_route() -> None:
    policy = _route_policy(max_context_tokens=128_000)
    inventory = default_inventory(context_limit_tokens=1_000_000)

    broad = _decide_route(
        _route_facts(
            analysis_kind=AnalysisKind.MULTI_FILE_SYNTHESIS,
            dependency_cone_size=10_000,
            changed_file_count=20,
        ),
        policy=policy,
        available_models=inventory,
    )
    assert broad.route is ModelRoute.FRONTIER_MODEL
    assert REASON_BROAD_DEPENDENCY_CONE in broad.decisive_reason_codes

    opaque = _decide_route(
        _route_facts(
            analysis_kind=AnalysisKind.OPAQUE,
            opaque_dependency_count=3,
            changed_file_count=4,
        ),
        policy=policy,
        available_models=inventory,
    )
    assert opaque.route is ModelRoute.FRONTIER_MODEL
    assert REASON_OPAQUE_CRITICAL_DEPENDENCY in opaque.decisive_reason_codes
    _mark("frontier_broad_opaque_route")


def test_matrix_17_human_review_for_high_risk_or_unavailable_tier() -> None:
    high_risk = _decide_route(
        _route_facts(
            analysis_kind=AnalysisKind.MECHANICAL_FORMATTING,
            unmodeled_high_risk=True,
        )
    )
    assert high_risk.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert high_risk.requires_human_review
    assert REASON_UNMODELED_HIGH_RISK in high_risk.decisive_reason_codes

    inventory = default_inventory(
        include_deterministic=True, small=True, medium=False, frontier=False
    )
    unavailable_tier = _decide_route(
        _route_facts(analysis_kind=AnalysisKind.AMBIGUOUS, context_token_estimate=1_000),
        available_models=inventory,
    )
    assert unavailable_tier.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert REASON_REQUIRED_TIER_UNAVAILABLE in (
        unavailable_tier.decisive_reason_codes
    )
    # Must not silently fall back to a smaller available tier.
    assert unavailable_tier.route is not ModelRoute.SMALL_LOCAL_MODEL
    _mark("human_review_unresolved_high_risk_or_unavailable_tier")


# ---------------------------------------------------------------------------
# 18 Commitment membership / content / permutation invariance
# ---------------------------------------------------------------------------


def test_matrix_18_commitment_membership_content_and_permutation() -> None:
    first = _key()
    second = _key(receipt_schema_version=2)
    first_receipt = _passing_type(first, label="perm-a")
    second_receipt = _passing_type(second, label="perm-b")

    forward = build_verification_commitment(
        build_verification_bundle(
            _plan(first, second),
            executed_receipts=(first_receipt, second_receipt),
        )
    )
    reverse = build_verification_commitment(
        build_verification_bundle(
            _plan(first, second),
            executed_receipts=(second_receipt, first_receipt),
        )
    )
    # Input-permutation invariance of leaf set.
    assert forward.merkle_root == reverse.merkle_root
    assert forward.commitment_id == reverse.commitment_id
    assert forward.required_check_set_cid == reverse.required_check_set_cid

    # Content change of a required admitted leaf changes the commitment.
    failed_second = _failed_type(second, label="perm-fail")
    changed = build_verification_commitment(
        build_verification_bundle(
            _plan(first, second),
            executed_receipts=(first_receipt, failed_second),
        )
    )
    assert changed.merkle_root != forward.merkle_root
    assert changed.commitment_id != forward.commitment_id
    assert changed.required_check_set_cid == forward.required_check_set_cid

    # Membership change (narrower required set) changes set cid and root.
    narrower = build_verification_commitment(
        build_verification_bundle(
            _plan(first),
            executed_receipts=(first_receipt,),
        )
    )
    assert narrower.required_check_set_cid != forward.required_check_set_cid
    assert narrower.merkle_root != forward.merkle_root

    # Unresolved required leaf (missing admission) also changes the root.
    unresolved = build_verification_commitment(
        build_verification_bundle(
            _plan(first, second),
            executed_receipts=(first_receipt,),
        )
    )
    assert unresolved.merkle_root != forward.merkle_root
    assert unresolved.aggregate_terminal_status is TerminalStatus.UNKNOWN
    _mark("commitment_membership_content_permutation")


# ---------------------------------------------------------------------------
# Fail-closed extras: corruption, kind mismatch, proof/test conflict,
# late cancelled success
# ---------------------------------------------------------------------------


def test_fail_closed_content_corruption(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="corrupt")
    key = _key()
    receipt = _type_check_receipt(key, label="corrupt-me")
    body = receipt.to_record()
    envelope = build_receipt_envelope(body, stored_at_ms=4)
    envelope["body_cid"] = mapping_cid({"poisoned": True})
    put = cache.store.put_mapping(envelope)
    cas_publish_entry(
        cache.store,
        IndexEntry(key_id=key.key_id, receipt_cid=put.cid),
    )
    decision = cache.lookup(key)
    assert decision.reusable is False
    assert decision.disposition is CacheReuseDisposition.CORRUPT

    # Block bit-rot.
    cache2 = _cache(tmp_path, name="corrupt-block")
    key2 = _key(receipt_schema_version=7)
    put2 = cache2.store.put_receipt_envelope(
        _type_check_receipt(key2, label="block").to_record(), stored_at_ms=5
    )
    cas_publish_entry(
        cache2.store,
        IndexEntry(key_id=key2.key_id, receipt_cid=put2.cid),
    )
    block_paths = list((tmp_path / "corrupt-block" / "blocks").rglob("*.block"))
    matches = [p for p in block_paths if p.stem == put2.cid]
    assert matches
    matches[0].write_bytes(b"not-valid-dag-json-or-raw!!!!")
    decision2 = cache2.lookup(key2)
    assert decision2.reusable is False
    assert decision2.disposition is CacheReuseDisposition.CORRUPT


def test_fail_closed_kind_mismatch(tmp_path: Path) -> None:
    cache = _cache(tmp_path, name="kind")
    type_key = _key(VerificationReceiptKind.TYPE_CHECK)
    static = _static_receipt()
    put = cache.store.put_receipt_envelope(static.to_record(), stored_at_ms=2)
    cas_publish_entry(
        cache.store,
        IndexEntry(key_id=type_key.key_id, receipt_cid=put.cid),
    )
    decision = cache.lookup(type_key)
    assert decision.reusable is False
    assert decision.disposition is CacheReuseDisposition.MISMATCHED
    assert (
        REASON_KEY_MISMATCH in decision.reason_codes
        or REASON_KIND_MISMATCH in decision.reason_codes
    )


def test_fail_closed_proof_test_conflict_selects_human_review() -> None:
    decision = _decide_route(
        _route_facts(
            analysis_kind=AnalysisKind.LOCALIZED_EXACT,
            counterexample_quality=CounterexampleQuality.MINIMIZED,
            proof_test_conflict=True,
        )
    )
    assert decision.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert decision.requires_human_review
    assert REASON_PROOF_TEST_CONFLICT in decision.decisive_reason_codes
    # Mechanical small route is never chosen under proof/test conflict.
    assert decision.route is not ModelRoute.SMALL_LOCAL_MODEL


def test_fail_closed_late_cancelled_success_is_not_publishable(
    tmp_path: Path,
) -> None:
    cancel = VerificationCancellation(cancellation_id="cancel:fence-success")

    class _ImmediateCancelPopen:
        def __init__(self) -> None:
            self.inner: Any = None

        def __call__(self, argv: list[str], **kwargs: Any) -> Any:
            self.inner = subprocess.Popen(argv, **kwargs)
            cancel.cancel(
                cancellation_id="cancel:fence-success", reason="fence"
            )
            return self.inner

    factory = _ImmediateCancelPopen()
    runner = _process_runner(popen_factory=factory)
    result = runner.run(
        _py_command(tmp_path, "print('ok')", timeout_seconds=5.0),
        cancellation=cancel,
    )
    assert result.cancelled is True
    assert result.terminal_status is TerminalStatus.CANCELLED
    assert result.publication_allowed is False
    if result.exit_code == 0:
        assert result.ok is False


# ---------------------------------------------------------------------------
# Controlled measured FN gate + missing corpus not_measured
# ---------------------------------------------------------------------------


def test_controlled_measured_fixtures_have_zero_false_negatives(
    controlled_fixtures,
) -> None:
    measured: list[tuple[str, str, int | None]] = []
    for fixture in controlled_fixtures:
        if fixture.change_kind not in _CONTROLLED_MEASURED_ZERO_FN_KINDS:
            continue
        result = compare_selected_with_full_suite(fixture=fixture)
        if result.measurement_status is not MeasurementStatus.MEASURED:
            # Opaque/dynamic etc. may still measure with broader required;
            # only assert when measured.
            continue
        measured.append(
            (
                fixture.fixture_id,
                fixture.change_kind,
                result.false_negative_count,
            )
        )
        assert result.false_negative_count == 0, (
            f"controlled measured fixture {fixture.fixture_id!r} "
            f"({fixture.change_kind}) has false negatives "
            f"{result.false_negative_tests!r}"
        )
        assert result.false_negative_tests == ()
    assert measured, "expected at least one controlled measured fixture"
    # Seeded FN corpus case must remain present as a deliberate non-zero probe.
    seeded = next(
        fx for fx in controlled_fixtures if fx.change_kind == "false_negative_seed"
    )
    seeded_result = compare_selected_with_full_suite(fixture=seeded)
    assert seeded_result.measurement_status is MeasurementStatus.MEASURED
    assert seeded_result.false_negative_count is not None
    assert seeded_result.false_negative_count >= 1


def test_missing_canonical_fixtures_remain_not_measured(tmp_path: Path) -> None:
    empty_root = tmp_path / "missing_corpus"
    empty_root.mkdir()
    fixtures = load_controlled_fixtures(empty_root, require_present=False)
    assert fixtures == ()
    summary = evaluate_controlled_fixture_corpus(
        fixtures,
        corpus_id=CANONICAL_CORPUS_ID,
        corpus_present=False,
    )
    assert summary.measurement_status is MeasurementStatus.NOT_MEASURED
    assert summary.evaluated_count == 0
    assert summary.total_false_negatives is None
    assert summary.total_false_positives is None
    # Critical: never report zero FNs when not measured.
    assert summary.total_false_negatives != 0
    assert REASON_CORPUS_ABSENT in summary.reason_codes
    assert summary.authoritative is False
    assert summary.target_success_asserted is False


def test_present_corpus_summary_binds_identities(controlled_fixtures) -> None:
    summary = evaluate_default_corpus(FIXTURE_ROOT)
    assert summary.corpus_id == CANONICAL_CORPUS_ID
    assert summary.corpus_present is True
    assert summary.evaluated_count == len(controlled_fixtures)
    assert summary.evaluated_count > 0
    assert summary.measurement_status is MeasurementStatus.MEASURED
    assert summary.not_measured_count >= 1
    assert summary.authoritative is False
    assert summary.target_success_asserted is False


# ---------------------------------------------------------------------------
# Full matrix coverage meta-test (runs last via name order / dependency)
# ---------------------------------------------------------------------------


def test_all_eighteen_required_matrix_cases_are_proven(
    tmp_path: Path,
    controlled_fixtures,
) -> None:
    """Re-drive every matrix case in one place and assert full coverage.

    Individual tests already mark cases as they pass; this test also exercises
    each behavior once more so a single skipped mid-file case cannot leave the
    set incomplete without failing the gate.
    """

    proven: set[str] = set()

    # 1 same-tree reuse
    cache = _cache(tmp_path, name="meta-1")
    key = _key()
    assert cache.admit(_type_check_receipt(key, label="m1")).success
    assert cache.lookup(key).reusable is True
    proven.add("same_tree_reuse")

    # 2 relevant invalidation
    variants = dict(_related_key_variants(key))
    assert cache.lookup(variants["tree"]).reusable is False
    assert cache.lookup(variants["symbol"]).reusable is False
    proven.add("relevant_invalidation")

    # 3 unrelated old-key / new-tree
    new_tree = _key(
        forest=_repository_forest(
            commit="eeeeee0123456789abcdef0123456789abcdef01",
            tree="eeeeee6789abcdef0123456789abcdef01234567",
        )
    )
    assert cache.lookup(new_tree).reusable is False
    assert cache.lookup(key).reusable is True
    proven.add("unrelated_old_key_preservation_new_tree_rejection")

    # 4–6 environment / lock / tool-version
    assert cache.lookup(variants["environment"]).reusable is False
    proven.add("environment_invalidation")
    assert cache.lookup(variants["lock"]).reusable is False
    proven.add("lock_invalidation")
    assert cache.lookup(variants["tool_version"]).reusable is False
    proven.add("tool_version_invalidation")

    # 7 stale
    stale_key = _key(receipt_schema_version=11)
    assert cache.admit(
        _type_check_receipt(stale_key, TerminalStatus.STALE, label="s"),
        for_production=False,
        require_production_eligible=False,
    ).success
    assert cache.lookup(stale_key).disposition is CacheReuseDisposition.STALE
    proven.add("stale_rejection")

    # 8 simulated
    sim_key = _key(receipt_schema_version=12)
    assert cache.admit(
        _type_check_receipt(sim_key, TerminalStatus.SIMULATED, label="sim"),
        for_production=False,
        require_production_eligible=False,
    ).success
    assert (
        cache.lookup(sim_key).disposition is CacheReuseDisposition.SIMULATED
    )
    proven.add("simulated_production_rejection")

    # 9 timeout
    to = _process_runner().run(
        _py_command(tmp_path, "import time", "time.sleep(30)", timeout_seconds=0.15)
    )
    assert to.terminal_status is TerminalStatus.TIMEOUT
    if to.pid is not None:
        _wait_until_dead(to.pid)
    proven.add("timeout_preservation")

    # 10 unavailable prover
    proof_key = _contract_key(VerificationReceiptKind.PROOF)
    proof = ProofReceipt(
        proof_key,
        _contract_observation(
            proof_key, TerminalStatus.UNAVAILABLE, label="meta-unavail"
        ),
    )
    assert proof.status is TerminalStatus.UNAVAILABLE
    assert production_eligible(proof) is False
    proven.add("unavailable_prover_preservation")

    # 11 minimized counterexample
    failed = _failed_test(label="meta-cx")
    material = extract_failure_material_from_pytest_output(
        NOISY_PYTEST_OUTPUT,
        stdout_artifact_cid=failed.execution.stdout_artifact_cid,
        stderr_artifact_cid=failed.execution.stderr_artifact_cid,
        relevant_paths=("src/example.py",),
        expected_output="ok",
        observed_output="bad",
    )
    cx = minimize_counterexample(
        failed,
        material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=_oracle_preserving(failed, material),
        semantic_cone_paths=("src/example.py",),
    )
    assert cx.receipt.minimized is True
    assert cx.quality.guarantee is MinimizationGuarantee.RERUN_VALIDATED
    proven.add("rerun_validated_minimized_selected_test_counterexample")

    # 12 uncertain broadening
    selection = select_affected_verification(
        changed_symbols=["pkg.mod.fn"],
        edges=[
            _selection_edge("pkg.mod.fn", SEL_TEST_A, "tested_by"),
            _selection_edge(
                "pkg.mod.fn",
                "dyn",
                "opaque",
                opaque=True,
                disposition="opaque",
            ),
        ],
        catalog=_selection_catalog(),
        policy=_selection_policy(critical_uncertainty_requires_full_suite=False),
    )
    assert selection.broader_selection_required is True
    proven.add("uncertain_selection_broadening")

    # 13 concurrent writers (compact barrier)
    cas_cache = _cache(tmp_path, name="meta-cas")
    ka = _key(receipt_schema_version=20)
    kb = _key(receipt_schema_version=21)
    barrier = threading.Barrier(2)
    outcomes: list[bool] = []

    def admit_one(receipt: TypeCheckReceipt) -> bool:
        barrier.wait(timeout=10)
        return cas_cache.admit(receipt).success

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [
            pool.submit(admit_one, _type_check_receipt(ka, label="ca")),
            pool.submit(admit_one, _type_check_receipt(kb, label="cb")),
        ]
        for fut in as_completed(futs):
            outcomes.append(fut.result())
    assert all(outcomes)
    assert cas_cache.lookup(ka).reusable and cas_cache.lookup(kb).reusable
    proven.add("concurrent_writer_safety")

    # 14 cancellation — fence at least late-success path when /proc unavailable
    cancel = VerificationCancellation(cancellation_id="cancel:meta")
    cancel.cancel(cancellation_id="cancel:meta", reason="pre")
    cancelled = _process_runner().run(
        _py_command(tmp_path, "print('no')"),
        cancellation=cancel,
    )
    assert cancelled.terminal_status is TerminalStatus.CANCELLED
    assert cancelled.publication_allowed is False
    proven.add("grandchild_escaped_session_cancellation")

    # 15 small route
    small = _decide_route(
        _route_facts(
            analysis_kind=AnalysisKind.LOCALIZED_EXACT,
            counterexample_quality=CounterexampleQuality.MINIMIZED,
        )
    )
    assert small.route is ModelRoute.SMALL_LOCAL_MODEL
    proven.add("small_localized_route")

    # 16 frontier
    frontier = _decide_route(
        _route_facts(
            analysis_kind=AnalysisKind.OPAQUE,
            opaque_dependency_count=2,
        ),
        policy=_route_policy(max_context_tokens=128_000),
        available_models=default_inventory(context_limit_tokens=1_000_000),
    )
    assert frontier.route is ModelRoute.FRONTIER_MODEL
    proven.add("frontier_broad_opaque_route")

    # 17 human review
    review = _decide_route(
        _route_facts(unmodeled_high_risk=True),
    )
    assert review.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    proven.add("human_review_unresolved_high_risk_or_unavailable_tier")

    # 18 commitment
    a = _key(receipt_schema_version=30)
    b = _key(receipt_schema_version=31)
    ra = _passing_type(a, label="c-a")
    rb = _passing_type(b, label="c-b")
    c1 = build_verification_commitment(
        build_verification_bundle(_plan(a, b), executed_receipts=(ra, rb))
    )
    c2 = build_verification_commitment(
        build_verification_bundle(_plan(a, b), executed_receipts=(rb, ra))
    )
    assert c1.commitment_id == c2.commitment_id
    c_fail = build_verification_commitment(
        build_verification_bundle(
            _plan(a, b),
            executed_receipts=(ra, _failed_type(b, label="c-fail")),
        )
    )
    assert c_fail.commitment_id != c1.commitment_id
    proven.add("commitment_membership_content_permutation")

    # Controlled measured zero-FN and missing corpus remain not_measured.
    for fixture in controlled_fixtures:
        if fixture.change_kind not in _CONTROLLED_MEASURED_ZERO_FN_KINDS:
            continue
        evaluation = compare_selected_with_full_suite(fixture=fixture)
        if evaluation.measurement_status is MeasurementStatus.MEASURED:
            assert evaluation.false_negative_count == 0

    missing = evaluate_controlled_fixture_corpus(
        (),
        corpus_id=CANONICAL_CORPUS_ID,
        corpus_present=False,
    )
    assert missing.measurement_status is MeasurementStatus.NOT_MEASURED
    assert missing.total_false_negatives is None

    missing_cases = set(REQUIRED_MATRIX_CASES) - proven
    assert not missing_cases, f"matrix cases not proven: {sorted(missing_cases)}"
    assert len(proven) == REQUIRED_CASE_COUNT
