"""Goal coverage and analyzer receipt tests (PTR-111)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    ObjectiveGoal,
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_goal_evidence import (
    ACCEPTANCE_COVERAGE_INTERFACE,
    ACCEPTANCE_COVERAGE_RECEIPT_INTERFACE,
    ANALYZER_HEALTH_INTERFACE,
    DEFAULT_CHANNEL_PROOF_REVISION,
    DEFAULT_PRODUCER_CHANNEL,
    EXHAUSTION_QUORUM_INTERFACE,
    PRODUCTION_WARM_REQUIREMENT_IDS,
    PRODUCING_TASK_ID,
    PROOF_REUSE_ANALYZER_RECEIPT_INTERFACE,
    PROOF_REUSE_BENCHMARK_RECEIPT_INTERFACE,
    PROOF_REUSE_POPULATION_RECEIPT_INTERFACE,
    PROOF_REUSE_ROLLBACK_DECISION_INTERFACE,
    PROOF_TEST_REUSE_GOAL_EVIDENCE_INTERFACE,
    REAL_ZK_REQUIREMENT_IDS,
    REQUIRED_ADVERSARIAL_POPULATIONS,
    REQUIRED_ANALYZER_CHANNELS,
    REQUIRED_QUORUM_MEMBERS,
    TEST_CERTIFICATE_ASSURANCE_RECEIPT_INTERFACE,
    AcceptanceCoverageReceipt,
    CoverageStatus,
    GoalAssuranceResult,
    GoalAssuranceRunner,
    GoalEvidenceGap,
    GoalEvidenceGapKind,
    GoalQuorumMember,
    ProofReuseAnalyzerReceipt,
    ProofReusePopulationReceipt,
    ProofTestReuseGoalEvidence,
    ProofTestReuseGoalEvidenceError,
    discover_requirement_ids_from_heap,
    goal_requirements_by_id,
)

NOW = 1_786_000_000.0
NOW_MS = int(NOW * 1_000)
REPO = "repository:sha256:current"
STATE = "baguqeera-state-current"
COMMIT = "f" * 40
TREE = "e" * 40
GITLINKS = "baguqeera-gitlinks-current"
FOREST = "baguqeera-forest-current"
OVERLAY = "baguqeera-overlay-clean"
POLICY = "baguqeera-policy-current"
CAPABILITY = "baguqeera-capability-current"
KEY = "baguqeera-key-current"
CIRCUIT = "baguqeera-circuit-current"
OBJECTIVE = "baguqeera-objective-current"
CHANNEL = DEFAULT_PRODUCER_CHANNEL
CHANNEL_REV = DEFAULT_CHANNEL_PROOF_REVISION

OBJECTIVES_PATH = (
    Path(__file__).resolve().parents[4]
    / "implementation_plan"
    / "docs"
    / "46-proof-backed-test-reuse.objectives.md"
)
# Workspace layout: test lives under external/ipfs_accelerate/test/api/
# so parents[4] may overshoot. Fall back to repo-relative discovery.
if not OBJECTIVES_PATH.is_file():
    for parent in Path(__file__).resolve().parents:
        candidate = (
            parent
            / "implementation_plan"
            / "docs"
            / "46-proof-backed-test-reuse.objectives.md"
        )
        if candidate.is_file():
            OBJECTIVES_PATH = candidate
            break


MINI_HEAP = """
## PTR-G010 Contracts

- Status: active
- Parent: PTR-G000
- Evidence: ptr/test-execution-contracts@1, ptr/reuse-authority-policy@1
- Acceptance criteria: ptr/test-execution-contracts@1; ptr/reuse-authority-policy@1
- Acceptance: contracts reject nonfinite inputs
- Validation: pytest -q
- Gap task: PTR-001
- Refinement: separate schemas
- Embedding query: contracts
- AST query: dataclasses
- Goal: Define contracts
- Outputs: contracts.py
- Fib priority: 2
- Priority: P0
- Track: foundation
- Bundle: foundation

## PTR-G100 Adversarial

- Status: active
- Parent: PTR-G000
- Evidence: ptr/degradation-matrix@1, ptr/invalidation-mutation-population@1, ptr/cross-repo-direct-node-conformance@1, ptr/security-concurrency-population@1
- Acceptance criteria: ptr/degradation-matrix@1; ptr/invalidation-mutation-population@1; ptr/cross-repo-direct-node-conformance@1; ptr/security-concurrency-population@1
- Acceptance: zero false skips
- Validation: pytest -q
- Gap task: PTR-090
- Refinement: populations
- Embedding query: adversarial
- AST query: plugin
- Goal: Adversarial assurance
- Outputs: tests.py
- Fib priority: 8
- Priority: P0
- Track: adversarial
- Bundle: adversarial
"""


def _runner(**overrides: Any) -> GoalAssuranceRunner:
    values: dict[str, Any] = {
        "repository_id": REPO,
        "repository_state_cid": STATE,
        "git_commit_id": COMMIT,
        "git_tree_id": TREE,
        "gitlink_state_cid": GITLINKS,
        "repository_forest_cid": FOREST,
        "dirty": False,
        "dirty_overlay_cid": OVERLAY,
        "objective_revision": OBJECTIVE,
        "policy_cid": POLICY,
        "capability_cid": CAPABILITY,
        "verifying_key_cid": KEY,
        "circuit_cid": CIRCUIT,
        "clock": lambda: NOW,
    }
    values.update(overrides)
    return GoalAssuranceRunner(**values)


def _base_binding() -> dict[str, Any]:
    return {
        "repository_id": REPO,
        "repository_state_cid": STATE,
        "git_commit_id": COMMIT,
        "git_tree_id": TREE,
        "gitlink_state_cid": GITLINKS,
        "repository_forest_cid": FOREST,
        "dirty": False,
        "dirty_overlay_cid": OVERLAY,
        "objective_revision": OBJECTIVE,
        "policy_cid": POLICY,
        "capability_cid": CAPABILITY,
        "verifying_key_cid": KEY,
        "circuit_cid": CIRCUIT,
        "producer_channel": CHANNEL,
        "channel_proof_revision": CHANNEL_REV,
        "observed_at_ms": NOW_MS - 1_000,
        "fresh_until_ms": NOW_MS + 30_000,
        "passed": True,
        "status": "passed",
    }


def _validation(requirement_id: str, **overrides: Any) -> dict[str, Any]:
    record = {
        **_base_binding(),
        "requirement_id": requirement_id,
        "validation_command": (
            "IPFS_TEST_PROOF_REUSE_MODE=off python3 -m pytest -q"
        ),
    }
    record.update(overrides)
    return record


def _analyzers() -> list[dict[str, Any]]:
    return [
        {
            **_base_binding(),
            "analyzer_id": analyzer_id,
            "producer_channel": f"analyzer:{analyzer_id}",
            "healthy": True,
            "exhaustive": True,
            "conclusive": True,
        }
        for analyzer_id in sorted(REQUIRED_ANALYZER_CHANNELS)
    ]


def _populations() -> list[dict[str, Any]]:
    return [
        {
            **_base_binding(),
            "population_id": population_id,
            "producer_channel": f"adversarial:{population_id}",
            "passed": True,
            "false_skips": 0,
        }
        for population_id in sorted(REQUIRED_ADVERSARIAL_POPULATIONS)
    ]


def _quorum() -> list[dict[str, Any]]:
    return [
        {
            "member_id": "exhaustive-scan",
            "evidence_channel": "static-dependency-exhaustive",
            "receipt_cid": "baguqeera-quorum-exhaustive",
            "healthy": True,
            "exhaustive": True,
            "conclusive": True,
            "fresh": True,
            "uncontradicted": True,
            "observed_at_ms": NOW_MS - 500,
            "fresh_until_ms": NOW_MS + 30_000,
        },
        {
            "member_id": "audit-scan",
            "evidence_channel": "independent-audit",
            "receipt_cid": "baguqeera-quorum-audit",
            "healthy": True,
            "exhaustive": True,
            "conclusive": True,
            "fresh": True,
            "uncontradicted": True,
            "observed_at_ms": NOW_MS - 400,
            "fresh_until_ms": NOW_MS + 30_000,
        },
    ]


def _capabilities_all_available() -> dict[str, Any]:
    return {
        name: {"available": True, "status": "available"}
        for name in ("groth16", "provekit", "cache", "ipfs")
    }


def _gap_kinds(result: GoalAssuranceResult) -> set[GoalEvidenceGapKind]:
    return {gap.kind for gap in result.gaps}


def _full_collect(
    heap: str = MINI_HEAP,
    *,
    capability_facts: dict[str, Any] | None = None,
    certificate_assurance: Any = None,
    benchmark_receipt: Any = None,
    requirement_registry: Any = None,
    mutate_validation: Any = None,
    mutate_population: Any = None,
    mutate_quorum: Any = None,
) -> GoalAssuranceResult:
    reqs = discover_requirement_ids_from_heap(heap)
    validations = {req: _validation(req) for req in reqs}
    if mutate_validation is not None:
        mutate_validation(validations)
    populations = _populations()
    if mutate_population is not None:
        mutate_population(populations)
    quorum = _quorum()
    if mutate_quorum is not None:
        mutate_quorum(quorum)
    return _runner().collect(
        heap,
        validation_by_requirement=validations,
        analyzer_inputs=_analyzers(),
        population_inputs=populations,
        quorum_inputs=quorum,
        capability_facts=capability_facts
        if capability_facts is not None
        else _capabilities_all_available(),
        certificate_assurance=certificate_assurance,
        benchmark_receipt=benchmark_receipt,
        requirement_registry=requirement_registry,
    )


# ---------------------------------------------------------------------------
# Interfaces + discovery
# ---------------------------------------------------------------------------


def test_interfaces_and_constants_are_stable() -> None:
    assert PROOF_TEST_REUSE_GOAL_EVIDENCE_INTERFACE == "ProofTestReuseGoalEvidence@1"
    assert ACCEPTANCE_COVERAGE_INTERFACE == "AcceptanceCoverage@1"
    assert ACCEPTANCE_COVERAGE_RECEIPT_INTERFACE == "AcceptanceCoverageReceipt@1"
    assert PROOF_REUSE_ANALYZER_RECEIPT_INTERFACE == "ProofReuseAnalyzerReceipt@1"
    assert (
        PROOF_REUSE_POPULATION_RECEIPT_INTERFACE == "ProofReusePopulationReceipt@1"
    )
    assert TEST_CERTIFICATE_ASSURANCE_RECEIPT_INTERFACE == (
        "TestCertificateAssuranceReceipt@1"
    )
    assert ANALYZER_HEALTH_INTERFACE == "AnalyzerHealth"
    assert EXHAUSTION_QUORUM_INTERFACE == "ExhaustionQuorum"
    assert PROOF_REUSE_BENCHMARK_RECEIPT_INTERFACE == "ProofReuseBenchmarkReceipt"
    assert PROOF_REUSE_ROLLBACK_DECISION_INTERFACE == "ProofReuseRollbackDecision"
    assert PRODUCING_TASK_ID == "PTR-111"
    assert REQUIRED_QUORUM_MEMBERS == 2
    assert REQUIRED_ANALYZER_CHANNELS == {
        "static-dependency",
        "runtime-dependency",
        "reuse-eligibility",
    }
    assert REQUIRED_ADVERSARIAL_POPULATIONS == {
        "mutation",
        "storage-security-concurrency",
        "cross-repository",
    }
    assert "ptr/real-zk-certificate-conformance@1" in REAL_ZK_REQUIREMENT_IDS
    assert "ptr/warm-reuse-benchmark@1" in PRODUCTION_WARM_REQUIREMENT_IDS


def test_requirement_ids_are_discovered_from_objective_heap_not_registry() -> None:
    assert OBJECTIVES_PATH.is_file(), f"missing objectives heap at {OBJECTIVES_PATH}"
    discovered = discover_requirement_ids_from_heap(OBJECTIVES_PATH)
    assert len(discovered) == 50
    assert len(set(discovered)) == 50
    assert all(item.startswith("ptr/") and item.endswith("@1") for item in discovered)

    by_goal = goal_requirements_by_id(OBJECTIVES_PATH)
    assert set(by_goal) == {
        "PTR-G000",
        "PTR-G010",
        "PTR-G020",
        "PTR-G030",
        "PTR-G040",
        "PTR-G050",
        "PTR-G060",
        "PTR-G070",
        "PTR-G080",
        "PTR-G090",
        "PTR-G100",
        "PTR-G110",
        "PTR-G120",
        "PTR-G130",
        "PTR-G140",
    }
    assert sum(len(items) for items in by_goal.values()) == 50

    # Explicit per-test registry is forbidden even when the heap is valid.
    result = _full_collect(
        MINI_HEAP, requirement_registry={"ptr/fake@1": "test_foo"}
    )
    assert GoalEvidenceGapKind.REQUIREMENT_REGISTRY_FORBIDDEN in _gap_kinds(result)
    assert not result.authoritative


def test_mini_heap_discovery_matches_declared_evidence() -> None:
    discovered = discover_requirement_ids_from_heap(MINI_HEAP)
    assert discovered == (
        "ptr/test-execution-contracts@1",
        "ptr/reuse-authority-policy@1",
        "ptr/degradation-matrix@1",
        "ptr/invalidation-mutation-population@1",
        "ptr/cross-repo-direct-node-conformance@1",
        "ptr/security-concurrency-population@1",
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_authoritative_result_binds_channels_identities_and_retained_bytes() -> None:
    result = _full_collect()
    assert result.authoritative
    assert result.authority == "authoritative"
    assert result.populations_passed
    assert result.analyzers_healthy
    assert result.quorum_satisfied
    assert not result.gaps
    assert set(result.required_requirement_ids) == set(
        discover_requirement_ids_from_heap(MINI_HEAP)
    )

    for receipt in result.coverage_receipts:
        assert receipt.verified
        assert receipt.producer_channel == CHANNEL
        assert receipt.channel_proof_revision == CHANNEL_REV
        assert receipt.repository_id == REPO
        assert receipt.git_tree_id == TREE
        assert receipt.repository_forest_cid == FOREST
        assert receipt.git_commit_id == COMMIT
        assert receipt.objective_revision == OBJECTIVE
        assert receipt.observed_at_ms < receipt.fresh_until_ms
        assert receipt.retained_validation_bytes_b64
        assert receipt.retained_validation_cid
        assert receipt.producing_task_id == "PTR-111"
        assert receipt.status is CoverageStatus.VERIFIED

    for analyzer in result.analyzer_receipts:
        assert analyzer.analyzer_id in REQUIRED_ANALYZER_CHANNELS
        assert analyzer.healthy and analyzer.exhaustive and analyzer.conclusive
        assert analyzer.producer_channel.startswith("analyzer:")
        assert analyzer.channel_proof_revision
        assert analyzer.retained_validation_cid
        assert analyzer.analyzer_health_interface == "AnalyzerHealth"

    for population in result.population_receipts:
        assert population.population_id in REQUIRED_ADVERSARIAL_POPULATIONS
        assert population.passed is True
        assert population.false_skips == 0
        assert population.producer_channel.startswith("adversarial:")
        assert population.retained_validation_cid

    assert len(result.quorum_members) >= 2
    member_ids = {item.member_id for item in result.quorum_members}
    channels = {item.evidence_channel for item in result.quorum_members}
    receipts = {item.receipt_cid for item in result.quorum_members}
    assert len(member_ids) >= 2
    assert len(channels) >= 2
    assert len(receipts) >= 2
    for member in result.quorum_members:
        assert member.admissible
        assert member.healthy
        assert member.exhaustive
        assert member.conclusive
        assert member.fresh
        assert member.uncontradicted

    assert set(result.evidence_by_goal) == {"PTR-G010", "PTR-G100"}
    for evidence in result.goal_evidence:
        assert evidence.authoritative
        assert evidence.status == "verified_complete"
        assert evidence.producer_channel == CHANNEL
        assert evidence.channel_proof_revision == CHANNEL_REV
        assert evidence.retained_validation_bytes_b64
        assert len(evidence.requirement_ids) == len(evidence.coverage_receipt_cids)


def test_receipts_round_trip_and_reject_tampering() -> None:
    result = _full_collect()
    replayed = GoalAssuranceResult.from_dict(result.to_record())
    assert replayed.content_id == result.content_id
    assert replayed.authoritative
    assert (
        replayed.coverage_receipts[0].content_id
        == result.coverage_receipts[0].content_id
    )

    tampered = deepcopy(result.to_record())
    tampered["coverage_receipts"][0]["git_tree_id"] = "forged"
    with pytest.raises(ProofTestReuseGoalEvidenceError):
        GoalAssuranceResult.from_dict(tampered)


# ---------------------------------------------------------------------------
# Adversarial populations + quorum
# ---------------------------------------------------------------------------


def test_all_three_adversarial_populations_must_pass_with_zero_false_skips() -> None:
    def fail_one(populations: list[dict[str, Any]]) -> None:
        populations[0]["passed"] = False

    failed = _full_collect(mutate_population=fail_one)
    assert GoalEvidenceGapKind.POPULATION_FAILED in _gap_kinds(failed)
    assert not failed.populations_passed
    assert not failed.authoritative

    def false_skip(populations: list[dict[str, Any]]) -> None:
        populations[1]["false_skips"] = 1

    skipped = _full_collect(mutate_population=false_skip)
    assert GoalEvidenceGapKind.FALSE_SKIP_DETECTED in _gap_kinds(skipped)
    assert not skipped.authoritative

    def drop_one(populations: list[dict[str, Any]]) -> None:
        populations.pop()

    missing = _full_collect(mutate_population=drop_one)
    assert GoalEvidenceGapKind.POPULATION_MISSING in _gap_kinds(missing)


def test_two_quorum_members_must_be_independent_healthy_exhaustive() -> None:
    def duplicate_channel(members: list[dict[str, Any]]) -> None:
        members[1]["evidence_channel"] = members[0]["evidence_channel"]

    dup = _full_collect(mutate_quorum=duplicate_channel)
    assert GoalEvidenceGapKind.QUORUM_NOT_INDEPENDENT in _gap_kinds(dup)
    assert not dup.quorum_satisfied

    def unhealthy(members: list[dict[str, Any]]) -> None:
        members[0]["healthy"] = False

    bad = _full_collect(mutate_quorum=unhealthy)
    assert GoalEvidenceGapKind.QUORUM_UNHEALTHY in _gap_kinds(bad)

    def one_member(members: list[dict[str, Any]]) -> None:
        del members[1]

    short = _full_collect(mutate_quorum=one_member)
    assert GoalEvidenceGapKind.QUORUM_INSUFFICIENT in _gap_kinds(short)

    def contradicted(members: list[dict[str, Any]]) -> None:
        members[1]["uncontradicted"] = False

    contrad = _full_collect(mutate_quorum=contradicted)
    assert GoalEvidenceGapKind.QUORUM_CONTRADICTED in _gap_kinds(contrad)


# ---------------------------------------------------------------------------
# Capabilities / real-ZK / synthetic benchmark
# ---------------------------------------------------------------------------


def test_unavailable_capabilities_are_typed_and_leave_real_zk_unverified() -> None:
    facts = {
        "groth16": {"available": False, "status": "missing"},
        "provekit": {"available": False, "status": "missing"},
        "cache": {"available": True, "status": "available"},
        "ipfs": {"available": True, "status": "available"},
    }
    # Mini heap has no real-ZK criteria — unavailable backends must not block
    # non-ZK authority.
    result = _full_collect(capability_facts=facts)
    assert "groth16" in result.unavailable_capabilities
    assert "provekit" in result.unavailable_capabilities
    assert result.authoritative

    # Heap that includes a real-ZK criterion.
    zk_heap = """
## PTR-G050 Datasets ZK

- Status: active
- Parent: PTR-G000
- Evidence: ptr/real-zk-certificate-conformance@1
- Acceptance criteria: ptr/real-zk-certificate-conformance@1
- Acceptance: real certificates bind public inputs
- Validation: pytest -q
- Gap task: PTR-041
- Refinement: conformance
- Embedding query: groth16
- AST query: certificate
- Goal: Real ZK
- Outputs: cert.py
- Fib priority: 3
- Priority: P0
- Track: datasets-zk
- Bundle: datasets-zk
"""
    unverified = _full_collect(zk_heap, capability_facts=facts)
    assert GoalEvidenceGapKind.REAL_ZK_UNVERIFIED in _gap_kinds(unverified)
    assert not unverified.authoritative
    assert "ptr/real-zk-certificate-conformance@1" not in {
        item.requirement_id
        for item in unverified.coverage_receipts
        if item.verified
    }

    # Reviewed real certificate restores authority even when backends are down.
    cert = {
        "interface": TEST_CERTIFICATE_ASSURANCE_RECEIPT_INTERFACE,
        "status": "verified",
        "authority": "authoritative",
        "backend": "groth16",
        "locally_verified": True,
        "verified": True,
    }
    restored = _full_collect(
        zk_heap, capability_facts=facts, certificate_assurance=cert
    )
    assert restored.authoritative
    assert "groth16" in restored.unavailable_capabilities


def test_unavailable_cache_leaves_production_warm_unverified() -> None:
    warm_heap = """
## PTR-G000 Root

- Status: active
- Evidence: ptr/warm-reuse-benchmark@1
- Acceptance criteria: ptr/warm-reuse-benchmark@1
- Acceptance: warm reuse
- Validation: pytest -q
- Gap task: PTR-100
- Refinement: bench
- Embedding query: warm
- AST query: cache
- Goal: Root warm
- Outputs: bench.py
- Fib priority: 1
- Priority: P0
- Track: root
- Bundle: root
"""
    facts = {
        "groth16": {"available": True, "status": "available"},
        "provekit": {"available": True, "status": "available"},
        "cache": {"available": False, "status": "missing"},
        "ipfs": {"available": False, "status": "missing"},
    }
    result = _full_collect(warm_heap, capability_facts=facts)
    assert GoalEvidenceGapKind.PRODUCTION_WARM_UNVERIFIED in _gap_kinds(result)
    assert not result.authoritative
    assert "cache" in result.unavailable_capabilities


def test_synthetic_always_verify_benchmark_is_never_deployment_authority() -> None:
    synthetic = {
        "interface": PROOF_REUSE_BENCHMARK_RECEIPT_INTERFACE,
        "authority": "authoritative",
        "passed": True,
        "false_admissions": 0,
        "verifier_id": "_AlwaysVerify",
        "corpus_id": "synthetic_benchmark_harness",
        "synthetic": True,
    }
    result = _full_collect(benchmark_receipt=synthetic)
    assert GoalEvidenceGapKind.SYNTHETIC_BENCHMARK_AUTHORITY in _gap_kinds(result)
    assert not result.authoritative

    # Non-synthetic benchmark does not by itself block mini-heap authority.
    real = {
        "interface": PROOF_REUSE_BENCHMARK_RECEIPT_INTERFACE,
        "authority": "authoritative",
        "passed": True,
        "false_admissions": 0,
        "verifier_id": "local-groth16@1",
        "corpus_id": "warm-eligible-v1",
    }
    ok = _full_collect(benchmark_receipt=real)
    assert ok.authoritative
    assert GoalEvidenceGapKind.SYNTHETIC_BENCHMARK_AUTHORITY not in _gap_kinds(ok)


# ---------------------------------------------------------------------------
# Fail-closed binding / freshness
# ---------------------------------------------------------------------------


def test_stale_or_mismatched_coverage_is_a_typed_gap() -> None:
    def stale(validations: dict[str, dict[str, Any]]) -> None:
        first = next(iter(validations))
        validations[first]["fresh_until_ms"] = NOW_MS - 1

    result = _full_collect(mutate_validation=stale)
    assert GoalEvidenceGapKind.COVERAGE_STALE in _gap_kinds(result)
    assert not result.authoritative

    def mismatch(validations: dict[str, dict[str, Any]]) -> None:
        first = next(iter(validations))
        validations[first]["repository_forest_cid"] = "wrong-forest"

    mismatched = _full_collect(mutate_validation=mismatch)
    assert GoalEvidenceGapKind.COVERAGE_BINDING_MISMATCH in _gap_kinds(mismatched)


def test_missing_coverage_and_missing_analyzers_fail_closed() -> None:
    result = _runner().collect(
        MINI_HEAP,
        validation_by_requirement={},
        analyzer_inputs=[],
        population_inputs=[],
        quorum_inputs=[],
        capability_facts=_capabilities_all_available(),
    )
    kinds = _gap_kinds(result)
    assert GoalEvidenceGapKind.COVERAGE_MISSING in kinds
    assert GoalEvidenceGapKind.ANALYZER_MISSING in kinds
    assert GoalEvidenceGapKind.POPULATION_MISSING in kinds
    assert GoalEvidenceGapKind.QUORUM_INSUFFICIENT in kinds
    assert not result.authoritative
    assert all(not gap.authoritative for gap in result.gaps)


def test_goal_quorum_member_construction_rejects_incomplete_records() -> None:
    with pytest.raises(ProofTestReuseGoalEvidenceError):
        GoalQuorumMember(
            member_id="",
            evidence_channel="c",
            receipt_cid="r",
            healthy=True,
            exhaustive=True,
            conclusive=True,
            fresh=True,
            uncontradicted=True,
            observed_at_ms=1,
            fresh_until_ms=2,
        )


def test_acceptance_coverage_receipt_requires_retained_byte_identity() -> None:
    from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_goal_evidence import (
        _encode_retained_bytes,
    )

    payload = {"ok": True, "requirement_id": "ptr/test-execution-contracts@1"}
    b64, cid = _encode_retained_bytes(payload)
    receipt = AcceptanceCoverageReceipt(
        requirement_id="ptr/test-execution-contracts@1",
        goal_id="PTR-G010",
        status=CoverageStatus.VERIFIED,
        producer_channel=CHANNEL,
        channel_proof_revision=CHANNEL_REV,
        repository_id=REPO,
        repository_state_cid=STATE,
        git_commit_id=COMMIT,
        git_tree_id=TREE,
        gitlink_state_cid=GITLINKS,
        repository_forest_cid=FOREST,
        dirty=False,
        dirty_overlay_cid=OVERLAY,
        policy_cid=POLICY,
        capability_cid=CAPABILITY,
        verifying_key_cid=KEY,
        circuit_cid=CIRCUIT,
        objective_revision=OBJECTIVE,
        observed_at_ms=NOW_MS - 10,
        fresh_until_ms=NOW_MS + 10_000,
        retained_validation_bytes_b64=b64,
        retained_validation_cid=cid,
    )
    assert receipt.verified
    assert receipt.authority == "authoritative"
    replayed = AcceptanceCoverageReceipt.from_dict(receipt.to_record())
    assert replayed.content_id == receipt.content_id

    with pytest.raises(ProofTestReuseGoalEvidenceError):
        AcceptanceCoverageReceipt(
            requirement_id="ptr/test-execution-contracts@1",
            goal_id="PTR-G010",
            status=CoverageStatus.VERIFIED,
            producer_channel=CHANNEL,
            channel_proof_revision=CHANNEL_REV,
            repository_id=REPO,
            repository_state_cid=STATE,
            git_commit_id=COMMIT,
            git_tree_id=TREE,
            gitlink_state_cid=GITLINKS,
            repository_forest_cid=FOREST,
            dirty=False,
            dirty_overlay_cid=OVERLAY,
            policy_cid=POLICY,
            capability_cid=CAPABILITY,
            verifying_key_cid=KEY,
            circuit_cid=CIRCUIT,
            objective_revision=OBJECTIVE,
            observed_at_ms=NOW_MS - 10,
            fresh_until_ms=NOW_MS + 10_000,
            retained_validation_bytes_b64=b64,
            retained_validation_cid="baguqeera-forged",
        )


def test_parse_goal_heap_integration_for_live_objectives() -> None:
    goals = parse_goal_heap(OBJECTIVES_PATH.read_text(encoding="utf-8"))
    assert isinstance(goals[0], ObjectiveGoal)
    discovered = discover_requirement_ids_from_heap(goals)
    assert len(discovered) == 50
