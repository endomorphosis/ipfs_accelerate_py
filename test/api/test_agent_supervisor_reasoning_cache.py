from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.reasoning_cache import (
    REASONING_CACHE_INTERFACE,
    REASONING_COMPUTATION_KEY_SCHEMA,
    CacheUseReceipt,
    ReasoningCacheCoordinator,
    ReasoningCacheError,
    ReasoningCacheReason,
    ReasoningCacheStatus,
    ReasoningComputationKey,
    ReasoningOutcome,
    ReasoningPrivateMaterialError,
    ReasoningSourceReceipt,
    build_reasoning_cache_key,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)


def _budget(**changes: object) -> ResourceBudget:
    values: dict[str, object] = {
        "wall_time_ms": 10_000,
        "cpu_time_ms": 8_000,
        "memory_bytes": 64 * 1024 * 1024,
        "max_processes": 2,
        "max_premises": 8,
        "network_allowed": False,
    }
    values.update(changes)
    return ResourceBudget(**values)


def _key(**changes: object) -> ReasoningComputationKey:
    values: dict[str, object] = {
        "operation": "analyze_property",
        "property": {"id": "property:1", "revision": "property@1"},
        "repository_forest": {
            "id": "tree:1",
            "superproject": "tree:1",
            "gitlinks": ["tree:submodule:1"],
        },
        "scope": {"id": "scope:1", "paths": ["src/a.py"]},
        "premises": (
            {"id": "premise:a", "digest": "sha256:premise-a"},
            {"id": "premise:b", "digest": "sha256:premise-b"},
        ),
        "assumptions": (
            {"id": "assumption:a", "digest": "sha256:assumption-a"},
        ),
        "parser": {"id": "parser:python", "revision": "parser@1"},
        "index": {"id": "index:ast", "revision": "index@1"},
        "translator": {"id": "translator:logic", "revision": "translator@1"},
        "toolchain": {"id": "toolchain:analysis", "digest": "sha256:toolchain-1"},
        "capability": {
            "id": "capability:analysis",
            "revision": "capability@1",
            "solver_id": "solver:1",
            "kernel_id": "kernel:1",
        },
        "policy": {"id": "policy:1", "revision": "policy@1"},
        "ir": {"id": "ir:1", "schema": "IntentIR@1"},
        "catalog": {"id": "catalog:1", "revision": "catalog@1"},
        "required_assurance": AssuranceLevel.CANDIDATE,
        "bounds": _budget().to_dict(),
        "dependencies": (
            "dependency:src/a.py@1",
            "dependency:catalog@1",
        ),
    }
    values.update(changes)
    return build_reasoning_cache_key(**values)


def _evidence() -> tuple[dict[str, object], ...]:
    return (
        {
            "kind": "static_observation",
            "authority": "analysis_provider",
            "verdict": "accepted",
            "independent": True,
            "artifact_id": "artifact:observation:1",
        },
    )


def _coordinator(
    tmp_path: Path,
    *,
    run_id: str = "run:1",
    assurance_deriver=None,
) -> ReasoningCacheCoordinator:
    return ReasoningCacheCoordinator(
        tmp_path / "analysis",
        tmp_path / "proof",
        tmp_path / "artifacts",
        run_id=run_id,
        assurance_deriver=assurance_deriver,
    )


def _store(
    coordinator: ReasoningCacheCoordinator,
    key: ReasoningComputationKey | None = None,
    *,
    payload: dict[str, object] | None = None,
):
    return coordinator.put_analysis(
        key or _key(),
        payload or {"finding_ids": ["finding:1"], "count": 1},
        evidence=_evidence(),
        claimed_assurance=AssuranceLevel.CANDIDATE,
    )


def test_interface_and_key_bind_every_semantic_dimension() -> None:
    assert REASONING_CACHE_INTERFACE == "ReasoningCacheCoordinator@1"
    baseline = _key()
    assert baseline.schema == REASONING_COMPUTATION_KEY_SCHEMA
    assert ReasoningComputationKey.from_dict(baseline.to_dict()) == baseline
    assert baseline.key_id == _key(
        premises=tuple(reversed(baseline.premises)),
        assumptions=tuple(reversed(baseline.assumptions)),
        dependencies=tuple(reversed(baseline.dependencies)),
    ).key_id

    mutations = {
        "operation": "prove_property",
        "property": {"id": "property:2"},
        "repository_forest": {"id": "tree:2"},
        "scope": {"id": "scope:2"},
        "premises": ({"id": "premise:c"},),
        "assumptions": ({"id": "assumption:b"},),
        "parser": {"id": "parser:2"},
        "index": {"id": "index:2"},
        "translator": {"id": "translator:2"},
        "toolchain": {"id": "toolchain:2"},
        "capability": {
            "id": "capability:2",
            "solver_id": "solver:2",
            "kernel_id": "kernel:2",
        },
        "policy": {"id": "policy:2"},
        "ir": {"id": "ir:2"},
        "catalog": {"id": "catalog:2"},
        "required_assurance": AssuranceLevel.SOLVER_CHECKED,
        "bounds": _budget(wall_time_ms=9_999).to_dict(),
        "dependencies": ("dependency:other@1",),
    }
    for name, value in mutations.items():
        assert _key(**{name: value}).key_id != baseline.key_id, name


def test_private_material_never_enters_keys_or_source_receipts() -> None:
    with pytest.raises(ReasoningPrivateMaterialError) as key_error:
        _key(scope={"id": "scope:1", "api_key": "do-not-copy"})
    assert key_error.value.reason_code == ReasoningCacheReason.PRIVATE_MATERIAL.value
    with pytest.raises(ReasoningPrivateMaterialError):
        ReasoningSourceReceipt.create(
            _key(),
            {"nested": {"private_witness": "do-not-copy"}},
            producer_run_id="run:1",
        )


def test_miss_is_explicitly_not_refutation(tmp_path: Path) -> None:
    result = _coordinator(tmp_path).lookup_analysis(_key())
    assert result.status is ReasoningCacheStatus.MISS
    assert result.reason_codes == (
        ReasoningCacheReason.CACHE_MISS.value,
        ReasoningCacheReason.CACHE_MISS_NOT_REFUTATION.value,
    )
    assert result.refuted is False
    assert result.is_refutation is False
    assert result.is_completion_evidence is False


def test_hit_reloads_cas_receipt_and_rederives_assurance(
    tmp_path: Path,
) -> None:
    derivations: list[str] = []

    def derive(receipt: ReasoningSourceReceipt) -> AssuranceLevel:
        derivations.append(receipt.receipt_id)
        assert receipt.payload["finding_ids"] == ["finding:1"]
        return AssuranceLevel.CANDIDATE

    coordinator = _coordinator(tmp_path, assurance_deriver=derive)
    stored = _store(coordinator)
    assert stored.stored
    assert stored.source_reference is not None
    assert derivations == [stored.source_receipt.receipt_id]

    hit = coordinator.lookup_analysis(_key())
    assert hit.status is ReasoningCacheStatus.HIT
    assert hit.payload == {"finding_ids": ["finding:1"], "count": 1}
    assert hit.assurance is AssuranceLevel.CANDIDATE
    assert hit.is_completion_evidence
    assert derivations == [
        stored.source_receipt.receipt_id,
        stored.source_receipt.receipt_id,
    ]
    assert hit.use_receipt is not None
    assert coordinator.verify_use_receipt(hit.use_receipt, _key())


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (
            {"repository_forest": {"id": "tree:wrong"}},
            ReasoningCacheReason.WRONG_REPOSITORY_FOREST,
        ),
        (
            {"scope": {"id": "scope:wrong"}},
            ReasoningCacheReason.WRONG_SCOPE,
        ),
        (
            {"toolchain": {"id": "toolchain:wrong"}},
            ReasoningCacheReason.WRONG_TOOLCHAIN,
        ),
        (
            {"policy": {"id": "policy:wrong"}},
            ReasoningCacheReason.WRONG_POLICY,
        ),
        (
            {"ir": {"id": "ir:wrong"}},
            ReasoningCacheReason.WRONG_IR,
        ),
        (
            {"catalog": {"id": "catalog:wrong"}},
            ReasoningCacheReason.WRONG_CATALOG,
        ),
    ],
)
def test_wrong_semantic_context_rejects_with_dimension_reason(
    tmp_path: Path,
    change: dict[str, object],
    reason: ReasoningCacheReason,
) -> None:
    coordinator = _coordinator(tmp_path)
    assert _store(coordinator)
    result = coordinator.lookup_analysis(_key(**change))
    assert result.status is ReasoningCacheStatus.REJECTED
    assert reason.value in result.reason_codes
    assert result.is_completion_evidence is False


def test_partial_and_insufficient_assurance_never_become_hits(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)
    partial = coordinator.put_analysis(
        _key(required_assurance=AssuranceLevel.UNVERIFIED),
        {"finding_ids": []},
        outcome=ReasoningOutcome.PARTIAL,
    )
    assert partial.stored
    rejected = coordinator.lookup_analysis(
        _key(required_assurance=AssuranceLevel.UNVERIFIED)
    )
    assert rejected.status is ReasoningCacheStatus.REJECTED
    assert ReasoningCacheReason.PARTIAL_ENTRY.value in rejected.reason_codes

    strong_key = _key(required_assurance=AssuranceLevel.SOLVER_CHECKED)
    assert coordinator.put_analysis(
        strong_key,
        {"finding_ids": ["candidate"]},
        evidence=_evidence(),
        claimed_assurance=AssuranceLevel.CANDIDATE,
    )
    insufficient = coordinator.lookup_analysis(strong_key)
    assert insufficient.status is ReasoningCacheStatus.REJECTED
    assert insufficient.reason_code == (
        ReasoningCacheReason.INSUFFICIENT_ASSURANCE.value
    )


def test_forged_claim_and_undeclared_dependency_fail_closed(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)
    forged = coordinator.put_analysis(
        _key(),
        {"finding_ids": ["forged"]},
        evidence=(),
        claimed_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )
    assert not forged.stored
    assert forged.reason_code == (
        ReasoningCacheReason.ASSURANCE_CLAIM_MISMATCH.value
    )

    with pytest.raises(ReasoningCacheError) as undeclared:
        ReasoningSourceReceipt.create(
            _key(),
            {"finding_ids": []},
            producer_run_id="run:1",
            observed_dependencies=("dependency:not-declared@1",),
        )
    assert undeclared.value.reason_code == (
        ReasoningCacheReason.UNDECLARED_DEPENDENCY.value
    )


def test_poisoned_native_index_and_corrupt_cas_fail_with_reasons(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)
    stored = _store(coordinator)
    assert stored.source_reference is not None

    native_path = coordinator.analysis_cache.entry_path(
        _key().to_analysis_cache_key()
    )
    native_payload = json.loads(native_path.read_text(encoding="utf-8"))
    native_payload["receipt"]["metadata"]["source_receipt_id"] = "forged"
    native_path.write_text(
        json.dumps(native_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    poisoned = coordinator.lookup_analysis(_key())
    assert poisoned.status is ReasoningCacheStatus.REJECTED
    assert poisoned.reason_code == ReasoningCacheReason.POISONED_ENTRY.value

    restored = _store(coordinator)
    assert restored
    projection_path = coordinator.artifact_store._projection_path(
        restored.source_reference
    )
    projection_path.write_bytes(b'{ "truncated": ')
    corrupt = coordinator.lookup_analysis(_key())
    assert corrupt.status is ReasoningCacheStatus.REJECTED
    assert corrupt.reason_code == (
        ReasoningCacheReason.ARTIFACT_INTEGRITY_FAILED.value
    )


def test_schema_and_receipt_identity_forgery_are_rejected() -> None:
    source = ReasoningSourceReceipt.create(
        _key(),
        {"finding_ids": []},
        producer_run_id="run:1",
    )
    wrong_schema = source.to_dict()
    wrong_schema["schema"] = "reasoning-source-receipt@999"
    with pytest.raises(ReasoningCacheError) as schema_error:
        ReasoningSourceReceipt.from_dict(wrong_schema)
    assert schema_error.value.reason_code == ReasoningCacheReason.WRONG_SCHEMA.value

    forged = source.to_dict()
    forged["payload"]["finding_ids"] = ["forged"]
    with pytest.raises(ReasoningCacheError) as forged_error:
        ReasoningSourceReceipt.from_dict(forged)
    assert forged_error.value.reason_code == (
        ReasoningCacheReason.FORGED_RECEIPT.value
    )


def test_use_receipts_are_fresh_run_bound_and_not_wire_authority(
    tmp_path: Path,
) -> None:
    first = _coordinator(tmp_path, run_id="run:1")
    assert _store(first)
    first_hit = first.lookup_analysis(_key())
    assert first_hit.use_receipt is not None

    restarted = _coordinator(tmp_path, run_id="run:2")
    second_hit = restarted.lookup_analysis(_key())
    assert second_hit.hit
    assert second_hit.source_receipt.producer_run_id == "run:1"
    assert second_hit.use_receipt.current_run_id == "run:2"
    assert second_hit.use_receipt.receipt_id != first_hit.use_receipt.receipt_id

    replay = restarted.verify_use_receipt(first_hit.use_receipt, _key())
    assert not replay
    assert replay.reason_code == ReasoningCacheReason.CROSS_RUN_REPLAY.value

    deserialized = CacheUseReceipt.from_dict(second_hit.use_receipt.to_dict())
    forged = restarted.verify_use_receipt(deserialized, _key())
    assert not forged
    assert forged.reason_code == ReasoningCacheReason.FORGED_RECEIPT.value


def test_dependency_invalidation_is_local_and_retains_shared_artifacts(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)
    key_a = _key(dependencies=("dependency:a@1",))
    key_b = _key(
        property={"id": "property:2"},
        dependencies=("dependency:b@1",),
    )
    stored_a = _store(coordinator, key_a, payload={"result": "a"})
    stored_b = _store(coordinator, key_b, payload={"result": "b"})
    assert stored_a and stored_b

    receipt = coordinator.invalidate_dependencies(("dependency:a@1",))
    assert receipt.invalidated_analysis_key_ids == (key_a.key_id,)
    assert receipt.invalidated_proof_key_ids == ()
    assert receipt.retained_artifact_ids == (
        stored_a.source_reference.artifact_id,
    )
    assert coordinator.lookup_analysis(key_a).miss
    assert coordinator.lookup_analysis(key_b).hit
    assert coordinator.artifact_store.read_projection(
        stored_a.source_reference
    )["receipt_id"] == stored_a.source_receipt.receipt_id


def test_identical_concurrent_misses_share_one_native_flight(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)
    barrier = threading.Barrier(8)
    calls = 0
    lock = threading.Lock()

    def producer() -> dict[str, object]:
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.15)
        return {"finding_ids": ["shared"], "count": 1}

    def worker():
        barrier.wait()
        return coordinator.get_or_compute_analysis(
            _key(),
            producer,
            evidence=_evidence(),
            claimed_assurance=AssuranceLevel.CANDIDATE,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = [future.result(timeout=10) for future in [
            executor.submit(worker) for _ in range(8)
        ]]

    assert calls == 1
    assert all(result.hit and result.is_completion_evidence for result in results)
    assert sum(result.produced for result in results) == 1
    assert all(
        result.status
        in {
            ReasoningCacheStatus.PRODUCED,
            ReasoningCacheStatus.SHARED,
            ReasoningCacheStatus.HIT,
        }
        for result in results
    )


def _proof_key() -> ReasoningComputationKey:
    return _key(
        operation="prove_property",
        property={"id": "obligation:1"},
        repository_forest={"id": "tree:1"},
        scope={"id": "scope:1"},
        premises=({"id": "premise:a"}, {"id": "premise:b"}),
        assumptions=(),
        parser={"id": "parser:logic"},
        index={"id": "index:logic"},
        translator={"id": "translator:1"},
        toolchain={"id": "toolchain:1"},
        capability={
            "id": "capability:proof",
            "solver_id": "solver:1",
            "kernel_id": "kernel:1",
        },
        policy={"id": "policy:1"},
        ir={"id": "ProofIR:1"},
        catalog={"id": "registry:1"},
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        dependencies=("dependency:proof-input@1",),
    )


def _proof_receipt() -> ProofReceipt:
    evidence = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:kernel:1",
        subject_id="obligation:1",
        verifier_id="kernel:1",
        independent=True,
    )
    return ProofReceipt(
        obligation_id="obligation:1",
        plan_id="plan:1",
        attempt_id="attempt:1",
        repository_id="repository:1",
        repository_tree_id="tree:1",
        ast_scope_ids=("scope:1",),
        premise_ids=("premise:a", "premise:b"),
        translator_id="translator:1",
        solver_id="solver:1",
        kernel_id="kernel:1",
        toolchain_id="toolchain:1",
        theorem_registry_id="registry:1",
        policy_id="policy:1",
        resource_budget=_budget(),
        verdict=ProofVerdict.PROVED,
        evidence=(evidence,),
        freshness=EvidenceFreshness.CURRENT,
        kernel_receipt_id="kernel-receipt:1",
    )


def test_proof_lane_reuses_the_only_trust_aware_proof_cache(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)
    key = _proof_key()
    calls = 0

    def produce() -> ProofReceipt:
        nonlocal calls
        calls += 1
        return _proof_receipt()

    first = coordinator.get_or_compute_proof(key, produce)
    second = coordinator.get_or_compute_proof(key, produce)
    assert first.status is ReasoningCacheStatus.PRODUCED
    assert second.status is ReasoningCacheStatus.HIT
    assert calls == 1
    assert second.assurance is AssuranceLevel.KERNEL_VERIFIED
    assert second.use_receipt is not None
    assert coordinator.verify_use_receipt(second.use_receipt, key)

    wrong_native_key = replace(
        key.to_proof_cache_key(),
        policy={"id": "policy:wrong"},
    )
    mismatch = coordinator.lookup_proof(key, proof_key=wrong_native_key)
    assert mismatch.status is ReasoningCacheStatus.REJECTED
    assert mismatch.reason_code == (
        ReasoningCacheReason.SEMANTIC_KEY_MISMATCH.value
    )

    restarted = _coordinator(tmp_path, run_id="run:2")
    invalidation = restarted.invalidate_dependencies(
        ("dependency:proof-input@1",)
    )
    assert invalidation.invalidated_proof_key_ids == (
        key.to_proof_cache_key().key_id,
    )
    assert restarted.lookup_proof(key).miss
