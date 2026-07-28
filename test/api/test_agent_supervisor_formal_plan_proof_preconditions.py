"""CBP-090: formal-plan requires_proof(property_id, assurance) preconditions."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.code_proof_obligations import (
    build_code_proof_cache_key,
)
from ipfs_accelerate_py.agent_supervisor.code_property_catalog import (
    DEFAULT_CODE_PROPERTY_CATALOG,
)
from ipfs_accelerate_py.agent_supervisor.formal_plan_compiler import (
    CompilationStatus,
    compile_formal_plan,
    is_requires_proof_precondition,
    requires_proof_precondition_bindings,
)
from ipfs_accelerate_py.agent_supervisor.formal_plan_conformance import (
    evaluate_requires_proof_admission,
    evaluate_requires_proof_preconditions,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_cache import (
    CacheLookupStatus,
    FormalVerificationCache,
    TrustAwareProofCache,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
    EvidenceAuthority,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)


PROPERTY_ID = "property:lease-uniqueness-and-fencing"
TREE_ID = "tree:cbp-090"
REPO_ID = "repo:cbp-090"


def _source_with_requires_proof(**task_extra: object) -> dict[str, object]:
    task: dict[str, object] = {
        "task_id": "CBP-090",
        "task_cid": "task:cid:cbp-090",
        "goal_id": "CBP-G090",
        "actor_id": "agent:implementer",
        "changed_ast_scopes": ["symbol:cid:formal-plan"],
        "acceptance_criteria": ["requires_proof preconditions admit correctly"],
        "validation_commands": [
            "python -m pytest test/api/test_agent_supervisor_formal_plan_proof_preconditions.py -q"
        ],
    }
    task.update(task_extra)
    return {
        "schema": "fixture/formal-plan-input@1",
        "repository_tree_id": TREE_ID,
        "objectives": [
            {
                "goal_id": "CBP-G090",
                "goal_cid": "goal:cid:cbp-g090",
                "owner_actor_id": "owner:supervisor",
                "title": "Formal-plan require_proof preconditions",
                "acceptance_criteria": [
                    "requires_proof preconditions compile and gate admission"
                ],
            }
        ],
        "taskboard": [task],
        "ast_records": [
            {
                "symbol_cid": "symbol:cid:formal-plan",
                "tree_cid": TREE_ID,
                "task_cid": "task:cid:cbp-090",
                "symbol": "RequiresProofPrecondition",
            }
        ],
        "proof_policy": {
            "policy_cid": "policy:cid:cbp-090",
            "minimum_code_assurance": "kernel_verified",
            "freshness_seconds": 3600,
        },
    }


def _budget() -> ResourceBudget:
    return ResourceBudget(
        wall_time_ms=10_000,
        cpu_time_ms=8_000,
        memory_bytes=64 * 1024 * 1024,
        max_processes=2,
        max_premises=4,
        network_allowed=False,
    )


def _obligation() -> CodeProofObligation:
    prop = DEFAULT_CODE_PROPERTY_CATALOG.require(PROPERTY_ID)
    return CodeProofObligation(
        repository_id=REPO_ID,
        repository_tree_id=TREE_ID,
        ast_scope_ids=("symbol:cid:formal-plan",),
        statement="Lease fencing token is unique and monotonic.",
        premise_ids=("premise:lease-state", "premise:token-order"),
        template_id=prop.template_id,
        template_version=prop.template_version,
        template_semantic_hash=prop.template_semantic_hash,
        invariant_class=prop.invariant_class or "lease_safety",
        task_id="CBP-090",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        fallback_checks=("pytest:test_lease",),
        metadata={"suite": "cbp-090", "property_id": PROPERTY_ID},
    )


def _kernel_evidence(obligation_id: str) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:kernel:cbp-090",
        subject_id=obligation_id,
        verifier_id="kernel:lean-4.19",
        independent=True,
        simulated=False,
    )


def _candidate_evidence(obligation_id: str) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.SMT_CANDIDATE,
        authority=EvidenceAuthority.SMT,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:candidate:cbp-090",
        subject_id=obligation_id,
        verifier_id="provider:claimed",
        independent=True,
        simulated=False,
    )


def _receipt(
    obligation: CodeProofObligation,
    *,
    evidence: tuple[ProofEvidence, ...] | None = None,
) -> ProofReceipt:
    return ProofReceipt(
        obligation_id=obligation.obligation_id,
        plan_id="plan:cbp-090",
        attempt_id="attempt:1",
        repository_id=obligation.repository_id,
        repository_tree_id=obligation.repository_tree_id,
        ast_scope_ids=obligation.ast_scope_ids,
        premise_ids=obligation.premise_ids,
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
        verdict=ProofVerdict.PROVED,
        evidence=evidence
        if evidence is not None
        else (_kernel_evidence(obligation.obligation_id),),
        provider_id="provider:hammer",
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
        started_at="2026-07-28T00:00:00Z",
        finished_at="2026-07-28T00:00:01Z",
        resource_usage={"wall_time_ms": 100, "peak_memory_bytes": 1_000},
        metadata={"property_id": PROPERTY_ID},
    )


def _cache_key(obligation: CodeProofObligation):
    prop = DEFAULT_CODE_PROPERTY_CATALOG.require(PROPERTY_ID)
    return build_code_proof_cache_key(
        obligation,
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
        property_id=PROPERTY_ID,
        catalog_version=DEFAULT_CODE_PROPERTY_CATALOG.catalog_version,
        catalog_id=DEFAULT_CODE_PROPERTY_CATALOG.catalog_id,
    )


def test_requires_proof_preconditions_compile_from_structured_field() -> None:
    result = compile_formal_plan(
        _source_with_requires_proof(
            requires_proof={
                "property_id": PROPERTY_ID,
                "assurance": "kernel_verified",
            }
        )
    )
    assert result.status is CompilationStatus.COMPILED
    assert result.plan is not None
    bindings = requires_proof_precondition_bindings(result.plan)
    assert len(bindings) == 1
    assert bindings[0]["property_id"] == PROPERTY_ID
    assert bindings[0]["assurance"] == AssuranceLevel.KERNEL_VERIFIED.value
    proof_preconditions = [
        item
        for item in result.plan.preconditions
        if is_requires_proof_precondition(item)
    ]
    assert len(proof_preconditions) == 1
    assert proof_preconditions[0].metadata["kind"] == "requires_proof"
    assert proof_preconditions[0].metadata["property_id"] == PROPERTY_ID


def test_requires_proof_preconditions_compile_from_preconditions_list() -> None:
    result = compile_formal_plan(
        _source_with_requires_proof(
            preconditions=[
                {
                    "kind": "requires_proof",
                    "property_id": PROPERTY_ID,
                    "assurance": "kernel_verified",
                }
            ]
        )
    )
    assert result.status is CompilationStatus.COMPILED
    assert result.plan is not None
    bindings = requires_proof_precondition_bindings(result.plan)
    assert len(bindings) == 1
    assert bindings[0]["property_id"] == PROPERTY_ID


def test_requires_proof_string_form_compiles() -> None:
    result = compile_formal_plan(
        _source_with_requires_proof(
            requires_proof=f"requires_proof({PROPERTY_ID}, kernel_verified)"
        )
    )
    assert result.status is CompilationStatus.COMPILED
    bindings = requires_proof_precondition_bindings(result.plan)
    assert bindings[0]["property_id"] == PROPERTY_ID
    assert bindings[0]["assurance"] == "kernel_verified"


def test_requires_proof_defaults_assurance_from_catalog() -> None:
    result = compile_formal_plan(
        _source_with_requires_proof(
            requires_proof={"property_id": PROPERTY_ID}
        )
    )
    assert result.status is CompilationStatus.COMPILED
    bindings = requires_proof_precondition_bindings(result.plan)
    prop = DEFAULT_CODE_PROPERTY_CATALOG.require(PROPERTY_ID)
    assert bindings[0]["assurance"] == prop.required_assurance.value


def test_unknown_property_id_fails_closed() -> None:
    result = compile_formal_plan(
        _source_with_requires_proof(
            requires_proof={
                "property_id": "property:does-not-exist",
                "assurance": "kernel_verified",
            }
        )
    )
    assert result.status is CompilationStatus.UNSUPPORTED
    assert result.plan is None
    assert any("unknown property_id" in item.message for item in result.issues)


def test_unsupported_nested_precondition_fails_closed() -> None:
    result = compile_formal_plan(
        _source_with_requires_proof(
            preconditions=[{"kind": "invented_predicate", "formula": "x > 0"}]
        )
    )
    assert result.status is CompilationStatus.UNSUPPORTED
    assert result.plan is None


def test_missing_receipt_fails_admission(tmp_path: Path) -> None:
    compiled = compile_formal_plan(
        _source_with_requires_proof(
            requires_proof={
                "property_id": PROPERTY_ID,
                "assurance": "kernel_verified",
            }
        )
    )
    assert compiled.plan is not None
    obligation = _obligation()
    key = _cache_key(obligation)
    cache: TrustAwareProofCache = FormalVerificationCache(tmp_path)

    admission = evaluate_requires_proof_preconditions(
        compiled.plan,
        proof_cache=cache,
        cache_keys={PROPERTY_ID: key},
    )
    assert admission.admitted is False
    assert "proof_receipt_missing" in admission.reason_codes
    assert len(admission.checks) == 1
    assert admission.checks[0].admitted is False
    assert admission.checks[0].cache_status == "miss"


def test_cache_hit_with_rederived_assurance_admits(tmp_path: Path) -> None:
    compiled = compile_formal_plan(
        _source_with_requires_proof(
            requires_proof={
                "property_id": PROPERTY_ID,
                "assurance": "kernel_verified",
            }
        )
    )
    assert compiled.plan is not None
    obligation = _obligation()
    key = _cache_key(obligation)
    receipt = _receipt(obligation)
    # Provider may claim ATTESTED; authoritative level is re-derived.
    assert receipt.provider_claimed_assurance is AssuranceLevel.ATTESTED
    assert receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED

    cache = FormalVerificationCache(tmp_path)
    stored = cache.put(key, receipt)
    assert stored.stored
    lookup = cache.lookup(key, required_assurance=AssuranceLevel.KERNEL_VERIFIED)
    assert lookup.status is CacheLookupStatus.HIT
    assert lookup.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED

    admission = evaluate_requires_proof_admission(
        compiled.plan,
        proof_cache=cache,
        cache_keys={PROPERTY_ID: key},
    )
    assert admission.admitted is True
    assert admission.reason_codes == ()
    assert len(admission.checks) == 1
    check = admission.checks[0]
    assert check.admitted is True
    assert check.from_cache is True
    assert check.cache_status == "hit"
    assert check.receipt_id == receipt.receipt_id
    assert check.derived_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert check.required_assurance is AssuranceLevel.KERNEL_VERIFIED


def test_candidate_only_does_not_admit() -> None:
    compiled = compile_formal_plan(
        _source_with_requires_proof(
            requires_proof={
                "property_id": PROPERTY_ID,
                "assurance": "kernel_verified",
            }
        )
    )
    assert compiled.plan is not None
    obligation = _obligation()
    candidate_receipt = _receipt(
        obligation,
        evidence=(_candidate_evidence(obligation.obligation_id),),
    )
    assert (
        candidate_receipt.authoritative_assurance is AssuranceLevel.CANDIDATE
    )

    # Candidate receipts cannot be stored as authoritative cache hits; evaluate
    # the direct binding path used for candidate-only rejection.
    admission = evaluate_requires_proof_preconditions(
        compiled.plan,
        receipts={PROPERTY_ID: candidate_receipt},
    )
    assert admission.admitted is False
    assert "candidate_only" in admission.reason_codes
    assert admission.checks[0].admitted is False
    assert admission.checks[0].derived_assurance is AssuranceLevel.CANDIDATE
    assert "candidate_only" in admission.checks[0].reason_codes


def test_candidate_only_cannot_become_authoritative_cache_hit(
    tmp_path: Path,
) -> None:
    obligation = _obligation()
    key = _cache_key(obligation)
    candidate_receipt = _receipt(
        obligation,
        evidence=(_candidate_evidence(obligation.obligation_id),),
    )
    cache = FormalVerificationCache(tmp_path)
    stored = cache.put(key, candidate_receipt)
    assert stored.stored is False
    lookup = cache.lookup(key, required_assurance=AssuranceLevel.KERNEL_VERIFIED)
    assert lookup.status is not CacheLookupStatus.HIT

    compiled = compile_formal_plan(
        _source_with_requires_proof(
            requires_proof={
                "property_id": PROPERTY_ID,
                "assurance": "kernel_verified",
            }
        )
    )
    admission = evaluate_requires_proof_preconditions(
        compiled.plan,
        proof_cache=cache,
        cache_keys={PROPERTY_ID: key},
    )
    assert admission.admitted is False
    assert "proof_receipt_missing" in admission.reason_codes


def test_vacuous_plan_without_requires_proof_admits() -> None:
    compiled = compile_formal_plan(_source_with_requires_proof())
    assert compiled.status is CompilationStatus.COMPILED
    assert compiled.plan is not None
    assert requires_proof_precondition_bindings(compiled.plan) == ()
    admission = evaluate_requires_proof_preconditions(compiled.plan)
    assert admission.admitted is True
    assert admission.checks == ()


def test_admission_result_round_trip(tmp_path: Path) -> None:
    compiled = compile_formal_plan(
        _source_with_requires_proof(
            requires_proof={
                "property_id": PROPERTY_ID,
                "assurance": "kernel_verified",
            }
        )
    )
    obligation = _obligation()
    key = _cache_key(obligation)
    cache = FormalVerificationCache(tmp_path)
    cache.put(key, _receipt(obligation))
    admission = evaluate_requires_proof_preconditions(
        compiled.plan,
        proof_cache=cache,
        cache_keys={PROPERTY_ID: key},
    )
    restored = type(admission).from_dict(admission.to_dict())
    assert restored.admission_id == admission.admission_id
    assert restored.admitted is True
    assert restored.checks[0].property_id == PROPERTY_ID
