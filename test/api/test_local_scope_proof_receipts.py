"""Authored evidence checks local scope boundaries; no prover is executed."""
from copy import deepcopy
from dataclasses import fields, replace
from datetime import datetime, timezone

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel, ContractValidationError, EvidenceAuthority, EvidenceKind,
    EvidenceVerdict, LocalProofScope, LOCAL_PROOF_COMPLETION_REJECTION,
    ProofEvidence, ProofReceipt, ProofVerdict, ResourceBudget, ScopedProofReceipt,
    MAX_LOCAL_PROOF_ID_BYTES, MAX_LOCAL_PROOF_SCOPE_IDS, MAX_LOCAL_PROOF_SCOPE_TEXT_BYTES,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import (
    FormalVerificationCache, ProofCacheEntry, build_proof_cache_key,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
SCOPE_BLIND_REASON = "conditional_local_proof_requires_scope_aware_consumer"


@pytest.fixture(autouse=True)
def isolated_orchestration(tmp_path, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))


def _legacy():
    evidence = ProofEvidence(kind=EvidenceKind.KERNEL_VERIFICATION, authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED, artifact_id="artifact:authored-local-kernel",
        subject_id="obligation:authored-local", verifier_id="kernel:authored-local", independent=True)
    return ProofReceipt(obligation_id="obligation:authored-local", plan_id="plan:authored-local",
        attempt_id="attempt:authored-local", repository_id="repository:authored-local",
        repository_tree_id="tree:authored-local", ast_scope_ids=("ast:authored-local",),
        premise_ids=("premise:authored-local",), translator_id="translator:authored-local",
        solver_id="solver:authored-local", kernel_id="kernel:authored-local",
        toolchain_id="toolchain:authored-local", policy_id="policy:authored-local",
        resource_budget=ResourceBudget(wall_time_ms=10_000), verdict=ProofVerdict.PROVED,
        evidence=(evidence,), theorem_registry_id="registry:authored-local", finished_at=NOW.isoformat())


def _scope(receipt=None, **changes):
    receipt = receipt or _legacy()
    values = dict(obligation_id=receipt.obligation_id, theorem_cid="theorem:authored-local",
        property_id="property:authored-local", consequence_ref="consequence:authored-local",
        proof_scope="Conditional local guard on converted ordinary strings.", premise_ids=receipt.premise_ids,
        assumption_ids=("assumption:declared-protocol", "assumption:reviewed-translation"),
        residual_obligation_ids=("residual:dynamic-python", "residual:security-ir"))
    values.update(changes)
    return LocalProofScope(**values)


def _scoped(receipt=None, **scope_changes):
    receipt = receipt or _legacy()
    return ScopedProofReceipt(**{item.name: getattr(receipt, item.name) for item in fields(ProofReceipt)},
                              scope=_scope(receipt, **scope_changes))


def _key(receipt):
    return build_proof_cache_key(obligation=receipt.obligation_id, premises=receipt.premise_ids,
        translator=receipt.translator_id, solver=receipt.solver_id, kernel=receipt.kernel_id,
        toolchain=receipt.toolchain_id, theorem_registry=receipt.theorem_registry_id,
        policy=receipt.policy_id, resource_budget=receipt.resource_budget.to_dict(),
        candidate_tree=receipt.repository_tree_id)


def test_legacy_canonical_identity_remains_published_v1_identity():
    # Golden identity was independently generated from published 58d0400 @1 code.
    receipt = _legacy()
    assert receipt.receipt_id == "baguqeeravjreebafxqiqj4swhkf2qcugqwkc7kpcqlibgsuzpkgobm4ywixq"
    assert receipt.evidence[0].evidence_id == "baguqeeraohdwf5fxdbzxjz2ksfv2zvcdm3fbuycmnfwch7zflmgdff5neq4q"
    assert receipt.to_dict()["contract_version"] == 1
    assert "scope" not in receipt.to_dict()
    restored = ProofReceipt.from_dict(receipt.to_record())
    assert type(restored) is ProofReceipt and restored.to_json() == receipt.to_json()
    assert receipt.satisfies_completion() is True


def test_local_scope_and_v2_receipt_keep_assurance_without_task_authority():
    receipt = _scoped()
    scope = receipt.scope
    for name in ("whole_program_proved", "completion_authority", "mutation_authority", "execution_authority"):
        assert getattr(scope, name) is False and scope.to_dict()[name] is False
    assert LocalProofScope.from_dict(scope.to_record()) == scope
    assert receipt.schema_version == receipt.to_dict()["contract_version"] == 2
    assert receipt.to_dict()["schema"].endswith("/proof-receipt@2")
    assert receipt.receipt_id != _legacy().receipt_id
    restored = ProofReceipt.from_dict(receipt.to_record())
    assert type(restored) is ScopedProofReceipt and restored == receipt
    assert restored.scope.assumption_ids == scope.assumption_ids
    assert restored.scope.residual_obligation_ids == scope.residual_obligation_ids
    assert restored.is_kernel_verified and restored.authoritative_verdict is ProofVerdict.PROVED
    assert restored.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert restored.satisfies(AssuranceLevel.KERNEL_VERIFIED)
    for required in AssuranceLevel:
        reference = restored.completion_reference(required)
        assert reference["assurance_satisfied"] is False and reference["completion_authority"] is False
        assert LOCAL_PROOF_COMPLETION_REJECTION in reference["reason_codes"]
        assert reference["receipt"] == restored.to_dict()
        assert not restored.satisfies_completion(required)


def test_scoped_receipt_persists_and_restarts_in_existing_generic_cache(tmp_path):
    receipt = _scoped()
    key = _key(receipt)
    with public_replay_without_metadata():
        cache = FormalVerificationCache(tmp_path / "cache")
        stored = cache.put(key, receipt)
        assert stored.stored, stored.reason_codes
        restarted = FormalVerificationCache(tmp_path / "cache")
        lookup = restarted.lookup(key, required_assurance=AssuranceLevel.KERNEL_VERIFIED)
    assert lookup.hit
    assert type(lookup.receipt) is ScopedProofReceipt
    assert lookup.receipt.to_json() == receipt.to_json()
    assert lookup.receipt.scope == receipt.scope
    assert lookup.receipt.is_kernel_verified and not lookup.receipt.satisfies_completion()


@pytest.mark.parametrize("change", ["obligation", "premises", "mapping"])
def test_constructor_requires_exact_typed_scope_and_semantic_joins(change):
    legacy = _legacy()
    scope = _scope(legacy)
    if change == "obligation":
        scope = replace(scope, obligation_id="different:obligation")
    elif change == "premises":
        scope = replace(scope, premise_ids=("different:premise",))
    else:
        scope = scope.to_dict()
    with pytest.raises(ContractValidationError):
        ScopedProofReceipt(**{item.name: getattr(legacy, item.name) for item in fields(ProofReceipt)}, scope=scope)


@pytest.mark.parametrize("name", ["whole_program_proved", "completion_authority", "mutation_authority", "execution_authority"])
def test_scope_authority_cannot_be_promoted_in_serialized_record(name):
    body = _scope().to_dict()
    body[name] = True
    with pytest.raises(ContractValidationError):
        LocalProofScope.from_dict(body)


@pytest.mark.parametrize("name", ["obligation_id", "theorem_cid", "property_id", "consequence_ref", "proof_scope",
                                 "premise_ids", "assumption_ids", "residual_obligation_ids"])
def test_scope_decode_requires_every_boundary_field(name):
    body = _scope().to_dict()
    body.pop(name)
    with pytest.raises(ContractValidationError):
        LocalProofScope.from_dict(body)


@pytest.mark.parametrize("kind", ["missing-scope", "metadata-only", "scope-extra", "receipt-extra", "missing-default",
                                 "wrong-version", "bool-version", "wrong-schema", "forged-cid", "second-forged-cid",
                                 "unsorted-assumptions", "scope-cid", "scope-version", "scope-arrays"])
def test_v2_decode_rejects_scope_stripping_and_noncanonical_bodies(kind):
    body = deepcopy(_scoped().to_record())
    if kind == "missing-scope":
        body.pop("scope")
    elif kind == "metadata-only":
        body["metadata"]["local_scope"] = body.pop("scope")
    elif kind == "scope-extra":
        body["scope"]["ignored_frontier"] = "must not drop"
    elif kind == "receipt-extra":
        body["completion_authority"] = True
    elif kind == "missing-default":
        body.pop("started_at")
    elif kind == "wrong-version":
        body["contract_version"] = 1
    elif kind == "bool-version":
        body["contract_version"] = True
    elif kind == "wrong-schema":
        body["schema"] = _legacy().SCHEMA
        body.pop("scope")
    elif kind == "forged-cid":
        body["content_id"] = "forged:identity"
    elif kind == "second-forged-cid":
        body["receipt_id"] = _scoped().receipt_id
        body["content_id"] = "forged:second-identity"
    elif kind == "unsorted-assumptions":
        body["scope"]["assumption_ids"].reverse()
    elif kind == "scope-cid":
        body["scope"]["content_id"] = "forged:scope"
    elif kind == "scope-version":
        body["scope"]["contract_version"] = True
    else:
        body["scope"]["premise_ids"] = "premise:authored-local"
    with pytest.raises(ContractValidationError):
        ProofReceipt.from_dict(body)


@pytest.mark.parametrize("changes", [
    {"theorem_cid": "x" * (MAX_LOCAL_PROOF_ID_BYTES + 1)},
    {"proof_scope": "x" * (MAX_LOCAL_PROOF_SCOPE_TEXT_BYTES + 1)},
    {"residual_obligation_ids": tuple("residual:" + str(index) for index in range(MAX_LOCAL_PROOF_SCOPE_IDS + 1))},
    {"assumption_ids": ("duplicate", "duplicate")},
    {"premise_ids": ["premise:authored-local"]},
])
def test_scope_construction_is_bounded_and_unambiguous(changes):
    with pytest.raises(ContractValidationError):
        _scope(**changes)


def test_public_completion_projection_and_validation_deny_only_local_authority():
    from ipfs_accelerate_py.agent_supervisor.objectives.goal_completion import CompletionEvidence, validate_completion_evidence
    scoped = _scoped()
    for receipt, expected in ((_legacy(), True), (scoped, False)):
        evidence = CompletionEvidence.from_proof_receipt(acceptance_criterion="Authored proof requirement",
            proof_receipt=receipt, required_assurance=AssuranceLevel.KERNEL_VERIFIED,
            validation_receipt={"attempted": True, "passed": True}, validation_passed=True, observed_at=NOW)
        public = evidence.to_dict()
        assert public["authoritative_assurance"] == "kernel_verified" and public["proof_verdict"] == "proved"
        assert public["assurance_satisfied"] is expected
        restored = CompletionEvidence.from_dict(public)
        assert restored.assurance_satisfied is expected
        result = validate_completion_evidence(restored, repository_id=receipt.repository_id,
            repository_tree=receipt.repository_tree_id, now=NOW)
        assert result.validation_succeeded is True
        assert result.assurance_satisfied is expected and result.valid is expected
        if not expected:
            assert result.reason_codes == (LOCAL_PROOF_COMPLETION_REJECTION,)
            assert public["proof_receipt"]["scope"] == scoped.scope.to_dict()


@pytest.mark.parametrize("cached", [False, True])
def test_flat_claim_adapter_refuses_local_scope_without_downgrading_kernel_assurance(cached):
    from ipfs_accelerate_py.agent_supervisor.assurance.formal_claim_adapter import (
        TypedIncompatibility, adapt_proof_cache_entry, adapt_proof_receipt,
    )
    receipt = _scoped()
    if cached:
        entry = ProofCacheEntry.create(_key(receipt), receipt, created_at_ms=1, expires_at_ms=10)
        adapted = adapt_proof_cache_entry(entry, now_ms=2)
    else:
        adapted = adapt_proof_receipt(receipt)
    assert type(adapted) is TypedIncompatibility and adapted.code == SCOPE_BLIND_REASON
    assert receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED


def test_flat_policy_outcome_refuses_local_scope():
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_policy import PolicyValidationError, ProofOutcome
    assert ProofOutcome.from_receipt("requirement:authored", _legacy()).authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    with pytest.raises(PolicyValidationError, match="scope-aware policy consumer"):
        ProofOutcome.from_receipt("requirement:authored", _scoped())


def test_requires_proof_admission_denies_scoped_receipt_with_valid_property_and_kernel():
    from test.api.test_agent_supervisor_formal_plan_proof_preconditions import PROPERTY_ID, _obligation, _receipt
    from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_conformance import evaluate_requires_proof_preconditions
    legacy = _receipt(_obligation())
    binding = {"property_id": PROPERTY_ID, "required_assurance": "kernel_verified"}
    with public_replay_without_metadata():
        admitted = evaluate_requires_proof_preconditions(bindings=(binding,), receipts={PROPERTY_ID: legacy})
        refused = evaluate_requires_proof_preconditions(bindings=(binding,),
            receipts={PROPERTY_ID: _scoped(legacy, property_id=PROPERTY_ID)})
    assert admitted.admitted and not refused.admitted
    assert refused.reason_codes == (SCOPE_BLIND_REASON,)
    assert refused.checks[0].derived_assurance is AssuranceLevel.KERNEL_VERIFIED


def test_codeproof_binding_denies_scoped_receipt_even_when_every_other_join_is_valid():
    from test.api.test_agent_supervisor_proof_goal_completion import _bound_code_receipt, _fresh_implementation_manifest
    from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import validate_code_proof_receipt_bindings
    manifest = _fresh_implementation_manifest()
    obligation = manifest.obligations[0]
    legacy = _bound_code_receipt(obligation, manifest.binding)
    admitted = validate_code_proof_receipt_bindings(legacy, manifest, obligation=obligation)
    refused = validate_code_proof_receipt_bindings(_scoped(legacy), manifest, obligation=obligation)
    assert admitted.valid
    assert not refused.valid and not refused.stale and not refused.contradictory
    assert refused.reason_codes == (SCOPE_BLIND_REASON,)
    assert refused.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED


def test_code_claim_projection_refuses_to_drop_typed_local_scope():
    from ipfs_accelerate_py.agent_supervisor.proof.code_claim_contracts import (
        CodeClaimContractError, claim_from_receipt,
    )
    legacy = _legacy()
    scoped = _scoped(legacy)
    prior = claim_from_receipt(legacy, property_id=scoped.scope.property_id,
                              assumption_ids=scoped.scope.assumption_ids)
    assert prior.receipt_id == legacy.receipt_id
    with pytest.raises(CodeClaimContractError,
                       match="conditional local proof requires a scope-aware code claim consumer"):
        claim_from_receipt(scoped, property_id=scoped.scope.property_id,
                           assumption_ids=scoped.scope.assumption_ids)
    assert scoped.is_kernel_verified and scoped.scope.residual_obligation_ids


def test_code_claim_cache_hit_refuses_to_drop_retained_scoped_receipt(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.proof.code_claim_contracts import (
        CodeClaimContractError, apply_cache_lookup, claim_from_receipt,
    )
    legacy = _legacy()
    scoped = _scoped(legacy)
    prior = claim_from_receipt(legacy, property_id=scoped.scope.property_id,
                              assumption_ids=scoped.scope.assumption_ids)
    with public_replay_without_metadata():
        cache = FormalVerificationCache(tmp_path / "scoped-claim-cache")
        key = _key(scoped)
        assert cache.put(key, scoped).stored
        hit = cache.lookup(key, required_assurance=AssuranceLevel.KERNEL_VERIFIED)
    assert hit.hit and type(hit.receipt) is ScopedProofReceipt
    assert hit.receipt.scope == scoped.scope and hit.receipt.is_kernel_verified
    with pytest.raises(CodeClaimContractError,
                       match="conditional local proof requires a scope-aware code claim consumer"):
        apply_cache_lookup(prior, outcome="hit", receipt=hit.receipt)
