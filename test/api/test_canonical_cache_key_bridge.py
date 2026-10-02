"""Native cross-package identity replay, drift and non-promotion controls."""
from copy import deepcopy
from dataclasses import replace

import pytest

from ipfs_datasets_py.logic.common.canonical_cache_key import (
    REQUIRED_IDENTITY_FIELDS,
    CanonicalCacheKeyError,
    CanonicalProofCacheKey,
    CandidateAsKernelError,
    CrossEnvironmentHitError,
    content_digest,
    make_identity_cid,
)
from ipfs_datasets_py.logic.ir_core.axes import LogicEvidenceAuthority, LogicEvidenceKind
from ipfs_accelerate_py.agent_supervisor.proof.canonical_cache_key_bridge import (
    CanonicalCacheBridgeError,
    IDENTITY_FIELDS,
    MAX_BRIDGE_BYTES,
    bridge_canonical_proof_cache_key,
    unbridge_canonical_proof_cache_key,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import (
    CacheRejectionReason,
    FormalVerificationCache,
    ProofCacheKey,
)
from test.api.test_agent_supervisor_formal_verification_cache import _key, _receipt


def _semantic():
    return CanonicalProofCacheKey.build(
        source={"path": "source.py", "sha256": "0" * 64},
        expression={"return": "left + right"}, formalization={"language": "Lean4"},
        slice={"function": "derive"}, obligation={"predicate": "result == expected"},
        assumptions=["exact arithmetic"], bounds={"left": [-1, 1]},
        translation={"source": "python", "target": "lean"}, provider="provider:v1",
        environment={"python": "3.12", "lean": "4.34.1"},
        policy={"network": False}, schema={"version": "case/v1"}, checker="checker:v1",
        network_policy={"allow": []}, evidence_kind=LogicEvidenceKind.SMT_CANDIDATE,
        authority_ceiling=LogicEvidenceAuthority.NONE,
        source_cid=make_identity_cid(b"source bytes"),
    )


def _changed(field):
    key = _semantic()
    if field in {"provider", "checker"}:
        value = "different:identity"
    elif field == "evidence_kind":
        value = LogicEvidenceKind.LLM_OUTPUT
    elif field == "authority_ceiling":
        value = LogicEvidenceAuthority.ADVISORY
    else:
        value = content_digest({"changed": field})
    return replace(key, **{field: value})


def _decode(bridged, *, semantic=None, execution=None):
    return unbridge_canonical_proof_cache_key(
        bridged, request=_semantic() if semantic is None else semantic,
        expected_execution_key=_key() if execution is None else execution,
    )


def test_roundtrip_retains_exact_native_payload_and_all_execution_fields():
    semantic = _semantic()
    execution = _key(obligation={"obligation_id": "obligation-1", "nested": [1, False, None]})
    bridged = bridge_canonical_proof_cache_key(semantic, execution_key=execution)
    recovered, context = _decode(ProofCacheKey.from_dict(bridged.to_dict()), execution=execution)
    assert type(recovered) is CanonicalProofCacheKey
    assert recovered.to_dict() == semantic.to_dict()
    assert recovered.key_id == semantic.key_id
    assert context.to_dict() == execution.to_dict()
    assert context.key_id == execution.key_id
    assert bridged.obligation["canonical_key"] == semantic.to_dict()
    assert bridged.obligation["scope"]["receipt_admission_supported"] is False
    assert tuple(REQUIRED_IDENTITY_FIELDS) == IDENTITY_FIELDS
    for field, row in zip(IDENTITY_FIELDS, bridged.obligation["field_correspondence"], strict=True):
        assert row == {"datasets_field": field, "supervisor_field": f"obligation.canonical_key.{field}",
                       "relation": "exact_identity_retention"}
    for field, value in execution.to_dict().items():
        if field != "obligation":
            assert bridged.to_dict()[field] == value


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
def test_every_semantic_dimension_changes_cache_identity_and_rejects_stale_request(field):
    original = bridge_canonical_proof_cache_key(_semantic(), execution_key=_key())
    changed = _changed(field)
    bridged = bridge_canonical_proof_cache_key(changed, execution_key=_key())
    assert bridged.key_id != original.key_id
    assert _decode(bridged, semantic=changed)[0] == changed
    error = CrossEnvironmentHitError if field == "environment" else CanonicalCacheKeyError
    with pytest.raises(error):
        _decode(original, semantic=changed)


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
def test_missing_native_dimension_cannot_be_recovered_from_other_context(field):
    bridged = bridge_canonical_proof_cache_key(_semantic(), execution_key=_key())
    body = deepcopy(bridged.to_dict())
    del body["obligation"]["canonical_key"][field]
    with pytest.raises(CanonicalCacheKeyError, match="missing required"):
        _decode(ProofCacheKey.from_dict(body))


@pytest.mark.parametrize("field", ["obligation", "premises", "translator", "solver", "kernel", "toolchain",
                                   "theorem_registry", "policy", "resource_budget", "candidate_tree"])
def test_every_declared_execution_dimension_stays_distinct_and_must_match(field):
    change = ("other-premise",) if field == "premises" else {"different": field}
    changed = replace(_key(), **{field: change})
    original = bridge_canonical_proof_cache_key(_semantic(), execution_key=_key())
    bridged = bridge_canonical_proof_cache_key(_semantic(), execution_key=changed)
    assert bridged.key_id != original.key_id
    assert _decode(bridged, execution=changed)[1] == changed
    with pytest.raises(CanonicalCacheBridgeError, match="execution context"):
        _decode(bridged)


@pytest.mark.parametrize("damage", ["interface", "key_id", "schema_absent", "source_cid",
                                    "correspondence", "scope", "canonical_id", "execution_id",
                                    "added_obligation_id", "missing_scope"])
def test_closed_envelope_rejects_ignored_metadata_and_identity_or_authority_tampering(damage):
    body = deepcopy(bridge_canonical_proof_cache_key(_semantic(), execution_key=_key()).to_dict())
    envelope = body["obligation"]
    if damage == "interface":
        envelope["canonical_key"]["interface"] = "forged"
    elif damage == "key_id":
        envelope["canonical_key"]["key_id"] = _semantic().key_id
    elif damage == "schema_absent":
        del envelope["canonical_key"]["schema_version"]
    elif damage == "source_cid":
        envelope["canonical_key"]["source_cid"] = make_identity_cid(b"different source")
    elif damage == "correspondence":
        envelope["field_correspondence"][0]["supervisor_field"] = "candidate_tree"
    elif damage == "scope":
        envelope["scope"]["proof_authority"] = True
    elif damage == "canonical_id":
        envelope["canonical_key_id"] = "forged"
    elif damage == "execution_id":
        envelope["execution_key_id"] = "forged"
    elif damage == "added_obligation_id":
        envelope["obligation_id"] = "obligation-1"
    else:
        del envelope["scope"]
    with pytest.raises((CanonicalCacheBridgeError, CanonicalCacheKeyError)):
        _decode(ProofCacheKey.from_dict(body))


def test_exact_native_types_required_not_dictionary_or_structural_lookalikes():
    class KeyLookalike:
        def to_dict(self):
            return _semantic().to_dict()
    class Subclass(CanonicalProofCacheKey):
        pass
    for value in (_semantic().to_dict(), KeyLookalike(), Subclass(**{
        name: getattr(_semantic(), name) for name in CanonicalProofCacheKey.__dataclass_fields__
    })):
        with pytest.raises(TypeError, match="exact native CanonicalProofCacheKey"):
            bridge_canonical_proof_cache_key(value, execution_key=_key())
    with pytest.raises(TypeError, match="exact native supervisor"):
        bridge_canonical_proof_cache_key(_semantic(), execution_key=_key().to_dict())


def test_native_candidate_as_kernel_guard_remains_active():
    with pytest.raises(CandidateAsKernelError):
        replace(_semantic(), authority_ceiling=LogicEvidenceAuthority.AUTHORITATIVE)
    # Even hostile post-construction mutation cannot bypass native replay.
    forged = _semantic()
    object.__setattr__(forged, "authority_ceiling", LogicEvidenceAuthority.AUTHORITATIVE)
    with pytest.raises(CandidateAsKernelError):
        bridge_canonical_proof_cache_key(forged, execution_key=_key())


def test_byte_budget_and_nested_bridges_refused():
    with pytest.raises(CanonicalCacheBridgeError, match="byte limit"):
        bridge_canonical_proof_cache_key(_semantic(), execution_key=_key(obligation="x" * MAX_BRIDGE_BYTES))
    bridged = bridge_canonical_proof_cache_key(_semantic(), execution_key=_key())
    with pytest.raises(CanonicalCacheBridgeError, match="nested"):
        bridge_canonical_proof_cache_key(_semantic(), execution_key=bridged)


def test_returned_values_do_not_alias_caller_or_stored_payload():
    execution = _key(obligation={"nested": [1]})
    bridged = bridge_canonical_proof_cache_key(_semantic(), execution_key=execution)
    before = bridged.key_id
    execution.obligation["nested"].append(2)
    assert bridged.key_id == before
    _, recovered = _decode(bridged, execution=_key(obligation={"nested": [1]}))
    recovered.obligation["nested"].append(3)
    assert bridged.key_id == before


def test_actual_cache_never_promotes_identity_envelope_or_checker_markers(tmp_path):
    cache = FormalVerificationCache(tmp_path / "cache")
    # A receipt eligible under its own native execution key remains unrelated
    # to the datasets semantic key, even with identically named checker/kernel.
    execution = _key()
    assert cache.put(execution, _receipt()).stored
    semantic = replace(_semantic(), checker="kernel-1", provider="solver-1")
    bridged = bridge_canonical_proof_cache_key(semantic, execution_key=execution)
    result = cache.put(bridged, _receipt())
    assert not result.stored
    assert CacheRejectionReason.BINDING_MISMATCH.value in result.reason_codes
    assert not cache.lookup(bridged).hit
    reopened = FormalVerificationCache(tmp_path / "cache")
    assert reopened.lookup(execution).hit
    assert not reopened.lookup(bridged).hit
    assert _decode(bridged, semantic=semantic)[0].authority_ceiling is LogicEvidenceAuthority.NONE
