"""IPS-030: allowlisted verification-key, proving-key, and signer trust policy."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.trust import (
    DEFAULT_PRODUCTION_SETUP_ORIGINS,
    KEY_REGISTRY_EVIDENCE,
    SIGNER_TRUST_EVIDENCE,
    ProvingKeyHandle,
    ProvingKeyRecord,
    SetupOrigin,
    SignerTrustRecord,
    SignerTrustRegistry,
    TrustError,
    TrustOutcome,
    TrustRejectionReason,
    TrustedProofPolicy,
    VerificationKeyRecord,
    VerificationKeyRegistry,
    build_production_policy,
    build_test_policy,
    closed_setup_origins,
    closed_trust_rejection_reasons,
)

_VK_CID = "bafybeigverificationkey000000000000000000000001"
_VK_CID_B = "bafybeigverificationkey000000000000000000000002"
_PK_CID = "bafybeigprovingkey00000000000000000000000000001"
_PK_CID_B = "bafybeigprovingkey00000000000000000000000000002"
_CIRCUIT = "circuit:ips-seal@1"
_CIRCUIT_B = "circuit:ips-other@1"
_SIGNER = "allowlist/operator-1"
_SIGNER_B = "allowlist/operator-2"


def _vk(**overrides) -> VerificationKeyRecord:
    payload = {
        "key_id": "vk/prod-1",
        "key_cid": _VK_CID,
        "circuit_ids": frozenset({_CIRCUIT}),
        "setup_origin": SetupOrigin.OPERATOR_REVIEWED,
        "test_only": False,
        "epoch": 1,
    }
    payload.update(overrides)
    return VerificationKeyRecord(**payload)


def _pk(**overrides) -> ProvingKeyRecord:
    payload = {
        "key_id": "pk/prod-1",
        "key_cid": _PK_CID,
        "circuit_ids": frozenset({_CIRCUIT}),
        "setup_origin": SetupOrigin.OPERATOR_REVIEWED,
        "test_only": False,
        "paired_verification_key_id": "vk/prod-1",
        "epoch": 1,
    }
    payload.update(overrides)
    return ProvingKeyRecord(**payload)


def _signer(**overrides) -> SignerTrustRecord:
    payload = {
        "signer_id": _SIGNER,
        "scopes": frozenset({"seal", "receipt"}),
        "trusted": True,
        "test_only": False,
        "revocation_epoch": None,
    }
    payload.update(overrides)
    return SignerTrustRecord(**payload)


def _production_policy(**kwargs) -> TrustedProofPolicy:
    return build_production_policy(
        verification_keys=kwargs.pop("verification_keys", (_vk(),)),
        proving_keys=kwargs.pop("proving_keys", (_pk(),)),
        signers=kwargs.pop("signers", (_signer(),)),
        current_epoch=kwargs.pop("current_epoch", 3),
        minimum_key_epoch=kwargs.pop("minimum_key_epoch", 1),
        **kwargs,
    )


def _test_policy(**kwargs) -> TrustedProofPolicy:
    return build_test_policy(
        verification_keys=kwargs.pop(
            "verification_keys",
            (
                _vk(
                    key_id="vk/test-1",
                    key_cid=_VK_CID_B,
                    test_only=True,
                    setup_origin=SetupOrigin.TEST_FIXTURE,
                ),
            ),
        ),
        proving_keys=kwargs.pop(
            "proving_keys",
            (
                _pk(
                    key_id="pk/test-1",
                    key_cid=_PK_CID_B,
                    test_only=True,
                    setup_origin=SetupOrigin.TEST_FIXTURE,
                    paired_verification_key_id="vk/test-1",
                ),
            ),
        ),
        signers=kwargs.pop(
            "signers",
            (_signer(signer_id="allowlist/test-signer", test_only=True),),
        ),
        current_epoch=kwargs.pop("current_epoch", 1),
        minimum_key_epoch=kwargs.pop("minimum_key_epoch", 0),
        **kwargs,
    )


def test_evidence_subsets_and_closed_vocabularies() -> None:
    assert KEY_REGISTRY_EVIDENCE == "ips/key-registry@1"
    assert SIGNER_TRUST_EVIDENCE == "ips/signer-trust@1"
    reasons = closed_trust_rejection_reasons()
    for required in (
        "unallowlisted_verification_key",
        "substituted_verification_key",
        "old_verification_key",
        "test_only_in_production",
        "unallowlisted_proving_key",
        "untrusted_signer",
        "revoked_signer",
        "out_of_scope_signer",
        "key_generation_forbidden",
        "key_download_forbidden",
        "circuit_incompatible",
    ):
        assert required in reasons
    origins = closed_setup_origins()
    assert "operator_reviewed" in origins
    assert "ceremony" in origins
    assert "test_fixture" in origins
    assert SetupOrigin.OPERATOR_REVIEWED.value in DEFAULT_PRODUCTION_SETUP_ORIGINS
    assert SetupOrigin.TEST_FIXTURE.value not in DEFAULT_PRODUCTION_SETUP_ORIGINS


def test_allowlisted_verification_key_accepted() -> None:
    policy = _production_policy()
    decision = policy.select_verification_key(
        "vk/prod-1", key_cid=_VK_CID, circuit_id=_CIRCUIT
    )
    assert decision.accepted is True
    assert decision.outcome is TrustOutcome.ACCEPTED
    assert decision.reason_code is None
    assert decision.evidence_subset == KEY_REGISTRY_EVIDENCE
    assert decision.details["key_cid"] == _VK_CID


def test_unallowlisted_verification_key_rejects() -> None:
    policy = _production_policy()
    decision = policy.select_verification_key("vk/unknown")
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.UNALLOWLISTED_VERIFICATION_KEY.value
    assert "not on the allowlist" in decision.message


def test_substituted_verification_key_rejects() -> None:
    policy = _production_policy()
    decision = policy.select_verification_key(
        "vk/prod-1", key_cid=_VK_CID_B, circuit_id=_CIRCUIT
    )
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.SUBSTITUTED_VERIFICATION_KEY.value
    assert decision.details["claimed_cid"] == _VK_CID_B
    assert decision.details["allowlisted_cid"] == _VK_CID


def test_old_superseded_verification_key_rejects() -> None:
    policy = _production_policy(
        verification_keys=(
            _vk(superseded_by="vk/prod-2"),
            _vk(key_id="vk/prod-2", key_cid=_VK_CID_B, epoch=2),
        )
    )
    decision = policy.select_verification_key("vk/prod-1", key_cid=_VK_CID)
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.OLD_VERIFICATION_KEY.value
    assert "superseded" in decision.message


def test_old_epoch_verification_key_rejects() -> None:
    policy = _production_policy(
        minimum_key_epoch=5,
        verification_keys=(_vk(epoch=1),),
    )
    decision = policy.select_verification_key("vk/prod-1", key_cid=_VK_CID)
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.OLD_VERIFICATION_KEY.value
    assert decision.details["required_epoch"] == 5


def test_revoked_verification_key_rejects() -> None:
    policy = _production_policy(verification_keys=(_vk(revoked=True),))
    decision = policy.select_verification_key("vk/prod-1", key_cid=_VK_CID)
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.REVOKED_VERIFICATION_KEY.value


def test_test_only_verification_key_in_production_rejects() -> None:
    # Production registry construction rejects test_only registration.
    with pytest.raises(TrustError, match="test_only"):
        build_production_policy(
            verification_keys=(
                _vk(test_only=True, setup_origin=SetupOrigin.TEST_FIXTURE),
            )
        )

    # Even if a non-production registry holds a test key, production evaluate rejects.
    test_registry = VerificationKeyRegistry(production=False)
    test_registry.register(
        _vk(
            key_id="vk/test-1",
            key_cid=_VK_CID_B,
            test_only=True,
            setup_origin=SetupOrigin.TEST_FIXTURE,
        )
    )
    decision = test_registry.evaluate("vk/test-1", production=True)
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.TEST_ONLY_IN_PRODUCTION.value


def test_circuit_incompatible_verification_key_rejects() -> None:
    policy = _production_policy()
    decision = policy.select_verification_key(
        "vk/prod-1", key_cid=_VK_CID, circuit_id=_CIRCUIT_B
    )
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.CIRCUIT_INCOMPATIBLE.value


def test_allowlisted_proving_key_returns_nonexportable_handle() -> None:
    policy = _production_policy()
    decision, handle = policy.select_proving_key_handle(
        "pk/prod-1",
        key_cid=_PK_CID,
        circuit_id=_CIRCUIT,
        paired_verification_key_id="vk/prod-1",
    )
    assert decision.accepted is True
    assert isinstance(handle, ProvingKeyHandle)
    assert handle.exportable is False
    assert handle.bytes_available is False
    assert handle.key_id == "pk/prod-1"
    assert handle.paired_verification_key_id == "vk/prod-1"
    with pytest.raises(TrustError, match="nonexportable"):
        handle.export_bytes()
    with pytest.raises(TrustError, match="download"):
        handle.download()
    public = handle.to_public_api()
    assert public["proving_key_exported"] is False
    assert public["exportable"] is False
    for forbidden in (
        "proving_key_bytes",
        "key_bytes",
        "private_key",
        "witness",
        "trapdoor",
    ):
        assert forbidden not in public


def test_unallowlisted_proving_key_rejects() -> None:
    policy = _production_policy()
    decision, handle = policy.select_proving_key_handle("pk/unknown")
    assert decision.accepted is False
    assert handle is None
    assert decision.reason_code == TrustRejectionReason.UNALLOWLISTED_PROVING_KEY.value


def test_substituted_proving_key_rejects() -> None:
    policy = _production_policy()
    decision, handle = policy.select_proving_key_handle(
        "pk/prod-1", key_cid=_PK_CID_B, circuit_id=_CIRCUIT
    )
    assert decision.accepted is False
    assert handle is None
    assert decision.reason_code == TrustRejectionReason.SUBSTITUTED_PROVING_KEY.value


def test_old_proving_key_rejects() -> None:
    policy = _production_policy(
        proving_keys=(_pk(superseded_by="pk/prod-2"),),
    )
    decision, handle = policy.select_proving_key_handle("pk/prod-1", key_cid=_PK_CID)
    assert decision.accepted is False
    assert handle is None
    assert decision.reason_code == TrustRejectionReason.OLD_PROVING_KEY.value


def test_test_only_proving_key_in_production_rejects() -> None:
    with pytest.raises(TrustError, match="test_only"):
        build_production_policy(
            proving_keys=(
                _pk(test_only=True, setup_origin=SetupOrigin.TEST_FIXTURE),
            )
        )


def test_paired_key_mismatch_rejects() -> None:
    policy = _production_policy()
    decision, handle = policy.select_proving_key_handle(
        "pk/prod-1",
        key_cid=_PK_CID,
        circuit_id=_CIRCUIT,
        paired_verification_key_id="vk/other",
    )
    assert decision.accepted is False
    assert handle is None
    assert decision.reason_code == TrustRejectionReason.SUBSTITUTED_PROVING_KEY.value


def test_evaluate_key_pair_accepts_matching_allowlisted_pair() -> None:
    policy = _production_policy()
    decision = policy.evaluate_key_pair(
        verification_key_id="vk/prod-1",
        proving_key_id="pk/prod-1",
        circuit_id=_CIRCUIT,
        verification_key_cid=_VK_CID,
        proving_key_cid=_PK_CID,
    )
    assert decision.accepted is True
    assert decision.subject_kind == "key_pair"
    assert decision.details["handle_only"] is True


def test_allowlisted_in_scope_signer_accepted() -> None:
    policy = _production_policy()
    decision = policy.select_signer(_SIGNER, scope="seal")
    assert decision.accepted is True
    assert decision.evidence_subset == SIGNER_TRUST_EVIDENCE
    assert "seal" in decision.details["scopes"]


def test_untrusted_signer_rejects() -> None:
    policy = _production_policy()
    decision = policy.select_signer("allowlist/unknown")
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.UNTRUSTED_SIGNER.value

    policy_untrusted = _production_policy(
        signers=(_signer(trusted=False),),
    )
    decision = policy_untrusted.select_signer(_SIGNER, scope="seal")
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.UNTRUSTED_SIGNER.value


def test_revoked_signer_rejects() -> None:
    policy = _production_policy(
        current_epoch=10,
        signers=(_signer(revocation_epoch=5),),
    )
    decision = policy.select_signer(_SIGNER, scope="seal")
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.REVOKED_SIGNER.value
    assert decision.details["revocation_epoch"] == 5

    # Before revocation epoch the same signer is still valid.
    policy_early = _production_policy(
        current_epoch=4,
        signers=(_signer(revocation_epoch=5),),
    )
    decision = policy_early.select_signer(_SIGNER, scope="seal")
    assert decision.accepted is True


def test_out_of_scope_signer_rejects() -> None:
    policy = _production_policy()
    decision = policy.select_signer(_SIGNER, scope="admin-rekey")
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.OUT_OF_SCOPE_SIGNER.value
    assert decision.details["requested_scope"] == "admin-rekey"


def test_test_only_signer_in_production_rejects_at_registration() -> None:
    with pytest.raises(TrustError, match="test_only"):
        build_production_policy(signers=(_signer(test_only=True),))

    registry = SignerTrustRegistry(production=False)
    registry.register(_signer(signer_id="allowlist/test-signer", test_only=True))
    decision = registry.evaluate("allowlist/test-signer", production=True)
    assert decision.reason_code == TrustRejectionReason.TEST_ONLY_IN_PRODUCTION.value


def test_production_never_generates_key_material() -> None:
    policy = _production_policy()
    decision = policy.generate_key_material(kind="verification_key", circuit_id=_CIRCUIT)
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.KEY_GENERATION_FORBIDDEN.value
    assert "never generates" in decision.message
    assert decision.details["key_material_generated"] is False
    # Construction also hard-clears the flags.
    assert policy.allow_key_generation is False
    assert policy.allow_key_download is False

    forced = TrustedProofPolicy(
        production=True,
        allow_key_generation=True,
        allow_key_download=True,
    )
    assert forced.allow_key_generation is False
    assert forced.allow_key_download is False
    assert forced.generate_key_material().reason_code == (
        TrustRejectionReason.KEY_GENERATION_FORBIDDEN.value
    )


def test_production_never_downloads_key_material() -> None:
    policy = _production_policy()
    decision = policy.download_key_material(
        key_id="vk/prod-1",
        kind="verification_key",
        source="https://example.invalid/keys/vk.bin",
    )
    assert decision.accepted is False
    assert decision.reason_code == TrustRejectionReason.KEY_DOWNLOAD_FORBIDDEN.value
    assert "never downloads" in decision.message
    assert decision.details["key_material_downloaded"] is False


def test_export_proving_key_bytes_always_forbidden() -> None:
    policy = _production_policy()
    decision = policy.export_proving_key_bytes("pk/prod-1")
    assert decision.accepted is False
    assert decision.reason_code == (
        TrustRejectionReason.PRODUCTION_KEY_EXPORT_FORBIDDEN.value
    )


def test_test_policy_admits_test_only_keys_but_still_blocks_generation() -> None:
    policy = _test_policy()
    vk = policy.select_verification_key(
        "vk/test-1", key_cid=_VK_CID_B, circuit_id=_CIRCUIT
    )
    assert vk.accepted is True
    pk_decision, handle = policy.select_proving_key_handle(
        "pk/test-1",
        key_cid=_PK_CID_B,
        circuit_id=_CIRCUIT,
        paired_verification_key_id="vk/test-1",
    )
    assert pk_decision.accepted is True
    assert handle is not None
    assert handle.test_only is True
    # Generation/download remain forbidden even in test policy builders.
    assert policy.generate_key_material().accepted is False
    assert policy.download_key_material(key_id="vk/test-1").accepted is False


def test_test_fixture_origin_requires_test_only_marker() -> None:
    with pytest.raises(TrustError, match="test_only"):
        VerificationKeyRecord(
            key_id="vk/bad",
            key_cid=_VK_CID,
            circuit_ids=frozenset({_CIRCUIT}),
            setup_origin=SetupOrigin.TEST_FIXTURE,
            test_only=False,
        )


def test_proving_key_record_rejects_sensitive_metadata() -> None:
    with pytest.raises(TrustError, match="sensitive field"):
        ProvingKeyRecord(
            key_id="pk/bad",
            key_cid=_PK_CID,
            circuit_ids=frozenset({_CIRCUIT}),
            setup_origin=SetupOrigin.OPERATOR_REVIEWED,
            test_only=False,
            paired_verification_key_id="vk/prod-1",
            metadata={"proving_key_bytes": b"secret"},
        )


def test_registry_rejects_duplicate_conflicting_bindings() -> None:
    registry = VerificationKeyRegistry(production=True)
    registry.register(_vk())
    with pytest.raises(TrustError, match="different binding"):
        registry.register(_vk(key_cid=_VK_CID_B))
    with pytest.raises(TrustError, match="already bound"):
        registry.register(_vk(key_id="vk/other", key_cid=_VK_CID))


def test_signer_registry_revocation_boundary() -> None:
    registry = SignerTrustRegistry(
        entries=(_signer(revocation_epoch=7),),
        production=True,
        current_epoch=7,
    )
    assert registry.evaluate(_SIGNER, scope="seal").reason_code == (
        TrustRejectionReason.REVOKED_SIGNER.value
    )
    assert registry.evaluate(_SIGNER, scope="seal", current_epoch=6).accepted is True


def test_policy_canonical_and_digest_are_stable_and_safe() -> None:
    policy = _production_policy()
    canonical = policy.to_canonical()
    assert canonical["production"] is True
    assert canonical["allow_key_generation"] is False
    assert canonical["allow_key_download"] is False
    assert canonical["key_material_generated"] is False
    assert canonical["key_material_downloaded"] is False
    assert canonical["proving_key_exported"] is False
    assert KEY_REGISTRY_EVIDENCE in canonical["evidence_subsets"]
    assert SIGNER_TRUST_EVIDENCE in canonical["evidence_subsets"]
    # Proving-key projection is handle-only.
    handles = canonical["proving_keys"]["handles"]
    assert "pk/prod-1" in handles
    assert handles["pk/prod-1"]["proving_key_exported"] is False
    assert "proving_key_bytes" not in handles["pk/prod-1"]
    digest = policy.policy_digest()
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64
    assert policy.policy_digest() == digest


def test_trust_decision_rejects_sensitive_details() -> None:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.trust import (
        TrustDecision,
    )

    with pytest.raises(TrustError, match="sensitive field"):
        TrustDecision(
            outcome=TrustOutcome.ACCEPTED,
            accepted=True,
            reason_code=None,
            message="ok",
            subject_kind="verification_key",
            subject_id="vk/1",
            evidence_subset=KEY_REGISTRY_EVIDENCE,
            details={"proving_key_bytes": "nope"},
        )


def test_unknown_setup_origin_rejected() -> None:
    with pytest.raises(TrustError, match="unknown setup_origin"):
        VerificationKeyRecord(
            key_id="vk/x",
            key_cid=_VK_CID,
            circuit_ids=frozenset({_CIRCUIT}),
            setup_origin="downloaded-from-internet",
            test_only=False,
        )


def test_disallowed_setup_origin_at_registration() -> None:
    registry = VerificationKeyRegistry(
        production=True,
        allowed_setup_origins=frozenset({SetupOrigin.CEREMONY.value}),
    )
    with pytest.raises(TrustError, match="not allowed"):
        registry.register(_vk(setup_origin=SetupOrigin.OPERATOR_REVIEWED))


def test_malformed_requests_reject_typed() -> None:
    policy = _production_policy()
    decision = policy.select_verification_key("   ")
    assert decision.reason_code == TrustRejectionReason.MALFORMED_REQUEST.value
    decision = policy.download_key_material(key_id="")
    assert decision.reason_code == TrustRejectionReason.MALFORMED_REQUEST.value
