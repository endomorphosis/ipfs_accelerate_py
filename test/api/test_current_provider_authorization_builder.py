"""Fresh signed model authority and immutable historical tuple separation."""
from copy import deepcopy
import json
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.control import profile_authority as authority
from ipfs_accelerate_py.agent_supervisor.entrypoints import protected_acceptance_advance_p019 as builder
from ipfs_accelerate_py.agent_supervisor.validation import prompt_v3_convergence as policy


@pytest.fixture
def signed_case(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profile"
    lifecycle_dir = tmp_path / "lifecycle"
    monkeypatch.setenv(authority.LIFECYCLE_DIR_ENV, str(lifecycle_dir))
    profile = authority.initialize_local_profile(
        repository_cid="repository:model-route-fixture", baseline_commit="a" * 40,
        profile_dir=profile_dir, lifecycle_dir=lifecycle_dir,
        effect_bounds=["edit", "isolated_worktree", "test"],
    )
    observed = int(time.time() * 1000)
    tree = "b" * 40
    witness = authority.export_local_profile_lifecycle_witness(
        repository_cid=profile.repository_cid, board_namespace=policy.BOARD_NAMESPACE,
        base_head=profile.baseline_commit, base_tree=tree, nonce="model-route-fixture",
        profile_dir=profile_dir, lifecycle_dir=lifecycle_dir, observed_at_ms=observed,
    )
    witness_raw = policy._canonical_json_bytes(witness)
    root_pin = dict(schema=policy.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_SCHEMA,
        board_namespace=policy.BOARD_NAMESPACE, base_head=profile.baseline_commit,
        base_tree=tree, root_identity_did=witness["root_identity_did"], pinned_at_ms=observed)
    root_pin["pin_id"] = policy._canonical_sha256(root_pin)
    root_raw = policy._canonical_json_bytes(root_pin)
    arguments = dict(witness=witness, witness_raw=witness_raw, root_pin=root_pin,
        root_pin_raw=root_raw, source_head=profile.baseline_commit, source_tree=tree,
        authorized_at_ms=observed, profile_dir=profile_dir)
    payload = json.loads(builder._build_provider_authorization_v2(**arguments))
    final = dict(reviewer_identity=profile.identity_did, profile_id=profile.profile_id,
        profile_content_id=profile.content_id, lifecycle_anchor_id=profile.lifecycle_anchor_id,
        lifecycle_generation=profile.lifecycle_generation, lifecycle_anchor_digest=witness["anchor_digest"])
    validation = dict(lifecycle_witness=policy.LocalOperatorLifecycleWitnessSnapshot(
        witness, witness_raw, builder._sha256_bytes(witness_raw)),
        root_pin=policy.LocalProfileLifecycleRootPinSnapshot(root_pin, root_raw, builder._sha256_bytes(root_raw)),
        expected_source_head=profile.baseline_commit, expected_source_tree=tree, expected_final_values=final)
    return SimpleNamespace(profile=profile, arguments=arguments, payload=payload,
        validation=validation, witness=witness, final=final)


def _resign(payload, profile_dir):
    review = {key: payload[key] for key in (
        "board_namespace", "route", "authority_bounds", "lifecycle_root_identity_did",
        "lifecycle_witness_nonce", "lifecycle_root_pin_path", "lifecycle_root_pin_sha256",
        "authorized_at_ms", "fallback_implementer_identity")}
    review["schema"] = policy.PROVIDER_FALLBACK_POLICY_REVIEW_V2_SCHEMA
    review["authorization_source"] = {key: payload["authorization_source"][key]
                                      for key in ("kind", "source_head", "source_tree")}
    review["reviewer"] = {key: value for key, value in payload["reviewer"].items() if key != "signature"}
    payload["reviewer"]["signature"] = authority.sign_profile_binding(
        profile_dir=profile_dir, payload=review)["signature"]


def test_fresh_builder_signs_current_route_and_portable_witness_validates(signed_case):
    case = signed_case
    assert case.payload["route"] == policy._CURRENT_PROVIDER_FALLBACK_AUTHORIZATION_ROUTE
    assert case.payload["route"]["route_id"] == case.profile.route_id
    assert policy.validate_local_operator_lifecycle_witness(case.witness,
        root_identity_did=case.witness["root_identity_did"],
        expected_final_values=case.final) == ()
    assert policy.ProviderFallbackPolicyAuthorization.from_dict(case.payload).validate(**case.validation) == ()


@pytest.mark.parametrize("damage", ["historical_tuple", "mixed_models", "unknown_route"])
def test_even_resigned_authorization_cannot_cross_route_or_profile_bounds(signed_case, damage):
    case = signed_case
    payload = deepcopy(case.payload)
    if damage == "historical_tuple":
        payload["route"] = deepcopy(policy._PROVIDER_FALLBACK_AUTHORIZATION_ROUTE)
    elif damage == "mixed_models":
        payload["route"]["fallback_model_id"] = "gpt-5.6-terra"
    else:
        payload["route"]["route_id"] = "unknown-route"
    _resign(payload, case.arguments["profile_dir"])
    errors = policy.ProviderFallbackPolicyAuthorization.from_dict(payload).validate(**case.validation)
    assert any(".route" in error for error in errors)
    assert not any("cryptographic verification failed" in error for error in errors)


def test_fresh_builder_refuses_historical_profile_before_signing(signed_case, monkeypatch):
    arguments = deepcopy(signed_case.arguments)
    historical = policy._PROVIDER_FALLBACK_AUTHORIZATION_ROUTE
    for key in ("route_id", "fallback_provider_id", "fallback_model_id", "fallback_reasoning_effort"):
        arguments["witness"]["profile"][key] = historical[key]
    monkeypatch.setattr(builder, "sign_profile_binding", lambda **kwargs: pytest.fail("mismatched profile signed"))
    with pytest.raises(builder.ProtectedAcceptanceDenied, match="signed reviewer profile"):
        builder._build_provider_authorization_v2(**arguments)


def test_historical_signed_tuple_remains_portably_verifiable():
    # Reuse the independently signed historical fixture without its obsolete
    # taskboard status transition. No retained archive is rewritten or upgraded.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from test.api import test_agent_supervisor_prompt_v3_convergence as fixtures
    root_key, active_key = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    head, tree, observed = "a" * 40, "b" * 40, 1_700_000_000_000
    witness, final = fixtures._lifecycle_witness_payload(root_key=root_key, active_key=active_key,
        base_head=head, base_tree=tree, observed_at_ms=observed)
    witness_raw = policy._canonical_json_bytes(witness)
    root_pin = fixtures._root_pin_payload(root_identity_did=witness["root_identity_did"],
        base_head=head, base_tree=tree, pinned_at_ms=observed)
    root_raw = policy._canonical_json_bytes(root_pin)
    payload = fixtures._fallback_authorization_v2_payload(active_key=active_key,
        witness=witness, witness_sha256=builder._sha256_bytes(witness_raw),
        root_pin=root_pin, root_pin_sha256=builder._sha256_bytes(root_raw),
        source_head=head, source_tree=tree, authorized_at_ms=observed)
    assert payload["route"] == policy._PROVIDER_FALLBACK_AUTHORIZATION_ROUTE
    assert payload["route"]["fallback_model_id"] == "gpt-5.6-terra"
    assert policy.validate_local_operator_lifecycle_witness(witness,
        root_identity_did=witness["root_identity_did"], expected_final_values=final) == ()
    assert policy.ProviderFallbackPolicyAuthorization.from_dict(payload).validate(
        lifecycle_witness=policy.LocalOperatorLifecycleWitnessSnapshot(
            witness, witness_raw, builder._sha256_bytes(witness_raw)),
        root_pin=policy.LocalProfileLifecycleRootPinSnapshot(root_pin, root_raw, builder._sha256_bytes(root_raw)),
        expected_source_head=head, expected_source_tree=tree, expected_final_values=final) == ()
