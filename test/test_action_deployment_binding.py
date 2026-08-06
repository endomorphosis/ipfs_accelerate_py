"""Signed deployment binding gate: catalog + adapter pins, confirm non-bypass."""

from __future__ import annotations

import dataclasses

import pytest

from ipfs_accelerate_py.action_runtime.catalog_211ai import (
    CATALOG_ID,
    catalog_digest,
    pilot_descriptors,
)
from ipfs_accelerate_py.action_runtime.deployment_binding import (
    PRODUCTION_ENVIRONMENT,
    SCHEMA,
    SIGNATURE_ALGORITHM,
    AdapterIdentity,
    DeploymentBindingError,
    SignedDeploymentBinding,
    adapter_identities_digest,
    adapter_identities_from_catalog,
    build_signed_pilot_binding,
    compare_adapter_identities,
    gate_production_execute,
    normalize_adapter_identities,
    pilot_adapter_identities,
    pilot_interface_identity,
    require_production_execute,
    sign_deployment_binding,
    verify_deployment_binding_signature,
)


OPERATOR_KEY = b"voice-action-032-test-operator-key"


@pytest.fixture
def identities() -> tuple[AdapterIdentity, ...]:
    return pilot_adapter_identities()


@pytest.fixture
def signed_binding(identities: tuple[AdapterIdentity, ...]) -> SignedDeploymentBinding:
    unsigned = SignedDeploymentBinding(
        binding_id="test-binding-v1",
        catalog_id=CATALOG_ID,
        catalog_digest=catalog_digest(),
        adapter_identities=identities,
        environment=PRODUCTION_ENVIRONMENT,
        issuer="test-operator",
        issued_at_epoch_s=1_700_000_000.0,
        expires_at_epoch_s=1_800_000_000.0,
        metadata={"task": "VOICE-ACTION-032"},
    )
    return sign_deployment_binding(unsigned, OPERATOR_KEY)


def test_schema_and_algorithm_constants() -> None:
    assert SCHEMA == "voice-action/deployment-binding@1"
    assert SIGNATURE_ALGORITHM == "hmac-sha256"
    assert PRODUCTION_ENVIRONMENT == "production"


def test_pilot_adapter_identities_cover_catalog() -> None:
    identities = pilot_adapter_identities()
    descriptors = pilot_descriptors()
    assert len(identities) == len(descriptors)
    by_id = {row.descriptor_id: row for row in identities}
    for descriptor in descriptors:
        identity = by_id[descriptor.descriptor_id]
        assert identity.logical_action == descriptor.logical_action
        assert identity.adapter == descriptor.adapter
        assert identity.interface_identity == pilot_interface_identity(
            descriptor.logical_action,
            descriptor_id=descriptor.descriptor_id,
        )


def test_normalize_rejects_duplicate_descriptor() -> None:
    row = pilot_adapter_identities()[0]
    with pytest.raises(ValueError, match="duplicate"):
        normalize_adapter_identities([row, row])


def test_adapter_identity_rejects_locator_tokens() -> None:
    with pytest.raises(ValueError, match="locator/credential"):
        AdapterIdentity(
            descriptor_id="voice.python.x.v1",
            logical_action="x",
            adapter="python",
            interface_identity="python:executable:/usr/bin/true",
        )


def test_sign_and_verify_round_trip(signed_binding: SignedDeploymentBinding) -> None:
    assert signed_binding.signature is not None
    assert verify_deployment_binding_signature(signed_binding, OPERATOR_KEY)
    assert not verify_deployment_binding_signature(signed_binding, b"wrong-key")
    assert signed_binding.catalog_digest == catalog_digest()
    assert signed_binding.adapter_identities_digest == adapter_identities_digest(
        signed_binding.adapter_identities
    )


def test_from_dict_round_trip(signed_binding: SignedDeploymentBinding) -> None:
    restored = SignedDeploymentBinding.from_dict(signed_binding.to_dict())
    assert restored.binding_id == signed_binding.binding_id
    assert restored.catalog_digest == signed_binding.catalog_digest
    assert restored.signature == signed_binding.signature
    assert restored.adapter_identities == signed_binding.adapter_identities
    assert verify_deployment_binding_signature(restored, OPERATOR_KEY)


def test_build_signed_pilot_binding_matches_live_catalog() -> None:
    binding = build_signed_pilot_binding(OPERATOR_KEY, binding_id="pilot-auto")
    assert binding.catalog_id == CATALOG_ID
    assert binding.catalog_digest == catalog_digest()
    assert binding.environment == PRODUCTION_ENVIRONMENT
    assert verify_deployment_binding_signature(binding, OPERATOR_KEY)


def test_gate_admits_matching_binding_without_confirm(
    signed_binding: SignedDeploymentBinding,
    identities: tuple[AdapterIdentity, ...],
) -> None:
    verdict = gate_production_execute(
        signed_binding,
        operator_key=OPERATOR_KEY,
        runtime_catalog_digest=catalog_digest(),
        runtime_adapter_identities=identities,
        confirmed=False,
        now_epoch_s=1_750_000_000.0,
    )
    assert verdict.admitted is True
    assert verdict.permits_execution is True
    assert verdict.signature_valid is True
    assert verdict.reason == "binding_match"
    assert verdict.confirmed is False


def test_gate_admits_matching_binding_with_confirm(
    signed_binding: SignedDeploymentBinding,
    identities: tuple[AdapterIdentity, ...],
) -> None:
    verdict = gate_production_execute(
        signed_binding,
        operator_key=OPERATOR_KEY,
        runtime_catalog_digest=catalog_digest(),
        runtime_adapter_identities=identities,
        confirmed=True,
        now_epoch_s=1_750_000_000.0,
    )
    assert verdict.admitted is True
    assert verdict.reason == "binding_match_confirmed"
    assert verdict.confirmed is True


def test_catalog_digest_mismatch_denies_even_when_confirmed(
    signed_binding: SignedDeploymentBinding,
    identities: tuple[AdapterIdentity, ...],
) -> None:
    bad_digest = "0" * 64
    assert bad_digest != signed_binding.catalog_digest
    verdict = gate_production_execute(
        signed_binding,
        operator_key=OPERATOR_KEY,
        runtime_catalog_digest=bad_digest,
        runtime_adapter_identities=identities,
        confirmed=True,
        now_epoch_s=1_750_000_000.0,
    )
    assert verdict.admitted is False
    assert verdict.permits_execution is False
    assert verdict.reason == "catalog_digest_mismatch"
    assert verdict.confirmed is True
    assert verdict.signature_valid is True
    assert verdict.expected_catalog_digest == signed_binding.catalog_digest
    assert verdict.actual_catalog_digest == bad_digest


def test_adapter_identity_mismatch_denies_even_when_confirmed(
    signed_binding: SignedDeploymentBinding,
    identities: tuple[AdapterIdentity, ...],
) -> None:
    # Mutate a single interface identity while keeping catalog digest intact.
    mutated = list(identities)
    target = mutated[0]
    mutated[0] = AdapterIdentity(
        descriptor_id=target.descriptor_id,
        logical_action=target.logical_action,
        adapter=target.adapter,
        interface_identity=f"tampered:{target.logical_action}:{target.descriptor_id}",
    )
    verdict = gate_production_execute(
        signed_binding,
        operator_key=OPERATOR_KEY,
        runtime_catalog_digest=catalog_digest(),
        runtime_adapter_identities=mutated,
        confirmed=True,
        now_epoch_s=1_750_000_000.0,
    )
    assert verdict.admitted is False
    assert verdict.permits_execution is False
    assert verdict.reason == f"adapter_identity_mismatch:{target.descriptor_id}"
    assert verdict.confirmed is True
    assert verdict.signature_valid is True


def test_adapter_identity_set_mismatch_denies_when_confirmed(
    signed_binding: SignedDeploymentBinding,
    identities: tuple[AdapterIdentity, ...],
) -> None:
    truncated = identities[:-1]
    assert len(truncated) < len(identities)
    verdict = gate_production_execute(
        signed_binding,
        operator_key=OPERATOR_KEY,
        runtime_catalog_digest=catalog_digest(),
        runtime_adapter_identities=truncated,
        confirmed=True,
        now_epoch_s=1_750_000_000.0,
    )
    assert verdict.admitted is False
    assert verdict.reason.startswith("adapter_identity_set_mismatch")
    assert verdict.confirmed is True


def test_invalid_signature_denies_even_when_confirmed(
    signed_binding: SignedDeploymentBinding,
    identities: tuple[AdapterIdentity, ...],
) -> None:
    forged = signed_binding.with_signature("a" * 64)
    verdict = gate_production_execute(
        forged,
        operator_key=OPERATOR_KEY,
        runtime_catalog_digest=catalog_digest(),
        runtime_adapter_identities=identities,
        confirmed=True,
        now_epoch_s=1_750_000_000.0,
    )
    assert verdict.admitted is False
    assert verdict.reason == "invalid_or_missing_binding_signature"
    assert verdict.signature_valid is False
    assert verdict.confirmed is True


def test_missing_signature_denies(
    identities: tuple[AdapterIdentity, ...],
) -> None:
    unsigned = SignedDeploymentBinding(
        binding_id="unsigned",
        catalog_id=CATALOG_ID,
        catalog_digest=catalog_digest(),
        adapter_identities=identities,
        environment=PRODUCTION_ENVIRONMENT,
    )
    verdict = gate_production_execute(
        unsigned,
        operator_key=OPERATOR_KEY,
        runtime_catalog_digest=catalog_digest(),
        runtime_adapter_identities=identities,
        confirmed=True,
    )
    assert verdict.admitted is False
    assert verdict.reason == "invalid_or_missing_binding_signature"


def test_expired_binding_denies_when_confirmed(
    signed_binding: SignedDeploymentBinding,
    identities: tuple[AdapterIdentity, ...],
) -> None:
    verdict = gate_production_execute(
        signed_binding,
        operator_key=OPERATOR_KEY,
        runtime_catalog_digest=catalog_digest(),
        runtime_adapter_identities=identities,
        confirmed=True,
        now_epoch_s=1_900_000_000.0,  # after expires_at
    )
    assert verdict.admitted is False
    assert verdict.reason == "binding_expired"
    assert verdict.signature_valid is True


def test_non_production_environment_denies(
    identities: tuple[AdapterIdentity, ...],
) -> None:
    pilot = sign_deployment_binding(
        SignedDeploymentBinding(
            binding_id="pilot-env",
            catalog_id=CATALOG_ID,
            catalog_digest=catalog_digest(),
            adapter_identities=identities,
            environment="pilot",
        ),
        OPERATOR_KEY,
    )
    verdict = gate_production_execute(
        pilot,
        operator_key=OPERATOR_KEY,
        runtime_catalog_digest=catalog_digest(),
        runtime_adapter_identities=identities,
        confirmed=True,
    )
    assert verdict.admitted is False
    assert verdict.reason == "binding_environment_not_production"


def test_require_production_execute_raises_on_mismatch(
    signed_binding: SignedDeploymentBinding,
    identities: tuple[AdapterIdentity, ...],
) -> None:
    with pytest.raises(DeploymentBindingError, match="catalog_digest_mismatch") as exc:
        require_production_execute(
            signed_binding,
            operator_key=OPERATOR_KEY,
            runtime_catalog_digest="f" * 64,
            runtime_adapter_identities=identities,
            confirmed=True,
            now_epoch_s=1_750_000_000.0,
        )
    assert exc.value.verdict.confirmed is True
    assert exc.value.verdict.admitted is False


def test_require_production_execute_returns_on_match(
    signed_binding: SignedDeploymentBinding,
    identities: tuple[AdapterIdentity, ...],
) -> None:
    verdict = require_production_execute(
        signed_binding,
        operator_key=OPERATOR_KEY,
        runtime_catalog_digest=catalog_digest(),
        runtime_adapter_identities=identities,
        confirmed=True,
        now_epoch_s=1_750_000_000.0,
    )
    assert verdict.admitted is True


def test_compare_adapter_identities_detects_field_drift() -> None:
    base = pilot_adapter_identities()
    other = list(base)
    first = other[0]
    other[0] = dataclasses.replace(first, adapter="cli")
    reason = compare_adapter_identities(base, other)
    assert reason == f"adapter_identity_mismatch:{first.descriptor_id}"
    assert compare_adapter_identities(base, base) is None


def test_adapter_identities_from_catalog_override() -> None:
    descriptors = pilot_descriptors()[:2]
    override_id = descriptors[0].descriptor_id
    rows = adapter_identities_from_catalog(
        descriptors,
        interface_identities={override_id: f"custom:surface:{override_id}"},
    )
    assert rows[0].interface_identity == f"custom:surface:{override_id}"


def test_tampered_payload_fails_signature(
    signed_binding: SignedDeploymentBinding,
) -> None:
    # Re-construct with a different catalog digest but keep the old signature.
    tampered = SignedDeploymentBinding(
        binding_id=signed_binding.binding_id,
        catalog_id=signed_binding.catalog_id,
        catalog_digest="1" * 64,
        adapter_identities=signed_binding.adapter_identities,
        environment=signed_binding.environment,
        issuer=signed_binding.issuer,
        signature_algorithm=signed_binding.signature_algorithm,
        nonce=signed_binding.nonce,
        signature=signed_binding.signature,
        issued_at_epoch_s=signed_binding.issued_at_epoch_s,
        expires_at_epoch_s=signed_binding.expires_at_epoch_s,
        metadata=signed_binding.metadata,
    )
    assert not verify_deployment_binding_signature(tampered, OPERATOR_KEY)
    verdict = gate_production_execute(
        tampered,
        operator_key=OPERATOR_KEY,
        runtime_catalog_digest="1" * 64,
        runtime_adapter_identities=signed_binding.adapter_identities,
        confirmed=True,
        now_epoch_s=1_750_000_000.0,
    )
    assert verdict.admitted is False
    assert verdict.reason == "invalid_or_missing_binding_signature"


def test_verdict_to_dict_is_json_friendly(
    signed_binding: SignedDeploymentBinding,
    identities: tuple[AdapterIdentity, ...],
) -> None:
    verdict = gate_production_execute(
        signed_binding,
        operator_key=OPERATOR_KEY,
        runtime_catalog_digest=catalog_digest(),
        runtime_adapter_identities=identities,
        confirmed=True,
        now_epoch_s=1_750_000_000.0,
    )
    payload = verdict.to_dict()
    assert payload["admitted"] is True
    assert payload["permits_execution"] is True
    assert payload["confirmed"] is True
    assert payload["binding_id"] == "test-binding-v1"
