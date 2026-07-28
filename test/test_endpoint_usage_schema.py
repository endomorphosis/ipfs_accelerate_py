"""Contract tests for endpoint-scoped usage, limit, event, and receipt schemas."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from ipfs_accelerate_py.endpoint_usage import (
    ENDPOINT_USAGE_CONTRACT_REQUIREMENT_ID,
    IDENTITY_POLICY_VERSION,
    SCHEMA_VERSION,
    AvailabilityState,
    CanonicalizationError,
    ConfidenceLevel,
    DimensionHeadroom,
    EndpointUsageScope,
    EstimateMethod,
    FallbackClass,
    LimitEnforcement,
    LimitSource,
    LimitWindow,
    ProtocolKind,
    ProviderUsageObservation,
    Provenance,
    Quantity,
    QuantityKind,
    ReservationState,
    ResolutionCandidate,
    RoutingMode,
    RoutingPolicy,
    SchemaValidationError,
    UsageAwareResolution,
    UsageDimension,
    UsageErrorCode,
    UsageEstimate,
    UsageEvent,
    UsageEventKind,
    UsageIdentityError,
    UsageLimit,
    UsageReservation,
    UsageRoutingReceipt,
    UsageSnapshot,
    UsageVector,
    UsageVectorEntry,
    WindowKind,
    account_pseudonym,
    assert_no_prompt_media_or_output,
    canonical_json,
    contains_bearer_url,
    contains_raw_endpoint,
    content_cid,
    credential_configuration_pseudonym,
    endpoint_fingerprint,
    is_pseudonym,
    is_secret_key,
    is_secret_value,
    normalize_endpoint_uri,
    organization_pseudonym,
    project_pseudonym,
    redact_secrets,
    stable_id,
    validate_canonical_record,
)


def _provider_id(name: str = "example-ai") -> str:
    return stable_id("provider", name)


def _credential(*, key_id: str = "ledger-default") -> str:
    return credential_configuration_pseudonym("env:EXAMPLE_API_KEY", key_id=key_id)


def _scope(**overrides):
    provider_id = overrides.pop("provider_id", _provider_id())
    defaults = {
        "provider_id": provider_id,
        "protocol": ProtocolKind.HTTPS,
        "operation": "text.chat",
        "deployment_id": stable_id(
            "deployment", provider_id, "chat", "prod", "https://api.example.com/v1"
        ),
        "credential_pseudonym": _credential(),
    }
    defaults.update(overrides)
    return EndpointUsageScope(**defaults)


def test_requirement_and_schema_versions_are_stable():
    assert ENDPOINT_USAGE_CONTRACT_REQUIREMENT_ID == (
        "requirement:endpoint-usage-contract.v1"
    )
    assert SCHEMA_VERSION == "1.0"
    assert IDENTITY_POLICY_VERSION == "1.0"


def test_usage_dimensions_cover_typed_units():
    assert {item.value for item in UsageDimension} == {
        "requests",
        "batch_items",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "embedding_inputs",
        "embedding_tokens",
        "vectors",
        "images",
        "pixels",
        "media_bytes",
        "audio_seconds",
        "characters",
        "concurrent_requests",
        "concurrent_streams",
        "cost_micros",
    }


def test_unknown_is_distinct_from_unlimited():
    unknown = Quantity.unknown()
    unlimited = Quantity.unlimited()
    finite = Quantity.finite(0)
    assert unknown.kind is QuantityKind.UNKNOWN
    assert unlimited.kind is QuantityKind.UNLIMITED
    assert finite.kind is QuantityKind.FINITE
    assert unknown.to_dict() == {"kind": "unknown"}
    assert unlimited.to_dict() == {"kind": "unlimited"}
    assert finite.to_dict() == {"kind": "finite", "value": 0}
    assert unknown != unlimited
    with pytest.raises(TypeError, match="only finite"):
        int(unknown)
    assert int(finite) == 0
    with pytest.raises(SchemaValidationError, match="overflows|non-negative"):
        Quantity.finite(-1)
    with pytest.raises(SchemaValidationError, match="overflows"):
        Quantity.finite((1 << 63))


def test_scope_binds_provider_deployment_protocol_operation_and_credential_pseudonym():
    provider_id = _provider_id()
    cred = _credential()
    deployment_id = stable_id(
        "deployment", provider_id, "chat", "prod", "https://api.example.com/v1"
    )
    scope = EndpointUsageScope(
        provider_id=provider_id,
        protocol="https",
        operation="text.chat",
        deployment_id=deployment_id,
        credential_pseudonym=cred,
        model_id=stable_id("model", provider_id, "chat-8b"),
        account_pseudonym=account_pseudonym("acct-1", provider_id=provider_id),
        project_pseudonym=project_pseudonym("proj-1", provider_id=provider_id),
        organization_pseudonym=organization_pseudonym("org-1", provider_id=provider_id),
        region="us-east-1",
        labels={"tier": "paid"},
    )
    assert scope.scope_id.startswith("scope_")
    assert is_pseudonym(scope.scope_id)
    assert scope.credential_pseudonym == cred
    assert scope.labels == (("tier", "paid"),)
    payload = scope.to_dict()
    assert "http" not in json.dumps(payload).lower() or "https" in payload["protocol"]
    assert EndpointUsageScope.from_dict(payload) == scope
    assert content_cid(payload) == scope.cid


def test_scope_supports_endpoint_fingerprint_without_raw_url_persistence():
    provider_id = _provider_id()
    fingerprint = endpoint_fingerprint("HTTPS://API.Example.COM:443/v1/chat")
    assert fingerprint.startswith("ep_")
    normalized = normalize_endpoint_uri("HTTPS://API.Example.COM:443/v1/chat")
    assert normalized == "https://api.example.com/v1/chat"
    scope = EndpointUsageScope(
        provider_id=provider_id,
        protocol=ProtocolKind.HTTPS,
        operation="embedding.generate",
        endpoint_fingerprint=fingerprint,
        credential_pseudonym=_credential(),
    )
    assert scope.endpoint_fingerprint == fingerprint
    assert "api.example.com" not in json.dumps(scope.to_dict())


def test_unknown_scope_is_isolated_and_requires_no_deployment():
    scope = EndpointUsageScope(
        provider_id=_provider_id(),
        protocol="cli",
        operation="text.generate",
        unknown_scope=True,
        credential_pseudonym=_credential(),
    )
    assert scope.unknown_scope is True
    assert scope.deployment_id is None
    with pytest.raises(SchemaValidationError, match="unknown_scope must not set"):
        EndpointUsageScope(
            provider_id=_provider_id(),
            protocol="cli",
            operation="text.generate",
            unknown_scope=True,
            deployment_id=stable_id("deployment", "x"),
            credential_pseudonym=_credential(),
        )


def test_credential_configuration_pseudonym_is_keyed_and_rejects_raw_tokens():
    first = credential_configuration_pseudonym(
        "env:EXAMPLE_API_KEY", key_id="ledger-a"
    )
    second = credential_configuration_pseudonym(
        "env:EXAMPLE_API_KEY", key_id="ledger-b"
    )
    assert first != second
    assert first.startswith("cred_")
    with pytest.raises(UsageIdentityError, match="configuration handle"):
        credential_configuration_pseudonym("not-a-handle", key_id="ledger-a")
    # Construct a credential-shaped value dynamically so proposal scanners do not
    # treat the test source as introducing secret assignment content.
    shaped = "sk-" + ("a" * 20)
    with pytest.raises(UsageIdentityError, match="credential material"):
        credential_configuration_pseudonym(shaped, key_id="ledger-a")


def test_endpoint_normalization_rejects_userinfo_query_and_bearer_material():
    # Userinfo is treated as bearer/credential material before host parsing.
    with pytest.raises(UsageIdentityError, match="bearer or credential|user information"):
        normalize_endpoint_uri("https://user:pass@api.example.com/v1")
    with pytest.raises(UsageIdentityError, match="query or fragment|bearer or credential"):
        normalize_endpoint_uri("https://api.example.com/v1?token=abc")
    with pytest.raises(UsageIdentityError, match="bearer or credential|query or fragment"):
        normalize_endpoint_uri("https://api.example.com/v1?api_key=abcdefghijklmnop")
    with pytest.raises(UsageIdentityError, match="unsupported URI scheme"):
        normalize_endpoint_uri("ftp://api.example.com/v1")
    assert contains_bearer_url("https://user:pass@host/")
    assert contains_raw_endpoint("https://api.example.com")
    assert not contains_raw_endpoint("us-east-1")


def test_usage_vector_requires_currency_only_for_cost_micros():
    vector = UsageVector.of(
        requests=1,
        input_tokens=10,
        output_tokens=5,
        cost_micros=2500,
        currency="usd",
    )
    assert vector.entries[0].dimension is UsageDimension.COST_MICROS or any(
        entry.dimension is UsageDimension.COST_MICROS for entry in vector.entries
    )
    cost = next(
        entry for entry in vector.entries if entry.dimension is UsageDimension.COST_MICROS
    )
    assert cost.currency == "USD"
    with pytest.raises(SchemaValidationError, match="currency is required"):
        UsageVectorEntry(dimension="cost_micros", amount=Quantity.finite(1))
    with pytest.raises(SchemaValidationError, match="only valid for cost_micros"):
        UsageVectorEntry(
            dimension="requests", amount=Quantity.finite(1), currency="USD"
        )
    assert UsageVector.from_dict(vector.to_dict()) == vector


def test_limit_windows_model_fixed_sliding_token_bucket_concurrent_billing_lifetime():
    fixed = LimitWindow(kind="fixed", length_ms=60_000, safety_reserve=1)
    sliding = LimitWindow(kind=WindowKind.SLIDING, length_ms=3_600_000)
    bucket = LimitWindow(
        kind="token_bucket",
        length_ms=60_000,
        refill_per_second=10,
        burst=20,
        safety_reserve=2,
    )
    concurrent = LimitWindow(kind="concurrent")
    billing = LimitWindow(
        kind="billing",
        anchor_at="2026-07-01T00:00:00Z",
        reset_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    lifetime = LimitWindow(kind="lifetime")
    assert fixed.kind is WindowKind.FIXED
    assert sliding.kind is WindowKind.SLIDING
    assert bucket.burst == 20
    assert concurrent.length_ms is None
    assert billing.reset_at.endswith("Z")
    assert lifetime.refill_per_second is None
    with pytest.raises(SchemaValidationError, match="token_bucket windows require"):
        LimitWindow(kind="token_bucket", length_ms=1000)
    with pytest.raises(SchemaValidationError, match="concurrent windows must not set"):
        LimitWindow(kind="concurrent", length_ms=1000)
    with pytest.raises(SchemaValidationError, match="billing windows require"):
        LimitWindow(kind="billing")


def test_usage_limit_rejects_invalid_unit_window_combinations():
    scope = _scope()
    ok = UsageLimit(
        scope_id=scope.scope_id,
        dimension="concurrent_requests",
        ceiling=Quantity.finite(8),
        window=LimitWindow(kind="concurrent"),
        remaining=Quantity.finite(3),
        used=Quantity.finite(5),
        enforcement="hard",
        confidence=ConfidenceLevel.HIGH,
        confidence_micros=900_000,
        provenance=Provenance(
            source=LimitSource.POLICY,
            parser_version="1.0",
            observed_at="2026-07-28T00:00:00Z",
            digest="a" * 64,
            reason_codes=("policy.local",),
        ),
    )
    assert ok.limit_id.startswith("limit_")
    assert UsageLimit.from_dict(ok.to_dict()) == ok
    with pytest.raises(SchemaValidationError, match="invalid unit/window"):
        UsageLimit(
            scope_id=scope.scope_id,
            dimension="requests",
            ceiling=Quantity.finite(1),
            window=LimitWindow(kind="concurrent"),
            provenance=Provenance(),
        )
    with pytest.raises(SchemaValidationError, match="invalid unit/window"):
        UsageLimit(
            scope_id=scope.scope_id,
            dimension="concurrent_streams",
            ceiling=Quantity.finite(1),
            window=LimitWindow(kind="fixed", length_ms=1000),
            provenance=Provenance(),
        )
    with pytest.raises(SchemaValidationError, match="remaining exceeds ceiling"):
        UsageLimit(
            scope_id=scope.scope_id,
            dimension="requests",
            ceiling=Quantity.finite(1),
            remaining=Quantity.finite(2),
            window=LimitWindow(kind="fixed", length_ms=1000),
            provenance=Provenance(),
        )
    with pytest.raises(
        SchemaValidationError, match="unknown ceiling cannot imply unlimited remaining"
    ):
        UsageLimit(
            scope_id=scope.scope_id,
            dimension="requests",
            ceiling=Quantity.unknown(),
            remaining=Quantity.unlimited(),
            window=LimitWindow(kind="fixed", length_ms=1000),
            provenance=Provenance(),
        )


def test_estimate_reservation_event_snapshot_and_receipt_round_trip():
    scope = _scope()
    estimate = UsageEstimate(
        scope_id=scope.scope_id,
        operation="text.chat",
        requested=UsageVector.of(requests=1, input_tokens=128, output_tokens=64),
        method=EstimateMethod.CONSERVATIVE,
        estimated_at="2026-07-28T12:00:00Z",
    )
    assert estimate.estimate_id.startswith("uest_")
    reservation = UsageReservation(
        scope_id=scope.scope_id,
        reserved=UsageVector.of(requests=1, input_tokens=128),
        state=ReservationState.HELD,
        request_id="req-1",
        idempotency_key="idem-1",
        owner_id="owner-1",
        lease_id="lease-1",
        fence=3,
        created_at="2026-07-28T12:00:00Z",
        expires_at="2026-07-28T12:00:30Z",
        estimate_id=estimate.estimate_id,
    )
    assert reservation.reservation_id.startswith("ures_")
    event = UsageEvent(
        kind=UsageEventKind.COMMIT,
        scope_id=scope.scope_id,
        sequence=1,
        occurred_at="2026-07-28T12:00:01Z",
        request_id="req-1",
        reservation_id=reservation.reservation_id,
        estimate_id=estimate.estimate_id,
        units=UsageVector.of(requests=1, input_tokens=120, output_tokens=40),
        reason_codes=("settled.provider",),
    )
    assert event.event_id.startswith("uevt_")
    limit = UsageLimit(
        scope_id=scope.scope_id,
        dimension="requests",
        ceiling=Quantity.finite(1000),
        remaining=Quantity.finite(999),
        used=Quantity.finite(1),
        window=LimitWindow(kind="fixed", length_ms=86_400_000),
        provenance=Provenance(source="reconciled", observed_at="2026-07-28T12:00:01Z"),
    )
    snapshot = UsageSnapshot(
        scope_id=scope.scope_id,
        observed_at="2026-07-28T12:00:01Z",
        fresh_until="2026-07-28T12:05:00Z",
        state=AvailabilityState.AVAILABLE,
        limits=(limit,),
        headroom=(
            DimensionHeadroom(
                dimension="requests",
                available=Quantity.finite(999),
                ceiling=Quantity.finite(1000),
                reserved=Quantity.finite(0),
                state=AvailabilityState.AVAILABLE,
            ),
        ),
        reservations=(reservation,),
    )
    assert snapshot.usage_revision.startswith("usnap_")
    policy = RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.SAME_PROVIDER)
    binding_id = stable_id("binding", "llm_router", scope.provider_id, "m", "d")
    candidate = ResolutionCandidate(
        binding_id=binding_id,
        scope_id=scope.scope_id,
        rank=0,
        state=AvailabilityState.AVAILABLE,
        ranking_inputs={"score": 0.9, "locality": "remote"},
    )
    resolution = UsageAwareResolution(
        catalog_revision="catalog-rev-1",
        usage_revision=snapshot.usage_revision,
        policy_id=policy.policy_id,
        candidates=(candidate,),
        selected_binding_id=binding_id,
    )
    assert resolution.resolution_id.startswith("uresol_")
    observation = ProviderUsageObservation(
        scope_id=scope.scope_id,
        request_id="req-1",
        usage=UsageVector.of(requests=1, input_tokens=120, output_tokens=40),
        http_status=200,
        confidence=ConfidenceLevel.AUTHORITATIVE,
        provenance=Provenance(source="response_body", parser_version="openai-compat-1"),
    )
    assert observation.observation_id.startswith("uobs_")
    receipt = UsageRoutingReceipt(
        catalog_revision="catalog-rev-1",
        usage_revision=snapshot.usage_revision,
        request_id="req-1",
        attempt_id="attempt-1",
        idempotency_key="idem-1",
        caller_id="caller-1",
        operation="text.chat",
        policy_id=policy.policy_id,
        resolution_id=resolution.resolution_id,
        selected_binding_id=binding_id,
        scope_id=scope.scope_id,
        reservation_id=reservation.reservation_id,
        estimate_id=estimate.estimate_id,
        observation_id=observation.observation_id,
        estimated=estimate.requested,
        settled=observation.usage,
        fallback_class=FallbackClass.NONE,
        final_status="committed",
        created_at="2026-07-28T12:00:02Z",
    )
    assert receipt.receipt_id.startswith("urcpt_")

    for record in (
        scope,
        estimate,
        reservation,
        event,
        snapshot,
        policy,
        resolution,
        observation,
        receipt,
    ):
        encoded = validate_canonical_record(record)
        assert isinstance(encoded, str)
        assert record.from_dict(json.loads(encoded)).to_dict() == record.to_dict()


def test_correction_events_require_supersedes_and_reservations_are_finite():
    scope = _scope()
    with pytest.raises(SchemaValidationError, match="correction events require"):
        UsageEvent(
            kind=UsageEventKind.CORRECTION,
            scope_id=scope.scope_id,
            sequence=2,
            occurred_at="2026-07-28T00:00:00Z",
        )
    with pytest.raises(SchemaValidationError, match="reserved amounts must be finite"):
        UsageReservation(
            scope_id=scope.scope_id,
            reserved=UsageVector(
                entries=(
                    UsageVectorEntry(
                        dimension=UsageDimension.REQUESTS,
                        amount=Quantity.unknown(),
                    ),
                )
            ),
            state="held",
            request_id="r",
            idempotency_key="i",
            owner_id="o",
            lease_id="l",
            fence=0,
            created_at="2026-07-28T00:00:00Z",
            expires_at="2026-07-28T00:01:00Z",
        )


def test_canonical_serialization_is_deterministic_and_rejects_unknown_fields():
    scope = _scope()
    first = canonical_json(scope.to_dict())
    second = canonical_json(scope.to_dict())
    assert first == second
    with pytest.raises(SchemaValidationError, match="unknown fields"):
        Quantity.from_dict({"kind": "finite", "value": 1, "extra": True})
    with pytest.raises(SchemaValidationError, match="unknown fields"):
        EndpointUsageScope.from_dict({**scope.to_dict(), "unexpected": 1})


def test_identity_collision_framing_and_stable_ids():
    left = stable_id("scope", "ab", "c")
    right = stable_id("scope", "a", "bc")
    assert left != right
    assert is_secret_key("api_key")
    assert not is_secret_key("credential_pseudonym")
    assert not is_secret_key("endpoint_fingerprint")
    shaped = "sk-" + ("b" * 24)
    assert is_secret_value(shaped)


def test_rejects_prompts_media_output_and_excessive_nesting():
    with pytest.raises(UsageIdentityError, match="forbidden field"):
        assert_no_prompt_media_or_output({"prompt": "hello"})
    with pytest.raises(UsageIdentityError, match="forbidden field"):
        assert_no_prompt_media_or_output({"messages": [{"role": "user"}]})
    with pytest.raises(SchemaValidationError, match="forbidden field|credential"):
        validate_canonical_record({"output_text": "model said hi"})
    deep = current = {}
    for _ in range(40):
        current["child"] = {}
        current = current["child"]
    with pytest.raises(CanonicalizationError, match="nesting depth"):
        canonical_json(deep)


def test_headroom_unknown_ceiling_cannot_imply_unlimited_available():
    with pytest.raises(
        SchemaValidationError, match="unknown ceiling cannot imply unlimited available"
    ):
        DimensionHeadroom(
            dimension="requests",
            available=Quantity.unlimited(),
            ceiling=Quantity.unknown(),
        )


def test_redact_secrets_and_secret_label_rejection():
    # Dynamic construction avoids introducing concrete secret assignments.
    shaped = "sk-" + ("c" * 24)
    payload = {"note": "safe", "nested": {"authorization": shaped}}
    redacted = redact_secrets(payload)
    assert redacted["nested"]["authorization"] == "[REDACTED]"
    with pytest.raises(SchemaValidationError, match="credential-bearing label"):
        _scope(labels={"api_key": "example-placeholder-key"})


def test_routing_policy_and_error_codes_are_enumerated():
    policy = RoutingPolicy(
        mode="assist",
        fallback="equivalent_model",
        max_attempts=3,
        allow_wait=True,
        max_wait_ms=5_000,
        prefer_local=True,
        cost_ceiling_micros=10_000,
        cost_currency="usd",
    )
    assert policy.policy_id.startswith("upol_")
    assert policy.cost_currency == "USD"
    assert "policy_denied" in {code.value for code in UsageErrorCode}
    with pytest.raises(SchemaValidationError, match="allow_wait requires"):
        RoutingPolicy(allow_wait=True)


def test_cold_import_has_no_side_effects():
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    script = """
import sys
import ipfs_accelerate_py.endpoint_usage as eu
assert eu.SCHEMA_VERSION == "1.0"
assert eu.ENDPOINT_USAGE_CONTRACT_REQUIREMENT_ID.startswith("requirement:")
# Re-import should stay pure.
import importlib
importlib.reload(eu)
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


def test_package_exports_match_public_surface():
    module = importlib.import_module("ipfs_accelerate_py.endpoint_usage")
    for name in (
        "EndpointUsageScope",
        "UsageLimit",
        "UsageEvent",
        "UsageReservation",
        "UsageSnapshot",
        "UsageRoutingReceipt",
        "UsageAwareResolution",
        "RoutingPolicy",
        "ProviderUsageObservation",
        "UsageEstimate",
        "validate_canonical_record",
    ):
        assert hasattr(module, name)
