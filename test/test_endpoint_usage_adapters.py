"""Offline tests for configured and provider-observed usage adapters."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from ipfs_accelerate_py.endpoint_usage.adapters import (
    ADAPTER_PARSER_VERSION,
    AdapterParseError,
    AdapterScopeError,
    ObservationInput,
    PROVIDER_USAGE_ADAPTER_REQUIREMENT_ID,
    apply_policy_ceiling_guard,
    normalize_configured_limits,
    parse_anthropic_observation,
    parse_cli_observation,
    parse_custom_observation,
    parse_huggingface_observation,
    parse_local_observation,
    parse_openai_compatible_observation,
    parse_provider_observation,
    retain_restrictive_cooldown,
)
from ipfs_accelerate_py.endpoint_usage.identity import (
    account_pseudonym,
    credential_configuration_pseudonym,
    project_pseudonym,
    stable_id,
)
from ipfs_accelerate_py.endpoint_usage.provider_registry import (
    ADAPTER_REGISTRY_VERSION,
    AdapterError,
    AdapterFamily,
    clear_custom_adapters,
    get_adapter_descriptor,
    is_registered_adapter,
    known_adapter_ids,
    list_adapter_descriptors,
    register_custom_adapter,
    resolve_adapter_family,
    unregister_custom_adapter,
)
from ipfs_accelerate_py.endpoint_usage.schema import (
    ConfidenceLevel,
    EndpointUsageScope,
    LimitSource,
    LimitWindow,
    ProtocolKind,
    ProviderUsageObservation,
    Quantity,
    QuantityKind,
    UsageDimension,
    UsageLimit,
    WindowKind,
)

FIXED_NOW = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
FIXED_NOW_TEXT = "2026-07-28T12:00:00.000000Z"


def _provider_id(name: str = "openai") -> str:
    return stable_id("provider", name)


def _credential(key_id: str = "ledger-default") -> str:
    return credential_configuration_pseudonym("env:EXAMPLE_API_KEY", key_id=key_id)


def _scope(**overrides) -> EndpointUsageScope:
    provider_id = overrides.pop("provider_id", _provider_id())
    defaults = {
        "provider_id": provider_id,
        "protocol": ProtocolKind.HTTPS,
        "operation": "text.chat",
        "deployment_id": stable_id(
            "deployment", provider_id, "chat", "prod", "https://api.example.com/v1"
        ),
        "credential_pseudonym": _credential(),
        "model_id": stable_id("model", provider_id, "chat-model"),
        "account_pseudonym": account_pseudonym("acct-1", provider_id=provider_id),
        "project_pseudonym": project_pseudonym("proj-1", provider_id=provider_id),
    }
    defaults.update(overrides)
    return EndpointUsageScope(**defaults)


def _base_input(**overrides):
    data = {
        "scope": _scope(),
        "request_id": "req-test-1",
        "observed_at": FIXED_NOW,
        "now": FIXED_NOW,
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_requirement_ids_and_registry_versions_are_stable():
    assert PROVIDER_USAGE_ADAPTER_REQUIREMENT_ID == (
        "requirement:provider-usage-adapter.v1"
    )
    assert ADAPTER_PARSER_VERSION == "1.0"
    assert ADAPTER_REGISTRY_VERSION == "1.0"


def test_builtin_adapter_families_cover_conformance_set():
    ids = set(known_adapter_ids())
    assert {
        "openai_compatible",
        "anthropic",
        "huggingface",
        "cli",
        "local",
        "custom",
        "unknown",
    } <= ids
    assert resolve_adapter_family("openai") is AdapterFamily.OPENAI_COMPATIBLE
    assert resolve_adapter_family("xai") is AdapterFamily.OPENAI_COMPATIBLE
    assert resolve_adapter_family("claude") is AdapterFamily.ANTHROPIC
    assert resolve_adapter_family("hf_tgi") is AdapterFamily.HUGGINGFACE
    assert resolve_adapter_family("codex") is AdapterFamily.CLI
    assert resolve_adapter_family("llama_cpp") is AdapterFamily.LOCAL
    assert resolve_adapter_family("not-a-real-provider") is AdapterFamily.UNKNOWN
    assert resolve_adapter_family(None, protocol=ProtocolKind.CLI) is AdapterFamily.CLI
    assert is_registered_adapter("openrouter")
    assert not is_registered_adapter("definitely-missing")
    descriptor = get_adapter_descriptor("openai_compatible")
    assert descriptor.supports_headers is True
    assert descriptor.supports_body_usage is True
    assert list_adapter_descriptors()


def test_custom_adapter_registration_is_process_local_and_non_overwriting():
    clear_custom_adapters()
    registered = register_custom_adapter(
        {
            "family": "custom",
            "adapter_id": "acme_widgets",
            "aliases": ("acme",),
            "description": "Acme widget usage map",
            "supports_headers": True,
            "supports_body_usage": True,
        }
    )
    assert registered.adapter_id == "acme_widgets"
    assert resolve_adapter_family("acme_widgets") is AdapterFamily.CUSTOM
    with pytest.raises(AdapterError, match="built-in"):
        register_custom_adapter(
            {
                "family": "custom",
                "adapter_id": "openai_compatible",
            }
        )
    assert unregister_custom_adapter("acme_widgets") is True
    clear_custom_adapters()


# ---------------------------------------------------------------------------
# Configured limits
# ---------------------------------------------------------------------------


def test_normalize_configured_limits_binds_scope_and_rejects_mismatch():
    scope = _scope()
    limits = normalize_configured_limits(
        scope,
        [
            {
                "dimension": "requests",
                "ceiling": 1000,
                "remaining": 900,
                "window": {"kind": "fixed", "length_ms": 60_000},
            },
            {
                "dimension": "concurrent_requests",
                "ceiling": {"kind": "finite", "value": 8},
                "window": {"kind": "concurrent"},
            },
        ],
        observed_at=FIXED_NOW,
    )
    assert len(limits) == 2
    assert all(item.scope_id == scope.scope_id for item in limits)
    assert limits[0].provenance.source is LimitSource.CONFIGURED
    assert limits[0].ceiling.value == 1000
    assert limits[1].dimension is UsageDimension.CONCURRENT_REQUESTS

    other = _scope(provider_id=_provider_id("other"))
    with pytest.raises(AdapterParseError, match="scope_id mismatch"):
        normalize_configured_limits(
            scope,
            [
                {
                    "scope_id": other.scope_id,
                    "dimension": "requests",
                    "ceiling": 1,
                    "window": {"kind": "fixed", "length_ms": 1000},
                }
            ],
        )


def test_normalize_configured_limits_rejects_negative_overflow_and_secrets():
    scope = _scope()
    with pytest.raises(AdapterParseError, match="non-negative|rejected"):
        normalize_configured_limits(
            scope,
            [
                {
                    "dimension": "requests",
                    "ceiling": -1,
                    "window": {"kind": "fixed", "length_ms": 1000},
                }
            ],
        )
    with pytest.raises(AdapterParseError, match="overflow|rejected"):
        normalize_configured_limits(
            scope,
            [
                {
                    "dimension": "requests",
                    "ceiling": 1 << 63,
                    "window": {"kind": "fixed", "length_ms": 1000},
                }
            ],
        )
    with pytest.raises(AdapterParseError, match="credential|forbidden"):
        normalize_configured_limits(
            scope,
            [
                {
                    "dimension": "requests",
                    "ceiling": 1,
                    "window": {"kind": "fixed", "length_ms": 1000},
                    "api_key": "sk-" + ("a" * 20),
                }
            ],
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible
# ---------------------------------------------------------------------------


def test_openai_compatible_parses_usage_body_and_rate_limit_headers():
    reset = FIXED_NOW + timedelta(seconds=30)
    observation = parse_openai_compatible_observation(
        _base_input(
            http_status=200,
            headers={
                "x-request-id": "req_abc123",
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-remaining-requests": "97",
                "x-ratelimit-reset-requests": "30s",
                "x-ratelimit-limit-tokens": "10000",
                "x-ratelimit-remaining-tokens": "8800",
                "x-ratelimit-reset-tokens": reset.isoformat().replace("+00:00", "Z"),
            },
            usage_body={
                "id": "chatcmpl-1",
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 40,
                    "total_tokens": 160,
                },
            },
            adapter_family="openai_compatible",
        )
    )
    assert isinstance(observation, ProviderUsageObservation)
    assert observation.scope_id == _scope().scope_id
    assert observation.request_id == "req-test-1"
    assert observation.provider_request_id in ("req_abc123", "chatcmpl-1")
    assert observation.http_status == 200
    usage = {entry.dimension: entry.amount.value for entry in observation.usage.entries}
    assert usage[UsageDimension.INPUT_TOKENS] == 120
    assert usage[UsageDimension.OUTPUT_TOKENS] == 40
    assert usage[UsageDimension.TOTAL_TOKENS] == 160
    dims = {limit.dimension for limit in observation.limits}
    assert UsageDimension.REQUESTS in dims
    assert UsageDimension.TOTAL_TOKENS in dims
    req_limit = next(
        limit
        for limit in observation.limits
        if limit.dimension is UsageDimension.REQUESTS
    )
    assert req_limit.ceiling.value == 100
    assert req_limit.remaining.value == 97
    assert req_limit.used.value == 3
    assert "raw_headers" not in observation.to_dict()
    assert "raw_body" not in observation.to_dict()
    assert observation.provenance.parser_version == ADAPTER_PARSER_VERSION
    # Round-trip through schema
    assert ProviderUsageObservation.from_dict(observation.to_dict()) == observation


def test_openai_compatible_429_retry_after_and_billing_exhaustion():
    obs_rate = parse_openai_compatible_observation(
        _base_input(
            http_status=429,
            headers={"retry-after": "12", "x-request-id": "r1"},
            error_body={
                "error": {
                    "type": "tokens",
                    "code": "rate_limit_exceeded",
                    "message": "Rate limit reached for requests",
                }
            },
        )
    )
    assert obs_rate.http_status == 429
    assert obs_rate.retry_after_ms == 12_000
    assert obs_rate.reset_at is not None
    assert any("subscription.usage_limit" in code for code in obs_rate.reason_codes)
    assert any(limit.remaining.value == 0 for limit in obs_rate.limits if limit.remaining.kind is QuantityKind.FINITE)

    obs_bill = parse_openai_compatible_observation(
        _base_input(
            http_status=429,
            error_body={
                "error": {
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                    "message": "You exceeded your current quota, please check your plan and billing details.",
                }
            },
        )
    )
    assert any("billing.exhausted" in code for code in obs_bill.reason_codes)
    assert obs_bill.retry_after_ms is not None


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_style_headers_and_usage_body():
    reset = FIXED_NOW + timedelta(seconds=45)
    observation = parse_anthropic_observation(
        _base_input(
            http_status=200,
            headers={
                "request-id": "req_ant_1",
                "anthropic-ratelimit-requests-limit": "50",
                "anthropic-ratelimit-requests-remaining": "49",
                "anthropic-ratelimit-requests-reset": reset.isoformat().replace(
                    "+00:00", "Z"
                ),
                "anthropic-ratelimit-input-tokens-limit": "40000",
                "anthropic-ratelimit-input-tokens-remaining": "39800",
                "anthropic-ratelimit-input-tokens-reset": reset.isoformat().replace(
                    "+00:00", "Z"
                ),
                "anthropic-ratelimit-output-tokens-limit": "8000",
                "anthropic-ratelimit-output-tokens-remaining": "7920",
                "anthropic-ratelimit-output-tokens-reset": reset.isoformat().replace(
                    "+00:00", "Z"
                ),
            },
            usage_body={
                "usage": {
                    "input_tokens": 200,
                    "output_tokens": 80,
                    "cache_read_input_tokens": 10,
                }
            },
        )
    )
    assert observation.provider_request_id == "req_ant_1"
    dims = {entry.dimension for entry in observation.usage.entries}
    assert UsageDimension.INPUT_TOKENS in dims
    assert UsageDimension.OUTPUT_TOKENS in dims
    limit_dims = {limit.dimension for limit in observation.limits}
    assert UsageDimension.REQUESTS in limit_dims
    assert UsageDimension.INPUT_TOKENS in limit_dims
    assert UsageDimension.OUTPUT_TOKENS in limit_dims


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------


def test_huggingface_429_and_estimated_time():
    observation = parse_huggingface_observation(
        _base_input(
            http_status=429,
            headers={
                "x-request-id": "hf-1",
                "x-ratelimit-limit": "300",
                "x-ratelimit-remaining": "0",
                "retry-after": "20",
            },
            error_body={
                "error": "Rate limit exceeded",
                "estimated_time": 20,
            },
            adapter_family="huggingface",
        )
    )
    assert observation.http_status == 429
    assert observation.retry_after_ms == 20_000
    assert observation.provider_request_id == "hf-1"
    assert any(
        limit.dimension is UsageDimension.REQUESTS and limit.remaining.value == 0
        for limit in observation.limits
        if limit.remaining.kind is QuantityKind.FINITE
    )


def test_huggingface_503_overloaded_retains_cooldown():
    observation = parse_huggingface_observation(
        _base_input(
            http_status=503,
            error_body={"error": "Model is currently overloaded", "estimated_time": 5},
        )
    )
    assert observation.http_status == 503
    assert observation.retry_after_ms is not None
    assert observation.retry_after_ms >= 5_000


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,kind",
    [
        ("codex", "usage_limit"),
        ("copilot", "usage_limit"),
        ("grok", "usage_limit"),
        ("gemini", "rate_limit"),
        ("goose", "capacity"),
        ("mistral", "usage_limit"),
    ],
)
def test_cli_structured_reset_metadata(provider, kind):
    scope = _scope(
        provider_id=_provider_id(provider),
        protocol=ProtocolKind.CLI,
        operation="text.generate",
    )
    observation = parse_cli_observation(
        {
            "scope": scope,
            "request_id": "cli-req-1",
            "observed_at": FIXED_NOW,
            "now": FIXED_NOW,
            "cli_metadata": {
                "provider": provider,
                "kind": kind,
                "resets_in_seconds": 3600,
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        }
    )
    assert observation.scope_id == scope.scope_id
    assert observation.retry_after_ms == 3_600_000
    assert any("subscription.usage_limit" in code for code in observation.reason_codes)
    assert any(code.startswith("cli.") for code in observation.reason_codes)
    usage = {entry.dimension for entry in observation.usage.entries}
    assert UsageDimension.INPUT_TOKENS in usage


def test_cli_billing_exhaustion_and_nested_events():
    scope = _scope(protocol=ProtocolKind.CLI)
    observation = parse_cli_observation(
        {
            "scope": scope,
            "request_id": "cli-bill-1",
            "observed_at": FIXED_NOW,
            "now": FIXED_NOW,
            "cli_metadata": {
                "provider": "codex",
                "kind": "quota_exceeded",
                "events": [
                    {"type": "error", "resets_in_seconds": 0},
                    {"type": "meta", "retry_after_seconds": 120},
                ],
            },
        }
    )
    assert any("billing.exhausted" in code for code in observation.reason_codes)
    assert observation.retry_after_ms is not None


# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------


def test_local_concurrency_and_memory_ceilings():
    scope = _scope(
        provider_id=_provider_id("local-runtime"),
        protocol=ProtocolKind.LOCAL,
        operation="text.generate",
    )
    observation = parse_local_observation(
        {
            "scope": scope,
            "request_id": "local-1",
            "observed_at": FIXED_NOW,
            "now": FIXED_NOW,
            "local_capacity": {
                "max_concurrent_requests": 4,
                "in_flight_requests": 2,
                "max_concurrent_streams": 2,
                "max_memory_bytes": 1_073_741_824,
                "memory_bytes_used": 536_870_912,
            },
        }
    )
    dims = {limit.dimension: limit for limit in observation.limits}
    assert dims[UsageDimension.CONCURRENT_REQUESTS].ceiling.value == 4
    assert dims[UsageDimension.CONCURRENT_REQUESTS].remaining.value == 2
    assert dims[UsageDimension.CONCURRENT_REQUESTS].window.kind is WindowKind.CONCURRENT
    assert dims[UsageDimension.CONCURRENT_STREAMS].ceiling.value == 2
    assert dims[UsageDimension.MEDIA_BYTES].ceiling.value == 1_073_741_824
    assert dims[UsageDimension.MEDIA_BYTES].remaining.value == 536_870_912


# ---------------------------------------------------------------------------
# Custom / unknown
# ---------------------------------------------------------------------------


def test_custom_adapter_with_field_map():
    clear_custom_adapters()
    register_custom_adapter(
        {
            "family": "custom",
            "adapter_id": "acme_obs",
            "description": "Acme",
        }
    )
    observation = parse_custom_observation(
        _base_input(
            http_status=200,
            headers={
                "x-acme-limit": "50",
                "x-acme-remaining": "10",
                "retry-after": "5",
            },
            custom_field_map={
                "x-acme-limit": "requests.limit",
                "x-acme-remaining": "requests.remaining",
            },
            adapter_family="custom",
        )
    )
    assert observation.retry_after_ms == 5_000
    req = next(
        (
            limit
            for limit in observation.limits
            if limit.dimension is UsageDimension.REQUESTS
        ),
        None,
    )
    assert req is not None
    assert req.ceiling.value == 50
    assert req.remaining.value == 10
    clear_custom_adapters()


def test_unknown_provider_still_parses_restrictive_signals():
    observation = parse_provider_observation(
        _base_input(
            http_status=429,
            headers={"retry-after": "9", "x-request-id": "u1"},
            adapter_family="totally-unknown-vendor",
        )
    )
    assert observation.http_status == 429
    assert observation.retry_after_ms == 9_000
    assert observation.provider_request_id == "u1"
    assert any(code.startswith("adapter.unknown") for code in observation.reason_codes)


# ---------------------------------------------------------------------------
# Safety: policy ceiling, cooldown retention, rejection
# ---------------------------------------------------------------------------


def test_never_raises_policy_ceiling_from_untrusted_observation():
    scope = _scope()
    configured = normalize_configured_limits(
        scope,
        [
            {
                "dimension": "requests",
                "ceiling": 100,
                "window": {"kind": "fixed", "length_ms": 60_000},
            }
        ],
        source=LimitSource.POLICY,
        observed_at=FIXED_NOW,
    )
    observation = parse_openai_compatible_observation(
        _base_input(
            scope=scope,
            headers={
                "x-ratelimit-limit-requests": "1000000",
                "x-ratelimit-remaining-requests": "999999",
            },
            policy_ceilings={"requests": 100},
            configured_limits=configured,
        )
    )
    req = next(
        limit
        for limit in observation.limits
        if limit.dimension is UsageDimension.REQUESTS
    )
    assert req.ceiling.value == 100
    assert any("policy.ceiling_clamped" in code for code in req.provenance.reason_codes)

    guarded = apply_policy_ceiling_guard(
        observation.limits, policy_ceilings={"requests": 50}
    )
    g_req = next(
        limit for limit in guarded if limit.dimension is UsageDimension.REQUESTS
    )
    assert g_req.ceiling.value == 50


def test_retains_restrictive_cooldown_when_unrelated_parsing_fails():
    # Malformed usage body should not erase Retry-After from headers/status.
    observation = parse_openai_compatible_observation(
        _base_input(
            http_status=429,
            headers={"retry-after": "15"},
            usage_body={"usage": {"prompt_tokens": "not-a-number", "extra": {"deep": 1}}},
        )
    )
    assert observation.retry_after_ms == 15_000
    assert observation.http_status == 429
    assert observation.reset_at is not None
    # Cooldown path also available as a direct helper.
    minimal = retain_restrictive_cooldown(
        scope=_scope(),
        request_id="req-cd",
        http_status=503,
        retry_after_ms=2_000,
        observed_at=FIXED_NOW,
    )
    assert minimal.retry_after_ms == 2_000
    assert minimal.http_status == 503
    assert any("cooldown.retained" in code for code in minimal.reason_codes)


def test_rejects_scope_mismatch_negative_stale_conflicting_and_credentials():
    scope = _scope()
    other = _scope(provider_id=_provider_id("other"))

    with pytest.raises(AdapterScopeError, match="scope_id"):
        parse_provider_observation(
            _base_input(scope=scope, claimed_scope_id=other.scope_id)
        )

    with pytest.raises(AdapterParseError, match="credential|forbidden"):
        parse_provider_observation(
            _base_input(
                headers={"authorization": "Bearer " + ("a" * 20)},
            )
        )

    with pytest.raises(AdapterParseError, match="credential|forbidden"):
        parse_provider_observation(
            _base_input(
                usage_body={"api_key": "sk-" + ("b" * 20), "prompt_tokens": 1},
            )
        )

    # Negative remaining is rejected at header parse (failure reason, not hard
    # crash) and must not produce a negative quantity.
    observation = parse_openai_compatible_observation(
        _base_input(
            headers={
                "x-ratelimit-limit-requests": "10",
                "x-ratelimit-remaining-requests": "-5",
            }
        )
    )
    for limit in observation.limits:
        if limit.remaining.kind is QuantityKind.FINITE:
            assert limit.remaining.value is not None and limit.remaining.value >= 0

    # Stale reset far in the past is dropped.
    stale = FIXED_NOW - timedelta(days=10)
    observation_stale = parse_openai_compatible_observation(
        _base_input(
            headers={
                "x-ratelimit-limit-requests": "10",
                "x-ratelimit-remaining-requests": "1",
                "x-ratelimit-reset-requests": stale.isoformat().replace("+00:00", "Z"),
            }
        )
    )
    assert any(
        "reset.stale" in code or "header.reset" in code
        for code in observation_stale.reason_codes
    )

    # Contradictory resets: prefer earlier, mark conflict.
    early = FIXED_NOW + timedelta(seconds=10)
    late = FIXED_NOW + timedelta(seconds=3600)
    observation_conflict = parse_openai_compatible_observation(
        _base_input(
            headers={
                "x-ratelimit-limit-requests": "10",
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-reset-requests": early.isoformat().replace("+00:00", "Z"),
                "x-ratelimit-limit-tokens": "100",
                "x-ratelimit-remaining-tokens": "0",
                "x-ratelimit-reset-tokens": late.isoformat().replace("+00:00", "Z"),
            }
        )
    )
    assert observation_conflict.reset_at is not None
    # More restrictive (earlier) wins when conflict exceeds tolerance.
    reset_dt = datetime.fromisoformat(
        observation_conflict.reset_at.replace("Z", "+00:00")
    )
    assert reset_dt <= early + timedelta(seconds=1)


def test_scope_reason_codes_distinguish_account_project_credential_model_operation():
    scope = _scope()
    observation = parse_openai_compatible_observation(
        _base_input(scope=scope, http_status=200, usage_body={"usage": {"total_tokens": 1}})
    )
    codes = set(observation.reason_codes)
    assert "scope.endpoint" in codes
    assert "scope.account" in codes
    assert "scope.project" in codes
    assert "scope.credential" in codes
    assert "scope.model" in codes
    assert "scope.operation" in codes


def test_unknown_fields_become_bounded_reason_codes_not_stored_payloads():
    observation = parse_openai_compatible_observation(
        _base_input(
            usage_body={
                "usage": {
                    "prompt_tokens": 1,
                    "mystery_counter": 99,
                    "totally_novel_field": "x",
                }
            },
            headers={"x-ratelimit-mystery": "1", "x-request-id": "id-1"},
        )
    )
    payload = observation.to_dict()
    # Raw values and header dumps are not retained; unknown names may appear only
    # as bounded reason codes (not as stored payload keys/values).
    assert "raw_headers" not in payload
    assert "raw_body" not in payload
    assert payload.get("usage", {}).get("entries")
    usage_blob = json.dumps(payload.get("usage"))
    # Unknown counter values must not be stored; only known prompt_tokens=1.
    assert '"value": 99' not in usage_blob
    assert any(code.startswith("usage.unknown_") for code in observation.reason_codes)
    assert any(code.startswith("header.unknown_") for code in observation.reason_codes)
    # Reason codes themselves are bounded tokens, not free-form dumps.
    for code in observation.reason_codes:
        assert len(code) <= 64
        assert code == code.casefold()


def test_observation_binds_to_exact_request_and_endpoint_scope():
    scope = _scope()
    observation = parse_provider_observation(
        ObservationInput(
            scope=scope,
            request_id="exact-req-42",
            http_status=200,
            usage_body={"usage": {"prompt_tokens": 3, "completion_tokens": 1}},
            observed_at=FIXED_NOW,
            now=FIXED_NOW,
            adapter_family=AdapterFamily.OPENAI_COMPATIBLE,
        )
    )
    assert observation.scope_id == scope.scope_id
    assert observation.request_id == "exact-req-42"
    assert observation.observation_id.startswith("uobs_")
    # Different request id => different observation identity.
    other = parse_provider_observation(
        ObservationInput(
            scope=scope,
            request_id="exact-req-43",
            http_status=200,
            usage_body={"usage": {"prompt_tokens": 3, "completion_tokens": 1}},
            observed_at=FIXED_NOW,
            now=FIXED_NOW,
            adapter_family=AdapterFamily.OPENAI_COMPATIBLE,
        )
    )
    assert other.observation_id != observation.observation_id


def test_string_count_depth_and_time_bounds_are_clamped():
    huge_headers = {("x-h-%d" % i): "1" for i in range(100)}
    with pytest.raises(AdapterParseError, match="headers exceeds"):
        parse_provider_observation(_base_input(headers=huge_headers))

    observation = parse_openai_compatible_observation(
        _base_input(
            headers={"retry-after": str(10**12)},  # absurd seconds -> clamped/rejected path
            http_status=429,
        )
    )
    # Either clamped to MAX_WINDOW_MS or defaulted; never unbounded.
    assert observation.retry_after_ms is not None
    assert observation.retry_after_ms <= 31_622_400_000


def test_malformed_and_adversarial_inputs_fail_closed():
    scope = _scope()
    with pytest.raises(AdapterParseError):
        parse_provider_observation({"request_id": "x"})  # missing scope
    with pytest.raises(AdapterParseError):
        parse_provider_observation(_base_input(http_status=99))
    with pytest.raises(AdapterParseError):
        parse_provider_observation(
            _base_input(usage_body=["not", "a", "mapping"])  # type: ignore[arg-type]
        )
    # Nested secret-shaped value rejected.
    with pytest.raises(AdapterParseError, match="credential"):
        parse_provider_observation(
            _base_input(
                error_body={"error": {"message": "ok", "token": "sk-" + ("z" * 24)}}
            )
        )
    # Empty request id rejected.
    with pytest.raises(AdapterParseError):
        parse_provider_observation(_base_input(request_id="   "))

    # Direct UsageLimit objects accepted for configured path.
    limit = UsageLimit(
        scope_id=scope.scope_id,
        dimension=UsageDimension.REQUESTS,
        ceiling=Quantity.finite(5),
        window=LimitWindow(kind=WindowKind.FIXED, length_ms=1000),
    )
    normalized = normalize_configured_limits(scope, [limit], observed_at=FIXED_NOW)
    assert normalized[0].limit_id == limit.limit_id


def test_confidence_reflects_sources_and_partial_failure():
    strong = parse_openai_compatible_observation(
        _base_input(
            headers={
                "x-ratelimit-limit-requests": "10",
                "x-ratelimit-remaining-requests": "9",
            },
            usage_body={"usage": {"total_tokens": 5}},
        )
    )
    assert strong.confidence in (
        ConfidenceLevel.HIGH,
        ConfidenceLevel.AUTHORITATIVE,
    )
    weak = parse_openai_compatible_observation(
        _base_input(
            http_status=429,
            headers={"retry-after": "1"},
            usage_body={"usage": {"prompt_tokens": "bad"}},
        )
    )
    assert weak.confidence in (ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM)
