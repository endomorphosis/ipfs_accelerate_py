"""Contracts for the hosted Meta Model API integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


_META_ENV_NAMES = (
    "META_API_KEY",
    "MODEL_API_KEY",
    "META_AI_API_KEY",
    "ipfs_accelerate_py_META_AI_API_KEY",
    "ipfs_accelerate_py_META_AI_BASE_URL",
    "ipfs_accelerate_py_META_AI_MODEL",
    "IPFS_ACCELERATE_PY_DISABLE_SECRET_MANAGER",
)


def _clear_meta_environment(monkeypatch) -> None:
    for name in _META_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


class _FakeSecretsManager:
    def __init__(self, value: str | None):
        self.value = value
        self.requested: list[str] = []

    def get_credential(self, name: str):
        self.requested.append(name)
        return self.value if name == "meta_ai_api_key" else None


def test_current_endpoint_model_and_legacy_aliases():
    from ipfs_accelerate_py.common.meta_model_api import (
        META_MODEL_API_BASE_URL,
        META_MODEL_API_DEFAULT_MODEL,
        normalize_meta_model_name,
    )

    assert META_MODEL_API_BASE_URL == "https://api.meta.ai/v1"
    assert META_MODEL_API_DEFAULT_MODEL == "muse-spark-1.1"
    assert normalize_meta_model_name(None) == "muse-spark-1.1"
    assert normalize_meta_model_name("meta-spark/Spark-1.1") == "muse-spark-1.1"
    assert normalize_meta_model_name("custom-model") == "custom-model"


def test_credential_precedence_is_explicit_then_environment_then_encrypted_store(
    monkeypatch,
):
    from ipfs_accelerate_py.common.meta_model_api import resolve_meta_model_api_key

    _clear_meta_environment(monkeypatch)
    manager = _FakeSecretsManager("stored-value")

    monkeypatch.setenv("MODEL_API_KEY", "environment-value")
    assert resolve_meta_model_api_key("explicit-value", secrets_manager=manager) == "explicit-value"
    assert resolve_meta_model_api_key(secrets_manager=manager) == "environment-value"
    assert manager.requested == []

    monkeypatch.delenv("MODEL_API_KEY")
    assert resolve_meta_model_api_key(secrets_manager=manager) == "stored-value"
    assert manager.requested == ["meta_ai_api_key"]


def test_secret_manager_can_be_disabled_for_isolated_processes(monkeypatch):
    from ipfs_accelerate_py.common.meta_model_api import resolve_meta_model_api_key

    _clear_meta_environment(monkeypatch)
    monkeypatch.setenv("IPFS_ACCELERATE_PY_DISABLE_SECRET_MANAGER", "1")
    manager = _FakeSecretsManager("stored-value")

    assert resolve_meta_model_api_key(secrets_manager=manager) is None
    assert manager.requested == []


def test_cache_fingerprint_does_not_contain_secret(monkeypatch):
    import ipfs_accelerate_py.common.meta_model_api as meta_contract

    secret = "synthetic-meta-secret"
    monkeypatch.setattr(
        meta_contract,
        "resolve_meta_model_api_key",
        lambda: secret,
    )
    fingerprint = meta_contract.meta_model_api_key_fingerprint()

    assert fingerprint
    assert secret not in fingerprint
    assert len(fingerprint) == 16


def test_llm_router_uses_encrypted_secret_and_current_wire_contract(monkeypatch):
    import ipfs_accelerate_py.common.secrets_manager as secrets_module
    from ipfs_accelerate_py.llm_router import _get_meta_ai_provider

    _clear_meta_environment(monkeypatch)
    manager = _FakeSecretsManager("synthetic-meta-secret")
    monkeypatch.setattr(
        secrets_module,
        "get_global_secrets_manager",
        lambda: manager,
    )

    captured: dict = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {
                    "model": "muse-spark-1.1",
                    "choices": [{"message": {"content": "OK"}}],
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    provider = _get_meta_ai_provider()
    assert provider is not None
    assert (
        provider.generate(
            "Reply with OK",
            model_name="meta-spark/Spark-1.1",
            max_tokens=512,
            timeout=17,
        )
        == "OK"
    )
    assert captured["url"] == "https://api.meta.ai/v1/chat/completions"
    assert captured["authorization"] == "Bearer synthetic-meta-secret"
    assert captured["timeout"] == 17
    assert captured["payload"]["model"] == "muse-spark-1.1"
    assert captured["payload"]["max_completion_tokens"] == 512
    assert "max_tokens" not in captured["payload"]


def test_api_backend_uses_secret_manager_without_persisting_remote_copy(
    monkeypatch,
    tmp_path: Path,
):
    import ipfs_accelerate_py.common.meta_model_api as meta_contract
    from ipfs_accelerate_py.api_backends.meta_ai import meta_ai
    from ipfs_accelerate_py.common.secrets_manager import SecretsManager

    _clear_meta_environment(monkeypatch)
    manager = SecretsManager(secrets_file=str(tmp_path / "secrets.enc"))
    assert manager.storage is None
    manager.set_credential("meta_ai_api_key", "synthetic-meta-secret")

    monkeypatch.setattr(
        meta_contract,
        "resolve_meta_model_api_key",
        lambda *args, **kwargs: "synthetic-meta-secret",
    )
    client = meta_ai(metadata={"api_key": "synthetic-meta-secret"})

    assert client.api_key == "synthetic-meta-secret"
    assert client.base_url == "https://api.meta.ai/v1"
    assert client.default_model == "muse-spark-1.1"
    assert b"synthetic-meta-secret" not in (tmp_path / "secrets.enc").read_bytes()


def test_spark_catalog_tiers_pricing_and_limits():
    from ipfs_accelerate_py.common.meta_model_api import (
        META_MODEL_API_RECOMMENDED_MODEL,
        MetaPricingTier,
        meta_model_context_window,
        meta_model_pricing,
        meta_model_rate_limits,
        meta_model_tier,
        normalize_meta_model_name,
    )

    assert META_MODEL_API_RECOMMENDED_MODEL == "muse-spark-1.3"
    assert normalize_meta_model_name("meta/muse-spark-1.3") == "muse-spark-1.3"
    assert meta_model_tier("muse-spark-1.3") is MetaPricingTier.STANDARD
    assert meta_model_tier("muse-spark-1.2-contributor") is MetaPricingTier.CONTRIBUTOR
    assert meta_model_context_window("muse-spark-1.3") == 1_048_576
    assert meta_model_pricing("muse-spark-1.3") == {
        "cached_input": 0.15,
        "input": 1.25,
        "output": 4.25,
    }
    assert meta_model_pricing("muse-spark-1.3-contributor")["input"] == 0.10
    assert meta_model_rate_limits("muse-spark-1.1") == {"rpm": 3000, "tpm": 4_000_000}
    assert meta_model_rate_limits("muse-spark-1.2-contributor") == {
        "rpm": 100,
        "tpm": 3_000_000,
    }


def test_parse_usage_and_cost_includes_cached_and_reasoning_tokens():
    from ipfs_accelerate_py.common.meta_model_api import (
        estimate_meta_cost_usd,
        parse_meta_usage,
    )

    chat = parse_meta_usage(
        {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "total_tokens": 1200,
                "prompt_tokens_details": {"cached_tokens": 400},
                "completion_tokens_details": {"reasoning_tokens": 50},
            }
        }
    )
    assert chat.source == "chat_completions"
    assert chat.cached_tokens == 400
    assert chat.reasoning_tokens == 50
    cost = estimate_meta_cost_usd(chat, model_name="muse-spark-1.2")
    expected = (600 / 1_000_000) * 1.25 + (400 / 1_000_000) * 0.15 + (200 / 1_000_000) * 4.25
    assert abs(cost - expected) < 1e-9

    responses = parse_meta_usage(
        {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens_details": {"reasoning_tokens": 1},
            }
        }
    )
    assert responses.source == "responses"
    assert responses.prompt_tokens == 10
    assert responses.completion_tokens == 5


def test_classify_http_errors_and_retry_policy():
    from ipfs_accelerate_py.common.meta_model_api import (
        MetaModelApiErrorKind,
        classify_meta_http_error,
        format_meta_http_error,
        meta_error_is_retryable,
        meta_retry_delay_seconds,
    )

    rate = classify_meta_http_error(
        429,
        json.dumps(
            {
                "error": {
                    "message": "Rate limit exceeded. Please retry after 15 seconds.",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            }
        ),
        headers={"Retry-After": "15", "x-ratelimit-remaining-requests": "0"},
    )
    assert rate.kind is MetaModelApiErrorKind.RATE_LIMIT
    assert rate.retryable is True
    assert rate.retry_after_seconds == 15.0
    assert "Meta AI HTTP 429:" in format_meta_http_error(rate)

    billing = classify_meta_http_error(
        402,
        json.dumps(
            {
                "error": {
                    "message": "Billing account issue.",
                    "type": "billing_error",
                    "code": "billing_not_configured",
                }
            }
        ),
    )
    assert billing.kind is MetaModelApiErrorKind.BILLING
    assert billing.retryable is False

    auth = classify_meta_http_error(
        401,
        json.dumps(
            {
                "error": {
                    "message": "Unauthorized",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }
            }
        ),
    )
    assert auth.kind is MetaModelApiErrorKind.AUTHENTICATION
    assert auth.retryable is False

    timeout = classify_meta_http_error(
        504,
        json.dumps(
            {
                "error": {
                    "message": "use streaming",
                    "type": "server_error",
                    "code": "gateway_timeout",
                }
            }
        ),
    )
    assert timeout.kind is MetaModelApiErrorKind.GATEWAY_TIMEOUT
    assert timeout.retryable is False

    overloaded = classify_meta_http_error(
        503,
        json.dumps(
            {
                "error": {
                    "message": "overloaded",
                    "type": "server_error",
                    "code": "service_overloaded",
                }
            }
        ),
        headers={"Retry-After": "60"},
    )
    assert overloaded.kind is MetaModelApiErrorKind.SERVICE_UNAVAILABLE
    assert overloaded.retryable is True
    assert overloaded.retry_after_seconds == 60.0
    assert meta_error_is_retryable(overloaded.kind)
    assert meta_retry_delay_seconds(0, retry_after=60.0) == 60.0

    context = classify_meta_http_error(
        400,
        json.dumps(
            {
                "error": {
                    "message": (
                        "You passed 1200064 input tokens and requested 1 output "
                        "tokens. However, the model's context length is only 1048576 tokens"
                    ),
                    "type": "invalid_request_error",
                    "code": None,
                }
            }
        ),
    )
    assert context.kind is MetaModelApiErrorKind.CONTEXT_WINDOW
    assert context.retryable is False


def test_llm_router_records_usage_and_retries_429(monkeypatch):
    from ipfs_accelerate_py.common.meta_model_api import (
        MetaModelApiErrorKind,
        clear_last_meta_observation,
        get_last_meta_observation,
    )
    from ipfs_accelerate_py.llm_router import LLMRouterError, _get_meta_ai_provider

    _clear_meta_environment(monkeypatch)
    monkeypatch.setenv("MODEL_API_KEY", "synthetic-meta-secret")
    monkeypatch.setattr(
        "ipfs_accelerate_py.llm_router.sleep_for_meta_retry", lambda *_a, **_k: None
    )
    clear_last_meta_observation()

    calls = {"n": 0}

    class _Headers(dict):
        def items(self):
            return super().items()

    class _Response:
        def __init__(self, payload, headers=None):
            self._payload = payload
            self.headers = _Headers(headers or {})

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    class _HTTPError(Exception):
        def __init__(self):
            self.code = 429
            self.reason = "Too Many Requests"
            self.fp = True
            self.headers = _Headers({"Retry-After": "1"})

        def read(self):
            return json.dumps(
                {
                    "error": {
                        "message": "Rate limit exceeded. Please retry after 15 seconds.",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        _ = (request, timeout)
        calls["n"] += 1
        if calls["n"] == 1:
            raise _HTTPError()
        return _Response(
            {
                "model": "muse-spark-1.2",
                "choices": [{"message": {"content": "OK"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "prompt_tokens_details": {"cached_tokens": 2},
                },
            },
            headers={
                "x-ratelimit-limit-requests": "3000",
                "x-ratelimit-remaining-requests": "2999",
                "x-ratelimit-limit-tokens": "4000000",
                "x-ratelimit-remaining-tokens": "3990000",
            },
        )

    import urllib.error
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(urllib.error, "HTTPError", _HTTPError)

    provider = _get_meta_ai_provider()
    assert provider is not None
    assert provider.generate("ping", model_name="muse-spark-1.2") == "OK"
    assert calls["n"] == 2
    observation = get_last_meta_observation()
    assert observation["model"] == "muse-spark-1.2"
    assert observation["tier"] == "standard"
    assert observation["usage"]["prompt_tokens"] == 10
    assert observation["usage"]["cached_tokens"] == 2
    assert observation["rate_limit"]["remaining_requests"] == 2999
    assert observation["estimated_cost_usd"] >= 0

    def always_429(request, timeout):
        _ = (request, timeout)
        raise _HTTPError()

    monkeypatch.setattr(urllib.request, "urlopen", always_429)
    clear_last_meta_observation()
    with pytest.raises(LLMRouterError, match="Meta AI HTTP 429") as caught:
        provider.generate("ping", model_name="muse-spark-1.2")
    assert caught.value.retryable is True
    assert caught.value.meta_error_kind is MetaModelApiErrorKind.RATE_LIMIT


def test_llm_router_does_not_retry_billing_or_auth(monkeypatch):
    import urllib.error
    import urllib.request

    from ipfs_accelerate_py.common.meta_model_api import MetaModelApiErrorKind
    from ipfs_accelerate_py.llm_router import LLMRouterError, _get_meta_ai_provider

    _clear_meta_environment(monkeypatch)
    monkeypatch.setenv("MODEL_API_KEY", "synthetic-meta-secret")
    sleeps: list[float] = []
    monkeypatch.setattr(
        "ipfs_accelerate_py.llm_router.sleep_for_meta_retry",
        lambda *_a, **_k: sleeps.append(1),
    )

    class _Headers(dict):
        pass

    class _HTTPError(Exception):
        def __init__(self, code, payload):
            self.code = code
            self.reason = "error"
            self.fp = True
            self.headers = _Headers()
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    calls = {"n": 0}

    def fake_urlopen(request, timeout):
        _ = (request, timeout)
        calls["n"] += 1
        raise _HTTPError(
            402,
            {
                "error": {
                    "message": "Billing account issue.",
                    "type": "billing_error",
                    "code": "billing_not_configured",
                }
            },
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(urllib.error, "HTTPError", _HTTPError)

    provider = _get_meta_ai_provider()
    with pytest.raises(LLMRouterError, match="Meta AI HTTP 402") as caught:
        provider.generate("ping")
    assert calls["n"] == 1
    assert sleeps == []
    assert caught.value.retryable is False
    assert caught.value.meta_error_kind is MetaModelApiErrorKind.BILLING

