"""Authored backend regressions for exact-model HF compatibility failures."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.endpoint_usage.routing import ErrorSafetyClass


@pytest.fixture
def authored_hf_backend(monkeypatch):
    calls = []
    fallback_discovery = []
    original_error = llm_router.LLMRouterError("authored model unsupported")

    class Backend:
        def generate(self, prompt, *, model_name=None, **kwargs):
            calls.append((prompt, model_name))
            if model_name == "authored-compatible-model":
                return "authored result"
            raise original_error

    def fallback_models(*, kwargs):
        fallback_discovery.append(dict(kwargs))
        return ["authored-compatible-model"]

    # Keep allocation/session accounting and model discovery outside this
    # routing test. No transport, credential lookup, or durable store is used.
    monkeypatch.setattr(llm_router, "_bind_cli_session_kwargs", lambda _provider, kwargs: kwargs)
    monkeypatch.setattr(llm_router, "_record_llm_allocation_observation", lambda **_kwargs: None)
    monkeypatch.setattr(llm_router, "_capture_cli_session", lambda *_args: None)
    monkeypatch.setattr(llm_router, "_classify_llm_invoke_exception", lambda _exc: ErrorSafetyClass.UNKNOWN)
    monkeypatch.setattr(llm_router, "_hf_llm_fallback_models", fallback_models)
    monkeypatch.setattr(llm_router, "_coalesce_env", lambda *_args: None)
    return Backend(), calls, fallback_discovery, original_error


def test_exact_model_failure_does_not_retry_hf_compatibility_models(authored_hf_backend):
    backend, calls, fallback_discovery, original_error = authored_hf_backend
    with pytest.raises(llm_router.LLMRouterError) as caught:
        llm_router._generate_with_provider_fallbacks(
            "hf_inference_api", backend, "authored prompt",
            model_name="authored-requested-model",
            kwargs={"disable_model_retry": True},
        )
    assert caught.value is original_error
    assert calls == [("authored prompt", "authored-requested-model")]
    assert fallback_discovery == []


def test_explicit_legacy_model_retry_keeps_hf_compatibility_fallback(authored_hf_backend):
    backend, calls, fallback_discovery, _original_error = authored_hf_backend
    output = llm_router._generate_with_provider_fallbacks(
        "hf_inference_api", backend, "authored prompt",
        model_name="authored-requested-model",
        kwargs={"disable_model_retry": False},
    )
    assert output == "authored result"
    assert calls == [
        ("authored prompt", "authored-requested-model"),
        ("authored prompt", None),
        ("authored prompt", "authored-compatible-model"),
    ]
    assert len(fallback_discovery) == 1
