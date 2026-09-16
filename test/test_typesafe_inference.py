from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.llm_allocation import (
    CallErrorKind,
    classify_provider_failure,
    path_for_provider,
)
from ipfs_accelerate_py.typesafe_inference import (
    Choice,
    Noul,
    Score,
    TypeSafeClient,
    TypeSafeInferenceError,
    get_last_typesafe_observation,
    serialize_questions,
    system_one,
)


class _FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status: int = 200,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._payload = json.dumps(payload).encode("utf-8")
        self.status = status
        self.headers = headers or {"X-Request-Id": "req-test"}

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _example_result() -> dict[str, Any]:
    return {
        "model": "jev-latest",
        "answers": {
            "billing": {"type": "noul", "noul": 0.91},
            "tone": {
                "type": "choice",
                "choice": "angry",
                "probabilities": {"calm": 0.1, "angry": 0.9},
                "confidence": 0.8,
            },
            "urgency": {
                "type": "score",
                "score": 1.6,
                "legend": {"0": "low", "1": "medium", "2": "high"},
                "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65},
                "confidence": 0.78,
            },
        },
        "usage": {"input_tokens": 312, "output_tokens": 48},
    }


def _example_questions() -> dict[str, Any]:
    return {
        "billing": Noul(instructions="Is this about billing?"),
        "tone": Choice(
            instructions="What is the tone?",
            criteria={"calm": None, "angry": None},
        ),
        "urgency": Score(
            instructions="How urgent is this?",
            criteria=["low", "medium", "high"],
        ),
    }


def test_serialize_questions_match_http_shape() -> None:
    payload = serialize_questions(_example_questions())
    assert payload["billing"] == {
        "type": "noul",
        "instructions": "Is this about billing?",
    }
    assert payload["tone"]["type"] == "choice"
    assert payload["tone"]["criteria"] == {"calm": None, "angry": None}
    assert payload["urgency"]["criteria"] == ["low", "medium", "high"]


def test_system_one_posts_state_and_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float = 0):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(_example_result())

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret-key")
    monkeypatch.setattr("ipfs_accelerate_py.typesafe_inference.urllib.request.urlopen", fake_urlopen)

    result = system_one("I was charged twice. Please help ASAP.", _example_questions())

    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert captured["method"] == "POST"
    assert captured["body"]["state"] == "I was charged twice. Please help ASAP."
    assert captured["body"]["model"] == "jev-latest"
    assert captured["body"]["questions"]["billing"]["type"] == "noul"
    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers["authorization"] == "Bearer ts-secret-key"
    assert result.nouls["billing"].noul == pytest.approx(0.91)
    assert result.choices["tone"].choice == "angry"
    assert result.scores["urgency"].score == pytest.approx(1.6)
    observation = get_last_typesafe_observation()
    assert observation["input_tokens"] == 312
    assert observation["output_tokens"] == 48
    dumped = json.dumps(result.to_dict())
    assert "ts-secret-key" not in dumped
    assert "ts-secret-key" not in json.dumps(observation)


def test_system_one_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("ipfs_accelerate_py_TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_PY_TYPESAFE_API_KEY", raising=False)
    with pytest.raises(TypeSafeInferenceError, match="TYPESAFE_API_KEY"):
        system_one("state", {"ok": {"type": "noul", "instructions": "yes?"}})


def test_http_error_redacts_secret_and_exposes_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float = 0):
        body = io.BytesIO(b'{"error":"invalid ts-secret-key"}')
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            hdrs=None,
            fp=body,
        )

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret-key")
    monkeypatch.setattr("ipfs_accelerate_py.typesafe_inference.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(TypeSafeInferenceError) as excinfo:
        system_one("state", {"ok": {"type": "noul", "instructions": "yes?"}})
    assert excinfo.value.status == 401
    assert "ts-secret-key" not in str(excinfo.value)
    assert "[redacted]" in str(excinfo.value)


def test_retries_overloaded_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_urlopen(request: urllib.request.Request, timeout: float = 0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                529,
                "Overloaded",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"overloaded"}'),
            )
        return _FakeResponse(_example_result())

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret-key")
    monkeypatch.setattr("ipfs_accelerate_py.typesafe_inference.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("ipfs_accelerate_py.typesafe_inference.time.sleep", lambda _delay: None)
    result = system_one("state", _example_questions())
    assert calls["n"] == 2
    assert result.model == "jev-latest"


def test_client_matches_sdk_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float = 0):
        return _FakeResponse(_example_result())

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret-key")
    monkeypatch.setattr("ipfs_accelerate_py.typesafe_inference.urllib.request.urlopen", fake_urlopen)
    with TypeSafeClient() as client:
        result = client.system_one("I was charged twice.", _example_questions())
    assert result.nouls["billing"].noul == pytest.approx(0.91)


def test_generate_text_typesafe_returns_json_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float = 0):
        return _FakeResponse(_example_result())

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret-key")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "0")
    monkeypatch.setattr("ipfs_accelerate_py.typesafe_inference.urllib.request.urlopen", fake_urlopen)
    llm_router.clear_llm_router_caches()
    text = llm_router.generate_text(
        "I was charged twice. Please help ASAP.",
        provider="typesafe",
        questions=_example_questions(),
        allow_local_fallback=False,
        allow_cross_provider_fallback=False,
    )
    payload = json.loads(text)
    assert payload["billing"]["noul"] == pytest.approx(0.91)
    assert payload["tone"]["choice"] == "angry"


def test_generate_text_typesafe_requires_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret-key")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "0")
    llm_router.clear_llm_router_caches()
    with pytest.raises(llm_router.LLMRouterError, match="questions"):
        llm_router.generate_text(
            "hello",
            provider="typesafe",
            allow_local_fallback=False,
            allow_cross_provider_fallback=False,
        )


def test_catalog_typesafe_authorized_without_leaking_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "configured-for-test")
    provider = llm_router.get_provider_descriptor("typesafe_ai")
    assert provider.name == "typesafe"
    assert provider.state.authorized is True
    assert "configured-for-test" not in json.dumps(provider.to_dict())
    models = llm_router.list_models("typesafe")
    assert dict(models[0].labels)["invocation_model"] == "jev-latest"


def test_typesafe_is_api_path_and_classifies_http_errors() -> None:
    assert path_for_provider("typesafe") == "api"
    auth = classify_provider_failure("typesafe", "TypeSafe HTTP 401: unauthorized", status_code=401)
    assert auth.kind is CallErrorKind.AUTHENTICATION
    invalid = classify_provider_failure("typesafe", "malformed question", status_code=422)
    assert invalid.kind is CallErrorKind.INVALID_REQUEST
    overloaded = classify_provider_failure("typesafe", "overloaded", status_code=529)
    assert overloaded.kind is CallErrorKind.SERVER
    assert overloaded.retryable is True


def test_api_backends_status_includes_typesafe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ipfs_accelerate_py.llm_allocation import AllocationStore, reset_allocation_store
    from ipfs_accelerate_py.llm_allocation.api_status import api_backends_status

    store = AllocationStore(tmp_path / "alloc.duckdb")
    reset_allocation_store(store)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret-key")
    report = api_backends_status(store=store, environ=__import__("os").environ)
    assert "typesafe" in report["backends"]
    assert report["backends"]["typesafe"]["authenticated"] is True
    assert "typesafe" in report["ready"]
    assert "ts-secret-key" not in json.dumps(report)
    reset_allocation_store(None)


def test_populate_typesafe_models(tmp_path: Path) -> None:
    from ipfs_accelerate_py.llm_allocation.model_manager_sync import populate_typesafe_models
    from ipfs_accelerate_py.model_manager import ModelManager

    manager = ModelManager(
        storage_path=str(tmp_path / "typesafe-models.json"),
        use_database=False,
        enable_ipfs=False,
        project_legacy_models=False,
    )
    result = populate_typesafe_models(manager, live=False)
    assert result["added"] >= 1
    sample = manager.get_model("typesafe:jev-latest")
    assert sample is not None
    assert sample.supported_backends == ["typesafe"]
