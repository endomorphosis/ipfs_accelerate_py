"""Regression coverage for canonical pinned SyMAI router contracts."""

from __future__ import annotations

import json
import urllib.request

import pytest

from ipfs_accelerate_py import llm_router


class _Response:
    def __init__(self, value: object) -> None:
        self.body = json.dumps(value).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


def test_pinned_symai_accepts_exact_source_withheld_realization_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[urllib.request.Request] = []
    model = llm_router._PINNED_SYMAI_LEANSTRAL_MODEL
    raw_realization = '{"text":"The agency shall file the notice."}'

    def urlopen(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> _Response:
        assert timeout > 0
        calls.append(request)
        if request.full_url.endswith("/models"):
            return _Response({"data": [{"id": model}]})
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["response_format"] == (
            llm_router._PINNED_SYMAI_REALIZATION_RESPONSE_FORMAT
        )
        return _Response(
            {
                "model": model,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": raw_realization},
                    }
                ],
            }
        )

    monkeypatch.setattr(llm_router, "_pinned_symai_urlopen", urlopen)

    text, trace = llm_router._generate_pinned_symai_leanstral(
        "realize this canonical IR",
        kwargs={
            "response_format": (
                llm_router._PINNED_SYMAI_REALIZATION_RESPONSE_FORMAT
            ),
            "temperature": 0.0,
            "max_tokens": 128,
        },
    )

    assert text == raw_realization
    assert trace == llm_router._PINNED_SYMAI_ROUTE_BINDING
    assert [request.full_url for request in calls] == [
        "http://127.0.0.1:8080/v1/models",
        "http://127.0.0.1:8080/v1/chat/completions",
    ]


def test_pinned_symai_still_rejects_arbitrary_json_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def forbidden_urlopen(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("rejected schema must not make an HTTP request")

    monkeypatch.setattr(
        llm_router, "_pinned_symai_urlopen", forbidden_urlopen
    )
    arbitrary = {
        "type": "json_schema",
        "json_schema": {
            "name": "arbitrary",
            "strict": True,
            "schema": {"type": "object"},
        },
    }

    with pytest.raises(RuntimeError, match="supported frozen"):
        llm_router._generate_pinned_symai_leanstral(
            "prompt",
            kwargs={
                "response_format": arbitrary,
                "temperature": 0.0,
                "max_tokens": 128,
            },
        )

    assert calls == []


def test_pinned_symai_completion_error_is_public_and_allow_listed() -> None:
    error = llm_router.PinnedSymaiCompletionError(
        llm_router.PinnedSymaiCompletionError.OUTPUT_TOKEN_LIMIT
    )

    assert isinstance(error, llm_router.LLMRouterError)
    assert error.safe_failure_class == "output_token_limit"
    with pytest.raises(ValueError, match="unsupported"):
        llm_router.PinnedSymaiCompletionError("raw-provider-detail")
