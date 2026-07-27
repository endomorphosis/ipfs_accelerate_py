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


def _semantic_canonical_response_format() -> dict[str, object]:
    qualifiers = ["after_deadline", "if_complete"]
    qualifier_schema = {
        "type": "array",
        "maxItems": 8,
        "items": {
            "type": "string",
            "enum": qualifiers,
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "semantic_roundtrip_canonical_ir_v1",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["rules"],
                "properties": {
                    "rules": {
                        "type": "array",
                        "maxItems": 16,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "modality",
                                "actor",
                                "action",
                                "object",
                                "conditions",
                                "exceptions",
                                "temporal",
                            ],
                            "properties": {
                                "modality": {
                                    "type": "string",
                                    "enum": ["O", "P", "F"],
                                },
                                "actor": {
                                    "type": "string",
                                    "enum": [
                                        "company_a",
                                        "company_b",
                                        "agency",
                                    ],
                                },
                                "action": {
                                    "type": "string",
                                    "enum": ["file", "review"],
                                },
                                "object": {
                                    "type": "string",
                                    "enum": ["", "notice", "record"],
                                },
                                "conditions": qualifier_schema,
                                "exceptions": json.loads(
                                    json.dumps(qualifier_schema)
                                ),
                                "temporal": json.loads(
                                    json.dumps(qualifier_schema)
                                ),
                            },
                        },
                    }
                },
            },
        },
    }


def _semantic_realization_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "semantic_roundtrip_realization_v1",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
        },
    }


def _semantic_generation_options(
    response_format: dict[str, object],
    *,
    max_tokens: int,
) -> dict[str, object]:
    return {
        "response_format": response_format,
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "stop": ["<|im_end|>"],
        "timeout": 120.0,
        "cache_prompt": False,
    }


@pytest.mark.parametrize(
    ("response_format", "max_tokens", "raw_completion"),
    [
        (
            _semantic_canonical_response_format(),
            3072,
            '{"rules":[]}',
        ),
        (
            _semantic_realization_response_format(),
            1536,
            '{"text":"The agency shall file the notice."}',
        ),
    ],
)
def test_pinned_symai_dispatches_exact_semantic_roundtrip_contracts(
    monkeypatch: pytest.MonkeyPatch,
    response_format: dict[str, object],
    max_tokens: int,
    raw_completion: str,
) -> None:
    calls: list[urllib.request.Request] = []
    model = llm_router._PINNED_SYMAI_LEANSTRAL_MODEL

    def urlopen(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> _Response:
        assert 0 < timeout <= 120.0
        calls.append(request)
        if request.full_url.endswith("/models"):
            return _Response({"data": [{"id": model}]})
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["response_format"] == response_format
        assert payload["max_tokens"] == max_tokens
        assert payload["temperature"] == 0
        assert payload["seed"] == 0
        assert payload["stop"] == ["<|im_end|>"]
        assert payload["cache_prompt"] is False
        return _Response(
            {
                "model": model,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": raw_completion},
                    }
                ],
            }
        )

    for name in (
        "ipfs_accelerate_py_ENABLE_IPFS_ACCELERATE",
        "IPFS_ACCELERATE_PY_ENABLE_IPFS_ACCELERATE",
        "IPFS_DATASETS_PY_ENABLE_IPFS_ACCELERATE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm_router, "_pinned_symai_urlopen", urlopen)
    deps = llm_router.RouterDeps(
        accelerate_managers={"llm_router": object()}
    )
    provider = llm_router._get_accelerate_provider(deps)
    assert provider is not None

    text = llm_router.generate_text(
        "bounded semantic round trip",
        model_name=llm_router._PINNED_SYMAI_LEANSTRAL_ALIAS,
        provider="ipfs_accelerate_py",
        provider_instance=provider,
        deps=deps,
        allow_local_fallback=False,
        disable_model_retry=True,
        _symai_route_binding=dict(
            llm_router._PINNED_SYMAI_ROUTE_BINDING
        ),
        **_semantic_generation_options(
            response_format,
            max_tokens=max_tokens,
        ),
    )

    assert text == raw_completion
    assert [request.full_url for request in calls] == [
        "http://127.0.0.1:8080/v1/models",
        "http://127.0.0.1:8080/v1/chat/completions",
    ]
    trace = llm_router.get_last_generation_trace()
    assert trace["resolved_provider_name"] == "leanstral_local"
    assert trace["resolved_model_name"] == model
    assert trace["service_endpoint"] == "http://127.0.0.1:8080/v1"
    assert trace["routing_backend"] == "llama.cpp"


def test_pinned_symai_request_validator_is_side_effect_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_urlopen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("contract validation must not perform HTTP")

    monkeypatch.setattr(
        llm_router, "_pinned_symai_urlopen", forbidden_urlopen
    )
    response_format = _semantic_canonical_response_format()

    normalized = llm_router.validate_pinned_symai_request_contract(
        model_name=llm_router._PINNED_SYMAI_LEANSTRAL_ALIAS,
        route_binding=dict(llm_router._PINNED_SYMAI_ROUTE_BINDING),
        generation_options=_semantic_generation_options(
            response_format,
            max_tokens=3072,
        ),
    )

    assert normalized["response_format"] == response_format
    assert normalized["max_tokens"] == 3072
    assert normalized["timeout"] == 120.0
    assert normalized["stop"] == ["<|im_end|>"]
    normalized_schema = normalized["response_format"]["json_schema"][
        "schema"
    ]
    actor_enum = normalized_schema["properties"]["rules"]["items"][
        "properties"
    ]["actor"]["enum"]
    assert actor_enum == ["company_a", "company_b", "agency"]


def test_pinned_symai_request_validator_rejects_backend_drift() -> None:
    route_binding = dict(llm_router._PINNED_SYMAI_ROUTE_BINDING)
    route_binding["routing_backend"] = "existing_leanstral_service"

    with pytest.raises(RuntimeError, match="incomplete or drifted"):
        llm_router.validate_pinned_symai_request_contract(
            model_name=llm_router._PINNED_SYMAI_LEANSTRAL_ALIAS,
            route_binding=route_binding,
            generation_options=_semantic_generation_options(
                _semantic_realization_response_format(),
                max_tokens=1536,
            ),
        )


def test_pinned_symai_rejects_semantic_setting_drift_before_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def forbidden_urlopen(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("drifted settings must not make an HTTP request")

    monkeypatch.setattr(
        llm_router, "_pinned_symai_urlopen", forbidden_urlopen
    )
    options = _semantic_generation_options(
        _semantic_canonical_response_format(),
        max_tokens=3071,
    )

    with pytest.raises(RuntimeError, match="token setting drifted"):
        llm_router._generate_pinned_symai_leanstral(
            "prompt",
            kwargs=options,
        )

    assert calls == []


@pytest.mark.parametrize("unsafe_change", ["unbounded_rules", "large_enum"])
def test_pinned_symai_rejects_unsafe_semantic_schemas_before_http(
    monkeypatch: pytest.MonkeyPatch,
    unsafe_change: str,
) -> None:
    calls: list[object] = []

    def forbidden_urlopen(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("unsafe schema must not make an HTTP request")

    monkeypatch.setattr(
        llm_router, "_pinned_symai_urlopen", forbidden_urlopen
    )
    response_format = _semantic_canonical_response_format()
    schema = response_format["json_schema"]["schema"]
    rule_array = schema["properties"]["rules"]
    if unsafe_change == "unbounded_rules":
        del rule_array["maxItems"]
    else:
        actor = rule_array["items"]["properties"]["actor"]
        actor["enum"] = [f"actor_{index:03d}" for index in range(257)]

    with pytest.raises(RuntimeError, match="schema|vocabulary"):
        llm_router._generate_pinned_symai_leanstral(
            "prompt",
            kwargs=_semantic_generation_options(
                response_format,
                max_tokens=3072,
            ),
        )

    assert calls == []


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
