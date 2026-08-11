"""Exact-provider routing regressions for supervisor LLM children."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.todo_daemon import llm as todo_llm


class _FailingProvider:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def generate(
        self,
        prompt: str,
        *,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        self._calls.append("primary")
        raise llm_router.LLMRouterError("primary provider failed")


class _AlternateProvider:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def generate(
        self,
        prompt: str,
        *,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        self._calls.append("alternate")
        return "alternate-result"


class _DefaultModelFallbackProvider:
    def __init__(self, models: list[str | None]) -> None:
        self._models = models

    def generate(
        self,
        prompt: str,
        *,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        self._models.append(model_name)
        if model_name is not None:
            raise llm_router.LLMRouterError("requested model failed")
        return "default-model-result"


def _install_provider_fixtures(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
) -> None:
    alternate = _AlternateProvider(calls)
    monkeypatch.setattr(llm_router, "_response_cache_enabled", lambda: False)
    monkeypatch.setattr(
        llm_router,
        "_iter_unpinned_optional_providers",
        lambda: [("grok_cli", alternate)],
    )
    monkeypatch.setattr(
        llm_router,
        "_get_accelerate_provider",
        lambda _deps: (_ for _ in ()).throw(
            AssertionError("accelerate fallback must not be resolved")
        ),
    )
    monkeypatch.setattr(
        llm_router,
        "_get_local_hf_provider",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("local fallback must not be resolved")
        ),
    )


def test_explicit_exact_provider_disables_cross_provider_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _install_provider_fixtures(monkeypatch, calls)

    with pytest.raises(llm_router.LLMRouterError, match="primary provider failed"):
        llm_router.generate_text(
            "audit this leaf",
            provider="codex_cli",
            provider_instance=_FailingProvider(calls),
            allow_local_fallback=False,
            allow_cross_provider_fallback=False,
        )

    assert calls == ["primary"]


def test_ordinary_default_route_retains_cross_provider_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    alternate = _AlternateProvider(calls)
    monkeypatch.setattr(llm_router, "_response_cache_enabled", lambda: False)
    monkeypatch.setattr(
        llm_router,
        "_iter_unpinned_optional_providers",
        lambda: [("codex_cli", alternate)],
    )

    result = llm_router.generate_text(
        "ordinary prompt",
        provider_instance=_FailingProvider(calls),
    )

    assert result == "alternate-result"
    assert calls == ["primary", "alternate"]


def test_remote_fallback_can_be_enabled_without_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    alternate = _AlternateProvider(calls)
    monkeypatch.setattr(llm_router, "_response_cache_enabled", lambda: False)
    monkeypatch.setattr(
        llm_router,
        "_iter_unpinned_optional_providers",
        lambda: [("grok_cli", alternate)],
    )

    result = llm_router.generate_text(
        "remote-only failover",
        provider="codex_cli",
        provider_instance=_FailingProvider(calls),
        allow_local_fallback=False,
        allow_cross_provider_fallback=True,
    )

    assert result == "alternate-result"
    assert calls == ["primary", "alternate"]


def test_remote_fallback_remains_default_when_local_fallback_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    alternate = _AlternateProvider(calls)
    monkeypatch.setattr(llm_router, "_response_cache_enabled", lambda: False)
    monkeypatch.setattr(
        llm_router,
        "_iter_unpinned_optional_providers",
        lambda: [("grok_cli", alternate)],
    )

    result = llm_router.generate_text(
        "ordinary remote failover",
        provider="codex_cli",
        provider_instance=_FailingProvider(calls),
        allow_local_fallback=False,
    )

    assert result == "alternate-result"
    assert calls == ["primary", "alternate"]


def test_exact_provider_does_not_consume_cross_provider_cached_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    deps = llm_router.RouterDeps()
    alternate = _AlternateProvider(calls)
    monkeypatch.setattr(llm_router, "_response_cache_enabled", lambda: True)
    monkeypatch.setattr(
        llm_router,
        "_iter_unpinned_optional_providers",
        lambda: [("grok_cli", alternate)],
    )

    assert (
        llm_router.generate_text(
            "same bound request",
            provider="codex_cli",
            provider_instance=_FailingProvider(calls),
            deps=deps,
            allow_local_fallback=False,
            allow_cross_provider_fallback=True,
        )
        == "alternate-result"
    )
    with pytest.raises(llm_router.LLMRouterError, match="primary provider failed"):
        llm_router.generate_text(
            "same bound request",
            provider="codex_cli",
            provider_instance=_FailingProvider(calls),
            deps=deps,
            allow_local_fallback=False,
            allow_cross_provider_fallback=False,
        )

    assert calls == ["primary", "alternate", "primary"]


def test_exact_provider_never_retries_or_caches_under_a_fallback_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models: list[str | None] = []
    deps = llm_router.RouterDeps()
    monkeypatch.setattr(llm_router, "_response_cache_enabled", lambda: True)
    provider = _DefaultModelFallbackProvider(models)

    for _attempt in range(2):
        with pytest.raises(llm_router.LLMRouterError, match="requested model failed"):
            llm_router.generate_text(
                "same exact model request",
                provider="codex_cli",
                provider_instance=provider,
                model_name="gpt-5.6-sol",
                deps=deps,
                allow_local_fallback=False,
                allow_cross_provider_fallback=False,
            )

    assert models == ["gpt-5.6-sol", "gpt-5.6-sol"]


def test_supervisor_child_passes_fail_closed_cross_provider_permission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_popen(command: list[str], **kwargs: Any) -> Any:
        captured["env"] = dict(kwargs.get("env") or {})
        captured["child_source"] = Path(command[1]).read_text(encoding="utf-8")

        class _Process:
            returncode = 0
            pid = 9191

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                return ("ok", "")

            def poll(self) -> int:
                return self.returncode

        return _Process()

    monkeypatch.setattr(todo_llm.subprocess, "Popen", fake_popen)
    config = todo_llm.LlmRouterInvocation(
        repo_root=tmp_path,
        provider="codex_cli",
        model_name="fixture-model",
        allow_local_fallback=False,
        allow_cross_provider_fallback=False,
        timeout_seconds=1,
        timeout_grace_seconds=0,
        python_executable=sys.executable,
        reject_effective_provider_name=None,
    )

    assert todo_llm.call_llm_router("exact review", config) == "ok"
    assert captured["env"]["TODO_DAEMON_LLM_ALLOW_LOCAL_FALLBACK"] == "0"
    assert captured["env"]["TODO_DAEMON_LLM_ALLOW_CROSS_PROVIDER_FALLBACK"] == "0"
    assert "allow_cross_provider_fallback=" in captured["child_source"]
    envelope = todo_llm.build_child_request_envelope(config)
    assert envelope.allow_cross_provider_fallback is False
