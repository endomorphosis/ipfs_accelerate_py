"""Router integration tests for the Goose CLI provider (GOOSE-005).

All execution uses fakes/mocks — no live Goose binary, network, or installer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

import ipfs_accelerate_py.llm_router as llm_router
from ipfs_accelerate_py.cli_runtime.installers.goose import GooseInstallResult
from ipfs_accelerate_py.cli_runtime.providers.goose import GooseProviderError, GooseErrorKind


_GOOSE_ENV_NAMES = (
    "IPFS_ACCELERATE_GOOSE_DISCOVERY",
    "IPFS_ACCELERATE_PY_GOOSE_DISCOVERY",
    "ipfs_accelerate_py_GOOSE_DISCOVERY",
    "IPFS_ACCELERATE_GOOSE_AUTO_INSTALL",
    "IPFS_ACCELERATE_PY_GOOSE_AUTO_INSTALL",
    "IPFS_ACCELERATE_GOOSE_PATH",
    "IPFS_ACCELERATE_PY_GOOSE_PATH",
    "ipfs_accelerate_py_GOOSE_BIN",
    "IPFS_ACCELERATE_PY_GOOSE_BIN",
    "IPFS_ACCELERATE_AGENT_GOOSE_BIN",
    "GOOSE_BIN",
    "GOOSE_MODEL",
    "GOOSE_PROVIDER",
    "ipfs_accelerate_py_GOOSE_CLI_MODEL",
    "IPFS_ACCELERATE_PY_GOOSE_CLI_MODEL",
    "ipfs_accelerate_py_GOOSE_PROVIDER",
    "IPFS_ACCELERATE_PY_GOOSE_PROVIDER",
    "IPFS_ACCELERATE_GOOSE_PROVIDER",
    "ipfs_accelerate_py_LLM_PROVIDER",
    "IPFS_ACCELERATE_PY_LLM_PROVIDER",
    "IPFS_DATASETS_PY_LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_HOST",
    "OPENAI_BASE_PATH",
)


def _clear_goose_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _GOOSE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _install_result(
    *,
    available: bool = True,
    executable: str = "/fake/goose",
    installed: bool = False,
    version: str = "1.44.0",
    reason: str = "already_installed",
) -> GooseInstallResult:
    return GooseInstallResult(
        available=available,
        installed=installed,
        executable=executable if available else "",
        version=version if available else "",
        method="path" if available else "not_found",
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Explicit vs implicit resolution / installation policy
# ---------------------------------------------------------------------------


def test_explicit_goose_provider_may_invoke_ensure_goose(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_goose_env(monkeypatch)
    ensure_calls: list[dict[str, Any]] = []

    def fake_ensure(**kwargs: Any) -> GooseInstallResult:
        ensure_calls.append(dict(kwargs))
        return _install_result(installed=True)

    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.goose.ensure_goose",
        fake_ensure,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.goose.discover_goose",
        lambda **_k: pytest.fail("explicit path must use ensure_goose, not discover_goose alone"),
    )

    provider = llm_router._builtin_provider_by_name("goose_cli", auto_install=True)

    assert provider is not None
    assert ensure_calls == [{"auto_install": True}]


def test_explicit_goose_alias_and_forced_env_count_as_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_goose_env(monkeypatch)
    ensure_calls: list[Any] = []

    def fake_ensure(**kwargs: Any) -> GooseInstallResult:
        ensure_calls.append(kwargs)
        return _install_result()

    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.goose.ensure_goose",
        fake_ensure,
    )
    # Preferred alias
    p1 = llm_router._resolve_provider_uncached(
        "goose", deps=llm_router.get_default_router_deps()
    )
    assert p1 is not None
    assert len(ensure_calls) == 1

    # Forced via env also counts as explicit
    monkeypatch.setenv("ipfs_accelerate_py_LLM_PROVIDER", "goose_cli")
    ensure_calls.clear()
    p2 = llm_router._resolve_provider_uncached(
        None, deps=llm_router.get_default_router_deps()
    )
    assert p2 is not None
    assert len(ensure_calls) == 1


def test_implicit_discovery_is_detect_only_and_never_installs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_goose_env(monkeypatch)
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.goose.ensure_goose",
        lambda **_k: pytest.fail("implicit discovery must not install"),
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.goose.discover_goose",
        lambda **_k: _install_result(available=False, reason="not_installed"),
    )
    monkeypatch.setattr(llm_router, "find_goose_cli", lambda: None)

    assert llm_router._builtin_provider_by_name("goose_cli") is None
    assert llm_router._builtin_provider_by_name("goose", auto_install=False) is None


def test_implicit_discovery_requires_opt_in_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_goose_env(monkeypatch)
    # Binary present, but discovery flag off → not considered during auto order.
    monkeypatch.setattr(llm_router, "find_goose_cli", lambda: "/fake/goose")
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.goose.discover_goose",
        lambda **_k: _install_result(),
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.goose.ensure_goose",
        lambda **_k: pytest.fail("auto-order must not ensure_goose"),
    )

    # Without discovery flag, unpinned iteration skips goose entirely.
    names = [name for name, _ in llm_router._iter_unpinned_optional_providers()]
    assert "goose_cli" not in names

    # With opt-in discovery, goose may appear (detect-only).
    monkeypatch.setenv("IPFS_ACCELERATE_GOOSE_DISCOVERY", "1")
    names_on = [name for name, _ in llm_router._iter_unpinned_optional_providers()]
    assert "goose_cli" in names_on


def test_detect_only_provider_returns_when_binary_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_goose_env(monkeypatch)
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.goose.discover_goose",
        lambda **_k: _install_result(executable='/opt/goose', version='1.44.0'),
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.goose.ensure_goose",
        lambda **_k: pytest.fail("detect-only must not call ensure_goose"),
    )

    provider = llm_router._get_goose_cli_provider(auto_install=False)
    assert provider is not None


# ---------------------------------------------------------------------------
# Model / underlying-provider mapping
# ---------------------------------------------------------------------------


def test_model_name_and_goose_provider_map_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_goose_env(monkeypatch)
    captured: dict[str, Any] = {}

    class FakeAdapter:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: Any) -> str:
            captured["prompt"] = prompt
            captured["model_name"] = model_name
            captured["kwargs"] = dict(kwargs)
            return "chat-ok"

    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.goose.ensure_goose",
        lambda **_k: _install_result(),
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.providers.goose.create_goose_provider",
        lambda **_k: FakeAdapter(),
    )

    provider = llm_router._get_goose_cli_provider(auto_install=True)
    assert provider is not None
    result = provider.generate(
        "hello",
        model_name="muse-spark-1.1",
        goose_provider="openai",
    )
    assert result == "chat-ok"
    assert isinstance(result, str)
    assert captured["model_name"]  # normalized model reaches adapter
    assert "spark" in str(captured["model_name"]).lower() or "muse" in str(
        captured["model_name"]
    ).lower()
    assert captured["kwargs"].get("goose_provider") == "openai"
    # model_name must not be collapsed into goose_provider
    assert captured["kwargs"].get("goose_provider") != captured["model_name"]


def test_chat_generate_text_returns_ordinary_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_goose_env(monkeypatch)

    class FakeBackend:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: Any) -> str:
            assert not kwargs.get("agent")
            assert not kwargs.get("side_effecting")
            return f"echo:{prompt}"

    monkeypatch.setattr(
        llm_router,
        "get_llm_provider",
        lambda *_a, **_k: FakeBackend(),
    )
    # Avoid optional cross-provider noise.
    text = llm_router.generate_text(
        "ping",
        provider="goose_cli",
        provider_instance=FakeBackend(),
        allow_local_fallback=False,
    )
    assert text == "echo:ping"
    assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Provider cache identity
# ---------------------------------------------------------------------------


def test_provider_cache_key_includes_goose_settings_without_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_goose_env(monkeypatch)
    monkeypatch.setattr(llm_router, "find_goose_cli", lambda: "/fake/goose")
    monkeypatch.setenv("GOOSE_MODEL", "muse-spark-1.1")
    monkeypatch.setenv("GOOSE_PROVIDER", "openai")
    monkeypatch.setenv("IPFS_ACCELERATE_GOOSE_DISCOVERY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value-must-not-appear")
    monkeypatch.setenv("OPENAI_HOST", "https://api.meta.ai")

    key = llm_router._provider_cache_key()
    serialized = repr(key)

    assert "muse-spark-1.1" in serialized
    assert "openai" in serialized
    assert "https://api.meta.ai" in serialized
    # Presence flag for credentials, never the secret value.
    assert "sk-secret-value-must-not-appear" not in serialized
    assert True in key  # OPENAI_API_KEY present marker and/or goose binary present

    # Changing a behavior-affecting Goose setting must change the key.
    monkeypatch.setenv("GOOSE_MODEL", "other-model")
    key2 = llm_router._provider_cache_key()
    assert key != key2


# ---------------------------------------------------------------------------
# Side-effect-aware cache / retry / fallback / batch policy
# ---------------------------------------------------------------------------


def test_agent_request_bypasses_response_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_goose_env(monkeypatch)
    monkeypatch.setattr(llm_router, "_response_cache_enabled", lambda: True)
    calls: list[str] = []

    class AgentBackend:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: Any) -> str:
            calls.append(prompt)
            assert kwargs.get("agent") or kwargs.get("side_effecting")
            return f"agent:{prompt}:{len(calls)}"

    deps = llm_router.RouterDeps()
    # Pre-seed a cache entry that must not be served for agent requests.
    cache_key = llm_router._response_cache_key(
        provider="goose_cli",
        model_name="m",
        prompt="work",
        kwargs={"agent": True, "workspace": "/tmp/ws", "path_root": "/tmp"},
    )
    deps.set_cached(cache_key, "CACHED_SHOULD_NOT_RETURN")

    first = llm_router.generate_text(
        "work",
        model_name="m",
        provider="goose_cli",
        provider_instance=AgentBackend(),
        deps=deps,
        allow_local_fallback=False,
        agent=True,
        side_effecting=True,
        workspace="/tmp/ws",
        path_root="/tmp",
    )
    second = llm_router.generate_text(
        "work",
        model_name="m",
        provider="goose_cli",
        provider_instance=AgentBackend(),
        deps=deps,
        allow_local_fallback=False,
        agent=True,
        side_effecting=True,
        workspace="/tmp/ws",
        path_root="/tmp",
    )
    assert first == "agent:work:1"
    assert second == "agent:work:2"
    assert "CACHED_SHOULD_NOT_RETURN" not in (first, second)
    assert len(calls) == 2


def test_agent_request_disables_default_model_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_goose_env(monkeypatch)
    attempts: list[Optional[str]] = []

    class FailOnce:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: Any) -> str:
            attempts.append(model_name)
            raise llm_router.LLMRouterError("agent failed before tools")

    with pytest.raises(llm_router.LLMRouterError, match="agent failed"):
        llm_router.generate_text(
            "do stuff",
            model_name="muse-spark-1.1",
            provider="goose_cli",
            provider_instance=FailOnce(),
            allow_local_fallback=False,
            agent=True,
            side_effecting=True,
            workspace="/tmp/ws",
            path_root="/tmp",
        )
    # Only the requested model — no retry with model_name=None.
    assert attempts == ["muse-spark-1.1"]


def test_agent_request_disables_automatic_provider_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_goose_env(monkeypatch)
    fallback_hits: list[str] = []

    class PrimaryFail:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: Any) -> str:
            raise llm_router.LLMRouterError("goose agent boom")

    def fake_iter() -> list[tuple[str, Any]]:
        class Other:
            def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: Any) -> str:
                fallback_hits.append("other")
                return "fallback-text"

        return [("mock", Other())]

    monkeypatch.setattr(llm_router, "_iter_unpinned_optional_providers", fake_iter)

    with pytest.raises(llm_router.LLMRouterError, match="goose agent boom"):
        llm_router.generate_text(
            "task",
            provider="goose_cli",
            provider_instance=PrimaryFail(),
            allow_local_fallback=True,
            agent=True,
            side_effecting=True,
            workspace="/tmp/ws",
            path_root="/tmp",
        )
    assert fallback_hits == []


def test_no_retry_after_side_effects_started(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_goose_env(monkeypatch)
    attempts = {"n": 0}

    class PartialSideEffect:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: Any) -> str:
            attempts["n"] += 1
            err = llm_router.LLMRouterError("tool already ran")
            setattr(err, "side_effects_started", True)
            raise err

    with pytest.raises(llm_router.LLMRouterError, match="tool already ran"):
        llm_router._generate_with_provider_fallbacks(
            "goose_cli",
            PartialSideEffect(),
            "prompt",
            model_name="m1",
            kwargs={"agent": True, "side_effecting": True},
        )
    assert attempts["n"] == 1


def test_agent_batch_forces_serial_workers() -> None:
    assert (
        llm_router._batch_worker_count(
            size=8,
            max_workers=8,
            provider="goose_cli",
            side_effecting=True,
        )
        == 1
    )
    # Chat (non-agent) may use limited concurrency.
    chat_workers = llm_router._batch_worker_count(
        size=8,
        max_workers=None,
        provider="goose_cli",
        side_effecting=False,
    )
    assert 1 <= chat_workers <= 2


def test_agent_generate_text_batch_is_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_goose_env(monkeypatch)
    concurrent_peaks: list[int] = []
    in_flight = {"n": 0}

    class CountingBackend:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: Any) -> str:
            in_flight["n"] += 1
            concurrent_peaks.append(in_flight["n"])
            try:
                return f"ok:{prompt}"
            finally:
                in_flight["n"] -= 1

    out = llm_router.generate_text_batch(
        ["a", "b", "c", "d"],
        provider="goose_cli",
        provider_instance=CountingBackend(),
        allow_local_fallback=False,
        max_workers=4,
        agent=True,
        side_effecting=True,
        workspace="/tmp/ws",
        path_root="/tmp",
    )
    assert out == ["ok:a", "ok:b", "ok:c", "ok:d"]
    assert max(concurrent_peaks) == 1


def test_goose_provider_error_maps_to_llm_router_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_goose_env(monkeypatch)

    class BoomAdapter:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: Any) -> str:
            raise GooseProviderError(
                "policy denied",
                kind=GooseErrorKind.POLICY_DENIAL,
                side_effects_started=False,
            )

    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.goose.ensure_goose",
        lambda **_k: _install_result(),
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.providers.goose.create_goose_provider",
        lambda **_k: BoomAdapter(),
    )
    provider = llm_router._get_goose_cli_provider(auto_install=True)
    assert provider is not None
    with pytest.raises(llm_router.LLMRouterError, match="policy denied"):
        provider.generate("x", agent=True, workspace="/tmp", path_root="/tmp")


def test_aliases_canonicalize_to_goose_cli() -> None:
    for alias in ("goose", "goose-cli", "block_goose", "block-goose", "aaif_goose", "goose_cli"):
        assert llm_router._canonicalize_provider(alias) == "goose_cli"
        assert llm_router._is_goose_provider_name(alias)


def test_find_goose_cli_prefers_configured_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_goose_env(monkeypatch)
    binary = tmp_path / "goose"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("IPFS_ACCELERATE_GOOSE_PATH", str(binary))
    assert llm_router.find_goose_cli() == str(binary)


def test_kwargs_are_side_effecting_helpers() -> None:
    assert llm_router._kwargs_are_side_effecting({"agent": True})
    assert llm_router._kwargs_are_side_effecting({"side_effecting": True})
    assert llm_router._kwargs_are_side_effecting({"with_tools": True})
    assert llm_router._kwargs_are_side_effecting({"agent_policy": MagicMock()})
    assert not llm_router._kwargs_are_side_effecting({})
    assert not llm_router._kwargs_are_side_effecting({"temperature": 0.2})


# ---------------------------------------------------------------------------
# GOOSE-011 security matrix anchors (router surface)
# ---------------------------------------------------------------------------


def test_matrix_no_install_on_implicit_and_no_retry_after_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Implicit discovery never installs; side-effect failures are non-retryable."""
    _clear_goose_env(monkeypatch)
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.goose.ensure_goose",
        lambda **_k: pytest.fail("implicit path must not install"),
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.goose.discover_goose",
        lambda **_k: _install_result(available=False, reason="not_installed"),
    )
    monkeypatch.setattr(llm_router, "find_goose_cli", lambda: None)
    assert llm_router._builtin_provider_by_name("goose_cli") is None

    from ipfs_accelerate_py.cli_runtime.providers.goose import (
        GooseErrorKind,
        GooseProviderError,
    )

    err = GooseProviderError(
        "partial tool activity",
        kind=GooseErrorKind.NONZERO_EXIT,
        side_effects_started=True,
        retryable=False,
    )
    assert err.side_effects_started is True
    assert err.retryable is False
    assert llm_router._kwargs_are_side_effecting(
        {"agent": True, "allow_side_effects": True}
    )



# ---------------------------------------------------------------------------
# GOOSE-012 operator documentation contracts and opt-in live smoke
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_doc(*parts: str) -> str:
    path = _repo_root().joinpath(*parts)
    assert path.is_file(), f"missing documentation file: {path}"
    return path.read_text(encoding="utf-8")


def test_goose_operator_docs_cover_router_guidance() -> None:
    """LLM_ROUTER.md documents Goose selection, env, install, and safety planes."""
    doc = _read_doc("docs", "LLM_ROUTER.md")
    # Provider table and aliases
    assert "`goose_cli`" in doc
    assert "goose" in doc.lower()
    # Canonical env vars and discovery/install policy (no secret values)
    for marker in (
        "IPFS_ACCELERATE_GOOSE_DISCOVERY",
        "IPFS_ACCELERATE_GOOSE_AUTO_INSTALL",
        "IPFS_ACCELERATE_GOOSE_PATH",
        "GOOSE_PROVIDER",
        "GOOSE_MODEL",
        "GOOSE_PATH_ROOT",
        "IPFS_ACCELERATE_GOOSE_MANAGED_ROOT",
        "IPFS_ACCELERATE_GOOSE_LIVE",
    ):
        assert marker in doc, f"missing env documentation: {marker}"
    assert "does not install" in doc.lower() or "detect-only" in doc.lower()
    assert "opt-in" in doc.lower() or "opt in" in doc.lower()
    # Chat vs agent separation
    assert "chat-only" in doc.lower() or "chat only" in doc.lower() or "GOOSE_MODE=chat" in doc
    assert "GooseAgentPolicy" in doc or "allow_side_effects" in doc
    assert "enable_agent" in doc
    # Managed install / pin / checksum
    assert "managed" in doc.lower()
    assert "PINNED_GOOSE_VERSION" in doc or "pinned" in doc.lower()
    assert "sha256" in doc.lower() or "checksum" in doc.lower()
    assert "goose_release_manifest" in doc
    # Shared vs isolated roots
    assert "GOOSE_PATH_ROOT" in doc
    assert "isolat" in doc.lower()
    # Health / cancel / recovery
    assert "readiness" in doc.lower()
    assert "liveness" in doc.lower()
    assert "cancel" in doc.lower()
    # P2P gates and no-replay
    assert "IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI" in doc
    assert "IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_AGENT" in doc
    assert "replay" in doc.lower() or "no-replay" in doc.lower() or "duplicate" in doc.lower()
    # Offline default + live smoke + rollback
    assert "offline" in doc.lower()
    assert "IPFS_ACCELERATE_GOOSE_LIVE" in doc
    assert "rollback" in doc.lower()
    assert "troubleshoot" in doc.lower()
    # Never document a literal secret value pattern as if it were real
    assert "sk-proj-" not in doc
    assert "BEGIN PRIVATE KEY" not in doc


def test_goose_operator_docs_indexes_and_quickstart() -> None:
    """Indexes and quickstart link Goose rollout without dropping other providers."""
    index = _read_doc("docs", "INDEX.md")
    assert "LLM_ROUTER" in index or "LLM router" in index
    assert "Goose" in index or "goose" in index

    quick = _read_doc("docs", "guides", "QUICKSTART.md")
    assert "goose_cli" in quick
    assert "IPFS_ACCELERATE_GOOSE_DISCOVERY" in quick
    assert "LLM_ROUTER" in quick or "llm_router" in quick.lower()

    readme = _read_doc("README.md")
    assert "Goose" in readme or "goose_cli" in readme
    assert "IPFS_ACCELERATE_GOOSE_LIVE" in readme
    assert "LLM_ROUTER" in readme or "LLM router" in readme
    # Preserve existing provider guidance anchors
    for marker in ("Codex", "Copilot", "OpenRouter", "llama.cpp"):
        assert marker.lower() in readme.lower() or marker in readme


def test_default_goose_suite_is_offline_by_design(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default Goose tests must not require live install or network."""
    import os

    monkeypatch.delenv("IPFS_ACCELERATE_GOOSE_LIVE", raising=False)
    assert not os.environ.get("IPFS_ACCELERATE_GOOSE_LIVE")
    # Implicit discovery remains off without opt-in.
    _clear_goose_env(monkeypatch)
    assert llm_router._goose_discovery_enabled() is False


@pytest.mark.skipif(
    not __import__("os").environ.get("IPFS_ACCELERATE_GOOSE_LIVE"),
    reason=(
        "live smoke requires IPFS_ACCELERATE_GOOSE_LIVE=1 and a configured "
        "Goose binary plus backend provider credentials"
    ),
)
def test_opt_in_live_goose_smoke() -> None:
    """Optional live chat through goose_cli when explicitly gated.

    Configure binary (PATH or IPFS_ACCELERATE_GOOSE_PATH) and backend auth
    (for example OPENAI_API_KEY / GOOSE_PROVIDER) in the environment. Never
    commit secrets.
    """
    import os

    binary = llm_router.find_goose_cli()
    if not binary:
        pytest.skip("goose binary not found; set IPFS_ACCELERATE_GOOSE_PATH")

    # Auth marker presence (value not inspected beyond emptiness).
    auth_names = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "OLLAMA_HOST",
        "DATABRICKS_TOKEN",
    )
    if not any(str(os.environ.get(n) or "").strip() for n in auth_names):
        # Package secret manager may still supply Meta credentials.
        try:
            from ipfs_accelerate_py.common.secrets_manager import (
                get_global_secrets_manager,
            )

            sm = get_global_secrets_manager()
            meta = sm.get_credential("meta_ai_api_key") if sm else None
            if not meta:
                pytest.skip("no Goose backend credential configured for live smoke")
        except Exception:
            pytest.skip("no Goose backend credential configured for live smoke")

    text = llm_router.generate_text(
        "Reply with exactly: goose-live-ok",
        provider="goose_cli",
        max_tokens=32,
        temperature=0.0,
    )
    assert isinstance(text, str)
    assert text.strip()
    # Live path must remain chat (no agent) and must not echo env secrets.
    for name in auth_names:
        secret = os.environ.get(name)
        if secret:
            assert secret not in text
