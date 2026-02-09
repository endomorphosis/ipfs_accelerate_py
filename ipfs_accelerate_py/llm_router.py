"""LLM router for ipfs_accelerate_py.

This module provides a reusable top-level entrypoint for text generation
that integrates with existing ipfs_accelerate_py infrastructure.

Design goals:
- Avoid import-time side effects.
- Allow optional hooks/providers (backend manager, remote endpoints).
- Provide a local HuggingFace transformers fallback when available.
- Reuse existing CLI/SDK wrappers (no duplication).
- Support endpoint multiplexing via InferenceBackendManager.

Environment variables:
- `IPFS_ACCELERATE_PY_LLM_PROVIDER`: force provider name (registered provider)
- `IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER`: allow backend manager routing
- `IPFS_ACCELERATE_PY_LLM_MODEL`: default HF model name for local fallback

Additional optional providers (opt-in by selecting provider):
- `openrouter`: OpenRouter chat completions
    - `OPENROUTER_API_KEY` or `IPFS_ACCELERATE_PY_OPENROUTER_API_KEY`
    - `IPFS_ACCELERATE_PY_OPENROUTER_MODEL` (default model)
    - `IPFS_ACCELERATE_PY_OPENROUTER_BASE_URL` (default: https://openrouter.ai/api/v1)
- `codex_cli`: OpenAI Codex CLI via existing wrapper
    - `IPFS_ACCELERATE_PY_CODEX_CLI_MODEL` / `IPFS_ACCELERATE_PY_CODEX_MODEL`
- `copilot_cli`: GitHub Copilot CLI via existing wrapper
- `copilot_sdk`: GitHub Copilot SDK via existing wrapper
    - `IPFS_ACCELERATE_PY_COPILOT_SDK_MODEL`, `IPFS_ACCELERATE_PY_COPILOT_SDK_TIMEOUT`
- `gemini_cli`: Gemini CLI via existing wrapper
- `claude_code`: Claude Code CLI via existing wrapper
- `backend_manager`: Use InferenceBackendManager for distributed/multiplexed inference
"""

from __future__ import annotations

import json
import os
import hashlib
import importlib
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Optional, Protocol, runtime_checkable

from .router_deps import RouterDeps, get_default_router_deps

logger = logging.getLogger(__name__)


def _resolve_transformers_module(*, deps: Optional[RouterDeps] = None, module_override: object | None = None) -> object | None:
    """Resolve the transformers module with optional RouterDeps injection/caching."""

    if module_override is not None:
        if deps is not None:
            deps.set_cached("pip::transformers", module_override)
        return module_override

    if deps is not None:
        cached = deps.get_cached("pip::transformers")
        if cached is not None:
            return cached

    try:
        module = importlib.import_module("transformers")
    except Exception:
        return None

    if deps is not None:
        deps.set_cached("pip::transformers", module)
    return module


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _cache_enabled() -> bool:
    return os.environ.get("IPFS_ACCELERATE_PY_ROUTER_CACHE", "1").strip() != "0"


def _response_cache_enabled() -> bool:
    # Default to enabled for determinism + speed
    value = os.environ.get("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE")
    if value is None:
        return True
    return str(value).strip() != "0"


def _response_cache_key_strategy() -> str:
    """Return the response-cache key strategy.

    - "sha256" (default): compact deterministic string key
    - "cid": content-addressed CID (sha2-256, CIDv1) for the request payload
    """

    return os.environ.get("IPFS_ACCELERATE_PY_ROUTER_CACHE_KEY", "sha256").strip().lower() or "sha256"


def _response_cache_cid_base() -> str:
    return os.environ.get("IPFS_ACCELERATE_PY_ROUTER_CACHE_CID_BASE", "base32").strip() or "base32"


def _stable_kwargs_digest(kwargs: Dict[str, object]) -> str:
    if not kwargs:
        return ""
    try:
        payload = json.dumps(kwargs, sort_keys=True, default=repr, ensure_ascii=False)
    except Exception:
        payload = repr(sorted(kwargs.items(), key=lambda x: str(x[0])))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _effective_model_key(*, provider_key: str, model_name: Optional[str], kwargs: Dict[str, object]) -> str:
    """Best-effort model identifier for caching.

    Callers are inconsistent about whether they pass the model via ``model_name``
    or via kwargs (e.g. ``model=...``). Some providers also use env defaults.
    Cache keys should include the effective model to avoid cross-model collisions.
    """

    direct = (model_name or "").strip()
    if direct:
        return direct

    for key in ("model", "model_name", "model_id"):
        try:
            value = kwargs.get(key)
        except Exception:
            value = None
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text

    pk = (provider_key or "auto").strip().lower()
    if pk == "openrouter":
        return (
            os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_MODEL")
            or os.getenv("IPFS_ACCELERATE_PY_LLM_MODEL")
            or "openai/gpt-4o-mini"
        ).strip()
    if pk in {"codex", "codex_cli"}:
        return (
            _coalesce_env("IPFS_ACCELERATE_PY_CODEX_CLI_MODEL", "IPFS_ACCELERATE_PY_CODEX_MODEL")
            or "gpt-4-turbo"
        ).strip()
    if pk == "copilot_sdk":
        return (os.environ.get("IPFS_ACCELERATE_PY_COPILOT_SDK_MODEL", "") or "gpt-4o").strip()
    if pk in {"hf", "huggingface", "local_hf"}:
        return (os.getenv("IPFS_ACCELERATE_PY_LLM_MODEL", "gpt2") or "gpt2").strip()

    # Provider unknown/auto: include the most common default.
    return (os.getenv("IPFS_ACCELERATE_PY_LLM_MODEL", "") or "").strip()


def _response_cache_key(*, provider: Optional[str], model_name: Optional[str], prompt: str, kwargs: Dict[str, object]) -> str:
    provider_key = (provider or "auto").strip().lower()
    model_key = _effective_model_key(provider_key=provider_key, model_name=model_name, kwargs=kwargs)

    strategy = _response_cache_key_strategy()
    if strategy == "cid":
        # Try to use CID-based caching if available
        try:
            from .ipfs_multiformats import cid_for_obj
            payload = {
                "type": "llm_response",
                "provider": provider_key,
                "model": model_key,
                "prompt": prompt or "",
                "kwargs": kwargs or {},
            }
            cid = cid_for_obj(payload, base=_response_cache_cid_base())
            return f"llm_response_cid::{cid}"
        except Exception:
            pass  # Fall back to sha256

    prompt_digest = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()[:16]
    kw_digest = _stable_kwargs_digest(kwargs)
    return f"llm_response::{provider_key}::{model_key}::{prompt_digest}::{kw_digest}"


@runtime_checkable
class LLMProvider(Protocol):
    def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str: ...


ProviderFactory = Callable[[], LLMProvider]


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    factory: ProviderFactory


_PROVIDER_REGISTRY: Dict[str, ProviderInfo] = {}


def register_llm_provider(name: str, factory: ProviderFactory) -> None:
    if not name or not name.strip():
        raise ValueError("Provider name must be non-empty")
    _PROVIDER_REGISTRY[name] = ProviderInfo(name=name, factory=factory)


def _coalesce_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _get_openrouter_provider() -> Optional[LLMProvider]:
    api_key = _coalesce_env("IPFS_ACCELERATE_PY_OPENROUTER_API_KEY", "OPENROUTER_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")

    class _OpenRouterProvider:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            model = (
                model_name
                or os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_MODEL")
                or os.getenv("IPFS_ACCELERATE_PY_LLM_MODEL")
                or "openai/gpt-4o-mini"
            )

            max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256))
            temperature = kwargs.get("temperature", 0.2)

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
            }

            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    **({"HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER")} if os.getenv("OPENROUTER_HTTP_REFERER") else {}),
                    **({"X-Title": os.getenv("OPENROUTER_APP_TITLE")} if os.getenv("OPENROUTER_APP_TITLE") else {}),
                },
            )

            try:
                with urllib.request.urlopen(req, timeout=float(kwargs.get("timeout", 120))) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail or exc.reason}") from exc
            except Exception as exc:
                raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

            try:
                data = json.loads(raw)
            except Exception as exc:
                raise RuntimeError("OpenRouter returned invalid JSON") from exc

            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"].strip()
                delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    return delta["content"].strip()
                text = choices[0].get("text") if isinstance(choices[0], dict) else None
                if isinstance(text, str):
                    return text.strip()
            raise RuntimeError("OpenRouter response missing choices")

    return _OpenRouterProvider()


def _get_codex_cli_provider() -> Optional[LLMProvider]:
    """Get Codex CLI provider using existing CLI integration wrapper."""
    try:
        from ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration import OpenAICodexCLI
    except Exception:
        return None

    class _CodexCLIProvider:
        def __init__(self):
            self._client = None

        def _get_client(self):
            if self._client is None:
                self._client = OpenAICodexCLI()
            return self._client

        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            client = self._get_client()
            model = model_name or _coalesce_env("IPFS_ACCELERATE_PY_CODEX_CLI_MODEL", "IPFS_ACCELERATE_PY_CODEX_MODEL")
            
            result = client.generate_code(
                prompt=prompt,
                model=model if model else None,
                max_tokens=int(kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256))),
                temperature=float(kwargs.get("temperature", 0.2)),
                use_cache=kwargs.get("use_cache", True)
            )
            
            if result.get("success"):
                return result.get("code", "").strip()
            raise RuntimeError(result.get("error", "Codex CLI failed"))

    return _CodexCLIProvider()


def _get_copilot_cli_provider() -> Optional[LLMProvider]:
    """Get Copilot CLI provider using existing copilot_cli wrapper."""
    try:
        from ipfs_accelerate_py.copilot_cli.wrapper import CopilotCLI
    except Exception:
        return None

    class _CopilotCLIProvider:
        def __init__(self):
            self._client = None

        def _get_client(self):
            if self._client is None:
                self._client = CopilotCLI()
            return self._client

        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            client = self._get_client()
            # Use suggest_command for general text generation
            result = client.suggest_command(
                prompt=prompt,
                use_cache=kwargs.get("use_cache", True)
            )
            
            if result.get("success"):
                return result.get("suggestion", "").strip()
            raise RuntimeError(result.get("error", "Copilot CLI failed"))

    return _CopilotCLIProvider()


def _get_copilot_sdk_provider() -> Optional[LLMProvider]:
    """Get Copilot SDK provider using existing copilot_sdk wrapper."""
    try:
        from ipfs_accelerate_py.copilot_sdk.wrapper import CopilotSDK
    except Exception:
        return None

    class _CopilotSDKProvider:
        def __init__(self):
            self._client = None

        def _get_client(self):
            if self._client is None:
                model = os.environ.get("IPFS_ACCELERATE_PY_COPILOT_SDK_MODEL", "gpt-4o")
                self._client = CopilotSDK(model=model)
            return self._client

        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            client = self._get_client()
            
            # Create a session
            session = client.create_session(model=model_name)
            
            try:
                # Send message and get response
                result = client.send_message(
                    session=session,
                    prompt=prompt,
                    use_cache=kwargs.get("use_cache", True)
                )
                
                if result.get("success") and result.get("messages"):
                    # Extract text from messages
                    messages = result["messages"]
                    text_parts = [
                        msg.get("content", "")
                        for msg in messages
                        if msg.get("type") == "message"
                    ]
                    return "\n".join(text_parts).strip()
                
                raise RuntimeError("Copilot SDK did not return valid response")
            finally:
                # Clean up session
                client.destroy_session(session)

    return _CopilotSDKProvider()


def _get_gemini_cli_provider() -> Optional[LLMProvider]:
    """Get Gemini CLI provider using existing CLI integration wrapper."""
    try:
        from ipfs_accelerate_py.cli_integrations.gemini_cli_integration import GeminiCLI
    except Exception:
        return None

    class _GeminiCLIProvider:
        def __init__(self):
            self._client = None

        def _get_client(self):
            if self._client is None:
                self._client = GeminiCLI()
            return self._client

        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            client = self._get_client()
            
            result = client.generate_text(
                prompt=prompt,
                model=model_name,
                max_tokens=int(kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256))),
                temperature=float(kwargs.get("temperature", 0.2)),
                use_cache=kwargs.get("use_cache", True)
            )
            
            if result.get("success"):
                return result.get("text", "").strip()
            raise RuntimeError(result.get("error", "Gemini CLI failed"))

    return _GeminiCLIProvider()


def _get_claude_code_provider() -> Optional[LLMProvider]:
    """Get Claude Code provider using existing CLI integration wrapper."""
    try:
        from ipfs_accelerate_py.cli_integrations.claude_code_cli_integration import ClaudeCodeCLI
    except Exception:
        return None

    class _ClaudeCodeProvider:
        def __init__(self):
            self._client = None

        def _get_client(self):
            if self._client is None:
                self._client = ClaudeCodeCLI()
            return self._client

        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            client = self._get_client()
            
            result = client.chat(
                message=prompt,
                model=model_name,
                max_tokens=int(kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256))),
                use_cache=kwargs.get("use_cache", True)
            )
            
            if result.get("success"):
                return result.get("response", "").strip()
            raise RuntimeError(result.get("error", "Claude Code failed"))

    return _ClaudeCodeProvider()


def _get_backend_manager_provider(deps: RouterDeps) -> Optional[LLMProvider]:
    """Get provider that uses InferenceBackendManager for distributed/multiplexed inference."""
    if not _truthy(os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER")):
        return None

    try:
        manager = deps.get_backend_manager(
            purpose="llm_router",
            enable_health_checks=True,
            load_balancing_strategy=os.getenv("IPFS_ACCELERATE_PY_LLM_LOAD_BALANCING", "round_robin"),
        )
        if manager is None:
            return None

        class _BackendManagerProvider:
            def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
                # Select backend for text-generation task
                backend = manager.select_backend_for_task(
                    task="text-generation",
                    model=model_name or os.getenv("IPFS_ACCELERATE_PY_LLM_MODEL", ""),
                    protocol="any"
                )
                
                if backend is None:
                    raise RuntimeError("No available backend for text-generation")
                
                # Execute inference via backend
                payload = {
                    "prompt": prompt,
                    "max_tokens": kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256)),
                    "temperature": kwargs.get("temperature", 0.2),
                }
                
                result = manager.execute_inference(
                    backend_id=backend["id"],
                    task="text-generation",
                    payload=payload
                )
                
                # Extract generated text from result
                text = result.get("generated_text") or result.get("text") or result.get("output")
                if isinstance(text, str) and text:
                    return text
                raise RuntimeError("Backend manager provider did not return generated text")

        return _BackendManagerProvider()
    except Exception as e:
        logger.debug(f"Backend manager provider unavailable: {e}")
        return None


def _provider_cache_key() -> tuple:
    # Include only env vars that change provider resolution.
    return (
        os.getenv("IPFS_ACCELERATE_PY_LLM_PROVIDER", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_API_KEY", "").strip(),
        os.getenv("OPENROUTER_API_KEY", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_MODEL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_BASE_URL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_CODEX_CLI_MODEL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_CODEX_MODEL", "").strip(),
    )


def _deps_provider_cache_key(preferred: Optional[str], cache_key: tuple) -> str:
    digest = hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()[:16]
    return f"llm_provider::{(preferred or '').strip().lower()}::{digest}"


@lru_cache(maxsize=32)
def _resolve_provider_cached(preferred: Optional[str], cache_key: tuple) -> LLMProvider:
    _ = cache_key
    # Use default deps here; custom deps are handled in get_llm_provider.
    return _resolve_provider_uncached(preferred, deps=get_default_router_deps())


def _get_local_hf_provider(*, deps: Optional[RouterDeps] = None) -> Optional[LLMProvider]:
    transformers = _resolve_transformers_module(deps=deps)
    if transformers is None:
        return None

    pipeline = getattr(transformers, "pipeline", None)
    if pipeline is None:
        return None

    class _LocalHFProvider:
        def __init__(self) -> None:
            self._pipelines: Dict[str, object] = {}

        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            model = model_name or os.getenv("IPFS_ACCELERATE_PY_LLM_MODEL", "gpt2")
            pipe = self._pipelines.get(model)
            if pipe is None:
                pipe = pipeline("text-generation", model=model)
                self._pipelines[model] = pipe

            max_new_tokens = int(kwargs.pop("max_new_tokens", kwargs.pop("max_tokens", 128)))
            out = pipe(prompt, max_new_tokens=max_new_tokens)
            if isinstance(out, list) and out:
                item = out[0]
                if isinstance(item, dict) and isinstance(item.get("generated_text"), str):
                    return item["generated_text"]
            return str(out)

    return _LocalHFProvider()


def _builtin_provider_by_name(name: str, deps: RouterDeps) -> Optional[LLMProvider]:
    key = (name or "").strip().lower()
    if not key:
        return None
    if key == "openrouter":
        return _get_openrouter_provider()
    if key in {"codex", "codex_cli"}:
        return _get_codex_cli_provider()
    if key in {"copilot_cli"}:
        return _get_copilot_cli_provider()
    if key in {"copilot_sdk"}:
        return _get_copilot_sdk_provider()
    if key in {"gemini_cli", "gemini"}:
        return _get_gemini_cli_provider()
    if key in {"claude_code", "claude"}:
        return _get_claude_code_provider()
    if key in {"backend_manager", "accelerate"}:
        return _get_backend_manager_provider(deps)
    if key in {"hf", "huggingface", "local_hf"}:
        return _get_local_hf_provider(deps=deps)
    return None


def _resolve_provider_uncached(preferred: Optional[str], *, deps: RouterDeps) -> LLMProvider:
    if preferred:
        info = _PROVIDER_REGISTRY.get(preferred)
        if info is not None:
            return info.factory()
        builtin = _builtin_provider_by_name(preferred, deps)
        if builtin is not None:
            return builtin
        raise ValueError(f"Unknown LLM provider: {preferred}")

    forced = os.getenv("IPFS_ACCELERATE_PY_LLM_PROVIDER", "").strip()
    if forced:
        info = _PROVIDER_REGISTRY.get(forced)
        if info is not None:
            return info.factory()
        builtin = _builtin_provider_by_name(forced, deps)
        if builtin is not None:
            return builtin
        raise ValueError(f"Unknown LLM provider: {forced}")

    # Try backend manager first (for distributed/multiplexed inference)
    backend_manager_provider = _get_backend_manager_provider(deps)
    if backend_manager_provider is not None:
        return backend_manager_provider

    # Try common optional CLI/API providers if available.
    for name in ["openrouter", "codex_cli", "copilot_cli", "copilot_sdk", "gemini_cli", "claude_code"]:
        candidate = _builtin_provider_by_name(name, deps)
        if candidate is not None:
            return candidate

    local_hf = _get_local_hf_provider(deps=deps)
    if local_hf is not None:
        return local_hf

    raise RuntimeError(
        "No LLM provider available. Install `transformers` or register a custom provider."
    )


def get_llm_provider(
    provider: Optional[str] = None,
    *,
    deps: Optional[RouterDeps] = None,
    use_cache: Optional[bool] = None,
) -> LLMProvider:
    """Resolve an LLM provider with optional dependency injection.

    - If ``deps`` is provided, the router will reuse injected/cached dependencies
      (e.g., InferenceBackendManager) stored on that object.
    - If caching is enabled, provider instances are reused in-process to avoid
      repeated initialization cascades.
    """

    resolved_deps = deps or get_default_router_deps()
    cache_ok = _cache_enabled() if use_cache is None else bool(use_cache)

    if not cache_ok:
        return _resolve_provider_uncached(provider, deps=resolved_deps)

    # If a deps container was explicitly provided, cache the provider instance on it.
    # This preserves per-provider internal caches (e.g., HF pipelines) and prevents
    # repeated initialization across call sites and repos.
    if deps is not None:
        cache_key = _provider_cache_key()
        deps_key = _deps_provider_cache_key(provider, cache_key)
        cached = resolved_deps.get_cached(deps_key)
        if cached is not None:
            return cached
        return resolved_deps.set_cached(deps_key, _resolve_provider_uncached(provider, deps=resolved_deps))

    # Process-global caching path.
    return _resolve_provider_cached(provider, _provider_cache_key())


def generate_text(
    prompt: str,
    *,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    provider_instance: Optional[LLMProvider] = None,
    deps: Optional[RouterDeps] = None,
    **kwargs: object,
) -> str:
    """Generate text from an LLM.
    
    Args:
        prompt: The text prompt to generate from
        model_name: Optional model name to use
        provider: Optional provider name to use
        provider_instance: Optional pre-created provider instance
        deps: Optional RouterDeps for dependency injection
        **kwargs: Additional arguments passed to the provider
        
    Returns:
        Generated text string
    """

    resolved_deps = deps or get_default_router_deps()
    if _response_cache_enabled():
        try:
            cache_key = _response_cache_key(provider=provider, model_name=model_name, prompt=prompt, kwargs=dict(kwargs))
            getter = getattr(resolved_deps, "get_cached_or_remote", None)
            cached = getter(cache_key) if callable(getter) else resolved_deps.get_cached(cache_key)
            if isinstance(cached, str):
                logger.debug(f"Using cached LLM response for prompt: {prompt[:50]}...")
                return cached
        except Exception:
            pass

    backend = provider_instance or get_llm_provider(provider, deps=resolved_deps)
    try:
        result = backend.generate(prompt, model_name=model_name, **kwargs)
        if _response_cache_enabled():
            try:
                cache_key = _response_cache_key(provider=provider, model_name=model_name, prompt=prompt, kwargs=dict(kwargs))
                setter = getattr(resolved_deps, "set_cached_and_remote", None)
                if callable(setter):
                    setter(cache_key, str(result))
                else:
                    resolved_deps.set_cached(cache_key, str(result))
            except Exception:
                pass
        return result
    except Exception:
        # If a specific model was requested but isn't available for this provider,
        # retry with the provider's default model before other fallbacks.
        if model_name is not None:
            try:
                result = backend.generate(prompt, model_name=None, **kwargs)
                if _response_cache_enabled():
                    try:
                        cache_key = _response_cache_key(provider=provider, model_name=None, prompt=prompt, kwargs=dict(kwargs))
                        setter = getattr(resolved_deps, "set_cached_and_remote", None)
                        if callable(setter):
                            setter(cache_key, str(result))
                        else:
                            resolved_deps.set_cached(cache_key, str(result))
                    except Exception:
                        pass
                return result
            except Exception:
                pass

        # Fall back to local HF provider if optional provider fails.
        if provider is None:
            local_hf = _get_local_hf_provider(deps=resolved_deps)
            if local_hf is not None and backend is not local_hf:
                try:
                    result = local_hf.generate(prompt, model_name=model_name, **kwargs)
                    if _response_cache_enabled():
                        try:
                            cache_key = _response_cache_key(provider=provider, model_name=model_name, prompt=prompt, kwargs=dict(kwargs))
                            setter = getattr(resolved_deps, "set_cached_and_remote", None)
                            if callable(setter):
                                setter(cache_key, str(result))
                            else:
                                resolved_deps.set_cached(cache_key, str(result))
                        except Exception:
                            pass
                    return result
                except Exception:
                    if model_name is not None:
                        result = local_hf.generate(prompt, model_name=None, **kwargs)
                        if _response_cache_enabled():
                            try:
                                cache_key = _response_cache_key(provider=provider, model_name=None, prompt=prompt, kwargs=dict(kwargs))
                                setter = getattr(resolved_deps, "set_cached_and_remote", None)
                                if callable(setter):
                                    setter(cache_key, str(result))
                                else:
                                    resolved_deps.set_cached(cache_key, str(result))
                            except Exception:
                                pass
                        return result
        raise


def clear_llm_router_caches() -> None:
    """Clear internal provider caches (useful for tests)."""

    _resolve_provider_cached.cache_clear()
