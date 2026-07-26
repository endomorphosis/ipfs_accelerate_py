"""LLM router.

This module provides a reusable top-level entrypoint for text generation.

Design goals:
- Avoid import-time side effects.
- Allow optional hooks/providers (ipfs_accelerate_py, remote endpoints).
- Provide a local HuggingFace transformers fallback when available.

Environment variables:
- `ipfs_accelerate_py_LLM_PROVIDER`: force provider name (registered provider)
- `ipfs_accelerate_py_ENABLE_IPFS_ACCELERATE`: control accelerate provider (best-effort hook)
    - unset: prefer accelerate when available
    - truthy: force-enable accelerate attempt
    - falsy (0/false/no): disable accelerate provider
- `ipfs_accelerate_py_LLM_MODEL`: default HF model name for local fallback

Additional optional providers (opt-in by selecting provider):
- `openrouter`: OpenRouter chat completions
    - `OPENROUTER_API_KEY` or `ipfs_accelerate_py_OPENROUTER_API_KEY`
    - `ipfs_accelerate_py_OPENROUTER_MODEL` (default model)
    - `ipfs_accelerate_py_OPENROUTER_BASE_URL` (default: https://openrouter.ai/api/v1)
- `codex_cli`: OpenAI Codex CLI via `codex exec`
    - `ipfs_accelerate_py_CODEX_CLI_MODEL` / `ipfs_accelerate_py_CODEX_MODEL`
- `copilot_cli`: GitHub Copilot CLI via command template
    - `ipfs_accelerate_py_COPILOT_CLI_CMD` (supports `{prompt}` placeholder)
- `copilot_sdk`: Python `copilot` SDK (if installed)
    - `ipfs_accelerate_py_COPILOT_SDK_MODEL`, `ipfs_accelerate_py_COPILOT_SDK_TIMEOUT`
- `gemini_cli`: Gemini CLI via `npx @google/gemini-cli`
    - `ipfs_accelerate_py_GEMINI_CLI_CMD` (supports `{prompt}` placeholder)
- `grok_cli`: xAI Grok Build CLI via the official `grok` binary
    - `ipfs_accelerate_py_GROK_CLI_CMD` (supports `{prompt}` and `{model}` placeholders)
    - `ipfs_accelerate_py_GROK_CLI_MODEL` (default: grok-4.5; run `grok models`)
    - Authenticate with `grok login` or `XAI_API_KEY`
- `gemini_py`: Python wrapper in `ipfs_accelerate_py.utils.gemini_cli.GeminiCLI`
- `claude_code`: Claude Code CLI command
    - `ipfs_accelerate_py_CLAUDE_CODE_CLI_CMD` (supports `{prompt}` placeholder)
- `claude_py`: Python wrapper in `ipfs_accelerate_py.utils.claude_cli.ClaudeCLI`
- `mistral_vibe`: Mistral Vibe CLI (`vibe`)
    - Explicit provider selection installs `mistral-vibe` with `uv tool` when missing
    - `IPFS_ACCELERATE_MISTRAL_VIBE_AUTO_INSTALL=0` disables installation
    - `IPFS_ACCELERATE_MISTRAL_VIBE_CLI_CMD` (supports `{prompt}`, `{model}`, and `{agent}`)
    - `IPFS_ACCELERATE_MISTRAL_VIBE_MODEL` (optional default model)
    - `MISTRAL_API_KEY` or `IPFS_ACCELERATE_MISTRAL_API_KEY` for auth
    - `ipfs_accelerate_py_MISTRAL_VIBE_CLI_CMD` (supports `{prompt}` and `{model}` placeholders)
    - `ipfs_accelerate_py_MISTRAL_VIBE_MODEL` (optional default model)
    - `MISTRAL_API_KEY` or `ipfs_accelerate_py_MISTRAL_API_KEY` for auth
- `xai`: xAI Grok AI (REST API, OpenAI-compatible)
    - `XAI_API_KEY` or `ipfs_accelerate_py_XAI_API_KEY`
    - `ipfs_accelerate_py_XAI_MODEL` (default model: grok-4.5)
    - `ipfs_accelerate_py_XAI_BASE_URL` (default: https://api.x.ai/v1)
- `meta_ai`: Meta Model API / Muse Spark (OpenAI-compatible)
    - encrypted credential `meta_ai_api_key`, `MODEL_API_KEY`,
      `META_AI_API_KEY`, or `ipfs_accelerate_py_META_AI_API_KEY`
    - `ipfs_accelerate_py_META_AI_MODEL` (default: muse-spark-1.1)
    - `ipfs_accelerate_py_META_AI_BASE_URL` (default: https://api.meta.ai/v1)
- `goose_cli`: Block/AAIF Goose CLI via `goose run`
    - chat-only by default (`GOOSE_MODE=chat`, no tools/extensions/session)
    - default backend is Meta Muse Spark through OpenAI-compatible env
      (`OPENAI_HOST=https://api.meta.ai`, package Meta API key)
    - `ipfs_accelerate_py_GOOSE_CLI_MODEL` / `GOOSE_MODEL` (default: muse-spark-1.1)
    - `ipfs_accelerate_py_GOOSE_BIN` or `goose` on PATH
    - pass `agent=True` / `side_effecting=True` only for explicitly authorized
      tool-using agent runs (developer extension, auto mode)
- `llama_cpp`: local llama.cpp OpenAI-compatible server
    - `IPFS_ACCELERATE_LLAMA_CPP_BASE_URL` (default from host/port: http://127.0.0.1:8080/v1)
    - `IPFS_ACCELERATE_LLAMA_CPP_MODEL` (chat-completions model id; defaults to Leanstral NVFP4 ref)
    - `IPFS_ACCELERATE_LLAMA_CPP_MODEL_REF` (server `-hf` model ref; defaults to Frosty40 Leanstral NVFP4)
    - `IPFS_ACCELERATE_LLAMA_CPP_HF_FILE` (optional exact GGUF file for `--hf-file`)
    - `IPFS_ACCELERATE_LLAMA_CPP_AUTOSTART=1` starts `llama serve`/`llama-server` when absent
    - `IPFS_ACCELERATE_LLAMA_CPP_PREFETCH_MODEL=1` downloads the configured GGUF before serving
    - `IPFS_ACCELERATE_LLAMA_CPP_AUTO_INSTALL=1` allows the configured installer when CLI is missing
    - `IPFS_ACCELERATE_LLAMA_CPP_AUTO_UPDATE=1` allows the configured updater before serving
- `llama_cpp_native`: local Python `llama_cpp.Llama` binding
    - `IPFS_ACCELERATE_LLAMA_CPP_NATIVE_MODEL_PATH` uses a local GGUF directly
    - `IPFS_ACCELERATE_LLAMA_CPP_NATIVE_MODEL_REF` / `IPFS_ACCELERATE_LLAMA_CPP_NATIVE_HF_FILE`
      select a Hugging Face-hosted GGUF for `Llama.from_pretrained`
    - `IPFS_ACCELERATE_LLAMA_CPP_NATIVE_AUTO_INSTALL=1` permits pip installing
      `llama-cpp-python[server]` when the binding is missing
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from html import unescape
import hashlib
import importlib
import importlib.util
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    runtime_checkable,
)

from .common.meta_model_api import (
    META_MODEL_API_BASE_URL,
    META_MODEL_API_DEFAULT_MODEL,
    meta_model_api_key_fingerprint,
    normalize_meta_model_name,
    resolve_meta_model_api_key,
)
from .model_catalog import (
    CapabilityDescriptor,
    CatalogSnapshot,
    LifecycleState,
    Modality,
    ModelDescriptor,
    Operation,
    OperationalState,
    ProviderDescriptor,
    Provenance,
    RouterBinding,
)
from .router_deps import RouterDeps, get_default_router_deps
from .utils.mistral_vibe import (
    MistralVibeInstallResult,
    ensure_mistral_vibe,
    mistral_vibe_auth_available,
)


class LLMRouterError(RuntimeError):
    """Errors raised by lightweight router helpers/providers.

    This is intentionally a RuntimeError subclass so existing call sites that
    catch RuntimeError continue to work.
    """


class PinnedSymaiCompletionError(LLMRouterError):
    """Secret-safe failure raised for a rejected pinned SyMAI completion."""

    OUTPUT_TOKEN_LIMIT = "output_token_limit"
    _SAFE_FAILURE_CLASSES = frozenset({OUTPUT_TOKEN_LIMIT})

    def __init__(self, safe_failure_class: str) -> None:
        if safe_failure_class not in self._SAFE_FAILURE_CLASSES:
            raise ValueError(
                "unsupported pinned SyMAI completion failure class"
            )
        self.safe_failure_class = safe_failure_class
        super().__init__(
            "pinned SyMAI completion failed: " + safe_failure_class
        )


_P2P_TASK_PREFIX = "p2p://"
_HF_ARCH_ROUTER_MODEL_ID = "katanemo/Arch-Router-1.5B"
_GROK_CLI_PROVIDER_ALIASES = {
    "grok_cli",
    "grok-cli",
    "xai_cli",
    "xai-cli",
    "grok_build",
    "grok-build",
    "grok_build_cli",
    "grok-build-cli",
}
_XAI_API_PROVIDER_ALIASES = {
    "xai",
    "xai_api",
    "xai-api",
    "xai_grok",
    "grok_api",
    "grok-api",
}
_LAST_GENERATION_TRACE = threading.local()
_PINNED_SYMAI_LEANSTRAL_ALIAS = "Leanstral-119B"
_PINNED_SYMAI_LEANSTRAL_INNER_PROVIDER = "leanstral_local"
_PINNED_SYMAI_LEANSTRAL_MODEL = (
    "Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4:NVFP4"
)
_PINNED_SYMAI_LEANSTRAL_ENDPOINT = "http://127.0.0.1:8080/v1"
_SYMAI_ROUTE_BINDING_KWARG = "_symai_route_binding"
_PINNED_SYMAI_TRACE_KEYS = frozenset(
    {
        "resolved_provider_name",
        "resolved_model_name",
        "service_endpoint",
        "routing_backend",
    }
)
_PINNED_SYMAI_ROUTE_BINDING = {
    "resolved_provider_name": _PINNED_SYMAI_LEANSTRAL_INNER_PROVIDER,
    "resolved_model_name": _PINNED_SYMAI_LEANSTRAL_MODEL,
    "service_endpoint": _PINNED_SYMAI_LEANSTRAL_ENDPOINT,
    "routing_backend": "existing_leanstral_service",
}
_PINNED_SYMAI_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "candidate_ir",
        "normalized_predicates",
        "quantifiers",
        "entities",
        "ambiguity_flags",
        "confidence",
        "validation_errors",
    ],
    "properties": {
        "candidate_ir": {
            "type": "object",
            "additionalProperties": False,
            "required": ["propositions"],
            "properties": {
                "propositions": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {"type": "string", "maxLength": 80},
                }
            },
        },
        "normalized_predicates": {
            "type": "array",
            "maxItems": 24,
            "items": {"type": "string", "maxLength": 80},
        },
        "quantifiers": {
            "type": "array",
            "maxItems": 24,
            "items": {"type": "string", "maxLength": 80},
        },
        "entities": {
            "type": "array",
            "maxItems": 24,
            "items": {"type": "string", "maxLength": 80},
        },
        "ambiguity_flags": {
            "type": "array",
            "maxItems": 24,
            "items": {"type": "string", "maxLength": 80},
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "validation_errors": {
            "type": "array",
            "maxItems": 24,
            "items": {"type": "string", "maxLength": 80},
        },
    },
}
_PINNED_SYMAI_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "hssl_symai_semantic_evidence",
        "strict": True,
        "schema": _PINNED_SYMAI_RESPONSE_SCHEMA,
    },
}
_PINNED_SYMAI_REALIZATION_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text"],
    "properties": {"text": {"type": "string"}},
}
_PINNED_SYMAI_REALIZATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "semantic_legal_realization",
        "strict": True,
        "schema": _PINNED_SYMAI_REALIZATION_RESPONSE_SCHEMA,
    },
}
_PINNED_SYMAI_ALLOWED_RESPONSE_FORMATS = (
    _PINNED_SYMAI_RESPONSE_FORMAT,
    _PINNED_SYMAI_REALIZATION_RESPONSE_FORMAT,
)
_PROVIDER_ALIASES = {
    "gpt4": "openai",
    "gpt-4": "openai",
    "codex": "codex_cli",
    "codex-cli": "codex_cli",
    "copilot": "copilot_cli",
    "gemini": "gemini_cli",
    "hf_api": "hf_inference_api",
    "hf_inference": "hf_inference_api",
    "huggingface_inference": "hf_inference_api",
    "p2p": "p2p_task_queue",
    "p2p_task": "p2p_task_queue",
    "remote_queue": "p2p_task_queue",
    "task_queue": "p2p_task_queue",
    "llamacpp": "llama_cpp",
    "llama.cpp": "llama_cpp",
    "openai_compatible": "llama_cpp",
    "local_openai": "llama_cpp",
    "leanstral_local": "llama_cpp",
    "llamacpp_native": "llama_cpp_native",
    "llama.cpp_native": "llama_cpp_native",
    "native_llama_cpp": "llama_cpp_native",
    "goose": "goose_cli",
    "goose-cli": "goose_cli",
    "block_goose": "goose_cli",
    "block-goose": "goose_cli",
    "aaif_goose": "goose_cli",
}
_GOOSE_CLI_PROVIDER_ALIASES = {
    "goose_cli",
    "goose",
    "goose-cli",
    "block_goose",
    "block-goose",
    "aaif_goose",
}
_UNPINNED_OPTIONAL_PROVIDER_ORDER = [
    "codex_cli",
    "copilot_cli",
    "goose_cli",
    "openai",
    "hf_inference_api",
    "openrouter",
    "gemini_cli",
    "claude_code",
    "mistral_vibe",
    "claude_py",
    "gemini_py",
    "copilot_sdk",
]

_LLM_GENERATE_PROVIDER_FORWARD_KEYS = (
    "max_new_tokens",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stop",
    "seed",
    "logprobs",
    "top_logprobs",
    "response_format",
    "timeout",
)
_LLM_GENERATE_TRACE_FORWARD_KEYS = (
    "trace",
    "trace_jsonl_path",
    "trace_dir",
)
_LLM_GENERATE_COPILOT_FORWARD_KEYS = (
    "copilot_config_dir",
    "copilot_log_dir",
)
_LLM_GENERATE_SESSION_FORWARD_KEYS = (
    "resume_session_id",
    "continue_session",
    "chat_session_id",
    "history_cid",
    "sticky_worker_id",
)
_LLM_GENERATE_SAFE_FORWARD_KEYS = (
    *_LLM_GENERATE_PROVIDER_FORWARD_KEYS,
    *_LLM_GENERATE_TRACE_FORWARD_KEYS,
    *_LLM_GENERATE_COPILOT_FORWARD_KEYS,
)
_LLM_GENERATE_TASK_FORWARD_KEYS = (
    *_LLM_GENERATE_SAFE_FORWARD_KEYS,
    *_LLM_GENERATE_SESSION_FORWARD_KEYS,
)


def _llm_generate_forwarded_kwargs(
    kwargs: Dict[str, object] | dict,
    *,
    include_session: bool = False,
) -> Dict[str, object]:
    keys = _LLM_GENERATE_TASK_FORWARD_KEYS if include_session else _LLM_GENERATE_SAFE_FORWARD_KEYS
    return {str(k): kwargs[k] for k in keys if k in kwargs}


def _truthy_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _p2p_auto_discovery_enabled() -> bool:
    # Default to enabled when libp2p is installed, but keep the attempt cheap:
    # we only auto-dial when we have a concrete hint (announce file / bootstrap)
    # or when the user explicitly requests it.
    explicit = os.environ.get("ipfs_accelerate_py_TASK_P2P_AUTO_DISCOVERY")
    if explicit is None:
        explicit = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_AUTO_DISCOVERY")
    if explicit is None:
        return True
    return str(explicit).strip().lower() in {"1", "true", "yes", "on"}


def _have_libp2p() -> bool:
    return importlib.util.find_spec("libp2p") is not None


def _default_task_p2p_announce_files() -> list[str]:
    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return [
        os.path.join(cache_root, "ipfs_accelerate_py", "task_p2p_announce.json"),
        os.path.join(cache_root, "ipfs_accelerate_py", "task_p2p_announce.json"),
    ]


def _read_task_p2p_announce() -> dict | None:
    # Optional env override.
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE")
        or os.environ.get("ipfs_accelerate_py_TASK_P2P_ANNOUNCE_FILE")
    )
    if raw is not None and str(raw).strip().lower() in {"0", "false", "no", "off"}:
        return None

    candidates: list[str] = []
    if raw is not None and str(raw).strip():
        candidates.append(str(raw).strip())
    candidates.extend(_default_task_p2p_announce_files())

    for path in candidates:
        try:
            if not path:
                continue
            if not os.path.exists(path):
                continue
            text = open(path, "r", encoding="utf-8").read().strip()
            if not text:
                continue
            info = json.loads(text)
            if isinstance(info, dict) and isinstance(info.get("multiaddr"), str) and "/p2p/" in str(info.get("multiaddr")):
                return info
        except Exception:
            continue
    return None


def _encode_p2p_task_id(*, peer_id: str, task_id: str) -> str:
    pid = (peer_id or "").strip()
    tid = (task_id or "").strip()
    if not pid or not tid:
        return tid
    return f"{_P2P_TASK_PREFIX}{pid}/{tid}"


def _decode_p2p_task_id(task_id: str) -> tuple[str, str] | None:
    text = str(task_id or "").strip()
    if not text.startswith(_P2P_TASK_PREFIX):
        return None
    rest = text[len(_P2P_TASK_PREFIX) :]
    if "/" not in rest:
        return None
    pid, tid = rest.split("/", 1)
    pid = pid.strip()
    tid = tid.strip()
    if not pid or not tid:
        return None
    return pid, tid


def _extract_peer_id_from_multiaddr(multiaddr: str) -> str:
    text = str(multiaddr or "").strip()
    if not text:
        return ""
    m = re.search(r"/p2p/([^/]+)$", text)
    return (m.group(1) if m else "").strip()


def submit_task(
    *,
    prompt: str,
    model_name: str = "gpt2",
    task_type: str = "text-generation",
    queue_path: Optional[str] = None,
    **kwargs: object,
) -> str:
    """Submit an LLM task to a local task queue, or to a remote peer via libp2p.

    This provides a simple multi-worker delegation mechanism.
    Workers can be run via `python -m ipfs_accelerate_py.p2p_tasks.worker`.
    """

    remote_peer_id = (
        os.environ.get("ipfs_accelerate_py_TASK_P2P_REMOTE_PEER_ID")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_PEER_ID")
        or ""
    ).strip()
    remote_multiaddr = (
        os.environ.get("ipfs_accelerate_py_TASK_P2P_REMOTE_MULTIADDR")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_MULTIADDR")
        or ""
    ).strip()

    announce = _read_task_p2p_announce() if not remote_multiaddr else None
    if announce and not remote_multiaddr:
        remote_multiaddr = str(announce.get("multiaddr") or "").strip()
        remote_peer_id = str(announce.get("peer_id") or "").strip() or remote_peer_id

    auto_discovery = _p2p_auto_discovery_enabled()

    try:
        from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
    except Exception as exc:
        raise LLMRouterError("Task delegation helpers not available") from exc

    payload: Dict[str, object] = {"prompt": str(prompt or "")}
    payload.update(_llm_generate_forwarded_kwargs(kwargs))

    # Session-affinity + provider routing for interactive LLM tasks.
    # This is primarily intended for mesh execution of copilot_cli prompts.
    ttype_norm = str(task_type or "").strip().lower()
    if ttype_norm in {"llm.generate", "llm_generate"}:
        provider = str(kwargs.get("provider") or "copilot_cli").strip() or "copilot_cli"
        payload["provider"] = provider

        # Optional: encrypt prompt to avoid transmitting/storing plaintext.
        # Routing keys must remain readable for claim-time enforcement.
        try:
            from ipfs_accelerate_py.p2p_tasks.protocol import encrypt_text

            enc = encrypt_text(str(payload.get("prompt") or ""))
            if isinstance(enc, dict):
                payload.pop("prompt", None)
                payload["prompt_enc"] = enc
        except Exception:
            pass

        payload.update(_llm_generate_forwarded_kwargs(kwargs, include_session=True))

        sid = kwargs.get("session_id")
        if not (isinstance(sid, str) and sid.strip()):
            sid = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_SESSION") or ""
        if isinstance(sid, str) and sid.strip():
            payload["session_id"] = sid.strip()

    # Avoid slow discovery attempts by default: only try when
    # - caller explicitly configured a multiaddr, OR
    # - we have an announce hint (local service), OR
    # - user explicitly enables auto-discovery, AND libp2p is installed.
    have_hint = bool(remote_multiaddr)
    explicit_discovery = os.environ.get("ipfs_accelerate_py_TASK_P2P_AUTO_DISCOVERY") is not None or os.environ.get(
        "IPFS_ACCELERATE_PY_TASK_P2P_AUTO_DISCOVERY"
    ) is not None

    should_try_p2p = bool(remote_multiaddr) or (explicit_discovery and auto_discovery)
    if not should_try_p2p and announce is not None:
        should_try_p2p = True

    if should_try_p2p and _have_libp2p():
        try:
            import anyio

            from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue
            from ipfs_accelerate_py.p2p_tasks.client import submit_task_with_info

            remote = RemoteQueue(peer_id=remote_peer_id or "", multiaddr=remote_multiaddr)

            async def _run() -> dict:
                return await submit_task_with_info(
                    remote=remote,
                    task_type=str(task_type),
                    model_name=str(model_name or ""),
                    payload=payload,  # type: ignore[arg-type]
                )

            info = anyio.run(_run, backend="trio")
            if isinstance(info, dict):
                tid = str(info.get("task_id") or "").strip()
                pid = str(info.get("peer_id") or "").strip() or remote_peer_id or _extract_peer_id_from_multiaddr(remote_multiaddr)
                if pid and tid:
                    return _encode_p2p_task_id(peer_id=pid, task_id=tid)
                if tid:
                    return tid
            raise RuntimeError(f"invalid_submit_response: {info}")
        except Exception as exc:
            # If the caller explicitly configured a remote multiaddr, fail loudly.
            if remote_multiaddr:
                raise LLMRouterError(f"P2P submit_task failed: {exc}") from exc
            # Auto-discovery is best-effort: fall back to local queue.
            pass

    q = TaskQueue(queue_path)
    return q.submit(task_type=str(task_type), model_name=str(model_name or ""), payload=payload)


def get_task(task_id: str, *, queue_path: Optional[str] = None) -> Optional[dict]:
    """Get task status/result from the local task queue, or from a remote peer via libp2p."""

    parsed = _decode_p2p_task_id(str(task_id))

    remote_peer_id = (
        os.environ.get("ipfs_accelerate_py_TASK_P2P_REMOTE_PEER_ID")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_PEER_ID")
        or ""
    ).strip()
    remote_multiaddr = (
        os.environ.get("ipfs_accelerate_py_TASK_P2P_REMOTE_MULTIADDR")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_MULTIADDR")
        or ""
    ).strip()
    auto_discovery = _p2p_auto_discovery_enabled()

    effective_peer_id = parsed[0] if parsed else remote_peer_id
    effective_task_id = parsed[1] if parsed else str(task_id)

    announce = _read_task_p2p_announce() if not remote_multiaddr else None
    if announce and not remote_multiaddr:
        remote_multiaddr = str(announce.get("multiaddr") or "").strip()
        remote_peer_id = str(announce.get("peer_id") or "").strip() or remote_peer_id

    explicit_discovery = os.environ.get("ipfs_accelerate_py_TASK_P2P_AUTO_DISCOVERY") is not None or os.environ.get(
        "IPFS_ACCELERATE_PY_TASK_P2P_AUTO_DISCOVERY"
    ) is not None
    should_try_p2p = bool(parsed is not None or remote_multiaddr)
    if not should_try_p2p and announce is not None:
        should_try_p2p = True
    if not should_try_p2p and explicit_discovery and auto_discovery and effective_peer_id:
        should_try_p2p = True

    if should_try_p2p and _have_libp2p():
        try:
            import anyio

            from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue
            from ipfs_accelerate_py.p2p_tasks.client import get_task as get_task_p2p

            remote = RemoteQueue(peer_id=effective_peer_id or "", multiaddr=remote_multiaddr)

            async def _run() -> Optional[dict]:
                task = await get_task_p2p(remote=remote, task_id=str(effective_task_id))
                return task if isinstance(task, dict) else None

            return anyio.run(_run, backend="trio")
        except Exception:
            return None

    try:
        from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
    except Exception:
        return None
    return TaskQueue(queue_path).get(task_id)


def wait_task(
    task_id: str,
    *,
    queue_path: Optional[str] = None,
    timeout_s: float = 60.0,
) -> Optional[dict]:
    """Wait for a task to complete.

    - Local: polls SQLite queue.
    - P2P: uses remote peer's wait RPC.
    """

    parsed = _decode_p2p_task_id(str(task_id))

    remote_peer_id = (
        os.environ.get("ipfs_accelerate_py_TASK_P2P_REMOTE_PEER_ID")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_PEER_ID")
        or ""
    ).strip()
    remote_multiaddr = (
        os.environ.get("ipfs_accelerate_py_TASK_P2P_REMOTE_MULTIADDR")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_MULTIADDR")
        or ""
    ).strip()
    auto_discovery = _p2p_auto_discovery_enabled()

    effective_peer_id = parsed[0] if parsed else remote_peer_id
    effective_task_id = parsed[1] if parsed else str(task_id)

    announce = _read_task_p2p_announce() if not remote_multiaddr else None
    if announce and not remote_multiaddr:
        remote_multiaddr = str(announce.get("multiaddr") or "").strip()
        remote_peer_id = str(announce.get("peer_id") or "").strip() or remote_peer_id

    explicit_discovery = os.environ.get("ipfs_accelerate_py_TASK_P2P_AUTO_DISCOVERY") is not None or os.environ.get(
        "IPFS_ACCELERATE_PY_TASK_P2P_AUTO_DISCOVERY"
    ) is not None
    should_try_p2p = bool(parsed is not None or remote_multiaddr)
    if not should_try_p2p and announce is not None:
        should_try_p2p = True
    if not should_try_p2p and explicit_discovery and auto_discovery and effective_peer_id:
        should_try_p2p = True

    if should_try_p2p and _have_libp2p():
        try:
            import anyio

            from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue
            from ipfs_accelerate_py.p2p_tasks.client import wait_task as wait_task_p2p

            remote = RemoteQueue(peer_id=effective_peer_id or "", multiaddr=remote_multiaddr)

            async def _run() -> Optional[dict]:
                task = await wait_task_p2p(remote=remote, task_id=str(effective_task_id), timeout_s=float(timeout_s))
                return task if isinstance(task, dict) else None

            return anyio.run(_run, backend="trio")
        except Exception:
            return None

    try:
        from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
    except Exception:
        return None

    import time

    q = TaskQueue(queue_path)
    deadline = time.time() + max(0.0, float(timeout_s))
    task = q.get(str(task_id))
    while task is not None and task.get("status") in {"queued", "running"} and time.time() < deadline:
        time.sleep(0.1)
        task = q.get(str(task_id))
    return task if isinstance(task, dict) else None


# In-process best-effort registry for routing chat session resumes.
#
# Keyed by either `chat_session_id` (preferred) or provider session ids such as
# `resume_session_id` when present.
_STICKY_SESSION_WORKER: dict[str, str] = {}


_CHAT_HISTORY_LOCK = threading.RLock()
_CHAT_HISTORY_INDEX: dict[str, str] = {}
_CHAT_HISTORY_INDEX_LOADED = False


def _chat_history_cache_dir() -> Path:
    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(cache_root) / "ipfs_accelerate_py" / "chat_history"


def _chat_history_index_path() -> Path:
    return _chat_history_cache_dir() / "index.json"


def _safe_cid_filename(cid: str) -> str:
    # CIDs are typically safe as filenames. Keep a conservative fallback.
    out = str(cid or "").strip()
    if not out:
        return ""
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in out)


def _cid_for_bytes(data: bytes) -> str:
    # Prefer multiformats CIDv1 (raw, sha2-256) when available.
    try:
        from multiformats import CID, multihash  # type: ignore

        mh = multihash.digest(data, "sha2-256")
        try:
            cid = CID("base32", 1, "raw", mh)
        except TypeError:
            # Older constructor variant.
            cid = CID("base32", "raw", mh)
        return str(cid)
    except Exception:
        return "sha256_" + hashlib.sha256(data).hexdigest()


def _cid_for_text(text: str) -> str:
    return _cid_for_bytes(str(text or "").encode("utf-8"))


def _load_chat_history_index() -> None:
    global _CHAT_HISTORY_INDEX_LOADED
    if _CHAT_HISTORY_INDEX_LOADED:
        return
    path = _chat_history_index_path()
    try:
        if path.exists() and path.stat().st_size > 0:
            data = json.loads(path.read_text("utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    ks = str(k or "").strip()
                    vs = str(v or "").strip()
                    if ks and vs:
                        _CHAT_HISTORY_INDEX[ks] = vs
    except Exception:
        pass
    _CHAT_HISTORY_INDEX_LOADED = True


def _save_chat_history_index() -> None:
    path = _chat_history_index_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_CHAT_HISTORY_INDEX, sort_keys=True), "utf-8")
    except Exception:
        pass


def _store_chat_history_text(text: str) -> str:
    cid = _cid_for_text(text)
    base = _chat_history_cache_dir() / "cids"
    try:
        base.mkdir(parents=True, exist_ok=True)
        fname = _safe_cid_filename(cid)
        if fname:
            path = base / f"{fname}.txt"
            if not path.exists():
                path.write_text(str(text or ""), "utf-8")
    except Exception:
        pass
    return cid


def _load_chat_history_text(cid: str) -> str | None:
    text = str(cid or "").strip()
    if not text:
        return None
    base = _chat_history_cache_dir() / "cids"
    try:
        path = base / f"{_safe_cid_filename(text)}.txt"
        if path.exists() and path.stat().st_size > 0:
            return path.read_text("utf-8")
    except Exception:
        return None
    return None


def _chat_history_get(chat_session_id: str) -> tuple[str | None, str | None]:
    sid = str(chat_session_id or "").strip()
    if not sid:
        return (None, None)
    with _CHAT_HISTORY_LOCK:
        _load_chat_history_index()
        cid = str(_CHAT_HISTORY_INDEX.get(sid) or "").strip()
    if not cid:
        return (None, None)
    return (_load_chat_history_text(cid), cid)


def _chat_history_append_turn(*, chat_session_id: str, user_prompt: str, assistant_text: str) -> str | None:
    sid = str(chat_session_id or "").strip()
    if not sid:
        return None

    # Simple plaintext transcript; intended for best-effort failover prompts.
    turn = f"User: {str(user_prompt or '').strip()}\nAssistant: {str(assistant_text or '').strip()}".strip()
    if not turn:
        return None

    with _CHAT_HISTORY_LOCK:
        _load_chat_history_index()
        prior_cid = str(_CHAT_HISTORY_INDEX.get(sid) or "").strip()
        prior_text = _load_chat_history_text(prior_cid) if prior_cid else None
        merged = (str(prior_text or "").strip() + "\n\n" + turn).strip() if str(prior_text or "").strip() else turn
        cid = _store_chat_history_text(merged)
        _CHAT_HISTORY_INDEX[sid] = cid
        _save_chat_history_index()
        return cid


def generate_text_mesh(
    prompt: str,
    *,
    model_name: Optional[str] = None,
    provider: str = "copilot_cli",
    session_id: str | None = None,
    chat_session_id: str | None = None,
    resume_session_id: str | None = None,
    continue_session: bool = False,
    history: str | None = None,
    timeout_s: float = 90.0,
    max_route_attempts: int = 3,
    queue_path: Optional[str] = None,
    **kwargs: object,
) -> str:
    """Generate text by delegating `llm.generate` tasks to the P2P mesh.

    Goals:
    - Highest throughput: any eligible worker may claim new sessions.
    - Session resume safety: resumed sessions can be pinned to the worker that
      originally handled the session via `sticky_worker_id`.
    - Failover: after `max_route_attempts` timeouts, fall back to starting a new
      session using the recovered `history` (if provided).

        Notes:
        - Sticky routing is best-effort and stored in-process. For long-lived
            orchestrators this is usually sufficient.
        - Session continuity flags (`resume_session_id` / `continue_session`) are
            only supported for provider='copilot_cli'.
    """

    provider_norm = str(provider or "").strip().lower() or "copilot_cli"
    if provider_norm != "copilot_cli":
        if (isinstance(resume_session_id, str) and resume_session_id.strip()) or bool(continue_session):
            raise LLMRouterError(
                "resume/continue session flags are only supported for provider='copilot_cli'"
            )

    try:
        attempts = int(max_route_attempts)
    except Exception:
        attempts = 3
    attempts = max(1, min(attempts, 10))

    # Build a base payload with only safe/expected keys.
    forwarded: Dict[str, object] = {}
    if isinstance(session_id, str) and session_id.strip():
        forwarded["session_id"] = session_id.strip()
    if isinstance(chat_session_id, str) and chat_session_id.strip():
        forwarded["chat_session_id"] = chat_session_id.strip()
    if isinstance(resume_session_id, str) and resume_session_id.strip():
        forwarded["resume_session_id"] = resume_session_id.strip()
    if continue_session:
        forwarded["continue_session"] = True

    # Forward a known-safe allowlist of provider args. This intentionally does
    # not allow provider-specific command overrides such as gemini_cmd.
    forwarded.update(_llm_generate_forwarded_kwargs(kwargs))

    def _sticky_key() -> str:
        if isinstance(chat_session_id, str) and chat_session_id.strip():
            return chat_session_id.strip()
        if isinstance(resume_session_id, str) and resume_session_id.strip():
            return resume_session_id.strip()
        return ""

    sticky_key = _sticky_key()
    sticky_worker_id = _STICKY_SESSION_WORKER.get(sticky_key, "") if sticky_key else ""

    # Best-effort: carry forward an archived history CID when available.
    history_cid: str | None = None
    if not (isinstance(history, str) and history.strip()) and isinstance(chat_session_id, str) and chat_session_id.strip():
        try:
            _hist_text, _hist_cid = _chat_history_get(chat_session_id.strip())
            if isinstance(_hist_cid, str) and _hist_cid.strip():
                history_cid = _hist_cid.strip()
        except Exception:
            pass

    last_task_id: str | None = None
    for i in range(attempts):
        # Submit to the (possibly remote) queue.
        submit_kwargs = dict(forwarded)
        if sticky_worker_id:
            submit_kwargs["sticky_worker_id"] = sticky_worker_id
        if isinstance(history_cid, str) and history_cid.strip():
            submit_kwargs["history_cid"] = history_cid.strip()
        task_id = submit_task(
            prompt=str(prompt or ""),
            model_name=str(model_name or "gpt2"),
            task_type="llm.generate",
            queue_path=queue_path,
            provider=provider_norm,
            **submit_kwargs,
        )
        last_task_id = task_id

        task = wait_task(task_id, queue_path=queue_path, timeout_s=float(timeout_s))
        if isinstance(task, dict) and str(task.get("status") or "") in {"completed", "failed"}:
            result = task.get("result")
            if isinstance(result, dict):
                wid = str(result.get("executor_worker_id") or "").strip()
                if sticky_key and wid:
                    _STICKY_SESSION_WORKER[sticky_key] = wid
            # Return text even if failed? Prefer raising with error.
            if str(task.get("status")) == "failed":
                err = str(task.get("error") or "") or str((result or {}).get("error") or "")
                raise LLMRouterError(err or "mesh llm.generate failed")
            if isinstance(result, dict) and "text" in result:
                text = str(result.get("text") or "")
                # Record successful turn for later failover.
                if isinstance(chat_session_id, str) and chat_session_id.strip():
                    try:
                        cid = _chat_history_append_turn(
                            chat_session_id=chat_session_id.strip(),
                            user_prompt=str(prompt or ""),
                            assistant_text=text,
                        )
                        if isinstance(cid, str) and cid.strip():
                            history_cid = cid.strip()
                    except Exception:
                        pass
                return text
            raise LLMRouterError("mesh llm.generate completed without generated text")

        # Timeout / not completed: cancel this queued work and retry.
        try:
            parsed = _decode_p2p_task_id(str(task_id))
            if parsed:
                import anyio

                from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue
                from ipfs_accelerate_py.p2p_tasks.client import cancel_task as cancel_task_p2p

                peer_id, inner_id = parsed
                remote_multiaddr = (
                    os.environ.get("ipfs_accelerate_py_TASK_P2P_REMOTE_MULTIADDR")
                    or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_MULTIADDR")
                    or ""
                ).strip()
                ann = _read_task_p2p_announce() if not remote_multiaddr else None
                if ann and not remote_multiaddr:
                    remote_multiaddr = str(ann.get("multiaddr") or "").strip()
                remote = RemoteQueue(peer_id=str(peer_id), multiaddr=str(remote_multiaddr))

                async def _run_cancel() -> None:
                    await cancel_task_p2p(remote=remote, task_id=str(inner_id), reason="route_timeout")

                anyio.run(_run_cancel, backend="trio")
            else:
                # Local cancellation.
                try:
                    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

                    TaskQueue(queue_path).cancel(task_id=str(task_id), reason="route_timeout")
                except Exception:
                    pass
        except Exception:
            pass

        # After the last retry, fall back to new session with history.
        if i == (attempts - 1):
            fallback_prompt = str(prompt or "")
            effective_history = str(history or "").strip() if isinstance(history, str) else ""
            effective_history_cid: str | None = None

            if not effective_history and isinstance(chat_session_id, str) and chat_session_id.strip():
                try:
                    cached_text, cached_cid = _chat_history_get(chat_session_id.strip())
                    if isinstance(cached_text, str) and cached_text.strip():
                        effective_history = cached_text.strip()
                    if isinstance(cached_cid, str) and cached_cid.strip():
                        effective_history_cid = cached_cid.strip()
                except Exception:
                    pass

            if effective_history:
                # Ensure the recovered history is persisted and content-addressed.
                try:
                    effective_history_cid = effective_history_cid or _store_chat_history_text(effective_history)
                except Exception:
                    pass
                fallback_prompt = (
                    "Continue this conversation, starting a fresh session if needed.\n\n"
                    "Conversation history:\n"
                    + effective_history
                    + "\n\nNext user message:\n"
                    + str(prompt or "")
                )

            # Clear resume/session flags and sticky pin.
            submit_kwargs2 = dict(forwarded)
            submit_kwargs2.pop("resume_session_id", None)
            submit_kwargs2.pop("continue_session", None)
            submit_kwargs2.pop("sticky_worker_id", None)

            # If the session-bound route timed out, resubmit with a fresh session
            # id on this machine so local workers can drain the queued task.
            if _truthy(os.environ.get("IPFS_ACCELERATE_PY_LLM_MESH_FAILOVER_REWRITE_SESSION_ID", "1")):
                failover_sid = str(
                    os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_FAILOVER_SESSION")
                    or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_SESSION")
                    or ""
                ).strip()
                if failover_sid:
                    submit_kwargs2["session_id"] = failover_sid
                else:
                    submit_kwargs2.pop("session_id", None)

            if isinstance(effective_history_cid, str) and effective_history_cid.strip():
                submit_kwargs2["history_cid"] = effective_history_cid.strip()

            task_id2 = submit_task(
                prompt=str(fallback_prompt),
                model_name=str(model_name or "gpt2"),
                task_type="llm.generate",
                queue_path=queue_path,
                provider=provider_norm,
                **submit_kwargs2,
            )
            task2 = wait_task(task_id2, queue_path=queue_path, timeout_s=float(timeout_s))
            if not isinstance(task2, dict):
                raise LLMRouterError("mesh llm.generate failed (no response)")
            result2 = task2.get("result") if isinstance(task2.get("result"), dict) else {}
            status2 = str(task2.get("status") or "").strip().lower()
            if status2 == "failed":
                err2 = str(task2.get("error") or "") or str((result2 or {}).get("error") or "")
                raise LLMRouterError(err2 or "mesh llm.generate failed")
            if status2 != "completed":
                raise LLMRouterError(f"mesh llm.generate did not complete after failover: {status2 or 'unknown'}")
            if isinstance(result2, dict) and "text" in result2:
                text2 = str(result2.get("text") or "")
                if isinstance(chat_session_id, str) and chat_session_id.strip():
                    try:
                        cid2 = _chat_history_append_turn(
                            chat_session_id=chat_session_id.strip(),
                            user_prompt=str(prompt or ""),
                            assistant_text=text2,
                        )
                        if isinstance(cid2, str) and cid2.strip():
                            history_cid = cid2.strip()
                    except Exception:
                        pass
                return text2
            raise LLMRouterError("mesh llm.generate completed without generated text")

    raise LLMRouterError("mesh llm.generate failed")


def get_remote_capabilities(*, timeout_s: float = 10.0, detail: bool = False) -> Dict[str, object]:
    """Get remote peer capabilities via libp2p.

    Uses:
    - ipfs_accelerate_py_TASK_P2P_REMOTE_MULTIADDR / IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_MULTIADDR
    - ipfs_accelerate_py_TASK_P2P_REMOTE_PEER_ID / IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_PEER_ID

    If multiaddr is not set, the client will attempt bootstrap+LAN mDNS discovery.
    """

    remote_peer_id = (
        os.environ.get("ipfs_accelerate_py_TASK_P2P_REMOTE_PEER_ID")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_PEER_ID")
        or ""
    ).strip()
    remote_multiaddr = (
        os.environ.get("ipfs_accelerate_py_TASK_P2P_REMOTE_MULTIADDR")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_MULTIADDR")
        or ""
    ).strip()

    try:
        import anyio

        from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue
        from ipfs_accelerate_py.p2p_tasks.client import get_capabilities as get_capabilities_p2p

        remote = RemoteQueue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)

        async def _run() -> Dict[str, object]:
            caps = await get_capabilities_p2p(remote=remote, timeout_s=float(timeout_s), detail=bool(detail))
            return caps if isinstance(caps, dict) else {}

        return anyio.run(_run, backend="trio")
    except Exception as exc:
        raise LLMRouterError(f"P2P get_remote_capabilities failed: {exc}") from exc


def call_remote_tool(
    *,
    tool_name: str,
    args: Optional[Dict[str, object]] = None,
    timeout_s: float = 30.0,
) -> Dict[str, object]:
    """Call a remote MCP tool via libp2p.

    Requires the remote peer to set `IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS=1`.
    """

    remote_peer_id = (
        os.environ.get("ipfs_accelerate_py_TASK_P2P_REMOTE_PEER_ID")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_PEER_ID")
        or ""
    ).strip()
    remote_multiaddr = (
        os.environ.get("ipfs_accelerate_py_TASK_P2P_REMOTE_MULTIADDR")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_MULTIADDR")
        or ""
    ).strip()

    try:
        import anyio

        from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue
        from ipfs_accelerate_py.p2p_tasks.client import call_tool as call_tool_p2p

        remote = RemoteQueue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
        safe_args: Dict[str, object] = args if isinstance(args, dict) else {}

        async def _run() -> Dict[str, object]:
            resp = await call_tool_p2p(remote=remote, tool_name=str(tool_name), args=safe_args, timeout_s=float(timeout_s))
            return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}

        return anyio.run(_run, backend="trio")
    except Exception as exc:
        raise LLMRouterError(f"P2P call_remote_tool failed: {exc}") from exc


def get_remote_cache_value(*, key: str, timeout_s: float = 10.0) -> Dict[str, object]:
    """Get a remote cache entry via libp2p.

    Requires the remote peer to set `IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_CACHE=1`.
    """

    remote_peer_id = (
        os.environ.get("ipfs_accelerate_py_TASK_P2P_REMOTE_PEER_ID")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_PEER_ID")
        or ""
    ).strip()
    remote_multiaddr = (
        os.environ.get("ipfs_accelerate_py_TASK_P2P_REMOTE_MULTIADDR")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_MULTIADDR")
        or ""
    ).strip()

    try:
        import anyio

        from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue
        from ipfs_accelerate_py.p2p_tasks.client import cache_get as cache_get_p2p

        remote = RemoteQueue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)

        async def _run() -> Dict[str, object]:
            resp = await cache_get_p2p(remote=remote, key=str(key), timeout_s=float(timeout_s))
            return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}

        return anyio.run(_run, backend="trio")
    except Exception as exc:
        raise LLMRouterError(f"P2P get_remote_cache_value failed: {exc}") from exc


def set_remote_cache_value(
    *,
    key: str,
    value: object,
    ttl_s: float | None = None,
    timeout_s: float = 10.0,
) -> Dict[str, object]:
    """Set a remote cache entry via libp2p.

    Requires the remote peer to set `IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_CACHE=1`.
    """

    remote_peer_id = (
        os.environ.get("ipfs_accelerate_py_TASK_P2P_REMOTE_PEER_ID")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_PEER_ID")
        or ""
    ).strip()
    remote_multiaddr = (
        os.environ.get("ipfs_accelerate_py_TASK_P2P_REMOTE_MULTIADDR")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_MULTIADDR")
        or ""
    ).strip()

    try:
        import anyio

        from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue
        from ipfs_accelerate_py.p2p_tasks.client import cache_set as cache_set_p2p

        remote = RemoteQueue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)

        async def _run() -> Dict[str, object]:
            resp = await cache_set_p2p(
                remote=remote,
                key=str(key),
                value=value,
                ttl_s=ttl_s,
                timeout_s=float(timeout_s),
            )
            return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}

        return anyio.run(_run, backend="trio")
    except Exception as exc:
        raise LLMRouterError(f"P2P set_remote_cache_value failed: {exc}") from exc


def _find_int_by_key(obj: object, key: str) -> Optional[int]:
    """Best-effort: find the first int-like value for a key anywhere in nested JSON."""

    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key:
                    if isinstance(v, bool):
                        return None
                    if isinstance(v, int):
                        return v
                    if isinstance(v, str) and v.strip().isdigit():
                        return int(v.strip())
                found = _find_int_by_key(v, key)
                if isinstance(found, int):
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _find_int_by_key(item, key)
                if isinstance(found, int):
                    return found
    except Exception:
        return None
    return None


def _extract_resets_in_seconds_from_codex_jsonl(text: str) -> Optional[int]:
    """Parse Codex --json output (JSONL) for a resets_in_seconds-like field."""

    if not isinstance(text, str) or not text.strip():
        return None
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        for candidate_key in (
            "resets_in_seconds",
            "reset_in_seconds",
            "retry_after_seconds",
            "retry_after",
        ):
            found = _find_int_by_key(obj, candidate_key)
            if isinstance(found, int) and found > 0:
                return found
    return None


def _extract_first_error_message_from_codex_jsonl(text: str) -> Optional[str]:
    """Best-effort: extract the first error message from Codex --json (JSONL) stdout."""

    if not isinstance(text, str) or not text.strip():
        return None
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "error":
            msg = obj.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
            err = obj.get("error")
            if isinstance(err, dict):
                msg2 = err.get("message")
                if isinstance(msg2, str) and msg2.strip():
                    return msg2.strip()
    return None


def _is_codex_quota_exceeded_message(msg: str) -> bool:
    """Detect errors that indicate a billing/quota hard-stop (waiting won't help)."""

    if not isinstance(msg, str) or not msg.strip():
        return False
    low = msg.lower()
    quota_markers = (
        "insufficient_quota",
        "exceeded your current quota",
        "quota has been exceeded",
        "billing",
        "hard limit",
        "billing limit",
        "check your plan and billing",
        "add a payment method",
        "your account is not active",
    )
    return any(m in low for m in quota_markers) and ("usage limit" not in low)


def _classify_codex_error_kind(*, stdout: str, stderr: str) -> Optional[str]:
    """Classify Codex failures into coarse kinds to guide retry vs fail-fast."""

    provider_msg = _extract_first_error_message_from_codex_jsonl(stdout or "")
    if provider_msg and _is_codex_quota_exceeded_message(provider_msg):
        return "quota_exceeded"

    combined = "\n".join([p for p in [provider_msg, stdout, stderr] if isinstance(p, str) and p.strip()])
    if _is_codex_quota_exceeded_message(combined):
        return "quota_exceeded"

    low = combined.lower() if isinstance(combined, str) else ""
    if "usage_limit" in low or "usage limit" in low:
        return "usage_limit"
    return None


def _extract_last_agent_message_from_codex_jsonl(text: str) -> Optional[str]:
    """Extract the most recent agent message from Codex --json (JSONL) stdout."""

    if not isinstance(text, str) or not text.strip():
        return None

    def _extract_text_from_message_like(obj: object) -> Optional[str]:
        if not isinstance(obj, dict):
            return None

        obj_type = obj.get("type")

        if obj_type in ("agent_message", "assistant_message"):
            txt = obj.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt

        if obj_type == "message" and obj.get("role") == "assistant":
            content = obj.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                for chunk in content:
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("type") in ("output_text", "text"):
                        chunk_text = chunk.get("text")
                        if isinstance(chunk_text, str) and chunk_text.strip():
                            parts.append(chunk_text)
                joined = "".join(parts).strip()
                return joined if joined else None

            txt = obj.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt

        return None

    last: Optional[str] = None
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        if isinstance(obj, dict) and obj.get("type") == "item.completed":
            item = obj.get("item")
            extracted = _extract_text_from_message_like(item)
            if isinstance(extracted, str) and extracted.strip():
                last = extracted
            continue

        if isinstance(obj, dict):
            extracted = _extract_text_from_message_like(obj.get("item"))
            if not extracted:
                extracted = _extract_text_from_message_like(obj.get("message"))
            if not extracted:
                extracted = _extract_text_from_message_like(obj)
            if isinstance(extracted, str) and extracted.strip():
                last = extracted

    return last.strip() if isinstance(last, str) and last.strip() else None


def get_accelerate_manager(
    *,
    deps: Optional[RouterDeps] = None,
    purpose: str = "llm",
    enable_distributed: bool = True,
    resources: Optional[Dict[str, object]] = None,
    ipfs_gateway: Optional[str] = None,
) -> object | None:
    """Return a cached AccelerateManager via RouterDeps.

    This is the preferred access path for accelerate integration from LLM-related
    call sites. It avoids importing `accelerate_integration` in those modules.
    """

    resolved = deps or get_default_router_deps()
    try:
        return resolved.get_accelerate_manager(
            purpose=purpose,
            enable_distributed=enable_distributed,
            resources=resources,
            ipfs_gateway=ipfs_gateway,
        )
    except Exception:
        return None


def get_accelerate_status() -> dict:
    """Best-effort accelerate status without forcing heavy imports.

    Note: This intentionally avoids importing `accelerate_integration` (or
    `ipfs_accelerate_py`) because those imports can trigger heavyweight optional
    initialization.
    """

    env_value = os.environ.get("IPFS_ACCELERATE_ENABLED", "1").lower()
    env_disabled = env_value in {"0", "false", "no", "disabled"}
    if env_disabled:
        return {"available": False, "enabled": False, "env_disabled": True, "env_var": env_value}

    try:
        import importlib.util

        backend_available = importlib.util.find_spec("ipfs_accelerate_py") is not None
    except Exception:
        backend_available = False

    return {"available": backend_available, "enabled": True, "env_disabled": False, "env_var": env_value}


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
    value = (
        os.environ.get("ipfs_accelerate_py_ROUTER_CACHE")
        or os.environ.get("IPFS_ACCELERATE_PY_ROUTER_CACHE")
        or os.environ.get("IPFS_DATASETS_PY_ROUTER_CACHE")
        or "1"
    )
    return value.strip() != "0"


def _response_cache_enabled() -> bool:
    # Default to enabled in benchmark contexts (determinism + speed), off otherwise.
    value = (
        os.environ.get("ipfs_accelerate_py_ROUTER_RESPONSE_CACHE")
        or os.environ.get("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE")
        or os.environ.get("IPFS_DATASETS_PY_ROUTER_RESPONSE_CACHE")
    )
    if value is None:
        return _truthy(
            os.environ.get("ipfs_accelerate_py_BENCHMARK")
            or os.environ.get("IPFS_ACCELERATE_PY_BENCHMARK")
            or os.environ.get("IPFS_DATASETS_PY_BENCHMARK")
        )
    return str(value).strip() != "0"


def _response_cache_key_strategy() -> str:
    """Return the response-cache key strategy.

    - "sha256" (default): compact deterministic string key
    - "cid": content-addressed CID (sha2-256, CIDv1) for the request payload
    """

    return (
        os.environ.get("ipfs_accelerate_py_ROUTER_CACHE_KEY")
        or os.environ.get("IPFS_ACCELERATE_PY_ROUTER_CACHE_KEY")
        or os.environ.get("IPFS_DATASETS_PY_ROUTER_CACHE_KEY")
        or "sha256"
    ).strip().lower() or "sha256"


def _response_cache_cid_base() -> str:
    return (
        os.environ.get("ipfs_accelerate_py_ROUTER_CACHE_CID_BASE")
        or os.environ.get("IPFS_ACCELERATE_PY_ROUTER_CACHE_CID_BASE")
        or os.environ.get("IPFS_DATASETS_PY_ROUTER_CACHE_CID_BASE")
        or "base32"
    ).strip() or "base32"


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

    pk = _canonicalize_provider(provider_key or "auto")
    if pk == "openrouter":
        return (
            _coalesce_env(
                "ipfs_accelerate_py_OPENROUTER_MODEL",
                "IPFS_ACCELERATE_PY_OPENROUTER_MODEL",
                "IPFS_DATASETS_PY_OPENROUTER_MODEL",
            )
            or _generic_llm_model_env()
            or "openai/gpt-4o-mini"
        ).strip()
    if pk in {"codex", "codex_cli"}:
        return (
            _coalesce_env(
                "ipfs_accelerate_py_CODEX_CLI_MODEL",
                "IPFS_ACCELERATE_PY_CODEX_CLI_MODEL",
                "IPFS_DATASETS_PY_CODEX_CLI_MODEL",
                "ipfs_accelerate_py_CODEX_MODEL",
                "IPFS_ACCELERATE_PY_CODEX_MODEL",
                "IPFS_DATASETS_PY_CODEX_MODEL",
            )
            or "chatgpt-5.6-terra"
        ).strip()
    if pk == "copilot_sdk":
        return _coalesce_env(
            "ipfs_accelerate_py_COPILOT_SDK_MODEL",
            "IPFS_ACCELERATE_PY_COPILOT_SDK_MODEL",
            "IPFS_DATASETS_PY_COPILOT_SDK_MODEL",
        )
    if pk == "hf_inference_api":
        return (
            _coalesce_env(
                "IPFS_ACCELERATE_PY_HF_INFERENCE_MODEL",
                "IPFS_DATASETS_PY_HF_INFERENCE_MODEL",
            )
            or _generic_llm_model_env()
            or "gpt2"
        ).strip()
    if pk in _GROK_CLI_PROVIDER_ALIASES or (
        pk == "grok" and _cli_available(_grok_cli_command())
    ):
        return (
            _coalesce_env(
                "ipfs_accelerate_py_GROK_CLI_MODEL",
                "IPFS_ACCELERATE_PY_GROK_CLI_MODEL",
                "IPFS_DATASETS_PY_GROK_CLI_MODEL",
                "GROK_CLI_MODEL",
                "IPFS_ACCELERATE_AGENT_GROK_MODEL",
            )
            or "grok-4.5"
        ).strip()
    if pk == "grok" or pk in _XAI_API_PROVIDER_ALIASES:
        return (
            _coalesce_env(
                "ipfs_accelerate_py_XAI_MODEL",
                "IPFS_ACCELERATE_PY_XAI_MODEL",
                "IPFS_DATASETS_PY_XAI_MODEL",
            )
            or _generic_llm_model_env()
            or "grok-4.5"
        ).strip()
    if pk in {"meta_ai", "meta-ai", "meta_llama", "meta", "meta_spark", "spark"}:
        return normalize_meta_model_name(
            _coalesce_env(
                "ipfs_accelerate_py_META_AI_MODEL",
                "IPFS_ACCELERATE_PY_META_AI_MODEL",
                "IPFS_DATASETS_PY_META_AI_MODEL",
            )
            or _generic_llm_model_env()
            or META_MODEL_API_DEFAULT_MODEL
        )
    if pk in {"hf", "huggingface", "local_hf"}:
        return (_generic_llm_model_env() or "gpt2").strip()

    # Provider unknown/auto: include the most common default.
    return _generic_llm_model_env()


def _response_cache_key(*, provider: Optional[str], model_name: Optional[str], prompt: str, kwargs: Dict[str, object]) -> str:
    provider_key = (provider or "auto").strip().lower()
    model_key = _effective_model_key(provider_key=provider_key, model_name=model_name, kwargs=kwargs)

    strategy = _response_cache_key_strategy()
    if strategy == "cid":
        from .utils.cid_utils import cid_for_obj

        payload = {
            "type": "llm_response",
            "provider": provider_key,
            "model": model_key,
            "prompt": prompt or "",
            "kwargs": kwargs or {},
        }
        cid = cid_for_obj(payload, base=_response_cache_cid_base())
        return f"llm_response_cid::{cid}"

    prompt_digest = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()[:16]
    kw_digest = _stable_kwargs_digest(kwargs)
    return f"llm_response::{provider_key}::{model_key}::{prompt_digest}::{kw_digest}"


@runtime_checkable
class LLMProvider(Protocol):
    def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str: ...


@runtime_checkable
class BatchLLMProvider(Protocol):
    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        model_name: Optional[str] = None,
        **kwargs: object,
    ) -> Sequence[str]: ...


@runtime_checkable
class TextBatchLLMProvider(Protocol):
    def generate_text_batch(
        self,
        prompts: Sequence[str],
        *,
        model_name: Optional[str] = None,
        **kwargs: object,
    ) -> Sequence[str]: ...


@runtime_checkable
class NativeMultimodalProvider(Protocol):
    def generate_multimodal(
        self,
        prompt: str,
        *,
        model_name: Optional[str] = None,
        image_paths: Sequence[str] | None = None,
        image_urls: Sequence[str] | None = None,
        system_prompt: Optional[str] = None,
        additional_text_blocks: Sequence[str] | None = None,
        messages: Sequence[dict] | None = None,
        **kwargs: object,
    ) -> str: ...


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class OpenAICompatTopLogProb:
    token: str
    logprob: float


@dataclass(frozen=True, slots=True)
class OpenAICompatLogProbsContentItem:
    top_logprobs: list[OpenAICompatTopLogProb]


@dataclass(frozen=True, slots=True)
class OpenAICompatLogProbs:
    content: list[OpenAICompatLogProbsContentItem]


@dataclass(frozen=True, slots=True)
class OpenAICompatMessage:
    content: str


@dataclass(frozen=True, slots=True)
class OpenAICompatChoice:
    message: OpenAICompatMessage
    logprobs: OpenAICompatLogProbs


@dataclass(frozen=True, slots=True)
class OpenAICompatResponse:
    choices: list[OpenAICompatChoice]


@runtime_checkable
class OpenAIChatCompletionsProvider(Protocol):
    def chat_completions(
        self,
        messages: Sequence[ChatMessage],
        *,
        model_name: Optional[str] = None,
        **kwargs: object,
    ) -> dict: ...


ProviderFactory = Callable[[], LLMProvider]


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    factory: ProviderFactory
    descriptor: Optional[ProviderDescriptor] = None
    models: Tuple[ModelDescriptor, ...] = ()


_PROVIDER_REGISTRY: Dict[str, ProviderInfo] = {}
_PROVIDER_REGISTRY_LOCK = threading.RLock()


def _registered_llm_provider_descriptor(
    name: str,
    descriptor: ProviderDescriptor | Mapping[str, object] | None,
) -> ProviderDescriptor:
    if descriptor is None:
        return ProviderDescriptor(
            name=name,
            description="Dynamically registered LLM provider.",
            capabilities=(_llm_capability(),),
            lifecycle=LifecycleState.DECLARED,
            state=OperationalState(
                known=True,
                configured=True,
                authorized=None,
                reachable=None,
                healthy=None,
                routable=None,
            ),
            provenance=(Provenance(source="llm_router.registry"),),
            labels={
                "access_requirement": "unknown",
                "batching": "supported",
                "device": "unknown",
                "locality": "unknown",
                "streaming": "unknown",
                "tools": "unknown",
            },
        )
    if isinstance(descriptor, ProviderDescriptor):
        resolved = descriptor
    elif isinstance(descriptor, Mapping):
        values = dict(descriptor)
        values.setdefault("name", name)
        resolved = ProviderDescriptor(**values)
    else:
        raise TypeError("descriptor must be a ProviderDescriptor, mapping, or None")
    if resolved.name != name:
        raise ValueError("Provider descriptor name must match the registered name")
    return resolved


def _registered_llm_model_descriptors(
    provider: ProviderDescriptor,
    models: Sequence[ModelDescriptor | Mapping[str, object]],
) -> Tuple[ModelDescriptor, ...]:
    if isinstance(models, (str, bytes, Mapping)):
        raise TypeError("models must be a sequence of model descriptors")
    output: list[ModelDescriptor] = []
    for model in models:
        if isinstance(model, ModelDescriptor):
            resolved = model
        elif isinstance(model, Mapping):
            values = dict(model)
            values.setdefault("provider_id", provider.provider_id)
            resolved = ModelDescriptor(**values)
        else:
            raise TypeError("models must contain ModelDescriptor records or mappings")
        if resolved.provider_id != provider.provider_id:
            raise ValueError("Model descriptor provider_id does not match provider")
        output.append(resolved)
    identities = [model.model_id for model in output]
    if len(identities) != len(set(identities)):
        raise ValueError("models contain duplicate identities")
    return tuple(sorted(output, key=lambda model: (model.name, model.model_id or "")))


def register_llm_provider(
    name: str,
    factory: ProviderFactory,
    *,
    descriptor: ProviderDescriptor | Mapping[str, object] | None = None,
    models: Sequence[ModelDescriptor | Mapping[str, object]] = (),
) -> None:
    """Register a provider and optional side-effect-free catalog metadata.

    Discovery retains ``factory`` without calling it.  When metadata is
    omitted, provider-specific facts stay explicitly unknown.
    """

    if not name or not name.strip():
        raise ValueError("Provider name must be non-empty")
    if not callable(factory):
        raise TypeError("Provider factory must be callable")
    normalized = name.strip().lower()
    provider_descriptor = _registered_llm_provider_descriptor(
        normalized,
        descriptor,
    )
    model_descriptors = _registered_llm_model_descriptors(
        provider_descriptor,
        models,
    )
    with _PROVIDER_REGISTRY_LOCK:
        _PROVIDER_REGISTRY[normalized] = ProviderInfo(
            name=normalized,
            factory=factory,
            descriptor=provider_descriptor,
            models=model_descriptors,
        )


def _coalesce_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _canonicalize_provider(name: Optional[str]) -> str:
    key = str(name or "").strip().lower()
    canonical = _PROVIDER_ALIASES.get(key, key)
    if canonical != key:
        return canonical
    # Registered descriptor aliases participate in invocation as well as
    # discovery. This lookup is metadata-only and never calls the factory.
    with _PROVIDER_REGISTRY_LOCK:
        matches = sorted(
            info.name
            for info in _PROVIDER_REGISTRY.values()
            if info.descriptor is not None and key in info.descriptor.aliases
        )
    return matches[0] if len(matches) == 1 else key


def _generic_llm_model_env() -> str:
    """Return the generic model override unless it actually names a provider."""

    value = _coalesce_env(
        "ipfs_accelerate_py_LLM_MODEL",
        "IPFS_ACCELERATE_PY_LLM_MODEL",
        "IPFS_DATASETS_PY_LLM_MODEL",
    )
    provider_names = {
        *_PROVIDER_ALIASES,
        *_PROVIDER_ALIASES.values(),
        *_GROK_CLI_PROVIDER_ALIASES,
        *_XAI_API_PROVIDER_ALIASES,
        "accelerate",
        "ipfs_accelerate_py",
        "mock",
        "dry_run",
        "dry-run",
        "openrouter",
        "openai",
        "copilot_cli",
        "copilot_sdk",
        "gemini_cli",
        "gemini_py",
        "claude",
        "claude_code",
        "claude_py",
        "mistral_vibe",
        "vibe",
        "xai",
        "meta_ai",
        "hf",
        "huggingface",
        "local_hf",
        "hf_inference_api",
        "p2p_task_queue",
        "llama_cpp",
        "llama_cpp_native",
    }
    return "" if value.strip().lower() in provider_names else value


def _resolve_hf_api_token() -> str:
    token = _coalesce_env(
        "IPFS_ACCELERATE_PY_HF_API_TOKEN",
        "ipfs_accelerate_py_HF_API_TOKEN",
        "IPFS_DATASETS_PY_HF_API_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HUGGINGFACE_API_TOKEN",
        "HF_TOKEN",
    )
    if token:
        return token
    try:
        hub = importlib.import_module("huggingface_hub")
        getter = getattr(hub, "get_token", None)
        resolved = getter() if callable(getter) else ""
        return str(resolved or "").strip()
    except Exception:
        return ""


def _resolve_openai_api_key() -> str:
    return _coalesce_env(
        "OPENAI_API_KEY",
        "OPENAI_KEY",
        "OPENAI_TOKEN",
        "IPFS_ACCELERATE_PY_OPENAI_API_KEY",
        "ipfs_accelerate_py_OPENAI_API_KEY",
    )


def _hf_token_fingerprint() -> str:
    token = _resolve_hf_api_token()
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12] if token else ""


def _resolve_hf_bill_to(*, kwargs: Optional[dict[str, object]] = None) -> str:
    if kwargs:
        for key in ("hf_bill_to", "bill_to", "organization", "org"):
            value = kwargs.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return _coalesce_env(
        "OPENROUTER_HF_BILL_TO",
        "IPFS_ACCELERATE_PY_HF_BILL_TO",
        "IPFS_DATASETS_PY_HF_BILL_TO",
        "HUGGINGFACE_BILL_TO",
        "HF_BILL_TO",
        "HF_ORGANIZATION",
        "HUGGINGFACE_ORG",
    )


def _resolve_hf_provider(*, kwargs: Optional[dict[str, object]] = None) -> str:
    if kwargs:
        for key in ("hf_provider", "hf_inference_provider", "huggingface_provider"):
            value = kwargs.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return _coalesce_env(
        "IPFS_ACCELERATE_PY_HF_PROVIDER",
        "IPFS_ACCELERATE_PY_HF_INFERENCE_PROVIDER",
        "IPFS_DATASETS_PY_HF_PROVIDER",
        "IPFS_DATASETS_PY_HF_INFERENCE_PROVIDER",
    )


def _build_hf_inference_client_kwargs(
    *,
    provider: str,
    token: str,
    timeout: float,
    bill_to: str = "",
) -> dict[str, object]:
    values: dict[str, object] = {
        "provider": provider,
        "token": token,
        "timeout": timeout,
    }
    if bill_to.strip():
        values["bill_to"] = bill_to.strip()
    return values


def _hf_model_uses_provider_policy(model_name: Optional[str]) -> bool:
    return ":" in str(model_name or "").strip()


def _hf_use_chat_completions(
    *,
    model_name: Optional[str],
    kwargs: dict[str, object],
) -> bool:
    raw = kwargs.get("hf_use_chat_completions")
    if raw is None:
        raw = _coalesce_env(
            "IPFS_ACCELERATE_PY_HF_USE_CHAT_COMPLETIONS",
            "IPFS_DATASETS_PY_HF_USE_CHAT_COMPLETIONS",
        )
    if raw not in (None, ""):
        return _truthy(str(raw))
    provider_name = _resolve_hf_provider(kwargs=kwargs).strip().lower()
    return bool(
        (provider_name and provider_name != "hf-inference")
        or _hf_model_uses_provider_policy(model_name)
    )


def _hf_to_jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_hf_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _hf_to_jsonable(item) for key, item in value.items()}
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _hf_to_jsonable(method())
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        try:
            return _hf_to_jsonable(vars(value))
        except Exception:
            pass
    return value


def _is_hf_inference_provider_name(name: Optional[str]) -> bool:
    return _canonicalize_provider(name) == "hf_inference_api"


def _is_hf_model_compatibility_error(exc: BaseException) -> bool:
    message = str(exc or "").lower()
    return (
        any(
            token in message
            for token in (
                "http 404",
                "not found",
                "model",
                "pipeline",
                "task",
                "unsupported",
                "does not support",
            )
        )
        and "http 402" not in message
    )


def _hf_llm_default_fallback_models() -> list[str]:
    return [
        "HuggingFaceH4/zephyr-7b-beta",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ]


def _is_probably_text_generation_model(model_id: str) -> bool:
    lower = str(model_id or "").strip().lower()
    if not lower:
        return False
    return not any(
        token in lower for token in ("bart", "pegasus", "t5", "mbart", "summar")
    )


def _hf_live_model_manager_candidate_models() -> list[str]:
    try:
        from ipfs_datasets_py.utils import model_manager

        records = model_manager.list_hf_inference_models(model_kind="llm")
    except Exception:
        return []

    def _score(record: dict[str, object]) -> tuple[int, int, str]:
        model_id = str(record.get("model_id") or "").strip()
        lower = model_id.lower()
        pipeline_tag = str(record.get("pipeline_tag") or "").strip().lower()
        if not model_id or not _is_probably_text_generation_model(model_id):
            return (-100, -100, model_id)
        score = 40 if pipeline_tag == "text-generation" else 20
        if pipeline_tag == "summarization":
            score -= 60
        if any(
            token in lower
            for token in (
                "instruct",
                "chat",
                "assistant",
                "gpt",
                "deepseek",
                "qwen",
                "mistral",
                "llama",
                "zephyr",
                "oss",
                "router",
            )
        ):
            score += 50
        return (score, len(model_id), model_id)

    ordered: list[str] = []
    for record in sorted(records, key=_score, reverse=True):
        model_id = str(record.get("model_id") or "").strip()
        if (
            model_id
            and model_id not in ordered
            and _is_probably_text_generation_model(model_id)
        ):
            ordered.append(model_id)
    return ordered


def _hf_dynamic_model_discovery_enabled(*, kwargs: dict[str, object]) -> bool:
    raw = kwargs.get("hf_dynamic_model_discovery")
    if raw is None:
        raw = _coalesce_env(
            "IPFS_ACCELERATE_PY_HF_DYNAMIC_MODEL_DISCOVERY",
            "IPFS_DATASETS_PY_HF_DYNAMIC_MODEL_DISCOVERY",
        ) or "1"
    return _truthy(str(raw))


def _hf_llm_discovery_limit(*, kwargs: dict[str, object]) -> int:
    raw = kwargs.get("hf_llm_discovery_limit")
    if raw is None:
        raw = _coalesce_env(
            "IPFS_ACCELERATE_PY_HF_LLM_DISCOVERY_LIMIT",
            "IPFS_DATASETS_PY_HF_LLM_DISCOVERY_LIMIT",
        ) or "20"
    try:
        return max(1, int(raw))
    except Exception:
        return 20


def _hf_llm_discovery_tags(*, kwargs: dict[str, object]) -> list[str]:
    raw = kwargs.get("hf_llm_discovery_tags")
    if raw is None:
        raw = _coalesce_env(
            "IPFS_ACCELERATE_PY_HF_LLM_DISCOVERY_TAGS",
            "IPFS_DATASETS_PY_HF_LLM_DISCOVERY_TAGS",
        ) or "text-generation,text2text-generation,summarization"
    return [item.strip() for item in str(raw).split(",") if item.strip()]


@lru_cache(maxsize=32)
def _discover_hf_models_for_pipeline(
    *,
    pipeline_tag: str,
    limit: int,
) -> tuple[str, ...]:
    try:
        hub = importlib.import_module("huggingface_hub")
        api_cls = getattr(hub, "HfApi", None)
        if api_cls is None:
            return ()
        models = api_cls().list_models(
            inference_provider="hf-inference",
            pipeline_tag=pipeline_tag,
            limit=max(1, int(limit)),
            token=_resolve_hf_api_token() or None,
        )
        output: list[str] = []
        for item in models:
            model_id = str(getattr(item, "id", "") or "").strip()
            if model_id and model_id not in output:
                output.append(model_id)
        return tuple(output)
    except Exception:
        return ()


def _hf_llm_fallback_models(*, kwargs: dict[str, object]) -> list[str]:
    raw = kwargs.get("hf_model_fallbacks")
    if raw is None:
        raw = _coalesce_env(
            "IPFS_ACCELERATE_PY_HF_LLM_FALLBACK_MODELS",
            "IPFS_DATASETS_PY_HF_LLM_FALLBACK_MODELS",
        )
    if str(raw or "").strip():
        return [item.strip() for item in str(raw).split(",") if item.strip()]

    models: list[str] = []
    for model_id in _hf_live_model_manager_candidate_models():
        if model_id not in models:
            models.append(model_id)
    if _hf_dynamic_model_discovery_enabled(kwargs=kwargs):
        for tag in _hf_llm_discovery_tags(kwargs=kwargs):
            for model_id in _discover_hf_models_for_pipeline(
                pipeline_tag=tag,
                limit=_hf_llm_discovery_limit(kwargs=kwargs),
            ):
                if model_id not in models:
                    models.append(model_id)
    for model_id in _hf_llm_default_fallback_models():
        if model_id not in models:
            models.append(model_id)
    return models


def _hf_arch_router_enabled(*, kwargs: dict[str, object]) -> bool:
    raw = kwargs.get("hf_use_arch_router")
    if raw is None:
        raw = _coalesce_env(
            "IPFS_ACCELERATE_PY_HF_USE_ARCH_ROUTER",
            "IPFS_DATASETS_PY_HF_USE_ARCH_ROUTER",
        ) or "1"
    return _truthy(str(raw))


def _hf_arch_router_model(*, kwargs: dict[str, object]) -> str:
    raw = kwargs.get("hf_arch_router_model")
    if raw is None:
        raw = _coalesce_env(
            "IPFS_ACCELERATE_PY_HF_ARCH_ROUTER_MODEL",
            "IPFS_DATASETS_PY_HF_ARCH_ROUTER_MODEL",
        )
    return str(raw or _HF_ARCH_ROUTER_MODEL_ID).strip() or _HF_ARCH_ROUTER_MODEL_ID


def _hf_arch_router_timeout(
    *,
    kwargs: dict[str, object],
    request_timeout: float,
) -> float:
    raw = kwargs.get("hf_arch_router_timeout")
    if raw is None:
        raw = _coalesce_env(
            "IPFS_ACCELERATE_PY_HF_ARCH_ROUTER_TIMEOUT",
            "IPFS_DATASETS_PY_HF_ARCH_ROUTER_TIMEOUT",
        )
    if raw not in (None, ""):
        try:
            return max(1.0, float(raw))
        except Exception:
            pass
    return max(1.0, min(float(request_timeout), 8.0))


def _hf_arch_router_candidate_models(*, kwargs: dict[str, object]) -> list[str]:
    raw = kwargs.get("hf_route_candidate_models")
    if raw is None:
        raw = _coalesce_env(
            "IPFS_ACCELERATE_PY_HF_ROUTE_CANDIDATE_MODELS",
            "IPFS_DATASETS_PY_HF_ROUTE_CANDIDATE_MODELS",
        )

    models: list[str] = []
    values = raw if isinstance(raw, (list, tuple, set)) else str(raw or "").split(",")
    for item in values:
        model_id = str(item or "").strip()
        if model_id and model_id not in models:
            models.append(model_id)
    if not models:
        models.extend(_hf_live_model_manager_candidate_models())
    if not models and _hf_dynamic_model_discovery_enabled(kwargs=kwargs):
        models.extend(
            _discover_hf_models_for_pipeline(
                pipeline_tag="text-generation",
                limit=_hf_llm_discovery_limit(kwargs=kwargs),
            )
        )
    if not models:
        models.extend(_hf_llm_default_fallback_models())
    router_model = _hf_arch_router_model(kwargs=kwargs)
    return [model_id for model_id in models if model_id and model_id != router_model]


def _describe_hf_route_candidate(model_id: str) -> str:
    lower = model_id.lower()
    if "llama" in lower and "instruct" in lower:
        return "General instruction-following, reasoning, coding, and multi-step assistant tasks."
    if "bart" in lower or "pegasus" in lower or "xsum" in lower:
        return "Summarization and concise rewriting of long passages or reports."
    if "t5" in lower:
        return "Instruction following, extraction, classification, and short structured responses."
    return "General Hugging Face inference model for text generation tasks."


def _build_hf_arch_router_prompt(
    *,
    route_config: list[dict[str, str]],
    prompt: str,
) -> str:
    return (
        "Choose the best route for the user request. Return JSON as "
        '{"route": "route_name"} or {"route": "other"}.\n'
        f"<routes>{json.dumps(route_config, ensure_ascii=False)}</routes>\n"
        f"<conversation>{json.dumps([{'role': 'user', 'content': prompt}], ensure_ascii=False)}</conversation>"
    )


def _parse_hf_arch_router_response(
    response_text: str,
    *,
    candidate_models: list[str],
) -> Optional[str]:
    text = str(response_text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            route = str(data.get("route") or "").strip()
            if route in candidate_models:
                return route
            if route.lower() == "other":
                return None
    except Exception:
        pass
    return next((model_id for model_id in candidate_models if model_id in text), None)


def _route_hf_model_with_arch_router(
    *,
    prompt: str,
    kwargs: dict[str, object],
    request_timeout: float,
    generate_fn: Callable[[str, str, float], str],
) -> Optional[str]:
    if not _hf_arch_router_enabled(kwargs=kwargs):
        return None
    candidate_models = _hf_arch_router_candidate_models(kwargs=kwargs)
    if not candidate_models:
        return None
    route_config = [
        {"name": model_id, "description": _describe_hf_route_candidate(model_id)}
        for model_id in candidate_models
    ]
    try:
        routed_text = generate_fn(
            _build_hf_arch_router_prompt(route_config=route_config, prompt=prompt),
            _hf_arch_router_model(kwargs=kwargs),
            _hf_arch_router_timeout(
                kwargs=kwargs,
                request_timeout=request_timeout,
            ),
        )
    except Exception:
        return None
    return _parse_hf_arch_router_response(
        routed_text,
        candidate_models=candidate_models,
    )


def _default_hf_inference_model(*, kwargs: dict[str, object]) -> str:
    explicit = _coalesce_env(
        "IPFS_ACCELERATE_PY_HF_INFERENCE_MODEL",
        "IPFS_DATASETS_PY_HF_INFERENCE_MODEL",
    )
    if explicit:
        return explicit
    generic_model = _generic_llm_model_env()
    if generic_model:
        return generic_model
    candidates = _hf_arch_router_candidate_models(kwargs=kwargs)
    return candidates[0] if candidates else "gpt2"


def _ordered_hf_generation_models(
    *,
    kwargs: dict[str, object],
    selected_model: str,
    routed_model: Optional[str],
) -> list[str]:
    ordered: list[str] = []
    for model_id in [
        selected_model,
        routed_model,
        *_hf_arch_router_candidate_models(kwargs=kwargs),
        *_hf_llm_fallback_models(kwargs=kwargs),
    ]:
        value = str(model_id or "").strip()
        if value and value not in ordered:
            ordered.append(value)
    return ordered


def _extract_hf_response_text(data: object) -> Optional[str]:
    if isinstance(data, str):
        return data.strip() or None
    if isinstance(data, list) and data:
        return _extract_hf_response_text(data[0])
    if isinstance(data, dict):
        for key in ("generated_text", "summary_text", "translation_text", "text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


_LEADING_MARKER_RE = re.compile(r"^[\s\u2022\u25CF\u25E6\u25AA\u25AB\u2219\u00B7\*\-]+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_copilot_output(text: str) -> str:
    cleaned = (text or "").strip()
    # Do not strip patch markers; they are semantically meaningful.
    if cleaned.lstrip().startswith("*** Begin Patch"):
        return cleaned.strip()
    cleaned = _LEADING_MARKER_RE.sub("", cleaned).strip()
    cleaned = unescape(cleaned)
    if "<" in cleaned and ">" in cleaned:
        cleaned = _HTML_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def _clean_codex_output(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.lstrip().startswith("*** Begin Patch"):
        return cleaned.strip()
    cleaned = _LEADING_MARKER_RE.sub("", cleaned).strip()
    cleaned = unescape(cleaned)
    if "<" in cleaned and ">" in cleaned:
        cleaned = _HTML_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def _clean_claude_output(text: str) -> str:
    return _clean_codex_output(text)


def _clean_gemini_output(text: str) -> str:
    return _clean_codex_output(text)


def _clean_mistral_vibe_output(text: str) -> str:
    return _clean_codex_output(text)


def _clean_grok_cli_output(text: str) -> str:
    return _clean_codex_output(text)


def _grok_cli_command() -> str:
    return (
        _coalesce_env(
            "ipfs_accelerate_py_GROK_CLI_CMD",
            "IPFS_ACCELERATE_PY_GROK_CLI_CMD",
            "IPFS_DATASETS_PY_GROK_CLI_CMD",
            "GROK_CLI_CMD",
        )
        or "grok"
    )


def _grok_cli_auth_path() -> Path:
    configured_home = os.getenv("GROK_HOME", "").strip()
    grok_home = Path(configured_home).expanduser() if configured_home else Path.home() / ".grok"
    return grok_home / "auth.json"


def _grok_cli_auth_available() -> bool:
    """Return whether headless Grok CLI authentication is configured."""

    if _coalesce_env(
        "XAI_API_KEY",
        "ipfs_accelerate_py_XAI_API_KEY",
        "IPFS_ACCELERATE_PY_XAI_API_KEY",
        "IPFS_DATASETS_PY_XAI_API_KEY",
        "GROK_AUTH_PROVIDER_COMMAND",
    ):
        return True
    try:
        auth_path = _grok_cli_auth_path()
        return auth_path.is_file() and auth_path.stat().st_size > 0
    except OSError:
        return False


def _grok_cli_auth_fingerprint() -> tuple[str, int, int]:
    """Return a non-secret cache fingerprint for the CLI OAuth credential."""

    auth_path = _grok_cli_auth_path()
    try:
        stat_result = auth_path.stat()
        return (str(auth_path), int(stat_result.st_mtime_ns), int(stat_result.st_size))
    except OSError:
        return (str(auth_path), 0, 0)


def _grok_cli_json_payload(text: str) -> Optional[dict[str, object]]:
    """Extract the final Grok headless JSON object from stdout."""

    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [raw, *reversed([line.strip() for line in raw.splitlines() if line.strip()])]
    for candidate in candidates:
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _grok_cli_error(stdout: str, stderr: str) -> LLMRouterError:
    payload = _grok_cli_json_payload(stdout)
    message = ""
    if payload is not None:
        message = str(payload.get("message") or payload.get("error") or "").strip()
    if not message:
        message = str(stderr or stdout or "Grok CLI failed").strip()
    if "not signed in" in message.lower() or "not authenticated" in message.lower():
        return LLMRouterError(
            "Grok CLI is not authenticated. Run 'grok login --device-code' "
            "or configure XAI_API_KEY."
        )
    return LLMRouterError(message or "Grok CLI failed")


def _redact_grok_cli_command(command: Sequence[str], prompt: str) -> list[str]:
    redacted: list[str] = []
    for value in command:
        text = str(value)
        redacted.append(text.replace(prompt, "<prompt>") if prompt else text)
    return redacted


_MISTRAL_LABS_PRIVACY_URL = "https://admin.mistral.ai/plateforme/privacy"


def _raise_mistral_vibe_access_error(exc: LLMRouterError) -> None:
    detail = str(exc).strip()
    lowered = detail.lower()
    if (
        "labs_not_enabled" in lowered
        or "labs model" in lowered
        or "code 1913" in lowered
        or '"code":"1913"' in lowered
    ):
        raise LLMRouterError(
            "Mistral Labs model access is disabled for this organization. "
            "An organization admin must enable API > Privacy > Labs models at "
            f"{_MISTRAL_LABS_PRIVACY_URL}. Enabling Labs permits Mistral to use "
            "Labs API data for model training regardless of the normal API opt-out. "
            f"Provider detail: {detail}"
        ) from exc
    raise exc


def _cli_available(command: str) -> bool:
    if not command:
        return False
    parts = shlex.split(command)
    if not parts:
        return False
    if parts[0] == "npx":
        return True
    return shutil.which(parts[0]) is not None


def find_standalone_copilot_cli() -> Optional[str]:
    """Return the standalone Copilot executable when installed."""

    return shutil.which("copilot")


@lru_cache(maxsize=8)
def _cli_help_text(command: str) -> str:
    if not command:
        return ""
    try:
        proc = subprocess.run(
            shlex.split(command) + ["--help"],
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
            env=os.environ.copy(),
        )
    except Exception:
        return ""
    return f"{proc.stdout or ''}\n{proc.stderr or ''}".strip()


def _copilot_cli_supports_image_inputs(command: str) -> bool:
    help_text = _cli_help_text(command).lower()
    return any(
        marker in help_text
        for marker in ("--image", "image input", "attach image")
    )


def _normalize_copilot_add_dirs(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        candidates: Sequence[object] = raw.split(os.pathsep)
    elif isinstance(raw, Sequence) and not isinstance(
        raw, (bytes, bytearray, str)
    ):
        candidates = raw
    else:
        candidates = [raw]
    output: list[str] = []
    for candidate in candidates:
        value = str(candidate or "").strip()
        if not value:
            continue
        normalized = os.path.abspath(os.path.expanduser(value))
        if normalized not in output:
            output.append(normalized)
    return output


def _copilot_cli_add_dirs_from_env() -> list[str]:
    return _normalize_copilot_add_dirs(
        _coalesce_env(
            "ipfs_accelerate_py_COPILOT_CLI_ADD_DIRS",
            "IPFS_ACCELERATE_PY_COPILOT_CLI_ADD_DIRS",
            "IPFS_DATASETS_PY_COPILOT_CLI_ADD_DIRS",
        )
    )


def _copilot_allow_all_paths_default() -> bool:
    raw = _coalesce_env(
        "ipfs_accelerate_py_COPILOT_CLI_ALLOW_ALL_PATHS",
        "IPFS_ACCELERATE_PY_COPILOT_CLI_ALLOW_ALL_PATHS",
        "IPFS_DATASETS_PY_COPILOT_CLI_ALLOW_ALL_PATHS",
    )
    return _truthy(raw)


def _run_cli_command(
    command: str,
    prompt: str,
    *,
    timeout_seconds: float = 120.0,
    template_vars: Optional[Dict[str, str]] = None,
    label: Optional[str] = None,
    extra_env: Optional[Dict[str, Optional[str]]] = None,
) -> str:
    if not command:
        raise RuntimeError("CLI command not configured")

    # Split the operator-owned command before substituting dynamic values so a
    # prompt, model, or agent remains one argv item even when it contains spaces.
    cmd = shlex.split(command)
    replacements = {
        "prompt": prompt,
        **{str(key): str(value) for key, value in (template_vars or {}).items()},
    }
    prompt_in_command = any("{prompt}" in part for part in cmd)
    for index, part in enumerate(cmd):
        for key, value in replacements.items():
            part = part.replace("{" + key + "}", value)
        cmd[index] = part
    input_text: str | None = None if prompt_in_command else prompt

    try:
        env = os.environ.copy()
        if extra_env:
            for key, value in extra_env.items():
                if not key:
                    continue
                if value is None:
                    env.pop(str(key), None)
                elif str(value).strip():
                    env[str(key)] = str(value)
        proc = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
            env=env,
        )
    except FileNotFoundError as exc:
        name = (label or "CLI").strip() or "CLI"
        raise LLMRouterError(f"{name} not found on PATH") from exc
    if proc.returncode != 0:
        name = (label or "CLI").strip() or "CLI"
        raise LLMRouterError(proc.stderr.strip() or f"{name} command failed")
    return (proc.stdout or "").strip()


def _get_openrouter_provider() -> Optional[LLMProvider]:
    api_key = _coalesce_env(
        "ipfs_accelerate_py_OPENROUTER_API_KEY",
        "IPFS_ACCELERATE_PY_OPENROUTER_API_KEY",
        "IPFS_DATASETS_PY_OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY",
    )
    if not api_key:
        return None

    base_url = (
        _coalesce_env(
            "ipfs_accelerate_py_OPENROUTER_BASE_URL",
            "IPFS_ACCELERATE_PY_OPENROUTER_BASE_URL",
            "IPFS_DATASETS_PY_OPENROUTER_BASE_URL",
        )
        or "https://openrouter.ai/api/v1"
    ).rstrip("/")

    def _request(payload: dict, *, timeout: float, bill_to: str = "") -> dict:
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
                **({"X-HF-Bill-To": bill_to} if bill_to else {}),
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
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
        if not isinstance(data, dict):
            raise RuntimeError("OpenRouter returned invalid JSON")
        return data

    class _OpenRouterProvider:
        def chat_completions(
            self,
            messages: Sequence[ChatMessage],
            *,
            model_name: Optional[str] = None,
            **kwargs: object,
        ) -> dict:
            model = (
                model_name
                or os.getenv("ipfs_accelerate_py_OPENROUTER_MODEL")
                or os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_MODEL")
                or os.getenv("IPFS_DATASETS_PY_OPENROUTER_MODEL")
                or os.getenv("ipfs_accelerate_py_LLM_MODEL")
                or os.getenv("IPFS_ACCELERATE_PY_LLM_MODEL")
                or os.getenv("IPFS_DATASETS_PY_LLM_MODEL")
                or "openai/gpt-4o-mini"
            )

            max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256))
            temperature = kwargs.get("temperature", 0.2)

            payload: dict = {
                "model": model,
                "messages": list(messages),
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
            }

            # Optional OpenAI-compatible fields.
            if "logprobs" in kwargs:
                payload["logprobs"] = bool(kwargs.get("logprobs"))
            if "top_logprobs" in kwargs and kwargs.get("top_logprobs") is not None:
                payload["top_logprobs"] = int(kwargs.get("top_logprobs"))
            if "response_format" in kwargs and kwargs.get("response_format") is not None:
                payload["response_format"] = kwargs.get("response_format")
            if "seed" in kwargs and kwargs.get("seed") is not None:
                payload["seed"] = int(kwargs.get("seed"))

            timeout = float(kwargs.get("timeout", 120))
            return _request(
                payload,
                timeout=timeout,
                bill_to=_resolve_hf_bill_to(kwargs=dict(kwargs)),
            )

        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            data = self.chat_completions(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                **kwargs,
            )

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


def _get_openai_provider() -> Optional[LLMProvider]:
    api_key = _resolve_openai_api_key()
    if not api_key:
        return None
    base_url = (
        _coalesce_env(
            "IPFS_ACCELERATE_PY_OPENAI_BASE_URL",
            "ipfs_accelerate_py_OPENAI_BASE_URL",
            "IPFS_DATASETS_PY_OPENAI_BASE_URL",
        )
        or "https://api.openai.com/v1"
    ).rstrip("/")

    def _request(payload: dict, *, timeout: float) -> dict:
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(
                f"OpenAI HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc
        try:
            data = json.loads(raw)
        except Exception as exc:
            raise RuntimeError("OpenAI returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("OpenAI returned invalid JSON")
        return data

    class _OpenAIProvider:
        def chat_completions(
            self,
            messages: Sequence[ChatMessage],
            *,
            model_name: Optional[str] = None,
            **kwargs: object,
        ) -> dict:
            model = model_name or _coalesce_env(
                "IPFS_ACCELERATE_PY_OPENAI_MODEL",
                "ipfs_accelerate_py_OPENAI_MODEL",
                "IPFS_DATASETS_PY_OPENAI_MODEL",
                "OPENAI_MODEL",
                "IPFS_ACCELERATE_PY_LLM_MODEL",
                "ipfs_accelerate_py_LLM_MODEL",
                "IPFS_DATASETS_PY_LLM_MODEL",
            ) or "gpt-4.1-mini"
            payload: dict[str, object] = {
                "model": model,
                "messages": list(messages),
                "max_tokens": int(
                    kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256))
                ),
                "temperature": float(kwargs.get("temperature", 0.2)),
            }
            for key in ("logprobs", "top_logprobs", "response_format", "seed"):
                value = kwargs.get(key)
                if value is not None:
                    payload[key] = value
            return _request(payload, timeout=float(kwargs.get("timeout", 120)))

        def generate(
            self,
            prompt: str,
            *,
            model_name: Optional[str] = None,
            **kwargs: object,
        ) -> str:
            data = self.chat_completions(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                **kwargs,
            )
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                item = choices[0] if isinstance(choices[0], dict) else {}
                message = item.get("message")
                if isinstance(message, dict) and isinstance(
                    message.get("content"), str
                ):
                    return message["content"].strip()
                if isinstance(item.get("text"), str):
                    return item["text"].strip()
            raise RuntimeError("OpenAI response missing choices")

    return _OpenAIProvider()


def _get_hf_inference_api_provider() -> Optional[LLMProvider]:
    api_token = _resolve_hf_api_token()
    if not api_token:
        return None
    base_url = (
        _coalesce_env(
            "IPFS_ACCELERATE_PY_HF_INFERENCE_BASE_URL",
            "IPFS_DATASETS_PY_HF_INFERENCE_BASE_URL",
        )
        or "https://router.huggingface.co/hf-inference/models"
    ).rstrip("/")

    class _HFInferenceAPIProvider:
        def chat_completions(
            self,
            messages: Sequence[ChatMessage],
            *,
            model_name: Optional[str] = None,
            **kwargs: object,
        ) -> dict:
            timeout = float(kwargs.get("timeout", 120))
            bill_to = _resolve_hf_bill_to(kwargs=dict(kwargs))
            provider_name = _resolve_hf_provider(kwargs=dict(kwargs)).strip() or "auto"
            selected_model = (
                model_name or _default_hf_inference_model(kwargs=dict(kwargs))
            ).strip()
            try:
                hub = importlib.import_module("huggingface_hub")
                client_cls = getattr(hub, "InferenceClient", None)
                if client_cls is None:
                    raise RuntimeError("huggingface_hub.InferenceClient not available")
                client = client_cls(
                    **_build_hf_inference_client_kwargs(
                        provider=provider_name,
                        token=api_token,
                        bill_to=bill_to,
                        timeout=timeout,
                    )
                )
                chat = getattr(client, "chat", None)
                completions = getattr(chat, "completions", None)
                create = getattr(completions, "create", None)
                if not callable(create):
                    raise RuntimeError(
                        "huggingface_hub.InferenceClient chat completions not available"
                    )
                payload: dict[str, object] = {
                    "messages": list(messages),
                    "model": selected_model,
                    "stream": False,
                }
                max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens"))
                if max_tokens is not None:
                    payload["max_tokens"] = int(max_tokens)
                for key in (
                    "temperature",
                    "top_p",
                    "logprobs",
                    "top_logprobs",
                    "response_format",
                    "seed",
                    "tools",
                    "tool_choice",
                    "tool_prompt",
                    "extra_body",
                    "frequency_penalty",
                    "presence_penalty",
                ):
                    value = kwargs.get(key)
                    if value is not None:
                        payload[key] = value
                stop = kwargs.get("stop")
                if stop is not None:
                    payload["stop"] = (
                        list(stop)
                        if isinstance(stop, (list, tuple))
                        else [str(stop)]
                    )
                result = create(**payload)
            except Exception as exc:
                raise RuntimeError(
                    f"HF Inference Providers chat request failed: {exc}"
                ) from exc
            data = _hf_to_jsonable(result)
            if not isinstance(data, dict):
                raise RuntimeError("HF Inference Providers chat response invalid")
            return data

        def generate(
            self,
            prompt: str,
            *,
            model_name: Optional[str] = None,
            **kwargs: object,
        ) -> str:
            effective_kwargs = dict(kwargs)
            if _hf_use_chat_completions(
                model_name=model_name,
                kwargs=effective_kwargs,
            ):
                data = self.chat_completions(
                    [{"role": "user", "content": prompt}],
                    model_name=model_name,
                    **kwargs,
                )
                parsed = _parse_openai_compat_response(data)
                if parsed.choices and parsed.choices[0].message.content:
                    return parsed.choices[0].message.content
                raise RuntimeError(
                    "HF Inference Providers chat response missing choices"
                )

            max_new_tokens = int(
                kwargs.get("max_new_tokens", kwargs.get("max_tokens", 128))
            )
            temperature = float(kwargs.get("temperature", 0.2))
            timeout = float(kwargs.get("timeout", 120))
            wait_for_model_raw = kwargs.get("wait_for_model", True)
            use_cache_raw = kwargs.get("use_cache", True)
            wait_for_model = (
                _truthy(wait_for_model_raw)
                if isinstance(wait_for_model_raw, str)
                else bool(wait_for_model_raw)
            )
            use_cache = (
                _truthy(use_cache_raw)
                if isinstance(use_cache_raw, str)
                else bool(use_cache_raw)
            )
            parameters: dict[str, object] = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }
            for key in (
                "top_p",
                "top_k",
                "repetition_penalty",
                "do_sample",
                "return_full_text",
            ):
                value = kwargs.get(key)
                if value is not None:
                    parameters[key] = value
            payload: dict[str, object] = {
                "inputs": prompt,
                "parameters": parameters,
                "options": {
                    "wait_for_model": wait_for_model,
                    "use_cache": use_cache,
                },
            }
            bill_to = _resolve_hf_bill_to(kwargs=effective_kwargs)
            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if bill_to:
                headers["X-HF-Bill-To"] = bill_to

            def _generate_with_model(selected_model: str) -> str:
                request = urllib.request.Request(
                    f"{base_url}/{selected_model}",
                    data=json.dumps(payload).encode("utf-8"),
                    method="POST",
                    headers=headers,
                )

                def _generate_via_inference_client() -> str:
                    try:
                        hub = importlib.import_module("huggingface_hub")
                        client_cls = getattr(hub, "InferenceClient", None)
                        if client_cls is None:
                            raise RuntimeError(
                                "huggingface_hub.InferenceClient not available"
                            )
                        client = client_cls(
                            **_build_hf_inference_client_kwargs(
                                provider="hf-inference",
                                token=api_token,
                                bill_to=bill_to,
                                timeout=timeout,
                            )
                        )
                        result = client.text_generation(
                            prompt,
                            model=selected_model,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=parameters.get("top_p"),
                            top_k=parameters.get("top_k"),
                            repetition_penalty=parameters.get("repetition_penalty"),
                            do_sample=parameters.get("do_sample"),
                            return_full_text=parameters.get("return_full_text"),
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"HF Inference Client request failed: {exc}"
                        ) from exc
                    if isinstance(result, str) and result:
                        return result
                    generated = getattr(result, "generated_text", None)
                    if isinstance(generated, str) and generated:
                        return generated
                    raise RuntimeError(
                        "HF Inference Client response missing generated text"
                    )

                try:
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        raw = response.read().decode("utf-8", errors="replace")
                except urllib.error.HTTPError as exc:
                    if exc.code in {400, 404, 422, 503}:
                        return _generate_via_inference_client()
                    detail = (
                        exc.read().decode("utf-8", errors="replace")
                        if exc.fp
                        else ""
                    )
                    raise RuntimeError(
                        f"HF Inference API HTTP {exc.code}: {detail or exc.reason}"
                    ) from exc
                except Exception as exc:
                    if "404" in str(exc) or "Not Found" in str(exc):
                        return _generate_via_inference_client()
                    raise RuntimeError(
                        f"HF Inference API request failed: {exc}"
                    ) from exc
                try:
                    data = json.loads(raw)
                except Exception as exc:
                    raise RuntimeError(
                        "HF Inference API returned invalid JSON"
                    ) from exc
                if isinstance(data, dict) and isinstance(data.get("error"), str):
                    raise RuntimeError(
                        f"HF Inference API error: {data.get('error')}"
                    )
                extracted = _extract_hf_response_text(data)
                if extracted is None:
                    raise RuntimeError(
                        "HF Inference API response missing generated text"
                    )
                return extracted

            selected_model = (
                model_name or _default_hf_inference_model(kwargs=effective_kwargs)
            ).strip()
            routed_model: Optional[str] = None
            if model_name is None and not bool(
                kwargs.get("hf_skip_model_routing")
            ):
                routed_model = _route_hf_model_with_arch_router(
                    prompt=prompt,
                    kwargs=effective_kwargs,
                    request_timeout=timeout,
                    generate_fn=lambda router_prompt, router_model, router_timeout: _HFInferenceAPIProvider().generate(
                        router_prompt,
                        model_name=router_model,
                        hf_skip_model_routing=True,
                        max_new_tokens=128,
                        temperature=0.0,
                        timeout=router_timeout,
                        return_full_text=False,
                    ),
                )
                if routed_model:
                    selected_model = routed_model
            candidates = [selected_model]
            if model_name is None:
                candidates = _ordered_hf_generation_models(
                    kwargs=effective_kwargs,
                    selected_model=selected_model,
                    routed_model=routed_model,
                )
            last_error: Optional[Exception] = None
            for candidate in candidates:
                try:
                    return _generate_with_model(candidate)
                except Exception as exc:
                    last_error = exc
                    if not _is_hf_model_compatibility_error(exc):
                        raise
            if last_error is not None:
                raise last_error
            return _generate_with_model(selected_model)

    return _HFInferenceAPIProvider()


def _get_llama_cpp_provider(*, auto_install: bool = False) -> Optional[LLMProvider]:
    """Return a local llama.cpp OpenAI-compatible provider."""

    try:
        from ipfs_accelerate_py.utils.llama_cpp import (
            DEFAULT_LEANSTRAL_MODEL_REF,
            config_from_env,
            ensure_llama_cpp_server,
            llama_cpp_server_ready,
        )
    except Exception:
        return None

    configured_base_url = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_BASE_URL",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_BASE_URL",
        "ipfs_accelerate_py_LLAMA_CPP_BASE_URL",
    ).rstrip("/")
    server_config = config_from_env()
    base_url = configured_base_url or server_config.base_url
    model_default = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_MODEL",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_MODEL",
        "ipfs_accelerate_py_LLAMA_CPP_MODEL",
    ) or server_config.model_ref or DEFAULT_LEANSTRAL_MODEL_REF
    api_key = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_API_KEY",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_API_KEY",
        "ipfs_accelerate_py_LLAMA_CPP_API_KEY",
    )

    def _request(payload: dict, *, timeout: float) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"llama.cpp HTTP {exc.code}: {detail or exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"llama.cpp request failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except Exception as exc:
            raise RuntimeError("llama.cpp returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("llama.cpp returned invalid JSON")
        return data

    class _LlamaCppProvider:
        def __init__(self) -> None:
            self._ready_checked = False

        def _ensure_ready(self) -> None:
            if self._ready_checked and llama_cpp_server_ready(base_url):
                return
            if llama_cpp_server_ready(base_url):
                self._ready_checked = True
                return

            autostart = _truthy_env("IPFS_ACCELERATE_LLAMA_CPP_AUTOSTART", default=False)
            allow_install = bool(auto_install) and _truthy_env(
                "IPFS_ACCELERATE_LLAMA_CPP_AUTO_INSTALL",
                default=False,
            )
            allow_update = _truthy_env("IPFS_ACCELERATE_LLAMA_CPP_AUTO_UPDATE", default=False)
            prefetch_model = _truthy_env("IPFS_ACCELERATE_LLAMA_CPP_PREFETCH_MODEL", default=False)
            startup_timeout = float(
                _coalesce_env(
                    "IPFS_ACCELERATE_LLAMA_CPP_STARTUP_TIMEOUT_SECONDS",
                    "IPFS_ACCELERATE_PY_LLAMA_CPP_STARTUP_TIMEOUT_SECONDS",
                    "ipfs_accelerate_py_LLAMA_CPP_STARTUP_TIMEOUT_SECONDS",
                )
                or "60"
            )
            if not autostart:
                raise LLMRouterError(
                    "llama.cpp server is not reachable at "
                    f"{base_url}. Start it with `ipfs-accelerate-llama-cpp-serve --serve` "
                    "or set IPFS_ACCELERATE_LLAMA_CPP_AUTOSTART=1."
                )

            result = ensure_llama_cpp_server(
                server_config,
                autostart=True,
                auto_install=allow_install,
                auto_update=allow_update,
                prefetch_model=prefetch_model,
                startup_timeout_seconds=startup_timeout,
            )
            if not result.running:
                cache = result.model_cache.to_dict()
                raise LLMRouterError(
                    "llama.cpp server did not become ready: "
                    f"{result.message}; base_url={result.base_url}; log={result.log_path}; "
                    f"model_cache={cache.get('message')}; "
                    f"partial_size_bytes={cache.get('partial_size_bytes', 0)}"
                )
            self._ready_checked = True

        def chat_completions(
            self,
            messages: Sequence[ChatMessage],
            *,
            model_name: Optional[str] = None,
            **kwargs: object,
        ) -> dict:
            self._ensure_ready()
            model = model_name or str(kwargs.pop("model", "") or "").strip() or model_default
            max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256))
            temperature = kwargs.get("temperature", 0.2)
            payload: dict = {
                "model": model,
                "messages": list(messages),
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
            }
            if "top_p" in kwargs and kwargs.get("top_p") is not None:
                payload["top_p"] = float(kwargs.get("top_p"))
            if "seed" in kwargs and kwargs.get("seed") is not None:
                payload["seed"] = int(kwargs.get("seed"))
            if "stop" in kwargs and kwargs.get("stop") is not None:
                payload["stop"] = kwargs.get("stop")
            if "logprobs" in kwargs:
                payload["logprobs"] = bool(kwargs.get("logprobs"))
            if "top_logprobs" in kwargs and kwargs.get("top_logprobs") is not None:
                payload["top_logprobs"] = int(kwargs.get("top_logprobs"))
            if "response_format" in kwargs and kwargs.get("response_format") is not None:
                payload["response_format"] = kwargs.get("response_format")
            timeout = float(kwargs.get("timeout", 300))
            return _request(payload, timeout=timeout)

        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            data = self.chat_completions(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                **kwargs,
            )
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg["content"].strip()
                    text = first.get("text")
                    if isinstance(text, str):
                        return text.strip()
            raise RuntimeError("llama.cpp response missing choices")

    return _LlamaCppProvider()


_LLAMA_CPP_SERVER_PROVIDER_ALIASES = {
    "llama_cpp",
    "llamacpp",
    "llama.cpp",
    "openai_compatible",
    "local_openai",
    "leanstral_local",
}

_LLAMA_CPP_NATIVE_PROVIDER_ALIASES = {
    "llama_cpp_native",
    "llamacpp_native",
    "native_llama_cpp",
    "llama.cpp_native",
}


def _get_llama_cpp_native_provider(*, auto_install: bool = False) -> Optional[LLMProvider]:
    """Return a local in-process llama-cpp-python provider when available."""

    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as import_exc:
        allow_install = bool(auto_install) and _truthy_env(
            "IPFS_ACCELERATE_LLAMA_CPP_NATIVE_AUTO_INSTALL",
            default=False,
        )
        if not allow_install:
            return None
        package = (
            _coalesce_env(
                "IPFS_ACCELERATE_LLAMA_CPP_NATIVE_PACKAGE",
                "IPFS_ACCELERATE_PY_LLAMA_CPP_NATIVE_PACKAGE",
            )
            or "llama-cpp-python[server]"
        )
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            text=True,
            capture_output=True,
            check=False,
            timeout=float(
                _coalesce_env("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_INSTALL_TIMEOUT_SECONDS")
                or "1800"
            ),
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or str(import_exc)).strip()
            raise LLMRouterError(f"llama-cpp-python install failed: {detail}") from import_exc
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:
            raise LLMRouterError("llama-cpp-python installed but could not be imported") from exc

    try:
        from ipfs_accelerate_py.utils.llama_cpp import (
            DEFAULT_LEANSTRAL_FILENAME,
            DEFAULT_LEANSTRAL_MODEL_REF,
            config_from_env,
            llama_cpp_model_cache_status,
        )
    except Exception:
        DEFAULT_LEANSTRAL_FILENAME = "Leanstral-1.5-119B-A6B-NVFP4.gguf"
        DEFAULT_LEANSTRAL_MODEL_REF = "Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4:NVFP4"
        config_from_env = None  # type: ignore[assignment]
        llama_cpp_model_cache_status = None  # type: ignore[assignment]

    server_config = config_from_env() if callable(config_from_env) else None
    default_model_ref = getattr(server_config, "model_ref", "") or DEFAULT_LEANSTRAL_MODEL_REF
    default_hf_file = getattr(server_config, "hf_file", "") or DEFAULT_LEANSTRAL_FILENAME
    model_path = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_NATIVE_MODEL_PATH",
        "IPFS_ACCELERATE_LLAMA_CPP_MODEL_PATH",
    )
    if not model_path and server_config is not None and callable(llama_cpp_model_cache_status):
        try:
            cache_status = llama_cpp_model_cache_status(server_config)
        except Exception:
            cache_status = None
        if (
            cache_status is not None
            and getattr(cache_status, "complete", False)
            and (
                getattr(cache_status, "cache_backend", "")
                in {"content_addressed_disk", "ipfs_kit", "model_path"}
                or str(getattr(cache_status, "cache_backend", "")).endswith("_hash_pending")
            )
        ):
            model_path = str(getattr(cache_status, "local_path", "") or "")
    model_ref = (
        _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_NATIVE_MODEL_REF",
            "IPFS_ACCELERATE_LLAMA_CPP_MODEL_REF",
        )
        or default_model_ref
    )
    repo_id = str(model_ref or "").split(":", 1)[0]
    hf_file = (
        _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_NATIVE_HF_FILE",
            "IPFS_ACCELERATE_LLAMA_CPP_HF_FILE",
        )
        or default_hf_file
    )

    def _int_env(*names: str, default: Optional[int] = None) -> Optional[int]:
        raw = _coalesce_env(*names)
        if not raw:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    context_size = _int_env(
        "IPFS_ACCELERATE_LLAMA_CPP_NATIVE_CONTEXT_SIZE",
        "IPFS_ACCELERATE_LLAMA_CPP_CONTEXT_SIZE",
        default=int(getattr(server_config, "context_size", 2048) or 2048),
    )
    threads = _int_env(
        "IPFS_ACCELERATE_LLAMA_CPP_NATIVE_THREADS",
        "IPFS_ACCELERATE_LLAMA_CPP_THREADS",
        default=int(getattr(server_config, "threads", 0) or 0),
    )
    gpu_layers = _int_env(
        "IPFS_ACCELERATE_LLAMA_CPP_NATIVE_GPU_LAYERS",
        "IPFS_ACCELERATE_LLAMA_CPP_GPU_LAYERS",
        default=getattr(server_config, "gpu_layers", None),
    )
    verbose = _truthy_env("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_VERBOSE", default=False)

    class _NativeLlamaCppProvider:
        def __init__(self) -> None:
            self._llm: object | None = None

        def _load(self) -> object:
            if self._llm is not None:
                return self._llm
            load_kwargs: dict[str, object] = {"verbose": verbose}
            if context_size and context_size > 0:
                load_kwargs["n_ctx"] = int(context_size)
            if threads and threads > 0:
                load_kwargs["n_threads"] = int(threads)
            if gpu_layers is not None:
                load_kwargs["n_gpu_layers"] = int(gpu_layers)
            if model_path:
                self._llm = Llama(model_path=model_path, **load_kwargs)
                return self._llm
            if not repo_id or not hf_file:
                raise LLMRouterError(
                    "llama_cpp_native requires IPFS_ACCELERATE_LLAMA_CPP_NATIVE_MODEL_PATH "
                    "or a Hugging Face repo/file pair."
                )
            from_pretrained = getattr(Llama, "from_pretrained", None)
            if not callable(from_pretrained):
                raise LLMRouterError("Installed llama-cpp-python lacks Llama.from_pretrained")
            self._llm = from_pretrained(repo_id=repo_id, filename=hf_file, **load_kwargs)
            return self._llm

        def chat_completions(
            self,
            messages: Sequence[ChatMessage],
            *,
            model_name: Optional[str] = None,
            **kwargs: object,
        ) -> dict:
            _ = model_name
            llm = self._load()
            max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256))
            native_kwargs: dict[str, object] = {
                "messages": [dict(message) for message in messages],
                "max_tokens": int(max_tokens),
                "temperature": float(kwargs.get("temperature", 0.2)),
            }
            for key in ("top_p", "stop", "seed", "logprobs", "top_logprobs", "response_format"):
                if key in kwargs and kwargs.get(key) is not None:
                    native_kwargs[key] = kwargs.get(key)
            create_chat_completion = getattr(llm, "create_chat_completion", None)
            if not callable(create_chat_completion):
                raise LLMRouterError("llama_cpp_native model lacks create_chat_completion")
            try:
                result = create_chat_completion(**native_kwargs)
            except TypeError:
                stable_kwargs = {
                    key: value
                    for key, value in native_kwargs.items()
                    if key in {"messages", "max_tokens", "temperature", "top_p", "stop"}
                }
                if stable_kwargs == native_kwargs:
                    raise
                result = create_chat_completion(**stable_kwargs)
            if not isinstance(result, dict):
                raise RuntimeError("llama_cpp_native returned invalid chat completion")
            return result

        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            data = self.chat_completions(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                **kwargs,
            )
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg["content"].strip()
                    text = first.get("text")
                    if isinstance(text, str):
                        return text.strip()
            raise RuntimeError("llama_cpp_native response missing choices")

    return _NativeLlamaCppProvider()


def _get_codex_cli_provider() -> Optional[LLMProvider]:
    if not shutil.which("codex"):
        return None

    class _CodexCLIProvider:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            model = (model_name or _coalesce_env("ipfs_accelerate_py_CODEX_CLI_MODEL", "ipfs_accelerate_py_CODEX_MODEL") or "chatgpt-5.6-terra").strip()
            sandbox = (os.getenv("ipfs_accelerate_py_CODEX_SANDBOX", "auto") or "auto").strip()
            skip_git_repo_check = os.getenv("ipfs_accelerate_py_CODEX_SKIP_GIT_REPO_CHECK", "1") != "0"
            timeout = float(kwargs.get("timeout", 180))

            trace_jsonl_path = kwargs.pop("trace_jsonl_path", None)
            trace_dir = kwargs.pop("trace_dir", None)
            trace_enabled = bool(kwargs.pop("trace", False) or trace_jsonl_path or trace_dir)

            json_mode = bool(trace_enabled or kwargs.pop("json", False))

            with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as last_msg:
                last_msg_path = last_msg.name

            cmd: list[str] = ["codex", "exec"]
            if skip_git_repo_check:
                cmd.append("--skip-git-repo-check")
            # Some Codex CLI builds do not accept '--sandbox auto'.
            # Treat 'auto' (the default) as "don't pass the flag" so the CLI can
            # pick its own default sandbox mode.
            if sandbox and sandbox.lower() != "auto":
                cmd.extend(["--sandbox", sandbox])
            if model:
                cmd.extend(["-m", model])
            cmd.extend(["--output-last-message", last_msg_path])
            if json_mode:
                cmd.append("--json")
            cmd.append("-")

            try:
                proc = subprocess.run(
                    cmd,
                    input=str(prompt),
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=timeout,
                )
            except FileNotFoundError as exc:
                raise LLMRouterError("codex CLI not found on PATH") from exc

            try:
                with open(last_msg_path, "r", encoding="utf-8", errors="replace") as handle:
                    text_out = handle.read().strip()
            except Exception:
                text_out = ""
            finally:
                try:
                    os.unlink(last_msg_path)
                except Exception:
                    pass

            if proc.returncode == 0 or text_out:
                if json_mode and proc.stdout:
                    extracted = _extract_last_agent_message_from_codex_jsonl(proc.stdout)
                    if extracted:
                        return _clean_codex_output(extracted)
                return _clean_codex_output(text_out)

            if trace_enabled and proc.stdout and isinstance(trace_jsonl_path, str) and trace_jsonl_path.strip():
                try:
                    os.makedirs(os.path.dirname(trace_jsonl_path.strip()) or ".", exist_ok=True)
                    with open(trace_jsonl_path.strip(), "a", encoding="utf-8") as handle:
                        handle.write(proc.stdout)
                        if not proc.stdout.endswith("\n"):
                            handle.write("\n")
                except OSError:
                    pass

            kind = _classify_codex_error_kind(stdout=proc.stdout or "", stderr=proc.stderr or "")
            resets = _extract_resets_in_seconds_from_codex_jsonl(proc.stdout or "")
            if kind == "quota_exceeded":
                raise LLMRouterError("Codex quota exceeded (billing/plan hard limit)")
            if kind == "usage_limit":
                suffix = f" (resets in ~{resets}s)" if isinstance(resets, int) else ""
                raise LLMRouterError(f"Codex usage limit reached{suffix}")
            raise LLMRouterError(proc.stderr.strip() or "codex exec failed")

    return _CodexCLIProvider()


def find_goose_cli() -> Optional[str]:
    """Locate the goose CLI binary without starting a process."""

    configured = _coalesce_env(
        "ipfs_accelerate_py_GOOSE_BIN",
        "IPFS_ACCELERATE_PY_GOOSE_BIN",
        "IPFS_ACCELERATE_AGENT_GOOSE_BIN",
        "GOOSE_BIN",
    )
    if configured:
        path = Path(configured).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return shutil.which("goose")


def _goose_default_model() -> str:
    return (
        _coalesce_env(
            "ipfs_accelerate_py_GOOSE_CLI_MODEL",
            "IPFS_ACCELERATE_PY_GOOSE_CLI_MODEL",
            "GOOSE_MODEL",
            "ipfs_accelerate_py_META_AI_MODEL",
            "ipfs_accelerate_py_LLM_MODEL",
        )
        or META_MODEL_API_DEFAULT_MODEL
    )


def _goose_openai_compatible_backend_env(
    base_env: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Build env so goose can talk to Meta Muse Spark (or an override host).

    Goose's OpenAI-compatible transport is the same family of boundary as
    ``meta_ai`` / ``xai`` HTTP providers: host + bearer key + model id.
    """

    env = dict(base_env or os.environ)
    # Prefer an explicit OPENAI_API_KEY when the operator already set one.
    if not str(env.get("OPENAI_API_KEY") or "").strip():
        meta_key = resolve_meta_model_api_key()
        if meta_key:
            env["OPENAI_API_KEY"] = meta_key
    host = (
        _coalesce_env(
            "ipfs_accelerate_py_GOOSE_OPENAI_HOST",
            "IPFS_ACCELERATE_AGENT_META_SPARK_HOST",
            "OPENAI_HOST",
        )
        or "https://api.meta.ai"
    )
    base_path = (
        _coalesce_env(
            "ipfs_accelerate_py_GOOSE_OPENAI_BASE_PATH",
            "IPFS_ACCELERATE_AGENT_META_SPARK_BASE_PATH",
            "OPENAI_BASE_PATH",
        )
        or "v1/chat/completions"
    )
    env.setdefault("OPENAI_HOST", host)
    env.setdefault("OPENAI_BASE_PATH", base_path)
    env.setdefault("GOOSE_PROVIDER", env.get("GOOSE_PROVIDER") or "openai")
    env.setdefault("GOOSE_DISABLE_KEYRING", env.get("GOOSE_DISABLE_KEYRING") or "1")
    local_bin = str(Path.home() / ".local" / "bin")
    path = env.get("PATH", "")
    if local_bin not in path.split(os.pathsep):
        env["PATH"] = local_bin + os.pathsep + path
    return env


def build_goose_cli_command(
    *,
    mode: str = "chat",
    workspace: Optional[str | Path] = None,
    model_name: Optional[str] = None,
    max_turns: Optional[int] = None,
    with_developer: bool = False,
    goose_bin: Optional[str] = None,
) -> list[str]:
    """Return argv for a goose CLI invocation (prompt is always stdin via ``-i -``).

    ``mode="chat"`` is the safe llm_router default: no tools, no session, no
    default profile extensions. ``mode="agent"`` is for explicitly authorized
    side-effecting runs (e.g. the agent supervisor implementation daemon).
    """

    binary = (goose_bin or find_goose_cli() or "").strip()
    if not binary:
        raise LLMRouterError("goose CLI not found on PATH")
    normalized = str(mode or "chat").strip().lower()
    if normalized not in {"chat", "agent"}:
        raise LLMRouterError(f"unsupported goose mode: {mode!r}")

    if max_turns is None:
        if normalized == "chat":
            max_turns = int(
                _coalesce_env("ipfs_accelerate_py_GOOSE_CLI_MAX_TURNS", "2") or "2"
            )
        else:
            max_turns = int(
                _coalesce_env(
                    "ipfs_accelerate_py_GOOSE_AGENT_MAX_TURNS",
                    "IPFS_ACCELERATE_AGENT_GOOSE_MAX_TURNS",
                    "40",
                )
                or "40"
            )
    max_turns = max(1, int(max_turns))

    cmd: list[str] = [
        binary,
        "run",
        "--no-session",
        "--quiet",
        "--max-turns",
        str(max_turns),
    ]
    if normalized == "chat":
        cmd.append("--no-profile")
    if normalized == "agent" or with_developer:
        cmd.extend(["--with-builtin", "developer"])
    # Instruction body is supplied by the caller on stdin.
    cmd.extend(["-i", "-"])
    if workspace is not None:
        # Goose itself does not take -C; callers must set cwd. Keep the
        # workspace argument out of argv so it cannot be mistaken for a
        # prompt path.
        _ = Path(workspace)
    _ = model_name  # model is carried via GOOSE_MODEL in the environment
    return cmd


def build_goose_cli_env(
    *,
    mode: str = "chat",
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    base_env: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Environment for goose CLI runs, defaulting backend to Meta Muse Spark."""

    env = _goose_openai_compatible_backend_env(base_env)
    normalized = str(mode or "chat").strip().lower()
    env["GOOSE_MODE"] = "chat" if normalized == "chat" else (env.get("GOOSE_MODE") or "auto")
    model = (model_name or "").strip() or _goose_default_model()
    env["GOOSE_MODEL"] = normalize_meta_model_name(model) if "spark" in model.lower() or model.startswith("muse-") else model
    if max_tokens is None:
        max_tokens = int(
            _coalesce_env(
                "ipfs_accelerate_py_GOOSE_MAX_TOKENS",
                "IPFS_ACCELERATE_AGENT_GOOSE_MAX_TOKENS",
                "4096",
            )
            or "4096"
        )
    env["GOOSE_MAX_TOKENS"] = str(max(64, int(max_tokens)))
    if not str(env.get("OPENAI_API_KEY") or "").strip():
        raise LLMRouterError(
            "goose_cli requires OPENAI_API_KEY or a Meta Spark credential "
            "(meta_ai_api_key / MODEL_API_KEY / META_AI_API_KEY)"
        )
    return env


def _get_goose_cli_provider() -> Optional[LLMProvider]:
    """Return the Goose CLI provider when the binary is present.

    Ordinary ``generate`` is chat-only and never enables tools/extensions.
    Pass ``agent=True`` (or ``side_effecting=True``) plus an explicit
    ``workspace`` only for authorized agent runs.
    """

    if not find_goose_cli():
        return None

    class _GooseCLIProvider:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            agent = bool(
                kwargs.pop("agent", False)
                or kwargs.pop("side_effecting", False)
                or kwargs.pop("with_tools", False)
            )
            workspace = kwargs.pop("workspace", None) or kwargs.pop("cwd", None)
            with_developer = bool(kwargs.pop("with_developer", agent))
            max_turns = kwargs.pop("max_turns", None)
            max_tokens = kwargs.pop("max_tokens", None) or kwargs.pop(
                "max_completion_tokens", None
            )
            timeout = float(kwargs.pop("timeout", 300 if agent else 180))
            mode = "agent" if agent else "chat"
            if agent and not workspace:
                raise LLMRouterError(
                    "goose_cli agent mode requires an explicit workspace/cwd"
                )

            try:
                cmd = build_goose_cli_command(
                    mode=mode,
                    workspace=workspace if workspace is not None else None,
                    model_name=model_name,
                    max_turns=int(max_turns) if max_turns is not None else None,
                    with_developer=with_developer and agent,
                )
                env = build_goose_cli_env(
                    mode=mode,
                    model_name=model_name,
                    max_tokens=int(max_tokens) if max_tokens is not None else None,
                )
            except LLMRouterError:
                raise
            except Exception as exc:
                raise LLMRouterError(f"goose_cli configuration failed: {exc}") from exc

            cwd = str(Path(workspace).expanduser().resolve()) if workspace else None
            try:
                proc = subprocess.run(
                    cmd,
                    input=str(prompt),
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=timeout,
                    env=env,
                    cwd=cwd,
                )
            except FileNotFoundError as exc:
                raise LLMRouterError("goose CLI not found on PATH") from exc
            except subprocess.TimeoutExpired as exc:
                raise LLMRouterError(
                    f"goose_cli timed out after {timeout}s"
                ) from exc

            text_out = (proc.stdout or "").strip()
            if proc.returncode == 0 and text_out:
                return text_out
            err = (proc.stderr or "").strip() or (proc.stdout or "").strip()
            lowered = err.lower()
            if "usage limit" in lowered or "rate limit" in lowered or "quota" in lowered:
                raise LLMRouterError(f"goose_cli capacity/quota error: {err[:500]}")
            if "api key" in lowered or "authentication" in lowered or "unauthorized" in lowered:
                raise LLMRouterError(f"goose_cli authentication failed: {err[:500]}")
            if proc.returncode != 0:
                raise LLMRouterError(err or f"goose run failed with exit {proc.returncode}")
            return text_out

    return _GooseCLIProvider()


def _get_copilot_cli_provider() -> Optional[LLMProvider]:
    # Default to the official Copilot CLI via npx. We run it in "interactive"
    # mode with an auto-executed prompt (`-i`) so this works in non-interactive
    # worker subprocesses.
    default_command = "npx --yes @github/copilot"
    command = _coalesce_env(
        "ipfs_accelerate_py_COPILOT_CLI_CMD",
        "IPFS_ACCELERATE_PY_COPILOT_CLI_CMD",
        "IPFS_DATASETS_PY_COPILOT_CLI_CMD",
    ) or default_command
    if not _cli_available(command):
        return None
    supports_image_inputs = _copilot_cli_supports_image_inputs(command)

    class _CopilotCLIProvider:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            model = (
                (model_name or "").strip()
                or _coalesce_env(
                    "ipfs_accelerate_py_COPILOT_CLI_MODEL",
                    "IPFS_ACCELERATE_PY_COPILOT_CLI_MODEL",
                    "IPFS_DATASETS_PY_COPILOT_CLI_MODEL",
                )
                or _generic_llm_model_env()
                or "gpt-5-mini"
            )
            timeout = float(kwargs.get("timeout", 180))

            trace_jsonl_path = kwargs.pop("trace_jsonl_path", None)
            trace_dir = kwargs.pop("trace_dir", None)
            trace_enabled = bool(kwargs.pop("trace", False) or trace_jsonl_path or trace_dir)

            copilot_config_dir = kwargs.pop("copilot_config_dir", None)
            copilot_log_dir = kwargs.pop("copilot_log_dir", None)
            resume_session_id = kwargs.pop("resume_session_id", None)
            continue_session = bool(kwargs.pop("continue_session", False))
            copilot_add_dirs = _normalize_copilot_add_dirs(
                kwargs.pop("copilot_add_dirs", None)
            )
            if not copilot_add_dirs:
                copilot_add_dirs = _copilot_cli_add_dirs_from_env()
            allow_paths_value = kwargs.pop("copilot_allow_all_paths", None)
            copilot_allow_all_paths = (
                _copilot_allow_all_paths_default()
                if allow_paths_value is None
                else (
                    _truthy(str(allow_paths_value))
                    if isinstance(allow_paths_value, str)
                    else bool(allow_paths_value)
                )
            )

            needs_native = bool(
                trace_enabled
                or copilot_config_dir
                or copilot_log_dir
                or resume_session_id
                or continue_session
                or copilot_add_dirs
                or copilot_allow_all_paths
            )

            # Template mode: allow deterministic stubs like `bash -lc "echo OK"`.
            # This mode can't safely support session/resume/continue flags.
            rendered_template_mode = "{prompt}" in str(command or "")
            base_parts = shlex.split(str(command or ""))
            base_exe = str(base_parts[0] if base_parts else "").strip().lower()
            structured_ok = bool(base_exe in {"npx", "copilot"} and not rendered_template_mode)

            if not structured_ok:
                if needs_native:
                    raise RuntimeError(
                        "copilot_cli session/tracing flags require a real Copilot CLI command (e.g. `npx --yes @github/copilot` or `copilot`). "
                        "Unset ipfs_accelerate_py_COPILOT_CLI_CMD or set it to a real copilot command; current value appears to be a template/stub."
                    )
                return _clean_copilot_output(
                    _run_cli_command(
                        command,
                        prompt,
                        timeout_seconds=timeout,
                        template_vars={"model": model},
                        label="Copilot CLI",
                    )
                )

            def _utc_stamp() -> str:
                return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            share_path: Optional[str] = None
            if trace_enabled:
                share_base_dir: Optional[str] = None
                if isinstance(trace_dir, str) and trace_dir.strip():
                    share_base_dir = trace_dir.strip()
                elif isinstance(trace_jsonl_path, str) and trace_jsonl_path.strip():
                    share_base_dir = os.path.dirname(trace_jsonl_path.strip()) or "."
                if share_base_dir:
                    os.makedirs(share_base_dir, exist_ok=True)
                    share_path = os.path.join(
                        share_base_dir,
                        f"copilot_session_{_utc_stamp()}_{os.getpid()}.md",
                    )

            use_standalone_file_mode = bool(
                copilot_add_dirs or copilot_allow_all_paths
            )
            if use_standalone_file_mode:
                standalone = find_standalone_copilot_cli()
                if standalone is None:
                    raise RuntimeError(
                        "copilot CLI binary not found on PATH (required for "
                        "trusted-directory flags)"
                    )
                cmd = [
                    standalone,
                    "--silent",
                    "--stream",
                    "off",
                    "--allow-all-tools",
                    "--no-ask-user",
                    "--model",
                    model,
                    "--prompt",
                    str(prompt),
                ]
                if copilot_allow_all_paths:
                    cmd.insert(5, "--allow-all-paths")
                for add_dir in copilot_add_dirs:
                    cmd.extend(["--add-dir", add_dir])
            else:
                # Use `-i` (interactive with an auto-executed prompt) for
                # non-interactive worker subprocesses.
                cmd = list(base_parts)
                cmd.extend(
                    [
                        "--silent",
                        "--stream",
                        "off",
                        "--model",
                        model,
                        "-i",
                        str(prompt),
                    ]
                )

            if isinstance(copilot_config_dir, str) and copilot_config_dir.strip():
                cmd.extend(["--config-dir", copilot_config_dir.strip()])

            if isinstance(copilot_log_dir, str) and copilot_log_dir.strip():
                cmd.extend(["--log-dir", copilot_log_dir.strip()])
            elif trace_enabled and isinstance(trace_dir, str) and trace_dir.strip():
                cmd.extend(["--log-dir", trace_dir.strip()])

            appended_continue = False
            if isinstance(resume_session_id, str) and resume_session_id.strip():
                cmd.extend(["--resume", resume_session_id.strip()])
            elif continue_session:
                cmd.append("--continue")
                appended_continue = True

            if share_path:
                cmd.extend(["--share", share_path])

            def _run_copilot(command_list: list[str]) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    command_list,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=timeout,
                    env=os.environ.copy(),
                )

            proc = _run_copilot(cmd)
            if proc.returncode != 0 and appended_continue:
                msg = ((proc.stderr or "") or "").lower()
                retryable_continue = any(
                    s in msg
                    for s in (
                        "no session",
                        "no previous session",
                        "nothing to continue",
                        "cannot continue",
                        "could not continue",
                        "unable to continue",
                        "not found",
                    )
                )
                if retryable_continue:
                    cmd2 = [x for x in cmd if x != "--continue"]
                    proc2 = _run_copilot(cmd2)
                    if proc2.returncode == 0:
                        cmd = cmd2
                        proc = proc2

            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or "").strip() or "copilot CLI failed")

            cleaned = _clean_copilot_output(proc.stdout or "")

            if trace_enabled and isinstance(trace_jsonl_path, str) and trace_jsonl_path.strip():
                record = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "provider": "copilot_cli",
                    "model": model,
                    "cmd": cmd,
                    "share_path": share_path,
                    "stdout_chars": len(proc.stdout or ""),
                    "stderr_chars": len(proc.stderr or ""),
                }
                try:
                    os.makedirs(os.path.dirname(trace_jsonl_path.strip()) or ".", exist_ok=True)
                    with open(trace_jsonl_path.strip(), "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                except OSError:
                    pass

            return cleaned

        def generate_multimodal(
            self,
            prompt: str,
            *,
            model_name: Optional[str] = None,
            image_paths: Sequence[str] | None = None,
            image_urls: Sequence[str] | None = None,
            system_prompt: Optional[str] = None,
            additional_text_blocks: Sequence[str] | None = None,
            messages: Sequence[dict] | None = None,
            **kwargs: object,
        ) -> str:
            if not supports_image_inputs:
                raise LLMRouterError(
                    "copilot_cli multimodal path unavailable: installed "
                    "Copilot CLI does not advertise image input support"
                )
            if image_urls:
                raise LLMRouterError(
                    "copilot_cli multimodal path requires local image_paths; "
                    "image_urls are not supported"
                )
            copilot_cli = find_standalone_copilot_cli()
            if copilot_cli is None:
                raise LLMRouterError("Copilot CLI not found on PATH")

            call_options = dict(kwargs)
            timeout = float(call_options.pop("timeout", 180))
            model = (
                str(model_name or "").strip()
                or _coalesce_env(
                    "ipfs_accelerate_py_COPILOT_CLI_MODEL",
                    "IPFS_ACCELERATE_PY_COPILOT_CLI_MODEL",
                    "IPFS_DATASETS_PY_COPILOT_CLI_MODEL",
                )
                or _generic_llm_model_env()
                or "gpt-5-mini"
            )
            add_dirs = _normalize_copilot_add_dirs(
                call_options.pop("copilot_add_dirs", None)
            )
            if not add_dirs:
                add_dirs = _copilot_cli_add_dirs_from_env()
            allow_value = call_options.pop("copilot_allow_all_paths", None)
            allow_all_paths = (
                _copilot_allow_all_paths_default()
                if allow_value is None
                else (
                    _truthy(str(allow_value))
                    if isinstance(allow_value, str)
                    else bool(allow_value)
                )
            )

            prompt_sections: list[str] = []
            if system_prompt and str(system_prompt).strip():
                prompt_sections.append(str(system_prompt).strip())
            if messages:
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    role = str(message.get("role") or "user").strip()
                    content = message.get("content")
                    if isinstance(content, list):
                        text_parts = [
                            str(part.get("text") or "").strip()
                            for part in content
                            if isinstance(part, dict)
                            and str(part.get("type") or "").strip() == "text"
                            and str(part.get("text") or "").strip()
                        ]
                        rendered = "\n".join(text_parts)
                    else:
                        rendered = str(content or "").strip()
                    if rendered:
                        prompt_sections.append(f"{role}: {rendered}")
            else:
                prompt_sections.append(str(prompt or "").strip())
                prompt_sections.extend(
                    str(block).strip()
                    for block in additional_text_blocks or ()
                    if str(block or "").strip()
                )

            cmd = [
                copilot_cli,
                "--silent",
                "--stream",
                "off",
                "--allow-all-tools",
                "--no-ask-user",
                "--model",
                model,
                "--prompt",
            ]
            if allow_all_paths:
                cmd.insert(5, "--allow-all-paths")
            image_dirs: list[str] = []
            for image_path in image_paths or ():
                candidate = str(image_path or "").strip()
                if not candidate:
                    continue
                image_dir = os.path.abspath(os.path.dirname(candidate) or ".")
                if image_dir not in image_dirs:
                    image_dirs.append(image_dir)
                cmd.extend(["--image", candidate])
            for add_dir in [*add_dirs, *image_dirs]:
                if add_dir:
                    cmd.extend(["--add-dir", add_dir])
            cmd.append(
                "\n\n".join(
                    section for section in prompt_sections if section
                ).strip()
            )
            try:
                proc = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=timeout,
                    env=os.environ.copy(),
                )
            except FileNotFoundError as exc:
                raise LLMRouterError("Copilot CLI not found on PATH") from exc
            if proc.returncode != 0:
                raise LLMRouterError(
                    (proc.stderr or "").strip() or "copilot CLI failed"
                )
            return _clean_copilot_output(proc.stdout or "")

    return _CopilotCLIProvider()


def _get_copilot_sdk_provider() -> Optional[LLMProvider]:
    try:
        import copilot  # type: ignore
    except Exception:
        return None

    class _CopilotSDKProvider:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            _ = model_name
            model = os.environ.get("ipfs_accelerate_py_COPILOT_SDK_MODEL", "").strip()
            timeout_seconds = float(os.environ.get("ipfs_accelerate_py_COPILOT_SDK_TIMEOUT", "120"))

            async def _run() -> str:
                options = {}
                client = copilot.CopilotClient(options or None)
                await client.start()
                if model:
                    session = await client.create_session({"model": model})
                else:
                    session = await client.create_session()
                try:
                    event = await session.send_and_wait({"prompt": prompt})
                    if event and getattr(event, "data", None) is not None:
                        content = getattr(event.data, "content", None)
                        if content is not None:
                            return str(content)
                    return ""
                finally:
                    await session.destroy()
                    await client.stop()

            try:
                from ipfs_accelerate_py.utils.anyio_compat import AsyncContextError, fail_after, run as run_anyio

                async def _run_with_timeout() -> str:
                    with fail_after(timeout_seconds):
                        return await _run()

                return run_anyio(_run_with_timeout())
            except AsyncContextError:
                raise RuntimeError("copilot-sdk requires a non-running event loop context")

    return _CopilotSDKProvider()


def _get_gemini_cli_provider() -> Optional[LLMProvider]:
    command = os.environ.get("ipfs_accelerate_py_GEMINI_CLI_CMD", "npx @google/gemini-cli {prompt}")
    if not _cli_available(command):
        return None

    class _GeminiCLIProvider:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            _ = model_name
            timeout = float(kwargs.get("timeout", 180))

            gemini_cmd = kwargs.pop("gemini_cmd", None)
            if isinstance(gemini_cmd, list) and gemini_cmd:
                base_cmd = [str(x) for x in gemini_cmd]
                rendered = None
            elif isinstance(gemini_cmd, str) and gemini_cmd.strip():
                rendered = gemini_cmd.strip()
                base_cmd = shlex.split(rendered)
            else:
                rendered = command
                base_cmd = shlex.split(rendered)

            def _run(cmd_list: list[str]) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    cmd_list,
                    input=str(prompt),
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=timeout,
                    env=os.environ.copy(),
                )

            try:
                proc = _run(base_cmd)
            except FileNotFoundError as exc:
                raise LLMRouterError("Gemini CLI not found on PATH") from exc

            if proc.returncode == 0:
                return _clean_gemini_output(proc.stdout or "")

            stderr = (proc.stderr or "")
            # Known failure mode when running on Node.js v18.
            node18_regex_error = ("invalid regular expression flags" in stderr.lower()) and ("node.js v18" in stderr.lower())
            if node18_regex_error:
                try:
                    proc2 = _run(base_cmd)
                except FileNotFoundError as exc:
                    raise LLMRouterError("Gemini CLI not found on PATH") from exc
                if proc2.returncode == 0:
                    return _clean_gemini_output(proc2.stdout or "")

            raise LLMRouterError(stderr.strip() or "Gemini CLI failed")

    return _GeminiCLIProvider()


def _get_gemini_py_provider() -> Optional[LLMProvider]:
    try:
        from ipfs_accelerate_py.utils.gemini_cli import GeminiCLI
    except Exception:
        return None

    class _GeminiPyProvider:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            _ = model_name
            client = GeminiCLI(use_accelerate=_truthy(os.getenv("ipfs_accelerate_py_ENABLE_IPFS_ACCELERATE")))
            timeout = int(float(kwargs.get("timeout", 180)))
            result = client.execute(["generate", prompt], capture_output=True, timeout=timeout)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "Gemini (python wrapper) failed")
            return (result.stdout or "").strip()

    return _GeminiPyProvider()


def _get_claude_code_provider() -> Optional[LLMProvider]:
    command = os.environ.get("ipfs_accelerate_py_CLAUDE_CODE_CLI_CMD", "claude {prompt}")
    if not _cli_available(command):
        return None

    class _ClaudeCodeProvider:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            _ = model_name
            timeout = float(kwargs.get("timeout", 180))
            return _clean_claude_output(_run_cli_command(command, prompt, timeout_seconds=timeout, label="Claude Code CLI"))

    return _ClaudeCodeProvider()


def _get_claude_py_provider() -> Optional[LLMProvider]:
    try:
        from ipfs_accelerate_py.utils.claude_cli import ClaudeCLI
    except Exception:
        return None

    class _ClaudePyProvider:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            _ = model_name
            client = ClaudeCLI(use_accelerate=_truthy(os.getenv("ipfs_accelerate_py_ENABLE_IPFS_ACCELERATE")))
            timeout = int(float(kwargs.get("timeout", 180)))
            result = client.execute(["chat", prompt], capture_output=True, timeout=timeout)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "Claude (python wrapper) failed")
            return (result.stdout or "").strip()

    return _ClaudePyProvider()


def _get_mistral_vibe_provider(*, auto_install: bool = False) -> Optional[LLMProvider]:
    configured_command = _coalesce_env(
        "IPFS_ACCELERATE_MISTRAL_VIBE_CLI_CMD",
        "IPFS_ACCELERATE_PY_MISTRAL_VIBE_CLI_CMD",
        "ipfs_accelerate_py_MISTRAL_VIBE_CLI_CMD",
        "IPFS_DATASETS_PY_MISTRAL_VIBE_CLI_CMD",
    )
    command = configured_command or "vibe --prompt {prompt} --output text --max-turns 1"
    if not _cli_available(command):
        # A custom command is operator-owned; do not install a different CLI to
        # compensate for a misspelled or unavailable override.
        if configured_command:
            return None
        if not auto_install:
            return None
        install_result = ensure_mistral_vibe(auto_install=True)
        if not install_result.available:
            detail = install_result.reason or "installation did not produce a vibe executable"
            raise LLMRouterError(f"Mistral Vibe provider unavailable: {detail}")
        command = (
            f"{shlex.quote(install_result.executable)} "
            "--prompt {prompt} --output text --max-turns 1"
        )

    def _mistral_auth_available() -> bool:
        return mistral_vibe_auth_available()

    class _MistralVibeProvider:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            model = (
                model_name
                or _coalesce_env(
                    "IPFS_ACCELERATE_MISTRAL_VIBE_MODEL",
                    "IPFS_ACCELERATE_PY_MISTRAL_VIBE_MODEL",
                    "ipfs_accelerate_py_MISTRAL_VIBE_MODEL",
                    "IPFS_DATASETS_PY_MISTRAL_VIBE_MODEL",
                )
                or ""
            ).strip()
            timeout = float(kwargs.get("timeout", 240))
            agent = str(kwargs.pop("mistral_vibe_agent", "") or "").strip()
            is_leanstral = model.casefold() in {"leanstral", "labs-leanstral-1-5"}
            if is_leanstral and not agent:
                agent = "lean"
            if agent and not re.fullmatch(r"[A-Za-z0-9_-]+", agent):
                raise ValueError("mistral_vibe_agent must contain only letters, digits, underscores, or hyphens")
            # Vibe's lean agent installs and selects its own versioned Leanstral
            # model. An active-model environment override is validated before
            # the agent profile and causes a misleading fallback warning.
            vibe_active_model = "" if agent.casefold() == "lean" else model
            command_for_call = command
            if agent and "{agent}" not in command_for_call:
                command_for_call = f"{command_for_call} --agent {{agent}}"
            per_call_key = kwargs.pop("mistral_api_key", None)
            mistral_api_key = (
                str(per_call_key).strip()
                if per_call_key is not None and str(per_call_key).strip()
                else _coalesce_env(
                    "IPFS_ACCELERATE_MISTRAL_API_KEY",
                    "IPFS_ACCELERATE_PY_MISTRAL_API_KEY",
                    "ipfs_accelerate_py_MISTRAL_API_KEY",
                    "IPFS_DATASETS_PY_MISTRAL_API_KEY",
                    "MISTRAL_API_KEY",
                )
            )

            try:
                raw = _run_cli_command(
                    command_for_call,
                    prompt,
                    timeout_seconds=timeout,
                    template_vars={"agent": agent, "model": model},
                    label="Mistral Vibe CLI",
                    extra_env={
                        **({"MISTRAL_API_KEY": mistral_api_key} if mistral_api_key else {}),
                        "VIBE_ACTIVE_MODEL": vibe_active_model or None,
                    }
                )
            except LLMRouterError as exc:
                if not mistral_api_key and not _mistral_auth_available():
                    raise LLMRouterError(
                        "Mistral Vibe call failed and no local auth markers were found. "
                        "If you are logged in via a non-env auth flow, keep using it; otherwise set "
                        "MISTRAL_API_KEY (or IPFS_ACCELERATE_MISTRAL_API_KEY) or run 'vibe --setup'."
                    ) from exc
                _raise_mistral_vibe_access_error(exc)

            return _clean_mistral_vibe_output(raw)

    return _MistralVibeProvider()


def _get_grok_cli_provider() -> Optional[LLMProvider]:
    """Return the official Grok CLI provider when its binary is available."""

    command = _grok_cli_command()
    if not _cli_available(command):
        return None

    class _GrokCLIProvider:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            model = (
                (model_name or "").strip()
                or _coalesce_env(
                    "ipfs_accelerate_py_GROK_CLI_MODEL",
                    "IPFS_ACCELERATE_PY_GROK_CLI_MODEL",
                    "IPFS_DATASETS_PY_GROK_CLI_MODEL",
                    "GROK_CLI_MODEL",
                )
            )
            timeout = float(kwargs.pop("timeout", 180))
            trace_jsonl_path = kwargs.pop("trace_jsonl_path", None)
            trace_dir = kwargs.pop("trace_dir", None)
            trace_enabled = bool(kwargs.pop("trace", False) or trace_jsonl_path or trace_dir)

            command_override = kwargs.pop("grok_cli_cmd", None)
            if isinstance(command_override, list) and command_override:
                base_parts = [str(value) for value in command_override]
                command_text = ""
                structured_cli = True
            else:
                command_text = (
                    str(command_override).strip()
                    if isinstance(command_override, str) and command_override.strip()
                    else command
                )
                base_parts = shlex.split(command_text)
                executable_name = Path(base_parts[0]).name.lower() if base_parts else ""
                structured_cli = executable_name in {"grok", "agent"}

            extra_env: Dict[str, Optional[str]] = {}
            if not os.getenv("XAI_API_KEY", "").strip():
                alternate_key = _coalesce_env(
                    "ipfs_accelerate_py_XAI_API_KEY",
                    "IPFS_ACCELERATE_PY_XAI_API_KEY",
                    "IPFS_DATASETS_PY_XAI_API_KEY",
                )
                if alternate_key:
                    extra_env["XAI_API_KEY"] = alternate_key

            if not structured_cli:
                raw = _run_cli_command(
                    command_text,
                    prompt,
                    timeout_seconds=timeout,
                    template_vars={"model": model},
                    label="Grok CLI",
                    extra_env=extra_env,
                )
                payload = _grok_cli_json_payload(raw)
                if payload is not None:
                    if str(payload.get("type") or "").lower() == "error":
                        raise _grok_cli_error(raw, "")
                    text = payload.get("text")
                    if isinstance(text, str):
                        return _clean_grok_cli_output(text)
                return _clean_grok_cli_output(raw)

            if not base_parts:
                raise LLMRouterError("Grok CLI command is empty")

            cmd = list(base_parts)
            if model and "--model" not in cmd and "-m" not in cmd:
                cmd.extend(["--model", model])
            if "--output-format" not in cmd:
                cmd.extend(["--output-format", "json"])
            if "--no-plan" not in cmd:
                cmd.append("--no-plan")
            if "--no-subagents" not in cmd:
                cmd.append("--no-subagents")
            if "--disable-web-search" not in cmd:
                cmd.append("--disable-web-search")
            if "--no-memory" not in cmd:
                cmd.append("--no-memory")
            if "--verbatim" not in cmd:
                cmd.append("--verbatim")

            max_turns = max(
                1,
                int(
                    kwargs.pop(
                        "grok_max_turns",
                        _coalesce_env(
                            "ipfs_accelerate_py_GROK_CLI_MAX_TURNS",
                            "IPFS_ACCELERATE_PY_GROK_CLI_MAX_TURNS",
                        )
                        or "1",
                    )
                ),
            )
            if "--max-turns" not in cmd:
                cmd.extend(["--max-turns", str(max_turns)])

            permission_mode = str(
                kwargs.pop(
                    "grok_permission_mode",
                    _coalesce_env(
                        "ipfs_accelerate_py_GROK_CLI_PERMISSION_MODE",
                        "IPFS_ACCELERATE_PY_GROK_CLI_PERMISSION_MODE",
                    )
                    or "dontAsk",
                )
            ).strip()
            if permission_mode and "--permission-mode" not in cmd:
                cmd.extend(["--permission-mode", permission_mode])

            tools = kwargs.pop(
                "grok_tools",
                os.getenv(
                    "ipfs_accelerate_py_GROK_CLI_TOOLS",
                    os.getenv("IPFS_ACCELERATE_PY_GROK_CLI_TOOLS", ""),
                ),
            )
            if "--tools" not in cmd:
                cmd.extend(["--tools", str(tools or "")])

            reasoning_effort = str(
                kwargs.pop(
                    "reasoning_effort",
                    _coalesce_env(
                        "ipfs_accelerate_py_GROK_CLI_REASONING_EFFORT",
                        "IPFS_ACCELERATE_PY_GROK_CLI_REASONING_EFFORT",
                    ),
                )
                or ""
            ).strip()
            if reasoning_effort and "--reasoning-effort" not in cmd and "--effort" not in cmd:
                cmd.extend(["--reasoning-effort", reasoning_effort])

            resume_session_id = str(kwargs.pop("resume_session_id", "") or "").strip()
            continue_session = bool(kwargs.pop("continue_session", False))
            chat_session_id = str(kwargs.pop("chat_session_id", "") or "").strip()
            if resume_session_id:
                cmd.extend(["--resume", resume_session_id])
            elif continue_session:
                cmd.append("--continue")
            elif chat_session_id:
                cmd.extend(["--session-id", chat_session_id])

            prompt_path = ""
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    prefix="llm-router-grok-prompt-",
                    suffix=".txt",
                    delete=False,
                ) as prompt_file:
                    prompt_file.write(str(prompt))
                    prompt_path = prompt_file.name
                cmd.extend(["--prompt-file", prompt_path])

                env = os.environ.copy()
                for key, value in extra_env.items():
                    if value is not None:
                        env[key] = value
                try:
                    proc = subprocess.run(
                        cmd,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=timeout,
                        env=env,
                    )
                except FileNotFoundError as exc:
                    raise LLMRouterError("Grok CLI not found on PATH") from exc
            finally:
                if prompt_path:
                    try:
                        os.unlink(prompt_path)
                    except OSError:
                        pass

            payload = _grok_cli_json_payload(proc.stdout or "")
            if proc.returncode != 0:
                raise _grok_cli_error(proc.stdout or "", proc.stderr or "")
            if payload is not None and str(payload.get("type") or "").lower() == "error":
                raise _grok_cli_error(proc.stdout or "", proc.stderr or "")

            if trace_enabled:
                trace_path = ""
                if isinstance(trace_jsonl_path, str) and trace_jsonl_path.strip():
                    trace_path = trace_jsonl_path.strip()
                elif isinstance(trace_dir, (str, Path)) and str(trace_dir).strip():
                    trace_path = os.path.join(str(trace_dir).strip(), "grok_cli_trace.jsonl")
                if trace_path:
                    record: dict[str, object] = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "provider": "grok_cli",
                        "model": model,
                        "cmd": _redact_grok_cli_command(cmd, prompt),
                        "stdout_chars": len(proc.stdout or ""),
                        "stderr_chars": len(proc.stderr or ""),
                    }
                    if payload is not None:
                        for key in (
                            "stopReason",
                            "sessionId",
                            "requestId",
                            "num_turns",
                            "usage",
                            "modelUsage",
                            "total_cost_usd",
                            "total_cost_usd_ticks",
                            "cost_is_partial",
                            "usage_is_incomplete",
                        ):
                            if key in payload:
                                record[key] = payload[key]
                    try:
                        os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
                        with open(trace_path, "a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    except OSError:
                        pass

            if payload is not None:
                text = payload.get("text")
                if isinstance(text, str):
                    cleaned_text = _clean_grok_cli_output(text)
                    if cleaned_text:
                        return cleaned_text
                    raise LLMRouterError("Grok CLI returned no response text")
            cleaned = _clean_grok_cli_output(proc.stdout or "")
            if cleaned:
                return cleaned
            raise LLMRouterError("Grok CLI returned no response text")

    return _GrokCLIProvider()


def _get_xai_provider() -> Optional[LLMProvider]:
    """Return an xAI Grok provider if XAI_API_KEY (or equivalent) is set."""
    api_key = _coalesce_env("XAI_API_KEY", "ipfs_accelerate_py_XAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("ipfs_accelerate_py_XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/")

    def _request(payload: dict, *, timeout: float) -> dict:
        import urllib.request
        import urllib.error

        url = f"{base_url}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"xAI HTTP {exc.code}: {detail or exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"xAI request failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except Exception as exc:
            raise RuntimeError("xAI returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("xAI returned invalid JSON")
        return data

    class _XAIProvider:
        def chat_completions(
            self,
            messages: Sequence[ChatMessage],
            *,
            model_name: Optional[str] = None,
            **kwargs: object,
        ) -> dict:
            model = (
                model_name
                or os.getenv("ipfs_accelerate_py_XAI_MODEL")
                or os.getenv("ipfs_accelerate_py_LLM_MODEL")
                or "grok-3"
            )
            max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256))
            temperature = kwargs.get("temperature", 0.2)
            payload: dict = {
                "model": model,
                "messages": list(messages),
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
            }
            if "logprobs" in kwargs:
                payload["logprobs"] = bool(kwargs.get("logprobs"))
            if "top_logprobs" in kwargs and kwargs.get("top_logprobs") is not None:
                payload["top_logprobs"] = int(kwargs.get("top_logprobs"))
            if "response_format" in kwargs and kwargs.get("response_format") is not None:
                payload["response_format"] = kwargs.get("response_format")
            if "seed" in kwargs and kwargs.get("seed") is not None:
                payload["seed"] = int(kwargs.get("seed"))
            timeout = float(kwargs.get("timeout", 120))
            return _request(payload, timeout=timeout)

        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            data = self.chat_completions(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                **kwargs,
            )
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"].strip()
                text = choices[0].get("text") if isinstance(choices[0], dict) else None
                if isinstance(text, str):
                    return text.strip()
            raise RuntimeError("xAI response missing choices")

    return _XAIProvider()


def _get_meta_ai_provider() -> Optional[LLMProvider]:
    """Return the Meta Model API provider when a credential is available."""

    api_key = resolve_meta_model_api_key()
    if not api_key:
        return None

    base_url = os.getenv(
        "ipfs_accelerate_py_META_AI_BASE_URL",
        META_MODEL_API_BASE_URL,
    ).rstrip("/")

    def _request(payload: dict, *, timeout: float) -> dict:
        import urllib.request
        import urllib.error

        url = f"{base_url}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"Meta AI HTTP {exc.code}: {detail or exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"Meta AI request failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except Exception as exc:
            raise RuntimeError("Meta AI returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Meta AI returned invalid JSON")
        return data

    class _MetaAIProvider:
        def chat_completions(
            self,
            messages: Sequence[ChatMessage],
            *,
            model_name: Optional[str] = None,
            **kwargs: object,
        ) -> dict:
            model = (
                model_name
                or os.getenv("ipfs_accelerate_py_META_AI_MODEL")
                or os.getenv("ipfs_accelerate_py_LLM_MODEL")
                or META_MODEL_API_DEFAULT_MODEL
            )
            model = normalize_meta_model_name(model)
            default_max = int(
                os.getenv("ipfs_accelerate_py_META_AI_MAX_COMPLETION_TOKENS", "1024")
            )
            max_completion_tokens = kwargs.get(
                "max_completion_tokens",
                kwargs.get("max_tokens", kwargs.get("max_new_tokens", default_max)),
            )
            temperature = kwargs.get("temperature", 0.2)
            payload: dict = {
                "model": model,
                "messages": list(messages),
                "max_completion_tokens": int(max_completion_tokens),
                "temperature": float(temperature),
            }
            if "logprobs" in kwargs:
                payload["logprobs"] = bool(kwargs.get("logprobs"))
            if "top_logprobs" in kwargs and kwargs.get("top_logprobs") is not None:
                payload["top_logprobs"] = int(kwargs.get("top_logprobs"))
            if "response_format" in kwargs and kwargs.get("response_format") is not None:
                payload["response_format"] = kwargs.get("response_format")
            if "seed" in kwargs and kwargs.get("seed") is not None:
                payload["seed"] = int(kwargs.get("seed"))
            timeout = float(kwargs.get("timeout", 120))
            return _request(payload, timeout=timeout)

        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            if (
                "max_completion_tokens" not in kwargs
                and "max_tokens" not in kwargs
                and "max_new_tokens" not in kwargs
            ):
                kwargs = {
                    **dict(kwargs),
                    "max_completion_tokens": int(
                        os.getenv(
                            "ipfs_accelerate_py_META_AI_MAX_COMPLETION_TOKENS",
                            "1024",
                        )
                    ),
                }
            data = self.chat_completions(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                **kwargs,
            )
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                text = choices[0].get("text") if isinstance(choices[0], dict) else None
                if isinstance(text, str) and text.strip():
                    return text.strip()
                raise RuntimeError(
                    "Meta AI response missing content "
                    f"(finish_reason={choices[0].get('finish_reason') if isinstance(choices[0], dict) else None})"
                )
            raise RuntimeError("Meta AI response missing choices")

    return _MetaAIProvider()


class _PinnedSymaiNoRedirect(urllib.request.HTTPRedirectHandler):
    """Reject redirects so the pinned service cannot change endpoints."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> None:
        del request, file_pointer, code, message, headers, new_url
        return None


def _pinned_symai_urlopen(
    request: urllib.request.Request,
    *,
    timeout: float,
) -> object:
    """Open one exact URL without ambient proxy or redirect behavior."""

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _PinnedSymaiNoRedirect(),
    )
    response = opener.open(request, timeout=timeout)
    try:
        final_url = response.geturl()
    except Exception:
        response.close()
        raise
    if final_url != request.full_url:
        response.close()
        raise RuntimeError("pinned Leanstral service changed its final URL")
    return response


def _bounded_json_response(
    request: urllib.request.Request,
    *,
    timeout: float,
    max_bytes: int,
) -> dict[str, object]:
    try:
        with _pinned_symai_urlopen(request, timeout=timeout) as response:
            raw = response.read(max_bytes + 1)
    except Exception as exc:
        raise RuntimeError(
            f"pinned Leanstral service request failed: {type(exc).__name__}"
        ) from exc
    if len(raw) > max_bytes:
        raise RuntimeError(
            "pinned Leanstral service response exceeded byte limit"
        )

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject_nonfinite(value: str) -> object:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            "pinned Leanstral service returned invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError("pinned Leanstral service returned a non-object")
    return value


def _served_model_ids(value: dict[str, object]) -> list[str]:
    records = value.get("data", value.get("models"))
    if not isinstance(records, list):
        raise RuntimeError("pinned Leanstral service omitted its model list")
    model_ids: list[str] = []
    for record in records:
        if isinstance(record, str):
            model_id = record
        elif isinstance(record, dict):
            model_id = str(
                record.get("id")
                or record.get("model")
                or record.get("name")
                or ""
            )
        else:
            continue
        if model_id:
            model_ids.append(model_id)
    return model_ids


def _generate_pinned_symai_leanstral(
    prompt: str,
    *,
    kwargs: dict[str, object],
) -> tuple[str, dict[str, str]]:
    """Use the already-running frozen Leanstral service without fallback."""

    response_format = kwargs.get("response_format")
    if (
        not isinstance(response_format, dict)
        or not any(
            response_format == allowed
            for allowed in _PINNED_SYMAI_ALLOWED_RESPONSE_FORMATS
        )
    ):
        raise RuntimeError(
            "pinned SyMAI Leanstral route requires a supported frozen "
            "JSON-schema contract"
        )

    timeout = max(1.0, min(float(kwargs.get("timeout", 30.0)), 60.0))
    deadline = time.monotonic() + timeout

    def remaining_timeout() -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "pinned SyMAI Leanstral aggregate timeout expired"
            )
        return remaining

    models_request = urllib.request.Request(
        f"{_PINNED_SYMAI_LEANSTRAL_ENDPOINT}/models",
        headers={"Accept": "application/json"},
        method="GET",
    )
    models = _bounded_json_response(
        models_request,
        timeout=remaining_timeout(),
        max_bytes=1024 * 1024,
    )
    served_ids = _served_model_ids(models)
    if served_ids.count(_PINNED_SYMAI_LEANSTRAL_MODEL) != 1:
        raise RuntimeError(
            "pinned Leanstral model is absent or ambiguous at the "
            "frozen endpoint"
        )

    max_tokens = int(
        kwargs.get("max_tokens", kwargs.get("max_new_tokens", 512))
    )
    if not 1 <= max_tokens <= 512:
        raise RuntimeError("pinned SyMAI Leanstral token bound is invalid")
    temperature = float(kwargs.get("temperature", 0.0))
    if temperature != 0.0:
        raise RuntimeError(
            "pinned SyMAI Leanstral route requires temperature zero"
        )
    payload: dict[str, object] = {
        "model": _PINNED_SYMAI_LEANSTRAL_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return exactly one valid JSON object. Do not use "
                    "Markdown fences or emit any text before or after the "
                    "object."
                ),
            },
            {"role": "user", "content": str(prompt)},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "cache_prompt": False,
        "response_format": response_format,
        "seed": 0,
        "stop": ["<|im_end|>", "<|tool_call_end|>", "<|im_start|>"],
    }
    if kwargs.get("top_p") is not None:
        payload["top_p"] = kwargs["top_p"]
    completion_request = urllib.request.Request(
        f"{_PINNED_SYMAI_LEANSTRAL_ENDPOINT}/chat/completions",
        data=json.dumps(
            payload, separators=(",", ":"), sort_keys=True
        ).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    completion = _bounded_json_response(
        completion_request,
        timeout=remaining_timeout(),
        max_bytes=64 * 1024,
    )
    if str(completion.get("model") or "") != _PINNED_SYMAI_LEANSTRAL_MODEL:
        raise RuntimeError(
            "pinned Leanstral service returned a different model"
        )
    choices = completion.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError(
            "pinned Leanstral service returned invalid choices"
        )
    first = choices[0]
    if (
        isinstance(first, dict)
        and first.get("finish_reason") == "length"
    ):
        raise PinnedSymaiCompletionError(
            PinnedSymaiCompletionError.OUTPUT_TOKEN_LIMIT
        )
    if not isinstance(first, dict) or first.get("finish_reason") != "stop":
        raise RuntimeError(
            "pinned Leanstral service returned an invalid choice"
        )
    message = first.get("message")
    text = (
        message.get("content")
        if isinstance(message, dict)
        else first.get("text")
    )
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("pinned Leanstral service returned empty text")
    return text.strip(), dict(_PINNED_SYMAI_ROUTE_BINDING)


def _get_accelerate_provider(deps: RouterDeps) -> Optional[LLMProvider]:
    enable_value = (
        os.getenv("ipfs_accelerate_py_ENABLE_IPFS_ACCELERATE")
        or os.getenv("IPFS_ACCELERATE_PY_ENABLE_IPFS_ACCELERATE")
        or os.getenv("IPFS_DATASETS_PY_ENABLE_IPFS_ACCELERATE")
    )
    if enable_value is not None and enable_value.strip() and not _truthy(enable_value):
        return None

    try:
        manager_factory = getattr(deps, "get_accelerate_manager", None)
        if callable(manager_factory):
            manager = manager_factory(
                purpose="llm_router",
                enable_distributed=True,
                resources={"purpose": "llm_router"},
            )
        else:
            # Compatibility for callers using this canonical router directly:
            # the datasets integration remains an optional manager provider,
            # not a second router implementation.
            manager_module = importlib.import_module(
                "ipfs_datasets_py.ml.accelerate_integration.manager"
            )
            manager_class = getattr(manager_module, "AccelerateManager", None)
            manager = manager_class() if callable(manager_class) else None
    except Exception:
        return None
    if manager is None:
        return None

    def _extract_generated_text(result: object) -> Optional[str]:
        if isinstance(result, str) and result.strip():
            return result
        if isinstance(result, list):
            for item in result:
                text = _extract_generated_text(item)
                if text:
                    return text
            return None
        if not isinstance(result, dict):
            return None
        if "heartbeat_ts" in result or (
            "phase" in result and "worker_id" in result
        ):
            return None
        for key in (
            "text",
            "generated_text",
            "output_text",
            "completion",
            "content",
        ):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
        choices = result.get("choices")
        if isinstance(choices, list):
            text = _extract_generated_text(choices)
            if text:
                return text
        for key in ("result", "data", "output", "response", "payload"):
            text = _extract_generated_text(result.get(key))
            if text:
                return text
        message = result.get("message")
        if isinstance(message, dict):
            text = _extract_generated_text(message)
            if text:
                return text
        return None

    def _image_path_to_data_url(path_value: str) -> str:
        import base64
        import mimetypes

        path = Path(path_value).expanduser()
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    class _AccelerateLLMProvider:
        router_provider_name = "accelerate"

        def __init__(self) -> None:
            self._generation_trace = threading.local()

        def get_last_generation_trace(self) -> dict[str, str]:
            value = getattr(self._generation_trace, "payload", None)
            return dict(value) if isinstance(value, dict) else {}

        def generate(
            self,
            prompt: str,
            *,
            model_name: Optional[str] = None,
            **kwargs: object,
        ) -> str:
            effective_model = model_name or _coalesce_env(
                "ipfs_accelerate_py_LLM_MODEL",
                "IPFS_ACCELERATE_PY_LLM_MODEL",
                "IPFS_DATASETS_PY_LLM_MODEL",
            )
            self._generation_trace.payload = {}
            route_binding = kwargs.pop(_SYMAI_ROUTE_BINDING_KWARG, None)
            if route_binding is not None:
                if effective_model != _PINNED_SYMAI_LEANSTRAL_ALIAS:
                    raise RuntimeError(
                        "private SyMAI route binding used with the wrong "
                        "model alias"
                    )
                if (
                    not isinstance(route_binding, dict)
                    or route_binding != _PINNED_SYMAI_ROUTE_BINDING
                ):
                    raise RuntimeError(
                        "private SyMAI route binding is incomplete or drifted"
                    )
                text, trace = _generate_pinned_symai_leanstral(
                    prompt,
                    kwargs=dict(kwargs),
                )
                self._generation_trace.payload = trace
                return text

            payload = {"prompt": prompt, **kwargs}
            result = manager.run_inference(
                effective_model,
                payload,
                task_type="text-generation",
            )
            if isinstance(result, dict):
                backend_name = result.get("backend")
                resolved_model = result.get("model")
                if isinstance(backend_name, str) and backend_name:
                    self._generation_trace.payload[
                        "routing_backend"
                    ] = backend_name
                if isinstance(resolved_model, str) and resolved_model:
                    self._generation_trace.payload[
                        "resolved_model_name"
                    ] = resolved_model
            text = _extract_generated_text(result)
            if text:
                return text
            if isinstance(result, dict) and isinstance(
                result.get("message"), str
            ):
                raise RuntimeError(str(result["message"]))
            raise RuntimeError(
                "AccelerateManager provider did not return generated text"
            )

        def generate_multimodal(
            self,
            prompt: str,
            *,
            model_name: Optional[str] = None,
            image_paths: Sequence[str] | None = None,
            image_urls: Sequence[str] | None = None,
            system_prompt: Optional[str] = None,
            additional_text_blocks: Sequence[str] | None = None,
            messages: Sequence[dict] | None = None,
            **kwargs: object,
        ) -> str:
            call_options = dict(kwargs)
            payload: dict[str, object] = {
                "prompt": str(prompt or ""),
                "image_urls": [
                    str(url)
                    for url in image_urls or ()
                    if str(url or "").strip()
                ],
                "image_data_urls": [
                    _image_path_to_data_url(path) for path in image_paths or ()
                ],
                "system_prompt": system_prompt,
                "additional_text_blocks": [
                    str(block)
                    for block in additional_text_blocks or ()
                    if str(block or "").strip()
                ],
            }
            if messages is not None:
                payload["messages"] = list(messages)
            for key in (
                "max_tokens",
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "timeout",
                "image_detail",
            ):
                if key in call_options:
                    payload[key] = call_options[key]
            result = manager.run_inference(
                model_name
                or _coalesce_env(
                    "ipfs_accelerate_py_LLM_MODEL",
                    "IPFS_ACCELERATE_PY_LLM_MODEL",
                    "IPFS_DATASETS_PY_LLM_MODEL",
                ),
                payload,
                task_type="multimodal-generation",
                **call_options,
            )
            text = _extract_generated_text(result)
            if text:
                return text
            raise RuntimeError(
                "AccelerateManager multimodal provider did not return generated text"
            )

    return _AccelerateLLMProvider()


def _provider_cache_key() -> tuple:
    # Include only env vars that change provider resolution.
    return (
        os.getenv("ipfs_accelerate_py_LLM_PROVIDER", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_LLM_PROVIDER", "").strip(),
        os.getenv("IPFS_DATASETS_PY_LLM_PROVIDER", "").strip(),
        os.getenv("ipfs_accelerate_py_ENABLE_IPFS_ACCELERATE", "").strip(),
        os.getenv("IPFS_DATASETS_PY_ENABLE_IPFS_ACCELERATE", "").strip(),
        os.getenv("ipfs_accelerate_py_OPENROUTER_API_KEY", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_API_KEY", "").strip(),
        os.getenv("IPFS_DATASETS_PY_OPENROUTER_API_KEY", "").strip(),
        os.getenv("OPENROUTER_API_KEY", "").strip(),
        os.getenv("ipfs_accelerate_py_OPENROUTER_MODEL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_MODEL", "").strip(),
        os.getenv("IPFS_DATASETS_PY_OPENROUTER_MODEL", "").strip(),
        os.getenv("ipfs_accelerate_py_OPENROUTER_BASE_URL", "").strip(),
        os.getenv("IPFS_DATASETS_PY_OPENROUTER_BASE_URL", "").strip(),
        bool(_resolve_openai_api_key()),
        _hf_token_fingerprint(),
        os.getenv("IPFS_ACCELERATE_PY_HF_INFERENCE_MODEL", "").strip(),
        os.getenv("IPFS_DATASETS_PY_HF_INFERENCE_MODEL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_HF_INFERENCE_BASE_URL", "").strip(),
        os.getenv("IPFS_DATASETS_PY_HF_INFERENCE_BASE_URL", "").strip(),
        os.getenv("ipfs_accelerate_py_CODEX_CLI_MODEL", "").strip(),
        os.getenv("ipfs_accelerate_py_CODEX_MODEL", "").strip(),
        os.getenv("ipfs_accelerate_py_COPILOT_CLI_CMD", "").strip(),
        os.getenv("ipfs_accelerate_py_GEMINI_CLI_CMD", "").strip(),
        os.getenv("ipfs_accelerate_py_GROK_CLI_CMD", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_GROK_CLI_CMD", "").strip(),
        os.getenv("IPFS_DATASETS_PY_GROK_CLI_CMD", "").strip(),
        os.getenv("GROK_CLI_CMD", "").strip(),
        os.getenv("ipfs_accelerate_py_GROK_CLI_MODEL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_GROK_CLI_MODEL", "").strip(),
        os.getenv("IPFS_DATASETS_PY_GROK_CLI_MODEL", "").strip(),
        os.getenv("GROK_CLI_MODEL", "").strip(),
        os.getenv("GROK_HOME", "").strip(),
        os.getenv("GROK_AUTH_PROVIDER_COMMAND", "").strip(),
        _grok_cli_auth_fingerprint(),
        os.getenv("ipfs_accelerate_py_CLAUDE_CODE_CLI_CMD", "").strip(),
        os.getenv("ipfs_accelerate_py_MISTRAL_VIBE_CLI_CMD", "").strip(),
        os.getenv("ipfs_accelerate_py_MISTRAL_VIBE_MODEL", "").strip(),
        os.getenv("XAI_API_KEY", "").strip(),
        os.getenv("ipfs_accelerate_py_XAI_API_KEY", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_XAI_API_KEY", "").strip(),
        os.getenv("IPFS_DATASETS_PY_XAI_API_KEY", "").strip(),
        os.getenv("ipfs_accelerate_py_XAI_MODEL", "").strip(),
        os.getenv("ipfs_accelerate_py_XAI_BASE_URL", "").strip(),
        meta_model_api_key_fingerprint(),
        os.getenv("ipfs_accelerate_py_META_AI_MODEL", "").strip(),
        os.getenv("ipfs_accelerate_py_META_AI_BASE_URL", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_BASE_URL", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_REF", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_HF_FILE", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_AUTOSTART", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_PREFETCH_MODEL", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_AUTO_INSTALL", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_AUTO_UPDATE", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_HOST", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_PORT", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_CONTEXT_SIZE", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_THREADS", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_GPU_LAYERS", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_PATH", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_MODEL_PATH", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_MODEL_REF", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_HF_FILE", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_CONTEXT_SIZE", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_THREADS", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_GPU_LAYERS", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_AUTO_INSTALL", "").strip(),
        os.getenv("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_PACKAGE", "").strip(),
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

        def _prepare_prompt(
            self,
            *,
            pipe: object,
            prompt: str,
            max_new_tokens: int,
        ) -> tuple[str, int]:
            tokenizer = getattr(pipe, "tokenizer", None)
            if tokenizer is None:
                return (
                    prompt[-8000:] if len(prompt) > 8000 else prompt,
                    max(1, min(max_new_tokens, 256)),
                )
            model_max_length = getattr(tokenizer, "model_max_length", None)
            if (
                not isinstance(model_max_length, int)
                or model_max_length <= 0
                or model_max_length > 100_000
            ):
                model_max_length = 1024
            safe_max_new_tokens = max(
                1,
                min(
                    max_new_tokens,
                    max(8, model_max_length // 4),
                    max(1, model_max_length // 2),
                ),
            )
            prompt_budget = max(1, model_max_length - safe_max_new_tokens)
            try:
                encoded = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=prompt_budget,
                    return_tensors=None,
                )
                input_ids = (
                    encoded.get("input_ids") if isinstance(encoded, dict) else None
                )
                if input_ids:
                    return (
                        tokenizer.decode(input_ids, skip_special_tokens=False),
                        safe_max_new_tokens,
                    )
            except Exception:
                pass
            return prompt[-max(512, prompt_budget * 4) :], safe_max_new_tokens

        def _retry_after_context_error(
            self,
            *,
            pipe: object,
            prompt: str,
            max_new_tokens: int,
        ) -> tuple[str, int]:
            tokenizer = getattr(pipe, "tokenizer", None)
            model_max_length = getattr(tokenizer, "model_max_length", None)
            if (
                not isinstance(model_max_length, int)
                or model_max_length <= 0
                or model_max_length > 100_000
            ):
                model_max_length = 1024
            retry_tokens = max(
                1,
                min(max_new_tokens, 32, max(1, model_max_length // 8)),
            )
            prompt_budget = max(1, min(128, max(1, model_max_length // 4)))
            if tokenizer is not None:
                try:
                    encoded = tokenizer(
                        prompt,
                        truncation=True,
                        max_length=prompt_budget,
                        return_tensors=None,
                    )
                    input_ids = (
                        encoded.get("input_ids")
                        if isinstance(encoded, dict)
                        else None
                    )
                    if input_ids:
                        return (
                            tokenizer.decode(
                                input_ids,
                                skip_special_tokens=False,
                            ),
                            retry_tokens,
                        )
                except Exception:
                    pass
            return prompt[-max(256, prompt_budget * 4) :], retry_tokens

        def generate(self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object) -> str:
            model = model_name or _generic_llm_model_env() or "gpt2"
            pipe = self._pipelines.get(model)
            if pipe is None:
                pipe = pipeline("text-generation", model=model)
                self._pipelines[model] = pipe

            max_new_tokens = int(kwargs.pop("max_new_tokens", kwargs.pop("max_tokens", 128)))
            prepared_prompt, safe_tokens = self._prepare_prompt(
                pipe=pipe,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            try:
                out = pipe(prepared_prompt, max_new_tokens=safe_tokens)
            except (IndexError, RuntimeError) as exc:
                if "index out of range" not in str(exc).lower():
                    raise
                retry_prompt, retry_tokens = self._retry_after_context_error(
                    pipe=pipe,
                    prompt=prompt,
                    max_new_tokens=safe_tokens,
                )
                out = pipe(retry_prompt, max_new_tokens=retry_tokens)
            if isinstance(out, list) and out:
                item = out[0]
                if isinstance(item, dict) and isinstance(item.get("generated_text"), str):
                    return item["generated_text"]
            return str(out)

    return _LocalHFProvider()


def _extract_generated_text_from_task_result(result: object) -> Optional[str]:
    """Extract text from the result shapes accepted by task-queue workers."""

    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return None
    for key in ("text", "generated_text", "output", "response", "content"):
        value = result.get(key)
        if isinstance(value, str):
            return value
    choices = result.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        first = choices[0]
        value = first.get("text")
        if isinstance(value, str):
            return value
        message = first.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    return None


def _get_p2p_task_queue_provider() -> LLMProvider:
    """Return the canonical TaskQueue-backed text-generation provider."""

    class _P2PTaskQueueProvider:
        router_provider_name = "p2p_task_queue"

        def generate(
            self,
            prompt: str,
            *,
            model_name: Optional[str] = None,
            **kwargs: object,
        ) -> str:
            call_options = dict(kwargs)
            queue_path = str(
                call_options.pop("queue_path", None)
                or call_options.pop("task_queue_path", None)
                or _coalesce_env(
                    "IPFS_ACCELERATE_PY_TASK_QUEUE_PATH",
                    "IPFS_DATASETS_PY_TASK_QUEUE_PATH",
                )
                or ""
            ).strip()
            timeout_value = (
                call_options.pop("wait_timeout_s", None)
                or call_options.pop("task_timeout_s", None)
                or call_options.pop("timeout_s", None)
                or call_options.pop("timeout", None)
                or _coalesce_env(
                    "IPFS_ACCELERATE_PY_TASK_QUEUE_WAIT_TIMEOUT_S",
                    "IPFS_DATASETS_PY_TASK_QUEUE_WAIT_TIMEOUT_S",
                )
                or 60.0
            )
            task_type = str(
                call_options.pop("task_type", None) or "text-generation"
            ).strip() or "text-generation"
            task_id = submit_task(
                prompt=str(prompt or ""),
                model_name=model_name
                or _coalesce_env(
                    "ipfs_accelerate_py_LLM_MODEL",
                    "IPFS_ACCELERATE_PY_LLM_MODEL",
                    "IPFS_DATASETS_PY_LLM_MODEL",
                )
                or "gpt2",
                task_type=task_type,
                queue_path=queue_path or None,
                **call_options,
            )
            task = wait_task(
                str(task_id),
                queue_path=queue_path or None,
                timeout_s=float(timeout_value),
            )
            if not isinstance(task, dict):
                raise LLMRouterError(
                    f"TaskQueue task did not complete before timeout: {task_id}"
                )
            status = str(task.get("status") or "").strip().lower()
            if status != "completed":
                error = str(
                    task.get("error")
                    or task.get("message")
                    or status
                    or "unknown error"
                )
                raise LLMRouterError(f"TaskQueue task failed: {error}")
            text = _extract_generated_text_from_task_result(task.get("result"))
            if isinstance(text, str) and text.strip():
                return text
            raise LLMRouterError(
                "TaskQueue task completed without generated text"
            )

    return _P2PTaskQueueProvider()


def _builtin_provider_by_name(name: str, *, auto_install: bool = False) -> Optional[LLMProvider]:
    key = _canonicalize_provider(name)
    if not key:
        return None
    if key in {"mock", "dry_run", "dry-run"}:
        return _get_mock_provider()
    if key == "openrouter":
        return _get_openrouter_provider()
    if key == "openai":
        return _get_openai_provider()
    if key == "hf_inference_api":
        return _get_hf_inference_api_provider()
    if key == "p2p_task_queue":
        return _get_p2p_task_queue_provider()
    if key in _LLAMA_CPP_SERVER_PROVIDER_ALIASES:
        return _get_llama_cpp_provider(auto_install=auto_install)
    if key in _LLAMA_CPP_NATIVE_PROVIDER_ALIASES:
        return _get_llama_cpp_native_provider(auto_install=auto_install)
    if key in {"codex", "codex_cli"}:
        return _get_codex_cli_provider()
    if key in _GOOSE_CLI_PROVIDER_ALIASES:
        return _get_goose_cli_provider()
    if key in {"copilot_cli"}:
        return _get_copilot_cli_provider()
    if key in {"copilot_sdk"}:
        return _get_copilot_sdk_provider()
    if key in {"gemini_cli"}:
        return _get_gemini_cli_provider()
    if key in _GROK_CLI_PROVIDER_ALIASES:
        return _get_grok_cli_provider()
    if key == "grok":
        return _get_grok_cli_provider() or _get_xai_provider()
    if key in {"gemini_py"}:
        return _get_gemini_py_provider()
    if key in {"claude_code"}:
        return _get_claude_code_provider()
    if key in {"claude", "claude_py"}:
        return _get_claude_py_provider()
    if key in {"mistral_vibe", "mistral-vibe", "vibe"}:
        return _get_mistral_vibe_provider(auto_install=auto_install)
    if key in _XAI_API_PROVIDER_ALIASES:
        return _get_xai_provider()
    if key in {"meta_ai", "meta-ai", "meta_llama", "meta", "meta_spark", "spark"}:
        return _get_meta_ai_provider()
    if key in {"hf", "huggingface", "local_hf"}:
        return _get_local_hf_provider(deps=get_default_router_deps())
    return None


def _get_mock_provider() -> LLMProvider:
    """Return an ultra-lightweight deterministic provider.

    This is intended for unit tests and offline environments. It avoids spawning
    external CLIs/SDKs and avoids loading local HF models.
    """

    class _MockProvider:
        def generate(self, prompt: str, *, model_name: Optional[str] = None, **_: object) -> str:
            lowered = str(prompt or "").lower()

            import re

            def _looks_like_json_contract(text: str) -> bool:
                if not text:
                    return False
                if "return a json" in text or "json object" in text:
                    return True
                # Heuristics for our contracted logic converters.
                if "foloutput" in text or "fol_formula" in text or '"fol_formula"' in text:
                    return True
                if "<output_data_model>" in text and "[[schema]]" in text:
                    return True
                # The ContractedFOLConverter prompt itself (used in symai contract pipelines).
                if "convert natural language statements into formal first-order logic" in text:
                    return True
                return False

            def _extract_output_format(text: str) -> str:
                if not text:
                    return "symbolic"

                # Prefer explicit markers used by our prompts.
                # This avoids false positives when the prompt *mentions* all formats
                # (e.g. in a "Format requirements" section).
                explicit = re.search(r"requested\s+output\s+format\s*:\s*([a-z0-9_-]+)", text)
                if explicit:
                    token = explicit.group(1).strip().lower()
                    if token in {"prolog", "tptp", "symbolic", "json"}:
                        return token

                # Also support JSON-style markers.
                json_style = re.search(r'"output[_\s-]*format"\s*:\s*"([a-z0-9_-]+)"', text)
                if json_style:
                    token = json_style.group(1).strip().lower()
                    if token in {"prolog", "tptp", "symbolic", "json"}:
                        return token

                # Fallback heuristics (keep conservative).
                if "tptp" in text or "fof(" in text:
                    return "tptp"
                if "prolog" in text:
                    return "prolog"
                if "symbolic" in text:
                    return "symbolic"
                if "json" in text:
                    return "json"
                return "symbolic"

            def _looks_like_fol_conversion_prompt(text: str) -> bool:
                if not text:
                    return False
                # Common phrasing used by our logic primitives.
                if "convert" not in text:
                    return False
                if "first-order logic" not in text and "fol" not in text:
                    return False
                if "return only" in text and "formula" in text:
                    return True
                # Also treat the ContractedFOLConverter prompt as a conversion prompt.
                if "convert natural language statements" in text and "fol" in text:
                    return True
                return False

            def _mock_fol_formula(fmt: str, text: str) -> str:
                # Keep these short, deterministic, and syntactically valid.
                cats = "cats" in text and "animals" in text
                if fmt == "prolog":
                    # Use ASCII tokens to satisfy tests that check for prolog-like syntax.
                    return "forall(X, (cat(X) -> animal(X)))." if cats else "exists(X, statement(X))."
                if fmt == "tptp":
                    return "fof(ax1, axiom, ! [X] : ( cat(X) => animal(X) ) )." if cats else "fof(ax1, axiom, ? [X] : statement(X) )."
                # symbolic/default
                return "∀x (Cat(x) → Animal(x))" if cats else "∃x Statement(x)"

            def _mock_contract_json(text: str) -> str:
                import json

                fmt = _extract_output_format(text)
                formula = _mock_fol_formula(fmt, text)
                payload = {
                    "fol_formula": formula,
                    "confidence": 0.9,
                    "logical_components": {
                        "quantifiers": ["∀" if fmt == "symbolic" else ("forall" if fmt == "prolog" else "!")],
                        "predicates": ["Cat", "Animal"],
                        "entities": ["cat", "animal"],
                        "connectives": ["→" if fmt == "symbolic" else ("->" if fmt == "prolog" else "=>")],
                    },
                    "reasoning_steps": ["mock"],
                    "validation_results": {"valid": True, "backend": "mock"},
                    "warnings": [],
                    "metadata": {"backend": "mock", "model": model_name or "mock", "output_format": fmt},
                }
                return json.dumps(payload, ensure_ascii=False)

            # SyMAI contract/type-validation prompts expect JSON.
            if _looks_like_json_contract(lowered):
                return _mock_contract_json(lowered)

            # FOL conversion prompts should yield a formula (not an extraction list).
            if _looks_like_fol_conversion_prompt(lowered):
                fmt = _extract_output_format(lowered)
                return _mock_fol_formula(fmt, lowered)

            # Heuristic outputs for common logic-tool extraction prompts.
            # Keep these checks strict to avoid accidentally matching SyMAI schema prompts.
            if "extract" in lowered and "quantifier" in lowered:
                return "all, some"
            if "extract" in lowered and "predicate" in lowered:
                return "is, are, has"
            if "extract" in lowered and "entit" in lowered:
                return "cat, animal"
            if "extract" in lowered and ("connective" in lowered or "logical connective" in lowered):
                return "and, or, not"

            if "first-order logic" in lowered or "fol" in lowered:
                # Keep output short and stable, but honor requested output format.
                fmt = _extract_output_format(lowered)
                return _mock_fol_formula(fmt, lowered)

            # Generic non-empty fallback.
            return "OK"

    return _MockProvider()


@dataclass(frozen=True)
class _LLMProviderSpec:
    name: str
    aliases: Tuple[str, ...]
    description: str
    locality: str
    device: str
    authorization: str
    model_env: Tuple[str, ...] = ()
    default_model: Optional[str] = None
    chat: bool = False
    streaming: str = "unknown"
    tools: str = "unknown"


_BUILTIN_LLM_PROVIDER_SPECS: Tuple[_LLMProviderSpec, ...] = (
    _LLMProviderSpec(
        name="accelerate",
        aliases=("ipfs_accelerate_py",),
        description="Distributed ipfs_accelerate_py text generation provider.",
        locality="distributed",
        device="runtime-selected",
        authorization="unknown",
        model_env=(
            "ipfs_accelerate_py_LLM_MODEL",
            "IPFS_ACCELERATE_PY_LLM_MODEL",
            "IPFS_DATASETS_PY_LLM_MODEL",
        ),
    ),
    _LLMProviderSpec(
        name="mock",
        aliases=("dry-run", "dry_run"),
        description="Deterministic in-process provider for tests and offline use.",
        locality="local",
        device="cpu",
        authorization="none",
        default_model="mock",
        streaming="not-supported",
        tools="not-supported",
    ),
    _LLMProviderSpec(
        name="openrouter",
        aliases=(),
        description="OpenRouter OpenAI-compatible chat completions API.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "ipfs_accelerate_py_OPENROUTER_MODEL",
            "IPFS_ACCELERATE_PY_OPENROUTER_MODEL",
            "IPFS_DATASETS_PY_OPENROUTER_MODEL",
            "ipfs_accelerate_py_LLM_MODEL",
            "IPFS_ACCELERATE_PY_LLM_MODEL",
            "IPFS_DATASETS_PY_LLM_MODEL",
        ),
        default_model="openai/gpt-4o-mini",
        chat=True,
    ),
    _LLMProviderSpec(
        name="openai",
        aliases=("gpt-4", "gpt4"),
        description="OpenAI chat completions API.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "IPFS_ACCELERATE_PY_OPENAI_MODEL",
            "ipfs_accelerate_py_OPENAI_MODEL",
            "IPFS_DATASETS_PY_OPENAI_MODEL",
            "OPENAI_MODEL",
            "IPFS_ACCELERATE_PY_LLM_MODEL",
            "ipfs_accelerate_py_LLM_MODEL",
            "IPFS_DATASETS_PY_LLM_MODEL",
        ),
        default_model="gpt-4.1-mini",
        chat=True,
    ),
    _LLMProviderSpec(
        name="hf_inference_api",
        aliases=("hf_api", "hf_inference", "huggingface_inference"),
        description="Hugging Face hosted inference and chat providers.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "IPFS_ACCELERATE_PY_HF_INFERENCE_MODEL",
            "IPFS_DATASETS_PY_HF_INFERENCE_MODEL",
            "ipfs_accelerate_py_LLM_MODEL",
            "IPFS_ACCELERATE_PY_LLM_MODEL",
            "IPFS_DATASETS_PY_LLM_MODEL",
        ),
        default_model="gpt2",
        chat=True,
        tools="supported",
    ),
    _LLMProviderSpec(
        name="p2p_task_queue",
        aliases=("p2p", "p2p_task", "remote_queue", "task_queue"),
        description="Distributed task-queue text generation provider.",
        locality="distributed",
        device="worker-selected",
        authorization="unknown",
        model_env=(
            "ipfs_accelerate_py_LLM_MODEL",
            "IPFS_ACCELERATE_PY_LLM_MODEL",
            "IPFS_DATASETS_PY_LLM_MODEL",
        ),
        default_model="gpt2",
    ),
    _LLMProviderSpec(
        name="llama_cpp",
        aliases=tuple(sorted(_LLAMA_CPP_SERVER_PROVIDER_ALIASES - {"llama_cpp"})),
        description="Local llama.cpp OpenAI-compatible server.",
        locality="local",
        device="cpu,cuda,metal",
        authorization="optional",
        model_env=(
            "IPFS_ACCELERATE_LLAMA_CPP_MODEL",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_MODEL",
            "ipfs_accelerate_py_LLAMA_CPP_MODEL",
            "IPFS_ACCELERATE_LLAMA_CPP_MODEL_REF",
        ),
        default_model="Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4:NVFP4",
        chat=True,
    ),
    _LLMProviderSpec(
        name="llama_cpp_native",
        aliases=tuple(sorted(_LLAMA_CPP_NATIVE_PROVIDER_ALIASES - {"llama_cpp_native"})),
        description="Local in-process llama-cpp-python provider.",
        locality="local",
        device="cpu,cuda,metal",
        authorization="none",
        model_env=(
            "IPFS_ACCELERATE_LLAMA_CPP_NATIVE_MODEL_PATH",
            "IPFS_ACCELERATE_LLAMA_CPP_NATIVE_MODEL_REF",
            "IPFS_ACCELERATE_LLAMA_CPP_MODEL_REF",
        ),
        default_model="Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4:NVFP4",
        chat=True,
    ),
    _LLMProviderSpec(
        name="codex_cli",
        aliases=("codex", "codex-cli"),
        description="OpenAI Codex CLI text generation provider.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "ipfs_accelerate_py_CODEX_CLI_MODEL",
            "IPFS_ACCELERATE_PY_CODEX_CLI_MODEL",
            "IPFS_DATASETS_PY_CODEX_CLI_MODEL",
            "ipfs_accelerate_py_CODEX_MODEL",
            "IPFS_ACCELERATE_PY_CODEX_MODEL",
            "IPFS_DATASETS_PY_CODEX_MODEL",
        ),
        default_model="chatgpt-5.6-terra",
        tools="supported",
    ),
    _LLMProviderSpec(
        name="goose_cli",
        aliases=("goose", "goose-cli", "block_goose", "block-goose", "aaif_goose"),
        description=(
            "Block/AAIF Goose CLI provider. Ordinary generation is chat-only; "
            "default model backend is Meta Muse Spark via OpenAI-compatible env."
        ),
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "ipfs_accelerate_py_GOOSE_CLI_MODEL",
            "IPFS_ACCELERATE_PY_GOOSE_CLI_MODEL",
            "GOOSE_MODEL",
            "ipfs_accelerate_py_META_AI_MODEL",
            "ipfs_accelerate_py_LLM_MODEL",
        ),
        default_model=META_MODEL_API_DEFAULT_MODEL,
        tools="supported",
    ),
    _LLMProviderSpec(
        name="copilot_cli",
        aliases=("copilot",),
        description="GitHub Copilot CLI text generation provider.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "ipfs_accelerate_py_COPILOT_CLI_MODEL",
            "IPFS_ACCELERATE_PY_COPILOT_CLI_MODEL",
            "IPFS_DATASETS_PY_COPILOT_CLI_MODEL",
        ),
        streaming="supported",
        tools="supported",
    ),
    _LLMProviderSpec(
        name="copilot_sdk",
        aliases=(),
        description="GitHub Copilot Python SDK provider.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "ipfs_accelerate_py_COPILOT_SDK_MODEL",
            "IPFS_ACCELERATE_PY_COPILOT_SDK_MODEL",
            "IPFS_DATASETS_PY_COPILOT_SDK_MODEL",
        ),
        streaming="supported",
        tools="supported",
    ),
    _LLMProviderSpec(
        name="gemini_cli",
        aliases=("gemini",),
        description="Google Gemini CLI text generation provider.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        tools="supported",
    ),
    _LLMProviderSpec(
        name="gemini_py",
        aliases=(),
        description="Python wrapper around the Google Gemini CLI.",
        locality="remote",
        device="provider-managed",
        authorization="required",
    ),
    _LLMProviderSpec(
        name="claude_code",
        aliases=(),
        description="Anthropic Claude Code CLI provider.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        tools="supported",
    ),
    _LLMProviderSpec(
        name="claude_py",
        aliases=("claude",),
        description="Python wrapper around the Anthropic Claude CLI.",
        locality="remote",
        device="provider-managed",
        authorization="required",
    ),
    _LLMProviderSpec(
        name="mistral_vibe",
        aliases=("mistral-vibe", "vibe"),
        description="Mistral Vibe CLI provider.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "IPFS_ACCELERATE_MISTRAL_VIBE_MODEL",
            "IPFS_ACCELERATE_PY_MISTRAL_VIBE_MODEL",
            "ipfs_accelerate_py_MISTRAL_VIBE_MODEL",
            "IPFS_DATASETS_PY_MISTRAL_VIBE_MODEL",
        ),
        tools="supported",
    ),
    _LLMProviderSpec(
        name="grok_cli",
        aliases=tuple(sorted(_GROK_CLI_PROVIDER_ALIASES - {"grok_cli"})),
        description="Official xAI Grok CLI provider.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "ipfs_accelerate_py_GROK_CLI_MODEL",
            "IPFS_ACCELERATE_PY_GROK_CLI_MODEL",
            "IPFS_DATASETS_PY_GROK_CLI_MODEL",
            "GROK_CLI_MODEL",
        ),
        tools="supported",
    ),
    _LLMProviderSpec(
        name="xai",
        aliases=tuple(sorted(_XAI_API_PROVIDER_ALIASES - {"xai"})),
        description="xAI Grok OpenAI-compatible chat API.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "ipfs_accelerate_py_XAI_MODEL",
            "IPFS_ACCELERATE_PY_XAI_MODEL",
            "IPFS_DATASETS_PY_XAI_MODEL",
            "ipfs_accelerate_py_LLM_MODEL",
        ),
        default_model="grok-3",
        chat=True,
    ),
    _LLMProviderSpec(
        name="meta_ai",
        aliases=("meta", "meta-ai", "meta_llama", "meta_spark", "spark"),
        description="Meta Model API / Muse Spark OpenAI-compatible chat API.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "ipfs_accelerate_py_META_AI_MODEL",
            "IPFS_ACCELERATE_PY_META_AI_MODEL",
            "IPFS_DATASETS_PY_META_AI_MODEL",
            "ipfs_accelerate_py_LLM_MODEL",
        ),
        default_model=META_MODEL_API_DEFAULT_MODEL,
        chat=True,
    ),
    _LLMProviderSpec(
        name="local_hf",
        aliases=("hf", "huggingface"),
        description="Local Hugging Face transformers text-generation pipeline.",
        locality="local",
        device="cpu,cuda,mps",
        authorization="none",
        model_env=(
            "ipfs_accelerate_py_LLM_MODEL",
            "IPFS_ACCELERATE_PY_LLM_MODEL",
            "IPFS_DATASETS_PY_LLM_MODEL",
        ),
        default_model="gpt2",
    ),
)
_BUILTIN_LLM_PROVIDER_SPEC_BY_NAME = {
    spec.name: spec for spec in _BUILTIN_LLM_PROVIDER_SPECS
}


def _llm_capability(
    *,
    chat: bool = False,
    streaming: bool = False,
    tools: bool = False,
    max_context_tokens: Optional[int] = None,
) -> CapabilityDescriptor:
    operations = [Operation.TEXT_GENERATE, Operation.BATCH]
    if chat:
        operations.append(Operation.TEXT_CHAT)
    if streaming:
        operations.append(Operation.STREAM)
    if tools:
        operations.append(Operation.TOOL_CALL)
    return CapabilityDescriptor(
        operations=tuple(operations),
        input_modalities=(Modality.TEXT,),
        output_modalities=(Modality.TEXT,),
        max_context_tokens=max_context_tokens,
    )


def _catalog_llm_model_name(value: object) -> str:
    """Normalize invocation identifiers into the shared catalog name grammar."""

    normalized = str(value or "").strip().casefold()
    normalized = re.sub(r"[^a-z0-9._/-]+", "-", normalized)
    normalized = re.sub(r"/{2,}", "/", normalized)
    normalized = re.sub(r"\.{2,}", ".", normalized)
    normalized = normalized.strip("._/-")
    if not normalized:
        normalized = "default"
    return normalized[:128].rstrip("._/-") or "default"


def _llm_model_facts(model_name: str) -> Tuple[Optional[int], Optional[str]]:
    """Return conservative facts for stable built-in model hints."""

    normalized = str(model_name or "").strip().casefold()
    if normalized == "openai/gpt-4o-mini":
        return 128_000, "transformer"
    if normalized == "gpt-4.1-mini":
        return 1_047_576, "transformer"
    if normalized == "gpt2":
        return 1_024, "transformer"
    if normalized == "grok-3":
        return 131_072, "transformer"
    if normalized == "meta-llama/llama-3.3-70b-instruct":
        return 131_072, "transformer"
    if "llama" in normalized or "leanstral" in normalized:
        return None, "transformer"
    return None, None


def _effective_llm_spec_model(spec: _LLMProviderSpec) -> Optional[str]:
    model_name = _coalesce_env(*spec.model_env) or spec.default_model
    if spec.name == "meta_ai":
        return normalize_meta_model_name(model_name)
    return model_name


def _llm_env_has_value(*names: str) -> bool:
    return bool(_coalesce_env(*names))


def _llm_provider_authorized(name: str) -> Optional[bool]:
    if name == "openrouter":
        return _llm_env_has_value(
            "ipfs_accelerate_py_OPENROUTER_API_KEY",
            "IPFS_ACCELERATE_PY_OPENROUTER_API_KEY",
            "IPFS_DATASETS_PY_OPENROUTER_API_KEY",
            "OPENROUTER_API_KEY",
        )
    if name == "openai":
        return _llm_env_has_value(
            "OPENAI_API_KEY",
            "OPENAI_KEY",
            "OPENAI_TOKEN",
            "IPFS_ACCELERATE_PY_OPENAI_API_KEY",
            "ipfs_accelerate_py_OPENAI_API_KEY",
        )
    if name == "hf_inference_api":
        # Do not call _resolve_hf_api_token: it imports huggingface_hub and
        # reads its credential store. With no environment token, auth is
        # unknown rather than false.
        if _llm_env_has_value(
            "IPFS_ACCELERATE_PY_HF_API_TOKEN",
            "ipfs_accelerate_py_HF_API_TOKEN",
            "IPFS_DATASETS_PY_HF_API_TOKEN",
            "HUGGINGFACEHUB_API_TOKEN",
            "HUGGINGFACE_API_TOKEN",
            "HF_TOKEN",
        ):
            return True
        return None
    if name == "xai":
        return _llm_env_has_value(
            "XAI_API_KEY",
            "ipfs_accelerate_py_XAI_API_KEY",
            "IPFS_ACCELERATE_PY_XAI_API_KEY",
            "IPFS_DATASETS_PY_XAI_API_KEY",
        )
    if name == "meta_ai":
        if _llm_env_has_value(
            "MODEL_API_KEY",
            "META_AI_API_KEY",
            "ipfs_accelerate_py_META_AI_API_KEY",
        ):
            return True
        # The runtime can also use the encrypted credentials manager.
        # Discovery must not inspect it, so absence from the environment is
        # unknown rather than proof that authorization is unavailable.
        return None
    if name == "mistral_vibe" and _llm_env_has_value(
        "MISTRAL_API_KEY",
        "IPFS_ACCELERATE_MISTRAL_API_KEY",
        "ipfs_accelerate_py_MISTRAL_API_KEY",
    ):
        return True
    if name == "grok_cli" and _llm_env_has_value("XAI_API_KEY"):
        return True
    if name in {
        "codex_cli",
        "copilot_cli",
        "copilot_sdk",
        "goose_cli",
        "gemini_cli",
        "gemini_py",
        "claude_code",
        "claude_py",
        "mistral_vibe",
        "grok_cli",
    }:
        # These integrations can use login/key stores. Discovery deliberately
        # does not inspect those stores.
        return None
    if name in {"mock", "local_hf", "llama_cpp_native"}:
        return True
    return None


def _builtin_llm_provider_state(
    spec: _LLMProviderSpec,
) -> Tuple[LifecycleState, OperationalState]:
    if spec.name == "mock":
        return (
            LifecycleState.READY,
            OperationalState(
                known=True,
                configured=True,
                authorized=True,
                reachable=True,
                healthy=True,
                routable=True,
            ),
        )
    if spec.name == "accelerate":
        raw = _coalesce_env(
            "ipfs_accelerate_py_ENABLE_IPFS_ACCELERATE",
            "IPFS_ACCELERATE_PY_ENABLE_IPFS_ACCELERATE",
            "IPFS_DATASETS_PY_ENABLE_IPFS_ACCELERATE",
        )
        configured = (
            None if not raw else str(raw).strip().lower() in {"1", "true", "yes", "on"}
        )
        return (
            LifecycleState.CONFIGURED if configured else LifecycleState.DECLARED,
            OperationalState(
                known=True,
                configured=configured,
                authorized=None,
                reachable=None,
                healthy=None,
                routable=None,
            ),
        )
    authorized = _llm_provider_authorized(spec.name)
    if spec.name in {"openrouter", "openai", "hf_inference_api", "xai", "meta_ai"}:
        configured = authorized
        return (
            LifecycleState.CONFIGURED if configured is True else LifecycleState.DECLARED,
            OperationalState(
                known=True,
                configured=configured,
                authorized=authorized,
                reachable=None,
                healthy=None,
                routable=None,
            ),
        )
    return (
        LifecycleState.DECLARED,
        OperationalState(
            known=True,
            configured=None,
            authorized=authorized,
            reachable=None,
            healthy=None,
            routable=None,
        ),
    )


def _builtin_llm_provider_descriptor(
    spec: _LLMProviderSpec,
) -> ProviderDescriptor:
    model_name = _effective_llm_spec_model(spec)
    context_tokens, _ = _llm_model_facts(model_name or "")
    lifecycle, state = _builtin_llm_provider_state(spec)
    return ProviderDescriptor(
        name=spec.name,
        aliases=spec.aliases,
        description=spec.description,
        capabilities=(
            _llm_capability(
                chat=spec.chat,
                streaming=spec.streaming == "supported",
                tools=spec.tools == "supported",
                max_context_tokens=context_tokens,
            ),
        ),
        lifecycle=lifecycle,
        state=state,
        provenance=(Provenance(source="llm_router.static"),),
        labels={
            "access_requirement": spec.authorization,
            "batching": "supported",
            "device": spec.device,
            "locality": spec.locality,
            "model_hint": model_name or "provider-default",
            "streaming": spec.streaming,
            "tools": spec.tools,
        },
    )


def _llm_provider_descriptors_by_name() -> Dict[str, ProviderDescriptor]:
    descriptors = {
        spec.name: _builtin_llm_provider_descriptor(spec)
        for spec in _BUILTIN_LLM_PROVIDER_SPECS
    }
    with _PROVIDER_REGISTRY_LOCK:
        registered = tuple(_PROVIDER_REGISTRY.values())
    for info in registered:
        # Dynamic registration has the same precedence as invocation.
        descriptors[info.name] = (
            info.descriptor
            or _registered_llm_provider_descriptor(info.name, None)
        )
    return descriptors


def list_providers() -> List[ProviderDescriptor]:
    """List LLM providers without resolving or constructing any provider."""

    return [
        descriptor
        for _, descriptor in sorted(_llm_provider_descriptors_by_name().items())
    ]


def _canonical_llm_catalog_provider_name(name: str) -> str:
    requested = str(name or "").strip().lower()
    if not requested:
        raise ValueError("LLM provider name must be non-empty")
    descriptors = _llm_provider_descriptors_by_name()
    if requested in descriptors:
        return requested
    # The historical "grok" selector prefers the CLI when installed and then
    # the xAI API. Checking PATH does not execute or install the CLI.
    if requested == "grok":
        return (
            "grok_cli"
            if _cli_available(_grok_cli_command())
            else "xai"
        )
    canonical = _PROVIDER_ALIASES.get(requested)
    if canonical in descriptors:
        return canonical
    matches = sorted(
        descriptor.name
        for descriptor in descriptors.values()
        if requested in descriptor.aliases
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous LLM provider alias {name!r}: {', '.join(matches)}"
        )
    raise ValueError(f"Unknown LLM provider: {name}")


def get_provider_descriptor(name: str) -> ProviderDescriptor:
    """Return the descriptor for a provider canonical name or alias."""

    canonical = _canonical_llm_catalog_provider_name(name)
    return _llm_provider_descriptors_by_name()[canonical]


def _llm_model_descriptor(
    provider: ProviderDescriptor,
    model_name: str,
) -> ModelDescriptor:
    context_tokens, architecture = _llm_model_facts(model_name)
    provider_labels = dict(provider.labels)
    with _PROVIDER_REGISTRY_LOCK:
        dynamically_registered = provider.name in _PROVIDER_REGISTRY
    if dynamically_registered:
        # Provider-authored capability sets may describe distinct chat, tool,
        # or streaming surfaces. Preserve them instead of flattening them into
        # the single built-in LLM capability shape.
        capabilities = provider.capabilities or (_llm_capability(),)
    else:
        capability = (
            provider.capabilities[0]
            if provider.capabilities
            else _llm_capability()
        )
        capabilities = (
            _llm_capability(
                chat=Operation.TEXT_CHAT in capability.operations,
                streaming=Operation.STREAM in capability.operations,
                tools=Operation.TOOL_CALL in capability.operations,
                max_context_tokens=context_tokens,
            ),
        )
    return ModelDescriptor(
        provider_id=provider.provider_id,
        name=_catalog_llm_model_name(model_name),
        architecture=architecture,
        capabilities=capabilities,
        lifecycle=provider.lifecycle,
        state=provider.state,
        provenance=(Provenance(source="llm_router.static"),),
        labels={
            "access_requirement": provider_labels.get(
                "access_requirement", "unknown"
            ),
            "batching": provider_labels.get("batching", "supported"),
            "device": provider_labels.get("device", "unknown"),
            "invocation_model": model_name,
            "locality": provider_labels.get("locality", "unknown"),
            "streaming": provider_labels.get("streaming", "unknown"),
            "tools": provider_labels.get("tools", "unknown"),
        },
    )


def _llm_models_for_provider(provider_name: str) -> Tuple[ModelDescriptor, ...]:
    descriptors = _llm_provider_descriptors_by_name()
    provider = descriptors[provider_name]
    with _PROVIDER_REGISTRY_LOCK:
        registered = _PROVIDER_REGISTRY.get(provider_name)
    if registered is not None:
        return registered.models
    spec = _BUILTIN_LLM_PROVIDER_SPEC_BY_NAME[provider_name]
    model_name = _effective_llm_spec_model(spec)
    if not model_name:
        return ()
    return (_llm_model_descriptor(provider, model_name),)


def list_models(provider: Optional[str] = None) -> List[ModelDescriptor]:
    """List statically known and dynamically registered LLM model hints."""

    if provider is not None:
        provider_names = (_canonical_llm_catalog_provider_name(provider),)
    else:
        provider_names = tuple(sorted(_llm_provider_descriptors_by_name()))
    models = [
        model
        for provider_name in provider_names
        for model in _llm_models_for_provider(provider_name)
    ]
    return sorted(
        models,
        key=lambda model: (model.provider_id, model.name, model.model_id or ""),
    )


def _select_llm_discovery_provider(
    provider: Optional[str],
    *,
    deps: Optional[RouterDeps],
) -> str:
    if provider:
        return _canonical_llm_catalog_provider_name(provider)

    forced = _coalesce_env(
        "ipfs_accelerate_py_LLM_PROVIDER",
        "IPFS_ACCELERATE_PY_LLM_PROVIDER",
        "IPFS_DATASETS_PY_LLM_PROVIDER",
    )
    if forced:
        return _canonical_llm_catalog_provider_name(forced)

    accelerate_enabled = _coalesce_env(
        "ipfs_accelerate_py_ENABLE_IPFS_ACCELERATE",
        "IPFS_ACCELERATE_PY_ENABLE_IPFS_ACCELERATE",
        "IPFS_DATASETS_PY_ENABLE_IPFS_ACCELERATE",
    )
    if accelerate_enabled and str(accelerate_enabled).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return "accelerate"

    for name in ("openrouter", "openai", "hf_inference_api", "xai", "meta_ai"):
        if _llm_provider_authorized(name):
            return name

    # Injected managers are already-live caller state and can be observed
    # without initializing a manager.
    resolved_deps = deps or get_default_router_deps()
    managers = getattr(resolved_deps, "accelerate_managers", {})
    if isinstance(managers, Mapping) and any(value is not None for value in managers.values()):
        return "accelerate"

    try:
        transformers_available = importlib.util.find_spec("transformers") is not None
    except Exception:
        transformers_available = False
    if transformers_available:
        return "local_hf"
    raise RuntimeError(
        "No LLM provider is statically resolvable for the requested constraints"
    )


def resolve_model(
    model_name: Optional[str] = None,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    device: Optional[str] = None,
    deps: Optional[RouterDeps] = None,
    **constraints: object,
) -> ModelDescriptor:
    """Resolve an LLM model using side-effect-free router selection metadata.

    Explicit model overrides remain open-ended just like ``generate_text``:
    an unlisted identifier is returned with unknown model-specific facts and
    is not rejected merely because it is absent from static hints.
    """

    if model is not None:
        if model_name is not None and str(model_name) != str(model):
            raise ValueError("model and model_name specify different values")
        model_name = str(model)
    operation = constraints.pop("operation", Operation.TEXT_GENERATE)
    if constraints:
        unknown = ", ".join(sorted(str(key) for key in constraints))
        raise TypeError(f"Unknown LLM resolution constraints: {unknown}")
    operation_value = (
        operation.value if isinstance(operation, Operation) else str(operation)
    )
    supported_operations = {
        Operation.TEXT_GENERATE.value,
        Operation.TEXT_CHAT.value,
        Operation.BATCH.value,
        Operation.STREAM.value,
        Operation.TOOL_CALL.value,
    }
    if operation_value not in supported_operations:
        raise ValueError(f"LLM router does not support operation {operation_value!r}")

    provider_name = _select_llm_discovery_provider(provider, deps=deps)
    provider_descriptor = get_provider_descriptor(provider_name)
    capability_operations = {
        item.value
        for capability in provider_descriptor.capabilities
        for item in capability.operations
    }
    if operation_value not in capability_operations:
        raise ValueError(
            f"LLM provider {provider_name!r} does not declare operation "
            f"{operation_value!r}"
        )
    if device:
        known_device = dict(provider_descriptor.labels).get("device", "unknown")
        requested_device = str(device).strip().casefold()
        known_devices = {
            item.strip().casefold() for item in known_device.split(",") if item.strip()
        }
        if (
            known_device not in {"unknown", "runtime-selected", "worker-selected", "provider-managed"}
            and requested_device not in known_devices
        ):
            raise ValueError(
                f"LLM provider {provider_name!r} does not declare device "
                f"{requested_device!r}"
            )

    known_models = _llm_models_for_provider(provider_name)
    requested_model = str(model_name or "").strip()
    if not requested_model:
        if not known_models:
            raise ValueError(
                f"LLM provider {provider_name!r} has no known default model; "
                "specify model_name explicitly"
            )
        return known_models[0]

    requested_key = requested_model.casefold()
    for descriptor in known_models:
        labels = dict(descriptor.labels)
        invocation_name = labels.get(
            "invocation_model",
            labels.get("router_model_name", descriptor.name),
        )
        if requested_key in {
            descriptor.name.casefold(),
            str(invocation_name).casefold(),
            *(alias.casefold() for alias in descriptor.aliases),
        }:
            return descriptor
    return _llm_model_descriptor(provider_descriptor, requested_model)


def get_catalog_snapshot() -> CatalogSnapshot:
    """Project LLM router discovery into a deterministic catalog snapshot."""

    providers = tuple(list_providers())
    models = tuple(list_models())
    provider_by_id = {provider.provider_id: provider for provider in providers}
    bindings = tuple(
        RouterBinding(
            router="llm_router",
            provider_id=model.provider_id,
            model_id=model.model_id,
            operations=model.capabilities[0].operations,
            priority=index,
            state=provider_by_id[model.provider_id].state,
            provenance=(Provenance(source="llm_router.static"),),
            labels={
                "invocation_model": dict(model.labels).get(
                    "invocation_model",
                    dict(model.labels).get("router_model_name", model.name),
                )
            },
        )
        for index, model in enumerate(models)
    )
    return CatalogSnapshot(
        providers=providers,
        models=models,
        bindings=bindings,
    )


def catalog_snapshot() -> CatalogSnapshot:
    """Compatibility alias for catalog source adapters."""

    return get_catalog_snapshot()


def _resolve_provider_uncached(preferred: Optional[str], *, deps: RouterDeps) -> LLMProvider:
    preferred_value = (preferred or "").strip()
    if preferred_value:
        name = _canonicalize_provider(preferred_value)
        if name in {"ipfs_accelerate_py", "accelerate"}:
            accelerate_provider = _get_accelerate_provider(deps)
            if accelerate_provider is None:
                raise LLMRouterError(
                    "Accelerate provider not available. Set ipfs_accelerate_py_ENABLE_IPFS_ACCELERATE=1 and ensure ipfs_accelerate_py is installed/configured."
                )
            return accelerate_provider

        info = _PROVIDER_REGISTRY.get(name)
        if info is not None:
            return info.factory()

        if name == "copilot_sdk":
            builtin = _get_copilot_sdk_provider()
            if builtin is None:
                raise LLMRouterError("Copilot Python SDK not installed (optional dependency).")
            return builtin

        if name in {"mistral_vibe", "mistral-vibe", "vibe"}:
            builtin = _get_mistral_vibe_provider(auto_install=True)
        elif name in _LLAMA_CPP_SERVER_PROVIDER_ALIASES:
            builtin = _get_llama_cpp_provider(auto_install=True)
        elif name in _LLAMA_CPP_NATIVE_PROVIDER_ALIASES:
            builtin = _get_llama_cpp_native_provider(auto_install=True)
            if builtin is None:
                raise LLMRouterError("llama-cpp-python not installed for llama_cpp_native provider.")
        else:
            builtin = _builtin_provider_by_name(name)
        if builtin is not None:
            return builtin
        raise ValueError(f"Unknown LLM provider: {preferred_value}")

    forced = _coalesce_env(
        "ipfs_accelerate_py_LLM_PROVIDER",
        "IPFS_ACCELERATE_PY_LLM_PROVIDER",
        "IPFS_DATASETS_PY_LLM_PROVIDER",
    )
    if forced:
        forced_name = _canonicalize_provider(forced)
        if forced_name in {"ipfs_accelerate_py", "accelerate"}:
            accelerate_provider = _get_accelerate_provider(deps)
            if accelerate_provider is None:
                raise LLMRouterError(
                    "Accelerate provider not available. Set ipfs_accelerate_py_ENABLE_IPFS_ACCELERATE=1 and ensure ipfs_accelerate_py is installed/configured."
                )
            return accelerate_provider

        info = _PROVIDER_REGISTRY.get(forced_name)
        if info is not None:
            return info.factory()

        if forced_name == "copilot_sdk":
            builtin = _get_copilot_sdk_provider()
            if builtin is None:
                raise LLMRouterError("Copilot Python SDK not installed (optional dependency).")
            return builtin

        if forced_name in {"mistral_vibe", "mistral-vibe", "vibe"}:
            builtin = _get_mistral_vibe_provider(auto_install=True)
        elif forced_name in _LLAMA_CPP_SERVER_PROVIDER_ALIASES:
            builtin = _get_llama_cpp_provider(auto_install=True)
        elif forced_name in _LLAMA_CPP_NATIVE_PROVIDER_ALIASES:
            builtin = _get_llama_cpp_native_provider(auto_install=True)
            if builtin is None:
                raise LLMRouterError("llama-cpp-python not installed for llama_cpp_native provider.")
        else:
            builtin = _builtin_provider_by_name(forced_name)
        if builtin is not None:
            return builtin
        raise ValueError(f"Unknown LLM provider: {forced}")
    accelerate_provider = _get_accelerate_provider(deps)
    if accelerate_provider is not None:
        return accelerate_provider

    # Try common optional CLI/API providers if available. Grok is only
    # auto-discovered when an API key, external auth provider, or OAuth token
    # is present; explicit ``provider="grok_cli"`` still returns an actionable
    # authentication error when the binary is installed but logged out.
    optional_provider_names = [
        "openrouter",
        "openai",
        "hf_inference_api",
        "xai",
        "meta_ai",
        "codex_cli",
        "copilot_cli",
        "goose_cli",
        "gemini_cli",
        "claude_code",
        "mistral_vibe",
        "claude_py",
        "gemini_py",
        "copilot_sdk",
    ]
    if _grok_cli_auth_available():
        optional_provider_names.insert(1, "grok_cli")
    for name in optional_provider_names:
        candidate = _builtin_provider_by_name(name)
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
      (e.g., AccelerateManager) stored on that object.
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


def _effective_llm_provider_name(explicit_provider: Optional[str]) -> str:
    """Return the canonical provider name used for diagnostics and traces."""

    key = (
        explicit_provider
        or os.getenv("ipfs_accelerate_py_LLM_PROVIDER")
        or os.getenv("IPFS_ACCELERATE_PY_LLM_PROVIDER")
        or os.getenv("IPFS_DATASETS_PY_LLM_PROVIDER")
        or ""
    ).strip().lower()
    if key == "grok":
        return "grok_cli" if _cli_available(_grok_cli_command()) else "xai"
    if key in _GROK_CLI_PROVIDER_ALIASES:
        return "grok_cli"
    if key in _XAI_API_PROVIDER_ALIASES:
        return "xai"
    aliases = {
        "codex": "codex_cli",
        "copilot": "copilot_cli",
        "gemini": "gemini_cli",
        "mistral-vibe": "mistral_vibe",
        "vibe": "mistral_vibe",
        "meta-ai": "meta_ai",
        "meta": "meta_ai",
        "spark": "meta_ai",
    }
    return _canonicalize_provider(aliases.get(key, key))


def _set_last_generation_trace(
    *,
    provider_name: str,
    model_name: Optional[str],
    route_trace: Optional[dict[str, object]] = None,
) -> None:
    payload = {
        "effective_provider_name": str(provider_name or "").strip(),
        "effective_model_name": str(model_name or "").strip(),
    }
    if route_trace:
        for key in _PINNED_SYMAI_TRACE_KEYS:
            value = route_trace.get(key)
            if isinstance(value, str) and value.strip():
                payload[key] = value.strip()
    _LAST_GENERATION_TRACE.payload = payload


def _clear_last_generation_trace() -> None:
    _LAST_GENERATION_TRACE.payload = {
        "effective_provider_name": "",
        "effective_model_name": "",
    }


def get_last_generation_trace() -> dict[str, str]:
    """Return the effective provider and model used by the latest call."""

    payload = getattr(_LAST_GENERATION_TRACE, "payload", None)
    return dict(payload) if isinstance(payload, dict) else {}


def _iter_unpinned_optional_providers() -> list[tuple[str, LLMProvider]]:
    providers: list[tuple[str, LLMProvider]] = []
    names = list(_UNPINNED_OPTIONAL_PROVIDER_ORDER)
    if _grok_cli_auth_available():
        names.insert(1, "grok_cli")
    for name in names:
        candidate = _builtin_provider_by_name(name)
        if candidate is not None:
            providers.append((name, candidate))
    return providers


def _generate_with_provider_fallbacks(
    provider_name: Optional[str],
    backend: LLMProvider,
    prompt: str,
    *,
    model_name: Optional[str],
    kwargs: dict[str, object],
) -> str:
    effective_provider_name = _canonicalize_provider(provider_name)
    disable_model_retry = bool(kwargs.pop("disable_model_retry", False))
    try:
        return backend.generate(prompt, model_name=model_name, **kwargs)
    except Exception as initial_error:
        if model_name is not None and not disable_model_retry:
            try:
                return backend.generate(prompt, model_name=None, **kwargs)
            except Exception:
                pass
        if (
            _is_hf_inference_provider_name(effective_provider_name)
            and _is_hf_model_compatibility_error(initial_error)
        ):
            attempted = {
                str(model_name or "").strip(),
                _coalesce_env(
                    "IPFS_ACCELERATE_PY_HF_INFERENCE_MODEL",
                    "IPFS_DATASETS_PY_HF_INFERENCE_MODEL",
                ),
            }
            for fallback_model in _hf_llm_fallback_models(kwargs=dict(kwargs)):
                if not fallback_model or fallback_model in attempted:
                    continue
                attempted.add(fallback_model)
                try:
                    return backend.generate(
                        prompt,
                        model_name=fallback_model,
                        **kwargs,
                    )
                except Exception:
                    continue
        raise initial_error


def generate_text(
    prompt: str,
    *,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    provider_instance: Optional[LLMProvider] = None,
    deps: Optional[RouterDeps] = None,
    allow_local_fallback: bool = True,
    **kwargs: object,
) -> str:
    """Generate text from an LLM."""

    resolved_deps = deps or get_default_router_deps()
    effective_provider_name = _effective_llm_provider_name(provider)
    _clear_last_generation_trace()
    # The pinned SyMAI engine owns its cache together with the four inner-route
    # receipt fields. A text-only router cache cannot reproduce those fields.
    response_cache_ok = (
        _response_cache_enabled()
        and kwargs.get(_SYMAI_ROUTE_BINDING_KWARG) is None
    )
    if response_cache_ok:
        try:
            cache_key = _response_cache_key(provider=provider, model_name=model_name, prompt=prompt, kwargs=dict(kwargs))
            getter = getattr(resolved_deps, "get_cached_or_remote", None)
            cached = getter(cache_key) if callable(getter) else resolved_deps.get_cached(cache_key)
            if isinstance(cached, str):
                _set_last_generation_trace(
                    provider_name=effective_provider_name,
                    model_name=model_name,
                )
                return cached
        except Exception:
            pass

    backend = provider_instance or get_llm_provider(provider, deps=resolved_deps)

    def _cache_result(value: str, *, used_model_name: Optional[str]) -> None:
        if not response_cache_ok:
            return
        try:
            cache_key = _response_cache_key(
                provider=provider,
                model_name=used_model_name,
                prompt=prompt,
                kwargs=dict(kwargs),
            )
            setter = getattr(resolved_deps, "set_cached_and_remote", None)
            if callable(setter):
                setter(cache_key, str(value))
            else:
                resolved_deps.set_cached(cache_key, str(value))
        except Exception:
            pass

    try:
        result = _generate_with_provider_fallbacks(
            effective_provider_name,
            backend,
            prompt,
            model_name=model_name,
            kwargs=dict(kwargs),
        )
        route_trace: dict[str, object] = {}
        route_trace_getter = getattr(
            backend, "get_last_generation_trace", None
        )
        if callable(route_trace_getter):
            try:
                candidate_trace = route_trace_getter()
            except Exception:
                candidate_trace = {}
            if isinstance(candidate_trace, dict):
                route_trace = candidate_trace
        _set_last_generation_trace(
            provider_name=effective_provider_name,
            model_name=model_name,
            route_trace=route_trace,
        )
        _cache_result(str(result), used_model_name=model_name)
        return result
    except Exception:
        pinned_provider = _canonicalize_provider(provider)
        pinned_optional = bool(
            provider is not None
            and pinned_provider in _UNPINNED_OPTIONAL_PROVIDER_ORDER
        )
        if provider is None or pinned_optional:
            for fallback_name, fallback_provider in _iter_unpinned_optional_providers():
                if fallback_provider is backend:
                    continue
                try:
                    result = _generate_with_provider_fallbacks(
                        fallback_name,
                        fallback_provider,
                        prompt,
                        model_name=model_name,
                        kwargs=dict(kwargs),
                    )
                    _set_last_generation_trace(
                        provider_name=fallback_name,
                        model_name=model_name,
                    )
                    _cache_result(str(result), used_model_name=model_name)
                    return result
                except Exception:
                    pass

        if pinned_optional:
            try:
                accelerate_provider = _get_accelerate_provider(resolved_deps)
                if accelerate_provider is not None and accelerate_provider is not backend:
                    result = _generate_with_provider_fallbacks(
                        "ipfs_accelerate_py",
                        accelerate_provider,
                        prompt,
                        model_name=model_name,
                        kwargs=dict(kwargs),
                    )
                    _set_last_generation_trace(
                        provider_name="ipfs_accelerate_py",
                        model_name=model_name,
                    )
                    _cache_result(str(result), used_model_name=model_name)
                    return result
            except Exception:
                pass

        if allow_local_fallback and (provider is None or pinned_optional):
            local_hf = _get_local_hf_provider(deps=resolved_deps)
            if local_hf is not None and backend is not local_hf:
                try:
                    result = local_hf.generate(prompt, model_name=model_name, **kwargs)
                    _set_last_generation_trace(
                        provider_name="local_hf",
                        model_name=model_name,
                    )
                    _cache_result(str(result), used_model_name=model_name)
                    return result
                except Exception:
                    if model_name is not None:
                        result = local_hf.generate(prompt, model_name=None, **kwargs)
                        _set_last_generation_trace(
                            provider_name="local_hf",
                            model_name=None,
                        )
                        _cache_result(str(result), used_model_name=None)
                        return result
        raise


def _batch_worker_count(
    *,
    size: int,
    max_workers: Optional[int],
    provider: Optional[str],
    default_cap: int = 4,
) -> int:
    if size <= 1:
        return 1
    if max_workers is not None:
        try:
            return max(1, min(int(max_workers), size))
        except (TypeError, ValueError):
            return 1

    raw = _coalesce_env(
        "IPFS_DATASETS_PY_LLM_ROUTER_BATCH_WORKERS",
        "IPFS_ACCELERATE_LLM_ROUTER_BATCH_WORKERS",
        "IPFS_ACCELERATE_PY_LLM_ROUTER_BATCH_WORKERS",
        "ipfs_accelerate_py_LLM_ROUTER_BATCH_WORKERS",
    )
    if raw:
        try:
            return max(1, min(int(raw), size))
        except (TypeError, ValueError):
            pass

    provider_key = str(provider or "").strip().lower()
    if provider_key in _LLAMA_CPP_NATIVE_PROVIDER_ALIASES:
        return 1
    return max(1, min(int(default_cap), size))


def _normalize_text_batch_result(
    value: object,
    *,
    expected_count: int,
) -> list[str]:
    if isinstance(value, str):
        if expected_count == 1:
            return [value]
        raise RuntimeError(
            "Batch LLM provider returned a single string for multiple prompts"
        )
    try:
        values = list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise RuntimeError(
            "Batch LLM provider returned a non-iterable result"
        ) from exc
    if len(values) != expected_count:
        raise RuntimeError(
            "Batch LLM provider returned "
            f"{len(values)} results for {expected_count} prompts"
        )
    return [str(item or "") for item in values]


def generate_text_batch(
    prompts: Sequence[str],
    *,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    provider_instance: Optional[LLMProvider] = None,
    deps: Optional[RouterDeps] = None,
    allow_local_fallback: bool = True,
    max_workers: Optional[int] = None,
    use_mesh: bool = False,
    timeout_s: float = 90.0,
    max_route_attempts: int = 3,
    queue_path: Optional[str] = None,
    **kwargs: object,
) -> list[str]:
    """Generate a prompt batch while preserving input order.

    Set ``use_mesh=True`` to submit the batch as ``llm.generate`` tasks so P2P
    workers can run prompts concurrently. For local in-process providers,
    ``max_workers`` controls thread-level concurrency.
    """

    prompt_list = [str(prompt or "") for prompt in list(prompts or [])]
    if not prompt_list:
        return []
    if use_mesh:
        try:
            return generate_text_mesh_batch(
                prompt_list,
                model_name=model_name,
                provider=str(provider or "copilot_cli"),
                timeout_s=timeout_s,
                max_route_attempts=max_route_attempts,
                queue_path=queue_path,
                max_workers=max_workers,
                **kwargs,
            )
        except Exception:
            if bool(kwargs.get("require_mesh") or kwargs.get("mesh_required")):
                raise

    resolved_deps = deps or get_default_router_deps()
    backend = provider_instance or get_llm_provider(provider, deps=resolved_deps)
    text_batch = getattr(backend, "generate_text_batch", None)
    if callable(text_batch):
        return _normalize_text_batch_result(
            text_batch(prompt_list, model_name=model_name, **kwargs),
            expected_count=len(prompt_list),
        )
    generic_batch = getattr(backend, "generate_batch", None)
    if callable(generic_batch):
        return _normalize_text_batch_result(
            generic_batch(prompt_list, model_name=model_name, **kwargs),
            expected_count=len(prompt_list),
        )

    def _generate_one(prompt: str) -> str:
        return str(
            generate_text(
                prompt,
                model_name=model_name,
                provider=provider,
                provider_instance=backend,
                deps=resolved_deps,
                allow_local_fallback=allow_local_fallback,
                **kwargs,
            )
        )

    workers = _batch_worker_count(
        size=len(prompt_list),
        max_workers=max_workers,
        provider=provider,
        default_cap=4,
    )
    if workers <= 1:
        return [_generate_one(prompt) for prompt in prompt_list]

    results: list[Optional[str]] = [None] * len(prompt_list)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_generate_one, prompt): idx
            for idx, prompt in enumerate(prompt_list)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [str(result or "") for result in results]


def generate_text_mesh_batch(
    prompts: Sequence[str],
    *,
    model_name: Optional[str] = None,
    provider: str = "copilot_cli",
    timeout_s: float = 90.0,
    max_route_attempts: int = 3,
    queue_path: Optional[str] = None,
    max_workers: Optional[int] = None,
    **kwargs: object,
) -> list[str]:
    """Generate a prompt batch through the P2P ``llm.generate`` mesh.

    This is intentionally a thin ordered wrapper around ``generate_text_mesh``
    so existing retry, sticky-session, and failover behavior stays centralized.
    """

    prompt_list = [str(prompt or "") for prompt in list(prompts or [])]
    if not prompt_list:
        return []

    provider_norm = str(provider or "").strip().lower() or "copilot_cli"

    def _generate_one(prompt: str) -> str:
        return generate_text_mesh(
            prompt,
            model_name=model_name,
            provider=provider_norm,
            timeout_s=timeout_s,
            max_route_attempts=max_route_attempts,
            queue_path=queue_path,
            **kwargs,
        )

    workers = _batch_worker_count(
        size=len(prompt_list),
        max_workers=max_workers,
        provider=provider_norm,
        default_cap=16,
    )
    if workers <= 1:
        return [_generate_one(prompt) for prompt in prompt_list]

    results: list[Optional[str]] = [None] * len(prompt_list)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_generate_one, prompt): idx
            for idx, prompt in enumerate(prompt_list)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [str(result or "") for result in results]


def chat_completions_batch_create(
    message_batches: Sequence[Sequence[ChatMessage]],
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    provider_instance: Optional[LLMProvider] = None,
    deps: Optional[RouterDeps] = None,
    max_workers: Optional[int] = None,
    **kwargs: object,
) -> list[OpenAICompatResponse]:
    """Run OpenAI-compatible chat completion requests in an ordered batch."""

    batches = [list(messages or []) for messages in list(message_batches or [])]
    if not batches:
        return []

    resolved_deps = deps or get_default_router_deps()
    backend = provider_instance or get_llm_provider(provider, deps=resolved_deps)

    def _create_one(messages: Sequence[ChatMessage]) -> OpenAICompatResponse:
        return chat_completions_create(
            messages=messages,
            model=model,
            provider=provider,
            provider_instance=backend,
            deps=resolved_deps,
            **kwargs,
        )

    workers = _batch_worker_count(
        size=len(batches),
        max_workers=max_workers,
        provider=provider,
        default_cap=4,
    )
    if workers <= 1:
        return [_create_one(messages) for messages in batches]

    results: list[Optional[OpenAICompatResponse]] = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_create_one, messages): idx
            for idx, messages in enumerate(batches)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [result for result in results if result is not None]


def clear_llm_router_caches() -> None:
    """Clear internal provider caches (useful for tests)."""

    _resolve_provider_cached.cache_clear()
    _discover_hf_models_for_pipeline.cache_clear()
    _clear_last_generation_trace()


def _messages_to_prompt(messages: Sequence[ChatMessage]) -> str:
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in list(messages))


def _parse_openai_compat_response(data: dict) -> OpenAICompatResponse:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Chat completions response missing choices")

    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("Chat completions response invalid choice")

    msg = first.get("message")
    content = ""
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        content = msg.get("content", "")
    elif isinstance(first.get("text"), str):
        content = str(first.get("text") or "")

    # Best-effort logprobs extraction.
    logprobs_obj = first.get("logprobs")
    top_logprobs: list[OpenAICompatTopLogProb] = []
    try:
        if isinstance(logprobs_obj, dict):
            content_items = logprobs_obj.get("content")
            if isinstance(content_items, list) and content_items:
                item0 = content_items[0]
                if isinstance(item0, dict):
                    raw_top = item0.get("top_logprobs")
                    if isinstance(raw_top, list):
                        for entry in raw_top:
                            if not isinstance(entry, dict):
                                continue
                            token = entry.get("token")
                            logprob = entry.get("logprob")
                            if isinstance(token, str) and isinstance(logprob, (int, float)):
                                top_logprobs.append(OpenAICompatTopLogProb(token=token, logprob=float(logprob)))
    except Exception:
        top_logprobs = []

    return OpenAICompatResponse(
        choices=[
            OpenAICompatChoice(
                message=OpenAICompatMessage(content=str(content).strip()),
                logprobs=OpenAICompatLogProbs(content=[OpenAICompatLogProbsContentItem(top_logprobs=top_logprobs)]),
            )
        ]
    )


def chat_completions_create(
    *,
    messages: Sequence[ChatMessage],
    model: Optional[str] = None,
    provider: Optional[str] = None,
    provider_instance: Optional[LLMProvider] = None,
    deps: Optional[RouterDeps] = None,
    **kwargs: object,
) -> OpenAICompatResponse:
    """OpenAI-compatible chat completions API via the router.

    Returns a small response object that supports attribute access compatible with
    common OpenAI usage patterns: `response.choices[0].message.content` and
    `response.choices[0].logprobs.content[0].top_logprobs`.

    Notes:
    - Not all providers support logprobs; when unavailable, `top_logprobs` is empty.
    """

    resolved_deps = deps or get_default_router_deps()
    backend = provider_instance or get_llm_provider(provider, deps=resolved_deps)

    # Prefer native chat completions when the provider supports it.
    if isinstance(backend, OpenAIChatCompletionsProvider):
        data = backend.chat_completions(messages, model_name=model, **kwargs)
        return _parse_openai_compat_response(data)

    # Fallback: flatten messages into a single prompt.
    prompt = _messages_to_prompt(messages)
    text = backend.generate(prompt, model_name=model, **kwargs)
    return OpenAICompatResponse(
        choices=[
            OpenAICompatChoice(
                message=OpenAICompatMessage(content=str(text).strip()),
                logprobs=OpenAICompatLogProbs(content=[OpenAICompatLogProbsContentItem(top_logprobs=[])]),
            )
        ]
    )


def get_openai_compat_async_client(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    deps: Optional[RouterDeps] = None,
):
    """Return an object shaped like `openai.AsyncOpenAI()` for chat completions.

    This is intentionally minimal: it only provides `.chat.completions.create(...)`.
    """

    import anyio

    resolved_deps = deps
    default_model = model

    class _ChatCompletions:
        async def create(self, *, messages: list[dict[str, str]], model: str, **kwargs: object) -> OpenAICompatResponse:
            effective_model = default_model or model

            def _run_sync() -> OpenAICompatResponse:
                return chat_completions_create(
                    messages=messages,  # type: ignore[arg-type]
                    model=effective_model,
                    provider=provider,
                    deps=resolved_deps,
                    **kwargs,
                )

            return await anyio.to_thread.run_sync(_run_sync)

    class _Chat:
        def __init__(self) -> None:
            self.completions = _ChatCompletions()

    class _Client:
        def __init__(self) -> None:
            self.chat = _Chat()

    return _Client()


def get_llm_interface(
    *,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    deps: Optional[RouterDeps] = None,
    **config_kwargs: object,
):
    """Return an `LLMInterface` backed by this router.

    This is a convenience bridge for the richer GraphRAG/validation tooling in
    `ipfs_accelerate_py.llm`.

    The returned interface supports:
    - `generate()` returning `{text, usage, ...}`
    - `generate_with_structured_output()` (best-effort JSON + schema validation)
    - embeddings via the optional embedding adapter
    """

    # Lazy import to keep llm_router lightweight.
    from ipfs_accelerate_py.llm.llm_interface import LLMConfig
    from ipfs_accelerate_py.llm.llm_router_interface import RoutedLLMInterface

    cfg_model = model_name or os.getenv("ipfs_accelerate_py_LLM_MODEL") or "mock-llm"
    config = LLMConfig(model_name=str(cfg_model), **{k: v for k, v in config_kwargs.items()})
    return RoutedLLMInterface(config, provider=provider, deps=deps)
