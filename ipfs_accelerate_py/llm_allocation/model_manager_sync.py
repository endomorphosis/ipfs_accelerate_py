"""Populate ModelManager with CLI models and the Intelligence Index matrix.

CLI discovery never installs tools. Artificial Analysis records are written as
language-model metadata with cost/intelligence in ``performance_metrics``.
"""

from __future__ import annotations

import os
import re
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from ipfs_accelerate_py.llm_allocation.intelligence_index import (
    CATALOG_REVISION,
    CLI_DEFAULT_MODELS,
    INDEX_SOURCE,
    INDEX_VERSION,
    SNAPSHOT_DATE,
    discover_available_providers,
    load_intelligence_index_models,
)
from ipfs_accelerate_py.llm_allocation.paths import CLI_PROVIDERS

HF_DEFAULT_MODELS: tuple[str, ...] = (
    "HuggingFaceH4/zephyr-7b-beta",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Llama-3.1-8B-Instruct",
)

OPENROUTER_LAB_PREFIX: Mapping[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "meta": "meta-llama",
    "spacexai": "x-ai",
    "xai": "x-ai",
    "zai": "z-ai",
    "z-ai": "z-ai",
    "alibaba": "qwen",
    "mistral": "mistralai",
    "deepseek": "deepseek",
    "kimi": "moonshotai",
    "xiaomi": "xiaomi",
}

TYPESAFE_DEFAULT_MODELS: tuple[str, ...] = (
    "jev-latest",
    "jev",
)

OPENROUTER_DEFAULT_MODELS: tuple[str, ...] = (
    "openai/gpt-4o-mini",
    "openai/gpt-6-astra",
    "anthropic/claude-sonnet-5",
    "google/gemini-3.8-flash",
    "z-ai/glm-5.3-flash",
    "x-ai/grok-4.6",
    "meta-llama/llama-4-maverick",
    "qwen/qwen3.8-2.4t",
    "mistralai/mistral-large-3",
)

_MODEL_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{1,79}$")
_CLI_LIST_ARGV: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "grok_cli": (("models",),),
    "muse_code": (("models",),),
    "codex_cli": (("models",),),
    "claude_code": (("models",),),
    "gemini_cli": (("models", "list"), ("models",)),
    "goose_cli": (("models",),),
    "copilot_cli": (("models",),),
    "mistral_vibe": (("models",),),
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _text_io_specs() -> list[Any]:
    from ipfs_accelerate_py.model_manager import DataType, IOSpec

    return [
        IOSpec(name="prompt", data_type=DataType.TEXT, description="Text prompt"),
        IOSpec(name="completion", data_type=DataType.TEXT, description="Generated text"),
    ]


def _parse_model_names(output: str) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for raw in str(output or "").splitlines():
        token = raw.strip().split()[0] if raw.strip() else ""
        token = token.strip("`,\"'")
        if not _MODEL_TOKEN.match(token):
            continue
        lowered = token.casefold()
        if lowered in {"models", "model", "name", "id", "provider"}:
            continue
        if lowered not in seen:
            seen.add(lowered)
            names.append(token)
        if len(names) >= 32:
            break
    return tuple(names)


def list_cli_provider_models(
    provider: str,
    *,
    executable: str = "",
    timeout_seconds: float = 4.0,
) -> tuple[str, ...]:
    """Best-effort CLI model list. Falls back to known defaults. Never installs."""

    key = str(provider or "").strip().casefold()
    defaults = CLI_DEFAULT_MODELS.get(key, ())
    exe = str(executable or "").strip()
    if not exe:
        try:
            from ipfs_accelerate_py.cli_runtime.installers.catalog import discover_cli_tool

            discovered = discover_cli_tool(key)
            exe = str(getattr(discovered, "executable", "") or "")
        except Exception:
            exe = ""
    if not exe:
        return defaults
    for argv in _CLI_LIST_ARGV.get(key, ()):
        try:
            completed = subprocess.run(
                [exe, *argv],
                capture_output=True,
                text=True,
                timeout=max(0.5, float(timeout_seconds)),
                check=False,
                env=os.environ.copy(),
            )
        except Exception:
            continue
        parsed = _parse_model_names((completed.stdout or "") + "\n" + (completed.stderr or ""))
        if parsed:
            return parsed
    return defaults


def populate_cli_models(
    manager: Any = None,
    *,
    timeout_seconds: float = 4.0,
) -> dict[str, Any]:
    """Register installed CLI tools and their models in ModelManager."""

    from ipfs_accelerate_py.model_manager import ModelManager, ModelMetadata, ModelType

    active = manager or ModelManager(enable_ipfs=False)
    now = _utcnow()
    added = 0
    providers: dict[str, Any] = {}
    try:
        from ipfs_accelerate_py.llm_allocation.cli_status import cli_tools_status

        status = cli_tools_status()
        tools = status.get("tools") or {}
    except Exception:
        tools = {}

    for provider in sorted(CLI_PROVIDERS):
        row = tools.get(provider) or {}
        installed = bool(row.get("installed"))
        executable = str(row.get("executable") or "")
        if not installed and not executable:
            providers[provider] = {"installed": False, "models": []}
            continue
        models = list_cli_provider_models(
            provider,
            executable=executable,
            timeout_seconds=timeout_seconds,
        )
        model_ids: list[str] = []
        for model_name in models:
            model_id = f"cli:{provider}:{model_name}"
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                model_type=ModelType.LANGUAGE_MODEL,
                architecture="decoder_only",
                inputs=_text_io_specs()[:1],
                outputs=_text_io_specs()[1:],
                supported_backends=[provider],
                performance_metrics={
                    "source": "cli",
                    "provider": provider,
                },
                tags=["cli", provider, "llm-router"],
                description=f"CLI model advertised by {provider}",
                created_at=now,
                updated_at=now,
            )
            if active.add_model(metadata):
                added += 1
                model_ids.append(model_id)
        providers[provider] = {
            "installed": True,
            "executable": executable,
            "models": list(models),
            "model_ids": model_ids,
        }
    return {"added": added, "providers": providers}


def _bulk_add(manager: Any, records: Sequence[Any]) -> int:
    added = 0
    lock = getattr(manager, "_model_lock", None)
    models = getattr(manager, "models", None)
    if not isinstance(models, dict):
        for metadata in records:
            if manager.add_model(metadata):
                added += 1
        return added
    context = lock if lock is not None else nullcontext()
    with context:
        for metadata in records:
            metadata.updated_at = _utcnow()
            models[metadata.model_id] = metadata
            added += 1
        saver = getattr(manager, "_save_data", None)
        if callable(saver):
            saver()
    sync = getattr(manager, "_sync_legacy_catalog", None)
    if callable(sync):
        try:
            sync()
        except Exception:
            pass
    return added


def populate_intelligence_index(manager: Any = None) -> dict[str, Any]:
    """Write the full Artificial Analysis cost/intelligence matrix into ModelManager."""

    from ipfs_accelerate_py.model_manager import ModelManager, ModelMetadata, ModelType

    active = manager or ModelManager(enable_ipfs=False)
    now = _utcnow()
    records = []
    for model in load_intelligence_index_models():
        model_id = f"aa:{INDEX_VERSION}:{model.slug}"
        records.append(
            ModelMetadata(
                model_id=model_id,
                model_name=model.model_name,
                model_type=ModelType.LANGUAGE_MODEL,
                architecture="decoder_only",
                inputs=_text_io_specs()[:1],
                outputs=_text_io_specs()[1:],
                huggingface_config={
                    "artificial_analysis_slug": model.slug,
                    "lab": model.lab,
                    "reasoning_effort": model.reasoning_effort,
                    "index_version": INDEX_VERSION,
                    "snapshot_date": SNAPSHOT_DATE,
                },
                supported_backends=list(model.providers),
                performance_metrics={
                    "intelligence_index": model.intelligence,
                    "cost_usd_per_task": model.cost_usd_per_task,
                    "index_version": INDEX_VERSION,
                    "pareto": model.pareto,
                    "reasoning_effort": model.reasoning_effort,
                    "catalog_revision": CATALOG_REVISION,
                },
                tags=[
                    "artificial-analysis",
                    f"intelligence-index-v{INDEX_VERSION}",
                    "llm-router",
                    *(["pareto"] if model.pareto else []),
                ],
                source_url=INDEX_SOURCE,
                license="benchmark-snapshot",
                description=model.name,
                created_at=now,
                updated_at=now,
            )
        )
    added = _bulk_add(active, records)
    return {
        "added": added,
        "index_version": INDEX_VERSION,
        "catalog_revision": CATALOG_REVISION,
        "records": added,
    }


def _aa_metrics_for_name(model_name: str) -> dict[str, Any]:
    needle = str(model_name or "").strip().casefold().rsplit("/", 1)[-1]
    best: dict[str, Any] = {}
    if not needle:
        return best
    for model in load_intelligence_index_models():
        if model.model_name.casefold() == needle and model.cost_usd_per_task > 0:
            best = {
                "intelligence_index": model.intelligence,
                "cost_usd_per_task": model.cost_usd_per_task,
                "index_version": INDEX_VERSION,
                "reasoning_effort": model.reasoning_effort,
                "catalog_revision": CATALOG_REVISION,
            }
            break
    return best


def list_openrouter_models(
    *,
    timeout_seconds: float = 4.0,
    live: bool = False,
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[str, ...]:
    """Return OpenRouter model ids. Live fetch is opt-in and fail-soft."""

    defaults = list(OPENROUTER_DEFAULT_MODELS)
    for model in load_intelligence_index_models():
        if "openrouter" not in model.providers:
            continue
        lab = str(model.lab or "").casefold()
        prefix = (
            OPENROUTER_LAB_PREFIX.get(lab)
            or OPENROUTER_LAB_PREFIX.get(lab.replace(" ", "-"))
            or OPENROUTER_LAB_PREFIX.get(lab.replace(" ", ""))
        )
        if not prefix:
            continue
        ident = f"{prefix}/{model.model_name}"
        if ident not in defaults:
            defaults.append(ident)
    if not live:
        return tuple(defaults[:64])
    env = os.environ if environ is None else environ
    token = ""
    for name in (
        "OPENROUTER_API_KEY",
        "ipfs_accelerate_py_OPENROUTER_API_KEY",
        "IPFS_ACCELERATE_PY_OPENROUTER_API_KEY",
    ):
        token = str(env.get(name) or "").strip()
        if token:
            break
    if not token:
        return tuple(defaults[:64])
    try:
        import json
        import urllib.request

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=max(0.5, float(timeout_seconds))) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        rows = payload.get("data") if isinstance(payload, dict) else None
        live_ids: list[str] = []
        for row in rows or ():
            ident = str((row or {}).get("id") or "").strip()
            if ident and ident not in live_ids:
                live_ids.append(ident)
            if len(live_ids) >= 64:
                break
        if live_ids:
            return tuple(live_ids)
    except Exception:
        pass
    return tuple(defaults[:64])


def list_hf_inference_models(*, live: bool = False) -> tuple[str, ...]:
    """Return Hugging Face Inference API model ids. Live listing is fail-soft."""

    names = list(HF_DEFAULT_MODELS)
    if live:
        try:
            from ipfs_accelerate_py.llm_router import (
                _hf_live_model_manager_candidate_models,
            )

            for item in _hf_live_model_manager_candidate_models() or ():
                ident = str(item or "").strip()
                if ident and ident not in names:
                    names.append(ident)
        except Exception:
            pass
    return tuple(names[:64])


def populate_openrouter_models(
    manager: Any = None,
    *,
    live: bool = False,
) -> dict[str, Any]:
    """Register OpenRouter models in ModelManager."""

    from ipfs_accelerate_py.model_manager import ModelManager, ModelMetadata, ModelType

    active = manager or ModelManager(enable_ipfs=False)
    now = _utcnow()
    models = list_openrouter_models(live=live)
    records = []
    for model_name in models:
        metrics = {"source": "openrouter", "provider": "openrouter"}
        metrics.update(_aa_metrics_for_name(model_name))
        records.append(
            ModelMetadata(
                model_id=f"openrouter:{model_name}",
                model_name=model_name,
                model_type=ModelType.LANGUAGE_MODEL,
                architecture="decoder_only",
                inputs=_text_io_specs()[:1],
                outputs=_text_io_specs()[1:],
                supported_backends=["openrouter"],
                performance_metrics=metrics,
                tags=["api", "openrouter", "llm-router"],
                description=f"OpenRouter model {model_name}",
                source_url="https://openrouter.ai/api/v1/models",
                created_at=now,
                updated_at=now,
            )
        )
    added = _bulk_add(active, records)
    return {"added": added, "models": list(models)}


def list_typesafe_models(*, live: bool = False) -> tuple[str, ...]:
    """Return TypeSafe System One model ids. Live listing is fail-soft."""

    try:
        from ipfs_accelerate_py.typesafe_inference import list_typesafe_models as _list

        return _list(live=live)
    except Exception:
        return TYPESAFE_DEFAULT_MODELS


def populate_typesafe_models(
    manager: Any = None,
    *,
    live: bool = False,
) -> dict[str, Any]:
    """Register TypeSafe System One models in ModelManager."""

    from ipfs_accelerate_py.model_manager import ModelManager, ModelMetadata, ModelType

    active = manager or ModelManager(enable_ipfs=False)
    now = _utcnow()
    models = list_typesafe_models(live=live)
    records = []
    for model_name in models:
        records.append(
            ModelMetadata(
                model_id=f"typesafe:{model_name}",
                model_name=model_name,
                model_type=ModelType.LANGUAGE_MODEL,
                architecture="decoder_only",
                inputs=_text_io_specs()[:1],
                outputs=_text_io_specs()[1:],
                supported_backends=["typesafe"],
                performance_metrics={"source": "typesafe", "provider": "typesafe"},
                tags=["api", "typesafe", "system-one", "llm-router"],
                description=f"TypeSafe System One model {model_name}",
                source_url="https://docs.typesafe.ai/sdk/python/usage",
                created_at=now,
                updated_at=now,
            )
        )
    added = _bulk_add(active, records)
    return {"added": added, "models": list(models)}


def populate_hf_inference_models(
    manager: Any = None,
    *,
    live: bool = False,
) -> dict[str, Any]:
    """Register Hugging Face Inference API models in ModelManager."""

    from ipfs_accelerate_py.model_manager import ModelManager, ModelMetadata, ModelType

    active = manager or ModelManager(enable_ipfs=False)
    now = _utcnow()
    models = list_hf_inference_models(live=live)
    records = []
    for model_name in models:
        records.append(
            ModelMetadata(
                model_id=f"hf:{model_name}",
                model_name=model_name,
                model_type=ModelType.LANGUAGE_MODEL,
                architecture="decoder_only",
                inputs=_text_io_specs()[:1],
                outputs=_text_io_specs()[1:],
                supported_backends=["hf_inference_api"],
                performance_metrics={"source": "hf_inference_api", "provider": "hf_inference_api"},
                tags=["api", "hf_inference_api", "llm-router"],
                description=f"Hugging Face Inference API model {model_name}",
                source_url="https://huggingface.co",
                created_at=now,
                updated_at=now,
            )
        )
    added = _bulk_add(active, records)
    return {"added": added, "models": list(models)}


def populate_router_catalog(manager: Any = None, *, live: bool = False) -> dict[str, Any]:
    """Populate CLI, API, and Intelligence Index catalogs together."""

    from ipfs_accelerate_py.model_manager import ModelManager

    active = manager or ModelManager(enable_ipfs=False)
    cli = populate_cli_models(active)
    index = populate_intelligence_index(active)
    openrouter = populate_openrouter_models(active, live=live)
    hf = populate_hf_inference_models(active, live=live)
    typesafe = populate_typesafe_models(active, live=live)
    return {
        "cli": cli,
        "intelligence_index": index,
        "openrouter": openrouter,
        "hf_inference_api": hf,
        "typesafe": typesafe,
        "available_providers": list(discover_available_providers()),
    }


def models_for_available_providers(
    manager: Any = None,
    *,
    available_providers: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """Return AA matrix rows whose backends intersect live providers."""

    live = tuple(available_providers) if available_providers is not None else discover_available_providers()
    allowed = {str(name).strip().casefold() for name in live if str(name).strip()}
    rows: list[dict[str, Any]] = []
    if manager is not None:
        for metadata in getattr(manager, "models", {}).values():
            metrics = getattr(metadata, "performance_metrics", None) or {}
            if "intelligence_index" not in metrics:
                continue
            backends = [
                str(item)
                for item in (getattr(metadata, "supported_backends", None) or [])
                if str(item).strip()
            ]
            matched = [name for name in backends if not allowed or name.casefold() in allowed]
            if allowed and not matched:
                continue
            rows.append(
                {
                    "model_id": metadata.model_id,
                    "model_name": metadata.model_name,
                    "intelligence": float(metrics.get("intelligence_index") or 0.0),
                    "cost_usd_per_task": float(metrics.get("cost_usd_per_task") or 0.0),
                    "providers": matched or backends,
                    "reasoning_effort": str(metrics.get("reasoning_effort") or ""),
                    "pareto": bool(metrics.get("pareto")),
                }
            )
        rows.sort(key=lambda item: (item["cost_usd_per_task"], -item["intelligence"]))
        if rows:
            return rows
    from ipfs_accelerate_py.llm_allocation.intelligence_index import intelligence_cost_matrix

    return intelligence_cost_matrix(available_only=True, available_providers=live)


__all__ = [
    "HF_DEFAULT_MODELS",
    "OPENROUTER_DEFAULT_MODELS",
    "TYPESAFE_DEFAULT_MODELS",
    "list_cli_provider_models",
    "list_hf_inference_models",
    "list_openrouter_models",
    "list_typesafe_models",
    "models_for_available_providers",
    "populate_cli_models",
    "populate_hf_inference_models",
    "populate_intelligence_index",
    "populate_openrouter_models",
    "populate_typesafe_models",
    "populate_router_catalog",
]
