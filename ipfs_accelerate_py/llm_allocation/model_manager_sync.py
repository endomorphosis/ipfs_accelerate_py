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


def populate_router_catalog(manager: Any = None) -> dict[str, Any]:
    """Populate CLI models and the Intelligence Index matrix together."""

    from ipfs_accelerate_py.model_manager import ModelManager

    active = manager or ModelManager(enable_ipfs=False)
    cli = populate_cli_models(active)
    index = populate_intelligence_index(active)
    return {
        "cli": cli,
        "intelligence_index": index,
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
    "list_cli_provider_models",
    "models_for_available_providers",
    "populate_cli_models",
    "populate_intelligence_index",
    "populate_router_catalog",
]
