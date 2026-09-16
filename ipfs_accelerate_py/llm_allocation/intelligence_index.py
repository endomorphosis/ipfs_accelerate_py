"""Artificial Analysis Intelligence Index v4.3 cost-efficient model routing.

Snapshot of Intelligence Index scores and weighted cost per task from
https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-3
(retrieved 2026-09-09 via the published v4.3 records). The supervisor uses this
to pick the cheapest model that still meets a task-difficulty intelligence floor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

INDEX_VERSION = "4.3"
INDEX_SOURCE = (
    "https://artificialanalysis.ai/articles/"
    "artificial-analysis-intelligence-index-v4-3"
)
SNAPSHOT_DATE = "2026-09-09"
CATALOG_REVISION = "aa-intelligence-index-v4.3"

AUTO_MODEL_NAMES = frozenset({"", "auto", "efficient"})

_OPENAI_PROVIDERS = ("openai", "openrouter", "codex_cli", "copilot_cli", "copilot_sdk")
_GROK_PROVIDERS = ("grok_cli", "xai")
_MUSE_PROVIDERS = ("muse_code", "meta_ai", "goose_cli")
_CLAUDE_PROVIDERS = ("claude_code", "claude_py")
_GEMINI_PROVIDERS = ("gemini_cli", "gemini_py", "openrouter")
_MISTRAL_PROVIDERS = ("mistral_vibe", "openrouter")
_OPENROUTER_PROVIDERS = ("openrouter",)

LAB_PROVIDERS: Mapping[str, tuple[str, ...]] = {
    "openai": _OPENAI_PROVIDERS,
    "anthropic": _CLAUDE_PROVIDERS,
    "meta": _MUSE_PROVIDERS,
    "xai": _GROK_PROVIDERS,
    "google": _GEMINI_PROVIDERS,
    "mistral": _MISTRAL_PROVIDERS,
    "zai": _OPENROUTER_PROVIDERS,
    "alibaba": _OPENROUTER_PROVIDERS,
    "deepseek": _OPENROUTER_PROVIDERS,
    "kimi": _OPENROUTER_PROVIDERS,
    "xiaomi": _OPENROUTER_PROVIDERS,
    "minimax": _OPENROUTER_PROVIDERS,
    "thinking-machines": _OPENROUTER_PROVIDERS,
    "nvidia": _OPENROUTER_PROVIDERS,
    "ibm": _OPENROUTER_PROVIDERS,
    "tencent": _OPENROUTER_PROVIDERS,
    "longcat": _OPENROUTER_PROVIDERS,
    "inclusionai": _OPENROUTER_PROVIDERS,
    "multiversecomputing": _OPENROUTER_PROVIDERS,
    "cohere": _OPENROUTER_PROVIDERS,
    "inception": _OPENROUTER_PROVIDERS,
    "arcee": _OPENROUTER_PROVIDERS,
    "upstage": _OPENROUTER_PROVIDERS,
    "celeris": _OPENROUTER_PROVIDERS,
}

CLI_DEFAULT_MODELS: Mapping[str, tuple[str, ...]] = {
    "grok_cli": ("grok-4.6",),
    "muse_code": ("muse-spark-1.3", "muse-spark-1.2"),
    "codex_cli": ("gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"),
    "claude_code": ("claude-fable-5.1", "claude-opus-5", "claude-sonnet-5"),
    "gemini_cli": ("gemini-3.8-flash", "gemini-3.7-flash"),
    "goose_cli": ("muse-spark-1.3",),
    "copilot_cli": ("gpt-5.6-sol",),
    "mistral_vibe": ("mistral-large",),
}

TASK_KIND_INTELLIGENCE: Mapping[str, float] = {
    "trivial": 18.0,
    "easy": 26.0,
    "inventory": 26.0,
    "extraction": 26.0,
    "rescan": 26.0,
    "retirement": 26.0,
    "legal": 26.0,
    "standard": 38.0,
    "review": 38.0,
    "unblock_review": 38.0,
    "planning": 38.0,
    "proposal": 38.0,
    "validation": 38.0,
    "facade": 38.0,
    "boundary": 38.0,
    "state": 38.0,
    "coding": 41.0,
    "implementation": 41.0,
    "repair": 41.0,
    "agent": 45.0,
    "merge": 45.0,
    "hard": 48.0,
    "scientific": 48.0,
    "rescue": 48.0,
    "frontier": 52.0,
}


@dataclass(frozen=True)
class IntelligenceIndexModel:
    """One Intelligence Index v4.3 configuration the router can select."""

    slug: str
    name: str
    lab: str
    intelligence: float
    cost_usd_per_task: float
    model_name: str
    providers: tuple[str, ...]
    reasoning_effort: str = ""
    pareto: bool = False


@dataclass(frozen=True)
class EfficientRoute:
    """Resolved model/provider for a difficulty floor."""

    model_name: str
    provider: str
    reasoning_effort: str
    intelligence: float
    cost_usd_per_task: float
    slug: str
    task_kind: str
    min_intelligence: float
    auto_selected: bool
    catalog_revision: str = CATALOG_REVISION
    index_version: str = INDEX_VERSION


# Pareto frontier plus supervisor-routable near-frontier models.
# Costs are weighted USD per Intelligence Index task.
INTELLIGENCE_INDEX_MODELS: tuple[IntelligenceIndexModel, ...] = (
    IntelligenceIndexModel(
        slug="mimo-v2-5-0424",
        name="MiMo-V2.5",
        lab="Xiaomi",
        intelligence=22.3039,
        cost_usd_per_task=0.0191,
        model_name="mimo-v2.5",
        providers=_OPENROUTER_PROVIDERS,
        pareto=True,
    ),
    IntelligenceIndexModel(
        slug="mimo-v2-5-pro",
        name="MiMo-V2.5-Pro",
        lab="Xiaomi",
        intelligence=26.4078,
        cost_usd_per_task=0.0541,
        model_name="mimo-v2.5-pro",
        providers=_OPENROUTER_PROVIDERS,
        pareto=True,
    ),
    IntelligenceIndexModel(
        slug="gpt-5-6-luna-xhigh",
        name="GPT-5.6 Luna (xhigh)",
        lab="OpenAI",
        intelligence=34.7720,
        cost_usd_per_task=0.0853,
        model_name="gpt-5.6-luna",
        providers=_OPENAI_PROVIDERS,
        reasoning_effort="xhigh",
        pareto=True,
    ),
    IntelligenceIndexModel(
        slug="gpt-5-6-luna",
        name="GPT-5.6 Luna (max)",
        lab="OpenAI",
        intelligence=37.5048,
        cost_usd_per_task=0.1783,
        model_name="gpt-5.6-luna",
        providers=_OPENAI_PROVIDERS,
        reasoning_effort="max",
        pareto=True,
    ),
    IntelligenceIndexModel(
        slug="glm-5-3-flash",
        name="GLM-5.3-Flash",
        lab="Z AI",
        intelligence=41.9074,
        cost_usd_per_task=0.2533,
        model_name="glm-5.3-flash",
        providers=_OPENROUTER_PROVIDERS,
        reasoning_effort="max",
        pareto=True,
    ),
    IntelligenceIndexModel(
        slug="gpt-5-6-sol-high",
        name="GPT-5.6 Sol (high)",
        lab="OpenAI",
        intelligence=42.4992,
        cost_usd_per_task=0.8080,
        model_name="gpt-5.6-sol",
        providers=_OPENAI_PROVIDERS,
        reasoning_effort="high",
        pareto=True,
    ),
    IntelligenceIndexModel(
        slug="gpt-5-6-sol",
        name="GPT-5.6 Sol (max)",
        lab="OpenAI",
        intelligence=47.0614,
        cost_usd_per_task=1.9885,
        model_name="gpt-5.6-sol",
        providers=_OPENAI_PROVIDERS,
        reasoning_effort="max",
    ),
    IntelligenceIndexModel(
        slug="gpt-5-6-terra",
        name="GPT-5.6 Terra (max)",
        lab="OpenAI",
        intelligence=42.2515,
        cost_usd_per_task=1.3987,
        model_name="gpt-5.6-terra",
        providers=_OPENAI_PROVIDERS,
        reasoning_effort="max",
    ),
    IntelligenceIndexModel(
        slug="gpt-6-astra-low",
        name="GPT-6 Astra (low)",
        lab="OpenAI",
        intelligence=45.9945,
        cost_usd_per_task=0.8175,
        model_name="gpt-6-astra",
        providers=_OPENAI_PROVIDERS,
        reasoning_effort="low",
        pareto=True,
    ),
    IntelligenceIndexModel(
        slug="gpt-6-astra-medium",
        name="GPT-6 Astra (medium)",
        lab="OpenAI",
        intelligence=49.6685,
        cost_usd_per_task=1.5406,
        model_name="gpt-6-astra",
        providers=_OPENAI_PROVIDERS,
        reasoning_effort="medium",
        pareto=True,
    ),
    IntelligenceIndexModel(
        slug="gpt-6-astra-high",
        name="GPT-6 Astra (high)",
        lab="OpenAI",
        intelligence=51.0481,
        cost_usd_per_task=1.7214,
        model_name="gpt-6-astra",
        providers=_OPENAI_PROVIDERS,
        reasoning_effort="high",
        pareto=True,
    ),
    IntelligenceIndexModel(
        slug="gpt-6-astra-xhigh",
        name="GPT-6 Astra (xhigh)",
        lab="OpenAI",
        intelligence=52.5059,
        cost_usd_per_task=2.3088,
        model_name="gpt-6-astra",
        providers=_OPENAI_PROVIDERS,
        reasoning_effort="xhigh",
        pareto=True,
    ),
    IntelligenceIndexModel(
        slug="gpt-6-astra",
        name="GPT-6 Astra (max)",
        lab="OpenAI",
        intelligence=52.8141,
        cost_usd_per_task=3.2575,
        model_name="gpt-6-astra",
        providers=_OPENAI_PROVIDERS,
        reasoning_effort="max",
        pareto=True,
    ),
    IntelligenceIndexModel(
        slug="claude-fable-5-1-xhigh",
        name="Claude Fable 5.1 (xhigh with fallback)",
        lab="Anthropic",
        intelligence=53.1840,
        cost_usd_per_task=5.9783,
        model_name="claude-fable-5.1",
        providers=_CLAUDE_PROVIDERS,
        reasoning_effort="xhigh",
        pareto=True,
    ),
    IntelligenceIndexModel(
        slug="claude-fable-5-1",
        name="Claude Fable 5.1 (max with fallback)",
        lab="Anthropic",
        intelligence=53.3738,
        cost_usd_per_task=7.6297,
        model_name="claude-fable-5.1",
        providers=_CLAUDE_PROVIDERS,
        reasoning_effort="max",
        pareto=True,
    ),
    IntelligenceIndexModel(
        slug="grok-4-6-low",
        name="Grok 4.6 (low)",
        lab="SpaceXAI",
        intelligence=35.4148,
        cost_usd_per_task=0.4753,
        model_name="grok-4.6",
        providers=_GROK_PROVIDERS,
        reasoning_effort="low",
    ),
    IntelligenceIndexModel(
        slug="grok-4-6-medium",
        name="Grok 4.6 (medium)",
        lab="SpaceXAI",
        intelligence=43.0112,
        cost_usd_per_task=1.4963,
        model_name="grok-4.6",
        providers=_GROK_PROVIDERS,
        reasoning_effort="medium",
    ),
    IntelligenceIndexModel(
        slug="grok-4-6",
        name="Grok 4.6 (high)",
        lab="SpaceXAI",
        intelligence=44.4050,
        cost_usd_per_task=1.8589,
        model_name="grok-4.6",
        providers=_GROK_PROVIDERS,
        reasoning_effort="high",
    ),
    IntelligenceIndexModel(
        slug="muse-spark-1-3-xhigh",
        name="Muse Spark 1.3 (xhigh)",
        lab="Meta",
        intelligence=45.1612,
        cost_usd_per_task=1.3678,
        model_name="muse-spark-1.3",
        providers=_MUSE_PROVIDERS,
        reasoning_effort="xhigh",
    ),
    IntelligenceIndexModel(
        slug="muse-spark-1-3",
        name="Muse Spark 1.3 (max)",
        lab="Meta",
        intelligence=48.1690,
        cost_usd_per_task=1.6049,
        model_name="muse-spark-1.3",
        providers=_MUSE_PROVIDERS,
        reasoning_effort="max",
    ),
)


def providers_for_lab(lab_slug: str) -> tuple[str, ...]:
    """Map an Artificial Analysis lab slug onto llm_router provider names."""

    return LAB_PROVIDERS.get(str(lab_slug or "").strip().casefold(), _OPENROUTER_PROVIDERS)


def _snapshot_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "intelligence_index_v4_3.json"


@lru_cache(maxsize=1)
def load_intelligence_index_models() -> tuple[IntelligenceIndexModel, ...]:
    """Load the full Intelligence Index v4.3 cost/intelligence matrix."""

    path = _snapshot_path()
    if not path.is_file():
        return INTELLIGENCE_INDEX_MODELS
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return INTELLIGENCE_INDEX_MODELS
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        return INTELLIGENCE_INDEX_MODELS
    loaded: list[IntelligenceIndexModel] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        try:
            intel = float(row.get("intelligence") or 0.0)
        except (TypeError, ValueError):
            continue
        if intel <= 0:
            continue
        raw_cost = row.get("cost_usd_per_task")
        try:
            cost = float(raw_cost) if raw_cost is not None else 0.0
        except (TypeError, ValueError):
            cost = 0.0
        lab_slug = str(row.get("lab_slug") or "")
        loaded.append(
            IntelligenceIndexModel(
                slug=str(row.get("slug") or ""),
                name=str(row.get("name") or row.get("slug") or ""),
                lab=str(row.get("lab") or ""),
                intelligence=intel,
                cost_usd_per_task=cost,
                model_name=str(row.get("model_name") or ""),
                providers=providers_for_lab(lab_slug),
                reasoning_effort=str(row.get("reasoning_effort") or ""),
                pareto=bool(row.get("pareto")),
            )
        )
    return tuple(loaded) or INTELLIGENCE_INDEX_MODELS


def discover_available_providers(
    *,
    include_model_manager: bool = False,
) -> tuple[str, ...]:
    """Return llm_router / CLI / model-manager providers that can actually run."""

    names: list[str] = []
    seen: set[str] = set()

    def _add(value: object) -> None:
        key = str(value or "").strip()
        lowered = key.casefold()
        if not key or lowered in seen:
            return
        seen.add(lowered)
        names.append(key)

    try:
        from ipfs_accelerate_py import llm_router

        for descriptor in llm_router.list_providers():
            state = getattr(descriptor, "state", None)
            if (
                getattr(state, "configured", None) is True
                or getattr(state, "authorized", None) is True
                or getattr(state, "routable", None) is True
            ):
                _add(getattr(descriptor, "name", ""))
    except Exception:
        pass

    try:
        from ipfs_accelerate_py.llm_allocation.cli_status import cli_tools_status

        status = cli_tools_status()
        for name in status.get("ready") or ():
            _add(name)
        for name, row in (status.get("tools") or {}).items():
            if isinstance(row, dict) and (
                row.get("ready") or (row.get("installed") and row.get("authenticated"))
            ):
                _add(name)
    except Exception:
        pass

    if include_model_manager:
        try:
            from ipfs_accelerate_py.model_manager import ModelManager

            manager = ModelManager(enable_ipfs=False, project_legacy_models=False)
            for metadata in getattr(manager, "models", {}).values():
                backends = getattr(metadata, "supported_backends", None) or []
                tags = getattr(metadata, "tags", None) or []
                if "cli" in tags or "artificial-analysis" in tags:
                    for backend in backends:
                        _add(backend)
        except Exception:
            pass
    return tuple(names)


def intelligence_cost_matrix(
    *,
    available_only: bool = False,
    available_providers: Optional[Sequence[str]] = None,
    min_intelligence: float = 0.0,
) -> list[dict[str, Any]]:
    """Return the full cost/intelligence matrix, optionally filtered to live providers."""

    live = (
        tuple(available_providers)
        if available_providers is not None
        else (discover_available_providers() if available_only else ())
    )
    allowed = {str(name).strip().casefold() for name in live if str(name).strip()}
    floor = float(min_intelligence or 0.0)
    rows: list[dict[str, Any]] = []
    for model in load_intelligence_index_models():
        if model.intelligence + 1e-9 < floor:
            continue
        matched = [
            provider
            for provider in model.providers
            if not allowed or provider.casefold() in allowed
        ]
        if allowed and not matched:
            continue
        rows.append(
            {
                "slug": model.slug,
                "name": model.name,
                "lab": model.lab,
                "intelligence": model.intelligence,
                "cost_usd_per_task": model.cost_usd_per_task,
                "model_name": model.model_name,
                "reasoning_effort": model.reasoning_effort,
                "providers": list(matched or model.providers),
                "pareto": model.pareto,
                "available": bool(matched) if allowed else None,
            }
        )
    rows.sort(key=lambda item: (item["cost_usd_per_task"], -item["intelligence"]))
    return rows


def intelligence_floor_for_task(
    task_kind: str = "",
    *,
    min_intelligence: float = 0.0,
) -> float:
    """Map a supervisor task kind onto an Intelligence Index floor."""

    if min_intelligence and float(min_intelligence) > 0:
        return float(min_intelligence)
    key = str(task_kind or "standard").strip().casefold()
    return float(TASK_KIND_INTELLIGENCE.get(key, TASK_KIND_INTELLIGENCE["standard"]))


def infer_task_kind(
    task_kind: str = "",
    *,
    backend_label: str = "",
    env_prefix: str = "",
) -> str:
    """Infer a closed task-kind token from invocation metadata."""

    explicit = str(task_kind or "").strip().casefold()
    if explicit:
        return explicit if explicit in TASK_KIND_INTELLIGENCE else "standard"
    blob = f"{backend_label} {env_prefix}".casefold()
    if "implement" in blob:
        return "implementation"
    if "review" in blob:
        return "review"
    if "legal" in blob or "parser" in blob:
        return "easy"
    if "plan" in blob or "proposal" in blob:
        return "planning"
    return "standard"


def is_auto_model_name(
    model_name: str,
    *,
    provider: str = "",
    default_model: str = "",
) -> bool:
    """True when the supervisor should pick an efficient model."""

    model = str(model_name or "").strip().casefold()
    if model in AUTO_MODEL_NAMES:
        return True
    if str(provider or "").strip():
        return False
    default = str(default_model or "").strip().casefold()
    return bool(default) and model == default


def _provider_match(
    model: IntelligenceIndexModel,
    provider: str,
    available: Optional[Sequence[str]],
) -> Optional[str]:
    wanted = str(provider or "").strip().casefold()
    allowed = {
        str(name).strip().casefold()
        for name in (available or ())
        if str(name).strip()
    }
    for candidate in model.providers:
        key = candidate.casefold()
        if wanted and key != wanted:
            continue
        if allowed and key not in allowed:
            continue
        return candidate
    return None


def select_efficient_model(
    *,
    min_intelligence: float = 38.0,
    provider: str = "",
    available_providers: Optional[Sequence[str]] = None,
    models: Optional[Sequence[IntelligenceIndexModel]] = None,
) -> IntelligenceIndexModel:
    """Return the cheapest model meeting *min_intelligence*.

    If none meet the floor, return the cheapest model at the highest available
    intelligence (still filtered by provider constraints).
    """

    floor = float(min_intelligence or 0.0)
    catalog = tuple(models) if models is not None else load_intelligence_index_models()
    matched: list[tuple[IntelligenceIndexModel, str]] = []
    for model in catalog:
        if model.cost_usd_per_task <= 0:
            continue
        chosen = _provider_match(model, provider, available_providers)
        if chosen is None:
            continue
        matched.append((model, chosen))
    if not matched:
        raise ValueError("no Intelligence Index models match the provider filter")
    meeting = [(model, chosen) for model, chosen in matched if model.intelligence + 1e-9 >= floor]
    pool = meeting or matched
    if not meeting:
        top = max(model.intelligence for model, _ in pool)
        pool = [(model, chosen) for model, chosen in pool if model.intelligence >= top - 1e-9]
    model, chosen = min(
        pool,
        key=lambda item: (item[0].cost_usd_per_task, -item[0].intelligence, item[0].slug),
    )
    return IntelligenceIndexModel(
        slug=model.slug,
        name=model.name,
        lab=model.lab,
        intelligence=model.intelligence,
        cost_usd_per_task=model.cost_usd_per_task,
        model_name=model.model_name,
        providers=(chosen,),
        reasoning_effort=model.reasoning_effort,
        pareto=model.pareto,
    )


def select_efficient_route(
    *,
    model_name: str = "",
    provider: str = "",
    task_kind: str = "",
    min_intelligence: float = 0.0,
    backend_label: str = "",
    env_prefix: str = "",
    available_providers: Optional[Iterable[str]] = None,
    default_model: str = "",
    force: bool = False,
) -> EfficientRoute:
    """Resolve an efficient llm_router route for a supervisor call."""

    kind = infer_task_kind(
        task_kind, backend_label=backend_label, env_prefix=env_prefix
    )
    floor = intelligence_floor_for_task(kind, min_intelligence=min_intelligence)
    auto = bool(force) or is_auto_model_name(
        model_name, provider=provider, default_model=default_model
    )
    if not auto:
        return EfficientRoute(
            model_name=str(model_name or "").strip(),
            provider=str(provider or "").strip(),
            reasoning_effort="",
            intelligence=0.0,
            cost_usd_per_task=0.0,
            slug="",
            task_kind=kind,
            min_intelligence=floor,
            auto_selected=False,
        )
    available = tuple(available_providers) if available_providers is not None else None
    selected = select_efficient_model(
        min_intelligence=floor,
        provider=provider,
        available_providers=available,
    )
    return EfficientRoute(
        model_name=selected.model_name,
        provider=selected.providers[0],
        reasoning_effort=selected.reasoning_effort,
        intelligence=selected.intelligence,
        cost_usd_per_task=selected.cost_usd_per_task,
        slug=selected.slug,
        task_kind=kind,
        min_intelligence=floor,
        auto_selected=True,
    )


__all__ = [
    "AUTO_MODEL_NAMES",
    "CATALOG_REVISION",
    "EfficientRoute",
    "INDEX_SOURCE",
    "INDEX_VERSION",
    "INTELLIGENCE_INDEX_MODELS",
    "IntelligenceIndexModel",
    "SNAPSHOT_DATE",
    "TASK_KIND_INTELLIGENCE",
    "CLI_DEFAULT_MODELS",
    "LAB_PROVIDERS",
    "infer_task_kind",
    "intelligence_cost_matrix",
    "intelligence_floor_for_task",
    "is_auto_model_name",
    "discover_available_providers",
    "load_intelligence_index_models",
    "providers_for_lab",
    "select_efficient_model",
    "select_efficient_route",
]
